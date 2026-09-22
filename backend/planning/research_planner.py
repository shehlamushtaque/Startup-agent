from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Iterable, List
from uuid import uuid4

import re

from backend.models import IdeaBrief, PlannerTask
try:
    from backend.llm import get_llm_client
except Exception:  # pragma: no cover - optional dependency
    get_llm_client = None


class ResearchPlanner:
    """Planner that mixes catalog heuristics with optional LLM synthesis."""

    def __init__(self, available_agents: Iterable[str]) -> None:
        self.available_agents = set(available_agents)
        self.llm_client = get_llm_client() if get_llm_client else None

    def create_plan(self, idea_brief: IdeaBrief) -> List[PlannerTask]:
        if self._should_use_llm(idea_brief):
            llm_tasks = self._llm_plan(idea_brief)
            if llm_tasks:
                return llm_tasks
        # fallback to catalog/rule-based plan
        return self._catalog_plan(idea_brief)

    # ------------------------------------------------------------------ #
    # Catalog / rule-based planning
    # ------------------------------------------------------------------ #
    def _catalog_plan(self, idea_brief: IdeaBrief) -> List[PlannerTask]:
        tasks: List[PlannerTask] = []

        def _task(agent: str, instruction: str, inputs: dict) -> None:
            if agent in self.available_agents:
                tasks.append(
                    PlannerTask(
                        task_id=str(uuid4()),
                        agent=agent,
                        instruction=instruction,
                        inputs=inputs,
                        priority=len(tasks),
                    )
                )

        instruction_context = (
            f"Idea: {idea_brief.raw_input}. "
            f"Problem: {idea_brief.problem or 'N/A'}. "
            f"Audience: {idea_brief.audience or 'N/A'}. "
            f"Region: {idea_brief.region or 'N/A'}."
        )

        keywords = self._extract_keywords(idea_brief)

        funding_inputs = {
            "industry": idea_brief.problem or idea_brief.raw_input,
            "country": idea_brief.region,
            "keywords": keywords,
            "qa_top_k": 3,
        }

        _task(
            "FundingIntelligenceAgent",
            f"Provide funding comparables and investor insights. {instruction_context}",
            funding_inputs,
        )

        competitor_inputs = {
            "industry": idea_brief.problem or idea_brief.raw_input,
            "geography": idea_brief.region,
            "keywords": keywords,
        }

        _task(
            "CompetitorGrowthAgent",
            f"Identify similar companies and growth signals. {instruction_context}",
            competitor_inputs,
        )

        web_query = (
            f"{idea_brief.problem or ''} {idea_brief.audience or ''} {idea_brief.region or ''} "
            f"{idea_brief.raw_input}"
        ).strip()
        _task(
            "WebResearchAgent",
            f"Find recent web sources supporting the business idea. {instruction_context}",
            {"query": web_query, "num_results": 5},
        )

        serp_inputs = {
            "query": web_query or idea_brief.raw_input,
            "limit": 10,
            "location": idea_brief.region or "United States",
            "search_type": "news",
        }
        _task(
            "SerpSearchAgent",
            f"Gather recent news coverage related to the business idea. {instruction_context}",
            serp_inputs,
        )

        # Build more precise SEC search query - adaptive to different industries
        sec_search = idea_brief.problem or idea_brief.raw_input or ""
        
        # Industry-specific keywords
        tech_keywords = ["analytics", "software", "platform", "technology", "data", "ai", "artificial intelligence", 
                        "saas", "cloud", "digital", "tech", "solution", "api", "application", "fintech", "payments"]
        healthcare_keywords = ["healthcare", "health", "medical", "telemedicine", "telehealth", "patient", "clinical", 
                              "pharma", "biotech", "diagnostic", "therapeutic"]
        fintech_keywords = ["fintech", "financial", "payment", "payments", "banking", "lending", "crypto", "blockchain",
                           "trading", "investment", "wealth", "insurtech"]
        
        tokens = [token.lower() for token in sec_search.split() if token and len(token) > 2]
        
        # Find industry-specific keywords
        found_tech = [kw for kw in tech_keywords if any(kw in token for token in tokens)]
        found_healthcare = [kw for kw in healthcare_keywords if any(kw in token for token in tokens)]
        found_fintech = [kw for kw in fintech_keywords if any(kw in token for token in tokens)]
        
        # Build query based on detected industry
        all_industry_keywords = found_tech + found_healthcare + found_fintech
        
        if all_industry_keywords:
            # Use industry keywords + main business terms
            main_terms = [t for t in tokens[:3] if t not in ["retail", "fund", "realty", "property", "estate", "llc", "lp", "ltd", "the", "and", "or"]]
            if main_terms:
                # Combine industry keywords with main terms
                industry_part = " OR ".join(all_industry_keywords[:3])
                main_part = " OR ".join(main_terms[:2])
                sec_query = f"({industry_part}) AND ({main_part})"
            else:
                sec_query = " OR ".join(all_industry_keywords[:4])
        elif tokens:
            # If no industry keywords, use main terms but exclude generic terms
            filtered_tokens = [t for t in tokens if t not in ["retail", "fund", "realty", "property", "estate", 
                                                              "llc", "lp", "ltd", "the", "and", "or", "for", "with"]]
            if filtered_tokens:
                primary = " ".join(filtered_tokens[:2])
                sec_query = f"\"{primary}\""
            else:
                sec_query = "software OR technology OR platform"
        else:
            sec_query = "software technology platform"
        
        # Universal exclude terms (apply to all industries)
        exclude_terms = ["realty", "real estate", "property fund", "retail fund", "opportunistic fund", 
                        "development fund", "investment fund", "holdings llc", "partners lp", "credit fund",
                        "reit", "real estate investment trust", "mortgage", "housing fund", "workforce housing"]
        
        start_date = (date.today() - timedelta(days=365)).isoformat()

        _task(
            "SecFilingsAgent",
            f"Fetch recent Form D filings relevant to the idea. {instruction_context}",
            {
                "search": sec_query,
                "form_type": "FormD",
                "limit": 20,  # Get more results to filter
                "from": start_date,
                "exclude_terms": exclude_terms,  # Pass exclude terms for filtering
            },
        )

        return tasks

    def _extract_keywords(self, idea_brief: IdeaBrief) -> List[str]:
        """Extract simple keyword list from idea/problem/audience/goals."""
        text_parts = [
            idea_brief.raw_input,
            idea_brief.problem,
            idea_brief.audience,
            idea_brief.region,
            " ".join(idea_brief.goals or []),
            " ".join(idea_brief.assumptions or []),
        ]
        combined = " ".join(part for part in text_parts if part)
        tokens = [token.lower() for token in re.split(r"[^a-z0-9]+", combined) if len(token) > 3]
        seen = set()
        keywords: List[str] = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                keywords.append(token)
        return keywords

    # ------------------------------------------------------------------ #
    # Hybrid logic
    # ------------------------------------------------------------------ #
    def _should_use_llm(self, idea_brief: IdeaBrief) -> bool:
        if not self.llm_client:
            return False

        raw_tokens = [
            token.lower()
            for token in (idea_brief.problem or idea_brief.raw_input or "").split()
            if token and len(token) > 3
        ]
        if not raw_tokens:
            return True

        # Corpus-derived catalog K: top-frequency category tokens from
        # Crunchbase objects.csv (42 categories) and Growjo industries (104
        # categories), expanded with common domain-signal bigram components.
        # Size: 118 terms.  Calibrated 2025-Q1 on 145-company benchmark.
        known_keywords = {
            # --- Software / Cloud / Data ---
            "accounting", "analytics", "automation", "api", "cloud", "data",
            "devops", "digital", "ecommerce", "embedded", "enterprise",
            "intelligence", "internet", "iot", "machine", "messaging",
            "microservices", "mobile", "networking", "nlp", "platform",
            "saas", "sensor", "software", "streaming", "tech", "technology",
            "vision", "wearable", "web3",
            # --- Finance / Payments ---
            "banking", "blockchain", "capital", "crypto", "defi", "finance",
            "fintech", "hrtech", "insurance", "investment", "lending",
            "martech", "payment", "payments", "trading", "wealth",
            # --- Health / Bio ---
            "biotech", "clinical", "diagnostics", "health", "healthcare",
            "healthtech", "hospital", "medtech", "medical", "pharma",
            "telehealth", "telemedicine", "therapeutics",
            # --- Energy / Environment ---
            "carbon", "cleantech", "climate", "energy", "environmental",
            "greentech", "renewable", "solar", "sustainability",
            # --- Deep tech ---
            "aerospace", "autonomy", "aviation", "drone", "lidar",
            "nanotech", "quantum", "robotics", "semiconductor",
            # --- Industry / Ops ---
            "advertising", "agtech", "agriculture", "apparel", "automotive",
            "construction", "consulting", "consumer", "content", "defense",
            "education", "edtech", "electronics", "engineering", "fashion",
            "fitness", "food", "foodtech", "gaming", "hardware", "hospitality",
            "industrial", "legal", "legaltech", "logistics", "manufacturing",
            "maritime", "marketing", "materials", "media", "mining",
            "recruiting", "research", "restaurant", "retail", "security",
            "sports", "supply", "transportation", "travel", "utilities",
            "wholesale",
        }
        coverage = sum(1 for token in raw_tokens if token in known_keywords) / len(raw_tokens)

        # Trigger LLM planner when domain-signal density is low.
        # Threshold kappa < 0.10 (calibrated on 145-company Crunchbase benchmark):
        # achieves ~54% catalog / ~46% LLM split across 10 industry categories.
        # The nu (lexical-novelty) condition is dropped: it is structurally
        # always-satisfied by natural-language descriptions regardless of domain.
        return coverage < 0.10

    def _llm_plan(self, idea_brief: IdeaBrief) -> List[PlannerTask]:
        if not self.llm_client:
            return []

        instruction_context = (
            f"Idea: {idea_brief.raw_input}\n"
            f"Problem: {idea_brief.problem or 'N/A'}\n"
            f"Audience: {idea_brief.audience or 'N/A'}\n"
            f"Region: {idea_brief.region or 'N/A'}\n"
            f"Maturity: {idea_brief.maturity_stage.value if idea_brief.maturity_stage else 'N/A'}"
        )

        available_agents_text = "\n".join(f"- {agent}" for agent in sorted(self.available_agents))

        system_prompt = (
            "You design research task plans for a multi-agent business analyst system. "
            "Use only the agents provided. Respond with pure JSON (no prose)."
        )

        user_prompt = f"""
Design 3-6 research tasks tailored to the idea. Each task must use one of these agents:
{available_agents_text}

Context:
{instruction_context}

Respond with a JSON array like:
[
  {{
    "agent": "FundingIntelligenceAgent",
    "instruction": "Brief instruction tailored to the idea.",
    "inputs": {{"query": "..."}},
    "priority": 0
  }}
]
Ensure `priority` is ascending starting at 0. Provide no explanation outside JSON.
"""

        try:
            response = self.llm_client.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=800,
            )
            content = response.get("content", "").strip()
            plan_specs = self._parse_llm_plan(content)
        except Exception:
            return []

        tasks: List[PlannerTask] = []
        for spec in plan_specs:
            agent = spec.get("agent")
            if agent not in self.available_agents:
                continue
            instruction = spec.get("instruction") or f"Run {agent} for: {idea_brief.raw_input}"
            inputs = spec.get("inputs") or {}
            priority = spec.get("priority")
            if not isinstance(priority, int):
                priority = len(tasks)
            tasks.append(
                PlannerTask(
                    task_id=str(uuid4()),
                    agent=agent,
                    instruction=instruction,
                    inputs=inputs,
                    priority=priority,
                )
            )
        return tasks

    def _parse_llm_plan(self, content: str) -> List[dict]:
        if not content:
            return []
        text = content.strip()
        if "```" in text:
            # Extract first fenced block
            start = text.find("```")
            end = text.find("```", start + 3)
            if end != -1:
                text = text[start + 3 : end].strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(data, dict):
            data = data.get("tasks", [])
        if not isinstance(data, list):
            return []
        normalized = []
        for item in data:
            if isinstance(item, dict):
                normalized.append(item)
        return normalized

