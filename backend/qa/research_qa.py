"""Interactive Q&A system for querying research results."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from backend.agents.base import AgentResponse, Evidence

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import LLM clients
try:
    from backend.llm import get_llm_client
    from backend.llm.client import DeepSeekClient
    from backend.llm.ollama_client import OllamaClient
except ImportError:
    get_llm_client = None
    DeepSeekClient = None
    OllamaClient = None


class ResearchQA:
    """Allows users to ask questions about collected research evidence."""

    def __init__(self, responses: Iterable[AgentResponse], llm_client=None):
        self.responses = list(responses)
        # Use provided client, or auto-detect (Ollama first, then DeepSeek)
        if llm_client:
            self.llm_client = llm_client
        elif get_llm_client:
            self.llm_client = get_llm_client()
        else:
            self.llm_client = None
        self.evidence_index = self._build_evidence_index()

    def _has_llm_key(self) -> bool:
        """Check if any LLM is available (Ollama or DeepSeek)."""
        import os
        # Check for Ollama (local, preferred)
        if OllamaClient:
            try:
                ollama = OllamaClient()
                if ollama._check_availability():
                    return True
            except Exception:
                pass
        # Check for DeepSeek
        return bool(os.getenv("DEEPSEEK_API_KEY"))

    def _build_evidence_index(self) -> List[Dict[str, Any]]:
        """Build a searchable index of all evidence."""
        index = []
        for response in self.responses:
            if response.agent_name == "SynthesisAgent":
                continue  # Skip synthesis agent
            
            for ev in response.evidence:
                entry = {
                    "agent": response.agent_name,
                    "title": getattr(ev, "title", ""),
                    "content": getattr(ev, "content", ""),
                    "source": getattr(ev, "source", ""),
                    "metadata": getattr(ev, "metadata", {}),
                    "full_text": self._extract_full_text(ev, response),
                }
                index.append(entry)
        return index

    def _extract_full_text(self, evidence: Evidence, response: AgentResponse) -> str:
        """Extract full searchable text from evidence."""
        parts = []
        if evidence.title:
            parts.append(evidence.title)
        if evidence.content:
            formatted = self._format_evidence_content(evidence.content)
            if formatted:
                parts.append(formatted)
        if evidence.source:
            parts.append(evidence.source)
        if evidence.metadata:
            if isinstance(evidence.metadata, dict):
                # Extract key metadata fields as readable text
                for key, value in evidence.metadata.items():
                    if value and key not in ["raw", "id"]:
                        if isinstance(value, (str, int, float)):
                            parts.append(f"{key} {value}")
                        elif isinstance(value, list):
                            parts.append(", ".join(str(v) for v in value[:5]))
        parts.append(response.summary)
        return " ".join(parts)

    @staticmethod
    def _format_evidence_content(content: Any) -> str:
        """Convert evidence content into a compact, readable string."""
        if not content:
            return ""
        if isinstance(content, dict):
            content_parts = []
            for key, value in content.items():
                if value and key not in ["raw", "id"]:
                    if isinstance(value, (str, int, float)):
                        content_parts.append(f"{key}: {value}")
                    elif isinstance(value, list):
                        preview = ", ".join(str(v) for v in value[:3])
                        if len(value) > 3:
                            preview += f" (+{len(value) - 3} more)"
                        content_parts.append(f"{key}: {preview}")
            return " | ".join(content_parts)
        return str(content)

    @staticmethod
    def _fmt_money(val: Any) -> Optional[str]:
        """Format numeric money values into readable abbreviations."""
        if val is None or val == "":
            return None
        try:
            v = float(val)
        except (TypeError, ValueError):
            return None
        if v >= 1_000_000_000:
            return f"${v / 1_000_000_000:.1f}B"
        if v >= 1_000_000:
            return f"${v / 1_000_000:.1f}M"
        if v >= 1_000:
            return f"${v / 1_000:.1f}K"
        return f"${v:,.0f}"

    def _format_metadata(self, meta: Any) -> str:
        """Render metadata dict into a structured summary string."""
        if not meta or not isinstance(meta, dict):
            return ""

        meta_parts = []

        company = meta.get("company") or meta.get("name_cb")
        if company:
            meta_parts.append(f"Company: {company}")

        funding = meta.get("funding_total_usd") or meta.get("total_offering_amount") or meta.get("amount_raised")
        funding_fmt = self._fmt_money(funding)
        if funding_fmt:
            meta_parts.append(f"Funding: {funding_fmt}")

        growth = meta.get("growjo_growth_percent")
        if growth is not None:
            try:
                meta_parts.append(f"Growth: {float(growth):.0f}%")
            except (TypeError, ValueError):
                pass

        revenue = meta.get("growjo_estimated_revenue_usd")
        revenue_fmt = self._fmt_money(revenue)
        if revenue_fmt:
            meta_parts.append(f"Revenue: {revenue_fmt}")

        employees = meta.get("growjo_employees")
        if employees is not None:
            try:
                meta_parts.append(f"Employees: {int(float(employees)):,}")
            except (TypeError, ValueError):
                pass

        investors = meta.get("investors") or meta.get("investor_names")
        if investors:
            if isinstance(investors, list):
                inv_str = ", ".join(str(inv) for inv in investors[:3])
                if len(investors) > 3:
                    inv_str += f" and {len(investors) - 3} more"
            else:
                inv_str = str(investors)
            meta_parts.append(f"Investors: {inv_str}")

        if meta.get("filed_at"):
            meta_parts.append(f"Filed: {meta['filed_at']}")

        other_fields = ["description", "form_type", "latest_investment_at", "funding_round_count"]
        for field in other_fields:
            if meta.get(field):
                meta_parts.append(f"{field}: {meta[field]}")

        return " | ".join(meta_parts)

    def _gather_agent_responses(self) -> List[Dict[str, Any]]:
        """Collect every agent's summary and evidence for full-context LLM synthesis."""
        agent_entries: List[Dict[str, Any]] = []
        for response in self.responses:
            if response.agent_name == "SynthesisAgent":
                continue
            evidence_entries = []
            for ev in response.evidence:
                evidence_entries.append(
                    {
                        "title": getattr(ev, "title", ""),
                        "content": self._format_evidence_content(getattr(ev, "content", "")),
                        "metadata": getattr(ev, "metadata", {}),
                        "source": getattr(ev, "source", ""),
                    }
                )
            agent_entries.append(
                {
                    "agent": response.agent_name,
                    "summary": response.summary,
                    "evidence": evidence_entries,
                }
            )
        return agent_entries

    def search_evidence(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Context-aware search through evidence - gets comprehensive context from research."""
        query_lower = query.lower()
        query_terms = [t for t in query_lower.split() if len(t) > 2]  # Filter out short words
        
        scored_results = []
        for entry in self.evidence_index:
            full_text_lower = entry["full_text"].lower()
            score = 0
            
            # Exact phrase match (highest weight)
            if query_lower in full_text_lower:
                score += 20
            
            # Semantic keyword matching - business context
            business_context_keywords = {
                "invest": ["funding", "investment", "capital", "raise", "money", "financing", "valuation"],
                "profit": ["revenue", "growth", "success", "viable", "profitable", "earnings", "income"],
                "competitor": ["competition", "similar", "market", "industry", "rival", "peer"],
                "market": ["trend", "industry", "sector", "demand", "opportunity", "size"],
                "funding": ["raised", "investment", "capital", "investor", "round", "series"],
            }
            
            # Check for semantic matches
            for query_word in query_terms:
                if query_word in business_context_keywords:
                    for related_word in business_context_keywords[query_word]:
                        if related_word in full_text_lower:
                            score += 3  # Semantic match bonus
            
            # Individual term matches
            for term in query_terms:
                if term in full_text_lower:
                    score += 2
            
            # Boost score for business-relevant agents
            agent = entry.get("agent", "")
            if agent in ["FundingIntelligenceAgent", "CompetitorGrowthAgent", "SecFilingsAgent"]:
                score += 5  # Prioritize structured business data
            
            # Boost for having actual metrics/data
            if entry.get("metadata"):
                meta = entry.get("metadata", {})
                if isinstance(meta, dict):
                    if any(key in meta for key in ["funding_total_usd", "growjo_growth_percent", 
                                                   "total_offering_amount", "growjo_estimated_revenue_usd"]):
                        score += 3  # Has concrete data
            
            if score > 0:
                scored_results.append((score, entry))
        
        # Sort by score (descending) and return top_k
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored_results[:top_k]]

    def answer_question(self, question: str, use_llm: bool = True, idea_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Answer a question based on collected evidence - gets comprehensive context from research."""
        # Search for relevant evidence - ONLY get top results that match the question
        # Don't add all business evidence - only what's actually relevant
        relevant_evidence = self.search_evidence(question, top_k=5)
        
        # Filter evidence by relevance to the specific question
        # Don't include generic business data unless it matches the question
        question_lower = question.lower()
        question_terms = [t for t in question_lower.split() if len(t) > 3]  # Focus on meaningful terms
        
        # Score evidence by how well it matches the question
        scored_evidence = []
        for ev in relevant_evidence:
            score = 0
            ev_text = (ev.get("title", "") + " " + ev.get("content", "")).lower()
            
            # Exact phrase match
            if question_lower in ev_text:
                score += 10
            
            # Term matches
            for term in question_terms:
                if term in ev_text:
                    score += 2
            
            # Boost if evidence mentions startup idea context
            if idea_context:
                idea = idea_context.get("idea", "").lower()
                problem = idea_context.get("problem", "").lower()
                if idea and any(word in ev_text for word in idea.split() if len(word) > 3):
                    score += 5
                if problem and any(word in ev_text for word in problem.split() if len(word) > 3):
                    score += 5
            
            if score > 0:
                scored_evidence.append((score, ev))
        
        # Sort by score and take top 5
        scored_evidence.sort(key=lambda x: x[0], reverse=True)
        relevant_evidence = [ev for _, ev in scored_evidence[:5]]
        
        # Ensure we have domain-relevant evidence if user asks about a known category
        if not relevant_evidence:
            category_specific = self._fallback_by_question_category(question_lower)
            if category_specific:
                relevant_evidence = category_specific

        if not relevant_evidence:
            if use_llm and self.llm_client:
                llm_answer = self._answer_with_llm_no_evidence(question, idea_context)
                return {
                    "question": question,
                    "answer": llm_answer,
                    "sources": [],
                    "evidence": [],
                    "llm_used": True,
                }
            return {
                "question": question,
                "answer": "I couldn't find relevant information in the research results to answer this question. Please try rephrasing or ask about funding, competitors, market trends, or SEC filings.",
                "sources": [],
                "evidence": [],
                "llm_used": False,
            }
        
        # Prefer LLM if available and requested
        if use_llm and self.llm_client:
            # Build context with startup idea information
            context_parts = []
            context_parts.append("=== STARTUP IDEA CONTEXT ===")
            if idea_context:
                idea = idea_context.get("idea", "")
                problem = idea_context.get("problem", "")
                audience = idea_context.get("audience", "")
                region = idea_context.get("region", "")
                if idea:
                    context_parts.append(f"Startup Idea: {idea}")
                if problem:
                    context_parts.append(f"Problem: {problem}")
                if audience:
                    context_parts.append(f"Target Audience: {audience}")
                if region:
                    context_parts.append(f"Region: {region}")
            context_parts.append("")
            context_parts.append("=== USER QUESTION ===")
            context_parts.append(f"{question}")
            context_parts.append("")
            context_parts.append("=== RELEVANT EVIDENCE (ONLY items that directly relate to the question) ===")
            sources = []
            for i, ev in enumerate(relevant_evidence, 1):
                context_parts.append(f"[Evidence {i} from {ev['agent']}]")
                if ev["title"]:
                    context_parts.append(f"Title: {ev['title']}")
                if ev["content"]:
                    content = self._format_evidence_content(ev["content"])
                    if content:
                        context_parts.append(f"Content: {content}")
                if ev["metadata"]:
                    meta_str = self._format_metadata(ev["metadata"])
                    if meta_str:
                        context_parts.append(f"Details: {meta_str}")
                if ev["source"]:
                    context_parts.append(f"Source: {ev['source']}")
                context_parts.append("")
                sources.append({
                    "id": ev.get("id"),
                    "agent": ev["agent"],
                    "title": ev["title"],
                    "source": ev["source"],
                    "metadata": ev.get("metadata", {}),
                })

            full_agent_entries = self._gather_agent_responses()
            if full_agent_entries:
                context_parts.append("")
                context_parts.append("=== COMPREHENSIVE AGENT RESPONSES (ALL RAW FINDINGS) ===")
                for entry in full_agent_entries:
                    context_parts.append(f"[Agent: {entry['agent']}]")
                    if entry["summary"]:
                        context_parts.append(f"Summary: {entry['summary']}")
                    for idx, evidence in enumerate(entry["evidence"], 1):
                        title = evidence["title"] or f"Evidence {idx}"
                        context_parts.append(f"- {title}")
                        if evidence["content"]:
                            context_parts.append(f"  Content: {evidence['content']}")
                        meta_str = self._format_metadata(evidence.get("metadata"))
                        if meta_str:
                            context_parts.append(f"  Metadata: {meta_str}")
                        if evidence.get("source"):
                            context_parts.append(f"  Source: {evidence['source']}")
                    context_parts.append("")
                context_parts.append("")
            
            context = "\n".join(context_parts)
            
            # Use LLM to analyze and synthesize
            try:
                answer = self._llm_answer(question, context)
                return {
                    "question": question,
                    "answer": answer,
                    "sources": sources,
                    "evidence": relevant_evidence,
                    "llm_used": True,
                }
            except Exception as e:
                # Fallback to template answer
                import sys
                error_msg = str(e)
                if "402" in error_msg or "Insufficient Balance" in error_msg:
                    print("[INFO] LLM Q&A unavailable: DeepSeek account needs credits. Using template-based answers.", file=sys.stderr)
                elif "401" in error_msg or "Unauthorized" in error_msg:
                    print("[WARNING] LLM Q&A failed: Invalid API key. Using template-based answers.", file=sys.stderr)
                # Continue to template fallback
        
        # Build context for template fallback (if LLM not available or failed)
        context_parts = []
        sources = []
        for i, ev in enumerate(relevant_evidence, 1):
            context_parts.append(f"[Evidence {i} from {ev['agent']}]")
            if ev["title"]:
                context_parts.append(f"Title: {ev['title']}")
            if ev["content"]:
                content = self._format_evidence_content(ev["content"])
                if content:
                    context_parts.append(f"Content: {content}")
            if ev["metadata"]:
                meta_str = self._format_metadata(ev["metadata"])
                if meta_str:
                    context_parts.append(f"Details: {meta_str}")
            if ev["source"]:
                context_parts.append(f"Source: {ev['source']}")
            context_parts.append("")
            sources.append({
                "id": ev.get("id"),
                "agent": ev["agent"],
                "title": ev["title"],
                "source": ev["source"],
                "metadata": ev.get("metadata", {}),
            })
        
        # Template-based answer with evidence
        answer = self._template_answer(question, relevant_evidence)
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "evidence": relevant_evidence,
            "llm_used": False,
        }

    def _fallback_by_question_category(self, question_lower: str) -> List[Dict[str, Any]]:
        """Provide targeted evidence if question mentions a specific category."""
        category_keywords = {
            "competitors": ["competitor", "competition", "similar company", "rival", "players"],
            "funding": ["funding", "invest", "raise", "capital", "valuation"],
            "market": ["market", "trend", "demand", "opportunity"],
            "regulatory": ["sec", "filing", "regulation", "form d"],
        }
        agent_map = {
            "competitors": {"CompetitorGrowthAgent"},
            "funding": {"FundingIntelligenceAgent"},
            "market": {"WebResearchAgent", "SerpSearchAgent", "TrendsAgent"},
            "regulatory": {"SecFilingsAgent"},
        }

        normalized_question = question_lower.lower()
        selected_agents: set[str] = set()
        for category, keywords in category_keywords.items():
            if any(keyword in normalized_question for keyword in keywords):
                selected_agents |= agent_map.get(category, set())

        if not selected_agents:
            return []

        targeted: List[Dict[str, Any]] = []
        for entry in self.evidence_index:
            if entry.get("agent") in selected_agents:
                targeted.append(entry)
                if len(targeted) >= 5:
                    break
        return targeted

    def _llm_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM with evidence analysis and synthesis."""
        system_prompt = """You are a precise business analyst answering questions about a SPECIFIC startup idea. Your answers must be directly relevant to the user's question and the startup context provided.

CRITICAL RULES:
1. Answer ONLY what the user asked - do not provide generic information about companies/trends unless directly relevant to the question.
2. If the evidence doesn't directly answer the question, say so clearly - don't make up answers or provide irrelevant data.
3. Lead with a direct answer (≤3 sentences) using ONLY facts from the evidence that relate to the question.
4. Each supporting fact must be directly relevant to the question - don't list random metrics.
5. Always cite evidence IDs inline (e.g., $954M [Evidence 1]).
6. If the question is about the startup idea specifically, connect the evidence to that idea.
7. Never provide the same generic answer for different questions - each answer must be unique to the question asked.
8. Output format:
Answer: <direct, question-specific answer with numbers if available>.
Supporting facts:
- <relevant fact> [Evidence #]
- <another relevant fact> [Evidence #]
9. If evidence is insufficient or irrelevant, say: "The available evidence doesn't directly answer this question. The research found [brief summary], but it doesn't address [what was asked]." """
        
        user_prompt = f"""{context}

IMPORTANT: 
- The evidence above is filtered to be relevant to the question. Use ONLY this evidence.
- If the evidence doesn't directly answer the question, explicitly state that.
- Do NOT provide generic answers that could apply to any startup.
- Connect your answer to the specific startup idea if the question relates to it.

Your task:
- Read the startup context and question carefully.
- Review ONLY the evidence provided (it's already filtered for relevance).
- Answer the SPECIFIC question asked - not a generic version of it.
- If evidence is irrelevant or insufficient, say so clearly.

Output format:
Answer: <direct answer to the specific question, or state if evidence doesn't address it>.
Supporting facts:
- <only include facts that directly support the answer> [Evidence #]

Do NOT:
- Provide the same answer for different questions
- List random metrics that don't relate to the question
- Make up connections between evidence and the question if they don't exist
- Use generic templates"""
        
        response = self.llm_client.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], temperature=0.5, max_tokens=1000)
        
        return response["content"]

    def _answer_with_llm_no_evidence(self, question: str, idea_context: Optional[Dict[str, Any]]) -> str:
        """Use LLM to answer even when no structured evidence exists."""
        context_parts = ["No directly matching research evidence was found for the question."]
        if idea_context:
            idea = idea_context.get("idea", "")
            problem = idea_context.get("problem", "")
            audience = idea_context.get("audience", "")
            region = idea_context.get("region", "")
            context_parts.append("Startup Idea Context:")
            if idea:
                context_parts.append(f"- Idea: {idea}")
            if problem:
                context_parts.append(f"- Problem: {problem}")
            if audience:
                context_parts.append(f"- Audience: {audience}")
            if region:
                context_parts.append(f"- Region: {region}")
        context_parts.append("If you must answer without data, explain the gap clearly.")
        context = "\n".join(context_parts)
        return self._llm_answer(question, context)

    def _template_answer(self, question: str, evidence: List[Dict[str, Any]]) -> str:
        """Generate template-based answer from evidence with better synthesis."""
        question_lower = question.lower()
        
        # Filter evidence by relevance - prioritize business/funding/competitor data
        relevant_evidence = []
        for ev in evidence:
            agent = ev.get("agent", "")
            # Prioritize business-relevant agents
            if agent in ["FundingIntelligenceAgent", "CompetitorGrowthAgent", "SecFilingsAgent"]:
                relevant_evidence.append(ev)
            elif agent in ["WebResearchAgent", "SerpSearchAgent"]:
                # Only include web/search results if they seem business-related
                title = (ev.get("title") or "").lower()
                content = (ev.get("content") or "").lower()
                business_keywords = ["funding", "investment", "startup", "company", "business", 
                                    "market", "revenue", "growth", "competitor", "industry",
                                    "venture", "capital", "financing", "raise", "valuation"]
                if any(kw in title or kw in content for kw in business_keywords):
                    relevant_evidence.append(ev)
        
        # If we filtered out too much, use original evidence
        if not relevant_evidence:
            relevant_evidence = evidence
        
        # Questions about investment/profitability
        if any(word in question_lower for word in ["invest", "profit", "profitable", "make money", "should i", "how much"]):
            # Synthesize funding, competitor, and market data
            funding_data = []
            competitor_data = []
            market_signals = []
            
            for ev in relevant_evidence:
                meta = ev.get("metadata", {})
                if isinstance(meta, dict):
                    # Funding info
                    if ev["agent"] == "FundingIntelligenceAgent":
                        company = meta.get("company") or meta.get("name_cb") or ev.get("title", "").replace("Funding record: ", "")
                        amount = meta.get("funding_total_usd")
                        if company and amount:
                            try:
                                amount_float = float(amount)
                                if amount_float >= 1_000_000_000:
                                    formatted = f"${amount_float / 1_000_000_000:.1f}B"
                                elif amount_float >= 1_000_000:
                                    formatted = f"${amount_float / 1_000_000:.1f}M"
                                else:
                                    formatted = f"${amount_float:,.0f}"
                                funding_data.append(f"{company}: {formatted}")
                            except (TypeError, ValueError):
                                pass
                    
                    # Competitor growth
                    if ev["agent"] == "CompetitorGrowthAgent":
                        company = ev.get("title", "").replace("Growth snapshot: ", "")
                        growth = meta.get("growjo_growth_percent")
                        revenue = meta.get("growjo_estimated_revenue_usd")
                        if company:
                            parts = [company]
                            if growth:
                                try:
                                    parts.append(f"{float(growth):.0f}% growth")
                                except (TypeError, ValueError):
                                    pass
                            if revenue:
                                try:
                                    rev_float = float(revenue)
                                    if rev_float >= 1_000_000:
                                        rev_str = f"${rev_float / 1_000_000:.1f}M revenue"
                                    else:
                                        rev_str = f"${rev_float:,.0f} revenue"
                                    parts.append(rev_str)
                                except (TypeError, ValueError):
                                    pass
                            competitor_data.append(" | ".join(parts))
                
                # Market trends
                if ev["agent"] in ["WebResearchAgent", "SerpSearchAgent"]:
                    title = ev.get("title", "")
                    if title:
                        market_signals.append(title)
            
            # Build comprehensive answer
            answer_parts = []
            if funding_data:
                answer_parts.append(f"Funding benchmarks: Similar companies have raised {'; '.join(funding_data[:5])}.")
            if competitor_data:
                answer_parts.append(f"Competitor performance: {'; '.join(competitor_data[:3])}.")
            if market_signals:
                answer_parts.append(f"Market signals: {market_signals[0]}.")
            
            if answer_parts:
                return " ".join(answer_parts) + " Use this data to assess investment requirements and market viability."
        
        # Simple pattern matching for common question types
        elif any(word in question_lower for word in ["funding", "raised", "investment", "money", "capital"]):
            # Look for funding information
            funding_info = []
            for ev in relevant_evidence:
                meta = ev.get("metadata", {})
                if isinstance(meta, dict):
                    company = meta.get("company") or meta.get("name_cb") or ev.get("title", "")
                    # Clean up company name
                    if "Funding record:" in company:
                        company = company.replace("Funding record: ", "")
                    amount = meta.get("funding_total_usd") or meta.get("total_offering_amount") or meta.get("amount_raised")
                    if company and amount:
                        try:
                            amount_float = float(amount)
                            if amount_float >= 1_000_000_000:
                                formatted = f"${amount_float / 1_000_000_000:.1f}B"
                            elif amount_float >= 1_000_000:
                                formatted = f"${amount_float / 1_000_000:.1f}M"
                            else:
                                formatted = f"${amount_float:,.0f}"
                            funding_info.append(f"{company}: {formatted}")
                        except (TypeError, ValueError):
                            pass
            if funding_info:
                return f"Based on the research: {'; '.join(funding_info[:5])}."
        
        elif any(word in question_lower for word in ["competitor", "competition", "similar", "competing"]):
            # Look for competitor information
            competitors = []
            for ev in relevant_evidence:
                if ev["agent"] == "CompetitorGrowthAgent":
                    company = ev.get("title", "").replace("Growth snapshot: ", "")
                    meta = ev.get("metadata", {})
                    if isinstance(meta, dict):
                        growth = meta.get("growjo_growth_percent")
                        revenue = meta.get("growjo_estimated_revenue_usd")
                        employees = meta.get("growjo_employees")
                        if company:
                            parts = [company]
                            if growth:
                                try:
                                    parts.append(f"{float(growth):.0f}% growth")
                                except (TypeError, ValueError):
                                    pass
                            if revenue:
                                try:
                                    rev_float = float(revenue)
                                    if rev_float >= 1_000_000:
                                        rev_str = f"${rev_float / 1_000_000:.1f}M revenue"
                                    else:
                                        rev_str = f"${rev_float:,.0f} revenue"
                                    parts.append(rev_str)
                                except (TypeError, ValueError):
                                    pass
                            competitors.append(" | ".join(parts))
            if competitors:
                return f"Key competitors identified: {', '.join(competitors[:5])}."
        
        elif any(word in question_lower for word in ["trend", "market", "news", "article"]):
            # Look for market trends (only business-relevant)
            trends = []
            for ev in relevant_evidence:
                if ev["agent"] in ["WebResearchAgent", "SerpSearchAgent"]:
                    title = ev.get("title", "")
                    if title:
                        trends.append(title)
            if trends:
                return f"Market trends and news found: {'; '.join(trends[:5])}."
        
        elif any(word in question_lower for word in ["filing", "sec", "form d", "regulatory"]):
            # Look for SEC filings
            filings = []
            for ev in relevant_evidence:
                if ev["agent"] == "SecFilingsAgent":
                    company = ev.get("title", "")
                    if " (CIK" in company:
                        company = company.split(" (CIK")[0]
                    content = ev.get("content", "")
                    if company:
                        filing_info = company
                        if content and content != "Form D filing":
                            filing_info += f" | {content}"
                        filings.append(filing_info)
            if filings:
                return f"Recent Form D filings: {'; '.join(filings[:5])}."
        
        # Generic answer - summarize what was found (only business-relevant)
        if relevant_evidence:
            summary_parts = []
            for ev in relevant_evidence[:3]:
                title = ev.get("title", "")
                content = ev.get("content", "")
                # Only include if it seems business-relevant
                text = (title + " " + content).lower()
                business_keywords = ["funding", "investment", "startup", "company", "business", 
                                    "market", "revenue", "growth", "competitor", "industry"]
                if any(kw in text for kw in business_keywords):
                    if title:
                        summary_parts.append(title)
                    elif content:
                        summary_parts.append(content[:100])
            
            if summary_parts:
                return f"Based on the research evidence, I found {len(relevant_evidence)} relevant business items. " + \
                       f"Key findings: {'; '.join(summary_parts[:3])}."
        
        # If no relevant business evidence, provide helpful message
        return f"I found {len(evidence)} items in the research, but they don't appear to be directly relevant to your business question. " + \
               f"Please try asking about: funding amounts, competitor performance, market trends, or SEC filings. " + \
               f"Or set DEEPSEEK_API_KEY to enable AI-powered analysis that can better understand your question."

