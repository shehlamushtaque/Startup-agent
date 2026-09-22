"""Prepare structured evidence for LLM synthesis."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List

from backend.agents.base import AgentResponse, Evidence


def prepare_structured_evidence(
    responses: Iterable[AgentResponse],
    idea_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert agent responses into structured JSON for LLM consumption."""
    evidence_by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    evidence_index: Dict[str, Dict[str, Any]] = {}
    agent_outputs: List[Dict[str, Any]] = []
    evidence_counter = 1

    for response in responses:
        category = _categorize_agent(response.agent_name)
        agent_entry = {
            "agent_name": response.agent_name,
            "task_id": response.task_id,
            "summary": response.summary,
            "confidence": response.confidence,
            "evidence": [],
        }
        for ev in response.evidence:
            ev_id = f"E{evidence_counter}"
            evidence_counter += 1

            metadata = _extract_metadata(ev)
            structured = {
                "id": ev_id,
                "category": category,
                "agent": response.agent_name,  # Include agent name for reference
                "title": getattr(ev, "title", ""),
                "content": getattr(ev, "content", ""),
                "source": getattr(ev, "source", ""),
                "metadata": metadata,
                "relevance_score": getattr(ev, "score", None),
                "region_match_score": _region_match_score(metadata, idea_context),
            }

            evidence_by_category[category].append(structured)
            evidence_index[ev_id] = structured
            agent_entry["evidence"].append(structured)

        if agent_entry["evidence"]:
            agent_outputs.append(agent_entry)

    # Extract key metrics
    metrics = _extract_metrics(responses)

    return {
        "idea_summary": {
            "idea": idea_context.get("idea", ""),
            "problem": idea_context.get("problem", ""),
            "audience": idea_context.get("audience", ""),
            "region": idea_context.get("region", ""),
            "maturity": idea_context.get("maturity", ""),
        },
        "evidence_by_category": dict(evidence_by_category),
        "evidence_index": evidence_index,
        "key_metrics": metrics,
        "total_evidence_count": evidence_counter - 1,
        "agent_outputs": agent_outputs,
    }


def _categorize_agent(agent_name: str) -> str:
    """Map agent name to evidence category."""
    mapping = {
        "FundingIntelligenceAgent": "funding",
        "CompetitorGrowthAgent": "competitors",
        "WebResearchAgent": "market_trends",
        "SerpSearchAgent": "market_trends",
        "SecFilingsAgent": "regulatory",
    }
    return mapping.get(agent_name, "general")


def _extract_metadata(evidence: Evidence) -> Dict[str, Any]:
    """Extract and normalize metadata from evidence item."""
    metadata = getattr(evidence, "metadata", {})
    if not isinstance(metadata, dict):
        return {}

    # Normalize common fields
    normalized: Dict[str, Any] = {}
    if "growjo_growth_percent" in metadata:
        normalized["growth_percent"] = metadata["growjo_growth_percent"]
    if "growjo_estimated_revenue_usd" in metadata:
        normalized["revenue_usd"] = metadata["growjo_estimated_revenue_usd"]
    if "growjo_employees" in metadata:
        normalized["employees"] = metadata["growjo_employees"]
    if "funding_total_usd" in metadata:
        normalized["funding_total_usd"] = metadata["funding_total_usd"]
    if "investor_names" in metadata:
        normalized["investors"] = metadata["investor_names"]
    if "company" in metadata:
        normalized["company"] = metadata["company"]

    # Preserve other fields
    for key, value in metadata.items():
        if key not in normalized and value not in (None, "", "nan"):
            normalized[key] = value

    return normalized


def _extract_metrics(responses: Iterable[AgentResponse]) -> Dict[str, Any]:
    """Extract key numeric metrics from responses."""
    metrics: Dict[str, Any] = {
        "funding": [],
        "competitors": [],
        "market_signals": 0,
        "regulatory_filings": 0,
    }

    for response in responses:
        if response.agent_name == "FundingIntelligenceAgent":
            for ev in response.evidence:
                meta = _extract_metadata(ev)
                if "funding_total_usd" in meta:
                    metrics["funding"].append(
                        {
                            "company": meta.get("company") or getattr(ev, "title", ""),
                            "amount_usd": meta["funding_total_usd"],
                        }
                    )
        elif response.agent_name == "CompetitorGrowthAgent":
            for ev in response.evidence:
                meta = _extract_metadata(ev)
                if "growth_percent" in meta or "revenue_usd" in meta:
                    metrics["competitors"].append(
                        {
                            "company": meta.get("company") or getattr(ev, "title", ""),
                            "growth_percent": meta.get("growth_percent"),
                            "revenue_usd": meta.get("revenue_usd"),
                            "employees": meta.get("employees"),
                        }
                    )
        elif response.agent_name in ("WebResearchAgent", "SerpSearchAgent"):
            metrics["market_signals"] += len(response.evidence)
        elif response.agent_name == "SecFilingsAgent":
            metrics["regulatory_filings"] += len(response.evidence)

    return metrics


def _region_match_score(metadata: Dict[str, Any], idea_context: Dict[str, Any]) -> float:
    """Score how well evidence metadata aligns with requested region."""
    region = (idea_context.get("region") or "").strip().lower()
    if not region or not metadata:
        return 0.0

    candidate_fields = [
        "region",
        "country",
        "hq_region",
        "hq_country",
        "hq_location",
        "location",
        "headquarters",
        "headquarters_location",
        "market",
        "description",
    ]

    score = 0.0
    for field in candidate_fields:
        value = metadata.get(field)
        if not value:
            continue
        value_str = str(value).lower()
        if region in value_str:
            score += 1.0
        elif "asia" in value_str and ("asia" in region or "south korea" in region):
            score += 0.4
        elif "korea" in value_str and "korea" in region:
            score += 0.8

    # Bonus if metadata explicitly mentions the region in free-form content
    free_text_fields = ["content", "summary", "notes"]
    for field in free_text_fields:
        value = metadata.get(field)
        if value and region in str(value).lower():
            score += 0.5

    return min(score, 3.0)


def curate_top_evidence(
    structured_evidence: Dict[str, Any],
    max_per_category: int = 5,
) -> Dict[str, Any]:
    """Select top N evidence items per category based on relevance."""
    curated: Dict[str, List[Dict[str, Any]]] = {}
    evidence_by_category = structured_evidence.get("evidence_by_category", {})

    for category, items in evidence_by_category.items():
        # Sort by region relevance first, then by explicit relevance score, finally preserve original order
        sorted_items = sorted(
            items,
            key=lambda x: (
                x.get("region_match_score", 0.0),
                x.get("relevance_score") or 0.0,
            ),
            reverse=True,
        )

        # Deduplicate companies to avoid repeating the same example (e.g., Lucid) in every report
        selected: List[Dict[str, Any]] = []
        seen_entities: set[str] = set()
        for item in sorted_items:
            metadata = item.get("metadata") or {}
            entity_key = (
                metadata.get("company")
                or metadata.get("normalized_name")
                or item.get("title")
                or item.get("source")
                or ""
            ).strip().lower()
            if entity_key and entity_key in seen_entities:
                continue
            if entity_key:
                seen_entities.add(entity_key)
            selected.append(item)
            if len(selected) >= max_per_category:
                break

        curated[category] = selected

    # Rebuild evidence_index with only curated items
    curated_index: Dict[str, Dict[str, Any]] = {}
    for items in curated.values():
        for item in items:
            curated_index[item["id"]] = item

    return {
        **structured_evidence,
        "evidence_by_category": curated,
        "evidence_index": curated_index,
    }

