"""Competitor and growth agent implementation."""

from __future__ import annotations

import math
from typing import Dict, List

from .base import AgentResponse, BaseAgent, Evidence, Task
from .tools import CompetitorGrowthTool


class CompetitorGrowthAgent(BaseAgent):
    name = "CompetitorGrowthAgent"

    def __init__(self, tools: Dict[str, object]):
        super().__init__(tools)
        if "CompetitorGrowthTool" not in tools:
            raise ValueError("CompetitorGrowthAgent requires CompetitorGrowthTool")

    def execute(self, task: Task) -> AgentResponse:
        tool: CompetitorGrowthTool = self.tools["CompetitorGrowthTool"]  # type: ignore
        results = tool.run(task.inputs)

        evidence: List[Evidence] = []
        geography = (task.inputs.get("geography") or task.inputs.get("region") or "").strip()

        for record in results.get("results", []):
            title = record.get("name_cb") or record.get("normalized_name")
            # Format content as readable text
            content_parts = []
            growth = record.get("growjo_growth_percent")
            if growth is not None:
                try:
                    content_parts.append(f"Growth: {float(growth):.0f}%")
                except (TypeError, ValueError):
                    pass
            revenue = record.get("growjo_estimated_revenue_usd")
            if revenue is not None:
                try:
                    rev_float = float(revenue)
                    if rev_float >= 1_000_000:
                        rev_str = f"${rev_float / 1_000_000:.1f}M"
                    else:
                        rev_str = f"${rev_float:,.0f}"
                    content_parts.append(f"Revenue: {rev_str}")
                except (TypeError, ValueError):
                    pass
            employees = record.get("growjo_employees")
            if employees is not None:
                try:
                    content_parts.append(f"Employees: {int(float(employees)):,}")
                except (TypeError, ValueError):
                    pass
            
            content = " | ".join(content_parts) if content_parts else f"Growth metrics for {title}"
            
            # Add source information to content
            source_info = "Data source: Growjo & Crunchbase (via internal company database)"
            if content:
                content += f" | {source_info}"
            else:
                content = source_info
            
            metadata = {
                **record,
                "_data_source": "Growjo + Crunchbase",
                "_source_type": "internal_database",
            }
            if geography:
                metadata.setdefault("region_hint", geography)

            evidence.append(
                Evidence(
                    title=f"Growth snapshot: {title}",
                    content=content,
                    source="Growjo & Crunchbase Database",  # More user-friendly source name
                    metadata=metadata,
                )
            )

        if results.get("results"):
            top = results["results"][0]
            growth = top.get("growjo_growth_percent")
            if isinstance(growth, (int, float)) and math.isnan(growth):
                growth = None
            company = top.get("name_cb") or top.get("normalized_name")
            if growth is not None:
                summary = f"Fastest comparable appears to be {company} with growth {growth}%."
            else:
                summary = f"Fastest comparable appears to be {company}, but no growth metric was disclosed."
            confidence = "medium"
        else:
            summary = "No competitor or growth data available for the current filters."
            confidence = "low"

        # Add citations with data source information
        citations = []
        # Add Growjo/Crunchbase as citation source for growth data
        if results.get("results"):
            citations.append({
                "label": "Growjo & Crunchbase - Company Growth Database",
                "url": "https://growjo.com",
                "excerpt": "Growth metrics sourced from Growjo and Crunchbase databases via internal data processing"
            })
        
        # Add any other actual URLs from evidence
        for ev in evidence:
            source = getattr(ev, "source", None)
            if source and source not in ["CompetitorGrowthTool", "Growjo & Crunchbase Database"]:
                if isinstance(source, str) and (source.startswith('http://') or source.startswith('https://')):
                    if source not in [c.get("url", "") if isinstance(c, dict) else "" for c in citations]:
                        citations.append({
                            "label": getattr(ev, "title", "Source"),
                            "url": source,
                            "excerpt": getattr(ev, "content", "")[:200] if hasattr(ev, "content") else ""
                        })
        
        return AgentResponse(
            agent_name=self.name,
            task_id=task.task_id,
            summary=summary,
            confidence=confidence,
            evidence=evidence,
            citations=citations,
        )


