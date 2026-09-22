"""Business trends agent leveraging Tracxn company data."""

from __future__ import annotations

from typing import Dict, List

from .base import AgentResponse, BaseAgent, Evidence, Task
from .tools import TracxnCompanyTool


class BusinessTrendsAgent(BaseAgent):
    name = "BusinessTrendsAgent"

    def __init__(self, tools: Dict[str, object]):
        super().__init__(tools)
        if "TracxnCompanyTool" not in tools:
            raise ValueError("BusinessTrendsAgent requires TracxnCompanyTool")

    def execute(self, task: Task) -> AgentResponse:
        tracxn_tool: TracxnCompanyTool = self.tools["TracxnCompanyTool"]  # type: ignore
        filter_payload = task.inputs.get("filter") or {}
        sort_payload = task.inputs.get("sort")
        size = task.inputs.get("size", 10)
        result = tracxn_tool.run({"filter": filter_payload, "sort": sort_payload, "size": size})

        entries = result.get("results", [])
        region_hint = (
            task.inputs.get("geography")
            or task.inputs.get("region")
            or (filter_payload.get("country") if isinstance(filter_payload, dict) else None)
            or ""
        ).strip()
        if result.get("error"):
            return AgentResponse(
                agent_name=self.name,
                task_id=task.task_id,
                summary=f"Tracxn query failed: {result['error']}",
                confidence="low",
                evidence=[],
                citations=[],
                raw_output=result,
            )
        evidence: List[Evidence] = []

        for company in entries:
            name = company.get("name") or "Unknown company"
            funding = company.get("total_equity_funding")
            summary_parts = [
                company.get("short_description") or company.get("detailed_description") or "",
                f"Stage: {company.get('stage')}" if company.get("stage") else "",
                f"Funding: ${funding:,}" if isinstance(funding, (int, float)) else "",
            ]
            investors = company.get("investors") or []
            if investors:
                summary_parts.append("Investors: " + ", ".join(investors[:5]))
            summary = " | ".join(filter(None, summary_parts))
            metadata = dict(company)
            if region_hint and not metadata.get("region_hint"):
                metadata["region_hint"] = region_hint
            evidence.append(
                Evidence(
                    title=f"{name} ({company.get('country') or 'N/A'})",
                    content=summary,
                    source=company.get("website") or "",
                    metadata=metadata,
                )
            )

        if evidence:
            top_names = [ev.metadata.get("name") or ev.title for ev in evidence[:3]]
            summary = f"Tracxn surfaced companies: {', '.join(top_names)}"
            confidence = "medium"
            citations = [ev.source for ev in evidence if ev.source][:3]
        else:
            summary = "No relevant Tracxn records returned."
            confidence = "low"
            citations = []

        return AgentResponse(
            agent_name=self.name,
            task_id=task.task_id,
            summary=summary,
            confidence=confidence,
            evidence=evidence,
            citations=citations,
            raw_output=result,
        )

