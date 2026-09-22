"""SerpApi-backed search agent."""

from __future__ import annotations

from typing import Dict, List

from .base import AgentResponse, BaseAgent, Evidence, Task
from .tools import SerpApiTool


class SerpSearchAgent(BaseAgent):
    name = "SerpSearchAgent"

    def __init__(self, tools: Dict[str, object]):
        super().__init__(tools)
        if "SerpApiTool" not in tools:
            raise ValueError("SerpSearchAgent requires SerpApiTool")

    def execute(self, task: Task) -> AgentResponse:
        tool: SerpApiTool = self.tools["SerpApiTool"]  # type: ignore
        payload = {
            "query": task.inputs.get("query") or task.instruction,
            "limit": task.inputs.get("limit", 10),
            "location": task.inputs.get("location"),
            "search_type": task.inputs.get("search_type"),
        }
        result = tool.run(payload)
        if result.get("error"):
            return AgentResponse(
                agent_name=self.name,
                task_id=task.task_id,
                summary=f"SerpApi search failed: {result['error']}",
                confidence="low",
                evidence=[],
                citations=[],
                raw_output=result,
            )

        entries = result.get("results", [])
        region_hint = (
            task.inputs.get("location")
            or task.inputs.get("geography")
            or task.inputs.get("region")
            or ""
        ).strip()
        evidence: List[Evidence] = []
        for item in entries:
            metadata = dict(item)
            if region_hint and not metadata.get("region_hint"):
                metadata["region_hint"] = region_hint
            evidence.append(
                Evidence(
                    title=item.get("title") or item.get("link") or "Search result",
                    content=item.get("snippet") or "",
                    source=item.get("link") or "",
                    metadata=metadata,
                )
            )

        if evidence:
            summary_titles = [ev.title for ev in evidence[:3]]
            summary = f"SerpApi surfaced sources: {', '.join(summary_titles)}"
            confidence = "medium"
            citations = [ev.source for ev in evidence if ev.source][:3]
        else:
            summary = "No relevant search results returned."
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

