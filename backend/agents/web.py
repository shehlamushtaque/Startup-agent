"""Web research agent leveraging Google Programmable Search."""

from __future__ import annotations

from typing import Dict, List

from .base import AgentResponse, BaseAgent, Evidence, Task
from .tools import BraveSearchTool


class WebResearchAgent(BaseAgent):
    name = "WebResearchAgent"

    def __init__(self, tools: Dict[str, object]):
        super().__init__(tools)
        if "BraveSearchTool" not in tools:
            raise ValueError("WebResearchAgent requires BraveSearchTool")

    def execute(self, task: Task) -> AgentResponse:
        search_tool: BraveSearchTool = self.tools["BraveSearchTool"]  # type: ignore
        query = task.inputs.get("query") or task.instruction
        results = search_tool.run({"query": query, "num_results": task.inputs.get("num_results", 5)})

        items = results.get("results", [])
        region_hint = (
            task.inputs.get("location")
            or task.inputs.get("geography")
            or task.inputs.get("region")
            or ""
        ).strip()
        evidence: List[Evidence] = []
        for item in items:
            metadata = dict(item)
            if region_hint and not metadata.get("region_hint"):
                metadata["region_hint"] = region_hint
            evidence.append(
                Evidence(
                    title=item.get("title") or item.get("display_link") or "Search result",
                    content=item.get("snippet", ""),
                    source=item.get("link", ""),
                    metadata=metadata,
                )
            )

        if evidence:
            top_titles = [ev.title for ev in evidence[:3]]
            summary = f"Top web sources: {', '.join(top_titles)}"
            confidence = "medium"
            citations = [ev.source for ev in evidence[:3] if ev.source]
        else:
            summary = "No web results returned for this query."
            confidence = "low"
            citations = []

        return AgentResponse(
            agent_name=self.name,
            task_id=task.task_id,
            summary=summary,
            confidence=confidence,
            evidence=evidence,
            citations=citations,
        )

