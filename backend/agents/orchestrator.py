"""Agent orchestrator responsible for executing planner tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .base import AgentResponse, Task
from .competitor import CompetitorGrowthAgent
from .funding import FundingIntelligenceAgent
from .registry import AgentConfig, ToolConfig, build_agent_instances, build_tool_instances
from .tools import (
    BraveSearchTool,
    CompanyFundingTool,
    CompanyQAIndexTool,
    CompetitorGrowthTool,
    SecFormDTool,
    SerpApiTool,
)
from .web import WebResearchAgent
from .sec import SecFilingsAgent
from .serp import SerpSearchAgent


DEFAULT_TOOL_CONFIGS: List[ToolConfig] = [
    ToolConfig(
        name="CompanyFundingTool",
        description="Access funding metrics from master companies table.",
        class_path="backend.agents.tools.CompanyFundingTool",
    ),
    ToolConfig(
        name="CompetitorGrowthTool",
        description="Retrieve competitor growth metrics.",
        class_path="backend.agents.tools.CompetitorGrowthTool",
    ),
    ToolConfig(
        name="CompanyQAIndexTool",
        description="Semantic search over the company QA index.",
        class_path="backend.agents.tools.CompanyQAIndexTool",
    ),
    ToolConfig(
        name="BraveSearchTool",
        description="Perform Brave Search queries.",
        class_path="backend.agents.tools.BraveSearchTool",
    ),
    ToolConfig(
        name="SerpApiTool",
        description="Perform Google search via SerpApi.",
        class_path="backend.agents.tools.SerpApiTool",
    ),
    ToolConfig(
        name="SecFormDTool",
        description="Fetch recent SEC Form D filings.",
        class_path="backend.agents.tools.SecFormDTool",
    ),
]

DEFAULT_AGENT_CONFIGS: List[AgentConfig] = [
    AgentConfig(
        name="FundingIntelligenceAgent",
        description="Summarise funding comparables and key investors.",
        class_path="backend.agents.funding.FundingIntelligenceAgent",
        tool_names=["CompanyFundingTool", "CompanyQAIndexTool"],
        requires=["industry", "geography"],
        outputs=["summary", "funding_totals", "investor_list"],
    ),
    AgentConfig(
        name="CompetitorGrowthAgent",
        description="Identify competitors and growth metrics.",
        class_path="backend.agents.competitor.CompetitorGrowthAgent",
        tool_names=["CompetitorGrowthTool"],
        requires=["industry"],
        outputs=["summary", "growth_signals"],
    ),
    AgentConfig(
        name="WebResearchAgent",
        description="Fetch supporting web references via Google search.",
        class_path="backend.agents.web.WebResearchAgent",
        tool_names=["BraveSearchTool"],
        requires=["query"],
        outputs=["summary", "citations", "evidence"],
    ),
    AgentConfig(
        name="SerpSearchAgent",
        description="Retrieve additional web intelligence via SerpApi.",
        class_path="backend.agents.serp.SerpSearchAgent",
        tool_names=["SerpApiTool"],
        requires=["query"],
        outputs=["summary", "citations", "evidence"],
    ),
    AgentConfig(
        name="SecFilingsAgent",
        description="Retrieve recent SEC Form D filings for comparable raises.",
        class_path="backend.agents.sec.SecFilingsAgent",
        tool_names=["SecFormDTool"],
        requires=["search"],
        outputs=["summary", "evidence", "citations"],
    ),
]


@dataclass
class Orchestrator:
    agent_configs: List[AgentConfig] = field(default_factory=lambda: DEFAULT_AGENT_CONFIGS)
    tool_configs: List[ToolConfig] = field(default_factory=lambda: DEFAULT_TOOL_CONFIGS)
    agents: Dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        tools = build_tool_instances(self.tool_configs)
        self.agents = build_agent_instances(self.agent_configs, tools)

    def run_tasks(self, tasks: List[Task]) -> List[AgentResponse]:
        responses: List[AgentResponse] = []
        for task in tasks:
            agent = self.agents.get(task.agent_name)
            if not agent:
                responses.append(
                    AgentResponse(
                        agent_name=task.agent_name,
                        task_id=task.task_id,
                        summary=f"Agent {task.agent_name} not available.",
                        confidence="low",
                    )
                )
                continue

            try:
                response = agent.execute(task)
                responses.append(response)
            except Exception as exc:  # pragma: no cover - defensive
                responses.append(
                    AgentResponse(
                        agent_name=task.agent_name,
                        task_id=task.task_id,
                        summary=f"Agent execution failed: {exc}",
                        confidence="low",
                        raw_output={"error": str(exc)},
                    )
                )
        return responses

