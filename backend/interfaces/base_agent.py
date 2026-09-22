from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

from backend.models import AgentResponse, PlannerTask


@dataclass
class AgentContext:
    task: PlannerTask
    prior_responses: Mapping[str, AgentResponse]
    tools: Mapping[str, object]


class AgentExecutionError(Exception):
    """Raised when an agent cannot complete its task."""


class BaseAgent(abc.ABC):
    name: str
    required_tools: Iterable[str] = ()
    required_fields: Iterable[str] = ()

    def __init__(self, tools: Mapping[str, object]) -> None:
        self.name = self.__class__.__name__
        self._tools = tools

    @property
    def tools(self) -> Mapping[str, object]:
        return self._tools

    def validate_requirements(self, task: PlannerTask) -> List[str]:
        missing: List[str] = []
        for field in self.required_fields:
            if field not in task.inputs:
                missing.append(field)
        return missing

    def ensure_tools(self) -> Dict[str, object]:
        missing = [tool for tool in self.required_tools if tool not in self._tools]
        if missing:
            raise AgentExecutionError(f"Agent {self.name} missing tools: {missing}")
        return {tool: self._tools[tool] for tool in self.required_tools if tool in self._tools}

    @abc.abstractmethod
    def execute(self, context: AgentContext) -> AgentResponse:
        """Execute agent logic using provided context."""

