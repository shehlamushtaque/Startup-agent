"""Base classes and dataclasses for the multi-agent orchestration system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class IdeaBrief:
    """Structured summary of the user's startup idea."""

    raw_idea: str
    industry: Optional[str] = None
    product: Optional[str] = None
    audience: Optional[str] = None
    geography: Optional[str] = None
    stage: Optional[str] = None
    pain_points: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Planner-issued instruction for a specific agent."""

    task_id: str
    agent_name: str
    instruction: str
    inputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    """Evidence snippet returned by an agent."""

    title: str
    content: str
    source: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Structured response from an agent execution."""

    agent_name: str
    task_id: str
    summary: str
    confidence: str = "unknown"
    evidence: List[Evidence] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    raw_output: Dict[str, Any] = field(default_factory=dict)


class BaseTool(Protocol):
    """Protocol that all tools should implement."""

    name: str

    def run(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool and return structured data."""
        ...


class BaseAgent:
    """Base class for specialist agents."""

    name: str = "BaseAgent"

    def __init__(self, tools: Dict[str, BaseTool]):
        self.tools = tools

    def execute(self, task: Task) -> AgentResponse:
        """Execute the task and produce a response."""
        raise NotImplementedError("Agent subclasses must override execute()")


