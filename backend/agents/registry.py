"""Agent and tool registry definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type

from .base import BaseAgent, BaseTool


@dataclass
class ToolConfig:
    name: str
    description: str
    class_path: str
    default_params: Dict[str, object] = field(default_factory=dict)


@dataclass
class AgentConfig:
    name: str
    description: str
    class_path: str
    tool_names: List[str]
    requires: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


def import_string(path: str):
    module_path, attr = path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[attr])
    return getattr(module, attr)


def build_tool_instances(configs: List[ToolConfig]) -> Dict[str, BaseTool]:
    tools: Dict[str, BaseTool] = {}
    for cfg in configs:
        tool_cls = import_string(cfg.class_path)
        tool: BaseTool = tool_cls(**cfg.default_params)
        tools[cfg.name] = tool
    return tools


def build_agent_instances(
    agent_configs: List[AgentConfig],
    tools: Dict[str, BaseTool],
) -> Dict[str, BaseAgent]:
    agents: Dict[str, BaseAgent] = {}
    for cfg in agent_configs:
        agent_cls: Type[BaseAgent] = import_string(cfg.class_path)
        agent_tools = {name: tools[name] for name in cfg.tool_names if name in tools}
        agent = agent_cls(agent_tools)
        agents[cfg.name] = agent
    return agents


