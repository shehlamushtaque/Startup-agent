from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import yaml

from backend.agents.registry import AgentConfig, ToolConfig


@dataclass
class AgentRegistry:
    agents: List[AgentConfig]
    tools: List[ToolConfig]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _normalise_tool_config(raw: Dict[str, Any]) -> ToolConfig:
    return ToolConfig(
        name=raw["name"],
        description=raw.get("description", ""),
        class_path=raw["class_path"],
        config=raw.get("config", {}),
    )


def _normalise_agent_config(raw: Dict[str, Any]) -> AgentConfig:
    return AgentConfig(
        name=raw["name"],
        description=raw.get("description", ""),
        class_path=raw["class_path"],
        tool_names=raw.get("tools", []),
        requires=raw.get("requires", []),
        outputs=raw.get("outputs", []),
    )


def load_agent_registry(path: Path) -> AgentRegistry:
    data = _load_yaml(path)
    tools = [_normalise_tool_config(entry) for entry in data.get("tools", [])]
    agents = [_normalise_agent_config(entry) for entry in data.get("agents", [])]
    return AgentRegistry(agents=agents, tools=tools)


def import_from_path(class_path: str) -> Type[Any]:
    module_name, _, class_name = class_path.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid class_path: {class_path}")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

