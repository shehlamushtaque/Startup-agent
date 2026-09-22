from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    payload: Dict[str, Any]
    metadata: Dict[str, Any]


class ToolError(Exception):
    """Raised when a tool fails after retries."""


class BaseTool(abc.ABC):
    name: str

    def __init__(self, name: Optional[str] = None, **config: Any) -> None:
        self.name = name or self.__class__.__name__
        self._config = config

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @abc.abstractmethod
    def execute(self, payload: Dict[str, Any]) -> ToolResult:
        """Execute tool with provided payload."""

    def validate_payload(self, payload: Dict[str, Any]) -> None:
        """Optional hook for validating payload before execution."""
        return None

