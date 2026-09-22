from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, MutableMapping, Optional

from backend.models import AgentResponse, EvidenceItem, IdeaBrief, PlannerTask


@dataclass
class SessionContext:
    session_id: str
    idea_brief: Optional[IdeaBrief] = None
    tasks: List[PlannerTask] = field(default_factory=list)
    responses: MutableMapping[str, AgentResponse] = field(default_factory=dict)
    evidence: List[EvidenceItem] = field(default_factory=list)


class ContextStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionContext] = {}

    def create_session(self, session_id: str) -> SessionContext:
        session = SessionContext(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        return self._sessions.get(session_id)

    def add_task(self, session_id: str, task: PlannerTask) -> None:
        session = self._require_session(session_id)
        session.tasks.append(task)

    def set_idea_brief(self, session_id: str, idea_brief: IdeaBrief) -> None:
        session = self._require_session(session_id)
        session.idea_brief = idea_brief

    def add_response(self, session_id: str, response: AgentResponse) -> None:
        session = self._require_session(session_id)
        session.responses[response.task_id] = response
        session.evidence.extend(response.evidence)

    def get_responses(self, session_id: str) -> Mapping[str, AgentResponse]:
        session = self._require_session(session_id)
        return session.responses

    def get_evidence(self, session_id: str) -> List[EvidenceItem]:
        session = self._require_session(session_id)
        return session.evidence

    def _require_session(self, session_id: str) -> SessionContext:
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError(f"Session {session_id} not found in context store.")
        return session

