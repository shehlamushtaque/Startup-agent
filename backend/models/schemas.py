from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator


class MaturityStage(str, Enum):
    concept = "concept"
    mvp = "mvp"
    pre_seed = "pre-seed"
    seed = "seed"
    growth = "growth"


class Citation(BaseModel):
    label: str
    url: str
    excerpt: str


class EvidenceSource(str, Enum):
    academic = "academic"
    news = "news"
    statistical = "statistical"
    linkedin = "linkedin"
    report = "report"
    internal = "internal"


class EvidenceItem(BaseModel):
    id: str
    content: str
    source_type: EvidenceSource
    source_ref: str
    published_at: datetime
    relevance_score: float = Field(ge=0.0, le=1.0)

    @validator("content")
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Evidence content must not be empty.")
        return value


class IdeaBrief(BaseModel):
    idea_id: str
    raw_input: str
    problem: Optional[str] = None
    audience: Optional[str] = None
    region: Optional[str] = None
    maturity_stage: Optional[MaturityStage] = None
    goals: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)

    @validator("idea_id", "raw_input")
    def ensure_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field must not be empty.")
        return value


class PlannerTask(BaseModel):
    task_id: str
    agent: str
    instruction: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    priority: int = 0

    @validator("task_id", "agent", "instruction")
    def ensure_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field must not be empty.")
        return value


class AgentConfidence(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class AgentResponse(BaseModel):
    task_id: str
    agent_name: str
    summary: str
    evidence: List[EvidenceItem] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    confidence: AgentConfidence = AgentConfidence.medium
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("task_id", "agent_name", "summary")
    def ensure_string_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field must not be empty.")
        return value


class Section(BaseModel):
    title: str
    narrative: str
    score: Optional[float] = None
    supporting_points: List[str] = Field(default_factory=list)

    @validator("title", "narrative")
    def ensure_section_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Section fields must not be empty.")
        return value

    @validator("score")
    def validate_score(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        if not 0.0 <= value <= 10.0:
            raise ValueError("Section score must be between 0 and 10.")
        return value


class BusinessReport(BaseModel):
    idea_id: str
    executive_summary: str
    vision_analysis: Section
    health_scores: Section
    forecasts: Section
    recommendations: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    references: List[Citation] = Field(default_factory=list)

    @validator("idea_id", "executive_summary")
    def ensure_report_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Report fields must not be empty.")
        return value

