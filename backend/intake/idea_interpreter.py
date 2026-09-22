from __future__ import annotations

from typing import Iterable, List, Optional

from backend.models import IdeaBrief, MaturityStage


class IdeaInterpreter:
    """Simple interpreter turning raw CLI input into an IdeaBrief."""

    REQUIRED_FIELDS: Iterable[str] = ("problem", "audience", "region")

    def interpret(
        self,
        *,
        idea: str,
        problem: Optional[str],
        audience: Optional[str],
        region: Optional[str],
        maturity: Optional[str],
        goals: Optional[str],
        assumptions: Optional[str],
    ) -> IdeaBrief:
        goals_list: List[str] = []
        if goals:
            goals_list = [entry.strip() for entry in goals.split(",") if entry.strip()]

        assumptions_list: List[str] = []
        if assumptions:
            assumptions_list = [entry.strip() for entry in assumptions.split(",") if entry.strip()]

        maturity_stage = None
        if maturity:
            try:
                maturity_stage = MaturityStage(maturity)
            except ValueError:
                maturity_stage = None

        field_pairs = [("problem", problem), ("audience", audience), ("region", region)]
        missing_fields = [field for field, value in field_pairs if value is None or not value.strip()]

        return IdeaBrief(
            idea_id=idea[:32].replace(" ", "_").lower(),
            raw_input=idea,
            problem=problem,
            audience=audience,
            region=region,
            maturity_stage=maturity_stage,
            goals=goals_list,
            assumptions=assumptions_list,
            missing_fields=missing_fields,
        )

