"""Utilities for persisting structured evidence snapshots for later analysis."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _slugify(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    normalized = normalized.strip("-")
    return normalized or fallback


def persist_structured_evidence(
    structured_data: Dict[str, Any],
    idea_context: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Path:
    """Persist structured evidence + context to disk for downstream LLM analysis."""
    output_dir = output_dir or Path("data/processed/report_runs")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    idea_slug = _slugify(str(idea_context.get("idea", "")), "idea")
    region_slug = _slugify(str(idea_context.get("region", "")), "global")
    filename = f"{timestamp}_{idea_slug}_{region_slug}.json"
    path = output_dir / filename

    payload = {
        "timestamp": timestamp,
        "idea_context": idea_context,
        "structured_evidence": structured_data,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path

