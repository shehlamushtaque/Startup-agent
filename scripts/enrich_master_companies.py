"""
Enrich the master_companies.parquet file with metrics sourced from growjo_clean.csv.

Usage:
    python scripts/enrich_master_companies.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = ROOT / "data" / "processed" / "master_companies.parquet"
GROWJO_PATH = ROOT / "data" / "processed" / "growjo_clean.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "master_companies_enriched.parquet"


def _normalize_name(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = "".join(ch for ch in value.lower() if ch.isalnum())
    return normalized or None


def _load_master() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master file not found at {MASTER_PATH}")
    master = pd.read_parquet(MASTER_PATH)
    candidate_cols = [
        "normalized_name",
        "name_cb",
        "company_name",
    ]
    norm_source = None
    for col in candidate_cols:
        if col in master.columns:
            norm_source = col
            break
    if norm_source is None:
        raise KeyError("No normalized name column found in master dataset.")
    master["__norm_name"] = master[norm_source].apply(_normalize_name)
    return master


def _identify_column(columns: pd.Index, keyword: str) -> Optional[str]:
    keyword = keyword.lower()
    for col in columns:
        if keyword in col.lower():
            return col
    return None


def _load_growjo() -> pd.DataFrame:
    if not GROWJO_PATH.exists():
        raise FileNotFoundError(f"Growjo file not found at {GROWJO_PATH}")
    growjo = pd.read_csv(GROWJO_PATH)
    name_col = _identify_column(growjo.columns, "company")
    if not name_col:
        raise KeyError("Could not find a company column in growjo dataset.")
    growjo["__norm_name"] = growjo[name_col].apply(_normalize_name)

    mapping: Dict[str, Optional[str]] = {
        "growjo_rank": _identify_column(growjo.columns, "rank"),
        "growjo_growth_percent": _identify_column(growjo.columns, "growth"),
        "growjo_employees": _identify_column(growjo.columns, "employee"),
        "growjo_estimated_revenue_usd": _identify_column(growjo.columns, "revenue"),
    }

    selected = growjo["__norm_name"].to_frame()
    for target, source in mapping.items():
        if source:
            selected[target] = growjo[source]
    return selected


def main() -> None:
    master = _load_master()
    growjo = _load_growjo()

    merged = master.merge(growjo, on="__norm_name", how="left", suffixes=("", "_growjo"))

    metrics = [
        "growjo_rank",
        "growjo_growth_percent",
        "growjo_employees",
        "growjo_estimated_revenue_usd",
    ]

    for metric in metrics:
        growjo_metric = f"{metric}_growjo"
        if growjo_metric in merged.columns:
            master[metric] = master.get(metric).combine_first(merged[growjo_metric])
            merged.drop(columns=[growjo_metric], inplace=True, errors="ignore")

    master.drop(columns=["__norm_name"], inplace=True, errors="ignore")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(OUTPUT_PATH, index=False)
    print(f"Enriched master dataset written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

