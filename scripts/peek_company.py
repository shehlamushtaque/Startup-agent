"""
Quick utility to inspect rows in master_companies_enriched for given keywords.

Usage:
    python scripts/peek_company.py MintM
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/peek_company.py <keyword1> [keyword2 ...]")
        sys.exit(1)

    keywords = [arg.lower() for arg in sys.argv[1:]]
    base_path = Path("data/processed")
    paths = [
        base_path / "master_companies_enriched.parquet",
        base_path / "master_companies.parquet",
    ]

    path = next((p for p in paths if p.exists()), None)
    if not path:
        raise FileNotFoundError("No master companies parquet file found.")

    df = pd.read_parquet(path)
    df["match"] = False
    for keyword in keywords:
        mask = df["normalized_name"].fillna("").str.lower().str.contains(keyword)
        df.loc[mask, "match"] = True

    matches = df[df["match"]]
    if matches.empty:
        print("No rows matched keywords:", ", ".join(keywords))
        return

    cols = [
        "normalized_name",
        "name_cb",
        "growjo_growth_percent",
        "growjo_estimated_revenue_usd",
        "growjo_employees",
        "growjo_rank",
        "funding_total_usd",
        "funding_round_count",
    ]
    existing_cols = [col for col in cols if col in matches.columns]
    print(matches[existing_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

