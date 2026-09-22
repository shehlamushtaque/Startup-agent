"""
Expand HARIS benchmark: 145 -> 300 companies (30 per category x 10 categories).

Strategy:
 - Load existing 145-company benchmark (retains seed-42 companies).
 - For each paper_cat, sample additional companies from objects.csv
   (no duplicates with existing benchmark) to reach 30 total.
 - legaltech: supplemented with 'consulting'/'public_relations' codes
   since only 11 'legal' companies exist in Crunchbase public data.
 - Output: data/processed/benchmark_companies_300.csv

Usage:
  python scripts/expand_benchmark.py
  python scripts/expand_benchmark.py --overwrite   # replaces benchmark_companies.csv too
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

# ── Category code mapping (Crunchbase → paper_cat) ──────────────────────────
CAT_MAP: dict[str, list[str]] = {
    "tech":        ["mobile", "software", "web", "enterprise", "hardware",
                    "analytics", "cloud", "network_hosting", "security"],
    "fintech":     ["finance"],
    "healthcare":  ["health", "medical"],
    "biotech":     ["biotech"],
    "cleantech":   ["cleantech"],
    "edtech":      ["education"],
    "retail":      ["ecommerce", "advertising"],
    "logistics":   ["manufacturing", "transportation"],
    "legaltech":   ["legal", "consulting", "public_relations"],  # extended for n=300
    "novel":       ["games_video", "social", "other"],
}

TARGET_PER_CAT = 30   # target per paper_cat


def load_raw_objects(n_per_cat: int = TARGET_PER_CAT) -> pd.DataFrame:
    """Load and filter objects.csv to candidate companies."""
    path = Path("data/raw/objects.csv")
    needed_codes = {c for codes in CAT_MAP.values() for c in codes}

    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    # Keep only relevant columns for efficiency
    keep_cols = ["id", "entity_type", "entity_id", "parent_id", "name",
                 "normalized_name", "permalink", "category_code", "status",
                 "founded_at", "closed_at", "domain", "homepage_url",
                 "twitter_username", "logo_url", "logo_width", "logo_height",
                 "short_description", "description", "overview", "tag_list",
                 "country_code", "state_code", "city", "region",
                 "first_investment_at", "last_investment_at", "investment_rounds",
                 "invested_companies", "first_funding_at", "last_funding_at",
                 "funding_rounds", "funding_total_usd", "first_milestone_at",
                 "last_milestone_at", "milestones", "relationships",
                 "created_by", "created_at", "updated_at"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # Filter: valid category, non-trivial description
    df = df[df["category_code"].isin(needed_codes)]
    df = df[df["short_description"].notna() & (df["short_description"].str.len() > 20)]
    df = df.drop_duplicates(subset=["id"])
    return df


def assign_paper_cat(df: pd.DataFrame) -> pd.DataFrame:
    """Assign paper_cat based on category_code using CAT_MAP."""
    # Build reverse map: category_code -> paper_cat
    rev_map: dict[str, str] = {}
    for paper_cat, codes in CAT_MAP.items():
        for code in codes:
            # If already assigned (e.g. overlap), first wins
            if code not in rev_map:
                rev_map[code] = paper_cat
    df = df.copy()
    df["paper_cat"] = df["category_code"].map(rev_map)
    return df.dropna(subset=["paper_cat"])


def expand_benchmark(overwrite: bool = False) -> pd.DataFrame:
    existing_path = Path("data/processed/benchmark_companies.csv")
    out_path_300  = Path("data/processed/benchmark_companies_300.csv")

    # 1. Load existing benchmark
    existing = pd.read_csv(existing_path, encoding="latin1", low_memory=False)
    existing["paper_cat"] = existing["paper_cat"].fillna("tech").astype(str)
    existing_ids = set(existing["id"])
    print(f"Existing benchmark: {len(existing)} companies | "
          f"cats: {existing['paper_cat'].value_counts().to_dict()}")

    # 2. Load raw objects and assign paper_cat
    raw = load_raw_objects()
    raw = assign_paper_cat(raw)
    # Exclude companies already in benchmark
    raw = raw[~raw["id"].isin(existing_ids)]
    print(f"Candidate pool (new, filtered): {len(raw)} companies")

    # 3. Stratified sampling to reach TARGET_PER_CAT per category
    new_chunks: list[pd.DataFrame] = []
    for paper_cat, current in existing.groupby("paper_cat"):
        n_have  = len(current)
        n_need  = max(0, TARGET_PER_CAT - n_have)
        pool    = raw[raw["paper_cat"] == paper_cat]
        n_avail = len(pool)
        n_take  = min(n_need, n_avail)

        if n_take == 0:
            print(f"  {paper_cat}: already at {n_have} (need {TARGET_PER_CAT},"
                  f" pool has {n_avail}) -- adding 0")
            continue

        sample = pool.sample(n=n_take, random_state=42)
        new_chunks.append(sample)
        print(f"  {paper_cat}: {n_have} -> {n_have + n_take}"
              f"  (pool: {n_avail}; took {n_take})")

    # 4. Combine
    new_all = pd.concat(new_chunks, ignore_index=True) if new_chunks else pd.DataFrame()
    expanded = pd.concat([existing, new_all], ignore_index=True)

    # 5. Summary
    final_counts = expanded["paper_cat"].value_counts()
    print(f"\nExpanded benchmark: {len(expanded)} companies")
    print("Per category:", final_counts.to_dict())

    # 6. Save
    expanded.to_csv(out_path_300, index=False, encoding="utf-8")
    print(f"Saved: {out_path_300}")

    if overwrite:
        expanded.to_csv(existing_path, index=False, encoding="utf-8")
        print(f"Overwritten: {existing_path}")

    return expanded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand HARIS benchmark to ~300 companies.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Also overwrite benchmark_companies.csv (used by all experiments)")
    args = parser.parse_args()
    expand_benchmark(overwrite=args.overwrite)
