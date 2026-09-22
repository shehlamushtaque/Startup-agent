"""
analyze_ratings.py
------------------
Takes completed rater CSV files, computes:
  - Per-dimension mean and SD for HARIS vs Template
  - Krippendorff's alpha per dimension and overall
  - Wilcoxon signed-rank test (HARIS vs Template per report)
  - Final Table 6 for the paper

Usage:
  # After raters submit their CSVs:
  python scripts/analyze_ratings.py \
      --ratings data/eval/ratings_R1.csv data/eval/ratings_R2.csv data/eval/ratings_R3.csv \
      --manifest data/eval/report_manifest.csv \
      --blind   data/eval/blind_manifest.csv

  # Or point to a folder with all rating files:
  python scripts/analyze_ratings.py --ratings_dir data/eval/

  # Demo mode with synthetic data (for testing the pipeline):
  python scripts/analyze_ratings.py --demo
"""

from __future__ import annotations

import argparse
import os
import sys
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import krippendorff
from scipy import stats


DIMENSIONS = ["relevance", "completeness", "actionability", "citation_grounding"]
DIMENSIONS_DISPLAY = ["Relevance", "Completeness", "Actionability", "Citation Grounding"]


# ── data loading ─────────────────────────────────────────────────────────────

def load_ratings(rating_files: list[str]) -> list[dict]:
    """Load and merge all rater CSV files into a list of dicts."""
    rows = []
    for path in rating_files:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip empty rows (template rows not filled in)
                if not any(row.get(d, "").strip() for d in DIMENSIONS):
                    continue
                rows.append({
                    "rater_id": row["rater_id"].strip(),
                    "blind_id": row["blind_id"].strip(),
                    "relevance": int(row["relevance"]),
                    "completeness": int(row["completeness"]),
                    "actionability": int(row["actionability"]),
                    "citation_grounding": int(row["citation_grounding"]),
                    "comments": row.get("comments", "").strip(),
                })
    return rows


def load_manifest(manifest_path: str) -> dict[str, dict]:
    """Load report_manifest.csv -> {report_id: {company, category, haris_is_A}}."""
    manifest = {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest[row["report_id"]] = {
                "company": row["company"],
                "category": row["category"],
                "haris_is_A": row["haris_is_A"].lower() in ("true", "1", "yes"),
            }
    return manifest


def load_blind_manifest(blind_path: str) -> dict[str, dict]:
    """Load blind_manifest.csv -> {blind_id: {report_id, version}}."""
    blind = {}
    with open(blind_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            blind[row["blind_id"]] = {
                "report_id": row["report_id"],
                "version": row["version"],
                "category": row["category"],
            }
    return blind


def resolve_system(blind_id: str, blind_manifest: dict, report_manifest: dict) -> str:
    """Given a blind_id like 'R01-A', return 'HARIS' or 'TEMPLATE'."""
    info = blind_manifest.get(blind_id, {})
    report_id = info.get("report_id", "")
    version = info.get("version", "A")
    rep = report_manifest.get(report_id, {})
    haris_is_A = rep.get("haris_is_A", True)
    if (version == "A" and haris_is_A) or (version == "B" and not haris_is_A):
        return "HARIS"
    return "TEMPLATE"


# ── statistics ───────────────────────────────────────────────────────────────

def compute_means_sds(ratings: list[dict], system: str) -> dict[str, tuple[float, float]]:
    """Return {dimension: (mean, sd)} for the given system."""
    result = {}
    for dim in DIMENSIONS:
        values = [r[dim] for r in ratings if r.get("system") == system]
        if not values:
            result[dim] = (float("nan"), float("nan"))
        else:
            result[dim] = (
                statistics.mean(values),
                statistics.stdev(values) if len(values) > 1 else 0.0,
            )
    return result


def compute_krippendorff(ratings: list[dict], system: str, dim: str) -> float:
    """
    Compute Krippendorff's alpha for one dimension, one system.
    Rows = raters, columns = items (report x system combinations).
    """
    raters = sorted(set(r["rater_id"] for r in ratings))
    items = sorted(set(r["blind_id"] for r in ratings if r.get("system") == system))
    if len(raters) < 2 or len(items) < 2:
        return float("nan")

    # Build matrix: raters x items, NaN for missing
    matrix = np.full((len(raters), len(items)), np.nan)
    rater_idx = {r: i for i, r in enumerate(raters)}
    item_idx = {it: i for i, it in enumerate(items)}
    for r in ratings:
        if r.get("system") != system:
            continue
        ri = rater_idx.get(r["rater_id"])
        ci = item_idx.get(r["blind_id"])
        if ri is not None and ci is not None:
            matrix[ri, ci] = r[dim]

    try:
        return krippendorff.alpha(matrix, level_of_measurement="ordinal")
    except Exception:
        return float("nan")


def compute_krippendorff_overall(ratings: list[dict]) -> float:
    """Overall Krippendorff's alpha across all dimensions and systems."""
    raters = sorted(set(r["rater_id"] for r in ratings))
    if len(raters) < 2:
        return float("nan")

    # One column per (blind_id, dimension) pair
    all_keys = sorted(set((r["blind_id"], d) for r in ratings for d in DIMENSIONS))
    matrix = np.full((len(raters), len(all_keys)), np.nan)
    rater_idx = {r: i for i, r in enumerate(raters)}
    key_idx = {k: i for i, k in enumerate(all_keys)}

    for r in ratings:
        ri = rater_idx.get(r["rater_id"])
        for dim in DIMENSIONS:
            ci = key_idx.get((r["blind_id"], dim))
            if ri is not None and ci is not None:
                matrix[ri, ci] = r.get(dim, np.nan)

    try:
        return krippendorff.alpha(matrix, level_of_measurement="ordinal")
    except Exception:
        return float("nan")


def wilcoxon_test(ratings: list[dict], dim: str) -> tuple[float, float]:
    """
    Wilcoxon signed-rank test: HARIS vs Template scores per report.
    Returns (statistic, p_value).
    """
    # Average across raters per (report_id, system, dim)
    from collections import defaultdict
    scores: dict[tuple, list] = defaultdict(list)
    for r in ratings:
        key = (r.get("report_id", r["blind_id"]), r.get("system", "?"))
        scores[key].append(r[dim])

    report_ids = sorted(set(k[0] for k in scores))
    haris_scores = []
    template_scores = []
    for rid in report_ids:
        h = scores.get((rid, "HARIS"), [])
        t = scores.get((rid, "TEMPLATE"), [])
        if h and t:
            haris_scores.append(statistics.mean(h))
            template_scores.append(statistics.mean(t))

    if len(haris_scores) < 3:
        return float("nan"), float("nan")

    try:
        result = stats.wilcoxon(haris_scores, template_scores, alternative="greater")
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return float("nan"), float("nan")


# ── report printing ──────────────────────────────────────────────────────────

def print_table6(haris_stats: dict, template_stats: dict, alpha: float) -> None:
    """Print the paper-ready Table 6."""
    print("\n" + "=" * 70)
    print("TABLE 6. Pilot synthesis quality (5-point Likert; n=10 reports, 3 raters)")
    print(f"Krippendorff's alpha (overall) = {alpha:.2f}")
    print("=" * 70)
    print(f"{'System':<20} {'Relevance':>12} {'Completeness':>14} {'Actionability':>15} {'Citation Grnd':>15}")
    print("-" * 70)

    for system, stat_dict in [("Template-only", template_stats), ("HARIS (LLM)", haris_stats)]:
        row_parts = []
        for dim in DIMENSIONS:
            mean, sd = stat_dict.get(dim, (float("nan"), float("nan")))
            if mean != mean:
                row_parts.append("  N/A")
            else:
                row_parts.append(f"{mean:.1f} +/- {sd:.1f}")
        print(f"{system:<20} {row_parts[0]:>12} {row_parts[1]:>14} {row_parts[2]:>15} {row_parts[3]:>15}")

    print("=" * 70)
    print()
    print("Paper copy-paste format (Markdown):")
    print()
    print("| System | Relevance | Completeness | Actionability | Citation Grounding |")
    print("|---|---|---|---|---|")
    for system, stat_dict in [("Template-only", template_stats), ("**HARIS (LLM synthesis)**", haris_stats)]:
        cells = []
        for dim in DIMENSIONS:
            mean, sd = stat_dict.get(dim, (float("nan"), float("nan")))
            if mean != mean:
                cells.append("N/A")
            else:
                cells.append(f"**{mean:.1f} +/- {sd:.1f}**" if "HARIS" in system else f"{mean:.1f} +/- {sd:.1f}")
        print(f"| {system} | {' | '.join(cells)} |")

    print()
    print(f"Table caption: Pilot synthesis quality (5-point Likert; n=10 reports, 3 raters;")
    print(f"               Krippendorff's alpha = {alpha:.2f})")


def print_wilcoxon(ratings: list[dict]) -> None:
    print("\n--- Wilcoxon Signed-Rank Tests (HARIS > Template) ---")
    print(f"{'Dimension':<25} {'W statistic':>12} {'p-value':>12} {'Significant (p<0.05)':>22}")
    print("-" * 75)
    for dim, disp in zip(DIMENSIONS, DIMENSIONS_DISPLAY):
        w, p = wilcoxon_test(ratings, dim)
        sig = "Yes" if p < 0.05 else "No" if p == p else "N/A"
        p_str = f"{p:.4f}" if p == p else "N/A"
        w_str = f"{w:.1f}" if w == w else "N/A"
        print(f"{disp:<25} {w_str:>12} {p_str:>12} {sig:>22}")


# ── demo mode ────────────────────────────────────────────────────────────────

def generate_demo_ratings() -> list[dict]:
    """Generate synthetic ratings consistent with paper Table 6 numbers."""
    import random
    random.seed(99)

    # Target means: HARIS ~4.0/3.7/3.8/3.6, Template ~2.9/2.3/2.7/1.8
    targets = {
        "HARIS": {"relevance": 4.0, "completeness": 3.7, "actionability": 3.8, "citation_grounding": 3.6},
        "TEMPLATE": {"relevance": 2.9, "completeness": 2.3, "actionability": 2.7, "citation_grounding": 1.8},
    }

    rows = []
    for rater_num in range(1, 4):
        rater_id = f"R{rater_num}"
        for report_num in range(1, 11):
            report_id = f"R{report_num:02d}"
            for version, system in [("A", "HARIS"), ("B", "TEMPLATE")]:
                blind_id = f"{report_id}-{version}"
                scores = {}
                for dim in DIMENSIONS:
                    target = targets[system][dim]
                    # Gaussian noise around target, clipped to [1,5]
                    val = round(random.gauss(target, 0.5))
                    val = max(1, min(5, val))
                    scores[dim] = val
                rows.append({
                    "rater_id": rater_id,
                    "blind_id": blind_id,
                    "report_id": report_id,
                    "system": system,
                    **scores,
                    "comments": "",
                })
    return rows


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze rater evaluation data.")
    parser.add_argument("--ratings", nargs="*", default=[], help="Rating CSV file paths")
    parser.add_argument("--ratings_dir", default=None, help="Directory with rating CSV files")
    parser.add_argument("--manifest", default="data/eval/report_manifest.csv")
    parser.add_argument("--blind", default="data/eval/blind_manifest.csv")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic demo data")
    parser.add_argument("--output", default="data/eval/analysis_results.json")
    args = parser.parse_args()

    if args.demo:
        print("DEMO MODE — using synthetic ratings consistent with paper Table 6")
        ratings = generate_demo_ratings()
        # System labels already embedded
    else:
        # Load real rating files
        rating_files = list(args.ratings)
        if args.ratings_dir:
            for fname in os.listdir(args.ratings_dir):
                if fname.startswith("ratings_") and fname.endswith(".csv"):
                    rating_files.append(os.path.join(args.ratings_dir, fname))
        if not rating_files:
            print("No rating files found. Use --demo to test with synthetic data.")
            sys.exit(1)

        print(f"Loading ratings from {len(rating_files)} file(s)...")
        ratings = load_ratings(rating_files)

        # Resolve system labels via manifests
        if os.path.exists(args.manifest) and os.path.exists(args.blind):
            report_manifest = load_manifest(args.manifest)
            blind_manifest = load_blind_manifest(args.blind)
            for r in ratings:
                r["system"] = resolve_system(r["blind_id"], blind_manifest, report_manifest)
                info = blind_manifest.get(r["blind_id"], {})
                r["report_id"] = info.get("report_id", r["blind_id"].split("-")[0])
        else:
            print("WARNING: Manifest files not found. Cannot resolve system labels.")

    print(f"Total rating rows: {len(ratings)}")
    n_haris = sum(1 for r in ratings if r.get("system") == "HARIS")
    n_template = sum(1 for r in ratings if r.get("system") == "TEMPLATE")
    print(f"  HARIS ratings: {n_haris}")
    print(f"  Template ratings: {n_template}")

    # Compute stats
    haris_stats = compute_means_sds(ratings, "HARIS")
    template_stats = compute_means_sds(ratings, "TEMPLATE")

    # Krippendorff's alpha per dimension
    print("\n--- Inter-rater Reliability (Krippendorff's alpha) ---")
    alphas = {}
    for dim, disp in zip(DIMENSIONS, DIMENSIONS_DISPLAY):
        alpha_h = compute_krippendorff(ratings, "HARIS", dim)
        alpha_t = compute_krippendorff(ratings, "TEMPLATE", dim)
        alphas[dim] = {"HARIS": alpha_h, "TEMPLATE": alpha_t}
        print(f"  {disp:<25}: HARIS alpha={alpha_h:.2f}  Template alpha={alpha_t:.2f}")

    overall_alpha = compute_krippendorff_overall(ratings)
    print(f"\n  Overall alpha (all dims + systems): {overall_alpha:.2f}")
    print("  Interpretation: >= 0.80 excellent, >= 0.67 acceptable, < 0.60 marginal")

    # Print Table 6
    print_table6(haris_stats, template_stats, overall_alpha)

    # Wilcoxon tests
    print_wilcoxon(ratings)

    # Save results
    results = {
        "n_ratings": len(ratings),
        "n_haris": n_haris,
        "n_template": n_template,
        "haris_stats": {dim: list(v) for dim, v in haris_stats.items()},
        "template_stats": {dim: list(v) for dim, v in template_stats.items()},
        "overall_krippendorff_alpha": overall_alpha,
        "per_dimension_alpha": alphas,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
