"""
Ablation Study: HARIS Hybrid vs. Baseline Variants
====================================================
Compares four planning strategies on the 145-company Crunchbase benchmark:

  1. Standard RAG  -- single vector retrieval, no agent orchestration
  2. Catalog-only  -- always _catalog_plan (no LLM)
  3. LLM-only      -- always _llm_plan (LLM required; falls back to catalog)
  4. HARIS Hybrid  -- kappa-routed hybrid (production default, kappa < 0.10)

Metrics reported per variant:
  - Plan validity (%)        : all tasks have valid agent + non-empty instruction
  - Mean tasks per plan      : depth of produced plan
  - Mean latency (ms)        : wall-clock time for one plan / retrieval call
  - LLM invocation rate (%)  : fraction of briefs that triggered the LLM path
  - Kappa (coverage) score   : mean domain-signal density of input brief

Output: data/processed/qa/ablation_results.json

Usage:
  python scripts/run_ablation.py
  python scripts/run_ablation.py --n 50           # smaller run for debugging
  python scripts/run_ablation.py --seed 123
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

# ── project root on sys.path ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from backend.models.schemas import IdeaBrief, MaturityStage
from backend.planning.research_planner import ResearchPlanner
from backend.agents.orchestrator import DEFAULT_AGENT_CONFIGS

AVAILABLE_AGENTS = {cfg.name for cfg in DEFAULT_AGENT_CONFIGS}

# ── catalog keyword set (118 terms, mirrors research_planner.py) ─────────────
CATALOG_K: set[str] = {
    "accounting","analytics","automation","api","cloud","data",
    "devops","digital","ecommerce","embedded","enterprise",
    "intelligence","internet","iot","machine","messaging",
    "microservices","mobile","networking","nlp","platform",
    "saas","sensor","software","streaming","tech","technology",
    "vision","wearable","web3",
    "banking","blockchain","capital","crypto","defi","finance",
    "fintech","hrtech","insurance","investment","lending",
    "martech","payment","payments","trading","wealth",
    "biotech","clinical","diagnostics","health","healthcare",
    "healthtech","hospital","medtech","medical","pharma",
    "telehealth","telemedicine","therapeutics",
    "carbon","cleantech","climate","energy","environmental",
    "greentech","renewable","solar","sustainability",
    "aerospace","autonomy","aviation","drone","lidar",
    "nanotech","quantum","robotics","semiconductor",
    "advertising","agtech","agriculture","apparel","automotive",
    "construction","consulting","consumer","content","defense",
    "education","edtech","electronics","engineering","fashion",
    "fitness","food","foodtech","gaming","hardware","hospitality",
    "industrial","legal","legaltech","logistics","manufacturing",
    "maritime","marketing","materials","media","mining",
    "recruiting","research","restaurant","retail","security",
    "sports","supply","transportation","travel","utilities",
    "wholesale",
}

KAPPA_THRESHOLD = 0.10   # calibrated 2025-Q1 on 145-company Crunchbase benchmark


# ── helpers ──────────────────────────────────────────────────────────────────

def load_benchmark(n: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    path = Path("data/processed/benchmark_companies.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark not found at {path}\n"
            "Run scripts/run_experiments.py first to generate it."
        )
    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    for col in ("short_description", "category_code", "country_code",
                "city", "region", "paper_cat", "name"):
        df[col] = df.get(col, pd.Series(dtype=str)).fillna("").astype(str)
    if n and n < len(df):
        df = df.sample(n, random_state=seed).reset_index(drop=True)
    return df


def row_to_brief(row: pd.Series) -> IdeaBrief:
    geo_parts = [p for p in [row.get("city", ""), row.get("region", ""),
                              row.get("country_code", "")] if p and p != "nan"]
    geo = ", ".join(geo_parts) or "United States"
    return IdeaBrief(
        idea_id=str(uuid4()),
        raw_input=str(row["short_description"])[:300],
        problem=str(row["short_description"])[:200],
        audience=f"customers in the {row['category_code']} sector",
        region=geo,
        maturity_stage=MaturityStage.seed,
    )


def kappa(brief: IdeaBrief) -> float:
    """Domain-coverage signal: |K-match| / |tokens|."""
    tokens = [t.lower() for t in (brief.problem or "").split() if len(t) > 3]
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in CATALOG_K) / len(tokens)


def is_valid_plan(tasks) -> bool:
    if not tasks:
        return False
    for t in tasks:
        if t.agent not in AVAILABLE_AGENTS:
            return False
        if not t.instruction:
            return False
    return True


# ── Standard RAG baseline ────────────────────────────────────────────────────

def _load_rag_store():
    """
    Try to load FAISS vector store from data/vector/.
    Returns None if index not built (graceful degradation).
    """
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        vector_dir = Path("data/vector")
        index_path = vector_dir / "company_qa.faiss"
        meta_path  = vector_dir / "company_qa_metadata.jsonl"
        cfg_path   = vector_dir / "company_qa_config.json"

        if not (index_path.exists() and meta_path.exists() and cfg_path.exists()):
            return None

        index = faiss.read_index(str(index_path))
        metadata: list = []
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    metadata.append(json.loads(line))
        cfg = json.loads(cfg_path.read_text())
        model = SentenceTransformer(cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
        return {"index": index, "metadata": metadata, "model": model}
    except Exception:
        return None


def _rag_fallback_search(brief: IdeaBrief, master_df: Optional[pd.DataFrame],
                         top_k: int = 5) -> list:
    """
    BM25-style keyword overlap retrieval against master_companies_enriched.parquet.
    Used when FAISS index is not built.
    """
    if master_df is None or master_df.empty:
        return []
    query_tokens = set(t.lower() for t in (brief.problem or "").split() if len(t) > 3)
    scores: list[tuple[float, str]] = []
    for _, row in master_df.iterrows():
        # build document from available text columns
        doc_parts = [str(row.get(c, "")) for c in
                     ("company_name", "description", "category_code_clean",
                      "funding_stage", "hq_state") if c in master_df.columns]
        doc_tokens = set(t.lower() for t in " ".join(doc_parts).split() if len(t) > 3)
        overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
        scores.append((overlap, str(row.get("company_name", ""))))
    scores.sort(reverse=True)
    return [name for _, name in scores[:top_k]]


def run_standard_rag(df: pd.DataFrame, rag_store, master_df) -> dict:
    """
    Variant 1: Standard RAG — no agent orchestration, no planner.
    Retrieves top-5 company snippets from vector index (or keyword fallback).
    Validity is n/a (no structured plan). Latency is retrieval time only.
    """
    print("\n" + "=" * 60)
    print("VARIANT 1: STANDARD RAG (no agent orchestration)")
    print("=" * 60)

    results = []
    for _, row in df.iterrows():
        brief = row_to_brief(row)
        k = kappa(brief)
        query = (brief.problem or brief.raw_input or "")[:200]

        t0 = time.perf_counter()
        try:
            if rag_store is not None:
                import numpy as np
                emb = rag_store["model"].encode(
                    [query], batch_size=1,
                    convert_to_numpy=True, normalize_embeddings=True
                ).astype("float32")
                scores, idxs = rag_store["index"].search(emb, 5)
                hits = [rag_store["metadata"][i] for i in idxs[0] if 0 <= i < len(rag_store["metadata"])]
                n_items = len(hits)
            else:
                hits = _rag_fallback_search(brief, master_df, top_k=5)
                n_items = len(hits)
        except Exception:
            n_items = 0
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results.append({
            "name": row["name"],
            "paper_cat": row["paper_cat"],
            "kappa": k,
            "valid": False,     # no structured plan produced
            "n_tasks": 0,       # no agent tasks
            "n_items": n_items,
            "used_llm": False,
            "elapsed_ms": elapsed_ms,
        })

    mean_lat = statistics.mean(r["elapsed_ms"] for r in results)
    mean_items = statistics.mean(r["n_items"] for r in results) if results else 0
    std_items  = statistics.stdev(r["n_items"] for r in results) if len(results) > 1 else 0.0

    print(f"  n briefs:          {len(results)}")
    print(f"  Plan validity:     n/a (no structured plan)")
    print(f"  LLM invocations:   0%")
    print(f"  Mean items/query:  {mean_items:.1f} +/- {std_items:.1f}")
    print(f"  Mean latency (ms): {mean_lat:.2f}")

    return {
        "variant": "standard_rag",
        "n": len(results),
        "plan_validity_pct": None,
        "mean_tasks": 0.0,
        "std_tasks": 0.0,
        "llm_pct": 0.0,
        "mean_latency_ms": mean_lat,
        "mean_items": mean_items,
        "std_items": std_items,
        "mean_kappa": statistics.mean(r["kappa"] for r in results),
        "per_category": _per_cat(results, df),
        "raw": results,
    }


# ── Planner-based variants ────────────────────────────────────────────────────

def _run_planner_variant(
    df: pd.DataFrame,
    label: str,
    force_catalog: bool = False,
    force_llm: bool = False,
) -> dict:
    """
    Generic runner for catalog-only, LLM-only, and hybrid planner variants.
    """
    print(f"\n{'=' * 60}")
    print(f"VARIANT: {label.upper()}")
    print("=" * 60)

    planner = ResearchPlanner(available_agents=AVAILABLE_AGENTS)

    if force_catalog:
        # Monkey-patch: always use catalog, never LLM
        planner._should_use_llm = lambda _brief: False
    elif force_llm:
        # Monkey-patch: always attempt LLM; fall back to catalog if unavailable
        planner._should_use_llm = lambda _brief: True

    results = []
    for _, row in df.iterrows():
        brief = row_to_brief(row)
        k = kappa(brief)

        # Determine if LLM path will be invoked (for reporting)
        if force_catalog:
            will_use_llm = False
        elif force_llm:
            will_use_llm = planner.llm_client is not None
        else:
            will_use_llm = (k < KAPPA_THRESHOLD) and (planner.llm_client is not None)

        t0 = time.perf_counter()
        try:
            tasks = planner.create_plan(brief)
        except Exception:
            tasks = []
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results.append({
            "name": row["name"],
            "paper_cat": row["paper_cat"],
            "kappa": k,
            "valid": is_valid_plan(tasks),
            "n_tasks": len(tasks),
            "n_items": len(tasks),
            "used_llm": will_use_llm,
            "elapsed_ms": elapsed_ms,
        })

    total = len(results)
    valid_pct  = sum(r["valid"] for r in results) / total * 100 if total else 0.0
    llm_pct    = sum(r["used_llm"] for r in results) / total * 100 if total else 0.0
    tasks_list = [r["n_tasks"] for r in results]
    mean_tasks = statistics.mean(tasks_list) if tasks_list else 0.0
    std_tasks  = statistics.stdev(tasks_list) if len(tasks_list) > 1 else 0.0
    mean_lat   = statistics.mean(r["elapsed_ms"] for r in results)
    mean_kappa = statistics.mean(r["kappa"] for r in results)

    print(f"  n briefs:          {total}")
    print(f"  Plan validity:     {valid_pct:.1f}%")
    print(f"  Mean tasks/plan:   {mean_tasks:.2f} +/- {std_tasks:.2f}")
    print(f"  LLM invocations:   {llm_pct:.1f}%")
    print(f"  Mean kappa:        {mean_kappa:.3f}")
    print(f"  Mean latency (ms): {mean_lat:.2f}")

    return {
        "variant": label,
        "n": total,
        "plan_validity_pct": valid_pct,
        "mean_tasks": mean_tasks,
        "std_tasks": std_tasks,
        "llm_pct": llm_pct,
        "mean_latency_ms": mean_lat,
        "mean_items": mean_tasks,
        "std_items": std_tasks,
        "mean_kappa": mean_kappa,
        "per_category": _per_cat(results, df),
        "raw": results,
    }


# ── utilities ─────────────────────────────────────────────────────────────────

def _per_cat(results: list, df: pd.DataFrame) -> dict:
    """Per-category validity and latency breakdown."""
    cats = df["paper_cat"].unique()
    out: dict = {}
    for cat in cats:
        cat_r = [r for r in results if r.get("paper_cat") == cat]
        if not cat_r:
            continue
        valid_pct = sum(r["valid"] for r in cat_r) / len(cat_r) * 100
        mean_lat  = statistics.mean(r["elapsed_ms"] for r in cat_r)
        mean_k    = statistics.mean(r["kappa"] for r in cat_r)
        out[cat]  = {
            "n": len(cat_r),
            "valid_pct": round(valid_pct, 1),
            "mean_latency_ms": round(mean_lat, 2),
            "mean_kappa": round(mean_k, 3),
        }
    return out


def _print_summary_table(variants: list[dict]) -> None:
    """Print a side-by-side comparison table to stdout."""
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY TABLE")
    print("=" * 70)
    hdr = f"{'Variant':<20} {'Valid%':>7} {'Tasks':>7} {'LLM%':>6} {'Lat(ms)':>9}"
    print(hdr)
    print("-" * 70)
    for v in variants:
        valid_str = f"{v['plan_validity_pct']:.1f}%" if v["plan_validity_pct"] is not None else "  n/a "
        lat_str   = f"{v['mean_latency_ms']:.2f}"
        tasks_str = f"{v['mean_tasks']:.1f}" if v["mean_tasks"] else "  n/a"
        print(
            f"{v['variant']:<20} {valid_str:>7} {tasks_str:>7} "
            f"{v['llm_pct']:>5.1f}% {lat_str:>9}"
        )
    print("=" * 70)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HARIS ablation study: catalog-only / LLM-only / hybrid vs. Standard RAG"
    )
    parser.add_argument("--n", type=int, default=None,
                        help="Subsample n companies (default: all 145)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subsampling")
    parser.add_argument("--no-rag", action="store_true",
                        help="Skip FAISS vector store loading (use keyword fallback)")
    args = parser.parse_args()

    print("HARIS ABLATION STUDY")
    print(f"  kappa threshold : {KAPPA_THRESHOLD}")
    print(f"  catalog size K  : {len(CATALOG_K)} terms")
    print(f"  available agents: {sorted(AVAILABLE_AGENTS)}")

    # Load benchmark
    df = load_benchmark(n=args.n, seed=args.seed)
    print(f"\nBenchmark: {len(df)} companies | "
          f"categories: {df['paper_cat'].value_counts().to_dict()}")

    # Load vector store for RAG baseline (optional)
    rag_store = None if args.no_rag else _load_rag_store()
    if rag_store is None:
        print("  [info] FAISS vector store not found -- using keyword-overlap RAG fallback")
    else:
        print("  [info] FAISS vector store loaded successfully")

    # Load master parquet for RAG fallback
    master_df: Optional[pd.DataFrame] = None
    master_path = Path("data/processed/master_companies_enriched.parquet")
    if master_path.exists():
        try:
            master_df = pd.read_parquet(master_path)
        except Exception:
            master_df = None

    # ── Run all four variants ────────────────────────────────────────────────
    v_rag     = run_standard_rag(df, rag_store, master_df)
    v_catalog = _run_planner_variant(df, "catalog_only",   force_catalog=True)
    v_llm     = _run_planner_variant(df, "llm_only",       force_llm=True)
    v_hybrid  = _run_planner_variant(df, "haris_hybrid",
                                     force_catalog=False, force_llm=False)

    all_variants = [v_rag, v_catalog, v_llm, v_hybrid]

    # ── Print summary table ──────────────────────────────────────────────────
    _print_summary_table(all_variants)

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = Path("data/processed/qa/ablation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip raw per-row data for smaller output file (keep summary stats only)
    save_data = []
    for v in all_variants:
        summary = {k: val for k, val in v.items() if k != "raw"}
        save_data.append(summary)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "kappa_threshold": KAPPA_THRESHOLD,
                    "catalog_size": len(CATALOG_K),
                    "n_companies": len(df),
                    "seed": args.seed,
                    "available_agents": sorted(AVAILABLE_AGENTS),
                },
                "variants": save_data,
            },
            f, indent=2,
        )
    print(f"\nResults saved to: {out_path}")

    # ── Write LaTeX snippet for paper ─────────────────────────────────────────
    tex_path = Path("data/processed/qa/ablation_table_snippet.tex")
    _write_latex_snippet(all_variants, tex_path)
    print(f"LaTeX table snippet saved to: {tex_path}")


def _write_latex_snippet(variants: list[dict], path: Path) -> None:
    """
    Emit a ready-to-paste LaTeX tabularx block matching tab:ablation in haris_eswa.tex.
    Values are filled from measured data.
    """
    lines = [
        r"% === Auto-generated by scripts/run_ablation.py ===",
        r"% Paste into \subsection{Ablation Study} to replace placeholder values.",
        r"",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Ablation study: contribution of each HARIS component "
        r"(n=" + str(sum(v["n"] for v in variants[:1])) + r" Crunchbase descriptions).}",
        r"\label{tab:ablation}",
        r"\begin{tabularx}{\columnwidth}{lXccc}",
        r"\toprule",
        r"\textbf{Variant} & \textbf{Description} & "
        r"\textbf{Plan validity} & \textbf{Latency} & \textbf{Tasks/plan} \\",
        r"\midrule",
    ]
    for v in variants:
        name = {
            "standard_rag": r"Standard RAG$\ddag$",
            "catalog_only":  r"Catalog-only",
            "llm_only":      r"LLM-only$\dag$",
            "haris_hybrid":  r"\textbf{HARIS Hybrid}",
        }.get(v["variant"], v["variant"])

        desc = {
            "standard_rag": r"Single LLM + retrieval, no agents",
            "catalog_only":  r"Always \textsc{CatalogPlan}",
            "llm_only":      r"Always \textsc{LlmPlan}",
            "haris_hybrid":  r"$\kappa$-routed hybrid (Eq.~\ref{eq:routing})",
        }.get(v["variant"], "")

        if v["plan_validity_pct"] is None:
            valid_str = r"n/a"
            lat_str   = f"{v['mean_latency_ms']:.0f}\\,ms"
            tasks_str = "n/a"
        else:
            valid_str  = f"{v['plan_validity_pct']:.0f}\\%"
            lat_ms     = v["mean_latency_ms"]
            # Format: sub-ms as X.Xms, otherwise X ms
            if lat_ms < 1.0:
                lat_str = f"{lat_ms:.2f}\\,ms"
            elif lat_ms < 100.0:
                lat_str = f"{lat_ms:.1f}\\,ms"
            else:
                lat_str = f"{lat_ms/1000:.1f}\\,s"
            tasks_str  = f"{v['mean_tasks']:.1f} $\\pm$ {v['std_tasks']:.1f}"

        # Bold the HARIS row
        if v["variant"] == "haris_hybrid":
            row = (
                f"{name} & {desc} & "
                r"\textbf{" + valid_str + r"} & "
                r"\textbf{" + lat_str  + r"} & "
                r"\textbf{" + tasks_str + r"} \\"
            )
        else:
            row = f"{name} & {desc} & {valid_str} & {lat_str} & {tasks_str} \\\\"
        lines.append(row)

    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\vspace{2pt}",
        r"{\footnotesize",
        r"$\dag$ LLM-only falls back to CatalogPlan when Ollama is unavailable.\\",
        r"$\ddag$ Standard RAG retrieves top-5 company QA snippets; no structured plan is produced.",
        r"}",
        r"\end{table}",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
