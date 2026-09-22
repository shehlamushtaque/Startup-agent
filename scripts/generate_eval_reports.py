"""
generate_eval_reports.py
------------------------
Generates 10 paired reports (HARIS vs Template) for blind rater evaluation.

Each brief is run through:
  - HARIS path: FundingIntelligenceAgent + CompetitorGrowthAgent (local) -> evidence
                -> LLM synthesis if available, else rich template
  - Template path: same evidence -> build_report() deterministic template

Output:
  data/eval/reports/  -- 20 text files (R01_HARIS.txt, R01_TEMPLATE.txt, ...)
  data/eval/report_manifest.csv -- maps report_id to brief details (for researcher use)
  data/eval/blind_manifest.csv  -- shuffled, system label hidden (for raters)

Usage:
  python scripts/generate_eval_reports.py
"""

from __future__ import annotations

import os
import sys
import csv
import random
import re
import textwrap
from datetime import datetime
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

from backend.models import IdeaBrief
from backend.agents.tools import CompanyFundingTool, CompetitorGrowthTool
from backend.agents.base import AgentResponse, Evidence
from backend.synthesis.report import build_report

# ── paths ──────────────────────────────────────────────────────────────────
BENCH_PATH = "data/processed/benchmark_companies.csv"
GROWJO_PATH = "data/processed/growjo_clean.csv"
OUT_DIR = "data/eval/reports"
MANIFEST_PATH = "data/eval/report_manifest.csv"
BLIND_PATH = "data/eval/blind_manifest.csv"

N_REPORTS = 10
RANDOM_SEED = 42


# ── helpers ─────────────────────────────────────────────────────────────────
def row_to_brief(row: pd.Series) -> IdeaBrief:
    desc = str(row.get("short_description", "") or str(row.get("name", "")))
    return IdeaBrief(
        idea_id=str(uuid4()),
        raw_input=desc,
        problem=desc,
        audience=str(row.get("paper_cat", "")),
        region=str(row.get("country_code", "US") or "US"),
    )


def run_local_agents(brief: IdeaBrief, growjo: pd.DataFrame):
    """Run FundingIntelligenceAgent and CompetitorGrowthAgent, return responses."""
    responses = []

    # --- FundingIntelligenceAgent ---
    funding_tool = CompanyFundingTool()
    try:
        result = funding_tool.run({
            "industry": brief.problem,
            "country": brief.region,
            "keywords": [],
            "qa_top_k": 3,
        })
        items = result.get("results", [])
        evidence = []
        for rec in items[:5]:
            name = rec.get("name_cb") or rec.get("normalized_name") or "Unknown"
            funding = rec.get("funding_total_usd")
            rounds = rec.get("funding_round_count")
            latest = rec.get("latest_investment_at", "")
            growth = rec.get("growjo_growth_percent")

            parts = [f"Funding: ${funding:,.0f}" if funding else "Funding: N/A"]
            if rounds:
                parts.append(f"{int(rounds)} rounds")
            if latest:
                parts.append(f"latest: {latest}")
            if growth and not (isinstance(growth, float) and growth != growth):
                parts.append(f"growth: {growth:.0f}%")

            evidence.append(Evidence(
                id=str(uuid4()),
                title=name,
                content=" | ".join(parts),
                source="CompanyFundingTool",
                metadata=rec,
                score=0.8,
            ))
        responses.append(AgentResponse(
            agent_name="FundingIntelligenceAgent",
            task_id=str(uuid4()),
            summary=f"Found {len(evidence)} comparable funding records.",
            evidence=evidence,
            confidence="high" if evidence else "low",
        ))
    except Exception as e:
        responses.append(AgentResponse(
            agent_name="FundingIntelligenceAgent",
            task_id=str(uuid4()),
            summary=f"No funding data retrieved: {e}",
            confidence="low",
        ))

    # --- CompetitorGrowthAgent (direct growjo query) ---
    try:
        text = (brief.problem or "").lower()
        tokens = [t for t in re.split(r"[^a-z0-9]+", text) if len(t) > 3]
        industry_col = next(
            (c for c in growjo.columns if "industry" in c.lower()), None
        )
        growth_col = next(
            (c for c in growjo.columns if "growth" in c.lower() and "percent" in c.lower()), None
        )
        revenue_col = next(
            (c for c in growjo.columns if "revenue" in c.lower()), None
        )
        rank_col = next((c for c in growjo.columns if "rank" in c.lower()), None)
        name_col = next((c for c in growjo.columns if c.lower() in ["name", "company_name", "growjo_name"]), None)

        if industry_col and tokens:
            mask = growjo[industry_col].fillna("").str.lower().apply(
                lambda x: any(tok in x for tok in tokens)
            )
            filtered = growjo[mask]
        else:
            filtered = growjo.head(0)

        if growth_col and revenue_col:
            has_metric = filtered[growth_col].notna() | filtered[revenue_col].notna()
            filtered = filtered[has_metric]

        if filtered.empty and rank_col:
            filtered = growjo.nsmallest(10, rank_col)

        filtered = filtered.head(10)
        evidence = []
        for _, rec in filtered.iterrows():
            name = str(rec.get(name_col, "Unknown")) if name_col else "Company"
            growth = rec.get(growth_col) if growth_col else None
            revenue = rec.get(revenue_col) if revenue_col else None
            parts = []
            if growth and not (isinstance(growth, float) and growth != growth):
                parts.append(f"YoY growth: {growth:.0f}%")
            if revenue and not (isinstance(revenue, float) and revenue != revenue):
                rev = revenue / 1e6 if revenue > 1e5 else revenue
                parts.append(f"revenue: ${rev:.1f}M" if revenue > 1e5 else f"revenue: ${revenue:,.0f}")
            content = " | ".join(parts) if parts else "Growth data available"
            evidence.append(Evidence(
                id=str(uuid4()),
                title=f"Growth snapshot: {name}",
                content=content,
                source="CompetitorGrowthTool",
                metadata=dict(rec),
                score=0.75,
            ))
        responses.append(AgentResponse(
            agent_name="CompetitorGrowthAgent",
            task_id=str(uuid4()),
            summary=f"Found {len(evidence)} competitor growth profiles.",
            evidence=evidence,
            confidence="high" if evidence else "low",
        ))
    except Exception as e:
        responses.append(AgentResponse(
            agent_name="CompetitorGrowthAgent",
            task_id=str(uuid4()),
            summary=f"No competitor data retrieved: {e}",
            confidence="low",
        ))

    return responses


def build_haris_report(brief: IdeaBrief, responses: list[AgentResponse]) -> str:
    """
    HARIS report: structured synthesis with evidence integration.
    Uses LLM if available; falls back to enriched template.
    """
    try:
        from backend.llm import get_llm_client
        client = get_llm_client()
    except Exception:
        client = None

    if client:
        # Prepare evidence JSON for LLM
        evidence_lines = []
        for resp in responses:
            for ev in resp.evidence[:5]:
                evidence_lines.append(f"[{ev.source}] {ev.title}: {ev.content}")
        evidence_text = "\n".join(evidence_lines) if evidence_lines else "No evidence retrieved."

        system_prompt = (
            "You are a senior business intelligence analyst. "
            "Given structured evidence from multiple sources, synthesize a comprehensive "
            "business intelligence report with four sections: "
            "Executive Summary, Key Findings, Strategic Implications, Recommendations. "
            "Cite evidence inline. Be specific with numbers."
        )
        user_prompt = (
            f"Business Idea: {brief.raw_input}\n"
            f"Region: {brief.region}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            "Generate a 4-section business intelligence report."
        )
        try:
            resp = client.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=1200,
            )
            content = resp.get("content", "").strip()
            if content:
                return content
        except Exception:
            pass

    # Enriched template fallback (richer than build_report)
    from backend.synthesis.summarizer import summarize_competitors, summarize_funding
    funding = summarize_funding(responses)
    competitors = summarize_competitors(responses)

    sections = []

    # Executive Summary
    comp_count = len(competitors["competitors"])
    fund_count = len(funding["highlights"])
    exec_sum = (
        f"Executive Summary\n"
        f"{'='*40}\n"
        f"This analysis covers the business opportunity: \"{brief.raw_input}\" "
        f"targeting the {brief.region} market. "
        f"Research identified {fund_count} comparable funding benchmarks and "
        f"{comp_count} high-growth competitors in the space."
    )
    sections.append(exec_sum)

    # Key Findings
    kf_lines = ["Key Findings\n" + "="*40]
    if funding["highlights"]:
        kf_lines.append("Funding Comparables:")
        for h in funding["highlights"][:4]:
            kf_lines.append(f"  - {h}")
    if competitors["growth_notes"]:
        kf_lines.append("Competitor Growth Signals:")
        for n in competitors["growth_notes"][:4]:
            kf_lines.append(f"  - {n}")
    if funding["top_investors"]:
        kf_lines.append(f"Active investors: {', '.join(funding['top_investors'][:4])}")
    sections.append("\n".join(kf_lines))

    # Strategic Implications
    si_lines = ["Strategic Implications\n" + "="*40]
    if competitors["competitors"]:
        si_lines.append(
            f"The space has {comp_count} identified high-growth players including "
            f"{', '.join(competitors['competitors'][:3])}. "
            "This indicates validated market demand but meaningful competitive intensity."
        )
    if funding["highlights"]:
        si_lines.append(
            "Funding activity in comparable companies suggests institutional capital "
            "is available for well-differentiated entrants."
        )
    sections.append("\n".join(si_lines))

    # Recommendations
    rec_lines = ["Recommendations\n" + "="*40]
    rec_lines.append("1. Benchmark pricing and go-to-market approach against identified comparables.")
    rec_lines.append("2. Prioritize differentiation from the " + str(comp_count) + " identified growth-stage competitors.")
    if funding["top_investors"]:
        rec_lines.append(f"3. Engage active investors: {', '.join(funding['top_investors'][:2])}.")
    rec_lines.append("4. Validate unit economics before Series A fundraise given comparable median funding levels.")
    sections.append("\n".join(rec_lines))

    return "\n\n".join(sections)


def build_template_report(responses: list[AgentResponse]) -> str:
    """
    Template baseline: deterministic build_report() from synthesis/report.py.
    This is the comparison condition for raters.
    """
    return build_report(responses)


def wrap_report(
    report_id: str,
    system_label: str,
    brief: IdeaBrief,
    body: str,
    blind: bool = False,
) -> str:
    """Format report as a clean text file for raters."""
    header_lines = [
        f"REPORT ID: {report_id}",
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
    ]
    if not blind:
        header_lines.append(f"System: {system_label}")
    header_lines += [
        f"Brief: {brief.raw_input[:120]}",
        f"Region: {brief.region}",
        "=" * 60,
        "",
    ]
    header = "\n".join(header_lines)
    wrapped_body = "\n".join(
        textwrap.fill(line, width=80) if line.strip() else line
        for line in body.split("\n")
    )
    return header + wrapped_body + "\n"


# ── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    random.seed(RANDOM_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    bench = pd.read_csv(BENCH_PATH)
    growjo = pd.read_csv(GROWJO_PATH)

    # Stratified sample: 1 per category from the 10 categories
    cats = sorted(bench["paper_cat"].unique())
    selected_rows = []
    for cat in cats:
        group = bench[bench["paper_cat"] == cat]
        row = group.sample(1, random_state=RANDOM_SEED).iloc[0]
        selected_rows.append(row)
    # If fewer than 10 categories, top up randomly
    while len(selected_rows) < N_REPORTS:
        selected_rows.append(bench.sample(1, random_state=RANDOM_SEED + len(selected_rows)).iloc[0])
    selected_rows = selected_rows[:N_REPORTS]

    manifest_rows = []
    blind_rows = []

    print(f"Generating {N_REPORTS} report pairs...")
    for i, row in enumerate(selected_rows, start=1):
        report_id = f"R{i:02d}"
        brief = row_to_brief(row)
        category = str(row.get("paper_cat", "unknown"))
        company_name = str(row.get("name", "Unknown"))

        print(f"  {report_id} [{category}] {company_name[:50]}...")

        # Run local agents once (shared evidence for both conditions)
        responses = run_local_agents(brief, growjo)

        # Generate both reports
        haris_body = build_haris_report(brief, responses)
        template_body = build_template_report(responses)
        if not template_body or template_body == "No synthesised insights available.":
            template_body = (
                f"Business area: {brief.raw_input}\n\n"
                "No structured data available for this input via the template system.\n"
                "Manual research recommended."
            )

        # Save full (non-blind) versions
        haris_file = os.path.join(OUT_DIR, f"{report_id}_HARIS.txt")
        template_file = os.path.join(OUT_DIR, f"{report_id}_TEMPLATE.txt")
        with open(haris_file, "w", encoding="utf-8") as f:
            f.write(wrap_report(report_id, "HARIS", brief, haris_body, blind=False))
        with open(template_file, "w", encoding="utf-8") as f:
            f.write(wrap_report(report_id, "TEMPLATE", brief, template_body, blind=False))

        # Save blind versions (A/B randomly assigned, label hidden)
        ab = random.choice(["A", "B"])
        if ab == "A":
            a_system, b_system = "HARIS", "TEMPLATE"
            a_body, b_body = haris_body, template_body
        else:
            a_system, b_system = "TEMPLATE", "HARIS"
            a_body, b_body = template_body, haris_body

        for version, sys_label, body in [("A", a_system, a_body), ("B", b_system, b_body)]:
            blind_file = os.path.join(OUT_DIR, f"{report_id}_{version}_blind.txt")
            with open(blind_file, "w", encoding="utf-8") as f:
                f.write(wrap_report(f"{report_id}-{version}", sys_label, brief, body, blind=True))
            blind_rows.append({
                "blind_id": f"{report_id}-{version}",
                "report_id": report_id,
                "version": version,
                "category": category,
                "filename": f"{report_id}_{version}_blind.txt",
            })

        manifest_rows.append({
            "report_id": report_id,
            "company": company_name,
            "category": category,
            "brief": brief.raw_input[:100],
            "region": brief.region,
            "haris_file": f"{report_id}_HARIS.txt",
            "template_file": f"{report_id}_TEMPLATE.txt",
            "haris_is_A": (a_system == "HARIS"),
        })

    # Write manifests
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    with open(BLIND_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(blind_rows[0].keys()))
        writer.writeheader()
        writer.writerows(blind_rows)

    print(f"\nDone. Files written to: {OUT_DIR}/")
    print(f"  Non-blind manifest: {MANIFEST_PATH}")
    print(f"  Blind manifest (give to raters): {BLIND_PATH}")
    print(f"\nGive raters the *_blind.txt files + rater_form_template.csv")
    print(f"Reveal A/B assignment ONLY after all ratings are collected.")


if __name__ == "__main__":
    main()
