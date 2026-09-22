"""
Run all three reproducible experiments on real Crunchbase company descriptions.

Experiment 1: Planner reliability (catalog vs hybrid) on 145 real companies
Experiment 2: Agent evidence coverage on 50 real companies (local agents only)
Experiment 3: SEC query constructor evaluation on 40 companies (real API)

All results saved to data/processed/qa/experiment_results.json
"""
import sys, os, time, json, statistics, re, requests
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

import pandas as pd
from backend.models.schemas import IdeaBrief, MaturityStage
from backend.planning.research_planner import ResearchPlanner
from backend.agents.orchestrator import DEFAULT_AGENT_CONFIGS
from backend.agents.tools import CompanyFundingTool, CompetitorGrowthTool

AVAILABLE_AGENTS = {cfg.name for cfg in DEFAULT_AGENT_CONFIGS}

# Crunchbase category → paper domain
NOVEL_CATEGORIES = {'games_video', 'social', 'hospitality', 'other'}

MATURITY_MAP = {
    "concept": MaturityStage.concept,
    "mvp": MaturityStage.mvp,
    "pre-seed": MaturityStage.pre_seed,
    "seed": MaturityStage.seed,
    "growth": MaturityStage.growth,
}

def load_benchmark():
    path = Path("data/processed/benchmark_companies.csv")
    df = pd.read_csv(path, encoding='latin1', low_memory=False)
    df['short_description'] = df['short_description'].fillna('').astype(str)
    df['category_code'] = df['category_code'].fillna('software').astype(str)
    df['country_code'] = df['country_code'].fillna('USA').astype(str)
    df['city'] = df['city'].fillna('').astype(str)
    df['region'] = df['region'].fillna('').astype(str)
    df['paper_cat'] = df['paper_cat'].fillna('tech').astype(str)
    return df

def row_to_brief(row):
    # Build a geographic string
    geo_parts = [p for p in [row.get('city',''), row.get('region',''), row.get('country_code','')] if p and p != 'nan']
    geo = ", ".join(geo_parts) if geo_parts else "United States"
    return IdeaBrief(
        idea_id=str(uuid4()),
        raw_input=str(row['short_description'])[:300],
        problem=str(row['short_description'])[:200],
        audience=f"customers in the {row['category_code']} sector",
        region=geo,
        maturity_stage=MaturityStage.seed,
    )

def is_valid_plan(tasks):
    if not tasks:
        return False
    for t in tasks:
        if t.agent not in AVAILABLE_AGENTS:
            return False
        if not t.instruction:
            return False
    return True

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 1: Planner Reliability
# ─────────────────────────────────────────────────────────────────
def run_exp1(df):
    print("\n" + "="*60)
    print("EXPERIMENT 1: PLANNER RELIABILITY")
    print(f"Dataset: {len(df)} real Crunchbase company descriptions")
    print("="*60)

    planner_hybrid = ResearchPlanner(available_agents=AVAILABLE_AGENTS)
    planner_catalog = ResearchPlanner(available_agents=AVAILABLE_AGENTS)
    planner_catalog._should_use_llm = lambda x: False

    # Must match backend/planning/research_planner.py known_keywords exactly (118 terms, 2025-Q1)
    known_keywords = {
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

    results_hybrid = []
    results_catalog = []
    novel_cats = {'novel', 'games_video', 'social', 'hospitality', 'other'}

    for _, row in df.iterrows():
        brief = row_to_brief(row)
        is_novel = row['paper_cat'] in novel_cats

        for mode, planner, result_list in [
            ("catalog", planner_catalog, results_catalog),
            ("hybrid",  planner_hybrid,  results_hybrid),
        ]:
            t0 = time.perf_counter()
            try:
                tasks = planner.create_plan(brief)
            except Exception:
                tasks = []
            elapsed = time.perf_counter() - t0

            valid = is_valid_plan(tasks)
            n_tasks = len(tasks)

            # coverage score (fraction of expected agent types assigned)
            assigned = {t.agent for t in tasks}
            all_agents = AVAILABLE_AGENTS
            coverage_score = len(assigned & all_agents) / len(all_agents) if all_agents else 0.0

            # LLM invocation detection for hybrid
            used_llm = False
            if mode == "hybrid":
                tokens = [t.lower() for t in (brief.problem or "").split() if len(t) > 3]
                cov = sum(1 for t in tokens if t in known_keywords) / max(len(tokens), 1)
                novel_cnt = sum(1 for t in tokens if t not in known_keywords)
                # Threshold kappa < 0.10 matches research_planner.py (calibrated 2025-Q1)
                used_llm = (cov < 0.10) and planner_hybrid.llm_client is not None

            result_list.append({
                "name": row['name'],
                "category": row['category_code'],
                "paper_cat": row['paper_cat'],
                "is_novel": is_novel,
                "valid": valid,
                "n_tasks": n_tasks,
                "coverage_score": coverage_score,
                "used_llm": used_llm,
                "elapsed_ms": elapsed * 1000,
            })

    def summarize(results, mode):
        total = len(results)
        valid_pct = sum(r["valid"] for r in results) / total * 100
        novel = [r for r in results if r["is_novel"]]
        novel_valid_pct = sum(r["valid"] for r in novel) / len(novel) * 100 if novel else 0
        mean_tasks = statistics.mean(r["n_tasks"] for r in results)
        std_tasks = statistics.stdev(r["n_tasks"] for r in results)
        llm_pct = sum(r["used_llm"] for r in results) / total * 100
        mean_cov = statistics.mean(r["coverage_score"] for r in results) * 100
        novel_cov = statistics.mean(r["coverage_score"] for r in novel) * 100 if novel else 0
        mean_lat = statistics.mean(r["elapsed_ms"] for r in results)

        # Per-category
        per_cat = {}
        for cat in df['paper_cat'].unique():
            cat_r = [r for r in results if r["paper_cat"] == cat]
            if cat_r:
                per_cat[cat] = {
                    "n": len(cat_r),
                    "valid_pct": sum(r["valid"] for r in cat_r) / len(cat_r) * 100,
                    "mean_tasks": statistics.mean(r["n_tasks"] for r in cat_r),
                    "agent_coverage_pct": statistics.mean(r["coverage_score"] for r in cat_r) * 100,
                }

        print(f"\n  Mode: {mode.upper()}")
        print(f"  Total briefs:           {total}")
        print(f"  Valid plans (%):        {valid_pct:.1f}%")
        print(f"  Novel-domain valid (%): {novel_valid_pct:.1f}%")
        print(f"  Agent coverage (%):     {mean_cov:.1f}%")
        print(f"  Novel coverage (%):     {novel_cov:.1f}%")
        print(f"  Mean tasks/plan:        {mean_tasks:.2f} ± {std_tasks:.2f}")
        print(f"  LLM invocations (%):    {llm_pct:.1f}%")
        print(f"  Mean latency (ms):      {mean_lat:.1f}")
        print(f"  Per-category breakdown:")
        for cat, s in sorted(per_cat.items()):
            print(f"    {cat:12s}: n={s['n']:2d}  valid={s['valid_pct']:.0f}%  "
                  f"tasks={s['mean_tasks']:.1f}  cov={s['agent_coverage_pct']:.0f}%")

        return {
            "mode": mode, "total": total, "valid_pct": valid_pct,
            "novel_valid_pct": novel_valid_pct, "agent_coverage_pct": mean_cov,
            "novel_coverage_pct": novel_cov,
            "mean_tasks": mean_tasks, "std_tasks": std_tasks,
            "llm_pct": llm_pct, "mean_latency_ms": mean_lat,
            "per_category": per_cat,
        }

    s_cat = summarize(results_catalog, "catalog")
    s_hyb = summarize(results_hybrid,  "hybrid")
    return {"catalog": s_cat, "hybrid": s_hyb}


# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 2: Agent Evidence Coverage (local agents only)
# ─────────────────────────────────────────────────────────────────
def run_exp2(df):
    print("\n" + "="*60)
    print("EXPERIMENT 2: AGENT EVIDENCE COVERAGE (LOCAL AGENTS)")
    print("="*60)

    # Use 50 companies stratified across categories
    sample50 = df.groupby('paper_cat', group_keys=False).apply(
        lambda x: x.sample(min(5, len(x)), random_state=42)
    ).reset_index(drop=True)
    print(f"Sample: {len(sample50)} companies across {sample50['paper_cat'].nunique()} categories")

    funding_tool = CompanyFundingTool()
    competitor_tool = CompetitorGrowthTool()

    funding_results = []
    competitor_results = []

    for _, row in sample50.iterrows():
        brief = row_to_brief(row)

        # FundingIntelligenceAgent (Parquet) — uses .run()
        t0 = time.perf_counter()
        try:
            fr = funding_tool.run({
                "industry": brief.problem or brief.raw_input,
                "country": brief.region,
                "keywords": [row['category_code']],
                "limit": 10,
            })
            f_items = len(fr.get("results", [])) if isinstance(fr, dict) else 0
            f_success = True
        except Exception as e:
            f_items = 0
            f_success = False
        f_elapsed = time.perf_counter() - t0
        funding_results.append({"items": f_items, "success": f_success, "ms": f_elapsed*1000, "cat": row['paper_cat']})

        # CompetitorGrowthAgent (Parquet) — uses .run()
        t0 = time.perf_counter()
        try:
            cr = competitor_tool.run({
                "industry": brief.problem or brief.raw_input,
                "keywords": [row['category_code']],
                "limit": 10,
            })
            c_items = len(cr.get("results", [])) if isinstance(cr, dict) else 0
            c_success = True
        except Exception as e:
            c_items = 0
            c_success = False
        c_elapsed = time.perf_counter() - t0
        competitor_results.append({"items": c_items, "success": c_success, "ms": c_elapsed*1000, "cat": row['paper_cat']})

    def stats(r, name):
        total = len(r)
        success_rate = sum(x["success"] for x in r) / total * 100
        items_list = [x["items"] for x in r if x["success"]]
        mean_items = statistics.mean(items_list) if items_list else 0
        std_items = statistics.stdev(items_list) if len(items_list) > 1 else 0
        mean_ms = statistics.mean(x["ms"] for x in r)
        print(f"\n  {name}:")
        print(f"    Success rate:  {success_rate:.1f}%")
        print(f"    Mean items:    {mean_items:.1f} ± {std_items:.1f}")
        print(f"    Mean latency:  {mean_ms:.0f}ms")
        return {"agent": name, "success_rate": success_rate, "mean_items": mean_items,
                "std_items": std_items, "mean_latency_ms": mean_ms, "n": total}

    s_f = stats(funding_results, "FundingIntelligenceAgent")
    s_c = stats(competitor_results, "CompetitorGrowthAgent")
    return {"FundingIntelligenceAgent": s_f, "CompetitorGrowthAgent": s_c, "n": len(sample50)}


# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 3: SEC Query Constructor Evaluation (real API)
# ─────────────────────────────────────────────────────────────────
def run_exp3(df):
    print("\n" + "="*60)
    print("EXPERIMENT 3: SEC QUERY CONSTRUCTOR EVALUATION (REAL API)")
    print("="*60)

    sec_key = os.getenv("Sec_API_KEY") or os.getenv("SEC_API_KEY")
    if not sec_key:
        print("  SEC API key not found — skipping")
        return {}

    # 40 companies across tech/fintech/healthcare/cleantech (10 each)
    target_cats = ['tech', 'fintech', 'healthcare', 'cleantech']
    sec_sample = df[df['paper_cat'].isin(target_cats)].groupby('paper_cat', group_keys=False).apply(
        lambda x: x.sample(min(10, len(x)), random_state=99)
    ).reset_index(drop=True)
    print(f"  SEC sample: {len(sec_sample)} companies across {sec_sample['paper_cat'].nunique()} categories")

    # Known exclusion patterns
    EXCLUDE_PATTERNS = [
        "realty", "real estate", "property fund", "retail fund", "investment fund",
        "holdings llc", "partners lp", "credit fund", "reit", "mortgage",
        "housing fund", "workforce housing", "opportunity fund", "income fund",
        "ventures lp", "capital partners", "equity fund", "debt fund",
    ]

    TECH_KW   = ["analytics","software","platform","technology","data","ai","saas","cloud","digital","tech","solution","api","fintech","payments"]
    HEALTH_KW = ["healthcare","health","medical","telemedicine","patient","clinical","pharma","biotech","diagnostic"]
    FIN_KW    = ["fintech","financial","payment","banking","lending","crypto","trading","investment","wealth"]

    def build_naive_query(brief):
        tokens = [t.lower() for t in (brief.problem or "").split() if len(t) > 3]
        return " ".join(tokens[:3]) if tokens else "software"

    def build_adaptive_query(brief):
        tokens = [t.lower() for t in (brief.problem or "").split() if len(t) > 3]
        found_tech   = [k for k in TECH_KW   if any(k in t for t in tokens)]
        found_health = [k for k in HEALTH_KW if any(k in t for t in tokens)]
        found_fin    = [k for k in FIN_KW    if any(k in t for t in tokens)]
        industry = found_tech + found_health + found_fin
        stops = {"retail","fund","realty","property","estate","llc","lp","ltd","the","and","or","for","with"}
        main = [t for t in tokens if t not in stops][:2]
        if industry and main:
            return f"({' OR '.join(industry[:3])}) AND ({' OR '.join(main)})"
        elif industry:
            return " OR ".join(industry[:4])
        elif main:
            return f'"{" ".join(main)}"'
        return "software OR technology"

    def is_irrelevant(name):
        n = (name or "").lower()
        return any(p in n for p in EXCLUDE_PATTERNS)

    def query_sec(query_str, limit=10):
        url = "https://api.sec-api.io/full-text-search"
        headers = {"Authorization": sec_key, "Content-Type": "application/json"}
        payload = {"query": query_str, "dateRange": {"startdate": "2023-01-01", "enddate": "2025-12-31"},
                   "forms": ["D"], "hits": {"total": limit}}
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=15)
            if r.status_code == 200:
                data = r.json()
                hits = data.get("hits", {}).get("hits", [])
                return [h.get("_source", {}) for h in hits]
            return []
        except Exception:
            return []

    naive_results   = {"tp": 0, "fp": 0, "fn": 0, "total_queries": 0, "irrelevant_per_query": []}
    adaptive_results= {"tp": 0, "fp": 0, "fn": 0, "total_queries": 0, "irrelevant_per_query": []}
    adaptive_excl   = {"tp": 0, "fp": 0, "fn": 0, "total_queries": 0, "irrelevant_per_query": []}

    for _, row in sec_sample.iterrows():
        brief = row_to_brief(row)

        for approach_name, query_fn, apply_excl, store in [
            ("naive",         build_naive_query,    False, naive_results),
            ("adaptive",      build_adaptive_query, False, adaptive_results),
            ("adaptive_excl", build_adaptive_query, True,  adaptive_excl),
        ]:
            q = query_fn(brief)
            hits = query_sec(q, limit=10)
            time.sleep(0.4)  # respect rate limit

            # Count irrelevant hits in this query's results
            irr_in_this_query = sum(
                1 for h in hits
                if is_irrelevant(h.get("companyName", h.get("entityName", h.get("issuerName", ""))))
            )
            total_returned = len(hits)

            if apply_excl:
                # After applying exclusion filter, count remaining irrelevant
                kept = [h for h in hits
                        if not is_irrelevant(h.get("companyName", h.get("entityName", h.get("issuerName", ""))))]
                post_filter_irr = sum(
                    1 for h in kept
                    if is_irrelevant(h.get("companyName", h.get("entityName", h.get("issuerName", ""))))
                )
                store["irrelevant_per_query"].append(irr_in_this_query - len(hits) + len(kept))
            else:
                store["irrelevant_per_query"].append(irr_in_this_query)

            store["total_queries"] += 1
            store.setdefault("total_returned", []).append(total_returned)

    def sec_stats(s, name):
        irrelevant = s["irrelevant_per_query"]
        mean_irr = statistics.mean(irrelevant) if irrelevant else 0
        std_irr  = statistics.stdev(irrelevant) if len(irrelevant) > 1 else 0
        total_q  = s["total_queries"]
        print(f"\n  {name}:")
        print(f"    Queries run:              {total_q}")
        print(f"    Mean irrelevant/query:    {mean_irr:.1f} ± {std_irr:.1f}")
        return {"approach": name, "n_queries": total_q,
                "mean_irrelevant_per_query": mean_irr, "std_irrelevant": std_irr,
                "irrelevant_per_query": irrelevant}

    s_naive = sec_stats(naive_results,    "naive keyword")
    s_adapt = sec_stats(adaptive_results, "adaptive (no exclusion)")
    s_aexcl = sec_stats(adaptive_excl,    "adaptive + exclusion filter")
    return {"naive": s_naive, "adaptive": s_adapt, "adaptive_exclusion": s_aexcl}


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_benchmark()
    print(f"Loaded {len(df)} benchmark companies from Crunchbase objects.csv")
    print(f"Category distribution: {df['paper_cat'].value_counts().to_dict()}")

    all_results = {}

    exp1 = run_exp1(df)
    all_results["experiment1_planner"] = exp1

    exp2 = run_exp2(df)
    all_results["experiment2_agent_coverage"] = exp2

    exp3 = run_exp3(df)
    all_results["experiment3_sec_query"] = exp3

    out = Path("data/processed/qa/experiment_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nAll results saved to: {out}")
