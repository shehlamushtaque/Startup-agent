"""Generate templated Q&A pairs for companies using cleaned datasets."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

PROCESSED_DIR = Path("data/processed")
QA_DIR = PROCESSED_DIR / "qa"
MASTER_PATH = PROCESSED_DIR / "master_companies.parquet"
INVESTOR_EDGES_PATH = PROCESSED_DIR / "company_investors.parquet"


def format_usd(value: Optional[float]) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    units = [
        (1e9, "B"),
        (1e6, "M"),
        (1e3, "K"),
    ]
    for factor, suffix in units:
        if abs(value) >= factor:
            return f"${value / factor:.1f}{suffix}"
    return f"${value:,.0f}"


def format_percent(value: Optional[float]) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return f"{value:.1f}%"


def best_company_name(row: pd.Series) -> str:
    for field in ("name_simplify", "name_cb", "normalized_name"):
        if field in row and isinstance(row[field], str) and row[field].strip():
            return row[field]
    return "Unknown company"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing master table at {MASTER_PATH}")
    master = pd.read_parquet(MASTER_PATH)

    if not INVESTOR_EDGES_PATH.exists():
        raise FileNotFoundError(f"Missing investor edges at {INVESTOR_EDGES_PATH}")
    edges = pd.read_parquet(INVESTOR_EDGES_PATH)

    return master, edges


def build_investor_lookup(edges: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    grouped = {}
    for object_id, group in edges.groupby("company_object_id"):
        names = [
            name for name in group["investor_display_name"].dropna().unique() if isinstance(name, str)
        ]
        grouped[object_id] = {
            "investor_names": names[:10],
            "investor_count": int(group["investor_object_id"].nunique()),
            "funding_round_ids": sorted(set(group["funding_round_id"].dropna().astype(int))),
        }
    return grouped


def build_records(master: pd.DataFrame, edges_lookup: Dict[str, Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for _, row in master.iterrows():
        name = best_company_name(row)
        normalized_name = row.get("normalized_name")
        company_object_id = row.get("object_id")
        records = []

        funding_total = row.get("funding_total_usd")
        funding_rounds = row.get("funding_round_count")
        latest_investment = row.get("latest_investment_at")
        investor_info = edges_lookup.get(company_object_id or "")
        investor_names = investor_info["investor_names"] if investor_info else []

        if funding_total or funding_rounds or latest_investment:
            prompt = f"What is the latest funding profile for {name}?"
            funding_parts = []
            funding_str = format_usd(funding_total)
            if funding_str:
                funding_parts.append(f"total disclosed funding of {funding_str}")
            if isinstance(funding_rounds, (int, float)) and not math.isnan(funding_rounds):
                funding_parts.append(f"{int(funding_rounds)} recorded funding rounds")
            if latest_investment and isinstance(latest_investment, str):
                funding_parts.append(f"latest investment on {latest_investment[:10]}")
            if not funding_parts:
                funding_parts.append("no disclosed funding information")
            response = f"{name} has " + ", ".join(funding_parts) + "."
            metadata = {
                "company": name,
                "normalized_name": normalized_name,
                "object_id": company_object_id,
                "funding_total_usd": funding_total,
                "funding_round_count": funding_rounds,
                "latest_investment_at": latest_investment,
                "source_tables": ["master_companies"],
            }
            records.append({"prompt": prompt, "response": response, "metadata": metadata})

        if investor_names:
            prompt = f"Who are the main investors in {name}?"
            listed = ", ".join(investor_names)
            response = f"Key investors in {name} include {listed}."
            metadata = {
                "company": name,
                "normalized_name": normalized_name,
                "object_id": company_object_id,
                "investor_names": investor_names,
                "source_tables": ["master_companies", "company_investors"],
            }
            records.append({"prompt": prompt, "response": response, "metadata": metadata})

        growth_percent = row.get("growjo_growth_percent")
        employees = row.get("growjo_employees")
        if not (growth_percent is None or (isinstance(growth_percent, float) and math.isnan(growth_percent))):
            growth_str = format_percent(growth_percent)
            employees_str = f" with {int(employees)} employees" if isinstance(employees, (int, float)) and not math.isnan(employees) else ""
            prompt = f"What is the employee growth rate of {name}?"
            response = f"{name} shows a {growth_str} employee growth rate{employees_str} according to Growjo."
            metadata = {
                "company": name,
                "normalized_name": normalized_name,
                "growjo_growth_percent": growth_percent,
                "growjo_employees": employees,
                "source_tables": ["master_companies", "growjo_clean"],
            }
            records.append({"prompt": prompt, "response": response, "metadata": metadata})

        for record in records:
            yield record


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> int:
    QA_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    master, edges = load_data()
    edges_lookup = build_investor_lookup(edges)
    records = list(build_records(master, edges_lookup))

    output_path = QA_DIR / "company_qa.jsonl"
    count = write_jsonl(records, output_path)

    print(f"Wrote {count} Q&A records to {output_path}")


if __name__ == "__main__":
    main()


