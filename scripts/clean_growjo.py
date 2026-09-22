"""Clean Growjo growth dataset and output processed growth metrics."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.utils.cleaning import (
    clean_text,
    normalize_name,
    parse_currency,
    parse_percentage,
)


RAW_PATH = Path("data/raw/Growjo-1k-list.csv")
PROCESSED_DIR = Path("data/processed")


def clean_growjo() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing Growjo dataset at {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    for column in [
        "company_name",
        "url",
        "city",
        "state",
        "country",
        "linkedin_url",
        "Industry",
        "keywords",
        "LeadInvestors",
        "Accelerator",
        "btype",
        "product_url",
        "indeed_url",
    ]:
        if column in df.columns:
            df[column] = df[column].apply(clean_text)

    df["normalized_name"] = df["company_name"].apply(normalize_name)
    df["slug"] = df["product_url"].apply(
        lambda url: clean_text(url.split("/")[-1]) if isinstance(url, str) else None
    )

    df["employees"] = pd.to_numeric(df["employees"], errors="coerce").astype("Int64")
    df["GrowjoRanking"] = pd.to_numeric(df["GrowjoRanking"], errors="coerce").astype("Int64")
    df["Previous Ranking"] = pd.to_numeric(df["Previous Ranking"], errors="coerce").astype("Int64")
    df["job_openings"] = pd.to_numeric(df["job_openings"], errors="coerce").astype("Int64")
    df["founded"] = pd.to_numeric(df["founded"], errors="coerce").astype("Int64")

    df["valuation_usd"] = df["valuation"].apply(parse_currency)
    df["total_funding_usd"] = df["total_funding"].apply(parse_currency)
    df["estimated_revenues_usd"] = pd.to_numeric(df["estimated_revenues"], errors="coerce")
    df["growth_percentage_value"] = df["growth_percentage"].apply(parse_percentage)

    selected = df[
        [
            "company_name",
            "normalized_name",
            "slug",
            "city",
            "state",
            "country",
            "Industry",
            "employees",
            "GrowjoRanking",
            "Previous Ranking",
            "estimated_revenues_usd",
            "job_openings",
            "LeadInvestors",
            "Accelerator",
            "btype",
            "valuation_usd",
            "total_funding_usd",
            "product_url",
            "indeed_url",
            "growth_percentage_value",
        ]
    ].copy()

    selected = selected.dropna(subset=["normalized_name"]).drop_duplicates("normalized_name").reset_index(drop=True)
    selected = selected.rename(
        columns={
            "Industry": "growjo_industry",
            "employees": "growjo_employees",
            "GrowjoRanking": "growjo_rank",
            "Previous Ranking": "growjo_previous_rank",
            "job_openings": "growjo_job_openings",
            "LeadInvestors": "growjo_lead_investors",
            "Accelerator": "growjo_accelerator",
            "btype": "growjo_business_type",
            "valuation_usd": "growjo_valuation_usd",
            "total_funding_usd": "growjo_total_funding_usd",
            "product_url": "growjo_product_url",
            "indeed_url": "growjo_indeed_url",
            "growth_percentage_value": "growjo_growth_percent",
            "estimated_revenues_usd": "growjo_estimated_revenue_usd",
        }
    )
    return selected


def write_output(df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PROCESSED_DIR / "growjo_clean.parquet"
    csv_path = PROCESSED_DIR / "growjo_clean.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)


def main() -> None:
    df = clean_growjo()
    write_output(df)
    print(f"Saved {len(df)} Growjo records to data/processed/growjo_clean.csv and .parquet")


if __name__ == "__main__":
    main()


