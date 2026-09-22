"""Clean Simplify company dataset and store processed output."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.utils.cleaning import (
    clean_text,
    company_size_to_estimate,
    normalize_name,
)


RAW_PATH = Path("data/raw/simplify-company-data.csv")
PROCESSED_DIR = Path("data/processed")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in df.columns:
        df[column] = df[column].apply(clean_text)

    df["normalized_name"] = df["name"].apply(normalize_name)
    df["slug"] = df["url_safe_slug"].apply(lambda x: clean_text(x.lower()) if isinstance(x, str) else None)
    df["year_founded"] = df["year_founded"].apply(lambda x: int(float(x)) if x not in (None, "", "nan") else None)
    df["company_size_bucket"] = df["company_size"].apply(lambda x: int(float(x)) if x not in (None, "", "nan") else None)
    df["employee_count_estimate"] = df["company_size_bucket"].apply(company_size_to_estimate)

    df["funding_total_usd"] = df["funding_total"].apply(
        lambda x: float(x) if x not in (None, "", "nan") else None
    )

    df["funding_stage_clean"] = df["funding_stage"].apply(lambda x: clean_text(x.replace("_", " ")) if x else None)
    df["customer_type_clean"] = df["customer_type"].apply(lambda x: clean_text(x.upper()) if x else None)

    df["benefits_list"] = df["benefits"].apply(parse_list_literal)
    df["values_list"] = df["values"].apply(parse_list_literal)

    df["crunchbase_permalink"] = df["crunchbase"].apply(extract_crunchbase_permalink)

    selected_columns = [
        "name",
        "normalized_name",
        "slug",
        "description",
        "short_description",
        "url",
        "twitter",
        "crunchbase",
        "crunchbase_permalink",
        "linkedin",
        "year_founded",
        "employee_count_estimate",
        "company_size_bucket",
        "funding_stage_clean",
        "customer_type_clean",
        "funding_total_usd",
        "benefits_list",
        "values_list",
        "verified",
    ]

    cleaned = df[selected_columns].dropna(subset=["normalized_name"])
    cleaned = cleaned.drop_duplicates("normalized_name")
    cleaned.sort_values("name", inplace=True)
    cleaned.reset_index(drop=True, inplace=True)
    return cleaned


def parse_list_literal(value: str | None) -> list[str] | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        parts = [item.strip().strip("'\"") for item in inner.split(",")]
        return [item for item in parts if item]
    return [value]


def extract_crunchbase_permalink(url: str | None) -> str | None:
    if not url:
        return None
    url = url.strip()
    if "crunchbase.com" not in url:
        return None
    parts = url.split("crunchbase.com")[-1]
    return parts if parts.startswith("/") else f"/{parts.lstrip('/') }"


def write_output(df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PROCESSED_DIR / "simplify_clean.parquet"
    csv_path = PROCESSED_DIR / "simplify_clean.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing Simplify dataset at {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    cleaned = clean_dataframe(df)
    write_output(cleaned)
    print(f"Saved {len(cleaned)} Simplify companies to data/processed/simplify_clean.csv and .parquet")


if __name__ == "__main__":
    main()


