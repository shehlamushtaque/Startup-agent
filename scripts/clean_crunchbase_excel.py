"""Clean Crunchbase Excel export into normalized CSV/Parquet outputs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.utils.cleaning import (
    clean_text,
    coerce_int,
    normalize_name,
    parse_currency,
    parse_employee_range,
    parse_money_range,
    strip_leading_numbering,
)


RAW_PATH = Path("data/raw/Crunchbase.xlsx")
PROCESSED_DIR = Path("data/processed")


def split_headquarters(value: str) -> Tuple[tuple[str | None, str | None, str | None], str | None]:
    text = clean_text(value)
    if text is None:
        return (None, None, None), None

    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) >= 3:
        city = ", ".join(parts[:-2]) or None
        state = parts[-2] or None
        country = parts[-1] or None
    elif len(parts) == 2:
        city = parts[0] or None
        state = None
        country = parts[1] or None
    elif parts:
        city = parts[0] or None
        state = None
        country = None
    else:
        city = state = country = None

    return (city, state, country), text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_text(col) or col for col in df.columns]
    df = df.rename(
        columns={
            "Unnamed: 0": "row_number",
            "Headquaters": "headquarters",
            "Nunmber of Lead Investors": "number_of_lead_investors",
            "Nmber of Investors": "number_of_investors",
            "Number of employees": "employee_range",
            "Number of Investment": "number_of_investments",
            "Number of Lead Investment": "number_of_lead_investments",
            "Funding Rounds": "funding_rounds",
            "Funding Status": "funding_status",
            "Funding Amount": "funding_amount",
            "Founders": "founders",
            "Investment Stage": "investment_stage",
            "Exit Date": "exit_date",
            "Estimated Revenue": "estimated_revenue",
            "Acquisition Status": "acquisition_status",
            "Acquired Price": "acquired_price",
            "Acquired By": "acquired_by",
            "IPO status": "ipo_status",
            "Full Name": "related_full_name",
            "Job Title": "related_job_title",
            "Primary Organization": "related_primary_org",
            "Founded Organization": "related_founded_org",
            "Portfolio Companies": "portfolio_companies",
            "Active Products": "active_products",
        }
    )

    for column in df.columns:
        df[column] = df[column].apply(clean_text)

    df["name"] = df["Name"].apply(strip_leading_numbering)
    df["normalized_name"] = df["name"].apply(normalize_name)

    headquarters_series = df.get("headquarters")
    if headquarters_series is not None:
        parsed_locations = [split_headquarters(val) for val in headquarters_series]
        if parsed_locations:
            location_tuples, cleaned_values = zip(*parsed_locations)
            hq_city, hq_state, hq_country = zip(*location_tuples)
            df["headquarters_clean"] = cleaned_values
            df["hq_city"] = hq_city
            df["hq_state"] = hq_state
            df["hq_country"] = hq_country

    df["founded_year"] = df["Founded Year"].apply(coerce_int)
    df["funding_rounds_count"] = df["funding_rounds"].apply(coerce_int)
    df["number_of_investments"] = df["number_of_investments"].apply(coerce_int)
    df["number_of_lead_investments"] = df["number_of_lead_investments"].apply(coerce_int)
    df["number_of_lead_investors"] = df["number_of_lead_investors"].apply(coerce_int)
    df["number_of_investors"] = df["number_of_investors"].apply(coerce_int)
    df["founders_count"] = df["founders"].apply(coerce_int)
    df["active_products_count"] = df["active_products"].apply(coerce_int)
    df["portfolio_companies_count"] = df["portfolio_companies"].apply(coerce_int)
    df["acquisitions_count"] = df["Acquisitions"].apply(coerce_int)
    df["employee_count_estimate"] = df["employee_range"].apply(parse_employee_range)
    df["funding_amount_usd"] = df["funding_amount"].apply(parse_currency)
    df["acquired_price_usd"] = df["acquired_price"].apply(parse_currency)

    revenue_min, revenue_max, revenue_avg = zip(*df["estimated_revenue"].apply(parse_money_range))
    df["estimated_revenue_min_usd"] = revenue_min
    df["estimated_revenue_max_usd"] = revenue_max
    df["estimated_revenue_avg_usd"] = revenue_avg

    df = df[df["normalized_name"].notna()].copy()
    df = df.drop_duplicates("normalized_name")

    columns_to_keep: Iterable[str] = [
        "name",
        "normalized_name",
        "Type",
        "headquarters_clean",
        "hq_city",
        "hq_state",
        "hq_country",
        "Status",
        "founded_year",
        "investment_stage",
        "exit_date",
        "Purpose",
        "number_of_investments",
        "number_of_lead_investments",
        "estimated_revenue_min_usd",
        "estimated_revenue_max_usd",
        "estimated_revenue_avg_usd",
        "founders_count",
        "employee_count_estimate",
        "funding_rounds_count",
        "funding_status",
        "active_products_count",
        "funding_amount_usd",
        "number_of_lead_investors",
        "number_of_investors",
        "acquired_price_usd",
        "acquisitions_count",
        "acquisition_status",
        "acquired_by",
        "ipo_status",
    ]

    optional_related = [
        "related_full_name",
        "related_job_title",
        "related_primary_org",
        "related_founded_org",
        "portfolio_companies_count",
    ]

    for col in optional_related:
        if col in df.columns:
            columns_to_keep.append(col)

    df_final = df.loc[:, [c for c in columns_to_keep if c in df.columns]].copy()
    df_final.sort_values("name", inplace=True)
    df_final.reset_index(drop=True, inplace=True)
    return df_final


def write_output(df: pd.DataFrame, name: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PROCESSED_DIR / f"{name}.parquet"
    csv_path = PROCESSED_DIR / f"{name}.csv"
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"Parquet write failed for {name}: {exc}. Falling back to CSV.")
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing Crunchbase dataset at {RAW_PATH}")

    df = pd.read_excel(RAW_PATH, dtype=str)
    cleaned = clean_dataframe(df)
    write_output(cleaned, "crunchbase_clean")
    print(f"Saved {len(cleaned)} cleaned Crunchbase companies to data/processed/crunchbase_clean.csv")


if __name__ == "__main__":
    main()


