"""Clean Crunchbase investments and funds datasets into normalized tables."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.utils.cleaning import clean_text, parse_currency, split_object_ref


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def clean_investments() -> pd.DataFrame:
    path = RAW_DIR / "investments.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing investments dataset at {path}")

    df = pd.read_csv(path)

    df["funding_round_id"] = pd.to_numeric(df["funding_round_id"], errors="coerce").astype("Int64")

    df["funded_object_id"] = df["funded_object_id"].apply(clean_text)
    df[["funded_object_type", "funded_object_uuid"]] = pd.DataFrame(
        df["funded_object_id"].apply(split_object_ref).tolist(), index=df.index
    )

    df["investor_object_id"] = df["investor_object_id"].apply(clean_text)
    df[["investor_object_type", "investor_object_uuid"]] = pd.DataFrame(
        df["investor_object_id"].apply(split_object_ref).tolist(), index=df.index
    )

    for column in ["created_at", "updated_at"]:
        df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)
        df[f"{column}_iso"] = df[column].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    return df[
        [
            "id",
            "funding_round_id",
            "funded_object_id",
            "funded_object_type",
            "funded_object_uuid",
            "investor_object_id",
            "investor_object_type",
            "investor_object_uuid",
            "created_at_iso",
            "updated_at_iso",
        ]
    ]


def clean_funds() -> pd.DataFrame:
    path = RAW_DIR / "funds.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing funds dataset at {path}")

    df = pd.read_csv(path)

    df["name"] = df["name"].apply(clean_text)
    df["object_id"] = df["object_id"].apply(clean_text)
    df[["object_type", "object_uuid"]] = pd.DataFrame(
        df["object_id"].apply(split_object_ref).tolist(), index=df.index
    )

    df["raised_amount_usd"] = df["raised_amount"].apply(parse_currency)
    df["raised_currency_code"] = df["raised_currency_code"].fillna("").str.upper().replace({"": None})

    df["funded_at"] = pd.to_datetime(df["funded_at"], errors="coerce", utc=True)
    df["funded_at_date"] = df["funded_at"].dt.strftime("%Y-%m-%d")

    for column in ["created_at", "updated_at"]:
        df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)
        df[f"{column}_iso"] = df[column].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    return df[
        [
            "id",
            "fund_id",
            "object_id",
            "object_type",
            "object_uuid",
            "name",
            "funded_at_date",
            "raised_amount_usd",
            "raised_currency_code",
            "source_url",
            "created_at_iso",
            "updated_at_iso",
        ]
    ]


def write_output(df: pd.DataFrame, name: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PROCESSED_DIR / f"{name}.parquet"
    csv_path = PROCESSED_DIR / f"{name}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)


def main() -> None:
    investments = clean_investments()
    funds = clean_funds()

    write_output(investments, "investments_clean")
    write_output(funds, "funds_clean")

    print(
        f"Saved {len(investments)} investments and {len(funds)} funds "
        "to data/processed/{investments_clean,funds_clean}.{csv,parquet}"
    )


if __name__ == "__main__":
    main()


