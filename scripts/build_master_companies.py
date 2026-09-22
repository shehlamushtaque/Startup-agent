"""Build master company table by joining cleaned Crunchbase data with objects."""

from __future__ import annotations

from rapidfuzz import process
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.utils.cleaning import clean_text, coerce_int, normalize_name, parse_currency


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def read_processed(name: str) -> pd.DataFrame:
    parquet_path = PROCESSED_DIR / f"{name}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    csv_path = PROCESSED_DIR / f"{name}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Processed dataset {name} not found in data/processed/")


def load_simplify_clean() -> pd.DataFrame:
    df = read_processed("simplify_clean")
    df["normalized_name"] = df["normalized_name"].apply(normalize_name)
    df["crunchbase_permalink"] = df["crunchbase_permalink"].apply(clean_text)
    return df


def load_crunchbase_clean() -> pd.DataFrame:
    df = read_processed("crunchbase_clean")
    df["normalized_name"] = df["normalized_name"].apply(normalize_name)
    df["type_clean"] = df["Type"].fillna("").str.lower()
    df["hq_country_clean"] = df["hq_country"].fillna("").apply(lambda x: clean_text(x) or "")
    return df


def load_growjo_clean() -> pd.DataFrame:
    df = read_processed("growjo_clean")
    df["normalized_name"] = df["normalized_name"].apply(normalize_name)
    return df


def load_objects() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "objects.csv", low_memory=False)
    df = df[df["entity_type"].str.lower() == "company"].copy()
    df["normalized_name"] = df["normalized_name"].apply(normalize_name)
    df["tag_list_clean"] = df["tag_list"].fillna("").str.lower()
    df["category_code_clean"] = df["category_code"].fillna("").str.lower()
    df["country_code_clean"] = df["country_code"].fillna("").str.upper()
    df["funding_total_usd"] = df["funding_total_usd"].apply(parse_currency)
    df["investment_rounds"] = df["investment_rounds"].apply(coerce_int)
    df["funding_rounds"] = df["funding_rounds"].apply(coerce_int)
    return df


def filter_us_saas(crunchbase: pd.DataFrame, objects: pd.DataFrame):
    us_aliases = {"united states", "united states of america", "usa", "us"}
    cb_mask_us = crunchbase["hq_country_clean"].str.lower().isin(us_aliases)
    cb_mask_saas = crunchbase["type_clean"].str.contains("saas", na=False)
    crunchbase_us = crunchbase[cb_mask_us & cb_mask_saas].copy()

    objects_us = objects[objects["country_code_clean"].isin({"USA", "US"})].copy()
    mask_saas = (
        objects_us["tag_list_clean"].str.contains("saas", na=False)
        | (objects_us["category_code_clean"] == "saas")
    )
    objects_us = objects_us[mask_saas]
    return crunchbase_us, objects_us


def build_master_table(
    crunchbase_us: pd.DataFrame,
    objects_us: pd.DataFrame,
    simplify: pd.DataFrame,
    growjo: pd.DataFrame,
) -> pd.DataFrame:
    master = crunchbase_us.merge(
        objects_us,
        on="normalized_name",
        how="left",
        suffixes=("_cb", "_objects"),
    )
    master["object_id"] = master.get("id")
    master["has_object_match"] = master["object_id"].notna()

    simplify_subset = simplify[
        [
            "normalized_name",
            "crunchbase_permalink",
            "name",
            "slug",
            "description",
            "short_description",
            "url",
            "linkedin",
            "year_founded",
            "employee_count_estimate",
            "funding_stage_clean",
            "customer_type_clean",
            "funding_total_usd",
        ]
    ].rename(
        columns={
            "name": "name_simplify",
            "url": "website",
            "linkedin": "linkedin_url",
        }
    )

    simplify_subset = simplify_subset.drop_duplicates("normalized_name")
    simplify_by_normalized = simplify_subset.set_index("normalized_name")

    master = master.merge(
        simplify_subset,
        on="normalized_name",
        how="left",
    )

    unmatched_index = master[master["name_simplify"].isna()].index
    if len(unmatched_index) > 0:
        simplify_by_permalink = simplify_subset.dropna(subset=["crunchbase_permalink"])\
            .drop_duplicates("crunchbase_permalink")
        simplify_by_permalink = simplify_by_permalink.set_index("crunchbase_permalink")
        simplify_dict = simplify_by_permalink.to_dict("index")
        records: list[dict[str, object]] = []
        indices: list[int] = []
        for idx, permalink in master.loc[unmatched_index, "permalink"].items():
            if not isinstance(permalink, str) or not permalink:
                continue
            data = simplify_dict.get(permalink)
            if data:
                records.append(data)
                indices.append(idx)
        if records:
            matched_df = pd.DataFrame(records, index=indices)
            master.loc[indices, matched_df.columns] = matched_df

    unmatched_index = master[master["name_simplify"].isna()].index
    if len(unmatched_index) > 0:
        simplify_name_dict = simplify_by_normalized.to_dict("index")
        simplify_names = [name for name in simplify_by_normalized.index if isinstance(name, str)]
        simplify_names_lower = {name.lower(): data for name, data in simplify_name_dict.items() if isinstance(name, str)}
        for idx, name in master.loc[unmatched_index, "normalized_name"].items():
            if not isinstance(name, str) or not name:
                continue
            lower = name.lower()
            if lower in simplify_names_lower:
                data = simplify_names_lower[lower]
                for col, value in data.items():
                    master.at[idx, col] = value
                continue
            matched = process.extractOne(name, simplify_names, score_cutoff=90)
            if matched:
                matched_name, score, _ = matched
                if matched_name[0] != name[0]:
                    continue
                if abs(len(matched_name) - len(name)) > 4:
                    continue
                data = simplify_name_dict.get(matched_name)
                if data:
                    for col, value in data.items():
                        master.at[idx, col] = value

    master["has_simplify_match"] = master["name_simplify"].notna()

    growjo_subset = growjo[
        [
            "normalized_name",
            "growjo_rank",
            "growjo_previous_rank",
            "growjo_employees",
            "growjo_estimated_revenue_usd",
            "growjo_growth_percent",
            "growjo_job_openings",
            "growjo_lead_investors",
            "growjo_accelerator",
            "growjo_business_type",
            "growjo_product_url",
            "growjo_indeed_url",
        ]
    ].drop_duplicates("normalized_name")
    growjo_by_normalized = growjo_subset.set_index("normalized_name")

    master = master.merge(growjo_subset, on="normalized_name", how="left")
    master["has_growjo_match"] = master["growjo_rank"].notna()

    unmatched_growjo = master[~master["has_growjo_match"] & master["normalized_name"].notna()].index
    if len(unmatched_growjo) > 0:
        growjo_names = [name for name in growjo_by_normalized.index if isinstance(name, str)]
        growjo_names_lower = {name.lower(): name for name in growjo_names}
        for idx in unmatched_growjo:
            name = master.at[idx, "normalized_name"]
            if not isinstance(name, str):
                continue
            lower = name.lower()
            if lower in growjo_names_lower:
                matched_name = growjo_names_lower[lower]
            else:
                matched = process.extractOne(name, growjo_names, score_cutoff=85)
                if not matched:
                    continue
                matched_name, score, _ = matched
                if matched_name and name and matched_name[0] != name[0]:
                    continue
                if abs(len(matched_name) - len(name)) > 6:
                    continue
            data = growjo_by_normalized.loc[matched_name].to_dict()
            for col, value in data.items():
                master.at[idx, col] = value
        master["has_growjo_match"] = master["growjo_rank"].notna()

    return master


def load_objects_lookup() -> pd.DataFrame:
    columns = ["id", "entity_type", "name", "permalink", "country_code", "category_code"]
    df = pd.read_csv(RAW_DIR / "objects.csv", usecols=columns, low_memory=False)
    df["id"] = df["id"].apply(clean_text)
    df["name"] = df["name"].apply(clean_text)
    return df


def enrich_with_investors(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    investments = read_processed("investments_clean")
    funds = read_processed("funds_clean")
    objects_lookup = load_objects_lookup()

    investments = investments[investments["funded_object_type"] == "c"].copy()
    investments = investments.dropna(subset=["funded_object_id"])

    investor_lookup = objects_lookup.rename(
        columns={
            "id": "investor_object_id",
            "entity_type": "investor_entity_type",
            "name": "investor_name",
            "permalink": "investor_permalink",
            "country_code": "investor_country_code",
            "category_code": "investor_category_code",
        }
    )

    investments = investments.merge(
        investor_lookup,
        on="investor_object_id",
        how="left",
    )

    funds_lookup = funds.rename(
        columns={
            "object_id": "investor_object_id",
            "name": "fund_name",
            "funded_at_date": "fund_funded_at_date",
            "raised_amount_usd": "fund_raised_amount_usd",
            "raised_currency_code": "fund_currency_code",
        }
    )[
        [
            "investor_object_id",
            "fund_name",
            "fund_funded_at_date",
            "fund_raised_amount_usd",
            "fund_currency_code",
        ]
    ]

    investments = investments.merge(
        funds_lookup,
        on="investor_object_id",
        how="left",
    )

    investments["investor_display_name"] = investments["investor_name"].fillna(investments["fund_name"])
    investments["investor_role"] = investments["investor_object_type"].map(
        {"f": "fund", "c": "company", "p": "person"}
    ).fillna("other")

    investments["created_at_iso"] = pd.to_datetime(investments["created_at_iso"], errors="coerce")

    agg = investments.groupby("funded_object_id").agg(
        investor_count=("investor_object_id", "nunique"),
        funding_round_count=("funding_round_id", "nunique"),
        latest_investment_at=("created_at_iso", "max"),
        investor_roles=("investor_role", lambda s: "; ".join(f"{role}:{count}" for role, count in s.value_counts().head(3).items())),
        investor_names=("investor_display_name", lambda s: "; ".join(sorted({name for name in s if isinstance(name, str)})[:10])),
    )

    agg["latest_investment_at"] = agg["latest_investment_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    agg = agg.reset_index().rename(columns={"funded_object_id": "object_id"})

    master = master.merge(agg, on="object_id", how="left")

    company_lookup = master[["object_id", "normalized_name", "name_cb", "name_simplify"]]
    edges = investments[
        [
            "funded_object_id",
            "investor_object_id",
            "investor_display_name",
            "investor_role",
            "investor_name",
            "fund_name",
            "funding_round_id",
            "created_at_iso",
            "updated_at_iso",
            "investor_entity_type",
            "investor_country_code",
            "investor_category_code",
            "investor_permalink",
            "fund_funded_at_date",
            "fund_raised_amount_usd",
            "fund_currency_code",
        ]
    ].copy()

    edges = edges.merge(
        company_lookup,
        left_on="funded_object_id",
        right_on="object_id",
        how="left",
    )
    edges = edges.rename(columns={"funded_object_id": "company_object_id"})
    edges["created_at_iso"] = edges["created_at_iso"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return master, edges


def ensure_processed_dir() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def write_outputs(frames: tuple[tuple[pd.DataFrame, str], ...]) -> None:
    for df, name in frames:
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
    ensure_processed_dir()

    crunchbase = load_crunchbase_clean()
    simplify = load_simplify_clean()
    growjo = load_growjo_clean()
    objects = load_objects()
    crunchbase_us, objects_us = filter_us_saas(crunchbase, objects)
    master = build_master_table(crunchbase_us, objects_us, simplify, growjo)
    master, investor_edges = enrich_with_investors(master)

    write_outputs(
        (
            (crunchbase_us, "crunchbase_us_saas"),
            (objects_us, "objects_us_saas"),
            (master, "master_companies"),
            (investor_edges, "company_investors"),
        )
    )

    joined_count = master["has_object_match"].sum()
    investor_company_count = master["investor_count"].fillna(0).astype(int).gt(0).sum()
    growth_company_count = master["has_growjo_match"].sum()
    print(
        "Saved:",
        len(crunchbase_us),
        "Crunchbase SaaS companies;",
        len(objects_us),
        "objects matches;",
        int(joined_count),
        "joined records;",
        int(investor_company_count),
        "companies with investor data;",
        int(growth_company_count),
        "companies with growth data",
    )


if __name__ == "__main__":
    main()


