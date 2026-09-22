"""Tool implementations used by specialist agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import re

import faiss  # type: ignore
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data/processed")
VECTOR_DIR = Path("data/vector")


class CompanyFundingTool:
    """Retrieve funding metrics from the master company table."""

    name = "CompanyFundingTool"

    TEXT_COLUMNS = [
        "Type",
        "normalized_name",
        "short_description",
        "description",
        "industries",
        "categories",
    ]

    def __init__(self, master_path: Optional[Path] = None):
        enriched_path = DATA_DIR / "master_companies_enriched.parquet"
        path = master_path or (enriched_path if enriched_path.exists() else DATA_DIR / "master_companies.parquet")
        if not path.exists():
            raise FileNotFoundError(f"Master company table not found at {path}")
        df = pd.read_parquet(path)
        numeric_columns = [
            "growjo_rank",
            "growjo_growth_percent",
            "growjo_employees",
            "growjo_estimated_revenue_usd",
            "funding_total_usd",
            "funding_round_count",
        ]
        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        # Pre-compute normalized columns for keyword filtering
        self.keyword_columns: List[str] = []
        for column in self.TEXT_COLUMNS:
            if column in df.columns:
                normalized_col = f"__norm_{column.lower()}"
                df[normalized_col] = (
                    df[column]
                    .fillna("")
                    .str.lower()
                    .str.replace(r"[^a-z0-9\s]+", " ", regex=True)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                )
                self.keyword_columns.append(normalized_col)

        self.df = df

    def _filter_by_keywords(self, df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
        if not keywords or not self.keyword_columns:
            return df
        tokens = [kw.lower().strip() for kw in keywords if kw]
        if not tokens:
            return df
        mask = pd.Series(False, index=df.index)
        for column in self.keyword_columns:
            if column in df.columns:
                series = df[column]
                column_mask = pd.Series(False, index=df.index)
                for token in tokens:
                    column_mask |= series.str.contains(token, na=False)
                mask |= column_mask
        if mask.any():
            return df[mask]
        return df

    def run(self, query: Dict[str, Any]) -> Dict[str, Any]:
        df = self.df
        industry = query.get("industry")
        country = query.get("country")
        keywords = query.get("keywords") or []
        limit = query.get("limit", 5)

        if isinstance(keywords, str):
            keywords = [keywords]

        if industry:
            mask = df["Type"].fillna("").str.contains(industry, case=False, na=False)
            if mask.any():
                df = df[mask]
        if keywords:
            filtered = self._filter_by_keywords(df, keywords)
            if not filtered.empty:
                df = filtered
        if country:
            mask = df["hq_country"].fillna("").str.contains(country, case=False, na=False)
            if mask.any():
                df = df[mask]

        if df.empty:
            df = self.df.copy()

        if "funding_total_usd" not in df.columns:
            return {"results": []}
        df = df[df["funding_total_usd"].notna()]
        if df.empty:
            return {"results": []}
        df = df.sort_values("funding_total_usd", ascending=False).head(limit)
        columns = [
            "normalized_name",
            "name_cb",
            "funding_total_usd",
            "funding_round_count",
            "investor_names",
            "latest_investment_at",
            "growjo_growth_percent",
            "hq_country",
            "hq_region",
            "hq_location",
            "Type",
            "industries",
        ]
        available_columns = [col for col in columns if col in df.columns]
        records = df[available_columns].fillna("").to_dict(orient="records")
        return {"results": records}


class CompetitorGrowthTool:
    """Surface competitors and growth metrics."""

    name = "CompetitorGrowthTool"

    _GEO_SYNONYMS = {
        "us": ["us", "usa", "united states", "united states of america", "america"],
        "usa": ["us", "usa", "united states", "united states of america", "america"],
        "united states": ["united states", "us", "usa", "united states of america", "america"],
        "europe": ["europe", "eu"],
        "uk": ["united kingdom", "uk", "england", "great britain", "gb"],
        "united kingdom": ["united kingdom", "uk", "england", "great britain", "gb"],
    }

    METRIC_COLUMNS = [
        "growjo_growth_percent",
        "growjo_estimated_revenue_usd",
        "growjo_employees",
    ]

    def __init__(self, master_path: Optional[Path] = None):
        enriched_path = DATA_DIR / "master_companies_enriched.parquet"
        path = master_path or (enriched_path if enriched_path.exists() else DATA_DIR / "master_companies.parquet")
        if not path.exists():
            raise FileNotFoundError(f"Master company table not found at {path}")
        df = pd.read_parquet(path)
        numeric_columns = self.METRIC_COLUMNS + ["growjo_rank"]
        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        df["normalized_industry"] = (
            df["Type"]
            .fillna("")
            .str.lower()
            .str.replace(r"[^a-z0-9\s]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df["normalized_country"] = (
            df["hq_country"]
            .fillna("")
            .str.lower()
            .str.replace(r"[^a-z0-9\s]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df["normalized_name"] = (
            df["normalized_name"]
            .fillna("")
            .str.lower()
            .str.replace(r"[^a-z0-9\s]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        self.keyword_columns = ["normalized_industry", "normalized_name"]
        self.excluded_names = {
            "lucid",
            "lucid software",
            "reputation.com",
            "reputation",
        }
        self.df = df

    @staticmethod
    def _tokenize(value: str) -> List[str]:
        return [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]

    def _filter_by_industry(self, df: pd.DataFrame, industry: str) -> pd.DataFrame:
        normalized = industry.lower().strip()
        if not normalized:
            return df
        tokens = self._tokenize(normalized)
        series = df["normalized_industry"]
        mask = series.str.contains(normalized, na=False)
        if mask.any():
            return df[mask]
        if tokens:
            all_mask = pd.Series(True, index=series.index)
            for token in tokens:
                all_mask &= series.str.contains(token, na=False)
            if all_mask.any():
                return df[all_mask]
            any_mask = pd.Series(False, index=series.index)
            for token in tokens:
                any_mask |= series.str.contains(token, na=False)
            if any_mask.any():
                return df[any_mask]
            # fall back to company names
            name_mask = pd.Series(False, index=series.index)
            for token in tokens:
                name_mask |= df["normalized_name"].str.contains(token, na=False)
            if name_mask.any():
                return df[name_mask]
        return df.head(0)

    def _filter_by_geography(self, df: pd.DataFrame, geography: str) -> pd.DataFrame:
        geo = geography.lower().strip()
        if not geo:
            return df
        synonyms = self._GEO_SYNONYMS.get(geo, [geo])
        mask = pd.Series(False, index=df.index)
        for synonym in synonyms:
            mask |= df["normalized_country"].str.contains(synonym, na=False)
        if mask.any():
            return df[mask]
        return df.head(0)

    def _filter_by_keywords(self, df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
        tokens = [kw.lower().strip() for kw in keywords if kw]
        if not tokens:
            return df
        mask = pd.Series(False, index=df.index)
        for column in self.keyword_columns:
            if column in df.columns:
                series = df[column]
                column_mask = pd.Series(False, index=df.index)
                for token in tokens:
                    column_mask |= series.str.contains(token, na=False)
                mask |= column_mask
        if mask.any():
            return df[mask]
        return df

    def run(self, query: Dict[str, Any]) -> Dict[str, Any]:
        df = self.df
        debug: Dict[str, Any] = {"applied_filters": {}}

        industry = query.get("industry")
        geography = query.get("geography")
        keywords = query.get("keywords") or []
        limit = query.get("limit", 5)

        if isinstance(keywords, str):
            keywords = [keywords]

        if industry:
            filtered = self._filter_by_industry(df, industry)
            if not filtered.empty:
                df = filtered
                debug["applied_filters"]["industry"] = industry
            else:
                debug["applied_filters"]["industry"] = "fallback-none"

        if geography:
            filtered_geo = self._filter_by_geography(df, geography)
            if not filtered_geo.empty:
                df = filtered_geo
                debug["applied_filters"]["geography"] = geography
            else:
                debug["applied_filters"]["geography"] = "fallback-none"

        if keywords:
            filtered_kw = self._filter_by_keywords(df, keywords)
            if not filtered_kw.empty:
                df = filtered_kw
                debug["applied_filters"]["keywords"] = keywords

        if df.empty:
            # fall back to top overall growth performers if filters fail completely
            df = self.df.copy()
            debug["applied_filters"]["fallback"] = "returning top growth companies"

        df = df.sort_values(
            ["growjo_growth_percent", "growjo_rank"],
            ascending=[False, True],
            na_position="last",
        )
        metric_cols = [col for col in self.METRIC_COLUMNS if col in df.columns]
        if metric_cols:
            metrics_mask = df[metric_cols].notna().any(axis=1)
            filtered_metrics = df[metrics_mask]
            if not filtered_metrics.empty:
                df = filtered_metrics
            else:
                global_metrics = self.df[self.df[metric_cols].notna().any(axis=1)]
                if not global_metrics.empty:
                    df = global_metrics.sort_values(
                        ["growjo_growth_percent", "growjo_rank"],
                        ascending=[False, True],
                        na_position="last",
                    )
                else:
                    df = df.head(0)
        include_blacklisted = any(kw.lower().strip() in self.excluded_names for kw in keywords)

        df = df.head(limit * 2 if not include_blacklisted else limit)

        if not include_blacklisted:
            filtered_no_blacklist = df[~df["normalized_name"].isin(self.excluded_names)]
            if not filtered_no_blacklist.empty:
                df = filtered_no_blacklist
        df = df.head(limit)

        columns = [
            "normalized_name",
            "name_cb",
            "growjo_rank",
            "growjo_growth_percent",
            "growjo_employees",
            "growjo_estimated_revenue_usd",
            "growjo_product_url",
            "hq_country",
            "hq_state",
            "hq_city",
            "Type",
            "category_code_clean",
        ]
        existing_cols = [col for col in columns if col in df.columns]
        selected = df[existing_cols].copy()
        records = selected.where(pd.notnull(selected), None).to_dict(orient="records")
        return {"results": records, "debug": debug}


class CompanyQAIndexTool:
    """Semantic search over the QA index."""

    name = "CompanyQAIndexTool"

    def __init__(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        index_path = index_path or VECTOR_DIR / "company_qa.faiss"
        metadata_path = metadata_path or VECTOR_DIR / "company_qa_metadata.jsonl"

        if not index_path.exists():
            raise FileNotFoundError(f"QA index not found at {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"QA metadata not found at {metadata_path}")

        self.index = faiss.read_index(str(index_path))
        self.metadata: List[Dict[str, Any]] = []
        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))
        self.model = SentenceTransformer(model_name)

    def run(self, query: Dict[str, Any]) -> Dict[str, Any]:
        question = query.get("question")
        top_k = query.get("top_k", 5)
        if not question:
            return {"results": []}

        embedding = self.model.encode(
            [question],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, indices = self.index.search(embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            entry = self.metadata[idx]
            results.append(
                {
                    "score": float(score),
                    "prompt": entry.get("prompt"),
                    "response": entry.get("response"),
                    "metadata": entry.get("metadata", {}),
                }
            )
        return {"results": results}


class BraveSearchTool:
    """Wrapper around the Brave Search API."""

    name = "BraveSearchTool"
    API_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(
        self,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ):
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError("BraveSearchTool requires BRAVE_API_KEY environment variable or api_key parameter.")
        self.session = session or requests.Session()

    def run(self, query: Dict[str, Any]) -> Dict[str, Any]:
        search_query = query.get("query")
        if not search_query:
            return {"results": []}

        num_results = int(query.get("num_results", 5))
        num_results = min(max(num_results, 1), 20)

        params = {
            "q": search_query,
            "count": num_results,
            "freshness": query.get("freshness", "month"),
        }
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }

        response = self.session.get(self.API_URL, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        payload = response.json()
        web_results = payload.get("web", {}).get("results", [])
        results: List[Dict[str, Any]] = []
        for item in web_results:
            results.append(
                {
                    "title": item.get("title"),
                    "link": item.get("url"),
                    "snippet": item.get("snippet"),
                    "source": item.get("source"),
                    "age": item.get("age"),
                }
            )
        return {"results": results, "query": search_query}


class SerpApiTool:
    """Query SerpApi for Google search results."""

    name = "SerpApiTool"
    API_URL = "https://serpapi.com/search"

    def __init__(self, api_key: Optional[str] = None, session: Optional[requests.Session] = None):
        self.api_key = api_key or os.getenv("SERP_API_KEY")
        if not self.api_key:
            raise ValueError("SerpApiTool requires SERP_API_KEY environment variable or api_key parameter.")
        self.session = session or requests.Session()

    def run(self, query: Dict[str, Any]) -> Dict[str, Any]:
        search_query = query.get("query")
        if not search_query:
            return {"results": []}

        engine = query.get("engine", "google")
        search_type = query.get("search_type")
        params: Dict[str, Any] = {
            "api_key": self.api_key,
            "engine": "google_news" if search_type == "news" else engine,
            "q": search_query,
            "num": min(int(query.get("limit", 10)), 20),
        }
        if location := query.get("location"):
            params["location"] = location
        if gl := query.get("gl"):
            params["gl"] = gl
        if hl := query.get("hl"):
            params["hl"] = hl

        try:
            response = self.session.get(self.API_URL, params=params, timeout=15)
            response.raise_for_status()
        except requests.RequestException as exc:
            return {"results": [], "error": str(exc), "params": params}

        try:
            data = response.json()
        except ValueError:
            return {"results": [], "error": "Invalid JSON response from SerpApi.", "params": params}

        results: List[Dict[str, Any]] = []
        if params["engine"] == "google_news":
            for item in data.get("news_results", []) or []:
                results.append(
                    {
                        "title": item.get("title"),
                        "link": item.get("link"),
                        "snippet": item.get("snippet"),
                        "source": item.get("source"),
                        "date": item.get("date"),
                    }
                )
        else:
            for item in data.get("organic_results", []) or []:
                results.append(
                    {
                        "title": item.get("title"),
                        "link": item.get("link"),
                        "snippet": item.get("snippet"),
                        "source": item.get("displayed_link"),
                    }
                )

        return {"results": results, "params": params}


class SecFormDTool:
    """Access recent SEC Form D filings via sec-api.io Full-Text Search API."""

    name = "SecFormDTool"
    API_URL = "https://api.sec-api.io/full-text-search"

    def __init__(
        self,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
        parse_xml: bool = True,
    ):
        self.api_key = api_key or os.getenv("SEC_API_KEY") or os.getenv("Sec_API_KEY")
        if not self.api_key:
            raise ValueError("SecFormDTool requires SEC_API_KEY environment variable or api_key parameter.")
        self.session = session or requests.Session()
        self.parse_xml = parse_xml

    def run(self, query: Dict[str, Any]) -> Dict[str, Any]:
        form_type = query.get("form_type")
        if form_type:
            form_type = form_type.replace("Form", "").strip().upper()

        payload: Dict[str, Any] = {
            "query": query.get("search") or "",
            "formTypes": [form_type or "D"],
            "page": str(query.get("page", 1)),
        }
        if limit := query.get("limit"):
            payload["size"] = int(limit)
        if start_date := query.get("from"):
            payload["startDate"] = start_date
        if end_date := query.get("to"):
            payload["endDate"] = end_date
        if cik := query.get("cik"):
            payload["ciks"] = [cik]

        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }
        try:
            response = self.session.post(self.API_URL, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
        except requests.RequestException as exc:
            return {"results": [], "error": str(exc), "payload": payload}

        try:
            data = response.json()
        except ValueError:
            return {"results": [], "error": "Invalid JSON response from SEC API.", "payload": payload}

        filings = data.get("filings") or []
        normalized: List[Dict[str, Any]] = []
        
        # Import here to avoid circular dependency
        if self.parse_xml:
            try:
                from backend.agents.xml_parser import enrich_sec_filing
            except ImportError:
                enrich_sec_filing = None
        else:
            enrich_sec_filing = None

        for filing in filings:
            base_data = {
                "id": filing.get("accessionNo") or filing.get("accessionNumber"),
                "company": filing.get("companyNameLong"),
                "cik": filing.get("cik"),
                "filed_at": filing.get("filedAt"),
                "form_type": filing.get("formType"),
                "description": filing.get("description"),
                "link": filing.get("filingUrl"),
                "raw": filing,
            }
            
            # Enrich with XML data if enabled
            if enrich_sec_filing and base_data.get("link"):
                try:
                    enriched = enrich_sec_filing(base_data, self.session)
                    # Only update if we actually got new data
                    if enriched != base_data:
                        base_data = enriched
                        # Add flag to indicate XML was parsed
                        base_data["xml_parsed"] = True
                except Exception as e:
                    # Continue without XML enrichment on error (SEC.gov may block or XML structure may differ)
                    base_data["xml_parsed"] = False
                    base_data["xml_error"] = str(e)[:100]  # Store error for debugging
            
            normalized.append(base_data)
        return {"results": normalized, "payload": payload}

