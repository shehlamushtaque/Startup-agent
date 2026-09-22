"""Reusable data cleaning helpers for startup datasets."""

from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np


_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def normalize_name(value: Optional[str]) -> Optional[str]:
    """Normalize entity names for fuzzy matching.

    - Lowercases text
    - Strips whitespace
    - Removes punctuation/non-alphanumerics, collapsing to single spaces
    """

    if value is None:
        return None
    if isinstance(value, float):
        if np.isnan(value):
            return None
        value = str(value)
    elif not isinstance(value, str):
        value = str(value)
    value = value.strip().lower()
    if not value:
        return None
    value = _NON_ALPHANUMERIC.sub(" ", value)
    return " ".join(value.split()) or None


def parse_currency(value: Optional[str]) -> Optional[float]:
    """Convert currency strings like "$1.2B" or "$300,000" to floats in USD."""

    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        return float(value)
    value = clean_text(value)
    if value is None:
        return None
    if not value or value.lower() in {"nan", "none"}:
        return None

    multipliers = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}
    cleaned = value.replace(",", "").replace("$", "")
    suffix = cleaned[-1].lower()
    multiplier = 1.0
    if suffix in multipliers:
        multiplier = multipliers[suffix]
        cleaned = cleaned[:-1]

    try:
        return float(cleaned) * multiplier
    except ValueError:
        return None


def coerce_int(value: Optional[str]) -> Optional[int]:
    """Best-effort conversion to integer."""

    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return int(value)

    value = str(value).strip()
    if not value:
        return None
    value = value.replace(",", "")
    try:
        return int(float(value))
    except ValueError:
        return None


def clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float):
        if np.isnan(value):
            return None
    value = str(value)
    if not value:
        return None

    replacements = {
        "\u2014": "",  # em dash
        "\u2013": "",
        "\u00a0": " ",
        "_x000D_": " ",
        "\n": " ",
        "\r": " ",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)

    value = re.sub(r"\s+", " ", value)
    value = value.strip()
    return value or None


_NUMBERING_PREFIX = re.compile(r"^\d+\.\s*")


def strip_leading_numbering(value: Optional[str]) -> Optional[str]:
    text = clean_text(value)
    if text is None:
        return None
    return _NUMBERING_PREFIX.sub("", text)


_EMPLOYEE_RANGE = re.compile(r"(?P<low>\d{1,3}(?:,\d{3})*)(?:\s*[-–]\s*(?P<high>\d{1,3}(?:,\d{3})*))?(?P<plus>\+)?")


def parse_employee_range(value: Optional[str]) -> Optional[int]:
    text = clean_text(value)
    if text is None:
        return None
    match = _EMPLOYEE_RANGE.search(text)
    if not match:
        return coerce_int(text)

    low = int(match.group("low").replace(",", ""))
    high = match.group("high")
    plus = match.group("plus")
    if high:
        high_val = int(high.replace(",", ""))
    elif plus:
        high_val = low
    else:
        high_val = low
    return int((low + high_val) / 2)


def parse_money_range(value: Optional[str]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    text = clean_text(value)
    if text is None:
        return None, None, None

    separators = [" to ", " - ", "–"]
    for sep in separators:
        if sep in text:
            left, right = text.split(sep, 1)
            min_val = parse_currency(left)
            max_val = parse_currency(right)
            avg = None
            if min_val is not None and max_val is not None:
                avg = (min_val + max_val) / 2
            return min_val, max_val, avg

    single = parse_currency(text)
    return single, single, single


def parse_percentage(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        if isinstance(value, float) and np.isnan(value):
            return None
        return float(value)
    text = clean_text(value)
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1]
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


COMPANY_SIZE_BUCKETS = {
    1: (1, 10),
    2: (11, 50),
    3: (51, 100),
    4: (101, 250),
    5: (251, 500),
    6: (501, 1000),
    7: (1001, 5000),
    8: (5001, 10000),
}


def company_size_to_estimate(bucket: Optional[int]) -> Optional[int]:
    if bucket is None:
        return None
    if isinstance(bucket, float):
        if np.isnan(bucket):
            return None
        bucket = int(bucket)
    if isinstance(bucket, str):
        if not bucket.strip():
            return None
        try:
            bucket = int(float(bucket))
        except ValueError:
            return None

    bounds = COMPANY_SIZE_BUCKETS.get(bucket)
    if not bounds:
        return None
    low, high = bounds
    return int((low + high) / 2)


def split_object_ref(value: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    text = clean_text(value)
    if not text or ":" not in text:
        return None, None
    prefix, identifier = text.split(":", 1)
    prefix = prefix.strip() or None
    identifier = identifier.strip() or None
    return prefix.lower() if prefix else None, identifier


def safe_lower(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value.lower() if value else None


def contains_keyword(text: Optional[str], keyword: str) -> bool:
    if not text or not keyword:
        return False
    return keyword.lower() in text.lower()


