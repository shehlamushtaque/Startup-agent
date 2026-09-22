"""
Quick diagnostic script to preview CompanyFundingTool output for manual queries.
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.agents.tools import CompanyFundingTool


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect funding comparables.")
    parser.add_argument("--industry", default=None)
    parser.add_argument("--country", default=None)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    tool = CompanyFundingTool()
    results = tool.run({"industry": args.industry, "country": args.country, "limit": args.limit})
    for row in results.get("results", []):
        print(row)


if __name__ == "__main__":
    main()

