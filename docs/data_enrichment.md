## Master Dataset Enrichment

- Run `python scripts/enrich_master_companies.py` after updating `data/processed/growjo_clean.csv`.  
  - Outputs `data/processed/master_companies_enriched.parquet` with Growjo growth, revenue, employee, and rank signals merged into the master table.
- Use `python scripts/peek_company.py <keyword>` to confirm that a specific company now carries numeric metrics.
- The competitor and funding tools automatically prefer the enriched parquet if it exists, so no config changes are required after regeneration.

## Funding Diagnostics

- Use `python scripts/debug_funding.py --industry <industry> --country <country> --limit 5` to preview what `CompanyFundingTool` will return for a given query.
- The tool drops companies that lack `funding_total_usd`, so ensure those figures exist before expecting them to show up in reports.

