import pandas as pd
from pathlib import Path


def summarize_dataset(path: Path, sample_rows: int = 100, head_rows: int = 3) -> None:
    print(f"=== {path.name} ===")
    if not path.exists():
        print("File missing")
        print()
        return

    try:
        df = pd.read_csv(path, nrows=sample_rows)
    except Exception as exc:
        print(f"Failed to read sample: {exc}")
        print()
        return

    print(f"Rows total (approx): {sum(1 for _ in path.open('r', encoding='utf-8', errors='ignore')) - 1}")
    print("Columns & dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col}: {dtype}")

    print("Sample rows:")
    for i, row in enumerate(df.head(head_rows).to_dict(orient="records"), start=1):
        print(f"  Row {i}: {row}")
    print()


if __name__ == "__main__":
    root = Path("data/raw")
    datasets = [
        "Financial-QA-10k.csv",
        "export.csv",
        "Sass_commpanies_dataset.csv",
        "funds.csv",
        "global_startup_success_dataset.csv",
        "top_100_saas_companies_2025.csv",
        "startup_valuation_dataset.csv",
        "objects.csv",
        "investments.csv",
        "Growjo-1k-list.csv",
    ]

    for dataset in datasets:
        summarize_dataset(root / dataset, sample_rows=500, head_rows=3)
