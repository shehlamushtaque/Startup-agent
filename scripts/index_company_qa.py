"""Index company QA records into a FAISS vector store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer


QA_PATH_DEFAULT = Path("data/processed/qa/company_qa.jsonl")
VECTOR_DIR_DEFAULT = Path("data/vector")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"QA file not found at {path}")

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            record_id = data.get("metadata", {}).get("object_id") or f"qa_{idx}"
            records.append({"id": str(record_id), **data})
    return records


def build_text(record: Dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    lines = [
        f"Company: {metadata.get('company', '')}",
        f"Prompt: {record.get('prompt', '')}",
        f"Response: {record.get('response', '')}",
    ]
    funding_total = metadata.get("funding_total_usd")
    if funding_total is not None:
        lines.append(f"Funding Total USD: {funding_total}")
    growth = metadata.get("growjo_growth_percent")
    if growth is not None:
        lines.append(f"Growth Percent: {growth}")
    investor_names = metadata.get("investor_names") or []
    if investor_names:
        lines.append("Investors: " + ", ".join(investor_names))
    lines.append("Source Tables: " + ", ".join(metadata.get("source_tables", [])))
    return "\n".join(lines)


def embed_texts(model_name: str, texts: Iterable[str]) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        list(texts),
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def save_metadata(metadata: Iterable[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_config(path: Path, *, model_name: str, qa_count: int) -> None:
    config = {
        "model_name": model_name,
        "qa_count": qa_count,
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index QA records into a FAISS vector store")
    parser.add_argument(
        "--qa-path",
        type=Path,
        default=QA_PATH_DEFAULT,
        help="Path to company QA JSONL file",
    )
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=VECTOR_DIR_DEFAULT,
        help="Directory to store FAISS index and metadata",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="SentenceTransformer model to use",
    )
    args = parser.parse_args()

    records = load_records(args.qa_path)
    if not records:
        raise ValueError(f"No QA records found in {args.qa_path}")

    texts = [build_text(record) for record in records]
    embeddings = embed_texts(args.model, texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    args.vector_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.vector_dir / "company_qa.faiss"
    metadata_path = args.vector_dir / "company_qa_metadata.jsonl"
    config_path = args.vector_dir / "company_qa_config.json"

    faiss.write_index(index, str(index_path))
    save_metadata(records, metadata_path)
    write_config(config_path, model_name=args.model, qa_count=len(records))

    print(f"Indexed {len(records)} QA records into {index_path}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()

