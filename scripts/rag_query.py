"""Query handler + RAG inference pipeline for company QA index."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer


VECTOR_DIR_DEFAULT = Path("data/vector")
QA_METADATA_FILENAME = "company_qa_metadata.jsonl"
INDEX_FILENAME = "company_qa.faiss"
CONFIG_FILENAME = "company_qa_config.json"


class VectorStore:
    def __init__(self, index: faiss.Index, metadata: List[Dict[str, Any]], model_name: str):
        self.index = index
        self.metadata = metadata
        self.model = SentenceTransformer(model_name)

    @classmethod
    def load(cls, vector_dir: Path) -> "VectorStore":
        index_path = vector_dir / INDEX_FILENAME
        metadata_path = vector_dir / QA_METADATA_FILENAME
        config_path = vector_dir / CONFIG_FILENAME

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        index = faiss.read_index(str(index_path))

        metadata: List[Dict[str, Any]] = []
        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    metadata.append(json.loads(line))

        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")

        return cls(index=index, metadata=metadata, model_name=model_name)

    def encode_query(self, question: str) -> np.ndarray:
        embedding = self.model.encode(
            [question],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.astype("float32")

    def search(self, question: str, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        embedding = self.encode_query(question)
        scores, indices = self.index.search(embedding, top_k)
        results: List[Tuple[float, Dict[str, Any]]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            results.append((float(score), self.metadata[idx]))
        return results


def build_context(snippets: List[Tuple[float, Dict[str, Any]]]) -> str:
    lines = []
    for rank, (score, entry) in enumerate(snippets, start=1):
        prompt = entry.get("prompt", "")
        response = entry.get("response", "")
        metadata = entry.get("metadata", {})
        company = metadata.get("company", metadata.get("normalized_name", "Unknown"))
        lines.append(f"[Snippet #{rank} | Score {score:.3f} | Company: {company}]\nQ: {prompt}\nA: {response}\n")
    return "\n".join(lines)


def build_prompt(question: str, context: str) -> str:
    return (
        "You are a startup research assistant. Use the provided context to answer the user's question.\n"
        "If the context lacks the answer, say you don't have enough data.\n\n"
        f"Context:\n{context}\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer:"
    )


def call_llm(prompt: str, llm: str = "none", model_name: str | None = None) -> str:
    llm = llm.lower()
    if llm == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package not installed. Install `openai` to use this option.") from exc
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")
        client = OpenAI(api_key=api_key)
        completion = client.responses.create(
            model=model_name or "gpt-4.1-mini",
            input=prompt,
        )
        return completion.output_text

    if llm == "hf-local":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise RuntimeError("transformers not available for local inference.") from exc
        model_id = model_name or "microsoft/Phi-3-mini-4k-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        output = generator(prompt, max_new_tokens=256, do_sample=False)
        return output[0]["generated_text"]

    # Default: no LLM, return context for debugging
    return (
        "LLM mode is set to 'none'.\n"
        "Here is the constructed prompt you can feed into your own model manually:\n\n"
        f"{prompt}"
    )


def rag_answer(question: str, vector_store: VectorStore, top_k: int, llm: str, llm_model: str | None) -> Dict[str, Any]:
    hits = vector_store.search(question, top_k=top_k)
    context = build_context(hits)
    prompt = build_prompt(question, context)
    answer = call_llm(prompt, llm=llm, model_name=llm_model)
    return {
        "question": question,
        "answer": answer,
        "context": context,
        "hits": hits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG query against company QA vector index")
    parser.add_argument("--question", type=str, required=True, help="User question to answer")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest QA entries to retrieve")
    parser.add_argument("--vector-dir", type=Path, default=VECTOR_DIR_DEFAULT, help="Directory with FAISS index and metadata")
    parser.add_argument("--llm", type=str, default="none", choices=["none", "openai", "hf-local"], help="LLM backend")
    parser.add_argument("--llm-model", type=str, default=None, help="Override model name for the chosen LLM backend")
    args = parser.parse_args()

    vector_store = VectorStore.load(args.vector_dir)
    result = rag_answer(args.question, vector_store, args.top_k, args.llm, args.llm_model)

    print("=== Question ===")
    print(result["question"])
    print("\n=== Retrieved Context ===")
    print(result["context"] or "[No context retrieved]")
    print("\n=== Answer ===")
    print(result["answer"])


if __name__ == "__main__":
    main()

