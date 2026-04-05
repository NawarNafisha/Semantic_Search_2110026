"""Script to download and store first 3000 StackOverflow rows as JSON.

Usage:
    python -m app.load_data
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

DATASET_ID = "MartinElMolon/stackoverflow_preguntas_con_embeddings"
MAX_ROWS = 3000


def _pick_split(dataset_obj: Any) -> Any:
    """Return the best available split (prefer train)."""
    if hasattr(dataset_obj, "keys"):
        keys = list(dataset_obj.keys())
        if "train" in keys:
            return dataset_obj["train"]
        if keys:
            return dataset_obj[keys[0]]
    return dataset_obj


def _find_key(sample: dict[str, Any], candidates: list[str], label: str) -> str:
    """Find first existing key from a list of candidates."""
    for key in candidates:
        if key in sample:
            return key
    raise KeyError(f"Could not find a `{label}` field. Available keys: {list(sample.keys())}")


def main() -> None:
    """Download dataset and persist required fields to data/dataset.json."""
    dataset_obj = load_dataset(DATASET_ID)
    dataset = _pick_split(dataset_obj)

    if len(dataset) == 0:
        raise ValueError("Dataset loaded but has no rows.")

    sample = dataset[0]
    question_key = _find_key(sample, ["question", "pregunta", "title"], "question")
    answer_key = _find_key(sample, ["answer", "respuesta", "body"], "answer")
    embedding_key = _find_key(sample, ["embeddings", "embedding", "vector"], "embeddings")

    dataset = dataset.select(range(min(MAX_ROWS, len(dataset))))

    rows = []
    for item in dataset:
        rows.append(
            {
                "question": item[question_key],
                "answer": item[answer_key],
                "embeddings": item[embedding_key],
            }
        )

    output_path = Path(__file__).resolve().parents[1] / "data" / "dataset.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
