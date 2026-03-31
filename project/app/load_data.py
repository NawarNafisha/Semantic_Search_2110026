"""Script to download and store first 3000 StackOverflow rows as JSON.

Usage:
    python -m app.load_data
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

# Dataset source requested by the user.
DATASET_ID = "MartinElMolon/stackoverflow_preguntas_con_embeddings"
MAX_ROWS = 3000


def main() -> None:
    """Download dataset and persist minimal fields to data/dataset.json."""
    # Load dataset split. Most HF datasets expose a "train" split.
    dataset = load_dataset(DATASET_ID, split="train")

    # Keep only the first 3000 rows.
    dataset = dataset.select(range(min(MAX_ROWS, len(dataset))))

    rows = []
    for item in dataset:
        # Keep only required fields.
        rows.append(
            {
                "question": item["question"],
                "answer": item["answer"],
                "embeddings": item["embeddings"],
            }
        )

    output_path = Path(__file__).resolve().parents[1] / "data" / "dataset.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save formatted JSON for readability.
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)

    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
