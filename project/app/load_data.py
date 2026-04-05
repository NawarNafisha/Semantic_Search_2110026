"""Download the first 3000 rows from HF and save question/answer/embeddings.

Run:
    python -m app.load_data
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

# Hugging Face dataset requested in the project requirements.
DATASET_ID = "MartinElMolon/stackoverflow_preguntas_con_embeddings"
# We only keep the first 3000 rows.
MAX_ROWS = 3000


def main() -> None:
    """Load dataset, keep required fields, and export JSON."""
    # Load the train split from Hugging Face.
    dataset = load_dataset(DATASET_ID, split="train")

    # Keep only the first 3000 rows (or less if dataset is smaller).
    dataset = dataset.select(range(min(MAX_ROWS, len(dataset))))

    # Build a clean list with only required fields.
    rows = []
    for item in dataset:
        rows.append(
            {
                "question": item["question"],
                "answer": item["answer"],
                "embeddings": item["embeddings"],
            }
        )

    # Write to project/data/dataset.json.
    output_path = Path(__file__).resolve().parents[1] / "data" / "dataset.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
