
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


INPUT_PATH = Path("data/processed/chunks_v2.json")
EMBEDDINGS_PATH = Path("data/processed/embeddings_v2.npy")
METADATA_PATH = Path("data/processed/embedding_metadata_v2.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def main() -> None:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH.resolve()}"
        )

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("No chunks found.")

    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["embedding_text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.save(EMBEDDINGS_PATH, embeddings)

    metadata = {
        "model_name": MODEL_NAME,
        "embedding_dimension": int(embeddings.shape[1]),
        "number_of_chunks": len(chunks),
        "chunks": chunks,
    }

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    # Report
    print(f"\nChunks embedded: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Embeddings saved to: {EMBEDDINGS_PATH.resolve()}")
    print(f"Metadata saved to: {METADATA_PATH.resolve()}")


if __name__ == "__main__":
    main()
