
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDINGS_PATH = Path("data/processed/embeddings_v2.npy")
METADATA_PATH = Path("data/processed/embedding_metadata_v2.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def load_data():
    embeddings = np.load(EMBEDDINGS_PATH)

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    return embeddings, metadata


def retrieve(query, top_k=TOP_K):

    model = SentenceTransformer(MODEL_NAME)
    embeddings, metadata = load_data()
     
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    similarities = np.dot(embeddings, query_embedding)

    top_indices = np.argsort(similarities, descending=True)[:top_k]

    results = []

    for index in top_indices:
        chunk = metadata["chunks"][index]

        results.append({
            "score": float(similarities[index]),
            "article": chunk
        })

    return results


def main():
    queries = ["Does the AI Act apply to AI systems used exclusively for military, defence or national security purposes, regardless of who provides them?"]

   

    for query in queries:
        results = retrieve(query)

        for rank, result in enumerate(results, start=1):
            article = result["article"]

            print(f"\n--- Rank: {rank} ---")
            print(f"Similarity: {result['score']:.4f}")
            print(f"Query: {query}")
            print(f"Article: {article.get('article_number', 'Unknown')}")
            print(f"Title: {article.get('article_title', 'Unknown')}")
            print(f"Chunk id: {article["chunk_id"]}")
            print(f"Text: {article["text"]}")


if __name__ == "__main__":
    main()
