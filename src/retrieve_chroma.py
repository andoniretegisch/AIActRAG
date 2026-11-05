
import chromadb as cmdb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def retrieve(query, model, collection, top_k=TOP_K):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )

    return results


def format_results(results):
    formatted_chunks = []

    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]

        chunk = {
            "rank": i + 1,
            "chunk_id": results["ids"][0][i],
            "article_id": metadata.get("article_id"),
            "article_number": metadata.get("article_number"),
            "article_title": metadata.get("article_title"),
            "chapter_title": metadata.get("chapter_title"),
            "section_title": metadata.get("section_title"),
            "distance": results["distances"][0][i],
            "text": results["documents"][0][i],
        }

        formatted_chunks.append(chunk)

    return formatted_chunks


def print_results(query, chunks):
    print(f"\nQuery: {query}")
    print("=" * 100)

    if not chunks:
        print("No chunks retrieved.")
        return

    for chunk in chunks:
        print(f"\nRank: {chunk['rank']}")
        print(
            f"Source: Article {chunk['article_number']} "
            f"- {chunk['article_title']}"
        )

        if chunk["chapter_title"]:
            print(f"Chapter: {chunk['chapter_title']}")

        if chunk["section_title"]:
            print(f"Section: {chunk['section_title']}")

        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Cosine distance: {chunk['distance']:.4f}")
        print("-" * 100)
        print(chunk["text"])
        print("-" * 100)


def main():
    queries = [
        "Does the AI Act apply to AI systems used exclusively for military, defence or national security purposes, regardless of who provides them?"
    ]

    model = SentenceTransformer(MODEL_NAME)

    persistent_client = cmdb.PersistentClient(path="data/chroma_db")
    collection = persistent_client.get_collection("ai_act_chunks")

    for query in queries:
        results = retrieve(query, model, collection)
        chunks = format_results(results)
        print_results(query, chunks)


if __name__ == "__main__":
    main()
