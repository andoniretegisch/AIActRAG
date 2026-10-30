
import json
from pathlib import Path

from retrieve import retrieve


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "evaluation" / "questions.json"
TOP_K = 5


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_question(item):
    question = item["question"]
    relevant_chunks = set(item["relevant_chunks"])

    results = retrieve(question, top_k=TOP_K)
   
    retrieved_ids = [
        str(result["article"]["chunk_id"])
        for result in results
    ]

    hits = [
        chunk_id for chunk_id in retrieved_ids
        if chunk_id in relevant_chunks
    ]

    recall = len(set(hits)) / len(relevant_chunks)

    precision = len(set(hits)) / TOP_K

    reciprocal_rank = 0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_chunks:
            reciprocal_rank = 1 / rank
            break

    return {
        "id": item["id"],
        "question": question,
        "recall@5": recall,
        "precision@5": precision,
        "reciprocal_rank": reciprocal_rank,
        "retrieved_chunks": retrieved_ids,
    }


def main():
    dataset = load_dataset(DATASET_PATH)

    results = [
        evaluate_question(item)
        for item in dataset
    ]

    n = len(results)

    if n == 0:
        print("Dataset is empty.")
        return

    # Aggregate metrics
    mean_recall = sum(
        r["recall@5"] for r in results
    ) / n

    mean_precision = sum(
        r["precision@5"] for r in results
    ) / n

    mrr = sum(
        r["reciprocal_rank"] for r in results
    ) / n

    print("\n=== Retrieval Evaluation ===")
    print(f"Questions: {n}")
    print(f"Recall@{TOP_K}: {mean_recall:.3f}")
    print(f"Precision@{TOP_K}: {mean_precision:.3f}")
    print(f"MRR: {mrr:.3f}")

    print("\n=== Per-question Results ===")

    for result in results:
        print(
            f"{result['id']} | "
            f"Recall@5: {result['recall@5']:.2f} | "
            f"MRR: {result['reciprocal_rank']:.2f}"
        )

    # Save detailed results
    output_path = ROOT / "data" / "processed" / "evaluation_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
