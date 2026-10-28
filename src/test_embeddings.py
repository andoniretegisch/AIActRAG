import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


INPUT_PATH = Path("data/processed/chunks_v2.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    model = SentenceTransformer(MODEL_NAME)
    tokenizer = model.tokenizer

    max_length = model.max_seq_length

    token_counts = []

    for chunk in chunks:
        tokens = tokenizer(
            chunk["text"],
            truncation=False,
            add_special_tokens=True,
        )

        token_count = len(tokens["input_ids"])
        token_counts.append(token_count)

    truncated_chunks = [
        count for count in token_counts
        if count > max_length
    ]

    print(f"Number of chunks: {len(chunks)}")
    print(f"Model maximum sequence length: {max_length}")
    print(f"Minimum token count: {min(token_counts)}")
    print(f"Median token count: {sorted(token_counts)[len(token_counts) // 2]}")
    print(f"Maximum token count: {max(token_counts)}")
    print(f"Chunks exceeding limit: {len(truncated_chunks)}")

    print("\nFive longest chunks:")

    longest = sorted(
        zip(chunks, token_counts),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    for chunk, count in longest:
        print(
            f"Article {chunk['article_number']}: "
            f"{count} tokens — {chunk['article_title']}"
        )


if __name__ == "__main__":
    main()