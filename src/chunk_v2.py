
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


INPUT_PATH = Path("data/processed/articles.json")
OUTPUT_PATH = Path("data/processed/chunks_v2.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 200
CHUNK_OVERLAP = 40


def create_chunks(articles: list[dict], model) -> list[dict]:
    tokenizer = model.tokenizer
    chunks = []

    for article in articles:
        token_ids = tokenizer.encode(
            article["text"],
            add_special_tokens=False,
        )

        start = 0
        chunk_index = 1

        while start < len(token_ids):
            end = start + CHUNK_SIZE
            chunk_tokens = token_ids[start:end]

            text = tokenizer.decode(
                chunk_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            chunks.append({
                "chunk_id": f"{article['article_id']}_{chunk_index}",
                "article_id": article["article_id"],
                "article_number": article["article_number"],
                "article_title": article["article_title"],
                "chapter_id": article["chapter_id"],
                "chapter_title": article["chapter_title"],
                "section_id": article["section_id"],
                "section_title": article["section_title"],
                "chunk_index": chunk_index,
                "start_token": start,
                "end_token": min(end, len(token_ids)),
                "text": text,
            })

            if end >= len(token_ids):
                break

            start = end - CHUNK_OVERLAP
            chunk_index += 1

    return chunks


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH.resolve()}"
        )

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        articles = json.load(file)

    model = SentenceTransformer(MODEL_NAME)

    chunks = create_chunks(articles, model)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print(f"Articles loaded: {len(articles)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Chunk size: {CHUNK_SIZE} tokens")
    print(f"Overlap: {CHUNK_OVERLAP} tokens")
    print(f"Chunks saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
