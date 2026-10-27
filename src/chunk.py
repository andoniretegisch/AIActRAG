
import json
from pathlib import Path


INPUT_PATH = Path("data/processed/articles.json")
OUTPUT_PATH = Path("data/processed/chunks_v1.json")


def create_chunks(articles: list[dict]) -> list[dict]:
    """Create one retrieval chunk per article."""
    chunks = []

    for article in articles:
        chunks.append(
            {
                "chapter_id": article["chapter_id"],
                "chapter_title": article["chapter_title"],
                "section_id": article["section_id"],
                "section_title": article["section_title"],
                "article_id": article["article_id"],
                "article_number": article["article_number"],
                "article_title": article["article_title"],
                "chunk_id": article["article_id"],
                "text": article["text"],
            }
        )

    return chunks


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH.resolve()}"
        )

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        articles = json.load(file)

    chunks = create_chunks(articles)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print(f"Articles loaded: {len(articles)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Chunks saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
