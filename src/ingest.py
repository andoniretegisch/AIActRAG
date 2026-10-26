
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag


INPUT_PATH = Path("data/raw/ai_act_2026_july_27.html")
OUTPUT_PATH = Path("data/processed/articles.json")

CHAPTER_ID_PATTERN = re.compile(r"^cpt_[IVXLCDM]+$")
ARTICLE_ID_PATTERN = re.compile(r"^art_\d+[a-z]?$", re.IGNORECASE)


def get_readable_text(element: Tag) -> str:
    """Extract readable text while preserving paragraph boundaries."""
    text = element.get_text("\n", strip=True)

    # Remove empty lines while preserving the order of the text.
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    return "\n".join(lines)


def get_title(element: Tag, title_class: str) -> str | None:
    """
    Find a title using its expected CSS class.
    Fall back to the first paragraph if that class is absent.
    """
    title_element = element.select_one(f"p.{title_class}")

    if title_element:
        return get_readable_text(title_element)

    first_paragraph = element.find("p")

    if first_paragraph:
        return get_readable_text(first_paragraph)

    return None


def parse_articles(html_path: Path) -> list[dict]:
    """Extract articles grouped by chapter from the Regulation HTML."""

    with html_path.open("r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "lxml")

    # 1. Find the outer Regulation container.
    eli_container = soup.find(
        "div",
        class_="eli-container",
    )

    if eli_container is None:
        raise ValueError("Could not find div.eli-container")

    # 2. Find the main body, excluding the annexes.
    main_body = eli_container.find(
        "div",
        class_="eli-subdivision",
        id="enc_1",
    )

    if main_body is None:
        raise ValueError(
            "Could not find div.eli-subdivision#enc_1"
        )

    articles = []

    # 3. Find chapters in document order.
    chapters = main_body.find_all(
        "div",
        id=CHAPTER_ID_PATTERN,
    )

    if not chapters:
        raise ValueError(
            "No chapters found. Check the chapter HTML structure."
        )

    for chapter in chapters:
        chapter_id = chapter["id"]

        # 4. Extract the chapter title.
        chapter_title = get_title(
            chapter,
            "title-chapter-norm",
        )

        # 5. Find articles inside this chapter.
        chapter_articles = chapter.find_all(
            "div",
            class_="eli-subdivision",
            id=ARTICLE_ID_PATTERN,
        )

        for article in chapter_articles:
            article_id = article["id"]

            article_title = get_title(
                article,
                "title-article-norm",
            )

            # Preserve all article text, including its paragraphs.
            article_text = get_readable_text(article)

            articles.append(
                {
                    "section_type": "article",
                    "article_id": article_id,
                    "article_number": article_id.removeprefix("art_"),
                    "chapter_id": chapter_id,
                    "chapter_number": chapter_id.removeprefix("cpt_"),
                    "chapter_title": chapter_title,
                    "article_title": article_title,
                    "text": article_text,
                }
            )

    return articles


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input HTML file not found: {INPUT_PATH.resolve()}"
        )

    articles = parse_articles(INPUT_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            articles,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Articles extracted: {len(articles)}")
    print(f"JSON saved to: {OUTPUT_PATH.resolve()}")

    if articles:
        print("\nFirst extracted article:")
        print(json.dumps(articles[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
