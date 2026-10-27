
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag


INPUT_PATH = Path("data/raw/ai_act_2026_july_27.html")
OUTPUT_PATH = Path("data/processed/articles.json")
SECTION_ID_PATTERN = re.compile(
    r"^cpt_[IVXLCDM]+\.sct_\d+$"
)

CHAPTER_ID_PATTERN = re.compile(r"^cpt_[IVXLCDM]+$")
ARTICLE_ID_PATTERN = re.compile(r"^art_\d+[a-z]?$", re.IGNORECASE)


def get_readable_text(element: Tag) -> str:
    """Extract readable text while preserving paragraph boundaries."""
    text = element.get_text("\n", strip=True)

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    return "\n".join(lines)

def get_section_title(section: Tag) -> str | None:
    """Extract the descriptive title of a section."""
    title_element = section.select_one("p.title-division-2")

    if title_element:
        return get_readable_text(title_element)

    return None


def get_title(element: Tag, title_class: str) -> str | None:
    """
    Find a title using its expected CSS class.
    Fall back to the first paragraph if the class is absent.
    """
    title_element = element.select_one(f"p.{title_class}")

    if title_element:
        return get_readable_text(title_element)

    first_paragraph = element.find("p")

    if first_paragraph:
        return get_readable_text(first_paragraph)

    return None


def get_chapter_title(chapter: Tag) -> str | None:
    """
    Extract the descriptive chapter title, e.g. 'HIGH-RISK AI SYSTEMS'.

    EUR-Lex separates the chapter label ('CHAPTER III') from its
    descriptive title in two different paragraphs.
    """
    title_element = chapter.select_one("p.title-division-2")

    if title_element:
        return get_readable_text(title_element)

    # Fallback for alternative chapter HTML structures.
    return get_title(chapter, "title-chapter-norm")


def get_article_title(article: Tag) -> str | None:
    """
    Extract the descriptive article title, e.g.
    'Classification rules for high-risk AI systems'.

    The article number and descriptive title are separate elements.
    """
    title_element = article.select_one(
        ":scope > div.eli-title > p.stitle-article-norm"
    )

    if title_element:
        return get_readable_text(title_element)

    # Fallback for alternative article HTML structures.
    return get_title(article, "stitle-article-norm")



def parse_articles(html_path: Path) -> list[dict]:
    """Extract articles with chapter and optional section metadata."""

    with html_path.open("r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "lxml")

    eli_container = soup.find("div", class_="eli-container")

    if eli_container is None:
        raise ValueError("Could not find div.eli-container")

    main_body = eli_container.find(
        "div",
        class_="eli-subdivision",
        id="enc_1",
    )

    if main_body is None:
        raise ValueError("Could not find div.eli-subdivision#enc_1")

    articles = []

    chapters = main_body.find_all(
        "div",
        id=CHAPTER_ID_PATTERN,
    )

    if not chapters:
        raise ValueError("No chapters found.")

    for chapter in chapters:
        chapter_id = chapter["id"]
        chapter_number = chapter_id.removeprefix("cpt_")
        chapter_title = get_chapter_title(chapter)

        current_section = {
            "section_id": None,
            "section_number": None,
            "section_title": None,
        }

        # Traverse only direct children to preserve document hierarchy.
        for child in chapter.find_all("div", recursive=False):

            child_id = child.get("id", "")

            # A section starts a new section context.
            if SECTION_ID_PATTERN.fullmatch(child_id):
                current_section = {
                    "section_id": child_id,
                    "section_number": child_id.rsplit(
                        ".sct_", 1
                    )[1],
                    "section_title": get_section_title(child),
                }

                # Extract articles directly belonging to this section.
                for article in child.find_all(
                    "div",
                    class_="eli-subdivision",
                    id=ARTICLE_ID_PATTERN,
                ):
                    article_id = article["id"]

                    articles.append(
                        {
                            "section_type": "article",
                            "article_id": article_id,
                            "article_number": article_id.removeprefix(
                                "art_"
                            ),
                            "chapter_id": chapter_id,
                            "chapter_number": chapter_number,
                            "chapter_title": chapter_title,
                            **current_section,
                            "article_title": get_article_title(article),
                            "text": get_readable_text(article),
                        }
                    )

            # Articles can also appear directly inside a chapter.
            elif ARTICLE_ID_PATTERN.fullmatch(child_id):
                articles.append(
                    {
                        "section_type": "article",
                        "article_id": child_id,
                        "article_number": child_id.removeprefix("art_"),
                        "chapter_id": chapter_id,
                        "chapter_number": chapter_number,
                        "chapter_title": chapter_title,
                        **current_section,
                        "article_title": get_article_title(child),
                        "text": get_readable_text(child),
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
