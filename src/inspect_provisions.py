
from pathlib import Path

from bs4 import BeautifulSoup, Tag


HTML_PATH = Path("data/raw/ai_act_2026_july_27.html")
ARTICLE_ID = "art_6"

with HTML_PATH.open("r", encoding="utf-8") as file:
    soup = BeautifulSoup(file, "lxml")

article = soup.find("div", id=ARTICLE_ID)

if article is None:
    raise ValueError(f"Article {ARTICLE_ID} not found")


def inspect(element: Tag, depth: int = 0) -> None:
    indent = "  " * depth

    classes = " ".join(element.get("class", []))
    element_id = element.get("id", "")

    # Print only the text directly owned by this element,
    # not all descendant text, to avoid repeated output.
    direct_text = " ".join(
        str(child).strip()
        for child in element.children
        if isinstance(child, str) and child.strip()
    )

    preview = direct_text[:100]

    print(
        f"{indent}<{element.name}> "
        f"id={element_id!r} "
        f"class={classes!r} "
        f"text={preview!r}"
    )

    for child in element.children:
        if isinstance(child, Tag):
            inspect(child, depth + 1)


inspect(article)
