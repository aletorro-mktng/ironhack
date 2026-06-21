"""Read novel manuscripts from the private knowledge base and split them into
chapters, so the app can generate per-chapter summaries and promos.

Manuscripts use Markdown headings like ``## Chapter 3: Tit for Tat {#anchor}``
(plus standalone sections such as ``## Epilogue``). We split on those ``## ``
headings, skip the table of contents, and return clean per-chapter text.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

# Anchored to the project root (parent of src/) so it works regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIVATE_MANUSCRIPT_DIR = _PROJECT_ROOT / "knowledge_base" / "private_manuscripts"

# Book display name -> candidate manuscript files (first existing/non-empty wins).
# Mirrors BOOK_MANUSCRIPT_FILES in context_filter and adds the sequel.
CHAPTER_BOOK_FILES: dict[str, tuple[str, ...]] = {
    "Mortal Vengeance": (
        "mortal_vengeance.md",
        "mortalvengeance.md",
    ),
    "Mortal Vengeance: A Grim Tale": (
        "mortal_vengeance_a_grim_tale.md",
        "mortalvengeance_agrimtale.md",
    ),
    "Mortal Vengeance II: To Reel or Not Too Real?": (
        "mortal_vengeance_ii_to_reel_or_not_too_real.md",
        "mortalvengeanceii_toreel_or_not_tooreal.md",
    ),
}

# Cap chapter text fed to the LLM (matches MANUSCRIPT_SOURCE_CHAR_CAP convention).
CHAPTER_TEXT_CAP = 24000

# A chapter heading, across all three manuscript styles:
#   "## Chapter 3: Tit for Tat {#anchor}"  (Mortal Vengeance)
#   "Chapter 1"                            (A Grim Tale — bare line, no title)
#   "CHAPTER 1: PAY ATTENTION, PUH-LEASE"  (Mortal Vengeance II — bare, uppercase)
# The title only counts when introduced by a ":" / "." / "-" separator, so a prose
# line like "Chapter 7 ended abruptly" (no separator) is NOT mistaken for a heading.
_CHAPTER_RE = re.compile(
    r"^(?:(?P<hash>#{1,3})\s+)?\*{0,2}\s*chapter\s+(?P<num>\d+)\s*\*{0,2}"
    r"(?:\s*[:.\-]\s*(?P<title>.+?))?"
    r"\s*(?:\{#[^}]*\})?\s*$",
    re.IGNORECASE,
)
# A markdown section heading (Epilogue, The B Roll, ...) that isn't a chapter.
_SECTION_RE = re.compile(r"^##\s+\*{0,2}(?P<title>[^\n{*]+?)\*{0,2}\s*(?:\{#[^}]*\})?\s*$")
_SKIP_TITLES = {"table of contents", "contents"}
# A table-of-contents entry is a page reference like ``[11](#chapter-1)`` — a
# bracketed page NUMBER linking to an anchor. Real headings don't carry one (their
# ``{#anchor}`` is curly-brace, not a markdown link), so this won't drop a heading
# that merely embeds a cross-reference link such as ``[see prologue](#prologue)``.
_PAGE_LINK_RE = re.compile(r"\[\d+\]\(#")


def _is_toc_line(line: str) -> bool:
    return bool(_PAGE_LINK_RE.search(line))


def _resolve_manuscript_path(book: str) -> Path | None:
    for filename in CHAPTER_BOOK_FILES.get(book, ()):
        path = PRIVATE_MANUSCRIPT_DIR / filename
        try:
            if path.exists() and path.stat().st_size > 0:
                return path
        except OSError:
            continue
    return None


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


@lru_cache(maxsize=16)
def parse_chapters(book: str) -> list[dict]:
    """Return ``[{id, label, title, text, char_count}]`` for a book.

    The table of contents and very short stub sections are skipped. Ids are stable
    (``chapter-3`` for numbered chapters, a title slug otherwise). Cached since
    manuscripts are static at runtime (avoids re-parsing on every page load).
    """
    path = _resolve_manuscript_path(book)
    if not path:
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []

    # Collect chapter/section boundaries (line index, label, id), skipping the ToC.
    boundaries: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        if _is_toc_line(line):
            continue
        chapter_match = _CHAPTER_RE.match(line)
        if chapter_match:
            number = chapter_match.group("num")
            rest = (chapter_match.group("title") or "").strip().rstrip(".").strip()
            label = f"Chapter {number}: {rest}" if rest else f"Chapter {number}"
            boundaries.append((i, label, f"chapter-{number}"))
            continue
        section_match = _SECTION_RE.match(line)
        if section_match:
            title = section_match.group("title").strip()
            if not title or title.lower() in _SKIP_TITLES:
                continue
            boundaries.append((i, title, _slugify(title) or f"section-{i}"))

    chapters: list[dict] = []
    used_ids: set[str] = set()
    for idx, (start_i, label, cid) in enumerate(boundaries):
        end_i = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
        body = "\n".join(lines[start_i + 1:end_i]).strip()
        if len(body) < 120:  # skip empty/stub headings
            continue
        if cid in used_ids:
            cid = f"{cid}-{idx}"
        used_ids.add(cid)
        chapters.append({
            "id": cid,
            "label": label,
            "title": label,
            "text": body,
            "char_count": len(body),
        })
    return chapters


def list_chapter_books() -> list[str]:
    """Books that have a manuscript on disk which parses into at least one chapter."""
    return [book for book in CHAPTER_BOOK_FILES if parse_chapters(book)]


def chapter_options(book: str) -> dict[str, str]:
    """`{id: label}` mapping for a book's chapters, for a ui.select."""
    return {chapter["id"]: chapter["label"] for chapter in parse_chapters(book)}


def get_chapter(book: str, chapter_id: str) -> dict | None:
    for chapter in parse_chapters(book):
        if chapter["id"] == chapter_id:
            return chapter
    return None


def chapter_text_for_prompt(chapter: dict) -> str:
    """Chapter body capped to a safe size for an LLM prompt."""
    text = str(chapter.get("text") or "")
    if len(text) <= CHAPTER_TEXT_CAP:
        return text
    return text[:CHAPTER_TEXT_CAP] + "\n\n[... chapter truncated for length ...]"
