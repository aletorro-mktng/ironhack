from __future__ import annotations

import re
from pathlib import Path

from document_processor import load_knowledge_base
from llm_integration import generate_text
from quote_semantic import query_relevance, quote_entry_id
from manuscript_semantic import (
    query_relevance as manuscript_query_relevance,
    chunk_id as manuscript_chunk_id,
)


MAX_CHUNK_CHARS = 1800
TOP_CHUNKS = 10
QUOTE_POST_ENTRY_CAP = 80
# Weight given to semantic (embedding) similarity when blended into the keyword
# score, and how strongly a matching mood/category tag boosts an entry.
SEMANTIC_SCORE_WEIGHT = 30
# Manuscript chunks are large, so keyword overlap alone can be noisy; weight the
# embedding-similarity signal higher than the quote-bank blend so genuinely
# relevant passages surface even with little literal keyword overlap.
MANUSCRIPT_SEMANTIC_WEIGHT = 80
MOOD_TAG_MATCH_BOOST = 40
# Two-tier source authority: primary (brand voice, playbook, canon, quote bank) is
# authoritative and outweighs secondary reference material (market/competitor/platform
# research) when ranking retrieved context. Other layers are neutral.
PRIMARY_LAYER_WEIGHT = 0.70
SECONDARY_LAYER_WEIGHT = 0.30
LAYER_WEIGHTS = {"primary": PRIMARY_LAYER_WEIGHT, "secondary": SECONDARY_LAYER_WEIGHT}


def layer_weight(layer: str) -> float:
    """Authority multiplier for a knowledge-base layer (primary > secondary)."""
    return LAYER_WEIGHTS.get(str(layer or "").lower(), 0.5)
QUOTE_POST_MANUSCRIPT_CAP = 24
MANUSCRIPT_CONTEXT_CAP = 18
MANUSCRIPT_SOURCE_CHAR_CAP = 24000
MIN_QUOTE_OPTIONS = 5
PRIVATE_MANUSCRIPT_DIR = Path("knowledge_base/private_manuscripts")
BOOK_MANUSCRIPT_FILES = {
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

# Long-form / editorial outputs draw from the whole knowledge base, not just the
# top keyword-matched chunks, so they can weave author interviews, awards, reader
# and critic reviews, themes, brand voice, book info, and pull quotes into one.
COMPREHENSIVE_CONTENT_TYPES = {
    "blog_post",
    "podcast",
    "press_release",
    "character_spotlight",
    "newsletter_blurb",
    "youtube_content",
    "linkedin_content",
}

# The quote bank is huge; cap how many of its chunks ride along for comprehensive runs.
QUOTE_BANK_CAP = 8
# Per-source character budget so each source is represented; large sources (e.g.
# real_reviews) keep their most relevant chunks, small ones are included in full.
PER_SOURCE_CHAR_CAP = 16000
# High ceiling — the per-source budget governs total size, not this count.
COMPREHENSIVE_CAP = 700


STOPWORDS = {
    "the", "and", "or", "for", "with", "from", "into", "that", "this",
    "about", "using", "only", "when", "what", "which", "should", "would",
    "create", "generate", "write", "post", "content", "quote", "quotes",
    "a", "an", "to", "of", "in", "on", "by", "as", "is", "are", "be"
}

QUOTE_BANK_REQUEST_PHRASES = (
    "quote bank",
    "book quote",
    "book quotes",
    "character quote",
    "character quotes",
    "funny quote",
    "sad quote",
    "scary quote",
    "romantic quote",
    "snarky quote",
    "brutal quote",
    "mortal vengeance quote",
    "a grim tale quote",
    "grim tale quote",
)

KNOWN_CHARACTER_ALIASES = {
    "alex": "Alex Herrera",
    "alex herrera": "Alex Herrera",
    "melissa": "Melissa Rocha",
    "melissa rocha": "Melissa Rocha",
    "monika": "Mónika Torres",
    "mónika": "Mónika Torres",
    "monika torres": "Mónika Torres",
    "mónika torres": "Mónika Torres",
    "mario": "Mario Stinga",
    "mario stinga": "Mario Stinga",
    "manuel": "Manuel Freites",
    "manuel freites": "Manuel Freites",
    "fernando": "Fernando Pepino",
    "fernando pepino": "Fernando Pepino",
    "enrique": "Enrique Hartling",
    "enrique hartling": "Enrique Hartling",
    "maria": "María García",
    "maría": "María García",
    "maria garcia": "María García",
    "maría garcía": "María García",
    "julian": "Julián Díaz",
    "julián": "Julián Díaz",
    "julian diaz": "Julián Díaz",
    "julián díaz": "Julián Díaz",
    "lucia": "Lucía Salgado",
    "lucía": "Lucía Salgado",
    "lucia salgado": "Lucía Salgado",
    "lucía salgado": "Lucía Salgado",
    "marcos": "Marcos",
    "lourdes": "Profesora Lourdes",
    "profesora lourdes": "Profesora Lourdes",
    "ricardo": "Lieutenant Ricardo García",
    "lieutenant ricardo": "Lieutenant Ricardo García",
    "lieutenant ricardo garcia": "Lieutenant Ricardo García",
    "lieutenant ricardo garcía": "Lieutenant Ricardo García",
    "grim cojuelo": "The Grim Cojuelo",
    "the grim cojuelo": "The Grim Cojuelo",
    "dona silvia": "Doña Silvia",
    "doña silvia": "Doña Silvia",
    "padre angel": "Padre Ángel",
    "padre ángel": "Padre Ángel",
    "angel": "Padre Ángel",
    "ángel": "Padre Ángel",
    "sister maria gracia": "Sister María Gracia",
    "sister maría gracia": "Sister María Gracia",
    "maria gracia": "Sister María Gracia",
    "maría gracia": "Sister María Gracia",
    "padre ignacio": "Padre Ignacio",
    "ignacio": "Padre Ignacio",
    # Mortal Vengeance II: To Reel or Not Too Real?
    "valeria": "Valeria Viccini",
    "valeria viccini": "Valeria Viccini",
    "camila": "Camila Álvarez",
    "camila alvarez": "Camila Álvarez",
    "camila álvarez": "Camila Álvarez",
    "rafa": "Rafael Montero",
    "rafael": "Rafael Montero",
    "rafa montero": "Rafael Montero",
    "rafael montero": "Rafael Montero",
    "shane": "Shane Harper",
    "shane harper": "Shane Harper",
}


def tokenize(text: str) -> set[str]:
    """
    Convert text into useful lowercase search tokens.
    """
    words = re.findall(r"[a-zA-ZÀ-ÿ0-9']+", text.lower())
    return {word for word in words if word not in STOPWORDS and len(word) > 2}


def normalize_lookup(text: str) -> str:
    """Normalize text for accent-insensitive character/source matching."""
    replacements = str.maketrans({
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n",
        "Á": "a", "É": "e", "Í": "i", "Ó": "o", "Ú": "u", "Ñ": "n",
    })
    text = (text or "").translate(replacements).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def extract_requested_line_values(topic: str, label: str) -> list[str]:
    """Pull comma-separated values from structured brief lines."""
    match = re.search(rf"^{re.escape(label)}:\s*(.+)$", topic or "", flags=re.MULTILINE | re.IGNORECASE)
    if not match:
        return []
    value = match.group(1).strip()
    if not value or value.lower() in {"not applicable", "not specified", "none"}:
        return []
    return [item.strip(" []'\"") for item in re.split(r",|\|", value) if item.strip(" []'\"")]


def requested_characters(topic: str) -> list[str]:
    """Return requested character names from structured fields and free text."""
    found = []
    for value in extract_requested_line_values(topic, "Requested character tags"):
        if value and value.lower() not in {"not applicable", "other / unnamed character"}:
            found.append(value)

    topic_lookup = f" {normalize_lookup(topic)} "
    for alias, character in KNOWN_CHARACTER_ALIASES.items():
        alias_lookup = f" {normalize_lookup(alias)} "
        if alias_lookup in topic_lookup and character not in found:
            found.append(character)

    return found


def requested_books(topic: str) -> list[str]:
    """Return selected/requested book-source values for canon grounding."""
    books = []
    for label in (
        "Related book/source",
        "Campaign related book/source",
        "Requested quote book/source",
        "Quote book/source",
        "Book/source",
    ):
        books.extend(extract_requested_line_values(topic, label))

    topic_lookup = normalize_lookup(topic)
    # Check the more specific titles first: "mortal vengeance ii" and
    # "...a grim tale" both contain the plain "mortal vengeance" substring.
    if "mortal vengeance ii" in topic_lookup and "Mortal Vengeance II: To Reel or Not Too Real?" not in books:
        books.append("Mortal Vengeance II: To Reel or Not Too Real?")
    elif "mortal vengeance a grim tale" in topic_lookup and "Mortal Vengeance: A Grim Tale" not in books:
        books.append("Mortal Vengeance: A Grim Tale")
    elif "mortal vengeance" in topic_lookup and not books:
        books.append("Mortal Vengeance")

    ignored = {"not applicable", "not specified", "none", "all books"}
    selected = []
    for book in books:
        if not book or book.lower() in ignored:
            continue
        if book not in selected:
            selected.append(book)
    return selected


def requested_moods(topic: str) -> list[str]:
    """Return requested quote mood/category tags so they can steer retrieval."""
    moods = []
    for label in (
        "Requested quote mood/category tags",
        "Quote mood/category tags",
        "Quote mood/category",
        "Quote moods",
    ):
        moods.extend(extract_requested_line_values(topic, label))

    ignored = {"not applicable", "not specified", "none"}
    selected = []
    for mood in moods:
        if not mood or mood.lower() in ignored:
            continue
        if mood not in selected:
            selected.append(mood)
    return selected


def load_private_manuscript_documents(book_names: list[str]) -> list[dict]:
    """Load selected private manuscripts for internal canon retrieval."""
    documents = []
    seen_paths = set()
    for book in book_names:
        for filename in BOOK_MANUSCRIPT_FILES.get(book, ()):
            path = PRIVATE_MANUSCRIPT_DIR / filename
            if not path.exists() or path in seen_paths:
                continue
            seen_paths.add(path)
            documents.append({
                "layer": "private_manuscript",
                "title": f"{book} manuscript",
                "book": book,
                "path": path,
                "content": path.read_text(encoding="utf-8", errors="ignore"),
            })
            break
    return documents


def markdown_field(entry: str, heading: str) -> str:
    """Extract a field body under a markdown ### heading."""
    match = re.search(
        rf"^###\s+{re.escape(heading)}\s*\n\n?(.*?)(?=\n###\s+|\n---|\Z)",
        entry,
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def field_values(entry: str, heading: str) -> list[str]:
    """Return comma/newline separated markdown field values."""
    field = markdown_field(entry, heading)
    return [item.strip(" -*") for item in re.split(r",|\n", field) if item.strip(" -*")]


def field_contains_any(field_items: list[str], requested_items: list[str]) -> bool:
    """Accent-insensitive exact-ish membership check."""
    field_lookup = {normalize_lookup(item) for item in field_items}
    return any(normalize_lookup(item) in field_lookup for item in requested_items)


def split_text_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Split markdown text into manageable chunks.
    Tries to preserve markdown sections when possible.
    """
    sections = re.split(r"(?=^#{1,4}\s+)", text, flags=re.MULTILINE)

    chunks = []

    for section in sections:
        section = section.strip()

        if not section:
            continue

        if len(section) <= max_chars:
            chunks.append(section)
        else:
            for start in range(0, len(section), max_chars):
                chunks.append(section[start:start + max_chars])

    return chunks


def split_quote_bank_entries(text: str) -> tuple[str, list[str]]:
    """
    Split the quote bank into its rules/preamble and individual quote entries.
    Generic markdown chunking can bury valid quotes in a very large file; quote
    posts need quote-level retrieval instead.
    """
    quote_headers = list(re.finditer(r"^##\s+Quote\s+[A-Z]+-\d+", text, flags=re.MULTILINE))

    if not quote_headers:
        return "", split_text_into_chunks(text)

    preamble = text[:quote_headers[0].start()].strip()
    entries = []

    for index, match in enumerate(quote_headers):
        start = match.start()
        end = quote_headers[index + 1].start() if index + 1 < len(quote_headers) else len(text)
        entry = text[start:end].strip()
        if entry:
            entries.append(entry)

    return preamble, entries


def quote_entry_score(
    entry: str,
    keywords: set[str],
    topic: str,
    requested_character_names: list[str] | None = None,
    requested_book_names: list[str] | None = None,
    requested_mood_names: list[str] | None = None,
    semantic_cosine: float = 0.0,
) -> int:
    """
    Score quote-bank entries with a few quote-specific boosts.
    The lightweight token overlap still does most of the work, but broad
    book-quote requests should prefer public-safe, reusable entries. Mood/category
    tags and an optional semantic-similarity signal steer selection toward quotes
    whose meaning fits the request, not just shared keywords.
    """
    entry_tokens = tokenize(entry)
    score = len(keywords.intersection(entry_tokens))
    topic_lower = topic.lower()
    entry_lower = entry.lower()
    if "[paste exact quote here.]" in entry_lower:
        return -999

    requested_character_names = requested_character_names or requested_characters(topic)
    requested_book_names = requested_book_names or requested_books(topic)
    if requested_mood_names is None:
        requested_mood_names = requested_moods(topic)
    entry_characters = field_values(entry, "Character Tags")
    entry_book = markdown_field(entry, "Book")

    # Mood/category tags are a first-class retrieval signal (not just keywords).
    if requested_mood_names:
        entry_mood_lookup = {normalize_lookup(mood) for mood in field_values(entry, "Mood Tags")}
        mood_hits = sum(1 for mood in requested_mood_names if normalize_lookup(mood) in entry_mood_lookup)
        if mood_hits:
            score += MOOD_TAG_MATCH_BOOST * mood_hits

    # Semantic similarity blended in (0 when embeddings are unavailable).
    if semantic_cosine:
        score += int(round(max(0.0, min(1.0, semantic_cosine)) * SEMANTIC_SCORE_WEIGHT))

    if requested_book_names:
        if any(normalize_lookup(book) == normalize_lookup(entry_book) for book in requested_book_names):
            score += 50
        else:
            score -= 90

    if requested_character_names:
        if field_contains_any(entry_characters, requested_character_names):
            score += 100
        else:
            # Character quote requests must not drift to a different character.
            # Keep a tiny possibility for manuscript fallback, but quote-bank
            # entries with the wrong character should fall far behind exact hits.
            score -= 100

    if "spoiler-free" in topic_lower or "avoid major spoilers" in topic_lower:
        if "spoiler-safe" in entry_lower:
            score += 6
        if "major-spoiler" in entry_lower:
            score -= 8

    if "instagram" in topic_lower and "instagram" in entry_lower:
        score += 4
    if "youtube" in topic_lower and "youtube" in entry_lower:
        score += 4
    if "linkedin" in topic_lower and "linkedin" in entry_lower:
        score += 4
    if "quote-graphic" in entry_lower or "quote graphic" in topic_lower:
        score += 2

    # Broad requests like "several book quotes" otherwise score nearly every
    # entry the same. Prefer reusable, public-facing quotes in that case.
    if not keywords or keywords.issubset({"book", "source", "several"}):
        if "spoiler-safe" in entry_lower:
            score += 5
        if "instagram" in entry_lower or "quote-graphic" in entry_lower:
            score += 2
        if "major-spoiler" in entry_lower:
            score -= 5

    return score


def requests_quote_bank_context(content_type: str, topic: str) -> bool:
    """
    True when any output type explicitly needs approved book quotes.
    """
    if content_type == "quote_post":
        return True

    topic_lower = topic.lower()
    return any(phrase in topic_lower for phrase in QUOTE_BANK_REQUEST_PHRASES)


def build_quote_post_candidates(document: dict, topic: str, keywords: set[str]) -> list[dict]:
    """
    Return quote-bank rules plus the most relevant individual quote entries.
    """
    preamble, entries = split_quote_bank_entries(document["content"])
    path = document["path"]
    candidates = []
    requested_character_names = requested_characters(topic)
    requested_book_names = requested_books(topic)
    requested_mood_names = requested_moods(topic)

    # Semantic pre-rank: embed the topic together with the mood/category tags so
    # the query reflects the intended emotional/psychological content, then blend
    # the cosine score into the keyword scorer. Returns {} (no effect) offline.
    semantic_query = " ".join(
        part for part in [topic, " ".join(requested_mood_names), " ".join(requested_character_names)] if part
    )
    semantic_scores = query_relevance(semantic_query, entries)

    if preamble:
        candidates.append({
            "layer": "primary",
            "title": "quote_bank",
            "path": path,
            "chunk_index": 0,
            "score": 999,
            "content": preamble,
        })

    scored_entries = []
    for index, entry in enumerate(entries, start=1):
        score = quote_entry_score(
            entry,
            keywords,
            topic,
            requested_character_names=requested_character_names,
            requested_book_names=requested_book_names,
            requested_mood_names=requested_mood_names,
            semantic_cosine=semantic_scores.get(quote_entry_id(entry), 0.0),
        )
        scored_entries.append({
            "layer": "primary",
            "title": "quote_bank",
            "path": path,
            "chunk_index": index,
            "score": score,
            "content": entry,
            "book": markdown_field(entry, "Book"),
        })

    exact_character_entries = [
        item for item in scored_entries
        if not requested_character_names
        or field_contains_any(field_values(item["content"], "Character Tags"), requested_character_names)
    ]
    useful_entries = [item for item in exact_character_entries if item["score"] > 0]
    if not useful_entries:
        useful_entries = [] if requested_character_names else [item for item in scored_entries if item["score"] > -900]

    ranked = sorted(useful_entries, key=lambda item: (item["score"], -item["chunk_index"]), reverse=True)

    if requested_book_names:
        selected_entries = ranked[:QUOTE_POST_ENTRY_CAP]
    else:
        # No specific book requested: fill the cap round-robin across distinct
        # books so later books are not starved by file order (BUG-QP-02).
        by_book: dict[str, list] = {}
        for item in ranked:
            by_book.setdefault(item.get("book") or "Unknown", []).append(item)
        selected_entries = []
        while len(selected_entries) < QUOTE_POST_ENTRY_CAP and any(by_book.values()):
            for book in list(by_book.keys()):
                bucket = by_book[book]
                if bucket:
                    selected_entries.append(bucket.pop(0))
                    if len(selected_entries) >= QUOTE_POST_ENTRY_CAP:
                        break

    selected_entries.sort(key=lambda item: item["chunk_index"])
    return candidates + selected_entries


def quote_candidate_matches_request(candidate: dict, topic: str) -> bool:
    """True when a quote-bank candidate matches requested character/book filters."""
    if candidate.get("chunk_index") == 0:
        return False
    content = candidate.get("content", "")
    character_names = requested_characters(topic)
    book_names = requested_books(topic)

    if character_names and not field_contains_any(field_values(content, "Character Tags"), character_names):
        return False
    if book_names:
        entry_book = markdown_field(content, "Book")
        if not any(normalize_lookup(book) == normalize_lookup(entry_book) for book in book_names):
            return False
    return True


def build_manuscript_quote_candidates(document: dict, topic: str, keywords: set[str]) -> list[dict]:
    """
    Return manuscript chunks as fallback quote discovery candidates.

    These are not pre-approved quote-bank entries; the generation prompt is told
    to label them as manuscript candidates and preserve exact text only.
    """
    requested_character_names = requested_characters(topic)
    requested_book_names = requested_books(topic)
    chunks = split_text_into_chunks(document["content"], max_chars=2400)

    # Semantic pre-rank over the manuscript passages themselves: this is what makes
    # quote discovery a true RAG over the novel text rather than keyword matching.
    # Returns {} (no effect) when embeddings are unavailable, so keyword retrieval
    # still works offline.
    semantic_query = " ".join(
        part for part in [topic, " ".join(requested_character_names)] if part
    )
    semantic_scores = manuscript_query_relevance(semantic_query, chunks)

    scored = []

    for index, chunk in enumerate(chunks, start=1):
        chunk_lookup = normalize_lookup(chunk)
        score = len(keywords.intersection(tokenize(chunk)))

        cosine = semantic_scores.get(manuscript_chunk_id(chunk), 0.0)
        if cosine:
            score += int(round(max(0.0, min(1.0, cosine)) * MANUSCRIPT_SEMANTIC_WEIGHT))

        if requested_character_names:
            if any(normalize_lookup(character) in chunk_lookup for character in requested_character_names):
                score += 80
            else:
                score -= 15

        if requested_book_names:
            if any(normalize_lookup(book) in chunk_lookup for book in requested_book_names):
                score += 20

        if '"' in chunk or "“" in chunk or "”" in chunk:
            score += 15

        if score > 0:
            scored.append({
                "layer": document.get("layer", "primary"),
                "title": f"{document['title']}_manuscript_quote_candidates",
                "path": document["path"],
                "chunk_index": index,
                "score": score,
                "content": (
                    "MANUSCRIPT FALLBACK CANDIDATE: Use only exact quoted lines or exact prose excerpts "
                    "from this chunk. If speaker/source is uncertain, label it as manuscript candidate and "
                    "do not claim quote-bank approval.\n\n"
                    f"{chunk}"
                ),
            })

    return sorted(scored, key=lambda item: item["score"], reverse=True)[:QUOTE_POST_MANUSCRIPT_CAP]


def build_private_manuscript_context_candidates(document: dict, topic: str, keywords: set[str]) -> list[dict]:
    """
    Return selected private manuscript chunks for any content type.

    These chunks are for internal canon grounding and excerpt discovery. The
    filter prompt decides how much to preserve for final generation.
    """
    requested_character_names = requested_characters(topic)
    topic_lookup = normalize_lookup(topic)
    chunks = split_text_into_chunks(document["content"], max_chars=2600)

    # Semantic pre-rank over manuscript passages so canon grounding, teasers, and
    # excerpt discovery retrieve the passages that actually match the request's
    # meaning. Empty (no effect) when embeddings are unavailable.
    semantic_query = " ".join(
        part for part in [topic, " ".join(requested_character_names)] if part
    )
    semantic_scores = manuscript_query_relevance(semantic_query, chunks)

    scored = []

    manuscript_intent_terms = {
        "chapter", "chapters", "title", "titles", "teaser", "summary", "summarize",
        "promo", "promotional", "excerpt", "excerpts", "scene", "quote", "quotes",
        "pull", "synopsis", "canon", "manuscript",
    }

    for index, chunk in enumerate(chunks, start=1):
        chunk_lookup = normalize_lookup(chunk)
        chunk_tokens = tokenize(chunk)
        score = len(keywords.intersection(chunk_tokens))

        cosine = semantic_scores.get(manuscript_chunk_id(chunk), 0.0)
        if cosine:
            score += int(round(max(0.0, min(1.0, cosine)) * MANUSCRIPT_SEMANTIC_WEIGHT))

        if requested_character_names:
            if any(normalize_lookup(character) in chunk_lookup for character in requested_character_names):
                score += 80
            else:
                score -= 8

        if any(term in topic_lookup for term in manuscript_intent_terms):
            score += 10

        if re.search(r"^#{1,4}\s+", chunk, flags=re.MULTILINE):
            score += 8
        if '"' in chunk or "“" in chunk or "”" in chunk:
            score += 5

        if score > 0:
            scored.append({
                "layer": document["layer"],
                "title": document["title"],
                "path": document["path"],
                "chunk_index": index,
                "score": score,
                "content": (
                    f"PRIVATE MANUSCRIPT CONTEXT for {document['book']}.\n"
                    "Use this as internal canon grounding for summaries, teasers, chapter references, "
                    "quote discovery, and exact excerpts. Preserve exact wording only when quoting or "
                    "pulling an excerpt; otherwise summarize. Do not expose long private manuscript "
                    "passages unless the user specifically requested excerpts.\n\n"
                    f"{chunk}"
                ),
            })

    selected = sorted(scored, key=lambda item: item["score"], reverse=True)[:MANUSCRIPT_CONTEXT_CAP]
    if not selected and chunks:
        selected = [{
            "layer": document["layer"],
            "title": document["title"],
            "path": document["path"],
            "chunk_index": 1,
            "score": 1,
            "content": (
                f"PRIVATE MANUSCRIPT CONTEXT for {document['book']}.\n"
                "Use this only as internal canon grounding. Summarize rather than reproducing long passages.\n\n"
                f"{chunks[0][:MANUSCRIPT_SOURCE_CHAR_CAP]}"
            ),
        }]

    selected.sort(key=lambda item: item["chunk_index"])
    return selected


def build_candidate_chunks(content_type: str, topic: str) -> list[dict]:
    """
    Load the knowledge base, split documents into chunks, and rank them
    using lightweight keyword matching before the LLM filtering step.
    """
    knowledge_base = load_knowledge_base()
    private_manuscripts = load_private_manuscript_documents(requested_books(topic))

    keywords = tokenize(topic + " " + content_type.replace("_", " "))
    needs_quote_bank = requests_quote_bank_context(content_type, topic)
    only_quote_bank = "use only quote bank" in topic.lower()

    if content_type == "quote_post":
        quote_candidates = []
        manuscript_candidates = []

        for layer_name, documents in knowledge_base.items():
            for document in documents:
                title = document["title"]
                if title == "quote_bank":
                    quote_candidates.extend(build_quote_post_candidates(document, topic, keywords))
                    continue

                if only_quote_bank:
                    continue

                title_lookup = normalize_lookup(title)
                path_lookup = normalize_lookup(str(document["path"]))
                if "manuscript" in title_lookup or "manuscript" in path_lookup:
                    manuscript_candidates.extend(build_manuscript_quote_candidates(document, topic, keywords))

        approved_quote_entries = [item for item in quote_candidates if quote_candidate_matches_request(item, topic)]
        if len(approved_quote_entries) >= MIN_QUOTE_OPTIONS or only_quote_bank:
            return quote_candidates

        for document in private_manuscripts:
            manuscript_candidates.extend(build_manuscript_quote_candidates(document, topic, keywords))

        return quote_candidates + manuscript_candidates

    candidates = []

    for layer_name, documents in knowledge_base.items():
        for document in documents:
            title = document["title"]
            if only_quote_bank and title != "quote_bank":
                continue

            path = document["path"]

            if title == "quote_bank" and needs_quote_bank:
                quote_candidates = build_quote_post_candidates(document, topic, keywords)
                if content_type == "quote_post" or only_quote_bank:
                    return quote_candidates
                candidates.extend(quote_candidates)
                continue

            chunks = split_text_into_chunks(document["content"])

            for index, chunk in enumerate(chunks, start=1):
                chunk_tokens = tokenize(chunk)
                # Weight by source authority so primary (authoritative) context ranks
                # ahead of secondary (reference) context for equal keyword relevance.
                score = len(keywords.intersection(chunk_tokens)) * layer_weight(layer_name)

                candidates.append({
                    "layer": layer_name,
                    "title": title,
                    "path": path,
                    "chunk_index": index,
                    "score": score,
                    "content": chunk
                })

    if not only_quote_bank:
        for document in private_manuscripts:
            candidates.extend(build_private_manuscript_context_candidates(document, topic, keywords))

    if content_type in COMPREHENSIVE_CONTENT_TYPES:
        return select_comprehensive_chunks(candidates)

    ranked_candidates = sorted(
        candidates,
        key=lambda item: item["score"],
        reverse=True
    )

    useful_candidates = [item for item in ranked_candidates if item["score"] > 0]

    if not useful_candidates:
        useful_candidates = ranked_candidates[:TOP_CHUNKS]

    return useful_candidates[:TOP_CHUNKS]


def select_comprehensive_chunks(candidates: list[dict]) -> list[dict]:
    """
    Represent every primary/secondary source for long-form outputs. Each source
    contributes up to a character budget (small docs in full; large docs keep
    their most relevant chunks), and the quote bank is capped to top chunks.
    Source order is preserved so the filter sees each document coherently.
    """
    by_source: dict[tuple, list[dict]] = {}
    for item in candidates:
        by_source.setdefault((item["layer"], item["title"]), []).append(item)

    selected = []
    for (_layer, title), chunks in by_source.items():
        if title == "quote_bank":
            selected.extend(sorted(chunks, key=lambda c: c["score"], reverse=True)[:QUOTE_BANK_CAP])
            continue

        total_chars = sum(len(c["content"]) for c in chunks)
        if total_chars <= PER_SOURCE_CHAR_CAP:
            selected.extend(chunks)
            continue

        # Large source: keep highest-scoring chunks until the per-source budget is hit.
        budget = 0
        for chunk in sorted(chunks, key=lambda c: c["score"], reverse=True):
            if budget + len(chunk["content"]) > PER_SOURCE_CHAR_CAP and budget:
                break
            selected.append(chunk)
            budget += len(chunk["content"])

    selected.sort(key=lambda item: (item["layer"], item["title"], item["chunk_index"]))
    return selected[:COMPREHENSIVE_CAP]


def format_candidate_chunks(candidates: list[dict]) -> str:
    """
    Format candidate chunks for the relevance-filtering prompt.
    """
    formatted_chunks = []

    for candidate in candidates:
        header = (
            f"## Source: {candidate['title']} "
            f"| Layer: {candidate['layer']} "
            f"| Chunk: {candidate['chunk_index']}"
        )

        formatted_chunks.append(f"{header}\n\n{candidate['content']}")

    return "\n\n---\n\n".join(formatted_chunks)


def build_relevance_filter_prompt(content_type: str, topic: str, candidate_context: str) -> str:
    """
    Build the first-stage LLM prompt.
    This prompt filters the knowledge base before final generation.
    """
    comprehensive_note = ""
    if content_type in COMPREHENSIVE_CONTENT_TYPES:
        comprehensive_note = """
This is a long-form / editorial output. Be COMPREHENSIVE, not minimal.

Draw from and combine ALL relevant primary sources into one rich brief:
- Author interviews (label exact quotes as author interview quotes).
- Awards and recognitions (only those present in the knowledge base).
- Reader reviews and critic/editorial reviews (keep exact wording and source attribution).
- Book themes, premise, and book/series info.
- Brand voice and positioning.
- Approved pull quotes from the quote bank.

Pull at least something from every source that has relevant material. Do not collapse to a single source. Keep the brief detailed enough that the writer can weave interviews, reviews, awards, themes, and quotes together.
"""

    return f"""
You are the context filtering layer for Tell Tales Ink.

Your task is NOT to write the final content.

Your task is to select and summarize the context that is relevant to the user's request.
{comprehensive_note}
Content type:
{content_type}

User request:
{topic}

Candidate knowledge base context:
{candidate_context}

Instructions:

1. Keep information relevant to the request.
2. Preserve exact real review quotes if they are relevant.
3. Preserve exact book quotes if they are relevant.
4. Keep source names when using review excerpts.
5. Preserve exact author interview quotes if they are relevant, and label them as author interview quotes.
6. Identify any spoiler restrictions.
7. Identify any factual or canon constraints.
8. Do not invent facts, reviews, quotes, awards, ratings, or book details.
9. Do not generate the final marketing content.
10. If no relevant quote or review exists, say so clearly.
11. For quote_post requests, preserve at least 5 quote options when the candidate context contains 5 or more relevant options.
12. For character quote_post requests, exact Character Tags are mandatory for quote-bank entries. Do not keep a quote-bank entry for Lieutenant Ricardo García when the user requested Mónika Torres, or any other mismatched character.
13. For quote_post requests, prefer approved Quote Bank entries. Use MANUSCRIPT FALLBACK CANDIDATE chunks only when fewer than 5 approved quote-bank matches are available.
14. Manuscript fallback quotes must be exact lines or exact prose excerpts from the manuscript chunk. Label them as manuscript candidates, not approved quote-bank quotes, unless they also appear in the quote bank.
15. If fewer than 5 total matching quotes/candidates are available, say exactly how many are available and what is missing.
16. PRIVATE MANUSCRIPT CONTEXT is available only for the selected/requested book. Use it for canon grounding, chapter titles, chapter-specific summaries, teasers, promos, exact quote discovery, and pull excerpts.
17. Do not reproduce long private manuscript passages unless the user explicitly requested excerpts. For summaries and promotional copy, synthesize the scene/chapter faithfully without revealing excessive manuscript text.
18. If a specific related book/source is selected, do not mix manuscript details from another book.
19. For quote_post requests with mood/category tags, prefer quotes whose Mood Tags match the requested mood and whose meaning genuinely fits it. Interpret the mood semantically: e.g. "Mental Health" means internal emotional experience, dissociation, self-blame, intrusive thoughts, or fear of abandonment — not merely a physical reaction adjacent to it. Rank mood-matching quotes ahead of off-mood ones.

Return the filtered context using this structure:

# Filtered Context

## Relevant Brand / Book Context

## Relevant Quotes or Review Excerpts

## Relevant Audience / Strategy Notes

## Spoiler or Accuracy Constraints

## Do Not Use / Avoid
"""


def select_relevant_context(content_type: str, topic: str) -> str:
    """
    First LLM call:
    Filter the knowledge base into a smaller, relevant context brief.
    """
    candidates = build_candidate_chunks(content_type, topic)
    candidate_context = format_candidate_chunks(candidates)

    relevance_prompt = build_relevance_filter_prompt(
        content_type=content_type,
        topic=topic,
        candidate_context=candidate_context
    )

    filtered_context = generate_text(relevance_prompt)

    return filtered_context


if __name__ == "__main__":
    test_context = select_relevant_context(
        content_type="instagram_caption",
        topic="Create a launch caption for Mortal Vengeance II."
    )

    print(test_context)
