from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Any

from backend.models import CampaignGenerationRequest, ChapterPromoRequest, ContentGenerationRequest, Draft
from backend.services import draft_service

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import chapter_promo_prompts as cpp  # noqa: E402
import chapter_reader  # noqa: E402
from context_filter import requested_characters  # noqa: E402
from content_pipeline import run_pipeline  # noqa: E402
from frontend_adapters import normalize_multi_select, selected_values  # noqa: E402
from image_generation import generate_external_visual, new_visual_output_dir, slugify_filename  # noqa: E402
from llm_integration import generate_text  # noqa: E402
from selection_options import CHARACTER_TAGS  # noqa: E402

SOURCE_EVIDENCE_TERMS = (
    "quote", "quotes", "dialogue", "dialog", "line", "lines", "funniest", "savage",
    "friendship", "friend", "moment", "moments", "top 10", "top 20", "listicle",
    "rank", "ranked", "ranking", "best character", "best characters", "favorite character",
    "favorite characters", "favourite character", "favourite characters",
)
SOURCE_EVIDENCE_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "that", "this", "when", "what",
    "which", "about", "blog", "article", "listicle", "times", "top", "best",
    "mortal", "vengeance", "grim", "tale", "characters", "character",
}


def _tokens(value: str) -> set[str]:
    words = re.findall(r"[a-zA-ZÀ-ÿ0-9']+", (value or "").lower())
    return {word for word in words if len(word) > 2 and word not in SOURCE_EVIDENCE_STOPWORDS}


def _expanded_focus_terms(topic: str, focus: str) -> set[str]:
    lookup = f"{topic} {focus}".lower()
    terms = _tokens(lookup)
    if "savage" in lookup:
        terms.update({"sharp", "retort", "smirk", "grin", "snapped", "dry", "challenge", "argument", "debate", "withering"})
    if "funny" in lookup or "funniest" in lookup:
        terms.update({"laugh", "smirk", "grin", "joke", "deadpan", "dry", "sarcasm", "teasing", "banter"})
    if "friend" in lookup or "friendship" in lookup:
        terms.update({"friend", "friends", "together", "help", "helped", "protect", "trust", "team", "smile", "support"})
    if "dialogue" in lookup or "line" in lookup or "quote" in lookup:
        terms.update({"said", "asked", "replied", "murmured", "called", "answered"})
    if "character" in lookup or "characters" in lookup:
        terms.update({"said", "asked", "looked", "turned", "felt", "thought", "wanted", "needed", "decided", "remembered"})
    if "best" in lookup or "rank" in lookup or "favorite" in lookup or "favourite" in lookup:
        terms.update({"choice", "decision", "changed", "saved", "protected", "confronted", "revealed", "refused", "risked", "truth"})
    return terms


def _candidate_blocks(text: str) -> list[str]:
    blocks = []
    for block in re.split(r"\n\s*\n", text or ""):
        cleaned = re.sub(r"\s+", " ", block).strip()
        if 40 <= len(cleaned) <= 900:
            blocks.append(cleaned)
    return blocks


def _character_hints(block: str, requested: list[str]) -> list[str]:
    hints = [name for name in requested if name.lower().split()[0] in block.lower()]
    for name in CHARACTER_TAGS:
        name_lookup = name.lower()
        first_name = name_lookup.split()[0]
        if (name_lookup in block.lower() or first_name in block.lower()) and name not in hints:
            hints.append(name)
    return hints[:5]


def _source_evidence_items_for_book(*, topic: str, source_focus: str, selected_book: str, knowledge_sources: list[str]) -> list[tuple[int, str, str, list[str], str]]:
    chapters = chapter_reader.parse_chapters(selected_book)
    query = f"{topic} {source_focus} {' '.join(selected_values(knowledge_sources or []))}"
    terms = _expanded_focus_terms(topic, source_focus)
    requested_names = requested_characters(query)
    scored: list[tuple[int, str, str, list[str], str]] = []
    for chapter in chapters:
        chapter_label = str(chapter.get("label") or chapter.get("title") or "Chapter")
        for block in _candidate_blocks(str(chapter.get("text") or "")):
            block_lookup = block.lower()
            score = sum(3 for term in terms if term in block_lookup)
            if any(mark in block for mark in ('"', "“", "”", "'")):
                score += 8
            hints = _character_hints(block, requested_names)
            if requested_names and hints:
                score += 20
            if "alex" in query.lower() and "alex" in block_lookup:
                score += 16
            if score > 0:
                scored.append((score, chapter_label, block, hints, selected_book))
    return scored


def source_evidence_pack_for(*, topic: str, source_focus: str = "", selected_book: str = "", selected_books: list[str] | None = None, knowledge_sources: list[str] | None = None) -> str:
    books = selected_values(selected_books or [])
    if selected_book and selected_book not in books:
        books.insert(0, selected_book)
    books = [book for index, book in enumerate(books) if book and book not in books[:index]]
    if not books:
        return ""

    scored: list[tuple[int, str, str, list[str], str]] = []
    for book in books:
        scored.extend(_source_evidence_items_for_book(topic=topic, source_focus=source_focus, selected_book=book, knowledge_sources=knowledge_sources or []))

    if not scored:
        return ""
    count_match = re.search(r"\btop\s+(\d+)\b", topic.lower())
    requested_count = int(count_match.group(1)) if count_match else 10
    limit = min(max(requested_count + 8, 16), 32)
    selected = sorted(scored, key=lambda item: item[0], reverse=True)[:limit]
    evidence = [
        "Verified source evidence from the selected book:",
        f"Selected book/source: {', '.join(books)}",
        "Use only these selected-book excerpts for ranked quote, dialogue, character, savage, or friendship items. Do not use another book. Do not invent phrases, characters, or explanations.",
    ]
    for index, (_score, chapter_label, block, hints, book) in enumerate(selected, start=1):
        hint_text = f" Character hints: {', '.join(hints)}." if hints else ""
        evidence.append(f"{index}. [{book} - {chapter_label}]{hint_text} Excerpt: \"\"\"{block}\"\"\"")
    return "\n".join(evidence)


def _source_evidence_pack(request: ContentGenerationRequest) -> str:
    return source_evidence_pack_for(
        topic=request.topic,
        source_focus=request.source_focus,
        selected_book=request.related_book,
        selected_books=request.related_books,
        knowledge_sources=request.knowledge_sources,
    )


def _selected_books(request: ContentGenerationRequest) -> list[str]:
    books = selected_values(request.related_books)
    if request.related_book and request.related_book not in books:
        books.insert(0, request.related_book)
    return [book for index, book in enumerate(books) if book and book not in books[:index]]


def _requested_top_count(topic: str, fallback: int = 10) -> int:
    count_match = re.search(r"\btop\s+(\d+)\b", topic.lower())
    return int(count_match.group(1)) if count_match else fallback


def _needs_character_ranking(request: ContentGenerationRequest) -> bool:
    lookup = f"{request.topic} {request.blog_format} {request.source_focus}".lower()
    return (
        request.content_type == "blog_post"
        and ("character" in lookup or "characters" in lookup)
        and any(term in lookup for term in ("best", "rank", "ranking", "ranked", "top", "favorite", "favourite"))
    )


def _character_ranking_rubric() -> str:
    return "\n".join(
        [
            "Character ranking rubric:",
            "Before the ranked list, include a concise Ranking Criteria section.",
            "Rank characters by a consistent rubric, not by arbitrary preference or generic praise.",
            "Use these criteria across the whole list: narrative impact, agency and decision-making, relationship significance, thematic relevance, memorability of voice, and available source support.",
            "For each ranked character, explain why that character belongs at that exact rank using the same criteria.",
            "For each placement, include a visible Book evidence line with a short source excerpt or concrete manuscript moment from the selected book/source, plus the relevant chapter or source note when available.",
            "When two characters are close, explain why the higher-ranked character edges out the lower-ranked one, or why the lower placement is limited by the available evidence.",
            "Do not rank characters without source support. If the evidence is thin, label the placement as provisional instead of presenting it as definitive.",
            "Attribute actions, quotes, and traits only to the character supported by the retrieved evidence.",
            "Mandatory ranked-entry format: Rank + character name; Why this rank; Book evidence; Criteria notes.",
            "If there is no Book evidence for a character, do not include that character in the ranked list.",
        ]
    )


def _character_name_matches_block(name: str, block_lookup: str) -> bool:
    name_lookup = name.lower()
    parts = [part for part in re.findall(r"[a-zA-ZÀ-ÿ']+", name_lookup) if len(part) > 2]
    if name_lookup in block_lookup:
        return True
    if not parts:
        return False
    if name_lookup.startswith("the "):
        return all(part in block_lookup for part in parts if part != "the")
    return parts[0] in block_lookup


def _character_evidence_pack(request: ContentGenerationRequest) -> str:
    books = _selected_books(request)
    if not books:
        return ""

    focus_terms = _expanded_focus_terms(request.topic, request.source_focus)
    per_character: dict[str, list[tuple[int, str, str, str]]] = {}
    for book in books:
        for chapter in chapter_reader.parse_chapters(book):
            chapter_label = str(chapter.get("label") or chapter.get("title") or "Chapter")
            for block in _candidate_blocks(str(chapter.get("text") or "")):
                block_lookup = block.lower()
                for character in CHARACTER_TAGS:
                    if not _character_name_matches_block(character, block_lookup):
                        continue
                    score = 20 + sum(3 for term in focus_terms if term in block_lookup)
                    if any(mark in block for mark in ('"', "“", "”", "'")):
                        score += 5
                    per_character.setdefault(character, []).append((score, book, chapter_label, block))

    if not per_character:
        return ""

    character_scores = sorted(
        (
            (sum(item[0] for item in entries), character, sorted(entries, key=lambda item: item[0], reverse=True)[:2])
            for character, entries in per_character.items()
        ),
        reverse=True,
    )
    limit = min(max(_requested_top_count(request.topic), 10), 16)
    selected = character_scores[:limit]
    lines = [
        "Character evidence map from the selected book/source:",
        f"Selected book/source: {', '.join(books)}",
        "Use these character-specific excerpts to justify rankings. Every ranked character must cite one of these evidence entries or be omitted.",
    ]
    for _score, character, entries in selected:
        lines.append(f"- {character}:")
        for _entry_score, book, chapter_label, block in entries:
            lines.append(f"  Evidence [{book} - {chapter_label}]: \"\"\"{block}\"\"\"")
    return "\n".join(lines)


def _suggested_blog_structure(request: ContentGenerationRequest, needs_evidence: bool) -> str:
    format_name = (request.blog_format or "article").lower()
    top_count = _requested_top_count(request.topic)
    if _needs_character_ranking(request):
        return "\n".join(
            [
                "Suggested article structure to use:",
                "1. SEO-ready title that names the selected book/source and the character ranking promise.",
                "2. Short opening paragraph that defines the ranking lens and spoiler boundary.",
                "3. Ranking Criteria section naming the criteria used for every placement.",
                f"4. Ranked list of up to {top_count} characters, each with a scannable heading.",
                "5. For each ranked character: rank rationale, Book evidence line, correct attribution, chapter/source note when available, and a clear reason for that exact placement.",
                "6. Closing takeaway that identifies what the ranking reveals about the selected book/source.",
                "7. Soft reader CTA only if the content type calls for one.",
            ]
        )
    if "listicle" in format_name or re.search(r"\btop\s+\d+\b", request.topic.lower()):
        evidence_rule = (
            "For each ranked item: exact quote/excerpt or clearly identified verified moment, correct character attribution, chapter/source note, and a short explanation tied directly to the excerpt."
            if needs_evidence
            else "For each ranked item: clear point, specific context, and a short explanation."
        )
        return "\n".join(
            [
                "Suggested article structure to use:",
                "1. SEO-ready title that names the book/source and the ranking promise.",
                "2. Short opening paragraph explaining the lens for the ranking and any spoiler boundary.",
                "3. One-sentence methodology/source note explaining that selections come from the selected book/source material.",
                f"4. Ranked list of up to {top_count} items, each with a scannable heading.",
                f"5. {evidence_rule}",
                "6. Closing takeaway that identifies the larger pattern across the selected moments.",
                "7. Soft reader CTA only if the content type calls for one.",
            ]
        )
    if "character" in format_name:
        return "\n".join(
            [
                "Suggested article structure to use:",
                "1. Title naming the character and the central insight.",
                "2. Opening thesis about what the character reveals in the story.",
                "3. Character role and relationship map grounded in the source material.",
                "4. Key scenes or moments, each tied to a concrete excerpt or canon detail.",
                "5. Reader-facing interpretation: why the character matters emotionally or thematically.",
                "6. Closing takeaway with a soft CTA if appropriate.",
            ]
        )
    if "review" in format_name:
        return "\n".join(
            [
                "Suggested article structure to use:",
                "1. Title focused on reader/reviewer response.",
                "2. Opening summary of the reception angle.",
                "3. Group reviews by theme, not randomly.",
                "4. Use exact review quotes only from review knowledge-base context.",
                "5. Connect review patterns to book positioning and reader fit.",
                "6. Close with a concise recommendation-style takeaway.",
            ]
        )
    if "q&a" in format_name or "author" in format_name:
        return "\n".join(
            [
                "Suggested article structure to use:",
                "1. Intro framing the author/story angle.",
                "2. 6-8 thoughtful Q&A sections ordered from accessible to deeper craft/theme questions.",
                "3. Each answer should include specific story, character, or process detail from context.",
                "4. End with where the book fits in the larger Mortal Vengeance world.",
            ]
        )
    return "\n".join(
        [
            "Suggested article structure to use:",
            "1. Strong title and opening hook.",
            "2. Context paragraph that names the selected book/source and reader promise.",
            "3. Main sections ordered from broad idea to specific source-backed support.",
            "4. Evidence from knowledge base or manuscript where selected.",
            "5. Reader takeaway section that explains why the topic matters.",
            "6. Soft CTA only if appropriate.",
        ]
    )


def content_brief(request: ContentGenerationRequest) -> str:
    topic_lookup = request.topic.lower()
    needs_exact_quotes = request.content_type == "blog_post" and any(
        term in topic_lookup
        for term in SOURCE_EVIDENCE_TERMS
    )
    needs_character_ranking = _needs_character_ranking(request)
    structure_values = selected_values(request.structure)
    should_suggest_structure = request.content_type == "blog_post" and any(
        value.strip().lower() == "suggest structure" for value in structure_values
    )
    lines = [
        request.topic.strip(),
        f"Related book/source: {', '.join(selected_values(request.related_books)) or request.related_book or 'Not specified'}",
        f"Platform: {request.platform or 'Not specified'}",
        f"Audience: {', '.join(selected_values(request.audience)) or 'Not specified'}",
        f"Objectives: {', '.join(selected_values(request.objectives)) or 'Not specified'}",
        f"Constraints: {', '.join(selected_values(request.constraints)) or 'Not specified'}",
    ]
    if request.content_type == "blog_post":
        lines.extend(
            [
                f"Blog format: {request.blog_format or 'Not specified'}",
                f"Blog length: {request.blog_length or request.word_count or request.length_preference or 'Not specified'}",
                f"Structure checklist: {', '.join(selected_values(request.structure)) or 'Not specified'}",
                f"SEO keyword: {request.seo_keyword or 'Not specified'}",
                f"Heading depth: {request.heading_depth or 'Not specified'}",
                f"Metadata controls: {request.metadata_controls or 'Not specified'}",
                f"Knowledge sources to use: {', '.join(selected_values(request.knowledge_sources)) or 'Not specified'}",
                f"Source focus: {request.source_focus or 'Not specified'}",
                f"Image generation: {request.image_generation or 'Not specified'}",
                f"Supporting image: {request.supporting_image or 'Not specified'}",
                f"Visual style: {request.visual_style or 'Not specified'}",
                f"Requested image formats: {', '.join(selected_values(request.visual_formats)) or 'Not specified'}",
            ]
        )
        if should_suggest_structure:
            lines.append(_suggested_blog_structure(request, needs_exact_quotes))
        if needs_character_ranking:
            lines.append(_character_ranking_rubric())
            character_evidence_pack = _character_evidence_pack(request)
            if character_evidence_pack:
                lines.append(character_evidence_pack)
        if needs_exact_quotes:
            lines.extend(
                [
                    "Quote grounding: this request requires exact source quotes or exact dialogue lines from the selected book/manuscript context.",
                    "Do not invent, paraphrase, modernize, or approximate dialogue. If there are fewer exact source lines available than requested, say so and use only verified lines.",
                    "For a ranked quote/dialogue/moment listicle, every numbered item must include one exact quoted line or clearly identified manuscript-grounded moment and a brief explanation grounded in the retrieved context.",
                    "When Character info is enabled, use retrieved character facts to identify who says or does each item.",
                    "When Reviews is enabled, use review context only for reception/positioning, not as invented book dialogue.",
                    "When Pull quotes is enabled, prefer approved quote-bank lines before manuscript fallback lines.",
                ]
            )
            evidence_pack = _source_evidence_pack(request)
            if evidence_pack:
                lines.append(evidence_pack)
    if request.content_type == "press_release":
        contact_parts = [
            request.media_contact_name,
            request.media_contact_email,
            request.media_contact_phone,
            request.media_contact_website,
        ]
        lines.extend(
            [
                "Format: professional press release.",
                "Do not frame this as a social promo. Do not use tone, spoiler level, hashtag, or call-to-action fields.",
                f"Release timing: {request.press_timing or 'FOR IMMEDIATE RELEASE'}",
                f"Publication date: {request.publication_date or 'Not specified'}",
                f"Byline city: {request.byline_city or 'Not specified'}",
                f"News angle / hook: {'Auto-generate the strongest media hook from the announcement.' if request.auto_news_angle else request.news_angle or 'Not specified'}",
                f"Boilerplate: {'Auto-generate an appropriate author/book boilerplate.' if request.auto_boilerplate else request.boilerplate or 'Not specified'}",
                f"Media contact: {request.media_contact or ' | '.join(part for part in contact_parts if part) or 'Not specified'}",
                f"Companion press assets: {', '.join(selected_values(request.companion_assets)) or 'Not specified'}",
            ]
        )
    else:
        lines.append(f"CTA: {request.cta or 'Not specified'}")
    return "\n".join(lines)


def _visuals_requested(request: ContentGenerationRequest) -> bool:
    mode = (request.image_generation or "").strip().lower()
    if mode in {"do not generate visuals", "no visuals", "none"}:
        return False
    if mode in {"create prompt only", "prompt only"}:
        return False
    return mode == "generate visuals" or bool(request.visual_formats)


def _default_visual_formats(request: ContentGenerationRequest) -> list[str]:
    formats = selected_values(request.visual_formats)
    if formats:
        return formats
    if request.content_type == "blog_post":
        return ["blog cover 16:9"]
    if request.content_type == "instagram_caption":
        return ["Instagram Post 4:5"]
    if request.content_type == "youtube_content":
        return ["YouTube thumbnail 16:9"]
    if request.content_type == "linkedin_content":
        return ["LinkedIn post horizontal 1.91:1"]
    return ["square 1:1"]


def _generate_visual_artifacts(request: ContentGenerationRequest, generated_content: str) -> dict[str, Any]:
    if not _visuals_requested(request):
        return {"paths": [], "errors": []}
    output_dir = new_visual_output_dir("generated_visuals")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    errors: list[dict[str, str]] = []
    source_books = ", ".join(selected_values(request.related_books)) or request.related_book
    for format_label in _default_visual_formats(request):
        result = generate_external_visual(
            use_case=f"{request.content_type} visual",
            content=generated_content[:1200],
            topic=request.topic,
            style=request.visual_style,
            attribution=source_books,
            format_label=format_label,
            output_dir=output_dir,
            file_stem=f"{request.content_type}_{request.topic}_{format_label}",
            content_kind=request.content_type,
        )
        if result.get("path"):
            paths.append(str(result["path"]))
        elif result.get("error"):
            errors.append({"format": format_label, "error": str(result["error"])})
    return {"paths": paths, "errors": errors, "output_dir": str(output_dir)}


async def generate_content(request: ContentGenerationRequest) -> dict[str, Any]:
    result = await asyncio.to_thread(run_pipeline, request.content_type, content_brief(request))
    visuals = await asyncio.to_thread(_generate_visual_artifacts, request, result["generated_content"])
    draft = draft_service.create_draft(
        title=request.topic.strip()[:90],
        content_type=request.content_type,
        content=result["generated_content"],
        source_path=result["draft_path"],
        metadata=request.model_dump() if hasattr(request, "model_dump") else request.dict(),
    )
    return {
        "content": result["generated_content"],
        "draft": draft,
        "artifacts": {
            "draft_path": str(result["draft_path"]),
            "prompt_path": str(result["prompt_path"]),
            "filtered_context_path": str(result["filtered_context_path"]),
            "visual_paths": visuals["paths"],
            "visual_errors": visuals["errors"],
            "visual_output_dir": visuals.get("output_dir", ""),
        },
    }


async def generate_campaign(request: CampaignGenerationRequest) -> dict[str, Any]:
    content_types = selected_values(request.content_types)
    items = []
    for content_type in content_types:
        generated = await generate_content(
            ContentGenerationRequest(
                content_type=content_type,
                topic=request.topic,
                related_book=request.related_book,
                platform=request.platform,
                audience=request.audience,
                objectives=request.objectives,
                constraints=request.constraints,
                cta=request.cta,
            )
        )
        items.append({"content_type": content_type, "draft": generated["draft"], "content": generated["content"]})
    bundle = "\n\n---\n\n".join(
        f"## {item['content_type'].replace('_', ' ').title()}\n\n{item['content']}"
        for item in items
    )
    draft = draft_service.create_draft(
        title=f"{request.topic.strip()[:70]} campaign",
        content_type="campaign",
        content=bundle,
        metadata={"content_types": content_types, "topic": request.topic.strip()},
    )
    return {"draft": draft, "items": items, "content": bundle}


def _book_for_chapter(chapter_key: str, books: list[str]) -> tuple[str, str] | None:
    if "::" in chapter_key:
        book, chapter_id = chapter_key.split("::", 1)
        return book, chapter_id
    for book in books:
        if chapter_reader.get_chapter(book, chapter_key):
            return book, chapter_key
    return None


async def generate_chapter_promos(request: ChapterPromoRequest) -> dict[str, Any]:
    books = normalize_multi_select(request.books)
    chapter_keys = normalize_multi_select(request.chapters)
    platforms = normalize_multi_select(request.platforms, ["Instagram"])
    moods = normalize_multi_select(request.moods, ["Ominous"])
    hooks = normalize_multi_select(request.hooks, ["Choose the Strongest Hook for Me"])
    promo_outputs = normalize_multi_select(request.promo_outputs, cpp.DEFAULT_PROMO_OUTPUTS)
    output: list[str] = []
    labels: list[str] = []

    for chapter_key in chapter_keys:
        resolved = _book_for_chapter(chapter_key, books)
        if not resolved:
            continue
        book, chapter_id = resolved
        chapter = chapter_reader.get_chapter(book, chapter_id)
        if not chapter:
            continue
        chapter_label = str(chapter.get("label") or chapter_id)
        labels.append(f"{book} - {chapter_label}")
        chapter_text = chapter_reader.chapter_text_for_prompt(chapter)
        summary_prompt = cpp.build_summary_prompt(
            novel_title=book,
            chapter_label=chapter_label,
            chapter_text=chapter_text,
            genre=request.genre,
        )
        summary = (await asyncio.to_thread(generate_text, summary_prompt)).strip()
        output.append(f"## {book} - {chapter_label}\n\n{summary}")
        if request.mode == "Chapter Summary":
            continue
        for platform in platforms:
            teaser_prompt = cpp.build_teaser_prompt(
                platform=platform,
                novel_title=book,
                chapter_label=chapter_label,
                chapter_text=chapter_text,
                summary=summary,
                genre=request.genre,
                teaser_pillar=request.teaser_pillar,
                spoiler_level=request.spoiler_level,
                promo_objective=request.promotional_goal,
                tone=", ".join(moods),
                cta=request.cta,
                optional_quote=request.optional_quote,
                style_variant=request.style_variant,
                primary_hook=", ".join(hooks),
                promo_outputs=promo_outputs,
                campaign_mode=request.mode,
                featured_character=request.focus_character,
                promo_duration=request.promo_duration,
                genre_promo_mode=request.genre_promo_mode,
                spoiler_notes=request.spoiler_notes,
            )
            generated = (await asyncio.to_thread(generate_text, teaser_prompt)).strip()
            output.append(f"### {platform} Promo\n\n{generated}")

    content = "\n\n---\n\n".join(output).strip()
    if not content:
        raise ValueError("No matching chapters found")
    draft = draft_service.create_draft(
        title=f"{', '.join(labels[:3])} chapter promos"[:90],
        content_type="chapter_promos",
        content=content,
        metadata={
            "books": books,
            "chapters": labels,
            "platforms": platforms,
            "primary_hooks": hooks,
            "featured_character": request.focus_character,
        },
    )
    return {"content": content, "draft": draft}
