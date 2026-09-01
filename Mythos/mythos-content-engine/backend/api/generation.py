from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.models import CampaignGenerationRequest, ChapterPromoRequest, ContentGenerationRequest
from backend.services.content_generator import generate_campaign, generate_chapter_promos, generate_content

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import chapter_promo_prompts as cpp  # noqa: E402
import chapter_reader  # noqa: E402
from prompt_templates import list_supported_content_types  # noqa: E402
from selection_options import (  # noqa: E402
    AUDIENCE_OPTIONS,
    CHARACTER_TAGS,
    CONSTRAINT_OPTIONS,
    PLATFORM_OPTIONS,
    PODCAST_DESTINATION_OPTIONS,
    PODCAST_FORMAT_OPTIONS,
    PODCAST_LENGTH_OPTIONS,
    PODCAST_TONE_OPTIONS,
    QUOTE_BOOK_OPTIONS,
    SOCIAL_OBJECTIVES,
)


router = APIRouter(prefix="/api", tags=["generation"])


@router.get("/options")
def options() -> dict[str, Any]:
    books = chapter_reader.list_chapter_books()
    chapters = []
    for book in books:
        for chapter_id, label in chapter_reader.chapter_options(book).items():
            chapters.append({"id": f"{book}::{chapter_id}", "book": book, "label": label})
    return {
        "contentTypes": list_supported_content_types(),
        "books": books or QUOTE_BOOK_OPTIONS,
        "chapters": chapters,
        "platforms": PLATFORM_OPTIONS,
        "characters": CHARACTER_TAGS,
        "audiences": AUDIENCE_OPTIONS,
        "objectives": SOCIAL_OBJECTIVES,
        "constraints": CONSTRAINT_OPTIONS,
        "podcast": {
            "destinations": PODCAST_DESTINATION_OPTIONS,
            "formats": PODCAST_FORMAT_OPTIONS,
            "tones": PODCAST_TONE_OPTIONS,
            "lengths": PODCAST_LENGTH_OPTIONS,
        },
        "chapterPromos": {
            "modes": cpp.CHAPTER_PROMO_MODES,
            "goals": cpp.PROMOTIONAL_GOALS,
            "moods": cpp.PROMOTIONAL_MOODS,
            "hooks": cpp.PROMO_HOOK_TYPES,
            "genres": cpp.GENRE_OPTIONS,
            "teaserPillars": list(cpp.TEASER_PILLARS.keys()),
            "durations": cpp.PROMO_DURATIONS,
            "genrePromoModes": cpp.GENRE_PROMO_MODE_OPTIONS,
            "styleVariants": [{"value": key, "label": label} for key, label in cpp.STYLE_VARIANTS["chapter_promos"]],
            "promoOutputs": [{"value": key, "label": label} for key, label in cpp.PROMO_OUTPUT_FORMATS.items()],
        },
    }


@router.post("/content/generate")
async def content_generate(request: ContentGenerationRequest) -> dict[str, Any]:
    if not request.content_type or not request.topic.strip():
        raise HTTPException(status_code=400, detail="content_type and topic are required")
    return await generate_content(request)


@router.post("/generate")
async def legacy_generate(request: ContentGenerationRequest) -> dict[str, Any]:
    generated = await content_generate(request)
    return {
        "content": generated["content"],
        "draft": generated["draft"],
        "result": {
            "generated_content": generated["content"],
            **generated["artifacts"],
        },
    }


@router.post("/campaigns/generate")
async def campaign_generate(request: CampaignGenerationRequest) -> dict[str, Any]:
    if not request.topic.strip() or not request.content_types:
        raise HTTPException(status_code=400, detail="topic and at least one content type are required")
    return await generate_campaign(request)


@router.post("/chapter-promos/generate")
async def chapter_promos_generate(request: ChapterPromoRequest) -> dict[str, Any]:
    if not request.books or not request.chapters:
        raise HTTPException(status_code=400, detail="Select at least one book and one chapter")
    try:
        return await generate_chapter_promos(request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
