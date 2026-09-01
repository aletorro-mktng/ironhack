from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models import DraftUpdateRequest, PodcastDraft, PodcastGenerationRequest
from backend.services import draft_service
from backend.services.podcast_generator import generate_podcast_script, to_podcast_draft


router = APIRouter(prefix="/api/podcasts", tags=["podcasts"])


@router.post("/scripts")
async def create_script(request: PodcastGenerationRequest) -> PodcastDraft:
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="topic is required")
    return await generate_podcast_script(request)


@router.post("/generate")
async def legacy_create_script(request: PodcastGenerationRequest) -> dict:
    draft = await create_script(request)
    return {"content": draft.content, "draft": draft}


@router.get("/{draft_id}")
def get_podcast(draft_id: str) -> PodcastDraft:
    draft = draft_service.get_draft(draft_id)
    if not draft or draft.content_type != "podcast":
        raise HTTPException(status_code=404, detail="Podcast draft not found")
    return to_podcast_draft(draft)


@router.patch("/{draft_id}")
def update_podcast(draft_id: str, request: DraftUpdateRequest) -> PodcastDraft:
    draft = draft_service.update_draft(draft_id, request)
    return to_podcast_draft(draft)
