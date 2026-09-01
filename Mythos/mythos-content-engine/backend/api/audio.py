from __future__ import annotations

from fastapi import APIRouter

from backend.models import Job, PodcastRenderRequest
from backend.services.audio_renderer import render_podcast_full, render_podcast_preview


router = APIRouter(prefix="/api", tags=["audio"])


@router.post("/podcasts/{draft_id}/preview")
def podcast_preview(draft_id: str, request: PodcastRenderRequest | None = None) -> Job:
    return render_podcast_preview(draft_id, request or PodcastRenderRequest())


@router.post("/podcasts/{draft_id}/render")
def podcast_render(draft_id: str, request: PodcastRenderRequest | None = None) -> Job:
    return render_podcast_full(draft_id, request or PodcastRenderRequest())
