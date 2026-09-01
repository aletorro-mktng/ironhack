from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models import Artifact, VoicePreviewRequest
from backend.repositories.artifact_repository import artifact_for_path
from backend.services.audio_renderer import preview_output_path
from backend.services.elevenlabs_service import get_voices, render_voice_preview


router = APIRouter(prefix="/api/voices", tags=["voices"])


@router.get("")
def voices() -> list[dict]:
    try:
        return get_voices()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/{voice_id}/preview")
def voice_preview(voice_id: str, request: VoicePreviewRequest) -> Artifact:
    try:
        path = render_voice_preview(voice_id, request.text, preview_output_path(voice_id))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return artifact_for_path(path, "audio/mpeg")
