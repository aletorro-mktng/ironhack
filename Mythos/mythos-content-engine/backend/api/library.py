from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter

from backend.repositories.artifact_repository import artifact_for_path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import outputs_browser  # noqa: E402


router = APIRouter(prefix="/api", tags=["library"])


@router.get("/gallery/images")
def gallery_images(content_type: str = "All types", sort: str = "Newest first") -> list[dict]:
    images = outputs_browser.load_gallery_images(content_type, sort)
    for image in images:
        if image.get("path"):
            image["url"] = f"/api/artifacts/{artifact_for_path(image['path']).id}/download"
    return images


@router.get("/library/audio")
def audio_library(sort: str = "Newest first") -> list[dict]:
    episodes = outputs_browser.load_audio_episodes(sort)
    for episode in episodes:
        if episode.get("mp3_path"):
            episode["url"] = f"/api/artifacts/{artifact_for_path(episode['mp3_path']).id}/download"
    return episodes
