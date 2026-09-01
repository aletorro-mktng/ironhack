from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from elevenlabs_integration import list_voices, synthesize_speech  # noqa: E402


def get_voices() -> list[dict]:
    return list_voices()


def render_voice_preview(voice_id: str, text: str, output_path: str | Path) -> Path:
    return synthesize_speech(text=text, voice_id=voice_id, output_path=output_path)
