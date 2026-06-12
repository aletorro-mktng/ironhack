"""ElevenLabs text-to-speech helpers."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib import error, request

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
ELEVENLABS_VOICES_URL = "https://api.elevenlabs.io/v1/voices"
DEFAULT_MODEL = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
DEFAULT_OUTPUT_FORMAT = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")

# Voices you create (cloned/professional/generated) sort before the stock premade ones.
VOICE_CATEGORY_ORDER = {"cloned": 0, "professional": 1, "generated": 2, "premade": 3}


def list_voices() -> list[dict]:
    """
    Fetch the voices available on the configured ElevenLabs account.

    Returns a list of {"voice_id", "name", "category"} dicts with the voices you
    created (cloned/professional/generated) sorted ahead of stock premade ones.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "ELEVENLABS_API_KEY was not found. Add it to your .env file."
        )

    api_request = request.Request(
        ELEVENLABS_VOICES_URL,
        headers={"xi-api-key": api_key},
        method="GET",
    )

    try:
        with request.urlopen(api_request) as response:
            data = json.loads(response.read())
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ElevenLabs voices request failed: {exc.code} {details}") from exc

    voices = [
        {
            "voice_id": voice.get("voice_id", ""),
            "name": voice.get("name", "(unnamed)"),
            "category": voice.get("category", "premade"),
            "preview_url": voice.get("preview_url", ""),
        }
        for voice in data.get("voices", [])
        if voice.get("voice_id")
    ]

    voices.sort(
        key=lambda voice: (
            VOICE_CATEGORY_ORDER.get(voice["category"], 4),
            voice["name"].lower(),
        )
    )

    return voices


PRODUCTION_TAG_PATTERN = re.compile(r"\[([^\]]+)\]")
PRODUCTION_TAG_PREFIXES = (
    "sfx:",
    "intro music:",
    "outro music:",
    "music bed:",
    "pause:",
    "cut if",
    "optional",
    "user edit:",
)
NATURAL_DELIVERY_TAGS = {
    "beat",
    "breath",
    "laughs",
    "sighs",
    "softly",
    "coughs",
    "smiles",
    "leans in",
    "lowers voice",
    "whispers",
    "under breath",
    "pause",
    "dryly",
    "warmer",
    "tense",
    "quietly",
    "excited",
}


def strip_production_tags(text: str) -> str:
    """
    Remove square-bracket production/editing tags before TTS rendering.

    Bracketed performance cues are preserved because some ElevenLabs models can
    use them for more natural delivery. Production tags and speaker labels are
    removed.
    """

    def replace_tag(match: re.Match) -> str:
        tag = match.group(1).strip()
        normalized = re.sub(r"\s+", " ", tag.lower())
        if normalized in NATURAL_DELIVERY_TAGS:
            return f"[{tag}]"
        if any(normalized.startswith(prefix) for prefix in PRODUCTION_TAG_PREFIXES):
            return ""
        if re.fullmatch(r"[A-Z0-9_ -]{2,40}", tag):
            return ""
        return ""

    cleaned_text = PRODUCTION_TAG_PATTERN.sub(replace_tag, text)
    return "\n".join(line.strip() for line in cleaned_text.splitlines() if line.strip())


def synthesize_speech(
    text: str,
    voice_id: str,
    output_path: str | Path,
    model_id: str = DEFAULT_MODEL,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    strip_tags: bool = True,
    stability: float | None = None,
    similarity_boost: float | None = None,
    style: float | None = None,
    speed: float | None = None,
    use_speaker_boost: bool | None = None,
) -> Path:
    """
    Render one text segment with one ElevenLabs voice.

    Multi-speaker podcasts should be split into speaker-specific segments before
    calling this helper, then stitched in editing software or a later pipeline step.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "ELEVENLABS_API_KEY was not found. Add it to your .env file."
        )

    if not voice_id:
        raise ValueError("voice_id is required for ElevenLabs speech synthesis.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"{ELEVENLABS_API_URL}/{voice_id}?output_format={output_format}"
    render_text = strip_production_tags(text) if strip_tags else text

    voice_settings = {}
    if stability is not None:
        voice_settings["stability"] = float(stability)
    if similarity_boost is not None:
        voice_settings["similarity_boost"] = float(similarity_boost)
    if style is not None:
        voice_settings["style"] = float(style)
    if speed is not None:
        voice_settings["speed"] = float(speed)
    if use_speaker_boost is not None:
        voice_settings["use_speaker_boost"] = bool(use_speaker_boost)

    payload_data = {
        "text": render_text,
        "model_id": model_id,
    }
    if voice_settings:
        payload_data["voice_settings"] = voice_settings

    payload = json.dumps(payload_data).encode("utf-8")

    api_request = request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "xi-api-key": api_key,
        },
        method="POST",
    )

    try:
        with request.urlopen(api_request) as response:
            output_path.write_bytes(response.read())
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ElevenLabs request failed: {exc.code} {details}") from exc

    return output_path


if __name__ == "__main__":
    print("ElevenLabs integration module loaded.")
    print("Set ELEVENLABS_API_KEY and pass a voice_id to synthesize_speech().")
