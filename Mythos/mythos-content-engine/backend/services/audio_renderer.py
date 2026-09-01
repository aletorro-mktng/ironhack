from __future__ import annotations

import asyncio
import os
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

from pydub import AudioSegment

from backend.models import Job, PodcastRenderRequest
from backend.repositories.artifact_repository import artifact_for_path
from backend.services import draft_service
from backend.workers import submit_job

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from elevenlabs_integration import DEFAULT_MODEL, DEFAULT_OUTPUT_FORMAT, strip_production_tags, synthesize_speech  # noqa: E402


PODCAST_PREVIEW_MAX_SEGMENTS = 3
PODCAST_TTS_MAX_CHARS = 2400
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def preview_output_path(voice_id: str) -> Path:
    return Path("outputs") / "voice_previews" / f"{voice_id}.mp3"


def slugify_filename(value: str, fallback: str = "podcast") -> str:
    value = (value or "").strip() or fallback
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    value = re.sub(r"[-\s]+", "_", value).strip("_").lower()
    return value or fallback


def normalize_speaker_label(value: str) -> str:
    value = re.sub(r"[*_`#>]", "", value or "")
    value = value.strip().strip("[]").strip()
    value = re.sub(r"[^A-Za-z0-9_ -]", "", value)
    value = re.sub(r"[-\s]+", "_", value).strip("_")
    return value.upper() or "HOST"


def parse_podcast_segments(script: str) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    current_label = ""
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_label, current_lines
        text = strip_production_tags("\n".join(current_lines).strip())
        if current_label and text:
            segments.append((current_label, text))
        current_lines = []

    for raw_line in (script or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        bracket_match = re.match(
            r"^\[(?!SFX:|INTRO MUSIC:|OUTRO MUSIC:|MUSIC BED:|PAUSE:|BEAT|CUT IF|OPTIONAL|USER EDIT:)([A-Za-z0-9_ -]+)\]\s*(.*)$",
            line,
            flags=re.IGNORECASE,
        )
        if bracket_match:
            flush()
            current_label = normalize_speaker_label(bracket_match.group(1))
            remainder = bracket_match.group(2).strip(" :-")
            current_lines = [remainder] if remainder else []
            continue
        colon_match = re.match(r"^([A-Za-z][A-Za-z0-9_ -]{1,40}):\s+(.+)$", line)
        if colon_match and not line.startswith("["):
            flush()
            current_label = normalize_speaker_label(colon_match.group(1))
            current_lines = [colon_match.group(2).strip()]
            continue
        if current_label:
            current_lines.append(line)

    flush()
    if segments:
        return segments

    fallback_lines = []
    for raw_line in (script or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("---"):
            continue
        if re.match(r"^\[(SFX:|INTRO MUSIC:|OUTRO MUSIC:|MUSIC BED:|PAUSE:|BEAT)", line, flags=re.IGNORECASE):
            continue
        fallback_lines.append(line)
    fallback_text = strip_production_tags("\n".join(fallback_lines).strip())
    return [("HOST", fallback_text)] if fallback_text else []


def split_tts_text(text: str, max_chars: int = PODCAST_TTS_MAX_CHARS) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        proposed = f"{current} {sentence}".strip()
        if len(proposed) <= max_chars:
            current = proposed
            continue
        if current:
            chunks.append(current)
        if len(sentence) <= max_chars:
            current = sentence
            continue
        words = sentence.split()
        current = ""
        for word in words:
            proposed_word = f"{current} {word}".strip()
            if len(proposed_word) > max_chars and current:
                chunks.append(current)
                current = word
            else:
                current = proposed_word
    if current:
        chunks.append(current)
    return chunks


def compact_podcast_segments(segments: list[tuple[str, str]]) -> list[tuple[str, str]]:
    compacted: list[tuple[str, str]] = []
    pending_label = ""
    pending_text = ""

    def flush() -> None:
        nonlocal pending_label, pending_text
        if pending_label and pending_text.strip():
            compacted.extend((pending_label, chunk) for chunk in split_tts_text(pending_text))
        pending_label = ""
        pending_text = ""

    for label, text in segments:
        text = text.strip()
        proposed = f"{pending_text}\n\n{text}".strip() if pending_text else text
        if label == pending_label and len(proposed) <= PODCAST_TTS_MAX_CHARS:
            pending_text = proposed
            continue
        flush()
        pending_label = label
        pending_text = text
    flush()
    return compacted


def selected_voice_ids(request: PodcastRenderRequest) -> list[str]:
    values = [request.host_voice_id, request.guest_voice_id, request.guest_2_voice_id, *request.voice_map.values()]
    return [voice_id for voice_id in dict.fromkeys(value.strip() for value in values if value and value.strip())]


def resolve_voice_for_label(label: str, request: PodcastRenderRequest, fallback_ids: list[str]) -> str:
    casting = {normalize_speaker_label(key): value for key, value in request.voice_map.items() if value}
    normalized = normalize_speaker_label(label)
    if normalized in casting:
        return casting[normalized]
    if normalized.startswith("GUEST_2") or normalized in {"COHOST", "SECOND_GUEST"}:
        return request.guest_2_voice_id or request.guest_voice_id or request.host_voice_id or fallback_ids[-1]
    if normalized.startswith("GUEST") or normalized in {"AUTHOR", "CRITIC", "INTERVIEWEE"}:
        return request.guest_voice_id or request.host_voice_id or fallback_ids[min(1, len(fallback_ids) - 1)]
    return request.host_voice_id or fallback_ids[0]


def render_podcast_audio(draft_id: str, request: PodcastRenderRequest, preview_only: bool, progress) -> dict:
    draft = draft_service.get_draft(draft_id)
    if not draft or draft.content_type != "podcast":
        raise ValueError("Podcast draft not found")
    if not os.getenv("ELEVENLABS_API_KEY"):
        raise EnvironmentError("ELEVENLABS_API_KEY is missing. Add it to .env and restart the API.")

    voices = selected_voice_ids(request)
    if not voices:
        raise ValueError("Choose at least one ElevenLabs voice before rendering podcast audio.")

    segments = compact_podcast_segments(parse_podcast_segments(draft.content))
    if preview_only:
        segments = segments[:PODCAST_PREVIEW_MAX_SEGMENTS]
    if not segments:
        raise ValueError("No renderable spoken text found in the podcast script.")

    progress(0.08, "Preparing audio render")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    render_mode = "preview" if preview_only else "full"
    file_stem = slugify_filename(draft.title or "podcast_audio")
    output_dir = PROJECT_ROOT / "outputs" / f"{file_stem}_{render_mode}_audio_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered_paths: list[Path] = []
    for index, (label, text) in enumerate(segments, start=1):
        progress(0.1 + (0.72 * ((index - 1) / max(len(segments), 1))), f"Rendering voice segment {index} of {len(segments)}")
        voice_id = resolve_voice_for_label(label, request, voices)
        segment_path = output_dir / f"{index:03d}_{label.lower()}.mp3"
        synthesize_speech(
            text=text,
            voice_id=voice_id,
            output_path=segment_path,
            model_id=request.model_id or DEFAULT_MODEL,
            output_format=DEFAULT_OUTPUT_FORMAT,
            strip_tags=True,
            stability=request.stability,
            similarity_boost=request.similarity_boost,
            style=request.style,
            speed=request.speed,
            use_speaker_boost=request.use_speaker_boost,
        )
        rendered_paths.append(segment_path)

    progress(0.86, "Combining segments into MP3")
    combined = AudioSegment.empty()
    pause = AudioSegment.silent(duration=650)
    for index, segment_path in enumerate(rendered_paths):
        if index:
            combined += pause
        combined += AudioSegment.from_file(segment_path)

    output_path = output_dir / f"{file_stem}_{render_mode}.mp3"
    combined.export(output_path, format="mp3", bitrate="128k")

    progress(0.94, "Packaging podcast artifacts")
    manifest_path = output_dir / "podcast_audio_manifest.md"
    manifest_path.write_text(
        "\n".join(
            [
                "# Podcast Audio Manifest",
                "",
                f"Draft ID: {draft_id}",
                f"Mode: {render_mode}",
                f"Document: {draft.title or file_stem}",
                f"Model: {request.model_id or DEFAULT_MODEL}",
                f"Rendered segments: {len(rendered_paths)}",
                "",
                "## Voice Casting",
                *[f"- {label}: {resolve_voice_for_label(label, request, voices)}" for label, _text in segments],
                "",
                "## Segment Files",
                *[f"- {path.name}" for path in rendered_paths],
                "",
                f"Combined MP3: {output_path.name}",
            ]
        ),
        encoding="utf-8",
    )

    zip_path = output_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(output_path, arcname=output_path.name)
        archive.write(manifest_path, arcname=manifest_path.name)
        for segment_path in rendered_paths:
            archive.write(segment_path, arcname=f"segments/{segment_path.name}")

    audio_draft = draft_service.create_draft(
        title=f"{draft.title or file_stem} {render_mode} audio",
        content_type="podcast_audio",
        content=manifest_path.read_text(encoding="utf-8"),
        source_path=output_path,
        metadata={
            "mode": render_mode,
            "package_path": str(zip_path),
            "segment_count": len(rendered_paths),
            "model": request.model_id or DEFAULT_MODEL,
        },
    )
    return {
        "draft_id": draft_id,
        "audio_draft_id": audio_draft.id,
        "render_type": render_mode,
        "segment_count": len(rendered_paths),
        "mp3": artifact_for_path(output_path, "audio/mpeg").model_dump(),
        "package": artifact_for_path(zip_path, "application/zip").model_dump(),
        "manifest": artifact_for_path(manifest_path, "text/markdown").model_dump(),
    }


def _submit_render(draft_id: str, request: PodcastRenderRequest, preview_only: bool) -> Job:
    async def handler(progress):
        return await asyncio.to_thread(render_podcast_audio, draft_id, request, preview_only, progress)

    return submit_job("Queued podcast preview" if preview_only else "Queued podcast render", handler)


def render_podcast_preview(draft_id: str, request: PodcastRenderRequest) -> Job:
    return _submit_render(draft_id, request, preview_only=True)


def render_podcast_full(draft_id: str, request: PodcastRenderRequest) -> Job:
    return _submit_render(draft_id, request, preview_only=False)
