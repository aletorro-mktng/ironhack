from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

from backend.models import Draft, PodcastDraft, PodcastGenerationRequest, PodcastSegment, PodcastSpeaker, PodcastTurn
from backend.services import draft_service

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from content_pipeline import run_pipeline  # noqa: E402
from context_filter import requested_books  # noqa: E402
from backend.services.content_generator import source_evidence_pack_for


def _slug(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or fallback


def _speaker_parts(raw_roles: str) -> list[tuple[str, str]]:
    parts = [part.strip() for part in re.split(r"[,/|]", raw_roles or "") if part.strip()]
    if not parts:
        parts = ["Host"]
    speakers: list[tuple[str, str]] = []
    for index, part in enumerate(parts, start=1):
        if ":" in part:
            role, name = [piece.strip() for piece in part.split(":", 1)]
        else:
            role, name = part, part
        speakers.append((_slug(role or name, f"speaker_{index}"), name or role))
    return speakers


def _speakers_from_request(request: PodcastGenerationRequest) -> list[PodcastSpeaker]:
    speakers = []
    for index, (speaker_id, name) in enumerate(_speaker_parts(request.speaker_roles), start=1):
        speakers.append(PodcastSpeaker(id=speaker_id, name=name, role=name if name != speaker_id else f"Speaker {index}"))
    return speakers


def _resolve_speaker_id(label: str, speakers: list[PodcastSpeaker]) -> str:
    normalized = _slug(label, "host")
    for speaker in speakers:
        if normalized == speaker.id or normalized in {_slug(speaker.name, ""), _slug(speaker.role, "")}:
            return speaker.id
    if normalized.startswith("guest"):
        guest = next((speaker for speaker in speakers if "guest" in speaker.id or "guest" in speaker.role.lower()), None)
        if guest:
            return guest.id
    return speakers[0].id if speakers else "host"


def _extract_turns(raw_script: str, speakers: list[PodcastSpeaker]) -> list[PodcastTurn]:
    turns: list[PodcastTurn] = []
    current_label = ""
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_label, current_lines
        text = "\n".join(current_lines).strip()
        if current_label and text:
            turns.append(
                PodcastTurn(
                    id=f"turn_{len(turns) + 1}",
                    speaker_id=_resolve_speaker_id(current_label, speakers),
                    text=text,
                    estimated_duration_seconds=round(len(text.split()) / 2.6, 1),
                )
            )
        current_lines = []

    for raw_line in (raw_script or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^\[?([A-Za-z][A-Za-z0-9_ -]{1,40})\]?:\s+(.+)$", line)
        bracket_match = re.match(r"^\[([A-Za-z][A-Za-z0-9_ -]{1,40})\]\s+(.+)$", line)
        speaker_match = match or bracket_match
        if speaker_match:
            flush()
            current_label = speaker_match.group(1)
            current_lines = [speaker_match.group(2)]
            continue
        if current_label:
            current_lines.append(line)
    flush()

    if turns:
        return turns
    fallback_text = "\n".join(line.strip() for line in (raw_script or "").splitlines() if line.strip())
    return [PodcastTurn(id="turn_1", speaker_id=speakers[0].id if speakers else "host", text=fallback_text)] if fallback_text else []


def to_podcast_draft(draft: Draft, request: PodcastGenerationRequest | None = None) -> PodcastDraft:
    metadata = draft.metadata or {}
    request_data = request.model_dump() if request else metadata
    speakers = _speakers_from_request(PodcastGenerationRequest(**request_data)) if request_data else [PodcastSpeaker(id="host", name="Host", role="Host")]
    turns = _extract_turns(draft.content, speakers)
    return PodcastDraft(
        **draft.model_dump(),
        show_title=str(request_data.get("show_title") or ""),
        episode_title=str(request_data.get("episode_title") or draft.title),
        speakers=speakers,
        segments=[PodcastSegment(id="segment_1", title="Generated Script", turns=turns)],
        raw_script=draft.content,
    )


def podcast_brief(request: PodcastGenerationRequest) -> str:
    topic_books = requested_books(request.topic)
    selected_book = request.source_title or (topic_books[0] if topic_books else "")
    lines = [
        request.topic.strip(),
        f"Source title: {request.source_title or 'Not specified'}",
        f"Podcast show title: {request.show_title or 'Not specified'}",
        f"Episode title: {request.episode_title or 'Not specified'}",
        f"Episode number: {request.episode_number or 'Not specified'}",
        f"Podcast destination: {request.destination}",
        f"Podcast format: {request.podcast_format}",
        f"Podcast speaker count: {request.speakers}",
        f"Podcast speaker roles/names: {request.speaker_roles}",
        f"Podcast tone: {', '.join(request.tone) or 'Not specified'}",
        f"Podcast audience: {', '.join(request.audience) or 'Not specified'}",
        f"Podcast constraints: {', '.join(request.constraints) or 'Not specified'}",
        f"Knowledge sources to use: {', '.join(request.knowledge_sources) or 'Not specified'}",
        f"Source focus: {request.source_focus or 'Not specified'}",
        "Source grounding: when manuscript/RAG moments, pull quotes, reviews, or character info are enabled, ground the script in those retrieved knowledge-base sources.",
        "For quote, dialogue, funniest-lines, savage-moment, friendship-moment, or top-ranked episode topics, use exact quoted lines where available and clearly identify manuscript-grounded moments. Do not invent dialogue.",
        "Use reviews only as reader/reception evidence, not as book dialogue. Use character info to correctly attribute actions, relationships, and speaker context.",
        f"Podcast performance cues: {', '.join(request.performance_cues) or 'Not specified'}",
        f"Podcast custom constraints: {request.custom_constraints or 'Not specified'}",
        f"ElevenLabs model preference: {request.model_id or 'Not specified'}",
        f"Podcast target length: {request.target_length}",
        f"CTA: {request.cta or 'Not specified'}",
    ]
    evidence_pack = source_evidence_pack_for(
        topic=request.topic,
        source_focus=request.source_focus,
        selected_book=selected_book,
        knowledge_sources=request.knowledge_sources,
    )
    if evidence_pack:
        lines.append(evidence_pack)
    return "\n".join(lines)


async def generate_podcast_script(request: PodcastGenerationRequest) -> PodcastDraft:
    result = await asyncio.to_thread(run_pipeline, "podcast", podcast_brief(request))
    draft = draft_service.create_draft(
        title=(request.episode_title or request.show_title or request.topic or "Podcast script")[:90],
        content_type="podcast",
        content=result["generated_content"],
        source_path=result["draft_path"],
        metadata=request.model_dump(),
    )
    return to_podcast_draft(draft, request)
