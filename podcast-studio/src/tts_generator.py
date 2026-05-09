import os
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.generators import Sine, WhiteNoise

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_VOICE_SEQUENCE = [
    "coral",
    "alloy",
    "fable",
    "sage",
    "ash",
    "nova",
]

TTS_1_SUPPORTED_VOICES = {
    "alloy",
    "ash",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
}

TTS_1_VOICE_FALLBACKS = {
    "cedar": "ash",
    "marin": "coral",
    "verse": "fable",
}

BRACKET_SPEAKER_LINE_PATTERN = re.compile(r"^\s*\[(HOST|CO-HOST|GUEST)\]\s*(.+)$", re.IGNORECASE)
COLON_SPEAKER_LINE_PATTERN = re.compile(r"^\s*\**\s*(Host|Co-host|Guest)\s*\**\s*:\s*\**\s*(.+)$", re.IGNORECASE)
PERFORMANCE_TAG_PATTERN = re.compile(r"(\[[^\]]+\]|\([A-Za-z][A-Za-z0-9 .:_-]{1,30}\)|<[^>]+>)")
MARKDOWN_HEADING_PATTERN = re.compile(r"^\s*\**\s*(title|script|podcast script)\s*:.*\**\s*$", re.IGNORECASE)

TAG_PAUSE_DURATIONS = {
    "pause": 700,
}

PERFORMANCE_EFFECT_TAGS = {
    "laugh",
    "laughs",
    "laughing",
    "sigh",
    "sighs",
    "cough",
    "coughs",
    "breath",
    "breathes",
}

MUSIC_STYLE_FREQUENCIES = {
    "upbeat": [523, 659, 784, 1046],
    "calm": [392, 494, 587, 784],
    "news": [330, 392, 523, 659],
    "tech": [220, 440, 660, 880],
    "warm": [349, 440, 523, 698],
}

ACCENT_INSTRUCTIONS = {
    "neutral": "Speak clearly with a neutral, natural podcast delivery.",
    "american": "Speak clearly with a natural American English accent.",
    "british": "Speak clearly with a natural British English accent.",
    "spanish": "Speak clearly with a natural Spanish accent while preserving the script wording.",
    "mexican spanish": "Speak clearly with a natural Mexican Spanish accent while preserving the script wording.",
    "french": "Speak clearly with a natural French accent while preserving the script wording.",
    "german": "Speak clearly with a natural German accent while preserving the script wording.",
    "italian": "Speak clearly with a natural Italian accent while preserving the script wording.",
    "indian english": "Speak clearly with a natural Indian English accent.",
    "australian": "Speak clearly with a natural Australian English accent.",
}


def parse_script_segments(script_text: str) -> list[tuple[str, str]]:
    segments = []
    current_speaker = "Narrator"
    current_lines = []

    for line in script_text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if MARKDOWN_HEADING_PATTERN.match(stripped):
            continue

        match = BRACKET_SPEAKER_LINE_PATTERN.match(stripped) or COLON_SPEAKER_LINE_PATTERN.match(stripped)

        if match:
            if current_lines:
                segments.append((current_speaker, " ".join(current_lines)))
                current_lines = []

            current_speaker = f"[{match.group(1).strip().upper()}]"
            current_lines.append(match.group(2).strip().strip("*").strip())
        else:
            current_lines.append(stripped.strip("*").strip())

    if current_lines:
        segments.append((current_speaker, " ".join(current_lines)))

    return [(speaker, text) for speaker, text in segments if text]


def normalize_speaker_name(speaker: str) -> str:
    return speaker.strip().strip("[]").lower().replace("_", "-")


def normalize_tag(tag: str) -> str:
    return tag.strip("[]()<> ").lower()


def pause_duration_for_tag(tag: str) -> int:
    normalized_tag = normalize_tag(tag)

    if normalized_tag.startswith("pause:"):
        pause_value = normalized_tag.split(":", 1)[1].strip().rstrip("s")

        try:
            return max(100, int(float(pause_value) * 1000))
        except ValueError:
            return TAG_PAUSE_DURATIONS["pause"]

    return TAG_PAUSE_DURATIONS.get(normalized_tag, 300)


def create_sound_effect(effect_name: str) -> AudioSegment:
    normalized_name = effect_name.strip().lower()

    if normalized_name == "ding":
        return (
            Sine(880).to_audio_segment(duration=160).fade_in(5).fade_out(60)
            + Sine(1320).to_audio_segment(duration=260).fade_in(5).fade_out(140)
        ) - 4

    if normalized_name == "chime":
        return (
            Sine(660).to_audio_segment(duration=180).fade_out(100)
            + Sine(880).to_audio_segment(duration=180).fade_out(100)
            + Sine(1100).to_audio_segment(duration=240).fade_out(150)
        ) - 6

    if normalized_name == "transition":
        return (
            Sine(440).to_audio_segment(duration=120).fade_out(80)
            + Sine(660).to_audio_segment(duration=120).fade_out(80)
            + Sine(880).to_audio_segment(duration=160).fade_out(100)
        ) - 5

    if normalized_name == "whoosh":
        return WhiteNoise().to_audio_segment(duration=650).fade_in(260).fade_out(220) - 22

    if normalized_name == "applause":
        burst = AudioSegment.silent(duration=0)

        for _ in range(9):
            burst += WhiteNoise().to_audio_segment(duration=55).fade_in(3).fade_out(35) - 18
            burst += AudioSegment.silent(duration=45)

        return burst

    return AudioSegment.silent(duration=300)


def create_performance_effect(tag_name: str) -> AudioSegment:
    normalized_name = tag_name.strip().lower()

    if normalized_name in ("sigh", "sighs"):
        sigh = WhiteNoise().to_audio_segment(duration=700).fade_in(120).fade_out(260) - 34
        fall = Sine(220).to_audio_segment(duration=300).fade_in(20).fade_out(260) - 24
        return sigh.overlay(fall, position=100)

    if normalized_name in ("cough", "coughs"):
        cough = AudioSegment.silent(duration=0)

        for _ in range(2):
            cough += WhiteNoise().to_audio_segment(duration=95).fade_in(5).fade_out(45) - 17
            cough += AudioSegment.silent(duration=90)

        return cough

    if normalized_name in ("breath", "breathes"):
        return WhiteNoise().to_audio_segment(duration=320).fade_in(120).fade_out(160) - 38

    return AudioSegment.silent(duration=300)


def generate_tts_performance_effect(
    tag_name: str,
    voice: str,
    accent: str,
    temp_dir: Path,
    segment_index: int
) -> tuple[AudioSegment, int]:
    normalized_name = tag_name.strip().lower()

    if normalized_name not in ("laugh", "laughs", "laughing"):
        return create_performance_effect(tag_name), segment_index

    effect_path = temp_dir / f"segment_{segment_index:03}_laugh.mp3"

    with client.audio.speech.with_streaming_response.create(
        model=os.getenv("TTS_MODEL", "tts-1"),
        voice=voice,
        instructions=f"{speech_instructions_for_accent(accent)} Give a brief, natural, friendly chuckle. Do not say any words.",
        input="Haha.",
    ) as response:
        response.stream_to_file(effect_path)

    return AudioSegment.from_file(effect_path) - 2, segment_index + 1


def get_audio_file_path(audio_file: Any) -> Optional[Path]:
    if not audio_file:
        return None

    if isinstance(audio_file, (str, Path)):
        return Path(audio_file)

    if isinstance(audio_file, dict):
        for key in ("path", "name", "orig_name"):
            value = audio_file.get(key)

            if value:
                return Path(value)

    for attribute in ("name", "path"):
        value = getattr(audio_file, attribute, None)

        if value:
            return Path(value)

    return None


def create_music_bed(style: str, duration_ms: int) -> AudioSegment:
    normalized_style = style.lower().replace("generated:", "").strip()
    frequencies = MUSIC_STYLE_FREQUENCIES.get(normalized_style, MUSIC_STYLE_FREQUENCIES["warm"])
    bed = AudioSegment.silent(duration=duration_ms)
    beat_ms = 420 if normalized_style in ("upbeat", "tech") else 650

    for index, start_ms in enumerate(range(0, duration_ms, beat_ms)):
        frequency = frequencies[index % len(frequencies)]
        tone = Sine(frequency).to_audio_segment(duration=min(beat_ms, 520)).fade_in(15).fade_out(180) - 22
        bed = bed.overlay(tone, position=start_ms)

        if normalized_style in ("upbeat", "news", "tech"):
            accent = Sine(frequency * 2).to_audio_segment(duration=90).fade_out(70) - 26
            bed = bed.overlay(accent, position=start_ms)

    return bed.fade_in(300).fade_out(700)


def load_music_bed(
    selected_music: str,
    uploaded_music: Any = None,
    duration_ms: int = 4500
) -> AudioSegment:
    uploaded_path = get_audio_file_path(uploaded_music)

    if uploaded_path:
        music = AudioSegment.from_file(uploaded_path)
        return music[:duration_ms].fade_in(250).fade_out(700) - 8

    if not selected_music or selected_music == "None":
        return AudioSegment.silent(duration=0)

    return create_music_bed(selected_music, duration_ms)


def parse_performance_parts(text: str) -> list[tuple[str, object]]:
    parts = []
    cursor = 0

    for match in PERFORMANCE_TAG_PATTERN.finditer(text):
        spoken_text = text[cursor:match.start()].strip()

        if spoken_text:
            parts.append(("text", spoken_text))

        normalized_tag = normalize_tag(match.group(0))

        if normalized_tag.startswith("sfx:"):
            parts.append(("sfx", normalized_tag.split(":", 1)[1].strip()))
        elif normalized_tag in PERFORMANCE_EFFECT_TAGS:
            parts.append(("effect", normalized_tag))
        else:
            parts.append(("pause", pause_duration_for_tag(match.group(0))))

        cursor = match.end()

    remaining_text = text[cursor:].strip()

    if remaining_text:
        parts.append(("text", remaining_text))

    return parts


def voice_for_speaker(
    speaker: str,
    voice_map: dict[str, str],
    configured_voices: Optional[dict[str, str]] = None
) -> str:
    normalized_speaker = speaker.lower()
    normalized_speaker = normalize_speaker_name(normalized_speaker)

    if configured_voices and normalized_speaker in configured_voices:
        return configured_voices[normalized_speaker]

    if normalized_speaker not in voice_map:
        voice_index = len(voice_map) % len(DEFAULT_VOICE_SEQUENCE)
        voice_map[normalized_speaker] = DEFAULT_VOICE_SEQUENCE[voice_index]

    return voice_map[normalized_speaker]


def resolve_voice_for_model(voice: str, model: str) -> str:
    normalized_voice = voice.strip().lower()

    if model == "tts-1" and normalized_voice not in TTS_1_SUPPORTED_VOICES:
        return TTS_1_VOICE_FALLBACKS.get(normalized_voice, "alloy")

    return normalized_voice


def accent_for_speaker(
    speaker: str,
    configured_accents: Optional[dict[str, str]] = None
) -> str:
    normalized_speaker = normalize_speaker_name(speaker)

    if configured_accents and normalized_speaker in configured_accents:
        return configured_accents[normalized_speaker]

    return "neutral"


def speech_instructions_for_accent(accent: str) -> str:
    normalized_accent = accent.strip().lower()
    return ACCENT_INSTRUCTIONS.get(normalized_accent, ACCENT_INSTRUCTIONS["neutral"])


def generate_audio(
    script_text: str,
    speaker_voices: Optional[dict[str, str]] = None,
    speaker_accents: Optional[dict[str, str]] = None,
    intro_music: str = "None",
    outro_music: str = "None",
    intro_music_file: Any = None,
    outro_music_file: Any = None,
    output_dir: str = "outputs"
) -> str:
    if not script_text or len(script_text.strip()) < 50:
        raise ValueError("Script is too short to generate audio.")

    Path(output_dir).mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"podcast_episode_{timestamp}.mp3"
    temp_dir = Path(output_dir) / f"segments_{timestamp}"
    temp_dir.mkdir(exist_ok=True)

    segments = parse_script_segments(script_text)

    if not segments:
        raise ValueError("No readable script segments were found.")

    voice_map = {}
    configured_voices = {
        normalize_speaker_name(speaker): voice
        for speaker, voice in (speaker_voices or {}).items()
        if speaker and voice
    }
    configured_accents = {
        normalize_speaker_name(speaker): accent
        for speaker, accent in (speaker_accents or {}).items()
        if speaker and accent
    }
    combined_audio = AudioSegment.empty()
    segment_index = 1

    for speaker, text in segments:
        tts_model = os.getenv("TTS_MODEL", "tts-1")
        voice = resolve_voice_for_model(voice_for_speaker(speaker, voice_map, configured_voices), tts_model)
        accent = accent_for_speaker(speaker, configured_accents)

        for part_type, part_value in parse_performance_parts(text):
            if part_type == "pause":
                combined_audio += AudioSegment.silent(duration=int(part_value))
                continue

            if part_type == "sfx":
                combined_audio += create_sound_effect(str(part_value))
                combined_audio += AudioSegment.silent(duration=150)
                continue

            if part_type == "effect":
                performance_audio, segment_index = generate_tts_performance_effect(
                    str(part_value),
                    voice,
                    accent,
                    temp_dir,
                    segment_index
                )
                combined_audio += performance_audio
                combined_audio += AudioSegment.silent(duration=120)
                continue

            segment_path = temp_dir / f"segment_{segment_index:03}.mp3"

            with client.audio.speech.with_streaming_response.create(
                model=tts_model,
                voice=voice,
                instructions=speech_instructions_for_accent(accent),
                input=str(part_value)[:4000],
            ) as response:
                response.stream_to_file(segment_path)

            combined_audio += AudioSegment.from_file(segment_path)
            segment_index += 1

        combined_audio += AudioSegment.silent(duration=350)

    intro_bed = load_music_bed(intro_music, intro_music_file, duration_ms=4500)
    outro_bed = load_music_bed(outro_music, outro_music_file, duration_ms=5500)

    if len(intro_bed):
        combined_audio = intro_bed.append(combined_audio, crossfade=min(700, len(intro_bed), len(combined_audio)))

    if len(outro_bed):
        combined_audio = combined_audio.append(outro_bed, crossfade=min(900, len(outro_bed), len(combined_audio)))

    combined_audio.export(output_path, format="mp3")

    for segment_file in temp_dir.glob("*.mp3"):
        segment_file.unlink()

    temp_dir.rmdir()

    return str(output_path)
