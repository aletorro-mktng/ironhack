"""Loaders for the Gallery (generated images) and Library Audio (podcast episodes) tabs.

These scan the real ``outputs/`` layout the app actually writes:

- Images:   outputs/generated_visuals/<ts>/<content_type>_<topic>_<format>_<aspect>.png
            outputs/campaign_quote_graphics/<ts>/...
            outputs/press_release_assets/<ts>/...
- Audio:    outputs/<slug>_<full|preview>_audio_<ts>/  containing numbered segment mp3s,
            a combined mp3, and a ``podcast_audio_manifest.md``.

Documents are not handled here — the Library Documents view reads the draft index via
``draft_store`` directly, which already carries clean title/type/date/metadata.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Directories that hold generated images (user uploads in outputs/uploads are excluded).
IMAGE_DIRS = ("generated_visuals", "campaign_quote_graphics", "press_release_assets")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Known content-type prefixes, longest first so the greedy match is correct.
_CONTENT_TYPE_PREFIXES = (
    "instagram_caption",
    "character_spotlight",
    "review_pull_quote",
    "linkedin_content",
    "youtube_content",
    "newsletter_blurb",
    "press_release",
    "quote_post",
    "blog_post",
)

# Trailing format tokens to strip so the topic reads cleanly (longest first).
_FORMAT_SUFFIXES = (
    "instagram_post",
    "instagram_reel",
    "instagram_story",
    "linkedin_post_horizontal",
    "linkedin_post_square",
    "blog_cover",
    "youtube_cover",
    "quote_card",
    "square",
    "post",
    "reel",
    "story",
    "cover",
    "horizontal",
    "vertical",
)

_ASPECT_RE = re.compile(r"_(\d+(?:\.\d+)?x\d+(?:\.\d+)?)$")
_CAROUSEL_RE = re.compile(r"_?(carousel_slide_\d+)$")
_DIR_TS_RE = re.compile(r"(\d{8})_(\d{6})")
_SEGMENT_RE = re.compile(r"^\d{3}_")


def _content_type_of(stem: str) -> str:
    for prefix in _CONTENT_TYPE_PREFIXES:
        if stem.startswith(prefix):
            return prefix
    return stem.split("_", 1)[0]


def _timestamp_for(path: Path) -> datetime:
    """Prefer the timestamped parent dir name; fall back to file mtime."""
    match = _DIR_TS_RE.search(path.parent.name)
    if match:
        try:
            return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def parse_image_filename(path: Path) -> dict:
    """Best-effort split of a generated image filename into type/topic/format/aspect."""
    stem = path.stem
    content_type = _content_type_of(stem)
    rest = stem[len(content_type):].lstrip("_") if stem.startswith(content_type) else stem

    aspect = ""
    match = _ASPECT_RE.search(rest)
    if match:
        aspect = match.group(1).replace("x", ":")
        rest = rest[: match.start()]

    fmt = ""
    carousel = _CAROUSEL_RE.search(rest)
    if carousel:
        fmt = carousel.group(1).replace("_", " ")
        rest = rest[: carousel.start()]
    else:
        for suffix in _FORMAT_SUFFIXES:
            if rest == suffix or rest.endswith("_" + suffix):
                fmt = suffix.replace("_", " ")
                rest = rest[: len(rest) - len(suffix)].rstrip("_")
                break

    topic = rest.replace("_", " ").strip().title() or content_type.replace("_", " ").title()
    return {
        "content_type": content_type,
        "topic": topic,
        "format": fmt.title(),
        "aspect": aspect or "—",
    }


def _iter_image_files():
    for name in IMAGE_DIRS:
        base = OUTPUTS_DIR / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                yield path


def image_content_types() -> list[str]:
    """Content types present among generated images, for the filter dropdown."""
    return sorted({_content_type_of(path.stem) for path in _iter_image_files()})


def load_gallery_images(content_type: str = "All types", sort: str = "Newest first") -> list[dict]:
    images: list[dict] = []
    for path in _iter_image_files():
        info = parse_image_filename(path)
        if content_type not in ("All types", "", None) and info["content_type"] != content_type:
            continue
        when = _timestamp_for(path)
        images.append({
            "path": str(path),
            "filename": path.name,
            "content_type": info["content_type"],
            "type_label": info["content_type"].replace("_", " ").title(),
            "topic": info["topic"],
            "format": info["format"],
            "aspect": info["aspect"],
            "date": when.strftime("%Y-%m-%d %H:%M"),
            "_sort": when.timestamp(),
            "size_kb": round(path.stat().st_size / 1024),
        })
    images.sort(key=lambda item: item["_sort"], reverse=(sort != "Oldest first"))
    return images


def _parse_audio_manifest(path: Path) -> dict:
    meta: dict[str, str] = {}
    if not path.exists():
        return meta
    fields = {
        "Mode:": "mode",
        "Document:": "document",
        "Model:": "model",
        "Rendered segments:": "segments",
        "Combined MP3:": "combined",
    }
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        for label, key in fields.items():
            if stripped.startswith(label):
                meta[key] = stripped.split(":", 1)[1].strip()
    return meta


def load_audio_episodes(sort: str = "Newest first") -> list[dict]:
    episodes: list[dict] = []
    if not OUTPUTS_DIR.exists():
        return episodes
    for directory in OUTPUTS_DIR.glob("*_audio_*"):
        if not directory.is_dir():
            continue
        manifest = directory / "podcast_audio_manifest.md"
        meta = _parse_audio_manifest(manifest)

        combined: Path | None = None
        if meta.get("combined") and (directory / meta["combined"]).exists():
            combined = directory / meta["combined"]
        else:
            for mp3 in sorted(directory.glob("*.mp3")):
                if not _SEGMENT_RE.match(mp3.name):
                    combined = mp3
                    break
        if not combined:
            continue

        match = _DIR_TS_RE.search(directory.name)
        when = (
            datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
            if match else datetime.fromtimestamp(directory.stat().st_mtime)
        )
        segment_count = len([m for m in directory.glob("*.mp3") if _SEGMENT_RE.match(m.name)])
        title = meta.get("document") or directory.name
        episodes.append({
            "title": title.replace("_", " ").strip().title(),
            "mode": (meta.get("mode") or "full").title(),
            "model": meta.get("model") or "—",
            "segments": int(meta["segments"]) if str(meta.get("segments", "")).isdigit() else segment_count,
            "mp3_path": str(combined),
            "manifest_path": str(manifest) if manifest.exists() else "",
            "date": when.strftime("%Y-%m-%d %H:%M"),
            "_sort": when.timestamp(),
            "size_mb": round(combined.stat().st_size / 1_000_000, 2),
        })
    episodes.sort(key=lambda item: item["_sort"], reverse=(sort != "Oldest first"))
    return episodes
