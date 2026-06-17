"""Persistent saved draft registry for generated Mythos content."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAFTS_DIR = PROJECT_ROOT / "outputs" / "saved_drafts"
DRAFT_INDEX_PATH = DRAFTS_DIR / "index.json"


def slugify(value: str, fallback: str = "draft") -> str:
    cleaned = (value or "").strip() or fallback
    cleaned = re.sub(r"[^\w\s-]", "", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"[-\s]+", "_", cleaned).strip("_").lower()
    return cleaned[:80] or fallback


def _read_index() -> list[dict[str, Any]]:
    if not DRAFT_INDEX_PATH.exists():
        return []
    try:
        data = json.loads(DRAFT_INDEX_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _write_index(records: list[dict[str, Any]]) -> None:
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    DRAFT_INDEX_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def first_content_line(content: str) -> str:
    for line in (content or "").splitlines():
        cleaned = line.strip().strip("#").strip()
        if cleaned:
            return cleaned[:90]
    return ""


def save_draft(
    *,
    title: str,
    content_type: str,
    content: str,
    source_path: str | Path = "",
    metadata: dict[str, Any] | None = None,
    status: str = "Draft",
) -> dict[str, Any]:
    """Save generated content as a user-facing draft and update the draft index."""

    now = datetime.now().isoformat(timespec="seconds")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    draft_title = (title or "").strip() or first_content_line(content) or f"{content_type} draft"
    draft_type = (content_type or "content").strip() or "content"
    draft_id = f"{timestamp}_{slugify(draft_type)}_{slugify(draft_title)}"
    file_path = DRAFTS_DIR / f"{draft_id}.md"
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content or "", encoding="utf-8")

    record = {
        "id": draft_id,
        "title": draft_title,
        "content_type": draft_type,
        "status": status,
        "created_at": now,
        "updated_at": now,
        "path": str(file_path),
        "source_path": str(source_path or ""),
        "metadata": metadata or {},
    }
    records = [record]
    records.extend(item for item in _read_index() if item.get("id") != draft_id)
    _write_index(records[:250])
    return record


def list_saved_drafts(limit: int = 20) -> list[dict[str, Any]]:
    records = _read_index()
    return records[: max(0, limit)]


def get_saved_draft(draft_id: str) -> dict[str, Any] | None:
    for record in _read_index():
        if record.get("id") == draft_id:
            return record
    return None


def read_saved_draft_content(draft_id: str) -> str:
    record = get_saved_draft(draft_id)
    if not record:
        return ""
    path = Path(str(record.get("path") or ""))
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def update_saved_draft(
    *,
    draft_id: str,
    title: str,
    content: str,
    status: str = "Draft",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update an existing user-facing draft while preserving its registry identity."""

    records = _read_index()
    updated_at = datetime.now().isoformat(timespec="seconds")
    for index, record in enumerate(records):
        if record.get("id") != draft_id:
            continue
        path_text = str(record.get("path") or "").strip()
        path = Path(path_text) if path_text else DRAFTS_DIR / f"{draft_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content or "", encoding="utf-8")
        merged_metadata = dict(record.get("metadata") or {})
        merged_metadata.update(metadata or {})
        updated_record = {
            **record,
            "title": (title or "").strip() or record.get("title") or first_content_line(content) or "Untitled draft",
            "status": status or record.get("status") or "Draft",
            "updated_at": updated_at,
            "path": str(path),
            "metadata": merged_metadata,
        }
        records[index] = updated_record
        records = [updated_record] + [item for item in records if item.get("id") != draft_id]
        _write_index(records[:250])
        return updated_record

    return save_draft(
        title=title,
        content_type="content",
        content=content,
        metadata=metadata,
        status=status,
    )


def delete_saved_draft(draft_id: str) -> bool:
    """Remove a draft from the index and delete its file. Returns True if removed."""
    records = _read_index()
    remaining = [item for item in records if item.get("id") != draft_id]
    if len(remaining) == len(records):
        return False
    for record in records:
        if record.get("id") == draft_id:
            path_text = str(record.get("path") or "").strip()
            if path_text:
                try:
                    Path(path_text).unlink(missing_ok=True)
                except OSError:
                    pass
    _write_index(remaining)
    return True


def archive_saved_draft(draft_id: str) -> dict[str, Any] | None:
    """Mark a draft as Archived without deleting its file."""
    records = _read_index()
    for index, record in enumerate(records):
        if record.get("id") == draft_id:
            record["status"] = "Archived"
            record["updated_at"] = datetime.now().isoformat(timespec="seconds")
            records[index] = record
            _write_index(records)
            return record
    return None


def render_saved_drafts_markdown(limit: int = 12) -> str:
    records = list_saved_drafts(limit)
    if not records:
        return "_No saved drafts yet. Generate content, a campaign, or podcast audio to create one._"

    lines = []
    for record in records:
        title = record.get("title") or "Untitled draft"
        content_type = record.get("content_type") or "content"
        status = record.get("status") or "Draft"
        created = record.get("created_at") or ""
        path = record.get("path") or ""
        lines.extend(
            [
                f"### {title}",
                f"- Type: `{content_type}`",
                f"- Status: {status}",
                f"- Saved: {created}",
                f"- Path: `{path}`",
                "",
            ]
        )
    return "\n".join(lines).strip()
