from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from backend.models import Draft, DraftUpdateRequest

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from draft_store import (  # noqa: E402
    delete_saved_draft,
    get_saved_draft,
    list_saved_drafts,
    read_saved_draft_content,
    save_draft,
    update_saved_draft,
)


def to_draft(record: dict[str, Any], include_content: bool = False) -> Draft:
    content = read_saved_draft_content(str(record.get("id") or "")) if include_content else ""
    return Draft(**{**record, "content": content})


def list_drafts(limit: int = 100) -> list[Draft]:
    return [to_draft(record) for record in list_saved_drafts(limit)]


def get_draft(draft_id: str) -> Draft | None:
    record = get_saved_draft(draft_id)
    return to_draft(record, include_content=True) if record else None


def create_draft(*, title: str, content_type: str, content: str, source_path: str | Path = "", metadata: dict[str, Any] | None = None) -> Draft:
    return to_draft(
        save_draft(
            title=title,
            content_type=content_type,
            content=content,
            source_path=source_path,
            metadata=metadata or {},
        ),
        include_content=True,
    )


def update_draft(draft_id: str, request: DraftUpdateRequest) -> Draft:
    return to_draft(
        update_saved_draft(
            draft_id=draft_id,
            title=request.title,
            content=request.content,
            status=request.status,
            metadata=request.metadata,
        ),
        include_content=True,
    )


def delete_draft(draft_id: str) -> bool:
    return delete_saved_draft(draft_id)
