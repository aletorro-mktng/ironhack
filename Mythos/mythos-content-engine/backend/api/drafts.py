from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models import Draft, DraftCreateRequest, DraftUpdateRequest
from backend.services import draft_service


router = APIRouter(prefix="/api/drafts", tags=["drafts"])


@router.get("")
def list_drafts(limit: int = 100) -> list[Draft]:
    return draft_service.list_drafts(limit)


@router.get("/{draft_id}")
def get_draft(draft_id: str) -> Draft:
    draft = draft_service.get_draft(draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    return draft


@router.post("")
def create_draft(request: DraftCreateRequest) -> Draft:
    return draft_service.create_draft(
        title=request.title,
        content_type=request.content_type,
        content=request.content,
        source_path=request.source_path,
        metadata=request.metadata,
    )


@router.patch("/{draft_id}")
def patch_draft(draft_id: str, request: DraftUpdateRequest) -> Draft:
    return draft_service.update_draft(draft_id, request)


@router.put("/{draft_id}")
def put_draft(draft_id: str, request: DraftUpdateRequest) -> Draft:
    return patch_draft(draft_id, request)


@router.delete("/{draft_id}")
def delete_draft(draft_id: str) -> dict[str, bool]:
    return {"deleted": draft_service.delete_draft(draft_id)}
