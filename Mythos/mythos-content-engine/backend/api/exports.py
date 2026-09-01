from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.models import Artifact, ExportRequest
from backend.repositories.artifact_repository import decode_artifact_id
from backend.services.export_service import export_content


router = APIRouter(prefix="/api", tags=["exports"])


@router.post("/exports")
def create_export(request: ExportRequest) -> Artifact:
    return export_content(request)


@router.get("/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: str) -> FileResponse:
    try:
        path = decode_artifact_id(artifact_id)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, filename=path.name)
