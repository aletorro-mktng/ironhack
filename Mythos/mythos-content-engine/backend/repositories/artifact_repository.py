from __future__ import annotations

import base64
from pathlib import Path

from backend.models import Artifact


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def encode_artifact_id(path: str | Path) -> str:
    resolved = Path(path).resolve()
    return base64.urlsafe_b64encode(str(resolved).encode("utf-8")).decode("ascii").rstrip("=")


def decode_artifact_id(artifact_id: str) -> Path:
    padded = artifact_id + "=" * (-len(artifact_id) % 4)
    path = Path(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("Artifact is outside the project") from exc
    return path


def artifact_for_path(path: str | Path, media_type: str = "application/octet-stream") -> Artifact:
    resolved = Path(path).resolve()
    return Artifact(
        id=encode_artifact_id(resolved),
        path=str(resolved),
        filename=resolved.name,
        media_type=media_type,
    )
