"""Small data adapters shared by the API and frontend-facing tests."""

from __future__ import annotations


def selected_values(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text.lower() in {"not specified", "not applicable"}:
        return []
    return [part.strip() for part in text.split(",") if part.strip()] or [text]


def normalize_multi_select(value, fallback=None) -> list[str]:
    values = selected_values(value)
    return values if values else selected_values(fallback)
