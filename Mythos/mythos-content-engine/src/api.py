"""Backward-compatible import path for the FastAPI app."""

from backend.main import app, create_app

__all__ = ["app", "create_app"]
