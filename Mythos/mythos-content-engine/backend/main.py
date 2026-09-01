from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend.api import audio, drafts, exports, generation, jobs, library, podcasts, voices
from backend.workers import start_job_worker, stop_job_worker


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await start_job_worker()
    try:
        yield
    finally:
        await stop_job_worker()


def create_app() -> FastAPI:
    app = FastAPI(title="TellTales Ink API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:7861",
            "http://localhost:7861",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(generation.router)
    app.include_router(drafts.router)
    app.include_router(podcasts.router)
    app.include_router(audio.router)
    app.include_router(exports.router)
    app.include_router(voices.router)
    app.include_router(jobs.router)
    app.include_router(library.router)

    frontend_dist = PROJECT_ROOT / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    return app


app = create_app()
