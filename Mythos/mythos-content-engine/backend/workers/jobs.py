from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.models import Job


ProgressCallback = Callable[[float, str], None]
JobHandler = Callable[[ProgressCallback], dict[str, Any] | Awaitable[dict[str, Any]]]


@dataclass
class QueuedJob:
    job_id: str
    handler: JobHandler


_JOBS: dict[str, Job] = {}
_EVENTS: dict[str, asyncio.Event] = {}
_QUEUE: asyncio.Queue[QueuedJob] | None = None
_WORKER_TASK: asyncio.Task | None = None
_JOBS_DIR = Path("outputs") / "jobs"
_JOBS_INDEX = _JOBS_DIR / "index.json"


def _persist_jobs() -> None:
    _JOBS_DIR.mkdir(parents=True, exist_ok=True)
    _JOBS_INDEX.write_text(
        json.dumps({job_id: job.model_dump() for job_id, job in _JOBS.items()}, indent=2),
        encoding="utf-8",
    )


def _load_jobs() -> None:
    if not _JOBS_INDEX.exists():
        return
    try:
        data = json.loads(_JOBS_INDEX.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    for job_id, payload in data.items():
        try:
            _JOBS[job_id] = Job(**payload)
        except Exception:
            continue


def _event(job_id: str) -> asyncio.Event:
    if job_id not in _EVENTS:
        _EVENTS[job_id] = asyncio.Event()
    return _EVENTS[job_id]


def _set_job(job: Job) -> None:
    _JOBS[job.id] = job
    _persist_jobs()
    event = _event(job.id)
    event.set()


def submit_job(message: str, handler: JobHandler) -> Job:
    if _QUEUE is None:
        raise RuntimeError("Job queue has not been started")
    job = Job(id=f"job_{uuid4().hex}", status="queued", message=message, progress=0.0)
    _set_job(job)
    _QUEUE.put_nowait(QueuedJob(job.id, handler))
    return job


def get_job(job_id: str) -> Job | None:
    if job_id not in _JOBS:
        _load_jobs()
    return _JOBS.get(job_id)


def update_job(job_id: str, *, status: str | None = None, message: str | None = None, progress: float | None = None, result: dict[str, Any] | None = None) -> Job:
    current = _JOBS[job_id]
    updated = current.model_copy(
        update={
            "status": status or current.status,
            "message": message if message is not None else current.message,
            "progress": max(0.0, min(1.0, progress if progress is not None else current.progress)),
            "result": result if result is not None else current.result,
        }
    )
    _set_job(updated)
    return updated


async def wait_for_job_change(job_id: str, timeout: float = 15.0) -> Job | None:
    job = get_job(job_id)
    if not job:
        return None
    event = _event(job_id)
    event.clear()
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
    except TimeoutError:
        pass
    return get_job(job_id)


async def _run_handler(queued: QueuedJob) -> None:
    def progress(value: float, message: str) -> None:
        update_job(queued.job_id, status="running", progress=value, message=message)

    try:
        update_job(queued.job_id, status="running", progress=0.02, message="Starting")
        result = queued.handler(progress)
        if inspect.isawaitable(result):
            result = await result
        update_job(queued.job_id, status="complete", progress=1.0, message="Complete", result=result)
    except Exception as exc:
        update_job(queued.job_id, status="failed", progress=1.0, message=str(exc), result={})


async def _worker() -> None:
    assert _QUEUE is not None
    while True:
        queued = await _QUEUE.get()
        try:
            await _run_handler(queued)
        finally:
            _QUEUE.task_done()


async def start_job_worker() -> None:
    global _QUEUE, _WORKER_TASK
    _load_jobs()
    if _QUEUE is None:
        _QUEUE = asyncio.Queue()
    if _WORKER_TASK is None or _WORKER_TASK.done():
        _WORKER_TASK = asyncio.create_task(_worker())


async def stop_job_worker() -> None:
    global _QUEUE, _WORKER_TASK
    if _WORKER_TASK is not None:
        _WORKER_TASK.cancel()
        try:
            await _WORKER_TASK
        except asyncio.CancelledError:
            pass
        _WORKER_TASK = None
    _QUEUE = None


def create_complete_job(message: str, result: dict) -> Job:
    job = Job(id=f"job_{uuid4().hex}", status="complete", message=message, progress=1.0, result=result)
    _set_job(job)
    return job
