from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.models import Job
from backend.workers import get_job, wait_for_job_change


router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}")
def job_status(job_id: str) -> Job:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/{job_id}/events")
def job_events(job_id: str) -> StreamingResponse:
    job_status(job_id)

    async def stream():
        while True:
            job = await wait_for_job_change(job_id)
            if not job:
                yield 'event: error\ndata: {"detail":"Job not found"}\n\n'
                return
            yield f"event: status\ndata: {json.dumps(job.model_dump())}\n\n"
            if job.status in {"complete", "failed"}:
                return

    return StreamingResponse(stream(), media_type="text/event-stream")
