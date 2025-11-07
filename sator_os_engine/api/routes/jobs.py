from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from starlette.status import HTTP_404_NOT_FOUND, HTTP_403_FORBIDDEN

from ...runtime.jobs import JobStatus
from ..deps import get_job_store, get_api_key


router = APIRouter()


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, api_key: str = Depends(get_api_key), job_store=Depends(get_job_store)):
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(HTTP_404_NOT_FOUND, detail="Job not found")
    if job.owner_key != api_key:
        raise HTTPException(HTTP_403_FORBIDDEN, detail="Forbidden")
    return {"job_id": job.id, "status": job.status, "error": job.error}


@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str, api_key: str = Depends(get_api_key), job_store=Depends(get_job_store)):
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(HTTP_404_NOT_FOUND, detail="Job not found")
    if job.owner_key != api_key:
        raise HTTPException(HTTP_403_FORBIDDEN, detail="Forbidden")
    if job.status != JobStatus.COMPLETED:
        return {"status": job.status, "error": job.error}
    return job.result or {}


