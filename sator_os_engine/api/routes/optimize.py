from __future__ import annotations

import asyncio
from fastapi import APIRouter, Depends, Header

from ...runtime.executor import Executor
from ...runtime.jobs import JobStatus
from ...core.models.optimize import OptimizeRequest
from ...core.optimizer.mobo_engine import run_optimization
from ..deps import get_job_store, rate_limit, idempotency, get_settings, get_idempotency_store


router = APIRouter()


@router.post("/optimize", status_code=202)
async def submit_optimize(
    payload: OptimizeRequest,
    api_key: str = Depends(rate_limit),
    idem_existing: str | None = Depends(idempotency),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    job_store=Depends(get_job_store),
    settings=Depends(get_settings),
    idem_store=Depends(get_idempotency_store),
):
    # Idempotency shortcut
    if idem_existing:
        return {"job_id": idem_existing}

    job = job_store.create_job(owner_key=api_key)
    if idempotency_key:
        idem_store.put(api_key, idempotency_key, job.id)

    executor = Executor(job_store, max_workers=settings.concurrency, timeout_sec=settings.job_timeout_sec)

    def work():
        return run_optimization(payload, device=settings.device)

    asyncio.create_task(executor.submit(job.id, work))

    return {"job_id": job.id, "status": JobStatus.QUEUED}


