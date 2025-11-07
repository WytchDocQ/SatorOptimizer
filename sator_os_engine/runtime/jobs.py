from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class Job:
    id: str
    owner_key: str
    status: JobStatus = JobStatus.QUEUED
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())


class JobStore:
    def __init__(self, ttl_sec: int = 600, timeout_sec: int = 300) -> None:
        self.ttl_sec = ttl_sec
        self.timeout_sec = timeout_sec
        self._jobs: Dict[str, Job] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def create_job(self, owner_key: str) -> Job:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = Job(id=job_id, owner_key=owner_key)
        self._jobs[job_id] = job
        self._locks[job_id] = asyncio.Lock()
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        job = self._jobs.get(job_id)
        if not job:
            return None
        if time.time() - job.created_at > self.ttl_sec and job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            # expired
            del self._jobs[job_id]
            return None
        return job

    async def set_status(self, job_id: str, status: JobStatus) -> None:
        job = self._jobs[job_id]
        async with self._locks[job_id]:
            job.status = status
            job.updated_at = time.time()

    async def complete(self, job_id: str, result: Dict[str, Any]) -> None:
        job = self._jobs[job_id]
        async with self._locks[job_id]:
            job.status = JobStatus.COMPLETED
            job.result = result
            job.updated_at = time.time()

    async def fail(self, job_id: str, error: str) -> None:
        job = self._jobs[job_id]
        async with self._locks[job_id]:
            job.status = JobStatus.FAILED
            job.error = error
            job.updated_at = time.time()


