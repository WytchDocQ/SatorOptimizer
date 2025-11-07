from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict

from .jobs import JobStore, JobStatus


class Executor:
    def __init__(self, store: JobStore, max_workers: int = 4, timeout_sec: int = 300) -> None:
        self.store = store
        self.timeout_sec = timeout_sec
        self._tp = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sator-worker")

    async def submit(self, job_id: str, func: Callable[[], Dict[str, Any]]) -> None:
        loop = asyncio.get_running_loop()
        await self.store.set_status(job_id, JobStatus.RUNNING)

        try:
            result = await asyncio.wait_for(loop.run_in_executor(self._tp, func), timeout=self.timeout_sec)
            await self.store.complete(job_id, result)
        except asyncio.TimeoutError:
            await self.store.fail(job_id, "Job timed out")
        except Exception as e:  # noqa: BLE001
            await self.store.fail(job_id, f"Job failed: {e}")


