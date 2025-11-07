from __future__ import annotations

import asyncio
import hmac
import json
from typing import Any, Dict, Optional

from nats.aio.client import Client as NATS

from ..core.models.optimize import OptimizeRequest
from ..core.models.reconstruct import ReconstructionRequest
from ..core.optimizer.mobo_engine import run_optimization
from ..reconstruction.slsqp_reconstructor import reconstruct as slsqp_reconstruct
from ..runtime.jobs import JobStore, JobStatus
from ..runtime.executor import Executor
from ..settings import Settings, get_settings


def _const_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def _get_key_from_headers(headers: Optional[Dict[str, str]]) -> Optional[str]:
    if not headers:
        return None
    key = headers.get("x-api-key")
    if key:
        return key
    auth = headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


async def run_nats(settings: Settings, store: Optional[JobStore] = None) -> None:
    if not settings.enable_nats:
        return

    nc = NATS()
    await nc.connect(servers=[settings.nats_url])

    job_store = store or JobStore(ttl_sec=settings.job_ttl_sec, timeout_sec=settings.job_timeout_sec)
    executor = Executor(job_store, max_workers=settings.concurrency, timeout_sec=settings.job_timeout_sec)

    async def _auth(headers: Optional[Dict[str, str]]) -> Optional[str]:
        provided = _get_key_from_headers(headers)
        if not provided:
            return None
        if settings.api_key and _const_eq(provided, settings.api_key):
            return provided
        return None

    async def on_optimize(msg):  # noqa: ANN001
        api_key = await _auth(msg.headers)
        if not api_key:
            if msg.reply:
                await nc.publish(msg.reply, json.dumps({"error": "unauthorized"}).encode())
            return
        try:
            payload = json.loads(msg.data.decode())
            req = OptimizeRequest(**payload)
        except Exception as e:  # noqa: BLE001
            if msg.reply:
                await nc.publish(msg.reply, json.dumps({"error": f"bad request: {e}"}).encode())
            return

        job = job_store.create_job(owner_key=api_key)
        # Ack with job id
        if msg.reply:
            await nc.publish(msg.reply, json.dumps({"job_id": job.id, "status": JobStatus.QUEUED}).encode())

        def work():
            return run_optimization(req, device=settings.device)

        async def run_and_publish():
            await executor.submit(job.id, work)
            final = job_store.get_job(job.id)
            subj = f"sator.v1.jobs.{job.id}"
            if final and final.status == JobStatus.COMPLETED:
                payload = json.dumps(final.result or {}).encode()
                await nc.publish(subj, payload)
            else:
                payload = json.dumps({"status": str(final.status) if final else "UNKNOWN"}).encode()
                await nc.publish(subj, payload)

        asyncio.create_task(run_and_publish())

    async def on_reconstruct(msg):  # noqa: ANN001
        api_key = await _auth(msg.headers)
        if not api_key:
            if msg.reply:
                await nc.publish(msg.reply, json.dumps({"error": "unauthorized"}).encode())
            return
        try:
            payload = json.loads(msg.data.decode())
            req = ReconstructionRequest(**payload)
        except Exception as e:  # noqa: BLE001
            if msg.reply:
                await nc.publish(msg.reply, json.dumps({"error": f"bad request: {e}"}).encode())
            return

        job = job_store.create_job(owner_key=api_key)
        if msg.reply:
            await nc.publish(msg.reply, json.dumps({"job_id": job.id, "status": JobStatus.QUEUED}).encode())

        def work():
            import numpy as np
            coords = np.array(req.coordinates, dtype=float)
            if req.pca_info and req.pca_info.pc_mins and req.pca_info.pc_maxs:
                pc_mins = np.array(req.pca_info.pc_mins, dtype=float)
                pc_maxs = np.array(req.pca_info.pc_maxs, dtype=float)
                coords = coords * (pc_maxs - pc_mins) + pc_mins

            if not (req.pca_info and req.pca_info.components):
                return {"success": False, "error": "Missing PCA components"}

            components = np.array(req.pca_info.components, dtype=float)
            mean = np.array(req.pca_info.mean, dtype=float) if req.pca_info.mean is not None else None

            ingredient_bounds = req.bounds.get("ingredients", []) if isinstance(req.bounds, dict) else []
            parameter_bounds = req.bounds.get("parameters", []) if isinstance(req.bounds, dict) else []

            res = slsqp_reconstruct(
                target_encoded=coords,
                encoder_components=components,
                encoder_mean=mean,
                ingredient_bounds=ingredient_bounds,
                parameter_bounds=parameter_bounds,
                n_ingredients=req.n_ingredients,
                target_precision=req.target_precision,
            )
            return {
                "success": res["success"],
                "reconstructed_formulation": {
                    "ingredients": res["ingredients"],
                    "parameters": res["parameters"],
                    "combined": res["solution"],
                },
                "reconstruction_metrics": {
                    "final_error": res["final_error"],
                    "iterations": res["iterations"],
                    "method": res["method"],
                },
            }

        async def run_and_publish():
            await executor.submit(job.id, work)
            final = job_store.get_job(job.id)
            subj = f"sator.v1.jobs.{job.id}"
            if final and final.status == JobStatus.COMPLETED:
                await nc.publish(subj, json.dumps(final.result or {}).encode())
            else:
                await nc.publish(subj, json.dumps({"status": str(final.status) if final else "UNKNOWN"}).encode())

        asyncio.create_task(run_and_publish())

    await nc.subscribe("sator.v1.optimize", cb=on_optimize)
    await nc.subscribe("sator.v1.reconstruct", cb=on_reconstruct)

    # Run forever
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await nc.drain()


def run() -> None:
    settings = get_settings()
    asyncio.run(run_nats(settings))



