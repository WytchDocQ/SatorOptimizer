from __future__ import annotations

import asyncio
from fastapi import APIRouter, Depends, Header

from ...runtime.executor import Executor
from ...runtime.jobs import JobStatus
from ...core.models.reconstruct import ReconstructionRequest
from ...reconstruction.slsqp_reconstructor import reconstruct as slsqp_reconstruct
from ..deps import get_job_store, rate_limit, idempotency, get_settings, get_idempotency_store


router = APIRouter()


@router.post("/reconstruct", status_code=202)
async def submit_reconstruct(
    payload: ReconstructionRequest,
    api_key: str = Depends(rate_limit),
    idem_existing: str | None = Depends(idempotency),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    job_store=Depends(get_job_store),
    settings=Depends(get_settings),
    idem_store=Depends(get_idempotency_store),
):
    if idem_existing:
        return {"job_id": idem_existing}

    job = job_store.create_job(owner_key=api_key)
    if idempotency_key:
        idem_store.put(api_key, idempotency_key, job.id)
    executor = Executor(job_store, max_workers=settings.concurrency, timeout_sec=settings.job_timeout_sec)

    def work():
        # Prepare target in natural PCA space
        coords = payload.coordinates
        if payload.pca_info and payload.pca_info.pc_mins and payload.pca_info.pc_maxs:
            import numpy as np
            pc_mins = np.array(payload.pca_info.pc_mins, dtype=float)
            pc_maxs = np.array(payload.pca_info.pc_maxs, dtype=float)
            coords_nat = np.array(coords, dtype=float) * (pc_maxs - pc_mins) + pc_mins
        else:
            import numpy as np
            coords_nat = np.array(coords, dtype=float)

        components = None
        mean = None
        if payload.pca_info and payload.pca_info.components:
            import numpy as np
            components = np.array(payload.pca_info.components, dtype=float)
            mean = np.array(payload.pca_info.mean, dtype=float) if payload.pca_info.mean is not None else None

        ingredient_bounds = payload.bounds.get("ingredients", []) if isinstance(payload.bounds, dict) else []
        parameter_bounds = payload.bounds.get("parameters", []) if isinstance(payload.bounds, dict) else []

        if components is None:
            # Cannot reconstruct without encoder components
            return {
                "success": False,
                "error": "Missing PCA components in pca_info",
            }

        res = slsqp_reconstruct(
            target_encoded=coords_nat,
            encoder_components=components,
            encoder_mean=mean,
            ingredient_bounds=ingredient_bounds,
            parameter_bounds=parameter_bounds,
            n_ingredients=payload.n_ingredients,
            target_precision=payload.target_precision,
            sum_target=payload.sum_target,
            ratio_constraints=payload.ratio_constraints,
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

    asyncio.create_task(executor.submit(job.id, work))

    return {"job_id": job.id, "status": JobStatus.QUEUED}


