from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PCAInfo(BaseModel):
    pc_mins: List[float]
    pc_maxs: List[float]
    components: Optional[List[List[float]]] = None
    mean: Optional[List[float]] = None


class ReconstructionRequest(BaseModel):
    coordinates: List[float]  # normalized [0,1]^D or natural depending on config
    pca_info: Optional[PCAInfo] = None
    bounds: Dict[str, Any]
    n_ingredients: int = Field(ge=0, default=0)
    target_precision: float = Field(1e-7, gt=0)
    sum_target: float = Field(1.0, gt=0)
    ratio_constraints: Optional[List[Dict[str, float]]] = None  # list of {i,j,min_ratio,max_ratio}


class ReconstructionResponse(BaseModel):
    job_id: Optional[str] = None
    success: Optional[bool] = None
    reconstructed_formulation: Optional[Dict[str, List[float]]] = None
    reconstruction_metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


