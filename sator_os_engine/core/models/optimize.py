from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OptimizationConfig(BaseModel):
    algorithm: str = Field("qnehvi")
    acquisition: str = Field("qnehvi")  # qnehvi|qehvi|parego|qei|qpi|qucb
    batch_size: int = Field(4, ge=1)
    max_evaluations: int = Field(100, ge=1)
    seed: Optional[int] = None
    use_pca: bool = False
    pca_dimension: Optional[int] = None
    parameter_scaling: Optional[str] = None  # none|standardize|minmax
    value_normalization: Optional[str] = None  # none|standardize|minmax
    target_tolerance: Optional[float] = None
    target_variance_penalty: Optional[float] = None
    sum_constraints: Optional[List[Dict[str, Any]]] = None  # [{"indices":[0,1,2],"target_sum":1.0}]
    ratio_constraints: Optional[List[Dict[str, Any]]] = None  # [{"i":0,"j":1,"min_ratio":0.5,"max_ratio":2.0}]
    # GP surface/volume maps for visualization (2D/3D only)
    return_maps: bool = False
    map_space: str = Field("input")  # input|pca
    map_resolution: Optional[List[int]] = None  # [nx,ny,(nz)]
    # Advanced numerical settings (optional)
    advanced: Optional[Dict[str, Any]] = None
    # Optional GP configuration and acquisition parameters
    gp_config: Optional[Dict[str, Any]] = None  # e.g., {"lengthscale": [..]|float, "outputscale": float, "noise": float, "ard": bool}
    acquisition_params: Optional[Dict[str, Any]] = None  # e.g., {"ucb_beta": 0.2}
    outputs: Dict[str, Any] = Field(default_factory=dict)


class OptimizeRequest(BaseModel):
    dataset: Dict[str, Any]
    search_space: Dict[str, Any]
    objectives: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None
    optimization_config: OptimizationConfig


class OptimizeResponse(BaseModel):
    job_id: Optional[str] = None
    predictions: Optional[List[Dict[str, Any]]] = None
    pareto: Optional[Dict[str, Any]] = None
    encoding_info: Optional[Dict[str, Any]] = None  # e.g., pc_mins/maxs
    diagnostics: Optional[Dict[str, Any]] = None


