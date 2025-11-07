from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize


def build_models(tX: torch.Tensor, tY: torch.Tensor, cfg) -> ModelListGP:
    models = []
    for i in range(tY.shape[-1]):
        yi = tY[..., i:i+1]
        m = SingleTaskGP(tX, yi, outcome_transform=Standardize(m=1))
        try:
            cfg_gp = {}
            if hasattr(cfg, "gp_config") and cfg.gp_config:
                cfg_gp = dict(cfg.gp_config)
            elif hasattr(cfg, "advanced") and cfg.advanced and isinstance(cfg.advanced, dict):
                cfg_gp = dict(cfg.advanced.get("gp", {}))
            if cfg_gp:
                ls = cfg_gp.get("lengthscale")
                if ls is not None:
                    if isinstance(ls, (list, tuple, np.ndarray)):
                        ls_t = torch.tensor(ls, dtype=tX.dtype, device=tX.device)
                    else:
                        ls_t = torch.tensor([float(ls)] * tX.shape[-1], dtype=tX.dtype, device=tX.device)
                    try:
                        m.covar_module.base_kernel.lengthscale = ls_t
                    except Exception:
                        pass
                oscale = cfg_gp.get("outputscale")
                if oscale is not None:
                    try:
                        m.covar_module.outputscale = float(oscale)
                    except Exception:
                        pass
                noise = cfg_gp.get("noise")
                if noise is not None:
                    try:
                        m.likelihood.noise_covar.initialize(noise=float(noise))
                    except Exception:
                        pass
        except Exception:
            pass
        mll = ExactMarginalLogLikelihood(m.likelihood, m)
        fit_gpytorch_mll(mll)
        models.append(m)
    return ModelListGP(*models)


def bounds_input(params: List[Dict[str, Any]], tdtype: torch.dtype, tdevice: torch.device) -> torch.Tensor:
    lbs = []
    ubs = []
    for p in params:
        if p.get("type", "float") not in ("float", "int"):
            raise RuntimeError("Only float/int parameters supported in v0.1")
        lbs.append(float(p["min"]))
        ubs.append(float(p["max"]))
    return torch.stack([
        torch.tensor(lbs, dtype=tdtype, device=tdevice),
        torch.tensor(ubs, dtype=tdtype, device=tdevice),
    ])


def bounds_model_pca(k: int, tdtype: torch.dtype, tdevice: torch.device) -> torch.Tensor:
    zmin = torch.zeros(k, dtype=tdtype, device=tdevice)
    zmax = torch.ones(k, dtype=tdtype, device=tdevice)
    return torch.stack([zmin, zmax])


