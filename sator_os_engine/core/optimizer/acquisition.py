from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from torch.quasirandom import SobolEngine
from botorch.optim.optimize import optimize_acqf

from .utils import feasible_mask as _feasible_mask, enforce_sum_constraints_np as _enforce_sum_constraints_np, build_linear_constraints as _build_linear_constraints


def select_candidates_single_objective(
    *,
    model,
    params: List[Dict[str, Any]],
    bounds_input: torch.Tensor,
    use_pca_model: bool,
    pca,
    pc_mins,
    pc_range,
    n: int,
    rng_seed: int | None,
    tdtype,
    tdevice,
    req,
    Y_np: np.ndarray,
) -> torch.Tensor:
    # Coerce bounds to tensors if lists were passed
    if not hasattr(bounds_input, "shape"):
        bounds_input = torch.tensor(bounds_input, dtype=tdtype, device=tdevice)
    sob = SobolEngine(dimension=bounds_input.shape[1], scramble=True, seed=rng_seed or 0)
    raw_n = 512
    grid01 = sob.draw(raw_n, dtype=tdtype)
    grid = bounds_input[0] + (bounds_input[1] - bounds_input[0]) * grid01
    if use_pca_model and pca is not None:
        grid_np = grid.detach().cpu().numpy()
        Zgrid = torch.tensor(pca.transform(grid_np), dtype=tdtype, device=tdevice)
        post = model.models[0].posterior(Zgrid)
    else:
        post = model.models[0].posterior(grid)
    mu = (-post.mean).detach().cpu().numpy().ravel()
    var = post.variance.detach().cpu().numpy().ravel()

    obj_cfgs = list(req.objectives.values()) if isinstance(req.objectives, dict) else []
    cfg0 = obj_cfgs[0] if obj_cfgs else {}
    goal = str(cfg0.get("goal", "min")).lower()
    target_val = cfg0.get("target_value")
    # Base score by direction
    if goal in ("min", "minimize", "minimize_below", "minimize_above"):
        score = -mu
    elif goal in ("max", "maximize", "maximize_below", "maximize_above"):
        score = mu
    elif goal in ("within_range",):
        score = np.zeros_like(mu)
    elif goal == "target" and target_val is not None:
        tol = float(getattr(req.optimization_config, "target_tolerance", 0.0) or 0.0)
        var_w = float(getattr(req.optimization_config, "target_variance_penalty", 0.05) or 0.05)
        if tol > 0:
            score = -(np.abs(mu - float(target_val)) / tol) + var_w * np.sqrt(var)
        else:
            score = -np.abs(mu - float(target_val)) + var_w * np.sqrt(var)
    else:
        score = -mu
    # Threshold / range shaping (demo-compatible, value scale)
    thr = cfg0.get("threshold") or cfg0.get("thresholds")
    rng = cfg0.get("range") or cfg0.get("ranges")
    if isinstance(thr, dict) and (thr.get("value") is not None):
        t_val = float(thr.get("value"))
        ttype_raw = str(thr.get("type", "")).lower()
        ttype = ttype_raw or ("<=" if ("below" in goal) else ">=" if ("above" in goal) else None)
        wthr = float(thr.get("weight", 0.25))
        if ttype in (">=", ">", "ge", "above"):
            score = score + wthr * np.maximum(mu - t_val, 0.0)
        elif ttype in ("<=", "<", "le", "below"):
            score = score + wthr * np.maximum(t_val - mu, 0.0)
    if isinstance(rng, dict) and (rng.get("min") is not None) and (rng.get("max") is not None):
        a = float(rng.get("min")); b = float(rng.get("max"))
        if a > b:
            a, b = b, a
        wr = float(rng.get("weight", 0.25))
        below = np.maximum(a - mu, 0.0)
        above = np.maximum(mu - b, 0.0)
        penalty = below + above
        score = score - wr * (penalty ** 2)
        if rng.get("ideal") is not None:
            ideal = float(rng.get("ideal"))
            wi = float(rng.get("ideal_weight", rng.get("weight", 0.25)))
            score = score - wi * ((mu - ideal) ** 2)
    # slight variance regularization
    score = score - 0.05 * np.sqrt(var)

    feas = _feasible_mask(grid.detach().cpu().numpy().tolist(), req, params)
    score = np.where(np.array(feas, dtype=bool), score, -np.inf)
    top_idx = np.argsort(score)[-n:][::-1]
    cand_input_np = grid.detach().cpu().numpy().copy()[top_idx]
    cand_input_np = _enforce_sum_constraints_np(cand_input_np, params, req)
    if use_pca_model and pca is not None:
        z_raw = pca.transform(cand_input_np)
        z_norm = (z_raw - pc_mins) / pc_range
        return torch.tensor(z_norm, dtype=tdtype, device=tdevice)
    return torch.tensor(cand_input_np, dtype=tdtype, device=tdevice)


def select_candidates_multiobjective(
    *,
    model,
    params: List[Dict[str, Any]],
    bounds_input: torch.Tensor,
    bounds_model: torch.Tensor,
    use_pca_model: bool,
    pca,
    pc_mins,
    pc_range,
    n: int,
    rng_seed: int | None,
    tdtype,
    tdevice,
    req,
    goals: List[str],
    Y_np: np.ndarray,
) -> torch.Tensor:
    # Coerce bounds to tensors if lists were passed
    if not hasattr(bounds_input, "shape"):
        bounds_input = torch.tensor(bounds_input, dtype=tdtype, device=tdevice)
    if not hasattr(bounds_model, "shape"):
        bounds_model = torch.tensor(bounds_model, dtype=tdtype, device=tdevice)
    has_advanced = any(g not in ("min", "max") for g in goals)
    tY = torch.tensor(Y_np * np.array([1.0 if g == "max" else -1.0 for g in goals], dtype=float), dtype=tdtype, device=tdevice)
    if not has_advanced:
        rp = tY.min(dim=0).values - 0.1 * tY.abs().mean(dim=0).clamp_min(1.0)
        part = NondominatedPartitioning(ref_point=rp.detach().cpu(), Y=tY.detach().cpu())
        acqf = qExpectedHypervolumeImprovement(model=model, ref_point=rp.tolist(), partitioning=part)
        # Only pass linear constraints when optimizing in the original input space.
        # In PCA space (reduced dimension), input-space linear indices do not align.
        d_model = int(bounds_model.shape[1])
        if use_pca_model or d_model != len(params):
            botorch_ineq = None
        else:
            ineq, _ = _build_linear_constraints(req, params)
            botorch_ineq = [
                (
                    torch.tensor(idxs, dtype=torch.long, device=tdevice),
                    torch.tensor(coeffs, dtype=tdtype, device=tdevice),
                    float(rhs),
                )
                for idxs, coeffs, rhs in ineq
            ]
        cand, _ = optimize_acqf(
            acqf,
            bounds=bounds_model,
            q=n,
            num_restarts=8,
            raw_samples=256,
            inequality_constraints=botorch_ineq if botorch_ineq else None,
            options={"batch_limit": 5, "maxiter": 200},
        )
        cand_np = cand.detach().cpu().numpy()
        if use_pca_model and pca is not None:
            z_norm = cand_np
            z_raw = z_norm * pc_range + pc_mins
            x_in = pca.inverse_transform(z_raw)
            x_in = _enforce_sum_constraints_np(x_in, params, req)
            z_norm = (pca.transform(x_in) - pc_mins) / pc_range
            return torch.tensor(z_norm, dtype=tdtype, device=tdevice)
        x_in = _enforce_sum_constraints_np(cand_np, params, req)
        return torch.tensor(x_in, dtype=tdtype, device=tdevice)

    # Advanced: sampling + scoring
    sob = SobolEngine(dimension=bounds_input.shape[1], scramble=True, seed=rng_seed or 0)
    raw_n = 1024
    grid01 = sob.draw(raw_n, dtype=tdtype)
    grid = bounds_input[0] + (bounds_input[1] - bounds_input[0]) * grid01
    if use_pca_model and pca is not None:
        grid_np = grid.detach().cpu().numpy()
        Zgrid = torch.tensor(pca.transform(grid_np), dtype=tdtype, device=tdevice)
        posts = [m.posterior(Zgrid) for m in model.models]
    else:
        posts = [m.posterior(grid) for m in model.models]
    mu_list = [(-p.mean).detach().cpu().numpy().ravel() for p in posts]
    var_list = [p.variance.detach().cpu().numpy().ravel() for p in posts]
    score = np.zeros(raw_n)
    obj_cfgs = list(req.objectives.values()) if isinstance(req.objectives, dict) else []
    for k, cfg_o in enumerate(obj_cfgs):
        goal = str(cfg_o.get("goal", "min")).lower()
        target_val = cfg_o.get("target_value")
        mu = mu_list[k]
        var = var_list[k]
        # base direction
        if goal in ("min", "minimize", "minimize_below", "minimize_above"):
            score_k = -mu
        elif goal in ("max", "maximize", "maximize_below", "maximize_above"):
            score_k = mu
        elif goal == "within_range":
            score_k = np.zeros_like(mu)
        elif goal == "target" and target_val is not None:
            tol = float(getattr(req.optimization_config, "target_tolerance", 0.0) or 0.0)
            var_w = float(getattr(req.optimization_config, "target_variance_penalty", 0.05) or 0.05)
            if tol > 0:
                score_k = -(np.abs(mu - float(target_val)) / tol) + var_w * np.sqrt(var)
            else:
                score_k = -np.abs(mu - float(target_val)) + var_w * np.sqrt(var)
        elif goal in ("explore", "probe"):
            score_k = np.sqrt(var)
        elif goal == "improve":
            best = float(np.min(Y_np))
            score_k = np.maximum(0.0, best - mu)
        else:
            score_k = -mu
        # threshold/range shaping
        thr = cfg_o.get("threshold") or cfg_o.get("thresholds")
        rng = cfg_o.get("range") or cfg_o.get("ranges")
        if isinstance(thr, dict) and (thr.get("value") is not None):
            t_val = float(thr.get("value"))
            ttype_raw = str(thr.get("type", "")).lower()
            ttype = ttype_raw or ("<=" if ("below" in goal) else ">=" if ("above" in goal) else None)
            wthr = float(thr.get("weight", 0.25))
            if ttype in (">=", ">", "ge", "above"):
                score_k = score_k + wthr * np.maximum(mu - t_val, 0.0)
            elif ttype in ("<=", "<", "le", "below"):
                score_k = score_k + wthr * np.maximum(t_val - mu, 0.0)
        if isinstance(rng, dict) and (rng.get("min") is not None) and (rng.get("max") is not None):
            a = float(rng.get("min")); b = float(rng.get("max"))
            if a > b:
                a, b = b, a
            wr = float(rng.get("weight", 0.25))
            below = np.maximum(a - mu, 0.0)
            above = np.maximum(mu - b, 0.0)
            penalty = below + above
            score_k = score_k - wr * (penalty ** 2)
            if rng.get("ideal") is not None:
                ideal = float(rng.get("ideal"))
                wi = float(rng.get("ideal_weight", rng.get("weight", 0.25)))
                score_k = score_k - wi * ((mu - ideal) ** 2)
        score_k = score_k - 0.05 * np.sqrt(var)
        score += score_k
    feas = _feasible_mask(grid.detach().cpu().numpy().tolist(), req, params)
    score = np.where(np.array(feas, dtype=bool), score, -np.inf)
    top_idx = np.argsort(score)[-n:][::-1]
    cand_np = grid.detach().cpu().numpy().copy()[top_idx]
    cand_np = _enforce_sum_constraints_np(cand_np, params, req)
    if use_pca_model and pca is not None:
        z_raw = pca.transform(cand_np)
        z_norm = (z_raw - pc_mins) / pc_range
        return torch.tensor(z_norm, dtype=tdtype, device=tdevice)
    return torch.tensor(cand_np, dtype=tdtype, device=tdevice)


