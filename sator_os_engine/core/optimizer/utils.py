from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def sample_candidates(search_space: Dict[str, Any], n: int, seed: int | None = None) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    params = search_space.get("parameters", [])
    cands: List[Dict[str, float]] = []
    for _ in range(n):
        cand: Dict[str, float] = {}
        for p in params:
            name = p["name"]
            ptype = p.get("type", "float")
            if ptype in ("float", "int"):
                lo, hi = float(p["min"]), float(p["max"])
                val = rng.uniform(lo, hi)
                cand[name] = float(int(val)) if ptype == "int" else float(val)
            elif ptype == "categorical":
                choices = p.get("choices", [])
                cand[name] = choices[int(rng.integers(0, len(choices)))] if choices else None
        cands.append(cand)
    return cands


def dummy_objective(cand: Dict[str, float]) -> List[float]:
    vals = np.array([v for v in cand.values() if isinstance(v, (int, float))], dtype=float)
    if vals.size == 0:
        return [0.0, 0.0]
    return [float(np.sum(vals)), float(-np.var(vals))]


def pareto_front(points: List[List[float]]) -> List[int]:
    P = np.array(points, dtype=float)
    n = P.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(P[j] <= P[i]) and np.any(P[j] < P[i]):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


def build_linear_constraints(req, params: List[Dict[str, Any]]) -> Tuple[List[Tuple[List[int], List[float], float]], List[Tuple[List[int], List[float], float]]]:
    ineq: List[Tuple[List[int], List[float], float]] = []
    eqpairs: List[Tuple[List[int], List[float], float]] = []
    dim = len(params)
    sums = (req.optimization_config.sum_constraints or []) if hasattr(req.optimization_config, "sum_constraints") else []
    for sc in sums:
        idxs = [int(i) for i in sc.get("indices", []) if 0 <= int(i) < dim]
        target = float(sc.get("target_sum", 0.0))
        if not idxs:
            continue
        coeff = [1.0] * len(idxs)
        coeff_neg = [-1.0] * len(idxs)
        ineq.append((idxs, coeff, target))
        ineq.append((idxs, coeff_neg, -target))
    ratios = (req.optimization_config.ratio_constraints or []) if hasattr(req.optimization_config, "ratio_constraints") else []
    for rc in ratios:
        i = int(rc.get("i", -1))
        j = int(rc.get("j", -1))
        if not (0 <= i < dim and 0 <= j < dim) or i == j:
            continue
        if rc.get("max_ratio") is not None:
            mr = float(rc["max_ratio"])
            ineq.append(([i, j], [1.0, -mr], 0.0))
        if rc.get("min_ratio") is not None:
            mrn = float(rc["min_ratio"])
            ineq.append(([i, j], [-1.0, mrn], 0.0))
    return ineq, eqpairs


def feasible_mask(points: List[List[float]], req, params: List[Dict[str, Any]], tol: float = 1e-6) -> List[bool]:
    X = np.asarray(points, dtype=float)
    dim = X.shape[1]
    mask = np.ones(X.shape[0], dtype=bool)
    sums = (req.optimization_config.sum_constraints or []) if hasattr(req.optimization_config, "sum_constraints") else []
    for sc in sums:
        idxs = [int(i) for i in sc.get("indices", []) if 0 <= int(i) < dim]
        target = float(sc.get("target_sum", 0.0))
        if idxs:
            s = X[:, idxs].sum(axis=1)
            mask &= np.isclose(s, target, atol=tol)
    ratios = (req.optimization_config.ratio_constraints or []) if hasattr(req.optimization_config, "ratio_constraints") else []
    for rc in ratios:
        i = int(rc.get("i", -1))
        j = int(rc.get("j", -1))
        if not (0 <= i < dim and 0 <= j < dim) or i == j:
            continue
        xi = X[:, i]
        xj = X[:, j]
        with np.errstate(divide='ignore', invalid='ignore'):
            r = xi / xj
        if rc.get("max_ratio") is not None:
            mask &= (r <= float(rc["max_ratio"]) + tol)
        if rc.get("min_ratio") is not None:
            mask &= (r >= float(rc["min_ratio"]) - tol)
    return mask.tolist()


def enforce_sum_constraints_np(cands: 'np.ndarray', params: List[Dict[str, Any]], req) -> 'np.ndarray':
    if cands.size == 0:
        return cands
    sums = (req.optimization_config.sum_constraints or []) if hasattr(req.optimization_config, "sum_constraints") else []
    if not sums:
        return cands
    bounds: List[Tuple[float | None, float | None]] = []
    for p in params:
        if p.get("type", "float") in ("float", "int"):
            bounds.append((float(p["min"]), float(p["max"])) )
        else:
            bounds.append((None, None))
    X = np.array(cands, dtype=float, copy=True)
    for sc in sums:
        idxs = [int(i) for i in sc.get("indices", []) if 0 <= int(i) < X.shape[1]]
        if not idxs:
            continue
        target = float(sc.get("target_sum", 0.0))
        sub = X[:, idxs]
        totals = sub.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.where(totals != 0.0, target / totals, 0.0)
        sub = sub * scale
        for j, idx in enumerate(idxs):
            lo, hi = bounds[idx]
            if lo is not None and hi is not None:
                sub[:, j] = np.clip(sub[:, j], lo, hi)
        totals2 = sub.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale2 = np.where(totals2 != 0.0, target / totals2, 0.0)
        sub = sub * scale2
        X[:, idxs] = sub
    return X


def infer_ingredient_and_param_indices(params: List[Dict[str, Any]], req) -> Tuple[List[int], List[int]]:
    dim = len(params)
    ing_set = set()
    sums = (req.optimization_config.sum_constraints or []) if hasattr(req.optimization_config, "sum_constraints") else []
    for sc in sums:
        idxs = [int(i) for i in sc.get("indices", []) if 0 <= int(i) < dim]
        for i in idxs:
            ing_set.add(i)
    ing_idx = sorted(list(ing_set))
    other_idx = [i for i in range(dim) if i not in ing_set]
    return ing_idx, other_idx


