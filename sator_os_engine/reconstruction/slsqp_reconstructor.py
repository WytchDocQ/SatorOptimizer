from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint


def _sum_constraint_factory(n_ingredients: int, target_sum: float):
    def sum_constraint(x: np.ndarray) -> float:
        if n_ingredients <= 0:
            return 0.0
        return float(np.sum(x[:n_ingredients]) - target_sum)

    return sum_constraint


def _combine_bounds(ingredient_bounds: List[List[float]], parameter_bounds: List[List[float]]) -> List[Tuple[float, float]]:
    bounds: List[Tuple[float, float]] = []
    bounds.extend([(float(a), float(b)) for a, b in ingredient_bounds])
    bounds.extend([(float(a), float(b)) for a, b in parameter_bounds])
    return bounds


def reconstruct(
    target_encoded: np.ndarray,
    encoder_components: np.ndarray,
    encoder_mean: Optional[np.ndarray],
    ingredient_bounds: List[List[float]],
    parameter_bounds: List[List[float]],
    n_ingredients: int,
    target_precision: float = 1e-7,
    sum_target: float = 1.0,
    ratio_constraints: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, Any]:
    dim_x = len(ingredient_bounds) + len(parameter_bounds)
    components = np.array(encoder_components, dtype=float)
    mean = np.array(encoder_mean, dtype=float) if encoder_mean is not None else np.zeros(dim_x)

    def encoder_func(x: np.ndarray) -> np.ndarray:
        x2d = np.atleast_2d(x)
        z = (x2d - mean) @ components.T
        return z.squeeze()

    bounds = _combine_bounds(ingredient_bounds, parameter_bounds)
    # Initial guess: mid of bounds
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    x0 = (lb + ub) / 2.0
    if n_ingredients > 0:
        s = np.sum(x0[:n_ingredients])
        if s > 0:
            x0[:n_ingredients] = x0[:n_ingredients] / s * float(sum_target)

    def objective(x: np.ndarray) -> float:
        enc = encoder_func(x)
        return float(np.linalg.norm(enc - target_encoded))

    constraints = []
    if n_ingredients > 0:
        sum_con = _sum_constraint_factory(n_ingredients, float(sum_target))
        constraints.append(NonlinearConstraint(sum_con, 0.0, 0.0))
    # Ratio constraints: x_i/x_j in [min_ratio, max_ratio] -> linear: x_i - max_ratio*x_j <= 0 and -x_i + min_ratio*x_j <= 0
    if ratio_constraints:
        for rc in ratio_constraints:
            i = int(rc.get("i", -1))
            j = int(rc.get("j", -1))
            if i < 0 or j < 0 or i == j:
                continue
            min_ratio = rc.get("min_ratio")
            max_ratio = rc.get("max_ratio")
            n = dim_x
            if max_ratio is not None:
                A = np.zeros((1, n))
                A[0, i] = 1.0
                A[0, j] = -float(max_ratio)
                constraints.append(LinearConstraint(A, -np.inf, 0.0))
            if min_ratio is not None:
                A = np.zeros((1, n))
                A[0, i] = -1.0
                A[0, j] = float(min_ratio)
                constraints.append(LinearConstraint(A, -np.inf, 0.0))

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"ftol": target_precision})

    final_solution = result.x
    final_error = objective(final_solution)

    return {
        "success": bool(result.success),
        "solution": final_solution.tolist(),
        "ingredients": final_solution[:n_ingredients].tolist() if n_ingredients > 0 else [],
        "parameters": final_solution[n_ingredients:].tolist() if n_ingredients < len(final_solution) else [],
        "final_error": final_error,
        "iterations": int(getattr(result, "nit", 0)),
        "method": "SLSQP_Constrained",
    }


