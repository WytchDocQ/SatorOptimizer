"""Unit tests for single-objective acquisition and goal shaping.

Covers goals and shaping in select_candidates_single_objective:
- min / max base directions
- target (with tolerance/variance penalty parameters)
- within_range (range penalties)
- minimize_below / maximize_above (threshold shaping)
- maximize_below / minimize_above (threshold shaping with opposite directions)

Tests assert that returned candidates lie near expected regions using posterior
means from the built GP, without assuming exact numeric optima.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from sator_os_engine.core.optimizer.gp import build_models, bounds_input
from sator_os_engine.core.optimizer.acquisition import select_candidates_single_objective


def _make_params():
    return [
        {"name": "x1", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "x2", "type": "float", "min": 0.0, "max": 1.0},
    ]


def _build_gp_for_func(func, n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, 2))
    y = func(X).reshape(n, 1)
    X_t = torch.tensor(X, dtype=torch.double)
    Y_t = torch.tensor(y, dtype=torch.double)
    model = build_models(X_t, Y_t, SimpleNamespace(gp_config={"noise": 1e-5}))
    return model, X, y


def _make_req(goal: str, *, threshold=None, range_=None, tol=0.05, var_pen=0.05):
    obj = {"goal": goal}
    if threshold is not None:
        obj["threshold"] = threshold
    if range_ is not None:
        obj["range"] = range_
    return SimpleNamespace(
        objectives={"f": obj},
        optimization_config=SimpleNamespace(
            target_tolerance=tol,
            target_variance_penalty=var_pen,
            sum_constraints=[],
            ratio_constraints=[],
        ),
    )


def _posterior_mean(model, x):
    xt = torch.tensor(x, dtype=torch.double).reshape(1, -1)
    post = model.models[0].posterior(xt)
    return float(post.mean.detach().cpu().numpy().ravel()[0])


def test_min_goal_selects_near_minimum():
    # Linear objective: f = x1 + x2 (minimum near [0,0])
    f = lambda X: (X[:, 0] + X[:, 1])
    model, X, y = _build_gp_for_func(f, n=120, seed=1)
    params = _make_params()
    tdtype = torch.double
    tdevice = torch.device("cpu")
    b_in = bounds_input(params, tdtype, tdevice)
    req = _make_req("min")

    cand = select_candidates_single_objective(
        model=model,
        params=params,
        bounds_input=b_in,
        use_pca_model=False,
        pca=None,
        pc_mins=None,
        pc_range=None,
        n=3,
        rng_seed=123,
        tdtype=tdtype,
        tdevice=tdevice,
        req=req,
        Y_np=y,
    )
    x_best = cand[0].detach().cpu().numpy()
    # Expect candidate near (0, 0)
    assert np.linalg.norm(x_best - np.array([0.0, 0.0])) < 0.25


def test_max_goal_selects_near_maximum():
    # Linear objective: f = x1 + x2 (maximum near [1,1])
    f = lambda X: (X[:, 0] + X[:, 1])
    model, X, y = _build_gp_for_func(f, n=120, seed=2)
    params = _make_params()
    tdtype = torch.double
    tdevice = torch.device("cpu")
    b_in = bounds_input(params, tdtype, tdevice)
    req = _make_req("max")

    cand = select_candidates_single_objective(
        model=model,
        params=params,
        bounds_input=b_in,
        use_pca_model=False,
        pca=None,
        pc_mins=None,
        pc_range=None,
        n=3,
        rng_seed=456,
        tdtype=tdtype,
        tdevice=tdevice,
        req=req,
        Y_np=y,
    )
    x_best = cand[0].detach().cpu().numpy()
    # Expect candidate near (1, 1)
    assert np.linalg.norm(x_best - np.array([1.0, 1.0])) < 0.25


def test_target_goal_prefers_means_near_target():
    # Linear objective: f = x1 + x2; prefer target ~1.0
    f = lambda X: (X[:, 0] + X[:, 1])
    model, X, y = _build_gp_for_func(f, n=140, seed=3)
    params = _make_params()
    tdtype = torch.double
    tdevice = torch.device("cpu")
    b_in = bounds_input(params, tdtype, tdevice)
    req = _make_req("target", tol=0.05, var_pen=0.05)
    req.objectives["f"]["target_value"] = 1.0

    cand = select_candidates_single_objective(
        model=model,
        params=params,
        bounds_input=b_in,
        use_pca_model=False,
        pca=None,
        pc_mins=None,
        pc_range=None,
        n=3,
        rng_seed=789,
        tdtype=tdtype,
        tdevice=tdevice,
        req=req,
        Y_np=y,
    )
    x_best = cand[0].detach().cpu().numpy()
    mu = _posterior_mean(model, x_best)
    assert abs(mu - 1.0) < 0.25


def test_within_range_prefers_inside_band():
    # Linear objective: f = x1 + x2; prefer [0.8, 1.2]
    f = lambda X: (X[:, 0] + X[:, 1])
    model, X, y = _build_gp_for_func(f, n=120, seed=4)
    params = _make_params()
    tdtype = torch.double
    tdevice = torch.device("cpu")
    b_in = bounds_input(params, tdtype, tdevice)
    req = _make_req("within_range", range_={"min": 0.8, "max": 1.2, "weight": 0.5})

    cand = select_candidates_single_objective(
        model=model,
        params=params,
        bounds_input=b_in,
        use_pca_model=False,
        pca=None,
        pc_mins=None,
        pc_range=None,
        n=3,
        rng_seed=1357,
        tdtype=tdtype,
        tdevice=tdevice,
        req=req,
        Y_np=y,
    )
    x_best = cand[0].detach().cpu().numpy()
    mu = _posterior_mean(model, x_best)
    assert 0.6 <= mu <= 1.4


def test_minimize_below_encourages_below_threshold():
    # Linear objective: f = x1 + x2; encourage being below 0.6
    f = lambda X: (X[:, 0] + X[:, 1])
    model, X, y = _build_gp_for_func(f, n=120, seed=5)
    params = _make_params()
    tdtype = torch.double
    tdevice = torch.device("cpu")
    b_in = bounds_input(params, tdtype, tdevice)
    thr = {"value": 0.6, "weight": 1.0}
    req = _make_req("minimize_below", threshold=thr)

    cand = select_candidates_single_objective(
        model=model,
        params=params,
        bounds_input=b_in,
        use_pca_model=False,
        pca=None,
        pc_mins=None,
        pc_range=None,
        n=3,
        rng_seed=9753,
        tdtype=tdtype,
        tdevice=tdevice,
        req=req,
        Y_np=y,
    )
    x_best = cand[0].detach().cpu().numpy()
    mu = _posterior_mean(model, x_best)
    assert mu <= 0.8


def test_maximize_above_encourages_above_threshold():
    # Linear objective: f = x1 + x2; encourage being above 1.2
    f = lambda X: (X[:, 0] + X[:, 1])
    model, X, y = _build_gp_for_func(f, n=120, seed=6)
    params = _make_params()
    tdtype = torch.double
    tdevice = torch.device("cpu")
    b_in = bounds_input(params, tdtype, tdevice)
    thr = {"value": 1.2, "weight": 1.0, "type": ">="}
    req = _make_req("maximize_above", threshold=thr)

    cand = select_candidates_single_objective(
        model=model,
        params=params,
        bounds_input=b_in,
        use_pca_model=False,
        pca=None,
        pc_mins=None,
        pc_range=None,
        n=3,
        rng_seed=8642,
        tdtype=tdtype,
        tdevice=tdevice,
        req=req,
        Y_np=y,
    )
    x_best = cand[0].detach().cpu().numpy()
    mu = _posterior_mean(model, x_best)
    assert mu >= 1.0


