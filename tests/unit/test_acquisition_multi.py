"""Unit tests for multi-objective acquisition paths.

Validates:
- qEHVI path is used for standard goals only (min/max) and passes inequality
  constraints when optimizing in input space, but suppresses them when using
  a PCA-modeled space (dimension mismatch or use_pca_model=True).
- Advanced goals trigger the sampling+scoring path and do not call optimize_acqf.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import pytest

from sator_os_engine.core.optimizer.gp import build_models, bounds_input, bounds_model_pca
from sator_os_engine.core.optimizer.acquisition import select_candidates_multiobjective


def _build_two_obj_model(d=2, n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, d))
    # Two simple objectives
    f1 = np.sum((X - 0.25) ** 2, axis=1)  # minimize near 0.25
    f2 = -np.sum((X - 0.75) ** 2, axis=1)  # maximize near 0.75
    Y = np.stack([f1, f2], axis=1)
    X_t = torch.tensor(X, dtype=torch.double)
    Y_t = torch.tensor(Y, dtype=torch.double)
    model = build_models(X_t, Y_t, SimpleNamespace(gp_config={"noise": 1e-5}))
    return model, X, Y


def test_qehvi_input_space_constraints_passed(monkeypatch: pytest.MonkeyPatch):
    # Model trained in input space (2D)
    model, X, Y = _build_two_obj_model(d=2, n=80, seed=1)
    params = [
        {"name": "x1", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "x2", "type": "float", "min": 0.0, "max": 1.0},
    ]
    tdtype = torch.double
    tdevice = torch.device("cpu")
    b_in = bounds_input(params, tdtype, tdevice)
    # qEHVI path → goals are only min/max
    goals = ["min", "max"]
    # Add a sum-to-target constraint to ensure inequality_constraints are built
    req = SimpleNamespace(
        objectives={"o1": {"goal": "min"}, "o2": {"goal": "max"}},
        optimization_config=SimpleNamespace(
            sum_constraints=[{"indices": [0, 1], "target_sum": 1.0}],
            ratio_constraints=[],
        ),
    )
    captured = {}

    def fake_optimize_acqf(acqf, bounds, q, num_restarts, raw_samples, inequality_constraints=None, options=None):
        captured["inequality_constraints"] = inequality_constraints
        # Return a fixed candidate with correct shape (q, d)
        d_model = bounds.shape[1]
        return torch.full((q, d_model), 0.5, dtype=tdtype), None

    monkeypatch.setattr("sator_os_engine.core.optimizer.acquisition.optimize_acqf", fake_optimize_acqf)

    out = select_candidates_multiobjective(
        model=model,
        params=params,
        bounds_input=b_in,
        bounds_model=b_in,
        use_pca_model=False,
        pca=None,
        pc_mins=None,
        pc_range=None,
        n=3,
        rng_seed=123,
        tdtype=tdtype,
        tdevice=tdevice,
        req=req,
        goals=goals,
        Y_np=Y,
    )
    assert out.shape == (3, 2)
    assert captured.get("inequality_constraints") is not None
    # Expect two inequalities from sum constraint (+target and -target)
    assert len(captured["inequality_constraints"]) == 2


def test_qehvi_pca_constraints_suppressed(monkeypatch: pytest.MonkeyPatch):
    # Model trained in PCA space (k=2), but declare 3 params so d_model != len(params)
    model, X, Y = _build_two_obj_model(d=2, n=60, seed=2)
    params = [
        {"name": "a", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "b", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "c", "type": "float", "min": 0.0, "max": 1.0},
    ]
    tdtype = torch.double
    tdevice = torch.device("cpu")
    b_in = bounds_input(params, tdtype, tdevice)  # not used by qEHVI
    b_model = bounds_model_pca(2, tdtype, tdevice)  # PCA bounds (2D)
    goals = ["min", "max"]
    req = SimpleNamespace(
        objectives={"o1": {"goal": "min"}, "o2": {"goal": "max"}},
        optimization_config=SimpleNamespace(
            sum_constraints=[{"indices": [0, 1], "target_sum": 1.0}],
            ratio_constraints=[],
        ),
    )
    captured = {}

    def fake_optimize_acqf(acqf, bounds, q, num_restarts, raw_samples, inequality_constraints=None, options=None):
        captured["inequality_constraints"] = inequality_constraints
        d_model = bounds.shape[1]
        return torch.full((q, d_model), 0.5, dtype=tdtype), None

    monkeypatch.setattr("sator_os_engine.core.optimizer.acquisition.optimize_acqf", fake_optimize_acqf)

    out = select_candidates_multiobjective(
        model=model,
        params=params,
        bounds_input=b_in,
        bounds_model=b_model,
        use_pca_model=True,
        pca=None,
        pc_mins=None,
        pc_range=None,
        n=3,
        rng_seed=456,
        tdtype=tdtype,
        tdevice=tdevice,
        req=req,
        goals=goals,
        Y_np=Y,
    )
    assert out.shape == (3, 2)
    # Constraints suppressed in PCA-modeled path
    assert captured.get("inequality_constraints") is None


def test_advanced_goals_sampling_path(monkeypatch: pytest.MonkeyPatch):
    # Ensure advanced goals do not call optimize_acqf (sampling+scoring path instead)
    model, X, Y = _build_two_obj_model(d=2, n=70, seed=3)
    params = [
        {"name": "x1", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "x2", "type": "float", "min": 0.0, "max": 1.0},
    ]
    tdtype = torch.double
    tdevice = torch.device("cpu")
    b_in = bounds_input(params, tdtype, tdevice)
    goals = ["min", "within_range"]
    req = SimpleNamespace(
        objectives={"o1": {"goal": "min"}, "o2": {"goal": "within_range", "range": {"min": -0.5, "max": 0.0}}},
        optimization_config=SimpleNamespace(sum_constraints=[], ratio_constraints=[]),
    )

    def raise_if_called(*args, **kwargs):
        raise AssertionError("optimize_acqf should not be called for advanced goals path")

    monkeypatch.setattr("sator_os_engine.core.optimizer.acquisition.optimize_acqf", raise_if_called)

    out = select_candidates_multiobjective(
        model=model,
        params=params,
        bounds_input=b_in,
        bounds_model=b_in,
        use_pca_model=False,
        pca=None,
        pc_mins=None,
        pc_range=None,
        n=3,
        rng_seed=789,
        tdtype=tdtype,
        tdevice=tdevice,
        req=req,
        goals=goals,
        Y_np=Y,
    )
    assert out.shape == (3, 2)


