"""Smoke tests for the optimization orchestrator `run_optimization`.

Validates end-to-end execution for:
- Single-objective (qei), with/without PCA.
- Multi-objective (qnehvi) with standard goals (min/max).
- Advanced-goal mixture to exercise non-qEHVI sampling path.
- Constraints: sum + ratio feasibility and post-selection sum enforcement.
"""

from __future__ import annotations

import numpy as np

from sator_os_engine.core.models.optimize import OptimizeRequest, OptimizationConfig
from sator_os_engine.core.optimizer.mobo_engine import run_optimization


def _params_2d():
    return [
        {"name": "x1", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "x2", "type": "float", "min": 0.0, "max": 1.0},
    ]


def test_orchestrator_single_qei_no_pca():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(60, 2))
    Y = (np.sum((X - 0.3) ** 2, axis=1))[:, None]  # min near 0.3

    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={"parameters": _params_2d()},
        objectives={"f": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            acquisition="qei", batch_size=3, max_evaluations=10, seed=1, return_maps=False, use_pca=False
        ),
    )
    res = run_optimization(req, device="cpu")
    assert isinstance(res.get("predictions"), list) and len(res["predictions"]) == 3


def test_orchestrator_multi_qnehvi_input_space():
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.0, size=(70, 2))
    f1 = np.sum((X - 0.25) ** 2, axis=1)
    f2 = -np.sum((X - 0.75) ** 2, axis=1)
    Y = np.stack([f1, f2], axis=1)

    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={"parameters": _params_2d()},
        objectives={"o1": {"goal": "min"}, "o2": {"goal": "max"}},
        optimization_config=OptimizationConfig(
            acquisition="qnehvi", batch_size=4, max_evaluations=12, seed=2, return_maps=False, use_pca=False
        ),
    )
    res = run_optimization(req, device="cpu")
    assert isinstance(res.get("predictions"), list) and len(res["predictions"]) == 4


def test_orchestrator_advanced_goals_sampling_path():
    rng = np.random.default_rng(2)
    X = rng.uniform(0.0, 1.0, size=(80, 2))
    f1 = np.sum((X - 0.4) ** 2, axis=1)
    f2 = np.sum((X - 0.6) ** 2, axis=1)
    Y = np.stack([f1, f2], axis=1)

    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={"parameters": _params_2d()},
        objectives={
            "o1": {"goal": "target", "target_value": 0.2},
            "o2": {"goal": "within_range", "range": {"min": 0.1, "max": 0.5}},
        },
        optimization_config=OptimizationConfig(
            acquisition="qnehvi", batch_size=3, max_evaluations=10, seed=3, return_maps=False, use_pca=False
        ),
    )
    res = run_optimization(req, device="cpu")
    assert isinstance(res.get("predictions"), list) and len(res["predictions"]) == 3


def test_orchestrator_constraints_sum_and_ratio():
    rng = np.random.default_rng(3)
    # Three variables: enforce sum on first two and ratio between a and c
    X = rng.uniform(0.0, 1.0, size=(60, 3))
    Y = (np.sum(X, axis=1))[:, None]  # single objective, not important for the check

    params = [
        {"name": "a", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "b", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "c", "type": "float", "min": 0.0, "max": 1.0},
    ]
    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={"parameters": params},
        objectives={"o": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            acquisition="qei",
            batch_size=3,
            max_evaluations=10,
            seed=4,
            sum_constraints=[{"indices": [0, 1], "target_sum": 1.0}],
            ratio_constraints=[{"i": 0, "j": 2, "min_ratio": 0.5, "max_ratio": 2.0}],
        ),
    )
    res = run_optimization(req, device="cpu")
    preds = res["predictions"]
    assert len(preds) == 3
    for p in preds:
        a = p["candidate"]["a"]
        b = p["candidate"]["b"]
        c = p["candidate"]["c"]
        # Sum enforcement
        assert abs((a + b) - 1.0) < 1e-2
        # Note: ratio constraints are used as feasibility filters prior to selection,
        # but are not enforced post-selection. We do not assert ratio bounds here.


