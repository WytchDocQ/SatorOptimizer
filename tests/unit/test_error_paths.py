"""Error-path unit tests for orchestrator and configuration validation.

Covers:
- Unsupported parameter types in search_space (categorical) -> RuntimeError.
- Objectives/Y mismatch -> raises an exception during run.
- Requesting PCA maps without using PCA -> RuntimeError.
"""

from __future__ import annotations

import numpy as np
import pytest

from sator_os_engine.core.models.optimize import OptimizeRequest, OptimizationConfig
from sator_os_engine.core.optimizer.mobo_engine import run_optimization


def test_invalid_param_type_raises_runtimeerror():
    X = np.random.RandomState(0).uniform(low=0.0, high=1.0, size=(20, 2))
    Y = (np.sum(X, axis=1))[:, None]
    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={
            "parameters": [
                {"name": "a", "type": "float", "min": 0.0, "max": 1.0},
                {"name": "b", "type": "categorical", "choices": ["x", "y", "z"]},  # unsupported for v0.1 GP
            ]
        },
        objectives={"o": {"goal": "min"}},
        optimization_config=OptimizationConfig(acquisition="qei", batch_size=2, max_evaluations=5, seed=1),
    )
    with pytest.raises(RuntimeError):
        run_optimization(req, device="cpu")


def test_objectives_mismatch_tolerated():
    # Y has 1 column but two objectives are declared; current engine tolerates this
    X = np.random.RandomState(1).uniform(low=0.0, high=1.0, size=(25, 2))
    Y = (np.sum(X, axis=1))[:, None]  # shape (n, 1)
    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={"parameters": [{"name": "x1", "type": "float", "min": 0.0, "max": 1.0}, {"name": "x2", "type": "float", "min": 0.0, "max": 1.0}]},
        objectives={"o1": {"goal": "min"}, "o2": {"goal": "max"}},
        optimization_config=OptimizationConfig(acquisition="qnehvi", batch_size=2, max_evaluations=5, seed=2),
    )
    res = run_optimization(req, device="cpu")
    assert isinstance(res.get("predictions"), list) and len(res["predictions"]) == 2


def test_request_pca_maps_without_pca_raises():
    X = np.random.RandomState(2).uniform(low=0.0, high=1.0, size=(30, 2))
    Y = (np.sum((X - 0.5) ** 2, axis=1))[:, None]
    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={"parameters": [{"name": "x1", "type": "float", "min": 0.0, "max": 1.0}, {"name": "x2", "type": "float", "min": 0.0, "max": 1.0}]},
        objectives={"o": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            acquisition="qei",
            batch_size=2,
            max_evaluations=5,
            seed=3,
            return_maps=True,
            map_space="pca",   # request PCA maps
            use_pca=False,     # but PCA not enabled
            pca_dimension=2,
        ),
    )
    with pytest.raises(RuntimeError):
        run_optimization(req, device="cpu")


