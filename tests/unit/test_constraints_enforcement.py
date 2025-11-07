from __future__ import annotations

import numpy as np

from sator_os_engine.core.models.optimize import OptimizeRequest, OptimizationConfig
from sator_os_engine.core.optimizer.mobo_engine import run_optimization


def test_sum_constraint_enforced_in_candidates():
    X = np.random.RandomState(0).uniform(low=[0.0, 0.0, 0.0], high=[1.0, 1.0, 1.0], size=(30, 3))
    Y = np.sum(X, axis=1, keepdims=True)  # single objective

    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={
            "parameters": [
                {"name": "a", "type": "float", "min": 0.0, "max": 1.0},
                {"name": "b", "type": "float", "min": 0.0, "max": 1.0},
                {"name": "c", "type": "float", "min": 0.0, "max": 1.0},
            ]
        },
        objectives={"o": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            acquisition="qei",
            batch_size=4,
            max_evaluations=10,
            seed=1,
            sum_constraints=[{"indices": [0, 1], "target_sum": 1.0}],
        ),
    )
    res = run_optimization(req, device="cpu")
    for p in res["predictions"]:
        a = p["candidate"]["a"]
        b = p["candidate"]["b"]
        assert abs((a + b) - 1.0) < 1e-2


