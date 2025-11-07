from __future__ import annotations

import numpy as np

from sator_os_engine.core.models.optimize import OptimizeRequest, OptimizationConfig
from sator_os_engine.core.optimizer.mobo_engine import run_optimization


def test_mo_goals_with_pca_maps():
    # Synthetic 2D dataset with 2 objectives
    rng = np.random.default_rng(0)
    X = rng.uniform(low=[0.0, -1.0], high=[1.0, 1.0], size=(50, 2))
    # objective 1: bowl near (0.3,0.0), objective 2: bowl near (0.7,0.5)
    f1 = np.sum((X - np.array([0.3, 0.0])) ** 2, axis=1)
    f2 = np.sum((X - np.array([0.7, 0.5])) ** 2, axis=1)
    Y = np.stack([f1, f2], axis=1)

    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={
            "parameters": [
                {"name": "x1", "type": "float", "min": 0.0, "max": 1.0},
                {"name": "x2", "type": "float", "min": -1.0, "max": 1.0},
            ]
        },
        objectives={
            "o1": {"goal": "target", "target_value": 0.05},
            "o2": {"goal": "min"},
        },
        optimization_config=OptimizationConfig(
            acquisition="qnehvi",
            batch_size=3,
            max_evaluations=20,
            seed=123,
            use_pca=True,
            pca_dimension=2,
            return_maps=True,
            map_space="pca",
            map_resolution=[16, 16],
            target_tolerance=0.05,
            target_variance_penalty=0.05,
        ),
    )

    res = run_optimization(req, device="cpu")
    assert isinstance(res.get("predictions"), list) and len(res["predictions"]) == 3
    # Ensure variances provided
    assert "variances" in res["predictions"][0]
    # Ensure maps present and in 2D
    gm = res.get("gp_maps")
    assert gm is not None
    assert gm.get("dimension") == 2
    assert "means" in gm["maps"] and "variances" in gm["maps"]


