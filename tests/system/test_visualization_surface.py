"""System test generating a Branin surface visualization artifact.

This test requires matplotlib (skipped if unavailable). It:
- Builds a small training set for the Branin function.
- Runs the optimizer to get a batch of predictions.
- Renders a contour plot of the ground-truth surface with training points and
  predicted candidates, saving the figure under tests/artifacts/.

It asserts that an image file is written and is non-trivial in size. The goal
is to validate the end-to-end coupling of optimization output with a simple
visualization flow, not to assess numerical optimality.
"""

from __future__ import annotations

import os
import math
import numpy as np
import pytest

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

from sator_os_engine.core.models.optimize import OptimizeRequest, OptimizationConfig
from sator_os_engine.core.optimizer.mobo_engine import run_optimization


def _branin(x: np.ndarray) -> np.ndarray:
    # x: (..., 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    a = 1.0
    b = 5.1 / (4.0 * math.pi ** 2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    return (a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s).astype(float)


@pytest.mark.skipif(not _HAS_MPL, reason="matplotlib not installed")
def test_visualize_branin_predictions(tmp_path):
    # Training dataset (random in the domain)
    rng = np.random.default_rng(0)
    bounds = np.array([[-5.0, 10.0], [0.0, 15.0]], dtype=float)  # (2,2)
    n_train = 80
    u = rng.uniform(low=0.0, high=1.0, size=(n_train, 2))
    X = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * u
    Y = _branin(X)[:, None]  # single objective, minimize

    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={
            "parameters": [
                {"name": "x1", "type": "float", "min": float(bounds[0, 0]), "max": float(bounds[0, 1])},
                {"name": "x2", "type": "float", "min": float(bounds[1, 0]), "max": float(bounds[1, 1])},
            ]
        },
        objectives={"f": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            acquisition="qnehvi",
            batch_size=5,
            max_evaluations=20,
            seed=123,
            return_maps=False,
        ),
    )

    res = run_optimization(req, device="cpu")
    assert isinstance(res.get("predictions"), list) and len(res["predictions"]) == 5

    # Prepare contour grid
    gx, gy = np.meshgrid(
        np.linspace(bounds[0, 0], bounds[0, 1], 120),
        np.linspace(bounds[1, 0], bounds[1, 1], 120),
        indexing="xy",
    )
    g = np.stack([gx.ravel(), gy.ravel()], axis=1)
    gz = _branin(g).reshape(gx.shape)

    # Extract predicted candidates
    preds = res["predictions"]
    P = np.array([[p["candidate"]["x1"], p["candidate"]["x2"]] for p in preds], dtype=float)

    # Plot & save
    plt.figure(figsize=(6, 5), dpi=120)
    cs = plt.contourf(gx, gy, gz, levels=20, cmap="viridis")
    plt.colorbar(cs, shrink=0.85)
    plt.scatter(X[:, 0], X[:, 1], s=10, c="white", alpha=0.5, label="train")
    plt.scatter(P[:, 0], P[:, 1], s=40, c="red", marker="x", label="pred")
    plt.legend(loc="upper right")
    out_dir = os.path.join("tests", "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "visual_branin.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 1000


