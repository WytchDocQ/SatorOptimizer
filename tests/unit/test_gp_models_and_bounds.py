"""Unit tests for GP model building and bounds helpers.

Validates:
- bounds_input: returns stacked [mins; maxs] tensor with correct shape/values.
- bounds_model_pca: returns [0, 1] bounds of length k.
- build_models: creates one SingleTaskGP per objective, and posteriors can be
  evaluated on input batches without errors (sanity check).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from sator_os_engine.core.optimizer.gp import build_models, bounds_input, bounds_model_pca


def test_bounds_input_tensor_values():
    params = [
        {"name": "x1", "type": "float", "min": -1.0, "max": 2.0},
        {"name": "x2", "type": "int", "min": 0, "max": 10},
    ]
    tdtype = torch.double
    tdevice = torch.device("cpu")
    b = bounds_input(params, tdtype, tdevice)
    assert b.shape == (2, 2)
    mins = b[0].cpu().numpy()
    maxs = b[1].cpu().numpy()
    assert np.allclose(mins, [-1.0, 0.0])
    assert np.allclose(maxs, [2.0, 10.0])


def test_bounds_model_pca_shape_and_values():
    tdtype = torch.double
    tdevice = torch.device("cpu")
    for k in (1, 2, 3):
        b = bounds_model_pca(k, tdtype, tdevice)
        assert b.shape == (2, k)
        assert torch.allclose(b[0], torch.zeros(k, dtype=tdtype))
        assert torch.allclose(b[1], torch.ones(k, dtype=tdtype))


def test_build_models_and_posterior_shapes():
    torch.manual_seed(0)
    # Small synthetic dataset: 2D inputs, 2 objectives
    X = torch.linspace(-1.0, 1.0, 24).reshape(12, 2).to(dtype=torch.double)
    # Objectives: simple quadratic bowls with small noise
    y1 = (X[:, 0] ** 2 + 0.1 * torch.randn(12, dtype=torch.double))
    y2 = ((X[:, 1] - 0.5) ** 2 + 0.1 * torch.randn(12, dtype=torch.double))
    Y = torch.stack([y1, y2], dim=1)  # (12, 2)

    cfg = SimpleNamespace(gp_config={"lengthscale": 0.5, "outputscale": 1.2, "noise": 1e-4})

    model = build_models(X, Y, cfg)
    # One model per objective
    assert hasattr(model, "models") and len(model.models) == 2

    # Posterior sanity: evaluate on a small batch
    Xq = torch.tensor([[-0.5, -0.2], [0.25, 0.75]], dtype=torch.double)
    for m in model.models:
        post = m.posterior(Xq)
        mu = post.mean
        var = post.variance
        assert mu.shape == (2, 1)
        assert var.shape == (2, 1)


