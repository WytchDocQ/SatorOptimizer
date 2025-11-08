"""Unit tests for GP map generation in PCA and input spaces.

Validates:
- PCA-space maps: correct dimension, resolution, and presence of mean/variance
  maps per objective when PCA is used.
- Input-space maps: correct shapes for 2D maps over selected input dimensions.
- Error path: requesting PCA maps without PCA in use raises RuntimeError.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import pytest

from sator_os_engine.core.optimizer.preprocess import fit_pca_normalize
from sator_os_engine.core.optimizer.gp import build_models, bounds_input
from sator_os_engine.core.optimizer.maps import compute_gp_maps


def _two_obj_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, 2))
    f1 = np.sum((X - 0.3) ** 2, axis=1)         # min near 0.3
    f2 = -np.sum((X - 0.7) ** 2, axis=1)        # max near 0.7
    Y = np.stack([f1, f2], axis=1)
    return X, Y


def _params_2d():
    return [
        {"name": "x1", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "x2", "type": "float", "min": 0.0, "max": 1.0},
    ]


def test_pca_space_maps_dimension_and_shapes():
    X, Y = _two_obj_data(n=90, seed=1)
    # Fit PCA and normalize
    pca, pc_mins, pc_maxs, pc_range, Z_norm = fit_pca_normalize(X, k=2)
    # Train GP on PCA-normalized space
    tX = torch.tensor(Z_norm, dtype=torch.double)
    tY = torch.tensor(Y, dtype=torch.double)
    model = build_models(tX, tY, SimpleNamespace(gp_config={"noise": 1e-5}))

    params = _params_2d()
    req = SimpleNamespace(objectives={"o1": {"goal": "min"}, "o2": {"goal": "max"}})
    cfg = SimpleNamespace(map_space="pca", pca_dimension=2, map_resolution=[8, 8])

    gm = compute_gp_maps(
        model=model,
        cfg=cfg,
        req=req,
        params=params,
        use_pca_model=True,
        pca=pca,
        Z=tX,
        X=torch.tensor(X, dtype=torch.double),
        tdtype=torch.double,
        tdevice=torch.device("cpu"),
        signs=[1.0, 1.0],
        pc_mins=pc_mins,
        pc_range=pc_range,
    )
    assert gm is not None
    assert gm["space"] == "pca"
    assert gm["dimension"] == 2
    assert gm["grid"]["resolution"] == [8, 8]
    # Means/variances for both objectives with shape (8,8)
    means = gm["maps"]["means"]
    variances = gm["maps"]["variances"]
    assert set(means.keys()) == {"o1", "o2"}
    assert np.array(means["o1"]).shape == (8, 8)
    assert np.array(variances["o2"]).shape == (8, 8)


def test_input_space_maps_dimension_and_shapes():
    X, Y = _two_obj_data(n=80, seed=2)
    tX = torch.tensor(X, dtype=torch.double)
    tY = torch.tensor(Y, dtype=torch.double)
    model = build_models(tX, tY, SimpleNamespace(gp_config={"noise": 1e-5}))

    params = _params_2d()
    req = SimpleNamespace(objectives={"o1": {"goal": "min"}, "o2": {"goal": "max"}})
    cfg = SimpleNamespace(map_space="input", pca_dimension=2, map_resolution=[12, 12])

    gm = compute_gp_maps(
        model=model,
        cfg=cfg,
        req=req,
        params=params,
        use_pca_model=False,
        pca=None,
        Z=tX,
        X=tX,
        tdtype=torch.double,
        tdevice=torch.device("cpu"),
        signs=[1.0, 1.0],
        pc_mins=None,
        pc_range=None,
    )
    assert gm is not None
    assert gm["space"] == "input"
    assert gm["dimension"] == 2
    assert gm["grid"]["resolution"] == [12, 12]
    means = gm["maps"]["means"]
    assert np.array(means["o1"]).shape == (12, 12)


def test_pca_maps_without_pca_raises():
    X, Y = _two_obj_data(n=40, seed=3)
    tX = torch.tensor(X, dtype=torch.double)
    tY = torch.tensor(Y, dtype=torch.double)
    model = build_models(tX, tY, SimpleNamespace(gp_config={"noise": 1e-5}))

    params = _params_2d()
    req = SimpleNamespace(objectives={"o1": {"goal": "min"}, "o2": {"goal": "max"}})
    cfg = SimpleNamespace(map_space="pca", pca_dimension=2, map_resolution=[4, 4])

    with pytest.raises(RuntimeError):
        compute_gp_maps(
            model=model,
            cfg=cfg,
            req=req,
            params=params,
            use_pca_model=False,  # Not using PCA, but requesting PCA maps
            pca=None,
            Z=tX,
            X=tX,
            tdtype=torch.double,
            tdevice=torch.device("cpu"),
            signs=[1.0, 1.0],
            pc_mins=None,
            pc_range=None,
        )


