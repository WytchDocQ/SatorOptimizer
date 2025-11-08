"""Unit tests for preprocessing: sum-to-target scaling and PCA normalization.

This module validates:
- enforce_sum_to_target_training: rows are scaled on specified indices to match
  a given target sum.
- fit_pca_normalize: clamps k to [1, d_in], returns normalized Z in [0,1],
  and strictly positive pc_range.
- input_to_z_norm / z_norm_to_input: round-trip conversion recovers inputs with
  small error when k equals the input dimension.
"""

from __future__ import annotations

import numpy as np

from sator_os_engine.core.optimizer.preprocess import (
    enforce_sum_to_target_training,
    fit_pca_normalize,
    input_to_z_norm,
    z_norm_to_input,
)


def test_enforce_sum_to_target_training_scales_rows():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(10, 4))
    sums = [{"indices": [0, 1], "target_sum": 1.0}]
    Xt = enforce_sum_to_target_training(X, sums)
    assert Xt.shape == X.shape
    s = Xt[:, [0, 1]].sum(axis=1)
    assert np.allclose(s, 1.0, atol=1e-6)


def test_fit_pca_normalize_k_bounds_and_ranges():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3))  # d_in=3

    for k in (1, 2, 3, 5):  # 5 will be clamped to 3
        pca, pc_mins, pc_maxs, pc_range, Z_norm = fit_pca_normalize(X, k)
        k_eff = min(max(1, k), X.shape[1])
        assert Z_norm.shape == (X.shape[0], k_eff)
        assert pca.components_.shape[0] == k_eff
        # pc_range strictly positive
        assert np.all(pc_range > 0.0)
        # Z_norm mostly within [0,1] allowing small numeric tolerance
        assert np.all(Z_norm >= -1e-9) and np.all(Z_norm <= 1.0 + 1e-9)


def test_roundtrip_input_to_z_norm_and_back():
    rng = np.random.default_rng(2)
    X = rng.uniform(-2.0, 2.0, size=(20, 3))  # d_in=3
    # Use full dimensionality for near-exact reconstruction
    pca, pc_mins, pc_maxs, pc_range, _ = fit_pca_normalize(X, k=3)
    z = input_to_z_norm(pca, pc_mins, pc_range, X)
    X_rec = z_norm_to_input(pca, pc_mins, pc_range, z)
    # Round-trip should be very close
    assert np.allclose(X_rec, X, atol=1e-6)


