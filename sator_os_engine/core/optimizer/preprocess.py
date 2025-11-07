from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def enforce_sum_to_target_training(X: np.ndarray, sums_cfg: List[Dict[str, Any]] | None) -> np.ndarray:
    if not sums_cfg:
        return X
    Xn = np.asarray(X, dtype=float).copy()
    for sc in sums_cfg:
        idxs = [int(i) for i in sc.get("indices", []) if 0 <= int(i) < Xn.shape[1]]
        if not idxs:
            continue
        target = float(sc.get("target_sum", 1.0))
        sub = Xn[:, idxs]
        totals = np.sum(sub, axis=1, keepdims=True)
        totals_safe = np.where(totals == 0.0, 1.0, totals)
        sub = sub * (target / totals_safe)
        Xn[:, idxs] = sub
    return Xn


def fit_pca_normalize(X: np.ndarray, k: int):
    from sklearn.decomposition import PCA

    k = int(k)
    d_in = X.shape[1]
    # Allow using up to the full input dimension. Previously capped at d_in - 1,
    # which caused a mismatch (e.g., 2D input with requested 2 components fell back to 1),
    # breaking downstream PCA map generation and GP posterior shapes.
    k = max(1, min(k, d_in))
    pca = PCA(n_components=k)
    Z_raw = pca.fit_transform(X)
    pc_mins = np.min(Z_raw, axis=0)
    pc_maxs = np.max(Z_raw, axis=0)
    pc_range = np.maximum(pc_maxs - pc_mins, 1e-12)
    Z_norm = (Z_raw - pc_mins) / pc_range
    return pca, pc_mins, pc_maxs, pc_range, Z_norm


def z_norm_to_input(pca, pc_mins: np.ndarray, pc_range: np.ndarray, z_norm: np.ndarray) -> np.ndarray:
    z_raw = z_norm * pc_range + pc_mins
    return pca.inverse_transform(z_raw)


def input_to_z_norm(pca, pc_mins: np.ndarray, pc_range: np.ndarray, X: np.ndarray) -> np.ndarray:
    z_raw = pca.transform(X)
    return (z_raw - pc_mins) / pc_range


