from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA


def fit_pca(combined: np.ndarray, n_components: int, scaling: Optional[str] = None) -> Tuple[PCA, Dict[str, np.ndarray]]:
    X = combined.astype(float)
    if scaling and scaling.lower() in {"standardize", "minmax"}:
        if scaling.lower() == "standardize":
            mean = X.mean(axis=0)
            std = X.std(axis=0) + 1e-12
            X = (X - mean) / std
            scale_info = {"mode": "standardize", "mean": mean, "std": std}
        else:
            minv = X.min(axis=0)
            maxv = X.max(axis=0)
            rng = (maxv - minv) + 1e-12
            X = (X - minv) / rng
            scale_info = {"mode": "minmax", "min": minv, "range": rng}
    else:
        scale_info = {"mode": "none"}

    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)
    pc_mins = Z.min(axis=0)
    pc_maxs = Z.max(axis=0)
    return pca, {"pc_mins": pc_mins, "pc_maxs": pc_maxs, "scale": scale_info}


def denormalize_coords(coords01: np.ndarray, pc_mins: np.ndarray, pc_maxs: np.ndarray) -> np.ndarray:
    return coords01 * (pc_maxs - pc_mins) + pc_mins


