from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch


def compute_gp_maps(
    *,
    model,
    cfg,
    req,
    params: List[Dict[str, Any]],
    use_pca_model: bool,
    pca,
    Z,
    X,
    tdtype,
    tdevice,
    signs,
    pc_mins=None,
    pc_range=None,
) -> Dict[str, Any] | None:
    """Compute GP posterior mean/variance maps either in PCA or input space.

    Returns a gp_maps dict or None if maps are not applicable.
    Raises RuntimeError on unexpected errors (caller decides policy).
    """
    from sklearn.decomposition import PCA

    # Determine mapping dimension and axes
    cont_idx = [i for i, p in enumerate(params) if p.get("type", "float") in ("float", "int")]
    ndim_input = len(cont_idx)

    map_dim = None
    if cfg.map_space == "pca" and (cfg.pca_dimension in (2, 3)):
        map_dim = int(cfg.pca_dimension)
    elif cfg.map_space == "input" and ndim_input >= 2:
        map_dim = 3 if (ndim_input >= 3 and cfg.map_resolution and len(cfg.map_resolution) == 3) else 2

    if map_dim is None:
        return None

    if cfg.map_space == "pca":
        if not (use_pca_model and pca is not None and pc_mins is not None and pc_range is not None):
            raise RuntimeError("PCA maps requested but PCA was not used")
        res = cfg.map_resolution or ([50, 50] if map_dim == 2 else [32, 32, 32])
        axes = [np.linspace(0.0, 1.0, res[d]) for d in range(map_dim)]
        if map_dim == 2:
            ZZ0, ZZ1 = np.meshgrid(axes[0], axes[1], indexing="xy")
            Zgrid = np.stack([ZZ0.ravel(), ZZ1.ravel()], axis=1)
        else:
            ZZ0, ZZ1, ZZ2 = np.meshgrid(axes[0], axes[1], axes[2], indexing="xy")
            Zgrid = np.stack([ZZ0.ravel(), ZZ1.ravel(), ZZ2.ravel()], axis=1)
        # Model is trained on normalized PCA coordinates (Z in [0,1])
        tXgrid_model = torch.tensor(Zgrid, dtype=tdtype, device=tdevice)
    else:
        res = cfg.map_resolution or ([50, 50] if map_dim == 2 else [32, 32, 32])
        chosen = cont_idx[:map_dim]
        lbs = np.array([float(params[i]["min"]) for i in chosen], dtype=float)
        ubs = np.array([float(params[i]["max"]) for i in chosen], dtype=float)
        axes = [np.linspace(lbs[d], ubs[d], res[d]) for d in range(map_dim)]
        if map_dim == 2:
            A0, A1 = np.meshgrid(axes[0], axes[1], indexing="xy")
            grid_pts = np.stack([A0.ravel(), A1.ravel()], axis=1)
        else:
            A0, A1, A2 = np.meshgrid(axes[0], axes[1], axes[2], indexing="xy")
            grid_pts = np.stack([A0.ravel(), A1.ravel(), A2.ravel()], axis=1)
        mid = [(float(p["min"]) + float(p["max"])) / 2.0 for p in params]
        Xgrid = np.tile(np.array(mid, dtype=float), (grid_pts.shape[0], 1))
        for j, dim in enumerate(chosen):
            Xgrid[:, dim] = grid_pts[:, j]
        tXgrid = torch.tensor(Xgrid, dtype=tdtype, device=tdevice)

    # Evaluate posterior means/vars
    maps_means: Dict[str, Any] = {}
    maps_vars: Dict[str, Any] = {}
    num_pts = (tXgrid_model if cfg.map_space == "pca" else tXgrid).shape[0]
    means_all = []
    vars_all = []
    bs = 4096
    for k, m in enumerate(model.models):
        mus = []
        vs = []
        for s in range(0, num_pts, bs):
            if cfg.map_space == "pca":
                post = m.posterior(tXgrid_model[s : s + bs])
            else:
                post = m.posterior(tXgrid[s : s + bs])
            mu_t = post.mean.detach().cpu().numpy().ravel()
            mus.append(mu_t)
            var = post.variance.detach().cpu().numpy().ravel()
            vs.append(var)
        mu_all = np.concatenate(mus, axis=0) * signs[k]
        var_all = np.concatenate(vs, axis=0)
        means_all.append(mu_all)
        vars_all.append(var_all)

    shape = tuple(res[:map_dim][::-1])
    for idx, key in enumerate([k for k in req.objectives.keys()]):
        maps_means[key] = means_all[idx].reshape(shape).tolist()
        maps_vars[key] = vars_all[idx].reshape(shape).tolist()

    return {
        "space": cfg.map_space,
        "dimension": map_dim,
        "grid": {"axes": [ax.tolist() for ax in axes], "resolution": res},
        "maps": {"means": maps_means, "variances": maps_vars},
    }


