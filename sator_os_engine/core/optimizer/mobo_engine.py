from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from ..models.optimize import OptimizeRequest
from .utils import (
    sample_candidates as _sample_candidates,
    dummy_objective as _dummy_objective,
    pareto_front as _pareto_front,
    build_linear_constraints as _build_linear_constraints,
    feasible_mask as _feasible_mask,
    enforce_sum_constraints_np as _enforce_sum_constraints_np,
    infer_ingredient_and_param_indices as _infer_ingredient_and_param_indices,
)
from .maps import compute_gp_maps


## moved to utils.sample_candidates


## moved to utils.dummy_objective


## moved to utils.pareto_front


## moved to utils.build_linear_constraints


## moved to utils.feasible_mask


## moved to utils.enforce_sum_constraints_np


## moved to utils.infer_ingredient_and_param_indices


def run_optimization(req: OptimizeRequest, device: str = "cpu") -> Dict[str, Any]:
    cfg = req.optimization_config
    num = int(cfg.batch_size)
    rng_seed = cfg.seed

    # Try BoTorch path if dataset provided; otherwise fallback
    dataset = req.dataset or {}
    X = dataset.get("X")
    Y = dataset.get("Y")

    if X is not None and Y is not None:
        try:
            import torch
            from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning

            tdevice = torch.device("cuda") if (device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")
            tdtype = torch.double

            X_np = np.asarray(X, dtype=float)
            params = req.search_space.get("parameters", [])
            sums_cfg = (req.optimization_config.sum_constraints or []) if hasattr(req.optimization_config, "sum_constraints") else []
            from .preprocess import enforce_sum_to_target_training, fit_pca_normalize, z_norm_to_input
            X_np = enforce_sum_to_target_training(X_np, sums_cfg)
            use_pca_model = bool(req.optimization_config.use_pca and (req.optimization_config.pca_dimension or 0) >= 1)
            pca = None
            if use_pca_model:
                k = int(req.optimization_config.pca_dimension or 2)
                pca, pc_mins, pc_maxs, pc_range, Z = fit_pca_normalize(X_np, k)
                X_model = Z
            else:
                X_model = X_np

            tX = torch.tensor(X_model, dtype=tdtype, device=tdevice)
            Y_np = np.asarray(Y, dtype=float)
            obj_cfgs = list(req.objectives.values()) if isinstance(req.objectives, dict) else []
            goals = [str(cfg.get("goal", "min")).lower() for cfg in obj_cfgs] if obj_cfgs else ["min"] * Y_np.shape[1]
            signs = np.array([1.0 if g == "max" else -1.0 for g in goals], dtype=float)
            tY = torch.tensor(Y_np * signs, dtype=tdtype, device=tdevice)
            n_obj = tY.shape[-1]

            from .gp import build_models, bounds_input as gp_bounds_input, bounds_model_pca
            model = build_models(tX, tY, cfg)

            # Bounds from search_space (input space)
            params = req.search_space.get("parameters", [])
            bounds_input = gp_bounds_input(params, tdtype, tdevice)

            # If modeling in PCA, define model bounds from observed Z mins/maxs
            if use_pca_model and pca is not None:
                bounds_model = gp_bounds_model_pca = bounds_model_pca(X_model.shape[1], tdtype, tdevice)
            else:
                bounds_model = bounds_input

            # Ref point for hypervolume (lower than all observed tY for maximize)
            rp = tY.min(dim=0).values - 0.1 * tY.abs().mean(dim=0).clamp_min(1.0)

            # Handle special single-objective goal variants via grid scoring
            obj_cfgs = list(req.objectives.values()) if isinstance(req.objectives, dict) else []
            acq_name = (req.optimization_config.acquisition or "qnehvi").lower()

            from .acquisition import select_candidates_single_objective, select_candidates_multiobjective
            if n_obj == 1:
                cand = select_candidates_single_objective(
                    model=model,
                    params=params,
                    bounds_input=bounds_input,
                    use_pca_model=use_pca_model,
                    pca=pca,
                    pc_mins=pc_mins if use_pca_model and pca is not None else None,
                    pc_range=pc_range if use_pca_model and pca is not None else None,
                    n=num,
                    rng_seed=rng_seed,
                    tdtype=tdtype,
                    tdevice=tdevice,
                    req=req,
                    Y_np=Y_np,
                )
            else:
                goals = [str(cfg.get("goal", "min")).lower() for cfg in (list(req.objectives.values()) if isinstance(req.objectives, dict) else [])]
                cand = select_candidates_multiobjective(
                        model=model,
                    params=params,
                    bounds_input=bounds_input,
                    bounds_model=bounds_model,
                    use_pca_model=use_pca_model,
                    pca=pca,
                    pc_mins=pc_mins if use_pca_model and pca is not None else None,
                    pc_range=pc_range if use_pca_model and pca is not None else None,
                    n=num,
                    rng_seed=rng_seed,
                    tdtype=tdtype,
                    tdevice=tdevice,
                    req=req,
                    goals=goals,
                    Y_np=Y_np,
                    )
                

            # Predict means for candidates (convert back to minimization by negating)
            preds: List[Dict[str, Any]] = []
            for i in range(cand.shape[0]):
                x_input = cand[i:i+1]
                if use_pca_model and pca is not None:
                    # model input is normalized PCA; keep as-is for posterior
                    z_norm = x_input.detach().cpu().numpy()
                    z_np = z_norm * pc_range + pc_mins
                    x_model = x_input
                    x_params = pca.inverse_transform(z_np)
                else:
                    x_model = x_input
                    x_params = x_input.detach().cpu().numpy()
                means = []
                variances = []
                for k, m in enumerate(model.models):
                    post = m.posterior(x_model)
                    mu_t = post.mean.detach().cpu().numpy().ravel()[0]
                    vr = post.variance.detach().cpu().numpy().ravel()[0]
                    mu_orig = mu_t * signs[k]
                    means.append(float(mu_orig))
                    variances.append(float(vr))
                x_params_vec = x_params.reshape(-1)
                cand_dict = {p["name"]: float(x_params_vec[j]) for j, p in enumerate(params)}
                pred_item: Dict[str, Any] = {"candidate": cand_dict, "objectives": means, "variances": variances}
                if use_pca_model and pca is not None:
                    # Include encoded PCA coordinates and reconstructed formulation (sum-to-one respected)
                    enc_coords_raw = z_np[0].tolist()
                    enc_coords_norm = ((z_np[0] - pc_mins) / pc_range).tolist()
                    pred_item["encoded"] = enc_coords_norm
                    try:
                        from ...reconstruction.slsqp_reconstructor import reconstruct as slsqp_reconstruct
                        ing_idx, other_idx = _infer_ingredient_and_param_indices(params, req)
                        ing_bounds = [[float(params[k]["min"]), float(params[k]["max"])] for k in ing_idx]
                        par_bounds = [[float(params[k]["min"]), float(params[k]["max"])] for k in other_idx]
                        sum_target = 1.0
                        sums = (req.optimization_config.sum_constraints or []) if hasattr(req.optimization_config, "sum_constraints") else []
                        if sums:
                            st = sums[0].get("target_sum")
                            if st is not None:
                                sum_target = float(st)
                        rec = slsqp_reconstruct(
                            target_encoded=np.array(enc_coords_raw, dtype=float),
                            encoder_components=pca.components_,
                            encoder_mean=pca.mean_,
                            ingredient_bounds=ing_bounds,
                            parameter_bounds=par_bounds,
                            n_ingredients=len(ing_idx),
                            target_precision=1e-7,
                            sum_target=sum_target,
                            ratio_constraints=None,
                        )
                        pred_item["reconstructed"] = {
                            "ingredients": rec.get("ingredients", []),
                            "parameters": rec.get("parameters", []),
                            "combined": rec.get("solution", []),
                            "success": rec.get("success", False),
                            "final_error": rec.get("final_error", None),
                        }
                    except Exception:
                        pass
                preds.append(pred_item)

            pareto_idx = _pareto_front([p["objectives"] for p in preds])
            pareto = {"indices": pareto_idx, "points": [preds[i]["objectives"] for i in pareto_idx]}

            encoding_info = {"pc_mins": [], "pc_maxs": []} if cfg.use_pca else None
            if use_pca_model and pca is not None:
                try:
                    encoding_info = {
                        "pc_mins": pc_mins.tolist(),
                        "pc_maxs": pc_maxs.tolist(),
                        "components": pca.components_.tolist(),
                        "mean": pca.mean_.tolist(),
                    }
                except Exception:
                    pass

            response: Dict[str, Any] = {
                "predictions": preds,
                "pareto": pareto,
                "encoding_info": encoding_info,
                "diagnostics": {"device": str(tdevice)},
                "objectives": req.objectives if isinstance(req.objectives, dict) else {},
                "encoded_dataset": Z.tolist() if (use_pca_model and pca is not None) else None,
            }

            # Optional GP maps for 2D/3D (delegated)
            if cfg.return_maps:
                try:
                    gp_maps = compute_gp_maps(
                        model=model,
                        cfg=cfg,
                        req=req,
                        params=params,
                        use_pca_model=use_pca_model,
                        pca=pca,
                        Z=Z if use_pca_model and pca is not None else None,
                        X=X,
                        tdtype=tdtype,
                        tdevice=tdevice,
                        signs=signs,
                        pc_mins=pc_mins if (use_pca_model and pca is not None) else None,
                        pc_range=pc_range if (use_pca_model and pca is not None) else None,
                    )
                    if gp_maps:
                        response["gp_maps"] = gp_maps
                except Exception as e:
                    raise RuntimeError(f"failed to compute gp maps: {e}")

            return response
        except Exception as e:
            # No fallback: propagate explicit failure so the job reports the error with traceback
            import traceback as _tb
            raise RuntimeError(f"Optimization failed: {e}: {_tb.format_exc()}")

    # If no dataset was provided, fail explicitly
    raise ValueError("OptimizeRequest must include dataset.X and dataset.Y")


