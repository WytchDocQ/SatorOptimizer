# Advanced Numerical Settings

All fields live under `optimization_config.advanced` (optional) and are safe to omit.

## GP & Fitting
- kernel: matern52|rbf (default: matern52)
  - Use `matern52` for most physical/industrial problems (allows rougher functions). Use `rbf` when you expect very smooth responses.
- noise: auto|value (default: auto)
  - Defines observation noise variance σ² (same units as Y²). Larger noise yields a smoother, more conservative model.
  - Typical ranges: if Y is standardized to unit variance, set σ² ∈ [1e-4, 1e-2]. If working in raw units, start with σ² ≈ (0.5–5%) of Var(Y).
  - Use `auto` to fit σ² from data (recommended). Use a fixed numeric value when you know your measurement noise or want to stabilize very noisy data.
  - Examples: `noise: 0.001` (on standardized Y) or `noise: 4.0` (if Y variance ≈ 200–500).
- jitter: float (default ~1e-8 in float64)
  - Small diagonal added to the kernel for Cholesky stability. If you see numerical errors ("cholesky"/"not positive definite"), increase jitter ×10.
  - Typical ranges: 1e-8–1e-6 (float64). Too large jitter over-smooths and underfits.
- fit_maxiter: int — max iterations for GP fitting (e.g., 100–500). Higher allows better hyperparameter convergence but costs time.
- fit_lr: float — learning rate for hyperparameter optimization (e.g., 0.05–0.2). Lower is more stable; higher converges faster but may overshoot.

## Acquisition & Optimization
- mc_samples: int — Monte Carlo samples for q* acquisitions.
  - Typical: 128–512. Increase for more stable estimates; cost grows linearly.
- num_restarts: int — multi-starts for acquisition optimization.
  - Typical: 5–20. More restarts improve global optimum chances; cost grows linearly.
- raw_samples: int — Sobol raw samples for initializing restarts.
  - Typical: 128–1024. Rule of thumb: 20–50× problem dimension.
- batch_limit: int — per-iteration batch size for the optimizer’s internal line-search (~5–20). Higher may speed convergence at higher memory cost.
- acq_maxiter: int — acquisition optimizer iterations (50–300). Diminishing returns beyond ~200.
- sequential: bool — propose q candidates one-by-one (`true`) or jointly (`false`).
  - Sequential is slower but can improve diversity in some problems.
- ucb_beta: float — exploration weight for qUCB.
  - Typical: 0.1–10. Larger β explores more; smaller β exploits.
- ei_gamma / pi_gamma: float — offsets for qEI/qPI baselines.
  - Typical: 0–1. Higher values push the method to favor improvements over tiny gains.

## Constraints Handling
- method: feasibility_weight|penalty — (planned) feasibility weighting in acquisition vs. penalty in scalarization.
- penalty_coef: float — penalty scale when using penalties (e.g., 1.0–100.0). Larger penalizes constraint violations more aggressively.

## Reference Point & Fantasization
- ref_point: [..] — explicit reference point for hypervolume (optional).
  - For minimization, choose ref point slightly worse (higher) than the worst observed Y along each objective.
- ref_point_strategy: observed_min_margin (default) — derive ref point from data with a margin.
- pending_as_fantasies: bool — model-as-if pending observations were drawn (default true for async) to reduce duplicate suggestions.

## Devices & Precision
- float64: bool — use double precision (default true). Improves numerical stability of GP solves.
- device: cpu|cuda — also controlled by SATOR_DEVICE env. CUDA accelerates large workloads but double precision on some GPUs is slower.

Example
```json
{
  "optimization_config": {
    "acquisition": "qnehvi",
    "batch_size": 4,
    "advanced": {
      "mc_samples": 256,
      "num_restarts": 8,
      "raw_samples": 256,
      "batch_limit": 8,
      "acq_maxiter": 200,
      "kernel": "matern52",
      "noise": "auto",
      "fit_maxiter": 200
    }
  }
}
```

Note: some settings are placeholders; support will be added progressively. Unrecognized keys are ignored.

