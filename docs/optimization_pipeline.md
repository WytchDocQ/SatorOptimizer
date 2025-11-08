### Optimization Pipeline

This document explains, end to end, how an optimization request flows through the SATOR engine: what happens, in what order, how data are normalized, how Gaussian Processes (GPs) are trained and queried, how candidates are selected, how constraints are enforced, how optional GP maps are produced, and how reconstruction back to original variables works. For each step, we point to the primary files and functions.


#### 1) Request ingestion
- What arrives: an `OptimizeRequest` JSON with `dataset` (X, Y), `search_space` (parameters, bounds, types), `objectives` (goals/thresholds/ranges), and `optimization_config` (acquisition, batch size, PCA, maps, constraints, seed).
- Where it lands:
  - API: `sator_os_engine/api/routes/optimize.py` → hands off to orchestrator.
  - Orchestrator: `sator_os_engine/core/optimizer/mobo_engine.py` → `run_optimization(...)`.

High level responsibilities in `run_optimization`:
- Parse and validate inputs.
- Prepare data structures (numpy/torch).
- Preprocess inputs (normalization/PCA if enabled).
- Fit GP models.
- Select candidate points using the configured acquisition goal(s).
- Enforce constraints and reconstruct responses.
- Optionally compute GP maps for visualization.


#### 2) Preprocessing and normalization
- Sum-to-target scaling for training data (optional):
  - File: `sator_os_engine/core/optimizer/preprocess.py`
  - Function: `enforce_sum_to_target_training(X, sums_cfg)`
  - Purpose: If some parameters are constrained to sum to a target (e.g., ingredients summing to 1.0), scale those columns row-wise before modeling so the model sees feasible exemplars.

- PCA fit and normalization (optional):
  - File: `sator_os_engine/core/optimizer/preprocess.py`
  - Functions:
    - `fit_pca_normalize(X, k)` → returns `(pca, pc_mins, pc_maxs, pc_range, Z_norm)`
    - `input_to_z_norm(pca, pc_mins, pc_range, X)`
    - `z_norm_to_input(pca, pc_mins, pc_range, z_norm)`
  - Purpose:
    - Reduce dimensionality when configured (`optimization_config.use_pca`, `pca_dimension`).
    - Normalize PCA coordinates to [0, 1] per component for stable GP training.

Inputs forwarded to modeling are either:
- Input space (no PCA): `X` normalized only by optional Standardize transforms in the GP.
- PCA space: normalized PCA coordinates `Z_norm` in [0, 1].


#### 3) Model building (Gaussian Processes)
- File: `sator_os_engine/core/optimizer/gp.py`
- Functions:
  - `build_models(tX, tY, cfg)` → returns a `ModelListGP` (one GP per objective)
  - `bounds_input(params, tdtype, tdevice)` → tensor bounds in input space
  - `bounds_model_pca(k, tdtype, tdevice)` → `[0, 1]` bounds in PCA space
- Details:
  - Uses BoTorch `SingleTaskGP` with `Standardize` outcome transform (one per objective).
  - Fits model hyperparameters via `fit_gpytorch_mll`.
  - Optional GP hyperparameter hints can be passed (`gp_config` in settings).


#### 4) Candidate selection (acquisition)
- File: `sator_os_engine/core/optimizer/acquisition.py`
- Functions:
  - Single objective: `select_candidates_single_objective(...)`
  - Multi objective: `select_candidates_multiobjective(...)`

Two paths for multi-objective selection:
- Standard goals only (`min`/`max`): use qEHVI
  - BoTorch: `qExpectedHypervolumeImprovement`
  - Reference partitioning: `NondominatedPartitioning`
- Advanced goals present (e.g., `target`, `within_range`, `minimize_below`, `maximize_above`, `explore`, `improve`): grid sampling + posterior scoring
  - Posterior means/variances are shaped according to goal semantics (e.g., distance to target, threshold penalties, within-range penalties, or variance preferences).

Single-objective path:
- Uses posterior scoring on a Sobol grid; chooses top-N by score.

Constraint handling during acquisition:
- Linear constraints (sum and ratio) are passed to BoTorch optimization only when optimizing in original input space.
- In PCA space, constraints are not passed to `optimize_acqf` (indices don’t align); instead, feasibility and sum enforcement are applied post-selection.


#### 5) Feasibility and constraint enforcement
- File: `sator_os_engine/core/optimizer/utils.py`
- Functions:
  - `build_linear_constraints(req, params)` → linear inequalities/equalities (input-space indices)
  - `feasible_mask(points, req, params, tol)` → boolean feasibility per point
  - `enforce_sum_constraints_np(cands, params, req)` → adjusts selected candidates to hit target sums while honoring per-parameter bounds

Usage points:
- During acquisition grid scoring: `feasible_mask` filters out infeasible grid points.
- After candidate selection: `enforce_sum_constraints_np` ensures final suggestions meet sum-to-target constraints without violating bounds.


#### 6) Optional GP maps (posterior mean/variance surfaces)
- File: `sator_os_engine/core/optimizer/maps.py`
- Function: `compute_gp_maps(model, cfg, req, params, use_pca_model, pca, Z, X, tdtype, tdevice, signs, pc_mins, pc_range)`
- Behavior:
  - If `optimization_config.return_maps` is true, generate 1D/2D/3D grids in either:
    - PCA space (normalized [0,1] coordinates), or
    - Input space (selected input dimensions across a grid).
  - Evaluate GP posterior means and variances in batches and return structured arrays per objective.


#### 7) Response assembly
- Orchestrator assembles:
  - `predictions`: suggested candidates (in input or PCA-normalized coordinates transformed back to input space), with means/variances and feasibility diagnostics.
  - `gp_maps` (optional): grid axes, mean/variance maps, and metadata.
  - Diagnostics: PCA meta, GP model kernel hints, fit details (when available).


#### 8) Reconstruction (inverse mapping from PCA)
- API: `sator_os_engine/api/routes/reconstruct.py` (routed by the FastAPI app)
- Preprocessing helpers (same as in Step 2):
  - `z_norm_to_input(pca, pc_mins, pc_range, z_norm)`
- Purpose:
  - Given PCA info (components, mean, pc_mins/range) and a point in PCA space (normalized or raw), reconstruct a feasible input vector.
  - For ingredients with sum constraints, use the same enforcement utilities to ensure feasibility.


#### 9) Order of operations summary
1. Parse request in API; call `run_optimization`.
2. Preprocess X:
   - Optional sum-to-target scaling for training (`enforce_sum_to_target_training`).
   - Optional PCA fit and normalization to obtain `Z_norm` (`fit_pca_normalize`).
3. Build GP(s) on the chosen space (`build_models`).
4. Determine bounds for acquisition (`bounds_input` or `bounds_model_pca`).
5. Select candidates:
   - Multi-objective:
     - All min/max → qEHVI; otherwise → sampling + scoring of goal-shaped objectives.
   - Single-objective: sampling + scoring.
6. Apply feasibility filters and sum enforcement to final candidates.
7. Optionally compute and return GP maps in the requested space.
8. Assemble response with predictions, diagnostics, and maps.
9. Separately, reconstruction endpoint maps normalized PCA coordinates back to input space on demand.


#### 10) Key files/functions index
- Orchestrator and flow:
  - `sator_os_engine/core/optimizer/mobo_engine.py` → `run_optimization`
- Preprocessing:
  - `sator_os_engine/core/optimizer/preprocess.py` → `enforce_sum_to_target_training`, `fit_pca_normalize`, `input_to_z_norm`, `z_norm_to_input`
- GP models and bounds:
  - `sator_os_engine/core/optimizer/gp.py` → `build_models`, `bounds_input`, `bounds_model_pca`
- Acquisition:
  - `sator_os_engine/core/optimizer/acquisition.py` → `select_candidates_single_objective`, `select_candidates_multiobjective`
- Constraints and feasibility:
  - `sator_os_engine/core/optimizer/utils.py` → `build_linear_constraints`, `feasible_mask`, `enforce_sum_constraints_np`
- GP maps:
  - `sator_os_engine/core/optimizer/maps.py` → `compute_gp_maps`
- API app and routes:
  - `sator_os_engine/api/app.py` → app factory
  - `sator_os_engine/api/routes/optimize.py` → optimize endpoints
  - `sator_os_engine/api/routes/reconstruct.py` → reconstruct endpoint


