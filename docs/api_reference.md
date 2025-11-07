# API Reference

Headers (HTTP)
- x-api-key: <key>
- Content-Type: application/json

## POST /v1/optimize
- Body: OptimizeRequest
  - optimization_config:
    - acquisition: qnehvi|qehvi|parego|qei|qpi|qucb
    - batch_size, max_evaluations, seed
    - use_pca, pca_dimension (1..D-1). Maps only when pca_dimension ∈ {2,3}.
    - parameter_scaling: none|standardize|minmax
    - value_normalization: none|standardize|minmax
    - target_tolerance: number — tolerance band for goal=target
    - target_variance_penalty: number — variance bonus/penalty weight for target goal
    - sum_constraints: [{ indices: [..], target_sum: 1.0 }]
    - ratio_constraints: [{ i: 0, j: 1, min_ratio: 0.5, max_ratio: 2.0 }]
  - objectives: per-objective config
    - goal: min|max|target|minimize_below|maximize_above|maximize_below|minimize_above|within_range|explore|improve
    - target_value (when goal=target)
    - threshold (for *_below / *_above): { "type": "<=|>=", "value": number, "weight?": number }
    - range (for within_range): { "min": number, "max": number, "ideal?": number, "weight?": number, "ideal_weight?": number }
    - weights (optional, parego scalarization)
  - dataset:
    - X: [[...], ...]
    - Y: [[...], ...] (objectives aligned with objectives order)
  - search_space:
    - parameters: [{ name, type(float|int|categorical), min, max, choices? }]
  - visualization (optional via optimization_config):
    - return_maps: true|false
    - map_space: input|pca (input maps need ≥2 continuous parameters; PCA maps need pca_dimension 2 or 3)
    - map_resolution: [nx,ny,(nz)]

Example
```json
{
  "dataset": {"X": [[0.1, 0.2],[0.7, -0.1]], "Y": [[1.2, 0.5],[0.7, 0.9]]},
  "search_space": {"parameters": [
    {"name": "x1", "type": "float", "min": 0.0, "max": 1.0},
    {"name": "x2", "type": "float", "min": -1.0, "max": 1.0}
  ]},
  "objectives": {"o1": {"goal": "min"}, "o2": {"goal": "target", "target_value": 0.8}},
  "optimization_config": {
    "acquisition": "qnehvi",
    "batch_size": 4,
    "max_evaluations": 50,
    "seed": 42,
    "use_pca": false,
    "pca_dimension": null,
    "parameter_scaling": "none",
    "value_normalization": "none",
    "target_tolerance": 0.5,
    "target_variance_penalty": 0.05,
    "sum_constraints": [{"indices": [0], "target_sum": 1.0}],
    "ratio_constraints": [{"i": 0, "j": 1, "min_ratio": 0.5, "max_ratio": 2.0}],
    "return_maps": true,
    "map_space": "input",
    "map_resolution": [60,60]
  }
}
```

Responses
- 202 Accepted (always async):
```json
{ "job_id": "job_abc123456789", "status": "QUEUED" }
```
- 200 OK (poll result):
```json
{
  "predictions": [
    { "candidate": {"x1": 0.33, "x2": -0.1}, "objectives": [0.91, 0.42], "variances": [0.12, 0.05] }
  ],
  "pareto": { "indices": [0], "points": [[0.91, 0.42]] },
  "encoding_info": null,
  "diagnostics": {"device": "cpu"},
  "gp_maps": {
    "space": "input",
    "dimension": 2,
    "grid": {"axes": [[0.0,0.02,...,1.0],[ -1.0, ... ,1.0 ]], "resolution": [60,60]},
    "maps": {
      "means": {"o1": [[...],[...],...], "o2": [[...],[...],...]},
      "variances": {"o1": [[...],[...],...], "o2": [[...],[...],...]}
    }
  }
}
```

## POST /v1/reconstruct
- Body: ReconstructionRequest
  - pca_info: pc_mins, pc_maxs, components, mean
  - bounds: { ingredients: [[min,max],..], parameters: [[min,max],..] }
  - n_ingredients, target_precision, sum_target, ratio_constraints

Example
```json
{
  "coordinates": [0.5, 0.7],
  "pca_info": {
    "pc_mins": [0.0, 0.0],
    "pc_maxs": [1.0, 1.0],
    "components": [[1,0,0],[0,1,0]],
    "mean": [0,0,0]
  },
  "bounds": {
    "ingredients": [[0,1],[0,1]],
    "parameters": [[0,1]]
  },
  "n_ingredients": 2,
  "sum_target": 1.0,
  "ratio_constraints": [{"i":0,"j":1,"min_ratio":0.5,"max_ratio":2.0}],
  "target_precision": 1e-7
}
```

Responses
- 202 Accepted:
```json
{ "job_id": "job_abc123", "status": "QUEUED" }
```
- 200 OK (poll result):
```json
{
  "success": true,
  "reconstructed_formulation": {
    "ingredients": [0.6, 0.4],
    "parameters": [0.2],
    "combined": [0.6,0.4,0.2]
  },
  "reconstruction_metrics": {"final_error": 1e-7, "iterations": 6, "method": "SLSQP_Constrained"}
}
```

## Jobs
- GET /v1/jobs/{job_id}
- GET /v1/jobs/{job_id}/result

NATS
- Subjects: `sator.v1.optimize`, `sator.v1.reconstruct`
- Send same JSON bodies; include `x-api-key` header; ack returns { job_id }, final result on `sator.v1.jobs.<job_id>`

---

Optimization parameters reference
- acquisition: qnehvi|qehvi|parego|qei|qpi|qucb — MOBO acquisition strategy
- batch_size: integer ≥1 — number of candidates per batch
- max_evaluations: integer ≥1 — overall budget hint
- seed: integer — RNG seed for reproducibility
- use_pca: boolean — whether to fit PCA for modeling/maps
- pca_dimension: 2|3 — required when use_pca=true
- parameter_scaling: none|standardize|minmax — preprocessing of inputs
- value_normalization: none|standardize|minmax — preprocessing of outputs (Y)
- sum_constraints: list of { indices: [int...], target_sum: number } — sum-to-target across selected variables
- ratio_constraints: list of { i: int, j: int, min_ratio?: number, max_ratio?: number } — ratio bounds across variables
- return_maps: boolean — return GP maps for visualization (2D/3D only)
- map_space: input|pca — coordinate system for maps
- map_resolution: [nx,ny,(nz)] — grid resolution for maps

Objective rules
- goal: min|max|target|minimize_below|maximize_above|maximize_below|minimize_above|within_range|explore|improve
- target_value: required when goal=target (goal-attainment)
- threshold / range: see shapes above (per-objective)
- weights: optional — used for ParEGO scalarization

