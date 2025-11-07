# Objective Goals — Concepts and How They Work

This guide explains objective goal types and how the optimizer interprets them.

## Goals

- min: minimize an objective f(x)
- max: maximize an objective f(x)
- target: drive f(x) toward a target value T (goal-attainment)
- minimize_below: prioritize being below a threshold T (softly penalize above T)
- maximize_above: prioritize being above a threshold T (softly penalize below T)
- maximize_below: maximize while discouraging exceeding a threshold T
- minimize_above: minimize while discouraging going under a threshold T
- within_range: keep f(x) inside [A,B] (soft penalty outside the interval)
- explore / probe: prioritize high-uncertainty regions (variance-driven)
- improve: prioritize immediate improvement over the current best

### target (goal-attainment)
Given target T, the optimizer seeks x that makes f(x) ≈ T. Internally this can be handled via desirability or scalarization (e.g., ParEGO) using |f(x)−T| or a smooth loss. Use when “hit a specific value” is more important than best min/max.

Examples
- Keep viscosity near 1200 cP with tolerance: set goal=target, target_value=1200.
- Hit pH 7.0 while also minimizing cost (multi-objective).

Notes
- Targets interact with other objectives via multi-objective (qEHVI/qNEHVI) or scalarization (ParEGO). Targets are normalized with the rest of Y.

### thresholds and ranges
- minimize_below (T): optimizer prefers solutions where f(x) ≤ T, penalizing f(x) > T.
- maximize_above (T): prefers f(x) ≥ T, penalizing f(x) < T.
- maximize_below (T): prefers high f(x) without exceeding T.
- minimize_above (T): prefers low f(x) without going under T.
- within_range [A,B]: prefers A ≤ f(x) ≤ B; slight exploration bonus may be added.

JSON shape (per-objective):
```json
"objectives": {
  "o": {
    "goal": "within_range",
    "range": { "min": 3.0, "max": 5.0, "ideal": 4.0, "weight": 0.4, "ideal_weight": 0.4 }
  }
}
```

### exploration modes
- explore / probe: encourages sampling where model variance is high to learn the landscape.
- improve: favors points with high expected improvement over current best.

## Combining Multiple Objectives

- Pareto (qEHVI/qNEHVI): treat each goal independently; the optimizer explores tradeoffs.
- ParEGO: scalarize objectives (weighted). Good when you want single best compromise quickly.

## Constraints vs Objectives

- Sum constraints: enforce sum across variables to a target; use sum_constraints for inputs.
- Ratio constraints: enforce x_i/x_j bounds; use ratio_constraints.
- If an objective is better stated as a constraint (e.g., must be ≤ spec), model it as a constraint rather than an objective.

## Normalization and Scaling

- Inputs (X): parameter_scaling none|standardize|minmax — improves GP conditioning.
- Outputs (Y): value_normalization none|standardize|minmax — used before applying targets/scalarization.

## PCA

- If use_pca is true, choose pca_dimension 2 or 3. PCA may be applied for modeling or just for visualization, depending on configuration.


