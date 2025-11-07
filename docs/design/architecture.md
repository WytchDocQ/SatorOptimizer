# Architecture Notes

- Stateless async-only core with HTTP and optional NATS transports.
- Single request object (`optimization_config`) drives algorithms, PCA, normalizations, constraints.
- In-memory jobs with TTL; no DB in v0.1.
- Optimize path: validate → (optional PCA) → fit GPs → propose candidates with MOBO (or sampling+scoring for advanced goals/PCA) → enforce linear constraints → return suggestions (+ maps/variances optionally).
- Reconstruct path: denormalize PCA coords if needed → SLSQP with sum/ratio constraints → return formulation with metrics.

