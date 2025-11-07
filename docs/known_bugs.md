# Known bugs and limitations

This page tracks noteworthy issues and gaps. Please open an issue if you hit something not listed here.

- PCA maps are returned only when `use_pca=true` and `pca_dimension ∈ {2,3}`.
- Input-space maps require at least two continuous parameters; categorical-only search spaces won’t render.
- Inequality constraints are not applied when optimizing in PCA space (dimension mismatch); constraints are enforced post-hoc on reconstructed inputs.
- Long-running jobs are bounded by `SATOR_JOB_TIMEOUT_SEC` and will fail explicitly rather than fallback.

