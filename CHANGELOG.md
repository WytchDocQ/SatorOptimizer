# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - Unreleased
- Server runs with single API key (`SATOR_API_KEY`); health endpoints `/livez`, `/readyz`.
- Refactor: modular optimizer (`preprocess.py`, `gp.py`, `acquisition.py`, `maps.py`, `utils.py`).
- Removed all fallback logic — failures are explicit and surfaced via job errors.
- PCA workflow: automatic encoded dataset return; predictions include encoded and reconstructed values.
- GP maps: correct PCA-space surface generation for PCA(2); plotted via examples.
- Constraints: sum-to-one (ingredients) and ratio constraints supported.
- Objectives: added threshold/range goals (`minimize_below`, `maximize_above`, `maximize_below`, `minimize_above`, `within_range`), plus `target`, `explore/probe`, `improve`.
- Examples: HTTP Branin (3D) and Chemical PCA (PCA(2) maps). Outputs saved under `examples/responses/`.
- Docs: quickstart, API reference, objectives, operations, how-to-run-examples updated.

## [0.1.0]
- Initial open-source codebase scaffolding
- HTTP server with health endpoints
- Single API key auth (`SATOR_API_KEY`)
- Basic optimization and reconstruction scaffolding

