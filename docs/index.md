# SATOR OS Engine Docs

Welcome. This documentation covers how to run the engine, call the API, understand objectives and constraints, and operate it in development/production.

## Start here

- Quickstart
  - `quickstart.md` — install, run, first request
- API reference
  - `api_reference.md` — endpoints and payload shapes (human-readable)
  - `openapi.yaml` — OpenAPI 3.1 source of truth (machine-readable)
- Objectives and constraints
  - `objectives.md` — goal types, thresholds/ranges, exploration, constraints
- Operations
  - `operations.md` — environment config, logging, health, TLS, reverse proxy
  - `local_tls_certs.md` — mkcert setup and running HTTPS locally
- Known bugs and limitations
  - `known_bugs.md`
- Advanced & design (optional)
  - `advanced_numerical_settings.md` — normalization, PCA, reconstruction notes
  - `design/architecture.md`, `design/reconstruction.md`

Tip: Use `.env` to configure the server. For local HTTPS, follow `local_tls_certs.md`; production TLS is best terminated at a reverse proxy (see `operations.md`).

