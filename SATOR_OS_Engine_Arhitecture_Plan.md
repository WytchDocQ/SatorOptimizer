## SATOR Open Source Engine — Architecture Plan

### 1) Purpose and Scope
- **Goal**: Provide a production-grade, black-box, multi-objective Bayesian optimization engine as a standalone Python server with a clean HTTP API and optional managed evaluation workers.
- **Use cases**: Parameter tuning, experiment design, process optimization, chemical/materials formulation, and other expensive black-box problems with multiple competing objectives and optional constraints.
- **Non-goals (v1)**: Not a general workflow orchestrator; not an autoscaler; no UI (server-only).

### 2) Core Capabilities
- **Optimization**: Multi-objective Bayesian Optimization (MOBO) with qEHVI/qNEHVI (BoTorch) and ParEGO fallback; constraint handling.
- **Spaces**: Continuous, integer, categorical, mixed; transforms/warping.
- **Stateless operations**: Async-only job processing for optimize and reconstruct.
- **Optional PCA**: Fit PCA from request data; include normalization info in response; client supplies PCA info for reconstruction.
- **Observability**: Logging (console/file) and optional metrics.

### 3) System Overview
- **Client** (SDK/transport): Sends Optimize/Reconstruct requests; receives job_id then polls or subscribes for results.
- **Transports**: HTTP (FastAPI) and/or NATS, each optional via env flags.
- **Execution Runtime**: Internal asyncio/thread/process pools; in-memory job queue and job store with TTL.
- **Optimizer**: BoTorch/PyTorch; stateless per request.
- **Observability**: Logging (console/file); optional Prometheus metrics.

### 4) Component Design
1) HTTP API (Optional, FastAPI)
   - Endpoints (async-only):
     - POST /v1/optimize → 202 { job_id }
     - POST /v1/reconstruct → 202 { job_id }
     - GET /v1/jobs/{job_id} → { status, progress?, eta? }
     - GET /v1/jobs/{job_id}/result → final payload (until TTL)
     - DELETE /v1/jobs/{job_id} → cancel (optional)
   - Auth: Single-class API key via headers (no scopes, no multi-tenant).
   - Validation: Pydantic v2 models; strict schema with enums.
   - Behavior driven by single `optimization_config` in requests (encoding, algorithms, outputs, normalizations).

2) Optimizer Service
   - BoTorch (primary), CPU/GPU via PyTorch; qEHVI/qNEHVI and ParEGO fallback; constraints supported.
   - Stateless: fit and evaluate per OptimizeRequest; return predictions/Pareto/diagnostics; GPU optional.

3) Orchestrator & Workers
   - Celery with Redis broker/back-end for idempotent tasks:
     - run_optimizer_iteration(study_id)
     - suggest_batch(study_id, batch_size)
     - run_managed_evaluations(study_id, candidates)
   - Concurrency controls, retries, backoff. Optional Flower dashboard for visibility.


5) Storage Layer
   - SQLModel/SQLAlchemy for schema; alembic migrations.
   - Entities: Project, Study, Space, Parameter, Objective, Constraint, Trial (candidate + status), Observation, Artifact, ParetoFront, ModelCheckpoint.
   - Default SQLite; Postgres for production; S3 (or local FS) for artifacts.

6) Observability
   - Metrics: request latency, task durations, queue depth, optimizer iterations, model fit time, EHVI improvements.
   - Tracing: request-to-worker spans; acquisition optimization spans.
   - Logging: Console logs by default; env toggles for level (`LOG_LEVEL`), format (`LOG_FORMAT=json|human`), and optional file logging (`LOG_TO_FILE=true`, `LOG_FILE_PATH`). In Docker, prefer stdout/stderr.

7) Encoding & Reconstruction (Optional)
   - Dimensionality Reduction: PCA via scikit-learn with optional parameter scaling (none|standardize|minmax); report `pc_mins`, `pc_maxs` with outputs when PCA is used.
   - No server cache: Clients retain PCA info from OptimizeResponse and provide it for ReconstructionRequest.
   - Reconstruction Service: SLSQP-based solver that finds original variables whose encoded value matches a target point. Enforces:
     - Ingredient sum-to-one equality and per-variable bounds
     - Parameter bounds
     - Optional ingredient ratio constraints (linear inequalities)
   - Precision Target: ≤ 1e-7 encoding error typical; reports metrics (initial/final error, iterations, constraint satisfaction, method).
   - API Usage: Reconstruct arbitrary additional encoded points using the cached PCA and normalization by providing normalized [0,1]^D coordinates.

### 5) Request/Response Schemas (High-level)
- OptimizeRequest: dataset, search_space, objectives/constraints, settings, optional encoding
- OptimizeResponse: predictions, optional Pareto, optional encoding_info, diagnostics
- ReconstructionRequest: coordinates, PCA info or data to fit, bounds
- ReconstructionResponse: solution (ingredients/parameters), error metrics

### 6) API Contracts (Summary)
- POST /v1/optimize: OptimizeRequest → OptimizeResponse
- POST /v1/reconstruct: ReconstructionRequest → ReconstructionResponse

NATS Transport (Optional)
- Subjects (request/reply):
  - sator.v1.optimize
  - sator.v1.reconstruct
- Authentication: include `x-api-key` in NATS message headers. Responses mirror HTTP JSON payloads.
 - Pattern: request-reply returns { job_id }; final result published on `sator.v1.jobs.<job_id>`.

Request/Response payloads are strictly validated with Pydantic models and versioned under /v1.

### 7) Request Flow (Stateless)
1. Client sends OptimizeRequest (dataset + config)
2. Engine fits and evaluates; returns OptimizeResponse (predictions, optional Pareto, optional encoding info)
3. For reconstruction, client sends ReconstructionRequest with PCA info and target coordinates; engine returns formulation

### 8) Security
- **Authentication**: Static API keys from environment (`SATOR_API_KEYS=key1,key2`); any valid key has full access; constant-time comparison; rotate by restarting with updated env; no key storage.
- **Rate limiting**: Env-configured per-key and per-IP (e.g., `SATOR_RATE_LIMIT_PER_MIN=300`).
- **IP controls**: Env-configured allow/deny lists (`SATOR_IP_WHITELIST`, `SATOR_IP_BLACKLIST` comma-separated). Deny list evaluated first.
- **Transport security**: TLS termination recommended; CORS allow-list.
- **Idempotency**: `Idempotency-Key` header for mutation endpoints (e.g., tell) with replay protection.
- **Secrets & storage**: Load keys from environment or a secret manager; do not persist keys to the database.
- **Dependency & supply chain**: Pin deps, enable vulnerability scanning, generate SBOM.

### 8) API Conventions & Productization
- **Versioning**: Prefix all routes with /v1; additive changes only; breaking changes via v2.
- **Error model**: Consistent JSON shape `{code, message, details, request_id}` with machine-readable codes.
- **Pagination**: Cursor-based pagination for list endpoints (`limit`, `cursor`).
- **Health/Readiness**: `/livez`, `/readyz` endpoints and `/metrics` for Prometheus.
- **Idempotency**: `Idempotency-Key` required for create/update style endpoints.
- **Request validation**: Strict Pydantic schemas; reject unknown fields when appropriate.

### 9) Deployment & Packaging
- Library: `sator-os-engine` (PyPI) with optional SDK
- Server entrypoint: `sator-server` (uvicorn)
- Docker image: server; docker-compose for server (+optional NATS)
- Helm chart (optional)

### 10) Configuration
- Env settings: LOG_LEVEL, LOG_FORMAT, LOG_TO_FILE, LOG_FILE_PATH
- Device: `SATOR_DEVICE=cpu|cuda`, optional `SATOR_CUDA_DEVICE=0`; fallback to CPU if CUDA unavailable
- Transports: `SATOR_ENABLE_HTTP`, `SATOR_HTTP_HOST`, `SATOR_HTTP_PORT`, `SATOR_ENABLE_NATS`, `SATOR_NATS_URL`
- Security: `SATOR_API_KEYS`, `SATOR_RATE_LIMIT_PER_MIN`, `SATOR_IP_WHITELIST`, `SATOR_IP_BLACKLIST`
- Runtime: job TTL, timeouts, concurrency caps

### 11) Testing Strategy
- Unit tests (pytest); property-based (Hypothesis) for search space handling
- Integration tests: HTTP API (and optional NATS) with docker-compose
- Determinism tests: fixed seeds, replay logs
- Load tests for concurrent jobs and batched computations

### 12) Licensing & Governance
- Recommended license: Apache-2.0 (patent grant, permissive)
- Third-party: BoTorch, PyTorch, Celery, Redis, SQLModel; credit in NOTICE
- Governance: CONTRIBUTING.md, CODE_OF_CONDUCT.md, issue templates, release cadence (semver)

### 13) Roadmap (High-level)
- v0.1: Stateless optimize + reconstruct (HTTP), PCA+SLSQP, logging, env auth, rate limiting, IP filters
- v0.2: Optional NATS transport; SDK wrappers; basic metrics
- v0.3: Performance tuning (GPU), docs/examples; Helm chart (optional)

### 14) Alternatives & Rationale
- BoTorch chosen for state-of-the-art MOBO, GPU support, and research velocity
- No external broker/DB in v0.1 to keep deployment simple; can add later if needed
- FastAPI chosen for speed, typing, and ecosystem maturity


