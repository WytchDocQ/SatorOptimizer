# Operations

- Env:
  - SATOR_API_KEY — single API key used for HTTP/NATS authentication
  - SATOR_RATE_LIMIT_PER_MIN, SATOR_IP_WHITELIST, SATOR_IP_BLACKLIST
  - SATOR_ENABLE_HTTP, SATOR_HTTP_HOST, SATOR_HTTP_PORT
  - SATOR_ENABLE_NATS, SATOR_NATS_URL
  - SATOR_DEVICE=cpu|cuda, SATOR_CUDA_DEVICE
  - SATOR_RESULT_TTL_SEC, SATOR_JOB_TIMEOUT_SEC, SATOR_CONCURRENCY

- Health endpoints:
  - `GET /livez` and `GET /readyz`

- Running:
  - `sator-server` (uses `.env` for `SATOR_API_KEY` and port/host)
  - Or `python -m sator_os_engine.server.main`

