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

## HTTPS (TLS)

- Env:
  - `SATOR_ENABLE_TLS=true`
  - `SATOR_TLS_CERT_FILE=/path/to/cert.pem`
  - `SATOR_TLS_KEY_FILE=/path/to/key.pem`
  - `SATOR_TLS_KEY_PASSWORD` (optional)
  - `SATOR_TLS_CA_CERTS` (optional, client CA bundle if needed)

- Direct TLS (Uvicorn):
  - The server will enable HTTPS automatically when `SATOR_ENABLE_TLS=true` and both cert/key files are provided.
  - Example `.env`:
    ```
    SATOR_API_KEY=dev-key
    SATOR_HTTP_HOST=0.0.0.0
    SATOR_HTTP_PORT=8443
    SATOR_ENABLE_TLS=true
    SATOR_TLS_CERT_FILE=certs/localhost.pem
    SATOR_TLS_KEY_FILE=certs/localhost-key.pem
    ```
  - Start: `sator-server` (or `python -m sator_os_engine.server.main`)
  - Access: `https://localhost:8443`
  - Local cert setup guide: see `local_tls_certs.md`

- Dev certificates (Windows/macOS/Linux):
  - Recommended: `mkcert` to create a locally trusted cert.
    - Install mkcert and trust local CA:
      - Windows (choco): `choco install mkcert` then `mkcert -install`
      - macOS (brew): `brew install mkcert nss` then `mkcert -install`
      - Linux: install mkcert, then `mkcert -install` (may require nss/ca-certificates)
    - Generate certs for localhost: `mkcert localhost 127.0.0.1 ::1`
    - Set `SATOR_TLS_CERT_FILE` to the `.pem` (or `.crt`) and `SATOR_TLS_KEY_FILE` to the key.
  - For Office.js add-in dev: `npx office-addin-dev-certs install` also generates a trusted localhost cert; you can point the engine to those files.

- Production (reverse proxy TLS recommended):
  - Terminate TLS with a reverse proxy and forward to the engine over HTTP.
  - Caddy (automatic Let's Encrypt):
    ```
    your.domain.com {
      reverse_proxy 127.0.0.1:8080
    }
    ```
  - NGINX (snippet):
    ```
    server {
      listen 443 ssl;
      server_name your.domain.com;
      ssl_certificate /etc/letsencrypt/live/your.domain.com/fullchain.pem;
      ssl_certificate_key /etc/letsencrypt/live/your.domain.com/privkey.pem;
      location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
      }
    }
    ```
  - Ensure your firewall/DNS point to the proxy; keep the engine bound to `127.0.0.1:8080`.

