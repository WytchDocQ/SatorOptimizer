"""Integration tests for health checks and authentication.

This module verifies:
- Health endpoints `/livez` and `/readyz` respond with expected JSON payloads.
- Requests without `x-api-key` to protected endpoints are rejected (401/403).

The FastAPI app is instantiated in-memory with a configured API key.
"""

from __future__ import annotations

import os
from fastapi.testclient import TestClient

from sator_os_engine.api.app import create_app
from sator_os_engine.settings import Settings


def _client_with_key() -> TestClient:
    os.environ["SATOR_API_KEY"] = "test-key"
    settings = Settings()
    app = create_app(settings)
    return TestClient(app)


def test_health_endpoints():
    client = _client_with_key()
    r1 = client.get("/livez")
    r2 = client.get("/readyz")
    assert r1.status_code == 200 and r1.json().get("status") == "ok"
    assert r2.status_code == 200 and r2.json().get("status") == "ready"


def test_auth_missing_key_rejected():
    client = _client_with_key()
    r = client.post("/v1/optimize", json={})
    assert r.status_code in (401, 403)


