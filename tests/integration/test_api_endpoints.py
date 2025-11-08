"""Integration tests for async API flows.

These tests spin up the FastAPI app in-memory and exercise:
- /v1/optimize: submit an asynchronous optimization job and poll until results
  are available; verify predictions are returned.
- /v1/reconstruct: submit a reconstruction job and poll until success; verify
  a reconstructed formulation is returned.

Authentication via x-api-key is included to mirror real usage.
"""

from __future__ import annotations

import os
import time

from fastapi.testclient import TestClient

from sator_os_engine.api.app import create_app
from sator_os_engine.settings import Settings


def _make_client() -> TestClient:
    os.environ["SATOR_API_KEY"] = "test-key"
    settings = Settings()
    app = create_app(settings)
    return TestClient(app)


def test_optimize_async_flow():
    client = _make_client()
    payload = {
        "dataset": {},
        "search_space": {
            "parameters": [
                {"name": "x1", "type": "float", "min": 0.0, "max": 1.0},
                {"name": "x2", "type": "float", "min": -1.0, "max": 1.0},
            ]
        },
        "objectives": {"o1": {"goal": "min"}, "o2": {"goal": "min"}},
        "optimization_config": {"algorithm": "qnehvi", "batch_size": 3, "max_evaluations": 10, "seed": 42},
    }
    r = client.post("/v1/optimize", json=payload, headers={"x-api-key": "test-key"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    # Poll for completion
    for _ in range(50):
        rr = client.get(f"/v1/jobs/{job_id}/result", headers={"x-api-key": "test-key"})
        if rr.status_code == 200 and "predictions" in rr.json():
            data = rr.json()
            assert isinstance(data["predictions"], list)
            break
        time.sleep(0.1)
    else:
        raise AssertionError("optimize result not ready in time")


def test_reconstruct_async_flow():
    client = _make_client()
    # Simple 3-dim (2 ingredients + 1 parameter), identity components
    pca_info = {
        "pc_mins": [0.0, 0.0],
        "pc_maxs": [1.0, 1.0],
        "components": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        "mean": [0.0, 0.0, 0.0],
    }
    payload = {
        "coordinates": [0.5, 0.5],
        "pca_info": pca_info,
        "bounds": {
            "ingredients": [[0.0, 1.0], [0.0, 1.0]],
            "parameters": [[0.0, 1.0]],
        },
        "n_ingredients": 2,
        "target_precision": 1e-7,
    }
    r = client.post("/v1/reconstruct", json=payload, headers={"x-api-key": "test-key"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    # Poll for completion
    for _ in range(50):
        rr = client.get(f"/v1/jobs/{job_id}/result", headers={"x-api-key": "test-key"})
        if rr.status_code == 200 and rr.json().get("success") is True:
            data = rr.json()
            assert "reconstructed_formulation" in data
            break
        time.sleep(0.1)
    else:
        raise AssertionError("reconstruct result not ready in time")


