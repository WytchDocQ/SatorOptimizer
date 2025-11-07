from __future__ import annotations

import math
import os
import time
from typing import Any, Dict
import json

import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import httpx


def branin(x: np.ndarray) -> np.ndarray:
    x1 = x[..., 0]
    x2 = x[..., 1]
    a = 1.0
    b = 5.1 / (4.0 * math.pi ** 2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    return (a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s).astype(float)


def main() -> None:
    api_key = os.environ.get("SATOR_API_KEY", "dev-key")
    base = os.environ.get("SATOR_BASE_URL", "http://localhost:8080")

    # Build dataset
    rng = np.random.default_rng(0)
    bounds = np.array([[-5.0, 10.0], [0.0, 15.0]], dtype=float)
    n_train = 80
    u = rng.uniform(low=0.0, high=1.0, size=(n_train, 2))
    X = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * u
    Y = branin(X)[:, None]

    payload: Dict[str, Any] = {
        "dataset": {"X": X.tolist(), "Y": Y.tolist()},
        "search_space": {
            "parameters": [
                {"name": "x1", "type": "float", "min": float(bounds[0, 0]), "max": float(bounds[0, 1])},
                {"name": "x2", "type": "float", "min": float(bounds[1, 0]), "max": float(bounds[1, 1])},
            ]
        },
        "objectives": {"f": {"goal": "min"}},
        "optimization_config": {
            "acquisition": "qnehvi",
            "batch_size": 5,
            "max_evaluations": 20,
            "seed": 123,
            "return_maps": False,
        },
    }

    # Print dataset and payload (and save copies)
    print("\n--- Branin dataset summary ---")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    print("X sample (first 5):", X[:5].tolist())
    print("Y sample (first 5):", Y[:5].ravel().tolist())
    os.makedirs("examples/responses", exist_ok=True)
    payload_path = "examples/responses/branin_request.json"
    with open(payload_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2))
    print("\n--- Optimize payload (summary) ---")
    oc = payload["optimization_config"]
    print({
        "acquisition": oc.get("acquisition"),
        "batch_size": oc.get("batch_size"),
        "max_evaluations": oc.get("max_evaluations"),
        "return_maps": oc.get("return_maps"),
        "saved_to": payload_path,
    })

    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{base}/v1/optimize", json=payload, headers={"x-api-key": api_key})
        if r.status_code != 202:
            print("/v1/optimize error:")
            print(r.text)
        r.raise_for_status()
        job_id = r.json()["job_id"]
        print("\n--- Optimize accepted ---")
        print({"job_id": job_id})

        # Poll for result
        result = None
        for _ in range(120):
            rr = client.get(f"{base}/v1/jobs/{job_id}/result", headers={"x-api-key": api_key})
            if rr.status_code == 200 and rr.json().get("predictions"):
                result = rr.json()
                break
            time.sleep(0.25)
        if result is None:
            raise RuntimeError("Result not ready in time")

    # Print and save the full result JSON
    # Save full result JSON, print compact summary
    result_json = json.dumps(result, indent=2)
    with open("examples/responses/branin_result.json", "w", encoding="utf-8") as f:
        f.write(result_json)
    print("\n--- Optimize result (summary) ---")
    preds = result.get("predictions", [])
    print(f"predictions: {len(preds)}")
    if preds:
        print("first prediction:", json.dumps({
            "candidate": preds[0].get("candidate"),
            "objectives": preds[0].get("objectives"),
            "variances": preds[0].get("variances")
        }, indent=2))

    preds = result.get("predictions", [])
    P = np.array([[p["candidate"]["x1"], p["candidate"]["x2"]] for p in preds], dtype=float)

    # Make 3D figure
    gx, gy = np.meshgrid(
        np.linspace(bounds[0, 0], bounds[0, 1], 140),
        np.linspace(bounds[1, 0], bounds[1, 1], 140),
        indexing="xy",
    )
    g = np.stack([gx.ravel(), gy.ravel()], axis=1)
    gz = branin(g).reshape(gx.shape)

    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(gx, gy, gz, cmap="viridis", linewidth=0, antialiased=True, alpha=0.85)
    fig.colorbar(surf, shrink=0.7, aspect=15, pad=0.08, label="branin(x1,x2)")
    z_train = branin(X)
    ax.scatter(X[:, 0], X[:, 1], z_train, c="white", s=12, alpha=0.6, label="train")
    if len(P) > 0:
        z_pred = branin(P)
        ax.scatter(P[:, 0], P[:, 1], z_pred, c="red", s=50, marker="x", depthshade=False, label="pred")
    ax.set_title("HTTP optimize: Branin (3D)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1,x2)")
    ax.view_init(elev=30, azim=-60)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


