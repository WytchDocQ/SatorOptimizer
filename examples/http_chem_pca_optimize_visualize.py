from __future__ import annotations

import os
import time
from typing import Any, Dict, List
import json

import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
import httpx


def synth_dataset(n_samples: int = 20, n_ing: int = 9, rng_seed: int = 7):
    rng = np.random.default_rng(rng_seed)
    W = rng.dirichlet(alpha=np.ones(n_ing), size=n_samples)  # sum-to-one ingredients
    pH = rng.uniform(2.0, 12.0, size=n_samples)
    temp = rng.uniform(20.0, 90.0, size=n_samples)
    visc = rng.uniform(100.0, 2000.0, size=n_samples)
    price = rng.uniform(1.0, 10.0, size=n_samples)
    press = rng.uniform(1.0, 5.0, size=n_samples)
    P = np.stack([pH, temp, visc, price, press], axis=1)
    X = np.concatenate([W, P], axis=1)
    # two objectives to minimize
    o1 = 0.6 * np.abs(pH - 7.0) + 0.001 * (temp - 45.0) ** 2 + 0.0004 * visc + 2.0 * (W[:, 3] - 0.15) ** 2 + 1.5 * (W[:, 1] - 0.1) ** 2
    o2 = 1.8 * price + 0.05 * temp + 4.0 * W[:, 5] + 3.0 * W[:, 2] + 0.3 * press
    Y = np.stack([o1, o2], axis=1)
    return X, Y


def main() -> None:
    api_key = os.environ.get("SATOR_API_KEY", "dev-key")
    base = os.environ.get("SATOR_BASE_URL", "http://localhost:8080")

    n_ing = 9
    param_bounds = np.array([
        [2.0, 12.0],    # pH
        [20.0, 90.0],   # temperature C
        [100.0, 2000.0],# viscosity cP
        [1.0, 10.0],    # price
        [1.0, 5.0],     # pressure bar
    ], dtype=float)

    X, Y = synth_dataset(n_samples=20, n_ing=n_ing, rng_seed=7)

    # search space
    params: List[dict] = []
    for i in range(n_ing):
        params.append({"name": f"w{i+1}", "type": "float", "min": 0.0, "max": 1.0})
    for j, (lo, hi) in enumerate(param_bounds):
        params.append({"name": ["pH", "temp", "visc", "price", "press"][j], "type": "float", "min": float(lo), "max": float(hi)})

    payload: Dict[str, Any] = {
        "dataset": {"X": X.tolist(), "Y": Y.tolist()},
        "search_space": {"parameters": params},
        "objectives": {"quality_loss": {"goal": "min"}, "cost": {"goal": "min"}},
        "optimization_config": {
            "acquisition": "qnehvi",
            "batch_size": 5,
            "max_evaluations": 32,
            "seed": 123,
            "use_pca": True,
            "pca_dimension": 2,
            "return_maps": True,
            "map_space": "pca",
            "map_resolution": [50, 50],
            "sum_constraints": [{"indices": list(range(n_ing)), "target_sum": 1.0}],
        },
    }

    # Print dataset summary and compact payload summary (save full JSON to file)
    print("\n--- Chem dataset summary ---")
    print(f"X shape: {X.shape} (n_ing+5), Y shape: {Y.shape}")
    print("X first row:", X[0].tolist())
    print("Y first row:", Y[0].tolist())

    os.makedirs("examples/responses", exist_ok=True)
    payload_path = "examples/responses/chem_request.json"
    with open(payload_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2))
    oc = payload["optimization_config"]
    print("\n--- Optimize payload (summary) ---")
    print({
        "acquisition": oc.get("acquisition"),
        "batch_size": oc.get("batch_size"),
        "max_evaluations": oc.get("max_evaluations"),
        "use_pca": oc.get("use_pca"),
        "pca_dimension": oc.get("pca_dimension"),
        "sum_constraints": len(oc.get("sum_constraints") or []),
        "saved_to": payload_path,
    })

    with httpx.Client(timeout=60.0) as client:
        r = client.post(f"{base}/v1/optimize", json=payload, headers={"x-api-key": api_key})
        if r.status_code != 202:
            print("/v1/optimize error:")
            print(r.text)
        r.raise_for_status()
        job_id = r.json()["job_id"]
        print("\n--- Optimize accepted ---")
        print({"job_id": job_id})
        result = None
        for _ in range(240):
            rr = client.get(f"{base}/v1/jobs/{job_id}/result", headers={"x-api-key": api_key})
            data = rr.json()
            if rr.status_code == 200 and data.get("predictions"):
                result = data
                break
            # if job failed, print error and abort early
            if data.get("status") == "FAILED":
                raise RuntimeError(f"Job failed: {data.get('error')}")
            time.sleep(0.25)
        if result is None:
            raise RuntimeError("Result not ready in time")

    # Save full result JSON, but print a compact summary to avoid flooding the console
    result_json = json.dumps(result, indent=2)
    with open("examples/responses/chem_result.json", "w", encoding="utf-8") as f:
        f.write(result_json)

    preds = result.get("predictions", [])
    print("\n--- Optimize result (summary) ---")
    print(f"predictions: {len(preds)}")
    if preds:
        first = preds[0]
        print("first prediction:", json.dumps({
            "candidate": first.get("candidate"),
            "objectives": first.get("objectives"),
            "variances": first.get("variances"),
            "encoded": first.get("encoded"),
            "reconstructed": first.get("reconstructed")
        }, indent=2))
    if result.get("diagnostics"):
        print("diagnostics:", result.get("diagnostics"))
    names = [f"w{i+1}" for i in range(n_ing)] + ["pH", "temp", "visc", "price", "press"]
    P = np.array([[*(pred["candidate"].get(n, 0.0) for n in names)] for pred in preds], dtype=float)

    # Use server-provided PCA model for visualization; do not fit locally
    enc = result.get("encoding_info") or {}
    components = np.array(enc.get("components", []), dtype=float) if enc.get("components") is not None else None
    mean = np.array(enc.get("mean", []), dtype=float) if enc.get("mean") is not None else None
    pc_mins = np.array(enc.get("pc_mins", []), dtype=float) if enc.get("pc_mins") is not None else None
    pc_maxs = np.array(enc.get("pc_maxs", []), dtype=float) if enc.get("pc_maxs") is not None else None
    if components is not None and mean is not None and mean.size == X.shape[1]:
        Z_raw = (X - mean) @ components.T
        if pc_mins is not None and pc_maxs is not None and pc_mins.size == Z_raw.shape[1]:
            pc_range = np.maximum(pc_maxs - pc_mins, 1e-12)
            Z = (Z_raw - pc_mins) / pc_range
        else:
            Z = Z_raw
    else:
        Z = np.zeros((X.shape[0], 2), dtype=float)
    # Prefer encoded dataset from server if provided
    Z_enc_server = result.get("encoded_dataset")
    if Z_enc_server is not None:
        Z = np.array(Z_enc_server, dtype=float)
    # Predicted encoded coords come directly from the server
    Zp = np.array([pred.get("encoded", [0.0, 0.0]) for pred in preds], dtype=float) if preds else np.zeros((0, 2))

    # Plot PCA maps if provided, overlay predictions
    # Plot PCA maps as 3D heightmaps if present; otherwise show a 2D scatter fallback
    gp = result.get("gp_maps") or {}
    axes = gp.get("grid", {}).get("axes") if gp else None
    maps_means = gp.get("maps", {}).get("means") if gp else None
    titles = list(result.get("objectives", {}).keys()) if result.get("objectives") else ["quality_loss", "cost"]
    if axes and maps_means:
        print("gp_maps:", {"dimension": gp.get("dimension"), "space": gp.get("space"), "resolution": gp.get("grid", {}).get("resolution")})
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        xax = np.array(axes[0], dtype=float)
        yax = np.array(axes[1], dtype=float)
        XX, YY = np.meshgrid(xax, yax, indexing="xy")

        # Bilinear interpolation helper so points lie on the plotted surface
        def interp2(ax_x: np.ndarray, ax_y: np.ndarray, M: np.ndarray, points: np.ndarray) -> np.ndarray:
            nx = ax_x.size
            ny = ax_y.size
            xs = np.clip(points[:, 0], ax_x[0], ax_x[-1])
            ys = np.clip(points[:, 1], ax_y[0], ax_y[-1])
            ix1 = np.clip(np.searchsorted(ax_x, xs) - 1, 0, nx - 2)
            iy1 = np.clip(np.searchsorted(ax_y, ys) - 1, 0, ny - 2)
            x0 = ax_x[ix1]
            x1 = ax_x[ix1 + 1]
            y0 = ax_y[iy1]
            y1 = ax_y[iy1 + 1]
            tx = (xs - x0) / np.maximum(x1 - x0, 1e-12)
            ty = (ys - y0) / np.maximum(y1 - y0, 1e-12)
            z00 = M[iy1, ix1]
            z10 = M[iy1, ix1 + 1]
            z01 = M[iy1 + 1, ix1]
            z11 = M[iy1 + 1, ix1 + 1]
            return (1 - tx) * (1 - ty) * z00 + tx * (1 - ty) * z10 + (1 - tx) * ty * z01 + tx * ty * z11

        fig = plt.figure(figsize=(12, 5), dpi=120)
        for i, key in enumerate(titles[:2]):
            ax = fig.add_subplot(1, 2, i + 1, projection="3d")
            if key in maps_means:
                M = np.array(maps_means[key], dtype=float)
                surf = ax.plot_surface(XX, YY, M, cmap="viridis", linewidth=0, antialiased=True, alpha=0.8)
                fig.colorbar(surf, ax=ax, shrink=0.7, aspect=15, pad=0.08)
                print(f"map[{key}] min={np.min(M):.3g} max={np.max(M):.3g} shape={M.shape}")
            # overlay train/pred exactly on the surface using bilinear interpolation
            if key in maps_means:
                M = np.array(maps_means[key], dtype=float)
                if Z.size > 0:
                    z_train = interp2(xax, yax, M, Z)
                    ax.scatter(
                        Z[:, 0],
                        Z[:, 1],
                        z_train,
                        c="blue",
                        s=24,
                        marker="o",
                        alpha=0.9,
                        depthshade=False,
                        label="train",
                    )
                if len(Zp) > 0:
                    z_pred = interp2(xax, yax, M, Zp)
                    ax.scatter(
                        Zp[:, 0],
                        Zp[:, 1],
                        z_pred,
                        c="red",
                        s=60,
                        marker="x",
                        depthshade=False,
                        label="pred",
                    )
            ax.set_title(f"PCA(2) map — {key}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel(key)
            ax.view_init(elev=30, azim=-60)
            ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        ax.scatter(Z[:, 0], Z[:, 1], s=20, c="blue", alpha=0.9, label="train")
        if len(Zp) > 0:
            ax.scatter(Zp[:, 0], Zp[:, 1], s=60, c="red", marker="x", label="pred")
        ax.set_title("PCA(2) scatter (no maps returned)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    # No reconstruct requests by default; predictions already include reconstruction when PCA is used


if __name__ == "__main__":
    main()


