from __future__ import annotations

import math
import os
from typing import Tuple

import numpy as np

try:
	# Use non-interactive backend for CI/headless
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore
	from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
	_HAS_MPL = True
except Exception:
	_HAS_MPL = False

from sator_os_engine.core.models.optimize import OptimizeRequest, OptimizationConfig
from sator_os_engine.core.optimizer.mobo_engine import run_optimization

try:
	from torch.quasirandom import SobolEngine
	_HAS_SOBOL = True
except Exception:
	_HAS_SOBOL = False


def branin(x: np.ndarray) -> np.ndarray:
	# x: (..., 2)
	x1 = x[..., 0]
	x2 = x[..., 1]
	a = 1.0
	b = 5.1 / (4.0 * math.pi ** 2)
	c = 5.0 / math.pi
	r = 6.0
	s = 10.0
	t = 1.0 / (8.0 * math.pi)
	return (a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s).astype(float)


def run_short_optimization(
	rng: np.random.Generator, *, n_init: int = 12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
	# Domain bounds
	bounds = np.array([[-5.0, 10.0], [0.0, 15.0]], dtype=float)  # (2,2)
	# Initial design
	if _HAS_SOBOL:
		sob = SobolEngine(dimension=2, scramble=True, seed=int(rng.integers(1, 2**31 - 1)))
		u = sob.draw(n_init).numpy()
	else:
		u = rng.uniform(low=0.0, high=1.0, size=(n_init, 2))
	X = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * u
	Y = branin(X)[:, None]

	search_space = {
		"parameters": [
			{"name": "x1", "type": "float", "min": float(bounds[0, 0]), "max": float(bounds[0, 1])},
			{"name": "x2", "type": "float", "min": float(bounds[1, 0]), "max": float(bounds[1, 1])},
		]
	}
	objectives = {"f": {"goal": "min"}}

	# Single BO suggestion (minimize)
	req = OptimizeRequest(
		dataset={"X": X.tolist(), "Y": Y.tolist()},
		search_space=search_space,
		objectives=objectives,
		optimization_config=OptimizationConfig(
			acquisition="qei",
			batch_size=1,
			max_evaluations=25,
			seed=int(rng.integers(0, 2**31 - 1)),
			return_maps=False,
		),
	)
	res = run_optimization(req, device="cpu")
	preds = res.get("predictions", []) or []
	if preds:
		P = np.array([[preds[0]["candidate"]["x1"], preds[0]["candidate"]["x2"]]], dtype=float)
	else:
		# fallback: random candidate if optimizer fails
		P = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * rng.uniform(size=(1, 2))
	Yp = branin(P)[:, None]

	return X, Y, P, Yp, n_init


def main() -> str:
	if not _HAS_MPL:
		raise RuntimeError("matplotlib not available; install it to generate the 3D surface")

	rng = np.random.default_rng(0)
	X, Y, P, Yp, n_init = run_short_optimization(rng, n_init=12)

	# Grid for true function surface
	bounds = np.array([[-5.0, 10.0], [0.0, 15.0]], dtype=float)
	gx, gy = np.meshgrid(
		np.linspace(bounds[0, 0], bounds[0, 1], 120),
		np.linspace(bounds[1, 0], bounds[1, 1], 120),
		indexing="xy",
	)
	g = np.stack([gx.ravel(), gy.ravel()], axis=1)
	gz = branin(g).reshape(gx.shape)

	# Plot
	fig = plt.figure(figsize=(8, 6), dpi=140)
	ax = fig.add_subplot(111, projection="3d")
	surf = ax.plot_surface(gx, gy, gz, cmap="viridis", linewidth=0, antialiased=True, alpha=0.75)
	fig.colorbar(surf, shrink=0.65, aspect=12, pad=0.08)

	# Overlay initial samples and suggested minimum
	z_eps = 1e-3
	# Initial (space-filling)
	ax.scatter(
		X[:n_init, 0],
		X[:n_init, 1],
		Y[:n_init, 0] + z_eps,
		s=36,
		c="#ffffff",
		marker="o",
		edgecolors="#222222",
		linewidths=0.7,
		alpha=1.0,
		depthshade=False,
		label="initial",
	)
	# Suggested next point (predicted minimum)
	ax.scatter(
		[P[0, 0]],
		[P[0, 1]],
		[Yp[0, 0] + 3 * z_eps],
		s=160,
		c="#ffd400",
		marker="*",
		edgecolors="#222222",
		linewidths=0.8,
		depthshade=False,
		label="suggested",
	)
	# No incumbent path and no annotation box

	ax.set_xlabel("x1")
	ax.set_ylabel("x2")
	ax.set_zlabel("f(x1, x2)")
	ax.set_title("Branin — short BO run (minimize) — 3D surface")
	ax.view_init(elev=35, azim=45)
	ax.legend(loc="upper left")
	plt.tight_layout()

	out_dir = os.path.join("tests", "artifacts")
	os.makedirs(out_dir, exist_ok=True)
	out_path = os.path.join(out_dir, "visual_branin_3d.png")
	plt.savefig(out_path)
	plt.close(fig)
	return out_path


if __name__ == "__main__":
	print(main())

