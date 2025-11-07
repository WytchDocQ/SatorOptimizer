# How to run examples

## 1) Start the server (manually, in a separate terminal)
- Windows PowerShell:
  - `./venv/Scripts/Activate.ps1`
  - `$env:SATOR_API_KEY = "dev-key"`
  - `python -m sator_os_engine.server.main`
- If your server runs elsewhere, set `SATOR_BASE_URL` accordingly (default is `http://localhost:8080`).

## 2) Install plotting deps (first time only)
- `pip install matplotlib scikit-learn httpx`

## 3) Run the examples
- Branin (3D) — builds dataset, sends optimize to server, polls, plots results:
  - `python .\examples\http_visualize_branin.py`
- Chemical PCA — 20 formulations (9 ingredients sum-to-one + 5 params), PCA(2) maps, reconstruct a few:
  - `python .\examples\http_chem_pca_optimize_visualize.py`

## 4) Configuration
- API key: `SATOR_API_KEY` (defaults to `dev-key`)
- Server address: `SATOR_BASE_URL` (defaults to `http://localhost:8080`)

If you see HTTP 401/403, check the API key. If you see 422, print the server error by temporarily adding `print(r.text)` before raising in the script to view validation details.

---

## What each example demonstrates

### Chemical PCA example (`examples/http_chem_pca_optimize_visualize.py`)
- **Dataset**: 20 synthetic formulations with 9 ingredients (sum-to-one) plus 5 other process parameters (pH, temp, viscosity, price, pressure). Two objectives are generated and both are set to **minimize**.
- **Server workflow exercised**:
  - Sum-to-one enforcement on training inputs.
  - Optional PCA (set to PCA(2)) with normalization of PC coordinates to [0, 1].
  - Per-objective Gaussian Process fitting in the chosen model space (PCA space here).
  - Multi-objective Bayesian optimization (qEHVI) to propose a batch of candidates balancing both objectives (i.e., move toward the Pareto front).
  - Automatic reconstruction of PCA-encoded predictions back to the original ingredient + parameter space (respecting sum-to-one and bounds).
  - Optional GP surface maps returned for visualization when PCA(2) is used.

- **What the plots show**:
  - Two side-by-side 3D heightmaps in the PCA(2) plane (`PC1`, `PC2`): one surface per objective (**quality_loss**, **cost**). These are the GP posterior mean surfaces learned from your dataset.
  - **Blue dots** are your training points, encoded into PCA space and plotted exactly on the corresponding GP surface by interpolation.
  - **Red crosses** are the newly suggested candidates (predictions), also shown exactly on-surface. The same PCA coordinates are evaluated on both surfaces to visualize the trade-off.
  - The color bars are the GP posterior mean values for each objective at each PCA coordinate.

- **What is being optimized**:
  - The server minimizes both objectives simultaneously. With qEHVI, it selects a batch of candidates that improve the overall Pareto front under constraints (sum-to-one for ingredients and variable bounds). The returned predictions include:
    - `encoded`: PCA(2) coordinates of each candidate.
    - `reconstructed`: the candidate reconstructed back to the original 14-D space (9 ingredients + 5 parameters), enforcing sum-to-one and respecting bounds.
    - `objectives` and `variances`: GP posterior means and variances for each objective at the candidate.

- **What do `quality_loss` and `cost` mean here? (synthetic demo)**
  - These are two synthetic objective functions used only for the example to mimic real goals; both are minimized.
  - `quality_loss` ≈ penalties for being away from target specs, e.g. pH near 7, temperature near 45 °C, higher viscosity, and specific ingredient ratio preferences:
    - \(0.6\,|pH-7| + 0.001\,(temp-45)^2 + 0.0004\,visc + 2.0\,(w_4-0.15)^2 + 1.5\,(w_2-0.10)^2\)
  - `cost` ≈ operational/material cost proxy from price, temperature, pressure, and some ingredient fractions:
    - \(1.8\,price + 0.05\,temp + 0.3\,press + 4.0\,w_6 + 3.0\,w_3\)
  - In a real project, you replace these with your true measured/derived targets; the server treats them generically.

- **Files saved**:
  - Full request: `examples/responses/chem_request.json`
  - Full result: `examples/responses/chem_result.json`

### Branin example (`examples/http_visualize_branin.py`)
- **Dataset**: Random samples from the well-known Branin function in 2D; single objective to minimize.
- **Server workflow exercised**:
  - GP fit in the original input space (no PCA).
  - Single-objective candidate selection.
- **What the plot shows**:
  - A 3D surface of the true Branin function to provide visual context (computed locally for display only).
  - The training points (white) and the predicted candidates (red X) overlaid on that surface.
- **Files saved**:
  - Full request: `examples/responses/branin_request.json`
  - Full result: `examples/responses/branin_result.json`

---

## Notes on interpretation
- GP surfaces are models learned from your data; they are not the ground truth physics. Peaks/valleys indicate the model’s belief about where the objective is high/low.
- In multi-objective mode, the same encoded candidate is shown on all objective surfaces. Its z-value differs per surface because each objective has its own GP.
- Reconstruction ensures suggested formulations are valid in the original space (e.g., ingredients sum exactly to one).
