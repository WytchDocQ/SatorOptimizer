# How to run tests

## Prerequisites
- Activate the project virtual environment
  - Windows PowerShell: `.\venv\Scripts\Activate.ps1`
  - Windows CMD: `venv\Scripts\activate.bat`
  - bash/zsh: `source venv/bin/activate`
- Install dependencies (first time or when `requirements*.txt` changes):
  - `pip install -r requirements.txt`
  - Optional CUDA build: `pip install -r requirements-cuda.txt`
  - Install package (editable): `pip install -e .`
  - If needed: `pip install pytest`

## Run all tests
- No server required. Unit and integration tests in this repo use an in-process TestClient.
- Terminal:
  - `pytest -q`
- VS Code / Cursor:
  - Run Task → "SATOR: Run Tests"

## Run a subset
- Unit tests only (fast, no external services required):
  - `pytest -q tests/unit`
- Single file:
  - `pytest tests/integration/test_health_and_auth.py -q`
- Single test node:
  - `pytest tests/unit/test_constraints_enforcement.py::test_sum_constraint_enforced_in_candidates -q`

## Useful options
- `-q` quieter output
- `-k expr` filter by test name substring, e.g. `-k optimize`
- `-x` stop after first failure

## Environment
- None required for this test suite. Unit and integration tests run in-process; no server, ports, or TLS setup needed.
