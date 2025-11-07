# How to run tests

## Prerequisites
- Activate the project virtual environment
  - Windows PowerShell: `./venv/Scripts/Activate.ps1`
- Ensure pytest is installed (already handled by the repo setup). If needed:
  - `pip install pytest`

## Run all tests
- Start the server first if you plan to run any HTTP/system tests
  - Windows PowerShell:
    - `./venv/Scripts/Activate.ps1`
    - `$env:SATOR_API_KEY = "dev-key"`
    - `python -m sator_os_engine.server.main`
- Terminal:
  - `pytest -q`
- VS Code / Cursor:
  - Run Task → "SATOR: Run Tests"

## Run a subset
- Single file:
  - `pytest tests/integration/test_health_and_auth.py -q`
- Single test node:
  - `pytest tests/unit/test_constraints_enforcement.py::test_sum_constraint_enforced_in_candidates -q`

## Useful options
- `-q` quieter output
- `-k expr` filter by test name substring, e.g. `-k optimize`
- `-x` stop after first failure

## Environment
- For server-backed tests, set the API key and (optionally) base URL:
  - `$env:SATOR_API_KEY = "dev-key"`
  - `$env:SATOR_BASE_URL = "http://localhost:8080"` (default if unset)
- Unit tests that use in-process TestClient do not require the server.
