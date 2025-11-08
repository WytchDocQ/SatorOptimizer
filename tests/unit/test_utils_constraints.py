"""Unit tests for constraint and feasibility utilities.

This file verifies that:
- `build_linear_constraints` correctly translates:
- Sum-to-target constraints into two linear inequalities: +target and -target
  for the same set of indices.
- Ratio constraints (i/j) into linear inequalities for both min_ratio and
  max_ratio forms.

It also tests:
- `feasible_mask` to ensure points satisfying sum/ratio constraints are marked
  feasible while violators are rejected.
- `enforce_sum_constraints_np` to ensure candidate vectors are adjusted to meet
  target sums while respecting parameter bounds.

The tests assert the expected structures and numeric tolerances relevant to
downstream optimization steps.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sator_os_engine.core.optimizer.utils import build_linear_constraints, feasible_mask, enforce_sum_constraints_np


def _make_req(sum_constraints=None, ratio_constraints=None):
    return SimpleNamespace(
        optimization_config=SimpleNamespace(
            sum_constraints=sum_constraints or [],
            ratio_constraints=ratio_constraints or [],
        )
    )


def test_build_linear_constraints_sum_and_ratio():
    params = [
        {"name": "a", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "b", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "c", "type": "float", "min": 0.0, "max": 1.0},
    ]
    # Sum-to-target on a+b == 1.0
    sums = [{"indices": [0, 1], "target_sum": 1.0}]
    # Ratio constraints: a/c in [0.5, 2.0]
    ratios = [{"i": 0, "j": 2, "min_ratio": 0.5, "max_ratio": 2.0}]
    req = _make_req(sum_constraints=sums, ratio_constraints=ratios)

    ineq, eqpairs = build_linear_constraints(req, params)

    # No equality constraints expected in current implementation
    assert isinstance(eqpairs, list) and len(eqpairs) == 0
    # Expect 2 from sum (± target), and 2 from ratio (min and max) => total 4
    assert isinstance(ineq, list) and len(ineq) == 4

    # Helper to find a constraint (indices, coeffs, rhs) regardless of ordering
    def has_constraint(target_idxs, target_coeffs, target_rhs):
        for idxs, coeffs, rhs in ineq:
            if idxs == target_idxs and coeffs == target_coeffs and abs(float(rhs) - float(target_rhs)) < 1e-12:
                return True
        return False

    # Sum ± target
    assert has_constraint([0, 1], [1.0, 1.0], 1.0)
    assert has_constraint([0, 1], [-1.0, -1.0], -1.0)

    # Ratio bounds encoded as linear inequalities on (a, c)
    # a/c <= 2.0  => a - 2.0*c <= 0
    assert has_constraint([0, 2], [1.0, -2.0], 0.0)
    # a/c >= 0.5  => -a + 0.5*c <= 0
    assert has_constraint([0, 2], [-1.0, 0.5], 0.0)


def test_feasible_mask_sum_and_ratio():
    params = [
        {"name": "a", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "b", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "c", "type": "float", "min": 0.0, "max": 1.0},
    ]
    sums = [{"indices": [0, 1], "target_sum": 1.0}]
    ratios = [{"i": 0, "j": 2, "min_ratio": 0.5, "max_ratio": 2.0}]
    req = _make_req(sum_constraints=sums, ratio_constraints=ratios)

    # Points: [a, b, c]
    pts = [
        [0.5, 0.5, 0.5],   # sum ok (1.0), ratio a/c=1.0 within [0.5, 2.0] -> True
        [0.6, 0.6, 0.5],   # sum violates (1.2) -> False
        [0.9, 0.1, 0.3],   # sum ok, ratio a/c=3.0 > 2.0 -> False
        [0.2, 0.8, 0.6],   # sum ok, ratio a/c≈0.333 < 0.5 -> False
    ]
    mask = feasible_mask(pts, req, params, tol=1e-6)
    assert mask == [True, False, False, False]


def test_enforce_sum_constraints_np_respects_target_and_bounds():
    params = [
        {"name": "a", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "b", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "c", "type": "float", "min": 0.0, "max": 1.0},
    ]
    sums = [{"indices": [0, 1], "target_sum": 1.0}]
    req = _make_req(sum_constraints=sums, ratio_constraints=[])

    # Candidates not summing to 1 on a+b
    X = np.array(
        [
            [0.3, 0.3, 0.1],  # sum 0.6
            [0.7, 0.7, 0.2],  # sum 1.4
            [1.2, -0.1, 0.5], # outside bounds, will be clipped then renormalized
        ],
        dtype=float,
    )

    Xe = enforce_sum_constraints_np(X, params, req)
    # Check shape preserved
    assert Xe.shape == X.shape
    # Check sums a+b close to 1.0
    s = Xe[:, [0, 1]].sum(axis=1)
    assert np.allclose(s, 1.0, atol=1e-2)
    # Check bounds respected
    assert np.all((Xe >= 0.0) & (Xe <= 1.0))


