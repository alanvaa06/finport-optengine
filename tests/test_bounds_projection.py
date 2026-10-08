"""Tests for project_to_bounds_iterated."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.optimizers._bounds import (
    InfeasibleBoundsError,
    project_to_bounds_iterated,
    project_to_constraints,
)


def test_already_feasible_input_unchanged():
    w = np.array([0.3, 0.3, 0.4])
    out = project_to_bounds_iterated(w, np.zeros(3), np.ones(3))
    np.testing.assert_allclose(out, w, atol=1e-10)


def test_residual_distributed_over_slack():
    w = np.array([0.5, 0.5, 0.5])  # sums to 1.5, must come down to 1.0
    out = project_to_bounds_iterated(w, np.zeros(3), np.full(3, 0.6))
    assert pytest.approx(out.sum(), abs=1e-8) == 1.0
    assert (out >= -1e-9).all() and (out <= 0.6 + 1e-9).all()


def test_clip_then_rescale_breaks_bounds_but_iterated_does_not():
    # A naive clip(0,0.4) + rescale would push the first weight back over 0.4.
    w = np.array([0.9, 0.05, 0.05])
    lb = np.zeros(3)
    ub = np.array([0.4, 1.0, 1.0])
    out = project_to_bounds_iterated(w, lb, ub)
    assert (out <= ub + 1e-9).all()
    assert (out >= lb - 1e-9).all()
    assert pytest.approx(out.sum(), abs=1e-8) == 1.0


def test_infeasible_lb_sum_raises():
    with pytest.raises(InfeasibleBoundsError):
        project_to_bounds_iterated(
            np.array([0.5, 0.5, 0.5]),
            np.full(3, 0.5),
            np.ones(3),
        )


def test_infeasible_ub_sum_raises():
    with pytest.raises(InfeasibleBoundsError):
        project_to_bounds_iterated(
            np.array([0.1, 0.1, 0.1]),
            np.zeros(3),
            np.full(3, 0.2),
        )


def test_lb_greater_than_ub_raises_valueerror():
    with pytest.raises(ValueError, match="lb must be"):
        project_to_bounds_iterated(
            np.array([0.5, 0.5]),
            np.array([0.6, 0.0]),
            np.array([0.4, 1.0]),
        )


def test_negative_residual_distributed_over_slack_above_lb():
    # w sums to 0.5 (deficit), pushed up via lower-side slack
    w = np.array([0.1, 0.1, 0.3])
    lb = np.array([0.05, 0.05, 0.05])
    ub = np.full(3, 1.0)
    out = project_to_bounds_iterated(w, lb, ub)
    assert pytest.approx(out.sum(), abs=1e-8) == 1.0
    assert (out >= lb - 1e-9).all()


def test_hrp_with_an_open_budget_keeps_it_open():
    """Review item O15: the fast projection forced ``sum = 1`` regardless.

    With no bucket budget or active-share cap the projection clipped and
    redistributed toward a unit budget, even under ``fully_invested=False``.
    Three 25% caps cannot reach one, so HRP raised InfeasibleBoundsError on a
    mandate that is satisfiable by any book inside the box. Without a budget
    the closest point in the box is HRP's own allocation, clipped.
    """
    import pandas as pd

    from optimization_engine.optimizers.base import PortfolioConstraints
    from optimization_engine.optimizers.hrp import HRPOptimizer

    names = ["A", "B", "C"]
    vol = np.array([0.10, 0.15, 0.20])
    corr = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.5], [0.1, 0.5, 1.0]])
    cov = pd.DataFrame(np.outer(vol, vol) * corr, index=names, columns=names)
    constraints = PortfolioConstraints(
        fully_invested=False, bounds={a: (0.0, 0.25) for a in names}
    )

    raw = HRPOptimizer(cov_matrix=cov).optimize().weights.values
    result = HRPOptimizer(cov_matrix=cov, constraints=constraints).optimize()

    np.testing.assert_allclose(result.weights.values, np.clip(raw, 0.0, 0.25), atol=1e-9)
    assert result.weights.sum() < 1.0
    assert result.is_compliant, result.violations


def test_the_fast_projection_honours_a_gross_exposure_cap():
    """Clip-and-redistribute cannot see gross exposure, so a cap now takes the exact path."""
    from optimization_engine.optimizers.base import PortfolioConstraints

    assets = ["A", "B", "C"]
    constraints = PortfolioConstraints(
        long_only=False, bounds={a: (-1.0, 1.0) for a in assets}, leverage=1.2
    )
    raw = np.array([0.9, 0.6, -0.5])  # sums to one, gross 2.0

    projected, distance = project_to_constraints(raw, assets, constraints)

    assert projected.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.abs(projected).sum() <= 1.2 + 1e-6
    assert distance > 0.0
