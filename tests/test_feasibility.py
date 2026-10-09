"""Structural feasibility diagnosis.

The module under test answers a question the solver cannot: *which* part of a
mandate is impossible, and what to change. These tests pin the two things that
make that answer worth reading — that a finding names the arithmetic behind it,
and that "the solver could not tell us" is never reported as "your constraints
have no solution".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.constraints import BASIS_PARENT, ConstraintLayer
from optimization_engine.optimizers import _cvxpy_helpers
from optimization_engine.optimizers._cvxpy_helpers import SolverFailure
from optimization_engine.optimizers.base import PortfolioConstraints
from optimization_engine.optimizers.feasibility import (
    STAGE_REACHABLE_RETURN,
    STAGE_STRUCTURAL,
    analyze_feasibility,
    knapsack_return_range,
    reachable_return_range,
    reduces_to_box_and_budget,
)

ASSETS = ["A", "B", "C", "D"]
MU = pd.Series([0.02, 0.05, 0.09, 0.14], index=ASSETS)
COV = pd.DataFrame(
    np.diag([0.02, 0.03, 0.05, 0.09]), index=ASSETS, columns=ASSETS
)
LEGACY = "Asset class"  # what the flat groups/group_bounds pair is called


def codes(report) -> set[str]:
    return {i.code for i in report.issues}


def issue(report, code):
    return next(i for i in report.issues if i.code == code)


# ---------------------------------------------------------------------------
# Stage 1: structural arithmetic, no solver
# ---------------------------------------------------------------------------


def test_feasibility_box_capacity():
    """Σ ub < 1 is fatal, and the report names the shortfall in weight terms."""
    cons = PortfolioConstraints(bounds={a: (0.0, 0.20) for a in ASSETS})
    report = analyze_feasibility(ASSETS, cons, MU, COV)

    assert not report.is_feasible
    found = issue(report, "max_weights_below_budget")
    assert found.fatal and found.severity == "fatal"
    assert "80.00%" in found.message  # the caps as they stand
    assert "20.00%" in found.suggestion  # what has to be added
    # A fatal structural finding stops the analysis: no range is claimed.
    assert report.stage_reached == STAGE_STRUCTURAL
    assert report.reachable_return is None
    assert report.min_return is None and report.max_return is None


def test_feasibility_box_capacity_from_below():
    """Σ lb > 1 is the same check from the other side."""
    cons = PortfolioConstraints(bounds={a: (0.30, 1.0) for a in ASSETS})
    report = analyze_feasibility(ASSETS, cons, MU, COV)

    found = issue(report, "min_weights_exceed_budget")
    assert found.fatal
    assert "120.00%" in found.message
    assert "20.00%" in found.suggestion


def test_a_gross_cap_below_one_is_only_a_problem_for_a_fully_invested_book():
    """A 0.5× cap on a book with no budget is a mandate, not a contradiction."""
    unbudgeted = PortfolioConstraints(
        fully_invested=False, long_only=False, leverage=0.5
    )
    assert analyze_feasibility(ASSETS, unbudgeted, MU, COV).is_feasible

    budgeted = PortfolioConstraints(leverage=0.5)
    assert "leverage_below_budget" in codes(analyze_feasibility(ASSETS, budgeted))


def test_feasibility_gross_cap_below_the_box_minimum():
    """Without a budget the box still forces a minimum gross exposure."""
    cons = PortfolioConstraints(
        bounds={"A": (0.60, 1.0), "B": (-1.0, -0.60), "C": (0.0, 1.0), "D": (0.0, 1.0)},
        fully_invested=False,
        long_only=False,
        leverage=1.0,
    )
    report = analyze_feasibility(ASSETS, cons, MU, COV)

    found = issue(report, "leverage_below_box_minimum")
    assert found.fatal
    assert "1.20x" in found.message and "1.00x" in found.message


def test_feasibility_layer_capacity():
    """A bucket budget its members' bounds cannot fund, in both directions."""
    tight_caps = PortfolioConstraints(
        bounds={"A": (0.0, 0.10), "B": (0.0, 0.10), "C": (0.0, 1.0), "D": (0.0, 1.0)},
        constraint_layers=(
            ConstraintLayer(
                name="Region",
                assignments={"A": "US", "B": "US", "C": "EU", "D": "EU"},
                limits={"US": (0.50, 1.0)},
            ),
        ),
    )
    report = analyze_feasibility(ASSETS, tight_caps, MU, COV)
    found = issue(report, "group_min_unreachable")
    assert found.fatal
    assert "50.00%" in found.message and "20.00%" in found.message

    tight_floors = PortfolioConstraints(
        bounds={"A": (0.20, 0.50), "B": (0.20, 0.50), "C": (0.0, 1.0), "D": (0.0, 1.0)},
        constraint_layers=(
            ConstraintLayer(
                name="Region",
                assignments={"A": "US", "B": "US", "C": "EU", "D": "EU"},
                limits={"US": (0.0, 0.30)},
            ),
        ),
    )
    report = analyze_feasibility(ASSETS, tight_floors, MU, COV)
    found = issue(report, "group_max_unreachable")
    assert found.fatal
    assert "30.00%" in found.message and "40.00%" in found.message


def test_feasibility_parent_coherence():
    """A child bucket's limits have to fit inside the sleeve that holds them."""
    # Percent-of-portfolio child: floors of 25% and 20% inside an equity
    # sleeve capped at 30% cannot both be met.
    floors_too_big = PortfolioConstraints(
        groups={"A": "Equity", "B": "Equity", "C": "Bonds", "D": "Bonds"},
        group_bounds={"Equity": (0.0, 0.30), "Bonds": (0.0, 1.0)},
        constraint_layers=(
            ConstraintLayer(
                name="Sub",
                assignments={"A": "DM", "B": "EM"},
                limits={"DM": (0.25, 1.0), "EM": (0.20, 1.0)},
                parent=LEGACY,
            ),
        ),
    )
    report = analyze_feasibility(ASSETS, floors_too_big, MU, COV)
    found = issue(report, "child_floors_exceed_parent_cap")
    assert found.fatal
    assert "Equity" in found.message
    assert "30.00%" in found.message and "45.00%" in found.message

    # Percent-of-parent child: caps summing to 60% of a sleeve that must hold
    # at least 20% of the book force the sleeve to zero.
    caps_too_small = PortfolioConstraints(
        groups={"A": "Equity", "B": "Equity", "C": "Bonds", "D": "Bonds"},
        group_bounds={"Equity": (0.20, 0.60), "Bonds": (0.0, 1.0)},
        constraint_layers=(
            ConstraintLayer(
                name="Sub",
                assignments={"A": "DM", "B": "EM"},
                limits={"DM": (0.0, 0.30), "EM": (0.0, 0.30)},
                basis=BASIS_PARENT,
                parent=LEGACY,
            ),
        ),
    )
    report = analyze_feasibility(ASSETS, caps_too_small, MU, COV)
    found = issue(report, "relative_caps_starve_parent_floor")
    assert found.fatal
    assert "60%" in found.message and "20.00%" in found.message


def test_feasibility_names_instructions_about_assets_that_are_not_there():
    """Bounds and layers about missing assets are reported, and never fatal."""
    cons = PortfolioConstraints(
        bounds={"A": (0.0, 1.0), "Z": (0.0, 0.10)},
        constraint_layers=(
            ConstraintLayer(
                name="Region",
                assignments={"A": "US", "Y": "US"},
                limits={"US": (0.0, 0.80)},
            ),
        ),
    )
    report = analyze_feasibility(ASSETS, cons, MU, COV)

    assert report.is_feasible
    assert "Z" in issue(report, "bounds_outside_universe").message
    assert "Y" in issue(report, "layer_assets_outside_universe").message
    assert all(not i.fatal for i in report.issues)


# ---------------------------------------------------------------------------
# Stage 2: the reachable-return range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cons", "tol"),
    [
        pytest.param(
            PortfolioConstraints(
                bounds={
                    "A": (0.05, 0.40),
                    "B": (0.00, 0.30),
                    "C": (0.10, 0.50),
                    "D": (0.00, 0.25),
                }
            ),
            1e-9,
            id="budgeted",
        ),
        # Without the budget row the optimum is a vertex of the box that the
        # interior-point solver only approaches from inside, so it stops a few
        # nano-units short of the exact corner the closed form lands on.
        pytest.param(
            PortfolioConstraints(
                bounds={
                    "A": (-0.20, 0.40),
                    "B": (-0.10, 0.30),
                    "C": (0.00, 0.50),
                    "D": (-0.30, 0.25),
                },
                fully_invested=False,
                long_only=False,
            ),
            1e-8,
            id="unbudgeted",
        ),
    ],
)
def test_feasibility_knapsack_matches_lp(cons, tol):
    """The closed form is the LP's answer, without the LP."""
    assert reduces_to_box_and_budget(cons)

    closed_form = knapsack_return_range(MU, cons, ASSETS)
    lp = reachable_return_range(MU, cons, ASSETS, use_closed_form=False)

    assert closed_form is not None and lp is not None
    assert closed_form[0] == pytest.approx(lp[0], abs=tol)
    assert closed_form[1] == pytest.approx(lp[1], abs=tol)
    # And it is the closed form that `analyze_feasibility` reports.
    report = analyze_feasibility(ASSETS, cons, MU)
    assert report.reachable_return == closed_form
    assert report.stage_reached == STAGE_REACHABLE_RETURN
    assert (report.min_return, report.max_return) == closed_form


def test_box_and_budget_range_needs_no_solver(monkeypatch):
    """No solver is called for a box-and-budget mandate, so none can fail."""
    monkeypatch.setattr(
        _cvxpy_helpers,
        "solve_problem",
        _raiser(RuntimeError("no solver should be called here")),
    )
    cons = PortfolioConstraints(bounds={a: (0.0, 0.50) for a in ASSETS})

    report = analyze_feasibility(ASSETS, cons, MU)

    assert report.is_feasible
    assert report.stage_reached == STAGE_REACHABLE_RETURN
    assert report.reachable_return is not None


def test_feasibility_jointly_impossible():
    """Box, budget and layer are each fine alone; together they are not."""
    layer = ConstraintLayer(
        name="Regional",
        assignments={"A": "US", "B": "US"},
        limits={"US": (0.0, 0.30)},
    )
    bounds = {"A": (0.0, 0.50), "B": (0.0, 0.50), "C": (0.0, 0.20), "D": (0.0, 0.20)}

    # Each part on its own: the box spans the budget (caps sum to 140%), and
    # the layer is satisfiable against unrestricted bounds.
    box_only = PortfolioConstraints(bounds=bounds)
    assert analyze_feasibility(ASSETS, box_only, MU, COV).is_feasible
    layer_only = PortfolioConstraints(constraint_layers=(layer,))
    assert analyze_feasibility(ASSETS, layer_only, MU, COV).is_feasible

    both = PortfolioConstraints(bounds=bounds, constraint_layers=(layer,))
    # Nothing in the structural stage can see it: every bucket's arithmetic adds up.
    assert analyze_feasibility(ASSETS, both).is_feasible

    report = analyze_feasibility(ASSETS, both, MU, COV)
    assert not report.is_feasible
    found = issue(report, "jointly_infeasible")
    assert "'Regional' layer" in found.message
    assert "each satisfiable on their own" in found.message
    assert "Regional" in found.suggestion
    assert report.reachable_return is None


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(RuntimeError("libclarabel is not installed"), id="crash"),
        pytest.param(
            SolverFailure("solver_error", ("CLARABEL", "SCS")), id="chain-exhausted"
        ),
    ],
)
def test_feasibility_solver_crash_is_not_infeasible(monkeypatch, exc):
    """A solver that cannot answer is a solver problem, not an impossible mandate."""
    monkeypatch.setattr(_cvxpy_helpers, "solve_problem", _raiser(exc))
    cons = PortfolioConstraints(
        constraint_layers=(
            ConstraintLayer(
                name="Region",
                assignments={"A": "US", "B": "US", "C": "EU", "D": "EU"},
                limits={"US": (0.0, 0.70), "EU": (0.0, 0.70)},
            ),
        ),
    )

    report = analyze_feasibility(ASSETS, cons, MU)

    assert "solver_error" in codes(report)
    assert "jointly_infeasible" not in codes(report)
    assert "lp_infeasible" not in codes(report)
    found = issue(report, "solver_error")
    assert not found.fatal
    assert str(exc) in found.message
    # The mandate has not been shown to be impossible, so nothing claims it is.
    assert report.is_feasible
    assert report.stage_reached == STAGE_STRUCTURAL
    assert report.reachable_return is None
    assert "no solution" not in report.describe()


def test_feasibility_reports_a_failed_volatility_floor_check(monkeypatch):
    """The volatility floor is either measured or reported as unmeasured."""
    from optimization_engine.optimizers import mean_variance

    monkeypatch.setattr(
        mean_variance.MinVarianceOptimizer,
        "optimize",
        lambda self: (_ for _ in ()).throw(RuntimeError("cholesky blew up")),
    )
    cons = PortfolioConstraints(target_volatility=0.01)

    report = analyze_feasibility(ASSETS, cons, cov_matrix=COV)

    found = issue(report, "gmv_solver_error")
    assert not found.fatal
    assert "cholesky blew up" in found.message
    assert report.is_feasible
    assert report.min_variance_return is None


def test_feasibility_flags_a_volatility_target_below_the_floor():
    """The floor itself, when the minimum-variance solve does answer."""
    cons = PortfolioConstraints(target_volatility=0.01)

    report = analyze_feasibility(ASSETS, cons, MU, COV)

    found = issue(report, "target_vol_below_gmv")
    assert found.fatal
    assert not report.is_feasible


def test_stage_reached_names_the_last_stage_that_answered():
    cons = PortfolioConstraints(bounds={a: (0.0, 0.50) for a in ASSETS})

    assert analyze_feasibility(ASSETS, cons).stage_reached == STAGE_STRUCTURAL
    full = analyze_feasibility(ASSETS, cons, MU, COV)
    assert full.stage_reached == STAGE_REACHABLE_RETURN
    assert full.reachable_return == (full.min_return, full.max_return)
    assert full.min_variance_return is not None


def _raiser(exc: BaseException):
    """A stand-in for ``solve_problem`` that always fails."""

    def _boom(*args, **kwargs):
        raise exc

    return _boom


def test_a_return_floor_below_the_reachable_range_is_a_warning():
    """Every allocation clears a floor below the range, so nothing is impossible.

    Since 0.7.0 ``target_return`` is a floor (``μ'w ≥ R*``), not an equality,
    and the solve with a 1% floor on this box succeeds with the floor slack.
    The pre-flight still called it fatal — written when the target was an
    equality — so ``raise_on_infeasible`` (the CLI's ``--strict``) refused a
    mandate the optimizer answers: reachable 5.10%–10.70%, target 1%,
    ``is_feasible=False``.
    """
    from optimization_engine.optimizers.mean_variance import MeanVarianceOptimizer

    cons = PortfolioConstraints(
        bounds={a: (0.10, 0.40) for a in ASSETS}, target_return=0.01
    )

    report = analyze_feasibility(ASSETS, cons, MU, COV)

    assert report.min_return == pytest.approx(0.051)
    found = issue(report, "target_return_too_low")
    assert not found.fatal
    assert report.is_feasible
    assert "floor" in found.suggestion

    solved = MeanVarianceOptimizer(
        expected_returns=MU, cov_matrix=COV, constraints=cons
    ).optimize()
    assert solved.extras["target_return_binding"] is False
