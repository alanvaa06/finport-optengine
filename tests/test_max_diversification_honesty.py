"""Max-diversification must not dress a failure up as an answer.

Three properties, all of which the optimizer used to get wrong:

* an infeasible mandate is a property of the *problem*, so it raises rather
  than quietly returning a portfolio that breaks the mandate;
* the projection fallback is reachable only from numerical failure; and
* when it is taken, the result says ``fallback_projection`` instead of
  inheriting the *unconstrained* solve's ``optimal``.
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

from optimization_engine.optimizers import max_diversification as max_div_module
from optimization_engine.optimizers._cvxpy_helpers import SolverFailure
from optimization_engine.optimizers.base import PortfolioConstraints
from optimization_engine.optimizers.max_diversification import (
    MaxDiversificationOptimizer,
)

ASSETS = ["A", "B", "C", "D"]


@pytest.fixture(scope="module")
def cov() -> pd.DataFrame:
    corr = np.array(
        [
            [1.0, 0.8, 0.2, 0.7],
            [0.8, 1.0, 0.1, 0.6],
            [0.2, 0.1, 1.0, 0.3],
            [0.7, 0.6, 0.3, 1.0],
        ]
    )
    vol = np.array([0.20, 0.15, 0.25, 0.18])
    return pd.DataFrame(np.outer(vol, vol) * corr, index=ASSETS, columns=ASSETS)


def _mandate(max_tracking_error: float, **kwargs) -> PortfolioConstraints:
    """A concentrated benchmark the 30% box cannot track closely."""
    return PortfolioConstraints(
        bounds={a: (0.0, 0.30) for a in ASSETS},
        benchmark_weights={"A": 1.0},
        max_tracking_error=max_tracking_error,
        **kwargs,
    )


def test_max_div_infeasible_mandate_raises(cov):
    """A TE budget below the reachable minimum is infeasible, not a fallback.

    The benchmark is 100% A and no weight may exceed 30%, so the closest the
    book can get to the index is a tracking error of roughly 12%. A 5% budget
    therefore has no feasible allocation at all — and an unreachable mandate
    is a property of the problem, which the caller has to be told about.
    """
    optimizer = MaxDiversificationOptimizer(
        cov_matrix=cov, constraints=_mandate(0.05)
    )

    with pytest.raises(SolverFailure) as raised:
        optimizer.optimize()

    assert raised.value.status == "infeasible"
    assert "infeasible" in str(raised.value)
    # Nothing was returned, so nothing was recorded as if it had been.
    assert "projection_distance" not in optimizer._diagnostics
    assert "fallback_reason" not in optimizer._diagnostics


def test_max_div_fallback_never_reports_optimal(cov, monkeypatch):
    """Numerical failure earns the fallback — and the fallback says so.

    The inner unconstrained solve genuinely returns ``optimal``; merging its
    ``SolveInfo`` wholesale is what used to label a projected, mandate-
    breaching book as the optimum.
    """
    real_solve = max_div_module.solve_problem
    calls = {"n": 0}

    def one_numerical_failure(problem, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise SolverFailure("solver_error", ("CLARABEL", "SCS"))
        return real_solve(problem, *args, **kwargs)

    monkeypatch.setattr(max_div_module, "solve_problem", one_numerical_failure)

    # A budget that is reachable, so only the monkeypatch can fail the solve.
    result = MaxDiversificationOptimizer(
        cov_matrix=cov, constraints=_mandate(0.20)
    ).optimize()

    assert calls["n"] == 2, "the constrained solve should have been retried once"
    assert result.extras["solver_status"] == "fallback_projection"
    assert result.extras["bounds_mode"] == "soft_iterated"
    assert "max_tracking_error" in result.extras["dropped_constraints"]
    # The unconstrained solve's objective describes a portfolio nobody got.
    assert "objective_value" not in result.extras
    # The banner's two keys survive untouched.
    assert result.extras["projection_distance"] >= 0.0
    assert "projection" in result.extras["fallback_reason"]


def test_max_div_dropped_constraints_names_only_what_projection_loses(cov):
    """``dropped_constraints`` is a claim about the projection, so it must be true.

    The projection re-imposes the mandate by solving ``min ‖x − w‖²`` under
    it, so an active-share cap does survive — and naming it as dropped would
    be the same class of lie this module exists to prevent.
    """
    optimizer = MaxDiversificationOptimizer(
        cov_matrix=cov,
        constraints=_mandate(0.20, max_active_share=0.40, leverage=1.0),
    )
    assert optimizer._dropped_by_projection() == ["max_tracking_error"]

    # No active-share cap and no bucket budget: the projection used to clip
    # and redistribute, blind to gross exposure. A cap that can bind now sends
    # it down the exact path (review item O15), so leverage is not dropped.
    bare = MaxDiversificationOptimizer(
        cov_matrix=cov,
        constraints=PortfolioConstraints(
            bounds={a: (-1.0, 1.0) for a in ASSETS},
            long_only=False,
            leverage=1.1,
        ),
    )
    assert bare._dropped_by_projection() == []


def test_max_div_refuses_an_inaccurate_answer_instead_of_projecting(cov, monkeypatch):
    """A refused ``optimal_inaccurate`` is a refusal, not a numerical failure.

    ``docs/ERRORS.md`` promises an answer no solver can verify is "refused, not
    returned". Max-diversification caught that refusal like any other solver
    failure and projected the *unconstrained* solve instead — dropping the
    tracking-error budget, labelling the result ``fallback_projection`` and
    returning weights for a solve the caller had just declined.
    """
    real_solve = max_div_module.solve_problem
    calls = {"n": 0}

    def inaccurate_once(problem, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise SolverFailure("optimal_inaccurate", ("CLARABEL", "SCS", "OSQP"))
        return real_solve(problem, *args, **kwargs)

    monkeypatch.setattr(max_div_module, "solve_problem", inaccurate_once)
    optimizer = MaxDiversificationOptimizer(
        cov_matrix=cov, constraints=_mandate(0.20), accept_inaccurate=False
    )

    with pytest.raises(SolverFailure) as raised:
        optimizer.optimize()

    assert raised.value.status == "optimal_inaccurate"
    assert calls["n"] == 1, "nothing was solved after the refusal"
    assert "fallback_reason" not in optimizer._diagnostics


def test_max_div_second_solve_reports_its_own_diagnostics(cov, monkeypatch):
    """A fallback on one solve must not be reported by the next.

    The fallback flipped ``bounds_mode`` on the instance and nothing reset
    ``_diagnostics`` between calls, so a second, clean solve on the same
    optimizer still said ``soft_iterated`` and carried the first solve's
    ``projection_distance``, ``fallback_reason`` and ``dropped_constraints``.
    """
    real_solve = max_div_module.solve_problem
    calls = {"n": 0}

    def one_numerical_failure(problem, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise SolverFailure("solver_error", ("CLARABEL", "SCS"))
        return real_solve(problem, *args, **kwargs)

    monkeypatch.setattr(max_div_module, "solve_problem", one_numerical_failure)
    optimizer = MaxDiversificationOptimizer(cov_matrix=cov, constraints=_mandate(0.20))

    first = optimizer.optimize()
    second = optimizer.optimize()

    assert first.extras["solver_status"] == "fallback_projection"
    assert second.extras["solver_status"] == "optimal"
    assert second.extras["bounds_mode"] == "hard"
    for stale in ("projection_distance", "fallback_reason", "dropped_constraints"):
        assert stale not in second.extras, stale
    # The first result is not rewritten by the second solve either.
    assert first.extras["bounds_mode"] == "soft_iterated"
