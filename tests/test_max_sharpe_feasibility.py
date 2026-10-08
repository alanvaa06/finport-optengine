"""Max-Sharpe must say why it has no answer, and must find one when it exists.

The tangency portfolio is solved on a ray: minimize ``y'Σy`` subject to
``(μ − rf)'y = 1`` and normalize ``w = y/Σy``. That transform has three ways to
go wrong that look alike from outside — every one used to surface as a bare
"infeasible" or as weights nobody should hold:

* no allocation the mandate allows earns more than cash, so no portfolio has a
  positive Sharpe ratio and the ray has nowhere to point;
* the best excess return is positive but tiny, so ``(μ − rf)'y = 1`` needs a
  ray so long the solver's tolerances call it infeasible; and
* with no finite box, the ratio keeps rising as the book levers up and no
  finite portfolio attains it.
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

from optimization_engine.optimizers._cvxpy_helpers import SolverFailure
from optimization_engine.optimizers.base import PortfolioConstraints
from optimization_engine.optimizers.mean_variance import MaxSharpeOptimizer


def test_unbounded_tangency_is_refused_rather_than_levered_without_limit():
    """With no finite box, a long-short tangency can have no maximizer.

    When ``1'Σ⁻¹(μ − rf) < 0`` the stationary point on the budget hyperplane is
    the *minimum* Sharpe ratio, and the maximum is only approached as the book
    levers up without limit. The ray solve then pins ``κ = Σy`` at its floor of
    1e-8 and ``y/κ`` came back as ±1.9e9 — reported ``optimal``. The degenerate
    check that existed for this compared ``κ`` against 1e-10, below the floor
    the program itself imposes, so it could never fire.
    """
    names = ["p", "q"]
    vols = np.array([0.10, 0.20])
    corr = np.array([[1.0, 0.95], [0.95, 1.0]])
    cov = pd.DataFrame(np.outer(vols, vols) * corr, index=names, columns=names)
    mu = pd.Series([0.01, 0.05], index=names)
    assert np.ones(2) @ np.linalg.inv(cov.values) @ mu.values < 0, "fixture drifted"

    optimizer = MaxSharpeOptimizer(
        expected_returns=mu,
        cov_matrix=cov,
        constraints=PortfolioConstraints(
            long_only=False, bounds={a: (-np.inf, np.inf) for a in names}
        ),
    )
    with pytest.raises(SolverFailure) as raised:
        optimizer.optimize()

    assert raised.value.status == "unbounded"
    assert "leverage" in str(raised.value)


NAMES = ["a", "b", "c"]


def _cov() -> pd.DataFrame:
    vol = np.array([0.10, 0.15, 0.20])
    corr = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.5], [0.3, 0.5, 1.0]])
    return pd.DataFrame(np.outer(vol, vol) * corr, index=NAMES, columns=NAMES)


def test_a_box_that_cannot_beat_cash_is_named_not_called_infeasible():
    """One asset beats cash and the box caps it: the best book earns −0.5% excess.

    The old pre-check looked asset by asset, saw ``a`` above the risk-free
    rate and let the solve run; the ray ``(μ − rf)'y = 1`` then has no
    solution with ``Σy > 0``, which came back as SolverFailure "no allocation
    satisfies every constraint" — false, the box and the budget are fine.
    The best feasible excess return is now computed (the knapsack closed form
    here, an LP under layers or benchmark budgets) and named.
    """
    from optimization_engine.optimizers.mean_variance import NoPositiveExcessReturnError

    mu = pd.Series([0.06, 0.02, 0.01], index=NAMES)
    constraints = PortfolioConstraints(
        bounds={"a": (0.0, 0.20), "b": (0.0, 0.50), "c": (0.0, 0.60)}
    )
    optimizer = MaxSharpeOptimizer(
        expected_returns=mu, cov_matrix=_cov(), constraints=constraints,
        risk_free_rate=0.03,
    )

    with pytest.raises(NoPositiveExcessReturnError) as raised:
        optimizer.optimize()

    # 0.2·3% + 0.5·(−1%) + 0.3·(−2%) = −0.5%: the knapsack's best.
    assert raised.value.best_excess_return == pytest.approx(-0.005)
    assert "-0.50%" in str(raised.value)
    assert isinstance(raised.value, ValueError)


def test_a_tiny_but_positive_excess_return_still_solves():
    """1e-7 of excess return is a tangency portfolio, not an infeasible problem.

    ``(μ − rf)'y = 1`` with an excess of 1e-7 needs a ray of length 1e7, past
    the feasibility tolerance of every solver in the chain — CLARABEL, SCS and
    OSQP all answered ``infeasible``, so walking further down the chain would
    not have helped. The ray is now normalized by the best feasible excess, so
    ``κ = Σy`` sits near one whatever the scale of the returns.
    """
    mu = pd.Series([0.0200001, 0.0199, 0.0198], index=NAMES)
    result = MaxSharpeOptimizer(
        expected_returns=mu, cov_matrix=_cov(), risk_free_rate=0.02
    ).optimize()

    assert result.extras["solver_status"] == "optimal"
    np.testing.assert_allclose(result.weights.values, [1.0, 0.0, 0.0], atol=1e-6)


def test_a_long_short_book_can_beat_cash_when_no_asset_does():
    """Review item O10: every asset below rf is not every *portfolio* below rf.

    Long ``a`` twice and short ``c`` once earns 2% over a 2% risk-free rate
    with assets that earn 1%, 0% and −2% — a Sharpe of 0.085. The per-asset
    guard refused the whole problem regardless of ``long_only``.
    """
    mu = pd.Series([0.01, 0.00, -0.02], index=NAMES)
    cov = _cov()
    constraints = PortfolioConstraints(
        long_only=False, bounds={a: (-3.0, 3.0) for a in NAMES}
    )
    witness = np.array([2.0, 0.0, -1.0])
    witness_sharpe = float(
        witness @ (mu.values - 0.02) / np.sqrt(witness @ cov.values @ witness)
    )
    assert witness_sharpe == pytest.approx(0.085, abs=1e-3)

    result = MaxSharpeOptimizer(
        expected_returns=mu, cov_matrix=cov, constraints=constraints,
        risk_free_rate=0.02,
    ).optimize()

    assert result.sharpe_ratio >= witness_sharpe - 1e-9
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert result.is_compliant, result.violations
