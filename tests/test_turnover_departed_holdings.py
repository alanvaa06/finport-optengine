"""Selling a holding that left the universe is turnover, and the budget counts it.

The turnover constraint and both turnover reports reindexed the previous book
onto today's universe, so a name the book held and can no longer hold dropped
out of ``Σ|w − w_prev|`` — selling it cost nothing. With 30% in such a name and
a 50% budget, the solve traded 50% inside the universe, the real turnover was
80%, the audit came back clean and ``diagnostics.turnover`` said 0.50.

The book cannot keep a name it cannot hold, so that leg is sold whatever the
solve decides: it is a constant ``Σ|w_prev,i|`` over the departed names, added
to the left side of the budget and to every reported figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.config import EngineConfig, OptimizerSpec  # noqa: E402
from optimization_engine.data.loader import prices_to_returns, sample_dataset  # noqa: E402
from optimization_engine.engine import run_engine  # noqa: E402
from optimization_engine.optimizers.base import PortfolioConstraints  # noqa: E402
from optimization_engine.optimizers.diagnostics import (  # noqa: E402
    check_constraints,
    portfolio_diagnostics,
)
from optimization_engine.optimizers.feasibility import (  # noqa: E402
    InfeasibleConstraintsError,
)


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset())


def _true_turnover(weights: pd.Series, previous: dict[str, float]) -> float:
    names = weights.index.union(pd.Index(list(previous)))
    held = pd.Series(previous).reindex(names).fillna(0.0)
    return float((weights.reindex(names).fillna(0.0) - held).abs().sum())


def _previous(assets: list[str]) -> dict[str, float]:
    """30% in a fund the universe no longer holds, the rest spread evenly."""
    return {"OLD_FUND": 0.30, **{a: 0.70 / len(assets) for a in assets}}


def test_the_budget_counts_the_sale_of_a_departed_name(returns):
    assets = list(returns.columns)
    previous = _previous(assets)
    config = EngineConfig(
        optimizer=OptimizerSpec(name="min_variance"),
        previous_weights=previous,
        turnover_limit=0.70,
    )
    run = run_engine(returns, config)

    # The solve used to spend the whole 0.70 inside the universe: 1.00 in all.
    traded = _true_turnover(run.result.weights, previous)
    assert traded <= 0.70 + 1e-4
    assert run.diagnostics.turnover == pytest.approx(traded, abs=1e-9)
    assert run.result.audit.is_clean


def test_a_budget_the_forced_sale_alone_exceeds_is_infeasible(returns):
    """The repro: a 50% budget, which used to return a book that traded 80%.

    Selling OLD_FUND is 0.30, and putting that 30% back to work inside a fully
    invested book is another 0.30, so nothing under 0.60 is reachable. The
    pre-solve analysis says so, and names the turnover budget.
    """
    assets = list(returns.columns)
    config = EngineConfig(
        optimizer=OptimizerSpec(name="min_variance"),
        previous_weights=_previous(assets),
        turnover_limit=0.50,
    )
    with pytest.raises(InfeasibleConstraintsError, match="turnover"):
        run_engine(returns, config, raise_on_infeasible=True)


def test_a_departed_holding_larger_than_the_budget_is_a_breach_not_a_free_trade():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    previous = {"A": 0.35, "B": 0.35, "OLD_FUND": 0.30}
    constraints = PortfolioConstraints(previous_weights=previous, turnover_limit=0.20)

    breaches = [v for v in check_constraints(weights, constraints) if v.kind == "turnover"]
    assert len(breaches) == 1
    assert breaches[0].actual == pytest.approx(0.60)

    diagnostics = portfolio_diagnostics(weights, constraints=constraints)
    assert diagnostics.turnover == pytest.approx(0.60)


def test_a_previous_book_inside_the_universe_is_measured_as_before():
    weights = pd.Series({"A": 0.6, "B": 0.4})
    constraints = PortfolioConstraints(
        previous_weights={"A": 0.5, "B": 0.5}, turnover_limit=0.25
    )
    assert portfolio_diagnostics(weights, constraints=constraints).turnover == pytest.approx(0.2)
    assert not check_constraints(weights, constraints)
