"""A benchmark member with no return is not a member that returned 0%.

``portfolio_returns_from_weights`` filled the panel's gaps with 0.0. A 50/50
benchmark whose second member listed half-way through was therefore exactly
half of the first member before the listing — 7.3% volatility against the
14.6% of anything it could actually have held — and a book identical in spirit
showed a 5.2% tracking error and an alpha of -16.6% a year, with no warning.
The same after a delisting, and in the buy-and-hold variant.

A weight-defined benchmark is now only defined on the periods where every
member it holds has a return; elsewhere its return is NaN, which the relative
analytics already handle for an external series that does not cover the
panel. Buy-and-hold is bought on the first such period, rather than parking
the unlisted member's share at zero return until it appears.
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

from optimization_engine.benchmark import (  # noqa: E402
    BenchmarkSpec,
    portfolio_returns_from_weights,
    resolve_benchmark,
)

T = 504
HALF = T // 2
FIFTY_FIFTY = pd.Series({"A": 0.5, "B": 0.5})


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2020-01-01", periods=T)
    return pd.DataFrame(
        {"A": rng.normal(0.0004, 0.01, T), "B": rng.normal(0.0004, 0.01, T)},
        index=index,
    )


def test_a_late_listing_leaves_the_benchmark_undefined_rather_than_diluted(panel):
    """The repro: B lists half-way, so the first half was exactly 0.5·A."""
    late = panel.copy()
    late.iloc[:HALF, 1] = np.nan
    bench = resolve_benchmark(
        BenchmarkSpec(kind="custom_weights", weights=FIFTY_FIFTY.to_dict()), late
    )
    assert bench is not None
    assert bench.returns.iloc[:HALF].isna().all()
    np.testing.assert_allclose(
        bench.returns.iloc[HALF:].to_numpy(),
        (0.5 * late["A"] + 0.5 * late["B"]).iloc[HALF:].to_numpy(),
    )


def test_a_delisting_ends_the_benchmark_where_the_member_stops(panel):
    gone = panel.copy()
    gone.iloc[3 * T // 4 :, 1] = np.nan
    stream = portfolio_returns_from_weights(gone, FIFTY_FIFTY)
    assert stream.iloc[3 * T // 4 :].isna().all()
    assert stream.iloc[: 3 * T // 4].notna().all()


def test_buy_and_hold_is_bought_when_every_member_trades(panel):
    late = panel.copy()
    late.iloc[:HALF, 1] = np.nan
    held = portfolio_returns_from_weights(late, FIFTY_FIFTY, "buy_and_hold")

    assert held.iloc[:HALF].isna().all()
    # The same hold, started on the listing date, on a panel with no gaps.
    reference = portfolio_returns_from_weights(
        late.iloc[HALF:], FIFTY_FIFTY, "buy_and_hold"
    )
    np.testing.assert_allclose(held.iloc[HALF:].to_numpy(), reference.to_numpy())


def test_a_member_with_no_weight_does_not_need_a_return(panel):
    late = panel.copy()
    late.iloc[:HALF, 1] = np.nan
    stream = portfolio_returns_from_weights(late, pd.Series({"A": 1.0, "B": 0.0}))
    np.testing.assert_allclose(stream.to_numpy(), late["A"].to_numpy())


@pytest.mark.parametrize("rebalance", ["periodic", "buy_and_hold"])
def test_a_complete_panel_is_unchanged(panel, rebalance):
    stream = portfolio_returns_from_weights(panel, FIFTY_FIFTY, rebalance)
    if rebalance == "periodic":
        expected = (panel * FIFTY_FIFTY).sum(axis=1)
    else:
        nav = (1.0 + panel).cumprod().mul(FIFTY_FIFTY, axis=1).sum(axis=1)
        expected = nav / nav.shift(1).fillna(1.0) - 1.0
    np.testing.assert_allclose(stream.to_numpy(), expected.to_numpy())
