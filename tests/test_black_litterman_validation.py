"""Input validation for Black-Litterman views.

A view the engine cannot express is not a view. These tests pin the two ways
that used to be papered over silently:

* a view naming an asset outside the optimization universe — dropped entirely
  when *every* leg was missing, and quietly reduced to its surviving legs when
  only some were, so "long AAPL vs short NOPE" became "long AAPL";
* a view whose pick portfolio has numerically zero prior variance — floored to
  ``1e-12``, i.e. handed near-infinite confidence, which then dominated the
  posterior it was meant to nudge.

Both now raise, and the message has to be actionable: the caller (the CLI, the
app's saved-scenario reload) surfaces it verbatim.
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

from optimization_engine.optimizers.base import PortfolioConstraints
from optimization_engine.optimizers.black_litterman import (
    BlackLittermanOptimizer,
    View,
    black_litterman_posterior,
    build_pick_matrix,
)

ASSETS = ["AAPL", "MSFT", "GLD"]


@pytest.fixture
def cov() -> pd.DataFrame:
    """A small, positive-definite, nowhere-collinear prior."""
    data = np.array(
        [
            [0.0400, 0.0180, 0.0020],
            [0.0180, 0.0320, 0.0015],
            [0.0020, 0.0015, 0.0250],
        ]
    )
    return pd.DataFrame(data, index=ASSETS, columns=ASSETS)


@pytest.fixture
def market(cov: pd.DataFrame) -> pd.Series:
    return pd.Series([0.5, 0.3, 0.2], index=cov.columns)


@pytest.fixture
def collinear_cov() -> pd.DataFrame:
    """``AAPL_CLONE`` is an exact duplicate column of ``AAPL``.

    The spread ``AAPL − AAPL_CLONE`` therefore has *exactly* zero prior
    variance: ``P · τΣ · Pᵀ`` is 0.0 in floating point, not merely small.
    """
    names = ["AAPL", "MSFT", "AAPL_CLONE"]
    data = np.array(
        [
            [0.0400, 0.0180, 0.0400],
            [0.0180, 0.0320, 0.0180],
            [0.0400, 0.0180, 0.0400],
        ]
    )
    return pd.DataFrame(data, index=names, columns=names)


# ---------------------------------------------------------------------------
# (a) Views on assets outside the universe
# ---------------------------------------------------------------------------


def test_bl_view_outside_universe_raises(cov: pd.DataFrame, market: pd.Series) -> None:
    """Both view forms raise, and the message names the missing asset."""
    # Absolute view, mapping form — the whole view used to be dropped.
    with pytest.raises(ValueError, match="NOPE") as absolute:
        black_litterman_posterior(cov, market, {"NOPE": 0.05})
    assert "universe" in str(absolute.value)

    # Basket / relative view — the missing leg used to be zeroed.
    basket = View({"AAPL": 1.0, "NOPE": -1.0}, 0.02, label="AAPL vs NOPE")
    with pytest.raises(ValueError, match="NOPE") as relative:
        black_litterman_posterior(cov, market, [basket])
    assert "universe" in str(relative.value)

    # And directly on the public pick-matrix builder, which is where the
    # dropping lived.
    with pytest.raises(ValueError, match="NOPE"):
        build_pick_matrix([basket], list(cov.columns))
    with pytest.raises(ValueError, match="NOPE"):
        build_pick_matrix([View({"NOPE": 1.0}, 0.05)], list(cov.columns))


def test_bl_view_outside_universe_names_every_missing_asset(cov: pd.DataFrame) -> None:
    """One message, every missing name — not just the first one hit."""
    views = [
        View({"AAPL": 1.0, "NOPE": -1.0}, 0.02, label="basket"),
        View({"ZZZZ": 1.0}, 0.05, label="absolute"),
    ]
    with pytest.raises(ValueError) as excinfo:
        build_pick_matrix(views, list(cov.columns))
    message = str(excinfo.value)
    assert "NOPE" in message
    assert "ZZZZ" in message


def test_bl_view_partially_outside_universe_raises(
    cov: pd.DataFrame, market: pd.Series
) -> None:
    """A half-held basket is the dangerous case: it used to solve, wrongly.

    ``{"AAPL": 1.0, "NOPE": -1.0}`` silently became the one-legged absolute
    view ``{"AAPL": 1.0}`` — a different opinion with a different sign on the
    posterior, asserted with no warning at all.
    """
    basket = View({"AAPL": 1.0, "NOPE": -1.0}, 0.02, label="AAPL vs NOPE")

    with pytest.raises(ValueError) as excinfo:
        build_pick_matrix([basket], list(cov.columns))
    message = str(excinfo.value)
    assert "NOPE" in message
    assert "AAPL vs NOPE" in message  # the offending view is identifiable

    # The one-legged reading is not merely unreported — it is not produced.
    posterior_absolute, _ = black_litterman_posterior(
        cov, market, [View({"AAPL": 1.0}, 0.02)]
    )
    with pytest.raises(ValueError):
        black_litterman_posterior(cov, market, [basket])

    # Control: with the second leg in the universe, the basket is a real
    # two-legged view and nothing raises.
    wide = cov.copy()
    wide.loc["NOPE", :] = [0.0, 0.0, 0.0]
    wide["NOPE"] = [0.0, 0.0, 0.0, 0.0300]
    P, Q, kept = build_pick_matrix([basket], list(wide.columns))
    assert P.shape == (1, 4)
    assert P[0].tolist() == [1.0, 0.0, 0.0, -1.0]
    assert Q.tolist() == [0.02]
    assert kept == [basket]
    assert posterior_absolute["AAPL"] > 0  # the fixture is sane


def test_bl_optimizer_surfaces_the_missing_asset(cov: pd.DataFrame) -> None:
    """The message is what the app's error surface shows, so it must act.

    Saved scenarios carry views for assets that may be absent from the panel
    now being optimized; the message has to name them and say what to do.
    """
    bl = BlackLittermanOptimizer(
        cov_matrix=cov,
        market_weights=pd.Series([0.5, 0.3, 0.2], index=cov.columns),
        views={"NOPE": 0.05},
    )
    with pytest.raises(ValueError) as excinfo:
        bl.optimize()
    message = str(excinfo.value)
    assert "NOPE" in message
    assert "universe" in message
    assert "Drop" in message or "drop" in message


# ---------------------------------------------------------------------------
# (b) Degenerate views
# ---------------------------------------------------------------------------


def test_bl_degenerate_view_raises(collinear_cov: pd.DataFrame) -> None:
    """A spread between two identical assets has no prior variance to scale Ω by."""
    market = pd.Series([0.4, 0.3, 0.3], index=collinear_cov.columns)
    labelled = View(
        {"AAPL": 1.0, "AAPL_CLONE": -1.0}, 0.02, label="AAPL beats its own clone"
    )

    # Precondition: the projection really is exactly zero, so this is the
    # floor's territory and not a tolerance argument.
    P, _, _ = build_pick_matrix([labelled], list(collinear_cov.columns))
    assert float(np.diag(P @ (0.05 * collinear_cov.values) @ P.T)[0]) == 0.0

    with pytest.raises(ValueError) as by_label:
        black_litterman_posterior(collinear_cov, market, [labelled])
    assert "AAPL beats its own clone" in str(by_label.value)

    # Unlabelled: identified by position and by the assets it names.
    unlabelled = View({"AAPL": 1.0, "AAPL_CLONE": -1.0}, 0.02)
    with pytest.raises(ValueError) as by_index:
        black_litterman_posterior(collinear_cov, market, [unlabelled])
    message = str(by_index.value)
    assert "AAPL_CLONE" in message
    assert "1" in message  # the view's 1-based position


def test_bl_zero_confidence_still_raises_on_positive_message(
    cov: pd.DataFrame, market: pd.Series
) -> None:
    """The ``confidence=0.0`` path is untouched — it has its own message."""
    with pytest.raises(ValueError, match="positive"):
        black_litterman_posterior(
            cov, market, [View({"GLD": 1.0}, 0.10, confidence=0.0)]
        )


# ---------------------------------------------------------------------------
# Regression guard: ordinary views are untouched
# ---------------------------------------------------------------------------


def test_bl_valid_views_still_solve(cov: pd.DataFrame, market: pd.Series) -> None:
    """In-universe views — absolute and relative — behave exactly as before."""
    prior, prior_cov = black_litterman_posterior(cov, market, None)

    absolute = View({"AAPL": 1.0}, 0.20, label="AAPL absolute")
    relative = View({"AAPL": 1.0, "MSFT": -1.0}, 0.03, label="AAPL over MSFT")

    P, Q, kept = build_pick_matrix([absolute, relative], list(cov.columns))
    assert P.shape == (2, 3)
    np.testing.assert_allclose(P[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(P[1], [1.0, -1.0, 0.0])
    np.testing.assert_allclose(Q, [0.20, 0.03])
    assert kept == [absolute, relative]

    post, post_cov = black_litterman_posterior(cov, market, [absolute])
    assert post["AAPL"] > prior["AAPL"]
    assert list(post.index) == list(cov.columns)
    assert post_cov.shape == cov.shape

    # A relative view moves the spread toward the view without raising.
    post_rel, _ = black_litterman_posterior(cov, market, [relative])
    prior_spread = float(prior["AAPL"] - prior["MSFT"])
    post_spread = float(post_rel["AAPL"] - post_rel["MSFT"])
    assert prior_spread < post_spread <= 0.03

    # The mapping form and an explicit confidence both still work end to end.
    bl = BlackLittermanOptimizer(
        cov_matrix=cov,
        market_weights=market,
        views={"AAPL": 0.20},
        view_confidences={"AAPL": 0.0004},
        constraints=PortfolioConstraints(
            long_only=True,
            fully_invested=True,
            bounds={a: (0.0, 1.0) for a in cov.columns},
        ),
    )
    weights = bl.optimize().weights
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-6)
    assert weights["AAPL"] > float(market["AAPL"])
    assert prior_cov.equals(cov)


# ---------------------------------------------------------------------------
# (c) Risk-aversion calibration with nothing to calibrate against
# ---------------------------------------------------------------------------


def test_bl_calibration_without_a_market_return_is_refused(cov: pd.DataFrame) -> None:
    """``calibrate_risk_aversion=True`` alone used to fall back to δ = 2.5.

    Calibration implies δ from the market's own Sharpe ratio, so it needs the
    market's expected return. Without one the solve quietly used the hand-set
    default the flag exists to replace, and reported no ``implied_risk_aversion``
    — the caller asked for a calibrated prior and got the uncalibrated one.
    """
    from optimization_engine.config import EngineConfig, OptimizerSpec
    from optimization_engine.optimizers import ConfigurationError
    from optimization_engine.optimizers.factory import effective_expected_returns

    with pytest.raises(ConfigurationError, match="market_return"):
        BlackLittermanOptimizer(cov_matrix=cov, calibrate_risk_aversion=True)

    # The pre-flight builds the same posterior without an optimizer, so it
    # refuses the same config rather than previewing a prior the solve rejects.
    cfg = EngineConfig(
        optimizer=OptimizerSpec(name="black_litterman", bl_calibrate_risk_aversion=True)
    )
    with pytest.raises(ConfigurationError, match="market_return"):
        effective_expected_returns(cfg, cov)
