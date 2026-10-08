"""Frontier uncertainty and Michaud resampling."""

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

from optimization_engine.config import EngineConfig, OptimizerSpec
from optimization_engine.data.covariance import covariance_matrix
from optimization_engine.data.loader import prices_to_returns, sample_dataset
from optimization_engine.frontier import efficient_frontier
from optimization_engine.resampling import (
    ResampledFrontier,
    bootstrap_frontier,
    resample_returns,
    resampled_efficient_frontier,
)

#: Captured before any monkeypatching so a mock can still call the real one.
_REAL_FRONTIER = efficient_frontier


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=252 * 5, seed=13))


@pytest.fixture(scope="module")
def config(returns: pd.DataFrame) -> EngineConfig:
    mu = (1 + returns).prod() ** (252 / len(returns)) - 1
    return EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.4] for a in returns.columns},
        optimizer=OptimizerSpec(name="mean_variance", risk_free_rate=0.03),
    )


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["iid", "block", "parametric"])
def test_resample_preserves_shape(returns, method):
    drawn = resample_returns(returns, method=method, rng=np.random.default_rng(0))
    assert drawn.shape == returns.shape
    assert list(drawn.columns) == list(returns.columns)


def test_resampling_is_reproducible(returns):
    a = resample_returns(returns, "block", rng=np.random.default_rng(7))
    b = resample_returns(returns, "block", rng=np.random.default_rng(7))
    pd.testing.assert_frame_equal(a, b)


def test_block_bootstrap_preserves_more_autocorrelation_than_iid():
    """That is the whole reason to prefer blocks for financial returns."""
    rng = np.random.default_rng(0)
    n = 2000
    noise = rng.normal(0, 0.01, n)
    series = np.zeros(n)
    for i in range(1, n):
        series[i] = 0.6 * series[i - 1] + noise[i]
    frame = pd.DataFrame({"x": series})

    block_rho, iid_rho = [], []
    for seed in range(12):
        block_rho.append(
            abs(resample_returns(frame, "block", block_size=60,
                                 rng=np.random.default_rng(seed))["x"].autocorr(1))
        )
        iid_rho.append(
            abs(resample_returns(frame, "iid",
                                 rng=np.random.default_rng(seed))["x"].autocorr(1))
        )
    assert np.mean(block_rho) > np.mean(iid_rho) + 0.1


def test_unknown_resampling_method_is_rejected(returns):
    with pytest.raises(ValueError, match="Unknown resampling method"):
        resample_returns(returns, method="jackknife")


# ---------------------------------------------------------------------------
# Frontier uncertainty
# ---------------------------------------------------------------------------


def test_bootstrap_frontier_produces_an_ordered_band(returns, config):
    u = bootstrap_frontier(returns, config, n_draws=12, n_points=8, seed=3)
    assert u.n_draws >= 8
    q = u.quantiles
    # Quantiles must not cross.
    assert (q["q05"] <= q["q50"] + 1e-9).all()
    assert (q["q50"] <= q["q95"] + 1e-9).all()
    width = u.band_width()
    assert (width >= -1e-12).all()
    assert width.notna().any()


def test_band_is_wider_on_a_shorter_sample(returns, config):
    """Less data means more estimation error, and the band should say so."""
    long_run = bootstrap_frontier(returns, config, n_draws=12, n_points=8, seed=5)
    short = returns.iloc[-250:]
    short_config = EngineConfig(
        expected_returns=config.expected_returns,
        bounds=config.bounds,
        optimizer=config.optimizer,
    )
    short_run = bootstrap_frontier(short, short_config, n_draws=12, n_points=8, seed=5)
    assert float(short_run.band_width().median()) > float(
        long_run.band_width().median()
    )


def test_summary_names_the_band(returns, config):
    u = bootstrap_frontier(returns, config, n_draws=8, n_points=6, seed=1)
    text = u.summary()
    assert "resampled histories" in text and "%" in text


def test_weight_dispersion_flags_the_unstable_positions(returns, config):
    u = bootstrap_frontier(returns, config, n_draws=12, n_points=6, seed=2)
    assert not u.weight_dispersion.empty
    assert (u.weight_dispersion >= 0).all()
    assert u.weight_dispersion.is_monotonic_decreasing


def test_bootstrap_requires_enough_draws(returns, config):
    with pytest.raises(ValueError, match="at least 2 draws"):
        bootstrap_frontier(returns, config, n_draws=1)


def test_point_estimate_is_carried_alongside_the_band(returns, config):
    u = bootstrap_frontier(returns, config, n_draws=8, n_points=6, seed=4)
    assert u.point_estimate.n_failed == 0
    assert not u.point_estimate.plot_frame().empty


# ---------------------------------------------------------------------------
# Michaud resampling
# ---------------------------------------------------------------------------


def test_resampled_frontier_weights_are_valid(returns, config):
    result = resampled_efficient_frontier(
        returns, config, n_draws=10, n_points=6, seed=1
    )
    assert isinstance(result, ResampledFrontier)
    averaged = result.weights
    assert result.n_draws + result.n_failed == 10
    assert not averaged.empty
    np.testing.assert_allclose(averaged.sum().values, 1.0, atol=1e-6)
    assert (averaged.values >= -1e-9).all()
    assert (averaged.values <= 0.4 + 1e-6).all()


def test_resampled_frontier_is_more_diversified_than_the_point_estimate(
    returns, config
):
    """Averaging weights across draws is the point: it stops the optimizer
    acting on differences the sample cannot resolve."""
    averaged = resampled_efficient_frontier(
        returns, config, n_draws=12, n_points=8, seed=1
    ).weights
    point = efficient_frontier(
        config,
        covariance_matrix(returns),
        pd.Series(config.expected_returns),
        n_points=8,
    ).weights

    def effective_n(frame):
        return float(np.mean([1.0 / (frame[c] ** 2).sum() for c in frame.columns]))

    assert effective_n(averaged) > effective_n(point)


def test_resampled_frontier_is_reproducible(returns, config):
    a = resampled_efficient_frontier(returns, config, n_draws=6, n_points=5, seed=9)
    b = resampled_efficient_frontier(returns, config, n_draws=6, n_points=5, seed=9)
    pd.testing.assert_frame_equal(a.weights, b.weights)
    assert (a.n_draws, a.n_failed) == (b.n_draws, b.n_failed)


def _mock_frontier_failures(monkeypatch, fails, message="solver exploded"):
    """Make ``fails(i)`` decide whether draw ``i`` raises. Counts the raises."""
    import optimization_engine.resampling as res

    calls = {"i": 0, "raised": 0}

    def flaky(*args, **kwargs):
        i = calls["i"]
        calls["i"] += 1
        if fails(i):
            calls["raised"] += 1
            raise RuntimeError(message)
        return _REAL_FRONTIER(*args, **kwargs)

    monkeypatch.setattr(res, "efficient_frontier", flaky)
    return calls


def test_michaud_reports_failed_draws(returns, config, monkeypatch):
    """A draw that raised is a draw that did not vote. Say how many."""
    calls = _mock_frontier_failures(monkeypatch, lambda i: i % 3 == 0)
    result = resampled_efficient_frontier(
        returns, config, n_draws=9, n_points=5, seed=2
    )
    assert calls["raised"] == 3
    assert result.n_failed == calls["raised"]
    assert result.n_draws == 9 - calls["raised"]
    assert result.first_error is not None
    assert "solver exploded" in result.first_error
    assert "3 draw(s) produced nothing to average" in result.summary()


def test_michaud_refuses_minority_average(returns, config, monkeypatch):
    """Averaging the third of draws where the mandate did not bind is not a
    resampled portfolio, so it refuses rather than returning a number."""
    # Two of every three draws raise, so six fail and three solve.
    calls = _mock_frontier_failures(
        monkeypatch, lambda i: i % 3 != 2, message="mandate infeasible"
    )
    with pytest.raises(ValueError, match="is not a resampled portfolio"):
        resampled_efficient_frontier(returns, config, n_draws=9, n_points=5, seed=2)
    assert calls["raised"] == 6


def test_michaud_counts_draws_that_solved_too_few_ranks(returns, config, monkeypatch):
    """The second silent drop: no exception, just a frontier too sparse to
    rank. It must be counted like any other lost draw."""
    import optimization_engine.resampling as res

    real = res.efficient_frontier
    calls = {"i": 0}

    def sparse(*args, **kwargs):
        result = real(*args, **kwargs)
        i = calls["i"]
        calls["i"] += 1
        if i % 3 == 0:
            # No exception anywhere — the frontier simply did not solve
            # enough ranks to be placed on the grid.
            result.summary["status"] = ["failed"] * len(result.summary)
        return result

    monkeypatch.setattr(res, "efficient_frontier", sparse)
    result = resampled_efficient_frontier(
        returns, config, n_draws=9, n_points=5, seed=2
    )
    assert result.n_failed == 3
    assert result.n_draws == 6
    assert result.first_error is not None
    assert "frontier ranks" in result.first_error
    # Nothing raised, so the message cannot be an exception repr.
    assert "Error" not in result.first_error


def _mock_rank_failures(monkeypatch, failing):
    """Make rank ``r`` of draw ``i`` fail whenever ``(i, r)`` is in ``failing``.

    The frontier still comes back — the failure is a status, not an
    exception — which is the case the rank grid has to survive.
    """
    import copy

    import optimization_engine.resampling as res

    calls = {"i": 0}

    def holed(*args, **kwargs):
        result = copy.deepcopy(_REAL_FRONTIER(*args, **kwargs))
        i = calls["i"]
        calls["i"] += 1
        for draw, rank in failing:
            if draw == i:
                result.summary.loc[result.summary.index[rank], "status"] = "failed"
                result.weights.iloc[:, rank] = np.nan
        return result

    monkeypatch.setattr(res, "efficient_frontier", holed)


def test_a_rank_lost_in_one_draw_does_not_shift_the_others(
    returns, config, monkeypatch
):
    """Ranks are averaged by position, each over the draws where it solved.

    The solved ranks used to be renumbered from zero per draw and the grid cut
    to the shortest draw, so one failed middle rank in one draw moved every
    rank above it down a slot in that draw, and the top rank vanished from
    the result altogether — while the summary said all eight draws averaged
    cleanly.
    """
    clean = resampled_efficient_frontier(returns, config, n_draws=8, n_points=6, seed=1)
    _mock_rank_failures(monkeypatch, {(0, 2)})
    holed = resampled_efficient_frontier(returns, config, n_draws=8, n_points=6, seed=1)

    assert list(holed.weights.columns) == list(clean.weights.columns)
    assert len(holed.weights.columns) == 6
    assert holed.rank_counts["rank_2"] == 7
    assert (holed.rank_counts.drop("rank_2") == 8).all()
    others = [c for c in clean.weights.columns if c != "rank_2"]
    pd.testing.assert_frame_equal(holed.weights[others], clean.weights[others])
    np.testing.assert_allclose(holed.weights.sum().values, 1.0, atol=1e-6)
    assert holed.n_failed == 0
    assert "rank_2" in holed.summary()


def test_a_rank_that_solved_in_a_minority_of_draws_is_dropped_and_named(
    returns, config, monkeypatch
):
    """The draw-level rule, applied per rank: a minority average is refused."""
    _mock_rank_failures(monkeypatch, {(i, 5) for i in range(5)})
    result = resampled_efficient_frontier(returns, config, n_draws=8, n_points=6, seed=1)

    assert "rank_5" not in result.weights.columns
    assert result.rank_counts["rank_5"] == 3
    assert "rank_5" in result.summary()


def test_the_point_frontier_is_centred_on_the_estimator_the_draws_use(returns):
    """The band re-estimates mu from each draw, so the centre must too.

    The point curve was traced on the config's own expected returns while
    every draw used historical ones, so a config carrying capital-market
    assumptions drew a flat 3% line through a band whose median ran from 0%
    to 6%.
    """
    from optimization_engine.data.covariance import (
        covariance_from_config,
        expected_returns_from_history,
    )

    assumptions = {a: 0.03 for a in returns.columns}
    with_cmas = EngineConfig(
        expected_returns=assumptions,
        bounds={a: [0.0, 0.4] for a in returns.columns},
        optimizer=OptimizerSpec(name="mean_variance"),
    )
    band = bootstrap_frontier(returns, with_cmas, n_draws=4, n_points=6, seed=0)

    historical = expected_returns_from_history(returns, method="mean")
    expected = efficient_frontier(
        EngineConfig(
            expected_returns=historical.to_dict(),
            bounds=with_cmas.bounds,
            optimizer=with_cmas.optimizer,
        ),
        covariance_from_config(returns, with_cmas),
        expected_returns=historical,
        returns=returns,
        n_points=6,
    )
    np.testing.assert_allclose(
        band.point_estimate.summary["expected_return"].values,
        expected.summary["expected_return"].values,
    )

    held = bootstrap_frontier(
        returns, with_cmas, n_draws=4, n_points=6, seed=0,
        reestimate_expected_returns=False,
    )
    np.testing.assert_allclose(held.point_estimate.summary["expected_return"], 0.03)
    np.testing.assert_allclose(held.quantiles.values, 0.03)


def test_bootstrap_estimates_mu_when_the_config_carries_none(returns):
    """Every draw estimates its own mu; the centre used to refuse instead."""
    bare = EngineConfig(
        bounds={a: [0.0, 0.4] for a in returns.columns},
        optimizer=OptimizerSpec(name="mean_variance"),
    )
    band = bootstrap_frontier(returns, bare, n_draws=4, n_points=5, seed=0)
    assert band.n_failed == 0
    assert band.point_estimate.n_failed == 0


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def test_point_estimate_is_clipped_to_the_band_range(returns, config):
    """A band that stops mid-plot beside a full-length line reads as broken."""
    from optimization_engine.reporting.plots import plot_frontier_uncertainty

    u = bootstrap_frontier(returns, config, n_draws=10, n_points=8, seed=8)
    fig = plot_frontier_uncertainty(u)
    point = next(t for t in fig.data if str(t.name).startswith("Point"))
    assert min(point.x) >= u.volatility.min() - 1e-9
    assert max(point.x) <= u.volatility.max() + 1e-9


def test_uncertainty_plots_render(returns, config):
    from optimization_engine.reporting.plots import (
        plot_frontier_uncertainty,
        plot_weight_dispersion,
    )

    u = bootstrap_frontier(returns, config, n_draws=8, n_points=6, seed=6)
    fig = plot_frontier_uncertainty(u)
    names = {t.name for t in fig.data}
    assert "Point estimate (observed sample)" in names
    assert any("percentile" in n for n in names)
    assert plot_weight_dispersion(u.weight_dispersion).data
