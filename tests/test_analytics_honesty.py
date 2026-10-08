"""Numbers the analytics reported that were not the number their label named.

Each section pins one defect found in the 2026-10-08 review, with the
reproduction that exposed it. None of them raised: every one produced a
plausible figure — a drawdown, a Sharpe, a tracking error — that was simply
the wrong one, which is the kind of error nothing downstream catches.
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

from optimization_engine.analytics.performance import (
    calmar_ratio,
    drawdown,
    summary_stats,
    time_under_water,
)
from optimization_engine.analytics.relative import relative_drawdown
from optimization_engine.analytics.risk import (
    drawdown_series,
    drawdown_table,
    max_drawdown_duration,
    ulcer_index,
)
from optimization_engine.backtest.sweep import SweepSpec, run_sweep
from optimization_engine.config import EngineConfig, OptimizerSpec

PPY = 252


def _days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def _reference_drawdown(returns: pd.Series) -> pd.Series:
    """Drawdown measured the long way: wealth from 1.0, peak including 1.0."""
    wealth = np.cumprod(np.r_[1.0, 1.0 + returns.to_numpy(dtype=float)])
    dd = wealth / np.maximum.accumulate(wealth) - 1.0
    return pd.Series(dd[1:], index=returns.index)


# ---------------------------------------------------------------------------
# 1. A drawdown is measured from the capital invested, not from the first close
# ---------------------------------------------------------------------------
#
# The running peak used to start at the wealth *after* the first return, so a
# series that opened with a loss never counted it: [-10%, -10%, +5%, +1%]
# reported a -10% maximum drawdown against a true -19%.


@pytest.fixture()
def opens_losing() -> pd.Series:
    return pd.Series([-0.10, -0.10, 0.05, 0.01], index=_days(4))


def test_drawdown_counts_a_loss_on_the_first_period(opens_losing):
    dd = drawdown_series(opens_losing)
    assert float(dd.min()) == pytest.approx(-0.19)
    pd.testing.assert_series_equal(dd, _reference_drawdown(opens_losing))


def test_a_monotone_loser_is_underwater_by_its_whole_loss():
    r = pd.Series([-0.05] * 3, index=_days(3))
    assert float(drawdown_series(r).min()) == pytest.approx(0.95**3 - 1.0)


def test_a_series_that_opens_with_a_gain_is_unchanged():
    r = pd.Series([0.10, -0.20, -0.10, 0.05, 0.40, 0.01, -0.05, 0.02], index=_days(8))
    growth = np.log1p(r).cumsum()
    legacy = np.expm1(growth - growth.cummax())
    pd.testing.assert_series_equal(drawdown_series(r), legacy, check_names=False)
    pd.testing.assert_series_equal(drawdown_series(r), _reference_drawdown(r))


def test_summary_stats_and_calmar_inherit_the_true_drawdown(opens_losing):
    stats = summary_stats(opens_losing.to_frame("p"), PPY, riskfree_rate=0.0, extended=True)
    assert float(stats.loc["p", "Max Drawdown"]) == pytest.approx(-0.19)
    cagr = float((1.0 + opens_losing).prod() ** (PPY / len(opens_losing)) - 1.0)
    assert float(calmar_ratio(opens_losing, PPY)) == pytest.approx(cagr / 0.19)
    assert float(stats.loc["p", "Calmar Ratio"]) == pytest.approx(cagr / 0.19)


def test_the_drawdown_frame_starts_its_peak_at_the_starting_wealth(opens_losing):
    frame = drawdown(opens_losing, starting_wealth=1000.0)
    assert float(frame["Peaks"].iloc[0]) == pytest.approx(1000.0)
    assert float(frame["Drawdown"].iloc[0]) == pytest.approx(-0.10)
    implied = frame["Wealth"] / frame["Peaks"] - 1.0
    pd.testing.assert_series_equal(frame["Drawdown"], implied, check_names=False)


def test_ulcer_index_and_time_under_water_see_the_opening_loss(opens_losing):
    reference = _reference_drawdown(opens_losing)
    assert ulcer_index(opens_losing) == pytest.approx(float(np.sqrt((reference**2).mean())))
    assert time_under_water(opens_losing) == pytest.approx(1.0)


def test_drawdown_table_dates_an_episode_from_inception(opens_losing):
    table = drawdown_table(opens_losing)
    assert len(table) == 1
    episode = table.iloc[0]
    assert episode["max_drawdown"] == pytest.approx(-0.19)
    # The peak is the starting capital, which has no date in the index.
    assert pd.isna(episode["peak_date"])
    assert episode["peak_wealth"] == pytest.approx(1.0)
    assert episode["trough_date"] == opens_losing.index[1]
    assert episode["decline_periods"] == 2
    assert episode["total_periods"] == 4
    assert not episode["recovered"]
    assert max_drawdown_duration(opens_losing) == 4


def test_drawdown_table_still_dates_a_peak_inside_the_sample():
    r = pd.Series([0.10, -0.20, 0.30, 0.0], index=_days(4))
    episode = drawdown_table(r).iloc[0]
    assert episode["peak_date"] == r.index[0]
    assert episode["recovery_date"] == r.index[2]
    assert episode["decline_periods"] == 1
    assert episode["total_periods"] == 2


def test_relative_drawdown_counts_falling_behind_on_the_first_period():
    p = pd.Series([-0.10, 0.0, 0.0], index=_days(3))
    b = pd.Series([0.0, 0.0, 0.0], index=_days(3))
    assert float(relative_drawdown(p, b).min().iloc[0]) == pytest.approx(-0.10)


def test_the_sweep_column_reports_the_true_drawdown(opens_losing):
    base = EngineConfig(optimizer=OptimizerSpec(name="equal_weight"))
    results = run_sweep(
        base,
        SweepSpec(params={"optimizer.name": ["equal_weight"]}),
        lambda cfg: opens_losing,
        periods_per_year=PPY,
    )
    assert float(results.frame.loc[0, "max_drawdown"]) == pytest.approx(-0.19)


def test_the_underwater_charts_start_from_the_starting_capital(opens_losing):
    pytest.importorskip("plotly")
    from optimization_engine.reporting.plots import plot_drawdown, plot_relative_wealth

    underwater = plot_drawdown(opens_losing)
    assert min(underwater.data[0].y) == pytest.approx(-0.19)

    flat = pd.Series(0.0, index=opens_losing.index)
    relative = plot_relative_wealth(opens_losing, flat)
    high_water_mark = relative.data[0].y
    assert high_water_mark[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. A deflation that could not run says why, instead of "no trial count"
# ---------------------------------------------------------------------------
#
# build_tearsheet caught every exception from the deflation and set it to
# None, and the caveat then read "No trial count was supplied" — for a run
# whose caller had supplied forty.


@pytest.fixture()
def oos_run():
    from optimization_engine.backtest.runner import run_backtest
    from optimization_engine.backtest.spec import BacktestSpec

    rng = np.random.default_rng(7)
    assets = pd.DataFrame(
        rng.normal(0.0004, 0.01, (300, 2)), index=_days(300), columns=["A", "B"]
    )
    run = run_backtest(
        assets, pd.Series({"A": 0.5, "B": 0.5}), BacktestSpec(is_out_of_sample=True)
    )
    return run, assets


def _trial_caveats(sheet) -> list[str]:
    return [c for c in sheet.caveats if "trial" in c.lower()]


@pytest.mark.parametrize(
    "trials",
    [pd.Series([0.8]), pd.Series([np.nan, np.nan, np.nan])],
    ids=["one survivor", "no usable trial"],
)
def test_too_few_trial_sharpes_fall_back_to_the_sampling_variance(oos_run, trials):
    from optimization_engine.analytics.selection import deflated_sharpe_ratio
    from optimization_engine.backtest.tearsheet import build_tearsheet

    run, assets = oos_run
    sheet = build_tearsheet(run, assets, n_trials=40, trial_sharpes=trials)

    assert sheet.deflated_sharpe is not None
    assert sheet.deflated_sharpe.n_trials == 40
    expected = deflated_sharpe_ratio(
        run.returns, n_trials=40, periods_per_year=run.periods_per_year
    )
    assert sheet.deflated_sharpe.deflated == pytest.approx(expected.deflated)

    caveats = _trial_caveats(sheet)
    assert not any("No trial count was supplied" in c for c in caveats)
    assert any("sampling variance" in c and "40" in c for c in caveats)
    assert sheet.metadata["deflation_note"]


def test_a_deflation_that_cannot_run_says_why(oos_run, monkeypatch):
    import optimization_engine.analytics.selection as selection
    from optimization_engine.backtest.tearsheet import build_tearsheet

    def refuse(*args, **kwargs):
        raise ValueError("Need at least 3 observations; got 2.")

    monkeypatch.setattr(selection, "deflated_sharpe_ratio", refuse)
    run, assets = oos_run
    sheet = build_tearsheet(run, assets, n_trials=40)

    assert sheet.deflated_sharpe is None
    caveats = _trial_caveats(sheet)
    assert not any("No trial count was supplied" in c for c in caveats)
    assert any(
        "could not be deflated" in c and "Need at least 3 observations" in c and "40" in c
        for c in caveats
    )
    assert "Need at least 3 observations" in sheet.metadata["deflation_note"]


def test_an_unexpected_failure_in_the_deflation_is_not_swallowed(oos_run, monkeypatch):
    import optimization_engine.analytics.selection as selection
    from optimization_engine.backtest.tearsheet import build_tearsheet

    def crash(*args, **kwargs):
        raise RuntimeError("a bug, not an input the deflation cannot use")

    monkeypatch.setattr(selection, "deflated_sharpe_ratio", crash)
    run, assets = oos_run
    with pytest.raises(RuntimeError, match="a bug"):
        build_tearsheet(run, assets, n_trials=40)


def test_no_trial_count_still_says_so(oos_run):
    from optimization_engine.backtest.tearsheet import build_tearsheet

    run, assets = oos_run
    sheet = build_tearsheet(run, assets)
    assert sheet.deflated_sharpe is None
    assert any("No trial count was supplied" in c for c in sheet.caveats)
    assert sheet.metadata["deflation_note"] is None


# ---------------------------------------------------------------------------
# 3. A constant stream has no Sharpe ratio, not one of 1e17
# ---------------------------------------------------------------------------
#
# The zero-variance guards compared the standard deviation with exactly 0. A
# constant 1bp stream has a standard deviation of ~1e-20 — rounding, not
# dispersion — so it scored a Sharpe near 1e17. In a sweep that one cell set
# the deflation benchmark to ~8e16 and every other cell's DSR to 0, and it
# collapsed the PBO of a skill-free grid from 0.314 to 0.000.


def _constant(value: float, n: int = 300) -> pd.Series:
    return pd.Series(value, index=_days(n))


def test_a_constant_stream_has_no_sharpe_ratio():
    from optimization_engine.analytics.performance import (
        probabilistic_sharpe_ratio,
        sharpe_ratio,
    )
    from optimization_engine.analytics.selection import (
        deflated_sharpe_ratio,
        minimum_track_record_length,
    )

    flat = _constant(0.0002)
    assert float(flat.std()) > 0, "the premise: rounding leaves a nonzero std"
    assert np.isnan(sharpe_ratio(flat))
    assert np.isnan(sharpe_ratio(flat, method="geometric"))
    assert np.isnan(probabilistic_sharpe_ratio(flat))
    deflated = deflated_sharpe_ratio(flat, n_trials=10)
    assert np.isnan(deflated.sharpe)
    assert np.isnan(deflated.deflated)
    assert not deflated.is_significant
    assert np.isnan(minimum_track_record_length(flat))


def test_the_frame_path_masks_only_the_constant_column():
    from optimization_engine.analytics.performance import sharpe_ratio

    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {"noisy": rng.normal(0.0004, 0.01, 300), "flat": 0.0001}, index=_days(300)
    )
    sharpes = sharpe_ratio(frame)
    assert np.isnan(sharpes["flat"])
    assert sharpes["noisy"] == pytest.approx(sharpe_ratio(frame["noisy"]))


def test_a_smooth_but_real_stream_keeps_its_sharpe():
    from optimization_engine.analytics.performance import sharpe_ratio

    rng = np.random.default_rng(4)
    smooth = pd.Series(1e-4 + rng.normal(0.0, 1e-7, 300), index=_days(300))
    assert np.isfinite(sharpe_ratio(smooth))
    assert sharpe_ratio(smooth) > 1000


def _sweep_with_a_constant_cell():
    rng = np.random.default_rng(7)
    idx = _days(800)
    streams = {
        "equal_weight": pd.Series(rng.normal(0.0004, 0.01, 800), index=idx),
        "inverse_volatility": pd.Series(rng.normal(0.0003, 0.01, 800), index=idx),
        "min_variance": pd.Series(0.0001, index=idx),
    }

    def evaluate(cfg):
        name = cfg.optimizer.name
        if name == "risk_parity":
            raise RuntimeError("solver blew up")
        return streams[name]

    sweep = SweepSpec(
        params={
            "optimizer.name": [
                "equal_weight", "inverse_volatility", "min_variance", "risk_parity",
            ]
        }
    )
    base = EngineConfig(optimizer=OptimizerSpec(name="equal_weight"))
    return run_sweep(base, sweep, evaluate, periods_per_year=PPY), streams


def test_a_constant_sweep_cell_no_longer_sets_the_deflation_benchmark():
    from optimization_engine.analytics.performance import sharpe_ratio
    from optimization_engine.analytics.selection import deflated_sharpe_ratio

    results, streams = _sweep_with_a_constant_cell()
    trials = results.trial_sharpes()
    assert np.isnan(trials["2"])
    assert np.isnan(results.frame.loc[2, "sharpe"])

    with pytest.warns(UserWarning, match="not finite"):
        deflated = results.deflated_sharpe(0)
    assert deflated.n_trials == 4  # the constant and the failed cell still count
    assert deflated.benchmark_sharpe < 10
    assert deflated.deflated > 0.0

    survivors = pd.Series(
        [sharpe_ratio(streams[k]) for k in ("equal_weight", "inverse_volatility")]
    )
    expected = deflated_sharpe_ratio(
        streams["equal_weight"], n_trials=4, trial_sharpes=survivors
    )
    assert deflated.deflated == pytest.approx(expected.deflated)


def test_an_absurd_trial_sharpe_is_left_out_of_the_dispersion():
    from optimization_engine.analytics.selection import deflated_sharpe_ratio

    rng = np.random.default_rng(5)
    x = pd.Series(rng.normal(0.0008, 0.01, 1000))
    clean = deflated_sharpe_ratio(x, n_trials=3, trial_sharpes=[0.5, 1.0])
    with pytest.warns(UserWarning, match="1 of the 3 trial Sharpes"):
        dirty = deflated_sharpe_ratio(x, n_trials=3, trial_sharpes=[0.5, 1.0, 1.17e17])
    assert dirty.deflated == pytest.approx(clean.deflated)


def test_a_constant_candidate_no_longer_decides_the_pbo():
    from optimization_engine.analytics.selection import (
        probability_of_backtest_overfitting,
    )

    rng = np.random.default_rng(7)
    idx = _days(800)
    noise = pd.DataFrame(rng.normal(0.0, 0.01, (800, 8)), index=idx)
    noise.columns = [f"n{i}" for i in range(8)]
    with_constant = noise.assign(const=0.0001)
    with_zeros = noise.assign(const=0.0)

    pbo = probability_of_backtest_overfitting(with_constant, 8).pbo
    # Scored like the exactly-zero-variance candidate it is, not as a winner.
    assert pbo == pytest.approx(probability_of_backtest_overfitting(with_zeros, 8).pbo)
    assert pbo > 0.0


# ---------------------------------------------------------------------------
# 4. Active risk counts the names only the benchmark holds
# ---------------------------------------------------------------------------
#
# active_risk_decomposition kept only the assets in ``weights``. A book that
# lists its holdings and omits what it does not hold lost every
# benchmark-only name: {A: .5, B: .5} against equal-weight A/B/C reported a
# 4.71% tracking error against a true 8.16%.


def test_active_risk_includes_benchmark_only_names():
    from optimization_engine.analytics.active import active_risk_decomposition

    cov = pd.DataFrame(np.diag([0.04] * 3), index=list("ABC"), columns=list("ABC"))
    book = pd.Series({"A": 0.5, "B": 0.5})
    bench = pd.Series({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
    active = np.array([0.5 - 1 / 3, 0.5 - 1 / 3, -1 / 3])
    true_te = float(np.sqrt(active @ cov.to_numpy() @ active))

    decomposition = active_risk_decomposition(book, bench, cov)
    assert list(decomposition.index) == list("ABC")
    assert float(decomposition["contribution"].sum()) == pytest.approx(true_te)
    assert decomposition.loc["C", "weight"] == 0.0
    assert decomposition.loc["C", "active_weight"] == pytest.approx(-1 / 3)

    explicit = active_risk_decomposition(book.reindex(list("ABC")).fillna(0.0), bench, cov)
    pd.testing.assert_frame_equal(decomposition, explicit)


# ---------------------------------------------------------------------------
# 5. Effective N stays between 1 and the number of assets
# ---------------------------------------------------------------------------
#
# 1/Σw² ignores gross exposure, so [1.5, -0.5] scored 0.4 positions; and a
# long-only pair at ρ = -0.9 has risk contributions {1.27, -0.27}, which put
# effective_n_risk at 0.598. Both docstrings promised "between 1 and N".


def test_effective_n_normalizes_by_gross_exposure():
    from optimization_engine.optimizers.diagnostics import effective_n

    assert effective_n(pd.Series([0.25] * 4)) == pytest.approx(4.0)
    assert effective_n(pd.Series([1.5, -0.5])) == pytest.approx(4.0 / 2.5)
    lever = pd.Series([0.65, 0.65, 0.65, -0.95])
    expected = lever.abs().sum() ** 2 / (lever**2).sum()
    assert effective_n(lever) == pytest.approx(expected)
    assert 1.0 <= effective_n(lever) <= 4.0
    # A book half in cash outside the frame is still two equal positions.
    assert effective_n(pd.Series([0.25, 0.25])) == pytest.approx(2.0)


def test_effective_n_risk_stays_in_range_with_a_hedge():
    from optimization_engine.optimizers.diagnostics import (
        effective_n_risk,
        portfolio_diagnostics,
        risk_contributions,
    )

    vols = np.array([0.20, 0.05])
    corr = np.array([[1.0, -0.9], [-0.9, 1.0]])
    cov = pd.DataFrame(np.outer(vols, vols) * corr, index=["EQ", "BOND"], columns=["EQ", "BOND"])
    book = pd.Series([0.5, 0.5], index=["EQ", "BOND"])
    rc = risk_contributions(book, cov)
    assert float(rc["BOND"]) < 0, "the premise: a negative risk contribution"

    shares = rc.abs() / rc.abs().sum()
    expected = 1.0 / float((shares**2).sum())
    assert effective_n_risk(book, cov) == pytest.approx(expected)
    assert 1.0 <= effective_n_risk(book, cov) <= 2.0
    assert portfolio_diagnostics(book, cov).effective_n_risk == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 6. Relative metrics pair each column with the benchmark on its own dates
# ---------------------------------------------------------------------------
#
# The frame was aligned on the dates where *any* column had data. A column
# that started later then met NaN: beta ran OLS over them and returned NaN
# without a word, and up-capture compounded the column over its own dates but
# the benchmark over all of them (1.0872 against 1.0841 on matched dates).


@pytest.fixture()
def staggered():
    rng = np.random.default_rng(7)
    idx = _days(500)
    bench = pd.Series(rng.normal(0.0004, 0.01, 500), index=idx, name="bench")
    early = bench * 0.9 + rng.normal(0, 0.002, 500)
    late = bench * 1.1 + rng.normal(0, 0.002, 500)
    late.iloc[:250] = np.nan
    frame = pd.DataFrame({"fitted": early, "walk_forward": late})
    return frame, bench


@pytest.mark.parametrize(
    "name",
    [
        "beta", "up_capture", "down_capture", "capture_ratio",
        "information_ratio", "tracking_error", "m_squared", "treynor_ratio",
    ],
)
def test_a_late_column_is_measured_on_its_own_dates(staggered, name):
    from optimization_engine.analytics import relative

    pytest.importorskip("statsmodels")
    frame, bench = staggered
    metric = getattr(relative, name)
    together = metric(frame, bench)
    for column in frame.columns:
        alone = metric(frame[column].dropna().to_frame(column), bench)
        assert np.isfinite(together[column])
        assert together[column] == pytest.approx(float(alone[column]), rel=1e-12)


def test_up_capture_matches_the_hand_computation_on_shared_dates(staggered):
    from optimization_engine.analytics.relative import up_capture

    frame, bench = staggered
    late = frame["walk_forward"].dropna()
    rising = bench.loc[late.index] > 0
    own, ref = late[rising], bench.loc[late.index][rising]
    expected = ((1 + own).prod() ** (1 / len(own)) - 1) / (
        (1 + ref).prod() ** (1 / len(ref)) - 1
    )
    assert up_capture(frame, bench)["walk_forward"] == pytest.approx(expected)


def test_conditional_beta_and_relative_drawdown_pair_each_column(staggered):
    from optimization_engine.analytics.relative import conditional_beta, relative_drawdown

    pytest.importorskip("statsmodels")
    frame, bench = staggered
    late = frame["walk_forward"].dropna().to_frame("walk_forward")
    pd.testing.assert_series_equal(
        conditional_beta(frame, bench).loc["walk_forward"],
        conditional_beta(late, bench).loc["walk_forward"],
    )
    together = relative_drawdown(frame, bench)["walk_forward"].dropna()
    alone = relative_drawdown(late, bench)["walk_forward"]
    pd.testing.assert_series_equal(together, alone)


# ---------------------------------------------------------------------------
# 7. The trial count is never below the number of trial Sharpes supplied
# ---------------------------------------------------------------------------
#
# Fifty trial Sharpes with n_trials=1 deflated against one trial: DSR = PSR
# = 0.999, where the fifty the caller had actually run give 0.793.


def test_n_trials_is_raised_to_the_trials_supplied():
    from optimization_engine.analytics.selection import deflated_sharpe_ratio

    rng = np.random.default_rng(7)
    x = pd.Series(rng.normal(0.0008, 0.01, 1000))
    trials = pd.Series(rng.normal(0.0, 0.5, 50))
    honest = deflated_sharpe_ratio(x, n_trials=50, trial_sharpes=trials)
    with pytest.warns(UserWarning, match="50 trial Sharpes"):
        undercounted = deflated_sharpe_ratio(x, n_trials=1, trial_sharpes=trials)
    assert undercounted.n_trials == 50
    assert undercounted.deflated == pytest.approx(honest.deflated)
    assert undercounted.deflated < undercounted.probabilistic


# ---------------------------------------------------------------------------
# 8. No observations is no return, not a 0% return
# ---------------------------------------------------------------------------


def test_annualize_returns_is_nan_without_observations():
    from optimization_engine.analytics.performance import annualize_returns

    assert np.isnan(annualize_returns(pd.Series(np.nan, index=_days(300))))
    assert np.isnan(annualize_returns(pd.Series([], dtype=float)))

    rng = np.random.default_rng(8)
    frame = pd.DataFrame(
        {"live": rng.normal(0.0004, 0.01, 300), "empty": np.nan}, index=_days(300)
    )
    cagr = annualize_returns(frame)
    assert np.isnan(cagr["empty"])
    assert cagr["live"] == pytest.approx(annualize_returns(frame["live"]))

    # The aggregation summary_stats builds its "Annualized Return" from.
    aggregated = frame.aggregate(annualize_returns, periods_per_year=PPY)
    assert np.isnan(aggregated["empty"])


# ---------------------------------------------------------------------------
# 9. The base configuration is a trial too, when the grid does not contain it
# ---------------------------------------------------------------------------
#
# The CLI deflates the headline run — the base config — against the sweep.
# With base risk_parity and --sweep optimizer.name=min_variance,hrp it used
# N = 2 against a dispersion that left the headline out; three were tried.


def _grid_around(base_name: str, names: list[str]):
    rng = np.random.default_rng(9)
    idx = _days(600)
    streams = {
        name: pd.Series(rng.normal(0.0003, 0.01, 600), index=idx, name=name)
        for name in ("risk_parity", "min_variance", "hrp", "equal_weight")
    }
    base = EngineConfig(optimizer=OptimizerSpec(name=base_name))
    results = run_sweep(
        base,
        SweepSpec(params={"optimizer.name": names}),
        lambda cfg: streams[cfg.optimizer.name],
        periods_per_year=PPY,
    )
    return results, streams


def test_a_base_outside_the_grid_is_counted_and_measured():
    from optimization_engine.analytics.performance import sharpe_ratio

    results, streams = _grid_around("risk_parity", ["min_variance", "hrp"])
    assert not results.base_is_cell()

    n_trials, sharpes = results.trials_with_base(streams["risk_parity"])
    assert n_trials == 3
    assert list(sharpes.index) == ["0", "1", "base"]
    assert sharpes["base"] == pytest.approx(
        sharpe_ratio(streams["risk_parity"], 0.0, PPY)
    )
    pd.testing.assert_series_equal(sharpes.drop("base"), results.trial_sharpes())


def test_a_base_inside_the_grid_is_not_counted_twice():
    results, streams = _grid_around("risk_parity", ["min_variance", "risk_parity"])
    assert results.base_is_cell()

    n_trials, sharpes = results.trials_with_base(streams["risk_parity"])
    assert n_trials == results.n_cells == 2
    pd.testing.assert_series_equal(sharpes, results.trial_sharpes())


def test_the_cli_deflates_the_headline_run_against_every_trial(tmp_path, capsys):
    import yaml

    from optimization_engine.cli import main

    config = yaml.safe_load((ROOT / "config" / "example_multi_asset.yaml").read_text())
    assert config["optimizer"]["name"] not in ("min_variance", "equal_weight")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    argv = [
        "backtest", "--config", str(path), "--sample",
        "--lookback", "252", "--rebalance-every", "252",
        "--commission-bps", "10",
        "--sweep", "optimizer.name=min_variance,equal_weight",
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "2 cells" in out
    assert "Across 3 trial(s)" in out


# ---------------------------------------------------------------------------
# 10. The docstrings name the units the code computes in
# ---------------------------------------------------------------------------
#
# Review sections 2.1 and 2.2: every ratio path converts an *annual*
# risk-free rate, turnover is Σ|Δw| (two-sided), and the one-number cost is
# charged per side. Docstrings said "per-period", "one-way" and "round-trip".


@pytest.mark.parametrize(
    "phrase",
    [
        "riskfree_rate: Per-period risk-free rate",
        "risk_free_rate: Per-period risk-free rate",
        "turnover: One-way turnover",
        "One-way turnover on each",
        "Sum of one-way turnover",
        "total_turnover: One-way traded notional",
        "Round-trip cost",
        "the number a trading desk actually budgets",
    ],
)
def test_no_docstring_misstates_a_unit(phrase):
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in sorted((SRC / "optimization_engine").rglob("*.py"))
        if phrase in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"{phrase!r} still appears in {offenders}"


# ---------------------------------------------------------------------------
# 11. An infinite condition number, and a matrix that needed repairing, say so
# ---------------------------------------------------------------------------
#
# The payload turned an infinite condition number into null — the value it
# uses for "not computed". And run_engine diagnosed the covariance *after*
# nearest_psd had repaired it, so "not positive semi-definite, repaired"
# could never be reported.


def _covariance_diagnostics(condition: float):
    from optimization_engine.data.covariance import CovarianceDiagnostics

    return CovarianceDiagnostics(
        n_assets=3, n_observations=2, observations_per_asset=2 / 3,
        condition_number=condition, min_eigenvalue=0.0, is_psd=True,
        effective_observations=2.0,
    )


def test_an_infinite_condition_number_is_distinguishable_from_missing():
    import json

    from optimization_engine.reporting.payloads import (
        SCHEMA_VERSION,
        covariance_diagnostics_payload,
    )

    singular = covariance_diagnostics_payload(_covariance_diagnostics(float("inf")))
    assert singular["condition_number"] is None
    assert singular["condition_number_infinite"] is True
    json.loads(json.dumps(singular, allow_nan=False))

    finite = covariance_diagnostics_payload(_covariance_diagnostics(250.0))
    assert finite["condition_number"] == 250.0
    assert finite["condition_number_infinite"] is False

    major, minor = SCHEMA_VERSION.split(".")[:2]
    assert major == "2" and int(minor) >= 3, "a new key is a minor bump"


def test_run_engine_diagnoses_the_covariance_before_repairing_it(monkeypatch):
    import optimization_engine.data.covariance as covariance
    from optimization_engine.data.loader import prices_to_returns, sample_dataset
    from optimization_engine.engine import run_engine

    def indefinite(returns, ddof=1):
        cov = returns.cov(ddof=ddof)
        eigenvalues = np.linalg.eigvalsh(cov.to_numpy())
        shift = eigenvalues.min() + 0.1 * eigenvalues.max()
        return cov - np.eye(len(cov)) * shift

    monkeypatch.setattr(covariance, "_sample", indefinite)
    returns = prices_to_returns(sample_dataset(n_periods=400, seed=3))
    config = EngineConfig(
        covariance_method="sample", optimizer=OptimizerSpec(name="equal_weight")
    )
    run = run_engine(returns, config)

    diagnostics = run.covariance_diagnostics
    assert diagnostics.min_eigenvalue < 0
    assert not diagnostics.is_psd
    assert any("not positive semi-definite" in w for w in diagnostics.warnings)
    # The matrix the run carries, and solved against, is the repaired one.
    assert np.linalg.eigvalsh(run.cov_matrix.to_numpy()).min() > -1e-10


# ---------------------------------------------------------------------------
# 12. A workbook that leaves a sheet out says which, and why
# ---------------------------------------------------------------------------
#
# run_sheets caught the relative-performance failure and dropped the
# performance sheets, and dropped the out-of-sample ones with a bare
# ``except (ValueError, KeyError): pass``. The workbook looked complete.


@pytest.fixture()
def benchmarked_run():
    from optimization_engine.benchmark import BenchmarkSpec
    from optimization_engine.data.loader import prices_to_returns, sample_dataset
    from optimization_engine.engine import run_engine

    returns = prices_to_returns(sample_dataset(n_periods=600, seed=11))
    config = EngineConfig(
        optimizer=OptimizerSpec(name="equal_weight"),
        benchmark=BenchmarkSpec(kind="equal_weight"),
    )
    return run_engine(returns, config)


def test_a_dropped_performance_report_is_recorded(benchmarked_run, monkeypatch):
    from optimization_engine.reporting.exporters import run_sheets

    def no_overlap(*args, **kwargs):
        raise ValueError("The portfolio and the benchmark share no dates")

    monkeypatch.setattr(benchmarked_run, "performance", no_overlap)
    with pytest.warns(UserWarning, match="share no dates"):
        sheets = run_sheets(benchmarked_run)
    notes = sheets["omitted_sheets"]
    assert "performance" in " ".join(notes["sheets"])
    assert notes["reason"].str.contains("share no dates").any()


def test_a_dropped_out_of_sample_report_is_recorded(benchmarked_run, monkeypatch):
    from optimization_engine.reporting.exporters import run_sheets

    walk = benchmarked_run.walk_forward(lookback=252, rebalance_every=126)
    performance = benchmarked_run.performance()
    real = benchmarked_run.performance

    def refuse_the_override(*args, **kwargs):
        if kwargs.get("returns_override") is not None:
            raise KeyError("walk-forward dates missing from the benchmark")
        return real(*args, **kwargs)

    monkeypatch.setattr(benchmarked_run, "performance", refuse_the_override)
    with pytest.warns(UserWarning, match="walk-forward dates missing"):
        sheets = run_sheets(benchmarked_run, walk_forward=walk, performance=performance)
    assert not any(name.startswith("oos_") for name in sheets)
    notes = sheets["omitted_sheets"]
    assert "oos_" in " ".join(notes["sheets"])
    assert notes["reason"].str.contains("walk-forward dates missing").any()


def test_a_complete_workbook_carries_no_omission_sheet(benchmarked_run):
    from optimization_engine.reporting.exporters import run_sheets

    assert "omitted_sheets" not in run_sheets(benchmarked_run)
