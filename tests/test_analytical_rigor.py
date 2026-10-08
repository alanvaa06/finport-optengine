"""Regression tests for the analytical-rigor fixes.

Each test here pins down a specific way the engine used to be wrong, so a
future refactor cannot quietly reintroduce it.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.analytics.risk import (
    downside_deviation,
    drawdown_table,
    max_drawdown_duration,
    omega_ratio,
    tail_ratio,
    ulcer_index,
)
from optimization_engine.config import EngineConfig, OptimizerSpec
from optimization_engine.data.covariance import (
    _semi,
    covariance_diagnostics,
    covariance_matrix,
    james_stein_shrinkage,
    nearest_psd,
)
from optimization_engine.data.loader import prices_to_returns, sample_dataset
from optimization_engine.engine import run_engine
from optimization_engine.optimizers.base import PortfolioConstraints
from optimization_engine.optimizers.diagnostics import (
    check_constraints,
    diversification_ratio,
    effective_n,
    herfindahl_index,
    risk_decomposition,
)
from optimization_engine.optimizers.feasibility import analyze_feasibility
from optimization_engine.optimizers.max_diversification import MaxDiversificationOptimizer


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=252 * 5, seed=7))


@pytest.fixture(scope="module")
def cov(returns: pd.DataFrame) -> pd.DataFrame:
    return covariance_matrix(returns, method="ledoit_wolf")


@pytest.fixture(scope="module")
def mu(returns: pd.DataFrame) -> pd.Series:
    return (1 + returns).prod() ** (252 / len(returns)) - 1


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------


def test_semicovariance_does_not_recentre_deviations(returns):
    """Semicovariance is D'D/(T-1), not the covariance *of* D.

    Using DataFrame.cov() subtracts D's own (negative) mean, understating
    downside co-movement.
    """
    deviation = (returns - returns.mean()).clip(upper=0.0)
    expected = deviation.T.values @ deviation.values / (len(deviation) - 1)
    got = _semi(returns).values
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-14)
    # And it genuinely differs from the re-centred version.
    recentred = deviation.cov(ddof=1).values
    assert np.abs(got - recentred).max() > 1e-8


def test_semicovariance_respects_explicit_mar(returns):
    zero_mar = _semi(returns, mar=0.0)
    mean_mar = _semi(returns, mar=None)
    assert not np.allclose(zero_mar.values, mean_mar.values)


def test_covariance_output_is_psd(returns):
    for method in ("sample", "ledoit_wolf", "oas", "ewma", "semi"):
        c = covariance_matrix(returns, method=method)
        eig = np.linalg.eigvalsh(c.values)
        assert eig.min() >= -1e-12, f"{method} produced a non-PSD matrix"


def test_nearest_psd_repairs_indefinite_matrix():
    bad = pd.DataFrame([[1.0, 0.99], [0.99, 0.5]], index=["a", "b"], columns=["a", "b"])
    assert np.linalg.eigvalsh(bad.values).min() < 0
    fixed = nearest_psd(bad)
    assert np.linalg.eigvalsh(fixed.values).min() >= -1e-12
    assert list(fixed.columns) == ["a", "b"]


def test_covariance_rejects_missing_values(returns):
    dirty = returns.copy()
    dirty.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        covariance_matrix(dirty)


def test_covariance_diagnostics_flag_thin_samples(returns):
    short = returns.iloc[:20]
    diag = covariance_diagnostics(
        covariance_matrix(short, method="ledoit_wolf"), n_observations=len(short)
    )
    assert not diag.is_reliable
    assert any("observations" in w for w in diag.warnings)


def test_james_stein_pulls_means_toward_the_centre(returns, cov, mu):
    shrunk, intensity = james_stein_shrinkage(mu, cov, len(returns))
    assert 0.0 <= intensity <= 1.0
    if intensity > 0:
        assert shrunk.std() < mu.std()


# ---------------------------------------------------------------------------
# Downside risk
# ---------------------------------------------------------------------------


def test_downside_deviation_averages_over_all_periods():
    r = pd.Series([0.02, -0.01, 0.03, -0.04, 0.01])
    expected = float(np.sqrt((np.minimum(r, 0.0) ** 2).mean()))
    assert downside_deviation(r) == pytest.approx(expected)
    # The old shortcut — std of the negative subset — is materially smaller.
    assert float(r[r < 0].std(ddof=0)) < expected


def test_downside_deviation_uses_the_mar():
    r = pd.Series([0.02, 0.01, 0.03, 0.015])
    assert downside_deviation(r, mar=0.0) == pytest.approx(0.0)
    assert downside_deviation(r, mar=0.05) > 0.0


def test_drawdown_table_reports_timing_and_recovery():
    idx = pd.date_range("2020-01-01", periods=8, freq="D")
    r = pd.Series([0.10, -0.20, -0.10, 0.05, 0.40, 0.01, -0.05, 0.02], index=idx)
    table = drawdown_table(r)
    assert not table.empty
    worst = table.iloc[0]
    assert worst["max_drawdown"] < 0
    assert worst["decline_periods"] >= 1
    assert max_drawdown_duration(r) >= worst["total_periods"]


def test_ulcer_index_separates_equal_max_drawdowns():
    """Same depth, different time underwater — Ulcer must tell them apart."""
    idx = pd.date_range("2020-01-01", periods=11, freq="D")
    # A peak first, so the drop is measured from somewhere.
    quick = pd.Series([0.05, -0.10, 0.12] + [0.0] * 8, index=idx)
    slow = pd.Series([0.05, -0.10] + [0.0] * 8 + [0.12], index=idx)
    assert ulcer_index(slow) > ulcer_index(quick)


def test_omega_and_tail_ratio_are_finite(returns):
    assert np.isfinite(omega_ratio(returns.iloc[:, 0]))
    assert np.isfinite(tail_ratio(returns.iloc[:, 0]))


# ---------------------------------------------------------------------------
# Optimizer correctness
# ---------------------------------------------------------------------------


def test_max_diversification_solves_the_bounded_problem(returns, cov):
    """Bounded max-DR must match the exact constrained optimum, not a projection."""
    import cvxpy as cp

    cons = PortfolioConstraints(bounds={a: (0.0, 0.15) for a in cov.columns})
    got = MaxDiversificationOptimizer(cov_matrix=cov, constraints=cons).optimize()

    std = np.sqrt(np.diag(cov.values))
    n = len(std)
    y = cp.Variable(n, nonneg=True)
    k = cp.Variable(nonneg=True)
    cp.Problem(
        cp.Minimize(cp.quad_form(y, cp.psd_wrap(cov.values))),
        [std @ y == 1, cp.sum(y) == k, y <= 0.15 * k, k >= 1e-8],
    ).solve()
    reference = np.asarray(y.value) / float(k.value)
    ref_dr = (reference @ std) / np.sqrt(reference @ cov.values @ reference)

    assert diversification_ratio(got.weights, cov) == pytest.approx(ref_dr, rel=1e-3)
    assert got.is_compliant


def test_max_diversification_honours_group_bounds(returns, cov):
    groups = {a: ("Equity" if "Equity" in a else "Other") for a in cov.columns}
    cons = PortfolioConstraints(
        bounds={a: (0.0, 0.5) for a in cov.columns},
        groups=groups,
        group_bounds={"Equity": (0.0, 0.20)},
    )
    res = MaxDiversificationOptimizer(cov_matrix=cov, constraints=cons).optimize()
    equity = sum(v for a, v in res.weights.items() if groups[a] == "Equity")
    assert equity <= 0.20 + 1e-6
    assert res.is_compliant


def test_long_only_tightens_negative_bounds():
    cons = PortfolioConstraints(bounds={"A": (-0.5, 0.5)}, long_only=True)
    assert cons.get_bounds("A") == (0.0, 0.5)
    cons.long_only = False
    assert cons.get_bounds("A") == (-0.5, 0.5)


def test_clean_weights_never_leaves_the_box(returns, cov, mu):
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.02, 0.20] for a in cov.columns},
        optimizer=OptimizerSpec(name="min_variance"),
    )
    run = run_engine(returns, cfg)
    w = run.result.weights
    assert (w >= 0.02 - 1e-6).all()
    assert (w <= 0.20 + 1e-6).all()
    assert run.result.is_compliant


def test_missing_expected_returns_warn_rather_than_silently_zero(cov):
    from optimization_engine.optimizers.mean_variance import MeanVarianceOptimizer

    partial = pd.Series({cov.columns[0]: 0.08})
    opt = MeanVarianceOptimizer(
        expected_returns=partial, cov_matrix=cov, risk_aversion=2.0
    )
    with pytest.warns(UserWarning, match="no expected return"):
        opt.optimize()


def test_turnover_budget_is_enforced_end_to_end(returns, mu):
    previous = {a: 1.0 / len(mu) for a in mu.index}
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 1.0] for a in mu.index},
        previous_weights=previous,
        turnover_limit=0.10,
        optimizer=OptimizerSpec(name="mean_variance", risk_aversion=1.0),
    )
    run = run_engine(returns, cfg)
    prev = pd.Series(previous).reindex(run.result.weights.index)
    turnover = float((run.result.weights - prev).abs().sum())
    assert turnover <= 0.10 + 1e-4
    assert run.result.is_compliant


def test_unconstrained_turnover_is_larger_than_the_budget(returns, mu):
    """The budget above is binding — otherwise the test proves nothing."""
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 1.0] for a in mu.index},
        optimizer=OptimizerSpec(name="mean_variance", risk_aversion=1.0),
    )
    run = run_engine(returns, cfg)
    prev = pd.Series({a: 1.0 / len(mu) for a in mu.index}).reindex(
        run.result.weights.index
    )
    assert float((run.result.weights - prev).abs().sum()) > 0.10


def test_leverage_and_short_selling_round_trip(returns, mu):
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [-0.3, 0.5] for a in mu.index},
        long_only=False,
        leverage=1.6,
        optimizer=OptimizerSpec(name="mean_variance", risk_aversion=1.0),
    )
    run = run_engine(returns, cfg)
    assert float(run.result.weights.abs().sum()) <= 1.6 + 1e-4
    assert run.result.weights.min() >= -0.3 - 1e-6


def test_risk_parity_reports_budget_error(returns, cov):
    from optimization_engine.optimizers.risk_parity import RiskParityOptimizer

    res = RiskParityOptimizer(
        cov_matrix=cov, constraints=PortfolioConstraints()
    ).optimize()
    assert res.extras["risk_budget_max_error"] < 1e-3


def test_risk_parity_rejects_zero_budget(cov):
    from optimization_engine.optimizers.risk_parity import RiskParityOptimizer

    budget = {a: 1.0 for a in cov.columns}
    budget[cov.columns[0]] = 0.0
    with pytest.raises(ValueError, match="zero"):
        RiskParityOptimizer(cov_matrix=cov, risk_budget=budget).optimize()


def test_cvar_reports_realized_tail_metrics(returns, mu):
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.4] for a in mu.index},
        optimizer=OptimizerSpec(name="cvar", cvar_alpha=0.05),
    )
    run = run_engine(returns, cfg)
    assert run.result.extras["cvar_period"] > 0
    assert run.result.extras["tail_observations"] > 0
    assert run.result.extras["cvar_sqrt_t_scaled"] > run.result.extras["cvar_period"]


def test_cvar_rejects_a_confidence_level_passed_as_alpha(returns):
    from optimization_engine.optimizers.cvar import CVaROptimizer

    with pytest.raises(ValueError, match="tail probability"):
        CVaROptimizer(returns=returns, alpha=0.95)


def test_hrp_warns_instead_of_silently_dropping_group_bounds(returns, cov):
    from optimization_engine.optimizers.hrp import HRPOptimizer

    cons = PortfolioConstraints(
        groups={a: "All" for a in cov.columns}, group_bounds={"All": (1.0, 1.0)}
    )
    with pytest.warns(UserWarning, match="bucket budgets"):
        HRPOptimizer(cov_matrix=cov, constraints=cons).optimize()


def test_max_sharpe_warns_that_turnover_cannot_bind(cov, mu):
    from optimization_engine.optimizers.mean_variance import MaxSharpeOptimizer

    cons = PortfolioConstraints(
        previous_weights={a: 1.0 / len(mu) for a in mu.index}, turnover_limit=0.05
    )
    with pytest.warns(UserWarning, match="turnover"):
        MaxSharpeOptimizer(
            expected_returns=mu, cov_matrix=cov, constraints=cons
        ).optimize()


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------


def test_relative_view_moves_the_spread_toward_the_view(cov):
    from optimization_engine.optimizers.black_litterman import (
        View,
        black_litterman_posterior,
        build_pick_matrix,
    )

    mkt = pd.Series(1.0 / len(cov.columns), index=cov.columns)
    view = View({"US_Equity": 1.0, "EM_Equity": -1.0}, 0.03)
    prior, _ = black_litterman_posterior(cov, mkt, None)
    post, _ = black_litterman_posterior(cov, mkt, [view])
    P, _, _ = build_pick_matrix([view], list(cov.columns))
    prior_spread = float((P @ prior.values)[0])
    post_spread = float((P @ post.values)[0])
    assert prior_spread < post_spread <= 0.03


def test_absolute_view_mapping_still_supported(cov):
    from optimization_engine.optimizers.black_litterman import black_litterman_posterior

    mkt = pd.Series(1.0 / len(cov.columns), index=cov.columns)
    prior, _ = black_litterman_posterior(cov, mkt, None)
    post, _ = black_litterman_posterior(cov, mkt, {"Gold": 0.10})
    assert post["Gold"] > prior["Gold"]


def test_he_litterman_omega_scales_with_the_view_portfolio(cov):
    from optimization_engine.optimizers.black_litterman import View, build_pick_matrix

    view = View({"US_Equity": 1.0, "EM_Equity": -1.0}, 0.03)
    P, _, _ = build_pick_matrix([view], list(cov.columns))
    omega = float(np.diag(P @ (0.05 * cov.values) @ P.T)[0])
    naive = 0.05 * float(cov.loc["US_Equity", "US_Equity"])
    # A spread view's uncertainty is not one leg's variance.
    assert abs(omega - naive) > 1e-8


def test_view_confidence_must_be_positive(cov):
    from optimization_engine.optimizers.black_litterman import View, black_litterman_posterior

    mkt = pd.Series(1.0 / len(cov.columns), index=cov.columns)
    with pytest.raises(ValueError, match="positive"):
        black_litterman_posterior(
            cov, mkt, [View({"Gold": 1.0}, 0.10, confidence=0.0)]
        )


def test_implied_risk_aversion_matches_market_sharpe(cov):
    from optimization_engine.optimizers.black_litterman import implied_risk_aversion

    mkt = pd.Series(1.0 / len(cov.columns), index=cov.columns)
    var = float(mkt.values @ cov.values @ mkt.values)
    delta = implied_risk_aversion(0.08, var, 0.02)
    assert delta == pytest.approx((0.08 - 0.02) / var)


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------


def test_feasibility_catches_caps_below_the_budget(cov, mu):
    cons = PortfolioConstraints(bounds={a: (0.0, 0.05) for a in cov.columns})
    report = analyze_feasibility(list(cov.columns), cons, mu, cov)
    assert not report.is_feasible
    assert any(i.code == "max_weights_below_budget" for i in report.issues)


def test_feasibility_catches_floors_above_the_budget(cov, mu):
    cons = PortfolioConstraints(bounds={a: (0.20, 1.0) for a in cov.columns})
    report = analyze_feasibility(list(cov.columns), cons, mu, cov)
    assert any(i.code == "min_weights_exceed_budget" for i in report.issues)


def test_feasibility_catches_unreachable_group_minimum(cov, mu):
    groups = {a: ("Equity" if "Equity" in a else "Other") for a in cov.columns}
    cons = PortfolioConstraints(
        bounds={a: (0.0, 0.05) for a in cov.columns},
        groups=groups,
        group_bounds={"Equity": (0.60, 1.0)},
    )
    report = analyze_feasibility(list(cov.columns), cons, mu, cov)
    assert any(i.code == "group_min_unreachable" for i in report.issues)


def test_feasibility_bounds_the_return_target(cov, mu):
    cons = PortfolioConstraints(
        bounds={a: (0.0, 0.15) for a in cov.columns}, target_return=5.0
    )
    report = analyze_feasibility(list(cov.columns), cons, mu, cov)
    assert any(i.code == "target_return_too_high" for i in report.issues)
    assert report.max_return is not None and report.max_return < 5.0


def test_feasibility_flags_a_dominated_target_without_blocking(cov, mu):
    """A sub-GMV target is still not fatal, but the advice had to change.

    The message used to promise "solvable but dominated". Since the return
    target became a floor (``μ'w ≥ R*``) that is no longer what happens:
    minimum variance already clears the target, so the minimum-variance
    portfolio is what comes back. Saying "dominated" would now be false.
    """
    cons = PortfolioConstraints(bounds={a: (0.0, 0.5) for a in cov.columns})
    base = analyze_feasibility(list(cov.columns), cons, mu, cov)
    cons.target_return = base.min_variance_return - 0.01
    report = analyze_feasibility(list(cov.columns), cons, mu, cov)
    assert report.is_feasible  # solvable, and it is the minimum-variance book
    issues = [i for i in report.issues if i.code == "target_return_inefficient"]
    assert len(issues) == 1
    issue = issues[0]
    assert not issue.fatal
    advice = f"{issue.message} {issue.suggestion}"
    assert "minimum-variance portfolio" in advice
    assert "dominated" not in advice.lower()


def test_feasibility_reports_are_actionable(cov, mu):
    cons = PortfolioConstraints(bounds={a: (0.0, 0.05) for a in cov.columns})
    report = analyze_feasibility(list(cov.columns), cons, mu, cov)
    for issue in report.issues:
        assert issue.suggestion, f"{issue.code} has no suggested fix"


def test_run_engine_can_raise_on_infeasible(returns, mu):
    from optimization_engine.optimizers.feasibility import InfeasibleConstraintsError

    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.05] for a in mu.index},
        optimizer=OptimizerSpec(name="min_variance"),
    )
    with pytest.raises(InfeasibleConstraintsError, match="cannot be satisfied"):
        run_engine(returns, cfg, raise_on_infeasible=True)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_effective_n_reflects_concentration():
    equal = pd.Series([0.25] * 4)
    concentrated = pd.Series([0.90, 0.05, 0.03, 0.02])
    assert effective_n(equal) == pytest.approx(4.0)
    assert effective_n(concentrated) < 1.5
    assert herfindahl_index(equal) == pytest.approx(0.25)


def test_risk_decomposition_sums_to_portfolio_volatility(cov):
    w = pd.Series(1.0 / len(cov.columns), index=cov.columns)
    table = risk_decomposition(w, cov)
    port_vol = float(np.sqrt(w.values @ cov.values @ w.values))
    assert table["contribution"].sum() == pytest.approx(port_vol, rel=1e-9)
    assert table["share_of_risk"].sum() == pytest.approx(1.0, rel=1e-9)


def test_check_constraints_detects_every_breach_kind():
    w = pd.Series({"A": 0.9, "B": 0.3, "C": -0.1})  # sums to 1.1, not 1
    cons = PortfolioConstraints(
        bounds={"A": (0.0, 0.5), "B": (0.0, 0.5), "C": (0.0, 0.5)},
        groups={"A": "G", "B": "G", "C": "H"},
        group_bounds={"G": (0.0, 0.6)},
        long_only=True,
    )
    kinds = {v.kind for v in check_constraints(w, cons)}
    assert {"bound", "budget", "group"} <= kinds


def test_compliant_solution_reports_no_violations(returns, mu):
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.25] for a in mu.index},
        optimizer=OptimizerSpec(name="max_sharpe", risk_free_rate=0.02),
    )
    run = run_engine(returns, cfg)
    assert run.result.is_compliant, run.result.violations
    assert run.diagnostics is not None
    assert run.diagnostics.effective_n > 1


# ---------------------------------------------------------------------------
# Frontier
# ---------------------------------------------------------------------------


def test_frontier_range_respects_binding_caps(returns, mu):
    """A 15% cap used to make more than half the frontier infeasible."""
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.15] for a in mu.index},
        optimizer=OptimizerSpec(name="mean_variance", risk_free_rate=0.02),
    )
    run = run_engine(returns, cfg, build_frontier=True, n_frontier_points=15)
    assert run.frontier.n_failed == 0, run.frontier.failures
    lo, hi = run.frontier.reachable_range
    assert lo < hi < float(mu.max())


def test_frontier_marks_the_anchor_portfolios(returns, mu):
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.30] for a in mu.index},
        optimizer=OptimizerSpec(name="mean_variance", risk_free_rate=0.02),
    )
    fr = run_engine(returns, cfg, build_frontier=True, n_frontier_points=12).frontier
    assert fr.min_variance is not None and fr.tangency is not None
    # Nothing on the frontier can have a lower volatility than the GMV.
    assert fr.efficient["expected_volatility"].min() >= (
        fr.min_variance["expected_volatility"] - 1e-6
    )
    # And nothing can beat the tangency portfolio's Sharpe ratio.
    assert fr.efficient["sharpe_ratio"].max() <= fr.tangency["sharpe_ratio"] + 1e-6


def test_frontier_cannot_trace_the_dominated_lower_branch(returns, mu):
    """Every swept point is efficient — the lower branch is unreachable now.

    This used to assert that the dominated branch was *filtered out* of
    ``efficient``. Under the mean-variance return floor (``μ'w ≥ R*``) there
    is nothing to filter: a target below the minimum-variance return returns
    the minimum-variance portfolio, so no solved point can land below it. The
    sweep is forced across the whole reachable range with
    ``efficient_only=False`` — the setting that used to produce the dominated
    branch — to show it produces none.
    """
    from optimization_engine.frontier import efficient_frontier

    cov = covariance_matrix(returns, method="ledoit_wolf")
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.30] for a in mu.index},
        optimizer=OptimizerSpec(name="mean_variance"),
    )
    fr = efficient_frontier(
        cfg, cov, expected_returns=mu, n_points=12, efficient_only=False
    )
    gmv_return = float(fr.min_variance["expected_return"])
    solved = fr.summary[fr.summary["status"] == "ok"]
    assert not solved.empty
    # Targets below the minimum-variance return were swept, and every one of
    # them still came back at or above it.
    assert (solved["target"] < gmv_return - 1e-6).any(), "no sub-GMV target swept"
    assert (solved["expected_return"] >= gmv_return - 1e-3).all()
    assert solved["is_efficient"].all()
    assert len(fr.efficient) == len(solved)
    # Volatility rises monotonically along the efficient branch.
    vols = fr.efficient["expected_volatility"].values
    assert (np.diff(vols) >= -1e-6).all()


def test_frontier_reports_anchor_failure(returns, mu):
    """A tangency portfolio that cannot be solved says so, and GMV survives.

    ``_anchor_portfolios`` used to swallow both anchor solves with
    ``except Exception: pass``. A mandate that makes max-Sharpe infeasible then
    produced a chart with no tangency marker and nothing to distinguish that
    from a chart that never asked for one.

    The mandate here: a 5% risk-free rate leaves exactly one asset with a
    positive excess return, and that asset is capped at 2%. No fully invested
    portfolio earns more than cash, so the tangency ray has nowhere to go —
    while minimum variance, which never looks at ``μ``, is untouched.
    """
    from optimization_engine.frontier import efficient_frontier

    cov = covariance_matrix(returns, method="ledoit_wolf")
    risk_free_rate = 0.05
    assert (mu > risk_free_rate).sum() == 1, "fixture no longer sets up the trap"
    bounds = {
        a: ([0.0, 0.02] if mu[a] > risk_free_rate else [0.0, 0.5]) for a in mu.index
    }
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds=bounds,
        optimizer=OptimizerSpec(
            name="mean_variance", risk_free_rate=risk_free_rate
        ),
    )
    fr = efficient_frontier(cfg, cov, expected_returns=mu, n_points=6)

    assert fr.tangency is None
    assert "tangency" in fr.anchor_failures
    assert "infeasible" in fr.anchor_failures["tangency"].lower()
    # The anchor that did solve is still drawn, and is not reported as failed.
    assert fr.min_variance is not None
    assert "min_variance" not in fr.anchor_failures


def test_frontier_with_both_anchors_reports_no_failures(returns, mu):
    """The failure map is empty when nothing failed — no phantom entries."""
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.30] for a in mu.index},
        optimizer=OptimizerSpec(name="mean_variance", risk_free_rate=0.02),
    )
    fr = run_engine(returns, cfg, build_frontier=True, n_frontier_points=6).frontier
    assert fr.min_variance is not None and fr.tangency is not None
    assert fr.anchor_failures == {}


def test_max_sharpe_index_is_nan_safe():
    from optimization_engine.frontier import FrontierResult

    summary = pd.DataFrame(
        {
            "target": [0.1, 0.2],
            "expected_return": [np.nan, np.nan],
            "expected_volatility": [np.nan, np.nan],
            "sharpe_ratio": [np.nan, np.nan],
            "is_efficient": [False, False],
            "status": ["failed: x", "failed: y"],
        }
    )
    fr = FrontierResult(summary=summary, weights=pd.DataFrame())
    assert fr.max_sharpe_index == -1
    assert fr.plot_frame().empty


def test_plot_frame_index_matches_the_highlighted_row(returns, mu):
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.30] for a in mu.index},
        optimizer=OptimizerSpec(name="mean_variance", risk_free_rate=0.02),
    )
    fr = run_engine(returns, cfg, build_frontier=True, n_frontier_points=10).frontier
    frame = fr.plot_frame()
    best = frame["sharpe_ratio"].idxmax()
    assert frame.loc[best, "sharpe_ratio"] == frame["sharpe_ratio"].max()


def test_black_litterman_target_checked_against_its_own_posterior(returns, cov, mu):
    """A BL return target must be judged against the equilibrium posterior.

    Checking it against the configured historical means passes for targets BL
    cannot reach, so the feasibility report says "feasible" and the solve then
    fails with a bare `infeasible`.
    """
    from optimization_engine.optimizers.factory import (
        constraints_from_config,
        effective_expected_returns,
    )

    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 1.0] for a in mu.index},
        optimizer=OptimizerSpec(
            name="black_litterman", target_return=float(mu.max()), risk_free_rate=0.04
        ),
    )
    posterior = effective_expected_returns(cfg, cov, mu)
    # The posterior is a different vector living at a different level.
    assert posterior.max() < mu.max()

    report = analyze_feasibility(
        list(cov.columns), constraints_from_config(cfg), posterior, cov
    )
    assert not report.is_feasible
    issue = next(i for i in report.issues if i.code == "target_return_too_high")
    assert "equilibrium posterior" in issue.suggestion


def test_effective_expected_returns_passes_other_methods_through(cov, mu):
    from optimization_engine.optimizers.factory import effective_expected_returns

    cfg = EngineConfig(
        expected_returns=mu.to_dict(), optimizer=OptimizerSpec(name="mean_variance")
    )
    pd.testing.assert_series_equal(effective_expected_returns(cfg, cov, mu), mu)


def test_run_engine_diagnoses_an_unreachable_black_litterman_target(returns, mu):
    from optimization_engine.optimizers.feasibility import InfeasibleConstraintsError

    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 1.0] for a in mu.index},
        optimizer=OptimizerSpec(
            name="black_litterman", target_return=float(mu.max()), risk_free_rate=0.04
        ),
    )
    with pytest.raises(InfeasibleConstraintsError, match="above the"):
        run_engine(returns, cfg, raise_on_infeasible=True)


def test_wealth_accumulation_is_overflow_proof():
    """A rolling product over a long high-return window overflows float64.

    Compounding in log space keeps the same answer where cumprod works, and
    keeps working where it does not.
    """
    from optimization_engine.analytics.performance import rolling_metrics
    from optimization_engine.analytics.risk import drawdown_series

    idx = pd.date_range("2000-01-01", periods=3000, freq="D")
    normal = pd.Series(np.linspace(-0.01, 0.01, 3000), index=idx)
    reference = (1 + normal).cumprod()
    # The peak includes the starting capital: this series opens with a loss.
    reference_dd = reference / reference.cummax().clip(lower=1.0) - 1
    np.testing.assert_allclose(
        drawdown_series(normal).values, reference_dd.values, atol=1e-12
    )

    extreme = pd.Series(np.full(3000, 0.5), index=idx)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        rolling = rolling_metrics(extreme, 2000)
        dd = drawdown_series(extreme)
    assert np.isfinite(rolling["rolling_return"].iloc[-1])
    assert float(dd.min()) == pytest.approx(0.0)


def test_total_loss_is_reported_not_warned():
    from optimization_engine.analytics.risk import drawdown_series

    s = pd.Series([0.1, -1.0, 0.5], index=pd.date_range("2020-01-01", periods=3))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        dd = drawdown_series(s)
    assert dd.iloc[1] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Solver dispatch
# ---------------------------------------------------------------------------


def test_solver_chain_prefers_an_exact_solution_over_an_inaccurate_one():
    """An `optimal_inaccurate` first answer must not end the fallback chain."""
    import cvxpy as cp

    from optimization_engine.optimizers._cvxpy_helpers import solve_problem

    calls: list[str] = []
    real_solve = cp.Problem.solve

    def fake_solve(self, *args, **kwargs):
        solver = kwargs.get("solver", "default")
        calls.append(str(solver))
        real_solve(self, *args, **kwargs)
        if len(calls) == 1:
            # Pretend the first solver converged only loosely.
            self._status = "optimal_inaccurate"

    w = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(w)), [cp.sum(w) == 1])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cp.Problem, "solve", fake_solve)
        info = solve_problem(problem, solvers=("CLARABEL", "SCS"))
    assert len(calls) == 2, "the chain stopped at the inaccurate answer"
    assert info.status == "optimal"


def _always_inaccurate_problem():
    """A feasible QP plus a patch making every solver report it inaccurate.

    Returns the variable, the problem and the patch function; the caller
    installs the patch so it can choose the scope.
    """
    import cvxpy as cp

    real_solve = cp.Problem.solve

    def always_inaccurate(self, *args, **kwargs):
        real_solve(self, *args, **kwargs)
        self._status = "optimal_inaccurate"

    w = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(w)), [cp.sum(w) == 1])
    return w, problem, always_inaccurate


def test_inaccurate_refused_by_default():
    """Nothing verified the answer, so nothing hands it back as if verified."""
    import cvxpy as cp

    from optimization_engine.optimizers._cvxpy_helpers import SolverFailure, solve_problem

    _w, problem, patch = _always_inaccurate_problem()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cp.Problem, "solve", patch)
        with pytest.raises(SolverFailure) as excinfo:
            solve_problem(problem, solvers=("CLARABEL", "SCS"))

    exc = excinfo.value
    # Not the last solver's verdict -- the one that describes what happened.
    assert exc.status == "optimal_inaccurate"
    assert exc.attempts == ("CLARABEL", "SCS")
    # The message has to name the way out, or the refusal is just a wall.
    assert "accept_inaccurate=True" in str(exc)
    assert "analyze_feasibility" in str(exc)


def test_inaccurate_refused_reports_inaccurate_not_the_last_solvers_verdict():
    """A later solver disagreeing must not relabel the failure.

    A chain that goes inaccurate and then hits a solver claiming the problem
    is infeasible used to raise with ``status="infeasible"`` and the message
    "no allocation satisfies every constraint at once" -- about a problem
    where a solver had already found an allocation.
    """
    import cvxpy as cp

    from optimization_engine.optimizers._cvxpy_helpers import SolverFailure, solve_problem

    real_solve = cp.Problem.solve
    calls: list[str] = []

    def inaccurate_then_infeasible(self, *args, **kwargs):
        calls.append(str(kwargs.get("solver", "default")))
        real_solve(self, *args, **kwargs)
        self._status = "optimal_inaccurate" if len(calls) == 1 else "infeasible"

    w = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(w)), [cp.sum(w) == 1])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cp.Problem, "solve", inaccurate_then_infeasible)
        with pytest.raises(SolverFailure) as excinfo:
            solve_problem(problem, solvers=("CLARABEL", "SCS"))

    assert excinfo.value.status == "optimal_inaccurate"
    assert "no allocation satisfies" not in str(excinfo.value)


def test_inaccurate_accepted_on_opt_in(caplog):
    """The opt-in returns the loose answer, and says out loud that it did."""
    import cvxpy as cp

    from optimization_engine.optimizers._cvxpy_helpers import solve_problem

    w, problem, patch = _always_inaccurate_problem()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cp.Problem, "solve", patch)
        with caplog.at_level(
            logging.WARNING, logger="optimization_engine.optimizers._cvxpy_helpers"
        ):
            info = solve_problem(
                problem, solvers=("CLARABEL", "SCS"), accept_inaccurate=True
            )

    assert info.status == "optimal_inaccurate"
    assert info.attempts == ("CLARABEL", "SCS")
    assert "approximate solution" in caplog.text
    # And the weights from the first usable attempt are still on the variable:
    # the later solver in the chain overwrote them, so they were restored.
    assert w.value is not None
    assert float(np.sum(w.value)) == pytest.approx(1.0, abs=1e-4)


def test_inaccurate_opt_in_travels_with_the_scope():
    """``accepting_inaccurate`` is how the setting reaches a nested solve.

    ``solve_problem`` is called with no keyword from inside every optimizer's
    ``_solve``; the scope is the only thing that carries a caller's decision
    that far down.
    """
    import cvxpy as cp

    from optimization_engine.optimizers._cvxpy_helpers import (
        SolverFailure,
        accepting_inaccurate,
        solve_problem,
    )

    _w, problem, patch = _always_inaccurate_problem()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cp.Problem, "solve", patch)
        with accepting_inaccurate(True):
            assert solve_problem(problem, solvers=("CLARABEL",)).status == (
                "optimal_inaccurate"
            )
            # None inherits rather than resetting: that is what lets NCO's
            # per-cluster optimizers run under the outer solve's settings.
            with accepting_inaccurate(None):
                assert solve_problem(problem, solvers=("CLARABEL",)).status == (
                    "optimal_inaccurate"
                )
            with accepting_inaccurate(False):
                with pytest.raises(SolverFailure):
                    solve_problem(problem, solvers=("CLARABEL",))
        # And the scope is closed again on the way out.
        with pytest.raises(SolverFailure):
            solve_problem(problem, solvers=("CLARABEL",))


def test_solver_failure_distinguishes_infeasible_from_numerical():
    import cvxpy as cp

    from optimization_engine.optimizers._cvxpy_helpers import SolverFailure, solve_problem

    w = cp.Variable(1)
    impossible = cp.Problem(cp.Minimize(w), [w >= 1, w <= 0])
    with pytest.raises(SolverFailure, match="infeasible"):
        solve_problem(impossible)

    unbounded = cp.Problem(cp.Minimize(w))
    with pytest.raises(SolverFailure, match="unbounded"):
        solve_problem(unbounded)


def test_solvers_do_not_leak_inaccuracy_warnings_to_callers(returns, mu):
    """The engine reports solver status itself; the duplicate is noise."""
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.4] for a in mu.index},
        optimizer=OptimizerSpec(name="risk_parity"),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = run_engine(returns, cfg, build_frontier=True, n_frontier_points=6)
    assert not [c for c in caught if "may be inaccurate" in str(c.message)]
    assert run.result.extras["solver_status"] in ("optimal", "optimal_inaccurate")


def test_cvar_frontier_is_plotted_against_cvar(returns, mu):
    """A mean-CVaR frontier drawn on a volatility axis is not the curve solved."""
    from optimization_engine.reporting.plots import plot_efficient_frontier

    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.3] for a in mu.index},
        optimizer=OptimizerSpec(name="cvar", cvar_alpha=0.05, risk_free_rate=0.03),
    )
    frontier = run_engine(
        returns, cfg, build_frontier=True, n_frontier_points=6
    ).frontier
    assert frontier.risk_measure == "CVaR"
    assert frontier.n_failed == 0, frontier.failures
    assert frontier.summary["cvar"].notna().all()

    fig = plot_efficient_frontier(frontier, risk_free_rate=0.03)
    assert "Conditional VaR" in fig.layout.xaxis.title.text
    # Anchors and the CAL live in volatility space; they must not be drawn here.
    names = {t.name for t in fig.data}
    assert "Capital allocation line" not in " ".join(names)
    assert "Minimum variance" not in names

    x = list(fig.data[0].x)
    np.testing.assert_allclose(
        x, frontier.plot_frame()["cvar"].values, rtol=1e-9
    )


def test_mean_variance_frontier_still_uses_volatility(returns, mu):
    from optimization_engine.reporting.plots import plot_efficient_frontier

    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.3] for a in mu.index},
        optimizer=OptimizerSpec(name="mean_variance", risk_free_rate=0.03),
    )
    frontier = run_engine(
        returns, cfg, build_frontier=True, n_frontier_points=6
    ).frontier
    fig = plot_efficient_frontier(frontier, risk_free_rate=0.03)
    assert "Volatility" in fig.layout.xaxis.title.text
    assert "Minimum variance" in {t.name for t in fig.data}


# ---------------------------------------------------------------------------
# Constraint projection for allocate-then-constrain methods
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grouped_constraints(cov):
    groups = {}
    for asset in cov.columns:
        if "Equity" in asset:
            groups[asset] = "Equity"
        elif asset in ("US_Treasuries", "TIPS", "IG_Credit", "HY_Credit", "EM_Debt", "Cash"):
            groups[asset] = "FixedIncome"
        else:
            groups[asset] = "Alternatives"
    return groups


@pytest.mark.parametrize("method", ["equal_weight", "inverse_vol", "hrp"])
def test_projection_methods_now_honour_group_budgets(
    returns, mu, grouped_constraints, method
):
    """1/N used to sail past an alternatives cap because projection ignored groups."""
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.5] for a in mu.index},
        groups=grouped_constraints,
        group_bounds={
            "Equity": [0.2, 0.7],
            "FixedIncome": [0.2, 0.7],
            "Alternatives": [0.0, 0.3],
        },
        optimizer=OptimizerSpec(name=method),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = run_engine(returns, cfg)
    assert run.result.is_compliant, run.result.violations
    grouped = run.result.weights.groupby(
        run.result.weights.index.map(grouped_constraints)
    ).sum()
    assert grouped["Alternatives"] <= 0.30 + 1e-6
    assert grouped["Equity"] >= 0.20 - 1e-6


def test_projection_reports_how_far_the_mandate_moved_the_answer(
    returns, mu, grouped_constraints
):
    loose = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 1.0] for a in mu.index},
        optimizer=OptimizerSpec(name="equal_weight"),
    )
    tight = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.5] for a in mu.index},
        groups=grouped_constraints,
        group_bounds={"Alternatives": [0.0, 0.10]},
        optimizer=OptimizerSpec(name="equal_weight"),
    )
    assert run_engine(returns, loose).result.extras["projection_distance"] == pytest.approx(
        0.0, abs=1e-9
    )
    moved = run_engine(returns, tight).result.extras["projection_distance"]
    assert moved > 0.1
    assert "moved" in run_engine(returns, tight).result.extras["bounds_note"].lower()


def test_projection_finds_the_closest_feasible_point(cov):
    """Not just *a* feasible point — the nearest one."""
    import cvxpy as cp

    from optimization_engine.optimizers._bounds import project_to_constraints

    assets = list(cov.columns)
    groups = {a: ("A" if i < 4 else "B") for i, a in enumerate(assets)}
    cons = PortfolioConstraints(
        bounds={a: (0.0, 1.0) for a in assets},
        groups=groups,
        group_bounds={"A": (0.0, 0.20)},
    )
    w0 = np.ones(len(assets)) / len(assets)
    projected, distance = project_to_constraints(w0, assets, cons)

    x = cp.Variable(len(assets))
    group_a = [i for i, a in enumerate(assets) if groups[a] == "A"]
    cp.Problem(
        cp.Minimize(cp.sum_squares(x - w0)),
        [cp.sum(x) == 1, x >= 0, x <= 1, cp.sum(x[group_a]) <= 0.20],
    ).solve()
    np.testing.assert_allclose(projected, np.asarray(x.value).flatten(), atol=1e-6)
    assert distance == pytest.approx(
        float(np.abs(projected - w0).sum()) / 2.0, rel=1e-9
    )


def test_projection_raises_when_nothing_is_feasible(cov):
    from optimization_engine.optimizers._bounds import (
        InfeasibleBoundsError,
        project_to_constraints,
    )

    assets = list(cov.columns)
    groups = {a: "A" for a in assets}
    cons = PortfolioConstraints(
        bounds={a: (0.0, 1.0) for a in assets},
        groups=groups,
        group_bounds={"A": (0.0, 0.20)},  # every asset is in A, but Sum(w)=1
    )
    with pytest.raises(InfeasibleBoundsError):
        project_to_constraints(np.ones(len(assets)) / len(assets), assets, cons)


def test_bounds_only_projection_still_used_without_groups(cov):
    from optimization_engine.optimizers._bounds import project_to_constraints

    assets = list(cov.columns)
    cons = PortfolioConstraints(bounds={a: (0.0, 0.10) for a in assets})
    projected, _ = project_to_constraints(np.ones(len(assets)) / len(assets), assets, cons)
    assert projected.max() <= 0.10 + 1e-9
    assert projected.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("with_groups", [False, True])
def test_projection_distance_is_one_way_on_both_paths(cov, with_groups):
    """The grouped and ungrouped branches must report the same measure.

    They are separate code paths, and a distance that means "one-way" on one
    and something else on the other silently rescales the number the UI
    escalates a warning on.
    """
    from optimization_engine.optimizers._bounds import project_to_constraints

    assets = list(cov.columns)
    n = len(assets)
    kwargs = {"bounds": {a: (0.0, 0.15) for a in assets}}
    if with_groups:
        kwargs["groups"] = {a: ("A" if i < 3 else "B") for i, a in enumerate(assets)}
        kwargs["group_bounds"] = {"A": (0.0, 0.20)}
    cons = PortfolioConstraints(**kwargs)

    start = np.zeros(n)
    start[0] = 0.6
    start[1:] = 0.4 / (n - 1)
    projected, distance = project_to_constraints(start, assets, cons)

    assert distance == pytest.approx(
        float(np.abs(projected - start).sum()) / 2.0, rel=1e-12
    )
    # Both vectors sum to 1, so one-way distance can never exceed 1.
    assert 0.0 <= distance <= 1.0
    assert distance > 0.0, "the constraints should have bound here"


def test_weight_cleaning_respects_group_budgets_too():
    """Dust removal plus rescaling must not push a group over its cap."""
    from optimization_engine.optimizers.base import BaseOptimizer

    class Dusty(BaseOptimizer):
        name = "dusty"

        def _solve(self):
            # Sub-tolerance dust on the last name; rescaling the rest would
            # take group G from 0.50 to 0.80 against a 0.60 cap.
            return np.array([0.4999995, 0.3, 0.2, 5e-7])

    assets = ["a", "b", "c", "d"]
    cov = pd.DataFrame(np.eye(4) * 0.04, index=assets, columns=assets)
    constraints = PortfolioConstraints(
        bounds={a: (0.0, 0.45) for a in assets},
        groups={"a": "G", "b": "G", "c": "H", "d": "H"},
        group_bounds={"G": (0.0, 0.60)},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = Dusty(cov_matrix=cov, constraints=constraints).optimize()

    assert result.weights.sum() == pytest.approx(1.0)
    assert result.weights[["a", "b"]].sum() <= 0.60 + 1e-6
    assert result.weights.max() <= 0.45 + 1e-6
    assert result.is_compliant, result.violations


# ---------------------------------------------------------------------------
# Bayes-Stein on the right units (review fix O5)
# ---------------------------------------------------------------------------


def test_james_stein_intensity_does_not_depend_on_the_annualization_basis():
    rng = np.random.default_rng(3)
    n, t = 6, 500
    per_period = pd.DataFrame(rng.normal(0.0, 0.01, size=(t, n)), columns=[f"X{i}" for i in range(n)])
    _, daily = james_stein_shrinkage(per_period.mean(), per_period.cov(), t, periods_per_year=1)
    _, annual = james_stein_shrinkage(
        per_period.mean() * 252, per_period.cov() * 252, t, periods_per_year=252
    )
    _, monthly = james_stein_shrinkage(
        per_period.mean() * 12, per_period.cov() * 12, t, periods_per_year=12
    )
    assert annual == pytest.approx(daily, rel=1e-9)
    assert monthly == pytest.approx(daily, rel=1e-9)
    # Six assets whose means differ by nothing but noise: most of the spread
    # is noise, and the intensity should say so. Before the fix, annualized
    # inputs gave an intensity about 250 times too small (~0.01 here).
    assert annual > 0.3


def test_shrunk_mean_actually_shrinks_noise():
    from optimization_engine.data.covariance import expected_returns_from_history

    rng = np.random.default_rng(5)
    n, t = 8, 750
    idx = pd.date_range("2020-01-01", periods=t, freq="B")
    noise = pd.DataFrame(
        rng.normal(0.0003, 0.012, size=(t, n)), index=idx, columns=[f"X{i}" for i in range(n)]
    )
    raw = expected_returns_from_history(noise, method="mean")
    shrunk = expected_returns_from_history(noise, method="shrunk_mean")
    assert shrunk.std() < 0.7 * raw.std(), (raw.std(), shrunk.std())
