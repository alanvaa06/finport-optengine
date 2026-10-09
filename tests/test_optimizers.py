"""Smoke tests for the optimization engine."""

from __future__ import annotations

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

from optimization_engine.config import EngineConfig, OptimizerSpec
from optimization_engine.data.covariance import covariance_matrix
from optimization_engine.data.loader import prices_to_returns, sample_dataset
from optimization_engine.engine import run_engine
from optimization_engine.optimizers import ConfigurationError
from optimization_engine.optimizers.factory import available_optimizers


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    prices = sample_dataset(n_periods=252 * 4, seed=7)
    return prices_to_returns(prices)


@pytest.fixture(scope="module")
def baseline_config(returns: pd.DataFrame) -> EngineConfig:
    expected = (1 + returns).prod() ** (252 / len(returns)) - 1
    return EngineConfig(
        expected_returns=expected.to_dict(),
        bounds={a: [0.0, 0.5] for a in returns.columns},
        groups={a: "All" for a in returns.columns},
        group_bounds={"All": [1.0, 1.0]},
        optimizer=OptimizerSpec(name="mean_variance", risk_free_rate=0.03),
    )


def test_lists_optimizers_includes_core_methods():
    names = available_optimizers()
    for required in (
        "mean_variance",
        "min_variance",
        "max_sharpe",
        "risk_parity",
        "hrp",
        "black_litterman",
        "cvar",
        "max_diversification",
        "equal_weight",
        "inverse_vol",
    ):
        assert required in names


def test_covariance_methods(returns: pd.DataFrame):
    for method in ["sample", "ledoit_wolf", "ewma", "semi"]:
        cov = covariance_matrix(returns, method=method)
        assert cov.shape == (returns.shape[1], returns.shape[1])
        eigvals = np.linalg.eigvalsh(cov.values)
        assert np.all(eigvals > -1e-8), f"{method}: cov has negative eigenvalues"


@pytest.mark.parametrize(
    "method",
    [
        "min_variance",
        "max_sharpe",
        "mean_variance",
        "risk_parity",
        "hrp",
        "max_diversification",
        "inverse_vol",
        "equal_weight",
    ],
)
def test_optimizer_runs(returns: pd.DataFrame, baseline_config: EngineConfig, method: str):
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        group_bounds=baseline_config.group_bounds,
        optimizer=OptimizerSpec(name=method, risk_free_rate=0.03),
    )
    run = run_engine(returns, cfg)
    w = run.result.weights
    assert pytest.approx(w.sum(), abs=1e-4) == 1.0
    assert (w >= -1e-6).all(), "Weights must be non-negative"
    assert (w <= 0.5 + 1e-6).all(), "Weights must respect upper bound"


def test_target_return(returns: pd.DataFrame, baseline_config: EngineConfig):
    """A target above the minimum-variance return binds, and says so.

    The target is a floor (``μ'w ≥ R*``), not an equality, so "the target was
    met" is only the whole story when the floor actually bound — otherwise the
    answer is minimum variance and the target had no say in it. 6% sits well
    above this panel's 0.43% minimum-variance return, so it binds and the
    realized return is the target.
    """
    target = 0.06
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        group_bounds=baseline_config.group_bounds,
        optimizer=OptimizerSpec(name="mean_variance", target_return=target),
    )
    run = run_engine(returns, cfg)
    assert run.result.expected_return == pytest.approx(target, abs=1e-3)
    assert run.result.extras["target_return_binding"] is True
    assert run.result.extras["target_return_slack"] == pytest.approx(0.0, abs=1e-6)


def test_target_return_below_gmv_returns_gmv(
    returns: pd.DataFrame, baseline_config: EngineConfig
):
    """A target under the minimum-variance return buys the GMV portfolio.

    Under the old equality target this returned a point on the *dominated*
    lower branch — the same volatility for less return — and reported the
    unreachable target as if it had been achieved. The floor cannot do that:
    minimum variance already clears it, so minimum variance is the answer, and
    ``target_return_binding`` says the target never entered into it.
    """
    common = dict(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        group_bounds=baseline_config.group_bounds,
    )
    gmv = run_engine(
        returns, EngineConfig(**common, optimizer=OptimizerSpec(name="min_variance"))
    ).result
    target = gmv.expected_return - 0.02
    assert target < gmv.expected_return

    run = run_engine(
        returns,
        EngineConfig(
            **common,
            optimizer=OptimizerSpec(name="mean_variance", target_return=target),
        ),
    )
    assert run.result.extras["target_return_binding"] is False
    assert run.result.extras["target_return_slack"] == pytest.approx(0.02, abs=1e-5)
    # It is the minimum-variance portfolio, to the accuracy the two QPs are
    # solved to. CLARABEL agrees with itself to ~1e-6 on the weights here (the
    # 0.5 cap is active, which is where a QP loses digits) and to ~1e-7 on the
    # summary statistics; both problems are re-solved from scratch, so this is
    # solver noise, not a difference in the answer.
    assert run.result.expected_return == pytest.approx(gmv.expected_return, abs=1e-6)
    assert run.result.expected_volatility == pytest.approx(
        gmv.expected_volatility, abs=1e-6
    )
    np.testing.assert_allclose(
        run.result.weights.values, gmv.weights.values, atol=5e-5
    )


def test_inverse_vol_zero_variance_raises(returns: pd.DataFrame):
    """A constant series cannot be inverse-vol weighted, and must not be dropped.

    ``1/σ`` is undefined at σ = 0. The old code gave the asset weight 0 and
    carried on, so a name the analyst had put in the universe silently left the
    book — the same failure mode max-diversification already refuses.
    """
    from optimization_engine.optimizers.naive import InverseVolatilityOptimizer

    cov = covariance_matrix(returns, method="sample")
    dead = [cov.columns[0], cov.columns[3]]
    for asset in dead:
        cov.loc[asset, :] = 0.0
        cov.loc[:, asset] = 0.0

    with pytest.raises(ValueError) as excinfo:
        InverseVolatilityOptimizer(cov_matrix=cov).optimize()
    message = str(excinfo.value)
    for asset in dead:
        assert asset in message, message
    assert "zero-variance" in message


def test_inverse_vol_still_solves_a_healthy_panel(returns: pd.DataFrame):
    """The guard only fires on a degenerate column, not on ordinary data."""
    from optimization_engine.optimizers.naive import InverseVolatilityOptimizer

    cov = covariance_matrix(returns, method="sample")
    weights = InverseVolatilityOptimizer(cov_matrix=cov).optimize().weights
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert (weights > 0).all()


def _indefinite_cov() -> pd.DataFrame:
    """ρ(a,b) = ρ(b,c) = 0.9 with ρ(a,c) = −0.9: no joint distribution has these."""
    names = ["a", "b", "c"]
    corr = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    vol = np.array([0.10, 0.20, 0.15])
    return pd.DataFrame(np.outer(vol, vol) * corr, index=names, columns=names)


def test_an_indefinite_covariance_used_to_solve_to_a_negative_variance():
    """The defect itself, pinned so the refusal below has a reason on record.

    ``cp.psd_wrap`` tells CVXPY to trust the matrix rather than check it. On
    this one the long-short minimum-variance solve came back ``optimal`` at
    ``w = [1, −1, 1]``, where ``w'Σw = −0.0445`` — reported as zero volatility,
    because the base class floors the variance at zero before rooting it.
    """
    from optimization_engine.optimizers.base import NonPSDCovarianceError
    from optimization_engine.optimizers.mean_variance import MinVarianceOptimizer

    cov = _indefinite_cov()
    w = np.array([1.0, -1.0, 1.0])
    assert float(w @ cov.values @ w) == pytest.approx(-0.0445)

    from optimization_engine.optimizers.base import PortfolioConstraints

    optimizer = MinVarianceOptimizer(
        cov_matrix=cov,
        constraints=PortfolioConstraints(
            long_only=False, bounds={a: (-1.0, 1.0) for a in cov.index}
        ),
    )
    with pytest.raises(NonPSDCovarianceError, match="eigenvalue") as raised:
        optimizer.optimize()
    assert raised.value.min_eigenvalue == pytest.approx(
        float(np.linalg.eigvalsh(cov.values)[0])
    )
    assert "nearest_psd" in str(raised.value)


@pytest.mark.parametrize("name", available_optimizers())
def test_every_optimizer_refuses_an_indefinite_covariance(name: str):
    """The check lives in ``BaseOptimizer.optimize``, so no method can skip it.

    The engine's own estimators repair an indefinite estimate with
    ``nearest_psd`` before any optimizer sees it, which is why only a caller of
    the optimizers directly could hand one over. Repairing it again here would
    solve a different matrix from the one passed, by an amount nobody chose;
    refusing names the problem and the one-line repair.
    """
    from optimization_engine.optimizers.base import NonPSDCovarianceError
    from optimization_engine.optimizers.factory import optimizer_factory

    cov = _indefinite_cov()
    rng = np.random.default_rng(0)
    history = pd.DataFrame(
        rng.normal(0.0004, 0.01, size=(300, 3)), columns=cov.columns
    )
    cfg = EngineConfig(
        optimizer=OptimizerSpec(name=name),
        benchmark_weights={a: 1 / 3 for a in cov.columns},
    )
    optimizer = optimizer_factory(
        cfg,
        cov,
        expected_returns=pd.Series([0.05, 0.08, 0.06], index=cov.columns),
        returns=history,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NonPSDCovarianceError):
            optimizer.optimize()


def test_a_covariance_on_the_psd_boundary_still_solves(returns: pd.DataFrame):
    """Eigenvalues at −1e-18 are rounding, not indefiniteness.

    Fewer observations than assets make the sample covariance singular, and
    the clipped estimate carries eigenvalues a few ulps either side of zero.
    The check is relative to the matrix's own trace, so that is not refused.
    """
    from optimization_engine.optimizers.mean_variance import MinVarianceOptimizer

    cov = covariance_matrix(returns.iloc[:8], method="sample")
    assert len(cov) > 8
    assert np.linalg.eigvalsh(cov.values)[0] < 1e-12
    weights = MinVarianceOptimizer(cov_matrix=cov).optimize().weights
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("name", ["inverse_vol", "hrp", "herc", "nco", "max_diversification"])
def test_a_constant_column_is_zero_variance_even_with_rounding(name: str):
    """A constant series is not low-risk; its sample variance is rounding.

    A cash column at a flat 0.01% a day has sample variance 1.86e-37, not 0,
    so the ``σ > 0`` guards let it through: inverse volatility and HRP put
    100% of the book in it, against the 0.7.0 changelog's "inverse volatility
    refuses a zero-variance asset". The guards are now relative to the largest
    variance in the matrix, so rounding residue counts as zero and a genuinely
    quiet asset does not.
    """
    from optimization_engine.optimizers.factory import optimizer_factory

    rng = np.random.default_rng(1)
    history = pd.DataFrame(
        rng.standard_normal((500, 4)) * 0.01 + 0.0004,
        columns=["w", "x", "y", "z"],
        index=pd.bdate_range("2020-01-01", periods=500),
    )
    history["cash"] = 0.0001
    cov = covariance_matrix(history, method="sample")
    assert 0.0 < cov.loc["cash", "cash"] < 1e-30, "fixture no longer rounds"

    optimizer = optimizer_factory(
        EngineConfig(optimizer=OptimizerSpec(name=name)), cov, returns=history
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="(?i)zero-variance") as raised:
            optimizer.optimize()
    assert "cash" in str(raised.value)


def test_a_quiet_asset_is_not_mistaken_for_a_constant_one(returns: pd.DataFrame):
    """The sample panel's Cash line has 0.5% volatility and must still be held."""
    from optimization_engine.optimizers.hrp import HRPOptimizer
    from optimization_engine.optimizers.naive import InverseVolatilityOptimizer

    cov = covariance_matrix(returns, method="sample")
    quiet = cov.values.diagonal().argmin()
    assert cov.values.diagonal()[quiet] < 1e-3 * cov.values.diagonal().max()
    for optimizer in (InverseVolatilityOptimizer, HRPOptimizer):
        weights = optimizer(cov_matrix=cov).optimize().weights
        assert weights.iloc[quiet] > 0.0


def test_cvar_extras_keys(returns: pd.DataFrame, baseline_config: EngineConfig):
    """``√ppy`` scaling is reported under a name that says what it is.

    ``cvar_annualized`` was the per-period CVaR times ``√252``, which
    annualizes a tail measure only if returns are iid Gaussian — the
    assumption a tail measure exists to avoid relying on. Both key pairs are
    written for one release, and the rename is announced once per solve
    because a read-shim cannot work: ``**extras`` and ``dict(extras)`` never
    call ``__getitem__``.
    """
    from optimization_engine.optimizers.cvar import DEPRECATED_EXTRAS_KEYS

    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        optimizer=OptimizerSpec(name="cvar", cvar_alpha=0.05),
    )
    with pytest.warns(DeprecationWarning, match="cvar_annualized"):
        extras = run_engine(returns, cfg).result.extras

    scale = np.sqrt(252)
    assert extras["cvar_sqrt_t_scaled"] == pytest.approx(extras["cvar_period"] * scale)
    assert extras["var_sqrt_t_scaled"] == pytest.approx(extras["var_period"] * scale)
    # The deprecated names still resolve, to exactly the same numbers.
    for old, new in DEPRECATED_EXTRAS_KEYS.items():
        assert extras[old] == extras[new]
    # ζ keeps its own name; it is the VaR, not the objective.
    assert "cvar_solver_zeta" in extras
    assert "sqrt(252)" in extras["cvar_note"]


def test_cvar_deprecation_names_both_renamed_keys(
    returns: pd.DataFrame, baseline_config: EngineConfig
):
    """One warning per solve, naming every key that moved."""
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        optimizer=OptimizerSpec(name="cvar", cvar_alpha=0.05),
    )
    with pytest.warns(DeprecationWarning) as caught:
        run_engine(returns, cfg)
    messages = [
        str(w.message)
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "cvar_annualized" in str(w.message)
    ]
    assert len(messages) == 1, messages
    for key in ("cvar_annualized", "var_annualized",
                "cvar_sqrt_t_scaled", "var_sqrt_t_scaled"):
        assert key in messages[0]


def test_cdar_extras_keys(returns: pd.DataFrame):
    """ζ and the objective are two different numbers, under two names.

    ``cdar_solver_objective`` used to hold ζ — the drawdown-at-risk threshold,
    which mean-CVaR already calls ``cvar_solver_zeta``. The objective the LP
    actually minimizes is ``ζ + Σz/(α·T)``, and it is strictly larger whenever
    any drawdown exceeds the threshold.
    """
    from optimization_engine.optimizers.cdar import CDaROptimizer

    cov = covariance_matrix(returns, method="sample")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        extras = CDaROptimizer(
            returns=returns,
            cov_matrix=cov,
            alpha=0.05,
        ).optimize().extras

    assert "cdar_solver_zeta" in extras
    assert "cdar_solver_objective" in extras
    # ζ is the threshold; the objective averages the tail beyond it, so it is
    # the larger of the two on any path with a drawdown worse than ζ.
    assert extras["cdar_solver_objective"] > extras["cdar_solver_zeta"]
    # The objective is the solved problem's own value.
    assert extras["cdar_solver_objective"] == pytest.approx(
        extras["objective_value"], rel=1e-9
    )


def test_cvar_optimizer(returns: pd.DataFrame, baseline_config: EngineConfig):
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        group_bounds=baseline_config.group_bounds,
        optimizer=OptimizerSpec(name="cvar", cvar_alpha=0.05),
    )
    run = run_engine(returns, cfg)
    assert pytest.approx(run.result.weights.sum(), abs=1e-3) == 1.0


def test_efficient_frontier(returns: pd.DataFrame, baseline_config: EngineConfig):
    run = run_engine(returns, baseline_config, build_frontier=True, n_frontier_points=10)
    assert run.frontier is not None
    summary = run.frontier.summary.dropna(subset=["expected_return"])
    assert (summary["expected_return"].diff().dropna() >= -1e-6).all()


def test_black_litterman(returns: pd.DataFrame, baseline_config: EngineConfig):
    spec = OptimizerSpec(
        name="black_litterman",
        bl_views={returns.columns[0]: 0.12},
        bl_view_confidences={returns.columns[0]: 0.0001},
        risk_free_rate=0.03,
    )
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        group_bounds=baseline_config.group_bounds,
        optimizer=spec,
    )
    run = run_engine(returns, cfg)
    assert run.result.weights.sum() == pytest.approx(1.0, abs=1e-3)


def test_risk_parity_equal_contributions(returns: pd.DataFrame):
    cfg = EngineConfig(
        expected_returns={a: 0.05 for a in returns.columns},
        bounds={a: [0.0, 1.0] for a in returns.columns},
        optimizer=OptimizerSpec(name="risk_parity"),
    )
    run = run_engine(returns, cfg)
    rc = run.risk_contributions().values
    rc = rc / rc.sum()
    target = np.ones_like(rc) / len(rc)
    assert np.max(np.abs(rc - target)) < 0.05  # ERC: roughly equal


def test_factory_raises_when_required_mu_missing(returns):
    # Test the factory directly: empty config + no override -> ConfigurationError.
    # (run_engine has a historical-mean fallback that fills mu in that case;
    # this validation matters for direct factory use.)
    from optimization_engine.data.covariance import covariance_matrix
    from optimization_engine.optimizers.factory import optimizer_factory

    cfg = EngineConfig(
        expected_returns={},
        bounds={a: [0.0, 1.0] for a in returns.columns},
        optimizer=OptimizerSpec(name="mean_variance"),
    )
    cov = covariance_matrix(returns, method="ledoit_wolf")
    with pytest.raises(ConfigurationError, match="expected_returns"):
        optimizer_factory(cfg, cov, expected_returns=None, returns=returns)


def test_factory_raises_when_returns_missing_for_cvar(returns):
    # CVaR needs the returns DataFrame; we exercise the factory directly
    # because the engine always supplies returns.
    from optimization_engine.data.covariance import covariance_matrix
    from optimization_engine.optimizers.factory import optimizer_factory

    cfg = EngineConfig(
        expected_returns={a: 0.05 for a in returns.columns},
        bounds={a: [0.0, 1.0] for a in returns.columns},
        optimizer=OptimizerSpec(name="cvar"),
    )
    cov = covariance_matrix(returns, method="ledoit_wolf")
    with pytest.raises(ConfigurationError, match="returns"):
        optimizer_factory(cfg, cov, expected_returns=None, returns=None)


def test_factory_warns_on_incompatible_target_return(returns, baseline_config, caplog):
    import logging
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        optimizer=OptimizerSpec(name="hrp", target_return=0.05),
    )
    with caplog.at_level(logging.WARNING):
        run_engine(returns, cfg)
    assert any("target_return" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    ("name", "target"),
    [("max_sharpe", {"target_return": 0.50}), ("risk_parity", {"target_volatility": 0.001})],
)
def test_an_ignored_target_is_recorded_and_not_preflighted(
    returns, baseline_config, name: str, target: dict
):
    """A target the method ignores is said to be ignored, and is not judged.

    Max-Sharpe asked for 9.5% delivered 7.19% and risk parity asked for 2%
    volatility delivered 7.18%, with only a log line to say so: nothing in
    ``extras``, nothing in the violations. Meanwhile the pre-flight validated
    the target the solve would ignore, so an unreachable one was a fatal
    finding and ``raise_on_infeasible`` refused a solve that never looks at it.
    """
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        optimizer=OptimizerSpec(name=name, risk_free_rate=0.0, **target),
    )

    run = run_engine(returns, cfg, raise_on_infeasible=True)

    (field,) = target
    assert field in run.result.extras["ignored_constraints"]
    assert not [i for i in run.feasibility.issues if "target" in i.code]


@pytest.mark.parametrize("name", ["max_sharpe", "risk_parity"])
def test_an_ignored_target_is_recorded_on_the_direct_api_too(returns, name: str):
    """The record comes from the optimizer, not only from the config path."""
    from optimization_engine.optimizers.base import PortfolioConstraints
    from optimization_engine.optimizers.mean_variance import MaxSharpeOptimizer
    from optimization_engine.optimizers.risk_parity import RiskParityOptimizer

    cov = covariance_matrix(returns, method="sample")
    mu = pd.Series(0.06, index=cov.columns) + np.linspace(0, 0.04, len(cov))
    constraints = PortfolioConstraints(target_return=0.095, target_volatility=0.02)
    if name == "max_sharpe":
        optimizer = MaxSharpeOptimizer(
            expected_returns=mu, cov_matrix=cov, constraints=constraints
        )
    else:
        optimizer = RiskParityOptimizer(cov_matrix=cov, constraints=constraints)

    ignored = optimizer.optimize().extras["ignored_constraints"]
    assert {"target_return", "target_volatility"} <= set(ignored)


def test_risk_parity_records_that_it_ignores_an_open_budget(returns):
    """Risk parity normalizes to one whatever the budget says, and now says so."""
    from optimization_engine.optimizers.base import PortfolioConstraints
    from optimization_engine.optimizers.risk_parity import RiskParityOptimizer

    cov = covariance_matrix(returns, method="sample")
    optimizer = RiskParityOptimizer(
        cov_matrix=cov, constraints=PortfolioConstraints(fully_invested=False)
    )
    with pytest.warns(UserWarning, match="fully_invested=False"):
        result = optimizer.optimize()
    assert "fully_invested" in result.extras["ignored_constraints"]
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_min_variance_says_how_much_an_open_budget_invests(returns):
    """With no budget the lowest-variance book is close to empty, and says so.

    It came back as weights around 1e-4, ``optimal``, a NaN Sharpe and nothing
    else. Minimum variance does honour the open budget — the pre-flight relies
    on exactly that solve for the volatility floor of an open-budget mandate —
    so the answer stands, with the invested fraction and a note beside it.
    """
    from optimization_engine.optimizers.base import PortfolioConstraints
    from optimization_engine.optimizers.mean_variance import MinVarianceOptimizer

    cov = covariance_matrix(returns, method="sample")
    result = MinVarianceOptimizer(
        cov_matrix=cov, constraints=PortfolioConstraints(fully_invested=False)
    ).optimize()

    assert 0.0 <= result.extras["invested_fraction"] < 0.01  # solver tolerance
    assert "open budget" in result.extras["budget_note"]


@pytest.mark.parametrize("linkage", ["single", "average", "complete", "ward"])
def test_hrp_linkage_methods(returns, baseline_config, linkage):
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        optimizer=OptimizerSpec(name="hrp", hrp_linkage=linkage),
    )
    run = run_engine(returns, cfg)
    w = run.result.weights
    assert pytest.approx(w.sum(), abs=1e-3) == 1.0
    assert (w >= -1e-6).all()
    assert (w <= 0.5 + 1e-6).all()


def test_max_diversification_respects_tight_bounds(returns):
    cfg = EngineConfig(
        expected_returns={a: 0.05 for a in returns.columns},
        bounds={a: [0.0, 0.3] for a in returns.columns},
        optimizer=OptimizerSpec(name="max_diversification"),
    )
    run = run_engine(returns, cfg)
    w = run.result.weights
    assert (w <= 0.3 + 1e-6).all(), w[w > 0.3].to_dict()
    assert (w >= -1e-6).all()
    assert pytest.approx(w.sum(), abs=1e-4) == 1.0


@pytest.mark.parametrize("method", ["equal_weight", "inverse_vol"])
def test_naive_methods_respect_tight_bounds(returns, method):
    cfg = EngineConfig(
        expected_returns={a: 0.05 for a in returns.columns},
        bounds={a: [0.0, 0.2] for a in returns.columns},
        optimizer=OptimizerSpec(name=method),
    )
    run = run_engine(returns, cfg)
    w = run.result.weights
    assert (w <= 0.2 + 1e-6).all()
    assert (w >= -1e-6).all()
    assert pytest.approx(w.sum(), abs=1e-4) == 1.0


def test_constrained_risk_parity_respects_bounds(returns):
    cfg = EngineConfig(
        expected_returns={a: 0.05 for a in returns.columns},
        bounds={a: [0.05, 0.25] for a in returns.columns},
        optimizer=OptimizerSpec(name="risk_parity"),
    )
    run = run_engine(returns, cfg)
    w = run.result.weights
    assert (w >= 0.05 - 1e-5).all(), w[w < 0.05].to_dict()
    assert (w <= 0.25 + 1e-5).all(), w[w > 0.25].to_dict()
    assert pytest.approx(w.sum(), abs=1e-4) == 1.0


def test_risk_parity_with_group_bounds(returns):
    cols = list(returns.columns)
    half = len(cols) // 2
    groups = {a: ("A" if i < half else "B") for i, a in enumerate(cols)}
    cfg = EngineConfig(
        expected_returns={a: 0.05 for a in cols},
        bounds={a: [0.0, 1.0] for a in cols},
        groups=groups,
        group_bounds={"A": [0.45, 0.55], "B": [0.45, 0.55]},
        optimizer=OptimizerSpec(name="risk_parity"),
    )
    run = run_engine(returns, cfg)
    g = run.result.weights.groupby(groups).sum()
    assert 0.45 - 1e-3 <= g["A"] <= 0.55 + 1e-3
    assert 0.45 - 1e-3 <= g["B"] <= 0.55 + 1e-3


def test_engine_uses_ema_expected_returns_when_specified(returns):
    cfg = EngineConfig(
        expected_returns={},  # empty -> engine seeds from history
        bounds={a: [0.0, 1.0] for a in returns.columns},
        expected_returns_method="ema",
        ema_span=120,
        optimizer=OptimizerSpec(name="min_variance"),  # min_variance ignores mu but engine still computes it
    )
    run = run_engine(returns, cfg)
    # Sanity: μ vector populated with finite values.
    assert run.expected_returns.notna().all()
    assert run.expected_returns.shape[0] == returns.shape[1]


def test_engine_uses_capm_expected_returns_when_specified(returns):
    cols = list(returns.columns)
    cfg = EngineConfig(
        expected_returns={},
        bounds={a: [0.0, 1.0] for a in cols},
        expected_returns_method="capm",
        market_weights={a: 1.0 / len(cols) for a in cols},
        market_return=0.08,
        optimizer=OptimizerSpec(name="min_variance", risk_free_rate=0.03),
    )
    run = run_engine(returns, cfg)
    assert run.expected_returns.notna().all()


def test_engine_default_method_unchanged(returns, baseline_config):
    # Existing test_optimizer_runs already covers this; just ensure default
    # historical_mean still works when method left at default.
    cfg = EngineConfig(
        expected_returns={},
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        optimizer=OptimizerSpec(name="min_variance"),
    )
    run = run_engine(returns, cfg)
    assert run.expected_returns.notna().all()


def test_cvar_with_target_return(returns, baseline_config):
    target = 0.05
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        optimizer=OptimizerSpec(
            name="cvar", cvar_alpha=0.05, target_return=target,
        ),
    )
    run = run_engine(returns, cfg)
    assert run.result.expected_return >= target - 1e-3


def test_black_litterman_no_views_runs(returns, baseline_config):
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        group_bounds=baseline_config.group_bounds,
        optimizer=OptimizerSpec(name="black_litterman", risk_aversion=2.5),
    )
    run = run_engine(returns, cfg)
    assert pytest.approx(run.result.weights.sum(), abs=1e-3) == 1.0
    assert (run.result.weights >= -1e-6).all()


@pytest.mark.parametrize("method", [
    "mean_variance", "min_variance", "max_sharpe", "cvar", "black_litterman",
])
def test_group_bounds_enforced_for_hard_methods(returns, method):
    cols = list(returns.columns)
    half = len(cols) // 2
    groups = {a: ("A" if i < half else "B") for i, a in enumerate(cols)}
    cfg = EngineConfig(
        expected_returns={a: 0.05 for a in cols},
        bounds={a: [0.0, 1.0] for a in cols},
        groups=groups,
        group_bounds={"A": [0.4, 0.6], "B": [0.4, 0.6]},
        optimizer=OptimizerSpec(name=method, risk_free_rate=0.0),
    )
    run = run_engine(returns, cfg)
    g = run.result.weights.groupby(groups).sum()
    assert 0.4 - 2e-3 <= g["A"] <= 0.6 + 2e-3, g.to_dict()
    assert 0.4 - 2e-3 <= g["B"] <= 0.6 + 2e-3, g.to_dict()


def test_infeasible_target_raises_clearly(returns, baseline_config):
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        optimizer=OptimizerSpec(name="mean_variance", target_return=10.0),
    )
    with pytest.raises(RuntimeError, match=r"infeasible|status|Solver"):
        run_engine(returns, cfg)


# ---------------------------------------------------------------------------
# Black-Litterman convention and idempotence; ray-space constraints
# (review fixes O1, O2, O3)
# ---------------------------------------------------------------------------


def _synthetic_cov(n: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    names = [f"X{i}" for i in range(n)]
    return pd.DataFrame(a @ a.T / 50 + np.eye(n) * 0.01, index=names, columns=names)


def test_black_litterman_with_no_views_returns_the_market_portfolio():
    from optimization_engine.optimizers.base import PortfolioConstraints
    from optimization_engine.optimizers.black_litterman import BlackLittermanOptimizer

    cov = _synthetic_cov()
    market = pd.Series([0.4, 0.25, 0.15, 0.12, 0.08], index=cov.index)
    for rf in (0.0, 0.03):
        bl = BlackLittermanOptimizer(
            cov_matrix=cov,
            market_weights=market,
            constraints=PortfolioConstraints(
                long_only=True, fully_invested=True, bounds={a: (0.0, 1.0) for a in cov.index}
            ),
            risk_free_rate=rf,
            risk_aversion=2.5,
        )
        weights = bl.optimize().weights
        # The defining check of the model: no views, no move. With the
        # risk-aversion conventions of the prior and the sub-solve out of step
        # this landed halfway to the minimum-variance portfolio.
        np.testing.assert_allclose(weights.to_numpy(), market.to_numpy(), atol=1e-3)


def test_black_litterman_solves_are_idempotent():
    from optimization_engine.optimizers.black_litterman import BlackLittermanOptimizer

    cov = _synthetic_cov()
    prior = cov.copy()
    bl = BlackLittermanOptimizer(
        cov_matrix=cov,
        market_weights=pd.Series(0.2, index=cov.index),
        views={"X0": 0.10},
        view_confidences={"X0": 0.0004},
    )
    first = bl.optimize()
    second = bl.optimize()
    third = bl.optimize()
    pd.testing.assert_series_equal(first.weights, second.weights)
    pd.testing.assert_series_equal(first.weights, third.weights)
    assert first.expected_return == pytest.approx(third.expected_return)
    pd.testing.assert_frame_equal(bl.cov_matrix, prior)  # the input is not overwritten
    pd.testing.assert_series_equal(
        first.extras["bl_prior_returns"], third.extras["bl_prior_returns"]
    )


def _levered_tangency_inputs():
    names = ["A", "B", "C", "D"]
    mu = pd.Series([0.10, 0.02, 0.08, -0.03], index=names)
    corr = np.array(
        [[1.0, 0.8, 0.2, 0.7], [0.8, 1.0, 0.1, 0.6], [0.2, 0.1, 1.0, 0.3], [0.7, 0.6, 0.3, 1.0]]
    )
    vol = np.array([0.20, 0.15, 0.25, 0.18])
    cov = pd.DataFrame(np.outer(vol, vol) * corr, index=names, columns=names)
    return mu, cov


def test_max_sharpe_honours_a_leverage_cap():
    from optimization_engine.optimizers.base import PortfolioConstraints
    from optimization_engine.optimizers.mean_variance import MaxSharpeOptimizer

    mu, cov = _levered_tangency_inputs()
    bounds = {a: (-1.0, 1.0) for a in cov.index}
    free = MaxSharpeOptimizer(
        expected_returns=mu,
        cov_matrix=cov,
        constraints=PortfolioConstraints(long_only=False, fully_invested=True, bounds=bounds),
    ).optimize()
    assert free.weights.abs().sum() > 1.3  # the unconstrained tangency wants leverage

    capped = MaxSharpeOptimizer(
        expected_returns=mu,
        cov_matrix=cov,
        constraints=PortfolioConstraints(
            long_only=False, fully_invested=True, bounds=bounds, leverage=1.2
        ),
    ).optimize()
    assert capped.weights.abs().sum() <= 1.2 + 1e-6
    assert capped.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert capped.extras["violations"] == []
    assert "ignored_constraints" not in capped.extras


def test_max_diversification_honours_a_leverage_cap():
    from optimization_engine.optimizers.base import PortfolioConstraints
    from optimization_engine.optimizers.max_diversification import MaxDiversificationOptimizer

    _, cov = _levered_tangency_inputs()
    bounds = {a: (-1.0, 1.0) for a in cov.index}
    capped = MaxDiversificationOptimizer(
        cov_matrix=cov,
        constraints=PortfolioConstraints(
            long_only=False, fully_invested=True, bounds=bounds, leverage=1.1
        ),
    ).optimize()
    assert capped.weights.abs().sum() <= 1.1 + 1e-6
    assert capped.extras["violations"] == []


@pytest.mark.parametrize("name", ["max_sharpe", "max_diversification"])
def test_ray_space_solves_report_an_open_budget_as_ignored(name):
    from optimization_engine.optimizers.base import PortfolioConstraints
    from optimization_engine.optimizers.max_diversification import MaxDiversificationOptimizer
    from optimization_engine.optimizers.mean_variance import MaxSharpeOptimizer

    mu, cov = _levered_tangency_inputs()
    constraints = PortfolioConstraints(
        long_only=True, fully_invested=False, bounds={a: (0.0, 1.0) for a in cov.index}
    )
    if name == "max_sharpe":
        optimizer = MaxSharpeOptimizer(expected_returns=mu, cov_matrix=cov, constraints=constraints)
    else:
        optimizer = MaxDiversificationOptimizer(cov_matrix=cov, constraints=constraints)
    with pytest.warns(UserWarning, match="fully_invested=False"):
        result = optimizer.optimize()
    assert "fully_invested" in result.extras["ignored_constraints"]
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Registry cross-check: the declared bounds mode is the delivered one
# ---------------------------------------------------------------------------

#: What a declared ``bounds_mode`` permits a solve to report. Only
#: ``"hard_or_projected"`` admits two answers, and it does so because the
#: method takes an exact bounded solve when it can and a projection when the
#: solver fails — the result says which, per solve.
_BOUNDS_MODE_ALLOWED = {
    "hard": {"hard"},
    "constrained": {"constrained"},
    "soft_iterated": {"soft_iterated"},
    "hard_or_projected": {"hard", "soft_iterated"},
}


@pytest.mark.parametrize("name", available_optimizers())
def test_result_bounds_mode_matches_its_requirements(returns: pd.DataFrame, name: str):
    """Every method's declared bounds mode must match what its result reports.

    The registry entry is what the CLI prints and the app captions; the
    result's ``extras["bounds_mode"]`` is what actually happened. A method
    that declares hard bounds and then projects is mislabelled, which is the
    bug this cross-check exists to catch.

    The mandate is deliberately binding — a 20% asset cap and a 20% budget on
    a three-name bucket that 1/N would otherwise overrun — so the projecting
    methods really do project.
    """
    from optimization_engine.benchmark import BenchmarkSpec
    from optimization_engine.optimizers.requirements import requirements_for

    assets = list(returns.columns)
    equities = assets[:3]
    cfg = EngineConfig(
        expected_returns=(
            (1 + returns).prod() ** (252 / len(returns)) - 1
        ).to_dict(),
        bounds={a: [0.0, 0.20] for a in assets},
        groups={a: ("Equity" if a in equities else "Other") for a in assets},
        group_bounds={"Equity": [0.0, 0.20]},
        # Only active_mean_variance needs one; the others ignore it because no
        # benchmark-relative budget is set.
        benchmark=BenchmarkSpec(kind="equal_weight"),
        optimizer=OptimizerSpec(name=name, risk_free_rate=0.02),
    )

    result = run_engine(returns, cfg).result

    declared = requirements_for(name).bounds_mode
    assert declared in _BOUNDS_MODE_ALLOWED, f"{name}: undeclared bounds mode {declared!r}"
    assert result.extras["bounds_mode"] in _BOUNDS_MODE_ALLOWED[declared], (
        f"{name} declares bounds_mode={declared!r} but reported "
        f"{result.extras['bounds_mode']!r}"
    )
    equity = sum(v for a, v in result.weights.items() if a in equities)
    assert equity <= 0.20 + 1e-6, f"{name}: the bucket budget did not bind"


# ---------------------------------------------------------------------------
# Registry cross-check: a hard-bounds solve owes a clean audit
# ---------------------------------------------------------------------------

#: Mandates for the audit cross-check, every one of them expressed purely in
#: weight terms — a box, the unit budget, bucket budgets on one or two layers.
#: That restriction is the test's whole validity. ``bounds_mode`` is a claim
#: about per-asset and group bounds and nothing else, while the audit checks
#: every limit the mandate carries, so a turnover or tracking-error budget here
#: would fail methods that are not lying about anything: ``max_sharpe``
#: declares hard bounds and *drops* a turnover limit, reporting it under
#: ``ignored_constraints``. That is a different promise, broken in a different
#: place, and it has its own tests.
def _audit_mandates(assets: list[str]) -> dict[str, dict]:
    """The weight-only mandates every method is cross-checked against."""
    from optimization_engine.constraints import ConstraintLayer

    equities = assets[:3]
    return {
        # The box and one bucket, both binding against 1/N.
        "box_and_bucket": dict(
            bounds={a: [0.0, 0.20] for a in assets},
            groups={a: ("Equity" if a in equities else "Other") for a in assets},
            group_bounds={"Equity": [0.0, 0.20]},
        ),
        # Floors as well as caps: the box binds from below, where a projection
        # that only clips has nothing to give.
        "floors_and_caps": dict(bounds={a: [0.03, 0.25] for a in assets}),
        # A second layer of policy, with both of its buckets fenced in.
        "layered_policy": dict(
            bounds={a: [0.0, 0.35] for a in assets},
            constraint_layers=[
                ConstraintLayer(
                    name="asset_class",
                    assignments={
                        a: ("Equity" if a in equities else "Bond") for a in assets
                    },
                    limits={"Equity": (0.10, 0.25), "Bond": (0.75, 0.90)},
                )
            ],
        ),
    }


@pytest.mark.parametrize("mandate", ["box_and_bucket", "floors_and_caps", "layered_policy"])
@pytest.mark.parametrize("name", available_optimizers())
def test_hard_bounds_solves_audit_clean(returns: pd.DataFrame, name: str, mandate: str):
    """A solve that reports ``bounds_mode="hard"`` must audit clean.

    The companion to ``test_result_bounds_mode_matches_its_requirements``: that
    one checks the *label* is the one the registry declares, this one checks the
    label is true. "Hard" means the box and the bucket budgets went into the
    convex program, so the post-solve audit has nothing to find — a hard method
    that comes back with a breach is either mislabelled or has stopped putting a
    constraint it claims into the program, and both are silent today.

    Keyed off the reported mode rather than the declared one, which is what
    brings ``max_diversification`` in: it declares ``"hard_or_projected"`` and
    is held to the hard standard exactly on the solves where it says it took the
    hard branch.
    """
    from optimization_engine.benchmark import BenchmarkSpec

    assets = list(returns.columns)
    cfg = EngineConfig(
        expected_returns=(
            (1 + returns).prod() ** (252 / len(returns)) - 1
        ).to_dict(),
        benchmark=BenchmarkSpec(kind="equal_weight"),
        optimizer=OptimizerSpec(name=name, risk_free_rate=0.02),
        **_audit_mandates(assets)[mandate],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_engine(returns, cfg).result

    assert result.audit is not None, f"{name}: the solve ran no audit"
    if result.extras["bounds_mode"] != "hard":
        pytest.skip(f"{name} reported {result.extras['bounds_mode']!r} on this mandate")
    assert result.audit.is_clean, (
        f"{name} reported hard bounds and breached the mandate:\n"
        f"{result.audit.describe()}"
    )
