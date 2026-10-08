"""The expected-return convention, and the single place it is defined.

Mean-variance optimization is a single-period model. It trades ``μ'w`` off
against ``λ·w'Σw`` over *one* period, so the ``μ`` it wants is the
expectation of one period's return — the arithmetic mean. The geometric mean
is the realized compound growth rate, lower by roughly ``σ²/2``, and it
answers a multi-period question instead. Pairing a geometric ``μ`` with an
arithmetic ``Σ`` is not conservatism; it measures reward and penalty on two
different conventions and the size of the discrepancy is exactly the quantity
being traded off.

These tests pin the convention, pin the geometric estimator that used to
occupy its name, and pin that neither formula is written out anywhere else.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.config import (
    EngineConfig,
    ExpectedReturnsMethod,
    expected_return_method_for_estimator,
)
from optimization_engine.data.covariance import (
    EXPECTED_RETURN_DESCRIPTIONS,
    covariance_matrix,
    expected_returns_from_history,
    james_stein_shrinkage,
)
from optimization_engine.data.loader import prices_to_returns, sample_dataset


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=252 * 4, seed=17))


# ---------------------------------------------------------------------------
# The convention
# ---------------------------------------------------------------------------


def test_mean_is_arithmetic(returns: pd.DataFrame):
    """``mean`` is ``r̄ · periods_per_year``, and nothing else."""
    mu = expected_returns_from_history(returns, method="mean", periods_per_year=252)
    expected = returns.mean() * 252
    pd.testing.assert_series_equal(mu, expected, check_names=False)
    assert list(mu.index) == list(returns.columns)


def test_geometric_mean_matches_old_formula(returns: pd.DataFrame):
    """The estimator that used to answer to ``mean`` still exists, named."""
    geometric = expected_returns_from_history(
        returns, method="geometric_mean", periods_per_year=252
    )
    old_formula = ((1 + returns).prod() ** (252 / len(returns))) - 1
    pd.testing.assert_series_equal(geometric, old_formula, check_names=False)


def test_arithmetic_exceeds_geometric_by_about_half_the_variance(
    returns: pd.DataFrame,
):
    """The gap is not a rounding difference — it is about ``σ²/2`` per asset.

    This is the whole reason the two cannot be used interchangeably: the
    error scales with variance, which is precisely the quantity the optimizer
    is penalizing, so swapping one convention for the other tilts the answer
    systematically toward the volatile names.

    The relation is a second-order approximation and it only reads cleanly
    where the variance term dominates. On a near-constant series (this
    panel's ``Cash``, at 0.5% annualized volatility) the *compounding* term
    ``≈ ppy(ppy−1)/2 · r̄²`` is the same size or larger and the geometric
    number ends up above the arithmetic one, so those assets are excluded
    rather than fudged into the tolerance.
    """
    arithmetic = expected_returns_from_history(returns, method="mean")
    geometric = expected_returns_from_history(returns, method="geometric_mean")
    gap = arithmetic - geometric
    half_variance = returns.var() * 252 / 2.0
    volatile = (returns.std() * np.sqrt(252)) > 0.05
    assert volatile.sum() >= 10
    assert (gap[volatile] > 0).all()
    # Second-order approximation, so allow it to be loose but not wrong.
    assert np.allclose(
        gap[volatile].values, half_variance[volatile].values, rtol=0.35, atol=2e-3
    )


def test_shrunk_mean_shrinks_the_arithmetic_mean(returns: pd.DataFrame):
    """Shrinkage pulls the arithmetic means in, not the geometric ones."""
    cov = covariance_matrix(returns, method="ledoit_wolf")
    shrunk = expected_returns_from_history(
        returns, method="shrunk_mean", cov_matrix=cov
    )
    arithmetic = expected_returns_from_history(returns, method="mean")
    geometric = expected_returns_from_history(returns, method="geometric_mean")
    # The shrunk vector is a convex combination of the arithmetic means and a
    # scalar target, so it must sit inside their range, not the geometric one.
    assert shrunk.min() >= arithmetic.min() - 1e-12
    assert shrunk.max() <= arithmetic.max() + 1e-12
    assert float(shrunk.mean()) > float(geometric.mean())


def test_capm_market_return_is_arithmetic(returns: pd.DataFrame):
    """The market premium is read as a single-period expectation too.

    It is multiplied by a beta and added to ``rf``; a geometric premium in
    that expression is the same inconsistency one level down.
    """
    cov = covariance_matrix(returns, method="ledoit_wolf")
    weights = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    implied = expected_returns_from_history(
        returns,
        method="capm",
        market_weights=weights,
        cov_matrix=cov,
        risk_free_rate=0.0,
    )
    market = (returns * weights).sum(axis=1)
    arithmetic_premium = float(market.mean() * 252)
    betas = (cov.values @ weights.values) / float(
        weights.values @ cov.values @ weights.values
    )
    pd.testing.assert_series_equal(
        implied,
        pd.Series(betas * arithmetic_premium, index=returns.columns),
        check_names=False,
    )


def test_ema_weights_the_window_it_was_given_not_its_first_observation():
    """``ema`` is a weighted *mean* of the window, with weights that sum to one.

    ``ewm(adjust=False)`` seeds the recursion with the first observation and
    leaves it whatever weight the decay has not yet handed to later rows:
    ``(1 − α)^(T−1)``, which on a 24-month window at the default span of 180
    is 0.774. One bad first month then sets the whole estimate — a −20% start
    to an otherwise steady +0.5% a month took μ to −86.7%. The normalized
    weights (``adjust=True``) give the same row ``(1 − α)^(T−1) / Σ(1 − α)^k``.
    """
    index = pd.date_range("2022-01-31", periods=24, freq="ME")
    monthly = pd.DataFrame({"A": [-0.20] + [0.005] * 23}, index=index)

    mu = expected_returns_from_history(
        monthly, method="ema", periods_per_year=12, span=180
    )

    alpha = 2.0 / (180 + 1.0)
    weights = (1.0 - alpha) ** np.arange(len(monthly))[::-1]
    by_hand = float((weights * monthly["A"].values).sum() / weights.sum()) * 12
    assert float(mu["A"]) == pytest.approx(by_hand, rel=1e-12)
    # The first month weighs about 1/24 of the window, not three quarters.
    assert weights[0] / weights.sum() < 0.06
    assert float(mu["A"]) > -0.10


def test_ema_is_annualized_like_the_mean_it_generalizes(returns: pd.DataFrame):
    """With no decay to speak of, ``ema`` is the arithmetic ``mean``.

    Both are single-period expectations, so both annualize as ``r̄ · ppy``.
    ``ema`` compounded instead — ``(1 + r̄)^ppy − 1`` — which on its own is a
    different convention from ``mean`` and, combined with the seeding above,
    put a span of ten million about 90 percentage points away from it.
    """
    mean = expected_returns_from_history(returns, method="mean")
    flat = expected_returns_from_history(returns, method="ema", span=10**9)
    pd.testing.assert_series_equal(flat, mean, check_names=False, rtol=0, atol=1e-6)


def test_geometric_mean_compounds_each_asset_over_its_own_history():
    """An asset that listed late is annualized over the periods it has.

    ``prod`` skips the missing returns while ``len`` counted them, so on a
    panel where Gold lists halfway through, its two years of compound growth
    were spread over all four: 4.6% instead of 9.45%.
    """
    prices = sample_dataset(n_periods=252 * 4, seed=3)[["US_Equity", "Gold"]].copy()
    prices.iloc[: 252 * 2, 1] = np.nan
    ragged = prices_to_returns(prices)

    on_the_panel = expected_returns_from_history(ragged, method="geometric_mean")
    on_its_own = expected_returns_from_history(
        ragged[["Gold"]].dropna(), method="geometric_mean"
    )
    assert float(on_the_panel["Gold"]) == pytest.approx(float(on_its_own["Gold"]))
    full = expected_returns_from_history(ragged[["US_Equity"]], method="geometric_mean")
    assert float(on_the_panel["US_Equity"]) == pytest.approx(float(full["US_Equity"]))


def test_unknown_method_names_every_available_one(returns: pd.DataFrame):
    with pytest.raises(ValueError, match="Unknown expected-return method"):
        expected_returns_from_history(returns, method="arithmetic")
    assert "geometric_mean" in EXPECTED_RETURN_DESCRIPTIONS
    assert set(EXPECTED_RETURN_DESCRIPTIONS) == {
        "mean", "geometric_mean", "ema", "capm", "shrunk_mean",
    }


# ---------------------------------------------------------------------------
# The config's own vocabulary
# ---------------------------------------------------------------------------


def test_config_vocabulary_translates_to_estimator_names():
    assert expected_return_method_for_estimator("historical_mean") == "mean"
    assert expected_return_method_for_estimator("geometric_mean") == "geometric_mean"
    for name in ("ema", "capm", "shrunk_mean"):
        assert expected_return_method_for_estimator(name) == name


def test_every_config_method_resolves_to_a_real_estimator(returns: pd.DataFrame):
    """A name the config accepts must be a name the estimator accepts."""
    import typing

    # Read the vocabulary from its own name, not from the class's
    # annotations: ``get_type_hints`` evaluates *every* annotation on
    # ``EngineConfig``, and several are PEP 604 unions, which are a runtime
    # ``TypeError`` on Python 3.9 even under ``from __future__ import
    # annotations``. The library never evaluates them, so only this test
    # ever hit it.
    allowed = typing.get_args(ExpectedReturnsMethod)
    assert "geometric_mean" in allowed
    cov = covariance_matrix(returns, method="ledoit_wolf")
    for name in allowed:
        mu = expected_returns_from_history(
            returns,
            method=expected_return_method_for_estimator(name),
            cov_matrix=cov,
        )
        assert list(mu.index) == list(returns.columns)


def test_config_round_trips_the_new_method():
    config = EngineConfig(expected_returns_method="geometric_mean")
    restored = EngineConfig.from_dict(config.to_dict())
    assert restored.expected_returns_method == "geometric_mean"


# ---------------------------------------------------------------------------
# Bayes-Stein: an unshrunk vector reports zero shrinkage
# ---------------------------------------------------------------------------


def test_bayes_stein_degenerate_reports_zero_intensity():
    """Nothing was shrunk, so the reported intensity has to be zero.

    ``λ = 1`` is the analytic limit of Jorion's formula as the quadratic
    form goes to zero, but the intensity is not being asked what the formula
    tends to — it is being asked what the estimator did to the vector it
    returned. It returned the sample means untouched. Reporting full
    shrinkage beside them is a statement about the output that is false, and
    the ``LinAlgError`` exit already reports ``0.0`` for the same reason.
    """
    assets = [f"a{i}" for i in range(5)]
    # Every mean identical ⇒ the deviation from the Jorion target is exactly
    # zero ⇒ the quadratic form is zero and no shrinkage is possible.
    mu = pd.Series(0.07, index=assets)
    rng = np.random.default_rng(0)
    data = rng.normal(size=(600, len(assets)))
    cov = pd.DataFrame(np.cov(data, rowvar=False) * 0.04, index=assets, columns=assets)

    shrunk, intensity = james_stein_shrinkage(mu, cov, n_observations=600)

    assert intensity == 0.0
    pd.testing.assert_series_equal(shrunk, mu)


def test_bayes_stein_does_not_depend_on_an_exact_cancellation():
    """A deviation at the level of floating-point residue is not a deviation.

    This is the case that actually broke: with identical means the Jorion
    target is that same mean recomputed through ``pinv``, so the deviation
    comes back as exactly zero on one BLAS and a few ulps on another. Both are
    the estimator doing the same nothing. Testing the quadratic form for
    ``<= 0`` alone caught only the first, and the second returned an intensity
    of 1.0 beside an unshrunk vector — the exact claim the zero exists to
    prevent, reached through a different door and invisible on whichever
    interpreter happened to cancel exactly.
    """
    assets = [f"a{i}" for i in range(5)]
    rng = np.random.default_rng(0)
    data = rng.normal(size=(600, len(assets)))
    cov = pd.DataFrame(np.cov(data, rowvar=False) * 0.04, index=assets, columns=assets)
    # Means that differ only at the last bit, as a recomputed target does.
    mu = pd.Series([0.07, 0.07 + 1e-17, 0.07 - 1e-17, 0.07, 0.07], index=assets)

    shrunk, intensity = james_stein_shrinkage(mu, cov, n_observations=600)

    assert intensity == 0.0
    pd.testing.assert_series_equal(shrunk, mu)


def test_bayes_stein_still_shrinks_a_real_deviation():
    """The residue guard must not swallow shrinkage that should happen."""
    assets = [f"a{i}" for i in range(5)]
    rng = np.random.default_rng(0)
    data = rng.normal(size=(600, len(assets)))
    cov = pd.DataFrame(np.cov(data, rowvar=False) * 0.04, index=assets, columns=assets)
    mu = pd.Series([0.02, 0.05, 0.09, 0.13, 0.20], index=assets)

    shrunk, intensity = james_stein_shrinkage(mu, cov, n_observations=600)

    assert 0.0 < intensity <= 1.0
    assert not np.allclose(shrunk.values, mu.values)
    # A convex combination, so it stays inside the sample range.
    assert mu.min() - 1e-12 <= shrunk.min() and shrunk.max() <= mu.max() + 1e-12


def test_bayes_stein_reports_a_real_intensity_when_it_shrinks():
    """The zero above must mean "nothing happened", not "always zero"."""
    assets = [f"a{i}" for i in range(5)]
    mu = pd.Series([0.02, 0.05, 0.09, 0.14, 0.20], index=assets)
    rng = np.random.default_rng(1)
    data = rng.normal(size=(600, len(assets)))
    cov = pd.DataFrame(np.cov(data, rowvar=False) * 0.04, index=assets, columns=assets)

    shrunk, intensity = james_stein_shrinkage(mu, cov, n_observations=600)

    assert 0.0 < intensity <= 1.0
    assert float(shrunk.std()) < float(mu.std())


# ---------------------------------------------------------------------------
# One definition site
# ---------------------------------------------------------------------------

#: Trees to scan: the library, the app, and the scripts that ship with it.
#: ``tests`` is deliberately outside it — ``test_geometric_mean_matches_old_
#: formula`` writes the formula out on purpose, as the pin.
_SCANNED = ("src", "app", "scripts")

#: Names that mark an exponent as an *annualization*, as opposed to the
#: ``1/n`` exponent of a plain geometric mean over a window.
_ANNUALIZERS = ("periods_per_year", "ppy", "252", "12", "52")


def _annualized_compounding_sites() -> list[tuple[str, int, str]]:
    """Every ``(...).prod() ** <annualizing exponent>`` in the scanned trees.

    Returns:
        ``(path, lineno, enclosing function)`` for each occurrence.
    """
    found: list[tuple[str, int, str]] = []
    for tree_name in _SCANNED:
        for path in sorted((ROOT / tree_name).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            module = ast.parse(source)
            enclosing: dict[int, str] = {}
            for node in ast.walk(module):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for inner in ast.walk(node):
                        enclosing.setdefault(getattr(inner, "lineno", -1), node.name)
            for node in ast.walk(module):
                if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)):
                    continue
                has_prod = any(
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "prod"
                    for sub in ast.walk(node.left)
                )
                if not has_prod:
                    continue
                exponent = ast.unparse(node.right)
                if not any(token in exponent for token in _ANNUALIZERS):
                    continue
                found.append(
                    (
                        # POSIX form, or this never matches on Windows.
                        path.relative_to(ROOT).as_posix(),
                        node.lineno,
                        enclosing.get(node.lineno, "<module>"),
                    )
                )
    return found


def test_no_other_mu_implementation():
    """The annualized compound-return formula lives in exactly one place.

    It used to be written out in six: twice inside ``covariance.py`` itself,
    twice in the Streamlit app, once in the doc-image script, and once as
    the ``mean`` branch. Six copies of a convention is six chances for them
    to disagree, and they did — the app's config table seeded a geometric μ
    into a solve the engine ran against a different one.

    ``analytics.performance.annualize_returns`` is deliberately *not* an
    exception here: it computes the compounding in two statements
    (``compounded = (1 + r).prod()`` then ``compounded ** ...``), so it does
    not match this shape, and it is the sanctioned single source for the
    *realized* annualized return, which is a different quantity from μ.
    """
    sites = _annualized_compounding_sites()
    where = [(path, function) for path, _, function in sites]
    assert where == [
        (
            "src/optimization_engine/data/covariance.py",
            "expected_returns_from_history",
        )
    ], f"annualized compounding written out in more than one place: {sites}"


def test_the_realized_annualizer_is_still_the_one_for_realized_returns():
    """The app's performance tiles must not grow a private copy either."""
    from optimization_engine.analytics.performance import annualize_returns

    series = pd.Series(np.full(504, 0.0004))
    assert float(annualize_returns(series, periods_per_year=252)) == pytest.approx(
        (1.0004**504) ** (252 / 504) - 1
    )
    app_source = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert ".prod()" not in app_source, (
        "the app compounds a return series itself again; route it through "
        "annualize_returns (realized) or expected_returns_from_history (μ)"
    )
    assert "annualize_returns(" in app_source


@pytest.mark.parametrize("method", ["cvar", "cdar"])
def test_tail_methods_compare_a_target_with_the_arithmetic_mean(method: str):
    """CVaR and CDaR annualized the history as ``(1+m)^ppy − 1`` (review §2.5).

    With no expected returns supplied they checked the return floor against
    the compounded mean, which runs above the arithmetic one the rest of the
    package uses. A 56.6% target, above the 49.4% best arithmetic mean, then
    "solved" with a book whose arithmetic return was 45.8%. The floor now uses
    ``expected_returns_from_history("mean")``, so the same target is out of
    reach and the solve says so.
    """
    from optimization_engine.optimizers._cvxpy_helpers import SolverFailure
    from optimization_engine.optimizers.cdar import CDaROptimizer
    from optimization_engine.optimizers.cvar import CVaROptimizer

    rng = np.random.default_rng(11)
    n_obs = 400
    a = rng.standard_normal(n_obs) * 0.006
    a[rng.choice(n_obs, 12, replace=False)] -= 0.05
    history = pd.DataFrame(
        {
            "A": a,
            "B": rng.standard_normal(n_obs) * 0.012 + 0.0004,
            "C": np.abs(rng.standard_normal(n_obs)) * 0.01 - 0.006,
        },
        index=pd.bdate_range("2020-01-01", periods=n_obs),
    )
    arithmetic = expected_returns_from_history(history, method="mean")
    compounded = (1 + history.mean()) ** 252 - 1
    target = float(arithmetic.max()) + 0.5 * float(compounded.max() - arithmetic.max())
    assert arithmetic.max() < target < compounded.max(), "fixture drifted"

    cls = CVaROptimizer if method == "cvar" else CDaROptimizer
    optimizer = cls(returns=history, target_return=target)
    with pytest.warns(UserWarning, match="historical means"):
        np.testing.assert_allclose(optimizer._target_mu_vector(), arithmetic.values)
    with pytest.warns(UserWarning):
        with pytest.raises(SolverFailure) as raised:
            optimizer.optimize()
    assert raised.value.status == "infeasible"
