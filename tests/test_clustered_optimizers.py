"""NCO, HERC and mean-CDaR.

The clustering methods are tested on a panel with a *known* block structure,
so "did it find the clusters" is a question with a right answer rather than a
judgement call.
"""

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
from optimization_engine.optimizers._clustering import (
    build_linkage,
    correlation_distance,
    correlation_from_covariance,
    optimal_clusters,
)
from optimization_engine.optimizers.base import PortfolioConstraints
from optimization_engine.optimizers.cdar import CDaROptimizer
from optimization_engine.optimizers.herc import HERCOptimizer
from optimization_engine.optimizers.nco import NCOOptimizer


@pytest.fixture(scope="module")
def blocked_returns() -> pd.DataFrame:
    """Three tight blocks of five assets each, with a weak market factor."""
    rng = np.random.default_rng(42)
    n_obs = 1500
    market = rng.normal(0, 0.004, size=n_obs)
    columns: dict[str, np.ndarray] = {}
    for block in range(3):
        driver = rng.normal(0, 0.010, size=n_obs)
        for member in range(5):
            columns[f"b{block}_a{member}"] = (
                market + driver + rng.normal(0, 0.004, size=n_obs)
            )
    return pd.DataFrame(columns)


@pytest.fixture(scope="module")
def blocked_cov(blocked_returns: pd.DataFrame) -> pd.DataFrame:
    return covariance_matrix(blocked_returns, method="sample")


@pytest.fixture(scope="module")
def sample_returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=252 * 4, seed=3))


# ---------------------------------------------------------------------------
# Shared clustering
# ---------------------------------------------------------------------------


def test_cluster_search_recovers_a_known_block_structure(blocked_cov: pd.DataFrame):
    corr = correlation_from_covariance(blocked_cov)
    link = build_linkage(correlation_distance(corr), method="ward")
    assignment = optimal_clusters(corr, link)

    assert assignment.n_clusters == 3
    for members in assignment.members.values():
        prefixes = {name.split("_")[0] for name in members}
        assert len(prefixes) == 1, f"cluster mixes blocks: {members}"
    assert assignment.silhouette > 0.3


def test_cluster_count_can_be_forced(blocked_cov: pd.DataFrame):
    corr = correlation_from_covariance(blocked_cov)
    link = build_linkage(correlation_distance(corr), method="ward")
    assert optimal_clusters(corr, link, n_clusters=5).n_clusters == 5


def test_correlation_distance_is_a_metric(blocked_cov: pd.DataFrame):
    distance = correlation_distance(correlation_from_covariance(blocked_cov))
    assert np.allclose(np.diag(distance), 0.0)
    assert np.allclose(distance, distance.T)
    # Triangle inequality on a random sample of triples.
    rng = np.random.default_rng(0)
    n = distance.shape[0]
    for _ in range(200):
        i, j, k = rng.integers(0, n, size=3)
        assert distance[i, j] <= distance[i, k] + distance[k, j] + 1e-9


def test_zero_variance_asset_is_named_rather_than_producing_nans():
    cov = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.0]], index=["good", "dead"], columns=["good", "dead"]
    )
    with pytest.raises(ValueError, match="dead"):
        correlation_from_covariance(cov)


# ---------------------------------------------------------------------------
# NCO
# ---------------------------------------------------------------------------


def test_nco_inverts_only_small_or_well_conditioned_matrices(
    blocked_cov: pd.DataFrame,
):
    result = NCOOptimizer(
        cov_matrix=blocked_cov, constraints=PortfolioConstraints()
    ).optimize()

    assert result.weights.sum() == pytest.approx(1.0)
    assert (result.weights >= -1e-8).all()
    assert result.extras["nco_n_clusters"] == 3
    assert (
        result.extras["nco_condition_worst_cluster"]
        < result.extras["nco_condition_direct"]
    )


def test_nco_weights_are_more_stable_than_a_direct_solve(
    blocked_returns: pd.DataFrame,
):
    """The claim NCO exists to make, tested rather than asserted.

    Estimate on two disjoint halves of the same generated history and compare
    how far each method's answer moves. The data-generating process is
    identical across halves, so any movement is estimation error.
    """
    from optimization_engine.optimizers.mean_variance import MinVarianceOptimizer

    first = blocked_returns.iloc[: len(blocked_returns) // 2]
    second = blocked_returns.iloc[len(blocked_returns) // 2 :]
    constraints = PortfolioConstraints()

    def solve(cls, sample, **kwargs):
        cov = covariance_matrix(sample, method="sample")
        return cls(cov_matrix=cov, constraints=constraints, **kwargs).optimize().weights

    direct_gap = float(
        np.abs(
            solve(MinVarianceOptimizer, first) - solve(MinVarianceOptimizer, second)
        ).sum()
    )
    nested_gap = float(
        np.abs(solve(NCOOptimizer, first) - solve(NCOOptimizer, second)).sum()
    )
    assert nested_gap < direct_gap


def test_nco_max_sharpe_requires_expected_returns(blocked_cov: pd.DataFrame):
    optimizer = NCOOptimizer(
        cov_matrix=blocked_cov,
        constraints=PortfolioConstraints(),
        objective="max_sharpe",
    )
    with pytest.raises(ValueError, match="needs expected returns"):
        optimizer.optimize()


def test_nco_rejects_an_unknown_objective(blocked_cov: pd.DataFrame):
    with pytest.raises(ValueError, match="Unknown NCO objective"):
        NCOOptimizer(cov_matrix=blocked_cov, objective="maximum_hope")


def test_nco_needs_a_universe_worth_clustering():
    cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.02]], index=["a", "b"], columns=["a", "b"]
    )
    with pytest.raises(ValueError, match="at least 3 assets"):
        NCOOptimizer(cov_matrix=cov, constraints=PortfolioConstraints()).optimize()


def test_nco_honours_weight_bounds_by_projection(blocked_cov: pd.DataFrame):
    constraints = PortfolioConstraints(
        bounds={a: (0.0, 0.10) for a in blocked_cov.columns}
    )
    result = NCOOptimizer(cov_matrix=blocked_cov, constraints=constraints).optimize()
    assert result.weights.max() <= 0.10 + 1e-6
    assert result.is_compliant
    assert "projection_distance" in result.extras


@pytest.mark.parametrize("objective", ["min_variance", "max_sharpe"])
def test_nco_layers_go_through_optimize(objective: str, blocked_cov: pd.DataFrame):
    """Both nested layers are solved by ``optimize()``, not by a bare ``_solve``.

    NCO used to reach past the public entry point straight into the
    sub-optimizer's ``_solve``, which meant the intra- and inter-cluster
    weights were the only ones in the engine that nothing cleaned, bounded or
    checked — no dust removal, no ``bounds_mode``, no solver record. The spy
    below fires only if the sub-solves go through ``optimize``, so the test
    fails outright on a regression rather than on a weakened assertion.

    The one thing a nested layer deliberately does *not* get is the post-solve
    mandate audit: see ``test_nco_sub_solves_are_not_audited_against_the_mandate``
    for why, and for what the opt-out is worth in cost.
    """
    from optimization_engine.optimizers.mean_variance import (
        MaxSharpeOptimizer,
        MinVarianceOptimizer,
    )

    sub_optimizer = (
        MinVarianceOptimizer if objective == "min_variance" else MaxSharpeOptimizer
    )
    assert "optimize" not in sub_optimizer.__dict__, (
        "this test restores the method by deleting it from the subclass"
    )
    mu = pd.Series(
        np.linspace(0.03, 0.12, len(blocked_cov.columns)), index=blocked_cov.columns
    )
    sub_results = []
    sub_kwargs = []
    unpatched = sub_optimizer.optimize

    def spy(self, **kwargs):
        result = unpatched(self, **kwargs)
        sub_results.append(result)
        sub_kwargs.append(kwargs)
        return result

    sub_optimizer.optimize = spy
    try:
        result = NCOOptimizer(
            cov_matrix=blocked_cov,
            expected_returns=mu,
            constraints=PortfolioConstraints(),
            objective=objective,
        ).optimize()
    finally:
        del sub_optimizer.optimize

    # One solve per cluster, plus the one across the synthetic cluster assets.
    n_clusters = result.extras["nco_n_clusters"]
    assert len(sub_results) == n_clusters + 1

    for sub in sub_results:
        assert isinstance(sub.extras["solver"], str) and sub.extras["solver"]
        assert sub.extras["solver_status"] == "optimal"
        assert sub.extras["solvers_attempted"]
        assert sub.extras["bounds_mode"] == "hard"
        assert sub.extras["optimizer"] == sub_optimizer.name
        assert sub.weights.sum() == pytest.approx(1.0)

    # Every layer asked `optimize` to skip the post-solve pass, explicitly.
    assert sub_kwargs == [{"run_post_solve_diagnostics": False}] * len(sub_results)

    # The last sub-solve is the inter-cluster layer: one weight per cluster.
    assert list(sub_results[-1].weights.index) == sorted(
        int(label) for label in result.extras["nco_cluster_weights"]
    )


def test_nco_sub_solves_are_not_audited_against_the_mandate(blocked_cov: pd.DataFrame):
    """A cluster weight is not a book weight, so the mandate cannot judge it.

    NCO solves each cluster against budget-and-sign only and applies the
    mandate's per-asset limits to the *combined* book by projection. Auditing a
    sub-solve against the mandate would therefore measure the wrong quantity —
    an intra-cluster weight of 0.30 is 30% of a cluster, not 30% of the
    portfolio, and a 10% cap would be "breached" by every layer of every NCO
    run. Auditing it against the budget-and-sign set it really was solved with
    is not wrong, only pointless: it re-checks, k+1 times, a constraint the
    convex program has already imposed.

    So the layers opt out, and the assertion that proves it is not that the
    sub-reports are clean — it is that the raw intra-cluster weights would
    *not* have been, had the mandate been the yardstick.
    """
    from optimization_engine.optimizers.mean_variance import MinVarianceOptimizer

    cap = 0.10
    constraints = PortfolioConstraints(
        bounds={a: (0.0, cap) for a in blocked_cov.columns}
    )
    sub_results = []
    unpatched = MinVarianceOptimizer.optimize

    def spy(self, **kwargs):
        result = unpatched(self, **kwargs)
        sub_results.append((self.constraints, result))
        return result

    MinVarianceOptimizer.optimize = spy
    try:
        result = NCOOptimizer(cov_matrix=blocked_cov, constraints=constraints).optimize()
    finally:
        del MinVarianceOptimizer.optimize

    assert sub_results, "no sub-solve was observed"
    for sub_constraints, sub in sub_results:
        # The sub-problem never sees the mandate's box in the first place.
        assert sub_constraints.bounds == {}
        assert sub.audit is None
        assert "diagnostics" not in sub.extras
        assert "violations" not in sub.extras

    # The heart of it: at least one layer breaks the mandate's cap in cluster
    # units, so an audit against `constraints` would have reported a breach
    # that does not exist in the book.
    assert any(
        sub.weights.max() > cap + 1e-6 for _constraints, sub in sub_results
    ), "expected a cluster weight above the book-level cap"

    # The book itself is audited, and complies.
    assert result.audit is not None
    assert result.audit.is_clean, result.audit.describe()
    assert result.weights.max() <= cap + 1e-6


def test_nco_survives_a_long_short_mandate(blocked_cov: pd.DataFrame):
    """``long_only=False`` is the path where the unit-budget invariant can break.

    Every cluster's weights are rescaled to sum to one before the outer
    product, so ``loadings @ inter`` sums to one. ``_clean_weights`` normally
    does that rescaling, but it declines to when a layer nets to ~0 — which
    only a long-short book can do. The nest has to hold the budget anyway.
    """
    constraints = PortfolioConstraints(
        long_only=False, bounds={a: (-0.30, 0.30) for a in blocked_cov.columns}
    )
    result = NCOOptimizer(cov_matrix=blocked_cov, constraints=constraints).optimize()

    assert result.weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert result.weights.min() < 0.0, "expected a genuinely long-short book"
    assert result.weights.min() >= -0.30 - 1e-6
    assert result.weights.max() <= 0.30 + 1e-6
    assert result.is_compliant, result.violations


def _two_block_cov() -> pd.DataFrame:
    """Two tight pairs whose unconstrained intra-cluster books lever past ±1."""
    names = ["p", "q", "r", "s"]
    corr = np.array(
        [
            [1.0, 0.95, 0.1, 0.1],
            [0.95, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.9],
            [0.1, 0.1, 0.9, 1.0],
        ]
    )
    vols = np.array([0.10, 0.20, 0.12, 0.30])
    return pd.DataFrame(np.outer(vols, vols) * corr, index=names, columns=names)


def _mlam_nco(cov: pd.DataFrame, clusters: list[list[str]], mu=None) -> np.ndarray:
    """López de Prado's NCO, *MLAM* snippet 7.6, with its closed-form ``optPort``."""

    def opt_port(sigma: np.ndarray, m: np.ndarray | None) -> np.ndarray:
        raw = np.linalg.inv(sigma) @ (np.ones(len(sigma)) if m is None else m)
        return raw / raw.sum()

    loadings = pd.DataFrame(0.0, index=cov.index, columns=range(len(clusters)))
    for k, members in enumerate(clusters):
        sub_mu = None if mu is None else mu.loc[members].values
        loadings.loc[members, k] = opt_port(cov.loc[members, members].values, sub_mu)
    reduced = loadings.T.values @ cov.values @ loadings.values
    reduced_mu = None if mu is None else loadings.T.values @ mu.values
    return loadings.values @ opt_port(reduced, reduced_mu)


@pytest.mark.parametrize("objective", ["min_variance", "max_sharpe"])
def test_nco_long_short_layers_carry_no_hidden_box(objective: str):
    """A long-short NCO is the reference NCO when the mandate does not bind.

    The sub-problems were built with no bounds, which ``get_bounds`` reads as
    the long-short default of (−1, 1): a box the mandate never set. Inside a
    tight pair the unconstrained minimum-variance book is 1.75 long and 0.75
    short, so under a (−10, 10) mandate the engine returned [0.6, 0, 0.4, 0]
    against the snippet's [1.155, −0.495, 0.495, −0.155] — with no violation
    and a clean audit, because the clipping happened where nothing audits.
    """
    cov = _two_block_cov()
    mu = pd.Series([0.05, 0.06, 0.04, 0.07], index=cov.index)
    constraints = PortfolioConstraints(
        long_only=False, bounds={a: (-10.0, 10.0) for a in cov.index}
    )
    result = NCOOptimizer(
        cov_matrix=cov,
        expected_returns=mu if objective == "max_sharpe" else None,
        constraints=constraints,
        objective=objective,
        n_clusters=2,
        detone_for_clustering=False,
    ).optimize()

    clusters = [["p", "q"], ["r", "s"]]
    assert sorted(sorted(m) for m in result.extras["nco_clusters"].values()) == clusters
    reference = _mlam_nco(cov, clusters, mu if objective == "max_sharpe" else None)

    np.testing.assert_allclose(result.weights.values, reference, atol=1e-6)
    assert np.abs(result.weights.values).max() > 1.0, "the reference levers past 1"
    assert result.extras["projection_distance"] < 1e-6
    assert result.is_compliant, result.violations


def test_nco_max_sharpe_survives_a_cluster_that_trails_cash():
    """One cluster entirely below the risk-free rate is not a failed solve.

    A bond pair at 1% and 1.5% against a 2% rate made the intra-cluster
    max-Sharpe raise "Every expected return is at or below the risk-free
    rate", and NCO with it — while plain max-Sharpe on the same inputs answers
    [0.61, 0.39, 0, 0]. Inside such a cluster there is nothing to maximize, so
    it is solved for minimum variance, and the inter-cluster layer decides how
    much of it to hold. The substitution is recorded, not silent.
    """
    names = ["e1", "e2", "b1", "b2"]
    corr = np.array(
        [[1, 0.8, 0, 0], [0.8, 1, 0, 0], [0, 0, 1, 0.8], [0, 0, 0.8, 1]], dtype=float
    )
    vols = np.array([0.15, 0.18, 0.05, 0.06])
    cov = pd.DataFrame(np.outer(vols, vols) * corr, index=names, columns=names)
    mu = pd.Series([0.08, 0.09, 0.01, 0.015], index=names)

    with pytest.warns(UserWarning, match="minimum variance"):
        result = NCOOptimizer(
            cov_matrix=cov,
            expected_returns=mu,
            objective="max_sharpe",
            risk_free_rate=0.02,
            n_clusters=2,
            detone_for_clustering=False,
        ).optimize()

    clusters = result.extras["nco_clusters"]
    bonds = next(label for label, members in clusters.items() if "b1" in members)
    fallback = result.extras["nco_min_variance_fallback"]
    assert list(fallback) == [bonds]
    assert "risk-free" in fallback[bonds]
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert result.weights[["e1", "e2"]].sum() > 0.5


def test_nco_max_sharpe_falls_back_where_a_cluster_has_no_finite_tangency():
    """Long-short, a pair with ``1'Σ⁻¹(μ − rf) < 0`` has a supremum, not a maximum.

    Once the hidden (−1, 1) box was removed from the sub-problems, that
    cluster's max-Sharpe raises ``unbounded``. Taking the cluster's minimum
    variance book keeps NCO answering, and says which cluster it did it for.
    The second pair is quiet enough that the layer across clusters still has
    a finite tangency.
    """
    names = ["p", "q", "r", "s"]
    corr = np.array(
        [
            [1.0, 0.95, 0.1, 0.1],
            [0.95, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.5],
            [0.1, 0.1, 0.5, 1.0],
        ]
    )
    vols = np.array([0.10, 0.20, 0.03, 0.04])
    cov = pd.DataFrame(np.outer(vols, vols) * corr, index=names, columns=names)
    mu = pd.Series([0.03, 0.08, 0.04, 0.05], index=names)
    constraints = PortfolioConstraints(
        long_only=False, bounds={a: (-10.0, 10.0) for a in names}
    )
    with pytest.warns(UserWarning, match="minimum variance"):
        result = NCOOptimizer(
            cov_matrix=cov,
            expected_returns=mu,
            constraints=constraints,
            objective="max_sharpe",
            n_clusters=2,
            detone_for_clustering=False,
        ).optimize()

    fallback = result.extras["nco_min_variance_fallback"]
    pair = next(
        label for label, members in result.extras["nco_clusters"].items() if "p" in members
    )
    assert list(fallback) == [pair]
    assert "unbounded" in fallback[pair]
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert result.is_compliant, result.violations


def test_nco_max_sharpe_names_an_unbounded_inter_cluster_layer():
    """Across clusters there is no fallback, and the advice has to fit NCO.

    The generic "add per-asset bounds" would send the reader to a box NCO
    applies only to the combined book, never inside a layer.
    """
    from optimization_engine.optimizers._cvxpy_helpers import SolverFailure

    cov = _two_block_cov()
    mu = pd.Series([0.01, 0.05, 0.04, 0.07], index=cov.index)
    constraints = PortfolioConstraints(
        long_only=False, bounds={a: (-10.0, 10.0) for a in cov.index}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(SolverFailure) as raised:
            NCOOptimizer(
                cov_matrix=cov,
                expected_returns=mu,
                constraints=constraints,
                objective="max_sharpe",
                n_clusters=2,
                detone_for_clustering=False,
            ).optimize()

    assert raised.value.status == "unbounded"
    assert "objective='min_variance'" in str(raised.value)


def test_nco_refuses_a_layer_whose_weights_net_to_zero(blocked_cov: pd.DataFrame):
    """A cancelled-out cluster is named, not quietly folded into the book.

    ``_clean_weights`` returns the weights *unnormalized* when they net to
    approximately zero, because rescaling is meaningless there. Passing that
    through would leave the cluster with no share of the book to allocate and
    the combined weights no longer summing to one — a silently un-invested
    portfolio. The sub-solve is forced into that state here, since a real
    covariance matrix reaches it only by accident.
    """
    from optimization_engine.optimizers.mean_variance import MinVarianceOptimizer

    def nets_to_zero(self):
        weights = np.ones(len(self.assets))
        weights[0] = -(len(weights) - 1.0)
        return weights

    unpatched = MinVarianceOptimizer._solve
    MinVarianceOptimizer._solve = nets_to_zero
    try:
        with pytest.raises(ValueError, match="net to") as excinfo:
            NCOOptimizer(
                cov_matrix=blocked_cov,
                constraints=PortfolioConstraints(long_only=False),
            ).optimize()
    finally:
        MinVarianceOptimizer._solve = unpatched

    assert "cluster" in str(excinfo.value)


# ---------------------------------------------------------------------------
# HERC
# ---------------------------------------------------------------------------


def test_herc_splits_the_budget_across_the_clusters_it_found(
    blocked_cov: pd.DataFrame,
):
    result = HERCOptimizer(
        cov_matrix=blocked_cov, constraints=PortfolioConstraints()
    ).optimize()

    assert result.weights.sum() == pytest.approx(1.0)
    assert (result.weights > 0).all()
    assert result.extras["herc_n_clusters"] == 3
    cluster_weights = pd.Series(result.extras["herc_cluster_weights"])
    assert cluster_weights.sum() == pytest.approx(1.0)
    # Three statistically identical blocks should get comparable budgets.
    assert cluster_weights.max() / cluster_weights.min() < 2.0


def test_herc_downside_measures_need_a_return_history(blocked_cov: pd.DataFrame):
    with pytest.raises(ValueError, match="needs a `returns` frame"):
        HERCOptimizer(cov_matrix=blocked_cov, risk_measure="cvar")


@pytest.mark.parametrize("measure", ["variance", "std", "cvar", "cdar", "equal_weight"])
def test_herc_supports_every_advertised_risk_measure(
    measure: str, blocked_cov: pd.DataFrame, blocked_returns: pd.DataFrame
):
    result = HERCOptimizer(
        cov_matrix=blocked_cov,
        constraints=PortfolioConstraints(),
        risk_measure=measure,
        returns=blocked_returns,
    ).optimize()
    assert result.weights.sum() == pytest.approx(1.0)
    assert (result.weights >= 0).all()


def test_herc_rejects_an_unknown_risk_measure(blocked_cov: pd.DataFrame):
    with pytest.raises(ValueError, match="Unknown HERC risk measure"):
        HERCOptimizer(cov_matrix=blocked_cov, risk_measure="vibes")


def test_herc_refuses_a_negative_minimum_weight(blocked_cov: pd.DataFrame):
    constraints = PortfolioConstraints(
        bounds={a: (-0.1, 0.5) for a in blocked_cov.columns}, long_only=False
    )
    with pytest.raises(ValueError, match="long-only"):
        HERCOptimizer(cov_matrix=blocked_cov, constraints=constraints).optimize()


def test_herc_differs_from_hrp_on_a_blocked_panel(blocked_cov: pd.DataFrame):
    """The two agree only by coincidence; the tree splits differ by design."""
    from optimization_engine.optimizers.hrp import HRPOptimizer

    constraints = PortfolioConstraints()
    herc = HERCOptimizer(cov_matrix=blocked_cov, constraints=constraints).optimize()
    hrp = HRPOptimizer(cov_matrix=blocked_cov, constraints=constraints).optimize()
    assert float(np.abs(herc.weights - hrp.weights).sum()) > 1e-3


# ---------------------------------------------------------------------------
# Mean-CDaR
# ---------------------------------------------------------------------------


def test_cdar_reduces_drawdown_against_equal_weight(sample_returns: pd.DataFrame):
    from optimization_engine.analytics.risk import drawdown_series

    constraints = PortfolioConstraints(
        bounds={a: (0.0, 0.4) for a in sample_returns.columns}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = CDaROptimizer(
            returns=sample_returns, constraints=constraints, alpha=0.05
        ).optimize()

    equal = pd.Series(
        np.ones(sample_returns.shape[1]) / sample_returns.shape[1],
        index=sample_returns.columns,
    )
    optimized_path = (sample_returns * result.weights).sum(axis=1)
    equal_path = (sample_returns * equal).sum(axis=1)
    assert drawdown_series(optimized_path).min() > drawdown_series(equal_path).min()
    assert result.is_compliant


def test_cdar_reports_the_drawdown_shape_it_optimized(sample_returns: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = CDaROptimizer(
            returns=sample_returns, constraints=PortfolioConstraints()
        ).optimize()
    for key in ("cdar_realized", "max_drawdown", "drawdown_episodes", "cdar_note"):
        assert key in result.extras
    # CDaR averages the worst tail of drawdowns, so it cannot exceed the worst.
    assert result.extras["cdar_realized"] <= abs(result.extras["max_drawdown"]) + 1e-9
    assert result.extras["cdar_realized"] >= result.extras["average_drawdown"]


def test_cdar_alpha_of_one_is_the_average_drawdown(sample_returns: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = CDaROptimizer(
            returns=sample_returns,
            constraints=PortfolioConstraints(),
            alpha=1.0,
        ).optimize()
    assert result.extras["cdar_realized"] == pytest.approx(
        result.extras["average_drawdown"], rel=1e-6
    )


def test_cdar_rejects_an_out_of_range_alpha(sample_returns: pd.DataFrame):
    with pytest.raises(ValueError, match="tail probability"):
        CDaROptimizer(returns=sample_returns, alpha=0.0)


def test_cdar_rejects_a_gappy_history(sample_returns: pd.DataFrame):
    gappy = sample_returns.copy()
    gappy.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        CDaROptimizer(returns=gappy)


# ---------------------------------------------------------------------------
# End to end through the engine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["nco", "herc", "cdar"])
def test_new_methods_run_through_the_engine(method: str, sample_returns: pd.DataFrame):
    config = EngineConfig(
        expected_returns={a: 0.05 for a in sample_returns.columns},
        bounds={a: [0.0, 0.4] for a in sample_returns.columns},
        optimizer=OptimizerSpec(name=method),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = run_engine(sample_returns, config)
    assert run.result.weights.sum() == pytest.approx(1.0)
    assert run.result.is_compliant, run.result.violations
    assert run.result.weights.max() <= 0.4 + 1e-6
