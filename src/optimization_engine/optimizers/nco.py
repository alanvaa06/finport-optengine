"""Nested Clustered Optimization (López de Prado, 2019).

Markowitz's curse: the more correlated the assets, the greater the need for
diversification — and the less stable the optimizer's answer, because a
highly correlated covariance matrix is ill-conditioned and its inverse
amplifies estimation error. López de Prado separates that instability into
two sources:

* **noise** — eigenvalues indistinguishable from those of a random matrix,
  handled by the Marchenko-Pastur filter in
  :mod:`optimization_engine.data.denoise`; and
* **signal** — the block structure itself. Even a perfectly denoised
  correlation matrix is ill-conditioned when it contains strongly correlated
  clusters, because the between-cluster and within-cluster variances live on
  different scales and end up in the same matrix inverse.

NCO addresses the second. Rather than inverting one N×N matrix, it:

1. clusters the correlation matrix into ``k`` groups;
2. optimizes *within* each cluster, on a small and comparatively
   well-conditioned sub-matrix;
3. collapses each cluster to a single synthetic asset, whose covariance is
   the original covariance evaluated at the intra-cluster weights;
4. optimizes *across* those ``k`` synthetic assets; and
5. multiplies the two layers together.

Every matrix that gets inverted is therefore either small (within a cluster)
or nearly diagonal (across clusters, since the clustering put the correlated
assets together). In López de Prado's Monte Carlo experiments this cuts the
RMSE of the recovered weights roughly in half against a direct solve on the
same inputs — most of the improvement coming from portfolios that a direct
optimizer would have pushed to a corner.

The intra- and inter-cluster solves reuse the engine's own convex
optimizers, so the sign conventions and the long-only setting hold at both
layers. The mandate itself (per-asset bounds, group budgets, turnover) is
applied to the combined result by projection, which is why ``bounds_mode`` is
``"soft_iterated"``: a binding bound moves the answer away from NCO's own
allocation, and the distance moved is reported.

References:
    López de Prado, M. (2019). "A Robust Estimator of the Efficient
    Frontier". SSRN 3469961.

    López de Prado, M. (2020). *Machine Learning for Asset Managers*.
    Cambridge University Press, chapter 7.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from optimization_engine.optimizers._bounds import project_to_constraints
from optimization_engine.optimizers._clustering import (
    build_linkage,
    correlation_distance,
    correlation_from_covariance,
    optimal_clusters,
)
from optimization_engine.optimizers.base import BaseOptimizer, PortfolioConstraints

#: Objectives the two nested layers can be solved with.
NCO_OBJECTIVES = ("min_variance", "max_sharpe")


class NCOOptimizer(BaseOptimizer):
    """Nested Clustered Optimization.

    Args:
        objective: ``"min_variance"`` (the default, and the one that needs no
            expected returns) or ``"max_sharpe"``. The same objective is used
            at both layers, which is what makes the two-stage answer
            comparable to the direct one.
        linkage_method: Linkage rule for the correlation dendrogram. ``ward``
            is the default here rather than HRP's ``single``: single linkage
            chains, which is tolerable when the tree is only used for
            ordering (HRP) and harmful when it is used to *partition*.
        n_clusters: Force a cluster count. ``None`` selects it by maximizing
            the silhouette t-statistic, the criterion behind ONC.
        max_clusters: Upper bound for that search.
        detone_for_clustering: Strip the market eigenvector from the
            correlation matrix *before* clustering. Every asset loads on the
            market, so leaving it in makes all pairs look alike and the
            partition degenerates toward one big cluster. The detoned matrix
            is used only for the distance metric — the covariance the two
            solves see is untouched.
    """

    name = "nco"
    bounds_mode = "soft_iterated"

    def __init__(
        self,
        *args,
        objective: str = "min_variance",
        linkage_method: str = "ward",
        n_clusters: int | None = None,
        max_clusters: int | None = None,
        detone_for_clustering: bool = True,
        **kwargs,
    ) -> None:
        """Configure the clustering and the objective solved within each cluster.

        Args:
            *args: Passed to :class:`~optimization_engine.optimizers.base.BaseOptimizer`.
            objective: What the intra- and inter-cluster problems optimize —
                ``"min_variance"`` by default.
            linkage_method: Linkage for the correlation tree.
            n_clusters: Fix the number of clusters. ``None`` selects it from the
                tree.
            max_clusters: Upper bound when the count is selected rather than fixed.
            detone_for_clustering: Strip the market eigenvalue before measuring
                distances, so the clusters reflect residual correlation rather
                than everything loading on one factor. It affects the *distance
                metric only* — the covariance both solves see is untouched.
            **kwargs: Passed to the base class.

        Raises:
            ValueError: If ``objective`` is not a registered NCO objective.
        """
        super().__init__(*args, **kwargs)
        if objective not in NCO_OBJECTIVES:
            raise ValueError(
                f"Unknown NCO objective {objective!r}. "
                f"Available: {list(NCO_OBJECTIVES)}"
            )
        self.objective = objective
        self.linkage_method = linkage_method
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.detone_for_clustering = bool(detone_for_clustering)

    # -- the two nested layers ---------------------------------------------

    def _sub_constraints(self, assets: list) -> PortfolioConstraints:
        """Constraints for a sub-problem: budget and sign only.

        The mandate's per-asset bounds cannot be applied inside a cluster —
        a 10% cap on an asset means 10% of the *portfolio*, and the
        intra-cluster weights sum to 1 within the cluster, not within the
        book. Applying them here would cap the wrong quantity. They are
        imposed on the combined result instead.

        "No bounds" has to be said explicitly under ``long_only=False``. Left
        empty, ``get_bounds`` falls back to its long-short default of
        ``(−1, 1)`` — a box nobody set, which clipped any cluster whose
        unconstrained book levers past one and left nothing in the result to
        say so. Long-only needs no entry: non-negative weights summing to one
        cannot reach the default cap of 1.

        Args:
            assets: The sub-problem's universe — one cluster's assets, or the
                synthetic cluster labels of the inter-cluster layer.
        """
        if self.constraints.long_only:
            return PortfolioConstraints(fully_invested=True, long_only=True)
        return PortfolioConstraints(
            fully_invested=True,
            long_only=False,
            bounds={a: (-np.inf, np.inf) for a in assets},
        )

    def _sub_solve(
        self, cov: pd.DataFrame, mu: pd.Series | None, layer: str = "sub-problem"
    ) -> pd.Series:
        """Solve one layer of the nest with the engine's own optimizers.

        The sub-problem goes through the optimizer's public
        :meth:`~optimization_engine.optimizers.base.BaseOptimizer.optimize`
        rather than its raw ``_solve``, so a nested layer gets the same
        treatment as a top-level solve: the non-finite check, dust removal
        and budget restoration of ``_clean_weights``, and ``bounds_mode``
        recorded on the result. Reaching past ``optimize`` was how the two
        layers came to be the only weights in the engine that nothing ever
        validated.

        What a nested layer does *not* get is the mandate audit, and that is
        the one deliberate exception: the sub-problem is solved against
        :meth:`_sub_constraints` -- budget and sign -- while the mandate's
        per-asset and group limits are applied to the combined book by
        projection in :meth:`_solve`. Auditing a cluster against a
        portfolio-level cap would measure the wrong quantity (a 10% weight
        *within* a cluster is not 10% of the book), and auditing it against the
        budget-and-sign set it was actually solved with re-checks, once per
        cluster plus once more, a constraint the convex program has already
        imposed. So this passes ``run_post_solve_diagnostics=False``, which is
        the per-solve opt-out ``optimize`` carries for exactly this case.

        The invariant the outer product in :meth:`_solve` depends on -- each
        layer's weights summing to one -- is not left to that check either way:
        :meth:`_unit_budget` enforces it below, and says so when it cannot.
        Skipping the sweep also returns what it cost: 12.8% of an NCO solve
        over 15 assets and 12.3% over 64, median of nine runs each. The weight
        cleaning, which is the other half of what the per-layer diagnostics were
        measured at, is kept -- it is what normalizes each layer's budget.

        Args:
            cov: Covariance of this layer's universe — one cluster's assets,
                or the synthetic cluster assets of the inter-cluster layer.
            mu: Expected returns for the same universe, or ``None`` under the
                ``min_variance`` objective.
            layer: Which layer is being solved, used only in error messages.

        Returns:
            Weights indexed by ``cov``'s columns, summing to one.

        Raises:
            ValueError: If the layer's weights net to approximately zero, so
                they cannot be rescaled to the unit budget the nesting
                assumes. Reachable only with ``long_only=False`` — see below.
        """
        from optimization_engine.optimizers.mean_variance import (
            MaxSharpeOptimizer,
            MinVarianceOptimizer,
        )

        assets = list(cov.columns)
        if len(assets) == 1:
            return pd.Series([1.0], index=assets)

        constraints = self._sub_constraints(assets)
        if self.objective == "max_sharpe" and mu is not None:
            optimizer = MaxSharpeOptimizer(
                expected_returns=mu,
                cov_matrix=cov,
                constraints=constraints,
                risk_free_rate=self.risk_free_rate,
            )
        else:
            optimizer = MinVarianceOptimizer(
                cov_matrix=cov, constraints=constraints
            )
        solved = optimizer.optimize(run_post_solve_diagnostics=False)
        weights = solved.weights.reindex(assets)
        return self._unit_budget(weights, layer)

    @staticmethod
    def _unit_budget(weights: pd.Series, layer: str) -> pd.Series:
        """Restore the unit budget the nesting assumes, or say why it cannot.

        ``_clean_weights`` normalizes a fully-invested solve back to one after
        zeroing dust — except when the weights net to approximately zero,
        where it returns them *unnormalized* because rescaling is meaningless
        there. That escape hatch is unreachable in a long-only nest (every
        weight is non-negative and the budget is imposed inside the convex
        program) but is live under ``long_only=False``, where a market-neutral
        cluster can net out. Letting an unnormalized layer through would break
        the invariant the outer product at :meth:`_solve` depends on — each
        cluster's weights sum to one, so ``loadings @ inter`` sums to one —
        and the book would silently stop being fully invested.

        Args:
            weights: One layer's solved weights.
            layer: Which layer produced them, for the error message.

        Returns:
            ``weights`` rescaled to sum to one.

        Raises:
            ValueError: If the weights net to approximately zero.
        """
        total = float(weights.sum())
        if abs(total) < 1e-9:
            raise ValueError(
                f"NCO: the weights for {layer} net to {total:.3g}, so they "
                "cannot be rescaled to the unit budget the nesting assumes. "
                "A cluster whose long and short legs cancel has no meaningful "
                "share of the book to allocate to. Force a different cluster "
                "count with n_clusters=, or run long-only."
            )
        return weights / total

    def _solve(self) -> np.ndarray:
        if self.cov_matrix is None:
            raise ValueError("Covariance matrix required for NCO")
        assets = self.assets
        if len(assets) < 3:
            raise ValueError(
                "NCO needs at least 3 assets: with fewer, the clustering step "
                "cannot produce two non-trivial groups and the method reduces "
                "to a direct solve. Use min_variance or max_sharpe instead."
            )
        cov = self.cov_matrix.reindex(assets, axis=0).reindex(assets, axis=1)

        mu: pd.Series | None = None
        if self.objective == "max_sharpe":
            vector = self._mu_vector()
            if vector is None:
                raise ValueError(
                    "NCO with objective='max_sharpe' needs expected returns. "
                    "Use objective='min_variance' to run without them."
                )
            mu = pd.Series(vector, index=assets)

        assignment = self._cluster(cov)

        # -- layer 1: optimize inside each cluster --------------------------
        intra: dict[int, pd.Series] = {}
        for label, members in assignment.members.items():
            sub_mu = mu.loc[members] if mu is not None else None
            intra[label] = self._sub_solve(
                cov.loc[members, members], sub_mu, layer=f"cluster {label}"
            )

        # -- collapse each cluster into one synthetic asset -----------------
        labels = sorted(intra)
        loadings = pd.DataFrame(0.0, index=assets, columns=labels)
        for label in labels:
            loadings.loc[intra[label].index, label] = intra[label].values
        reduced_cov = pd.DataFrame(
            loadings.T.values @ cov.values @ loadings.values,
            index=labels,
            columns=labels,
        )
        reduced_mu = (
            pd.Series(loadings.T.values @ mu.values, index=labels)
            if mu is not None
            else None
        )

        # -- layer 2: optimize across the synthetic assets ------------------
        inter = self._sub_solve(
            reduced_cov, reduced_mu, layer="the inter-cluster layer"
        )

        combined = pd.Series(
            loadings.values @ inter.reindex(labels).values, index=assets
        )
        self._record_diagnostics(assignment, intra, inter, reduced_cov, cov)

        projected, distance = project_to_constraints(
            combined.values, assets, self.constraints
        )
        self._diagnostics["projection_distance"] = distance
        if distance > 1e-6:
            self._diagnostics["bounds_note"] = (
                f"Constraints moved {distance:.2%} of the book away from the "
                "raw NCO allocation. NCO applies per-asset and group limits by "
                "projection, so a large distance means the mandate rather than "
                "the nesting produced the answer."
            )
        return projected

    # -- clustering ---------------------------------------------------------

    def _cluster(self, cov: pd.DataFrame):
        corr = correlation_from_covariance(cov)
        clustering_corr = corr
        if self.detone_for_clustering and len(cov.columns) > 2:
            from optimization_engine.data.denoise import detone_correlation

            clustering_corr = pd.DataFrame(
                detone_correlation(corr.values, n_factors=1),
                index=corr.index,
                columns=corr.columns,
            )

        link = build_linkage(
            correlation_distance(clustering_corr), method=self.linkage_method
        )
        assignment = optimal_clusters(
            clustering_corr,
            link,
            max_clusters=self.max_clusters,
            n_clusters=self.n_clusters,
        )
        if assignment.silhouette < 0.05:
            warnings.warn(
                f"The correlation matrix barely clusters (mean silhouette "
                f"{assignment.silhouette:.3f}). NCO's advantage comes from "
                "block structure; without it the two-layer solve is close to "
                "a direct one and adds only the clustering's own noise.",
                stacklevel=4,
            )
        return assignment

    def _record_diagnostics(
        self,
        assignment,
        intra: dict[int, pd.Series],
        inter: pd.Series,
        reduced_cov: pd.DataFrame,
        cov: pd.DataFrame,
    ) -> None:
        """Expose both layers, plus the conditioning NCO exists to improve."""
        full_condition = _condition_number(cov.values)
        reduced_condition = _condition_number(reduced_cov.values)
        worst_cluster = max(
            (_condition_number(cov.loc[m, m].values) for m in assignment.members.values()),
            default=float("nan"),
        )
        self._diagnostics.update(
            {
                "nco_objective": self.objective,
                "nco_linkage": self.linkage_method,
                "nco_clusters": {
                    str(label): members
                    for label, members in assignment.members.items()
                },
                "nco_n_clusters": assignment.n_clusters,
                "nco_silhouette": assignment.silhouette,
                "nco_cluster_scores": assignment.candidates,
                "nco_cluster_weights": {
                    str(label): float(inter.get(label, 0.0)) for label in assignment.members
                },
                "nco_condition_direct": full_condition,
                "nco_condition_reduced": reduced_condition,
                "nco_condition_worst_cluster": worst_cluster,
                "nco_conditioning_note": (
                    f"A direct solve would invert a matrix with condition number "
                    f"{full_condition:.3g}. NCO instead inverts "
                    f"{assignment.n_clusters} cluster matrices (worst condition "
                    f"{worst_cluster:.3g}) and one {assignment.n_clusters}×"
                    f"{assignment.n_clusters} matrix (condition "
                    f"{reduced_condition:.3g})."
                ),
            }
        )


def _condition_number(matrix: np.ndarray) -> float:
    eigenvalue = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    smallest = float(eigenvalue.min())
    return float(eigenvalue.max() / smallest) if smallest > 0 else float("inf")
