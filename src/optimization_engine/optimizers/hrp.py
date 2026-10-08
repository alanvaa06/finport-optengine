"""Hierarchical Risk Parity (López de Prado, 2016).

HRP avoids matrix inversion entirely. Steps:

1. Build a correlation distance ``d = √(½(1 − ρ))``.
2. Cluster with single linkage (or any linkage method).
3. Quasi-diagonalize: reorder assets so similar items sit together.
4. Recursive bisection: walk the cluster tree, splitting risk between
   sub-clusters using inverse-variance weights at each split.

The result is robust to ill-conditioned covariance matrices and to noisy
estimates — particularly useful with many assets or limited history, and the
natural choice when ``T/N`` is too small for mean-variance to mean anything.

Constraints are *not* part of the recursion: HRP allocates top-down and the
result is then projected onto the closest feasible allocation, group budgets
included. That makes the mandate binding but the method approximate, which is
why ``bounds_mode`` is ``"soft_iterated"`` and the distance the projection
moved is reported alongside the weights.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from optimization_engine.optimizers._bounds import project_to_constraints
from optimization_engine.optimizers.base import BaseOptimizer, zero_variance_assets

LINKAGE_METHODS = ("single", "average", "complete", "ward")


def _correl_distance(corr: pd.DataFrame) -> pd.DataFrame:
    return ((1 - corr) / 2.0) ** 0.5


def _quasi_diag(link: np.ndarray) -> list[int]:
    link = link.astype(int)
    sort_ix = pd.Series([int(link[-1, 0]), int(link[-1, 1])])
    num_items = int(link[-1, 3])
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index.to_numpy()
        j = df0.values - num_items
        sort_ix.loc[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.astype(int).tolist()


def _ivp_weights(cov: pd.DataFrame, items: list[str]) -> np.ndarray:
    sub = np.diag(cov.loc[items, items].values)
    inv = 1.0 / sub
    inv = inv / inv.sum()
    return inv


def _cluster_var(cov: pd.DataFrame, items: list[str]) -> float:
    sub_cov = cov.loc[items, items].values
    w = _ivp_weights(cov, items)
    return float(w @ sub_cov @ w)


def _recursive_bisection(cov: pd.DataFrame, sort_ix: list[str]) -> pd.Series:
    weights = pd.Series(np.ones(len(sort_ix)), index=sort_ix, dtype=float).copy()
    clusters: list[list[str]] = [list(sort_ix)]
    while clusters:
        next_clusters: list[list[str]] = []
        for c in clusters:
            if len(c) <= 1:
                continue
            split = len(c) // 2
            left, right = c[:split], c[split:]
            v_left = _cluster_var(cov, left)
            v_right = _cluster_var(cov, right)
            denom = v_left + v_right
            alpha = 1 - v_left / denom if denom > 0 else 0.5
            weights.loc[left] = weights.loc[left].values * alpha
            weights.loc[right] = weights.loc[right].values * (1 - alpha)
            next_clusters.extend([left, right])
        clusters = next_clusters
    return weights


class HRPOptimizer(BaseOptimizer):
    """Hierarchical Risk Parity (HRP).

    Linkage is configurable: ``single`` (default, López de Prado),
    ``average``, ``complete``, ``ward``.
    """

    name = "hrp"
    bounds_mode = "soft_iterated"

    def __init__(self, *args, linkage_method: str = "single", **kwargs) -> None:
        """Choose the linkage the correlation tree is built with.

        Args:
            *args: Passed to :class:`~optimization_engine.optimizers.base.BaseOptimizer`.
            linkage_method: ``"single"`` (the default, and López de Prado's own),
                ``"average"``, ``"complete"`` or ``"ward"``.
            **kwargs: Passed to the base class.

        Raises:
            ValueError: If ``linkage_method`` is not one of those four.
        """
        super().__init__(*args, **kwargs)
        if linkage_method not in LINKAGE_METHODS:
            raise ValueError(
                f"Unknown HRP linkage {linkage_method!r}. "
                f"Available: {list(LINKAGE_METHODS)}"
            )
        self.linkage_method = linkage_method

    def _solve(self) -> np.ndarray:
        if self.cov_matrix is None:
            raise ValueError("Covariance matrix required for HRP")
        if len(self.assets) < 2:
            raise ValueError(
                "HRP needs at least 2 assets to build a cluster tree."
            )
        if self.constraints.has_layer_limits:
            warnings.warn(
                "HRP allocates down its own correlation-derived hierarchy, "
                "which generally disagrees with a hand-specified one. The "
                "layered bucket budgets will be met by projecting the result "
                "onto the constraint set, which moves it away from HRP's own "
                "answer — use risk_parity or mean_variance to have them "
                "enforced inside the solve.",
                stacklevel=3,
            )
        if (np.array([self.constraints.get_bounds(a)[0] for a in self.assets]) < 0).any():
            raise ValueError(
                "HRP produces long-only weights by construction; a negative "
                "minimum weight cannot be honoured."
            )

        cov = self.cov_matrix
        zero = zero_variance_assets(list(cov.columns), np.diag(cov.values))
        if zero:
            raise ValueError(
                f"Zero-variance asset(s) {zero}: the correlation distance is "
                "undefined. Drop them from the universe."
            )
        std = np.sqrt(np.diag(cov.values))
        corr = cov.values / np.outer(std, std)
        corr = np.clip(corr, -1.0, 1.0)
        corr_df = pd.DataFrame(corr, index=cov.index, columns=cov.columns)

        dist = np.array(_correl_distance(corr_df).values, copy=True)
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method=self.linkage_method)
        sort_ix = _quasi_diag(link)
        ordered = [cov.columns[i] for i in sort_ix]

        w = _recursive_bisection(cov, ordered)
        w = w.reindex(self.assets).fillna(0.0)
        weights = w.values.astype(float)

        self._record_cluster_diagnostics(link, ordered, corr_df)

        projected, drift = project_to_constraints(
            weights, self.assets, self.constraints
        )
        self._diagnostics["projection_distance"] = drift
        if drift > 1e-6:
            self._diagnostics["bounds_note"] = (
                f"Constraints moved {drift:.2%} of the book away from the raw "
                "HRP allocation. HRP applies them by projection, so a large "
                "distance means the mandate — not the hierarchy — is driving "
                "the result."
            )
        return projected

    def _record_cluster_diagnostics(
        self, link: np.ndarray, ordered: list[str], corr: pd.DataFrame
    ) -> None:
        """Expose the hierarchy so the analyst can sanity-check the clustering.

        HRP's whole premise is that the tree it finds is economically
        sensible. Showing the ordering and the natural cluster split is what
        lets someone confirm that, rather than trusting it.
        """
        n = len(ordered)
        k = max(2, min(int(np.sqrt(n)), n - 1))
        try:
            labels = fcluster(link, t=k, criterion="maxclust")
            members: dict[int, list[str]] = {}
            for asset, lab in zip(corr.columns, labels):
                members.setdefault(int(lab), []).append(str(asset))
            self._diagnostics["hrp_clusters"] = members
            self._diagnostics["hrp_n_clusters"] = len(members)
        except Exception:  # pragma: no cover - scipy edge cases
            pass
        self._diagnostics["hrp_order"] = [str(a) for a in ordered]
        self._diagnostics["hrp_linkage"] = self.linkage_method
        off_diag = corr.values[~np.eye(n, dtype=bool)]
        self._diagnostics["mean_correlation"] = float(off_diag.mean())
