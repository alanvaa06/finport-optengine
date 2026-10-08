"""Correlation clustering shared by the hierarchical methods.

HRP, HERC and NCO all start the same way — turn a covariance matrix into a
correlation distance, build a dendrogram, and decide how many clusters the
data actually supports — and then diverge in how they allocate down it. That
common prefix lives here so the three methods cannot quietly disagree about
what a cluster is.

The one genuinely contestable step is *how many* clusters to cut the tree
into. López de Prado's ONC algorithm (*Machine Learning for Asset Managers*,
§4.4) picks the partition that maximizes the t-statistic of the silhouette
scores; the implementation below applies the same criterion directly to the
hierarchical tree rather than to the k-means restarts ONC uses, which keeps
the clustering deterministic and consistent with the dendrogram the caller
already has. The chosen ``k`` and its score are always reported, so a caller
who disagrees can override it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from optimization_engine.optimizers.base import zero_variance_assets

LINKAGE_METHODS = ("single", "average", "complete", "ward")


def correlation_from_covariance(cov: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix, with a clear error for degenerate assets.

    Args:
        cov: A covariance matrix, indexed and columned by asset.

    Returns:
        The correlation matrix over the same universe.

    Raises:
        ValueError: If any asset has zero variance — exactly, or to rounding,
            as a constant series does (see ``zero_variance_assets``) — which
            leaves its correlation, and so its distance to everything else,
            undefined.
    """
    variances = np.diag(np.asarray(cov.values, dtype=float))
    dead = [str(a) for a in zero_variance_assets(list(cov.columns), variances)]
    if dead:
        raise ValueError(
            f"Zero-variance asset(s) {dead}: the correlation distance is "
            "undefined. Drop them from the universe."
        )
    std = np.sqrt(variances)
    corr = np.asarray(cov.values, dtype=float) / np.outer(std, std)
    return pd.DataFrame(
        np.clip(corr, -1.0, 1.0), index=cov.index, columns=cov.columns
    )


def correlation_distance(corr: pd.DataFrame) -> np.ndarray:
    """``d = √(½(1 − ρ))`` — López de Prado's correlation-based metric.

    This is a true metric (it satisfies the triangle inequality), which
    ordinary ``1 − ρ`` is not; hierarchical clustering on a non-metric
    dissimilarity can produce inversions in the dendrogram.

    Args:
        corr: A correlation matrix.

    Returns:
        A square distance matrix in ``[0, 1]``, zero on the diagonal.
    """
    values = np.asarray(corr.values, dtype=float)
    dist = np.sqrt(np.clip((1.0 - values) / 2.0, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    return (dist + dist.T) / 2.0


def build_linkage(distance: np.ndarray, method: str = "ward") -> np.ndarray:
    """Hierarchical linkage from a square correlation-distance matrix.

    Args:
        distance: A square correlation-distance matrix.
        method: ``"single"``, ``"average"``, ``"complete"`` or ``"ward"``.

    Returns:
        A SciPy linkage matrix.

    Raises:
        ValueError: On an unknown linkage rule.
    """
    if method not in LINKAGE_METHODS:
        raise ValueError(
            f"Unknown linkage {method!r}. Available: {list(LINKAGE_METHODS)}"
        )
    return linkage(squareform(distance, checks=False), method=method)


@dataclass(frozen=True)
class ClusterAssignment:
    """A partition of the universe, and how well the data supported it.

    Attributes:
        labels: ``asset -> cluster id``.
        members: ``cluster id -> assets``, in the caller's asset order.
        n_clusters: Number of clusters chosen.
        silhouette: Mean silhouette score of the chosen partition. Above ~0.5
            the clusters are well separated; near 0 the correlation structure
            does not really cluster and the hierarchy is arbitrary.
        score: The selection criterion — the t-statistic of the silhouette
            scores (mean / standard deviation), which is what ONC maximizes.
            Preferring it to the plain mean stops the search picking a
            partition that is excellent for a few assets and poor for the rest.
        candidates: ``k -> score`` for every partition considered, so the
            choice can be inspected rather than trusted.
    """

    labels: dict[str, int]
    members: dict[int, list[str]]
    n_clusters: int
    silhouette: float
    score: float
    candidates: dict[int, float]


def optimal_clusters(
    corr: pd.DataFrame,
    link: np.ndarray,
    max_clusters: int | None = None,
    n_clusters: int | None = None,
) -> ClusterAssignment:
    """Cut the dendrogram at the number of clusters the data best supports.

    Args:
        corr: Correlation matrix the tree was built from.
        link: Linkage produced by :func:`build_linkage`.
        max_clusters: Largest partition to consider. Defaults to
            ``min(10, N − 1)``: beyond roughly ten clusters the "clusters"
            are mostly singletons and the silhouette stops being informative.
        n_clusters: Force this many clusters, skipping the search. The
            silhouette of the forced partition is still reported.

    Returns:
        A :class:`ClusterAssignment`.

    Raises:
        ValueError: If fewer than two assets are supplied.
    """
    from sklearn.metrics import silhouette_samples

    assets = [str(a) for a in corr.columns]
    n = len(assets)
    if n < 2:
        raise ValueError("Clustering needs at least 2 assets.")
    distance = correlation_distance(corr)

    def assess(k: int) -> tuple[float, float, np.ndarray]:
        """Score one candidate cluster count.

        Args:
            k: Number of clusters to cut the tree into.

        Returns:
            ``(score, mean_silhouette, labels)``. The score is the ONC criterion —
            the mean silhouette per unit of its own dispersion — so a clustering
            that is good *and* consistent beats one that is good only on average.
            A cut that collapses to a single cluster scores ``-inf``.
        """
        labels = fcluster(link, t=k, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            return float("-inf"), float("nan"), labels
        samples = silhouette_samples(distance, labels, metric="precomputed")
        mean = float(np.mean(samples))
        spread = float(np.std(samples))
        # The ONC criterion: mean silhouette per unit of its own dispersion.
        return (mean / spread if spread > 0 else mean * 1e6), mean, labels

    if n_clusters is not None:
        k = int(np.clip(n_clusters, 2, n))
        score, mean, labels = assess(k)
        candidates = {k: score}
    else:
        upper = max_clusters if max_clusters is not None else min(10, n - 1)
        upper = int(np.clip(upper, 2, max(2, n - 1)))
        candidates = {}
        best: tuple[float, float, np.ndarray, int] | None = None
        for candidate in range(2, upper + 1):
            score, mean, labels = assess(candidate)
            candidates[candidate] = score
            if best is None or score > best[0]:
                best = (score, mean, labels, candidate)
        assert best is not None
        score, mean, labels, k = best

    members: dict[int, list[str]] = {}
    for asset, label in zip(assets, labels):
        members.setdefault(int(label), []).append(asset)

    return ClusterAssignment(
        labels={a: int(lbl) for a, lbl in zip(assets, labels)},
        members=members,
        n_clusters=len(members),
        silhouette=float(mean),
        score=float(score),
        candidates=candidates,
    )


def inverse_variance_weights(cov: pd.DataFrame, items: list[str]) -> pd.Series:
    """Naive risk parity inside a cluster: weights ∝ ``1/σ²``.

    This is the allocation HRP and HERC use where they do not recurse
    further — it ignores the off-diagonal terms, which is exactly the point
    within a cluster whose members are correlated by construction.

    Args:
        cov: The full covariance matrix.
        items: The cluster's members.

    Returns:
        Weights over ``items``, summing to 1.
    """
    variance = np.diag(cov.loc[items, items].values)
    inverse = 1.0 / np.where(variance > 0, variance, np.inf)
    total = float(inverse.sum())
    if total <= 0:
        return pd.Series(np.ones(len(items)) / len(items), index=items)
    return pd.Series(inverse / total, index=items)


def cluster_variance(cov: pd.DataFrame, items: list[str]) -> float:
    """Variance of a cluster held at its own inverse-variance weights.

    Args:
        cov: The full covariance matrix.
        items: The cluster's members.

    Returns:
        The cluster's variance, in the covariance's own units. This is what
        the tree's bisection step compares two branches on.
    """
    weights = inverse_variance_weights(cov, items).values
    return float(weights @ cov.loc[items, items].values @ weights)
