"""Effective number of bets (Meucci, Santangelo & Deguest, 2013/2015).

The engine already reports two concentration measures: the effective number
of *positions* (inverse Herfindahl of the weights) and the effective number
of *risk* contributions (inverse Herfindahl of the Euler risk shares). Both
have the same blind spot — they are computed asset by asset, and assets are
correlated.

Hold ten European bank stocks and the weight-based measure says ten. The
risk-contribution measure says roughly ten too, because each name does carry
a tenth of the risk. Neither notices that all ten are the same bet.

Meucci's answer is to stop measuring bets in assets and start measuring them
in *uncorrelated factors*. Rotate the assets into a set of uncorrelated
factors, express the portfolio in that basis, and the resulting variance
shares form a genuine probability distribution — the **diversification
distribution**. Its exponential entropy is the effective number of bets:

    ``p_n = (θ_n² · λ_n) / Σ_m θ_m² λ_m``,   ``ENB = exp(−Σ_n p_n · ln p_n)``

Uniform shares give ``ENB = N``; a portfolio whose variance all comes from a
single factor gives ``ENB = 1``, regardless of how many line items it holds.

Which rotation you use matters, and the two on offer measure genuinely
different things. This is not a detail to pick a default for and forget:

* ``"pca"`` — the principal components, ordered by variance. The leading
  component absorbs whatever the assets have in common, so ENB collapses
  toward 1 exactly when one driver dominates the book. It reads concentration
  correctly. The cost is that the factors are statistical artefacts: the third
  principal component of a multi-asset panel is rarely something anyone can
  name, and it reorders and flips sign as the sample changes.
* ``"minimum_torsion"`` — the rotation that is uncorrelated *and* stays as
  close as possible to the original assets in a tracking-error sense. Each
  factor remains recognizably "the equity bet" or "the duration bet", so the
  distribution can be read and acted on. The cost is the mirror image:
  because it refuses to move far from the asset basis, it does not
  concentrate common variation into one factor. In the limiting case of a
  perfectly equicorrelated matrix held at equal weights, symmetry forces
  ``ENB = N`` however high the correlation — the rotation has no asymmetry to
  work with.

So they answer two different questions. PCA asks *how much independent
variation* the portfolio is exposed to; minimum torsion asks *how many
distinct, nameable positions* it takes. Neither is the "true" number, and the
useful diagnostic is usually the **gap between them**: on the engine's sample
panel an equal-weight book scores 9.8 effective bets on the minimum-torsion
factors and 1.6 on the principal components. Thirteen distinct positions,
one dominant driver. :func:`compare_diversification` reports both side by
side for exactly that reason.

References:
    Meucci, A. (2009). "Managing Diversification". *Risk* 22(5).

    Meucci, A., Santangelo, A. and Deguest, R. (2015). "Risk Budgeting and
    Diversification Based on Optimized Uncorrelated Factors". *Risk* 28(11).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TORSION_MODELS = ("minimum_torsion", "pca")


def minimum_torsion(
    cov_matrix: pd.DataFrame,
    max_iterations: int = 10_000,
    tolerance: float = 1e-14,
) -> pd.DataFrame:
    """The minimum-torsion rotation into uncorrelated factors.

    Finds the linear map ``t`` such that the factors ``t·r`` are mutually
    uncorrelated and the total tracking error between each factor and its
    corresponding original asset is as small as possible. Meucci's iterative
    algorithm works on the correlation matrix's Riccati root and converges
    quickly; the volatilities are scaled back in at the end.

    Args:
        cov_matrix: Asset covariance matrix.
        max_iterations: Cap on the fixed-point iteration.
        tolerance: Relative convergence tolerance on the Frobenius residual.

    Returns:
        The torsion matrix ``t``, indexed by factor (named after the asset
        each factor tracks) and columned by asset.

    Raises:
        ValueError: If any asset has zero variance.
    """
    import scipy.linalg

    assets = list(cov_matrix.columns)
    sigma = np.asarray(cov_matrix.values, dtype=float)
    std = np.sqrt(np.diag(sigma))
    if not (std > 0).all():
        raise ValueError(
            "The minimum-torsion rotation is undefined when an asset has zero "
            "variance. Drop constant series first."
        )
    n = len(assets)
    corr = sigma / np.outer(std, std)
    # Riccati root: the symmetric square root of the correlation matrix.
    c = np.real(scipy.linalg.sqrtm(corr))

    d = np.ones(n)
    previous = np.inf
    perturbation = c
    for _ in range(max_iterations):
        u = np.real(scipy.linalg.sqrtm(np.diag(d) @ c @ c @ np.diag(d)))
        q = np.linalg.pinv(u) @ np.diag(d) @ c
        d = np.diag(q @ c).copy()
        perturbation = np.diag(d) @ q
        residual = float(np.linalg.norm(c - perturbation, "fro"))
        if previous < np.inf and abs(residual - previous) / max(residual, 1e-16) / n <= tolerance:
            break
        previous = residual

    rotation = perturbation @ np.linalg.pinv(c)
    torsion = np.diag(std) @ rotation @ np.diag(1.0 / std)
    return pd.DataFrame(torsion, index=assets, columns=assets)


def pca_torsion(cov_matrix: pd.DataFrame) -> pd.DataFrame:
    """Principal-component rotation, eigenvalues descending.

    Signs are fixed so each component's largest loading is positive, which
    stops the factors flipping arbitrarily between samples and makes the
    diversification distribution comparable across runs.

    Args:
        cov_matrix: Asset covariance, indexed and columned by asset.

    Returns:
        The rotation matrix, assets down the index and one column per factor.
        Unlike the minimum-torsion rotation, its factors are uncorrelated but
        need not resemble the assets they came from.
    """
    assets = list(cov_matrix.columns)
    sigma = np.asarray(cov_matrix.values, dtype=float)
    eigenvalue, eigenvector = np.linalg.eigh((sigma + sigma.T) / 2.0)
    order = np.argsort(eigenvalue)[::-1]
    eigenvector = eigenvector[:, order]
    for j in range(eigenvector.shape[1]):
        column = eigenvector[:, j]
        if column[np.argmax(np.abs(column))] < 0:
            eigenvector[:, j] = -column
    return pd.DataFrame(
        eigenvector.T,
        index=[f"PC{j + 1}" for j in range(len(assets))],
        columns=assets,
    )


@dataclass(frozen=True)
class DiversificationReport:
    """The diversification distribution and the number it summarizes.

    Attributes:
        effective_number_of_bets: ``exp(−Σ p ln p)``, between 1 and N.
        distribution: Share of portfolio variance carried by each
            uncorrelated factor, summing to 1 and sorted descending.
        model: Which rotation produced it.
        n_factors: ``N``.
        concentration: ``ENB / N`` — the share of the theoretical maximum
            achieved, which is what makes the number comparable across
            universes of different sizes.
        largest_bet: The single factor's share of variance. The number that
            says "68% of this book is one bet".
    """

    effective_number_of_bets: float
    distribution: pd.Series
    model: str
    n_factors: int
    concentration: float
    largest_bet: float

    def describe(self) -> str:
        """How many independent bets the book really holds, and where they sit.

        Returns:
            A sentence giving the effective number of bets against the maximum
            available, the factor model used, and the share of variance carried by
            the single largest bet.
        """
        top = self.distribution.index[0]
        return (
            f"{self.effective_number_of_bets:.2f} effective bets out of "
            f"{self.n_factors} possible ({self.concentration:.0%} of the "
            f"maximum), on the {self.model.replace('_', '-')} factors. The "
            f"largest single bet - the factor tracking {top} - carries "
            f"{self.largest_bet:.0%} of portfolio variance."
        )


def diversification_distribution(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
    model: str = "minimum_torsion",
) -> DiversificationReport:
    """Variance shares of a portfolio across uncorrelated factors.

    Args:
        weights: Portfolio weights. Active weights work equally well and
            answer the more interesting question — how many independent bets
            the *active* book takes.
        cov_matrix: Covariance over the same universe.
        model: ``"minimum_torsion"`` or ``"pca"``.

    Returns:
        A :class:`DiversificationReport`.

    Raises:
        ValueError: On an unknown model, a non-overlapping universe, or a
            portfolio with zero variance.
    """
    if model not in TORSION_MODELS:
        raise ValueError(
            f"Unknown torsion model {model!r}. Available: {list(TORSION_MODELS)}"
        )
    assets = [a for a in cov_matrix.columns if a in weights.index]
    if len(assets) < 2:
        raise ValueError(
            "The weight vector and the covariance matrix must share at least "
            "2 assets."
        )
    sigma = cov_matrix.loc[assets, assets]
    w = weights.reindex(assets).fillna(0.0).values.astype(float)

    torsion = (
        minimum_torsion(sigma) if model == "minimum_torsion" else pca_torsion(sigma)
    )
    t = torsion.values
    # Factor exposures: w'r = w' t⁻¹ (t r) = θ'z, so θ = (t')⁻¹ w.
    exposures = np.linalg.pinv(t.T) @ w
    factor_cov = t @ sigma.values @ t.T
    variances = np.diag(factor_cov)

    contributions = (exposures**2) * variances
    total = float(contributions.sum())
    if total <= 0:
        raise ValueError(
            "The portfolio has zero variance, so it makes no bets to count."
        )
    shares = pd.Series(contributions / total, index=torsion.index)
    shares = shares.sort_values(ascending=False)

    positive = shares[shares > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    enb = float(np.exp(entropy))
    n = len(assets)
    return DiversificationReport(
        effective_number_of_bets=enb,
        distribution=shares,
        model=model,
        n_factors=n,
        concentration=enb / n,
        largest_bet=float(shares.iloc[0]),
    )


def effective_number_of_bets(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
    model: str = "minimum_torsion",
) -> float:
    """Just the number, for callers that do not need the distribution.

    Args:
        weights: Portfolio weights, as fractions of the book.
        cov_matrix: Asset covariance over the same universe.
        model: The rotation used to build uncorrelated factors —
            ``"minimum_torsion"`` (the default, Meucci's) or ``"pca"``.

    Returns:
        The effective number of independent bets, between ``1`` and the number
        of assets.
    """
    return diversification_distribution(
        weights, cov_matrix, model=model
    ).effective_number_of_bets


def compare_diversification(
    weights: pd.Series, cov_matrix: pd.DataFrame
) -> pd.DataFrame:
    """Effective number of bets under both rotations, side by side.

    The two numbers bracket the honest answer, and their ratio is the
    diagnostic: close together means the portfolio's distinct positions are
    also distinct risks; far apart means it holds many positions that share
    one driver, which is the situation every concentration measure computed
    asset-by-asset will miss.

    Args:
        weights: Portfolio weights, as fractions of the book.
        cov_matrix: Asset covariance over the same universe.

    Returns:
        A frame indexed by model with the effective number of bets, its share
        of the maximum, and the largest single factor's variance share.
    """
    rows = {}
    for model in TORSION_MODELS:
        report = diversification_distribution(weights, cov_matrix, model=model)
        rows[model] = {
            "effective_bets": report.effective_number_of_bets,
            "share_of_maximum": report.concentration,
            "largest_bet": report.largest_bet,
        }
    frame = pd.DataFrame(rows).T
    frame.index.name = "model"
    return frame
