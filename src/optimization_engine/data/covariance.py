"""Covariance and expected-return estimation.

Wraps the common estimators (sample, Ledoit-Wolf, OAS, EWMA, semicovariance)
behind a single, ergonomic API.

Every method name resolves to exactly one estimator, on every machine. That
is a deliberate constraint rather than an obvious one: ``"shrink"`` used to
route through ``riskfolio-lib`` when it happened to be installed and fall
back to Ledoit-Wolf when it was not, which meant the same config on the same
data produced different numbers depending on the environment, with nothing
in the output saying which had run. An engine that reports what an
allocation rests on cannot have an estimator that varies with the virtualenv.

Every estimator returns a matrix that is symmetric and positive semi-definite:
estimators that can produce indefinite matrices (EWMA on short samples,
semicovariance) are passed through :func:`nearest_psd`. Downstream convex
solvers rely on that invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

CovarianceMethod = Literal[
    "sample", "ledoit_wolf", "oas", "shrink", "ewma", "semi", "denoised"
]

ExpectedReturnMethod = Literal[
    "mean", "geometric_mean", "ema", "capm", "shrunk_mean"
]

#: Human-facing description of each estimator, surfaced by the UI so the
#: analyst can see the assumption they are buying into.
COVARIANCE_DESCRIPTIONS: dict[str, str] = {
    "sample": (
        "Unbiased sample covariance. Unbiased but noisy: needs T >> N "
        "observations or the matrix becomes ill-conditioned (and singular "
        "once T <= N)."
    ),
    "ledoit_wolf": (
        "Ledoit-Wolf shrinkage toward a scaled identity. Well-conditioned "
        "and the safe default when the number of assets is large relative "
        "to the sample."
    ),
    "oas": (
        "Oracle Approximating Shrinkage. Like Ledoit-Wolf but with a "
        "shrinkage intensity tuned for Gaussian data - usually shrinks "
        "harder on short samples."
    ),
    "shrink": (
        "Alias for ledoit_wolf, kept so that configs and saved scenarios "
        "written against it keep loading. Prefer ledoit_wolf in new work."
    ),
    "ewma": (
        "RiskMetrics exponentially-weighted covariance. Reacts fast to "
        "regime changes; effective sample is only ~1/(1-lambda) observations, "
        "so it is noisier than it looks."
    ),
    "semi": (
        "Semicovariance built from downside deviations below the MAR. "
        "Targets downside co-movement rather than total variance; "
        "portfolio 'volatility' from this matrix is a semi-deviation, not "
        "a standard deviation."
    ),
    "denoised": (
        "Sample covariance with the Marchenko-Pastur noise eigenvalues "
        "replaced by their average (Lopez de Prado, 2020). Keeps the factor "
        "structure intact instead of shrinking it along with the noise, so "
        "it conditions the matrix without flattening the signal."
    ),
}


# ---------------------------------------------------------------------------
# PSD repair + diagnostics
# ---------------------------------------------------------------------------


def nearest_psd(matrix: pd.DataFrame | np.ndarray, eig_floor: float = 0.0):
    """Return the nearest positive semi-definite version of ``matrix``.

    Symmetrizes, then clips eigenvalues at ``eig_floor``. When the input is
    already PSD this is (numerically) the identity transform.

    Args:
        matrix: A square covariance-like matrix.
        eig_floor: Minimum eigenvalue to keep. ``0.0`` gives PSD; pass a
            small positive number (e.g. ``1e-12``) to force strict positive
            definiteness for solvers that need it.
    """
    is_frame = isinstance(matrix, pd.DataFrame)
    values = np.asarray(matrix.values if is_frame else matrix, dtype=float)
    sym = (values + values.T) / 2.0
    eigval, eigvec = np.linalg.eigh(sym)
    if (eigval >= eig_floor).all():
        repaired = sym
    else:
        repaired = (eigvec * np.clip(eigval, eig_floor, None)) @ eigvec.T
        repaired = (repaired + repaired.T) / 2.0
    if is_frame:
        return pd.DataFrame(repaired, index=matrix.index, columns=matrix.columns)
    return repaired


@dataclass(frozen=True)
class CovarianceDiagnostics:
    """Conditioning and sample-adequacy diagnostics for a covariance estimate.

    Attributes:
        n_assets: Number of assets ``N``.
        n_observations: Number of return observations ``T``.
        observations_per_asset: ``T / N``. Below ~10 the sample covariance is
            badly conditioned; at ``T <= N`` it is singular.
        condition_number: Ratio of largest to smallest eigenvalue.
        min_eigenvalue: Smallest eigenvalue of the estimate.
        is_psd: Whether every eigenvalue is non-negative (within tolerance).
        effective_observations: The sample the estimator actually leans on —
            for EWMA ``1/(1−λ)`` (see
            :func:`effective_sample_size`), otherwise ``T``. Unrounded here;
            the denoiser rounds it because the Marchenko-Pastur ratio needs
            an observation count.
        warnings: Human-facing warnings, ordered most severe first.
    """

    n_assets: int
    n_observations: int
    observations_per_asset: float
    condition_number: float
    min_eigenvalue: float
    is_psd: bool
    effective_observations: float
    warnings: tuple[str, ...] = ()

    @property
    def is_reliable(self) -> bool:
        """True when nothing worth warning the analyst about was found."""
        return not self.warnings


def _ewma_effective_observations(ewma_lambda: float) -> float:
    """Observations an EWMA estimator effectively leans on: ``1 / (1 − λ)``.

    **The single definition of the EWMA effective sample in this library.**
    The geometric weights ``(1 − λ)λ^k`` put almost all their mass on the
    recent past, so the estimate rests on about ``1 / (1 − λ)`` observations —
    roughly 17 at the RiskMetrics default λ = 0.94 — however long the panel
    is. Two places need that number and they used to compute it separately,
    which meant the diagnostics could warn about a 17-observation sample
    while the denoiser was told it had two thousand.

    Args:
        ewma_lambda: The decay factor, in ``[0, 1)``.

    Returns:
        The effective observation count, unrounded.
    """
    return 1.0 / (1.0 - float(ewma_lambda))


def effective_sample_size(
    n_observations: int,
    method: str = "sample",
    ewma_lambda: float = 0.94,
) -> tuple[int, str]:
    """How many observations an estimator actually leans on, and why.

    An equally-weighted estimator uses every row exactly once, so its
    effective sample *is* the row count. EWMA does not, and anything
    downstream that reasons about sampling error — a T/N ratio, the
    Marchenko-Pastur noise edge — has to be handed the smaller number or it
    will treat two thousand rows of exponentially-decayed history as two
    thousand independent observations.

    Args:
        n_observations: Rows the estimator saw.
        method: The covariance estimator's name.
        ewma_lambda: The decay, used only when ``method == "ewma"``.

    Returns:
        ``(n_effective, note)``. ``note`` is empty when the effective sample
        is the row count, and one sentence naming the decay when it is not.
    """
    if method != "ewma" or not 0.0 <= float(ewma_lambda) < 1.0:
        return int(n_observations), ""
    exact = _ewma_effective_observations(ewma_lambda)
    effective = min(int(n_observations), max(1, int(round(exact))))
    if effective == int(n_observations):
        return effective, ""
    return effective, (
        f"EWMA with lambda = {float(ewma_lambda):g} leans on about "
        f"1/(1-lambda) ~ {exact:.0f} observations regardless of the "
        f"{int(n_observations)} rows supplied."
    )


def covariance_diagnostics(
    cov: pd.DataFrame,
    n_observations: int,
    method: str = "sample",
    ewma_lambda: float = 0.94,
) -> CovarianceDiagnostics:
    """Assess whether a covariance estimate is fit to optimize against.

    The checks are the ones that actually bite in portfolio construction:
    too few observations per asset, near-singularity (which makes
    mean-variance weights explode), and non-PSD matrices.

    Args:
        cov: The estimated covariance, indexed and columned by asset.
        n_observations: How many periods it was estimated from. This is the
            *common* sample — the rows every asset has at once — not the
            length of the longest column.
        method: The estimator that produced it, used to phrase the advice.
        ewma_lambda: The decay used, when the estimator was EWMA. It sets the
            effective sample size, which is shorter than the nominal one.

    Returns:
        A :class:`CovarianceDiagnostics` carrying the condition number, the
        observations-per-asset ratio, whether the matrix is PSD, and the
        findings phrased for a reader.
    """
    values = np.asarray(cov.values, dtype=float)
    n_assets = values.shape[0]
    eigval = np.linalg.eigvalsh((values + values.T) / 2.0)
    min_eig = float(eigval.min())
    max_eig = float(eigval.max())
    cond = float(max_eig / min_eig) if min_eig > 0 else float("inf")
    tol = max(abs(max_eig), 1.0) * 1e-10
    obs_per_asset = (
        float(n_observations) / n_assets if n_assets else float("nan")
    )
    eff_obs = (
        _ewma_effective_observations(ewma_lambda)
        if method == "ewma" and ewma_lambda < 1.0
        else float(n_observations)
    )

    msgs: list[str] = []
    if n_observations <= n_assets:
        msgs.append(
            f"Only {n_observations} observations for {n_assets} assets (T <= N): "
            "the sample covariance is singular. Use a shrinkage estimator "
            "(ledoit_wolf / oas) or shorten the universe."
        )
    elif obs_per_asset < 10:
        msgs.append(
            f"Only {obs_per_asset:.1f} observations per asset (T/N). Below ~10, "
            "covariance estimates are dominated by noise - prefer a shrinkage "
            "estimator or HRP, which does not invert the matrix."
        )
    if eff_obs < 2 * n_assets:
        msgs.append(
            f"Effective sample size is ~{eff_obs:.0f} for {n_assets} assets. "
            "Estimation error will dominate optimizer output."
        )
    if not np.isfinite(cond) or cond > 1e8:
        msgs.append(
            f"Condition number is {cond:.3g}. The matrix is near-singular, so "
            "mean-variance weights are unstable - small changes in expected "
            "returns will produce very different portfolios."
        )
    elif cond > 1e4:
        msgs.append(
            f"Condition number is {cond:.3g}: the covariance is poorly "
            "conditioned. Consider shrinkage or fewer, less-collinear assets."
        )
    if min_eig < -tol:
        msgs.append(
            f"Smallest eigenvalue is {min_eig:.3g} (< 0): the matrix is not "
            "positive semi-definite and has been repaired by eigenvalue "
            "clipping."
        )

    return CovarianceDiagnostics(
        n_assets=n_assets,
        n_observations=int(n_observations),
        observations_per_asset=obs_per_asset,
        condition_number=cond,
        min_eigenvalue=min_eig,
        is_psd=min_eig >= -tol,
        effective_observations=float(eff_obs),
        warnings=tuple(msgs),
    )


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def _sample(returns: pd.DataFrame, ddof: int = 1) -> pd.DataFrame:
    return returns.cov(ddof=ddof)


def _ledoit_wolf(returns: pd.DataFrame) -> pd.DataFrame:
    from sklearn.covariance import LedoitWolf

    cov = LedoitWolf().fit(returns.values).covariance_
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def _oas(returns: pd.DataFrame) -> pd.DataFrame:
    from sklearn.covariance import OAS

    cov = OAS().fit(returns.values).covariance_
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def _ewma(returns: pd.DataFrame, lam: float = 0.94, demean: bool = True) -> pd.DataFrame:
    """Exponentially weighted (RiskMetrics-style) covariance.

    ``demean=False`` reproduces the original RiskMetrics recursion, which
    treats the conditional mean as zero — the usual choice for daily data,
    where the mean is small relative to the noise in estimating it.
    """
    r = returns - returns.mean() if demean else returns
    weights = (1 - lam) * lam ** np.arange(len(r))[::-1]
    weights /= weights.sum()
    cov = (r.T * weights) @ r
    return cov


def _semi(returns: pd.DataFrame, mar: float | pd.Series | None = None) -> pd.DataFrame:
    """Semicovariance from downside deviations below a minimum acceptable return.

    ``Σ_semi = (1/(T−1)) · Dᵀ D`` with ``D = min(r − MAR, 0)``.

    The deviations are *not* re-centred: ``D`` already measures shortfall
    against the MAR, so subtracting its own (negative) mean would understate
    downside co-movement. ``mar`` defaults to the per-asset sample mean,
    which is the Markowitz (1959) semivariance convention; pass ``0.0`` for
    the zero-threshold convention, or a Series for per-asset thresholds.
    """
    threshold = returns.mean() if mar is None else mar
    deviation = returns.sub(threshold, axis=1) if not np.isscalar(threshold) else returns - threshold
    deviation = deviation.where(deviation < 0, 0.0)
    n = len(deviation)
    denom = max(n - 1, 1)
    cov = deviation.T.values @ deviation.values / denom
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def covariance_matrix(
    returns: pd.DataFrame,
    method: CovarianceMethod = "ledoit_wolf",
    annualize: bool = True,
    periods_per_year: int = 252,
    ewma_lambda: float = 0.94,
    semi_mar: float | pd.Series | None = None,
    ensure_psd: bool = True,
    denoise: bool = False,
    denoise_method: str = "constant_residual",
    denoise_alpha: float = 0.0,
    detone: int = 0,
) -> pd.DataFrame:
    """Estimate a covariance matrix on returns.

    Set ``annualize=True`` to scale by ``periods_per_year`` (e.g. daily ⇒ annual).

    Args:
        returns: Wide frame of periodic returns, one column per asset.
        method: Estimator name; see :data:`COVARIANCE_DESCRIPTIONS`.
        annualize: Scale the result by ``periods_per_year``.
        periods_per_year: Observations per year in ``returns``.
        ewma_lambda: Decay factor when ``method == "ewma"``.
        semi_mar: Minimum acceptable return for ``method == "semi"``.
            ``None`` uses the per-asset sample mean.
        ensure_psd: Repair the estimate with :func:`nearest_psd`. Leave on
            unless you specifically want the raw estimator output.
        denoise: Apply the Marchenko-Pastur eigenvalue filter after
            estimating (López de Prado, 2020). Implied by
            ``method="denoised"``, and composable with any other estimator —
            denoising a Ledoit-Wolf matrix is legitimate, if belt-and-braces.
            The noise edge is fitted against the sample the estimator
            actually leant on, not the row count: under ``method="ewma"``
            that is ``round(1/(1−ewma_lambda))`` — about 17 at the default
            decay — so the cutoff sits far higher than it would on the same
            panel estimated equally-weighted, and a universe wider than that
            effective sample cannot be denoised at all. See
            :func:`effective_sample_size` and
            :func:`~optimization_engine.data.denoise.denoise_covariance`.
        denoise_method: ``"constant_residual"`` or ``"targeted_shrinkage"``;
            see :func:`~optimization_engine.data.denoise.denoise_correlation`.
        denoise_alpha: Noise-block shrinkage retained under
            ``"targeted_shrinkage"``.
        detone: Remove this many leading eigenvectors (the market component)
            after denoising. Non-zero values make the result **singular** —
            correct for clustering methods, fatal for anything that inverts
            the matrix.

    Returns:
        The estimate. When denoising ran, the :class:`DenoiseReport` is
        attached as ``result.attrs["denoise_report"]`` so callers that want
        the spectrum diagnostics can reach them without a second API.

    Raises:
        ValueError: If ``returns`` is empty or ``method`` is unknown.
    """
    if returns is None or returns.empty:
        raise ValueError("Cannot estimate a covariance matrix from empty returns.")
    if returns.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 return observations to estimate covariance; got "
            f"{returns.shape[0]}."
        )
    if returns.isna().any().any():
        raise ValueError(
            "Returns contain missing values. Align or fill the price panel "
            "before estimating covariance - silently dropping rows would "
            "change the sample each asset is estimated on."
        )

    if method in ("sample", "denoised"):
        cov = _sample(returns)
    elif method == "ledoit_wolf":
        cov = _ledoit_wolf(returns)
    elif method == "oas":
        cov = _oas(returns)
    elif method == "shrink":
        # An alias, not a third shrinkage estimator — see the module docstring.
        cov = _ledoit_wolf(returns)
    elif method == "ewma":
        cov = _ewma(returns, lam=ewma_lambda)
    elif method == "semi":
        cov = _semi(returns, mar=semi_mar)
    else:
        raise ValueError(
            f"Unknown covariance method: {method!r}. "
            f"Available: {sorted(COVARIANCE_DESCRIPTIONS)}"
        )

    if annualize:
        cov = cov * periods_per_year

    report = None
    if denoise or method == "denoised" or detone:
        from optimization_engine.data.denoise import denoise_covariance

        n_effective, note = effective_sample_size(
            len(returns), method=method, ewma_lambda=ewma_lambda
        )
        cov, report = denoise_covariance(
            cov,
            n_observations=n_effective,
            method=denoise_method,
            alpha=denoise_alpha,
            detone=detone,
            n_observations_nominal=len(returns),
            effective_sample_note=note,
        )

    if ensure_psd:
        cov = nearest_psd(cov)
    if report is not None:
        cov.attrs["denoise_report"] = report
    return cov


def covariance_from_config(
    returns: pd.DataFrame, config, ensure_psd: bool = True
) -> pd.DataFrame:
    """Estimate the covariance the way one :class:`EngineConfig` asks for.

    Every part of the engine that re-estimates a covariance — the frontier
    sweep, the bootstrap, the walk-forward, the CLI, the UI — has to make the
    same choices about estimator, annualization, decay and denoising. Doing
    that inline five times is how a run ends up bootstrapping a different
    matrix than it optimized. This is the single place those choices live.

    Args:
        returns: Periodic returns, one column per asset.
        config: The configuration supplying the estimator, the annualization
            basis, the EWMA decay and the denoising settings.
        ensure_psd: Repair the estimate with :func:`nearest_psd`. Off only
            for a caller that diagnoses the raw estimate and repairs it
            itself, as :func:`~optimization_engine.engine.run_engine` does.

    Returns:
        The annualized covariance, indexed and columned by asset.
    """
    return covariance_matrix(
        returns,
        method=config.covariance_method,
        periods_per_year=config.periods_per_year,
        ewma_lambda=config.ewma_lambda,
        denoise=getattr(config, "denoise", False),
        denoise_method=getattr(config, "denoise_method", "constant_residual"),
        denoise_alpha=getattr(config, "denoise_alpha", 0.0),
        detone=getattr(config, "detone", 0),
        ensure_psd=ensure_psd,
    )


# ---------------------------------------------------------------------------
# Expected returns
# ---------------------------------------------------------------------------

EXPECTED_RETURN_DESCRIPTIONS: dict[str, str] = {
    "mean": (
        "Annualized arithmetic mean of realized returns - the single-period "
        "expected return mean-variance optimization is defined against. "
        "Simple, but the standard error of a mean estimate is huge: decades "
        "of data are needed to distinguish two assets' means."
    ),
    "geometric_mean": (
        "Annualized geometric (compound) mean of realized returns. This is "
        "the growth rate an investor actually earned, and it is the right "
        "number for a multi-period question - but it is not the mu a "
        "single-period mean-variance model wants, and pairing it with an "
        "arithmetic covariance understates every expected return by roughly "
        "half the variance."
    ),
    "ema": (
        "Exponentially-weighted mean: recent returns count more. Responsive "
        "but even noisier than the full-sample mean."
    ),
    "capm": (
        "CAPM-implied returns: rf + beta*(market premium). Cross-sectionally "
        "disciplined, so far more stable than raw historical means."
    ),
    "shrunk_mean": (
        "James-Stein shrinkage of the arithmetic historical means toward the "
        "grand mean. Keeps the ranking information while pulling in the "
        "extremes that drive mean-variance to corner solutions."
    ),
}


def james_stein_shrinkage(
    sample_mean: pd.Series,
    cov_matrix: pd.DataFrame,
    n_observations: int,
    target: float | None = None,
    periods_per_year: int = 252,
) -> tuple[pd.Series, float]:
    """Shrink a vector of sample means toward a common target.

    Implements the Jorion (1986) Bayes-Stein estimator: the shrinkage
    intensity grows with the number of assets and the dispersion of the
    covariance, and shrinks as the sample lengthens.

    The intensity is Jorion's ``λ = (N+2) / ((N+2) + T·(μ−μ₀)'Σ⁻¹(μ−μ₀))``,
    a statistic of the *per-observation* moments: ``T`` counts observations,
    so the quadratic form has to be in the units of one observation too.
    Annualized inputs inflate it by ``periods_per_year`` — the means scale by
    the basis and the covariance by the basis, so their ratio does — and
    with daily data that made the shrinkage some 250 times too weak. The
    quadratic form is therefore divided by ``periods_per_year`` before the
    intensity is formed.

    Args:
        sample_mean: Annualized sample means, one per asset.
        cov_matrix: Annualized covariance matrix for the same assets.
        n_observations: Number of return observations behind the means.
        target: Grand mean to shrink toward. ``None`` uses the
            minimum-variance portfolio's expected return, which is the
            Jorion prior.
        periods_per_year: The annualization basis of ``sample_mean`` and
            ``cov_matrix`` — 252 for daily inputs, 12 for monthly, ``1``
            when they are already per-period.

    **A degenerate quadratic form reports zero intensity, not one.** When
    ``(μ−μ₀)'Σ⁻¹(μ−μ₀)`` comes back non-positive — every mean already sits on
    the target, or the pseudo-inverse of a singular Σ annihilates the
    deviation — the formula's analytic limit is ``λ = 1``. But ``λ`` is not
    the answer here; it is a description of what the estimator did, and what
    it did was return the sample means untouched. Reporting full shrinkage
    beside an unshrunk vector is a lie about the estimator's own output, and
    a caller that logs the intensity or gates on it would draw exactly the
    wrong conclusion. The :class:`numpy.linalg.LinAlgError` exit below
    already reports ``0.0`` for the same reason; this matches it.

    Returns:
        ``(shrunk_means, intensity)`` where ``intensity`` is in ``[0, 1]``.
        The intensity is the share of the returned vector that came from the
        target, so a returned ``sample_mean`` always carries ``0.0``.
    """
    mu = sample_mean.astype(float)
    assets = list(mu.index)
    sigma = cov_matrix.reindex(assets, axis=0).reindex(assets, axis=1).values
    n = len(assets)
    if n < 3 or n_observations <= n + 2:
        # Stein shrinkage has no advantage below 3 assets and the intensity
        # formula is undefined once the sample is too short.
        return mu, 0.0

    try:
        inv = np.linalg.pinv(sigma)
        ones = np.ones(n)
        denom = float(ones @ inv @ ones)
        mu_target = (
            float(target)
            if target is not None
            else float(ones @ inv @ mu.values) / denom
        )
        diff = mu.values - mu_target
        quad = float(diff @ inv @ diff) / float(periods_per_year)
        # A deviation at the level of floating-point residue is not a
        # deviation. Testing ``quad <= 0`` alone made this branch depend on an
        # exact cancellation: when every mean already sits on the target, the
        # target is that mean recomputed through ``pinv``, so ``diff`` comes
        # back as either exactly zero or a few ulps, depending on the BLAS.
        # Both are the same estimator doing the same nothing, and only one of
        # them used to be reported as such — the other returned an intensity
        # of 1.0 beside an unshrunk vector, which is the very claim this
        # branch exists to prevent.
        scale = float(np.max(np.abs(mu.values))) or 1.0
        if quad <= 0 or not bool(np.any(np.abs(diff) > 1e-12 * scale)):
            # Nothing was shrunk — see the docstring. The analytic limit of
            # the formula is 1.0, but the intensity reports what happened to
            # the vector, and what happened to it was nothing.
            return mu, 0.0
        lam = (n + 2) / (n + 2 + n_observations * quad)
    except np.linalg.LinAlgError:
        return mu, 0.0

    intensity = float(np.clip(lam, 0.0, 1.0))
    shrunk = (1 - intensity) * mu.values + intensity * mu_target
    return pd.Series(shrunk, index=assets, name=mu.name), intensity


def expected_returns_from_history(
    returns: pd.DataFrame,
    method: ExpectedReturnMethod = "mean",
    periods_per_year: int = 252,
    span: int = 180,
    market_return: float | None = None,
    risk_free_rate: float = 0.0,
    market_weights: pd.Series | None = None,
    cov_matrix: pd.DataFrame | None = None,
) -> pd.Series:
    """Build an expected-return vector from realized history.

    **The single definition of μ in this library.** Every other place that
    needs expected returns — the app, the doc-image script, the resampler,
    the shrinkage and CAPM branches below — calls this rather than writing
    the formula out again, so there is one place where the convention lives
    and one place to change it.

    * ``mean``           — annualized *arithmetic* historical mean,
      ``r̄ · periods_per_year``.
    * ``geometric_mean`` — annualized compound growth rate,
      ``(∏(1+r))^(ppy/T) − 1``, with ``T`` counted per asset so a series that
      starts late is compounded over its own history.
    * ``ema``            — exponentially-weighted arithmetic mean with the
      given ``span``, weights normalized over the window, annualized as
      ``ema · periods_per_year`` like ``mean``.
    * ``capm``           — implied returns from a single-factor CAPM where
      the market portfolio is approximated by ``market_weights`` (defaulting
      to equal weights).
    * ``shrunk_mean``    — ``mean`` pulled toward the minimum-variance
      portfolio's return via :func:`james_stein_shrinkage`.

    **Why ``mean`` is arithmetic.** Mean-variance optimization is a
    single-period model: it maximizes ``μ'w − λ·w'Σw`` over one period's
    return, and the expectation of a one-period return is its arithmetic
    mean. The geometric mean is the *realized* compound growth rate, lower
    than the arithmetic mean by roughly ``σ²/2``, and it is the answer to a
    different question — what an investor earned over many periods. Pairing a
    geometric μ with an arithmetic Σ is not conservative, it is inconsistent:
    the penalty term is measured on one convention and the reward term on
    another, and the size of the error is exactly the quantity the optimizer
    is trading off. ``geometric_mean`` is kept, named for what it is, for
    callers who want that number deliberately.

    Args:
        returns: Periodic returns, one column per asset.
        method: Which of the five estimators above to use.
        periods_per_year: Annualization basis for the result.
        span: EWMA span in periods. Only used by ``ema``.
        market_return: The market's expected return, for ``capm``. Estimated
            from the market portfolio when omitted.
        risk_free_rate: Annualized risk-free rate, used by ``capm``.
        market_weights: The market portfolio for ``capm``. Defaults to equal
            weights, which is an assumption rather than a neutral choice.
        cov_matrix: A pre-computed covariance, needed by ``capm`` and
            ``shrunk_mean``. Estimated from ``returns`` when omitted.

    Returns:
        Annualized expected returns, one per column of ``returns``.

    Raises:
        ValueError: On an unknown ``method``, empty ``returns``, CAPM market
            weights summing to zero, or a CAPM market portfolio with zero
            variance, which leaves the betas undefined.
    """

    if returns is None or returns.empty:
        raise ValueError("Cannot estimate expected returns from empty data.")

    if method == "mean":
        return returns.mean() * periods_per_year
    if method == "geometric_mean":
        # ``prod`` skips a missing return, so the exponent has to count per
        # column too: ``len`` read a series that listed halfway through as
        # having compounded its growth over the whole panel.
        return ((1 + returns).prod() ** (periods_per_year / returns.count())) - 1
    if method == "ema":
        # ``adjust=True`` normalizes the weights over the window. The
        # recursive form (``adjust=False``) seeds itself with the first row and
        # leaves it all the weight the decay has not yet handed on —
        # ``(1 − α)^(T−1)``, 0.774 on two years of months at the default span.
        # Annualized as ``mean`` is, for the same single-period reason.
        ema = returns.ewm(span=span, adjust=True).mean().iloc[-1]
        return ema * periods_per_year
    if method == "shrunk_mean":
        raw = expected_returns_from_history(
            returns, method="mean", periods_per_year=periods_per_year
        )
        if cov_matrix is None:
            cov_matrix = covariance_matrix(
                returns, method="ledoit_wolf", periods_per_year=periods_per_year
            )
        shrunk, _ = james_stein_shrinkage(
            raw, cov_matrix, len(returns), periods_per_year=periods_per_year
        )
        return shrunk
    if method == "capm":
        if cov_matrix is None:
            cov_matrix = covariance_matrix(
                returns, method="ledoit_wolf", periods_per_year=periods_per_year
            )
        if market_weights is None:
            market_weights = pd.Series(
                np.ones(returns.shape[1]) / returns.shape[1], index=returns.columns
            )
        market_weights = market_weights.reindex(returns.columns).fillna(0.0)
        if float(market_weights.sum()) <= 0:
            raise ValueError(
                "CAPM market weights sum to zero - cannot define a market "
                "portfolio."
            )
        market_weights = market_weights / float(market_weights.sum())
        market_return_est = market_return
        if market_return_est is None:
            # The market premium is multiplied by a beta and read as a
            # single-period expectation, so it has to be the arithmetic mean
            # for the same reason ``mean`` is — and it goes through the same
            # code path so it cannot drift from it.
            mkt = (returns * market_weights).sum(axis=1)
            market_return_est = float(
                expected_returns_from_history(
                    mkt.to_frame("market"),
                    method="mean",
                    periods_per_year=periods_per_year,
                ).iloc[0]
            )
        market_var = float(market_weights.values @ cov_matrix.values @ market_weights.values)
        if market_var <= 0:
            raise ValueError(
                "CAPM market portfolio has zero variance; betas are undefined."
            )
        betas = (cov_matrix.values @ market_weights.values) / market_var
        return pd.Series(
            risk_free_rate + betas * (market_return_est - risk_free_rate),
            index=returns.columns,
        )
    raise ValueError(
        f"Unknown expected-return method: {method!r}. "
        f"Available: {sorted(EXPECTED_RETURN_DESCRIPTIONS)}"
    )
