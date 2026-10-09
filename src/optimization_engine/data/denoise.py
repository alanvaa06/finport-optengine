"""Random-matrix-theory denoising and detoning of correlation matrices.

Shrinkage estimators (Ledoit-Wolf, OAS) pull *every* entry of the covariance
toward a target. That is a blunt instrument: it treats the part of the matrix
that carries real structure — the market factor, the sector blocks — exactly
like the part that is sampling noise, so conditioning improves at the cost of
attenuating the signal too.

Random matrix theory offers a sharper cut. López de Prado (*Machine Learning
for Asset Managers*, CUP 2020, ch. 2) observes that the eigenvalues of a
correlation matrix estimated from ``T`` observations of ``N`` assets split
into two populations:

* eigenvalues below the Marchenko-Pastur upper edge ``λ₊``, which are
  statistically indistinguishable from those of a matrix of pure noise, and
* eigenvalues above it, which carry the factor structure.

**Denoising** replaces the noise eigenvalues with their common average (or
shrinks them toward it), leaving the signal eigenvectors untouched.
Conditioning improves dramatically — the near-zero eigenvalues that make
mean-variance weights explode are exactly the ones being replaced — while the
factor structure survives intact.

**Detoning** goes one step further and *removes* the first eigenvector, the
one every asset loads on. What is left is the correlation structure after the
market has been taken out, which is what the clustering methods (HRP, HERC,
NCO) actually want to see: with the market component in place, every pair of
equities looks similar and the hierarchy degenerates.

The two operations answer different questions and compose in one direction
only: denoise first (to decide which eigenvalues are signal), then detone.

References:
    López de Prado, M. (2020). *Machine Learning for Asset Managers*.
    Cambridge University Press, chapter 2.

    Laloux, L., Cizeau, P., Bouchaud, J-P. and Potters, M. (1999). "Noise
    dressing of financial correlation matrices". *Physical Review Letters*
    83(7).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Denoising strategies for the sub-``λ₊`` eigenvalues.
DENOISE_METHODS = ("constant_residual", "targeted_shrinkage")


def cov_to_corr(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a covariance matrix into ``(correlation, standard deviations)``.

    Args:
        cov: A covariance matrix.

    Returns:
        A ``(correlation, std)`` pair, the second in the covariance's own
        volatility units.

    Raises:
        ValueError: If any asset has zero or negative variance, which makes
            the correlation undefined rather than merely awkward.
    """
    std = np.sqrt(np.diag(cov))
    if not (std > 0).all():
        raise ValueError(
            "Cannot convert covariance to correlation: at least one asset has "
            "zero variance. Drop constant series before denoising."
        )
    corr = cov / np.outer(std, std)
    return np.clip(corr, -1.0, 1.0), std


def corr_to_cov(corr: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Rebuild a covariance matrix from a correlation and its volatilities.

    Args:
        corr: A correlation matrix.
        std: Per-asset standard deviations, aligned to it.

    Returns:
        The covariance matrix, in the units ``std`` carries.
    """
    return corr * np.outer(std, std)


def _rescale_to_unit_diagonal(matrix: np.ndarray) -> np.ndarray:
    """Renormalize a matrix so its diagonal is exactly 1.

    Both denoising and detoning change the diagonal (they redistribute or
    remove variance), and a "correlation" matrix whose diagonal is 0.83 will
    silently rescale every volatility downstream. This puts it back.
    """
    diag = np.diag(matrix).copy()
    diag[diag <= 0] = 1.0
    scale = np.sqrt(diag)
    out = matrix / np.outer(scale, scale)
    np.fill_diagonal(out, 1.0)
    return np.clip((out + out.T) / 2.0, -1.0, 1.0)


def _sorted_eigen(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition with eigenvalues in *descending* order."""
    eigenvalue, eigenvector = np.linalg.eigh((matrix + matrix.T) / 2.0)
    order = np.argsort(eigenvalue)[::-1]
    return eigenvalue[order], eigenvector[:, order]


# ---------------------------------------------------------------------------
# The Marchenko-Pastur law
# ---------------------------------------------------------------------------


def marchenko_pastur_pdf(
    variance: float, q: float, n_points: int = 1000
) -> pd.Series:
    """Density of the Marchenko-Pastur law on its support.

    This is the eigenvalue distribution of a correlation matrix built from
    ``T`` observations of ``N`` *independent* series with common variance
    ``variance``. Everything inside the support ``[λ₋, λ₊]`` is what pure
    noise looks like.

    Args:
        variance: Variance of the underlying noise. ``1.0`` for a correlation
            matrix with no signal at all; the fitted value is lower whenever
            some of the variance has been claimed by real factors.
        q: The aspect ratio ``T / N``. Must exceed 1 — with fewer
            observations than assets the sample matrix is singular and the
            law's support collapses onto zero.
        n_points: Resolution of the returned density.

    Returns:
        Density indexed by eigenvalue.

    Raises:
        ValueError: If ``q <= 1`` or ``variance <= 0``.
    """
    if q <= 1:
        raise ValueError(
            f"The Marchenko-Pastur fit needs more observations than assets "
            f"(T/N > 1); got T/N = {q:.3f}. Shorten the universe or lengthen "
            "the history."
        )
    if variance <= 0:
        raise ValueError(f"Noise variance must be positive; got {variance}.")

    lambda_min = variance * (1.0 - np.sqrt(1.0 / q)) ** 2
    lambda_max = variance * (1.0 + np.sqrt(1.0 / q)) ** 2
    grid = np.linspace(lambda_min, lambda_max, int(n_points))
    density = q / (2.0 * np.pi * variance * grid) * np.sqrt(
        np.clip((lambda_max - grid) * (grid - lambda_min), 0.0, None)
    )
    return pd.Series(density, index=grid, name="mp_pdf")


def _kde_density(
    observations: np.ndarray, bandwidth: float, grid: np.ndarray
) -> np.ndarray:
    from sklearn.neighbors import KernelDensity

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(np.asarray(observations, dtype=float).reshape(-1, 1))
    return np.exp(kde.score_samples(np.asarray(grid, dtype=float).reshape(-1, 1)))


def fit_marchenko_pastur(
    eigenvalues: np.ndarray, q: float, bandwidth: float = 0.25
) -> tuple[float, float]:
    """Fit the noise variance implied by an observed eigenvalue spectrum.

    The naive cutoff assumes the whole matrix is noise (``σ² = 1``), which
    over-states ``λ₊`` and throws away real factors. Instead the noise
    variance is fitted by matching a kernel density of the observed
    eigenvalues to the Marchenko-Pastur density — the procedure in *Machine
    Learning for Asset Managers*, §2.5.

    Args:
        eigenvalues: Observed eigenvalues of a **correlation** matrix.
        q: ``T / N``.
        bandwidth: Kernel bandwidth for the empirical density. Larger values
            smooth over the fine structure of the spectrum; 0.25 is López de
            Prado's default and is stable for typical panel sizes.

    Returns:
        ``(fitted_variance, lambda_max)``. ``lambda_max`` is the cutoff above
        which an eigenvalue is treated as signal. When the fit does not
        converge the variance falls back to 1.0 — the pure-noise edge, the
        widest cutoff there is — and a ``UserWarning`` says so.
    """
    variance, failure = _fit_noise_variance(eigenvalues, q, bandwidth)
    if failure:
        _warn_fit_failed(failure)
    lambda_max = variance * (1.0 + np.sqrt(1.0 / q)) ** 2
    return variance, float(lambda_max)


def _warn_fit_failed(reason: str) -> None:
    warnings.warn(
        f"The Marchenko-Pastur fit did not converge ({reason}), so the noise "
        "variance fell back to 1.0. That is the widest noise edge there is: "
        "weak factors below it are treated as noise.",
        stacklevel=3,
    )


def _fit_noise_variance(
    eigenvalues: np.ndarray, q: float, bandwidth: float
) -> tuple[float, str]:
    """The fitted ``σ²``, and why the fit failed — empty when it did not.

    Split from :func:`fit_marchenko_pastur` so the fallback can travel to the
    :class:`DenoiseReport` as a flag rather than only as a value of 1.0 that
    is indistinguishable from a genuinely pure-noise panel.
    """
    from scipy.optimize import minimize_scalar

    observed = np.asarray(eigenvalues, dtype=float)

    def sse(variance: float) -> float:
        """Squared error between the empirical and theoretical eigenvalue densities.

        Args:
            variance: Candidate noise variance to evaluate.

        Returns:
            The sum of squared differences. This is the objective the fit
            minimizes over.
        """
        theoretical = marchenko_pastur_pdf(float(variance), q)
        empirical = _kde_density(observed, bandwidth, theoretical.index.values)
        return float(np.sum((empirical - theoretical.values) ** 2))

    try:
        found = minimize_scalar(sse, bounds=(1e-4, 1.0 - 1e-4), method="bounded")
    except Exception as exc:  # pragma: no cover - optimizer edge cases
        return 1.0, f"{type(exc).__name__}: {exc}"
    if not found.success:
        return 1.0, "the optimizer reported no success"
    return float(found.x), ""


# ---------------------------------------------------------------------------
# Denoising / detoning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DenoiseReport:
    """What the random-matrix filter found and what it did about it.

    Attributes:
        n_assets: Universe size ``N``.
        n_observations: Nominal sample length — the number of rows the
            estimator saw.
        n_observations_effective: The ``T`` the Marchenko-Pastur edge was
            actually computed from, which is what ``q`` divides. Equal to
            ``n_observations`` for an equally-weighted estimator; strictly
            smaller under EWMA, where the exponential decay means the fit
            rests on roughly ``1 / (1 − λ)`` observations no matter how long
            the panel is.
        effective_sample_note: Why the two differ, in words, or empty when
            they do not.
        q: ``T_effective / N``.
        noise_variance: Fitted Marchenko-Pastur variance ``σ²``. Well below
            1 means most of the matrix's variance is real structure; close to
            1 means the panel is close to noise.
        eigenvalue_cutoff: The fitted ``λ₊``.
        n_signal_eigenvalues: How many eigenvalues sit above the cutoff — the
            number of factors the data can actually support. Zero on a panel
            of pure noise; the caller's ``n_signal`` when one was given.
        n_factors_kept: How many the filter actually kept as signal. Equal to
            ``n_signal_eigenvalues`` unless that was forced into ``[1, N−1]``:
            the leading eigenvalue is always kept, because filtering all of
            them would erase the correlation structure, and keeping all would
            make the filter a no-op.
        signal_share: Fraction of total variance carried by the kept factors.
        noise_fit_failed: The Marchenko-Pastur fit did not converge, so
            ``noise_variance`` is the 1.0 fallback rather than an estimate —
            the widest noise edge, which can class weak factors as noise.
        method: Which denoising rule was applied.
        detoned_factors: How many leading eigenvectors were removed.
        condition_before: Condition number of the input matrix — the
            covariance when the caller went through
            :func:`denoise_covariance`, so it is directly comparable to what
            :func:`~optimization_engine.data.covariance.covariance_diagnostics`
            reports.
        condition_after: The same after filtering.
        correlation_condition_before: Condition number of the *correlation*
            matrix before filtering.
        correlation_condition_after: And after.

    Reporting both pairs separates two different causes of an
    ill-conditioned covariance. Denoising only ever acts on the correlation,
    so it is the correlation pair that measures whether it worked. A
    covariance whose condition number barely moves — or rises — after a large
    improvement in the correlation is telling you its conditioning was never
    about correlation noise in the first place: it comes from the spread of
    the volatilities themselves, and no eigenvalue filter will touch that.
    """

    n_assets: int
    n_observations: int
    q: float
    noise_variance: float
    eigenvalue_cutoff: float
    n_signal_eigenvalues: int
    signal_share: float
    method: str
    detoned_factors: int
    condition_before: float
    condition_after: float
    correlation_condition_before: float = float("nan")
    correlation_condition_after: float = float("nan")
    n_observations_effective: int = -1
    effective_sample_note: str = ""
    n_factors_kept: int = -1
    noise_fit_failed: bool = False

    def __post_init__(self) -> None:
        """Default the effective sample to the nominal one.

        Only a caller that knows the estimator down-weights its own history
        (EWMA) can tell the two apart, so the sentinel means "nobody said
        otherwise" rather than "unknown". ``n_factors_kept`` defaults the same
        way, to the fitted count.
        """
        if self.n_observations_effective < 0:
            object.__setattr__(
                self, "n_observations_effective", int(self.n_observations)
            )
        if self.n_factors_kept < 0:
            object.__setattr__(
                self, "n_factors_kept", int(self.n_signal_eigenvalues)
            )

    def describe(self) -> str:
        """One-paragraph summary suitable for a UI panel or a report cell."""
        sample = f"{self.n_observations} observations"
        if self.n_observations_effective != self.n_observations:
            sample = (
                f"{self.n_observations_effective} effective observations "
                f"(of {self.n_observations} rows)"
            )
        line = (
            f"Marchenko-Pastur fit on {sample} of "
            f"{self.n_assets} assets (T/N = {self.q:.1f}) put the noise edge at "
            f"lambda+ = {self.eigenvalue_cutoff:.3f}. "
            f"{self.n_signal_eigenvalues} of {self.n_assets} eigenvalues sit "
            "above it"
        )
        if self.n_factors_kept != self.n_signal_eigenvalues:
            line += (
                f", but {self.n_factors_kept} were kept anyway - the filter "
                "never treats the whole matrix as noise, nor none of it"
            )
        line += (
            f". The kept factors carry {self.signal_share:.1%} of total "
            f"variance; the remaining {self.n_assets - self.n_factors_kept} "
            f"eigenvalues were treated as noise "
            f"({self.method.replace('_', ' ')}). "
            f"The correlation's condition number went "
            f"{self.correlation_condition_before:.3g} -> "
            f"{self.correlation_condition_after:.3g}; the covariance's went "
            f"{self.condition_before:.3g} -> {self.condition_after:.3g}"
        )
        if (
            np.isfinite(self.condition_after)
            and np.isfinite(self.correlation_condition_after)
            and self.correlation_condition_after
            < 0.9 * self.correlation_condition_before
            and self.condition_after > 0.9 * self.condition_before
        ):
            line += (
                " - so this covariance's conditioning is driven by the spread "
                "of the volatilities, not by correlation noise, and no "
                "eigenvalue filter will improve it."
            )
        else:
            line += "."
        if self.noise_fit_failed:
            line += (
                " The Marchenko-Pastur fit did not converge, so the noise "
                "variance is the 1.0 fallback rather than an estimate - the "
                "widest edge there is, which can class weak factors as noise."
            )
        if self.effective_sample_note:
            line += " " + self.effective_sample_note
        if self.detoned_factors:
            line += (
                f" The top {self.detoned_factors} eigenvector(s) were then "
                "removed, so the result is a market-neutral correlation and is "
                "singular by construction - use it for clustering and distance, "
                "not for anything that inverts it."
            )
        return line


def denoise_correlation(
    corr: np.ndarray,
    q: float,
    method: str = "constant_residual",
    bandwidth: float = 0.25,
    alpha: float = 0.0,
    n_signal: int | None = None,
) -> tuple[np.ndarray, DenoiseReport]:
    """Filter the noise eigenvalues out of a correlation matrix.

    Args:
        corr: Correlation matrix (unit diagonal).
        q: ``T / N``, the sample aspect ratio.
        method:
            ``"constant_residual"`` — replace every sub-``λ₊`` eigenvalue with
            their common mean. Trace-preserving, and the strongest available
            improvement in conditioning.
            ``"targeted_shrinkage"`` — shrink the noise *sub-matrix* toward
            its own diagonal by ``1 − alpha``, leaving the signal block alone.
            Gentler: use it when you suspect the cutoff has classified some
            weak-but-real factors as noise.
        bandwidth: Kernel bandwidth for the density fit.
        alpha: Shrinkage retained on the noise block under
            ``"targeted_shrinkage"``. ``0`` removes the noise correlations
            entirely; ``1`` is a no-op.
        n_signal: Override the fitted number of signal eigenvalues. Useful
            when a factor model already tells you how many factors there are.

    Returns:
        ``(denoised_correlation, report)``.

    Raises:
        ValueError: On an unknown ``method`` or an out-of-range ``alpha``.
    """
    if method not in DENOISE_METHODS:
        raise ValueError(
            f"Unknown denoising method {method!r}. Available: {list(DENOISE_METHODS)}"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1]; got {alpha}.")

    corr = np.asarray(corr, dtype=float)
    n = corr.shape[0]
    eigenvalue, eigenvector = _sorted_eigen(corr)
    cond_before = _condition(eigenvalue)

    variance, failure = _fit_noise_variance(eigenvalue, q, bandwidth)
    if failure:
        _warn_fit_failed(failure)
    cutoff = variance * (1.0 + np.sqrt(1.0 / q)) ** 2
    if n_signal is None:
        n_found = int((eigenvalue > cutoff).sum())
    else:
        n_found = int(np.clip(n_signal, 0, n))
    # Keeping every eigenvalue would make the filter a no-op; keeping none
    # would erase the correlation structure entirely. Both are degenerate
    # readings of a fit, so the market factor is always retained — and the
    # report carries the count the fit found beside the one that was kept.
    n_facts = int(np.clip(n_found, 1, n - 1)) if n > 1 else n

    filtered = eigenvalue.copy()
    if method == "constant_residual":
        residual = eigenvalue[n_facts:]
        if residual.size:
            filtered[n_facts:] = residual.sum() / residual.size
        denoised = eigenvector @ np.diag(filtered) @ eigenvector.T
    else:
        signal_vec, signal_val = eigenvector[:, :n_facts], eigenvalue[:n_facts]
        noise_vec, noise_val = eigenvector[:, n_facts:], eigenvalue[n_facts:]
        corr_signal = signal_vec @ np.diag(signal_val) @ signal_vec.T
        corr_noise = noise_vec @ np.diag(noise_val) @ noise_vec.T
        denoised = (
            corr_signal
            + alpha * corr_noise
            + (1.0 - alpha) * np.diag(np.diag(corr_noise))
        )

    denoised = _rescale_to_unit_diagonal(denoised)
    report = DenoiseReport(
        n_assets=n,
        n_observations=int(round(q * n)),
        q=float(q),
        noise_variance=float(variance),
        eigenvalue_cutoff=float(cutoff),
        n_signal_eigenvalues=n_found,
        n_factors_kept=n_facts,
        noise_fit_failed=bool(failure),
        signal_share=float(eigenvalue[:n_facts].sum() / eigenvalue.sum())
        if eigenvalue.sum() > 0
        else float("nan"),
        method=method,
        detoned_factors=0,
        condition_before=cond_before,
        condition_after=_condition(_sorted_eigen(denoised)[0]),
        correlation_condition_before=cond_before,
        correlation_condition_after=_condition(_sorted_eigen(denoised)[0]),
    )
    return denoised, report


def detone_correlation(
    corr: np.ndarray, n_factors: int = 1
) -> np.ndarray:
    """Remove the leading eigenvector(s) — the market component.

    Every equity loads on the market, so the first eigenvector dominates the
    correlation matrix and every pair looks alike. Clustering on the raw
    matrix therefore produces a hierarchy driven by beta rather than by the
    sector and style structure the method is supposed to find. Detoning
    removes that component before the tree is built.

    The result is **singular** by construction: ``n_factors`` eigenvalues have
    been set to zero. That is fine for distance and clustering, and fatal for
    anything that inverts the matrix.

    Args:
        corr: Correlation matrix, ideally already denoised.
        n_factors: How many leading eigenvectors to strip. ``1`` (the market)
            is almost always the right answer.

    Raises:
        ValueError: If ``n_factors`` is not in ``[1, N-1]``.
    """
    corr = np.asarray(corr, dtype=float)
    n = corr.shape[0]
    if not 1 <= n_factors < n:
        raise ValueError(
            f"Can remove between 1 and {n - 1} factors from a {n}-asset "
            f"correlation; got {n_factors}."
        )
    eigenvalue, eigenvector = _sorted_eigen(corr)
    market_vec = eigenvector[:, :n_factors]
    market_val = np.diag(eigenvalue[:n_factors])
    detoned = corr - market_vec @ market_val @ market_vec.T
    return _rescale_to_unit_diagonal(detoned)


def denoise_covariance(
    cov: pd.DataFrame,
    n_observations: int,
    method: str = "constant_residual",
    bandwidth: float = 0.25,
    alpha: float = 0.0,
    detone: int = 0,
    n_signal: int | None = None,
    n_observations_nominal: int | None = None,
    effective_sample_note: str = "",
) -> tuple[pd.DataFrame, DenoiseReport]:
    """Denoise (and optionally detone) a covariance matrix.

    The filtering happens on the correlation matrix — eigenvalue cutoffs are
    only comparable once the volatilities are scaled out — and the original
    per-asset volatilities are put back afterwards. Denoising is a statement
    about *co-movement*, and rescaling anyone's volatility would be a
    different claim entirely.

    Args:
        cov: Covariance matrix to filter.
        n_observations: The **effective** number of return observations ``T``
            behind it — the sample the estimator actually leant on. The
            Marchenko-Pastur cutoff is a function of ``T / N``, so passing a
            wrong ``T`` silently moves the cutoff. For an equally-weighted
            estimator this is the row count; for EWMA it is roughly
            ``1 / (1 − λ)``, which is far smaller and is the number the
            cutoff has to be built from.
        method: See :func:`denoise_correlation`.
        bandwidth: Kernel bandwidth for the density fit.
        alpha: Noise-block shrinkage under ``"targeted_shrinkage"``.
        detone: Number of leading eigenvectors to remove after denoising.
            ``0`` (the default) leaves the market component in place.
        n_signal: Override the fitted signal-eigenvalue count.
        n_observations_nominal: The row count, when it differs from the
            effective sample. Reported alongside it so the reader can see
            both, and named in the error when the effective sample is what
            made the fit impossible.
        effective_sample_note: One sentence explaining why the two differ.

    Returns:
        ``(filtered_covariance, report)``, the covariance carrying the same
        index and columns as the input.

    Raises:
        ValueError: If ``T <= N``, where the Marchenko-Pastur law has no
            usable support. See the note below on why this is a refusal
            rather than a fallback.

    Note:
        **Why ``q <= 1`` refuses instead of degrading.** The Marchenko-Pastur
        law is defined for ``q < 1`` too, but not the law this module fits.
        With ``T < N`` the sample correlation matrix is singular: ``N − T``
        of its eigenvalues are exactly zero, and the true limiting law puts
        an atom of mass ``1 − q`` at the origin beside its continuous part.
        :func:`marchenko_pastur_pdf` models only the continuous part and
        normalizes it to integrate to one, and
        :func:`fit_marchenko_pastur` fits it by matching a kernel density of
        *every* observed eigenvalue against it. Feed that a spectrum whose
        mass sits on a zero atom the density does not have and the fitted
        ``σ²`` — and therefore ``λ₊`` — is an artefact of the missing atom
        rather than an estimate of the noise edge. On top of that,
        ``"constant_residual"`` would then average a block of exact zeros and
        hand back a matrix whose conditioning it cannot vouch for. The cutoff
        is not merely different below ``q = 1``; it is not estimable by this
        procedure, so the honest answer is to say so and stop.
    """
    assets = list(cov.columns)
    n = len(assets)
    nominal = int(
        n_observations if n_observations_nominal is None else n_observations_nominal
    )
    values = np.asarray(cov.values, dtype=float)
    q = float(n_observations) / n if n else float("nan")
    if q <= 1.0:
        message = (
            f"Denoising needs more observations than assets: got "
            f"T = {n_observations}, N = {n} (T/N = {q:.2f}). Below 1 the "
            "sample spectrum is degenerate and the Marchenko-Pastur cutoff is "
            "not defined. Use a shrinkage estimator instead."
        )
        if nominal != n_observations:
            message = (
                f"Denoising needs more observations than assets, and the "
                f"effective sample is too short: T = {n_observations} "
                f"effective observations (from {nominal} rows) against "
                f"N = {n} assets gives T/N = {q:.2f}. "
                f"{effective_sample_note} "
                "The Marchenko-Pastur cutoff is not defined below T/N = 1 - "
                "the sample correlation is singular there and the fit has no "
                "noise edge to find - so this combination cannot be denoised "
                "however long the panel is. Either shorten the universe to "
                f"fewer than {n_observations} assets, raise the decay so the "
                "estimator leans on more history, or denoise an "
                "equally-weighted estimator instead."
            ).replace("  ", " ")
        raise ValueError(message)

    corr, std = cov_to_corr(values)
    denoised, report = denoise_correlation(
        corr, q, method=method, bandwidth=bandwidth, alpha=alpha, n_signal=n_signal
    )
    # Re-measure conditioning on the covariance rather than the correlation:
    # that is the matrix the solvers invert, and the only one comparable to
    # what the covariance diagnostics report.
    report = DenoiseReport(
        **{
            **report.__dict__,
            "n_observations": nominal,
            "n_observations_effective": int(n_observations),
            "effective_sample_note": effective_sample_note,
            "condition_before": _condition(_sorted_eigen(values)[0]),
            "condition_after": _condition(
                _sorted_eigen(corr_to_cov(denoised, std))[0]
            ),
        }
    )
    if detone:
        denoised = detone_correlation(denoised, n_factors=detone)
        report = DenoiseReport(
            **{
                **report.__dict__,
                "detoned_factors": int(detone),
                "condition_after": _condition(_sorted_eigen(denoised)[0]),
            }
        )
        warnings.warn(
            "A detoned covariance is singular by construction - the market "
            "eigenvector has been removed. Use it for clustering methods "
            "(HRP, HERC, NCO) and not for solves that invert the matrix.",
            stacklevel=2,
        )

    out = corr_to_cov(denoised, std)
    return (
        pd.DataFrame(out, index=cov.index, columns=cov.columns),
        report,
    )


def _condition(eigenvalue: np.ndarray) -> float:
    """Condition number from a spectrum, ``inf`` when singular."""
    smallest = float(np.min(eigenvalue))
    largest = float(np.max(eigenvalue))
    if smallest <= 0:
        return float("inf")
    return largest / smallest
