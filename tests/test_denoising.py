"""Random-matrix denoising and detoning.

The tests are written against properties that must hold rather than against
recorded numbers, so they keep their meaning if the kernel bandwidth or the
optimizer inside the Marchenko-Pastur fit changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.data.covariance import covariance_matrix
from optimization_engine.data.denoise import (
    cov_to_corr,
    denoise_correlation,
    denoise_covariance,
    detone_correlation,
    fit_marchenko_pastur,
    marchenko_pastur_pdf,
)
from optimization_engine.data.loader import prices_to_returns, sample_dataset


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=252 * 4, seed=11))


def _factor_correlation(n_assets: int, n_obs: int, n_factors: int, seed: int):
    """A correlation matrix with a known number of real factors."""
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(n_assets, n_factors))
    factors = rng.normal(size=(n_obs, n_factors))
    noise = rng.normal(size=(n_obs, n_assets)) * 1.5
    data = factors @ loadings.T + noise
    frame = pd.DataFrame(data, columns=[f"a{i}" for i in range(n_assets)])
    return frame.corr(), frame


# ---------------------------------------------------------------------------
# The Marchenko-Pastur law itself
# ---------------------------------------------------------------------------


def test_marchenko_pastur_density_integrates_to_one():
    pdf = marchenko_pastur_pdf(1.0, q=10.0, n_points=20_000)
    # Trapezoid by hand: numpy renamed ``trapz`` to ``trapezoid`` in 2.0 and
    # this suite runs on both.
    x, y = pdf.index.values, pdf.values
    area = float(np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(x)))
    assert area == pytest.approx(1.0, abs=5e-3)


def test_marchenko_pastur_support_widens_as_the_sample_shortens():
    wide = marchenko_pastur_pdf(1.0, q=2.0)
    narrow = marchenko_pastur_pdf(1.0, q=50.0)
    assert (wide.index.max() - wide.index.min()) > (
        narrow.index.max() - narrow.index.min()
    )


def test_marchenko_pastur_needs_more_observations_than_assets():
    with pytest.raises(ValueError, match="T/N"):
        marchenko_pastur_pdf(1.0, q=0.5)


def test_pure_noise_spectrum_fits_a_variance_near_one():
    rng = np.random.default_rng(3)
    n_assets, n_obs = 60, 600
    data = pd.DataFrame(
        rng.normal(size=(n_obs, n_assets)),
        columns=[f"a{i}" for i in range(n_assets)],
    )
    eigenvalues = np.linalg.eigvalsh(data.corr().values)
    variance, cutoff = fit_marchenko_pastur(eigenvalues, q=n_obs / n_assets)
    # With no signal at all, essentially all the variance is noise.
    assert variance > 0.9
    # And essentially every eigenvalue falls under the cutoff.
    assert (eigenvalues > cutoff).sum() <= 2


def test_signal_eigenvalues_are_detected_when_factors_are_present():
    corr, _ = _factor_correlation(n_assets=40, n_obs=800, n_factors=3, seed=5)
    eigenvalues = np.linalg.eigvalsh(corr.values)
    _, cutoff = fit_marchenko_pastur(eigenvalues, q=800 / 40)
    detected = int((eigenvalues > cutoff).sum())
    assert 1 <= detected <= 6, f"expected roughly 3 factors, detected {detected}"


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------


def test_denoising_improves_conditioning_and_keeps_a_correlation():
    corr, _ = _factor_correlation(n_assets=40, n_obs=200, n_factors=3, seed=9)
    denoised, report = denoise_correlation(corr.values, q=200 / 40)

    assert np.allclose(np.diag(denoised), 1.0)
    assert np.allclose(denoised, denoised.T)
    assert report.condition_after < report.condition_before
    assert 1 <= report.n_signal_eigenvalues < corr.shape[0]


def test_a_spectrum_with_no_signal_reports_zero_factors_and_the_one_kept():
    """The fit's count and the filter's floor are two different numbers.

    The filter always keeps the leading eigenvalue — treating the whole matrix
    as noise would erase it — but on a pure-noise panel the fit finds nothing
    above the edge, and the report used to say it had found one.
    """
    rng = np.random.default_rng(11)
    noise = pd.DataFrame(rng.normal(size=(1000, 50)) * 0.01)
    cov = noise.cov()
    eigenvalues = np.linalg.eigvalsh(cov_to_corr(cov.values)[0])
    _, cutoff = fit_marchenko_pastur(eigenvalues, q=1000 / 50)
    assert (eigenvalues > cutoff).sum() == 0

    _, report = denoise_covariance(cov, n_observations=1000)

    assert report.n_signal_eigenvalues == 0
    assert report.n_factors_kept == 1
    assert "kept anyway" in report.describe()


def test_a_marchenko_pastur_fit_that_fails_is_flagged_not_silently_replaced(
    monkeypatch,
):
    """σ² = 1 is the pure-noise edge, and it moves the answer.

    When the optimizer behind the fit reports failure the variance falls back
    to 1.0. On a panel with weak factors that pushed λ₊ from 1.553 to 1.732
    and the signal count from 4 to 2, and the minimum-variance book moved by
    0.13 in L1 — with nothing in the report to say the fit had not run.
    """
    import types

    import scipy.optimize

    real = scipy.optimize.minimize_scalar

    def unconverged(*args, **kwargs):
        found = real(*args, **kwargs)
        return types.SimpleNamespace(x=found.x, success=False)

    monkeypatch.setattr(scipy.optimize, "minimize_scalar", unconverged)
    corr, _ = _factor_correlation(n_assets=40, n_obs=400, n_factors=3, seed=9)

    with pytest.warns(UserWarning, match="Marchenko-Pastur fit"):
        _, report = denoise_correlation(corr.values, q=10.0)

    assert report.noise_fit_failed
    assert report.noise_variance == 1.0
    assert "did not converge" in report.describe()


def test_a_fit_that_converges_carries_no_failure_flag():
    corr, _ = _factor_correlation(n_assets=40, n_obs=400, n_factors=3, seed=9)
    _, report = denoise_correlation(corr.values, q=10.0)
    assert not report.noise_fit_failed
    assert report.n_factors_kept == report.n_signal_eigenvalues


def test_denoising_preserves_the_trace_of_the_correlation():
    """Constant-residual denoising redistributes eigenvalues, it does not add.

    The trace of a correlation matrix is N by definition, so preserving it is
    the check that no variance was invented or destroyed.
    """
    corr, _ = _factor_correlation(n_assets=30, n_obs=300, n_factors=2, seed=13)
    denoised, _ = denoise_correlation(corr.values, q=10.0)
    assert np.trace(denoised) == pytest.approx(30.0)


def test_denoising_leaves_the_signal_eigenvectors_alone():
    corr, _ = _factor_correlation(n_assets=40, n_obs=400, n_factors=2, seed=17)
    denoised, report = denoise_correlation(corr.values, q=10.0)

    _, before = np.linalg.eigh(corr.values)
    _, after = np.linalg.eigh(denoised)
    # eigh returns ascending eigenvalues, so the leading eigenvector is last.
    leading_before = before[:, -1]
    leading_after = after[:, -1]
    alignment = abs(float(leading_before @ leading_after))
    assert alignment > 0.99, "denoising rotated the dominant factor"
    assert report.n_signal_eigenvalues >= 1


def test_targeted_shrinkage_with_alpha_one_is_a_no_op():
    corr, _ = _factor_correlation(n_assets=20, n_obs=400, n_factors=2, seed=21)
    denoised, _ = denoise_correlation(
        corr.values, q=20.0, method="targeted_shrinkage", alpha=1.0
    )
    assert np.allclose(denoised, corr.values, atol=1e-8)


def test_targeted_shrinkage_is_gentler_than_constant_residual():
    corr, _ = _factor_correlation(n_assets=30, n_obs=300, n_factors=2, seed=23)
    hard, _ = denoise_correlation(corr.values, q=10.0)
    soft, _ = denoise_correlation(
        corr.values, q=10.0, method="targeted_shrinkage", alpha=0.5
    )
    assert np.linalg.norm(soft - corr.values) < np.linalg.norm(hard - corr.values)


def test_unknown_denoise_method_is_rejected():
    corr, _ = _factor_correlation(n_assets=10, n_obs=200, n_factors=1, seed=1)
    with pytest.raises(ValueError, match="Unknown denoising method"):
        denoise_correlation(corr.values, q=20.0, method="wishful")


# ---------------------------------------------------------------------------
# Detoning
# ---------------------------------------------------------------------------


def test_detoning_removes_the_market_factor_and_leaves_it_singular():
    corr, _ = _factor_correlation(n_assets=25, n_obs=500, n_factors=1, seed=29)
    detoned = detone_correlation(corr.values, n_factors=1)

    assert np.allclose(np.diag(detoned), 1.0)
    eigenvalues = np.linalg.eigvalsh(detoned)
    assert eigenvalues.min() < 1e-8, "the removed factor should leave a null space"
    # The average pairwise correlation must fall: that is the whole point.
    off = ~np.eye(25, dtype=bool)
    assert abs(detoned[off].mean()) < abs(corr.values[off].mean())


def test_detoning_rejects_removing_every_factor():
    corr, _ = _factor_correlation(n_assets=8, n_obs=200, n_factors=1, seed=31)
    with pytest.raises(ValueError, match="Can remove between"):
        detone_correlation(corr.values, n_factors=8)


# ---------------------------------------------------------------------------
# Covariance-level integration
# ---------------------------------------------------------------------------


def test_denoise_covariance_preserves_the_volatilities(returns: pd.DataFrame):
    cov = covariance_matrix(returns, method="sample")
    denoised, _ = denoise_covariance(cov, n_observations=len(returns))
    assert np.allclose(np.diag(denoised.values), np.diag(cov.values))
    assert list(denoised.columns) == list(cov.columns)


def test_denoise_ewma_uses_effective_observations(returns: pd.DataFrame):
    """EWMA leans on ~1/(1−λ) observations, not on every row it was handed.

    Feeding the row count to the Marchenko-Pastur fit treats 1000 rows of
    exponentially-decayed history as 1000 independent observations, which
    puts the noise edge far too low and classes almost every eigenvalue as
    signal. The two numbers used to disagree silently: the diagnostics
    warned about a ~17-observation sample while the denoiser was told it had
    a thousand.
    """
    plain = covariance_matrix(returns, method="sample", denoise=True)
    ewma = covariance_matrix(returns, method="ewma", denoise=True)
    plain_report = plain.attrs["denoise_report"]
    ewma_report = ewma.attrs["denoise_report"]

    # Both saw the same panel...
    assert plain_report.n_observations == len(returns)
    assert ewma_report.n_observations == len(returns)
    # ...but only one of them leant on all of it.
    assert plain_report.n_observations_effective == plain_report.n_observations
    assert ewma_report.n_observations_effective < ewma_report.n_observations
    assert ewma_report.n_observations_effective == round(1 / (1 - 0.94))

    # A shorter effective sample means a wider noise band, so the edge moves
    # up and fewer eigenvalues survive as signal.
    assert ewma_report.q < plain_report.q
    assert ewma_report.eigenvalue_cutoff > plain_report.eigenvalue_cutoff
    assert ewma_report.n_signal_eigenvalues <= plain_report.n_signal_eigenvalues
    assert "effective observations" in ewma_report.describe()
    assert "λ = 0.94" in ewma_report.describe()


def test_denoise_ewma_refuses_a_universe_wider_than_its_effective_sample():
    """T/N below 1 is not a cutoff this fit can find, so it refuses.

    The Marchenko-Pastur law exists for q < 1, but the law *this module
    fits* does not: the sample correlation is singular there, N−T of its
    eigenvalues are exactly zero, and ``marchenko_pastur_pdf`` models only
    the continuous part of the law without the zero atom the true limit
    carries. Fitting one against the other would return a σ² that is an
    artefact of the missing atom. The honest answer is to say so.
    """
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        rng.normal(0, 0.01, size=(1000, 25)),
        columns=[f"a{i}" for i in range(25)],
    )
    # A thousand rows: nothing about the panel is short.
    with pytest.raises(ValueError) as exc:
        covariance_matrix(frame, method="ewma", denoise=True)
    message = str(exc.value)
    # The message has to name the effective sample, or the reader concludes
    # their loader dropped 983 rows.
    assert "effective observations" in message
    assert "from 1000 rows" in message
    assert "λ = 0.94" in message
    assert "T/N = 0.68" in message
    # The same panel denoises fine without the exponential weighting.
    assert covariance_matrix(frame, method="sample", denoise=True) is not None


def test_denoise_ewma_survives_when_the_universe_is_narrow_enough():
    """Below N = 1/(1−λ) the fit is defined, so it runs rather than refusing."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        rng.normal(0, 0.01, size=(1000, 8)),
        columns=[f"a{i}" for i in range(8)],
    )
    cov = covariance_matrix(frame, method="ewma", denoise=True)
    report = cov.attrs["denoise_report"]
    assert report.n_observations_effective == 17
    assert report.q == pytest.approx(17 / 8)


def test_denoise_report_defaults_effective_to_nominal():
    """A caller that says nothing about weighting gets T back unchanged."""
    frame = pd.DataFrame(
        np.random.default_rng(3).normal(size=(400, 6)),
        columns=[f"a{i}" for i in range(6)],
    )
    _, report = denoise_covariance(frame.cov(), n_observations=400)
    assert report.n_observations == 400
    assert report.n_observations_effective == 400
    assert report.effective_sample_note == ""
    assert "effective observations" not in report.describe()


def test_denoise_covariance_refuses_a_degenerate_sample():
    frame = pd.DataFrame(
        np.random.default_rng(0).normal(size=(5, 8)),
        columns=[f"a{i}" for i in range(8)],
    )
    cov = frame.cov()
    with pytest.raises(ValueError, match="more observations than assets"):
        denoise_covariance(cov, n_observations=5)


def test_covariance_matrix_denoised_method_attaches_its_report(returns: pd.DataFrame):
    cov = covariance_matrix(returns, method="denoised")
    report = cov.attrs.get("denoise_report")
    assert report is not None
    assert report.n_assets == returns.shape[1]
    assert "Marchenko-Pastur" in report.describe()


def test_detone_flag_makes_the_covariance_singular(returns: pd.DataFrame):
    with pytest.warns(UserWarning, match="singular by construction"):
        cov = covariance_matrix(returns, method="denoised", detone=1)
    eigenvalues = np.linalg.eigvalsh(cov.values)
    assert eigenvalues.min() < 1e-12


def test_denoising_is_optional_and_off_by_default(returns: pd.DataFrame):
    plain = covariance_matrix(returns, method="sample")
    assert "denoise_report" not in plain.attrs


def test_cov_to_corr_rejects_a_constant_series():
    cov = np.array([[0.04, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="zero variance"):
        cov_to_corr(cov)
