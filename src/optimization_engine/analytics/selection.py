"""Selection bias: what a Sharpe ratio is worth once you count the trials.

Everything else in this library estimates a portfolio. This module estimates
the *search* that produced it.

The engine makes it easy to run a hundred configurations — ten methods, five
covariance estimators, a grid of bounds — and report the best one. The
maximum of a hundred noisy estimates is a biased estimate of the best true
value, and the bias is large: with a hundred independent strategies that all
have a true Sharpe of zero, the best in-sample Sharpe will land near 2.6
standard errors above zero purely by chance. Nothing about that number is
visible in the number itself.

Three tools, all from Bailey and López de Prado:

* :func:`deflated_sharpe_ratio` — the probability the strategy's true Sharpe
  exceeds what the *best of N trials* would have produced under the null.
  This is the headline correction: it takes the observed Sharpe, the
  dispersion of Sharpes across the trials you actually ran, the number of
  trials, and the skew and kurtosis of the returns, and returns a probability.
* :func:`minimum_track_record_length` — how much history you would need
  before the observed Sharpe is significant at a chosen confidence. Usually a
  bracing answer.
* :func:`probability_of_backtest_overfitting` — the combinatorially
  symmetric cross-validation (CSCV) procedure: across every balanced split of
  the sample into train and test halves, how often does the in-sample winner
  land in the bottom half out of sample? Above ~50% the selection process is
  no better than picking at random.

All Sharpe ratios in this module are handled **per period** internally, and
they are *arithmetic* — the mean excess return over its standard deviation,
which is the estimator Bailey and López de Prado's standard error is derived
for. That is exactly
:func:`~optimization_engine.analytics.performance.sharpe_ratio` with
``periods_per_year=1``, and this module calls it rather than reimplementing
it. Annualized figures divide or multiply by ``√periods_per_year``, so an
annualized Sharpe handed in here — ``trial_sharpes`` in particular — must
come from the same arithmetic convention; a geometric one deflates a number
against a distribution measured a different way. The public functions take
and return annualized figures where that is the natural unit, and say which
is which.

References:
    Bailey, D. and López de Prado, M. (2012). "The Sharpe Ratio Efficient
    Frontier". *Journal of Risk* 15(2).

    Bailey, D. and López de Prado, M. (2014). "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting and Non-Normality".
    *The Journal of Portfolio Management* 40(5).

    Bailey, D., Borwein, J., López de Prado, M. and Zhu, Q. (2017). "The
    Probability of Backtest Overfitting". *Journal of Computational Finance*
    20(4).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats

from optimization_engine.analytics.performance import (
    _DEGENERATE_STD_RTOL,
    _dispersion_is_rounding,
    sharpe_ratio,
)
from optimization_engine.analytics.risk import kurtosis, skewness

#: Euler-Mascheroni constant, from the expected maximum of N Gaussians.
EULER_MASCHERONI = 0.5772156649015329


def _period_sharpe(returns: pd.Series) -> float:
    """Per-period Sharpe of an excess-return series.

    :func:`~optimization_engine.analytics.performance.sharpe_ratio` on a
    one-period-per-year clock, which is the arithmetic ratio
    ``mean / std``. Routed through it rather than reimplemented so this
    module's Sharpe and the library's headline Sharpe cannot drift apart —
    the deflation below compares one against a distribution of the other.

    A stream that is constant up to float rounding scores NaN, as it does in
    ``sharpe_ratio``: a series that never moved has no risk-adjusted
    performance to report. It used to score 0.0 at exactly zero variance and
    ~1e16 once rounding left a standard deviation of 1e-20, which handed a
    constant stream a deflated Sharpe of 1.0 and a one-period track record.
    """
    series = returns.dropna()
    return float(sharpe_ratio(series, riskfree_rate=0.0, periods_per_year=1))


def _sharpe_standard_error(
    sharpe: float, n: int, skew: float, kurt: float
) -> float:
    """Standard error of a per-period Sharpe estimate (Mertens / Lo).

    ``√[(1 − γ₃·SR + (γ₄ − 1)/4 · SR²) / (T − 1)]``. Negative skew and fat
    tails — the return shapes optimizers gravitate toward — *inflate* this,
    which is the whole reason a headline Sharpe is not comparable across
    strategies with different return shapes.
    """
    if n < 3:
        return float("nan")
    variance = 1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * sharpe**2
    return float(np.sqrt(max(variance, 1e-12) / (n - 1)))


def expected_maximum_sharpe(
    n_trials: int,
    sharpe_variance: float,
    mean_sharpe: float = 0.0,
) -> float:
    """Expected best Sharpe from ``n_trials`` strategies that have no skill.

    The benchmark a strategy has to beat is not zero. If you tried ``N``
    configurations whose Sharpe estimates were spread with variance
    ``sharpe_variance``, the best of them lands, in expectation, at

    ``E[SR] + √V[SR] · ((1 − γ)·Z⁻¹[1 − 1/N] + γ·Z⁻¹[1 − 1/(N·e)])``

    with ``γ`` the Euler-Mascheroni constant. That is the level to deflate
    against.

    Args:
        n_trials: Number of configurations actually tried — including the
            ones you discarded, which is the count people forget.
        sharpe_variance: Variance of the (per-period) Sharpe estimates
            across those trials.
        mean_sharpe: Their mean. Usually 0 under the null of no skill.

    Returns:
        The expected maximum, in the same per-period units as the inputs.

    Raises:
        ValueError: If ``n_trials < 1`` or ``sharpe_variance < 0``.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1; got {n_trials}.")
    if sharpe_variance < 0:
        raise ValueError(
            f"sharpe_variance must be non-negative; got {sharpe_variance}."
        )
    if n_trials == 1:
        return float(mean_sharpe)
    n = float(n_trials)
    quantile = (1.0 - EULER_MASCHERONI) * scipy.stats.norm.ppf(1.0 - 1.0 / n)
    quantile += EULER_MASCHERONI * scipy.stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(mean_sharpe + np.sqrt(sharpe_variance) * quantile)


@dataclass(frozen=True)
class DeflatedSharpe:
    """The result of deflating one Sharpe ratio for selection bias.

    Attributes:
        sharpe: Observed annualized Sharpe of the selected strategy.
        benchmark_sharpe: The annualized threshold it was deflated against —
            the expected maximum across the trials, not zero.
        deflated: ``P(true Sharpe > benchmark)``. Read it as a confidence
            level: below 0.95 the strategy has not demonstrably beaten what
            the search itself would have produced by luck.
        probabilistic: ``P(true Sharpe > 0)`` — the undeflated PSR, for
            comparison. The gap between the two is the price of the search.
        n_trials: Trials the deflation assumed.
        n_observations: Sample length behind the Sharpe.
        skewness: Return skewness. Negative widens the standard error.
        kurtosis: Return kurtosis (3 = normal). Fat tails widen it too.
        standard_error: Standard error of the per-period Sharpe estimate.
    """

    sharpe: float
    benchmark_sharpe: float
    deflated: float
    probabilistic: float
    n_trials: int
    n_observations: int
    skewness: float
    kurtosis: float
    standard_error: float

    @property
    def is_significant(self) -> bool:
        """True at the conventional 95% level."""
        return self.deflated >= 0.95

    def describe(self) -> str:
        """The Sharpe ratio, and what it is worth after the trial count.

        Returns:
            A sentence giving the raw Sharpe, the best result expected under the
            null across the declared trials, the deflated Sharpe, the
            probabilistic Sharpe before deflation, and the verdict at the 95%
            level — plus the skew and kurtosis the adjustment used.
        """
        verdict = (
            "clears the selection-bias threshold"
            if self.is_significant
            else "does NOT clear the selection-bias threshold"
        )
        return (
            f"Sharpe {self.sharpe:.2f} over {self.n_observations} observations. "
            f"Across {self.n_trials} trial(s) the best result expected under "
            f"the null is {self.benchmark_sharpe:.2f}; the deflated Sharpe — "
            f"P(true Sharpe > that) — is {self.deflated:.1%}, against "
            f"{self.probabilistic:.1%} before the trial count is taken into "
            f"account. The strategy {verdict} at 95%. "
            f"(Skew {self.skewness:.2f}, kurtosis {self.kurtosis:.2f}.)"
        )


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    trial_sharpes: pd.Series | np.ndarray | None = None,
    sharpe_variance: float | None = None,
    riskfree_rate: float = 0.0,
    periods_per_year: int = 252,
) -> DeflatedSharpe:
    """Deflate a Sharpe ratio for the number of trials behind it.

    Args:
        returns: Periodic returns of the *selected* strategy.
        n_trials: How many configurations were tried in total. If you swept a
            grid, this is the size of the grid — not the number you reported.
            Never fewer than the entries in ``trial_sharpes``: a smaller
            count is raised to that number, with a warning, and
            :attr:`DeflatedSharpe.n_trials` reports the count used.
        trial_sharpes: The annualized Sharpe of every trial, if you kept
            them. Their variance is the right dispersion to deflate against
            and is far better than a guess. They must be *arithmetic*
            Sharpes — :func:`~optimization_engine.analytics.performance.sharpe_ratio`
            with its default ``method`` — measured over one common sample;
            :meth:`~optimization_engine.backtest.sweep.SweepResults.trial_sharpes`
            produces exactly that.
        sharpe_variance: Variance of the *per-period* trial Sharpes, if you
            would rather supply it directly. Ignored when ``trial_sharpes``
            is given. Defaults to the estimated sampling variance of the
            observed Sharpe, which is the standard fallback when the trials
            themselves were not retained.
        riskfree_rate: Annual risk-free rate.
        periods_per_year: Observations per year.

    Returns:
        A :class:`DeflatedSharpe`. For a stream that is constant up to float
        rounding its Sharpe and both probabilities are NaN — there is no
        Sharpe to deflate — and it is not significant.

    Raises:
        ValueError: If fewer than 3 observations are supplied.
    """
    series = returns.dropna()
    n = len(series)
    if n < 3:
        raise ValueError(
            f"Need at least 3 observations to estimate a Sharpe ratio and its "
            f"standard error; got {n}."
        )
    rf_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess = series - rf_period

    sharpe = _period_sharpe(excess)
    skew = float(skewness(excess))
    kurt = float(kurtosis(excess))
    std_error = _sharpe_standard_error(sharpe, n, skew, kurt)

    if trial_sharpes is not None:
        n_supplied = int(np.asarray(trial_sharpes, dtype=float).size)
        if n_trials < n_supplied:
            # Every Sharpe handed in is a configuration that was tried, so the
            # count of them is a floor on the trial count. Fifty Sharpes with
            # n_trials=1 used to deflate against one trial — DSR 0.999 where
            # the fifty give 0.793.
            warnings.warn(
                f"n_trials={n_trials} is fewer than the {n_supplied} trial Sharpes "
                f"supplied; deflating against {n_supplied}.",
                UserWarning,
                stacklevel=2,
            )
            n_trials = n_supplied
        annual = _usable_trial_sharpes(trial_sharpes, periods_per_year)
        if annual.size < 2:
            raise ValueError(
                "trial_sharpes needs at least 2 entries to have a variance; "
                "pass sharpe_variance instead, or omit both."
            )
        variance = float(np.var(annual / np.sqrt(periods_per_year), ddof=1))
    elif sharpe_variance is not None:
        variance = float(sharpe_variance)
    else:
        # No record of the trials: the sampling variance of this Sharpe is
        # the honest stand-in, and the usual one in the literature.
        variance = std_error**2

    threshold = expected_maximum_sharpe(n_trials, variance)
    deflated = _psr(sharpe, threshold, n, skew, kurt)
    undeflated = _psr(sharpe, 0.0, n, skew, kurt)

    scale = np.sqrt(periods_per_year)
    return DeflatedSharpe(
        sharpe=float(sharpe * scale),
        benchmark_sharpe=float(threshold * scale),
        deflated=deflated,
        probabilistic=undeflated,
        n_trials=int(n_trials),
        n_observations=n,
        skewness=skew,
        kurtosis=kurt,
        standard_error=std_error,
    )


def _usable_trial_sharpes(
    trial_sharpes: pd.Series | np.ndarray, periods_per_year: int
) -> np.ndarray:
    """The trial Sharpes that can enter a dispersion, annualized.

    A NaN is a trial with no Sharpe — a constant return stream, or a cell
    too short to measure — and so is an infinite one. So is a finite Sharpe
    beyond what any stream with real dispersion can produce, the residue of
    a constant measured before that was refused: a single 1e17 used to set
    the deflation benchmark of a whole grid. They are left out of the
    *dispersion* with a warning; they still count in ``n_trials``, because
    they were configurations someone tried.
    """
    supplied = pd.Series(np.asarray(trial_sharpes, dtype=float).ravel())
    per_period = supplied.abs() / np.sqrt(periods_per_year)
    usable = np.isfinite(supplied) & (per_period < 1.0 / _DEGENERATE_STD_RTOL)
    dropped = int((~usable).sum())
    if dropped:
        warnings.warn(
            f"{dropped} of the {len(supplied)} trial Sharpes are not finite, or too "
            "large for a stream with any dispersion — a constant return stream has "
            "no Sharpe ratio — and were left out of the dispersion. They still "
            "count as trials.",
            UserWarning,
            stacklevel=3,
        )
    return supplied[usable].to_numpy(dtype=float)


def _psr(sharpe: float, benchmark: float, n: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe ratio, all arguments per period."""
    std_error = _sharpe_standard_error(sharpe, n, skew, kurt)
    if not np.isfinite(std_error) or std_error <= 0:
        return float("nan")
    return float(scipy.stats.norm.cdf((sharpe - benchmark) / std_error))


def minimum_track_record_length(
    returns: pd.Series,
    benchmark_sharpe: float = 0.0,
    confidence: float = 0.95,
    riskfree_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Observations needed before this Sharpe is significant at ``confidence``.

    ``MinTRL = 1 + [1 − γ₃·SR + (γ₄ − 1)/4 · SR²] · (Z_α / (SR − SR*))²``

    Returned in *periods*, on the same clock as ``returns``. Divide by
    ``periods_per_year`` for years. The number grows without bound as the
    observed Sharpe approaches the benchmark — a strategy that barely beats
    its benchmark can never be shown to beat it — and ``inf`` is returned when
    it does not beat it at all. A stream that is constant up to float rounding
    has no Sharpe ratio, and returns NaN.

    Args:
        returns: Periodic returns.
        benchmark_sharpe: Annualized Sharpe to beat. Pass the deflated
            benchmark from :func:`deflated_sharpe_ratio` to ask "how long
            before this survives the trial count".
        confidence: Required confidence level, e.g. ``0.95``.
        riskfree_rate: Annual risk-free rate.
        periods_per_year: Observations per year.

    Raises:
        ValueError: If ``confidence`` is not in ``(0, 1)``.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must lie in (0, 1); got {confidence}.")
    series = returns.dropna()
    if len(series) < 3:
        raise ValueError(
            f"Need at least 3 observations; got {len(series)}."
        )
    rf_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess = series - rf_period

    sharpe = _period_sharpe(excess)
    if not np.isfinite(sharpe):
        # A constant stream: no Sharpe, so no track record makes it significant
        # or insignificant. It used to come back as one period.
        return float("nan")
    benchmark = benchmark_sharpe / np.sqrt(periods_per_year)
    if sharpe <= benchmark:
        return float("inf")

    skew = float(skewness(excess))
    kurt = float(kurtosis(excess))
    z = scipy.stats.norm.ppf(confidence)
    variance = 1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * sharpe**2
    return float(1.0 + max(variance, 1e-12) * (z / (sharpe - benchmark)) ** 2)


# ---------------------------------------------------------------------------
# Combinatorially symmetric cross-validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverfittingReport:
    """Output of the CSCV probability-of-backtest-overfitting procedure.

    Attributes:
        pbo: Probability of backtest overfitting — the fraction of splits in
            which the in-sample winner ranked below median out of sample.
            Above 0.5 the selection is worse than a coin flip; the literature
            treats anything above ~0.2 as a warning.
        n_splits: Number of balanced train/test splits evaluated.
        n_trials: Number of candidate strategies compared.
        n_partitions: How many blocks the sample was cut into.
        median_logit: Median of the logits of the winner's out-of-sample
            rank. Negative means the winner typically underperforms.
        performance_degradation: OLS slope of out-of-sample performance on
            in-sample performance across splits. A slope at or below zero
            means in-sample performance carries no information about
            out-of-sample performance at all.
        probability_of_loss: Fraction of splits in which the selected
            strategy lost money out of sample.
        in_sample: Selected strategy's in-sample performance per split.
        out_of_sample: Its out-of-sample performance per split.
    """

    pbo: float
    n_splits: int
    n_trials: int
    n_partitions: int
    median_logit: float
    performance_degradation: float
    probability_of_loss: float
    in_sample: np.ndarray
    out_of_sample: np.ndarray

    def describe(self) -> str:
        """What the CSCV procedure found, and whether the selection means anything.

        Returns:
            A sentence giving the probability of backtest overfitting across the
            balanced splits, the verdict it implies (no better than chance at
            ``>= 0.5``, real information at ``<= 0.2``, weak in between), and the
            slope of out-of-sample performance regressed on in-sample.
        """
        verdict = (
            "the selection process is no better than chance"
            if self.pbo >= 0.5
            else "the selection carries real information"
            if self.pbo <= 0.2
            else "the selection is weak"
        )
        return (
            f"Across {self.n_splits} balanced splits of {self.n_trials} "
            f"candidate strategies, the in-sample winner landed below the "
            f"out-of-sample median {self.pbo:.1%} of the time — "
            f"{verdict}. Out-of-sample performance regressed on in-sample has "
            f"slope {self.performance_degradation:.2f} (≤ 0 means in-sample "
            f"ranking is uninformative), and the winner lost money out of "
            f"sample in {self.probability_of_loss:.1%} of splits."
        )


def probability_of_backtest_overfitting(
    trial_returns: pd.DataFrame,
    n_partitions: int = 16,
    metric: str = "sharpe",
) -> OverfittingReport:
    """CSCV: how often does the in-sample winner fail out of sample?

    The procedure (Bailey, Borwein, López de Prado & Zhu, 2017):

    1. Cut the sample into ``S`` contiguous blocks of equal length.
    2. Form every combination of ``S/2`` blocks as the training set; the
       remaining blocks are the test set. There are ``C(S, S/2)`` of them, and
       the construction is *symmetric* — every split's complement is also a
       split — which is what removes the arbitrariness of a single holdout.
    3. On each training set pick the best strategy; look up its rank among all
       strategies on the matching test set.
    4. PBO is the share of splits where that rank sits below the median.

    What makes this the right tool rather than a walk-forward: it holds the
    *number of strategies* fixed and asks whether choosing between them
    works. A walk-forward tells you whether one strategy survives; CSCV tells
    you whether your selection procedure does.

    Args:
        trial_returns: Periodic returns, one column per candidate strategy,
            all on the same index. This is the matrix of everything you tried.
        n_partitions: ``S``, which must be even. Larger gives more splits —
            ``C(16, 8) = 12,870`` — at a cost that grows fast.
        metric: ``"sharpe"`` or ``"mean"``. Sharpe is the usual choice;
            ``"mean"`` is cheaper and appropriate when the strategies share a
            volatility target.

    Returns:
        An :class:`OverfittingReport`.

    Raises:
        ValueError: If ``n_partitions`` is odd or below 4, if fewer than 2
            strategies are supplied, or if the sample is too short to cut.
    """
    if n_partitions % 2 or n_partitions < 4:
        raise ValueError(
            f"n_partitions must be an even number of at least 4; got "
            f"{n_partitions}. CSCV splits the blocks into two equal halves."
        )
    frame = trial_returns.dropna(how="any")
    n_obs, n_trials = frame.shape
    if n_trials < 2:
        raise ValueError(
            "CSCV compares candidate strategies against each other; it needs "
            f"at least 2 columns, got {n_trials}."
        )
    if n_obs < n_partitions * 2:
        raise ValueError(
            f"Need at least {n_partitions * 2} observations to cut {n_partitions} "
            f"usable blocks; got {n_obs}."
        )

    block_length = n_obs // n_partitions
    blocks = [
        frame.iloc[i * block_length : (i + 1) * block_length].values
        for i in range(n_partitions)
    ]

    def performance(rows: np.ndarray) -> np.ndarray:
        """Score every candidate over one set of rows.

        Args:
            rows: A ``periods x candidates`` block of returns.

        Returns:
            One score per candidate: the mean return under ``metric="mean"``, or
            the mean over the standard deviation under ``"sharpe"``, with a
            candidate that is constant up to float rounding scored zero rather
            than as a division by (almost) zero. Comparing with exactly zero
            let a constant 1bp candidate score ~1e16 and win every split.

        Raises:
            ValueError: If ``metric`` is neither ``"sharpe"`` nor ``"mean"``.
        """
        mean = rows.mean(axis=0)
        if metric == "mean":
            return mean
        if metric != "sharpe":
            raise ValueError(
                f"Unknown metric {metric!r}. Use 'sharpe' or 'mean'."
            )
        std = rows.std(axis=0, ddof=1)
        constant = _dispersion_is_rounding(std, np.abs(rows).mean(axis=0))
        return np.divide(mean, std, out=np.zeros_like(mean), where=~constant)

    logits: list[float] = []
    in_sample: list[float] = []
    out_of_sample: list[float] = []
    losses = 0

    indices = range(n_partitions)
    for train_idx in combinations(indices, n_partitions // 2):
        test_idx = [i for i in indices if i not in train_idx]
        train = np.vstack([blocks[i] for i in train_idx])
        test = np.vstack([blocks[i] for i in test_idx])

        train_perf = performance(train)
        test_perf = performance(test)
        winner = int(np.argmax(train_perf))

        # Relative rank of the winner out of sample, in (0, 1).
        order = scipy.stats.rankdata(test_perf)
        omega = float(order[winner]) / (n_trials + 1)
        omega = float(np.clip(omega, 1e-9, 1 - 1e-9))
        logits.append(float(np.log(omega / (1.0 - omega))))
        in_sample.append(float(train_perf[winner]))
        out_of_sample.append(float(test_perf[winner]))
        if test[:, winner].mean() < 0:
            losses += 1

    logit_array = np.asarray(logits)
    is_array = np.asarray(in_sample)
    oos_array = np.asarray(out_of_sample)
    slope = (
        float(np.polyfit(is_array, oos_array, 1)[0])
        if np.std(is_array) > 0
        else float("nan")
    )

    return OverfittingReport(
        pbo=float(np.mean(logit_array <= 0.0)),
        n_splits=len(logits),
        n_trials=n_trials,
        n_partitions=n_partitions,
        median_logit=float(np.median(logit_array)),
        performance_degradation=slope,
        probability_of_loss=float(losses / len(logits)),
        in_sample=is_array,
        out_of_sample=oos_array,
    )
