"""Risk metrics: VaR, CVaR, downside deviation, drawdown shape, risk contributions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats


def downside_deviation(
    r: pd.Series | pd.DataFrame,
    mar: float = 0.0,
    full_sample: bool = True,
) -> float | pd.Series:
    """Root-mean-square shortfall below a minimum acceptable return.

    ``√( mean( min(r − MAR, 0)² ) )`` — the correct denominator for the
    Sortino ratio.

    The distinction that matters: the average is taken over *every* period,
    not only the losing ones. Taking the standard deviation of the negative
    subset instead (a common shortcut) both changes the denominator and
    re-centres the losses on their own mean, which understates downside risk —
    on the engine's sample panel by roughly 16%, inflating Sortino by ~19%.

    Args:
        r: Periodic returns.
        mar: Minimum acceptable return per period. ``0.0`` is the usual
            choice; pass the per-period risk-free rate for an excess-return
            Sortino.
        full_sample: Divide by the total number of periods (the standard
            definition). ``False`` divides by the number of shortfall periods
            instead, which answers "how bad is a bad period" rather than "how
            much downside risk does this carry".
    """
    shortfall = np.minimum(r - mar, 0.0)
    squared = shortfall**2
    if full_sample:
        return np.sqrt(squared.mean())
    below = (shortfall < 0).sum()
    if isinstance(r, pd.DataFrame):
        return np.sqrt(squared.sum() / below.replace(0, np.nan))
    return float(np.sqrt(squared.sum() / below)) if below else float("nan")


def semideviation(r: pd.Series | pd.DataFrame, mar: float = 0.0) -> float | pd.Series:
    """Alias for :func:`downside_deviation` with the standard denominator.

    Args:
        r: A return stream, or a frame of them.
        mar: Minimum acceptable return per period. Deviations below it count;
            everything above is treated as zero.

    Returns:
        Per-period downside deviation — not annualized. One value per column
        for a frame.
    """
    return downside_deviation(r, mar=mar, full_sample=True)


def skewness(r: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Population skewness. Negative means the left tail is the long one.

    Args:
        r: A return stream, or a frame of them.

    Returns:
        The third standardized moment, dimensionless. One value per column for
        a frame.
    """
    demeaned = r - r.mean()
    sigma = r.std(ddof=0)
    return (demeaned**3).mean() / sigma**3


def kurtosis(r: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Population kurtosis, *not* excess: a normal distribution scores 3.

    Args:
        r: A return stream, or a frame of them.

    Returns:
        The fourth standardized moment, dimensionless. Subtract 3 for excess
        kurtosis. One value per column for a frame.
    """
    demeaned = r - r.mean()
    sigma = r.std(ddof=0)
    return (demeaned**4).mean() / sigma**4


def is_normal(r: pd.Series, level: float = 0.01) -> bool:
    """Jarque-Bera normality test, returning True if we accept normality.

    Args:
        r: A return stream.
        level: Significance level as a fraction — ``0.01`` for a 1% test.

    Returns:
        ``True`` when the null of normality is *not* rejected. Note that a
        short sample fails to reject almost anything, so a ``True`` here is
        weak evidence.
    """
    _, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


def var_historic(r: pd.Series | pd.DataFrame, level: float = 5) -> float | pd.Series:
    """Historic Value at Risk at the ``level`` percentile (5 = 5%).

    Args:
        r: A return stream, or a frame of them.
        level: The percentile, in percent rather than as a fraction — pass
            ``5`` for the 95% VaR, not ``0.05``.

    Returns:
        A positive number: a VaR of ``0.02`` means "lose 2% or more in the
        worst 5% of periods". One value per column for a frame. NaN where
        there is no observation: ``np.percentile`` raises on an empty array,
        and one never-observed column used to take down a whole
        ``summary_stats`` table.

    Raises:
        TypeError: If ``r`` is neither a Series nor a DataFrame.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    if isinstance(r, pd.Series):
        observed = r.dropna()
        if observed.empty:
            return float("nan")
        return -np.percentile(observed, level)
    raise TypeError("Expected Series or DataFrame")


def var_gaussian(
    r: pd.Series | pd.DataFrame, level: float = 5, modified: bool = False
) -> float | pd.Series:
    """Parametric Value at Risk, optionally Cornish-Fisher adjusted.

    The Cornish-Fisher expansion corrects the Gaussian quantile for skew and
    kurtosis. It is an asymptotic expansion, so it is only trustworthy for
    moderate departures from normality — with kurtosis far above ~7 the
    adjusted quantile can stop being monotone in the confidence level. Prefer
    :func:`cvar_historic` when the sample is that wild.

    Args:
        r: A return stream, or a frame of them.
        level: The percentile, in percent rather than as a fraction — pass
            ``5`` for the 95% VaR, not ``0.05``.
        modified: Apply the Cornish-Fisher adjustment.

    Returns:
        A positive number: the loss threshold at that confidence. One value
        per column for a frame.
    """
    z = scipy.stats.norm.ppf(level / 100)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (
            z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * (k - 3) / 24
            - (2 * z**3 - 5 * z) * (s**2) / 36
        )
    return -(r.mean() + z * r.std(ddof=0))


def cvar_historic(r: pd.Series | pd.DataFrame, level: float = 5) -> float | pd.Series:
    """Historic Conditional VaR (expected shortfall) beyond the VaR threshold.

    Args:
        r: A return stream, or a frame of them.
        level: The percentile, in percent rather than as a fraction — pass
            ``5`` for the 95% CVaR, not ``0.05``.

    Returns:
        A positive number: the average loss across the periods worse than the
        VaR threshold. One value per column for a frame. NaN where there is
        no observation, as for :func:`var_historic`: no period lies beyond a
        NaN threshold, and the mean of none is NaN.

    Raises:
        TypeError: If ``r`` is neither a Series nor a DataFrame.
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    raise TypeError("Expected Series or DataFrame")


def tail_ratio(r: pd.Series | pd.DataFrame, level: float = 5) -> float | pd.Series:
    """Right-tail magnitude divided by left-tail magnitude.

    Above 1 means the good surprises are bigger than the bad ones. A useful
    companion to skewness because it reads off the actual quantiles rather
    than a third moment that a single outlier can dominate.

    Args:
        r: A return stream, or a frame of them.
        level: The percentile defining each tail, in percent — ``5`` compares
            the 95th percentile against the 5th.

    Returns:
        A dimensionless ratio. One value per column for a frame.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(tail_ratio, level=level)
    right = float(np.percentile(r, 100 - level))
    left = float(abs(np.percentile(r, level)))
    return right / left if left > 0 else float("nan")


def omega_ratio(
    r: pd.Series | pd.DataFrame, threshold: float = 0.0
) -> float | pd.Series:
    """Probability-weighted gains above a threshold over losses below it.

    Omega uses the whole return distribution rather than its first two
    moments, so it distinguishes portfolios that Sharpe cannot.

    Args:
        r: A return stream, or a frame of them.
        threshold: The per-period return separating a gain from a loss. Must
            be on the same periodicity as ``r``.

    Returns:
        A dimensionless ratio, above 1 when the probability-weighted gains
        exceed the losses. One value per column for a frame.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(omega_ratio, threshold=threshold)
    excess = r - threshold
    gains = float(excess[excess > 0].sum())
    losses = float(-excess[excess < 0].sum())
    return gains / losses if losses > 0 else float("inf")


# ---------------------------------------------------------------------------
# Drawdown shape
# ---------------------------------------------------------------------------


def log_wealth(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Cumulative log growth, ``Σ log(1 + r)``.

    Accumulating in log space rather than with ``cumprod`` keeps long or
    high-return series from overflowing float64, and removes the rounding
    that a product of thousands of terms accumulates. A return of exactly
    −100% yields ``-inf``, which is the correct reading: the capital is gone,
    and no later return brings it back. That is a deliberate result rather
    than a numerical accident, so the divide-by-zero it raises is silenced.

    Args:
        returns: A return stream, or a frame of them.

    Returns:
        Cumulative log growth, indexed like the input. Exponentiate for a
        wealth multiple, or use :func:`wealth_index`.
    """
    with np.errstate(divide="ignore"):
        return np.log1p(returns).cumsum()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown from the running peak, as a negative fraction.

    Computed as ``expm1(log_wealth − peak)`` so that the peak ratio is
    evaluated without ever materializing the wealth level itself.

    The running peak starts at the capital invested — log-wealth zero — not
    at the wealth after the first return. Starting it at the first close
    used to make a loss on the first period invisible: ``[-10%, -10%, +5%,
    +1%]`` reported a -10% maximum drawdown when the investor was 19% down.
    The capital before the first return is a peak like any other.

    Args:
        returns: A return stream.

    Returns:
        A series indexed like ``returns``, zero at each new peak and negative
        below one — ``-0.20`` is a 20% drawdown.
    """
    growth = log_wealth(returns)
    peak = np.maximum(growth.cummax(), 0.0)
    return pd.Series(np.expm1(growth - peak), index=returns.index)


def drawdown_table(returns: pd.Series, top: int = 5) -> pd.DataFrame:
    """The ``top`` worst drawdown episodes, with their timing.

    Max drawdown alone says how deep the hole was. What determines whether an
    investor actually sat through it is how *long* it lasted — so this reports
    peak, trough, recovery, depth, and both the decline and recovery lengths.
    An episode still underwater at the end of the sample has ``NaT`` recovery
    and is flagged.

    An episode that begins with the first return has the starting capital as
    its peak, which has no date in the index: its ``peak_date`` is ``NaT``,
    its ``peak_wealth`` is 1.0, and its lengths count from inception — the
    first period is one period of decline, not zero.

    Args:
        returns: A return stream.
        top: How many episodes to report, deepest first.

    Returns:
        One row per episode with the peak, trough and recovery dates, the
        depth as a negative fraction, and the decline and recovery lengths in
        periods.
    """
    dd = drawdown_series(returns)
    wealth = wealth_index(returns)
    peak = wealth.cummax()

    underwater = dd < -1e-12
    if not underwater.any():
        return pd.DataFrame(
            columns=[
                "peak_date", "trough_date", "recovery_date", "max_drawdown",
                "decline_periods", "recovery_periods", "total_periods", "recovered",
            ]
        )

    # Episodes by position. A start of -1 is the starting capital, the peak
    # before the first return.
    episodes: list[tuple[int, int | None]] = []
    start: int | None = None
    for i, is_under in enumerate(underwater.to_numpy()):
        if is_under and start is None:
            start = i - 1
        elif not is_under and start is not None:
            episodes.append((start, i))
            start = None
    if start is not None:
        episodes.append((start, None))

    idx = returns.index
    rows: list[dict[str, object]] = []
    for s_pos, e_pos in episodes:
        window = dd.iloc[max(s_pos, 0) : (None if e_pos is None else e_pos + 1)]
        if window.empty:
            continue
        trough_date = window.idxmin()
        depth = float(window.min())
        t_pos = idx.get_loc(trough_date)
        rows.append(
            {
                "peak_date": idx[s_pos] if s_pos >= 0 else pd.NaT,
                "trough_date": trough_date,
                "recovery_date": idx[e_pos] if e_pos is not None else pd.NaT,
                "max_drawdown": depth,
                "decline_periods": int(t_pos - s_pos),
                "recovery_periods": (
                    int(e_pos - t_pos) if e_pos is not None else np.nan
                ),
                "total_periods": (
                    int(e_pos - s_pos) if e_pos is not None else int(len(idx) - 1 - s_pos)
                ),
                "recovered": e_pos is not None,
                "peak_wealth": float(peak.iloc[s_pos]) if s_pos >= 0 else 1.0,
            }
        )

    table = pd.DataFrame(rows).sort_values("max_drawdown").head(top)
    return table.reset_index(drop=True)


def max_drawdown_duration(returns: pd.Series) -> float:
    """Longest run of periods spent below a previous peak.

    Args:
        returns: A return stream.

    Returns:
        The length in periods, on whatever periodicity ``returns`` carries.
    """
    table = drawdown_table(returns, top=10_000)
    if table.empty:
        return 0.0
    return float(table["total_periods"].max())


def wealth_index(returns: pd.Series, starting: float = 1.0) -> pd.Series:
    """Cumulative wealth, accumulated in log space for numerical safety.

    Args:
        returns: A return stream.
        starting: Value at the first period. ``1.0`` makes the series read as
            a growth multiple.

    Returns:
        A series indexed like ``returns``, compounding them from ``starting``.
    """
    return starting * np.exp(log_wealth(returns))


def ulcer_index(returns: pd.Series) -> float:
    """Root-mean-square drawdown: depth *and* duration in one number.

    Two portfolios can share a max drawdown while one spent a month
    underwater and the other three years. The Ulcer Index separates them.

    Args:
        returns: A return stream.

    Returns:
        A positive fraction. Larger is worse, and it has no upper bound.
    """
    dd = drawdown_series(returns)
    return float(np.sqrt((dd**2).mean()))


# ---------------------------------------------------------------------------
# Risk decomposition
# ---------------------------------------------------------------------------


def risk_contribution(weights: np.ndarray | pd.Series, cov_matrix: pd.DataFrame) -> pd.Series:
    """Decompose portfolio variance into per-asset risk contributions.

    Args:
        weights: Portfolio weights, as fractions of the book.
        cov_matrix: Asset covariance over the same universe.

    Returns:
        A series whose entries sum to 1 — each value is the share of total
        portfolio risk attributable to that asset. Because the Euler
        decomposition splits volatility and variance in the same proportions,
        this share reads correctly either way.
    """
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    total_var = float(w @ cov @ w)
    index = cov_matrix.columns if isinstance(cov_matrix, pd.DataFrame) else None
    if total_var <= 0:
        return pd.Series(np.zeros_like(w), index=index)
    marginal = cov @ w
    contribution = w * marginal / total_var
    return pd.Series(contribution, index=index)


def marginal_risk_contribution(
    weights: np.ndarray | pd.Series, cov_matrix: pd.DataFrame
) -> pd.Series:
    """``∂σ_p/∂w_i`` — the volatility added by one more unit of each asset.

    This is the number that says which position to trim first: the asset with
    the highest marginal risk per unit of expected return.

    Args:
        weights: Portfolio weights, as fractions of the book.
        cov_matrix: Asset covariance. Its periodicity sets the units of the
            result.

    Returns:
        One marginal volatility per asset, in the covariance's own units.
    """
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    port_vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    index = cov_matrix.columns if isinstance(cov_matrix, pd.DataFrame) else None
    if port_vol <= 0:
        return pd.Series(np.zeros_like(w), index=index)
    return pd.Series((cov @ w) / port_vol, index=index)


def group_risk_contribution(
    weights: pd.Series, cov_matrix: pd.DataFrame, groups: dict[str, str]
) -> pd.Series:
    """Aggregate risk contributions up to asset-class level.

    Risk shares are additive, so the group total is simply the sum of its
    members' contributions — which is what makes "equity is 85% of our risk"
    a statement you can compute rather than estimate.

    Args:
        weights: Portfolio weights, as fractions of the book.
        cov_matrix: Asset covariance over the same universe.
        groups: ``asset -> group`` mapping.

    Returns:
        One risk share per group, summing to 1.
    """
    rc = risk_contribution(weights, cov_matrix)
    labels = pd.Series(
        {a: groups.get(str(a), "Unassigned") for a in rc.index}, name="group"
    )
    return rc.groupby(labels).sum().sort_values(ascending=False)
