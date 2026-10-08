"""Performance metrics: annualized return/volatility, Sharpe, Sortino, drawdown."""

from __future__ import annotations

import numpy as np
import pandas as pd

from optimization_engine.analytics.risk import (
    cvar_historic,
    downside_deviation,
    drawdown_series,
    kurtosis,
    max_drawdown_duration,
    omega_ratio,
    skewness,
    tail_ratio,
    ulcer_index,
    var_gaussian,
    var_historic,
    wealth_index,
)


def drawdown(return_series: pd.Series, starting_wealth: float = 1000.0) -> pd.DataFrame:
    """Wealth index, running peak, and drawdown for a return series.

    The drawdown column comes from :func:`drawdown_series`, which works in
    log space, so it stays exact even where the wealth level itself would
    overflow. The peak starts at ``starting_wealth`` — the capital invested
    is the first high-water mark — so the three columns agree.

    Args:
        return_series: A return stream.
        starting_wealth: The capital invested before the first return, for
            the wealth and peak columns.

    Returns:
        A frame indexed like the input with ``wealth``, ``peak`` and
        ``drawdown`` columns, the last as a negative fraction.
    """
    wealth = wealth_index(return_series, starting_wealth)
    return pd.DataFrame(
        {
            "Wealth": wealth,
            "Peaks": wealth.cummax().clip(lower=starting_wealth),
            "Drawdown": drawdown_series(return_series),
        }
    )


def annualize_volatility(
    r: pd.Series | pd.DataFrame, periods_per_year: int = 252, prices: bool = False
) -> float | pd.Series:
    """Scale periodic volatility by ``√periods_per_year``.

    The square-root rule assumes returns are serially uncorrelated. Momentum
    or mean reversion breaks it — see :func:`annualize_volatility_newey_west`
    when autocorrelation is material.

    Args:
        r: A return stream, or a frame of them.
        periods_per_year: Annualization basis — 252 for daily, 12 for monthly.
        prices: Treat ``r`` as a price panel and difference it first. Gaps
            are *not* forward-filled — an interior missing price costs two
            returns rather than becoming a 0% period followed by a
            compounded jump, and the answer no longer depends on which
            pandas is installed.

    Returns:
        Annualized volatility as a fraction. One value per column for a
        frame, a scalar for a series.
    """
    if prices:
        r = r.pct_change(fill_method=None).dropna()
    return r.std() * np.sqrt(periods_per_year)


def annualize_volatility_newey_west(
    r: pd.Series | pd.DataFrame, periods_per_year: int = 252, lags: int = 5
) -> float | pd.Series:
    """Annualized volatility corrected for serial correlation.

    Applies the Newey-West adjustment ``σ² · (1 + 2·Σ_k (1 − k/(L+1))·ρ_k)``
    before scaling. Positively autocorrelated returns (trend-following,
    illiquid or appraisal-priced assets) understate their true annual risk
    under the plain square-root rule; this corrects for that.

    Args:
        r: A return stream, or a frame of them.
        periods_per_year: Annualization basis — 252 for daily, 12 for monthly.
        lags: How many autocorrelation lags ``L`` to include. More lags
            capture longer dependence at the cost of a noisier estimate.

    Returns:
        Annualized volatility as a fraction. One value per column for a
        frame, a scalar for a series.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(
            annualize_volatility_newey_west,
            periods_per_year=periods_per_year,
            lags=lags,
        )
    series = r.dropna()
    variance = float(series.var(ddof=1))
    if variance <= 0 or len(series) <= lags + 1:
        return float(np.sqrt(max(variance, 0.0)) * np.sqrt(periods_per_year))
    adjustment = 1.0
    for k in range(1, lags + 1):
        rho = float(series.autocorr(lag=k))
        if np.isnan(rho):
            continue
        adjustment += 2.0 * (1.0 - k / (lags + 1)) * rho
    adjustment = max(adjustment, 1e-6)
    return float(np.sqrt(variance * adjustment) * np.sqrt(periods_per_year))


def annualize_returns(
    r: pd.Series | pd.DataFrame, periods_per_year: int = 252, prices: bool = False
) -> float | pd.Series:
    """Geometric (compound) annualized return.

    Args:
        r: A return stream, or a frame of them.
        periods_per_year: Annualization basis — 252 for daily, 12 for monthly.
        prices: Treat ``r`` as a price panel and difference it first. Gaps
            are *not* forward-filled — an interior missing price costs two
            returns rather than becoming a 0% period followed by a
            compounded jump, and the answer no longer depends on which
            pandas is installed.

    Returns:
        The compound annual growth rate as a fraction. One value per column
        for a frame, a scalar for a series.
    """
    if prices:
        r = r.pct_change(fill_method=None).dropna()
    compounded = (1 + r).prod()
    # Count the observations that exist, not the rows: ``prod`` skips a NaN,
    # so counting its row would price a missing period as a zero return.
    n = r.count()
    return compounded ** (periods_per_year / n) - 1


def _rf_per_period(riskfree_rate: float, periods_per_year: int) -> float:
    return (1 + riskfree_rate) ** (1 / periods_per_year) - 1


#: The Sharpe conventions this library will compute, and nothing else.
SHARPE_METHODS = ("arithmetic", "geometric")


def sharpe_ratio(
    r: pd.Series | pd.DataFrame,
    riskfree_rate: float = 0.0,
    periods_per_year: int = 252,
    *,
    method: str = "arithmetic",
) -> float | pd.Series:
    """Annualized excess return over annualized volatility.

    **The single source of the Sharpe ratio in this library.** Three
    definitions used to coexist — this one, an arithmetic one inside
    :func:`rolling_metrics`, and a per-period one inside
    ``analytics.selection`` — so a deflated Sharpe deflated an arithmetic
    number against a distribution of geometric ones. Every Sharpe in the tree
    now routes through here; the only thing that varies is ``method``.

    The gap between the two is not a rounding error: on the eight-year sample
    panel, equal-weighted, the geometric Sharpe is 0.5950 and the arithmetic
    one 0.6238 — 4.8% higher.

    The usual shorthand, that the geometric numerator is the arithmetic one
    less half the variance, holds for the volatilities most return series
    have but is not a rule. Compounding is convex, and its second-order term
    pushes the geometric numerator back up; it wins whenever the per-period
    volatility falls below roughly ``sqrt(periods_per_year - 1)`` times the
    mean — about 15.8x on daily data. On a low-volatility, high-mean series
    the geometric Sharpe is therefore the *higher* of the two. This is the
    same effect that makes the sample panel's cash column the one asset whose
    arithmetic mean sits below its geometric one.

    Args:
        r: A return stream, or a frame of them.
        riskfree_rate: Annualized risk-free rate, as a fraction.
        periods_per_year: Annualization basis — 252 for daily, 12 for monthly.
        method: ``"arithmetic"`` (the default) annualizes the mean excess
            return by multiplying by ``periods_per_year``. This is the
            convention Bailey and López de Prado's deflated and probabilistic
            Sharpe ratios assume, so it is the one the selection-bias
            diagnostics need. ``"geometric"`` annualizes it by compounding,
            via :func:`annualize_returns`; it answers "what compound excess
            growth did this earn per unit of risk", and it is the number this
            function returned before the conventions were unified.

    Returns:
        A dimensionless ratio. One value per column for a frame, a scalar
        for a series.

    Raises:
        ValueError: If ``method`` is neither ``"arithmetic"`` nor
            ``"geometric"``.
    """
    if method not in SHARPE_METHODS:
        raise ValueError(
            f"Unknown Sharpe method {method!r}; expected one of "
            f"{', '.join(repr(m) for m in SHARPE_METHODS)}."
        )
    rf = _rf_per_period(riskfree_rate, periods_per_year)
    excess = r - rf
    ann_vol = annualize_volatility(r, periods_per_year)
    # A zero-variance stream has an infinite Sharpe, which is the answer, not
    # an error. pandas already suppresses the warning for the frame path; this
    # makes the scalar path behave the same way, which is what lets
    # ``rolling_metrics`` call this per window without changing what it warns.
    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "geometric":
            return annualize_returns(excess, periods_per_year) / ann_vol
        return excess.mean() * periods_per_year / ann_vol


def probabilistic_sharpe_ratio(
    r: pd.Series,
    benchmark_sharpe: float = 0.0,
    riskfree_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Probability that the true Sharpe ratio exceeds ``benchmark_sharpe``.

    A Sharpe ratio is an estimate, and its standard error grows with negative
    skew and fat tails — exactly the return shapes that optimizers gravitate
    toward. The PSR (Bailey & López de Prado, 2012) converts the point
    estimate into a confidence statement: below ~0.95 the portfolio has not
    demonstrably beaten the benchmark, however good the headline number looks.

    Its internal Sharpe is arithmetic per period — it always was, and it had
    to be, because the standard error below is derived for that estimator.
    Before the conventions were unified it therefore *disagreed* with
    :func:`sharpe_ratio`, which annualized geometrically; now the two agree,
    and ``sharpe_ratio(r, rf, periods_per_year=1)`` is exactly the per-period
    Sharpe this function tests.

    Args:
        r: A return stream.
        benchmark_sharpe: The annualized Sharpe to beat. ``0.0`` asks only
            whether the strategy makes money at all.
        riskfree_rate: Annualized risk-free rate, as a fraction.
        periods_per_year: Annualization basis — 252 for daily, 12 for monthly.

    Returns:
        A probability in ``[0, 1]``. It does *not* account for how many
        strategies you tried — for that, deflate it with
        :func:`~optimization_engine.analytics.selection.deflated_sharpe_ratio`.
    """
    import scipy.stats

    series = r.dropna()
    n = len(series)
    if n < 3:
        return float("nan")
    rf = _rf_per_period(riskfree_rate, periods_per_year)
    excess = series - rf
    sr_period = float(excess.mean() / excess.std(ddof=1)) if excess.std(ddof=1) > 0 else 0.0
    benchmark_period = benchmark_sharpe / np.sqrt(periods_per_year)
    g = float(skewness(excess))
    k = float(kurtosis(excess))
    denom = np.sqrt(
        max(1.0 - g * sr_period + ((k - 1.0) / 4.0) * sr_period**2, 1e-12)
    )
    z = (sr_period - benchmark_period) * np.sqrt(n - 1) / denom
    return float(scipy.stats.norm.cdf(z))


def sortino_ratio(
    r: pd.Series | pd.DataFrame,
    riskfree_rate: float = 0.0,
    periods_per_year: int = 252,
    mar: float | None = None,
) -> float | pd.Series:
    """Annualized excess return over annualized downside deviation.

    **Convention:** the numerator is *geometric* — :func:`annualize_returns`
    compounds it — while :func:`sharpe_ratio` now annualizes arithmetically by
    default. The two ratios in the same summary therefore do not share a
    numerator, and a Sortino cannot be read as "the Sharpe with a different
    denominator". This is deliberate and unchanged; it is recorded here so the
    mixing is visible rather than silent.

    Args:
        r: A return stream, or a frame of them.
        riskfree_rate: Annualized risk-free rate, as a fraction.
        periods_per_year: Annualization basis — 252 for daily, 12 for monthly.
        mar: Minimum acceptable return per period. Defaults to the
            per-period risk-free rate, so the numerator and the denominator
            measure shortfall against the same threshold — the pair that makes
            the ratio internally consistent.

    Returns:
        A dimensionless ratio. One value per column for a frame, a scalar
        for a series.
    """
    rf = _rf_per_period(riskfree_rate, periods_per_year)
    threshold = rf if mar is None else mar
    excess = r - rf
    ann_excess = annualize_returns(excess, periods_per_year)
    downside = downside_deviation(r, mar=threshold) * np.sqrt(periods_per_year)
    return ann_excess / downside


def calmar_ratio(
    r: pd.Series | pd.DataFrame, periods_per_year: int = 252
) -> float | pd.Series:
    """Annualized return over the absolute worst drawdown.

    **Convention:** the numerator is *geometric* — :func:`annualize_returns`
    compounds it — while :func:`sharpe_ratio` now annualizes arithmetically by
    default. Deliberate and unchanged, but stated so a summary carrying both
    is not read as though they shared a numerator.

    Args:
        r: A return stream, or a frame of them.
        periods_per_year: Annualization basis — 252 for daily, 12 for monthly.

    Returns:
        A dimensionless ratio. One bad fortnight decides it — see
        :func:`martin_ratio` for the version that prices the whole drawdown
        path. One value per column for a frame, a scalar for a series.
    """
    ann = annualize_returns(r, periods_per_year)
    if isinstance(r, pd.DataFrame):
        max_dd = r.aggregate(lambda x: drawdown(x).Drawdown.min())
    else:
        max_dd = drawdown(r).Drawdown.min()
    return ann / abs(max_dd)


def hit_rate(r: pd.Series | pd.DataFrame, threshold: float = 0.0) -> float | pd.Series:
    """Fraction of periods returning more than ``threshold``.

    Args:
        r: A return stream, or a frame of them.
        threshold: The per-period return a period must beat to count as a
            win, on the same periodicity as ``r``.

    Returns:
        A fraction in ``[0, 1]``. Read next to :func:`win_loss_ratio`, never
        instead of it. One value per column for a frame, a scalar for a
        series.
    """
    # A NaN period is neither a win nor a loss, so it leaves the denominator
    # as well as the numerator; ``(r > t).mean()`` would count it as a loss.
    return (r > threshold).sum() / r.count()


def gain_to_pain_ratio(r: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Total return divided by the total of the losing periods.

    Schwager's ratio. Where Sharpe divides by a symmetric dispersion — which
    charges the portfolio for its good months as much as its bad ones — this
    divides by the losses alone, and it needs no distributional assumption to
    do it. A value of 1 means every unit earned was matched by a unit lost
    along the way.

    Args:
        r: A return stream, or a frame of them.

    Returns:
        A dimensionless ratio. One value per column for a frame, a scalar
        for a series.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(gain_to_pain_ratio)
    series = r.dropna()
    pain = float(series[series < 0].sum())
    if pain == 0:
        return float("inf") if float(series.sum()) > 0 else float("nan")
    return float(series.sum() / abs(pain))


def win_loss_ratio(r: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Average winning period over the average losing period, in absolute size.

    Read next to the hit rate, never instead of it: a strategy can win four
    times out of five and still lose money if this ratio is small enough.

    Args:
        r: A return stream, or a frame of them.

    Returns:
        A dimensionless ratio. One value per column for a frame, a scalar
        for a series.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(win_loss_ratio)
    series = r.dropna()
    wins = series[series > 0]
    losses = series[series < 0]
    if len(losses) == 0:
        return float("inf") if len(wins) else float("nan")
    if len(wins) == 0:
        return 0.0
    return float(wins.mean() / abs(losses.mean()))


def martin_ratio(
    r: pd.Series | pd.DataFrame,
    riskfree_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """Annualized excess return over the Ulcer index.

    Calmar divides by the single worst drawdown, so one bad fortnight decides
    it. The Ulcer index is the root-mean-square of the whole drawdown path,
    which prices the *duration* of the pain as well as its depth — the
    difference between a portfolio that fell 30% and recovered in a month and
    one that spent three years down 20%.

    **Convention:** like :func:`sortino_ratio` and :func:`calmar_ratio`, the
    numerator is *geometric* — :func:`annualize_returns` compounds it — while
    :func:`sharpe_ratio` now annualizes arithmetically by default. Deliberate
    and unchanged; recorded so the mixing is visible.

    Args:
        r: A return stream, or a frame of them.
        riskfree_rate: Annualized risk-free rate, as a fraction.
        periods_per_year: Annualization basis — 252 for daily, 12 for monthly.

    Returns:
        A dimensionless ratio. One value per column for a frame, a scalar
        for a series.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(
            martin_ratio,
            riskfree_rate=riskfree_rate,
            periods_per_year=periods_per_year,
        )
    ulcer = float(ulcer_index(r))
    if ulcer <= 0:
        return float("nan")
    rf = _rf_per_period(riskfree_rate, periods_per_year)
    return float(annualize_returns(r - rf, periods_per_year) / ulcer)


def time_under_water(r: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Share of periods spent below the previous high-water mark.

    The statistic an investor experiences directly. A high Sharpe with 80%
    time under water describes a portfolio that is uncomfortable to hold
    regardless of how well it scores.

    Args:
        r: A return stream, or a frame of them.

    Returns:
        A fraction in ``[0, 1]``. One value per column for a frame, a scalar
        for a series.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(time_under_water)
    dd = drawdown_series(r.dropna())
    if len(dd) == 0:
        return float("nan")
    return float((dd < -1e-12).mean())


def rolling_metrics(
    r: pd.Series,
    window: int,
    riskfree_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Rolling annualized return, volatility, Sharpe and drawdown.

    A single full-sample Sharpe hides whether the strategy worked throughout
    or earned everything in one window. This is the frame that answers that.

    ``rolling_sharpe`` is :func:`sharpe_ratio` evaluated on each window, so
    the last row of a full-length window is the headline Sharpe to the last
    bit. It used to be an open-coded arithmetic ratio while the headline was
    geometric, which meant the two disagreed by a few per cent and neither
    said so.

    Args:
        r: The return stream.
        window: Rolling window length, in periods. At least 2.
        riskfree_rate: Annualized risk-free rate, as a fraction — the same
            unit :func:`sharpe_ratio` takes, converted to a per-period rate
            internally.
        periods_per_year: Annualization basis.

    Returns:
        A frame indexed like ``r``, one column per metric, with NaN over the
        first ``window - 1`` rows.

    Raises:
        ValueError: If ``window`` is below 2.
    """
    if window < 2:
        raise ValueError(f"Rolling window must be at least 2 periods; got {window}.")
    roll = r.rolling(window)
    # Compound in log space. A rolling np.prod over a long window can overflow
    # float64 on high-return series, and it accumulates rounding besides;
    # summing logs is exact to machine precision and cannot overflow.
    growth = np.log1p(r.clip(lower=-1 + 1e-12))
    ann_ret = np.expm1(growth.rolling(window).sum() * (periods_per_year / window))
    ann_vol = roll.std() * np.sqrt(periods_per_year)
    # Calling ``sharpe_ratio`` per window rather than open-coding the ratio
    # again costs about 0.2 s on 2,000 rows against 0.01 s vectorised. That is
    # the price of the guarantee that this column and the headline Sharpe can
    # never disagree, on a function called a handful of times per report and
    # never inside a solve or a sweep.
    # ``raw=False`` is load-bearing: on a bare ndarray ``.std()`` is ddof=0,
    # which is a different — and wrong — denominator.
    rolling_sharpe = roll.apply(
        sharpe_ratio, raw=False, args=(riskfree_rate, periods_per_year)
    )
    return pd.DataFrame(
        {
            "rolling_return": ann_ret,
            "rolling_volatility": ann_vol,
            "rolling_sharpe": rolling_sharpe,
            "rolling_drawdown": drawdown_series(r),
        }
    )


def summary_stats(
    r: pd.DataFrame | pd.Series,
    periods_per_year: int = 252,
    riskfree_rate: float = 0.03,
    var_level: float = 5,
    extended: bool = False,
) -> pd.DataFrame:
    """Aggregate summary statistics per column of a returns frame.

    **Mixed annualization conventions, on purpose.** ``"Sharpe Ratio"`` is
    arithmetic: the mean excess return times ``periods_per_year``. Every other
    ratio here — Sortino, Calmar, Martin — keeps a *geometric* (compounded)
    numerator, as does ``"Annualized Return"``. They are therefore not
    rescalings of one another, and a Sortino above a Sharpe may say something
    about the convention rather than about downside risk. Unifying them is out
    of scope; naming the split is not.

    ``"Sharpe Ratio (geometric)"`` carries the number ``"Sharpe Ratio"`` held
    before the conventions were unified, so a reader can see exactly what
    moved. It is a one-release migration aid and will be removed.

    Args:
        r: Periodic returns, one column per series.
        periods_per_year: Observations per year.
        riskfree_rate: Annual risk-free rate for Sharpe and Sortino.
        var_level: Tail percentile for VaR/CVaR (5 ⇒ 5%).
        extended: Add the ratios that do not assume symmetric risk —
            Calmar, Omega, Martin, gain-to-pain, win/loss — plus the tail
            ratio, hit rate, Ulcer index, drawdown duration, time under
            water, the best and worst single period, and the
            probabilistic Sharpe ratio.
    """
    if isinstance(r, pd.Series):
        r = r.to_frame()
    ann_r = r.aggregate(annualize_returns, periods_per_year=periods_per_year)
    ann_vol = r.aggregate(annualize_volatility, periods_per_year=periods_per_year)
    ann_sr = r.aggregate(
        sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=periods_per_year
    )
    geo_sr = r.aggregate(
        sharpe_ratio,
        riskfree_rate=riskfree_rate,
        periods_per_year=periods_per_year,
        method="geometric",
    )
    ann_sortino = r.aggregate(
        sortino_ratio, riskfree_rate=riskfree_rate, periods_per_year=periods_per_year
    )
    dd = r.aggregate(lambda s: drawdown(s).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var = r.aggregate(var_gaussian, level=var_level, modified=True)
    hist_var = r.aggregate(var_historic, level=var_level)
    hist_cvar = r.aggregate(cvar_historic, level=var_level)

    out = pd.DataFrame(
        {
            "Annualized Return": ann_r,
            "Annualized Vol": ann_vol,
            "Skewness": skew,
            "Kurtosis": kurt,
            f"Historic VaR({var_level:.0f}%)": hist_var,
            f"Cornish-Fisher VaR({var_level:.0f}%)": cf_var,
            f"Historic CVaR({var_level:.0f}%)": hist_cvar,
            "Sharpe Ratio": ann_sr,
            "Sharpe Ratio (geometric)": geo_sr,
            "Sortino Ratio": ann_sortino,
            "Max Drawdown": dd,
        }
    )
    # Preserve the historical column label so existing formatters keep working.
    if var_level == 5:
        out = out.rename(
            columns={
                "Cornish-Fisher VaR(5%)": "Cornish-Fisher VaR(5%)",
                "Historic CVaR(5%)": "Historic CVaR(5%)",
            }
        )

    if extended:
        out["Calmar Ratio"] = r.aggregate(
            calmar_ratio, periods_per_year=periods_per_year
        )
        out["Omega Ratio"] = r.aggregate(omega_ratio)
        out["Tail Ratio"] = r.aggregate(tail_ratio, level=var_level)
        out["Hit Rate"] = r.aggregate(hit_rate)
        out["Ulcer Index"] = r.aggregate(ulcer_index)
        out["Max DD Duration"] = r.aggregate(max_drawdown_duration)
        out["Martin Ratio"] = r.aggregate(
            martin_ratio,
            riskfree_rate=riskfree_rate,
            periods_per_year=periods_per_year,
        )
        out["Gain-to-Pain"] = r.aggregate(gain_to_pain_ratio)
        out["Win/Loss Ratio"] = r.aggregate(win_loss_ratio)
        out["Time Under Water"] = r.aggregate(time_under_water)
        out["Best Period"] = r.max()
        out["Worst Period"] = r.min()
        out["Prob. Sharpe > 0"] = r.aggregate(
            lambda s: probabilistic_sharpe_ratio(
                s, 0.0, riskfree_rate, periods_per_year
            )
        )
    return out
