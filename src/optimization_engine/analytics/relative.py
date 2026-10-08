"""Relative-performance metrics versus a benchmark.

Every function here aligns the portfolio and the benchmark on their common
dates before computing anything. Comparing two series that merely happen to
be the same length — one starting a month later, say — produces a beta and a
tracking error that describe no real portfolio, and nothing in the output
would reveal it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from optimization_engine._optional import LazyModule
from optimization_engine.analytics.performance import (
    annualize_returns,
    annualize_volatility,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from optimization_engine.analytics.risk import downside_deviation

#: statsmodels is only needed by the OLS-based metrics below (beta,
#: conditional beta, regression stats). Everything else in this module is
#: numpy and pandas, so it stays importable without the ``stats`` extra.
sm = LazyModule("statsmodels.api", extra="stats", purpose="OLS regression metrics")


def spread(
    series_1: pd.Series | pd.DataFrame, series_2: pd.Series | pd.DataFrame
) -> pd.DataFrame | pd.Series:
    """Arithmetic difference of two aligned return streams.

    Args:
        series_1: The stream to subtract from.
        series_2: The stream to subtract.

    Returns:
        ``series_1 - series_2`` over their common dates. Note this is an
        arithmetic difference of returns, not a relative return: for a
        compounding view of "how far behind" see :func:`relative_drawdown`.
    """
    if isinstance(series_2, pd.DataFrame):
        series_2 = series_2.iloc[:, 0]
    if isinstance(series_1, pd.DataFrame):
        return series_1.sub(series_2, axis=0)
    return series_1 - series_2


def _ensure_frame(s: pd.Series | pd.DataFrame) -> pd.DataFrame:
    return s.to_frame() if isinstance(s, pd.Series) else s


def _as_series(rb: pd.Series | pd.DataFrame) -> pd.Series:
    return rb.iloc[:, 0] if isinstance(rb, pd.DataFrame) else rb


def _aligned(
    r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Restrict both inputs to the dates on which both are observed.

    The frame keeps every date on which *any* of its columns is observed, so
    a column that starts late still carries NaN rows. Anything computed per
    column has to pair that column with the benchmark on its own dates — see
    :func:`_pair` — or it compares the column over one window with the
    benchmark over another.

    Raises:
        ValueError: When the two series share no dates at all — a silent
            empty result would otherwise propagate as NaN metrics.
    """
    frame = _ensure_frame(r)
    bench = _as_series(rb)
    common = frame.dropna(how="all").index.intersection(bench.dropna().index)
    if len(common) == 0:
        raise ValueError(
            "The portfolio and the benchmark share no dates, so no relative "
            "metric can be computed. Check that both cover the same period."
        )
    return frame.loc[common], bench.loc[common]


def _pair(frame: pd.DataFrame, bench: pd.Series, column) -> tuple[pd.Series, pd.Series]:
    """One column of an aligned frame and the benchmark, on that column's dates.

    A walk-forward column that starts a year after the fitted one used to
    meet the benchmark over the fitted one's dates: beta ran OLS over its NaN
    rows and came back NaN without a word, and up-capture compounded the
    column over its own dates but the benchmark over all of them.
    """
    series = frame[column]
    observed = series.notna()
    return series[observed], bench[observed]


def _geometric_capture(
    r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame, upside: bool
) -> pd.Series:
    frame, aligned_bench = _aligned(r, rb)
    out: dict[str, float] = {}
    for col in frame.columns:
        series, bench = _pair(frame, aligned_bench, col)
        mask = bench > 0 if upside else bench < 0
        n = int(mask.sum())
        if n == 0:
            out[col] = float("nan")
            continue
        bench_geo = float((1 + bench[mask]).prod() ** (1 / n) - 1)
        if bench_geo == 0:
            out[col] = float("nan")
            continue
        geo = float((1 + series[mask]).prod() ** (1 / n) - 1)
        out[col] = geo / bench_geo
    return pd.Series(out)


def up_capture(r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame) -> pd.Series:
    """Geometric up-capture: share of the benchmark's gains that was captured.

    Args:
        r: Portfolio returns — a stream, or a frame of them.
        rb: Benchmark returns, over the same dates.

    Returns:
        One ratio per column of ``r``, computed only over the periods in which
        the benchmark rose. Above 1 means the portfolio outran the index on
        the way up.
    """
    return _geometric_capture(r, rb, upside=True)


def down_capture(r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame) -> pd.Series:
    """Geometric down-capture: share of the benchmark's losses that was taken.

    Args:
        r: Portfolio returns — a stream, or a frame of them.
        rb: Benchmark returns, over the same dates.

    Returns:
        One ratio per column of ``r``, computed only over the periods in which
        the benchmark fell. Below 1 is the good direction here.
    """
    return _geometric_capture(r, rb, upside=False)


def capture_ratio(r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame) -> pd.Series:
    """Up-capture over down-capture. Above 1 is the asymmetry worth paying for.

    Args:
        r: Portfolio returns — a stream, or a frame of them.
        rb: Benchmark returns, over the same dates.

    Returns:
        One value per column of ``r``.
    """
    return up_capture(r, rb) / down_capture(r, rb)


def regression_stats(
    r: pd.Series | pd.DataFrame,
    rb: pd.Series | pd.DataFrame,
    riskfree_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Full CAPM regression of each column on the benchmark.

    Beta on its own says how much market exposure a portfolio carries; it says
    nothing about whether the rest of the return was skill or noise. The alpha
    t-statistic and R² are what turn a number into a claim: an alpha of 2%
    with a t-stat of 0.4 has not been demonstrated.

    Args:
        r: Portfolio returns — a stream, or a frame of them.
        rb: Benchmark returns, over the same dates.
        riskfree_rate: Annualized risk-free rate, as a fraction.
        periods_per_year: Annualization basis — 252 for daily, 12 for monthly.

    Returns:
        One row per column of ``r`` with the annualized alpha, beta, the
        t-statistic of alpha, R², and residual (idiosyncratic) volatility.

    Raises:
        MissingDependencyError: If statsmodels is not installed. Install it
            with ``finport-optengine[stats]``.
    """
    frame, bench = _aligned(r, rb)
    rf = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    x = sm.add_constant((bench - rf).rename("benchmark"))

    rows: dict[str, dict[str, float]] = {}
    for col in frame.columns:
        y = frame[col] - rf
        valid = y.notna()
        model = sm.OLS(y[valid], x.loc[valid.index[valid]]).fit()
        alpha_period = float(model.params.iloc[0])
        rows[col] = {
            "Alpha (annualized)": (1 + alpha_period) ** periods_per_year - 1,
            "Beta": float(model.params.iloc[1]),
            "Alpha t-stat": float(model.tvalues.iloc[0]),
            "R-squared": float(model.rsquared),
            "Residual Vol": float(
                np.sqrt(model.mse_resid) * np.sqrt(periods_per_year)
            ),
        }
    return pd.DataFrame(rows).T


def beta(r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame) -> pd.Series:
    """OLS beta of each column of ``r`` against benchmark ``rb``.

    Args:
        r: Portfolio returns — a stream, or a frame of them.
        rb: Benchmark returns, over the same dates.

    Returns:
        One value per column of ``r``.

    Raises:
        MissingDependencyError: If statsmodels is not installed. Install it
            with ``finport-optengine[stats]``.
    """
    frame, bench = _aligned(r, rb)
    out: dict[str, float] = {}
    for col in frame.columns:
        y, x = _pair(frame, bench, col)
        out[col] = float(
            sm.OLS(y, sm.add_constant(x.rename("benchmark"))).fit().params.iloc[1]
        )
    return pd.Series(out, name="Beta")


def conditional_beta(
    r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame
) -> pd.DataFrame:
    """Beta measured separately in up and down benchmark periods.

    A portfolio with the same beta in both directions is symmetric; one whose
    down-beta exceeds its up-beta is the shape investors dislike most, and a
    single full-sample beta averages the two into silence.

    Args:
        r: Portfolio returns — a stream, or a frame of them.
        rb: Benchmark returns, over the same dates.

    Returns:
        One row per column of ``r``, with an up-beta and a down-beta column.

    Raises:
        MissingDependencyError: If statsmodels is not installed. Install it
            with ``finport-optengine[stats]``.
    """
    frame, aligned_bench = _aligned(r, rb)
    rows: dict[str, dict[str, float]] = {}
    for col in frame.columns:
        series, bench = _pair(frame, aligned_bench, col)
        entry: dict[str, float] = {}
        for label, mask in (("Up Beta", bench > 0), ("Down Beta", bench < 0)):
            if int(mask.sum()) < 3:
                entry[label] = float("nan")
                continue
            x = sm.add_constant(bench[mask].rename("benchmark"))
            entry[label] = float(sm.OLS(series[mask], x).fit().params.iloc[1])
        entry["Beta Asymmetry"] = entry.get("Down Beta", np.nan) - entry.get(
            "Up Beta", np.nan
        )
        rows[col] = entry
    return pd.DataFrame(rows).T


def tracking_error(
    r: pd.Series | pd.DataFrame,
    rb: pd.Series | pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.Series | float:
    """Annualized volatility of the return difference versus the benchmark.

    Args:
        r: Portfolio returns — a stream, or a frame of them.
        rb: Benchmark returns, over the same dates.
        periods_per_year: Annualization basis — 252 for daily, 12 for monthly.

    Returns:
        Annualized tracking error as a fraction. One value per column of
        ``r``.
    """
    frame, bench = _aligned(r, rb)
    return annualize_volatility(frame.sub(bench, axis=0), periods_per_year)


def information_ratio(
    r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame, periods_per_year: int = 252
) -> pd.Series:
    """Annualized excess return over annualized tracking error.

    The numerator is the difference of *geometric* annualized returns while
    the denominator annualizes the volatility of the *arithmetic* difference.
    That is the industry convention, and it is worth knowing it is a mismatch:
    over long, volatile samples the two conventions can differ by enough to
    move an IR by a tenth.

    Args:
        r: Portfolio returns — a stream, or a frame of them.
        rb: Benchmark returns, over the same dates.
        periods_per_year: Annualization basis — 252 for daily, 12 for monthly.

    Returns:
        One value per column of ``r``.
    """
    frame, bench = _aligned(r, rb)
    pairs = {col: _pair(frame, bench, col) for col in frame.columns}
    ann_excess = pd.Series(
        {
            col: annualize_returns(y, periods_per_year)
            - annualize_returns(x, periods_per_year)
            for col, (y, x) in pairs.items()
        },
        dtype=float,
    )
    te = pd.Series(
        {col: annualize_volatility(y - x, periods_per_year) for col, (y, x) in pairs.items()},
        dtype=float,
    )
    return ann_excess / te


def batting_average(
    r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame
) -> pd.Series:
    """Share of periods in which the portfolio beat the benchmark.

    The most honest single number about consistency, and the one that most
    often contradicts the information ratio: a portfolio can win 4 periods in
    10 and still post a strong IR because the 4 were large. Reported next to
    :func:`up_down_number_ratio` so the reader can see which of the two
    stories the excess return is telling.

    Args:
        r: Portfolio returns — a stream, or a frame of them.
        rb: Benchmark returns, over the same dates.

    Returns:
        A fraction in ``[0, 1]``, one per column of ``r``.
    """
    frame, bench = _aligned(r, rb)
    diff = frame.sub(bench, axis=0)
    return pd.Series(
        {col: float((diff[col].dropna() > 0).mean()) for col in diff.columns},
        name="Batting Average",
    )


def up_down_number_ratio(
    r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame
) -> pd.DataFrame:
    """How often the portfolio outperformed in rising and in falling markets.

    Capture ratios are magnitudes; these are counts. A manager can post a
    flattering down-capture from a single well-timed month while having lost
    to the benchmark in most of the others, and only the count reveals it.

    Args:
        r: Portfolio returns — a stream, or a frame of them.
        rb: Benchmark returns, over the same dates.

    Returns:
        One row per column of ``r``, with the share of rising and of falling
        benchmark periods in which the portfolio came out ahead.
    """
    frame, bench = _aligned(r, rb)
    diff = frame.sub(bench, axis=0)
    up, down = bench > 0, bench < 0
    rows: dict[str, dict[str, float]] = {}
    for col in frame.columns:
        rows[col] = {
            "Up Number Ratio": (
                float((diff.loc[up, col].dropna() > 0).mean())
                if int(up.sum()) else float("nan")
            ),
            "Down Number Ratio": (
                float((diff.loc[down, col].dropna() > 0).mean())
                if int(down.sum()) else float("nan")
            ),
        }
    return pd.DataFrame(rows).T


def treynor_ratio(
    r: pd.Series | pd.DataFrame,
    rb: pd.Series | pd.DataFrame,
    riskfree_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """Annualized excess-over-cash return per unit of *systematic* risk.

    Sharpe divides by total volatility, which charges a portfolio for
    diversifiable risk it may be holding deliberately. Treynor divides by beta
    instead, so it ranks portfolios by the return earned on the market
    exposure alone — the right comparison for a sleeve inside a larger book,
    the wrong one for a standalone allocation.

    **Convention:** the numerator is *geometric* — :func:`annualize_returns`
    compounds it — while
    :func:`~optimization_engine.analytics.performance.sharpe_ratio` now
    annualizes arithmetically. The two ratios in the same summary do not share
    a numerator, so a Treynor cannot be read as "the Sharpe with beta in the
    denominator". Deliberate and unchanged; stated so the mixing is visible.

    Args:
        r: Portfolio returns — a stream, or a frame of them.
        rb: Benchmark returns, over the same dates.
        riskfree_rate: Annualized risk-free rate, as a fraction.
        periods_per_year: Annualization basis — 252 for daily, 12 for monthly.

    Returns:
        One value per column of ``r``.

    Raises:
        MissingDependencyError: If statsmodels is not installed. Install it
            with ``finport-optengine[stats]``.
    """
    frame, bench = _aligned(r, rb)
    betas = beta(frame, bench)
    ann_excess = annualize_returns(frame, periods_per_year) - riskfree_rate
    out = ann_excess / betas.replace(0.0, np.nan)
    return pd.Series(out, name="Treynor Ratio")


def m_squared(
    r: pd.Series | pd.DataFrame,
    rb: pd.Series | pd.DataFrame,
    riskfree_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """Modigliani-Modigliani: the portfolio's return at the benchmark's risk.

    Levers the portfolio with cash until its volatility equals the
    benchmark's, then reports the resulting return. It says the same thing as
    the Sharpe ratio — the ranking is identical — but in percentage points
    rather than in ratio units, which is what makes it answerable: "this
    portfolio earned 1.8% a year more than the index at the same risk".

    **Convention:** it is built from
    :func:`~optimization_engine.analytics.performance.sharpe_ratio`, so it
    inherits that function's *arithmetic* annualization and moves with it —
    unlike :func:`information_ratio` and :func:`treynor_ratio`, whose
    numerators stay geometric. On the sample panel the switch raised the
    equal-weight Sharpe from 0.5950 to 0.6238, and M-squared rises in the
    same proportion.

    Args:
        r: Portfolio returns — a stream, or a frame of them.
        rb: Benchmark returns, over the same dates.
        riskfree_rate: Annualized risk-free rate, as a fraction.
        periods_per_year: Annualization basis — 252 for daily, 12 for monthly.

    Returns:
        An annualized return as a fraction, one per column of ``r``. Compare
        it against the benchmark's own annualized return.
    """
    frame, bench = _aligned(r, rb)
    out: dict[str, float] = {}
    for col in frame.columns:
        y, x = _pair(frame, bench, col)
        sr = float(sharpe_ratio(y, riskfree_rate, periods_per_year))
        out[col] = sr * float(annualize_volatility(x, periods_per_year)) + riskfree_rate
    return pd.Series(out, name="M-squared")


def appraisal_ratio(
    r: pd.Series | pd.DataFrame,
    rb: pd.Series | pd.DataFrame,
    riskfree_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """Jensen's alpha per unit of residual (stock-specific) volatility.

    Treynor-Black's measure of pure selection skill. Where the information
    ratio divides excess return by total tracking error — which includes any
    deliberate beta tilt — this isolates the part of the deviation that the
    benchmark cannot explain at all.

    Args:
        r: Portfolio returns — a stream, or a frame of them.
        rb: Benchmark returns, over the same dates.
        riskfree_rate: Annualized risk-free rate, as a fraction.
        periods_per_year: Annualization basis — 252 for daily, 12 for monthly.

    Returns:
        One value per column of ``r``.

    Raises:
        MissingDependencyError: If statsmodels is not installed. Install it
            with ``finport-optengine[stats]``.
    """
    stats = regression_stats(r, rb, riskfree_rate, periods_per_year)
    return pd.Series(
        stats["Alpha (annualized)"] / stats["Residual Vol"].replace(0.0, np.nan),
        name="Appraisal Ratio",
    )


def excess_returns(
    r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame
) -> pd.DataFrame:
    """Per-period portfolio-minus-benchmark returns, on their common dates.

    Args:
        r: Portfolio returns — a stream, or a frame of them.
        rb: Benchmark returns, over the same dates.

    Returns:
        A frame indexed by the shared dates, one column per column of ``r``.
    """
    frame, bench = _aligned(r, rb)
    return frame.sub(bench, axis=0)


def relative_drawdown(
    r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame
) -> pd.DataFrame:
    """Drawdown of the portfolio's wealth *relative to* the benchmark's.

    The series a plan sponsor actually watches: how far the portfolio has
    fallen behind its index since the last time it was ahead. It is not the
    drawdown of the excess return — that would compound a difference of
    returns as if it were a return — but the drawdown of the ratio of the two
    wealth curves, which is what "behind by 8% since 2022" means.

    The relative high-water mark starts at 1 — level with the benchmark at
    inception — for the reason
    :func:`~optimization_engine.analytics.risk.drawdown_series` starts its
    peak at the capital invested: a portfolio that falls behind on the first
    period is behind from that period on.

    Args:
        r: Portfolio returns — a stream, or a frame of them.
        rb: Benchmark returns, over the same dates.

    Returns:
        A frame indexed by the shared dates, one column per column of ``r``,
        zero whenever the portfolio is at a new high against the benchmark and
        negative below it.
    """
    frame, bench = _aligned(r, rb)
    out: dict[str, pd.Series] = {}
    for col in frame.columns:
        y, x = _pair(frame, bench, col)
        ratio = (1.0 + y).cumprod() / (1.0 + x).cumprod()
        out[col] = ratio / ratio.cummax().clip(lower=1.0) - 1.0
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def relative_summary_extras(
    r: pd.Series | pd.DataFrame,
    rb: pd.Series | pd.DataFrame,
    riskfree_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """The second tier of relative statistics, for each column of ``r``.

    Kept separate from :func:`summary_relative` so the headline table stays
    short enough to read at a glance, and so a caller that wants everything
    can ask for it explicitly.

    Args:
        r: Portfolio returns — a stream, or a frame of them.
        rb: Benchmark returns, over the same dates.
        riskfree_rate: Annualized risk-free rate, as a fraction.
        periods_per_year: Annualization basis — 252 for daily, 12 for monthly.

    Returns:
        One row per column of ``r`` with the batting average, Treynor ratio,
        M², appraisal ratio, correlation to the benchmark, downside tracking
        error, worst relative drawdown, the probability that the excess
        return is positive, and the up/down outperformance counts.

    Raises:
        MissingDependencyError: If statsmodels is not installed. Install it
            with ``finport-optengine[stats]``.
    """
    frame, bench = _aligned(r, rb)
    diff = frame.sub(bench, axis=0)
    rel_dd = relative_drawdown(frame, bench)
    out = pd.DataFrame(
        {
            "Batting Average": batting_average(frame, bench),
            "Treynor Ratio": treynor_ratio(
                frame, bench, riskfree_rate, periods_per_year
            ),
            "M-squared": m_squared(frame, bench, riskfree_rate, periods_per_year),
            "Appraisal Ratio": appraisal_ratio(
                frame, bench, riskfree_rate, periods_per_year
            ),
            "Correlation": pd.Series(
                {c: float(frame[c].corr(bench)) for c in frame.columns}
            ),
            "Downside T.E.": pd.Series(
                {
                    c: float(
                        downside_deviation(diff[c].dropna(), mar=0.0)
                        * np.sqrt(periods_per_year)
                    )
                    for c in diff.columns
                }
            ),
            "Worst Relative Drawdown": rel_dd.min(),
            "Prob. Excess > 0": pd.Series(
                {
                    c: probabilistic_sharpe_ratio(
                        diff[c].dropna(), 0.0, 0.0, periods_per_year
                    )
                    for c in diff.columns
                }
            ),
        }
    )
    return out.join(up_down_number_ratio(frame, bench))


def active_share(weights: pd.Series, benchmark_weights: pd.Series) -> float:
    """Half the sum of absolute weight differences versus the benchmark.

    0 means the portfolio is the benchmark; 1 means it shares no holding with
    it. Unlike tracking error, active share is computed from positions alone,
    so it cannot be flattered by a quiet market.

    Args:
        weights: Portfolio weights, as fractions of the book.
        benchmark_weights: The benchmark's weights over the same universe.
            Assets it does not hold count as zero.

    Returns:
        A fraction in ``[0, 1]`` for a long-only book.
    """
    assets = weights.index.union(benchmark_weights.index)
    w = weights.reindex(assets).fillna(0.0)
    b = benchmark_weights.reindex(assets).fillna(0.0)
    return float((w - b).abs().sum() / 2.0)


def summary_relative(
    r: pd.Series | pd.DataFrame,
    rb: pd.Series | pd.DataFrame,
    periods_per_year: int = 252,
    riskfree_rate: float = 0.0,
    extended: bool = False,
) -> pd.DataFrame:
    """Benchmark-relative summary for each column of ``r``.

    Args:
        r: Portfolio returns.
        rb: Benchmark returns.
        periods_per_year: Observations per year.
        riskfree_rate: Annual risk-free rate, used for the CAPM alpha.
        extended: Add alpha, its t-statistic, R², residual volatility,
            up/down betas, and the second tier from
            :func:`relative_summary_extras` — batting average, Treynor, M²,
            the appraisal ratio, downside tracking error and the worst
            relative drawdown.
    """
    frame, bench = _aligned(r, rb)
    ann_excess = annualize_returns(frame, periods_per_year) - annualize_returns(
        bench, periods_per_year
    )
    te = annualize_volatility(frame.sub(bench, axis=0), periods_per_year)
    up = up_capture(frame, bench)
    down = down_capture(frame, bench)

    out = pd.DataFrame(
        {
            "Annualized Excess": ann_excess,
            "Annualized T.E.": te,
            "Information Ratio": ann_excess / te,
            "Beta": beta(frame, bench),
            "Up Capture": up,
            "Down Capture": down,
            "Capture": up / down,
        }
    )
    if extended:
        out = (
            out.join(
                regression_stats(frame, bench, riskfree_rate, periods_per_year)[
                    ["Alpha (annualized)", "Alpha t-stat", "R-squared", "Residual Vol"]
                ]
            )
            .join(conditional_beta(frame, bench))
            .join(
                relative_summary_extras(
                    frame, bench, riskfree_rate, periods_per_year
                )
            )
        )
    return out
