"""One performance report: absolute, relative, and the evidence behind both.

The engine already computes every statistic here somewhere. What it lacked
was a single object that holds the *pair* — how the portfolio did, and how it
did against the thing it is supposed to beat — computed on one aligned sample
and exported from one place. Assembling the two halves separately is how a
report ends up quoting a Sharpe ratio from the full history next to an
information ratio from the benchmark's shorter one.

Everything is derived from the return streams, so the same report describes an
in-sample replay, a costed backtest or a walk-forward track record depending
only on what it is handed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from optimization_engine.analytics.performance import (
    rolling_metrics,
    summary_stats,
)
from optimization_engine.analytics.relative import (
    active_share,
    relative_drawdown,
    summary_relative,
)
from optimization_engine.analytics.risk import drawdown_table

#: Column names used throughout, so the UI and the exporter agree on them.
PORTFOLIO = "Portfolio"
BENCHMARK = "Benchmark"
EXCESS = "Excess"

#: Calendar aggregations offered by :func:`period_returns`, newest pandas
#: alias first. Pandas renamed the period-end offsets in 2.2 and dropped the
#: old spellings in 3.0, so each entry carries its fallback rather than
#: pinning the report to one pandas generation.
_FREQ_ALIASES: dict[str, tuple[str, ...]] = {
    "yearly": ("YE", "A"),
    "quarterly": ("QE", "Q"),
    "monthly": ("ME", "M"),
}

_PERIOD_LABEL_FORMAT = {
    "yearly": "%Y",
    "quarterly": "%Y-Q",
    "monthly": "%Y-%m",
}


def _resample(frame: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Compound returns into calendar buckets, tolerating either pandas alias."""
    last_error: Exception | None = None
    for alias in _FREQ_ALIASES[freq]:
        try:
            return frame.resample(alias).apply(lambda s: (1.0 + s).prod() - 1.0)
        except ValueError as exc:  # unsupported alias on this pandas
            last_error = exc
    raise ValueError(
        f"Could not resample to {freq!r} with any known pandas alias "
        f"({', '.join(_FREQ_ALIASES[freq])}): {last_error}"
    )


def period_returns(
    returns: pd.DataFrame | pd.Series, freq: str = "yearly"
) -> pd.DataFrame:
    """Compounded calendar-period returns, one row per period.

    Annualized figures hide the shape of a track record: two portfolios with
    the same 7% a year can have reached it steadily or through one enormous
    year and four flat ones, and only the period table separates them.

    Args:
        returns: Periodic returns; a Series is promoted to one column.
        freq: ``yearly``, ``quarterly`` or ``monthly``.

    Raises:
        ValueError: When ``freq`` is unknown or the index is not datetime-like
            — calendar buckets are undefined without dates.
    """
    if freq not in _FREQ_ALIASES:
        raise ValueError(
            f"Unknown period {freq!r}; expected one of {', '.join(_FREQ_ALIASES)}."
        )
    frame = returns.to_frame() if isinstance(returns, pd.Series) else returns
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(
            "Calendar-period returns need a DatetimeIndex; the returns frame "
            f"is indexed by {type(frame.index).__name__}."
        )
    out = _resample(frame.dropna(how="all"), freq)
    if PORTFOLIO in out.columns and BENCHMARK in out.columns:
        # Compounding the per-period difference is not the difference of the
        # compounded returns, and a table where the third column does not
        # equal the first minus the second is read as an error every time.
        out[EXCESS] = out[PORTFOLIO] - out[BENCHMARK]
    if freq == "quarterly":
        labels = [f"{i.year}-Q{i.quarter}" for i in out.index]
    else:
        labels = out.index.strftime(_PERIOD_LABEL_FORMAT[freq])
    out.index = pd.Index(labels, name=freq.capitalize())
    return out


def rolling_relative(
    portfolio: pd.Series,
    benchmark: pd.Series,
    window: int,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Rolling excess return, tracking error, information ratio and beta.

    A full-sample information ratio is one number for a decade. This is the
    frame that says whether the outperformance was earned throughout or in a
    single regime — and whether the beta the portfolio is running today is the
    one the full-sample regression reports.

    Args:
        portfolio: The portfolio's return stream.
        benchmark: The benchmark's, over the same dates.
        window: Rolling window length, in periods. At least 2.
        periods_per_year: Annualization basis for the excess return, the
            tracking error and the information ratio.

    Returns:
        A frame indexed like the inputs, with one column per rolling metric
        and NaN over the first ``window - 1`` rows.

    Raises:
        ValueError: If ``window`` is below 2.
    """
    if window < 2:
        raise ValueError(f"Rolling window must be at least 2 periods; got {window}.")
    common = portfolio.dropna().index.intersection(benchmark.dropna().index)
    p = portfolio.loc[common]
    b = benchmark.loc[common]
    diff = p - b

    ann_excess = diff.rolling(window).mean() * periods_per_year
    te = diff.rolling(window).std() * np.sqrt(periods_per_year)
    covariance = p.rolling(window).cov(b)
    variance = b.rolling(window).var()
    return pd.DataFrame(
        {
            "rolling_excess": ann_excess,
            "rolling_tracking_error": te,
            "rolling_information_ratio": ann_excess / te.replace(0.0, np.nan),
            "rolling_beta": covariance / variance.replace(0.0, np.nan),
            "rolling_correlation": p.rolling(window).corr(b),
        }
    )


@dataclass
class PerformanceReport:
    """Absolute and relative performance of one portfolio, assembled once.

    Attributes:
        returns: Portfolio (and, when set, benchmark and excess) return
            streams on their common dates.
        absolute: Extended :func:`summary_stats` for every stream.
        relative: Extended :func:`summary_relative`, or ``None`` without a
            benchmark.
        periods: Calendar-period returns, portfolio versus benchmark.
        rolling_absolute: Rolling return/volatility/Sharpe/drawdown.
        rolling_relative: Rolling excess/TE/IR/beta, or ``None``.
        drawdowns: The portfolio's worst drawdown episodes.
        relative_drawdown: Shortfall against the benchmark's wealth curve.
        benchmark_label: Display name of the benchmark, if any.
        active_share: Position-based active share, when the benchmark has
            weights and the portfolio's are known.
        periods_per_year: Observations per year, carried for the reader.
        riskfree_rate: Annual rate the ratios were computed at.
        metadata: Free-form provenance (which backtest, what costs).
    """

    returns: pd.DataFrame
    absolute: pd.DataFrame
    relative: pd.DataFrame | None
    periods: pd.DataFrame
    rolling_absolute: pd.DataFrame
    rolling_relative_frame: pd.DataFrame | None
    drawdowns: pd.DataFrame
    relative_drawdown_series: pd.Series | None
    benchmark_label: str | None
    active_share: float | None
    periods_per_year: int
    riskfree_rate: float
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- views --------------------------------------------------------------

    @property
    def has_benchmark(self) -> bool:
        """Whether this report carries benchmark-relative figures.

        Returns:
            ``True`` when a benchmark was resolved and the relative block was
            computed. Every relative accessor returns ``nan`` when it is ``False``.
        """
        return self.relative is not None

    def _abs(self, metric: str) -> float:
        try:
            return float(self.absolute.loc[PORTFOLIO, metric])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    def _rel(self, metric: str) -> float:
        if self.relative is None:
            return float("nan")
        try:
            return float(self.relative.loc[PORTFOLIO, metric])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    def headline(self) -> dict[str, float]:
        """The numbers a reader looks at first, absolute and relative together.

        Returned as one flat mapping rather than two so a caller cannot render
        the absolute block from this report and the relative block from
        another.
        """
        out = {
            "annualized_return": self._abs("Annualized Return"),
            "annualized_volatility": self._abs("Annualized Vol"),
            "sharpe_ratio": self._abs("Sharpe Ratio"),
            "sortino_ratio": self._abs("Sortino Ratio"),
            "max_drawdown": self._abs("Max Drawdown"),
            "calmar_ratio": self._abs("Calmar Ratio"),
            "hit_rate": self._abs("Hit Rate"),
            "probabilistic_sharpe": self._abs("Prob. Sharpe > 0"),
        }
        if self.has_benchmark:
            out.update(
                {
                    "excess_return": self._rel("Annualized Excess"),
                    "tracking_error": self._rel("Annualized T.E."),
                    "information_ratio": self._rel("Information Ratio"),
                    "beta": self._rel("Beta"),
                    "alpha": self._rel("Alpha (annualized)"),
                    "alpha_t_stat": self._rel("Alpha t-stat"),
                    "up_capture": self._rel("Up Capture"),
                    "down_capture": self._rel("Down Capture"),
                    "batting_average": self._rel("Batting Average"),
                    "worst_relative_drawdown": self._rel("Worst Relative Drawdown"),
                }
            )
            if self.active_share is not None:
                out["active_share"] = float(self.active_share)
        return out

    def metrics(self) -> pd.DataFrame:
        """Every statistic as tidy long-form rows: block, series, metric, value.

        The shape that survives a CSV round-trip and a pivot table, and the
        one that lets a reader diff two runs without lining up column orders.
        """
        rows: list[dict[str, Any]] = []
        for block, frame in (("Absolute", self.absolute), ("Relative", self.relative)):
            if frame is None:
                continue
            for series_name, row in frame.iterrows():
                for metric, value in row.items():
                    rows.append(
                        {
                            "block": block,
                            "series": str(series_name),
                            "metric": str(metric),
                            "value": (
                                float(value)
                                if isinstance(value, (int, float, np.floating))
                                else value
                            ),
                        }
                    )
        if self.active_share is not None:
            rows.append(
                {
                    "block": "Relative",
                    "series": PORTFOLIO,
                    "metric": "Active Share",
                    "value": float(self.active_share),
                }
            )
        return pd.DataFrame(rows)

    def to_frames(self) -> dict[str, pd.DataFrame]:
        """Named frames for a workbook, skipping whatever this run lacks."""
        frames: dict[str, pd.DataFrame] = {
            "performance_absolute": self.absolute,
            "performance_metrics": self.metrics(),
            "performance_periods": self.periods,
            "performance_rolling": self.rolling_absolute.dropna(how="all"),
            "performance_drawdowns": self.drawdowns,
            "performance_returns": self.returns,
        }
        if self.relative is not None:
            frames["performance_relative"] = self.relative
        if self.rolling_relative_frame is not None:
            frames["performance_rolling_relative"] = (
                self.rolling_relative_frame.dropna(how="all")
            )
        if self.relative_drawdown_series is not None:
            frames["performance_relative_drawdown"] = (
                self.relative_drawdown_series.to_frame("relative_drawdown")
            )
        return frames

    def describe(self) -> str:
        """One paragraph stating what the numbers say, hedged where they should be.

        Written so the caller does not have to decide whether an information
        ratio of 0.3 on two years of data is a claim — it is not, and the
        sentence says so.
        """
        h = self.headline()
        n = int(len(self.returns))
        years = n / max(self.periods_per_year, 1)
        parts = [
            f"Over {years:.1f} years ({n:,} observations) the portfolio "
            f"returned {h['annualized_return']:.2%} a year with "
            f"{h['annualized_volatility']:.2%} volatility "
            f"(Sharpe {h['sharpe_ratio']:.2f}, worst drawdown "
            f"{h['max_drawdown']:.2%})."
        ]
        if self.has_benchmark:
            verb = "ahead" if h["excess_return"] >= 0 else "behind"
            parts.append(
                f"Against {self.benchmark_label or 'the benchmark'} it was "
                f"{abs(h['excess_return']):.2%} a year {verb}, at "
                f"{h['tracking_error']:.2%} tracking error - an information "
                f"ratio of {h['information_ratio']:.2f}, with a beta of "
                f"{h['beta']:.2f}."
            )
            t_stat = h.get("alpha_t_stat", float("nan"))
            if np.isfinite(t_stat) and abs(t_stat) < 2.0:
                parts.append(
                    f"The CAPM alpha of {h['alpha']:.2%} carries a t-statistic "
                    f"of {t_stat:.2f}, which is not distinguishable from zero "
                    "on this sample."
                )
            elif np.isfinite(t_stat):
                parts.append(
                    f"The CAPM alpha of {h['alpha']:.2%} is significant on this "
                    f"sample (t = {t_stat:.2f}); whether it survives out of "
                    "sample is a separate question."
                )
        return " ".join(parts)


def performance_report(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    periods_per_year: int = 252,
    riskfree_rate: float = 0.0,
    portfolio_weights: pd.Series | None = None,
    benchmark_weights: pd.Series | None = None,
    benchmark_label: str | None = None,
    rolling_window: int | None = None,
    period_freq: str = "yearly",
    top_drawdowns: int = 5,
    metadata: dict[str, Any] | None = None,
) -> PerformanceReport:
    """Build a :class:`PerformanceReport` from one or two return streams.

    The portfolio and benchmark are restricted to their common dates *before*
    anything is computed, and the absolute statistics are then computed on
    that same restricted sample — so the Sharpe ratio quoted next to an
    information ratio describes the period the information ratio covers.

    Args:
        portfolio_returns: Periodic portfolio returns.
        benchmark_returns: Periodic benchmark returns. ``None`` produces an
            absolute-only report rather than an error.
        periods_per_year: Observations per year.
        riskfree_rate: Annual risk-free rate for the ratios and the CAPM alpha.
        portfolio_weights: Solved weights, needed only for active share.
        benchmark_weights: Benchmark weights, likewise. Active share is
            reported only when both are given — an external index has no
            positions in the universe and inventing them would be worse than
            leaving the cell empty.
        benchmark_label: Display name for the benchmark.
        rolling_window: Window for the rolling frames. Defaults to one year.
        period_freq: ``yearly``, ``quarterly`` or ``monthly`` calendar table.
        top_drawdowns: How many drawdown episodes to tabulate.
        metadata: Free-form provenance carried through to the export.

    Raises:
        ValueError: When the portfolio series is empty, or when it shares no
            dates with the benchmark.
    """
    portfolio = pd.Series(portfolio_returns).dropna()
    if portfolio.empty:
        raise ValueError(
            "performance_report received an empty portfolio return series."
        )
    portfolio.name = PORTFOLIO
    window = int(rolling_window or max(periods_per_year, 2))
    window = max(2, min(window, len(portfolio)))

    frame = portfolio.to_frame()
    bench: pd.Series | None = None
    if benchmark_returns is not None:
        bench = pd.Series(benchmark_returns).dropna()
        common = portfolio.index.intersection(bench.index)
        if len(common) == 0:
            raise ValueError(
                "The portfolio and the benchmark share no dates, so no "
                "relative metric could be computed. Check that both cover the "
                "same period and the same frequency."
            )
        portfolio = portfolio.loc[common]
        bench = bench.loc[common]
        bench.name = BENCHMARK
        frame = pd.concat([portfolio, bench], axis=1)
        frame[EXCESS] = frame[PORTFOLIO] - frame[BENCHMARK]

    absolute = summary_stats(
        frame[[c for c in frame.columns if c != EXCESS]],
        periods_per_year=periods_per_year,
        riskfree_rate=riskfree_rate,
        extended=True,
    )

    relative = None
    rolling_rel = None
    rel_dd = None
    share = None
    if bench is not None:
        relative = summary_relative(
            portfolio.to_frame(PORTFOLIO),
            bench,
            periods_per_year=periods_per_year,
            riskfree_rate=riskfree_rate,
            extended=True,
        )
        rolling_rel = rolling_relative(portfolio, bench, window, periods_per_year)
        rel_dd = relative_drawdown(portfolio.to_frame(PORTFOLIO), bench)[PORTFOLIO]
        rel_dd.name = "relative_drawdown"
        if portfolio_weights is not None and benchmark_weights is not None:
            share = active_share(portfolio_weights, benchmark_weights)

    return PerformanceReport(
        returns=frame,
        absolute=absolute,
        relative=relative,
        periods=period_returns(frame, period_freq),
        rolling_absolute=rolling_metrics(
            portfolio, window, riskfree_rate, periods_per_year
        ),
        rolling_relative_frame=rolling_rel,
        drawdowns=drawdown_table(portfolio, top=top_drawdowns),
        relative_drawdown_series=rel_dd,
        benchmark_label=(benchmark_label if bench is not None else None),
        active_share=share,
        periods_per_year=int(periods_per_year),
        riskfree_rate=float(riskfree_rate),
        metadata=dict(metadata or {}),
    )


def compare_performance(
    streams: dict[str, pd.Series],
    benchmark_returns: pd.Series | None = None,
    periods_per_year: int = 252,
    riskfree_rate: float = 0.0,
) -> pd.DataFrame:
    """Absolute and relative statistics for several portfolios, side by side.

    Used by the comparison views, where the question is not "how did this do"
    but "which of these did better, and by how much against the same
    benchmark".

    Args:
        streams: ``label -> return stream``. The labels become the columns.
        benchmark_returns: A common benchmark. When ``None``, only the
            absolute statistics are computed.
        periods_per_year: Annualization basis for every stream.
        riskfree_rate: Annual risk-free rate used by the ratio metrics.

    Returns:
        A frame with one row per statistic and one column per stream, every
        stream scored over the periods they *all* cover. A stream that starts
        later — a walk-forward beside a fitted replay — shortens the window
        for the others rather than being padded to their length, because a
        padded period would be scored as a zero return and the comparison
        would flatter whichever stream was shortest. With a benchmark, the
        relative statistics are joined on with a ``" (rel)"`` suffix where a
        name would otherwise collide.

    Raises:
        ValueError: If ``streams`` is empty.
        MissingDependencyError: If a benchmark is supplied and statsmodels is
            not installed. Install it with ``finport-optengine[stats]``.
    """
    if not streams:
        raise ValueError("compare_performance needs at least one return stream.")
    frame = pd.concat(streams, axis=1, join="inner").dropna(how="any")
    out = summary_stats(
        frame,
        periods_per_year=periods_per_year,
        riskfree_rate=riskfree_rate,
        extended=True,
    )
    if benchmark_returns is None:
        return out
    return out.join(
        summary_relative(
            frame,
            benchmark_returns,
            periods_per_year=periods_per_year,
            riskfree_rate=riskfree_rate,
            extended=True,
        ),
        rsuffix=" (rel)",
    )


__all__ = [
    "BENCHMARK",
    "EXCESS",
    "PORTFOLIO",
    "PerformanceReport",
    "compare_performance",
    "performance_report",
    "period_returns",
    "rolling_relative",
]
