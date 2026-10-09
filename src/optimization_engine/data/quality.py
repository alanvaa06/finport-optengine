"""Data-quality analysis for price and return panels.

Optimizers are exquisitely sensitive to their inputs and completely silent
about them: a stale price series looks like a low-volatility asset, a short
history looks like a confident estimate, and a panel where two assets never
overlap produces a correlation the sample never observed. All three are
invisible in the weights.

This module surfaces those problems *before* the solve, with the severity
and the fix attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Severity = Literal["error", "warning", "info"]

#: A return this large is almost always a corporate action or a bad tick
#: rather than a market move, at daily frequency.
EXTREME_RETURN_THRESHOLD = 0.50

#: Runs of identical prices at or above this length suggest a stale feed.
STALE_RUN_THRESHOLD = 5

#: Share of exactly-zero returns above which a series is flagged. Past one
#: period in four the price is updating less often than the panel is sampled
#: — a weekly NAV in a daily panel, a quote nobody trades — and its volatility
#: and correlations are read off the few periods in which it moved. Short,
#: scattered repeats never reach :data:`STALE_RUN_THRESHOLD`, so this catches
#: what the run-length check cannot.
ZERO_RETURN_SHARE_THRESHOLD = 0.25


@dataclass(frozen=True)
class DataIssue:
    """One problem found in the input panel."""

    severity: Severity
    code: str
    asset: str | None
    message: str
    suggestion: str

    def describe(self) -> str:
        """The issue and its fix, on one line.

        Returns:
            Something like ``"EM_Equity: 34 missing observations → align the panel
            or shorten the window"``, prefixed with the asset when the issue is
            scoped to one.
        """
        where = f"{self.asset}: " if self.asset else ""
        return f"{where}{self.message} -> {self.suggestion}"


@dataclass
class DataQualityReport:
    """Everything worth knowing about a price panel before optimizing.

    Attributes:
        issues: Problems found, errors first.
        per_asset: One row per asset — coverage, gaps, stale runs, extremes.
        overlap: Pairwise count of jointly observed periods. The diagonal is
            each asset's own observation count.
        common_start: First date on which every asset has data.
        common_end: Last such date.
        n_common_periods: Observations available to *all* assets at once —
            the sample the covariance matrix is really estimated on.
    """

    issues: tuple[DataIssue, ...] = field(default_factory=tuple)
    per_asset: pd.DataFrame = field(default_factory=pd.DataFrame)
    overlap: pd.DataFrame = field(default_factory=pd.DataFrame)
    common_start: pd.Timestamp | None = None
    common_end: pd.Timestamp | None = None
    n_common_periods: int = 0

    @property
    def errors(self) -> tuple[DataIssue, ...]:
        """The issues serious enough to stop an optimization, in the order found."""
        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[DataIssue, ...]:
        """The issues worth reading before trusting the numbers, but not fatal."""
        return tuple(i for i in self.issues if i.severity == "warning")

    @property
    def is_clean(self) -> bool:
        """Whether the panel raised nothing at all — no errors and no warnings."""
        return not self.errors and not self.warnings

    @property
    def is_usable(self) -> bool:
        """Whether the panel can be optimized on.

        Returns:
            ``True`` when there are no errors. Warnings do not block a run: the
            CLI proceeds and prints them, and only ``--strict`` turns an error
            into a refusal.
        """
        return not self.errors

    def describe(self) -> str:
        """Every issue found, one bulleted line each.

        Returns:
            A multi-line string, or a statement that nothing was found.
        """
        if not self.issues:
            return "No data-quality issues found."
        return "\n".join(f"* {i.describe()}" for i in self.issues)


def _longest_repeat_run(series: pd.Series) -> int:
    """Longest run of consecutive identical values."""
    values = series.dropna().values
    if len(values) < 2:
        return 0
    changed = np.r_[True, values[1:] != values[:-1]]
    run_ids = np.cumsum(changed)
    counts = np.bincount(run_ids)
    return int(counts.max()) if len(counts) else 0


def analyze_prices(
    prices: pd.DataFrame,
    periods_per_year: int = 252,
    min_observations_per_asset: int = 30,
) -> DataQualityReport:
    """Inspect a price panel for the problems that quietly break optimizers.

    Args:
        prices: Wide price panel, ``DatetimeIndex`` × assets.
        periods_per_year: Used to express coverage in years and to judge
            whether the history is long enough for the universe size.
        min_observations_per_asset: Below this an asset's statistics are not
            worth estimating.
    """
    issues: list[DataIssue] = []
    if prices is None or prices.empty:
        return DataQualityReport(
            issues=(
                DataIssue(
                    "error", "empty_panel", None,
                    "The price panel is empty.",
                    "Load a file with at least one asset and two dates.",
                ),
            )
        )

    if not isinstance(prices.index, pd.DatetimeIndex):
        issues.append(
            DataIssue(
                "error", "non_datetime_index", None,
                "The index is not a set of dates.",
                "Make the first column the date column when loading the file.",
            )
        )
        return DataQualityReport(issues=tuple(issues))

    if not prices.index.is_monotonic_increasing:
        issues.append(
            DataIssue(
                "warning", "unsorted_index", None,
                "Dates are not in ascending order.",
                "Sort the panel by date; returns computed from an unsorted "
                "index are meaningless.",
            )
        )
    duplicated = int(prices.index.duplicated().sum())
    if duplicated:
        issues.append(
            DataIssue(
                "error", "duplicate_dates", None,
                f"{duplicated} duplicate date(s) in the index.",
                "Remove or aggregate duplicate rows before loading.",
            )
        )

    rows: list[dict[str, object]] = []
    # ``fill_method=None`` explicitly: pandas 2.0-2.2 pad by default and 3.0
    # does not, and a padded gap shows up here as a fabricated zero return
    # followed by a fabricated extreme one — the two statistics this function
    # exists to count.
    returns = prices.pct_change(fill_method=None)

    for asset in prices.columns:
        series = prices[asset]
        observed = series.dropna()
        n_obs = len(observed)
        first = observed.index.min() if n_obs else pd.NaT
        last = observed.index.max() if n_obs else pd.NaT
        # Gaps interior to the asset's own history, not the leading NaNs of a
        # series that simply started later.
        interior = series.loc[first:last] if n_obs else series
        n_missing = int(interior.isna().sum())
        stale_run = _longest_repeat_run(observed)
        asset_returns = returns[asset].dropna()
        n_extreme = int((asset_returns.abs() > EXTREME_RETURN_THRESHOLD).sum())
        n_zero = int((asset_returns == 0).sum())
        n_nonpositive = int((observed <= 0).sum())

        rows.append(
            {
                "observations": n_obs,
                "first_date": first,
                "last_date": last,
                "years": n_obs / periods_per_year if periods_per_year else np.nan,
                "missing_interior": n_missing,
                "longest_stale_run": stale_run,
                "extreme_returns": n_extreme,
                "zero_return_share": (
                    n_zero / len(asset_returns) if len(asset_returns) else np.nan
                ),
                "annualized_vol": (
                    float(asset_returns.std() * np.sqrt(periods_per_year))
                    if len(asset_returns) > 1
                    else np.nan
                ),
            }
        )

        if n_obs == 0:
            issues.append(
                DataIssue(
                    "error", "all_missing", str(asset),
                    "Every value is missing.",
                    "Drop this asset from the universe or fix its source.",
                )
            )
            continue
        if n_nonpositive:
            issues.append(
                DataIssue(
                    "error", "non_positive_price", str(asset),
                    f"{n_nonpositive} price(s) at or below zero.",
                    "Percentage returns are undefined across a zero. Fix the "
                    "series or drop the asset.",
                )
            )
        if n_obs < min_observations_per_asset:
            issues.append(
                DataIssue(
                    "error", "too_few_observations", str(asset),
                    f"Only {n_obs} observations.",
                    f"At least {min_observations_per_asset} are needed to "
                    "estimate anything. Drop the asset or load more history.",
                )
            )
        if n_missing:
            share = n_missing / max(len(interior), 1)
            issues.append(
                DataIssue(
                    "error" if share > 0.10 else "warning",
                    "interior_gaps", str(asset),
                    f"{n_missing} missing value(s) inside its own history "
                    f"({share:.1%}).",
                    "Forward-fill, interpolate, or drop the asset - gaps make "
                    "each pairwise covariance rest on a different sample.",
                )
            )
        if stale_run >= STALE_RUN_THRESHOLD:
            issues.append(
                DataIssue(
                    "warning", "stale_prices", str(asset),
                    f"Price unchanged for {stale_run} consecutive periods.",
                    "A stale feed looks like low volatility to the optimizer, "
                    "which will overweight this asset. Check the source.",
                )
            )
        zero_share = n_zero / len(asset_returns) if len(asset_returns) else 0.0
        if zero_share > ZERO_RETURN_SHARE_THRESHOLD:
            issues.append(
                DataIssue(
                    "warning", "zero_returns", str(asset),
                    f"{zero_share:.0%} of its returns are exactly zero.",
                    "The price updates less often than the panel is sampled, "
                    "so its volatility and correlations are understated. "
                    "Use a lower frequency for this asset, or check the source.",
                )
            )
        if n_extreme:
            issues.append(
                DataIssue(
                    "warning", "extreme_returns", str(asset),
                    f"{n_extreme} period(s) with a move beyond "
                    f"+/-{EXTREME_RETURN_THRESHOLD:.0%}.",
                    "Usually an unadjusted split or dividend rather than a "
                    "market move. Use adjusted prices.",
                )
            )

    per_asset = pd.DataFrame(rows, index=prices.columns)

    # Overlap: what the covariance matrix is really estimated on.
    mask = prices.notna()
    overlap = pd.DataFrame(
        mask.T.values.astype(int) @ mask.values.astype(int),
        index=prices.columns,
        columns=prices.columns,
    )
    complete = prices.dropna(how="any")
    common_start = complete.index.min() if len(complete) else None
    common_end = complete.index.max() if len(complete) else None
    n_common = len(complete)

    n_assets = prices.shape[1]
    if n_common == 0 and n_assets > 1:
        issues.append(
            DataIssue(
                "error", "no_common_history", None,
                "No date has data for every asset at once.",
                "The assets' histories do not overlap. Shorten the universe or "
                "the date range so a common sample exists.",
            )
        )
    elif n_assets > 1:
        longest = int(per_asset["observations"].max())
        if n_common < 0.5 * longest:
            issues.append(
                DataIssue(
                    "warning", "short_common_history", None,
                    f"Only {n_common} of {longest} periods have every asset "
                    "present.",
                    "The covariance matrix will be estimated on the common "
                    f"window ({n_common} periods). Drop the newest assets to "
                    "recover history, or accept the shorter sample.",
                )
            )
        if n_common <= n_assets:
            issues.append(
                DataIssue(
                    "error", "singular_sample", None,
                    f"{n_common} common observations for {n_assets} assets "
                    "(T <= N).",
                    "The sample covariance is singular. Use a shrinkage "
                    "estimator or HRP, or shorten the universe.",
                )
            )
        elif n_common < 10 * n_assets:
            issues.append(
                DataIssue(
                    "warning", "thin_sample", None,
                    f"{n_common} common observations for {n_assets} assets "
                    f"(T/N = {n_common / n_assets:.1f}).",
                    "Below T/N ~ 10 the covariance is mostly noise. Prefer "
                    "ledoit_wolf/oas shrinkage or HRP over the sample estimator.",
                )
            )

    order = {"error": 0, "warning": 1, "info": 2}
    issues.sort(key=lambda i: order[i.severity])

    return DataQualityReport(
        issues=tuple(issues),
        per_asset=per_asset,
        overlap=overlap,
        common_start=common_start,
        common_end=common_end,
        n_common_periods=n_common,
    )


def align_panel(
    prices: pd.DataFrame,
    method: Literal["common", "ffill", "drop_assets"] = "common",
    max_ffill: int | None = 5,
    min_observations: int = 30,
) -> tuple[pd.DataFrame, list[str]]:
    """Resolve missing data explicitly, and say what was done.

    Silent alignment is how a panel ends up with each pairwise covariance
    estimated on a different sample. The caller gets back the cleaned panel
    *and* a log of every change, so the choice is visible in the report.

    Args:
        prices: Raw price panel.
        method:
            ``"common"`` — keep only dates where every asset is present.
            ``"ffill"`` — carry the last price forward across short gaps,
            then drop any remaining incomplete rows.
            ``"drop_assets"`` — drop assets that are too short, then take the
            common window of what remains.
        max_ffill: Maximum consecutive periods to carry forward. ``None``
            fills without limit, which fabricates prices — not recommended.
        min_observations: Threshold used by ``"drop_assets"``.

    Returns:
        ``(aligned_prices, actions)``.
    """
    actions: list[str] = []
    out = prices.sort_index()
    if out.index.duplicated().any():
        n = int(out.index.duplicated().sum())
        out = out[~out.index.duplicated(keep="last")]
        actions.append(f"Dropped {n} duplicate date(s), keeping the last.")

    if method == "drop_assets":
        counts = out.notna().sum()
        too_short = counts[counts < min_observations].index.tolist()
        if too_short:
            out = out.drop(columns=too_short)
            actions.append(
                f"Dropped {len(too_short)} asset(s) with fewer than "
                f"{min_observations} observations: {', '.join(map(str, too_short))}."
            )
        method = "common"

    if method == "ffill":
        before = int(out.isna().sum().sum())
        out = out.ffill(limit=max_ffill)
        after = int(out.isna().sum().sum())
        if before - after:
            actions.append(
                f"Forward-filled {before - after} missing price(s) across gaps "
                f"of at most {max_ffill} period(s)."
            )

    n_before = len(out)
    out = out.dropna(how="any")
    if len(out) < n_before:
        actions.append(
            f"Kept the {len(out)} date(s) where every asset is present "
            f"(dropped {n_before - len(out)})."
        )
    return out, actions
