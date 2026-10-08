"""How many periods make a year, read off the data rather than assumed.

Every annualized number in the engine — volatility, expected return, Sharpe,
tracking error — is a per-period figure scaled by ``periods_per_year``. The
config's default is 252, which is right for daily bars and wrong by a factor
of 21 in variance for monthly ones: a monthly panel annualized on 252 reports
a 6% volatility as 29% and an 8% return as 170%, and nothing downstream can
tell.

The dates already say which it is. :func:`resolve_periods_per_year` decides
the factor in one place, in this order: a value the config states, checked
against the dates; then the ingest interval the panel was fetched at, checked
the same way; then the spacing of the dates themselves; and only when none of
those can say, the config's default. A stated value or an interval that the
dates contradict is refused rather than overridden, because either way one of
the two inputs is not what its author thinks it is.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from optimization_engine.ingest.spec import INTERVALS


class FrequencyMismatchError(ValueError):
    """A stated ``periods_per_year`` or interval contradicts the data's dates."""


class _Band(NamedTuple):
    """One standard frequency: the spacing that identifies it, what it implies."""

    label: str
    min_days: float
    max_days: float
    periods_per_year: int
    #: Values a caller may state for data at this spacing. Daily bars are
    #: annualized on 252 trading days by convention, but 250, 260 and the
    #: calendar's 365 (seven-day markets) are all in use and all defensible.
    accepts: tuple[int, int]


#: Median spacing in calendar days for each frequency. The bands are wide
#: enough to absorb weekends, holidays and month lengths, and far enough
#: apart that a panel falls in at most one. Anything else — intraday bars,
#: an irregular series — infers nothing.
_BANDS: tuple[_Band, ...] = (
    _Band("daily", 0.5, 4.0, 252, (240, 366)),
    _Band("weekly", 5.0, 9.0, 52, (48, 53)),
    _Band("monthly", 26.0, 35.0, 12, (12, 12)),
    _Band("quarterly", 85.0, 96.0, 4, (4, 4)),
    _Band("annual", 350.0, 380.0, 1, (1, 1)),
)


def _median_spacing_days(index: pd.Index) -> float | None:
    """Median gap between consecutive distinct dates, in days."""
    if not isinstance(index, pd.DatetimeIndex):
        return None
    dates = index.dropna().unique().sort_values()
    if len(dates) < 3:
        return None
    # ``.values`` is UTC for an aware index and carries its own unit, so the
    # division is right at any resolution pandas happened to build.
    gaps = np.diff(dates.values) / np.timedelta64(1, "D")
    return float(np.median(gaps))


def _band_for(index: pd.Index) -> tuple[_Band | None, float | None]:
    spacing = _median_spacing_days(index)
    if spacing is None:
        return None, None
    for band in _BANDS:
        if band.min_days <= spacing <= band.max_days:
            return band, spacing
    return None, spacing


def infer_periods_per_year(index: pd.Index) -> int | None:
    """Periods per year implied by the median spacing of a date index.

    Args:
        index: The dates of a price or return panel.

    Returns:
        252, 52, 12, 4 or 1 for daily, weekly, monthly, quarterly or annual
        spacing; ``None`` when the index is not dates, has fewer than three of
        them, or is spaced like none of those.
    """
    band, _ = _band_for(index)
    return band.periods_per_year if band is not None else None


def resolve_periods_per_year(
    index: pd.Index,
    *,
    stated: int | None = None,
    interval: str | None = None,
    default: int = 252,
) -> tuple[int, str]:
    """Decide the annualization factor for a panel, and say how.

    Args:
        index: The panel's dates.
        stated: ``periods_per_year`` as the config states it, or ``None`` when
            the config does not set it — a default the author never wrote is
            not a statement about the data.
        interval: The ingest interval the panel was fetched at (a key of
            :data:`~optimization_engine.ingest.spec.INTERVALS`), or ``None``
            when it did not come through an ingest.
        default: The value to fall back on when nothing else can say.

    Returns:
        ``(periods_per_year, note)``. The note is one sentence naming where a
        value other than ``stated`` or ``default`` came from, and empty when
        the result is simply one of those two.

    Raises:
        FrequencyMismatchError: When ``stated`` or ``interval`` implies a
            frequency the dates' spacing contradicts.
    """
    band, spacing = _band_for(index)
    seen = (
        f"the dates are {band.label} (median spacing {spacing:.1f} days)"
        if band is not None
        else ""
    )

    if stated is not None:
        stated = int(stated)
        if band is not None and not band.accepts[0] <= stated <= band.accepts[1]:
            raise FrequencyMismatchError(
                f"The config sets periods_per_year: {stated}, but {seen}, which "
                f"annualize on {band.periods_per_year}. Every annualized "
                "volatility, return and Sharpe would be off by that ratio. "
                f"Set periods_per_year: {band.periods_per_year}, or remove it "
                "and let the dates decide."
            )
        return stated, ""

    if interval is not None:
        implied = INTERVALS[interval]
        if band is not None and not band.accepts[0] <= implied <= band.accepts[1]:
            raise FrequencyMismatchError(
                f"The ingest interval is {interval!r} ({implied} periods per "
                f"year), but {seen}. Pass the interval the data actually has, "
                f"or set periods_per_year: {band.periods_per_year} in the config."
            )
        note = (
            f"{implied} periods per year, from the {interval!r} ingest interval."
            if implied != default
            else ""
        )
        return implied, note

    if band is not None:
        note = (
            f"{band.periods_per_year} periods per year, because {seen}."
            if band.periods_per_year != default
            else ""
        )
        return band.periods_per_year, note
    return default, ""


__all__ = [
    "FrequencyMismatchError",
    "infer_periods_per_year",
    "resolve_periods_per_year",
]
