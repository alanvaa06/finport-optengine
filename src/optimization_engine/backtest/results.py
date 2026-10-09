"""The canonical result bundle a simulation returns.

One run produces one :class:`RunResult`: a fixed set of frames — NAV, net and
gross returns, held weights, targets, trades, costs — plus a
:class:`RunMeta` recording what was asked for and what came back. Everything
downstream (transaction-cost analysis, position statistics, the tearsheet,
the sweep) reads this bundle and nothing else, which is what keeps those
layers from having to know how the loop works.

The result carries a ``result_hash`` alongside the spec's ``spec_hash``. The
pair answers two different questions: the spec hash says *what was asked*,
the result hash says *what came back*. Same spec and same data must give the
same result hash — if it does not, something in the pipeline is not
deterministic, and finding that out cheaply is worth the hash.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from optimization_engine.backtest.spec import BacktestSpec

#: Rounding applied before hashing, in **significant figures** — not decimal
#: places. Float arithmetic is deterministic given a fixed operation order, but
#: BLAS threading and platform libm are not guaranteed to agree in the last
#: bits, so the digest has to round before it reads. Twelve *absolute* decimals
#: does not do that job at the scales this library is actually run at: a NAV
#: path denominated in real money — ten million, say — has only about nine
#: fractional decimals of float64 precision left, so rounding to twelve sat
#: below the noise floor, rounded nothing, and let one ulp of BLAS disagreement
#: flip the hash. Twelve significant figures is the same tightness at every
#: scale — far finer than any decision taken on these numbers, far coarser than
#: the last three or four bits where platforms differ.
_HASH_SIG_FIGS = 12

#: Hash format version. Bumped when :func:`compute_result_hash` changes what it
#: reads or how it rounds, so two digests are only ever compared when they were
#: produced the same way. Version 2 rounds relatively (see ``_HASH_SIG_FIGS``)
#: and includes the realized weight path.
HASH_VERSION = 2


def _q(value: float) -> str:
    """One float, rounded to ``_HASH_SIG_FIGS`` significant figures, as text."""
    return f"{float(value):.{_HASH_SIG_FIGS}g}"


def _round_significant(values: np.ndarray, sig_figs: int = _HASH_SIG_FIGS) -> np.ndarray:
    """``values`` rounded to ``sig_figs`` significant figures, elementwise.

    The array form of :func:`_q`, and the reason the weight path can be hashed
    at all: formatting a 20-year, 50-asset weight frame cell by cell costs the
    better part of a second, and a parameter sweep pays that once per cell.
    Rounding the whole array and hashing its bytes costs microseconds.

    Args:
        values: Any float array. NaN and infinity pass through untouched.
        sig_figs: Significant figures to keep.

    Returns:
        A new float64 array. Negative zero is normalized to positive zero so
        two runs that agree numerically cannot disagree in the sign bit.
    """
    arr = np.asarray(values, dtype=float)
    out = np.array(arr, dtype=float, copy=True)
    scalable = np.isfinite(arr) & (arr != 0.0)
    if scalable.any():
        magnitude = np.floor(np.log10(np.abs(arr[scalable])))
        factor = np.power(10.0, sig_figs - 1 - magnitude)
        out[scalable] = np.round(arr[scalable] * factor) / factor
    out[out == 0.0] = 0.0
    out[np.isnan(out)] = np.nan
    return out


#: Columns of the trades frame, in canonical order.
TRADE_COLUMNS = (
    "date",
    "asset",
    "side",
    "traded_weight",
    "notional",
    "commission",
    "slippage",
    "cost",
)

#: Columns of the per-date costs frame, in canonical order.
COST_COLUMNS = ("date", "commission", "slippage", "total", "turnover")


@dataclass(frozen=True)
class RunMeta:
    """What the run was asked to do, and what it reported back.

    Attributes:
        spec: The spec, as a plain dict, so the meta serializes on its own.
        spec_hash: Provenance hash of the spec.
        result_hash: Provenance hash of the frames this run produced.
        n_periods: Evaluated periods.
        n_assets: Universe size.
        start: First evaluated date.
        end: Last evaluated date.
        is_out_of_sample: Copied off the spec so a bare result still carries
            the caveat.
        degradations: Reasons any cost component switched itself off. Empty
            is the normal case; non-empty means the reported cost is a lower
            bound on the modelled one.
        notes: Free-form annotations added by whatever produced the run —
            the walk-forward runner records its window geometry here, and the
            simulation loop records anything it had to move or drop. Notes are
            deliberately **not** hashed: they describe the run, they are not
            part of what it computed.
        hash_version: Which recipe produced ``result_hash``. Two digests are
            comparable only when this matches — version 2 rounds relatively
            and hashes the weight path, version 1 (anything written before
            this field existed) did neither. Defaults to the current version,
            which is the only one this release can produce.
    """

    spec: dict[str, Any]
    spec_hash: str
    result_hash: str
    n_periods: int
    n_assets: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    is_out_of_sample: bool
    degradations: tuple[str, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)
    hash_version: int = HASH_VERSION


@dataclass
class RunResult:
    """The frames one simulation produced.

    Attributes:
        returns: Portfolio returns net of costs — the headline series.
        gross_returns: The same series before costs.
        nav: Net asset value, starting at ``spec.initial_capital``.
        weights: Weights actually held at the start of each period, after
            drift. Not the targets: the difference between the two is the
            drift the book was carrying.
        targets: Target weights on each date one became effective.
        trades: One row per asset per traded date, with the cost split.
        costs: Per-date cost totals and turnover.
        rebalance_dates: Dates on which the book was traded back to target.
        meta: See :class:`RunMeta`.
    """

    returns: pd.Series
    gross_returns: pd.Series
    nav: pd.Series
    weights: pd.DataFrame
    targets: pd.DataFrame
    trades: pd.DataFrame
    costs: pd.DataFrame
    rebalance_dates: pd.DatetimeIndex
    meta: RunMeta

    # -- derived views ------------------------------------------------------

    @property
    def turnover(self) -> pd.Series:
        """Two-sided turnover, ``Σ|Δw|`` (buys plus sells), on each traded date."""
        if self.costs.empty:
            return pd.Series(dtype=float, name="turnover")
        return pd.Series(
            self.costs["turnover"].values,
            index=pd.DatetimeIndex(self.costs["date"]),
            name="turnover",
        )

    @property
    def cost_series(self) -> pd.Series:
        """Total cost charged on each traded date, as a fraction of NAV."""
        if self.costs.empty:
            return pd.Series(dtype=float, name="cost")
        return pd.Series(
            self.costs["total"].values,
            index=pd.DatetimeIndex(self.costs["date"]),
            name="cost",
        )

    @property
    def total_turnover(self) -> float:
        """Sum of two-sided turnover, ``Σ|Δw|``, over the run, as a fraction of NAV."""
        return float(self.costs["turnover"].sum()) if not self.costs.empty else 0.0

    @property
    def total_cost(self) -> float:
        """Every cost charged over the whole run, as a fraction of NAV."""
        return float(self.costs["total"].sum()) if not self.costs.empty else 0.0

    @property
    def periods_per_year(self) -> int:
        """Annualization basis the run was simulated on, read back from the spec."""
        return int(self.meta.spec.get("periods_per_year", 252))

    @property
    def annualized_turnover(self) -> float:
        """Two-sided turnover per year, ``Σ|Δw|``; one-way is half of it."""
        years = len(self.returns) / self.periods_per_year
        return float(self.total_turnover / years) if years > 0 else float("nan")

    def wealth(self, starting: float = 1.0) -> pd.Series:
        """The net-of-cost equity curve.

        Args:
            starting: Value at the first period. Defaults to ``1.0``, which makes
                the series read as a growth multiple.

        Returns:
            A series indexed like :attr:`returns`, compounding them from
            ``starting``.
        """
        return starting * (1 + self.returns).cumprod()

    def summary(
        self, periods_per_year: int | None = None, riskfree_rate: float = 0.0
    ) -> pd.DataFrame:
        """Standard performance summary of the net-of-cost return stream.

        Args:
            periods_per_year: Annualization basis. Defaults to the run's own.
            riskfree_rate: Annual risk-free rate for the ratio metrics, as a
                fraction; converted to a per-period rate internally.

        Returns:
            A one-row-per-statistic summary frame, computed on the extended set.
        """
        from optimization_engine.analytics.performance import summary_stats

        return summary_stats(
            self.returns.to_frame("portfolio"),
            periods_per_year=periods_per_year or self.periods_per_year,
            riskfree_rate=riskfree_rate,
            extended=True,
        )

    def cost_drag(self, periods_per_year: int | None = None) -> float:
        """Annualized return given up to transaction costs.

        Args:
            periods_per_year: Annualization basis. Defaults to the run's own.

        Returns:
            The gross annualized return minus the net one, as a fraction. This is
            what trading cost, expressed the way a return is.
        """
        from optimization_engine.analytics.performance import annualize_returns

        ppy = periods_per_year or self.periods_per_year
        return float(
            annualize_returns(self.gross_returns, ppy)
            - annualize_returns(self.returns, ppy)
        )

    def describe(self) -> str:
        """One line: sample, length, annualized turnover, total cost, degradations.

        Returns:
            A summary that names any cost degradation rather than letting a
            cheap-looking backtest hide a cost model that switched itself off.
        """
        sample = "out-of-sample" if self.meta.is_out_of_sample else "in-sample"
        degraded = (
            "" if not self.meta.degradations
            else f"; {len(self.meta.degradations)} cost degradation(s)"
        )
        return (
            f"{self.meta.spec.get('name', 'backtest')} - {len(self.returns)} periods "
            f"({sample}), turnover {self.annualized_turnover:.2f}x/yr, "
            f"cost {self.total_cost * 100:.2f}% of NAV{degraded}"
        )


def empty_trades() -> pd.DataFrame:
    """A trades frame with the canonical columns and no rows."""
    return pd.DataFrame({name: pd.Series(dtype=_trade_dtype(name)) for name in TRADE_COLUMNS})


def empty_costs() -> pd.DataFrame:
    """A costs frame with the canonical columns and no rows."""
    frame = pd.DataFrame({name: pd.Series(dtype=float) for name in COST_COLUMNS})
    frame["date"] = pd.Series(dtype="datetime64[ns]")
    return frame[list(COST_COLUMNS)]


def _trade_dtype(column: str) -> str:
    if column == "date":
        return "datetime64[ns]"
    if column in {"asset", "side"}:
        return "object"
    return "float64"


def compute_result_hash(
    nav: pd.Series,
    trades: pd.DataFrame,
    costs: pd.DataFrame,
    weights: pd.DataFrame | None = None,
) -> str:
    """A deterministic fingerprint of what the run produced.

    Hashes the NAV path, the trade and cost totals, and the realized weight
    path — the four things that would have to change for the run to mean
    something different. The weight path is in there because NAV and trades do
    not pin it down: two books can hold different assets, trade the same
    notional and print the same NAV, and without the weights those two runs
    share a digest.

    Values are rounded to twelve *significant figures* first (see
    ``_HASH_SIG_FIGS``) and the rows are consumed in their canonical order, so
    the hash is stable across platforms and across scales — a million-unit book
    and a one-unit book are rounded equally hard — without being so loose that
    a real change slips past it.

    Args:
        nav: The NAV path.
        trades: The trade frame, in canonical column order.
        costs: The cost frame, in canonical column order.
        weights: The realized weight path. Optional only so that a caller
            holding the three frames an older release produced can still hash
            them; every run this library builds passes it.

    Returns:
        A hex SHA-256 digest. See :data:`HASH_VERSION` for which recipe it is.
    """
    digest = hashlib.sha256()
    for date, value in nav.items():
        digest.update(f"{pd.Timestamp(date).isoformat()}|{_q(value)}|".encode())
    if not trades.empty:
        ordered = trades.sort_values(["date", "asset", "side"], kind="mergesort")
        for row in ordered.itertuples(index=False):
            digest.update(
                (
                    f"{pd.Timestamp(row.date).isoformat()}|{row.asset}|{row.side}|"
                    f"{_q(row.traded_weight)}|{_q(row.cost)}|"
                ).encode()
            )
    if not costs.empty:
        for row in costs.sort_values("date", kind="mergesort").itertuples(index=False):
            digest.update(
                (
                    f"{pd.Timestamp(row.date).isoformat()}|"
                    f"{_q(row.total)}|{_q(row.turnover)}|"
                ).encode()
            )
    if weights is not None and not weights.empty:
        # Labels first, values second, and the values as one block of bytes:
        # formatting a weight frame cell by cell is the difference between a
        # hash that costs microseconds and one that costs the better part of a
        # second, paid once per sweep cell.
        digest.update(b"|weights|")
        digest.update("|".join(str(column) for column in weights.columns).encode())
        digest.update(b"|")
        digest.update(
            "|".join(pd.Timestamp(date).isoformat() for date in weights.index).encode()
        )
        rounded = _round_significant(weights.to_numpy(dtype=float))
        # Little-endian explicitly: the digest must not depend on the host.
        digest.update(np.ascontiguousarray(rounded, dtype="<f8").tobytes())
    return digest.hexdigest()


def build_meta(
    spec: BacktestSpec,
    *,
    nav: pd.Series,
    trades: pd.DataFrame,
    costs: pd.DataFrame,
    weights: pd.DataFrame,
    degradations: tuple[str, ...] = (),
    notes: dict[str, Any] | None = None,
) -> RunMeta:
    """Assemble the metadata block, hashing the frames as it goes.

    Args:
        spec: The spec the run was simulated under.
        nav: The NAV path.
        trades: The trade frame.
        costs: The cost frame.
        weights: The realized weight path. Hashed, not just measured.
        degradations: Cost-model degradation notes collected during the run.
        notes: Anything else worth recording alongside the result.

    Returns:
        A :class:`RunMeta` carrying the serialized spec, the result hash and
        the degradations — everything needed to tell two runs apart.
    """
    index = pd.DatetimeIndex(weights.index)
    return RunMeta(
        spec=spec.to_dict(),
        spec_hash=spec.spec_hash,
        result_hash=compute_result_hash(nav, trades, costs, weights),
        n_periods=int(len(index)),
        n_assets=int(weights.shape[1]),
        start=index[0] if len(index) else None,
        end=index[-1] if len(index) else None,
        is_out_of_sample=bool(spec.is_out_of_sample),
        degradations=tuple(degradations),
        notes=dict(notes or {}),
    )


def sanitize_weights(weights: pd.Series, assets: list[str]) -> np.ndarray:
    """A weight vector aligned to ``assets``, with missing names at zero.

    Args:
        weights: The solved weights, possibly over a different universe.
        assets: The universe to align to, in order.

    Returns:
        A float array of ``len(assets)``. An asset the solve did not hold
        contributes zero rather than a NaN that would poison the NAV path.
    """
    return weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)


__all__ = [
    "COST_COLUMNS",
    "HASH_VERSION",
    "TRADE_COLUMNS",
    "RunMeta",
    "RunResult",
    "build_meta",
    "compute_result_hash",
    "empty_costs",
    "empty_trades",
    "sanitize_weights",
]
