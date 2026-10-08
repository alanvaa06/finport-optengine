"""Benchmark selection: what the portfolio is measured — and optimized — against.

A benchmark enters the engine in two distinct places, and conflating them is
how a report ends up internally inconsistent:

* as a **return stream**, for relative performance (excess return, tracking
  error, beta, capture, alpha);
* as a **weight vector**, for active positions (active share, active-risk
  decomposition) and for the benchmark-relative constraints and objectives in
  :mod:`optimization_engine.optimizers`.

A :class:`BenchmarkSpec` describes the choice once, declaratively, so both
uses resolve from the same statement instead of from two widgets that can
drift apart. Not every kind supplies both: an external index is a return
stream with no weights in the investable universe, and asking it for an
active share raises rather than inventing one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

#: How the benchmark's returns are produced.
#:
#: ``none``            — no benchmark; relative analytics are unavailable.
#: ``equal_weight``    — 1/N over the investable universe.
#: ``single_asset``    — one asset of the universe carries the whole weight.
#: ``custom_weights``  — an explicit weight vector (a policy portfolio).
#: ``external``        — a return series from outside the universe (an index
#:                      or a peer fund), carried alongside the panel.
BenchmarkKind = Literal[
    "none", "equal_weight", "single_asset", "custom_weights", "external"
]

#: Whether the weight vector is restored every period or left to drift.
#:
#: ``periodic`` rebalances back to the stated weights at every observation —
#: the convention behind most published index returns and the one that makes
#: ``(returns · w).sum()`` correct. ``buy_and_hold`` holds the initial
#: allocation and lets the winners grow, which is what an untouched policy
#: portfolio actually did.
BenchmarkRebalance = Literal["periodic", "buy_and_hold"]

_KIND_LABELS: dict[str, str] = {
    "none": "No benchmark",
    "equal_weight": "Equal weight (1/N)",
    "single_asset": "Single asset",
    "custom_weights": "Custom weights",
    "external": "External series",
}


class BenchmarkError(ValueError):
    """The benchmark cannot be resolved from the data it was given."""


@dataclass
class BenchmarkSpec:
    """A declarative statement of what the portfolio is measured against.

    Attributes:
        kind: One of :data:`BenchmarkKind`.
        asset: Universe member carrying the whole weight when
            ``kind == "single_asset"``.
        weights: Explicit ``asset -> weight`` map when
            ``kind == "custom_weights"``. Assets outside the map get zero.
        series_name: Column of the external returns frame to use when
            ``kind == "external"``.
        label: Display name. Defaults to something descriptive of ``kind``.
        rebalance: See :data:`BenchmarkRebalance`.
        normalize: Rescale the weight vector to sum to one. Left on by
            default because a benchmark that is not fully invested compares
            the portfolio against a different amount of money.
    """

    kind: BenchmarkKind = "none"
    asset: str | None = None
    weights: dict[str, float] | None = None
    series_name: str | None = None
    label: str | None = None
    rebalance: BenchmarkRebalance = "periodic"
    normalize: bool = True

    def __post_init__(self) -> None:
        """Normalize the spec and reject one that names nothing resolvable.

        Raises:
            BenchmarkError: On an unknown ``kind``, or a rebalance rule that is
                neither ``"periodic"`` nor ``"buy_and_hold"``.
        """
        self.kind = str(self.kind or "none")
        if self.kind not in _KIND_LABELS:
            raise BenchmarkError(
                f"Unknown benchmark kind {self.kind!r}. "
                f"Choose one of: {', '.join(sorted(_KIND_LABELS))}."
            )
        if self.rebalance not in ("periodic", "buy_and_hold"):
            raise BenchmarkError(
                f"Unknown rebalance rule {self.rebalance!r}; expected "
                "'periodic' or 'buy_and_hold'."
            )
        if self.weights is not None:
            self.weights = {str(k): float(v) for k, v in self.weights.items()}

    # -- identity -----------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """Whether this spec names a benchmark at all."""
        return self.kind != "none"

    @property
    def has_weights(self) -> bool:
        """Whether the benchmark can be expressed as universe weights.

        False for an external series, which is a return stream and nothing
        more — active share and the active-risk decomposition need positions
        and are correctly unavailable for it.
        """
        return self.kind in ("equal_weight", "single_asset", "custom_weights")

    @property
    def display_label(self) -> str:
        """What to call this benchmark in a report, a chart or the UI.

        Returns:
            The explicit ``label`` when one was set; otherwise the asset name for
            a single-asset benchmark, the series name for an external one, and
            the kind's own label for the rest.
        """
        if self.label:
            return str(self.label)
        if self.kind == "single_asset" and self.asset:
            return str(self.asset)
        if self.kind == "external" and self.series_name:
            return str(self.series_name)
        return _KIND_LABELS[self.kind]

    # -- resolution ---------------------------------------------------------

    def weight_vector(self, assets: list[str]) -> pd.Series | None:
        """The benchmark's weights over ``assets``, or None when it has none.

        Args:
            assets: The universe to expand the spec over.

        Returns:
            One weight per asset, normalized when ``normalize`` is set, or
            ``None`` for an external-index benchmark.

        Raises:
            BenchmarkError: When the spec names an asset or weights that the
                universe does not contain, when the universe is empty, or when the
                weights sum to zero under ``normalize``. Silently zero-filling
                would make the benchmark quietly different from the one that was
                chosen.
        """
        if not self.has_weights:
            return None
        assets = [str(a) for a in assets]
        if not assets:
            raise BenchmarkError("Cannot build a benchmark over an empty universe.")

        if self.kind == "equal_weight":
            w = pd.Series(1.0 / len(assets), index=assets)
        elif self.kind == "single_asset":
            if self.asset is None:
                raise BenchmarkError(
                    "A single-asset benchmark needs an asset; none was set."
                )
            if str(self.asset) not in assets:
                raise BenchmarkError(
                    f"Benchmark asset {self.asset!r} is not in the universe "
                    f"({', '.join(assets[:8])}{' …' if len(assets) > 8 else ''})."
                )
            w = pd.Series(0.0, index=assets)
            w[str(self.asset)] = 1.0
        else:  # custom_weights
            if not self.weights:
                raise BenchmarkError(
                    "A custom-weight benchmark needs a weight vector; none was set."
                )
            unknown = [a for a in self.weights if a not in assets]
            if unknown:
                raise BenchmarkError(
                    f"Benchmark weights name {len(unknown)} asset(s) outside the "
                    f"universe: {', '.join(unknown[:8])}"
                    f"{' …' if len(unknown) > 8 else ''}."
                )
            w = pd.Series(self.weights).reindex(assets).fillna(0.0)

        if self.normalize:
            total = float(w.sum())
            if abs(total) < 1e-12:
                raise BenchmarkError(
                    "The benchmark weights sum to zero, so they cannot be "
                    "normalized into a portfolio. Set at least one non-zero weight."
                )
            w = w / total
        return w.astype(float)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The spec as a plain, YAML-serializable dict.

        Returns:
            One key per field, with ``weights`` as a plain mapping or ``None``.
        """
        return {
            "kind": self.kind,
            "asset": self.asset,
            "weights": dict(self.weights) if self.weights else None,
            "series_name": self.series_name,
            "label": self.label,
            "rebalance": self.rebalance,
            "normalize": self.normalize,
        }

    @classmethod
    def from_dict(cls, data: Any) -> BenchmarkSpec:
        """Build a spec from a mapping, a bare kind string, or ``None``.

        Args:
            data: A full mapping, the kind alone as a string (the common
                shorthand in a config), or ``None`` for no benchmark.

        Returns:
            The :class:`BenchmarkSpec`.

        Raises:
            BenchmarkError: If ``data`` is some other type, or the resulting spec
                names an unknown kind or rebalance rule.
        """
        if data is None:
            return cls()
        if isinstance(data, BenchmarkSpec):
            return data
        if isinstance(data, str):
            return cls(kind=data)  # type: ignore[arg-type]
        if not isinstance(data, dict):
            raise BenchmarkError(
                f"Cannot read a benchmark from {type(data).__name__}; expected "
                "a mapping, a kind string, or None."
            )
        raw_weights = data.get("weights")
        return cls(
            kind=str(data.get("kind", "none")),  # type: ignore[arg-type]
            asset=(str(data["asset"]) if data.get("asset") else None),
            weights=(
                {str(k): float(v) for k, v in raw_weights.items()}
                if raw_weights
                else None
            ),
            series_name=(
                str(data["series_name"]) if data.get("series_name") else None
            ),
            label=(str(data["label"]) if data.get("label") else None),
            rebalance=str(data.get("rebalance", "periodic")),  # type: ignore[arg-type]
            normalize=bool(data.get("normalize", True)),
        )


@dataclass(frozen=True)
class ResolvedBenchmark:
    """A benchmark bound to actual data: its returns, and its weights if any."""

    label: str
    returns: pd.Series
    weights: pd.Series | None
    spec: BenchmarkSpec

    @property
    def has_weights(self) -> bool:
        """Whether this benchmark holds positions in the investable universe.

        Returns:
            ``True`` for the weight-defined kinds. It is ``False`` for an external
            index, which is why a tracking-error or active-share budget cannot be
            measured against one: a series of returns has no holdings to be active
            against.
        """
        return self.weights is not None

    def summary(self) -> pd.DataFrame:
        """One-row description of the benchmark, for the report and the export."""
        row = {
            "label": self.label,
            "kind": self.spec.kind,
            "rebalance": self.spec.rebalance,
            "start": str(getattr(self.returns.index.min(), "date", lambda: "")() or ""),
            "end": str(getattr(self.returns.index.max(), "date", lambda: "")() or ""),
            "observations": int(self.returns.notna().sum()),
            "position_based": self.has_weights,
        }
        if self.weights is not None:
            held = self.weights[self.weights.abs() > 1e-12]
            row["holdings"] = int(len(held))
            row["top_holding"] = (
                str(held.abs().idxmax()) if len(held) else ""
            )
        return pd.DataFrame([row])

    def weights_frame(self) -> pd.DataFrame | None:
        """The benchmark's weight vector as a one-column frame, or None."""
        if self.weights is None:
            return None
        return self.weights.to_frame("benchmark_weight")


def portfolio_returns_from_weights(
    returns: pd.DataFrame,
    weights: pd.Series,
    rebalance: BenchmarkRebalance = "periodic",
) -> pd.Series:
    """Return stream of a fixed weight vector held over ``returns``.

    Args:
        returns: Periodic asset returns.
        weights: The allocation to hold, as fractions of the book.
        rebalance: ``"periodic"`` restores the weights every period — the
            assumption behind ``(returns · w).sum(axis=1)`` and behind most
            published index series. ``"buy_and_hold"`` invests once and lets
            the weights drift, which is a materially different track record
            over long samples: the winners compound into a larger share and
            the result is no longer the stated allocation.

    Returns:
        One return per row of ``returns``, NaN on every row where a member
        with non-zero weight has no return. A member that has not listed yet,
        or has delisted, earned nothing knowable — not 0%. Filling it with zero
        made a 50/50 benchmark exactly half of its live member before the other
        listed: half the volatility, and a spurious tracking error and alpha
        for any book that held what could be held. The relative analytics
        already handle a benchmark with gaps, as they must for an external
        series that does not cover the panel. Under ``"buy_and_hold"`` the
        book is bought on the first row where every member trades, rather
        than parking an unlisted member's share at zero return until it
        appears.
    """
    aligned = weights.reindex(returns.columns).fillna(0.0).astype(float)
    held = [c for c in returns.columns if aligned[c] != 0.0]
    undefined = returns[held].isna().any(axis=1)
    if rebalance == "periodic":
        return (returns.fillna(0.0) * aligned).sum(axis=1).mask(undefined)

    out = pd.Series(float("nan"), index=returns.index, dtype=float)
    defined = (~undefined).to_numpy()
    if not defined.any():
        return out
    first = int(defined.argmax())
    growth = (1.0 + returns.iloc[first:].fillna(0.0)).cumprod()
    nav = growth.mul(aligned, axis=1).sum(axis=1)
    previous = nav.shift(1)
    previous.iloc[0] = float(aligned.sum())
    out.iloc[first:] = (nav / previous - 1.0).to_numpy()
    out = out.replace([float("inf"), float("-inf")], float("nan"))
    return out.mask(undefined)


def resolve_benchmark(
    spec: BenchmarkSpec | dict | str | None,
    returns: pd.DataFrame,
    external_returns: pd.DataFrame | pd.Series | None = None,
) -> ResolvedBenchmark | None:
    """Bind a :class:`BenchmarkSpec` to data.

    Args:
        spec: The benchmark statement. ``None`` or ``kind="none"`` returns
            ``None``, so callers can write ``if bench is not None``.
        returns: The investable universe's periodic returns.
        external_returns: Returns of series outside the universe, needed only
            when ``spec.kind == "external"``. A Series is taken as the
            benchmark itself; a frame is indexed by ``spec.series_name``.

    Returns:
        A :class:`ResolvedBenchmark`, or ``None`` when no benchmark is set.

    Raises:
        BenchmarkError: When the spec cannot be satisfied by the data —
            an unknown asset, a missing external series, or an overlap of
            zero dates between the benchmark and the panel.
    """
    spec = BenchmarkSpec.from_dict(spec)
    if not spec.is_active:
        return None
    if returns is None or returns.empty:
        raise BenchmarkError("Cannot resolve a benchmark against an empty panel.")

    if spec.kind == "external":
        series = _external_series(spec, external_returns)
        common = series.dropna().index.intersection(returns.index)
        if len(common) == 0:
            raise BenchmarkError(
                f"The external benchmark {spec.display_label!r} shares no dates "
                "with the panel, so no relative metric could be computed. Check "
                "that both cover the same period and frequency."
            )
        return ResolvedBenchmark(
            label=spec.display_label,
            returns=series.reindex(returns.index).astype(float),
            weights=None,
            spec=spec,
        )

    weights = spec.weight_vector(list(returns.columns))
    assert weights is not None  # has_weights is true for every remaining kind
    stream = portfolio_returns_from_weights(returns, weights, spec.rebalance)
    stream.name = spec.display_label
    return ResolvedBenchmark(
        label=spec.display_label, returns=stream, weights=weights, spec=spec
    )


def _external_series(
    spec: BenchmarkSpec, external: pd.DataFrame | pd.Series | None
) -> pd.Series:
    if external is None:
        raise BenchmarkError(
            "An external benchmark was selected but no external return series "
            "was supplied. Load the index alongside the universe first."
        )
    if isinstance(external, pd.Series):
        return external.astype(float)
    if external.shape[1] == 0:
        raise BenchmarkError("The external returns frame has no columns.")
    name = spec.series_name
    if name is None:
        if external.shape[1] > 1:
            raise BenchmarkError(
                "The external returns frame holds several series "
                f"({', '.join(map(str, external.columns[:5]))}); name the one "
                "to use with series_name."
            )
        return external.iloc[:, 0].astype(float)
    if name not in external.columns:
        raise BenchmarkError(
            f"External benchmark {name!r} is not among the loaded series "
            f"({', '.join(map(str, external.columns[:8]))})."
        )
    return external[name].astype(float)


__all__ = [
    "BenchmarkError",
    "BenchmarkKind",
    "BenchmarkRebalance",
    "BenchmarkSpec",
    "ResolvedBenchmark",
    "portfolio_returns_from_weights",
    "resolve_benchmark",
]
