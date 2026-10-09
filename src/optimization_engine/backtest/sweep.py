"""Parameter sweeps, and the discipline that makes them safe to run.

Running a grid is easy and reporting the best cell is fatal. The maximum of N
noisy estimates is a biased estimate of the best true value, and the bias
grows with N: a hundred strategies with a true Sharpe of zero produce a best
in-sample Sharpe near 2.6 standard errors above zero, purely by chance.
Nothing about that number looks wrong on its own.

So this module takes a structural stance rather than a documentary one:

* **The grid is evaluated whole.** :class:`SweepResults` cannot represent a
  partial grid — the invariant is checked on construction. A cell that fails
  to build or fails to solve becomes an error *row*, never a missing one.
* **Nothing here selects, ranks, or recommends.** There is no ``best()``,
  no ``top_n()``, no sorting by Sharpe. What the module does offer is the
  diagnostics that tell you what the search cost you: the deflated Sharpe of
  a cell given every trial you ran, and the probability of backtest
  overfitting across the grid.

The number of trials is the input those diagnostics need and the one nobody
records. Running the grid through here records it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Union

import pandas as pd

from optimization_engine.config import EngineConfig

#: Absolute ceiling on grid size. Not raisable through the public API: a grid
#: this large is a search, and a search needs the diagnostics below far more
#: than it needs another thousand cells.
HARD_CELL_CAP = 1000

ParamScalar = Union[int, float, str, bool, None]

#: Columns every results frame carries, before the parameter and metric columns.
STATUS_COLUMNS = ("cell_id", "status", "error")

#: Metrics computed per cell from its net return stream.
METRIC_COLUMNS = (
    "annual_return",
    "annual_volatility",
    "sharpe",
    "max_drawdown",
    "n_periods",
)


class SweepValidationError(ValueError):
    """A grid that cannot be expanded against the base configuration."""


@dataclass(frozen=True)
class SweepSpec:
    """A declarative grid: dotted config path -> values to try.

    Attributes:
        params: ``{"optimizer.name": ["mean_variance", "risk_parity"],
            "covariance_method": ["sample", "ledoit_wolf"]}`` and so on. Paths
            are validated against the base configuration before anything runs.
        max_cells: Refuse to expand past this. Defaults low on purpose.
    """

    params: dict[str, list[ParamScalar]]
    max_cells: int = 200

    def __post_init__(self) -> None:
        """Validate the grid before anything is run.

        Raises:
            SweepValidationError: If there are no parameters, if any parameter has
                an empty value list, if ``max_cells`` is not positive, or if it
                exceeds the hard cap — which is not raisable, because a grid that
                large is a selection-bias problem rather than a compute one.
        """
        if not self.params:
            raise SweepValidationError("A sweep needs at least one parameter.")
        for path, values in self.params.items():
            if not values:
                raise SweepValidationError(f"Parameter {path!r} has an empty value list.")
        if self.max_cells <= 0:
            raise SweepValidationError(f"max_cells must be positive; got {self.max_cells}.")
        if self.max_cells > HARD_CELL_CAP:
            raise SweepValidationError(
                f"max_cells {self.max_cells} exceeds the hard cap of {HARD_CELL_CAP}. "
                "Shrink the grid - the cap is not raisable."
            )

    def cell_count(self) -> int:
        """How many configurations the grid expands to.

        Returns:
            The product of the value-list lengths. Compare against
            :attr:`max_cells` before running: this is also the trial count that
            feeds the deflated Sharpe ratio.
        """
        count = 1
        for values in self.params.values():
            count *= len(values)
        return count


@dataclass(frozen=True)
class GridCell:
    """One cell: its overrides, and either the built config or the build error."""

    cell_id: int
    params: dict[str, ParamScalar]
    config: EngineConfig | None
    build_error: str | None


def _config_document(config: EngineConfig) -> dict[str, Any]:
    """The config as a plain dict, keeping optional optimizer fields visible.

    ``OptimizerSpec.to_dict`` drops ``None`` values, which would make a path
    like ``optimizer.target_volatility`` look nonexistent on a config that has
    not set it — exactly the parameter someone would want to sweep.
    """
    document = config.to_dict()
    document["optimizer"] = dict(config.optimizer.__dict__)
    return document


def _assert_path_exists(document: dict[str, Any], path: str) -> None:
    node: Any = document
    for segment in path.split("."):
        if not isinstance(node, dict) or segment not in node:
            raise SweepValidationError(
                f"Sweep path {path!r} does not exist on the base configuration "
                f"(failed at {segment!r}). Available top-level keys: "
                f"{sorted(document)}"
            )
        node = node[segment]


def _get_path(document: dict[str, Any], path: str) -> Any:
    node: Any = document
    for segment in path.split("."):
        node = node[segment]
    return node


def _set_path(document: dict[str, Any], path: str, value: ParamScalar) -> None:
    segments = path.split(".")
    node: Any = document
    for segment in segments[:-1]:
        node = node[segment]
    node[segments[-1]] = value


def expand_grid(base_config: EngineConfig, sweep: SweepSpec) -> list[GridCell]:
    """Validate the grid against the base config and expand it in full.

    Deterministic order: paths sorted lexicographically, values in the order
    given, the last path varying fastest. A cell whose overrides fail
    validation is still a cell — it carries its ``build_error`` forward.

    Args:
        base_config: The configuration the grid perturbs.
        sweep: The grid.

    Returns:
        One :class:`GridCell` per configuration, in the order above.

    Raises:
        SweepValidationError: If a path names nothing on the base
            configuration, or the grid expands past ``max_cells``.
    """
    document = _config_document(base_config)
    for path in sweep.params:
        _assert_path_exists(document, path)
    count = sweep.cell_count()
    if count > sweep.max_cells:
        raise SweepValidationError(
            f"The grid expands to {count} cells, over max_cells={sweep.max_cells}. "
            f"Drop values or raise max_cells (hard cap {HARD_CELL_CAP})."
        )

    paths = sorted(sweep.params)
    cells: list[GridCell] = []
    for cell_id, combination in enumerate(
        itertools.product(*[sweep.params[path] for path in paths])
    ):
        params = dict(zip(paths, combination))
        config: EngineConfig | None = None
        build_error: str | None = None
        try:
            cell_document = _config_document(base_config)
            for path, value in params.items():
                _set_path(cell_document, path, value)
            config = EngineConfig.from_dict(cell_document)
        except Exception as exc:  # noqa: BLE001 — an unbuildable cell is a row
            build_error = f"{type(exc).__name__}: {exc}"
        cells.append(
            GridCell(
                cell_id=cell_id, params=params, config=config, build_error=build_error
            )
        )
    return cells


def _cell_metrics(returns: pd.Series, periods_per_year: int) -> dict[str, float]:
    from optimization_engine.analytics.performance import (
        annualize_returns,
        annualize_volatility,
        sharpe_ratio,
    )
    from optimization_engine.analytics.risk import drawdown_series

    clean = returns.dropna()
    if clean.empty:
        return {name: float("nan") for name in METRIC_COLUMNS}
    return {
        "annual_return": float(annualize_returns(clean, periods_per_year)),
        "annual_volatility": float(annualize_volatility(clean, periods_per_year)),
        # Arithmetic, stated rather than defaulted: this column has to match
        # what ``SweepResults.trial_sharpes`` feeds the deflation, and the
        # deflation's standard error is derived for the arithmetic estimator.
        "sharpe": float(sharpe_ratio(clean, 0.0, periods_per_year, method="arithmetic")),
        "max_drawdown": float(drawdown_series(clean).min()),
        "n_periods": float(len(clean)),
    }


@dataclass
class SweepResults:
    """Every cell of an evaluated grid, with the diagnostics the search needs.

    The frame has one row per cell — always, including the ones that failed.
    Constructing it with fewer rows than cells raises, because a grid that
    quietly lost its failures is a grid whose trial count is wrong, and the
    trial count is precisely what the deflation below depends on.
    """

    frame: pd.DataFrame
    returns: dict[int, pd.Series]
    n_cells: int
    base_config: EngineConfig
    sweep: SweepSpec
    periods_per_year: int = 252
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Assert the full-grid invariant.

        Every cell is a row, including the ones that failed to build or solve.
        Dropping failures would silently shrink the trial count that the
        selection-bias diagnostics are computed from, which is the one number a
        sweep exists to report honestly.

        Raises:
            ValueError: If the frame has a different number of rows than cells.
        """
        if len(self.frame) != self.n_cells:
            raise ValueError(
                f"Full-grid invariant violated: {len(self.frame)} rows for "
                f"{self.n_cells} cells. Failed cells must be rows, never drops."
            )

    @property
    def n_ok(self) -> int:
        """How many cells evaluated successfully."""
        return int((self.frame["status"] == "ok").sum())

    @property
    def n_failed(self) -> int:
        """How many cells failed to build or solve.

        These still count as trials: they were configurations you tried.
        """
        return self.n_cells - self.n_ok

    def return_matrix(self) -> pd.DataFrame:
        """A ``T x N`` frame of the successful cells' return streams.

        Columns are named by cell id and ordered by it, and the rows are the
        dates every cell has in common — the input the CSCV procedure wants.
        """
        if not self.returns:
            return pd.DataFrame()
        ordered = [self.returns[cell_id] for cell_id in sorted(self.returns)]
        matrix = pd.concat(
            ordered, axis=1, join="inner",
            keys=[str(cell_id) for cell_id in sorted(self.returns)],
        )
        return matrix.dropna(how="any")

    def trial_sharpes(self, *, aligned: bool = True) -> pd.Series:
        """Annualized Sharpe of every successful cell.

        Args:
            aligned: Measure the Sharpes on the columns of
                :meth:`return_matrix` — the dates every surviving cell has in
                common, which is the block :meth:`overfitting_report` scores.
                Default, and what the deflation wants: the two diagnostics
                then describe one sample. When ``False``, read the
                pre-computed Sharpes off the results frame instead; those are
                measured over each cell's *own* full history, so a grid that
                varies an estimation window compares a Sharpe from 2,000
                observations against one from 1,800.

        Returns:
            A series named ``"sharpe"``, indexed by :meth:`return_matrix`'s
            columns when aligned — string cell ids — and by integer cell id
            otherwise. Empty when no cell produced a return stream. A cell
            whose stream is constant up to float rounding has no Sharpe and
            reads NaN; :meth:`deflated_sharpe` leaves it out of the dispersion,
            with a warning, and still counts it as a trial.
        """
        if not aligned:
            ok = self.frame[self.frame["status"] == "ok"]
            return pd.Series(
                ok["sharpe"].to_numpy(dtype=float),
                index=ok["cell_id"].to_numpy(),
                name="sharpe",
            )

        from optimization_engine.analytics.performance import sharpe_ratio

        matrix = self.return_matrix()
        if matrix.empty:
            return pd.Series(dtype=float, name="sharpe")
        sharpes = sharpe_ratio(
            matrix, 0.0, self.periods_per_year, method="arithmetic"
        )
        return pd.Series(sharpes, dtype=float, name="sharpe")

    def deflated_sharpe(self, cell_id: int):
        """Deflate one cell's Sharpe for the whole grid behind it.

        The grid *is* the trial count. Deflating against it is the difference
        between "this configuration had a Sharpe of 1.4" and "this
        configuration had a Sharpe of 1.4 after we tried forty-eight of them".

        Every cell counts, including the ones that failed to build or failed
        to solve — see :attr:`n_failed`. A configuration you tried and threw
        away is still a draw from the search, and deflating against
        :attr:`n_ok` would quietly reward a grid for breaking.

        The dispersion comes from :meth:`trial_sharpes`, which measures every
        surviving cell over the dates they all share — the same block
        :meth:`overfitting_report` scores. This cell's *own* Sharpe is still
        taken from its full stream, which is the number the results frame
        reports and the honest sample size for its standard error; on a grid
        whose cells span different windows the two are not measured over the
        same dates, and the deflation is the more conservative for it.

        Args:
            cell_id: Which cell to deflate. Must be one that evaluated.

        Returns:
            A :class:`~optimization_engine.analytics.selection.DeflatedSharpe`,
            with the trial count and the grid's own Sharpe distribution taken from
            these results rather than supplied by the caller.

        Raises:
            KeyError: If the cell has no return stream — it failed to build or to
                solve. The message carries its status.
            ValueError: If fewer than two cells produce a Sharpe over the dates
                they share — a one-cell grid, or cells whose streams barely
                overlap. There is no dispersion to deflate against.
        """
        from optimization_engine.analytics.selection import deflated_sharpe_ratio

        if cell_id not in self.returns:
            raise KeyError(
                f"Cell {cell_id} has no return stream "
                f"(status: {self._status(cell_id)!r})."
            )
        trials = self.trial_sharpes()
        n_usable = int(trials.notna().sum())
        if n_usable < 2:
            raise ValueError(
                "Deflation needs at least two trial Sharpes to have a "
                f"dispersion; the grid's {self.n_ok} evaluated cell(s) yield "
                f"{n_usable} over the {len(self.return_matrix())} date(s) they "
                "share. Widen the grid, align the cells' return streams, or "
                "deflate by hand with deflated_sharpe_ratio(..., "
                "sharpe_variance=...)."
            )
        return deflated_sharpe_ratio(
            self.returns[cell_id],
            n_trials=self.n_cells,
            trial_sharpes=trials,
            periods_per_year=self.periods_per_year,
        )

    def base_is_cell(self) -> bool:
        """Whether the base configuration is itself one of the grid's cells.

        The grid is the full product of the swept values over the base, so
        the base is a cell exactly when its own value at every swept path is
        among the values swept there.

        Returns:
            ``True`` when some cell carries the base configuration unchanged.
        """
        document = _config_document(self.base_config)
        return all(
            _get_path(document, path) in values
            for path, values in self.sweep.params.items()
        )

    def trials_with_base(self, base_returns: pd.Series) -> tuple[int, pd.Series]:
        """Trial count and trial Sharpes for deflating the *base* run.

        A caller that reports the base configuration's own run — the CLI's
        headline backtest — and sweeps around it has tried the base too.
        When the base is a cell it is already counted and measured. When it
        is not, deflating against :attr:`n_cells` undercounts by one and
        measures a dispersion the reported run is not part of: base
        ``risk_parity`` swept over ``min_variance`` and ``hrp`` is three
        trials, not two.

        Args:
            base_returns: The base configuration's return stream, evaluated
                the way the cells were.

        Returns:
            ``(n_trials, trial_sharpes)``. When the base is a cell, these are
            :attr:`n_cells` and :meth:`trial_sharpes` unchanged. When it is
            not, the count is one higher and the Sharpes gain a ``"base"``
            entry, every Sharpe measured over the dates the cells and the
            base all share.
        """
        if self.base_is_cell():
            return self.n_cells, self.trial_sharpes()

        from optimization_engine.analytics.performance import sharpe_ratio

        base = base_returns.rename("base")
        matrix = self.return_matrix()
        joint = (
            base.to_frame()
            if matrix.empty
            else pd.concat([matrix, base], axis=1, join="inner")
        ).dropna(how="any")
        if joint.empty:
            sharpes = pd.Series(dtype=float, name="sharpe")
        else:
            sharpes = pd.Series(
                sharpe_ratio(joint, 0.0, self.periods_per_year, method="arithmetic"),
                dtype=float,
                name="sharpe",
            )
        return self.n_cells + 1, sharpes

    def overfitting_report(self, n_partitions: int = 16):
        """CSCV across the grid: does the in-sample winner survive out of sample?

        Scored on :meth:`return_matrix` — the same block :meth:`trial_sharpes`
        measures, so this report and :meth:`deflated_sharpe` describe one
        sample rather than two.

        Args:
            n_partitions: How many balanced blocks to split the sample into.
                More partitions give more splits and a finer estimate, at the
                cost of shorter blocks.

        Returns:
            An :class:`~optimization_engine.analytics.selection.OverfittingReport`
            with the probability of backtest overfitting and the slope of
            out-of-sample performance on in-sample.

        Raises:
            ValueError: If fewer than two cells produced a return stream — there
                is no selection to diagnose.
        """
        from optimization_engine.analytics.selection import (
            probability_of_backtest_overfitting,
        )

        matrix = self.return_matrix()
        if matrix.shape[1] < 2:
            raise ValueError(
                "Backtest-overfitting analysis needs at least two successful "
                f"cells; the grid produced {matrix.shape[1]}."
            )
        return probability_of_backtest_overfitting(matrix, n_partitions=n_partitions)

    def _status(self, cell_id: int) -> str:
        row = self.frame[self.frame["cell_id"] == cell_id]
        return "unknown" if row.empty else str(row.iloc[0]["status"])

    def describe(self) -> str:
        """One line: the grid's size, its parameters, and how many cells survived.

        Returns:
            Something like ``"12 cells over ['optimizer.name']: 11 evaluated,
            1 failed."``
        """
        return (
            f"{self.n_cells} cells over {sorted(self.sweep.params)}: "
            f"{self.n_ok} evaluated, {self.n_failed} failed."
        )


def run_sweep(
    base_config: EngineConfig,
    sweep: SweepSpec,
    evaluate: Callable[[EngineConfig], pd.Series],
    *,
    periods_per_year: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> SweepResults:
    """Evaluate every cell of the grid and return all of them.

    Args:
        base_config: The configuration the grid perturbs.
        sweep: The grid.
        evaluate: Takes one cell's config and returns its net return stream.
            Normally a walk-forward run — evaluating cells in-sample makes
            the overfitting diagnostics measure the wrong thing.
        periods_per_year: Annualization basis. Defaults to the base config's.
        progress: Called as ``progress(done, total)`` after each cell.

    Returns:
        Every cell, evaluated or failed. See :class:`SweepResults`.

    Raises:
        SweepValidationError: If a grid path names nothing on the base
            configuration, or the grid expands past ``max_cells``.
        ValueError: If ``evaluate`` returns an empty stream for a cell.
    """
    cells = expand_grid(base_config, sweep)
    total = len(cells)
    ppy = int(periods_per_year or base_config.periods_per_year)

    rows: list[dict[str, Any]] = []
    streams: dict[int, pd.Series] = {}
    for done, cell in enumerate(cells, start=1):
        row: dict[str, Any] = {"cell_id": cell.cell_id, "status": "ok", "error": None}
        row.update(cell.params)
        if cell.config is None:
            row["status"] = "build_error"
            row["error"] = cell.build_error
            row.update({name: float("nan") for name in METRIC_COLUMNS})
        else:
            try:
                stream = evaluate(cell.config)
                if stream is None or len(stream) == 0:
                    raise ValueError("evaluate returned an empty return stream")
                streams[cell.cell_id] = stream.rename(str(cell.cell_id))
                row.update(_cell_metrics(stream, ppy))
            except Exception as exc:  # noqa: BLE001 — per-cell isolation
                row["status"] = "run_error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                row.update({name: float("nan") for name in METRIC_COLUMNS})
        rows.append(row)
        if progress is not None:
            progress(done, total)

    columns = (
        list(STATUS_COLUMNS) + sorted(sweep.params) + list(METRIC_COLUMNS)
    )
    frame = pd.DataFrame(rows, columns=columns)
    return SweepResults(
        frame=frame,
        returns=streams,
        n_cells=total,
        base_config=base_config,
        sweep=sweep,
        periods_per_year=ppy,
        metadata={"n_trials": total},
    )


def sweep_from_optimizers(
    names: list[str], extra: dict[str, list[ParamScalar]] | None = None
) -> SweepSpec:
    """The common case: try these optimizers, optionally crossed with more.

    Args:
        names: Optimizer names to sweep over, as ``optimizer.name`` values.
        extra: Further dotted config paths to cross with them, e.g.
            ``{"covariance_method": ["sample", "ledoit_wolf"]}``.

    Returns:
        A :class:`SweepSpec` over the product of everything given.

    Raises:
        SweepValidationError: If ``names`` is empty, or any value list is.
    """
    params: dict[str, list[ParamScalar]] = {"optimizer.name": list(names)}
    params.update(extra or {})
    return SweepSpec(params=params)


__all__ = [
    "HARD_CELL_CAP",
    "METRIC_COLUMNS",
    "STATUS_COLUMNS",
    "GridCell",
    "SweepResults",
    "SweepSpec",
    "SweepValidationError",
    "expand_grid",
    "run_sweep",
    "sweep_from_optimizers",
]
