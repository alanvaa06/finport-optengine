"""An MCP server exposing the engine as tools.

Run it with ``optengine-mcp``, or point an MCP client at
``python -m optimization_engine.mcp_server``. It speaks stdio, which is what
desktop clients launch.

Every tool returns the same payloads the CLI's ``--json`` emits, from
:mod:`optimization_engine.reporting.payloads`. That is the point of having
written the contract as a module: this server is a transport, not a second
serialisation layer that could drift from the first.

**What this server can reach.** It runs as a local process with the
permissions of whoever launched it, and two tools take a filesystem path —
``config_path`` for a mandate and ``prices_path`` for a price panel. The
caller is a model, and a model can be steered by text it read somewhere
else, so those paths are confined:

- only under the *allowed roots* — ``--root DIR`` (repeatable) on the
  command line, else ``OPTENGINE_MCP_ROOTS`` (directories separated by
  ``os.pathsep``), else the working directory the server started in. A
  server started from a filesystem root gets no implicit root at all, since
  that default would be the whole disk. A relative path resolves against the
  first root, and a symlink or ``..`` that leads outside is refused;
- never a network or device path (``\\\\host\\share``, ``//host/share``,
  ``\\\\?\\…``) — on Windows merely looking one up authenticates to that host
  over SMB with the user's credentials;
- only with the extensions a mandate (``.yaml``, ``.yml``, ``.json``) or a
  price panel (``.csv``, ``.xlsx``, ``.xls``, ``.xlsm``, ``.parquet``) has,
  and under :data:`MAX_CONFIG_BYTES` / :data:`MAX_PRICES_BYTES` — both checked
  before a byte is read.

A file that does not parse is reported by path and kind of problem — the
exception type, and a line and column where the parser gives one — never by
the parser's message, which quotes the file: a YAML snippet, a value that
would not convert, the first cell of a CSV. Nothing here writes to disk,
fetches over the network, or reaches a data provider that needs a key; a
caller that wants live data should ingest it separately and hand over a
file. Start from ``sample=True``, which needs no paths at all.

**Solving blocks.** A large mean-variance solve is seconds of CPU, and these
tools are synchronous, so a client waiting on one waits for the whole thing.
That is the honest behaviour for an optimizer; it is not a hung server. What
*is* bounded is how much work one call may ask for: a panel of at most
:data:`MAX_ASSETS` assets and :data:`MAX_ROWS` rows, and a backtest of at most
:data:`MAX_RESOLVES` re-solves. The CLI has none of these limits.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, TypeVar

import pandas as pd
import yaml

from optimization_engine._optional import require
from optimization_engine.reporting.payloads import (
    SCHEMA_VERSION,
    backtest_payload,
    check_payload,
    describe_payload,
    optimization_payload,
)

INSTRUCTIONS = """\
A multi-asset portfolio optimization engine.

Call `describe_optimizer` before building a config: a method that does not
support a constraint ignores it silently rather than rejecting it, so the
support flags are the only way to know a turnover budget or a benchmark
limit will actually bind.

Then `check_mandate` before `optimize`. It reports whether the constraint
set is satisfiable at all and what expected-return range is reachable,
which turns an infeasible solve from an error into a number you can act on.

Read the diagnostics, not just the weights. `effective_n` counts positions
by capital and `effective_n_risk` counts them by risk contribution; a book
that is wide by capital and narrow by risk looks diversified in a weights
table and is not.

Data: pass `sample=True` for a built-in synthetic panel, or `prices_path`
for a CSV, Excel or Parquet file of prices (not returns - the engine
differences them). Paths must lie under the directories this server was
started with - its working directory unless it was told otherwise.
"""


def _build() -> tuple[Any, type[Exception]]:
    """Build the server, or explain what is missing.

    Returns the server and the SDK's ``ToolError``. That second value
    matters more than it looks: it is the only exception class whose
    message reaches the client. Anything else is wrapped in
    ``UnexpectedToolError`` and the caller sees "Error executing tool
    optimize" with the actual reason discarded — which for an agent means
    a failure it cannot act on or explain.

    The ``mcp`` SDK is an extra and needs Python 3.10+, one minor version
    above this package's own floor. Someone on 3.9 gets a resolver error
    from pip rather than anything from here, which is why the message names
    the version too.
    """
    mcpserver = require(
        "mcp.server.mcpserver",
        extra="mcp",
        purpose="running the MCP server (needs Python 3.10 or newer)",
    )
    exceptions = require(
        "mcp.server.mcpserver.exceptions",
        extra="mcp",
        purpose="running the MCP server",
    )
    from optimization_engine import __version__

    server = mcpserver.MCPServer(
        name="optimization-engine",
        title="Portfolio Optimization Engine",
        version=__version__,
        instructions=INSTRUCTIONS,
    )
    return server, exceptions.ToolError


mcp, ToolError = _build()

#: pydantic ships with the SDK. Taken through ``require`` rather than imported
#: at the top, so a core install still gets the install command from
#: ``_build`` instead of a bare ImportError naming a package it never asked for.
Field = require("pydantic", extra="mcp", purpose="running the MCP server").Field

#: Environment variable naming the directories the path-taking tools may read,
#: separated by ``os.pathsep``. ``--root`` on the command line takes precedence.
ROOTS_ENV = "OPTENGINE_MCP_ROOTS"

#: Largest mandate file read, in bytes. A real one is a few kilobytes.
MAX_CONFIG_BYTES = 1024 * 1024
#: Largest price file read, in bytes — twenty years of daily closes for a few
#: hundred assets fits with room to spare.
MAX_PRICES_BYTES = 64 * 1024 * 1024
#: Widest panel one call may solve on.
MAX_ASSETS = 200
#: Longest panel one call may solve on — about forty years of daily bars.
MAX_ROWS = 10_000
#: Most re-solves one ``backtest`` call may ask for. A re-solve on every bar of
#: the sample panel is ~1,500 solves, and the server answers nothing else while
#: it runs; 250 is a weekly cadence over five years of daily data.
MAX_RESOLVES = 250

CONFIG_SUFFIXES = (".yaml", ".yml", ".json")
PRICE_SUFFIXES = (".csv", ".xlsx", ".xls", ".xlsm", ".parquet")

#: Set by ``--root`` in :func:`main`; ``None`` defers to the environment.
_command_line_roots: tuple[Path, ...] | None = None


def allowed_roots() -> tuple[Path, ...]:
    """The directories the path-taking tools may read, resolved.

    ``--root`` first, then :data:`ROOTS_ENV`, then the working directory —
    unless the working directory is a filesystem root, where the default would
    be the whole disk and there is none.

    Returns:
        The roots, possibly empty. Empty means no path is readable until the
        server is told where.
    """
    if _command_line_roots is not None:
        return _command_line_roots
    configured = os.environ.get(ROOTS_ENV, "")
    if configured.strip():
        return tuple(
            Path(entry).expanduser().resolve()
            for entry in configured.split(os.pathsep)
            if entry.strip()
        )
    cwd = Path.cwd().resolve()
    return () if cwd == Path(cwd.anchor) else (cwd,)


def _is_within(path: Path, root: Path) -> bool:
    # normcase so that C:\Data and c:\data are one directory on Windows, and
    # stay two on a case-sensitive filesystem.
    return Path(os.path.normcase(path)).is_relative_to(Path(os.path.normcase(root)))


def _readable(raw: str, *, what: str, suffixes: tuple[str, ...], limit: int) -> Path:
    """Resolve a caller's path, or refuse it before anything is read.

    The checks run cheapest and least revealing first: the shape of the string,
    then its extension, then where it resolves, and only then the filesystem.

    Args:
        raw: The path as the caller sent it.
        what: ``"config"`` or ``"price"``, for the messages.
        suffixes: The extensions a file of this kind may have.
        limit: The largest size accepted, in bytes.

    Returns:
        The resolved path, inside an allowed root, of a regular file within
        ``limit``.

    Raises:
        ToolError: Naming the path and which check it failed.
    """
    if raw.startswith(("\\\\", "//")):
        raise ToolError(
            f"{raw} is a network or device path; this server reads local files only."
        )
    suffix = Path(raw).suffix.lower()
    if suffix not in suffixes:
        raise ToolError(
            f"{raw}: a {what} file must end in {', '.join(suffixes)}; "
            f"got {suffix or 'no extension'}."
        )
    roots = allowed_roots()
    if not roots:
        raise ToolError(
            "This server was started from a filesystem root, so it reads no files "
            f"until told where: start it with --root DIR, or set {ROOTS_ENV}."
        )
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = roots[0] / candidate
    resolved = candidate.resolve()
    if not any(_is_within(resolved, root) for root in roots):
        raise ToolError(
            f"{raw} is outside the directories this server may read "
            f"({', '.join(str(r) for r in roots)}). Start it with --root DIR, or "
            f"set {ROOTS_ENV}, to allow another."
        )
    if not resolved.exists():
        raise ToolError(f"No such {what} file: {raw}")
    if not resolved.is_file():
        raise ToolError(f"{raw} is not a regular file.")
    size = resolved.stat().st_size
    if size > limit:
        raise ToolError(
            f"{raw} is {size:,} bytes; this server reads {what} files of up to "
            f"{limit:,} bytes."
        )
    return resolved


def _withheld(path: str, what: str, exc: BaseException) -> str:
    """Say a file could not be used, without quoting what is in it.

    A parser's message quotes its input — a YAML snippet, a value float() would
    not take, the first cell of a CSV — and a path-taking tool can be pointed at
    any file under its roots. The type and, where the parser gives one, the
    position are enough to find the problem in a file you own.
    """
    where = ""
    mark = getattr(exc, "problem_mark", None)
    if isinstance(exc, yaml.MarkedYAMLError) and mark is not None:
        where = f" at line {mark.line + 1}, column {mark.column + 1}"
    elif isinstance(exc, json.JSONDecodeError):
        where = f" at line {exc.lineno}, column {exc.colno}"
    return (
        f"Could not read {path} as {what}: {type(exc).__name__}{where}. The "
        "parser's message is withheld because it can quote the file; run "
        "`optengine check` on it locally to see it."
    )


def _check_size(prices: pd.DataFrame) -> None:
    """Refuse a panel larger than one call should solve on."""
    rows, assets = prices.shape
    if assets > MAX_ASSETS or rows > MAX_ROWS:
        raise ToolError(
            f"The panel has {assets} assets and {rows} rows; this server solves "
            f"on at most {MAX_ASSETS} assets and {MAX_ROWS} rows per call. Use "
            "the optengine CLI, which has no such limit, for a larger one."
        )


_Tool = TypeVar("_Tool", bound=Callable[..., Any])


def _reasons_reach_the_client(tool: _Tool) -> _Tool:
    """Re-raise the engine's own refusals as ``ToolError``, reason intact.

    Anything else that escapes a tool is wrapped by the SDK as "Error executing
    tool X" with the reason discarded. The engine's refusals are ``ValueError``
    and ``RuntimeError`` subclasses — a shock outside the panel, an invalid
    backtest spec, a singular covariance — plus the ``KeyError`` for an
    unknown name, and all of them say what to change. By the time one is raised
    the mandate and the panel have been read and validated, so the message
    describes them rather than the bytes of an arbitrary file; read failures
    are reported, withheld, before this point.
    """

    @functools.wraps(tool)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return tool(*args, **kwargs)
        except ToolError:
            raise
        except KeyError as exc:
            reason = exc.args[0] if exc.args else exc
            raise ToolError(f"KeyError: {reason}") from exc
        except (ValueError, RuntimeError) as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper  # type: ignore[return-value]


def _panel(
    sample: bool,
    prices_path: str | None,
    config: Any = None,
    config_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], Any]:
    """Resolve a price panel, its returns, how they were aligned, and its quality.

    Prices and returns are both returned because the two are used for
    different things — data-quality analysis reads prices, the optimizers
    read returns — and reconstructing one from the other would introduce a
    rounding difference between what was checked and what was solved.

    The third element is the alignment log. A panel is made rectangular
    before it is differenced, and the asset that listed last decides where
    every other series starts; a client that only ever sees weights has no
    way to find that out. It travels in the payload rather than on a
    stream: this server speaks the protocol over stdio, so there is no
    stdout to narrate on.

    Every entry point here takes *prices*, because that is what a data file
    holds and differencing them in one place keeps the two from being
    confused. Handing this a returns file silently builds a portfolio on
    second differences, so the parameter is named for what it wants.

    Given the mandate, it also settles ``config.periods_per_year`` from the
    dates, as the CLI does — see
    :func:`~optimization_engine.data.frequency.resolve_periods_per_year` —
    so a monthly file is not annualized on the default 252, and reads the
    data-quality report off the panel *before* alignment, which would
    otherwise remove the gaps the report exists to name. The fourth element
    is that report, or ``None`` when no mandate was given.
    """
    from optimization_engine.data.loader import load_prices, prices_to_returns, sample_dataset
    from optimization_engine.data.quality import align_panel, analyze_prices

    if sample and prices_path:
        raise ToolError(
            "Pass either sample=True or prices_path, not both - otherwise it "
            "is ambiguous which panel the result describes."
        )
    if sample:
        prices = sample_dataset()
    elif prices_path:
        path = _readable(
            prices_path, what="price", suffixes=PRICE_SUFFIXES, limit=MAX_PRICES_BYTES
        )
        try:
            prices = load_prices(path)
        except Exception as exc:  # noqa: BLE001 — every reader failure, reported without its text
            raise ToolError(_withheld(prices_path, "a price panel", exc)) from exc
        text_columns = [
            str(position)
            for position, column in enumerate(prices.columns, start=1)
            if not pd.api.types.is_numeric_dtype(prices[column])
        ]
        if text_columns:
            # By position, not by name or value: the header and the cells of a
            # file that is not a price panel are exactly what must not echo.
            raise ToolError(
                f"{prices_path}: column(s) {', '.join(text_columns)} after the "
                "date column are not numeric; a price panel holds numbers only."
            )
    else:
        raise ToolError(
            "No data given. Pass sample=True for the built-in panel, or "
            "prices_path pointing at a CSV, Excel or Parquet file of prices."
        )
    _check_size(prices)
    quality = None
    if config is not None:
        _annualize(config, config_path, prices.index)
        quality = analyze_prices(prices, periods_per_year=config.periods_per_year)
    # `method="common"` is the CLI's choice, for the CLI's reasons — and
    # the two surfaces must not disagree about what the same file means.
    # See `cli._prepare_inputs`, including the note on why an interior gap
    # is treated differently from the bare `dropna(how="any")` this
    # replaced.
    aligned, actions = align_panel(prices, method="common")
    returns = prices_to_returns(aligned)
    n_rows = len(returns)
    returns = returns.dropna(how="any")
    if len(returns) < n_rows:
        actions.append(
            f"Dropped {n_rows - len(returns)} period(s) whose return could "
            "not be computed from the aligned prices."
        )
    return aligned, returns, actions, quality


def _source(sample: bool, prices_path: str | None) -> dict[str, Any]:
    """The panel a result was computed on, in the CLI's ``data_source`` shape.

    ``_panel`` has already refused anything but exactly one of the two, so
    this only describes; the path is the one the client named.
    """
    return {
        "kind": "sample" if sample else "file",
        "synthetic": bool(sample),
        "path": None if sample else prices_path,
        "provider": None,
        "identifiers": None,
    }


def _annualize(config: Any, config_path: str | None, index: pd.Index) -> None:
    """Set the annualization factor from the dates, refusing a contradiction."""
    from optimization_engine.config import stated_keys
    from optimization_engine.data.frequency import (
        FrequencyMismatchError,
        resolve_periods_per_year,
    )

    stated = None
    if config_path:
        # The same confined, already-validated file ``_config`` read — never
        # the raw argument, which may be relative to a root, not to the cwd.
        path = _readable(
            config_path, what="config", suffixes=CONFIG_SUFFIXES, limit=MAX_CONFIG_BYTES
        )
        if "periods_per_year" in stated_keys(path):
            stated = config.periods_per_year
    try:
        config.periods_per_year, _ = resolve_periods_per_year(
            index, stated=stated, default=config.periods_per_year
        )
    except FrequencyMismatchError as exc:
        raise ToolError(str(exc)) from exc


def _config(config_path: str | None, optimizer: str | None) -> Any:
    """Load a mandate, or build the smallest one that will solve.

    The method name is checked here, for all three tools at once: unchecked,
    an unknown name surfaced as a ``KeyError`` deep in the solve, which the
    SDK reported as "Error executing tool optimize" and nothing more.
    """
    from optimization_engine.config import EngineConfig, OptimizerSpec, load_config
    from optimization_engine.optimizers.factory import available_optimizers

    if optimizer and optimizer not in available_optimizers():
        raise ToolError(
            f"No optimizer named {optimizer!r}. Available: "
            + ", ".join(available_optimizers())
        )
    if config_path:
        path = _readable(
            config_path, what="config", suffixes=CONFIG_SUFFIXES, limit=MAX_CONFIG_BYTES
        )
        try:
            config = load_config(path)
        except Exception as exc:  # noqa: BLE001 — every parse failure, reported without its text
            raise ToolError(_withheld(config_path, "a mandate", exc)) from exc
        if optimizer:
            # Swap the method, keep the rest: the risk-free rate, the return
            # target, the risk budget and the views are part of the mandate.
            # Replacing the whole spec silently solved max-Sharpe against a
            # cash rate of zero on a config that said 4%.
            config.optimizer = replace(config.optimizer, name=optimizer)
        elif config.optimizer.name not in available_optimizers():
            # The name came from the file, so it is not repeated back.
            raise ToolError(
                f"{config_path} names an optimizer this engine does not have. "
                "Available: " + ", ".join(available_optimizers())
            )
        return config
    return EngineConfig(optimizer=OptimizerSpec(name=optimizer or "risk_parity"))


@mcp.tool(
    title="List optimizers",
    description=(
        "Every optimization method this engine can run, with a one-line "
        "summary of each. Start here when you do not know which method a "
        "mandate needs."
    ),
)
def list_optimizers() -> dict[str, Any]:
    """Enumerate the available methods."""
    from optimization_engine.optimizers.factory import available_optimizers
    from optimization_engine.optimizers.requirements import requirements_for

    entries = []
    for name in available_optimizers():
        req = requirements_for(name)
        entries.append({"name": name, "label": req.label, "summary": req.summary})
    return {
        "schema_version": SCHEMA_VERSION,
        "optimizers": entries,
    }


@mcp.tool(
    title="Describe an optimizer",
    description=(
        "What one method needs as input and which constraints it will "
        "honour. Read this before building a config: a constraint a method "
        "does not support is ignored silently, not rejected."
    ),
)
def describe_optimizer(name: str) -> dict[str, Any]:
    """Report one optimizer's contract.

    Args:
        name: An optimizer name, e.g. ``"risk_parity"``. Use
            ``list_optimizers`` if you are unsure.

    Returns:
        What the method requires, what it supports, and what it assumes.

    Raises:
        ToolError: If no optimizer carries that name. The message lists the
            available ones.
    """
    from optimization_engine.optimizers.factory import available_optimizers
    from optimization_engine.optimizers.requirements import requirements_for

    try:
        req = requirements_for(name)
    except KeyError as exc:
        # Naming the alternatives turns a dead end into the next call.
        raise ToolError(
            f"No optimizer named {name!r}. Available: "
            + ", ".join(available_optimizers())
        ) from exc
    return describe_payload(req)


@mcp.tool(
    title="Check a mandate before solving",
    description=(
        "Pre-flight the data and the constraints. Reports whether the "
        "constraint set can be satisfied at all, what expected-return range "
        "is reachable, and whether the covariance estimate is conditioned "
        "well enough to be worth optimizing on."
    ),
)
@_reasons_reach_the_client
def check_mandate(
    config_path: str | None = None,
    sample: bool = False,
    prices_path: str | None = None,
    optimizer: str | None = None,
) -> dict[str, Any]:
    """Say whether this mandate is solvable, before spending a solve on it.

    Args:
        config_path: Path to a YAML mandate. Omit for a minimal default.
        sample: Use the built-in synthetic price panel.
        prices_path: A CSV, Excel or Parquet file of prices.
        optimizer: Override the config's method.

    Returns:
        A payload whose ``ready`` field is the single boolean to branch on.
        ``alignment`` says what the panel lost to become rectangular — an
        empty list means nothing was dropped.
    """
    from optimization_engine.data.covariance import (
        covariance_diagnostics,
        covariance_from_config,
    )
    from optimization_engine.engine import resolve_expected_returns
    from optimization_engine.optimizers.factory import (
        constraints_from_config,
        effective_expected_returns,
    )
    from optimization_engine.optimizers.feasibility import analyze_feasibility

    config = _config(config_path, optimizer)
    _, returns, alignment, quality = _panel(sample, prices_path, config, config_path)

    cov = covariance_from_config(returns, config)
    diagnostics = covariance_diagnostics(
        cov, len(returns), config.covariance_method, config.ewma_lambda
    )
    assets = list(returns.columns)
    # Must be the vector `optimize` will use, or this tool validates a
    # different mandate from the one that gets solved.
    mu = resolve_expected_returns(config, returns, cov)
    feasibility = analyze_feasibility(
        assets,
        constraints_from_config(config, assets),
        expected_returns=effective_expected_returns(config, cov, mu),
        cov_matrix=cov,
    )
    return check_payload(
        quality,
        feasibility,
        diagnostics,
        alignment=alignment,
        data_source=_source(sample, prices_path),
    )


@mcp.tool(
    title="Optimize a portfolio",
    description=(
        "Solve for weights and report what they rest on: which solver "
        "answered, whether the constraints held, how well-conditioned the "
        "covariance estimate was, and how concentrated the book is in risk "
        "rather than capital. Blocks while solving."
    ),
)
@_reasons_reach_the_client
def optimize(
    config_path: str | None = None,
    sample: bool = False,
    prices_path: str | None = None,
    optimizer: str | None = None,
) -> dict[str, Any]:
    """Run one optimization end to end.

    Args:
        config_path: Path to a YAML mandate. Omit for a minimal default.
        sample: Use the built-in synthetic price panel.
        prices_path: A CSV, Excel or Parquet file of prices.
        optimizer: Override the config's method — see ``list_optimizers``.

    Returns:
        Weights under ``weights``, and the evidence under ``diagnostics``,
        ``covariance`` and ``feasibility``. Reporting the weights alone
        discards what distinguishes this engine. ``alignment`` names every
        change made to the panel before it was differenced.

    Raises:
        ToolError: If the mandate has no solution, or no solver could
            produce one. The message carries the feasibility report naming
            the binding constraint.
    """
    from optimization_engine.engine import run_engine
    from optimization_engine.optimizers._cvxpy_helpers import SolverFailure
    from optimization_engine.optimizers.feasibility import InfeasibleConstraintsError

    config = _config(config_path, optimizer)
    _, returns, alignment, quality = _panel(sample, prices_path, config, config_path)
    try:
        run = run_engine(returns, config, raise_on_infeasible=True)
    except InfeasibleConstraintsError as exc:
        # Anticipated, and the most useful failure this tool has: the
        # message names which constraints cannot hold together. Call
        # `check_mandate` to get the same finding as data rather than text.
        # ``raise_on_infeasible`` is what makes this branch reachable — the
        # engine's default is to let the solver fail instead.
        raise ToolError(f"The mandate has no solution: {exc}") from exc
    except SolverFailure as exc:
        raise ToolError(f"No solver could produce an allocation: {exc}") from exc
    return optimization_payload(
        run,
        alignment=alignment,
        quality=quality,
        data_source=_source(sample, prices_path),
    )


@mcp.tool(
    title="Backtest the process",
    description=(
        "Walk the allocation process forward over history, priced with "
        "commission, slippage and square-root market impact. Returns the "
        "spec and result hashes that identify the run, so a re-run can be "
        "told apart from a real change. Blocks while running."
    ),
)
@_reasons_reach_the_client
def backtest(
    config_path: str | None = None,
    sample: bool = False,
    prices_path: str | None = None,
    optimizer: str | None = None,
    lookback: Annotated[int | None, Field(ge=2, le=MAX_ROWS)] = None,
    rebalance_every: Annotated[int | None, Field(ge=1)] = None,
    commission_bps: Annotated[float, Field(ge=0.0, le=10_000.0)] = 5.0,
    slippage_bps: Annotated[float, Field(ge=0.0, le=10_000.0)] = 5.0,
) -> dict[str, Any]:
    """Simulate the process, rather than replaying a fitted allocation.

    The run is shaped by the config the same way ``optengine backtest`` is:
    annualized on the config's ``periods_per_year``, traded only when the
    process re-solves, and its Sharpe measured against the config's
    risk-free rate.

    Args:
        config_path: Path to a YAML mandate. Omit for a minimal default.
        sample: Use the built-in synthetic price panel.
        prices_path: A CSV, Excel or Parquet file of prices.
        optimizer: Override the config's method.
        lookback: Estimation window, in periods, at least 2. Defaults to
            two years on the config's ``periods_per_year``.
        rebalance_every: Periods between re-solves, at least 1. Defaults
            to one quarter. Zero used to be read as "the default" and
            run without a word; the schema now refuses it. The walk may
            ask for at most :data:`MAX_RESOLVES` re-solves in all.
        commission_bps: Broker commission, in basis points of traded value,
            per side. Not negative.
        slippage_bps: Slippage, in basis points of traded value, per side.
            Not negative.

    Returns:
        ``spec_hash`` and ``result_hash`` identify the run; ``degradations``
        names anywhere the simulation had to fall back, which is where its
        costs are optimistic. ``alignment`` says how much history the panel
        lost before the walk started.
    """
    from optimization_engine.backtest.spec import BacktestSpec, CostSpec
    from optimization_engine.engine import run_engine
    from optimization_engine.optimizers._cvxpy_helpers import SolverFailure

    config = _config(config_path, optimizer)
    _, returns, alignment, quality = _panel(sample, prices_path, config, config_path)
    # Counted before the walk starts, on the walk's own defaults: the cost of
    # a call is roughly one solve per re-solve, and a server busy with one
    # call answers no other.
    ppy = config.periods_per_year
    window = lookback or max(2 * ppy, 24)
    cadence = rebalance_every or max(ppy // 4, 1)
    resolves = math.ceil(max(len(returns) - window, 0) / cadence)
    if resolves > MAX_RESOLVES:
        raise ToolError(
            f"This walk-forward would make {resolves} re-solves; this server "
            f"runs at most {MAX_RESOLVES} per call. Raise rebalance_every to "
            f"at least {math.ceil(max(len(returns) - window, 0) / MAX_RESOLVES)}, "
            "shorten the panel, or use `optengine backtest`, which has no such "
            "limit."
        )
    spec = BacktestSpec(
        costs=CostSpec(commission_bps=commission_bps, slippage_bps=slippage_bps),
        periods_per_year=config.periods_per_year,
        frequency="none",
    )
    # Feasibility is checked by `check_mandate`; re-running it here would
    # reject a mandate the walk-forward could still say something useful
    # about on the windows where it does solve.
    try:
        run = run_engine(returns, config, check_feasibility=False)
        walk = run.walk_forward_run(
            lookback=lookback,
            rebalance_every=rebalance_every,
            spec=spec,
        )
    except SolverFailure as exc:
        raise ToolError(f"The initial solve failed: {exc}") from exc
    except ValueError as exc:
        raise ToolError(f"The walk-forward could not run: {exc}") from exc
    return backtest_payload(
        walk.run,
        tearsheet=run.tearsheet(walk.run),
        alignment=alignment,
        quality=quality,
        data_source=_source(sample, prices_path),
    )


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point: serve over stdio.

    Args:
        argv: Arguments to parse, defaulting to ``sys.argv[1:]``. ``--root
            DIR``, repeatable, names a directory the path-taking tools may
            read and replaces the default of the working directory.
    """
    global _command_line_roots

    parser = argparse.ArgumentParser(
        prog="optengine-mcp",
        description="Serve the optimization engine as MCP tools over stdio.",
    )
    parser.add_argument(
        "--root",
        action="append",
        metavar="DIR",
        help=(
            "A directory config_path and prices_path may point into. Repeat "
            f"for several. Overrides {ROOTS_ENV} and the default, which is the "
            "working directory the server started in."
        ),
    )
    args = parser.parse_args(argv)
    if args.root:
        _command_line_roots = tuple(Path(r).expanduser().resolve() for r in args.root)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
