"""Command-line entrypoint: ``optengine``."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from optimization_engine.benchmark import BenchmarkError, BenchmarkSpec
from optimization_engine.config import load_config
from optimization_engine.data.fred import FREDError, load_fred_series
from optimization_engine.data.fx import FXError
from optimization_engine.data.loader import load_prices, prices_to_returns, sample_dataset
from optimization_engine.data.yahoo import YahooFinanceError, load_prices_yahoo
from optimization_engine.engine import (
    apply_fx_conversion,
    resolve_expected_returns,
    run_engine,
)
from optimization_engine.ingest import (
    IngestError,
    IngestRequest,
    describe_providers,
    ingest,
    load_dotenv,
)
from optimization_engine.ingest import fields as ingest_fields
from optimization_engine.optimizers._cvxpy_helpers import SolverFailure
from optimization_engine.optimizers.factory import available_optimizers
from optimization_engine.optimizers.requirements import requirements_for
from optimization_engine.reporting.exporters import (
    excel_writer,
    run_sheets,
    write_excel_report,
)
from optimization_engine.reporting.payloads import (
    SCHEMA_VERSION,
    backtest_payload,
    check_payload,
    describe_payload,
    optimization_payload,
)
from optimization_engine.stress import StressError, load_shocks
from optimization_engine.universe.eligibility import MASK_POLICIES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optengine",
        description="Multi-asset portfolio optimization engine.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    optimize = sub.add_parser("optimize", help="Run an optimization end-to-end.")
    optimize.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit the result as JSON on stdout instead of a formatted report. Human-readable narration moves to stderr, so stdout stays parseable."
        ),
    )
    optimize.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    optimize.add_argument("--prices", help="Excel/CSV/Parquet file of prices.")
    optimize.add_argument("--sheet", default="Precios", help="Excel sheet name.")
    optimize.add_argument("--sample", action="store_true", help="Use built-in sample data.")
    optimize.add_argument(
        "--yahoo",
        help="Download prices from Yahoo Finance. "
             "Pass tickers as a comma- or space-separated list.",
    )
    optimize.add_argument(
        "--yahoo-period", default="5y",
        help="Yahoo period when --yahoo is used (default: 5y).",
    )
    optimize.add_argument("--yahoo-start", help="Yahoo start date (YYYY-MM-DD).")
    optimize.add_argument("--yahoo-end", help="Yahoo end date (YYYY-MM-DD).")
    optimize.add_argument(
        "--base-currency",
        help="Override config.base_currency. Conversion uses FRED FX rates.",
    )
    _add_ingest_arguments(optimize)
    optimize.add_argument(
        "--output",
        help="Excel path for the report. Defaults to outputs.xlsx in the "
             "working directory, except under --json, where no workbook is "
             "written unless this is given. An existing file is replaced, "
             "and the run says so on stderr.",
    )
    optimize.add_argument(
        "--accept-inaccurate",
        action="store_true",
        help=(
            "Accept an approximate solution when no solver converges exactly. Off by default: a solve that only reaches 'optimal_inaccurate' fails rather than report unverified weights as optimal. Turn it on when an indicative book beats no book, and read solver_status on the result to see which one you got. No-op for HRP, HERC and the naive weightings, which never call a solver."
        ),
    )
    optimize.add_argument("--frontier", action="store_true", help="Also compute the frontier.")
    optimize.add_argument("--frontier-points", type=int, default=25)
    optimize.add_argument(
        "--walk-forward", action="store_true",
        help="Also run an out-of-sample walk-forward evaluation of the config.",
    )
    optimize.add_argument(
        "--lookback", type=int,
        help="Walk-forward estimation window in periods (default: two years).",
    )
    optimize.add_argument(
        "--rebalance-every", type=int,
        help="Walk-forward periods between re-solves (default: one quarter).",
    )
    optimize.add_argument(
        "--cost-bps", type=float, default=0.0,
        help="One-way transaction cost in basis points, applied to the backtest.",
    )
    optimize.add_argument(
        "--resample", type=int, metavar="N",
        help="Estimate the frontier's confidence band from N resampled "
             "histories. Requires --frontier.",
    )
    optimize.add_argument(
        "--denoise", action="store_true",
        help="Filter the covariance's noise eigenvalues through the "
             "Marchenko-Pastur cutoff before optimizing.",
    )
    optimize.add_argument(
        "--detone", type=int, default=None, metavar="K",
        help="Remove the K leading eigenvectors (the market component) after "
             "denoising. Makes the covariance singular — use only with the "
             "clustering methods (hrp, herc, nco).",
    )
    optimize.add_argument(
        "--trials", type=int, default=1, metavar="N",
        help="How many configurations you tried before settling on this one. "
             "Used to deflate the walk-forward Sharpe for selection bias; "
             "leaving it at 1 claims you tried exactly one.",
    )
    optimize.add_argument(
        "--mcos", type=int, metavar="N",
        help="Run Monte Carlo Optimization Selection over N simulated "
             "histories, ranking the methods by how reliably each recovers "
             "the allocation the fitted distribution implies.",
    )
    optimize.add_argument(
        "--mcos-methods",
        default="mean_variance,min_variance,hrp,herc,nco",
        help="Comma-separated optimizer names for --mcos.",
    )
    optimize.add_argument(
        "--benchmark", metavar="SPEC",
        help="Benchmark to measure — and optionally optimize — against. Use "
             "'equal_weight' for 1/N, an asset name for a single-asset index, "
             "or 'none' to override the config. Omit to use the config's own "
             "benchmark block.",
    )
    optimize.add_argument(
        "--max-tracking-error", type=float, metavar="TE",
        help="Cap annualized tracking error against the benchmark "
             "(0.03 = 300bp). Imposed inside the solve by the mean-variance "
             "family, mean-CVaR/CDaR and active_mean_variance.",
    )
    optimize.add_argument(
        "--max-active-share", type=float, metavar="AS",
        help="Cap active share against the benchmark (0.4 = 40%%).",
    )
    optimize.add_argument(
        "--stress", metavar="FILE",
        help="YAML/JSON file of stress scenarios to apply to the solved book. "
             "Each entry needs a 'name' and a 'returns' map of asset -> "
             "one-period return; 'covariance_scale' (a number or a full "
             "matrix) and 'notes' are optional. A shock naming an asset "
             "outside the panel is refused, not silently zeroed.",
    )
    optimize.add_argument(
        "--strict", action="store_true",
        help="Refuse to run when the constraints are infeasible, naming the "
             "constraint instead of letting the solver fail.",
    )
    optimize.add_argument(
        "--strict-mandate", action="store_true",
        help="Refuse a solved book that breaches the mandate instead of "
             "reporting the breach. The flag form of the config's "
             "'strict_mandate'. Distinct from --strict, which is a pre-solve "
             "check on the constraints: this one fires *after* a successful "
             "solve, and it is the methods that apply bounds by projection "
             "(HRP, HERC, NCO, the naive weightings) it most often stops.",
    )

    backtest = sub.add_parser(
        "backtest",
        help="Walk-forward the configured process, price the trading, and "
             "report what the search cost. Optionally sweep a grid.",
    )
    backtest.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit the result as JSON on stdout instead of a formatted report. Human-readable narration moves to stderr, so stdout stays parseable."
        ),
    )
    backtest.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    backtest.add_argument("--prices", help="Excel/CSV/Parquet file of prices.")
    backtest.add_argument("--sheet", default="Precios", help="Excel sheet name.")
    backtest.add_argument("--sample", action="store_true", help="Use built-in sample data.")
    backtest.add_argument(
        "--yahoo", help="Comma- or space-separated tickers to download from Yahoo."
    )
    backtest.add_argument(
        "--yahoo-period", default="10y", help="Yahoo lookback when no dates are given."
    )
    backtest.add_argument("--yahoo-start", help="Yahoo start date (YYYY-MM-DD).")
    backtest.add_argument("--yahoo-end", help="Yahoo end date (YYYY-MM-DD).")
    backtest.add_argument(
        "--accept-inaccurate",
        action="store_true",
        help=(
            "Accept an approximate solution when no solver converges exactly. Off by default: a solve that only reaches 'optimal_inaccurate' fails rather than report unverified weights as optimal. Turn it on when an indicative book beats no book, and read solver_status on the result to see which one you got. No-op for HRP, HERC and the naive weightings, which never call a solver. Applies to every re-solve in the walk-forward."
        ),
    )
    backtest.add_argument(
        "--lookback", type=int, metavar="N",
        help="Estimation window in periods. Defaults to two years.",
    )
    backtest.add_argument(
        "--rebalance-every", type=int, metavar="N",
        help="Periods between re-solves — the re-optimization cadence. "
             "Defaults to one quarter.",
    )
    backtest.add_argument(
        "--rebalance", default="none",
        choices=["none", "daily", "weekly", "monthly", "quarterly", "annual"],
        help="How often the book is traded back to the current target "
             "*between* re-solves — the rebalancing cadence, which is a "
             "separate decision from --rebalance-every. Defaults to 'none': "
             "hold each solution untouched until the next one and let the "
             "weights drift. A committee that re-solves quarterly but "
             "rebalances monthly wants --rebalance-every 63 --rebalance "
             "monthly on a daily panel.",
    )
    backtest.add_argument(
        "--expanding", action="store_true",
        help="Grow the estimation window from the start instead of rolling it.",
    )
    backtest.add_argument(
        "--commission-bps", type=float, default=0.0, metavar="BPS",
        help="One-way broker commission on traded notional.",
    )
    backtest.add_argument(
        "--slippage-bps", type=float, default=0.0, metavar="BPS",
        help="One-way spread cost on traded notional.",
    )
    backtest.add_argument(
        "--impact-eta", type=float, default=0.0, metavar="ETA",
        help="Square-root market-impact coefficient. Non-zero makes cost grow "
             "with the square root of trade size, which is the only way "
             "capacity shows up in a backtest.",
    )
    backtest.add_argument(
        "--impact-participation-source", default="fixed", choices=["fixed", "adv"],
        help=(
            "Where the impact model's participation rate comes from. 'fixed' "
            "(default) uses --impact-participation and needs no volume data "
            "at all, which is what lets an index universe be backtested. "
            "'adv' derives it from traded volume, falling back to the fixed "
            "rate — and saying so — for any asset that has none."
        ),
    )
    backtest.add_argument(
        "--impact-adv-share", type=float, default=0.10, metavar="S",
        help="Share of an asset's average daily traded notional this book is "
             "willing to be, under --impact-participation-source adv.",
    )
    backtest.add_argument(
        "--impact-adv-lookback", type=int, default=21, metavar="N",
        help="Trailing periods averaged when computing ADV.",
    )
    backtest.add_argument(
        "--initial-capital", type=float, default=1.0, metavar="NAV",
        help=(
            "Starting NAV, in the currency the prices are quoted in. Cosmetic "
            "for returns, but required by --impact-participation-source adv: "
            "capacity is a currency amount, so the fund's size is what decides "
            "whether a name's daily volume is deep or thin for this book."
        ),
    )
    backtest.add_argument(
        "--impact-participation", type=float, default=0.05, metavar="Q",
        help="Fraction of the book tradable in one name, in one period, "
             "without impact. Smaller means a thinner market.",
    )
    backtest.add_argument(
        "--execution-lag", type=int, default=1, metavar="N",
        help="Periods between a decision and its fill. Defaults to 1 — a desk "
             "does not trade on a close it has not seen. Pass 0 for the "
             "conventional (optimistic) same-period fill.",
    )
    backtest.add_argument(
        "--holdout", metavar="YYYY-MM-DD",
        help="Withhold everything after this date from the walk-forward, then "
             "evaluate on it once and append the visit to the audit log.",
    )
    backtest.add_argument(
        "--audit-log", default="runs/holdout_audit.jsonl",
        help="Where the holdout audit trail is appended.",
    )
    backtest.add_argument(
        "--sweep", metavar="PATH=V1,V2",
        action="append",
        help="Sweep a config path over values, e.g. "
             "'optimizer.name=min_variance,risk_parity'. Repeat for a grid. "
             "Every cell is walk-forwarded and the trial count is carried "
             "into the deflated Sharpe.",
    )
    backtest.add_argument(
        "--stress", metavar="FILE",
        help="YAML/JSON file of stress scenarios. The tearsheet gains a "
             "stress panel over the book the walk-forward ended on.",
    )
    backtest.add_argument(
        "--strict-mandate", action="store_true",
        help="Refuse a solved book that breaches the mandate. Read the "
             "warning on --strict-mandate for `optimize` first: inside a "
             "walk-forward a refused window is caught per-window and recorded "
             "as a failed solve, so this does not stop the run — it turns "
             "non-compliant windows into carried-forward ones, counted in "
             "the 'solve(s) failed' line.",
    )
    backtest.add_argument(
        "--universe", metavar="FILE",
        help="YAML/JSON rules file defining point-in-time membership: which "
             "names were investable, and when. Without one the universe is "
             "'every column of the panel, from the first bar', which is "
             "survivorship bias and look-ahead in the same frame. See "
             "optimization_engine.universe.rules for the schema.",
    )
    backtest.add_argument(
        "--universe-policy", default="exclude",
        choices=list(MASK_POLICIES),
        help="How a date/asset cell the rules could not evaluate is read. "
             "'exclude' (the default) treats it as ineligible and prints how "
             "many cells that silently removed; 'include' admits a name "
             "nothing screened; 'raise' refuses to guess and stops the run. "
             "The library API has no default here on purpose — the CLI picks "
             "one because a non-interactive run cannot ask, and then says "
             "what the choice cost.",
    )
    backtest.add_argument(
        "--delisting-grace", type=int, metavar="N",
        help="Bars of silence after which a name counts as delisted, dropped "
             "from the solve and sold at its last mark. Separate from "
             "--universe and opt-in on its own: a screen says what the "
             "mandate permits, this says what still trades. Omitted, "
             "delisting is not diagnosed at all and a name that stopped "
             "printing is simply held.",
    )
    backtest.add_argument(
        "--output", help="Optional Excel path for the tearsheet frames."
    )

    _add_ingest_arguments(backtest)

    check = sub.add_parser(
        "check",
        help="Validate data and constraints without solving. Reports data "
             "quality, covariance conditioning and constraint feasibility.",
    )
    check.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit the result as JSON on stdout instead of a formatted report. Human-readable narration moves to stderr, so stdout stays parseable."
        ),
    )
    check.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    check.add_argument("--prices", help="Excel/CSV/Parquet file of prices.")
    check.add_argument("--sheet", default="Precios", help="Excel sheet name.")
    check.add_argument("--sample", action="store_true", help="Use built-in sample data.")
    check.add_argument(
        "--denoise", action="store_true",
        help="Report the conditioning of the denoised covariance instead.",
    )
    check.add_argument(
        "--detone", type=int, default=None, metavar="K",
        help="Remove K leading eigenvectors after denoising.",
    )
    check.add_argument(
        "--base-currency", help="Override the config's base currency for FX conversion."
    )
    check.add_argument(
        "--benchmark",
        help="Benchmark to check the mandate against: a kind or an asset name.",
    )
    check.add_argument(
        "--accept-inaccurate",
        action="store_true",
        help=(
            "Let the reachable-return LPs settle for an approximate answer. "
            "Off by default, in which case a range the solver could not "
            "verify is reported as 'we could not tell' rather than as a "
            "range. This is the pre-flight's half of the same flag "
            "'optimize' and 'backtest' pass to the solve itself."
        ),
    )
    check.add_argument("--max-tracking-error", type=float, default=None)
    check.add_argument("--max-active-share", type=float, default=None)

    _add_ingest_arguments(check)

    describe = sub.add_parser(
        "describe", help="Explain one optimizer: what it needs and what it assumes."
    )
    describe.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit the result as JSON on stdout instead of a formatted report. Human-readable narration moves to stderr, so stdout stays parseable."
        ),
    )
    describe.add_argument("name", help="Optimizer name (see list-optimizers).")

    sub.add_parser("list-optimizers", help="List available optimizer names.")

    providers = sub.add_parser(
        "providers",
        help="List data providers, what each can serve, and whether its key is set.",
    )
    providers.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON."
    )
    providers.add_argument(
        "--env-file", help="Load API keys from this .env file before reporting."
    )

    ingest_cmd = sub.add_parser(
        "ingest",
        help="Fetch a price panel from a provider and write it to disk.",
    )
    _add_ingest_arguments(ingest_cmd)
    ingest_cmd.add_argument(
        "--output", default="prices.csv",
        help="Where to write the close panel (.csv, .xlsx or .parquet).",
    )
    ingest_cmd.add_argument(
        "--volume-output",
        help=(
            "Also write the volume panel here. Skipped with a note when the "
            "universe carries no volume, which is the norm for indices."
        ),
    )

    fred = sub.add_parser("fred", help="Fetch one or more FRED series and write to disk.")
    fred.add_argument("series", help="Comma- or space-separated series ids (e.g. 'DGS10,VIXCLS').")
    fred.add_argument("--start", help="Start date (YYYY-MM-DD).")
    fred.add_argument("--end", help="End date (YYYY-MM-DD).")
    fred.add_argument("--output", default="fred_data.csv", help="Output CSV path.")

    sample = sub.add_parser("sample-data", help="Write a synthetic price panel to disk.")
    sample.add_argument("--output", default="data/sample/sample_prices.csv")
    sample.add_argument("--periods", type=int, default=252 * 8)

    return parser


def _load_stress_into(config, args: argparse.Namespace) -> int:
    """Read ``--stress`` into ``config.stress``, or report why it could not.

    Args:
        config: The configuration to attach the scenarios to, in place.
        args: The parsed command line. ``--stress`` is optional; without it
            the config keeps whatever its own file declared.

    Returns:
        ``0`` on success (including when no file was named), ``2`` when the
        file could not be read or parsed — the message is on stderr.
    """
    path = getattr(args, "stress", None)
    if not path:
        return 0
    try:
        config.stress = load_shocks(path)
    except (OSError, StressError) as exc:
        return _fail(args, f"Could not read stress scenarios from {path}: {exc}")
    print(f"Loaded {len(config.stress)} stress scenario(s) from {path}")
    return 0


def _load_universe_for(args: argparse.Namespace, returns, prices):
    """Read ``--universe`` into an :class:`Eligibility`, and say what it costs.

    The rules file carries the paths to any characteristic panel it needs —
    ADV, market capitalisation — because :func:`_prepare_inputs` loads prices
    and nothing else; see :mod:`optimization_engine.universe.rules` for why
    the data lives with the rules rather than behind a second flag. The run's
    own panels are handed in here under the names ``returns`` and ``prices``,
    so a screen written only over those needs no data files at all.

    Args:
        args: The parsed command line. ``--universe`` is optional.
        returns: The aligned return panel, available to rules as ``returns``.
        prices: The aligned price panel, available as ``prices``.

    Returns:
        The :class:`~optimization_engine.universe.eligibility.Eligibility`,
        ``None`` when no rules file was named, or ``2`` when the file could
        not be read — the message is on stderr.
    """
    from optimization_engine.universe.rules import (
        count_unresolved,
        load_universe_rules,
    )
    from optimization_engine.universe.signal import UniverseError

    path = getattr(args, "universe", None)
    if not path:
        return None
    try:
        rules = load_universe_rules(path)
        universe = rules.build(returns=returns, prices=prices)
    except (OSError, UniverseError) as exc:
        return _fail(args, f"Could not read the universe from {path}: {exc}")
    print(rules.describe())

    policy = getattr(args, "universe_policy", "exclude")
    cells, bars, names = count_unresolved(universe, returns.index, list(returns.columns))
    if cells:
        # Unconditionally on stderr, like the alignment log and for the same
        # reason: the library refuses to pick a collapse policy, this command
        # picked one, and the size of what that decided is not an advanced-mode
        # detail. Under "raise" the run is about to stop on these very cells,
        # so the count is the diagnosis rather than a footnote.
        verdict = {
            "exclude": "reads them as ineligible",
            "include": "admits them",
            "raise": "will stop the run on them",
        }[policy]
        print(
            f"  Universe: {cells} date/asset cell(s) across {bars} bar(s) and "
            f"{len(names)} name(s) were not evaluable, and the {policy!r} "
            f"policy — not a screen — {verdict}: "
            f"{', '.join(names[:5])}{' …' if len(names) > 5 else ''}.",
            file=sys.stderr,
        )
    else:
        print(
            "  Universe: every date/asset cell was evaluable, so the "
            f"{policy!r} policy decides nothing.",
            file=sys.stderr,
        )
    return universe


def _cmd_optimize(args: argparse.Namespace) -> int:
    from optimization_engine.optimizers.audit import MandateViolationError
    from optimization_engine.optimizers.feasibility import InfeasibleConstraintsError

    inputs = _prepare_inputs(args)
    if isinstance(inputs, int):
        return inputs
    config, returns, quality = inputs.config, inputs.returns, inputs.quality
    alignment = inputs.alignment
    if _load_stress_into(config, args) != 0:
        return 2
    for issue in quality.errors:
        print(f"Data error — {issue.describe()}", file=sys.stderr)
    for issue in quality.warnings:
        print(f"Data warning — {issue.describe()}", file=sys.stderr)
    if quality.errors and args.strict:
        return _fail(
            args,
            "Refusing to optimize on data with errors. Drop --strict to "
            "proceed anyway.",
        )

    # Every refusal below goes through `_fail` rather than a bare print, so
    # the reason reaches the --json payload as well as stderr. Printed and
    # returned, it left the payload saying only that the command "exited
    # before producing a result".
    try:
        run = run_engine(
            returns,
            config,
            build_frontier=args.frontier,
            n_frontier_points=args.frontier_points,
            raise_on_infeasible=args.strict,
            run_stress=bool(config.stress),
        )
    except InfeasibleConstraintsError as exc:
        return _fail(args, str(exc))
    except StressError as exc:
        return _fail(args, f"Stress test failed: {exc}")
    except MandateViolationError as exc:
        # Raised only under --strict-mandate, and *after* a successful solve:
        # the answer arrived and does not comply. It is a ValueError while
        # SolverFailure is a RuntimeError, so the clause below never sees it —
        # without this one the CLI would leak a traceback for the one failure
        # the flag exists to produce.
        return _fail(
            args,
            f"{exc}\n"
            "  The mandate is satisfiable; this method did not satisfy it. "
            "Pick a method whose bounds are enforced inside the convex "
            "program, loosen the limit named above, or drop --strict-mandate "
            "and read the audit on the result.",
        )
    except SolverFailure as exc:
        message = f"Optimization failed: {exc}"
        if config.max_tracking_error is not None or config.max_active_share is not None:
            message += (
                "\n  A tracking-error or active-share budget is in force. A "
                "benchmark holding an asset your bounds cap below its index "
                "weight sets a floor on tracking error that no allocation can "
                "go below — raise the limit, or relax the bound."
            )
        return _fail(args, message)

    for warning in run.warnings:
        print(f"Warning — {warning}", file=sys.stderr)

    if run.stress is not None:
        print(run.stress.describe())

    walk_forward = None
    if args.walk_forward:
        try:
            walk_forward = run.walk_forward(
                lookback=args.lookback,
                rebalance_every=args.rebalance_every,
                transaction_cost_bps=args.cost_bps,
            )
        except ValueError as exc:
            print(f"Walk-forward skipped: {exc}", file=sys.stderr)

    uncertainty = None
    if args.resample:
        if not args.frontier:
            print(
                "--resample needs --frontier; skipping the uncertainty band.",
                file=sys.stderr,
            )
        else:
            from optimization_engine.resampling import bootstrap_frontier

            try:
                uncertainty = bootstrap_frontier(
                    returns, config,
                    n_draws=int(args.resample),
                    n_points=max(args.frontier_points // 2, 6),
                )
                print(f"  {uncertainty.summary()}")
            except ValueError as exc:
                print(f"Frontier resampling skipped: {exc}", file=sys.stderr)

    print(
        f"{config.optimizer.name}: expected return "
        f"{run.result.expected_return:.2%}, volatility "
        f"{run.result.expected_volatility:.2%}, Sharpe "
        f"{run.result.sharpe_ratio:.2f}"
    )
    if run.diagnostics is not None:
        print(
            f"  {run.diagnostics.n_positions} position(s) · effective N "
            f"{run.diagnostics.effective_n:.1f} · diversification ratio "
            f"{run.diagnostics.diversification_ratio:.2f}"
        )
    _report_layer_exposures(run)
    performance = _report_versus_benchmark(run, config)

    # Under --json the result is the document on stdout, so a workbook is
    # written only when one is asked for; the human default stays, because
    # a person running `optimize` expects a report to open.
    output = args.output or (None if args.json else "outputs.xlsx")
    out = None
    sheets: dict = {}
    if output is not None:
        sheets = run_sheets(
            run,
            riskfree_rate=config.optimizer.risk_free_rate,
            data_quality=quality,
            walk_forward=walk_forward,
            frontier_uncertainty=uncertainty,
            performance=performance,
        )
        _announce_overwrite(output)
        out = write_excel_report(output, sheets)
    if walk_forward is not None:
        comparison = run.in_vs_out_of_sample(
            walk_forward, config.optimizer.risk_free_rate
        )
        gap = float(comparison.loc["Sharpe Ratio", "Degradation"])
        print(
            f"  Walk-forward: {walk_forward.n_resolves} re-solve(s) over "
            f"{walk_forward.n_trade_dates} trade date(s), "
            f"Sharpe falls {gap:.2f} out of sample"
        )
        _report_deflated_sharpe(walk_forward.returns, args, config)

    if args.mcos:
        from optimization_engine.resampling import monte_carlo_optimization_selection

        methods = tuple(
            m.strip() for m in str(args.mcos_methods).replace(" ", ",").split(",") if m.strip()
        )
        try:
            selection = monte_carlo_optimization_selection(
                returns, config, methods=methods, n_simulations=int(args.mcos)
            )
        except ValueError as exc:
            print(f"MCOS skipped: {exc}", file=sys.stderr)
        else:
            print(f"  {selection.describe()}")
            for name, row in selection.ranking().iterrows():
                print(
                    f"    {name:<18} weight RMSE {row['weight_rmse']:.2%} · "
                    f"worst position {row['max_weight_drift']:.2%}"
                )
    if out is not None:
        print(f"Wrote {out} ({len(sheets)} sheets)")
    _capture(
        args,
        optimization_payload(
            run,
            output_path=str(out) if out is not None else None,
            alignment=alignment,
            quality=quality,
            ingest=inputs.ingest,
            data_source=inputs.data_source,
        ),
    )
    return 0


def _announce_overwrite(path: str | Path) -> None:
    """Say on stderr that a file is about to be replaced.

    Replacing is still what happens — a re-run that refused to overwrite
    its own last report would break the commonest workflow there is — but
    not in silence: `optimize` used to replace an `outputs.xlsx` it had
    never been asked to write, and the first anyone heard of it was the
    missing workbook.
    """
    if Path(path).exists():
        print(f"  Overwriting {path}.", file=sys.stderr)


def _apply_benchmark_flags(
    config, args: argparse.Namespace, assets: list[str]
) -> None:
    """Let ``--benchmark`` and the two limits override the config's block.

    ``--benchmark`` accepts a kind or an asset name, because on the command
    line "compare this against SPY" is the common case and forcing the
    ``kind: single_asset`` YAML for it would be ceremony.

    Raises:
        BenchmarkError: When the argument names neither a known kind nor an
            asset in the universe — better than silently comparing against
            something the caller did not ask for.
    """
    raw = getattr(args, "benchmark", None)
    if raw:
        value = str(raw).strip()
        if value.lower() in ("none", "off"):
            config.benchmark = BenchmarkSpec(kind="none")
            config.benchmark_weights = None
        elif value.lower() in ("equal_weight", "equal-weight", "ew", "1/n"):
            config.benchmark = BenchmarkSpec(kind="equal_weight")
        elif value in assets:
            config.benchmark = BenchmarkSpec(kind="single_asset", asset=value)
        else:
            raise BenchmarkError(
                f"--benchmark {value!r} is neither a benchmark kind nor an "
                f"asset in the universe ({', '.join(assets[:8])}"
                f"{' …' if len(assets) > 8 else ''}). Use 'equal_weight', an "
                "asset name, or define the benchmark in the config."
            )
    if getattr(args, "max_tracking_error", None) is not None:
        config.max_tracking_error = float(args.max_tracking_error)
    if getattr(args, "max_active_share", None) is not None:
        config.max_active_share = float(args.max_active_share)


def _report_layer_exposures(run) -> None:
    """Print each layer's bucket exposures next to their caps.

    Only the bucket that is *binding* explains the allocation, so the marker
    goes there: an allocator scanning the output should be able to see which
    line of the policy produced the portfolio without opening the workbook.
    """
    exposures = run.layer_exposures()
    if exposures.empty:
        return
    for layer_name, block in exposures.groupby("layer", sort=False):
        print(f"  {layer_name}:")
        for _, row in block.iterrows():
            limits = ""
            if pd.notna(row["effective_max"]):
                floor = (
                    f"{row['effective_min']:.1%}"
                    if pd.notna(row["effective_min"]) and row["effective_min"] > 0
                    else "0%"
                )
                limits = f" / limit {floor}–{row['effective_max']:.1%}"
                if row["basis"] == "parent":
                    limits += (
                        f" ({row['min']:.0%}–{row['max']:.0%} of {row['parent']})"
                    )
            mark = "  ←binding" if row["binding"] else ""
            print(f"    {row['bucket']:<24} {row['weight']:>7.2%}{limits}{mark}")


def _report_versus_benchmark(run, config):
    """Print the relative headline and return the report for the workbook.

    Returns ``None`` when the run has no benchmark, which is also what tells
    :func:`run_sheets` there is nothing relative to write.
    """
    if run.benchmark is None:
        return None
    try:
        report = run.performance(riskfree_rate=config.optimizer.risk_free_rate)
    except ValueError as exc:
        print(f"  Relative performance skipped: {exc}", file=sys.stderr)
        return None
    h = report.headline()
    print(
        f"  vs {run.benchmark_label}: excess {h['excess_return']:+.2%} · "
        f"T.E. {h['tracking_error']:.2%} · IR {h['information_ratio']:.2f} · "
        f"beta {h['beta']:.2f}"
        + (
            f" · active share {h['active_share']:.1%}"
            if "active_share" in h
            else ""
        )
    )
    return report


def _report_deflated_sharpe(returns, args: argparse.Namespace, config) -> None:
    """Print the walk-forward Sharpe deflated for the number of trials.

    An out-of-sample Sharpe is still a selected number when the configuration
    that produced it was itself chosen by looking at results. ``--trials``
    is how the analyst declares that search; the default of 1 is a claim, and
    the printed line says so.
    """
    from optimization_engine.analytics.selection import (
        deflated_sharpe_ratio,
        minimum_track_record_length,
    )

    try:
        deflated = deflated_sharpe_ratio(
            returns,
            n_trials=max(int(args.trials), 1),
            riskfree_rate=config.optimizer.risk_free_rate,
            periods_per_year=config.periods_per_year,
        )
    except ValueError as exc:
        print(f"  Deflated Sharpe skipped: {exc}", file=sys.stderr)
        return
    print(f"  {deflated.describe()}")
    try:
        needed = minimum_track_record_length(
            returns,
            benchmark_sharpe=deflated.benchmark_sharpe,
            riskfree_rate=config.optimizer.risk_free_rate,
            periods_per_year=config.periods_per_year,
        )
    except ValueError:
        return
    if needed == float("inf"):
        print(
            "  Minimum track record: unreachable — this Sharpe does not "
            "exceed the selection-bias threshold at any sample length."
        )
    else:
        print(
            f"  Minimum track record to call it significant at 95%: "
            f"{needed / config.periods_per_year:.1f} year(s) "
            f"({needed:.0f} periods); this run has "
            f"{len(returns)}."
        )


def _cmd_list_optimizers() -> int:
    width = max(len(n) for n in available_optimizers())
    for name in available_optimizers():
        print(f"{name:<{width}}  {requirements_for(name).summary}")
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    try:
        req = requirements_for(args.name)
    except KeyError as exc:
        return _fail(args, str(exc).strip("\""))
    print(f"{req.display_name}  ({req.name})")
    print(f"\n  {req.summary}")
    print(f"\nUse it when:\n  {req.when_to_use}")
    if req.assumptions:
        print("\nAssumptions:")
        for a in req.assumptions:
            print(f"  - {a}")
    needs = [
        label
        for flag, label in (
            (req.requires_mu, "expected returns"),
            (req.requires_cov, "a covariance matrix"),
            (req.requires_returns, "the full return history"),
        )
        if flag
    ]
    print(f"\nInputs: {', '.join(needs) if needs else 'none beyond the universe'}")
    supports = [
        label
        for flag, label in (
            (req.supports_target_return, "target return"),
            (req.supports_target_volatility, "target volatility"),
            (req.supports_risk_aversion, "risk-aversion utility"),
            (req.supports_group_bounds, "group bounds"),
            (req.supports_turnover, "turnover budget"),
            (req.supports_frontier, "efficient frontier"),
        )
        if flag
    ]
    print(f"Supports: {', '.join(supports) if supports else 'no optional settings'}")
    print(f"Bounds: {req.bounds_note}")
    _capture(args, describe_payload(req))
    return 0


def _apply_estimator_flags(config, args: argparse.Namespace) -> None:
    """Let command-line flags override the config file's estimator settings.

    Kept in one place so ``check`` and ``optimize`` cannot diverge: a
    pre-flight that reports the conditioning of a matrix the solve will not
    use is worse than no pre-flight at all. The same argument covers
    ``--accept-inaccurate``: the command that says the mandate is reachable
    and the command that solves it have to agree about what counts as an
    answer.

    ``getattr`` throughout because the three subcommands that share
    :func:`_prepare_inputs` do not carry identical flags.
    """
    if getattr(args, "denoise", False):
        config.denoise = True
    detone = getattr(args, "detone", None)
    if detone is not None:
        config.detone = int(detone)
    if getattr(args, "accept_inaccurate", False):
        # One-way on purpose. The flag is an opt-in to a weaker answer, so
        # its absence must not overwrite a config that already asked for one.
        config.optimizer.accept_inaccurate = True
    if getattr(args, "strict_mandate", False):
        # One-way for the same reason, in the other direction: the flag is an
        # opt-in to a *stricter* answer, and its absence must not switch off a
        # config that already asked to refuse a non-compliant book.
        config.strict_mandate = True




#: Field presets exposed on the command line. ``close`` is the default because
#: it is all the optimizer needs; ``ohlcv`` is what a capacity-aware backtest
#: needs, and asking for it from a provider that has no volume fails loudly at
#: preflight instead of quietly returning a short panel.
_FIELD_PRESETS = {
    "close": ingest_fields.PRICE_ONLY,
    "ohlc": ingest_fields.OHLC,
    "ohlcv": ingest_fields.OHLCV,
}


def _add_ingest_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the multi-provider ingest flags to a subcommand.

    Added to every command that needs a price panel, so switching a run from a
    spreadsheet to a live provider is one flag rather than a different
    workflow.
    """
    group = parser.add_argument_group("data ingest (multi-provider)")
    group.add_argument(
        "--provider",
        help=(
            "Data provider to fetch from: "
            f"{', '.join(_provider_names())}. Requires --identifiers."
        ),
    )
    group.add_argument(
        "--identifiers",
        help="Comma- or space-separated universe, e.g. 'SPY,AGG,GLD' or 'SP500,IPC'.",
    )
    group.add_argument("--ingest-start", help="Inclusive start date (YYYY-MM-DD).")
    group.add_argument("--ingest-end", help="Inclusive end date (YYYY-MM-DD).")
    group.add_argument(
        "--ingest-period", default="5y",
        help="Window when no start date is given (1y, 2y, 3y, 5y, 10y, 20y).",
    )
    group.add_argument(
        "--ingest-interval", default="1d", choices=["1d", "1wk", "1mo"],
        help="Bar size.",
    )
    group.add_argument(
        "--ingest-fields", default="close", choices=sorted(_FIELD_PRESETS),
        help=(
            "Which fields to fetch. 'close' is enough to optimize and "
            "backtest; 'ohlcv' adds the volume a capacity-aware cost model "
            "needs, where the provider publishes it."
        ),
    )
    group.add_argument(
        "--ingest-currency",
        help="Convert every series into this ISO currency (e.g. USD, MXN).",
    )
    group.add_argument(
        "--require-volume", action="store_true",
        help=(
            "Fail when an instrument that should report volume does not. Off "
            "by default: indices have no volume and are backtested from a "
            "fixed participation rate instead."
        ),
    )
    group.add_argument(
        "--cache-dir",
        help="Directory for the on-disk panel cache. Omit to disable caching.",
    )
    group.add_argument(
        "--file-path",
        help="Path read by the 'file' provider (CSV, Excel or Parquet).",
    )
    group.add_argument(
        "--env-file", help="Load API keys from this .env file before fetching."
    )


def _provider_names() -> tuple[str, ...]:
    from optimization_engine.ingest import available_providers

    return available_providers()


def _ingest_request_from(args: argparse.Namespace) -> IngestRequest:
    """Build an :class:`IngestRequest` from the shared ingest flags."""
    return IngestRequest(
        identifiers=getattr(args, "identifiers", "") or "",
        provider=args.provider,
        start=getattr(args, "ingest_start", None),
        end=getattr(args, "ingest_end", None),
        period=None if getattr(args, "ingest_start", None) else args.ingest_period,
        interval=args.ingest_interval,
        fields=_FIELD_PRESETS[args.ingest_fields],
        currency=getattr(args, "ingest_currency", None),
        require_volume=bool(getattr(args, "require_volume", False)),
        cache_dir=getattr(args, "cache_dir", None),
    )


def _run_ingest(args: argparse.Namespace):
    """Fetch a panel and print the per-identifier outcome.

    Returns the :class:`~optimization_engine.ingest.IngestResult` so callers
    can take both the prices and the volume, which is what makes a
    capacity-aware backtest possible from the command line.
    """
    if getattr(args, "env_file", None):
        loaded = load_dotenv(args.env_file)
        print(f"Loaded {loaded} variable(s) from {args.env_file}")

    options = {}
    if getattr(args, "file_path", None):
        options["path"] = args.file_path

    result = ingest(_ingest_request_from(args), **options)
    print(f"Ingest: {result.summary()}")
    for outcome in result.failed:
        print(f"  ! {outcome.identifier}: {outcome.status} — {outcome.message}")
    for note in result.warnings:
        print(f"  · {note}")
    return result


def _load_prices_for(args: argparse.Namespace):
    """Resolve the price panel from whichever legacy flag names it.

    ``--provider`` is handled by :func:`_prepare_inputs`, which needs the
    whole ingest result rather than the prices alone. Exactly one source has
    been named by the time this runs — :func:`_data_source` checked.
    """
    if getattr(args, "yahoo", None):
        if args.yahoo_start:
            return load_prices_yahoo(
                args.yahoo, start=args.yahoo_start, end=args.yahoo_end
            )
        return load_prices_yahoo(args.yahoo, period=args.yahoo_period)
    if args.sample:
        return sample_dataset()
    return load_prices(args.prices, sheet_name=args.sheet)


def _split_identifiers(raw: str | None) -> list[str]:
    return [token for token in str(raw or "").replace(",", " ").split() if token]


def _data_source(args: argparse.Namespace) -> dict[str, object]:
    """The one panel this run reads, described the way the payload reports it.

    A source has to be named. ``--sample`` used to be what a command fell
    back to when ``--prices`` was missing, so a forgotten flag produced an
    exit-0 allocation on synthetic data with nothing on any stream saying
    so. Naming two is refused for the same reason: ``--sample`` silently won
    over ``--prices``, which is the same substitution reached the other way.

    Raises:
        ValueError: When the command line names no source, or more than one.
            The message lists the flags.
    """
    named = [
        flag
        for flag, present in (
            ("--prices", getattr(args, "prices", None)),
            ("--provider", getattr(args, "provider", None)),
            ("--yahoo", getattr(args, "yahoo", None)),
            ("--sample", getattr(args, "sample", False)),
        )
        if present
    ]
    if not named:
        raise ValueError(
            "No price data given. Pass --prices FILE, --provider NAME with "
            "--identifiers, --yahoo TICKERS, or --sample for the built-in "
            "synthetic panel."
        )
    if len(named) > 1:
        raise ValueError(
            f"Pass one data source, not {len(named)}: {', '.join(named)}. "
            "It would otherwise be ambiguous which panel the result describes."
        )
    source: dict[str, object] = {
        "kind": {"--prices": "file", "--provider": "provider", "--yahoo": "yahoo",
                 "--sample": "sample"}[named[0]],
        "synthetic": False,
        "path": None,
        "provider": None,
        "identifiers": None,
    }
    if named[0] == "--prices":
        source["path"] = args.prices
    elif named[0] == "--yahoo":
        source["identifiers"] = _split_identifiers(args.yahoo)
    elif named[0] == "--provider":
        source["provider"] = args.provider
        source["identifiers"] = _split_identifiers(getattr(args, "identifiers", None))
        source["path"] = getattr(args, "file_path", None) if args.provider == "file" else None
        # The ingest layer has a synthetic provider of its own; it is no more
        # market data than --sample is.
        source["synthetic"] = args.provider == "sample"
    else:
        source["synthetic"] = True
    return source


@dataclass
class _Inputs:
    """What every solving command starts from, built one way."""

    config: object
    prices: pd.DataFrame
    returns: pd.DataFrame
    quality: object
    volumes: pd.DataFrame | None
    #: One sentence per change alignment made to the panel. Empty means
    #: nothing was dropped, which is a claim worth being able to make.
    alignment: list[str]
    #: Where the prices came from, as :func:`_data_source` describes it.
    data_source: dict[str, object]
    #: The ingest the panel came from, for the payload's resolved window;
    #: ``None`` for ``--prices``, ``--sample`` and ``--yahoo``.
    ingest: object = None


def _fail(args: argparse.Namespace, message: str, code: int = 2) -> int:
    """Report a failure on stderr — and into the JSON payload, when there is one."""
    print(message, file=sys.stderr)
    sink = getattr(args, "_json_sink", None)
    if sink is not None:
        sink.setdefault("error", message)
    return code


def _prepare_inputs(args: argparse.Namespace) -> _Inputs | int:
    """Load the config and the panel, and shape both the way the solve needs.

    ``check``, ``optimize`` and ``backtest`` all start here, which is what
    makes a pre-flight worth running: it validates the mandate the solve
    goes on to see — same panel, same currency, same universe, same
    benchmark flags — rather than a mandate assembled slightly differently.
    Each used to build its inputs by hand, and they drifted: ``optimize``
    refused a config with no ``expected_returns`` block that ``check`` had
    just called ready, ``check`` never saw ``--base-currency`` or
    ``--benchmark``, and ``backtest`` seeded zero expected returns that
    ``resolve_expected_returns`` would otherwise have estimated.

    The order of operations: estimator flags, then the panel, then currency
    conversion, then the universe, then alignment, then the benchmark flags.
    Currency before universe because a conversion needs every column it is
    told about; universe before alignment because an asset the config never
    asked for must not be allowed to truncate the sample; alignment before
    benchmark because a single-asset benchmark has to name an asset that
    survived.

    Missing data is resolved by :func:`~optimization_engine.data.quality.align_panel`
    rather than by dropping incomplete rows in passing. The operation is
    the same one; what changes is that the caller is told. One asset that
    listed three years after the rest truncates the sample for every other
    asset, and a covariance estimated on three years of a twenty-year
    panel is not the estimate the config asked for.

    Returns:
        The inputs — including the alignment log — or an exit code when
        something the caller can fix is wrong: printed on stderr, and
        carried into the ``--json`` payload.
    """
    from optimization_engine.data.quality import align_panel, analyze_prices

    try:
        data_source = _data_source(args)
    except ValueError as exc:
        return _fail(args, str(exc))
    if data_source["synthetic"]:
        # On stderr, like the alignment log: under --json it is the only
        # stream a person reads, and the payload carries the same fact.
        print(
            "  Data: a synthetic panel — the numbers below describe no market.",
            file=sys.stderr,
        )

    # Inside a handler, because a config that cannot be read is the most
    # ordinary input error there is. Outside one, a missing file or a
    # misspelt key escaped as a traceback with exit 1 — the code that means
    # the engine ran and the answer is no — and under --json it fell into
    # the net meant for defects. The exception's type stays in the message,
    # which is how a caller tells a missing file from a malformed one.
    try:
        config = load_config(args.config)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        return _fail(
            args,
            f"Could not load the config {args.config}: it is not valid YAML "
            f"or JSON ({type(exc).__name__}: {exc})",
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return _fail(
            args, f"Could not load the config {args.config}: {type(exc).__name__}: {exc}"
        )
    _apply_estimator_flags(config, args)

    volumes = None
    ingested = None
    ingested_currency = None
    try:
        if getattr(args, "provider", None):
            # One request serves both prices and volume, so a capacity-aware
            # backtest cannot price impact off a different panel than the
            # one it traded.
            ingested = _run_ingest(args)
            prices, volumes = ingested.prices, ingested.volumes
            ingested_currency = getattr(args, "ingest_currency", None)
        else:
            prices = _load_prices_for(args)
    except YahooFinanceError as exc:
        return _fail(args, f"Yahoo Finance error: {exc}")
    except IngestError as exc:
        return _fail(args, f"Ingest error: {exc}")
    except (OSError, ValueError) as exc:
        # A price file that is missing, has an extension no reader takes, or
        # whose index will not parse as dates: the same input error as a
        # bad config, and the same exit code.
        return _fail(
            args, f"Could not load the price panel: {type(exc).__name__}: {exc}"
        )

    if getattr(args, "base_currency", None):
        config.base_currency = args.base_currency.upper()
    if ingested is not None and ingested_currency:
        # The ingest step converted the series whose currency the provider
        # declared, and stamped them with the base. Applying
        # ``config.currencies`` to those would convert them twice, so their
        # entries are set aside — and only theirs: a series that declared no
        # currency (every file series) arrived unconverted, and the config
        # is the only thing that says what it is quoted in.
        base = ingested_currency.upper()
        converted = {
            name for name, record in ingested.panel.meta.items()
            if record.currency == base
        }
        exempt = sorted(a for a in config.currencies if a in converted)
        if exempt:
            print(
                f"  Currency: {', '.join(exempt)} were converted to {base} on "
                "ingest, so the config's currencies map is not applied to "
                "them again.",
                file=sys.stderr,
            )
        config.currencies = {
            a: c for a, c in config.currencies.items() if a not in converted
        }
        config.base_currency = base
    if config.currencies:
        try:
            prices = apply_fx_conversion(prices, config)
        except FXError as exc:
            return _fail(args, f"FX conversion failed: {exc}")

    if config.expected_returns:
        common = [c for c in prices.columns if c in config.expected_returns]
        if not common:
            return _fail(
                args, "Config has no expected returns matching the price columns."
            )
        prices = prices[common]

    # Before anything is annualized — the quality report included. A config
    # that never set periods_per_year left 252 in place for monthly data, so
    # the factor comes from the ingest interval or the dates instead, and a
    # value the config does state is checked against them.
    from optimization_engine.config import stated_keys
    from optimization_engine.data.frequency import (
        FrequencyMismatchError,
        resolve_periods_per_year,
    )

    try:
        config.periods_per_year, annualization = resolve_periods_per_year(
            prices.index,
            stated=(
                config.periods_per_year
                if "periods_per_year" in stated_keys(args.config)
                else None
            ),
            interval=args.ingest_interval if ingested is not None else None,
            default=config.periods_per_year,
        )
    except FrequencyMismatchError as exc:
        return _fail(args, f"Annualization error: {exc}")
    if annualization:
        print(f"  Annualization: {annualization}", file=sys.stderr)

    # Quality is read off the *raw* panel on purpose: aligning first would
    # hide the very gaps the report exists to name.
    quality = analyze_prices(prices, periods_per_year=config.periods_per_year)

    # `method="common"` keeps the dates on which every asset is present.
    # The alternatives were rejected deliberately: `"ffill"` fabricates
    # prices, and a fabricated flat period understates volatility and
    # correlation in exactly the sample the covariance is estimated on;
    # `"drop_assets"` silently edits the *universe*, which is a mandate
    # decision the config makes and the CLI must not make for it. Losing
    # history is the least dishonest of the three, so long as the loss is
    # stated. The app offers all three (`app/streamlit_app.py`) and defaults
    # to the same one; a non-interactive run does not get to guess.
    #
    # For a late listing — the case this replaced — the surviving sample is
    # identical to what the bare `dropna(how="any")` produced, so no number
    # moves. An *interior* gap does differ, and deliberately: differencing
    # first left NaN at the gap and at the period after it, quietly costing
    # two observations, whereas aligning first drops the gap date and books
    # the move across it as one period. That is the split `prices_to_returns`
    # documents and hands to "the alignment step"; this is that step.
    prices, alignment = align_panel(prices, method="common")
    returns = prices_to_returns(prices)
    # Nothing is missing after alignment, so a NaN surviving here is a
    # degenerate price rather than a listing date. Counting it keeps the
    # guard without restoring the silence.
    n_rows = len(returns)
    returns = returns.dropna(how="any")
    if len(returns) < n_rows:
        alignment.append(
            f"Dropped {n_rows - len(returns)} period(s) whose return could "
            "not be computed from the aligned prices."
        )
    # Unconditionally on stderr: stdout is the parsed stream under
    # `--json`, and a truncated sample is not an advanced-mode detail.
    for action in alignment:
        print(f"  Alignment: {action}", file=sys.stderr)
    if returns.empty:
        return _fail(args, "No usable returns after alignment.")
    if config.expected_returns:
        config.expected_returns = {a: config.expected_returns[a] for a in returns.columns}

    try:
        _apply_benchmark_flags(config, args, list(returns.columns))
    except (BenchmarkError, ValueError) as exc:
        return _fail(args, f"Benchmark error: {exc}")

    return _Inputs(
        config=config,
        prices=prices,
        returns=returns,
        quality=quality,
        volumes=volumes,
        alignment=alignment,
        data_source=data_source,
        ingest=ingested,
    )


def _parse_sweep_arguments(raw: list[str] | None):
    """``--sweep path=a,b`` into the grid the sweep runner expects.

    Values are parsed as JSON when they can be, so numbers and booleans reach
    the config as numbers and booleans rather than as strings a validator
    would later reject.
    """
    import json as _json

    if not raw:
        return None
    params: dict[str, list] = {}
    for entry in raw:
        if "=" not in entry:
            raise ValueError(
                f"--sweep expects PATH=V1,V2 but got {entry!r}. "
                "Example: --sweep optimizer.name=min_variance,risk_parity"
            )
        path, _, values = entry.partition("=")
        parsed = []
        for token in values.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                parsed.append(_json.loads(token))
            except ValueError:
                parsed.append(token)
        if not parsed:
            raise ValueError(f"--sweep {path!r} lists no values.")
        params[path.strip()] = parsed
    from optimization_engine.backtest import SweepSpec

    return SweepSpec(params=params)


def _cmd_backtest(args: argparse.Namespace) -> int:
    """Walk the process forward, price the trading, and count the trials.

    The three things this prints that a plain optimize run cannot: what the
    strategy earned out of sample, what the trading cost to get it, and how
    much of the remaining Sharpe survives being deflated for the size of the
    search that produced it.
    """
    from optimization_engine.backtest import (
        BacktestSpec,
        CostSpec,
        final_holdout_run,
        gate_returns,
        run_backtest,
    )
    from optimization_engine.backtest.spec import SpecValidationError
    from optimization_engine.optimizers.audit import MandateViolationError
    from optimization_engine.universe.signal import UniverseError

    inputs = _prepare_inputs(args)
    if isinstance(inputs, int):
        return inputs
    config, prices, returns, volumes = (
        inputs.config, inputs.prices, inputs.returns, inputs.volumes
    )
    alignment = inputs.alignment
    if _load_stress_into(config, args) != 0:
        return 2
    universe = _load_universe_for(args, returns, prices)
    if isinstance(universe, int):
        return universe

    try:
        spec = BacktestSpec(
            frequency=args.rebalance,
            costs=CostSpec(
                commission_bps=args.commission_bps,
                slippage_bps=args.slippage_bps,
                impact_coefficient=args.impact_eta,
                impact_participation=args.impact_participation,
                impact_participation_source=args.impact_participation_source,
                impact_adv_share=args.impact_adv_share,
                impact_adv_lookback=args.impact_adv_lookback,
            ),
            execution_lag=args.execution_lag,
            periods_per_year=config.periods_per_year,
            initial_capital=args.initial_capital,
            name=Path(args.config).stem,
        )
    except SpecValidationError as exc:
        # The hint only for the one validation it answers: appended to every
        # spec error, it told a caller with a negative --execution-lag to pass
        # --initial-capital.
        hint = " Pass --initial-capital." if "initial_capital" in str(exc) else ""
        return _fail(args, f"{exc}{hint}")
    print(spec.describe())
    if spec.costs.uses_volume and volumes is None:
        print(
            "  Liquidity: no volume panel available, so ADV-based impact "
            "falls back to the fixed participation rate of "
            f"{spec.costs.impact_participation:.1%}. Every affected trade is "
            "listed in the run's degradation notes."
        )

    evaluation = returns
    if args.holdout:
        evaluation = gate_returns(returns, args.holdout)
        print(
            f"  Holdout: walk-forward sees {len(evaluation)} of {len(returns)} "
            f"observations; everything after {args.holdout} is withheld."
        )

    try:
        run = run_engine(evaluation, config, check_feasibility=False)
    except MandateViolationError as exc:
        # --strict-mandate on the anchor solve. Every per-window refusal is
        # caught inside the walk-forward and recorded as a failed solve, but
        # this one is outside it and would otherwise leak a traceback.
        return _fail(args, f"The initial solve was refused: {exc}")
    except SolverFailure as exc:
        return _fail(args, f"The initial solve failed: {exc}")
    try:
        walk = run.walk_forward_run(
            lookback=args.lookback,
            rebalance_every=args.rebalance_every,
            spec=spec,
            expanding=args.expanding,
            prices=prices,
            volumes=volumes,
            universe=universe,
            # Only when there is a universe to read it under. Passed with no
            # universe the runner logs that the policy decides nothing, which
            # is noise on every run that does not use the feature.
            universe_policy=args.universe_policy if universe is not None else None,
            delisting_grace=args.delisting_grace,
        )
    except UniverseError as exc:
        # The 'raise' collapse policy lands here, and it is the one case where
        # the message has to say which flag chose it: a run that stopped
        # because a warm-up period could not be evaluated has not found a
        # problem with the data.
        return _fail(args, f"Universe failed: {exc}")
    except ValueError as exc:
        return _fail(args, f"Walk-forward failed: {exc}")

    print(f"  {walk.run.describe()}")
    print(f"  {walk.describe()}")
    if walk.n_failures:
        print(f"  {walk.n_failures} solve(s) failed; the previous book was carried forward.")

    sweep_results = None
    overfitting = None
    n_trials = 1
    trial_sharpes = None
    try:
        sweep_spec = _parse_sweep_arguments(args.sweep)
    except ValueError as exc:
        print(f"Sweep skipped: {exc}", file=sys.stderr)
        sweep_spec = None
    if sweep_spec is not None:
        # The grid is evaluated under the *same* universe as the headline run.
        # Its cell Sharpes are what the deflated Sharpe is deflated against, so
        # a grid run on the survivors while the run itself was screened would
        # deflate one universe's result by another universe's dispersion.
        sweep_results = run.sweep(
            sweep_spec,
            lookback=args.lookback,
            rebalance_every=args.rebalance_every,
            spec=spec,
            expanding=args.expanding,
            prices=prices,
            volumes=volumes,
            universe=universe,
            universe_policy=args.universe_policy if universe is not None else None,
            delisting_grace=args.delisting_grace,
        )
        print(f"  {sweep_results.describe()}")
        n_trials = max(sweep_results.n_cells, 1)
        trial_sharpes = sweep_results.trial_sharpes()
        try:
            overfitting = sweep_results.overfitting_report()
        except ValueError as exc:
            print(f"  Overfitting analysis skipped: {exc}", file=sys.stderr)
        else:
            print(f"  {overfitting.describe()}")

    try:
        sheet = run.tearsheet(
            walk.run,
            n_trials=n_trials if sweep_results is not None else None,
            trial_sharpes=trial_sharpes,
            overfitting=overfitting,
        )
    except StressError as exc:
        # ``tearsheet`` applies ``config.stress`` to the book the walk-forward
        # ended on, so a shock naming an asset outside the panel surfaces here
        # rather than at the solve.
        return _fail(args, f"Stress test failed: {exc}")
    print(f"  {sheet.tca.describe()}")
    if sheet.deflated_sharpe is not None:
        print(f"  {sheet.deflated_sharpe.describe()}")
    if sheet.stress is not None:
        print(sheet.stress.describe())
    for caveat in sheet.caveats:
        print(f"  ! {caveat}")

    if args.holdout:
        # The last book the gated walk-forward chose is what a desk would
        # actually have been holding when the boundary arrived. Replaying it
        # forward is the one honest question the held-out segment can answer.
        locked = walk.weights_history.iloc[-1]
        holdout_spec = spec.with_(is_out_of_sample=True, name=f"{spec.name}-holdout")
        outcome = final_holdout_run(
            returns,
            args.holdout,
            # The held-out replay is a real run and must be priced the same
            # way: without the volume panel it would silently fall back to the
            # fixed participation rate, so the one segment that is supposed to
            # be the honest answer would be the cheapest.
            lambda segment: run_backtest(
                segment,
                locked,
                holdout_spec,
                prices=prices.reindex(segment.index) if volumes is not None else None,
                volumes=(
                    volumes.reindex(index=segment.index, columns=segment.columns)
                    if volumes is not None
                    else None
                ),
            ).returns,
            strategy={"config": args.config, "spec_hash": spec.spec_hash},
            audit_path=args.audit_log,
            periods_per_year=config.periods_per_year,
        )
        print(f"  {outcome.describe()}")
        print(
            "  Held-out Sharpe: "
            f"{float(outcome.summary.loc['holdout', 'Sharpe Ratio']):.2f}"
        )

    out = None
    if args.output:
        frames = sheet.to_frames()
        if sweep_results is not None:
            frames["sweep"] = sweep_results.frame
        _announce_overwrite(args.output)
        out = write_excel_report(args.output, frames)
        print(f"Wrote {out} ({len(frames)} sheets)")
    _capture(
        args,
        backtest_payload(
            walk.run,
            tearsheet=sheet,
            output_path=str(out) if out is not None else None,
            alignment=alignment,
            quality=inputs.quality,
            ingest=inputs.ingest,
            data_source=inputs.data_source,
        ),
    )
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """Pre-flight the inputs and constraints, and say what would go wrong."""
    from optimization_engine.data.covariance import (
        covariance_diagnostics,
        covariance_from_config,
    )
    from optimization_engine.optimizers._cvxpy_helpers import accepting_inaccurate
    from optimization_engine.optimizers.factory import (
        constraints_from_config,
        effective_expected_returns,
    )
    from optimization_engine.optimizers.feasibility import (
        STAGE_STRUCTURAL,
        analyze_feasibility,
    )

    inputs = _prepare_inputs(args)
    if isinstance(inputs, int):
        return inputs
    config, prices, returns, quality = (
        inputs.config, inputs.prices, inputs.returns, inputs.quality
    )
    alignment = inputs.alignment

    print("== Data quality ==")
    print(quality.describe())
    print(
        f"\nCommon history: {quality.n_common_periods} period(s) "
        f"across {prices.shape[1]} asset(s)."
    )

    print("\n== Estimation ==")
    cov = covariance_from_config(returns, config)
    diag = covariance_diagnostics(
        cov, len(returns), config.covariance_method, config.ewma_lambda
    )
    print(
        f"T/N = {diag.observations_per_asset:.1f} · "
        f"condition number = {diag.condition_number:.3g}"
    )
    denoise_report = cov.attrs.get("denoise_report")
    if denoise_report is not None:
        print(f"  {denoise_report.describe()}")
    for w in diag.warnings:
        print(f"  ! {w}")
    if not diag.warnings:
        print("  Covariance estimate looks well conditioned.")

    print("\n== Constraints ==")
    # The same vector the solve will use. Deriving it differently here —
    # zeros, when the config carries no expected_returns block — would have
    # this command validate a mandate the optimizer never sees.
    mu = resolve_expected_returns(config, returns, cov)
    # The reachable-return LPs run on the same solver chain as the solve, so
    # they inherit the same refusal of an unverified answer. ``check`` builds
    # no optimizer, so there is no constructor to carry the setting down --
    # the scope is opened here instead, which is what keeps the pre-flight
    # and the solve answering the same question.
    with accepting_inaccurate(config.optimizer.accept_inaccurate):
        report = analyze_feasibility(
            list(returns.columns),
            constraints_from_config(config, list(returns.columns)),
            expected_returns=effective_expected_returns(config, cov, mu),
            cov_matrix=cov,
        )
    # Fatal findings and warnings print differently on purpose: the whole
    # point of the two-stage analysis is that "no allocation satisfies these
    # constraints" and "the solver could not answer" are different answers,
    # and a reader who cannot tell them apart is back where they started.
    for issue in report.fatal_issues:
        print(f"  x {issue.message}")
        if issue.suggestion:
            print(f"    -> {issue.suggestion}")
    for issue in report.warnings:
        print(f"  ! {issue.message}")
        if issue.suggestion:
            print(f"    -> {issue.suggestion}")
    if not report.issues:
        print("  Constraints are satisfiable.")

    if report.reachable_return is not None:
        low, high = report.reachable_return
        line = f"Reachable expected return: {low:.2%} to {high:.2%}"
        if report.min_variance_return is not None:
            line += f" (efficient above {report.min_variance_return:.2%})"
        print(line)
    elif report.stage_reached == STAGE_STRUCTURAL and not report.fatal_issues:
        # Saying nothing here would let "Ready to optimize." read as though a
        # return target had been validated against a reachable range that was
        # never computed.
        print(
            "Reachable expected return: not computed — "
            "the range needs a solver and none answered."
        )

    _capture(
        args,
        check_payload(
            quality,
            report,
            diag,
            alignment=alignment,
            ingest=inputs.ingest,
            data_source=inputs.data_source,
        ),
    )
    if report.fatal_issues:
        print("\nNot ready to optimize.", file=sys.stderr)
        return 2
    if quality.errors:
        print("\nNot ready to optimize.", file=sys.stderr)
        return 1
    print("\nReady to optimize.")
    return 0


def _cmd_providers(args: argparse.Namespace) -> int:
    """Show every provider, what it serves, and whether it is usable now.

    The single most common failure in a multi-provider setup is asking a
    provider for something it does not publish, or forgetting a key. Both are
    visible here in one screen, before any run.
    """
    if getattr(args, "env_file", None):
        load_dotenv(args.env_file)

    rows = describe_providers()
    if args.json:
        import json

        print(json.dumps(list(rows), indent=2, default=str))
        return 0

    print(f"{len(rows)} data providers\n")
    for row in rows:
        mark = "ready" if row["ready"] else "needs key"
        print(f"  {row['provider']:<8} [{mark}]  {row['description']}")
        print(
            f"    fields:    {', '.join(f.removeprefix('m_') for f in row['fields'])}"
        )
        print(f"    intervals: {', '.join(row['intervals'])}")
        print(
            f"    volume:    {'yes' if row['serves_volume'] else 'no — index-style levels only'}"
        )
        print(f"    key:       {row['key_label']}")
        if row["signup_url"]:
            print(f"    sign up:   {row['signup_url']}")
        if row["notes"]:
            print(f"    note:      {row['notes']}")
        print()
    print(
        "Set a key with the provider's environment variable, or put it in a "
        ".env file and pass --env-file."
    )
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Fetch a panel and write it out, prices and volume separately."""
    if not args.provider:
        print("--provider is required for `ingest`.", file=sys.stderr)
        return 2
    if not args.identifiers:
        print("--identifiers is required for `ingest`.", file=sys.stderr)
        return 2

    try:
        result = _run_ingest(args)
    except IngestError as exc:
        print(f"Ingest error: {exc}", file=sys.stderr)
        return 2

    written = _write_panel(Path(args.output), result.prices)
    if written is None:
        return 2
    print(f"Wrote {args.output} ({result.prices.shape[0]} rows × "
          f"{result.prices.shape[1]} series)")

    if args.volume_output:
        volumes = result.volumes
        if volumes is None:
            print(
                "No volume to write: this universe carries none. The backtest "
                "will price impact from a fixed participation rate."
            )
        elif _write_panel(Path(args.volume_output), volumes) is not None:
            print(f"Wrote {args.volume_output} ({volumes.shape[1]} series)")

    print()
    print(result.panel.coverage().to_string())
    return 0 if result.is_complete else 1


def _write_panel(path: Path, frame: pd.DataFrame) -> Path | None:
    """Write a frame in the format its extension names.

    Parquet is the one format that needs a package the project does not depend
    on, so a missing engine is reported as the install it needs rather than as
    pandas' own two-paragraph ImportError.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        # Through the shared writer: a provider's or a file's column names
        # land in the header row, and must not be written as formulas.
        with excel_writer(path) as writer:
            frame.to_excel(writer, sheet_name="Precios")
    elif suffix == ".parquet":
        try:
            frame.to_parquet(path)
        except ImportError:
            print(
                "Writing Parquet needs pyarrow, which is not installed. "
                'Install it with: pip install -e ".[data]" — or write .csv '
                "or .xlsx instead.",
                file=sys.stderr,
            )
            return None
    else:
        print(f"Unsupported output extension: {suffix}", file=sys.stderr)
        return None
    return path


def _cmd_sample_data(args: argparse.Namespace) -> int:
    prices = sample_dataset(n_periods=args.periods)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        prices.to_csv(out)
    elif out.suffix.lower() in {".xlsx", ".xls"}:
        with excel_writer(out) as writer:
            prices.to_excel(writer, sheet_name="Precios")
    elif out.suffix.lower() == ".parquet":
        prices.to_parquet(out)
    else:
        print(f"Unsupported output extension: {out.suffix}", file=sys.stderr)
        return 2
    print(f"Wrote {out} ({prices.shape[0]} rows × {prices.shape[1]} cols)")
    return 0


def _cmd_fred(args: argparse.Namespace) -> int:
    try:
        df = load_fred_series(args.series, start=args.start, end=args.end)
    except FREDError as exc:
        print(f"FRED error: {exc}", file=sys.stderr)
        return 2
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        df.to_csv(out)
    elif out.suffix.lower() in {".xlsx", ".xls"}:
        with excel_writer(out) as writer:
            df.to_excel(writer)
    elif out.suffix.lower() == ".parquet":
        df.to_parquet(out)
    else:
        print(f"Unsupported output extension: {out.suffix}", file=sys.stderr)
        return 2
    print(f"Wrote {out} ({df.shape[0]} rows × {df.shape[1]} series)")
    return 0


def _emit_json(
    args: argparse.Namespace, command: Callable[[argparse.Namespace], int]
) -> int:
    """Run a command in JSON mode: payload on stdout, narration on stderr.

    The commands below print as they work — data-quality findings, solver
    fallbacks, constraint warnings — and that narration is useful even to a
    machine caller, but not on the stream it is parsing. So stdout is
    redirected to stderr for the duration and the payload is written to the
    real stdout afterwards, which keeps every existing ``print`` where it is
    instead of threading a formatter through several hundred lines.

    A command that fails before producing a payload still emits one. A
    caller parsing stdout should never have to distinguish "no JSON" from
    "JSON I could not read"; an error object with the exit code is
    unambiguous, and the human-readable reason is on stderr.

    That promise covers a raised exception as much as a non-zero return.
    An unreadable config or an unwritable output directory raises rather
    than returning, and letting it propagate would print a traceback and
    leave stdout empty — precisely the case this mode exists to remove.
    The traceback still goes to stderr, where the human debugging it looks;
    the caller parsing stdout gets the exception's type and message.
    """
    import json

    sink: dict[str, object] = {}
    args._json_sink = sink
    failure: str | None = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            code = command(args)
    except Exception as exc:  # noqa: BLE001 — the payload *is* the contract
        traceback.print_exc()
        failure = f"{type(exc).__name__}: {exc}"
        code = 1
    payload = sink.get("payload")
    if failure is not None or payload is None:
        # A run that raised reports the failure even if it had already
        # captured a payload: emitting that payload under a non-zero exit
        # code would describe a result the command did not finish producing.
        # A run that *returned* a code carries the reason it printed.
        payload = {
            "schema_version": SCHEMA_VERSION,
            "command": args.command,
            "error": (
                failure
                or sink.get("error")
                or "the command exited before producing a result"
            ),
            "exit_code": code,
        }
    print(json.dumps(payload, indent=2, default=str))
    return code


def _capture(args: argparse.Namespace, payload: dict[str, object]) -> None:
    """Hand a payload to :func:`_emit_json`, if this run is in JSON mode."""
    sink = getattr(args, "_json_sink", None)
    if sink is not None:
        sink["payload"] = payload


def _escape_what_the_stream_cannot_encode() -> None:
    """Write ``\\uXXXX`` for a character the output encoding lacks, not crash.

    The narration uses a few characters outside the legacy Windows code
    pages — α and δ in the method summaries, an arrow marking a binding
    bucket. A console renders them, but a *piped* run on Windows encodes
    with the ANSI code page (cp1252 on most Western machines), and there one
    arrow raised ``UnicodeEncodeError`` half-way through the report: exit 1,
    a traceback, and no workbook.

    ``backslashreplace`` keeps the encoding the environment chose rather than
    switching to UTF-8, so a consumer decoding with the code page still reads
    every other character correctly. The ``--json`` document is unaffected
    either way: ``json.dumps`` escapes non-ASCII itself.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments and dispatch to the requested subcommand.

    Args:
        argv: Arguments to parse, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code. ``0`` on success; ``1`` when the command ran and
        the answer is negative — unusable data from ``check``, an incomplete
        panel from ``ingest``, or, under ``--json``, a command that raised;
        ``2`` when the command could not run at all, which includes a config
        or price file that cannot be read and a mandate ``check`` finds
        impossible. See ``docs/ERRORS.md`` for the full contract.
    """
    _escape_what_the_stream_cannot_encode()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "optimize":
        if args.json:
            return _emit_json(args, _cmd_optimize)
        return _cmd_optimize(args)
    if args.command == "list-optimizers":
        return _cmd_list_optimizers()
    if args.command == "sample-data":
        return _cmd_sample_data(args)
    if args.command == "fred":
        return _cmd_fred(args)
    if args.command == "check":
        if args.json:
            return _emit_json(args, _cmd_check)
        return _cmd_check(args)
    if args.command == "backtest":
        if args.json:
            return _emit_json(args, _cmd_backtest)
        return _cmd_backtest(args)
    if args.command == "describe":
        if args.json:
            return _emit_json(args, _cmd_describe)
        return _cmd_describe(args)
    if args.command == "providers":
        return _cmd_providers(args)
    if args.command == "ingest":
        return _cmd_ingest(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
