"""End-to-end script: run an optimization, write Excel + plots.

Usage::

    python scripts/run_optimization.py --config config/example_multi_asset.yaml --sample
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.config import load_config  # noqa: E402
from optimization_engine.data.loader import (  # noqa: E402
    load_prices,
    prices_to_returns,
    sample_dataset,
)
from optimization_engine.data.quality import (  # noqa: E402
    align_panel,
    analyze_prices,
)
from optimization_engine.engine import run_engine  # noqa: E402
from optimization_engine.reporting.exporters import (  # noqa: E402
    run_sheets,
    write_excel_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prices", default=None)
    parser.add_argument("--sheet", default="Precios")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--output", default="outputs/optimization_report.xlsx")
    parser.add_argument("--frontier", action="store_true", default=True)
    parser.add_argument("--frontier-points", type=int, default=25)
    args = parser.parse_args()

    # One source, named. Falling back to the synthetic panel when --prices was
    # forgotten wrote a report on data that describes no market; see
    # `cli._data_source` for the same rule on the command line.
    if bool(args.sample) == bool(args.prices):
        print("Pass exactly one of --prices FILE or --sample.", file=sys.stderr)
        return 2
    config = load_config(args.config)
    if args.sample:
        prices = sample_dataset()
    else:
        prices = load_prices(args.prices, sheet_name=args.sheet)

    common = sorted(set(prices.columns) & set(config.expected_returns))
    if not common:
        print("No overlap between prices and config.expected_returns", file=sys.stderr)
        return 2
    prices = prices[common]
    config.expected_returns = {a: config.expected_returns[a] for a in common}

    quality = analyze_prices(prices, periods_per_year=config.periods_per_year)
    for issue in quality.issues:
        print(f"Data {issue.severity} - {issue.describe()}", file=sys.stderr)

    # `method="common"` matches `optengine` — see `cli._prepare_inputs` for
    # why that method and not one of the other two. The panel is aligned
    # *after* the quality report so the report still sees the gaps.
    prices, alignment = align_panel(prices, method="common")
    for action in alignment:
        print(f"  Alignment: {action}", file=sys.stderr)

    returns = prices_to_returns(prices)
    n_rows = len(returns)
    returns = returns.dropna(how="any")
    if len(returns) < n_rows:
        print(
            f"  Alignment: Dropped {n_rows - len(returns)} period(s) whose "
            "return could not be computed from the aligned prices.",
            file=sys.stderr,
        )
    if returns.empty:
        print("No usable returns after alignment.", file=sys.stderr)
        return 2

    run = run_engine(
        returns,
        config,
        build_frontier=args.frontier,
        n_frontier_points=args.frontier_points,
    )
    for warning in run.warnings:
        print(f"Warning - {warning}", file=sys.stderr)

    sheets = run_sheets(
        run,
        riskfree_rate=config.optimizer.risk_free_rate,
        data_quality=quality,
    )

    out = write_excel_report(args.output, sheets)
    print(f"Wrote {out}")
    print(f"Weights:\n{run.result.weights.round(4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
