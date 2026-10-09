"""Render the figures used by the README.

Kept in the repo so the images are reproducible rather than one-off artefacts:
every figure below is produced by the engine's own plotting code on the
built-in sample dataset, so what the README shows is what the library draws.

Static export goes through headless Chromium rather than kaleido — the browser
is already present for the UI tests, and this avoids adding an image-export
dependency to a library that otherwise needs none.

Usage::

    python scripts/render_docs_images.py [--output docs/images]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import plotly.io as pio  # noqa: E402

from optimization_engine import (  # noqa: E402
    BenchmarkSpec,
    EngineConfig,
    OptimizerSpec,
    prices_to_returns,
    run_engine,
    sample_dataset,
)
from optimization_engine.analytics.report import BENCHMARK, PORTFOLIO  # noqa: E402
from optimization_engine.data.covariance import (  # noqa: E402
    expected_returns_from_history,
)
from optimization_engine.reporting.plots import (  # noqa: E402
    plot_efficient_frontier,
    plot_frontier_uncertainty,
    plot_period_returns,
    plot_relative_wealth,
    plot_walk_forward_comparison,
    plot_weight_vs_risk,
)
from optimization_engine.resampling import bootstrap_frontier  # noqa: E402

#: Wide enough to read on GitHub without forcing a click-through.
WIDTH = 1040
SCALE = 2

GROUPS = {
    "US_Equity": "Equity", "Intl_Equity": "Equity", "EM_Equity": "Equity",
    "Real_Estate": "Alternatives", "Commodities": "Alternatives",
    "Infra": "Alternatives", "Gold": "Alternatives",
    "US_Treasuries": "FixedIncome", "TIPS": "FixedIncome",
    "IG_Credit": "FixedIncome", "HY_Credit": "FixedIncome",
    "EM_Debt": "FixedIncome", "Cash": "FixedIncome",
}


def shoot(fig, path: Path, width: int = WIDTH, height: int = 520) -> None:
    """Screenshot a Plotly figure through headless Chromium."""
    from playwright.sync_api import sync_playwright

    fig.update_layout(width=width, height=height)
    html = pio.to_html(fig, include_plotlyjs=True, full_html=True)
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as fh:
        fh.write(html)
        tmp = fh.name

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/opt/pw-browsers/chromium")
        page = browser.new_page(
            viewport={"width": width, "height": height},
            device_scale_factor=SCALE,
        )
        page.goto(f"file://{tmp}")
        page.wait_for_selector(".plot-container", timeout=30_000)
        page.wait_for_timeout(1200)
        page.locator(".plot-container").first.screenshot(path=str(path))
        browser.close()
    Path(tmp).unlink(missing_ok=True)
    print(f"  {path.name}  {path.stat().st_size // 1024} KB")


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    returns = prices_to_returns(sample_dataset(252 * 8))
    mu = expected_returns_from_history(
        returns, method="mean", periods_per_year=252
    ).to_dict()

    def config(name: str, **spec) -> EngineConfig:
        return EngineConfig(
            expected_returns=mu,
            bounds={a: [0.0, 0.35] for a in mu},
            groups=GROUPS,
            optimizer=OptimizerSpec(name=name, risk_free_rate=0.03, **spec),
        )

    print("efficient frontier...")
    # A conservative utility portfolio, so its marker does not land on top of
    # the tangency dot and the two read as the different portfolios they are.
    run = run_engine(
        returns, config("mean_variance", risk_aversion=14.0),
        build_frontier=True, n_frontier_points=25,
    )
    shoot(
        plot_efficient_frontier(
            run.frontier,
            title="Efficient frontier, with the portfolios worth marking",
            risk_free_rate=0.03,
            current_portfolio=(
                run.result.expected_volatility,
                run.result.expected_return,
                "Your portfolio",
            ),
        ),
        output / "frontier.png",
        height=560,
    )

    print("frontier uncertainty...")
    uncertainty = bootstrap_frontier(
        returns, config("mean_variance"), n_draws=60, n_points=14, seed=0
    )
    shoot(
        plot_frontier_uncertainty(
            uncertainty,
            title=f"The same frontier, resampled {uncertainty.n_draws} times",
        ),
        output / "frontier-uncertainty.png",
        height=520,
    )
    print(f"    band: {uncertainty.summary()}")

    print("capital vs risk...")
    rp = run_engine(returns, config("risk_parity"))
    mv = run_engine(returns, config("mean_variance", risk_aversion=4.0))
    decomposition = mv.risk_decomposition()
    shoot(
        plot_weight_vs_risk(
            decomposition[decomposition["weight"].abs() > 1e-4],
            title="Capital weight vs. share of risk (mean-variance)",
        ),
        output / "capital-vs-risk.png",
        height=470,
    )
    print(
        f"    mean-variance effective N {mv.diagnostics.effective_n:.1f} "
        f"(risk {mv.diagnostics.effective_n_risk:.1f}) | "
        f"risk parity {rp.diagnostics.effective_n:.1f} "
        f"(risk {rp.diagnostics.effective_n_risk:.1f})"
    )

    print("walk-forward...")
    chosen = run_engine(returns, config("max_sharpe"))
    # reestimate_expected_returns defaults to True: each window derives its own
    # mu. Reusing the config's full-sample vector would leak the future into
    # every window and roughly double the apparent out-of-sample Sharpe.
    wf = chosen.walk_forward(lookback=504, rebalance_every=63, transaction_cost_bps=15)
    # Cost and rebalancing matched on both sides, so the gap is overfitting
    # rather than a trading-cost artefact.
    in_sample = chosen.backtest(
        frequency="quarterly", transaction_cost_bps=15
    ).returns.reindex(wf.returns.index)
    shoot(
        plot_walk_forward_comparison(
            in_sample, wf.returns,
            title="What the backtest promised, and what it delivered",
        ),
        output / "walk-forward.png",
        height=470,
    )
    comparison = chosen.in_vs_out_of_sample(wf, riskfree_rate=0.03)
    print(
        "    Sharpe "
        f"{comparison.loc['Sharpe Ratio', 'In-sample (fitted)']:.2f} in-sample vs "
        f"{comparison.loc['Sharpe Ratio', 'Out-of-sample (walk-forward)']:.2f} "
        "walk-forward"
    )

    print("relative performance...")
    # A mandate written relative to an index: beat 1/N, at no more than 3% of
    # tracking error. The budget is what makes the two curves comparable.
    relative_config = config("mean_variance", risk_aversion=3.0)
    relative_config.benchmark = BenchmarkSpec(kind="equal_weight")
    relative_config.max_tracking_error = 0.03
    relative_run = run_engine(returns, relative_config)
    report = relative_run.performance(
        riskfree_rate=0.03, frequency="quarterly", transaction_cost_bps=15
    )
    shoot(
        plot_relative_wealth(
            report.returns[PORTFOLIO],
            report.returns[BENCHMARK],
            title=(
                "Ahead or behind: portfolio wealth relative to the benchmark's"
            ),
        ),
        output / "relative-performance.png",
        height=470,
    )
    shoot(
        plot_period_returns(
            report.periods,
            title="Year by year, against the benchmark it is measured on",
        ),
        output / "period-returns.png",
        height=470,
    )
    head = report.headline()
    print(
        f"    excess {head['excess_return']:.2%} at "
        f"{head['tracking_error']:.2%} tracking error "
        f"(IR {head['information_ratio']:.2f}, "
        f"active share {head['active_share']:.1%})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "docs" / "images"))
    args = parser.parse_args()
    build(Path(args.output))
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
