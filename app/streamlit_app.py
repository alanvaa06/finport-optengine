"""Streamlit UI for the optimization engine.

Run with::

    streamlit run app/streamlit_app.py

The app is driven by the same `EngineConfig` machinery as the CLI, so
anything that works here also works headless.

The flow is deliberately linear — load data, check it, choose a method, set
assumptions, state constraints, solve, then interrogate the result — and each
step surfaces what could make the next one wrong: data-quality problems before
estimation, constraint feasibility before solving, and the gap between the
in-sample and walk-forward track records before anyone believes the backtest.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make src/ and this directory importable when running
# ``streamlit run app/streamlit_app.py``.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
for path in (str(SRC), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from components import (  # noqa: E402
    ELIGIBILITY_STATES,
    align_scenario_table,
    describe_policy_cost,
    eligibility_state_counts,
    empty_scenario_table,
    empty_shock_table,
    format_table,
    metric_row,
    num,
    pct,
    plot_eligibility_heatmap,
    plot_universe_breadth,
    plot_universe_turnover,
    render_assumptions,
    render_benchmark,
    render_compliance,
    render_covariance_diagnostics,
    render_data_quality,
    render_feasibility,
    render_frontier_health,
    render_method_card,
    render_portfolio_diagnostics,
    render_projection_distance,
    render_stress_report,
    render_universe_notes,
    shock_dicts_from_tables,
    tables_from_shocks,
    thin_rows,
    unheld_shocked_assets,
    validated_shock_dicts,
)
from data_sources import (  # noqa: E402
    render_empty_state,
    render_ingest_panel,
    render_liquidity_selector,
    render_source_picker,
)
from layer_editor import (  # noqa: E402
    current_layers,
    render_layer_builder,
    render_layer_exposures,
    seed_layers_from_config,
)

from optimization_engine.analytics.performance import (  # noqa: E402
    annualize_returns,
    rolling_metrics,
    summary_stats,
)
from optimization_engine.analytics.relative import (  # noqa: E402
    active_share,
    summary_relative,
)
from optimization_engine.analytics.report import BENCHMARK, PORTFOLIO  # noqa: E402
from optimization_engine.analytics.risk import drawdown_table  # noqa: E402
from optimization_engine.backtest import (  # noqa: E402
    BacktestSpec,
    CostSpec,
    SpecValidationError,
    compute_tca,
    cost_by_asset,
    run_backtest,
)
from optimization_engine.benchmark import BenchmarkError, BenchmarkSpec  # noqa: E402
from optimization_engine.config import (  # noqa: E402
    EngineConfig,
    OptimizerSpec,
    expected_return_method_for_estimator,
)
from optimization_engine.data.covariance import (  # noqa: E402
    COVARIANCE_DESCRIPTIONS,
    EXPECTED_RETURN_DESCRIPTIONS,
    covariance_from_config,
    covariance_matrix,
    expected_returns_from_history,
)
from optimization_engine.data.fx import (  # noqa: E402
    FXError,
    convert_prices_to_base,
    supported_currencies,
)
from optimization_engine.data.loader import prices_to_returns  # noqa: E402
from optimization_engine.data.quality import align_panel, analyze_prices  # noqa: E402
from optimization_engine.engine import run_engine  # noqa: E402
from optimization_engine.optimizers.factory import (  # noqa: E402
    available_optimizers,
    constraints_from_config,
    effective_expected_returns,
)
from optimization_engine.optimizers.feasibility import analyze_feasibility  # noqa: E402
from optimization_engine.optimizers.requirements import requirements_for  # noqa: E402
from optimization_engine.reporting.exporters import (  # noqa: E402
    excel_writer,
    performance_sheets,
    run_sheets,
    unique_sheet_name,
)
from optimization_engine.reporting.plots import (  # noqa: E402
    plot_correlation_heatmap,
    plot_drawdown,
    plot_efficient_frontier,
    plot_frontier_uncertainty,
    plot_period_returns,
    plot_portfolio_composition,
    plot_relative_wealth,
    plot_return_distribution,
    plot_risk_contributions,
    plot_risk_return_scatter,
    plot_rolling_metrics,
    plot_rolling_relative,
    plot_walk_forward_comparison,
    plot_wealth_index,
    plot_weight_dispersion,
    plot_weight_evolution,
    plot_weight_vs_risk,
    plot_weights_bar,
)
from optimization_engine.resampling import bootstrap_frontier  # noqa: E402
from optimization_engine.scenarios import (  # noqa: E402
    NOTES_MAX_LEN,
    Scenario,
    config_signature,
    dump_scenarios_yaml,
    load_scenarios_yaml,
    now_iso,
    scenario_signature,
)
from optimization_engine.scenarios import (
    delete_scenario as _delete_scenario,
)
from optimization_engine.scenarios import (
    rename_scenario as _rename_scenario,
)
from optimization_engine.stress import (  # noqa: E402
    StressError,
    dump_shocks_yaml,
    load_shocks_yaml,
    shocks_from_dicts,
    stress_test,
)
from optimization_engine.ui_state import (  # noqa: E402
    derive_widget_state,
)
from optimization_engine.universe import (  # noqa: E402
    MASK_POLICIES,
    UniverseError,
    UniverseRules,
    count_unresolved,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Optimization Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .small-muted { color: #6b7280; font-size: 0.85rem; }
    .step-done { color: #16a34a; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_DEFAULT_STATE = {
    "scenarios": {},
    "active_scenario": None,
    "last_run_by_scenario": {},
    "scenario_load_warning": None,
    "last_run": None,
    "last_error": None,
    "walk_forward": None,
    "frontier_uncertainty": None,
    "ingest": None,
    # The stress grids exist from the first run because ``_build_config``
    # reads them: a config assembled before the Stress tab has rendered must
    # still be a config, and an absent key would make it one that quietly
    # carries no scenarios rather than one that carries none.
    "stress_shock_table": empty_shock_table(),
    "stress_scenario_table": empty_scenario_table(),
    "stress_unknown_assets": "raise",
    "universe_run_notes": None,
}
for key, default in _DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _adopt_shock_tables(shock_rows, scenario_rows) -> None:
    """Replace both stress grids wholesale — a file load, or a preset.

    Both go together or neither does: a covariance multiplier left over from
    the scenarios that were on the page a moment ago would attach itself to
    whichever new scenario happened to take the same name.
    """
    st.session_state.stress_shock_table = shock_rows
    st.session_state.stress_scenario_table = scenario_rows


def _seed_table_from_config(cfg) -> None:
    """Replace the editable assets / groups / FX tables with values from cfg."""
    assets = list(cfg.expected_returns.keys())
    st.session_state.config_table = pd.DataFrame(
        {
            "Expected Return": pd.Series(cfg.expected_returns).reindex(assets),
            "Min Weight": pd.Series(
                {a: float(cfg.bounds.get(a, [0.0, 1.0])[0]) for a in assets}
            ),
            "Max Weight": pd.Series(
                {a: float(cfg.bounds.get(a, [0.0, 1.0])[1]) for a in assets}
            ),
            "Group": pd.Series({a: str(cfg.groups.get(a, "Other")) for a in assets}),
            "Currency": pd.Series(
                {a: str(cfg.currencies.get(a, cfg.base_currency)) for a in assets}
            ),
        },
        index=assets,
    )
    if cfg.group_bounds:
        st.session_state.group_bounds = pd.DataFrame(
            {
                "Min Weight": [float(v[0]) for v in cfg.group_bounds.values()],
                "Max Weight": [float(v[1]) for v in cfg.group_bounds.values()],
            },
            index=list(cfg.group_bounds.keys()),
        )
    st.session_state.asset_currency = {
        a: str(cfg.currencies.get(a, cfg.base_currency)) for a in assets
    }
    # The saved mandate, not an empty builder: a scenario is only reproducible
    # if its layered limits come back with it.
    seed_layers_from_config(cfg)
    if getattr(cfg.optimizer, "risk_budget", None):
        st.session_state.risk_budget = pd.DataFrame(
            {"Risk Budget": pd.Series(cfg.optimizer.risk_budget)}
        )
    # The scenarios are part of the mandate, not a scratch pad: a config that
    # declares what a bad day does to the book has to reopen carrying them,
    # the same way its layered limits do.
    _adopt_shock_tables(*tables_from_shocks(cfg.stress))
    if getattr(cfg.optimizer, "bl_views", None) and isinstance(
        cfg.optimizer.bl_views, dict
    ):
        idx = sorted(
            set(cfg.optimizer.bl_views) | set(cfg.optimizer.bl_view_confidences or {})
        )
        st.session_state.bl_views = pd.DataFrame(
            {
                "View": pd.Series(cfg.optimizer.bl_views).reindex(idx),
                "Confidence (variance)": pd.Series(
                    cfg.optimizer.bl_view_confidences or {}
                ).reindex(idx),
            },
            index=idx,
        )


_pending = st.session_state.pop("pending_scenario_load", None)
if _pending and _pending in st.session_state.scenarios:
    _scn = st.session_state.scenarios[_pending]
    _cfg = _scn.config
    # Seed sidebar-widget keys *before* the widgets render this run.
    st.session_state["optimizer_name"] = _cfg.optimizer.name
    st.session_state["cov_method"] = _cfg.covariance_method
    st.session_state["ewma_lambda"] = float(_cfg.ewma_lambda)
    st.session_state["base_currency"] = _cfg.base_currency
    st.session_state["risk_free_rate"] = float(_cfg.optimizer.risk_free_rate)
    st.session_state["risk_aversion"] = float(_cfg.optimizer.risk_aversion)
    st.session_state["cvar_alpha"] = float(_cfg.optimizer.cvar_alpha)
    st.session_state["cdar_alpha"] = float(_cfg.optimizer.cdar_alpha)
    st.session_state["denoise"] = bool(_cfg.denoise)
    st.session_state["detone"] = int(_cfg.detone)
    st.session_state["cluster_linkage"] = str(_cfg.optimizer.cluster_linkage)
    st.session_state["auto_clusters"] = _cfg.optimizer.n_clusters is None
    st.session_state["n_clusters"] = _cfg.optimizer.n_clusters
    st.session_state["herc_risk_measure"] = str(_cfg.optimizer.herc_risk_measure)
    st.session_state["nco_objective"] = str(_cfg.optimizer.nco_objective)
    st.session_state["nco_detone"] = bool(_cfg.optimizer.nco_detone_for_clustering)
    st.session_state["accept_inaccurate"] = bool(_cfg.optimizer.accept_inaccurate)
    st.session_state["long_only"] = bool(_cfg.long_only)
    st.session_state["benchmark_kind"] = _cfg.benchmark.kind
    if _cfg.benchmark.asset:
        st.session_state["benchmark_asset"] = _cfg.benchmark.asset
    if _cfg.benchmark.series_name:
        st.session_state["benchmark_series"] = _cfg.benchmark.series_name
    st.session_state["benchmark_rebalance"] = _cfg.benchmark.rebalance
    if _cfg.benchmark.weights:
        st.session_state.benchmark_weights_table = pd.DataFrame(
            {"Weight": pd.Series(_cfg.benchmark.weights)}
        )
    st.session_state["use_te_limit"] = _cfg.max_tracking_error is not None
    if _cfg.max_tracking_error is not None:
        st.session_state["max_tracking_error"] = float(_cfg.max_tracking_error)
    st.session_state["use_as_limit"] = _cfg.max_active_share is not None
    if _cfg.max_active_share is not None:
        st.session_state["max_active_share"] = float(_cfg.max_active_share)
    if _cfg.optimizer.target_return is not None:
        st.session_state["mv_mode"] = "Target return"
        st.session_state["target_return"] = float(_cfg.optimizer.target_return)
    elif _cfg.optimizer.target_volatility is not None:
        st.session_state["mv_mode"] = "Target volatility"
        st.session_state["target_volatility"] = float(_cfg.optimizer.target_volatility)
    else:
        st.session_state["mv_mode"] = "Utility"
    _seed_table_from_config(_cfg)
    st.session_state.active_scenario = _pending


st.title("📈 Multi-Asset Optimization Engine")
st.caption(
    "Mean-variance · Risk parity · HRP · Black-Litterman · CVaR · Max diversification"
)


# ---------------------------------------------------------------------------
# Cached data helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=16)
def _frame_hash(df: pd.DataFrame) -> str:
    return pd.util.hash_pandas_object(df, index=True).values.tobytes().hex()


@st.cache_data(show_spinner=False, max_entries=16)
def _covariance_cached(
    returns_hash: str,
    method: str,
    ewma_lambda: float,
    periods_per_year: int,
    annualize: bool,
    _returns: pd.DataFrame,
) -> pd.DataFrame:
    return covariance_matrix(
        _returns, method=method, ewma_lambda=ewma_lambda,
        periods_per_year=periods_per_year, annualize=annualize,
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _historical_mu_cached(
    returns_hash: str,
    periods_per_year: int,
    _returns: pd.DataFrame,
) -> pd.Series:
    """Seed the config table's μ column from the library's own estimator.

    The formula used to be written out here as well, which meant the table
    the analyst edits could disagree with the μ the engine solved against.
    """
    return expected_returns_from_history(
        _returns, method="mean", periods_per_year=periods_per_year
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _quality_cached(prices_hash: str, periods_per_year: int, _prices: pd.DataFrame):
    return analyze_prices(_prices, periods_per_year=periods_per_year)


def _load_uploaded(file: io.BytesIO, sheet: str) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(file, sheet_name=sheet, index_col=0, parse_dates=True)
    if name.endswith(".csv"):
        return pd.read_csv(file, index_col=0, parse_dates=True)
    if name.endswith(".parquet"):
        return pd.read_parquet(file)
    raise ValueError(f"Unsupported file: {file.name}")


# ---------------------------------------------------------------------------
# Sidebar — step 1: data
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("1 · Data")
    data_source = st.radio(
        "Source",
        options=["Data provider", "Upload file"],
        index=0,
        horizontal=True,
        help=(
            "Providers are fetched live and return the same column names "
            "whichever one you pick. Upload is for a panel you already have."
        ),
    )

    if data_source == "Data provider":
        raw_prices, ingest_result = render_source_picker(st.session_state)
    else:
        uploaded = st.file_uploader(
            "Price file (Excel/CSV/Parquet)",
            type=["xlsx", "xls", "xlsm", "csv", "parquet"],
        )
        sheet = st.text_input("Sheet name (Excel)", value="Precios")
        # An uploaded panel carries no ingest provenance; the Data tab then
        # shows the quality report alone rather than an empty source block.
        ingest_result = None
        raw_prices = None
        if uploaded is not None:
            raw_prices = _load_uploaded(uploaded, sheet)
            raw_prices.index = pd.to_datetime(raw_prices.index)
            raw_prices = raw_prices.sort_index().dropna(how="all")
    st.session_state["ingest"] = ingest_result

# The guidance is drawn in the main area, where the data is about to appear,
# rather than in a narrow sidebar column beside it.
if raw_prices is None:
    render_empty_state(awaiting_upload=data_source == "Upload file")
    st.stop()

with st.sidebar:
    selected_assets = st.multiselect(
        "Universe (assets to include)",
        options=list(raw_prices.columns),
        default=list(raw_prices.columns),
        help=(
            "Anything you load but leave out here stays available as an "
            "external benchmark in step 6 — load an index alongside the "
            "universe and deselect it to compare against it."
        ),
    )
    if not selected_assets:
        st.warning("Select at least one asset.")
        st.stop()
    # Kept whole: the series left out of the universe are exactly the ones
    # that can serve as an external benchmark, and slicing them away here
    # would make that impossible without a second download.
    raw_prices_loaded = raw_prices
    raw_prices = raw_prices[selected_assets]

    st.caption(f"{raw_prices.shape[0]:,} rows × {raw_prices.shape[1]} assets loaded.")

    missing_alignment = st.selectbox(
        "Missing data",
        options=["common", "ffill", "drop_assets"],
        index=0,
        format_func=lambda m: {
            "common": "Use only dates where every asset is present",
            "ffill": "Carry the last price across short gaps, then align",
            "drop_assets": "Drop short series, then align",
        }[m],
        help=(
            "How to reconcile assets with different histories. Every choice "
            "changes the sample the covariance is estimated on, so it is "
            "made explicitly rather than silently."
        ),
    )
    max_ffill = (
        st.number_input(
            "Max gap to fill (periods)", min_value=1, max_value=60, value=5
        )
        if missing_alignment == "ffill"
        else 5
    )

# Alignment happens outside the sidebar so its log can be shown in the tabs.
prices, alignment_actions = align_panel(
    raw_prices, method=missing_alignment, max_ffill=int(max_ffill)
)
if prices.empty or prices.shape[1] == 0:
    st.error(
        "No usable data after alignment. The assets' histories may not "
        "overlap — try 'Carry the last price across short gaps' or shorten "
        "the universe."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — step 2: currency
# ---------------------------------------------------------------------------

with st.sidebar:
    st.divider()
    st.header("2 · Currency")
    currency_options = supported_currencies()
    base_currency = st.selectbox(
        "Base currency",
        options=currency_options,
        index=currency_options.index("USD"),
        key="base_currency",
        help=(
            "All asset prices are converted into this currency before "
            "computing returns. FX rates come from FRED."
        ),
    )

if "asset_currency" not in st.session_state or set(st.session_state.asset_currency) != set(prices.columns):
    st.session_state.asset_currency = {a: base_currency for a in prices.columns}

unique_currencies = {
    st.session_state.asset_currency.get(a, base_currency) for a in prices.columns
}
fx_note = None
if unique_currencies != {base_currency}:
    try:
        prices = convert_prices_to_base(
            prices,
            asset_currency=st.session_state.asset_currency,
            base=base_currency,
        )
        fx_note = (
            f"FX-converted {len(prices.columns)} series → {base_currency} via FRED."
        )
    except FXError as exc:
        st.sidebar.error(f"FX conversion failed: {exc}")
        st.stop()

returns = prices_to_returns(prices).dropna(how="any")
if returns.empty:
    st.error("The price panel produced no usable returns.")
    st.stop()

# Series loaded but left out of the universe: candidate external benchmarks.
_excluded = [c for c in raw_prices_loaded.columns if c not in selected_assets]
external_returns: pd.DataFrame | None = None
if _excluded:
    _external_prices = raw_prices_loaded[_excluded].sort_index().dropna(how="all")
    external_returns = prices_to_returns(_external_prices).dropna(how="all")
    if external_returns.empty:
        external_returns = None

# ---------------------------------------------------------------------------
# Sidebar — step 3: method
# ---------------------------------------------------------------------------

_METHOD_ORDER = [
    "mean_variance", "active_mean_variance", "min_variance", "max_sharpe",
    "risk_parity",
    "hrp", "black_litterman", "cvar", "max_diversification",
    "inverse_vol", "equal_weight",
]
_methods = [m for m in _METHOD_ORDER if m in available_optimizers()]
_methods += [m for m in available_optimizers() if m not in _methods]

with st.sidebar:
    st.divider()
    st.header("3 · Method")
    optimizer_name = st.selectbox(
        "Optimizer",
        options=_methods,
        index=_methods.index("mean_variance"),
        format_func=lambda m: requirements_for(m).display_name,
        key="optimizer_name",
        help="Ordered from most to least assumption-heavy.",
    )
    req = requirements_for(optimizer_name)
    st.caption(req.summary)
    ws = derive_widget_state(optimizer_name)

    st.divider()
    st.header("4 · Assumptions")
    risk_free_rate = st.number_input(
        "Risk-free rate (annual)",
        min_value=0.0, max_value=0.30,
        value=0.04, step=0.005, format="%.4f",
        key="risk_free_rate",
        disabled=not ws["risk_free_rate"]["enabled"],
        help=ws["risk_free_rate"]["tooltip"]
        or "Used for Sharpe, the tangency portfolio, and CAPM/Black-Litterman.",
    )
    periods_per_year = st.number_input(
        "Periods per year", min_value=1, max_value=365, value=252,
        key="periods_per_year",
        help="252 for daily data, 52 for weekly, 12 for monthly.",
    )
    cov_options = [
        "ledoit_wolf", "oas", "sample", "ewma", "semi", "shrink", "denoised"
    ]
    cov_method = st.selectbox(
        "Covariance estimator",
        options=cov_options,
        index=0,
        key="cov_method",
        disabled=not ws["cov_method"]["enabled"],
        help=ws["cov_method"]["tooltip"] or "Shrinkage estimators first.",
    )
    if ws["cov_method"]["enabled"]:
        st.caption(COVARIANCE_DESCRIPTIONS.get(cov_method, ""))
    denoise = st.checkbox(
        "Denoise (Marchenko-Pastur)",
        value=(cov_method == "denoised"),
        key="denoise",
        disabled=not ws["cov_method"]["enabled"] or cov_method == "denoised",
        help=(
            "Replace the eigenvalues that are indistinguishable from noise "
            "with their average, leaving the factor structure alone. Composes "
            "with any estimator."
        ),
    )
    detone = int(
        st.number_input(
            "Detone: eigenvectors to remove",
            min_value=0, max_value=3, value=0, step=1,
            key="detone",
            disabled=not ws["cov_method"]["enabled"],
            help=(
                "Strip the market component before clustering. Makes the "
                "covariance singular, so use it only with HRP, HERC or NCO."
            ),
        )
    )
    if detone and st.session_state.get("optimizer_name") not in ("hrp", "herc", "nco"):
        st.warning(
            "Detoning removes the market eigenvector, which leaves the "
            "covariance singular. Only the clustering methods (HRP, HERC, NCO) "
            "can use it — everything else inverts the matrix."
        )
    ewma_lambda = (
        st.slider(
            "EWMA λ", 0.80, 0.999, 0.94, 0.005,
            key="ewma_lambda",
            help="Effective sample is about 1/(1−λ) observations.",
        )
        if cov_method == "ewma" and ws["cov_method"]["enabled"]
        else 0.94
    )

    st.divider()
    st.header("5 · Objective")
    target_return: float | None = None
    target_volatility: float | None = None
    risk_aversion = 1.0
    cvar_alpha = 0.05
    risk_budget: dict[str, float] | None = None

    show_mode_radio = (
        optimizer_name != "cvar"
        and (ws["target_return"]["enabled"] or ws["target_volatility"]["enabled"])
        and ws["risk_aversion"]["enabled"]
    )
    if show_mode_radio:
        modes = []
        if ws["target_return"]["enabled"]:
            modes.append("Target return")
        if ws["target_volatility"]["enabled"]:
            modes.append("Target volatility")
        if ws["risk_aversion"]["enabled"]:
            modes.append("Utility")
        mode = st.radio("Mode", modes, horizontal=True, key="mv_mode")
        if mode == "Target return":
            target_return = st.number_input(
                "Target return (annual)", value=0.07, step=0.005, format="%.4f",
                key="target_return",
                help="Minimize risk subject to hitting exactly this return.",
            )
            if optimizer_name == "black_litterman":
                st.caption(
                    "Black-Litterman targets its **equilibrium posterior**, "
                    "which usually sits well below historical means. The "
                    "constraints tab reports the range this method can "
                    "actually reach."
                )
        elif mode == "Target volatility":
            target_volatility = st.number_input(
                "Target volatility (annual)", value=0.10, step=0.005, format="%.4f",
                key="target_volatility",
                help="Maximize return subject to staying under this volatility.",
            )
        else:
            risk_aversion = st.slider(
                "Risk aversion λ", 0.1, 20.0, 2.5, key="risk_aversion",
                help="Maximize μ'w − λ·w'Σw. Higher λ means a safer portfolio.",
            )
    elif not ws["risk_aversion"]["enabled"] and optimizer_name != "cvar":
        st.caption(
            f"{req.display_name} has no objective to tune — it is fully "
            "determined by the covariance and the constraints."
        )

    if optimizer_name == "cvar":
        cvar_alpha = st.slider(
            "CVaR tail probability α", 0.01, 0.20, 0.05, 0.01,
            key="cvar_alpha",
            help="0.05 ⇒ the 95% CVaR: the average of the worst 5% of periods.",
        )
        n_tail = int(cvar_alpha * len(returns))
        st.caption(
            f"About {n_tail} of {len(returns):,} observations drive the estimate."
            + ("  ⚠️ That is very few." if n_tail < 10 else "")
        )
        _tr = st.number_input(
            "Minimum return (optional, 0 = none)",
            value=0.0, step=0.005, format="%.4f", key="target_return_cvar",
        )
        target_return = None if _tr == 0.0 else _tr

    st.divider()
    st.header("6 · Benchmark")
    st.caption(
        "What the portfolio is measured against — and, once a limit is set, "
        "what it is optimized against."
    )
    _bench_kinds = ["none", "equal_weight", "single_asset", "custom_weights"]
    if external_returns is not None:
        _bench_kinds.append("external")
    if req.requires_benchmark:
        # The method is undefined without one, so "no benchmark" is not
        # offered rather than offered and then rejected at solve time.
        _bench_kinds = [k for k in _bench_kinds if k != "none"]
    _bench_labels = {
        "none": "No benchmark",
        "equal_weight": "Equal weight (1/N)",
        "single_asset": "Single asset",
        "custom_weights": "Custom weights",
        "external": "External series",
    }
    benchmark_kind = st.selectbox(
        "Benchmark",
        options=_bench_kinds,
        format_func=lambda k: _bench_labels[k],
        key="benchmark_kind",
        help=(
            "Absolute numbers say what happened; relative numbers say whether "
            "the process earned its fee. Everything downstream — the "
            "performance tab, the workbook, and the active-risk limits below "
            "— reads this one choice."
        ),
    )
    if req.requires_benchmark:
        st.caption(
            f"{req.display_name} measures return and risk against the "
            "benchmark, so one has to be chosen — 'no benchmark' is not an "
            "option for this method."
        )

    benchmark_asset: str | None = None
    benchmark_custom: dict[str, float] | None = None
    benchmark_series: str | None = None
    if benchmark_kind == "single_asset":
        benchmark_asset = st.selectbox(
            "Benchmark asset", options=list(returns.columns), key="benchmark_asset"
        )
    elif benchmark_kind == "custom_weights":
        if (
            "benchmark_weights_table" not in st.session_state
            or set(st.session_state.benchmark_weights_table.index)
            != set(returns.columns)
        ):
            st.session_state.benchmark_weights_table = pd.DataFrame(
                {"Weight": [1.0 / returns.shape[1]] * returns.shape[1]},
                index=list(returns.columns),
            )
        st.session_state.benchmark_weights_table = st.data_editor(
            st.session_state.benchmark_weights_table,
            num_rows="fixed",
            column_config={
                "Weight": st.column_config.NumberColumn(
                    min_value=-1.0, max_value=1.5, step=0.01, format="%.3f"
                ),
            },
            key="bench_editor",
        )
        benchmark_custom = {
            str(a): float(v)
            for a, v in st.session_state.benchmark_weights_table["Weight"].items()
        }
        _bench_total = sum(benchmark_custom.values())
        if abs(_bench_total) < 1e-9:
            st.error("Benchmark weights sum to zero — pick a different mix.")
        elif abs(_bench_total - 1.0) > 1e-6:
            st.caption(
                f"Weights sum to {_bench_total:.3f}; they are normalized to 1 "
                "so the comparison is against the same amount of money."
            )
    elif benchmark_kind == "external":
        benchmark_series = st.selectbox(
            "External series",
            options=list(external_returns.columns),
            key="benchmark_series",
            help=(
                "Loaded alongside the universe but excluded from it. It has "
                "no holdings here, so active share and the active-risk "
                "decomposition are unavailable — every return-based metric "
                "still is."
            ),
        )
        if unique_currencies != {base_currency}:
            st.caption(
                "⚠️ FX conversion is applied to the universe only. This series "
                f"is compared as quoted, not in {base_currency}."
            )

    benchmark_rebalance = "periodic"
    if benchmark_kind in ("equal_weight", "single_asset", "custom_weights"):
        benchmark_rebalance = st.radio(
            "Benchmark rebalancing",
            options=["periodic", "buy_and_hold"],
            format_func=lambda r: (
                "Rebalanced every period" if r == "periodic" else "Bought and held"
            ),
            horizontal=True,
            key="benchmark_rebalance",
            help=(
                "Published indices are rebalanced; an untouched policy "
                "portfolio drifts. Over a long sample the two are materially "
                "different track records."
            ),
        )

    max_tracking_error: float | None = None
    max_active_share: float | None = None
    if benchmark_kind != "none":
        _limits_enabled = ws["benchmark_limits"]["enabled"]
        if not _limits_enabled:
            st.caption(f"ℹ️ {ws['benchmark_limits']['tooltip']}")
        if st.checkbox(
            "Limit tracking error",
            value=False,
            key="use_te_limit",
            disabled=not _limits_enabled,
            help=(
                "Cap √((w−b)'Σ(w−b)) inside the solve. This is the constraint "
                "that turns a benchmark from a reporting choice into an "
                "optimization input."
            ),
        ):
            max_tracking_error = st.slider(
                "Max tracking error (annual)", 0.005, 0.20, 0.03, 0.005,
                format="%.3f", key="max_tracking_error",
            )
        if st.checkbox(
            "Limit active share",
            value=False,
            key="use_as_limit",
            disabled=not _limits_enabled,
            help=(
                "Cap ½·Σ|wᵢ−bᵢ|. Binds on positions, so it holds in a calm "
                "market where a tracking-error budget quietly permits a "
                "portfolio that shares almost nothing with its index."
            ),
        ):
            max_active_share = st.slider(
                "Max active share", 0.05, 1.0, 0.40, 0.05,
                format="%.2f", key="max_active_share",
            )

    st.divider()
    st.header("7 · Exposure")
    long_only = st.checkbox(
        "Long only", value=True, key="long_only",
        help="Off allows short positions, subject to each asset's minimum weight.",
    )
    leverage_cap: float | None = None
    if not long_only:
        leverage_cap = st.slider(
            "Gross-exposure cap (Σ|w|)", 1.0, 3.0, 1.5, 0.1,
            help="1.5 means 125% long against 25% short, or any equivalent mix.",
        )
    use_turnover = st.checkbox(
        "Turnover budget",
        value=False,
        help=(
            "Limit how much of the book may change versus a previous "
            "allocation. Honoured by mean-variance and CVaR; other methods "
            "warn instead of silently ignoring it."
        ),
    )
    turnover_limit: float | None = None
    if use_turnover:
        turnover_limit = st.slider(
            "Max one-way turnover", 0.01, 2.0, 0.20, 0.01,
            help="0.20 means at most 20% of the portfolio changes hands.",
        )
        if not req.supports_turnover:
            st.warning(
                f"{req.display_name} cannot enforce a turnover budget. "
                "Use mean-variance or CVaR to make it bind."
            )

    st.divider()
    st.header("8 · Frontier")
    build_frontier = st.checkbox(
        "Build efficient frontier",
        value=True,
        disabled=not ws["frontier"]["enabled"],
        help=ws["frontier"]["tooltip"],
    )
    n_frontier_points = st.slider(
        "Frontier points", 5, 100, 25,
        disabled=not ws["frontier"]["enabled"],
    )
    if not ws["frontier"]["enabled"]:
        build_frontier = False

    st.divider()
    with st.expander("9 · Advanced", expanded=False):
        accept_inaccurate = st.checkbox(
            "Accept approximate solutions",
            value=False,
            key="accept_inaccurate",
            help=(
                "Off: a solve that no solver can verify fails, and says so. "
                "On: the approximate weights are used and flagged in the "
                "compliance banner. Turn it on only when an indicative book "
                "is more use than none — the constraints it reports as "
                "satisfied are satisfied to the solver's own loose tolerance, "
                "not to yours."
            ),
        )
        st.caption(
            "Has no effect on HRP, HERC or the naive weightings: they never "
            "call a solver."
        )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

st.markdown("---")
(
    tab_overview,
    tab_assets,
    tab_constraints,
    tab_optimize,
    tab_stress,
    tab_backtest,
    tab_universe,
    tab_performance,
    tab_compare,
    tab_whatif,
    tab_report,
) = st.tabs(
    [
        "🌐 Data",
        "📊 Assets",
        "⚙️ Assumptions & constraints",
        "🚀 Optimize",
        "💥 Stress",
        "📉 Backtest",
        "🌍 Universe",
        "🎯 Performance",
        "🆚 Compare",
        "🎚️ What-if",
        "📤 Report",
    ]
)

if st.session_state.scenario_load_warning:
    st.warning(st.session_state.scenario_load_warning)
    st.session_state.scenario_load_warning = None


quality = _quality_cached(_frame_hash(raw_prices), int(periods_per_year), raw_prices)


# ---------------------------------------------------------------------------
# 🌐 Data
# ---------------------------------------------------------------------------

with tab_overview:
    metric_row(
        [
            ("Assets", str(returns.shape[1]), None),
            ("Periods", f"{returns.shape[0]:,}", None),
            ("Start", str(returns.index.min().date()), None),
            ("End", str(returns.index.max().date()), None),
        ]
    )

    render_ingest_panel(ingest_result)

    render_data_quality(quality)
    if alignment_actions:
        with st.expander("What alignment did to the panel", expanded=False):
            for action in alignment_actions:
                st.markdown(f"- {action}")
    if fx_note:
        st.caption(fx_note)

    st.subheader("Cumulative returns")
    st.plotly_chart(plot_wealth_index(returns), width="stretch")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Correlation")
        st.plotly_chart(
            plot_correlation_heatmap(returns.corr()), width="stretch"
        )
    with right:
        st.subheader("Coverage")
        st.caption(
            "Observations available per asset, before alignment. Wide "
            "differences mean the common sample is much shorter than the "
            "longest series."
        )
        coverage = quality.per_asset[
            ["observations", "first_date", "last_date", "annualized_vol"]
        ].copy()
        coverage["annualized_vol"] = coverage["annualized_vol"].map(
            lambda v: f"{v:.1%}" if pd.notna(v) else "—"
        )
        st.dataframe(coverage, width="stretch")


# ---------------------------------------------------------------------------
# 📊 Assets
# ---------------------------------------------------------------------------

with tab_assets:
    st.subheader("Per-asset statistics")
    extended = st.toggle(
        "Show extended metrics",
        value=False,
        help=(
            "Adds Calmar, Omega, tail ratio, hit rate, the Ulcer index, "
            "drawdown duration, and the probabilistic Sharpe ratio."
        ),
    )
    stats = summary_stats(
        returns,
        periods_per_year=int(periods_per_year),
        riskfree_rate=float(risk_free_rate),
        extended=extended,
    )
    st.dataframe(format_table(stats), width="stretch")
    if extended:
        st.caption(
            "**Prob. Sharpe > 0** is the probability the true Sharpe ratio is "
            "positive, given the sample's length, skew and kurtosis. Below "
            "~95% the headline Sharpe is not statistically distinguishable "
            "from zero."
        )

    st.subheader("Drawdowns")
    sel = st.multiselect(
        "Series to plot",
        options=list(returns.columns),
        default=list(returns.columns[:3]),
    )
    if sel:
        st.plotly_chart(plot_drawdown(returns[sel]), width="stretch")
        focus = st.selectbox("Drawdown episodes for", options=sel, index=0)
        episodes = drawdown_table(returns[focus], top=5)
        if episodes.empty:
            st.caption("No drawdown episodes in this sample.")
        else:
            display = episodes.copy()
            display["max_drawdown"] = display["max_drawdown"].map("{:.2%}".format)
            st.dataframe(
                display[
                    [
                        "peak_date", "trough_date", "recovery_date",
                        "max_drawdown", "decline_periods", "recovery_periods",
                        "recovered",
                    ]
                ],
                width="stretch", hide_index=True,
            )
            st.caption(
                "Depth is only half the story — how long a portfolio stayed "
                "underwater is what decides whether it was actually held."
            )


# ---------------------------------------------------------------------------
# ⚙️ Assumptions & constraints
# ---------------------------------------------------------------------------

with tab_constraints:
    render_method_card(req)
    st.divider()

    historical_mu = _historical_mu_cached(
        _frame_hash(returns), int(periods_per_year), returns,
    )

    if "config_table" not in st.session_state or set(st.session_state.config_table.index) != set(returns.columns):
        st.session_state.config_table = pd.DataFrame(
            {
                "Expected Return": historical_mu.round(4),
                "Min Weight": 0.0,
                "Max Weight": 1.0,
                "Group": "Other",
                "Currency": [
                    st.session_state.asset_currency.get(a, base_currency)
                    for a in returns.columns
                ],
            },
            index=returns.columns,
        )
    elif "Currency" not in st.session_state.config_table.columns:
        st.session_state.config_table["Currency"] = [
            st.session_state.asset_currency.get(a, base_currency)
            for a in returns.columns
        ]

    if ws["expected_returns_method"]["enabled"]:
        st.markdown("**Expected returns**")
        er_options = [
            "historical_mean", "geometric_mean", "shrunk_mean", "ema", "capm",
        ]
        er_method = st.radio(
            "Method",
            options=er_options,
            index=er_options.index(
                st.session_state.get("expected_returns_method", "historical_mean")
            )
            if st.session_state.get("expected_returns_method", "historical_mean")
            in er_options
            else 0,
            horizontal=True,
            key="expected_returns_method",
        )
        st.caption(
            EXPECTED_RETURN_DESCRIPTIONS.get(
                expected_return_method_for_estimator(er_method), ""
            )
        )
        if er_method == "ema":
            st.session_state.ema_span = st.slider(
                "EMA span (periods)",
                min_value=30, max_value=504,
                value=int(st.session_state.get("ema_span", 180)),
                step=10,
                key="ema_span_slider",
            )
        elif er_method == "capm":
            st.session_state.market_return = st.number_input(
                "Market return (annual, optional)",
                value=float(st.session_state.get("market_return") or 0.08),
                step=0.005, format="%.4f",
                key="market_return_input",
            )
            mw_idx = list(returns.columns)
            if (
                "market_weights_table" not in st.session_state
                or set(st.session_state.market_weights_table.index) != set(mw_idx)
            ):
                st.session_state.market_weights_table = pd.DataFrame(
                    {"Market weight": [1.0 / len(mw_idx)] * len(mw_idx)}, index=mw_idx
                )
            st.session_state.market_weights_table = st.data_editor(
                st.session_state.market_weights_table,
                num_rows="fixed",
                column_config={
                    "Market weight": st.column_config.NumberColumn(
                        min_value=0.0, max_value=1.0, step=0.01, format="%.3f",
                    ),
                },
            )

        if st.button("Reset μ to method default", key="reset_mu_btn"):
            mw_for_capm = (
                pd.Series(st.session_state.market_weights_table["Market weight"])
                if er_method == "capm"
                and "market_weights_table" in st.session_state
                else None
            )
            try:
                seeded = expected_returns_from_history(
                    returns,
                    method=expected_return_method_for_estimator(er_method),
                    periods_per_year=int(periods_per_year),
                    span=int(st.session_state.get("ema_span", 180)),
                    market_return=float(st.session_state.get("market_return") or 0.0) or None,
                    risk_free_rate=float(risk_free_rate),
                    market_weights=mw_for_capm,
                    cov_matrix=_covariance_cached(
                        _frame_hash(returns), cov_method, float(ewma_lambda),
                        int(periods_per_year), True, returns,
                    ),
                )
                st.session_state.config_table["Expected Return"] = seeded.round(4)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    st.markdown("**Per-asset expected returns and weight bounds**")
    st.caption(
        "Edit any cell. Min/Max are weight bounds; Group drives the "
        "asset-class constraints below."
    )
    if ws["soft_bounds_caption"]["enabled"]:
        st.info(req.bounds_note)

    edited = st.data_editor(
        st.session_state.config_table,
        width="stretch",
        num_rows="fixed",
        column_config={
            "Expected Return": st.column_config.NumberColumn(
                format="%.4f",
                disabled=not ws["expected_returns_column"]["enabled"],
                help=ws["expected_returns_column"]["tooltip"],
            ),
            "Min Weight": st.column_config.NumberColumn(
                min_value=-1.0, max_value=1.0, step=0.01, format="%.2f"
            ),
            "Max Weight": st.column_config.NumberColumn(
                min_value=0.0, max_value=1.5, step=0.01, format="%.2f"
            ),
            "Group": st.column_config.TextColumn(),
            "Currency": st.column_config.SelectboxColumn(
                "Currency",
                options=supported_currencies(),
                help="ISO code of the currency the price series is quoted in.",
            ),
        },
    )
    st.session_state.config_table = edited
    st.session_state.asset_currency = {
        a: str(edited.loc[a, "Currency"]) for a in returns.columns
    }

    asset_groups = {a: str(edited.loc[a, "Group"]) for a in returns.columns}

    if ws["group_bounds"]["enabled"]:
        st.divider()
        st.markdown("**Layer 1 · Asset-class budgets**")
        st.caption(
            "One row per value in the Group column above. Leave a row at "
            "0.00–1.00 to leave that asset class uncapped."
        )
        unique_groups = sorted(edited["Group"].dropna().unique().tolist())
        if unique_groups:
            gb_default = pd.DataFrame(
                {"Min Weight": 0.0, "Max Weight": 1.0}, index=unique_groups
            )
            if "group_bounds" not in st.session_state or list(st.session_state.group_bounds.index) != unique_groups:
                st.session_state.group_bounds = gb_default
            st.session_state.group_bounds = st.data_editor(
                st.session_state.group_bounds,
                width="stretch",
                num_rows="fixed",
                column_config={
                    "Min Weight": st.column_config.NumberColumn(
                        min_value=0.0, max_value=1.5, step=0.01, format="%.2f"
                    ),
                    "Max Weight": st.column_config.NumberColumn(
                        min_value=0.0, max_value=1.5, step=0.01, format="%.2f"
                    ),
                },
            )

        base_limits = {}
        if "group_bounds" in st.session_state:
            base_limits = {
                str(g): (float(row["Min Weight"]), float(row["Max Weight"]))
                for g, row in st.session_state.group_bounds.iterrows()
            }
        st.divider()
        _last_run = st.session_state.get("last_run")
        render_layer_builder(
            list(returns.columns),
            groups=asset_groups,
            currencies=dict(st.session_state.asset_currency),
            base_currency=base_currency,
            base_layer_limits=base_limits,
            # The previous solve, so each limit sits next to where the book
            # actually landed. That turns the builder into a loop instead of
            # a form filled in blind.
            current_weights=(
                None if _last_run is None else _last_run.result.weights
            ),
        )
    else:
        st.caption(
            f"_{req.display_name} does not enforce group or layered bucket "
            "budgets — the editors are hidden._"
        )

    if optimizer_name == "risk_parity":
        st.markdown("**Risk budgets** (each asset's target share of total risk)")
        if "risk_budget" not in st.session_state or set(st.session_state.risk_budget.index) != set(returns.columns):
            st.session_state.risk_budget = pd.DataFrame(
                {"Risk Budget": 1.0 / len(returns.columns)}, index=returns.columns
            )
        st.session_state.risk_budget = st.data_editor(
            st.session_state.risk_budget,
            width="stretch",
            num_rows="fixed",
            column_config={
                "Risk Budget": st.column_config.NumberColumn(
                    min_value=0.001, max_value=1.0, step=0.01, format="%.3f",
                    help="Must be strictly positive for every asset.",
                ),
            },
        )
        budget_total = float(st.session_state.risk_budget["Risk Budget"].sum())
        if abs(budget_total - 1.0) > 1e-6:
            st.caption(
                f"Budgets sum to {budget_total:.3f}; they will be renormalized to 1."
            )
        risk_budget = st.session_state.risk_budget["Risk Budget"].to_dict()

    elif optimizer_name == "black_litterman":
        st.markdown("**Black-Litterman views**")
        st.caption(
            "An **absolute** view names the return you expect from one asset. "
            "A **relative** view names a spread — 'A beats B by 2%' — and "
            "leaves the overall market level alone, which is usually how "
            "opinions are actually held."
        )
        view_kind = st.radio(
            "View type", ["Absolute", "Relative (spread)"], horizontal=True,
            key="bl_view_kind",
        )
        if view_kind == "Absolute":
            if "bl_views" not in st.session_state or set(st.session_state.bl_views.index) != set(returns.columns):
                st.session_state.bl_views = pd.DataFrame(
                    {"View": np.nan, "Confidence (variance)": np.nan},
                    index=returns.columns,
                )
            st.session_state.bl_views = st.data_editor(
                st.session_state.bl_views,
                width="stretch",
                num_rows="fixed",
                column_config={
                    "View": st.column_config.NumberColumn(
                        format="%.4f", help="Annualized expected return."
                    ),
                    "Confidence (variance)": st.column_config.NumberColumn(
                        format="%.6f",
                        help="Variance of the view's error. Blank uses the "
                        "He-Litterman default; smaller means more confident.",
                    ),
                },
            )
        else:
            if "bl_relative_views" not in st.session_state:
                st.session_state.bl_relative_views = pd.DataFrame(
                    {
                        "Outperforms": [returns.columns[0]],
                        "Underperforms": [returns.columns[-1]],
                        "By (annual)": [0.02],
                        "Confidence (variance)": [np.nan],
                    }
                )
            st.session_state.bl_relative_views = st.data_editor(
                st.session_state.bl_relative_views,
                width="stretch",
                num_rows="dynamic",
                column_config={
                    "Outperforms": st.column_config.SelectboxColumn(
                        options=list(returns.columns)
                    ),
                    "Underperforms": st.column_config.SelectboxColumn(
                        options=list(returns.columns)
                    ),
                    "By (annual)": st.column_config.NumberColumn(format="%.4f"),
                    "Confidence (variance)": st.column_config.NumberColumn(
                        format="%.6f"
                    ),
                },
            )

        st.markdown("**Equilibrium prior**")
        c1, c2 = st.columns(2)
        with c1:
            st.session_state["bl_tau"] = st.slider(
                "τ (prior uncertainty)", 0.01, 0.5,
                float(st.session_state.get("bl_tau", 0.05)), 0.01,
                key="bl_tau_slider",
                help="Scales the prior covariance. Smaller τ means a firmer prior.",
            )
        with c2:
            st.session_state["bl_calibrate"] = st.checkbox(
                "Imply δ from a market return",
                value=bool(st.session_state.get("bl_calibrate", False)),
                help=(
                    "Back the risk-aversion coefficient out of the market's "
                    "own Sharpe ratio rather than guessing 2.5."
                ),
            )
            if st.session_state["bl_calibrate"]:
                st.session_state["bl_market_return"] = st.number_input(
                    "Market return (annual)",
                    value=float(st.session_state.get("bl_market_return", 0.07)),
                    step=0.005, format="%.4f",
                )
        if (
            "bl_market_caps_table" not in st.session_state
            or set(st.session_state.bl_market_caps_table.index) != set(returns.columns)
        ):
            st.session_state.bl_market_caps_table = pd.DataFrame(
                {"Market cap weight": [1.0 / len(returns.columns)] * len(returns.columns)},
                index=returns.columns,
            )
        st.session_state.bl_market_caps_table = st.data_editor(
            st.session_state.bl_market_caps_table,
            num_rows="fixed",
            column_config={
                "Market cap weight": st.column_config.NumberColumn(
                    min_value=0.0, max_value=1.0, step=0.01, format="%.3f",
                    help="Equal weights → equilibrium under no views.",
                ),
            },
        )

    elif optimizer_name == "hrp":
        st.session_state["hrp_linkage"] = st.selectbox(
            "HRP linkage method",
            options=["single", "average", "complete", "ward"],
            index=["single", "average", "complete", "ward"].index(
                st.session_state.get("hrp_linkage", "single")
            ),
            key="hrp_linkage_select",
            help="Hierarchical clustering linkage rule.",
        )

    elif optimizer_name in ("herc", "nco"):
        linkages = ["ward", "average", "complete", "single"]
        st.session_state["cluster_linkage"] = st.selectbox(
            "Linkage method",
            options=linkages,
            index=linkages.index(st.session_state.get("cluster_linkage", "ward")),
            key="cluster_linkage_select",
            help=(
                "These methods partition the tree rather than merely order it, "
                "so single linkage's chaining tends to produce one dominant "
                "cluster."
            ),
        )
        auto_k = st.checkbox(
            "Choose the number of clusters automatically",
            value=st.session_state.get("auto_clusters", True),
            key="auto_clusters",
            help=(
                "Maximizes the silhouette t-statistic — the criterion behind "
                "López de Prado's ONC."
            ),
        )
        st.session_state["n_clusters"] = (
            None
            if auto_k
            else int(
                st.number_input(
                    "Number of clusters", min_value=2,
                    max_value=max(2, len(returns.columns) - 1),
                    value=min(4, max(2, len(returns.columns) - 1)), step=1,
                    key="n_clusters_input",
                )
            )
        )
        if optimizer_name == "herc":
            measures = ["variance", "std", "cvar", "cdar", "equal_weight"]
            st.session_state["herc_risk_measure"] = st.selectbox(
                "Cluster risk measure",
                options=measures,
                index=measures.index(
                    st.session_state.get("herc_risk_measure", "variance")
                ),
                key="herc_risk_measure_select",
                help=(
                    "How the budget is split between two sibling branches. "
                    "CVaR and CDaR let downside risk drive the split."
                ),
            )
        else:
            objectives = ["min_variance", "max_sharpe"]
            st.session_state["nco_objective"] = st.selectbox(
                "Objective at both layers",
                options=objectives,
                index=objectives.index(
                    st.session_state.get("nco_objective", "min_variance")
                ),
                key="nco_objective_select",
                help="Solved inside each cluster and again across clusters.",
            )
            st.session_state["nco_detone"] = st.checkbox(
                "Detone before clustering",
                value=bool(st.session_state.get("nco_detone", True)),
                key="nco_detone_check",
                help=(
                    "Remove the market eigenvector from the distance metric. "
                    "Without it every pair looks alike and the partition "
                    "degenerates."
                ),
            )

    elif optimizer_name == "cdar":
        st.session_state["cdar_alpha"] = float(
            st.slider(
                "CDaR tail probability α", 0.01, 1.0,
                float(st.session_state.get("cdar_alpha", 0.05)), 0.01,
                key="cdar_alpha_slider",
                help=(
                    "0.05 averages the worst 5% of the drawdown path; 1.0 is "
                    "the average drawdown."
                ),
            )
        )
        st.caption(
            "Drawdown is a path statistic: reorder the same returns and this "
            "objective changes. Check the walk-forward before believing it."
        )

    if use_turnover:
        st.markdown("**Previous allocation** (what the turnover budget trades from)")
        if (
            "previous_weights_table" not in st.session_state
            or set(st.session_state.previous_weights_table.index) != set(returns.columns)
        ):
            st.session_state.previous_weights_table = pd.DataFrame(
                {"Previous weight": [1.0 / len(returns.columns)] * len(returns.columns)},
                index=returns.columns,
            )
        st.session_state.previous_weights_table = st.data_editor(
            st.session_state.previous_weights_table,
            width="stretch",
            num_rows="fixed",
            column_config={
                "Previous weight": st.column_config.NumberColumn(
                    min_value=-1.0, max_value=1.5, step=0.01, format="%.3f"
                ),
            },
        )


def _current_shock_dicts() -> tuple[list[dict], list[str]]:
    """The Stress tab's grids as scenario mappings, and what would not parse.

    ``align_scenario_table`` is re-run here rather than trusted from session
    state because the per-scenario grid is rebuilt one rerun *behind* the leg
    grid: a scenario renamed and immediately saved would otherwise carry the
    old name's covariance multiplier into the preset.
    """
    rows = st.session_state.get("stress_shock_table")
    if rows is None or rows.empty:
        return [], []
    meta = align_scenario_table(rows, st.session_state.get("stress_scenario_table"))
    return validated_shock_dicts(shock_dicts_from_tables(rows, meta))


def _build_config() -> EngineConfig:
    """Assemble the EngineConfig from every widget and editable table."""
    table = st.session_state.config_table
    bounds = {
        a: [float(table.loc[a, "Min Weight"]), float(table.loc[a, "Max Weight"])]
        for a in returns.columns
    }
    groups = {a: str(table.loc[a, "Group"]) for a in returns.columns}
    group_bounds: dict[str, list[float]] = {}
    if "group_bounds" in st.session_state:
        for g, row in st.session_state.group_bounds.iterrows():
            group_bounds[str(g)] = [float(row["Min Weight"]), float(row["Max Weight"])]

    expected_returns = {
        a: float(table.loc[a, "Expected Return"]) for a in returns.columns
    }

    spec = OptimizerSpec(
        name=optimizer_name,
        target_return=target_return,
        target_volatility=target_volatility,
        risk_free_rate=float(risk_free_rate),
        risk_aversion=float(risk_aversion),
        cvar_alpha=float(cvar_alpha),
        bl_tau=float(st.session_state.get("bl_tau", 0.05)),
        hrp_linkage=str(st.session_state.get("hrp_linkage", "single")),
        cdar_alpha=float(st.session_state.get("cdar_alpha", 0.05)),
        cluster_linkage=str(st.session_state.get("cluster_linkage", "ward")),
        n_clusters=st.session_state.get("n_clusters"),
        herc_risk_measure=str(
            st.session_state.get("herc_risk_measure", "variance")
        ),
        nco_objective=str(st.session_state.get("nco_objective", "min_variance")),
        nco_detone_for_clustering=bool(st.session_state.get("nco_detone", True)),
        accept_inaccurate=bool(accept_inaccurate),
    )

    if optimizer_name == "risk_parity" and "risk_budget" in st.session_state:
        spec.risk_budget = st.session_state.risk_budget["Risk Budget"].to_dict()

    if optimizer_name == "black_litterman":
        kind = st.session_state.get("bl_view_kind", "Absolute")
        if kind == "Absolute" and "bl_views" in st.session_state:
            v = st.session_state.bl_views
            spec.bl_views = {
                a: float(v.loc[a, "View"])
                for a in v.index
                if pd.notna(v.loc[a, "View"])
            }
            spec.bl_view_confidences = {
                a: float(v.loc[a, "Confidence (variance)"])
                for a in v.index
                if pd.notna(v.loc[a, "Confidence (variance)"])
            }
        elif "bl_relative_views" in st.session_state:
            rows = []
            for _, row in st.session_state.bl_relative_views.iterrows():
                over, under = row.get("Outperforms"), row.get("Underperforms")
                spread = row.get("By (annual)")
                if pd.isna(over) or pd.isna(under) or pd.isna(spread):
                    continue
                if over == under:
                    continue
                entry = {
                    "weights": {str(over): 1.0, str(under): -1.0},
                    "expected_return": float(spread),
                    "label": f"{over} > {under}",
                }
                conf = row.get("Confidence (variance)")
                if pd.notna(conf):
                    entry["confidence"] = float(conf)
                rows.append(entry)
            spec.bl_views = rows or None
        if "bl_market_caps_table" in st.session_state:
            spec.bl_market_caps = (
                st.session_state.bl_market_caps_table["Market cap weight"].to_dict()
            )
        if st.session_state.get("bl_calibrate"):
            spec.bl_calibrate_risk_aversion = True
            spec.bl_market_return = float(st.session_state.get("bl_market_return", 0.07))

    market_weights = None
    if (
        st.session_state.get("expected_returns_method") == "capm"
        and "market_weights_table" in st.session_state
    ):
        market_weights = (
            st.session_state.market_weights_table["Market weight"].to_dict()
        )

    previous_weights = None
    if use_turnover and "previous_weights_table" in st.session_state:
        previous_weights = (
            st.session_state.previous_weights_table["Previous weight"].to_dict()
        )

    benchmark_spec = BenchmarkSpec(
        kind=benchmark_kind,
        asset=benchmark_asset,
        weights=benchmark_custom,
        series_name=benchmark_series,
        rebalance=benchmark_rebalance,
    )

    return EngineConfig(
        expected_returns=expected_returns,
        bounds=bounds,
        groups=groups,
        group_bounds=group_bounds,
        constraint_layers=current_layers(list(returns.columns)),
        currencies=dict(st.session_state.asset_currency),
        base_currency=base_currency,
        periods_per_year=int(periods_per_year),
        covariance_method=cov_method,
        ewma_lambda=float(ewma_lambda),
        denoise=bool(denoise),
        detone=int(detone),
        expected_returns_method=str(
            st.session_state.get("expected_returns_method", "historical_mean")
        ),
        ema_span=int(st.session_state.get("ema_span", 180)),
        market_return=(
            float(st.session_state.get("market_return"))
            if st.session_state.get("expected_returns_method") == "capm"
            and st.session_state.get("market_return")
            else None
        ),
        market_weights=market_weights,
        optimizer=spec,
        benchmark=benchmark_spec,
        max_tracking_error=max_tracking_error,
        max_active_share=max_active_share,
        long_only=bool(long_only),
        leverage=leverage_cap,
        previous_weights=previous_weights,
        turnover_limit=turnover_limit,
        # Only the scenarios that are scenarios. What did not parse is named
        # on the Stress tab rather than dropped in silence.
        stress=_current_shock_dicts()[0],
    )


@st.cache_data(show_spinner=False, max_entries=16)
def _feasibility_cached(signature: str, returns_hash: str, _returns: pd.DataFrame):
    cfg = EngineConfig.from_dict(json.loads(signature))
    cov = covariance_from_config(_returns, cfg)
    mu = pd.Series(cfg.expected_returns).reindex(_returns.columns).fillna(0.0)
    return analyze_feasibility(
        list(_returns.columns),
        constraints_from_config(cfg, list(_returns.columns)),
        expected_returns=effective_expected_returns(cfg, cov, mu),
        cov_matrix=cov,
    )


# The live feasibility panel closes the loop on the constraints tab: the
# analyst sees whether what they just typed can be solved, before solving.
with tab_constraints:
    st.divider()
    st.markdown("**Feasibility check**")
    try:
        live_config = _build_config()
        live_feasibility = _feasibility_cached(
            config_signature(live_config), _frame_hash(returns), returns
        )
        render_feasibility(live_feasibility)
    except Exception as exc:  # a half-edited table should not crash the tab
        live_feasibility = None
        st.info(f"Feasibility check unavailable: {exc}")


# ---------------------------------------------------------------------------
# Cached scenario solver (shared by Compare & What-if)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=32)
def _solve_scenario_cached(
    signature: str, returns_df: pd.DataFrame, _external: pd.DataFrame | None = None
):
    """Solve a config given its JSON signature; cached on (signature, returns).

    ``_external`` is leading-underscored so Streamlit does not try to hash it;
    the benchmark it feeds is already part of ``signature``, so the cache key
    stays correct.
    """
    cfg = EngineConfig.from_dict(json.loads(signature))
    return run_engine(returns_df, cfg, build_frontier=False, external_returns=_external)


def _summarize_run(name: str, run) -> dict:
    w = run.result.weights.sort_values(ascending=False)
    top3 = ", ".join(f"{a}: {v:.1%}" for a, v in w.head(3).items())
    diag = run.diagnostics
    return {
        "Scenario": name,
        "Optimizer": run.config.optimizer.name,
        "Expected Return": run.result.expected_return,
        "Expected Vol": run.result.expected_volatility,
        "Sharpe": run.result.sharpe_ratio,
        "Effective N": diag.effective_n if diag else float("nan"),
        "Div. ratio": diag.diversification_ratio if diag else float("nan"),
        "Active": int((w.abs() > 1e-4).sum()),
        "Top 3 holdings": top3,
    }


def _scenario_returns_subset(scn: Scenario, full_returns: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in full_returns.columns if c in scn.config.expected_returns]
    if not keep:
        raise ValueError("Scenario universe disjoint from loaded data.")
    return full_returns[keep]


# ---------------------------------------------------------------------------
# 🚀 Optimize
# ---------------------------------------------------------------------------

with tab_optimize:
    head_l, head_r = st.columns([3, 1])
    with head_l:
        st.subheader("Run optimization")
        st.caption(
            f"{req.display_name} · {cov_method} covariance · "
            f"{returns.shape[1]} assets · {returns.shape[0]:,} periods"
        )
    with head_r:
        solve_clicked = st.button(
            "Optimize portfolio", type="primary", width="stretch"
        )

    if live_feasibility is not None and not live_feasibility.is_feasible:
        # Show the fix here rather than pointing at another tab: the analyst is
        # about to press Optimize, and the next click should be the correction,
        # not navigation.
        render_feasibility(live_feasibility, show_ok=False)

    if solve_clicked:
        st.session_state.walk_forward = None
        try:
            config = _build_config()
            with st.spinner("Solving…"):
                # Deliberately without ``run_stress``. The engine applies the
                # scenarios inside the solve and *raises* on a shock naming an
                # asset the book cannot hold — correct for a CLI, wrong here,
                # where it would mean a mistyped scenario destroys a good
                # allocation. The 💥 Stress tab applies the same scenarios to
                # the same book through the same ``stress_test`` call, and
                # shows that refusal as a message instead.
                st.session_state["last_run"] = run_engine(
                    returns,
                    config,
                    build_frontier=build_frontier,
                    n_frontier_points=n_frontier_points,
                    external_returns=external_returns,
                )
            st.session_state["last_error"] = None
        except Exception as exc:
            st.session_state["last_run"] = None
            st.session_state["last_error"] = str(exc)

    if st.session_state.get("last_error"):
        st.error(f"Optimization failed: {st.session_state['last_error']}")
        if live_feasibility is not None and live_feasibility.issues:
            st.markdown("**Most likely cause:**")
            render_feasibility(live_feasibility, show_ok=False)

    run = st.session_state.get("last_run")
    if run is None:
        st.info(
            "Set the method and assumptions in the sidebar, review the "
            "constraints tab, then click **Optimize portfolio**."
        )
    else:
        render_compliance(run.result)
        for warning in run.warnings:
            if warning not in run.result.violations:
                st.warning(warning)

        weights = run.result.weights
        metric_row(
            [
                ("Expected Return", pct(run.result.expected_return), None),
                ("Expected Volatility", pct(run.result.expected_volatility), None),
                (
                    "Sharpe Ratio",
                    num(run.result.sharpe_ratio),
                    f"Excess of the {risk_free_rate:.2%} risk-free rate over volatility.",
                ),
                (
                    "Solve time",
                    f"{run.result.extras.get('solve_seconds', 0):.2f}s",
                    f"Solver: {run.result.extras.get('solver', '—')}",
                ),
            ]
        )
        render_portfolio_diagnostics(run.diagnostics)
        render_projection_distance(run.result)
        render_layer_exposures(run)

        if run.benchmark is not None and run.benchmark.weights is not None:
            st.markdown("**Against the benchmark, before the fact**")
            st.caption(
                f"Ex-ante numbers, from the covariance the optimizer solved "
                f"with — not from a replay. Benchmark: {run.benchmark.label}."
            )
            _bench_w = run.benchmark.weights.reindex(weights.index).fillna(0.0)
            _active = weights - _bench_w
            _ex_ante_te = float(
                np.sqrt(
                    max(float(_active.values @ run.cov_matrix.values @ _active.values), 0.0)
                )
            )
            _bench_return = float(
                run.expected_returns.reindex(weights.index).fillna(0.0) @ _bench_w
            )
            _active_return = run.result.expected_return - _bench_return
            metric_row(
                [
                    (
                        "Expected active return",
                        pct(_active_return),
                        "Portfolio minus benchmark, on the expected-return "
                        "vector the solve used.",
                    ),
                    (
                        "Ex-ante tracking error",
                        pct(_ex_ante_te),
                        "√((w−b)'Σ(w−b)). The realized figure on the "
                        "🎯 Performance tab will differ — this is what the "
                        "covariance predicts, that is what happened.",
                    ),
                    (
                        "Implied information ratio",
                        num(_active_return / _ex_ante_te)
                        if _ex_ante_te > 1e-9
                        else "—",
                        "Expected active return per unit of expected active "
                        "risk.",
                    ),
                    (
                        "Active share",
                        pct(float(_active.abs().sum() / 2.0), 1),
                        "Half the sum of absolute weight differences.",
                    ),
                ]
            )
            with st.expander("Active weights vs. the benchmark", expanded=False):
                _active_frame = pd.DataFrame(
                    {
                        "Portfolio": weights,
                        "Benchmark": _bench_w,
                        "Active": _active,
                    }
                ).sort_values("Active", ascending=False)
                st.dataframe(
                    _active_frame.style.format("{:.2%}"), width="stretch"
                )
                st.caption(
                    "Where the risk *differs* from the benchmark's, which is "
                    "not where the risk is: a large index position carries "
                    "absolute risk and no active risk at all."
                )

        st.divider()
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Weights**")
            st.plotly_chart(
                plot_weights_bar(
                    weights[weights.abs() > 1e-4],
                    title="Allocation",
                    bounds=st.session_state.config_table,
                ),
                width="stretch",
            )
        with right:
            st.markdown("**Capital vs. risk**")
            decomposition = run.risk_decomposition()
            st.plotly_chart(
                plot_weight_vs_risk(
                    decomposition[decomposition["weight"].abs() > 1e-4]
                ),
                width="stretch",
            )
            st.caption(
                "A position can be small in capital and large in risk. The gap "
                "between the bars is what risk budgeting exists to close."
            )

        with st.expander("Full risk decomposition (Euler)", expanded=False):
            table = run.risk_decomposition()
            display = table.copy()
            for col in ("weight", "contribution", "share_of_risk", "standalone_vol"):
                display[col] = display[col].map("{:.2%}".format)
            display["marginal_risk"] = table["marginal_risk"].map("{:.4f}".format)
            st.dataframe(display, width="stretch")
            st.caption(
                "`contribution` is in annualized volatility units and sums "
                f"exactly to the portfolio's {run.result.expected_volatility:.2%}."
            )
            # Weight limits are set on capital; what they are trying to
            # control is risk. Showing both per layer is what makes the gap
            # visible — a 30% bond sleeve rarely carries 30% of the risk.
            for _layer in run.constraint_layers:
                try:
                    _rc = run.layer_risk_contributions(_layer.name)
                except ValueError:
                    continue
                if len(_rc) < 2:
                    continue
                _frame = _rc.to_frame("Share of risk")
                _frame["Weight"] = (
                    run.result.weights.groupby(_layer.assignments).sum()
                )
                st.markdown(f"**Risk by {_layer.name.lower()}**")
                st.dataframe(
                    _frame[["Weight", "Share of risk"]].style.format(
                        {"Weight": "{:.2%}", "Share of risk": "{:.2%}"}
                    ),
                    width="stretch",
                )

        if run.frontier is not None and ws["frontier"]["enabled"]:
            st.divider()
            st.markdown("### Efficient frontier")
            c1, c2 = st.columns([1, 1])
            with c1:
                show_cal = st.checkbox(
                    "Show capital allocation line", value=True,
                    help="Risk/return reachable by mixing the tangency "
                    "portfolio with cash at the risk-free rate.",
                )
            with c2:
                show_dominated = st.checkbox(
                    "Show dominated branch", value=False,
                    help="Portfolios below the minimum-variance point: same "
                    "risk, less return.",
                )
            render_frontier_health(run.frontier)
            on_cvar_axis = run.frontier.risk_measure == "CVaR"
            if on_cvar_axis:
                st.caption(
                    "The x-axis is Conditional VaR, the risk measure this "
                    "frontier was traced against. The minimum-variance and "
                    "tangency anchors and the capital allocation line live in "
                    "volatility space and have no position here."
                )
            st.plotly_chart(
                plot_efficient_frontier(
                    run.frontier,
                    risk_free_rate=(
                        float(risk_free_rate) if show_cal and not on_cvar_axis else None
                    ),
                    current_portfolio=(
                        None
                        if on_cvar_axis
                        else (
                            run.result.expected_volatility,
                            run.result.expected_return,
                            "Your portfolio",
                        )
                    ),
                    show_dominated=show_dominated,
                ),
                width="stretch",
            )

            with st.expander(
                "How much can you trust this frontier?", expanded=False
            ):
                st.caption(
                    "The frontier is a point estimate of a curve. Resampling "
                    "the return history and re-tracing it shows how far that "
                    "curve moves when the sample changes — differences "
                    "narrower than the band are not distinguishable from "
                    "estimation noise. This re-solves the whole frontier once "
                    "per draw, so it is deliberately opt-in."
                )
                u1, u2 = st.columns([2, 1])
                with u1:
                    n_draws = st.slider(
                        "Resampled histories", 10, 200, 40, 10,
                        help="More draws give a smoother band at linear cost.",
                    )
                with u2:
                    resample_method = st.selectbox(
                        "Resampling",
                        options=["block", "iid", "parametric"],
                        index=0,
                        format_func=lambda m: {
                            "block": "Block bootstrap",
                            "iid": "IID bootstrap",
                            "parametric": "Parametric (normal)",
                        }[m],
                        help=(
                            "Block preserves volatility clustering and is the "
                            "right default for returns; IID destroys it; "
                            "parametric imposes normality."
                        ),
                    )
                if st.button("Estimate frontier uncertainty", key="run_bootstrap"):
                    with st.spinner(f"Re-tracing the frontier {n_draws} times…"):
                        try:
                            st.session_state["frontier_uncertainty"] = (
                                bootstrap_frontier(
                                    returns,
                                    _build_config(),
                                    n_draws=int(n_draws),
                                    n_points=max(int(n_frontier_points) // 2, 6),
                                    method=resample_method,
                                )
                            )
                        except Exception as exc:
                            st.session_state["frontier_uncertainty"] = None
                            st.error(f"Resampling failed: {exc}")

                uncertainty = st.session_state.get("frontier_uncertainty")
                if uncertainty is not None:
                    st.info(uncertainty.summary())
                    if uncertainty.n_failed:
                        st.caption(
                            f"{uncertainty.n_failed} draw(s) could not be "
                            "traced and were dropped."
                        )
                    st.plotly_chart(
                        plot_frontier_uncertainty(uncertainty), width="stretch"
                    )
                    band = uncertainty.volatility
                    st.caption(
                        f"The band covers {band.min():.1%} to {band.max():.1%} "
                        "volatility — the risk levels every resampled history "
                        "could reach. Outside that range the draws disagree on "
                        "what is even attainable, which is its own kind of "
                        "uncertainty."
                    )
                    if not uncertainty.weight_dispersion.empty:
                        st.plotly_chart(
                            plot_weight_dispersion(
                                uncertainty.weight_dispersion.head(15)
                            ),
                            width="stretch",
                        )
                        st.caption(
                            "Positions at the top of this chart are ones the "
                            "optimizer sizes differently on every resampled "
                            "history — its conviction there comes from this "
                            "particular sample, not from the asset."
                        )

            f1, f2 = st.columns(2)
            with f1:
                st.plotly_chart(
                    plot_portfolio_composition(
                        run.frontier.weights, "Weights along the frontier"
                    ),
                    width="stretch",
                )
            with f2:
                if (
                    run.frontier.group_weights is not None
                    and not run.frontier.group_weights.empty
                ):
                    st.plotly_chart(
                        plot_portfolio_composition(
                            run.frontier.group_weights, "Group weights along the frontier"
                        ),
                        width="stretch",
                    )

        with st.expander("Estimation quality", expanded=False):
            render_covariance_diagnostics(run.covariance_diagnostics)
            st.markdown("**Expected returns used**")
            st.dataframe(
                run.expected_returns.to_frame("Annualized").style.format(
                    {"Annualized": "{:.2%}"}
                ),
                width="stretch",
            )
            if "bl_posterior_returns" in run.result.extras:
                st.markdown("**Black-Litterman: prior vs. posterior**")
                bl = pd.DataFrame(
                    {
                        "Equilibrium (prior)": run.result.extras["bl_prior_returns"],
                        "Posterior": run.result.extras["bl_posterior_returns"],
                        "View impact": run.result.extras["bl_view_impact"],
                    }
                )
                st.dataframe(
                    bl.style.format("{:.2%}"), width="stretch"
                )
                for v in run.result.extras.get("bl_views", []):
                    st.caption(f"View: {v}")
            if "hrp_clusters" in run.result.extras:
                st.markdown("**HRP clusters**")
                for label, members in run.result.extras["hrp_clusters"].items():
                    st.caption(f"Cluster {label}: {', '.join(members)}")
            if "risk_budget_achieved" in run.result.extras:
                st.markdown("**Risk budget: target vs. achieved**")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Target": run.result.extras["risk_budget_target"],
                            "Achieved": run.result.extras["risk_budget_achieved"],
                        }
                    ).style.format("{:.2%}"),
                    width="stretch",
                )

        with st.expander("Assumptions behind this result", expanded=False):
            render_assumptions(run.assumptions())


# ---------------------------------------------------------------------------
# 💥 Stress
# ---------------------------------------------------------------------------

with tab_stress:
    st.subheader("What does a named bad day do to this book?")
    st.caption(
        "Each scenario is a one-period return shock per asset applied to the "
        "solved weights: P&L = Σ wᵢ·rᵢ, decomposed back onto the positions "
        "that produced it. No horizon, no distribution, no simulation — which "
        "is why the contributions add to the loss exactly rather than "
        "approximately. An asset a scenario does not name is unmoved, not "
        "undefined."
    )

    st.markdown("**Scenarios**")
    st.caption(
        "One row per leg: a scenario is its legs, and a name it does not "
        "mention is unmoved. This book can hold "
        f"{', '.join(returns.columns)} — a leg on anything else is refused."
    )
    st.session_state.stress_shock_table = st.data_editor(
        st.session_state.stress_shock_table,
        width="stretch",
        num_rows="dynamic",
        hide_index=True,
        # Keyless, like every other grid on this page and unlike the layered
        # limits: ``data_editor`` stores its edits against the key as a delta
        # from the data it was handed, so a keyed dynamic grid replays "add a
        # row" onto a frame that already has it. Writing the returned frame
        # back to session state is the whole state model here.
        column_config={
            "Scenario": st.column_config.TextColumn(
                help="Scenario name. Every report line is keyed by it, and "
                "two legs sharing a name are two legs of one scenario."
            ),
            # Deliberately free text rather than a picker over the loaded
            # universe. A scenario library is written against the world, not
            # against today's book, and the engine's refusal to apply a shock
            # to a name that cannot be held is the safeguard — hiding the case
            # behind a dropdown would hide the safeguard too.
            "Asset": st.column_config.TextColumn(
                help="An asset name. One this book does not hold is refused "
                "rather than zeroed; the message below says so."
            ),
            "Shock return": st.column_config.NumberColumn(
                format="%.4f",
                help="One-period simple return, as a fraction: -0.32 is a 32% "
                "fall.",
            ),
        },
    )

    st.session_state.stress_scenario_table = align_scenario_table(
        st.session_state.stress_shock_table,
        st.session_state.get("stress_scenario_table"),
    )
    if not st.session_state.stress_scenario_table.empty:
        st.session_state.stress_scenario_table = st.data_editor(
            st.session_state.stress_scenario_table,
            width="stretch",
            num_rows="fixed",
            column_config={
                "Covariance ×": st.column_config.NumberColumn(
                    format="%.2f",
                    help="What the scenario does to *risk*: a multiple of the "
                    "base covariance, so 4.00 doubles every volatility with "
                    "correlations unchanged. Blank leaves risk unstressed.",
                ),
                "Notes": st.column_config.TextColumn(
                    help="Where the numbers came from, or which episode they "
                    "are calibrated to. Carried into the report."
                ),
            },
        )

    up_left, up_right = st.columns([3, 2])
    with up_left:
        shocks_file = st.file_uploader(
            "⬆ Load scenarios from YAML",
            type=["yaml", "yml", "json"],
            key="stress_upload",
            help="A --stress file, in either shape the CLI accepts: a mapping "
            "with a 'shocks' list, or a bare list of scenarios.",
        )
        if shocks_file is not None:
            try:
                uploaded_shocks = load_shocks_yaml(
                    shocks_file.getvalue().decode("utf-8")
                )
            except (StressError, UnicodeDecodeError) as exc:
                st.error(f"Could not read those scenarios: {exc}")
            else:
                digest = (shocks_file.name, len(shocks_file.getvalue()))
                if st.session_state.get("stress_upload_applied") != digest:
                    _adopt_shock_tables(*tables_from_shocks(uploaded_shocks))
                    st.session_state["stress_upload_applied"] = digest
                    st.success(
                        f"Loaded {len(uploaded_shocks)} scenario(s) from "
                        f"{shocks_file.name}."
                    )
                    st.rerun()

    shock_dicts, shock_problems = _current_shock_dicts()
    for problem in shock_problems:
        st.error(f"Not a scenario yet — {problem}")

    with up_right:
        st.download_button(
            "⬇ Download scenarios (YAML)",
            data=(
                dump_shocks_yaml(shocks_from_dicts(shock_dicts))
                if shock_dicts
                else ""
            ),
            file_name="shocks.yaml",
            mime="text/yaml",
            disabled=not shock_dicts,
            width="stretch",
            help="The same file `optengine optimize --stress` reads.",
        )
        st.caption(
            "Scenarios are part of the configuration, so saving a preset in "
            "the sidebar saves them with it."
        )

    st.divider()

    run = st.session_state.get("last_run")
    if not shock_dicts:
        st.info(
            "No scenarios yet. Add a row above, or load a file — "
            "`config/shocks.yaml` is a worked example over this panel."
        )
    elif run is None:
        st.info(
            "Run an optimization first — a scenario is applied to a book, and "
            "there is not one yet."
        )
    else:
        unheld = unheld_shocked_assets(shock_dicts, returns.columns)
        unknown_policy = st.radio(
            "A scenario naming an asset this book cannot hold",
            options=["raise", "ignore"],
            format_func=lambda p: {
                "raise": "Refuse the scenario (default)",
                "ignore": "Apply it anyway, recording what was dropped",
            }[p],
            horizontal=True,
            key="stress_unknown_assets",
            help=(
                "The loss a scenario describes on a name the book cannot hold "
                "cannot reach the book, so applying it anyway reports a "
                "smaller loss than the scenario says. Refusing is the default "
                "for that reason; the other option makes the narrowing "
                "visible rather than assumed."
            ),
        )
        try:
            report = stress_test(
                run.result.weights,
                shocks_from_dicts(shock_dicts),
                cov_matrix=run.cov_matrix,
                unknown_assets=unknown_policy,
            )
        except StressError as exc:
            # The library's refusal, shown rather than swallowed: the scenario
            # is still in the grid, still in the config, and still wrong.
            st.error(f"**Stress test refused.** {exc}")
            if unheld:
                st.markdown(
                    "\n".join(
                        f"- **{name}** moves {', '.join(missing)}, which this "
                        "book does not hold."
                        for name, missing in unheld.items()
                    )
                )
                st.caption(
                    "Drop those legs, widen the universe in the sidebar, or "
                    "switch the choice above to apply the scenario with them "
                    "recorded as dropped."
                )
        else:
            render_stress_report(report)


# ---------------------------------------------------------------------------
# 📉 Backtest
# ---------------------------------------------------------------------------

with tab_backtest:
    run = st.session_state.get("last_run")
    if run is None:
        st.info("Run an optimization first — the backtest replays its weights.")
    else:
        st.subheader("How would this allocation have behaved?")
        st.warning(
            "**Everything on this page except the walk-forward section is "
            "in-sample.** The optimizer estimated its inputs from these same "
            "returns, so it already knew which assets won. Treat these "
            "numbers as a description of the fit, not as a forecast."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            frequency = st.selectbox(
                "Rebalancing",
                options=["none", "monthly", "quarterly", "annual", "weekly", "daily"],
                index=1,
                format_func=lambda f: f.title() if f != "none" else "Buy and hold",
                help=(
                    "Between rebalances, weights drift with performance. "
                    "Assuming constant weights silently assumes free, "
                    "continuous rebalancing."
                ),
            )
            execution_lag = st.number_input(
                "Execution lag (periods)", min_value=0, max_value=10, value=0, step=1,
                help=(
                    "Periods between choosing a target and trading it. At zero "
                    "the book fills on a close it has not seen — the "
                    "conventional, optimistic assumption."
                ),
            )
        with c2:
            commission_bps = st.slider(
                "Commission (bps, one-way)", 0, 100, 10,
                help="The broker's share. Charged on traded notional.",
            )
            slippage_bps = st.slider(
                "Slippage (bps, one-way)", 0, 100, 0,
                help="The half-spread you cross. A market property, not a broker one.",
            )
        with c3:
            impact_eta = st.slider(
                "Market impact (eta)", 0.0, 2.0, 0.0, step=0.1,
                help=(
                    "Square-root impact: cost grows with the square root of "
                    "trade size, so the same allocation gets more expensive as "
                    "the book grows. Zero switches impact off."
                ),
            )
            participation = st.slider(
                "Tradable per period (% of book)", 0.5, 20.0, 5.0, step=0.5,
                help=(
                    "How much of the book can be traded in one name, in one "
                    "period, without moving the price. Smaller is a thinner "
                    "market and a more expensive rebalance."
                ),
            )

        # Volume is optional throughout. With none — the normal state for an
        # index universe — the impact model prices from the fixed rate above,
        # and the selector says so rather than letting the fallback be a
        # surprise in the run log.
        backtest_volumes = (
            ingest_result.volumes if ingest_result is not None else None
        )
        if float(impact_eta) > 0.0:
            liquidity, initial_capital = render_liquidity_selector(backtest_volumes)
        else:
            liquidity, initial_capital = {"impact_participation_source": "fixed"}, 1.0

        cost_bps = float(commission_bps) + float(slippage_bps)
        try:
            backtest_spec = BacktestSpec(
                frequency=frequency,
                costs=CostSpec(
                    commission_bps=float(commission_bps),
                    slippage_bps=float(slippage_bps),
                    impact_coefficient=float(impact_eta),
                    impact_participation=float(participation) / 100.0,
                    **liquidity,
                ),
                execution_lag=int(execution_lag),
                periods_per_year=int(periods_per_year),
                initial_capital=float(initial_capital),
            )
        except SpecValidationError as exc:
            # A cost model the numbers cannot express is something to say, not
            # something to fall over on.
            st.error(str(exc))
            st.stop()
        bt = run.simulate(
            backtest_spec,
            prices=prices.reindex(returns.index) if backtest_volumes is not None else None,
            volumes=(
                backtest_volumes.reindex(index=returns.index, columns=returns.columns)
                if backtest_volumes is not None
                else None
            ),
        )
        metric_row(
            [
                (
                    "Annualized return (net)",
                    pct(
                        float(
                            annualize_returns(
                                bt.returns, periods_per_year=int(periods_per_year)
                            )
                        )
                    ),
                    None,
                ),
                ("Turnover per year", num(bt.annualized_turnover, 2), "One-way."),
                ("Total cost", pct(bt.total_cost), None),
                (
                    "Cost drag",
                    pct(bt.cost_drag(int(periods_per_year))),
                    "Annualized return given up to trading.",
                ),
            ]
        )

        with st.expander("What did the trading cost, and where did it go?"):
            panel = compute_tca(bt)
            st.caption(panel.describe())
            t1, t2 = st.columns(2)
            with t1:
                st.dataframe(panel.to_frame(), width="stretch")
            with t2:
                by_asset = cost_by_asset(bt)
                if by_asset.empty:
                    st.info("Nothing traded, so there is no cost to attribute.")
                else:
                    st.dataframe(by_asset.head(15), width="stretch")
            for reason in panel.reasons.values():
                st.caption(f"— {reason}")
            for degradation in panel.degradations:
                st.warning(
                    f"{degradation}. The reported cost is a lower bound on the "
                    "modelled one."
                )

        st.plotly_chart(
            plot_wealth_index(
                bt.returns.to_frame("portfolio"), "Portfolio wealth (start = 1)"
            ),
            width="stretch",
        )
        b1, b2 = st.columns(2)
        with b1:
            st.plotly_chart(
                plot_drawdown(bt.returns, "Drawdown"), width="stretch"
            )
        with b2:
            st.plotly_chart(
                plot_return_distribution(
                    bt.returns,
                    var=float(np.percentile(bt.returns, 5)),
                    cvar=float(bt.returns[bt.returns <= np.percentile(bt.returns, 5)].mean()),
                    title="Return distribution with tail cuts",
                ),
                width="stretch",
            )

        st.markdown("**Summary**")
        st.dataframe(
            format_table(
                bt.summary(int(periods_per_year), float(risk_free_rate))
            ),
            width="stretch",
        )

        st.divider()
        st.markdown("### Versus a benchmark")
        render_benchmark(run)
        benchmark_returns = run.benchmark_returns

        if benchmark_returns is not None:
            # One selection, made once in the sidebar, drives this panel, the
            # performance page and the export. A second picker here is how the
            # tabs end up quoting information ratios against different indices.
            relative = summary_relative(
                bt.returns.to_frame("portfolio"),
                benchmark_returns.reindex(bt.returns.index),
                periods_per_year=int(periods_per_year),
                riskfree_rate=float(risk_free_rate),
                extended=True,
            ).T
            share = None
            if run.benchmark.weights is not None:
                share = active_share(run.result.weights, run.benchmark.weights)
            metric_row(
                [
                    (
                        "Annualized excess",
                        pct(float(relative.loc["Annualized Excess", "portfolio"])),
                        None,
                    ),
                    (
                        "Tracking error",
                        pct(float(relative.loc["Annualized T.E.", "portfolio"])),
                        None,
                    ),
                    (
                        "Information ratio",
                        num(float(relative.loc["Information Ratio", "portfolio"])),
                        "Excess return per unit of tracking error.",
                    ),
                    (
                        "Active share",
                        pct(share) if share is not None else "—",
                        "Half the sum of absolute weight differences. 0 is the "
                        "benchmark; 1 shares no holding with it.",
                    ),
                ]
            )
            st.plotly_chart(
                plot_wealth_index(
                    pd.concat(
                        {
                            "Portfolio": bt.returns,
                            "Benchmark": benchmark_returns.reindex(bt.returns.index),
                        },
                        axis=1,
                    ),
                    "Portfolio vs. benchmark",
                ),
                width="stretch",
            )
            st.dataframe(format_table(relative), width="stretch")
            t_stat = float(relative.loc["Alpha t-stat", "portfolio"])
            alpha = float(relative.loc["Alpha (annualized)", "portfolio"])
            if abs(t_stat) < 2:
                st.info(
                    f"CAPM alpha is {alpha:.2%} with a t-statistic of "
                    f"{t_stat:.2f}. Below |2| the alpha is not statistically "
                    "distinguishable from zero on this sample — and this is "
                    "still the in-sample fit."
                )
            else:
                st.success(
                    f"CAPM alpha of {alpha:.2%} (t = {t_stat:.2f}) is "
                    "statistically significant in sample. Confirm it survives "
                    "the walk-forward below before believing it."
                )
            st.caption(
                "The 🎯 Performance tab carries the full relative picture — "
                "capture, batting average, rolling tracking error and the "
                "exportable tables."
            )

        if frequency != "daily":
            with st.expander("Weight drift between rebalances", expanded=False):
                st.plotly_chart(
                    plot_weight_evolution(bt.weights, "Actual held weights"),
                    width="stretch",
                )

        window = st.slider(
            "Rolling window (periods)",
            min_value=int(periods_per_year // 4),
            max_value=min(int(periods_per_year * 3), max(len(bt.returns) - 1, 2)),
            value=min(int(periods_per_year), max(len(bt.returns) - 1, 2)),
            help=(
                "A full-sample Sharpe cannot tell a strategy that worked "
                "throughout from one that earned everything in a single window."
            ),
        )
        st.plotly_chart(
            plot_rolling_metrics(
                rolling_metrics(
                    bt.returns, window, float(risk_free_rate), int(periods_per_year)
                )
            ),
            width="stretch",
        )

        st.divider()
        st.markdown("### Out-of-sample walk-forward")
        st.caption(
            "Re-estimates and re-solves on a rolling window, then holds each "
            "solution over returns the optimizer never saw. The gap against "
            "the in-sample curve is how much of the backtest was hindsight. "
            "Two cadences to set, and they are different decisions: how often "
            "the optimizer **re-solves**, and how often the book is **traded "
            "back** to whatever target is current."
        )
        w1, w2, w3 = st.columns(3)
        with w1:
            lookback = st.number_input(
                "Estimation window (periods)",
                min_value=int(periods_per_year // 2),
                max_value=max(int(len(returns) - periods_per_year // 4), 10),
                value=min(int(periods_per_year * 2), max(len(returns) // 2, 10)),
            )
        with w2:
            rebalance_every = st.number_input(
                "Re-solve every (periods)",
                min_value=1, max_value=int(periods_per_year * 2),
                value=max(int(periods_per_year // 4), 1),
                help=(
                    "The re-optimization cadence: how often the optimizer "
                    "sees new data and re-solves. Separate from how often the "
                    "book trades — set that below."
                ),
            )
            wf_frequency = st.selectbox(
                "Rebalance between re-solves",
                options=["none", "monthly", "quarterly", "annual", "weekly", "daily"],
                index=0,
                format_func=lambda f: (
                    "Only when re-solving" if f == "none" else f.title()
                ),
                help=(
                    "The rebalancing cadence, which is a different question "
                    "from the one above. 'Only when re-solving' holds each "
                    "solution untouched until the next one and lets the "
                    "weights drift. Anything else pulls the book back to the "
                    "current target on that calendar — more turnover, more "
                    "cost, less drift. A quarterly committee with a monthly "
                    "rebalancing discipline re-solves four times a year and "
                    "trades twelve."
                ),
            )
        with w3:
            expanding = st.checkbox(
                "Expanding window", value=False,
                help="Grow the sample from inception instead of rolling it.",
            )
            reestimate_mu = st.checkbox(
                "Re-estimate expected returns per window",
                value=True,
                help=(
                    "On by default, and leaving it on is what makes this "
                    "out-of-sample. The expected returns in the constraints "
                    "tab are seeded from the whole history, so reusing them "
                    "would hand every window an estimate built partly from its "
                    "own future. Turn it off only if your expected returns are "
                    "genuine forward-looking assumptions rather than estimates "
                    "from this data."
                ),
            )

        if st.button("Run walk-forward", key="run_wf"):
            with st.spinner("Re-solving through history…"):
                try:
                    st.session_state.walk_forward = run.walk_forward(
                        lookback=int(lookback),
                        rebalance_every=int(rebalance_every),
                        transaction_cost_bps=float(cost_bps),
                        expanding=expanding,
                        reestimate_expected_returns=bool(reestimate_mu),
                        rebalance_frequency=wf_frequency,
                    )
                except Exception as exc:
                    st.session_state.walk_forward = None
                    st.error(f"Walk-forward failed: {exc}")

        wf = st.session_state.get("walk_forward")
        if wf is not None:
            metric_row(
                [
                    (
                        "Re-solves",
                        str(wf.n_resolves),
                        "Times the optimizer re-estimated and re-solved.",
                    ),
                    (
                        "Trade dates",
                        str(wf.n_trade_dates),
                        "Dates the book actually traded. Above the re-solve "
                        "count when a rebalancing cadence sits between them.",
                    ),
                    (
                        "OOS annualized return",
                        pct(
                            float(
                                annualize_returns(
                                    wf.returns, periods_per_year=int(periods_per_year)
                                )
                            )
                        ),
                        None,
                    ),
                    ("Turnover per year", num(wf.backtest.annualized_turnover, 2), None),
                    (
                        "Failed solves",
                        str(len(wf.failures)),
                        "Failed re-solves carry the previous book forward.",
                    ),
                ]
            )
            if not wf.backtest.metadata.get("reestimated_expected_returns", True):
                st.warning(
                    "Expected returns were held fixed across every window. If "
                    "they came from this history rather than from forward-"
                    "looking assumptions, the numbers below are not fully "
                    "out-of-sample."
                )
            in_sample = bt.returns.reindex(wf.returns.index)
            st.plotly_chart(
                plot_walk_forward_comparison(in_sample, wf.returns),
                width="stretch",
            )
            comparison = run.in_vs_out_of_sample(wf, float(risk_free_rate))
            st.markdown("**In-sample vs. out-of-sample**")
            st.dataframe(format_table(comparison), width="stretch")
            degradation = comparison.loc["Sharpe Ratio", "Degradation"]
            if degradation > 0.5:
                st.error(
                    f"The Sharpe ratio falls by {degradation:.2f} out of sample. "
                    "Most of the in-sample result was fitted, not earned — "
                    "consider a more robust method (HRP, minimum variance, "
                    "risk parity) or shrunk expected returns."
                )
            elif degradation > 0.2:
                st.warning(
                    f"The Sharpe ratio falls by {degradation:.2f} out of sample — "
                    "a normal amount of optimism, but size positions on the "
                    "out-of-sample number."
                )
            else:
                st.success(
                    f"The Sharpe ratio holds up out of sample (gap "
                    f"{degradation:.2f})."
                )

            stability = wf.weight_stability()
            if not stability.empty:
                with st.expander("Weight stability across re-solves", expanded=False):
                    st.caption(
                        "Average absolute change in each asset's weight between "
                        "re-solves. Large values mean the optimizer is chasing "
                        "estimation noise, and the turnover it implies is real."
                    )
                    st.dataframe(
                        stability.sort_values(ascending=False)
                        .to_frame("Mean absolute change")
                        .style.format("{:.2%}"),
                        width="stretch",
                    )
            if wf.failures:
                with st.expander(f"{len(wf.failures)} failed re-solve(s)"):
                    for f in wf.failures[:20]:
                        st.text(f)


# ---------------------------------------------------------------------------
# 🌍 Universe
# ---------------------------------------------------------------------------

#: Falls back to this when ``config/universe.yaml`` is not beside the app —
#: a screen over nothing but the return panel, so it needs no data files.
_DEFAULT_UNIVERSE_RULES = """schema_version: 1
combine: all
rules:
  - kind: rolling
    panel: returns
    window: 63
    agg: std
    op: "<"
    value: 0.0126
    name: trailing 63-session volatility under 20% annualized
"""

_UNIVERSE_EXAMPLE = ROOT / "config" / "universe.yaml"


@st.cache_data(show_spinner=False, max_entries=8)
def _universe_cached(rules_text: str, base_dir: str, returns_hash: str, _returns, _prices):
    """Parse and evaluate a rules document. Cached on the text, not the object.

    Returns ``(rules, eligibility)``. Raising is left to the caller so the
    error lands on the page beside the text box that caused it.
    """
    import yaml as _yaml

    try:
        document = _yaml.safe_load(rules_text)
    except _yaml.YAMLError as exc:
        # Re-raised as the package's own error so the caller has one except
        # clause rather than two, and so a syntax slip reads like every other
        # thing wrong with the document.
        raise UniverseError(f"That is not readable YAML: {exc}") from exc
    if document is None:
        raise UniverseError("The rules document is empty.")
    rules = UniverseRules.from_dict(document, base_dir=Path(base_dir))
    if rules.panels:
        # A `panels:` entry is a path on the machine running this page, and
        # `streamlit run` listens on every interface: honouring it would let
        # anyone who can reach the page have the server read a file of their
        # choosing, with the parse error quoting what it found. The page
        # supplies `returns` and `prices`; a screen over a characteristic
        # panel runs from `optengine backtest --universe`, where the operator
        # names the files.
        raise UniverseError(
            "This page does not read panels from disk, so a rules document "
            f"here cannot declare 'panels:' ({', '.join(p.name for p in rules.panels)}). "
            "Screen on the run's own `returns` or `prices`, or run the file "
            "with `optengine backtest --universe`."
        )
    return rules, rules.build(returns=_returns, prices=_prices)


with tab_universe:
    st.subheader("Who was investable, and when?")
    st.caption(
        "Every other tab takes the universe from the columns of the return "
        "panel — the universe as it looks *today*. Names that were delisted "
        "or dropped from the index never appear, and the ones that were added "
        "appear from the first day of the sample. That is survivorship bias "
        "and look-ahead in the same panel, and no amount of care in the "
        "optimizer removes it. A rules file puts the missing axis back."
    )

    if "universe_rules_text" not in st.session_state:
        st.session_state["universe_rules_text"] = (
            _UNIVERSE_EXAMPLE.read_text(encoding="utf-8")
            if _UNIVERSE_EXAMPLE.exists()
            else _DEFAULT_UNIVERSE_RULES
        )
        st.session_state["universe_rules_base"] = str(
            _UNIVERSE_EXAMPLE.parent if _UNIVERSE_EXAMPLE.exists() else ROOT
        )

    # No path box. A free-text "Rules file on disk" read any file the server
    # could see, for anyone who could reach the page — `streamlit run` listens
    # on every interface — and the parse error quoted what it found. The
    # shipped example is one click away and anything else comes in through
    # the uploader, which never names a server path: the same reasoning that
    # keeps the `file` provider out of the Data tab (`data_sources.py`).
    if _UNIVERSE_EXAMPLE.exists() and st.button(
        "Load the example rules",
        key="universe_load_example",
        help=(
            "Replace the box below with config/universe.yaml — the volatility "
            "band with hysteresis the README walks through."
        ),
    ):
        st.session_state["universe_rules_text"] = _UNIVERSE_EXAMPLE.read_text(
            encoding="utf-8"
        )
        st.session_state["universe_rules_base"] = str(_UNIVERSE_EXAMPLE.parent)
        st.rerun()

    rules_file = st.file_uploader(
        "⬆ Rules file (YAML/JSON)",
        type=["yaml", "yml", "json"],
        key="universe_upload",
        help=(
            "The same --universe file the CLI reads. A rules document that "
            "screens on a characteristic panel — ADV, market capitalisation — "
            "names that panel's path itself, which is why the data comes with "
            "the rules rather than behind a second control."
        ),
    )
    if rules_file is not None:
        digest = (rules_file.name, len(rules_file.getvalue()))
        if st.session_state.get("universe_upload_applied") != digest:
            st.session_state["universe_rules_text"] = rules_file.getvalue().decode(
                "utf-8", errors="replace"
            )
            # An uploaded document has no directory of its own, so a relative
            # panel path in it has nothing to resolve against. Saying so is
            # better than resolving it against a directory it never meant.
            st.session_state["universe_rules_base"] = str(ROOT)
            st.session_state["universe_upload_applied"] = digest
            st.rerun()

    # Keyed on the session value itself, so replacing the text from the
    # uploader or the path loader actually redraws the box: a keyed widget's
    # own state wins over a ``value=`` argument on every rerun after the first.
    rules_text = st.text_area(
        "Rules",
        height=260,
        key="universe_rules_text",
        help="Edited here, evaluated as soon as the box loses focus.",
    )

    universe = None
    try:
        parsed_rules, universe = _universe_cached(
            rules_text,
            st.session_state["universe_rules_base"],
            _frame_hash(returns),
            returns,
            prices.reindex(returns.index),
        )
    except (UniverseError, OSError, ValueError) as exc:
        st.error(f"**These rules do not describe a universe.** {exc}")
    else:
        st.caption(parsed_rules.describe())

    if universe is not None:
        eligibility_frame = universe.frame.reindex(columns=list(returns.columns))
        counts = eligibility_state_counts(eligibility_frame)
        cells, bars, decided_names = count_unresolved(
            universe, returns.index, list(returns.columns)
        )

        st.divider()
        st.markdown("**How the unknowns are read**")
        st.caption(
            "There is deliberately no default. `exclude` quietly shrinks the "
            "book across every warm-up, `include` quietly admits names no "
            "rule has screened, and `raise` stops any run whose rules have a "
            "warm-up at all. Which of those is wrong depends on the mandate, "
            "so this page will not choose one for you."
        )
        policy = st.radio(
            "Collapse policy",
            options=list(MASK_POLICIES),
            index=None,
            horizontal=True,
            key="universe_policy",
            format_func=lambda p: {
                "exclude": "exclude — unknown means not eligible",
                "include": "include — unknown means eligible",
                "raise": "raise — refuse to guess",
            }[p],
        )
        st.info(describe_policy_cost(policy, cells, bars, decided_names))

        st.divider()
        st.markdown("**Eligibility, three states**")
        metric_row(
            [
                (label, f"{counts[label]:,} cells", None)
                for label in ELIGIBILITY_STATES
            ]
            + [
                (
                    "Names the policy decides",
                    f"{len(decided_names):,}",
                    "Assets touched by at least one cell no rule reached.",
                )
            ]
        )
        heatmap_frame, step = thin_rows(eligibility_frame, 400)
        st.plotly_chart(
            plot_eligibility_heatmap(heatmap_frame), width="stretch"
        )
        st.caption(
            (
                f"Every {step}th bar is drawn, ending on the last one — "
                "sampled, not aggregated, because there is no honest way to "
                "collapse a fortnight containing both an eligible and an "
                "unevaluated day into one cell. "
                if step > 1
                else ""
            )
            + "Amber is *not evaluable*: no rule reached that cell. It is a "
            "third state, not a shade of ineligible — reading it as "
            "'excluded' is exactly the mistake the colour exists to prevent."
        )

        st.markdown("**Breadth and churn**")
        st.plotly_chart(
            plot_universe_breadth(universe.breadth(), universe.unknown_count()),
            width="stretch",
        )
        turnover = universe.turnover()
        st.plotly_chart(plot_universe_turnover(turnover), width="stretch")
        st.caption(
            f"{int(turnover['entries'].sum()):,} entries and "
            f"{int(turnover['exits'].sum()):,} exits over the sample. Only "
            "transitions between two *evaluated* states count: a name going "
            "from unknown to eligible has not entered the universe, it has "
            "become knowable."
        )

        st.markdown("**Why was a name in or out?**")
        index_dates = pd.DatetimeIndex(universe.index)
        first_day, last_day = index_dates[0].date(), index_dates[-1].date()
        universe_assets = [str(a) for a in universe.assets]
        # Both controls are keyed, so their remembered value outlives the
        # universe that produced it: edit the rules, or reload the panel, and
        # yesterday's date or a name this universe never heard of would be
        # handed back to a widget that refuses it. Reset rather than raise.
        if not (first_day <= st.session_state.get(
            "universe_explain_date", last_day
        ) <= last_day):
            st.session_state["universe_explain_date"] = last_day
        if st.session_state.get("universe_explain_asset") not in universe_assets:
            st.session_state.pop("universe_explain_asset", None)
        e1, e2 = st.columns([1, 1])
        with e1:
            explain_date = st.date_input(
                "As of",
                value=last_day,
                min_value=first_day,
                max_value=last_day,
                key="universe_explain_date",
            )
        with e2:
            explain_asset = st.selectbox(
                "Asset",
                options=universe_assets,
                key="universe_explain_asset",
            )
        try:
            st.info(universe.explain(explain_date, explain_asset))
        except UniverseError as exc:
            st.warning(str(exc))

        st.divider()
        st.markdown("**Apply it to a run**")
        run = st.session_state.get("last_run")
        if run is None:
            st.info("Run an optimization first — a universe is applied to a book.")
        elif policy is None:
            st.info(
                "Choose a collapse policy above. A run needs one, and the "
                "library will not pick it either."
            )
        else:
            universe_mode = st.radio(
                "How to apply it",
                options=["replay", "walk_forward"],
                horizontal=True,
                key="universe_mode",
                format_func=lambda m: {
                    "replay": "Replay this book under the universe",
                    "walk_forward": "Re-solve each window inside the universe",
                }[m],
                help=(
                    "The replay keeps the weights already solved and only "
                    "sells what the universe stops admitting. The "
                    "walk-forward re-optimizes inside the eligible set at "
                    "each decision, which is the honest version and the "
                    "slower one."
                ),
            )
            u1, u2 = st.columns([1, 1])
            with u1:
                universe_frequency = st.selectbox(
                    "Rebalancing",
                    options=["monthly", "quarterly", "annual", "weekly", "none"],
                    index=0,
                    format_func=lambda f: f.title() if f != "none" else "Buy and hold",
                    key="universe_frequency",
                    help=(
                        "Eligibility is read at each *decision*, never at the "
                        "execution, so a rebalance acts on the universe the "
                        "desk could see when it chose the target."
                    ),
                )
            with u2:
                if universe_mode == "walk_forward":
                    delisting_grace = st.number_input(
                        "Delisting grace (bars of silence)",
                        min_value=0, max_value=250, value=5, step=1,
                        key="universe_delisting_grace",
                        help=(
                            "A separate opt-in from the screen, because it "
                            "answers a different question: the universe says "
                            "what the mandate permits, this says what still "
                            "trades. A name silent for longer than this is "
                            "liquidated at its last print and never traded "
                            "again."
                        ),
                    )
                    universe_lag = 1
                else:
                    delisting_grace = None
                    universe_lag = st.number_input(
                        "Execution lag (periods)",
                        min_value=0, max_value=10, value=1, step=1,
                        key="universe_lag",
                    )
            if st.button("Run it", key="run_universe"):
                spec = BacktestSpec(
                    frequency=universe_frequency,
                    execution_lag=int(universe_lag),
                    periods_per_year=int(periods_per_year),
                )
                try:
                    with st.spinner("Applying the universe…"):
                        if universe_mode == "walk_forward":
                            universe_result = run.walk_forward_run(
                                spec=spec,
                                rebalance_frequency=universe_frequency,
                                universe=universe,
                                universe_policy=policy,
                                delisting_grace=int(delisting_grace),
                            ).run
                        else:
                            universe_result = run_backtest(
                                returns,
                                run.result.weights,
                                spec,
                                universe=universe,
                                universe_policy=policy,
                            )
                except (UniverseError, ValueError) as exc:
                    st.session_state["universe_run_notes"] = None
                    # Under 'raise' this is the run stopping on the very cells
                    # the policy line above counted, which is the diagnosis
                    # rather than a surprise.
                    st.error(f"The run stopped: {exc}")
                else:
                    st.session_state["universe_run_notes"] = dict(
                        universe_result.meta.notes
                    )

            notes = st.session_state.get("universe_run_notes")
            if notes is not None:
                render_universe_notes(notes)
                if "delistings" not in notes:
                    st.caption(
                        "Delisting was not diagnosed: a replay only applies "
                        "the screen. Re-solve each window with a grace period "
                        "to find names that stopped printing."
                    )
                st.caption(
                    "A name the universe drops is sold at the next bar and "
                    "the proceeds sit in cash: the runner never renormalises "
                    "the rest of the book to hide the gap, because that would "
                    "be a trade nobody decided to make."
                )


# ---------------------------------------------------------------------------
# 🎯 Performance
# ---------------------------------------------------------------------------


def _performance_downloads(report, key_prefix: str) -> None:
    """Excel workbook and tidy CSV of everything on this page.

    Two formats because they answer different questions: the workbook is what
    gets circulated, the long-form CSV is what gets pivoted against last
    quarter's run without anyone having to align column orders by hand.
    """
    left, right = st.columns(2)
    buf = io.BytesIO()
    frames = performance_sheets(report)
    with excel_writer(buf) as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name[:31], index=True)
    buf.seek(0)
    left.download_button(
        "📥 Performance workbook (Excel)",
        data=buf,
        file_name="performance_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        key=f"{key_prefix}_xlsx",
        width="stretch",
    )
    right.download_button(
        "📄 Metrics (CSV)",
        data=report.metrics().to_csv(index=False).encode("utf-8"),
        file_name="performance_metrics.csv",
        mime="text/csv",
        key=f"{key_prefix}_csv",
        width="stretch",
        help=(
            "One row per metric: block, series, metric, value. The shape that "
            "survives a pivot table and a diff against another run."
        ),
    )


with tab_performance:
    run = st.session_state.get("last_run")
    if run is None:
        st.info("Run an optimization first — this page measures its track record.")
    else:
        st.subheader("Performance")
        render_benchmark(run)

        c1, c2, c3 = st.columns(3)
        with c1:
            perf_frequency = st.selectbox(
                "Rebalancing",
                options=["monthly", "quarterly", "annual", "weekly", "daily", "none"],
                index=0,
                format_func=lambda f: f.title() if f != "none" else "Buy and hold",
                key="perf_frequency",
                help=(
                    "The track record is a replay of the solved weights under "
                    "this rule, net of the cost below."
                ),
            )
        with c2:
            perf_cost_bps = st.slider(
                "Transaction cost (bps, one-way)", 0, 100, 10, key="perf_cost",
            )
        with c3:
            perf_period_freq = st.selectbox(
                "Period table",
                options=["yearly", "quarterly", "monthly"],
                index=0,
                key="perf_period_freq",
            )

        perf_window = st.slider(
            "Rolling window (periods)",
            min_value=max(int(periods_per_year) // 4, 5),
            max_value=min(int(periods_per_year) * 3, max(len(returns) - 1, 10)),
            value=min(int(periods_per_year), max(len(returns) - 1, 10)),
            key="perf_window",
            help=(
                "One year by default. A full-sample ratio cannot tell a "
                "strategy that worked throughout from one that earned "
                "everything in a single quarter; this can."
            ),
        )

        try:
            report = run.performance(
                riskfree_rate=float(risk_free_rate),
                frequency=perf_frequency,
                transaction_cost_bps=float(perf_cost_bps),
                rolling_window=int(perf_window),
                period_freq=perf_period_freq,
            )
        except (ValueError, BenchmarkError) as exc:
            report = None
            st.error(f"Performance report unavailable: {exc}")

        if report is not None:
            st.markdown(f"> {report.describe()}")
            if not report.metadata.get("out_of_sample", False):
                st.warning(
                    "**In-sample.** The optimizer estimated its inputs from "
                    "these same returns, so it already knew which assets won. "
                    "The walk-forward section of the Backtest tab is the "
                    "honest version of this page."
                )

            head = report.headline()

            st.markdown("#### Absolute")
            metric_row(
                [
                    ("Annualized return", pct(head["annualized_return"]), None),
                    ("Volatility", pct(head["annualized_volatility"]), None),
                    (
                        "Sharpe",
                        num(head["sharpe_ratio"]),
                        "Excess return per unit of total risk.",
                    ),
                    (
                        "Sortino",
                        num(head["sortino_ratio"]),
                        "Excess return per unit of *downside* deviation.",
                    ),
                    ("Max drawdown", pct(head["max_drawdown"]), None),
                    (
                        "Calmar",
                        num(head["calmar_ratio"]),
                        "Annualized return over the worst drawdown.",
                    ),
                ]
            )
            metric_row(
                [
                    (
                        "Hit rate",
                        pct(head["hit_rate"], 1),
                        "Share of periods with a positive return.",
                    ),
                    (
                        "Prob. Sharpe > 0",
                        pct(head["probabilistic_sharpe"], 1),
                        "Probability the true Sharpe exceeds zero, given the "
                        "skew and fat tails of this sample. Below ~95% the "
                        "Sharpe has not been demonstrated.",
                    ),
                    (
                        "Gain-to-pain",
                        num(report.absolute.loc[PORTFOLIO, "Gain-to-Pain"]),
                        "Total return over the sum of the losing periods.",
                    ),
                    (
                        "Time under water",
                        pct(report.absolute.loc[PORTFOLIO, "Time Under Water"], 1),
                        "Share of periods spent below the previous high-water "
                        "mark — the statistic an investor experiences.",
                    ),
                    (
                        "Turnover / year",
                        num(report.metadata.get("annualized_turnover"), 2),
                        "One-way.",
                    ),
                    (
                        "Ulcer index",
                        num(report.absolute.loc[PORTFOLIO, "Ulcer Index"], 3),
                        "Root-mean-square of the whole drawdown path: depth "
                        "and duration together.",
                    ),
                ]
            )

            if report.has_benchmark:
                st.markdown(f"#### Relative to {report.benchmark_label}")
                metric_row(
                    [
                        ("Annualized excess", pct(head["excess_return"]), None),
                        ("Tracking error", pct(head["tracking_error"]), None),
                        (
                            "Information ratio",
                            num(head["information_ratio"]),
                            "Excess return per unit of tracking error.",
                        ),
                        ("Beta", num(head["beta"]), None),
                        (
                            "Alpha (CAPM)",
                            pct(head["alpha"]),
                            f"t-statistic {head['alpha_t_stat']:.2f}. Below |2| "
                            "the alpha is not distinguishable from zero.",
                        ),
                        (
                            "Active share",
                            pct(head["active_share"], 1)
                            if "active_share" in head
                            else "—",
                            "Half the sum of absolute weight differences. "
                            "0 is the benchmark; 1 shares no holding with it. "
                            "Unavailable for an external index.",
                        ),
                    ]
                )
                metric_row(
                    [
                        (
                            "Up capture",
                            num(head["up_capture"]),
                            "Share of the benchmark's gains captured.",
                        ),
                        (
                            "Down capture",
                            num(head["down_capture"]),
                            "Share of the benchmark's losses taken. Lower is "
                            "better.",
                        ),
                        (
                            "Batting average",
                            pct(head["batting_average"], 1),
                            "Share of periods that beat the benchmark. It "
                            "often disagrees with the information ratio — a "
                            "few large wins can carry a low batting average.",
                        ),
                        (
                            "Worst relative drawdown",
                            pct(head["worst_relative_drawdown"]),
                            "Furthest the portfolio has fallen behind the "
                            "benchmark since last being ahead.",
                        ),
                        (
                            "M²",
                            pct(report.relative.loc[PORTFOLIO, "M-squared"]),
                            "The portfolio's return levered to the benchmark's "
                            "risk — the Sharpe ranking, in percentage points.",
                        ),
                        (
                            "Prob. excess > 0",
                            pct(report.relative.loc[PORTFOLIO, "Prob. Excess > 0"], 1),
                            "Probability the true excess return is positive, "
                            "given this sample's shape.",
                        ),
                    ]
                )

            st.divider()
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(
                    plot_wealth_index(
                        report.returns[
                            [c for c in report.returns.columns if c != "Excess"]
                        ],
                        "Cumulative wealth (start = 1)",
                    ),
                    width="stretch",
                )
            with g2:
                if report.has_benchmark:
                    st.plotly_chart(
                        plot_relative_wealth(
                            report.returns[PORTFOLIO],
                            report.returns[BENCHMARK],
                            f"Relative to {report.benchmark_label}",
                        ),
                        width="stretch",
                    )
                else:
                    st.plotly_chart(
                        plot_return_distribution(
                            report.returns[PORTFOLIO],
                            title="Return distribution",
                        ),
                        width="stretch",
                    )

            g3, g4 = st.columns(2)
            with g3:
                st.plotly_chart(
                    plot_drawdown(
                        report.returns[
                            [c for c in report.returns.columns if c != "Excess"]
                        ],
                        "Drawdown",
                    ),
                    width="stretch",
                )
            with g4:
                st.plotly_chart(
                    plot_period_returns(
                        report.periods,
                        f"{perf_period_freq.capitalize()} returns",
                    ),
                    width="stretch",
                )

            scatter_points = summary_stats(
                returns, periods_per_year=int(periods_per_year)
            )
            scatter_points = pd.concat(
                [
                    scatter_points,
                    summary_stats(
                        report.returns[
                            [c for c in report.returns.columns if c != "Excess"]
                        ],
                        periods_per_year=int(periods_per_year),
                    ),
                ]
            )
            highlight = {PORTFOLIO: "portfolio"}
            if report.has_benchmark:
                highlight[BENCHMARK] = "benchmark"
            st.plotly_chart(
                plot_risk_return_scatter(
                    scatter_points,
                    highlight,
                    "Where the portfolio sits against its own universe",
                ),
                width="stretch",
            )

            r1, r2 = st.columns(2)
            with r1:
                st.plotly_chart(
                    plot_rolling_metrics(
                        report.rolling_absolute.dropna(how="all"),
                        f"Rolling {int(perf_window)}-period performance",
                    ),
                    width="stretch",
                )
            with r2:
                if report.rolling_relative_frame is not None:
                    st.plotly_chart(
                        plot_rolling_relative(
                            report.rolling_relative_frame.dropna(how="all"),
                            f"Rolling {int(perf_window)}-period vs. benchmark",
                        ),
                        width="stretch",
                    )
                else:
                    st.caption(
                        "Choose a benchmark in the sidebar to see the rolling "
                        "excess return, tracking error, information ratio and "
                        "beta here."
                    )

            st.divider()
            st.markdown("#### Tables")
            t_abs, t_rel, t_per, t_dd = st.tabs(
                ["Absolute", "Relative", "Periods", "Drawdowns"]
            )
            with t_abs:
                st.dataframe(format_table(report.absolute.T), width="stretch")
            with t_rel:
                if report.relative is not None:
                    st.dataframe(format_table(report.relative.T), width="stretch")
                    if report.active_share is not None:
                        st.caption(
                            f"Active share {report.active_share:.1%} — computed "
                            "from positions, so unlike tracking error it cannot "
                            "be flattered by a quiet market."
                        )
                else:
                    st.info("No benchmark set.")
            with t_per:
                st.dataframe(format_table(report.periods), width="stretch")
            with t_dd:
                st.dataframe(format_table(report.drawdowns), width="stretch")

            st.divider()
            st.markdown("#### Export")
            st.caption(
                "Everything on this page, in the two formats it gets used in."
            )
            _performance_downloads(report, "perf")

# ---------------------------------------------------------------------------
# 🆚 Compare
# ---------------------------------------------------------------------------

with tab_compare:
    if not st.session_state.scenarios:
        st.info(
            "Save at least one scenario from the **📚 Scenarios** sidebar block "
            "to compare."
        )
    else:
        names_all = sorted(st.session_state.scenarios.keys())
        # Default to everything saved (up to the cap): a comparison tab that
        # opens on a single scenario is not comparing anything.
        default_sel = names_all[:5]
        if (
            st.session_state.active_scenario in names_all
            and st.session_state.active_scenario not in default_sel
        ):
            default_sel = [st.session_state.active_scenario] + default_sel[:4]
        chosen = st.multiselect(
            "Scenarios to compare",
            options=names_all,
            default=default_sel,
            max_selections=5,
            help="Up to 5 scenarios at once.",
        )
        if not chosen:
            st.info("Pick one or more scenarios.")
        else:
            runs: dict[str, object] = {}
            for n in chosen:
                scn = st.session_state.scenarios[n]
                try:
                    sub = _scenario_returns_subset(scn, returns)
                    runs[n] = _solve_scenario_cached(
                        scenario_signature(scn), sub, external_returns
                    )
                    covered = len(
                        [c for c in returns.columns if c in scn.config.expected_returns]
                    )
                    total = len(scn.config.expected_returns)
                    if covered != total:
                        st.caption(
                            f"_{n}: covers {covered}/{total} of its assets in the "
                            "loaded data._"
                        )
                except Exception as exc:
                    st.error(f"{n}: {exc}")

            if runs:
                summary_df = pd.DataFrame(
                    [_summarize_run(n, r) for n, r in runs.items()]
                ).set_index("Scenario")
                st.dataframe(
                    summary_df.style.format(
                        {
                            "Expected Return": "{:.2%}",
                            "Expected Vol": "{:.2%}",
                            "Sharpe": "{:.3f}",
                            "Effective N": "{:.1f}",
                            "Div. ratio": "{:.2f}",
                        }
                    ),
                    width="stretch",
                )

                weights_df = pd.DataFrame(
                    {n: r.result.weights for n, r in runs.items()}
                ).fillna(0.0)
                st.plotly_chart(
                    plot_portfolio_composition(weights_df, title="Weights by scenario"),
                    width="stretch",
                )
                grouped = plot_risk_contributions(weights_df)
                grouped.update_layout(title="Per-asset weights")
                st.plotly_chart(grouped, width="stretch")

                st.markdown("**Backtest comparison**")
                st.caption(
                    "In-sample, costless, rebalanced every period — a like-for-"
                    "like comparison, not a forecast for any of them."
                )
                bt_df = pd.concat(
                    {n: r.backtest_returns()["portfolio"] for n, r in runs.items()},
                    axis=1,
                )
                st.plotly_chart(
                    plot_wealth_index(bt_df, "Backtest comparison"),
                    width="stretch",
                )


# ---------------------------------------------------------------------------
# 🎚️ What-if
# ---------------------------------------------------------------------------

with tab_whatif:
    if not st.session_state.scenarios:
        st.info("Save a scenario first; What-if needs an anchor.")
    else:
        names_all = sorted(st.session_state.scenarios.keys())
        default_idx = (
            names_all.index(st.session_state.active_scenario)
            if st.session_state.active_scenario in names_all
            else 0
        )
        anchor_name = st.selectbox(
            "Anchor scenario", options=names_all, index=default_idx,
            key="whatif_anchor",
        )

        if st.session_state.get("whatif_last_anchor") != anchor_name:
            st.session_state.whatif_overrides = {}
            st.session_state.whatif_extra = {}
            st.session_state.whatif_run = None
            st.session_state.whatif_error = None
            st.session_state.whatif_last_anchor = anchor_name

        anchor_scn = st.session_state.scenarios[anchor_name]
        anchor_cfg = anchor_scn.config
        anchor_assets = list(anchor_cfg.expected_returns.keys())
        n_assets = len(anchor_assets)
        is_slow = anchor_cfg.optimizer.name == "cvar" or n_assets > 25

        if is_slow:
            st.info(
                f"Live re-solve is off (optimizer='{anchor_cfg.optimizer.name}', "
                f"{n_assets} assets). Drag sliders, then press **Recompute**."
            )

        st.markdown("**Per-asset weight bounds**")
        cols = st.columns(2)
        overrides: dict[str, tuple[float, float]] = dict(
            st.session_state.get("whatif_overrides", {})
        )
        for i, a in enumerate(anchor_assets):
            lo0, hi0 = anchor_cfg.bounds.get(a, [0.0, 1.0])
            current = overrides.get(a, (float(lo0), float(hi0)))
            with cols[i % 2]:
                lo, hi = st.slider(
                    f"{a}", min_value=-1.0, max_value=1.5, step=0.01,
                    value=(float(current[0]), float(current[1])),
                    key=f"whatif_bnd_{anchor_name}_{a}",
                )
            overrides[a] = (lo, hi)
        st.session_state.whatif_overrides = overrides

        st.markdown("**Optimizer settings**")
        anchor_req = requirements_for(anchor_cfg.optimizer.name)
        extras: dict[str, object] = dict(st.session_state.get("whatif_extra", {}))

        if anchor_req.supports_target_return or anchor_req.supports_target_volatility:
            modes = []
            if anchor_req.supports_target_return:
                modes.append("Target return")
            if anchor_req.supports_target_volatility:
                modes.append("Target volatility")
            if anchor_req.supports_risk_aversion:
                modes.append("Utility")
            if anchor_cfg.optimizer.target_return is not None:
                default_mode = "Target return"
            elif anchor_cfg.optimizer.target_volatility is not None:
                default_mode = "Target volatility"
            else:
                default_mode = modes[0]
            wf_mode = st.radio(
                "Mode", modes,
                index=modes.index(default_mode) if default_mode in modes else 0,
                horizontal=True, key="whatif_mv_mode",
            )
            if wf_mode == "Target return":
                tr = st.number_input(
                    "Target return (annual)",
                    value=float(anchor_cfg.optimizer.target_return or 0.07),
                    step=0.005, format="%.4f", key="whatif_target_return",
                )
                extras = {"target_return": float(tr), "target_volatility": None}
            elif wf_mode == "Target volatility":
                tv = st.number_input(
                    "Target volatility (annual)",
                    value=float(anchor_cfg.optimizer.target_volatility or 0.10),
                    step=0.005, format="%.4f", key="whatif_target_vol",
                )
                extras = {"target_return": None, "target_volatility": float(tv)}
            else:
                ra = st.slider(
                    "Risk aversion λ", 0.1, 20.0,
                    float(anchor_cfg.optimizer.risk_aversion or 2.5),
                    key="whatif_risk_aversion",
                )
                extras = {
                    "target_return": None,
                    "target_volatility": None,
                    "risk_aversion": float(ra),
                }

        for extra in anchor_req.extras:
            if extra.kind == "scalar" and extra.key == "cvar_alpha":
                extras["cvar_alpha"] = float(
                    st.slider(
                        "CVaR tail probability α", 0.01, 0.20,
                        float(anchor_cfg.optimizer.cvar_alpha or 0.05), 0.01,
                        key="whatif_cvar_alpha",
                    )
                )
            elif extra.kind == "scalar" and extra.key == "bl_tau":
                extras["bl_tau"] = float(
                    st.slider(
                        "τ (prior uncertainty)", 0.01, 0.5,
                        float(anchor_cfg.optimizer.bl_tau or 0.05), 0.01,
                        key="whatif_bl_tau",
                    )
                )
            elif extra.kind == "choice" and extra.key == "hrp_linkage":
                extras["hrp_linkage"] = str(
                    st.selectbox(
                        "HRP linkage", list(extra.choices or ()),
                        index=(extra.choices or ("single",)).index(
                            anchor_cfg.optimizer.hrp_linkage or "single"
                        ),
                        key="whatif_hrp_linkage",
                    )
                )
            elif extra.kind == "per_asset" and extra.key == "risk_budget":
                rb_idx = list(anchor_cfg.expected_returns.keys())
                default_rb = anchor_cfg.optimizer.risk_budget or {
                    a: 1.0 / len(rb_idx) for a in rb_idx
                }
                rb_df = st.data_editor(
                    pd.DataFrame(
                        {"Risk Budget": [default_rb.get(a, 0.0) for a in rb_idx]},
                        index=rb_idx,
                    ),
                    num_rows="fixed",
                    column_config={
                        "Risk Budget": st.column_config.NumberColumn(
                            min_value=0.001, max_value=1.0, step=0.01, format="%.3f",
                        ),
                    },
                    key="whatif_rb_editor",
                )
                extras["risk_budget"] = rb_df["Risk Budget"].to_dict()

        st.session_state.whatif_extra = extras

        def _live_config():
            cfg_dict = anchor_cfg.to_dict()
            cfg_dict["bounds"] = {
                a: list(
                    st.session_state.whatif_overrides.get(
                        a, anchor_cfg.bounds.get(a, [0.0, 1.0])
                    )
                )
                for a in anchor_assets
            }
            opt = dict(cfg_dict["optimizer"])
            for k, v in st.session_state.whatif_extra.items():
                opt[k] = v
            cfg_dict["optimizer"] = opt
            return EngineConfig.from_dict(cfg_dict)

        should_solve = (
            st.button("Recompute", type="primary", key="whatif_recompute")
            if is_slow
            else True
        )

        if should_solve:
            try:
                cfg_live = _live_config()
                sub = returns[[c for c in returns.columns if c in anchor_assets]]
                st.session_state.whatif_run = _solve_scenario_cached(
                    config_signature(cfg_live), sub, external_returns
                )
                st.session_state.whatif_error = None
            except Exception as exc:
                st.session_state.whatif_run = None
                st.session_state.whatif_error = str(exc)

        wf_run = st.session_state.get("whatif_run")
        if st.session_state.get("whatif_error"):
            st.error(f"Solver: {st.session_state.whatif_error}")
            try:
                cfg_live = _live_config()
                sub = returns[[c for c in returns.columns if c in anchor_assets]]
                render_feasibility(
                    _feasibility_cached(
                        config_signature(cfg_live), _frame_hash(sub), sub
                    ),
                    show_ok=False,
                )
            except Exception:
                pass
            if st.button("Reset to anchor", key="whatif_reset"):
                st.session_state.whatif_overrides = {}
                st.session_state.whatif_extra = {}
                st.session_state.whatif_error = None
                st.rerun()
        elif wf_run is not None:
            metric_row(
                [
                    ("Expected Return", pct(wf_run.result.expected_return), None),
                    ("Expected Vol", pct(wf_run.result.expected_volatility), None),
                    ("Sharpe", num(wf_run.result.sharpe_ratio), None),
                    (
                        "Effective N",
                        num(wf_run.diagnostics.effective_n, 1)
                        if wf_run.diagnostics
                        else "—",
                        None,
                    ),
                ]
            )
            try:
                anchor_run = _solve_scenario_cached(
                    scenario_signature(anchor_scn),
                    _scenario_returns_subset(anchor_scn, returns),
                    external_returns,
                )
                weights_df = pd.DataFrame(
                    {
                        "Anchor": anchor_run.result.weights,
                        "What-if": wf_run.result.weights,
                    }
                ).fillna(0.0)
                delta = weights_df["What-if"] - weights_df["Anchor"]
                st.caption(
                    f"One-way turnover versus the anchor: {delta.abs().sum():.2%}"
                )
            except Exception:
                weights_df = pd.DataFrame({"What-if": wf_run.result.weights}).fillna(0.0)
            st.plotly_chart(
                plot_portfolio_composition(weights_df, title="Anchor vs. What-if"),
                width="stretch",
            )

            if st.button("Save these as new scenario", key="whatif_save"):
                st.session_state["whatif_save_pending"] = True

            if st.session_state.get("whatif_save_pending"):
                new_name = st.text_input(
                    "Scenario name", key="whatif_save_name",
                    placeholder="e.g. Anchor + tighter equities",
                )
                cA, cB = st.columns(2)
                if cA.button("Save", key="whatif_save_confirm", type="primary"):
                    if not new_name:
                        st.error("Name is required.")
                    elif new_name in st.session_state.scenarios:
                        st.error(f"Scenario {new_name!r} already exists.")
                    else:
                        st.session_state.scenarios[new_name] = Scenario(
                            name=new_name,
                            config=_live_config(),
                            notes=f"Forked from {anchor_name}",
                            created_at=now_iso(),
                            updated_at=now_iso(),
                        )
                        st.session_state.active_scenario = new_name
                        st.session_state["whatif_save_pending"] = False
                        st.success(f"Saved as {new_name!r}.")
                        st.rerun()
                if cB.button("Cancel", key="whatif_save_cancel"):
                    st.session_state["whatif_save_pending"] = False
                    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar — 📚 Scenarios (rendered after _build_config exists)
# ---------------------------------------------------------------------------

with st.sidebar.expander("📚 Scenarios", expanded=False):
    names_in_state = sorted(st.session_state.scenarios.keys())
    sel_options = ["—"] + names_in_state
    default_idx = (
        sel_options.index(st.session_state.active_scenario)
        if st.session_state.active_scenario in names_in_state
        else 0
    )
    selected = st.selectbox(
        "Active scenario", options=sel_options, index=default_idx, key="scn_select"
    )
    has_selection = selected != "—"

    new_name = st.text_input("Name", key="scn_new_name").strip()
    notes = st.text_area(
        "Notes (optional)", key="scn_notes", height=70, max_chars=NOTES_MAX_LEN
    )
    cA, cB = st.columns(2)
    save_clicked = cA.button("Save", width="stretch", key="scn_save")
    update_clicked = cB.button(
        "Update", disabled=not has_selection, width="stretch", key="scn_update"
    )
    cC, cD = st.columns(2)
    load_clicked = cC.button(
        "Load", disabled=not has_selection, width="stretch", key="scn_load"
    )
    delete_clicked = cD.button(
        "Delete", disabled=not has_selection, width="stretch", key="scn_delete"
    )

    rename_to = st.text_input(
        "Rename to", key="scn_rename_to", disabled=not has_selection
    ).strip()
    rename_clicked = st.button(
        "Rename", disabled=(not has_selection) or (not rename_to), key="scn_rename"
    )

    st.divider()
    st.download_button(
        "⬇ Download all (YAML)",
        data=dump_scenarios_yaml(st.session_state.scenarios)
        if st.session_state.scenarios
        else "",
        file_name="scenarios.yaml",
        mime="text/yaml",
        disabled=not st.session_state.scenarios,
    )
    upl = st.file_uploader(
        "⬆ Upload scenarios YAML", type=["yaml", "yml"], key="scn_upload"
    )
    merge_mode = st.radio(
        "On name collision",
        options=["Skip", "Overwrite", "Suffix"],
        horizontal=True,
        key="scn_merge_mode",
    )

    if save_clicked:
        if not new_name:
            st.error("Name is required.")
        elif new_name in st.session_state.scenarios:
            st.error(
                f"Scenario {new_name!r} already exists. Use Update or pick "
                "another name."
            )
        else:
            ts = now_iso()
            st.session_state.scenarios[new_name] = Scenario(
                name=new_name, config=_build_config(), notes=notes,
                created_at=ts, updated_at=ts,
            )
            st.session_state.active_scenario = new_name
            st.success(f"Saved scenario {new_name!r}.")
            st.rerun()

    if update_clicked and has_selection:
        prev = st.session_state.scenarios[selected]
        st.session_state.scenarios[selected] = Scenario(
            name=selected,
            config=_build_config(),
            notes=notes or prev.notes,
            created_at=prev.created_at or now_iso(),
            updated_at=now_iso(),
        )
        st.session_state.last_run_by_scenario.pop(selected, None)
        st.success(f"Updated {selected!r}.")
        st.rerun()

    if load_clicked and has_selection:
        target_cfg = st.session_state.scenarios[selected].config
        loaded_assets = set(returns.columns)
        missing = sorted(set(target_cfg.expected_returns) - loaded_assets)
        if missing:
            kept = [a for a in target_cfg.expected_returns if a in loaded_assets]
            if not kept:
                st.error(
                    f"Cannot load {selected!r}: none of its assets are in the "
                    "loaded data."
                )
            else:
                trimmed_dict = target_cfg.to_dict()
                trimmed_dict["expected_returns"] = {
                    a: trimmed_dict["expected_returns"][a] for a in kept
                }
                trimmed_dict["bounds"] = {
                    a: trimmed_dict["bounds"].get(a, [0.0, 1.0]) for a in kept
                }
                trimmed_dict["groups"] = {
                    a: trimmed_dict["groups"].get(a, "Other") for a in kept
                }
                trimmed_dict["currencies"] = {
                    a: trimmed_dict["currencies"].get(a, trimmed_dict["base_currency"])
                    for a in kept
                }
                prev = st.session_state.scenarios[selected]
                st.session_state.scenarios[selected] = Scenario(
                    name=selected,
                    config=EngineConfig.from_dict(trimmed_dict),
                    notes=prev.notes,
                    created_at=prev.created_at,
                    updated_at=prev.updated_at,
                )
                st.session_state.scenario_load_warning = (
                    f"Loaded {selected!r}; dropped {len(missing)} missing "
                    "asset(s): " + ", ".join(missing)
                )
                st.session_state.pending_scenario_load = selected
                st.rerun()
        else:
            st.session_state.pending_scenario_load = selected
            st.rerun()

    if delete_clicked and has_selection:
        st.session_state.scenarios = _delete_scenario(
            st.session_state.scenarios, selected
        )
        st.session_state.last_run_by_scenario.pop(selected, None)
        if st.session_state.active_scenario == selected:
            st.session_state.active_scenario = None
        st.success(f"Deleted {selected!r}.")
        st.rerun()

    if rename_clicked and has_selection and rename_to:
        try:
            st.session_state.scenarios = _rename_scenario(
                st.session_state.scenarios, selected, rename_to
            )
            old_run = st.session_state.last_run_by_scenario.pop(selected, None)
            if old_run is not None:
                st.session_state.last_run_by_scenario[rename_to] = old_run
            if st.session_state.active_scenario == selected:
                st.session_state.active_scenario = rename_to
            st.success(f"Renamed to {rename_to!r}.")
            st.rerun()
        except (KeyError, ValueError) as exc:
            st.error(str(exc))

    if upl is not None:
        try:
            incoming = load_scenarios_yaml(upl.read().decode("utf-8"))
            applied = 0
            for k, v in incoming.items():
                if k in st.session_state.scenarios:
                    if merge_mode == "Skip":
                        continue
                    if merge_mode == "Overwrite":
                        st.session_state.scenarios[k] = v
                        st.session_state.last_run_by_scenario.pop(k, None)
                        applied += 1
                    else:
                        suffix = 2
                        candidate = f"{k} ({suffix})"
                        while candidate in st.session_state.scenarios:
                            suffix += 1
                            candidate = f"{k} ({suffix})"
                        st.session_state.scenarios[candidate] = Scenario(
                            name=candidate, config=v.config, notes=v.notes,
                            created_at=v.created_at, updated_at=v.updated_at,
                        )
                        applied += 1
                else:
                    st.session_state.scenarios[k] = v
                    applied += 1
            st.success(f"Uploaded {applied} scenario(s).")
        except Exception as exc:
            st.error(f"Upload failed: {exc}")


# ---------------------------------------------------------------------------
# 📤 Report
# ---------------------------------------------------------------------------

with tab_report:
    run = st.session_state.get("last_run")
    if run is None:
        st.info("Run an optimization first.")
    else:
        st.subheader("Export")
        st.caption(
            "The workbook carries the assumptions and diagnostics alongside "
            "the weights, so a reader can see what the numbers rest on."
        )

        assumptions = run.assumptions()
        report = None
        try:
            report = run.performance(riskfree_rate=float(risk_free_rate))
        except (ValueError, BenchmarkError) as exc:
            st.warning(f"Performance sheets omitted: {exc}")
        # Same builder the CLI uses, so a workbook exported from the app and
        # one written by `optengine optimize` carry identical sheets.
        sheets = run_sheets(
            run,
            riskfree_rate=float(risk_free_rate),
            data_quality=quality,
            walk_forward=st.session_state.get("walk_forward"),
            frontier_uncertainty=st.session_state.get("frontier_uncertainty"),
            performance=report,
        )

        buf = io.BytesIO()
        # Excel caps sheet names at 31 characters and two long names can
        # truncate onto each other, so the shared de-duplicating writer is
        # used rather than a bare slice that would silently drop a sheet.
        _used: set[str] = set()
        with excel_writer(buf) as writer:
            for name, df in sheets.items():
                if df is None:
                    continue
                sheet = unique_sheet_name(name, _used)
                _used.add(sheet)
                df.to_excel(writer, sheet_name=sheet, index=True)
        buf.seek(0)
        dl_left, dl_right = st.columns(2)
        dl_left.download_button(
            label="📥 Download Excel report",
            data=buf,
            file_name="optimization_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            width="stretch",
        )
        dl_right.download_button(
            label="📄 Weights (CSV)",
            data=(
                run.result.weights.to_frame("weight")
                .to_csv()
                .encode("utf-8")
            ),
            file_name="weights.csv",
            mime="text/csv",
            width="stretch",
        )
        st.caption(f"{len(sheets)} sheets: {', '.join(sheets)}")

        if report is not None:
            st.divider()
            st.markdown("**Performance**")
            st.caption(report.describe())
            _performance_downloads(report, "report_tab")

        st.markdown("**Assumptions**")
        render_assumptions(assumptions)

        st.markdown("**Config used (YAML)**")
        import yaml as _yaml

        st.code(
            _yaml.safe_dump(run.config.to_dict(), sort_keys=False), language="yaml"
        )
