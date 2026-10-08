"""Plotting helpers (Plotly).

Plotly is preferred over Matplotlib here because the same figure objects
render natively in Streamlit and in Jupyter without extra setup.

A note on the frontier chart: it takes a :class:`FrontierResult` rather than
a bare summary frame. Passing the frame alone is how the "max Sharpe" star
ends up on the wrong point — the caller drops NaN rows for plotting, then
highlights a position computed against the undropped frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from optimization_engine._optional import LazyModule

#: Plotly is an optional extra (``viz``). Bound lazily so that importing
#: this module — which the reporting package does eagerly — costs nothing
#: until a figure is actually built. Return annotations naming ``go.Figure``
#: are strings under ``from __future__ import annotations``, so they never
#: trigger the import either.
px = LazyModule("plotly.express", extra="viz", purpose="plotting")
go = LazyModule("plotly.graph_objects", extra="viz", purpose="plotting")

#: Categorical palette, in fixed slot order. The ordering is the
#: colour-vision-deficiency safety mechanism, not cosmetics: every *adjacent*
#: pair has to stay separable, because adjacent slots are what land next to each
#: other in a stack or a grouped bar.
#:
#: Validated against a white chart surface — lightness band, chroma floor,
#: adjacent-pair CVD separation (worst ΔE 9.1, protan) and normal-vision floor
#: (worst ΔE 19.6) all pass. The previous Tableau-10 order failed: its red
#: (#E45756) and green (#54A24B) sat in adjacent slots at ΔE 1.2 under
#: deuteranopia — indistinguishable to roughly one man in twelve, and they were
#: neighbours in every stacked weight chart.
PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

#: Anything past the categorical slots folds into this neutral rather than
#: cycling back to slot 1. A repeated hue in a 13-asset stack is a lie about
#: identity; a labelled "Other" band is not.
OTHER_COLOR = "#898781"

#: How :func:`fold_to_slots` names the folded band. Colour is assigned by label
#: rather than by position so the neutral tracks the band itself — the folded
#: band sits at whatever index the fold leaves it, not reliably past the slots.
OTHER_LABEL_PREFIX = "Other ("

#: Frontier-anchor field names, as an analyst reads them. Keyed by the
#: ``FrontierResult`` field the anchor lands in, which is also how
#: ``anchor_failures`` is keyed.
_ANCHOR_LABELS = {
    "min_variance": "Minimum variance",
    "tangency": "Maximum Sharpe",
}

#: Recessive furniture, so the data carries the ink.
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

_LAYOUT = dict(
    template="plotly_white",
    colorway=PALETTE,
    margin=dict(l=60, r=30, t=60, b=50),
    hoverlabel=dict(bgcolor="white", font_size=12),
    font=dict(color="#0b0b0b"),
)


def series_color(i: int, label: object = None) -> str:
    """Colour for the ``i``-th series in fixed order.

    The folded "Other" band and anything past the categorical slots take the
    neutral: a hue is only ever spent on a series that means one thing.

    Args:
        i: Zero-based series index.
        label: The series' name. ``"Other"`` always takes the neutral,
            whatever its index.

    Returns:
        A CSS colour string.
    """
    if label is not None and str(label).startswith(OTHER_LABEL_PREFIX):
        return OTHER_COLOR
    return PALETTE[i] if i < len(PALETTE) else OTHER_COLOR


def fold_to_slots(
    frame: pd.DataFrame, max_series: int = len(PALETTE), axis: int = 0
) -> pd.DataFrame:
    """Keep the largest ``max_series - 1`` rows and sum the rest into "Other".

    Stacked charts assign adjacent palette slots to adjacent bands, so a
    universe larger than the palette would otherwise cycle and paint two
    different assets the same colour. Folding the small tail into one labelled
    neutral band keeps every colour meaning exactly one thing; the full detail
    stays in the weights table and the Excel export.

    Args:
        frame: Assets × series weights.
        max_series: Number of distinct bands to keep, "Other" included.
        axis: ``0`` to fold rows (the usual assets-in-rows layout).

    Returns:
        The frame unchanged when it already fits, otherwise the top rows plus
        an ``Other`` row.
    """
    if axis != 0:
        return fold_to_slots(frame.T, max_series=max_series).T
    if len(frame) <= max_series:
        return frame
    ranked = frame.abs().sum(axis=1).sort_values(ascending=False)
    keep = list(ranked.index[: max_series - 1])
    rest = [i for i in frame.index if i not in keep]
    folded = frame.loc[keep].copy()
    folded.loc[f"{OTHER_LABEL_PREFIX}{len(rest)} assets)"] = frame.loc[rest].sum()
    return folded


def plot_efficient_frontier(
    frontier,
    highlight_index: int | None = None,
    title: str = "Efficient Frontier",
    risk_free_rate: float | None = None,
    current_portfolio: tuple[float, float, str] | None = None,
    show_dominated: bool = False,
) -> go.Figure:
    """Draw a frontier with its anchor portfolios and, optionally, the CAL.

    An anchor portfolio that failed to solve is named in a footnote below the
    legend rather than left off the chart in silence — a missing marker and a
    marker that was never requested look identical otherwise.

    Args:
        frontier: A ``FrontierResult``. A bare summary DataFrame is still
            accepted for backwards compatibility, but then the anchors and
            the anchor-failure footnote cannot be drawn.
        highlight_index: Position within the plotted frame to star. Defaults
            to the highest-Sharpe point.
        title: Chart title.
        risk_free_rate: When given, draws the capital allocation line from
            ``rf`` through the tangency portfolio — the set of risk/return
            combinations reachable by mixing it with cash.
        current_portfolio: ``(risk, return, label)`` for the allocation
            actually chosen, so the analyst can see where it sits relative to
            the frontier. ``risk`` must be measured on the same axis the
            frontier uses — pass ``None`` rather than a volatility when the
            frontier is drawn against CVaR.
        show_dominated: **Deprecated and ignored.** It used to draw the
            branch below the minimum-variance point. The mean-variance sweep
            imposes its return target as ``μ'w ≥ R*``, so every target below
            the minimum-variance return now resolves to the minimum-variance
            portfolio itself and there is no dominated branch left to draw.
            Accepted so existing callers keep working; remove after 0.6.x.
    """
    is_result = hasattr(frontier, "plot_frame")
    if is_result:
        df = frontier.plot_frame(efficient_only=True)
    else:
        # The full summary was also kept here, to filter the dominated branch
        # out of. There is no dominated branch any more, so the plotted frame
        # is the only one this function needs.
        df = frontier.dropna(
            subset=["expected_volatility", "expected_return"]
        ).reset_index(drop=True)

    # Plot the risk measure the frontier was actually traced against. Drawing
    # a mean-CVaR frontier on a volatility axis shows a curve that was never
    # optimized, and the shapes differ wherever returns are skewed.
    risk_measure = getattr(frontier, "risk_measure", "volatility") if is_result else "volatility"
    if risk_measure == "CVaR" and "cvar" in df.columns and df["cvar"].notna().any():
        risk_col = "cvar"
        risk_label = "Conditional VaR (annualized)"
        # The minimum-variance and tangency anchors, and the capital
        # allocation line, are defined in volatility space. They have no
        # position on a CVaR axis, so they are not drawn there.
        show_anchors = False
    else:
        risk_col = "expected_volatility"
        risk_label = "Volatility (annualized)"
        show_anchors = True

    fig = go.Figure()

    if not df.empty:
        fig.add_trace(
            go.Scatter(
                x=df[risk_col],
                y=df["expected_return"],
                mode="markers+lines",
                line=dict(color=PALETTE[0], width=2),
                marker=dict(
                    size=8,
                    color=df["sharpe_ratio"],
                    colorscale="Viridis",
                    colorbar=dict(title="Sharpe", thickness=12),
                    showscale=True,
                ),
                name="Efficient frontier",
                hovertemplate=(
                    f"{risk_label.split(' (')[0]}: %{{x:.2%}}<br>"
                    "Return: %{y:.2%}<br>"
                    "Sharpe: %{marker.color:.3f}<extra></extra>"
                ),
            )
        )

        has_exact_tangency = is_result and show_anchors and frontier.tangency is not None
        if highlight_index is None and not has_exact_tangency and "sharpe_ratio" in df.columns:
            valid = df["sharpe_ratio"].dropna()
            highlight_index = int(valid.idxmax()) if len(valid) else None
        if highlight_index is not None and 0 <= highlight_index < len(df):
            row = df.iloc[highlight_index]
            fig.add_trace(
                go.Scatter(
                    x=[row[risk_col]],
                    y=[row["expected_return"]],
                    mode="markers",
                    marker=dict(size=16, color=PALETTE[7], symbol="star"),
                    name="Max Sharpe on frontier",
                    hovertemplate=(
                        "Max Sharpe<br>Vol: %{x:.2%}<br>Return: %{y:.2%}"
                        "<extra></extra>"
                    ),
                )
            )

    if is_result and show_anchors:
        for anchor, symbol, color in (
            (frontier.min_variance, "diamond", PALETTE[5]),
            (frontier.tangency, "circle", PALETTE[1]),
        ):
            if anchor is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[anchor["expected_volatility"]],
                    y=[anchor["expected_return"]],
                    mode="markers",
                    marker=dict(size=13, color=color, symbol=symbol,
                                line=dict(width=1.5, color="white")),
                    name=str(anchor["label"]),
                    hovertemplate=(
                        f"{anchor['label']}<br>Vol: %{{x:.2%}}<br>"
                        "Return: %{y:.2%}<extra></extra>"
                    ),
                )
            )

        if risk_free_rate is not None and frontier.tangency is not None:
            tan_vol = float(frontier.tangency["expected_volatility"])
            tan_ret = float(frontier.tangency["expected_return"])
            if tan_vol > 0:
                x_max = float(
                    max(
                        df["expected_volatility"].max() if not df.empty else tan_vol,
                        tan_vol,
                    )
                    * 1.15
                )
                slope = (tan_ret - risk_free_rate) / tan_vol
                fig.add_trace(
                    go.Scatter(
                        x=[0.0, x_max],
                        y=[risk_free_rate, risk_free_rate + slope * x_max],
                        mode="lines",
                        line=dict(dash="dash", color=PALETTE[1], width=1.5),
                        name=f"Capital allocation line (slope {slope:.2f})",
                        hovertemplate=(
                            "Vol: %{x:.2%}<br>Return: %{y:.2%}<br>"
                            "<i>Mix of cash and the tangency portfolio</i>"
                            "<extra></extra>"
                        ),
                    )
                )

    if current_portfolio is not None:
        vol, ret, label = current_portfolio
        fig.add_trace(
            go.Scatter(
                x=[vol], y=[ret],
                mode="markers+text",
                marker=dict(size=15, color=PALETTE[4], symbol="x-thin",
                            line=dict(width=3, color=PALETTE[4])),
                text=[label], textposition="bottom center",
                name=label,
                hovertemplate=(
                    f"{label}<br>Vol: %{{x:.2%}}<br>Return: %{{y:.2%}}"
                    "<extra></extra>"
                ),
            )
        )

    anchor_failures = getattr(frontier, "anchor_failures", None) if is_result else None
    if anchor_failures:
        fig.add_annotation(
            text="<br>".join(
                f"<b>{_ANCHOR_LABELS.get(name, name)} not drawn:</b> {reason}"
                for name, reason in anchor_failures.items()
            ),
            xref="paper",
            yref="paper",
            # The legend sits at y=-0.18; the footnote has to clear it, or the
            # one line explaining a missing marker lands underneath the legend
            # entry for the marker that is there.
            x=0,
            y=-0.30,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            align="left",
            font=dict(size=11, color=MUTED),
        )

    # ``_LAYOUT`` is applied last, so the extra bottom margin the footnote
    # needs has to go in with it rather than in an earlier call it would undo.
    layout = dict(_LAYOUT)
    if anchor_failures:
        layout["margin"] = {**layout["margin"], "b": 120}

    fig.update_layout(
        title=title,
        xaxis_title=risk_label,
        yaxis_title="Expected Return (annualized)",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        legend=dict(orientation="h", y=-0.18, x=0),
        **layout,
    )
    return fig


def plot_portfolio_composition(
    weights: pd.DataFrame, title: str = "Portfolio Composition", as_percent: bool = True
) -> go.Figure:
    """Stacked weights, one bar per column of ``weights``.

    Args:
        weights: Assets down the index, one column per portfolio to compare.
        title: Figure title.
        as_percent: Label the axis in percent rather than as fractions.

    Returns:
        A Plotly figure.

    Raises:
        MissingDependencyError: If plotly is not installed. Install it with
            ``finport-optengine[viz]``.
    """
    folded = fold_to_slots(weights)
    df = folded.T.copy()
    if as_percent:
        df = df * 100.0
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_trace(
            go.Bar(
                name=str(col),
                x=df.index.astype(str),
                y=df[col],
                marker=dict(
                    color=series_color(i, col),
                    # A hairline of surface between stacked bands, so adjacent
                    # segments read as separate even at small sizes.
                    line=dict(width=1, color="white"),
                ),
                hovertemplate=f"{col}<br>%{{x}}: %{{y:.2f}}%<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="",
        yaxis_title="Weight (%)" if as_percent else "Weight",
        legend=dict(orientation="v", x=1.02, y=1),
        **_LAYOUT,
    )
    return fig


def plot_weights_bar(
    weights: pd.Series, title: str = "Weights", bounds: pd.DataFrame | None = None
) -> go.Figure:
    """Horizontal weight bars, optionally showing where each bound binds.

    Seeing which positions are pinned to a cap is the fastest way to tell
    whether the constraints or the optimizer produced the answer.

    Args:
        weights: One weight per asset, as fractions of the book.
        title: Figure title.
        bounds: A frame indexed by asset with ``min`` and ``max`` columns.
            When given, the bounds are drawn behind the bars.

    Returns:
        A Plotly figure.

    Raises:
        MissingDependencyError: If plotly is not installed. Install it with
            ``finport-optengine[viz]``.
    """
    w = weights.sort_values()
    colors = [PALETTE[0] if v >= 0 else PALETTE[7] for v in w.values]
    at_bound: list[str] = []
    if bounds is not None:
        for asset, value in w.items():
            lo = float(bounds.loc[asset, "Min Weight"])
            hi = float(bounds.loc[asset, "Max Weight"])
            if abs(value - hi) < 1e-4:
                at_bound.append("at max")
            elif abs(value - lo) < 1e-4 and lo > 0:
                at_bound.append("at min")
            else:
                at_bound.append("")
    fig = go.Figure(
        go.Bar(
            x=w.values * 100,
            y=[str(i) for i in w.index],
            orientation="h",
            marker_color=colors,
            text=[
                f"{v:.1%}" + (f"  ({b})" if b else "")
                for v, b in zip(w.values, at_bound or [""] * len(w))
            ],
            textposition="auto",
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=title, xaxis_title="Weight (%)", yaxis_title="", showlegend=False,
        height=max(320, 26 * len(w)), **_LAYOUT,
    )
    return fig


def plot_risk_contributions(rc: pd.DataFrame, title: str = "Risk Contributions") -> go.Figure:
    """Grouped bars of risk shares (or any per-asset frame).

    Args:
        rc: Assets down the index, one column per series to compare.
        title: Figure title.

    Returns:
        A Plotly figure.

    Raises:
        MissingDependencyError: If plotly is not installed. Install it with
            ``finport-optengine[viz]``.
    """
    df = rc * 100.0
    fig = px.bar(df, barmode="group", title=title, color_discrete_sequence=PALETTE)
    fig.update_layout(yaxis_title="% of risk", xaxis_title="Asset", **_LAYOUT)
    return fig


def plot_weight_vs_risk(
    decomposition: pd.DataFrame, title: str = "Capital vs. risk"
) -> go.Figure:
    """Capital weight beside risk share for each asset.

    The gap between the two bars is the whole argument for risk parity: a
    position can be small in capital and large in risk.

    Args:
        decomposition: A risk decomposition frame, as returned by
            :func:`~optimization_engine.optimizers.diagnostics.risk_decomposition`.
        title: Figure title.

    Returns:
        A Plotly figure.

    Raises:
        MissingDependencyError: If plotly is not installed. Install it with
            ``finport-optengine[viz]``.
    """
    df = decomposition.sort_values("share_of_risk", ascending=False)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Capital weight", x=[str(i) for i in df.index],
            y=df["weight"] * 100, marker_color=PALETTE[0],
            hovertemplate="%{x}<br>Capital: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Risk share", x=[str(i) for i in df.index],
            y=df["share_of_risk"] * 100, marker_color=PALETTE[1],
            hovertemplate="%{x}<br>Risk: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        barmode="group", title=title, yaxis_title="% of portfolio",
        xaxis_title="", legend=dict(orientation="h", y=-0.25), **_LAYOUT,
    )
    return fig


def plot_wealth_index(returns: pd.DataFrame, title: str = "Wealth Index") -> go.Figure:
    """Cumulative growth of 1 unit.

    Args:
        returns: Periodic returns, one column per series to compare.
        title: Figure title.

    Returns:
        A Plotly figure.

    Raises:
        MissingDependencyError: If plotly is not installed. Install it with
            ``finport-optengine[viz]``.
    """
    wealth = (1 + returns).cumprod()
    fig = px.line(wealth, title=title, color_discrete_sequence=PALETTE)
    fig.update_layout(
        yaxis_title="Wealth (start = 1)", xaxis_title="", **_LAYOUT
    )
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>")
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    """Correlation matrix with values annotated for small universes.

    Args:
        corr: A square correlation matrix.
        title: Figure title.

    Returns:
        A Plotly figure.

    Raises:
        MissingDependencyError: If plotly is not installed. Install it with
            ``finport-optengine[viz]``.
    """
    fig = px.imshow(
        corr.values,
        x=corr.columns,
        y=corr.index,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=title,
        aspect="auto",
        text_auto=".2f" if len(corr) <= 15 else False,
    )
    fig.update_layout(**_LAYOUT)
    fig.update_traces(
        hovertemplate="%{y} vs %{x}<br>ρ = %{z:.3f}<extra></extra>"
    )
    return fig


def plot_drawdown(returns: pd.Series | pd.DataFrame, title: str = "Drawdown") -> go.Figure:
    """Underwater chart.

    The peak starts at the capital invested, as in
    :func:`~optimization_engine.analytics.risk.drawdown_series`, so a series
    that opens with a loss is drawn underwater from its first period.

    Args:
        returns: Periodic returns, or a frame of them.
        title: Figure title.

    Returns:
        A Plotly figure.

    Raises:
        MissingDependencyError: If plotly is not installed. Install it with
            ``finport-optengine[viz]``.
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    dd = (wealth - peak) / peak
    fig = px.area(dd, title=title, color_discrete_sequence=PALETTE)
    fig.update_layout(
        yaxis_title="Drawdown", yaxis_tickformat=".0%", xaxis_title="", **_LAYOUT
    )
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2%}<extra></extra>")
    return fig


def plot_rolling_metrics(
    rolling: pd.DataFrame, title: str = "Rolling performance"
) -> go.Figure:
    """Rolling return, volatility, Sharpe and drawdown, in stacked panels.

    A full-sample Sharpe cannot distinguish a strategy that worked throughout
    from one that earned everything in a single quarter. This can.

    The panels are stacked rather than overlaid on twin axes: a Sharpe series
    and an annualized-return series happen to occupy similar numeric ranges,
    so on one plot they trace nearly the same path and read as a duplicate
    line rather than two different measurements.

    Args:
        rolling: The frame from
            :func:`~optimization_engine.analytics.performance.rolling_metrics`.
        title: Figure title.

    Returns:
        A Plotly figure.

    Raises:
        MissingDependencyError: If plotly is not installed. Install it with
            ``finport-optengine[viz]``.
    """
    from plotly.subplots import make_subplots

    panels = [
        ("rolling_return", "Return", ".1%", ".0%"),
        ("rolling_volatility", "Volatility", ".1%", ".0%"),
        ("rolling_sharpe", "Sharpe", ".2f", ".1f"),
        ("rolling_drawdown", "Drawdown", ".1%", ".0%"),
    ]
    present = [p for p in panels if p[0] in rolling.columns]
    if not present:
        return go.Figure()

    fig = make_subplots(
        rows=len(present), cols=1, shared_xaxes=True, vertical_spacing=0.045,
        subplot_titles=[p[1] for p in present],
    )
    for row, (col, label, hover_fmt, tick_fmt) in enumerate(present, start=1):
        fig.add_trace(
            go.Scatter(
                x=rolling.index,
                y=rolling[col],
                name=label,
                line=dict(color=PALETTE[(row - 1) % len(PALETTE)], width=1.6),
                fill="tozeroy" if col == "rolling_drawdown" else None,
                hovertemplate=(
                    f"{label}<br>%{{x|%Y-%m-%d}}: %{{y:{hover_fmt}}}<extra></extra>"
                ),
            ),
            row=row, col=1,
        )
        fig.update_yaxes(tickformat=tick_fmt, row=row, col=1)
        if col == "rolling_sharpe":
            fig.add_hline(y=0, line_width=1, line_color=MUTED, row=row, col=1)

    fig.update_layout(
        title=title,
        showlegend=False,
        height=170 * len(present) + 90,
        **_LAYOUT,
    )
    fig.update_annotations(font_size=12)
    return fig


def plot_walk_forward_comparison(
    in_sample: pd.Series,
    out_of_sample: pd.Series,
    title: str = "In-sample vs. out-of-sample",
) -> go.Figure:
    """Fitted and walk-forward wealth curves on one axis.

    The distance between the lines is how much of the backtest was hindsight.

    Args:
        in_sample: The in-sample return stream.
        out_of_sample: The walk-forward stream from the same process.
        title: Figure title.

    Returns:
        A Plotly figure.

    Raises:
        MissingDependencyError: If plotly is not installed. Install it with
            ``finport-optengine[viz]``.
    """
    fig = go.Figure()
    for name, series, color, dash in (
        ("In-sample (fitted)", in_sample, PALETTE[0], None),
        ("Out-of-sample (walk-forward)", out_of_sample, PALETTE[3], "dash"),
    ):
        wealth = (1 + series).cumprod()
        fig.add_trace(
            go.Scatter(
                x=wealth.index, y=wealth.values, name=name,
                line=dict(color=color, width=2, dash=dash),
                hovertemplate=f"{name}<br>%{{x|%Y-%m-%d}}: %{{y:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title, yaxis_title="Wealth (start = 1)", xaxis_title="",
        legend=dict(orientation="h", y=-0.2), **_LAYOUT,
    )
    return fig


def plot_weight_evolution(
    weights: pd.DataFrame, title: str = "Weights through time"
) -> go.Figure:
    """Stacked area of held weights over a backtest.

    Args:
        weights: Dates down the index, one column per asset.
        title: Figure title.

    Returns:
        A Plotly figure.

    Raises:
        MissingDependencyError: If plotly is not installed. Install it with
            ``finport-optengine[viz]``.
    """
    folded = fold_to_slots(weights, axis=1)
    fig = go.Figure()
    for i, col in enumerate(folded.columns):
        fig.add_trace(
            go.Scatter(
                x=folded.index, y=folded[col] * 100,
                name=str(col), mode="lines", stackgroup="one",
                line=dict(width=0.5, color=series_color(i, col)),
                fillcolor=series_color(i, col),
                hovertemplate=f"{col}<br>%{{x|%Y-%m-%d}}: %{{y:.2f}}%<extra></extra>",
            )
        )
    fig.update_layout(
        title=title, yaxis_title="Weight (%)", xaxis_title="",
        legend=dict(orientation="v", x=1.02, y=1), **_LAYOUT,
    )
    return fig


def plot_return_distribution(
    returns: pd.Series,
    var: float | None = None,
    cvar: float | None = None,
    title: str = "Return distribution",
) -> go.Figure:
    """Histogram of periodic returns with the VaR and CVaR thresholds marked.

    Seeing where the tail cut falls relative to the actual histogram is what
    makes a CVaR number mean something.

    Args:
        returns: A periodic return stream.
        var: Value at Risk as a positive fraction, drawn as a vertical line.
            ``None`` omits it.
        cvar: Conditional VaR as a positive fraction, drawn likewise.
        title: Figure title.

    Returns:
        A Plotly figure.

    Raises:
        MissingDependencyError: If plotly is not installed. Install it with
            ``finport-optengine[viz]``.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=returns.values, nbinsx=60, marker_color=PALETTE[0],
            name="Returns", opacity=0.85,
            hovertemplate="%{x:.2%}<br>count %{y}<extra></extra>",
        )
    )
    for value, label, color in (
        (var, "VaR", PALETTE[1]),
        (cvar, "CVaR", PALETTE[3]),
    ):
        if value is None or not np.isfinite(value):
            continue
        fig.add_vline(
            x=-abs(value), line_dash="dash", line_color=color,
            annotation_text=f"{label} {abs(value):.2%}",
            annotation_position="top left",
        )
    fig.update_layout(
        title=title, xaxis_title="Periodic return", yaxis_title="Frequency",
        xaxis_tickformat=".1%", showlegend=False, **_LAYOUT,
    )
    return fig


def plot_frontier_uncertainty(
    uncertainty,
    title: str = "How much can you trust this frontier?",
) -> go.Figure:
    """The efficient frontier as a confidence band rather than a single line.

    The point estimate is drawn on top of the quantile band from resampled
    histories. Where the band is wide, two portfolios that look different on
    the point-estimate curve are not distinguishable given the data.

    Args:
        uncertainty: A
            :class:`~optimization_engine.resampling.FrontierUncertainty`.
        title: Figure title.

    Returns:
        A Plotly figure.

    Raises:
        MissingDependencyError: If plotly is not installed. Install it with
            ``finport-optengine[viz]``.
    """
    q = uncertainty.quantiles
    x = list(q.index)
    fig = go.Figure()

    for lo, hi, opacity, label in (
        ("q05", "q95", 0.14, "5th-95th percentile"),
        ("q25", "q75", 0.24, "25th-75th percentile"),
    ):
        if lo not in q.columns or hi not in q.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=list(q[hi]) + list(q[lo])[::-1],
                fill="toself",
                fillcolor=f"rgba(42, 120, 214, {opacity})",
                line=dict(width=0),
                name=label,
                hoverinfo="skip",
            )
        )

    if "q50" in q.columns:
        fig.add_trace(
            go.Scatter(
                x=x, y=q["q50"], mode="lines",
                line=dict(color=PALETTE[0], width=2, dash="dot"),
                name="Median across draws",
                hovertemplate="Vol: %{x:.2%}<br>Median return: %{y:.2%}<extra></extra>",
            )
        )

    point = uncertainty.point_estimate.plot_frame()
    if not point.empty and x:
        # Clip the point estimate to the band's range. The band spans only the
        # risk levels *every* draw could reach, so drawing the full
        # point-estimate curve beside it makes the band look truncated when it
        # is simply the region where a comparison is defined.
        lo, hi = min(x), max(x)
        clipped = point[
            (point["expected_volatility"] >= lo - 1e-12)
            & (point["expected_volatility"] <= hi + 1e-12)
        ]
        if clipped.empty:
            clipped = point
        fig.add_trace(
            go.Scatter(
                x=clipped["expected_volatility"], y=clipped["expected_return"],
                mode="lines+markers",
                line=dict(color=PALETTE[7], width=2.5),
                marker=dict(size=6),
                name="Point estimate (observed sample)",
                hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Volatility (annualized)",
        yaxis_title="Expected Return (annualized)",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        legend=dict(orientation="h", y=-0.2),
        **_LAYOUT,
    )
    return fig


def plot_weight_dispersion(
    dispersion: pd.Series, title: str = "Which weights the data cannot pin down"
) -> go.Figure:
    """Standard deviation of each asset's weight across resampled frontiers.

    A large bar means the optimizer's conviction in that position comes from
    the particular sample it was handed, not from the asset.

    Args:
        dispersion: One standard deviation per asset, in weight units.
        title: Figure title.

    Returns:
        A Plotly figure.

    Raises:
        MissingDependencyError: If plotly is not installed. Install it with
            ``finport-optengine[viz]``.
    """
    d = dispersion.sort_values()
    fig = go.Figure(
        go.Bar(
            x=d.values * 100,
            y=[str(i) for i in d.index],
            orientation="h",
            marker_color=PALETTE[1],
            hovertemplate="%{y}: ±%{x:.2f}pp across draws<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Weight standard deviation across resampled histories (pp)",
        yaxis_title="", showlegend=False,
        height=max(300, 24 * len(d)), **_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# Benchmark-relative
# ---------------------------------------------------------------------------


def plot_relative_wealth(
    portfolio: pd.Series,
    benchmark: pd.Series,
    title: str = "Cumulative performance vs. benchmark",
) -> go.Figure:
    """Growth of the portfolio's wealth *relative to* the benchmark's.

    The ratio of the two wealth curves, baselined at 1. Rising means the
    portfolio is pulling ahead; the shaded area below the running maximum is
    the relative drawdown — the stretch a plan sponsor experiences as "we have
    been behind the index since 2022", which no absolute chart shows.

    Plotting the ratio rather than the cumulative sum of excess returns is
    deliberate: compounding a difference of returns as though it were a return
    overstates the gap, and the error grows with the sample.

    Args:
        portfolio: The portfolio's return stream.
        benchmark: The benchmark's, over the same dates.
        title: Figure title.

    Returns:
        A Plotly figure.

    Raises:
        MissingDependencyError: If plotly is not installed. Install it with
            ``finport-optengine[viz]``.
    """
    common = portfolio.dropna().index.intersection(benchmark.dropna().index)
    p = portfolio.loc[common]
    b = benchmark.loc[common]
    ratio = (1.0 + p).cumprod() / (1.0 + b).cumprod()
    # The high-water mark starts level with the benchmark, at 1, so falling
    # behind on the first period shows as the drawdown it is.
    peak = ratio.cummax().clip(lower=1.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ratio.index, y=peak, name="Relative high-water mark",
            line=dict(color=BASELINE, width=1, dash="dot"),
            hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ratio.index, y=ratio, name="Relative wealth",
            line=dict(color=PALETTE[0], width=2),
            fill="tonexty", fillcolor="rgba(42,120,214,0.10)",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}×<extra></extra>",
        )
    )
    fig.add_hline(y=1.0, line_width=1, line_color=MUTED)
    fig.update_layout(
        title=title,
        yaxis_title="Portfolio wealth ÷ benchmark wealth",
        xaxis_title="",
        showlegend=False,
        **_LAYOUT,
    )
    return fig


def plot_period_returns(
    periods: pd.DataFrame,
    title: str = "Calendar-period returns",
    excess_column: str = "Excess",
) -> go.Figure:
    """Grouped bars per calendar period, with the excess drawn as a marker.

    Annualized numbers say what happened on average; this says whether the
    average is a description of anything. The excess is a marker rather than a
    third bar so the eye compares portfolio against benchmark first, which is
    the comparison the chart exists to make.

    Args:
        periods: The frame from
            :func:`~optimization_engine.analytics.report.period_returns`.
        title: Figure title.
        excess_column: Which column holds the excess return.

    Returns:
        A Plotly figure.

    Raises:
        MissingDependencyError: If plotly is not installed. Install it with
            ``finport-optengine[viz]``.
    """
    if periods is None or periods.empty:
        return go.Figure()
    bars = [c for c in periods.columns if c != excess_column]
    fig = go.Figure()
    for i, col in enumerate(bars):
        fig.add_trace(
            go.Bar(
                x=list(periods.index), y=periods[col], name=str(col),
                marker_color=series_color(i, col),
                hovertemplate=f"{col}<br>%{{x}}: %{{y:.2%}}<extra></extra>",
            )
        )
    if excess_column in periods.columns:
        fig.add_trace(
            go.Scatter(
                x=list(periods.index), y=periods[excess_column],
                name=excess_column, mode="markers",
                marker=dict(
                    symbol="diamond", size=9, color=PALETTE[3],
                    line=dict(width=1, color="#0b0b0b"),
                ),
                hovertemplate=f"{excess_column}<br>%{{x}}: %{{y:.2%}}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_width=1, line_color=BASELINE)
    fig.update_layout(
        title=title, barmode="group", yaxis_tickformat=".0%",
        yaxis_title="Return", xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **_LAYOUT,
    )
    return fig


def plot_rolling_relative(
    rolling: pd.DataFrame, title: str = "Rolling performance vs. benchmark"
) -> go.Figure:
    """Rolling excess return, tracking error, information ratio and beta.

    Stacked panels for the same reason as :func:`plot_rolling_metrics`: an
    information ratio and a beta occupy similar numeric ranges and would read
    as one duplicated line on a shared axis.

    Args:
        rolling: The frame from
            :func:`~optimization_engine.analytics.report.rolling_relative`.
        title: Figure title.

    Returns:
        A Plotly figure.

    Raises:
        MissingDependencyError: If plotly is not installed. Install it with
            ``finport-optengine[viz]``.
    """
    from plotly.subplots import make_subplots

    panels = [
        ("rolling_excess", "Excess return", ".1%", ".0%"),
        ("rolling_tracking_error", "Tracking error", ".1%", ".0%"),
        ("rolling_information_ratio", "Information ratio", ".2f", ".1f"),
        ("rolling_beta", "Beta", ".2f", ".1f"),
    ]
    present = [p for p in panels if p[0] in rolling.columns]
    if not present:
        return go.Figure()

    fig = make_subplots(
        rows=len(present), cols=1, shared_xaxes=True, vertical_spacing=0.045,
        subplot_titles=[p[1] for p in present],
    )
    for row, (col, label, hover_fmt, tick_fmt) in enumerate(present, start=1):
        fig.add_trace(
            go.Scatter(
                x=rolling.index, y=rolling[col], name=label,
                line=dict(color=PALETTE[(row - 1) % len(PALETTE)], width=1.6),
                hovertemplate=(
                    f"{label}<br>%{{x|%Y-%m-%d}}: %{{y:{hover_fmt}}}<extra></extra>"
                ),
            ),
            row=row, col=1,
        )
        fig.update_yaxes(tickformat=tick_fmt, row=row, col=1)
        # Zero for the excess measures, one for beta: each panel's own
        # "no active decision" line, which is what the eye should compare to.
        fig.add_hline(
            y=1.0 if col == "rolling_beta" else 0.0,
            line_width=1, line_color=MUTED, row=row, col=1,
        )

    fig.update_layout(
        title=title, showlegend=False, height=170 * len(present) + 90, **_LAYOUT
    )
    fig.update_annotations(font_size=12)
    return fig


def plot_risk_return_scatter(
    points: pd.DataFrame,
    highlight: dict[str, str] | None = None,
    title: str = "Risk and return",
) -> go.Figure:
    """Volatility against annualized return, one marker per series.

    Args:
        points: Frame with ``Annualized Vol`` and ``Annualized Return``
            columns, indexed by series name — the layout
            :func:`~optimization_engine.analytics.performance.summary_stats`
            already returns.
        highlight: ``name -> role`` for the series that should stand out
            (``"portfolio"`` and ``"benchmark"``). Everything else is drawn as
            recessive context, because the assets are the backdrop against
            which the two decisions are read, not the subject.
        title: Chart title.
    """
    required = {"Annualized Vol", "Annualized Return"}
    if points is None or points.empty or not required.issubset(points.columns):
        return go.Figure()
    highlight = highlight or {}
    styles = {
        "portfolio": dict(color=PALETTE[0], size=15, symbol="star"),
        "benchmark": dict(color=PALETTE[1], size=13, symbol="diamond"),
    }

    fig = go.Figure()
    context = [i for i in points.index if str(i) not in highlight]
    if context:
        sub = points.loc[context]
        fig.add_trace(
            go.Scatter(
                x=sub["Annualized Vol"], y=sub["Annualized Return"],
                mode="markers+text", text=[str(i) for i in sub.index],
                textposition="top center", textfont=dict(size=9, color=MUTED),
                marker=dict(color=OTHER_COLOR, size=8, opacity=0.65),
                name="Universe",
                hovertemplate=(
                    "%{text}<br>vol %{x:.2%}<br>return %{y:.2%}<extra></extra>"
                ),
            )
        )
    for name, role in highlight.items():
        if name not in points.index:
            continue
        row = points.loc[name]
        fig.add_trace(
            go.Scatter(
                x=[row["Annualized Vol"]], y=[row["Annualized Return"]],
                mode="markers+text", text=[str(name)],
                textposition="bottom center", textfont=dict(size=11),
                marker=styles.get(role, styles["portfolio"]), name=str(name),
                hovertemplate=(
                    f"{name}<br>vol %{{x:.2%}}<br>return %{{y:.2%}}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Annualized volatility", yaxis_title="Annualized return",
        xaxis_tickformat=".0%", yaxis_tickformat=".0%",
        showlegend=False,
        **_LAYOUT,
    )
    fig.add_hline(y=0, line_width=1, line_color=BASELINE)
    return fig
