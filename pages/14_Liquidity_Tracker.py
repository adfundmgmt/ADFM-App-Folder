"""Deterministic Liquidity Conditions Monitor.

The analytical definitions are loaded from ``adfm_core/_liquidity_tracker_base.py``
without executing that file's legacy Streamlit presentation block. This page
owns all controls and rendering, so every Streamlit rerun follows one code path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from adfm_core.palette import PASTEL

BASE_PAGE = Path(__file__).resolve().parents[1] / "adfm_core" / "_liquidity_tracker_base.py"
_ENGINE_SPLIT = "\nrender_page_header(\n"

_source = BASE_PAGE.read_text(encoding="utf-8")
if _ENGINE_SPLIT not in _source:
    raise RuntimeError("Liquidity engine boundary could not be found.")

_engine: dict[str, Any] = {
    "__file__": str(BASE_PAGE),
    "__name__": "adfm_liquidity_engine",
}
exec(
    compile(_source.split(_ENGINE_SPLIT, 1)[0], str(BASE_PAGE), "exec"),
    _engine,
)
# st.set_page_config is executed once from the shared engine loaded above.

# Engine bindings
TITLE = _engine["TITLE"]
BLACK = _engine["BLACK"]
BLUE = PASTEL["blue"]
GREEN = PASTEL["sage"]
RED = _engine["RED"]
ORANGE = PASTEL["coral"]
PURPLE = PASTEL["plum"]
GRAY = _engine["GRAY"]
GRID = _engine["GRID"]

FRED_IDS = _engine["FRED_IDS"]
FRED_LABELS = _engine["FRED_LABELS"]
PRIMARY_SPECS = _engine["PRIMARY_SPECS"]
MARKET_SPECS = _engine["MARKET_SPECS"]

PageHeader = _engine["PageHeader"]
render_page_header = _engine["render_page_header"]
render_section_header = _engine["render_section_header"]
render_footer = _engine["render_footer"]

plot_layout = _engine["plot_layout"]
latest = _engine["latest"]
fmt_raw = _engine["fmt_raw"]
score_bucket = _engine["score_bucket"]
filter_lookback = _engine["filter_lookback"]
load_fred = _engine["load_fred"]
load_market = _engine["load_market"]
market_tickers = _engine["market_tickers"]
build_primary = _engine["build_primary"]
build_market_components = _engine["build_market_components"]
component_scores = _engine["component_scores"]
sleeve_composite = _engine["sleeve_composite"]
scorecard = _engine["scorecard"]
load_fcig = _engine["load_fcig"]


def _read_bucket(value: float, positive: str, negative: str) -> str:
    if pd.isna(value):
        return "Unavailable"
    if value >= 0.35:
        return positive
    if value <= -0.35:
        return negative
    return "Mixed"


def _color_score(value: object) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    if pd.isna(x):
        return ""
    if x >= 0.90:
        return "background-color:#d9ead3;color:#274e13;"
    if x >= 0.35:
        return "background-color:#e2f0d9;color:#385723;"
    if x > -0.35:
        return "background-color:#f2f2f2;color:#404040;"
    if x > -0.90:
        return "background-color:#fce4d6;color:#843c0c;"
    return "background-color:#f4cccc;color:#990000;"


def _style_scores(
    frame: pd.DataFrame,
    score_columns: list[str],
    formats: dict[str, str],
):
    styler = frame.style
    available = [column for column in score_columns if column in frame.columns]
    if available:
        if hasattr(styler, "map"):
            styler = styler.map(_color_score, subset=available)
        else:
            styler = styler.applymap(_color_score, subset=available)
    return styler.format(formats, na_rep="N/A")


def _add_regime_bands(
    fig: go.Figure,
    *,
    row: int,
    positive_label: str,
    negative_label: str,
) -> None:
    fig.add_hrect(
        y0=-0.35,
        y1=0.35,
        fillcolor="rgba(107,114,128,.07)",
        line_width=0,
        row=row,
        col=1,
    )
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color=GRAY,
        row=row,
        col=1,
    )
    fig.add_hline(
        y=0.35,
        line_dash="dot",
        line_color="rgba(112,173,71,.55)",
        row=row,
        col=1,
    )
    fig.add_hline(
        y=-0.35,
        line_dash="dot",
        line_color="rgba(192,0,0,.45)",
        row=row,
        col=1,
    )
    fig.add_annotation(
        text=positive_label,
        xref="paper",
        x=0.995,
        yref=f"y{'' if row == 1 else row}",
        y=0.43,
        showarrow=False,
        xanchor="right",
        font=dict(size=10, color="#548235"),
    )
    fig.add_annotation(
        text=negative_label,
        xref="paper",
        x=0.995,
        yref=f"y{'' if row == 1 else row}",
        y=-0.43,
        showarrow=False,
        xanchor="right",
        font=dict(size=10, color="#9C0006"),
    )


render_page_header(
    PageHeader(
        title=TITLE,
        description=(
            "Primary-source liquidity level and marginal impulse from Federal Reserve "
            "balance-sheet plumbing, overnight funding, credit spreads, the broad dollar, "
            "and real yields. Market prices remain a small confirmation sleeve."
        ),
        eyebrow="ADFM Liquidity Regimes",
    )
)

with st.sidebar:
    st.markdown("## About This Tool")
    st.markdown(
        """
        **Purpose:** Separate the level of system liquidity from its marginal direction.

        - **35% Balance Sheet:** reserves, Fed assets, TGA, ON RRP.
        - **25% Funding:** SOFR and EFFR relative to IORB.
        - **25% Transmission:** HY OAS, IG OAS, broad dollar, real yields.
        - **15% Market Confirmation:** breadth, speculative beta, banks, EM, Bitcoin, volatility.

        Market history is fixed at ten years so changing the display window cannot alter the formula.
        """
    )
    st.markdown("### Display Controls")
    lookback = st.selectbox(
        "Display lookback",
        ["6m", "1y", "2y", "3y", "5y", "10y", "max"],
        index=4,
    )
    z_window = st.number_input(
        "Score lookback, business days",
        min_value=252,
        max_value=1260,
        value=756,
        step=21,
    )
    min_periods = st.number_input(
        "Minimum score observations",
        min_value=126,
        max_value=756,
        value=252,
        step=21,
    )
    smoothing = st.number_input(
        "Composite smoothing, business days",
        min_value=1,
        max_value=21,
        value=3,
        step=1,
    )
    show_drivers = st.checkbox("Show primary liquidity drivers", value=True)
    show_scorecards = st.checkbox("Show component scorecards", value=True)
    show_fcig = st.checkbox("Show Fed financial conditions", value=True)
    show_download = st.checkbox("Show download", value=True)
    st.caption(
        "Display lookback changes the visible date range only. The market-confirmation "
        "history is fixed at ten years."
    )

with st.spinner("Loading primary-source liquidity data..."):
    fred, fred_errors = load_fred(FRED_IDS)

if fred.empty:
    st.error(
        "Primary-source liquidity data could not be loaded. Failed series will be "
        "retried on the next rerun."
    )
    if fred_errors:
        with st.expander("Primary-source diagnostics"):
            for series_id, error in fred_errors.items():
                st.write(f"**{FRED_LABELS.get(series_id, series_id)}:** {error}")
    st.stop()

primary, primary_specs = build_primary(fred)
if primary.empty:
    st.error("The primary liquidity components could not be constructed.")
    st.stop()

# Fixed history prevents a display control from changing the signal definition.
market_period = "10y"
prices = load_market(tuple(market_tickers()), market_period)
market, market_specs = (
    build_market_components(prices)
    if not prices.empty
    else (pd.DataFrame(), [])
)

primary_levels, primary_impulses = component_scores(
    primary,
    primary_specs,
    int(z_window),
    int(min_periods),
)
market_levels, market_impulses = (
    component_scores(
        market,
        market_specs,
        int(z_window),
        int(min_periods),
    )
    if not market.empty
    else (pd.DataFrame(), pd.DataFrame())
)

all_impulses = pd.concat(
    [primary_impulses, market_impulses],
    axis=1,
).sort_index()
all_specs = primary_specs + market_specs

sleeve_impulses, liquidity_impulse, easing_breadth, impulse_coverage = (
    sleeve_composite(
        all_impulses,
        all_specs,
        0.65,
        0.70,
        3,
    )
)
sleeve_levels, liquidity_level, _, level_coverage = sleeve_composite(
    primary_levels,
    primary_specs,
    0.65,
    0.70,
    2,
)

if int(smoothing) > 1:
    liquidity_impulse = liquidity_impulse.rolling(
        int(smoothing),
        min_periods=1,
    ).mean()
    liquidity_level = liquidity_level.rolling(
        int(smoothing),
        min_periods=1,
    ).mean()

market_confirmation = (
    sleeve_impulses["Market Confirmation"]
    if "Market Confirmation" in sleeve_impulses
    else pd.Series(index=liquidity_impulse.index, dtype=float)
)

display_level = filter_lookback(liquidity_level, lookback)
display_impulse = filter_lookback(liquidity_impulse, lookback)
display_sleeve_impulses = filter_lookback(sleeve_impulses, lookback)

current_level = latest(display_level)
current_impulse = latest(display_impulse)
level_read = _read_bucket(current_level, "Easy", "Tight")
impulse_read = _read_bucket(current_impulse, "Improving", "Deteriorating")

if fred_errors:
    st.warning(
        "Unavailable primary series: "
        + ", ".join(
            FRED_LABELS.get(series_id, series_id)
            for series_id in fred_errors
        )
        + ". Coverage rules prevent incomplete sleeves from printing as full signals."
    )

render_section_header(
    "Liquidity Level and Marginal Impulse",
    (
        f"Current level: {current_level:+.2f} ({level_read}). "
        f"Current impulse: {current_impulse:+.2f} ({impulse_read}). "
        "The first panel answers whether liquidity is easy or tight. "
        "The second answers whether it is improving or deteriorating."
    )
    if pd.notna(current_level) and pd.notna(current_impulse)
    else (
        "The first panel answers whether liquidity is easy or tight. "
        "The second answers whether it is improving or deteriorating."
    ),
)

fig_main = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.10,
    row_heights=[0.5, 0.5],
    subplot_titles=("Liquidity level", "Marginal impulse"),
)
_add_regime_bands(
    fig_main,
    row=1,
    positive_label="Easy",
    negative_label="Tight",
)
_add_regime_bands(
    fig_main,
    row=2,
    positive_label="Improving",
    negative_label="Deteriorating",
)
fig_main.add_trace(
    go.Scatter(
        x=display_level.index,
        y=display_level,
        name="Liquidity Level",
        mode="lines",
        line=dict(color=BLUE, width=2.8),
        showlegend=False,
        hovertemplate="%{x|%b %d, %Y}<br>Level: %{y:+.2f}<extra></extra>",
    ),
    row=1,
    col=1,
)
fig_main.add_trace(
    go.Scatter(
        x=display_impulse.index,
        y=display_impulse,
        name="Liquidity Impulse",
        mode="lines",
        line=dict(color=BLACK, width=2.8),
        showlegend=False,
        hovertemplate="%{x|%b %d, %Y}<br>Impulse: %{y:+.2f}<extra></extra>",
    ),
    row=2,
    col=1,
)
plot_layout(
    fig_main,
    600,
    margin=dict(l=56, r=30, t=78, b=44),
    showlegend=False,
)
fig_main.update_yaxes(title_text="Level score", row=1, col=1)
fig_main.update_yaxes(title_text="Impulse score", row=2, col=1)
st.plotly_chart(fig_main, width="stretch")

if show_fcig:
    render_section_header(
        "Federal Reserve FCI-G Overlay",
        (
            "Above zero means financial conditions are a growth headwind. "
            "Below zero means they are a growth tailwind. FCI-G measures transmission "
            "and is therefore kept separate from the liquidity score."
        ),
    )
    fcig, fcig_errors = load_fcig()
    if fcig.empty:
        st.info(
            "Federal Reserve FCI-G is temporarily unavailable. "
            "The primary liquidity composite is unaffected."
        )
    else:
        fcig_display = filter_lookback(fcig, lookback)
        fig_fcig = go.Figure()
        fcig_colors = {
            "FCI-G Baseline": BLUE,
            "FCI-G 1Y Lookback": ORANGE,
        }
        for column in fcig_display.columns:
            fig_fcig.add_trace(
                go.Scatter(
                    x=fcig_display.index,
                    y=fcig_display[column],
                    name=column,
                    mode="lines",
                    line=dict(
                        color=fcig_colors.get(column),
                        width=2.4,
                    ),
                )
            )

        y_values = pd.to_numeric(
            fcig_display.stack(),
            errors="coerce",
        ).dropna()
        if not y_values.empty:
            y_min = min(float(y_values.min()), -0.25)
            y_max = max(float(y_values.max()), 0.25)
            fig_fcig.add_hrect(
                y0=0,
                y1=y_max,
                fillcolor="rgba(192,0,0,.055)",
                line_width=0,
                annotation_text="Growth headwind",
                annotation_position="top left",
            )
            fig_fcig.add_hrect(
                y0=y_min,
                y1=0,
                fillcolor="rgba(112,173,71,.055)",
                line_width=0,
                annotation_text="Growth tailwind",
                annotation_position="bottom left",
            )

        fig_fcig.add_hline(
            y=0,
            line_dash="dot",
            line_color=GRAY,
        )
        plot_layout(
            fig_fcig,
            430,
            margin=dict(l=52, r=28, t=68, b=44),
        )
        fig_fcig.update_yaxes(title_text="FCI-G")
        st.plotly_chart(fig_fcig, width="stretch")

if show_drivers:
    render_section_header(
        "Primary Liquidity Drivers",
        (
            "The line chart shows which primary sleeve is changing. "
            "The bar chart ranks the individual components driving the latest reading. "
            "Positive scores ease liquidity; negative scores tighten it."
        ),
    )

    primary_sleeves = [
        sleeve
        for sleeve in ("Balance Sheet", "Funding", "Transmission")
        if sleeve in display_sleeve_impulses.columns
    ]
    if primary_sleeves:
        sleeve_colors = {
            "Balance Sheet": BLUE,
            "Funding": ORANGE,
            "Transmission": PURPLE,
        }
        fig_sleeves = go.Figure()
        for sleeve in primary_sleeves:
            fig_sleeves.add_trace(
                go.Scatter(
                    x=display_sleeve_impulses.index,
                    y=display_sleeve_impulses[sleeve],
                    name=sleeve,
                    mode="lines",
                    line=dict(
                        color=sleeve_colors[sleeve],
                        width=2.4,
                    ),
                )
            )
        fig_sleeves.add_hrect(
            y0=-0.35,
            y1=0.35,
            fillcolor="rgba(107,114,128,.07)",
            line_width=0,
        )
        fig_sleeves.add_hline(
            y=0,
            line_dash="dot",
            line_color=GRAY,
        )
        plot_layout(
            fig_sleeves,
            420,
            margin=dict(l=52, r=28, t=64, b=44),
        )
        fig_sleeves.update_yaxes(title_text="Sleeve impulse")
        st.plotly_chart(fig_sleeves, width="stretch")

    latest_components = pd.Series(
        {
            column: latest(primary_impulses[column])
            for column in primary_impulses.columns
        },
        dtype=float,
    ).dropna().sort_values()

    if not latest_components.empty:
        bar_colors = [
            RED if value < -0.35 else GREEN if value > 0.35 else GRAY
            for value in latest_components
        ]
        fig_components = go.Figure()
        fig_components.add_vline(
            x=0,
            line_dash="dot",
            line_color=GRAY,
        )
        fig_components.add_trace(
            go.Bar(
                x=latest_components.values,
                y=latest_components.index,
                orientation="h",
                marker_color=bar_colors,
                text=[
                    f"{value:+.2f}"
                    for value in latest_components.values
                ],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}<br>Impulse: %{x:+.2f}<extra></extra>",
            )
        )
        plot_layout(
            fig_components,
            max(390, 36 * len(latest_components) + 90),
            margin=dict(l=190, r=58, t=30, b=42),
            showlegend=False,
            hovermode="closest",
        )
        fig_components.update_xaxes(
            title_text="Latest component impulse"
        )
        fig_components.update_yaxes(showgrid=False)
        st.plotly_chart(fig_components, width="stretch")

primary_card = scorecard(
    primary,
    primary_levels,
    primary_impulses,
    primary_specs,
)
market_card = (
    scorecard(
        market,
        market_levels,
        market_impulses,
        market_specs,
    )
    if not market.empty
    else pd.DataFrame()
)

if show_scorecards:
    render_section_header(
        "Component Audit",
        (
            "Each component shows its latest raw level, level score, marginal impulse, "
            "source, weight, and interpretation."
        ),
    )
    tabs = st.tabs(
        ["Primary Sources", "Market Confirmation", "Source Diagnostics"]
    )

    with tabs[0]:
        if not primary_card.empty:
            styled = _style_scores(
                primary_card,
                ["Level Score", "Impulse Score"],
                {
                    "Level Score": "{:+.2f}",
                    "Impulse Score": "{:+.2f}",
                    "Within-Sleeve Weight": "{:.0%}",
                },
            )
            st.dataframe(
                styled,
                width="stretch",
                hide_index=True,
            )

    with tabs[1]:
        if market_card.empty:
            st.info("Market confirmation data are unavailable.")
        else:
            styled = _style_scores(
                market_card,
                ["Impulse Score"],
                {
                    "Impulse Score": "{:+.2f}",
                    "Within-Sleeve Weight": "{:.0%}",
                },
            )
            st.dataframe(
                styled,
                width="stretch",
                hide_index=True,
            )

    with tabs[2]:
        diagnostics = pd.DataFrame(
            [
                {
                    "Series": FRED_LABELS.get(series_id, series_id),
                    "FRED ID": series_id,
                    "Status": (
                        "Unavailable"
                        if series_id in fred_errors
                        else "Loaded"
                    ),
                    "Latest Observation": (
                        fred[series_id]
                        .dropna()
                        .index.max()
                        .date()
                        .isoformat()
                        if (
                            series_id in fred
                            and fred[series_id].notna().any()
                        )
                        else "N/A"
                    ),
                    "Error": fred_errors.get(series_id, ""),
                }
                for series_id in FRED_IDS
            ]
        )
        st.dataframe(
            diagnostics,
            width="stretch",
            hide_index=True,
        )

if show_download:
    render_section_header(
        "Download",
        "Exports preserve numeric values for independent audit and backtesting.",
    )
    export = pd.concat(
        {
            "Liquidity Level": liquidity_level,
            "Liquidity Impulse": liquidity_impulse,
            "Weighted Breadth": easing_breadth,
            "Coverage": impulse_coverage,
            "Market Confirmation": market_confirmation,
        },
        axis=1,
    ).reset_index(names="Date")
    st.download_button(
        "Download liquidity history",
        export.to_csv(index=False).encode("utf-8"),
        "adfm_liquidity_conditions.csv",
        "text/csv",
    )

render_footer(
    data_note=(
        "Primary inputs: Federal Reserve FRED and H.4.1 series, New York Fed "
        "overnight rates and reverse-repo usage, ICE BofA spread indexes distributed "
        "through FRED, Federal Reserve FCI-G, and Yahoo Finance market prices. "
        "Market history is fixed at ten years so display controls cannot alter the "
        "signal definition."
    )
)
