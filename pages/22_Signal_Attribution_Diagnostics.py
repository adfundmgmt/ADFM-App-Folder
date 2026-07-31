from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from adfm_core.data_registry import market_symbols
from adfm_core.market_data import close_panel, fetch_daily_ohlcv
from adfm_core.research_diagnostics import (
    build_snapshot_history,
    dashboard_route,
    evidence_weights,
    signal_attribution,
    signal_performance,
)
from adfm_core.ui import (
    PageHeader,
    inject_explorer_style,
    render_footer,
    render_page_header,
    render_section_header,
)

st.set_page_config(page_title="Signal Attribution + Diagnostics", layout="wide")
inject_explorer_style()


@st.cache_data(ttl=3600, show_spinner=False)
def load_research_data():
    symbols = market_symbols()
    frames, missing = fetch_daily_ohlcv(symbols, period="10y")
    prices = close_panel(frames, symbols, adjusted=True)
    history = build_snapshot_history(prices)
    if history.empty:
        return prices, history, pd.DataFrame(), pd.DataFrame(), missing
    dates = sorted(pd.to_datetime(history["Snapshot Date"]).dropna().unique())
    current = history.loc[history["Snapshot Date"] == dates[-1]]
    previous = (
        history.loc[history["Snapshot Date"] == dates[-2]]
        if len(dates) > 1
        else pd.DataFrame()
    )
    attribution = signal_attribution(current, previous)
    diagnostics = signal_performance(history, prices)
    return prices, history, attribution, diagnostics, missing


prices, history, attribution, diagnostics, missing = load_research_data()
render_page_header(
    PageHeader(
        title="Signal Attribution + Diagnostics",
        description="Audit composite changes, inspect input contributions, and evaluate subsequent signal performance before changing weights.",
        as_of=(
            f"Snapshot through {pd.to_datetime(history['Snapshot Date']).max().date()}"
            if not history.empty
            else "Snapshot unavailable"
        ),
        source_note="Causal weekly reconstructions from the shared market registry",
    )
)

if attribution.empty:
    st.warning("Insufficient history to calculate attribution.")
else:
    signal = st.selectbox("Signal", sorted(attribution["Signal"].unique()))
    selected = attribution.loc[attribution["Signal"] == signal].copy()
    change = pd.to_numeric(selected["Composite Change"], errors="coerce").iloc[0]
    st.metric(
        "Composite change since prior snapshot",
        f"{change:+.3f}" if pd.notna(change) else "Unavailable",
    )
    render_section_header(
        "Contribution audit",
        "Current and prior inputs use the active normalized weights behind the composite.",
    )
    st.dataframe(
        selected[
            [
                "Component",
                "Current Input",
                "Prior Input",
                "Normalized Weight",
                "Current Contribution",
                "Prior Contribution",
                "Contribution Change",
                "Data Through",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(format="%+.3f")
            for column in (
                "Current Input",
                "Prior Input",
                "Current Contribution",
                "Prior Contribution",
                "Contribution Change",
            )
        },
    )

    signal_key = str(selected["Key"].iloc[0])
    route = dashboard_route(signal_key)
    st.caption(
        f"Drilldown context: {route['instrument']} over {route['lookback']}. "
        "The destination page remains unchanged; the handoff context is retained "
        "in the URL."
    )
    st.session_state["adfm_cross_dashboard_context"] = dict(route)
    route_slug = Path(route["page"]).stem.split("_", 1)[-1]
    route_query = urlencode(
        {
            "instrument": route["instrument"],
            "lookback": route["lookback"],
            "signal": signal_key,
        }
    )
    st.link_button(
        label="Open exact analytical dashboard",
        url=f"/{route_slug}?{route_query}",
    )

tabs = st.tabs(["Forward performance", "Evidence weights", "Data gaps"])
with tabs[0]:
    render_section_header(
        "Out-of-sample diagnostics",
        "Signal direction is evaluated against its own constructive proxy using non-overlapping forward windows.",
    )
    st.dataframe(
        diagnostics,
        width="stretch",
        hide_index=True,
        column_config={
            "Hit Rate": st.column_config.NumberColumn(format="%.1%%"),
            "Average Forward Return": st.column_config.NumberColumn(format="%.2%%"),
            "Average Signed Return": st.column_config.NumberColumn(format="%.2%%"),
            "Worst Strategy Drawdown": st.column_config.NumberColumn(format="%.2%%"),
            "Turnover": st.column_config.NumberColumn(format="%.3f"),
        },
    )
with tabs[1]:
    proposed = evidence_weights(diagnostics)
    st.warning(
        "These are bounded research weights. They are diagnostics, not automatic production changes."
    )
    st.dataframe(
        proposed[
            [
                "Signal",
                "Group",
                "Observations",
                "Hit Rate",
                "Average Signed Return",
                "Worst Strategy Drawdown",
                "Proposed Weight",
                "Evidence Status",
            ]
        ]
        if not proposed.empty
        else proposed,
        width="stretch",
        hide_index=True,
    )
with tabs[2]:
    st.dataframe(missing, width="stretch", hide_index=True)

render_footer()
