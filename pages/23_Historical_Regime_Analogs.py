from __future__ import annotations

import streamlit as st

from adfm_core.data_registry import market_symbols
from adfm_core.market_data import close_panel, fetch_daily_ohlcv
from adfm_core.primary_data import fetch_fred_series
from adfm_core.research_diagnostics import (
    historical_regime_analogs,
    regime_feature_panel,
)
from adfm_core.ui import (
    PageHeader,
    inject_explorer_style,
    render_footer,
    render_page_header,
    render_section_header,
)

st.set_page_config(page_title="Historical Regime Analogs", layout="wide")
inject_explorer_style()


@st.cache_data(ttl=3600, show_spinner=False)
def load_analog_data():
    symbols = market_symbols()
    frames, market_missing = fetch_daily_ohlcv(symbols, period="10y")
    prices = close_panel(frames, symbols, adjusted=True)
    macro, source_status = fetch_fred_series(start="2006-01-01")
    features = regime_feature_panel(prices, macro)
    return prices, features, source_status, market_missing


prices, features, source_status, market_missing = load_analog_data()
match_count = st.sidebar.slider("Historical matches", 5, 25, 12)
exclusion = st.sidebar.slider("Recent-session exclusion", 21, 126, 63)
result = historical_regime_analogs(
    features,
    prices,
    matches=match_count,
    exclusion_sessions=exclusion,
)

render_page_header(
    PageHeader(
        title="Historical Regime Analogs",
        description="Find prior configurations with similar rates, inflation, dollar, liquidity, credit, volatility, and breadth, then inspect the full forward-return distribution.",
        as_of=f"Current regime through {result.current_date or 'unavailable'}",
        source_note="Federal Reserve FRED and shared market proxies",
    )
)

st.caption(
    "Distance is calculated across standardized available features. The most recent "
    f"{exclusion} sessions are excluded to reduce overlap with the current regime. "
    "Macro observations use a bounded 10-session carry between official releases."
)

left, right = st.columns([1.2, 1])
with left:
    render_section_header(
        "Closest historical regimes",
        "Each row is an independent historical date with subsequent cross-asset returns.",
    )
    st.dataframe(
        result.matches,
        width="stretch",
        hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(format="%.2%%")
            for column in result.matches.columns
            if column not in {"Analog Date", "Distance"}
        },
    )
with right:
    render_section_header(
        "Forward-return distribution",
        "Median, tails, and positive frequency preserve dispersion around the analog set.",
    )
    st.dataframe(
        result.distribution,
        width="stretch",
        hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(format="%.2%%")
            for column in (
                "Median",
                "Mean",
                "10th Percentile",
                "90th Percentile",
                "Positive Rate",
            )
        },
    )

with st.expander("Source status and missing observations"):
    st.dataframe(source_status, width="stretch", hide_index=True)
    st.dataframe(market_missing, width="stretch", hide_index=True)

render_footer()
