import streamlit as st

st.set_page_config(page_title="AD Fund Management Tools", layout="wide")

TOOLS_BY_CATEGORY = {
    "Market Structure & Price Action": [
        (
            "Monthly Seasonality Explorer",
            "Median monthly returns and hit rates for stocks, indices, and commodities using Yahoo Finance and FRED data.",
        ),
        (
            "Market Memory Explorer",
            "Compares the current year to prior years with similar return paths to frame potential roadmaps.",
        ),
        (
            "Technical Chart Explorer",
            "Technical indicator dashboard for Yahoo Finance tickers across equities, commodities, and credit proxies.",
        ),
        (
            "Breakout Scanner",
            "Screens for 20D/50D/100D/200D breakouts with multi-timeframe RSI context.",
        ),
        (
            "Ratio Charts",
            "Tracks equal-weight cyclical vs defensive sector ETF performance to monitor risk-on/risk-off behavior.",
        ),
    ],
    "Cross-Asset, Macro & Liquidity": [
        (
            "Cross Asset Volatility Surface",
            "Monitors implied and realized volatility across assets and translates it into a practical regime read.",
        ),
        (
            "ETF Flows Dashboard",
            "Maps thematic and global ETF flow trends to show where capital is rotating.",
        ),
        (
            "US Inflation Dashboard",
            "Breaks down headline and core CPI into YoY and MoM components.",
        ),
        (
            "Real Yield Dashboard",
            "Follows nominal vs real 10Y yields and inverted 63-day real-yield momentum to gauge macro regime shifts.",
        ),
        (
            "Liquidity Tracker",
            "Tracks Net Liquidity (WALCL - RRP - TGA) in billions against policy context and equity behavior.",
        ),
        (
            "Market Stress Composite",
            "Single 0–100 stress score blending volatility, credit, curve, funding, and equity drawdown inputs.",
        ),
        (
            "Weekly Cross-Asset Compass",
            "Weekly cross-asset readout covering regime shifts, correlation changes, and volatility-driven trade framing.",
        ),
    ],
    "Positioning, Flows & Internal Coverage": [
        (
            "Sector Breadth and Rotation",
            "Monitors S&P 500 sector leadership and breadth to identify rotation dynamics.",
        ),
        (
            "Unusual Options Flow Tracker",
            "Tracks unusual options flow activity to surface notable positioning signals.",
        ),
        (
            "Factor Momentum Leadership",
            "Scores style and macro factor leadership using ETF-relative momentum, trend, and inflection structure.",
        ),
        (
            "VIX Spike Deep Dive",
            "Evaluates short-horizon SPX behavior after volatility spikes to support tactical timing and sizing.",
        ),
        (
            "ADFM Public Equities Baskets",
            "Consolidated dashboard for daily equal-weight basket performance and technicals benchmarked to SPY.",
        ),
        (
            "Home Value to Rent Ratio",
            "Tracks housing valuation pressure via home-price-to-rent dynamics.",
        ),
        (
            "Hedge Timer",
            "Monitors timing context for hedging decisions under changing market stress and trend conditions.",
        ),
    ],
}

total_tools = sum(len(tools) for tools in TOOLS_BY_CATEGORY.values())

st.title("AD Fund Management LP - Analytics Suite")
st.caption(
    "Internal decision-support dashboards for market structure, macro regime, and positioning intelligence."
)

left, mid, right = st.columns(3)
left.metric("Dashboards", total_tools)
mid.metric("Coverage Areas", len(TOOLS_BY_CATEGORY))
right.metric("Primary Data Feeds", "Yahoo Finance + FRED")

st.markdown("---")

st.subheader("How to Use This Suite")
col1, col2, col3 = st.columns(3)
col1.info("**1) Pick a tool** from the sidebar based on your immediate question.")
col2.info("**2) Validate context** with at least one macro or liquidity dashboard.")
col3.info("**3) Cross-check signals** using positioning and flow tools before actioning.")

st.markdown("---")

st.subheader("Tool Library")
for category, tools in TOOLS_BY_CATEGORY.items():
    with st.expander(f"{category} ({len(tools)})", expanded=True):
        for name, description in tools:
            st.markdown(f"- **{name}** — {description}")

st.markdown("---")
st.success(
    "These dashboards are built to speed up decision-making with repeatable context. "
    "Feedback and feature requests are always welcome."
)
