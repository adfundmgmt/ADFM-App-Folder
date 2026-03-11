import streamlit as st

st.set_page_config(page_title="AD Fund Management Tools", layout="wide")

TOOLS_BY_CATEGORY = {
    "Priority Dashboards": [
        (
            "ADFM Public Equities Baskets",
            "Daily equal-weight basket performance, trend status, and relative strength versus SPY.",
        ),
        (
            "Unusual Options Flow Tracker",
            "Scans S&P 500 options chains for outsized premium and positioning anomalies.",
        ),
        (
            "Weekly Cross-Asset Compass",
            "Weekly regime summary across rates, equities, credit, commodities, and FX.",
        ),
        (
            "Market Stress Composite",
            "Single 0-100 stress gauge combining volatility, credit, curve, funding, and breadth inputs.",
        ),
        (
            "Liquidity Tracker",
            "Tracks Net Liquidity (WALCL - RRP - TGA) alongside policy and financial-conditions context.",
        ),
    ],
    "Market & Macro Monitoring": [
        (
            "ETF Flows Dashboard",
            "Estimates ETF creations/redemptions over selectable windows to map capital rotation.",
        ),
        (
            "Factor Momentum Leadership",
            "Ranks factor leadership using relative momentum, trend, and inflection signals.",
        ),
        (
            "VIX Spike Deep Dive",
            "Studies forward SPX behavior after volatility spikes for tactical timing context.",
        ),
        (
            "Hedge Timer",
            "Combines trend, stress, and drawdown context to frame hedge timing decisions.",
        ),
        (
            "Sector Breadth and Rotation",
            "Tracks sector leadership and participation to identify rotation strength or deterioration.",
        ),
        (
            "Market Internals Breadth Monitor",
            "Monitors participation breadth, new highs/lows, and concentration risk proxies like SPY/RSP and QQQ/SPY.",
        ),
        (
            "Sentiment & Positioning Composite",
            "Combines cross-asset risk appetite proxies into a single positioning and sentiment regime score.",
        ),
        (
            "Cross-Asset Volume Dashboard",
            "Volume-led monitor that scores cross-asset setups and weights chart paths by relative participation.",
        ),
    ],
    "Supporting Research Tools": [
        (
            "Breakout Scanner",
            "Screens for key moving-average breakouts with RSI confirmation context.",
        ),
        (
            "Technical Chart Explorer",
            "Multi-indicator charting workspace for equities, commodities, rates proxies, and FX.",
        ),
        (
            "Ratio Charts",
            "Monitors cyclical-vs-defensive and risk-on-vs-risk-off ratio trends.",
        ),
        (
            "Cross Asset Volatility Surface",
            "Compares implied and realized volatility conditions across major asset classes.",
        ),
        (
            "US Inflation Dashboard",
            "Tracks headline/core CPI trend, momentum, and inflation regime shifts.",
        ),
        (
            "Real Yield Dashboard",
            "Monitors nominal vs real 10Y yields and real-rate momentum for macro regime read-through.",
        ),
        (
            "Market Memory Explorer",
            "Finds historical analog years with similar YTD paths to frame scenario roadmaps.",
        ),
        (
            "Monthly Seasonality Explorer",
            "Shows month-by-month return tendencies and hit rates across selected assets.",
        ),
        (
            "Home Value to Rent Ratio",
            "Tracks housing valuation pressure through home-price-to-rent dynamics.",
        ),
        (
            "Event Risk Calendar",
            "Maintains an editable catalyst calendar and quantifies SPY pre/post-event reaction windows.",
        ),
    ],
}

st.title("AD Fund Management LP - Analytics Suite")
st.caption(
    "Internal decision-support dashboards for market structure, macro regime, and positioning intelligence."
)

st.markdown("---")

st.subheader("How to Use This Suite")
col1, col2, col3 = st.columns(3)
col1.info("**1) Start with priority dashboards** to establish market regime and risk context.")
col2.info("**2) Validate with market/macro monitors** before forming a directional view.")
col3.info("**3) Use supporting tools** for deeper drill-down and signal confirmation.")

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
