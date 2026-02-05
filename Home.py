# Home.py
import streamlit as st
st.set_page_config(page_title="AD Fund Management Tools", layout="wide")

st.title("AD Fund Management LP - Analytics Suite")

st.markdown("""
Welcome to the internal analytics dashboard for **AD Fund Management LP**.

These are proprietary tools ADFM has built in-house to support daily decision-making. The goal is simple: make it easier to spot patterns, track shifts, and move faster with more context.

---

### Tools Available (use the sidebar to launch):

- **Monthly Seasonality Explorer**  
  Looks at median monthly returns and hit rates for any stock, index, or commodity. Pulls from Yahoo Finance and FRED for deeper historical data. Helps us stay aware of recurring calendar trends and seasonality biases.

- **Market Memory Explorer**  
  Compares how this year is tracking versus past years with similar return paths. Useful for putting the current tape in historical context and building intuition around where we might go next.

- **Technical Chart Explorer**  
  Visualize key technical indicators for any publicly traded ticker that is on Yahoo Finance (equities, commodities, credit).

- **Sector Breadth and Rotation**  
  This dashboard monitors the relative performance and rotation of S&P 500 sectors to help identify leadership trends and market breadth dynamics.

- **Breakout Scanner**  
  Screen for stocks breaking out to 20D, 50D, 100D, or 200D highs and view multi-timeframe RSI.

- **Ratio Charts**  
  Tracks the relative performance of S&P cyclical and defensive sector ETFs (equal-weighted) to visualize risk-on/risk-off regime shifts in US equities.

- **Cross Asset Volatility Surface**  
  Track cross asset implied and realized volatility in one place and map it to a simple regime read that can steer gross and hedging.

- **ETF Flows Dashboard**  
  A dashboard of thematic, and global ETF flows - see where money is moving among major macro and innovation trades.
  
- **US Inflation Dashboard**  
  Dissects the latest CPI print with YoY and MoM breakdowns across headline, and core categories.

- **Real Yield Dashboard**  
  Tracks nominal and real 10-year Treasury yields alongside the 63-day momentum of real yields (inverted). It’s a clean way to monitor macro regime changes—whether the environment is tilting toward tightening or easing.

- **Liquidity Tracker**  
  Tracks Net Liquidity and relate it to policy and equities. Definition: Net Liquidity = WALCL - RRP - TGA. Units are billions. Sustained rises often coincide with risk-on conditions.

- **Market Stress Composite**  
  A single 0 to 100 stress score blending volatility, credit, curve, funding, and equity drawdown. Higher = more stress.

- **Unusual Options Flow Tracker**  
  Tracks unusual options flow tracker.

- **Factor Momentum Leadership**  
  Tracks style and macro factor leadership using ETF pairs. Each factor is a relative strength ratio scored on short and long momentum, trend structure, and inflection.
  
- **Weekly Cross-Asset Compass**  
  A weekly cross-asset readout that maps regime shifts, correlations, and volatility changes to clear actions, invalidations, and trade expressions in one place.

- **VIX Spike Deep Dive**  
  Evaluates short-horizon SPX behavior after a VIX spike. Support sizing and timing of tactical bounce or fade setups.
st.set_page_config(page_title="AD Fund Management Tools", layout="wide")

- **ADFM Public Equities Baskets**  
  A consolidated dashboard that tracks daily, equal-weight performance and technicals across all ADFM baskets, benchmarked to SPY, with aligned business-day panels and category-level drilldowns.
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
            "Monitors implied/realized volatility across assets and translates it into a practical regime read.",
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
            "Single 0-100 stress score blending volatility, credit, curve, funding, and equity drawdown inputs.",
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

These tools aren’t overly complex, but they’ve become a key part of how we stay organized, especially in fast moving markets. Always a work in progress, open to feedback or feature ideas.
""")
st.title("AD Fund Management LP - Analytics Suite")
st.caption("Internal decision-support dashboards for market structure, macro regime, and positioning intelligence.")

left, mid, right = st.columns(3)
left.metric("Dashboards", total_tools)
mid.metric("Coverage Areas", len(TOOLS_BY_CATEGORY))
right.metric("Primary Data Feeds", "Yahoo Finance + FRED")

st.markdown("---")

st.subheader("How to Use This Suite")
col1, col2, col3 = st.columns(3)
col1.info("**1) Pick a tool** from the sidebar based on your immediate question.")
col2.info("**2) Validate context** with at least one macro/liquidity dashboard.")
col3.info("**3) Cross-check signals** using positioning/flow tools before actioning.")

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
