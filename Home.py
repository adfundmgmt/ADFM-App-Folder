import streamlit as st


st.set_page_config(page_title="AD Fund Management Tools", layout="wide")

st.title("AD Fund Management Tools")
st.caption("Simple navigation hub for all dashboards.")


TOOLS = [
    # Priority Dashboards
    ("Priority Dashboards", "ADFM Public Equities Baskets", "Daily equal-weight basket performance, trend status, and relative strength versus SPY.", "pages/1_ADFM_Public_Equities_Baskets.py", "Live"),
    ("Priority Dashboards", "Unusual Options Flow Tracker", "Scans S&P 500 options chains for outsized premium and positioning anomalies.", "pages/2_Unusual_Options_Flows.py", "Live"),
    ("Priority Dashboards", "Weekly Cross-Asset Compass", "Weekly regime summary across rates, equities, credit, commodities, and FX.", "pages/3_Weekly_Cross_-_Asset_Compass.py", "Live"),
    ("Priority Dashboards", "Market Stress Composite", "Single 0-100 stress gauge combining volatility, credit, curve, funding, and breadth inputs.", "pages/4_Market_Stress_Composite.py", "Live"),
    ("Priority Dashboards", "Liquidity Tracker", "Tracks Net Liquidity (WALCL - RRP - TGA) alongside policy and financial-conditions context.", "pages/5_Liquidity_Tracker.py", "Live"),
    ("Priority Dashboards", "Fed Speeches Tone Underwriter", "Scores live Fed communications for hawkish/dovish drift.", "pages/15_Fed_Speeches_Tone_Underwriter.py", "Live"),

    # Market & Macro Monitoring
    ("Market & Macro Monitoring", "ETF Flows Dashboard", "Estimates ETF creations/redemptions over selectable windows to map capital rotation.", "pages/6_ETF_Flows_Dashboard.py", "Live"),
    ("Market & Macro Monitoring", "Factor Momentum Leadership", "Ranks factor leadership using relative momentum, trend, and inflection signals.", "pages/7_Factor_Momentum_Leadership.py", "Live"),
    ("Market & Macro Monitoring", "VIX Spike Deep Dive", "Studies forward SPX behavior after volatility spikes for tactical timing context.", "pages/8_VIX_Spike_Deep_Dive.py", "Live"),
    ("Market & Macro Monitoring", "Hedge Timer", "Combines trend, stress, and drawdown context to frame hedge timing decisions.", "pages/9_Hedge_Timer.py", "Live"),
    ("Market & Macro Monitoring", "Sector Breadth and Rotation", "Tracks sector leadership and participation to identify rotation strength or deterioration.", "pages/10_Sector_Breadth_and_Rotation.py", "Live"),
    ("Market & Macro Monitoring", "Alert Engine", "Flags risk-on/risk-off conditions when key market thresholds align.", "pages/21_Alert_Engine.py", "Live"),
    ("Market & Macro Monitoring", "Consensus Regime Score", "Composite 0-100 market regime score blending trend, volatility, credit, and breadth internals.", "pages/22_Consensus_Regime_Score.py", "Live"),

    # Supporting Research Tools
    ("Supporting Research Tools", "Breakout Scanner", "Screens for key moving-average breakouts with RSI confirmation context.", "pages/11_Breakout_Scanner.py", "Live"),
    ("Supporting Research Tools", "Technical Chart Explorer", "Multi-indicator charting workspace for equities, commodities, rates proxies, and FX.", "pages/12_Technical_Chart_Explorer.py", "Live"),
    ("Supporting Research Tools", "Ratio Charts", "Monitors cyclical-vs-defensive and risk-on-vs-risk-off ratio trends.", "pages/13_Ratio_Charts.py", "Live"),
    ("Supporting Research Tools", "Cross Asset Volatility Surface", "Compares implied and realized volatility conditions across major asset classes.", "pages/14_Cross_Asset_Volatility_Surface.py", "Live"),
    ("Supporting Research Tools", "RoW Central Bank Tone Underwriter", "Tracks policy tone and sentiment shifts from major non-US central banks.", "pages/16_RoW_Central_Bank_Tone_Underwriter.py", "Live"),
    ("Supporting Research Tools", "Market Memory Explorer", "Finds historical analog years with similar YTD paths to frame scenario roadmaps.", "pages/17_Market_Memory_Explorer.py", "Live"),
    ("Supporting Research Tools", "Monthly Seasonality Explorer", "Shows month-by-month return tendencies and hit rates across selected assets.", "pages/18_Monthly_Seasonality_Explorer.py", "Live"),
    ("Supporting Research Tools", "Home Value to Rent Ratio", "Tracks housing valuation pressure through home-price-to-rent dynamics.", "pages/19_Home_Value_to_Rent_Ratio.py", "Live"),
    ("Supporting Research Tools", "Volume Based Sentiment Indicator", "Measures risk appetite shifts from participation and directional volume behavior.", "pages/20_Volume_Based_Sentiment_Indicator.py", "Live"),
    ("Supporting Research Tools", "US Inflation Dashboard", "Tracks headline/core CPI trend, momentum, and inflation regime shifts.", None, "Planned"),
    ("Supporting Research Tools", "Real Yield Dashboard", "Monitors nominal vs real 10Y yields and real-rate momentum for macro regime read-through.", None, "Planned"),
]


def matches_search(tool, search_text):
    category, name, description, page, status = tool
    text = f"{category} {name} {description} {status}".lower()
    return search_text in text


def show_tool(tool):
    category, name, description, page, status = tool

    st.write(f"**{name}** | {status}")
    st.caption(description)

    if page:
        st.page_link(page, label=f"Open {name}")
    else:
        st.caption("No page assigned yet.")

    st.divider()


live_tools = [tool for tool in TOOLS if tool[4] == "Live"]
planned_tools = [tool for tool in TOOLS if tool[4] == "Planned"]
priority_tools = [tool for tool in TOOLS if tool[0] == "Priority Dashboards"]


col1, col2, col3 = st.columns(3)
col1.metric("Live dashboards", len(live_tools))
col2.metric("Planned tools", len(planned_tools))
col3.metric("Priority dashboards", len(priority_tools))


st.subheader("Quick Launch")

quick_cols = st.columns(3)

for i, tool in enumerate(priority_tools):
    category, name, description, page, status = tool

    if page:
        with quick_cols[i % 3]:
            st.page_link(page, label=name)


st.subheader("Tool Index")

search_text = st.text_input("Find a tool", placeholder="Search by name or keyword").strip().lower()
show_planned = st.checkbox("Show planned tools", value=True)

categories = list(dict.fromkeys(tool[0] for tool in TOOLS))

for category in categories:
    visible_tools = [
        tool for tool in TOOLS
        if tool[0] == category
        and (show_planned or tool[4] != "Planned")
        and (not search_text or matches_search(tool, search_text))
    ]

    st.markdown(f"### {category} ({len(visible_tools)})")

    if not visible_tools:
        st.write("No tools match the current filter.")
        continue

    for tool in visible_tools:
        show_tool(tool)
