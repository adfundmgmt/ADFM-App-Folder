import streamlit as st

st.set_page_config(page_title="AD Fund Management Tools", layout="wide")

TOOL_LIBRARY = {
    "Priority Dashboards": [
        {
            "name": "ADFM Public Equities Baskets",
            "description": "Daily equal-weight basket performance, trend status, and relative strength versus SPY.",
            "page": "pages/1_ADFM_Public_Equities_Baskets.py",
            "priority": "High",
            "status": "Live",
        },
        {
            "name": "Unusual Options Flow Tracker",
            "description": "Scans S&P 500 options chains for outsized premium and positioning anomalies.",
            "page": "pages/2_Unusual_Options_Flows.py",
            "priority": "High",
            "status": "Live",
        },
        {
            "name": "Weekly Cross-Asset Compass",
            "description": "Weekly regime summary across rates, equities, credit, commodities, and FX.",
            "page": "pages/3_Weekly_Cross_-_Asset_Compass.py",
            "priority": "High",
            "status": "Live",
        },
        {
            "name": "Market Stress Composite",
            "description": "Single 0-100 stress gauge combining volatility, credit, curve, funding, and breadth inputs.",
            "page": "pages/4_Market_Stress_Composite.py",
            "priority": "High",
            "status": "Live",
        },
        {
            "name": "Liquidity Tracker",
            "description": "Tracks Net Liquidity (WALCL - RRP - TGA) alongside policy and financial-conditions context.",
            "page": "pages/5_Liquidity_Tracker.py",
            "priority": "High",
            "status": "Live",
        },
        {
            "name": "Fed Speeches Tone Underwriter",
            "description": "Flagship policy-tone monitor that scores live Fed communications for hawkish/dovish drift.",
            "page": "pages/15_Fed_Speeches_Tone_Underwriter.py",
            "priority": "High",
            "status": "Live",
        },
    ],
    "Market & Macro Monitoring": [
        {
            "name": "ETF Flows Dashboard",
            "description": "Estimates ETF creations/redemptions over selectable windows to map capital rotation.",
            "page": "pages/6_ETF_Flows_Dashboard.py",
            "priority": "Medium",
            "status": "Live",
        },
        {
            "name": "Factor Momentum Leadership",
            "description": "Ranks factor leadership using relative momentum, trend, and inflection signals.",
            "page": "pages/7_Factor_Momentum_Leadership.py",
            "priority": "Medium",
            "status": "Live",
        },
        {
            "name": "VIX Spike Deep Dive",
            "description": "Studies forward SPX behavior after volatility spikes for tactical timing context.",
            "page": "pages/8_VIX_Spike_Deep_Dive.py",
            "priority": "Medium",
            "status": "Live",
        },
        {
            "name": "Hedge Timer",
            "description": "Combines trend, stress, and drawdown context to frame hedge timing decisions.",
            "page": "pages/9_Hedge_Timer.py",
            "priority": "Medium",
            "status": "Live",
        },
        {
            "name": "Sector Breadth and Rotation",
            "description": "Tracks sector leadership and participation to identify rotation strength or deterioration.",
            "page": "pages/10_Sector_Breadth_and_Rotation.py",
            "priority": "Medium",
            "status": "Live",
        },
        {
            "name": "Alert Engine",
            "description": "Multi-signal trigger board that flags risk-on/risk-off conditions when key market thresholds align.",
            "page": "pages/21_Alert_Engine.py",
            "priority": "Medium",
            "status": "Live",
        },
        {
            "name": "Consensus Regime Score",
            "description": "Composite 0-100 market regime score blending trend, volatility, credit, and breadth internals.",
            "page": "pages/22_Consensus_Regime_Score.py",
            "priority": "Medium",
            "status": "Live",
        },
    ],
    "Supporting Research Tools": [
        {
            "name": "Breakout Scanner",
            "description": "Screens for key moving-average breakouts with RSI confirmation context.",
            "page": "pages/11_Breakout_Scanner.py",
            "priority": "Normal",
            "status": "Live",
        },
        {
            "name": "Technical Chart Explorer",
            "description": "Multi-indicator charting workspace for equities, commodities, rates proxies, and FX.",
            "page": "pages/12_Technical_Chart_Explorer.py",
            "priority": "Normal",
            "status": "Live",
        },
        {
            "name": "Ratio Charts",
            "description": "Monitors cyclical-vs-defensive and risk-on-vs-risk-off ratio trends.",
            "page": "pages/13_Ratio_Charts.py",
            "priority": "Normal",
            "status": "Live",
        },
        {
            "name": "Cross Asset Volatility Surface",
            "description": "Compares implied and realized volatility conditions across major asset classes.",
            "page": "pages/14_Cross_Asset_Volatility_Surface.py",
            "priority": "Normal",
            "status": "Live",
        },
        {
            "name": "RoW Central Bank Tone Underwriter",
            "description": "Tracks policy tone and sentiment shifts from major non-US central banks.",
            "page": "pages/16_RoW_Central_Bank_Tone_Underwriter.py",
            "priority": "Normal",
            "status": "Live",
        },
        {
            "name": "Market Memory Explorer",
            "description": "Finds historical analog years with similar YTD paths to frame scenario roadmaps.",
            "page": "pages/17_Market_Memory_Explorer.py",
            "priority": "Normal",
            "status": "Live",
        },
        {
            "name": "Monthly Seasonality Explorer",
            "description": "Shows month-by-month return tendencies and hit rates across selected assets.",
            "page": "pages/18_Monthly_Seasonality_Explorer.py",
            "priority": "Normal",
            "status": "Live",
        },
        {
            "name": "Home Value to Rent Ratio",
            "description": "Tracks housing valuation pressure through home-price-to-rent dynamics.",
            "page": "pages/19_Home_Value_to_Rent_Ratio.py",
            "priority": "Normal",
            "status": "Live",
        },
        {
            "name": "Volume Based Sentiment Indicator",
            "description": "Measures risk appetite shifts from participation and directional volume behavior.",
            "page": "pages/20_Volume_Based_Sentiment_Indicator.py",
            "priority": "Normal",
            "status": "Live",
        },
        {
            "name": "US Inflation Dashboard",
            "description": "Tracks headline/core CPI trend, momentum, and inflation regime shifts.",
            "priority": "Normal",
            "status": "Planned",
        },
        {
            "name": "Real Yield Dashboard",
            "description": "Monitors nominal vs real 10Y yields and real-rate momentum for macro regime read-through.",
            "priority": "Normal",
            "status": "Planned",
        },
    ],
}

st.markdown("""
<h1>AD Fund Management Tools</h1>
<p>Simple home page for accessing all analytics tools.</p>
<p>Choose a section below and open any live tool.</p>
""", unsafe_allow_html=True)

for category, tools in TOOL_LIBRARY.items():
    st.markdown(f"## {category}")
    for tool in tools:
        st.markdown(f"**{tool['name']}**")
        st.write(tool["description"])
        st.write(f"Status: {tool['status']}")
        if tool.get("page"):
            st.page_link(tool["page"], label=f"Open {tool['name']}")
        st.markdown("---")
