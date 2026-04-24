import streamlit as st

st.set_page_config(page_title="AD Fund Management Tools", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background-color: #f4f4ef;
            color: #111;
        }
        .retro-wrap {
            border: 2px solid #000;
            background: #fffef7;
            padding: 0.8rem 1rem;
            margin-bottom: 0.8rem;
            box-shadow: 4px 4px 0 #bdbdbd;
        }
        .retro-title {
            font-family: "Courier New", monospace;
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: 0.5px;
        }
        .retro-subtitle {
            margin: 0.3rem 0 0 0;
            font-family: "Courier New", monospace;
            font-size: 0.95rem;
        }
        .retro-heading {
            font-family: "Courier New", monospace;
            font-size: 1.15rem;
            font-weight: 700;
            border-bottom: 1px solid #111;
            padding-bottom: 0.25rem;
            margin-bottom: 0.5rem;
        }
        .retro-row {
            border: 1px solid #111;
            background: #fff;
            padding: 0.6rem 0.7rem;
            margin-bottom: 0.45rem;
        }
        .retro-note {
            font-size: 0.9rem;
            font-family: "Courier New", monospace;
            border: 1px dashed #111;
            padding: 0.5rem 0.6rem;
            background: #fafafa;
        }
        .status-live {
            color: #006400;
            font-weight: 700;
            font-family: "Courier New", monospace;
        }
        .status-planned {
            color: #8b0000;
            font-weight: 700;
            font-family: "Courier New", monospace;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

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


def show_tool(tool: dict) -> None:
    status = tool["status"]
    st.write(f"**{tool['name']}** ({status})")
    st.caption(tool["description"])
    if tool.get("page"):
        st.page_link(tool["page"], label=f"Open {tool['name']}")


all_tools = [tool for group in TOOL_LIBRARY.values() for tool in group]
live_tools = [tool for tool in all_tools if tool["status"] == "Live"]
planned_tools = [tool for tool in all_tools if tool["status"] == "Planned"]

st.markdown(
    """
    <div class='retro-wrap'>
        <p class='retro-title'>AD FUND MANAGEMENT LP</p>
        <p class='retro-subtitle'>Market Analytics Home Page | Last Update: Current Session</p>
    </div>
    """,
    unsafe_allow_html=True,
)

summary_col1, summary_col2 = st.columns([1, 2])
with summary_col1:
    st.markdown("<div class='retro-wrap'><p class='retro-heading'>Site Summary</p>", unsafe_allow_html=True)
    st.write(f"Live dashboards: {len(live_tools)}")
    st.write(f"Planned tools: {len(all_tools) - len(live_tools)}")
    st.write(f"Priority dashboards: {len(TOOL_LIBRARY['Priority Dashboards'])}")
    st.markdown("</div>", unsafe_allow_html=True)

with summary_col2:
    st.markdown(
        """
        <div class='retro-wrap'>
            <p class='retro-heading'>How To Use This Site</p>
            <div class='retro-note'>
                1. Start in <b>Priority Dashboards</b> for daily monitoring.<br>
                2. Use <b>Market & Macro Monitoring</b> to confirm regime and leadership.<br>
                3. Use <b>Supporting Research Tools</b> to validate or challenge your thesis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div class='retro-wrap'><p class='retro-heading'>Quick Launch</p></div>", unsafe_allow_html=True)
quick_cols = st.columns(3)
for i, tool in enumerate(TOOL_LIBRARY["Priority Dashboards"]):
    with quick_cols[i % len(quick_cols)]:
        st.page_link(tool["page"], label=tool["name"])

st.markdown("<div class='retro-wrap'><p class='retro-heading'>Full Tool Index</p></div>", unsafe_allow_html=True)

search_term = st.text_input("Find a tool", placeholder="Type a keyword such as liquidity, volatility, seasonality")
show_planned = st.checkbox("Show planned tools", value=True)
needle = search_term.strip().lower()

for category, tools in TOOL_LIBRARY.items():
    filtered = []
    for tool in tools:
        if tool["status"] == "Planned" and not show_planned:
            continue
        haystack = f"{tool['name']} {tool['description']}".lower()
        if needle and needle not in haystack:
            continue
        filtered.append(tool)

    st.markdown(f"### {category} ({len(filtered)})")

    if not filtered:
        st.write("No tools match the current filter.")
        continue

    for tool in filtered:
        status_class = "status-live" if tool["status"] == "Live" else "status-planned"
        st.markdown(
            f"""
            <div class='retro-row'>
                <b>{tool['name']}</b> | <span class='{status_class}'>{tool['status'].upper()}</span><br>
                {tool['description']}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if tool.get("page"):
            st.page_link(tool["page"], label=f"Open {tool['name']}")

st.markdown(
    """
    <div class='retro-wrap'>
        <p class='retro-subtitle'>Functional layout designed for fast navigation and readability.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
