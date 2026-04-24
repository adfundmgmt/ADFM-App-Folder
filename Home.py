import streamlit as st
from pathlib import Path


st.set_page_config(
    page_title="AD Fund Management",
    layout="wide",
)


TOOLS = [
    {
        "name": "Weekly Cross-Asset Compass",
        "description": "Weekly regime map across equities, rates, credit, commodities, and FX.",
        "page": "pages/3_Weekly_Cross_-_Asset_Compass.py",
        "group": "Start Here",
    },
    {
        "name": "Market Stress Composite",
        "description": "A single stress gauge built from volatility, credit, curve, funding, and breadth inputs.",
        "page": "pages/4_Market_Stress_Composite.py",
        "group": "Start Here",
    },
    {
        "name": "Liquidity Tracker",
        "description": "Tracks liquidity, Fed balance sheet mechanics, RRP, TGA, and policy pressure.",
        "page": "pages/5_Liquidity_Tracker.py",
        "group": "Start Here",
    },
    {
        "name": "ADFM Public Equities Baskets",
        "description": "Equal-weight basket performance, trend status, and relative strength versus SPY.",
        "page": "pages/1_ADFM_Public_Equities_Baskets.py",
        "group": "Equities",
    },
    {
        "name": "Unusual Options Flow Tracker",
        "description": "Scans options chains for outsized premium, positioning anomalies, and risk appetite.",
        "page": "pages/2_Unusual_Options_Flows.py",
        "group": "Equities",
    },
    {
        "name": "Sector Breadth and Rotation",
        "description": "Tracks sector leadership, breadth, participation, and rotation strength.",
        "page": "pages/10_Sector_Breadth_and_Rotation.py",
        "group": "Equities",
    },
    {
        "name": "Factor Momentum Leadership",
        "description": "Ranks factor leadership using momentum, trend, and inflection signals.",
        "page": "pages/7_Factor_Momentum_Leadership.py",
        "group": "Equities",
    },
    {
        "name": "ETF Flows Dashboard",
        "description": "Estimates ETF creations and redemptions to map capital rotation.",
        "page": "pages/6_ETF_Flows_Dashboard.py",
        "group": "Flows",
    },
    {
        "name": "VIX Spike Deep Dive",
        "description": "Studies forward SPX behavior after volatility shocks.",
        "page": "pages/8_VIX_Spike_Deep_Dive.py",
        "group": "Risk",
    },
    {
        "name": "Hedge Timer",
        "description": "Frames hedge timing through trend, stress, and drawdown context.",
        "page": "pages/9_Hedge_Timer.py",
        "group": "Risk",
    },
    {
        "name": "Alert Engine",
        "description": "Flags risk-on and risk-off conditions when key thresholds align.",
        "page": "pages/21_Alert_Engine.py",
        "group": "Risk",
    },
    {
        "name": "Consensus Regime Score",
        "description": "Composite regime score blending trend, volatility, credit, and breadth.",
        "page": "pages/22_Consensus_Regime_Score.py",
        "group": "Risk",
    },
    {
        "name": "Technical Chart Explorer",
        "description": "Charting workspace for equities, commodities, rates proxies, and FX.",
        "page": "pages/12_Technical_Chart_Explorer.py",
        "group": "Research",
    },
    {
        "name": "Breakout Scanner",
        "description": "Screens for moving-average breakouts with RSI confirmation.",
        "page": "pages/11_Breakout_Scanner.py",
        "group": "Research",
    },
    {
        "name": "Ratio Charts",
        "description": "Monitors cyclical versus defensive and risk-on versus risk-off ratios.",
        "page": "pages/13_Ratio_Charts.py",
        "group": "Research",
    },
    {
        "name": "Cross Asset Volatility Surface",
        "description": "Compares implied and realized volatility across major asset classes.",
        "page": "pages/14_Cross_Asset_Volatility_Surface.py",
        "group": "Research",
    },
    {
        "name": "Fed Speeches Tone Underwriter",
        "description": "Scores Fed communication for hawkish or dovish drift.",
        "page": "pages/15_Fed_Speeches_Tone_Underwriter.py",
        "group": "Policy",
    },
    {
        "name": "RoW Central Bank Tone Underwriter",
        "description": "Tracks tone and policy shifts from major non-US central banks.",
        "page": "pages/16_RoW_Central_Bank_Tone_Underwriter.py",
        "group": "Policy",
    },
    {
        "name": "Market Memory Explorer",
        "description": "Finds historical analog years with similar YTD paths.",
        "page": "pages/17_Market_Memory_Explorer.py",
        "group": "Research",
    },
    {
        "name": "Monthly Seasonality Explorer",
        "description": "Shows month-by-month return tendencies and hit rates.",
        "page": "pages/18_Monthly_Seasonality_Explorer.py",
        "group": "Research",
    },
    {
        "name": "Home Value to Rent Ratio",
        "description": "Tracks housing valuation pressure through home-price-to-rent dynamics.",
        "page": "pages/19_Home_Value_to_Rent_Ratio.py",
        "group": "Macro",
    },
    {
        "name": "Volume Based Sentiment Indicator",
        "description": "Measures risk appetite through participation and directional volume.",
        "page": "pages/20_Volume_Based_Sentiment_Indicator.py",
        "group": "Flows",
    },
]


def page_exists(page_path: str | None) -> bool:
    if not page_path:
        return False

    return Path(page_path).exists()


def open_button(tool: dict, label: str = "Open") -> None:
    if page_exists(tool["page"]):
        st.page_link(tool["page"], label=label)
    else:
        st.caption("Page file not found.")


def tool_tile(tool: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**{tool['name']}**")
        st.caption(tool["description"])
        open_button(tool)


def find_tools(search: str, group: str) -> list[dict]:
    search = search.strip().lower()

    results = []

    for tool in TOOLS:
        if group != "All" and tool["group"] != group:
            continue

        searchable = " ".join(
            [
                tool["name"],
                tool["description"],
                tool["group"],
            ]
        ).lower()

        if search and search not in searchable:
            continue

        results.append(tool)

    return results


start_here = [tool for tool in TOOLS if tool["group"] == "Start Here"]
groups = ["All"] + sorted(set(tool["group"] for tool in TOOLS))


st.title("AD Fund Management")
st.caption("Internal market dashboard workspace.")

st.write(
    "Start with the regime, then move into the tools that explain what is driving the tape. "
    "This page is meant to be a clean launch point, not a database of every dashboard at the top of the screen."
)

st.divider()


left, right = st.columns([2, 1])

with left:
    st.subheader("Start Here")

    st.write(
        "These are the first dashboards to open when you want a fast read on the market: "
        "macro regime, stress, and liquidity."
    )

    main_cols = st.columns(3)

    for index, tool in enumerate(start_here):
        with main_cols[index % 3]:
            tool_tile(tool)

with right:
    st.subheader("Workspace")

    with st.container(border=True):
        st.metric("Dashboards", len(TOOLS))
        st.metric("Core regime tools", len(start_here))
        st.metric("Research lanes", len(groups) - 1)

    st.caption(
        "The rest of the workspace is organized below by research lane so the home page stays clean."
    )


st.divider()


st.subheader("Research Lanes")

lane_cols = st.columns(3)

lanes = [
    {
        "title": "Equities",
        "text": "Baskets, sector rotation, options flow, and factor leadership.",
        "group": "Equities",
    },
    {
        "title": "Risk",
        "text": "Stress, volatility, hedge timing, alerts, and regime scoring.",
        "group": "Risk",
    },
    {
        "title": "Policy & Macro",
        "text": "Liquidity, Fed tone, central bank tone, housing, and macro pressure.",
        "group": "Policy",
    },
]

for index, lane in enumerate(lanes):
    with lane_cols[index]:
        with st.container(border=True):
            st.markdown(f"**{lane['title']}**")
            st.caption(lane["text"])

            matching_tools = [
                tool for tool in TOOLS
                if tool["group"] == lane["group"]
            ]

            for tool in matching_tools[:3]:
                open_button(tool, label=tool["name"])


st.divider()


st.subheader("Find a Dashboard")

search_col, group_col = st.columns([3, 1])

with search_col:
    search = st.text_input(
        "Search",
        placeholder="Search by dashboard, asset class, signal, or workflow",
        label_visibility="collapsed",
    )

with group_col:
    selected_group = st.selectbox(
        "Group",
        groups,
        label_visibility="collapsed",
    )


results = find_tools(search, selected_group)

if not results:
    st.info("No dashboards match this search.")
else:
    result_cols = st.columns(2)

    for index, tool in enumerate(results):
        with result_cols[index % 2]:
            tool_tile(tool)


st.divider()


with st.expander("How to use this page"):
    st.write(
        "Open the weekly compass first when you need the top-down read. "
        "Use the stress and liquidity tools next to confirm whether the market is being driven by policy, credit, funding, volatility, or breadth. "
        "Then use the equity, flow, and technical tools to underwrite the actual trade expression."
    )
