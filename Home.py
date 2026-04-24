import streamlit as st


st.set_page_config(
    page_title="AD Fund Management Tools",
    page_icon="📊",
    layout="wide",
)


TOOLS = [
    {
        "category": "Priority Dashboards",
        "name": "ADFM Public Equities Baskets",
        "description": "Daily equal-weight basket performance, trend status, and relative strength versus SPY.",
        "page": "pages/1_ADFM_Public_Equities_Baskets.py",
        "status": "Live",
    },
    {
        "category": "Priority Dashboards",
        "name": "Unusual Options Flow Tracker",
        "description": "Scans S&P 500 options chains for outsized premium and positioning anomalies.",
        "page": "pages/2_Unusual_Options_Flows.py",
        "status": "Live",
    },
    {
        "category": "Priority Dashboards",
        "name": "Weekly Cross-Asset Compass",
        "description": "Weekly regime summary across rates, equities, credit, commodities, and FX.",
        "page": "pages/3_Weekly_Cross_-_Asset_Compass.py",
        "status": "Live",
    },
    {
        "category": "Priority Dashboards",
        "name": "Market Stress Composite",
        "description": "Single 0-100 stress gauge combining volatility, credit, curve, funding, and breadth inputs.",
        "page": "pages/4_Market_Stress_Composite.py",
        "status": "Live",
    },
    {
        "category": "Priority Dashboards",
        "name": "Liquidity Tracker",
        "description": "Tracks Net Liquidity alongside policy and financial-conditions context.",
        "page": "pages/5_Liquidity_Tracker.py",
        "status": "Live",
    },
    {
        "category": "Priority Dashboards",
        "name": "Fed Speeches Tone Underwriter",
        "description": "Scores live Fed communications for hawkish or dovish drift.",
        "page": "pages/15_Fed_Speeches_Tone_Underwriter.py",
        "status": "Live",
    },

    {
        "category": "Market & Macro Monitoring",
        "name": "ETF Flows Dashboard",
        "description": "Estimates ETF creations and redemptions over selectable windows to map capital rotation.",
        "page": "pages/6_ETF_Flows_Dashboard.py",
        "status": "Live",
    },
    {
        "category": "Market & Macro Monitoring",
        "name": "Factor Momentum Leadership",
        "description": "Ranks factor leadership using relative momentum, trend, and inflection signals.",
        "page": "pages/7_Factor_Momentum_Leadership.py",
        "status": "Live",
    },
    {
        "category": "Market & Macro Monitoring",
        "name": "VIX Spike Deep Dive",
        "description": "Studies forward SPX behavior after volatility spikes for tactical timing context.",
        "page": "pages/8_VIX_Spike_Deep_Dive.py",
        "status": "Live",
    },
    {
        "category": "Market & Macro Monitoring",
        "name": "Hedge Timer",
        "description": "Combines trend, stress, and drawdown context to frame hedge timing decisions.",
        "page": "pages/9_Hedge_Timer.py",
        "status": "Live",
    },
    {
        "category": "Market & Macro Monitoring",
        "name": "Sector Breadth and Rotation",
        "description": "Tracks sector leadership and participation to identify rotation strength or deterioration.",
        "page": "pages/10_Sector_Breadth_and_Rotation.py",
        "status": "Live",
    },
    {
        "category": "Market & Macro Monitoring",
        "name": "Alert Engine",
        "description": "Flags risk-on or risk-off conditions when key market thresholds align.",
        "page": "pages/21_Alert_Engine.py",
        "status": "Live",
    },
    {
        "category": "Market & Macro Monitoring",
        "name": "Consensus Regime Score",
        "description": "Composite 0-100 market regime score blending trend, volatility, credit, and breadth internals.",
        "page": "pages/22_Consensus_Regime_Score.py",
        "status": "Live",
    },

    {
        "category": "Supporting Research Tools",
        "name": "Breakout Scanner",
        "description": "Screens for key moving-average breakouts with RSI confirmation context.",
        "page": "pages/11_Breakout_Scanner.py",
        "status": "Live",
    },
    {
        "category": "Supporting Research Tools",
        "name": "Technical Chart Explorer",
        "description": "Multi-indicator charting workspace for equities, commodities, rates proxies, and FX.",
        "page": "pages/12_Technical_Chart_Explorer.py",
        "status": "Live",
    },
    {
        "category": "Supporting Research Tools",
        "name": "Ratio Charts",
        "description": "Monitors cyclical-vs-defensive and risk-on-vs-risk-off ratio trends.",
        "page": "pages/13_Ratio_Charts.py",
        "status": "Live",
    },
    {
        "category": "Supporting Research Tools",
        "name": "Cross Asset Volatility Surface",
        "description": "Compares implied and realized volatility conditions across major asset classes.",
        "page": "pages/14_Cross_Asset_Volatility_Surface.py",
        "status": "Live",
    },
    {
        "category": "Supporting Research Tools",
        "name": "RoW Central Bank Tone Underwriter",
        "description": "Tracks policy tone and sentiment shifts from major non-US central banks.",
        "page": "pages/16_RoW_Central_Bank_Tone_Underwriter.py",
        "status": "Live",
    },
    {
        "category": "Supporting Research Tools",
        "name": "Market Memory Explorer",
        "description": "Finds historical analog years with similar YTD paths to frame scenario roadmaps.",
        "page": "pages/17_Market_Memory_Explorer.py",
        "status": "Live",
    },
    {
        "category": "Supporting Research Tools",
        "name": "Monthly Seasonality Explorer",
        "description": "Shows month-by-month return tendencies and hit rates across selected assets.",
        "page": "pages/18_Monthly_Seasonality_Explorer.py",
        "status": "Live",
    },
    {
        "category": "Supporting Research Tools",
        "name": "Home Value to Rent Ratio",
        "description": "Tracks housing valuation pressure through home-price-to-rent dynamics.",
        "page": "pages/19_Home_Value_to_Rent_Ratio.py",
        "status": "Live",
    },
    {
        "category": "Supporting Research Tools",
        "name": "Volume Based Sentiment Indicator",
        "description": "Measures risk appetite shifts from participation and directional volume behavior.",
        "page": "pages/20_Volume_Based_Sentiment_Indicator.py",
        "status": "Live",
    },
    {
        "category": "Supporting Research Tools",
        "name": "US Inflation Dashboard",
        "description": "Tracks headline and core CPI trend, momentum, and inflation regime shifts.",
        "page": None,
        "status": "Planned",
    },
    {
        "category": "Supporting Research Tools",
        "name": "Real Yield Dashboard",
        "description": "Monitors nominal versus real 10Y yields and real-rate momentum for macro regime read-through.",
        "page": None,
        "status": "Planned",
    },
]


def filter_tools(tools, query, show_planned):
    query = query.strip().lower()

    filtered = []

    for tool in tools:
        searchable_text = " ".join(
            [
                tool["category"],
                tool["name"],
                tool["description"],
                tool["status"],
            ]
        ).lower()

        matches_query = query in searchable_text if query else True
        matches_status = show_planned or tool["status"] == "Live"

        if matches_query and matches_status:
            filtered.append(tool)

    return filtered


def tool_card(tool):
    with st.container(border=True):
        top_left, top_right = st.columns([4, 1])

        with top_left:
            st.markdown(f"**{tool['name']}**")
            st.caption(tool["description"])

        with top_right:
            if tool["status"] == "Live":
                st.success("Live")
            else:
                st.warning("Planned")

        if tool["page"]:
            st.page_link(tool["page"], label="Open dashboard")
        else:
            st.caption("Page not assigned yet.")


def category_view(category, tools):
    category_tools = [tool for tool in tools if tool["category"] == category]

    if not category_tools:
        st.info("No tools match this filter.")
        return

    cols = st.columns(2)

    for index, tool in enumerate(category_tools):
        with cols[index % 2]:
            tool_card(tool)


live_count = sum(1 for tool in TOOLS if tool["status"] == "Live")
planned_count = sum(1 for tool in TOOLS if tool["status"] == "Planned")
priority_count = sum(1 for tool in TOOLS if tool["category"] == "Priority Dashboards")


st.title("AD Fund Management Tools")
st.caption("A simple launch page for market dashboards, macro monitors, and internal research tools.")

st.divider()


metric_col_1, metric_col_2, metric_col_3 = st.columns(3)

metric_col_1.metric("Live dashboards", live_count)
metric_col_2.metric("Planned tools", planned_count)
metric_col_3.metric("Priority dashboards", priority_count)


st.subheader("Priority Launch")

priority_tools = [
    tool for tool in TOOLS
    if tool["category"] == "Priority Dashboards" and tool["status"] == "Live"
]

priority_cols = st.columns(3)

for index, tool in enumerate(priority_tools):
    with priority_cols[index % 3]:
        with st.container(border=True):
            st.markdown(f"**{tool['name']}**")
            st.caption(tool["description"])

            if tool["page"]:
                st.page_link(tool["page"], label="Open")


st.divider()


st.subheader("Tool Index")

search_col, toggle_col = st.columns([3, 1])

with search_col:
    search_text = st.text_input(
        "Search tools",
        placeholder="Search by name, category, status, or keyword",
        label_visibility="collapsed",
    )

with toggle_col:
    show_planned = st.checkbox("Show planned", value=True)


visible_tools = filter_tools(TOOLS, search_text, show_planned)

categories = []
for tool in TOOLS:
    if tool["category"] not in categories:
        categories.append(tool["category"])

tabs = st.tabs(categories)

for tab, category in zip(tabs, categories):
    with tab:
        category_view(category, visible_tools)


with st.expander("About this workspace"):
    st.write(
        "This landing page is designed to stay close to Streamlit's default UI. "
        "It uses native Streamlit components only: metrics, columns, tabs, containers, page links, "
        "text inputs, checkboxes, and expanders. The goal is to keep the home page clean, stable, "
        "and easy to maintain as new dashboards are added."
    )
