import streamlit as st

st.set_page_config(page_title="AD Fund Management Tools", layout="wide")

st.markdown(
    """
    <style>
        .hero {
            padding: 1.35rem 1.6rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #0b1220, #121f36 55%, #183a66);
            color: #f3f7ff;
            border: 1px solid rgba(160, 199, 255, 0.25);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2rem;
        }
        .hero p {
            margin: 0;
            opacity: 0.92;
            font-size: 1rem;
        }
        .section-card {
            border: 1px solid rgba(120, 140, 180, 0.3);
            border-radius: 12px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.75rem;
            background: rgba(16, 24, 40, 0.45);
        }
        .featured-card {
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            background: rgba(15, 23, 42, 0.55);
            margin-bottom: 0.75rem;
        }
        .featured-card p {
            margin: 0.35rem 0 0 0;
            font-size: 0.92rem;
            opacity: 0.94;
        }
        .status-chip {
            display: inline-block;
            font-size: 0.72rem;
            padding: 0.15rem 0.45rem;
            border-radius: 999px;
            margin-left: 0.35rem;
            background: rgba(251, 191, 36, 0.12);
            border: 1px solid rgba(251, 191, 36, 0.4);
            color: #fbbf24;
            vertical-align: middle;
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

FEATURED_UPDATES = [
    {
        "title": "Fed Speeches Tone Underwriter",
        "description": "Institutional nowcasting for Fed rhetoric. Prioritize this module ahead of high-impact macro events and FOMC windows.",
        "page": "pages/15_Fed_Speeches_Tone_Underwriter.py",
    },
    {
        "title": "RoW Central Bank Tone Underwriter",
        "description": "Cross-check global policy impulse to capture divergence risk across ECB, BOE, BOJ, and broader G10 messaging.",
        "page": "pages/16_RoW_Central_Bank_Tone_Underwriter.py",
    },
    {
        "title": "Market Memory + Seasonality Stack",
        "description": "Pair analog-year pattern matching with monthly return tendencies to tighten scenario framing and timing decisions.",
        "page": "pages/17_Market_Memory_Explorer.py",
    },
]

all_tools = [tool for group in TOOL_LIBRARY.values() for tool in group]
live_tools = [tool for tool in all_tools if tool["status"] == "Live"]

st.markdown(
    """
    <div class='hero'>
        <h1>AD Fund Management LP · Analytics Suite</h1>
        <p>Professional decision-support platform for market structure, macro regime, liquidity, and policy-tone intelligence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Live Dashboards", len(live_tools))
m2.metric("Planned Modules", len(all_tools) - len(live_tools))
m3.metric("Priority Dashboards", len(TOOL_LIBRARY["Priority Dashboards"]))
m4.metric("Research Modules", len(TOOL_LIBRARY["Supporting Research Tools"]))

st.markdown("---")

st.subheader("What\'s New / Featured Intelligence")
feat_cols = st.columns(3)
for i, item in enumerate(FEATURED_UPDATES):
    with feat_cols[i % len(feat_cols)]:
        st.markdown(
            f"""
            <div class='featured-card'>
                <strong>{item['title']}</strong>
                <p>{item['description']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link(item["page"], label=f"Open {item['title']}", icon="🧭")

st.markdown("---")

st.subheader("Quick Launch")
quick_cols = st.columns(3)
for i, tool in enumerate(TOOL_LIBRARY["Priority Dashboards"]):
    with quick_cols[i % len(quick_cols)]:
        st.page_link(tool["page"], label=tool["name"], icon="📌")

st.markdown("---")

st.subheader("Decision Workflow")
wf1, wf2, wf3, wf4 = st.columns(4)
wf1.info("**1) Set regime context**\n\nStart with stress, liquidity, and cross-asset dashboards.")
wf2.info("**2) Read policy signal**\n\nRun Fed + RoW tone underwriters to identify policy drift and message asymmetry.")
wf3.info("**3) Validate participation**\n\nConfirm breadth, factor leadership, and flows before acting.")
wf4.info("**4) Pressure-test ideas**\n\nUse scanners, analog studies, and seasonality to challenge conviction.")

st.markdown("---")

st.subheader("Tool Library")
search_term = st.text_input("Filter tools by name or description", placeholder="e.g., liquidity, volatility, seasonality")
show_planned = st.toggle("Show planned tools", value=True)
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

    with st.expander(f"{category} ({len(filtered)})", expanded=True):
        if not filtered:
            st.caption("No tools match your current filter.")
            continue

        for tool in filtered:
            badge = ""
            if tool["status"] != "Live":
                badge = "<span class='status-chip'>PLANNED</span>"

            st.markdown(
                f"""
                <div class='section-card'>
                    <strong>{tool['name']}</strong>{badge}<br>
                    <span>{tool['description']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if tool.get("page"):
                st.page_link(tool["page"], label=f"Open {tool['name']}", icon="➡️")

st.markdown("---")
st.success(
    "Built for institutional speed and repeatability: frame regime, validate policy tone, "
    "confirm participation, and execute with disciplined risk context."
)
