import streamlit as st
from pathlib import Path


st.set_page_config(
    page_title="AD Fund Management",
    page_icon="ADFM",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# DATA
# =============================================================================

TOOLS = [
    {
        "name": "Weekly Cross-Asset Compass",
        "short_name": "Cross-Asset Compass",
        "description": "Weekly regime map across equities, rates, credit, commodities, and FX.",
        "page": "pages/3_Weekly_Cross_-_Asset_Compass.py",
        "category": "Macro Command",
        "pillar": "Regime",
        "status": "Live",
        "priority": True,
    },
    {
        "name": "Market Stress Composite",
        "short_name": "Stress Composite",
        "description": "Single stress gauge combining volatility, credit, curve, funding, and breadth inputs.",
        "page": "pages/4_Market_Stress_Composite.py",
        "category": "Macro Command",
        "pillar": "Risk",
        "status": "Live",
        "priority": True,
    },
    {
        "name": "Liquidity Tracker",
        "short_name": "Liquidity",
        "description": "Tracks Net Liquidity, Fed balance sheet mechanics, RRP, TGA, and policy pressure.",
        "page": "pages/5_Liquidity_Tracker.py",
        "category": "Macro Command",
        "pillar": "Liquidity",
        "status": "Live",
        "priority": True,
    },
    {
        "name": "ADFM Public Equities Baskets",
        "short_name": "Equity Baskets",
        "description": "Equal-weight basket performance, trend status, and relative strength versus SPY.",
        "page": "pages/1_ADFM_Public_Equities_Baskets.py",
        "category": "Equity Engine",
        "pillar": "Equities",
        "status": "Live",
        "priority": True,
    },
    {
        "name": "Unusual Options Flow Tracker",
        "short_name": "Options Flow",
        "description": "Scans S&P 500 options chains for outsized premium and positioning anomalies.",
        "page": "pages/2_Unusual_Options_Flows.py",
        "category": "Equity Engine",
        "pillar": "Positioning",
        "status": "Live",
        "priority": True,
    },
    {
        "name": "Fed Speeches Tone Underwriter",
        "short_name": "Fed Tone",
        "description": "Scores live Fed communication for hawkish or dovish drift.",
        "page": "pages/15_Fed_Speeches_Tone_Underwriter.py",
        "category": "Policy",
        "pillar": "Policy",
        "status": "Live",
        "priority": True,
    },
    {
        "name": "ETF Flows Dashboard",
        "short_name": "ETF Flows",
        "description": "Estimates ETF creations and redemptions over selectable windows to map capital rotation.",
        "page": "pages/6_ETF_Flows_Dashboard.py",
        "category": "Flows & Positioning",
        "pillar": "Flows",
        "status": "Live",
        "priority": False,
    },
    {
        "name": "Factor Momentum Leadership",
        "short_name": "Factor Leadership",
        "description": "Ranks factor leadership using relative momentum, trend, and inflection signals.",
        "page": "pages/7_Factor_Momentum_Leadership.py",
        "category": "Equity Engine",
        "pillar": "Factors",
        "status": "Live",
        "priority": False,
    },
    {
        "name": "VIX Spike Deep Dive",
        "short_name": "VIX Spikes",
        "description": "Studies forward SPX behavior after volatility spikes for tactical timing context.",
        "page": "pages/8_VIX_Spike_Deep_Dive.py",
        "category": "Risk & Hedges",
        "pillar": "Volatility",
        "status": "Live",
        "priority": False,
    },
    {
        "name": "Hedge Timer",
        "short_name": "Hedge Timer",
        "description": "Combines trend, stress, and drawdown context to frame hedge timing decisions.",
        "page": "pages/9_Hedge_Timer.py",
        "category": "Risk & Hedges",
        "pillar": "Hedges",
        "status": "Live",
        "priority": False,
    },
    {
        "name": "Sector Breadth and Rotation",
        "short_name": "Sector Rotation",
        "description": "Tracks sector leadership and participation to identify rotation strength or deterioration.",
        "page": "pages/10_Sector_Breadth_and_Rotation.py",
        "category": "Equity Engine",
        "pillar": "Breadth",
        "status": "Live",
        "priority": False,
    },
    {
        "name": "Breakout Scanner",
        "short_name": "Breakouts",
        "description": "Screens for key moving-average breakouts with RSI confirmation context.",
        "page": "pages/11_Breakout_Scanner.py",
        "category": "Technical Research",
        "pillar": "Technicals",
        "status": "Live",
        "priority": False,
    },
    {
        "name": "Technical Chart Explorer",
        "short_name": "Chart Explorer",
        "description": "Multi-indicator charting workspace for equities, commodities, rates proxies, and FX.",
        "page": "pages/12_Technical_Chart_Explorer.py",
        "category": "Technical Research",
        "pillar": "Charts",
        "status": "Live",
        "priority": False,
    },
    {
        "name": "Ratio Charts",
        "short_name": "Ratio Charts",
        "description": "Monitors cyclical-vs-defensive and risk-on-vs-risk-off ratio trends.",
        "page": "pages/13_Ratio_Charts.py",
        "category": "Technical Research",
        "pillar": "Ratios",
        "status": "Live",
        "priority": False,
    },
    {
        "name": "Cross Asset Volatility Surface",
        "short_name": "Vol Surface",
        "description": "Compares implied and realized volatility conditions across major asset classes.",
        "page": "pages/14_Cross_Asset_Volatility_Surface.py",
        "category": "Risk & Hedges",
        "pillar": "Volatility",
        "status": "Live",
        "priority": False,
    },
    {
        "name": "RoW Central Bank Tone Underwriter",
        "short_name": "RoW Central Banks",
        "description": "Tracks policy tone and sentiment shifts from major non-US central banks.",
        "page": "pages/16_RoW_Central_Bank_Tone_Underwriter.py",
        "category": "Policy",
        "pillar": "Policy",
        "status": "Live",
        "priority": False,
    },
    {
        "name": "Market Memory Explorer",
        "short_name": "Market Memory",
        "description": "Finds historical analog years with similar YTD paths to frame scenario roadmaps.",
        "page": "pages/17_Market_Memory_Explorer.py",
        "category": "Research Lab",
        "pillar": "Analogs",
        "status": "Live",
        "priority": False,
    },
    {
        "name": "Monthly Seasonality Explorer",
        "short_name": "Seasonality",
        "description": "Shows month-by-month return tendencies and hit rates across selected assets.",
        "page": "pages/18_Monthly_Seasonality_Explorer.py",
        "category": "Research Lab",
        "pillar": "Seasonality",
        "status": "Live",
        "priority": False,
    },
    {
        "name": "Home Value to Rent Ratio",
        "short_name": "Home Value / Rent",
        "description": "Tracks housing valuation pressure through home-price-to-rent dynamics.",
        "page": "pages/19_Home_Value_to_Rent_Ratio.py",
        "category": "Macro Command",
        "pillar": "Housing",
        "status": "Live",
        "priority": False,
    },
    {
        "name": "Volume Based Sentiment Indicator",
        "short_name": "Volume Sentiment",
        "description": "Measures risk appetite shifts from participation and directional volume behavior.",
        "page": "pages/20_Volume_Based_Sentiment_Indicator.py",
        "category": "Flows & Positioning",
        "pillar": "Sentiment",
        "status": "Live",
        "priority": False,
    },
    {
        "name": "Alert Engine",
        "short_name": "Alert Engine",
        "description": "Flags risk-on and risk-off conditions when key market thresholds align.",
        "page": "pages/21_Alert_Engine.py",
        "category": "Risk & Hedges",
        "pillar": "Alerts",
        "status": "Live",
        "priority": False,
    },
    {
        "name": "Consensus Regime Score",
        "short_name": "Regime Score",
        "description": "Composite market regime score blending trend, volatility, credit, and breadth internals.",
        "page": "pages/22_Consensus_Regime_Score.py",
        "category": "Macro Command",
        "pillar": "Regime",
        "status": "Live",
        "priority": False,
    },
    {
        "name": "US Inflation Dashboard",
        "short_name": "Inflation",
        "description": "Tracks headline and core CPI trend, momentum, and inflation regime shifts.",
        "page": None,
        "category": "Macro Command",
        "pillar": "Inflation",
        "status": "Planned",
        "priority": False,
    },
    {
        "name": "Real Yield Dashboard",
        "short_name": "Real Yields",
        "description": "Monitors nominal versus real 10Y yields and real-rate momentum.",
        "page": None,
        "category": "Macro Command",
        "pillar": "Rates",
        "status": "Planned",
        "priority": False,
    },
]


LANES = [
    {
        "title": "Macro Command",
        "subtitle": "Regime, liquidity, policy, inflation, housing, and rates.",
        "caption": "Start here when the market is moving because the top-down tape changed.",
    },
    {
        "title": "Equity Engine",
        "subtitle": "Baskets, factors, breadth, options, and equity leadership.",
        "caption": "Use this lane to translate macro regime into expression.",
    },
    {
        "title": "Risk & Hedges",
        "subtitle": "Stress, vol, drawdowns, alerts, and hedge timing.",
        "caption": "Use this lane to decide whether to press, pause, hedge, or cut.",
    },
    {
        "title": "Flows & Positioning",
        "subtitle": "ETF flows, volume sentiment, and market participation.",
        "caption": "Use this lane to understand who is chasing, fading, or trapped.",
    },
    {
        "title": "Technical Research",
        "subtitle": "Charts, ratios, breakouts, and tactical confirmation.",
        "caption": "Use this lane to confirm price, trend, and timing.",
    },
    {
        "title": "Policy",
        "subtitle": "Fed and global central bank communication.",
        "caption": "Use this lane when policy language is driving the reaction function.",
    },
    {
        "title": "Research Lab",
        "subtitle": "Analogs, seasonality, and repeatable market memory.",
        "caption": "Use this lane to frame probability, precedent, and path dependency.",
    },
]


# =============================================================================
# STYLE
# =============================================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.3rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }

        div[data-testid="stSidebar"] {
            border-right: 1px solid rgba(49, 51, 63, 0.12);
        }

        .adfm-hero {
            padding: 2.0rem 2.1rem;
            border-radius: 24px;
            border: 1px solid rgba(49, 51, 63, 0.12);
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.18), transparent 34%),
                linear-gradient(135deg, rgba(248, 250, 252, 1), rgba(241, 245, 249, 0.88));
            box-shadow: 0 18px 55px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.4rem;
        }

        .adfm-kicker {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.76rem;
            font-weight: 700;
            color: #475569;
            margin-bottom: 0.55rem;
        }

        .adfm-title {
            font-size: clamp(2.2rem, 5vw, 4.2rem);
            line-height: 1.02;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.055em;
            margin-bottom: 0.8rem;
        }

        .adfm-subtitle {
            font-size: 1.06rem;
            line-height: 1.65;
            color: #334155;
            max-width: 860px;
            margin-bottom: 0.2rem;
        }

        .adfm-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.25rem;
        }

        .adfm-pill {
            font-size: 0.78rem;
            font-weight: 650;
            color: #0f172a;
            padding: 0.42rem 0.72rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.35);
        }

        .adfm-section-label {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            color: #64748b;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }

        .adfm-section-title {
            font-size: 1.45rem;
            font-weight: 780;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }

        .adfm-muted {
            color: #64748b;
            font-size: 0.94rem;
            line-height: 1.55;
        }

        .adfm-card {
            padding: 1.05rem 1.1rem 1rem 1.1rem;
            min-height: 174px;
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.28);
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 8px 28px rgba(15, 23, 42, 0.055);
            margin-bottom: 0.75rem;
        }

        .adfm-card-title {
            font-size: 1.03rem;
            font-weight: 760;
            color: #0f172a;
            letter-spacing: -0.02em;
            margin-bottom: 0.35rem;
        }

        .adfm-card-desc {
            color: #475569;
            font-size: 0.91rem;
            line-height: 1.48;
            margin-bottom: 0.95rem;
        }

        .adfm-card-meta {
            display: flex;
            gap: 0.45rem;
            align-items: center;
            flex-wrap: wrap;
            margin-top: 0.55rem;
        }

        .adfm-tag {
            display: inline-flex;
            align-items: center;
            width: fit-content;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.25rem 0.52rem;
            border-radius: 999px;
            color: #1e293b;
            background: #f1f5f9;
            border: 1px solid rgba(148, 163, 184, 0.25);
        }

        .adfm-live {
            color: #14532d;
            background: #dcfce7;
            border-color: rgba(34, 197, 94, 0.22);
        }

        .adfm-planned {
            color: #713f12;
            background: #fef3c7;
            border-color: rgba(245, 158, 11, 0.24);
        }

        .adfm-lane {
            padding: 1.05rem 1.05rem 1rem 1.05rem;
            border-radius: 20px;
            border: 1px solid rgba(148, 163, 184, 0.28);
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.82));
            min-height: 185px;
            box-shadow: 0 10px 32px rgba(15, 23, 42, 0.055);
            margin-bottom: 0.75rem;
        }

        .adfm-lane-title {
            font-size: 1.06rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.35rem;
        }

        .adfm-lane-subtitle {
            font-size: 0.91rem;
            color: #475569;
            line-height: 1.45;
            margin-bottom: 0.55rem;
        }

        .adfm-lane-caption {
            font-size: 0.82rem;
            color: #64748b;
            line-height: 1.45;
        }

        .adfm-divider-space {
            height: 0.4rem;
        }

        .adfm-small-note {
            color: #64748b;
            font-size: 0.84rem;
            line-height: 1.45;
        }

        @media (prefers-color-scheme: dark) {
            .adfm-hero {
                border-color: rgba(148, 163, 184, 0.18);
                background:
                    radial-gradient(circle at top left, rgba(59, 130, 246, 0.22), transparent 35%),
                    linear-gradient(135deg, rgba(15, 23, 42, 1), rgba(30, 41, 59, 0.92));
                box-shadow: 0 18px 55px rgba(0, 0, 0, 0.24);
            }

            .adfm-title,
            .adfm-card-title,
            .adfm-section-title,
            .adfm-lane-title {
                color: #f8fafc;
            }

            .adfm-subtitle,
            .adfm-card-desc,
            .adfm-lane-subtitle {
                color: #cbd5e1;
            }

            .adfm-kicker,
            .adfm-section-label,
            .adfm-muted,
            .adfm-lane-caption,
            .adfm-small-note {
                color: #94a3b8;
            }

            .adfm-pill,
            .adfm-card,
            .adfm-lane {
                background: rgba(15, 23, 42, 0.78);
                border-color: rgba(148, 163, 184, 0.20);
                color: #e2e8f0;
            }

            .adfm-tag {
                background: rgba(51, 65, 85, 0.85);
                color: #e2e8f0;
                border-color: rgba(148, 163, 184, 0.18);
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HELPERS
# =============================================================================

def page_exists(page: str | None) -> bool:
    if not page:
        return False
    return Path(page).exists()


def get_tools_by_category(category: str) -> list[dict]:
    return [tool for tool in TOOLS if tool["category"] == category]


def search_tools(query: str, category: str, status: str) -> list[dict]:
    query = query.strip().lower()

    results = []

    for tool in TOOLS:
        searchable = " ".join(
            [
                tool["name"],
                tool["short_name"],
                tool["description"],
                tool["category"],
                tool["pillar"],
                tool["status"],
            ]
        ).lower()

        if query and query not in searchable:
            continue

        if category != "All" and tool["category"] != category:
            continue

        if status != "All" and tool["status"] != status:
            continue

        results.append(tool)

    return results


def render_tool_card(tool: dict) -> None:
    status_class = "adfm-live" if tool["status"] == "Live" else "adfm-planned"

    st.markdown(
        f"""
        <div class="adfm-card">
            <div class="adfm-card-title">{tool["short_name"]}</div>
            <div class="adfm-card-desc">{tool["description"]}</div>
            <div class="adfm-card-meta">
                <span class="adfm-tag">{tool["pillar"]}</span>
                <span class="adfm-tag">{tool["category"]}</span>
                <span class="adfm-tag {status_class}">{tool["status"]}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page_exists(tool["page"]):
        st.page_link(tool["page"], label=f"Open {tool['short_name']}")
    elif tool["page"]:
        st.caption("Page file not found.")
    else:
        st.caption("Planned. No page assigned yet.")


def render_lane_card(lane: dict) -> None:
    tools = get_tools_by_category(lane["title"])
    live_count = sum(1 for tool in tools if tool["status"] == "Live")

    st.markdown(
        f"""
        <div class="adfm-lane">
            <div class="adfm-lane-title">{lane["title"]}</div>
            <div class="adfm-lane-subtitle">{lane["subtitle"]}</div>
            <div class="adfm-lane-caption">{lane["caption"]}</div>
            <div class="adfm-card-meta" style="margin-top: 0.85rem;">
                <span class="adfm-tag">{len(tools)} tools</span>
                <span class="adfm-tag adfm-live">{live_count} live</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(label: str, title: str, body: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="adfm-section-label">{label}</div>
        <div class="adfm-section-title">{title}</div>
        """,
        unsafe_allow_html=True,
    )

    if body:
        st.markdown(f'<div class="adfm-muted">{body}</div>', unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### AD Fund Management")
    st.caption("Internal market dashboard workspace.")

    st.divider()

    st.markdown("**Core workflow**")
    st.caption("Regime first. Confirm with stress and liquidity. Then move into equities, positioning, and hedges.")

    st.divider()

    st.markdown("**Quick links**")

    sidebar_tools = [tool for tool in TOOLS if tool["priority"] and tool["status"] == "Live"]

    for tool in sidebar_tools:
        if page_exists(tool["page"]):
            st.page_link(tool["page"], label=tool["short_name"])

    st.divider()

    live_count = sum(1 for tool in TOOLS if tool["status"] == "Live")
    planned_count = sum(1 for tool in TOOLS if tool["status"] == "Planned")

    st.metric("Live dashboards", live_count)
    st.metric("Planned tools", planned_count)


# =============================================================================
# HERO
# =============================================================================

st.markdown(
    """
    <div class="adfm-hero">
        <div class="adfm-kicker">ADFM Internal Workspace</div>
        <div class="adfm-title">Market Command Center</div>
        <div class="adfm-subtitle">
            A cleaner home base for regime work, risk monitoring, liquidity tracking,
            equity leadership, flows, technicals, and policy language. Start with the market regime,
            confirm the pressure points, then move into trade expression.
        </div>
        <div class="adfm-pill-row">
            <span class="adfm-pill">Macro</span>
            <span class="adfm-pill">Equities</span>
            <span class="adfm-pill">Rates</span>
            <span class="adfm-pill">Credit</span>
            <span class="adfm-pill">FX</span>
            <span class="adfm-pill">Volatility</span>
            <span class="adfm-pill">Liquidity</span>
            <span class="adfm-pill">Policy</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# TOP METRICS
# =============================================================================

live_count = sum(1 for tool in TOOLS if tool["status"] == "Live")
planned_count = sum(1 for tool in TOOLS if tool["status"] == "Planned")
priority_count = sum(1 for tool in TOOLS if tool["priority"])
category_count = len(set(tool["category"] for tool in TOOLS))

m1, m2, m3, m4 = st.columns(4)

m1.metric("Live dashboards", live_count)
m2.metric("Priority tools", priority_count)
m3.metric("Research lanes", category_count)
m4.metric("Planned builds", planned_count)


st.markdown('<div class="adfm-divider-space"></div>', unsafe_allow_html=True)


# =============================================================================
# START THE DAY
# =============================================================================

render_section_header(
    "Start the day",
    "Open the dashboards that define the tape first.",
    "This section is intentionally narrow. It is the clean morning workflow before going deeper into baskets, flows, technicals, and hedges.",
)

priority_tools = [tool for tool in TOOLS if tool["priority"] and tool["status"] == "Live"]

priority_cols = st.columns(3)

for i, tool in enumerate(priority_tools):
    with priority_cols[i % 3]:
        render_tool_card(tool)


st.divider()


# =============================================================================
# RESEARCH LANES
# =============================================================================

render_section_header(
    "Research lanes",
    "Move from regime into expression.",
    "Each lane exists to answer a different part of the portfolio question: what changed, who is exposed, where is pressure building, and which expression has the cleanest asymmetry.",
)

lane_cols = st.columns(3)

for i, lane in enumerate(LANES):
    with lane_cols[i % 3]:
        render_lane_card(lane)


st.divider()


# =============================================================================
# DASHBOARD LIBRARY
# =============================================================================

render_section_header(
    "Dashboard library",
    "Search the full workspace.",
    "Use this when you know the signal you want, or when you want to go directly into a specific tool.",
)

filter_col_1, filter_col_2, filter_col_3 = st.columns([2.2, 1.2, 1.0])

with filter_col_1:
    query = st.text_input(
        "Search",
        placeholder="Search by tool, asset class, signal, or workflow",
        label_visibility="collapsed",
    )

with filter_col_2:
    categories = ["All"] + sorted(set(tool["category"] for tool in TOOLS))
    selected_category = st.selectbox(
        "Category",
        categories,
        label_visibility="collapsed",
    )

with filter_col_3:
    selected_status = st.selectbox(
        "Status",
        ["All", "Live", "Planned"],
        label_visibility="collapsed",
    )


results = search_tools(query, selected_category, selected_status)

st.caption(f"{len(results)} dashboard{'s' if len(results) != 1 else ''} shown")

if not results:
    st.info("No dashboards match the current search.")
else:
    result_cols = st.columns(3)

    for i, tool in enumerate(results):
        with result_cols[i % 3]:
            render_tool_card(tool)


st.divider()


# =============================================================================
# FOOTER
# =============================================================================

footer_left, footer_right = st.columns([2, 1])

with footer_left:
    st.markdown("### Operating logic")
    st.write(
        "The workspace is built around one sequence: define the regime, confirm stress and liquidity, "
        "map equity leadership, read positioning, then choose the cleanest expression. "
        "The home page should stay fast, visual, and useful every morning."
    )

with footer_right:
    with st.container(border=True):
        st.markdown("**Default workflow**")
        st.caption("Weekly Compass")
        st.caption("Stress Composite")
        st.caption("Liquidity Tracker")
        st.caption("Equity Baskets")
        st.caption("Options Flow")
