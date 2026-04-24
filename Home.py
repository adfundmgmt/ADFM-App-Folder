import streamlit as st
from pathlib import Path


st.set_page_config(
    page_title="AD Fund Management",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# PAGE LINKS
# Keep this small and intentional. This page should feel like a landing page,
# not an inventory management screen.
# =============================================================================

CORE_WORKFLOW = [
    {
        "title": "Cross-Asset Compass",
        "eyebrow": "Regime",
        "description": "Start here. Establish the market regime across equities, rates, credit, commodities, and FX.",
        "page": "pages/3_Weekly_Cross_-_Asset_Compass.py",
    },
    {
        "title": "Market Stress Composite",
        "eyebrow": "Risk",
        "description": "Check whether volatility, credit, curve, funding, or breadth is driving pressure.",
        "page": "pages/4_Market_Stress_Composite.py",
    },
    {
        "title": "Liquidity Tracker",
        "eyebrow": "Liquidity",
        "description": "Read the policy and liquidity backdrop before forcing a trade-level conclusion.",
        "page": "pages/5_Liquidity_Tracker.py",
    },
]

COMMAND_MODULES = [
    {
        "title": "Equity Leadership",
        "description": "Baskets, sector rotation, factors, and breadth. Use this to see where the tape is actually confirming the macro view.",
        "links": [
            ("Public Equities Baskets", "pages/1_ADFM_Public_Equities_Baskets.py"),
            ("Sector Breadth & Rotation", "pages/10_Sector_Breadth_and_Rotation.py"),
            ("Factor Momentum Leadership", "pages/7_Factor_Momentum_Leadership.py"),
        ],
    },
    {
        "title": "Positioning & Flow",
        "description": "Options, ETF flows, and volume behavior. Use this to see where positioning is crowded, chasing, or trapped.",
        "links": [
            ("Unusual Options Flow", "pages/2_Unusual_Options_Flows.py"),
            ("ETF Flows Dashboard", "pages/6_ETF_Flows_Dashboard.py"),
            ("Volume Sentiment", "pages/20_Volume_Based_Sentiment_Indicator.py"),
        ],
    },
    {
        "title": "Risk & Hedging",
        "description": "Stress, volatility shocks, alerts, and hedge timing. Use this when the book needs a risk decision.",
        "links": [
            ("VIX Spike Deep Dive", "pages/8_VIX_Spike_Deep_Dive.py"),
            ("Hedge Timer", "pages/9_Hedge_Timer.py"),
            ("Alert Engine", "pages/21_Alert_Engine.py"),
        ],
    },
    {
        "title": "Policy & Macro",
        "description": "Fed language, global central banks, housing pressure, and regime scoring. Use this when the reaction function is moving.",
        "links": [
            ("Fed Tone Underwriter", "pages/15_Fed_Speeches_Tone_Underwriter.py"),
            ("RoW Central Bank Tone", "pages/16_RoW_Central_Bank_Tone_Underwriter.py"),
            ("Consensus Regime Score", "pages/22_Consensus_Regime_Score.py"),
        ],
    },
    {
        "title": "Technical Workbench",
        "description": "Charts, ratios, breakouts, and tactical confirmation. Use this when price needs to confirm the thesis.",
        "links": [
            ("Technical Chart Explorer", "pages/12_Technical_Chart_Explorer.py"),
            ("Breakout Scanner", "pages/11_Breakout_Scanner.py"),
            ("Ratio Charts", "pages/13_Ratio_Charts.py"),
        ],
    },
    {
        "title": "Market Memory",
        "description": "Analogs, seasonality, and forward path work. Use this to frame what has happened before under similar conditions.",
        "links": [
            ("Market Memory Explorer", "pages/17_Market_Memory_Explorer.py"),
            ("Monthly Seasonality Explorer", "pages/18_Monthly_Seasonality_Explorer.py"),
            ("Home Value / Rent", "pages/19_Home_Value_to_Rent_Ratio.py"),
        ],
    },
]


# =============================================================================
# STYLING
# =============================================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3.5rem;
            max-width: 1260px;
        }

        .adfm-hero {
            border: 1px solid rgba(120, 120, 120, 0.18);
            border-radius: 28px;
            padding: 2.6rem 2.8rem;
            margin-bottom: 1.4rem;
            background:
                radial-gradient(circle at 10% 10%, rgba(80, 120, 255, 0.22), transparent 28%),
                radial-gradient(circle at 90% 20%, rgba(15, 23, 42, 0.10), transparent 26%),
                linear-gradient(135deg, rgba(250, 250, 250, 1), rgba(239, 242, 247, 0.95));
            box-shadow: 0 24px 70px rgba(15, 23, 42, 0.10);
        }

        .adfm-kicker {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: #64748b;
            font-weight: 800;
            margin-bottom: 0.7rem;
        }

        .adfm-title {
            font-size: clamp(2.6rem, 6vw, 5.6rem);
            line-height: 0.95;
            letter-spacing: -0.07em;
            font-weight: 850;
            color: #0f172a;
            margin-bottom: 1.1rem;
        }

        .adfm-subtitle {
            max-width: 780px;
            font-size: 1.05rem;
            line-height: 1.65;
            color: #334155;
        }

        .adfm-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.35rem;
        }

        .adfm-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.42rem 0.72rem;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            background: rgba(255, 255, 255, 0.72);
            color: #0f172a;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .section-eyebrow {
            margin-top: 0.4rem;
            margin-bottom: 0.25rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.73rem;
            color: #64748b;
            font-weight: 800;
        }

        .section-title {
            font-size: 1.55rem;
            line-height: 1.15;
            color: #0f172a;
            font-weight: 800;
            letter-spacing: -0.035em;
            margin-bottom: 0.35rem;
        }

        .section-copy {
            color: #64748b;
            font-size: 0.95rem;
            line-height: 1.55;
            max-width: 840px;
            margin-bottom: 1.0rem;
        }

        .workflow-card {
            min-height: 220px;
            border-radius: 24px;
            padding: 1.25rem 1.25rem 1.05rem 1.25rem;
            border: 1px solid rgba(148, 163, 184, 0.30);
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 250, 252, 0.86));
            box-shadow: 0 14px 40px rgba(15, 23, 42, 0.07);
            margin-bottom: 0.75rem;
        }

        .workflow-number {
            width: 2.0rem;
            height: 2.0rem;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #0f172a;
            color: #f8fafc;
            font-weight: 800;
            font-size: 0.85rem;
            margin-bottom: 0.85rem;
        }

        .workflow-eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.68rem;
            color: #64748b;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .workflow-title {
            font-size: 1.15rem;
            color: #0f172a;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.45rem;
        }

        .workflow-copy {
            color: #475569;
            line-height: 1.5;
            font-size: 0.92rem;
            margin-bottom: 0.55rem;
        }

        .module-card {
            min-height: 260px;
            border-radius: 24px;
            padding: 1.25rem;
            border: 1px solid rgba(148, 163, 184, 0.28);
            background: rgba(255, 255, 255, 0.78);
            box-shadow: 0 12px 36px rgba(15, 23, 42, 0.055);
            margin-bottom: 0.9rem;
        }

        .module-title {
            color: #0f172a;
            font-size: 1.1rem;
            font-weight: 820;
            letter-spacing: -0.03em;
            margin-bottom: 0.45rem;
        }

        .module-copy {
            color: #64748b;
            font-size: 0.9rem;
            line-height: 1.5;
            margin-bottom: 1rem;
        }

        .library-shell {
            border-radius: 26px;
            border: 1px solid rgba(148, 163, 184, 0.28);
            background: rgba(248, 250, 252, 0.72);
            padding: 1.2rem;
            margin-top: 0.5rem;
        }

        .small-muted {
            color: #64748b;
            font-size: 0.84rem;
            line-height: 1.45;
        }

        @media (prefers-color-scheme: dark) {
            .adfm-hero {
                border-color: rgba(148, 163, 184, 0.20);
                background:
                    radial-gradient(circle at 10% 10%, rgba(80, 120, 255, 0.24), transparent 30%),
                    radial-gradient(circle at 90% 20%, rgba(148, 163, 184, 0.12), transparent 26%),
                    linear-gradient(135deg, rgba(15, 23, 42, 1), rgba(30, 41, 59, 0.96));
                box-shadow: 0 24px 70px rgba(0, 0, 0, 0.30);
            }

            .adfm-title,
            .section-title,
            .workflow-title,
            .module-title {
                color: #f8fafc;
            }

            .adfm-subtitle,
            .workflow-copy,
            .module-copy {
                color: #cbd5e1;
            }

            .adfm-kicker,
            .section-eyebrow,
            .section-copy,
            .workflow-eyebrow,
            .small-muted {
                color: #94a3b8;
            }

            .adfm-chip,
            .workflow-card,
            .module-card,
            .library-shell {
                background: rgba(15, 23, 42, 0.72);
                border-color: rgba(148, 163, 184, 0.20);
            }

            .adfm-chip {
                color: #e2e8f0;
            }

            .workflow-number {
                background: #f8fafc;
                color: #0f172a;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HELPERS
# =============================================================================

def safe_page_link(page: str, label: str) -> None:
    if page and Path(page).exists():
        st.page_link(page, label=label)
    else:
        st.caption(f"{label} · page file not found")


def render_section(eyebrow: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-eyebrow">{eyebrow}</div>
        <div class="section-title">{title}</div>
        <div class="section-copy">{copy}</div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_card(item: dict, number: int) -> None:
    st.markdown(
        f"""
        <div class="workflow-card">
            <div class="workflow-number">{number}</div>
            <div class="workflow-eyebrow">{item["eyebrow"]}</div>
            <div class="workflow-title">{item["title"]}</div>
            <div class="workflow-copy">{item["description"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    safe_page_link(item["page"], f"Open {item['title']}")


def render_module_card(module: dict) -> None:
    st.markdown(
        f"""
        <div class="module-card">
            <div class="module-title">{module["title"]}</div>
            <div class="module-copy">{module["description"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for label, page in module["links"]:
        safe_page_link(page, label)


# =============================================================================
# HERO
# =============================================================================

st.markdown(
    """
    <div class="adfm-hero">
        <div class="adfm-kicker">AD Fund Management</div>
        <div class="adfm-title">Market Command Center</div>
        <div class="adfm-subtitle">
            A focused home base for reading the tape, separating signal from noise,
            and moving from regime to expression without turning the landing page
            into a spreadsheet of tools.
        </div>
        <div class="adfm-chip-row">
            <span class="adfm-chip">Regime</span>
            <span class="adfm-chip">Liquidity</span>
            <span class="adfm-chip">Stress</span>
            <span class="adfm-chip">Equity Leadership</span>
            <span class="adfm-chip">Positioning</span>
            <span class="adfm-chip">Hedges</span>
            <span class="adfm-chip">Policy</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# START THE DAY
# =============================================================================

render_section(
    "Start the day",
    "Three dashboards before anything else.",
    "The home page should force the right sequence. First define the regime, then confirm whether risk is building, then check liquidity and policy pressure before going into individual expressions.",
)

workflow_cols = st.columns(3)

for idx, item in enumerate(CORE_WORKFLOW, start=1):
    with workflow_cols[idx - 1]:
        render_workflow_card(item, idx)


st.divider()


# =============================================================================
# COMMAND MODULES
# =============================================================================

render_section(
    "Command modules",
    "Go deeper only after the regime is clear.",
    "Each module is a research lane. The goal is to keep the page visually calm while still giving fast access to the tools that matter.",
)

module_cols = st.columns(3)

for idx, module in enumerate(COMMAND_MODULES):
    with module_cols[idx % 3]:
        render_module_card(module)


st.divider()


# =============================================================================
# LIGHTWEIGHT LIBRARY
# =============================================================================

render_section(
    "Search",
    "Find a dashboard directly.",
    "This is intentionally pushed lower on the page. Search is useful, but it should not define the first impression of the workspace.",
)

all_links = []

for item in CORE_WORKFLOW:
    all_links.append((item["title"], item["eyebrow"], item["description"], item["page"]))

for module in COMMAND_MODULES:
    for label, page in module["links"]:
        all_links.append((label, module["title"], module["description"], page))

query = st.text_input(
    "Search dashboards",
    placeholder="Type a tool, asset class, signal, or workflow",
    label_visibility="collapsed",
)

query_clean = query.strip().lower()

if query_clean:
    matches = [
        item for item in all_links
        if query_clean in " ".join(item).lower()
    ]
else:
    matches = all_links

with st.container():
    st.markdown('<div class="library-shell">', unsafe_allow_html=True)

    if not matches:
        st.info("No dashboards match that search.")
    else:
        cols = st.columns(3)

        for idx, (label, group, description, page) in enumerate(matches):
            with cols[idx % 3]:
                st.markdown(f"**{label}**")
                st.caption(group)
                safe_page_link(page, "Open")

    st.markdown("</div>", unsafe_allow_html=True)


st.divider()


# =============================================================================
# FOOTER
# =============================================================================

footer_left, footer_right = st.columns([2.2, 1])

with footer_left:
    st.markdown("### Operating logic")
    st.write(
        "The page is built around the daily decision sequence: define the regime, confirm stress, check liquidity, "
        "then move into leadership, positioning, technicals, and hedges. Everything else should stay secondary."
    )

with footer_right:
    with st.container(border=True):
        st.markdown("**Default sequence**")
        st.caption("Cross-Asset Compass")
        st.caption("Market Stress Composite")
        st.caption("Liquidity Tracker")
        st.caption("Equity Leadership")
        st.caption("Positioning & Hedges")
