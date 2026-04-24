import streamlit as st
from pathlib import Path


st.set_page_config(
    page_title="AD Fund Management",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CORE_PAGES = {
    "Market Compass": {
        "path": "pages/3_Weekly_Cross_-_Asset_Compass.py",
        "description": "Cross-asset regime snapshot to frame risk-on vs risk-off.",
    },
    "Liquidity Tracker": {
        "path": "pages/5_Liquidity_Tracker.py",
        "description": "Fed balance sheet, RRP, TGA, and broad liquidity pressure.",
    },
    "Market Stress": {
        "path": "pages/4_Market_Stress_Composite.py",
        "description": "Composite stress read across vol, funding, credit, and breadth.",
    },
    "Equity Baskets": {
        "path": "pages/1_ADFM_Public_Equities_Baskets.py",
        "description": "Leadership and trend behavior across ADFM equity sleeves.",
    },
}


TOOL_GROUPS = {
    "Macro + Risk Regime": [
        "Weekly Cross - Asset Compass",
        "Market Stress Composite",
        "Liquidity Tracker",
        "VIX Spike Deep Dive",
        "Consensus Regime Score",
    ],
    "Equities + Positioning": [
        "ADFM Public Equities Baskets",
        "Sector Breadth and Rotation",
        "Factor Momentum Leadership",
        "Breakout Scanner",
        "Technical Chart Explorer",
        "Ratio Charts",
        "Market Memory Explorer",
        "Volume Based Sentiment Indicator",
    ],
    "Flows + Derivatives": [
        "Unusual Options Flows",
        "ETF Flows Dashboard",
        "Cross Asset Volatility Surface",
        "Hedge Timer",
    ],
    "Rates + Macro Tone": [
        "Fed Speeches Tone Underwriter",
        "RoW Central Bank Tone Underwriter",
        "Monthly Seasonality Explorer",
        "Home Value to Rent Ratio",
        "Alert Engine",
    ],
}


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2.5rem;
            padding-bottom: 2.5rem;
        }

        .hero {
            padding: 1.6rem 1.8rem;
            border: 1px solid rgba(120, 120, 120, 0.2);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.06), rgba(2, 132, 199, 0.03));
            margin-bottom: 1.4rem;
        }

        .eyebrow {
            font-size: 0.74rem;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            font-weight: 700;
            color: #64748b;
            margin-bottom: 0.45rem;
        }

        .title {
            font-size: clamp(2rem, 5vw, 3.1rem);
            line-height: 1.05;
            letter-spacing: -0.03em;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.6rem;
        }

        .subtitle {
            max-width: 780px;
            font-size: 1rem;
            line-height: 1.6;
            color: #475569;
        }

        .panel {
            border: 1px solid rgba(120, 120, 120, 0.2);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            background: rgba(255, 255, 255, 0.72);
            margin-bottom: 0.9rem;
            min-height: 145px;
        }

        .panel-title {
            font-size: 0.97rem;
            font-weight: 760;
            color: #0f172a;
            margin-bottom: 0.32rem;
        }

        .panel-copy {
            font-size: 0.89rem;
            line-height: 1.5;
            color: #64748b;
        }

        .section-label {
            font-size: 1.02rem;
            font-weight: 760;
            color: #0f172a;
            margin-top: 0.4rem;
            margin-bottom: 0.2rem;
        }

        .section-copy {
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 0.6rem;
        }

        .tool-list {
            margin: 0;
            padding-left: 1rem;
            color: #475569;
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .tool-list li { margin-bottom: 0.15rem; }

        .footer {
            margin-top: 1.2rem;
            color: #94a3b8;
            font-size: 0.84rem;
        }

        @media (prefers-color-scheme: dark) {
            .title,
            .panel-title,
            .section-label {
                color: #f8fafc;
            }

            .subtitle,
            .panel-copy,
            .section-copy,
            .tool-list {
                color: #cbd5e1;
            }

            .eyebrow,
            .footer {
                color: #94a3b8;
            }

            .hero,
            .panel {
                background: rgba(15, 23, 42, 0.52);
                border-color: rgba(148, 163, 184, 0.24);
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def page_link(path: str, label: str) -> None:
    if Path(path).exists():
        st.page_link(path, label=label)
    else:
        st.caption("Page file not found.")


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AD Fund Management</div>
        <div class="title">Market Workspace</div>
        <div class="subtitle">
            A clean command center for top-down market reads. Start with regime,
            confirm liquidity and stress, then move into flows, leadership, and tactical tools.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="section-label">Core workflow</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">Use these four pages in order for a simple, repeatable daily process.</div>',
    unsafe_allow_html=True,
)

core_cols = st.columns(4)
for col, (name, item) in zip(core_cols, CORE_PAGES.items()):
    with col:
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-title">{name}</div>
                <div class="panel-copy">{item['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page_link(item["path"], f"Open {name}")


st.markdown('<div class="section-label">Tool map</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">Everything in the app grouped by purpose, so you can find the right page fast.</div>',
    unsafe_allow_html=True,
)

map_cols = st.columns(2)
for idx, (group_name, tool_names) in enumerate(TOOL_GROUPS.items()):
    with map_cols[idx % 2]:
        tools_html = "".join(f"<li>{tool}</li>" for tool in tool_names)
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-title">{group_name}</div>
                <ul class="tool-list">{tools_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown(
    """
    <div class="footer">
        Suggested sequence: Regime → Liquidity → Stress → Expression.
    </div>
    """,
    unsafe_allow_html=True,
)
