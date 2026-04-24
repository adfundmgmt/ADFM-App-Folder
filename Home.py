import streamlit as st
from pathlib import Path


st.set_page_config(
    page_title="AD Fund Management",
    layout="wide",
    initial_sidebar_state="collapsed",
)


PAGES = {
    "Market Compass": {
        "path": "pages/3_Weekly_Cross_-_Asset_Compass.py",
        "description": "Start here for the cross-asset regime read.",
    },
    "Liquidity": {
        "path": "pages/5_Liquidity_Tracker.py",
        "description": "Track Fed balance sheet, RRP, TGA, and liquidity pressure.",
    },
    "Market Stress": {
        "path": "pages/4_Market_Stress_Composite.py",
        "description": "Read volatility, credit, curve, funding, and breadth stress.",
    },
    "Equity Baskets": {
        "path": "pages/1_ADFM_Public_Equities_Baskets.py",
        "description": "Monitor leadership, trend, and relative strength across ADFM baskets.",
    },
}


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1120px;
            padding-top: 3.2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 3.2rem 0 2.2rem 0;
            border-bottom: 1px solid rgba(120, 120, 120, 0.18);
            margin-bottom: 2rem;
        }

        .eyebrow {
            font-size: 0.78rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-weight: 700;
            color: #64748b;
            margin-bottom: 0.85rem;
        }

        .title {
            font-size: clamp(3rem, 7vw, 6rem);
            line-height: 0.95;
            letter-spacing: -0.075em;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 1rem;
        }

        .subtitle {
            max-width: 720px;
            font-size: 1.08rem;
            line-height: 1.7;
            color: #475569;
        }

        .section-title {
            font-size: 1.1rem;
            font-weight: 750;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }

        .section-copy {
            color: #64748b;
            font-size: 0.93rem;
            line-height: 1.55;
            margin-bottom: 1.1rem;
        }

        .card {
            border: 1px solid rgba(120, 120, 120, 0.18);
            border-radius: 18px;
            padding: 1.15rem 1.2rem;
            min-height: 145px;
            background: rgba(255, 255, 255, 0.72);
            margin-bottom: 0.65rem;
        }

        .card-title {
            font-size: 1.02rem;
            font-weight: 750;
            color: #0f172a;
            margin-bottom: 0.35rem;
            letter-spacing: -0.02em;
        }

        .card-copy {
            font-size: 0.9rem;
            line-height: 1.5;
            color: #64748b;
        }

        .footer {
            margin-top: 2.6rem;
            padding-top: 1.2rem;
            border-top: 1px solid rgba(120, 120, 120, 0.18);
            color: #94a3b8;
            font-size: 0.85rem;
        }

        @media (prefers-color-scheme: dark) {
            .title,
            .section-title,
            .card-title {
                color: #f8fafc;
            }

            .subtitle,
            .section-copy,
            .card-copy {
                color: #cbd5e1;
            }

            .eyebrow,
            .footer {
                color: #94a3b8;
            }

            .card {
                background: rgba(15, 23, 42, 0.55);
                border-color: rgba(148, 163, 184, 0.22);
            }

            .hero,
            .footer {
                border-color: rgba(148, 163, 184, 0.22);
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
            A simple home base for reading the market regime, checking liquidity and stress,
            and moving into the dashboards that matter without clutter.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="section-title">Start here</div>
    <div class="section-copy">
        Four core dashboards. Open the market compass first, then check liquidity, stress, and equity leadership.
    </div>
    """,
    unsafe_allow_html=True,
)


cols = st.columns(4)

for col, (name, item) in zip(cols, PAGES.items()):
    with col:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">{name}</div>
                <div class="card-copy">{item["description"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page_link(item["path"], f"Open {name}")


st.markdown(
    """
    <div class="footer">
        Regime first. Liquidity second. Stress third. Expression last.
    </div>
    """,
    unsafe_allow_html=True,
)
