import streamlit as st

from adfm_core.catalog import tool_descriptions, tool_groups, tool_order
from adfm_core.observability import current_data_health


st.set_page_config(
    page_title="AD Fund Management",
    layout="wide",
    initial_sidebar_state="collapsed",
)


TOOL_ORDER = tool_order()
TOOL_GROUPS = tool_groups()
TOOL_DESCRIPTIONS = tool_descriptions()


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1180px;
            padding-top: 3.75rem;
            padding-bottom: 2.5rem;
        }

        .hero {
            padding: 2.1rem 2rem 2.2rem 2rem;
            border: 1px solid rgba(120, 120, 120, 0.2);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.06), rgba(2, 132, 199, 0.03));
            margin-top: 0.35rem;
            margin-bottom: 1.4rem;
            overflow: visible;
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
            font-size: clamp(2.4rem, 5vw, 3.35rem);
            line-height: 1.05;
            letter-spacing: -0.045em;
            font-weight: 850;
            color: #0f172a;
            margin-bottom: 0.65rem;
        }

        .subtitle {
            max-width: 900px;
            font-size: 1.03rem;
            line-height: 1.65;
            color: #475569;
        }

        .chip-row {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-top: 1.15rem;
        }

        .chip {
            border: 1px solid rgba(100, 116, 139, 0.28);
            padding: 0.28rem 0.72rem;
            border-radius: 999px;
            font-size: 0.78rem;
            line-height: 1;
            color: #475569;
            background: rgba(248, 250, 252, 0.74);
        }

        .section-title {
            font-size: 1.08rem;
            font-weight: 750;
            letter-spacing: -0.02em;
            color: #0f172a;
            margin-top: 0.25rem;
            margin-bottom: 0.9rem;
        }

        .tool-card {
            border: 1px solid rgba(120, 120, 120, 0.2);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            background: rgba(255, 255, 255, 0.72);
            margin-bottom: 0.9rem;
            min-height: 104px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        }

        .tool-title {
            font-size: 0.97rem;
            font-weight: 760;
            color: #0f172a;
            margin-bottom: 0.32rem;
        }

        .tool-copy {
            font-size: 0.89rem;
            line-height: 1.5;
            color: #64748b;
        }

        div[data-testid="stTextInput"] {
            margin-bottom: 0.9rem;
        }

        div[data-testid="stTextInput"] input {
            border-radius: 999px;
        }

        @media (prefers-color-scheme: dark) {
            .title,
            .section-title,
            .tool-title {
                color: #f8fafc;
            }

            .subtitle,
            .tool-copy,
            .chip {
                color: #cbd5e1;
            }

            .eyebrow {
                color: #94a3b8;
            }

            .hero,
            .tool-card,
            .chip {
                background: rgba(15, 23, 42, 0.52);
                border-color: rgba(148, 163, 184, 0.24);
            }

            .tool-card {
                box-shadow: none;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">ADFM Analytics Platform</div>
        <div class="title">AD Fund Management</div>
        <div class="subtitle">
            A focused command center for equity leadership, technical structure,
            flows, macro regimes, rates, credit, liquidity, stress, event risk,
            seasonality, analogs, hedge timing, and currency tension.
        </div>
        <div class="chip-row">
            <span class="chip">Equity Leadership</span>
            <span class="chip">Technicals</span>
            <span class="chip">Relative Value</span>
            <span class="chip">Flows</span>
            <span class="chip">Macro Regime</span>
            <span class="chip">Rates</span>
            <span class="chip">FX</span>
            <span class="chip">Credit</span>
            <span class="chip">Liquidity</span>
            <span class="chip">Event Risk</span>
            <span class="chip">Hedging</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="section-title">Tool Map</div>', unsafe_allow_html=True)

with st.expander("Data health", expanded=False):
    health = current_data_health()
    if health is None:
        st.caption("No shared Yahoo Finance request has run in this session yet.")
    else:
        st.caption(
            f"Provider: {health.provider} | Last pull: {health.recorded_at_utc} | "
            f"Data through: {health.data_through or 'unavailable'} | "
            f"Returned: {health.returned_symbols}/{health.requested_symbols} | "
            f"Failed: {health.failed_symbols}"
        )


if hasattr(st, "segmented_control"):
    selected_group = st.segmented_control(
        "Filter by group",
        options=list(TOOL_GROUPS.keys()),
        default="All tools",
        label_visibility="collapsed",
    )
else:
    selected_group = st.radio(
        "Filter by group",
        options=list(TOOL_GROUPS.keys()),
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )


query = st.text_input(
    "Search tools",
    placeholder="Try: inflation, rates, credit, liquidity, catalyst...",
    label_visibility="collapsed",
)


filtered_tools = TOOL_GROUPS[selected_group]


if query:
    q = query.lower().strip()
    filtered_tools = [
        tool
        for tool in filtered_tools
        if q in tool.lower() or q in TOOL_DESCRIPTIONS.get(tool, "").lower()
    ]


if filtered_tools:
    quick_cols = st.columns(2)

    for idx, tool in enumerate(filtered_tools):
        with quick_cols[idx % 2]:
            st.markdown(
                f"""
                <div class="tool-card">
                    <div class="tool-title">{tool}</div>
                    <div class="tool-copy">{TOOL_DESCRIPTIONS.get(tool, "Description coming soon.")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.info("No tools matched your search. Try a shorter keyword.")
