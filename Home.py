import numpy as np
import pandas as pd
import streamlit as st

from adfm_core.catalog import (
    tool_definitions,
    tool_descriptions,
    tool_groups,
    tool_order,
)
from adfm_core.data_registry import market_symbols
from adfm_core.market_data import close_panel, fetch_daily_ohlcv
from adfm_core.observability import current_data_health
from adfm_core.pm_cockpit import (
    build_signal_snapshot,
    group_scores,
    largest_changes,
    summarize_snapshot,
)
from adfm_core.signal_ledger import (
    latest_score_changes,
    record_signal_snapshot,
)

st.set_page_config(
    page_title="AD Fund Management LP",
    layout="wide",
    initial_sidebar_state="collapsed",
)


TOOL_ORDER = tool_order()
TOOL_GROUPS = tool_groups()
TOOL_DESCRIPTIONS = tool_descriptions()
TOOL_DEFINITIONS = {tool.title: tool for tool in tool_definitions()}


@st.cache_data(ttl=900, show_spinner=False)
def load_pm_command_center():
    """Load one coherent market snapshot for the landing-page command center."""

    symbols = market_symbols()
    frames, missing = fetch_daily_ohlcv(symbols, period="3y")
    prices = close_panel(frames, symbols, adjusted=True)
    snapshot = build_signal_snapshot(prices)
    summary = summarize_snapshot(snapshot)
    groups = group_scores(snapshot)
    movers = largest_changes(snapshot)
    ledger_error = None
    score_changes = pd.DataFrame()
    try:
        history = record_signal_snapshot(snapshot)
        score_changes = latest_score_changes(history)
    except Exception as exc:
        ledger_error = f"{type(exc).__name__}: {exc}"
    if not score_changes.empty:
        snapshot = snapshot.merge(score_changes, on="Key", how="left")
    return snapshot, summary, groups, movers, missing, ledger_error


def fmt_score(value: float) -> str:
    """Format a bounded signal score without fabricating unavailable values."""

    return f"{value:+.2f}" if np.isfinite(value) else "Unavailable"


def fmt_percent(value: float) -> str:
    """Format a zero-to-one fraction as a whole percentage."""

    return f"{value:.0%}" if np.isfinite(value) else "Unavailable"


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

        .cockpit-banner {
            display: grid;
            grid-template-columns: minmax(0, 2fr) minmax(220px, 1fr);
            gap: 1rem;
            padding: 1.1rem 1.2rem;
            border: 1px solid rgba(120, 120, 120, 0.2);
            border-left: 5px solid #315f95;
            border-radius: 14px;
            background: rgba(248, 250, 252, 0.74);
            margin: 0.35rem 0 0.9rem;
        }

        .cockpit-label {
            color: #64748b;
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .cockpit-regime {
            color: #0f172a;
            font-size: 1.45rem;
            font-weight: 800;
            margin: 0.2rem 0 0.35rem;
        }

        .cockpit-copy {
            color: #475569;
            font-size: 0.86rem;
            line-height: 1.5;
        }

        .cockpit-score {
            color: #0f172a;
            font-size: 2rem;
            font-weight: 820;
            text-align: right;
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
            .tool-title,
            .cockpit-regime,
            .cockpit-score {
                color: #f8fafc;
            }

            .subtitle,
            .tool-copy,
            .chip,
            .cockpit-copy {
                color: #cbd5e1;
            }

            .eyebrow,
            .cockpit-label {
                color: #94a3b8;
            }

            .hero,
            .tool-card,
            .chip,
            .cockpit-banner {
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
        <div class="eyebrow">AD Fund Management LP</div>
        <div class="title">ADFM Analytics Platform</div>
        <div class="subtitle">
            A command center built by ADFM team for equity leadership, technical structure,
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


st.markdown(
    '<div class="section-title">PM Command Center</div>', unsafe_allow_html=True
)

with st.spinner("Loading one coherent cross-asset snapshot..."):
    (
        signal_snapshot,
        cockpit_summary,
        cockpit_groups,
        cockpit_movers,
        cockpit_missing,
        cockpit_ledger_error,
    ) = load_pm_command_center()

if cockpit_summary.available_signals == 0:
    st.warning(
        "The command center could not build a current cross-asset snapshot. "
        "The tool map remains available below."
    )
else:
    constructive = int(
        (pd.to_numeric(signal_snapshot["Composite"], errors="coerce") > 0.10).sum()
    )
    defensive = int(
        (pd.to_numeric(signal_snapshot["Composite"], errors="coerce") < -0.10).sum()
    )
    st.markdown(
        f"""
        <div class="cockpit-banner">
            <div>
                <div class="cockpit-label">Current cross-asset regime</div>
                <div class="cockpit-regime">{cockpit_summary.regime}</div>
                <div class="cockpit-copy">
                    {constructive} signals are constructive and {defensive} are defensive.
                    Dispersion is {cockpit_summary.dispersion:.2f}; higher dispersion means
                    the tape is giving a less coherent message. Data through
                    {cockpit_summary.as_of or "unavailable"}.
                </div>
            </div>
            <div>
                <div class="cockpit-label" style="text-align:right">Composite score</div>
                <div class="cockpit-score">{fmt_score(cockpit_summary.composite)}</div>
                <div class="cockpit-copy" style="text-align:right">
                    Confidence {fmt_percent(cockpit_summary.confidence)}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Signal breadth", fmt_percent(cockpit_summary.breadth))
    metric_cols[1].metric("Near-term impulse", fmt_score(cockpit_summary.impulse))
    metric_cols[2].metric(
        "Cross-signal dispersion", f"{cockpit_summary.dispersion:.2f}"
    )
    metric_cols[3].metric(
        "Signals available",
        f"{cockpit_summary.available_signals}/{cockpit_summary.total_signals}",
    )
    metric_cols[4].metric(
        "Missing tickers",
        str(len(cockpit_missing)),
    )

    mover_cols = st.columns(2)
    with mover_cols[0]:
        st.caption("Largest improvements: 1-week score versus 3-month score")
        improving = cockpit_movers["improving"][
            ["Signal", "Group", "Composite", "Impulse"]
        ].copy()
        st.dataframe(
            improving,
            width="stretch",
            hide_index=True,
            column_config={
                "Composite": st.column_config.NumberColumn(format="%+.2f"),
                "Impulse": st.column_config.NumberColumn(format="%+.2f"),
            },
        )
    with mover_cols[1]:
        st.caption("Largest deteriorations: 1-week score versus 3-month score")
        deteriorating = cockpit_movers["deteriorating"][
            ["Signal", "Group", "Composite", "Impulse"]
        ].copy()
        st.dataframe(
            deteriorating,
            width="stretch",
            hide_index=True,
            column_config={
                "Composite": st.column_config.NumberColumn(format="%+.2f"),
                "Impulse": st.column_config.NumberColumn(format="%+.2f"),
            },
        )

    with st.expander(
        "Signal sleeves, source coverage, and methodology", expanded=False
    ):
        st.dataframe(
            cockpit_groups,
            width="stretch",
            hide_index=True,
            column_config={
                "Composite": st.column_config.NumberColumn(format="%+.2f"),
                "Impulse": st.column_config.NumberColumn(format="%+.2f"),
            },
        )
        st.caption(
            "Scores compare current market moves with prior observations only. "
            "Higher scores are constructed to represent easier liquidity or stronger "
            "risk confirmation. The landing page uses Yahoo Finance market proxies; "
            "exact macro levels belong to registered primary-source series."
        )
        if not cockpit_missing.empty:
            st.dataframe(cockpit_missing, width="stretch", hide_index=True)
        if cockpit_ledger_error:
            st.caption(
                "Point-in-time history could not be updated in this runtime. "
                "The current snapshot remains available."
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
            definition = TOOL_DEFINITIONS[tool]
            st.markdown(
                f"""
                <div class="tool-card">
                    <div class="tool-title">{tool}</div>
                    <div class="tool-copy">{TOOL_DESCRIPTIONS.get(tool, "Description coming soon.")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.page_link(
                f"pages/{definition.page_filename}",
                label=f"Open {tool}",
                use_container_width=True,
            )
else:
    st.info("No tools matched your search. Try a shorter keyword.")
