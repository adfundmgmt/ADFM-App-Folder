from __future__ import annotations

import pandas as pd
import streamlit as st

from adfm_core.data_registry import market_symbols
from adfm_core.market_data import close_panel, fetch_daily_ohlcv
from adfm_core.observability import current_data_health
from adfm_core.operations import (
    daily_brief_tables,
    evaluate_alerts,
    load_decision_journal,
    new_alert_transitions,
)
from adfm_core.pm_cockpit import build_signal_snapshot, summarize_snapshot
from adfm_core.ui import (
    PageHeader,
    inject_explorer_style,
    render_footer,
    render_kpi_cards,
    render_page_header,
    render_section_header,
)

st.set_page_config(page_title="Daily PM Brief", layout="wide")
inject_explorer_style()


@st.cache_data(ttl=900, show_spinner=False)
def load_brief_data():
    symbols = market_symbols()
    frames, missing = fetch_daily_ohlcv(symbols, period="3y")
    prices = close_panel(frames, symbols, adjusted=True)
    snapshot = build_signal_snapshot(prices)
    return snapshot, summarize_snapshot(snapshot), missing


snapshot, summary, missing = load_brief_data()
journal = load_decision_journal()
tables = daily_brief_tables(summary, snapshot, journal)
alerts = evaluate_alerts(
    summary,
    snapshot,
    data_health=current_data_health(),
    journal=journal,
)
new_alerts = new_alert_transitions(alerts)

render_page_header(
    PageHeader(
        title="Daily PM Brief",
        description="One morning operating view of regime, cross-asset reversals, data risk, and decisions requiring review.",
        as_of=f"Market data through {summary.as_of or 'unavailable'}",
        source_note="Shared market registry and local point-in-time decision ledger",
    )
)

render_kpi_cards(
    [
        (
            "Regime",
            summary.regime,
            f"Composite {summary.composite:+.2f}"
            if pd.notna(summary.composite)
            else "Unavailable",
        ),
        (
            "Breadth",
            f"{summary.breadth:.0%}",
            "Signals above the constructive threshold",
        ),
        (
            "Impulse",
            f"{summary.impulse:+.2f}" if pd.notna(summary.impulse) else "N/A",
            "Average 1-week versus 3-month score",
        ),
        (
            "Dispersion",
            f"{summary.dispersion:.2f}" if pd.notna(summary.dispersion) else "N/A",
            "Higher means less coherent tape",
        ),
        ("New alerts", str(len(new_alerts)), f"{len(alerts)} currently active"),
        (
            "Reviews due",
            str(len(tables["reviews"])),
            "Open decisions requiring PM attention",
        ),
    ]
)

if new_alerts:
    render_section_header(
        "New threshold crossings",
        "Only alerts that were inactive on the prior run appear here.",
    )
    for alert in new_alerts:
        message = f"{alert.title}: {alert.detail}"
        st.error(message) if alert.severity == "high" else st.warning(message)

left, right = st.columns([1.2, 1])
with left:
    render_section_header(
        "Largest risk changes",
        "Signals ranked by the absolute gap between short- and medium-horizon evidence.",
    )
    st.dataframe(
        tables["movers"],
        width="stretch",
        hide_index=True,
        column_config={
            "Composite": st.column_config.NumberColumn(format="%+.2f"),
            "Impulse": st.column_config.NumberColumn(format="%+.2f"),
        },
    )
with right:
    render_section_header(
        "Decision review queue",
        "Open trades with a reached review date, no review date, or missing invalidation.",
    )
    if tables["reviews"].empty:
        st.success("No journal decisions require review.")
    else:
        st.dataframe(
            tables["reviews"][
                ["Instrument", "Direction", "Status", "Review Date", "Review Reason"]
            ],
            width="stretch",
            hide_index=True,
        )

render_section_header(
    "Operating gaps",
    "Unavailable portfolio and catalyst inputs remain explicit rather than inferred.",
)
gap_cols = st.columns(3)
gap_cols[0].info(
    "Portfolio exposures: connect an approved position source or upload positions in a future private risk-console workflow."
)
gap_cols[1].info(
    "Catalysts: use the existing Event Risk + Catalyst Calendar for the current configured calendar."
)
gap_cols[2].info(
    f"Provider gaps: {len(missing)} market symbols failed; {len(tables['stale'])} signal rows are stale."
)

render_footer()
