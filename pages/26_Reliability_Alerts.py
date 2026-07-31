from __future__ import annotations

from pathlib import Path

import streamlit as st

from adfm_core.data_registry import market_symbols
from adfm_core.market_data import close_panel, fetch_daily_ohlcv
from adfm_core.observability import current_data_health
from adfm_core.operations import (
    evaluate_alerts,
    load_decision_journal,
    new_alert_transitions,
    reliability_report,
)
from adfm_core.pm_cockpit import build_signal_snapshot, summarize_snapshot
from adfm_core.ui import (
    PageHeader,
    inject_explorer_style,
    render_footer,
    render_page_header,
    render_section_header,
)

st.set_page_config(page_title="Reliability + Alerts", layout="wide")
inject_explorer_style()
ROOT = Path(__file__).resolve().parents[1]


@st.cache_data(ttl=900, show_spinner=False)
def load_alert_snapshot():
    symbols = market_symbols()
    frames, missing = fetch_daily_ohlcv(symbols, period="3y")
    prices = close_panel(frames, symbols, adjusted=True)
    snapshot = build_signal_snapshot(prices)
    return snapshot, summarize_snapshot(snapshot), missing


snapshot, summary, missing = load_alert_snapshot()
health = current_data_health()
journal = load_decision_journal()
active_alerts = evaluate_alerts(
    summary,
    snapshot,
    data_health=health,
    journal=journal,
)
new_alerts = new_alert_transitions(active_alerts)
reliability = reliability_report(data_health=health, repository_root=ROOT)

render_page_header(
    PageHeader(
        title="Reliability + Threshold Alerts",
        description="Audit provider failures, stale sources, persistence, calculation version, CI readiness, and sparse PM threshold crossings.",
        as_of=f"Market data through {summary.as_of or 'unavailable'}",
        source_note="Shared observability state and repository health contracts",
    )
)

render_section_header(
    "Platform reliability",
    "Status is explicit for provider coverage, snapshots, ledgers, calculation contract, tests, and fallbacks.",
)
st.dataframe(reliability, width="stretch", hide_index=True)

left, right = st.columns(2)
with left:
    render_section_header(
        "Active alerts",
        "Current conditions above threshold. Alert keys remain active until the condition clears.",
    )
    if not active_alerts:
        st.success("No thresholds are currently breached.")
    for alert in active_alerts:
        st.warning(f"{alert.title}: {alert.detail}")
with right:
    render_section_header(
        "New transitions",
        "Conditions that crossed from inactive to active on this run.",
    )
    if not new_alerts:
        st.success("No new threshold crossings.")
    for alert in new_alerts:
        st.error(
            f"{alert.title}: {alert.detail}"
        ) if alert.severity == "high" else st.warning(f"{alert.title}: {alert.detail}")

with st.expander("Provider failures"):
    st.dataframe(missing, width="stretch", hide_index=True)

st.caption(
    "Alerts are currently in-platform and stateful. External delivery requires an approved private destination and credentials."
)
render_footer()
