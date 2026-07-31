from __future__ import annotations

from datetime import date, timedelta
from io import StringIO

import pandas as pd
import streamlit as st

from adfm_core.operations import (
    load_decision_journal,
    save_decision,
    weekly_process_review,
)
from adfm_core.signal_ledger import load_signal_history
from adfm_core.ui import (
    PageHeader,
    inject_explorer_style,
    render_footer,
    render_page_header,
    render_section_header,
)

st.set_page_config(page_title="Decision Journal", layout="wide")
inject_explorer_style()

render_page_header(
    PageHeader(
        title="Decision Journal + Weekly Review",
        description="Preserve the thesis, expected path, sizing, catalyst, invalidation, and platform state that existed when a decision was made.",
        source_note="Private local ledger excluded from Git",
    )
)

signal_history = load_signal_history()
latest_snapshot = (
    signal_history.sort_values("Captured At UTC").drop_duplicates("Key", keep="last")
    if not signal_history.empty
    else pd.DataFrame()
)

with st.form("new_decision", clear_on_submit=True):
    cols = st.columns(4)
    instrument = cols[0].text_input("Instrument")
    direction = cols[1].selectbox(
        "Direction", ["Long", "Short", "Relative Value", "Hedge"]
    )
    size = cols[2].text_input("Size / risk budget")
    trade_date = cols[3].date_input("Trade date", value=date.today())
    thesis = st.text_area("Thesis")
    catalyst = st.text_area("Catalyst")
    expected_path = st.text_area("Expected path and timing")
    invalidation = st.text_area("Explicit invalidation")
    review_date = st.date_input("Review date", value=date.today() + timedelta(days=7))
    submitted = st.form_submit_button("Record decision")

if submitted:
    try:
        updated = save_decision(
            {
                "Trade Date": trade_date.isoformat(),
                "Instrument": instrument,
                "Direction": direction,
                "Size": size,
                "Thesis": thesis,
                "Catalyst": catalyst,
                "Invalidation": invalidation,
                "Expected Path": expected_path,
                "Review Date": review_date.isoformat(),
                "Entry Regime": st.session_state.get("adfm_entry_regime", ""),
                "Entry Composite": st.session_state.get("adfm_entry_composite", pd.NA),
            },
            snapshot=latest_snapshot,
        )
        st.success(f"Decision recorded. Journal now contains {len(updated)} entries.")
    except ValueError as exc:
        st.error(str(exc))

journal = load_decision_journal()
tabs = st.tabs(["Open decisions", "Weekly process review", "Snapshot audit"])
with tabs[0]:
    render_section_header(
        "Open decisions",
        "The journal is a process record. Position and account data remain outside the repository.",
    )
    open_decisions = journal.loc[journal["Status"].astype(str).str.lower() == "open"]
    st.dataframe(
        open_decisions.drop(columns=["Entry Snapshot"], errors="ignore"),
        width="stretch",
        hide_index=True,
    )
with tabs[1]:
    if not journal.empty:
        review_id = st.selectbox(
            "Decision to review",
            journal["Decision ID"].dropna().astype(str).tolist(),
        )
        existing = journal.loc[journal["Decision ID"].astype(str) == review_id].iloc[-1]
        existing_outcome = existing.get("Outcome")
        existing_notes = existing.get("Review Notes")
        with st.form("decision_review"):
            review_cols = st.columns(3)
            status = review_cols[0].selectbox(
                "Status",
                ["Open", "Closed", "Invalidated", "Scaled", "Watch"],
                index=0,
            )
            outcome = review_cols[1].text_input(
                "Outcome",
                value="" if pd.isna(existing_outcome) else str(existing_outcome),
            )
            grade_options = ["", "A", "B", "C", "D", "F"]
            thesis_grade = review_cols[2].selectbox("Thesis grade", grade_options)
            grade_cols = st.columns(4)
            timing_grade = grade_cols[0].selectbox("Timing grade", grade_options)
            sizing_grade = grade_cols[1].selectbox("Sizing grade", grade_options)
            execution_grade = grade_cols[2].selectbox("Execution grade", grade_options)
            luck = grade_cols[3].selectbox("Luck", ["", "Helped", "Neutral", "Hurt"])
            review_notes = st.text_area(
                "Review notes",
                value="" if pd.isna(existing_notes) else str(existing_notes),
            )
            save_review = st.form_submit_button("Save review")
        if save_review:
            payload = existing.to_dict()
            payload.update(
                {
                    "Status": status,
                    "Outcome": outcome,
                    "Thesis Grade": thesis_grade,
                    "Timing Grade": timing_grade,
                    "Sizing Grade": sizing_grade,
                    "Execution Grade": execution_grade,
                    "Luck": luck,
                    "Review Notes": review_notes,
                }
            )
            save_decision(payload)
            st.success("Decision review saved.")
    review = weekly_process_review(journal)
    st.dataframe(
        review.drop(columns=["Entry Snapshot"], errors="ignore"),
        width="stretch",
        hide_index=True,
    )
with tabs[2]:
    decision_ids = journal["Decision ID"].dropna().astype(str).tolist()
    if not decision_ids:
        st.info("No entry snapshots have been recorded.")
    else:
        selected_id = st.selectbox("Decision ID", decision_ids)
        payload = journal.loc[
            journal["Decision ID"].astype(str) == selected_id, "Entry Snapshot"
        ].iloc[-1]
        try:
            st.dataframe(
                pd.read_json(StringIO(payload)),
                width="stretch",
                hide_index=True,
            )
        except (TypeError, ValueError):
            st.warning("The stored entry snapshot is unavailable.")

render_footer()
