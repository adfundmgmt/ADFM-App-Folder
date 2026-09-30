from __future__ import annotations

from datetime import date, timedelta
from typing import List

import pandas as pd
import streamlit as st

from adfm_core import catalyst_calendar_page as base
from adfm_core.ui import render_sidebar_about

TITLE = "Catalyst Calendar"


def _dated_calendar(start: date, horizon_days: int, include_fed: bool) -> pd.DataFrame:
    """Return the existing recurring calendar with single-date event labels."""
    df = base._build_rule_calendar(start, horizon_days, include_fed)
    if df.empty:
        return df

    replacements = {
        "CPI Inflation Window": "CPI Inflation",
        "PPI Inflation Window": "PPI Inflation",
        "PCE Inflation Window": "PCE Inflation",
        "JOLTS Job Openings Window": "JOLTS Job Openings",
        "ISM Manufacturing Window": "ISM Manufacturing",
        "ISM Services Window": "ISM Services",
        "Retail Sales Window": "Retail Sales",
        "FOMC Decision Window": "FOMC Decision",
        "GDP Release Window": "GDP Release",
        "Quarterly Treasury Refunding Window": "Quarterly Treasury Refunding",
    }
    df["Event"] = df["Event"].replace(replacements)
    return df


def _format_event_date(d: date) -> str:
    return d.strftime("%b %d, %Y")


def _format_days(days: int) -> str:
    if days == 0:
        return "Today"
    if days == 1:
        return "Tomorrow"
    return f"In {days} days"


def render_catalyst_calendar() -> None:
    st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
            .block-container {padding-top: 2.4rem; padding-bottom: 2rem; max-width: 1580px;}
            .section-title {font-size:1.03rem; font-weight:760; color:#0f172a; margin-top:.85rem; margin-bottom:.45rem;}
            .section-note {font-size:.78rem; color:#64748b; margin-top:-.20rem; margin-bottom:.55rem; line-height:1.4;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    base.inject_institutional_tool_finish()

    with st.sidebar:
        render_sidebar_about("20_Catalyst_Calendar.py")
        st.header("Controls")
        horizon_days = st.select_slider("Event horizon", options=[14, 30, 60, 90, 120, 180], value=90)
        include_macro = st.checkbox("Include recurring macro catalysts", value=True)
        include_fed = st.checkbox("Include FOMC dates", value=True)
        hide_low = st.checkbox("Hide low-risk rows", value=False)
        st.divider()
        st.header("Custom Events")
        custom_text = st.text_area(
            "Paste custom event CSV",
            value="",
            height=145,
            placeholder=(
                "Date,Event,Type,Region,Why It Matters\n"
                "2026-09-16,FOMC Decision,Fed,U.S.,Policy rate and press conference catalyst"
            ),
        )

    today = date.today()
    market = base._fetch_market(min(date(today.year, 1, 1) - timedelta(days=10), today - timedelta(days=460)).isoformat())
    stress_bonus, stress_label = base._market_stress(market)

    frames: List[pd.DataFrame] = []
    if include_macro:
        frames.append(_dated_calendar(today, horizon_days, include_fed))

    custom = base._parse_custom_events(custom_text)
    if not custom.empty:
        custom["Source"] = "Custom"
        frames.append(custom)

    calendar = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not calendar.empty:
        if "Precision" not in calendar.columns:
            calendar["Precision"] = "Rule"
        if "Source" not in calendar.columns:
            calendar["Source"] = "Calendar rule"
        calendar["Precision"] = calendar["Precision"].fillna("Rule")
        calendar["Source"] = calendar["Source"].fillna("Custom")
        calendar = calendar[(calendar["Date"] >= today) & (calendar["Date"] <= today + timedelta(days=horizon_days))]
        calendar = calendar.drop_duplicates(subset=["Date", "Event", "Type"], keep="last")
        calendar = base._score_events(calendar, today, stress_bonus)
        if hide_low:
            calendar = calendar[calendar["Risk Score"] >= 65].reset_index(drop=True)

    base.render_page_header(
        base.PageHeader(
            title=TITLE,
            description="Confirmed agency dates plus deterministic market-calendar events and the latest macro prints defining the setup into each catalyst.",
            eyebrow="ADFM Risk + Catalysts",
        )
    )

    if calendar.empty:
        st.info("No events to show. Enable recurring macro catalysts or paste a custom CSV.")
        return

    st.caption(f"Volatility backdrop: {stress_label} · Event-risk add-on: {stress_bonus:+.1f}")

    st.markdown("<div class='section-title'>Upcoming Catalyst Dates</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-note'>Confirmed = published by the named source. Rule-based = deterministic market-calendar convention, not an estimated macro release date.</div>",
        unsafe_allow_html=True,
    )
    decision = calendar[["Date", "Days", "Event", "Type", "Precision", "Source", "Risk Score", "Exposure", "Action"]].copy()
    decision["Date"] = pd.to_datetime(decision["Date"])
    decision["When"] = decision["Days"].map(lambda x: _format_days(int(x)))
    decision["Status"] = decision["Precision"].replace({"Official": "Confirmed", "Rule": "Rule-based", "Custom": "Custom", "Estimated": "Estimated"})
    decision["Risk"] = decision["Risk Score"].map(lambda x: base._risk_label(float(x)))
    decision = decision[["Date", "When", "Event", "Type", "Status", "Source", "Risk", "Risk Score", "Exposure", "Action"]]
    st.dataframe(
        decision, width="stretch", hide_index=True, height=390,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="MMM DD, YYYY"),
            "Risk Score": st.column_config.NumberColumn("Risk Score", format="%.0f"),
        },
    )

    with st.expander("Catalyst charts and market backdrop", expanded=False, on_change="rerun") as charts:
        if charts.open:
            st.plotly_chart(base._timeline(calendar, today), use_container_width=True)
            perf = base._build_market_table(market, today)
            if perf.empty:
                st.info("Market data unavailable.")
            else:
                st.plotly_chart(base._heatmap(perf), use_container_width=True)

    with st.expander("Latest macro prints", expanded=False, on_change="rerun") as macro_detail:
        if macro_detail.open:
            macro_panel, macro_status = base._fetch_macro(date(today.year - 3, 1, 1).isoformat(), today.isoformat())
            st.caption("Latest and previous values from primary U.S. releases distributed through FRED.")
            macro = base._macro_prints(macro_panel)
            if macro.empty:
                st.info("Primary macro data is temporarily unavailable.")
            else:
                st.dataframe(macro, use_container_width=True, hide_index=True, height=420)
            with st.expander("Macro data status"):
                if not macro_status.empty:
                    st.dataframe(macro_status[["key", "symbol", "provider", "data_through", "status"]], use_container_width=True, hide_index=True)

    with st.expander("Full event details", expanded=False, on_change="rerun") as event_detail:
        if event_detail.open:
            details = calendar.copy()
            details["Date"] = pd.to_datetime(details["Date"])
            details["When"] = details["Days"].map(lambda x: _format_days(int(x)))
            details["Status"] = details["Precision"].replace({"Official": "Confirmed", "Rule": "Rule-based", "Custom": "Custom", "Estimated": "Estimated"})
            st.dataframe(
                details[["Date", "When", "Event", "Type", "Status", "Source", "Region", "Risk Score", "Cluster", "Why It Matters", "Exposure", "Action"]],
                use_container_width=True,
                hide_index=True,
                column_config={"Date": st.column_config.DateColumn("Date", format="MMM DD, YYYY")},
            )

    base.render_footer()
