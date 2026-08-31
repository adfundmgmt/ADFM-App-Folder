from __future__ import annotations

from datetime import date, timedelta
from typing import List

import pandas as pd
import streamlit as st

from adfm_core import catalyst_calendar_page as base


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
            .metric-card {background: linear-gradient(180deg,#fff 0%,#fafafa 100%); border:1px solid #e5e7eb; border-radius:14px; padding:13px 15px 10px; min-height:98px; box-shadow:0 1px 4px rgba(15,23,42,.05);}
            .metric-label {font-size:.70rem; color:#64748b; text-transform:uppercase; letter-spacing:.055em; margin-bottom:.42rem;}
            .metric-value {font-size:1.12rem; font-weight:760; color:#0f172a; line-height:1.18;}
            .metric-footnote {font-size:.76rem; color:#94a3b8; margin-top:.42rem; line-height:1.35;}
            .section-title {font-size:1.03rem; font-weight:760; color:#0f172a; margin-top:.85rem; margin-bottom:.45rem;}
            .section-note {font-size:.78rem; color:#64748b; margin-top:-.20rem; margin-bottom:.55rem; line-height:1.4;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    base.inject_institutional_tool_finish()

    with st.sidebar:
        st.header("About This Tool")
        st.markdown(
            "Forward catalyst planner for macro releases, FOMC decisions, options expirations, "
            "Treasury supply, quarter-end flows, and user-defined events. Each catalyst is shown "
            "on one assigned calendar date rather than as a vague event window."
        )
        st.caption(
            "Recurring macro dates are calendar-planning dates generated from release patterns. "
            "Confirm official agency schedules before trading directly around a release."
        )
        st.divider()
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
    macro_panel, macro_status = base._fetch_macro(date(today.year - 3, 1, 1).isoformat(), today.isoformat())

    frames: List[pd.DataFrame] = []
    if include_macro:
        frames.append(_dated_calendar(today, horizon_days, include_fed))

    custom = base._parse_custom_events(custom_text)
    if not custom.empty:
        frames.append(custom)

    calendar = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not calendar.empty:
        calendar = calendar[(calendar["Date"] >= today) & (calendar["Date"] <= today + timedelta(days=horizon_days))]
        calendar = calendar.drop_duplicates(subset=["Date", "Event", "Type"], keep="last")
        calendar = base._score_events(calendar, today, stress_bonus)
        if hide_low:
            calendar = calendar[calendar["Risk Score"] >= 65].reset_index(drop=True)

    perf = base._build_market_table(market, today)
    base.render_page_header(
        base.PageHeader(
            title=TITLE,
            description="Exact catalyst dates plus the latest macro prints that define the setup going into each event.",
            eyebrow="ADFM Risk + Catalysts",
        )
    )

    if calendar.empty:
        st.info("No events to show. Enable recurring macro catalysts or paste a custom CSV.")
        return

    next_event = calendar.iloc[0]
    highest = calendar.sort_values("Risk Score", ascending=False).iloc[0]
    next_week = calendar[calendar["Days"] <= 7]
    cluster_days = int((calendar["Cluster"] > 0).sum())

    cols = st.columns(5)
    with cols[0]:
        base._metric_card(
            "Next Catalyst",
            str(next_event["Event"]),
            f"{_format_event_date(next_event['Date'])} · {_format_days(int(next_event['Days']))}",
            base.TYPE_COLORS.get(str(next_event["Type"]), base.RISK_COLORS["neutral"]),
        )
    with cols[1]:
        base._metric_card(
            "Highest Risk",
            str(highest["Event"]),
            f"{_format_event_date(highest['Date'])} · {base._risk_label(float(highest['Risk Score']))} risk, score {float(highest['Risk Score']):.0f}",
            base.RISK_COLORS["high"] if float(highest["Risk Score"]) >= 82 else base.RISK_COLORS["medium"],
        )
    with cols[2]:
        base._metric_card(
            "Next 7 Days",
            str(len(next_week)),
            f"{int((next_week['Risk Score'] >= 82).sum())} high-risk event(s)",
            base.RISK_COLORS["high"] if int((next_week["Risk Score"] >= 82).sum()) else base.RISK_COLORS["neutral"],
        )
    with cols[3]:
        base._metric_card(
            "Clustered Days",
            str(cluster_days),
            "Same-day or nearby catalysts",
            base.RISK_COLORS["medium"] if cluster_days else base.RISK_COLORS["neutral"],
        )
    with cols[4]:
        base._metric_card(
            "Vol Backdrop",
            stress_label,
            f"+{stress_bonus:.1f} added to event score" if stress_bonus else "No risk-score add-on",
            base.RISK_COLORS["high"] if stress_bonus >= 5 else base.RISK_COLORS["neutral"],
        )

    left, right = st.columns([1.12, 0.88])
    with left:
        st.markdown("<div class='section-title'>Catalyst Tape</div>", unsafe_allow_html=True)
        st.plotly_chart(base._timeline(calendar, today), use_container_width=True)
    with right:
        st.markdown("<div class='section-title'>Market Backdrop: Today, 1W, 1M, 3M, YTD</div>", unsafe_allow_html=True)
        if perf.empty:
            st.info("Market data unavailable.")
        else:
            st.plotly_chart(base._heatmap(perf), use_container_width=True)

    st.markdown("<div class='section-title'>Latest Macro Prints</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-note'>The actual numbers behind the recurring catalyst dates. Latest and previous values update from primary U.S. data distributed through FRED.</div>",
        unsafe_allow_html=True,
    )
    macro = base._macro_prints(macro_panel)
    if macro.empty:
        st.info("Primary macro data is temporarily unavailable.")
    else:
        st.dataframe(macro, use_container_width=True, hide_index=True, height=420)

    st.markdown("<div class='section-title'>Upcoming Catalyst Dates</div>", unsafe_allow_html=True)
    decision = calendar[["Date", "Days", "Event", "Type", "Risk Score", "Exposure", "Action"]].copy()
    decision["Date"] = decision["Date"].map(_format_event_date)
    decision["When"] = decision["Days"].map(lambda x: _format_days(int(x)))
    decision["Risk"] = decision["Risk Score"].map(lambda x: base._risk_label(float(x)))
    decision["Risk Score"] = decision["Risk Score"].map(lambda x: f"{float(x):.0f}")
    decision = decision[["Date", "When", "Event", "Type", "Risk", "Risk Score", "Exposure", "Action"]]
    st.dataframe(decision, use_container_width=True, hide_index=True, height=390)

    with st.expander("Full event details"):
        details = calendar.copy()
        details["Date"] = details["Date"].map(_format_event_date)
        details["When"] = details["Days"].map(lambda x: _format_days(int(x)))
        st.dataframe(
            details[["Date", "When", "Event", "Type", "Region", "Risk Score", "Cluster", "Why It Matters", "Exposure", "Action"]],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Macro data status"):
        if not macro_status.empty:
            st.dataframe(
                macro_status[["key", "symbol", "provider", "data_through", "status"]],
                use_container_width=True,
                hide_index=True,
            )

    base.render_footer()
