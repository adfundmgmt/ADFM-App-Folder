from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="Event Risk Calendar", layout="wide")


# =========================================================
# Helpers
# =========================================================
def default_event_table() -> pd.DataFrame:
    today = date.today()
    rows = [
        {"Event": "CPI", "Date": today + timedelta(days=7), "Priority": "High"},
        {"Event": "FOMC", "Date": today + timedelta(days=14), "Priority": "High"},
        {"Event": "NFP", "Date": today + timedelta(days=21), "Priority": "High"},
        {"Event": "OPEX", "Date": today + timedelta(days=28), "Priority": "Medium"},
        {"Event": "PCE", "Date": today + timedelta(days=35), "Priority": "High"},
    ]
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_spy(period: str = "15y") -> pd.Series:
    raw = yf.download(
        "SPY",
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=True,
        threads=False,
    )

    if raw is None or raw.empty:
        return pd.Series(dtype=float)

    close = raw.get("Close")
    if close is None:
        return pd.Series(dtype=float)

    if isinstance(close, pd.DataFrame):
        if "SPY" in close.columns:
            close = close["SPY"]
        else:
            close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce").dropna()

    if close.empty:
        return pd.Series(dtype=float)

    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close.name = "SPY"
    return close.sort_index()


def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Event", "Date", "Priority"])

    out = df.copy()

    if "Event" not in out.columns:
        out["Event"] = ""
    if "Date" not in out.columns:
        out["Date"] = pd.NaT
    if "Priority" not in out.columns:
        out["Priority"] = "Medium"

    out["Event"] = out["Event"].astype(str).str.strip()
    out["Priority"] = out["Priority"].astype(str).str.strip().replace("", "Medium")
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()

    out = out[(out["Event"] != "") & out["Date"].notna()].copy()
    out = out.sort_values("Date").reset_index(drop=True)

    return out[["Event", "Date", "Priority"]]


def analyze_event_windows(
    spy: pd.Series,
    events_df: pd.DataFrame,
    pre_days: int,
    post_days: int,
) -> pd.DataFrame:
    if spy.empty or events_df.empty:
        return pd.DataFrame()

    idx = spy.index
    results = []

    for _, row in events_df.iterrows():
        event_name = str(row["Event"]).strip()
        priority = str(row["Priority"]).strip()
        event_date = pd.to_datetime(row["Date"]).normalize()

        nearest_pos = idx.get_indexer([event_date], method="nearest")[0]
        if nearest_pos < 0:
            continue

        matched_date = idx[nearest_pos]
        start_pos = max(0, nearest_pos - pre_days)
        end_pos = min(len(spy) - 1, nearest_pos + post_days)

        start_px = float(spy.iloc[start_pos])
        event_px = float(spy.iloc[nearest_pos])
        end_px = float(spy.iloc[end_pos])

        pre_move = (event_px / start_px - 1.0) if nearest_pos > start_pos else 0.0
        post_move = (end_px / event_px - 1.0) if end_pos > nearest_pos else 0.0
        total_move = (end_px / start_px - 1.0) if end_pos > start_pos else 0.0

        results.append(
            {
                "Event": event_name,
                "Input Date": event_date.date(),
                "Matched Trading Date": matched_date.date(),
                "Priority": priority,
                f"Pre {pre_days}D": pre_move,
                f"Post {post_days}D": post_move,
                f"Total {pre_days + post_days}D": total_move,
            }
        )

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return result_df

    numeric_cols = [f"Pre {pre_days}D", f"Post {post_days}D", f"Total {pre_days + post_days}D"]
    for col in numeric_cols:
        result_df[col] = pd.to_numeric(result_df[col], errors="coerce")

    return result_df


def style_display_df(result_df: pd.DataFrame, pre_days: int, post_days: int) -> pd.DataFrame:
    display_df = result_df.copy()
    pct_cols = [f"Pre {pre_days}D", f"Post {post_days}D", f"Total {pre_days + post_days}D"]

    for col in pct_cols:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").mul(100).round(2)

    return display_df


def make_bar_chart(display_df: pd.DataFrame, pre_days: int, post_days: int) -> go.Figure:
    pre_col = f"Pre {pre_days}D"
    post_col = f"Post {post_days}D"
    total_col = f"Total {pre_days + post_days}D"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=display_df["Event"],
            y=display_df[pre_col],
            name=pre_col,
            customdata=np.stack(
                [
                    display_df["Matched Trading Date"].astype(str),
                    display_df["Priority"].astype(str),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Matched Date: %{customdata[0]}<br>"
                "Priority: %{customdata[1]}<br>"
                "Return: %{y:.2f}%<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=display_df["Event"],
            y=display_df[post_col],
            name=post_col,
            customdata=np.stack(
                [
                    display_df["Matched Trading Date"].astype(str),
                    display_df["Priority"].astype(str),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Matched Date: %{customdata[0]}<br>"
                "Priority: %{customdata[1]}<br>"
                "Return: %{y:.2f}%<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=display_df["Event"],
            y=display_df[total_col],
            mode="lines+markers",
            name=total_col,
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Total Window: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="SPY Reaction Around Event Windows",
        barmode="group",
        height=500,
        xaxis_title="Event",
        yaxis_title="Pre / Post Return (%)",
        yaxis2=dict(
            title="Total Window Return (%)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def priority_score(priority: str) -> int:
    mapping = {"High": 3, "Medium": 2, "Low": 1}
    return mapping.get(str(priority).title(), 2)


# =========================================================
# Header
# =========================================================
st.title("Event Risk Calendar")
st.caption("Track upcoming catalysts and quantify historical SPY reaction around event windows.")


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("Event Window Settings")
    pre_days = st.slider("Days before event", min_value=1, max_value=10, value=3)
    post_days = st.slider("Days after event", min_value=1, max_value=10, value=3)

    st.markdown("---")
    st.header("About This Tool")
    st.markdown(
        """
This page helps you maintain a live catalyst list and quickly translate those dates into a market-risk lens.

It maps each event to the nearest SPY trading session, then calculates:
- pre-event drift over the selected lookback window
- post-event drift over the selected forward window
- total move across the full event window

Use it operationally, not mechanically. The point is to frame tape behavior around event clusters, compare expected path against realized path, and think through gross, hedge ratios, and optionality into catalysts.
"""
    )


# =========================================================
# Event editor
# =========================================================
st.markdown("### Upcoming Event List")

editable_events = st.data_editor(
    default_event_table(),
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Event": st.column_config.TextColumn("Event", required=True),
        "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
        "Priority": st.column_config.SelectboxColumn(
            "Priority",
            options=["High", "Medium", "Low"],
            required=True,
        ),
    },
)

events_df = clean_events(editable_events)

if events_df.empty:
    st.warning("Add at least one valid event name and date.")
    st.stop()


# =========================================================
# Data fetch
# =========================================================
spy = fetch_spy()

if spy.empty:
    st.error("Unable to fetch SPY history from Yahoo Finance.")
    st.stop()


# =========================================================
# Analysis
# =========================================================
result_df = analyze_event_windows(spy=spy, events_df=events_df, pre_days=pre_days, post_days=post_days)

if result_df.empty:
    st.warning("No valid event windows could be calculated.")
    st.stop()

display_df = style_display_df(result_df, pre_days=pre_days, post_days=post_days)

pre_col = f"Pre {pre_days}D"
post_col = f"Post {post_days}D"
total_col = f"Total {pre_days + post_days}D"

numeric_summary = result_df[[pre_col, post_col, total_col]].apply(pd.to_numeric, errors="coerce")
summary = numeric_summary.mean().mul(100)

hit_rate_post = (numeric_summary[post_col] > 0).mean() * 100
avg_abs_total = numeric_summary[total_col].abs().mean() * 100


# =========================================================
# Metrics
# =========================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Avg Pre {pre_days}D", f"{summary[pre_col]:.2f}%")
c2.metric(f"Avg Post {post_days}D", f"{summary[post_col]:.2f}%")
c3.metric("Avg Total Window", f"{summary[total_col]:.2f}%")
c4.metric(f"Post {post_days}D Hit Rate", f"{hit_rate_post:.1f}%")

st.markdown("### Event Window Price Reaction (SPY)")
st.dataframe(
    display_df[
        ["Event", "Input Date", "Matched Trading Date", "Priority", pre_col, post_col, total_col]
    ],
    use_container_width=True,
    hide_index=True,
)

fig = make_bar_chart(display_df=display_df, pre_days=pre_days, post_days=post_days)
st.plotly_chart(fig, use_container_width=True)


# =========================================================
# Risk framing
# =========================================================
st.markdown("### Operational Readout")

high_priority = display_df.copy()
high_priority["PriorityScore"] = high_priority["Priority"].map(priority_score)
high_priority = high_priority.sort_values(["PriorityScore", "Matched Trading Date"], ascending=[False, True])

next_event_text = "None"
if not high_priority.empty:
    next_row = high_priority.iloc[0]
    next_event_text = f"{next_row['Event']} on {next_row['Input Date']} ({next_row['Priority']})"

st.info(
    f"Next event focus: {next_event_text}. "
    f"Across the current list, the average absolute total move over the selected window is {avg_abs_total:.2f}%. "
    f"Use this as a first-pass framing tool for gross exposure, hedge ratios, and whether event premium is worth owning or fading."
)
