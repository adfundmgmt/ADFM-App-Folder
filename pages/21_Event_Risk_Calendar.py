from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Event Risk Calendar", layout="wide")
st.title("Event Risk Calendar")
st.caption("Track upcoming catalysts and quantify historical SPY reaction around event windows.")

with st.sidebar:
    st.header("Event Window Settings")
    pre_days = st.slider("Days before event", 1, 10, 3)
    post_days = st.slider("Days after event", 1, 10, 3)

st.markdown("### Upcoming Event List")

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

editable_events = st.data_editor(
    default_event_table(),
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
)

@st.cache_data(ttl=3600)
def fetch_spy(period: str = "10y") -> pd.Series:
    raw = yf.download("SPY", period=period, interval="1d", progress=False, auto_adjust=True)
    if raw.empty:
        return pd.Series(dtype=float)
    return raw["Close"].dropna()

spy = fetch_spy()
if spy.empty:
    st.error("Unable to fetch SPY history.")
    st.stop()


window_results = []
for _, row in editable_events.iterrows():
    event_name = str(row.get("Event", "Unknown"))
    try:
        event_date = pd.to_datetime(row.get("Date")).tz_localize(None)
    except Exception:
        continue

    nearest_idx = spy.index.get_indexer([event_date], method="nearest")[0]
    if nearest_idx < 0:
        continue

    start_idx = max(0, nearest_idx - pre_days)
    end_idx = min(len(spy) - 1, nearest_idx + post_days)

    pre_move = spy.iloc[nearest_idx] / spy.iloc[start_idx] - 1 if nearest_idx > start_idx else 0.0
    post_move = spy.iloc[end_idx] / spy.iloc[nearest_idx] - 1 if end_idx > nearest_idx else 0.0
    total_move = spy.iloc[end_idx] / spy.iloc[start_idx] - 1 if end_idx > start_idx else 0.0

    window_results.append(
        {
            "Event": event_name,
            "Date": spy.index[nearest_idx].date(),
            f"Pre {pre_days}D": pre_move,
            f"Post {post_days}D": post_move,
            f"Total {pre_days + post_days}D": total_move,
        }
    )

result_df = pd.DataFrame(window_results)

if result_df.empty:
    st.warning("No valid event dates were provided.")
    st.stop()

for col in [f"Pre {pre_days}D", f"Post {post_days}D", f"Total {pre_days + post_days}D"]:
    result_df[col] = (result_df[col] * 100).round(2)

st.markdown("### Event Window Price Reaction (SPY)")
st.dataframe(result_df, use_container_width=True, hide_index=True)

summary = result_df[[f"Pre {pre_days}D", f"Post {post_days}D", f"Total {pre_days + post_days}D"]].mean()
c1, c2, c3 = st.columns(3)
c1.metric(f"Avg Pre {pre_days}D", f"{summary.iloc[0]:.2f}%")
c2.metric(f"Avg Post {post_days}D", f"{summary.iloc[1]:.2f}%")
c3.metric("Avg Total Window", f"{summary.iloc[2]:.2f}%")

fig = go.Figure()
fig.add_trace(go.Bar(x=result_df["Event"], y=result_df[f"Pre {pre_days}D"], name=f"Pre {pre_days}D"))
fig.add_trace(go.Bar(x=result_df["Event"], y=result_df[f"Post {post_days}D"], name=f"Post {post_days}D"))
fig.update_layout(
    title="Pre/Post Event Return Profile (SPY)",
    barmode="group",
    yaxis_title="Return (%)",
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

st.info(
    "Use this page as an operational risk calendar: maintain your event list weekly, then review pre/post drift behavior "
    "to size gross, hedge ratios, and option structures around catalyst windows."
)
