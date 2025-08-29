# streamlit_app.py

import os
import io
import json
import datetime as dt
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional GA4 import. The app runs without GA4 if the package is missing.
try:
    from google.analytics.data_v1beta import BetaAnalyticsDataClient
    from google.analytics.data_v1beta.types import DateRange, Dimension, Metric, RunReportRequest
    GA4_AVAILABLE = True
except Exception:
    GA4_AVAILABLE = False

# =====================
# App Config
# =====================
st.set_page_config(page_title="Website Traffic Tracker", page_icon="📈", layout="wide")

st.sidebar.title("📈 Website Traffic Tracker")
st.sidebar.caption("Source: GA4 API or CSV export")

# =====================
# Utilities
# =====================

@st.cache_data(show_spinner=False)
def zscore_anomalies(series: pd.Series, lookback: int = 14, z: float = 2.5) -> pd.Series:
    """Return boolean mask where values are anomalous by rolling z score."""
    s = series.astype(float)
    roll_mean = s.rolling(lookback, min_periods=max(3, lookback // 2)).mean()
    roll_std = s.rolling(lookback, min_periods=max(3, lookback // 2)).std(ddof=0)
    zscores = (s - roll_mean) / roll_std.replace(0, np.nan)
    return zscores.abs() >= z

@st.cache_data(show_spinner=False)
def parse_ga4_service_account(sa_json_text: str):
    """Create a GA4 client from a service account JSON string."""
    from google.oauth2 import service_account
    from google.analytics.data_v1beta import BetaAnalyticsDataClient

    info = json.loads(sa_json_text)
    creds = service_account.Credentials.from_service_account_info(info)
    scoped = creds.with_scopes(["https://www.googleapis.com/auth/analytics.readonly"])
    client = BetaAnalyticsDataClient(credentials=scoped)
    return client

@st.cache_data(show_spinner=False)
def fetch_ga4_timeseries(client, property_id: str, start_date: str, end_date: str,
                         granularity: str = "daily",
                         dims: Optional[List[str]] = None,
                         metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """Fetch GA4 timeseries using RunReport. Returns a tidy DataFrame."""
    if dims is None:
        dims = ["date"]
    if metrics is None:
        metrics = ["totalUsers", "activeUsers", "newUsers", "sessions"]

    if granularity == "weekly" and "week" not in dims:
        dims = ["week"] + [d for d in dims if d != "date"]
    elif granularity == "monthly" and "month" not in dims:
        dims = ["month"] + [d for d in dims if d != "date"]

    request = RunReportRequest(
        property=f"properties/{property_id}",
        date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        dimensions=[Dimension(name=d) for d in dims],
        metrics=[Metric(name=m) for m in metrics],
    )
    resp = client.run_report(request)

    rows = []
    for r in resp.rows:
        row = {d.name: v.value for d, v in zip(resp.dimension_headers, r.dimension_values)}
        row.update({m.name: float(v.value) for m, v in zip(resp.metric_headers, r.metric_values)})
        rows.append(row)

    df = pd.DataFrame(rows)
    # Normalize date fields
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    if "week" in df.columns:
        df["date"] = pd.to_datetime(df["week"] + "-1", format="%Y-W%W-%w")
    if "month" in df.columns and "date" not in df.columns:
        df["date"] = pd.to_datetime(df["month"] + "-01")
    df = df.sort_values("date")
    return df

@st.cache_data(show_spinner=False)
def load_csv(upload: io.BytesIO) -> pd.DataFrame:
    df = pd.read_csv(upload)
    # Attempt to coerce common date column names
    for c in ["date", "Date", "day", "Day"]:
        if c in df.columns:
            df["date"] = pd.to_datetime(df[c])
            break
    if "date" not in df.columns:
        raise ValueError("CSV must include a date column. Example headers: date,totalUsers,sessions,pageViews")
    return df

# =====================
# Sidebar Controls
# =====================

source = st.sidebar.radio("Data source", ["GA4 API", "CSV upload"], index=0)
start_default = (dt.date.today() - dt.timedelta(days=90)).isoformat()
end_default = dt.date.today().isoformat()
start_date = st.sidebar.date_input("Start", value=pd.to_datetime(start_default))
end_date = st.sidebar.date_input("End", value=pd.to_datetime(end_default))
agg = st.sidebar.selectbox("Granularity", ["daily", "weekly", "monthly"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Metrics**")
metric_choices = ["totalUsers", "activeUsers", "newUsers", "sessions", "screenPageViews", "engagedSessions"]
selected_metrics = st.sidebar.multiselect("Select up to 6", metric_choices, default=["totalUsers", "sessions", "screenPageViews"])

# =====================
# Data Ingest
# =====================

df: Optional[pd.DataFrame] = None
error_msg = None

if source == "GA4 API":
    if not GA4_AVAILABLE:
        st.warning("google-analytics-data is not installed. Switch to CSV upload or add it to requirements.txt")
    property_id = st.sidebar.text_input("GA4 Property ID", "123456789")
    sa_file = st.sidebar.file_uploader("Service account JSON", type=["json"], help="Create a GA4 service account and grant it Viewer on your property")
    sa_text = None
    if sa_file is not None:
        sa_text = sa_file.read().decode("utf-8")

    if st.sidebar.checkbox("Fetch GA4 data", value=False):
        try:
            if sa_text is None:
                st.stop()
            client = parse_ga4_service_account(sa_text)
            df = fetch_ga4_timeseries(
                client=client,
                property_id=property_id,
                start_date=pd.to_datetime(start_date).date().isoformat(),
                end_date=pd.to_datetime(end_date).date().isoformat(),
                granularity=agg,
                dims=["date"],
                metrics=selected_metrics or ["totalUsers", "sessions"],
            )
        except Exception as e:
            error_msg = str(e)
else:
    csv_file = st.sidebar.file_uploader("Upload CSV export", type=["csv"], help="Headers example: date,totalUsers,sessions,screenPageViews")
    if csv_file is not None:
        try:
            df = load_csv(csv_file)
        except Exception as e:
            error_msg = str(e)

# =====================
# Main Layout
# =====================

st.title("Website Traffic Tracker")
st.caption("Track users, sessions, and page views with anomalies, top pages, sources, and cohort views")

if error_msg:
    st.error(error_msg)

if df is None or df.empty:
    st.info("Load data using GA4 or upload a CSV to see charts")
    st.stop()

# Basic normalization
num_cols = [c for c in df.columns if c not in {"date"}]
for c in num_cols:
    try:
        df[c] = pd.to_numeric(df[c])
    except Exception:
        pass

df = df.sort_values("date").reset_index(drop=True)

# KPIs
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

kpi_cols = st.columns(3)
metric_names = [m for m in ["totalUsers", "sessions", "screenPageViews"] if m in df.columns]
metric_names = metric_names[:3]

for i, m in enumerate(metric_names):
    delta = ((latest[m] - prev[m]) / prev[m] * 100.0) if prev[m] else 0.0
    kpi_cols[i].metric(label=m, value=f"{latest[m]:,.0f}", delta=f"{delta:+.1f}%")

# Timeseries chart
st.subheader("Timeseries")
value_col = metric_names[0] if metric_names else selected_metrics[0]
fig_ts = px.line(df, x="date", y=[c for c in df.columns if c in selected_metrics], markers=False)
fig_ts.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig_ts, use_container_width=True)

# Anomaly detection on the first metric
st.subheader("Anomalies")
lookback = st.slider("Rolling window", 7, 56, 21, step=1)
thresh = st.slider("Z threshold", 1.5, 4.0, 2.5, step=0.1)
mask = zscore_anomalies(df[value_col], lookback=lookback, z=thresh)

anom_df = df.loc[mask, ["date", value_col]].copy()
fig_anom = go.Figure()
fig_anom.add_trace(go.Scatter(x=df["date"], y=df[value_col], mode="lines", name=value_col))
fig_anom.add_trace(go.Scatter(x=anom_df["date"], y=anom_df[value_col], mode="markers", name="anomaly", marker=dict(size=10)))
fig_anom.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig_anom, use_container_width=True)

col_a, col_b = st.columns([1, 1])

with col_a:
    st.subheader("Top days")
    top_days = df.sort_values(value_col, ascending=False).head(15)
    st.dataframe(top_days[["date", value_col]].reset_index(drop=True))

with col_b:
    st.subheader("Moving averages")
    for w in [7, 14, 28]:
        df[f"ma_{w}"] = df[value_col].rolling(w, min_periods=w//2).mean()
    fig_ma = px.line(df, x="date", y=[value_col, "ma_7", "ma_14", "ma_28"]).update_layout(height=320)
    st.plotly_chart(fig_ma, use_container_width=True)

st.markdown("---")

# Optional page-level breakdown if present
page_cols = [c for c in df.columns if c.lower() in {"pagepath", "pagetitle", "page", "landingpage"}]
if page_cols:
    key = page_cols[0]
    st.subheader("Pages")
    agg_pages = df.groupby(key, as_index=False)[value_col].sum().sort_values(value_col, ascending=False).head(25)
    fig_pages = px.bar(agg_pages, x=value_col, y=key, orientation="h")
    fig_pages.update_layout(height=600)
    st.plotly_chart(fig_pages, use_container_width=True)
else:
    st.caption("Upload a CSV with a page column to unlock the Pages view. Example header: pagePath")

# Optional source breakdown if present
source_cols = [c for c in df.columns if c.lower() in {"source", "medium", "sourcemedium", "channel"}]
if source_cols:
    key = source_cols[0]
    st.subheader("Sources")
    agg_src = df.groupby(key, as_index=False)[value_col].sum().sort_values(value_col, ascending=False).head(20)
    fig_src = px.bar(agg_src, x=key, y=value_col)
    st.plotly_chart(fig_src, use_container_width=True)
else:
    st.caption("Upload a CSV with source or medium to unlock the Sources view")

# =====================
# Notes and Setup
# =====================
with st.expander("Setup notes"):
    st.markdown(
        """
        **GA4 mode**
        1. Create a Google Cloud project and service account.
        2. Grant the service account Viewer on your GA4 property in Admin.
        3. Upload the service account JSON and enter your Property ID.
        
        **CSV mode**
        - Include columns like: `date,totalUsers,sessions,screenPageViews,pagePath,source`.
        - Dates can be in `YYYY-MM-DD` and will be parsed automatically.
        
        **Deploy**
        - Add to `requirements.txt`: `streamlit plotly pandas numpy google-analytics-data` (GA4 optional)
        - Run: `streamlit run streamlit_app.py`
        """
    )
