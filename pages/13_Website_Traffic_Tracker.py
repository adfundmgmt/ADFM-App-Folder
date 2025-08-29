# streamlit_app.py

import re
import datetime as dt
from typing import List, Optional

import numpy as pd
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# External source: Google Trends via pytrends (no API key required)
# pip install pytrends
from pytrends.request import TrendReq

st.set_page_config(page_title="Website Interest Tracker", page_icon="📈", layout="wide")

# =====================
# Helpers
# =====================

def normalize_domains(text: str) -> List[str]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    cleaned = []
    for p in parts:
        p = re.sub(r"^https?://", "", p, flags=re.I)
        p = re.sub(r"^www\.", "", p, flags=re.I)
        cleaned.append(p.lower())
    return list(dict.fromkeys(cleaned))  # de-dupe in order

@st.cache_data(show_spinner=False)
def fetch_trends(domains: List[str], timeframe: str, region: str, property_kind: str) -> pd.DataFrame:
    gprop = {"Web": "", "News": "news", "Images": "images", "YouTube": "youtube"}[property_kind]
    pytrends = TrendReq(hl="en-US", tz=0)

    # pytrends supports up to 5 terms per request; chunk if needed
    chunks = [domains[i:i+5] for i in range(0, len(domains), 5)]
    frames = []
    for chunk in chunks:
        pytrends.build_payload(kw_list=chunk, timeframe=timeframe, geo=region, gprop=gprop)
        df = pytrends.interest_over_time()
        if df.empty:
            continue
        df = df.drop(columns=[c for c in ["isPartial"] if c in df.columns])
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out = out.groupby(level=0, axis=1).max()  # if duplicates
    out.reset_index(inplace=True)
    out.rename(columns={"date": "Date"}, inplace=True)
    return out

def annotate_breakouts(df_long: pd.DataFrame, lookback: int = 90) -> pd.DataFrame:
    rows = []
    for k, g in df_long.groupby("Site"):
        s = g.set_index("Date")["Value"].astype(float)
        if len(s) < 10:
            continue
        recent = s.iloc[-1]
        roll_max = s.rolling(lookback, min_periods=max(10, lookback//3)).max()
        is_breakout = recent >= roll_max.iloc[-1]
        chg_7 = pct(s, 7)
        chg_30 = pct(s, 30)
        rows.append({"Site": k, "Latest": recent, "7d%": chg_7, "30d%": chg_30, "90dHighBreakout": bool(is_breakout)})
    return pd.DataFrame(rows)

def pct(s: pd.Series, n: int) -> float:
    if len(s) <= n or s.iloc[-n-1] == 0:
        return np.nan
    return (s.iloc[-1] / s.iloc[-n-1] - 1) * 100.0

# =====================
# Sidebar
# =====================

st.sidebar.title("📈 Website Interest Tracker")
st.sidebar.caption("Type domains. Data source: Google Trends interest over time. No GA4. No CSV.")

sites_input = st.sidebar.text_area("Websites (comma separated)", value="alibaba.com, tesla.com, openai.com")
region = st.sidebar.text_input("Region code (geo)", value="")

# Timeframe presets supported by Google Trends
preset = st.sidebar.selectbox("Timeframe", [
    "today 3-m", "today 12-m", "today 5-y", "all"
], index=1)

prop_kind = st.sidebar.selectbox("Search property", ["Web", "News", "Images", "YouTube"], index=0)

smooth = st.sidebar.slider("SMA window", 1, 12, 3, help="Simple moving average on the chart")

st.sidebar.markdown("---")
show_table = st.sidebar.checkbox("Show data table", value=False)

# =====================
# Main
# =====================

st.title("Website Trend Tracker")
st.caption("Interest over time for typed domains using Google Trends. Values are indexed 0 to 100 per Google methodology.")

sites = normalize_domains(sites_input)
if not sites:
    st.info("Enter at least one domain to begin")
    st.stop()

with st.spinner("Fetching Google Trends..."):
    df = fetch_trends(sites, timeframe=preset, region=region, property_kind=prop_kind)

if df.empty:
    st.warning("No trend data returned. Try fewer sites, another timeframe, or switch property.")
    st.stop()

# Long format for plotting
long = df.melt(id_vars=["Date"], var_name="Site", value_name="Value").sort_values("Date")

# Apply smoothing for display
if smooth > 1:
    long["Value"] = long.groupby("Site")["Value"].transform(lambda s: s.rolling(smooth, min_periods=1).mean())

# Chart
fig = go.Figure()
for site, g in long.groupby("Site"):
    fig.add_trace(go.Scatter(x=g["Date"], y=g["Value"], mode="lines", name=site))

fig.update_layout(
    height=520,
    margin=dict(l=10, r=10, t=35, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    yaxis_title="Interest (0-100)",
)

st.plotly_chart(fig, use_container_width=True)

# Breakout and momentum table
st.subheader("Signals")
sig = annotate_breakouts(long, lookback=90)
st.dataframe(sig.sort_values(["90dHighBreakout", "30d%", "7d%"], ascending=[False, False, False]), use_container_width=True)

# Notes
st.markdown("""
**How to read this**
- Values are relative interest from Google search, indexed per query. They are not absolute traffic counts. For cross-site comparison, the index is shared within the same call which allows relative ranking.
- Use it as a proxy for attention and top-of-funnel demand. For actual session counts, wire GA4 later if desired.
""")

# Optional raw table and download
if show_table:
    st.subheader("Raw data")
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="website_trends.csv", mime="text/csv")

# Footer
with st.expander("Tips"):
    st.markdown(
        """
        - Region codes use ISO geo codes accepted by Google Trends. Example: US, GB, TR. Leave blank for worldwide.
        - Timeframes: try `today 3-m`, `today 12-m`, `today 5-y`, or `all`.
        - Keep to 5 sites per comparison for the cleanest scaling.
        """
    )
