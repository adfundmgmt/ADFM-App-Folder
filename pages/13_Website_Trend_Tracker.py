# streamlit_app.py

import re
import time
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pytrends.request import TrendReq

st.set_page_config(page_title="Website Trend Tracker", page_icon="📈", layout="wide")

# =====================
# Defaults
# =====================
DEFAULT_SITES = [
    # AI front-end
    "openai.com", "anthropic.com", "perplexity.ai",
    # China
    "alibaba.com",
    # Cloud / hyperscalers
    "aws.amazon.com", "azure.microsoft.com", "cloud.google.com", "coreweave.com",
]

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
    # keep order, drop dups
    seen = set()
    out = []
    for c in cleaned:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

@st.cache_data(show_spinner=False)
def fetch_trends(domains: List[str], timeframe: str, region: str, property_kind: str,
                 per_chunk_sleep: float = 1.5, max_retries: int = 4, base_backoff: float = 2.0) -> pd.DataFrame:
    """Fetch Google Trends interest_over_time for up to N domains with retry/backoff.
    Returns a wide DataFrame with Date index and one column per domain.
    """
    gprop = {"Web": "", "News": "news", "Images": "images", "YouTube": "youtube"}[property_kind]
    pytrends = TrendReq(hl="en-US", tz=0)

    # pytrends supports up to 5 terms at once
    chunks = [domains[i:i+5] for i in range(0, len(domains), 5)]
    frames = []
    for chunk in chunks:
        # basic throttle between requests
        time.sleep(per_chunk_sleep)
        # build payload
        pytrends.build_payload(kw_list=chunk, timeframe=timeframe, geo=region, gprop=gprop)
        # retry with exponential backoff on rate limits
        attempt = 0
        last_err = None
        while attempt <= max_retries:
            try:
                df = pytrends.interest_over_time()
                if df is None or df.empty:
                    break
                df = df.drop(columns=[c for c in df.columns if c.lower() == "ispartial"], errors="ignore")
                frames.append(df)
                break
            except Exception as e:
                last_err = e
                sleep_for = base_backoff * (2 ** attempt) + np.random.random()
                time.sleep(sleep_for)
                attempt += 1
        else:
            # if loop did not break, raise last error
            raise last_err

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1)
    # if duplicate columns due to overlapping chunks, take max across
    out = out.groupby(level=0, axis=1).max()
    out.reset_index(inplace=True)
    out.rename(columns={"date": "Date"}, inplace=True)
    return out

def pct_change(series: pd.Series, periods: int) -> float:
    if len(series) <= periods:
        return np.nan
    prev = series.iloc[-periods-1]
    if prev == 0:
        return np.nan
    return (series.iloc[-1] / prev - 1.0) * 100.0

def build_signals(long_df: pd.DataFrame, lookback: int = 90) -> pd.DataFrame:
    rows = []
    for site, g in long_df.groupby("Site"):
        s = g.set_index("Date")["Value"].astype(float)
        if len(s) < 10:
            continue
        recent = s.iloc[-1]
        roll_max = s.rolling(lookback, min_periods=max(10, lookback // 3)).max()
        breakout = bool(recent >= roll_max.iloc[-1])
        rows.append({
            "Site": site,
            "Latest": recent,
            "7d%": pct_change(s, 7),
            "30d%": pct_change(s, 30),
            "90dHighBreakout": breakout,
        })
    df = pd.DataFrame(rows)
    return df

# =====================
# Sidebar
# =====================

st.sidebar.title("📈 Website Interest Tracker")
st.sidebar.caption("Type domains. Uses Google Trends. No GA4. No CSV.")

sites_input = st.sidebar.text_area(
    "Websites (comma separated)",
    value=", ".join(DEFAULT_SITES),
    height=100,
)
region = st.sidebar.text_input("Region code (geo)", value="")

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["today 3-m", "today 12-m", "today 5-y", "all"],
    index=1,
)
prop_kind = st.sidebar.selectbox("Search property", ["Web", "News", "Images", "YouTube"], index=0)

smooth = st.sidebar.slider("SMA window", 1, 12, 3)

st.sidebar.markdown("---")
auto_fetch = st.sidebar.checkbox("Auto fetch on first load", value=True)
fetch_click = st.sidebar.button("Fetch trends", type="primary")

# =====================
# Main
# =====================

st.title("Website Trend Tracker")
st.caption("Interest over time for typed domains using Google Trends. Values are indexed 0 to 100 per Google methodology.")

sites = normalize_domains(sites_input)
if not sites:
    st.info("Enter at least one domain to begin")
    st.stop()

# control when to fetch
should_fetch = False
if auto_fetch and "_autofetched" not in st.session_state:
    should_fetch = True
    st.session_state["_autofetched"] = True
if fetch_click:
    should_fetch = True

if should_fetch:
    with st.spinner("Fetching Google Trends..."):
        df = fetch_trends(sites, timeframe=timeframe, region=region, property_kind=prop_kind)
    if df.empty:
        st.warning("No trend data returned. Try fewer sites, another timeframe, or different property.")
        st.stop()

    # Reshape to long for plotting
    long = df.melt(id_vars=["Date"], var_name="Site", value_name="Value").sort_values("Date")

    # smooth for display if selected
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

    # Signals
    st.subheader("Signals")
    sig = build_signals(long, lookback=90)
    st.dataframe(sig.sort_values(["90dHighBreakout", "30d%", "7d%"], ascending=[False, False, False]), use_container_width=True)

    with st.expander("Notes"):
        st.markdown(
            """
            - Google Trends measures relative search interest. Use it as an attention proxy. It is not absolute traffic.
            - Compare no more than five at a time for the cleanest scaling. This app chunks if you enter more.
            - Region code examples: US, GB, DE, TR. Leave blank for worldwide.
            - Timeframes: today 3-m, today 12-m, today 5-y, all.
            """
        )
else:
    st.info("Click Fetch trends to load data.")
