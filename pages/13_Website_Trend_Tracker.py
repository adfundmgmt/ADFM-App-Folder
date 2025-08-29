# streamlit_app.py

import re
import time
from typing import List, Tuple

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

def _chunk_with_anchor(domains: List[str], anchor: str) -> List[List[str]]:
    """Build chunks of up to 5 terms, ensuring every chunk includes the anchor for cross-chunk scaling."""
    others = [d for d in domains if d != anchor]
    chunks = []
    # first chunk: up to 5 including anchor
    first = [anchor] + others[:4]
    chunks.append(first)
    # remaining chunks: anchor + next four
    i = 4
    while i < len(others):
        chunk = [anchor] + others[i:i+4]
        chunks.append(chunk)
        i += 4
    return chunks

def _stitch_with_anchor(frames: List[pd.DataFrame], anchor: str) -> pd.DataFrame:
    """Scale each frame to the first frame using the anchor column's overlap average ratio."""
    # Normalize column names and set Date
    normed = []
    for df in frames:
        df = df.copy()
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        df = df.rename(columns={"date": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])  # ensure ts
        normed.append(df)

    base = normed[0]
    if anchor not in base.columns:
        # If anchor missing (rare), just concat
        out = base
        for df in normed[1:]:
            out = out.merge(df, on="Date", how="outer")
        out = out.sort_values("Date").reset_index(drop=True)
        return out

    # Use the base anchor as reference level
    out = base.copy()
    for df in normed[1:]:
        if anchor not in df.columns:
            # Merge without scaling if anchor absent (fallback)
            out = out.merge(df, on="Date", how="outer")
            continue
        # compute scaling on overlapping dates where both anchors > 0
        merged_anchor = pd.merge(out[["Date", anchor]], df[["Date", anchor]], on="Date", how="inner", suffixes=("_base", "_new"))
        overlap = merged_anchor[(merged_anchor[f"{anchor}_base"] > 0) & (merged_anchor[f"{anchor}_new"] > 0)]
        if len(overlap) == 0:
            scale = 1.0
        else:
            # ratio to scale new frame so that its anchor matches base anchor on average
            scale = (overlap[f"{anchor}_base"].astype(float).mean()) / (overlap[f"{anchor}_new"].astype(float).mean())
        df_scaled = df.copy()
        for c in df_scaled.columns:
            if c not in {"Date"}:
                df_scaled[c] = df_scaled[c].astype(float) * scale
        out = out.merge(df_scaled, on="Date", how="outer")

    # If duplicate columns resulted (same site from multiple chunks), take the max
    out = out.groupby(level=0, axis=1).max()
    out = out.sort_values("Date").reset_index()
    return out

@st.cache_data(show_spinner=False)
def fetch_trends(domains: List[str], timeframe: str, region: str, property_kind: str,
                 per_chunk_sleep: float = 1.5, max_retries: int = 4, base_backoff: float = 2.0) -> pd.DataFrame:
    """Fetch Google Trends interest_over_time for up to N domains with retry/backoff and
    correct cross-chunk comparability via an anchor term (first domain).
    Returns a wide DataFrame with Date and one column per domain, all on a consistent scale.
    """
    if not domains:
        return pd.DataFrame()
    anchor = domains[0]
    gprop = {"Web": "", "News": "news", "Images": "images", "YouTube": "youtube"}[property_kind]
    pytrends = TrendReq(hl="en-US", tz=0)

    frames = []
    for chunk in _chunk_with_anchor(domains, anchor):
        time.sleep(per_chunk_sleep)
        pytrends.build_payload(kw_list=chunk, timeframe=timeframe, geo=region, gprop=gprop)
        attempt = 0
        last_err = None
        while attempt <= max_retries:
            try:
                df = pytrends.interest_over_time()
                if df is None or df.empty:
                    break
                frames.append(df)
                break
            except Exception as e:
                last_err = e
                sleep_for = base_backoff * (2 ** attempt) + np.random.random()
                time.sleep(sleep_for)
                attempt += 1
        else:
            raise last_err

    if not frames:
        return pd.DataFrame()

    out = _stitch_with_anchor(frames, anchor=anchor)
    out.rename(columns={"date": "Date"}, inplace=True)
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"])  # ensure ts
    return out

# Date-aware pct change using asof for nearest prior observation
def pct_change_days(s: pd.Series, days: int) -> float:
    s = s.dropna().astype(float)
    if s.empty:
        return np.nan
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    last_dt = s.index[-1]
    ref_dt = last_dt - pd.Timedelta(days=days)
    # align to the last available on or before ref_dt
    s_asof = s[:ref_dt]
    if s_asof.empty or s_asof.iloc[-1] == 0:
        return np.nan
    return (s.iloc[-1] / s_asof.iloc[-1] - 1.0) * 100.0

def build_signals(df_wide: pd.DataFrame, lookback: int = 90) -> pd.DataFrame:
    rows = []
    for c in [col for col in df_wide.columns if col != "Date"]:
        s = df_wide.set_index("Date")[c].astype(float)
        if len(s) < 10:
            continue
        recent = s.iloc[-1]
        roll_max = s.rolling(lookback, min_periods=max(10, lookback // 3)).max()
        breakout = bool(recent >= roll_max.iloc[-1])
        rows.append({
            "Site": c,
            "Latest": float(recent),
            "7d%": pct_change_days(s, 7),
            "30d%": pct_change_days(s, 30),
            "90dHighBreakout": breakout,
        })
    return pd.DataFrame(rows)

# =====================
# Sidebar
# =====================

st.sidebar.title("Website Interest Tracker")
with st.sidebar.expander("About This Tool", expanded=True):
    st.write(
        """
        Track **relative web search interest** for any set of domains using Google Trends.
        Useful as an **attention proxy** for AI adoption and data-center buildout narratives.
        - Type domains, choose timeframe and region, click **Fetch trends**.
        - The app compares more than five sites by **stitching chunks with an anchor** (the first domain)
          so scales are comparable.
        - Outputs: timeseries chart and a **Signals** table with 7d/30d momentum and 90d-high breakouts.
        """
    )

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

smooth = st.sidebar.slider("SMA window (chart only)", 1, 12, 3)

st.sidebar.markdown("---")
auto_fetch = st.sidebar.checkbox("Auto fetch on first load", value=True)
fetch_click = st.sidebar.button("Fetch trends", type="primary")

# =====================
# Main
# =====================

st.title("Website Trend Tracker")
st.caption("Google Trends interest over time. Values are indexed 0 to 100; cross-chunk comparability fixed via anchor scaling.")

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
        df_wide = fetch_trends(sites, timeframe=timeframe, region=region, property_kind=prop_kind)
    if df_wide.empty:
        st.warning("No trend data returned. Try fewer sites, another timeframe, or different property.")
        st.stop()

    # Build signals from RAW values (not smoothed)
    st.subheader("Signals")
    sig = build_signals(df_wide, lookback=90)
    st.dataframe(sig.sort_values(["90dHighBreakout", "30d%", "7d%"], ascending=[False, False, False]), use_container_width=True)

    # Prepare long for chart; smoothing applied only for display
    long = df_wide.melt(id_vars=["Date"], var_name="Site", value_name="Value").sort_values("Date")
    if smooth > 1:
        long["Value"] = long.groupby("Site")["Value"].transform(lambda s: s.rolling(smooth, min_periods=1).mean())

    st.subheader("Timeseries")
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

    with st.expander("Notes"):
        st.markdown(
            """
            - Google Trends is **relative interest**, not absolute traffic. Use alongside your GA/BI data.
            - Scaling across more than 5 sites is corrected using an **anchor** (the first domain you list).
            - 7d/30d changes are **calendar-day aware** and use the nearest available prior observation.
            - Region codes: US, GB, DE, TR; blank for worldwide.
            """
        )
else:
    st.info("Click Fetch trends to load data.")
