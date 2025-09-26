# streamlit_app.py
# Unusual Options Flow Tracker using Yahoo Finance and public sources
# Notes:
# - Relies on yfinance for options chains. Yahoo does NOT expose individual trade prints, sweeps, or block tags.
# - We approximate "big flow" using notional size, Volume/Open Interest ratio, and day-over-day OI change.
# - The app persists daily snapshots of OI to compute changes over time. Run it daily for best results.
# - Optional: you can drop CSVs of external option trade logs into the ./ingest folder and the app will unify them.
# - No manual refresh button is included. Optionally enable auto-refresh in the sidebar.

import os
import io
import math
import json
import time
import glob
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

############################################################
# Config & Paths
############################################################
APP_TITLE = "Unusual Options Flow Tracker"
DATA_DIR = "data"  # local persistence for snapshots
INGEST_DIR = "ingest"  # optional external CSVs with option flows
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INGEST_DIR, exist_ok=True)

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

############################################################
# Sidebar Controls
############################################################
with st.sidebar:
    st.markdown("### Universe")
    tickers_input = st.text_area(
        "Tickers (comma or whitespace separated)",
        value="AAPL, AMD, NVDA, TSLA, MSFT, META, AMZN, GOOGL",
        height=90,
        help="Add liquid names first to avoid API throttles."
    )
    tickers = sorted(list({t.strip().upper() for t in tickers_input.replace("\n", ",").split(",") if t.strip()}))

    st.markdown("### Scan Parameters")
    max_expiries = st.number_input(
        "Max expirations per ticker", min_value=1, max_value=12, value=4, step=1,
        help="To manage rate limits, cap the number of expirations pulled per ticker."
    )
    min_notional = st.number_input(
        "Min notional per line, USD", min_value=0, value=1_000_000, step=100_000,
        help="Underlying price x 100 x option volume."
    )
    min_vol_oi = st.number_input(
        "Min Volume / OI", min_value=0.0, value=0.5, step=0.1,
        help="Higher suggests flow that exceeds existing open interest."
    )
    only_unusual = st.checkbox(
        "Show only unusual flow (meets thresholds)", value=True,
        help="Filter to lines that cross notional and Volume/OI thresholds."
    )

    st.markdown("### Auto-Refresh (optional)")
    use_autorefresh = st.checkbox("Enable auto-refresh", value=False)
    refresh_secs = st.number_input("Refresh every N seconds", min_value=10, value=120, step=10)

    st.markdown("### Advanced")
    st.caption("Historical OI deltas require at least one prior snapshot saved by this app.")
    risk_free_rate = st.number_input("Risk-free rate %, annualized", min_value=-1.0, value=1.5, step=0.1)
    dividend_yield = st.number_input("Dividend yield %, annualized", min_value=0.0, value=0.0, step=0.1)

if use_autorefresh:
    st.experimental_set_query_params(ts=int(time.time()))  # change URL param to trigger reruns

############################################################
# Helper Functions
############################################################

def bs_delta(S, K, T, r, q, iv, is_call=True):
    """Black-Scholes delta using continuous compounding.
    S: spot, K: strike, T: years to expiry, r: rate, q: dividend yield, iv: implied vol (decimal)
    """
    try:
        if T <= 0 or iv <= 0 or S <= 0 or K <= 0:
            return np.nan
        from math import log, sqrt, exp
        d1 = (math.log(S / K) + (r - q + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
        # standard normal CDF
        def phi(x):
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))
        if is_call:
            return math.exp(-q * T) * phi(d1)
        else:
            return -math.exp(-q * T) * phi(-d1)
    except Exception:
        return np.nan


def load_previous_snapshot_path(snapshot_date: str) -> str:
    return os.path.join(DATA_DIR, f"options_snapshot_{snapshot_date}.parquet")


def list_snapshots() -> List[str]:
    return sorted(glob.glob(os.path.join(DATA_DIR, "options_snapshot_*.parquet")))


def save_snapshot(df: pd.DataFrame):
    snapshot_date = datetime.now().strftime("%Y%m%d")
    path = load_previous_snapshot_path(snapshot_date)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        # fallback to csv if parquet not available
        path = path.replace(".parquet", ".csv")
        df.to_csv(path, index=False)
    return path


def load_most_recent_snapshot() -> pd.DataFrame:
    files = list_snapshots()
    if not files:
        return pd.DataFrame()
    latest = files[-1]
    try:
        if latest.endswith(".parquet"):
            return pd.read_parquet(latest)
        else:
            return pd.read_csv(latest)
    except Exception:
        return pd.DataFrame()


def fetch_option_chain(ticker: str, max_exp: int) -> Tuple[pd.DataFrame, float]:
    tk = yf.Ticker(ticker)
    spot = float('nan')
    try:
        spot = float(tk.fast_info.get("last_price") or tk.info.get("regularMarketPrice") or tk.history(period="1d").iloc[-1]["Close"])  # robust fallback
    except Exception:
        pass

    expirations = []
    try:
        expirations = tk.options
    except Exception:
        pass

    chains = []
    if expirations:
        for exp in expirations[:max_exp]:
            try:
                oc = tk.option_chain(exp)
                calls = oc.calls.copy()
                puts = oc.puts.copy()
                for side, df in [("call", calls), ("put", puts)]:
                    if df is None or df.empty:
                        continue
                    df["ticker"] = ticker
                    df["side"] = side
                    df["expiration"] = pd.to_datetime(exp)
                    chains.append(df)
            except Exception:
                continue
    if not chains:
        return pd.DataFrame(), spot
    all_df = pd.concat(chains, ignore_index=True)
    return all_df, spot


def compute_metrics(df: pd.DataFrame, spot: float, r: float, q: float) -> pd.DataFrame:
    if df.empty:
        return df
    # Normalize columns present in yfinance
    # Expected columns: ['contractSymbol','lastTradeDate','strike','lastPrice','bid','ask','change','percentChange','volume','openInterest','impliedVolatility','inTheMoney','contractSize','currency']
    df = df.copy()
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")
    df["openInterest"] = pd.to_numeric(df.get("openInterest"), errors="coerce")
    df["impliedVolatility"] = pd.to_numeric(df.get("impliedVolatility"), errors="coerce")
    df["strike"] = pd.to_numeric(df.get("strike"), errors="coerce")
    df["lastPrice"] = pd.to_numeric(df.get("lastPrice"), errors="coerce")

    # Notional approximation
    df["underlying_price"] = spot
    df["notional_usd"] = df["volume"].fillna(0) * spot * 100.0

    # Volume / OI ratio
    df["vol_oi"] = df.apply(lambda x: (x["volume"] / x["openInterest"]) if x.get("openInterest", 0) not in [0, np.nan, None] else np.nan, axis=1)

    # Time to expiry in years
    now = datetime.now(timezone.utc)
    df["ttm_years"] = (pd.to_datetime(df["expiration"]).dt.tz_localize("UTC") - now).dt.total_seconds() / (365.0 * 24 * 3600)
    df.loc[df["ttm_years"] < 0, "ttm_years"] = 0.0

    # Delta approximation via Black-Scholes using IV
    r_d = r / 100.0
    q_d = q / 100.0
    df["delta"] = df.apply(lambda x: bs_delta(
        S=spot,
        K=x["strike"],
        T=max(float(x["ttm_years"]), 1e-6),
        r=r_d,
        q=q_d,
        iv=float(x["impliedVolatility"]) if pd.notna(x["impliedVolatility"]) else np.nan,
        is_call=True if x["side"] == "call" else False
    ), axis=1)

    # Moneyness
    df["moneyness"] = spot / df["strike"]

    # Clean contract size if present
    if "contractSize" in df.columns:
        df["contractSize"] = df["contractSize"].fillna("REGULAR")

    # Select compact view
    keep = [
        "ticker","contractSymbol","side","expiration","strike","lastPrice","bid","ask",
        "volume","openInterest","vol_oi","impliedVolatility","delta","moneyness",
        "underlying_price","notional_usd","ttm_years"
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    return df[keep]


def merge_with_previous_oi(curr_df: pd.DataFrame, prev_df: pd.DataFrame) -> pd.DataFrame:
    if curr_df.empty:
        return curr_df
    curr_df = curr_df.copy()
    key_cols = ["ticker","contractSymbol"]
    prev = prev_df[key_cols + ["openInterest"]].rename(columns={"openInterest": "openInterest_prev"}) if not prev_df.empty else pd.DataFrame(columns=key_cols + ["openInterest_prev"])
    curr_df = curr_df.merge(prev, on=key_cols, how="left")
    curr_df["oi_change"] = curr_df["openInterest"] - curr_df["openInterest_prev"]
    curr_df["oi_change_pct"] = np.where(
        curr_df["openInterest_prev"].fillna(0) > 0,
        curr_df["oi_change"] / curr_df["openInterest_prev"],
        np.nan
    )
    return curr_df


def read_external_flows() -> pd.DataFrame:
    # Optional: unify any CSV files in ./ingest with columns like
    # [datetime, ticker, side, strike, expiration, size, price, notional, exchange]
    files = sorted(glob.glob(os.path.join(INGEST_DIR, "*.csv")))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "ticker" in df.columns and "expiration" in df.columns:
                # normalize
                df["expiration"] = pd.to_datetime(df["expiration"])  # try parse
                df["source_file"] = os.path.basename(f)
                dfs.append(df)
        except Exception:
            continue
    if dfs:
        ext = pd.concat(dfs, ignore_index=True)
        return ext
    return pd.DataFrame()

############################################################
# Data Pipeline
############################################################
progress = st.progress(0.0, text="Fetching chains...")

rows = []
spots = {}
for i, tk in enumerate(tickers):
    try:
        chain, spot = fetch_option_chain(tk, max_expiries)
        spots[tk] = spot
        if not chain.empty:
            enriched = compute_metrics(chain, spot, r=risk_free_rate, q=dividend_yield)
            rows.append(enriched)
    except Exception as e:
        st.warning(f"{tk}: {e}")
    progress.progress((i + 1) / max(1, len(tickers)), text=f"Fetched {i+1}/{len(tickers)}")

progress.empty()

if rows:
    current_df = pd.concat(rows, ignore_index=True)
else:
    st.stop()

# Load previous snapshot to compute OI deltas
previous_df = load_most_recent_snapshot()
aug = merge_with_previous_oi(current_df, previous_df)

# Save today snapshot for future comparisons
saved_path = save_snapshot(current_df)

############################################################
# Filtering and Display
############################################################
# Apply thresholds
mask = (
    (aug["notional_usd"] >= min_notional) &
    (aug["vol_oi"].fillna(0) >= min_vol_oi)
)
filtered = aug[mask].copy() if only_unusual else aug.copy()

# Derived flags
filtered["days_to_exp"] = (pd.to_datetime(filtered["expiration"]) - pd.Timestamp.utcnow()).dt.days
filtered["moneyness_flag"] = np.where(filtered["moneyness"].between(0.95, 1.05), "~ATM",
                               np.where(filtered["moneyness"] > 1.05, "ITM call / OTM put", "OTM call / ITM put"))

# Sort by notional, then vol_oi
filtered = filtered.sort_values(["notional_usd","vol_oi"], ascending=[False, False])

st.subheader("Flow Summary")
st.caption("Approximation based on option volumes and OI. Not a trade tape.")

summary_cols = [
    "ticker","side","expiration","days_to_exp","strike","underlying_price","lastPrice",
    "volume","openInterest","vol_oi","oi_change","oi_change_pct","impliedVolatility","delta",
    "notional_usd","moneyness_flag"
]

# Pretty formatting
def fmt_pct(x):
    return "" if pd.isna(x) else f"{x*100:.1f}%"

def fmt_usd(x):
    return "" if pd.isna(x) else f"${x:,.0f}"

show = filtered[summary_cols].copy()
show["expiration"] = pd.to_datetime(show["expiration"]).dt.date.astype(str)
show["oi_change_pct"] = show["oi_change_pct"].apply(fmt_pct)
show["impliedVolatility"] = show["impliedVolatility"].apply(fmt_pct)
show["delta"] = show["delta"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}")
show["vol_oi"] = show["vol_oi"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}")
show["notional_usd"] = show["notional_usd"].apply(fmt_usd)
show["underlying_price"] = show["underlying_price"].apply(lambda v: "" if pd.isna(v) else f"${v:,.2f}")
show["lastPrice"] = show["lastPrice"].apply(lambda v: "" if pd.isna(v) else f"${v:,.2f}")

st.dataframe(show, use_container_width=True, hide_index=True)

############################################################
# Universe Level Stats
############################################################
st.subheader("By Ticker Aggregates")
agg = filtered.groupby(["ticker","side"], as_index=False).agg(
    total_notional=("notional_usd","sum"),
    lines=("contractSymbol","count"),
    avg_vol_oi=("vol_oi","mean"),
    sum_volume=("volume","sum"),
    sum_openInterest=("openInterest","sum"),
)
agg = agg.sort_values(["total_notional"], ascending=False)
agg_disp = agg.copy()
agg_disp["total_notional"] = agg_disp["total_notional"].apply(fmt_usd)
agg_disp["avg_vol_oi"] = agg_disp["avg_vol_oi"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}")
st.dataframe(agg_disp, use_container_width=True, hide_index=True)

############################################################
# External Feeds (optional)
############################################################
with st.expander("External ingest (CSV)"):
    st.write(
        "Drop CSV files into ./ingest with columns like "
        "[datetime, ticker, side, strike, expiration, size, price, notional, exchange]. "
        "They will be unified and shown here."
    )
    ext = read_external_flows()
    if not ext.empty:
        st.dataframe(ext, use_container_width=True, hide_index=True)
    else:
        st.info("No external CSVs detected.")

############################################################
# Footer / Diagnostics
############################################################
st.caption(
    "Data: Yahoo Finance option chains via yfinance. Flow is inferred using Volume/OI and OI deltas. "
    f"Saved snapshot: {os.path.basename(saved_path)}. Run daily to build OI history."
)
