# streamlit_app.py
# Unusual Options Flow: Rule-based market scanner using yfinance chains
# Notes:
# - Approximates “big flow” via notional, Vol/OI, OTM %, and DTE.
# - No tape-level sweep/venue metadata in free sources.
# - Scans preset universe (S&P500 top 200) or your custom list.
# - Persists daily snapshots for OI delta comparisons when run regularly.

import os
import math
import glob
import time
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

APP_TITLE = "Unusual Options Flow – Rule Scanner"
DATA_DIR = "data"
INGEST_DIR = "ingest"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INGEST_DIR, exist_ok=True)

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# ---------- Preset Universe ----------
SP500_TOP200 = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","LLY","AVGO","JPM","V","WMT",
    "XOM","UNH","JNJ","PG","MA","COST","HD","ABBV","ORCL","BAC","MRK","PEP","KO","NFLX",
    "ASML","AMD","TMUS","CRM","ADBE","CSCO","LIN","TMO","ACN","INTU","MCD","WFC","VZ",
    "CMCSA","TXN","ABT","DHR","IBM","PM","CAT","GE","PFE","NOW","AMAT","SPGI","NEE","NKE",
    "AMGN","LOW","RTX","ETN","HON","BKNG","BX","ISRG","PLTR","QCOM","UBER","SCHW","MDLZ",
    "AMT","INTC","UNP","SO","ADP","BLK","ELV","DE","MS","MMC","SYK","GS","CME","TJX","GILD",
    "PYPL","BA","C","PH","MO","LMT","CI","T","CVX","PANW","REGN","VRTX","KLAC","MU","FISV",
    "ZTS","EQIX","CSX","COP","BDX","EMR","ICE","HCA","AON","CDNS","MAR","SHW","WM","PGR",
    "HUM","ITW","DUK","EOG","ADSK","NSC","PXD","FDX","AEP","PLD","LRCX","ORLY","AZO","MRVL",
    "SBUX","ROST","PSA","TRV","MNST","WDAY","IDXX","CEG","PCAR","TT","F","MET","DG","D",
    "AIG","GM","CNC","FCX","CMG","CTAS","PRU","ED","HPQ","GIS","KR","HAL","CTVA","MSCI",
    "SNPS","HES","VLO","PSX","KHC","ADM","DVN","KDP","MCHP","WBA","NUE","LULU","ROK","RMD",
    "PAYX","CDW","TGT","PHM","LEN","CPRT","NEM","ON","AFL","ALL","A","ALB","AAL","UAL"
]

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.markdown("### Universe")
    universe_preset = st.selectbox("Preset", ["S&P 500 Top 200", "Custom"], index=0)

    if universe_preset == "Custom":
        tickers_input = st.text_area(
            "Tickers (comma/space separated)",
            value="AAPL, AMD, NVDA, TSLA, MSFT, META, AMZN, GOOGL",
            height=90
        )
        tickers = sorted(list({t.strip().upper()
                               for t in tickers_input.replace("\n", ",").split(",")
                               if t.strip()}))
    else:
        tickers = SP500_TOP200

    st.markdown("### Data Limits")
    max_expiries = st.number_input("Max expirations per ticker", 1, 12, 4)
    st.caption("Tip: 3 to 5 expirations is plenty to find short-dated activity.")

    st.markdown("### Thresholds")
    min_notional = st.number_input("Min notional per line, USD", 0, 100_000_000, 1_000_000, 100_000)
    min_vol_oi = st.number_input("Min Volume / OI", 0.0, 50.0, 0.5, 0.1)

    st.markdown("### Rule-based Scanner")
    rule_side = st.selectbox("Side", ["call", "put"], index=0)
    rule_otm_pct = st.number_input("OTM threshold %", 0.0, 200.0, 10.0, 1.0)
    rule_max_dte = st.number_input("Max DTE (days)", 1, 365, 30)
    st.caption("Example: 10 percent OTM calls with ≤ 30 DTE.")

    st.markdown("### Greeks Setup")
    risk_free_rate = st.number_input("Risk-free rate %", -1.0, 10.0, 1.5, 0.1)
    dividend_yield = st.number_input("Dividend yield %", 0.0, 20.0, 0.0, 0.1)

    st.markdown("### Auto-refresh")
    use_autorefresh = st.checkbox("Enable auto-refresh", value=False)
    refresh_secs = st.number_input("Refresh every N seconds", 10, 3600, 120, 10)

if use_autorefresh:
    st.experimental_set_query_params(ts=int(time.time()))

# ---------- Helpers ----------
def bs_delta(S, K, T, r, q, iv, is_call=True):
    try:
        if T <= 0 or iv <= 0 or S <= 0 or K <= 0:
            return np.nan
        d1 = (math.log(S / K) + (r - q + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
        def phi(x):  # standard normal CDF
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))
        return math.exp(-q * T) * (phi(d1) if is_call else -phi(-d1))
    except Exception:
        return np.nan

def snapshot_path(tag_date: str) -> str:
    return os.path.join(DATA_DIR, f"options_snapshot_{tag_date}.parquet")

def list_snapshots() -> List[str]:
    return sorted(glob.glob(os.path.join(DATA_DIR, "options_snapshot_*.parquet")))

def save_snapshot(df: pd.DataFrame):
    d = datetime.now().strftime("%Y%m%d")
    p = snapshot_path(d)
    try:
        df.to_parquet(p, index=False)
    except Exception:
        p = p.replace(".parquet", ".csv")
        df.to_csv(p, index=False)
    return p

def load_latest_snapshot() -> pd.DataFrame:
    files = list_snapshots()
    if not files:
        return pd.DataFrame()
    f = files[-1]
    try:
        if f.endswith(".parquet"):
            return pd.read_parquet(f)
        else:
            return pd.read_csv(f)
    except Exception:
        return pd.DataFrame()

def fetch_chain(ticker: str, max_exp: int) -> Tuple[pd.DataFrame, float]:
    tk = yf.Ticker(ticker)
    spot = np.nan
    try:
        spot = float(tk.fast_info.get("last_price")
                     or tk.info.get("regularMarketPrice")
                     or tk.history(period="1d").iloc[-1]["Close"])
    except Exception:
        pass
    try:
        expirations = tk.options or []
    except Exception:
        expirations = []
    rows = []
    for exp in expirations[:max_exp]:
        try:
            oc = tk.option_chain(exp)
            for side, df in [("call", oc.calls), ("put", oc.puts)]:
                if df is None or df.empty:
                    continue
                tmp = df.copy()
                tmp["ticker"] = ticker
                tmp["side"] = side
                tmp["expiration"] = pd.to_datetime(exp)
                rows.append(tmp)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(), spot
    return pd.concat(rows, ignore_index=True), spot

def enrich(df: pd.DataFrame, spot: float, r: float, q: float) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for c in ["volume","openInterest","impliedVolatility","strike","lastPrice","bid","ask"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["underlying_price"] = spot
    out["notional_usd"] = out["volume"].fillna(0) * spot * 100.0
    out["vol_oi"] = out.apply(lambda x: (x["volume"] / x["openInterest"])
                              if pd.notna(x.get("openInterest")) and x.get("openInterest", 0) > 0 else np.nan, axis=1)
    now_utc = pd.Timestamp.now(tz="UTC")
    out["expiration"] = pd.to_datetime(out["expiration"], utc=True)
    out["ttm_years"] = (out["expiration"] - now_utc).dt.total_seconds() / (365 * 24 * 3600)
    out.loc[out["ttm_years"] < 0, "ttm_years"] = 0.0
    out["delta"] = out.apply(
        lambda x: bs_delta(
            S=spot, K=x["strike"], T=max(float(x["ttm_years"]), 1e-6),
            r=r/100.0, q=q/100.0,
            iv=float(x["impliedVolatility"]) if pd.notna(x["impliedVolatility"]) else np.nan,
            is_call=True if x["side"] == "call" else False
        ), axis=1
    )
    out["moneyness"] = spot / out["strike"]
    keep = [
        "ticker","contractSymbol","side","expiration","strike","lastPrice","bid","ask",
        "volume","openInterest","vol_oi","impliedVolatility","delta","moneyness",
        "underlying_price","notional_usd","ttm_years"
    ]
    for k in keep:
        if k not in out.columns:
            out[k] = np.nan
    return out[keep]

def merge_prev_oi(curr: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    if curr.empty:
        return curr
    key = ["ticker","contractSymbol"]
    prev_slim = prev[key + ["openInterest"]].rename(columns={"openInterest":"openInterest_prev"}) if not prev.empty else pd.DataFrame(columns=key + ["openInterest_prev"])
    out = curr.merge(prev_slim, on=key, how="left")
    out["oi_change"] = out["openInterest"] - out["openInterest_prev"]
    out["oi_change_pct"] = np.where(out["openInterest_prev"].fillna(0) > 0, out["oi_change"] / out["openInterest_prev"], np.nan)
    return out

# ---------- Fetch + Build ----------
progress = st.progress(0.0, text="Fetching chains...")
spots = {}
rows = []
for i, tk in enumerate(tickers):
    try:
        chain, spot = fetch_chain(tk, max_expiries)
        spots[tk] = spot
        if not chain.empty:
            rows.append(enrich(chain, spot, risk_free_rate, dividend_yield))
    except Exception as e:
        st.warning(f"{tk}: {e}")
    progress.progress((i + 1) / max(1, len(tickers)), text=f"Fetched {i+1}/{len(tickers)}")
progress.empty()

if not rows:
    st.stop()

current = pd.concat(rows, ignore_index=True)
previous = load_latest_snapshot()
aug = merge_prev_oi(current, previous)
saved = save_snapshot(current)

# ---------- Unusual Flow View (thresholds) ----------
aug = aug.copy()
aug["days_to_exp"] = (pd.to_datetime(aug["expiration"], utc=True) - pd.Timestamp.now(tz="UTC")).dt.days
mask_unusual = (aug["notional_usd"] >= min_notional) & (aug["vol_oi"].fillna(0) >= min_vol_oi)
unusual = aug.loc[mask_unusual].copy()
unusual["moneyness_flag"] = np.where(
    unusual["moneyness"].between(0.95, 1.05), "~ATM",
    np.where(unusual["moneyness"] > 1.05, "ITM call / OTM put", "OTM call / ITM put")
)
unusual = unusual.sort_values(["notional_usd","vol_oi"], ascending=[False, False])

st.subheader("Unusual Flow (by thresholds)")
cols = [
    "ticker","side","expiration","days_to_exp","strike","underlying_price","lastPrice",
    "volume","openInterest","vol_oi","oi_change","oi_change_pct","impliedVolatility","delta",
    "notional_usd","moneyness_flag"
]
view = unusual[cols].copy()
view["expiration"] = pd.to_datetime(view["expiration"]).dt.date.astype(str)
fmt_pct = lambda x: "" if pd.isna(x) else f"{x*100:.1f}%"
fmt_usd = lambda x: "" if pd.isna(x) else f"${x:,.0f}"
view["oi_change_pct"] = view["oi_change_pct"].apply(fmt_pct)
view["impliedVolatility"] = view["impliedVolatility"].apply(fmt_pct)
view["delta"] = view["delta"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}")
view["vol_oi"] = view["vol_oi"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}")
view["notional_usd"] = view["notional_usd"].apply(fmt_usd)
view["underlying_price"] = view["underlying_price"].apply(lambda v: "" if pd.isna(v) else f"${v:,.2f}")
view["lastPrice"] = view["lastPrice"].apply(lambda v: "" if pd.isna(v) else f"${v:,.2f}")
st.dataframe(view, use_container_width=True, hide_index=True)

# ---------- Rule-based Market Scan ----------
st.subheader("Rule-based flags")
aug2 = aug.copy()
otm = rule_otm_pct / 100.0
if rule_side == "call":
    side_mask = aug2["side"].eq("call")
    moneyness_mask = aug2["strike"] >= (1 + otm) * aug2["underlying_price"]
else:
    side_mask = aug2["side"].eq("put")
    moneyness_mask = aug2["strike"] <= (1 - otm) * aug2["underlying_price"]

rule_mask = (
    side_mask &
    moneyness_mask &
    (aug2["days_to_exp"] >= 0) & (aug2["days_to_exp"] <= rule_max_dte) &
    (aug2["notional_usd"] >= min_notional) &
    (aug2["vol_oi"].fillna(0) >= min_vol_oi)
)
hits = aug2.loc[rule_mask].copy().sort_values(["notional_usd","vol_oi"], ascending=[False, False])

if hits.empty:
    st.info("No contracts matched the current rules in the selected universe.")
else:
    h = hits[[
        "ticker","side","expiration","strike","underlying_price","lastPrice","volume","openInterest",
        "vol_oi","impliedVolatility","delta","notional_usd","days_to_exp"
    ]].copy()
    h["expiration"] = pd.to_datetime(h["expiration"]).dt.date.astype(str)
    h["impliedVolatility"] = h["impliedVolatility"].apply(fmt_pct)
    h["vol_oi"] = h["vol_oi"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    h["delta"] = h["delta"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    h["underlying_price"] = h["underlying_price"].apply(lambda v: "" if pd.isna(v) else f"${v:,.2f}")
    h["lastPrice"] = h["lastPrice"].apply(lambda v: "" if pd.isna(v) else f"${v:,.2f}")
    h["notional_usd"] = h["notional_usd"].apply(fmt_usd)
    st.dataframe(h, use_container_width=True, hide_index=True)

# ---------- Aggregates ----------
st.subheader("By Ticker Aggregates (thresholded)")
agg = unusual.groupby(["ticker","side"], as_index=False).agg(
    total_notional=("notional_usd","sum"),
    lines=("contractSymbol","count"),
    avg_vol_oi=("vol_oi","mean"),
    sum_volume=("volume","sum"),
    sum_openInterest=("openInterest","sum")
).sort_values("total_notional", ascending=False)
agg["total_notional"] = agg["total_notional"].apply(fmt_usd)
agg["avg_vol_oi"] = agg["avg_vol_oi"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}")
st.dataframe(agg, use_container_width=True, hide_index=True)

st.caption(
    "Data: Yahoo Finance chains via yfinance. Flags are inferred using rules and thresholds; "
    "they are not trade-tape sweeps. Snapshot saved for OI history."
)
