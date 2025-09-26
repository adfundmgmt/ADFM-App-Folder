# streamlit_app.py (Safe ASCII baseline)
# Unusual Options Flow – Free UOA Scanner
# - Avoids emojis / pandas Styler / exotic APIs to prevent parse issues.
# - Industry-style UOA rule: Volume >= OI, Notional >= floor, DTE <= cap, OTM by threshold.
# - Z-score vs recent snapshots (approx 30 days, using saved parquet/csv files).

import os
import glob
import math
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

APP_TITLE = "Unusual Options Flow Tracker"
DATA_DIR = "data"
INGEST_DIR = "ingest"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INGEST_DIR, exist_ok=True)

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

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

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown("### Universe")
    preset = st.selectbox("Preset", ["S&P 500 Top 200","Custom"], index=0)
    if preset == "Custom":
        tickers_input = st.text_area("Tickers (comma/space)", value="AAPL, AMD, NVDA, TSLA")
        parts = tickers_input.replace("\n", ",").replace(" ", ",").split(",")
        tickers = sorted(list({t.strip().upper() for t in parts if t.strip()}))
    else:
        tickers = SP500_TOP200

    st.markdown("### Data Limits")
    max_expiries = st.number_input("Max expirations per ticker", min_value=1, max_value=12, value=4, step=1)

    st.markdown("### UOA Thresholds")
    min_notional = st.number_input("Min notional per line, USD", min_value=0, max_value=100_000_000, value=500_000, step=100_000)
    rule_otm_pct = st.number_input("OTM threshold %", min_value=0.0, max_value=200.0, value=10.0, step=1.0)
    rule_max_dte = st.number_input("Max DTE (days)", min_value=1, max_value=365, value=30, step=1)
    extra_min_vol_oi = st.number_input("Extra floor for Vol/OI (optional)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)

    st.markdown("### Greeks Setup")
    risk_free_rate = st.number_input("Risk-free rate %", min_value=-1.0, max_value=10.0, value=1.5, step=0.1)
    dividend_yield = st.number_input("Dividend yield %", min_value=0.0, max_value=20.0, value=0.0, step=0.1)

    st.markdown("### Auto-refresh")
    use_autorefresh = st.checkbox("Enable auto-refresh", value=False)
    refresh_secs = st.number_input("Refresh every N seconds", min_value=10, max_value=3600, value=120, step=10)

if use_autorefresh:
    # Streamlit query params setter; keep ASCII only
    st.query_params = {"ts": str(int(time.time()))}

# -------------------------
# Helpers
# -------------------------

def bs_delta(S: float, K: float, T: float, r: float, q: float, iv: float, is_call: bool) -> float:
    try:
        if T <= 0 or iv <= 0 or S <= 0 or K <= 0:
            return float("nan")
        d1 = (math.log(S / K) + (r - q + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
        # standard normal CDF
        return math.exp(-q * T) * (0.5 * (1.0 + math.erf(d1 / math.sqrt(2))) if is_call else -0.5 * (1.0 - math.erf(d1 / math.sqrt(2))))
    except Exception:
        return float("nan")


def list_snapshots() -> List[str]:
    return sorted(glob.glob(os.path.join(DATA_DIR, "options_snapshot_*.parquet")))


def save_snapshot(df: pd.DataFrame) -> str:
    tag = datetime.now().strftime("%Y%m%d")
    p = os.path.join(DATA_DIR, f"options_snapshot_{tag}.parquet")
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
        return pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
    except Exception:
        return pd.DataFrame()


def load_history(max_files: int = 40) -> pd.DataFrame:
    files = list_snapshots()
    if not files:
        return pd.DataFrame()
    files = files[-max_files:]
    out = []
    for f in files:
        try:
            df = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
            tag = os.path.splitext(os.path.basename(f))[0].replace("options_snapshot_", "")
            df["snapshot_date"] = tag
            out.append(df)
        except Exception:
            pass
    if not out:
        return pd.DataFrame()
    hist = pd.concat(out, ignore_index=True)
    for c in ["notional_usd","impliedVolatility","ttm_years","moneyness","side","ticker"]:
        if c not in hist.columns:
            hist[c] = np.nan
    hist["DTE"] = (hist.get("ttm_years", pd.Series(np.nan, index=hist.index)).astype(float) * 365.0).round()
    def m_band(m):
        try:
            d = abs(float(m) - 1.0)
        except Exception:
            return "unknown"
        if d <= 0.05:
            return "ATM"
        if d <= 0.10:
            return "Near-OTM"
        if d <= 0.20:
            return "OTM"
        return "Deep-OTM"
    def dte_band(d):
        try:
            x = float(d)
        except Exception:
            return "unknown"
        if x <= 7:
            return "0-7d"
        if x <= 30:
            return "8-30d"
        if x <= 90:
            return "31-90d"
        return ">90d"
    hist["moneyness_band"] = hist["moneyness"].apply(m_band)
    hist["dte_band"] = hist["DTE"].apply(dte_band)
    return hist


def fetch_chain(ticker: str, max_exp: int) -> Tuple[pd.DataFrame, float]:
    tk = yf.Ticker(ticker)
    spot = float("nan")
    try:
        spot = float(tk.fast_info.get("last_price") or tk.info.get("regularMarketPrice") or tk.history(period="1d").iloc[-1]["Close"])
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
            pass
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
    out["vol_oi"] = out.apply(lambda x: (x["volume"] / x["openInterest"]) if pd.notna(x.get("openInterest")) and x.get("openInterest", 0) > 0 else np.nan, axis=1)
    now_utc = pd.Timestamp.now(tz="UTC")
    out["expiration"] = pd.to_datetime(out["expiration"], utc=True)
    out["ttm_years"] = (out["expiration"] - now_utc).dt.total_seconds() / (365 * 24 * 3600)
    out.loc[out["ttm_years"] < 0, "ttm_years"] = 0.0
    out["delta"] = out.apply(lambda x: bs_delta(
        S=spot, K=x["strike"], T=max(float(x["ttm_years"]), 1e-6), r=r/100.0, q=q/100.0,
        iv=float(x["impliedVolatility"]) if pd.notna(x["impliedVolatility"]) else float("nan"),
        is_call=True if x["side"] == "call" else False
    ), axis=1)
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
    prev_slim = prev[key + ["openInterest"]].rename(columns={"openInterest": "openInterest_prev"}) if not prev.empty else pd.DataFrame(columns=key + ["openInterest_prev"])
    out = curr.merge(prev_slim, on=key, how="left")
    out["oi_change"] = out["openInterest"] - out["openInterest_prev"]
    out["oi_change_pct"] = np.where(out["openInterest_prev"].fillna(0) > 0, out["oi_change"] / out["openInterest_prev"], np.nan)
    return out


def read_external() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(INGEST_DIR, "*.csv")))
    acc = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "ticker" in df.columns and "expiration" in df.columns:
                df["expiration"] = pd.to_datetime(df["expiration"])  # normalize
                df["source_file"] = os.path.basename(f)
                acc.append(df)
        except Exception:
            pass
    return pd.concat(acc, ignore_index=True) if acc else pd.DataFrame()

# -------------------------
# Pipeline
# -------------------------
progress = st.progress(0.0, text="Fetching chains...")
rows = []
spots = {}
for i, tk in enumerate(tickers):
    try:
        chain, spot = fetch_chain(tk, max_expiries)
        spots[tk] = spot
        if not chain.empty:
            rows.append(enrich(chain, spot, r=risk_free_rate, q=dividend_yield))
    except Exception as e:
        st.warning(f"{tk}: {e}")
    progress.progress((i + 1) / max(1, len(tickers)), text=f"Fetched {i+1}/{len(tickers)}")
progress.empty()

if not rows:
    st.stop()

current = pd.concat(rows, ignore_index=True)
previous = load_latest_snapshot()
aug = merge_prev_oi(current, previous)
saved_path = save_snapshot(current)

# Z-scores vs history
hist = load_history(max_files=40)
if not hist.empty:
    def m_band_now(m):
        try:
            d = abs(float(m) - 1.0)
        except Exception:
            return "unknown"
        if d <= 0.05:
            return "ATM"
        if d <= 0.10:
            return "Near-OTM"
        if d <= 0.20:
            return "OTM"
        return "Deep-OTM"
    def dte_band_now(days):
        try:
            d = float(days)
        except Exception:
            return "unknown"
        if d <= 7:
            return "0-7d"
        if d <= 30:
            return "8-30d"
        if d <= 90:
            return "31-90d"
        return ">90d"
    if "ttm_years" in hist.columns:
        hist["DTE"] = (hist["ttm_years"].astype(float) * 365.0).round()
    if "moneyness_band" not in hist.columns:
        hist["moneyness_band"] = hist["moneyness"].apply(m_band_now)
    if "dte_band" not in hist.columns:
        hist["dte_band"] = hist.get("DTE", pd.Series(np.nan, index=hist.index)).apply(dte_band_now)
    stats = hist.groupby(["ticker","side","moneyness_band","dte_band"]).agg(
        mean_notional=("notional_usd","mean"),
        std_notional=("notional_usd","std")
    ).reset_index()
    aug["days_to_exp"] = (pd.to_datetime(aug["expiration"], utc=True) - pd.Timestamp.now(tz="UTC")).dt.days
    aug["moneyness_band"] = aug["moneyness"].apply(m_band_now)
    aug["dte_band"] = aug["days_to_exp"].apply(dte_band_now)
    aug = aug.merge(stats, on=["ticker","side","moneyness_band","dte_band"], how="left")
    aug["z_notional"] = (aug["notional_usd"] - aug["mean_notional"]) / aug["std_notional"]
    aug.loc[~np.isfinite(aug["z_notional"]), "z_notional"] = np.nan
else:
    aug["days_to_exp"] = (pd.to_datetime(aug["expiration"], utc=True) - pd.Timestamp.now(tz="UTC")).dt.days
    aug["z_notional"] = np.nan

# UOA rule
otm = rule_otm_pct / 100.0
call_otm = aug["side"].eq("call") & (aug["strike"] >= (1 + otm) * aug["underlying_price"]) 
put_otm  = aug["side"].eq("put")  & (aug["strike"] <= (1 - otm) * aug["underlying_price"]) 
mask_unusual = (
    (aug["notional_usd"] >= min_notional) &
    (aug["volume"] >= aug["openInterest"].fillna(0)) &
    (aug["days_to_exp"].between(0, rule_max_dte)) &
    (call_otm | put_otm)
)
if extra_min_vol_oi and extra_min_vol_oi > 0:
    mask_unusual &= (aug["vol_oi"].fillna(0) >= extra_min_vol_oi)

unusual = aug.loc[mask_unusual].copy()

# Priority Summary
st.subheader("Priority Summary")
if unusual.empty:
    st.info("No contracts matched the current UOA rules in the selected universe.")
else:
    with st.expander("Display filters", expanded=False):
        min_z = st.slider("Min max_zscore", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
        min_not = st.number_input("Min total_notional (league)", min_value=0, max_value=100_000_000_000, value=0, step=1_000_000)
    league = unusual.groupby("ticker", as_index=False).agg(
        unusual_trades=("contractSymbol","count"),
        total_notional=("notional_usd","sum"),
        max_zscore=("z_notional","max")
    )
    league = league[(league["max_zscore"].fillna(0) >= min_z) & (league["total_notional"] >= min_not)]
    league = league.sort_values(["total_notional","max_zscore","unusual_trades"], ascending=[False, False, False])
    league_display = league.copy()
    league_display["total_notional"] = league_display["total_notional"].apply(lambda x: f"${x:,.0f}")
    league_display["max_zscore"] = league_display["max_zscore"].apply(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    st.dataframe(league_display, use_container_width=True, hide_index=True)

    st.subheader("Top contracts per ticker (head 5)")
    top = unusual.sort_values(["ticker","notional_usd"], ascending=[True, False]).groupby("ticker").head(5).copy()
    top["expiration"] = pd.to_datetime(top["expiration"]).dt.date.astype(str)
    def make_flag(row):
        flags = []
        z = pd.to_numeric(row.get("z_notional"), errors="coerce")
        dte = pd.to_numeric(row.get("days_to_exp"), errors="coerce")
        voi = pd.to_numeric(row.get("vol_oi"), errors="coerce")
        if pd.notna(z) and z >= 3: flags.append("Z>=3")
        if pd.notna(dte) and dte <= 7: flags.append("<=7d")
        if pd.notna(voi) and voi >= 2: flags.append("Vol/OI>=2")
        return " | ".join(flags)
    top["flags"] = top.apply(make_flag, axis=1)
    top_display = top[[
        "ticker","flags","side","expiration","strike","underlying_price","lastPrice","volume","openInterest",
        "vol_oi","impliedVolatility","delta","notional_usd","days_to_exp","z_notional"
    ]].copy()
    top_display["underlying_price"] = top_display["underlying_price"].apply(lambda v: "" if pd.isna(v) else f"${v:,.2f}")
    top_display["lastPrice"] = top_display["lastPrice"].apply(lambda v: "" if pd.isna(v) else f"${v:,.2f}")
    top_display["notional_usd"] = top_display["notional_usd"].apply(lambda v: "" if pd.isna(v) else f"${v:,.0f}")
    top_display["impliedVolatility"] = top_display["impliedVolatility"].apply(lambda v: "" if pd.isna(v) else f"{v*100:.1f}%")
    top_display["delta"] = top_display["delta"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    top_display["vol_oi"] = top_display["vol_oi"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    top_display["z_notional"] = top_display["z_notional"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    st.dataframe(top_display, use_container_width=True, hide_index=True)

# Flow Summary
st.subheader("Flow Summary")
st.caption("Contracts that satisfy: Volume >= OI, notional >= floor, short-dated, and OTM by threshold.")
if unusual.empty:
    st.stop()
view = unusual[[
    "ticker","side","expiration","days_to_exp","strike","underlying_price","lastPrice",
    "volume","openInterest","vol_oi","oi_change","oi_change_pct","impliedVolatility","delta",
    "notional_usd","moneyness"
]].copy()
view["expiration"] = pd.to_datetime(view["expiration"]).dt.date.astype(str)
view["oi_change_pct"] = view["oi_change_pct"].apply(lambda x: "" if pd.isna(x) else f"{x*100:.1f}%")
view["impliedVolatility"] = view["impliedVolatility"].apply(lambda x: "" if pd.isna(x) else f"{x*100:.1f}%")
view["delta"] = view["delta"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}")
view["vol_oi"] = view["vol_oi"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}")
view["notional_usd"] = view["notional_usd"].apply(lambda v: "" if pd.isna(v) else f"${v:,.0f}")
view["underlying_price"] = view["underlying_price"].apply(lambda v: "" if pd.isna(v) else f"${v:,.2f}")
view["lastPrice"] = view["lastPrice"].apply(lambda v: "" if pd.isna(v) else f"${v:,.2f}")
# comma-format counts in Flow Summary
view["volume"] = pd.to_numeric(view["volume"], errors="coerce").astype("Int64").map(lambda x: f"{x:,}" if pd.notna(x) else "")
view["openInterest"] = pd.to_numeric(view["openInterest"], errors="coerce").astype("Int64").map(lambda x: f"{x:,}" if pd.notna(x) else "")

view = view.sort_values(["ticker","notional_usd"], ascending=[True, False])
st.dataframe(view, use_container_width=True, hide_index=True)

with st.expander("External ingest (CSV)"):
    st.write("Drop CSV files into ./ingest with columns: datetime, ticker, side, strike, expiration, size, price, notional, exchange.")
    ext = read_external()
    if not ext.empty:
        st.dataframe(ext, use_container_width=True, hide_index=True)
    else:
        st.info("No external CSVs detected.")

st.caption("Data: Yahoo Finance chains via yfinance. UOA rules applied. Snapshot saved. Run daily to build history.")
