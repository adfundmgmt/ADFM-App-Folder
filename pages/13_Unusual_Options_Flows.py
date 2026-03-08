# streamlit_app.py
# Unusual Options Flow Tracker
# Full S&P 500 universe, premium-first signal engine, cached concurrent scan

import os
import io
import glob
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

APP_TITLE = "Unusual Options Flow Tracker"
DATA_DIR = "data"
INGEST_DIR = "ingest"
SCAN_CACHE_BASENAME = "last_good_scan"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INGEST_DIR, exist_ok=True)

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# -----------------------------------------------------------------------------
# Generic file helpers
# -----------------------------------------------------------------------------
def _safe_write_table(df: pd.DataFrame, path_no_ext: str) -> str:
    parquet_path = f"{path_no_ext}.parquet"
    csv_path = f"{path_no_ext}.csv"
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        df.to_csv(csv_path, index=False)
        return csv_path


def _safe_read_table(path: str) -> pd.DataFrame:
    try:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def list_snapshot_files() -> List[str]:
    parquet_files = glob.glob(os.path.join(DATA_DIR, "options_snapshot_*.parquet"))
    csv_files = glob.glob(os.path.join(DATA_DIR, "options_snapshot_*.csv"))
    return sorted(parquet_files + csv_files)


def save_snapshot(df: pd.DataFrame) -> str:
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(DATA_DIR, f"options_snapshot_{tag}")
    return _safe_write_table(df, base)


def load_latest_snapshot() -> pd.DataFrame:
    files = list_snapshot_files()
    if not files:
        return pd.DataFrame()
    return _safe_read_table(files[-1])


def load_history(max_files: int = 80) -> pd.DataFrame:
    files = list_snapshot_files()
    if not files:
        return pd.DataFrame()

    files = files[-max_files:]
    out = []
    for f in files:
        df = _safe_read_table(f)
        if df.empty:
            continue
        tag = os.path.splitext(os.path.basename(f))[0].replace("options_snapshot_", "")
        df["snapshot_date"] = tag
        out.append(df)

    if not out:
        return pd.DataFrame()

    hist = pd.concat(out, ignore_index=True)

    needed = [
        "ticker", "contractSymbol", "side", "expiration", "strike",
        "underlying_price", "volume", "openInterest", "vol_oi",
        "impliedVolatility", "delta", "moneyness", "notional_usd",
        "premium_usd", "ttm_years"
    ]
    for c in needed:
        if c not in hist.columns:
            hist[c] = np.nan

    hist["expiration"] = pd.to_datetime(hist["expiration"], errors="coerce")
    hist["DTE"] = np.floor(pd.to_numeric(hist["ttm_years"], errors="coerce") * 365.0)

    hist["moneyness_band"] = hist["moneyness"].apply(moneyness_band_from_ratio)
    hist["dte_band"] = hist["DTE"].apply(dte_band_from_days)

    return hist


def save_last_good_scan(df: pd.DataFrame) -> str:
    base = os.path.join(DATA_DIR, SCAN_CACHE_BASENAME)
    return _safe_write_table(df, base)


def load_last_good_scan() -> pd.DataFrame:
    parquet_path = os.path.join(DATA_DIR, f"{SCAN_CACHE_BASENAME}.parquet")
    csv_path = os.path.join(DATA_DIR, f"{SCAN_CACHE_BASENAME}.csv")
    if os.path.exists(parquet_path):
        return _safe_read_table(parquet_path)
    if os.path.exists(csv_path):
        return _safe_read_table(csv_path)
    return pd.DataFrame()


# -----------------------------------------------------------------------------
# Universe helpers
# -----------------------------------------------------------------------------
def _normalize_symbols(symbols: List[str]) -> List[str]:
    out = []
    seen = set()
    for s in symbols:
        if not isinstance(s, str):
            continue
        x = s.replace(".", "-").upper().strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def get_sp500_symbols() -> List[str]:
    try:
        import requests
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        r.raise_for_status()
        tables = pd.read_html(r.text)
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if "symbol" in cols:
                colname = t.columns[cols.index("symbol")]
                return _normalize_symbols(t[colname].astype(str).tolist())
    except Exception:
        pass

    try:
        import requests
        url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        for c in df.columns:
            if str(c).strip().lower() in ("symbol", "ticker"):
                return _normalize_symbols(df[c].astype(str).tolist())
    except Exception:
        pass

    try:
        local = os.path.join(INGEST_DIR, "sp500_symbols.csv")
        if os.path.exists(local):
            df = pd.read_csv(local)
            for c in df.columns:
                if str(c).strip().lower() in ("symbol", "ticker"):
                    return _normalize_symbols(df[c].astype(str).tolist())
    except Exception:
        pass

    return [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","LLY","AVGO","JPM","V","WMT",
        "XOM","UNH","JNJ","PG","MA","COST","HD","ABBV","ORCL","BAC","MRK","PEP","KO","NFLX"
    ]


# -----------------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta_scalar(S: float, K: float, T: float, r: float, q: float, iv: float, is_call: bool) -> float:
    try:
        if any(pd.isna(v) for v in [S, K, T, r, q, iv]):
            return np.nan
        if T <= 0 or iv <= 0 or S <= 0 or K <= 0:
            return np.nan
        d1 = (math.log(S / K) + (r - q + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
        if is_call:
            return math.exp(-q * T) * norm_cdf(d1)
        return math.exp(-q * T) * (norm_cdf(d1) - 1.0)
    except Exception:
        return np.nan


def compute_vol_oi(volume: pd.Series, open_interest: pd.Series) -> pd.Series:
    vol = pd.to_numeric(volume, errors="coerce")
    oi = pd.to_numeric(open_interest, errors="coerce")
    out = vol / oi
    out = out.where(oi > 0, np.nan)
    return out


def moneyness_band_from_ratio(m: float) -> str:
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


def dte_band_from_days(d: float) -> str:
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


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# -----------------------------------------------------------------------------
# Option chain fetch and enrich
# -----------------------------------------------------------------------------
def get_spot_price(tk: yf.Ticker) -> float:
    try:
        fi = getattr(tk, "fast_info", {}) or {}
        spot = fi.get("last_price")
        if spot is not None and pd.notna(spot):
            return float(spot)
    except Exception:
        pass

    try:
        info = getattr(tk, "info", {}) or {}
        spot = info.get("regularMarketPrice")
        if spot is not None and pd.notna(spot):
            return float(spot)
    except Exception:
        pass

    try:
        hist = tk.history(period="5d", auto_adjust=False)
        if not hist.empty:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass

    return np.nan


def fetch_chain_raw(ticker: str, max_expiries: int) -> Tuple[pd.DataFrame, float, str]:
    try:
        tk = yf.Ticker(ticker)
        spot = get_spot_price(tk)

        try:
            expirations = list(tk.options or [])
        except Exception:
            expirations = []

        if not expirations:
            return pd.DataFrame(), spot, "no_expirations"

        rows = []
        for exp in expirations[:max_expiries]:
            try:
                oc = tk.option_chain(exp)
                for side, df in (("call", oc.calls), ("put", oc.puts)):
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
            return pd.DataFrame(), spot, "empty_chain"

        return pd.concat(rows, ignore_index=True), spot, ""
    except Exception as e:
        return pd.DataFrame(), np.nan, str(e)


def enrich_chain(df: pd.DataFrame, spot: float, r: float, q: float) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    numeric_cols = ["volume", "openInterest", "impliedVolatility", "strike", "lastPrice", "bid", "ask"]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = np.nan

    out["underlying_price"] = pd.to_numeric(spot, errors="coerce")
    out["expiration"] = pd.to_datetime(out["expiration"], errors="coerce")
    now_utc = pd.Timestamp.now(tz="UTC")
    exp_utc = pd.to_datetime(out["expiration"], utc=True, errors="coerce")
    out["ttm_years"] = (exp_utc - now_utc).dt.total_seconds() / (365.0 * 24.0 * 3600.0)
    out["ttm_years"] = out["ttm_years"].clip(lower=0.0)

    out["days_to_exp"] = (
        pd.to_datetime(out["expiration"]).dt.normalize() - pd.Timestamp.utcnow().normalize()
    ).dt.days

    out["notional_usd"] = out["volume"].fillna(0.0) * out["underlying_price"].fillna(0.0) * 100.0
    out["premium_usd"] = out["volume"].fillna(0.0) * out["lastPrice"].fillna(0.0) * 100.0
    out["vol_oi"] = compute_vol_oi(out["volume"], out["openInterest"])
    out["moneyness"] = safe_div(out["underlying_price"], out["strike"])

    mids = (out["bid"].fillna(0.0) + out["ask"].fillna(0.0)) / 2.0
    spreads = out["ask"] - out["bid"]
    out["mid"] = mids.where((out["bid"].notna()) & (out["ask"].notna()), np.nan)
    out["spread"] = spreads.where((out["bid"].notna()) & (out["ask"].notna()), np.nan)
    out["spread_pct_mid"] = safe_div(out["spread"], out["mid"])
    out["last_vs_mid"] = out["lastPrice"] - out["mid"]

    def execution_bias(last_price, bid, ask):
        if pd.isna(last_price) or pd.isna(bid) or pd.isna(ask) or ask <= bid:
            return np.nan
        return (last_price - bid) / (ask - bid)

    out["exec_bias"] = [
        execution_bias(lp, b, a)
        for lp, b, a in zip(out["lastPrice"], out["bid"], out["ask"])
    ]

    out["delta"] = [
        bs_delta_scalar(
            S=float(s) if pd.notna(s) else np.nan,
            K=float(k) if pd.notna(k) else np.nan,
            T=max(float(t), 1e-6) if pd.notna(t) else np.nan,
            r=r / 100.0,
            q=q / 100.0,
            iv=float(iv) if pd.notna(iv) else np.nan,
            is_call=(side == "call"),
        )
        for s, k, t, iv, side in zip(
            out["underlying_price"], out["strike"], out["ttm_years"], out["impliedVolatility"], out["side"]
        )
    ]

    out["delta_abs"] = out["delta"].abs()
    out["delta_notional_usd"] = out["notional_usd"] * out["delta_abs"].fillna(0.0)

    keep = [
        "ticker", "contractSymbol", "side", "expiration", "days_to_exp",
        "strike", "lastPrice", "bid", "ask", "mid", "spread", "spread_pct_mid",
        "last_vs_mid", "exec_bias", "volume", "openInterest", "impliedVolatility",
        "delta", "delta_abs", "moneyness", "underlying_price",
        "notional_usd", "delta_notional_usd", "premium_usd", "vol_oi", "ttm_years"
    ]
    for c in keep:
        if c not in out.columns:
            out[c] = np.nan
    return out[keep]


@st.cache_data(ttl=15 * 60, show_spinner=False)
def load_market_scan(
    tickers: Tuple[str, ...],
    max_expiries: int,
    max_workers: int,
    risk_free_rate: float,
    dividend_yield: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    collected = []
    failures = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(fetch_chain_raw, ticker, max_expiries): ticker
            for ticker in tickers
        }
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                chain_df, spot, err = fut.result()
                if err:
                    failures.append({"ticker": ticker, "error": err})
                    continue
                if chain_df.empty:
                    failures.append({"ticker": ticker, "error": "empty"})
                    continue
                enriched = enrich_chain(chain_df, spot, risk_free_rate, dividend_yield)
                if enriched.empty:
                    failures.append({"ticker": ticker, "error": "enrich_empty"})
                    continue
                collected.append(enriched)
            except Exception as e:
                failures.append({"ticker": ticker, "error": str(e)})

    fail_df = pd.DataFrame(failures)

    if collected:
        full = pd.concat(collected, ignore_index=True)
        save_last_good_scan(full)
        return full, fail_df

    fallback = load_last_good_scan()
    return fallback, fail_df


# -----------------------------------------------------------------------------
# Historical features
# -----------------------------------------------------------------------------
def merge_prev_oi(curr: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    if curr.empty:
        return curr.copy()

    key = ["ticker", "contractSymbol"]
    prev_cols = key + ["openInterest"]
    prev_s = pd.DataFrame(columns=key + ["openInterest_prev"])
    if not prev.empty and all(c in prev.columns for c in prev_cols):
        prev_s = prev[prev_cols].copy().rename(columns={"openInterest": "openInterest_prev"})

    out = curr.merge(prev_s, on=key, how="left")
    out["oi_change"] = pd.to_numeric(out["openInterest"], errors="coerce") - pd.to_numeric(out["openInterest_prev"], errors="coerce")
    out["oi_change_pct"] = np.where(
        pd.to_numeric(out["openInterest_prev"], errors="coerce").fillna(0) > 0,
        out["oi_change"] / pd.to_numeric(out["openInterest_prev"], errors="coerce"),
        np.nan
    )
    return out


def attach_history_stats(curr: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    out = curr.copy()
    if out.empty:
        return out

    out["moneyness_band"] = out["moneyness"].apply(moneyness_band_from_ratio)
    out["dte_band"] = out["days_to_exp"].apply(dte_band_from_days)

    if hist.empty:
        out["mean_notional"] = np.nan
        out["std_notional"] = np.nan
        out["mean_premium"] = np.nan
        out["std_premium"] = np.nan
        out["z_notional"] = np.nan
        out["z_premium"] = np.nan
        return out

    grp = ["ticker", "side", "moneyness_band", "dte_band"]
    stats = (
        hist.groupby(grp, dropna=False)
        .agg(
            mean_notional=("notional_usd", "mean"),
            std_notional=("notional_usd", "std"),
            mean_premium=("premium_usd", "mean"),
            std_premium=("premium_usd", "std"),
        )
        .reset_index()
    )

    out = out.merge(stats, on=grp, how="left")
    out["z_notional"] = (out["notional_usd"] - out["mean_notional"]) / out["std_notional"]
    out["z_premium"] = (out["premium_usd"] - out["mean_premium"]) / out["std_premium"]
    out.loc[~np.isfinite(out["z_notional"]), "z_notional"] = np.nan
    out.loc[~np.isfinite(out["z_premium"]), "z_premium"] = np.nan
    return out


# -----------------------------------------------------------------------------
# Signal engine
# -----------------------------------------------------------------------------
def add_side_aware_otm_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["otm_pct"] = np.nan

    call_mask = out["side"].eq("call") & out["underlying_price"].gt(0)
    put_mask = out["side"].eq("put") & out["underlying_price"].gt(0)

    out.loc[call_mask, "otm_pct"] = (out.loc[call_mask, "strike"] / out.loc[call_mask, "underlying_price"]) - 1.0
    out.loc[put_mask, "otm_pct"] = 1.0 - (out.loc[put_mask, "strike"] / out.loc[put_mask, "underlying_price"])

    out["is_otm"] = out["otm_pct"].fillna(-999.0) >= 0
    return out


def apply_uoa_screen(
    df: pd.DataFrame,
    min_notional: float,
    min_premium: float,
    rule_otm_pct: float,
    rule_max_dte: int,
    extra_min_vol_oi: float,
    require_exec_bias: bool,
) -> pd.DataFrame:
    out = add_side_aware_otm_fields(df)

    otm_req = rule_otm_pct / 100.0
    mask = (
        (out["notional_usd"] >= min_notional) &
        (out["premium_usd"] >= min_premium) &
        (pd.to_numeric(out["volume"], errors="coerce").fillna(0) >= pd.to_numeric(out["openInterest"], errors="coerce").fillna(0)) &
        (out["days_to_exp"].between(0, rule_max_dte)) &
        (out["otm_pct"].fillna(-999.0) >= otm_req)
    )

    if extra_min_vol_oi > 0:
        mask &= out["vol_oi"].fillna(0.0) >= extra_min_vol_oi

    if require_exec_bias:
        mask &= out["exec_bias"].fillna(0.5) >= 0.60

    return out.loc[mask].copy()


def classify_tier(row: pd.Series) -> str:
    prem = float(row.get("premium_usd", np.nan))
    voi = float(row.get("vol_oi", np.nan))
    dte = float(row.get("days_to_exp", np.nan))
    zprem = float(row.get("z_premium", np.nan))
    exec_bias = float(row.get("exec_bias", np.nan))

    if all(pd.notna(v) for v in [prem, voi, dte]):
        if prem >= 10_000_000 and voi >= 3 and dte <= 21:
            if pd.isna(exec_bias) or exec_bias >= 0.60:
                return "Tier 1"
        if prem >= 2_000_000 and voi >= 2 and dte <= 45:
            if pd.isna(exec_bias) or exec_bias >= 0.55:
                return "Tier 2"

    if pd.notna(zprem) and zprem >= 3 and prem >= 1_000_000 and voi >= 1:
        return "Tier 2"

    return "Contextual Flow"


def build_bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    x = df.copy()
    x["strike_bucket"] = (pd.to_numeric(x["strike"], errors="coerce") / 5.0).round() * 5.0

    grp = ["ticker", "side", "expiration", "strike_bucket", "tier"]
    bucket = (
        x.groupby(grp, dropna=False)
        .agg(
            lines=("contractSymbol", "count"),
            total_premium=("premium_usd", "sum"),
            total_notional=("notional_usd", "sum"),
            total_delta_notional=("delta_notional_usd", "sum"),
            max_vol_oi=("vol_oi", "max"),
            avg_exec_bias=("exec_bias", "mean"),
            avg_otm_pct=("otm_pct", "mean"),
        )
        .reset_index()
        .sort_values(["total_premium", "total_notional"], ascending=[False, False])
    )
    return bucket


def build_priority_summary(signal: pd.DataFrame, ranking_mode: str) -> pd.DataFrame:
    if signal.empty:
        return pd.DataFrame()

    league = (
        signal.groupby("ticker", as_index=False)
        .agg(
            tier1_trades=("tier", lambda s: int((s == "Tier 1").sum())),
            tier2_trades=("tier", lambda s: int((s == "Tier 2").sum())),
            total_premium=("premium_usd", "sum"),
            total_notional=("notional_usd", "sum"),
            total_delta_notional=("delta_notional_usd", "sum"),
            max_premium=("premium_usd", "max"),
            best_z_premium=("z_premium", "max"),
            avg_exec_bias=("exec_bias", "mean"),
        )
    )

    if ranking_mode == "Premium":
        league = league.sort_values(
            ["total_premium", "tier1_trades", "max_premium"],
            ascending=[False, False, False]
        )
    elif ranking_mode == "Notional":
        league = league.sort_values(
            ["total_notional", "total_premium", "tier1_trades"],
            ascending=[False, False, False]
        )
    elif ranking_mode == "Delta-adjusted":
        league = league.sort_values(
            ["total_delta_notional", "total_premium", "tier1_trades"],
            ascending=[False, False, False]
        )
    else:
        league = league.sort_values(
            ["best_z_premium", "total_premium", "tier1_trades"],
            ascending=[False, False, False]
        )

    return league.reset_index(drop=True)


# -----------------------------------------------------------------------------
# External ingest
# -----------------------------------------------------------------------------
def read_external() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(INGEST_DIR, "*.csv")))
    acc = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "ticker" in df.columns and "expiration" in df.columns:
                df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")
                df["source_file"] = os.path.basename(f)
                acc.append(df)
        except Exception:
            continue
    return pd.concat(acc, ignore_index=True) if acc else pd.DataFrame()


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------
def fmt_money(x):
    return "" if pd.isna(x) else f"${x:,.0f}"


def fmt_money_2(x):
    return "" if pd.isna(x) else f"${x:,.2f}"


def fmt_pct(x, mult=100.0, decimals=1):
    return "" if pd.isna(x) else f"{x * mult:.{decimals}f}%"


def fmt_num(x, decimals=2):
    return "" if pd.isna(x) else f"{x:.{decimals}f}"


def fmt_int(x):
    return "" if pd.isna(x) else f"{int(x):,}"


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Objective: detect institutional-grade unusual options activity using the full S&P 500.

        Philosophy: premium first, intensity second. Focus on flows large enough to matter,
        filtered by moneyness, expiration window, volume versus open interest, and basic execution quality.

        Improvements in this build
        • Concurrent option-chain fetch across the full universe
        • Cached raw scan so threshold changes do not hit the network
        • Timestamped snapshots for richer history
        • Last-good scan fallback when live fetch degrades
        • History-aware z-scores by ticker, side, moneyness band, and DTE band
        • Bucket summaries that cluster related contracts
        """
    )

    st.markdown("### Universe")
    st.caption("Using full S&P 500 universe with public-source fallback to local CSV.")
    uploaded = st.file_uploader("Upload sp500_symbols.csv (optional)", type=["csv"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            df_up.to_csv(os.path.join(INGEST_DIR, "sp500_symbols.csv"), index=False)
            st.success("Saved to ./ingest/sp500_symbols.csv. Reload to use it.")
        except Exception as e:
            st.warning(f"Upload failed: {e}")

    tickers = get_sp500_symbols()
    st.caption(f"Loaded {len(tickers)} symbols.")

    st.markdown("### Data")
    max_expiries = st.number_input("Max expirations per ticker", min_value=1, max_value=12, value=4, step=1)
    max_workers = st.number_input("Concurrent workers", min_value=2, max_value=24, value=10, step=1)

    st.markdown("### UOA Thresholds")
    min_notional = st.number_input("Min notional per line, USD", min_value=0, max_value=250_000_000, value=500_000, step=100_000)
    min_premium = st.number_input("Min premium per line, USD", min_value=0, max_value=250_000_000, value=0, step=100_000)
    rule_otm_pct = st.number_input("OTM threshold %", min_value=0.0, max_value=200.0, value=10.0, step=1.0)
    rule_max_dte = st.number_input("Max DTE (days)", min_value=1, max_value=365, value=30, step=1)
    extra_min_vol_oi = st.number_input("Extra floor for Vol/OI", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
    require_exec_bias = st.checkbox("Require last trade to lean toward ask", value=False)

    st.markdown("### Greeks Setup")
    risk_free_rate = st.number_input("Risk-free rate %", min_value=-1.0, max_value=15.0, value=4.25, step=0.05)
    dividend_yield = st.number_input("Dividend yield %", min_value=0.0, max_value=20.0, value=0.0, step=0.1)

    st.markdown("### Ranking")
    ranking_mode = st.selectbox("Priority ranking", ["Premium", "Notional", "Delta-adjusted", "Z-score"])

    st.markdown("### History")
    history_files = st.number_input("History files for z-scores", min_value=10, max_value=200, value=80, step=10)


# -----------------------------------------------------------------------------
# Load raw market scan
# -----------------------------------------------------------------------------
with st.status("Scanning option chains...", expanded=False) as status:
    raw_scan, fail_df = load_market_scan(
        tuple(tickers),
        int(max_expiries),
        int(max_workers),
        float(risk_free_rate),
        float(dividend_yield),
    )
    if raw_scan.empty:
        status.update(label="Live scan unavailable. No last-good scan found.", state="error")
    else:
        status.update(label="Scan completed.", state="complete")

if raw_scan.empty:
    st.error("No option-chain data available. Live scan failed and no last-good dataset was found.")
    st.stop()

# Persist latest timestamped snapshot after a successful or fallback scan
saved_path = save_snapshot(raw_scan)

# -----------------------------------------------------------------------------
# Build signal table locally only
# -----------------------------------------------------------------------------
previous = load_latest_snapshot()
hist = load_history(max_files=int(history_files))

aug = merge_prev_oi(raw_scan, previous)
aug = attach_history_stats(aug, hist)

unusual = apply_uoa_screen(
    aug,
    min_notional=float(min_notional),
    min_premium=float(min_premium),
    rule_otm_pct=float(rule_otm_pct),
    rule_max_dte=int(rule_max_dte),
    extra_min_vol_oi=float(extra_min_vol_oi),
    require_exec_bias=bool(require_exec_bias),
)

if unusual.empty:
    c1, c2, c3 = st.columns(3)
    c1.metric("Universe", f"{len(tickers):,}")
    c2.metric("Contracts scanned", f"{len(raw_scan):,}")
    c3.metric("Fetch failures", f"{len(fail_df):,}")
    st.info("No contracts matched the current UOA screen.")
    st.caption(f"Latest snapshot saved: {saved_path}")
    st.stop()

unusual["tier"] = unusual.apply(classify_tier, axis=1)
signal = unusual[unusual["tier"].isin(["Tier 1", "Tier 2"])].copy()

# -----------------------------------------------------------------------------
# Top status row
# -----------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Universe", f"{len(tickers):,}")
c2.metric("Contracts scanned", f"{len(raw_scan):,}")
c3.metric("UOA lines", f"{len(unusual):,}")
c4.metric("Tier 1 / 2 lines", f"{len(signal):,}")
c5.metric("Fetch failures", f"{len(fail_df):,}")

if not fail_df.empty:
    with st.expander("Fetch failures and empty chains"):
        fail_show = fail_df.copy()
        st.dataframe(fail_show.sort_values(["error", "ticker"]), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Priority summary
# -----------------------------------------------------------------------------
st.subheader("Priority Summary")

if signal.empty:
    st.info("No Tier 1 or Tier 2 contracts matched today in the selected universe.")
else:
    league = build_priority_summary(signal, ranking_mode=ranking_mode)

    league_display = league.copy()
    league_display["total_premium"] = league_display["total_premium"].map(fmt_money)
    league_display["total_notional"] = league_display["total_notional"].map(fmt_money)
    league_display["total_delta_notional"] = league_display["total_delta_notional"].map(fmt_money)
    league_display["max_premium"] = league_display["max_premium"].map(fmt_money)
    league_display["best_z_premium"] = league_display["best_z_premium"].map(lambda x: fmt_num(x, 2))
    league_display["avg_exec_bias"] = league_display["avg_exec_bias"].map(lambda x: fmt_num(x, 2))
    league_display["tier1_trades"] = league_display["tier1_trades"].map(fmt_int)
    league_display["tier2_trades"] = league_display["tier2_trades"].map(fmt_int)

    st.dataframe(league_display, use_container_width=True, hide_index=True)

    st.subheader("Top contracts per ticker")
    top_raw = (
        signal.sort_values(["ticker", "premium_usd"], ascending=[True, False])
        .groupby("ticker", as_index=False, sort=False)
        .head(3)
        .copy()
    )

    top_display = top_raw[[
        "ticker", "tier", "side", "expiration", "days_to_exp", "strike",
        "underlying_price", "lastPrice", "volume", "openInterest", "vol_oi",
        "impliedVolatility", "delta", "premium_usd", "notional_usd",
        "delta_notional_usd", "z_premium", "exec_bias"
    ]].copy()

    top_display["expiration"] = pd.to_datetime(top_display["expiration"]).dt.date.astype(str)
    top_display["strike"] = top_display["strike"].map(fmt_money_2)
    top_display["underlying_price"] = top_display["underlying_price"].map(fmt_money_2)
    top_display["lastPrice"] = top_display["lastPrice"].map(fmt_money_2)
    top_display["volume"] = top_raw["volume"].map(fmt_int)
    top_display["openInterest"] = top_raw["openInterest"].map(fmt_int)
    top_display["vol_oi"] = top_raw["vol_oi"].map(lambda x: fmt_num(x, 2))
    top_display["impliedVolatility"] = top_raw["impliedVolatility"].map(lambda x: fmt_pct(x, mult=100.0, decimals=1))
    top_display["delta"] = top_raw["delta"].map(lambda x: fmt_num(x, 2))
    top_display["premium_usd"] = top_raw["premium_usd"].map(fmt_money)
    top_display["notional_usd"] = top_raw["notional_usd"].map(fmt_money)
    top_display["delta_notional_usd"] = top_raw["delta_notional_usd"].map(fmt_money)
    top_display["z_premium"] = top_raw["z_premium"].map(lambda x: fmt_num(x, 2))
    top_display["exec_bias"] = top_raw["exec_bias"].map(lambda x: fmt_num(x, 2))

    st.dataframe(top_display, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Bucket summary
# -----------------------------------------------------------------------------
st.subheader("Clustered Flow Buckets")
bucket = build_bucket_summary(signal if not signal.empty else unusual)

if bucket.empty:
    st.info("No grouped buckets available.")
else:
    bucket_display = bucket.head(200).copy()
    bucket_display["expiration"] = pd.to_datetime(bucket_display["expiration"]).dt.date.astype(str)
    bucket_display["strike_bucket"] = bucket_display["strike_bucket"].map(fmt_money_2)
    bucket_display["lines"] = bucket_display["lines"].map(fmt_int)
    bucket_display["total_premium"] = bucket_display["total_premium"].map(fmt_money)
    bucket_display["total_notional"] = bucket_display["total_notional"].map(fmt_money)
    bucket_display["total_delta_notional"] = bucket_display["total_delta_notional"].map(fmt_money)
    bucket_display["max_vol_oi"] = bucket_display["max_vol_oi"].map(lambda x: fmt_num(x, 2))
    bucket_display["avg_exec_bias"] = bucket_display["avg_exec_bias"].map(lambda x: fmt_num(x, 2))
    bucket_display["avg_otm_pct"] = bucket_display["avg_otm_pct"].map(lambda x: fmt_pct(x, mult=100.0, decimals=1))
    st.dataframe(bucket_display, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Signal table
# -----------------------------------------------------------------------------
st.subheader("Flow Summary")

signal_base = signal if not signal.empty else unusual
view_raw = signal_base[[
    "ticker", "tier", "side", "expiration", "days_to_exp", "strike",
    "underlying_price", "lastPrice", "volume", "openInterest", "vol_oi",
    "oi_change", "oi_change_pct", "impliedVolatility", "delta", "delta_abs",
    "premium_usd", "notional_usd", "delta_notional_usd", "moneyness",
    "otm_pct", "z_premium", "z_notional", "exec_bias"
]].copy()

sort_col = {
    "Premium": "premium_usd",
    "Notional": "notional_usd",
    "Delta-adjusted": "delta_notional_usd",
    "Z-score": "z_premium",
}[ranking_mode]

view_raw = view_raw.sort_values(sort_col, ascending=False)

view = view_raw.copy()
view["expiration"] = pd.to_datetime(view["expiration"]).dt.date.astype(str)
view["strike"] = view["strike"].map(fmt_money_2)
view["underlying_price"] = view["underlying_price"].map(fmt_money_2)
view["lastPrice"] = view["lastPrice"].map(fmt_money_2)
view["volume"] = view_raw["volume"].map(fmt_int)
view["openInterest"] = view_raw["openInterest"].map(fmt_int)
view["vol_oi"] = view_raw["vol_oi"].map(lambda x: fmt_num(x, 2))
view["oi_change"] = view_raw["oi_change"].map(lambda x: fmt_num(x, 0))
view["oi_change_pct"] = view_raw["oi_change_pct"].map(lambda x: fmt_pct(x, mult=100.0, decimals=1))
view["impliedVolatility"] = view_raw["impliedVolatility"].map(lambda x: fmt_pct(x, mult=100.0, decimals=1))
view["delta"] = view_raw["delta"].map(lambda x: fmt_num(x, 2))
view["delta_abs"] = view_raw["delta_abs"].map(lambda x: fmt_num(x, 2))
view["premium_usd"] = view_raw["premium_usd"].map(fmt_money)
view["notional_usd"] = view_raw["notional_usd"].map(fmt_money)
view["delta_notional_usd"] = view_raw["delta_notional_usd"].map(fmt_money)
view["moneyness"] = view_raw["moneyness"].map(lambda x: fmt_num(x, 3))
view["otm_pct"] = view_raw["otm_pct"].map(lambda x: fmt_pct(x, mult=100.0, decimals=1))
view["z_premium"] = view_raw["z_premium"].map(lambda x: fmt_num(x, 2))
view["z_notional"] = view_raw["z_notional"].map(lambda x: fmt_num(x, 2))
view["exec_bias"] = view_raw["exec_bias"].map(lambda x: fmt_num(x, 2))

st.dataframe(view, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Ticker drilldown
# -----------------------------------------------------------------------------
st.subheader("Ticker Drilldown")
candidate_names = league["ticker"].tolist() if not signal.empty else sorted(signal_base["ticker"].dropna().unique().tolist())
if candidate_names:
    selected_ticker = st.selectbox("Select ticker", candidate_names)
    ticker_df = signal_base[signal_base["ticker"] == selected_ticker].copy()

    if not ticker_df.empty:
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Lines", f"{len(ticker_df):,}")
        t2.metric("Premium", fmt_money(ticker_df["premium_usd"].sum()))
        t3.metric("Notional", fmt_money(ticker_df["notional_usd"].sum()))
        t4.metric("Delta-adjusted", fmt_money(ticker_df["delta_notional_usd"].sum()))

        ticker_view_raw = ticker_df.sort_values(["premium_usd", "notional_usd"], ascending=False).copy()
        ticker_view = ticker_view_raw[[
            "tier", "side", "expiration", "days_to_exp", "contractSymbol", "strike",
            "underlying_price", "lastPrice", "volume", "openInterest", "vol_oi",
            "impliedVolatility", "delta", "premium_usd", "notional_usd",
            "delta_notional_usd", "otm_pct", "z_premium", "exec_bias"
        ]].copy()

        ticker_view["expiration"] = pd.to_datetime(ticker_view["expiration"]).dt.date.astype(str)
        ticker_view["strike"] = ticker_view_raw["strike"].map(fmt_money_2)
        ticker_view["underlying_price"] = ticker_view_raw["underlying_price"].map(fmt_money_2)
        ticker_view["lastPrice"] = ticker_view_raw["lastPrice"].map(fmt_money_2)
        ticker_view["volume"] = ticker_view_raw["volume"].map(fmt_int)
        ticker_view["openInterest"] = ticker_view_raw["openInterest"].map(fmt_int)
        ticker_view["vol_oi"] = ticker_view_raw["vol_oi"].map(lambda x: fmt_num(x, 2))
        ticker_view["impliedVolatility"] = ticker_view_raw["impliedVolatility"].map(lambda x: fmt_pct(x, mult=100.0, decimals=1))
        ticker_view["delta"] = ticker_view_raw["delta"].map(lambda x: fmt_num(x, 2))
        ticker_view["premium_usd"] = ticker_view_raw["premium_usd"].map(fmt_money)
        ticker_view["notional_usd"] = ticker_view_raw["notional_usd"].map(fmt_money)
        ticker_view["delta_notional_usd"] = ticker_view_raw["delta_notional_usd"].map(fmt_money)
        ticker_view["otm_pct"] = ticker_view_raw["otm_pct"].map(lambda x: fmt_pct(x, mult=100.0, decimals=1))
        ticker_view["z_premium"] = ticker_view_raw["z_premium"].map(lambda x: fmt_num(x, 2))
        ticker_view["exec_bias"] = ticker_view_raw["exec_bias"].map(lambda x: fmt_num(x, 2))

        st.dataframe(ticker_view, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# External ingest
# -----------------------------------------------------------------------------
with st.expander("External ingest (CSV)"):
    st.write("Drop CSV files into ./ingest with columns such as datetime, ticker, side, strike, expiration, size, price, notional, exchange.")
    ext = read_external()
    if not ext.empty:
        st.dataframe(ext, use_container_width=True, hide_index=True)
    else:
        st.info("No external CSVs detected.")

st.caption("Data source: Yahoo Finance option chains via yfinance. Signal quality is limited by source granularity and OI update cadence.")
st.caption(f"Latest snapshot saved: {saved_path}")
st.caption("© 2026 AD Fund Management LP")
