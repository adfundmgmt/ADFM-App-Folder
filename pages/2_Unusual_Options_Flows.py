# streamlit_app.py
# ADFM | Unusual Options Flow Tracker
# Nasdaq 100 version with pre-scan underlying filters, lazy loading, and hardened metrics/state handling.

import json
import math
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Unusual Options Flow Tracker", layout="wide")

APP_TITLE = "Unusual Options Flow Tracker"
APP_SUBTITLE = (
    "Nasdaq 100 public-data options anomaly scanner. Wikipedia for constituents, "
    "Yahoo Finance for chains, and anomaly filters tuned for practical tape triage."
)

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.0rem; padding-bottom: 1.5rem; max-width: 1680px;}
    h1, h2, h3 {font-weight: 700; letter-spacing: 0.1px;}
    .stMetric {
        background: #fafafa;
        border: 1px solid #ececec;
        border-radius: 14px;
        padding: 10px 14px;
    }
    .stDataFrame, .stPlotlyChart {
        border-radius: 14px;
    }
    div[data-testid="stSidebarContent"] {
        padding-top: 0.6rem;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.88rem;
        line-height: 1.4;
    }
    .status-card {
        background: #fbfbfb;
        border: 1px solid #ececec;
        border-radius: 14px;
        padding: 12px 14px;
        margin-bottom: 12px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

# -----------------------------------------------------------------------------
# Constants and paths
# -----------------------------------------------------------------------------
CACHE_DIR = Path(".uw_cache")
CACHE_DIR.mkdir(exist_ok=True)

UNIVERSE_META_CACHE = CACHE_DIR / "ndx100_meta.pkl"
LAST_GOOD_SCAN = CACHE_DIR / "last_good_flow.pkl"
LAST_GOOD_DIAGS = CACHE_DIR / "last_good_diags.pkl"
LAST_GOOD_META = CACHE_DIR / "last_good_meta.json"

NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

DEFAULT_MIN_PREMIUM = 50_000.0
DEFAULT_MIN_VOLUME = 100
DEFAULT_MIN_VOL_OI = 1.5
DEFAULT_MAX_SPREAD_PCT = 30.0
DEFAULT_TOP_N = 200
DEFAULT_MAX_WORKERS = 10
DEFAULT_DTE_MIN = 0
DEFAULT_DTE_MAX = 60
DEFAULT_MAX_EXPIRIES_PER_SYMBOL = 8
DEFAULT_MIN_UNUSUAL_SCORE = 55.0
DEFAULT_MIN_UNDERLYING_DOLLAR_VOL_MM = 100.0
DEFAULT_BATCH_SIZE = 40

DEFAULT_MIN_PRICE = 10.0
DEFAULT_MIN_20D_VOL = 1_000_000

NASDAQ100_FALLBACK_SYMBOLS = sorted([
    "AAPL","ABNB","ADBE","ADI","ADP","ADSK","AEP","AMAT","AMD","AMGN","AMZN","ANSS","APP","ARM",
    "ASML","AVGO","AXON","AZN","BIIB","BKNG","CDNS","CEG","CHTR","CMCSA","COST","CPRT","CRWD",
    "CSCO","CSGP","CSX","CTAS","CTSH","DASH","DDOG","DXCM","EA","EXC","FANG","FAST","FTNT",
    "GEHC","GFS","GILD","GOOG","GOOGL","HON","IDXX","INTC","INTU","ISRG","KDP","KHC","KLAC",
    "LIN","LRCX","LULU","MAR","MCHP","MDLZ","MELI","META","MNST","MRVL","MSFT","MU","NFLX",
    "NVDA","NXPI","ODFL","ON","ORLY","PANW","PAYX","PCAR","PDD","PEP","PLTR","PYPL","QCOM",
    "REGN","ROP","ROST","SBUX","SNPS","TEAM","TMUS","TSLA","TTD","TTWO","TXN","VRTX","WBD",
    "WDAY","XEL","ZS"
])

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def human_money(x: float) -> str:
    x = 0.0 if pd.isna(x) else float(x)
    ax = abs(x)
    if ax >= 1_000_000_000:
        return f"${x/1_000_000_000:.2f}B"
    if ax >= 1_000_000:
        return f"${x/1_000_000:.2f}M"
    if ax >= 1_000:
        return f"${x/1_000:.1f}K"
    return f"${x:,.0f}"

def human_num(x: float) -> str:
    x = 0.0 if pd.isna(x) else float(x)
    ax = abs(x)
    if ax >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    if ax >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if ax >= 1_000:
        return f"{x/1_000:.1f}K"
    return f"{x:,.0f}"

def safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def safe_int(x, default=0) -> int:
    try:
        if x is None:
            return int(default)
        if isinstance(x, float) and pd.isna(x):
            return int(default)
        return int(x)
    except Exception:
        return int(default)

def chunked(items: List[str], n: int) -> List[List[str]]:
    return [items[i:i + n] for i in range(0, len(items), n)]

def load_pickle(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def save_pickle(obj, path: Path):
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    except Exception:
        pass

def load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_json(obj: dict, path: Path):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def normalize_symbol(sym: str) -> str:
    sym = str(sym).strip().upper()
    return sym.replace(".", "-")

def classify_side(fill: float, bid: float, ask: float) -> str:
    bid = 0.0 if pd.isna(bid) else float(bid)
    ask = 0.0 if pd.isna(ask) else float(ask)
    fill = 0.0 if pd.isna(fill) else float(fill)

    if bid <= 0 and ask <= 0:
        return "UNKNOWN"
    if ask <= 0:
        return "BID"
    if bid <= 0:
        return "ASK"

    spread = ask - bid
    if spread <= 0:
        return "MID"

    near_bid = abs(fill - bid)
    near_ask = abs(fill - ask)
    tol = max(spread * 0.20, 0.02)

    if near_ask <= tol and near_ask <= near_bid:
        return "ASK"
    if near_bid <= tol and near_bid < near_ask:
        return "BID"
    return "MID"

def classify_direction(option_type: str, side: str) -> str:
    option_type = str(option_type).upper()
    side = str(side).upper()

    if option_type == "CALL":
        if side in ("ASK", "MID"):
            return "BULLISH"
        if side == "BID":
            return "BEARISH"
    elif option_type == "PUT":
        if side == "BID":
            return "BULLISH"
        if side in ("ASK", "MID"):
            return "BEARISH"
    return "NEUTRAL"

def strike_bucket(strike: float, spot: float) -> str:
    if spot <= 0 or pd.isna(spot) or pd.isna(strike):
        return "UNK"
    pct = abs(float(strike) - float(spot)) / float(spot)
    if pct <= 0.01:
        return "0-1%"
    if pct <= 0.03:
        return "1-3%"
    if pct <= 0.05:
        return "3-5%"
    if pct <= 0.10:
        return "5-10%"
    return "10%+"

def dte_bucket(dte: int) -> str:
    dte = int(dte)
    if dte == 0:
        return "0DTE"
    if dte <= 7:
        return "1-7D"
    if dte <= 30:
        return "8-30D"
    if dte <= 60:
        return "31-60D"
    if dte <= 120:
        return "61-120D"
    return "120D+"

# -----------------------------------------------------------------------------
# Universe loader
# -----------------------------------------------------------------------------
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_nasdaq100_from_wikipedia() -> pd.DataFrame:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(NASDAQ100_URL, headers=headers, timeout=20)
        r.raise_for_status()
        tables = pd.read_html(r.text)

        candidate = None
        for tbl in tables:
            cols = [str(c).strip().lower() for c in tbl.columns]
            if ("ticker" in cols or "ticker symbol" in cols or "symbol" in cols) and ("company" in cols or "security" in cols):
                candidate = tbl.copy()
                break

        if candidate is None:
            for tbl in tables:
                cols = [str(c).strip().lower() for c in tbl.columns]
                if "ticker" in cols or "ticker symbol" in cols or "symbol" in cols:
                    candidate = tbl.copy()
                    break

        if candidate is None:
            raise ValueError("Could not find Nasdaq 100 table.")

        candidate.columns = [str(c).strip().lower() for c in candidate.columns]

        rename_map = {}
        if "ticker symbol" in candidate.columns:
            rename_map["ticker symbol"] = "symbol"
        if "ticker" in candidate.columns:
            rename_map["ticker"] = "symbol"
        if "security" in candidate.columns:
            rename_map["security"] = "company"
        candidate = candidate.rename(columns=rename_map)

        if "symbol" not in candidate.columns:
            raise ValueError("No symbol column found in Nasdaq 100 table.")

        if "company" not in candidate.columns:
            candidate["company"] = candidate["symbol"]

        keep_cols = [c for c in ["symbol", "company"] if c in candidate.columns]
        candidate = candidate[keep_cols].copy()
        candidate["symbol"] = candidate["symbol"].astype(str).map(normalize_symbol)
        candidate["company"] = candidate["company"].astype(str)

        candidate = candidate.dropna(subset=["symbol"]).drop_duplicates("symbol").reset_index(drop=True)

        if len(candidate) < 80:
            raise ValueError(f"Nasdaq 100 table looked incomplete. Rows: {len(candidate)}")

        save_pickle(candidate, UNIVERSE_META_CACHE)
        return candidate

    except Exception:
        cached = load_pickle(UNIVERSE_META_CACHE)
        if isinstance(cached, pd.DataFrame) and not cached.empty and "symbol" in cached.columns:
            cached = cached.copy()
            cached["symbol"] = cached["symbol"].astype(str).map(normalize_symbol)
            cached = cached.drop_duplicates("symbol").reset_index(drop=True)
            return cached

        df = pd.DataFrame({
            "symbol": [normalize_symbol(x) for x in NASDAQ100_FALLBACK_SYMBOLS],
            "company": [normalize_symbol(x) for x in NASDAQ100_FALLBACK_SYMBOLS],
        })
        return df.drop_duplicates("symbol").reset_index(drop=True)

# -----------------------------------------------------------------------------
# Spot snapshot
# -----------------------------------------------------------------------------
def _extract_snapshot_from_download(data: pd.DataFrame, symbols: List[str]) -> List[dict]:
    rows = []
    if data is None or len(data) == 0:
        return rows

    multi = isinstance(data.columns, pd.MultiIndex)

    for sym in symbols:
        try:
            if multi:
                level0 = set(data.columns.get_level_values(0))
                if sym not in level0:
                    continue
                sub = data[sym].copy()
            else:
                sub = data.copy()

            sub = sub.dropna(how="all")
            if sub.empty or "Close" not in sub.columns:
                continue

            close = sub["Close"].dropna()
            vol = sub["Volume"].dropna() if "Volume" in sub.columns else pd.Series(dtype=float)

            if close.empty:
                continue

            spot = float(close.iloc[-1])
            prev_close = float(close.iloc[-2]) if len(close) >= 2 else spot
            avg_20d_vol = float(vol.tail(20).mean()) if len(vol) > 0 else np.nan
            day_volume = float(vol.iloc[-1]) if len(vol) > 0 else np.nan

            aligned = pd.concat([close.rename("close"), vol.rename("vol")], axis=1).dropna()
            avg_dollar_vol = float((aligned["close"] * aligned["vol"]).tail(20).mean()) if not aligned.empty else np.nan

            rows.append({
                "symbol": sym,
                "spot": spot,
                "prev_close": prev_close,
                "avg_20d_vol": avg_20d_vol,
                "day_volume": day_volume,
                "avg_dollar_vol": avg_dollar_vol,
            })
        except Exception:
            continue

    return rows

def fetch_spot_snapshot(symbols: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["symbol", "spot", "prev_close", "avg_20d_vol", "day_volume", "avg_dollar_vol"])

    all_rows: List[dict] = []

    for batch in chunked(symbols, batch_size):
        try:
            data = yf.download(
                tickers=batch,
                period="3mo",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="ticker",
            )
            rows = _extract_snapshot_from_download(data, batch)
            all_rows.extend(rows)
        except Exception:
            continue

    if not all_rows:
        return pd.DataFrame(columns=["symbol", "spot", "prev_close", "avg_20d_vol", "day_volume", "avg_dollar_vol"])

    return pd.DataFrame(all_rows).drop_duplicates("symbol", keep="last").reset_index(drop=True)

def filter_underlyings(
    snapshot: pd.DataFrame,
    min_price: float,
    min_avg_dollar_vol_mm: float,
    min_avg_20d_vol: int,
) -> pd.DataFrame:
    if snapshot is None or snapshot.empty:
        return pd.DataFrame(columns=snapshot.columns if snapshot is not None else [])

    out = snapshot.copy()
    out = out[pd.to_numeric(out["spot"], errors="coerce") >= float(min_price)]
    out = out[pd.to_numeric(out["avg_dollar_vol"], errors="coerce") >= float(min_avg_dollar_vol_mm) * 1_000_000]
    out = out[pd.to_numeric(out["avg_20d_vol"], errors="coerce") >= float(min_avg_20d_vol)]
    return out.reset_index(drop=True)

# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------
def compute_unusual_score(
    premium: float,
    volume: float,
    open_interest: float,
    spread_pct: float,
    pct_otm: float,
    dte: int,
    side: str,
    avg_dollar_vol: float,
) -> float:
    premium_score = min(math.log10(max(premium, 0) + 1.0) / 6.7, 1.0)
    abs_size_score = min(max(volume, 0) / 3000.0, 1.0)
    voi_score = min((max(volume, 0) / max(open_interest, 1.0)) / 6.0, 1.0)
    otm_score = min(max(pct_otm, 0) / 0.10, 1.0)
    dte_score = 1.0 if dte <= 30 else max(0.25, 1.0 - ((dte - 30) / 365.0))
    spread_penalty = min(max(spread_pct, 0) / 80.0, 1.0)
    side_bonus = 0.05 if side in ("ASK", "BID") else 0.02 if side == "MID" else 0.0

    if pd.isna(avg_dollar_vol) or avg_dollar_vol <= 0:
        liquidity_score = 0.4
    else:
        liquidity_score = min(math.log10(avg_dollar_vol + 1.0) / 9.0, 1.0)

    raw = (
        0.34 * premium_score
        + 0.20 * voi_score
        + 0.14 * abs_size_score
        + 0.08 * otm_score
        + 0.08 * dte_score
        + 0.08 * liquidity_score
        + side_bonus
        - 0.16 * spread_penalty
    )
    return float(np.clip(raw, 0.0, 1.0) * 100.0)

# -----------------------------------------------------------------------------
# One-symbol scan
# -----------------------------------------------------------------------------
def scan_one_symbol(
    symbol: str,
    spot_map: Dict[str, float],
    adv_map: Dict[str, float],
    min_premium: float,
    min_volume: int,
    min_vol_oi: float,
    max_spread_pct: float,
    dte_min: int,
    dte_max: int,
    max_expiries_per_symbol: int,
    min_unusual_score: float,
) -> Tuple[List[dict], dict]:
    diagnostics = {
        "symbol": symbol,
        "expiries_seen": 0,
        "contracts_seen": 0,
        "contracts_kept": 0,
        "errors": 0,
        "reason": "",
    }
    rows: List[dict] = []

    try:
        tkr = yf.Ticker(symbol)
        expiries = list(tkr.options or [])
    except Exception:
        diagnostics["errors"] = 1
        diagnostics["reason"] = "options_load_failed"
        return rows, diagnostics

    if not expiries:
        diagnostics["reason"] = "no_listed_options"
        return rows, diagnostics

    spot = safe_float(spot_map.get(symbol, np.nan), np.nan)
    avg_dollar_vol = safe_float(adv_map.get(symbol, np.nan), np.nan)

    if pd.isna(spot) or spot <= 0:
        diagnostics["reason"] = "spot_missing"
        return rows, diagnostics

    today = pd.Timestamp.utcnow().date()

    parsed_expiries = []
    for exp in expiries:
        try:
            exp_date = pd.Timestamp(exp).date()
            dte = (exp_date - today).days
            if dte_min <= dte <= dte_max:
                parsed_expiries.append((exp, dte))
        except Exception:
            continue

    parsed_expiries = sorted(parsed_expiries, key=lambda x: x[1])[:max_expiries_per_symbol]

    if not parsed_expiries:
        diagnostics["reason"] = "no_expiry_in_range"
        return rows, diagnostics

    for exp, dte in parsed_expiries:
        diagnostics["expiries_seen"] += 1
        try:
            chain = tkr.option_chain(exp)
            chain_parts = [("CALL", chain.calls), ("PUT", chain.puts)]
        except Exception:
            diagnostics["errors"] += 1
            continue

        for option_type, df in chain_parts:
            if df is None or df.empty:
                continue

            local = df.copy()
            for col in ["contractSymbol", "strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]:
                if col not in local.columns:
                    local[col] = np.nan

            local = local[["contractSymbol", "strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]].copy()
            diagnostics["contracts_seen"] += len(local)

            local["strike"] = pd.to_numeric(local["strike"], errors="coerce")
            local["lastPrice"] = pd.to_numeric(local["lastPrice"], errors="coerce").fillna(0.0)
            local["bid"] = pd.to_numeric(local["bid"], errors="coerce").fillna(0.0)
            local["ask"] = pd.to_numeric(local["ask"], errors="coerce").fillna(0.0)
            local["volume"] = pd.to_numeric(local["volume"], errors="coerce").fillna(0.0)
            local["openInterest"] = pd.to_numeric(local["openInterest"], errors="coerce").fillna(0.0)
            local["impliedVolatility"] = pd.to_numeric(local["impliedVolatility"], errors="coerce")

            local = local[(local["strike"] > 0) & (local["volume"] >= min_volume)].copy()
            if local.empty:
                continue

            mids = np.where(
                (local["bid"] > 0) & (local["ask"] > 0),
                (local["bid"] + local["ask"]) / 2.0,
                np.where(local["ask"] > 0, local["ask"], np.where(local["bid"] > 0, local["bid"], 0.0))
            )
            fills = np.where(
                local["lastPrice"] > 0, local["lastPrice"],
                np.where(mids > 0, mids, np.where(local["ask"] > 0, local["ask"], local["bid"]))
            )

            local["mid"] = mids
            local["fill_est"] = fills
            local = local[local["fill_est"] > 0].copy()
            if local.empty:
                continue

            local["premium"] = local["fill_est"] * local["volume"] * 100.0
            local = local[local["premium"] >= min_premium].copy()
            if local.empty:
                continue

            local["vol_oi"] = local["volume"] / np.maximum(local["openInterest"], 1.0)
            local = local[local["vol_oi"] >= min_vol_oi].copy()
            if local.empty:
                continue

            local["spread_pct"] = np.where(
                (local["mid"] > 0) & (local["ask"] >= local["bid"]) & (local["bid"] > 0),
                ((local["ask"] - local["bid"]) / local["mid"]) * 100.0,
                np.nan
            )
            local = local[(local["spread_pct"].isna()) | (local["spread_pct"] <= max_spread_pct)].copy()
            if local.empty:
                continue

            local["side"] = [
                classify_side(f, b, a)
                for f, b, a in zip(local["fill_est"], local["bid"], local["ask"])
            ]
            local["direction"] = [
                classify_direction(option_type, s)
                for s in local["side"]
            ]

            if option_type == "CALL":
                local["pct_otm_dec"] = np.maximum((local["strike"] - spot) / spot, 0.0)
                local["moneyness"] = np.where(
                    local["strike"] > spot, "OTM",
                    np.where(local["strike"] < spot, "ITM", "ATM")
                )
            else:
                local["pct_otm_dec"] = np.maximum((spot - local["strike"]) / spot, 0.0)
                local["moneyness"] = np.where(
                    local["strike"] < spot, "OTM",
                    np.where(local["strike"] > spot, "ITM", "ATM")
                )

            local["unusual_score"] = [
                compute_unusual_score(
                    premium=prem,
                    volume=vol,
                    open_interest=oi,
                    spread_pct=0.0 if pd.isna(sp) else sp,
                    pct_otm=pct_otm,
                    dte=dte,
                    side=side,
                    avg_dollar_vol=avg_dollar_vol,
                )
                for prem, vol, oi, sp, pct_otm, side in zip(
                    local["premium"],
                    local["volume"],
                    local["openInterest"],
                    local["spread_pct"],
                    local["pct_otm_dec"],
                    local["side"]
                )
            ]

            local = local[local["unusual_score"] >= min_unusual_score].copy()
            if local.empty:
                continue

            local["symbol"] = symbol
            local["type"] = option_type
            local["expiry"] = exp
            local["dte"] = int(dte)
            local["spot"] = float(spot)
            local["pct_otm"] = local["pct_otm_dec"] * 100.0
            local["avg_dollar_vol"] = avg_dollar_vol

            rows.extend(local[[
                "symbol", "contractSymbol", "type", "expiry", "dte", "spot", "strike",
                "moneyness", "pct_otm", "side", "direction", "bid", "ask", "mid",
                "fill_est", "volume", "openInterest", "vol_oi", "premium",
                "spread_pct", "impliedVolatility", "unusual_score", "avg_dollar_vol"
            ]].rename(columns={
                "contractSymbol": "contract_symbol",
                "openInterest": "open_interest",
                "impliedVolatility": "iv"
            }).to_dict("records"))

            diagnostics["contracts_kept"] += len(local)

    if diagnostics["contracts_kept"] == 0 and diagnostics["reason"] == "":
        diagnostics["reason"] = "all_filtered_out"

    return rows, diagnostics

# -----------------------------------------------------------------------------
# Cluster logic
# -----------------------------------------------------------------------------
def build_cluster_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    out = df.copy()
    stale_cols = [
        "strike_bucket", "cluster_key", "cluster_contracts", "cluster_premium",
        "cluster_volume", "cluster_score", "cluster_flag"
    ]
    out = out.drop(columns=[c for c in stale_cols if c in out.columns], errors="ignore")

    required = ["symbol", "expiry", "type", "direction", "strike", "spot", "premium", "volume", "unusual_score"]
    if any(c not in out.columns for c in required):
        out["cluster_contracts"] = np.nan
        out["cluster_premium"] = np.nan
        out["cluster_volume"] = np.nan
        out["cluster_score"] = np.nan
        out["cluster_flag"] = ""
        return out

    out["strike_bucket"] = [strike_bucket(k, s) for k, s in zip(out["strike"], out["spot"])]
    out["cluster_key"] = (
        out["symbol"].astype(str) + "|" +
        out["expiry"].astype(str) + "|" +
        out["type"].astype(str) + "|" +
        out["direction"].astype(str) + "|" +
        out["strike_bucket"].astype(str)
    )

    grp = (
        out.groupby("cluster_key", dropna=False)
        .agg(
            cluster_contracts=("contract_symbol", "count"),
            cluster_premium=("premium", "sum"),
            cluster_volume=("volume", "sum"),
            cluster_score=("unusual_score", "mean"),
        )
        .reset_index()
    )

    out = out.merge(grp, on="cluster_key", how="left", validate="many_to_one")
    out["cluster_contracts"] = out["cluster_contracts"].fillna(1)
    out["cluster_premium"] = out["cluster_premium"].fillna(0.0)
    out["cluster_volume"] = out["cluster_volume"].fillna(0.0)
    out["cluster_score"] = out["cluster_score"].fillna(0.0)

    out["cluster_flag"] = np.where(
        (out["cluster_contracts"] >= 3) & (out["cluster_premium"] >= 500_000),
        "CLUSTER",
        ""
    )
    return out

# -----------------------------------------------------------------------------
# Full scan
# -----------------------------------------------------------------------------
def run_full_scan(
    symbols: List[str],
    spot_snapshot: pd.DataFrame,
    min_premium: float,
    min_volume: int,
    min_vol_oi: float,
    max_spread_pct: float,
    dte_min: int,
    dte_max: int,
    max_workers: int,
    max_expiries_per_symbol: int,
    min_unusual_score: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    spot_map: Dict[str, float] = {}
    adv_map: Dict[str, float] = {}

    if spot_snapshot is not None and not spot_snapshot.empty:
        spot_map = dict(zip(spot_snapshot["symbol"], spot_snapshot["spot"]))
        adv_map = dict(zip(spot_snapshot["symbol"], spot_snapshot["avg_dollar_vol"]))

    all_rows: List[dict] = []
    all_diags: List[dict] = []

    progress_bar = st.progress(0.0)
    status = st.empty()

    total = len(symbols)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                scan_one_symbol,
                sym,
                spot_map,
                adv_map,
                min_premium,
                min_volume,
                min_vol_oi,
                max_spread_pct,
                dte_min,
                dte_max,
                max_expiries_per_symbol,
                min_unusual_score,
            ): sym
            for sym in symbols
        }

        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                rows, diag = fut.result()
                if rows:
                    all_rows.extend(rows)
                all_diags.append(diag)
            except Exception:
                all_diags.append({
                    "symbol": sym,
                    "expiries_seen": 0,
                    "contracts_seen": 0,
                    "contracts_kept": 0,
                    "errors": 1,
                    "reason": "worker_exception",
                })

            completed += 1
            progress_bar.progress(completed / max(total, 1))
            status.caption(f"Scanning chains... {completed:,}/{total:,} symbols completed")

    progress_bar.empty()
    status.empty()

    flow = pd.DataFrame(all_rows)
    diags = pd.DataFrame(all_diags)

    if not flow.empty:
        flow = build_cluster_flags(flow)
        flow = flow.sort_values(
            ["unusual_score", "premium", "vol_oi", "volume"],
            ascending=[False, False, False, False]
        ).reset_index(drop=True)

    return flow, diags

# -----------------------------------------------------------------------------
# Commentary and schema helpers
# -----------------------------------------------------------------------------
def summarize_tape(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No qualifying anomalies made it through the current public-data filters."

    bullish = float(df.loc[df["direction"] == "BULLISH", "premium"].sum()) if "direction" in df.columns else 0.0
    bearish = float(df.loc[df["direction"] == "BEARISH", "premium"].sum()) if "direction" in df.columns else 0.0
    total = float(df["premium"].sum()) if "premium" in df.columns else 0.0

    top = df.iloc[0]
    top_tickers = (
        df.groupby("symbol", as_index=False)["premium"].sum()
        .sort_values("premium", ascending=False)
        .head(5)
    )
    concentration = float(top_tickers["premium"].sum() / total) if total > 0 else 0.0

    dte_mix = (
        df.assign(dte_bucket=df["dte"].map(dte_bucket))
        .groupby("dte_bucket", as_index=False)["premium"]
        .sum()
        .sort_values("premium", ascending=False)
    )
    lead_bucket = dte_mix.iloc[0]["dte_bucket"] if not dte_mix.empty else "N/A"

    clusters = int((df["cluster_flag"] == "CLUSTER").sum()) if "cluster_flag" in df.columns else 0
    cluster_premium = float(df.loc[df["cluster_flag"] == "CLUSTER", "premium"].sum()) if "cluster_flag" in df.columns else 0.0

    leader = (
        f"Top print is {top['symbol']} {top['type'].lower()} flow in the {top['expiry']} "
        f"{top['strike']:.0f} strike, estimated premium {human_money(top['premium'])}, "
        f"volume/OI {top['vol_oi']:.2f}x, score {top['unusual_score']:.1f}."
    )

    if bullish > bearish * 1.25:
        tilt = f"Directional premium skews bullish at {human_money(bullish)} versus {human_money(bearish)} bearish."
    elif bearish > bullish * 1.25:
        tilt = f"Directional premium skews bearish at {human_money(bearish)} versus {human_money(bullish)} bullish."
    else:
        tilt = f"Directional premium is relatively balanced with {human_money(bullish)} bullish and {human_money(bearish)} bearish."

    breadth = (
        f"Concentration is {'high' if concentration >= 0.60 else 'moderate' if concentration >= 0.40 else 'broad'}, "
        f"with the top five tickers accounting for {concentration:.0%} of qualifying premium."
    )

    tenor = f"The tape is led by the {lead_bucket} bucket."
    cluster_line = (
        f"Cluster behavior is meaningful, with {clusters:,} flagged rows representing {human_money(cluster_premium)} of premium."
        if clusters > 0 else
        "The board is more single-print than cluster-driven."
    )

    return " ".join([leader, tilt, breadth, tenor, cluster_line])

def ensure_flow_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    defaults = {
        "avg_dollar_vol": np.nan,
        "iv": np.nan,
        "spread_pct": np.nan,
        "pct_otm": np.nan,
        "cluster_flag": "",
        "cluster_contracts": np.nan,
        "cluster_premium": np.nan,
        "cluster_volume": np.nan,
        "cluster_score": np.nan,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    return out

def prep_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = ensure_flow_schema(df)

    out["Premium"] = out["premium"].map(human_money) if "premium" in out.columns else ""
    out["Vol"] = out["volume"].map(human_num) if "volume" in out.columns else ""
    out["OI"] = out["open_interest"].map(human_num) if "open_interest" in out.columns else ""
    out["Vol/OI"] = out["vol_oi"].map(lambda x: f"{x:.2f}x") if "vol_oi" in out.columns else ""
    out["Fill"] = out["fill_est"].map(lambda x: f"{x:.2f}") if "fill_est" in out.columns else ""
    out["Bid"] = out["bid"].map(lambda x: f"{x:.2f}") if "bid" in out.columns else ""
    out["Ask"] = out["ask"].map(lambda x: f"{x:.2f}") if "ask" in out.columns else ""
    out["Spread %"] = out["spread_pct"].map(lambda x: "" if pd.isna(x) else f"{x:.1f}%") if "spread_pct" in out.columns else ""
    out["IV"] = out["iv"].map(lambda x: "" if pd.isna(x) else f"{x:.1%}") if "iv" in out.columns else ""
    out["OTM %"] = out["pct_otm"].map(lambda x: "" if pd.isna(x) else f"{x:.1f}%") if "pct_otm" in out.columns else ""
    out["Score"] = out["unusual_score"].map(lambda x: f"{x:.1f}") if "unusual_score" in out.columns else ""
    out["ADV"] = out["avg_dollar_vol"].map(lambda x: "" if pd.isna(x) else human_money(x)) if "avg_dollar_vol" in out.columns else ""

    keep = [
        "symbol", "type", "expiry", "dte", "direction", "side", "strike", "spot",
        "moneyness", "OTM %", "Fill", "Bid", "Ask", "Spread %", "Vol", "OI",
        "Vol/OI", "Premium", "IV", "Score", "ADV", "cluster_flag", "contract_symbol"
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep].copy()

def render_top_flow_dashboard(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("No top-flow rows to show.")
        return

    df = ensure_flow_schema(df).copy().head(30)
    df["label"] = (
        df["symbol"].astype(str) + " | " +
        df["type"].astype(str) + " | " +
        df["expiry"].astype(str) + " | " +
        df["strike"].round(0).astype(int).astype(str)
    )

    fig = px.bar(
        df.sort_values(["premium", "unusual_score"], ascending=[True, True]),
        x="premium",
        y="label",
        color="direction",
        orientation="h",
        hover_data={
            "symbol": True,
            "type": True,
            "expiry": True,
            "strike": ":.2f",
            "dte": True,
            "vol_oi": ":.2f",
            "spread_pct": ":.1f",
            "unusual_score": ":.1f",
            "premium": ":,.0f",
        },
        title="Top Flow Leaderboard"
    )
    fig.update_layout(height=780, margin=dict(l=20, r=20, t=55, b=20), yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    inspect_choice = st.selectbox(
        "Inspect contract",
        options=df["label"].tolist(),
        index=0,
        key="top_flow_inspect_choice"
    )
    row = df.loc[df["label"] == inspect_choice].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Premium", human_money(row["premium"]))
    c2.metric("Vol/OI", f"{row['vol_oi']:.2f}x")
    c3.metric("Score", f"{row['unusual_score']:.1f}")
    c4.metric("Spread %", "" if pd.isna(row["spread_pct"]) else f"{row['spread_pct']:.1f}%")

    st.markdown(
        f"""
        **{row['symbol']} {row['type']} {int(round(row['strike']))} {row['expiry']}**

        Direction: **{row['direction']}**  
        Side estimate: **{row['side']}**  
        DTE: **{int(row['dte'])}**  
        Spot: **{row['spot']:.2f}**  
        Moneyness: **{row['moneyness']}**  
        OTM distance: **{row['pct_otm']:.1f}%**  
        Cluster flag: **{row['cluster_flag'] if str(row['cluster_flag']).strip() else 'None'}**
        """
    )

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
def init_state():
    defaults = {
        "flow_raw": None,
        "diags_raw": None,
        "scan_meta": None,
        "scan_ran": False,
        "filtered_snapshot": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# -----------------------------------------------------------------------------
# Universe loads fast
# -----------------------------------------------------------------------------
universe = fetch_nasdaq100_from_wikipedia()
symbols_all = universe["symbol"].dropna().astype(str).map(normalize_symbol).unique().tolist()

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Purpose: unusual options-flow scanner for outsized listed-options activity across Nasdaq 100 names.

        Improvements here:
        • smaller universe
        • pre-scan underlying filters
        • lazy loading of heavy data
        • chart-driven review
        """
    )

    st.subheader("Underlying Pre-Filters")
    min_price = st.number_input("Minimum stock price ($)", min_value=0.0, value=float(DEFAULT_MIN_PRICE), step=5.0, format="%.0f")
    min_avg_20d_vol = st.number_input("Minimum 20D avg share volume", min_value=0, value=int(DEFAULT_MIN_20D_VOL), step=250_000)
    min_underlying_dollar_vol_mm = st.number_input(
        "Minimum underlying ADV ($mm)",
        min_value=0.0,
        value=float(DEFAULT_MIN_UNDERLYING_DOLLAR_VOL_MM),
        step=25.0,
        format="%.0f"
    )

    st.subheader("Anomaly Filters")
    min_premium = st.number_input("Minimum premium ($)", min_value=0.0, value=float(DEFAULT_MIN_PREMIUM), step=25_000.0, format="%.0f")
    min_volume = st.number_input("Minimum contract volume", min_value=1, value=int(DEFAULT_MIN_VOLUME), step=25)
    min_vol_oi = st.number_input("Minimum volume / OI", min_value=0.0, value=float(DEFAULT_MIN_VOL_OI), step=0.25, format="%.2f")
    max_spread_pct = st.number_input("Maximum spread %", min_value=0.0, value=float(DEFAULT_MAX_SPREAD_PCT), step=5.0, format="%.1f")
    min_unusual_score = st.slider("Minimum unusual score", min_value=0.0, max_value=100.0, value=float(DEFAULT_MIN_UNUSUAL_SCORE), step=1.0)
    dte_range = st.slider("DTE range", min_value=0, max_value=365, value=(DEFAULT_DTE_MIN, DEFAULT_DTE_MAX), step=1)

    st.subheader("Scan Engine")
    max_expiries_per_symbol = st.slider("Max expiries per symbol", min_value=1, max_value=16, value=DEFAULT_MAX_EXPIRIES_PER_SYMBOL, step=1)
    max_workers = st.slider("Concurrent workers", min_value=2, max_value=24, value=DEFAULT_MAX_WORKERS, step=2)
    top_n = st.slider("Rows to display", min_value=25, max_value=300, value=DEFAULT_TOP_N, step=25)
    allow_cached_fallback = st.toggle("Use last-good cached scan when live scan is degraded", value=True)

    run_scan = st.button("Run Live Scan", use_container_width=True, type="primary")

    st.markdown(
        '<div class="small-note">This version fetches the Nasdaq 100 universe quickly, then only pulls spot data and option chains after you click Run Live Scan. It also filters underlyings before scanning options.</div>',
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Cached startup load
# -----------------------------------------------------------------------------
if not st.session_state["scan_ran"]:
    cached_flow = load_pickle(LAST_GOOD_SCAN)
    cached_diags = load_pickle(LAST_GOOD_DIAGS)
    cached_meta = load_json(LAST_GOOD_META)

    if isinstance(cached_flow, pd.DataFrame) and not cached_flow.empty:
        st.session_state["flow_raw"] = ensure_flow_schema(build_cluster_flags(cached_flow.copy()))
        st.session_state["diags_raw"] = cached_diags if isinstance(cached_diags, pd.DataFrame) else pd.DataFrame()
        st.session_state["scan_meta"] = {
            "mode": "Cached startup snapshot",
            "saved_at": cached_meta.get("saved_at", "unknown") if isinstance(cached_meta, dict) else "unknown",
            "elapsed": None,
            "success_ratio": cached_meta.get("success_ratio", None) if isinstance(cached_meta, dict) else None,
            "raw_contracts": cached_meta.get("raw_contracts", None) if isinstance(cached_meta, dict) else None,
            "symbols_requested": safe_int(cached_meta.get("symbols_requested", len(symbols_all))) if isinstance(cached_meta, dict) else len(symbols_all),
            "symbols_scanned": safe_int(cached_meta.get("symbols_scanned", 0)) if isinstance(cached_meta, dict) else 0,
        }
    else:
        st.session_state["flow_raw"] = pd.DataFrame()
        st.session_state["diags_raw"] = pd.DataFrame()
        st.session_state["scan_meta"] = {
            "mode": "No scan loaded yet",
            "saved_at": None,
            "elapsed": None,
            "success_ratio": None,
            "raw_contracts": None,
            "symbols_requested": len(symbols_all),
            "symbols_scanned": 0,
        }

# -----------------------------------------------------------------------------
# Run scan
# -----------------------------------------------------------------------------
if run_scan:
    scan_started = time.time()

    with st.spinner(f"Fetching Nasdaq 100 spot data for {len(symbols_all):,} symbols..."):
        snapshot = fetch_spot_snapshot(symbols_all, batch_size=DEFAULT_BATCH_SIZE)

    filtered_snapshot = filter_underlyings(
        snapshot=snapshot,
        min_price=min_price,
        min_avg_dollar_vol_mm=min_underlying_dollar_vol_mm,
        min_avg_20d_vol=min_avg_20d_vol,
    )
    symbols_scanned_list = filtered_snapshot["symbol"].dropna().astype(str).tolist()
    symbols_scanned_count = len(symbols_scanned_list)

    st.session_state["filtered_snapshot"] = filtered_snapshot.copy()

    with st.spinner(f"Running live options scan across {symbols_scanned_count:,} filtered Nasdaq 100 symbols..."):
        try:
            flow_live, diags_live = run_full_scan(
                symbols=symbols_scanned_list,
                spot_snapshot=filtered_snapshot,
                min_premium=min_premium,
                min_volume=min_volume,
                min_vol_oi=min_vol_oi,
                max_spread_pct=max_spread_pct,
                dte_min=dte_range[0],
                dte_max=dte_range[1],
                max_workers=max_workers,
                max_expiries_per_symbol=max_expiries_per_symbol,
                min_unusual_score=min_unusual_score,
            )
        except Exception as e:
            st.warning(f"Live scan hit an error: {e}")
            flow_live = pd.DataFrame()
            diags_live = pd.DataFrame()

    success_ratio = 0.0
    raw_contracts = 0
    if diags_live is not None and not diags_live.empty:
        raw_contracts = int(diags_live["contracts_seen"].sum()) if "contracts_seen" in diags_live.columns else 0
        successes = int((diags_live["errors"].fillna(0) == 0).sum()) if "errors" in diags_live.columns else 0
        success_ratio = successes / max(len(diags_live), 1)

    degraded = (
        diags_live is None or diags_live.empty or
        success_ratio < 0.50 or
        raw_contracts < 1_000
    )

    if degraded and allow_cached_fallback:
        cached_flow = load_pickle(LAST_GOOD_SCAN)
        cached_diags = load_pickle(LAST_GOOD_DIAGS)
        cached_meta = load_json(LAST_GOOD_META)

        if isinstance(cached_flow, pd.DataFrame) and not cached_flow.empty:
            st.session_state["flow_raw"] = ensure_flow_schema(build_cluster_flags(cached_flow.copy()))
            st.session_state["diags_raw"] = cached_diags if isinstance(cached_diags, pd.DataFrame) else pd.DataFrame()
            st.session_state["scan_meta"] = {
                "mode": "Cached fallback",
                "saved_at": cached_meta.get("saved_at", "unknown") if isinstance(cached_meta, dict) else "unknown",
                "elapsed": round(time.time() - scan_started, 1),
                "success_ratio": success_ratio,
                "raw_contracts": raw_contracts,
                "symbols_requested": len(symbols_all),
                "symbols_scanned": symbols_scanned_count,
            }
            st.session_state["scan_ran"] = True
            st.warning("Live scan looked degraded, so the app loaded the last-good cached snapshot.")
        else:
            st.session_state["flow_raw"] = ensure_flow_schema(flow_live)
            st.session_state["diags_raw"] = diags_live
            st.session_state["scan_meta"] = {
                "mode": "Live",
                "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed": round(time.time() - scan_started, 1),
                "success_ratio": success_ratio,
                "raw_contracts": raw_contracts,
                "symbols_requested": len(symbols_all),
                "symbols_scanned": symbols_scanned_count,
            }
            st.session_state["scan_ran"] = True
    else:
        st.session_state["flow_raw"] = ensure_flow_schema(flow_live)
        st.session_state["diags_raw"] = diags_live
        st.session_state["scan_meta"] = {
            "mode": "Live",
            "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed": round(time.time() - scan_started, 1),
            "success_ratio": success_ratio,
            "raw_contracts": raw_contracts,
            "symbols_requested": len(symbols_all),
            "symbols_scanned": symbols_scanned_count,
        }
        st.session_state["scan_ran"] = True

        if flow_live is not None and not flow_live.empty:
            save_pickle(flow_live, LAST_GOOD_SCAN)
            save_pickle(diags_live, LAST_GOOD_DIAGS)
            save_json(
                {
                    "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "rows": int(len(flow_live)),
                    "symbols_requested": int(len(symbols_all)),
                    "symbols_scanned": int(symbols_scanned_count),
                    "raw_contracts": raw_contracts,
                    "success_ratio": success_ratio,
                },
                LAST_GOOD_META,
            )

# -----------------------------------------------------------------------------
# Main frames
# -----------------------------------------------------------------------------
flow = st.session_state["flow_raw"] if st.session_state["flow_raw"] is not None else pd.DataFrame()
diags = st.session_state["diags_raw"] if st.session_state["diags_raw"] is not None else pd.DataFrame()
scan_meta = st.session_state["scan_meta"]
filtered_snapshot = st.session_state["filtered_snapshot"]

if flow is not None and not flow.empty:
    flow = ensure_flow_schema(build_cluster_flags(flow.copy()))

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
symbols_requested = safe_int(len(symbols_all))
symbols_scanned = safe_int(scan_meta.get("symbols_scanned", 0) if isinstance(scan_meta, dict) else 0)
symbols_completed = safe_int(diags["symbol"].nunique()) if not diags.empty and "symbol" in diags.columns else 0
contracts_seen = safe_int(diags["contracts_seen"].sum()) if not diags.empty and "contracts_seen" in diags.columns else 0
scan_errors = safe_int(diags["errors"].sum()) if not diags.empty and "errors" in diags.columns else 0
qualifying_rows = safe_int(len(flow)) if flow is not None and not flow.empty else 0

bullish_premium = float(flow.loc[flow["direction"] == "BULLISH", "premium"].sum()) if qualifying_rows > 0 else 0.0
bearish_premium = float(flow.loc[flow["direction"] == "BEARISH", "premium"].sum()) if qualifying_rows > 0 else 0.0
total_premium = float(flow["premium"].sum()) if qualifying_rows > 0 else 0.0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Universe size", f"{symbols_requested:,}")
c2.metric("Symbols scanned", f"{symbols_scanned:,}")
c3.metric("Symbols completed", f"{symbols_completed:,}")
c4.metric("Contracts seen", f"{contracts_seen:,}")
c5.metric("Qualifying rows", f"{qualifying_rows:,}")
c6.metric("Total premium", human_money(total_premium))

if scan_meta:
    mode = scan_meta.get("mode", "Unknown")
    saved_at = scan_meta.get("saved_at", "Unknown")
    elapsed = scan_meta.get("elapsed", None)
    success_ratio = scan_meta.get("success_ratio", None)
    raw_contracts = scan_meta.get("raw_contracts", None)

    meta_text = f"Scan mode: {mode}"
    if saved_at:
        meta_text += f" | Snapshot: {saved_at}"
    meta_text += f" | Bullish premium: {human_money(bullish_premium)} | Bearish premium: {human_money(bearish_premium)} | Errors: {scan_errors:,}"
    if elapsed is not None:
        meta_text += f" | Elapsed: {elapsed:.1f}s"
    if success_ratio is not None:
        meta_text += f" | Success ratio: {success_ratio:.0%}"
    if raw_contracts is not None:
        meta_text += f" | Raw contracts: {safe_int(raw_contracts):,}"
    st.caption(meta_text)

# -----------------------------------------------------------------------------
# No-scan and empty states
# -----------------------------------------------------------------------------
if not st.session_state["scan_ran"] and flow.empty:
    st.markdown(
        """
        <div class="status-card">
        <b>No scan has been run yet.</b><br>
        This version uses the Nasdaq 100 plus pre-scan underlying filters. Click <b>Run Live Scan</b> in the sidebar.
        </div>
        """,
        unsafe_allow_html=True,
    )

if flow.empty:
    left, right = st.columns([1.1, 1.0])

    with left:
        st.subheader("Why the board is empty")
        if diags.empty:
            st.info("No diagnostics yet. Run the live scan first.")
        else:
            reason_counts = (
                diags["reason"]
                .fillna("unknown")
                .replace("", "kept_none")
                .value_counts()
                .rename_axis("reason")
                .reset_index(name="count")
            )
            fig_reason = px.bar(
                reason_counts.sort_values("count", ascending=True),
                x="count",
                y="reason",
                orientation="h",
                title="Top Failure Reasons"
            )
            fig_reason.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20), yaxis_title="")
            st.plotly_chart(fig_reason, use_container_width=True)

    with right:
        st.subheader("Practical fixes")
        st.markdown(
            """
            Most useful levers:
            • lower minimum premium
            • lower minimum volume
            • lower minimum volume / OI
            • raise max spread %
            • lower minimum unusual score
            • expand DTE range
            • loosen pre-scan underlying filters
            """
        )
        if filtered_snapshot is not None and isinstance(filtered_snapshot, pd.DataFrame) and not filtered_snapshot.empty:
            st.dataframe(filtered_snapshot[["symbol", "spot", "avg_20d_vol", "avg_dollar_vol"]], use_container_width=True, hide_index=True)

    st.stop()

# -----------------------------------------------------------------------------
# Derived views
# -----------------------------------------------------------------------------
flow = flow.copy()
flow["rank"] = np.arange(1, len(flow) + 1)

overall = flow.nlargest(top_n, ["unusual_score", "premium", "vol_oi"])
bullish = flow[flow["direction"] == "BULLISH"].nlargest(top_n, ["unusual_score", "premium", "vol_oi"])
bearish = flow[flow["direction"] == "BEARISH"].nlargest(top_n, ["unusual_score", "premium", "vol_oi"])
zero_dte = flow[flow["dte"] == 0].nlargest(top_n, ["unusual_score", "premium", "vol_oi"])
clusters_only = flow[flow["cluster_flag"] == "CLUSTER"].nlargest(top_n, ["cluster_premium", "unusual_score", "premium"])

bullish_prem_by_symbol = flow.loc[flow["direction"] == "BULLISH"].groupby("symbol")["premium"].sum()
bearish_prem_by_symbol = flow.loc[flow["direction"] == "BEARISH"].groupby("symbol")["premium"].sum()

ticker_summary = (
    flow.groupby("symbol", as_index=False)
    .agg(
        rows=("symbol", "size"),
        total_premium=("premium", "sum"),
        avg_score=("unusual_score", "mean"),
        max_score=("unusual_score", "max"),
        avg_dte=("dte", "mean"),
    )
)

ticker_summary["bullish_premium"] = ticker_summary["symbol"].map(bullish_prem_by_symbol).fillna(0.0)
ticker_summary["bearish_premium"] = ticker_summary["symbol"].map(bearish_prem_by_symbol).fillna(0.0)
ticker_summary["net_premium"] = ticker_summary["bullish_premium"] - ticker_summary["bearish_premium"]
ticker_summary["flow_bias"] = np.where(
    ticker_summary["net_premium"] > 0, "BULLISH",
    np.where(ticker_summary["net_premium"] < 0, "BEARISH", "BALANCED")
)
ticker_summary = ticker_summary.sort_values(["total_premium", "max_score", "rows"], ascending=[False, False, False]).reset_index(drop=True)

# -----------------------------------------------------------------------------
# Commentary
# -----------------------------------------------------------------------------
st.subheader("Tape Commentary")
st.write(summarize_tape(overall))

leaderboard_cols = st.columns([1.2, 1.2, 1.1, 1.1])
top_contract = f"{overall.iloc[0]['symbol']} {overall.iloc[0]['expiry']} {overall.iloc[0]['type']} {overall.iloc[0]['strike']:.0f}"
leaderboard_cols[0].metric("Top ticker by premium", str(ticker_summary.iloc[0]["symbol"]))
leaderboard_cols[1].metric("Top contract", top_contract)
leaderboard_cols[2].metric("0DTE rows", f"{len(zero_dte):,}")
leaderboard_cols[3].metric("Cluster flags", f"{int((flow['cluster_flag'] == 'CLUSTER').sum()):,}")

# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    direction_dte = (
        flow.assign(dte_bucket=flow["dte"].map(dte_bucket))
        .groupby(["dte_bucket", "direction"], as_index=False)["premium"]
        .sum()
    )
    dte_order = ["0DTE", "1-7D", "8-30D", "31-60D", "61-120D", "120D+"]
    direction_dte["dte_bucket"] = pd.Categorical(direction_dte["dte_bucket"], categories=dte_order, ordered=True)
    direction_dte = direction_dte.sort_values("dte_bucket")

    fig_dte = px.bar(
        direction_dte,
        x="dte_bucket",
        y="premium",
        color="direction",
        barmode="group",
        title="Premium by Direction and DTE Bucket"
    )
    fig_dte.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig_dte, use_container_width=True)

with right:
    bubble = overall.copy().head(300)
    bubble["label"] = bubble["symbol"] + " " + bubble["type"] + " " + bubble["expiry"]
    bubble["premium_log"] = np.log10(np.maximum(bubble["premium"], 1.0))

    fig_bubble = px.scatter(
        bubble,
        x="vol_oi",
        y="premium_log",
        size="unusual_score",
        color="direction",
        hover_name="label",
        hover_data={
            "strike": True,
            "dte": True,
            "side": True,
            "pct_otm": ":.1f",
            "spread_pct": ":.1f",
            "unusual_score": ":.1f",
            "vol_oi": ":.2f",
            "premium": ":,.0f",
            "premium_log": False,
        },
        title="Flow Map: log10(Premium) vs Volume/OI"
    )
    fig_bubble.update_yaxes(title_text="log10(Premium)")
    fig_bubble.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig_bubble, use_container_width=True)

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Top Flow", "Bullish", "Bearish", "0DTE", "Clusters", "By Ticker", "Diagnostics"]
)

with tab1:
    render_top_flow_dashboard(overall)

with tab2:
    st.dataframe(prep_display(bullish), use_container_width=True, hide_index=True)

with tab3:
    st.dataframe(prep_display(bearish), use_container_width=True, hide_index=True)

with tab4:
    if zero_dte.empty:
        st.info("No 0DTE anomalies passed the current filters.")
    else:
        st.dataframe(prep_display(zero_dte), use_container_width=True, hide_index=True)

with tab5:
    if clusters_only.empty:
        st.info("No cluster anomalies passed the current filters.")
    else:
        cluster_detail = (
            clusters_only.groupby(["symbol", "expiry", "type", "direction", "strike_bucket"], as_index=False)
            .agg(
                cluster_premium=("premium", "sum"),
                cluster_volume=("volume", "sum"),
                cluster_contracts=("contract_symbol", "count"),
                cluster_score=("unusual_score", "mean"),
            )
            .sort_values(["cluster_premium", "cluster_score"], ascending=[False, False])
            .head(50)
        )

        fig_cluster_detail = px.scatter(
            cluster_detail,
            x="cluster_contracts",
            y="cluster_premium",
            size="cluster_score",
            color="direction",
            hover_data={
                "symbol": True,
                "expiry": True,
                "type": True,
                "strike_bucket": True,
                "cluster_volume": ":,.0f",
                "cluster_premium": ":,.0f",
                "cluster_score": ":.1f",
            },
            title="Cluster Detail Map"
        )
        fig_cluster_detail.update_layout(height=520, margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(fig_cluster_detail, use_container_width=True)

with tab6:
    display_ticker = ticker_summary.copy()
    display_ticker["Total Premium"] = display_ticker["total_premium"].map(human_money)
    display_ticker["Bullish Premium"] = display_ticker["bullish_premium"].map(human_money)
    display_ticker["Bearish Premium"] = display_ticker["bearish_premium"].map(human_money)
    display_ticker["Net Premium"] = display_ticker["net_premium"].map(human_money)
    display_ticker["Avg Score"] = display_ticker["avg_score"].map(lambda x: f"{x:.1f}")
    display_ticker["Max Score"] = display_ticker["max_score"].map(lambda x: f"{x:.1f}")
    display_ticker["Avg DTE"] = display_ticker["avg_dte"].map(lambda x: f"{x:.1f}")

    left_ticker, right_ticker = st.columns([1.15, 1.0])

    with left_ticker:
        fig_ticker = px.bar(
            ticker_summary.head(25).sort_values("total_premium", ascending=True),
            x="total_premium",
            y="symbol",
            color="flow_bias",
            orientation="h",
            hover_data={
                "rows": True,
                "bullish_premium": ":,.0f",
                "bearish_premium": ":,.0f",
                "avg_score": ":.1f",
                "max_score": ":.1f",
            },
            title="Top Tickers by Total Premium"
        )
        fig_ticker.update_layout(height=760, margin=dict(l=20, r=20, t=55, b=20), yaxis_title="")
        st.plotly_chart(fig_ticker, use_container_width=True)

    with right_ticker:
        st.dataframe(
            display_ticker[[
                "symbol", "rows", "Total Premium", "Bullish Premium", "Bearish Premium",
                "Net Premium", "flow_bias", "Avg Score", "Max Score", "Avg DTE"
            ]].head(100),
            use_container_width=True,
            hide_index=True
        )

    ticker_pick = st.selectbox("Inspect ticker", options=ticker_summary["symbol"].tolist(), index=0)
    ticker_rows = flow[flow["symbol"] == ticker_pick].nlargest(200, ["unusual_score", "premium", "vol_oi"])
    st.dataframe(prep_display(ticker_rows), use_container_width=True, hide_index=True)

with tab7:
    st.subheader("Diagnostics")
    if filtered_snapshot is not None and isinstance(filtered_snapshot, pd.DataFrame) and not filtered_snapshot.empty:
        st.caption(f"Underlying pre-filters reduced the universe from {len(symbols_all):,} names to {len(filtered_snapshot):,} names before option-chain scanning.")
    if not diags.empty:
        st.dataframe(diags.sort_values(["errors", "contracts_seen"], ascending=[False, False]), use_container_width=True, hide_index=True)
    else:
        st.info("No diagnostic frame loaded yet. Run the live scan first.")
