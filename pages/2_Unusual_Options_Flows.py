# streamlit_app.py
# ADFM | Options Flow Trend Monitor
#
# What Yahoo can provide: option chains, volume, open interest, bid, ask, last price, IV, expiries.
# What Yahoo cannot provide reliably: institutional trade prints, sweep/block tags, buyer/seller IDs,
# true time-and-sales, opening vs closing trade classification, or real-time exchange-grade flow.
#
# The right public-data approximation is therefore:
# 1) scan option chains for unusual contracts,
# 2) classify likely direction using bid/ask/last-price heuristics,
# 3) save qualifying rows into a local database,
# 4) build rolling ticker trend scores from the saved database.

from __future__ import annotations

import json
import math
import pickle
import random
import sqlite3
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(page_title="ADFM Options Flow Trend Monitor", layout="wide")

APP_TITLE = "ADFM | Options Flow Trend Monitor"
APP_SUBTITLE = (
    "Yahoo Finance option-chain scanner with a persistent unusual-flow database, "
    "rolling bull/bear scores, and ticker-level trend boards."
)

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.0rem; padding-bottom: 1.8rem; max-width: 1680px;}
    h1, h2, h3 {font-weight: 650; letter-spacing: 0.1px;}
    .stMetric {
        background: #fafafa;
        border: 1px solid #ececec;
        border-radius: 14px;
        padding: 10px 14px;
    }
    .stDataFrame, .stPlotlyChart {border-radius: 14px;}
    div[data-testid="stSidebarContent"] {padding-top: 0.6rem;}
    .small-note {color: #6b7280; font-size: 0.88rem; line-height: 1.4;}
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

# =============================================================================
# Constants and paths
# =============================================================================
CACHE_DIR = Path(".uw_cache")
CACHE_DIR.mkdir(exist_ok=True)

FLOW_DB = CACHE_DIR / "adfm_options_flow.sqlite"
UNIVERSE_META_CACHE = CACHE_DIR / "ndx100_meta.pkl"
LAST_GOOD_SNAPSHOT = CACHE_DIR / "last_good_underlying_snapshot.pkl"
LAST_GOOD_SCAN = CACHE_DIR / "last_good_live_scan.pkl"
LAST_GOOD_DIAGS = CACHE_DIR / "last_good_live_diags.pkl"
LAST_GOOD_META = CACHE_DIR / "last_good_live_meta.json"

NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

DEFAULT_MIN_PREMIUM = 50_000.0
DEFAULT_MIN_VOLUME = 100
DEFAULT_MIN_VOL_OI = 1.5
DEFAULT_MAX_SPREAD_PCT = 35.0
DEFAULT_TOP_N = 150
DEFAULT_MAX_WORKERS = 4
DEFAULT_DTE_MIN = 0
DEFAULT_DTE_MAX = 90
DEFAULT_MAX_EXPIRIES_PER_SYMBOL = 8
DEFAULT_MIN_UNUSUAL_SCORE = 55.0
DEFAULT_MIN_UNDERLYING_DOLLAR_VOL_MM = 100.0
DEFAULT_BATCH_SIZE = 25
DEFAULT_MIN_PRICE = 10.0
DEFAULT_MIN_20D_VOL = 1_000_000

RETRY_ATTEMPTS = 3
RETRY_SLEEP_BASE = 0.85
CHAIN_JITTER_MAX = 0.35

DEFAULT_EXTRA_SYMBOLS = sorted([
    "SPY", "QQQ", "IWM", "DIA", "TLT", "HYG", "LQD", "GLD", "SLV", "USO", "XLE", "XLF",
    "XLK", "SMH", "SOXX", "KRE", "XBI", "ARKK", "FXI", "EEM", "EWZ", "TSM", "VST", "AXON",
    "HOOD", "COIN", "MSTR", "CAR", "NKTR", "FICO", "KKR", "TPL", "AMD", "NVDA", "META", "MSFT",
])

NASDAQ100_FALLBACK_SYMBOLS = sorted([
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN", "AMZN", "ANSS", "APP", "ARM",
    "ASML", "AVGO", "AXON", "AZN", "BIIB", "BKNG", "CDNS", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD",
    "CSCO", "CSGP", "CSX", "CTAS", "CTSH", "DASH", "DDOG", "DXCM", "EA", "EXC", "FANG", "FAST", "FTNT",
    "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "INTC", "INTU", "ISRG", "KDP", "KHC", "KLAC",
    "LIN", "LRCX", "LULU", "MAR", "MCHP", "MDLZ", "MELI", "META", "MNST", "MRVL", "MSFT", "MU", "NFLX",
    "NVDA", "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP", "PLTR", "PYPL", "QCOM",
    "REGN", "ROP", "ROST", "SBUX", "SNPS", "TEAM", "TMUS", "TSLA", "TTD", "TTWO", "TXN", "VRTX", "WBD",
    "WDAY", "XEL", "ZS"
])

# =============================================================================
# Formatting helpers
# =============================================================================
def human_money(x: float) -> str:
    x = 0.0 if pd.isna(x) else float(x)
    ax = abs(x)
    if ax >= 1_000_000_000:
        return f"${x / 1_000_000_000:.2f}B"
    if ax >= 1_000_000:
        return f"${x / 1_000_000:.2f}M"
    if ax >= 1_000:
        return f"${x / 1_000:.1f}K"
    return f"${x:,.0f}"


def human_num(x: float) -> str:
    x = 0.0 if pd.isna(x) else float(x)
    ax = abs(x)
    if ax >= 1_000_000_000:
        return f"{x / 1_000_000_000:.2f}B"
    if ax >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    if ax >= 1_000:
        return f"{x / 1_000:.1f}K"
    return f"{x:,.0f}"


def safe_float(x, default=np.nan) -> float:
    try:
        if x is None or pd.isna(x):
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


def normalize_symbol(sym: str) -> str:
    sym = str(sym).strip().upper()
    return sym.replace(".", "-")


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


def save_pickle(obj, path: Path) -> None:
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


def save_json(obj: dict, path: Path) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def with_retry_sleep(attempt: int) -> None:
    time.sleep(RETRY_SLEEP_BASE * attempt + random.uniform(0.0, CHAIN_JITTER_MAX))

# =============================================================================
# Local database
# =============================================================================
def init_db() -> None:
    with sqlite3.connect(FLOW_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS flow_events (
                scan_date TEXT NOT NULL,
                scan_ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                contract_symbol TEXT NOT NULL,
                type TEXT,
                expiry TEXT,
                dte INTEGER,
                spot REAL,
                strike REAL,
                moneyness TEXT,
                pct_otm REAL,
                side TEXT,
                direction TEXT,
                bid REAL,
                ask REAL,
                mid REAL,
                fill_est REAL,
                volume REAL,
                open_interest REAL,
                vol_oi REAL,
                premium REAL,
                spread_pct REAL,
                iv REAL,
                unusual_score REAL,
                avg_dollar_vol REAL,
                cluster_flag TEXT,
                cluster_contracts REAL,
                cluster_premium REAL,
                cluster_volume REAL,
                cluster_score REAL,
                PRIMARY KEY (scan_date, symbol, contract_symbol)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flow_symbol_date ON flow_events(symbol, scan_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flow_date ON flow_events(scan_date)")
        conn.commit()


def save_flow_to_db(flow: pd.DataFrame, scan_ts: Optional[pd.Timestamp] = None) -> int:
    if flow is None or flow.empty:
        return 0

    init_db()
    scan_ts = scan_ts or pd.Timestamp.now()
    scan_date = scan_ts.date().isoformat()
    scan_ts_str = scan_ts.strftime("%Y-%m-%d %H:%M:%S")

    out = ensure_flow_schema(build_cluster_flags(flow.copy()))
    out["scan_date"] = scan_date
    out["scan_ts"] = scan_ts_str

    cols = [
        "scan_date", "scan_ts", "symbol", "contract_symbol", "type", "expiry", "dte", "spot", "strike",
        "moneyness", "pct_otm", "side", "direction", "bid", "ask", "mid", "fill_est", "volume",
        "open_interest", "vol_oi", "premium", "spread_pct", "iv", "unusual_score", "avg_dollar_vol",
        "cluster_flag", "cluster_contracts", "cluster_premium", "cluster_volume", "cluster_score"
    ]

    for col in cols:
        if col not in out.columns:
            out[col] = np.nan if col not in ["cluster_flag"] else ""

    out = out[cols].copy()

    numeric_cols = [
        "dte", "spot", "strike", "pct_otm", "bid", "ask", "mid", "fill_est", "volume", "open_interest",
        "vol_oi", "premium", "spread_pct", "iv", "unusual_score", "avg_dollar_vol", "cluster_contracts",
        "cluster_premium", "cluster_volume", "cluster_score"
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    inserted = 0
    with sqlite3.connect(FLOW_DB) as conn:
        placeholders = ",".join(["?"] * len(cols))
        update_cols = [c for c in cols if c not in ["scan_date", "symbol", "contract_symbol"]]
        update_clause = ", ".join([f"{c}=excluded.{c}" for c in update_cols])
        sql = f"""
            INSERT INTO flow_events ({','.join(cols)})
            VALUES ({placeholders})
            ON CONFLICT(scan_date, symbol, contract_symbol)
            DO UPDATE SET {update_clause}
        """
        records = out.where(pd.notna(out), None).values.tolist()
        conn.executemany(sql, records)
        conn.commit()
        inserted = len(records)

    return inserted


def load_flow_db(days: int = 90) -> pd.DataFrame:
    init_db()
    start_date = (date.today() - timedelta(days=int(days))).isoformat()
    with sqlite3.connect(FLOW_DB) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM flow_events WHERE scan_date >= ? ORDER BY scan_date DESC, premium DESC",
            conn,
            params=(start_date,),
        )
    if df.empty:
        return df
    df["scan_date"] = pd.to_datetime(df["scan_date"]).dt.date
    df["scan_ts"] = pd.to_datetime(df["scan_ts"], errors="coerce")
    return ensure_flow_schema(df)


def clear_flow_db() -> None:
    init_db()
    with sqlite3.connect(FLOW_DB) as conn:
        conn.execute("DELETE FROM flow_events")
        conn.commit()

# =============================================================================
# Universe loader
# =============================================================================
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

        candidate = candidate[["symbol", "company"]].copy()
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
            return cached.drop_duplicates("symbol").reset_index(drop=True)

        df = pd.DataFrame({
            "symbol": [normalize_symbol(x) for x in NASDAQ100_FALLBACK_SYMBOLS],
            "company": [normalize_symbol(x) for x in NASDAQ100_FALLBACK_SYMBOLS],
        })
        return df.drop_duplicates("symbol").reset_index(drop=True)


def parse_custom_symbols(raw: str) -> List[str]:
    if not raw:
        return []
    parts = raw.replace("\n", ",").replace(";", ",").split(",")
    return sorted({normalize_symbol(x) for x in parts if str(x).strip()})

# =============================================================================
# Spot snapshot
# =============================================================================
def _extract_snapshot_from_download(data: pd.DataFrame, symbols: List[str]) -> List[dict]:
    rows: List[dict] = []
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
            avg_20d_vol = float(vol.tail(20).mean()) if len(vol) else np.nan
            day_volume = float(vol.iloc[-1]) if len(vol) else np.nan
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


@st.cache_data(ttl=20 * 60, show_spinner=False)
def fetch_spot_snapshot(symbols: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> Tuple[pd.DataFrame, dict]:
    if not symbols:
        return pd.DataFrame(columns=["symbol", "spot", "prev_close", "avg_20d_vol", "day_volume", "avg_dollar_vol"]), {
            "batches": 0,
            "rows": 0,
            "failures": 0,
        }

    all_rows: List[dict] = []
    failures = 0
    batches = 0

    for batch in chunked(symbols, batch_size):
        batches += 1
        batch_rows: List[dict] = []
        batch_success = False

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                data = yf.download(
                    tickers=batch,
                    period="3mo",
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    group_by="ticker",
                    ignore_tz=True,
                )
                batch_rows = _extract_snapshot_from_download(data, batch)
                if batch_rows:
                    batch_success = True
                    break
            except Exception:
                pass
            with_retry_sleep(attempt)

        if not batch_success:
            failures += 1
        all_rows.extend(batch_rows)

    if not all_rows:
        cached = load_pickle(LAST_GOOD_SNAPSHOT)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached.copy(), {
                "batches": batches,
                "rows": len(cached),
                "failures": failures,
                "used_cached_snapshot": True,
            }
        return pd.DataFrame(columns=["symbol", "spot", "prev_close", "avg_20d_vol", "day_volume", "avg_dollar_vol"]), {
            "batches": batches,
            "rows": 0,
            "failures": failures,
            "used_cached_snapshot": False,
        }

    out = pd.DataFrame(all_rows).drop_duplicates("symbol", keep="last").reset_index(drop=True)
    save_pickle(out, LAST_GOOD_SNAPSHOT)
    return out, {
        "batches": batches,
        "rows": len(out),
        "failures": failures,
        "used_cached_snapshot": False,
    }


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

# =============================================================================
# Option classification and scoring
# =============================================================================
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
    premium_score = min(math.log10(max(premium, 0) + 1.0) / 6.8, 1.0)
    abs_size_score = min(max(volume, 0) / 3000.0, 1.0)
    voi_score = min((max(volume, 0) / max(open_interest, 1.0)) / 6.0, 1.0)
    otm_score = min(max(pct_otm, 0) / 0.12, 1.0)
    dte_score = 1.0 if dte <= 30 else max(0.25, 1.0 - ((dte - 30) / 365.0))
    spread_penalty = min(max(spread_pct, 0) / 80.0, 1.0)
    side_bonus = 0.06 if side in ("ASK", "BID") else 0.025 if side == "MID" else 0.0

    if pd.isna(avg_dollar_vol) or avg_dollar_vol <= 0:
        liquidity_score = 0.4
    else:
        liquidity_score = min(math.log10(avg_dollar_vol + 1.0) / 9.0, 1.0)

    raw = (
        0.35 * premium_score
        + 0.20 * voi_score
        + 0.14 * abs_size_score
        + 0.08 * otm_score
        + 0.08 * dte_score
        + 0.07 * liquidity_score
        + side_bonus
        - 0.16 * spread_penalty
    )
    return float(np.clip(raw, 0.0, 1.0) * 100.0)

# =============================================================================
# Yahoo option helpers
# =============================================================================
def get_ticker_with_retries(symbol: str) -> Optional[yf.Ticker]:
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return yf.Ticker(symbol)
        except Exception:
            with_retry_sleep(attempt)
    return None


def get_options_dates(tkr: yf.Ticker) -> Tuple[List[str], Optional[str]]:
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            expiries = list(tkr.options or [])
            if expiries:
                return expiries, None
            return [], "no_listed_options"
        except Exception:
            with_retry_sleep(attempt)
    return [], "options_load_failed"


def get_option_chain(tkr: yf.Ticker, expiry: str):
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return tkr.option_chain(expiry), None
        except Exception:
            with_retry_sleep(attempt)
    return None, "chain_load_failed"

# =============================================================================
# One-symbol scanner
# =============================================================================
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
        "detail": "",
    }
    rows: List[dict] = []

    tkr = get_ticker_with_retries(symbol)
    if tkr is None:
        diagnostics["errors"] = 1
        diagnostics["reason"] = "ticker_init_failed"
        return rows, diagnostics

    expiries, expiry_err = get_options_dates(tkr)
    if expiry_err is not None:
        diagnostics["errors"] = 1 if expiry_err == "options_load_failed" else 0
        diagnostics["reason"] = expiry_err
        return rows, diagnostics

    spot = safe_float(spot_map.get(symbol, np.nan), np.nan)
    avg_dollar_vol = safe_float(adv_map.get(symbol, np.nan), np.nan)

    if pd.isna(spot) or spot <= 0:
        diagnostics["reason"] = "spot_missing"
        return rows, diagnostics

    today_utc = pd.Timestamp.now(tz="UTC").date()
    parsed_expiries: List[Tuple[str, int]] = []
    for exp in expiries:
        try:
            exp_date = pd.Timestamp(exp).date()
            dte = (exp_date - today_utc).days
            if dte_min <= dte <= dte_max:
                parsed_expiries.append((exp, dte))
        except Exception:
            continue

    parsed_expiries = sorted(parsed_expiries, key=lambda x: x[1])[:max_expiries_per_symbol]
    if not parsed_expiries:
        diagnostics["reason"] = "no_expiry_in_range"
        return rows, diagnostics

    time.sleep(random.uniform(0.02, CHAIN_JITTER_MAX))

    for exp, dte in parsed_expiries:
        diagnostics["expiries_seen"] += 1
        chain, chain_err = get_option_chain(tkr, exp)

        if chain_err is not None or chain is None:
            diagnostics["errors"] += 1
            if diagnostics["reason"] == "":
                diagnostics["reason"] = chain_err
            continue

        chain_parts = [("CALL", getattr(chain, "calls", pd.DataFrame())), ("PUT", getattr(chain, "puts", pd.DataFrame()))]

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
                np.where(local["ask"] > 0, local["ask"], np.where(local["bid"] > 0, local["bid"], 0.0)),
            )
            fills = np.where(
                local["lastPrice"] > 0,
                local["lastPrice"],
                np.where(mids > 0, mids, np.where(local["ask"] > 0, local["ask"], local["bid"])),
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
                np.nan,
            )
            local = local[(local["spread_pct"].isna()) | (local["spread_pct"] <= max_spread_pct)].copy()
            if local.empty:
                continue

            local["side"] = [classify_side(f, b, a) for f, b, a in zip(local["fill_est"], local["bid"], local["ask"])]
            local["direction"] = [classify_direction(option_type, s) for s in local["side"]]

            if option_type == "CALL":
                local["pct_otm_dec"] = np.maximum((local["strike"] - spot) / spot, 0.0)
                local["moneyness"] = np.where(local["strike"] > spot, "OTM", np.where(local["strike"] < spot, "ITM", "ATM"))
            else:
                local["pct_otm_dec"] = np.maximum((spot - local["strike"]) / spot, 0.0)
                local["moneyness"] = np.where(local["strike"] < spot, "OTM", np.where(local["strike"] > spot, "ITM", "ATM"))

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
                    local["side"],
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
                "spread_pct", "impliedVolatility", "unusual_score", "avg_dollar_vol",
            ]].rename(columns={
                "contractSymbol": "contract_symbol",
                "openInterest": "open_interest",
                "impliedVolatility": "iv",
            }).to_dict("records"))

            diagnostics["contracts_kept"] += len(local)

    if diagnostics["contracts_kept"] == 0 and diagnostics["reason"] == "":
        diagnostics["reason"] = "empty_chains" if diagnostics["contracts_seen"] == 0 else "all_filtered_out"

    return rows, diagnostics

# =============================================================================
# Clusters and schema
# =============================================================================
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


def build_cluster_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    out = df.copy()
    stale_cols = [
        "strike_bucket", "cluster_key", "cluster_contracts", "cluster_premium",
        "cluster_volume", "cluster_score", "cluster_flag",
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
    out["cluster_flag"] = np.where((out["cluster_contracts"] >= 3) & (out["cluster_premium"] >= 500_000), "CLUSTER", "")
    return out

# =============================================================================
# Full scan
# =============================================================================
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

    if total == 0:
        progress_bar.empty()
        status.empty()
        return pd.DataFrame(), pd.DataFrame()

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
            except Exception as e:
                all_diags.append({
                    "symbol": sym,
                    "expiries_seen": 0,
                    "contracts_seen": 0,
                    "contracts_kept": 0,
                    "errors": 1,
                    "reason": "worker_exception",
                    "detail": str(e),
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
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    return flow, diags

# =============================================================================
# Trend engine
# =============================================================================
def build_trend_board(db: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    if db is None or db.empty:
        return pd.DataFrame()

    start = date.today() - timedelta(days=int(lookback_days))
    sub = db[pd.to_datetime(db["scan_date"]).dt.date >= start].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["bullish_premium"] = np.where(sub["direction"] == "BULLISH", sub["premium"], 0.0)
    sub["bearish_premium"] = np.where(sub["direction"] == "BEARISH", sub["premium"], 0.0)
    sub["bullish_rows"] = np.where(sub["direction"] == "BULLISH", 1, 0)
    sub["bearish_rows"] = np.where(sub["direction"] == "BEARISH", 1, 0)
    sub["cluster_int"] = np.where(sub["cluster_flag"].astype(str).str.upper() == "CLUSTER", 1, 0)

    trend = (
        sub.groupby("symbol", as_index=False)
        .agg(
            unusual_trades=("contract_symbol", "count"),
            active_days=("scan_date", lambda x: pd.Series(x).nunique()),
            total_premium=("premium", "sum"),
            bullish_premium=("bullish_premium", "sum"),
            bearish_premium=("bearish_premium", "sum"),
            bullish_trades=("bullish_rows", "sum"),
            bearish_trades=("bearish_rows", "sum"),
            avg_score=("unusual_score", "mean"),
            max_score=("unusual_score", "max"),
            avg_dte=("dte", "mean"),
            largest_print=("premium", "max"),
            clusters=("cluster_int", "sum"),
        )
    )

    trend["net_premium"] = trend["bullish_premium"] - trend["bearish_premium"]
    trend["net_trades"] = trend["bullish_trades"] - trend["bearish_trades"]
    trend["bull_bear_score"] = np.where(
        trend["total_premium"] > 0,
        100.0 * trend["net_premium"] / trend["total_premium"],
        0.0,
    )
    trend["persistence"] = trend["active_days"] / max(lookback_days, 1)

    trend["tape"] = np.select(
        [
            trend["bull_bear_score"] >= 35,
            trend["bull_bear_score"] <= -35,
        ],
        ["BULLISH", "BEARISH"],
        default="MIXED",
    )

    trend["conviction_score"] = (
        trend["avg_score"].fillna(0) * 0.40
        + np.clip(np.log10(trend["total_premium"].fillna(0) + 1) / 7.0 * 100, 0, 100) * 0.30
        + np.clip(trend["unusual_trades"].fillna(0) / 20 * 100, 0, 100) * 0.20
        + np.clip(trend["active_days"].fillna(0) / max(min(lookback_days, 20), 1) * 100, 0, 100) * 0.10
    )

    return trend.sort_values(["conviction_score", "total_premium"], ascending=[False, False]).reset_index(drop=True)


def build_multi_window_scores(db: pd.DataFrame) -> pd.DataFrame:
    windows = [1, 5, 20, 60]
    boards = []
    for w in windows:
        b = build_trend_board(db, w)
        if b.empty:
            continue
        keep = b[["symbol", "unusual_trades", "total_premium", "net_premium", "bull_bear_score", "conviction_score", "tape"]].copy()
        keep = keep.rename(columns={
            "unusual_trades": f"{w}D Trades",
            "total_premium": f"{w}D Premium",
            "net_premium": f"{w}D Net Premium",
            "bull_bear_score": f"{w}D B/B Score",
            "conviction_score": f"{w}D Conviction",
            "tape": f"{w}D Tape",
        })
        boards.append(keep)

    if not boards:
        return pd.DataFrame()

    out = boards[0]
    for b in boards[1:]:
        out = out.merge(b, on="symbol", how="outer")

    sort_col = "20D Conviction" if "20D Conviction" in out.columns else [c for c in out.columns if c.endswith("Conviction")][0]
    return out.sort_values(sort_col, ascending=False).reset_index(drop=True)

# =============================================================================
# Display helpers
# =============================================================================
def prep_flow_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = ensure_flow_schema(df).copy()
    out["Premium"] = out["premium"].map(human_money)
    out["Vol"] = out["volume"].map(human_num)
    out["OI"] = out["open_interest"].map(human_num)
    out["Vol/OI"] = out["vol_oi"].map(lambda x: f"{safe_float(x, 0):.2f}x")
    out["Fill"] = out["fill_est"].map(lambda x: f"{safe_float(x, 0):.2f}")
    out["Bid"] = out["bid"].map(lambda x: f"{safe_float(x, 0):.2f}")
    out["Ask"] = out["ask"].map(lambda x: f"{safe_float(x, 0):.2f}")
    out["Spread %"] = out["spread_pct"].map(lambda x: "" if pd.isna(x) else f"{x:.1f}%")
    out["IV"] = out["iv"].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    out["OTM %"] = out["pct_otm"].map(lambda x: "" if pd.isna(x) else f"{x:.1f}%")
    out["Score"] = out["unusual_score"].map(lambda x: f"{safe_float(x, 0):.1f}")
    if "scan_date" in out.columns:
        out["Date"] = out["scan_date"].astype(str)
    else:
        out["Date"] = ""

    keep = [
        "Date", "symbol", "type", "expiry", "dte", "direction", "side", "strike", "spot",
        "moneyness", "OTM %", "Fill", "Bid", "Ask", "Spread %", "Vol", "OI", "Vol/OI",
        "Premium", "IV", "Score", "cluster_flag", "contract_symbol",
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep].copy()


def prep_trend_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    money_cols = [c for c in out.columns if "Premium" in c or c in ["total_premium", "bullish_premium", "bearish_premium", "net_premium", "largest_print"]]
    for col in money_cols:
        if col in out.columns:
            out[col] = out[col].map(human_money)
    score_cols = [c for c in out.columns if "Score" in c or "Conviction" in c or c in ["avg_score", "max_score", "bull_bear_score"]]
    for col in score_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    if "avg_dte" in out.columns:
        out["avg_dte"] = out["avg_dte"].map(lambda x: "" if pd.isna(x) else f"{x:.0f}")
    return out


def render_top_metrics(db: pd.DataFrame, live_flow: pd.DataFrame, scan_meta: dict) -> None:
    db_rows = 0 if db is None or db.empty else len(db)
    db_symbols = 0 if db is None or db.empty else db["symbol"].nunique()
    live_rows = 0 if live_flow is None or live_flow.empty else len(live_flow)
    live_premium = 0.0 if live_flow is None or live_flow.empty else float(live_flow["premium"].sum())
    live_bull = 0.0 if live_flow is None or live_flow.empty else float(live_flow.loc[live_flow["direction"] == "BULLISH", "premium"].sum())
    live_bear = 0.0 if live_flow is None or live_flow.empty else float(live_flow.loc[live_flow["direction"] == "BEARISH", "premium"].sum())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Live rows", f"{live_rows:,}")
    c2.metric("Live premium", human_money(live_premium))
    c3.metric("Live bull", human_money(live_bull))
    c4.metric("Live bear", human_money(live_bear))
    c5.metric("DB rows", f"{db_rows:,}")
    c6.metric("DB tickers", f"{db_symbols:,}")

    if scan_meta:
        meta = f"Mode: {scan_meta.get('mode', 'Unknown')}"
        if scan_meta.get("saved_at"):
            meta += f" | Snapshot: {scan_meta.get('saved_at')}"
        if scan_meta.get("symbols_scanned") is not None:
            meta += f" | Symbols scanned: {safe_int(scan_meta.get('symbols_scanned')):,}"
        if scan_meta.get("raw_contracts") is not None:
            meta += f" | Raw contracts: {safe_int(scan_meta.get('raw_contracts')):,}"
        if scan_meta.get("success_ratio") is not None:
            meta += f" | Success ratio: {safe_float(scan_meta.get('success_ratio'), 0):.0%}"
        if scan_meta.get("elapsed") is not None:
            meta += f" | Elapsed: {safe_float(scan_meta.get('elapsed'), 0):.1f}s"
        st.caption(meta)


def render_trend_bar(trend: pd.DataFrame, lookback_days: int, top_n: int = 30) -> None:
    if trend.empty:
        st.info("No saved trend data for this window yet. Run and save a scan first.")
        return

    top = trend.head(top_n).copy()
    fig = px.bar(
        top.sort_values("net_premium", ascending=True),
        x="net_premium",
        y="symbol",
        color="tape",
        orientation="h",
        hover_data={
            "unusual_trades": True,
            "active_days": True,
            "total_premium": ":,.0f",
            "bullish_premium": ":,.0f",
            "bearish_premium": ":,.0f",
            "bull_bear_score": ":.1f",
            "conviction_score": ":.1f",
        },
        title=f"Top Net Flow by Ticker | {lookback_days}D Window",
    )
    fig.update_layout(height=760, margin=dict(l=20, r=20, t=55, b=20), yaxis_title="", xaxis_title="Net bullish minus bearish premium")
    st.plotly_chart(fig, use_container_width=True)


def render_ticker_timeline(db: pd.DataFrame, ticker: str) -> None:
    if db.empty or not ticker:
        return

    sub = db[db["symbol"] == ticker].copy()
    if sub.empty:
        st.info("No saved rows for this ticker.")
        return

    sub["scan_date_dt"] = pd.to_datetime(sub["scan_date"])
    sub["bullish_premium"] = np.where(sub["direction"] == "BULLISH", sub["premium"], 0.0)
    sub["bearish_premium"] = np.where(sub["direction"] == "BEARISH", sub["premium"], 0.0)

    daily = (
        sub.groupby("scan_date_dt", as_index=False)
        .agg(
            bullish_premium=("bullish_premium", "sum"),
            bearish_premium=("bearish_premium", "sum"),
            rows=("contract_symbol", "count"),
        )
    )
    daily["net_premium"] = daily["bullish_premium"] - daily["bearish_premium"]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily["scan_date_dt"], y=daily["bullish_premium"], name="Bullish premium"))
    fig.add_trace(go.Bar(x=daily["scan_date_dt"], y=-daily["bearish_premium"], name="Bearish premium"))
    fig.add_trace(go.Scatter(x=daily["scan_date_dt"], y=daily["net_premium"], mode="lines+markers", name="Net premium"))
    fig.update_layout(
        title=f"{ticker} | Saved Unusual Flow by Day",
        barmode="relative",
        height=500,
        margin=dict(l=20, r=20, t=55, b=20),
        yaxis_title="Premium",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# Session state
# =============================================================================
def init_state() -> None:
    defaults = {
        "flow_raw": None,
        "diags_raw": None,
        "scan_meta": None,
        "scan_ran": False,
        "filtered_snapshot": None,
        "spot_snapshot_meta": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
init_db()

# =============================================================================
# Universe
# =============================================================================
universe = fetch_nasdaq100_from_wikipedia()
ndx_symbols = universe["symbol"].dropna().astype(str).map(normalize_symbol).unique().tolist()

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** A public-data approximation of a curated unusual-options-flow service.

        **What it does**
        - Scans Yahoo Finance option chains for unusual contracts.
        - Saves qualifying rows into a local SQLite database.
        - Builds ticker-level bull/bear trend scores over rolling windows.
        - Lets you drill down into repeated attacks on the same ticker.

        **Yahoo limitation**
        - This is chain-snapshot data, not exchange-grade time-and-sales.
        - Side and direction are estimates from last price versus bid/ask.
        """
    )

    st.subheader("Universe")
    universe_mode = st.selectbox(
        "Universe mode",
        ["Nasdaq 100 + ADFM extras", "Nasdaq 100 only", "Custom only"],
        index=0,
    )
    custom_raw = st.text_area(
        "Custom tickers",
        value="",
        placeholder="Example: SLV, GLD, VST, HOOD, CAR, NKTR",
        height=80,
    )
    custom_symbols = parse_custom_symbols(custom_raw)

    if universe_mode == "Nasdaq 100 only":
        symbols_all = sorted(set(ndx_symbols))
    elif universe_mode == "Custom only":
        symbols_all = sorted(set(custom_symbols))
    else:
        symbols_all = sorted(set(ndx_symbols + DEFAULT_EXTRA_SYMBOLS + custom_symbols))

    st.subheader("Underlying Pre-Filters")
    min_price = st.number_input("Minimum stock price ($)", min_value=0.0, value=float(DEFAULT_MIN_PRICE), step=5.0, format="%.0f")
    min_avg_20d_vol = st.number_input("Minimum 20D avg share volume", min_value=0, value=int(DEFAULT_MIN_20D_VOL), step=250_000)
    min_underlying_dollar_vol_mm = st.number_input(
        "Minimum underlying ADV ($mm)",
        min_value=0.0,
        value=float(DEFAULT_MIN_UNDERLYING_DOLLAR_VOL_MM),
        step=25.0,
        format="%.0f",
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
    max_workers = st.slider("Concurrent workers", min_value=1, max_value=10, value=DEFAULT_MAX_WORKERS, step=1)
    top_n = st.slider("Rows to display", min_value=25, max_value=500, value=DEFAULT_TOP_N, step=25)
    allow_cached_fallback = st.toggle("Use last-good cached scan when live scan is degraded", value=True)
    save_to_database = st.toggle("Save qualifying rows to local trend database", value=True)

    run_scan = st.button("Run Live Scan", use_container_width=True, type="primary")

    st.subheader("Database")
    db_days = st.slider("Database lookback loaded", min_value=5, max_value=365, value=90, step=5)
    if st.button("Clear local flow database", use_container_width=True):
        clear_flow_db()
        st.success("Local flow database cleared.")

# =============================================================================
# Cached startup load
# =============================================================================
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

# =============================================================================
# Run scan
# =============================================================================
if run_scan:
    scan_started = time.time()
    scan_ts = pd.Timestamp.now()

    if not symbols_all:
        st.error("No symbols selected.")
        st.stop()

    with st.spinner(f"Fetching spot data for {len(symbols_all):,} symbols..."):
        snapshot, spot_meta = fetch_spot_snapshot(symbols_all, batch_size=DEFAULT_BATCH_SIZE)

    st.session_state["spot_snapshot_meta"] = spot_meta

    filtered_snapshot = filter_underlyings(
        snapshot=snapshot,
        min_price=min_price,
        min_avg_dollar_vol_mm=min_underlying_dollar_vol_mm,
        min_avg_20d_vol=min_avg_20d_vol,
    )
    symbols_scanned_list = filtered_snapshot["symbol"].dropna().astype(str).tolist() if not filtered_snapshot.empty else []
    symbols_scanned_count = len(symbols_scanned_list)
    st.session_state["filtered_snapshot"] = filtered_snapshot.copy()

    if symbols_scanned_count == 0:
        flow_live = pd.DataFrame()
        diags_live = pd.DataFrame()
        success_ratio = None
        raw_contracts = 0
        empty_reason = "All underlyings failed the pre-scan filters."
    else:
        with st.spinner(f"Running Yahoo options-chain scan across {symbols_scanned_count:,} filtered symbols..."):
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
            non_error_rows = int((diags_live["errors"].fillna(0) == 0).sum()) if "errors" in diags_live.columns else 0
            success_ratio = non_error_rows / max(len(diags_live), 1)
        empty_reason = None

    degraded = (
        diags_live is None
        or diags_live.empty
        or (success_ratio is not None and success_ratio < 0.35)
        or raw_contracts < 250
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
                "empty_reason": "Live scan looked degraded, so cached data was loaded.",
            }
            st.session_state["scan_ran"] = True
            st.warning("Live scan looked degraded, so the app loaded the last-good cached snapshot.")
        else:
            st.session_state["flow_raw"] = ensure_flow_schema(flow_live)
            st.session_state["diags_raw"] = diags_live
            st.session_state["scan_meta"] = {
                "mode": "Live",
                "saved_at": scan_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed": round(time.time() - scan_started, 1),
                "success_ratio": success_ratio,
                "raw_contracts": raw_contracts,
                "symbols_requested": len(symbols_all),
                "symbols_scanned": symbols_scanned_count,
                "empty_reason": empty_reason or "Live scan returned weak chain coverage and no cached snapshot was strong enough to use.",
            }
            st.session_state["scan_ran"] = True
    else:
        flow_live = ensure_flow_schema(flow_live)
        st.session_state["flow_raw"] = flow_live
        st.session_state["diags_raw"] = diags_live
        st.session_state["scan_meta"] = {
            "mode": "Live",
            "saved_at": scan_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed": round(time.time() - scan_started, 1),
            "success_ratio": success_ratio,
            "raw_contracts": raw_contracts,
            "symbols_requested": len(symbols_all),
            "symbols_scanned": symbols_scanned_count,
            "empty_reason": empty_reason,
        }
        st.session_state["scan_ran"] = True

        if flow_live is not None and not flow_live.empty:
            save_pickle(flow_live, LAST_GOOD_SCAN)
            save_pickle(diags_live, LAST_GOOD_DIAGS)
            save_json(
                {
                    "saved_at": scan_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "rows": int(len(flow_live)),
                    "symbols_requested": int(len(symbols_all)),
                    "symbols_scanned": int(symbols_scanned_count),
                    "raw_contracts": raw_contracts,
                    "success_ratio": success_ratio,
                },
                LAST_GOOD_META,
            )

            if save_to_database:
                inserted = save_flow_to_db(flow_live, scan_ts=scan_ts)
                st.success(f"Saved {inserted:,} qualifying rows to the local trend database.")

# =============================================================================
# Main data frames
# =============================================================================
flow = st.session_state["flow_raw"] if st.session_state["flow_raw"] is not None else pd.DataFrame()
diags = st.session_state["diags_raw"] if st.session_state["diags_raw"] is not None else pd.DataFrame()
scan_meta = st.session_state["scan_meta"] or {}
filtered_snapshot = st.session_state["filtered_snapshot"]
spot_snapshot_meta = st.session_state["spot_snapshot_meta"] or {}

if flow is not None and not flow.empty:
    flow = ensure_flow_schema(build_cluster_flags(flow.copy()))

db = load_flow_db(days=db_days)
render_top_metrics(db, flow, scan_meta)

# =============================================================================
# Empty state
# =============================================================================
if not st.session_state["scan_ran"] and flow.empty and db.empty:
    st.markdown(
        """
        <div class="status-card">
        <b>No scan has been run yet and the local trend database is empty.</b><br>
        Click <b>Run Live Scan</b> in the sidebar. Qualifying rows can be saved into the local database, which is what powers the rolling trend board.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# Tabs
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Trend Board",
    "Today Odd Action",
    "Ticker Drilldown",
    "Live Scan Diagnostics",
    "Method Notes",
])

with tab1:
    st.subheader("Rolling Bull/Bear Trend Board")

    if db.empty:
        st.info("No saved database rows yet. Run a live scan with 'Save qualifying rows to local trend database' turned on.")
    else:
        lookback = st.selectbox("Trend window", [1, 5, 20, 60, 90], index=2, key="trend_window")
        trend = build_trend_board(db, lookback)
        multi = build_multi_window_scores(db)

        if trend.empty:
            st.info("No saved rows in this lookback window.")
        else:
            render_trend_bar(trend, lookback_days=lookback, top_n=30)

            display_cols = [
                "symbol", "tape", "unusual_trades", "active_days", "total_premium", "bullish_premium",
                "bearish_premium", "net_premium", "bull_bear_score", "conviction_score", "avg_score",
                "avg_dte", "largest_print", "clusters",
            ]
            display_cols = [c for c in display_cols if c in trend.columns]
            st.dataframe(prep_trend_display(trend[display_cols]).head(150), use_container_width=True, hide_index=True)

        st.subheader("Multi-Window Scoreboard")
        if multi.empty:
            st.info("No multi-window score available yet.")
        else:
            st.dataframe(prep_trend_display(multi).head(150), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Today Odd Action")

    if flow.empty:
        st.info("No live or cached scan rows loaded. Run a live scan first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(flow):,}")
        c2.metric("Premium", human_money(float(flow["premium"].sum())))
        c3.metric("Bullish premium", human_money(float(flow.loc[flow["direction"] == "BULLISH", "premium"].sum())))
        c4.metric("Bearish premium", human_money(float(flow.loc[flow["direction"] == "BEARISH", "premium"].sum())))

        chart = flow.head(30).copy()
        chart["label"] = (
            chart["symbol"].astype(str) + " | " +
            chart["type"].astype(str) + " | " +
            chart["expiry"].astype(str) + " | " +
            chart["strike"].round(0).astype(int).astype(str)
        )
        fig = px.bar(
            chart.sort_values(["premium", "unusual_score"], ascending=[True, True]),
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
            title="Top Live Unusual Flow",
        )
        fig.update_layout(height=780, margin=dict(l=20, r=20, t=55, b=20), yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(prep_flow_display(flow.head(top_n)), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Ticker Drilldown")

    if db.empty:
        st.info("No saved database rows yet.")
    else:
        symbols_in_db = sorted(db["symbol"].dropna().astype(str).unique().tolist())
        default_symbol = symbols_in_db[0] if symbols_in_db else ""
        ticker_pick = st.selectbox("Ticker", options=symbols_in_db, index=0 if symbols_in_db else None)

        if ticker_pick:
            sub = db[db["symbol"] == ticker_pick].copy().sort_values(["scan_date", "premium"], ascending=[False, False])
            render_ticker_timeline(db, ticker_pick)

            trend_5 = build_trend_board(db[db["symbol"] == ticker_pick], 5)
            trend_20 = build_trend_board(db[db["symbol"] == ticker_pick], 20)
            trend_60 = build_trend_board(db[db["symbol"] == ticker_pick], 60)

            cols = st.columns(3)
            for col, label, frame in zip(cols, ["5D", "20D", "60D"], [trend_5, trend_20, trend_60]):
                if frame.empty:
                    col.metric(f"{label} B/B Score", "N/A")
                else:
                    row = frame.iloc[0]
                    col.metric(f"{label} B/B Score", f"{row['bull_bear_score']:.1f}", delta=human_money(row["net_premium"]))

            st.dataframe(prep_flow_display(sub.head(300)), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Live Scan Diagnostics")

    if filtered_snapshot is not None and isinstance(filtered_snapshot, pd.DataFrame) and not filtered_snapshot.empty:
        st.caption(f"Underlying pre-filters reduced the universe from {len(symbols_all):,} names to {len(filtered_snapshot):,} names before option-chain scanning.")
        st.dataframe(filtered_snapshot[["symbol", "spot", "avg_20d_vol", "avg_dollar_vol"]], use_container_width=True, hide_index=True)

    if not diags.empty:
        st.dataframe(diags.sort_values(["errors", "contracts_seen"], ascending=[False, False]), use_container_width=True, hide_index=True)

        reason_counts = (
            diags["reason"]
            .fillna("unknown")
            .replace("", "blank")
            .value_counts()
            .rename_axis("reason")
            .reset_index(name="count")
        )
        fig_diag = px.bar(
            reason_counts.sort_values("count", ascending=True),
            x="count",
            y="reason",
            orientation="h",
            title="Reason Breakdown",
        )
        fig_diag.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20), yaxis_title="")
        st.plotly_chart(fig_diag, use_container_width=True)
    else:
        st.info("No diagnostic frame loaded yet. Run the live scan first.")

with tab5:
    st.subheader("Method Notes")
    st.markdown(
        """
        This is intentionally designed as a public-data approximation of a curated unusual-options-flow service. Yahoo Finance can provide option-chain snapshots, including bid, ask, last price, volume, open interest, implied volatility, strike, and expiry. That is enough to identify unusual contracts, estimate premium, calculate volume/open-interest anomalies, and build a persistent trend database.

        It cannot fully replicate institutional flow products because Yahoo does not give the full exchange-grade tape: no reliable sweep/block flags, no exact trade sequence, no opening-versus-closing classification, no real buyer/seller identity, and no true institutional tag. The app therefore treats direction as an estimate based on the relationship between last price and bid/ask, then scores persistence by ticker over time.

        """
    )

st.caption("© 2026 AD Fund Management LP")
