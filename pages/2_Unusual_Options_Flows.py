# streamlit_app.py
# ADFM | Options Flow Outlier Monitor
# Revised rebuild: designed to capture true outlier contracts rather than broad option-chain activity.
# Public Yahoo chain snapshots are filtered through premium, volume/OI, spread, DTE, direction confidence,
# OTM distance, and premium-versus-underlying-liquidity checks.

from __future__ import annotations

import hashlib
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
from typing import Dict, Iterable, List, Optional, Tuple

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
st.set_page_config(page_title="ADFM Options Flow Outlier Monitor", layout="wide")

APP_TITLE = "ADFM | Options Flow Outlier Monitor"
APP_SUBTITLE = (
    "Yahoo Finance option-chain scanner focused on outlier contracts: high premium, high volume/OI, "
    "cleaner quotes, aggressive side estimates, and fast DTE toggles."
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

# Use a v2 DB so old same-day overwrite behavior does not poison new events.
FLOW_DB = CACHE_DIR / "adfm_options_flow_outliers_v3.sqlite"
LEGACY_FLOW_DB = CACHE_DIR / "adfm_options_flow.sqlite"
UNIVERSE_META_CACHE = CACHE_DIR / "ndx100_meta.pkl"
LAST_GOOD_SNAPSHOT = CACHE_DIR / "last_good_underlying_snapshot.pkl"
LAST_GOOD_SCAN = CACHE_DIR / "last_good_live_scan_outliers_v3.pkl"
LAST_GOOD_DIAGS = CACHE_DIR / "last_good_live_diags_outliers_v3.pkl"
LAST_GOOD_META = CACHE_DIR / "last_good_live_meta_outliers_v3.json"

NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

DEFAULT_MIN_PREMIUM = 150_000.0
DEFAULT_MIN_VOLUME = 250
DEFAULT_MIN_VOL_OI = 2.0
DEFAULT_MAX_SPREAD_PCT = 30.0
DEFAULT_TOP_N = 150
DEFAULT_MAX_WORKERS = 4
DEFAULT_DTE_MIN = 0
DEFAULT_DTE_MAX = 60
DEFAULT_MAX_EXPIRIES_PER_SYMBOL = 8
DEFAULT_MIN_UNUSUAL_SCORE = 70.0
DEFAULT_MIN_UNDERLYING_DOLLAR_VOL_MM = 100.0
DEFAULT_BATCH_SIZE = 25
DEFAULT_MIN_PRICE = 10.0
DEFAULT_MIN_20D_VOL = 1_000_000

RETRY_ATTEMPTS = 3
RETRY_SLEEP_BASE = 0.85
CHAIN_JITTER_MAX = 0.35

DTE_PRESETS = [20, 30, 60, 90, 120, 360]
TREND_WINDOWS = [1, 5, 20, 30, 60, 90, 120, 360]

OUTLIER_PROFILES = {
    "Outliers only": {
        "min_premium": 150_000.0,
        "min_volume": 250,
        "min_vol_oi": 2.0,
        "max_spread_pct": 30.0,
        "min_unusual_score": 70.0,
        "min_outlier_score": 72.0,
        "min_premium_adv_bps": 0.10,
    },
    "Strict outliers": {
        "min_premium": 300_000.0,
        "min_volume": 500,
        "min_vol_oi": 3.0,
        "max_spread_pct": 25.0,
        "min_unusual_score": 78.0,
        "min_outlier_score": 82.0,
        "min_premium_adv_bps": 0.25,
    },
    "Exploratory": {
        "min_premium": 75_000.0,
        "min_volume": 100,
        "min_vol_oi": 1.25,
        "max_spread_pct": 40.0,
        "min_unusual_score": 58.0,
        "min_outlier_score": 60.0,
        "min_premium_adv_bps": 0.00,
    },
}

PASTEL_GREEN = "#BFE8C5"
PASTEL_RED = "#F4B6B6"
PASTEL_GREY = "#D1D5DB"
NET_LINE_GREY = "#6B7280"

DIRECTION_COLOR_MAP = {
    "BULLISH": PASTEL_GREEN,
    "BEARISH": PASTEL_RED,
    "NEUTRAL": PASTEL_GREY,
    "MIXED": PASTEL_GREY,
    "UNKNOWN": PASTEL_GREY,
}

TAPE_COLOR_MAP = {
    "BULLISH": PASTEL_GREEN,
    "BEARISH": PASTEL_RED,
    "NEUTRAL": PASTEL_GREY,
    "MIXED": PASTEL_GREY,
}

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
# Formatting and small utilities
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
    return str(sym).strip().upper().replace(".", "-")


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
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass


def with_retry_sleep(attempt: int) -> None:
    time.sleep(RETRY_SLEEP_BASE * attempt + random.uniform(0.0, CHAIN_JITTER_MAX))


def stable_hash(payload) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def normalize_scan_date_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "scan_date" in out.columns:
        out["scan_date"] = pd.to_datetime(out["scan_date"], errors="coerce").dt.date
    if "scan_ts" in out.columns:
        out["scan_ts"] = pd.to_datetime(out["scan_ts"], errors="coerce")
    return out

# =============================================================================
# Database
# =============================================================================
def init_db() -> None:
    with sqlite3.connect(FLOW_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS flow_events (
                event_id TEXT PRIMARY KEY,
                scan_date TEXT NOT NULL,
                scan_ts TEXT NOT NULL,
                scan_signature TEXT,
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
                direction_confidence REAL,
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
                outlier_score REAL,
                outlier_tier TEXT,
                premium_adv_bps REAL,
                expiry_bucket TEXT,
                outlier_reason TEXT,
                avg_dollar_vol REAL,
                cluster_flag TEXT,
                cluster_contracts REAL,
                cluster_premium REAL,
                cluster_volume REAL,
                cluster_score REAL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flow_symbol_date ON flow_events(symbol, scan_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flow_date ON flow_events(scan_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flow_scan_ts ON flow_events(scan_ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flow_signature ON flow_events(scan_signature)")
        conn.commit()


def ensure_event_id(row: pd.Series, scan_ts_str: str, scan_signature: str) -> str:
    payload = {
        "scan_ts": scan_ts_str,
        "signature": scan_signature,
        "symbol": row.get("symbol"),
        "contract_symbol": row.get("contract_symbol"),
        "premium": safe_float(row.get("premium"), 0),
        "volume": safe_float(row.get("volume"), 0),
    }
    return stable_hash(payload)


def save_flow_to_db(flow: pd.DataFrame, scan_ts: Optional[pd.Timestamp] = None, scan_signature: str = "") -> int:
    if flow is None or flow.empty:
        return 0

    init_db()
    scan_ts = scan_ts or pd.Timestamp.now()
    scan_date = scan_ts.date().isoformat()
    scan_ts_str = scan_ts.strftime("%Y-%m-%d %H:%M:%S")

    out = ensure_flow_schema(build_cluster_flags(flow.copy()))
    out["scan_date"] = scan_date
    out["scan_ts"] = scan_ts_str
    out["scan_signature"] = scan_signature

    if "direction_confidence" not in out.columns:
        out["direction_confidence"] = np.where(out.get("side", "").astype(str).isin(["ASK", "BID"]), 1.0, 0.0)

    cols = [
        "event_id", "scan_date", "scan_ts", "scan_signature", "symbol", "contract_symbol", "type", "expiry", "dte", "spot", "strike",
        "moneyness", "pct_otm", "side", "direction", "direction_confidence", "bid", "ask", "mid", "fill_est", "volume",
        "open_interest", "vol_oi", "premium", "spread_pct", "iv", "unusual_score", "outlier_score", "outlier_tier", "premium_adv_bps", "expiry_bucket", "outlier_reason", "avg_dollar_vol",
        "cluster_flag", "cluster_contracts", "cluster_premium", "cluster_volume", "cluster_score"
    ]

    for col in cols:
        if col not in out.columns:
            out[col] = "" if col in ["event_id", "cluster_flag", "scan_signature", "outlier_tier", "expiry_bucket", "outlier_reason"] else np.nan

    out["event_id"] = [ensure_event_id(row, scan_ts_str, scan_signature) for _, row in out.iterrows()]

    numeric_cols = [
        "dte", "spot", "strike", "pct_otm", "direction_confidence", "bid", "ask", "mid", "fill_est", "volume", "open_interest",
        "vol_oi", "premium", "spread_pct", "iv", "unusual_score", "outlier_score", "premium_adv_bps", "avg_dollar_vol", "cluster_contracts",
        "cluster_premium", "cluster_volume", "cluster_score"
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[cols].copy()

    with sqlite3.connect(FLOW_DB) as conn:
        placeholders = ",".join(["?"] * len(cols))
        update_cols = [c for c in cols if c != "event_id"]
        update_clause = ", ".join([f"{c}=excluded.{c}" for c in update_cols])
        sql = f"""
            INSERT INTO flow_events ({','.join(cols)})
            VALUES ({placeholders})
            ON CONFLICT(event_id)
            DO UPDATE SET {update_clause}
        """
        records = out.where(pd.notna(out), None).values.tolist()
        conn.executemany(sql, records)
        conn.commit()

    return len(out)


def migrate_legacy_db_if_needed() -> None:
    """One-time migration from the old same-day-overwrite DB, if present."""
    init_db()
    if not LEGACY_FLOW_DB.exists():
        return
    try:
        with sqlite3.connect(FLOW_DB) as conn:
            existing = pd.read_sql_query("SELECT COUNT(*) AS n FROM flow_events", conn)["n"].iloc[0]
        if existing > 0:
            return
        with sqlite3.connect(LEGACY_FLOW_DB) as conn_old:
            legacy = pd.read_sql_query("SELECT * FROM flow_events", conn_old)
        if legacy.empty:
            return
        legacy = ensure_flow_schema(legacy)
        legacy["scan_ts"] = pd.to_datetime(legacy.get("scan_ts", legacy.get("scan_date")), errors="coerce").fillna(pd.Timestamp.now())
        inserted = 0
        for ts, frame in legacy.groupby("scan_ts"):
            inserted += save_flow_to_db(frame, scan_ts=pd.Timestamp(ts), scan_signature="legacy_migration")
        if inserted:
            st.sidebar.caption(f"Migrated {inserted:,} legacy rows into v2 database.")
    except Exception:
        pass


def load_flow_db(days: int = 120) -> pd.DataFrame:
    init_db()
    with sqlite3.connect(FLOW_DB) as conn:
        df = pd.read_sql_query("SELECT * FROM flow_events ORDER BY scan_ts DESC, premium DESC", conn)
    if df.empty:
        return df
    df = normalize_scan_date_col(df)
    latest = pd.to_datetime(df["scan_date"], errors="coerce").max()
    if pd.notna(latest):
        start = latest.date() - timedelta(days=int(days))
        df = df[pd.to_datetime(df["scan_date"], errors="coerce").dt.date >= start].copy()
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
        return pd.DataFrame({
            "symbol": [normalize_symbol(x) for x in NASDAQ100_FALLBACK_SYMBOLS],
            "company": [normalize_symbol(x) for x in NASDAQ100_FALLBACK_SYMBOLS],
        }).drop_duplicates("symbol").reset_index(drop=True)


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
                level1 = set(data.columns.get_level_values(1))
                if sym in level0:
                    sub = data[sym].copy()
                elif sym in level1:
                    sub = data.xs(sym, axis=1, level=1).copy()
                else:
                    continue
            else:
                sub = data.copy()

            sub = sub.dropna(how="all")
            if sub.empty or "Close" not in sub.columns:
                continue
            close = pd.to_numeric(sub["Close"], errors="coerce").dropna()
            vol = pd.to_numeric(sub["Volume"], errors="coerce").dropna() if "Volume" in sub.columns else pd.Series(dtype=float)
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
            "batches": 0, "rows": 0, "failures": 0, "used_cached_snapshot": False
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
            return cached.copy(), {"batches": batches, "rows": len(cached), "failures": failures, "used_cached_snapshot": True}
        return pd.DataFrame(columns=["symbol", "spot", "prev_close", "avg_20d_vol", "day_volume", "avg_dollar_vol"]), {
            "batches": batches, "rows": 0, "failures": failures, "used_cached_snapshot": False
        }

    out = pd.DataFrame(all_rows).drop_duplicates("symbol", keep="last").reset_index(drop=True)
    save_pickle(out, LAST_GOOD_SNAPSHOT)
    return out, {"batches": batches, "rows": len(out), "failures": failures, "used_cached_snapshot": False}


def filter_underlyings(snapshot: pd.DataFrame, min_price: float, min_avg_dollar_vol_mm: float, min_avg_20d_vol: int) -> pd.DataFrame:
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
    if ask <= 0 or bid <= 0:
        return "ONE_SIDED"
    spread = ask - bid
    if spread <= 0:
        return "UNKNOWN"
    near_bid = abs(fill - bid)
    near_ask = abs(fill - ask)
    tol = max(spread * 0.20, 0.02)
    if near_ask <= tol and near_ask <= near_bid:
        return "ASK"
    if near_bid <= tol and near_bid < near_ask:
        return "BID"
    return "MID"


def classify_direction(option_type: str, side: str) -> Tuple[str, float]:
    """Return direction and confidence. MID is deliberately neutral now."""
    option_type = str(option_type).upper()
    side = str(side).upper()
    if side not in {"ASK", "BID"}:
        return "NEUTRAL", 0.0
    if option_type == "CALL":
        return ("BULLISH", 1.0) if side == "ASK" else ("BEARISH", 0.85)
    if option_type == "PUT":
        return ("BEARISH", 1.0) if side == "ASK" else ("BULLISH", 0.85)
    return "NEUTRAL", 0.0


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
    side_bonus = 0.06 if side in ("ASK", "BID") else 0.0
    liquidity_score = 0.4 if pd.isna(avg_dollar_vol) or avg_dollar_vol <= 0 else min(math.log10(avg_dollar_vol + 1.0) / 9.0, 1.0)
    raw = (
        0.30 * premium_score
        + 0.26 * voi_score
        + 0.14 * abs_size_score
        + 0.10 * otm_score
        + 0.08 * dte_score
        + 0.06 * liquidity_score
        + side_bonus
        - 0.18 * spread_penalty
    )
    return float(np.clip(raw, 0.0, 1.0) * 100.0)


def compute_outlier_metadata(
    premium: float,
    volume: float,
    open_interest: float,
    vol_oi: float,
    spread_pct: float,
    pct_otm_pct: float,
    dte: int,
    side: str,
    unusual_score: float,
    avg_dollar_vol: float,
) -> Tuple[float, str, float, str, str]:
    """Stricter public-chain outlier model: size, vol/OI, quote quality, aggression, DTE, OTM distance, and size versus underlying liquidity."""
    premium = safe_float(premium, 0.0)
    volume = safe_float(volume, 0.0)
    open_interest = safe_float(open_interest, 0.0)
    vol_oi = safe_float(vol_oi, 0.0)
    spread_pct = safe_float(spread_pct, np.nan)
    pct_otm_pct = safe_float(pct_otm_pct, 0.0)
    dte = safe_int(dte, 0)
    unusual_score = safe_float(unusual_score, 0.0)
    avg_dollar_vol = safe_float(avg_dollar_vol, np.nan)

    premium_adv_bps = 0.0 if pd.isna(avg_dollar_vol) or avg_dollar_vol <= 0 else premium / avg_dollar_vol * 10_000.0
    score = unusual_score * 0.62
    reasons = []

    if premium >= 1_000_000:
        score += 14; reasons.append(">$1M premium")
    elif premium >= 500_000:
        score += 10; reasons.append(">$500k premium")
    elif premium >= 250_000:
        score += 6; reasons.append(">$250k premium")

    if vol_oi >= 10:
        score += 15; reasons.append("vol/OI >=10x")
    elif vol_oi >= 5:
        score += 12; reasons.append("vol/OI >=5x")
    elif vol_oi >= 3:
        score += 8; reasons.append("vol/OI >=3x")
    elif vol_oi >= 1.5:
        score += 4; reasons.append("volume > OI")

    if volume >= 10_000:
        score += 9; reasons.append(">10k contracts")
    elif volume >= 3_000:
        score += 6; reasons.append(">3k contracts")
    elif volume >= 1_000:
        score += 3; reasons.append(">1k contracts")

    if pd.notna(spread_pct):
        if spread_pct <= 12:
            score += 6; reasons.append("tight spread")
        elif spread_pct <= 25:
            score += 3

    if side in {"ASK", "BID"}:
        score += 7; reasons.append("aggressive side")

    if dte <= 20:
        score += 6; reasons.append("<=20D")
    elif dte <= 60:
        score += 4
    elif dte >= 180:
        score -= 4

    if 3 <= pct_otm_pct <= 25:
        score += 6; reasons.append("meaningful OTM")
    elif pct_otm_pct > 40:
        score -= 5

    if premium_adv_bps >= 2.0:
        score += 8; reasons.append(">=2 bps ADV")
    elif premium_adv_bps >= 0.5:
        score += 5; reasons.append(">=0.5 bps ADV")
    elif premium_adv_bps >= 0.1:
        score += 2

    score = float(np.clip(score, 0.0, 100.0))
    if score >= 88:
        tier = "EXTREME"
    elif score >= 78:
        tier = "HIGH"
    elif score >= 68:
        tier = "WATCH"
    else:
        tier = "NOISE"
    return score, tier, premium_adv_bps, dte_bucket(dte), "; ".join(reasons[:5]) if reasons else "generic activity"

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
    min_outlier_score: float,
    min_premium_adv_bps: float,
    require_two_sided_quotes: bool,
) -> Tuple[List[dict], dict]:
    diagnostics = {
        "symbol": symbol,
        "expiries_seen": 0,
        "contracts_seen": 0,
        "contracts_kept": 0,
        "api_errors": 0,
        "usable_chain": 0,
        "reason": "",
        "detail": "",
    }
    rows: List[dict] = []

    tkr = get_ticker_with_retries(symbol)
    if tkr is None:
        diagnostics["api_errors"] = 1
        diagnostics["reason"] = "ticker_init_failed"
        return rows, diagnostics

    expiries, expiry_err = get_options_dates(tkr)
    if expiry_err is not None:
        diagnostics["api_errors"] = 1 if expiry_err == "options_load_failed" else 0
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
            diagnostics["api_errors"] += 1
            if diagnostics["reason"] == "":
                diagnostics["reason"] = chain_err or "chain_load_failed"
            continue

        chain_parts = [("CALL", getattr(chain, "calls", pd.DataFrame())), ("PUT", getattr(chain, "puts", pd.DataFrame()))]
        for option_type, df in chain_parts:
            if df is None or df.empty:
                continue
            local = df.copy()
            needed = ["contractSymbol", "strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]
            for col in needed:
                if col not in local.columns:
                    local[col] = np.nan
            local = local[needed].copy()
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
            fills = np.where(local["lastPrice"] > 0, local["lastPrice"], mids)
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

            local["two_sided_quote"] = (local["bid"] > 0) & (local["ask"] > 0) & (local["ask"] >= local["bid"])
            local["spread_pct"] = np.where(
                local["two_sided_quote"] & (local["mid"] > 0),
                ((local["ask"] - local["bid"]) / local["mid"]) * 100.0,
                np.nan,
            )
            if require_two_sided_quotes:
                local = local[local["two_sided_quote"]].copy()
            local = local[(local["spread_pct"].notna()) & (local["spread_pct"] <= max_spread_pct)].copy()
            if local.empty:
                continue

            local["side"] = [classify_side(f, b, a) for f, b, a in zip(local["fill_est"], local["bid"], local["ask"])]
            directions = [classify_direction(option_type, s) for s in local["side"]]
            local["direction"] = [d[0] for d in directions]
            local["direction_confidence"] = [d[1] for d in directions]

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
                    spread_pct=sp,
                    pct_otm=pct_otm,
                    dte=dte,
                    side=side,
                    avg_dollar_vol=avg_dollar_vol,
                )
                for prem, vol, oi, sp, pct_otm, side in zip(
                    local["premium"], local["volume"], local["openInterest"], local["spread_pct"], local["pct_otm_dec"], local["side"]
                )
            ]
            local = local[local["unusual_score"] >= min_unusual_score].copy()
            if local.empty:
                continue

            local["pct_otm"] = local["pct_otm_dec"] * 100.0
            meta_rows = [
                compute_outlier_metadata(
                    premium=prem,
                    volume=vol,
                    open_interest=oi,
                    vol_oi=voi,
                    spread_pct=sp,
                    pct_otm_pct=pct_otm,
                    dte=dte,
                    side=side,
                    unusual_score=score,
                    avg_dollar_vol=avg_dollar_vol,
                )
                for prem, vol, oi, voi, sp, pct_otm, side, score in zip(
                    local["premium"], local["volume"], local["openInterest"], local["vol_oi"],
                    local["spread_pct"], local["pct_otm"], local["side"], local["unusual_score"]
                )
            ]
            local["outlier_score"] = [x[0] for x in meta_rows]
            local["outlier_tier"] = [x[1] for x in meta_rows]
            local["premium_adv_bps"] = [x[2] for x in meta_rows]
            local["expiry_bucket"] = [x[3] for x in meta_rows]
            local["outlier_reason"] = [x[4] for x in meta_rows]
            local = local[
                (local["outlier_score"] >= min_outlier_score)
                & (local["premium_adv_bps"] >= min_premium_adv_bps)
                & (local["outlier_tier"].isin(["WATCH", "HIGH", "EXTREME"]))
            ].copy()
            if local.empty:
                continue

            local["symbol"] = symbol
            local["type"] = option_type
            local["expiry"] = exp
            local["dte"] = int(dte)
            local["spot"] = float(spot)
            local["avg_dollar_vol"] = avg_dollar_vol

            rows.extend(local[[
                "symbol", "contractSymbol", "type", "expiry", "dte", "spot", "strike", "moneyness",
                "pct_otm", "side", "direction", "direction_confidence", "bid", "ask", "mid", "fill_est",
                "volume", "openInterest", "vol_oi", "premium", "spread_pct", "impliedVolatility",
                "unusual_score", "outlier_score", "outlier_tier", "premium_adv_bps", "expiry_bucket", "outlier_reason", "avg_dollar_vol",
            ]].rename(columns={
                "contractSymbol": "contract_symbol",
                "openInterest": "open_interest",
                "impliedVolatility": "iv",
            }).to_dict("records"))
            diagnostics["contracts_kept"] += len(local)
            diagnostics["usable_chain"] = 1

    if diagnostics["contracts_kept"] == 0 and diagnostics["reason"] == "":
        diagnostics["reason"] = "empty_chains" if diagnostics["contracts_seen"] == 0 else "all_filtered_out"
    return rows, diagnostics

# =============================================================================
# Schema and clusters
# =============================================================================
def ensure_flow_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    defaults = {
        "event_id": "",
        "scan_date": None,
        "scan_ts": pd.NaT,
        "scan_signature": "",
        "avg_dollar_vol": np.nan,
        "iv": np.nan,
        "spread_pct": np.nan,
        "pct_otm": np.nan,
        "direction_confidence": np.nan,
        "outlier_score": np.nan,
        "outlier_tier": "",
        "premium_adv_bps": np.nan,
        "expiry_bucket": "",
        "outlier_reason": "",
        "cluster_flag": "",
        "cluster_contracts": np.nan,
        "cluster_premium": np.nan,
        "cluster_volume": np.nan,
        "cluster_score": np.nan,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    if "scan_date" in out.columns:
        out = normalize_scan_date_col(out)
    numeric_cols = [
        "dte", "spot", "strike", "pct_otm", "direction_confidence", "bid", "ask", "mid", "fill_est", "volume",
        "open_interest", "vol_oi", "premium", "spread_pct", "iv", "unusual_score", "outlier_score", "premium_adv_bps", "avg_dollar_vol",
        "cluster_contracts", "cluster_premium", "cluster_volume", "cluster_score"
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_cluster_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out = df.copy()
    stale_cols = ["strike_bucket", "cluster_key", "cluster_contracts", "cluster_premium", "cluster_volume", "cluster_score", "cluster_flag"]
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
        out["symbol"].astype(str) + "|" + out["expiry"].astype(str) + "|" + out["type"].astype(str) + "|" +
        out["direction"].astype(str) + "|" + out["strike_bucket"].astype(str)
    )
    grp = out.groupby("cluster_key", dropna=False).agg(
        cluster_contracts=("contract_symbol", "count"),
        cluster_premium=("premium", "sum"),
        cluster_volume=("volume", "sum"),
        cluster_score=("unusual_score", "mean"),
    ).reset_index()
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
    min_outlier_score: float,
    min_premium_adv_bps: float,
    require_two_sided_quotes: bool,
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
                min_outlier_score,
                min_premium_adv_bps,
                require_two_sided_quotes,
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
                    "api_errors": 1,
                    "usable_chain": 0,
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
        flow = ensure_flow_schema(build_cluster_flags(flow))
        flow = flow.sort_values(["outlier_score", "premium", "vol_oi", "volume"], ascending=[False, False, False, False]).reset_index(drop=True)
    return flow, diags

# =============================================================================
# Dashboard filters and trend engine
# =============================================================================
def anchor_date_for_db(db: pd.DataFrame) -> Optional[date]:
    if db is None or db.empty or "scan_date" not in db.columns:
        return None
    s = pd.to_datetime(db["scan_date"], errors="coerce").dropna()
    return s.max().date() if not s.empty else None


def filter_to_lookback(db: pd.DataFrame, lookback_days: int, anchor: Optional[date] = None) -> pd.DataFrame:
    if db is None or db.empty:
        return pd.DataFrame()
    out = normalize_scan_date_col(db)
    anchor = anchor or anchor_date_for_db(out) or date.today()
    start = anchor - timedelta(days=int(lookback_days))
    return out[pd.to_datetime(out["scan_date"], errors="coerce").dt.date >= start].copy()


def apply_dashboard_filters(
    df: pd.DataFrame,
    symbols: List[str],
    directions: List[str],
    dte_range: Tuple[int, int],
    min_premium: float,
    min_score: float,
    min_outlier_score: float,
    min_confidence: float,
    outlier_tiers: List[str],
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = ensure_flow_schema(df)
    if symbols:
        allowed = {normalize_symbol(s) for s in symbols}
        out = out[out["symbol"].astype(str).map(normalize_symbol).isin(allowed)]
    if directions and "All" not in directions:
        out = out[out["direction"].astype(str).isin(directions)]
    if outlier_tiers and "All" not in outlier_tiers and "outlier_tier" in out.columns:
        out = out[out["outlier_tier"].astype(str).isin(outlier_tiers)]
    out = out[pd.to_numeric(out["dte"], errors="coerce").between(dte_range[0], dte_range[1], inclusive="both")]
    out = out[pd.to_numeric(out["premium"], errors="coerce") >= float(min_premium)]
    out = out[pd.to_numeric(out["unusual_score"], errors="coerce") >= float(min_score)]
    out = out[pd.to_numeric(out["outlier_score"], errors="coerce").fillna(0) >= float(min_outlier_score)]
    out = out[pd.to_numeric(out["direction_confidence"], errors="coerce").fillna(0) >= float(min_confidence)]
    return out.copy()


def build_trend_board(db: pd.DataFrame, lookback_days: int, anchor: Optional[date] = None) -> pd.DataFrame:
    if db is None or db.empty:
        return pd.DataFrame()
    sub = filter_to_lookback(db, lookback_days, anchor=anchor)
    if sub.empty:
        return pd.DataFrame()

    sub = ensure_flow_schema(sub)
    sub["weighted_premium"] = sub["premium"].fillna(0) * sub["direction_confidence"].fillna(0)
    sub["bullish_premium"] = np.where(sub["direction"] == "BULLISH", sub["weighted_premium"], 0.0)
    sub["bearish_premium"] = np.where(sub["direction"] == "BEARISH", sub["weighted_premium"], 0.0)
    sub["neutral_premium"] = np.where(~sub["direction"].isin(["BULLISH", "BEARISH"]), sub["premium"], 0.0)
    sub["bullish_rows"] = np.where(sub["direction"] == "BULLISH", 1, 0)
    sub["bearish_rows"] = np.where(sub["direction"] == "BEARISH", 1, 0)
    sub["neutral_rows"] = np.where(~sub["direction"].isin(["BULLISH", "BEARISH"]), 1, 0)
    sub["cluster_int"] = np.where(sub["cluster_flag"].astype(str).str.upper() == "CLUSTER", 1, 0)

    group_key = "event_id" if "event_id" in sub.columns and sub["event_id"].astype(str).str.len().gt(0).any() else "contract_symbol"
    trend = sub.groupby("symbol", as_index=False).agg(
        unusual_trades=(group_key, "count"),
        active_days=("scan_date", lambda x: pd.Series(x).nunique()),
        total_premium=("premium", "sum"),
        bullish_premium=("bullish_premium", "sum"),
        bearish_premium=("bearish_premium", "sum"),
        neutral_premium=("neutral_premium", "sum"),
        bullish_trades=("bullish_rows", "sum"),
        bearish_trades=("bearish_rows", "sum"),
        neutral_trades=("neutral_rows", "sum"),
        avg_score=("unusual_score", "mean"),
        avg_outlier_score=("outlier_score", "mean"),
        max_score=("unusual_score", "max"),
        max_outlier_score=("outlier_score", "max"),
        avg_dte=("dte", "mean"),
        largest_print=("premium", "max"),
        clusters=("cluster_int", "sum"),
        last_seen=("scan_ts", "max"),
    )
    trend["directional_premium"] = trend["bullish_premium"] + trend["bearish_premium"]
    trend["net_premium"] = trend["bullish_premium"] - trend["bearish_premium"]
    trend["net_trades"] = trend["bullish_trades"] - trend["bearish_trades"]
    trend["bull_bear_score"] = np.where(
        trend["directional_premium"] > 0,
        100.0 * trend["net_premium"] / trend["directional_premium"],
        0.0,
    )
    trend["persistence"] = trend["active_days"] / max(lookback_days, 1)
    trend["tape"] = np.select(
        [trend["bull_bear_score"] >= 35, trend["bull_bear_score"] <= -35],
        ["BULLISH", "BEARISH"],
        default="MIXED",
    )
    trend["conviction_score"] = (
        trend["avg_outlier_score"].fillna(trend["avg_score"]).fillna(0) * 0.45
        + np.clip(np.log10(trend["directional_premium"].fillna(0) + 1) / 7.0 * 100, 0, 100) * 0.28
        + np.clip(trend["unusual_trades"].fillna(0) / 20 * 100, 0, 100) * 0.15
        + np.clip(trend["active_days"].fillna(0) / max(min(lookback_days, 20), 1) * 100, 0, 100) * 0.12
    )
    return trend.sort_values(["conviction_score", "directional_premium"], ascending=[False, False]).reset_index(drop=True)


def build_multi_window_scores(db: pd.DataFrame, windows: List[int], anchor: Optional[date] = None, sort_window: int = 20) -> pd.DataFrame:
    boards = []
    unique_windows = []
    for w in windows:
        if int(w) not in unique_windows:
            unique_windows.append(int(w))
    for w in unique_windows:
        b = build_trend_board(db, w, anchor=anchor)
        if b.empty:
            continue
        keep = b[["symbol", "unusual_trades", "directional_premium", "net_premium", "bull_bear_score", "conviction_score", "tape"]].copy()
        keep = keep.rename(columns={
            "unusual_trades": f"{w}D Trades",
            "directional_premium": f"{w}D Directional Premium",
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
    sort_col = f"{sort_window}D Conviction" if f"{sort_window}D Conviction" in out.columns else [c for c in out.columns if c.endswith("Conviction")][0]
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
    out["Outlier"] = out["outlier_score"].map(lambda x: f"{safe_float(x, 0):.1f}")
    out["ADV bps"] = out["premium_adv_bps"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    out["Conf"] = out["direction_confidence"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    out["Date"] = out["scan_date"].astype(str) if "scan_date" in out.columns else ""
    out["Time"] = pd.to_datetime(out["scan_ts"], errors="coerce").dt.strftime("%H:%M:%S") if "scan_ts" in out.columns else ""
    keep = [
        "Date", "Time", "symbol", "type", "expiry", "dte", "expiry_bucket", "direction", "Conf", "side", "strike", "spot",
        "moneyness", "OTM %", "Fill", "Bid", "Ask", "Spread %", "Vol", "OI", "Vol/OI", "Premium", "ADV bps", "IV",
        "Score", "Outlier", "outlier_tier", "outlier_reason", "cluster_flag", "contract_symbol",
    ]
    return out[[c for c in keep if c in out.columns]].copy()


def prep_trend_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    money_cols = [c for c in out.columns if "Premium" in c or c in ["total_premium", "directional_premium", "bullish_premium", "bearish_premium", "neutral_premium", "net_premium", "largest_print"]]
    for col in money_cols:
        if col in out.columns:
            out[col] = out[col].map(human_money)
    score_cols = [c for c in out.columns if "Score" in c or "Conviction" in c or c in ["avg_score", "max_score", "avg_outlier_score", "max_outlier_score", "bull_bear_score"]]
    for col in score_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    if "avg_dte" in out.columns:
        out["avg_dte"] = out["avg_dte"].map(lambda x: "" if pd.isna(x) else f"{x:.0f}")
    if "last_seen" in out.columns:
        out["last_seen"] = pd.to_datetime(out["last_seen"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    return out


def render_top_metrics(db: pd.DataFrame, live_flow: pd.DataFrame, scan_meta: dict, dashboard_db: pd.DataFrame) -> None:
    db_rows = 0 if db is None or db.empty else len(db)
    dash_rows = 0 if dashboard_db is None or dashboard_db.empty else len(dashboard_db)
    db_symbols = 0 if dashboard_db is None or dashboard_db.empty else dashboard_db["symbol"].nunique()
    live_rows = 0 if live_flow is None or live_flow.empty else len(live_flow)
    live_premium = 0.0 if live_flow is None or live_flow.empty else float(live_flow["premium"].sum())
    live_bull = 0.0 if live_flow is None or live_flow.empty else float(live_flow.loc[live_flow["direction"] == "BULLISH", "premium"].sum())
    live_bear = 0.0 if live_flow is None or live_flow.empty else float(live_flow.loc[live_flow["direction"] == "BEARISH", "premium"].sum())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Loaded live rows", f"{live_rows:,}")
    c2.metric("Live premium", human_money(live_premium))
    c3.metric("Live bull", human_money(live_bull))
    c4.metric("Live bear", human_money(live_bear))
    c5.metric("DB rows / filtered", f"{db_rows:,} / {dash_rows:,}")
    c6.metric("Filtered tickers", f"{db_symbols:,}")

    if scan_meta:
        meta = f"Mode: {scan_meta.get('mode', 'Unknown')}"
        if scan_meta.get("saved_at"):
            meta += f" | Snapshot: {scan_meta.get('saved_at')}"
        if scan_meta.get("symbols_scanned") is not None:
            meta += f" | Symbols scanned: {safe_int(scan_meta.get('symbols_scanned')):,}"
        if scan_meta.get("raw_contracts") is not None:
            meta += f" | Raw contracts: {safe_int(scan_meta.get('raw_contracts')):,}"
        if scan_meta.get("usable_ratio") is not None:
            meta += f" | Usable-chain ratio: {safe_float(scan_meta.get('usable_ratio'), 0):.0%}"
        if scan_meta.get("api_success_ratio") is not None:
            meta += f" | API success: {safe_float(scan_meta.get('api_success_ratio'), 0):.0%}"
        if scan_meta.get("elapsed") is not None:
            meta += f" | Elapsed: {safe_float(scan_meta.get('elapsed'), 0):.1f}s"
        if scan_meta.get("cache_warning"):
            meta += f" | {scan_meta.get('cache_warning')}"
        st.caption(meta)


def render_trend_bar(trend: pd.DataFrame, lookback_days: int, top_n: int, rank_by: str) -> None:
    if trend.empty:
        st.info("No saved trend data for this window yet. Run and save a scan first, or loosen dashboard filters.")
        return
    if rank_by == "Absolute net premium":
        top = trend.assign(abs_net=trend["net_premium"].abs()).sort_values("abs_net", ascending=False).head(top_n).copy()
    elif rank_by == "Directional premium":
        top = trend.sort_values("directional_premium", ascending=False).head(top_n).copy()
    else:
        top = trend.sort_values("conviction_score", ascending=False).head(top_n).copy()

    fig = px.bar(
        top.sort_values("net_premium", ascending=True),
        x="net_premium",
        y="symbol",
        color="tape",
        color_discrete_map=TAPE_COLOR_MAP,
        category_orders={"tape": ["BULLISH", "MIXED", "NEUTRAL", "BEARISH"]},
        orientation="h",
        hover_data={
            "unusual_trades": True,
            "active_days": True,
            "directional_premium": ":,.0f",
            "total_premium": ":,.0f",
            "bullish_premium": ":,.0f",
            "bearish_premium": ":,.0f",
            "neutral_premium": ":,.0f",
            "bull_bear_score": ":.1f",
            "conviction_score": ":.1f",
            "avg_outlier_score": ":.1f",
        },
        title=f"Ticker Net Directional Flow | {lookback_days}D Window | Ranked by {rank_by}",
    )
    fig.update_layout(height=max(420, min(900, 35 * len(top) + 120)), margin=dict(l=20, r=20, t=55, b=20), yaxis_title="", xaxis_title="Net bullish minus bearish premium")
    st.plotly_chart(fig, use_container_width=True, key=f"trend_bar_{lookback_days}_{top_n}_{rank_by}_{len(trend)}_{float(trend['net_premium'].sum()):.0f}")


def render_ticker_timeline(db: pd.DataFrame, ticker: str, lookback_days: int, anchor: Optional[date]) -> None:
    if db.empty or not ticker:
        return
    sub = db[db["symbol"] == ticker].copy()
    sub = filter_to_lookback(sub, lookback_days, anchor=anchor)
    if sub.empty:
        st.info("No saved rows for this ticker in the selected dashboard window.")
        return
    sub["scan_date_dt"] = pd.to_datetime(sub["scan_date"], errors="coerce")
    sub["weighted_premium"] = sub["premium"].fillna(0) * sub["direction_confidence"].fillna(0)
    sub["bullish_premium"] = np.where(sub["direction"] == "BULLISH", sub["weighted_premium"], 0.0)
    sub["bearish_premium"] = np.where(sub["direction"] == "BEARISH", sub["weighted_premium"], 0.0)
    sub["neutral_premium"] = np.where(~sub["direction"].isin(["BULLISH", "BEARISH"]), sub["premium"], 0.0)
    daily = sub.groupby("scan_date_dt", as_index=False).agg(
        bullish_premium=("bullish_premium", "sum"),
        bearish_premium=("bearish_premium", "sum"),
        neutral_premium=("neutral_premium", "sum"),
        rows=("contract_symbol", "count"),
    )
    daily["net_premium"] = daily["bullish_premium"] - daily["bearish_premium"]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily["scan_date_dt"], y=daily["bullish_premium"], name="Bullish premium", marker_color=PASTEL_GREEN))
    fig.add_trace(go.Bar(x=daily["scan_date_dt"], y=-daily["bearish_premium"], name="Bearish premium", marker_color=PASTEL_RED))
    fig.add_trace(go.Bar(x=daily["scan_date_dt"], y=daily["neutral_premium"], name="Neutral premium", marker_color=PASTEL_GREY))
    fig.add_trace(go.Scatter(x=daily["scan_date_dt"], y=daily["net_premium"], mode="lines+markers", name="Net premium", line=dict(color=NET_LINE_GREY), marker=dict(color=NET_LINE_GREY)))
    fig.update_layout(
        title=f"{ticker} | Saved Unusual Flow by Day | {lookback_days}D Window",
        barmode="relative",
        height=500,
        margin=dict(l=20, r=20, t=55, b=20),
        yaxis_title="Premium",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"ticker_timeline_{ticker}_{lookback_days}_{len(sub)}")

# =============================================================================
# State
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
migrate_legacy_db_if_needed()

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
        **Purpose:** Capture option-chain outliers, not generic active contracts.

        **Scan filters** affect the next Yahoo scan. **Dashboard filters** affect existing live/cache/database views immediately.
        The DTE controls below let you flip through 20, 30, 60, 90, 120, and 360-day contracts without dragging ranges.
        """
    )

    st.subheader("Universe")
    universe_mode = st.selectbox("Universe mode", ["Nasdaq 100 + ADFM extras", "Nasdaq 100 only", "Custom only"], index=0)
    custom_raw = st.text_area("Custom tickers", value="", placeholder="Example: SLV, GLD, VST, HOOD, CAR, NKTR", height=80)
    custom_symbols = parse_custom_symbols(custom_raw)
    if universe_mode == "Nasdaq 100 only":
        symbols_all = sorted(set(ndx_symbols))
    elif universe_mode == "Custom only":
        symbols_all = sorted(set(custom_symbols))
    else:
        symbols_all = sorted(set(ndx_symbols + DEFAULT_EXTRA_SYMBOLS + custom_symbols))

    st.subheader("Outlier Scan Profile")
    scan_profile = st.selectbox("Profile", ["Outliers only", "Strict outliers", "Exploratory"], index=0)
    profile_defaults = OUTLIER_PROFILES[scan_profile]
    scan_dte_max = st.select_slider("Scan option expiry max DTE", options=DTE_PRESETS, value=60)
    dte_range = (0, int(scan_dte_max))

    min_price = st.number_input("Minimum stock price ($)", min_value=0.0, value=float(DEFAULT_MIN_PRICE), step=5.0, format="%.0f")
    min_avg_20d_vol = st.number_input("Minimum 20D avg share volume", min_value=0, value=int(DEFAULT_MIN_20D_VOL), step=250_000)
    min_underlying_dollar_vol_mm = st.number_input("Minimum underlying ADV ($mm)", min_value=0.0, value=float(DEFAULT_MIN_UNDERLYING_DOLLAR_VOL_MM), step=25.0, format="%.0f")

    min_premium = st.number_input("Minimum premium ($)", min_value=0.0, value=float(profile_defaults["min_premium"]), step=25_000.0, format="%.0f", key=f"scan_min_premium_{scan_profile}")
    min_volume = st.number_input("Minimum contract volume", min_value=1, value=int(profile_defaults["min_volume"]), step=25, key=f"scan_min_volume_{scan_profile}")
    min_vol_oi = st.number_input("Minimum volume / OI", min_value=0.0, value=float(profile_defaults["min_vol_oi"]), step=0.25, format="%.2f", key=f"scan_min_vol_oi_{scan_profile}")
    max_spread_pct = st.number_input("Maximum spread %", min_value=0.0, value=float(profile_defaults["max_spread_pct"]), step=5.0, format="%.1f", key=f"scan_max_spread_{scan_profile}")
    min_unusual_score = st.slider("Minimum unusual score", min_value=0.0, max_value=100.0, value=float(profile_defaults["min_unusual_score"]), step=1.0, key=f"scan_min_unusual_{scan_profile}")
    min_outlier_score = st.slider("Minimum outlier score", min_value=0.0, max_value=100.0, value=float(profile_defaults["min_outlier_score"]), step=1.0, key=f"scan_min_outlier_{scan_profile}")
    min_premium_adv_bps = st.number_input("Minimum premium / underlying ADV, bps", min_value=0.0, value=float(profile_defaults["min_premium_adv_bps"]), step=0.05, format="%.2f", key=f"scan_min_adv_bps_{scan_profile}")
    require_two_sided_quotes = st.toggle("Require two-sided option quotes", value=True)

    st.subheader("Scan Engine")
    max_expiries_per_symbol = st.slider("Max expiries per symbol", min_value=1, max_value=24, value=10, step=1)
    max_workers = st.slider("Concurrent workers", min_value=1, max_value=10, value=DEFAULT_MAX_WORKERS, step=1)
    allow_cached_fallback = st.toggle("Use same-filter last-good cached scan when live scan is degraded", value=True)
    save_to_database = st.toggle("Save qualifying rows to local trend database", value=True)
    run_scan = st.button("Run Outlier Scan", use_container_width=True, type="primary")

    st.subheader("Dashboard Filters")
    dashboard_lookback = st.selectbox("Trend lookback", TREND_WINDOWS, index=2, key="dashboard_window")
    dashboard_dte_max = st.select_slider("Option expiry max DTE", options=DTE_PRESETS, value=60, key="dashboard_dte_max")
    dashboard_dte_range = (0, int(dashboard_dte_max))
    top_n = st.slider("Rows / tickers to display", min_value=10, max_value=500, value=DEFAULT_TOP_N, step=10)
    rank_by = st.selectbox("Trend chart ranking", ["Conviction", "Absolute net premium", "Directional premium"], index=0)
    dashboard_directions = st.multiselect("Direction filter", ["All", "BULLISH", "BEARISH", "NEUTRAL"], default=["BULLISH", "BEARISH"])
    dashboard_tiers = st.multiselect("Outlier tier", ["All", "EXTREME", "HIGH", "WATCH", "NOISE"], default=["EXTREME", "HIGH", "WATCH"])
    dashboard_min_premium = st.number_input("Dashboard min premium ($)", min_value=0.0, value=float(profile_defaults["min_premium"]), step=25_000.0, format="%.0f")
    dashboard_min_score = st.slider("Dashboard min unusual score", min_value=0.0, max_value=100.0, value=float(profile_defaults["min_unusual_score"]), step=1.0)
    dashboard_min_outlier_score = st.slider("Dashboard min outlier score", min_value=0.0, max_value=100.0, value=float(profile_defaults["min_outlier_score"]), step=1.0)
    dashboard_min_confidence = st.slider("Dashboard min direction confidence", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
    dashboard_symbol_filter_raw = st.text_input("Dashboard ticker filter", value="", placeholder="Optional: NVDA, AMD, META")
    dashboard_symbols = parse_custom_symbols(dashboard_symbol_filter_raw)

    st.subheader("Database")
    db_days = st.slider("Database history loaded", min_value=5, max_value=730, value=max(360, int(dashboard_lookback)), step=5)
    if st.button("Clear local flow database", use_container_width=True):
        clear_flow_db()
        st.success("Local flow database cleared.")

scan_filter_payload = {
    "universe_mode": universe_mode,
    "symbols": symbols_all,
    "min_price": min_price,
    "min_avg_20d_vol": min_avg_20d_vol,
    "min_underlying_dollar_vol_mm": min_underlying_dollar_vol_mm,
    "min_premium": min_premium,
    "min_volume": min_volume,
    "min_vol_oi": min_vol_oi,
    "max_spread_pct": max_spread_pct,
    "scan_profile": scan_profile,
    "min_unusual_score": min_unusual_score,
    "min_outlier_score": min_outlier_score,
    "min_premium_adv_bps": min_premium_adv_bps,
    "dte_range": dte_range,
    "max_expiries_per_symbol": max_expiries_per_symbol,
    "require_two_sided_quotes": require_two_sided_quotes,
}
scan_signature = stable_hash(scan_filter_payload)

# =============================================================================
# Startup cached load
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
            "api_success_ratio": cached_meta.get("api_success_ratio", None) if isinstance(cached_meta, dict) else None,
            "usable_ratio": cached_meta.get("usable_ratio", None) if isinstance(cached_meta, dict) else None,
            "raw_contracts": cached_meta.get("raw_contracts", None) if isinstance(cached_meta, dict) else None,
            "symbols_requested": safe_int(cached_meta.get("symbols_requested", len(symbols_all))) if isinstance(cached_meta, dict) else len(symbols_all),
            "symbols_scanned": safe_int(cached_meta.get("symbols_scanned", 0)) if isinstance(cached_meta, dict) else 0,
            "signature": cached_meta.get("scan_signature", "") if isinstance(cached_meta, dict) else "",
        }
    else:
        st.session_state["flow_raw"] = pd.DataFrame()
        st.session_state["diags_raw"] = pd.DataFrame()
        st.session_state["scan_meta"] = {"mode": "No scan loaded yet", "saved_at": None, "elapsed": None, "api_success_ratio": None, "usable_ratio": None, "raw_contracts": None, "symbols_requested": len(symbols_all), "symbols_scanned": 0, "signature": ""}

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

    filtered_snapshot = filter_underlyings(snapshot, min_price, min_underlying_dollar_vol_mm, min_avg_20d_vol)
    symbols_scanned_list = filtered_snapshot["symbol"].dropna().astype(str).tolist() if not filtered_snapshot.empty else []
    symbols_scanned_count = len(symbols_scanned_list)
    st.session_state["filtered_snapshot"] = filtered_snapshot.copy()

    flow_live = pd.DataFrame()
    diags_live = pd.DataFrame()
    api_success_ratio = None
    usable_ratio = None
    raw_contracts = 0
    empty_reason = None

    if symbols_scanned_count == 0:
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
                    min_outlier_score=min_outlier_score,
                    min_premium_adv_bps=min_premium_adv_bps,
                    require_two_sided_quotes=require_two_sided_quotes,
                )
            except Exception as e:
                st.warning(f"Live scan hit an error: {e}")
                flow_live = pd.DataFrame()
                diags_live = pd.DataFrame()

        if diags_live is not None and not diags_live.empty:
            raw_contracts = int(diags_live["contracts_seen"].sum()) if "contracts_seen" in diags_live.columns else 0
            api_success_ratio = float((diags_live["api_errors"].fillna(0) == 0).sum() / max(len(diags_live), 1)) if "api_errors" in diags_live.columns else None
            usable_ratio = float(diags_live["usable_chain"].fillna(0).sum() / max(len(diags_live), 1)) if "usable_chain" in diags_live.columns else None

    expected_contract_floor = max(10, min(250, symbols_scanned_count * 8))
    degraded = (
        diags_live is None
        or diags_live.empty
        or (api_success_ratio is not None and api_success_ratio < 0.35)
        or raw_contracts < expected_contract_floor
    )

    if degraded and allow_cached_fallback:
        cached_flow = load_pickle(LAST_GOOD_SCAN)
        cached_diags = load_pickle(LAST_GOOD_DIAGS)
        cached_meta = load_json(LAST_GOOD_META)
        same_signature = isinstance(cached_meta, dict) and cached_meta.get("scan_signature") == scan_signature
        if same_signature and isinstance(cached_flow, pd.DataFrame) and not cached_flow.empty:
            st.session_state["flow_raw"] = ensure_flow_schema(build_cluster_flags(cached_flow.copy()))
            st.session_state["diags_raw"] = cached_diags if isinstance(cached_diags, pd.DataFrame) else pd.DataFrame()
            st.session_state["scan_meta"] = {
                "mode": "Same-filter cached fallback",
                "saved_at": cached_meta.get("saved_at", "unknown"),
                "elapsed": round(time.time() - scan_started, 1),
                "api_success_ratio": api_success_ratio,
                "usable_ratio": usable_ratio,
                "raw_contracts": raw_contracts,
                "symbols_requested": len(symbols_all),
                "symbols_scanned": symbols_scanned_count,
                "signature": scan_signature,
                "cache_warning": "Live scan degraded; same-filter cache loaded.",
            }
            st.session_state["scan_ran"] = True
            st.warning("Live scan looked degraded, so the app loaded the same-filter cached snapshot.")
        else:
            flow_live = ensure_flow_schema(flow_live)
            st.session_state["flow_raw"] = flow_live
            st.session_state["diags_raw"] = diags_live
            st.session_state["scan_meta"] = {
                "mode": "Live degraded",
                "saved_at": scan_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed": round(time.time() - scan_started, 1),
                "api_success_ratio": api_success_ratio,
                "usable_ratio": usable_ratio,
                "raw_contracts": raw_contracts,
                "symbols_requested": len(symbols_all),
                "symbols_scanned": symbols_scanned_count,
                "signature": scan_signature,
                "cache_warning": "No same-filter cache used.",
                "empty_reason": empty_reason or "Live scan returned weak chain coverage.",
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
            "api_success_ratio": api_success_ratio,
            "usable_ratio": usable_ratio,
            "raw_contracts": raw_contracts,
            "symbols_requested": len(symbols_all),
            "symbols_scanned": symbols_scanned_count,
            "signature": scan_signature,
            "empty_reason": empty_reason,
        }
        st.session_state["scan_ran"] = True
        if flow_live is not None and not flow_live.empty:
            save_pickle(flow_live, LAST_GOOD_SCAN)
            save_pickle(diags_live, LAST_GOOD_DIAGS)
            save_json({
                "saved_at": scan_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "rows": int(len(flow_live)),
                "symbols_requested": int(len(symbols_all)),
                "symbols_scanned": int(symbols_scanned_count),
                "raw_contracts": raw_contracts,
                "api_success_ratio": api_success_ratio,
                "usable_ratio": usable_ratio,
                "scan_signature": scan_signature,
                "scan_profile": scan_profile,
                "scan_dte_range": dte_range,
                "min_outlier_score": min_outlier_score,
            }, LAST_GOOD_META)
            if save_to_database:
                inserted = save_flow_to_db(flow_live, scan_ts=scan_ts, scan_signature=scan_signature)
                st.success(f"Saved {inserted:,} qualifying event rows to the local trend database.")

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

# Always load enough DB history for the largest visible window and the selected database load.
needed_db_days = max(int(db_days), int(dashboard_lookback), 120)
db_raw = load_flow_db(days=needed_db_days)
dashboard_db = apply_dashboard_filters(
    db_raw,
    symbols=dashboard_symbols,
    directions=dashboard_directions,
    dte_range=dashboard_dte_range,
    min_premium=dashboard_min_premium,
    min_score=dashboard_min_score,
    min_outlier_score=dashboard_min_outlier_score,
    min_confidence=dashboard_min_confidence,
    outlier_tiers=dashboard_tiers,
)
flow_view = apply_dashboard_filters(
    flow,
    symbols=dashboard_symbols,
    directions=dashboard_directions,
    dte_range=dashboard_dte_range,
    min_premium=dashboard_min_premium,
    min_score=dashboard_min_score,
    min_outlier_score=dashboard_min_outlier_score,
    min_confidence=dashboard_min_confidence,
    outlier_tiers=dashboard_tiers,
)
anchor = anchor_date_for_db(dashboard_db)
render_top_metrics(db_raw, flow_view, scan_meta, dashboard_db)

if scan_meta.get("empty_reason"):
    st.info(scan_meta.get("empty_reason"))

if not st.session_state["scan_ran"] and flow_view.empty and dashboard_db.empty:
    st.markdown(
        """
        <div class="status-card">
        <b>No scan has been run yet and the local trend database is empty.</b><br>
        Click <b>Run Live Scan</b> in the sidebar. Qualifying rows can be saved into the local database, which powers the rolling trend board.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# Tabs
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Trend Board",
    "Outlier Tape",
    "Ticker Drilldown",
    "Live Scan Diagnostics",
    "Method Notes",
])

with tab1:
    st.subheader("Rolling Bull/Bear Trend Board")
    if dashboard_db.empty:
        st.info("No database rows match the current dashboard filters. Run a live scan with database saving on, or loosen the dashboard filters.")
    else:
        trend = build_trend_board(dashboard_db, dashboard_lookback, anchor=anchor)
        windows = [20, 30, 60, 90, 120, 360, dashboard_lookback]
        multi = build_multi_window_scores(dashboard_db, windows=windows, anchor=anchor, sort_window=dashboard_lookback)

        if trend.empty:
            st.info("No saved rows in this dashboard window.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trend window", f"{dashboard_lookback}D")
            c2.metric("Option DTE", f"0-{dashboard_dte_max}D")
            c3.metric("Trend tickers", f"{len(trend):,}")
            c4.metric("Net premium", human_money(float(trend["net_premium"].sum())))
            render_trend_bar(trend, lookback_days=dashboard_lookback, top_n=min(top_n, 100), rank_by=rank_by)

            display_cols = [
                "symbol", "tape", "unusual_trades", "active_days", "total_premium", "directional_premium",
                "bullish_premium", "bearish_premium", "neutral_premium", "net_premium", "bull_bear_score",
                "conviction_score", "avg_outlier_score", "avg_score", "avg_dte", "largest_print", "clusters", "last_seen",
            ]
            display_cols = [c for c in display_cols if c in trend.columns]
            st.dataframe(prep_trend_display(trend[display_cols]).head(top_n), use_container_width=True, hide_index=True)

        st.subheader("Multi-Window Scoreboard")
        if multi.empty:
            st.info("No multi-window score available yet.")
        else:
            st.dataframe(prep_trend_display(multi).head(top_n), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Loaded Outlier Tape")
    if flow_view.empty:
        st.info("No live or cached scan rows match the current dashboard filters. Run a live scan first or loosen dashboard filters.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(flow_view):,}")
        c2.metric("Premium", human_money(float(flow_view["premium"].sum())))
        c3.metric("Bullish premium", human_money(float(flow_view.loc[flow_view["direction"] == "BULLISH", "premium"].sum())))
        c4.metric("Bearish premium", human_money(float(flow_view.loc[flow_view["direction"] == "BEARISH", "premium"].sum())))

        chart = flow_view.sort_values(["outlier_score", "premium", "vol_oi"], ascending=[False, False, False]).head(min(top_n, 60)).copy()
        chart["label"] = (
            chart["symbol"].astype(str) + " | " + chart["type"].astype(str) + " | " + chart["expiry"].astype(str) + " | " + chart["strike"].round(0).astype("Int64").astype(str)
        )
        fig = px.bar(
            chart.sort_values(["outlier_score", "premium"], ascending=[True, True]),
            x="premium",
            y="label",
            color="direction",
            color_discrete_map=DIRECTION_COLOR_MAP,
            category_orders={"direction": ["BULLISH", "NEUTRAL", "MIXED", "BEARISH"]},
            orientation="h",
            hover_data={
                "symbol": True,
                "type": True,
                "expiry": True,
                "strike": ":.2f",
                "dte": True,
                "vol_oi": ":.2f",
                "spread_pct": ":.1f",
                "direction_confidence": ":.2f",
                "unusual_score": ":.1f",
                "outlier_score": ":.1f",
                "outlier_tier": True,
                "premium_adv_bps": ":.2f",
                "premium": ":,.0f",
            },
            title=f"Top Loaded Option Outliers | 0-{dashboard_dte_max}D Options | Dashboard Filters Applied",
        )
        fig.update_layout(height=max(420, min(900, 30 * len(chart) + 140)), margin=dict(l=20, r=20, t=55, b=20), yaxis_title="")
        st.plotly_chart(fig, use_container_width=True, key=f"today_flow_{dashboard_lookback}_{len(flow_view)}_{top_n}")
        st.dataframe(prep_flow_display(flow_view.sort_values(["outlier_score", "premium", "vol_oi"], ascending=[False, False, False]).head(top_n)), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Ticker Drilldown")
    if dashboard_db.empty:
        st.info("No saved database rows match the current dashboard filters.")
    else:
        symbols_in_db = sorted(dashboard_db["symbol"].dropna().astype(str).unique().tolist())
        ticker_pick = st.selectbox("Ticker", options=symbols_in_db, index=0 if symbols_in_db else None)
        if ticker_pick:
            sub = dashboard_db[dashboard_db["symbol"] == ticker_pick].copy().sort_values(["scan_ts", "outlier_score", "premium"], ascending=[False, False, False])
            sub_window = filter_to_lookback(sub, dashboard_lookback, anchor=anchor)
            render_ticker_timeline(dashboard_db, ticker_pick, dashboard_lookback, anchor=anchor)

            trend_selected = build_trend_board(dashboard_db[dashboard_db["symbol"] == ticker_pick], dashboard_lookback, anchor=anchor)
            trend_1 = build_trend_board(dashboard_db[dashboard_db["symbol"] == ticker_pick], 1, anchor=anchor)
            trend_5 = build_trend_board(dashboard_db[dashboard_db["symbol"] == ticker_pick], 5, anchor=anchor)
            trend_20 = build_trend_board(dashboard_db[dashboard_db["symbol"] == ticker_pick], 20, anchor=anchor)

            cols = st.columns(4)
            metric_frames = [(f"{dashboard_lookback}D", trend_selected), ("1D", trend_1), ("5D", trend_5), ("20D", trend_20)]
            for col, label, frame in zip(cols, [x[0] for x in metric_frames], [x[1] for x in metric_frames]):
                if frame.empty:
                    col.metric(f"{label} B/B Score", "N/A")
                else:
                    row = frame.iloc[0]
                    col.metric(f"{label} B/B Score", f"{row['bull_bear_score']:.1f}", delta=human_money(row["net_premium"]))
            st.dataframe(prep_flow_display(sub_window.head(top_n)), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Live Scan Diagnostics")
    if filtered_snapshot is not None and isinstance(filtered_snapshot, pd.DataFrame) and not filtered_snapshot.empty:
        st.caption(f"Underlying pre-filters reduced the universe from {len(symbols_all):,} names to {len(filtered_snapshot):,} names before option-chain scanning.")
        st.dataframe(filtered_snapshot[["symbol", "spot", "avg_20d_vol", "avg_dollar_vol"]], use_container_width=True, hide_index=True)
    if not diags.empty:
        sort_cols = [c for c in ["api_errors", "contracts_seen"] if c in diags.columns]
        st.dataframe(diags.sort_values(sort_cols, ascending=[False, False] if len(sort_cols) == 2 else False), use_container_width=True, hide_index=True)
        reason_counts = diags["reason"].fillna("unknown").replace("", "blank").value_counts().rename_axis("reason").reset_index(name="count")
        fig_diag = px.bar(reason_counts.sort_values("count", ascending=True), x="count", y="reason", orientation="h", title="Reason Breakdown")
        fig_diag.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20), yaxis_title="")
        st.plotly_chart(fig_diag, use_container_width=True, key=f"diag_reason_{len(diags)}")
    else:
        st.info("No diagnostic frame loaded yet. Run the live scan first.")

with tab5:
    st.subheader("Method Notes")
    st.markdown(
        """
        This app is a public-data approximation of a curated option-outlier service. Yahoo Finance provides option-chain snapshots: bid, ask, last price, volume, open interest, implied volatility, strike, and expiry. The app deliberately filters harder than a generic chain scanner: premium, volume/open-interest, outlier score, quote quality, aggressive side estimate, DTE, OTM distance, and premium versus underlying ADV.

        It does not provide exchange-grade time-and-sales, sweep/block tags, opening-versus-closing classification, or buyer/seller identity. Direction is therefore estimated conservatively. Ask-side call prints and bid-side put prints are treated as bullish; ask-side put prints and bid-side call prints are treated as bearish. Midpoint and one-sided quotes are neutral unless you change the source logic.

        The selected trend window is applied consistently across the Trend Board, chart, ticker drilldown, and ticker timeline. The selected option DTE max is applied to both database and loaded-flow views, so you can toggle quickly between 20, 30, 60, 90, 120, and 360-day contracts.
        """
    )

st.caption("© 2026 AD Fund Management LP")
