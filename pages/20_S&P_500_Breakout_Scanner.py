# pages/20_S&P_500_Breakout_Scanner.py

from __future__ import annotations

import io
import os
import time
import sqlite3
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="S&P 500 Breakout Scanner", layout="wide")
st.title("S&P 500 Breakout Scanner")

DEFAULT_LOOKBACK_DAYS = 220
DEFAULT_CONSOLIDATION_DAYS = 30
DEFAULT_BREAKOUT_BUFFER_PCT = 0.01
DEFAULT_VOL_SURGE_THRESHOLD = 1.20
DEFAULT_WATCHLIST_VOL_THRESHOLD = 1.15
DEFAULT_ROC_THRESHOLD_PCT = 1.0
DEFAULT_MIN_DOLLAR_VOLUME = 20_000_000

SCANNER_SECRETS = st.secrets.get("scanner", {})
LOOKBACK_DAYS = int(SCANNER_SECRETS.get("lookback_days", DEFAULT_LOOKBACK_DAYS))
CONSOLIDATION_DAYS = int(SCANNER_SECRETS.get("consolidation_days", DEFAULT_CONSOLIDATION_DAYS))
BREAKOUT_BUFFER_PCT = float(SCANNER_SECRETS.get("breakout_buffer_pct", DEFAULT_BREAKOUT_BUFFER_PCT))
VOL_SURGE_THRESHOLD = float(SCANNER_SECRETS.get("vol_surge_threshold", DEFAULT_VOL_SURGE_THRESHOLD))
WATCHLIST_VOL_THRESHOLD = float(SCANNER_SECRETS.get("watchlist_vol_threshold", DEFAULT_WATCHLIST_VOL_THRESHOLD))
ROC_THRESHOLD_PCT = float(SCANNER_SECRETS.get("roc_threshold_pct", DEFAULT_ROC_THRESHOLD_PCT))
MIN_DOLLAR_VOLUME = float(SCANNER_SECRETS.get("min_dollar_volume", DEFAULT_MIN_DOLLAR_VOLUME))

DB_PATH = os.path.join(tempfile.gettempdir(), "scanner.db")

def clamp_int(x: int, lo: int, hi: int) -> int:
    if hi < lo:
        return lo
    return max(lo, min(int(x), int(hi)))

# ============================== DB ==============================

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=30000;")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS signals (
            scan_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            direction TEXT,
            price REAL,
            volume_ratio REAL,
            roc REAL,
            consolidation_std REAL,
            consolidation_slope REAL,
            consolidation_max_dev REAL,
            dollar_volume_20d REAL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (scan_date, ticker, signal_type)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS forward_returns (
            scan_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            day_n INTEGER NOT NULL,
            return_pct REAL,
            PRIMARY KEY (scan_date, ticker, day_n)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS universe_cache (
            asof_utc TEXT NOT NULL,
            source TEXT NOT NULL,
            tickers_csv TEXT NOT NULL,
            PRIMARY KEY (asof_utc, source)
        )
        """
    )
    conn.commit()
    return conn

def db_cache_universe(conn: sqlite3.Connection, source: str, tickers: List[str]) -> None:
    if not tickers:
        return
    asof_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tickers_csv = ",".join(tickers)
    conn.execute(
        "INSERT INTO universe_cache (asof_utc, source, tickers_csv) VALUES (?, ?, ?)",
        (asof_utc, source, tickers_csv),
    )
    conn.commit()

def db_load_last_universe(conn: sqlite3.Connection) -> Optional[List[str]]:
    try:
        row = conn.execute(
            "SELECT tickers_csv FROM universe_cache ORDER BY asof_utc DESC LIMIT 1"
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if not row:
        return None
    tickers = [t.strip().upper().replace(".", "-") for t in row[0].split(",") if t.strip()]
    tickers = sorted(list(dict.fromkeys(tickers)))
    return tickers if tickers else None

def db_upsert_signals(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    cols = [
        "scan_date", "ticker", "signal_type", "direction", "price", "volume_ratio", "roc",
        "consolidation_std", "consolidation_slope", "consolidation_max_dev", "dollar_volume_20d",
        "created_at"
    ]
    df2 = df[cols].copy()
    conn.executemany(
        """
        INSERT OR REPLACE INTO signals
        (scan_date, ticker, signal_type, direction, price, volume_ratio, roc,
         consolidation_std, consolidation_slope, consolidation_max_dev, dollar_volume_20d, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        df2.itertuples(index=False, name=None)
    )
    conn.commit()

def db_upsert_forward_returns(conn: sqlite3.Connection, fr: pd.DataFrame) -> None:
    if fr is None or fr.empty:
        return
    df2 = fr[["scan_date", "ticker", "direction", "entry_price", "day_n", "return_pct"]].copy()
    conn.executemany(
        """
        INSERT OR REPLACE INTO forward_returns
        (scan_date, ticker, direction, entry_price, day_n, return_pct)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        df2.itertuples(index=False, name=None)
    )
    conn.commit()

def db_read_forward_returns(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM forward_returns ORDER BY scan_date DESC, ticker ASC, day_n ASC LIMIT 200000",
        conn
    )

# ============================== UNIVERSE ==============================

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def _http_get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")

def _parse_sp500_from_html(html: str) -> List[str]:
    tables = pd.read_html(io.StringIO(html))
    sp = tables[0]
    tickers = sp["Symbol"].astype(str).tolist()
    tickers = [t.strip().upper().replace(".", "-") for t in tickers if t and isinstance(t, str)]
    return sorted(list(dict.fromkeys(tickers)))

@st.cache_data(ttl=6 * 60 * 60)
def fetch_sp500_tickers_robust() -> Tuple[List[str], str]:
    last_err = None
    for attempt in range(1, 4):
        try:
            html = _http_get(WIKI_URL, timeout=30)
            tickers = _parse_sp500_from_html(html)
            if len(tickers) < 450:
                raise ValueError(f"Parsed unusually small universe ({len(tickers)})")
            return tickers, "Wikipedia (live)"
        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempt)
    return [], f"Wikipedia failed: {type(last_err).__name__}: {last_err}"

def parse_manual_tickers(text: str) -> List[str]:
    raw = text.replace("\n", ",").replace(" ", ",").split(",")
    tickers = [t.strip().upper() for t in raw if t.strip()]
    tickers = [t.replace(".", "-") for t in tickers]
    return sorted(list(dict.fromkeys(tickers)))

# ============================== DATA ==============================

@st.cache_data(ttl=60 * 60)
def download_ohlcv(tickers: List[str], lookback_days: int) -> pd.DataFrame:
    period = "400d" if lookback_days <= 260 else "730d"
    return yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="column",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

def normalize_download(df: pd.DataFrame, tickers: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))
        fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        if fields.issubset(lvl0):
            for t in tickers:
                if t not in lvl1:
                    continue
                sub = df.loc[:, pd.IndexSlice[:, t]].copy()
                sub.columns = [c[0] for c in sub.columns]
                sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna(how="any")
                if len(sub) > 0:
                    out[t] = sub
        else:
            for t in tickers:
                if t not in lvl0:
                    continue
                sub = df[t].copy()
                needed = ["Open", "High", "Low", "Close", "Volume"]
                if all(c in sub.columns for c in needed):
                    sub = sub[needed].dropna(how="any")
                    if len(sub) > 0:
                        out[t] = sub
    return out

# ============================== SIGNAL LOGIC ==============================

@dataclass
class ConsolidationMetrics:
    std: float
    slope: float
    max_dev: float

def compute_consolidation_metrics(prices: pd.Series) -> ConsolidationMetrics:
    mean = float(prices.mean())
    if mean <= 0:
        return ConsolidationMetrics(std=np.nan, slope=np.nan, max_dev=np.nan)
    std = float(prices.std(ddof=0) / mean)
    x = np.arange(len(prices), dtype=float)
    coeff = np.polyfit(x, prices.values.astype(float), 1)
    slope = float(abs(coeff[0]) / mean)
    max_dev = float(((prices - mean).abs() / mean).max())
    return ConsolidationMetrics(std=std, slope=slope, max_dev=max_dev)

def is_consolidating(df: pd.DataFrame, consolidation_days: int) -> Tuple[bool, ConsolidationMetrics]:
    if df is None or len(df) < consolidation_days + 15:
        return False, ConsolidationMetrics(std=np.nan, slope=np.nan, max_dev=np.nan)
    window = df.iloc[-(consolidation_days + 1):-1]
    m = compute_consolidation_metrics(window["Close"])
    return bool(m.std < 0.04 and m.slope < 0.0006 and m.max_dev < 0.065), m

def rising_volume_ratio(df: pd.DataFrame) -> Tuple[bool, float]:
    if df is None or len(df) < 18:
        return False, 0.0
    recent = float(df["Volume"].iloc[-3:].mean())
    base = float(df["Volume"].iloc[-13:-3].mean())
    if base <= 0:
        return False, 0.0
    ratio = recent / base
    return bool(ratio >= WATCHLIST_VOL_THRESHOLD), float(ratio)

def dollar_volume_20d(df: pd.DataFrame) -> float:
    if df is None or len(df) < 25:
        return 0.0
    return float((df["Close"].iloc[-20:] * df["Volume"].iloc[-20:]).mean())

def inside_consolidation(df: pd.DataFrame, consolidation_days: int) -> bool:
    cons = df.iloc[-(consolidation_days + 1):-1]
    hi = float(cons["High"].max())
    lo = float(cons["Low"].min())
    close = float(df["Close"].iloc[-1])
    return bool((close <= hi * (1.0 + BREAKOUT_BUFFER_PCT)) and (close >= lo * (1.0 - BREAKOUT_BUFFER_PCT)))

def breakout_today(df: pd.DataFrame, consolidation_days: int) -> Tuple[bool, Dict[str, float]]:
    cons = df.iloc[-(consolidation_days + 1):-1]
    hi = float(cons["High"].max())
    lo = float(cons["Low"].min())

    close = float(df["Close"].iloc[-1])
    vol = float(df["Volume"].iloc[-1])

    up = close > hi * (1.0 + BREAKOUT_BUFFER_PCT)
    down = close < lo * (1.0 - BREAKOUT_BUFFER_PCT)
    direction = "UP" if up else "DOWN" if down else "NONE"

    vol_base = float(df["Volume"].iloc[-11:-1].mean())
    vol_ratio = (vol / vol_base) if vol_base > 0 else 0.0
    vol_ok = vol_ratio >= VOL_SURGE_THRESHOLD

    past_close = float(df["Close"].iloc[-11])
    roc = ((close - past_close) / past_close) * 100.0 if past_close > 0 else 0.0
    roc_ok = (roc >= ROC_THRESHOLD_PCT) if direction == "UP" else (roc <= -ROC_THRESHOLD_PCT) if direction == "DOWN" else False

    ok = bool(direction != "NONE" and vol_ok and roc_ok)
    return ok, {"direction": direction, "volume_ratio": float(vol_ratio), "roc": float(roc)}

# ============================== CHARTS ==============================

def plot_candles(df: pd.DataFrame, ticker: str, consolidation_days: int, title: str) -> go.Figure:
    d = df.tail(90).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name=ticker))

    if len(d) >= consolidation_days + 2:
        cons = d.iloc[-(consolidation_days + 1):-1]
        cons_high = float(cons["High"].max())
        cons_low = float(cons["Low"].min())
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=cons.index[0],
            x1=cons.index[-1],
            y0=cons_low,
            y1=cons_high,
            line=dict(width=1),
            fillcolor="rgba(0, 0, 255, 0.08)",
        )

    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
    )
    return fig

# ============================== SCAN ==============================

def run_full_scan(tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    raw = download_ohlcv(tickers, LOOKBACK_DAYS)
    data = normalize_download(raw, tickers)

    scan_date = datetime.now(timezone.utc).date().isoformat()
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    breakout_rows = []
    watch_rows = []

    total = len(tickers)
    prog = st.progress(0)
    status = st.empty()

    for i, t in enumerate(tickers, start=1):
        df = data.get(t)
        if df is None or df.empty or len(df) < CONSOLIDATION_DAYS + 25:
            if i % 25 == 0 or i == total:
                prog.progress(int(i / total * 100))
                status.write(f"Scanning {i}/{total} ...")
            continue

        dv = dollar_volume_20d(df)
        if dv < MIN_DOLLAR_VOLUME:
            if i % 25 == 0 or i == total:
                prog.progress(int(i / total * 100))
                status.write(f"Scanning {i}/{total} ...")
            continue

        cons_ok, cons_m = is_consolidating(df, CONSOLIDATION_DAYS)
        if not cons_ok:
            if i % 25 == 0 or i == total:
                prog.progress(int(i / total * 100))
                status.write(f"Scanning {i}/{total} ...")
            continue

        brk_ok, brk_m = breakout_today(df, CONSOLIDATION_DAYS)
        vol_rising_ok, vol_rising_ratio = rising_volume_ratio(df)

        px = float(df["Close"].iloc[-1])

        if brk_ok:
            breakout_rows.append({
                "scan_date": scan_date,
                "ticker": t,
                "signal_type": "breakout",
                "direction": brk_m["direction"],
                "price": px,
                "volume_ratio": brk_m["volume_ratio"],
                "roc": brk_m["roc"],
                "consolidation_std": cons_m.std,
                "consolidation_slope": cons_m.slope,
                "consolidation_max_dev": cons_m.max_dev,
                "dollar_volume_20d": dv,
                "created_at": created_at,
            })

        if vol_rising_ok and inside_consolidation(df, CONSOLIDATION_DAYS) and (not brk_ok):
            watch_rows.append({
                "scan_date": scan_date,
                "ticker": t,
                "signal_type": "watchlist",
                "direction": None,
                "price": px,
                "volume_ratio": vol_rising_ratio,
                "roc": float("nan"),
                "consolidation_std": cons_m.std,
                "consolidation_slope": cons_m.slope,
                "consolidation_max_dev": cons_m.max_dev,
                "dollar_volume_20d": dv,
                "created_at": created_at,
            })

        if i % 25 == 0 or i == total:
            prog.progress(int(i / total * 100))
            status.write(f"Scanning {i}/{total} ...")

    status.write("Scan complete.")
    prog.progress(100)

    bdf = pd.DataFrame(breakout_rows)
    wdf = pd.DataFrame(watch_rows)

    if not bdf.empty:
        bdf["abs_roc"] = bdf["roc"].abs()
        bdf = bdf.sort_values(["abs_roc", "volume_ratio"], ascending=[False, False]).drop(columns=["abs_roc"])
    if not wdf.empty:
        wdf = wdf.sort_values(["volume_ratio", "dollar_volume_20d"], ascending=[False, False])

    return bdf, wdf, data

# ============================== SIDEBAR + UNIVERSE ==============================

conn = db_connect()

with st.sidebar:
    st.header("Scanner Controls")
    st.write(f"DB path: {DB_PATH}")

    manual_text = st.text_area(
        "Manual tickers (comma/space/newline). Leave blank for Wikipedia/DB fallback.",
        value="",
        height=90,
    )

    MIN_DOLLAR_VOLUME = float(
        st.number_input("Min avg daily $ volume (20d)", min_value=0.0, value=float(MIN_DOLLAR_VOLUME), step=5_000_000.0)
    )

    CONSOLIDATION_DAYS = st.slider("Consolidation window (sessions)", 20, 60, int(CONSOLIDATION_DAYS), 1)
    BREAKOUT_BUFFER_PCT = st.slider("Breakout buffer (%)", 0.0, 3.0, float(BREAKOUT_BUFFER_PCT * 100.0), 0.1) / 100.0
    VOL_SURGE_THRESHOLD = st.slider("Breakout volume threshold (x)", 1.0, 3.0, float(VOL_SURGE_THRESHOLD), 0.05)
    WATCHLIST_VOL_THRESHOLD = st.slider("Watchlist volume trend threshold (x)", 1.0, 2.0, float(WATCHLIST_VOL_THRESHOLD), 0.05)
    ROC_THRESHOLD_PCT = st.slider("ROC threshold (%)", 0.0, 5.0, float(ROC_THRESHOLD_PCT), 0.1)

def resolve_universe() -> Tuple[List[str], str]:
    manual = parse_manual_tickers(manual_text) if manual_text.strip() else []
    if manual:
        return manual, "Manual input"

    live, live_source = fetch_sp500_tickers_robust()
    if live:
        try:
            db_cache_universe(conn, "wikipedia", live)
        except sqlite3.OperationalError:
            pass
        return live, live_source

    cached = db_load_last_universe(conn)
    if cached:
        return cached, f"DB cache (fallback). {live_source}"

    return [], f"Universe unavailable. {live_source}"

tickers, universe_source = resolve_universe()
st.caption(f"Universe source: {universe_source}")
st.caption(f"Universe size: {len(tickers)}")
if not tickers:
    st.error("No universe available. Paste a manual ticker list in the sidebar to proceed.")
    st.stop()

# ============================== MAIN ==============================

colA, colB = st.columns([1, 1])
run_scan = colA.button("Run Scan", type="primary")
clear_cache = colB.button("Clear Streamlit cache")

if clear_cache:
    st.cache_data.clear()
    st.success("Cleared Streamlit cache. Reload the page.")

if "last_breakouts" not in st.session_state:
    st.session_state["last_breakouts"] = pd.DataFrame()
if "last_watch" not in st.session_state:
    st.session_state["last_watch"] = pd.DataFrame()
if "last_data" not in st.session_state:
    st.session_state["last_data"] = {}

if run_scan:
    st.info("Downloading data and scanning the full universe.")
    breakouts_df, watch_df, data_map = run_full_scan(tickers)
    st.session_state["last_breakouts"] = breakouts_df
    st.session_state["last_watch"] = watch_df
    st.session_state["last_data"] = data_map

breakouts_df = st.session_state.get("last_breakouts", pd.DataFrame())
watch_df = st.session_state.get("last_watch", pd.DataFrame())
data_map = st.session_state.get("last_data", {})

st.divider()
st.subheader("Charts")

c1, c2 = st.columns([1, 1])

with c1:
    st.write("Breakout charts")
    if breakouts_df is None or breakouts_df.empty:
        st.write("Run a scan to render charts.")
    else:
        max_n = min(25, len(breakouts_df))
        default_n = clamp_int(st.session_state.get("breakout_charts_n", min(8, max_n)), 1, max_n)
        top_n = st.slider("Breakout charts to render", 1, max_n, default_n, key="breakout_charts_n")
        for _, r in breakouts_df.head(top_n).iterrows():
            t = r["ticker"]
            df = data_map.get(t)
            if df is None or df.empty:
                continue
            fig = plot_candles(
                df,
                t,
                CONSOLIDATION_DAYS,
                title=f"{t} | {r['direction']} breakout | Close ${r['price']:.2f} | Vol {r['volume_ratio']:.2f}x | ROC {r['roc']:+.2f}%"
            )
            st.plotly_chart(fig, use_container_width=True)

with c2:
    st.write("Watchlist charts")
    if watch_df is None or watch_df.empty:
        st.write("Run a scan to render charts.")
    else:
        max_n2 = min(25, len(watch_df))
        default_n2 = clamp_int(st.session_state.get("watchlist_charts_n", min(8, max_n2)), 1, max_n2)
        top_n2 = st.slider("Watchlist charts to render", 1, max_n2, default_n2, key="watchlist_charts_n")
        for _, r in watch_df.head(top_n2).iterrows():
            t = r["ticker"]
            df = data_map.get(t)
            if df is None or df.empty:
                continue
            fig = plot_candles(
                df,
                t,
                CONSOLIDATION_DAYS,
                title=f"{t} | watchlist | Close ${r['price']:.2f} | VolTrend {r['volume_ratio']:.2f}x"
            )
            st.plotly_chart(fig, use_container_width=True)
