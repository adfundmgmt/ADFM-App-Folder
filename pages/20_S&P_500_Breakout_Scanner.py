# app.py
# S&P 500 Breakout Scanner (Streamlit)
#
# What this is:
# - Runs inside Streamlit and scans the full current S&P 500 universe (scraped from Wikipedia)
# - Uses batch yfinance download for speed and reliability
# - Produces two outputs:
#   1) Breakouts today: tight consolidation over the prior month, then a confirmed range break today with volume + momentum
#   2) Pre-break watchlist: still coiling inside the range, but volume trend is rising
# - Stores daily signals and forward returns in a local SQLite database (scanner.db)
#
# Notes:
# - Streamlit is not a scheduler. If you want it to run daily automatically, keep the app deployed and trigger it
#   via an external scheduler (cron/Task Scheduler) calling an endpoint, or just open the app and click Run Scan.
# - Do NOT hardcode credentials. If you enable email, use Streamlit secrets.
#
# Optional Streamlit secrets (create .streamlit/secrets.toml):
# [email]
# enabled = true
# sender_email = "you@gmail.com"
# sender_app_password = "your_gmail_app_password"
# recipients = ["you@gmail.com", "other@gmail.com"]
#
# [scanner]
# lookback_days = 220
# consolidation_days = 30
# breakout_buffer_pct = 0.01
# vol_surge_threshold = 1.20
# watchlist_vol_threshold = 1.15
# roc_threshold_pct = 1.0
# min_dollar_volume = 20000000
#
# Dependencies:
# pip install streamlit yfinance pandas numpy plotly lxml

from __future__ import annotations

import io
import os
import time
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ============================== UI + CONFIG ==============================

st.set_page_config(page_title="S&P 500 Breakout Scanner", layout="wide")

st.title("S&P 500 Breakout Scanner")

# Default config (can be overridden by st.secrets["scanner"])
DEFAULT_LOOKBACK_DAYS = 220  # trading sessions desired; we will request calendar buffer via period
DEFAULT_CONSOLIDATION_DAYS = 30
DEFAULT_BREAKOUT_BUFFER_PCT = 0.01
DEFAULT_VOL_SURGE_THRESHOLD = 1.20
DEFAULT_WATCHLIST_VOL_THRESHOLD = 1.15
DEFAULT_ROC_THRESHOLD_PCT = 1.0
DEFAULT_MIN_DOLLAR_VOLUME = 20_000_000  # avg daily $ volume filter

SCANNER_SECRETS = st.secrets.get("scanner", {})
LOOKBACK_DAYS = int(SCANNER_SECRETS.get("lookback_days", DEFAULT_LOOKBACK_DAYS))
CONSOLIDATION_DAYS = int(SCANNER_SECRETS.get("consolidation_days", DEFAULT_CONSOLIDATION_DAYS))
BREAKOUT_BUFFER_PCT = float(SCANNER_SECRETS.get("breakout_buffer_pct", DEFAULT_BREAKOUT_BUFFER_PCT))
VOL_SURGE_THRESHOLD = float(SCANNER_SECRETS.get("vol_surge_threshold", DEFAULT_VOL_SURGE_THRESHOLD))
WATCHLIST_VOL_THRESHOLD = float(SCANNER_SECRETS.get("watchlist_vol_threshold", DEFAULT_WATCHLIST_VOL_THRESHOLD))
ROC_THRESHOLD_PCT = float(SCANNER_SECRETS.get("roc_threshold_pct", DEFAULT_ROC_THRESHOLD_PCT))
MIN_DOLLAR_VOLUME = float(SCANNER_SECRETS.get("min_dollar_volume", DEFAULT_MIN_DOLLAR_VOLUME))

DB_PATH = "scanner.db"


# ============================== DB ==============================

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS signals (
            scan_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            signal_type TEXT NOT NULL,  -- 'breakout' or 'watchlist'
            direction TEXT,             -- 'UP', 'DOWN', or NULL
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
    conn.commit()
    return conn


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


def db_read_recent(conn: sqlite3.Connection, limit_days: int = 30) -> pd.DataFrame:
    q = """
    SELECT * FROM signals
    ORDER BY scan_date DESC, signal_type ASC, ticker ASC
    LIMIT 20000
    """
    return pd.read_sql_query(q, conn)


def db_read_forward_returns(conn: sqlite3.Connection, limit_days: int = 30) -> pd.DataFrame:
    q = """
    SELECT * FROM forward_returns
    ORDER BY scan_date DESC, ticker ASC, day_n ASC
    LIMIT 200000
    """
    return pd.read_sql_query(q, conn)


# ============================== UNIVERSE ==============================

@st.cache_data(ttl=6 * 60 * 60)
def fetch_sp500_tickers() -> List[str]:
    # Uses Wikipedia; pandas.read_html needs lxml installed.
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp = tables[0]
    tickers = sp["Symbol"].astype(str).tolist()
    # Yahoo uses '-' for '.' tickers
    tickers = [t.replace(".", "-").strip().upper() for t in tickers if isinstance(t, str)]
    tickers = sorted(list(dict.fromkeys(tickers)))
    return tickers


# ============================== DATA PULL ==============================

@st.cache_data(ttl=60 * 60)
def download_ohlcv(tickers: List[str], lookback_days: int) -> pd.DataFrame:
    """
    Batch download via yfinance.
    Returns a DataFrame with columns MultiIndex:
      (Field, Ticker) or (Ticker, Field) depending on yfinance version.
    We will normalize downstream.
    """
    # Request extra calendar buffer since "days" are trading sessions.
    # 400d period is usually enough to get ~220 trading sessions.
    period = "400d" if lookback_days <= 260 else "730d"
    df = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="column",
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    return df


def normalize_download(df: pd.DataFrame, tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Produce dict[ticker] -> OHLCV DataFrame with columns: Open, High, Low, Close, Volume
    """
    out: Dict[str, pd.DataFrame] = {}

    if df is None or df.empty:
        return out

    # Two common formats:
    # 1) Columns MultiIndex: (Field, Ticker)
    # 2) Columns MultiIndex: (Ticker, Field)
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))
        fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        if fields.issubset(lvl0):
            # (Field, Ticker)
            for t in tickers:
                if t not in lvl1:
                    continue
                sub = df.loc[:, pd.IndexSlice[:, t]].copy()
                sub.columns = [c[0] for c in sub.columns]
                sub = sub.rename(columns={"Adj Close": "AdjClose"})
                sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna(how="any")
                if len(sub) > 0:
                    out[t] = sub
        else:
            # (Ticker, Field)
            for t in tickers:
                if t not in lvl0:
                    continue
                sub = df[t].copy()
                if "Adj Close" in sub.columns:
                    sub = sub.rename(columns={"Adj Close": "AdjClose"})
                needed = ["Open", "High", "Low", "Close", "Volume"]
                missing = [c for c in needed if c not in sub.columns]
                if missing:
                    continue
                sub = sub[needed].dropna(how="any")
                if len(sub) > 0:
                    out[t] = sub
    else:
        # Single ticker fallback case
        needed = {"Open", "High", "Low", "Close", "Volume"}
        if needed.issubset(set(df.columns)):
            t = tickers[0] if tickers else "TICKER"
            out[t] = df[list(needed)].dropna(how="any")

    return out


# ============================== SCORING ==============================

@dataclass
class ConsolidationMetrics:
    std: float
    slope: float
    max_dev: float


def compute_consolidation_metrics(prices: pd.Series) -> ConsolidationMetrics:
    price_mean = float(prices.mean())
    if price_mean <= 0:
        return ConsolidationMetrics(std=np.nan, slope=np.nan, max_dev=np.nan)

    price_std = float(prices.std(ddof=0) / price_mean)

    x = np.arange(len(prices), dtype=float)
    coeff = np.polyfit(x, prices.values.astype(float), 1)
    slope = float(abs(coeff[0]) / price_mean)

    dev = (prices - price_mean).abs() / price_mean
    max_dev = float(dev.max())

    return ConsolidationMetrics(std=price_std, slope=slope, max_dev=max_dev)


def is_consolidating(df: pd.DataFrame, consolidation_days: int) -> Tuple[bool, ConsolidationMetrics]:
    # Prior consolidation window excludes today
    if df is None or len(df) < consolidation_days + 15:
        return False, ConsolidationMetrics(std=np.nan, slope=np.nan, max_dev=np.nan)

    window = df.iloc[-(consolidation_days + 1):-1]
    prices = window["Close"]
    m = compute_consolidation_metrics(prices)

    low_volatility = m.std < 0.04
    sideways = m.slope < 0.0006
    near_mean = m.max_dev < 0.065

    ok = bool(low_volatility and sideways and near_mean)
    return ok, m


def rising_volume_ratio(df: pd.DataFrame) -> Tuple[bool, float]:
    # Last 3 sessions vs prior 10 sessions
    if df is None or len(df) < 3 + 10 + 5:
        return False, 0.0

    recent = float(df["Volume"].iloc[-3:].mean())
    base = float(df["Volume"].iloc[-13:-3].mean())
    if base <= 0:
        return False, 0.0
    ratio = recent / base
    return bool(ratio >= WATCHLIST_VOL_THRESHOLD), float(ratio)


def breakout_today(df: pd.DataFrame, consolidation_days: int) -> Tuple[bool, Dict[str, float]]:
    """
    Confirmed breakout uses CLOSE, not wick.
    - Range defined by prior consolidation window (excludes today)
    - Upside breakout: today's close > prior high * (1 + buffer)
    - Downside breakout: today's close < prior low  * (1 - buffer)
    - Volume surge: today's volume >= avg volume prior 10 sessions * threshold
    - Momentum: 10 session ROC in breakout direction exceeds threshold
    """
    if df is None or len(df) < consolidation_days + 20:
        return False, {}

    cons = df.iloc[-(consolidation_days + 1):-1]
    cons_high = float(cons["High"].max())
    cons_low = float(cons["Low"].min())

    today = df.iloc[-1]
    close = float(today["Close"])
    vol = float(today["Volume"])

    up = close > cons_high * (1.0 + BREAKOUT_BUFFER_PCT)
    down = close < cons_low * (1.0 - BREAKOUT_BUFFER_PCT)
    direction = "UP" if up else "DOWN" if down else "NONE"

    # Volume surge: today vs prior 10 sessions excluding today
    vol_base = float(df["Volume"].iloc[-11:-1].mean())
    if vol_base <= 0:
        vol_ratio = 0.0
        vol_ok = False
    else:
        vol_ratio = vol / vol_base
        vol_ok = vol_ratio >= VOL_SURGE_THRESHOLD

    # 10 session ROC
    past_close = float(df["Close"].iloc[-11])
    roc = ((close - past_close) / past_close) * 100.0 if past_close > 0 else 0.0
    if direction == "UP":
        roc_ok = roc >= ROC_THRESHOLD_PCT
    elif direction == "DOWN":
        roc_ok = roc <= -ROC_THRESHOLD_PCT
    else:
        roc_ok = False

    price_ok = (direction != "NONE")
    ok = bool(price_ok and vol_ok and roc_ok)

    return ok, {
        "direction": direction,
        "volume_ratio": float(vol_ratio),
        "roc": float(roc),
        "cons_high": cons_high,
        "cons_low": cons_low,
    }


def inside_consolidation(df: pd.DataFrame, consolidation_days: int) -> bool:
    if df is None or len(df) < consolidation_days + 5:
        return False
    cons = df.iloc[-(consolidation_days + 1):-1]
    cons_high = float(cons["High"].max())
    cons_low = float(cons["Low"].min())
    close = float(df["Close"].iloc[-1])
    return bool((close <= cons_high * (1.0 + BREAKOUT_BUFFER_PCT)) and (close >= cons_low * (1.0 - BREAKOUT_BUFFER_PCT)))


def dollar_volume_20d(df: pd.DataFrame) -> float:
    if df is None or len(df) < 25:
        return 0.0
    px = df["Close"].iloc[-20:]
    vol = df["Volume"].iloc[-20:]
    dv = float((px * vol).mean())
    return dv


# ============================== FORWARD RETURNS (TRADING SESSIONS) ==============================

def compute_forward_returns_for_signals(
    data: Dict[str, pd.DataFrame],
    breakouts: pd.DataFrame,
    max_days: int = 10
) -> pd.DataFrame:
    """
    Uses the already-downloaded OHLCV history and computes forward returns
    by stepping through the next available trading sessions in the series.
    """
    if breakouts is None or breakouts.empty:
        return pd.DataFrame()

    rows = []
    for _, r in breakouts.iterrows():
        t = r["ticker"]
        if t not in data:
            continue
        df = data[t].copy()
        if df.empty:
            continue

        # Signal is anchored to the last bar in this scan.
        entry_idx = len(df) - 1
        entry_price = float(df["Close"].iloc[entry_idx])
        direction = str(r.get("direction", "UP"))

        for day_n in range(1, max_days + 1):
            idx = entry_idx + day_n
            if idx >= len(df):
                break
            px = float(df["Close"].iloc[idx])
            ret = ((px - entry_price) / entry_price) * 100.0 if entry_price > 0 else np.nan
            rows.append({
                "scan_date": r["scan_date"],
                "ticker": t,
                "direction": direction,
                "entry_price": entry_price,
                "day_n": int(day_n),
                "return_pct": float(ret),
            })

    return pd.DataFrame(rows)


# ============================== CHARTING ==============================

def plot_candles(df: pd.DataFrame, ticker: str, consolidation_days: int, title: str) -> go.Figure:
    d = df.tail(90).copy()
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=d.index,
            open=d["Open"],
            high=d["High"],
            low=d["Low"],
            close=d["Close"],
            name=ticker,
        )
    )

    if len(d) >= consolidation_days + 2:
        cons = d.iloc[-(consolidation_days + 1):-1]
        cons_high = float(cons["High"].max())
        cons_low = float(cons["Low"].min())
        x0 = cons.index[0]
        x1 = cons.index[-1]

        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=cons_low,
            y1=cons_high,
            line=dict(width=1),
            fillcolor="rgba(0, 0, 255, 0.08)",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
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

    # Scan all tickers, no early exit
    for i, t in enumerate(tickers, start=1):
        if t not in data:
            if i % 25 == 0 or i == total:
                prog.progress(int(i / total * 100))
                status.write(f"Scanning {i}/{total} ...")
            continue

        df = data[t]
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

        # Breakout (confirmed by close)
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

        # Watchlist: still inside the box, volume rising
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

    breakouts_df = pd.DataFrame(breakout_rows)
    watch_df = pd.DataFrame(watch_rows)

    # Sort
    if not breakouts_df.empty:
        breakouts_df["abs_roc"] = breakouts_df["roc"].abs()
        breakouts_df = breakouts_df.sort_values(["abs_roc", "volume_ratio"], ascending=[False, False]).drop(columns=["abs_roc"])

    if not watch_df.empty:
        watch_df = watch_df.sort_values(["volume_ratio", "dollar_volume_20d"], ascending=[False, False])

    return breakouts_df, watch_df, data


# ============================== OPTIONAL EMAIL ==============================

def email_settings() -> Dict[str, object]:
    e = st.secrets.get("email", {})
    enabled = bool(e.get("enabled", False))
    sender = str(e.get("sender_email", "")).strip()
    password = str(e.get("sender_app_password", "")).strip()
    recipients = e.get("recipients", [])
    if isinstance(recipients, str):
        recipients = [x.strip() for x in recipients.split(",") if x.strip()]
    return {"enabled": enabled, "sender": sender, "password": password, "recipients": recipients}


def send_email_report(subject: str, body: str) -> Tuple[bool, str]:
    # Optional, only if secrets exist. We keep code minimal and safe.
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    es = email_settings()
    if not es["enabled"]:
        return False, "Email disabled (enable it in secrets.toml)."
    if not es["sender"] or not es["password"] or not es["recipients"]:
        return False, "Email secrets missing."

    msg = MIMEMultipart()
    msg["From"] = es["sender"]
    msg["To"] = ", ".join(es["recipients"])
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(es["sender"], es["password"])
        server.send_message(msg)
        server.quit()
        return True, "Email sent."
    except Exception as ex:
        return False, f"Email failed: {ex}"


def format_report(breakouts: pd.DataFrame, watch: pd.DataFrame) -> str:
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nb = 0 if breakouts is None else len(breakouts)
    nw = 0 if watch is None else len(watch)

    lines = []
    lines.append(f"S&P 500 Breakout Scanner Report")
    lines.append(f"Run: {dt}")
    lines.append("")
    lines.append(f"Breakouts: {nb}")
    lines.append(f"Watchlist: {nw}")
    lines.append("")

    if breakouts is not None and not breakouts.empty:
        lines.append("BREAKOUTS")
        for _, r in breakouts.iterrows():
            lines.append(
                f"{r['ticker']} | {r['direction']} | ${r['price']:.2f} | Vol {r['volume_ratio']:.2f}x | ROC {r['roc']:+.2f}% | $Vol20d {r['dollar_volume_20d'] / 1e6:.1f}M"
            )
        lines.append("")

    if watch is not None and not watch.empty:
        lines.append("WATCHLIST")
        for _, r in watch.iterrows():
            lines.append(
                f"{r['ticker']} | ${r['price']:.2f} | VolTrend {r['volume_ratio']:.2f}x | $Vol20d {r['dollar_volume_20d'] / 1e6:.1f}M"
            )
        lines.append("")

    return "\n".join(lines)


# ============================== SIDEBAR ==============================

with st.sidebar:
    st.header("Scanner Controls")

    st.write("Universe: current S&P 500 constituents (Wikipedia).")
    st.write("Data: daily OHLCV from Yahoo via yfinance.")
    st.write("Signals are stored in scanner.db.")

    st.subheader("Filters")
    min_dv = st.number_input("Min avg daily $ volume (20d)", min_value=0.0, value=float(MIN_DOLLAR_VOLUME), step=5_000_000.0)
    MIN_DOLLAR_VOLUME = float(min_dv)

    st.subheader("Parameters")
    CONSOLIDATION_DAYS = st.slider("Consolidation window (sessions)", min_value=20, max_value=60, value=int(CONSOLIDATION_DAYS), step=1)
    BREAKOUT_BUFFER_PCT = st.slider("Breakout buffer (%)", min_value=0.0, max_value=3.0, value=float(BREAKOUT_BUFFER_PCT * 100.0), step=0.1) / 100.0
    VOL_SURGE_THRESHOLD = st.slider("Breakout volume threshold (x)", min_value=1.0, max_value=3.0, value=float(VOL_SURGE_THRESHOLD), step=0.05)
    WATCHLIST_VOL_THRESHOLD = st.slider("Watchlist volume trend threshold (x)", min_value=1.0, max_value=2.0, value=float(WATCHLIST_VOL_THRESHOLD), step=0.05)
    ROC_THRESHOLD_PCT = st.slider("ROC threshold (%)", min_value=0.0, max_value=5.0, value=float(ROC_THRESHOLD_PCT), step=0.1)

    st.subheader("Email")
    es = email_settings()
    st.write(f"Enabled: {es['enabled']}")
    if es["enabled"]:
        st.write(f"Sender: {es['sender']}")
        st.write(f"Recipients: {', '.join(es['recipients']) if es['recipients'] else '(none)'}")


# ============================== MAIN ==============================

colA, colB = st.columns([1, 1])

with colA:
    run_scan = st.button("Run Scan", type="primary")
with colB:
    load_latest = st.button("Load Latest From DB")

tickers = fetch_sp500_tickers()
st.caption(f"Universe size: {len(tickers)} tickers")

conn = db_connect()

if "last_breakouts" not in st.session_state:
    st.session_state["last_breakouts"] = pd.DataFrame()
if "last_watch" not in st.session_state:
    st.session_state["last_watch"] = pd.DataFrame()
if "last_data" not in st.session_state:
    st.session_state["last_data"] = {}

if run_scan:
    st.info("Downloading data and scanning the full S&P 500 universe.")
    breakouts_df, watch_df, data_map = run_full_scan(tickers)

    st.session_state["last_breakouts"] = breakouts_df
    st.session_state["last_watch"] = watch_df
    st.session_state["last_data"] = data_map

    # Persist signals
    combined = pd.concat([breakouts_df, watch_df], ignore_index=True) if (not breakouts_df.empty or not watch_df.empty) else pd.DataFrame()
    if not combined.empty:
        db_upsert_signals(conn, combined)

    # Persist forward returns for breakouts (based on already-downloaded series)
    if breakouts_df is not None and not breakouts_df.empty:
        fr = compute_forward_returns_for_signals(data_map, breakouts_df, max_days=10)
        if not fr.empty:
            db_upsert_forward_returns(conn, fr)

    # Optional email, without attachments (Streamlit friendly)
    report = format_report(breakouts_df, watch_df)
    st.text_area("Run Report (for copy/paste)", report, height=240)
    if es["enabled"]:
        if st.button("Send Email Report"):
            subject = f"Breakout Scanner | {len(breakouts_df)} breakouts, {len(watch_df)} watchlist | {datetime.now().strftime('%Y-%m-%d')}"
            ok, msg = send_email_report(subject, report)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

if load_latest:
    st.info("Loading recent signals from the local database.")
    recent = db_read_recent(conn)
    st.dataframe(recent, use_container_width=True)

# Display last scan results if present
breakouts_df = st.session_state.get("last_breakouts", pd.DataFrame())
watch_df = st.session_state.get("last_watch", pd.DataFrame())
data_map = st.session_state.get("last_data", {})

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Breakouts Today")
    if breakouts_df is None or breakouts_df.empty:
        st.write("No breakouts found in the last run.")
    else:
        st.dataframe(
            breakouts_df[[
                "ticker", "direction", "price", "volume_ratio", "roc",
                "consolidation_std", "consolidation_slope", "consolidation_max_dev",
                "dollar_volume_20d"
            ]],
            use_container_width=True
        )

        csv = breakouts_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Breakouts CSV", data=csv, file_name="breakouts.csv", mime="text/csv")

with right:
    st.subheader("Pre-break Watchlist")
    if watch_df is None or watch_df.empty:
        st.write("No watchlist names in the last run.")
    else:
        st.dataframe(
            watch_df[[
                "ticker", "price", "volume_ratio",
                "consolidation_std", "consolidation_slope", "consolidation_max_dev",
                "dollar_volume_20d"
            ]],
            use_container_width=True
        )

        csv2 = watch_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Watchlist CSV", data=csv2, file_name="watchlist.csv", mime="text/csv")

st.divider()

st.subheader("Charts")

chart_cols = st.columns([1, 1])

with chart_cols[0]:
    st.write("Breakout charts")
    if breakouts_df is not None and not breakouts_df.empty:
        top_n = st.slider("Breakout charts to render", min_value=1, max_value=min(25, len(breakouts_df)), value=min(8, len(breakouts_df)))
        for _, r in breakouts_df.head(top_n).iterrows():
            t = r["ticker"]
            if t in data_map:
                fig = plot_candles(
                    data_map[t],
                    t,
                    CONSOLIDATION_DAYS,
                    title=f"{t} | {r['direction']} breakout | Close ${r['price']:.2f} | Vol {r['volume_ratio']:.2f}x | ROC {r['roc']:+.2f}%"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Run a scan to render charts.")

with chart_cols[1]:
    st.write("Watchlist charts")
    if watch_df is not None and not watch_df.empty:
        top_n2 = st.slider("Watchlist charts to render", min_value=1, max_value=min(25, len(watch_df)), value=min(8, len(watch_df)))
        for _, r in watch_df.head(top_n2).iterrows():
            t = r["ticker"]
            if t in data_map:
                fig = plot_candles(
                    data_map[t],
                    t,
                    CONSOLIDATION_DAYS,
                    title=f"{t} | watchlist | Close ${r['price']:.2f} | VolTrend {r['volume_ratio']:.2f}x"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Run a scan to render charts.")

st.divider()

st.subheader("Forward Return Tracking (Breakouts)")
fr_db = db_read_forward_returns(conn)
if fr_db is None or fr_db.empty:
    st.write("No forward returns recorded yet. Run scans on multiple days to build history.")
else:
    # Summarize by scan_date
    fr_db["scan_date"] = pd.to_datetime(fr_db["scan_date"])
    latest_dates = sorted(fr_db["scan_date"].dt.date.unique(), reverse=True)[:10]
    pick_date = st.selectbox("Select scan date", latest_dates)
    subset = fr_db[fr_db["scan_date"].dt.date == pick_date].copy()
    st.dataframe(subset, use_container_width=True)

    # Simple summary: Day 10 stats where available
    d10 = subset[subset["day_n"] == 10].dropna(subset=["return_pct"])
    if len(d10) > 0:
        avg = float(d10["return_pct"].mean())
        win = float((d10["return_pct"] > 0).mean() * 100.0)
        st.write(f"Day 10 average return: {avg:+.2f}% | win rate: {win:.1f}% | sample: {len(d10)}")
    else:
        st.write("No Day 10 observations for that date yet.")


st.caption(
    "Implementation details: consolidation uses normalized close std, normalized linear slope, and max deviation over the prior window; "
    "breakout confirmation uses close beyond the range with a buffer, plus volume surge and 10 session ROC filter; "
    "watchlist requires still inside the range and rising 3 session volume versus prior 10 sessions."
)
