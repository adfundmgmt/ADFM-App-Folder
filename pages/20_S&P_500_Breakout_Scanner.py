# pages/20_S&P_500_Breakout_Scanner.py
# Minimal S&P 500 Breakout Scanner (Streamlit)
# Outputs: tables + CSV downloads, no charts, no email, no SQLite.

from __future__ import annotations

import io
import time
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="S&P 500 Breakout Scanner", layout="wide")
st.title("S&P 500 Breakout Scanner (Minimal)")

# ----------------------------
# Parameters
# ----------------------------
DEFAULT_LOOKBACK_DAYS = 260
DEFAULT_CONSOL_DAYS = 30
DEFAULT_BREAKOUT_BUFFER_PCT = 1.0
DEFAULT_VOL_SURGE = 1.20
DEFAULT_WATCH_VOL = 1.15
DEFAULT_ROC_PCT = 1.0
DEFAULT_MIN_DOLLAR_VOL = 20_000_000
DEFAULT_BATCH_SIZE = 60

with st.sidebar:
    st.header("Controls")

    manual_universe = st.text_area(
        "Manual tickers (optional). Comma, space, or newline separated.",
        value="",
        height=90,
    )

    st.subheader("Scan settings")
    lookback_days = st.slider("Lookback (calendar days)", 120, 900, DEFAULT_LOOKBACK_DAYS, 10)
    consol_days = st.slider("Consolidation window (sessions)", 20, 60, DEFAULT_CONSOL_DAYS, 1)

    st.subheader("Breakout rules")
    breakout_buffer_pct = st.slider("Breakout buffer (%)", 0.0, 3.0, DEFAULT_BREAKOUT_BUFFER_PCT, 0.1) / 100.0
    vol_surge = st.slider("Volume surge threshold (x)", 1.0, 3.0, DEFAULT_VOL_SURGE, 0.05)
    roc_threshold = st.slider("ROC threshold (%)", 0.0, 5.0, DEFAULT_ROC_PCT, 0.1)

    st.subheader("Watchlist rules")
    watch_vol = st.slider("Rising volume threshold (x)", 1.0, 2.0, DEFAULT_WATCH_VOL, 0.05)

    st.subheader("Liquidity filter")
    min_dollar_vol = st.number_input("Min avg daily $ volume (20d)", min_value=0.0, value=float(DEFAULT_MIN_DOLLAR_VOL), step=5_000_000.0)

    st.subheader("yfinance batching")
    batch_size = st.slider("Batch size", 20, 120, DEFAULT_BATCH_SIZE, 5)

    run_scan = st.button("Run scan", type="primary")

# ----------------------------
# Helpers
# ----------------------------

def parse_tickers(text: str) -> List[str]:
    raw = text.replace("\n", ",").replace(" ", ",").split(",")
    tickers = [t.strip().upper() for t in raw if t.strip()]
    tickers = [t.replace(".", "-") for t in tickers]
    tickers = list(dict.fromkeys(tickers))
    return tickers

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def http_get(url: str, timeout: int = 30) -> str:
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

@st.cache_data(ttl=6 * 60 * 60)
def fetch_sp500_tickers() -> Tuple[List[str], str]:
    last_err = None
    for attempt in range(1, 4):
        try:
            html = http_get(WIKI_URL, timeout=30)
            tables = pd.read_html(io.StringIO(html))
            sp = tables[0]
            if "Symbol" not in sp.columns:
                raise ValueError("Wikipedia table missing Symbol column")
            tickers = sp["Symbol"].astype(str).tolist()
            tickers = [t.strip().upper().replace(".", "-") for t in tickers if t]
            tickers = list(dict.fromkeys(tickers))
            if len(tickers) < 450:
                raise ValueError(f"Parsed unusually small universe: {len(tickers)}")
            return tickers, "Wikipedia (live)"
        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempt)
    return [], f"Wikipedia failed: {type(last_err).__name__}: {last_err}"

def chunk_list(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i:i+n] for i in range(0, len(xs), n)]

def normalize_download(df: pd.DataFrame, tickers: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out

    # yfinance can return MultiIndex with either (Field, Ticker) or (Ticker, Field)
    if not isinstance(df.columns, pd.MultiIndex):
        return out

    lvl0 = list(df.columns.get_level_values(0))
    lvl1 = list(df.columns.get_level_values(1))
    lvl0_set = set(lvl0)
    fields = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}

    # Case A: (Field, Ticker)
    if fields.issubset(lvl0_set):
        for t in tickers:
            if t not in set(lvl1):
                continue
            sub = df.loc[:, pd.IndexSlice[:, t]].copy()
            sub.columns = [c[0] for c in sub.columns]
            needed = ["Open", "High", "Low", "Close", "Volume"]
            if all(c in sub.columns for c in needed):
                sub = sub[needed].dropna(how="any")
                if len(sub) > 0:
                    out[t] = sub
        return out

    # Case B: (Ticker, Field)
    for t in tickers:
        if t not in lvl0_set:
            continue
        sub = df[t].copy()
        needed = ["Open", "High", "Low", "Close", "Volume"]
        if all(c in sub.columns for c in needed):
            sub = sub[needed].dropna(how="any")
            if len(sub) > 0:
                out[t] = sub
    return out

def consolidation_metrics(closes: pd.Series) -> Tuple[float, float, float]:
    mean = float(closes.mean())
    if mean <= 0:
        return np.nan, np.nan, np.nan
    std = float(closes.std(ddof=0) / mean)
    x = np.arange(len(closes), dtype=float)
    slope = float(abs(np.polyfit(x, closes.values.astype(float), 1)[0]) / mean)
    max_dev = float(((closes - mean).abs() / mean).max())
    return std, slope, max_dev

def is_consolidating(df: pd.DataFrame, window: int) -> Tuple[bool, Dict[str, float]]:
    if df is None or len(df) < window + 15:
        return False, {}
    cons = df.iloc[-(window + 1):-1]
    std, slope, max_dev = consolidation_metrics(cons["Close"])
    ok = bool((std < 0.04) and (slope < 0.0006) and (max_dev < 0.065))
    return ok, {"std": std, "slope": slope, "max_dev": max_dev}

def dollar_volume_20d(df: pd.DataFrame) -> float:
    if df is None or len(df) < 25:
        return 0.0
    return float((df["Close"].iloc[-20:] * df["Volume"].iloc[-20:]).mean())

def rising_volume(df: pd.DataFrame, threshold: float) -> Tuple[bool, float]:
    if df is None or len(df) < 18:
        return False, 0.0
    recent = float(df["Volume"].iloc[-3:].mean())
    base = float(df["Volume"].iloc[-13:-3].mean())
    if base <= 0:
        return False, 0.0
    ratio = recent / base
    return bool(ratio >= threshold), float(ratio)

def inside_consolidation_range(df: pd.DataFrame, window: int, buffer_pct: float) -> bool:
    if df is None or len(df) < window + 5:
        return False
    cons = df.iloc[-(window + 1):-1]
    hi = float(cons["High"].max())
    lo = float(cons["Low"].min())
    close = float(df["Close"].iloc[-1])
    return bool((close <= hi * (1.0 + buffer_pct)) and (close >= lo * (1.0 - buffer_pct)))

def breakout_today(df: pd.DataFrame, window: int, buffer_pct: float, vol_surge_x: float, roc_pct: float) -> Tuple[bool, Dict[str, float]]:
    if df is None or len(df) < window + 20:
        return False, {}
    cons = df.iloc[-(window + 1):-1]
    hi = float(cons["High"].max())
    lo = float(cons["Low"].min())

    close = float(df["Close"].iloc[-1])
    vol = float(df["Volume"].iloc[-1])

    up = close > hi * (1.0 + buffer_pct)
    down = close < lo * (1.0 - buffer_pct)
    direction = "UP" if up else "DOWN" if down else "NONE"

    base_vol = float(df["Volume"].iloc[-11:-1].mean())
    vol_ratio = (vol / base_vol) if base_vol > 0 else 0.0
    vol_ok = bool(vol_ratio >= vol_surge_x)

    past_close = float(df["Close"].iloc[-11])
    roc = ((close - past_close) / past_close) * 100.0 if past_close > 0 else 0.0
    roc_ok = bool((roc >= roc_pct) if direction == "UP" else (roc <= -roc_pct) if direction == "DOWN" else False)

    ok = bool(direction != "NONE" and vol_ok and roc_ok)
    return ok, {"direction": direction, "volume_ratio": float(vol_ratio), "roc": float(roc)}

# ----------------------------
# Universe
# ----------------------------
manual = parse_tickers(manual_universe) if manual_universe.strip() else []
if manual:
    tickers = manual
    universe_source = "Manual"
else:
    tickers, universe_source = fetch_sp500_tickers()

st.caption(f"Universe source: {universe_source} | Size: {len(tickers)}")

if not tickers:
    st.error("No universe available. Paste tickers manually in the sidebar and rerun.")
    st.stop()

# ----------------------------
# Run scan
# ----------------------------
if not run_scan:
    st.info("Set your parameters in the sidebar, then click Run scan.")
    st.stop()

start_ts = datetime.now(timezone.utc)
st.write(f"Started: {start_ts.isoformat(timespec='seconds')} UTC")

progress = st.progress(0)
status = st.empty()

breakouts_rows = []
watch_rows = []

batches = chunk_list(tickers, batch_size)
total_batches = len(batches)

for bi, batch in enumerate(batches, start=1):
    status.write(f"Downloading batch {bi}/{total_batches} (size {len(batch)})")
    try:
        df_raw = yf.download(
            tickers=batch,
            period=f"{max(lookback_days, 200)}d",
            interval="1d",
            group_by="column",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
    except Exception:
        df_raw = pd.DataFrame()

    data_map = normalize_download(df_raw, batch)

    for t in batch:
        df = data_map.get(t)
        if df is None or df.empty:
            continue
        if len(df) < consol_days + 25:
            continue

        dv20 = dollar_volume_20d(df)
        if dv20 < float(min_dollar_vol):
            continue

        cons_ok, cons_m = is_consolidating(df, consol_days)
        if not cons_ok:
            continue

        brk_ok, brk_m = breakout_today(df, consol_days, breakout_buffer_pct, vol_surge, roc_threshold)
        vol_rising_ok, vol_rising_ratio = rising_volume(df, watch_vol)

        px = float(df["Close"].iloc[-1])
        asof = df.index[-1].date().isoformat()

        if brk_ok:
            breakouts_rows.append({
                "ticker": t,
                "asof": asof,
                "direction": brk_m["direction"],
                "close": px,
                "volume_ratio": brk_m["volume_ratio"],
                "roc_10d_pct": brk_m["roc"],
                "cons_std": cons_m["std"],
                "cons_slope": cons_m["slope"],
                "cons_max_dev": cons_m["max_dev"],
                "avg_dollar_vol_20d": dv20,
            })

        if (not brk_ok) and vol_rising_ok and inside_consolidation_range(df, consol_days, breakout_buffer_pct):
            watch_rows.append({
                "ticker": t,
                "asof": asof,
                "close": px,
                "vol_trend_ratio": vol_rising_ratio,
                "cons_std": cons_m["std"],
                "cons_slope": cons_m["slope"],
                "cons_max_dev": cons_m["max_dev"],
                "avg_dollar_vol_20d": dv20,
            })

    progress.progress(int(bi / total_batches * 100))

end_ts = datetime.now(timezone.utc)
status.write(f"Done. Runtime: {(end_ts - start_ts).total_seconds():.0f}s")
progress.progress(100)

breakouts_df = pd.DataFrame(breakouts_rows)
watch_df = pd.DataFrame(watch_rows)

# Sort outputs
if not breakouts_df.empty:
    breakouts_df["abs_roc"] = breakouts_df["roc_10d_pct"].abs()
    breakouts_df = breakouts_df.sort_values(["abs_roc", "volume_ratio"], ascending=[False, False]).drop(columns=["abs_roc"])
if not watch_df.empty:
    watch_df = watch_df.sort_values(["vol_trend_ratio", "avg_dollar_vol_20d"], ascending=[False, False])

# ----------------------------
# Outputs
# ----------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader(f"Breakouts ({len(breakouts_df)})")
    if breakouts_df.empty:
        st.write("None today under current thresholds.")
    else:
        st.dataframe(breakouts_df, use_container_width=True)
        st.download_button(
            "Download breakouts CSV",
            breakouts_df.to_csv(index=False).encode("utf-8"),
            file_name=f"sp500_breakouts_{end_ts.date().isoformat()}.csv",
            mime="text/csv",
        )

with c2:
    st.subheader(f"Watchlist ({len(watch_df)})")
    if watch_df.empty:
        st.write("None today under current thresholds.")
    else:
        st.dataframe(watch_df, use_container_width=True)
        st.download_button(
            "Download watchlist CSV",
            watch_df.to_csv(index=False).encode("utf-8"),
            file_name=f"sp500_watchlist_{end_ts.date().isoformat()}.csv",
            mime="text/csv",
        )

st.caption(
    "Logic: consolidation uses last N sessions excluding today (std, slope, max deviation thresholds). "
    "Breakout requires close beyond consolidation range by buffer, plus volume surge vs prior 10 sessions, plus 10-session ROC filter. "
    "Watchlist requires consolidation, rising 3-session volume vs prior 10 sessions, and still inside the consolidation range."
)
