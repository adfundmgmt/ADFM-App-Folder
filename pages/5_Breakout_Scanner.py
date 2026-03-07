import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.style.use("default")
pd.options.mode.chained_assignment = None

# ── Page Setup ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Breakout Scanner", layout="wide")
st.title("Breakout Scanner")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Screen a custom ticker list for fresh price breakouts, momentum context, and RSI regime.

        • Uses Yahoo Finance daily data over a fixed 2 year lookback  
        • Detects true breakouts versus prior 20 / 50 / 100 / 200 day highs using a 1 day shift  
        • Computes RSI(7 / 14 / 21) from Typical Price ((H+L+C)/3)  
        • Smooths each RSI line with a 3 period EMA for cleaner reads  
        • Adds breakout distance, 200DMA distance, and volume confirmation  
        • Includes a per-ticker chart for price, prior highs, moving averages, and RSI  

        Notes  
        • Breakout logic uses adjusted close when available  
        • RSI uses raw High / Low / Close typical price  
        • Volume confirmation = latest volume / 20 day average volume
        """,
        unsafe_allow_html=True,
    )

# ── Inputs ──────────────────────────────────────────────────────────────────
tickers_input = st.sidebar.text_input(
    "Tickers (comma separated):",
    "NVDA, MSFT, AAPL, AMZN, GOOGL, META, TSLA, AVGO, TSM"
).upper()

def normalize_tickers(raw: str) -> List[str]:
    seen = set()
    cleaned = []
    for x in raw.split(","):
        t = x.strip().upper()
        if t and t not in seen:
            seen.add(t)
            cleaned.append(t)
    return cleaned

tickers = normalize_tickers(tickers_input)
if not tickers:
    st.warning("Please enter at least one ticker.")
    st.stop()

# Fixed parameters
RSI_WINDOWS = [7, 14, 21]
RSI_SMOOTH_SPAN = 3
LOOKBACK_PERIOD = "2y"
INTERVAL = "1d"
MIN_BARS = 200
BREAKOUT_WINDOWS = (20, 50, 100, 200)
MA_WINDOWS = (20, 50, 200)

# ── Data Fetch ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ohlcv_batch(symbols: Tuple[str, ...], period: str, interval: str) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Returns:
        frames: {ticker: DataFrame[Open, High, Low, Close, Adj Close, Volume]}
        failed: [tickers with no usable data]
    """
    frames: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []

    try:
        raw = yf.download(
            list(symbols),
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        return {}, list(symbols)

    required_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    if raw.empty:
        return {}, list(symbols)

    # Multi-ticker path
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = list(raw.columns.get_level_values(0).unique())
        level1 = list(raw.columns.get_level_values(1).unique())

        # yfinance can return either [ticker, field] or [field, ticker]
        ticker_first = any(sym in level0 for sym in symbols)

        for sym in symbols:
            try:
                if ticker_first and sym in level0:
                    sub = raw[sym].copy()
                elif sym in level1:
                    sub = raw.xs(sym, axis=1, level=1).copy()
                else:
                    failed.append(sym)
                    continue

                keep = [c for c in required_cols if c in sub.columns]
                sub = sub[keep].copy()
                sub = sub.dropna(how="all")
                if sub.empty or "Close" not in sub.columns:
                    failed.append(sym)
                    continue

                for col in required_cols:
                    if col not in sub.columns:
                        sub[col] = np.nan

                sub = sub[required_cols].sort_index()
                frames[sym] = sub
            except Exception:
                failed.append(sym)

    # Single-ticker flat-column path
    else:
        sym = symbols[0]
        sub = raw.copy()
        keep = [c for c in required_cols if c in sub.columns]
        sub = sub[keep].copy().dropna(how="all")
        if sub.empty or "Close" not in sub.columns:
            failed.append(sym)
        else:
            for col in required_cols:
                if col not in sub.columns:
                    sub[col] = np.nan
            sub = sub[required_cols].sort_index()
            frames[sym] = sub

        for other in symbols[1:]:
            failed.append(other)

    # Final validation
    final_frames = {}
    for sym, df_ in frames.items():
        usable = df_.dropna(subset=["Close"], how="all")
        if usable.empty:
            failed.append(sym)
        else:
            final_frames[sym] = usable

    failed = sorted(list(set(failed) - set(final_frames.keys())))
    return final_frames, failed

data_map, failed_tickers = fetch_ohlcv_batch(tuple(sorted(tickers)), LOOKBACK_PERIOD, INTERVAL)

if not data_map:
    st.error("No valid price data returned. Check symbols or connectivity.")
    st.stop()

# Keep only names with enough history
valid_map = {}
insufficient = []
for sym, df_sym in data_map.items():
    px = df_sym["Adj Close"].copy()
    if px.isna().all():
        px = df_sym["Close"].copy()
    px = px.dropna()
    if len(px) >= MIN_BARS:
        valid_map[sym] = df_sym
    else:
        insufficient.append(sym)

data_map = valid_map

if not data_map:
    st.info("No tickers have at least 200 daily observations.")
    st.stop()

# ── Helpers ─────────────────────────────────────────────────────────────────
def get_price_series(df_sym: pd.DataFrame) -> pd.Series:
    s = df_sym["Adj Close"].copy()
    if s.isna().all():
        s = df_sym["Close"].copy()
    return s.dropna()

def typical_price(df_sym: pd.DataFrame) -> pd.Series:
    cols = ["High", "Low", "Close"]
    base = df_sym[cols].copy().dropna()
    if base.empty:
        return get_price_series(df_sym)
    return (base["High"] + base["Low"] + base["Close"]) / 3.0

def rsi_wilder(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    roll_down = down.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def pct_diff(current: float, reference: float) -> float:
    if pd.isna(current) or pd.isna(reference) or reference == 0:
        return np.nan
    return (current / reference - 1.0) * 100.0

def build_signal_table(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []

    for sym, df_sym in frames.items():
        price = get_price_series(df_sym)
        if len(price) < MIN_BARS:
            continue

        latest = float(price.iloc[-1])

        prior_highs = {}
        breakout_flags = {}
        breakout_distances = {}

        for w in BREAKOUT_WINDOWS:
            prior_high = price.rolling(w).max().shift(1).iloc[-1]
            prior_highs[w] = float(prior_high) if pd.notna(prior_high) else np.nan
            breakout_flags[w] = bool(pd.notna(prior_high) and latest >= prior_high)
            breakout_distances[w] = pct_diff(latest, prior_high)

        tp = typical_price(df_sym).dropna()
        rsi_vals = {}
        for w in RSI_WINDOWS:
            if len(tp) >= w:
                r = rsi_wilder(tp, w).ewm(span=RSI_SMOOTH_SPAN, adjust=False).mean()
                rsi_vals[w] = float(r.iloc[-1]) if pd.notna(r.iloc[-1]) else np.nan
            else:
                rsi_vals[w] = np.nan

        ma_20 = price.rolling(20).mean().iloc[-1]
        ma_50 = price.rolling(50).mean().iloc[-1]
        ma_200 = price.rolling(200).mean().iloc[-1]

        vol = df_sym["Volume"].dropna()
        latest_vol = float(vol.iloc[-1]) if not vol.empty else np.nan
        avg_vol_20 = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else np.nan
        vol_ratio = latest_vol / avg_vol_20 if pd.notna(latest_vol) and pd.notna(avg_vol_20) and avg_vol_20 != 0 else np.nan

        rec = {
            "Ticker": sym,
            "Price": latest,
            "20DMA": ma_20,
            "50DMA": ma_50,
            "200DMA": ma_200,
            "% vs 200DMA": pct_diff(latest, ma_200),
            "Volume": latest_vol,
            "20D Avg Vol": avg_vol_20,
            "Vol Ratio": vol_ratio,
        }

        for w in BREAKOUT_WINDOWS:
            rec[f"Prior {w}D High"] = prior_highs[w]
            rec[f"Breakout {w}D"] = breakout_flags[w]
            rec[f"% vs Prior {w}D High"] = breakout_distances[w]

        for w in RSI_WINDOWS:
            rec[f"RSI ({w})"] = rsi_vals[w]

        # Composite ranking
        breakout_score = (
            1.0 * int(rec["Breakout 20D"]) +
            1.5 * int(rec["Breakout 50D"]) +
            2.0 * int(rec["Breakout 100D"]) +
            3.0 * int(rec["Breakout 200D"])
        )

        momentum_score = 0.0
        for w in BREAKOUT_WINDOWS:
            v = rec.get(f"% vs Prior {w}D High", np.nan)
            if pd.notna(v):
                momentum_score += max(v, 0)

        rsi_score = 0.0
        r14 = rec.get("RSI (14)", np.nan)
        if pd.notna(r14):
            rsi_score = max(min(r14 - 50, 30), -30)

        volume_score = 0.0
        if pd.notna(vol_ratio):
            volume_score = min(vol_ratio, 3.0)

        rec["Signal Score"] = breakout_score * 10 + momentum_score + 0.4 * rsi_score + 2.0 * volume_score
        records.append(rec)

    out = pd.DataFrame(records)
    if out.empty:
        return out

    out = out.sort_values(
        by=["Signal Score", "Breakout 200D", "Breakout 100D", "Breakout 50D", "Breakout 20D", "Vol Ratio"],
        ascending=[False, False, False, False, False, False]
    ).reset_index(drop=True)

    return out

df = build_signal_table(data_map)

if df.empty:
    st.info("No valid signals found.")
    st.stop()

# ── Status Messages ─────────────────────────────────────────────────────────
if failed_tickers:
    st.caption(f"Failed to retrieve usable data for: {', '.join(sorted(failed_tickers))}")
if insufficient:
    st.caption(f"Insufficient history (< {MIN_BARS} bars): {', '.join(sorted(insufficient))}")

# ── Display Table ───────────────────────────────────────────────────────────
st.markdown("### Breakout & RSI Signals")

break_cols = [c for c in df.columns if c.startswith("Breakout ")]
pct_cols = [c for c in df.columns if c.startswith("% vs")]
rsi_cols = [c for c in df.columns if c.startswith("RSI")]
price_cols = ["Price", "20DMA", "50DMA", "200DMA"] + [c for c in df.columns if c.startswith("Prior ")]
vol_cols = ["Volume", "20D Avg Vol", "Vol Ratio", "Signal Score"]

display_df = df.copy()

for col in break_cols:
    display_df[col] = display_df[col].map(lambda x: "✅" if bool(x) else "")

for col in price_cols:
    if col in display_df.columns:
        display_df[col] = display_df[col].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")

for col in pct_cols:
    if col in display_df.columns:
        display_df[col] = display_df[col].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "")

for col in rsi_cols:
    if col in display_df.columns:
        display_df[col] = display_df[col].map(lambda x: f"{x:.1f}" if pd.notna(x) else "")

for col in ["Volume", "20D Avg Vol"]:
    if col in display_df.columns:
        display_df[col] = display_df[col].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")

for col in ["Vol Ratio", "Signal Score"]:
    if col in display_df.columns:
        display_df[col] = display_df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.download_button(
    "Download as CSV",
    df.to_csv(index=False),
    file_name="breakout_signals.csv",
    mime="text/csv"
)

# ── Per-Ticker Chart ────────────────────────────────────────────────────────
st.markdown("### Per-Ticker Inspection")

selected = st.selectbox("Select ticker to chart:", df["Ticker"].tolist(), index=0)
df_sel = data_map[selected].copy()
price_sel = get_price_series(df_sel)
tp_sel = typical_price(df_sel).dropna()

if len(price_sel) < MIN_BARS:
    st.info(f"{selected} does not have enough data to draw rolling highs.")
else:
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1]},
        constrained_layout=True
    )

    # Price panel
    ax1.plot(price_sel.index, price_sel, label="Adj Close / Close", color="black", linewidth=2.0)

    for w, color in zip(MA_WINDOWS, ["#1f77b4", "#9467bd", "#2ca02c"]):
        ma = price_sel.rolling(w).mean()
        ax1.plot(price_sel.index, ma, label=f"{w}DMA", color=color, linewidth=1.2)

    for w, color in zip(BREAKOUT_WINDOWS, ["#9ecae1", "#fdae6b", "#a1d99b", "#fcae91"]):
        prior_high = price_sel.rolling(w).max().shift(1)
        ax1.plot(
            price_sel.index,
            prior_high,
            linestyle="--",
            linewidth=1.0,
            color=color,
            alpha=0.95,
            label=f"Prior {w}D High"
        )

    latest_px = float(price_sel.iloc[-1])
    ax1.axhline(latest_px, color="#444444", linestyle=":", linewidth=1.0, alpha=0.8)

    ax1.set_title(f"{selected} Price, Prior Highs, and Moving Averages", fontweight="bold")
    ax1.set_ylabel("Price")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8, ncol=3, frameon=False)
    ax1.margins(x=0.01)

    # RSI panel
    for w, color in zip(RSI_WINDOWS, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        r = rsi_wilder(tp_sel, w).ewm(span=RSI_SMOOTH_SPAN, adjust=False).mean()
        ax2.plot(r.index, r, label=f"RSI({w})", linewidth=1.5, color=color)

    ax2.axhline(80, linestyle="--", color="gray", linewidth=0.9)
    ax2.axhline(50, linestyle=":", color="gray", linewidth=0.9)
    ax2.axhline(20, linestyle="--", color="gray", linewidth=0.9)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    ax2.set_xlabel("Date")
    ax2.set_title(f"{selected} RSI on Typical Price", fontweight="bold")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8, frameon=False)
    ax2.margins(x=0.01)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Quick diagnostic strip
    latest_row = df[df["Ticker"] == selected].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Signal Score", f"{latest_row['Signal Score']:.2f}")
    c2.metric("% vs Prior 50D High", f"{latest_row['% vs Prior 50D High']:.2f}%" if pd.notna(latest_row["% vs Prior 50D High"]) else "N/A")
    c3.metric("% vs 200DMA", f"{latest_row['% vs 200DMA']:.2f}%" if pd.notna(latest_row["% vs 200DMA"]) else "N/A")
    c4.metric("Vol Ratio", f"{latest_row['Vol Ratio']:.2f}" if pd.notna(latest_row["Vol Ratio"]) else "N/A")

# ── Footer ──────────────────────────────────────────────────────────────────
latest_dates = []
for _, df_sym in data_map.items():
    px = get_price_series(df_sym)
    if not px.empty:
        latest_dates.append(px.index.max())

if latest_dates:
    data_through = max(latest_dates).date()
    st.caption(f"Data through: {data_through}")

st.caption("© 2026 AD Fund Management LP")
