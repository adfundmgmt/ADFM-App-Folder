import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
pd.options.mode.chained_assignment = None

import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("default")

# ── Page Setup ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Breakout Scanner", layout="wide")
st.title("Breakout Scanner")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Screen for stocks breaking out to 20D, 50D, 100D, or 200D highs and view multi-timeframe RSI.

        This version:
        • Always uses daily RSI
        • RSI from Typical Price ((H+L+C)/3)
        • RSI lines smoothed with EMA(3)
        • No price filter
        """
    )

# ── Inputs ─────────────────────────────────────────────────────────────────
tickers_input = st.sidebar.text_input(
    "Tickers (comma separated):",
    "NVDA, MSFT, AAPL, AMZN, GOOGL, META, TSLA, AVGO, TSM"
).upper()

tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.warning("Please enter at least one ticker.")
    st.stop()

# Fixed parameters
RSI_WINDOWS = [7, 14, 21]   # fixed RSI periods
RSI_SMOOTH_SPAN = 3         # smoothing span
MIN_BARS = 200
LOOKBACK_PERIOD = "2y"
INTERVAL = "1d"

# ── Data Fetch (Batch + Cached) ─────────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_batch(ticks, period=LOOKBACK_PERIOD, interval=INTERVAL):
    try:
        raw = yf.download(
            ticks, period=period, interval=interval,
            progress=False, group_by="ticker", auto_adjust=False
        )
    except Exception:
        return {}

    def extract(field):
        out = {}
        if isinstance(raw.columns, pd.MultiIndex):
            for t in ticks:
                try:
                    s = raw[(t, field)].dropna()
                    if not s.empty:
                        out[t] = s
                except Exception:
                    continue
        else:
            if field in raw.columns:
                t0 = ticks[0] if isinstance(ticks, list) and len(ticks) == 1 else "TICKER"
                out[t0] = raw[field].dropna()
        if not out:
            return pd.DataFrame()
        return pd.DataFrame(out).sort_index().dropna(how="all")

    adj_close = extract("Adj Close")
    close = extract("Close")
    high = extract("High")
    low = extract("Low")
    return {"Adj Close": adj_close, "Close": close, "High": high, "Low": low}

data = fetch_batch(tickers)

# Fallback to Close if Adj Close missing
prices = data.get("Adj Close", pd.DataFrame())
if prices.empty:
    prices = data.get("Close", pd.DataFrame())

if prices.empty:
    st.error("No valid price data. Check tickers or connectivity.")
    st.stop()

# Ensure enough bars
sufficient_len = [c for c in prices.columns if prices[c].dropna().shape[0] >= MIN_BARS]
prices = prices[sufficient_len]
if prices.empty:
    st.info("No tickers have at least 200 daily observations.")
    st.stop()

prices = prices.dropna(how="all").copy()

# ── RSI (Wilder) ────────────────────────────────────────────────────────────
def rsi_wilder(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# Always Typical Price for RSI
def typical_price(sym: str) -> pd.Series:
    H = data["High"].get(sym)
    L = data["Low"].get(sym)
    C = data["Close"].get(sym)
    if H is None or L is None or C is None:
        return prices[sym]
    tp = (H.align(L, join="inner")[0]
          + L.align(C, join="inner")[0]
          + C.align(H, join="inner")[0]) / 3.0
    return tp.dropna()

# ── Build Signal Table ──────────────────────────────────────────────────────
WINDOWS_HI = (20, 50, 100, 200)
records = []

for sym in prices.columns:
    s = prices[sym].dropna()
    if s.shape[0] < MIN_BARS:
        continue

    latest = float(s.iloc[-1])
    highs = {w: float(s.rolling(w).max().iloc[-1]) for w in WINDOWS_HI}

    base = typical_price(sym).dropna()
    rsi_vals = {w: float(rsi_wilder(base, w).ewm(span=RSI_SMOOTH_SPAN, adjust=False).mean().iloc[-1])
                for w in RSI_WINDOWS if base.shape[0] >= w}

    rec = {"Ticker": sym, "Price": round(latest, 2)}
    for w in WINDOWS_HI:
        rec[f"{w}D High"] = round(highs[w], 2)
        rec[f"Breakout {w}D"] = latest >= highs[w]
    for w in RSI_WINDOWS:
        rec[f"RSI ({w})"] = round(rsi_vals.get(w, np.nan), 1)
    records.append(rec)

df = pd.DataFrame(records)
if df.empty:
    st.info("No breakouts detected or insufficient data.")
    st.stop()

break_cols = [c for c in df.columns if c.startswith("Breakout ")]
df = df.sort_values(by=break_cols + ["Price"], ascending=False).reset_index(drop=True)

# ── Table Styling ───────────────────────────────────────────────────────────
def styled_table(dfin: pd.DataFrame):
    df_disp = dfin.copy()
    for col in break_cols:
        df_disp[col] = df_disp[col].map(lambda x: "✅" if bool(x) else "")
    float_cols = [c for c in df_disp.columns if "High" in c or c == "Price"]
    rsi_cols = [c for c in df_disp.columns if c.startswith("RSI")]
    for c in float_cols:
        df_disp[c] = df_disp[c].map(lambda x: f"{x:,.2f}")
    for c in rsi_cols:
        df_disp[c] = df_disp[c].map(lambda x: f"{x:.1f}" if pd.notna(x) else "")
    def color_rsi(val):
        try:
            v = float(val)
        except Exception:
            return ""
        if v >= 80:
            return "color: #d62728; font-weight: bold;"
        if v <= 20:
            return "color: #1f77b4; font-weight: bold;"
        return ""
    styled = (
        df_disp.style
        .set_table_styles([
            {"selector": "th", "props": [("font-size", "13px"), ("text-align", "center"), ("font-weight", "bold")]},
            {"selector": "td", "props": [("font-size", "13px"), ("text-align", "center")]},
        ])
        .applymap(color_rsi, subset=rsi_cols)
        .set_properties(**{"background-color": "#f8fafc"}, subset=pd.IndexSlice[:, ["Price"]])
        .set_properties(**{"border": "1.5px solid #e0e0e0"})
    )
    return styled

st.markdown("### Breakout & RSI Signals")
st.dataframe(styled_table(df), use_container_width=True)
st.download_button("Download as CSV", df.to_csv(index=False), file_name="breakout_signals.csv")

# ── Per-Ticker Charts ───────────────────────────────────────────────────────
sel = st.selectbox("Select ticker to chart:", df["Ticker"], index=0)
s_close = prices[sel].dropna()

if s_close.shape[0] < MIN_BARS:
    st.info(f"{sel} does not have enough data to draw rolling highs.")
else:
    base = typical_price(sel).dropna()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=False, constrained_layout=True)

    # Price and rolling highs
    ax1.plot(s_close.index, s_close, label="Close", color="black", linewidth=2)
    for w, col in zip(WINDOWS_HI, ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")):
        ax1.plot(s_close.index, s_close.rolling(w).max(), lw=1.1, color=col, label=f"{w}D High")
    ax1.set_title(f"{sel} Price & Rolling Highs", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_ylabel("Price")
    ax1.margins(x=0)
    ax1.set_xlim(s_close.index.min(), s_close.index.max())

    # RSI panel (daily)
    for w in RSI_WINDOWS:
        r = rsi_wilder(base, w).ewm(span=RSI_SMOOTH_SPAN, adjust=False).mean()
        ax2.plot(r.index, r, label=f"RSI({w})", linewidth=1.4)
    ax2.axhline(80, ls="--", color="gray", lw=0.9)
    ax2.axhline(20, ls="--", color="gray", lw=0.9)
    ax2.set_title(f"{sel} RSI Indicators", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_ylabel("RSI")
    ax2.set_xlabel("Date")
    ax2.margins(x=0)
    ax2.set_ylim(0, 100)

    st.pyplot(fig, use_container_width=True)

st.caption(f"Data through: {prices.index.max().date()}")
st.caption("© 2025 AD Fund Management LP")
