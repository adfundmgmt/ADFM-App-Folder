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

        Tips to reduce choppiness:
        • Use longer RSI windows or smooth RSI lines
        • Switch to weekly RSI for higher signal to noise
        • Optionally compute RSI from Typical Price ((H+L+C)/3)
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

min_price = st.sidebar.number_input("Min Price filter", value=2.0, step=0.5)

# RSI options
rsi_wins = st.sidebar.multiselect(
    "RSI windows",
    options=[7, 14, 21, 28],
    default=[7, 14, 21]
)
rsi_smooth = st.sidebar.checkbox("Smooth RSI lines visually", value=True,
                                 help="Applies a short EMA to RSI outputs for cleaner visuals")
rsi_smooth_span = st.sidebar.slider("Smoothing span", 2, 10, 3)
rsi_weekly = st.sidebar.checkbox("Use weekly RSI", value=False,
                                 help="Computes RSI from weekly closes for less noise")
rsi_typical = st.sidebar.checkbox("Use Typical Price for RSI", value=False,
                                  help="RSI on (High+Low+Close)/3 instead of Close")

# ── Data Fetch (Batch + Cached) ─────────────────────────────────────────────
LOOKBACK_PERIOD = "2y"
INTERVAL = "1d"
MIN_BARS = 200

@st.cache_data(ttl=1800)
def fetch_batch(ticks, period=LOOKBACK_PERIOD, interval=INTERVAL):
    """Return dict of DataFrames per field to support Typical Price if needed."""
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

prices = data.get("Adj Close", pd.DataFrame())
if prices.empty:
    st.error("No valid Adjusted Close data. Check tickers or connectivity.")
    st.stop()

# Filter by last valid price
last_valid = prices.apply(lambda s: s.dropna().iloc[-1] if s.dropna().shape[0] else np.nan)
valid_by_price = last_valid[last_valid >= min_price].index.tolist()
prices = prices[valid_by_price]

if prices.empty:
    st.error(f"No tickers above min price ${min_price:.2f}.")
    st.stop()

# Ensure enough bars
sufficient_len = [c for c in prices.columns if prices[c].dropna().shape[0] >= MIN_BARS]
prices = prices[sufficient_len]
if prices.empty:
    st.info("No tickers have at least 200 daily observations.")
    st.stop()

prices = prices.dropna(how="all").copy()

# ── RSI (Wilder with proper initialization) ─────────────────────────────────
def rsi_wilder(series: pd.Series, window: int) -> pd.Series:
    """
    Wilder RSI using RMA with min_periods=window for stable initialization.
    Optionally resampled to weekly by the caller before passing in.
    """
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Proper min_periods avoids early noise
    roll_up = up.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def maybe_weekly(s: pd.Series) -> pd.Series:
    return s.resample("W-FRI").last().dropna() if rsi_weekly else s

def price_for_rsi(sym: str) -> pd.Series:
    if rsi_typical:
        # Typical Price = (H + L + C) / 3, use Close if Adj Close missing in that step
        H = data["High"].get(sym)
        L = data["Low"].get(sym)
        C = data["Close"].get(sym)
        if H is None or L is None or C is None:
            return prices[sym]  # fallback to Adj Close
        tp = (H.align(L, join="inner")[0]
              + L.align(C, join="inner")[0]
              + C.align(H, join="inner")[0]) / 3.0
        return tp.dropna()
    return prices[sym]

# ── Build Signal Table ──────────────────────────────────────────────────────
WINDOWS_HI = (20, 50, 100, 200)

records = []
for sym in prices.columns:
    s = prices[sym].dropna()
    if s.shape[0] < MIN_BARS:
        continue

    latest = float(s.iloc[-1])
    highs = {w: float(s.rolling(w).max().iloc[-1]) for w in WINDOWS_HI}

    base = maybe_weekly(price_for_rsi(sym)).dropna()
    rsi_vals = {w: float(rsi_wilder(base, w).iloc[-1]) for w in rsi_wins if base.shape[0] >= w}

    rec = {"Ticker": sym, "Price": round(latest, 2)}
    for w in WINDOWS_HI:
        rec[f"{w}D High"] = round(highs[w], 2)
        rec[f"Breakout {w}D"] = latest >= highs[w]
    for w in rsi_wins:
        rec[f"RSI ({w})"] = round(rsi_vals.get(w, np.nan), 1)
    records.append(rec)

df = pd.DataFrame(records)
if df.empty:
    st.info("No breakouts detected or insufficient data after filters.")
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

st.download_button(
    "Download as CSV",
    df.to_csv(index=False),
    file_name="breakout_signals.csv"
)

# ── Per-Ticker Charts ───────────────────────────────────────────────────────
default_choice = st.session_state.get("selected_ticker", df["Ticker"].iloc[0])
if default_choice not in df["Ticker"].values:
    default_choice = df["Ticker"].iloc[0]

sel = st.selectbox("Select ticker to chart:", df["Ticker"], index=df["Ticker"].tolist().index(default_choice))
st.session_state["selected_ticker"] = sel

s_close = prices[sel].dropna()

if s_close.shape[0] < MIN_BARS:
    st.info(f"{sel} does not have enough data to draw rolling highs.")
else:
    # Choose base series for RSI
    base = maybe_weekly(price_for_rsi(sel)).dropna()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=False, constrained_layout=True)

    # Price and rolling highs
    ax1.plot(s_close.index, s_close, label="Adj Close", color="black", linewidth=2)
    for w, col in zip((20, 50, 100, 200), ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")):
        ax1.plot(s_close.index, s_close.rolling(w).max(), lw=1.1, color=col, label=f"{w}D High")
    ax1.set_title(f"{sel} Price & Rolling Highs", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_ylabel("Price")
    ax1.margins(x=0)
    ax1.set_xlim(s_close.index.min(), s_close.index.max())

    # RSI panel, possibly weekly index
    for w in rsi_wins:
        r = rsi_wilder(base, w)
        if rsi_smooth:
            r = r.ewm(span=rsi_smooth_span, adjust=False).mean()
        ax2.plot(r.index, r, label=f"RSI({w})" + (" W" if rsi_weekly else ""), linewidth=1.4,
                 color=("black" if w == 7 else None))
    ax2.axhline(80, ls="--", color="gray", lw=0.9, label="RSI 80")
    ax2.axhline(20, ls="--", color="gray", lw=0.9, label="RSI 20")
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
