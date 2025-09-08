import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
import io
import matplotlib.pyplot as plt

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("About This Tool")
st.sidebar.markdown("""
Visualize key technical indicators for any stock using Yahoo Finance data.

**Features:**
- Interactive OHLC candlesticks
- 8/20/50/100/200-day moving averages with full continuity
- Volume bars color-coded by up/down days
- RSI (14-day) with Wilder smoothing
- MACD (12,26,9) with histogram
- Optional Bollinger Bands (20, 2.0)
- Static image panel for easy copy or download
""")

ticker = st.sidebar.text_input("Ticker", "^GSPC").upper()
period = st.sidebar.selectbox(
    "Period", ["1mo","3mo","6mo","1y","2y","3y","5y","10y","max"], index=3
)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)

show_bbands = st.sidebar.checkbox("Show Bollinger Bands (20, 2.0)", value=True)
show_static = st.sidebar.checkbox("Show static image panel", value=True)

# ── Helpers ────────────────────────────────────────────────────────────────
def start_date_from_period(p: str) -> pd.Timestamp | None:
    today = pd.Timestamp.today().normalize()
    if p == "max":
        return None
    if p.endswith("mo"):
        months = int(p[:-2])
        return today - pd.DateOffset(months=months)
    if p.endswith("y"):
        years = int(p[:-1])
        return today - pd.DateOffset(years=years)
    return None

@st.cache_data(ttl=3600)
def get_full_history(tkr: str, intrvl: str) -> pd.DataFrame:
    try:
        return yf.Ticker(tkr).history(period="max", interval=intrvl, auto_adjust=False)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# ── Fetch full history for the chosen interval ─────────────────────────────
df_full = get_full_history(ticker, interval)
if df_full.empty:
    st.error("No data returned. Check symbol or internet.")
    st.stop()

# Clean index
if df_full.index.tz is not None:
    df_full.index = df_full.index.tz_localize(None)
if interval == "1d":
    df_full = df_full[df_full.index.weekday < 5]

# Optional hard cap to avoid huge payloads
CAP_MAX_ROWS = 200000
if len(df_full) > CAP_MAX_ROWS:
    df_full = df_full.tail(CAP_MAX_ROWS)

# ── Date-based windowing with a long warmup for indicators ─────────────────
start_dt = start_date_from_period(period)

# Warmup history length for indicators
BUFFER_YEARS = 50
if start_dt is not None:
    warmup_start = start_dt - pd.DateOffset(years=BUFFER_YEARS)
    df_ind = df_full[df_full.index >= warmup_start].copy()
else:
    df_ind = df_full.copy()

# ── Indicators computed on the warmup+window set ───────────────────────────
for w in (8, 20, 50, 100, 200):
    df_ind[f"MA{w}"] = df_ind["Close"].rolling(w).mean()

# RSI (14)
delta = df_ind["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
rs = avg_gain / avg_loss
df_ind["RSI14"] = 100 - (100 / (1 + rs))

# MACD (12, 26, 9)
df_ind["EMA12"]  = df_ind["Close"].ewm(span=12, adjust=False).mean()
df_ind["EMA26"]  = df_ind["Close"].ewm(span=26, adjust=False).mean()
df_ind["MACD"]   = df_ind["EMA12"] - df_ind["EMA26"]
df_ind["Signal"] = df_ind["MACD"].ewm(span=9, adjust=False).mean()
df_ind["Hist"]   = df_ind["MACD"] - df_ind["Signal"]

# Bollinger Bands (20, 2.0)
if show_bbands:
    bb_window = 20
    bb_mult = 2.0
    df_ind["BB_MA"] = df_ind["Close"].rolling(bb_window).mean()
    df_ind["BB_STD"] = df_ind["Close"].rolling(bb_window).std()
    df_ind["BB_UPPER"] = df_ind["BB_MA"] + bb_mult * df_ind["BB_STD"]
    df_ind["BB_LOWER"] = df_ind["BB_MA"] - bb_mult * df_ind["BB_STD"]

# Final display slice (now indicators retain full continuity)
if start_dt is not None:
    df_display = df_ind[df_ind.index >= start_dt].copy()
    if df_display.empty:
        df_display = df_ind.tail(300).copy()
else:
    df_display = df_ind.copy()

# ── Plotly Interactive Plot ────────────────────────────────────────────────
df_display["DateStr"] = df_display.index.strftime("%Y-%m-%d")
available_mas = [w for w in (8, 20, 50, 100, 200) if len(df_display) >= w]
missing_mas = [w for w in (8, 20, 50, 100, 200) if len(df_display) < w]
if missing_mas:
    st.warning("Some MAs may appear choppy due to limited visible data: " + ", ".join(f"MA{w}" for w in missing_mas))

fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    row_heights=[0.60, 0.14, 0.13, 0.13],
    vertical_spacing=0.04,
    specs=[[{"type": "candlestick"}], [{"type": "bar"}], [{"type": "scatter"}], [{"type": "scatter"}]]
)

# Candles
fig.add_trace(go.Candlestick(
    x=df_display["DateStr"], open=df_display["Open"], high=df_display["High"],
    low=df_display["Low"], close=df_display["Close"],
    increasing_line_color="#43A047", decreasing_line_color="#E53935", name="Price"
), row=1, col=1)

# Bollinger on price panel
if show_bbands and "BB_UPPER" in df_display and "BB_LOWER" in df_display:
    # Plot upper first, then lower with fill to create a band
    fig.add_trace(go.Scatter(
        x=df_display["DateStr"], y=df_display["BB_UPPER"],
        mode="lines", line=dict(width=1, color="#607D8B"),
        name="BB Upper"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_display["DateStr"], y=df_display["BB_LOWER"],
        mode="lines", line=dict(width=1, color="#607D8B"),
        fill="tonexty", fillcolor="rgba(96,125,139,0.15)",
        name="BB Lower"
    ), row=1, col=1)
    # Mid band
    fig.add_trace(go.Scatter(
        x=df_display["DateStr"], y=df_display.get("BB_MA", pd.Series(index=df_display.index)),
        mode="lines", line=dict(width=1, color="#455A64", dash="dot"),
        name="BB Mid"
    ), row=1, col=1)

# Moving averages
for w, color in zip(available_mas, ("green", "purple", "blue", "orange", "gray")):
    fig.add_trace(go.Scatter(
        x=df_display["DateStr"], y=df_display[f"MA{w}"],
        mode="lines", line=dict(color=color, width=1.3), name=f"MA{w}"
    ), row=1, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)

# Volume
vol_colors = [
    "#B0BEC5" if i == 0 else (
        "#43A047" if df_display["Close"].iat[i] > df_display["Close"].iat[i-1] else "#E53935"
    )
    for i in range(len(df_display))
]
fig.add_trace(go.Bar(
    x=df_display["DateStr"], y=df_display["Volume"], width=0.4,
    marker_color=vol_colors, opacity=0.7, name="Volume"
), row=2, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1)

# RSI
fig.add_trace(go.Scatter(
    x=df_display["DateStr"], y=df_display["RSI14"],
    mode="lines", line=dict(width=1.5, color="purple"), name="RSI (14)"
), row=3, col=1)
fig.update_yaxes(range=[0, 100], title_text="RSI", row=3, col=1, title_standoff=10)
fig.add_hline(y=80, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=20, line_dash="dash", line_color="gray", row=3, col=1)

# MACD
hist_colors = ["#43A047" if h > 0 else "#E53935" for h in df_display["Hist"]]
fig.add_trace(go.Bar(
    x=df_display["DateStr"], y=df_display["Hist"], marker_color=hist_colors, opacity=0.6, width=0.4, name="MACD Hist"
), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df_display["DateStr"], y=df_display["MACD"], mode="lines", line=dict(width=1.5, color="blue"), name="MACD"
), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df_display["DateStr"], y=df_display["Signal"], mode="lines", line=dict(width=1.2, color="orange"), name="Signal"
), row=4, col=1)
fig.update_yaxes(title_text="MACD", row=4, col=1)

# Month tick labels
month_starts = df_display.index.to_series().groupby(df_display.index.to_period("M")).first()
tickvals = month_starts.dt.strftime("%Y-%m-%d").tolist()
ticktext = month_starts.dt.strftime("%b-%y").tolist()
fig.update_xaxes(
    type="category", tickmode="array",
    tickvals=tickvals, ticktext=ticktext,
    tickangle=-45, showgrid=True, gridwidth=0.5, gridcolor="#e1e5ed"
)

title_suffix = " + Bollinger Bands" if show_bbands else ""
fig.update_layout(
    height=950,
    title=dict(text=f"{ticker} — OHLC + RSI & MACD{title_suffix}", x=0.5),
    plot_bgcolor="white", paper_bgcolor="white",
    hovermode="x unified",
    margin=dict(l=60, r=20, t=60, b=40),
    xaxis=dict(rangeslider_visible=False, showline=True, linewidth=1, linecolor='black'),
    legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center"),
    font=dict(family="Arial, sans-serif", size=12)
)

st.plotly_chart(fig, use_container_width=True)

# ── Static Image Panel (copy friendly) ─────────────────────────────────────
def render_static_panel(df: pd.DataFrame, show_bbands_flag: bool) -> bytes:
    """
    Returns PNG bytes for a clean static price panel.
    Plots Close, MA20, MA50, MA200, optional Bollinger Bands.
    """
    # Choose a compact subset of lines for readability
    cols_needed = ["Close", "MA20", "MA50", "MA200"]
    if show_bbands_flag and ("BB_UPPER" in df and "BB_LOWER" in df and "BB_MA" in df):
        cols_needed += ["BB_UPPER", "BB_LOWER", "BB_MA"]

    data = df[cols_needed].dropna(how="all").copy()
    if data.empty:
        # Fallback to Close only
        data = df[["Close"]].copy()

    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=160)
    ax.plot(data.index, data["Close"], linewidth=1.3, label="Close")

    if "MA20" in data:
        ax.plot(data.index, data["MA20"], linewidth=1.0, label="MA20")
    if "MA50" in data:
        ax.plot(data.index, data["MA50"], linewidth=1.0, label="MA50")
    if "MA200" in data:
        ax.plot(data.index, data["MA200"], linewidth=1.0, label="MA200")

    if show_bbands_flag and all(c in data for c in ["BB_UPPER", "BB_LOWER"]):
        ax.plot(data.index, data["BB_UPPER"], linewidth=0.9, label="BB Upper")
        ax.plot(data.index, data["BB_LOWER"], linewidth=0.9, label="BB Lower")
        # Fill band lightly
        ax.fill_between(data.index, data["BB_LOWER"], data["BB_UPPER"], alpha=0.12)

    ax.set_title(f"{ticker} Static Panel")
    ax.set_xlabel("")
    ax.set_ylabel("Price")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.legend(ncol=6, fontsize=8, frameon=False, loc="upper left")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

if show_static:
    st.subheader("Static image panel")
    png_bytes = render_static_panel(df_display, show_bbands)
    st.image(png_bytes, caption=f"{ticker} static price panel", use_column_width=True)
    st.download_button(
        label="Download static panel as PNG",
        data=png_bytes,
        file_name=f"{ticker}_static_panel.png",
        mime="image/png"
    )

st.caption("© 2025 AD Fund Management LP")
