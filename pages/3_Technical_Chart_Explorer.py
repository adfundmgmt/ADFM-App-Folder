import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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
""")

ticker = st.sidebar.text_input("Ticker", "^GSPC").upper()
period = st.sidebar.selectbox(
    "Period", ["1mo","3mo","6mo","1y","2y","3y","5y","10y","max"], index=3
)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)

# ── Data fetch ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_full_history(tkr, intrvl):
    try:
        df = yf.Ticker(tkr).history(period="max", interval=intrvl, auto_adjust=False)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

df_full = get_full_history(ticker, interval)
if df_full.empty:
    st.error("No data returned. Check symbol or internet.")
    st.stop()

# optional cap to avoid pathological long histories
CAP_MAX_ROWS = 20000
if len(df_full) > CAP_MAX_ROWS:
    df_full = df_full.tail(CAP_MAX_ROWS)

# Remove weekends for daily bars to keep spacing clean
if interval == "1d":
    df_full = df_full[df_full.index.weekday < 5]

# ── Indicators on full history ─────────────────────────────────────────────
for w in (8, 20, 50, 100, 200):
    df_full[f"MA{w}"] = df_full["Close"].rolling(w).mean()

delta = df_full["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
rs = avg_gain / avg_loss
df_full["RSI14"] = 100 - (100 / (1 + rs))

df_full["EMA12"]  = df_full["Close"].ewm(span=12, adjust=False).mean()
df_full["EMA26"]  = df_full["Close"].ewm(span=26, adjust=False).mean()
df_full["MACD"]   = df_full["EMA12"] - df_full["EMA26"]
df_full["Signal"] = df_full["MACD"].ewm(span=9, adjust=False).mean()
df_full["Hist"]   = df_full["MACD"] - df_full["Signal"]

# ── Date-based windowing (fixes 1wk/1mo overshoot) ─────────────────────────
def period_to_offset(p: str) -> pd.DateOffset | None:
    if p == "1mo":  return pd.DateOffset(months=1)
    if p == "3mo":  return pd.DateOffset(months=3)
    if p == "6mo":  return pd.DateOffset(months=6)
    if p == "1y":   return pd.DateOffset(years=1)
    if p == "2y":   return pd.DateOffset(years=2)
    if p == "3y":   return pd.DateOffset(years=3)
    if p == "5y":   return pd.DateOffset(years=5)
    if p == "10y":  return pd.DateOffset(years=10)
    return None  # max

end_ts = df_full.index.max()
off = period_to_offset(period)
if off is None:
    df_display = df_full.copy()
else:
    start_ts = end_ts - off
    # pad a little for nicer MA continuity at left edge
    pad = pd.DateOffset(days=30) if interval == "1d" else pd.DateOffset(months=3)
    df_display = df_full[df_full.index >= (start_ts - pad)].copy()

available_mas = [w for w in (8, 20, 50, 100, 200) if len(df_display) >= w]
missing_mas = [w for w in (8, 20, 50, 100, 200) if len(df_display) < w]
if missing_mas:
    st.warning("Some MAs may look choppy due to limited visible data: " +
               ", ".join(f"MA{w}" for w in missing_mas))

# ── Axis helper ────────────────────────────────────────────────────────────
def xaxis_params(p: str):
    if p == "1mo":   return dict(dtick="D2",  tickformat="%b %d")
    if p == "3mo":   return dict(dtick="W1",  tickformat="%b %d")
    if p in ("6mo","1y"):  return dict(dtick="M1",  tickformat="%b %Y")
    if p in ("2y","3y"):   return dict(dtick="M3",  tickformat="%b %Y")
    if p == "5y":    return dict(dtick="M6",  tickformat="%b %Y")
    if p == "10y":   return dict(dtick="M12", tickformat="%Y")
    return dict(dtick="M24", tickformat="%Y")

xa = xaxis_params(period)

# ── Figure ─────────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    row_heights=[0.60, 0.14, 0.13, 0.13],
    vertical_spacing=0.04,
    specs=[[{"type": "candlestick"}],
           [{"type": "bar"}],
           [{"type": "scatter"}],
           [{"type": "scatter"}]]
)

fig.add_trace(go.Candlestick(
    x=df_display.index,
    open=df_display["Open"], high=df_display["High"],
    low=df_display["Low"], close=df_display["Close"],
    increasing_line_color="#43A047", decreasing_line_color="#E53935",
    name="Price",
    hovertemplate="Date=%{x|%Y-%m-%d}<br>O=%{open:.2f} H=%{high:.2f} L=%{low:.2f} C=%{close:.2f}<extra></extra>"
), row=1, col=1)

for w, color in zip(available_mas, ("green","purple","blue","orange","gray")):
    fig.add_trace(go.Scatter(
        x=df_display.index, y=df_display[f"MA{w}"],
        mode="lines", line=dict(color=color, width=1.3), name=f"MA{w}"
    ), row=1, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)

# Volume
cl = df_display["Close"].to_numpy()
vol_colors = ["#B0BEC5"] + [
    "#43A047" if cl[i] > cl[i-1] else "#E53935" for i in range(1, len(cl))
]
fig.add_trace(go.Bar(
    x=df_display.index, y=df_display["Volume"],
    marker_color=vol_colors, opacity=0.7, name="Volume"
), row=2, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1, rangemode="tozero")

# RSI
fig.add_trace(go.Scatter(
    x=df_display.index, y=df_display["RSI14"],
    mode="lines", line=dict(width=1.5, color="purple"), name="RSI (14)"
), row=3, col=1)
fig.update_yaxes(range=[0, 100], title_text="RSI", row=3, col=1, title_standoff=10)
fig.add_hline(y=80, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=20, line_dash="dash", line_color="gray", row=3, col=1)

# MACD
hist_colors = ["#43A047" if h > 0 else "#E53935" for h in df_display["Hist"]]
fig.add_trace(go.Bar(
    x=df_display.index, y=df_display["Hist"], marker_color=hist_colors,
    opacity=0.6, name="MACD Hist"
), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df_display.index, y=df_display["MACD"], mode="lines",
    line=dict(width=1.5, color="blue"), name="MACD"
), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df_display.index, y=df_display["Signal"], mode="lines",
    line=dict(width=1.2, color="orange"), name="Signal"
), row=4, col=1)
fig.update_yaxes(title_text="MACD", row=4, col=1)

# X axis
range_breaks = [dict(bounds=["sat", "mon"])] if interval == "1d" else []
fig.update_xaxes(
    type="date",
    tickmode="auto",
    dtick=xa["dtick"],
    tickformat=xa["tickformat"],
    showgrid=True, gridwidth=0.5, gridcolor="#e1e5ed",
    tickangle=-45 if period in ("1mo","3mo") else 0,
    rangeslider_visible=False,
    rangebreaks=range_breaks
)

fig.update_layout(
    height=950,
    title=dict(text=f"{ticker} | OHLC + RSI & MACD", x=0.5),
    plot_bgcolor="white", paper_bgcolor="white",
    hovermode="x unified",
    margin=dict(l=60, r=20, t=60, b=40),
    xaxis=dict(showline=True, linewidth=1, linecolor='black'),
    legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center"),
    font=dict(family="Arial, sans-serif", size=12),
    bargap=0.05
)

st.plotly_chart(fig, use_container_width=True)
st.caption("© 2025 AD Fund Management LP")
