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

# Trading day approximations for visible window sizing
base_trading_days = {
    "1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504, "3y": 756,
    "5y": 1260, "10y": 2520, "max": 252 * 25
}

# ── Cache with parameters ───────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_full_history(tkr, intrvl):
    try:
        df = yf.Ticker(tkr).history(period="max", interval=intrvl, auto_adjust=False)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

df_full = get_full_history(ticker, interval)
if df_full.empty:
    st.error("No data returned. Check symbol or internet.")
    st.stop()

# Cap at ~25 years
CAP_MAX_DAYS = 252 * 25
if len(df_full) > CAP_MAX_DAYS:
    df_full = df_full.tail(CAP_MAX_DAYS)

# Clean index to naive datetime
if df_full.index.tz is not None:
    df_full.index = df_full.index.tz_localize(None)

# Remove weekends for daily interval (helps spacing and grid)
if interval == "1d":
    df_full = df_full[df_full.index.weekday < 5]

# Indicators on full history (continuity kept even when slicing view)
for w in (8, 20, 50, 100, 200):
    df_full[f"MA{w}"] = df_full["Close"].rolling(w).mean()

# RSI (Wilder smoothing via EWM with alpha=1/14)
delta = df_full["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
rs = avg_gain / avg_loss
df_full["RSI14"] = 100 - (100 / (1 + rs))

# MACD
df_full["EMA12"]  = df_full["Close"].ewm(span=12, adjust=False).mean()
df_full["EMA26"]  = df_full["Close"].ewm(span=26, adjust=False).mean()
df_full["MACD"]   = df_full["EMA12"] - df_full["EMA26"]
df_full["Signal"] = df_full["MACD"].ewm(span=9, adjust=False).mean()
df_full["Hist"]   = df_full["MACD"] - df_full["Signal"]

# Slice for display while keeping MA continuity already computed above
display_days = base_trading_days[period]
df_display = df_full if len(df_full) <= display_days else df_full.tail(display_days).copy()

available_mas = [w for w in (8, 20, 50, 100, 200) if len(df_display) >= w]
missing_mas = [w for w in (8, 20, 50, 100, 200) if len(df_display) < w]
if missing_mas:
    st.warning(
        "Some MAs may look choppy due to limited visible data: "
        + ", ".join(f"MA{w}" for w in missing_mas)
    )

# ── Axis helpers: dynamic tick density and formatting ───────────────────────
def xaxis_params(selected_period: str):
    """
    Returns Plotly date-axis params tuned for readable labels.
    - dtick uses Plotly's calendar strings (e.g., 'M3' every 3 months).
    - tickformat controls label text.
    """
    if selected_period == "1mo":
        return dict(dtick="D2", tickformat="%b %d")
    if selected_period == "3mo":
        return dict(dtick="W1", tickformat="%b %d")
    if selected_period in ("6mo", "1y"):
        return dict(dtick="M1", tickformat="%b %Y")
    if selected_period in ("2y", "3y"):
        return dict(dtick="M3", tickformat="%b %Y")
    if selected_period == "5y":
        return dict(dtick="M6", tickformat="%b %Y")
    if selected_period == "10y":
        return dict(dtick="M12", tickformat="%Y")
    # max or anything larger
    return dict(dtick="M24", tickformat="%Y")

xa = xaxis_params(period)

# ── Build figure ────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    row_heights=[0.60, 0.14, 0.13, 0.13],
    vertical_spacing=0.04,
    specs=[[{"type": "candlestick"}], [{"type": "bar"}], [{"type": "scatter"}], [{"type": "scatter"}]]
)

# Price + MAs
fig.add_trace(go.Candlestick(
    x=df_display.index, open=df_display["Open"], high=df_display["High"],
    low=df_display["Low"], close=df_display["Close"],
    increasing_line_color="#43A047", decreasing_line_color="#E53935", name="Price",
    hovertemplate="Date=%{x|%Y-%m-%d}<br>O=%{open:.2f} H=%{high:.2f} L=%{low:.2f} C=%{close:.2f}<extra></extra>"
), row=1, col=1)

for w, color in zip(available_mas, ("green", "purple", "blue", "orange", "gray")):
    fig.add_trace(go.Scatter(
        x=df_display.index, y=df_display[f"MA{w}"],
        mode="lines", line=dict(color=color, width=1.3),
        name=f"MA{w}",
        hovertemplate="Date=%{x|%Y-%m-%d}<br>MA"+str(w)+"=%{y:.2f}<extra></extra>"
    ), row=1, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)

# Volume
vol_colors = []
cl = df_display["Close"].to_numpy()
for i in range(len(df_display)):
    if i == 0:
        vol_colors.append("#B0BEC5")
    else:
        vol_colors.append("#43A047" if cl[i] > cl[i-1] else "#E53935")

fig.add_trace(go.Bar(
    x=df_display.index, y=df_display["Volume"],
    marker_color=vol_colors, opacity=0.7, name="Volume"
), row=2, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1, rangemode="tozero")

# RSI
fig.add_trace(go.Scatter(
    x=df_display.index, y=df_display["RSI14"],
    mode="lines", line=dict(width=1.5, color="purple"), name="RSI (14)",
    hovertemplate="Date=%{x|%Y-%m-%d}<br>RSI14=%{y:.2f}<extra></extra>"
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

# X-axis: true date axis with dynamic density and readable labels
# Range breaks remove weekends for daily data to avoid awkward spacing
range_breaks = [dict(bounds=["sat", "mon"])] if interval == "1d" else []
fig.update_xaxes(
    type="date",
    tickmode="auto",
    dtick=xa["dtick"],
    tickformat=xa["tickformat"],
    showgrid=True, gridwidth=0.5, gridcolor="#e1e5ed",
    tickangle=0 if period not in ("1mo","3mo") else -45,
    rangeslider_visible=False,
    rangebreaks=range_breaks
)

# Layout
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
