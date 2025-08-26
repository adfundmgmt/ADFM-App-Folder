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
- 8/20/50/100/200 moving averages (per selected bar frequency)
- Volume bars color-coded by up/down days
- RSI (14) with Wilder smoothing
- MACD (12,26,9) with histogram
""")

ticker = st.sidebar.text_input("Ticker", "^GSPC").upper()
period = st.sidebar.selectbox("Period", ["1mo","3mo","6mo","1y","2y","3y","5y","10y","max"], index=3)
interval_choice = st.sidebar.selectbox("Interval", ["Auto","1d","1wk","1mo"], index=0)

# ── Helpers ────────────────────────────────────────────────────────────────
def period_to_offset(p: str) -> pd.DateOffset | None:
    return {
        "1mo": pd.DateOffset(months=1),
        "3mo": pd.DateOffset(months=3),
        "6mo": pd.DateOffset(months=6),
        "1y":  pd.DateOffset(years=1),
        "2y":  pd.DateOffset(years=2),
        "3y":  pd.DateOffset(years=3),
        "5y":  pd.DateOffset(years=5),
        "10y": pd.DateOffset(years=10),
        "max": None,
    }[p]

def auto_interval(p: str) -> str:
    if p in ("1mo","3mo","6mo","1y"): return "1d"
    if p in ("2y","3y","5y"):         return "1wk"
    return "1mo"  # 10y or max

def xaxis_params(p: str):
    if p == "1mo": return dict(dtick="D2",  fmt="%b %d", angle=-45)
    if p == "3mo": return dict(dtick="W1",  fmt="%b %d", angle=-45)
    if p in ("6mo","1y"): return dict(dtick="M1",  fmt="%b %Y", angle=0)
    if p in ("2y","3y"):  return dict(dtick="M3",  fmt="%b %Y", angle=0)
    if p == "5y":         return dict(dtick="M6",  fmt="%b %Y", angle=0)
    if p == "10y":        return dict(dtick="M12", fmt="%Y",     angle=0)
    return dict(dtick="M24", fmt="%Y", angle=0)

# ── Data fetch (always daily max; we resample later) ───────────────────────
@st.cache_data(ttl=3600)
def get_daily_max(tkr: str) -> pd.DataFrame:
    df = yf.Ticker(tkr).history(period="max", interval="1d", auto_adjust=False)
    if df.empty: return df
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    df = df[df.index.weekday < 5]  # drop weekends
    return df

df_daily = get_daily_max(ticker)
if df_daily.empty:
    st.error("No data returned. Check symbol or internet.")
    st.stop()

# ── Window bounds from period (freeze the pan) ─────────────────────────────
end_ts = df_daily.index.max()
off = period_to_offset(period)
if off is None:
    start_ts = df_daily.index.min()
else:
    start_ts = end_ts - off
# Left padding to stabilize MAs near window edge
pad = pd.DateOffset(days=30)
win_start = max(df_daily.index.min(), start_ts - pad)
win_end = end_ts

# ── Choose interval (cascade or manual) ────────────────────────────────────
interval = auto_interval(period) if interval_choice == "Auto" else interval_choice

# ── Resample inside the fixed window ───────────────────────────────────────
df_win = df_daily.loc[(df_daily.index >= win_start) & (df_daily.index <= win_end)].copy()

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "Open": "first", "High": "max", "Low": "min", "Close": "last",
        "Volume": "sum"
    }
    return df.resample(rule).apply(agg).dropna()

if interval == "1d":
    df_plot = df_win.copy()
elif interval == "1wk":
    df_plot = resample_ohlc(df_win, "W-FRI")
else:  # "1mo"
    df_plot = resample_ohlc(df_win, "M")

# ── Indicators on resampled bars (bar-based windows: 8/20/50/100/200) ─────
for w in (8, 20, 50, 100, 200):
    df_plot[f"MA{w}"] = df_plot["Close"].rolling(w).mean()

# RSI (Wilder) on chosen bar frequency
chg = df_plot["Close"].diff()
gain = chg.clip(lower=0)
loss = -chg.clip(upper=0)
avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
rs = avg_gain / avg_loss
df_plot["RSI14"] = 100 - (100 / (1 + rs))

# MACD on chosen bar frequency
df_plot["EMA12"]  = df_plot["Close"].ewm(span=12, adjust=False).mean()
df_plot["EMA26"]  = df_plot["Close"].ewm(span=26, adjust=False).mean()
df_plot["MACD"]   = df_plot["EMA12"] - df_plot["EMA26"]
df_plot["Signal"] = df_plot["MACD"].ewm(span=9, adjust=False).mean()
df_plot["Hist"]   = df_plot["MACD"] - df_plot["Signal"]

# ── Build figure ───────────────────────────────────────────────────────────
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    row_heights=[0.60, 0.14, 0.13, 0.13], vertical_spacing=0.04,
    specs=[[{"type": "candlestick"}], [{"type": "bar"}],
           [{"type": "scatter"}], [{"type": "scatter"}]]
)

fig.add_trace(go.Candlestick(
    x=df_plot.index, open=df_plot["Open"], high=df_plot["High"],
    low=df_plot["Low"], close=df_plot["Close"],
    increasing_line_color="#43A047", decreasing_line_color="#E53935",
    name="Price",
    hovertemplate="Date=%{x|%Y-%m-%d}<br>"
                  "O=%{open:.2f} H=%{high:.2f} L=%{low:.2f} C=%{close:.2f}<extra></extra>"
), row=1, col=1)

for w, color in zip((8,20,50,100,200), ("green","purple","blue","orange","gray")):
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot[f"MA{w}"],
        mode="lines", line=dict(color=color, width=1.2), name=f"MA{w}"
    ), row=1, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)

# Volume
close_vals = df_plot["Close"].to_numpy()
vol_colors = ["#B0BEC5"] + ["#43A047" if close_vals[i] > close_vals[i-1] else "#E53935"
                            for i in range(1, len(close_vals))]
fig.add_trace(go.Bar(
    x=df_plot.index, y=df_plot["Volume"],
    marker_color=vol_colors, opacity=0.7, name="Volume"
), row=2, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1, rangemode="tozero")

# RSI
fig.add_trace(go.Scatter(
    x=df_plot.index, y=df_plot["RSI14"], mode="lines",
    line=dict(width=1.5, color="purple"), name="RSI (14)"
), row=3, col=1)
fig.update_yaxes(range=[0, 100], title_text="RSI", row=3, col=1, title_standoff=10)
fig.add_hline(y=80, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=20, line_dash="dash", line_color="gray", row=3, col=1)

# MACD
hist_colors = ["#43A047" if h > 0 else "#E53935" for h in df_plot["Hist"].fillna(0)]
fig.add_trace(go.Bar(
    x=df_plot.index, y=df_plot["Hist"], marker_color=hist_colors,
    opacity=0.6, name="MACD Hist"
), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df_plot.index, y=df_plot["MACD"], mode="lines",
    line=dict(width=1.5, color="blue"), name="MACD"
), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df_plot.index, y=df_plot["Signal"], mode="lines",
    line=dict(width=1.2, color="orange"), name="Signal"
), row=4, col=1)
fig.update_yaxes(title_text="MACD", row=4, col=1)

# X-axis: freeze to the requested period
xa = xaxis_params(period)
range_breaks = [dict(bounds=["sat", "mon"])] if interval == "1d" else []
fig.update_xaxes(
    type="date", tickmode="auto", dtick=xa["dtick"], tickformat=xa["fmt"],
    showgrid=True, gridwidth=0.5, gridcolor="#e1e5ed",
    tickangle=xa["angle"], rangeslider_visible=False, rangebreaks=range_breaks,
    range=[start_ts, end_ts]  # hard clamp to requested window
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
