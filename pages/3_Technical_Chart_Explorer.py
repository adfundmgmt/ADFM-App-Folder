import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# ========= UI =========
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
interval_mode = st.sidebar.selectbox("Interval", ["Auto","1d","1wk","1mo"], index=0)

# ========= Helpers =========
def period_to_offset(p: str) -> pd.DateOffset | None:
    table = {
        "1mo": pd.DateOffset(months=1),
        "3mo": pd.DateOffset(months=3),
        "6mo": pd.DateOffset(months=6),
        "1y":  pd.DateOffset(years=1),
        "2y":  pd.DateOffset(years=2),
        "3y":  pd.DateOffset(years=3),
        "5y":  pd.DateOffset(years=5),
        "10y": pd.DateOffset(years=10),
        "max": None,
    }
    return table[p]

def auto_interval(p: str) -> str:
    if p in ("1mo","3mo","6mo","1y"):
        return "1d"
    if p in ("2y","3y","5y"):
        return "1wk"
    return "1mo"  # 10y or max

def xaxis_params(p: str):
    # Simple and readable; Plotly handles tick count
    if p in ("1mo","3mo"): return dict(fmt="%b %d", angle=-45)
    if p in ("6mo","1y"):  return dict(fmt="%b %Y", angle=0)
    return dict(fmt="%Y", angle=0)

# ========= Data fetch (daily only, bounded by dates) =========
@st.cache_data(ttl=3600)
def get_daily_range(tkr: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # fetch only what we need to avoid surprises
    df = yf.download(tkr, start=start, end=end + pd.Timedelta(days=1), interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        return df
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    # drop weekends
    df = df[df.index.weekday < 5]
    return df

# compute target window
today = pd.Timestamp.today().normalize()
off = period_to_offset(period)
if off is None:
    start = pd.Timestamp("1900-01-01")  # effectively max history
else:
    start = today - off

# decide interval
interval = auto_interval(period) if interval_mode == "Auto" else interval_mode

# buffer so long MAs have continuity BEFORE the visible window
bars_needed = 200  # for MA200 continuity on chosen bar size
if interval == "1d":
    buffer_days = bars_needed
elif interval == "1wk":
    buffer_days = bars_needed * 7
else:  # 1mo (approx)
    buffer_days = bars_needed * 30

dl_start = start - pd.DateOffset(days=buffer_days)
daily = get_daily_range(ticker, dl_start, today)

if daily.empty:
    st.error("No data returned. Check symbol or internet.")
    st.stop()

# ========= Resample strictly inside [dl_start, today] =========
def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    out = df.resample(rule).apply(agg).dropna()
    return out

if interval == "1d":
    bars = daily.copy()
elif interval == "1wk":
    bars = resample_ohlc(daily, "W-FRI")
else:
    bars = resample_ohlc(daily, "M")

# ========= Indicators on resampled bars (with buffer) =========
for w in (8,20,50,100,200):
    bars[f"MA{w}"] = bars["Close"].rolling(w).mean()

chg = bars["Close"].diff()
gain = chg.clip(lower=0)
loss = -chg.clip(upper=0)
avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
rs = avg_gain / avg_loss
bars["RSI14"] = 100 - (100 / (1 + rs))

bars["EMA12"] = bars["Close"].ewm(span=12, adjust=False).mean()
bars["EMA26"] = bars["Close"].ewm(span=26, adjust=False).mean()
bars["MACD"]   = bars["EMA12"] - bars["EMA26"]
bars["Signal"] = bars["MACD"].ewm(span=9, adjust=False).mean()
bars["Hist"]   = bars["MACD"] - bars["Signal"]

# ========= Trim to the exact visible window [start, today] =========
plot_df = bars[(bars.index >= start) & (bars.index <= today)].copy()
if plot_df.empty:
    st.error("No data in selected window.")
    st.stop()

# ========= Figure =========
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    row_heights=[0.60, 0.14, 0.13, 0.13], vertical_spacing=0.04,
    specs=[[{"type":"candlestick"}],[{"type":"bar"}],[{"type":"scatter"}],[{"type":"scatter"}]]
)

# Price
fig.add_trace(go.Candlestick(
    x=plot_df.index,
    open=plot_df["Open"], high=plot_df["High"],
    low=plot_df["Low"], close=plot_df["Close"],
    increasing_line_color="#43A047", decreasing_line_color="#E53935",
    name="Price"
), row=1, col=1)

for w, color in zip((8,20,50,100,200), ("green","purple","blue","orange","gray")):
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df[f"MA{w}"],
        mode="lines", line=dict(color=color, width=1.2), name=f"MA{w}"
    ), row=1, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)

# Volume
close_vals = plot_df["Close"].to_numpy()
vol_colors = ["#B0BEC5"] + ["#43A047" if close_vals[i] > close_vals[i-1] else "#E53935"
                            for i in range(1, len(close_vals))]
fig.add_trace(go.Bar(
    x=plot_df.index, y=plot_df["Volume"], marker_color=vol_colors, opacity=0.7, name="Volume"
), row=2, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1, rangemode="tozero")

# RSI
fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df["RSI14"], mode="lines",
    line=dict(width=1.5, color="purple"), name="RSI (14)"
), row=3, col=1)
fig.update_yaxes(range=[0,100], title_text="RSI", row=3, col=1)
fig.add_hline(y=80, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=20, line_dash="dash", line_color="gray", row=3, col=1)

# MACD
hist_colors = ["#43A047" if h > 0 else "#E53935" for h in plot_df["Hist"].fillna(0)]
fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["Hist"], marker_color=hist_colors, opacity=0.6, name="MACD Hist"), row=4, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD"],   mode="lines", line=dict(width=1.5, color="blue"),   name="MACD"), row=4, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Signal"], mode="lines", line=dict(width=1.2, color="orange"), name="Signal"), row=4, col=1)
fig.update_yaxes(title_text="MACD", row=4, col=1)

# X axis clamp and formatting
xa = xaxis_params(period)
range_breaks = [dict(bounds=["sat","mon"])] if interval == "1d" else []
fig.update_xaxes(
    type="date",
    tickformat=xa["fmt"],
    tickangle=xa["angle"],
    rangeslider_visible=False,
    rangebreaks=range_breaks,
    range=[start, today],  # hard clamp
    showgrid=True, gridwidth=0.5, gridcolor="#e1e5ed"
)

fig.update_layout(
    height=950,
    title=dict(text=f"{ticker} | OHLC + RSI & MACD", x=0.5),
    plot_bgcolor="white", paper_bgcolor="white",
    hovermode="x unified",
    margin=dict(l=60, r=20, t=60, b=40),
    xaxis=dict(showline=True, linewidth=1, linecolor="black"),
    legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center"),
    font=dict(family="Arial, sans-serif", size=12),
    bargap=0.05
)

st.plotly_chart(fig, use_container_width=True)
st.caption("© 2025 AD Fund Management LP")
