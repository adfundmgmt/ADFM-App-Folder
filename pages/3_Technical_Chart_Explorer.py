import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# ── UI ─────────────────────────────────────────────────────────────────────
st.sidebar.header("About This Tool")
st.sidebar.markdown("""
Visualize key technical indicators using Yahoo Finance.

Features:
- Interactive OHLC candlesticks
- 8/20/50/100/200 moving averages (on the selected bar size)
- Volume color-coded by up/down days
- RSI(14) and MACD(12,26,9)
""")

ticker  = st.sidebar.text_input("Ticker", "^GSPC").upper()
period  = st.sidebar.selectbox("Period", ["1mo","3mo","6mo","1y","2y","3y","5y","10y","max"], index=3)
interval_mode = st.sidebar.selectbox("Interval", ["Auto","1d","1wk","1mo"], index=0)

# ── Helpers ────────────────────────────────────────────────────────────────
def period_offset(p: str) -> pd.DateOffset | None:
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
    return "1mo"

def xaxis_fmt(p: str):
    if p in ("1mo","3mo"): return dict(fmt="%b %d", angle=-45)
    if p in ("6mo","1y"):  return dict(fmt="%b %Y", angle=0)
    return dict(fmt="%Y", angle=0)

# strict Yahoo fetch at target bar size
@st.cache_data(ttl=3600)
def fetch_bars(tkr: str, start: pd.Timestamp, end: pd.Timestamp, intrvl: str) -> pd.DataFrame:
    df = yf.download(
        tkr,
        start=start,
        end=end + pd.Timedelta(days=1),
        interval=intrvl,
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        return df
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

# indicators computed on the same bar size
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for w in (8,20,50,100,200):
        out[f"MA{w}"] = out["Close"].rolling(w).mean()
    chg = out["Close"].diff()
    gain = chg.clip(lower=0)
    loss = -chg.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    out["RSI14"] = 100 - (100 / (1 + rs))
    out["EMA12"]  = out["Close"].ewm(span=12, adjust=False).mean()
    out["EMA26"]  = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"]   = out["EMA12"] - out["EMA26"]
    out["Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["Hist"]   = out["MACD"] - out["Signal"]
    return out

# ── Determine window and interval ──────────────────────────────────────────
# Step 1: get last available daily bar to anchor 'end'
daily_probe = fetch_bars(ticker, pd.Timestamp("1990-01-01"), pd.Timestamp.today().normalize(), "1d")
if daily_probe.empty:
    st.error("No data returned. Check symbol or internet.")
    st.stop()
end_bar = daily_probe.index.max()

off = period_offset(period)
if off is None:
    # max: span full history
    start_target = daily_probe.index.min()
else:
    start_target = end_bar - off

interval = auto_interval(period) if interval_mode == "Auto" else interval_mode

# Buffer for MA continuity at left edge
bars_needed = 220
if interval == "1d":   buffer_days = bars_needed
elif interval == "1wk": buffer_days = bars_needed * 7
else:                  buffer_days = bars_needed * 30

fetch_start = start_target - pd.DateOffset(days=buffer_days)

# ── Fetch only the needed bars at the chosen bar size ──────────────────────
raw = fetch_bars(ticker, fetch_start, end_bar, interval)
if interval == "1d":
    raw = raw[raw.index.weekday < 5]  # remove weekends

if raw.empty:
    st.error("No data returned for selected settings.")
    st.stop()

# ── Indicators and final trim ──────────────────────────────────────────────
bars = add_indicators(raw)
plot_df = bars[(bars.index >= start_target) & (bars.index <= end_bar)].copy()
if plot_df.empty:
    st.error("No data in selected window.")
    st.stop()

# ── Build figure ───────────────────────────────────────────────────────────
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
cl = plot_df["Close"].to_numpy()
vol_colors = ["#B0BEC5"] + ["#43A047" if cl[i] > cl[i-1] else "#E53935" for i in range(1, len(cl))]
fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["Volume"], marker_color=vol_colors, opacity=0.7, name="Volume"), row=2, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1, rangemode="tozero")

# RSI
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["RSI14"], mode="lines",
                         line=dict(width=1.5, color="purple"), name="RSI (14)"), row=3, col=1)
fig.update_yaxes(range=[0,100], title_text="RSI", row=3, col=1)
fig.add_hline(y=80, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=20, line_dash="dash", line_color="gray", row=3, col=1)

# MACD
hist_colors = ["#43A047" if h > 0 else "#E53935" for h in plot_df["Hist"].fillna(0)]
fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["Hist"], marker_color=hist_colors, opacity=0.6, name="MACD Hist"), row=4, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD"],   mode="lines", line=dict(width=1.5, color="blue"),   name="MACD"), row=4, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Signal"], mode="lines", line=dict(width=1.2, color="orange"), name="Signal"), row=4, col=1)
fig.update_yaxes(title_text="MACD", row=4, col=1)

# Clamp the x-range on ALL subplot axes
fmt = xaxis_fmt(period)
range_breaks = [dict(bounds=["sat","mon"])] if interval == "1d" else []
for ax in ["xaxis","xaxis2","xaxis3","xaxis4"]:
    fig["layout"][ax].update(
        type="date",
        tickformat=fmt["fmt"],
        tickangle=fmt["angle"],
        rangeslider_visible=False,
        rangebreaks=range_breaks,
        range=[start_target, end_bar],
        showgrid=True, gridwidth=0.5, gridcolor="#e1e5ed"
    )

fig.update_layout(
    height=950,
    title=dict(text=f"{ticker} | OHLC + RSI & MACD", x=0.5),
    plot_bgcolor="white", paper_bgcolor="white",
    hovermode="x unified",
    margin=dict(l=60, r=20, t=60, b=40),
    legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center"),
    font=dict(family="Arial, sans-serif", size=12),
    bargap=0.05
)

st.plotly_chart(fig, use_container_width=True)

# Debug footer: shows the actual plotted window and interval
st.caption(f"Window: {start_target.date()} → {end_bar.date()}   |   Interval: {interval}   |   Rows: {len(plot_df)}")
st.caption("© 2025 AD Fund Management LP")
