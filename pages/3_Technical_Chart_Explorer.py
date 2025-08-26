import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Optional

# ===================== UI =====================
st.set_page_config(page_title="ADFM Chart", layout="wide")

st.sidebar.header("About This Tool")
st.sidebar.markdown(
    "Visualize OHLC candlesticks with MAs, RSI, and MACD using Yahoo Finance daily data."
)

ticker = st.sidebar.text_input("Ticker", "^GSPC").upper()
period = st.sidebar.selectbox(
    "Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "max"],
    index=3,
)

# ===================== Helpers =====================
def period_offset(p: str) -> Optional[pd.DateOffset]:
    return {
        "1mo": pd.DateOffset(months=1),
        "3mo": pd.DateOffset(months=3),
        "6mo": pd.DateOffset(months=6),
        "1y": pd.DateOffset(years=1),
        "2y": pd.DateOffset(years=2),
        "3y": pd.DateOffset(years=3),
        "5y": pd.DateOffset(years=5),
        "10y": pd.DateOffset(years=10),
        "max": None,
    }[p]

def xaxis_fmt(p: str):
    if p in ("1mo", "3mo"):
        return {"fmt": "%b %d", "angle": -45}
    if p in ("6mo", "1y"):
        return {"fmt": "%b %Y", "angle": 0}
    return {"fmt": "%Y", "angle": 0}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_daily(tkr: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    df = yf.download(
        tkr,
        start=start_dt,
        end=end_dt + pd.Timedelta(days=1),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        return df
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    # weekdays only
    df = df[df.index.weekday < 5]
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for w in (8, 20, 50, 100, 200):
        out[f"MA{w}"] = out["Close"].rolling(w).mean()

    # RSI(14)
    chg = out["Close"].diff()
    gain = chg.clip(lower=0)
    loss = -chg.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    out["RSI14"] = 100 - (100 / (1 + rs))

    # MACD(12,26,9)
    out["EMA12"] = out["Close"].ewm(span=12, adjust=False).mean()
    out["EMA26"] = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = out["EMA12"] - out["EMA26"]
    out["Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["Hist"] = out["MACD"] - out["Signal"]
    return out

# ===================== Data window =====================
probe = fetch_daily(ticker, pd.Timestamp("1990-01-01"), pd.Timestamp.today().normalize())
if probe.empty:
    st.error("No data returned. Check symbol or internet.")
    st.stop()

end_bar = probe.index.max()
off = period_offset(period)
if off is None:
    start_target = probe.index.min()
else:
    start_target = end_bar - off

# indicator buffer so edges are smooth
buffer_days = 260
fetch_start = start_target - pd.DateOffset(days=buffer_days)

raw = fetch_daily(ticker, fetch_start, end_bar)
if raw.empty:
    st.error("No data returned for selected settings.")
    st.stop()

bars = add_indicators(raw)
plot_df = bars[(bars.index >= start_target) & (bars.index <= end_bar)].copy()
if plot_df.empty:
    st.error("No data in selected window.")
    st.stop()

# ===================== Figure =====================
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    row_heights=[0.60, 0.14, 0.13, 0.13],
    vertical_spacing=0.04,
    specs=[[{"type": "candlestick"}],
           [{"type": "bar"}],
           [{"type": "scatter"}],
           [{"type": "scatter"}]]
)

# Price
fig.add_trace(
    go.Candlestick(
        x=plot_df.index,
        open=plot_df["Open"],
        high=plot_df["High"],
        low=plot_df["Low"],
        close=plot_df["Close"],
        increasing_line_color="#43A047",
        decreasing_line_color="#E53935",
        name="Price",
    ),
    row=1, col=1,
)

for w, color in zip((8, 20, 50, 100, 200), ("green", "purple", "blue", "orange", "gray")):
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df[f"MA{w}"], mode="lines",
                   line=dict(color=color, width=1.2), name=f"MA{w}"),
        row=1, col=1,
    )
fig.update_yaxes(title_text="Price", row=1, col=1)

# Volume
cl = plot_df["Close"].to_numpy()
vol_colors = ["#B0BEC5"] + [
    "#43A047" if cl[i] > cl[i - 1] else "#E53935" for i in range(1, len(cl))
]
fig.add_trace(
    go.Bar(x=plot_df.index, y=plot_df["Volume"],
           marker_color=vol_colors, opacity=0.7, name="Volume"),
    row=2, col=1,
)
fig.update_yaxes(title_text="Volume", row=2, col=1, rangemode="tozero")

# RSI
fig.add_trace(
    go.Scatter(x=plot_df.index, y=plot_df["RSI14"], mode="lines",
               line=dict(width=1.5, color="purple"), name="RSI (14)"),
    row=3, col=1,
)
fig.update_yaxes(range=[0, 100], title_text="RSI", row=3, col=1)
# horizontal guides via shapes so it works reliably with subplots
for y in (80, 20):
    fig.add_shape(type="line",
                  xref="x3", yref="y3",
                  x0=plot_df.index.min(), x1=plot_df.index.max(),
                  y0=y, y1=y,
                  line=dict(width=1, dash="dash", color="gray"))

# MACD
hist_colors = ["#43A047" if h > 0 else "#E53935" for h in plot_df["Hist"].fillna(0)]
fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["Hist"],
                     marker_color=hist_colors, opacity=0.6, name="MACD Hist"),
              row=4, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD"], mode="lines",
                         line=dict(width=1.5), name="MACD"),
              row=4, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Signal"], mode="lines",
                         line=dict(width=1.2), name="Signal"),
              row=4, col=1)
fig.update_yaxes(title_text="MACD", row=4, col=1)

# X axis clamp, weekend breaks, period-specific tickformat
fmt = xaxis_fmt(period)
for i, ax in enumerate(["xaxis", "xaxis2", "xaxis3", "xaxis4"], start=1):
    fig["layout"][ax].update(
        type="date",
        tickformat=fmt["fmt"],
        tickangle=fmt["angle"],
        rangeslider_visible=False,
        rangebreaks=[dict(bounds=["sat", "mon"])],
        range=[start_target, end_bar],
        showgrid=True,
        gridwidth=0.5,
        gridcolor="#e1e5ed",
    )

fig.update_layout(
    height=950,
    title=dict(text=f"{ticker} | Daily OHLC + RSI & MACD", x=0.5),
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="x unified",
    margin=dict(l=60, r=20, t=60, b=40),
    legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center"),
    font=dict(family="Arial, sans-serif", size=12),
    bargap=0.05,
)

st.plotly_chart(fig, use_container_width=True)

st.caption(f"Window: {start_target.date()} → {end_bar.date()} | Interval: 1d | Rows: {len(plot_df)}")
st.caption("© 2025 AD Fund Management LP")
