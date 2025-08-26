import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# ===================== UI =====================
st.sidebar.header("About This Tool")
st.sidebar.markdown("""
Visualize key technical indicators using Yahoo Finance data.

Features:
- OHLC candlesticks (daily only)
- 8/20/50/100/200 MAs
- RSI(14) and MACD(12,26,9)
""")

ticker = st.sidebar.text_input("Ticker", "^GSPC").upper()
period = st.sidebar.selectbox("Period", ["1mo","3mo","6mo","1y","2y","3y","5y","10y","max"], index=3)

# ===================== Helpers =====================
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

def xaxis_style(p: str):
    # Tick density and label format tuned by horizon
    if p in ("1mo","3mo"):
        return dict(tickformat="%b %d", ticklabelmode="instant", tickangle=-45, nticks=30)
    if p in ("6mo","1y"):
        return dict(tickformat="%b %Y", ticklabelmode="instant", tickangle=0, nticks=24)
    # 2y+
    return dict(tickformat="%Y", ticklabelmode="instant", tickangle=0, nticks=10)

# ===================== Data (daily only) =====================
@st.cache_data(ttl=3600, show_spinner=False)
def get_daily(tkr: str) -> pd.DataFrame:
    # Hard-locked to daily candles. No interval option anywhere.
    df = yf.download(tkr, period="max", interval="1d", auto_adjust=False, progress=False)
    if df.empty:
     
rs = avg_gain / avg_loss
df["RSI14"] = 100 - (100 / (1 + rs))

# MACD
df["EMA12"]  = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA26"]  = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"]   = df["EMA12"] - df["EMA26"]
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["Hist"]   = df["MACD"] - df["Signal"]

plot_df = df.loc[(df.index >= start) & (df.index <= end)].copy()
if plot_df.empty:
    st.error("No data in selected window.")
    st.stop()

# ===================== Figure =====================
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    row_heights=[0.60, 0.14, 0.13, 0.13], vertical_spacing=0.04,
    specs=[[{"type":"candlestick"}],[{"type":"bar"}],[{"type":"scatter"}],[{"type":"scatter"}]]
)

# Price + MAs
fig.add_trace(go.Candlestick(
    x=plot_df.index,
    open=plot_df["Open"], high=plot_df["High"], low=plot_df["Low"], close=plot_df["Close"],
    increasing_line_color="#43A047", decreasing_line_color="#E53935",
    name="Price"
), row=1, col=1)

for w, color in zip((8,20,50,100,200), ("green","purple","blue","orange","gray")):
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df[f"MA{w}"], mode="lines",
        line=dict(color=color, width=1.2), name=f"MA{w}"
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

# X-axis style and clamp
style = xaxis_style(period)
for ax in ["xaxis","xaxis2","xaxis3","xaxis4"]:
    fig["layout"][ax].update(
        type="date",
        tickformat=style["tickformat"],
        ticklabelmode=style["ticklabelmode"],
        tickangle=style["tickangle"],
        nticks=style["nticks"],
        rangeslider_visible=False,
        range=[start, end],
        showgrid=True, gridwidth=0.5, gridcolor="#e1e5ed",
        # Remove weekend gaps to keep spacing clean even if a few slipped through
        rangebreaks=[dict(bounds=["sat","mon"])]
    )

fig.update_layout(
    height=950,
    title=dict(text=f"{ticker} | OHLC + RSI & MACD (Daily)", x=0.5),
    plot_bgcolor="white", paper_bgcolor="white",
    hovermode="x unified",
    margin=dict(l=60, r=20, t=60, b=40),
    legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center"),
    font=dict(family="Arial, sans-serif", size=12),
    bargap=0.05
)

st.plotly_chart(fig, use_container_width=True)
st.caption(f"Window: {start.date()} → {end.date()}   |   Interval: Daily   |   Bars: {len(plot_df)}")
