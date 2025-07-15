import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

CYCLICALS  = ["XLK", "XLI", "XLF", "XLC", "XLY"]
DEFENSIVES = ["XLP", "XLE", "XLV", "XLRE", "XLB", "XLU"]

st.set_page_config(layout="wide", page_title="S&P Cyclicals vs Defensives Dashboard")
st.title("S&P Cyclicals Relative to Defensives — Equal‑Weight")

# Sidebar: About section and lookback selector
with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
    This dashboard tracks the **relative performance of S&P cyclical and defensive sector ETFs** (equal-weighted) to visualize risk-on/risk-off regime shifts in US equities.

    - **Cyclical basket:** XLK, XLI, XLF, XLC, XLY
    - **Defensive basket:** XLP, XLE, XLV, XLRE, XLB, XLU
    - Shows cumulative return ratio, 50/200-day moving averages, and RSI (14).

    There are also other miscellaneous ratio charts below (Semis/Software, QQQ/IWM) for context and macro leadership.
    """)
    st.header("Look‑back")
    spans = {"3 M":90,"6 M":180,"9 M":270,"YTD":None,"1 Y":365,
             "3 Y":365*3,"5 Y":365*5,"10 Y":365*10}
    span_key = st.selectbox("", list(spans.keys()), index=list(spans).index("5 Y"))

today = datetime.today()
hist_start = today - timedelta(days=365*10+220)
disp_start = datetime(today.year,1,1) if span_key=="YTD" else today - timedelta(days=spans[span_key])

@st.cache_data(ttl=3600, show_spinner="Fetching ETF prices…")
def fetch_etfs(tickers, start, end):
    df = yf.download(tickers, start, end, group_by="ticker", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        closes = df.xs('Close', level=1, axis=1)
    else:
        closes = df[["Close"]]
    closes = closes.fillna(method='ffill').dropna()
    return closes

def calc_ratio(etfs1, etfs2):
    b1 = (1 + fetch_etfs(etfs1, hist_start, today).pct_change()).cumprod().mean(axis=1)
    b2 = (1 + fetch_etfs(etfs2, hist_start, today).pct_change()).cumprod().mean(axis=1)
    df = pd.DataFrame({'b1': b1, 'b2': b2}).dropna()
    ratio = (df['b1'] / df['b2']) * 100
    return ratio

def calc_ratio_simple(ticker1, ticker2, mult=1.0):
    closes = fetch_etfs([ticker1, ticker2], hist_start, today)
    df = closes[[ticker1, ticker2]].dropna()
    ratio = (df[ticker1] / df[ticker2]) * mult
    return ratio

def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    ma_up = up.rolling(n).mean()
    ma_dn = dn.rolling(n).mean()
    rs = ma_up / ma_dn
    return 100 - 100 / (1 + rs)

def plot_ratio_panel(ratio, disp_start, title, ylab="Ratio"):
    mask = ratio.index >= disp_start
    ratio_disp = ratio[mask]
    ma50 = ratio.rolling(50).mean()[mask]
    ma200 = ratio.rolling(200).mean()[mask]
    rsi_panel = rsi(ratio)[mask]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=ratio_disp.index, y=ratio_disp,
                             line=dict(color="black", width=2), name=title), row=1, col=1)
    fig.add_trace(go.Scatter(x=ma50.index, y=ma50,
                             line=dict(color="blue", width=2), name="50‑DMA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ma200.index, y=ma200,
                             line=dict(color="red", width=2), name="200‑DMA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=rsi_panel.index, y=rsi_panel,
                             line=dict(color="black", width=2), name="RSI‑14", showlegend=False), row=2, col=1)
    for y, clr, txt in [(70, "red", "Overbought"), (30, "green", "Oversold")]:
        fig.add_hline(y=y, row=2, col=1, line_dash="dot", line_color=clr)
        fig.add_annotation(x=rsi_panel.index[0], y=y, text=txt, yshift=4 if y == 70 else -8,
                           showarrow=False, font=dict(color=clr), row=2, col=1)
    fig.update_layout(height=600, margin=dict(l=20, r=20, t=30, b=20),
                      plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", font=dict(size=13)),
                      title=title)
    fig.update_yaxes(title_text=ylab, row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

# --- Panel 1: Cyclicals vs Defensives
cyc_def_ratio = calc_ratio(CYCLICALS, DEFENSIVES)
plot_ratio_panel(cyc_def_ratio, disp_start, "Cyclicals / Defensives (Equal-Weight)", ylab="Relative Ratio")

# --- Panel 2: SMH / IGV
smh_igv_ratio = calc_ratio_simple("SMH", "IGV")
plot_ratio_panel(smh_igv_ratio, disp_start, "SMH / IGV Relative Strength & RSI", ylab="SMH / IGV")

# --- Panel 3: QQQ / IWM
qqq_iwm_ratio = calc_ratio_simple("QQQ", "IWM")
plot_ratio_panel(qqq_iwm_ratio, disp_start, "QQQ / IWM Relative Strength & RSI", ylab="QQQ / IWM")
