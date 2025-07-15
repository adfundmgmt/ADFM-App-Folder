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

# Sidebar: lookback period
with st.sidebar:
    st.header("Look‑back")
    spans = {"3 M":90,"6 M":180,"9 M":270,"YTD":None,"1 Y":365,
             "3 Y":365*3,"5 Y":365*5,"10 Y":365*10}
    span_key = st.selectbox("", list(spans.keys()), index=list(spans).index("5 Y"))

today = datetime.today()
hist_start = today - timedelta(days=365*10+220)
disp_start = datetime(today.year,1,1) if span_key=="YTD" else today - timedelta(days=spans[span_key])

@st.cache_data(ttl=3600, show_spinner="Fetching ETF prices…")
def fetch_etfs(tickers,start,end):
    df = yf.download(tickers, start, end, group_by="ticker", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        closes = df.xs('Close',level=1,axis=1)
    else:
        closes = df[["Close"]]
    closes = closes.fillna(method='ffill').dropna()
    return closes

def calc_ratio(etfs1, etfs2):
    b1 = (1 + fetch_etfs(etfs1, hist_start, today).pct_change()).cumprod()
    b2 = (1 + fetch_etfs(etfs2, hist_start, today).pct_change()).cumprod()
    b1 = b1.mean(axis=1)
    b2 = b2.mean(axis=1)
    df = pd.DataFrame({'b1': b1, 'b2': b2}).dropna()
    ratio = (df['b1'] / df['b2']) * 100
    return ratio

def calc_ratio_simple(ticker1, ticker2):
    closes = fetch_etfs([ticker1, ticker2], hist_start, today)
    df = closes[[ticker1, ticker2]].dropna()
    ratio = df[ticker1] / df[ticker2]
    return ratio

def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    ma_up = up.rolling(n).mean()
    ma_dn = dn.rolling(n).mean()
    rs = ma_up / ma_dn
    return 100 - 100 / (1 + rs)

# --- Main Cyc/Def Ratio
cyc_def_ratio = calc_ratio(CYCLICALS, DEFENSIVES)
mask = cyc_def_ratio.index >= disp_start
cyc_def_ratio_disp = cyc_def_ratio[mask]
ma50 = cyc_def_ratio.rolling(50).mean()[mask]
ma200 = cyc_def_ratio.rolling(200).mean()[mask]
rsi_main = rsi(cyc_def_ratio)[mask]

fig = make_subplots(rows=2,cols=1,shared_xaxes=True,
                    row_heights=[0.75,0.25],vertical_spacing=0.03)
fig.add_trace(go.Scatter(x=cyc_def_ratio_disp.index, y=cyc_def_ratio_disp, line=dict(color="#355E3B", width=2), name="Cyc/Def"), row=1, col=1)
fig.add_trace(go.Scatter(x=ma50.index, y=ma50, line=dict(color="blue", width=2), name="50‑DMA"), row=1, col=1)
fig.add_trace(go.Scatter(x=ma200.index, y=ma200, line=dict(color="red", width=2), name="200‑DMA"), row=1, col=1)
fig.add_trace(go.Scatter(x=rsi_main.index, y=rsi_main, line=dict(color="black", width=2), name="RSI‑14", showlegend=False), row=2, col=1)
for y, clr, txt in [(70, "red", "Overbought"), (30, "green", "Oversold")]:
    fig.add_hline(y=y, row=2, col=1, line_dash="dot", line_color=clr)
    fig.add_annotation(x=rsi_main.index[0], y=y, text=txt, yshift=4 if y == 70 else -8,
                       showarrow=False, font=dict(color=clr), row=2, col=1)
fig.update_layout(height=700, margin=dict(l=20, r=20, t=30, b=20),
                  plot_bgcolor="white",
                  legend=dict(orientation="h", y=1.02, x=1, xanchor="right", font=dict(size=13)))
st.plotly_chart(fig, use_container_width=True)

# --- SMH / IGV Ratio
smh_igv_ratio = calc_ratio_simple("SMH", "IGV")
mask2 = smh_igv_ratio.index >= disp_start
smh_igv_ratio_disp = smh_igv_ratio[mask2]
ma50_2 = smh_igv_ratio.rolling(50).mean()[mask2]
ma200_2 = smh_igv_ratio.rolling(200).mean()[mask2]
rsi_smh_igv = rsi(smh_igv_ratio)[mask2]

fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     row_heights=[0.75, 0.25], vertical_spacing=0.03)
fig2.add_trace(go.Scatter(x=smh_igv_ratio_disp.index, y=smh_igv_ratio_disp, line=dict(color="#0a5ba1", width=2), name="SMH/IGV Ratio"), row=1, col=1)
fig2.add_trace(go.Scatter(x=ma50_2.index, y=ma50_2, line=dict(color="blue", width=2), name="50‑DMA"), row=1, col=1)
fig2.add_trace(go.Scatter(x=ma200_2.index, y=ma200_2, line=dict(color="red", width=2), name="200‑DMA"), row=1, col=1)
fig2.add_trace(go.Scatter(x=rsi_smh_igv.index, y=rsi_smh_igv, line=dict(color="black", width=2), name="RSI‑14", showlegend=False), row=2, col=1)
for y, clr, txt in [(70, "red", "Overbought"), (30, "green", "Oversold")]:
    fig2.add_hline(y=y, row=2, col=1, line_dash="dot", line_color=clr)
    fig2.add_annotation(x=rsi_smh_igv.index[0], y=y, text=txt, yshift=4 if y == 70 else -8,
                        showarrow=False, font=dict(color=clr), row=2, col=1)
fig2.update_layout(height=550, margin=dict(l=20, r=20, t=30, b=20),
                   plot_bgcolor="white",
                   legend=dict(orientation="h", y=1.02, x=1, xanchor="right", font=dict(size=13)),
                   title="SMH / IGV Relative Strength & RSI")
st.plotly_chart(fig2, use_container_width=True)
