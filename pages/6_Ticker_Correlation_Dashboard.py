import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

CYCLICALS  = ["XLK", "XLI", "XLF", "XLC", "XLY"]
DEFENSIVES = ["XLP", "XLE", "XLV", "XLRE", "XLB", "XLU"]

st.set_page_config(layout="wide", page_title="S&P Cyclicals vs Defensives Dashboard")
st.title("S&P Cyclicals Relative to Defensives – Equal-Weight")

start_date = "2022-07-01"
end_date = datetime.today().strftime('%Y-%m-%d')

def basket_price(etfs, start, end):
    data = yf.download(etfs, start=start, end=end)["Adj Close"]
    basket = data.pct_change().mean(axis=1)
    basket_cum = (1 + basket).cumprod()
    return basket_cum

cyc = basket_price(CYCLICALS, start_date, end_date)
defn = basket_price(DEFENSIVES, start_date, end_date)

rel = (cyc / defn) * 100  # scale for better visual match

rel_ma50  = rel.rolling(50).mean()
rel_ma200 = rel.rolling(200).mean()

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

rsi = compute_rsi(rel, window=14)

signal = pd.Series(index=rel.index, dtype="object")
signal[(rel_ma50 > rel_ma200) & (rel_ma50.shift(1) <= rel_ma200.shift(1))] = "up"
signal[(rel_ma50 < rel_ma200) & (rel_ma50.shift(1) >= rel_ma200.shift(1))] = "down"

fig = go.Figure()
fig.add_trace(go.Scatter(x=rel.index, y=rel, mode='lines', name='Cyc/Def Rel', line=dict(color='#355E3B')))
fig.add_trace(go.Scatter(x=rel_ma50.index, y=rel_ma50, mode='lines', name='50D MA', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=rel_ma200.index, y=rel_ma200, mode='lines', name='200D MA', line=dict(color='red')))

for date, sig in signal.dropna().items():
    if sig == "up":
        fig.add_trace(go.Scatter(x=[date], y=[rel_ma50[date]], mode='markers',
                                 marker_symbol='arrow-up', marker_size=15, marker_color='green', name='Positive Signal'))
    else:
        fig.add_trace(go.Scatter(x=[date], y=[rel_ma50[date]], mode='markers',
                                 marker_symbol='arrow-down', marker_size=15, marker_color='red', name='Negative Signal'))

fig.update_layout(
    height=600, width=1000,
    margin=dict(l=20, r=20, t=60, b=40),
    title=None,
    yaxis=dict(range=[rel.min()-10, rel.max()+10], title="Relative Ratio"),
    xaxis=dict(title="Date"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=rel.index, y=rsi, mode='lines', name='RSI (14)', line=dict(color='black')))
fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
fig_rsi.update_layout(height=250, width=1000,
    margin=dict(l=20, r=20, t=40, b=40),
    yaxis=dict(title="RSI", range=[0, 100]),
    xaxis=dict(title="Date"),
    title="Overbought / Oversold"
)

st.plotly_chart(fig, use_container_width=True)
st.plotly_chart(fig_rsi, use_container_width=True)
