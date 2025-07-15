import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

CYCLICALS  = ["XLK", "XLI", "XLF", "XLC", "XLY"]
DEFENSIVES = ["XLP", "XLE", "XLV", "XLRE", "XLB", "XLU"]

st.set_page_config(layout="wide", page_title="S&P Cyclicals vs Defensives Dashboard")
st.title("S&P Cyclicals Relative to Defensives – Equal-Weight")

# Sidebar
with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
    This dashboard tracks the relative performance of S&P cyclical and defensive sector ETFs (equal-weighted) to visualize risk-on/risk-off regime shifts in US equities.

    **How it works:**
    - Cyclical basket: XLK, XLI, XLF, XLC, XLY
    - Defensive basket: XLP, XLE, XLV, XLRE, XLB, XLU
    - The ratio of cumulative returns (Cyc/Def) is shown, with 50D & 200D moving averages and RSI (14).
    - Select your preferred lookback.
    """)

    st.subheader("Time Frame")
    time_options = {
        "3 Months": 90,
        "6 Months": 180,
        "9 Months": 270,
        "YTD": None,
        "1 Year": 365,
        "3 Years": 365*3,
        "5 Years": 365*5,
        "10 Years": 365*10
    }
    default_ix = list(time_options.keys()).index("5 Years")
    time_choice = st.selectbox(
        "Select the lookback period:",
        list(time_options.keys()),
        index=default_ix
    )

today = datetime.today()
data_start_date = today - timedelta(days=365*10 + 220)  # 10 years + 200D buffer
data_start_str = data_start_date.strftime('%Y-%m-%d')
end_date = today.strftime('%Y-%m-%d')

if time_choice == "YTD":
    display_start = datetime(today.year, 1, 1)
else:
    display_start = today - timedelta(days=time_options[time_choice])
display_start_str = display_start.strftime('%Y-%m-%d')

def basket_price(etfs, start, end):
    data = yf.download(etfs, start=start, end=end, group_by="ticker", auto_adjust=True, progress=False)
    price_df = pd.DataFrame()
    for etf in etfs:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                px = data[etf]["Close"]
            else:
                px = data["Close"]
            price_df[etf] = px
        except Exception:
            continue
    price_df = price_df.fillna(method='ffill').dropna()
    basket = price_df.pct_change().mean(axis=1)
    basket_cum = (1 + basket).cumprod()
    return basket_cum

cyc = basket_price(CYCLICALS, data_start_str, end_date)
defn = basket_price(DEFENSIVES, data_start_str, end_date)

rel = (cyc / defn) * 100
rel = rel.dropna()

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

# Slice for display period (keep MA's full length, slice only rel and RSI)
display_mask = rel.index >= display_start_str
rel_disp  = rel[display_mask]
rsi_disp  = rsi[display_mask]

# Main chart (no signals, always full MA curves)
fig = go.Figure()
fig.add_trace(go.Scatter(x=rel_disp.index, y=rel_disp, mode='lines', name='Cyc/Def Rel', line=dict(color='#355E3B', width=2)))
fig.add_trace(go.Scatter(x=rel_ma50.index, y=rel_ma50, mode='lines', name='50D MA', line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=rel_ma200.index, y=rel_ma200, mode='lines', name='200D MA', line=dict(color='red', width=2)))

fig.update_layout(
    height=600, width=1000,
    margin=dict(l=20, r=20, t=40, b=40),
    font=dict(size=16, family="Arial"),
    yaxis=dict(title="Relative Ratio"),
    xaxis=dict(title="Date", range=[rel_disp.index.min(), rel_disp.index.max()]),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1,
        font=dict(size=14, family="Arial")
    ),
    plot_bgcolor="white"
)

# RSI subplot
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=rsi_disp.index, y=rsi_disp, mode='lines', name='RSI (14)', line=dict(color='black', width=2)))
fig_rsi.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Overbought", annotation_position="top left")
fig_rsi.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold", annotation_position="bottom left")
fig_rsi.update_layout(
    height=220, width=1000,
    margin=dict(l=20, r=20, t=25, b=40),
    font=dict(size=15, family="Arial"),
    yaxis=dict(title="RSI", range=[0, 100]),
    xaxis=dict(title="Date", range=[rel_disp.index.min(), rel_disp.index.max()]),
    legend=dict(
        orientation="h", font=dict(size=14, family="Arial")
    ),
    title="<b>Overbought / Oversold</b>",
    plot_bgcolor="white"
)

st.plotly_chart(fig, use_container_width=True)
st.plotly_chart(fig_rsi, use_container_width=True)
