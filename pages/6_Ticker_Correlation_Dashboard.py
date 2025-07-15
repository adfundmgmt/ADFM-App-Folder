import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime as dt

st.set_page_config(page_title="Cyclical vs Defensive Regime Dashboard", layout="wide")
st.title("S&P Cyclicals Relative to Defensives — Equal Weight")

cyclicals = ["XLK", "XLI", "XLF", "XLC", "XLY"]
defensives = ["XLP", "XLE", "XLV", "XLRE", "XLB", "XLU"]
all_sectors = cyclicals + defensives

with st.sidebar:
    st.header("Settings")
    st.markdown(
        "Plots equal-weighted Cyclicals vs Defensives with 50/200-day moving averages and custom RSI."
    )
    start_date = st.date_input("Start Date", value=dt.date.today() - dt.timedelta(days=1100))
    end_date = st.date_input("End Date", value=dt.date.today())
    rsi_window = st.slider("RSI Window", 5, 30, 14)
    ma_short = st.slider("Short MA (days)", 10, 100, 50)
    ma_long = st.slider("Long MA (days)", 100, 300, 200)

@st.cache_data
def load_sector_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            df = df["Adj Close"]
        else:
            last_lvl = df.columns.get_level_values(0).unique()[-1]
            df = df.xs(last_lvl, axis=1, level=0)
    else:
        df = df.to_frame()
    df = df.dropna(axis=1, how="all")
    return df

prices = load_sector_data(all_sectors, start_date, end_date)
returns = prices.pct_change().dropna()

cyclicals_live = [x for x in cyclicals if x in prices.columns]
defensives_live = [x for x in defensives if x in prices.columns]

# Equal-weighted cumulative index
cyc_index = (1 + returns[cyclicals_live]).cumprod().mean(axis=1)
def_index = (1 + returns[defensives_live]).cumprod().mean(axis=1)
ratio = cyc_index / def_index
ratio.name = "Cyclicals/Defensives Ratio"

# Moving averages
ratio_ma_short = ratio.rolling(ma_short).mean()
ratio_ma_long = ratio.rolling(ma_long).mean()

# Custom RSI
def compute_rsi(series, window):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=window).mean()
    roll_down = down.rolling(window=window).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

rsi_series = compute_rsi(ratio, rsi_window)

# Election day vertical line (2024-11-05 for US Election)
election_date = pd.Timestamp("2024-11-05")
if election_date not in ratio.index:
    # find closest available date
    idx = ratio.index.get_indexer([election_date], method='nearest')[0]
    election_date = ratio.index[idx]

# --- Build composite Plotly figure with subplots ---
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True,
    row_heights=[0.66, 0.34],
    vertical_spacing=0.05,
    subplot_titles=(
        "Cyclicals Relative to Defensives (Equal Weight, 50/200-day MA)", 
        f"RSI ({rsi_window}-day) — Overbought / Oversold"
    )
)

# Top panel: Ratio + MAs
fig.add_trace(go.Scatter(
    x=ratio.index, y=ratio, 
    mode='lines', name='Cyc/Def Ratio',
    line=dict(color='green', width=2)
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=ratio_ma_short.index, y=ratio_ma_short,
    mode='lines', name=f'{ma_short}-day MA',
    line=dict(color='blue', width=1, dash='dot')
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=ratio_ma_long.index, y=ratio_ma_long,
    mode='lines', name=f'{ma_long}-day MA',
    line=dict(color='firebrick', width=1.5, dash='dash')
), row=1, col=1)
fig.add_vline(
    x=election_date, line_width=1.5, line_dash="dot", line_color="black", row=1, col=1
)
fig.add_annotation(
    x=election_date, y=ratio.max(), text="Election Day 2024", 
    showarrow=True, arrowhead=1, ay=-40, ax=30, 
    font=dict(size=12), row=1, col=1
)

# Bottom panel: RSI
fig.add_trace(go.Scatter(
    x=rsi_series.index, y=rsi_series, 
    mode='lines', name=f'RSI({rsi_window})', 
    line=dict(color='black', width=1.5)
), row=2, col=1)
fig.add_hline(y=70, line_dash='dash', line_color='red', row=2, col=1)
fig.add_hline(y=30, line_dash='dash', line_color='green', row=2, col=1)
fig.update_yaxes(title_text="Ratio", row=1, col=1)
fig.update_yaxes(title_text="RSI", row=2, col=1)
fig.update_layout(
    height=720, width=1100,
    showlegend=True,
    margin=dict(t=80, l=40, r=40, b=40),
    title=dict(
        text="S&P Cyclicals Relative to Defensives — Equal Weight",
        font=dict(size=22)
    ),
    plot_bgcolor="#fff",
    paper_bgcolor="#fff",
)

st.plotly_chart(fig, use_container_width=True)

# Current regime
latest_ratio = ratio.dropna().iloc[-1]
latest_rsi = rsi_series.dropna().iloc[-1]
if latest_rsi > 70:
    regime = "Overbought"
elif latest_rsi < 30:
    regime = "Oversold"
else:
    regime = "Neutral"

st.markdown("### Regime Table")
regime_df = pd.DataFrame({
    "Latest Ratio": [f"{latest_ratio:.2f}"],
    f"RSI ({rsi_window})": [f"{latest_rsi:.1f}"],
    "Status": [regime],
})
st.table(regime_df)

st.sidebar.markdown("---")
st.sidebar.markdown("**Download Series**")
export_df = pd.DataFrame({
    "Cyc_Def_Ratio": ratio, 
    f"MA_{ma_short}": ratio_ma_short, 
    f"MA_{ma_long}": ratio_ma_long,
    f"RSI_{rsi_window}": rsi_series
})
csv = export_df.to_csv().encode()
st.sidebar.download_button("Download CSV", csv, "cyc_def_ratio_ma_rsi.csv")
