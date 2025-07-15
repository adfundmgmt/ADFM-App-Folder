import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import datetime as dt

st.set_page_config(page_title="Cyclicals vs. Defensives Ratio", layout="wide")
st.title("Cyclicals vs. Defensives: Ratio & RSI Regime Dashboard")

# ---- Sector definitions ----
cyclicals = ["XLK", "XLI", "XLF", "XLC", "XLY"]
defensives = ["XLP", "XLE", "XLV", "XLRE", "XLB", "XLU"]
all_sectors = cyclicals + defensives

# ---- Sidebar: Settings ----
with st.sidebar:
    st.header("Settings")
    st.markdown(
        "Tracks the equal-weighted performance of Cyclicals (XLK, XLI, XLF, XLC, XLY) "
        "vs. Defensives (XLP, XLE, XLV, XLRE, XLB, XLU). Plots the ratio and computes custom RSI."
    )
    start_date = st.date_input("Start Date", value=dt.date.today() - dt.timedelta(days=365*3))
    end_date = st.date_input("End Date", value=dt.date.today())
    rsi_window = st.slider("RSI Window", 5, 30, 14)

# ---- Data Loader ----
@st.cache_data
def load_sector_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)
    # MultiIndex (multi ticker), else single ticker
    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            df = df["Adj Close"]
        else:
            st.warning("Adj Close not found in columns. Using last available level.")
            last_lvl = df.columns.get_level_values(0).unique()[-1]
            df = df.xs(last_lvl, axis=1, level=0)
    else:
        df = df.to_frame()
    df = df.dropna(axis=1, how="all")
    return df

prices = load_sector_data(all_sectors, start_date, end_date)
returns = prices.pct_change().dropna()

# Only use tickers that have price data
cyclicals_live = [x for x in cyclicals if x in prices.columns]
defensives_live = [x for x in defensives if x in prices.columns]
st.write(f"**Cyclicals available:** {cyclicals_live}")
st.write(f"**Defensives available:** {defensives_live}")

# ---- Equal-weighted indices ----
cyc_index = (1 + returns[cyclicals_live]).cumprod().mean(axis=1)
def_index = (1 + returns[defensives_live]).cumprod().mean(axis=1)

# ---- Ratio series ----
ratio = cyc_index / def_index
ratio.name = "Cyclicals/Defensives Ratio"

# ---- Custom RSI function ----
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

# ---- Plot ratio ----
st.subheader("Equal-Weighted Ratio: Cyclicals vs. Defensives")
fig = px.line(ratio, labels={"value": "Cyc/Def Ratio", "index": "Date"}, title="Cyclicals / Defensives Ratio (Equal Weighted)")
st.plotly_chart(fig, use_container_width=True)

# ---- Plot RSI ----
st.subheader(f"RSI ({rsi_window}-day) of Cyclicals/Defensives Ratio")
fig_rsi = px.line(rsi_series, labels={"value": f"RSI({rsi_window})", "index": "Date"})
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
st.plotly_chart(fig_rsi, use_container_width=True)

# ---- Overbought/Oversold Table ----
latest_ratio = ratio.dropna().iloc[-1]
latest_rsi = rsi_series.dropna().iloc[-1]
if latest_rsi > 70:
    regime = "Overbought"
elif latest_rsi < 30:
    regime = "Oversold"
else:
    regime = "Neutral"

st.subheader("Current Regime Table")
regime_df = pd.DataFrame(
    {
        "Latest Ratio": [f"{latest_ratio:.2f}"],
        f"RSI ({rsi_window})": [f"{latest_rsi:.1f}"],
        "Status": [regime],
    }
)
st.table(regime_df)

# ---- Download Data ----
st.sidebar.markdown("---")
st.sidebar.markdown("**Download Series**")
export_df = pd.DataFrame({"Cyc_Def_Ratio": ratio, f"RSI_{rsi_window}": rsi_series})
csv = export_df.to_csv().encode()
st.sidebar.download_button("Download CSV", csv, "cyc_def_ratio_and_rsi.csv")
