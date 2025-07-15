import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import datetime as dt

st.set_page_config(page_title="Sector Correlation Dashboard", layout="wide")
st.title("Sector Correlation Dashboard: Cyclicals vs. Defensives")

# ---- Define sector groups ----
cyclicals = ["XLK", "XLI", "XLF", "XLC", "XLY"]
defensives = ["XLP", "XLE", "XLV", "XLRE", "XLU"]
all_sectors = cyclicals + defensives

# ---- Sidebar options ----
with st.sidebar:
    st.header("Settings")
    st.markdown("Analyze rolling and static correlations between key S&P 500 sector ETFs.")
    start_date = st.date_input("Start Date", value=dt.date.today() - dt.timedelta(days=365*3))
    end_date = st.date_input("End Date", value=dt.date.today())
    roll_window = st.slider("Rolling Window (days)", 20, 120, 60)

# ---- Load price data ----
@st.cache_data
def load_sector_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)["Adj Close"]
    return df

prices = load_sector_data(all_sectors, start_date, end_date)
returns = prices.pct_change().dropna()

# ---- Helper: Correlation Table ----
def styled_corr_table(corr_df):
    styled = corr_df.style.format("{:.2f}").background_gradient(
        cmap="RdYlGn", axis=None, gmap=corr_df.values)
    return styled

# ---- Static Correlation Tables ----
st.subheader("Pairwise Correlations")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Cyclicals**")
    corr_cyc = returns[cyclicals].corr(method="spearman")
    st.dataframe(corr_cyc.style.background_gradient(cmap="RdYlGn"))

with col2:
    st.markdown("**Defensives**")
    corr_def = returns[defensives].corr(method="spearman")
    st.dataframe(corr_def.style.background_gradient(cmap="RdYlGn"))

st.markdown("**Cyclicals vs. Defensives (Average Correlation)**")
cyc_def_corr = returns[cyclicals].corrwith(returns[defensives].mean(axis=1), method="spearman")
st.write(cyc_def_corr.to_frame("Corr with Defensives Mean").style.background_gradient(cmap="RdYlGn"))

# ---- Rolling Correlations ----
st.subheader("Rolling Correlation Heatmaps")

# 1. Cyclicals rolling correlation
rolling_corr_cyc = returns[cyclicals].rolling(roll_window).corr().dropna()
rolling_corr_def = returns[defensives].rolling(roll_window).corr().dropna()

# Show average rolling pairwise correlation over time
cyc_pairs = [(i, j) for idx, i in enumerate(cyclicals) for j in cyclicals[idx+1:]]
def_pairs = [(i, j) for idx, i in enumerate(defensives) for j in defensives[idx+1:]]

avg_rolling_corr_cyc = pd.DataFrame({
    f"{i}-{j}": rolling_corr_cyc.xs(i, level=1)[j].values
    for i, j in cyc_pairs
}, index=rolling_corr_cyc.index.levels[0][roll_window-1:])

avg_rolling_corr_def = pd.DataFrame({
    f"{i}-{j}": rolling_corr_def.xs(i, level=1)[j].values
    for i, j in def_pairs
}, index=rolling_corr_def.index.levels[0][roll_window-1:])

tab1, tab2 = st.tabs(["Cyclicals", "Defensives"])
with tab1:
    st.markdown("**Cyclicals: Rolling Pairwise Correlations**")
    fig = px.line(avg_rolling_corr_cyc, labels={"value":"Corr", "index":"Date"})
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("**Defensives: Rolling Pairwise Correlations**")
    fig = px.line(avg_rolling_corr_def, labels={"value":"Corr", "index":"Date"})
    st.plotly_chart(fig, use_container_width=True)

# ---- Cyclicals vs Defensives rolling correlation ----
st.subheader("Rolling Mean: Cyclicals vs. Defensives")
rolling_cyc = returns[cyclicals].mean(axis=1)
rolling_def = returns[defensives].mean(axis=1)
rolling_corr = rolling_cyc.rolling(roll_window).corr(rolling_def)

fig = px.line(rolling_corr, labels={"value":"Corr", "index":"Date"}, title="Rolling Correlation: Mean Cyclicals vs Mean Defensives")
st.plotly_chart(fig, use_container_width=True)

# ---- Option: Download Data ----
st.sidebar.markdown("---")
st.sidebar.markdown("**Download Data**")
csv = returns.to_csv().encode()
st.sidebar.download_button("Download Returns CSV", csv, "sector_returns.csv")

