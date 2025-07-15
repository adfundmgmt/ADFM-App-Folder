import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import datetime as dt

st.set_page_config(page_title="Sector Correlation Dashboard", layout="wide")
st.title("Sector Correlation Dashboard: Cyclicals vs. Defensives")

# ---- Define sector groups ----
cyclicals = ["XLK", "XLI", "XLF", "XLC", "XLY"]
defensives = ["XLP", "XLE", "XLV", "XLRE", "XLB", "XLU"]
all_sectors = cyclicals + defensives

# ---- Sidebar options ----
with st.sidebar:
    st.header("Settings")
    st.markdown(
        "Analyze rolling and static correlations between key S&P 500 sector ETFs.\n\n"
        "Cyclicals: XLK, XLI, XLF, XLC, XLY\n"
        "Defensives: XLP, XLE, XLV, XLRE, XLB, XLU"
    )
    start_date = st.date_input("Start Date", value=dt.date.today() - dt.timedelta(days=365*3))
    end_date = st.date_input("End Date", value=dt.date.today())
    roll_window = st.slider("Rolling Window (days)", 20, 120, 60)

# ---- Load price data ----
@st.cache_data
def load_sector_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)
    # If MultiIndex, select Adj Close; else, just wrap single series as DataFrame
    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            df = df["Adj Close"]
        else:
            # fallback: just try to get price columns (shouldn't happen)
            df = df.xs("Adj Close", axis=1, level=0, drop_level=True)
    else:
        df = df.to_frame()
    # Drop tickers with all NaNs
    df = df.dropna(axis=1, how="all")
    return df

prices = load_sector_data(all_sectors, start_date, end_date)
returns = prices.pct_change().dropna()

st.write("**Loaded sector tickers:**", list(prices.columns))

# ---- Static Correlation Tables ----
st.subheader("Pairwise Correlations")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Cyclicals**")
    corr_cyc = returns[cyclicals].corr(method="spearman")
    st.dataframe(
        corr_cyc.style.background_gradient(cmap="RdYlGn").format("{:.2f}"),
        use_container_width=True,
    )

with col2:
    st.markdown("**Defensives**")
    corr_def = returns[defensives].corr(method="spearman")
    st.dataframe(
        corr_def.style.background_gradient(cmap="RdYlGn").format("{:.2f}"),
        use_container_width=True,
    )

# Show correlation of each cyclical with the defensives average
st.markdown("**Cyclicals vs. Defensives (Correlation with Defensives Mean)**")
def_mean = returns[defensives].mean(axis=1)
cyc_def_corr = returns[cyclicals].corrwith(def_mean, method="spearman")
st.dataframe(
    cyc_def_corr.to_frame("Corr with Defensives Mean").style.background_gradient(cmap="RdYlGn").format("{:.2f}"),
    use_container_width=True,
)

# ---- Rolling Correlations ----
st.subheader("Rolling Correlation Heatmaps")

def get_rolling_corr(df, tickers, window):
    pairs = [(i, j) for idx, i in enumerate(tickers) for j in tickers[idx+1:]]
    out = pd.DataFrame(index=df.index)
    for i, j in pairs:
        out[f"{i}-{j}"] = df[i].rolling(window).corr(df[j])
    return out.dropna()

# 1. Cyclicals rolling
rolling_corr_cyc = get_rolling_corr(returns, cyclicals, roll_window)
# 2. Defensives rolling
rolling_corr_def = get_rolling_corr(returns, defensives, roll_window)

tab1, tab2 = st.tabs(["Cyclicals", "Defensives"])
with tab1:
    st.markdown("**Cyclicals: Rolling Pairwise Correlations**")
    fig = px.line(rolling_corr_cyc, labels={"value": "Corr", "index": "Date"})
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    st.markdown("**Defensives: Rolling Pairwise Correlations**")
    fig = px.line(rolling_corr_def, labels={"value": "Corr", "index": "Date"})
    st.plotly_chart(fig, use_container_width=True)

# ---- Cyclicals vs Defensives rolling correlation ----
st.subheader("Rolling Mean: Cyclicals vs. Defensives")
rolling_cyc = returns[cyclicals].mean(axis=1)
rolling_def = returns[defensives].mean(axis=1)
rolling_corr = rolling_cyc.rolling(roll_window).corr(rolling_def)

fig = px.line(
    rolling_corr,
    labels={"value": "Corr", "index": "Date"},
    title="Rolling Correlation: Mean Cyclicals vs Mean Defensives",
)
st.plotly_chart(fig, use_container_width=True)

# ---- Download Data ----
st.sidebar.markdown("---")
st.sidebar.markdown("**Download Returns Data**")
csv = returns.to_csv().encode()
st.sidebar.download_button("Download Returns CSV", csv, "sector_returns.csv")
