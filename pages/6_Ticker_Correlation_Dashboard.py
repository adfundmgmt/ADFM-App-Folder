import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import plotly.express as px
import io
import zipfile

st.set_page_config(
    page_title="Correlation Dashboard — AD Fund Management LP",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Ticker Correlation Dashboard")

# ─── Sidebar: About + Inputs ──────────────────────────────
with st.sidebar:
    st.markdown("## About This Tool")
    st.markdown("""
    **Purpose**  
    Provides a robust, institutional view of correlations and volatility for up to 3 tickers.

    **Features**  
    • Supports daily, weekly, monthly log returns  
    • Spearman correlation (rank-based, default for all analytics)  
    • Interactive, professional Plotly charts  
    • Rolling volatility  
    • Correlation matrix heatmap  
    • Download all outputs  
    """)
    st.markdown("---")
    st.header("Inputs")
    ticker_x = st.text_input("Ticker X", value="AAPL", help="Primary security").strip().upper()
    ticker_y = st.text_input("Ticker Y", value="MSFT").strip().upper()
    ticker_z = st.text_input(
        "Ticker Z (optional)", 
        value="", 
        help="Benchmark/index (optional; leave blank to skip)."
    ).strip().upper()
    freq = st.selectbox("Return Frequency", options=["Daily", "Weekly", "Monthly"], index=0)
    roll_window = st.slider("Rolling Window (periods)", 20, 120, value=60)

# ─── Helper: Fetch and Validate ───────────────────────────
@st.cache_data(show_spinner=False)
def fetch_prices(symbols, start, end):
    try:
        df = yf.download(
            symbols,
            start=start,
            end=end + dt.timedelta(days=1),
            progress=False,
            auto_adjust=False,
        )
    except Exception as e:
        return None, f"Data fetch failed: {e}"
    if df.empty:
        return None, "No data returned (bad ticker or unavailable data)."
    if isinstance(df.columns, pd.MultiIndex):
        adj = df["Adj Close"].copy()
        adj.columns = adj.columns.get_level_values(0)
    else:
        adj = df[["Adj Close"]].rename(columns={"Adj Close": symbols[0]})
    return adj.dropna(how="all"), None

def validate_ticker(ticker):
    if not ticker: return False
    try:
        info = yf.Ticker(ticker).info
        return 'shortName' in info or 'symbol' in info
    except Exception:
        return False

def get_rolling_corr(s1, s2, window):
    # Spearman by default: rank transform before rolling Pearson
    s1r = s1.rank()
    s2r = s2.rank()
    return s1r.rolling(window).corr(s2r)

# Validate tickers up front
bad_tickers = []
for tk in [ticker_x, ticker_y, ticker_z]:
    if tk and not validate_ticker(tk):
        bad_tickers.append(tk)
if bad_tickers:
    st.error(f"Invalid ticker(s): {', '.join(bad_tickers)}. Please correct.")
    st.stop()

# Prepare date range
end_date = dt.date.today()
windows = {
    "YTD": dt.date(end_date.year, 1, 1),
    "3M": end_date - relativedelta(months=3),
    "6M": end_date - relativedelta(months=6),
    "9M": end_date - relativedelta(months=9),
    "1Y": end_date - relativedelta(years=1),
    "3Y": end_date - relativedelta(years=3),
    "5Y": end_date - relativedelta(years=5),
    "10Y": end_date - relativedelta(years=10),
}
earliest_date = min(windows.values()) - relativedelta(months=1)
symbols = sorted(set(filter(bool, [ticker_x, ticker_y, ticker_z])))

prices, fetch_err = fetch_prices(symbols, start=earliest_date, end=end_date)
if fetch_err or prices is None or prices.empty:
    st.error(fetch_err or "No price data returned — check ticker symbols and try again.")
    st.stop()

# ─── Frequency Handling ───────────────────────────
def resample_prices(prices, freq):
    if freq == "Daily":
        return prices
    elif freq == "Weekly":
        return prices.resample("W-FRI").last()
    elif freq == "Monthly":
        return prices.resample("M").last()
    else:
        raise ValueError("Unknown frequency.")

prices = resample_prices(prices, freq)
returns = np.log(prices / prices.shift(1)).dropna(how="all")

# ─── Overlay: Indexed Price Chart (Plotly) ──────────────
st.subheader("Indexed Price Overlays")
overlay_tickers = [ticker_x, ticker_y] + ([ticker_z] if ticker_z else [])
overlay_tickers = [t for t in overlay_tickers if t in prices.columns]
fig = go.Figure()
for tk in overlay_tickers:
    indexed = prices[tk] / prices[tk].iloc[0] * 100
    fig.add_trace(go.Scatter(
        x=indexed.index,
        y=indexed,
        mode="lines",
        name=tk,
        line=dict(width=2)
    ))
fig.update_layout(
    yaxis_title="Indexed Price (Base=100)",
    xaxis_title="Date",
    legend_title="Ticker",
    template="plotly_white",
    height=400,
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# ─── Rolling Correlation Chart (Plotly, Spearman) ───────
st.subheader("Rolling Correlation Chart (Spearman)")
corr_df = pd.DataFrame(index=returns.index)
corr_df[f"{ticker_x} vs {ticker_y}"] = get_rolling_corr(returns[ticker_x], returns[ticker_y], roll_window)
if ticker_z and ticker_z in returns.columns:
    corr_df[f"{ticker_x} vs {ticker_z}"] = get_rolling_corr(returns[ticker_x], returns[ticker_z], roll_window)
    corr_df[f"{ticker_y} vs {ticker_z}"] = get_rolling_corr(returns[ticker_y], returns[ticker_z], roll_window)
corr_df = corr_df.dropna(how="all")

fig_corr = go.Figure()
for col in corr_df.columns:
    fig_corr.add_trace(go.Scatter(
        x=corr_df.index,
        y=corr_df[col],
        mode="lines",
        name=col,
        line=dict(width=2)
    ))
fig_corr.update_layout(
    yaxis_title="Rolling Correlation (Spearman)",
    xaxis_title="Date",
    legend_title="Pairs",
    template="plotly_white",
    height=400,
    hovermode="x unified",
    yaxis=dict(range=[-1, 1])
)
st.plotly_chart(fig_corr, use_container_width=True)

# ─── Rolling Volatility Chart (Plotly) ────────────────
st.subheader("Rolling Volatility (Annualized)")
fig_vol = go.Figure()
for tk in overlay_tickers:
    vol = returns[tk].rolling(roll_window).std() * np.sqrt({"Daily":252, "Weekly":52, "Monthly":12}[freq])
    fig_vol.add_trace(go.Scatter(
        x=vol.index,
        y=vol,
        mode="lines",
        name=tk,
        line=dict(width=2)
    ))
fig_vol.update_layout(
    yaxis_title="Annualized Volatility",
    xaxis_title="Date",
    legend_title="Ticker",
    template="plotly_white",
    height=400,
    hovermode="x unified"
)
st.plotly_chart(fig_vol, use_container_width=True)

# ─── Correlation Matrix Heatmap (Spearman) ─────────────
st.subheader("Correlation Matrix (Look-back Windows, Spearman)")
def spearman_corr_matrix(df):
    # Rank transform then pearson
    ranked = df.rank(axis=0)
    return ranked.corr(method="pearson")

corr_matrices = {}
for label, since in windows.items():
    ret_slice = returns.loc[returns.index >= pd.Timestamp(since)]
    mat = spearman_corr_matrix(ret_slice[overlay_tickers])
    corr_matrices[label] = mat

selected_window = st.selectbox("Select window for heatmap:", list(windows.keys()), index=0)
fig_heat = px.imshow(
    corr_matrices[selected_window].round(2),
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdBu",
    zmin=-1, zmax=1,
    title=f"Correlation Heatmap (Spearman, {selected_window})"
)
fig_heat.update_layout(height=340)
st.plotly_chart(fig_heat, use_container_width=True)

# ─── Correlation Table by Window (Spearman) ────────────
st.subheader("Correlation Table by Look-Back Window (Spearman)")
rows = []
for label, since in windows.items():
    ret_slice = returns.loc[returns.index >= pd.Timestamp(since)]
    mat = spearman_corr_matrix(ret_slice[overlay_tickers])
    row = {"Window": label}
    # X↔Y
    try:
        row["X vs Y"] = mat.loc[ticker_x, ticker_y].round(3)
    except: row["X vs Y"] = np.nan
    if ticker_z:
        try:
            row["X vs Z"] = mat.loc[ticker_x, ticker_z].round(3)
        except: row["X vs Z"] = np.nan
        try:
            row["Y vs Z"] = mat.loc[ticker_y, ticker_z].round(3)
        except: row["Y vs Z"] = np.nan
    rows.append(row)
df_corr = pd.DataFrame(rows)
st.dataframe(df_corr.set_index("Window"), height=340)

# ─── Data Download (ZIP all outputs) ───────────────────
with st.expander("Download All Outputs (.zip)"):
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zipf:
        zipf.writestr("indexed_prices.csv", prices.to_csv(index=True))
        zipf.writestr("log_returns.csv", returns.to_csv(index=True))
        zipf.writestr("correlation_table.csv", df_corr.to_csv(index=False))
        for label, mat in corr_matrices.items():
            zipf.writestr(f"corr_matrix_{label}.csv", mat.to_csv(index=True))
    st.download_button(
        "Download ZIP",
        data=zbuf.getvalue(),
        file_name="correlation_dashboard_all_outputs.zip",
        mime="application/zip"
    )

st.caption("© 2025 AD Fund Management LP")
