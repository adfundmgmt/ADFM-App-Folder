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
import re

st.set_page_config(
    page_title="Correlation Dashboard — AD Fund Management LP",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Ticker Correlation Dashboard")

# ─── Sidebar: About + Inputs ──────────────────────────────
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    Institutional correlation regime dashboard  
    • Up to 3 tickers (equity, ETF, or index)  
    • Daily, weekly, or monthly log returns  
    • Spearman (rank-based) correlations  
    • Rolling annualized volatility  
    • Download all outputs  
    """)
    st.markdown("---")
    st.header("Inputs")
    ticker_x = st.text_input("Ticker X", value="AAPL", help="First ticker (required)").strip().upper()
    ticker_y = st.text_input("Ticker Y", value="MSFT", help="Second ticker (required)").strip().upper()
    ticker_z = st.text_input("Ticker Z (optional)", value="", help="Third ticker or benchmark (optional)").strip().upper()
    freq = st.selectbox("Return Frequency", options=["Daily", "Weekly", "Monthly"], index=0)
    roll_window = st.slider("Rolling Window (periods)", 20, 120, value=60)

# ─── Helper Functions ──────────────────────────────
def clean_ticker(t):
    """Remove extra spaces and only allow A-Z0-9-.^ for tickers."""
    return re.sub(r'[^A-Z0-9\-\.\^]', '', t.upper().strip())

ticker_x, ticker_y, ticker_z = map(clean_ticker, [ticker_x, ticker_y, ticker_z])

def validate_ticker(ticker):
    if not ticker:
        return False
    try:
        hist = yf.download(ticker, period="5d", progress=False)
        return not hist.empty
    except Exception:
        return False

bad_tickers = [tk for tk in [ticker_x, ticker_y, ticker_z] if tk and not validate_ticker(tk)]
if bad_tickers:
    st.error(f"Invalid ticker(s): {', '.join(bad_tickers)}. Please correct and retry.")
    st.stop()

symbols = list(filter(bool, [ticker_x, ticker_y, ticker_z]))
symbols = sorted(set(symbols))  # De-duplicate

# ─── Date Range Setup ──────────────────────────────
end_date = dt.date.today()
windows = {
    "Year-to-date": dt.date(end_date.year, 1, 1),
    "3 Months": end_date - relativedelta(months=3),
    "6 Months": end_date - relativedelta(months=6),
    "9 Months": end_date - relativedelta(months=9),
    "1 Year": end_date - relativedelta(years=1),
    "3 Years": end_date - relativedelta(years=3),
    "5 Years": end_date - relativedelta(years=5),
    "10 Years": end_date - relativedelta(years=10),
}
earliest_date = min(windows.values()) - relativedelta(months=1)

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
        # Single ticker, not multi-index
        adj = df[["Adj Close"]].rename(columns={"Adj Close": symbols[0]})
    adj = adj.dropna(axis=1, how="all")
    if adj.empty:
        return None, "No valid price data found for tickers."
    return adj.dropna(how="all"), None

prices, fetch_err = fetch_prices(symbols, start=earliest_date, end=end_date)
if fetch_err or prices is None or prices.empty:
    st.error(fetch_err or "No price data returned — check ticker symbols and try again.")
    st.stop()

# ─── Frequency Handling ──────────────────────────────
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

# ─── Data Coverage Warning ──────────────────────────────
min_required = roll_window + 10  # Require at least this many points for robust results
for tk in symbols:
    if tk not in returns.columns or returns[tk].dropna().size < min_required:
        st.warning(
            f"Warning: {tk} has insufficient price history for the selected rolling window. "
            "Results for this ticker may be unstable or missing in charts below."
        )

# ─── Indexed Price Chart (Plotly) ──────────────────────────────
st.subheader("Indexed Price Performance")
plot_tickers = [t for t in [ticker_x, ticker_y, ticker_z] if t in prices.columns]
fig = go.Figure()
for tk in plot_tickers:
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

# ─── Rolling Correlation Chart ──────────────────────────────
def get_rolling_corr(s1, s2, window):
    s1r = s1.rank()
    s2r = s2.rank()
    return s1r.rolling(window).corr(s2r)

st.subheader("Rolling Correlation (Spearman)")
corr_df = pd.DataFrame(index=returns.index)
label_xy = f"{ticker_x} vs {ticker_y}"
if ticker_x in returns.columns and ticker_y in returns.columns:
    corr_df[label_xy] = get_rolling_corr(returns[ticker_x], returns[ticker_y], roll_window)
if ticker_z and ticker_z in returns.columns:
    label_xz = f"{ticker_x} vs {ticker_z}"
    label_yz = f"{ticker_y} vs {ticker_z}"
    if ticker_x in returns.columns and ticker_z in returns.columns:
        corr_df[label_xz] = get_rolling_corr(returns[ticker_x], returns[ticker_z], roll_window)
    if ticker_y in returns.columns and ticker_z in returns.columns:
        corr_df[label_yz] = get_rolling_corr(returns[ticker_y], returns[ticker_z], roll_window)
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
    yaxis_title="Rolling Correlation",
    xaxis_title="Date",
    legend_title="Pair",
    template="plotly_white",
    height=400,
    hovermode="x unified",
    yaxis=dict(range=[-1, 1])
)
st.plotly_chart(fig_corr, use_container_width=True)

# ─── Rolling Volatility Chart ──────────────────────────────
st.subheader("Rolling Annualized Volatility")
fig_vol = go.Figure()
periods_per_year = {"Daily": 252, "Weekly": 52, "Monthly": 12}[freq]
for tk in plot_tickers:
    vol = returns[tk].rolling(roll_window).std() * np.sqrt(periods_per_year)
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

# ─── Spearman Correlation Matrix (heatmap) ──────────────────────────────
st.subheader("Correlation Heatmap (Spearman)")
def spearman_corr_matrix(df):
    ranked = df.rank(axis=0)
    return ranked.corr(method="pearson")

corr_matrices = {}
for label, since in windows.items():
    ret_slice = returns.loc[returns.index >= pd.Timestamp(since), plot_tickers]
    mat = spearman_corr_matrix(ret_slice)
    corr_matrices[label] = mat

selected_window = st.selectbox("Select look-back window:", list(windows.keys()), index=0)
fig_heat = px.imshow(
    corr_matrices[selected_window].round(3),
    text_auto='.3f',
    aspect="auto",
    color_continuous_scale="RdBu",
    zmin=-1, zmax=1,
    title=f"Correlation Heatmap, {selected_window}"
)
fig_heat.update_layout(height=340)
st.plotly_chart(fig_heat, use_container_width=True)

# ─── Correlation Table by Window (Spearman, 3 decimals + color codes) ──────────────────────────────
st.subheader("Correlation Regime Table by Window (Spearman)")
def format_corr(val):
    """Return emoji + formatted float string."""
    if pd.isna(val): return ""
    absval = abs(val)
    if absval >= 0.7: emoji = "🟢"
    elif absval >= 0.3: emoji = "🟡"
    else: emoji = "🔴"
    return f"{emoji} {val:.3f}"

rows = []
for label, since in windows.items():
    ret_slice = returns.loc[returns.index >= pd.Timestamp(since), plot_tickers]
    mat = spearman_corr_matrix(ret_slice)
    row = {"Window": label}
    # X↔Y
    try: row["X vs Y"] = mat.loc[ticker_x, ticker_y]
    except: row["X vs Y"] = np.nan
    if ticker_z:
        try: row["X vs Z"] = mat.loc[ticker_x, ticker_z]
        except: row["X vs Z"] = np.nan
        try: row["Y vs Z"] = mat.loc[ticker_y, ticker_z]
        except: row["Y vs Z"] = np.nan
    rows.append(row)

df_corr = pd.DataFrame(rows)
df_corr_fmt = df_corr.copy()
for col in df_corr.columns[1:]:
    df_corr_fmt[col] = df_corr[col].apply(format_corr)
st.dataframe(df_corr_fmt.set_index("Window"), height=340)

# ─── Data Download (ZIP) ──────────────────────────────
with st.expander("Download All Outputs (.zip)"):
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zipf:
        zipf.writestr("indexed_prices.csv", prices.to_csv(index=True))
        zipf.writestr("log_returns.csv", returns.to_csv(index=True))
        zipf.writestr("correlation_table.csv", df_corr.to_csv(index=False))
        for label, mat in corr_matrices.items():
            zipf.writestr(f"corr_matrix_{label.replace(' ', '_').lower()}.csv", mat.to_csv(index=True))
        # Add simple README
        zipf.writestr("README.txt",
            "Institutional Correlation Dashboard Output\n"
            "• indexed_prices.csv: Price data, base-100\n"
            "• log_returns.csv: Log returns (resampled)\n"
            "• correlation_table.csv: Lookback regime summary\n"
            "• corr_matrix_*.csv: Spearman correlation matrices by window"
        )
    st.download_button(
        "Download ZIP",
        data=zbuf.getvalue(),
        file_name="correlation_dashboard_outputs.zip",
        mime="application/zip"
    )

st.caption("© 2025 AD Fund Management LP | All rights reserved")
