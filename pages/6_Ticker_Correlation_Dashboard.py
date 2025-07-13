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
    st.markdown("## About")
    st.markdown("""
    Correlation regime dashboard for two or three tickers.  
    All correlations are **Spearman** (rank-based).  
    Data: Yahoo Finance (Adj Close, total return).
    """)
    st.markdown("---")
    st.header("Inputs")
    ticker_x = st.text_input("Ticker X", value="AAPL").strip().upper()
    ticker_y = st.text_input("Ticker Y", value="MSFT").strip().upper()
    ticker_z = st.text_input("Ticker Z (optional)", value="", help="Benchmark or index (optional)").strip().upper()
    freq = st.selectbox("Return Frequency", ["Daily", "Weekly", "Monthly"], index=0)
    roll_window = st.slider("Rolling Window (periods)", 20, 120, value=60)

# ─── Helpers ───────────────────────────────────────────────
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

def spearman_corr_matrix(df):
    # Rank transform then pearson
    ranked = df.rank(axis=0)
    return ranked.corr(method="pearson")

def emoji_corr(val):
    if pd.isnull(val): return ""
    if val > 0.7: return "🟢"
    if val > 0.3: return "🟡"
    if val < -0.3: return "🔴"
    return "⚪️"

def pct_fmt(val, nan_as_blank=True):
    if pd.isnull(val):
        return "" if nan_as_blank else "N/A"
    return f"{val*100:.1f}%"

# ─── Validate ─────────────────────────────────────────────
bad_tickers = []
for tk in [ticker_x, ticker_y, ticker_z]:
    if tk and not validate_ticker(tk):
        bad_tickers.append(tk)
if bad_tickers:
    st.error(f"Invalid ticker(s): {', '.join(bad_tickers)}. Please correct.")
    st.stop()

# ─── Download data ───────────────────────────────────────
end_date = dt.date.today()
windows = {
    "YTD": dt.date(end_date.year, 1, 1),
    "3M": end_date - relativedelta(months=3),
    "6M": end_date - relativedelta(months=6),
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

# ─── Resample to frequency ───────────────────────────────
def resample_prices(prices, freq):
    if freq == "Daily": return prices
    if freq == "Weekly": return prices.resample("W-FRI").last()
    if freq == "Monthly": return prices.resample("M").last()
    raise ValueError("Unknown frequency.")

prices = resample_prices(prices, freq)
returns = np.log(prices / prices.shift(1)).dropna(how="all")

# --- Key ticker pairs for correlations ---
pairs = [(ticker_x, ticker_y)]
if ticker_z:
    pairs += [(ticker_x, ticker_z), (ticker_y, ticker_z)]

# ─── 1. Key regime stats summary ─────────────────────────
with st.container():
    st.subheader("Latest & Regime Stats")
    regime_rows = []
    for pair in pairs:
        c_label = f"{pair[0]} vs {pair[1]}"
        roll = get_rolling_corr(returns[pair[0]], returns[pair[1]], roll_window)
        latest = roll.dropna().iloc[-1] if not roll.dropna().empty else np.nan
        minv = roll.min()
        maxv = roll.max()
        medv = roll.median()
        regime_rows.append({
            "Pair": c_label,
            "Latest": pct_fmt(latest),
            "Median": pct_fmt(medv),
            "Min": pct_fmt(minv),
            "Max": pct_fmt(maxv),
            "Regime": emoji_corr(latest)
        })
    df_regime = pd.DataFrame(regime_rows).set_index("Pair")
    st.table(df_regime)

# ─── 2. Rolling Correlation Plot (Spearman, % y-axis) ────
st.subheader("Rolling Correlation")
fig_corr = go.Figure()
for pair in pairs:
    roll = get_rolling_corr(returns[pair[0]], returns[pair[1]], roll_window)
    fig_corr.add_trace(go.Scatter(
        x=roll.index, y=roll*100,
        mode="lines",
        name=f"{pair[0]} vs {pair[1]}",
        line=dict(width=2)
    ))
fig_corr.update_layout(
    yaxis_title="Rolling Correlation (Spearman, %)",
    xaxis_title="Date",
    legend_title="Pair",
    template="plotly_white",
    height=400,
    hovermode="x unified",
    yaxis=dict(range=[-100, 100])
)
st.plotly_chart(fig_corr, use_container_width=True)

# ─── 3. Correlation Table by Window (Spearman, %) ────────
st.subheader("Look-Back Window Correlations")
rows = []
for label, since in windows.items():
    ret_slice = returns.loc[returns.index >= pd.Timestamp(since)]
    mat = spearman_corr_matrix(ret_slice)
    row = {"Window": label}
    for pair in pairs:
        try:
            val = mat.loc[pair[0], pair[1]]
            row[f"{pair[0]} vs {pair[1]}"] = f"{pct_fmt(val)} {emoji_corr(val)}"
        except:
            row[f"{pair[0]} vs {pair[1]}"] = ""
    rows.append(row)
df_corr_disp = pd.DataFrame(rows).set_index("Window")
st.dataframe(df_corr_disp, height=320)

# ─── 4. Heatmap for Selected Window (Spearman, %) ────────
st.subheader("Correlation Matrix Heatmap")
selected_window = st.selectbox("Window", list(windows.keys()), index=0)
ret_slice = returns.loc[returns.index >= pd.Timestamp(windows[selected_window])]
mat = spearman_corr_matrix(ret_slice)
matrix_pct = mat.applymap(lambda v: v*100)
fig_heat = px.imshow(
    matrix_pct.round(1),
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdBu",
    zmin=-100, zmax=100,
    title=f"Spearman Correlation Heatmap ({selected_window})"
)
fig_heat.update_layout(height=340)
st.plotly_chart(fig_heat, use_container_width=True)

# ─── 5. Download ─────────────────────────────────────────
with st.expander("Download Raw Data (.zip)"):
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zipf:
        zipf.writestr("prices.csv", prices.to_csv(index=True))
        zipf.writestr("returns.csv", returns.to_csv(index=True))
        zipf.writestr("correlation_table.csv", df_corr_disp.to_csv())
        zipf.writestr("window_corr_matrix.csv", mat.to_csv(index=True))
    st.download_button(
        "Download ZIP",
        data=zbuf.getvalue(),
        file_name="correlation_dashboard_all_outputs.zip",
        mime="application/zip"
    )

st.caption("© 2025 AD Fund Management LP")
