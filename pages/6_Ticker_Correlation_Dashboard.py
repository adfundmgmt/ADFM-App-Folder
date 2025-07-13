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
    All correlations are **Spearman** (rank-based, rolling).  
    Data: Yahoo Finance (Adj Close, total return).
    """)
    st.markdown("---")
    st.header("Inputs")
    ticker_x = st.text_input("Ticker X", value="AAPL").strip().upper()
    ticker_y = st.text_input("Ticker Y", value="MSFT").strip().upper()
    ticker_z = st.text_input("Ticker Z (optional)", value="", help="Optional: benchmark, ETF, or index").strip().upper()
    freq = st.selectbox("Return Frequency", ["Daily", "Weekly", "Monthly"], index=0)
    roll_window = st.slider("Rolling Window (periods)", 20, 120, value=60)

# ─── Helpers ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_prices(symbols, start, end):
    """Fetch adjusted close for each symbol as flat columns."""
    try:
        data = yf.download(
            symbols,
            start=start,
            end=end + dt.timedelta(days=1),
            progress=False,
            auto_adjust=False,
        )
    except Exception as e:
        return None, f"Data fetch failed: {e}"
    if data is None or data.empty:
        return None, "No data returned (bad ticker or unavailable data)."
    # MultiIndex for >1 symbol, else single index
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data:
            adj = data["Adj Close"].copy()
        else:
            return None, "Missing 'Adj Close' in downloaded data."
        adj.columns = [str(col) for col in adj.columns]
    else:
        # single symbol, must ensure proper column name
        col = symbols[0]
        if "Adj Close" in data:
            adj = data[["Adj Close"]].rename(columns={"Adj Close": col})
        else:
            return None, "Missing 'Adj Close' in downloaded data."
    # Defensive dropna: only drop rows where *all* are nan
    return adj.dropna(how="all"), None

def validate_ticker(ticker):
    """Minimal validation by Yahoo Finance Ticker.info presence."""
    if not ticker: return False
    try:
        info = yf.Ticker(ticker).info
        return bool(info and ('shortName' in info or 'symbol' in info))
    except Exception:
        return False

def get_rolling_corr(s1, s2, window):
    """Rolling Spearman correlation: rank before rolling Pearson."""
    s1r = s1.rank()
    s2r = s2.rank()
    # If too short, return NaNs
    if len(s1r.dropna()) < window or len(s2r.dropna()) < window:
        return pd.Series(index=s1.index, data=np.nan)
    return s1r.rolling(window).corr(s2r)

def spearman_corr_matrix(df):
    """Spearman corr matrix via rank transform, Pearson corr."""
    if df.shape[1] < 2:  # Can't compute matrix for 1 col
        mat = pd.DataFrame(index=df.columns, columns=df.columns)
        for c in df.columns:
            mat.loc[c, c] = 1.0
        return mat
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

# ─── Validate Tickers Before Proceeding ───────────────────
bad_tickers = []
for tk in [ticker_x, ticker_y, ticker_z]:
    if tk and not validate_ticker(tk):
        bad_tickers.append(tk)
if bad_tickers:
    st.error(f"Invalid ticker(s): {', '.join(bad_tickers)}. Please correct.")
    st.stop()

# ─── Download Data ────────────────────────────────────────
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

# ─── Resample to Frequency ────────────────────────────────
def resample_prices(prices, freq):
    if freq == "Daily": return prices
    if freq == "Weekly": return prices.resample("W-FRI").last()
    if freq == "Monthly": return prices.resample("M").last()
    raise ValueError("Unknown frequency.")

try:
    prices = resample_prices(prices, freq)
except Exception as e:
    st.error(f"Resampling failed: {e}")
    st.stop()

returns = np.log(prices / prices.shift(1)).dropna(how="all")

# Defensive: Ensure all tickers have data for downstream use
missing = [t for t in symbols if t not in returns.columns]
if missing:
    st.warning(f"No return data for: {', '.join(missing)} — excluded from analysis.")
    for m in missing:
        if m in symbols:
            symbols.remove(m)

# --- Key ticker pairs for correlations ---
pairs = []
if ticker_x in returns.columns and ticker_y in returns.columns:
    pairs.append((ticker_x, ticker_y))
if ticker_z and ticker_z in returns.columns:
    if ticker_x in returns.columns: pairs.append((ticker_x, ticker_z))
    if ticker_y in returns.columns: pairs.append((ticker_y, ticker_z))

if not pairs:
    st.error("Insufficient data for correlation analysis. Please check tickers or data availability.")
    st.stop()

# ─── 1. Key Regime Stats Summary ──────────────────────────
with st.container():
    st.subheader("Latest & Regime Stats")
    regime_rows = []
    for pair in pairs:
        s1 = returns[pair[0]]
        s2 = returns[pair[1]]
        roll = get_rolling_corr(s1, s2, roll_window)
        latest = roll.dropna().iloc[-1] if not roll.dropna().empty else np.nan
        minv = roll.min()
        maxv = roll.max()
        medv = roll.median()
        regime_rows.append({
            "Pair": f"{pair[0]} vs {pair[1]}",
            "Latest": pct_fmt(latest),
            "Median": pct_fmt(medv),
            "Min": pct_fmt(minv),
            "Max": pct_fmt(maxv),
            "Regime": emoji_corr(latest)
        })
    df_regime = pd.DataFrame(regime_rows).set_index("Pair")
    st.table(df_regime)

# ─── 2. Rolling Correlation Plot (Spearman, % y-axis) ─────
st.subheader("Rolling Correlation")
fig_corr = go.Figure()
for pair in pairs:
    s1, s2 = returns[pair[0]], returns[pair[1]]
    roll = get_rolling_corr(s1, s2, roll_window)
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

# ─── 3. Correlation Table by Window (Spearman, %) ─────────
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
        except Exception:
            row[f"{pair[0]} vs {pair[1]}"] = ""
    rows.append(row)
df_corr_disp = pd.DataFrame(rows).set_index("Window")
st.dataframe(df_corr_disp, height=320)

# ─── 4. Heatmap for Selected Window (Spearman, %) ─────────
st.subheader("Correlation Matrix Heatmap")
selected_window = st.selectbox("Window", list(windows.keys()), index=0)
ret_slice = returns.loc[returns.index >= pd.Timestamp(windows[selected_window])]
mat = spearman_corr_matrix(ret_slice)
matrix_pct = mat.applymap(lambda v: v*100 if pd.notnull(v) else np.nan)
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

# ─── 5. Download ──────────────────────────────────────────
with st.expander("Download Raw Data (.zip)"):
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zipf:
        zipf.writestr("prices.csv", prices.to_csv(index=True))
        zipf.writestr("returns.csv", returns.to_csv(index=True))
        zipf.writestr("correlation_table.csv", df_corr_disp.to_csv())
        zipf.writestr("window_corr_matrix.csv", mat.to_csv(index=True))
    zbuf.seek(0)
    st.download_button(
        "Download ZIP",
        data=zbuf.getvalue(),
        file_name="correlation_dashboard_all_outputs.zip",
        mime="application/zip"
    )

st.caption("© 2025 AD Fund Management LP")
