import math
import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="Cross-Asset Correlations and Vol", layout="wide")

st.title("Cross-Asset Correlation and Volatility Dashboard")
st.caption("Data source: Yahoo Finance via yfinance.")

# -------------------------------
# Utilities
# -------------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_prices(tickers, start, end, interval="1d"):
    """
    Download Adjusted Close for a list of tickers.
    Returns a wide DataFrame of prices with Date index.
    """
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if isinstance(data.columns, pd.MultiIndex):
        frames = []
        for t in tickers:
            if t not in data.columns.get_level_values(0):
                continue
            df = data[t].copy()
            col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else df.columns[0])
            s = df[col].rename(t).to_frame()
            frames.append(s)
        if len(frames) == 0:
            return pd.DataFrame()
        wide = pd.concat(frames, axis=1)
    else:
        # single ticker path
        if data.empty:
            return pd.DataFrame()
        col = "Adj Close" if "Adj Close" in data.columns else ("Close" if "Close" in data.columns else data.columns[0])
        wide = data[[col]].rename(columns={col: tickers[0]})
    wide.index.name = "Date"
    wide = wide.sort_index()
    return wide

def pct_returns(prices):
    return prices.pct_change().dropna(how="all")

def realized_vol(returns, window=21, annualization=252):
    return returns.rolling(window).std() * math.sqrt(annualization)

def zscores(series, lookback=2520):
    s = pd.Series(series).dropna()
    if s.empty:
        return series * np.nan
    m = s.rolling(lookback, min_periods=max(20, lookback//10)).mean()
    sd = s.rolling(lookback, min_periods=max(20, lookback//10)).std()
    return (s - m) / sd

def safe_plot(fig):
    st.pyplot(fig, clear_figure=True, use_container_width=True)

def annotate_latest(ax, series, fmt="{:.2f}"):
    s = pd.Series(series).dropna()
    if s.empty:
        return
    x = s.index[-1]
    y = s.iloc[-1]
    ax.scatter([x], [y])
    ax.text(x, y, " " + fmt.format(y))

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("Settings")

default_start = datetime.now() - timedelta(days=365*12)
start_date = st.sidebar.date_input("Start date", value=default_start.date())
end_date = st.sidebar.date_input("End date", value=datetime.now().date())
rolling_days = st.sidebar.number_input("Rolling window (trading days)", min_value=5, max_value=126, value=21, step=1)

zlb = st.sidebar.number_input("Z-score lookback (days)", min_value=252, max_value=5040, value=2520, step=21)

st.sidebar.divider()
st.sidebar.caption("Tickers are editable. Keep Yahoo Finance symbols.")

# Equity indices
eq_tickers = st.sidebar.text_area("Equity indices (comma separated)",
                                  value="^GSPC,^RUT,^N225,EEM")
# Credit ETFs
credit_tickers = st.sidebar.text_area("Credit ETFs (comma separated)", value="LQD,HYG")
# Rates proxies (yields or bond ETFs)
rates_tickers = st.sidebar.text_area("Rates and bond proxies (comma separated)",
                                     value="^TNX,^TYX,IEF,TLT")
# Commodities
cmd_tickers = st.sidebar.text_area("Commodities (comma separated)",
                                   value="CL=F,GC=F,HG=F,GLD,USO")
# FX
fx_tickers = st.sidebar.text_area("FX pairs (comma separated)", value="EURUSD=X,USDJPY=X,GBPUSD=X")

# Implied vol indices on Yahoo (optional)
impvol_tickers = st.sidebar.text_area("Implied vol indices, optional (comma separated)",
                                      value="^VIX,^OVX")

def parse_tickers(text):
    return [t.strip() for t in text.split(",") if t.strip()]

groups = {
    "Equities": parse_tickers(eq_tickers),
    "Credit": parse_tickers(credit_tickers),
    "Rates": parse_tickers(rates_tickers),
    "Commodities": parse_tickers(cmd_tickers),
    "FX": parse_tickers(fx_tickers),
}
implied_vol_list = parse_tickers(impvol_tickers)

all_tickers = sorted(list(set(sum(groups.values(), []))))

st.sidebar.write(f"Total tickers: {len(all_tickers)}")

# -------------------------------
# Download data
# -------------------------------
if start_date >= end_date:
    st.error("Start date must be earlier than end date.")
    st.stop()

with st.spinner("Downloading market data from Yahoo Finance..."):
    price_wide = fetch_prices(all_tickers, str(start_date), str(end_date), interval="1d")

if price_wide.empty:
    st.error("No data downloaded. Check tickers and date range.")
    st.stop()
else:
    st.success(f"Fetched {price_wide.shape[1]} series and {price_wide.shape[0]} rows.")

# -------------------------------
# Correlation Matrix (1M)
# -------------------------------
st.header("Cross-Asset Correlation Matrix (1M)")
# Use last N trading days
rets_all = pct_returns(price_wide)
rets_last = rets_all.tail(rolling_days)

corr_matrix = rets_last.corr().replace([-np.inf, np.inf], np.nan)

def build_group_labels(cols):
    label_map = {}
    for gname, ticklist in groups.items():
        for t in ticklist:
            label_map[t] = gname
    return pd.Series([label_map.get(c, "Other") for c in cols], index=cols, name="Group")

group_labels = build_group_labels(corr_matrix.columns)
corr_display = corr_matrix.copy()
corr_display.index = corr_display.index.map(lambda x: f"{x}  [{group_labels.loc[x]}]")

st.dataframe(corr_display.style.format("{:.0%}"))

# -------------------------------
# Correlation Analysis Plots
# -------------------------------
st.header("Cross-Asset Correlation Analysis")

def plot_rolling_corr(prices_a, prices_b, label_a, label_b, window=21):
    r = pd.DataFrame({label_a: prices_a, label_b: prices_b}).dropna()
    rcorr = r[label_a].pct_change().rolling(window).corr(r[label_b].pct_change())
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(rcorr.index, rcorr.values)
    ax.set_title(f"Rolling {window}-day Correlation: {label_a} vs {label_b}")
    ax.set_ylabel("Correlation")
    ax.grid(True, alpha=0.3)
    annotate_latest(ax, rcorr, "{:.0%}")
    safe_plot(fig)

pairs = [
    ("^GSPC", "^TNX", "SPX", "US 10Y Yield"),
    ("^RUT", "LQD", "RTY", "IG Credit (LQD)"),
    ("^GSPC", "CL=F", "SPX", "WTI Crude (CL=F)"),
    ("^GSPC", "GLD", "SPX", "Gold (GLD)"),
    ("EEM", "EURUSD=X", "EM Eq (EEM)", "EURUSD"),
    ("^N225", "USDJPY=X", "Nikkei 225", "USDJPY"),
]

for a, b, la, lb in pairs:
    if a in price_wide.columns and b in price_wide.columns:
        plot_rolling_corr(price_wide[a], price_wide[b], la, lb, window=rolling_days)

# -------------------------------
# Volatility Monitor
# -------------------------------
st.header("Cross-Asset Volatility Monitor")

realized = realized_vol(rets_all, window=rolling_days)

# Equity vol: VIX vs SPX realized
if "^GSPC" in price_wide.columns:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(realized.index, realized["^GSPC"] * 100, label="SPX 1M Realized Vol")
    if "^VIX" in implied_vol_list:
        vix = fetch_prices(["^VIX"], str(start_date), str(end_date))
        if not vix.empty:
            ax.plot(vix.index, vix["^VIX"], label="VIX Index")
    ax.set_title("Equity Volatility: implied vs realized")
    ax.set_ylabel("Percent")
    ax.grid(True, alpha=0.3)
    ax.legend()
    safe_plot(fig)

# Rates vol: realized for 10Y yield
if "^TNX" in price_wide.columns:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(realized.index, realized["^TNX"] * 100, label="US 10Y Yield 1M Realized Vol")
    ax.set_title("Rates Volatility: realized proxy")
    ax.set_ylabel("Percent")
    ax.grid(True, alpha=0.3)
    ax.legend()
    safe_plot(fig)

# Credit vol: realized for LQD and HYG
for tkr, name in [("LQD", "IG Credit"), ("HYG", "HY Credit")]:
    if tkr in realized.columns:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(realized.index, realized[tkr] * 100, label=f"{name} 1M Realized Vol")
        ax.set_title(f"{name} Volatility: realized proxy")
        ax.set_ylabel("Percent")
        ax.grid(True, alpha=0.3)
        ax.legend()
        safe_plot(fig)

# Oil vol: OVX vs Oil realized
if "CL=F" in price_wide.columns:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(realized.index, realized["CL=F"] * 100, label="Oil 1M Realized Vol")
    if "^OVX" in implied_vol_list:
        ovx = fetch_prices(["^OVX"], str(start_date), str(end_date))
        if not ovx.empty:
            ax.plot(ovx.index, ovx["^OVX"], label="OVX Index")
    ax.set_title("Oil Volatility: implied vs realized")
    ax.set_ylabel("Percent")
    ax.grid(True, alpha=0.3)
    ax.legend()
    safe_plot(fig)

# FX vol: realized only
for tkr in ["EURUSD=X", "USDJPY=X"]:
    if tkr in realized.columns:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(realized.index, realized[tkr] * 100, label=f"{tkr} 1M Realized Vol")
        ax.set_title(f"FX Volatility: realized proxy, {tkr}")
        ax.set_ylabel("Percent")
        ax.grid(True, alpha=0.3)
        ax.legend()
        safe_plot(fig)

# -------------------------------
# Z-Score Snapshot
# -------------------------------
st.header("Cross-Asset Volatility Snapshot (Z-Scores)")

z_panel = {}

# Implied indices
for sym in implied_vol_list:
    iv = fetch_prices([sym], str(start_date), str(end_date))
    if not iv.empty:
        z_panel[sym] = iv.iloc[:, 0]

# Realized proxies (percent)
for sym, label in [("^GSPC", "SPX Realized"),
                   ("^TNX", "US10Y Realized"),
                   ("LQD", "IG Realized"),
                   ("HYG", "HY Realized"),
                   ("CL=F", "Oil Realized"),
                   ("GLD", "Gold Realized"),
                   ("EURUSD=X", "EURUSD Realized"),
                   ("USDJPY=X", "USDJPY Realized")]:
    if sym in realized.columns:
        z_panel[label] = realized[sym] * 100

if len(z_panel) >= 2:
    zdf = pd.DataFrame(z_panel).dropna(how="all")
    zdf = zdf.apply(lambda s: zscores(s, lookback=zlb))
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in zdf.columns:
        ax.plot(zdf.index, zdf[c], label=c)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("Vol standardized to long lookback where available")
    ax.set_ylabel("Z-Score")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    safe_plot(fig)
else:
    st.info("Not enough series to compute a z-score snapshot.")

# -------------------------------
# Data and Downloads
# -------------------------------
st.header("Data")
st.write("Spot prices")
st.dataframe(price_wide.tail().style.format("{:,.4f}"))

st.write("1M correlation matrix")
st.dataframe(corr_matrix.style.format("{:.0%}"))

def to_csv_bytes(df):
    buf = io.StringIO()
    df.to_csv(buf)
    return buf.getvalue().encode()

st.download_button("Download current prices (CSV)", data=to_csv_bytes(price_wide), file_name="prices.csv")
st.download_button("Download 1M correlation matrix (CSV)", data=to_csv_bytes(corr_matrix), file_name="corr_1m.csv")

st.caption("Coverage notes: Yahoo provides VIX and OVX. MOVE, CDX IG/HY, and most FX implied vols are not on Yahoo. The app uses realized volatility proxies where implied data are unavailable.")
