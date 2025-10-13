import math
import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------------
# App Config and style
# -------------------------------
st.set_page_config(page_title="Cross-Asset Correlations and Vol", layout="wide")

plt.rcParams.update({
    "figure.figsize": (8, 3),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

st.title("Cross-Asset Correlation and Volatility Dashboard")
st.caption("Data source: Yahoo Finance via yfinance. Layout mirrors your reference, with the matrix up top and 3x2 panels below.")

# -------------------------------
# Utilities
# -------------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_prices(tickers, start, end, interval="1d"):
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

def annotate_latest(ax, series, fmt="{:.0%}"):
    s = pd.Series(series).dropna()
    if s.empty:
        return
    x = s.index[-1]
    y = s.iloc[-1]
    ax.scatter([x], [y])
    ax.text(x, y, " " + fmt.format(y))

def style_corr(df):
    # diverging background from -100% to +100%
    return (df.style
            .format("{:.0%}")
            .background_gradient(cmap="PiYG", vmin=-1, vmax=1)
            .set_table_styles([{"selector": "th.col_heading",
                                "props": [("text-align", "center")]},
                               {"selector": "th.row_heading",
                                "props": [("text-align", "left")]}]))

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
st.sidebar.caption("Tickers are editable. Keep Yahoo symbols.")

eq_tickers = st.sidebar.text_area("Equity indices",
                                  value="^GSPC,^RUT,^N225,EEM")
credit_tickers = st.sidebar.text_area("Credit ETFs", value="LQD,HYG")
rates_tickers = st.sidebar.text_area("Rates and bond proxies",
                                     value="^TNX,^TYX,IEF,TLT")
cmd_tickers = st.sidebar.text_area("Commodities",
                                   value="CL=F,GC=F,HG=F,GLD,USO")
fx_tickers = st.sidebar.text_area("FX pairs", value="EURUSD=X,USDJPY=X,GBPUSD=X")
impvol_tickers = st.sidebar.text_area("Implied vol tickers (optional)", value="^VIX,^OVX")

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

rets_all = pct_returns(price_wide)
rets_last = rets_all.tail(rolling_days)
corr_matrix = rets_last.corr().replace([-np.inf, np.inf], np.nan)

# -------------------------------
# TOP: Big Correlation Matrix
# -------------------------------
st.subheader("Cross-Asset Correlation Matrix (1M)")
st.caption("Rolling 1M returns, percentage format, diverging color map. Use the CSV below to share.")
st.dataframe(style_corr(corr_matrix), use_container_width=True)

c1, c2 = st.columns([1, 1])
with c1:
    st.download_button("Download matrix (CSV)",
                       data=corr_matrix.to_csv().encode(),
                       file_name="corr_1m.csv")

with c2:
    st.download_button("Download prices (CSV)",
                       data=price_wide.to_csv().encode(),
                       file_name="prices.csv")

# -------------------------------
# MIDDLE: Correlation Analysis, 3 x 2
# -------------------------------
st.subheader("Cross-Asset Correlation Analysis (Rolling 1M)")
pairs = [
    ("^GSPC", "^TNX", "SPX", "US 10Y Yield"),
    ("^RUT", "LQD", "RTY", "IG Credit (LQD)"),
    ("^GSPC", "CL=F", "SPX", "WTI Crude (CL=F)"),
    ("^GSPC", "GLD", "SPX", "Gold (GLD)"),
    ("EEM", "EURUSD=X", "EM Eq (EEM)", "EURUSD"),
    ("^N225", "USDJPY=X", "Nikkei 225", "USDJPY"),
]

def plot_rolling_corr(ax, prices_a, prices_b, la, lb, window=21):
    r = pd.DataFrame({la: prices_a, lb: prices_b}).dropna()
    rcorr = r[la].pct_change().rolling(window).corr(r[lb].pct_change())
    ax.plot(rcorr.index, rcorr.values)
    ax.set_title(f"{la} vs {lb}")
    ax.set_ylabel("Correlation")
    annotate_latest(ax, rcorr, "{:.0%}")

rows = [pairs[:3], pairs[3:]]
for row in rows:
    col1, col2, col3 = st.columns(3)
    for (a, b, la, lb), col in zip(row, [col1, col2, col3]):
        with col:
            if a in price_wide.columns and b in price_wide.columns:
                fig, ax = plt.subplots()
                plot_rolling_corr(ax, price_wide[a], price_wide[b], la, lb, window=rolling_days)
                st.pyplot(fig, use_container_width=True)

# -------------------------------
# LOWER: Volatility Monitor, 3 x 2
# -------------------------------
st.subheader("Cross-Asset Volatility Monitor")
real_vol = realized_vol(rets_all, window=rolling_days)

def plot_series(ax, x, y_list, title, ylabel="Percent"):
    for y, label in y_list:
        ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, ncol=2)

# Row 1
col1, col2, col3 = st.columns(3)

with col1:
    if "^GSPC" in real_vol.columns:
        fig, ax = plt.subplots()
        y = [(real_vol["^GSPC"] * 100, "SPX 1M Realized Vol")]
        if "^VIX" in implied_vol_list:
            vix = fetch_prices(["^VIX"], str(start_date), str(end_date))
            if not vix.empty:
                y.append((vix["^VIX"], "VIX Index"))
        plot_series(ax, real_vol.index, y, "Equity Vol, implied vs realized")
        st.pyplot(fig, use_container_width=True)

with col2:
    if "^TNX" in real_vol.columns:
        fig, ax = plt.subplots()
        plot_series(ax, real_vol.index, [(real_vol["^TNX"] * 100, "US 10Y Realized Vol")], "Rates Vol, realized proxy")
        st.pyplot(fig, use_container_width=True)

with col3:
    if "CL=F" in real_vol.columns:
        fig, ax = plt.subplots()
        y = [(real_vol["CL=F"] * 100, "Oil 1M Realized Vol")]
        if "^OVX" in implied_vol_list:
            ovx = fetch_prices(["^OVX"], str(start_date), str(end_date))
            if not ovx.empty:
                y.append((ovx["^OVX"], "OVX Index"))
        plot_series(ax, real_vol.index, y, "Oil Vol, implied vs realized")
        st.pyplot(fig, use_container_width=True)

# Row 2
col4, col5, col6 = st.columns(3)

with col4:
    if "LQD" in real_vol.columns:
        fig, ax = plt.subplots()
        plot_series(ax, real_vol.index, [(real_vol["LQD"] * 100, "IG 1M Realized Vol")], "IG Credit Vol, realized proxy")
        st.pyplot(fig, use_container_width=True)

with col5:
    if "HYG" in real_vol.columns:
        fig, ax = plt.subplots()
        plot_series(ax, real_vol.index, [(real_vol["HYG"] * 100, "HY 1M Realized Vol")], "HY Credit Vol, realized proxy")
        st.pyplot(fig, use_container_width=True)

with col6:
    to_plot = []
    if "EURUSD=X" in real_vol.columns:
        to_plot.append((real_vol["EURUSD=X"] * 100, "EURUSD Realized Vol"))
    if "USDJPY=X" in real_vol.columns:
        to_plot.append((real_vol["USDJPY=X"] * 100, "USDJPY Realized Vol"))
    if len(to_plot) > 0:
        fig, ax = plt.subplots()
        plot_series(ax, real_vol.index, to_plot, "FX Vol, realized proxies")
        st.pyplot(fig, use_container_width=True)

# -------------------------------
# Bottom: Long Lookback Z-Scores, full width
# -------------------------------
st.subheader("Cross-Asset Volatility Snapshot, Z-Scores (long lookback)")
z_panel = {}
for sym in implied_vol_list:
    iv = fetch_prices([sym], str(start_date), str(end_date))
    if not iv.empty:
        z_panel[sym] = iv.iloc[:, 0]
for sym, label in [("^GSPC", "SPX Realized"),
                   ("^TNX", "US10Y Realized"),
                   ("LQD", "IG Realized"),
                   ("HYG", "HY Realized"),
                   ("CL=F", "Oil Realized"),
                   ("GLD", "Gold Realized"),
                   ("EURUSD=X", "EURUSD Realized"),
                   ("USDJPY=X", "USDJPY Realized")]:
    if sym in real_vol.columns:
        z_panel[label] = real_vol[sym] * 100

if len(z_panel) >= 2:
    zdf = pd.DataFrame(z_panel).dropna(how="all")
    zdf = zdf.apply(lambda s: zscores(s, lookback=zlb))
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in zdf.columns:
        ax.plot(zdf.index, zdf[c], label=c)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_ylabel("Z-Score")
    ax.set_title("Standardized to long history where available")
    ax.legend(ncol=3, fontsize=8)
    st.pyplot(fig, use_container_width=True)
else:
    st.info("Not enough series to compute a z-score snapshot.")

st.caption("Notes: VIX and OVX are available on Yahoo. MOVE and CDX are not, so rates and credit use realized proxies.")
