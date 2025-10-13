import math
import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------------
# App config and baseline style
# -------------------------------
st.set_page_config(page_title="Cross-Asset Correlations and Volatility", layout="wide")

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
st.caption("Data: Yahoo Finance via yfinance. Matrix and panels arranged to mirror your reference. Tickers are fixed to ensure consistency.")

# -------------------------------
# Fixed ticker set and ordering (Yahoo symbols)
# -------------------------------
GROUPS = {
    "Equities": ["^GSPC", "^RUT", "^STOXX50E", "^N225", "EEM"],              # SPX, RTY, SX5E, NKY, MXEF proxy
    "Corporate Credit": ["LQD", "HYG"],                                     # IG, HY ETFs
    "Rates": ["^TNX", "^TYX"],                                              # US 10Y, 30Y yields
    "Commodities": ["CL=F", "GLD", "HG=F"],                                 # WTI, Gold ETF, Copper futures
    "Foreign Exchange": ["EURUSD=X", "USDJPY=X", "GBPUSD=X"],               # EURUSD, USDJPY, GBPUSD
}
ORDER = sum(GROUPS.values(), [])

# -------------------------------
# Sidebar: methodology notes (no inputs)
# -------------------------------
st.sidebar.header("How to read this dashboard")
st.sidebar.markdown(
"""
**Scope**
- Fixed basket and ordering to match your template.
- 1M = rolling 21 trading days.

**Correlation matrix**
- Lower-triangle only, diagonal blank, group headers across columns.
- ETFs stand in for indices when necessary, for example LQD/HYG for IG/HY, EEM for MXEF.

**Volatility**
- Realized vol = rolling 21-day std of daily returns, annualized.
- Implied vol shown where Yahoo provides it: VIX for SPX, OVX for oil.
- MOVE, CDX, and most FX implied vols are not on Yahoo, so those panes use realized proxies.

**Downloads**
- Matrix and prices available as CSV for your records.
"""
)

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
    # Normalize to wide frame of Adj Close (or Close)
    if isinstance(data.columns, pd.MultiIndex):
        frames = []
        for t in tickers:
            if t not in data.columns.get_level_values(0):
                continue
            df = data[t].copy()
            col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else df.columns[0])
            frames.append(df[col].rename(t).to_frame())
        if not frames:
            return pd.DataFrame()
        wide = pd.concat(frames, axis=1)
    else:
        if data.empty:
            return pd.DataFrame()
        col = "Adj Close" if "Adj Close" in data.columns else ("Close" if "Close" in data.columns else data.columns[0])
        wide = data[[col]].rename(columns={col: tickers[0]})
    wide.index.name = "Date"
    return wide.sort_index()

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

def style_corr_matrix(df, groups, highlight_thresh=0.75):
    """Return a Styler that mimics the look of the reference matrix:
       - lower triangle only
       - diagonal blank
       - group headers across columns
       - subtle formatting and conditional highlight for |rho| >= threshold
    """
    # Reindex to fixed ORDER and build MultiIndex columns for group header
    cols = [c for c in ORDER if c in df.columns]
    df = df.loc[cols, cols]
    col_groups = []
    for g, lst in groups.items():
        for t in lst:
            if t in df.columns:
                col_groups.append(g)
    columns = pd.MultiIndex.from_arrays([col_groups, df.columns], names=["Group", "Ticker"])
    df.columns = columns
    df.index.name = "Ticker"

    # Mask upper triangle and diagonal
    base = df.copy()
    base[:] = df.values
    mask_upper = np.triu(np.ones(base.shape, dtype=bool), k=1)  # True for upper triangle
    mask_diag = np.eye(base.shape[0], dtype=bool)

    # Build display frame with lower triangle only
    display = base.copy()
    display.values[mask_upper] = np.nan
    display.values[mask_diag] = np.nan

    # Styling helpers
    def fmt_percent(x):
        return pd.DataFrame(np.where(np.isnan(x), "", np.round(x*100).astype(int).astype(str) + "%"),
                            index=x.index, columns=x.columns)

    def highlight_extremes(x):
        css = np.full(x.shape, "", dtype=object)
        cond = np.abs(x.values) >= highlight_thresh
        # only highlight visible cells
        cond = np.logical_and(cond, ~np.isnan(display.values))
        css[cond] = "background-color: #f8d7da;"  # light red
        # grey diagonal
        css[mask_diag] = "background-color: #e6e6e6;"
        return pd.DataFrame(css, index=x.index, columns=x.columns)

    # Group header colors
    group_colors = {
        "Equities": "#d9ead3",
        "Corporate Credit": "#d0e0e3",
        "Rates": "#fce5cd",
        "Commodities": "#fff2cc",
        "Foreign Exchange": "#ead1dc",
    }
    header_styles = []
    # Color the top header by group
    for i, (g, t) in enumerate(display.columns.tolist()):
        header_styles.append({
            "selector": f"th.col_heading.level0.col{i}",
            "props": [("background-color", group_colors.get(g, "#f2f2f2"))]
        })

    sty = (display.style
           .format(lambda v: "" if pd.isna(v) else f"{int(round(v*100,0))}%")
           .apply(highlight_extremes, axis=None)
           .set_table_styles(header_styles)
           .set_properties(**{"text-align": "center"}))

    # Row band for left group labels
    # Build a left-side narrow column with group names by block
    left_labels = []
    for g, tickers in groups.items():
        for t in tickers:
            if t in display.index:
                left_labels.append(g)
    # We cannot inject a new left column easily into Styler, so we provide a legend below.
    st.caption("Group colors in header: Equities, Corporate Credit, Rates, Commodities, Foreign Exchange.")
    return sty

def align_series_to_x(x_index, series):
    """Align a pandas Series to x_index for plotting, returning (x, y) aligned.
       Fixes the ValueError: x and y must have same first dimension.
    """
    if isinstance(series, pd.Series):
        y = series.reindex(pd.Index(x_index)).astype(float)
        return x_index, y.values
    # assume numpy-like
    return x_index, np.array(series)

# -------------------------------
# Data download
# -------------------------------
end_date = datetime.now().date()
start_date = (datetime.now() - timedelta(days=365*12)).date()
with st.spinner("Downloading market data from Yahoo Finance..."):
    prices = fetch_prices(ORDER, str(start_date), str(end_date), interval="1d")

if prices.empty:
    st.error("No data downloaded. Check connectivity.")
    st.stop()

rets = pct_returns(prices)
window = 21  # 1M ~ 21 trading days
corr_1m = rets.tail(window).corr().replace([-np.inf, np.inf], np.nan)

# -------------------------------
# TOP: Big correlation matrix
# -------------------------------
st.subheader("Cross-Asset Correlation Matrix (1M)")
st.dataframe(
    style_corr_matrix(corr_1m, GROUPS),
    use_container_width=True
)
c1, c2 = st.columns([1, 1])
with c1:
    st.download_button("Download matrix (CSV)", data=corr_1m.to_csv().encode(), file_name="corr_1m.csv")
with c2:
    st.download_button("Download prices (CSV)", data=prices.to_csv().encode(), file_name="prices.csv")

# -------------------------------
# MIDDLE: Correlation analysis, 3 x 2
# -------------------------------
st.subheader("Cross-Asset Correlation Analysis, Rolling 1M")

def plot_rolling_corr(ax, a_sym, b_sym, la, lb, window=21):
    if a_sym not in prices.columns or b_sym not in prices.columns:
        ax.set_visible(False)
        return
    df = pd.DataFrame({la: prices[a_sym], lb: prices[b_sym]}).dropna()
    rcorr = df[la].pct_change().rolling(window).corr(df[lb].pct_change())
    ax.plot(rcorr.index, rcorr.values)
    ax.set_title(f"{la} vs {lb}")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1, 1)

pairs = [
    ("^GSPC", "^TNX", "SPX", "US 10Y Yield"),
    ("^RUT", "LQD", "RTY", "IG Credit"),
    ("^GSPC", "CL=F", "SPX", "WTI Crude"),
    ("^GSPC", "GLD", "SPX", "Gold"),
    ("EEM", "EURUSD=X", "EM Eq (EEM)", "EURUSD"),
    ("^N225", "USDJPY=X", "Nikkei 225", "USDJPY"),
]

rows = [pairs[:3], pairs[3:]]
for row in rows:
    col1, col2, col3 = st.columns(3)
    for (a, b, la, lb), col in zip(row, [col1, col2, col3]):
        with col:
            fig, ax = plt.subplots()
            plot_rolling_corr(ax, a, b, la, lb, window=window)
            st.pyplot(fig, use_container_width=True)

# -------------------------------
# LOWER: Volatility monitor, 3 x 2
# -------------------------------
st.subheader("Cross-Asset Volatility Monitor")
real_vol = realized_vol(rets, window=window)

def plot_series(ax, x_index, series_list, title, ylabel="Percent"):
    for s, label in series_list:
        x_aligned, y_aligned = align_series_to_x(x_index, s)
        ax.plot(x_aligned, y_aligned, label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, ncol=2)

# Row 1
c1, c2, c3 = st.columns(3)
with c1:
    fig, ax = plt.subplots()
    ys = [(real_vol["^GSPC"] * 100, "SPX 1M Realized Vol")]
    vix = fetch_prices(["^VIX"], str(start_date), str(end_date))
    if not vix.empty:
        ys.append((vix["^VIX"], "VIX Index"))
    plot_series(ax, real_vol.index, ys, "Equity Vol, implied vs realized")
    st.pyplot(fig, use_container_width=True)

with c2:
    fig, ax = plt.subplots()
    plot_series(ax, real_vol.index, [(real_vol["^TNX"] * 100, "US 10Y Realized Vol")], "Rates Vol, realized proxy")
    st.pyplot(fig, use_container_width=True)

with c3:
    fig, ax = plt.subplots()
    ys = [(real_vol["CL=F"] * 100, "Oil 1M Realized Vol")]
    ovx = fetch_prices(["^OVX"], str(start_date), str(end_date))
    if not ovx.empty:
        ys.append((ovx["^OVX"], "OVX Index"))
    plot_series(ax, real_vol.index, ys, "Oil Vol, implied vs realized")
    st.pyplot(fig, use_container_width=True)

# Row 2
c4, c5, c6 = st.columns(3)
with c4:
    fig, ax = plt.subplots()
    plot_series(ax, real_vol.index, [(real_vol["LQD"] * 100, "IG 1M Realized Vol")], "IG Credit Vol, realized proxy")
    st.pyplot(fig, use_container_width=True)

with c5:
    fig, ax = plt.subplots()
    plot_series(ax, real_vol.index, [(real_vol["HYG"] * 100, "HY 1M Realized Vol")], "HY Credit Vol, realized proxy")
    st.pyplot(fig, use_container_width=True)

with c6:
    fig, ax = plt.subplots()
    to_plot = []
    if "EURUSD=X" in real_vol.columns:
        to_plot.append((real_vol["EURUSD=X"] * 100, "EURUSD Realized Vol"))
    if "USDJPY=X" in real_vol.columns:
        to_plot.append((real_vol["USDJPY=X"] * 100, "USDJPY Realized Vol"))
    if "GBPUSD=X" in real_vol.columns:
        to_plot.append((real_vol["GBPUSD=X"] * 100, "GBPUSD Realized Vol"))
    plot_series(ax, real_vol.index, to_plot, "FX Vol, realized proxies")
    st.pyplot(fig, use_container_width=True)

# -------------------------------
# Bottom: Z-score snapshot, full width
# -------------------------------
st.subheader("Cross-Asset Volatility Snapshot, Z-Scores (10Y style lookback)")
z_panel = {}
# Implied if available
for sym in ["^VIX", "^OVX"]:
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
                   ("USDJPY=X", "USDJPY Realized"),
                   ("GBPUSD=X", "GBPUSD Realized")]:
    if sym in real_vol.columns:
        z_panel[label] = real_vol[sym] * 100

if len(z_panel) >= 2:
    zdf = pd.DataFrame(z_panel).dropna(how="all")
    zdf = zdf.apply(lambda s: zscores(s, lookback=2520))
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
