import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import streamlit as st

# -------------------------------
# App config and style
# -------------------------------
st.set_page_config(page_title="Cross-Asset Correlations and Volatility", layout="wide")

plt.rcParams.update({
    "figure.figsize": (8, 3),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 14,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 2.0,
})

st.title("Cross-Asset Correlation and Volatility Dashboard")
st.caption("Data: Yahoo Finance via yfinance. Layout mirrors your reference. Universe is locked.")

# -------------------------------
# Fixed universe and order (Yahoo symbols)
# -------------------------------
GROUPS = {
    "Equities": ["^GSPC", "^RUT", "^STOXX50E", "^N225", "EEM"],      # SPX, RTY, SX5E, NKY, MXEF proxy (EEM)
    "Corporate Credit": ["LQD", "HYG"],                              # IG, HY ETFs
    "Rates": ["^TNX", "^TYX"],                                       # US 10Y, 30Y yields
    "Commodities": ["CL=F", "GLD", "HG=F"],                          # WTI, Gold ETF, Copper futures
    "Foreign Exchange": ["EURUSD=X", "USDJPY=X", "GBPUSD=X"],        # EURUSD, USDJPY, GBPUSD
}
ORDER = sum(GROUPS.values(), [])

# -------------------------------
# Sidebar: methodology notes
# -------------------------------
st.sidebar.header("How to read this dashboard")
st.sidebar.markdown(
"""
**Scope**
- Fixed basket and ordering to match your template.
- 1M window uses 21 trading days.

**Correlation matrix**
- Lower-triangle only, gray diagonal, group-colored header, highlight when absolute rho ≥ 0.75.
- ETFs proxy indices where needed, for example LQD/HYG for IG/HY, EEM for MXEF.

**Volatility**
- Realized vol = 21-day rolling standard deviation of daily returns, annualized.
- Implied vol shown where Yahoo provides it, VIX for SPX, OVX for oil.
- MOVE, CDX, and most FX implied vols are unavailable on Yahoo, so realized proxies are used.

**Downloads**
- CSV buttons for matrix and prices.
"""
)

# -------------------------------
# Helpers
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
    # Reindex to fixed order and make MultiIndex columns for group header
    cols = [c for c in ORDER if c in df.columns]
    df = df.loc[cols, cols]
    col_groups = []
    for g, lst in groups.items():
        for t in lst:
            if t in df.columns:
                col_groups.append(g)
    df.columns = pd.MultiIndex.from_arrays([col_groups, df.columns], names=["Group", "Ticker"])
    df.index.name = "Ticker"

    # Keep lower triangle only, blank diagonal
    base = df.copy()
    mask_upper = np.triu(np.ones(base.shape, dtype=bool), k=1)
    mask_diag = np.eye(base.shape[0], dtype=bool)
    display = base.copy()
    display.values[mask_upper] = np.nan
    display.values[mask_diag] = np.nan

    # Header colors by group
    group_colors = {
        "Equities": "#d9ead3",
        "Corporate Credit": "#d0e0e3",
        "Rates": "#fce5cd",
        "Commodities": "#fff2cc",
        "Foreign Exchange": "#ead1dc",
    }
    header_styles = []
    for i, (g, t) in enumerate(display.columns.tolist()):
        header_styles.append({
            "selector": f"th.col_heading.level0.col{i}",
            "props": [("background-color", group_colors.get(g, "#f2f2f2"))]
        })

    def highlight_extremes(x):
        css = np.full(x.shape, "", dtype=object)
        cond = np.abs(x.values) >= highlight_thresh
        cond = np.logical_and(cond, ~np.isnan(display.values))
        css[cond] = "background-color: #f8d7da;"  # light red
        css[mask_diag] = "background-color: #e6e6e6;"
        return pd.DataFrame(css, index=x.index, columns=x.columns)

    sty = (display.style
           .format(lambda v: "" if pd.isna(v) else f"{int(round(v*100,0))}%")
           .apply(highlight_extremes, axis=None)
           .set_table_styles(header_styles)
           .set_properties(**{"text-align": "center"}))
    st.caption("Header colors: Equities, Corporate Credit, Rates, Commodities, Foreign Exchange.")
    return sty

def align_series_to_x(x_index, series):
    """Return y aligned to x_index as a numpy array."""
    if isinstance(series, pd.Series):
        return x_index, series.reindex(pd.Index(x_index)).astype(float).values
    return x_index, np.array(series)

# -------------------------------
# Data download
# -------------------------------
end_date = datetime.now().date()
start_date = (datetime.now() - timedelta(days=365*12)).date()

with st.spinner("Downloading market data from Yahoo Finance..."):
    PRICES = fetch_prices(ORDER, str(start_date), str(end_date), interval="1d")

if PRICES.empty:
    st.error("No data downloaded. Check connectivity.")
    st.stop()

RET = pct_returns(PRICES)
WIN = 21  # 1M ~ 21 trading days

# -------------------------------
# Top: big correlation matrix, no vertical scroll
# -------------------------------
st.subheader("Cross-Asset Correlation Matrix (1M)")
corr_1m = RET.tail(WIN).corr().replace([-np.inf, np.inf], np.nan)

# Auto-fit height to rows so the whole matrix is visible at first glance
rows = corr_1m.shape[0]            # number of tickers
row_px = 40                         # row height estimate
header_px = 110                     # header height estimate
table_height = header_px + row_px * rows
table_height = min(table_height, 2000)  # safety cap for small screens

sty = style_corr_matrix(corr_1m, GROUPS)
st.dataframe(sty, use_container_width=True, height=table_height)

c1, c2 = st.columns([1, 1])
with c1:
    st.download_button("Download matrix (CSV)", data=corr_1m.to_csv().encode(), file_name="corr_1m.csv")
with c2:
    st.download_button("Download prices (CSV)", data=PRICES.to_csv().encode(), file_name="prices.csv")

# -------------------------------
# Middle: 3×2 correlation panels
# -------------------------------
st.subheader("Cross-Asset Correlation Analysis")

PANEL_SPECS = [
    dict(a="^GSPC", b="^TNX", la="SPX", lb="US 10Y Tsy Yield",
         title="Equity-Rates Correlation (1M)", ymin=-0.8, ymax=0.8),
    dict(a="^RUT", b="LQD", la="RTY", lb="IG Bonds (LQD)",
         title="Equity-Corp Bond Correlation (1M)", ymin=-0.6, ymax=1.2),
    dict(a="^GSPC", b="CL=F", la="SPX", lb="Oil (WTI)",
         title="Equity-Oil Correlation (1M)", ymin=-0.8, ymax=0.8),
    dict(a="^GSPC", b="GLD", la="SPX", lb="GLD",
         title="Equity-Gold Correlation (1M)", ymin=-1.0, ymax=0.8),
    dict(a="^STOXX50E", b="EURUSD=X", la="SX5E", lb="EURUSD",
         title="Equity-FX Correlation (1M)", ymin=-0.6, ymax=0.6),
    dict(a="^N225", b="USDJPY=X", la="NKY", lb="USDJPY",
         title="Equity-FX Correlation (1M)", ymin=-0.8, ymax=0.8),
]

def roll_corr(series_a, series_b, window=21):
    df = pd.DataFrame({"A": series_a, "B": series_b}).dropna()
    return df["A"].pct_change().rolling(window).corr(df["B"].pct_change())

def panel(ax, spec):
    if spec["a"] not in PRICES.columns or spec["b"] not in PRICES.columns:
        ax.set_visible(False)
        return
    rc = roll_corr(PRICES[spec["a"]], PRICES[spec["b"]], WIN)
    ax.plot(rc.index, rc.values)
    ax.axhline(0, linewidth=1)
    ax.set_ylim(spec["ymin"], spec["ymax"])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title(spec["title"])
    ax.set_ylabel("Correlation (%)")
    ax.legend([f"{spec['la']} vs. {spec['lb']}"], loc="upper left", fontsize=9)
    ymin, ymax = ax.get_ylim()
    ax.text(rc.index.min(), ymin + 0.02*(ymax - ymin), "Source: Yahoo Finance", fontsize=8)

rows_specs = [PANEL_SPECS[:3], PANEL_SPECS[3:]]
for row in rows_specs:
    col1, col2, col3 = st.columns(3)
    for spec, col in zip(row, [col1, col2, col3]):
        with col:
            fig, ax = plt.subplots()
            panel(ax, spec)
            st.pyplot(fig, use_container_width=True)

# -------------------------------
# Lower: 3×2 volatility monitor
# -------------------------------
st.subheader("Cross-Asset Volatility Monitor")
REALVOL = realized_vol(RET, window=WIN)

def align_plot(ax, x_index, series_list, title, ylabel="Percent"):
    for s, label in series_list:
        x_aligned, y_aligned = x_index, s.reindex(x_index).astype(float).values
        ax.plot(x_aligned, y_aligned, label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, ncol=2)

# Row 1
c1, c2, c3 = st.columns(3)
with c1:
    fig, ax = plt.subplots()
    ys = [(REALVOL["^GSPC"] * 100, "SPX 1M Realized Vol")]
    vix = fetch_prices(["^VIX"], str(start_date), str(end_date))
    if not vix.empty:
        ys.append((vix["^VIX"], "VIX Index"))
    align_plot(ax, REALVOL.index, ys, "Equity Vol, implied vs realized")
    st.pyplot(fig, use_container_width=True)

with c2:
    fig, ax = plt.subplots()
    align_plot(ax, REALVOL.index, [(REALVOL["^TNX"] * 100, "US 10Y Realized Vol")], "Rates Vol, realized proxy")
    st.pyplot(fig, use_container_width=True)

with c3:
    fig, ax = plt.subplots()
    ys = [(REALVOL["CL=F"] * 100, "Oil 1M Realized Vol")]
    ovx = fetch_prices(["^OVX"], str(start_date), str(end_date))
    if not ovx.empty:
        ys.append((ovx["^OVX"], "OVX Index"))
    align_plot(ax, REALVOL.index, ys, "Oil Vol, implied vs realized")
    st.pyplot(fig, use_container_width=True)

# Row 2
c4, c5, c6 = st.columns(3)
with c4:
    fig, ax = plt.subplots()
    align_plot(ax, REALVOL.index, [(REALVOL["LQD"] * 100, "IG 1M Realized Vol")], "IG Credit Vol, realized proxy")
    st.pyplot(fig, use_container_width=True)

with c5:
    fig, ax = plt.subplots()
    align_plot(ax, REALVOL.index, [(REALVOL["HYG"] * 100, "HY 1M Realized Vol")], "HY Credit Vol, realized proxy")
    st.pyplot(fig, use_container_width=True)

with c6:
    fig, ax = plt.subplots()
    fx_list = []
    if "EURUSD=X" in REALVOL.columns:
        fx_list.append((REALVOL["EURUSD=X"] * 100, "EURUSD Realized Vol"))
    if "USDJPY=X" in REALVOL.columns:
        fx_list.append((REALVOL["USDJPY=X"] * 100, "USDJPY Realized Vol"))
    if "GBPUSD=X" in REALVOL.columns:
        fx_list.append((REALVOL["GBPUSD=X"] * 100, "GBPUSD Realized Vol"))
    align_plot(ax, REALVOL.index, fx_list, "FX Vol, realized proxies")
    st.pyplot(fig, use_container_width=True)

# -------------------------------
# Bottom: z-score snapshot
# -------------------------------
st.subheader("Cross-Asset Volatility Snapshot, Z-Scores")
z_panel = {}

for sym in ["^VIX", "^OVX"]:
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
                   ("USDJPY=X", "USDJPY Realized"),
                   ("GBPUSD=X", "GBPUSD Realized")]:
    if sym in REALVOL.columns:
        z_panel[label] = REALVOL[sym] * 100

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
