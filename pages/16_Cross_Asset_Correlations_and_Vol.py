# Cross-Asset Correlation and Volatility Dashboard (optimized)
# Notes:
# - Single batched Yahoo fetch with cache
# - Rolling correlations on returns, not double differencing
# - Safer styling and guards for partial data
# - Optional clipping, focus since 2017, regime shading
# - Data freshness banner

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import streamlit as st

# =========================
# App config and style
# =========================
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
st.caption("Data: Yahoo Finance via yfinance. Universe is locked. Layout mirrors your reference.")

# =========================
# Fixed universe and order (Yahoo symbols)
# =========================
GROUPS = {
    "Equities": ["^GSPC", "^RUT", "^STOXX50E", "^N225", "EEM"],      # SPX, RTY, SX5E, NKY, MXEF proxy (EEM)
    "Corporate Credit": ["LQD", "HYG"],                              # IG, HY ETFs
    "Rates": ["^TNX", "^TYX"],                                       # US 10Y, 30Y yields
    "Commodities": ["CL=F", "GLD", "HG=F"],                          # WTI, Gold ETF, Copper futures
    "Foreign Exchange": ["EURUSD=X", "USDJPY=X", "GBPUSD=X"],        # EURUSD, USDJPY, GBPUSD
}
ORDER = sum(GROUPS.values(), [])

# IV proxies we will fetch once and reuse
IV_TICKERS = ["^VIX", "^OVX"]

# =========================
# Sidebar: reading guide + view controls
# =========================
st.sidebar.header("Reading guide")
st.sidebar.markdown(
"""
- **1M** = 21 trading days.
- **Matrix**: lower triangle, gray diagonal, group-colored header, highlight |rho| ≥ 0.75.
- **Vol**: realized = 21d stdev, annualized. Implied where Yahoo offers it (VIX, OVX).
- **MOVE/CDX/FX IV** not on Yahoo, so realized proxies stand in.

**View controls**
- Focus since **2017** to avoid 2020 skew.
- Clip y-axes by percentile to keep visuals readable.
- Regime shading for COVID and the 2022–23 hike cycle.
"""
)

focus_since_2017 = st.sidebar.checkbox("Focus on 2017–present", value=True)
clip_extremes = st.sidebar.checkbox("Clip y-axes at 1st–99th percentiles", value=True)
show_regimes = st.sidebar.checkbox("Show regime shading", value=True)

# =========================
# Helpers
# =========================
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_prices(tickers, start, end, interval="1d"):
    # Robust Yahoo fetch that returns a wide DataFrame of Adj Close where available
    try:
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
    except Exception as e:
        st.error(f"Yahoo Finance error: {e}")
        return pd.DataFrame()

    if data is None or len(data) == 0:
        return pd.DataFrame()

    # Normalize to wide frame of Adj Close (or Close)
    if isinstance(data.columns, pd.MultiIndex):
        frames = []
        for t in tickers:
            try:
                if t not in data.columns.get_level_values(0):
                    continue
                df = data[t].copy()
                col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else df.columns[0])
                frames.append(df[col].rename(t).to_frame())
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        wide = pd.concat(frames, axis=1)
    else:
        # Single ticker case
        col = "Adj Close" if "Adj Close" in data.columns else ("Close" if "Close" in data.columns else data.columns[0])
        wide = data[[col]]
        if len(tickers) == 1:
            wide = wide.rename(columns={col: tickers[0]})
    wide.index.name = "Date"
    return wide.sort_index()

def pct_returns(prices):
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")

def realized_vol(returns, window=21, annualization=252):
    return returns.rolling(window).std() * math.sqrt(annualization)

def zscores_full(series_like):
    s = pd.Series(series_like).dropna()
    if s.empty:
        return s
    m, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.nan, index=s.index)
    return (s - m) / sd

def percentile_rank(series):
    s = pd.Series(series).dropna()
    if s.empty:
        return np.nan
    return float(s.rank(pct=True).iloc[-1] * 100.0)

def apply_focus_window(index):
    if not focus_since_2017:
        return index
    start = pd.Timestamp("2017-01-01")
    return index[index >= start]

def clip_limits(array_like, pmin=1, pmax=99):
    if not clip_extremes:
        return None
    a = pd.Series(np.asarray(array_like).astype(float)).dropna()
    if a.empty:
        return None
    lo, hi = np.percentile(a, [pmin, pmax])
    span = hi - lo
    return (lo - 0.05*span, hi + 0.05*span)

def add_regime_shading(ax):
    if not show_regimes:
        return
    spans = [
        ("2020-02-15", "2020-06-30"),   # COVID shock
        ("2022-03-01", "2023-11-01"),   # Fed hiking cycle
    ]
    for a, b in spans:
        ax.axvspan(pd.to_datetime(a), pd.to_datetime(b), color="#cccccc", alpha=0.25)

def style_corr_matrix(corr_df, groups, highlight_thresh=0.75):
    # Guard for empty or degenerate input
    if corr_df is None or corr_df.empty:
        st.error("Correlation matrix is empty.")
        return None

    # Reindex to fixed order present in corr_df
    cols = [c for c in ORDER if c in corr_df.columns]
    if not cols:
        st.error("No valid tickers found for correlation matrix.")
        return None

    base = corr_df.loc[cols, cols].copy()

    # Build MultiIndex columns for group headers
    col_groups = []
    for t in base.columns:
        g_found = None
        for g, lst in groups.items():
            if t in lst:
                g_found = g
                break
        col_groups.append(g_found or "Other")
    base.columns = pd.MultiIndex.from_arrays([col_groups, base.columns], names=["Group", "Ticker"])
    base.index.name = "Ticker"

    # Keep lower triangle only, blank diagonal
    n = base.shape[0]
    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    mask_diag = np.eye(n, dtype=bool)
    display = base.copy()
    vals = display.values.astype(float)
    vals[mask_upper] = np.nan
    vals[mask_diag] = np.nan
    display.iloc[:, :] = vals

    # Header colors by group
    group_colors = {
        "Equities": "#d9ead3",
        "Corporate Credit": "#d0e0e3",
        "Rates": "#fce5cd",
        "Commodities": "#fff2cc",
        "Foreign Exchange": "#ead1dc",
        "Other": "#f2f2f2",
    }

    header_styles = []
    for i, (g, _) in enumerate(display.columns.tolist()):
        header_styles.append({
            "selector": f"th.col_heading.level0.col{i}",
            "props": [("background-color", group_colors.get(g, "#f2f2f2"))]
        })

    def highlight_extremes(x):
        css = np.full(x.shape, "", dtype=object)
        cond = np.abs(x.values) >= highlight_thresh
        # do not highlight NaNs
        cond = np.logical_and(cond, ~np.isnan(x.values))
        css[cond] = "background-color: #f8d7da;"  # light red
        # gray diagonal
        for i in range(min(x.shape)):
            css[i, i] = "background-color: #e6e6e6;"
        return pd.DataFrame(css, index=x.index, columns=x.columns)

    sty = (display.style
           .format(lambda v: "" if pd.isna(v) else f"{int(round(v*100, 0))}%")
           .apply(highlight_extremes, axis=None)
           .set_table_styles(header_styles, overwrite=False)
           .set_properties(**{"text-align": "center"}))
    st.caption("Header colors: Equities, Corporate Credit, Rates, Commodities, Foreign Exchange.")
    return sty

# =========================
# Data download (single batched call)
# =========================
end_date = datetime.now().date()
start_date = (datetime.now() - timedelta(days=365*12)).date()

ALL_TICKERS = list(dict.fromkeys(ORDER + IV_TICKERS))  # de-duplicate, preserve order

with st.spinner("Downloading market data from Yahoo Finance..."):
    PRICES = fetch_prices(ALL_TICKERS, str(start_date), str(end_date), interval="1d")

if PRICES.empty:
    st.error("No data downloaded. Check connectivity.")
    st.stop()

# Split slices for convenience
BASE = PRICES[[c for c in ORDER if c in PRICES.columns]].copy()
IV   = PRICES[[c for c in IV_TICKERS if c in PRICES.columns]].copy()

# Freshness banner
last_date = BASE.index.max()
st.info(f"Data last date: {last_date.strftime('%Y-%m-%d')}" if pd.notna(last_date) else "No dates available.")

RET = pct_returns(BASE)
WIN = 21  # 1M ~ 21 trading days
REALVOL = realized_vol(RET, window=WIN)

# =========================
# Top: big correlation matrix
# =========================
st.subheader("Cross-Asset Correlation Matrix (1M)")
# Build corr from returns over last WIN days
if len(RET) >= WIN:
    recent_ret = RET.tail(WIN)
    corr_1m = recent_ret.corr().replace([-np.inf, np.inf], np.nan)
else:
    corr_1m = pd.DataFrame()

if corr_1m.empty:
    st.warning("Not enough data to compute the 1M correlation matrix.")
else:
    rows = corr_1m.shape[0]
    row_px = 40
    header_px = 110
    table_height = min(header_px + row_px * rows, 2400)

    sty = style_corr_matrix(corr_1m, GROUPS)
    if sty is not None:
        st.dataframe(sty, use_container_width=True, height=table_height)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.download_button("Download matrix (CSV)", data=corr_1m.to_csv().encode(), file_name="corr_1m.csv")
        with c2:
            st.download_button("Download prices (CSV)", data=BASE.to_csv().encode(), file_name="prices.csv")

# =========================
# Middle: 3×2 correlation panels
# =========================
st.subheader("Cross-Asset Correlation Analysis")

PANEL_SPECS = [
    dict(a="^GSPC", b="^TNX", la="SPX", lb="US 10Y Tsy Yield", title="Equity-Rates Correlation (1M)"),
    dict(a="^RUT", b="LQD", la="RTY", lb="IG Bonds (LQD)",     title="Equity-Corp Bond Correlation (1M)"),
    dict(a="^GSPC", b="CL=F", la="SPX", lb="Oil (WTI)",        title="Equity-Oil Correlation (1M)"),
    dict(a="^GSPC", b="GLD",  la="SPX", lb="GLD",              title="Equity-Gold Correlation (1M)"),
    dict(a="^STOXX50E", b="EURUSD=X", la="SX5E", lb="EURUSD",  title="Equity-FX Correlation (1M)"),
    dict(a="^N225", b="USDJPY=X", la="NKY", lb="USDJPY",       title="Equity-FX Correlation (1M)"),
]

def roll_corr_from_returns(ret_df, a, b, window=21):
    if a not in ret_df.columns or b not in ret_df.columns:
        return pd.Series(dtype=float)
    df = ret_df[[a, b]].dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return df[a].rolling(window).corr(df[b])

def corr_panel(ax, spec):
    rc = roll_corr_from_returns(RET, spec["a"], spec["b"], WIN)
    if rc.empty:
        ax.set_visible(False)
        return
    rc = rc.loc[apply_focus_window(rc.index)]
    ax.plot(rc.index, rc.values)
    ax.axhline(0, linewidth=1)
    lims = clip_limits(rc.values, 1, 99)
    if lims:
        ax.set_ylim(max(-1, lims[0]), min(1, lims[1]))
    else:
        ax.set_ylim(-1, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title(spec["title"])
    ax.set_ylabel("Correlation (%)")
    ax.legend([f"{spec['la']} vs. {spec['lb']}"], loc="upper left", fontsize=9)
    add_regime_shading(ax)
    ymin, ymax = ax.get_ylim()
    ax.text(rc.index.min(), ymin + 0.02*(ymax - ymin), "Source: Yahoo Finance", fontsize=8)

for row_specs in [PANEL_SPECS[:3], PANEL_SPECS[3:]]:
    col1, col2, col3 = st.columns(3)
    for spec, col in zip(row_specs, [col1, col2, col3]):
        with col:
            fig, ax = plt.subplots()
            corr_panel(ax, spec)
            st.pyplot(fig, use_container_width=True)

# =========================
# Lower: 3×2 volatility monitor
# =========================
st.subheader("Cross-Asset Volatility Monitor")

def align_plot(ax, s, label, title):
    if s is None or s.empty:
        ax.set_visible(False)
        return
    s = s.loc[apply_focus_window(s.index)]
    if s.empty:
        ax.set_visible(False)
        return
    x = s.index
    y = s.astype(float).values
    ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.set_ylabel("Percent")
    add_regime_shading(ax)

# Row 1
c1, c2, c3 = st.columns(3)

with c1:
    fig, ax = plt.subplots()
    if "^GSPC" in REALVOL.columns:
        align_plot(ax, REALVOL["^GSPC"] * 100, "SPX 1M Realized Vol", "Equity Vol, implied vs realized")
        if "^VIX" in IV.columns:
            v = IV["^VIX"].loc[apply_focus_window(IV.index)]
            ax.plot(v.index, v.values, label="VIX Index")
        lims = clip_limits((REALVOL["^GSPC"]*100).values, 1, 99)
        if lims: ax.set_ylim(lims)
        ax.legend(fontsize=8, ncol=2)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("SPX series unavailable for realized vol.")

with c2:
    fig, ax = plt.subplots()
    if "^TNX" in REALVOL.columns:
        align_plot(ax, REALVOL["^TNX"] * 100, "US 10Y Realized Vol", "Rates Vol, realized proxy")
        lims = clip_limits((REALVOL["^TNX"]*100).values, 1, 99)
        if lims: ax.set_ylim(lims)
        ax.legend(fontsize=8)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("US 10Y series unavailable for realized vol.")

with c3:
    fig, ax = plt.subplots()
    if "CL=F" in REALVOL.columns:
        align_plot(ax, REALVOL["CL=F"] * 100, "Oil 1M Realized Vol", "Oil Vol, implied vs realized")
        if "^OVX" in IV.columns:
            o = IV["^OVX"].loc[apply_focus_window(IV.index)]
            ax.plot(o.index, o.values, label="OVX Index")
        lims = clip_limits((REALVOL["CL=F"]*100).values, 1, 99)
        if lims: ax.set_ylim(lims)
        ax.legend(fontsize=8, ncol=2)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("CL=F series unavailable for realized vol.")

# Row 2
c4, c5, c6 = st.columns(3)

with c4:
    fig, ax = plt.subplots()
    if "LQD" in REALVOL.columns:
        align_plot(ax, REALVOL["LQD"] * 100, "IG 1M Realized Vol", "IG Credit Vol, realized proxy")
        lims = clip_limits((REALVOL["LQD"]*100).values, 1, 99)
        if lims: ax.set_ylim(lims)
        ax.legend(fontsize=8)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("LQD series unavailable for realized vol.")

with c5:
    fig, ax = plt.subplots()
    if "HYG" in REALVOL.columns:
        align_plot(ax, REALVOL["HYG"] * 100, "HY 1M Realized Vol", "HY Credit Vol, realized proxy")
        lims = clip_limits((REALVOL["HYG"]*100).values, 1, 99)
        if lims: ax.set_ylim(lims)
        ax.legend(fontsize=8)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("HYG series unavailable for realized vol.")

with c6:
    fig, ax = plt.subplots()
    fx_cols = [c for c in REALVOL.columns if c.endswith("=X")]
    plotted = False
    for fx in ["EURUSD=X", "USDJPY=X", "GBPUSD=X"]:
        if fx in REALVOL.columns:
            s = (REALVOL[fx] * 100).loc[apply_focus_window(REALVOL.index)]
            ax.plot(s.index, s.values, label=f"{fx} Realized Vol")
            plotted = True
    if plotted:
        all_fx = pd.concat([(REALVOL[c]*100).dropna() for c in fx_cols], axis=0) if fx_cols else pd.Series(dtype=float)
        lims = clip_limits(all_fx.values if len(all_fx) else [], 1, 99)
        if lims: ax.set_ylim(lims)
        ax.set_title("FX Vol, realized proxies")
        ax.set_ylabel("Percent")
        ax.legend(fontsize=8)
        add_regime_shading(ax)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("FX realized vol series unavailable.")

# =========================
# Bottom: standardized vol snapshot + ranked table
# =========================
st.subheader("Volatility Snapshot — Standardized and Ranked")

z_panel = {}

# Implied vol indices
for sym in ["^VIX", "^OVX"]:
    if sym in IV.columns:
        z_panel[sym] = IV[sym].copy()

# Realized vol to percent units
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
    # Full-sample standardization per series
    zdf = zdf.apply(zscores_full, axis=0)
    zdf = zdf.loc[apply_focus_window(zdf.index)]

    if not zdf.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        for c in zdf.columns:
            ax.plot(zdf.index, zdf[c], label=c)
        ax.axhline(0, linestyle="--", linewidth=1)
        add_regime_shading(ax)
        lims = clip_limits(zdf.values.flatten(), 1, 99)
        if lims:
            ax.set_ylim(lims)
        ax.set_ylabel("Z-Score")
        ax.set_title("Standardized to full available history")
        ax.legend(ncol=3, fontsize=8)
        st.pyplot(fig, use_container_width=True)

        latest = zdf.tail(1).T.rename(columns={zdf.index[-1]: "Current Z"})
        latest["Percentile"] = pd.Series({c: percentile_rank(zdf[c]) for c in zdf.columns})
        latest = latest.sort_values("Current Z", ascending=False)
        st.markdown("**Current standardized vol (ranked)**")
        st.dataframe(
            latest.style.format({"Current Z": "{:.2f}", "Percentile": "{:.0f}%"}).background_gradient(
                subset=["Current Z"], cmap="RdYlGn_r"
            ),
            use_container_width=True
        )
    else:
        st.info("Not enough overlapping history to plot standardized snapshot.")
else:
    st.info("Not enough series to compute standardized snapshot.")
