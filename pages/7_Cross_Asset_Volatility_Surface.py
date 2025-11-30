import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta

# ---------------- Config ----------------
st.set_page_config(page_title="Cross Asset Volatility Surface Monitor", layout="wide")
plt.style.use("default")

PASTEL = [
    "#A8DADC", "#F4A261", "#90BE6D", "#FFD6A5", "#BDE0FE",
    "#CDB4DB", "#E2F0CB", "#F1C0E8", "#B9FBC0", "#F7EDE2"
]

GRID = "#e8e8e8"
TEXT = "#222222"


# ---------------- Helpers ----------------
def zscore(x: pd.Series) -> pd.Series:
    x = pd.Series(x).dropna()
    if x.empty:
        return x * np.nan
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return x * np.nan
    return (x - x.mean()) / sd


def realized_vol(series: pd.Series, window: int = 21, ann: int = 252) -> pd.Series:
    r = series.pct_change().dropna()
    if r.empty:
        return pd.Series(dtype=float, index=series.index)
    return r.rolling(window).std(ddof=0) * np.sqrt(ann)


@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_series(tickers, start="2010-01-01"):
    """
    Robust loader for multiple tickers using yfinance.

    Returns a dict: {ticker: Close series} with all series aligned on their
    own trading calendars.
    """
    if not tickers:
        return {}

    try:
        data = yf.download(
            tickers=tickers,
            start=start,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
        )
    except Exception as e:
        st.error(f"yfinance error: {e}")
        return {}

    out = {}

    # MultiIndex: level 0 = ticker, level 1 = field
    if isinstance(data.columns, pd.MultiIndex):
        tick_level = data.columns.get_level_values(0)
        for t in tickers:
            if t in tick_level:
                df_t = data[t]
                col = "Adj Close" if "Adj Close" in df_t.columns else "Close"
                if col in df_t.columns:
                    s = df_t[col].dropna()
                    if not s.empty:
                        out[t] = s
    else:
        # Single ticker case
        cols = data.columns.tolist()
        col = None
        for c in ["Adj Close", "Close"]:
            if c in cols:
                col = c
                break
        if col is not None:
            s = data[col].dropna()
            if not s.empty:
                # When only one ticker is passed, yfinance uses that name for the series
                t_name = tickers[0]
                out[t_name] = s

    return out


def pastel_cmap():
    return LinearSegmentedColormap.from_list(
        "pastel_scale",
        ["#F1FAEE", "#A8DADC", "#457B9D"],
        N=256
    )


# ---------------- Sidebar ----------------
st.title("Cross Asset Volatility Surface Monitor")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
**Objective**

Monitor implied and realized volatility across major risk assets in a single view.

**Design**

- Implied vol indices for equities, Treasuries, gold, and oil  
- Realized 1M vol for SPX, TLT, GLD, and WTI  
- VIX term structure snapshot (1M vs 3M)  
- Time series panels for each vol index  
- Daily shock heatmap across all series

**Use cases**

- Check whether vol is rich or cheap versus its own history  
- See if stress is local to one asset or broad across curves and commodities  
- Sanity check risk-taking against the current vol regime
        """
    )

    st.divider()
    st.header("Settings")
    start_date = st.date_input("History start", datetime(2010, 1, 1))
    lookback = st.slider("Realized vol window (days)", 15, 90, 21)
    show_z = st.checkbox("Overlay Z score on time series", value=True)
    st.caption("Data Yahoo Finance. Research utility only.")


# ---------------- Data universe ----------------
VOL_IDX = [
    "^VIX",      # S&P implied vol
    "^VIX3M",    # 3M implied vol
    "^VVIX",     # Vol of vol
    "^MOVE",     # Treasury vol (may be missing)
    "^GVZ",      # Gold vol (may be missing)
    "^OVX",      # Oil vol (may be missing)
]

PRICE_UNDERLYING = [
    "^GSPC",
    "TLT",
    "GLD",
    "CL=F"
]

ALL_TICKERS = VOL_IDX + PRICE_UNDERLYING

# ---------------- Load data ----------------
series = load_series(ALL_TICKERS, start=str(start_date))

if not series:
    st.error("No data returned from Yahoo Finance for the selected universe.")
    st.stop()

# ---------------- Realized vol ----------------
realized = {}
for p in PRICE_UNDERLYING:
    s = series.get(p)
    if s is None or s.empty:
        continue
    rv = realized_vol(s, window=lookback)
    if not rv.dropna().empty:
        realized[p] = rv


# ---------------- Build table ----------------
rows = []

# Implied vols
for idx in VOL_IDX:
    s = series.get(idx)
    if s is None or s.empty:
        continue
    lvl = float(s.iloc[-1])
    z_val = zscore(s).iloc[-1] if show_z else np.nan
    rows.append([
        idx,
        round(lvl, 2),
        round(z_val, 2) if show_z and not pd.isna(z_val) else ""
    ])

# Realized vols
for p, rv in realized.items():
    if rv.empty:
        continue
    lvl = float(rv.iloc[-1])
    z_val = zscore(rv).iloc[-1] if show_z else np.nan
    rows.append([
        f"{p} 1M RV",
        round(lvl, 2),
        round(z_val, 2) if show_z and not pd.isna(z_val) else ""
    ])

df = pd.DataFrame(rows, columns=["Series", "Level", "Z"])

st.subheader("Current Volatility Levels")

if df.empty:
    st.info("No valid volatility series available under current settings.")
else:
    st.dataframe(df, use_container_width=True)


# ---------------- Term structure panel ----------------
def plot_term_structure(vol, vol3m):
    fig, ax = plt.subplots(figsize=(7, 3))
    points = [0.0, 0.25]
    vals = [np.nan, np.nan]

    if vol is not None and not np.isnan(vol):
        vals[0] = vol
    if vol3m is not None and not np.isnan(vol3m):
        vals[1] = vol3m

    ax.plot(points, vals, marker="o", linewidth=2, color=PASTEL[1])
    ax.set_xticks(points)
    ax.set_xticklabels(["1M", "3M"])
    ax.set_title("VIX Term Structure", color=TEXT)
    ax.grid(color=GRID, linewidth=0.6)
    st.pyplot(fig, clear_figure=True)


st.subheader("VIX Term Structure")
v = series.get("^VIX")
v3 = series.get("^VIX3M")

v_last = float(v.iloc[-1]) if v is not None and not v.empty else None
v3_last = float(v3.iloc[-1]) if v3 is not None and not v3.empty else None

if v_last is None and v3_last is None:
    st.info("VIX or VIX3M data not available for this date range.")
else:
    plot_term_structure(v_last, v3_last)


# ---------------- Multi panel time series ----------------
st.subheader("Volatility Time Series")

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.ravel()

targets = [
    "^VIX", "^VVIX", "^MOVE",
    "^GVZ", "^OVX", "^VIX3M"
]

for i, t in enumerate(targets):
    ax = axes[i]
    s = series.get(t)
    if s is None or s.empty:
        ax.text(0.5, 0.5, f"{t} missing", ha="center", va="center", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        continue

    ax.plot(s.index, s.values, color=PASTEL[i % len(PASTEL)], linewidth=2)
    ax.set_title(t, color=TEXT, fontsize=11)
    ax.grid(color=GRID, linewidth=0.6)

    if show_z:
        zs = zscore(s)
        ax2 = ax.twinx()
        ax2.plot(zs.index, zs.values, color="#999999", linewidth=1, alpha=0.5)
        ax2.axhline(0, color="#999999", linewidth=0.8)
        ax2.set_yticks([])

plt.tight_layout()
st.pyplot(fig, clear_figure=True)


# ---------------- Heatmap of daily changes ----------------
st.subheader("Cross Asset Vol Shock Map")


def daily_change_map(series_dict):
    names = []
    vals = []
    for k, v in series_dict.items():
        if v is None or len(v) < 2:
            continue
        ch = (v.iloc[-1] - v.iloc[-2]) / v.iloc[-2]
        if pd.isna(ch):
            continue
        names.append(k)
        vals.append(float(ch))
    if not names:
        return pd.DataFrame(columns=["Series", "Daily % Change"])
    return pd.DataFrame({"Series": names, "Daily % Change": vals})


heat = daily_change_map({k: v for k, v in series.items() if k in VOL_IDX})

if heat.empty:
    st.info("Insufficient data to compute daily vol shocks.")
else:
    heat = heat.set_index("Series")
    fig_h, ax_h = plt.subplots(figsize=(6, max(2.5, 0.4 * len(heat))))
    cmap = pastel_cmap()
    im = ax_h.imshow(heat.values, cmap=cmap, aspect="auto")

    ax_h.set_yticks(range(len(heat.index)))
    ax_h.set_yticklabels(heat.index)
    ax_h.set_xticks([0])
    ax_h.set_xticklabels(["Daily %"])
    ax_h.grid(False)

    cbar = plt.colorbar(im, ax=ax_h)
    cbar.ax.set_ylabel("Daily change", rotation=270, labelpad=12)

    st.pyplot(fig_h, clear_figure=True)

st.caption("© 2025 AD Fund Management LP · Cross Asset Volatility Surface Monitor")
