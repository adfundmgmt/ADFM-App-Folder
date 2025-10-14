# Cross-Asset Correlation & Volatility Dashboard - Insight-first, final
# Key features
# - Single batched Yahoo fetch (cached)
# - Rolling correlations on returns
# - Correlation matrix: red->green for signed, Greens for absolute, threshold highlights, compact Top pairs, correct group-to-group view
# - Correlation panels: 6M mean ±1σ band, metric tiles, subtle regime shading
# - Vol monitor: IQR shading, metric tiles, implied overlays
# - Z-score snapshot: grouped into tabs, tight layout
# - Sidebar reading guide restored

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
st.caption("Data: Yahoo Finance via yfinance. Universe is locked. Views add percentiles, bands, tiles, and group structure.")

# =========================
# Fixed universe and order (Yahoo symbols)
# =========================
GROUPS = {
    "Equities": ["^GSPC", "^RUT", "^STOXX50E", "^N225", "EEM"],
    "Corporate Credit": ["LQD", "HYG"],
    "Rates": ["^TNX", "^TYX"],
    "Commodities": ["CL=F", "GLD", "HG=F"],
    "Foreign Exchange": ["EURUSD=X", "USDJPY=X", "GBPUSD=X"],
}
ORDER = sum(GROUPS.values(), [])
IV_TICKERS = ["^VIX", "^OVX"]

# =========================
# Sidebar: guide and controls
# =========================
st.sidebar.header("Reading guide")
st.sidebar.markdown(
"""
**Core definitions**
- 1M = 21 trading days.
- Correlation panels show 21D rolling correlation with a 6M mean and ±1σ band to separate noise from drift.
- Matrix uses the last 21 sessions of returns. Lower triangle only. Bold cells cross a user-set |ρ| threshold. Use absolute mode to see structure, signed mode for hedge direction.
- Volatility: realized = 21D stdev of returns, annualized and in percent. Implied overlays use ^VIX for SPX and ^OVX for oil.
- Z-scores standardize each series to its own full history, grouped in tabs for readability.

**Reading tips**
- Negative equity-rates correlation means rate rallies often support equities.
- Positive equity-credit correlation means credit sells off with equities. Watch HY.
- Use Top pairs tables for immediate hedge and pair ideas.
"""
)

st.sidebar.header("View controls")
focus_since_2017 = st.sidebar.checkbox("Focus on 2017 to present", value=True)
clip_extremes = st.sidebar.checkbox("Clip y-axes at 1st to 99th percentiles", value=True)
show_regimes = st.sidebar.checkbox("Show regime shading (no labels)", value=True)
z_snapshot_mode = st.sidebar.selectbox("Z-score layout", ["Tabs (recommended)", "Stacked"], index=0)

# =========================
# Helpers
# =========================
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_prices(tickers, start, end, interval="1d"):
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

def percentile_of_last(series):
    s = pd.Series(series).dropna()
    if s.empty:
        return np.nan
    return float(s.rank(pct=True).iloc[-1] * 100.0)

def current_value(series):
    s = pd.Series(series).dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan

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
        ("2020-02-15", "2020-06-30"),
        ("2022-03-01", "2023-11-01"),
    ]
    for a, b in spans:
        ax.axvspan(pd.to_datetime(a), pd.to_datetime(b), color="#cccccc", alpha=0.18)

# =========================
# Data download (single batched call)
# =========================
end_date = datetime.now().date()
start_date = (datetime.now() - timedelta(days=365*12)).date()
ALL_TICKERS = list(dict.fromkeys(ORDER + IV_TICKERS))

with st.spinner("Downloading market data from Yahoo Finance..."):
    PRICES = fetch_prices(ALL_TICKERS, str(start_date), str(end_date), interval="1d")

if PRICES.empty:
    st.error("No data downloaded. Check connectivity.")
    st.stop()

BASE = PRICES[[c for c in ORDER if c in PRICES.columns]].copy()
IV   = PRICES[[c for c in IV_TICKERS if c in PRICES.columns]].copy()

last_date = BASE.index.max()
st.info(f"Data last date: {last_date.strftime('%Y-%m-%d')}" if pd.notna(last_date) else "No dates available.")

RET = pct_returns(BASE)
WIN = 21
REALVOL = realized_vol(RET, window=WIN)

# =========================
# Correlation Matrix - insight-first
# =========================
st.subheader("Cross-Asset Correlation Matrix (1M)")

with st.expander("Matrix options", expanded=True):
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        rho_thresh = st.slider("|ρ| highlight threshold", 0.0, 1.0, 0.50, 0.05)
    with c2:
        show_abs = st.checkbox("Show absolute correlations", value=False)
    with c3:
        sort_mode = st.selectbox("Ordering", ["Original groups", "By average |ρ|"], index=0)

if len(RET) >= WIN:
    recent_ret = RET.tail(WIN)
    corr_raw = recent_ret.corr().replace([-np.inf, np.inf], np.nan)
else:
    corr_raw = pd.DataFrame()

if corr_raw.empty:
    st.warning("Not enough data to compute the 1M correlation matrix.")
else:
    corr_signed = corr_raw.copy()
    corr = corr_raw.abs() if show_abs else corr_raw.copy()

    # ordering
    if sort_mode == "By average |ρ|":
        avg_abs = corr_raw.abs().mean().sort_values(ascending=False).index.tolist()
        order = [t for t in avg_abs if t in corr.columns]
    else:
        order = [t for t in ORDER if t in corr.columns]

    corr = corr.loc[order, order]
    corr_signed = corr_signed.loc[order, order]

    # lower triangle, blank diagonal
    n = corr.shape[0]
    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    mask_diag  = np.eye(n, dtype=bool)
    disp = corr.copy().astype(float)
    disp.values[mask_upper] = np.nan
    disp.values[mask_diag]  = np.nan

    # multiindex headers with groups
    group_colors = {"Equities":"#d9ead3","Corporate Credit":"#d0e0e3","Rates":"#fce5cd","Commodities":"#fff2cc","Foreign Exchange":"#ead1dc"}
    col_groups = []
    for t in disp.columns:
        g = next((g for g, lst in GROUPS.items() if t in lst), "Other")
        col_groups.append(g)
    disp.columns = pd.MultiIndex.from_arrays([col_groups, disp.columns], names=["Group", "Ticker"])
    disp.index.name = "Ticker"

    # styling: red->green for signed, Greens for abs
    cmap = "RdYlGn" if not show_abs else "Greens"

    def _highlight_threshold(x):
        css = np.full(x.shape, "", dtype=object)
        cond = (np.abs(x.values) >= rho_thresh) & ~np.isnan(x.values)
        css[cond] = "font-weight:700; border:1px solid #777;"
        for i in range(min(x.shape)):
            css[i, i] = "background-color:#efefef;"
        return pd.DataFrame(css, index=x.index, columns=x.columns)

    header_styles = []
    for i, (g, _) in enumerate(disp.columns.tolist()):
        header_styles.append({
            "selector": f"th.col_heading.level0.col{i}",
            "props": [("background-color", group_colors.get(g, "#f2f2f2"))]
        })

    sty = (
        disp.style
            .background_gradient(cmap=cmap, vmin=(-1 if not show_abs else 0), vmax=1)
            .format(na_rep="", formatter=lambda v: "" if pd.isna(v) else f"{int(round(v*100,0))}%")
            .apply(_highlight_threshold, axis=None)
            .set_table_styles(header_styles, overwrite=False)
            .set_properties(**{"text-align":"center"})
    )
    rows = disp.shape[0]
    st.dataframe(sty, use_container_width=True, height=min(110 + 40*rows, 2400))
    st.caption(f"Lower triangle only. Bold cells meet |ρ| ≥ {rho_thresh:.2f}. Red=negative, green=positive. Absolute mode removes direction.")

    # compact top pairs
    def pair_snapshot(ret_df, a, b):
        r = ret_df[[a, b]].dropna()
        if r.empty:
            return None
        rc = r[a].rolling(WIN).corr(r[b]).dropna()
        if rc.empty:
            return None
        now = rc.iloc[-1]
        pct = float(rc.rank(pct=True).iloc[-1] * 100.0)
        return now, pct

    pairs = []
    tickers = disp.columns.get_level_values("Ticker").tolist()
    for i in range(1, len(tickers)):
        for j in range(0, i):
            a, b = tickers[i], tickers[j]
            ps = pair_snapshot(RET, a, b)
            if ps is None:
                continue
            now, pct = ps
            pairs.append({"A": a, "B": b, "ρ": now, "|ρ|": abs(now), "Percentile(5y)": pct})

    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty:
        filt = pairs_df[pairs_df["|ρ|"] >= rho_thresh].copy().sort_values("|ρ|", ascending=False)

        def _render_pairs(df):
            if df.empty:
                return df
            out = df.copy()
            out["Corr"] = out["ρ"].map(lambda x: f"{x: .2f}")
            out["Percentile(5y)"] = out["Percentile(5y)"].map(lambda x: f"{x:.0f}th")
            return out[["A","B","Corr","Percentile(5y)"]].rename(columns={"A":"Leg A","B":"Leg B"})

        c1, c2, c3 = st.columns([1.2, 1.2, 1])
        with c1:
            st.markdown("**Top positive pairs (now)**")
            pos = filt.sort_values("ρ", ascending=False).head(6)
            st.dataframe(_render_pairs(pos), use_container_width=True, height=262)
        with c2:
            st.markdown("**Top negative pairs (now)**")
            neg = filt.sort_values("ρ", ascending=True).head(6)
            st.dataframe(_render_pairs(neg), use_container_width=True, height=262)
        with c3:
            st.markdown("**Stats**")
            st.metric("Pairs ≥ threshold", f"{len(filt)}/{len(pairs_df)}")
            st.metric("Median |ρ| (all)", f"{pairs_df['|ρ|'].median():.2f}")
            st.metric("Max |ρ| now", f"{pairs_df['|ρ|'].max():.2f}")

        # correct group-to-group averages from same window
        def group_avg_corr_from_returns(ret_df, groups, window=WIN):
            if len(ret_df) < window:
                return pd.DataFrame()
            C = ret_df.tail(window).corr().replace([-np.inf, np.inf], np.nan)
            group_names = [g for g in groups.keys() if any(t in C.columns for t in groups[g])]
            M = pd.DataFrame(index=group_names, columns=group_names, dtype=float)
            for ga in group_names:
                Ta = [t for t in groups[ga] if t in C.columns]
                for gb in group_names:
                    Tb = [t for t in groups[gb] if t in C.columns]
                    if not Ta or not Tb:
                        M.loc[ga, gb] = np.nan
                        continue
                    sub = C.loc[Ta, Tb].to_numpy()
                    if ga == gb:
                        if sub.shape[0] == sub.shape[1]:
                            mask = np.ones(sub.shape, dtype=bool)
                            np.fill_diagonal(mask, False)
                            vals = sub[mask]
                        else:
                            vals = sub.flatten()
                    else:
                        vals = sub.flatten()
                    vals = vals[~np.isnan(vals)]
                    M.loc[ga, gb] = np.nan if vals.size == 0 else float(np.mean(vals))
            return M

        GAVG = group_avg_corr_from_returns(RET, GROUPS, WIN)

        st.markdown("**Group-to-group average correlation (signed, last 21 sessions)**")
        if GAVG.empty:
            st.info("Not enough data for group summary.")
        else:
            fig, ax = plt.subplots(figsize=(6.0, 4.8))
            im = ax.imshow(GAVG.values, vmin=-1, vmax=1, cmap="RdYlGn")
            ax.set_xticks(range(len(GAVG.columns))); ax.set_xticklabels(GAVG.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(GAVG.index)));   ax.set_yticklabels(GAVG.index)
            for i in range(GAVG.shape[0]):
                for j in range(GAVG.shape[1]):
                    val = GAVG.iat[i, j]
                    ax.text(j, i, "" if np.isnan(val) else f"{val*100:.0f}%", ha="center", va="center", fontsize=9)
            ax.set_title("Average signed correlation by asset class")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel("Correlation", rotation=270, labelpad=12)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)

# =========================
# Correlation panels with bands and tiles
# =========================
st.subheader("Cross-Asset Correlation Analysis")

PANEL_SPECS = [
    dict(a="^GSPC", b="^TNX", la="SPX", lb="US 10Y",      title="Equity - Rates (1M)"),
    dict(a="^RUT",  b="LQD",  la="RTY", lb="IG (LQD)",     title="Equity - IG Credit (1M)"),
    dict(a="^GSPC", b="CL=F", la="SPX", lb="Oil (WTI)",    title="Equity - Oil (1M)"),
    dict(a="^GSPC", b="GLD",  la="SPX", lb="GLD",          title="Equity - Gold (1M)"),
    dict(a="^STOXX50E", b="EURUSD=X", la="SX5E", lb="EURUSD", title="Equity - EURUSD (1M)"),
    dict(a="^N225", b="USDJPY=X", la="NKY", lb="USDJPY",   title="Equity - USDJPY (1M)"),
]

def roll_corr_from_returns(ret_df, a, b, window=21):
    if a not in ret_df.columns or b not in ret_df.columns:
        return pd.Series(dtype=float)
    df = ret_df[[a, b]].dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return df[a].rolling(window).corr(df[b])

def corr_with_context(ax, rc):
    rc = rc.loc[apply_focus_window(rc.index)]
    if rc.empty:
        return None, None, None
    mean_rc = rc.rolling(126).mean()
    std_rc  = rc.rolling(126).std()
    ax.fill_between(rc.index, (mean_rc-std_rc).values, (mean_rc+std_rc).values, alpha=0.15)
    ax.plot(mean_rc.index, mean_rc.values, linewidth=1.5, label="6M mean")
    ax.plot(rc.index, rc.values, linewidth=1.0, label="21D corr")
    ax.axhline(0, linewidth=1)
    lims = clip_limits(rc.values, 1, 99)
    if lims:
        ax.set_ylim(max(-1, lims[0]), min(1, lims[1]))
    else:
        ax.set_ylim(-1, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.margins(y=0.05)
    add_regime_shading(ax)
    return current_value(rc), percentile_of_last(rc), rc

for row_specs in [PANEL_SPECS[:3], PANEL_SPECS[3:]]:
    col1, col2, col3 = st.columns(3)
    for spec, col in zip(row_specs, [col1, col2, col3]):
        with col:
            fig, ax = plt.subplots()
            rc = roll_corr_from_returns(RET, spec["a"], spec["b"], WIN)
            curr, pct, series = corr_with_context(ax, rc)
            ax.set_title(spec["title"])
            ax.legend(loc="lower left", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Current corr", f"{curr:.1%}" if curr is not None and not np.isnan(curr) else "NA")
            with m2:
                st.metric("Percentile vs history", f"{pct:.0f}th" if pct is not None and not np.isnan(pct) else "NA")

# =========================
# Volatility Monitor
# =========================
st.subheader("Cross-Asset Volatility Monitor")

def iqr_band(ax, s):
    s = pd.Series(s).dropna()
    if s.empty:
        return
    q25, q75 = np.nanpercentile(s, [25, 75])
    ax.axhspan(q25, q75, color="lightgray", alpha=0.25)

def vol_panel(series, label, title, overlay=None):
    fig, ax = plt.subplots()
    s = series.loc[apply_focus_window(series.index)] if series is not None else pd.Series(dtype=float)
    if s is None or s.empty:
        st.info(f"{label} unavailable.")
        return
    ax.plot(s.index, s.values, label=label)
    if overlay is not None and not overlay.empty:
        o = overlay.loc[apply_focus_window(overlay.index)]
        ax.plot(o.index, o.values, label=overlay.name)
    iqr_band(ax, s)
    lims = clip_limits(s.values, 1, 99)
    if lims: ax.set_ylim(lims)
    ax.set_title(title)
    ax.set_ylabel("Percent")
    ax.margins(y=0.05)
    add_regime_shading(ax)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    m1, m2 = st.columns(2)
    with m1: st.metric("Current", f"{current_value(s):.1f}%")
    with m2: st.metric("Percentile vs history", f"{percentile_of_last(s):.0f}th")

# Row 1
c1, c2, c3 = st.columns(3)
with c1:
    if "^GSPC" in REALVOL.columns:
        vix = IV["^VIX"] if "^VIX" in IV.columns else None
        if vix is not None: vix.name = "VIX Index"
        vol_panel(REALVOL["^GSPC"] * 100, "SPX 1M Realized Vol", "Equity vol, implied vs realized", overlay=vix)
    else:
        st.info("SPX realized vol unavailable.")

with c2:
    if "^TNX" in REALVOL.columns:
        vol_panel(REALVOL["^TNX"] * 100, "US 10Y Realized Vol", "Rates vol, realized proxy")
    else:
        st.info("US 10Y realized vol unavailable.")

with c3:
    if "CL=F" in REALVOL.columns:
        ovx = IV["^OVX"] if "^OVX" in IV.columns else None
        if ovx is not None: ovx.name = "OVX Index"
        vol_panel(REALVOL["CL=F"] * 100, "Oil 1M Realized Vol", "Oil vol, implied vs realized", overlay=ovx)
    else:
        st.info("Oil realized vol unavailable.")

# Row 2
c4, c5, c6 = st.columns(3)
with c4:
    if "LQD" in REALVOL.columns:
        vol_panel(REALVOL["LQD"] * 100, "IG 1M Realized Vol", "IG credit vol, realized proxy")
    else:
        st.info("IG realized vol unavailable.")

with c5:
    if "HYG" in REALVOL.columns:
        vol_panel(REALVOL["HYG"] * 100, "HY 1M Realized Vol", "HY credit vol, realized proxy")
    else:
        st.info("HY realized vol unavailable.")

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
        ax.set_title("FX vol, realized proxies")
        ax.set_ylabel("Percent")
        ax.margins(y=0.05)
        add_regime_shading(ax)
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        pick = None
        for name in ["EURUSD=X", "USDJPY=X", "GBPUSD=X"]:
            if name in REALVOL.columns:
                pick = (REALVOL[name] * 100).loc[apply_focus_window(REALVOL.index)]
                break
        if pick is not None and not pick.empty:
            m1, m2 = st.columns(2)
            with m1: st.metric("FX realized, current", f"{current_value(pick):.1f}%")
            with m2: st.metric("FX realized, percentile", f"{percentile_of_last(pick):.0f}th")
    else:
        st.info("FX realized vol unavailable.")

# =========================
# Z-score snapshot - grouped
# =========================
st.subheader("Volatility Snapshot - Standardized, Grouped")

z_panel = {}
if "^VIX" in IV.columns:  z_panel["^VIX"] = IV["^VIX"]
if "^OVX" in IV.columns:  z_panel["^OVX"] = IV["^OVX"]

mapping = [("^GSPC", "SPX Realized"),
           ("^TNX", "US10Y Realized"),
           ("LQD", "IG Realized"),
           ("HYG", "HY Realized"),
           ("CL=F", "Oil Realized"),
           ("GLD", "Gold Realized"),
           ("EURUSD=X", "EURUSD Realized"),
           ("USDJPY=X", "USDJPY Realized"),
           ("GBPUSD=X", "GBPUSD Realized")]
for sym, label in mapping:
    if sym in REALVOL.columns:
        z_panel[label] = REALVOL[sym] * 100

if len(z_panel) >= 2:
    zdf = pd.DataFrame(z_panel).dropna(how="all")
    zdf = zdf.apply(zscores_full, axis=0)
    zdf = zdf.loc[apply_focus_window(zdf.index)]

    if not zdf.empty:
        groups = {
            "Implied vol": [c for c in ["^VIX", "^OVX"] if c in zdf.columns],
            "Rates and credit": [c for c in ["US10Y Realized", "IG Realized", "HY Realized"] if c in zdf.columns],
            "Commodities and FX": [c for c in ["Oil Realized", "Gold Realized", "EURUSD Realized", "USDJPY Realized", "GBPUSD Realized"] if c in zdf.columns],
        }

        def plot_group(cols, title):
            fig, ax = plt.subplots(figsize=(10, 3.8))
            for c in cols:
                ax.plot(zdf.index, zdf[c], label=c)
            ax.axhline(0, linestyle="--", linewidth=1)
            add_regime_shading(ax)
            lims = clip_limits(zdf[cols].values.flatten(), 1, 99)
            if lims:
                ax.set_ylim(lims)
            ax.set_ylabel("Z-Score")
            ax.set_title(title + " - standardized to full history")
            ax.legend(ncol=min(3, len(cols)), fontsize=8)
            ax.margins(y=0.05)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        if z_snapshot_mode.startswith("Tabs"):
            tabs = st.tabs(list(groups.keys()))
            for (name, cols), t in zip(groups.items(), tabs):
                with t:
                    if cols: plot_group(cols, name)
        else:
            for name, cols in groups.items():
                if cols: plot_group(cols, name)

        latest = zdf.tail(1).T.rename(columns={zdf.index[-1]: "Current Z"})
        latest["Percentile"] = pd.Series({c: percentile_of_last(zdf[c]) for c in zdf.columns})
        latest = latest.sort_values("Current Z", ascending=False)
        st.markdown("**Current standardized vol (ranked)**")
        st.dataframe(
            latest.style.format({"Current Z": "{:.2f}", "Percentile": "{:.0f}%"}).background_gradient(
                subset=["Current Z"], cmap="RdYlGn_r"),
            use_container_width=True
        )
    else:
        st.info("Not enough overlapping history for standardized snapshot.")
else:
    st.info("Not enough series to compute standardized snapshot.")
