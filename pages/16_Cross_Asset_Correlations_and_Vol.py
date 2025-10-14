# Cross-Asset Correlations & Vol — 6 panels, minimal matrix, insight-first

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import ListedColormap
import streamlit as st

# =========================
# App config and style
# =========================
st.set_page_config(page_title="Cross-Asset Correlations & Vol (6-Panel, Minimal Matrix)", layout="wide")

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

st.title("Cross-Asset Correlations & Volatility — Six Panels, Minimal Matrix")
st.caption("Data: Yahoo Finance via yfinance. Matrix is text-first, low-ink. Six correlation panels stay.")

# =========================
# Universe
# =========================
GROUPS = {
    "Equities": ["^GSPC", "^RUT", "^STOXX50E", "^N225", "EEM"],
    "Corporate Credit": ["LQD", "HYG"],
    "Rates": ["^TNX", "^TYX"],
    "Commodities": ["CL=F", "GLD", "HG=F"],
    "FX": ["EURUSD=X", "USDJPY=X", "GBPUSD=X"],
}
ORDER = sum(GROUPS.values(), [])
IV_TICKERS = ["^VIX", "^OVX"]

# =========================
# Sidebar: slim controls
# =========================
st.sidebar.header("Controls")
rho_thresh = st.sidebar.slider("|ρ| threshold", 0.30, 0.90, 0.50, 0.05)
show_abs = st.sidebar.checkbox("Matrix in absolute terms", value=False)
matrix_style = st.sidebar.selectbox("Matrix style", ["Minimal text-only", "Greyscale heat"], index=0)

st.sidebar.caption("Panels: 21D rolling corr, 6M mean ±1σ band, focus 2017+.")
focus_since_2017 = True

# =========================
# Helpers
# =========================
def pastelize_cmap(name="RdYlGn", lighten=0.65, n=256):
    base = plt.cm.get_cmap(name, n)
    colors = base(np.linspace(0, 1, n))
    colors[:, :3] = 1.0 - lighten * (1.0 - colors[:, :3])  # toward white
    return ListedColormap(colors)

PASTEL_RdYlGn = pastelize_cmap("RdYlGn", lighten=0.65)

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_prices(tickers, start, end, interval="1d"):
    try:
        data = yf.download(
            tickers=tickers, start=start, end=end, interval=interval,
            auto_adjust=True, progress=False, group_by="ticker", threads=True
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

def zscores(series):
    s = pd.Series(series).dropna()
    if s.empty:
        return s
    m, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.nan, index=s.index)
    return (s - m) / sd

def percentile_of_last(series):
    s = pd.Series(series).dropna()
    return float(s.rank(pct=True).iloc[-1] * 100.0) if not s.empty else np.nan

def current_value(series):
    s = pd.Series(series).dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan

def apply_focus_window(index):
    if not focus_since_2017:
        return index
    start = pd.Timestamp("2017-01-01")
    return index[index >= start]

def clip_limits(array_like, pmin=1, pmax=99):
    a = pd.Series(np.asarray(array_like).astype(float)).dropna()
    if a.empty:
        return None
    lo, hi = np.percentile(a, [pmin, pmax])
    span = hi - lo
    return (lo - 0.05*span, hi + 0.05*span)

def add_regime_shading(ax):
    spans = [("2020-02-15", "2020-06-30"), ("2022-03-01", "2023-11-01")]
    for a, b in spans:
        ax.axvspan(pd.to_datetime(a), pd.to_datetime(b), color="#cccccc", alpha=0.18)

# =========================
# Data
# =========================
end_date = datetime.now().date()
start_date = (datetime.now() - timedelta(days=365*12)).date()
ALL = list(dict.fromkeys(ORDER + IV_TICKERS))

with st.spinner("Downloading market data..."):
    PRICES = fetch_prices(ALL, str(start_date), str(end_date), interval="1d")

if PRICES.empty:
    st.error("No data downloaded. Check connectivity.")
    st.stop()

BASE = PRICES[[c for c in ORDER if c in PRICES.columns]].copy()
IV   = PRICES[[c for c in IV_TICKERS if c in PRICES.columns]].copy()
RET = pct_returns(BASE)
WIN = 21
REALVOL = realized_vol(RET, window=WIN)

last_date = BASE.index.max()
st.info(f"Data last date: {last_date.strftime('%Y-%m-%d')}" if pd.notna(last_date) else "No dates available.")

# =========================
# Minimal Matrix (lower triangle, text-first)
# =========================
st.subheader("Cross-Asset Correlation Matrix (1M) — Minimal")

def _style_matrix_text(df, thresh=0.5, sign_color=True):
    """Return a Styler that colors text subtly by sign, bolds above thresh, no background fill."""
    def fmt(v):
        return "" if pd.isna(v) else f"{int(round(v*100,0))}%"

    def style_fn(x):
        css = np.full(x.shape, "", dtype=object)
        vals = x.values
        # Bold with thin border above threshold
        hit = (np.abs(vals) >= thresh) & ~np.isnan(vals)
        css[hit] = "font-weight:700; border:1px solid #999;"

        if sign_color:
            # Soft red/green text, opacity scales with |ρ|
            with np.errstate(invalid="ignore"):
                alpha = np.clip(np.abs(vals), 0.2, 0.9)  # floor/ceiling
                red = (vals < 0) & ~np.isnan(vals)
                green = (vals > 0) & ~np.isnan(vals)
            def rgba(a, g=False):
                # muted red or green
                return f"color: rgba({60 if g else 200}, {150 if g else 60}, {80 if g else 60}, {a:.2f});"
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    if red[i, j]:
                        css[i, j] += " " + rgba(alpha[i, j], g=False)
                    elif green[i, j]:
                        css[i, j] += " " + rgba(alpha[i, j], g=True)
        # Center text, uniform font
        css = np.where(css == "", "text-align:center; font-size:0.95rem;", css + " text-align:center; font-size:0.95rem;")
        return pd.DataFrame(css, index=x.index, columns=x.columns)

    return (
        df.style
        .format(fmt)
        .apply(style_fn, axis=None)
        .set_properties(**{"background-color": "white"})  # keep blank background
    )

if len(RET) >= WIN:
    corr_raw = RET.tail(WIN).corr().replace([-np.inf, np.inf], np.nan)
else:
    corr_raw = pd.DataFrame()

if corr_raw.empty:
    st.warning("Not enough data to compute the 1M correlation matrix.")
else:
    order = [t for t in ORDER if t in corr_raw.columns]
    corr_signed = corr_raw.loc[order, order]
    corr = corr_raw.abs().loc[order, order] if show_abs else corr_signed.copy()

    # lower triangle, blank diagonal
    n = corr.shape[0]
    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    mask_diag  = np.eye(n, dtype=bool)
    disp = corr.copy().astype(float)
    disp.values[mask_upper] = np.nan
    disp.values[mask_diag]  = np.nan

    if matrix_style == "Minimal text-only":
        sty = _style_matrix_text(disp, thresh=rho_thresh, sign_color=not show_abs)
        st.dataframe(sty, use_container_width=True, height=min(110 + 40*disp.shape[0], 1200))
    else:
        # Greyscale heat uses |ρ| to avoid color confusion, still blanks upper/diag
        abs_disp = disp.abs()
        sty = (
            abs_disp.style
            .background_gradient(cmap="Greys", vmin=0, vmax=1)
            .format(lambda v: "" if pd.isna(v) else f"{int(round(v*100,0))}%")
            .apply(lambda x: np.where((x.values >= rho_thresh) & ~np.isnan(x.values),
                                      "font-weight:700; border:1px solid #999; text-align:center;",
                                      "text-align:center;"),
                   axis=None)
        )
        st.dataframe(sty, use_container_width=True, height=min(110 + 40*disp.shape[0], 1200))

    st.caption(f"Lower triangle only, blanks on diagonal. Bold marks |ρ| ≥ {rho_thresh:.2f}. "
               + ("Absolute mode removes direction." if show_abs else "Subtle text color shows sign."))

# =========================
# Six correlation panels (two rows of three)
# =========================
st.subheader("Cross-Asset Correlation Panels (21D)")

PANEL_SPECS = [
    dict(a="^GSPC", b="^TNX", la="SPX", lb="US 10Y",      title="Equity vs Rates"),
    dict(a="^RUT",  b="LQD",  la="RTY", lb="IG (LQD)",     title="Small Caps vs IG Credit"),
    dict(a="^GSPC", b="HYG",  la="SPX", lb="HY (HYG)",     title="Equity vs HY Credit"),
    dict(a="^GSPC", b="CL=F", la="SPX", lb="Oil (WTI)",    title="Equity vs Oil"),
    dict(a="^STOXX50E", b="EURUSD=X", la="SX5E", lb="EURUSD", title="Europe vs EURUSD"),
    dict(a="^N225", b="USDJPY=X", la="NKY", lb="USDJPY",   title="Japan vs USDJPY"),
]

def roll_corr(ret_df, a, b, window=21):
    if a not in ret_df.columns or b not in ret_df.columns:
        return pd.Series(dtype=float)
    df = ret_df[[a, b]].dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return df[a].rolling(window).corr(df[b])

def corr_with_context(ax, rc):
    rc = rc.loc[apply_focus_window(rc.index)]
    if rc.empty:
        return None, None
    mean_rc = rc.rolling(126).mean()
    std_rc  = rc.rolling(126).std()
    ax.fill_between(rc.index, (mean_rc-std_rc).values, (mean_rc+std_rc).values, alpha=0.15)
    ax.plot(mean_rc.index, mean_rc.values, linewidth=1.5, label="6M mean")
    ax.plot(rc.index, rc.values, linewidth=1.0, label="21D corr")
    ax.axhline(0, linewidth=1)
    lims = clip_limits(rc.values, 1, 99)
    ax.set_ylim(max(-1, lims[0]) if lims else -1, min(1, lims[1]) if lims else 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.margins(y=0.05); add_regime_shading(ax)
    return rc.iloc[-1], percentile_of_last(rc)

# Row 1
row1 = st.columns(3)
for spec, col in zip(PANEL_SPECS[:3], row1):
    with col:
        fig, ax = plt.subplots()
        rc = roll_corr(RET, spec["a"], spec["b"], WIN)
        curr, pct = corr_with_context(ax, rc)
        ax.set_title(spec["title"]); ax.legend(loc="lower left", fontsize=8); plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        m1, m2 = st.columns(2)
        with m1: st.metric("Current corr", f"{curr:.1%}" if curr is not None and not np.isnan(curr) else "NA")
        with m2: st.metric("Percentile", f"{pct:.0f}th" if pct is not None and not np.isnan(pct) else "NA")

# Row 2
row2 = st.columns(3)
for spec, col in zip(PANEL_SPECS[3:], row2):
    with col:
        fig, ax = plt.subplots()
        rc = roll_corr(RET, spec["a"], spec["b"], WIN)
        curr, pct = corr_with_context(ax, rc)
        ax.set_title(spec["title"]); ax.legend(loc="lower left", fontsize=8); plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        m1, m2 = st.columns(2)
        with m1: st.metric("Current corr", f"{curr:.1%}" if curr is not None and not np.isnan(curr) else "NA")
        with m2: st.metric("Percentile", f"{pct:.0f}th" if pct is not None and not np.isnan(pct) else "NA")

# =========================
# Volatility snapshot (ranked)
# =========================
st.subheader("Vol snapshot — standardized, ranked")

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
    zdf = pd.DataFrame(z_panel).dropna(how="all").apply(zscores, axis=0)
    zdf = zdf.loc[apply_focus_window(zdf.index)]
    if not zdf.empty:
        latest = zdf.tail(1).T.rename(columns={zdf.index[-1]: "Current Z"})
        latest["Percentile"] = pd.Series({c: percentile_of_last(zdf[c]) for c in zdf.columns})
        latest = latest.sort_values("Current Z", ascending=False)
        st.dataframe(
            latest.style.format({"Current Z": "{:.2f}", "Percentile": "{:.0f}%"}).background_gradient(
                subset=["Current Z"], cmap="RdYlGn_r"),
            use_container_width=True,
            height=min(60 + 28*len(latest), 700),
        )
else:
    st.info("Not enough series for standardized snapshot.")
