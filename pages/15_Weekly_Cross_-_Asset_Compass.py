# cross_asset_weekly_compass.py
# Weekly decision-grade view: correlations, vol, regime, conclusions, actions.
# Tables are rendered at the BOTTOM with clean formatting and clear columns.

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import streamlit as st

# =========================
# App config and style
# =========================
st.set_page_config(page_title="Weekly Cross-Asset Compass", layout="wide")

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

st.title("Weekly Cross-Asset Compass")
st.caption("Data: Yahoo Finance via yfinance. Focus: what changed this week, why it matters, and how to express it.")

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
IV_TICKERS = ["^VIX", "^OVX", "^VIX3M"]

PAIR_SPECS = [
    dict(a="^GSPC", b="^TNX", la="SPX", lb="US10Y", title="Equity vs Rates"),
    dict(a="^GSPC", b="HYG",  la="SPX", lb="HY",    title="Equity vs HY Credit"),
    dict(a="^GSPC", b="CL=F", la="SPX", lb="WTI",   title="Equity vs Oil"),
    dict(a="^RUT",  b="LQD",  la="RTY", lb="IG",    title="Small Caps vs IG"),
    dict(a="^STOXX50E", b="EURUSD=X", la="SX5E", lb="EURUSD", title="Europe vs EUR"),
    dict(a="^N225", b="USDJPY=X", la="NKY", lb="USDJPY", title="Japan vs USDJPY"),
]

# =========================
# Sidebar
# =========================
st.sidebar.header("Controls")
lookback_years = st.sidebar.slider("History window (years)", 5, 15, 10)
win = st.sidebar.selectbox("Rolling window", [21, 63], index=0)  # 1M or ~3M
wk_delta = st.sidebar.selectbox("Compare vs", ["1 week ago", "2 weeks ago"], index=0)
wk_back = 5 if wk_delta == "1 week ago" else 10  # business days
rho_alert = st.sidebar.slider("|ρ| alert threshold", 0.30, 0.90, 0.50, 0.05)
z_hot = st.sidebar.slider("Hot vol Z threshold", 1.0, 3.0, 1.5, 0.1)
z_cold = st.sidebar.slider("Cold vol Z threshold", -3.0, -1.0, -1.5, 0.1)
clip_extremes = st.sidebar.checkbox("Clip chart y-axes at 1st–99th pct", value=True)

# =========================
# Helpers
# =========================
def pastelize_cmap(name="RdYlGn", lighten=0.70, n=256):
    base = plt.cm.get_cmap(name, n)
    colors = base(np.linspace(0, 1, n))
    colors[:, :3] = 1.0 - lighten * (1.0 - colors[:, :3])
    return ListedColormap(colors)

def pastel_red_white_green():
    return LinearSegmentedColormap.from_list(
        "RwG_pastel",
        [(0.93, 0.60, 0.60), (1.0, 1.0, 1.0), (0.60, 0.85, 0.60)],
        N=256
    )

PASTEL_RdYlGn = pastelize_cmap("RdYlGn", lighten=0.70)
PASTEL_RWG = pastel_red_white_green()

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

def clip_limits(array_like, pmin=1, pmax=99):
    if not clip_extremes:
        return None
    a = pd.Series(np.asarray(array_like).astype(float)).dropna()
    if a.empty:
        return None
    lo, hi = np.percentile(a, [pmin, pmax])
    span = hi - lo
    return (lo - 0.05*span, hi + 0.05*span)

def roll_corr(ret_df, a, b, window=21):
    if a not in ret_df.columns or b not in ret_df.columns:
        return pd.Series(dtype=float)
    df = ret_df[[a, b]].dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return df[a].rolling(window).corr(df[b])

def computed_height(n_rows, row_px=32, header_px=38, pad_px=16, max_px=800):
    return min(max_px, int(header_px + row_px * n_rows + pad_px))

# =========================
# Data
# =========================
end_date = datetime.now().date()
start_date = (datetime.now() - timedelta(days=365*lookback_years)).date()
ALL = list(dict.fromkeys(ORDER + IV_TICKERS))

with st.spinner("Downloading market data..."):
    PRICES = fetch_prices(ALL, str(start_date), str(end_date), interval="1d")

if PRICES.empty:
    st.error("No data downloaded. Check connectivity.")
    st.stop()

BASE = PRICES[[c for c in ORDER if c in PRICES.columns]].copy()
IV   = PRICES[[c for c in IV_TICKERS if c in PRICES.columns]].copy()
RET = pct_returns(BASE)
REALVOL = realized_vol(RET, window=21)

last_date = BASE.index.max()
wk_ago_idx = BASE.index.get_loc(last_date) - wk_back if last_date is not None else None
st.info(f"Data last date: {last_date.strftime('%Y-%m-%d')}" if pd.notna(last_date) else "No dates available.")

# =========================
# Regime classification
# =========================
def vix_term_structure(iv_df):
    if "^VIX" in iv_df.columns and "^VIX3M" in iv_df.columns:
        ts = iv_df["^VIX3M"] / iv_df["^VIX"] - 1.0
        return ts.dropna()
    return pd.Series(dtype=float)

def classify_regime():
    ts = vix_term_structure(IV)
    vix = IV["^VIX"] if "^VIX" in IV.columns else pd.Series(dtype=float)
    rv_spx = REALVOL["^GSPC"] * 100 if "^GSPC" in REALVOL.columns else pd.Series(dtype=float)
    corr_er = roll_corr(RET, "^GSPC", "^TNX", win)

    ts_now = current_value(ts)
    vix_now = current_value(vix)
    rv_now = current_value(rv_spx)
    corr_now = current_value(corr_er)

    stress = 0
    if ts_now is not None and not np.isnan(ts_now) and ts_now < 0:
        stress += 1
    if vix_now is not None and not np.isnan(vix_now) and vix_now >= 25:
        stress += 1
    if rv_now is not None and not np.isnan(rv_now) and percentile_of_last(rv_spx) >= 70:
        stress += 1
    if corr_now is not None and not np.isnan(corr_now) and corr_now >= 0.5:
        stress += 1

    if stress >= 3:
        regime = "Risk-off"
    elif stress == 2:
        regime = "Cautious"
    else:
        regime = "Risk-on to neutral"

    return dict(regime=regime, ts_now=ts_now, vix_now=vix_now, rv_now=rv_now, corr_now=corr_now)

reg = classify_regime()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Regime", reg["regime"])
c2.metric("VIX3M/VIX − 1", f"{reg['ts_now']:+.3f}" if reg["ts_now"] is not None and not np.isnan(reg["ts_now"]) else "NA")
c3.metric("VIX", f"{reg['vix_now']:.1f}" if reg["vix_now"] is not None and not np.isnan(reg["vix_now"]) else "NA")
c4.metric("SPX vs 10Y ρ", f"{reg['corr_now']:.1%}" if reg["corr_now"] is not None and not np.isnan(reg["corr_now"]) else "NA")

# =========================
# Conclusions and actions
# =========================
st.subheader("Conclusions and actions")

bullets = []

corr_er = roll_corr(RET, "^GSPC", "^TNX", win)
if not corr_er.empty:
    rho = corr_er.iloc[-1]
    if rho <= -0.30:
        bullets.append(f"Equity vs rates negative at {rho:.0%}. Duration rallies should support equities.")
    elif rho >= 0.30:
        bullets.append(f"Equity vs rates positive at {rho:.0%}. Rate selloffs are pressuring equities.")

if "^GSPC" in REALVOL.columns and "^VIX" in IV.columns:
    rv = (REALVOL["^GSPC"] * 100).dropna()
    vix = IV["^VIX"].dropna()
    both = rv.to_frame("rv").join(vix.to_frame("vix"), how="inner")
    if not both.empty:
        spread = both["vix"] - both["rv"]
        z = zscores(spread)
        sp_now = float(spread.iloc[-1])
        z_now  = float(z.iloc[-1]) if not z.empty else np.nan
        if z_now >= 1.0:
            bullets.append(f"VIX rich to realized by {sp_now:.1f} pts. Overwrite or short-vol structures are attractive.")
        elif z_now <= -1.0:
            bullets.append(f"VIX cheap to realized by {sp_now:.1f} pts. Consider buying convexity or collars.")

ts = vix_term_structure(IV)
if not ts.dropna().empty:
    ts_now = ts.iloc[-1]
    if ts_now < 0:
        bullets.append("VIX term structure in backwardation. Respect gap risk and keep leverage modest.")
    elif ts_now > 0.10:
        bullets.append("VIX term structure in contango. Carry and relative value should work better.")

def add_hot_cold(label, s):
    if s is None or s.dropna().empty:
        return
    z = zscores(s.dropna())
    if z.empty:
        return
    z_now = float(z.iloc[-1])
    if z_now >= z_hot:
        bullets.append(f"{label} volatility hot on a Z of {z_now:.2f}. Tighten risk on exposures tied to this factor.")
    if z_now <= z_cold:
        bullets.append(f"{label} volatility cold on a Z of {z_now:.2f}. Harvest carry where risk is bounded.")

if "HYG" in REALVOL.columns:
    add_hot_cold("High yield", REALVOL["HYG"] * 100)
if "CL=F" in REALVOL.columns:
    add_hot_cold("Oil", REALVOL["CL=F"] * 100)

if reg["regime"] == "Risk-off":
    bullets.append("Overall read is risk-off. Favor defensives, reduce gross, add convexity.")
elif reg["regime"] == "Cautious":
    bullets.append("Overall read is cautious. Keep nets modest and rely on pairs and relative value.")
else:
    bullets.append("Overall read is neutral to constructive. Let carry work but define invalidations.")

express = [
    "If VIX rich to realized, run covered calls or short-dated call spreads against longs.",
    "If HY vol stays hot and equities track credit, hedge with HYG puts or IG vs HY relative value."
]
invalid = [
    "Flip or reduce if SPX vs 10Y correlation crosses back through zero and holds for a week.",
    "Flip or reduce if VIX term structure turns to contango and realized vol falls.",
    "Re-assess if the VIX minus realized spread mean reverts by more than 1 standard deviation."
]

st.markdown("**Why this matters and how to act**")
if bullets:
    st.markdown("\n".join(f"- {b}" for b in bullets))
else:
    st.info("No strong weekly conclusions at current thresholds.")

st.markdown("**Expressions**")
st.markdown("\n".join(f"- {e}" for e in express))
st.markdown("**Invalidations**")
st.markdown("\n".join(f"- {x}" for x in invalid))

# =========================
# Compact visuals
# =========================
st.subheader("Correlation panels, rolling")
def corr_with_context(ax, rc):
    rc = rc.dropna()
    if rc.empty:
        return
    mean_rc = rc.rolling(126).mean()
    std_rc  = rc.rolling(126).std()
    ax.fill_between(rc.index, (mean_rc-std_rc).values, (mean_rc+std_rc).values, alpha=0.15)
    ax.plot(mean_rc.index, mean_rc.values, linewidth=1.5, label="6M mean")
    ax.plot(rc.index, rc.values, linewidth=1.0, label=f"{win}D corr")
    ax.axhline(0, linewidth=1)
    lims = clip_limits(rc.values, 1, 99)
    if lims:
        ax.set_ylim(max(-1, lims[0]), min(1, lims[1]))
    else:
        ax.set_ylim(-1, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.margins(y=0.05)
    ax.legend(loc="lower left", fontsize=8)

for row in [PAIR_SPECS[:3], PAIR_SPECS[3:]]:
    cols = st.columns(3)
    for spec, col in zip(row, cols):
        with col:
            fig, ax = plt.subplots()
            rc = roll_corr(RET, spec["a"], spec["b"], win)
            if rc.dropna().empty:
                st.info("No data")
            else:
                corr_with_context(ax, rc)
                ax.set_title(spec["title"])
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

# =========================
# Correlation matrix quickscan
# =========================
st.subheader("Cross-Asset Correlation Matrix (1M, lower triangle)")
corr_raw = RET.tail(21).corr().replace([-np.inf, np.inf], np.nan) if len(RET) >= 21 else pd.DataFrame()
if corr_raw.empty:
    st.info("Insufficient data for matrix.")
else:
    order = [t for t in ORDER if t in corr_raw.columns]
    mat = corr_raw.loc[order, order].copy().astype(float)
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            if j >= i:
                mat.iat[i, j] = np.nan
            if i == j:
                mat.iat[i, j] = np.nan
    def _fmt(v): return "" if pd.isna(v) else f"{int(round(v*100, 0))}%"
    sty = (
        mat.style
           .background_gradient(cmap=PASTEL_RWG, vmin=-0.75, vmax=0.75)
           .format(_fmt)
           .apply(lambda x: np.where((np.abs(x.values.astype(float)) >= rho_alert) & ~np.isnan(x.values),
                                     "color:black; font-weight:700; border:1px solid #888; text-align:center; font-size:0.95rem;",
                                     "color:black; text-align:center; font-size:0.95rem;"),
                  axis=None)
           .set_properties(**{"background-color": "white"})
    )
    st.dataframe(
        sty,
        use_container_width=True,
        height=computed_height(mat.shape[0])
    )
    st.caption(f"Bold marks |ρ| ≥ {rho_alert:.2f}. Use matrix to sanity-check panel reads.")

# =========================
# WEEKLY SNAPSHOTS — placed at the BOTTOM
# =========================
# Build correlation snapshot table (clean, no blanks, clarified columns)
rows = []
for spec in PAIR_SPECS:
    rc = roll_corr(RET, spec["a"], spec["b"], win)
    if rc.empty or np.isnan(rc.iloc[-1]):
        continue
    curr = float(rc.iloc[-1])
    delta = np.nan
    if rc.shape[0] > wk_back + 1 and not np.isnan(rc.iloc[-wk_back-1]):
        delta = float(rc.iloc[-1] - rc.iloc[-wk_back-1])
    pct = percentile_of_last(rc)
    rows.append([spec["title"], spec["a"], spec["b"], curr, delta, pct])

corr_tbl = pd.DataFrame(rows, columns=[
    "Pair",
    "Series A",
    "Series B",
    "ρ now (rolling)",
    "Δρ w/w",
    "Percentile rank"
]).dropna(how="all")

if not corr_tbl.empty:
    corr_tbl = corr_tbl.sort_values("Δρ w/w", key=lambda s: s.abs(), ascending=False)

# Build volatility snapshot with explicit units/classes
vol_rows = []
def add_vol(label, s, vclass, units):
    if s is None or s.dropna().empty:
        return
    s = s.dropna()
    lvl = float(s.iloc[-1])
    dv  = np.nan
    if s.shape[0] > wk_back + 1 and not np.isnan(s.iloc[-wk_back-1]):
        dv = float(s.iloc[-1] - s.iloc[-wk_back-1])
    pr  = percentile_of_last(s)
    vol_rows.append([label, vclass, units, lvl, dv, pr])

# Realized vols in percent
if "^GSPC" in REALVOL.columns: add_vol("SPX 1M RV", REALVOL["^GSPC"] * 100, "Realized", "%")
if "HYG" in REALVOL.columns:   add_vol("HY 1M RV",  REALVOL["HYG"] * 100,   "Realized", "%")
if "CL=F" in REALVOL.columns:  add_vol("WTI 1M RV", REALVOL["CL=F"] * 100,  "Realized", "%")
fx_cols = [c for c in REALVOL.columns if c.endswith("=X")]
if "USDJPY=X" in fx_cols:      add_vol("USDJPY 1M RV", REALVOL["USDJPY=X"] * 100, "Realized", "%")
elif fx_cols:                  add_vol(f"{fx_cols[0]} 1M RV", REALVOL[fx_cols[0]] * 100, "Realized", "%")

# Implied vols in points
if "^VIX" in IV.columns:       add_vol("VIX", IV["^VIX"], "Implied", "pts")
if "^OVX" in IV.columns:       add_vol("OVX", IV["^OVX"], "Implied", "pts")

vol_tbl = pd.DataFrame(vol_rows, columns=["Series", "Class", "Units", "Level", "Δ w/w", "Percentile rank"]).dropna(how="all")

# FIX: create temp abs-change column, sort by it, then drop
if not vol_tbl.empty:
    vol_tbl["abschg"] = vol_tbl["Δ w/w"].abs()
    vol_tbl = vol_tbl.sort_values(["Class", "abschg"], ascending=[True, False], na_position="last").drop(columns=["abschg"])

st.subheader("This week vs last: correlations and vol")

if not corr_tbl.empty:
    corr_style = (
        corr_tbl.style
            .format({
                "ρ now (rolling)": "{:+.2%}",
                "Δρ w/w": "{:+.2%}",
                "Percentile rank": "{:.0f}%"
            })
            .background_gradient(subset=["Δρ w/w"], cmap="RdYlGn")
            .hide_index()
    )
    st.dataframe(
        corr_style,
        use_container_width=True,
        height=computed_height(corr_tbl.shape[0])
    )
    st.caption("ρ now is the current rolling correlation of daily returns over the selected window. Δρ w/w is change versus the prior week. Percentile rank is within the chosen history window.")

if not vol_tbl.empty:
    vol_style = (
        vol_tbl.style
            .format({
                "Level": lambda v: f"{v:.1f}",
                "Δ w/w": lambda v: "" if pd.isna(v) else f"{v:+.1f}",
                "Percentile rank": "{:.0f}%"
            })
            .background_gradient(subset=["Δ w/w"], cmap="RdYlGn")
            .hide_index()
    )
    st.dataframe(
        vol_style,
        use_container_width=True,
        height=computed_height(vol_tbl.shape[0])
    )
    st.caption("Class identifies implied vs realized volatility. Units are points for implied indices (VIX/OVX) and percent for realized 1M vol. Δ w/w is change versus the prior week; Percentile rank is within the chosen history window.")

# =========================
# Downloads
# =========================
with st.expander("Download snapshots"):
    if not corr_tbl.empty:
        st.download_button("Weekly correlation snapshot CSV", corr_tbl.to_csv(index=False).encode(), file_name="weekly_corr_snapshot.csv")
    if not vol_tbl.empty:
        st.download_button("Weekly vol snapshot CSV", vol_tbl.to_csv(index=False).encode(), file_name="weekly_vol_snapshot.csv")

st.caption("© 2025 AD Fund Management LP")
