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
st.set_page_config(page_title="Cross-Asset Correlations & Vol", layout="wide")

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

st.title("Cross-Asset Correlations & Volatility")
st.caption("Data: Yahoo Finance via yfinance. Insight-first, minimal ink, pastel where helpful.")

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

# Matrix spec from user
EXCLUDE_ROWS = ["EURUSD=X", "USDJPY=X", "GBPUSD=X"]  # drop bottom FX rows

# =========================
# Sidebar, slim by design
# =========================
st.sidebar.header("Controls")
rho_thresh = st.sidebar.slider("|ρ| highlight threshold", 0.30, 0.90, 0.50, 0.05)
clip_extremes = st.sidebar.checkbox("Clip chart y-axes at 1st–99th pct", value=True)
focus_since_2017 = st.sidebar.checkbox("Focus on 2017 to present", value=True)

st.sidebar.header("Playbook knobs")
pos_cut = st.sidebar.slider("Strong positive corr cutoff", 0.60, 0.95, 0.80, 0.05)
neg_cut = st.sidebar.slider("Strong negative corr cutoff", -0.95, -0.60, -0.80, 0.05)
z_hot = st.sidebar.slider("Hot vol Z threshold", 1.0, 3.0, 1.5, 0.1)
z_cold = st.sidebar.slider("Cold vol Z threshold", -3.0, -1.0, -1.5, 0.1)

# =========================
# Helpers
# =========================
def pastelize_cmap(name="RdYlGn", lighten=0.70, n=256):
    base = plt.cm.get_cmap(name, n)
    colors = base(np.linspace(0, 1, n))
    colors[:, :3] = 1.0 - lighten * (1.0 - colors[:, :3])  # toward white
    return ListedColormap(colors)

def pastel_red_white_green():
    # very soft red -> white -> green, matches: -75% red, 0 white, +75% green when used with vmin/vmax
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
# Header KPIs, quick read
# =========================
def roll_corr(ret_df, a, b, window=21):
    if a not in ret_df.columns or b not in ret_df.columns:
        return pd.Series(dtype=float)
    df = ret_df[[a, b]].dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return df[a].rolling(window).corr(df[b])

k1, k2, k3, k4 = st.columns(4)

rc_er = roll_corr(RET, "^GSPC", "^TNX", WIN)
with k1:
    st.metric("SPX vs US10Y, 21D corr",
              value=("NA" if rc_er.empty else f"{rc_er.iloc[-1]:.1%}"),
              delta=(None if rc_er.empty else f"{percentile_of_last(rc_er):.0f}th pct"))

if "^GSPC" in REALVOL.columns and "^VIX" in IV.columns:
    rv = (REALVOL["^GSPC"] * 100).dropna()
    vix = IV["^VIX"].dropna()
    common = rv.to_frame("rv").join(vix.to_frame("vix"), how="inner")
    if not common.empty:
        spread = common["vix"] - common["rv"]
        with k2:
            st.metric("VIX minus SPX realized",
                      value=f"{spread.iloc[-1]:.1f} pts",
                      delta=f"{percentile_of_last(spread):.0f}th pct")

if "HYG" in REALVOL.columns:
    hyg = (REALVOL["HYG"] * 100).dropna()
    with k3:
        st.metric("HY realized vol, now", f"{current_value(hyg):.1f}%", f"{percentile_of_last(hyg):.0f}th pct")

fx_pick = None
for fx in ["USDJPY=X", "EURUSD=X", "GBPUSD=X"]:
    if fx in REALVOL.columns:
        fx_pick = (REALVOL[fx] * 100).dropna()
        break
with k4:
    if fx_pick is not None and not fx_pick.empty:
        st.metric(f"{fx_pick.name} realized vol, now", f"{fx_pick.iloc[-1]:.1f}%", f"{percentile_of_last(fx_pick):.0f}th pct")
    else:
        st.metric("FX realized vol, now", "NA")

# =========================
# Six correlation panels
# =========================
st.subheader("Correlation panels, 21D, with 6M bands")

PANEL_SPECS = [
    dict(a="^GSPC", b="^TNX", la="SPX", lb="US 10Y",      title="Equity vs Rates"),
    dict(a="^GSPC", b="HYG",  la="SPX", lb="HY (HYG)",     title="Equity vs HY Credit"),
    dict(a="^GSPC", b="CL=F", la="SPX", lb="Oil (WTI)",    title="Equity vs Oil"),
    dict(a="^RUT",  b="LQD",  la="RTY", lb="IG (LQD)",     title="Small Caps vs IG Credit"),
    dict(a="^STOXX50E", b="EURUSD=X", la="SX5E", lb="EURUSD", title="Europe vs EURUSD"),
    dict(a="^N225", b="USDJPY=X", la="NKY", lb="USDJPY",   title="Japan vs USDJPY"),
]

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

for row_specs in [PANEL_SPECS[:3], PANEL_SPECS[3:]]:
    cols = st.columns(3)
    for spec, col in zip(row_specs, cols):
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
# Correlation Matrix — minimal, user spec
# =========================
st.subheader("Cross-Asset Correlation Matrix (1M)")

def style_matrix_black_text(df, thresh=0.5):
    # black numerals, bold above |ρ| threshold, thin border on hits
    def style_fn(x):
        css = np.full(x.shape, "color:black; text-align:center; font-size:0.95rem;", dtype=object)
        vals = x.values.astype(float)
        hit = (np.abs(vals) >= thresh) & ~np.isnan(vals)
        css[hit] = "color:black; font-weight:700; border:1px solid #888; text-align:center; font-size:0.95rem;"
        return pd.DataFrame(css, index=x.index, columns=x.columns)
    return df.style.apply(style_fn, axis=None).set_properties(**{"background-color": "white"})

# compute 1M corr
if len(RET) >= WIN:
    corr_raw = RET.tail(WIN).corr().replace([-np.inf, np.inf], np.nan)
else:
    corr_raw = pd.DataFrame()

if corr_raw.empty:
    st.warning("Not enough data to compute the 1M correlation matrix.")
else:
    order = [t for t in ORDER if t in corr_raw.columns]
    corr_signed = corr_raw.loc[order, order]

    # drop bottom FX rows per spec
    row_keep = [t for t in order if t not in EXCLUDE_ROWS]
    corr_signed = corr_signed.loc[row_keep, order]

    # lower triangle relative to our row subset, blank diagonal
    n_rows, n_cols = corr_signed.shape
    disp = corr_signed.copy().astype(float)
    for i in range(n_rows):
        for j in range(n_cols):
            if j >= i:  # keep strictly lower triangle
                disp.iat[i, j] = np.nan
    for i in range(min(n_rows, n_cols)):
        disp.iat[i, i] = np.nan

    # formatter, blanks for NaN
    def _fmt(v):
        return "" if pd.isna(v) else f"{int(round(v*100, 0))}%"

    # cell background gradient, −75% red, 0 white, +75% green
    sty = (
        disp.style
            .background_gradient(cmap=PASTEL_RWG, vmin=-0.75, vmax=0.75)
            .format(_fmt)
    )
    sty = style_matrix_black_text(disp, thresh=rho_thresh)

    # NOTE: Styler operations are not strictly composable, so re-apply gradient after text styling
    sty = (
        disp.style
            .background_gradient(cmap=PASTEL_RWG, vmin=-0.75, vmax=0.75)
            .format(_fmt)
            .apply(lambda x: np.where((np.abs(x.values.astype(float)) >= rho_thresh) & ~np.isnan(x.values),
                                      "color:black; font-weight:700; border:1px solid #888; text-align:center; font-size:0.95rem;",
                                      "color:black; text-align:center; font-size:0.95rem;"),
                   axis=None)
            .set_properties(**{"background-color": "white"})
    )

    st.dataframe(sty, use_container_width=True, height=min(110 + 40*disp.shape[0], 1200))
    st.caption(
        f"Lower triangle only. FX rows removed. Blank cells are intentional. "
        f"Bold marks |ρ| ≥ {rho_thresh:.2f}. Cell color shows sign and strength, 0% white, −75% red, +75% green."
    )

# =========================
# Vol monitor, realized vs implied
# =========================
st.subheader("Volatility monitor")

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
        return None, None
    ax.plot(s.index, s.values, label=label)
    if overlay is not None and not overlay.dropna().empty:
        o = overlay.loc[apply_focus_window(overlay.index)]
        ax.plot(o.index, o.values, label=overlay.name)
    iqr_band(ax, s)
    lims = clip_limits(s.values, 1, 99)
    if lims:
        ax.set_ylim(lims)
    ax.set_title(title); ax.set_ylabel("Percent"); ax.margins(y=0.05); add_regime_shading(ax)
    ax.legend(fontsize=8, ncol=2); plt.tight_layout(); st.pyplot(fig, use_container_width=True)
    return current_value(s), percentile_of_last(s)

c1, c2, c3 = st.columns(3)
with c1:
    vix = IV["^VIX"] if "^VIX" in IV.columns else None
    if vix is not None: vix.name = "VIX"
    if "^GSPC" in REALVOL.columns:
        v_now, v_pct = vol_panel(REALVOL["^GSPC"] * 100, "SPX 1M Realized", "Equity vol, implied vs realized", overlay=vix)
        if v_now is not None:
            m1, m2 = st.columns(2)
            with m1: st.metric("SPX realized, now", f"{v_now:.1f}%")
            with m2: st.metric("Percentile", f"{v_pct:.0f}th")
with c2:
    ovx = IV["^OVX"] if "^OVX" in IV.columns else None
    if ovx is not None: ovx.name = "OVX"
    if "CL=F" in REALVOL.columns:
        o_now, o_pct = vol_panel(REALVOL["CL=F"] * 100, "WTI 1M Realized", "Oil vol, implied vs realized", overlay=ovx)
        if o_now is not None:
            m1, m2 = st.columns(2)
            with m1: st.metric("Oil realized, now", f"{o_now:.1f}%")
            with m2: st.metric("Percentile", f"{o_pct:.0f}th")
with c3:
    fx_cols = [c for c in REALVOL.columns if c.endswith("=X")]
    fx_name = None
    if fx_cols:
        fx_name = "USDJPY=X" if "USDJPY=X" in REALVOL.columns else fx_cols[0]
        f_now, f_pct = vol_panel(REALVOL[fx_name] * 100, f"{fx_name} 1M Realized", "FX vol, realized proxy")
        if f_now is not None:
            m1, m2 = st.columns(2)
            with m1: st.metric("FX realized, now", f"{f_now:.1f}%")
            with m2: st.metric("Percentile", f"{f_pct:.0f}th")

# =========================
# Standardized vol snapshot, ranked table
# =========================
st.subheader("Vol snapshot, standardized, ranked")

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

# =========================
# Playbook, auto generated
# =========================
st.subheader("Playbook, auto generated")

bullets = []

def matrix_extremes(ret_df, tickers, window=21, pos_cut=0.8, neg_cut=-0.8):
    if len(ret_df) < window:
        return [], []
    corr = ret_df.tail(window).corr().replace([-np.inf, np.inf], np.nan)
    order = [t for t in tickers if t in corr.columns]
    pos, neg = [], []
    for i in range(1, len(order)):
        for j in range(0, i):
            a, b = order[i], order[j]
            val = corr.loc[a, b]
            if pd.isna(val):
                continue
            if val >= pos_cut:
                pos.append((a, b, float(val)))
            if val <= neg_cut:
                neg.append((a, b, float(val)))
    pos = sorted(pos, key=lambda x: -x[2])[:6]
    neg = sorted(neg, key=lambda x: x[2])[:6]
    return pos, neg

pos_ext, neg_ext = matrix_extremes(RET, ORDER, WIN, pos_cut=pos_cut, neg_cut=neg_cut)
if pos_ext:
    bullets.append("Strong positives, consider relative value rather than directional hedge:")
    for a, b, v in pos_ext:
        bullets.append(f"• {a} with {b}: {v:.2f}")
if neg_ext:
    bullets.append("Strong negatives, hedge candidates and macro pivots:")
    for a, b, v in neg_ext:
        bullets.append(f"• {a} with {b}: {v:.2f}")

if not rc_er.empty:
    if rc_er.iloc[-1] <= -0.5:
        bullets.append("• Equities vs rates negative, rallies in duration tend to support equities.")
    elif rc_er.iloc[-1] >= 0.5:
        bullets.append("• Equities vs rates positive, rate selloffs are pressuring equities.")

rc_ehy = roll_corr(RET, "^GSPC", "HYG", WIN)
if not rc_ehy.empty and rc_ehy.iloc[-1] >= 0.5:
    bullets.append("• Equities vs HY positive, equity drawdowns likely travel with credit widening.")

def last_z(name, series_map):
    s = series_map.get(name)
    if s is None:
        return None
    z = zscores(s.dropna())
    return float(z.iloc[-1]) if not z.empty else None

series_map = {}
if "^VIX" in IV.columns: series_map["VIX"] = IV["^VIX"]
if "^OVX" in IV.columns: series_map["OVX"] = IV["^OVX"]
if "^GSPC" in REALVOL.columns: series_map["SPX RV"] = REALVOL["^GSPC"] * 100
if "HYG" in REALVOL.columns: series_map["HY RV"] = REALVOL["HYG"] * 100
if "CL=F" in REALVOL.columns: series_map["Oil RV"] = REALVOL["CL=F"] * 100

vix_z = last_z("VIX", series_map)
spxrv_z = last_z("SPX RV", series_map)
hyrv_z = last_z("HY RV", series_map)
oilrv_z = last_z("Oil RV", series_map)

if vix_z is not None and spxrv_z is not None and vix_z >= z_hot and spxrv_z < vix_z:
    bullets.append("• Implied equity vol rich to realized, consider short vol structures or overwrite.")
if vix_z is not None and vix_z <= z_cold:
    bullets.append("• Implied equity vol depressed, consider buying convexity or collars.")

if hyrv_z is not None and hyrv_z >= z_hot:
    bullets.append("• HY vol hot, watch for spillover into equities, reduce gross or add credit hedges.")

if oilrv_z is not None and oilrv_z >= z_hot:
    bullets.append("• Oil vol elevated, check energy beta in book and rebalance exposures.")

if fx_pick is not None and not fx_pick.empty:
    fx_pct = percentile_of_last(fx_pick)
    if fx_pct >= 80:
        bullets.append("• FX vol in higher regime, cross-asset correlations can rotate faster, tighten stops.")
    elif fx_pct <= 20:
        bullets.append("• FX vol subdued, carry and pair trades tend to persist longer.")

if bullets:
    st.markdown("\n".join(bullets))
else:
    st.info("No strong signals based on current thresholds.")

st.caption("© 2025 AD Fund Management LP")

# =========================
# Footer notes
# =========================
st.caption("Notes: 21 trading days used for 1M windows, realized vol is 21D standard deviation annualized. Rolling correlations are 21D. Implied overlays use VIX and OVX where available.")
