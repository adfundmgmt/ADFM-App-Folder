# cross_asset_weekly_compass.py
# Weekly decision-grade view: cross-asset rolling correlations, regime, and key weekly shifts.

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
st.caption("Data: Yahoo Finance via yfinance. Focus: correlations, regime, and weekly shifts you can act on.")

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

# Curated non-redundant rolling-correlation universe (21 pairs)
PAIR_SPECS = [
    # SPX vs macro sleeves
    dict(a="^GSPC", b="^TNX",      la="SPX", lb="US10Y",  title="SPX vs 10Y rates"),
    dict(a="^GSPC", b="^TYX",      la="SPX", lb="US30Y",  title="SPX vs 30Y rates"),
    dict(a="^GSPC", b="HYG",       la="SPX", lb="HY",     title="SPX vs high yield"),
    dict(a="^GSPC", b="LQD",       la="SPX", lb="IG",     title="SPX vs IG credit"),
    dict(a="^GSPC", b="CL=F",      la="SPX", lb="WTI",    title="SPX vs oil"),
    dict(a="^GSPC", b="GLD",       la="SPX", lb="Gold",   title="SPX vs gold"),
    dict(a="^GSPC", b="HG=F",      la="SPX", lb="Copper", title="SPX vs copper"),
    dict(a="^GSPC", b="EURUSD=X",  la="SPX", lb="EURUSD", title="SPX vs EURUSD"),
    dict(a="^GSPC", b="USDJPY=X",  la="SPX", lb="USDJPY", title="SPX vs USDJPY"),

    # Small caps and EM vs macro
    dict(a="^RUT",  b="^TNX",      la="RTY", lb="US10Y",  title="Small caps vs 10Y"),
    dict(a="^RUT",  b="HYG",       la="RTY", lb="HY",     title="Small caps vs HY"),
    dict(a="^RUT",  b="CL=F",      la="RTY", lb="WTI",    title="Small caps vs oil"),

    dict(a="EEM",   b="HYG",       la="EM",  lb="HY",     title="EM vs high yield"),
    dict(a="EEM",   b="USDJPY=X",  la="EM",  lb="USDJPY", title="EM vs USDJPY"),
    dict(a="EEM",   b="CL=F",      la="EM",  lb="WTI",    title="EM vs oil"),

    # Regional indices vs FX
    dict(a="^STOXX50E", b="EURUSD=X", la="SX5E", lb="EURUSD", title="Europe vs EURUSD"),
    dict(a="^N225",     b="USDJPY=X", la="NKY",  lb="USDJPY", title="Japan vs USDJPY"),

    # Credit, commodities, and rates links
    dict(a="HYG",   b="CL=F",      la="HY",  lb="WTI",    title="HY credit vs oil"),
    dict(a="LQD",   b="^TNX",      la="IG",  lb="US10Y",  title="IG credit vs 10Y"),
    dict(a="GLD",   b="^TNX",      la="Gold",lb="US10Y",  title="Gold vs 10Y"),
    dict(a="CL=F",  b="USDJPY=X",  la="WTI", lb="USDJPY", title="Oil vs USDJPY"),
]

# =========================
# Sidebar
# =========================
st.sidebar.header("About this tool")
st.sidebar.markdown(
    """
**Objective**  
Show how equities, credit, rates, commodities, and FX are trading against each other on rolling windows, and translate that into a regime read and weekly shifts you can act on.

**Design**  
- Daily closes from Yahoo Finance  
- 1M or 3M rolling correlations of daily returns  
- Regime built from VIX term structure, VIX level, SPX realized vol, and SPX vs 10Y correlation  
- Up to 21 curated rolling-correlation panels across the main macro linkages
"""
)

st.sidebar.header("Controls")
lookback_years = st.sidebar.slider("History window (years)", 5, 15, 10)
win = st.sidebar.selectbox("Rolling window", [21, 63], index=0)  # 1M or ~3M
wk_delta = st.sidebar.selectbox("Compare vs", ["1 week ago", "2 weeks ago"], index=0)
wk_back = 5 if wk_delta == "1 week ago" else 10  # business days
rho_alert = st.sidebar.slider("|ρ| alert threshold", 0.30, 0.90, 0.50, 0.05)
z_hot = st.sidebar.slider("Hot vol Z threshold", 1.0, 3.0, 1.5, 0.1)
z_cold = st.sidebar.slider("Cold vol Z threshold", -3.0, -1.0, -1.5, 0.1)
clip_extremes = st.sidebar.checkbox("Clip chart y-axes at 1st to 99th pct", value=True)

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

def hide_index_compat(styler: pd.io.formats.style.Styler) -> pd.io.formats.style.Styler:
    if hasattr(styler, "hide"):
        try:
            return styler.hide(axis="index")
        except Exception:
            pass
    if hasattr(styler, "hide_index"):
        try:
            return styler.hide_index()
        except Exception:
            pass
    return styler

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
# Conclusions and actions (dynamic narrative)
# =========================
st.subheader("Conclusions and actions")

corr_er = roll_corr(RET, "^GSPC", "^TNX", win)
rho = float(corr_er.dropna().iloc[-1]) if not corr_er.dropna().empty else np.nan

vix_rv_spread = None
vix_rv_z = None
if "^GSPC" in REALVOL.columns and "^VIX" in IV.columns:
    rv_series = (REALVOL["^GSPC"] * 100).dropna()
    vix_series = IV["^VIX"].dropna()
    both = rv_series.to_frame("rv").join(vix_series.to_frame("vix"), how="inner")
    if not both.empty:
        spread = both["vix"] - both["rv"]
        z = zscores(spread)
        if not spread.empty:
            vix_rv_spread = float(spread.iloc[-1])
        if not z.empty:
            vix_rv_z = float(z.iloc[-1])

ts_series = vix_term_structure(IV)
ts_now = float(ts_series.iloc[-1]) if not ts_series.dropna().empty else np.nan

hy_z = None
if "HYG" in REALVOL.columns:
    hz = zscores((REALVOL["HYG"] * 100).dropna())
    if not hz.empty:
        hy_z = float(hz.iloc[-1])

oil_z = None
if "CL=F" in REALVOL.columns:
    oz = zscores((REALVOL["CL=F"] * 100).dropna())
    if not oz.empty:
        oil_z = float(oz.iloc[-1])

lines = []

# Regime paragraph
if not np.isnan(reg["vix_now"]) and not np.isnan(reg["rv_now"]):
    lines.append(
        f"Regime screens {reg['regime'].lower()}. VIX around {reg['vix_now']:.1f} with 1M SPX realized volatility near "
        f"{reg['rv_now']:.1f}% and VIX3M/VIX − 1 at {reg['ts_now']:+.3f} tells you this is a {('stressy' if reg['regime']=='Risk-off' else 'workable')} tape rather than a full-vol panic."
    )

# Equity vs rates
if not np.isnan(rho):
    if rho >= 0.3:
        lines.append(
            f"Equity and rates are trading with a positive correlation of about {rho:.0%}, "
            "so moves in yields behave like a levered SPX trade and pure duration hedges will not neutralize the AI basket by themselves."
        )
    elif rho <= -0.3:
        lines.append(
            f"Equity vs rates correlation is meaningfully negative at roughly {rho:.0%}, "
            "so duration rallies cushion drawdowns and long TLT or 10Y futures work as clean hedges against the AI and small cap risk."
        )
    else:
        lines.append(
            f"Equity vs rates correlation is close to flat at about {rho:.0%}, "
            "so you should treat duration as a separate macro bet rather than a reliable hedge or amplifier."
        )

# VIX vs realized
if vix_rv_spread is not None and vix_rv_z is not None:
    if vix_rv_z >= 1.0:
        lines.append(
            f"VIX is rich to realized by about {vix_rv_spread:.1f} points which is roughly a {vix_rv_z:.1f} standard deviation premium; "
            "that favours overwrites and short-dated call spreads over outright de-grossing if you still like the AI and power names."
        )
    elif vix_rv_z <= -1.0:
        lines.append(
            f"VIX is cheap to realized by about {vix_rv_spread:.1f} points at roughly {vix_rv_z:.1f} standard deviations below its mean; "
            "in this state collars and long optionality are a better hedge than just leaning on index futures."
        )

# HY and oil vol
if hy_z is not None:
    if hy_z >= z_hot:
        lines.append(
            f"High yield volatility screens hot with a Z score around {hy_z:.1f}, which means credit is doing real work and HYG puts or IG vs HY pairs "
            "are credible overlay hedges if your equities are tracking credit beta."
        )
    elif hy_z <= z_cold:
        lines.append(
            f"High yield volatility is unusually calm with a Z score near {hy_z:.1f}, so you can harvest carry in credit legs but should not rely on HY as a crash hedge."
        )

if oil_z is not None:
    if oil_z >= z_hot:
        lines.append(
            f"Oil volatility is elevated with a Z score near {oil_z:.1f}; energy shocks can leak into factor dispersion and you should size WTI-linked trades accordingly."
        )
    elif oil_z <= z_cold:
        lines.append(
            f"Oil volatility is suppressed with a Z score around {oil_z:.1f}, which keeps energy from dominating the macro tape and gives more room for AI and growth leadership."
        )

# Action bias and invalidations
action_bits = []
if reg["regime"] == "Risk-off":
    action_bits.append("Keep gross lighter, skew toward defensives and avoid chasing upside breakouts in the AI basket until stress eases.")
elif reg["regime"] == "Cautious":
    action_bits.append("Run moderate nets, lean on pair trades and baskets, and avoid letting single-name tails dominate portfolio risk.")
else:
    action_bits.append("Let carry work in the AI and power legs but keep position size honest and avoid leverage that assumes vol will stay pinned.")

if vix_rv_z is not None and vix_rv_z >= 1.0:
    action_bits.append("Express a chunk of downside protection through overwrites and call spreads rather than outright long vol.")
elif vix_rv_z is not None and vix_rv_z <= -1.0:
    action_bits.append("Express downside and regime change risk through index or basket optionality while implied vol is discounted.")

if not np.isnan(rho) and rho >= 0.3:
    action_bits.append("Treat duration shorts as risk-on expressions and do not assume TLT or 10Y futures will hedge a selloff in your AI and growth names.")

invalid_bits = [
    "This stance flips if SPX versus 10Y correlation holds the opposite sign for a full week, "
    "if VIX term structure moves sharply into contango with realized vol collapsing, "
    "or if the VIX minus realized spread mean reverts by more than one standard deviation in the other direction."
]

if lines:
    st.markdown(" ".join(lines))
if action_bits:
    st.markdown("**Action bias**: " + " ".join(action_bits))
st.markdown("**Invalidation**: " + " ".join(invalid_bits))

# =========================
# Rolling correlation panels
# =========================
st.subheader(f"Rolling correlations ({win}-day)")

def corr_with_context(ax, rc):
    rc = rc.dropna()
    if rc.empty:
        return
    mean_rc = rc.rolling(126).mean()
    std_rc  = rc.rolling(126).std()
    ax.fill_between(rc.index, (mean_rc - std_rc).values, (mean_rc + std_rc).values, alpha=0.15)
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

max_charts = 30
specs_to_plot = PAIR_SPECS[:max_charts]

for i in range(0, len(specs_to_plot), 3):
    row_specs = specs_to_plot[i:i+3]
    cols = st.columns(len(row_specs))
    for spec, col in zip(row_specs, cols):
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
# Key weekly shifts (compact and actionable)
# =========================
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
    rows.append([spec["Pair"] if "Pair" in spec else spec["title"], spec["a"], spec["b"], curr, delta, pct])

corr_tbl = pd.DataFrame(rows, columns=[
    "Pair", "Series A", "Series B", "ρ now (rolling)", "Δρ w/w", "Percentile rank"
]).dropna(how="all")
if not corr_tbl.empty:
    corr_tbl = corr_tbl.sort_values("Δρ w/w", key=lambda s: s.abs(), ascending=False)

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

if "^GSPC" in REALVOL.columns: add_vol("SPX 1M RV", REALVOL["^GSPC"] * 100, "Realized", "%")
if "HYG" in REALVOL.columns:   add_vol("HY 1M RV",  REALVOL["HYG"] * 100,   "Realized", "%")
if "CL=F" in REALVOL.columns:  add_vol("WTI 1M RV", REALVOL["CL=F"] * 100,  "Realized", "%")
fx_cols = [c for c in REALVOL.columns if c.endswith("=X")]
if "USDJPY=X" in fx_cols:      add_vol("USDJPY 1M RV", REALVOL["USDJPY=X"] * 100, "Realized", "%")
elif fx_cols:                  add_vol(f"{fx_cols[0]} 1M RV", REALVOL[fx_cols[0]] * 100, "Realized", "%")
if "^VIX" in IV.columns:       add_vol("VIX", IV["^VIX"], "Implied", "pts")
if "^OVX" in IV.columns:       add_vol("OVX", IV["^OVX"], "Implied", "pts")

vol_tbl = pd.DataFrame(vol_rows, columns=["Series", "Class", "Units", "Level", "Δ w/w", "Percentile rank"]).dropna(how="all")
if not vol_tbl.empty:
    vol_tbl["abschg"] = vol_tbl["Δ w/w"].abs()
    vol_tbl = vol_tbl.sort_values(["Class", "abschg"], ascending=[True, False], na_position="last").drop(columns=["abschg"])

st.subheader("Key weekly shifts in correlations and volatility")

summary_lines = []

if not corr_tbl.empty:
    top_corr = corr_tbl.head(3)
    corr_bits = []
    for _, r in top_corr.iterrows():
        corr_bits.append(
            f"{r['Pair']} now at about {r['ρ now (rolling)']:.0%} with a week-on-week change of {r['Δρ w/w']:+.0%} "
            f"around the {r['Percentile rank']:.0f}th percentile for this history window."
        )
    summary_lines.append("Largest correlation moves: " + " ".join(corr_bits))

if not vol_tbl.empty:
    top_vol = vol_tbl.head(3)
    vol_bits = []
    for _, r in top_vol.iterrows():
        delta_str = "" if pd.isna(r["Δ w/w"]) else f"{r['Δ w/w']:+.1f}{r['Units']}"
        vol_bits.append(
            f"{r['Series']} at {r['Level']:.1f}{r['Units']} ({delta_str} on the week) around the "
            f"{r['Percentile rank']:.0f}th percentile."
        )
    summary_lines.append("Largest volatility shifts: " + " ".join(vol_bits))

if summary_lines:
    st.markdown(" ".join(summary_lines))

if not corr_tbl.empty:
    corr_style = (
        corr_tbl.head(5).style
            .format({"ρ now (rolling)": "{:+.2%}", "Δρ w/w": "{:+.2%}", "Percentile rank": "{:.0f}%"})
    )
    corr_style = hide_index_compat(corr_style)
    st.dataframe(corr_style, use_container_width=True, height=computed_height(min(5, corr_tbl.shape[0])))

if not vol_tbl.empty:
    vol_style = (
        vol_tbl.head(5).style
            .format({"Level": "{:.1f}", "Δ w/w": lambda v: "" if pd.isna(v) else f"{v:+.1f}", "Percentile rank": "{:.0f}%"})
    )
    vol_style = hide_index_compat(vol_style)
    st.dataframe(vol_style, use_container_width=True, height=computed_height(min(5, vol_tbl.shape[0])))

st.caption("© 2025 AD Fund Management LP")
