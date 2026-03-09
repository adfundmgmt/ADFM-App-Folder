# 15_Weekly_Cross_Asset_Compass.py
# Weekly Cross-Asset Compass
# Clean rebuild focused on regime, signal ranking, and strong dynamic commentary

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Weekly Cross-Asset Compass",
    layout="wide",
)

# =========================
# Global styling
# =========================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 1.2rem;
        max-width: 1500px;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.2;
        margin-top: 0.15rem;
        margin-bottom: 0.2rem;
        padding-top: 0.1rem;
        overflow: visible;
    }

    .subtle {
        color: #6b7280;
        font-size: 0.96rem;
        margin-bottom: 1rem;
    }

    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        height: 100%;
    }

    .card-title {
        font-size: 0.80rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }

    .card-value {
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }

    .card-sub {
        font-size: 0.92rem;
        color: #4b5563;
        line-height: 1.35;
    }

    .section-label {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 0.4rem;
        margin-bottom: 0.65rem;
    }

    .commentary-box {
        background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }

    .signal-chip {
        display: inline-block;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.35rem;
    }

    .chip-green { background: #e8f7ee; color: #166534; }
    .chip-yellow { background: #fff7e6; color: #92400e; }
    .chip-red { background: #fdecec; color: #991b1b; }
    .chip-blue { background: #eaf2ff; color: #1d4ed8; }

    div[data-testid="stDataFrame"] div[role="table"] {
        border-radius: 16px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Header
# =========================
st.markdown('<div class="main-title">Weekly Cross-Asset Compass</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">A decision dashboard for regime, transmission, and hedge mix across equities, credit, rates, FX, commodities, and volatility.</div>',
    unsafe_allow_html=True,
)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Purpose: Weekly cross-asset regime summary for positioning and risk framing.

        What it covers
        • Core signals and summary outputs for this dashboard
        • Key context needed to interpret current regime or setup
        • Practical view designed for quick internal decision support

        Data source
        • Public market and macro data feeds used throughout the app
        """
    )

    st.header("Controls")
    lookback_years = st.slider("History window (years)", 3, 12, 6)
    corr_window = st.selectbox("Rolling correlation window", [21, 63], index=0)
    compare_weeks = st.selectbox("Compare against", ["1 week ago", "2 weeks ago"], index=0)
    week_back = 5 if compare_weeks == "1 week ago" else 10

    top_pairs_to_show = st.slider("Top pair signals to display", 4, 12, 8)
    show_detail_charts = st.slider("Detailed charts", 2, 8, 4)

# =========================
# Universe
# =========================
ASSETS = {
    "SPY": {"label": "SPX", "group": "Equities", "transform": "pct"},
    "IWM": {"label": "RTY", "group": "Equities", "transform": "pct"},
    "FEZ": {"label": "Europe", "group": "Equities", "transform": "pct"},
    "EWJ": {"label": "Japan", "group": "Equities", "transform": "pct"},
    "EEM": {"label": "EM", "group": "Equities", "transform": "pct"},
    "HYG": {"label": "HY", "group": "Credit", "transform": "pct"},
    "LQD": {"label": "IG", "group": "Credit", "transform": "pct"},
    "TLT": {"label": "TLT", "group": "Rates", "transform": "pct"},
    "^TNX": {"label": "US10Y", "group": "Rates", "transform": "diff"},
    "UUP": {"label": "Dollar", "group": "FX", "transform": "pct"},
    "USDJPY=X": {"label": "USDJPY", "group": "FX", "transform": "pct"},
    "GLD": {"label": "Gold", "group": "Macro", "transform": "pct"},
    "USO": {"label": "Oil", "group": "Macro", "transform": "pct"},
    "^VIX": {"label": "VIX", "group": "Vol", "transform": "level"},
    "^VIX3M": {"label": "VIX3M", "group": "Vol", "transform": "level"},
}

PAIR_SPECS = [
    {"a": "SPY", "b": "^TNX", "pair": "SPX vs 10Y"},
    {"a": "SPY", "b": "TLT", "pair": "SPX vs TLT"},
    {"a": "SPY", "b": "HYG", "pair": "SPX vs HY"},
    {"a": "SPY", "b": "LQD", "pair": "SPX vs IG"},
    {"a": "SPY", "b": "UUP", "pair": "SPX vs Dollar"},
    {"a": "SPY", "b": "USO", "pair": "SPX vs Oil"},
    {"a": "SPY", "b": "GLD", "pair": "SPX vs Gold"},
    {"a": "IWM", "b": "^TNX", "pair": "RTY vs 10Y"},
    {"a": "IWM", "b": "HYG", "pair": "RTY vs HY"},
    {"a": "EEM", "b": "UUP", "pair": "EM vs Dollar"},
    {"a": "EEM", "b": "USO", "pair": "EM vs Oil"},
    {"a": "EWJ", "b": "USDJPY=X", "pair": "Japan vs USDJPY"},
    {"a": "LQD", "b": "^TNX", "pair": "IG vs 10Y"},
    {"a": "GLD", "b": "^TNX", "pair": "Gold vs 10Y"},
    {"a": "HYG", "b": "USO", "pair": "HY vs Oil"},
]

HEATMAP_TICKERS = ["SPY", "IWM", "HYG", "LQD", "TLT", "UUP", "USDJPY=X", "GLD", "USO", "^TNX"]

ALL_TICKERS = list(ASSETS.keys())

# =========================
# Helpers
# =========================
def safe_last(series):
    s = pd.Series(series).dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan

def week_change(series, n=5):
    s = pd.Series(series).dropna()
    if len(s) <= n:
        return np.nan
    return float(s.iloc[-1] - s.iloc[-n - 1])

def week_return(series, n=5):
    s = pd.Series(series).dropna()
    if len(s) <= n:
        return np.nan
    base = s.iloc[-n - 1]
    if base == 0 or pd.isna(base):
        return np.nan
    return float((s.iloc[-1] / base - 1.0) * 100.0)

def pct_rank_last(series):
    s = pd.Series(series).dropna()
    if s.empty:
        return np.nan
    return float(s.rank(pct=True).iloc[-1] * 100.0)

def zscore_last(series, window=504):
    s = pd.Series(series).dropna()
    if len(s) > window:
        s = s.iloc[-window:]
    if s.empty:
        return np.nan
    sd = s.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return np.nan
    z = (s - s.mean()) / sd
    return float(z.iloc[-1])

def realized_vol(ret_series, window=21, annualization=252):
    return ret_series.rolling(window).std() * math.sqrt(annualization) * 100

def rolling_corr(a, b, window):
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < window:
        return pd.Series(dtype=float)
    return df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1])

def signal_strength(rho_now, rho_delta, pctile):
    level = min(abs(rho_now) / 0.80, 1.0) * 45 if pd.notna(rho_now) else 0
    delta = min(abs(rho_delta) / 0.25, 1.0) * 40 if pd.notna(rho_delta) else 0
    extreme = (abs(pctile - 50) / 50) * 15 if pd.notna(pctile) else 0
    return round(level + delta + extreme, 1)

def bucket_from_score(score):
    if score >= 80:
        return "Very strong"
    if score >= 65:
        return "Strong"
    if score >= 50:
        return "Moderate"
    return "Low"

def chip_class(label):
    if label in ["Constructive", "Supportive", "Working", "Contained"]:
        return "chip-green"
    if label in ["Mixed", "Watchful", "Moderate", "Relevant", "Balanced", "Hot", "Quiet", "Loose", "Unclear"]:
        return "chip-yellow"
    if label in ["Fragile", "Broken", "Elevated", "Tightening"]:
        return "chip-red"
    return "chip-blue"

def rolling_mean_std(series, window=126):
    s = pd.Series(series).dropna()
    return s.rolling(window).mean(), s.rolling(window).std()

# =========================
# Data fetch
# =========================
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_prices(tickers, start, end):
    try:
        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception:
        return pd.DataFrame()

    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        frames = []
        top = raw.columns.get_level_values(0)
        for t in tickers:
            if t not in top:
                continue
            df = raw[t].copy()
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            if col in df.columns:
                frames.append(df[col].rename(t))
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, axis=1)
    else:
        col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        if col not in raw.columns:
            return pd.DataFrame()
        out = raw[[col]].rename(columns={col: tickers[0]})

    out.index.name = "Date"
    return out.sort_index()

# =========================
# Load data
# =========================
end_date = datetime.now().date() + timedelta(days=1)
start_date = datetime.now().date() - timedelta(days=365 * lookback_years)

with st.spinner("Downloading market data..."):
    prices = fetch_prices(ALL_TICKERS, str(start_date), str(end_date))

if prices.empty:
    st.error("No data downloaded. Check Yahoo Finance connectivity and try again.")
    st.stop()

prices = prices.dropna(axis=1, how="all").copy()

# =========================
# Return engine
# =========================
transformed = {}
pct_returns = {}

for ticker in prices.columns:
    mode = ASSETS[ticker]["transform"]
    s = prices[ticker]
    if mode == "diff":
        transformed[ticker] = s.diff()
    elif mode == "pct":
        transformed[ticker] = s.pct_change()
        pct_returns[ticker] = s.pct_change()
    else:
        transformed[ticker] = s.copy()

transformed = pd.DataFrame(transformed).replace([np.inf, -np.inf], np.nan)
pct_returns = pd.DataFrame(pct_returns).replace([np.inf, -np.inf], np.nan)

# =========================
# Core regime features
# =========================
vix = prices["^VIX"] if "^VIX" in prices.columns else pd.Series(dtype=float)
vix3m = prices["^VIX3M"] if "^VIX3M" in prices.columns else pd.Series(dtype=float)
term_structure = ((vix3m / vix) - 1.0).dropna() if (not vix.empty and not vix3m.empty) else pd.Series(dtype=float)

spy_rv = realized_vol(pct_returns["SPY"], 21) if "SPY" in pct_returns.columns else pd.Series(dtype=float)
hyg_rv = realized_vol(pct_returns["HYG"], 21) if "HYG" in pct_returns.columns else pd.Series(dtype=float)
uso_rv = realized_vol(pct_returns["USO"], 21) if "USO" in pct_returns.columns else pd.Series(dtype=float)
uup_rv = realized_vol(pct_returns["UUP"], 21) if "UUP" in pct_returns.columns else pd.Series(dtype=float)

spy_tnx_corr = rolling_corr(transformed["SPY"], transformed["^TNX"], corr_window) if {"SPY", "^TNX"}.issubset(transformed.columns) else pd.Series(dtype=float)
spy_tlt_corr = rolling_corr(transformed["SPY"], transformed["TLT"], corr_window) if {"SPY", "TLT"}.issubset(transformed.columns) else pd.Series(dtype=float)
spy_hyg_corr = rolling_corr(transformed["SPY"], transformed["HYG"], corr_window) if {"SPY", "HYG"}.issubset(transformed.columns) else pd.Series(dtype=float)
eem_uup_corr = rolling_corr(transformed["EEM"], transformed["UUP"], corr_window) if {"EEM", "UUP"}.issubset(transformed.columns) else pd.Series(dtype=float)
gld_tnx_corr = rolling_corr(transformed["GLD"], transformed["^TNX"], corr_window) if {"GLD", "^TNX"}.issubset(transformed.columns) else pd.Series(dtype=float)

vix_rv_spread = pd.Series(dtype=float)
if not vix.empty and not spy_rv.empty:
    both = pd.concat([vix.rename("vix"), spy_rv.rename("rv")], axis=1).dropna()
    if not both.empty:
        vix_rv_spread = both["vix"] - both["rv"]

# =========================
# State classification
# =========================
vix_now = safe_last(vix)
term_now = safe_last(term_structure)
spy_rv_now = safe_last(spy_rv)
spy_rv_pctile = pct_rank_last(spy_rv)
spy_tnx_now = safe_last(spy_tnx_corr)
spy_tlt_now = safe_last(spy_tlt_corr)
spy_hyg_now = safe_last(spy_hyg_corr)
eem_uup_now = safe_last(eem_uup_corr)
gld_tnx_now = safe_last(gld_tnx_corr)
hy_vol_z = zscore_last(hyg_rv)
oil_vol_z = zscore_last(uso_rv)
dollar_vol_z = zscore_last(uup_rv)
vix_rv_z = zscore_last(vix_rv_spread)

# Vol state
vol_score = 0
if pd.notna(term_now) and term_now < 0:
    vol_score += 1
if pd.notna(vix_now) and vix_now >= 22:
    vol_score += 1
if pd.notna(spy_rv_pctile) and spy_rv_pctile >= 70:
    vol_score += 1

if vol_score >= 2:
    vol_state = "Elevated"
elif vol_score == 1:
    vol_state = "Watchful"
else:
    vol_state = "Constructive"

# Hedge state
if pd.isna(spy_tnx_now):
    hedge_state = "Unclear"
elif spy_tnx_now >= 0.30:
    hedge_state = "Broken"
elif spy_tnx_now <= -0.30 or (pd.notna(spy_tlt_now) and spy_tlt_now <= -0.30):
    hedge_state = "Working"
else:
    hedge_state = "Mixed"

# Credit state
credit_score = 0
if pd.notna(hy_vol_z) and hy_vol_z >= 1.0:
    credit_score += 1
if pd.notna(spy_hyg_now) and spy_hyg_now >= 0.50:
    credit_score += 1
if credit_score >= 2:
    credit_state = "Tightening"
elif credit_score == 1:
    credit_state = "Relevant"
else:
    credit_state = "Contained"

# Dollar state
if pd.isna(eem_uup_now):
    dollar_state = "Mixed"
elif eem_uup_now <= -0.35:
    dollar_state = "Tightening"
elif eem_uup_now >= 0.10:
    dollar_state = "Loose"
else:
    dollar_state = "Mixed"

# Commodity state
if pd.notna(oil_vol_z) and oil_vol_z >= 1.2:
    commodity_state = "Hot"
elif pd.notna(oil_vol_z) and oil_vol_z <= -1.0:
    commodity_state = "Quiet"
else:
    commodity_state = "Balanced"

# Master regime
if vol_state == "Elevated" and hedge_state == "Broken":
    regime = "Fragile"
elif vol_state == "Constructive" and hedge_state == "Working" and credit_state == "Contained":
    regime = "Supportive"
else:
    regime = "Mixed"

# =========================
# Top cards
# =========================
c1, c2, c3, c4, c5 = st.columns(5)

cards = [
    ("Regime", regime, "Overall tape quality and macro transmission"),
    ("Volatility", vol_state, f"VIX {vix_now:.1f}" if pd.notna(vix_now) else "VIX unavailable"),
    ("Hedge State", hedge_state, f"SPX vs 10Y rho {spy_tnx_now:+.2f}" if pd.notna(spy_tnx_now) else "Rates linkage unavailable"),
    ("Credit", credit_state, f"SPX vs HY rho {spy_hyg_now:+.2f}" if pd.notna(spy_hyg_now) else "Credit linkage unavailable"),
    ("Dollar", dollar_state, f"EM vs Dollar rho {eem_uup_now:+.2f}" if pd.notna(eem_uup_now) else "Dollar linkage unavailable"),
]

for col, (title, value, sub) in zip([c1, c2, c3, c4, c5], cards):
    with col:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">{title}</div>
                <div class="card-value">{value}</div>
                <div class="card-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")

# =========================
# Commentary engine
# =========================
def build_state_commentary():
    pieces = []

    if regime == "Supportive":
        pieces.append(
            "The tape still screens constructive. Volatility is not doing enough damage to force broad de-risking, credit is behaving, and the cross-asset structure is closer to carry than panic. In that state, leadership can keep extending, but the right posture is still selective rather than complacent because macro linkages can reprice faster than index levels suggest."
        )
    elif regime == "Fragile":
        pieces.append(
            "The tape screens fragile. The important point is that volatility is higher and the cross-asset plumbing is less helpful. If rates are trading like equity beta and credit is tightening at the same time, portfolios lose natural shock absorbers and gross has to earn its place."
        )
    else:
        pieces.append(
            "The tape is mixed. You do not have a clean one-factor regime here, which usually means fewer clean index-level conclusions and more emphasis on internal rotation, factor selection, and hedge efficiency. In mixed states, the market often feels fine until one linkage starts doing real transmission work."
        )

    if hedge_state == "Broken":
        pieces.append(
            "Duration is not giving you clean protection right now. When SPX and yields are positively linked, long-duration exposure behaves more like a macro expression than a hedge, so downside protection has to come from index optionality, credit hedges, or lower gross rather than habit."
        )
    elif hedge_state == "Working":
        pieces.append(
            "Duration is still doing real hedge work. That matters because it lets you carry risk in the book with more confidence, especially if your growth exposures are still sensitive to macro discount-rate swings."
        )

    if credit_state == "Tightening":
        pieces.append(
            "Credit deserves respect here. Once HY volatility rises and the equity-credit linkage strengthens, the market is telling you that financing conditions and quality spreads are starting to matter again. That usually hits lower-quality cyclicals and speculative beta before it hits the index narrative."
        )
    elif credit_state == "Contained":
        pieces.append(
            "Credit is relatively contained. That still does not make the tape safe, but it does mean the market is not yet broadcasting broader financing stress through HY, which keeps the burden of proof on equity weakness rather than assuming it spills everywhere."
        )

    if dollar_state == "Tightening":
        pieces.append(
            "The dollar channel is firm enough to matter. When EM and the dollar are trading with a strongly negative linkage, global financial conditions are tighter than headline equity resilience may imply, and that usually bleeds into commodity sensitivity, international beta, and marginal liquidity appetite."
        )

    if pd.notna(vix_rv_z):
        if vix_rv_z >= 1.0:
            pieces.append(
                "Implied volatility is rich to realized. That pushes the playbook toward spreads, overwrites, and more efficient hedges rather than paying top dollar for naked protection."
            )
        elif vix_rv_z <= -1.0:
            pieces.append(
                "Implied volatility is cheap to realized. That gives you a better entry point for optionality than the headline tape might suggest."
            )

    return " ".join(pieces)

def build_action_commentary():
    actions = []

    if regime == "Supportive":
        actions.append("Let your winners work, but keep sizing honest and avoid leverage that assumes correlations stay benign.")
    elif regime == "Fragile":
        actions.append("Run lighter gross, tighten the hedge mix, and avoid pretending index resilience solves a broken cross-asset structure.")
    else:
        actions.append("Favor relative-value expressions and internal rotation over simple index conviction.")

    if hedge_state == "Broken":
        actions.append("Reduce reliance on duration hedges and lean more on equity or credit overlays.")
    elif hedge_state == "Working":
        actions.append("Duration can still earn a seat in the hedge stack.")

    if credit_state == "Tightening":
        actions.append("Watch lower-quality beta and cyclical exposure closely.")
    if dollar_state == "Tightening":
        actions.append("Respect dollar strength as a headwind for EM and externally sensitive risk.")
    if commodity_state == "Hot":
        actions.append("Treat commodity volatility as a source of factor dispersion rather than isolated sector noise.")

    return " ".join(actions)

def build_invalidation_commentary():
    return (
        "This read should be challenged if SPX versus rates flips sign and stays there for a week, if credit volatility fades while equity-credit linkage loosens, or if VIX term structure heals enough that stress pricing stops leaking into broader asset relationships."
    )

left, right = st.columns([1.25, 0.9])

with left:
    st.markdown('<div class="section-label">Dynamic Commentary</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="commentary-box">
            <div style="font-weight:700; margin-bottom:0.5rem;">State</div>
            <div style="line-height:1.55; color:#1f2937;">{build_state_commentary()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="commentary-box">
            <div style="font-weight:700; margin-bottom:0.5rem;">Action Bias</div>
            <div style="line-height:1.55; color:#1f2937;">{build_action_commentary()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="commentary-box">
            <div style="font-weight:700; margin-bottom:0.5rem;">Invalidation</div>
            <div style="line-height:1.55; color:#1f2937;">{build_invalidation_commentary()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="section-label">Macro State Chips</div>', unsafe_allow_html=True)
    chips = [
        ("Regime", regime),
        ("Vol", vol_state),
        ("Hedge", hedge_state),
        ("Credit", credit_state),
        ("Dollar", dollar_state),
        ("Commodities", commodity_state),
    ]
    chip_html = ""
    for label, value in chips:
        chip_html += f'<div style="margin-bottom:0.55rem;"><span style="font-size:0.88rem; color:#6b7280; margin-right:0.5rem;">{label}</span><span class="signal-chip {chip_class(value)}">{value}</span></div>'
    st.markdown(f'<div class="card">{chip_html}</div>', unsafe_allow_html=True)

    score_rows = [
        ("VIX", f"{vix_now:.1f}" if pd.notna(vix_now) else "NA"),
        ("VIX3M/VIX - 1", f"{term_now:+.3f}" if pd.notna(term_now) else "NA"),
        ("SPX 1M RV", f"{spy_rv_now:.1f}%" if pd.notna(spy_rv_now) else "NA"),
        ("HY vol z", f"{hy_vol_z:+.2f}" if pd.notna(hy_vol_z) else "NA"),
        ("Oil vol z", f"{oil_vol_z:+.2f}" if pd.notna(oil_vol_z) else "NA"),
        ("VIX - RV z", f"{vix_rv_z:+.2f}" if pd.notna(vix_rv_z) else "NA"),
    ]
    score_html = "".join(
        [
            f'<div style="display:flex; justify-content:space-between; padding:0.32rem 0; border-bottom:1px solid #f0f0f0;"><span style="color:#6b7280;">{k}</span><span style="font-weight:600;">{v}</span></div>'
            for k, v in score_rows
        ]
    )
    st.markdown('<div class="section-label" style="margin-top:1rem;">Diagnostic Snapshot</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card">{score_html}</div>', unsafe_allow_html=True)

# =========================
# Pair signal table
# =========================
pair_rows = []
for spec in PAIR_SPECS:
    a, b, pair = spec["a"], spec["b"], spec["pair"]
    if a not in transformed.columns or b not in transformed.columns:
        continue
    rc = rolling_corr(transformed[a], transformed[b], corr_window).dropna()
    if len(rc) < week_back + 5:
        continue

    rho_now = safe_last(rc)
    rho_delta = week_change(rc, week_back)
    pctile = pct_rank_last(rc.tail(504))
    score = signal_strength(rho_now, rho_delta, pctile)
    bucket = bucket_from_score(score)

    if pair == "SPX vs 10Y":
        if pd.notna(rho_now) and rho_now >= 0.30:
            interp = "Rates are acting as equity beta"
            action = "Do not trust duration as primary hedge"
        elif pd.notna(rho_now) and rho_now <= -0.30:
            interp = "Duration is cushioning equity risk"
            action = "Long duration can still work as hedge"
        else:
            interp = "Rates linkage is mixed"
            action = "Treat duration as its own macro bet"
    elif pair == "SPX vs HY":
        if pd.notna(rho_now) and pd.notna(rho_delta) and rho_now >= 0.50 and rho_delta >= 0.10:
            interp = "Credit is increasingly driving equities"
            action = "Watch low-quality beta and cyclicals"
        else:
            interp = "Credit matters but is not accelerating sharply"
            action = "Use HY as confirmation signal"
    elif pair == "EM vs Dollar":
        if pd.notna(rho_now) and rho_now <= -0.35:
            interp = "Dollar pressure is tightening global conditions"
            action = "Respect EM and commodity sensitivity"
        else:
            interp = "Dollar channel is present but not dominant"
            action = "Cross-check with credit and oil"
    elif pair == "Gold vs 10Y":
        if pd.notna(rho_now) and rho_now >= 0.15:
            interp = "Gold is holding up despite firmer rates"
            action = "Policy distrust or reserve demand may matter"
        elif pd.notna(rho_now) and rho_now <= -0.35:
            interp = "Gold is following the real-rate channel"
            action = "Frame gold through rates and dollar"
        else:
            interp = "Gold-rate linkage is mixed"
            action = "Treat gold as diversification sleeve"
    elif pair == "Japan vs USDJPY":
        if pd.notna(rho_now) and rho_now >= 0.35:
            interp = "Japan is still benefiting from yen weakness"
            action = "Do not fade Japan without a clean FX view"
        else:
            interp = "FX transmission into Japan is loosening"
            action = "Japan can trade more on equity beta"
    else:
        interp = "Cross-asset linkage is moving"
        action = "Use as context, then confirm elsewhere"

    pair_rows.append(
        {
            "Pair": pair,
            "ρ now": rho_now,
            "Δρ w/w": rho_delta,
            "2Y pctile": pctile,
            "Signal score": score,
            "Bucket": bucket,
            "Interpretation": interp,
            "Action": action,
            "Series": rc,
        }
    )

pair_tbl = pd.DataFrame(pair_rows)
if not pair_tbl.empty:
    pair_tbl = pair_tbl.sort_values(["Signal score", "Δρ w/w"], ascending=[False, False], na_position="last").reset_index(drop=True)

st.markdown('<div class="section-label">Top Cross-Asset Shifts</div>', unsafe_allow_html=True)

if pair_tbl.empty:
    st.info("No pair signals available.")
else:
    top_tbl = pair_tbl.head(top_pairs_to_show).copy()
    st.dataframe(
        top_tbl.drop(columns=["Series"]).style.format(
            {
                "ρ now": "{:+.2f}",
                "Δρ w/w": "{:+.2f}",
                "2Y pctile": "{:.0f}%",
                "Signal score": "{:.1f}",
            }
        ),
        use_container_width=True,
        height=min(500, 40 * (len(top_tbl) + 1)),
    )

# =========================
# Heatmap + asset tape
# =========================
h1, h2 = st.columns([1.05, 0.95])

with h1:
    st.markdown('<div class="section-label">Current Cross-Asset Correlation Map</div>', unsafe_allow_html=True)
    hm_tickers = [t for t in HEATMAP_TICKERS if t in transformed.columns]
    hm_data = transformed[hm_tickers].dropna()
    if len(hm_data) >= corr_window:
        hm = hm_data.tail(corr_window).corr()
        labels = [ASSETS[t]["label"] for t in hm.index]

        cmap = LinearSegmentedColormap.from_list(
            "soft_rwg",
            [(0.91, 0.55, 0.55), (1.0, 1.0, 1.0), (0.55, 0.82, 0.62)],
            N=256,
        )

        fig, ax = plt.subplots(figsize=(9.2, 6.2))
        im = ax.imshow(hm.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        for i in range(len(labels)):
            for j in range(len(labels)):
                val = hm.values[i, j]
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8)

        cbar = plt.colorbar(im, ax=ax, shrink=0.82)
        cbar.set_label("Correlation")
        ax.set_title(f"{corr_window}-day rolling correlation matrix")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Not enough data for heatmap.")

with h2:
    st.markdown('<div class="section-label">Weekly Tape Snapshot</div>', unsafe_allow_html=True)

    asset_rows = []
    watch_list = ["SPY", "IWM", "HYG", "LQD", "TLT", "UUP", "GLD", "USO", "USDJPY=X"]
    for t in watch_list:
        if t not in prices.columns:
            continue
        label = ASSETS[t]["label"]
        px = prices[t]
        one_w = week_return(px, week_back)
        rv = realized_vol(pct_returns[t], 21) if t in pct_returns.columns else pd.Series(dtype=float)
        rv_now = safe_last(rv)
        rv_z = zscore_last(rv)

        asset_rows.append(
            {
                "Asset": label,
                "1W %": one_w,
                "1M RV": rv_now,
                "RV z": rv_z,
            }
        )

    asset_tbl = pd.DataFrame(asset_rows)
    if not asset_tbl.empty:
        asset_tbl = asset_tbl.sort_values("RV z", ascending=False, na_position="last")
        st.dataframe(
            asset_tbl.style.format(
                {
                    "1W %": "{:+.2f}%",
                    "1M RV": "{:.1f}%",
                    "RV z": "{:+.2f}",
                }
            ),
            use_container_width=True,
            height=min(420, 40 * (len(asset_tbl) + 1)),
        )
    else:
        st.info("No asset snapshot available.")

# =========================
# Detailed signal charts
# =========================
st.markdown('<div class="section-label">Detailed Signal Charts</div>', unsafe_allow_html=True)

def plot_corr_chart(series, title):
    rc = pd.Series(series).dropna()
    if rc.empty:
        return None

    mean6, std6 = rolling_mean_std(rc, 126)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(
        rc.index,
        (mean6 - std6).values,
        (mean6 + std6).values,
        alpha=0.15,
        label="6M band",
    )
    ax.plot(rc.index, rc.values, label=f"{corr_window}D corr", linewidth=1.8)
    ax.plot(mean6.index, mean6.values, label="6M mean", linewidth=1.4)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(0.30, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(-0.30, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0.50, color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(-0.50, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    return fig

if pair_tbl.empty:
    st.info("No charts to show.")
else:
    chart_df = pair_tbl.head(show_detail_charts).copy()
    for i in range(0, len(chart_df), 2):
        cols = st.columns(2)
        subset = chart_df.iloc[i:i + 2]
        for col, (_, row) in zip(cols, subset.iterrows()):
            with col:
                fig = plot_corr_chart(row["Series"], row["Pair"])
                if fig is not None:
                    st.pyplot(fig, use_container_width=True)
                    st.caption(f"{row['Interpretation']}. {row['Action']}.")

# =========================
# Footer
# =========================
last_date = prices.index.max()
footer_date = last_date.strftime("%Y-%m-%d") if pd.notna(last_date) else "N/A"
st.caption(f"As of {footer_date} | © 2026 AD Fund Management LP")
