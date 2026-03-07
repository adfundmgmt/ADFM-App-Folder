# cross_asset_weekly_compass_v2.py
# Weekly Cross-Asset Compass
# Decision-focused cross-asset signal engine for macro and overlay work

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import streamlit as st

# =========================
# Page config
# =========================
st.set_page_config(page_title="Weekly Cross-Asset Compass", layout="wide")
st.title("Weekly Cross-Asset Compass")
st.caption("Data: Yahoo Finance via yfinance | Focus: regime state, ranked cross-asset signals, and hedge/action mapping")

# =========================
# Style
# =========================
plt.rcParams.update({
    "figure.figsize": (8, 3.5),
    "axes.grid": True,
    "grid.alpha": 0.20,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 2.0,
})

# =========================
# Universe
# =========================
ASSET_META = {
    "SPY":       {"label": "SPX",      "group": "Equities", "transform": "pct"},
    "IWM":       {"label": "RTY",      "group": "Equities", "transform": "pct"},
    "FEZ":       {"label": "Europe",   "group": "Equities", "transform": "pct"},
    "EWJ":       {"label": "Japan",    "group": "Equities", "transform": "pct"},
    "EEM":       {"label": "EM",       "group": "Equities", "transform": "pct"},
    "HYG":       {"label": "HY",       "group": "Credit",   "transform": "pct"},
    "LQD":       {"label": "IG",       "group": "Credit",   "transform": "pct"},
    "TLT":       {"label": "TLT",      "group": "Rates",    "transform": "pct"},
    "^TNX":      {"label": "US10Y",    "group": "Rates",    "transform": "diff"},
    "^TYX":      {"label": "US30Y",    "group": "Rates",    "transform": "diff"},
    "GLD":       {"label": "Gold",     "group": "Macro",    "transform": "pct"},
    "USO":       {"label": "Oil",      "group": "Macro",    "transform": "pct"},
    "UUP":       {"label": "Dollar",   "group": "FX",       "transform": "pct"},
    "USDJPY=X":  {"label": "USDJPY",   "group": "FX",       "transform": "pct"},
    "^VIX":      {"label": "VIX",      "group": "Vol",      "transform": "diff"},
    "^VIX3M":    {"label": "VIX3M",    "group": "Vol",      "transform": "diff"},
}

CORE_HEATMAP = ["SPY", "IWM", "HYG", "LQD", "TLT", "GLD", "USO", "UUP", "USDJPY=X", "^TNX"]

PAIR_SPECS = [
    {"a": "SPY",      "b": "^TNX",     "title": "SPX vs 10Y"},
    {"a": "SPY",      "b": "TLT",      "title": "SPX vs TLT"},
    {"a": "SPY",      "b": "HYG",      "title": "SPX vs HY"},
    {"a": "SPY",      "b": "LQD",      "title": "SPX vs IG"},
    {"a": "SPY",      "b": "USO",      "title": "SPX vs oil"},
    {"a": "SPY",      "b": "GLD",      "title": "SPX vs gold"},
    {"a": "IWM",      "b": "^TNX",     "title": "Small caps vs 10Y"},
    {"a": "IWM",      "b": "HYG",      "title": "Small caps vs HY"},
    {"a": "EEM",      "b": "USO",      "title": "EM vs oil"},
    {"a": "EEM",      "b": "UUP",      "title": "EM vs dollar"},
    {"a": "FEZ",      "b": "UUP",      "title": "Europe vs dollar"},
    {"a": "EWJ",      "b": "USDJPY=X", "title": "Japan vs USDJPY"},
    {"a": "HYG",      "b": "USO",      "title": "HY vs oil"},
    {"a": "LQD",      "b": "^TNX",     "title": "IG vs 10Y"},
    {"a": "GLD",      "b": "^TNX",     "title": "Gold vs 10Y"},
    {"a": "USO",      "b": "USDJPY=X", "title": "Oil vs USDJPY"},
]

ALL_TICKERS = list(dict.fromkeys(list(ASSET_META.keys())))

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("About this tool")
    st.markdown(
        """
This version is built to answer three questions fast:

- What regime are we in right now?
- What changed materially this week across the key cross-asset linkages?
- What does that imply for gross, hedge mix, and internal rotation?

Design choices:
- One shared feature engine for regime, signals, tables, and charts
- Ranked signals instead of a wall of equally weighted correlation panels
- Mixed transformations that treat rates as changes and risk assets as returns
- Optional detailed charts only for the highest-conviction shifts
        """
    )

    st.header("Controls")
    lookback_years = st.slider("History window (years)", 3, 15, 7)
    corr_window = st.selectbox("Rolling correlation window", [21, 63], index=0)
    compare_weeks = st.selectbox("Compare vs", ["1 week ago", "2 weeks ago"], index=0)
    week_back = 5 if compare_weeks == "1 week ago" else 10

    abs_rho_threshold = st.slider("Absolute correlation threshold", 0.20, 0.90, 0.35, 0.05)
    delta_threshold = st.slider("Weekly correlation shift threshold", 0.05, 0.50, 0.12, 0.01)
    vol_z_hot = st.slider("Hot vol z-score", 0.5, 3.0, 1.5, 0.1)
    vol_z_cold = st.slider("Cold vol z-score", -3.0, -0.5, -1.5, 0.1)
    show_chart_count = st.slider("Detailed charts to show", 3, 10, 6)

# =========================
# Helpers
# =========================
def safe_last(series):
    s = pd.Series(series).dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan

def pct_rank_last(series):
    s = pd.Series(series).dropna()
    if s.empty:
        return np.nan
    return float(s.rank(pct=True).iloc[-1] * 100.0)

def zscore_last(series, window=None):
    s = pd.Series(series).dropna()
    if window is not None and len(s) > window:
        s = s.iloc[-window:]
    if s.empty or s.std(ddof=0) == 0 or np.isnan(s.std(ddof=0)):
        return np.nan
    z = (s - s.mean()) / s.std(ddof=0)
    return float(z.iloc[-1])

def week_delta(series, n=5):
    s = pd.Series(series).dropna()
    if len(s) <= n:
        return np.nan
    return float(s.iloc[-1] - s.iloc[-n-1])

def annualized_realized_vol(ret_series, window=21, annualization=252):
    return ret_series.rolling(window).std() * math.sqrt(annualization)

def signal_bucket(score):
    if score >= 80:
        return "Very strong"
    if score >= 65:
        return "Strong"
    if score >= 50:
        return "Moderate"
    return "Low"

def classify_vol_state(vix_now, ts_now, rv_pctile):
    score = 0
    if not np.isnan(ts_now) and ts_now < 0:
        score += 1
    if not np.isnan(vix_now) and vix_now >= 25:
        score += 1
    if not np.isnan(rv_pctile) and rv_pctile >= 70:
        score += 1

    if score >= 2:
        return "Elevated"
    if score == 1:
        return "Watchful"
    return "Calm"

def classify_hedge_state(spx_rate_rho):
    if np.isnan(spx_rate_rho):
        return "Unclear"
    if spx_rate_rho >= 0.30:
        return "Duration hedge broken"
    if spx_rate_rho <= -0.30:
        return "Duration hedge working"
    return "Duration hedge mixed"

def classify_credit_state(hy_vol_z, spy_hyg_rho):
    score = 0
    if not np.isnan(hy_vol_z) and hy_vol_z >= 1.0:
        score += 1
    if not np.isnan(spy_hyg_rho) and spy_hyg_rho >= 0.50:
        score += 1

    if score >= 2:
        return "Credit tightly driving equities"
    if score == 1:
        return "Credit relevant"
    return "Credit contained"

def state_confidence(valid_count, total_count):
    if total_count == 0:
        return "Low"
    ratio = valid_count / total_count
    if ratio >= 0.90:
        return "High"
    if ratio >= 0.70:
        return "Medium"
    return "Low"

def explain_pair_signal(pair_row):
    title = pair_row["Pair"]
    rho = pair_row["rho_now"]
    delta = pair_row["rho_delta_w"]
    bucket = str(pair_row.get("Signal bucket", "Low")).lower()

    if title == "SPX vs 10Y":
        if rho >= 0.30:
            return (
                "Rates are behaving like equity beta rather than ballast.",
                "Do not lean on duration as the primary hedge. Prefer index optionality or credit hedges.",
            )
        if rho <= -0.30:
            return (
                "Duration is cushioning equity drawdowns again.",
                "Long duration can do real hedge work against equity risk.",
            )
        return (
            "The equity-rates link is weak or transitional.",
            "Treat duration as its own macro bet, not a dependable hedge.",
        )

    if title == "SPX vs TLT":
        if rho <= -0.30:
            return (
                "Bond beta is offsetting equity risk.",
                "TLT works as a cleaner hedge when growth scares or disinflation dominate.",
            )
        if rho >= 0.30:
            return (
                "Stocks and long bonds are moving together.",
                "Reduce faith in long-duration hedges and diversify the overlay mix.",
            )
        return (
            "The SPX-TLT relationship is muddy.",
            "Keep hedge mix diversified across vol, credit, and duration.",
        )

    if title in ["SPX vs HY", "Small caps vs HY"]:
        if rho >= 0.50 and delta >= 0.10:
            return (
                "Credit is increasingly driving equity tape quality.",
                "Watch low-quality cyclicals and use HY stress as an early warning for de-grossing.",
            )
        return (
            "Credit remains relevant but is not accelerating sharply.",
            "Keep HY on the dashboard as confirmation rather than the sole trigger.",
        )

    if title == "IG vs 10Y":
        if rho <= -0.40:
            return (
                "Rate sensitivity is dominating IG behavior.",
                "IG is behaving like duration. Hedge or size it accordingly.",
            )
        return (
            "IG is less purely a rates trade here.",
            "Spread behavior and carry matter more than just Treasury direction.",
        )

    if title == "Gold vs 10Y":
        if rho >= 0.20:
            return (
                "Gold is holding up despite firmer rates.",
                "That usually points to policy distrust, stress demand, or reserve-diversification support.",
            )
        if rho <= -0.40:
            return (
                "Gold is trading in the classic real-rate channel.",
                "Use rates and dollar views to frame gold rather than treating it as a standalone call.",
            )
        return (
            "Gold-rate linkage is mixed.",
            "Keep gold framed as a portfolio diversifier, not a single-factor rate trade.",
        )

    if title in ["EM vs dollar", "Europe vs dollar"]:
        if rho <= -0.40:
            return (
                "The dollar is tightening global financial conditions again.",
                "Respect dollar strength as a headwind for EM and externally sensitive risk.",
            )
        return (
            "Dollar pressure is present but not dominant.",
            "Cross-check with commodity strength and credit before cutting international beta.",
        )

    if title == "Japan vs USDJPY":
        if rho >= 0.40:
            return (
                "Japanese equities are still leveraging yen weakness.",
                "Be careful fading Japan without a view that USDJPY will actually reverse.",
            )
        return (
            "The yen-equity transmission is loosening.",
            "Japan can trade more on domestic or global equity beta than purely on FX.",
        )

    if title in ["SPX vs oil", "HY vs oil", "EM vs oil"]:
        if delta >= 0.10 and rho >= 0.30:
            return (
                "Energy is feeding through more directly into risk assets.",
                "Oil is no longer just sector noise. Watch inflation impulse and factor dispersion.",
            )
        return (
            "Oil is moving, but transmission into broad risk is not yet dominant.",
            "Avoid overreacting to commodity headlines unless the linkage keeps strengthening.",
        )

    return (
        f"Cross-asset linkage is moving with {bucket} conviction.",
        "Use it as a portfolio context signal and confirm it with price action and positioning.",
    )

def action_bias_from_states(vol_state, hedge_state, credit_state, vix_rv_z):
    bits = []

    if vol_state == "Elevated":
        bits.append("Keep gross more selective and avoid assuming correlation relief will appear on its own.")
    elif vol_state == "Watchful":
        bits.append("Run moderate nets and prefer baskets or paired exposures over isolated single-name heroics.")
    else:
        bits.append("Carry can keep working, but size positions as if vol can reprice quickly.")

    if hedge_state == "Duration hedge broken":
        bits.append("Do not assume long-duration exposure offsets an equity drawdown. Use index vol or credit overlays.")
    elif hedge_state == "Duration hedge working":
        bits.append("Duration can still do real hedge work against equity stress.")

    if credit_state == "Credit tightly driving equities":
        bits.append("Credit is central to the tape. Watch HY and financing-sensitive cyclicals closely.")
    elif credit_state == "Credit relevant":
        bits.append("Credit deserves respect as a confirming risk signal.")

    if not np.isnan(vix_rv_z):
        if vix_rv_z >= 1.0:
            bits.append("Implied vol screens rich to realized. Favor spreads, overwrites, or better-priced hedges over outright long vol.")
        elif vix_rv_z <= -1.0:
            bits.append("Implied vol screens cheap to realized. Optionality is more attractive than usual here.")

    return " ".join(bits)

# =========================
# Data fetch
# =========================
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_prices(tickers, start, end):
    try:
        data = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as e:
        st.error(f"Yahoo Finance download failed: {e}")
        return pd.DataFrame()

    if data is None or len(data) == 0:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        frames = []
        for t in tickers:
            if t not in data.columns.get_level_values(0):
                continue
            df = data[t].copy()
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            if col not in df.columns:
                continue
            frames.append(df[col].rename(t))
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, axis=1)
    else:
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        if col not in data.columns:
            return pd.DataFrame()
        out = data[[col]].rename(columns={col: tickers[0]})

    out.index.name = "Date"
    return out.sort_index()

# =========================
# Feature engine
# =========================
def build_transformed_returns(price_df, asset_meta):
    transformed = {}
    for ticker in price_df.columns:
        s = price_df[ticker].copy()
        mode = asset_meta.get(ticker, {}).get("transform", "pct")
        if mode == "diff":
            transformed[ticker] = s.diff()
        else:
            transformed[ticker] = s.pct_change()
    out = pd.DataFrame(transformed).replace([np.inf, -np.inf], np.nan)
    return out.dropna(how="all")

def build_pct_returns(price_df, tickers):
    tickers = [t for t in tickers if t in price_df.columns]
    if not tickers:
        return pd.DataFrame()
    out = price_df[tickers].pct_change().replace([np.inf, -np.inf], np.nan)
    return out.dropna(how="all")

def rolling_corr(series_a, series_b, window):
    df = pd.concat([series_a, series_b], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1])

def build_pair_feature_table(transformed_ret, pair_specs, corr_window, week_back, long_rank_window=504):
    rows = []

    for spec in pair_specs:
        a = spec["a"]
        b = spec["b"]
        title = spec["title"]

        if a not in transformed_ret.columns or b not in transformed_ret.columns:
            continue

        rc = rolling_corr(transformed_ret[a], transformed_ret[b], corr_window).dropna()
        if len(rc) < max(corr_window + 10, week_back + 2):
            continue

        rho_now = safe_last(rc)
        rho_delta_w = week_delta(rc, week_back)
        pctile_full = pct_rank_last(rc)
        pctile_2y = pct_rank_last(rc.tail(long_rank_window))
        persist = float((rc.tail(5).abs() >= 0.30).mean() * 100.0) if len(rc) >= 5 else np.nan

        level_score = min(abs(rho_now) / 0.80, 1.0) * 40 if not np.isnan(rho_now) else 0
        delta_score = min(abs(rho_delta_w) / 0.25, 1.0) * 35 if not np.isnan(rho_delta_w) else 0
        extremity_score = (abs(pctile_2y - 50) / 50) * 15 if not np.isnan(pctile_2y) else 0
        persistence_score = (persist / 100.0) * 10 if not np.isnan(persist) else 0

        signal_strength = round(level_score + delta_score + extremity_score + persistence_score, 1)
        bucket = signal_bucket(signal_strength)

        rows.append({
            "Pair": title,
            "a": a,
            "b": b,
            "a_label": ASSET_META.get(a, {}).get("label", a),
            "b_label": ASSET_META.get(b, {}).get("label", b),
            "rho_now": rho_now,
            "rho_delta_w": rho_delta_w,
            "pctile_full": pctile_full,
            "pctile_2y": pctile_2y,
            "persistence_5d": persist,
            "Signal strength": signal_strength,
            "Signal bucket": bucket,
            "Series": rc,
        })

    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return tbl

    explanations = tbl.apply(explain_pair_signal, axis=1)
    tbl["Interpretation"] = explanations.apply(lambda x: x[0])
    tbl["Action"] = explanations.apply(lambda x: x[1])
    tbl = tbl.sort_values(
        ["Signal strength", "rho_delta_w"],
        ascending=[False, False],
        na_position="last"
    ).reset_index(drop=True)
    return tbl

def build_regime_features(prices, pct_ret, transformed_ret, corr_window, week_back):
    features = {}

    vix = prices["^VIX"] if "^VIX" in prices.columns else pd.Series(dtype=float)
    vix3m = prices["^VIX3M"] if "^VIX3M" in prices.columns else pd.Series(dtype=float)

    if not vix.empty and not vix3m.empty:
        term_structure = (vix3m / vix) - 1.0
    else:
        term_structure = pd.Series(dtype=float)

    spy_rv = annualized_realized_vol(pct_ret["SPY"], window=21) * 100 if "SPY" in pct_ret.columns else pd.Series(dtype=float)
    hyg_rv = annualized_realized_vol(pct_ret["HYG"], window=21) * 100 if "HYG" in pct_ret.columns else pd.Series(dtype=float)
    uso_rv = annualized_realized_vol(pct_ret["USO"], window=21) * 100 if "USO" in pct_ret.columns else pd.Series(dtype=float)
    usdjpy_rv = annualized_realized_vol(pct_ret["USDJPY=X"], window=21) * 100 if "USDJPY=X" in pct_ret.columns else pd.Series(dtype=float)

    spy_tnx_corr = rolling_corr(transformed_ret["SPY"], transformed_ret["^TNX"], corr_window) if {"SPY", "^TNX"}.issubset(transformed_ret.columns) else pd.Series(dtype=float)
    spy_hyg_corr = rolling_corr(transformed_ret["SPY"], transformed_ret["HYG"], corr_window) if {"SPY", "HYG"}.issubset(transformed_ret.columns) else pd.Series(dtype=float)

    vix_rv_z = np.nan
    if not vix.empty and not spy_rv.empty:
        both = pd.concat([vix.rename("vix"), spy_rv.rename("rv")], axis=1).dropna()
        if not both.empty:
            spread = both["vix"] - both["rv"]
            vix_rv_z = zscore_last(spread, window=504)

    features["VIX"] = safe_last(vix)
    features["VIX term"] = safe_last(term_structure)
    features["SPY 1M RV"] = safe_last(spy_rv)
    features["SPY RV pctile"] = pct_rank_last(spy_rv)
    features["SPY vs 10Y rho"] = safe_last(spy_tnx_corr)
    features["SPY vs HY rho"] = safe_last(spy_hyg_corr)
    features["VIX minus RV z"] = vix_rv_z
    features["HY vol z"] = zscore_last(hyg_rv, window=504)
    features["Oil vol z"] = zscore_last(uso_rv, window=504)
    features["USDJPY vol z"] = zscore_last(usdjpy_rv, window=504)
    features["SPY RV Δw"] = week_delta(spy_rv, week_back)
    features["VIX Δw"] = week_delta(vix, week_back)
    features["Term Δw"] = week_delta(term_structure, week_back)

    return features

def classify_regime_state(feat):
    vol_state = classify_vol_state(
        feat.get("VIX", np.nan),
        feat.get("VIX term", np.nan),
        feat.get("SPY RV pctile", np.nan),
    )
    hedge_state = classify_hedge_state(feat.get("SPY vs 10Y rho", np.nan))
    credit_state = classify_credit_state(
        feat.get("HY vol z", np.nan),
        feat.get("SPY vs HY rho", np.nan),
    )

    valid_count = sum(pd.notna(list(feat.values())))
    total_count = len(feat)

    if vol_state == "Elevated" and hedge_state == "Duration hedge broken":
        regime_label = "Fragile risk tape"
    elif vol_state == "Calm" and hedge_state == "Duration hedge working":
        regime_label = "Constructive risk tape"
    elif vol_state == "Calm":
        regime_label = "Carry-friendly but watch structure"
    else:
        regime_label = "Mixed / transitional"

    return {
        "Regime": regime_label,
        "Vol state": vol_state,
        "Hedge state": hedge_state,
        "Credit state": credit_state,
        "Confidence": state_confidence(valid_count, total_count),
    }

def build_asset_stress_table(pct_ret, prices, week_back):
    rows = []

    asset_list = ["SPY", "HYG", "LQD", "TLT", "GLD", "USO", "UUP", "USDJPY=X"]
    for t in asset_list:
        if t not in prices.columns:
            continue

        label = ASSET_META[t]["label"]
        series = prices[t].dropna()
        daily_ret = pct_ret[t].dropna() if t in pct_ret.columns else pd.Series(dtype=float)
        rv = annualized_realized_vol(daily_ret, window=21) * 100 if not daily_ret.empty else pd.Series(dtype=float)

        one_w_pct = np.nan
        if len(series) > week_back + 1:
            one_w_pct = float((series.iloc[-1] / series.iloc[-week_back-1] - 1.0) * 100.0)

        rows.append({
            "Asset": label,
            "Last": safe_last(series),
            "1W %": one_w_pct,
            "1M RV": safe_last(rv),
            "RV z": zscore_last(rv, window=504),
            "RV pctile": pct_rank_last(rv),
        })

    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return tbl
    return tbl.sort_values("RV z", ascending=False, na_position="last").reset_index(drop=True)

# =========================
# Load data
# =========================
end_date = datetime.now().date()
start_date = (datetime.now() - timedelta(days=365 * lookback_years)).date()

with st.spinner("Downloading market data..."):
    prices = fetch_prices(ALL_TICKERS, str(start_date), str(end_date + timedelta(days=1)))

if prices.empty:
    st.error("No data downloaded. Check connectivity or try again.")
    st.stop()

prices = prices[[c for c in prices.columns if c in ALL_TICKERS]].copy()
prices = prices.dropna(how="all", axis=1)

transformed_ret = build_transformed_returns(prices, ASSET_META)
pct_ret = build_pct_returns(
    prices,
    [t for t, meta in ASSET_META.items() if meta["transform"] == "pct" and t in prices.columns]
)

pair_tbl = build_pair_feature_table(transformed_ret, PAIR_SPECS, corr_window, week_back)
regime_features = build_regime_features(prices, pct_ret, transformed_ret, corr_window, week_back)
regime_state = classify_regime_state(regime_features)
asset_stress_tbl = build_asset_stress_table(pct_ret, prices, week_back)

last_date = prices.dropna(how="all").index.max()

# =========================
# Header state
# =========================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Regime", regime_state["Regime"])
c2.metric("Vol state", regime_state["Vol state"])
c3.metric("Hedge state", regime_state["Hedge state"])
c4.metric("Credit state", regime_state["Credit state"])
c5.metric("Confidence", regime_state["Confidence"])

# =========================
# Top conclusions
# =========================
st.subheader("State, implication, invalidation")

state_line = (
    f"State: {regime_state['Regime']} | "
    f"VIX {regime_features.get('VIX', np.nan):.1f} | "
    f"VIX3M/VIX - 1 {regime_features.get('VIX term', np.nan):+.3f} | "
    f"SPY 1M realized vol {regime_features.get('SPY 1M RV', np.nan):.1f}% | "
    f"SPY vs 10Y rolling rho {regime_features.get('SPY vs 10Y rho', np.nan):+.0%}"
)

implication_line = (
    "Implication: " +
    action_bias_from_states(
        regime_state["Vol state"],
        regime_state["Hedge state"],
        regime_state["Credit state"],
        regime_features.get("VIX minus RV z", np.nan),
    )
)

invalidation_line = (
    "Invalidation: This read loses force if the SPY versus 10Y rolling correlation flips sign and holds for a week, "
    "if VIX term structure heals materially while realized vol compresses, or if the highest-ranked cross-asset shifts fade back below your thresholds."
)

st.markdown(state_line)
st.markdown(implication_line)
st.markdown(invalidation_line)

# =========================
# Top ranked signals
# =========================
st.subheader("Top ranked cross-asset signals")

if pair_tbl.empty:
    st.warning("No pair signals available for the selected configuration.")
else:
    signal_view = pair_tbl.copy()
    signal_view["Pass"] = (
        (signal_view["rho_now"].abs() >= abs_rho_threshold) |
        (signal_view["rho_delta_w"].abs() >= delta_threshold)
    )

    signal_view = signal_view.sort_values(
        ["Pass", "Signal strength"],
        ascending=[False, False]
    ).reset_index(drop=True)

    display_tbl = signal_view[[
        "Pair", "rho_now", "rho_delta_w", "pctile_2y", "persistence_5d",
        "Signal strength", "Signal bucket", "Interpretation", "Action"
    ]].copy()

    display_tbl = display_tbl.rename(columns={
        "rho_now": "ρ now",
        "rho_delta_w": "Δρ w/w",
        "pctile_2y": "2Y pctile",
        "persistence_5d": "5D persistence",
    })

    st.dataframe(
        display_tbl.style.format({
            "ρ now": "{:+.2f}",
            "Δρ w/w": "{:+.2f}",
            "2Y pctile": "{:.0f}%",
            "5D persistence": "{:.0f}%",
            "Signal strength": "{:.1f}",
        }),
        use_container_width=True,
        height=min(500, 38 * (len(display_tbl) + 1)),
    )

    passed = signal_view[signal_view["Pass"]].head(3)
    if not passed.empty:
        summary_bits = []
        for _, r in passed.iterrows():
            summary_bits.append(
                f"{r['Pair']} at {r['rho_now']:+.2f} with a weekly change of {r['rho_delta_w']:+.2f}; {r['Interpretation']}"
            )
        st.markdown(" ".join(summary_bits))

# =========================
# Asset stress table
# =========================
st.subheader("Asset stress map")

if asset_stress_tbl.empty:
    st.info("No asset stress data available.")
else:
    st.dataframe(
        asset_stress_tbl.style.format({
            "Last": "{:.2f}",
            "1W %": "{:+.2f}%",
            "1M RV": "{:.1f}%",
            "RV z": "{:+.2f}",
            "RV pctile": "{:.0f}%",
        }),
        use_container_width=True,
        height=min(350, 38 * (len(asset_stress_tbl) + 1)),
    )

# =========================
# Heatmap
# =========================
st.subheader(f"Current {corr_window}-day cross-asset correlation map")

heatmap_assets = [t for t in CORE_HEATMAP if t in transformed_ret.columns]
if len(heatmap_assets) >= 3:
    hm_data = transformed_ret[heatmap_assets].dropna()
    if len(hm_data) >= corr_window:
        hm = hm_data.tail(corr_window).corr()
        labels = [ASSET_META[t]["label"] for t in hm.index]

        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        im = ax.imshow(hm.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        for i in range(len(labels)):
            for j in range(len(labels)):
                val = hm.values[i, j]
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8)

        cbar = plt.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label("Correlation")
        ax.set_title(f"{corr_window}-day correlation matrix")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Not enough history for the selected rolling window.")
else:
    st.info("Not enough assets available for the heatmap.")

# =========================
# Detailed charts
# =========================
st.subheader("Detailed charts")

if pair_tbl.empty:
    st.info("No charts to display.")
else:
    with st.expander("Open top signal charts", expanded=False):
        chart_rows = pair_tbl.head(show_chart_count)

        def plot_pair_chart(series, title, corr_window):
            rc = pd.Series(series).dropna()
            if rc.empty:
                return None

            mean_6m = rc.rolling(126).mean()
            std_6m = rc.rolling(126).std()

            fig, ax = plt.subplots(figsize=(9.5, 3.5))
            ax.fill_between(
                rc.index,
                (mean_6m - std_6m).values,
                (mean_6m + std_6m).values,
                alpha=0.15,
                label="6M mean ± 1σ"
            )
            ax.plot(mean_6m.index, mean_6m.values, linewidth=1.5, label="6M mean")
            ax.plot(rc.index, rc.values, linewidth=1.6, label=f"{corr_window}D corr")
            ax.axhline(0, color="black", linewidth=1)
            ax.axhline(0.30, color="gray", linewidth=0.8, linestyle="--")
            ax.axhline(-0.30, color="gray", linewidth=0.8, linestyle="--")
            ax.axhline(0.50, color="gray", linewidth=0.8, linestyle=":")
            ax.axhline(-0.50, color="gray", linewidth=0.8, linestyle=":")

            ax.set_ylim(-1, 1)
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
            ax.set_title(title)
            ax.legend(loc="lower left", fontsize=8)
            plt.tight_layout()
            return fig

        for i in range(0, len(chart_rows), 2):
            cols = st.columns(2)
            subset = chart_rows.iloc[i:i+2]

            for col, (_, row) in zip(cols, subset.iterrows()):
                with col:
                    fig = plot_pair_chart(row["Series"], row["Pair"], corr_window)
                    if fig is not None:
                        st.pyplot(fig, use_container_width=True)
                        st.caption(f"{row['Interpretation']} Action: {row['Action']}")

# =========================
# Regime diagnostics
# =========================
st.subheader("Regime diagnostics")

diag_rows = [
    {"Feature": "VIX", "Value": regime_features.get("VIX", np.nan), "Comment": "Spot implied volatility"},
    {"Feature": "VIX3M/VIX - 1", "Value": regime_features.get("VIX term", np.nan), "Comment": "Term structure health"},
    {"Feature": "SPY 1M realized vol", "Value": regime_features.get("SPY 1M RV", np.nan), "Comment": "Realized equity stress"},
    {"Feature": "SPY RV percentile", "Value": regime_features.get("SPY RV pctile", np.nan), "Comment": "Where current realized vol sits in history"},
    {"Feature": "SPY vs 10Y rho", "Value": regime_features.get("SPY vs 10Y rho", np.nan), "Comment": "Duration hedge behavior"},
    {"Feature": "SPY vs HY rho", "Value": regime_features.get("SPY vs HY rho", np.nan), "Comment": "Credit transmission into equities"},
    {"Feature": "VIX minus RV z", "Value": regime_features.get("VIX minus RV z", np.nan), "Comment": "Rich or cheap implied vol"},
    {"Feature": "HY vol z", "Value": regime_features.get("HY vol z", np.nan), "Comment": "Credit volatility stress"},
    {"Feature": "Oil vol z", "Value": regime_features.get("Oil vol z", np.nan), "Comment": "Commodity shock risk"},
    {"Feature": "USDJPY vol z", "Value": regime_features.get("USDJPY vol z", np.nan), "Comment": "FX stress / carry stability"},
]
diag_tbl = pd.DataFrame(diag_rows)

st.dataframe(
    diag_tbl.style.format({"Value": "{:+.2f}"}),
    use_container_width=True,
    height=min(450, 38 * (len(diag_tbl) + 1)),
)

# =========================
# Footer
# =========================
footer_date = last_date.strftime("%Y-%m-%d") if pd.notna(last_date) else "N/A"
st.caption(f"As of {footer_date} | © 2026 AD Fund Management LP")
