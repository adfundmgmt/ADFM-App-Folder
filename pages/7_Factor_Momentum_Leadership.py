import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =========================================================
# Config
# =========================================================
st.set_page_config(page_title="Factor Momentum", layout="wide")
plt.style.use("default")

TITLE = "Factor Momentum"
SUBTITLE = "Factor momentum dashboard with broken-out factor charts and improved tape commentary."

PASTELS = [
    "#6FB9C3",
    "#E07A2F",
    "#6FA85A",
    "#F2B874",
    "#8CC7F2",
    "#A889C7",
    "#C7E29E",
    "#D89AD3",
    "#8FE3A1",
    "#E8CFC3",
]
TEXT = "#222222"
SUBTLE = "#666666"
GRID = "#E6E6E6"
BORDER = "#E0E0E0"
CARD_BG = "#FAFAFA"

CACHE_DIR = Path(".adfm_factor_cache")
CACHE_DIR.mkdir(exist_ok=True)

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
    h1, h2, h3 {font-weight: 600; letter-spacing: 0.15px;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================================================
# Helpers
# =========================================================
def card_box(inner_html: str) -> None:
    st.markdown(
        f"""
        <div style="border:1px solid {BORDER}; border-radius:10px;
                    padding:14px; background:{CARD_BG}; color:{TEXT};
                    font-size:14px; line-height:1.45;">
          {inner_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def _chunk(lst: List[str], n: int) -> List[List[str]]:
    n = max(1, int(n))
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def _safe_upper(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return str(value).upper().strip()

def _cache_path(name: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)
    return CACHE_DIR / f"{safe_name}.parquet"

def _load_last_good_cache(name: str) -> pd.DataFrame:
    path = _cache_path(name)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception:
        return pd.DataFrame()

def _save_last_good_cache(name: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    try:
        df.to_parquet(_cache_path(name))
    except Exception:
        pass

# =========================================================
# Factor ETFs
# =========================================================
FACTOR_ETFS: Dict[str, Tuple[str, Optional[str]]] = {
    "Growth vs Value": ("VUG", "VTV"),
    "Quality vs Junk": ("QUAL", "JNK"),
    "High Beta vs Low Vol": ("SPHB", "SPLV"),
    "Small vs Large": ("IWM", "SPY"),
    "Tech vs Broad": ("XLK", "SPY"),
    "Cyclicals vs Defensives": ("XLY", "XLP"),
    "US vs World": ("SPY", "VEA"),
    "Momentum": ("MTUM", "SPY"),
    "Equal Weight vs Cap": ("RSP", "SPY"),
}
BENCH = "SPY"

WINDOW_MAP_DAYS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
    "3Y": 252 * 3,
    "5Y": 252 * 5,
    "10Y": 252 * 10,
}

# =========================================================
# Math helpers
# =========================================================
def pct_change_window(series: pd.Series, days: int) -> float:
    s = series.dropna()
    if len(s) <= days:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-(days + 1)] - 1.0)

def momentum(series: pd.Series, win: int = 20) -> float:
    r = series.pct_change().dropna()
    if len(r) < 2:
        return np.nan
    win = int(min(win, len(r)))
    if win < 2:
        return np.nan
    return float(r.rolling(win).mean().iloc[-1])

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def trend_class(series: pd.Series) -> str:
    s = series.dropna()
    if len(s) < 50:
        return "Neutral"
    e10 = ema(s, 10).iloc[-1]
    e20 = ema(s, 20).iloc[-1]
    e40 = ema(s, 40).iloc[-1]
    if e10 > e20 > e40:
        return "Up"
    if e10 < e20 < e40:
        return "Down"
    return "Neutral"

def inflection(short_mom: float, long_mom: float) -> str:
    if pd.isna(short_mom) or pd.isna(long_mom):
        return "Neutral"
    if short_mom > 0 and long_mom < 0:
        return "Turning Up"
    if short_mom < 0 and long_mom > 0:
        return "Turning Down"
    if short_mom > 0 and long_mom > 0:
        return "Confirmed Up"
    if short_mom < 0 and long_mom < 0:
        return "Confirmed Down"
    return "Mixed"

def slope_zscore(series: pd.Series, lookback: int = 20) -> float:
    s = series.dropna()
    if len(s) < lookback + 5:
        return np.nan
    r = s.pct_change().dropna()
    roll = r.rolling(lookback).mean().dropna()
    if len(roll) < lookback:
        return np.nan
    mu = roll.iloc[-lookback:].mean()
    sd = roll.iloc[-lookback:].std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float((roll.iloc[-1] - mu) / sd)

def trend_strength(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 50:
        return np.nan
    e10 = ema(s, 10)
    e20 = ema(s, 20)
    e40 = ema(s, 40)
    spread = (e10 - e40) / e40
    return float(spread.iloc[-1])

def normalized_series(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return s
    return s / s.iloc[0] * 100.0

def robust_z_cross_section(series: pd.Series) -> pd.Series:
    s = series.astype(float).replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty:
        return pd.Series(index=s.index, dtype=float)
    median = s.median(skipna=True)
    mad = (s - median).abs().median(skipna=True)
    if pd.isna(mad) or mad == 0:
        std = s.std(skipna=True, ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=s.index)
        z = (s - s.mean(skipna=True)) / std
        return z.clip(-3, 3).fillna(0.0)
    z = 0.6745 * (s - median) / mad
    return z.clip(-3, 3).fillna(0.0)

def build_relative_series(a: pd.Series, b: pd.Series, min_obs: int = 60) -> pd.Series:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < min_obs:
        return pd.Series(dtype=float)
    out = aligned.iloc[:, 0] / aligned.iloc[:, 1]
    out.name = f"{a.name}_vs_{b.name}"
    return out

# =========================================================
# Commentary helpers
# =========================================================
def bucket_breadth(breadth: float) -> str:
    if breadth < 15:
        return "very narrow"
    if breadth < 35:
        return "narrow"
    if breadth < 55:
        return "mixed"
    if breadth < 75:
        return "broad"
    return "very broad"

def bucket_regime(regime_score: float) -> str:
    if regime_score < 25:
        return "defensive"
    if regime_score < 40:
        return "cautious"
    if regime_score < 60:
        return "balanced"
    if regime_score < 75:
        return "constructive"
    return "aggressively risk-seeking"

def build_commentary(mom_df: pd.DataFrame, breadth: float, regime_score: float) -> str:
    trend_counts = mom_df["Trend"].value_counts()
    up_count = int(trend_counts.get("Up", 0))
    down_count = int(trend_counts.get("Down", 0))
    neutral_count = int(trend_counts.get("Neutral", 0))

    confirmed_up = mom_df[mom_df["Inflection"] == "Confirmed Up"].sort_values("Short", ascending=False)
    confirmed_down = mom_df[mom_df["Inflection"] == "Confirmed Down"].sort_values("Short", ascending=True)
    turning_up = mom_df[mom_df["Inflection"] == "Turning Up"].sort_values("Short", ascending=False)
    turning_down = mom_df[mom_df["Inflection"] == "Turning Down"].sort_values("Short", ascending=True)

    short_sorted = mom_df.sort_values("Short", ascending=False)
    long_sorted = mom_df.sort_values("Long", ascending=False)

    leadership_names = short_sorted.index.tolist()[:3]
    laggard_names = short_sorted.index.tolist()[-3:]
    long_leaders = long_sorted.index.tolist()[:3]

    short_dispersion = (
        float(short_sorted["Short"].max() - short_sorted["Short"].min())
        if len(short_sorted) else np.nan
    )
    long_dispersion = (
        float(long_sorted["Long"].max() - long_sorted["Long"].min())
        if len(long_sorted) else np.nan
    )

    alignment = float(((mom_df["Short"] > 0) & (mom_df["Long"] > 0)).mean())
    conflict = float(
        (
            ((mom_df["Short"] > 0) & (mom_df["Long"] < 0))
            | ((mom_df["Short"] < 0) & (mom_df["Long"] > 0))
        ).mean()
    )

    leadership_text = ", ".join(leadership_names) if leadership_names else "none"
    laggard_text = ", ".join(laggard_names) if laggard_names else "none"
    long_leaders_text = ", ".join(long_leaders) if long_leaders else "none"
    turning_up_text = ", ".join(turning_up.index.tolist()[:3]) if not turning_up.empty else "none"
    turning_down_text = ", ".join(turning_down.index.tolist()[:3]) if not turning_down.empty else "none"

    if breadth >= 60 and alignment >= 0.45:
        tape_read = (
            f"The factor board is fairly healthy. Breadth is {bucket_breadth(breadth)} and the tape reads {bucket_regime(regime_score)}, "
            f"with short and long windows lining up in a decent share of the board. Leadership is being carried by {leadership_text}, "
            f"while the longer-duration trend structure is strongest in {long_leaders_text}."
        )
    elif breadth < 40 and conflict >= 0.30:
        tape_read = (
            f"The factor board is unstable. Breadth is {bucket_breadth(breadth)} and the tape reads {bucket_regime(regime_score)}, "
            f"but a meaningful share of factors are fighting between short and long windows, which usually means rotation is happening faster than conviction is building. "
            f"Near-term leadership is in {leadership_text}, but that leadership still lacks clean confirmation across the full board."
        )
    else:
        tape_read = (
            f"The factor board is mixed. Breadth is {bucket_breadth(breadth)} and the tape reads {bucket_regime(regime_score)}. "
            f"The market is rewarding {leadership_text} on the short horizon, while the better anchored longer-window leadership sits in {long_leaders_text}. "
            f"That usually argues for selectivity rather than broad aggression."
        )

    internal_message = (
        f"Confirmed strength is concentrated in {', '.join(confirmed_up.index.tolist()[:4]) if not confirmed_up.empty else 'very few factor pairs'}, "
        f"while the weakest confirmed areas are {', '.join(confirmed_down.index.tolist()[:4]) if not confirmed_down.empty else 'not deeply entrenched'}. "
        f"Fresh improvement is showing up in {turning_up_text}, and deterioration is showing up in {turning_down_text}."
    )

    risk_message = (
        f"Short-window dispersion is {short_dispersion * 100:.1f}% and long-window dispersion is {long_dispersion * 100:.1f}%, "
        f"which tells you how concentrated the relative-strength trade has become. The weakest groups right now are {laggard_text}. "
        f"If that spread starts compressing while the leaders stall, the next move usually becomes rotation rather than straightforward continuation."
    )

    stats_message = (
        f"Uptrends: {up_count}. Downtrends: {down_count}. Neutral: {neutral_count}. "
        f"Alignment ratio: {alignment * 100:.1f}%. Conflict ratio: {conflict * 100:.1f}%. "
        f"Breadth index: {breadth:.1f}%. Regime score: {regime_score:.1f}."
    )

    body = (
        '<div style="font-weight:700; margin-bottom:6px;">Conclusion</div>'
        f"<div>{tape_read}</div>"
        '<div style="font-weight:700; margin:10px 0 6px;">Internal read</div>'
        f"<div>{internal_message}</div>"
        '<div style="font-weight:700; margin:10px 0 6px;">What to watch</div>'
        f"<div>{risk_message}</div>"
        '<div style="font-weight:700; margin:10px 0 6px;">Key stats</div>'
        f"<div>{stats_message}</div>"
    )
    return body

# =========================================================
# Data download and normalization
# =========================================================
def _normalize_yf_download(df: pd.DataFrame, requested: List[str]) -> pd.DataFrame:
    requested = [_safe_upper(x) for x in requested if x]
    if df is None or df.empty:
        return pd.DataFrame()

    out: Optional[pd.DataFrame] = None

    if not isinstance(df.columns, pd.MultiIndex):
        cols_lower = {str(c).lower(): c for c in df.columns}
        price_col = None
        for candidate in ("close", "adj close"):
            if candidate in cols_lower:
                price_col = cols_lower[candidate]
                break
        if price_col is not None:
            s = df[price_col].copy()
            ticker = requested[0] if requested else "TICKER"
            s.name = ticker
            out = s.to_frame()

    if out is None and isinstance(df.columns, pd.MultiIndex):
        level0 = [str(x) for x in df.columns.get_level_values(0)]
        level1 = [str(x) for x in df.columns.get_level_values(1)]

        if "Close" in level0 or "Adj Close" in level0:
            field = "Close" if "Close" in level0 else "Adj Close"
            try:
                tmp = df[field].copy()
                if isinstance(tmp, pd.Series):
                    tmp = tmp.to_frame()
                tmp.columns = [_safe_upper(c) for c in tmp.columns]
                out = tmp
            except Exception:
                out = None

        if out is None:
            candidate_fields = {"Close", "Adj Close"}
            if any(f in level1 for f in candidate_fields):
                frames = {}
                for t in requested:
                    for fld in ("Close", "Adj Close"):
                        key = (t, fld)
                        if key in df.columns:
                            frames[t] = df[key]
                            break
                if frames:
                    out = pd.DataFrame(frames)

    if out is None or out.empty:
        return pd.DataFrame()

    out = out.copy()
    out.index = pd.to_datetime(out.index).normalize()
    out = out.sort_index()
    out = out.loc[:, ~out.columns.duplicated()]
    valid_cols = [c for c in requested if c in out.columns]
    out = out[valid_cols]
    return out

def _download_batch_once(batch: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    batch = [_safe_upper(x) for x in batch if x]
    if not batch:
        return pd.DataFrame()

    try:
        data = yf.download(
            tickers=batch if len(batch) > 1 else batch[0],
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column",
        )
        return _normalize_yf_download(data, batch)
    except Exception:
        return pd.DataFrame()

def _download_one_by_one(batch: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for ticker in batch:
        out = pd.DataFrame()
        for _ in range(3):
            out = _download_batch_once([ticker], start, end)
            if not out.empty and ticker in out.columns and out[ticker].dropna().shape[0] > 0:
                break
            time.sleep(0.5)
        if not out.empty:
            frames.append(out)
    if not frames:
        return pd.DataFrame()
    wide = pd.concat(frames, axis=1)
    wide = wide.loc[:, ~wide.columns.duplicated()].sort_index()
    return wide

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_daily_levels(
    tickers: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_size: int = 25
) -> pd.DataFrame:
    uniq = sorted({_safe_upper(t) for t in tickers if t})
    if not uniq:
        return pd.DataFrame()

    cache_key = f"levels_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_{'_'.join(uniq)}"

    frames: List[pd.DataFrame] = []
    found: set = set()

    for batch in _chunk(uniq, chunk_size):
        out = pd.DataFrame()
        for _ in range(3):
            out = _download_batch_once(batch, start, end)
            if not out.empty:
                break
            time.sleep(0.5)
        if not out.empty:
            frames.append(out)
            found.update(out.columns.tolist())

    missing = [t for t in uniq if t not in found]
    if missing:
        rescue = _download_one_by_one(missing, start, end)
        if not rescue.empty:
            frames.append(rescue)
            found.update(rescue.columns.tolist())

    if not frames:
        return _load_last_good_cache(cache_key)

    wide = pd.concat(frames, axis=1)
    wide = wide.loc[:, ~wide.columns.duplicated()]
    wide = wide.sort_index()
    wide = wide[[c for c in uniq if c in wide.columns]]
    wide = wide.ffill(limit=3)

    if not wide.empty:
        _save_last_good_cache(cache_key, wide)

    return wide

# =========================================================
# Sidebar
# =========================================================
st.title(TITLE)
st.caption(SUBTITLE)

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Purpose: Factor leadership monitor based on relative momentum, trend, and inflection signals.

        What it covers
        • Core signals and summary outputs for this dashboard
        • Key context needed to interpret current regime or setup
        • Practical view designed for quick internal decision support

        Data source
        • Public market and macro data feeds used throughout the app
        """
    )
    st.markdown("---")

with st.sidebar:
    st.header("Settings")
    history_start = st.date_input("History start", datetime(2018, 1, 1))
    window_choice = st.selectbox(
        "Analysis window",
        ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "10Y"],
        index=3,
    )
    lookback_short = st.slider("Short momentum window (days)", 10, 60, 20)
    lookback_long = st.slider("Long momentum window (days)", 30, 180, 60)
    normalize_charts = st.checkbox("Normalize factor charts to 100", value=False)
    min_overlap_obs = st.slider("Minimum overlap observations per factor pair", 40, 252, 60, 5)
    st.caption("Data source: Yahoo Finance. Internal use only.")

# =========================================================
# Compute window
# =========================================================
today = date.today()

if window_choice == "YTD":
    window_start = pd.Timestamp(date(datetime.now().year, 1, 1))
    requested_days = None
else:
    requested_days = WINDOW_MAP_DAYS[window_choice]
    window_start = pd.Timestamp(today) - pd.Timedelta(days=int(requested_days * 1.6))

window_end = pd.Timestamp(today) + pd.Timedelta(days=1)

# =========================================================
# Build universe and fetch
# =========================================================
factor_tickers = sorted({t for pair in FACTOR_ETFS.values() for t in pair if t is not None} | {BENCH})

levels = fetch_daily_levels(
    factor_tickers,
    start=pd.Timestamp(history_start),
    end=window_end,
)

if levels.empty:
    st.error("Price download failed for all requested ETFs. Yahoo returned no usable close data and no last-good cache was available.")
    st.stop()

if BENCH not in levels.columns or levels[BENCH].dropna().empty:
    st.error("SPY data missing or empty after fetch normalization.")
    st.stop()

# =========================================================
# Factor series with diagnostics
# =========================================================
factor_levels_full: Dict[str, pd.Series] = {}
pair_diagnostics: List[Dict[str, str]] = []

for name, (up, down) in FACTOR_ETFS.items():
    up_u = _safe_upper(up)
    down_u = _safe_upper(down)

    if up_u not in levels.columns or levels[up_u].dropna().empty:
        pair_diagnostics.append({
            "Factor": name,
            "Status": "Skipped",
            "Reason": f"Missing usable data for {up_u}",
        })
        continue

    if down_u is None:
        factor_levels_full[name] = levels[up_u].dropna().copy()
        pair_diagnostics.append({
            "Factor": name,
            "Status": "OK",
            "Reason": f"Using standalone series {up_u}",
        })
        continue

    if down_u not in levels.columns or levels[down_u].dropna().empty:
        pair_diagnostics.append({
            "Factor": name,
            "Status": "Skipped",
            "Reason": f"Missing usable data for {down_u}",
        })
        continue

    rel = build_relative_series(levels[up_u], levels[down_u], min_obs=min_overlap_obs)
    if rel.empty:
        overlap = pd.concat([levels[up_u], levels[down_u]], axis=1).dropna().shape[0]
        pair_diagnostics.append({
            "Factor": name,
            "Status": "Skipped",
            "Reason": f"Overlap too short: {overlap} observations",
        })
        continue

    factor_levels_full[name] = rel
    pair_diagnostics.append({
        "Factor": name,
        "Status": "OK",
        "Reason": f"{up_u} / {down_u}",
    })

factor_df_full = pd.DataFrame(factor_levels_full).dropna(how="all")

if factor_df_full.empty:
    st.error("No factor series could be constructed from the available ETF data.")
    with st.expander("Diagnostics", expanded=True):
        st.dataframe(pd.DataFrame(pair_diagnostics), use_container_width=True, hide_index=True)
    st.stop()

if requested_days is None:
    factor_df = factor_df_full[factor_df_full.index >= window_start].copy()
else:
    factor_df = factor_df_full.tail(min(requested_days, len(factor_df_full))).copy()
    if not factor_df.empty:
        window_start = factor_df.index.min()

if factor_df.empty:
    st.error("No data available for the selected window.")
    with st.expander("Diagnostics", expanded=True):
        st.dataframe(pd.DataFrame(pair_diagnostics), use_container_width=True, hide_index=True)
    st.stop()

# =========================================================
# Momentum snapshot
# =========================================================
rows = []
for factor_name in factor_df.columns:
    s = factor_df[factor_name].dropna()
    if len(s) < 15:
        continue

    eff_short = min(lookback_short, max(5, len(s) - 2))
    eff_long = min(lookback_long, max(eff_short + 1, len(s) - 2))

    if len(s) <= eff_long:
        continue

    r5 = pct_change_window(s, min(5, len(s) - 2))
    r_short = pct_change_window(s, eff_short)
    r_long = pct_change_window(s, eff_long)
    mom_val = momentum(s, win=min(eff_short, max(2, len(s) - 1)))
    tclass = trend_class(s)
    infl = inflection(r_short, r_long)
    slope_z = slope_zscore(s, lookback=min(20, max(10, len(s) // 2)))
    t_strength = trend_strength(s)

    rows.append([
        factor_name,
        r5,
        r_short,
        r_long,
        mom_val,
        tclass,
        infl,
        eff_short,
        eff_long,
        slope_z,
        t_strength,
    ])

mom_df = pd.DataFrame(
    rows,
    columns=[
        "Factor",
        "%5D",
        "Short",
        "Long",
        "Momentum",
        "Trend",
        "Inflection",
        "Eff Short",
        "Eff Long",
        "Slope Z",
        "Trend Strength",
    ],
).set_index("Factor")

if mom_df.empty:
    st.error("No factors passed data checks for this window. Try a longer analysis window or reduce the short and long lookbacks.")
    with st.expander("Diagnostics", expanded=True):
        st.dataframe(pd.DataFrame(pair_diagnostics), use_container_width=True, hide_index=True)
    st.stop()

mom_df = mom_df.sort_values("Short", ascending=False)

# =========================================================
# Tape score
# =========================================================
trend_counts = mom_df["Trend"].value_counts()
num_up = int(trend_counts.get("Up", 0))
num_down = int(trend_counts.get("Down", 0))
num_neutral = int(trend_counts.get("Neutral", 0))

breadth = num_up / len(mom_df) * 100.0

short_z = robust_z_cross_section(mom_df["Short"])
long_z = robust_z_cross_section(mom_df["Long"])
slope_z = robust_z_cross_section(mom_df["Slope Z"].fillna(0.0))

turning_balance = (
    (mom_df["Inflection"] == "Turning Up").mean()
    - (mom_df["Inflection"] == "Turning Down").mean()
)
trend_balance = (
    (mom_df["Trend"] == "Up").mean()
    - (mom_df["Trend"] == "Down").mean()
)

raw_score = (
    0.30 * short_z.mean()
    + 0.25 * long_z.mean()
    + 0.15 * slope_z.mean()
    + 0.15 * turning_balance * 3.0
    + 0.15 * trend_balance * 3.0
)

regime_score = float(np.clip(50.0 + 15.0 * raw_score, 0.0, 100.0))

# =========================================================
# Summary
# =========================================================
st.subheader(f"Factor Tape Summary ({window_choice})")
summary_html = build_commentary(mom_df, breadth, regime_score)
card_box(summary_html)

ok_pairs = [d for d in pair_diagnostics if d["Status"] == "OK"]
skipped_pairs = [d for d in pair_diagnostics if d["Status"] != "OK"]

diag_title = f"Diagnostics: {len(ok_pairs)} factors built, {len(skipped_pairs)} skipped"
with st.expander(diag_title, expanded=False):
    st.dataframe(pd.DataFrame(pair_diagnostics), use_container_width=True, hide_index=True)

# =========================================================
# Broken-out factor images
# =========================================================
st.subheader(f"Factor Time Series ({window_choice})")

plot_df = factor_df.copy()
if normalize_charts:
    for col in plot_df.columns:
        plot_df[col] = normalized_series(plot_df[col])

n_factors = len(plot_df.columns)
ncols = 3
nrows = int(np.ceil(n_factors / ncols))

fig_ts, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), squeeze=False)
axes = axes.ravel()

if len(plot_df.index) > 1:
    span_days = (plot_df.index[-1] - plot_df.index[0]).days
else:
    span_days = 0

if span_days <= 120:
    locator = mdates.WeekdayLocator(interval=2)
    formatter = mdates.DateFormatter("%b %d")
elif span_days <= 420:
    locator = mdates.MonthLocator()
    formatter = mdates.DateFormatter("%b")
else:
    locator = mdates.YearLocator()
    formatter = mdates.DateFormatter("%Y")

last_valid_i = -1
for i, factor_name in enumerate(plot_df.columns):
    last_valid_i = i
    ax = axes[i]
    s = plot_df[factor_name].dropna()
    if s.empty:
        ax.axis("off")
        continue

    color = PASTELS[i % len(PASTELS)]
    ax.plot(s.index, s.values, color=color, linewidth=2.1)

    if len(s) >= 20 and not normalize_charts:
        e20 = ema(s, 20)
        ax.plot(e20.index, e20.values, color="#888888", linewidth=1.1, alpha=0.9)

    latest = s.iloc[-1]
    latest_short = mom_df.loc[factor_name, "Short"] * 100.0 if factor_name in mom_df.index else np.nan
    latest_long = mom_df.loc[factor_name, "Long"] * 100.0 if factor_name in mom_df.index else np.nan

    title = f"{factor_name}\nS {latest_short:.1f}% | L {latest_long:.1f}%"
    ax.set_title(title, color=TEXT, fontsize=11, pad=8)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.scatter(s.index[-1], latest, s=18, color=color, zorder=3)

for j in range(last_valid_i + 1, len(axes)):
    axes[j].axis("off")

fig_ts.tight_layout()
st.pyplot(fig_ts, clear_figure=True)
plt.close(fig_ts)

# =========================================================
# Snapshot table
# =========================================================
st.subheader("Factor Momentum Snapshot")

display_df = mom_df.copy()
for col in ["%5D", "Short", "Long", "Momentum", "Trend Strength"]:
    display_df[col] = display_df[col] * 100.0

display_df = display_df[
    ["%5D", "Short", "Long", "Momentum", "Trend", "Inflection", "Eff Short", "Eff Long", "Slope Z", "Trend Strength"]
]

st.dataframe(
    display_df.style.format(
        {
            "%5D": "{:.1f}%",
            "Short": "{:.1f}%",
            "Long": "{:.1f}%",
            "Momentum": "{:.2f}%",
            "Slope Z": "{:.2f}",
            "Trend Strength": "{:.2f}%",
            "Eff Short": "{:.0f}",
            "Eff Long": "{:.0f}",
        }
    ),
    use_container_width=True,
)

# =========================================================
# Leadership map
# =========================================================
st.subheader("Leadership Map (Short vs Long Momentum)")

fig_lead, ax_lead = plt.subplots(figsize=(8.6, 6.4))
short_vals = mom_df["Short"] * 100.0
long_vals = mom_df["Long"] * 100.0

x_abs = max(abs(short_vals.min()), abs(short_vals.max()), 1.0)
y_abs = max(abs(long_vals.min()), abs(long_vals.max()), 1.0)
pad_x = x_abs * 0.15
pad_y = y_abs * 0.15

ax_lead.set_xlim(-x_abs - pad_x, x_abs + pad_x)
ax_lead.set_ylim(-y_abs - pad_y, y_abs + pad_y)

x_min, x_max = ax_lead.get_xlim()
y_min, y_max = ax_lead.get_ylim()

ax_lead.fill_between([0, x_max], 0, y_max, color="#E1F5E0", alpha=0.55)
ax_lead.fill_between([x_min, 0], 0, y_max, color="#FFF9C4", alpha=0.55)
ax_lead.fill_between([x_min, 0], y_min, 0, color="#FDE0DC", alpha=0.55)
ax_lead.fill_between([0, x_max], y_min, 0, color="#FFE9B3", alpha=0.55)

ax_lead.axvline(0, color="#888888", linewidth=1)
ax_lead.axhline(0, color="#888888", linewidth=1)

ax_lead.text(
    x_max * 0.62, y_max * 0.76,
    "Short ↑ / Long ↑\nEstablished leaders",
    fontsize=9, ha="center", va="center", color="#333333"
)
ax_lead.text(
    x_min * 0.62, y_max * 0.76,
    "Short ↓ / Long ↑\nMean reversion",
    fontsize=9, ha="center", va="center", color="#333333"
)
ax_lead.text(
    x_min * 0.62, y_min * 0.76,
    "Short ↓ / Long ↓\nPersistent laggards",
    fontsize=9, ha="center", va="center", color="#333333"
)
ax_lead.text(
    x_max * 0.62, y_min * 0.76,
    "Short ↑ / Long ↓\nNew rotations",
    fontsize=9, ha="center", va="center", color="#333333"
)

for k, factor_name in enumerate(mom_df.index):
    x = short_vals.loc[factor_name]
    y = long_vals.loc[factor_name]
    ax_lead.scatter(
        x, y, s=75, color=PASTELS[k % len(PASTELS)],
        edgecolor="#444444", linewidth=0.6, zorder=3
    )
    ax_lead.annotate(
        factor_name,
        xy=(x, y),
        xytext=(4, 3),
        textcoords="offset points",
        fontsize=9,
        va="center",
        color="#111111",
    )

ax_lead.set_xlabel("Short window return %", color=TEXT)
ax_lead.set_ylabel("Long window return %", color=TEXT)
ax_lead.set_title("Factors by Short vs Long Momentum", color=TEXT, pad=10)
ax_lead.grid(color=GRID, linewidth=0.6, alpha=0.6)

for spine in ["top", "right"]:
    ax_lead.spines[spine].set_visible(False)

fig_lead.tight_layout()
st.pyplot(fig_lead, clear_figure=True)
plt.close(fig_lead)

st.caption("© 2026 AD Fund Management LP")
