import time
from datetime import datetime, date
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
def card_box(inner_html: str):
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

def metric_box(label: str, value: str, sub: Optional[str] = None):
    sub_html = f'<div style="font-size:11px; color:{SUBTLE}; margin-top:4px;">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div style="border:1px solid {BORDER}; border-radius:10px; padding:12px 14px; background:white;">
            <div style="font-size:12px; color:{SUBTLE};">{label}</div>
            <div style="font-size:25px; font-weight:700; color:{TEXT}; margin-top:4px;">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def _chunk(lst: List[str], n: int) -> List[List[str]]:
    n = max(1, int(n))
    return [lst[i:i+n] for i in range(0, len(lst), n)]

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

def rs(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    out = aligned.iloc[:, 0] / aligned.iloc[:, 1]
    out.name = f"{series_a.name}_vs_{series_b.name}"
    return out

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

    short_dispersion = float(short_sorted["Short"].max() - short_sorted["Short"].min()) if len(short_sorted) else np.nan
    long_dispersion = float(long_sorted["Long"].max() - long_sorted["Long"].min()) if len(long_sorted) else np.nan

    alignment = float(((mom_df["Short"] > 0) & (mom_df["Long"] > 0)).mean())
    conflict = float(((mom_df["Short"] > 0) & (mom_df["Long"] < 0) | (mom_df["Short"] < 0) & (mom_df["Long"] > 0)).mean())

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
        f"If that spread starts compressing while the leaders stall, the next move is usually rotation rather than straightforward continuation."
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
# Data download
# =========================================================
def _download_close_batch(batch: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    batch = [str(x).upper() for x in batch if x]
    if not batch:
        return pd.DataFrame()

    try:
        if len(batch) == 1:
            t = batch[0]
            df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False, threads=False)
            if df is None or df.empty:
                return pd.DataFrame()
            col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
            if col is None:
                return pd.DataFrame()
            s = df[col].copy()
            s.name = t
            return s.to_frame()

        df = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = set(df.columns.get_level_values(0))
            if "Close" in lvl0:
                close = df["Close"].copy()
            elif "Adj Close" in lvl0:
                close = df["Adj Close"].copy()
            else:
                return pd.DataFrame()
            close.columns = [str(c).upper() for c in close.columns]
            return close

    except Exception:
        return pd.DataFrame()

    return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_daily_levels(tickers: List[str], start: pd.Timestamp, end: pd.Timestamp, chunk_size: int = 50) -> pd.DataFrame:
    uniq = sorted(list({str(t).upper() for t in tickers if t}))
    frames = []

    for batch in _chunk(uniq, chunk_size):
        out = pd.DataFrame()
        for _ in range(3):
            out = _download_close_batch(batch, start, end)
            if not out.empty:
                break
            time.sleep(0.4)
        if not out.empty:
            frames.append(out)

    if not frames:
        return pd.DataFrame()

    wide = pd.concat(frames, axis=1)
    wide = wide.loc[:, ~wide.columns.duplicated()].sort_index()
    if wide.empty:
        return wide

    bidx = pd.bdate_range(wide.index.min(), wide.index.max(), name=wide.index.name)
    wide = wide.reindex(bidx).ffill()
    return wide

# =========================================================
# Sidebar
# =========================================================
st.title(TITLE)
st.caption(SUBTITLE)

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
    end=window_end
)

if levels.empty:
    st.error("No data returned.")
    st.stop()

if BENCH not in levels.columns or levels[BENCH].dropna().empty:
    st.error("SPY data missing or empty.")
    st.stop()

# =========================================================
# Factor series
# =========================================================
factor_levels_full = {}
for name, (up, down) in FACTOR_ETFS.items():
    up = up.upper()
    down_u = down.upper() if down is not None else None

    if down_u is None:
        if up in levels.columns:
            factor_levels_full[name] = levels[up]
        continue

    if up in levels.columns and down_u in levels.columns:
        rel = rs(levels[up], levels[down_u])
        if not rel.empty:
            factor_levels_full[name] = rel

factor_df_full = pd.DataFrame(factor_levels_full).dropna(how="all")
if factor_df_full.empty:
    st.error("No factor series could be constructed.")
    st.stop()

if requested_days is None:
    factor_df = factor_df_full[factor_df_full.index >= window_start].copy()
else:
    factor_df = factor_df_full.tail(min(requested_days, len(factor_df_full))).copy()
    if not factor_df.empty:
        window_start = factor_df.index.min()

if factor_df.empty:
    st.error("No data available for the selected window.")
    st.stop()

# =========================================================
# Momentum snapshot
# =========================================================
rows = []
for f in factor_df.columns:
    s = factor_df[f].dropna()
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
        f,
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
    st.error(
        "No factors passed data checks for this window. Try a longer analysis window or reduce the short and long lookbacks."
    )
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

raw_score = (
    0.30 * mom_df["Short"].mean()
    + 0.20 * mom_df["Long"].mean()
    + 0.20 * ((mom_df["Inflection"] == "Turning Up").mean() - (mom_df["Inflection"] == "Turning Down").mean())
    + 0.20 * ((mom_df["Trend"] == "Up").mean() - (mom_df["Trend"] == "Down").mean())
    + 0.10 * mom_df["Slope Z"].fillna(0).mean()
)
regime_score = max(0.0, min(100.0, 50.0 + 50.0 * (raw_score / 5.0)))

# =========================================================
# Summary
# =========================================================
st.subheader(f"Factor Tape Summary ({window_choice})")

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_box("Breadth", f"{breadth:.1f}%", "Share of factors in uptrends")
with c2:
    metric_box("Regime score", f"{regime_score:.1f}", "0 to 100 scale")
with c3:
    metric_box("Uptrends", str(num_up), "10 / 20 / 40 EMA stack")
with c4:
    metric_box("Downtrends", str(num_down), "10 / 20 / 40 EMA stack")

summary_html = build_commentary(mom_df, breadth, regime_score)
card_box(summary_html)

# =========================================================
# Broken-out factor images
# =========================================================
st.subheader(f"Factor Time Series ({window_choice})")

plot_df = factor_df.copy()
if normalize_charts:
    for c in plot_df.columns:
        plot_df[c] = normalized_series(plot_df[c])

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

for i, f in enumerate(plot_df.columns):
    ax = axes[i]
    s = plot_df[f].dropna()
    if s.empty:
        ax.axis("off")
        continue

    ax.plot(s.index, s.values, color=PASTELS[i % len(PASTELS)], linewidth=2.1)

    if len(s) >= 20:
        e20 = ema(s, 20)
        ax.plot(e20.index, e20.values, color="#888888", linewidth=1.1, alpha=0.9)

    ax.set_title(f, color=TEXT, fontsize=11, pad=8)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    latest = s.iloc[-1]
    ax.scatter(s.index[-1], latest, s=18, color=PASTELS[i % len(PASTELS)], zorder=3)

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

fig_ts.tight_layout()
st.pyplot(fig_ts, clear_figure=True)

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

ax_lead.text(x_max * 0.62, y_max * 0.76, "Short ↑ / Long ↑\nEstablished leaders", fontsize=9, ha="center", va="center", color="#333333")
ax_lead.text(x_min * 0.62, y_max * 0.76, "Short ↓ / Long ↑\nMean reversion", fontsize=9, ha="center", va="center", color="#333333")
ax_lead.text(x_min * 0.62, y_min * 0.76, "Short ↓ / Long ↓\nPersistent laggards", fontsize=9, ha="center", va="center", color="#333333")
ax_lead.text(x_max * 0.62, y_min * 0.76, "Short ↑ / Long ↓\nNew rotations", fontsize=9, ha="center", va="center", color="#333333")

for k, factor in enumerate(mom_df.index):
    x = short_vals.loc[factor]
    y = long_vals.loc[factor]
    ax_lead.scatter(x, y, s=75, color=PASTELS[k % len(PASTELS)], edgecolor="#444444", linewidth=0.6, zorder=3)
    ax_lead.annotate(
        factor,
        xy=(x, y),
        xytext=(4, 3),
        textcoords="offset points",
        fontsize=9,
        va="center",
        color="#111111"
    )

ax_lead.set_xlabel("Short window return %", color=TEXT)
ax_lead.set_ylabel("Long window return %", color=TEXT)
ax_lead.set_title("Factors by Short vs Long Momentum", color=TEXT, pad=10)
ax_lead.grid(color=GRID, linewidth=0.6, alpha=0.6)

for spine in ["top", "right"]:
    ax_lead.spines[spine].set_visible(False)

fig_lead.tight_layout()
st.pyplot(fig_lead, clear_figure=True)

st.caption("© 2026 AD Fund Management LP")
