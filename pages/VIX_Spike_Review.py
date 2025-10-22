# vix_spike_deep_dive.py
# ADFM Analytics Platform — VIX 20%+ Spike Deep Dive
# Light theme, pastel palette, matplotlib only, no seaborn.

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st

# ------------------------------- Page config -------------------------------
st.set_page_config(
    page_title="ADFM | VIX 20%+ Spike Deep Dive",
    layout="wide"
)
plt.style.use("default")

# ------------------------------- Pastel palette ----------------------------
PASTELS = [
    "#A8DADC",  # soft teal
    "#F4A261",  # sand
    "#90BE6D",  # sage
    "#FFCC99",  # peach
    "#BDE0FE",  # powder blue
    "#CDB4DB",  # lavender
    "#FFD6A5",  # light apricot
    "#E2F0CB",  # mint
    "#F1C0E8",  # pink lavender
    "#B9FBC0"   # soft green
]

GRID_COLOR = "#e6e6e6"
BAR_EDGE = "#666666"
TEXT_COLOR = "#222222"

# ------------------------------- Helpers -----------------------------------
@st.cache_data(show_spinner=False, ttl=60*60)
def load_history(start="1990-01-01"):
    vix = yf.download("^VIX", start=start, auto_adjust=False, progress=False)
    spx = yf.download("^GSPC", start=start, auto_adjust=False, progress=False)
    vix = vix.rename(columns=str.lower)
    spx = spx.rename(columns=str.lower)
    df = pd.DataFrame(index=vix.index)
    df["vix_close"] = vix["close"]
    df["vix_prev"] = df["vix_close"].shift(1)
    df["vix_pctchg"] = (df["vix_close"] / df["vix_prev"] - 1.0)
    df["spx_close"] = spx["close"].reindex(df.index).ffill()
    df["spx_next2"] = df["spx_close"].shift(-2)
    df["spx_fwd2_ret"] = (df["spx_next2"] / df["spx_close"] - 1.0) * 100.0
    # 200dma for regime
    df["spx_200dma"] = df["spx_close"].rolling(200).mean()
    df["regime"] = np.where(df["spx_close"] >= df["spx_200dma"], "Bull", "Bear")
    # RSI for oversold
    df["rsi14"] = compute_rsi(df["spx_close"], 14)
    return df.dropna(subset=["vix_close", "spx_close"])

def compute_rsi(series, n=14):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/n, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def bucket_vix_base(x):
    # Buckets tuned to your screenshot
    bins = [0, 12, 16, 20, 24, 30, 99]
    labels = ["0-12", "12-16", "16-20", "20-24", "24-30", "30+"]
    return pd.cut([x], bins=bins, labels=labels, right=False)[0]

def bucket_spike_mag(pct):
    if pct < 0.30:
        return "Moderate (20-30%)"
    elif pct < 0.50:
        return "Large (30-50%)"
    else:
        return "Extreme (50%+)"

def decade_label(dt):
    y = dt.year
    return f"{y//10*10}s"

def winrate(series):
    return 100.0 * (series > 0).mean() if len(series) else np.nan

def barplot(ax, categories, values, colors, title, ylabel="2-Day Win Rate (%)"):
    ax.bar(categories, values, color=colors, edgecolor=BAR_EDGE)
    ax.axhline(50, linestyle="--", linewidth=1, color="#888888")
    ax.set_title(title, color=TEXT_COLOR, fontsize=12, pad=8)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    ax.set_ylim(0, max(70, np.nanmax(values) + 10))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    for i, v in enumerate(values):
        if np.isnan(v): 
            continue
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", color=TEXT_COLOR, fontsize=10)

def card_box(text):
    st.markdown(
        f"""
        <div style="border:1px solid #e0e0e0; border-radius:8px; padding:14px; background:#fafafa; color:{TEXT_COLOR}; font-size:14px;">
        {text}
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------- Sidebar -----------------------------------
st.title("VIX 20%+ Spike Deep Dive • ADFM")

with st.sidebar:
    st.header("Controls")
    start_date = st.date_input("History start", value=datetime(1990,1,1))
    fwd_days = st.number_input("Forward horizon (days)", min_value=1, max_value=10, value=2, step=1)
    spike_threshold = st.slider("VIX spike threshold (%)", 20, 100, 20, step=5)
    rsi_thresh = st.slider("Oversold RSI threshold", 10, 40, 30, step=1)
    ma_window = st.selectbox("Regime MA window", [100, 150, 200], index=2)
    decade_filter = st.multiselect("Decades to include", [], help="Filled after data loads")

# ------------------------------- Data --------------------------------------
df = load_history(start=str(start_date))
# allow custom forward horizon for returns
df[f"spx_next{fwd_days}"] = df["spx_close"].shift(-fwd_days)
df[f"spx_fwd{fwd_days}_ret"] = (df[f"spx_next{fwd_days}"] / df["spx_close"] - 1.0) * 100.0

# regime by custom MA window
df[f"spx_{ma_window}dma"] = df["spx_close"].rolling(ma_window).mean()
df["regime"] = np.where(df["spx_close"] >= df[f"spx_{ma_window}dma"], "Bull", "Bear")

# spike flags and features
events = df.copy()
events["vix_base"] = events["vix_close"].shift(1)
events["vix_spike"] = events["vix_pctchg"] >= (spike_threshold / 100.0)
events = events[events["vix_spike"]].dropna(subset=["vix_base"])

# add engineered columns
events["vix_base_bucket"] = events["vix_base"].apply(bucket_vix_base)
events["spike_mag"] = events["vix_pctchg"].apply(bucket_spike_mag)
events["oversold"] = np.where(events["rsi14"] <= rsi_thresh, "Oversold", "Not Oversold")
events["month_name"] = events.index.month_name()
events["is_october"] = np.where(events.index.month == 10, "October", "Other Months")
events["decade"] = [decade_label(d) for d in events.index]

# decade filter control
all_decades = sorted(events["decade"].unique().tolist())
with st.sidebar:
    if not decade_filter:
        st.session_state["__decades"] = all_decades
    chosen_decades = st.multiselect("Decades to include", options=all_decades, default=all_decades, key="__decades")

events = events[events["decade"].isin(chosen_decades)]

# ------------------------------- Summary top row ---------------------------
left, right = st.columns([2,1])
with left:
    st.subheader("Key Findings")
    msg = []
    msg.append(f"Sample size: {len(events)} spike events where VIX rose at least {spike_threshold}% day over day.")
    best_bucket = (
        events.groupby("vix_base_bucket")[f"spx_fwd{fwd_days}_ret"].apply(winrate).sort_values(ascending=False)
    )
    if len(best_bucket) > 0 and not best_bucket.isna().all():
        msg.append(f"Best VIX base bucket by {fwd_days}-day win rate: {best_bucket.index[0]} at {best_bucket.iloc[0]:.1f}%.")
    best_mag = (
        events.groupby("spike_mag")[f"spx_fwd{fwd_days}_ret"].apply(winrate).sort_values(ascending=False)
    )
    if len(best_mag) > 0 and not best_mag.isna().all():
        msg.append(f"Spike magnitude favors: {best_mag.index[0]} at {best_mag.iloc[0]:.1f}% win rate.")
    regime_stats = events.groupby("regime")[f"spx_fwd{fwd_days}_ret"].apply(winrate)
    if "Bull" in regime_stats and "Bear" in regime_stats:
        msg.append(f"Regime effect: Bull {regime_stats['Bull']:.1f}% vs Bear {regime_stats['Bear']:.1f}%.")
    oversold_stats = events.groupby("oversold")[f"spx_fwd{fwd_days}_ret"].apply(winrate)
    if "Oversold" in oversold_stats and "Not Oversold" in oversold_stats:
        msg.append(f"Oversold helps: {oversold_stats['Oversold']:.1f}% vs {oversold_stats['Not Oversold']:.1f}%.")
    seasonality_stats = events.groupby("is_october")[f"spx_fwd{fwd_days}_ret"].apply(winrate)
    if "October" in seasonality_stats and "Other Months" in seasonality_stats:
        msg.append(f"Seasonality: October {seasonality_stats['October']:.1f}% vs Other {seasonality_stats['Other Months']:.1f}%.")

    card_box("<br>".join(msg))

with right:
    st.subheader("Filters")
    card_box(
        f"""
        <b>Spike threshold:</b> {spike_threshold}%<br>
        <b>Forward horizon:</b> {fwd_days} trading days<br>
        <b>RSI oversold:</b> ≤ {rsi_thresh}<br>
        <b>Regime MA:</b> {ma_window}-day<br>
        <b>Decades:</b> {', '.join(chosen_decades)}
        """
    )

# ------------------------------- Plots grid --------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.subplots_adjust(wspace=0.35, hspace=0.45)

# 1) Win rate by VIX base bucket
base_wr = events.groupby("vix_base_bucket")[f"spx_fwd{fwd_days}_ret"].apply(winrate).reindex(
    ["0-12", "12-16", "16-20", "20-24", "24-30", "30+"]
)
barplot(
    axes[0,0],
    base_wr.index.tolist(),
    base_wr.values.astype(float),
    PASTELS[:6],
    "Win Rate by VIX Base Level (Granular)"
)

# 2) Win rate by spike magnitude
mag_wr = events.groupby("spike_mag")[f"spx_fwd{fwd_days}_ret"].apply(winrate).reindex(
    ["Moderate (20-30%)", "Large (30-50%)", "Extreme (50%+)"]
)
barplot(
    axes[0,1],
    mag_wr.index.tolist(),
    mag_wr.values.astype(float),
    PASTELS[6:9],
    "Win Rate by Spike Magnitude"
)

# 3) Win rate by regime
reg_wr = events.groupby("regime")[f"spx_fwd{fwd_days}_ret"].apply(winrate).reindex(["Bull", "Bear"])
barplot(
    axes[0,2],
    reg_wr.index.tolist(),
    reg_wr.values.astype(float),
    [PASTELS[0], PASTELS[2]],
    "Win Rate by Market Regime"
)

# 4) Scatter: VIX base vs forward returns (colored by decade)
ax = axes[1,0]
for i, dec in enumerate(sorted(events["decade"].unique())):
    e = events[events["decade"] == dec]
    ax.scatter(e["vix_base"], e[f"spx_fwd{fwd_days}_ret"], s=28, alpha=0.85, label=dec, color=PASTELS[i % len(PASTELS)], edgecolors=BAR_EDGE, linewidths=0.3)
ax.axhline(0, color="#888888", linewidth=1)
ax.set_title("VIX Base Level vs Forward Returns", color=TEXT_COLOR, fontsize=12, pad=8)
ax.set_xlabel("VIX Level Before Spike", color=TEXT_COLOR)
ax.set_ylabel(f"SPX {fwd_days}-Day Return (%)", color=TEXT_COLOR)
ax.grid(color=GRID_COLOR, linewidth=0.6)
ax.legend(fontsize=8, frameon=False, ncol=2)

# 5) Win rate by oversold
ov_wr = events.groupby("oversold")[f"spx_fwd{fwd_days}_ret"].apply(winrate).reindex(["Oversold", "Not Oversold"])
barplot(
    axes[1,1],
    ov_wr.index.tolist(),
    ov_wr.values.astype(float),
    [PASTELS[3], PASTELS[1]],
    "Win Rate by Oversold Conditions"
)

# 6) Win rate by seasonality
sea_wr = events.groupby("is_october")[f"spx_fwd{fwd_days}_ret"].apply(winrate).reindex(["October", "Other Months"])
barplot(
    axes[1,2],
    sea_wr.index.tolist(),
    sea_wr.values.astype(float),
    [PASTELS[4], PASTELS[5]],
    "Win Rate by Seasonality"
)

st.pyplot(fig, clear_figure=True)

# ------------------------------- Dynamic commentary ------------------------
st.subheader("Dynamic Commentary")

def latest_context_box():
    # Find the latest spike event under current threshold
    latest = events.iloc[-1] if not events.empty else None
    if latest is None:
        card_box("No qualifying VIX spike events in the filtered sample.")
        return
    dt = latest.name.strftime("%Y-%m-%d")
    vix_now = latest["vix_close"]
    vix_base = latest["vix_base"]
    vix_spike = latest["vix_pctchg"] * 100.0
    spx_now = latest["spx_close"]
    rsi_now = latest["rsi14"]
    regime_now = latest["regime"]
    bucket = latest["vix_base_bucket"]
    magcat = latest["spike_mag"]
    over = latest["oversold"]

    # Historical comps matching the same triplet of conditions
    mask = (
        (events["vix_base_bucket"] == bucket) &
        (events["spike_mag"] == magcat) &
        (events["regime"] == regime_now)
    )
    comps = events[mask]
    wr = winrate(comps[f"spx_fwd{fwd_days}_ret"])
    avg = comps[f"spx_fwd{fwd_days}_ret"].mean() if len(comps) else np.nan
    n = len(comps)

    text = f"""
    <b>Latest qualifying event:</b> {dt}<br>
    VIX {vix_now:.2f} (base {vix_base:.2f}), spike {vix_spike:.1f}%<br>
    SPX {spx_now:.2f}, RSI14 {rsi_now:.1f} ({over}), Regime {regime_now}<br>
    VIX base bucket {bucket}, Spike {magcat}<br><br>
    <b>Historical comps:</b> same bucket, magnitude, and regime<br>
    Count {n}, win rate {wr:.1f}% over {fwd_days} days, avg return {avg:.2f}%<br><br>
    <i>Heuristic:</i> Best setup tends to be low base VIX buckets with Large spikes in Bull regimes. 
    Extreme spikes carry higher dispersion. Oversold filters improve average outcomes, but sample sizes drop.
    """
    card_box(text)

latest_context_box()

# ------------------------------- Table of events (optional) ---------------
with st.expander("Show events table"):
    show_cols = [
        "vix_close", "vix_base", "vix_pctchg", "spx_close", f"spx_fwd{fwd_days}_ret",
        "vix_base_bucket", "spike_mag", "regime", "rsi14", "oversold", "is_october", "decade"
    ]
    tbl = events[show_cols].copy()
    tbl.rename(columns={
        "vix_close":"VIX",
        "vix_base":"VIX_Base",
        "vix_pctchg":"VIX_Spike_%", 
        "spx_close":"SPX",
        f"spx_fwd{fwd_days}_ret":f"SPX_Fwd{fwd_days}_Ret_%", 
        "rsi14":"RSI14"
    }, inplace=True)
    tbl["VIX_Spike_%"] = (tbl["VIX_Spike_%"] * 100.0).round(2)
    st.dataframe(tbl.round(2), use_container_width=True)

# ------------------------------- Footer -----------------------------------
st.caption("ADFM Analytics Platform · VIX 20%+ Spike Deep Dive · Data source: Yahoo Finance")
