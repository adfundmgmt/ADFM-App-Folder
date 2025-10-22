# vix_spike_deep_dive.py
# ADFM Analytics Platform, VIX 20%+ Spike Deep Dive
# Light theme, pastel palette, matplotlib only.

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import streamlit as st

# ------------------------------- Page config -------------------------------
st.set_page_config(page_title="ADFM | VIX 20%+ Spike Deep Dive", layout="wide")
plt.style.use("default")

# ------------------------------- Pastel palette ----------------------------
PASTELS = [
    "#A8DADC", "#F4A261", "#90BE6D", "#FFCC99", "#BDE0FE",
    "#CDB4DB", "#FFD6A5", "#E2F0CB", "#F1C0E8", "#B9FBC0"
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

    # RSI(14) for oversold context
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
    bins = [0, 12, 16, 20, 24, 30, 99]
    labels = ["0-12", "12-16", "16-20", "20-24", "24-30", "30+"]
    return pd.cut([x], bins=bins, labels=labels, right=False)[0]

def bucket_spike_mag(pct):
    # pct is a fraction, compare to thresholds
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
    series = pd.to_numeric(series, errors="coerce").dropna()
    return 100.0 * (series > 0).mean() if len(series) else np.nan

def pct_bands(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        return np.nan, np.nan, np.nan
    return np.percentile(series, [10, 50, 90])

def mean_abs_deviation(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        return np.nan
    return np.mean(np.abs(series - np.mean(series)))

def barplot(ax, categories, values, colors, title, ylabel="2-Day Win Rate (%)"):
    ax.bar(categories, values, color=colors, edgecolor=BAR_EDGE)
    ax.axhline(50, linestyle="--", linewidth=1, color="#888888")
    ax.set_title(title, color=TEXT_COLOR, fontsize=12, pad=8)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    ymax = np.nanmax(values) if len(values) else 0
    ax.set_ylim(0, max(70, (ymax if np.isfinite(ymax) else 60) + 10))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", color=TEXT_COLOR, fontsize=10)

def card_box(text):
    st.markdown(
        f"""
        <div style="border:1px solid #e0e0e0; border-radius:10px; padding:14px; background:#fafafa; color:{TEXT_COLOR}; font-size:14px; line-height:1.35;">
        {text}
        </div>
        """,
        unsafe_allow_html=True
    )

def stats_row(label, wr, avg, med, mad, n):
    wr_txt  = f"{wr:.1f}%" if pd.notna(wr) else "NA"
    avg_txt = f"{avg:.2f}%" if pd.notna(avg) else "NA"
    med_txt = f"{med:.2f}%" if pd.notna(med) else "NA"
    mad_txt = f"{mad:.2f}%" if pd.notna(mad) else "NA"
    return f"<b>{label}</b>: WR {wr_txt}, Avg {avg_txt}, Med {med_txt}, Disp {mad_txt}, N={n}"

# ------------------------------- Sidebar -----------------------------------
st.title("VIX 20%+ Spike Deep Dive • ADFM")

with st.sidebar:
    st.header("Controls")
    start_date = st.date_input("History start", value=datetime(1990, 1, 1))
    fwd_days = st.number_input("Forward horizon (days)", min_value=1, max_value=10, value=2, step=1)
    spike_threshold = st.slider("VIX spike threshold (%)", 20, 100, 20, step=5)
    rsi_thresh = st.slider("Oversold RSI threshold", 10, 40, 30, step=1)
    ma_window = st.selectbox("Regime MA window", [100, 150, 200], index=2)

# ------------------------------- Data --------------------------------------
df = load_history(start=str(start_date))

# dynamic forward returns
df[f"spx_next{fwd_days}"] = df["spx_close"].shift(-fwd_days)
df[f"spx_fwd{fwd_days}_ret"] = (df[f"spx_next{fwd_days}"] / df["spx_close"] - 1.0) * 100.0

# regime by chosen MA
df[f"spx_{ma_window}dma"] = df["spx_close"].rolling(ma_window).mean()
df["regime"] = np.where(df["spx_close"] >= df[f"spx_{ma_window}dma"], "Bull", "Bear")

# spike features
events = df.copy()
events["vix_base"] = events["vix_close"].shift(1)
events["vix_spike"] = events["vix_pctchg"] >= (spike_threshold / 100.0)
events = events[events["vix_spike"]].dropna(subset=["vix_base"])

events["vix_base_bucket"] = events["vix_base"].apply(bucket_vix_base)
events["spike_mag"] = events["vix_pctchg"].apply(bucket_spike_mag)
events["oversold"] = np.where(events["rsi14"] <= rsi_thresh, "Oversold", "Not Oversold")
events["decade"] = [decade_label(d) for d in events.index]

# unified decade control
all_decades = sorted(events["decade"].unique().tolist())
decade_filter = st.sidebar.multiselect("Decades to include", options=all_decades, default=all_decades)
events = events[events["decade"].isin(decade_filter)]

# ------------------------------- Latest setup and comps ---------------------
latest = events.iloc[-1] if not events.empty else None
if latest is not None:
    bucket = latest["vix_base_bucket"]
    magcat = latest["spike_mag"]
    regime_now = latest["regime"]
    comps_mask = (
        (events["vix_base_bucket"] == bucket) &
        (events["spike_mag"] == magcat) &
        (events["regime"] == regime_now)
    )
    comps = events[comps_mask].copy()
else:
    comps = pd.DataFrame(columns=events.columns)

# ------------------------------- Summary top row ---------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Key Findings")

    # overall stats
    overall = events[f"spx_fwd{fwd_days}_ret"]
    wr_all = winrate(overall)
    avg_all = overall.mean() if len(overall) else np.nan
    med_all = overall.median() if len(overall) else np.nan
    mad_all = mean_abs_deviation(overall)
    n_all = len(overall)

    # best buckets
    bucket_wr = events.groupby("vix_base_bucket")[f"spx_fwd{fwd_days}_ret"].apply(winrate).sort_values(ascending=False)
    best_bucket_label = bucket_wr.index[0] if len(bucket_wr) else "NA"
    best_bucket_wr = bucket_wr.iloc[0] if len(bucket_wr) else np.nan

    mag_wr = events.groupby("spike_mag")[f"spx_fwd{fwd_days}_ret"].apply(winrate).sort_values(ascending=False)
    best_mag_label = mag_wr.index[0] if len(mag_wr) else "NA"
    best_mag_wr = mag_wr.iloc[0] if len(mag_wr) else np.nan

    reg_wr = events.groupby("regime")[f"spx_fwd{fwd_days}_ret"].apply(winrate)
    bull_wr = reg_wr.get("Bull", np.nan)
    bear_wr = reg_wr.get("Bear", np.nan)

    ov_wr = events.groupby("oversold")[f"spx_fwd{fwd_days}_ret"].apply(winrate)
    ov_yes = ov_wr.get("Oversold", np.nan)
    ov_no = ov_wr.get("Not Oversold", np.nan)

    p10, p50, p90 = pct_bands(overall)

    msg = []
    msg.append(stats_row("All spikes", wr_all, avg_all, med_all, mad_all, n_all))
    if pd.notna(best_bucket_wr):
        msg.append(f"<b>Best base bucket</b>: {best_bucket_label}, WR {best_bucket_wr:.1f}%")
    if pd.notna(best_mag_wr):
        msg.append(f"<b>Favored magnitude</b>: {best_mag_label}, WR {best_mag_wr:.1f}%")
    if pd.notna(bull_wr) and pd.notna(bear_wr):
        msg.append(f"<b>Regime split</b>: Bull {bull_wr:.1f}% vs Bear {bear_wr:.1f}%")
    if pd.notna(ov_yes) and pd.notna(ov_no):
        msg.append(f"<b>RSI filter</b>: Oversold {ov_yes:.1f}% vs Not {ov_no:.1f}%")
    if pd.notna(p10):
        msg.append(f"<b>Outcome bands</b>: p10 {p10:.2f}%, p50 {p50:.2f}%, p90 {p90:.2f}%")
    card_box("<br>".join(msg))

with right:
    st.subheader("Filters")
    card_box(
        f"""
        <b>Spike threshold</b>: {spike_threshold}%<br>
        <b>Forward horizon</b>: {fwd_days} trading days<br>
        <b>RSI oversold</b>: ≤ {rsi_thresh}<br>
        <b>Regime MA</b>: {ma_window}-day<br>
        <b>Decades</b>: {', '.join(decade_filter) if decade_filter else 'None'}
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
    axes[0, 0],
    base_wr.index.tolist(),
    base_wr.values.astype(float),
    PASTELS[:6],
    "Win Rate by VIX Base Level"
)

# 2) Win rate by spike magnitude
mag_wr = events.groupby("spike_mag")[f"spx_fwd{fwd_days}_ret"].apply(winrate).reindex(
    ["Moderate (20-30%)", "Large (30-50%)", "Extreme (50%+)"]
)
barplot(
    axes[0, 1],
    mag_wr.index.tolist(),
    mag_wr.values.astype(float),
    PASTELS[6:9],
    "Win Rate by Spike Magnitude"
)

# 3) Win rate by regime
reg_wr = events.groupby("regime")[f"spx_fwd{fwd_days}_ret"].apply(winrate).reindex(["Bull", "Bear"])
barplot(
    axes[0, 2],
    reg_wr.index.tolist(),
    reg_wr.values.astype(float),
    [PASTELS[0], PASTELS[2]],
    "Win Rate by Market Regime"
)

# 4) Scatter: VIX base vs forward returns, colored by decade
ax = axes[1, 0]
for i, dec in enumerate(sorted(events["decade"].unique())):
    e = events[events["decade"] == dec]
    ax.scatter(
        e["vix_base"], e[f"spx_fwd{fwd_days}_ret"],
        s=28, alpha=0.85, label=dec, color=PASTELS[i % len(PASTELS)],
        edgecolors=BAR_EDGE, linewidths=0.3
    )
ax.axhline(0, color="#888888", linewidth=1)
ax.set_title("VIX Base Level vs Forward Returns", color=TEXT_COLOR, fontsize=12, pad=8)
ax.set_xlabel("VIX Level Before Spike", color=TEXT_COLOR)
ax.set_ylabel(f"SPX {fwd_days}-Day Return (%)", color=TEXT_COLOR)
ax.grid(color=GRID_COLOR, linewidth=0.6)
ax.legend(fontsize=8, frameon=False, ncol=2)

# 5) Oversold split
ov_wr = events.groupby("oversold")[f"spx_fwd{fwd_days}_ret"].apply(winrate).reindex(["Oversold", "Not Oversold"])
barplot(
    axes[1, 1],
    ov_wr.index.tolist(),
    ov_wr.values.astype(float),
    [PASTELS[3], PASTELS[1]],
    "Win Rate by RSI Oversold"
)

# 6) Setup Distribution: histogram on the grid (FIX)
axd = axes[1, 2]
axd.set_title("Setup Distribution", color=TEXT_COLOR, fontsize=12, pad=8)
axd.set_xlabel(f"SPX {fwd_days}-Day Return (%)", color=TEXT_COLOR)
axd.set_ylabel("Frequency", color=TEXT_COLOR)
axd.grid(color=GRID_COLOR, linewidth=0.6)

vals = comps[f"spx_fwd{fwd_days}_ret"].dropna().values if not comps.empty else np.array([])
if vals.size > 0:
    axd.hist(vals, bins=min(20, max(8, int(np.sqrt(len(vals))))), edgecolor=BAR_EDGE)
    axd.axvline(0, color="#888888", linewidth=1)
else:
    axd.text(
        0.5, 0.5,
        "No matching comps for latest setup\nunder current filters",
        ha="center", va="center", transform=axd.transAxes, color="#555555"
    )

st.pyplot(fig, clear_figure=True)

# ------------------------------- Dynamic commentary ------------------------
st.subheader("Dynamic Commentary")

def latest_context_box():
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

    wr = winrate(comps[f"spx_fwd{fwd_days}_ret"]) if not comps.empty else np.nan
    avg = comps[f"spx_fwd{fwd_days}_ret"].mean() if not comps.empty else np.nan
    med = comps[f"spx_fwd{fwd_days}_ret"].median() if not comps.empty else np.nan
    mad = mean_abs_deviation(comps[f"spx_fwd{fwd_days}_ret"]) if not comps.empty else np.nan
    p10, p50, p90 = pct_bands(comps[f"spx_fwd{fwd_days}_ret"]) if not comps.empty else (np.nan, np.nan, np.nan)
    n = len(comps) if not comps.empty else 0

    edge_score = np.nan
    if pd.notna(wr) and pd.notna(med) and pd.notna(mad) and mad > 0:
        edge_score = (wr - 50.0) * (med / mad)

    text = f"""
    <b>Latest event</b>: {dt}<br>
    VIX {vix_now:.2f} (base {vix_base:.2f}), spike {vix_spike:.1f}%<br>
    SPX {spx_now:.2f}, RSI14 {rsi_now:.1f} ({over}), Regime {regime_now}<br>
    Setup: base {bucket}, magnitude {magcat}<br><br>
    <b>Setup stats</b> (same bucket, magnitude, regime)<br>
    WR {wr:.1f}%, Avg {avg:.2f}%, Med {med:.2f}%, Disp {mad:.2f}%, N={n}<br>
    Bands: p10 {p10:.2f}%, p50 {p50:.2f}%, p90 {p90:.2f}%<br>
    Edge score: {edge_score:.2f} (higher is stronger, scale is relative)<br><br>
    <i>Read</i>: Low base buckets with Large spikes in Bull regimes tend to work best. Extreme spikes widen tails, RSI oversold improves skew but reduces N.
    """
    card_box(text)

    if n > 0:
        show_cols = [
            "vix_close", "vix_base", "vix_pctchg", "spx_close", f"spx_fwd{fwd_days}_ret", "rsi14"
        ]
        analogs = (
            comps[show_cols]
            .rename(columns={
                "vix_close": "VIX",
                "vix_base": "VIX_Base",
                "vix_pctchg": "VIX_Spike_Frac",
                "spx_close": "SPX",
                f"spx_fwd{fwd_days}_ret": f"SPX_Fwd{fwd_days}_Ret_%",
                "rsi14": "RSI14"
            })
            .copy()
        )
        analogs["VIX_Spike_%"] = (analogs["VIX_Spike_Frac"] * 100.0).round(2)
        analogs.drop(columns=["VIX_Spike_Frac"], inplace=True)
        analogs = analogs.round(2).sort_index(ascending=False).head(10)
        st.markdown("**Nearest analogs, same setup, last 10 occurrences**")
        st.dataframe(analogs, use_container_width=True)

latest_context_box()

# ------------------------------- Events table ------------------------------
with st.expander("Show events table"):
    show_cols = [
        "vix_close", "vix_base", "vix_pctchg", "spx_close", f"spx_fwd{fwd_days}_ret",
        "vix_base_bucket", "spike_mag", "regime", "rsi14", "oversold", "decade"
    ]
    tbl = events[show_cols].copy()
    tbl.rename(columns={
        "vix_close": "VIX",
        "vix_base": "VIX_Base",
        "vix_pctchg": "VIX_Spike_Fr",
        "spx_close": "SPX",
        f"spx_fwd{fwd_days}_ret": f"SPX_Fwd{fwd_days}_Ret_%",
        "rsi14": "RSI14"
    }, inplace=True)
    tbl["VIX_Spike_%"] = (tbl["VIX_Spike_Fr"] * 100.0).round(2)
    tbl.drop(columns=["VIX_Spike_Fr"], inplace=True)
    st.dataframe(tbl.round(2), use_container_width=True)

# ------------------------------- Footer ------------------------------------
st.caption("ADFM Analytics Platform · VIX 20%+ Spike Deep Dive · Data source: Yahoo Finance")
