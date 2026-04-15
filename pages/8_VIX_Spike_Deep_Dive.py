# vix_spike_deep_dive.py
# ADFM Analytics Platform | VIX Spike Deep Dive
# Captures both VIX spikes up and VIX drops down by absolute % threshold

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
import streamlit as st

# ------------------------------- Page config -------------------------------
st.set_page_config(page_title="ADFM | VIX Spike Deep Dive", layout="wide")
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
@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_history(start="1990-01-01"):
    vix = yf.download("^VIX", start=start, auto_adjust=False, progress=False)
    spx = yf.download("^GSPC", start=start, auto_adjust=False, progress=False)

    vix = vix.rename(columns=str.lower)
    spx = spx.rename(columns=str.lower)

    df = pd.DataFrame(index=vix.index)
    df["vix_close"] = vix["close"]
    df["vix_prev"] = df["vix_close"].shift(1)
    df["vix_pctchg"] = (df["vix_close"] / df["vix_prev"] - 1.0)
    df["vix_abs_pctchg"] = df["vix_pctchg"].abs()
    df["spx_close"] = spx["close"].reindex(df.index).ffill()
    df["rsi14"] = compute_rsi(df["spx_close"], 14)
    return df.dropna(subset=["vix_close", "spx_close"])


def compute_rsi(series, n=14):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1 / n, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def bucket_vix_base(x, bins, labels):
    if pd.isna(x):
        return np.nan
    try:
        return pd.cut([x], bins=bins, labels=labels, right=False, include_lowest=True)[0]
    except Exception:
        return np.nan


def bucket_move_mag(abs_pct):
    if abs_pct < 0.30:
        return "Moderate (25-30%)"
    elif abs_pct < 0.50:
        return "Large (30-50%)"
    else:
        return "Extreme (50%+)"


def move_type_label(pct):
    return "Spike Up" if pct >= 0 else "Spike Down"


def decade_label(dt):
    y = dt.year
    return f"{y // 10 * 10}s"


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


def fmt_pct(x, digits=1):
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}%"


def card_box(inner_html):
    st.markdown(
        f"""
        <div style="border:1px solid #e0e0e0; border-radius:10px; padding:14px; background:#fafafa; color:{TEXT_COLOR}; font-size:14px; line-height:1.35;">
          {inner_html}
        </div>
        """.strip(),
        unsafe_allow_html=True
    )


def make_vix_bins_and_labels(vix_min, vix_1, vix_2, vix_3, vix_4, vix_5, vix_max=99):
    raw_edges = [vix_min, vix_1, vix_2, vix_3, vix_4, vix_5, vix_max]
    bins = [raw_edges[0]]
    for edge in raw_edges[1:]:
        bins.append(max(edge, bins[-1] + 1))

    labels = [
        f"{bins[0]}-{bins[1]}",
        f"{bins[1]}-{bins[2]}",
        f"{bins[2]}-{bins[3]}",
        f"{bins[3]}-{bins[4]}",
        f"{bins[4]}-{bins[5]}",
        f"{bins[5]}+",
    ]
    return bins, labels


# ------------------------------- Sidebar -----------------------------------
st.title("VIX Spike Deep Dive")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Purpose: Evaluate post-event equity behavior after large VIX moves.

        What it covers
        • Forward SPX return distributions after configurable VIX moves
        • Hit-rate, tail-risk, and regime-conditioned outcome diagnostics
        • Event-study and heatmap views to support tactical timing and sizing

        Data source
        • Yahoo Finance VIX and SPX history
        """
    )
    st.divider()

    st.header("Controls")
    start_date = st.date_input("History start", value=datetime(1990, 1, 1))
    fwd_days = st.number_input("Forward horizon (days)", min_value=1, max_value=30, value=2, step=1)
    spike_threshold = st.slider("Absolute VIX move threshold (%)", 10, 100, 25, step=5)
    rsi_thresh = st.slider("Oversold RSI threshold", 10, 40, 30, step=1)
    ma_window = st.selectbox("Regime MA window", [100, 150, 200], index=2)
    show_legend = st.checkbox("Show legend", value=False)

    st.divider()
    st.subheader("VIX Base Buckets")
    st.caption("Adjust the starting VIX regime buckets used in the analysis.")

    vix_min = st.slider("Min VIX", 0, 20, 0)
    vix_1 = st.slider("Bucket 1 upper", 8, 20, 12)
    vix_2 = st.slider("Bucket 2 upper", 12, 25, 16)
    vix_3 = st.slider("Bucket 3 upper", 15, 30, 20)
    vix_4 = st.slider("Bucket 4 upper", 20, 35, 24)
    vix_5 = st.slider("Bucket 5 upper", 25, 40, 30)

vix_bins, vix_labels = make_vix_bins_and_labels(
    vix_min=vix_min,
    vix_1=vix_1,
    vix_2=vix_2,
    vix_3=vix_3,
    vix_4=vix_4,
    vix_5=vix_5,
    vix_max=99
)

# ------------------------------- Data --------------------------------------
df = load_history(start=str(start_date))

df[f"spx_next{fwd_days}"] = df["spx_close"].shift(-fwd_days)
df[f"spx_fwd{fwd_days}_ret"] = (df[f"spx_next{fwd_days}"] / df["spx_close"] - 1.0) * 100.0

df[f"spx_{ma_window}dma"] = df["spx_close"].rolling(ma_window).mean()
df["regime"] = np.where(df["spx_close"] >= df[f"spx_{ma_window}dma"], "Bull", "Bear")

events = df.copy()
events["vix_base"] = events["vix_close"].shift(1)
events["qualifying_move"] = events["vix_abs_pctchg"] >= (spike_threshold / 100.0)
events = events[events["qualifying_move"]].dropna(subset=["vix_base"]).copy()

if not events.empty:
    events["vix_base_bucket"] = events["vix_base"].apply(
        lambda x: bucket_vix_base(x, vix_bins, vix_labels)
    )
    events["move_type"] = events["vix_pctchg"].apply(move_type_label)
    events["move_mag"] = events["vix_abs_pctchg"].apply(bucket_move_mag)
    events["oversold"] = np.where(events["rsi14"] <= rsi_thresh, "Oversold", "Not Oversold")
    events["decade"] = [decade_label(d) for d in events.index]
else:
    events = pd.DataFrame(
        columns=[
            "vix_base_bucket", "move_type", "move_mag", "oversold",
            "decade", "regime", f"spx_fwd{fwd_days}_ret"
        ]
    )

all_decades = sorted(events["decade"].unique().tolist()) if not events.empty else []
decade_filter = st.sidebar.multiselect("Decades to include", options=all_decades, default=all_decades)
if decade_filter:
    events = events[events["decade"].isin(decade_filter)]

latest = events.iloc[-1] if not events.empty else None
if latest is not None:
    bucket_now = latest["vix_base_bucket"]
    move_type_now = latest["move_type"]
    mag_now = latest["move_mag"]
    regime_now = latest["regime"]
    rsi_state = latest["oversold"]

    comps_mask = (
        (events["vix_base_bucket"] == bucket_now) &
        (events["move_type"] == move_type_now) &
        (events["move_mag"] == mag_now) &
        (events["regime"] == regime_now)
    )
    comps = events[comps_mask].copy()
else:
    bucket_now = move_type_now = mag_now = regime_now = rsi_state = None
    comps = pd.DataFrame(columns=events.columns)

# ------------------------------- Stats for commentary ----------------------
overall = events[f"spx_fwd{fwd_days}_ret"] if not events.empty else pd.Series(dtype=float)
bucket_wr = (
    events.groupby("vix_base_bucket")[f"spx_fwd{fwd_days}_ret"].apply(winrate)
    if not events.empty else pd.Series(dtype=float)
)
move_type_wr = (
    events.groupby("move_type")[f"spx_fwd{fwd_days}_ret"].apply(winrate)
    if not events.empty else pd.Series(dtype=float)
)
mag_wr = (
    events.groupby("move_mag")[f"spx_fwd{fwd_days}_ret"].apply(winrate)
    if not events.empty else pd.Series(dtype=float)
)
reg_wr = (
    events.groupby("regime")[f"spx_fwd{fwd_days}_ret"].apply(winrate)
    if not events.empty else pd.Series(dtype=float)
)
ov_wr = (
    events.groupby("oversold")[f"spx_fwd{fwd_days}_ret"].apply(winrate)
    if not events.empty else pd.Series(dtype=float)
)

p10, p50, p90 = pct_bands(overall) if len(overall) else (np.nan, np.nan, np.nan)

wr_all = winrate(overall)
avg_all = overall.mean() if len(overall) else np.nan
med_all = overall.median() if len(overall) else np.nan
mad_all = mean_abs_deviation(overall)
n_all = len(overall)

best_bucket_label = bucket_wr.idxmax() if len(bucket_wr) else None
best_bucket_wr = float(bucket_wr.max()) if len(bucket_wr) else np.nan
best_mag_label = mag_wr.idxmax() if len(mag_wr) else None
best_mag_wr = float(mag_wr.max()) if len(mag_wr) else np.nan
up_wr = move_type_wr.get("Spike Up", np.nan)
down_wr = move_type_wr.get("Spike Down", np.nan)
bull_wr = reg_wr.get("Bull", np.nan)
bear_wr = reg_wr.get("Bear", np.nan)
ov_yes = ov_wr.get("Oversold", np.nan)
ov_no = ov_wr.get("Not Oversold", np.nan)

# ------------------------------- Commentary engine -------------------------
def generate_commentary(ctx):
    wr_all, med, avg, n = ctx["wr_all"], ctx["med_all"], ctx["avg_all"], ctx["n_all"]
    p10, p50, p90 = ctx["p10"], ctx["p50"], ctx["p90"]
    best_bucket_label, best_bucket_wr = ctx["best_bucket_label"], ctx["best_bucket_wr"]
    best_mag_label, best_mag_wr = ctx["best_mag_label"], ctx["best_mag_wr"]
    up_wr, down_wr = ctx["up_wr"], ctx["down_wr"]
    bull_wr, bear_wr = ctx["bull_wr"], ctx["bear_wr"]
    ov_yes, ov_no = ctx["ov_yes"], ctx["ov_no"]
    threshold = ctx["spike_threshold"]
    fwd_days = ctx["fwd_days"]
    bucket_now = ctx["bucket_now"]
    move_type_now = ctx["move_type_now"]
    mag_now = ctx["mag_now"]
    regime_now = ctx["regime_now"]
    rsi_state = ctx["rsi_state"]
    ma_window = ctx["ma_window"]
    show_legend = ctx["show_legend"]
    current_order_base = ctx["order_base"]

    bucket_idx = current_order_base.index(str(bucket_now)) if bucket_now in current_order_base else 0
    regime_idx = 0 if regime_now == "Bull" else 1
    dir_idx = 0 if move_type_now == "Spike Up" else 1
    rsi_idx = 0 if rsi_state == "Oversold" else 1

    if move_type_now == "Spike Up":
        headline_bank = [
            "Vol shock higher, relief odds depend on the tape.",
            "Panic spike registered, now the question is whether buyers absorb it.",
            "Upside VIX shock, reflex bounce setup depends on trend support.",
            "Fear surged, now the tape has to prove it can stabilize.",
        ]
        why = f"Whether a ≥{threshold}% upside VIX shock creates a buyable short-term reset."
    else:
        headline_bank = [
            "Vol collapsed, calm can persist or fade fast.",
            "Fear came out of the tape, now the question is whether complacency bites.",
            "Downside VIX shock, grind higher can continue but reward may compress.",
            "Stress unwound quickly, now follow-through matters more than relief.",
        ]
        why = f"Whether a ≥{threshold}% downside VIX move signals cleaner follow-through or near-term exhaustion."

    headline = headline_bank[(bucket_idx + regime_idx + rsi_idx) % len(headline_bank)]

    if regime_idx == 0:
        regime_line = f"Above the {ma_window}-DMA, hit rate improves to {fmt_pct(bull_wr, 1)}."
    else:
        regime_line = f"Below the {ma_window}-DMA, hit rate slips to {fmt_pct(bear_wr, 1)}."

    direction_line = f"Direction split, Spike Up WR {fmt_pct(up_wr,1)} vs Spike Down WR {fmt_pct(down_wr,1)}."

    if pd.notna(wr_all) and pd.notna(med):
        if wr_all >= 53 and med >= 0:
            conclusion_tail = "Setup supports tactical participation."
        elif 47 <= wr_all < 53:
            conclusion_tail = "Edge is mixed, execution matters."
        else:
            conclusion_tail = "Edge is weak, caution first."
    else:
        conclusion_tail = "Insufficient sample."

    drivers = []
    drivers.append(f"Base case, WR {fmt_pct(wr_all, 1)}, median {fmt_pct(med, 1)}, average {fmt_pct(avg, 1)}, N {n}.")
    drivers.append(direction_line)
    if pd.notna(best_bucket_wr) and best_bucket_label is not None:
        drivers.append(f"Base VIX {best_bucket_label} works best, WR {fmt_pct(best_bucket_wr, 1)}.")
    if pd.notna(best_mag_wr) and best_mag_label is not None:
        drivers.append(f"Move magnitude {best_mag_label}, WR {fmt_pct(best_mag_wr, 1)}.")
    drivers.append(regime_line)
    if pd.notna(ov_yes) and pd.notna(ov_no):
        drivers.append(f"RSI filter, Oversold {fmt_pct(ov_yes, 1)} vs Not {fmt_pct(ov_no, 1)}.")
    drivers.append(f"Expected range for {fwd_days} days, p10 {fmt_pct(p10)}, p50 {fmt_pct(p50)}, p90 {fmt_pct(p90)}.")

    body = (
        f'<div style="font-weight:700; margin-bottom:6px;">Conclusion</div>'
        f'<div>{headline} {conclusion_tail}</div>'
        f'<div style="font-weight:700; margin:10px 0 6px;">Why it matters</div>'
        f'<div>{why}</div>'
        f'<div style="font-weight:700; margin:10px 0 6px;">Key drivers</div>'
        f'<ul style="margin-top:4px; margin-bottom:4px;">'
        + "".join(f"<li>{d}</li>" for d in drivers) +
        "</ul>"
    )

    if show_legend:
        body += (
            '<hr style="border-top:1px solid #eee; margin:10px 0;">'
            '<div style="font-size:13px;">'
            "<b>Legend</b> WR win rate, Avg and Med forward SPX returns, N sample size. "
            "Base bucket is VIX level before the move. Move type separates VIX spikes up from drops down. "
            "Magnitude uses the absolute VIX move. Regime uses your DMA. RSI uses your slider. "
            "p10, p50, p90 are percentile bands of forward returns for the chosen horizon."
            "</div>"
        )

    lights = 0
    high_buckets = current_order_base[-2:] if len(current_order_base) >= 2 else current_order_base
    if bucket_now in high_buckets:
        lights += 1
    if mag_now in ["Large (30-50%)", "Extreme (50%+)"]:
        lights += 1
    if regime_now == "Bull":
        lights += 1
    if rsi_state == "Oversold" and pd.notna(ov_yes) and pd.notna(ov_no) and ov_yes >= ov_no:
        lights += 1

    return body, lights


ctx = {
    "wr_all": wr_all,
    "med_all": med_all,
    "avg_all": avg_all,
    "n_all": n_all,
    "p10": p10,
    "p50": p50,
    "p90": p90,
    "best_bucket_label": best_bucket_label,
    "best_bucket_wr": best_bucket_wr,
    "best_mag_label": best_mag_label,
    "best_mag_wr": best_mag_wr,
    "up_wr": up_wr,
    "down_wr": down_wr,
    "bull_wr": bull_wr,
    "bear_wr": bear_wr,
    "ov_yes": ov_yes,
    "ov_no": ov_no,
    "spike_threshold": spike_threshold,
    "fwd_days": fwd_days,
    "bucket_now": bucket_now,
    "move_type_now": move_type_now,
    "mag_now": mag_now,
    "regime_now": regime_now,
    "rsi_state": rsi_state,
    "ma_window": ma_window,
    "show_legend": show_legend,
    "order_base": vix_labels,
}
summary_html, lights = generate_commentary(ctx)

# ------------------------------- Top row -----------------------------------
col1, col2 = st.columns([1.8, 1])

with col1:
    st.subheader("Decision Box")
    card_box(summary_html)

with col2:
    st.subheader("Filters")
    card_box(
        f"""
        <b>Absolute move threshold</b>: {spike_threshold}%<br>
        <b>Forward horizon</b>: {fwd_days} trading days<br>
        <b>RSI oversold</b>: ≤ {rsi_thresh}<br>
        <b>Regime MA</b>: {ma_window}-day<br>
        <b>VIX buckets</b>: {", ".join(vix_labels)}<br>
        <b>Decades</b>: {", ".join(decade_filter) if decade_filter else "None"}<br>
        <b>Four lights</b>: {lights} / 4
        """.strip()
    )

# ------------------------------- Panels ------------------------------------
def safe_group_lists(g, col, order):
    data = []
    labels = []
    for k in order:
        if (not events.empty) and (k in g.groups):
            arr = pd.to_numeric(g.get_group(k)[col], errors="coerce").dropna().values
        else:
            arr = np.array([])
        if arr.size > 0:
            data.append(arr)
            labels.append(k)
    return data, labels


def ensure_fwd_cols(df_, horizons):
    for h in horizons:
        nxt = f"spx_next{h}"
        ret = f"spx_fwd{h}_ret"
        if ret not in df_.columns:
            df_[nxt] = df_["spx_close"].shift(-h)
            df_[ret] = (df_[nxt] / df_["spx_close"] - 1.0) * 100.0
    return df_


fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.subplots_adjust(wspace=0.35, hspace=0.45)

# 1) Boxplot by base bucket
ax1 = axes[0, 0]
order_base = vix_labels

if not events.empty:
    g = events.groupby("vix_base_bucket", observed=False)
    data, labels = safe_group_lists(g, f"spx_fwd{fwd_days}_ret", order_base)
else:
    data, labels = [], []

bp = ax1.boxplot(
    data if data else [np.array([np.nan])],
    tick_labels=labels if labels else ["no data"],
    patch_artist=True,
    notch=True,
    showfliers=False
)
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(PASTELS[i % len(PASTELS)])
    patch.set_edgecolor(BAR_EDGE)
for med in bp["medians"]:
    med.set_color("#444444")
ax1.axhline(0, color="#888888", linewidth=1)
ax1.set_title(f"Distribution by VIX Base, {fwd_days}-Day %", color=TEXT_COLOR, fontsize=12, pad=8)
ax1.grid(axis="y", color=GRID_COLOR, linewidth=0.6)

# 2) Heatmap by Base × Direction
ax2 = axes[0, 1]
order_direction = ["Spike Up", "Spike Down"]

if not events.empty:
    pivot = events.pivot_table(
        index="vix_base_bucket",
        columns="move_type",
        values=f"spx_fwd{fwd_days}_ret",
        aggfunc=lambda s: winrate(pd.Series(s))
    ).reindex(index=order_base, columns=order_direction)
else:
    pivot = pd.DataFrame(np.nan, index=order_base, columns=order_direction)

cmap = LinearSegmentedColormap.from_list("pastelWR", ["#EAF7FB", "#A8DADC", "#4DB6AC"])
ax2.imshow(pivot.values.astype(float), aspect="auto", cmap=cmap, vmin=0, vmax=100)
ax2.set_xticks(range(len(order_direction)))
ax2.set_xticklabels(order_direction, fontsize=10)
ax2.set_yticks(range(len(order_base)))
ax2.set_yticklabels(order_base, fontsize=10)
ax2.set_title("Win Rate by Base × Direction", color=TEXT_COLOR, fontsize=12, pad=8)

for i in range(len(order_base)):
    for j in range(len(order_direction)):
        val = pivot.values[i, j]
        txt = "" if np.isnan(val) else f"{val:.0f}%"
        ax2.text(j, i, txt, ha="center", va="center", color="#1f1f1f", fontsize=9)

# 3) ECDF by regime
ax3 = axes[0, 2]
if not events.empty:
    for name, color in [("Bull", PASTELS[0]), ("Bear", PASTELS[2])]:
        vals = pd.to_numeric(
            events.loc[events["regime"] == name, f"spx_fwd{fwd_days}_ret"],
            errors="coerce"
        ).dropna().values
        if vals.size > 0:
            x = np.sort(vals)
            y = np.arange(1, x.size + 1) / x.size
            ax3.plot(x, y, label=f"{name} (N={x.size})", linewidth=2, color=color)
ax3.axvline(0, color="#888888", linewidth=1)
ax3.set_title(f"ECDF of {fwd_days}-Day Returns by Regime", color=TEXT_COLOR, fontsize=12, pad=8)
ax3.set_xlabel("SPX Forward Return (%)", color=TEXT_COLOR)
ax3.set_ylabel("Cumulative Probability", color=TEXT_COLOR)
ax3.grid(color=GRID_COLOR, linewidth=0.6)
ax3.legend(frameon=False, fontsize=9)

# 4) Scatter
ax4 = axes[1, 0]
if not events.empty:
    vals = events.copy()
    vals["MovePct"] = vals["vix_pctchg"] * 100.0
    colors = np.where(vals["move_type"] == "Spike Up", PASTELS[1], PASTELS[3])
    sizes = 10 + np.clip(vals["MovePct"].abs(), 0, 100)
    ax4.scatter(
        vals["vix_base"],
        vals[f"spx_fwd{fwd_days}_ret"],
        s=sizes,
        c=colors,
        alpha=0.85,
        edgecolors=BAR_EDGE,
        linewidths=0.3
    )
    good = vals[["vix_base", f"spx_fwd{fwd_days}_ret"]].dropna()
    if len(good) > 5:
        m, b = np.polyfit(good["vix_base"], good[f"spx_fwd{fwd_days}_ret"], 1)
        xs = np.linspace(good["vix_base"].min(), good["vix_base"].max(), 100)
        ax4.plot(xs, m * xs + b, linewidth=1.5, color="#444444")
ax4.axhline(0, color="#888888", linewidth=1)
ax4.set_title("VIX Base vs Forward Return, size=|Move| color=Direction", color=TEXT_COLOR, fontsize=12, pad=8)
ax4.set_xlabel("VIX Level Before Move", color=TEXT_COLOR)
ax4.set_ylabel(f"SPX {fwd_days}-Day Return (%)", color=TEXT_COLOR)
ax4.grid(color=GRID_COLOR, linewidth=0.6)

# 5) Event study
ax5 = axes[1, 1]
horizons = [1, 2, 3, 5, 10, 20]
df = ensure_fwd_cols(df, horizons)

if not events.empty:
    idx = events.index
    means, p10s, p90s = [], [], []
    for h in horizons:
        x = df.loc[idx, f"spx_fwd{h}_ret"].astype(float)
        means.append(np.nanmean(x))
        p10s.append(np.nanpercentile(x.dropna(), 10) if x.notna().sum() else np.nan)
        p90s.append(np.nanpercentile(x.dropna(), 90) if x.notna().sum() else np.nan)
    ax5.plot(horizons, means, linewidth=2, marker="o")
    ax5.fill_between(horizons, p10s, p90s, alpha=0.25)

ax5.axhline(0, color="#888888", linewidth=1)
ax5.set_title("Event Study, Mean Path after Qualifying Move", color=TEXT_COLOR, fontsize=12, pad=8)
ax5.set_xlabel("Horizon (trading days)", color=TEXT_COLOR)
ax5.set_ylabel("Forward Return (%)", color=TEXT_COLOR)
ax5.grid(color=GRID_COLOR, linewidth=0.6)

# 6) Yearly count
ax6 = axes[1, 2]
if not events.empty:
    yr_counts = events.groupby(events.index.year).size()
    years = yr_counts.index.astype(int).tolist()
    counts = yr_counts.values.tolist()
    ax6.bar(years, counts, edgecolor=BAR_EDGE, color=PASTELS[9])
    ax6.set_xlim(min(years) - 0.5, max(years) + 0.5)
ax6.set_title("Yearly Count of Qualifying VIX Moves", color=TEXT_COLOR, fontsize=12, pad=8)
ax6.set_xlabel("Year", color=TEXT_COLOR)
ax6.set_ylabel("Count", color=TEXT_COLOR)
ax6.grid(axis="y", color=GRID_COLOR, linewidth=0.6)

st.pyplot(fig, clear_figure=True)

# ------------------------------- Dynamic commentary ------------------------
st.subheader("Dynamic Commentary")

def latest_context_box():
    if latest is None:
        card_box("No qualifying VIX move events in the filtered sample.")
        return

    dt = latest.name.strftime("%Y-%m-%d")
    vix_now = latest["vix_close"]
    vix_base = latest["vix_base"]
    vix_move = latest["vix_pctchg"] * 100.0
    spx_now = latest["spx_close"]
    rsi_now = latest["rsi14"]
    regime_now_local = latest["regime"]
    bucket = latest["vix_base_bucket"]
    move_type = latest["move_type"]
    magcat = latest["move_mag"]
    over = latest["oversold"]

    wr = winrate(comps[f"spx_fwd{fwd_days}_ret"]) if not comps.empty else np.nan
    avg = comps[f"spx_fwd{fwd_days}_ret"].mean() if not comps.empty else np.nan
    med = comps[f"spx_fwd{fwd_days}_ret"].median() if not comps.empty else np.nan
    mad = mean_abs_deviation(comps[f"spx_fwd{fwd_days}_ret"]) if not comps.empty else np.nan
    p10_c, p50_c, p90_c = pct_bands(comps[f"spx_fwd{fwd_days}_ret"]) if not comps.empty else (np.nan, np.nan, np.nan)
    n = len(comps) if not comps.empty else 0

    text = f"""
    <b>Latest event</b> {dt}. VIX {vix_now:.2f} from base {vix_base:.2f}, move {vix_move:.1f}% ({move_type}).
    SPX {spx_now:.2f}, RSI14 {rsi_now:.1f} ({over}), regime {regime_now_local}.
    Setup, base {bucket}, magnitude {magcat}.
    <br><br>
    <b>Setup stats</b> same bucket, direction, magnitude, regime. WR {fmt_pct(wr, 1)}, Avg {fmt_pct(avg, 1)}, Med {fmt_pct(med, 1)}, Disp {fmt_pct(mad, 1)}, N {n}.
    Bands p10 {fmt_pct(p10_c)}, p50 {fmt_pct(p50_c)}, p90 {fmt_pct(p90_c)}.
    """.strip()
    card_box(text)

    if n > 0:
        show_cols = [
            "vix_close", "vix_base", "vix_pctchg", "spx_close", f"spx_fwd{fwd_days}_ret", "rsi14", "move_type"
        ]
        analogs = (
            comps[show_cols]
            .rename(columns={
                "vix_close": "VIX",
                "vix_base": "VIX_Base",
                "vix_pctchg": "VIX_Move_Frac",
                "spx_close": "SPX",
                f"spx_fwd{fwd_days}_ret": f"SPX_Fwd{fwd_days}_Ret_%",
                "rsi14": "RSI14",
                "move_type": "Move_Type"
            })
            .copy()
        )
        analogs["VIX_Move_%"] = (analogs["VIX_Move_Frac"] * 100.0).round(2)
        analogs.drop(columns=["VIX_Move_Frac"], inplace=True)
        analogs = analogs.round(2).sort_index(ascending=False).head(10)
        st.markdown("**Nearest analogs, same setup, last 10 occurrences**")
        st.dataframe(analogs, use_container_width=True)

latest_context_box()

# ------------------------------- Events table ------------------------------
with st.expander("Show events table"):
    if not events.empty:
        show_cols = [
            "vix_close", "vix_base", "vix_pctchg", "vix_abs_pctchg", "spx_close", f"spx_fwd{fwd_days}_ret",
            "vix_base_bucket", "move_type", "move_mag", "regime", "rsi14", "oversold", "decade"
        ]
        tbl = events[show_cols].copy()
        tbl.rename(columns={
            "vix_close": "VIX",
            "vix_base": "VIX_Base",
            "vix_pctchg": "VIX_Move_Fr",
            "vix_abs_pctchg": "VIX_Abs_Move_Fr",
            "spx_close": "SPX",
            f"spx_fwd{fwd_days}_ret": f"SPX_Fwd{fwd_days}_Ret_%",
            "rsi14": "RSI14"
        }, inplace=True)
        tbl["VIX_Move_%"] = (tbl["VIX_Move_Fr"] * 100.0).round(2)
        tbl["VIX_Abs_Move_%"] = (tbl["VIX_Abs_Move_Fr"] * 100.0).round(2)
        tbl.drop(columns=["VIX_Move_Fr", "VIX_Abs_Move_Fr"], inplace=True)
        st.dataframe(tbl.round(2), use_container_width=True)
    else:
        st.write("No events under current filters.")

# ------------------------------- Footer ------------------------------------
st.caption("ADFM Analytics Platform | VIX Spike Deep Dive | Data source: Yahoo Finance")
st.caption("© 2026 AD Fund Management LP")
