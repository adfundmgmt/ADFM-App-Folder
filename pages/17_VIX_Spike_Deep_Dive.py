# vix_spike_deep_dive.py
# ADFM Analytics Platform — VIX Spike Deep Dive
# Light theme, pastel palette, matplotlib only. Decision Box unchanged. Six revised panels.

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
    # pct is a fraction
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

# ------------------------------- Sidebar -----------------------------------
st.title("VIX Spike Deep Dive")

with st.sidebar:
    st.header("Controls")
    start_date = st.date_input("History start", value=datetime(1990, 1, 1))
    fwd_days = st.number_input("Forward horizon (days)", min_value=1, max_value=30, value=2, step=1)
    spike_threshold = st.slider("VIX spike threshold (%)", 20, 100, 20, step=5)
    rsi_thresh = st.slider("Oversold RSI threshold", 10, 40, 30, step=1)
    ma_window = st.selectbox("Regime MA window", [100, 150, 200], index=2)
    show_legend = st.checkbox("Show legend", value=False)

# ------------------------------- Data --------------------------------------
df = load_history(start=str(start_date))

df[f"spx_next{fwd_days}"] = df["spx_close"].shift(-fwd_days)
df[f"spx_fwd{fwd_days}_ret"] = (df[f"spx_next{fwd_days}"] / df["spx_close"] - 1.0) * 100.0

df[f"spx_{ma_window}dma"] = df["spx_close"].rolling(ma_window).mean()
df["regime"] = np.where(df["spx_close"] >= df[f"spx_{ma_window}dma"], "Bull", "Bear")

events = df.copy()
events["vix_base"] = events["vix_close"].shift(1)
events["vix_spike"] = events["vix_pctchg"] >= (spike_threshold / 100.0)
events = events[events["vix_spike"]].dropna(subset=["vix_base"]).copy()

if not events.empty:
    events["vix_base_bucket"] = events["vix_base"].apply(bucket_vix_base)
    events["spike_mag"] = events["vix_pctchg"].apply(bucket_spike_mag)
    events["oversold"] = np.where(events["rsi14"] <= rsi_thresh, "Oversold", "Not Oversold")
    events["decade"] = [decade_label(d) for d in events.index]
else:
    events = pd.DataFrame(columns=["vix_base_bucket","spike_mag","oversold","decade","regime",f"spx_fwd{fwd_days}_ret"])

all_decades = sorted(events["decade"].unique().tolist()) if not events.empty else []
decade_filter = st.sidebar.multiselect("Decades to include", options=all_decades, default=all_decades)
if decade_filter:
    events = events[events["decade"].isin(decade_filter)]

latest = events.iloc[-1] if not events.empty else None
if latest is not None:
    bucket_now = latest["vix_base_bucket"]
    mag_now = latest["spike_mag"]
    regime_now = latest["regime"]
    rsi_state = latest["oversold"]
    comps_mask = (
        (events["vix_base_bucket"] == bucket_now) &
        (events["spike_mag"] == mag_now) &
        (events["regime"] == regime_now)
    )
    comps = events[comps_mask].copy()
else:
    bucket_now = mag_now = regime_now = rsi_state = None
    comps = pd.DataFrame(columns=events.columns)

# ------------------------------- Stats for commentary ----------------------
overall = events[f"spx_fwd{fwd_days}_ret"] if not events.empty else pd.Series(dtype=float)
bucket_wr = events.groupby("vix_base_bucket")[f"spx_fwd{fwd_days}_ret"].apply(winrate) if not events.empty else pd.Series(dtype=float)
mag_wr = events.groupby("spike_mag")[f"spx_fwd{fwd_days}_ret"].apply(winrate) if not events.empty else pd.Series(dtype=float)
reg_wr = events.groupby("regime")[f"spx_fwd{fwd_days}_ret"].apply(winrate) if not events.empty else pd.Series(dtype=float)
ov_wr = events.groupby("oversold")[f"spx_fwd{fwd_days}_ret"].apply(winrate) if not events.empty else pd.Series(dtype=float)
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
    bull_wr, bear_wr = ctx["bull_wr"], ctx["bear_wr"]
    ov_yes, ov_no = ctx["ov_yes"], ctx["ov_no"]
    threshold = ctx["spike_threshold"]
    fwd_days = ctx["fwd_days"]
    bucket_now = ctx["bucket_now"]
    mag_now = ctx["mag_now"]
    regime_now = ctx["regime_now"]
    rsi_state = ctx["rsi_state"]
    ma_window = ctx["ma_window"]
    show_legend = ctx["show_legend"]

    bucket_order = ["0-12","12-16","16-20","20-24","24-30","30+"]
    mag_order = ["Moderate (20-30%)","Large (30-50%)","Extreme (50%+)"]
    bucket_idx = bucket_order.index(str(bucket_now)) if bucket_now in bucket_order else 0
    mag_idx = mag_order.index(str(mag_now)) if mag_now in mag_order else 0
    regime_idx = 0 if regime_now == "Bull" else 1
    rsi_idx = 0 if rsi_state == "Oversold" else 1

    headline_bank_bull = [
        "Vol shock fades, drift favors buyers.",
        "Panic cooled, bias tilts up in trend.",
        "Spike absorbed, path of least resistance is higher.",
        "Stress unwinds, reflex bid shows up.",
        "Trend steady, bounce odds improve.",
        "Compression after shock supports a tactical long.",
    ]
    headline_bank_bear = [
        "Relief pops exist, trend resists.",
        "Shock inside a weak tape, edge thinner.",
        "Pops fade faster in this regime.",
        "Down-trend blunts the reflex bid.",
        "Counter-trend carry is hostile.",
        "Stress persists, bounce quality is lower.",
    ]
    headline = (headline_bank_bull if regime_idx == 0 else headline_bank_bear)[(bucket_idx + mag_idx) % 6]

    wim_bank = [
        f"Whether to press for a {fwd_days}-day relief move after a ≥{threshold}% VIX jump.",
        f"Signal quality for a short tactical hold into mean reversion.",
        f"Gauge if panic creates a near-term buyable skew.",
        f"Decide if a brief probe long is justified after stress.",
        f"Size and timing for a quick bounce attempt.",
        f"Should you lean into a fast mean-revert move.",
    ]
    why = wim_bank[(regime_idx + rsi_idx + mag_idx) % len(wim_bank)]

    bucket_desc = [
        "low base VIX leaves less fuel",
        "mid base VIX offers some spring",
        "upper-mid base VIX carries energy",
        "elevated base VIX adds thrust",
        "high base VIX loads the coil",
        "very high base VIX, tails widen",
    ][bucket_idx]

    mag_desc = [
        "a manageable shock",
        "a heavy jolt",
        "a sharp dislocation",
        "a disorderly jump",
        "a capitulation-style surge",
        "a face-ripper spike",
    ][(mag_idx + bucket_idx) % 6]

    if regime_idx == 0:
        regime_line = f"Above the {ma_window}-DMA, hit rate improves to {fmt_pct(bull_wr,1)}."
    else:
        regime_line = f"Below the {ma_window}-DMA, hit rate slips to {fmt_pct(bear_wr,1)}."

    if pd.notna(wr_all) and pd.notna(med):
        if wr_all >= 53 and med >= 0:
            conclusion_tail = "Setup supports a tactical long."
        elif 47 <= wr_all < 53:
            conclusion_tail = "Edge is mixed, execution matters."
        else:
            conclusion_tail = "Edge is weak, caution first."
    else:
        conclusion_tail = "Insufficient sample."

    drivers = []
    drivers.append(f"Base case, WR {fmt_pct(wr_all,1)}, median {fmt_pct(med,1)}, average {fmt_pct(avg,1)}, N {n}.")
    if pd.notna(best_bucket_wr) and best_bucket_label is not None:
        drivers.append(f"Base VIX {best_bucket_label} works best, WR {fmt_pct(best_bucket_wr,1)} ({bucket_desc}).")
    if pd.notna(best_mag_wr) and best_mag_label is not None:
        drivers.append(f"Spike magnitude {best_mag_label}, WR {fmt_pct(best_mag_wr,1)} ({mag_desc}).")
    drivers.append(regime_line)
    if pd.notna(ov_yes) and pd.notna(ov_no):
        rsi_line = "Oversold helps" if rsi_idx == 0 else "No stretch tempers the pop"
        drivers.append(f"RSI filter, Oversold {fmt_pct(ov_yes,1)} vs Not {fmt_pct(ov_no,1)} ({rsi_line}).")
    drivers.append(f"Expected range for {fwd_days} days, p10 {fmt_pct(p10)}, p50 {fmt_pct(p50)}, p90 {fmt_pct(p90)}.")

    body = (
        f'<div style="font-weight:700; margin-bottom:6px;">Conclusion</div>'
        f'<div>{headline} {conclusion_tail}</div>'
        f'<div style="font-weight:700; margin:10px 0 6px;">Why it matters</div>'
        f'<div>{why}</div>'
        f'<div style="font-weight:700; margin:10px 0 6px;">Key drivers</div>'
        f'<ul style="margin-top:4px; margin-bottom:4px;">'
        + ''.join(f'<li>{d}</li>' for d in drivers) +
        '</ul>'
    )

    if show_legend:
        body += (
            '<hr style="border-top:1px solid #eee; margin:10px 0;">'
            '<div style="font-size:13px;">'
            '<b>Legend</b> WR win rate, Avg and Med forward SPX returns, N sample size. '
            'Base bucket is VIX level before the spike. Magnitude is spike size. '
            'Regime uses your DMA. RSI uses your slider. '
            'p10, p50, p90 are percentile bands of forward returns for the chosen horizon.'
            '</div>'
        )

    lights = 0
    if bucket_now in ["24-30","30+"]:
        lights += 1
    if mag_now in ["Large (30-50%)","Extreme (50%+)"]:
        lights += 1
    if regime_now == "Bull":
        lights += 1
    if rsi_state == "Oversold" and pd.notna(ov_yes) and pd.notna(ov_no) and ov_yes >= ov_no:
        lights += 1

    return body, lights

# Build context and commentary
ctx = {
    "wr_all": wr_all, "med_all": med_all, "avg_all": avg_all, "n_all": n_all,
    "p10": p10, "p50": p50, "p90": p90,
    "best_bucket_label": best_bucket_label, "best_bucket_wr": best_bucket_wr,
    "best_mag_label": best_mag_label, "best_mag_wr": best_mag_wr,
    "bull_wr": bull_wr, "bear_wr": bear_wr,
    "ov_yes": ov_yes, "ov_no": ov_no,
    "spike_threshold": spike_threshold, "fwd_days": fwd_days,
    "bucket_now": bucket_now, "mag_now": mag_now, "regime_now": regime_now, "rsi_state": rsi_state,
    "ma_window": ma_window, "show_legend": show_legend,
}
summary_html, lights = generate_commentary(ctx)

# ------------------------------- Top row: Decision Box + Filters -----------
col1, col2 = st.columns([1.8, 1])

with col1:
    st.subheader("Decision Box")
    card_box(summary_html)

with col2:
    st.subheader("Filters")
    card_box(
        f"""
        <b>Spike threshold</b>: {spike_threshold}%<br>
        <b>Forward horizon</b>: {fwd_days} trading days<br>
        <b>RSI oversold</b>: ≤ {rsi_thresh}<br>
        <b>Regime MA</b>: {ma_window}-day<br>
        <b>Decades</b>: {', '.join(decade_filter) if decade_filter else 'None'}<br>
        <b>Four lights</b>: {lights} / 4
        """.strip()
    )

# ------------------------------- Panels v2 ---------------------------------
# Six revised charts. Decision Box above remains unchanged.

def safe_group_lists(g, col, order):
    data = []
    labels = []
    for k in order:
        arr = pd.to_numeric(g.get_group(k)[col], errors="coerce").dropna().values if (not events.empty and k in g.groups) else np.array([])
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

# 1) Boxplot: forward returns by VIX base bucket
ax1 = axes[0, 0]
order_base = ["0-12","12-16","16-20","20-24","24-30","30+"]
if not events.empty:
    g = events.groupby("vix_base_bucket")
    data, labels = safe_group_lists(g, f"spx_fwd{fwd_days}_ret", order_base)
else:
    data, labels = [], []
bp = ax1.boxplot(
    data if data else [np.array([np.nan])],
    labels=labels if labels else ["no data"],
    patch_artist=True, notch=True, showfliers=False
)
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(PASTELS[i % len(PASTELS)])
    patch.set_edgecolor(BAR_EDGE)
for med in bp["medians"]:
    med.set_color("#444444")
ax1.axhline(0, color="#888888", linewidth=1)
ax1.set_title(f"Distribution by VIX Base, {fwd_days}-Day %", color=TEXT_COLOR, fontsize=12, pad=8)
ax1.grid(axis="y", color=GRID_COLOR, linewidth=0.6)

# 2) Heatmap: win rate by Base × Magnitude
ax2 = axes[0, 1]
order_mag = ["Moderate (20-30%)","Large (30-50%)","Extreme (50%+)"]
if not events.empty:
    pivot = events.pivot_table(
        index="vix_base_bucket", columns="spike_mag",
        values=f"spx_fwd{fwd_days}_ret",
        aggfunc=lambda s: winrate(pd.Series(s))
    ).reindex(index=order_base, columns=order_mag)
else:
    pivot = pd.DataFrame(np.nan, index=order_base, columns=order_mag)
cmap = LinearSegmentedColormap.from_list("pastelWR", ["#EAF7FB", "#A8DADC", "#4DB6AC"])
im = ax2.imshow(pivot.values.astype(float), aspect="auto", cmap=cmap, vmin=0, vmax=100)
ax2.set_xticks(range(len(order_mag)))
ax2.set_xticklabels(["20–30%", "30–50%", "50%+"], fontsize=10)
ax2.set_yticks(range(len(order_base)))
ax2.set_yticklabels(order_base, fontsize=10)
ax2.set_title("Win Rate by Base × Magnitude", color=TEXT_COLOR, fontsize=12, pad=8)
for i in range(len(order_base)):
    for j in range(len(order_mag)):
        val = pivot.values[i, j]
        txt = "" if np.isnan(val) else f"{val:.0f}%"
        ax2.text(j, i, txt, ha="center", va="center", color="#1f1f1f", fontsize=9)

# 3) ECDF: Bull vs Bear
ax3 = axes[0, 2]
if not events.empty:
    for name, color in [("Bull", PASTELS[0]), ("Bear", PASTELS[2])]:
        vals = pd.to_numeric(events.loc[events["regime"] == name, f"spx_fwd{fwd_days}_ret"], errors="coerce").dropna().values
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

# 4) Scatter: VIX base vs forward return, size=Spike%, color=RSI Oversold
ax4 = axes[1, 0]
if not events.empty:
    vals = events.copy()
    vals["SpikePct"] = vals["vix_pctchg"] * 100.0
    colors = np.where(vals["oversold"] == "Oversold", PASTELS[3], PASTELS[1])
    sizes = 10 + np.clip(vals["SpikePct"].abs(), 0, 100)
    ax4.scatter(vals["vix_base"], vals[f"spx_fwd{fwd_days}_ret"], s=sizes, c=colors,
                alpha=0.85, edgecolors=BAR_EDGE, linewidths=0.3)
    good = vals[["vix_base", f"spx_fwd{fwd_days}_ret"]].dropna()
    if len(good) > 5:
        m, b = np.polyfit(good["vix_base"], good[f"spx_fwd{fwd_days}_ret"], 1)
        xs = np.linspace(good["vix_base"].min(), good["vix_base"].max(), 100)
        ax4.plot(xs, m*xs + b, linewidth=1.5, color="#444444")
ax4.axhline(0, color="#888888", linewidth=1)
ax4.set_title("VIX Base vs Forward Return, size=Spike% color=RSI", color=TEXT_COLOR, fontsize=12, pad=8)
ax4.set_xlabel("VIX Level Before Spike", color=TEXT_COLOR)
ax4.set_ylabel(f"SPX {fwd_days}-Day Return (%)", color=TEXT_COLOR)
ax4.grid(color=GRID_COLOR, linewidth=0.6)

# 5) Event study: mean path and 10/90 bands over horizons
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
ax5.set_title("Event Study, Mean Path after Spike", color=TEXT_COLOR, fontsize=12, pad=8)
ax5.set_xlabel("Horizon (trading days)", color=TEXT_COLOR)
ax5.set_ylabel("Forward Return (%)", color=TEXT_COLOR)
ax5.grid(color=GRID_COLOR, linewidth=0.6)

# 6) Yearly count of qualifying spikes
ax6 = axes[1, 2]
if not events.empty:
    yr_counts = events.groupby(events.index.year).size()
    years = yr_counts.index.astype(int).tolist()
    counts = yr_counts.values.tolist()
    ax6.bar(years, counts, edgecolor=BAR_EDGE, color=PASTELS[9])
    ax6.set_xlim(min(years) - 0.5, max(years) + 0.5)
ax6.set_title("Yearly Count of ≥ Threshold Spikes", color=TEXT_COLOR, fontsize=12, pad=8)
ax6.set_xlabel("Year", color=TEXT_COLOR)
ax6.set_ylabel("Count", color=TEXT_COLOR)
ax6.grid(axis="y", color=GRID_COLOR, linewidth=0.6)

st.pyplot(fig, clear_figure=True)
# ----------------------------- End Panels v2 -------------------------------

# ------------------------------- Dynamic commentary for latest event -------
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
    p10_c, p50_c, p90_c = pct_bands(comps[f"spx_fwd{fwd_days}_ret"]) if not comps.empty else (np.nan, np.nan, np.nan)
    n = len(comps) if not comps.empty else 0

    text = f"""
    <b>Latest event</b> {dt}. VIX {vix_now:.2f} from base {vix_base:.2f}, spike {vix_spike:.1f}%.
    SPX {spx_now:.2f}, RSI14 {rsi_now:.1f} ({over}), regime {regime_now}.
    Setup, base {bucket}, magnitude {magcat}.
    <br><br>
    <b>Setup stats</b> same bucket, magnitude, regime. WR {fmt_pct(wr,1)}, Avg {fmt_pct(avg,1)}, Med {fmt_pct(med,1)}, Disp {fmt_pct(mad,1)}, N {n}.
    Bands p10 {fmt_pct(p10_c)}, p50 {fmt_pct(p50_c)}, p90 {fmt_pct(p90_c)}.
    """.strip()
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
    if not events.empty:
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
    else:
        st.write("No events under current filters.")

# ------------------------------- Footer ------------------------------------
st.caption("ADFM Analytics Platform · VIX Spike Deep Dive · Data source: Yahoo Finance")
