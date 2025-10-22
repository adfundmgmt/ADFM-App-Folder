# vix_spike_deep_dive.py
# ADFM Analytics Platform, VIX 20%+ Spike Deep Dive
# Light theme, pastel palette, matplotlib only, dynamic regime-aware commentary.

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
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

    # RSI(14)
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

def fmt_pct(x, digits=2, sign=False):
    if pd.isna(x):
        return "NA"
    s = f"{x:.{digits}f}%"
    if sign and x > 0:
        s = "+" + s
    return s

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

# ------------------------------- Sidebar -----------------------------------
st.title("VIX Spike Deep Dive")

with st.sidebar:
    st.header("Controls")
    start_date = st.date_input("History start", value=datetime(1990, 1, 1))
    fwd_days = st.number_input("Forward horizon (days)", min_value=1, max_value=10, value=2, step=1)
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
events = events[events["vix_spike"]].dropna(subset=["vix_base"])

events["vix_base_bucket"] = events["vix_base"].apply(bucket_vix_base)
events["spike_mag"] = events["vix_pctchg"].apply(bucket_spike_mag)
events["oversold"] = np.where(events["rsi14"] <= rsi_thresh, "Oversold", "Not Oversold")
events["decade"] = [decade_label(d) for d in events.index]

all_decades = sorted(events["decade"].unique().tolist())
decade_filter = st.sidebar.multiselect("Decades to include", options=all_decades, default=all_decades)
events = events[events["decade"].isin(decade_filter)]

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

# ------------------------------- Stats for commentary ----------------------
overall = events[f"spx_fwd{fwd_days}_ret"]
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
    """
    Build a regime-aware, data-driven note from phrase banks.
    No randomness, selection keyed to regime, VIX bucket, spike mag, RSI, and stats.
    """
    # Context unpack
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
    mad_all = ctx["mad_all"]

    # Category indices for deterministic variety
    bucket_idx = ["0-12","12-16","16-20","20-24","24-30","30+"].index(str(bucket_now)) if bucket_now is not None else 0
    mag_idx = ["Moderate (20-30%)","Large (30-50%)","Extreme (50%+)"].index(str(mag_now)) if mag_now is not None else 0
    regime_idx = 0 if regime_now == "Bull" else 1
    rsi_idx = 0 if rsi_state == "Oversold" else 1
    edge_up = 1 if (wr_all is not None and pd.notna(wr_all) and wr_all >= 50 and pd.notna(med) and med >= 0) else 0

    # Headline banks (12 variants)
    headline_bank_bull = [
        "Short window leans green after a spike",
        "Tape supports a quick bounce after stress",
        "Panic cooled, bias tilts up in trend",
        "Vol shock fades, drift favors buyers",
        "Bid returns with trend at the back",
        "Spike absorbed, path of least resistance is higher",
    ]
    headline_bank_bear = [
        "Relief bounces exist, but trend resists",
        "Vol shock in a down tape, edge is thinner",
        "Panic meets gravity, bounces are shorter",
        "Stress signal inside a weak regime",
        "Counter-trend pops possible, carry is hostile",
        "Snapbacks fade faster in this regime",
    ]
    headline = (headline_bank_bull if regime_idx == 0 else headline_bank_bear)[(bucket_idx + mag_idx) % 6]

    # Why it matters (6 variants)
    wim_bank = [
        f"Quick read on a {fwd_days}-day hold after a ≥{threshold}% VIX jump.",
        f"Whether to press for a {fwd_days}-day relief move after a hard vol shock.",
        f"Signal quality for a short tactical hold into mean reversion.",
        f"Gauge if panic creates a near-term buyable skew.",
        f"Should you lean into a fast bounce or stand down.",
        f"Context to size a brief risk-on probe after stress.",
    ]
    why = wim_bank[(regime_idx + rsi_idx + mag_idx) % len(wim_bank)]

    # Bucket phrasing (6 variants)
    bucket_bank = [
        "low base VIX leaves less fuel",
        "mid base VIX offers some spring",
        "upper-mid base VIX carries energy",
        "elevated base VIX adds thrust",
        "high base VIX loads the coil",
        "very high base VIX, tails get wider",
    ]
    bucket_phrase = bucket_bank[bucket_idx]

    # Magnitude phrasing (6 variants)
    mag_bank = [
        "a manageable shock",
        "a heavy jolt",
        "a disorderly jump",
        "a sharp dislocation",
        "a capitulation-style surge",
        "a face-ripper spike",
    ]
    mag_phrase = mag_bank[(mag_idx + bucket_idx) % len(mag_bank)]

    # RSI phrasing (4 variants)
    rsi_bank_oversold = [
        "oversold helps the snapback",
        "stretch adds mean-revert energy",
        "pressure built for a reflex bounce",
        "decompression flows favor upside",
    ]
    rsi_bank_not = [
        "no stretch lowers the spring",
        "lack of stretch tempers the pop",
        "neutral RSI, less pent-up energy",
        "without stretch, moves are cleaner but smaller",
    ]
    rsi_phrase = (rsi_bank_oversold if rsi_idx == 0 else rsi_bank_not)[(bucket_idx + mag_idx) % 4]

    # Regime split phrasing (8 variants)
    if regime_idx == 0:
        regime_phrase = [
            f"Trend filter shows Bull {fmt_pct(bull_wr,1)} vs Bear {fmt_pct(bear_wr,1)}, trend helps.",
            f"Above the {ma_window}-DMA, hit rate improves to {fmt_pct(bull_wr,1)}.",
            f"Bull regime carries a better base rate at {fmt_pct(bull_wr,1)}.",
            f"With trend up, win rate outpaces bear at {fmt_pct(bull_wr,1)}.",
        ][(mag_idx + rsi_idx) % 4]
    else:
        regime_phrase = [
            f"Below the {ma_window}-DMA, hit rate slips to {fmt_pct(bear_wr,1)}.",
            f"Bear regime drags base rate at {fmt_pct(bear_wr,1)}.",
            f"Counter-trend carry cuts the edge to {fmt_pct(bear_wr,1)}.",
            f"Trend is down, base rate sits near {fmt_pct(bear_wr,1)}.",
        ][(mag_idx + rsi_idx) % 4]

    # Edge classification from stats
    edge_level = "positive" if (pd.notna(wr_all) and wr_all >= 53 and pd.notna(med) and med >= 0) else \
                 "balanced" if (pd.notna(wr_all) and 47 <= wr_all < 53) else "negative"

    # Conclusion variants (12 total)
    if edge_level == "positive":
        concl_bank = [
            "Short window favors a bounce.",
            "Expect a modest reflex higher.",
            "Bias is up, not huge, but real.",
            "Mean-reversion tilt is on.",
            "Setup supports a tactical long.",
            "Relief pop is the base case.",
        ]
    elif edge_level == "balanced":
        concl_bank = [
            "Edge is mixed, discipline over conviction.",
            "Distribution is balanced, sizing matters.",
            "No strong tilt, pick your spots.",
            "Signal is flat, respect levels.",
            "Even split, execution is the edge.",
            "Neutral skew, wait for better alignment.",
        ]
    else:
        concl_bank = [
            "Edge is weak, caution first.",
            "Pops are for sale unless trend turns.",
            "Downside risk dominates the short window.",
            "Snapbacks are fragile in this tape.",
            "Avoid forcing the bounce.",
            "Defense until the filter improves.",
        ]
    conclusion = concl_bank[(bucket_idx + regime_idx + rsi_idx) % 6]

    # Drivers assembled from actual numbers
    drivers = []
    drivers.append(f"Base case, WR {fmt_pct(wr_all,1)}, median {fmt_pct(med)}, average {fmt_pct(avg)}, N {n}.")
    if pd.notna(best_bucket_wr) and best_bucket_label is not None:
        drivers.append(f"Starting VIX level, best bucket {best_bucket_label}, WR {fmt_pct(best_bucket_wr,1)} ({bucket_phrase}).")
    if pd.notna(best_mag_wr) and best_mag_label is not None:
        drivers.append(f"Spike size, favored magnitude {best_mag_label}, WR {fmt_pct(best_mag_wr,1)} ({mag_phrase}).")
    drivers.append(regime_phrase)
    if pd.notna(ov_yes) and pd.notna(ov_no):
        if rsi_idx == 0:
            drivers.append(f"RSI filter, Oversold {fmt_pct(ov_yes,1)} vs Not {fmt_pct(ov_no,1)} ({rsi_phrase}).")
        else:
            drivers.append(f"RSI filter, Oversold {fmt_pct(ov_yes,1)} vs Not {fmt_pct(ov_no,1)} ({rsi_phrase}).")
    drivers.append(f"Expected range for {fwd_days} days, p10 {fmt_pct(p10)}, p50 {fmt_pct(p50)}, p90 {fmt_pct(p90)}.")

    # Risks and reversals banks (10 variants chosen deterministically)
    risk_bank = [
        "Low base VIX buckets reduce bounce quality.",
        "Bear regime increases failure rate and drawdown risk.",
        "Extreme spikes widen both tails, entries matter.",
        "Tight liquidity windows can invert the edge.",
        "Weak medians erase thin WR advantages.",
        "Event clustering changes forward drift.",
        "Overnight gaps can dominate short horizons.",
        "Macro catalysts can swamp historical patterns.",
        "Slippage and carry can turn small edges negative.",
        "Sample changes as you tighten filters.",
    ]
    risks = [
        risk_bank[(bucket_idx) % 10],
        risk_bank[(bucket_idx + mag_idx + 2) % 10],
        risk_bank[(regime_idx + rsi_idx + 4) % 10],
    ]

    # Next steps banks (12 variants)
    if edge_level == "positive":
        next_bank = [
            f"Prefer entries when base VIX is {bucket_now} and spike is {mag_now}.",
            f"Size modestly, lean long into the {fwd_days}-day window.",
            f"Respect p10 near {fmt_pct(p10)} for risk control.",
            "Tight stop or hedge, keep it tactical.",
            "If two filters flip against you, pass.",
            f"Keep regime filter on {ma_window}-DMA to avoid whips.",
        ]
    elif edge_level == "balanced":
        next_bank = [
            "Trade small or wait for alignment of filters.",
            f"Let price confirm, above yesterday’s high or {ma_window}-DMA.",
            f"Guardrail with p10 near {fmt_pct(p10)}.",
            "Favor options structures with defined risk.",
            "Stand down if liquidity is thin around data.",
            "Recheck after the next session’s close.",
        ]
    else:
        next_bank = [
            "Skip the bounce setup, wait for regime to improve.",
            "If participating, use tiny size and tight timing.",
            f"Use spreads or collars, respect p10 near {fmt_pct(p10)}.",
            "Only reconsider if RSI flips to oversold and base VIX lifts.",
            "Let vol decay first before probing.",
            f"Require reclaim of the {ma_window}-DMA before attempts.",
        ]
    next_steps = [
        next_bank[(bucket_idx + mag_idx) % 6],
        next_bank[(regime_idx + rsi_idx + 2) % 6],
        next_bank[(bucket_idx + regime_idx + 3) % 6],
    ]

    # Assemble HTML
    body = f"""
    <div style="font-size:14px;">
      <div style="font-weight:700; margin-bottom:6px;">Conclusion</div>
      <div>{headline}. {conclusion}</div>
      <div style="font-weight:700; margin:10px 0 6px;">Why it matters</div>
      <div>{why}</div>
      <div style="font-weight:700; margin:10px 0 6px;">Key drivers</div>
      <ul>
        {''.join(f'<li>{d}</li>' for d in drivers)}
      </ul>
      <div style="font-weight:700; margin:10px 0 6px;">Risks and reversals</div>
      <ul>
        {''.join(f'<li>{r}</li>' for r in risks)}
      </ul>
      <div style="font-weight:700; margin:10px 0 6px;">Next steps</div>
      <ul>
        {''.join(f'<li>{n}</li>' for n in next_steps)}
      </ul>
    </div>
    """

    # Optional legend
    if ctx.get("show_legend", False):
        legend = """
        <hr style="border-top:1px solid #eee; margin:10px 0;">
        <div style="font-size:13px;">
          <b>Legend</b> WR win rate, Avg and Med forward SPX returns, Disp mean absolute deviation, N sample size.
          Base bucket is VIX level before the spike. Magnitude is spike size. Regime uses your DMA. RSI uses your slider.
          p10, p50, p90 are percentile bands of forward returns for the chosen horizon.
        </div>
        """
        body += legend

    # Four-lights score for the filter panel
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

# Build context and render
bucket_now = latest["vix_base_bucket"] if latest is not None else None
mag_now = latest["spike_mag"] if latest is not None else None
regime_now = latest["regime"] if latest is not None else None
rsi_state = latest["oversold"] if latest is not None else None

ctx = {
    "wr_all": wr_all, "med_all": med_all, "avg_all": avg_all, "mad_all": mad_all, "n_all": n_all,
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

# ------------------------------- Summary top row ---------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Decision Box")
    card_box(summary_html)

with right:
    st.subheader("Filters")
    card_box(
        f"""
        <b>Spike threshold</b>: {spike_threshold}%<br>
        <b>Forward horizon</b>: {fwd_days} trading days<br>
        <b>RSI oversold</b>: ≤ {rsi_thresh}<br>
        <b>Regime MA</b>: {ma_window}-day<br>
        <b>Decades</b>: {', '.join(decade_filter) if decade_filter else 'None'}<br>
        <b>Four lights</b>: {lights} / 4
        """
    )

# ------------------------------- Plots grid --------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.subplots_adjust(wspace=0.35, hspace=0.45)

base_wr_plot = events.groupby("vix_base_bucket")[f"spx_fwd{fwd_days}_ret"].apply(winrate).reindex(
    ["0-12", "12-16", "16-20", "20-24", "24-30", "30+"]
)
barplot(
    axes[0, 0],
    base_wr_plot.index.tolist(),
    base_wr_plot.values.astype(float),
    PASTELS[:6],
    "Win Rate by VIX Base Level"
)

mag_wr_plot = events.groupby("spike_mag")[f"spx_fwd{fwd_days}_ret"].apply(winrate).reindex(
    ["Moderate (20-30%)", "Large (30-50%)", "Extreme (50%+)"]
)
mag_display = {
    "Moderate (20-30%)": "Moderate\n20-30%",
    "Large (30-50%)":    "Large\n30-50%",
    "Extreme (50%+)":    "Extreme\n50%+",
}
mag_categories_draw = [mag_display.get(k, k) for k in mag_wr_plot.index.tolist()]
ax_mag = axes[0, 1]
barplot(
    ax_mag,
    mag_categories_draw,
    mag_wr_plot.values.astype(float),
    PASTELS[6:9],
    "Win Rate by Spike Magnitude"
)
ax_mag.margins(x=0.05)
ax_mag.tick_params(axis="x", labelsize=10)

reg_wr_plot = events.groupby("regime")[f"spx_fwd{fwd_days}_ret"].apply(winrate).reindex(["Bull", "Bear"])
barplot(
    axes[0, 2],
    reg_wr_plot.index.tolist(),
    reg_wr_plot.values.astype(float),
    [PASTELS[0], PASTELS[2]],
    "Win Rate by Market Regime"
)

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

ov_wr_plot = events.groupby("oversold")[f"spx_fwd{fwd_days}_ret"].apply(winrate).reindex(["Oversold", "Not Oversold"])
barplot(
    axes[1, 1],
    ov_wr_plot.index.tolist(),
    ov_wr_plot.values.astype(float),
    [PASTELS[3], PASTELS[1]],
    "Win Rate by RSI Oversold"
)

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
    <b>Setup stats</b> same bucket, magnitude, regime. WR {fmt_pct(wr,1)}, Avg {fmt_pct(avg)}, Med {fmt_pct(med)}, Disp {fmt_pct(mad)}, N {n}.
    Bands p10 {fmt_pct(p10_c)}, p50 {fmt_pct(p50_c)}, p90 {fmt_pct(p90_c)}.
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
