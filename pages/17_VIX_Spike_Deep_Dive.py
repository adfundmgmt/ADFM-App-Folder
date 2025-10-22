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

def fmt_pct(x, digits=2, sign=False):
    if pd.isna(x):
        return "NA"
    s = f"{x:.{digits}f}%"
    if sign and x > 0:
        s = "+" + s
    return s

# ------------------------------- Sidebar -----------------------------------
st.title("VIX 20%+ Spike Deep Dive • ADFM")

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

# ------------------------------- Human summary -----------------------------
def decision_box(overall, bucket_wr, mag_wr, reg_wr, ov_wr, p10, p50, p90):
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

    # Four-lights score for quick read
    lights = 0
    if pd.notna(best_bucket_label) and str(best_bucket_label) in ["24-30", "30+"]:
        lights += 1
    if pd.notna(best_mag_label) and best_mag_label in ["Large (30-50%)", "Extreme (50%+)"]:
        lights += 1
    if pd.notna(bull_wr) and bull_wr >= 50:
        lights += 1
    # oversold helps if its win rate is higher than not-oversold
    if pd.notna(ov_yes) and pd.notna(ov_no) and ov_yes >= ov_no:
        lights += 1

    conclusion = "Short window favors a bounce after a ≥{}% VIX spike. Edge improves with elevated base VIX, larger spikes, bull regime, oversold tape.".format(spike_threshold)
    why = "You want a quick read on whether buying SPX for the next {} sessions has positive expectancy and when that tilt strengthens.".format(fwd_days)

    drivers = []
    drivers.append(f"Base case, WR {wr_all:.1f}%, median {fmt_pct(med_all)}, average {fmt_pct(avg_all)}, N {n_all}.")
    if pd.notna(best_bucket_wr):
        drivers.append(f"Starting VIX level, best bucket {best_bucket_label}, WR {best_bucket_wr:.1f}%.")
    if pd.notna(best_mag_wr):
        drivers.append(f"Spike size, favored magnitude {best_mag_label}, WR {best_mag_wr:.1f}%.")
    if pd.notna(bull_wr) and pd.notna(bear_wr):
        drivers.append(f"Regime split, Bull {bull_wr:.1f}% vs Bear {bear_wr:.1f}%.")
    if pd.notna(ov_yes) and pd.notna(ov_no):
        drivers.append(f"RSI filter, Oversold {ov_yes:.1f}% vs Not {ov_no:.1f}%.")

    risks = []
    risks.append("Bear regimes and low base VIX buckets weaken the edge.")
    risks.append("Extreme spikes widen both tails, gains can be larger, drawdowns can bite.")
    risks.append("Small average gains can be eaten by slippage and entry timing.")

    next_steps = []
    next_steps.append("Prefer trades when four lights align, elevated base VIX, larger spike, bull regime, oversold.")
    next_steps.append(f"Size for tails, respect p10 near {fmt_pct(p10)} for a {fwd_days}-day horizon.")
    next_steps.append("If two or more lights are off, cut size or pass.")

    body = f"""
    <div style="font-size:14px;">
      <div style="font-weight:700; margin-bottom:6px;">Conclusion</div>
      <div>{conclusion}</div>
      <div style="font-weight:700; margin:10px 0 6px;">Why it matters</div>
      <div>{why}</div>
      <div style="font-weight:700; margin:10px 0 6px;">Key drivers</div>
      <ul>
        {''.join(f'<li>{d}</li>' for d in drivers)}
      </ul>
      <div><b>Expected range</b>, p10 {fmt_pct(p10)}, p50 {fmt_pct(p50)}, p90 {fmt_pct(p90)}.</div>
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

    # optional legend toggle
    if show_legend:
        legend = """
        <hr style="border-top:1px solid #eee; margin:10px 0;">
        <div style="font-size:13px;">
          <b>Legend</b> WR win rate, share of cases with positive forward return. Avg and Med are average and median forward SPX returns.
          Disp is mean absolute deviation as a simple volatility proxy. N is sample size.
          Base bucket is VIX level before the spike. Magnitude is size of the VIX jump.
          Regime is SPX vs your selected moving average. RSI oversold uses your slider threshold.
          Outcome bands p10, p50, p90 show downside, typical, and strong upside for the chosen horizon.
        </div>
        """
        body += legend

    return body, lights

# ------------------------------- Compute stats for summary ------------------
overall = events[f"spx_fwd{fwd_days}_ret"]
bucket_wr = events.groupby("vix_base_bucket")[f"spx_fwd{fwd_days}_ret"].apply(winrate) if not events.empty else pd.Series(dtype=float)
mag_wr = events.groupby("spike_mag")[f"spx_fwd{fwd_days}_ret"].apply(winrate) if not events.empty else pd.Series(dtype=float)
reg_wr = events.groupby("regime")[f"spx_fwd{fwd_days}_ret"].apply(winrate) if not events.empty else pd.Series(dtype=float)
ov_wr = events.groupby("oversold")[f"spx_fwd{fwd_days}_ret"].apply(winrate) if not events.empty else pd.Series(dtype=float)
p10, p50, p90 = pct_bands(overall) if len(overall) else (np.nan, np.nan, np.nan)

summary_html, lights = decision_box(overall, bucket_wr, mag_wr, reg_wr, ov_wr, p10, p50, p90)

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

# 2) Win rate by spike magnitude  (two-line labels)
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

# 3) Win rate by regime
reg_wr_plot = events.groupby("regime")[f"spx_fwd{fwd_days}_ret"].apply(winrate).reindex(["Bull", "Bear"])
barplot(
    axes[0, 2],
    reg_wr_plot.index.tolist(),
    reg_wr_plot.values.astype(float),
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
ov_wr_plot = events.groupby("oversold")[f"spx_fwd{fwd_days}_ret"].apply(winrate).reindex(["Oversold", "Not Oversold"])
barplot(
    axes[1, 1],
    ov_wr_plot.index.tolist(),
    ov_wr_plot.values.astype(float),
    [PASTELS[3], PASTELS[1]],
    "Win Rate by RSI Oversold"
)

# 6) Setup Distribution: histogram of comps
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

    edge_score = np.nan
    if pd.notna(wr) and pd.notna(med) and pd.notna(mad) and mad > 0:
        edge_score = (wr - 50.0) * (med / mad)

    text = f"""
    <b>Latest event</b> {dt}. VIX {vix_now:.2f} from base {vix_base:.2f}, spike {vix_spike:.1f}%.
    SPX {spx_now:.2f}, RSI14 {rsi_now:.1f} ({over}), regime {regime_now}.
    Setup, base {bucket}, magnitude {magcat}.
    <br><br>
    <b>Setup stats</b> same bucket, magnitude, regime. WR {fmt_pct(wr, 1)}, Avg {fmt_pct(avg)}, Med {fmt_pct(med)}, Disp {fmt_pct(mad)}, N {n}.
    Bands p10 {fmt_pct(p10_c)}, p50 {fmt_pct(p50_c)}, p90 {fmt_pct(p90_c)}.
    Edge score {edge_score:.2f}.
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
