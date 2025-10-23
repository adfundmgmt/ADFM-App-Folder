# vix_spike_deep_dive.py
# ADFM Analytics Platform — VIX Spike Deep Dive (simplified)
# Light theme, pastel palette, matplotlib only. Six fixed charts.

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
    "#A8DADC","#F4A261","#90BE6D","#FFCC99","#BDE0FE",
    "#CDB4DB","#FFD6A5","#E2F0CB","#F1C0E8","#B9FBC0"
]
GRID_COLOR = "#e6e6e6"
BAR_EDGE = "#666666"
TEXT_COLOR = "#222222"

# ------------------------------- Utilities ---------------------------------
def compute_rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/n, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def winrate(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float((x > 0).mean() * 100) if len(x) else np.nan

def pct_bands(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return np.percentile(x, [10, 50, 90]) if len(x) else (np.nan, np.nan, np.nan)

def bucket_vix_base(v: float) -> str:
    bins = [0, 12, 16, 20, 24, 30, 999]
    labels = ["0-12","12-16","16-20","20-24","24-30","30+"]
    return pd.cut([v], bins=bins, labels=labels, right=False)[0]

def bucket_spike_mag(frac: float) -> str:
    if frac < 0.30:  # 30%
        return "Moderate (20-30%)"
    elif frac < 0.50:
        return "Large (30-50%)"
    else:
        return "Extreme (50%+)"

def decade_label(ts: pd.Timestamp) -> str:
    y = ts.year
    return f"{(y//10)*10}s"

def fmt_pct(x, d=1):
    return "NA" if pd.isna(x) else f"{x:.{d}f}%"

def barplot(ax, cats, vals, colors, title, ylabel):
    ax.bar(cats, vals, color=colors[:len(cats)], edgecolor=BAR_EDGE)
    ax.axhline(50, linestyle="--", linewidth=1, color="#888888")
    ax.set_title(title, color=TEXT_COLOR, fontsize=12, pad=8)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    ymax = np.nanmax(vals) if len(vals) else 0
    top = 0 if not np.isfinite(ymax) else max(70, ymax + 10)
    ax.set_ylim(0, top if top > 0 else 70)
    for i, v in enumerate(vals):
        if not pd.isna(v):
            ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", color=TEXT_COLOR, fontsize=10)

def card_box(html):
    st.markdown(
        f"""
        <div style="border:1px solid #e0e0e0; border-radius:10px; padding:14px; background:#fafafa; color:{TEXT_COLOR}; font-size:14px; line-height:1.35;">
          {html}
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------- Data --------------------------------------
@st.cache_data(show_spinner=False, ttl=60*60)
def load_history(start="1990-01-01"):
    vix = yf.download("^VIX", start=start, auto_adjust=False, progress=False)
    spx = yf.download("^GSPC", start=start, auto_adjust=False, progress=False)
    vix = vix.rename(columns=str.lower)
    spx = spx.rename(columns=str.lower)

    df = pd.DataFrame(index=vix.index)
    df["vix_close"] = vix["close"]
    df["spx_close"] = spx["close"].reindex(df.index).ffill()
    df["vix_pctchg"] = df["vix_close"].pct_change()
    df["rsi14"] = compute_rsi(df["spx_close"], 14)
    return df.dropna(subset=["vix_close", "spx_close"])

# ------------------------------- Sidebar -----------------------------------
st.title("VIX Spike Deep Dive")

with st.sidebar:
    st.header("Controls")
    start_date = st.date_input("History start", value=datetime(1990, 1, 1))
    fwd_days = st.number_input("Forward horizon (days)", min_value=1, max_value=30, value=2, step=1)
    spike_threshold = st.slider("VIX spike threshold (%)", 20, 100, 20, step=5)
    rsi_thresh = st.slider("Oversold RSI threshold", 10, 40, 30, step=1)
    ma_window = st.selectbox("Regime MA window", [100, 150, 200], index=2)

df = load_history(start=str(start_date))

# Forward returns and regime
df[f"spx_next{fwd_days}"] = df["spx_close"].shift(-fwd_days)
df[f"spx_fwd{fwd_days}_ret"] = (df[f"spx_next{fwd_days}"] / df["spx_close"] - 1.0) * 100.0
df[f"spx_{ma_window}dma"] = df["spx_close"].rolling(ma_window).mean()
df["regime"] = np.where(df["spx_close"] >= df[f"spx_{ma_window}dma"], "Bull", "Bear")

# Build events
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

# Decade filter
all_decades = sorted(events["decade"].unique().tolist()) if not events.empty else []
decade_filter = st.sidebar.multiselect("Decades to include", options=all_decades, default=all_decades)
if decade_filter:
    events = events[events["decade"].isin(decade_filter)]

# Latest and comps
latest = events.iloc[-1] if not events.empty else None
if latest is not None:
    bucket_now = latest["vix_base_bucket"]
    mag_now = latest["spike_mag"]
    regime_now = latest["regime"]
    comps = events[
        (events["vix_base_bucket"] == bucket_now) &
        (events["spike_mag"] == mag_now) &
        (events["regime"] == regime_now)
    ].copy()
else:
    bucket_now = mag_now = regime_now = None
    comps = pd.DataFrame(columns=events.columns)

# ------------------------------- Summary stats -----------------------------
overall = events[f"spx_fwd{fwd_days}_ret"] if not events.empty else pd.Series(dtype=float)
wr_all = winrate(overall)
avg_all = float(overall.mean()) if len(overall) else np.nan
med_all = float(overall.median()) if len(overall) else np.nan
p10, p50, p90 = pct_bands(overall)

bucket_wr = events.groupby("vix_base_bucket")[f"spx_fwd{fwd_days}_ret"].apply(winrate) if not events.empty else pd.Series(dtype=float)
mag_wr = events.groupby("spike_mag")[f"spx_fwd{fwd_days}_ret"].apply(winrate) if not events.empty else pd.Series(dtype=float)
reg_wr = events.groupby("regime")[f"spx_fwd{fwd_days}_ret"].apply(winrate) if not events.empty else pd.Series(dtype=float)
ov_wr  = events.groupby("oversold")[f"spx_fwd{fwd_days}_ret"].apply(winrate) if not events.empty else pd.Series(dtype=float)

bull_wr = reg_wr.get("Bull", np.nan)
bear_wr = reg_wr.get("Bear", np.nan)
ov_yes = ov_wr.get("Oversold", np.nan)
ov_no  = ov_wr.get("Not Oversold", np.nan)

# ------------------------------- Decision Box ------------------------------
def decision_text() -> str:
    if pd.isna(wr_all):
        return "<b>Conclusion</b><br>No qualifying events under the current filters."

    edge = "tactical long bias" if (wr_all >= 53 and med_all >= 0) else ("mixed edge" if 47 <= wr_all < 53 else "weak edge")
    parts = [
        "<b>Conclusion</b>",
        f"After a ≥{spike_threshold}% VIX jump, the {fwd_days}-day base case shows {fmt_pct(wr_all)} hit rate, median {fmt_pct(med_all)}, average {fmt_pct(avg_all)}.",
        f"Expected band p10 {fmt_pct(p10)}, p50 {fmt_pct(p50)}, p90 {fmt_pct(p90)}.",
        f"Regime split Bull {fmt_pct(bull_wr)} vs Bear {fmt_pct(bear_wr)}. RSI filter Oversold {fmt_pct(ov_yes)} vs Not {fmt_pct(ov_no)}.",
        f"Working view, {edge}."
    ]
    return "<br>".join(parts)

col1, col2 = st.columns([1.8, 1])
with col1:
    st.subheader("Decision Box")
    card_box(decision_text())
with col2:
    st.subheader("Filters")
    lights = 0
    if bucket_now in ["24-30","30+"]: lights += 1
    if mag_now in ["Large (30-50%)","Extreme (50%+)"]: lights += 1
    if regime_now == "Bull": lights += 1
    # Prefer Oversold only if its win rate is at least as good
    if pd.notna(ov_yes) and pd.notna(ov_no) and ov_yes >= ov_no: lights += 1

    details = f"""
    <b>Spike threshold</b>: {spike_threshold}%<br>
    <b>Forward horizon</b>: {fwd_days} trading days<br>
    <b>RSI oversold</b>: ≤ {rsi_thresh}<br>
    <b>Regime MA</b>: {ma_window}-day<br>
    <b>Decades</b>: {', '.join(decade_filter) if decade_filter else 'All'}<br>
    <b>Four lights</b>: {lights} / 4
    """
    card_box(details)

# ------------------------------- Six Charts --------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.subplots_adjust(wspace=0.35, hspace=0.45)
ylabel = f"{fwd_days}-Day Win Rate (%)"

# 1) Win rate by VIX base bucket
order_base = ["0-12","12-16","16-20","20-24","24-30","30+"]
base_wr_plot = bucket_wr.reindex(order_base)
barplot(axes[0,0], order_base, base_wr_plot.values.astype(float) if base_wr_plot is not None else [], PASTELS, "Win Rate by VIX Base Level", ylabel)

# 2) Win rate by spike magnitude
order_mag = ["Moderate (20-30%)","Large (30-50%)","Extreme (50%+)"]
mag_wr_plot = mag_wr.reindex(order_mag)
cats_mag = ["Moderate\n20-30%","Large\n30-50%","Extreme\n50%+"]
barplot(axes[0,1], cats_mag, mag_wr_plot.values.astype(float) if mag_wr_plot is not None else [], PASTELS[6:], "Win Rate by Spike Magnitude", ylabel)

# 3) Win rate by regime
order_reg = ["Bull","Bear"]
reg_wr_plot = reg_wr.reindex(order_reg)
barplot(axes[0,2], order_reg, reg_wr_plot.values.astype(float) if reg_wr_plot is not None else [], [PASTELS[0],PASTELS[2]], "Win Rate by Market Regime", ylabel)

# 4) Scatter: VIX base vs forward returns by decade
ax = axes[1,0]
if not events.empty:
    decs = sorted(events["decade"].unique())
    for i, dec in enumerate(decs):
        e = events[events["decade"] == dec]
        ax.scatter(
            e["vix_base"],
            e[f"spx_fwd{fwd_days}_ret"],
            s=28, alpha=0.85, label=dec, color=PASTELS[i % len(PASTELS)],
            edgecolors=BAR_EDGE, linewidths=0.3
        )
ax.axhline(0, color="#888888", linewidth=1)
ax.set_title("VIX Base Level vs Forward Returns", color=TEXT_COLOR, fontsize=12, pad=8)
ax.set_xlabel("VIX Level Before Spike", color=TEXT_COLOR)
ax.set_ylabel(f"SPX {fwd_days}-Day Return (%)", color=TEXT_COLOR)
ax.grid(color=GRID_COLOR, linewidth=0.6)
if not events.empty:
    ax.legend(fontsize=8, frameon=False, ncol=2)

# 5) Win rate by RSI oversold vs not
ov_order = ["Oversold","Not Oversold"]
ov_wr_plot = ov_wr.reindex(ov_order) if not events.empty else pd.Series([np.nan, np.nan], index=ov_order)
barplot(axes[1,1], ov_order, ov_wr_plot.values.astype(float), [PASTELS[3],PASTELS[1]], "Win Rate by RSI Filter", ylabel)

# 6) Histogram of comps for the latest setup
axh = axes[1,2]
axh.set_title("Setup Distribution, Latest Analog Set", color=TEXT_COLOR, fontsize=12, pad=8)
axh.set_xlabel(f"SPX {fwd_days}-Day Return (%)", color=TEXT_COLOR)
axh.set_ylabel("Frequency", color=TEXT_COLOR)
axh.grid(color=GRID_COLOR, linewidth=0.6)
vals = comps[f"spx_fwd{fwd_days}_ret"].dropna().values if not comps.empty else np.array([])
if vals.size > 0:
    bins = min(20, max(8, int(np.sqrt(len(vals)))))
    axh.hist(vals, bins=bins, edgecolor=BAR_EDGE)
    axh.axvline(0, color="#888888", linewidth=1)
else:
    axh.text(0.5, 0.5, "No matching analogs under current filters", ha="center", va="center", transform=axh.transAxes, color="#555555")

st.pyplot(fig, clear_figure=True)

# ------------------------------- Latest event box --------------------------
st.subheader("Latest Event Context")
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
    over = latest["oversold"]
    regime_curr = latest["regime"]
    bucket = latest["vix_base_bucket"]
    magcat = latest["spike_mag"]

    wr = winrate(comps[f"spx_fwd{fwd_days}_ret"]) if not comps.empty else np.nan
    avg = float(comps[f"spx_fwd{fwd_days}_ret"].mean()) if not comps.empty else np.nan
    med = float(comps[f"spx_fwd{fwd_days}_ret"].median()) if not comps.empty else np.nan
    p10_c, p50_c, p90_c = pct_bands(comps[f"spx_fwd{fwd_days}_ret"]) if not comps.empty else (np.nan, np.nan, np.nan)
    n = len(comps) if not comps.empty else 0

    html = f"""
    <b>Date</b> {dt}. VIX {vix_now:.2f} from base {vix_base:.2f}, spike {vix_spike:.1f}%.
    SPX {spx_now:.2f}, RSI14 {rsi_now:.1f} ({over}), regime {regime_curr}.<br>
    Setup base {bucket}, magnitude {magcat}.<br><br>
    <b>Analog set</b> WR {fmt_pct(wr)}, Avg {fmt_pct(avg)}, Med {fmt_pct(med)}, N {n}. Bands p10 {fmt_pct(p10_c)}, p50 {fmt_pct(p50_c)}, p90 {fmt_pct(p90_c)}.
    """
    card_box(html)

    if n > 0:
        show_cols = ["vix_close","vix_base","vix_pctchg","spx_close",f"spx_fwd{fwd_days}_ret","rsi14"]
        analogs = (comps[show_cols]
                   .rename(columns={
                       "vix_close":"VIX",
                       "vix_base":"VIX_Base",
                       "vix_pctchg":"VIX_Spike_Frac",
                       "spx_close":"SPX",
                       f"spx_fwd{fwd_days}_ret":f"SPX_Fwd{fwd_days}_Ret_%",
                       "rsi14":"RSI14"
                   }).copy())
        analogs["VIX_Spike_%"] = (analogs["VIX_Spike_Frac"] * 100.0).round(2)
        analogs.drop(columns=["VIX_Spike_Frac"], inplace=True)
        analogs = analogs.round(2).sort_index(ascending=False).head(10)
        st.markdown("**Nearest analogs, same setup, last 10 occurrences**")
        st.dataframe(analogs, use_container_width=True)

latest_context_box()

# ------------------------------- Events table ------------------------------
with st.expander("Show events table"):
    if not events.empty:
        cols = ["vix_close","vix_base","vix_pctchg","spx_close",f"spx_fwd{fwd_days}_ret","vix_base_bucket","spike_mag","regime","rsi14","oversold","decade"]
        tbl = events[cols].copy()
        tbl.rename(columns={
            "vix_close":"VIX","vix_base":"VIX_Base","vix_pctchg":"VIX_Spike_Fr","spx_close":"SPX","rsi14":"RSI14"
        }, inplace=True)
        tbl["VIX_Spike_%"] = (tbl["VIX_Spike_Fr"] * 100.0).round(2)
        tbl.drop(columns=["VIX_Spike_Fr"], inplace=True)
        st.dataframe(tbl.round(2), use_container_width=True)
    else:
        st.write("No events under current filters.")

# ------------------------------- Footer ------------------------------------
st.caption("ADFM Analytics Platform · VIX Spike Deep Dive · Data source: Yahoo Finance")
