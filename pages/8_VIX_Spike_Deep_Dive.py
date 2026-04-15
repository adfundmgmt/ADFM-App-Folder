# vix_spike_deep_dive_rebuilt.py
# ADFM Analytics Platform | VIX Event Deep Dive
# Rebuilt as an event-study and analog engine using Yahoo Finance only

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------- Page config -------------------------------
st.set_page_config(page_title="VIX Spike Deep Dive", layout="wide")
plt.style.use("default")

# ------------------------------- Style -------------------------------------
PASTELS = [
    "#A8DADC", "#F4A261", "#90BE6D", "#FFCC99", "#BDE0FE",
    "#CDB4DB", "#FFD6A5", "#E2F0CB", "#F1C0E8", "#B9FBC0"
]
GRID_COLOR = "#e6e6e6"
BAR_EDGE = "#666666"
TEXT_COLOR = "#222222"

# ------------------------------- Helpers -----------------------------------
def fmt_pct(x, digits=1):
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}%"

def fmt_num(x, digits=2):
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"

def compute_rsi(series, n=14):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1 / n, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def winrate(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return 100.0 * (s > 0).mean()

def pct_bands(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan, np.nan, np.nan
    return np.percentile(s, [10, 50, 90])

def max_drawdown_from_peak(price):
    peak = price.cummax()
    dd = price / peak - 1.0
    return dd * 100.0

def zscore(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)

def percentile_rank_rolling(series, window):
    vals = []
    arr = series.values
    idx = series.index
    for i in range(len(arr)):
        if i < window - 1 or pd.isna(arr[i]):
            vals.append(np.nan)
            continue
        chunk = arr[i - window + 1:i + 1]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk) == 0:
            vals.append(np.nan)
            continue
        vals.append(100.0 * (chunk <= arr[i]).mean())
    return pd.Series(vals, index=idx)

def future_return(price, h):
    return (price.shift(-h) / price - 1.0) * 100.0

def make_event_label(row):
    return f"{row['event_type']} | {row['context_label']}"

def safe_std(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    v = x.std()
    return np.nan if v == 0 else v

def render_metric_box(title, lines):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        for line in lines:
            st.markdown(line)

@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_market_history(start="1990-01-01"):
    tickers = {
        "vix": "^VIX",
        "spx": "^GSPC",
        "qqq": "QQQ",
        "iwm": "IWM",
        "hyg": "HYG",
        "ief": "IEF",
        "tlt": "TLT",
    }

    out = {}
    for key, ticker in tickers.items():
        df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
        if df.empty:
            out[key] = pd.DataFrame()
        else:
            df = df.rename(columns=str.lower)
            out[key] = df

    if out["vix"].empty or out["spx"].empty:
        return pd.DataFrame()

    idx = out["spx"].index.union(out["vix"].index).sort_values()
    df = pd.DataFrame(index=idx)

    df["vix_close"] = out["vix"]["close"].reindex(idx).ffill()
    df["spx_close"] = out["spx"]["close"].reindex(idx).ffill()
    df["qqq_close"] = out["qqq"]["close"].reindex(idx).ffill() if not out["qqq"].empty else np.nan
    df["iwm_close"] = out["iwm"]["close"].reindex(idx).ffill() if not out["iwm"].empty else np.nan
    df["hyg_close"] = out["hyg"]["close"].reindex(idx).ffill() if not out["hyg"].empty else np.nan
    df["ief_close"] = out["ief"]["close"].reindex(idx).ffill() if not out["ief"].empty else np.nan
    df["tlt_close"] = out["tlt"]["close"].reindex(idx).ffill() if not out["tlt"].empty else np.nan

    df = df.dropna(subset=["vix_close", "spx_close"]).copy()

    df["vix_prev"] = df["vix_close"].shift(1)
    df["vix_pctchg"] = df["vix_close"] / df["vix_prev"] - 1.0
    df["vix_abs_pctchg"] = df["vix_pctchg"].abs()
    df["vix_change"] = df["vix_close"].diff()
    df["vix_z20"] = zscore(df["vix_change"], 20)
    df["vix_z60"] = zscore(df["vix_change"], 60)

    df["spx_ret_1d"] = df["spx_close"].pct_change() * 100.0
    df["spx_ret_3d"] = df["spx_close"].pct_change(3) * 100.0
    df["spx_ret_5d"] = df["spx_close"].pct_change(5) * 100.0
    df["spx_ret_10d"] = df["spx_close"].pct_change(10) * 100.0

    df["spx_20dma"] = df["spx_close"].rolling(20).mean()
    df["spx_50dma"] = df["spx_close"].rolling(50).mean()
    df["spx_200dma"] = df["spx_close"].rolling(200).mean()

    df["dist_20dma"] = (df["spx_close"] / df["spx_20dma"] - 1.0) * 100.0
    df["dist_50dma"] = (df["spx_close"] / df["spx_50dma"] - 1.0) * 100.0
    df["dist_200dma"] = (df["spx_close"] / df["spx_200dma"] - 1.0) * 100.0

    df["ma20_slope_5d"] = (df["spx_20dma"] / df["spx_20dma"].shift(5) - 1.0) * 100.0
    df["ma50_slope_10d"] = (df["spx_50dma"] / df["spx_50dma"].shift(10) - 1.0) * 100.0

    df["rsi14"] = compute_rsi(df["spx_close"], 14)
    df["spx_drawdown_pct"] = max_drawdown_from_peak(df["spx_close"])

    df["rv10"] = df["spx_close"].pct_change().rolling(10).std() * np.sqrt(252) * 100.0
    df["rv20"] = df["spx_close"].pct_change().rolling(20).std() * np.sqrt(252) * 100.0
    df["rv20_pctile_252"] = percentile_rank_rolling(df["rv20"], 252)

    df["qqq_iwm_ratio"] = df["qqq_close"] / df["iwm_close"]
    df["qqq_iwm_5d"] = (df["qqq_iwm_ratio"] / df["qqq_iwm_ratio"].shift(5) - 1.0) * 100.0

    df["hyg_ief_ratio"] = df["hyg_close"] / df["ief_close"]
    df["hyg_ief_5d"] = (df["hyg_ief_ratio"] / df["hyg_ief_ratio"].shift(5) - 1.0) * 100.0
    df["tlt_ret_1d"] = df["tlt_close"].pct_change() * 100.0
    df["ief_ret_1d"] = df["ief_close"].pct_change() * 100.0

    for h in [1, 2, 3, 5, 10, 20]:
        df[f"spx_fwd_{h}d"] = future_return(df["spx_close"], h)

    return df.dropna(how="all")

def classify_context(row, oversold_rsi=30, drawdown_cut=-8.0):
    above_200 = pd.notna(row["dist_200dma"]) and row["dist_200dma"] >= 0
    oversold = pd.notna(row["rsi14"]) and row["rsi14"] <= oversold_rsi
    deep_dd = pd.notna(row["spx_drawdown_pct"]) and row["spx_drawdown_pct"] <= drawdown_cut
    trend_down = pd.notna(row["ma20_slope_5d"]) and row["ma20_slope_5d"] < 0

    if above_200 and oversold:
        return "Bull Pullback"
    if above_200 and not oversold:
        return "Bull Shock"
    if (not above_200) and deep_dd and oversold:
        return "Washout"
    if (not above_200) and trend_down:
        return "Trend Break"
    return "Mixed"

def classify_outcome(row):
    r2 = row.get("spx_fwd_2d", np.nan)
    r5 = row.get("spx_fwd_5d", np.nan)
    r10 = row.get("spx_fwd_10d", np.nan)

    if pd.isna(r2) or pd.isna(r5):
        return "Incomplete"

    if r2 > 0 and r5 > 0:
        return "Exhaustion / Bounce"
    if r2 <= 0 and r5 <= 0:
        return "Continuation"
    if r2 > 0 and r5 <= 0:
        return "Failed Bounce"
    if r2 <= 0 and r5 > 0:
        return "Delayed Recovery"
    if pd.notna(r10) and r10 > 0:
        return "Recovery"
    return "Mixed"

def build_events(
    df,
    event_mode="Absolute % move",
    abs_move_threshold=25.0,
    z_threshold=2.0,
    spx_down_confirm=-1.5,
    lookback_gap=5,
    oversold_rsi=30,
    drawdown_cut=-8.0
):
    if df.empty:
        return pd.DataFrame()

    events = df.copy()
    events["vix_base"] = events["vix_close"].shift(1)

    if event_mode == "Absolute % move":
        mask = events["vix_abs_pctchg"] >= (abs_move_threshold / 100.0)
    elif event_mode == "VIX z-score":
        mask = events["vix_z20"].abs() >= z_threshold
    elif event_mode == "Panic day":
        mask = (
            (events["vix_pctchg"] >= (abs_move_threshold / 100.0)) &
            (events["spx_ret_1d"] <= spx_down_confirm)
        )
    elif event_mode == "Relief day":
        mask = (
            (events["vix_pctchg"] <= -(abs_move_threshold / 100.0)) &
            (events["spx_ret_5d"].shift(1) <= -3.0)
        )
    else:
        mask = events["vix_abs_pctchg"] >= (abs_move_threshold / 100.0)

    events = events[mask].copy()
    if events.empty:
        return events

    events["event_type"] = np.where(events["vix_pctchg"] >= 0, "Panic", "Relief")
    events["move_mag"] = events["vix_abs_pctchg"] * 100.0
    events["context_label"] = events.apply(
        lambda row: classify_context(row, oversold_rsi=oversold_rsi, drawdown_cut=drawdown_cut), axis=1
    )
    events["outcome_label"] = events.apply(classify_outcome, axis=1)
    events["event_label"] = events.apply(make_event_label, axis=1)

    if lookback_gap > 0:
        keep_idx = []
        last_kept = None
        for idx in events.index:
            if last_kept is None or (idx - last_kept).days > lookback_gap:
                keep_idx.append(idx)
                last_kept = idx
        events = events.loc[keep_idx].copy()

    return events

def compute_similarity(events, latest_idx, feature_cols):
    if events.empty or latest_idx not in events.index:
        return pd.Series(dtype=float)

    X = events[feature_cols].copy()
    latest = X.loc[latest_idx]

    stds = X.apply(safe_std, axis=0)
    scores = pd.Series(index=events.index, dtype=float)

    for idx, row in X.iterrows():
        diffs = []
        for col in feature_cols:
            a = row[col]
            b = latest[col]
            s = stds[col]
            if pd.isna(a) or pd.isna(b) or pd.isna(s):
                continue
            diffs.append(abs(a - b) / s)
        scores.loc[idx] = np.nan if len(diffs) == 0 else np.mean(diffs)

    return scores.sort_values()

def summarize_setup(events, label_col, ret_col):
    if events.empty:
        return pd.DataFrame()

    rows = []
    for name, g in events.groupby(label_col):
        s = pd.to_numeric(g[ret_col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        p10, p50, p90 = pct_bands(s)
        rows.append({
            label_col: name,
            "N": len(s),
            "WR_%": winrate(s),
            "Avg_%": s.mean(),
            "Med_%": s.median(),
            "P10_%": p10,
            "P50_%": p50,
            "P90_%": p90,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["N", "WR_%"], ascending=[False, False])
    return out

def build_path_stats(events, horizons):
    data = []
    for h in horizons:
        col = f"spx_fwd_{h}d"
        s = pd.to_numeric(events[col], errors="coerce").dropna()
        if len(s) == 0:
            data.append({"h": h, "mean": np.nan, "median": np.nan, "p25": np.nan, "p75": np.nan, "p10": np.nan, "p90": np.nan})
        else:
            data.append({
                "h": h,
                "mean": s.mean(),
                "median": s.median(),
                "p25": np.percentile(s, 25),
                "p75": np.percentile(s, 75),
                "p10": np.percentile(s, 10),
                "p90": np.percentile(s, 90),
            })
    return pd.DataFrame(data)

# ------------------------------- Sidebar -----------------------------------
st.title("VIX Spike Deep Dive")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
Purpose: classify large VIX events, score historical analogs, and map forward SPX paths.

What it covers
• Panic versus relief events
• Context-aware event study
• Analog matching on tape, trend, drawdown, vol, and cross-asset proxies
• Forward path cones and failure rates

Data source
• Yahoo Finance
        """
    )
    st.divider()

    start_date = st.date_input("History start", value=datetime(1998, 1, 1))
    event_mode = st.selectbox(
        "Event definition",
        ["Absolute % move", "VIX z-score", "Panic day", "Relief day"],
        index=0
    )
    abs_move_threshold = st.slider("Absolute VIX move threshold (%)", 10, 80, 15, step=5)
    z_threshold = st.slider("VIX z-score threshold", 1.0, 4.0, 2.0, step=0.1)
    spx_down_confirm = st.slider("SPX down confirm for panic day (%)", -5.0, 0.0, -1.5, step=0.1)
    dedupe_gap = st.slider("Minimum gap between events (calendar days)", 0, 20, 5, step=1)
    oversold_rsi = st.slider("Oversold RSI threshold", 10, 40, 30, step=1)
    drawdown_cut = st.slider("Washout drawdown threshold (%)", -25.0, -3.0, -8.0, step=1.0)
    forward_focus = st.selectbox("Primary forward horizon", [1, 2, 3, 5, 10, 20], index=3)
    analog_count = st.slider("Nearest analogs to show", 3, 15, 8, step=1)
    event_type_filter = st.multiselect("Event types", ["Panic", "Relief"], default=["Panic", "Relief"])
    context_filter = st.multiselect(
        "Contexts",
        ["Bull Pullback", "Bull Shock", "Washout", "Trend Break", "Mixed"],
        default=["Bull Pullback", "Bull Shock", "Washout", "Trend Break", "Mixed"]
    )

# ------------------------------- Data load ---------------------------------
df = load_market_history(start=str(start_date))

if df.empty:
    st.error("Failed to load VIX and SPX history.")
    st.stop()

events = build_events(
    df=df,
    event_mode=event_mode,
    abs_move_threshold=abs_move_threshold,
    z_threshold=z_threshold,
    spx_down_confirm=spx_down_confirm,
    lookback_gap=dedupe_gap,
    oversold_rsi=oversold_rsi,
    drawdown_cut=drawdown_cut
)

if events.empty:
    st.warning("No events matched the current definition.")
    st.stop()

events = events[events["event_type"].isin(event_type_filter)].copy()
events = events[events["context_label"].isin(context_filter)].copy()

if events.empty:
    st.warning("No events remained after the current filters.")
    st.stop()

latest_idx = events.index.max()
latest = events.loc[latest_idx].copy()

feature_cols = [
    "vix_close",
    "move_mag",
    "spx_ret_1d",
    "spx_ret_5d",
    "dist_20dma",
    "dist_50dma",
    "dist_200dma",
    "spx_drawdown_pct",
    "rsi14",
    "rv20",
    "rv20_pctile_252",
    "qqq_iwm_5d",
    "hyg_ief_5d",
    "ief_ret_1d",
]

sim_scores = compute_similarity(events, latest_idx, feature_cols)
sim_scores = sim_scores.drop(index=latest_idx, errors="ignore")
analogs_idx = sim_scores.head(analog_count).index.tolist()
analogs = events.loc[analogs_idx].copy() if len(analogs_idx) else pd.DataFrame()

latest_setup = events[
    (events["event_type"] == latest["event_type"]) &
    (events["context_label"] == latest["context_label"])
].copy()

latest_wr_setup = winrate(latest_setup[f"spx_fwd_{forward_focus}d"])
latest_median_setup = latest_setup[f"spx_fwd_{forward_focus}d"].median()

latest_continuation_rate = (
    100.0 * (latest_setup["outcome_label"] == "Continuation").mean()
    if not latest_setup.empty else np.nan
)

if pd.notna(latest_wr_setup) and pd.notna(latest_median_setup):
    if latest["event_type"] == "Panic":
        if latest_wr_setup >= 60 and latest_median_setup > 0:
            verdict = "Exhaustion bias"
        elif latest_continuation_rate >= 50:
            verdict = "Continuation risk"
        else:
            verdict = "Transition / mixed"
    else:
        if latest_wr_setup >= 60 and latest_median_setup > 0:
            verdict = "Follow-through bias"
        elif latest_continuation_rate >= 40:
            verdict = "Complacency risk"
        else:
            verdict = "Transition / mixed"
else:
    verdict = "Insufficient sample"

# ------------------------------- Top cards ---------------------------------
col1, col2 = st.columns([1.6, 1.0])

with col1:
    render_metric_box(
        "Latest Event Snapshot",
        [
            f"**Date**: {latest_idx.strftime('%Y-%m-%d')}",
            f"**Type**: {latest['event_type']}",
            f"**Context**: {latest['context_label']}",
            f"**Verdict**: {verdict}",
            f"**VIX**: {fmt_num(latest['vix_close'])} | **Daily move**: {fmt_pct(latest['vix_pctchg'] * 100.0)}",
            f"**SPX 1D**: {fmt_pct(latest['spx_ret_1d'])} | **SPX 5D**: {fmt_pct(latest['spx_ret_5d'])}",
            f"**Drawdown from peak**: {fmt_pct(latest['spx_drawdown_pct'])} | **RSI14**: {fmt_num(latest['rsi14'], 1)}",
            f"**Dist to 20DMA**: {fmt_pct(latest['dist_20dma'])} | **Dist to 50DMA**: {fmt_pct(latest['dist_50dma'])} | **Dist to 200DMA**: {fmt_pct(latest['dist_200dma'])}",
            f"**RV20**: {fmt_pct(latest['rv20'])} | **RV20 percentile**: {fmt_pct(latest['rv20_pctile_252'])}",
            f"**Setup sample**: WR {fmt_pct(latest_wr_setup)}, median {fmt_pct(latest_median_setup)}, continuation risk {fmt_pct(latest_continuation_rate)}",
        ]
    )

with col2:
    render_metric_box(
        "Current Definition",
        [
            f"**Event mode**: {event_mode}",
            f"**Abs threshold**: {abs_move_threshold}%",
            f"**Z threshold**: {z_threshold}",
            f"**SPX confirm**: {spx_down_confirm}%",
            f"**Dedup gap**: {dedupe_gap} days",
            f"**Primary horizon**: {forward_focus} days",
            f"**Included events**: {len(events)}",
            f"**Latest setup label**: {latest['event_label']}",
        ]
    )

# ------------------------------- Dynamic readout ---------------------------
st.subheader("What just happened and what usually comes next")

same_setup = latest_setup.copy()

text = []

if latest["event_type"] == "Panic":
    text.append(
        f"This was a panic event inside a {latest['context_label'].lower()} context. "
        f"The market came in with SPX {fmt_pct(latest['spx_ret_5d'])} over five days, drawdown {fmt_pct(latest['spx_drawdown_pct'])} from peak, "
        f"and VIX jumped {fmt_pct(latest['vix_pctchg'] * 100.0)} on the day."
    )
else:
    text.append(
        f"This was a relief event inside a {latest['context_label'].lower()} context. "
        f"Fear came out of the tape with VIX down {fmt_pct(latest['vix_pctchg'] * 100.0)}, while SPX had moved {fmt_pct(latest['spx_ret_5d'])} over the prior five days."
    )

if len(same_setup) > 0:
    wr = winrate(same_setup[f"spx_fwd_{forward_focus}d"])
    med = same_setup[f"spx_fwd_{forward_focus}d"].median()
    p10, p50, p90 = pct_bands(same_setup[f"spx_fwd_{forward_focus}d"])
    text.append(
        f"For this exact setup label, the {forward_focus}-day base rate is WR {fmt_pct(wr)}, median {fmt_pct(med)}, "
        f"with a 10th to 90th percentile range of {fmt_pct(p10)} to {fmt_pct(p90)} on {len(same_setup)} events."
    )

if len(analogs) > 0:
    analog_med_5 = analogs["spx_fwd_5d"].median()
    analog_med_10 = analogs["spx_fwd_10d"].median()
    text.append(
        f"The nearest analog set points to median SPX returns of {fmt_pct(analog_med_5)} over 5 days and {fmt_pct(analog_med_10)} over 10 days. "
        f"That gives you a more tape-aware read than a raw bucket average."
    )

with st.container(border=True):
    for paragraph in text:
        st.write(paragraph)

# ------------------------------- Main charts -------------------------------
st.subheader("Forward path and analog engine")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.subplots_adjust(wspace=0.25, hspace=0.35)

# Chart 1: path cone for same setup
ax1 = axes[0, 0]
horizons = [1, 2, 3, 5, 10, 20]
path_same_setup = build_path_stats(same_setup, horizons)
if not path_same_setup.empty:
    ax1.plot(path_same_setup["h"], path_same_setup["median"], linewidth=2, marker="o", label="Median")
    ax1.fill_between(path_same_setup["h"], path_same_setup["p25"], path_same_setup["p75"], alpha=0.25, label="25-75")
    ax1.fill_between(path_same_setup["h"], path_same_setup["p10"], path_same_setup["p90"], alpha=0.12, label="10-90")
ax1.axhline(0, color="#888888", linewidth=1)
ax1.set_title("Forward Path Cone | Same Setup", color=TEXT_COLOR, fontsize=12)
ax1.set_xlabel("Trading days")
ax1.set_ylabel("SPX forward return (%)")
ax1.grid(color=GRID_COLOR, linewidth=0.6)
ax1.legend(frameon=False, fontsize=9)

# Chart 2: path cone for analogs
ax2 = axes[0, 1]
path_analogs = build_path_stats(analogs, horizons)
if not path_analogs.empty:
    ax2.plot(path_analogs["h"], path_analogs["median"], linewidth=2, marker="o", label="Median")
    ax2.fill_between(path_analogs["h"], path_analogs["p25"], path_analogs["p75"], alpha=0.25, label="25-75")
    ax2.fill_between(path_analogs["h"], path_analogs["p10"], path_analogs["p90"], alpha=0.12, label="10-90")
ax2.axhline(0, color="#888888", linewidth=1)
ax2.set_title("Forward Path Cone | Nearest Analogs", color=TEXT_COLOR, fontsize=12)
ax2.set_xlabel("Trading days")
ax2.set_ylabel("SPX forward return (%)")
ax2.grid(color=GRID_COLOR, linewidth=0.6)
ax2.legend(frameon=False, fontsize=9)

# Chart 3: context expectancy
ax3 = axes[1, 0]
context_table = summarize_setup(events, "context_label", f"spx_fwd_{forward_focus}d")
if not context_table.empty:
    plot_df = context_table.sort_values("Med_%")
    ax3.barh(plot_df["context_label"], plot_df["Med_%"], edgecolor=BAR_EDGE)
ax3.axvline(0, color="#888888", linewidth=1)
ax3.set_title(f"Median SPX Return by Context | {forward_focus}D", color=TEXT_COLOR, fontsize=12)
ax3.set_xlabel("Median return (%)")
ax3.grid(axis="x", color=GRID_COLOR, linewidth=0.6)

# Chart 4: scatter of event severity vs forward path
ax4 = axes[1, 1]
scatter = events.copy()
colors = np.where(scatter["event_type"] == "Panic", PASTELS[1], PASTELS[0])
sizes = 20 + np.clip(scatter["move_mag"], 0, 80) * 2
ax4.scatter(
    scatter["spx_drawdown_pct"],
    scatter[f"spx_fwd_{forward_focus}d"],
    s=sizes,
    c=colors,
    alpha=0.75,
    edgecolors=BAR_EDGE,
    linewidths=0.3
)
ax4.axhline(0, color="#888888", linewidth=1)
ax4.axvline(latest["spx_drawdown_pct"], color="#444444", linewidth=1, linestyle="--")
ax4.set_title(f"Forward Return vs Drawdown at Event | {forward_focus}D", color=TEXT_COLOR, fontsize=12)
ax4.set_xlabel("SPX drawdown from peak (%)")
ax4.set_ylabel("SPX forward return (%)")
ax4.grid(color=GRID_COLOR, linewidth=0.6)

st.pyplot(fig, clear_figure=True)

# ------------------------------- Outcome diagnostics -----------------------
st.subheader("Failure rates and classification")

col3, col4 = st.columns([1, 1])

with col3:
    if not same_setup.empty:
        outcome_table = (
            same_setup["outcome_label"]
            .value_counts(normalize=True)
            .mul(100.0)
            .rename_axis("Outcome")
            .reset_index(name="Rate_%")
        )

        if not outcome_table.empty and {"Outcome", "Rate_%"} <= set(outcome_table.columns):
            fig2, ax = plt.subplots(figsize=(7, 4))
            ax.bar(outcome_table["Outcome"], outcome_table["Rate_%"], edgecolor=BAR_EDGE)
            ax.set_title("Outcome Mix | Same Setup", color=TEXT_COLOR, fontsize=12)
            ax.set_ylabel("Rate (%)")
            ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
            plt.xticks(rotation=20, ha="right")
            st.pyplot(fig2, clear_figure=True)
        else:
            st.write("Outcome table could not be constructed for current setup.")
    else:
        st.write("No outcome sample for current setup.")

with col4:
    type_table = summarize_setup(events, "event_type", f"spx_fwd_{forward_focus}d")
    if not type_table.empty:
        fig3, ax = plt.subplots(figsize=(7, 4))
        ax.bar(type_table["event_type"], type_table["Med_%"], edgecolor=BAR_EDGE)
        ax.axhline(0, color="#888888", linewidth=1)
        ax.set_title(f"Median Return | Panic vs Relief | {forward_focus}D", color=TEXT_COLOR, fontsize=12)
        ax.set_ylabel("Median return (%)")
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
        st.pyplot(fig3, clear_figure=True)
    else:
        st.write("No type sample for current filters.")

# ------------------------------- Tables ------------------------------------
st.subheader("Nearest analogs")

if len(analogs) > 0:
    analog_table = analogs.copy()
    analog_table["similarity_score"] = sim_scores.loc[analog_table.index]
    keep_cols = [
        "similarity_score",
        "event_type",
        "context_label",
        "vix_close",
        "move_mag",
        "spx_ret_1d",
        "spx_ret_5d",
        "spx_drawdown_pct",
        "rsi14",
        "rv20_pctile_252",
        "spx_fwd_1d",
        "spx_fwd_2d",
        "spx_fwd_3d",
        "spx_fwd_5d",
        "spx_fwd_10d",
        "spx_fwd_20d",
        "outcome_label",
    ]
    analog_table = analog_table[keep_cols].copy()
    analog_table.columns = [
        "Similarity",
        "Type",
        "Context",
        "VIX",
        "Move_%",
        "SPX_1D_%",
        "SPX_5D_%",
        "Drawdown_%",
        "RSI14",
        "RV20_Pctile_%",
        "Fwd_1D_%",
        "Fwd_2D_%",
        "Fwd_3D_%",
        "Fwd_5D_%",
        "Fwd_10D_%",
        "Fwd_20D_%",
        "Outcome",
    ]
    st.dataframe(analog_table.round(2).sort_values("Similarity"), use_container_width=True)
else:
    st.write("No analogs available.")

st.subheader("Setup expectancy tables")

col5, col6 = st.columns([1, 1])

with col5:
    context_expectancy = summarize_setup(events, "context_label", f"spx_fwd_{forward_focus}d")
    if not context_expectancy.empty:
        st.markdown("**By context**")
        st.dataframe(context_expectancy.round(2), use_container_width=True)

with col6:
    setup_expectancy = summarize_setup(events, "event_label", f"spx_fwd_{forward_focus}d")
    if not setup_expectancy.empty:
        st.markdown("**By event label**")
        st.dataframe(setup_expectancy.round(2), use_container_width=True)

with st.expander("Show full events table"):
    show_cols = [
        "event_type",
        "context_label",
        "event_label",
        "vix_close",
        "vix_pctchg",
        "move_mag",
        "spx_ret_1d",
        "spx_ret_3d",
        "spx_ret_5d",
        "spx_drawdown_pct",
        "dist_20dma",
        "dist_50dma",
        "dist_200dma",
        "rsi14",
        "rv20",
        "rv20_pctile_252",
        "qqq_iwm_5d",
        "hyg_ief_5d",
        "spx_fwd_1d",
        "spx_fwd_2d",
        "spx_fwd_3d",
        "spx_fwd_5d",
        "spx_fwd_10d",
        "spx_fwd_20d",
        "outcome_label",
    ]
    full_tbl = events[show_cols].copy()
    full_tbl["vix_pctchg"] = full_tbl["vix_pctchg"] * 100.0
    full_tbl = full_tbl.rename(columns={
        "event_type": "Type",
        "context_label": "Context",
        "event_label": "Event_Label",
        "vix_close": "VIX",
        "vix_pctchg": "VIX_Move_%",
        "move_mag": "Abs_Move_%",
        "spx_ret_1d": "SPX_1D_%",
        "spx_ret_3d": "SPX_3D_%",
        "spx_ret_5d": "SPX_5D_%",
        "spx_drawdown_pct": "Drawdown_%",
        "dist_20dma": "Dist_20DMA_%",
        "dist_50dma": "Dist_50DMA_%",
        "dist_200dma": "Dist_200DMA_%",
        "rsi14": "RSI14",
        "rv20": "RV20_%",
        "rv20_pctile_252": "RV20_Pctile_%",
        "qqq_iwm_5d": "QQQ_IWM_5D_%",
        "hyg_ief_5d": "HYG_IEF_5D_%",
        "spx_fwd_1d": "Fwd_1D_%",
        "spx_fwd_2d": "Fwd_2D_%",
        "spx_fwd_3d": "Fwd_3D_%",
        "spx_fwd_5d": "Fwd_5D_%",
        "spx_fwd_10d": "Fwd_10D_%",
        "spx_fwd_20d": "Fwd_20D_%",
        "outcome_label": "Outcome",
    })
    st.dataframe(full_tbl.round(2), use_container_width=True)

# ------------------------------- Footer ------------------------------------
st.caption("ADFM Analytics Platform | VIX Event Deep Dive | Data source: Yahoo Finance")
st.caption("© 2026 AD Fund Management LP")
