# vix_spike_deep_dive_clean_rewrite.py
# ADFM Analytics Platform | VIX Event Deep Dive
# Clean state model: current tape, live trigger, historical base rates, explicit analog anchor
# Data source: Yahoo Finance only

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =============================================================================
# Page config
# =============================================================================

st.set_page_config(
    page_title="VIX Spike Deep Dive",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Style constants
# =============================================================================

PASTEL_GREEN = "#52b788"
PASTEL_RED = "#e85d5d"
PASTEL_GREY = "#8b949e"

TEXT_COLOR = "#222222"
MUTED_TEXT = "#5f6b76"
GRID_COLOR = "#e6e6e6"
BORDER_COLOR = "#e8eaed"
BAR_EDGE = "#4f5963"

PANIC_COLOR = PASTEL_RED
RELIEF_COLOR = PASTEL_GREEN
NEUTRAL_COLOR = PASTEL_GREY

st.markdown(
    """
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.0rem;
}
div[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e8eaed;
    padding: 14px 16px;
    border-radius: 14px;
}
.adfm-small {
    color: #5f6b76;
    font-size: 0.92rem;
    line-height: 1.45;
}
.adfm-card-line {
    margin-bottom: 0.22rem;
}
.adfm-muted {
    color: #5f6b76;
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Formatting helpers
# =============================================================================

def fmt_pct(x, digits=1):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}%"


def fmt_num(x, digits=2):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def fmt_int(x):
    if pd.isna(x):
        return "NA"
    return f"{int(x):,}"


def fmt_date(idx):
    if idx is None or pd.isna(idx):
        return "NA"
    return pd.Timestamp(idx).strftime("%Y-%m-%d")


def rgba_from_hex(hex_color, alpha=0.25):
    h = hex_color.lstrip("#")
    r, g, b = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def safe_round_table(df, digits=2):
    if df is None or df.empty:
        return df
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(digits)
    return out


def render_card(title, lines, note=None):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        for line in lines:
            st.markdown(f"<div class='adfm-card-line'>{line}</div>", unsafe_allow_html=True)
        if note:
            st.markdown(f"<div class='adfm-small'>{note}</div>", unsafe_allow_html=True)


# =============================================================================
# Quant helpers
# =============================================================================

def normalize_yf_columns(df):
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]

    out.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in out.columns
    ]

    return out


def get_price_series(df, prefer_adjusted=False):
    if df is None or df.empty:
        return pd.Series(dtype=float)

    cols = df.columns

    if prefer_adjusted and "adj_close" in cols:
        return pd.to_numeric(df["adj_close"], errors="coerce")

    if "close" in cols:
        return pd.to_numeric(df["close"], errors="coerce")

    if "adj_close" in cols:
        return pd.to_numeric(df["adj_close"], errors="coerce")

    return pd.Series(index=df.index, dtype=float)


def compute_rsi(series, n=14):
    s = pd.to_numeric(series, errors="coerce")
    delta = s.diff()

    up = pd.Series(np.where(delta > 0, delta, 0.0), index=s.index, dtype=float)
    down = pd.Series(np.where(delta < 0, -delta, 0.0), index=s.index, dtype=float)

    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi = rsi.where(~((roll_down == 0) & (roll_up > 0)), 100.0)
    rsi = rsi.where(~((roll_up == 0) & (roll_down > 0)), 0.0)

    return rsi


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
    s = pd.to_numeric(price, errors="coerce")
    peak = s.cummax()
    return (s / peak - 1.0) * 100.0


def zscore(series, window):
    s = pd.to_numeric(series, errors="coerce")
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    return (s - mean) / std.replace(0, np.nan)


def percentile_rank_rolling(series, window):
    s = pd.to_numeric(series, errors="coerce")
    values = s.to_numpy(dtype=float)
    ranks = []

    for i in range(len(values)):
        current_value = values[i]

        if i < window - 1 or np.isnan(current_value):
            ranks.append(np.nan)
            continue

        chunk = values[i - window + 1:i + 1]
        chunk = chunk[~np.isnan(chunk)]

        if len(chunk) == 0:
            ranks.append(np.nan)
            continue

        ranks.append(100.0 * (chunk <= current_value).mean())

    return pd.Series(ranks, index=s.index)


def future_return(price, h):
    s = pd.to_numeric(price, errors="coerce")
    return (s.shift(-h) / s - 1.0) * 100.0


def safe_std(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return np.nan

    val = s.std()
    if pd.isna(val) or val == 0:
        return np.nan

    return val


def sample_quality_label(n):
    if pd.isna(n) or n == 0:
        return "No sample"
    if n < 5:
        return "Very thin"
    if n < 10:
        return "Thin"
    if n < 20:
        return "Usable"
    return "Deep enough"


# =============================================================================
# Data loading
# =============================================================================

@st.cache_data(show_spinner=False, ttl=60 * 60)
def download_single_ticker(ticker, start, prefer_adjusted=False):
    try:
        raw = yf.download(
            ticker,
            start=start,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.Series(dtype=float)

    if raw is None or raw.empty:
        return pd.Series(dtype=float)

    raw = normalize_yf_columns(raw)
    px = get_price_series(raw, prefer_adjusted=prefer_adjusted)
    px.name = ticker

    return px.dropna(how="all")


@st.cache_data(show_spinner=True, ttl=60 * 60)
def load_market_history(start="1998-01-01"):
    tickers = {
        "vix": {"ticker": "^VIX", "prefer_adjusted": False},
        "vix9d": {"ticker": "^VIX9D", "prefer_adjusted": False},
        "vix3m": {"ticker": "^VIX3M", "prefer_adjusted": False},
        "spx": {"ticker": "^GSPC", "prefer_adjusted": False},
        "qqq": {"ticker": "QQQ", "prefer_adjusted": True},
        "iwm": {"ticker": "IWM", "prefer_adjusted": True},
        "hyg": {"ticker": "HYG", "prefer_adjusted": True},
        "ief": {"ticker": "IEF", "prefer_adjusted": True},
        "tlt": {"ticker": "TLT", "prefer_adjusted": True},
    }

    series_map = {}

    for key, cfg in tickers.items():
        series_map[key] = download_single_ticker(
            ticker=cfg["ticker"],
            start=start,
            prefer_adjusted=cfg["prefer_adjusted"],
        )

    if series_map["vix"].empty or series_map["spx"].empty:
        return pd.DataFrame()

    indexes = [s.index for s in series_map.values() if isinstance(s, pd.Series) and not s.empty]
    idx = indexes[0]

    for item in indexes[1:]:
        idx = idx.union(item)

    idx = idx.sort_values()
    df = pd.DataFrame(index=idx)

    for key, s in series_map.items():
        col = f"{key}_close"
        if s.empty:
            df[col] = np.nan
        else:
            df[col] = s.reindex(idx).ffill()

    df = df.dropna(subset=["vix_close", "spx_close"]).copy()
    df = df[~df.index.duplicated(keep="last")].copy()
    df["trading_pos"] = np.arange(len(df))

    # VIX features
    df["vix_prev"] = df["vix_close"].shift(1)
    df["vix_pctchg"] = df["vix_close"] / df["vix_prev"] - 1.0
    df["vix_abs_pctchg"] = df["vix_pctchg"].abs()
    df["vix_change"] = df["vix_close"].diff()
    df["vix_z20"] = zscore(df["vix_change"], 20)
    df["vix_z60"] = zscore(df["vix_change"], 60)

    # SPX returns
    for h in [1, 2, 3, 5, 10, 20]:
        df[f"spx_ret_{h}d"] = df["spx_close"].pct_change(h) * 100.0

    df["spx_ret_5d_prior"] = df["spx_ret_5d"].shift(1)

    # Trend and distance from moving averages
    for ma in [20, 50, 100, 200]:
        df[f"spx_{ma}dma"] = df["spx_close"].rolling(ma).mean()
        df[f"dist_{ma}dma"] = (df["spx_close"] / df[f"spx_{ma}dma"] - 1.0) * 100.0

    df["ma20_slope_5d"] = (df["spx_20dma"] / df["spx_20dma"].shift(5) - 1.0) * 100.0
    df["ma50_slope_10d"] = (df["spx_50dma"] / df["spx_50dma"].shift(10) - 1.0) * 100.0
    df["ma200_slope_20d"] = (df["spx_200dma"] / df["spx_200dma"].shift(20) - 1.0) * 100.0

    df["rsi14"] = compute_rsi(df["spx_close"], 14)
    df["spx_drawdown_pct"] = max_drawdown_from_peak(df["spx_close"])

    # Realized volatility
    spx_daily_ret = df["spx_close"].pct_change()
    df["rv10"] = spx_daily_ret.rolling(10).std() * np.sqrt(252) * 100.0
    df["rv20"] = spx_daily_ret.rolling(20).std() * np.sqrt(252) * 100.0
    df["rv60"] = spx_daily_ret.rolling(60).std() * np.sqrt(252) * 100.0
    df["rv20_pctile_252"] = percentile_rank_rolling(df["rv20"], 252)

    # Cross-asset proxies
    df["qqq_iwm_ratio"] = df["qqq_close"] / df["iwm_close"]
    df["qqq_iwm_5d"] = (df["qqq_iwm_ratio"] / df["qqq_iwm_ratio"].shift(5) - 1.0) * 100.0
    df["qqq_iwm_20d"] = (df["qqq_iwm_ratio"] / df["qqq_iwm_ratio"].shift(20) - 1.0) * 100.0

    df["hyg_ief_ratio"] = df["hyg_close"] / df["ief_close"]
    df["hyg_ief_5d"] = (df["hyg_ief_ratio"] / df["hyg_ief_ratio"].shift(5) - 1.0) * 100.0
    df["hyg_ief_20d"] = (df["hyg_ief_ratio"] / df["hyg_ief_ratio"].shift(20) - 1.0) * 100.0

    df["ief_ret_1d"] = df["ief_close"].pct_change() * 100.0
    df["ief_ret_5d"] = df["ief_close"].pct_change(5) * 100.0
    df["tlt_ret_1d"] = df["tlt_close"].pct_change() * 100.0
    df["tlt_ret_5d"] = df["tlt_close"].pct_change(5) * 100.0

    # VIX curve proxies
    df["vix9d_vix_spread"] = df["vix9d_close"] - df["vix_close"]
    df["vix_vix3m_spread"] = df["vix_close"] - df["vix3m_close"]
    df["vix_vix3m_ratio"] = df["vix_close"] / df["vix3m_close"] - 1.0

    # Forward returns
    for h in [1, 2, 3, 5, 10, 20]:
        df[f"spx_fwd_{h}d"] = future_return(df["spx_close"], h)

    return df.dropna(how="all")


# =============================================================================
# Classification and event logic
# =============================================================================

def classify_context(
    row,
    oversold_rsi=30,
    drawdown_cut=-8.0,
    credit_stress_cut=-2.0,
    vol_pctile_cut=80.0,
):
    above_200 = pd.notna(row.get("dist_200dma")) and row.get("dist_200dma") >= 0
    above_50 = pd.notna(row.get("dist_50dma")) and row.get("dist_50dma") >= 0
    oversold = pd.notna(row.get("rsi14")) and row.get("rsi14") <= oversold_rsi
    deep_dd = pd.notna(row.get("spx_drawdown_pct")) and row.get("spx_drawdown_pct") <= drawdown_cut
    trend_down = pd.notna(row.get("ma20_slope_5d")) and row.get("ma20_slope_5d") < 0
    credit_stress = pd.notna(row.get("hyg_ief_5d")) and row.get("hyg_ief_5d") <= credit_stress_cut
    vol_hot = pd.notna(row.get("rv20_pctile_252")) and row.get("rv20_pctile_252") >= vol_pctile_cut

    if above_200 and above_50 and oversold and not credit_stress:
        return "Bull Pullback"

    if above_200 and not oversold and not credit_stress:
        return "Bull Shock"

    if deep_dd and oversold and (credit_stress or vol_hot):
        return "Washout"

    if (not above_200) and trend_down and (credit_stress or vol_hot):
        return "Trend Break"

    if (not above_200) and deep_dd:
        return "Late-Cycle Stress"

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
        if pd.notna(r10) and r10 > 0:
            return "Delayed 10D Recovery"
        return "Continuation"

    if r2 > 0 and r5 <= 0:
        return "Failed Bounce"

    if r2 <= 0 and r5 > 0:
        return "Delayed Recovery"

    return "Mixed"


def build_event_mask(
    df,
    event_mode,
    abs_move_threshold,
    z_threshold,
    spx_down_confirm,
    relief_prior_spx_5d,
):
    move_cut = abs_move_threshold / 100.0

    if event_mode == "Panic spike":
        mask = df["vix_pctchg"] >= move_cut

    elif event_mode == "Panic confirmation":
        mask = (df["vix_pctchg"] >= move_cut) & (df["spx_ret_1d"] <= spx_down_confirm)

    elif event_mode == "Relief crush":
        mask = df["vix_pctchg"] <= -move_cut

    elif event_mode == "Relief after weakness":
        mask = (df["vix_pctchg"] <= -move_cut) & (df["spx_ret_5d_prior"] <= relief_prior_spx_5d)

    elif event_mode == "Two-sided absolute move":
        mask = df["vix_abs_pctchg"] >= move_cut

    elif event_mode == "Two-sided z-score":
        mask = df["vix_z20"].abs() >= z_threshold

    else:
        mask = df["vix_pctchg"] >= move_cut

    return mask.fillna(False)


def make_event_label(row):
    return f"{row.get('event_type', 'NA')} | {row.get('context_label', 'NA')}"


def cluster_events_by_trading_gap(events, gap):
    if events.empty or gap <= 0:
        return events

    events = events.sort_index().copy()
    clusters = []
    current_cluster = []
    last_pos = None

    for idx, row in events.iterrows():
        pos = row.get("trading_pos")

        if last_pos is None:
            current_cluster = [idx]
            last_pos = pos
            continue

        if pd.notna(pos) and pd.notna(last_pos) and (pos - last_pos) <= gap:
            current_cluster.append(idx)
        else:
            clusters.append(current_cluster)
            current_cluster = [idx]

        last_pos = pos

    if current_cluster:
        clusters.append(current_cluster)

    keep_idx = []

    for cluster in clusters:
        cluster_df = events.loc[cluster].copy()
        keep_idx.append(cluster_df["move_mag"].idxmax())

    return events.loc[keep_idx].sort_index().copy()


def build_events(
    df,
    event_mode="Panic spike",
    abs_move_threshold=15.0,
    z_threshold=2.0,
    spx_down_confirm=-1.5,
    relief_prior_spx_5d=-3.0,
    trading_gap=5,
    oversold_rsi=30,
    drawdown_cut=-8.0,
    credit_stress_cut=-2.0,
    vol_pctile_cut=80.0,
):
    if df.empty:
        return pd.DataFrame()

    events = df.copy()
    mask = build_event_mask(
        events,
        event_mode=event_mode,
        abs_move_threshold=abs_move_threshold,
        z_threshold=z_threshold,
        spx_down_confirm=spx_down_confirm,
        relief_prior_spx_5d=relief_prior_spx_5d,
    )

    events = events[mask].copy()

    if events.empty:
        return events

    events["event_type"] = np.where(events["vix_pctchg"] >= 0, "Panic", "Relief")
    events["move_mag"] = events["vix_abs_pctchg"] * 100.0

    events["context_label"] = events.apply(
        lambda row: classify_context(
            row,
            oversold_rsi=oversold_rsi,
            drawdown_cut=drawdown_cut,
            credit_stress_cut=credit_stress_cut,
            vol_pctile_cut=vol_pctile_cut,
        ),
        axis=1,
    )

    events["outcome_label"] = events.apply(classify_outcome, axis=1)
    events["event_label"] = events.apply(make_event_label, axis=1)

    events = cluster_events_by_trading_gap(events, trading_gap)

    return events


def build_current_event_row(current, current_context):
    row = current.copy()
    row["event_type"] = "Panic" if row.get("vix_pctchg", np.nan) >= 0 else "Relief"
    row["move_mag"] = abs(row.get("vix_pctchg", np.nan)) * 100.0
    row["context_label"] = current_context
    row["outcome_label"] = "Incomplete"
    row["event_label"] = make_event_label(row)
    return row


def vix_curve_label(row):
    spread = row.get("vix_vix3m_spread", np.nan)

    if pd.isna(spread):
        return "Curve unavailable"

    if spread > 0:
        return "Front-end stress"

    if spread < -3:
        return "Contained term structure"

    return "Flat / transitional"


def trigger_readout(
    row,
    event_mode,
    abs_move_threshold,
    z_threshold,
    spx_down_confirm,
    relief_prior_spx_5d,
):
    vix_move = row.get("vix_pctchg", np.nan) * 100.0
    abs_move = abs(vix_move) if pd.notna(vix_move) else np.nan
    z20 = row.get("vix_z20", np.nan)
    spx_1d = row.get("spx_ret_1d", np.nan)
    prior_5d = row.get("spx_ret_5d_prior", np.nan)

    lines = []

    if event_mode == "Panic spike":
        distance = abs_move_threshold - vix_move if pd.notna(vix_move) else np.nan
        lines.append(f"VIX move today: **{fmt_pct(vix_move)}** vs trigger **+{fmt_pct(abs_move_threshold)}**")
        lines.append(f"Distance to trigger: **{fmt_pct(max(distance, 0) if pd.notna(distance) else np.nan)}**")

    elif event_mode == "Panic confirmation":
        distance = abs_move_threshold - vix_move if pd.notna(vix_move) else np.nan
        spx_ok = pd.notna(spx_1d) and spx_1d <= spx_down_confirm
        lines.append(f"VIX move today: **{fmt_pct(vix_move)}** vs trigger **+{fmt_pct(abs_move_threshold)}**")
        lines.append(f"SPX confirm: **{fmt_pct(spx_1d)}** vs required **≤ {fmt_pct(spx_down_confirm)}**")
        lines.append(f"SPX confirm met: **{'Yes' if spx_ok else 'No'}**")
        lines.append(f"VIX distance to trigger: **{fmt_pct(max(distance, 0) if pd.notna(distance) else np.nan)}**")

    elif event_mode == "Relief crush":
        distance = vix_move + abs_move_threshold if pd.notna(vix_move) else np.nan
        lines.append(f"VIX move today: **{fmt_pct(vix_move)}** vs trigger **-{fmt_pct(abs_move_threshold)}**")
        lines.append(f"Distance to trigger: **{fmt_pct(max(distance, 0) if pd.notna(distance) else np.nan)}**")

    elif event_mode == "Relief after weakness":
        distance = vix_move + abs_move_threshold if pd.notna(vix_move) else np.nan
        weakness_ok = pd.notna(prior_5d) and prior_5d <= relief_prior_spx_5d
        lines.append(f"VIX move today: **{fmt_pct(vix_move)}** vs trigger **-{fmt_pct(abs_move_threshold)}**")
        lines.append(f"Prior SPX 5D: **{fmt_pct(prior_5d)}** vs required **≤ {fmt_pct(relief_prior_spx_5d)}**")
        lines.append(f"Weakness filter met: **{'Yes' if weakness_ok else 'No'}**")
        lines.append(f"VIX distance to trigger: **{fmt_pct(max(distance, 0) if pd.notna(distance) else np.nan)}**")

    elif event_mode == "Two-sided absolute move":
        distance = abs_move_threshold - abs_move if pd.notna(abs_move) else np.nan
        lines.append(f"Absolute VIX move today: **{fmt_pct(abs_move)}** vs trigger **{fmt_pct(abs_move_threshold)}**")
        lines.append(f"Distance to trigger: **{fmt_pct(max(distance, 0) if pd.notna(distance) else np.nan)}**")

    elif event_mode == "Two-sided z-score":
        abs_z = abs(z20) if pd.notna(z20) else np.nan
        distance = z_threshold - abs_z if pd.notna(abs_z) else np.nan
        lines.append(f"VIX 20D z-score today: **{fmt_num(z20, 2)}** vs trigger **±{fmt_num(z_threshold, 2)}**")
        lines.append(f"Distance to trigger: **{fmt_num(max(distance, 0) if pd.notna(distance) else np.nan, 2)} z**")

    return lines


# =============================================================================
# Stats engine
# =============================================================================

def summarize_setup(events, label_col, ret_col):
    if events.empty or ret_col not in events.columns:
        return pd.DataFrame()

    rows = []

    for name, group in events.groupby(label_col):
        s = pd.to_numeric(group[ret_col], errors="coerce").dropna()

        if len(s) == 0:
            continue

        p10, p50, p90 = pct_bands(s)

        rows.append(
            {
                label_col: name,
                "N": len(s),
                "Quality": sample_quality_label(len(s)),
                "WR_%": winrate(s),
                "Avg_%": s.mean(),
                "Med_%": s.median(),
                "P10_%": p10,
                "P50_%": p50,
                "P90_%": p90,
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(["N", "WR_%"], ascending=[False, False])

    return out


def build_path_stats(events, horizons):
    rows = []

    for h in horizons:
        col = f"spx_fwd_{h}d"

        if events.empty or col not in events.columns:
            rows.append(
                {
                    "h": h,
                    "n": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "p25": np.nan,
                    "p75": np.nan,
                    "p10": np.nan,
                    "p90": np.nan,
                }
            )
            continue

        s = pd.to_numeric(events[col], errors="coerce").dropna()

        if len(s) == 0:
            rows.append(
                {
                    "h": h,
                    "n": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "p25": np.nan,
                    "p75": np.nan,
                    "p10": np.nan,
                    "p90": np.nan,
                }
            )
            continue

        rows.append(
            {
                "h": h,
                "n": len(s),
                "mean": s.mean(),
                "median": s.median(),
                "p25": np.percentile(s, 25),
                "p75": np.percentile(s, 75),
                "p10": np.percentile(s, 10),
                "p90": np.percentile(s, 90),
            }
        )

    return pd.DataFrame(rows)


def compute_weighted_similarity(events, target_row, feature_cols, weights):
    if events.empty:
        return pd.Series(dtype=float)

    available_cols = [
        c for c in feature_cols
        if c in events.columns and c in target_row.index
    ]

    if len(available_cols) == 0:
        return pd.Series(dtype=float)

    X = events[available_cols].copy()
    stds = X.apply(safe_std, axis=0)

    scores = pd.Series(index=events.index, dtype=float)

    for idx, row in X.iterrows():
        weighted_distance = 0.0
        total_weight = 0.0

        for col in available_cols:
            a = row.get(col)
            b = target_row.get(col)
            s = stds.get(col)
            w = float(weights.get(col, 1.0))

            if pd.isna(a) or pd.isna(b) or pd.isna(s) or s == 0:
                continue

            weighted_distance += w * abs(a - b) / s
            total_weight += w

        if total_weight <= 0:
            scores.loc[idx] = np.nan
        else:
            scores.loc[idx] = weighted_distance / total_weight

    return scores.sort_values()


def build_outcome_table(sample):
    if sample.empty or "outcome_label" not in sample.columns:
        return pd.DataFrame(columns=["Outcome", "Count", "Rate_%"])

    valid = sample[sample["outcome_label"] != "Incomplete"].copy()

    if valid.empty:
        return pd.DataFrame(columns=["Outcome", "Count", "Rate_%"])

    out = (
        valid["outcome_label"]
        .value_counts()
        .rename_axis("Outcome")
        .reset_index(name="Count")
    )

    out["Rate_%"] = out["Count"] / out["Count"].sum() * 100.0
    return out


def infer_live_verdict(event_type, wr, med, continuation_rate, n):
    if pd.isna(wr) or pd.isna(med) or n < 8:
        return "Thin sample, no standalone signal"

    if event_type == "Panic":
        if wr >= 60 and med > 0 and (pd.isna(continuation_rate) or continuation_rate < 45):
            return "Exhaustion bias"
        if med < 0 and pd.notna(continuation_rate) and continuation_rate >= 45:
            return "Continuation risk"
        return "Mixed panic tape"

    if event_type == "Relief":
        if wr >= 60 and med > 0:
            return "Follow-through bias"
        if med < 0 or (pd.notna(continuation_rate) and continuation_rate >= 40):
            return "Failed relief risk"
        return "Mixed relief tape"

    return "Mixed"


# =============================================================================
# Plot helpers
# =============================================================================

def base_layout(fig, height=420):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=42, r=30, t=55, b=42),
        font=dict(color=TEXT_COLOR, size=12),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)

    return fig


def make_path_cone_chart(path_stats, title):
    fig = go.Figure()

    if path_stats.empty or path_stats["median"].dropna().empty:
        fig.add_annotation(
            text="No forward path sample available",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=MUTED_TEXT),
        )
        fig.update_layout(title=title)
        return base_layout(fig)

    x = path_stats["h"]

    fig.add_trace(
        go.Scatter(
            x=x,
            y=path_stats["p90"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=path_stats["p10"],
            mode="lines",
            fill="tonexty",
            fillcolor=rgba_from_hex(NEUTRAL_COLOR, 0.16),
            line=dict(width=0),
            name="10-90",
            hovertemplate="Day %{x}<br>P10/P90 band<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=path_stats["p75"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=path_stats["p25"],
            mode="lines",
            fill="tonexty",
            fillcolor=rgba_from_hex(PASTEL_GREEN, 0.22),
            line=dict(width=0),
            name="25-75",
            hovertemplate="Day %{x}<br>P25/P75 band<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=path_stats["median"],
            mode="lines+markers",
            line=dict(color=TEXT_COLOR, width=2.5),
            marker=dict(size=8, color=TEXT_COLOR),
            name="Median",
            customdata=np.stack(
                [
                    path_stats["n"],
                    path_stats["mean"],
                    path_stats["p10"],
                    path_stats["p90"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "Day %{x}<br>"
                "Median: %{y:.2f}%<br>"
                "Mean: %{customdata[1]:.2f}%<br>"
                "N: %{customdata[0]:.0f}<br>"
                "P10: %{customdata[2]:.2f}%<br>"
                "P90: %{customdata[3]:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(y=0, line_width=1, line_color=NEUTRAL_COLOR)
    fig.update_layout(title=title, xaxis_title="Trading days forward", yaxis_title="SPX forward return (%)")

    return base_layout(fig)


def make_context_bar_chart(context_table, forward_focus):
    fig = go.Figure()

    if context_table.empty:
        fig.add_annotation(
            text="No context sample available",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=MUTED_TEXT),
        )
        fig.update_layout(title=f"Median SPX Return by Context | {forward_focus}D")
        return base_layout(fig)

    plot_df = context_table.sort_values("Med_%").copy()

    colors = [PASTEL_GREEN if v >= 0 else PASTEL_RED for v in plot_df["Med_%"]]

    fig.add_trace(
        go.Bar(
            x=plot_df["Med_%"],
            y=plot_df["context_label"],
            orientation="h",
            marker=dict(color=colors, line=dict(color=BAR_EDGE, width=0.5)),
            customdata=np.stack(
                [
                    plot_df["N"],
                    plot_df["WR_%"],
                    plot_df["P10_%"],
                    plot_df["P90_%"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "%{y}<br>"
                "Median: %{x:.2f}%<br>"
                "N: %{customdata[0]:.0f}<br>"
                "Win rate: %{customdata[1]:.1f}%<br>"
                "P10: %{customdata[2]:.2f}%<br>"
                "P90: %{customdata[3]:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vline(x=0, line_width=1, line_color=NEUTRAL_COLOR)
    fig.update_layout(
        title=f"Median SPX Return by Context | {forward_focus}D",
        xaxis_title="Median return (%)",
        yaxis_title=None,
    )

    return base_layout(fig)


def make_event_scatter(events, target_row, target_label, forward_focus):
    fig = go.Figure()

    ret_col = f"spx_fwd_{forward_focus}d"

    if events.empty or ret_col not in events.columns:
        fig.add_annotation(
            text="No event sample available",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=MUTED_TEXT),
        )
        fig.update_layout(title=f"Forward Return vs Drawdown | {forward_focus}D")
        return base_layout(fig)

    plot_df = events.dropna(subset=["spx_drawdown_pct", ret_col]).copy()

    if plot_df.empty:
        fig.add_annotation(
            text="No complete forward-return rows available",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=MUTED_TEXT),
        )
        fig.update_layout(title=f"Forward Return vs Drawdown | {forward_focus}D")
        return base_layout(fig)

    colors = [PANIC_COLOR if x == "Panic" else RELIEF_COLOR for x in plot_df["event_type"]]
    sizes = 9 + np.clip(plot_df["move_mag"].fillna(0), 0, 80) * 0.45

    hover_text = []

    for idx, row in plot_df.iterrows():
        hover_text.append(
            f"Date: {idx.strftime('%Y-%m-%d')}<br>"
            f"Type: {row.get('event_type', 'NA')}<br>"
            f"Context: {row.get('context_label', 'NA')}<br>"
            f"VIX: {fmt_num(row.get('vix_close'))}<br>"
            f"VIX move: {fmt_pct(row.get('vix_pctchg') * 100.0)}<br>"
            f"SPX drawdown: {fmt_pct(row.get('spx_drawdown_pct'))}<br>"
            f"Forward {forward_focus}D: {fmt_pct(row.get(ret_col))}<br>"
            f"Outcome: {row.get('outcome_label', 'NA')}"
        )

    fig.add_trace(
        go.Scatter(
            x=plot_df["spx_drawdown_pct"],
            y=plot_df[ret_col],
            mode="markers",
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.78,
                line=dict(color=BAR_EDGE, width=0.5),
            ),
            text=hover_text,
            hoverinfo="text",
            showlegend=False,
        )
    )

    if target_row is not None and "spx_drawdown_pct" in target_row.index:
        x_val = target_row.get("spx_drawdown_pct")
        if pd.notna(x_val):
            fig.add_vline(
                x=x_val,
                line_width=1,
                line_dash="dash",
                line_color=TEXT_COLOR,
                annotation_text=target_label,
                annotation_position="top left",
            )

    fig.add_hline(y=0, line_width=1, line_color=NEUTRAL_COLOR)
    fig.update_layout(
        title=f"Forward Return vs Drawdown | {forward_focus}D",
        xaxis_title="SPX drawdown from peak (%)",
        yaxis_title="SPX forward return (%)",
    )

    return base_layout(fig)


def make_outcome_mix_chart(outcome_table, title):
    fig = go.Figure()

    if outcome_table.empty:
        fig.add_annotation(
            text="No outcome sample available",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=MUTED_TEXT),
        )
        fig.update_layout(title=title)
        return base_layout(fig, height=380)

    colors = []

    for outcome in outcome_table["Outcome"]:
        if outcome in ["Exhaustion / Bounce", "Delayed Recovery", "Delayed 10D Recovery"]:
            colors.append(PASTEL_GREEN)
        elif outcome in ["Continuation", "Failed Bounce"]:
            colors.append(PASTEL_RED)
        else:
            colors.append(PASTEL_GREY)

    fig.add_trace(
        go.Bar(
            x=outcome_table["Outcome"],
            y=outcome_table["Rate_%"],
            marker=dict(color=colors, line=dict(color=BAR_EDGE, width=0.5)),
            customdata=outcome_table["Count"],
            hovertemplate=(
                "%{x}<br>"
                "Rate: %{y:.1f}%<br>"
                "Count: %{customdata:.0f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(title=title, xaxis_title=None, yaxis_title="Rate (%)")
    fig.update_xaxes(tickangle=20)

    return base_layout(fig, height=380)


def make_type_bar_chart(type_table, forward_focus):
    fig = go.Figure()

    if type_table.empty:
        fig.add_annotation(
            text="No type sample available",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=MUTED_TEXT),
        )
        fig.update_layout(title=f"Median Return | Panic vs Relief | {forward_focus}D")
        return base_layout(fig, height=380)

    colors = [PANIC_COLOR if x == "Panic" else RELIEF_COLOR for x in type_table["event_type"]]

    fig.add_trace(
        go.Bar(
            x=type_table["event_type"],
            y=type_table["Med_%"],
            marker=dict(color=colors, line=dict(color=BAR_EDGE, width=0.5)),
            customdata=np.stack(
                [
                    type_table["N"],
                    type_table["WR_%"],
                    type_table["P10_%"],
                    type_table["P90_%"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "%{x}<br>"
                "Median: %{y:.2f}%<br>"
                "N: %{customdata[0]:.0f}<br>"
                "Win rate: %{customdata[1]:.1f}%<br>"
                "P10: %{customdata[2]:.2f}%<br>"
                "P90: %{customdata[3]:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(y=0, line_width=1, line_color=NEUTRAL_COLOR)
    fig.update_layout(
        title=f"Median Return | Panic vs Relief | {forward_focus}D",
        xaxis_title=None,
        yaxis_title="Median return (%)",
    )

    return base_layout(fig, height=380)


# =============================================================================
# Sidebar
# =============================================================================

st.title("VIX Spike Deep Dive")
st.caption("ADFM Analytics Platform | Yahoo Finance data | Event-study and analog engine")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
This app separates the live tape from the historical event study.

It will only print a live signal verdict when the latest market row actually meets the selected VIX event definition. When there is no trigger, the app shows current tape, trigger distance, and historical base rates without pretending there is an active signal.

Data source: Yahoo Finance.
        """
    )

    st.divider()

    start_date = st.date_input("History start", value=datetime(1998, 1, 1))

    event_mode = st.selectbox(
        "Event definition",
        [
            "Panic spike",
            "Panic confirmation",
            "Relief crush",
            "Relief after weakness",
            "Two-sided absolute move",
            "Two-sided z-score",
        ],
        index=0,
    )

    abs_move_threshold = st.slider(
        "VIX move threshold (%)",
        min_value=5,
        max_value=80,
        value=15,
        step=5,
    )

    z_threshold = st.slider(
        "VIX z-score threshold",
        min_value=1.0,
        max_value=4.0,
        value=2.0,
        step=0.1,
    )

    spx_down_confirm = st.slider(
        "SPX down confirm for panic (%)",
        min_value=-5.0,
        max_value=0.0,
        value=-1.5,
        step=0.1,
    )

    relief_prior_spx_5d = st.slider(
        "Prior SPX 5D weakness for relief (%)",
        min_value=-10.0,
        max_value=0.0,
        value=-3.0,
        step=0.25,
    )

    trading_gap = st.slider(
        "Minimum gap between event waves",
        min_value=0,
        max_value=20,
        value=5,
        step=1,
    )

    forward_focus = st.selectbox(
        "Primary forward horizon",
        [1, 2, 3, 5, 10, 20],
        index=3,
    )

    analog_count = st.slider(
        "Nearest analogs to show",
        min_value=3,
        max_value=20,
        value=8,
        step=1,
    )

    st.divider()

    with st.expander("Context classification settings", expanded=False):
        oversold_rsi = st.slider(
            "Oversold RSI threshold",
            min_value=10,
            max_value=40,
            value=30,
            step=1,
        )

        drawdown_cut = st.slider(
            "Washout drawdown threshold (%)",
            min_value=-30.0,
            max_value=-3.0,
            value=-8.0,
            step=1.0,
        )

        credit_stress_cut = st.slider(
            "Credit stress: HYG/IEF 5D (%)",
            min_value=-8.0,
            max_value=1.0,
            value=-2.0,
            step=0.25,
        )

        vol_pctile_cut = st.slider(
            "Hot vol: RV20 percentile",
            min_value=50.0,
            max_value=99.0,
            value=80.0,
            step=1.0,
        )

    with st.expander("Filters", expanded=False):
        event_type_filter = st.multiselect(
            "Event types",
            ["Panic", "Relief"],
            default=["Panic", "Relief"],
        )

        context_filter = st.multiselect(
            "Contexts",
            [
                "Bull Pullback",
                "Bull Shock",
                "Washout",
                "Trend Break",
                "Late-Cycle Stress",
                "Mixed",
            ],
            default=[
                "Bull Pullback",
                "Bull Shock",
                "Washout",
                "Trend Break",
                "Late-Cycle Stress",
                "Mixed",
            ],
        )


# =============================================================================
# Data load and event state
# =============================================================================

df = load_market_history(start=str(start_date))

if df.empty:
    st.error("Failed to load VIX and SPX history from Yahoo Finance.")
    st.stop()

current_idx = df.index.max()
current = df.loc[current_idx].copy()

current_context = classify_context(
    current,
    oversold_rsi=oversold_rsi,
    drawdown_cut=drawdown_cut,
    credit_stress_cut=credit_stress_cut,
    vol_pctile_cut=vol_pctile_cut,
)

current_event_mask = build_event_mask(
    df.loc[[current_idx]].copy(),
    event_mode=event_mode,
    abs_move_threshold=abs_move_threshold,
    z_threshold=z_threshold,
    spx_down_confirm=spx_down_confirm,
    relief_prior_spx_5d=relief_prior_spx_5d,
)

live_trigger = bool(current_event_mask.iloc[0])
current_event = build_current_event_row(current, current_context)

raw_events = build_events(
    df=df,
    event_mode=event_mode,
    abs_move_threshold=abs_move_threshold,
    z_threshold=z_threshold,
    spx_down_confirm=spx_down_confirm,
    relief_prior_spx_5d=relief_prior_spx_5d,
    trading_gap=trading_gap,
    oversold_rsi=oversold_rsi,
    drawdown_cut=drawdown_cut,
    credit_stress_cut=credit_stress_cut,
    vol_pctile_cut=vol_pctile_cut,
)

if raw_events.empty:
    st.warning("No historical events matched the selected definition.")
    st.stop()

events = raw_events.copy()

if event_type_filter:
    events = events[events["event_type"].isin(event_type_filter)].copy()
else:
    events = events.iloc[0:0].copy()

if context_filter:
    events = events[events["context_label"].isin(context_filter)].copy()
else:
    events = events.iloc[0:0].copy()

if events.empty:
    st.warning("No events remained after the current filters.")
    st.stop()

primary_ret_col = f"spx_fwd_{forward_focus}d"
historical_events = events.dropna(subset=[primary_ret_col]).copy()

last_event_idx = events.index.max()
last_event = events.loc[last_event_idx].copy()

if live_trigger:
    target_row = current_event.copy()
    target_label = "Live trigger"
    target_event_type = current_event["event_type"]
    target_context = current_context
else:
    target_row = current.copy()
    target_row["move_mag"] = abs(target_row.get("vix_pctchg", np.nan)) * 100.0
    target_row["context_label"] = current_context
    target_row["event_type"] = "Current"
    target_label = "Current tape"
    target_event_type = None
    target_context = current_context

if live_trigger:
    setup_sample = historical_events[
        (historical_events["event_type"] == target_event_type) &
        (historical_events["context_label"] == target_context)
    ].copy()

    if setup_sample.empty:
        reference_sample = historical_events.copy()
        reference_label = "Selected event definition"
    else:
        reference_sample = setup_sample.copy()
        reference_label = f"Live setup: {target_event_type} | {target_context}"
else:
    setup_sample = pd.DataFrame()
    reference_sample = historical_events.copy()
    reference_label = "Selected event definition"

reference_n = len(reference_sample)
reference_wr = winrate(reference_sample[primary_ret_col]) if primary_ret_col in reference_sample.columns else np.nan
reference_median = reference_sample[primary_ret_col].median() if primary_ret_col in reference_sample.columns else np.nan
reference_p10, reference_p50, reference_p90 = pct_bands(reference_sample[primary_ret_col]) if primary_ret_col in reference_sample.columns else (np.nan, np.nan, np.nan)

outcome_sample = reference_sample.dropna(subset=["spx_fwd_2d", "spx_fwd_5d"]).copy()
continuation_rate = (
    100.0 * (outcome_sample["outcome_label"] == "Continuation").mean()
    if not outcome_sample.empty else np.nan
)

if live_trigger:
    live_verdict = infer_live_verdict(
        event_type=target_event_type,
        wr=reference_wr,
        med=reference_median,
        continuation_rate=continuation_rate,
        n=reference_n,
    )
else:
    live_verdict = "No live verdict"

feature_cols = [
    "vix_close",
    "move_mag",
    "spx_ret_1d",
    "spx_ret_5d",
    "dist_20dma",
    "dist_50dma",
    "dist_100dma",
    "dist_200dma",
    "ma20_slope_5d",
    "ma50_slope_10d",
    "spx_drawdown_pct",
    "rsi14",
    "rv20",
    "rv20_pctile_252",
    "qqq_iwm_5d",
    "hyg_ief_5d",
    "ief_ret_1d",
    "tlt_ret_5d",
    "vix_vix3m_spread",
]

feature_weights = {
    "vix_close": 1.25,
    "move_mag": 1.50,
    "spx_ret_1d": 1.00,
    "spx_ret_5d": 1.15,
    "dist_20dma": 1.00,
    "dist_50dma": 1.20,
    "dist_100dma": 1.10,
    "dist_200dma": 1.50,
    "ma20_slope_5d": 0.85,
    "ma50_slope_10d": 0.90,
    "spx_drawdown_pct": 1.70,
    "rsi14": 1.30,
    "rv20": 1.00,
    "rv20_pctile_252": 1.30,
    "qqq_iwm_5d": 0.90,
    "hyg_ief_5d": 1.45,
    "ief_ret_1d": 0.55,
    "tlt_ret_5d": 0.70,
    "vix_vix3m_spread": 1.20,
}

analogs_base = historical_events.copy()

if live_trigger and current_idx in analogs_base.index:
    analogs_base = analogs_base.drop(index=current_idx, errors="ignore")

sim_scores = compute_weighted_similarity(
    events=analogs_base,
    target_row=target_row,
    feature_cols=feature_cols,
    weights=feature_weights,
).dropna()

analogs_idx = sim_scores.head(analog_count).index.tolist()
analogs = analogs_base.loc[analogs_idx].copy() if analogs_idx else pd.DataFrame()


# =============================================================================
# Top section
# =============================================================================

top1, top2, top3 = st.columns([1.2, 1.15, 1.05])

with top1:
    render_card(
        "Current Tape",
        [
            f"<b>Date:</b> {fmt_date(current_idx)}",
            f"<b>Context:</b> {current_context}",
            f"<b>VIX:</b> {fmt_num(current.get('vix_close'))} | <b>Daily move:</b> {fmt_pct(current.get('vix_pctchg') * 100.0)}",
            f"<b>SPX 1D:</b> {fmt_pct(current.get('spx_ret_1d'))} | <b>SPX 5D:</b> {fmt_pct(current.get('spx_ret_5d'))}",
            f"<b>Drawdown:</b> {fmt_pct(current.get('spx_drawdown_pct'))} | <b>RSI14:</b> {fmt_num(current.get('rsi14'), 1)}",
            f"<b>Dist 20/50/200DMA:</b> {fmt_pct(current.get('dist_20dma'))} / {fmt_pct(current.get('dist_50dma'))} / {fmt_pct(current.get('dist_200dma'))}",
            f"<b>RV20:</b> {fmt_pct(current.get('rv20'))} | <b>RV20 pctile:</b> {fmt_pct(current.get('rv20_pctile_252'))}",
            f"<b>Credit proxy HYG/IEF 5D:</b> {fmt_pct(current.get('hyg_ief_5d'))}",
            f"<b>VIX curve:</b> {vix_curve_label(current)}",
        ],
    )

with top2:
    trigger_lines = [
        f"<b>Status:</b> {'Triggered' if live_trigger else 'No live trigger'}",
        f"<b>Event definition:</b> {event_mode}",
        f"<b>Primary horizon:</b> {forward_focus} trading days",
    ]
    trigger_lines.extend(trigger_readout(
        current,
        event_mode=event_mode,
        abs_move_threshold=abs_move_threshold,
        z_threshold=z_threshold,
        spx_down_confirm=spx_down_confirm,
        relief_prior_spx_5d=relief_prior_spx_5d,
    ))

    render_card(
        "Live Trigger Monitor",
        trigger_lines,
        note="This card is the only place where the app decides whether the latest market row qualifies as a live VIX event.",
    )

with top3:
    render_card(
        "Last Matched Event",
        [
            f"<b>Date:</b> {fmt_date(last_event_idx)}",
            f"<b>Type:</b> {last_event.get('event_type', 'NA')}",
            f"<b>Context:</b> {last_event.get('context_label', 'NA')}",
            f"<b>VIX:</b> {fmt_num(last_event.get('vix_close'))} | <b>Daily move:</b> {fmt_pct(last_event.get('vix_pctchg') * 100.0)}",
            f"<b>SPX 1D:</b> {fmt_pct(last_event.get('spx_ret_1d'))} | <b>SPX 5D:</b> {fmt_pct(last_event.get('spx_ret_5d'))}",
            f"<b>Drawdown:</b> {fmt_pct(last_event.get('spx_drawdown_pct'))} | <b>RSI14:</b> {fmt_num(last_event.get('rsi14'), 1)}",
            f"<b>Forward {forward_focus}D:</b> {fmt_pct(last_event.get(primary_ret_col))}",
            f"<b>Filtered events:</b> {fmt_int(len(events))}",
            f"<b>Complete forward rows:</b> {fmt_int(len(historical_events))}",
        ],
        note="Reference only. This card does not create a live signal when today has not triggered.",
    )


# =============================================================================
# Signal state
# =============================================================================

st.subheader("Signal State")

if live_trigger:
    st.success(
        f"Live VIX event triggered on {fmt_date(current_idx)}. Setup: {target_event_type} | {target_context}."
    )

    with st.container(border=True):
        st.markdown(f"**Verdict:** {live_verdict}")
        st.markdown(f"**Sample:** {reference_label}")
        st.markdown(f"**{forward_focus}D base rate:** N {fmt_int(reference_n)} | WR {fmt_pct(reference_wr)} | Median {fmt_pct(reference_median)} | P10/P90 {fmt_pct(reference_p10)} / {fmt_pct(reference_p90)}")
        st.markdown(f"**Continuation rate:** {fmt_pct(continuation_rate)}")
        st.markdown(f"**Sample quality:** {sample_quality_label(reference_n)}")

    if reference_n < 10:
        st.warning(
            f"The live setup sample is thin: N={reference_n}. Treat the verdict as directional evidence, not a standalone trade signal."
        )
else:
    st.info(
        "No live signal under the selected definition. The rest of the page shows current tape, trigger distance, and historical behavior for the selected event definition."
    )

    with st.container(border=True):
        st.markdown(f"**Historical reference sample:** {reference_label}")
        st.markdown(f"**{forward_focus}D base rate:** N {fmt_int(reference_n)} | WR {fmt_pct(reference_wr)} | Median {fmt_pct(reference_median)} | P10/P90 {fmt_pct(reference_p10)} / {fmt_pct(reference_p90)}")
        st.markdown(f"**Continuation rate:** {fmt_pct(continuation_rate)}")
        st.markdown("**Live verdict:** No live verdict because the latest market row did not trigger.")


# =============================================================================
# Charts
# =============================================================================

st.subheader("Historical Return Profile")

horizons = [1, 2, 3, 5, 10, 20]
path_reference = build_path_stats(reference_sample, horizons)
context_table = summarize_setup(historical_events, "context_label", primary_ret_col)
type_table = summarize_setup(historical_events, "event_type", primary_ret_col)
outcome_table = build_outcome_table(reference_sample)

chart1, chart2 = st.columns([1, 1])

with chart1:
    fig_path = make_path_cone_chart(
        path_reference,
        title=f"Forward Path Cone | {reference_label}",
    )
    st.plotly_chart(fig_path, use_container_width=True)

with chart2:
    fig_scatter = make_event_scatter(
        historical_events,
        target_row=target_row,
        target_label=target_label,
        forward_focus=forward_focus,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

chart3, chart4 = st.columns([1, 1])

with chart3:
    fig_context = make_context_bar_chart(context_table, forward_focus)
    st.plotly_chart(fig_context, use_container_width=True)

with chart4:
    fig_outcome = make_outcome_mix_chart(
        outcome_table,
        title=f"Outcome Mix | {reference_label}",
    )
    st.plotly_chart(fig_outcome, use_container_width=True)

if len(type_table) > 1:
    fig_type = make_type_bar_chart(type_table, forward_focus)
    st.plotly_chart(fig_type, use_container_width=True)


# =============================================================================
# Analogs
# =============================================================================

st.subheader("Nearest Analogs")

if live_trigger:
    st.caption("Analogs are anchored to the live VIX event.")
else:
    st.caption("Analogs are anchored to the current tape for reference only. There is no live signal unless the trigger monitor says triggered.")

if len(analogs) > 0:
    analog_table = analogs.copy()
    analog_table["similarity_score"] = sim_scores.loc[analog_table.index]

    keep_cols = [
        "similarity_score",
        "event_type",
        "context_label",
        "vix_close",
        "move_mag",
        "vix_z20",
        "spx_ret_1d",
        "spx_ret_5d",
        "spx_drawdown_pct",
        "dist_50dma",
        "dist_200dma",
        "rsi14",
        "rv20_pctile_252",
        "hyg_ief_5d",
        "qqq_iwm_5d",
        "vix_vix3m_spread",
        "spx_fwd_1d",
        "spx_fwd_2d",
        "spx_fwd_3d",
        "spx_fwd_5d",
        "spx_fwd_10d",
        "spx_fwd_20d",
        "outcome_label",
    ]

    keep_cols = [c for c in keep_cols if c in analog_table.columns]
    analog_table = analog_table[keep_cols].copy()

    rename_map = {
        "similarity_score": "Similarity",
        "event_type": "Type",
        "context_label": "Context",
        "vix_close": "VIX",
        "move_mag": "Move_%",
        "vix_z20": "VIX_Z20",
        "spx_ret_1d": "SPX_1D_%",
        "spx_ret_5d": "SPX_5D_%",
        "spx_drawdown_pct": "Drawdown_%",
        "dist_50dma": "Dist_50DMA_%",
        "dist_200dma": "Dist_200DMA_%",
        "rsi14": "RSI14",
        "rv20_pctile_252": "RV20_Pctile_%",
        "hyg_ief_5d": "HYG_IEF_5D_%",
        "qqq_iwm_5d": "QQQ_IWM_5D_%",
        "vix_vix3m_spread": "VIX_minus_VIX3M",
        "spx_fwd_1d": "Fwd_1D_%",
        "spx_fwd_2d": "Fwd_2D_%",
        "spx_fwd_3d": "Fwd_3D_%",
        "spx_fwd_5d": "Fwd_5D_%",
        "spx_fwd_10d": "Fwd_10D_%",
        "spx_fwd_20d": "Fwd_20D_%",
        "outcome_label": "Outcome",
    }

    analog_table = analog_table.rename(columns=rename_map)
    analog_table.index = analog_table.index.strftime("%Y-%m-%d")
    analog_table.index.name = "Date"

    st.dataframe(
        safe_round_table(analog_table.sort_values("Similarity"), 2),
        use_container_width=True,
    )

    analog_med_5 = analogs["spx_fwd_5d"].median() if "spx_fwd_5d" in analogs.columns else np.nan
    analog_med_10 = analogs["spx_fwd_10d"].median() if "spx_fwd_10d" in analogs.columns else np.nan
    analog_wr_5 = winrate(analogs["spx_fwd_5d"]) if "spx_fwd_5d" in analogs.columns else np.nan

    with st.container(border=True):
        st.markdown("**Analog Takeaway**")
        st.markdown(f"**5D median:** {fmt_pct(analog_med_5)}")
        st.markdown(f"**10D median:** {fmt_pct(analog_med_10)}")
        st.markdown(f"**5D win rate:** {fmt_pct(analog_wr_5)}")
else:
    st.write("No analogs available under the current filters.")


# =============================================================================
# Event tables
# =============================================================================

st.subheader("Historical Event Log")

base_table = historical_events.copy()

show_cols = [
    "event_type",
    "context_label",
    "vix_close",
    "vix_pctchg",
    "move_mag",
    "vix_z20",
    "spx_ret_1d",
    "spx_ret_5d",
    "spx_drawdown_pct",
    "dist_20dma",
    "dist_50dma",
    "dist_200dma",
    "rsi14",
    "rv20_pctile_252",
    "qqq_iwm_5d",
    "hyg_ief_5d",
    "vix_vix3m_spread",
    "spx_fwd_1d",
    "spx_fwd_2d",
    "spx_fwd_3d",
    "spx_fwd_5d",
    "spx_fwd_10d",
    "spx_fwd_20d",
    "outcome_label",
]

show_cols = [c for c in show_cols if c in base_table.columns]

event_table = base_table[show_cols].copy()

if "vix_pctchg" in event_table.columns:
    event_table["vix_pctchg"] = event_table["vix_pctchg"] * 100.0

rename_map = {
    "event_type": "Type",
    "context_label": "Context",
    "vix_close": "VIX",
    "vix_pctchg": "VIX_Move_%",
    "move_mag": "Abs_Move_%",
    "vix_z20": "VIX_Z20",
    "spx_ret_1d": "SPX_1D_%",
    "spx_ret_5d": "SPX_5D_%",
    "spx_drawdown_pct": "Drawdown_%",
    "dist_20dma": "Dist_20DMA_%",
    "dist_50dma": "Dist_50DMA_%",
    "dist_200dma": "Dist_200DMA_%",
    "rsi14": "RSI14",
    "rv20_pctile_252": "RV20_Pctile_%",
    "qqq_iwm_5d": "QQQ_IWM_5D_%",
    "hyg_ief_5d": "HYG_IEF_5D_%",
    "vix_vix3m_spread": "VIX_minus_VIX3M",
    "spx_fwd_1d": "Fwd_1D_%",
    "spx_fwd_2d": "Fwd_2D_%",
    "spx_fwd_3d": "Fwd_3D_%",
    "spx_fwd_5d": "Fwd_5D_%",
    "spx_fwd_10d": "Fwd_10D_%",
    "spx_fwd_20d": "Fwd_20D_%",
    "outcome_label": "Outcome",
}

event_table = event_table.rename(columns=rename_map)
event_table.index = event_table.index.strftime("%Y-%m-%d")
event_table.index.name = "Date"

st.dataframe(
    safe_round_table(event_table.sort_index(ascending=False), 2),
    use_container_width=True,
)


# =============================================================================
# Supporting tables and diagnostics
# =============================================================================

with st.expander("Supporting base-rate tables", expanded=False):
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("**By context**")
        st.dataframe(safe_round_table(context_table, 2), use_container_width=True)

    with col_b:
        st.markdown("**By event type**")
        st.dataframe(safe_round_table(type_table, 2), use_container_width=True)

with st.expander("Diagnostics", expanded=False):
    diag_rows = [
        {"Item": "Market rows loaded", "Value": fmt_int(len(df))},
        {"Item": "Raw events before filters", "Value": fmt_int(len(raw_events))},
        {"Item": "Events after filters", "Value": fmt_int(len(events))},
        {"Item": f"Events with {forward_focus}D forward returns", "Value": fmt_int(len(historical_events))},
        {"Item": "Reference sample", "Value": fmt_int(reference_n)},
        {"Item": "Nearest analogs returned", "Value": fmt_int(len(analogs))},
        {"Item": "Current data date", "Value": fmt_date(current_idx)},
        {"Item": "Live trigger", "Value": "Yes" if live_trigger else "No"},
        {"Item": "Last matched event date", "Value": fmt_date(last_event_idx)},
        {"Item": "Analog anchor", "Value": target_label},
    ]

    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)

st.caption("ADFM Analytics Platform | VIX Spike Deep Dive | Data source: Yahoo Finance")
st.caption("© 2026 AD Fund Management LP")
