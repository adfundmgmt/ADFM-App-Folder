# vix_spike_deep_dive_rebuilt.py
# ADFM Analytics Platform | VIX Event Deep Dive
# Event-study and analog engine using Yahoo Finance only

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go


# ------------------------------- Page config -------------------------------

st.set_page_config(
    page_title="VIX Spike Deep Dive",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------- Style -------------------------------------

PASTEL_GREEN = "#52b788"
PASTEL_RED = "#e85d5d"
PASTEL_GREY = "#8b949e"

TEXT_COLOR = "#222222"
MUTED_TEXT = "#5f6b76"
GRID_COLOR = "#e6e6e6"
BAR_EDGE = "#4f5963"
CARD_BG = "#ffffff"

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
.adfm-card {
    background: #ffffff;
    border: 1px solid #e8eaed;
    border-radius: 16px;
    padding: 16px 18px;
}
.adfm-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #222222;
    margin-bottom: 0.35rem;
}
.adfm-line {
    margin-bottom: 0.18rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------- Formatting helpers -------------------------

def fmt_pct(x, digits=1):
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}%"


def fmt_num(x, digits=2):
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


def fmt_int(x):
    if pd.isna(x):
        return "NA"
    return f"{int(x):,}"


def rgba_from_hex(hex_color, alpha=0.25):
    h = hex_color.lstrip("#")
    r, g, b = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def render_metric_box(title, lines, note=None):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        for line in lines:
            st.markdown(line)
        if note:
            st.markdown(f"<div class='adfm-small'>{note}</div>", unsafe_allow_html=True)


def safe_round_table(df, digits=2):
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(digits)
    return out


# ------------------------------- Quant helpers ------------------------------

def normalize_yf_columns(df):
    if df.empty:
        return df

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            c[0] if isinstance(c, tuple) else c
            for c in out.columns
        ]

    out.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in out.columns
    ]

    return out


def get_price_series(df, prefer_adjusted=False):
    if df.empty:
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
    delta = series.diff()

    up = pd.Series(
        np.where(delta > 0, delta, 0.0),
        index=series.index,
        dtype=float,
    )

    down = pd.Series(
        np.where(delta < 0, -delta, 0.0),
        index=series.index,
        dtype=float,
    )

    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()

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
    return (price / peak - 1.0) * 100.0


def zscore(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def percentile_rank_rolling(series, window):
    values = series.to_numpy(dtype=float)
    ranks = []

    for i in range(len(values)):
        if i < window - 1 or np.isnan(values[i]):
            ranks.append(np.nan)
            continue

        chunk = values[i - window + 1:i + 1]
        chunk = chunk[~np.isnan(chunk)]

        if len(chunk) == 0:
            ranks.append(np.nan)
            continue

        ranks.append(100.0 * (chunk <= values[i]).mean())

    return pd.Series(ranks, index=series.index)


def future_return(price, h):
    return (price.shift(-h) / price - 1.0) * 100.0


def safe_std(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return np.nan

    val = s.std()
    if pd.isna(val) or val == 0:
        return np.nan

    return val


def make_event_label(row):
    return f"{row['event_type']} | {row['context_label']}"


def sample_quality_label(n):
    if pd.isna(n) or n == 0:
        return "No historical evidence"
    if n < 5:
        return "Very thin sample"
    if n < 10:
        return "Thin sample"
    if n < 20:
        return "Usable sample"
    return "Deeper sample"


# ------------------------------- Data loading -------------------------------

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

    return px


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

    all_indexes = [
        s.index for s in series_map.values()
        if isinstance(s, pd.Series) and not s.empty
    ]

    idx = all_indexes[0]
    for item in all_indexes[1:]:
        idx = idx.union(item)

    idx = idx.sort_values()
    df = pd.DataFrame(index=idx)

    for key, s in series_map.items():
        if s.empty:
            df[f"{key}_close"] = np.nan
        else:
            df[f"{key}_close"] = s.reindex(idx).ffill()

    df = df.dropna(subset=["vix_close", "spx_close"]).copy()
    df = df[~df.index.duplicated(keep="last")].copy()
    df["trading_pos"] = np.arange(len(df))

    # VIX and SPX event features
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
    df["spx_ret_20d"] = df["spx_close"].pct_change(20) * 100.0

    # Trend and momentum
    df["spx_20dma"] = df["spx_close"].rolling(20).mean()
    df["spx_50dma"] = df["spx_close"].rolling(50).mean()
    df["spx_100dma"] = df["spx_close"].rolling(100).mean()
    df["spx_200dma"] = df["spx_close"].rolling(200).mean()

    df["dist_20dma"] = (df["spx_close"] / df["spx_20dma"] - 1.0) * 100.0
    df["dist_50dma"] = (df["spx_close"] / df["spx_50dma"] - 1.0) * 100.0
    df["dist_100dma"] = (df["spx_close"] / df["spx_100dma"] - 1.0) * 100.0
    df["dist_200dma"] = (df["spx_close"] / df["spx_200dma"] - 1.0) * 100.0

    df["ma20_slope_5d"] = (df["spx_20dma"] / df["spx_20dma"].shift(5) - 1.0) * 100.0
    df["ma50_slope_10d"] = (df["spx_50dma"] / df["spx_50dma"].shift(10) - 1.0) * 100.0
    df["ma200_slope_20d"] = (df["spx_200dma"] / df["spx_200dma"].shift(20) - 1.0) * 100.0

    df["rsi14"] = compute_rsi(df["spx_close"], 14)
    df["spx_drawdown_pct"] = max_drawdown_from_peak(df["spx_close"])

    # Realized volatility
    df["rv10"] = df["spx_close"].pct_change().rolling(10).std() * np.sqrt(252) * 100.0
    df["rv20"] = df["spx_close"].pct_change().rolling(20).std() * np.sqrt(252) * 100.0
    df["rv60"] = df["spx_close"].pct_change().rolling(60).std() * np.sqrt(252) * 100.0
    df["rv20_pctile_252"] = percentile_rank_rolling(df["rv20"], 252)

    # Cross-asset proxies
    df["qqq_iwm_ratio"] = df["qqq_close"] / df["iwm_close"]
    df["qqq_iwm_5d"] = (df["qqq_iwm_ratio"] / df["qqq_iwm_ratio"].shift(5) - 1.0) * 100.0
    df["qqq_iwm_20d"] = (df["qqq_iwm_ratio"] / df["qqq_iwm_ratio"].shift(20) - 1.0) * 100.0

    df["hyg_ief_ratio"] = df["hyg_close"] / df["ief_close"]
    df["hyg_ief_5d"] = (df["hyg_ief_ratio"] / df["hyg_ief_ratio"].shift(5) - 1.0) * 100.0
    df["hyg_ief_20d"] = (df["hyg_ief_ratio"] / df["hyg_ief_ratio"].shift(20) - 1.0) * 100.0

    df["tlt_ret_1d"] = df["tlt_close"].pct_change() * 100.0
    df["ief_ret_1d"] = df["ief_close"].pct_change() * 100.0
    df["tlt_ret_5d"] = df["tlt_close"].pct_change(5) * 100.0
    df["ief_ret_5d"] = df["ief_close"].pct_change(5) * 100.0

    # VIX curve proxies where Yahoo returns clean history
    df["vix9d_vix_spread"] = df["vix9d_close"] - df["vix_close"]
    df["vix_vix3m_spread"] = df["vix_close"] - df["vix3m_close"]
    df["vix_vix3m_ratio"] = df["vix_close"] / df["vix3m_close"] - 1.0

    # Forward SPX returns
    for h in [1, 2, 3, 5, 10, 20]:
        df[f"spx_fwd_{h}d"] = future_return(df["spx_close"], h)

    return df.dropna(how="all")


# ------------------------------- Classification -----------------------------

def classify_context(
    row,
    oversold_rsi=30,
    drawdown_cut=-8.0,
    credit_stress_cut=-2.0,
    vol_pctile_cut=80.0,
):
    above_200 = pd.notna(row.get("dist_200dma")) and row.get("dist_200dma") >= 0
    oversold = pd.notna(row.get("rsi14")) and row.get("rsi14") <= oversold_rsi
    deep_dd = pd.notna(row.get("spx_drawdown_pct")) and row.get("spx_drawdown_pct") <= drawdown_cut
    trend_down = pd.notna(row.get("ma20_slope_5d")) and row.get("ma20_slope_5d") < 0
    credit_stress = pd.notna(row.get("hyg_ief_5d")) and row.get("hyg_ief_5d") <= credit_stress_cut
    vol_hot = pd.notna(row.get("rv20_pctile_252")) and row.get("rv20_pctile_252") >= vol_pctile_cut

    if above_200 and oversold and not credit_stress:
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


def evaluate_event_condition(
    row,
    event_mode,
    abs_move_threshold,
    z_threshold,
    spx_down_confirm,
):
    if event_mode == "Absolute % move":
        return pd.notna(row.get("vix_abs_pctchg")) and row.get("vix_abs_pctchg") >= abs_move_threshold / 100.0

    if event_mode == "VIX z-score":
        return pd.notna(row.get("vix_z20")) and abs(row.get("vix_z20")) >= z_threshold

    if event_mode == "Panic day":
        return (
            pd.notna(row.get("vix_pctchg")) and
            pd.notna(row.get("spx_ret_1d")) and
            row.get("vix_pctchg") >= abs_move_threshold / 100.0 and
            row.get("spx_ret_1d") <= spx_down_confirm
        )

    if event_mode == "Relief day":
        return (
            pd.notna(row.get("vix_pctchg")) and
            pd.notna(row.get("spx_ret_5d")) and
            row.get("vix_pctchg") <= -abs_move_threshold / 100.0 and
            row.get("spx_ret_5d") <= -3.0
        )

    return False


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
        if "move_mag" in cluster_df.columns:
            keep_idx.append(cluster_df["move_mag"].idxmax())
        else:
            keep_idx.append(cluster_df.index[-1])

    return events.loc[keep_idx].sort_index().copy()


def build_events(
    df,
    event_mode="Absolute % move",
    abs_move_threshold=25.0,
    z_threshold=2.0,
    spx_down_confirm=-1.5,
    trading_gap=5,
    oversold_rsi=30,
    drawdown_cut=-8.0,
    credit_stress_cut=-2.0,
    vol_pctile_cut=80.0,
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


# ------------------------------- Stats engine -------------------------------

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
        else:
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

    available_cols = [c for c in feature_cols if c in events.columns and c in target_row.index]

    if len(available_cols) == 0:
        return pd.Series(dtype=float)

    X = events[available_cols].copy()
    stds = X.apply(safe_std, axis=0)

    scores = pd.Series(index=events.index, dtype=float)
    used_features = pd.Series(index=events.index, dtype=float)

    for idx, row in X.iterrows():
        weighted_diffs = []
        total_weight = 0.0

        for col in available_cols:
            a = row.get(col)
            b = target_row.get(col)
            s = stds.get(col)
            w = float(weights.get(col, 1.0))

            if pd.isna(a) or pd.isna(b) or pd.isna(s) or s == 0:
                continue

            weighted_diffs.append(w * abs(a - b) / s)
            total_weight += w

        if total_weight <= 0 or len(weighted_diffs) == 0:
            scores.loc[idx] = np.nan
            used_features.loc[idx] = 0
        else:
            scores.loc[idx] = np.sum(weighted_diffs) / total_weight
            used_features.loc[idx] = len(weighted_diffs)

    return scores.sort_values()


def get_historical_sample(events, forward_focus):
    col = f"spx_fwd_{forward_focus}d"

    if events.empty or col not in events.columns:
        return pd.DataFrame()

    return events.dropna(subset=[col]).copy()


def infer_verdict(event_type, wr, med, continuation_rate, n):
    if pd.isna(wr) or pd.isna(med) or n < 5:
        return "Insufficient sample"

    if event_type == "Panic":
        if wr >= 60 and med > 0:
            return "Exhaustion bias"
        if pd.notna(continuation_rate) and continuation_rate >= 50:
            return "Continuation risk"
        return "Transition / mixed"

    if event_type == "Relief":
        if wr >= 60 and med > 0:
            return "Follow-through bias"
        if pd.notna(continuation_rate) and continuation_rate >= 40:
            return "Complacency risk"
        return "Transition / mixed"

    return "Transition / mixed"


def vix_curve_label(row):
    spread = row.get("vix_vix3m_spread", np.nan)

    if pd.isna(spread):
        return "Curve unavailable"

    if spread > 0:
        return "Front-end stress"

    if spread < -3:
        return "Contained term structure"

    return "Flat / transitional"


# ------------------------------- Plot helpers -------------------------------

def base_layout(fig, height=420):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=40, r=30, t=55, b=40),
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
    fig.update_layout(title=title)

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

    colors = [
        PASTEL_GREEN if v >= 0 else PASTEL_RED
        for v in plot_df["Med_%"]
    ]

    fig.add_trace(
        go.Bar(
            x=plot_df["Med_%"],
            y=plot_df["context_label"],
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color=BAR_EDGE, width=0.5),
            ),
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


def make_event_scatter(events, target_event, forward_focus):
    fig = go.Figure()

    if events.empty:
        fig.add_annotation(
            text="No event sample available",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=MUTED_TEXT),
        )
        fig.update_layout(title=f"Forward Return vs Drawdown at Event | {forward_focus}D")
        return base_layout(fig)

    plot_df = events.copy()
    ret_col = f"spx_fwd_{forward_focus}d"
    plot_df = plot_df.dropna(subset=["spx_drawdown_pct", ret_col])

    colors = [
        PANIC_COLOR if x == "Panic" else RELIEF_COLOR
        for x in plot_df["event_type"]
    ]

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

    if target_event is not None and "spx_drawdown_pct" in target_event.index:
        if pd.notna(target_event.get("spx_drawdown_pct")):
            fig.add_vline(
                x=target_event.get("spx_drawdown_pct"),
                line_width=1,
                line_dash="dash",
                line_color=TEXT_COLOR,
            )

    fig.add_hline(y=0, line_width=1, line_color=NEUTRAL_COLOR)
    fig.update_layout(
        title=f"Forward Return vs Drawdown at Event | {forward_focus}D",
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
            marker=dict(
                color=colors,
                line=dict(color=BAR_EDGE, width=0.5),
            ),
            customdata=outcome_table["Count"],
            hovertemplate=(
                "%{x}<br>"
                "Rate: %{y:.1f}%<br>"
                "Count: %{customdata:.0f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title="Rate (%)",
    )

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

    colors = [
        PANIC_COLOR if x == "Panic" else RELIEF_COLOR
        for x in type_table["event_type"]
    ]

    fig.add_trace(
        go.Bar(
            x=type_table["event_type"],
            y=type_table["Med_%"],
            marker=dict(
                color=colors,
                line=dict(color=BAR_EDGE, width=0.5),
            ),
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


# ------------------------------- Sidebar -----------------------------------

st.title("VIX Spike Deep Dive")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
**Purpose:** classify large VIX events, score historical analogs, and map forward SPX paths.

**Core improvements in this version**
- Separates current tape from the last matched VIX event.
- Uses adjusted ETF prices for QQQ, IWM, HYG, IEF, and TLT.
- Excludes incomplete forward-return rows from expectancy stats.
- Uses weighted analog scoring instead of equal-weight feature matching.
- Clusters volatility waves by trading-day gaps and keeps the largest event in each wave.

**Data source:** Yahoo Finance.
        """
    )

    st.divider()

    start_date = st.date_input("History start", value=datetime(1998, 1, 1))

    event_mode = st.selectbox(
        "Event definition",
        ["Absolute % move", "VIX z-score", "Panic day", "Relief day"],
        index=0,
    )

    abs_move_threshold = st.slider(
        "Absolute VIX move threshold (%)",
        min_value=10,
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
        "SPX down confirm for panic day (%)",
        min_value=-5.0,
        max_value=0.0,
        value=-1.5,
        step=0.1,
    )

    trading_gap = st.slider(
        "Minimum gap between event waves (trading days)",
        min_value=0,
        max_value=20,
        value=5,
        step=1,
    )

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
        "Credit stress threshold: HYG/IEF 5D (%)",
        min_value=-8.0,
        max_value=1.0,
        value=-2.0,
        step=0.25,
    )

    vol_pctile_cut = st.slider(
        "Hot vol threshold: RV20 percentile",
        min_value=50.0,
        max_value=99.0,
        value=80.0,
        step=1.0,
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


# ------------------------------- Data load ---------------------------------

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

current_event_hit = evaluate_event_condition(
    row=current,
    event_mode=event_mode,
    abs_move_threshold=abs_move_threshold,
    z_threshold=z_threshold,
    spx_down_confirm=spx_down_confirm,
)

events_raw = build_events(
    df=df,
    event_mode=event_mode,
    abs_move_threshold=abs_move_threshold,
    z_threshold=z_threshold,
    spx_down_confirm=spx_down_confirm,
    trading_gap=trading_gap,
    oversold_rsi=oversold_rsi,
    drawdown_cut=drawdown_cut,
    credit_stress_cut=credit_stress_cut,
    vol_pctile_cut=vol_pctile_cut,
)

if events_raw.empty:
    st.warning("No events matched the current definition.")
    st.stop()

events = events_raw.copy()

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

last_event_idx = events.index.max()
last_event = events.loc[last_event_idx].copy()

historical_events = get_historical_sample(events, forward_focus)
historical_ex_target = historical_events.drop(index=last_event_idx, errors="ignore").copy()

same_setup = historical_ex_target[
    (historical_ex_target["event_type"] == last_event["event_type"]) &
    (historical_ex_target["context_label"] == last_event["context_label"])
].copy()

primary_ret_col = f"spx_fwd_{forward_focus}d"

same_setup_n = len(same_setup)
same_setup_wr = winrate(same_setup[primary_ret_col]) if primary_ret_col in same_setup else np.nan
same_setup_median = same_setup[primary_ret_col].median() if primary_ret_col in same_setup else np.nan

outcome_sample = same_setup.dropna(subset=["spx_fwd_2d", "spx_fwd_5d"]).copy()

same_setup_continuation_rate = (
    100.0 * (outcome_sample["outcome_label"] == "Continuation").mean()
    if not outcome_sample.empty else np.nan
)

verdict = infer_verdict(
    event_type=last_event["event_type"],
    wr=same_setup_wr,
    med=same_setup_median,
    continuation_rate=same_setup_continuation_rate,
    n=same_setup_n,
)

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
    "move_mag": 1.60,
    "spx_ret_1d": 1.00,
    "spx_ret_5d": 1.15,
    "dist_20dma": 1.00,
    "dist_50dma": 1.20,
    "dist_100dma": 1.15,
    "dist_200dma": 1.60,
    "ma20_slope_5d": 0.85,
    "ma50_slope_10d": 0.90,
    "spx_drawdown_pct": 1.75,
    "rsi14": 1.30,
    "rv20": 1.00,
    "rv20_pctile_252": 1.35,
    "qqq_iwm_5d": 0.90,
    "hyg_ief_5d": 1.50,
    "ief_ret_1d": 0.55,
    "tlt_ret_5d": 0.70,
    "vix_vix3m_spread": 1.20,
}

sim_scores = compute_weighted_similarity(
    events=historical_ex_target,
    target_row=last_event,
    feature_cols=feature_cols,
    weights=feature_weights,
)

sim_scores = sim_scores.dropna()
analogs_idx = sim_scores.head(analog_count).index.tolist()
analogs = historical_ex_target.loc[analogs_idx].copy() if analogs_idx else pd.DataFrame()


# ------------------------------- Top section --------------------------------

st.caption("ADFM Analytics Platform | VIX Event Deep Dive | Yahoo Finance data")

top1, top2, top3 = st.columns([1.25, 1.25, 1.0])

with top1:
    render_metric_box(
        "Current Tape",
        [
            f"**Date**: {current_idx.strftime('%Y-%m-%d')}",
            f"**Current context**: {current_context}",
            f"**Event trigger today**: {'Yes' if current_event_hit else 'No'}",
            f"**VIX**: {fmt_num(current.get('vix_close'))} | **Daily move**: {fmt_pct(current.get('vix_pctchg') * 100.0)}",
            f"**SPX 1D**: {fmt_pct(current.get('spx_ret_1d'))} | **SPX 5D**: {fmt_pct(current.get('spx_ret_5d'))}",
            f"**Drawdown from peak**: {fmt_pct(current.get('spx_drawdown_pct'))} | **RSI14**: {fmt_num(current.get('rsi14'), 1)}",
            f"**Dist 20DMA**: {fmt_pct(current.get('dist_20dma'))} | **Dist 50DMA**: {fmt_pct(current.get('dist_50dma'))} | **Dist 200DMA**: {fmt_pct(current.get('dist_200dma'))}",
            f"**RV20**: {fmt_pct(current.get('rv20'))} | **RV20 percentile**: {fmt_pct(current.get('rv20_pctile_252'))}",
            f"**VIX curve**: {vix_curve_label(current)}",
        ],
        note="This block always reflects the latest market data available from Yahoo Finance, regardless of whether today qualifies as a VIX event.",
    )

with top2:
    render_metric_box(
        "Last Matched Event",
        [
            f"**Date**: {last_event_idx.strftime('%Y-%m-%d')}",
            f"**Type**: {last_event['event_type']}",
            f"**Context**: {last_event['context_label']}",
            f"**Verdict**: {verdict}",
            f"**VIX**: {fmt_num(last_event.get('vix_close'))} | **Daily move**: {fmt_pct(last_event.get('vix_pctchg') * 100.0)}",
            f"**SPX 1D**: {fmt_pct(last_event.get('spx_ret_1d'))} | **SPX 5D**: {fmt_pct(last_event.get('spx_ret_5d'))}",
            f"**Drawdown from peak**: {fmt_pct(last_event.get('spx_drawdown_pct'))} | **RSI14**: {fmt_num(last_event.get('rsi14'), 1)}",
            f"**Dist 20DMA**: {fmt_pct(last_event.get('dist_20dma'))} | **Dist 50DMA**: {fmt_pct(last_event.get('dist_50dma'))} | **Dist 200DMA**: {fmt_pct(last_event.get('dist_200dma'))}",
            f"**Setup sample**: N {fmt_int(same_setup_n)} | WR {fmt_pct(same_setup_wr)} | Median {fmt_pct(same_setup_median)}",
            f"**Continuation risk**: {fmt_pct(same_setup_continuation_rate)}",
        ],
        note="This block reflects the most recent historical event that passed the sidebar definition and filters.",
    )

with top3:
    render_metric_box(
        "Current Definition",
        [
            f"**Event mode**: {event_mode}",
            f"**Abs threshold**: {abs_move_threshold}%",
            f"**Z threshold**: {z_threshold}",
            f"**SPX confirm**: {spx_down_confirm}%",
            f"**Event wave gap**: {trading_gap} trading days",
            f"**Primary horizon**: {forward_focus} days",
            f"**Filtered events**: {fmt_int(len(events))}",
            f"**Historical events with {forward_focus}D forward return**: {fmt_int(len(historical_events))}",
            f"**Latest setup label**: {last_event['event_label']}",
        ],
    )


if same_setup_n < 10:
    st.warning(
        f"The exact setup sample is thin: N={same_setup_n}. Treat the win rate and median as directional evidence, not a standalone signal."
    )


# ------------------------------- Dynamic readout ----------------------------

st.subheader("What just happened and what usually comes next")

paragraphs = []

if current_event_hit:
    paragraphs.append(
        f"The current tape qualifies under the selected event definition. VIX moved {fmt_pct(current.get('vix_pctchg') * 100.0)} on the latest available session, SPX was {fmt_pct(current.get('spx_ret_1d'))} on the day, and the market sits {fmt_pct(current.get('spx_drawdown_pct'))} below its prior peak. The live context reads as {current_context.lower()}, which is the regime label to anchor before leaning on any historical average."
    )
else:
    paragraphs.append(
        f"The current tape does not qualify under the selected VIX event definition. That matters because the last matched event was {last_event_idx.strftime('%Y-%m-%d')}, so the event-study section is describing the most recent qualifying volatility episode rather than claiming that today is a fresh signal."
    )

if last_event["event_type"] == "Panic":
    paragraphs.append(
        f"The last matched event was a panic event inside a {last_event['context_label'].lower()} context. SPX had moved {fmt_pct(last_event.get('spx_ret_5d'))} over the prior five sessions, the index was {fmt_pct(last_event.get('spx_drawdown_pct'))} below its peak, and VIX jumped {fmt_pct(last_event.get('vix_pctchg') * 100.0)} on the day."
    )
else:
    paragraphs.append(
        f"The last matched event was a relief event inside a {last_event['context_label'].lower()} context. VIX fell {fmt_pct(last_event.get('vix_pctchg') * 100.0)} on the day, while SPX had moved {fmt_pct(last_event.get('spx_ret_5d'))} over the prior five sessions."
    )

if same_setup_n > 0:
    p10, p50, p90 = pct_bands(same_setup[primary_ret_col])
    paragraphs.append(
        f"For the same event type and context, the {forward_focus}-day historical base rate is WR {fmt_pct(same_setup_wr)}, median {fmt_pct(same_setup_median)}, with a 10th to 90th percentile range of {fmt_pct(p10)} to {fmt_pct(p90)} across {same_setup_n} prior events. Sample quality: {sample_quality_label(same_setup_n).lower()}."
    )

if len(analogs) > 0:
    analog_med_5 = analogs["spx_fwd_5d"].median()
    analog_med_10 = analogs["spx_fwd_10d"].median()
    analog_wr_5 = winrate(analogs["spx_fwd_5d"])
    paragraphs.append(
        f"The weighted analog set points to median SPX returns of {fmt_pct(analog_med_5)} over 5 trading days and {fmt_pct(analog_med_10)} over 10 trading days, with a 5-day win rate of {fmt_pct(analog_wr_5)}. This is usually more useful than the raw bucket average because it gives more weight to drawdown, trend, VIX severity, realized volatility, and credit stress."
    )

with st.container(border=True):
    for p in paragraphs:
        st.write(p)


# ------------------------------- Main charts --------------------------------

st.subheader("Forward path and analog engine")

horizons = [1, 2, 3, 5, 10, 20]

path_same_setup = build_path_stats(same_setup, horizons)
path_analogs = build_path_stats(analogs, horizons)

chart1, chart2 = st.columns([1, 1])

with chart1:
    fig_same = make_path_cone_chart(
        path_same_setup,
        title="Forward Path Cone | Same Setup",
    )
    st.plotly_chart(fig_same, use_container_width=True)

with chart2:
    fig_analog = make_path_cone_chart(
        path_analogs,
        title="Forward Path Cone | Nearest Analogs",
    )
    st.plotly_chart(fig_analog, use_container_width=True)

chart3, chart4 = st.columns([1, 1])

with chart3:
    context_table = summarize_setup(
        historical_events,
        "context_label",
        primary_ret_col,
    )

    fig_context = make_context_bar_chart(
        context_table,
        forward_focus=forward_focus,
    )
    st.plotly_chart(fig_context, use_container_width=True)

with chart4:
    fig_scatter = make_event_scatter(
        historical_events,
        target_event=last_event,
        forward_focus=forward_focus,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ------------------------------- Outcome diagnostics ------------------------

st.subheader("Failure rates and classification")

out1, out2 = st.columns([1, 1])

with out1:
    if not outcome_sample.empty:
        outcome_table = (
            outcome_sample["outcome_label"]
            .value_counts()
            .rename_axis("Outcome")
            .reset_index(name="Count")
        )

        outcome_table["Rate_%"] = outcome_table["Count"] / outcome_table["Count"].sum() * 100.0
    else:
        outcome_table = pd.DataFrame(columns=["Outcome", "Count", "Rate_%"])

    fig_outcome = make_outcome_mix_chart(
        outcome_table,
        title="Outcome Mix | Same Setup",
    )
    st.plotly_chart(fig_outcome, use_container_width=True)

with out2:
    type_table = summarize_setup(
        historical_events,
        "event_type",
        primary_ret_col,
    )

    fig_type = make_type_bar_chart(
        type_table,
        forward_focus=forward_focus,
    )
    st.plotly_chart(fig_type, use_container_width=True)


# ------------------------------- Tables -------------------------------------

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
else:
    st.write("No analogs available under the current filters.")


st.subheader("Setup expectancy tables")

table1, table2 = st.columns([1, 1])

with table1:
    context_expectancy = summarize_setup(
        historical_events,
        "context_label",
        primary_ret_col,
    )

    if not context_expectancy.empty:
        st.markdown("**By context**")
        st.dataframe(
            safe_round_table(context_expectancy, 2),
            use_container_width=True,
        )
    else:
        st.write("No context expectancy table available.")

with table2:
    setup_expectancy = summarize_setup(
        historical_events,
        "event_label",
        primary_ret_col,
    )

    if not setup_expectancy.empty:
        st.markdown("**By event label**")
        st.dataframe(
            safe_round_table(setup_expectancy, 2),
            use_container_width=True,
        )
    else:
        st.write("No setup expectancy table available.")


with st.expander("Show full events table"):
    show_cols = [
        "event_type",
        "context_label",
        "event_label",
        "vix_close",
        "vix_pctchg",
        "move_mag",
        "vix_z20",
        "spx_ret_1d",
        "spx_ret_3d",
        "spx_ret_5d",
        "spx_ret_10d",
        "spx_drawdown_pct",
        "dist_20dma",
        "dist_50dma",
        "dist_100dma",
        "dist_200dma",
        "ma20_slope_5d",
        "ma50_slope_10d",
        "rsi14",
        "rv20",
        "rv20_pctile_252",
        "qqq_iwm_5d",
        "hyg_ief_5d",
        "ief_ret_1d",
        "tlt_ret_5d",
        "vix_vix3m_spread",
        "spx_fwd_1d",
        "spx_fwd_2d",
        "spx_fwd_3d",
        "spx_fwd_5d",
        "spx_fwd_10d",
        "spx_fwd_20d",
        "outcome_label",
    ]

    show_cols = [c for c in show_cols if c in events.columns]

    full_tbl = events[show_cols].copy()
    full_tbl["vix_pctchg"] = full_tbl["vix_pctchg"] * 100.0

    full_tbl = full_tbl.rename(
        columns={
            "event_type": "Type",
            "context_label": "Context",
            "event_label": "Event_Label",
            "vix_close": "VIX",
            "vix_pctchg": "VIX_Move_%",
            "move_mag": "Abs_Move_%",
            "vix_z20": "VIX_Z20",
            "spx_ret_1d": "SPX_1D_%",
            "spx_ret_3d": "SPX_3D_%",
            "spx_ret_5d": "SPX_5D_%",
            "spx_ret_10d": "SPX_10D_%",
            "spx_drawdown_pct": "Drawdown_%",
            "dist_20dma": "Dist_20DMA_%",
            "dist_50dma": "Dist_50DMA_%",
            "dist_100dma": "Dist_100DMA_%",
            "dist_200dma": "Dist_200DMA_%",
            "ma20_slope_5d": "MA20_Slope_5D_%",
            "ma50_slope_10d": "MA50_Slope_10D_%",
            "rsi14": "RSI14",
            "rv20": "RV20_%",
            "rv20_pctile_252": "RV20_Pctile_%",
            "qqq_iwm_5d": "QQQ_IWM_5D_%",
            "hyg_ief_5d": "HYG_IEF_5D_%",
            "ief_ret_1d": "IEF_1D_%",
            "tlt_ret_5d": "TLT_5D_%",
            "vix_vix3m_spread": "VIX_minus_VIX3M",
            "spx_fwd_1d": "Fwd_1D_%",
            "spx_fwd_2d": "Fwd_2D_%",
            "spx_fwd_3d": "Fwd_3D_%",
            "spx_fwd_5d": "Fwd_5D_%",
            "spx_fwd_10d": "Fwd_10D_%",
            "spx_fwd_20d": "Fwd_20D_%",
            "outcome_label": "Outcome",
        }
    )

    full_tbl.index = full_tbl.index.strftime("%Y-%m-%d")
    full_tbl.index.name = "Date"

    st.dataframe(
        safe_round_table(full_tbl, 2),
        use_container_width=True,
    )


with st.expander("Diagnostics"):
    diag_rows = [
        {"Item": "Market rows loaded", "Value": fmt_int(len(df))},
        {"Item": "Raw events before filters", "Value": fmt_int(len(events_raw))},
        {"Item": "Events after filters", "Value": fmt_int(len(events))},
        {"Item": f"Events with {forward_focus}D forward returns", "Value": fmt_int(len(historical_events))},
        {"Item": "Same setup historical sample", "Value": fmt_int(same_setup_n)},
        {"Item": "Nearest analogs returned", "Value": fmt_int(len(analogs))},
        {"Item": "Current data date", "Value": current_idx.strftime("%Y-%m-%d")},
        {"Item": "Last matched event date", "Value": last_event_idx.strftime("%Y-%m-%d")},
    ]

    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)


# ------------------------------- Footer -------------------------------------

st.caption("ADFM Analytics Platform | VIX Event Deep Dive | Data source: Yahoo Finance")
st.caption("© 2026 AD Fund Management LP")
