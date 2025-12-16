# app.py
# Hedge Timer
# Static (image) charts, no toggles, no sliders.
# Sidebar: About This Tool + Sanity check since 2020 (forward risk stats).
#
# Install:
#   pip install streamlit yfinance pandas numpy matplotlib
#
# Run:
#   streamlit run app.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf


# ============================== Page + Style ==============================
st.set_page_config(page_title="Hedge Timer", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 0.8rem; padding-bottom: 1.1rem; max-width: 1420px; }
h1,h2,h3 { letter-spacing: -0.2px; }
.small { opacity: 0.85; font-size: 0.92rem; }
.card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.09);
  border-radius: 16px;
  padding: 14px 16px;
}
.row { display:flex; justify-content:space-between; gap:14px; align-items:baseline; flex-wrap:wrap; }
.badge {
  display:inline-block; padding: 6px 10px; border-radius: 999px;
  border: 1px solid rgba(0,0,0,0.10);
  font-weight: 800; font-size: 0.92rem;
  background: rgba(0,0,0,0.03);
}
.b_good { background: rgba(16,185,129,0.16); border-color: rgba(16,185,129,0.25); }
.b_mid  { background: rgba(245,158,11,0.16); border-color: rgba(245,158,11,0.25); }
.b_bad  { background: rgba(239,68,68,0.16); border-color: rgba(239,68,68,0.25); }

.kv { display:flex; gap:10px; flex-wrap:wrap; margin-top:10px; }
.kv > div {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.09);
  padding: 7px 10px; border-radius: 12px; font-size: 0.92rem;
}
hr { opacity: 0.20; margin: 0.9rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

plt.rcParams["figure.dpi"] = 200


# ============================== Sidebar ==============================
with st.sidebar:
    st.markdown("## Hedge Timer")
    st.markdown("---")
    st.markdown("### About This Tool")
    st.markdown(
        """
Daily metrics: credit and breadth ratios (HYG/LQD, RSP/SPY as breadth proxy, XLY/XLP), volatility stress and term structure (VIX, VIX9D, VIX3M, VVIX when available), multi-timeframe RSI and MACD (daily, weekly, monthly), and trend confirmation on the decision index.

Decision output is a composite score (0 to 100). Thresholds are calibrated on the full sample since 2020-01-01 to balance early warning against over-trading.

Assumptions: close-to-close data, weekly is Friday close, monthly is month-end close, no transaction costs, borrow costs, or execution model. If Yahoo misses VIX9D, VIX3M, or VVIX, the model renormalizes around the remaining layers. If ^SPX is unavailable from Yahoo in your region, the app uses ^GSPC as the data source but labels it ^SPX in the UI.
""".strip()
    )
    st.markdown("---")
    st.markdown("### Sanity check since 2020")
    sanity_box = st.empty()


# ============================== Targets + Data Universe ==============================
CALIBRATION_START = "2020-01-01"
DISPLAY_START = "2020-01-01"

HORIZON_DAYS = 20
LEAD_LOOKBACK = 40

# Canonical decision instruments (what we show in headlines)
CANON_SDX = "^SPX"
CANON_NDX = "^NDX"

# Yahoo reality: ^SPX may fail, ^GSPC usually works for S&P 500
SPX_CANDIDATES = ["^SPX", "^GSPC"]
NDX_CANDIDATES = ["^NDX"]

AUX_TICKERS = [
    "SPY",  # used only as breadth proxy denominator for RSP/SPY when available
    "RSP",
    "XLY",
    "XLP",
    "HYG",
    "LQD",
    "IWM",
    "TLT",
    "^VIX",
    "^VIX9D",
    "^VIX3M",
    "^VVIX",
]

DOWNLOAD_TICKERS = list(dict.fromkeys(SPX_CANDIDATES + NDX_CANDIDATES + AUX_TICKERS))


# ============================== Helpers ==============================
def _today() -> date:
    return date.today()

def _start_date() -> date:
    # enough history to compute MA200 and multi-timeframe indicators cleanly
    return _today() - timedelta(days=int(10 * 365.25) + 200)

@st.cache_data(ttl=900, show_spinner=False)
def yf_download(tickers: List[str], start: date) -> pd.DataFrame:
    return yf.download(
        tickers=tickers,
        start=start.isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

def extract_close(df_raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    if isinstance(df_raw.columns, pd.MultiIndex):
        out = {}
        for t in tickers:
            if (t, "Close") in df_raw.columns:
                out[t] = df_raw[(t, "Close")]
            elif (t, "Adj Close") in df_raw.columns:
                out[t] = df_raw[(t, "Adj Close")]
        df = pd.DataFrame(out)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    # single ticker fallback
    if "Close" in df_raw.columns:
        df = df_raw[["Close"]].rename(columns={"Close": tickers[0]})
    elif "Adj Close" in df_raw.columns:
        df = df_raw[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
    else:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

def choose_source(df: pd.DataFrame, candidates: List[str], min_obs: int = 800) -> Optional[str]:
    for t in candidates:
        if t in df.columns:
            n = int(df[t].dropna().shape[0])
            if n >= min_obs:
                return t
    return None

def last_valid(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")

def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "NA"
    return f"{x*100:.2f}%"

def fmt_num(x: float, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"

def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b
    return out.replace([np.inf, -np.inf], np.nan)

def rolling_ma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).mean()

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))

def macd_hist(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    m = ema(s, fast) - ema(s, slow)
    sig = ema(m, signal)
    return m - sig

def resample_last(s: pd.Series, rule: str) -> pd.Series:
    return s.dropna().resample(rule).last()

def mtf_to_daily(daily_index: pd.DatetimeIndex, mtf_series: pd.Series) -> pd.Series:
    return mtf_series.reindex(daily_index, method="ffill")

def drawdown(px: pd.Series) -> pd.Series:
    s = px.dropna()
    peak = s.cummax()
    return s / peak - 1.0

def forward_min_return(px: pd.Series, h: int) -> pd.Series:
    s = px.dropna()
    fut_min = s[::-1].rolling(h, min_periods=1).min()[::-1].shift(-1)
    return fut_min / s - 1.0

def signal_events(score: pd.Series, t_short: int) -> pd.Series:
    sc = score.copy()
    evt = (sc >= t_short) & (sc.shift(1) < t_short)
    evt = evt.fillna(False)
    return evt

def find_drawdown_episodes_full(
    px: pd.Series,
    threshold: float = -0.08,
    recovery: float = -0.02
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, float]]:
    """
    Returns episodes as (start, end, trough, depth) where:
    start = first date drawdown <= threshold,
    end = first date drawdown >= recovery after start,
    trough = min drawdown date within episode,
    depth = min drawdown value (negative).
    """
    s = px.dropna()
    if len(s) < 260:
        return []
    dd = drawdown(s)

    episodes: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, float]] = []
    in_ep = False
    start = None

    for ts, v in dd.items():
        if (not in_ep) and (v <= threshold):
            in_ep = True
            start = ts
        if in_ep and (v >= recovery):
            seg = dd.loc[start:ts]
            trough = seg.idxmin()
            depth = float(seg.min())
            end = ts
            episodes.append((start, end, trough, depth))
            in_ep = False
            start = None

    if in_ep and start is not None:
        seg = dd.loc[start:]
        trough = seg.idxmin()
        depth = float(seg.min())
        end = seg.index[-1]
        episodes.append((start, end, trough, depth))

    return sorted(episodes, key=lambda x: x[3])  # deepest first


# ============================== Signal Model ==============================
@dataclass
class Component:
    key: str
    label: str
    weight: int

COMPONENTS: List[Component] = [
    Component("credit_risk", "Credit risk-off", 18),
    Component("breadth_risk", "Breadth weak", 14),
    Component("defensive_tape", "Defensive tape", 10),
    Component("vol_stress", "Vol stress", 18),
    Component("mtf_momentum", "RSI/MACD rollover (D/W/M)", 22),
    Component("trend_confirm", "Trend confirms", 18),
]

def compute_components(df: pd.DataFrame, target_col: str) -> Tuple[Dict[str, pd.Series], int]:
    idx = df.index

    # Breadth proxy denominator: prefer SPY for RSP/SPY, else fall back to target index
    spx_proxy = df.get("SPY", df.get(CANON_SDX, pd.Series(index=idx, dtype=float)))
    rsp = df.get("RSP", pd.Series(index=idx, dtype=float))
    xly = df.get("XLY", pd.Series(index=idx, dtype=float))
    xlp = df.get("XLP", pd.Series(index=idx, dtype=float))
    hyg = df.get("HYG", pd.Series(index=idx, dtype=float))
    lqd = df.get("LQD", pd.Series(index=idx, dtype=float))

    vix = df.get("^VIX", pd.Series(index=idx, dtype=float))
    vix9 = df.get("^VIX9D", pd.Series(index=idx, dtype=float))
    vix3m = df.get("^VIX3M", pd.Series(index=idx, dtype=float))
    vvix = df.get("^VVIX", pd.Series(index=idx, dtype=float))

    tgt = df.get(target_col, pd.Series(index=idx, dtype=float))

    # Credit
    credit = safe_ratio(hyg, lqd)
    credit_ma200 = rolling_ma(credit, 200)
    credit_ma20 = rolling_ma(credit, 20)
    credit_rollover = (credit < credit_ma200) & (credit_ma20 < rolling_ma(credit_ma200, 20))

    # Breadth
    rsp_base = safe_ratio(rsp, spx_proxy)
    rsp_ma200 = rolling_ma(rsp_base, 200)
    rsp_ma20 = rolling_ma(rsp_base, 20)
    breadth_rollover = (rsp_base < rsp_ma200) & (rsp_ma20 < rolling_ma(rsp_ma200, 20))

    # Defensive tape
    xly_xlp = safe_ratio(xly, xlp)
    xlyxlp_ma200 = rolling_ma(xly_xlp, 200)
    defensive = xly_xlp < xlyxlp_ma200

    # Vol stress
    vix_ma50 = rolling_ma(vix, 50)
    vol_1 = (vix > vix_ma50) & (vix.diff(10) > 0)
    vol_2 = (safe_ratio(vix9, vix) >= 1.00) if vix9.notna().sum() > 50 else pd.Series(False, index=idx)
    vol_3 = (safe_ratio(vix, vix3m) >= 1.00) if vix3m.notna().sum() > 50 else pd.Series(False, index=idx)
    vol_4 = (vvix >= vvix.rolling(252, min_periods=126).quantile(0.70)) if vvix.notna().sum() > 100 else pd.Series(False, index=idx)
    vol_stress = (vol_1 | vol_2 | vol_3) & (vol_4 | (vix >= 18))

    # RSI/MACD multi-timeframe
    rsi_d = rsi(tgt, 14)
    macd_d = macd_hist(tgt)

    w = resample_last(tgt, "W-FRI")
    m = resample_last(tgt, "M")

    rsi_w = mtf_to_daily(idx, rsi(w, 14))
    rsi_m = mtf_to_daily(idx, rsi(m, 14))
    macd_w = mtf_to_daily(idx, macd_hist(w))
    macd_m = mtf_to_daily(idx, macd_hist(m))

    # Rollover heuristics
    rsi_d_roll = ((rsi_d.shift(1) >= 70) & (rsi_d < 70)) | ((rsi_d >= 68) & (rsi_d.diff(5) < 0))
    rsi_w_roll = ((rsi_w.shift(1) >= 65) & (rsi_w < 65)) | ((rsi_w >= 62) & (rsi_w.diff(3) < 0))
    rsi_m_roll = ((rsi_m.shift(1) >= 60) & (rsi_m < 60)) | ((rsi_m >= 58) & (rsi_m.diff(2) < 0))

    macd_d_bear = (macd_d < 0) & (macd_d.shift(5) > macd_d)
    macd_w_bear = (macd_w < 0) & (macd_w.shift(3) > macd_w)
    macd_m_bear = (macd_m < 0) & (macd_m.shift(2) > macd_m)

    rsi_votes = (rsi_d_roll.astype(int) + rsi_w_roll.astype(int) + rsi_m_roll.astype(int))
    macd_votes = (macd_d_bear.astype(int) + macd_w_bear.astype(int) + macd_m_bear.astype(int))
    mtf_momentum = (rsi_votes >= 2) | (macd_votes >= 2) | ((rsi_votes >= 1) & (macd_votes >= 1))

    # Trend confirmation
    ma50 = rolling_ma(tgt, 50)
    ma200 = rolling_ma(tgt, 200)
    trend_confirm = (tgt < ma50) | (tgt < ma200)

    cond = {
        "credit_risk": credit_rollover,
        "breadth_risk": breadth_rollover,
        "defensive_tape": defensive,
        "vol_stress": vol_stress,
        "mtf_momentum": mtf_momentum,
        "trend_confirm": trend_confirm,
    }

    denom = 0
    for c in COMPONENTS:
        s = cond.get(c.key)
        if s is not None and s.notna().sum() > 200:
            denom += c.weight

    return cond, max(denom, 1)

def compute_score(df: pd.DataFrame, target_col: str) -> Tuple[pd.Series, int]:
    cond, denom = compute_components(df, target_col)
    score = pd.Series(0.0, index=df.index)
    for c in COMPONENTS:
        s = cond.get(c.key)
        if s is None or s.notna().sum() <= 200:
            continue
        score = score.add(s.astype(float) * c.weight, fill_value=0.0)
    score = (score / denom) * 100.0
    return score.clip(0, 100), denom

def calibrate_threshold(score_spx: pd.Series, score_ndx: pd.Series, px_spx: pd.Series, px_ndx: pd.Series) -> int:
    s0 = pd.to_datetime(CALIBRATION_START)
    best_t, best_obj = 68, -1e9

    for t in range(50, 86):
        objs = []
        for score, px in [(score_spx, px_spx), (score_ndx, px_ndx)]:
            sc = score.loc[score.index >= s0].dropna()
            pr = px.loc[px.index >= s0].dropna()
            idx = sc.index.intersection(pr.index)
            sc = sc.reindex(idx)
            pr = pr.reindex(idx)
            if len(sc) < 800:
                continue

            evt = signal_events(sc, t)
            rate = float(evt.mean())  # event-days as % of days
            fwd_min = forward_min_return(pr, HORIZON_DAYS).reindex(idx)
            hit = fwd_min[evt].dropna()

            if hit.shape[0] < 18:
                continue

            avg_worst = float(hit.mean())  # negative
            # objective: bigger negative forward risk after signal is good, too many signals is bad
            obj = (-avg_worst * 100.0) - (rate * 100.0 * 4.5)

            # mild preferences: avoid too rare or too frequent
            if rate < 0.010:
                obj -= 2.5
            if rate > 0.085:
                obj -= 3.5

            objs.append(obj)

        if not objs:
            continue

        obj_avg = float(np.mean(objs))
        if obj_avg > best_obj:
            best_obj = obj_avg
            best_t = t

    return int(best_t)

def stance_from_score(x: float, t_short: int) -> Tuple[str, str]:
    t_bias = max(40, t_short - 12)
    if x >= t_short:
        return "SHORT ALLOWED", "b_bad"
    if x >= t_bias:
        return "HEDGE BIAS", "b_mid"
    return "STAND DOWN", "b_good"

def pick_target_today(df: pd.DataFrame) -> str:
    # pick weaker: ^NDX vs ^SPX
    rs = safe_ratio(df[CANON_NDX], df[CANON_SDX]).dropna()
    if len(rs) < 260:
        return CANON_SDX
    rs_ma200 = rolling_ma(rs, 200)
    rs_ma20 = rolling_ma(rs, 20)
    if (last_valid(rs) < last_valid(rs_ma200)) and (last_valid(rs_ma20) < last_valid(rolling_ma(rs_ma200, 20))):
        return CANON_NDX
    return CANON_SDX

def forward_stats(score: pd.Series, px: pd.Series, t_short: int) -> Dict[str, float]:
    idx = score.dropna().index.intersection(px.dropna().index)
    sc = score.reindex(idx)
    pr = px.reindex(idx)

    evt = signal_events(sc, t_short)
    fwd_min = forward_min_return(pr, HORIZON_DAYS).reindex(idx)
    hit = fwd_min[evt].dropna()
    miss = fwd_min[~evt].dropna()

    def q(x: pd.Series, p: float) -> float:
        return float(np.nanquantile(x.values, p)) if len(x) else float("nan")

    return {
        "signal_rate": float(evt.mean()),
        "signals": float(evt.sum()),
        "avg_worst_signal": float(hit.mean()) if len(hit) else float("nan"),
        "med_worst_signal": q(hit, 0.50),
        "avg_worst_nosig": float(miss.mean()) if len(miss) else float("nan"),
        "med_worst_nosig": q(miss, 0.50),
    }

def lead_before_episode(score: pd.Series, start_ts: pd.Timestamp, t_short: int, lookback: int) -> int:
    s = score.dropna()
    if len(s) == 0:
        return -1

    # map start_ts onto nearest trading day in index
    if start_ts not in s.index:
        loc = s.index.get_indexer([start_ts], method="nearest")[0]
        start_ts = s.index[loc]

    evt = signal_events(s, t_short)

    loc = s.index.get_loc(start_ts)
    lo = max(0, loc - lookback)
    window_evt = evt.iloc[lo:loc + 1]
    hits = window_evt[window_evt]
    if hits.empty:
        return -1

    first = hits.index[0]
    return int(loc - s.index.get_loc(first))


# ============================== Static Chart (no calendar gaps) ==============================
def plot_price_and_score_image(
    price: pd.Series,
    score: pd.Series,
    t_short: int,
    episodes: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, float]],
    title: str,
) -> plt.Figure:
    dfp = pd.DataFrame({"price": price, "score": score}).dropna()
    dfp = dfp[dfp.index >= pd.Timestamp(DISPLAY_START)]
    if dfp.empty:
        fig = plt.figure(figsize=(13, 6))
        plt.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return fig

    # Integer x-axis removes weekend/holiday spacing
    x = np.arange(len(dfp))
    idx = dfp.index

    ma50 = rolling_ma(price, 50).reindex(idx)
    ma200 = rolling_ma(price, 200).reindex(idx)

    t_bias = max(40, t_short - 12)

    fig = plt.figure(figsize=(13.6, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 1.25], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Shade drawdown start->end (recovery)
    for start, end, trough, depth in episodes[:8]:
        if end < idx.min() or start > idx.max():
            continue
        s_loc = idx.get_indexer([start], method="nearest")[0]
        e_loc = idx.get_indexer([end], method="nearest")[0]
        if e_loc < s_loc:
            s_loc, e_loc = e_loc, s_loc
        ax1.axvspan(s_loc, e_loc, alpha=0.09)

    # Price lines
    ax1.plot(x, dfp["price"].values, linewidth=2.2, label="Price", color="#1f77b4")
    ax1.plot(x, ma50.values, linewidth=1.5, label="MA50", color="#d62728")
    ax1.plot(x, ma200.values, linewidth=1.5, label="MA200", color="#2ca02c")

    # Signal markers: crossing events only
    evt = signal_events(dfp["score"], t_short).values
    if np.any(evt):
        ax1.scatter(
            x[evt],
            dfp["price"].values[evt],
            marker="v",
            s=62,
            color="#9467bd",
            label="Short signal",
            zorder=5,
        )

    ax1.grid(False)
    ax1.set_ylabel("")
    ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    # Score panel
    ax2.plot(x, dfp["score"].values, linewidth=1.9, label="Score", color="#ff7f0e")
    ax2.axhline(t_short, linewidth=1.0, alpha=0.70, color="#111827")
    ax2.axhline(t_bias, linewidth=1.0, alpha=0.35, color="#111827")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Score (0-100)")
    ax2.grid(False)

    # Yearly ticks only (no collisions)
    year_starts = pd.date_range(idx.min().normalize(), idx.max().normalize(), freq="YS")
    tick_pos: List[int] = []
    tick_lbl: List[str] = []
    for d in year_starts:
        loc = idx.get_indexer([d], method="nearest")[0]
        if 0 <= loc < len(idx):
            if len(tick_pos) == 0 or abs(loc - tick_pos[-1]) > 40:
                tick_pos.append(int(loc))
                tick_lbl.append(d.strftime("%Y"))
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_lbl)
    ax2.tick_params(axis="x", labelsize=10)

    # Title + legend, clean spacing
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.985)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=5,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
    return fig


def render_table_image(df: pd.DataFrame, title: str) -> plt.Figure:
    # simple static table figure
    fig = plt.figure(figsize=(13.6, 4.2))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # build table
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.35)

    # header styling
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(cell.get_height() * 1.15)

    fig.tight_layout()
    return fig


# ============================== App ==============================
st.title("Hedge Timer")
st.caption("Decision slip for hedging equity beta using ^SPX or ^NDX, with early-warning stress plus multi-timeframe momentum and trend confirmation.")

start = _start_date()
raw = yf_download(DOWNLOAD_TICKERS, start)
df0 = extract_close(raw, DOWNLOAD_TICKERS)

if df0.empty:
    st.error("Yahoo feed failed. Retry later.")
    st.stop()

# Choose data sources for SPX and NDX, then canonicalize column names to ^SPX and ^NDX
spx_src = choose_source(df0, SPX_CANDIDATES, min_obs=900)
ndx_src = choose_source(df0, NDX_CANDIDATES, min_obs=900)

if spx_src is None:
    st.error("Could not load ^SPX (or fallback ^GSPC) from Yahoo.")
    st.stop()

if ndx_src is None:
    st.error("Could not load ^NDX from Yahoo.")
    st.stop()

df = df0.copy()
if spx_src != CANON_SDX:
    df[CANON_SDX] = df[spx_src]
if ndx_src != CANON_NDX:
    df[CANON_NDX] = df[ndx_src]

# Ensure canonical cols exist
if CANON_SDX not in df.columns or CANON_NDX not in df.columns:
    st.error("Data load incomplete for ^SPX/^NDX.")
    st.stop()

# Score series and coverage denominators
score_spx, denom_spx = compute_score(df, CANON_SDX)
score_ndx, denom_ndx = compute_score(df, CANON_NDX)

t_short = calibrate_threshold(score_spx, score_ndx, df[CANON_SDX], df[CANON_NDX])
t_bias = max(40, t_short - 12)

target = pick_target_today(df)

spx_last = last_valid(df[CANON_SDX])
ndx_last = last_valid(df[CANON_NDX])
vix_last = last_valid(df["^VIX"]) if "^VIX" in df.columns else float("nan")

dd_spx = last_valid(drawdown(df[CANON_SDX]))
dd_ndx = last_valid(drawdown(df[CANON_NDX]))

score_today_spx = float(last_valid(score_spx))
score_today_ndx = float(last_valid(score_ndx))

stance_spx, badge_spx = stance_from_score(score_today_spx, t_short)
stance_ndx, badge_ndx = stance_from_score(score_today_ndx, t_short)

stance_target = stance_ndx if target == CANON_NDX else stance_spx
badge_target = badge_ndx if target == CANON_NDX else badge_spx
score_target = score_today_ndx if target == CANON_NDX else score_today_spx
coverage_target = denom_ndx if target == CANON_NDX else denom_spx

# Sidebar sanity check
stats_spx = forward_stats(score_spx[score_spx.index >= CALIBRATION_START], df[CANON_SDX][df.index >= CALIBRATION_START], t_short)
stats_ndx = forward_stats(score_ndx[score_ndx.index >= CALIBRATION_START], df[CANON_NDX][df.index >= CALIBRATION_START], t_short)

sanity_box.markdown(
    f"""
Forward risk stats (next {HORIZON_DAYS} sessions). The key number is worst forward move after a short signal event versus when standing down.

**{CANON_SDX}** | signal event rate: {stats_spx["signal_rate"]*100:.1f}% | events: {int(stats_spx["signals"])}  
Avg worst next {HORIZON_DAYS}d after signal: {stats_spx["avg_worst_signal"]*100:.2f}% | when no signal: {stats_spx["avg_worst_nosig"]*100:.2f}%  
Median worst next {HORIZON_DAYS}d after signal: {stats_spx["med_worst_signal"]*100:.2f}% | when no signal: {stats_spx["med_worst_nosig"]*100:.2f}%  

**{CANON_NDX}** | signal event rate: {stats_ndx["signal_rate"]*100:.1f}% | events: {int(stats_ndx["signals"])}  
Avg worst next {HORIZON_DAYS}d after signal: {stats_ndx["avg_worst_signal"]*100:.2f}% | when no signal: {stats_ndx["avg_worst_nosig"]*100:.2f}%  
Median worst next {HORIZON_DAYS}d after signal: {stats_ndx["med_worst_signal"]*100:.2f}% | when no signal: {stats_ndx["med_worst_nosig"]*100:.2f}%
""".strip()
)

st.markdown(
    f"""
<div class="card">
  <div class="row">
    <div>
      <span class="badge {badge_target}">{stance_target}</span>
      <span class="small" style="margin-left:10px;">Decision target: <b>{target}</b></span>
    </div>
    <div class="small">Score: <b>{int(round(score_target))}/100</b> | Threshold: <b>{t_short}</b> | Hedge bias: <b>{t_bias}</b> | Data coverage: <b>{coverage_target}</b> weight-points</div>
  </div>

  <div class="kv">
    <div>{CANON_SDX}: <b>{fmt_num(spx_last,2)}</b> | DD: <b>{fmt_pct(dd_spx)}</b> | Stance: <b>{stance_spx}</b></div>
    <div>{CANON_NDX}: <b>{fmt_num(ndx_last,2)}</b> | DD: <b>{fmt_pct(dd_ndx)}</b> | Stance: <b>{stance_ndx}</b></div>
    <div>VIX: <b>{fmt_num(vix_last,2)}</b></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<hr/>", unsafe_allow_html=True)

c1, c2 = st.columns([1, 1], gap="large")
with c1:
    st.subheader("How to execute")
    if stance_target == "SHORT ALLOWED":
        st.write(
            f"Short is permitted in {target}. Sizing: add when the score stays above {t_short} and price stays below MA50. Cover discipline: if the score falls below {t_bias} for 3 sessions, reduce. If it snaps below {t_bias - 5}, cover fast."
        )
    elif stance_target == "HEDGE BIAS":
        st.write(
            f"Keep it smaller. If you hedge, use partial size or options. Wait for score to clear {t_short} for full permission."
        )
    else:
        st.write("Stand down. If you hedge, keep it light and tactical.")
with c2:
    st.subheader("What flips it")
    st.write(f"At or above {t_short}: short allowed. Between {t_bias} and {t_short}: hedge bias. Below {t_bias}: stand down.")

st.markdown("<hr/>", unsafe_allow_html=True)

episodes_target = find_drawdown_episodes_full(df[target], threshold=-0.08, recovery=-0.02)
score_target_series = score_ndx if target == CANON_NDX else score_spx

fig = plot_price_and_score_image(
    price=df[target],
    score=score_target_series,
    t_short=t_short,
    episodes=episodes_target,
    title=f"{target}: price, moving averages, short signals, and score (from Jan 2020, no calendar gaps)",
)
st.pyplot(fig, use_container_width=True)

st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("Did it warn before major selloffs?")

st.write(f"We take the largest drawdowns and ask if the score crossed {t_short} within the prior {LEAD_LOOKBACK} sessions.")

def summarize_eps(name: str, eps: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, float]], score: pd.Series) -> pd.DataFrame:
    rows = []
    for start_ts, end_ts, trough_ts, depth in eps[:10]:
        lead = lead_before_episode(score, start_ts, t_short, LEAD_LOOKBACK)
        rows.append(
            {
                "Index": name,
                "Start": start_ts.date().isoformat(),
                "End": end_ts.date().isoformat(),
                "Trough": trough_ts.date().isoformat(),
                "Depth": f"{depth*100:.2f}%",
                "Lead (sessions)": lead if lead >= 0 else "No",
            }
        )
    return pd.DataFrame(rows)

tbl = pd.concat(
    [
        summarize_eps(CANON_SDX, find_drawdown_episodes_full(df[CANON_SDX]), score_spx),
        summarize_eps(CANON_NDX, find_drawdown_episodes_full(df[CANON_NDX]), score_ndx),
    ],
    ignore_index=True,
)

tbl_fig = render_table_image(tbl, "Largest drawdowns and whether the signal fired beforehand")
st.pyplot(tbl_fig, use_container_width=True)
