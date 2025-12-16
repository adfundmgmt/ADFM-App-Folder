# app.py
# Hedge Timing v2: clean visuals + pre-drawdown signal design + built-in historical check
#
# Install:
#   pip install streamlit yfinance pandas numpy plotly
#
# Run:
#   streamlit run app.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ============================== Page + Style ==============================
st.set_page_config(page_title="Hedge Timing (SPY / QQQ)", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 0.9rem; padding-bottom: 1.1rem; max-width: 1400px; }
h1,h2,h3 { letter-spacing: -0.2px; }
.small { opacity: 0.86; font-size: 0.92rem; }
.card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.09);
  border-radius: 16px;
  padding: 14px 16px;
}
.badge {
  display:inline-block; padding: 6px 10px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  font-weight: 700; font-size: 0.92rem;
}
.b_good { background: rgba(16,185,129,0.14); }
.b_mid  { background: rgba(245,158,11,0.14); }
.b_bad  { background: rgba(239,68,68,0.14); }
.kv { display:flex; gap:10px; flex-wrap:wrap; margin-top:10px; }
.kv > div {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.09);
  padding: 7px 10px; border-radius: 12px; font-size: 0.92rem;
}
hr { opacity: 0.22; margin: 0.8rem 0; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================== Core Parameters ==============================
DEFAULT_TICKERS = {
    # Equity beta
    "SPY": "SPY",
    "QQQ": "QQQ",
    "RSP": "RSP",
    "IWM": "IWM",
    "XLY": "XLY",
    "XLP": "XLP",
    # Credit
    "HYG": "HYG",
    "LQD": "LQD",
    # Rates
    "TLT": "TLT",
    "^TNX": "^TNX",  # 10Y yield index (usually 10x the yield)
    "^IRX": "^IRX",  # 13-week index (often 10x the yield)
    # Vol regime
    "^VIX": "^VIX",
    "^VIX9D": "^VIX9D",   # may be missing sometimes
    "^VIX3M": "^VIX3M",   # may be missing sometimes
    "^VVIX": "^VVIX",     # may be missing sometimes
}


# ============================== Utilities ==============================
def _today() -> date:
    return date.today()

def _start_date(years: int) -> date:
    return _today() - timedelta(days=int(years * 365.25) + 80)

@st.cache_data(ttl=900, show_spinner=False)
def yf_download(tickers: List[str], start: date) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    return df

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
    else:
        # single ticker
        if "Close" in df_raw.columns:
            df = df_raw[["Close"]].rename(columns={"Close": tickers[0]})
        elif "Adj Close" in df_raw.columns:
            df = df_raw[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        else:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

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
    out = out.replace([np.inf, -np.inf], np.nan)
    return out

def drawdown(px: pd.Series) -> pd.Series:
    peak = px.cummax()
    return px / peak - 1.0

def ma(px: pd.Series, w: int) -> pd.Series:
    return px.rolling(w).mean()

def slope_z(px: pd.Series, w: int = 20) -> pd.Series:
    # normalized slope proxy: rolling z-scored slope of prices (fast and stable)
    x = np.arange(w, dtype=float)
    x = (x - x.mean()) / (x.std() + 1e-12)

    def _fit(y: np.ndarray) -> float:
        if np.any(np.isnan(y)):
            return np.nan
        y = (y - y.mean()) / (y.std() + 1e-12)
        b = np.polyfit(x, y, 1)[0]
        return float(b)

    return px.rolling(w).apply(lambda y: _fit(np.array(y, dtype=float)), raw=False)

def forward_return(px: pd.Series, h: int) -> pd.Series:
    return px.shift(-h) / px - 1.0

def forward_min_return(px: pd.Series, h: int) -> pd.Series:
    # worst forward move over next h sessions: min_future / today - 1
    fut_min = px[::-1].rolling(h, min_periods=1).min()[::-1].shift(-1)
    return fut_min / px - 1.0

def pad_range(series: pd.Series, pad: float = 0.02) -> Tuple[float, float]:
    s = series.dropna()
    if len(s) < 10:
        return (None, None)
    lo = float(s.min())
    hi = float(s.max())
    span = max(hi - lo, 1e-9)
    return lo - pad * span, hi + pad * span


# ============================== Signal Model ==============================
@dataclass
class SignalComponent:
    key: str
    label: str
    weight: int

@dataclass
class SignalState:
    score_series: pd.Series
    score_today: int
    coverage_today: int
    stance: str
    badge_class: str
    triggered: bool
    reasons_today: List[str]
    contrib_today: pd.DataFrame


COMPONENTS: List[SignalComponent] = [
    # Vol term structure and stress: tends to show up early when fragility rises
    SignalComponent("vix_rising", "VIX above MA50 and rising", 14),
    SignalComponent("vix9_invert", "VIX9D/VIX >= 1.00 (front stress)", 18),
    SignalComponent("vix_term_invert", "VIX/VIX3M >= 1.00 (term inversion)", 14),
    SignalComponent("vvix_spike", "VVIX elevated (vol of vol)", 6),

    # Credit and breadth: these are the main early-warning layers
    SignalComponent("credit_200", "HYG/LQD below 200d MA (credit risk-off)", 18),
    SignalComponent("credit_down", "HYG/LQD trending down (30d slope < 0)", 8),

    SignalComponent("rsp_200", "RSP/SPY below 200d MA (breadth weak)", 10),
    SignalComponent("xly_xlp_200", "XLY/XLP below 200d MA (defensive tape)", 8),
    SignalComponent("iwm_200", "IWM/SPY below 200d MA (small caps lag)", 4),

    # Price confirmation: used as permission to size shorts, not as the early trigger
    SignalComponent("trend_50", "Target below 50d MA", 8),
    SignalComponent("trend_200", "Target below 200d MA", 10),
    SignalComponent("mom_20", "Target 20d return < 0", 2),
]

def compute_signal(px: pd.DataFrame, target: str) -> SignalState:
    df = px.copy()

    # Convenience
    tgt = df[target].dropna()

    # Build series we need (best effort, degrade gracefully)
    vix = df.get("^VIX", pd.Series(index=df.index, dtype=float))
    vix9 = df.get("^VIX9D", pd.Series(index=df.index, dtype=float))
    vix3m = df.get("^VIX3M", pd.Series(index=df.index, dtype=float))
    vvix = df.get("^VVIX", pd.Series(index=df.index, dtype=float))

    hyg = df.get("HYG", pd.Series(index=df.index, dtype=float))
    lqd = df.get("LQD", pd.Series(index=df.index, dtype=float))

    spy = df.get("SPY", pd.Series(index=df.index, dtype=float))
    rsp = df.get("RSP", pd.Series(index=df.index, dtype=float))
    iwm = df.get("IWM", pd.Series(index=df.index, dtype=float))
    xly = df.get("XLY", pd.Series(index=df.index, dtype=float))
    xlp = df.get("XLP", pd.Series(index=df.index, dtype=float))

    # Derived
    vix_ma50 = ma(vix, 50)
    vix_slope20 = slope_z(vix, 20)

    vix9_ratio = safe_ratio(vix9, vix)
    vix_term = safe_ratio(vix, vix3m)

    credit = safe_ratio(hyg, lqd)
    credit_ma200 = ma(credit, 200)
    credit_slope30 = slope_z(credit, 30)

    rsp_spy = safe_ratio(rsp, spy)
    rsp_spy_ma200 = ma(rsp_spy, 200)

    iwm_spy = safe_ratio(iwm, spy)
    iwm_spy_ma200 = ma(iwm_spy, 200)

    xly_xlp = safe_ratio(xly, xlp)
    xly_xlp_ma200 = ma(xly_xlp, 200)

    tgt_ma50 = ma(tgt, 50)
    tgt_ma200 = ma(tgt, 200)
    tgt_ret20 = tgt.pct_change(20)

    # Conditions as series
    cond: Dict[str, pd.Series] = {}

    cond["vix_rising"] = (vix > vix_ma50) & (vix_slope20 > 0.10)
    cond["vix9_invert"] = (vix9_ratio >= 1.00)
    cond["vix_term_invert"] = (vix_term >= 1.00)
    cond["vvix_spike"] = (vvix >= vvix.rolling(252).quantile(0.70))

    cond["credit_200"] = (credit < credit_ma200)
    cond["credit_down"] = (credit_slope30 < -0.05)

    cond["rsp_200"] = (rsp_spy < rsp_spy_ma200)
    cond["xly_xlp_200"] = (xly_xlp < xly_xlp_ma200)
    cond["iwm_200"] = (iwm_spy < iwm_spy_ma200)

    # Align to full index
    cond["trend_50"] = (df[target] < df[target].rolling(50).mean())
    cond["trend_200"] = (df[target] < df[target].rolling(200).mean())
    cond["mom_20"] = (df[target].pct_change(20) < 0)

    # Active weight coverage (handles missing series)
    active_weights = []
    for c in COMPONENTS:
        s = cond.get(c.key)
        if s is not None and s.dropna().shape[0] > 50:
            active_weights.append(c.weight)
    denom = max(1, int(sum(active_weights)))

    score = pd.Series(index=df.index, dtype=float)
    score[:] = 0.0

    for c in COMPONENTS:
        s = cond.get(c.key)
        if s is None or s.dropna().shape[0] <= 50:
            continue
        score = score.add(s.astype(float) * c.weight, fill_value=0.0)

    score = (score / denom) * 100.0
    score = score.clip(lower=0, upper=100)

    score_today = int(round(float(last_valid(score))))
    coverage_today = int(round(denom))

    # Stance mapping (tight, action-oriented)
    if score_today >= 70:
        stance = "SHORT ALLOWED"
        badge_class = "b_bad"
        triggered = True
    elif score_today >= 55:
        stance = "HEDGE BIAS"
        badge_class = "b_mid"
        triggered = True
    else:
        stance = "STAND DOWN"
        badge_class = "b_good"
        triggered = False

    # Reasons today + contribution breakdown
    today = score.dropna().index[-1]
    rows = []
    reasons = []
    for c in COMPONENTS:
        s = cond.get(c.key)
        if s is None or s.dropna().shape[0] <= 50:
            continue
        is_on = bool(s.loc[today]) if today in s.index and pd.notna(s.loc[today]) else False
        pts = (c.weight / denom) * 100.0 if is_on else 0.0
        rows.append({"Signal": c.label, "On": is_on, "WeightPts": round((c.weight / denom) * 100.0, 1), "ContribPts": round(pts, 1)})
        if is_on:
            reasons.append(c.label)

    contrib = pd.DataFrame(rows).sort_values("ContribPts", ascending=False)

    return SignalState(
        score_series=score,
        score_today=score_today,
        coverage_today=coverage_today,
        stance=stance,
        badge_class=badge_class,
        triggered=triggered,
        reasons_today=reasons[:6],
        contrib_today=contrib,
    )


# ============================== Drawdown Episode Finder ==============================
@dataclass
class DrawdownEpisode:
    start: pd.Timestamp
    trough: pd.Timestamp
    end: pd.Timestamp
    depth: float

def find_drawdown_episodes(px: pd.Series, threshold: float = -0.08, recovery: float = -0.02) -> List[DrawdownEpisode]:
    s = px.dropna()
    if len(s) < 260:
        return []
    dd = drawdown(s)

    episodes: List[DrawdownEpisode] = []
    i = 0
    idx = dd.index

    while i < len(dd):
        if dd.iloc[i] <= threshold:
            # start: last time drawdown was near 0 before this break
            pre = dd.iloc[:i]
            if len(pre) == 0:
                start_i = i
            else:
                near0 = pre[pre >= -0.003]
                start_i = dd.index.get_loc(near0.index[-1]) if len(near0) else max(0, i - 5)

            # end: when we recover above recovery
            j = i
            while j < len(dd) and dd.iloc[j] <= recovery:
                j += 1
            end_i = min(len(dd) - 1, max(i, j - 1))

            seg = dd.iloc[i:end_i + 1]
            trough_ts = seg.idxmin()
            depth = float(seg.min())

            episodes.append(
                DrawdownEpisode(
                    start=idx[start_i],
                    trough=trough_ts,
                    end=idx[end_i],
                    depth=depth,
                )
            )
            i = end_i + 1
        else:
            i += 1

    # Deduplicate overlapping (keep deeper one)
    episodes_sorted = sorted(episodes, key=lambda e: e.start)
    pruned: List[DrawdownEpisode] = []
    for ep in episodes_sorted:
        if not pruned:
            pruned.append(ep)
            continue
        last = pruned[-1]
        if ep.start <= last.end:
            # overlap: keep deeper episode, extend end
            if ep.depth < last.depth:
                pruned[-1] = DrawdownEpisode(start=last.start, trough=ep.trough, end=max(last.end, ep.end), depth=ep.depth)
            else:
                pruned[-1] = DrawdownEpisode(start=last.start, trough=last.trough, end=max(last.end, ep.end), depth=last.depth)
        else:
            pruned.append(ep)

    # Biggest first
    pruned = sorted(pruned, key=lambda e: e.depth)  # more negative first
    return pruned


# ============================== Plotting (no grids, tight ranges) ==============================
def fig_price_signals(
    px: pd.Series,
    score: pd.Series,
    threshold: int,
    episodes: List[DrawdownEpisode],
    title: str,
    lookback_days: int,
) -> go.Figure:
    s = px.dropna()
    s = s.iloc[-lookback_days:] if len(s) > lookback_days else s

    ma50 = s.rolling(50).mean()
    ma200 = s.rolling(200).mean()

    sc = score.reindex(s.index).dropna()
    sig = sc >= threshold

    ylo, yhi = pad_range(s, pad=0.015)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=s.index, y=s, name="Price", mode="lines", line=dict(width=2.2)))
    fig.add_trace(go.Scatter(x=ma50.index, y=ma50, name="MA50", mode="lines", line=dict(width=1.2)))
    fig.add_trace(go.Scatter(x=ma200.index, y=ma200, name="MA200", mode="lines", line=dict(width=1.2)))

    # Signal markers
    sig_idx = sc.index[sig.reindex(sc.index, fill_value=False)]
    sig_idx = sig_idx[(sig_idx >= s.index.min()) & (sig_idx <= s.index.max())]
    if len(sig_idx):
        fig.add_trace(
            go.Scatter(
                x=sig_idx,
                y=s.reindex(sig_idx),
                name="Signal",
                mode="markers",
                marker=dict(symbol="triangle-down", size=9),
                hovertemplate="Signal<br>%{x|%Y-%m-%d}<extra></extra>",
            )
        )

    # Shade drawdown episodes inside window
    for ep in episodes:
        if ep.end < s.index.min() or ep.start > s.index.max():
            continue
        x0 = max(ep.start, s.index.min())
        x1 = min(ep.end, s.index.max())
        fig.add_vrect(x0=x0, x1=x1, opacity=0.10, line_width=0)

    fig.update_layout(
        title=title,
        height=460,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, range=[ylo, yhi] if (ylo is not None and yhi is not None) else None)
    return fig

def fig_score(score: pd.Series, threshold: int, lookback_days: int) -> go.Figure:
    s = score.dropna()
    s = s.iloc[-lookback_days:] if len(s) > lookback_days else s

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s, name="Score", mode="lines", line=dict(width=2.0)))
    fig.add_hline(y=threshold, line_width=1, opacity=0.45)

    fig.update_layout(
        title="Signal score (0 to 100) vs threshold",
        height=260,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, range=[0, 100])
    return fig

def fig_episode_bar(episodes: List[DrawdownEpisode], leads: List[int], title: str) -> go.Figure:
    # Bar of drawdown depth with lead time in hover
    labels = []
    depths = []
    hover = []
    for ep, ld in zip(episodes, leads):
        lbl = f"{ep.trough:%Y-%m}"
        labels.append(lbl)
        depths.append(ep.depth * 100)
        hover.append(f"Start: {ep.start:%Y-%m-%d}<br>Trough: {ep.trough:%Y-%m-%d}<br>End: {ep.end:%Y-%m-%d}<br>Depth: {ep.depth*100:.2f}%<br>Lead time: {ld} sessions")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=depths, name="Depth (%)", hovertext=hover, hoverinfo="text"))
    fig.update_layout(
        title=title,
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, title="Drawdown depth (%)")
    return fig


# ============================== Sidebar ==============================
with st.sidebar:
    st.header("Hedge Timing v2")

    target_mode = st.selectbox("Decision target", ["Auto (pick weaker)", "SPY", "QQQ"], index=0)
    years = st.slider("History (years)", 2, 15, 7)
    window_days = st.slider("Charts window (days)", 120, 1500, 540, step=30)

    st.divider()
    threshold = st.slider("Short threshold (score)", 40, 85, 65)
    episode_thresh = st.slider("Define 'significant drawdown' as", 5, 20, 8)  # %
    episode_recover = st.slider("Episode ends when drawdown recovers above", 0, 8, 2)  # %
    lead_lookback = st.slider("Count lead time within prior sessions", 10, 90, 40)

    st.divider()
    show_contrib = st.toggle("Show score breakdown", value=True)
    show_backtest = st.toggle("Show episode check", value=True)


# ============================== Load Data ==============================
st.title("SPY / QQQ Hedge Timing")
st.caption("Goal: a clean, early-warning signal set that tends to turn on before big drawdowns, then uses trend as permission to size.")

tickers = list(DEFAULT_TICKERS.values())
start = _start_date(years)

raw = yf_download(tickers, start)
close = extract_close(raw, tickers)

if close.empty or "SPY" not in close.columns or "QQQ" not in close.columns:
    st.error("Yahoo feed failed for SPY/QQQ. Retry later.")
    st.stop()

# Work on a unified panel with everything we can get
df = close.copy()

# Normalize yield indices if present (usually 10x)
for ysym in ["^TNX", "^IRX"]:
    if ysym in df.columns:
        df[ysym] = df[ysym] / 10.0

# Decide target (auto picks weaker by relative trend)
def pick_target(df_: pd.DataFrame) -> str:
    if target_mode in ["SPY", "QQQ"]:
        return target_mode
    rs = safe_ratio(df_["QQQ"], df_["SPY"]).dropna()
    if len(rs) < 260:
        return "SPY"
    rs_ma200 = rs.rolling(200).mean()
    rs_slope30 = slope_z(rs, 30)
    if last_valid(rs) < last_valid(rs_ma200) and last_valid(rs_slope30) < 0:
        return "QQQ"
    return "SPY"

target = pick_target(df)

# ============================== Compute Signal ==============================
sig = compute_signal(df, target=target)

# Basic tape metrics
tgt_px = df[target].dropna()
tgt_dd = drawdown(tgt_px)
vix_last = last_valid(df["^VIX"]) if "^VIX" in df.columns else float("nan")
tgt_last = last_valid(tgt_px)

# ============================== Decision Panel ==============================
badge = f"<span class='badge {sig.badge_class}'>{sig.stance}</span>"
st.markdown(
    f"""
<div class="card">
  <div style="display:flex; align-items:baseline; justify-content:space-between; gap:16px;">
    <div style="font-size:1.25rem; font-weight:800;">{badge} <span class="small">Target: <b>{target}</b></span></div>
    <div class="small">Score: <b>{sig.score_today}/100</b> | Data coverage: <b>{sig.coverage_today}</b> weight-points</div>
  </div>
  <div class="kv">
    <div>{target} last: <b>{fmt_num(tgt_last,2)}</b></div>
    <div>{target} drawdown: <b>{fmt_pct(last_valid(tgt_dd))}</b></div>
    <div>VIX: <b>{fmt_num(vix_last,2)}</b></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2 = st.columns([1, 1], gap="large")
with c1:
    st.subheader("Why it is on (today)")
    if sig.reasons_today:
        st.write("\n".join([f"• {r}" for r in sig.reasons_today]))
    else:
        st.write("• None. The early-warning layers are quiet.")
with c2:
    st.subheader("How to execute (tight)")
    if sig.score_today >= threshold:
        st.write(
            f"• You have permission to run an index short in {target}. If price is still above MA50, size smaller. If price is below MA50 and credit is weak, size can be larger.\n"
            f"• Your first cover trigger is mechanical: score drops back below {threshold} and stays there for 3 sessions."
        )
    elif sig.score_today >= (threshold - 10):
        st.write(
            f"• You are in the gray zone. Keep it small or hedge with optionality. Demand confirmation from trend (below MA50) and credit (HYG/LQD down)."
        )
    else:
        st.write("• Stand down. This is usually a tape that punishes premature shorts.")

st.markdown("<hr/>", unsafe_allow_html=True)

# ============================== Charts ==============================
episodes = find_drawdown_episodes(
    tgt_px,
    threshold=-(episode_thresh / 100.0),
    recovery=-(episode_recover / 100.0),
)

st.plotly_chart(
    fig_price_signals(
        px=tgt_px,
        score=sig.score_series,
        threshold=threshold,
        episodes=episodes,
        title=f"{target} price with signals (triangles) and significant drawdown windows (shaded)",
        lookback_days=window_days,
    ),
    use_container_width=True,
)
st.plotly_chart(fig_score(sig.score_series, threshold=threshold, lookback_days=window_days), use_container_width=True)

# ============================== Score Breakdown ==============================
if show_contrib:
    st.subheader("Score breakdown (today)")
    st.caption("This tells you exactly what is driving the short permission slip.")
    if sig.contrib_today.empty:
        st.write("No breakdown available (missing data in key inputs).")
    else:
        # Minimal, readable table (no grids in charts already; table is compact)
        show = sig.contrib_today.copy()
        show["On"] = show["On"].map(lambda x: "Yes" if x else "No")
        show = show[["Signal", "On", "WeightPts", "ContribPts"]]
        st.dataframe(show, use_container_width=True, hide_index=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ============================== Episode Check (this answers your Aug 2024, Mar/Apr 2025 request) ==============================
if show_backtest:
    st.subheader("Did this framework flip on before big drawdowns?")
    st.caption(
        "We detect significant drawdown episodes from price itself, then measure whether the score crossed the threshold before the episode."
    )

    # Compute lead times to episode start (first signal within lookback)
    score = sig.score_series.dropna()
    sig_on = score >= threshold

    shown_eps = episodes[:10]  # biggest 10
    leads = []
    for ep in shown_eps:
        # search for any signal in [start - lead_lookback, start]
        start_loc = score.index.get_indexer([ep.start], method="nearest")[0]
        lo = max(0, start_loc - lead_lookback)
        hi = start_loc
        window = sig_on.iloc[lo:hi + 1]
        if window.any():
            first_sig_idx = window[window].index[0]
            lead = int((ep.start - first_sig_idx).days)  # calendar day proxy
            # convert to sessions proxy using index positions
            lead_sessions = int(score.index.get_loc(ep.start) - score.index.get_loc(first_sig_idx))
            leads.append(max(0, lead_sessions))
        else:
            leads.append(-1)

    if len(shown_eps) == 0:
        st.write("No significant drawdown episodes found in this window. Increase history or lower episode threshold.")
    else:
        st.plotly_chart(
            fig_episode_bar(shown_eps, leads, title="Largest drawdowns and how many sessions the signal led (if any)"),
            use_container_width=True,
        )

        # Highlight specifically around Aug 2024 and Mar/Apr 2025 if present
        def tag_episode(ep: DrawdownEpisode) -> Optional[str]:
            m = ep.trough.strftime("%Y-%m")
            if m == "2024-08":
                return "Aug 2024"
            if m in ["2025-03", "2025-04"]:
                return "Mar/Apr 2025"
            return None

        hits = []
        for ep, ld in zip(shown_eps, leads):
            tag = tag_episode(ep)
            if tag is None:
                continue
            hits.append((tag, ep.start, ep.trough, ep.depth, ld))

        if hits:
            st.markdown("**Your referenced episodes (if detected):**")
            for tag, stt, tr, depth, ld in hits:
                if ld >= 0:
                    st.write(f"• {tag}: signal fired **{ld} sessions** before episode start; depth {depth*100:.2f}% (start {stt:%Y-%m-%d}, trough {tr:%Y-%m-%d}).")
                else:
                    st.write(f"• {tag}: no signal inside the prior {lead_lookback} sessions (start {stt:%Y-%m-%d}, trough {tr:%Y-%m-%d}, depth {depth*100:.2f}%).")
        else:
            st.write("Your specific months were not detected as top episodes under your drawdown definition. Raise history, lower episode threshold, or switch target.")

st.caption("Reminder: Yahoo can miss VIX9D/VIX3M/VVIX on some days. The score automatically renormalizes around what is available.")
