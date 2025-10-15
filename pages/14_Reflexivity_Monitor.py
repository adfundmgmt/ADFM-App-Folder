# reflexivity_monitor.py
# Streamlit page: Reflexivity Monitor
# Hardened data access, stable caching, better edge handling, pastel UI, clear regimes.

import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pandas_datareader import data as pdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------- Optional fredapi ----------------
FRED_AVAILABLE, FRED = False, None
try:
    from fredapi import Fred
    fred_key = None
    try:
        fred_key = st.secrets.get("FRED_API_KEY", None)
    except Exception:
        fred_key = None
    fred_key = fred_key or os.getenv("FRED_API_KEY")
    # If no key, fredapi still works for public series but may throttle
    FRED = Fred(api_key=fred_key) if fred_key else Fred()
    FRED_AVAILABLE = True
except Exception:
    FRED_AVAILABLE = False


# ---------------- Helpers ----------------
def _zscore(x: pd.Series, clip: float = 3.0) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        z = pd.Series(0.0, index=x.index)
    else:
        z = (x - mu) / (sd + 1e-12)
    return z.clip(-clip, clip) if clip is not None else z


def _as_tuple_or_str(seq: List[str]) -> Tuple[str, ...]:
    # Stable cache key
    return tuple(str(s).strip() for s in seq)


def _retry(fn, tries=3, sleep=0.8):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(sleep)
    raise last


@st.cache_data(show_spinner=False)
def load_prices(tickers: Tuple[str, ...], start="2010-01-01", freq="W-FRI") -> pd.DataFrame:
    def _fetch():
        return yf.download(list(tickers), start=start, auto_adjust=True, progress=False, group_by='column')

    raw = _retry(_fetch, tries=3, sleep=0.8)
    if raw.empty:
        return pd.DataFrame(index=pd.date_range(start, periods=1, freq="D"))

    # yfinance returns MultiIndex if many tickers, Series if one
    if isinstance(raw, pd.DataFrame) and "Close" in raw.columns:
        px = raw["Close"].copy()
    else:
        # Fallback shape handling
        if isinstance(raw, pd.Series):
            px = raw.to_frame(name=tickers[0])
        else:
            # Try to locate Close level by name
            close_cols = [c for c in raw.columns if isinstance(c, tuple) and c[0] == "Close"]
            if close_cols:
                px = raw[close_cols]
                # Flatten columns to tickers only
                px.columns = [c[1] for c in close_cols]
            else:
                # Last resort, treat as single close column
                px = raw.copy()
                if isinstance(px, pd.Series):
                    px = px.to_frame(name=tickers[0])

    if isinstance(px, pd.Series):
        px = px.to_frame()

    px = px.sort_index().ffill()
    if freq:
        px = px.resample(freq).last()

    # Drop columns that are entirely NaN
    return px.dropna(how="all")


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_fred_series(series_map: Dict[str, str], start="2010-01-01", freq="W-FRI") -> pd.DataFrame:
    frames = []

    # Try fredapi first, per-series guarded
    if FRED_AVAILABLE:
        for name, sid in series_map.items():
            try:
                # fredapi allows start, but not all backends respect it, so slice after
                s = FRED.get_series(sid)
                if s is None or len(s) == 0:
                    continue
                s = s.to_frame(name=name)
                s.index = pd.to_datetime(s.index)
                frames.append(s)
            except Exception:
                continue

    # Fallback to pandas_datareader for any missing
    have = set([f.columns[0] for f in frames])
    for name, sid in series_map.items():
        if name in have:
            continue
        try:
            s = pdr.DataReader(sid, "fred", start)
            if s.empty:
                continue
            s = s.rename(columns={sid: name})
            frames.append(s)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1).sort_index()
    df = df.loc[pd.to_datetime(start):]

    # Build a uniform weekly index to align with prices tightly
    if freq:
        weekly = pd.date_range(df.index.min(), pd.Timestamp.today().normalize(), freq=freq)
        df = df.reindex(weekly).ffill()
    else:
        daily = pd.date_range(df.index.min(), pd.Timestamp.today().normalize(), freq="B")
        df = df.reindex(daily).ffill()

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


# ---------------- Macro builders ----------------
def net_liquidity(df: pd.DataFrame) -> pd.Series:
    need = {"WALCL", "RRPONTSYD", "WTREGEN"}
    if not need.issubset(df.columns):
        return pd.Series(dtype=float)
    nl = df["WALCL"] - df["RRPONTSYD"] - df["WTREGEN"]
    nl.name = "NetLiquidity"
    return nl


def real_yield(df: pd.DataFrame) -> pd.Series:
    if {"DGS10", "T10YIE"}.issubset(df.columns):
        ry = df["DGS10"] - df["T10YIE"]
        ry.name = "Real10y"
        return ry
    return pd.Series(dtype=float)


def growth_yoy(s: pd.Series, months=12, freq_weeks=True) -> pd.Series:
    if s.empty:
        return s
    s = s.astype(float)
    if freq_weeks:
        # 52 weeks approximates YoY for weekly series
        return s.pct_change(52).replace([np.inf, -np.inf], np.nan)
    # 252 trading days approximates YoY for daily
    return s.pct_change(252).replace([np.inf, -np.inf], np.nan)


# ---------------- Rolling lead/lag utilities ----------------
def rolling_lag_corr(asset_ret: pd.Series, fund_delta: pd.Series, window=52, lag_steps=4) -> pd.Series:
    dF_fwd = fund_delta.shift(-int(lag_steps))
    aligned = pd.concat([asset_ret.rename("r"), dF_fwd.rename("dF_fwd")], axis=1).dropna()
    if len(aligned) < int(window):
        return pd.Series(dtype=float)
    return aligned["r"].rolling(int(window)).corr(aligned["dF_fwd"])


def rolling_lead_corr(asset_ret: pd.Series, fund_delta: pd.Series, window=52, lead_steps=4) -> pd.Series:
    r_fwd = asset_ret.shift(-int(lead_steps))
    aligned = pd.concat([fund_delta.rename("dF"), r_fwd.rename("r_fwd")], axis=1).dropna()
    if len(aligned) < int(window):
        return pd.Series(dtype=float)
    return aligned["dF"].rolling(int(window)).corr(aligned["r_fwd"])


# ---------------- Main score ----------------
def make_reflexivity_score(asset_px: pd.Series, fundamentals: pd.DataFrame, params: dict):
    """
    Composite uses standardized spreads of Corr(price -> future dFactor) minus Corr(dFactor -> future price)
    across lags, weighted by factor weights, with windowed rolling correlations at weekly frequency.
    """
    r = asset_px.pct_change().rename("ret").replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    scores, meta = [], []
    window = int(params.get("window", 52))
    lags = [int(x) for x in params.get("lags", [4, 12])]

    fdf = fundamentals.copy()
    fdf = fdf.loc[r.index].ffill()

    for fcol in fdf.columns:
        f = fdf[fcol].astype(float)
        if f.dropna().empty:
            continue
        for lag in lags:
            # lag-matched diffs to stabilize low-frequency series
            dF = f.diff(int(lag)).rename(f"d_{fcol}")
            rc = rolling_lag_corr(r, dF, window=window, lag_steps=lag)
            fc = rolling_lead_corr(r, dF, window=window, lead_steps=lag)

            spread = (rc - fc).rename(f"spread_{fcol}_{lag}")
            scores.append(spread)
            meta.append({"factor": fcol, "lag": lag})

    if not scores:
        return pd.Series(dtype=float), pd.DataFrame()

    S = pd.concat(scores, axis=1)
    S = S.loc[~S.isna().all(axis=1)]
    if S.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    # Build weights aligned to S.columns order
    valid_meta = [m for m in meta if f"spread_{m['factor']}_{m['lag']}" in S.columns]
    w = np.array([params.get("factor_weights", {}).get(m["factor"], 1.0) for m in valid_meta], dtype=float)
    if not np.isfinite(w).all() or w.sum() == 0:
        w = np.ones(len(valid_meta), dtype=float)
    w = w / w.sum()

    Z = S.apply(_zscore)
    composite = pd.Series(Z.values @ w, index=Z.index, name="ReflexivityIntensity")
    return composite, S


# ---------------- Percentile and streak utils ----------------
def _percentile_rank(series: pd.Series, value: float) -> float:
    if series.empty or not np.isfinite(value):
        return np.nan
    vals = np.array(series.dropna().values)
    vals.sort()
    # Percentile of the last observation within history
    k = np.searchsorted(vals, value, side="right")
    return 100.0 * k / len(vals)


def _streak_lengths(mask: pd.Series) -> List[int]:
    if mask.empty:
        return []
    lengths, run = [], 0
    for v in mask.astype(bool).values:
        if v:
            run += 1
        elif run > 0:
            lengths.append(run)
            run = 0
    if run > 0:
        lengths.append(run)
    return lengths


# ---------------- UI ----------------
st.title("Reflexivity Monitor")
st.caption("Quantify when prices are feeding back into fundamentals. Gauge is weekly, 0 to 100, for position sizing and reversal timing.")

# Sidebar
st.sidebar.header("About this tool")
st.sidebar.markdown(
    """
**What it is**: A meta-signal that measures when **prices drive fundamentals** versus when fundamentals drive prices.

**How it works**: Rolling correlations both ways  
1) Price → future changes in macro factors  
2) Factor changes → future price

We standardize, weight, and take the spread to form a **0–100 gauge**.

**How to read it**  
- **> 65** trend supportive  
- **35–65** neutral  
- **< 35** mean reversion bias

Weekly frequency for stability and speed.
"""
)

st.sidebar.header("Settings")
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2005-01-01"))
window = st.sidebar.number_input("Rolling window, weeks", 20, 156, 26, step=2)
sel_lags = st.sidebar.multiselect("Lead/Lag, weeks", [2, 4, 8, 12, 26], [4, 8, 12])

st.sidebar.subheader("FRED series")
series_map = {
    "WALCL": "WALCL",
    "RRPONTSYD": "RRPONTSYD",
    "WTREGEN": "WTREGEN",
    "DGS10": "DGS10",
    "T10YIE": "T10YIE",
    "HY_OAS": "BAMLH0A0HYM2",
    "M2SL": "M2SL",
    "INDPRO": "INDPRO",
}
custom_series = st.sidebar.text_area("Extra FRED series, format name=ID per line")
if custom_series.strip():
    for line in custom_series.splitlines():
        if "=" in line:
            name, sid = [x.strip() for x in line.split("=", 1)]
            if name and sid:
                series_map[name] = sid

assets_default = ["SPY", "QQQ", "IWM", "HYG", "XLF", "XHB", "XLE", "GLD", "BTC-USD"]
assets_str = st.sidebar.text_input("Assets, comma separated", ",".join(assets_default))
assets = [a.strip() for a in assets_str.split(",") if a.strip()]

# Data
with st.spinner("Loading prices"):
    px = load_prices(_as_tuple_or_str(assets), start=str(start_date), freq="W-FRI")

with st.spinner("Loading FRED series"):
    fred_df = load_fred_series(series_map, start=str(start_date), freq="W-FRI")

# Fundamentals
fund = pd.DataFrame(index=px.index)
warn_msgs = []

if not fred_df.empty:
    fred_df = fred_df.reindex(px.index).ffill()

    nl = net_liquidity(fred_df)
    if not nl.empty:
        fund["NetLiquidity"] = nl
    else:
        warn_msgs.append("NetLiquidity inputs missing, skipped.")

    ry = real_yield(fred_df)
    if not ry.empty:
        fund["Real10y"] = ry
    else:
        warn_msgs.append("Real10y inputs missing, skipped.")

    if "HY_OAS" in fred_df.columns:
        fund["HY_OAS"] = fred_df["HY_OAS"]
    else:
        warn_msgs.append("HY_OAS missing, skipped.")

    if "M2SL" in fred_df.columns:
        fund["M2_YoY"] = growth_yoy(fred_df["M2SL"], freq_weeks=True)
    else:
        warn_msgs.append("M2SL missing, skipped.")

    if "INDPRO" in fred_df.columns:
        fund["INDPRO_YoY"] = growth_yoy(fred_df["INDPRO"], freq_weeks=True)
    else:
        warn_msgs.append("INDPRO missing, skipped.")
else:
    warn_msgs.append("All FRED series failed, fundamentals empty.")

fund = fund.ffill()

if warn_msgs:
    with st.expander("Data warnings"):
        for m in warn_msgs:
            st.write("• " + m)

base_weights = {"NetLiquidity": 1.25, "Real10y": 1.0, "HY_OAS": 1.25, "M2_YoY": 0.75, "INDPRO_YoY": 0.75}
params = {"lags": sel_lags, "window": int(window), "factor_weights": base_weights}

# ---------------- Main panel ----------------
st.subheader("Reflexivity intensity by asset")
valid_assets = [a for a in assets if a in px.columns]
tabs = st.tabs(valid_assets if valid_assets else ["No assets"])

for i, tkr in enumerate(valid_assets):
    with tabs[i]:
        s = px[tkr].dropna()
        if s.empty or fund.dropna(how="all").empty:
            st.info("Need both prices and fundamentals to compute the index.")
            continue

        ri, details = make_reflexivity_score(s, fund.dropna(how="all"), params)
        if ri.empty:
            st.info("Insufficient data for rolling correlations.")
            continue

        ri = ri.replace([np.inf, -np.inf], np.nan).dropna()
        if ri.empty:
            st.info("Gauge history is empty after cleaning.")
            continue

        ri_z = _zscore(ri)
        ri_idx = (50 + 15 * ri_z).clip(0, 100)

        latest_val = float(ri_idx.iloc[-1])
        delta4 = float(ri_idx.iloc[-1] - ri_idx.iloc[-5]) if len(ri_idx) > 5 else np.nan
        regime = "Self-reinforcing" if latest_val >= 65 else ("Self-correcting" if latest_val <= 35 else "Neutral")

        # Summary cards
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2.2])
        c1.metric(f"{tkr} gauge", f"{latest_val:.1f}", None if np.isnan(delta4) else f"{delta4:+.1f} / 4w")
        c2.metric("Regime", regime)

        # Time in current regime
        tir = np.nan
        last_reg = (ri_idx >= 65).astype(int) - (ri_idx <= 35).astype(int)  # 1, 0, -1
        cur = 1 if latest_val >= 65 else (-1 if latest_val <= 35 else 0)
        if cur != 0:
            cnt = 0
            for v in last_reg.iloc[::-1]:
                if v == cur:
                    cnt += 1
                else:
                    break
            tir = cnt
        c3.metric("Time in regime", "NA" if np.isnan(tir) else f"{int(tir)}w")

        # Regime context
        c4.markdown("**Regime context**")
        pct_rank = _percentile_rank(ri_idx, latest_val)
        # Historical median streak length
        if regime == "Self-reinforcing":
            mask = ri_idx >= 65
        elif regime == "Self-correcting":
            mask = ri_idx <= 35
        else:
            mask = pd.Series([], dtype=bool)

        lengths = _streak_lengths(mask) if len(mask) else []
        median_streak = int(np.median(lengths)) if lengths else np.nan

        ctx_lines = [
            f"<div style='font-size:22px; line-height:1.25; font-weight:600'>Percentile: {'NA' if np.isnan(pct_rank) else f'{pct_rank:.0f}th'}</div>",
            f"<div style='font-size:18px; line-height:1.25'>Streak: {'NA' if np.isnan(tir) or np.isnan(median_streak) else f'{int(tir)}w vs {int(median_streak)}w median'}</div>",
        ]
        c4.markdown("<br/>".join(ctx_lines), unsafe_allow_html=True)

        # Chart: gauge + regime strip
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2]
        )
        # Regime bands
        fig.add_hrect(y0=0, y1=35, line_width=0, fillcolor="#fde2e1", opacity=0.6, row=1, col=1)
        fig.add_hrect(y0=35, y1=65, line_width=0, fillcolor="#eef0f2", opacity=0.5, row=1, col=1)
        fig.add_hrect(y0=65, y1=100, line_width=0, fillcolor="#e5f5ea", opacity=0.6, row=1, col=1)
        fig.add_trace(go.Scatter(x=ri_idx.index, y=ri_idx, name="Gauge 0–100", mode="lines"), row=1, col=1)

        first_valid = ri_idx.first_valid_index()
        if (first_valid is not None) and (pd.to_datetime(first_valid) > pd.to_datetime(start_date)):
            fig.add_vrect(
                x0=pd.to_datetime(start_date),
                x1=pd.to_datetime(first_valid),
                fillcolor="lightgray",
                opacity=0.15,
                line_width=0,
                annotation_text="warm-up",
                annotation_position="top left",
                row=1, col=1,
            )

        # Regime strip
        strip_vals = np.where(ri_idx <= 35, 0, np.where(ri_idx >= 65, 2, 1))
        fig.add_trace(
            go.Heatmap(
                x=ri_idx.index,
                z=strip_vals.reshape(1, -1),
                colorscale=[[0.0, "#f8b4ad"], [0.5, "#d9dce1"], [1.0, "#a6e3b8"]],
                showscale=False,
            ),
            row=2, col=1,
        )

        # Axes/format
        fig.update_yaxes(range=[0, 100], title_text="Reflexivity gauge", row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=2, col=1)
        fig.update_xaxes(
            range=[pd.to_datetime(start_date), ri_idx.index.max()],
            dtick="M12",
            tickformat="%Y",
            hoverformat="%b %Y",
            title_text="Year",
            row=2, col=1,
        )
        fig.update_layout(height=600, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        # Factor panel
        with st.expander("Factor contributions, latest"):
            if details is not None and not details.empty:
                det_nonan = details.dropna(how="all")
                if det_nonan.empty:
                    st.info("No valid factor spreads yet for this asset and settings.")
                else:
                    last_row = det_nonan.iloc[-1].sort_values(ascending=False)
                    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in last_row.values]
                    fig2 = go.Figure(
                        data=[go.Bar(x=last_row.index, y=last_row.values, marker=dict(color=colors))]
                    )
                    fig2.update_layout(height=320, margin=dict(l=20, r=20, t=10, b=10))
                    st.plotly_chart(fig2, use_container_width=True)
                    st.dataframe(det_nonan.tail(5))
            else:
                st.info("No factor details available with current window.")

        # Download
        with st.expander("Download data"):
            out = pd.concat([s.rename("price"), fund], axis=1)
            out["ReflexivityIntensity"] = ri
            st.download_button("Download CSV", out.to_csv().encode(), file_name=f"{tkr}_reflexivity_monitor.csv")

st.divider()
st.subheader("Notes")
st.markdown(
    """
- Weekly computation for stability and speed.  
- FRED fetch tries fredapi per series, then pandas_datareader as fallback.  
- Fundamentals are forward-filled and aligned to avoid look-ahead.  
- Score compares price→future fundamentals vs fundamentals→future price. Positive spread implies a reflexive loop.
"""
)

st.caption("© 2025 AD Fund Management LP")
