# reflexivity_monitor.py
# Streamlit page: Reflexivity Monitor
# FRED pulls: try fredapi, fallback to pandas_datareader (no key needed).
# Weekly computation only. Visual polish: regime bands, action badges, sorted factor bars, clearer hover.

import os
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
    FRED = Fred(api_key=fred_key) if fred_key else Fred()
    FRED_AVAILABLE = True
except Exception:
    FRED_AVAILABLE = False


# ---------------- Helpers ----------------
def _zscore(x: pd.Series, clip=3.0):
    z = (x - x.mean()) / (x.std(ddof=0) + 1e-9)
    return z.clip(-clip, clip) if clip is not None else z


@st.cache_data(show_spinner=False)
def load_prices(tickers, start="2010-01-01", freq="W-FRI"):
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.ffill()
    if freq:
        px = px.resample(freq).last()
    return px.dropna(how="all")


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_fred_series(series_map: dict, start="2010-01-01", freq="W-FRI"):
    frames = []
    # Try fredapi first
    if FRED_AVAILABLE:
        try:
            for name, sid in series_map.items():
                s = FRED.get_series(sid)
                s.name = name
                frames.append(s.to_frame())
        except Exception:
            frames = []
    # Fallback to pandas_datareader (no key)
    if not frames:
        for name, sid in series_map.items():
            try:
                s = pdr.DataReader(sid, "fred", start)[sid]
                s.name = name
                frames.append(s.to_frame())
            except Exception:
                pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1)
    df = df.loc[pd.to_datetime(start):]
    if freq:
        df = df.resample(freq).ffill()
    else:
        daily = pd.date_range(df.index.min(), pd.Timestamp.today().normalize(), freq="B")
        df = df.reindex(daily).ffill()
    return df


# ---------------- Macro builders ----------------
def net_liquidity(df):
    need = {"WALCL", "RRPONTSYD", "WTREGEN"}
    if not need.issubset(df.columns):
        return pd.Series(dtype=float)
    nl = df["WALCL"] - df["RRPONTSYD"] - df["WTREGEN"]
    nl.name = "NetLiquidity"
    return nl


def real_yield(df):
    if {"DGS10", "T10YIE"}.issubset(df.columns):
        ry = df["DGS10"] - df["T10YIE"]
        ry.name = "Real10y"
        return ry
    return pd.Series(dtype=float)


def growth_yoy(s: pd.Series, months=12, freq_weeks=True):
    if s.empty:
        return s
    if freq_weeks:
        return s.pct_change(int(months * 4.33))  # ~weeks per month
    return s.pct_change(int(months * 21))


# ---------------- Rolling lead/lag utilities ----------------
def rolling_lag_corr(asset_ret: pd.Series, fund_delta: pd.Series, window=52, lag_steps=4):
    dF_fwd = fund_delta.shift(-lag_steps)
    aligned = pd.concat([asset_ret, dF_fwd], axis=1).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)
    return aligned[asset_ret.name].rolling(window).corr(aligned[dF_fwd.name])


def rolling_lead_corr(asset_ret: pd.Series, fund_delta: pd.Series, window=52, lead_steps=4):
    r_fwd = asset_ret.shift(-lead_steps)
    aligned = pd.concat([fund_delta, r_fwd], axis=1).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)
    return aligned[fund_delta.name].rolling(window).corr(aligned[r_fwd.name])


# ---------------- Main score ----------------
def make_reflexivity_score(asset_px: pd.Series, fundamentals: pd.DataFrame, params: dict):
    """
    Compute reflexivity score from two-way rolling correlations.
    Lag-matched deltas (diff(lag)) avoid zero-variance windows for monthly series.
    """
    r = asset_px.pct_change().rename("ret")
    scores, meta = [], []
    window = params.get("window", 52)
    lags = params.get("lags", [4, 12])

    for fcol in fundamentals.columns:
        f = fundamentals[fcol].rename(fcol)
        for lag in lags:
            dF = f.diff(int(lag)).rename(f"d_{fcol}")
            rc = rolling_lag_corr(r, dF, window=window, lag_steps=int(lag))
            fc = rolling_lead_corr(r, dF, window=window, lead_steps=int(lag))
            spread = rc - fc
            spread.name = f"spread_{fcol}_{lag}"
            scores.append(spread)
            meta.append({"factor": fcol, "lag": lag})

    if not scores:
        return pd.Series(dtype=float), pd.DataFrame()

    S = pd.concat(scores, axis=1).dropna(axis=1, how="all")
    if S.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    valid_meta = [m for m in meta if f"spread_{m['factor']}_{m['lag']}" in S.columns]
    w = np.array([params.get("factor_weights", {}).get(m["factor"], 1.0) for m in valid_meta], dtype=float)
    w = w / (w.sum() if w.sum() != 0 else 1.0)

    Z = S.apply(_zscore)
    composite = (Z @ w).rename("ReflexivityIntensity")
    return composite, S


# ---------------- Small utilities for context panel ----------------
def _percentile_rank(series: pd.Series, value: float) -> float:
    if series.empty or np.isnan(value):
        return np.nan
    return float(series.rank(pct=True).iloc[-1] * 100.0)

def _streak_lengths(mask: pd.Series) -> list:
    # mask is boolean series for a given regime
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
st.caption("Quantify when prices are feeding back into fundamentals. Output is a real-time reflexivity intensity index for position sizing and reversal timing.")

# Sidebar
st.sidebar.header("About this tool")
st.sidebar.markdown(
    """
**What it is**: A meta-signal that measures when **prices drive fundamentals** (reflexive loop) vs when fundamentals drive prices.

**How it works**: Rolling correlations both ways:
1) **Price → future changes** in macro factors (Net Liquidity, Real 10y, HY OAS, M2 YoY, INDPRO YoY)  
2) **Factor changes → future price**

The weighted, standardized spread becomes a **0–100 gauge**.

**How to read it**
- **> 65**: Self-reinforcing loop; trend-friendly.
- **35–65**: Neutral.
- **< 35**: Self-correcting; mean-reversion bias.

We compute at **weekly frequency** for stability and speed.
"""
)

st.sidebar.header("Settings")
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2005-01-01"))
window = st.sidebar.number_input("Rolling window (weeks)", 20, 156, 26, step=2)
sel_lags = st.sidebar.multiselect("Lead/Lag steps (weeks)", [2, 4, 8, 12, 26], [4, 8, 12])

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
custom_series = st.sidebar.text_area("Extra FRED series (name=ID per line)")
if custom_series.strip():
    for line in custom_series.splitlines():
        if "=" in line:
            name, sid = [x.strip() for x in line.split("=", 1)]
            series_map[name] = sid

# Assets
assets_default = ["SPY", "QQQ", "IWM", "HYG", "XLF", "XHB", "XLE", "GLD", "BTC-USD"]
assets = st.sidebar.text_input("Assets (comma separated)", ",".join(assets_default)).replace(" ", "").split(",")

# Data
with st.spinner("Loading prices"):
    px = load_prices(assets, start=str(start_date), freq="W-FRI")
with st.spinner("Loading FRED series"):
    fred_df = load_fred_series(series_map, start=str(start_date), freq="W-FRI")

# Fundamentals
fund = pd.DataFrame(index=px.index)
if not fred_df.empty:
    fred_df = fred_df.reindex(px.index).ffill()
    nl = net_liquidity(fred_df)
    if not nl.empty:
        fund["NetLiquidity"] = nl
    ry = real_yield(fred_df)
    if not ry.empty:
        fund["Real10y"] = ry
    if "HY_OAS" in fred_df:
        fund["HY_OAS"] = fred_df["HY_OAS"]
    if "M2SL" in fred_df:
        fund["M2_YoY"] = growth_yoy(fred_df["M2SL"], freq_weeks=True)
    if "INDPRO" in fred_df:
        fund["INDPRO_YoY"] = growth_yoy(fred_df["INDPRO"], freq_weeks=True)
fund = fund.ffill()

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

        ri_z = _zscore(ri)
        ri_idx = (50 + 15 * ri_z).clip(0, 100)

        latest_series = ri_idx.dropna()
        latest_val = float(latest_series.iloc[-1]) if not latest_series.empty else np.nan
        delta4 = latest_series.iloc[-1] - latest_series.iloc[-5] if len(latest_series) > 5 else np.nan
        regime = "Self-reinforcing" if (not np.isnan(latest_val) and latest_val >= 65) else (
            "Self-correcting" if (not np.isnan(latest_val) and latest_val <= 35) else "Neutral"
        )

        # ---------------- Summary cards (no ellipsis) ----------------
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2.2])

        gauge_text = "NA" if np.isnan(latest_val) else f"{latest_val:.1f}"
        delta_text = None if np.isnan(delta4) else f"{delta4:+.1f} / 4w"
        c1.metric(f"{tkr} gauge", gauge_text, delta_text)

        c2.metric("Regime", regime)

        # Time-in-regime (consecutive weeks in current extreme)
        tir = np.nan
        if not latest_series.empty:
            last_reg = (latest_series >= 65).astype(int) - (latest_series <= 35).astype(int)  # 1/0/-1
            cur = 1 if (not np.isnan(latest_val) and latest_val >= 65) else (-1 if (not np.isnan(latest_val) and latest_val <= 35) else 0)
            if cur != 0:
                cnt = 0
                for v in last_reg.iloc[::-1]:
                    if v == cur:
                        cnt += 1
                    else:
                        break
                tir = cnt
        c3.metric("Time in regime", "NA" if np.isnan(tir) else f"{int(tir)}w")

        # ---------------- Regime context (replaces Suggested action) ----------------
        # Percentile rank of latest gauge vs history
        pct_rank = _percentile_rank(latest_series, latest_val) if not np.isnan(latest_val) else np.nan

        # Historical median streak length for this regime
        if regime == "Self-reinforcing":
            mask = latest_series >= 65
            lengths = _streak_lengths(mask)
            median_streak = int(np.median(lengths)) if len(lengths) else np.nan
        elif regime == "Self-correcting":
            mask = latest_series <= 35
            lengths = _streak_lengths(mask)
            median_streak = int(np.median(lengths)) if len(lengths) else np.nan
        else:
            median_streak = np.nan

        c4.markdown("**Regime context**")
        ctx_lines = []
        ctx_lines.append(
            f"<div style='font-size:22px; line-height:1.25; font-weight:600'>Percentile: "
            f"{'NA' if np.isnan(pct_rank) else f'{pct_rank:.0f}th'}</div>"
        )
        if not np.isnan(tir) and not np.isnan(median_streak):
            ctx_lines.append(
                f"<div style='font-size:18px; line-height:1.25'>Streak: {int(tir)}w vs {int(median_streak)}w median</div>"
            )
        else:
            ctx_lines.append("<div style='font-size:18px; line-height:1.25'>Streak: NA</div>")
        c4.markdown("<br/>".join(ctx_lines), unsafe_allow_html=True)

        # ---------------- Chart: gauge + regime strip ----------------
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2]
        )

        # Regime bands (row 1)
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

        # Regime strip (row 2)
        strip_vals = ri_idx.copy()
        if not strip_vals.empty:
            strip_vals[:] = np.where(ri_idx <= 35, 0, np.where(ri_idx >= 65, 2, 1))
            fig.add_trace(
                go.Heatmap(
                    x=strip_vals.index,
                    z=strip_vals.values.reshape(1, -1),
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

        # ---------------- Factor panel ----------------
        with st.expander("Factor contributions (latest)"):
            if not details.empty:
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

        # ---------------- Download ----------------
        with st.expander("Download data"):
            out = pd.concat([s.rename("price"), fund], axis=1)
            out["ReflexivityIntensity"] = ri
            st.download_button("Download CSV", out.to_csv().encode(), file_name=f"{tkr}_reflexivity_monitor.csv")

st.divider()

st.subheader("Notes")
st.markdown(
    """
- Computation is **weekly** for stability and speed.
- FRED fetching tries fredapi first, then falls back to pandas_datareader with no key.
- Fundamentals are forward-filled and aligned to avoid lookahead.
- The score compares price→future fundamentals vs fundamentals→future price. Positive spread implies a reflexive loop.
"""
)
