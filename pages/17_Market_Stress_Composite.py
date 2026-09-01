from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from adfm_core.ui import PageHeader, inject_explorer_style, render_page_header

TITLE = "Market Stress Composite"
st.set_page_config(page_title=TITLE, layout="wide")
inject_explorer_style()
st.markdown("<style>.block-container { padding-top: 2.15rem; }</style>", unsafe_allow_html=True)
render_page_header(
    PageHeader(
        title=TITLE,
        description="Global fracture monitor: ex-U.S. risk-off pressure and cross-market dislocation versus U.S. equities.",
        eyebrow="ADFM Global Fracture Monitor",
    )
)

# ---------------- Universe ----------------
SPX = "^GSPC"
IXIC = "^IXIC"

FOREIGN_EQUITIES: Dict[str, str] = {
    "^N225": "Nikkei 225",
    "^GDAXI": "DAX",
    "^FTSE": "FTSE 100",
    "^FCHI": "CAC 40",
    "^STOXX50E": "Euro Stoxx 50",
    "^HSI": "Hang Seng",
    "^AXJO": "ASX 200",
}

CARRY_FX: Dict[str, str] = {
    "AUDJPY=X": "AUD/JPY",
    "NZDJPY=X": "NZD/JPY",
}

HAVEN_FX: Dict[str, str] = {
    "JPY=X": "USD/JPY",
    "CHF=X": "USD/CHF",
}

FOREIGN_BONDS: Dict[str, str] = {
    "IGLT.L": "UK Gilts ETF",
    "IEGA.AS": "Euro Govt Bonds ETF",
    "2510.T": "Japan Govt Bonds ETF",
    "IGB.AX": "Australia Govt Bonds ETF",
}

ALL_TICKERS = sorted(
    set([SPX, IXIC] + list(FOREIGN_EQUITIES) + list(CARRY_FX) + list(HAVEN_FX) + list(FOREIGN_BONDS))
)

# ---------------- Defaults ----------------
DEFAULT_YEARS = 5
DEFAULT_Z_WINDOW = 252 * 3
DEFAULT_SIGNAL_SMOOTH = 3
FORWARD_HORIZON = 20

with st.sidebar:
    st.header("Settings")

    lookback_years = st.selectbox(
        "Chart lookback",
        [1, 2, 3, 5, 10],
        index=3,
        format_func=lambda x: f"{x} year" if x == 1 else f"{x} years",
    )

    target_mode = st.radio(
        "U.S. overlay",
        ["Auto", "S&P 500", "Nasdaq Composite"],
        index=0,
        help="Auto selects the U.S. index with the larger absolute 21-day move and stronger relationship to the current foreign-risk signal.",
    )

    z_window_years = st.slider("Normalization window", 1, 5, 3, 1)
    smooth_days = st.slider("Signal smoothing", 1, 10, DEFAULT_SIGNAL_SMOOTH, 1)

    st.markdown("---")
    st.header("Signal construction")
    st.markdown(
        """
        **Global Risk-Off**
        - Foreign equity weakness and breadth deterioration.
        - Foreign equity underperformance versus U.S. equities.
        - AUD/JPY and NZD/JPY carry unwind.
        - Yen and Swiss franc haven strength.

        **Global Dislocation**
        - Abnormally large foreign bond moves in either direction.
        - Large FX shocks.
        - Large foreign equity shocks.
        - Cross-country return dispersion.

        The two scores are intentionally separate. A directional risk-off signal and a rates-led dislocation do not mean the same thing.
        """
    )


# ---------------- Data ----------------
@st.cache_data(ttl=900, show_spinner=False)
def load_prices(tickers: List[str], start: date) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    out = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            if (ticker, "Close") in raw.columns:
                out[ticker] = pd.to_numeric(raw[(ticker, "Close")], errors="coerce")
            elif (ticker, "Adj Close") in raw.columns:
                out[ticker] = pd.to_numeric(raw[(ticker, "Adj Close")], errors="coerce")
    else:
        col = "Close" if "Close" in raw.columns else "Adj Close" if "Adj Close" in raw.columns else None
        if col and tickers:
            out[tickers[0]] = pd.to_numeric(raw[col], errors="coerce")

    df = pd.DataFrame(out)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def robust_z(s: pd.Series, window: int, min_periods: int = 126) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    minp = min(min_periods, max(40, window // 4))
    mean = s.rolling(window, min_periods=minp).mean()
    std = s.rolling(window, min_periods=minp).std(ddof=0)
    z = (s - mean) / std.replace(0, np.nan)
    return z.clip(-5, 5)


def pct_return(px: pd.Series, days: int) -> pd.Series:
    return px.pct_change(days, fill_method=None)


def safe_mean(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    return frame.mean(axis=1, skipna=True)


def available_cols(px: pd.DataFrame, universe: Dict[str, str]) -> List[str]:
    return [t for t in universe if t in px.columns and px[t].notna().sum() >= 80]


def latest_valid(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def fmt_score(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x:+.2f}"


def fmt_pct(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x * 100:+.2f}%"


def regime_label(risk: float, dislocation: float) -> str:
    if pd.isna(risk) or pd.isna(dislocation):
        return "Insufficient data"
    if risk >= 1.5 and dislocation >= 1.5:
        return "Global fracture"
    if risk >= 1.0 and dislocation >= 0.5:
        return "Risk-off broadening"
    if dislocation >= 1.5 and risk < 1.0:
        return "Dislocation without broad risk-off"
    if risk <= -0.75 and dislocation < 1.0:
        return "Risk-on / calm"
    return "Mixed / neutral"


history_start = date.today() - timedelta(days=int(12 * 365.25))
px = load_prices(ALL_TICKERS, history_start)

if px.empty or SPX not in px.columns:
    st.error("Yahoo Finance did not return enough market data to build the Global Fracture Monitor.")
    st.stop()

# Align on SPX calendar while allowing foreign markets to carry through one U.S. session.
calendar = px[SPX].dropna().index
px = px.reindex(calendar).ffill(limit=2)

eq_cols = available_cols(px, FOREIGN_EQUITIES)
carry_cols = available_cols(px, CARRY_FX)
haven_cols = available_cols(px, HAVEN_FX)
bond_cols = available_cols(px, FOREIGN_BONDS)

if len(eq_cols) < 3 or len(carry_cols) < 1:
    st.warning("Some foreign-market series are unavailable today. Scores are automatically reweighted across available inputs.")

# ---------------- Signal engineering ----------------
z_window = int(252 * z_window_years)

# Foreign equity directional deterioration.
eq_r21 = pd.DataFrame({t: pct_return(px[t], 21) for t in eq_cols})
eq_r63 = pd.DataFrame({t: pct_return(px[t], 63) for t in eq_cols})
eq_weak_21 = safe_mean(pd.DataFrame({t: -robust_z(eq_r21[t], z_window) for t in eq_cols}))
eq_weak_63 = safe_mean(pd.DataFrame({t: -robust_z(eq_r63[t], z_window) for t in eq_cols}))

# Breadth: share of foreign markets with negative 21D returns and below 100D MA.
breadth_neg21 = (eq_r21 < 0).mean(axis=1) if not eq_r21.empty else pd.Series(index=calendar, dtype=float)
below_ma = pd.DataFrame(
    {t: (px[t] < px[t].rolling(100, min_periods=60).mean()).astype(float) for t in eq_cols}
)
breadth_ma = below_ma.mean(axis=1) if not below_ma.empty else pd.Series(index=calendar, dtype=float)
breadth_raw = 0.5 * breadth_neg21 + 0.5 * breadth_ma
breadth_z = robust_z(breadth_raw, z_window)

# Foreign equity underperformance versus SPX.
spx_r21 = pct_return(px[SPX], 21)
relative = pd.DataFrame({t: eq_r21[t] - spx_r21 for t in eq_cols})
relative_weak = safe_mean(pd.DataFrame({t: -robust_z(relative[t], z_window) for t in eq_cols}))

# Carry unwind: falling AUDJPY/NZDJPY is stress.
carry_5 = pd.DataFrame({t: -robust_z(pct_return(px[t], 5), z_window) for t in carry_cols})
carry_21 = pd.DataFrame({t: -robust_z(pct_return(px[t], 21), z_window) for t in carry_cols})
carry_stress = 0.4 * safe_mean(carry_5) + 0.6 * safe_mean(carry_21)

# Haven FX: falling USDJPY/USDCHF means JPY/CHF strength, treated as risk-off.
haven_5 = pd.DataFrame({t: -robust_z(pct_return(px[t], 5), z_window) for t in haven_cols})
haven_21 = pd.DataFrame({t: -robust_z(pct_return(px[t], 21), z_window) for t in haven_cols})
haven_stress = 0.4 * safe_mean(haven_5) + 0.6 * safe_mean(haven_21)

risk_components = pd.DataFrame(
    {
        "Foreign equity weakness": 0.30 * eq_weak_21 + 0.10 * eq_weak_63,
        "Foreign breadth": 0.20 * breadth_z,
        "Foreign vs U.S.": 0.15 * relative_weak,
        "Carry unwind": 0.15 * carry_stress,
        "Haven FX": 0.10 * haven_stress,
    }
)
risk_raw = risk_components.sum(axis=1, min_count=2)
risk_score = robust_z(risk_raw, z_window).rolling(smooth_days, min_periods=1).mean()

# Dislocation: magnitude matters, not direction.
eq_shock = safe_mean(
    pd.DataFrame({t: robust_z(pct_return(px[t], 5), z_window).abs() for t in eq_cols})
)
fx_all = carry_cols + haven_cols
fx_shock = safe_mean(
    pd.DataFrame({t: robust_z(pct_return(px[t], 5), z_window).abs() for t in fx_all})
)
bond_shock = safe_mean(
    pd.DataFrame({t: robust_z(pct_return(px[t], 5), z_window).abs() for t in bond_cols})
)

# Cross-country dispersion of 21D returns, normalized through time.
eq_dispersion = eq_r21.std(axis=1, skipna=True) if not eq_r21.empty else pd.Series(index=calendar, dtype=float)
dispersion_z = robust_z(eq_dispersion, z_window)

dislocation_components = pd.DataFrame(
    {
        "Foreign bond shock": 0.35 * bond_shock,
        "FX shock": 0.25 * fx_shock,
        "Foreign equity shock": 0.25 * eq_shock,
        "Cross-country dispersion": 0.15 * dispersion_z,
    }
)
dislocation_raw = dislocation_components.sum(axis=1, min_count=2)
dislocation_score = robust_z(dislocation_raw, z_window).rolling(smooth_days, min_periods=1).mean()

# ---------------- Auto target ----------------
def corr_with_future_drawdown(signal: pd.Series, target_px: pd.Series, horizon: int = 20) -> float:
    future_min = target_px.shift(-1).rolling(horizon, min_periods=max(5, horizon // 3)).min().shift(-(horizon - 1))
    fwd_dd = future_min / target_px - 1.0
    aligned = pd.concat([signal, fwd_dd], axis=1).dropna()
    if len(aligned) < 100:
        return np.nan
    return float(aligned.iloc[:, 0].corr(-aligned.iloc[:, 1]))


def choose_target() -> tuple[str, str]:
    if target_mode == "S&P 500":
        return SPX, "S&P 500"
    if target_mode == "Nasdaq Composite":
        return IXIC if IXIC in px.columns else SPX, "Nasdaq Composite" if IXIC in px.columns else "S&P 500"

    candidates = [(SPX, "S&P 500")]
    if IXIC in px.columns:
        candidates.append((IXIC, "Nasdaq Composite"))

    best = candidates[0]
    best_metric = -np.inf
    for ticker, label in candidates:
        relationship = corr_with_future_drawdown(risk_score, px[ticker], FORWARD_HORIZON)
        move = abs(latest_valid(pct_return(px[ticker], 21)))
        metric = (0 if pd.isna(relationship) else relationship) + (0 if pd.isna(move) else move)
        if metric > best_metric:
            best_metric = metric
            best = (ticker, label)
    return best


target_ticker, target_label = choose_target()
target_px = px[target_ticker].dropna()

risk_now = latest_valid(risk_score)
dislocation_now = latest_valid(dislocation_score)
risk_5d = risk_now - latest_valid(risk_score.shift(5))
dislocation_5d = dislocation_now - latest_valid(dislocation_score.shift(5))
regime = regime_label(risk_now, dislocation_now)

# ---------------- Top readout ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Global Risk-Off", fmt_score(risk_now), fmt_score(risk_5d))
c2.metric("Global Dislocation", fmt_score(dislocation_now), fmt_score(dislocation_5d))
c3.metric("Regime", regime)
c4.metric("U.S. overlay", target_label)

if risk_now >= 1.5 and dislocation_now >= 1.5:
    st.error("Foreign markets are showing both broad directional risk-off pressure and unusually large cross-market dislocation.")
elif risk_now >= 1.0:
    st.warning("Global risk-off pressure is broadening outside the U.S. tape.")
elif dislocation_now >= 1.5:
    st.warning("Cross-market dislocation is elevated even though directional risk-off breadth is not yet extreme.")
else:
    st.info("No broad global-fracture condition is currently confirmed.")

# ---------------- Main interactive overlay ----------------
cutoff = pd.Timestamp(date.today() - timedelta(days=int(lookback_years * 365.25)))
plot_idx = target_px.index[target_px.index >= cutoff]

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(
        x=plot_idx,
        y=target_px.reindex(plot_idx),
        name=target_label,
        mode="lines",
        line=dict(width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>" + target_label + ": %{y:,.2f}<extra></extra>",
    ),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(
        x=plot_idx,
        y=risk_score.reindex(plot_idx),
        name="Global Risk-Off",
        mode="lines",
        line=dict(width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>Risk-Off: %{y:+.2f}<extra></extra>",
    ),
    secondary_y=True,
)
fig.add_trace(
    go.Scatter(
        x=plot_idx,
        y=dislocation_score.reindex(plot_idx),
        name="Global Dislocation",
        mode="lines",
        line=dict(width=2, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>Dislocation: %{y:+.2f}<extra></extra>",
    ),
    secondary_y=True,
)

for level in [1.0, 2.0]:
    fig.add_hline(y=level, line_dash="dash", line_width=1, opacity=0.35, secondary_y=True)

fig.update_layout(
    height=610,
    margin=dict(l=25, r=25, t=45, b=25),
    hovermode="x unified",
    legend=dict(orientation="h", y=1.05, x=0),
    xaxis_title=None,
)
fig.update_yaxes(title_text=target_label, secondary_y=False)
fig.update_yaxes(title_text="Signal Z-score", secondary_y=True)
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# ---------------- Interactive drill-down ----------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Signal drivers", "Global map of stress", "Forward drawdowns", "Data health"]
)

with tab1:
    left, right = st.columns(2)

    with left:
        st.subheader("Risk-Off drivers")
        latest_risk_parts = risk_components.iloc[-1].dropna().sort_values(ascending=False)
        risk_driver_fig = go.Figure(
            go.Bar(
                x=latest_risk_parts.values,
                y=latest_risk_parts.index,
                orientation="h",
                text=[f"{x:+.2f}" for x in latest_risk_parts.values],
                textposition="auto",
            )
        )
        risk_driver_fig.update_layout(height=330, margin=dict(l=10, r=10, t=15, b=20), xaxis_title="Weighted contribution")
        st.plotly_chart(risk_driver_fig, use_container_width=True, config={"displaylogo": False})

    with right:
        st.subheader("Dislocation drivers")
        latest_dis_parts = dislocation_components.iloc[-1].dropna().sort_values(ascending=False)
        dis_driver_fig = go.Figure(
            go.Bar(
                x=latest_dis_parts.values,
                y=latest_dis_parts.index,
                orientation="h",
                text=[f"{x:+.2f}" for x in latest_dis_parts.values],
                textposition="auto",
            )
        )
        dis_driver_fig.update_layout(height=330, margin=dict(l=10, r=10, t=15, b=20), xaxis_title="Weighted contribution")
        st.plotly_chart(dis_driver_fig, use_container_width=True, config={"displaylogo": False})

    st.subheader("Latest market moves")
    rows = []
    for universe, bucket in [
        (FOREIGN_EQUITIES, "Foreign equities"),
        (CARRY_FX, "Carry FX"),
        (HAVEN_FX, "Haven FX"),
        (FOREIGN_BONDS, "Foreign bonds"),
    ]:
        for ticker, name in universe.items():
            if ticker not in px.columns:
                continue
            rows.append(
                {
                    "Bucket": bucket,
                    "Market": name,
                    "Ticker": ticker,
                    "5D": latest_valid(pct_return(px[ticker], 5)),
                    "21D": latest_valid(pct_return(px[ticker], 21)),
                    "63D": latest_valid(pct_return(px[ticker], 63)),
                }
            )
    moves = pd.DataFrame(rows)
    if not moves.empty:
        styler = moves.style.format({"5D": "{:+.2%}", "21D": "{:+.2%}", "63D": "{:+.2%}"})
        st.dataframe(styler, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Cross-market stress matrix")
    matrix_rows = []
    for ticker in eq_cols:
        matrix_rows.append(
            {
                "Market": FOREIGN_EQUITIES[ticker],
                "Type": "Equity",
                "5D shock Z": latest_valid(robust_z(pct_return(px[ticker], 5), z_window)),
                "21D direction Z": latest_valid(robust_z(pct_return(px[ticker], 21), z_window)),
            }
        )
    for ticker in carry_cols:
        matrix_rows.append(
            {
                "Market": CARRY_FX[ticker],
                "Type": "Carry FX",
                "5D shock Z": latest_valid(robust_z(pct_return(px[ticker], 5), z_window)),
                "21D direction Z": latest_valid(-robust_z(pct_return(px[ticker], 21), z_window)),
            }
        )
    for ticker in haven_cols:
        matrix_rows.append(
            {
                "Market": HAVEN_FX[ticker],
                "Type": "Haven FX",
                "5D shock Z": latest_valid(robust_z(pct_return(px[ticker], 5), z_window)),
                "21D direction Z": latest_valid(-robust_z(pct_return(px[ticker], 21), z_window)),
            }
        )
    for ticker in bond_cols:
        matrix_rows.append(
            {
                "Market": FOREIGN_BONDS[ticker],
                "Type": "Bond",
                "5D shock Z": abs(latest_valid(robust_z(pct_return(px[ticker], 5), z_window))),
                "21D direction Z": np.nan,
            }
        )

    stress_matrix = pd.DataFrame(matrix_rows)
    if not stress_matrix.empty:
        st.dataframe(
            stress_matrix.style.format({"5D shock Z": "{:+.2f}", "21D direction Z": "{:+.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

with tab3:
    st.subheader(f"Signal vs subsequent {FORWARD_HORIZON}D {target_label} drawdown")

    future_min = target_px.shift(-1).rolling(FORWARD_HORIZON, min_periods=5).min().shift(-(FORWARD_HORIZON - 1))
    fwd_dd = future_min / target_px - 1.0
    test = pd.DataFrame(
        {
            "Risk-Off": risk_score,
            "Dislocation": dislocation_score,
            "Forward drawdown": fwd_dd,
        }
    ).dropna()

    threshold = st.slider("Historical signal threshold", 0.5, 3.0, 1.5, 0.25)
    signal_type = st.radio("Backtest signal", ["Risk-Off", "Dislocation", "Both"], horizontal=True)

    if signal_type == "Both":
        events = test[(test["Risk-Off"] >= threshold) & (test["Dislocation"] >= threshold)].copy()
        x_for_scatter = events[["Risk-Off", "Dislocation"]].mean(axis=1)
        x_label = "Average of both scores"
    else:
        events = test[test[signal_type] >= threshold].copy()
        x_for_scatter = events[signal_type]
        x_label = signal_type

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Signal days", f"{len(events):,}")
    b2.metric("Avg forward drawdown", fmt_pct(events["Forward drawdown"].mean() if len(events) else np.nan))
    b3.metric("Median forward drawdown", fmt_pct(events["Forward drawdown"].median() if len(events) else np.nan))
    b4.metric(
        "Hit rate: >5% drawdown",
        "NA" if not len(events) else f"{(events['Forward drawdown'] <= -0.05).mean() * 100:.1f}%",
    )

    if len(events):
        scatter = go.Figure(
            go.Scatter(
                x=x_for_scatter,
                y=events["Forward drawdown"] * 100,
                mode="markers",
                customdata=events.index.strftime("%Y-%m-%d"),
                hovertemplate="%{customdata}<br>Signal: %{x:.2f}<br>Forward drawdown: %{y:.2f}%<extra></extra>",
            )
        )
        scatter.update_layout(
            height=430,
            margin=dict(l=25, r=25, t=25, b=30),
            xaxis_title=x_label,
            yaxis_title=f"Worst subsequent {FORWARD_HORIZON}D close (%)",
        )
        st.plotly_chart(scatter, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("No historical observations meet the selected threshold.")

with tab4:
    health_rows = []
    for ticker in ALL_TICKERS:
        s = px[ticker].dropna() if ticker in px.columns else pd.Series(dtype=float)
        health_rows.append(
            {
                "Ticker": ticker,
                "Observations": len(s),
                "First": s.index.min().date().isoformat() if len(s) else "",
                "Last": s.index.max().date().isoformat() if len(s) else "",
                "Status": "OK" if len(s) >= 80 else "Missing / thin",
            }
        )
    st.dataframe(pd.DataFrame(health_rows), use_container_width=True, hide_index=True)

st.caption(
    "Live inputs are Yahoo Finance market proxies. Foreign bond ETFs do not provide a valid 1987/2000 history, so this page is a live and modern-history monitor rather than a synthetic long-history backtest."
)
