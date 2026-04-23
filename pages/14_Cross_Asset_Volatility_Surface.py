import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Cross-Asset Volatility Monitor", layout="wide")

TEXT = "#1f2937"
SUBTLE = "#6b7280"
BORDER = "#e5e7eb"
BG = "#ffffff"
CARD_BG = "#fafafa"

SLEEVE_COLORS = {
    "Equity": "#A8DADC",
    "Rates": "#BDE0FE",
    "FX": "#CDB4DB",
    "Commodity": "#90BE6D",
    "Relative Value": "#F4A261",
}

Z_DIVERGING = "RdYlGn_r"
PCT_SEQUENTIAL = "YlOrRd"

IMPLIED_SPECS = [
    {"ticker": "^VIX", "label": "Equity | VIX", "sleeve": "Equity"},
    {"ticker": "^VIX3M", "label": "Equity | VIX3M", "sleeve": "Equity"},
    {"ticker": "^VVIX", "label": "Equity | VVIX", "sleeve": "Equity"},
    {"ticker": "^MOVE", "label": "Rates | MOVE", "sleeve": "Rates"},
    {"ticker": "^GVZ", "label": "Commodity | GVZ", "sleeve": "Commodity"},
    {"ticker": "^OVX", "label": "Commodity | OVX", "sleeve": "Commodity"},
]

UNDERLYING_SPECS = [
    {"ticker": "^GSPC", "label": "Equity | SPX", "sleeve": "Equity"},
    {"ticker": "TLT", "label": "Rates | TLT", "sleeve": "Rates"},
    {"ticker": "GLD", "label": "Commodity | GLD", "sleeve": "Commodity"},
    {"ticker": "CL=F", "label": "Commodity | WTI", "sleeve": "Commodity"},
    {"ticker": "DX-Y.NYB", "label": "FX | DXY", "sleeve": "FX"},
    {"ticker": "JPY=X", "label": "FX | USDJPY", "sleeve": "FX"},
]

SLEEVE_ORDER = ["Equity", "Rates", "FX", "Commodity", "Relative Value"]


@dataclass
class FeatureSpec:
    name: str
    sleeve: str
    kind: str
    horizon: str
    unit: str
    change_type: str
    series: pd.Series


def realized_vol_close_to_close(price: pd.Series, window: int, ann: int = 252) -> pd.Series:
    ret = price.pct_change()
    return ret.rolling(window).std() * np.sqrt(ann) * 100.0


def realized_vol_parkinson(high: pd.Series, low: pd.Series, window: int, ann: int = 252) -> pd.Series:
    hl = np.log(high / low)
    rs = (hl ** 2) / (4.0 * np.log(2.0))
    return np.sqrt(rs.rolling(window).mean() * ann) * 100.0


def realized_vol_gk(high: pd.Series, low: pd.Series, close: pd.Series, open_: pd.Series, window: int, ann: int = 252) -> pd.Series:
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    var = var.clip(lower=0)
    return np.sqrt(var.rolling(window).mean() * ann) * 100.0


def empirical_percentile_series(s: pd.Series, lookback: int) -> pd.Series:
    out = pd.Series(index=s.index, dtype=float)
    vals = s.astype(float)
    min_periods = max(20, min(lookback, 60))
    for i in range(len(vals)):
        start = max(0, i - lookback + 1)
        window = vals.iloc[start:i + 1].dropna()
        if len(window) < min_periods:
            continue
        out.iloc[i] = 100.0 * (window <= vals.iloc[i]).mean()
    return out


def rolling_zscore_series(s: pd.Series, lookback: int) -> pd.Series:
    min_periods = max(20, min(lookback, 60))
    mu = s.rolling(lookback, min_periods=min_periods).mean()
    sig = s.rolling(lookback, min_periods=min_periods).std(ddof=0)
    return (s - mu) / sig.replace(0, np.nan)


def normalize_spark_minmax(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    s = pd.Series(arr).dropna()
    if s.empty:
        return np.full(arr.shape, np.nan)
    mn, mx = s.min(), s.max()
    if np.isclose(mx, mn):
        out = np.zeros(arr.shape, dtype=float)
        out[np.isnan(arr)] = np.nan
        return out
    return (arr - mn) / (mx - mn)


def point_change_5d(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 6:
        return np.nan
    return float(s.iloc[-1] - s.iloc[-6])


def pct_change_5d(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 6 or np.isclose(s.iloc[-6], 0):
        return np.nan
    return float((s.iloc[-1] / s.iloc[-6] - 1.0) * 100.0)


def fmt_num(x: float, decimals: int = 2, suffix: str = "") -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{decimals}f}{suffix}"


def sleeve_color(sleeve: str) -> str:
    return SLEEVE_COLORS.get(sleeve, "#d1d5db")


def make_card(title: str, value: str, subtitle: str):
    st.markdown(
        f"""
        <div style="border:1px solid {BORDER}; border-radius:14px; padding:16px 16px 14px 16px; background:{CARD_BG};">
          <div style="font-size:12px; color:{SUBTLE}; font-weight:700; letter-spacing:0.2px; text-transform:uppercase;">{title}</div>
          <div style="font-size:24px; color:{TEXT}; font-weight:800; margin-top:6px;">{value}</div>
          <div style="font-size:12px; color:{SUBTLE}; margin-top:6px; line-height:1.35;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False, ttl=1800)
def load_ohlc_data(tickers, start="2010-01-01"):
    raw = yf.download(
        tickers=tickers,
        start=start,
        progress=False,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
    )
    out = {}
    if raw is None or len(raw) == 0:
        return out

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()
        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
        return df[keep].dropna(how="all")

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        for t in tickers:
            if t in lvl0:
                out[t] = _clean(raw[t])
    else:
        if len(tickers) == 1:
            out[tickers[0]] = _clean(raw)
    return out


def build_feature_specs(raw_data: dict, rv_windows: list[int], estimator: str, spread_anchor_window: int):
    features: list[FeatureSpec] = []
    missing: list[str] = []

    implied_name_map = {}
    for spec in IMPLIED_SPECS:
        t = spec["ticker"]
        if t not in raw_data or raw_data[t].empty or "Close" not in raw_data[t]:
            missing.append(t)
            continue
        s = raw_data[t]["Close"].dropna().astype(float)
        if s.empty:
            missing.append(t)
            continue
        implied_name_map[t] = spec["label"]
        features.append(
            FeatureSpec(
                name=spec["label"],
                sleeve=spec["sleeve"],
                kind="Implied",
                horizon="Index",
                unit="vol pts",
                change_type="pts",
                series=s,
            )
        )

    rv_map = {}
    for spec in UNDERLYING_SPECS:
        t = spec["ticker"]
        if t not in raw_data or raw_data[t].empty:
            missing.append(t)
            continue
        df = raw_data[t].copy()
        need_cols = {"Open", "High", "Low", "Close"}
        if not need_cols.issubset(set(df.columns)):
            missing.append(t)
            continue
        close = df["Close"].dropna()
        if close.empty:
            missing.append(t)
            continue

        for w in rv_windows:
            if estimator == "Close to close":
                rv = realized_vol_close_to_close(close, w)
            elif estimator == "Parkinson":
                rv = realized_vol_parkinson(df["High"], df["Low"], w)
            else:
                rv = realized_vol_gk(df["High"], df["Low"], df["Close"], df["Open"], w)
            rv = rv.dropna().astype(float)
            if rv.empty:
                continue
            name = f'{spec["label"]} | RV{w}'
            features.append(
                FeatureSpec(
                    name=name,
                    sleeve=spec["sleeve"],
                    kind="Realized",
                    horizon=f"RV{w}",
                    unit="%",
                    change_type="%",
                    series=rv,
                )
            )
            rv_map[(t, w)] = rv

    spread_specs = [
        ("^VIX", "^GSPC", "Equity | VIX - SPX RV", "Relative Value", "Spread", f"VIX-RV{spread_anchor_window}", "vol pts"),
        ("^MOVE", "TLT", "Rates | MOVE - TLT RV", "Relative Value", "Spread", f"MOVE-RV{spread_anchor_window}", "vol pts"),
        ("^GVZ", "GLD", "Commodity | GVZ - GLD RV", "Relative Value", "Spread", f"GVZ-RV{spread_anchor_window}", "vol pts"),
        ("^OVX", "CL=F", "Commodity | OVX - WTI RV", "Relative Value", "Spread", f"OVX-RV{spread_anchor_window}", "vol pts"),
    ]

    for imp_ticker, und_ticker, name, sleeve, kind, horizon, unit in spread_specs:
        imp_name = implied_name_map.get(imp_ticker)
        imp = next((f.series for f in features if f.name == imp_name), None)
        rv = rv_map.get((und_ticker, spread_anchor_window))
        if imp is None or rv is None or rv.empty:
            continue
        idx = imp.index.union(rv.index)
        spread = imp.reindex(idx).sort_index().ffill() - rv.reindex(idx).sort_index().ffill()
        spread = spread.dropna().astype(float)
        if spread.empty:
            continue
        features.append(
            FeatureSpec(
                name=name,
                sleeve=sleeve,
                kind=kind,
                horizon=horizon,
                unit=unit,
                change_type="pts",
                series=spread,
            )
        )

    if "^VIX3M" in raw_data and "^VIX" in raw_data:
        vix3m = raw_data["^VIX3M"]["Close"].dropna().astype(float)
        vix = raw_data["^VIX"]["Close"].dropna().astype(float)
        idx = vix3m.index.union(vix.index)
        ts = vix3m.reindex(idx).sort_index().ffill() - vix.reindex(idx).sort_index().ffill()
        ts = ts.dropna()
        if not ts.empty:
            features.append(
                FeatureSpec(
                    name="Equity | VIX3M - VIX",
                    sleeve="Relative Value",
                    kind="Spread",
                    horizon="Term Structure",
                    unit="vol pts",
                    change_type="pts",
                    series=ts.astype(float),
                )
            )

    if "^VVIX" in raw_data and "^VIX" in raw_data:
        vvix = raw_data["^VVIX"]["Close"].dropna().astype(float)
        vix = raw_data["^VIX"]["Close"].dropna().astype(float)
        idx = vvix.index.union(vix.index)
        ratio = vvix.reindex(idx).sort_index().ffill() / vix.reindex(idx).sort_index().ffill()
        ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
        if not ratio.empty:
            features.append(
                FeatureSpec(
                    name="Equity | VVIX / VIX",
                    sleeve="Relative Value",
                    kind="Ratio",
                    horizon="Vol of Vol",
                    unit="x",
                    change_type="%",
                    series=ratio.astype(float),
                )
            )

    return features, sorted(set(missing))


def build_feature_frame(features: list[FeatureSpec], norm_obs: int):
    rows = []
    histories = {}

    for feat in features:
        s = feat.series.dropna().astype(float)
        if s.empty:
            continue

        z_hist = rolling_zscore_series(s, norm_obs)
        pct_hist = empirical_percentile_series(s, norm_obs)
        latest_z = z_hist.dropna().iloc[-1] if not z_hist.dropna().empty else np.nan
        latest_pct = pct_hist.dropna().iloc[-1] if not pct_hist.dropna().empty else np.nan
        z20_delta = np.nan
        z_valid = z_hist.dropna()
        if len(z_valid) >= 21:
            z20_delta = float(z_valid.iloc[-1] - z_valid.iloc[-21])

        chg_5d = point_change_5d(s) if feat.change_type == "pts" else pct_change_5d(s)

        pct_valid = pct_hist.dropna()
        state_shift = "Stable"
        if len(pct_valid) >= 2:
            prev, curr = pct_valid.iloc[-2], pct_valid.iloc[-1]
            if prev < 80 <= curr:
                state_shift = "Entered >80th pct"
            elif prev > 60 >= curr:
                state_shift = "Fell <60th pct"
            elif prev > 20 >= curr:
                state_shift = "Entered <20th pct"
            elif prev < 40 <= curr:
                state_shift = "Recovered >40th pct"

        spark = normalize_spark_minmax(s.tail(90).to_numpy())

        rows.append(
            {
                "Series": feat.name,
                "Sleeve": feat.sleeve,
                "Kind": feat.kind,
                "Horizon": feat.horizon,
                "Unit": feat.unit,
                "ChangeType": feat.change_type,
                "Level": float(s.iloc[-1]),
                "5D Change": chg_5d,
                "Z": float(latest_z) if pd.notna(latest_z) else np.nan,
                "Percentile": float(latest_pct) if pd.notna(latest_pct) else np.nan,
                "Z 20D Δ": z20_delta,
                "State Shift": state_shift,
                "Last Date": s.index[-1].date() if len(s.index) else None,
                "Spark": spark,
            }
        )
        histories[feat.name] = {
            "level": s,
            "z": z_hist,
            "pct": pct_hist,
            "unit": feat.unit,
            "kind": feat.kind,
            "horizon": feat.horizon,
            "sleeve": feat.sleeve,
        }

    df = pd.DataFrame(rows)
    if not df.empty:
        df["AbsZ"] = df["Z"].abs()
        df["Sleeve"] = pd.Categorical(df["Sleeve"], categories=SLEEVE_ORDER, ordered=True)
        df = df.sort_values(["Sleeve", "Kind", "Series"]).reset_index(drop=True)
    return df, histories


def composite_summary(frame: pd.DataFrame, sleeve: str) -> dict:
    sub = frame[frame["Sleeve"] == sleeve].copy()
    if sub.empty:
        return {"med_z": np.nan, "med_pct": np.nan, "hot": 0, "count": 0, "trend": np.nan}
    return {
        "med_z": float(sub["Z"].median(skipna=True)),
        "med_pct": float(sub["Percentile"].median(skipna=True)),
        "hot": int((sub["AbsZ"] >= 1.0).sum()),
        "count": int(len(sub)),
        "trend": float(sub["Z 20D Δ"].median(skipna=True)),
    }


def build_regime_summary(frame: pd.DataFrame) -> dict:
    eq = composite_summary(frame, "Equity")
    rt = composite_summary(frame, "Rates")
    fx = composite_summary(frame, "FX")
    cmd = composite_summary(frame, "Commodity")
    rv = composite_summary(frame, "Relative Value")

    def pos(x):
        return 0.0 if pd.isna(x) or x < 0 else float(x)

    breadth = float((frame["AbsZ"] >= 1.5).mean()) if len(frame) else 0.0
    stress = (
        0.45 * pos(eq["med_z"]) +
        0.35 * pos(rt["med_z"]) +
        0.20 * pos(cmd["med_z"]) +
        0.15 * pos(fx["med_z"]) +
        0.15 * breadth
    )
    if stress >= 1.9:
        regime = "Risk off"
    elif stress >= 1.05:
        regime = "Cautious"
    else:
        regime = "Neutral to constructive"

    dispersion = float(frame["Z"].dropna().std(ddof=0)) if frame["Z"].notna().any() else np.nan
    broadening = float(frame["Z 20D Δ"].dropna().median()) if frame["Z 20D Δ"].notna().any() else np.nan

    return {
        "regime": regime,
        "stress": stress,
        "dispersion": dispersion,
        "broadening": broadening,
        "hedge_richness": rv["med_z"],
        "Equity": eq,
        "Rates": rt,
        "FX": fx,
        "Commodity": cmd,
        "Relative Value": rv,
    }


def generate_commentary(summary: dict) -> str:
    def describe(med_z, trend):
        if pd.isna(med_z):
            return "unavailable"
        if med_z >= 1.5:
            base = "elevated"
        elif med_z >= 0.5:
            base = "firm"
        elif med_z <= -0.5:
            base = "contained"
        else:
            base = "middling"
        if pd.isna(trend):
            return base
        if trend >= 0.35:
            return f"{base} and rising"
        if trend <= -0.35:
            return f"{base} but fading"
        return base

    eq = summary["Equity"]
    rt = summary["Rates"]
    fx = summary["FX"]
    cmd = summary["Commodity"]
    rv = summary["Relative Value"]
    disp = summary["dispersion"]
    broad = summary["broadening"]

    disp_txt = "broadening" if pd.notna(disp) and disp >= 1.2 else "contained"
    broad_txt = "accelerating" if pd.notna(broad) and broad > 0.25 else "cooling" if pd.notna(broad) and broad < -0.25 else "stable"

    if pd.isna(rv["med_z"]):
        hedge_txt = "hedge richness is unavailable"
    elif rv["med_z"] >= 1.0:
        hedge_txt = "index and commodity hedges screen rich versus delivered movement"
    elif rv["med_z"] <= -1.0:
        hedge_txt = "hedges screen cheap versus delivered movement"
    else:
        hedge_txt = "hedge pricing versus delivered movement looks near fair"

    return (
        f"The regime reads {summary['regime'].lower()}. Equity volatility is {describe(eq['med_z'], eq['trend'])}, "
        f"rates volatility is {describe(rt['med_z'], rt['trend'])}, FX volatility is {describe(fx['med_z'], fx['trend'])}, "
        f"and commodity volatility is {describe(cmd['med_z'], cmd['trend'])}. Cross-asset dispersion is {disp_txt} "
        f"and the recent change in stress looks {broad_txt}. On relative value, {hedge_txt}."
    )


def add_global_style():
    st.markdown(
        f"""
        <style>
        .block-container {{
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }}
        div[data-testid="stDataFrame"] div[role="table"] {{
            border: 1px solid {BORDER};
            border-radius: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


add_global_style()

st.title("Cross-Asset Volatility Monitor")
st.caption("Implied volatility, realized volatility, term structure, and implied-versus-realized spreads in one regime dashboard. Data source: Yahoo Finance.")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Cross-asset volatility regime monitor for stress concentration and hedge richness.

        **What this tab shows**
        - Realized-volatility regime reads across key asset classes.
        - Concentration checks to see whether stress is isolated or broadening.
        - Practical context for hedge pricing, demand, and risk posture.

        **Data source**
        - Public market and macro data feeds used throughout the app.
        """
    )
    st.divider()
    st.header("Controls")
    start_date = st.date_input("History start", datetime(2010, 1, 1))
    rv_windows = st.multiselect("Realized vol horizons (trading days)", [5, 10, 21, 63], default=[21, 63])
    if not rv_windows:
        rv_windows = [21]
    estimator = st.selectbox("Realized vol estimator", ["Close to close", "Parkinson", "Garman-Klass"], index=0)
    norm_window = st.selectbox("Normalization window", ["1Y", "3Y", "5Y", "Max"], index=1)
    spread_anchor_window = st.selectbox("Spread anchor RV window", sorted(rv_windows), index=0)
    ts_lookback = st.selectbox("Time-series lookback", ["1Y", "3Y", "5Y", "Max"], index=1)
    show_only_latest = st.checkbox("Details: show current outliers only", value=False)
    st.caption("Normalization uses trailing valid observations rather than calendar days.")

norm_obs = {"1Y": 252, "3Y": 756, "5Y": 1260, "Max": 5000}[norm_window]
ts_obs = {"1Y": 252, "3Y": 756, "5Y": 1260, "Max": 5000}[ts_lookback]

all_tickers = sorted({x["ticker"] for x in IMPLIED_SPECS + UNDERLYING_SPECS})
raw_data = load_ohlc_data(all_tickers, start=str(start_date))
features, missing = build_feature_specs(raw_data, sorted(rv_windows), estimator, spread_anchor_window)
df, histories = build_feature_frame(features, norm_obs)

if df.empty:
    st.error("No usable volatility features were built from the selected inputs.")
    st.stop()

latest_common_date = max((meta["level"].index.max() for meta in histories.values() if not meta["level"].empty), default=None)
if latest_common_date is not None:
    df["Lag vs Latest"] = df["Last Date"].apply(lambda d: (latest_common_date.date() - d).days if pd.notna(d) else np.nan)
else:
    df["Lag vs Latest"] = np.nan

summary = build_regime_summary(df)
commentary = generate_commentary(summary)

tab_overview, tab_heatmap, tab_details, tab_ts = st.tabs(["Overview", "Heatmap", "Details", "Time Series"])

with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        make_card("Regime", summary["regime"], f"Stress score {fmt_num(summary['stress'], 2)} | Normalization {norm_window}")
    with c2:
        eq = summary["Equity"]
        make_card("Equity vol", f"Z {fmt_num(eq['med_z'], 2)}", f"Pct {fmt_num(eq['med_pct'], 0, '%')} | Hot {eq['hot']}/{eq['count']}")
    with c3:
        rt = summary["Rates"]
        make_card("Rates vol", f"Z {fmt_num(rt['med_z'], 2)}", f"Pct {fmt_num(rt['med_pct'], 0, '%')} | Hot {rt['hot']}/{rt['count']}")
    with c4:
        fx = summary["FX"]
        make_card("FX vol", f"Z {fmt_num(fx['med_z'], 2)}", f"Pct {fmt_num(fx['med_pct'], 0, '%')} | Hot {fx['hot']}/{fx['count']}")
    with c5:
        make_card("Hedge richness", f"Z {fmt_num(summary['hedge_richness'], 2)}", f"Dispersion {fmt_num(summary['dispersion'], 2)} | Breadth trend {fmt_num(summary['broadening'], 2)}")

    st.subheader("Commentary")
    st.markdown(
        f"""
        <div style="border:1px solid {BORDER}; border-radius:14px; padding:16px; background:{BG}; line-height:1.55;">
        {commentary}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Top outliers")
    out = df.dropna(subset=["Z"]).sort_values("AbsZ", ascending=False).head(10).copy()
    if not out.empty:
        out["Level Display"] = out.apply(lambda r: f'{r["Level"]:.2f}x' if r["Unit"] == "x" else f'{r["Level"]:.2f}%', axis=1)
        out["5D Display"] = out.apply(lambda r: f'{r["5D Change"]:+.2f} pts' if r["ChangeType"] == "pts" else f'{r["5D Change"]:+.1f}%', axis=1)
        st.dataframe(
            out[["Series", "Sleeve", "Kind", "Horizon", "Level Display", "5D Display", "Z", "Percentile", "Z 20D Δ", "State Shift"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Z": st.column_config.NumberColumn(format="%.2f"),
                "Percentile": st.column_config.NumberColumn(format="%.0f%%"),
                "Z 20D Δ": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    st.subheader("Cross-asset Z snapshot")
    z_df = df.dropna(subset=["Z"]).sort_values(["Sleeve", "AbsZ"], ascending=[True, False]).copy()
    fig_bar = go.Figure()
    for sleeve in SLEEVE_ORDER:
        sub = z_df[z_df["Sleeve"] == sleeve]
        if sub.empty:
            continue
        fig_bar.add_trace(
            go.Bar(
                x=sub["Z"],
                y=sub["Series"],
                orientation="h",
                name=sleeve,
                marker_color=sleeve_color(sleeve),
                customdata=np.column_stack([
                    sub["Kind"],
                    sub["Horizon"],
                    sub["Level"],
                    sub["Unit"],
                    sub["Percentile"],
                    sub["Z 20D Δ"],
                    sub["State Shift"],
                ]),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Z: %{x:.2f}<br>"
                    "Kind: %{customdata[0]}<br>"
                    "Horizon: %{customdata[1]}<br>"
                    "Level: %{customdata[2]:.2f} %{customdata[3]}<br>"
                    "Percentile: %{customdata[4]:.0f}%<br>"
                    "Z 20D Δ: %{customdata[5]:.2f}<br>"
                    "State: %{customdata[6]}<extra></extra>"
                ),
            )
        )
    fig_bar.add_vline(x=0, line_color="#9ca3af", line_width=1)
    for x in [-2, -1, 1, 2]:
        fig_bar.add_vline(x=x, line_color="#d1d5db", line_width=1, line_dash="dot")
    fig_bar.update_layout(
        height=max(500, 28 * len(z_df) + 120),
        margin=dict(l=10, r=10, t=10, b=10),
        barmode="group",
        xaxis_title="Z-score",
        yaxis_title="",
        legend_title_text="Sleeve",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with tab_heatmap:
    st.subheader("Heatmaps")
    st.caption("Z-score heatmap is centered at zero. Percentile heatmap ranks each series within the selected normalization lookback.")

    heat = df.copy()
    heat["Col"] = heat.apply(lambda r: f'{r["Kind"]} | {r["Horizon"]}', axis=1)

    z_pivot = heat.pivot_table(index="Series", columns="Col", values="Z", aggfunc="last")
    z_pivot = z_pivot.loc[z_pivot.abs().max(axis=1).sort_values(ascending=False).index]
    finite_z = z_pivot.to_numpy(dtype=float)
    finite_z = finite_z[np.isfinite(finite_z)]
    zmax = max(2.0, min(6.0, np.nanmax(np.abs(finite_z)) if finite_z.size else 2.0))

    fig_hm_z = px.imshow(
        z_pivot,
        aspect="auto",
        color_continuous_scale=Z_DIVERGING,
        zmin=-zmax,
        zmax=zmax,
        text_auto=".1f",
        labels=dict(color="Z"),
        height=min(1000, 30 * len(z_pivot) + 170),
    )
    fig_hm_z.update_xaxes(side="top")
    fig_hm_z.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_hm_z, use_container_width=True)

    pct_pivot = heat.pivot_table(index="Series", columns="Col", values="Percentile", aggfunc="last")
    pct_pivot = pct_pivot.loc[pct_pivot.max(axis=1).sort_values(ascending=False).index]
    fig_hm_pct = px.imshow(
        pct_pivot,
        aspect="auto",
        color_continuous_scale=PCT_SEQUENTIAL,
        zmin=0,
        zmax=100,
        text_auto=".0f",
        labels=dict(color="Pct"),
        height=min(1000, 30 * len(pct_pivot) + 170),
    )
    fig_hm_pct.update_xaxes(side="top")
    fig_hm_pct.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_hm_pct, use_container_width=True)

with tab_details:
    st.subheader("Details")
    st.caption("Implied indices and spread features show 5-day point change. Realized volatility and ratios show 5-day % change. Sparklines are min-max normalized over the last 90 observations for shape only.")

    detail = df.copy()
    if show_only_latest:
        detail = detail[detail["AbsZ"] >= 1.0].copy()

    detail["Level Display"] = detail.apply(lambda r: f'{r["Level"]:.2f}x' if r["Unit"] == "x" else f'{r["Level"]:.2f}%', axis=1)
    detail["5D Display"] = detail.apply(lambda r: f'{r["5D Change"]:+.2f} pts' if r["ChangeType"] == "pts" else f'{r["5D Change"]:+.1f}%', axis=1)

    st.dataframe(
        detail[["Series", "Sleeve", "Kind", "Horizon", "Level Display", "5D Display", "Z", "Percentile", "Z 20D Δ", "State Shift", "Lag vs Latest", "Spark"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Z": st.column_config.NumberColumn(format="%.2f"),
            "Percentile": st.column_config.NumberColumn(format="%.0f%%"),
            "Z 20D Δ": st.column_config.NumberColumn(format="%.2f"),
            "Lag vs Latest": st.column_config.NumberColumn(format="%.0f d"),
            "Spark": st.column_config.LineChartColumn("90D", width="small", y_min=0.0, y_max=1.0),
        },
    )

    stale = df[df["Lag vs Latest"].fillna(0) > 1][["Series", "Last Date", "Lag vs Latest"]].copy()
    if not stale.empty:
        st.warning("Some series are stale relative to the latest common date.")
        st.dataframe(stale, use_container_width=True, hide_index=True)

    if missing:
        st.info("Missing or unusable source series: " + ", ".join(sorted(set(missing))))

with tab_ts:
    st.subheader("Time series")
    st.caption("Each panel shows the raw level on the left axis and rolling Z-score on the right axis, with one and two sigma guide rails.")

    names = list(df["Series"])
    selected = st.multiselect("Select series", names, default=names[:min(9, len(names))])
    plot_items = selected if selected else names[:min(9, len(names))]
    cols_per_row = 3

    for start in range(0, len(plot_items), cols_per_row):
        row_items = plot_items[start:start + cols_per_row]
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            with cols[j]:
                if j >= len(row_items):
                    st.empty()
                    continue

                name = row_items[j]
                hist = histories[name]
                level = hist["level"].dropna()
                z = hist["z"].dropna()

                if ts_obs < 5000:
                    level = level.tail(ts_obs)
                z_plot = z.reindex(level.index)

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Scatter(
                        x=level.index,
                        y=level.values,
                        mode="lines",
                        name="Level",
                        line=dict(color=sleeve_color(hist["sleeve"]), width=2),
                    ),
                    secondary_y=False,
                )
                if z_plot.notna().any():
                    fig.add_trace(
                        go.Scatter(
                            x=z_plot.index,
                            y=z_plot.values,
                            mode="lines",
                            name="Z",
                            line=dict(color="#374151", width=1.6, dash="dot"),
                        ),
                        secondary_y=True,
                    )
                for thr in [-2, -1, 1, 2]:
                    fig.add_hline(y=thr, line_color="#d1d5db", line_width=1, line_dash="dot", secondary_y=True)

                fig.update_layout(
                    height=320,
                    margin=dict(l=10, r=10, t=38, b=10),
                    title=dict(text=name, x=0.02, xanchor="left", font=dict(size=13, color=TEXT)),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, font=dict(size=10)),
                    xaxis=dict(title="", showgrid=False),
                )
                fig.update_yaxes(title_text=hist["unit"], showgrid=True, gridcolor="#f3f4f6", secondary_y=False)
                fig.update_yaxes(title_text="Z", range=[-3.5, 3.5], showgrid=False, secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)

st.caption("© 2026 AD Fund Management LP")
