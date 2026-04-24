import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="ADFM | Cross-Asset Vol Tape", layout="wide")

APP_VERSION = "2026.04.24"

TEXT = "#1f2937"
SUBTLE = "#6b7280"
BORDER = "#e5e7eb"
BG = "#ffffff"
CARD_BG = "#fafafa"

PASTEL_GREEN = "#52b788"
PASTEL_RED = "#e85d5d"
PASTEL_GREY = "#8b949e"
AMBER = "#f4a261"

SLEEVE_COLORS = {
    "Equity": "#A8DADC",
    "Rates": "#BDE0FE",
    "FX": "#CDB4DB",
    "Commodity": "#90BE6D",
    "Credit": "#FFD6A5",
    "Relative Value": "#F4A261",
    "Term Structure": "#B8C0FF",
}

SLEEVE_ORDER = [
    "Equity",
    "Rates",
    "FX",
    "Commodity",
    "Credit",
    "Relative Value",
    "Term Structure",
]

Z_DIVERGING = "RdYlGn_r"
PCT_SEQUENTIAL = "YlOrRd"

IMPLIED_SPECS = [
    {"ticker": "^VIX", "label": "Equity | VIX", "sleeve": "Equity", "unit": "vol pts"},
    {"ticker": "^VIX9D", "label": "Equity | VIX9D", "sleeve": "Equity", "unit": "vol pts"},
    {"ticker": "^VIX3M", "label": "Equity | VIX3M", "sleeve": "Equity", "unit": "vol pts"},
    {"ticker": "^VVIX", "label": "Equity | VVIX", "sleeve": "Equity", "unit": "vol pts"},
    {"ticker": "^SKEW", "label": "Equity | SKEW", "sleeve": "Equity", "unit": "index pts"},
    {"ticker": "^MOVE", "label": "Rates | MOVE", "sleeve": "Rates", "unit": "index pts"},
    {"ticker": "^GVZ", "label": "Commodity | GVZ", "sleeve": "Commodity", "unit": "vol pts"},
    {"ticker": "^OVX", "label": "Commodity | OVX", "sleeve": "Commodity", "unit": "vol pts"},
]

UNDERLYING_SPECS = [
    {"ticker": "^GSPC", "label": "Equity | SPX", "sleeve": "Equity"},
    {"ticker": "SPY", "label": "Equity | SPY", "sleeve": "Equity"},
    {"ticker": "QQQ", "label": "Equity | QQQ", "sleeve": "Equity"},
    {"ticker": "IWM", "label": "Equity | IWM", "sleeve": "Equity"},
    {"ticker": "SMH", "label": "Equity | SMH", "sleeve": "Equity"},
    {"ticker": "XLK", "label": "Equity | XLK", "sleeve": "Equity"},
    {"ticker": "XLF", "label": "Equity | XLF", "sleeve": "Equity"},
    {"ticker": "XLE", "label": "Equity | XLE", "sleeve": "Equity"},
    {"ticker": "XLU", "label": "Equity | XLU", "sleeve": "Equity"},
    {"ticker": "TLT", "label": "Rates | TLT", "sleeve": "Rates"},
    {"ticker": "IEF", "label": "Rates | IEF", "sleeve": "Rates"},
    {"ticker": "SHY", "label": "Rates | SHY", "sleeve": "Rates"},
    {"ticker": "EDV", "label": "Rates | EDV", "sleeve": "Rates"},
    {"ticker": "HYG", "label": "Credit | HYG", "sleeve": "Credit"},
    {"ticker": "LQD", "label": "Credit | LQD", "sleeve": "Credit"},
    {"ticker": "GLD", "label": "Commodity | GLD", "sleeve": "Commodity"},
    {"ticker": "SLV", "label": "Commodity | SLV", "sleeve": "Commodity"},
    {"ticker": "CPER", "label": "Commodity | Copper", "sleeve": "Commodity"},
    {"ticker": "URA", "label": "Commodity | Uranium", "sleeve": "Commodity"},
    {"ticker": "USO", "label": "Commodity | USO", "sleeve": "Commodity"},
    {"ticker": "CL=F", "label": "Commodity | WTI", "sleeve": "Commodity"},
    {"ticker": "BZ=F", "label": "Commodity | Brent", "sleeve": "Commodity"},
    {"ticker": "NG=F", "label": "Commodity | Nat Gas", "sleeve": "Commodity"},
    {"ticker": "DX-Y.NYB", "label": "FX | DXY", "sleeve": "FX"},
    {"ticker": "UUP", "label": "FX | UUP", "sleeve": "FX"},
    {"ticker": "UDN", "label": "FX | UDN", "sleeve": "FX"},
    {"ticker": "JPY=X", "label": "FX | USDJPY", "sleeve": "FX"},
    {"ticker": "MXN=X", "label": "FX | USDMXN", "sleeve": "FX"},
    {"ticker": "EURUSD=X", "label": "FX | EURUSD", "sleeve": "FX"},
    {"ticker": "CHF=X", "label": "FX | USDCHF", "sleeve": "FX"},
    {"ticker": "CNH=X", "label": "FX | USDCNH", "sleeve": "FX"},
]

SPREAD_SPECS = [
    {"implied": "^VIX", "underlying": "^GSPC", "name": "Equity | VIX - SPX RV", "sleeve": "Relative Value"},
    {"implied": "^VIX", "underlying": "QQQ", "name": "Equity | VIX - QQQ RV", "sleeve": "Relative Value"},
    {"implied": "^VIX", "underlying": "IWM", "name": "Equity | VIX - IWM RV", "sleeve": "Relative Value"},
    {"implied": "^MOVE", "underlying": "TLT", "name": "Rates | MOVE - TLT RV", "sleeve": "Relative Value"},
    {"implied": "^GVZ", "underlying": "GLD", "name": "Commodity | GVZ - GLD RV", "sleeve": "Relative Value"},
    {"implied": "^OVX", "underlying": "CL=F", "name": "Commodity | OVX - WTI RV", "sleeve": "Relative Value"},
    {"implied": "^OVX", "underlying": "USO", "name": "Commodity | OVX - USO RV", "sleeve": "Relative Value"},
]

TERM_STRUCTURE_SPECS = [
    {"a": "^VIX3M", "b": "^VIX", "name": "Equity | VIX3M - VIX", "unit": "vol pts", "change_type": "pts"},
    {"a": "^VIX", "b": "^VIX9D", "name": "Equity | VIX - VIX9D", "unit": "vol pts", "change_type": "pts"},
    {"a": "^VIX3M", "b": "^VIX", "name": "Equity | VIX3M / VIX", "unit": "x", "change_type": "%", "ratio": True},
    {"a": "^VVIX", "b": "^VIX", "name": "Equity | VVIX / VIX", "unit": "x", "change_type": "%", "ratio": True},
    {"a": "^SKEW", "b": "^VIX", "name": "Equity | SKEW / VIX", "unit": "x", "change_type": "%", "ratio": True},
]


@dataclass
class FeatureSpec:
    name: str
    sleeve: str
    kind: str
    horizon: str
    unit: str
    change_type: str
    series: pd.Series
    source: str = "Yahoo Finance"


def add_global_style():
    st.markdown(
        f"""
        <style>
        .block-container {{
            padding-top: 1.15rem;
            padding-bottom: 2.25rem;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.35rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            border: 1px solid {BORDER};
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            background: {BG};
        }}
        div[data-testid="stDataFrame"] div[role="table"] {{
            border: 1px solid {BORDER};
            border-radius: 12px;
        }}
        .soft-note {{
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 13px 15px;
            background: {CARD_BG};
            color: {TEXT};
            line-height: 1.5;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def realized_vol_close_to_close(price: pd.Series, window: int, ann: int = 252) -> pd.Series:
    ret = price.astype(float).pct_change()
    return ret.rolling(window).std() * np.sqrt(ann) * 100.0


def realized_vol_parkinson(high: pd.Series, low: pd.Series, window: int, ann: int = 252) -> pd.Series:
    high = high.astype(float)
    low = low.astype(float)
    hl = np.log(high / low.replace(0, np.nan))
    rs = (hl ** 2) / (4.0 * np.log(2.0))
    return np.sqrt(rs.rolling(window).mean() * ann) * 100.0


def realized_vol_gk(high: pd.Series, low: pd.Series, close: pd.Series, open_: pd.Series, window: int, ann: int = 252) -> pd.Series:
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    open_ = open_.astype(float)

    log_hl = np.log(high / low.replace(0, np.nan))
    log_co = np.log(close / open_.replace(0, np.nan))
    var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    var = var.clip(lower=0)

    return np.sqrt(var.rolling(window).mean() * ann) * 100.0


def empirical_percentile_series(s: pd.Series, lookback: int) -> pd.Series:
    vals = s.astype(float).replace([np.inf, -np.inf], np.nan)
    out = pd.Series(index=vals.index, dtype=float)

    min_periods = max(20, min(lookback, 60))

    for i in range(len(vals)):
        if pd.isna(vals.iloc[i]):
            continue

        start = max(0, i - lookback + 1)
        window = vals.iloc[start:i + 1].dropna()

        if len(window) < min_periods:
            continue

        out.iloc[i] = 100.0 * (window <= vals.iloc[i]).mean()

    return out


def rolling_zscore_series(s: pd.Series, lookback: int) -> pd.Series:
    vals = s.astype(float).replace([np.inf, -np.inf], np.nan)
    min_periods = max(20, min(lookback, 60))

    mu = vals.rolling(lookback, min_periods=min_periods).mean()
    sig = vals.rolling(lookback, min_periods=min_periods).std(ddof=0)

    return (vals - mu) / sig.replace(0, np.nan)


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


def point_change(s: pd.Series, periods: int) -> float:
    s = s.dropna()

    if len(s) < periods + 1:
        return np.nan

    return float(s.iloc[-1] - s.iloc[-periods - 1])


def pct_change(s: pd.Series, periods: int) -> float:
    s = s.dropna()

    if len(s) < periods + 1 or np.isclose(s.iloc[-periods - 1], 0):
        return np.nan

    return float((s.iloc[-1] / s.iloc[-periods - 1] - 1.0) * 100.0)


def sleeve_color(sleeve: str) -> str:
    return SLEEVE_COLORS.get(str(sleeve), "#d1d5db")


def pressure_color(score: float) -> str:
    if pd.isna(score):
        return PASTEL_GREY

    if score >= 0.35:
        return PASTEL_RED

    if score <= -0.35:
        return PASTEL_GREEN

    return PASTEL_GREY


def signal_color(signal: str) -> str:
    s = str(signal).lower()

    if "rising" in s or "upper-tail" in s or "backwardation" in s or "shock" in s:
        return PASTEL_RED

    if "cheap" in s or "compress" in s or "fading" in s or "calming" in s:
        return PASTEL_GREEN

    if "rich" in s:
        return AMBER

    return PASTEL_GREY


@st.cache_data(show_spinner=False, ttl=1800)
def load_ohlc_data(tickers: tuple[str, ...], start: str = "2010-01-01") -> dict:
    tickers = list(dict.fromkeys(tickers))

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

    def clean_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index).tz_localize(None)

        df = df.sort_index()
        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]

        return df[keep].dropna(how="all")

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        lvl1 = raw.columns.get_level_values(1)

        if set(["Open", "High", "Low", "Close"]).intersection(set(lvl0)):
            if len(tickers) == 1:
                out[tickers[0]] = clean_df(raw)
        else:
            for ticker in tickers:
                if ticker in lvl0:
                    out[ticker] = clean_df(raw[ticker])
                elif ticker in lvl1:
                    out[ticker] = clean_df(raw.xs(ticker, axis=1, level=1))
    else:
        if len(tickers) == 1:
            out[tickers[0]] = clean_df(raw)

    return out


def close_series(raw_data: dict, ticker: str) -> pd.Series | None:
    if ticker not in raw_data:
        return None

    df = raw_data[ticker]

    if df is None or df.empty or "Close" not in df.columns:
        return None

    s = df["Close"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()

    return s if not s.empty else None


def build_realized_vol(df: pd.DataFrame, window: int, estimator: str) -> pd.Series:
    if estimator == "Close to close":
        return realized_vol_close_to_close(df["Close"], window)

    if estimator == "Parkinson":
        return realized_vol_parkinson(df["High"], df["Low"], window)

    return realized_vol_gk(df["High"], df["Low"], df["Close"], df["Open"], window)


def align_binary_series(a: pd.Series, b: pd.Series, operation: str = "diff") -> pd.Series:
    idx = a.index.union(b.index)

    aa = a.reindex(idx).sort_index().ffill()
    bb = b.reindex(idx).sort_index().ffill()

    if operation == "ratio":
        out = aa / bb.replace(0, np.nan)
    else:
        out = aa - bb

    return out.replace([np.inf, -np.inf], np.nan).dropna().astype(float)


def build_feature_specs(
    raw_data: dict,
    rv_windows: list[int],
    estimator: str,
    spread_windows: list[int],
) -> tuple[list[FeatureSpec], list[str]]:
    features: list[FeatureSpec] = []
    missing: list[str] = []
    rv_map: dict[tuple[str, int], pd.Series] = {}

    for spec in IMPLIED_SPECS:
        s = close_series(raw_data, spec["ticker"])

        if s is None:
            missing.append(spec["ticker"])
            continue

        features.append(
            FeatureSpec(
                name=spec["label"],
                sleeve=spec["sleeve"],
                kind="Implied",
                horizon="Index",
                unit=spec.get("unit", "index pts"),
                change_type="pts",
                series=s,
            )
        )

    for spec in UNDERLYING_SPECS:
        ticker = spec["ticker"]

        if ticker not in raw_data or raw_data[ticker].empty:
            missing.append(ticker)
            continue

        df = raw_data[ticker].copy()
        needed = {"Open", "High", "Low", "Close"}

        if not needed.issubset(set(df.columns)):
            missing.append(ticker)
            continue

        for w in rv_windows:
            rv = build_realized_vol(df, w, estimator).dropna().astype(float)

            if rv.empty:
                continue

            rv_map[(ticker, w)] = rv

            features.append(
                FeatureSpec(
                    name=f'{spec["label"]} | RV{w}',
                    sleeve=spec["sleeve"],
                    kind="Realized",
                    horizon=f"RV{w}",
                    unit="%",
                    change_type="%",
                    series=rv,
                )
            )

    for spread in SPREAD_SPECS:
        imp = close_series(raw_data, spread["implied"])

        if imp is None:
            continue

        for w in spread_windows:
            rv = rv_map.get((spread["underlying"], w))

            if rv is None or rv.empty:
                continue

            s = align_binary_series(imp, rv, operation="diff")

            if s.empty:
                continue

            features.append(
                FeatureSpec(
                    name=f'{spread["name"]} | RV{w}',
                    sleeve=spread["sleeve"],
                    kind="Spread",
                    horizon=f"IV-RV{w}",
                    unit="vol pts",
                    change_type="pts",
                    series=s,
                )
            )

    for spec in TERM_STRUCTURE_SPECS:
        a = close_series(raw_data, spec["a"])
        b = close_series(raw_data, spec["b"])

        if a is None or b is None:
            continue

        op = "ratio" if spec.get("ratio") else "diff"
        s = align_binary_series(a, b, operation=op)

        if s.empty:
            continue

        features.append(
            FeatureSpec(
                name=spec["name"],
                sleeve="Term Structure",
                kind="Ratio" if op == "ratio" else "Spread",
                horizon="Term Structure" if op == "diff" else "Convexity",
                unit=spec["unit"],
                change_type=spec["change_type"],
                series=s,
            )
        )

    return features, sorted(set(missing))


def classify_signal(row: pd.Series, stale_threshold: int) -> str:
    if pd.notna(row.get("Lag vs Latest")) and row.get("Lag vs Latest") > stale_threshold:
        return "Stale"

    z = row.get("Z")
    pct = row.get("Percentile")
    zchg = row.get("Z 20D Change")
    kind = str(row.get("Kind", ""))
    name = str(row.get("Series", ""))

    if kind == "Spread":
        if pd.notna(z) and z >= 1.25:
            return "Hedge rich"

        if pd.notna(z) and z <= -1.25:
            return "Hedge cheap"

    if "VIX3M - VIX" in name:
        if pd.notna(z) and z <= -1.0:
            return "Front-end stress"

        if pd.notna(z) and z >= 1.0:
            return "Term premium calm"

    if pd.notna(z) and pd.notna(zchg):
        if z >= 1.5 and zchg >= 0.35:
            return "Stress rising"

        if z >= 1.5 and zchg <= -0.35:
            return "Stress high, fading"

        if z <= -1.0 and zchg <= -0.35:
            return "Vol compressing"

        if z <= -1.0 and zchg >= 0.35:
            return "Compression reversing"

    if pd.notna(pct):
        if pct >= 90:
            return "Upper-tail pressure"

        if pct <= 10:
            return "Lower-tail calm"

    return "Neutral"


def trade_read(row: pd.Series) -> str:
    signal = str(row.get("Signal", ""))
    sleeve = str(row.get("Sleeve", ""))
    kind = str(row.get("Kind", ""))
    zchg = row.get("Z 20D Change")

    if signal == "Stale":
        return "Ignore until data refreshes"

    if signal == "Hedge cheap":
        return "Optionality screens cheap versus delivered movement"

    if signal == "Hedge rich":
        return "Avoid chasing premium unless catalyst is immediate"

    if signal in ["Stress rising", "Upper-tail pressure", "Compression reversing"]:
        if sleeve == "Equity":
            return "Equity hedge demand is confirming"

        if sleeve == "Rates":
            return "Rates shock is leading the tape"

        if sleeve == "FX":
            return "FX stress is moving up the board"

        if sleeve == "Commodity":
            return "Commodity shock risk is active"

        if sleeve == "Credit":
            return "Credit vol deserves closer watch"

        return "Stress impulse is active"

    if signal in ["Stress high, fading", "Vol compressing", "Lower-tail calm", "Term premium calm"]:
        return "Pressure is fading unless price confirms again"

    if signal == "Front-end stress":
        return "Front-end protection demand is elevated"

    if kind == "Realized" and pd.notna(zchg) and zchg > 0.5:
        return "Delivered movement is accelerating"

    return "No strong edge from vol alone"


def build_feature_frame(features: list[FeatureSpec], norm_obs: int) -> tuple[pd.DataFrame, dict]:
    rows = []
    histories = {}

    for feat in features:
        s = feat.series.dropna().astype(float).replace([np.inf, -np.inf], np.nan).dropna()

        if s.empty:
            continue

        z_hist = rolling_zscore_series(s, norm_obs)
        pct_hist = empirical_percentile_series(s, norm_obs)

        z_valid = z_hist.dropna()
        pct_valid = pct_hist.dropna()

        latest_z = z_valid.iloc[-1] if not z_valid.empty else np.nan
        latest_pct = pct_valid.iloc[-1] if not pct_valid.empty else np.nan

        z20_change = np.nan

        if len(z_valid) >= 21:
            z20_change = float(z_valid.iloc[-1] - z_valid.iloc[-21])

        chg_5d = point_change(s, 5) if feat.change_type == "pts" else pct_change(s, 5)
        chg_20d = point_change(s, 20) if feat.change_type == "pts" else pct_change(s, 20)
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
                "20D Change": chg_20d,
                "Z": float(latest_z) if pd.notna(latest_z) else np.nan,
                "AbsZ": abs(float(latest_z)) if pd.notna(latest_z) else np.nan,
                "Percentile": float(latest_pct) if pd.notna(latest_pct) else np.nan,
                "Z 20D Change": z20_change,
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
            "change_type": feat.change_type,
        }

    frame = pd.DataFrame(rows)

    if frame.empty:
        return frame, histories

    frame["Sleeve"] = pd.Categorical(frame["Sleeve"], categories=SLEEVE_ORDER, ordered=True)

    return frame.reset_index(drop=True), histories


def add_lag_and_rank(frame: pd.DataFrame, histories: dict, stale_threshold: int) -> pd.DataFrame:
    out = frame.copy()

    latest_date = max(
        (meta["level"].index.max() for meta in histories.values() if not meta["level"].empty),
        default=None,
    )

    if latest_date is not None:
        out["Lag vs Latest"] = out["Last Date"].apply(
            lambda d: (latest_date.date() - d).days if pd.notna(d) else np.nan
        )
    else:
        out["Lag vs Latest"] = np.nan

    z_mat = pd.DataFrame({name: meta["z"] for name, meta in histories.items()}).sort_index().ffill()
    z_mat = z_mat.dropna(how="all")

    if z_mat.empty:
        out["Rank"] = np.nan
        out["Rank 5D Change"] = np.nan
        out["Rank 20D Change"] = np.nan
    else:
        current = z_mat.iloc[-1].abs().rank(ascending=False, method="min")
        out["Rank"] = out["Series"].map(current)

        if len(z_mat) >= 6:
            prior = z_mat.iloc[-6].abs().rank(ascending=False, method="min")
            out["Rank 5D Change"] = out["Series"].map(prior - current)
        else:
            out["Rank 5D Change"] = np.nan

        if len(z_mat) >= 21:
            prior20 = z_mat.iloc[-21].abs().rank(ascending=False, method="min")
            out["Rank 20D Change"] = out["Series"].map(prior20 - current)
        else:
            out["Rank 20D Change"] = np.nan

    out["Signal"] = out.apply(lambda r: classify_signal(r, stale_threshold), axis=1)
    out["Trade Read"] = out.apply(trade_read, axis=1)

    out["Pressure Score"] = out.apply(
        lambda r: np.nan if r["Signal"] == "Stale" else (
            0.65 * (0.0 if pd.isna(r["Z"]) else r["Z"]) +
            0.35 * (0.0 if pd.isna(r["Z 20D Change"]) else r["Z 20D Change"])
        ),
        axis=1,
    )

    out["Pressure Color"] = out["Pressure Score"].apply(pressure_color)
    out["Signal Color"] = out["Signal"].apply(signal_color)

    out = out.sort_values(["AbsZ", "Rank 5D Change"], ascending=[False, False]).reset_index(drop=True)

    return out


def build_regime_map(frame: pd.DataFrame, stale_threshold: int) -> pd.DataFrame:
    fresh = frame[
        (frame["Lag vs Latest"].fillna(999) <= stale_threshold) &
        frame["Z"].notna()
    ].copy()

    rows = []

    for sleeve in SLEEVE_ORDER:
        sub = fresh[fresh["Sleeve"].astype(str) == sleeve]

        if sub.empty:
            continue

        med_z = float(sub["Z"].median(skipna=True))
        med_pct = float(sub["Percentile"].median(skipna=True))
        impulse = float(sub["Z 20D Change"].median(skipna=True))
        hot_count = int((sub["AbsZ"] >= 1.5).sum())
        count = int(len(sub))

        if med_z >= 1.25 and impulse >= 0.25:
            state = "Leading stress"
        elif med_z >= 1.25 and impulse < 0.25:
            state = "High but slower"
        elif med_z <= -0.75:
            state = "Compressed"
        elif impulse >= 0.5:
            state = "Building"
        elif impulse <= -0.5:
            state = "Cooling"
        else:
            state = "Neutral"

        rows.append(
            {
                "Sleeve": sleeve,
                "Median Z": med_z,
                "Median Percentile": med_pct,
                "Median 20D Z Change": impulse,
                "Hot Series": hot_count,
                "Series Count": count,
                "State": state,
            }
        )

    return pd.DataFrame(rows)


def generate_tape_read(regime_map: pd.DataFrame, board: pd.DataFrame) -> str:
    if regime_map.empty or board.empty:
        return "Tape read unavailable until enough series have valid history."

    leading = regime_map.sort_values("Median Z", ascending=False).head(2)
    impulse = regime_map.sort_values("Median 20D Z Change", ascending=False).head(2)
    top = board[board["Signal"] != "Stale"].sort_values("AbsZ", ascending=False).head(5)
    movers = board[board["Signal"] != "Stale"].sort_values("Rank 5D Change", ascending=False).head(3)

    lead_txt = ", ".join([f"{r['Sleeve']} Z {r['Median Z']:.2f}" for _, r in leading.iterrows()])
    impulse_txt = ", ".join([f"{r['Sleeve']} 20D change {r['Median 20D Z Change']:.2f}" for _, r in impulse.iterrows()])
    top_txt = ", ".join(top["Series"].astype(str).tolist()) if not top.empty else "no clean outliers"
    mover_txt = ", ".join(movers["Series"].astype(str).tolist()) if not movers.empty else "no major rank change"

    return (
        f"The vol tape is led by {lead_txt}. The strongest impulse is in {impulse_txt}. "
        f"The current outlier stack is {top_txt}. The fastest rank climbers over five sessions are {mover_txt}. "
        "Use this as a pressure board first and a regime label second: the useful signal is where volatility is migrating, "
        "whether implied premium is leading or lagging realized movement, and whether the stress is isolated or spreading across sleeves."
    )


def display_value(row: pd.Series) -> str:
    unit = row.get("Unit")
    level = row.get("Level")

    if pd.isna(level):
        return "NA"

    if unit == "x":
        return f"{level:.2f}x"

    if unit == "%":
        return f"{level:.2f}%"

    return f"{level:.2f}"


def display_change(row: pd.Series, col: str) -> str:
    val = row.get(col)

    if pd.isna(val):
        return "NA"

    if row.get("ChangeType") == "pts":
        return f"{val:+.2f} pts"

    return f"{val:+.1f}%"


def prepare_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d["Level Display"] = d.apply(display_value, axis=1)
    d["5D Display"] = d.apply(lambda r: display_change(r, "5D Change"), axis=1)
    d["20D Display"] = d.apply(lambda r: display_change(r, "20D Change"), axis=1)
    d["Sleeve"] = d["Sleeve"].astype(str)

    return d


def dataframe_config():
    return {
        "Rank": st.column_config.NumberColumn(format="%.0f"),
        "Rank 5D Change": st.column_config.NumberColumn(format="%+.0f"),
        "Rank 20D Change": st.column_config.NumberColumn(format="%+.0f"),
        "Z": st.column_config.NumberColumn(format="%.2f"),
        "AbsZ": st.column_config.NumberColumn(format="%.2f"),
        "Percentile": st.column_config.NumberColumn(format="%.0f%%"),
        "Z 20D Change": st.column_config.NumberColumn(format="%+.2f"),
        "Lag vs Latest": st.column_config.NumberColumn(format="%.0f d"),
        "Spark": st.column_config.LineChartColumn("90D", width="small", y_min=0.0, y_max=1.0),
    }


def plot_pressure_bar(board: pd.DataFrame, max_rows: int = 35) -> go.Figure:
    plot_df = board[board["Z"].notna()].copy().head(max_rows)
    plot_df = plot_df.sort_values("Z", ascending=True)

    colors = plot_df["Signal Color"].tolist()

    fig = go.Figure(
        go.Bar(
            x=plot_df["Z"],
            y=plot_df["Series"],
            orientation="h",
            marker=dict(color=colors),
            customdata=np.column_stack(
                [
                    plot_df["Sleeve"].astype(str),
                    plot_df["Kind"],
                    plot_df["Horizon"],
                    plot_df["Percentile"],
                    plot_df["Z 20D Change"],
                    plot_df["Rank 5D Change"],
                    plot_df["Signal"],
                    plot_df["Trade Read"],
                ]
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Z: %{x:.2f}<br>"
                "Sleeve: %{customdata[0]}<br>"
                "Kind: %{customdata[1]}<br>"
                "Horizon: %{customdata[2]}<br>"
                "Percentile: %{customdata[3]:.0f}%<br>"
                "20D Z change: %{customdata[4]:+.2f}<br>"
                "5D rank change: %{customdata[5]:+.0f}<br>"
                "Signal: %{customdata[6]}<br>"
                "Read: %{customdata[7]}<extra></extra>"
            ),
        )
    )

    fig.add_vline(x=0, line_color="#9ca3af", line_width=1)

    for x in [-2, -1, 1, 2]:
        fig.add_vline(x=x, line_color="#d1d5db", line_width=1, line_dash="dot")

    fig.update_layout(
        height=max(520, 24 * len(plot_df) + 120),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Z-score",
        yaxis_title="",
        showlegend=False,
    )

    return fig


def plot_sleeve_map(regime_map: pd.DataFrame) -> go.Figure:
    d = regime_map.copy()
    d["Color"] = d["Median 20D Z Change"].apply(pressure_color)

    fig = go.Figure(
        go.Bar(
            x=d["Median Z"],
            y=d["Sleeve"],
            orientation="h",
            marker=dict(color=d["Color"]),
            customdata=np.column_stack(
                [
                    d["Median Percentile"],
                    d["Median 20D Z Change"],
                    d["Hot Series"],
                    d["Series Count"],
                    d["State"],
                ]
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Median Z: %{x:.2f}<br>"
                "Median percentile: %{customdata[0]:.0f}%<br>"
                "Median 20D Z change: %{customdata[1]:+.2f}<br>"
                "Hot series: %{customdata[2]:.0f}/%{customdata[3]:.0f}<br>"
                "State: %{customdata[4]}<extra></extra>"
            ),
        )
    )

    fig.add_vline(x=0, line_color="#9ca3af", line_width=1)
    fig.add_vline(x=1.25, line_color="#d1d5db", line_width=1, line_dash="dot")
    fig.add_vline(x=-0.75, line_color="#d1d5db", line_width=1, line_dash="dot")

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Median Z",
        yaxis_title="",
        showlegend=False,
    )

    return fig


def plot_iv_rv_spreads(board: pd.DataFrame) -> go.Figure:
    d = board[board["Kind"].eq("Spread") & board["Z"].notna()].copy()

    if d.empty:
        return go.Figure()

    d = d.sort_values("Z", ascending=True)
    colors = d["Signal Color"].tolist()

    fig = go.Figure(
        go.Bar(
            x=d["Z"],
            y=d["Series"],
            orientation="h",
            marker=dict(color=colors),
            customdata=np.column_stack(
                [
                    d["Level"],
                    d["5D Change"],
                    d["Percentile"],
                    d["Signal"],
                    d["Trade Read"],
                ]
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Z: %{x:.2f}<br>"
                "Spread level: %{customdata[0]:.2f}<br>"
                "5D change: %{customdata[1]:+.2f}<br>"
                "Percentile: %{customdata[2]:.0f}%<br>"
                "Signal: %{customdata[3]}<br>"
                "Read: %{customdata[4]}<extra></extra>"
            ),
        )
    )

    fig.add_vline(x=0, line_color="#9ca3af", line_width=1)

    for x in [-2, -1, 1, 2]:
        fig.add_vline(x=x, line_color="#d1d5db", line_width=1, line_dash="dot")

    fig.update_layout(
        height=max(420, 26 * len(d) + 120),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Spread Z-score",
        yaxis_title="",
        showlegend=False,
    )

    return fig


def plot_heatmap(
    board: pd.DataFrame,
    value_col: str,
    color_scale: str,
    zmin=None,
    zmax=None,
    title_color="",
) -> go.Figure:
    heat = board.copy()
    heat["Sleeve"] = heat["Sleeve"].astype(str)
    heat["Col"] = heat.apply(lambda r: f'{r["Kind"]} | {r["Horizon"]}', axis=1)

    pivot = heat.pivot_table(index="Series", columns="Col", values=value_col, aggfunc="last")

    if pivot.empty:
        return go.Figure()

    if value_col == "Z":
        order = pivot.abs().max(axis=1).sort_values(ascending=False).index
    else:
        order = pivot.max(axis=1).sort_values(ascending=False).index

    pivot = pivot.loc[order]

    if value_col == "Z":
        finite = pivot.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        span = max(2.0, min(6.0, np.nanmax(np.abs(finite)) if finite.size else 2.0))
        zmin = -span
        zmax = span

    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale=color_scale,
        zmin=zmin,
        zmax=zmax,
        text_auto=".1f" if value_col == "Z" else ".0f",
        labels=dict(color=title_color or value_col),
        height=min(1200, 27 * len(pivot) + 170),
    )

    fig.update_xaxes(side="top")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))

    return fig


def plot_series_panel(name: str, histories: dict, ts_obs: int) -> go.Figure:
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
                line=dict(color="#374151", width=1.7, dash="dot"),
            ),
            secondary_y=True,
        )

    for thr in [-2, -1, 1, 2]:
        fig.add_hline(
            y=thr,
            line_color="#d1d5db",
            line_width=1,
            line_dash="dot",
            secondary_y=True,
        )

    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=42, b=10),
        title=dict(text=name, x=0.02, xanchor="left", font=dict(size=13, color=TEXT)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=10),
        ),
        xaxis=dict(title="", showgrid=False),
    )

    fig.update_yaxes(
        title_text=hist["unit"],
        showgrid=True,
        gridcolor="#f3f4f6",
        secondary_y=False,
    )

    fig.update_yaxes(
        title_text="Z",
        range=[-3.5, 3.5],
        showgrid=False,
        secondary_y=True,
    )

    return fig


add_global_style()

st.title("ADFM | Cross-Asset Vol Tape")
st.caption(
    "Implied volatility, realized volatility, vol-of-vol, term structure, and implied-versus-realized richness. Data source: Yahoo Finance."
)

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** A cross-asset pressure board for volatility migration, hedge richness, and stress concentration.

        **How to read it:** Start with the Vol Tape tab. Sort by rank change for what is moving, by absolute Z for where stress is highest, and by Signal for whether option premium is rich, cheap, rising, or fading.
        """
    )

    st.divider()
    st.header("Controls")

    start_date = st.date_input("History start", datetime(2010, 1, 1))

    rv_windows = st.multiselect(
        "Realized vol horizons",
        [5, 10, 21, 42, 63],
        default=[5, 21, 63],
    )

    if not rv_windows:
        rv_windows = [21]

    estimator = st.selectbox(
        "Realized vol estimator",
        ["Close to close", "Parkinson", "Garman-Klass"],
        index=0,
    )

    norm_window = st.selectbox(
        "Normalization window",
        ["1Y", "3Y", "5Y", "Max"],
        index=1,
    )

    spread_windows = st.multiselect(
        "IV-RV spread windows",
        sorted(rv_windows),
        default=[w for w in [5, 21, 63] if w in rv_windows],
    )

    if not spread_windows:
        spread_windows = [sorted(rv_windows)[0]]

    ts_lookback = st.selectbox(
        "Time-series lookback",
        ["6M", "1Y", "3Y", "5Y", "Max"],
        index=2,
    )

    stale_threshold = st.slider(
        "Stale threshold, calendar days",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )

    max_bar_rows = st.slider(
        "Max rows in main bar chart",
        min_value=15,
        max_value=80,
        value=40,
        step=5,
    )

    show_stale = st.checkbox("Show stale series in tables", value=True)

    st.caption(f"Version {APP_VERSION}")

norm_obs = {"1Y": 252, "3Y": 756, "5Y": 1260, "Max": 5000}[norm_window]
ts_obs = {"6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260, "Max": 5000}[ts_lookback]

all_tickers = tuple(sorted({x["ticker"] for x in IMPLIED_SPECS + UNDERLYING_SPECS}))

raw_data = load_ohlc_data(all_tickers, start=str(start_date))

features, missing = build_feature_specs(
    raw_data,
    sorted(rv_windows),
    estimator,
    sorted(spread_windows),
)

base_frame, histories = build_feature_frame(features, norm_obs)

if base_frame.empty:
    st.error("No usable volatility features were built from the selected inputs. Try a longer history window or fewer series.")
    st.stop()

board = add_lag_and_rank(base_frame, histories, stale_threshold)

if not show_stale:
    board = board[board["Signal"] != "Stale"].copy()

regime_map = build_regime_map(board, stale_threshold)
tape_read = generate_tape_read(regime_map, board)
display_board = prepare_display_frame(board)

vol_tape, iv_rv_tab, term_tab, heatmap_tab, explorer_tab, data_tab = st.tabs(
    [
        "Vol Tape",
        "Implied vs Realized",
        "Term Structure",
        "Heatmap",
        "Series Explorer",
        "Data Quality",
    ]
)

with vol_tape:
    st.subheader("Tape read")
    st.markdown(f"<div class='soft-note'>{tape_read}</div>", unsafe_allow_html=True)

    st.subheader("Vol pressure board")
    st.plotly_chart(plot_pressure_bar(board, max_rows=max_bar_rows), use_container_width=True)

    left, right = st.columns([1.15, 1.0])

    with left:
        st.subheader("What moved up the board")

        movers = display_board.sort_values("Rank 5D Change", ascending=False).head(15)

        st.dataframe(
            movers[
                [
                    "Rank",
                    "Rank 5D Change",
                    "Series",
                    "Sleeve",
                    "Signal",
                    "Trade Read",
                    "Level Display",
                    "5D Display",
                    "Z",
                    "Percentile",
                    "Z 20D Change",
                    "Lag vs Latest",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config=dataframe_config(),
        )

    with right:
        st.subheader("Sleeve regime map")

        if not regime_map.empty:
            st.plotly_chart(plot_sleeve_map(regime_map), use_container_width=True)

            st.dataframe(
                regime_map,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Median Z": st.column_config.NumberColumn(format="%.2f"),
                    "Median Percentile": st.column_config.NumberColumn(format="%.0f%%"),
                    "Median 20D Z Change": st.column_config.NumberColumn(format="%+.2f"),
                },
            )

    st.subheader("Full board")

    st.dataframe(
        display_board[
            [
                "Rank",
                "Rank 5D Change",
                "Rank 20D Change",
                "Series",
                "Sleeve",
                "Kind",
                "Horizon",
                "Signal",
                "Trade Read",
                "Level Display",
                "5D Display",
                "20D Display",
                "Z",
                "Percentile",
                "Z 20D Change",
                "Lag vs Latest",
                "Spark",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config=dataframe_config(),
    )

with iv_rv_tab:
    st.subheader("Implied versus realized")
    st.caption(
        "Positive Z means implied premium is rich versus delivered movement. Negative Z means optionality screens cheap versus delivered movement."
    )

    spreads = prepare_display_frame(board[board["Kind"].eq("Spread")].copy())

    if spreads.empty:
        st.info("No IV-RV spreads available with the current ticker pulls and realized-vol windows.")
    else:
        st.plotly_chart(plot_iv_rv_spreads(board), use_container_width=True)

        st.dataframe(
            spreads[
                [
                    "Rank",
                    "Rank 5D Change",
                    "Series",
                    "Signal",
                    "Trade Read",
                    "Level Display",
                    "5D Display",
                    "Z",
                    "Percentile",
                    "Z 20D Change",
                    "Spark",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config=dataframe_config(),
        )

with term_tab:
    st.subheader("Term structure and convexity")
    st.caption("This isolates VIX curve shape, front-end pressure, VVIX/VIX, and SKEW/VIX where Yahoo has usable history.")

    term = prepare_display_frame(board[board["Sleeve"].astype(str).eq("Term Structure")].copy())

    if term.empty:
        st.info("No term-structure series available from the current data pull.")
    else:
        st.dataframe(
            term[
                [
                    "Rank",
                    "Rank 5D Change",
                    "Series",
                    "Signal",
                    "Trade Read",
                    "Level Display",
                    "5D Display",
                    "Z",
                    "Percentile",
                    "Z 20D Change",
                    "Spark",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config=dataframe_config(),
        )

        selected_term = st.multiselect(
            "Select term-structure series",
            list(term["Series"]),
            default=list(term["Series"].head(min(4, len(term)))),
        )

        for start_idx in range(0, len(selected_term), 2):
            cols = st.columns(2)

            for j, name in enumerate(selected_term[start_idx:start_idx + 2]):
                with cols[j]:
                    st.plotly_chart(plot_series_panel(name, histories, ts_obs), use_container_width=True)

with heatmap_tab:
    st.subheader("Z-score heatmap")
    st.plotly_chart(plot_heatmap(board, "Z", Z_DIVERGING, title_color="Z"), use_container_width=True)

    st.subheader("Percentile heatmap")
    st.plotly_chart(
        plot_heatmap(
            board,
            "Percentile",
            PCT_SEQUENTIAL,
            zmin=0,
            zmax=100,
            title_color="Pct",
        ),
        use_container_width=True,
    )

with explorer_tab:
    st.subheader("Series explorer")

    names = list(board.sort_values("Rank")["Series"])
    default_names = names[:min(9, len(names))]

    selected = st.multiselect("Select series", names, default=default_names)
    plot_items = selected if selected else default_names

    cols_per_row = 3

    for start_idx in range(0, len(plot_items), cols_per_row):
        row_items = plot_items[start_idx:start_idx + cols_per_row]
        cols = st.columns(cols_per_row)

        for j in range(cols_per_row):
            with cols[j]:
                if j >= len(row_items):
                    st.empty()
                    continue

                st.plotly_chart(plot_series_panel(row_items[j], histories, ts_obs), use_container_width=True)

with data_tab:
    st.subheader("Data quality")
    st.caption("Stale series are greyed out in the signal logic and excluded from the sleeve regime map after the selected threshold.")

    dq = prepare_display_frame(board.copy())

    st.dataframe(
        dq[
            [
                "Series",
                "Sleeve",
                "Kind",
                "Last Date",
                "Lag vs Latest",
                "Signal",
                "Level Display",
                "Z",
                "Percentile",
            ]
        ].sort_values("Lag vs Latest", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config=dataframe_config(),
    )

    if missing:
        st.info("Missing or unusable source series: " + ", ".join(sorted(set(missing))))
    else:
        st.success("All requested source tickers returned usable data.")

st.caption("© 2026 AD Fund Management LP")
