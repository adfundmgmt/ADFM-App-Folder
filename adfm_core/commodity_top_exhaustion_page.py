from __future__ import annotations

import tempfile
import time
import warnings
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from adfm_core.cftc_positioning import (
    PRICE_PROXIES,
    add_metrics,
    fetch_contract_history,
)
from adfm_core.palette import PASTEL
from adfm_core.ui import (
    PageHeader,
    inject_explorer_style,
    render_footer,
    render_page_header,
    render_sidebar_about,
)

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

try:
    _yf_cache_dir = Path(tempfile.gettempdir()) / "adfm-yfinance-cache"
    _yf_cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(_yf_cache_dir))
except Exception:
    pass

TITLE = "Commodity Event Study"
CACHE_TTL_SECONDS = 3600

COMMODITY_GROUPS: Dict[str, List[Tuple[str, str]]] = {
    "Energy": [
        ("WTI Crude Oil", "CL=F"),
        ("Brent Crude Oil", "BZ=F"),
        ("Natural Gas", "NG=F"),
        ("Heating Oil", "HO=F"),
        ("RBOB Gasoline", "RB=F"),
        ("Mont Belvieu Propane", "B0=F"),
    ],
    "Metals": [
        ("Gold", "GC=F"),
        ("Micro Gold", "MGC=F"),
        ("Silver", "SI=F"),
        ("Micro Silver", "SIL=F"),
        ("Copper", "HG=F"),
        ("Platinum", "PL=F"),
        ("Palladium", "PA=F"),
    ],
    "Grains + Oilseeds": [
        ("Corn", "ZC=F"),
        ("Chicago Wheat", "ZW=F"),
        ("KC HRW Wheat", "KE=F"),
        ("Soybeans", "ZS=F"),
        ("Soybean Meal", "ZM=F"),
        ("Soybean Oil", "ZL=F"),
        ("Oats", "ZO=F"),
        ("Rough Rice", "ZR=F"),
    ],
    "Livestock": [
        ("Live Cattle", "LE=F"),
        ("Feeder Cattle", "GF=F"),
        ("Lean Hogs", "HE=F"),
    ],
    "Softs": [
        ("Cocoa", "CC=F"),
        ("Coffee", "KC=F"),
        ("Sugar #11", "SB=F"),
        ("Cotton", "CT=F"),
        ("Orange Juice", "OJ=F"),
        ("Random Length Lumber", "LBS=F"),
    ],
}

CONTRACT_LABEL_TO_SYMBOL: Dict[str, str] = {}
CONTRACT_SYMBOL_TO_NAME: Dict[str, str] = {}
for _group, _contracts in COMMODITY_GROUPS.items():
    for _name, _symbol in _contracts:
        _label = f"{_group} · {_name} ({_symbol})"
        CONTRACT_LABEL_TO_SYMBOL[_label] = _symbol
        CONTRACT_SYMBOL_TO_NAME[_symbol] = _name

CONTRACT_OPTIONS = list(CONTRACT_LABEL_TO_SYMBOL) + ["Custom Yahoo futures symbol…"]
RETURN_WINDOWS = {"1M": 21, "2M": 42, "3M": 63, "6M": 126, "12M": 252}
FORWARD_HORIZONS = {
    "1D": 1,
    "1W": 5,
    "2W": 10,
    "3W": 15,
    "1M": 21,
    "2M": 42,
    "3M": 63,
    "6M": 126,
    "9M": 189,
    "12M": 252,
}
SPACING_OPTIONS = {"1M": 21, "2M": 42, "3M": 63, "6M": 126, "12M": 252}
LOOKBACK_OPTIONS = {"Max": None, "10Y": 10, "25Y": 25, "50Y": 50}
PROFILE_PRESETS = {
    "Early Warning": {
        "return_pctile": 92.0,
        "trend_z": 1.50,
        "rsi": 70.0,
        "vol_pctile": 65.0,
        "crowding_pctile": 85.0,
        "memory": 8,
        "reversal_components": 0,
    },
    "Confirmed Exhaustion": {
        "return_pctile": 94.0,
        "trend_z": 1.75,
        "rsi": 72.0,
        "vol_pctile": 70.0,
        "crowding_pctile": 88.0,
        "memory": 10,
        "reversal_components": 2,
    },
    "Crowded Blow-Off": {
        "return_pctile": 97.0,
        "trend_z": 2.00,
        "rsi": 75.0,
        "vol_pctile": 80.0,
        "crowding_pctile": 90.0,
        "memory": 10,
        "reversal_components": 2,
    },
}


def _inject_page_style() -> None:
    inject_explorer_style(max_width_px=1540)
    st.markdown(
        """
        <style>
        .event-study-status {
            display:flex; flex-wrap:wrap; gap:.48rem 1.05rem; align-items:center;
            border-top:1px solid #d7d7d7; border-bottom:1px solid #d7d7d7;
            margin:.18rem 0 .55rem; padding:.55rem 0; color:#222;
            font-family:Arial,Helvetica,sans-serif; font-size:.78rem; line-height:1.35;
        }
        .event-study-status strong { color:#000; font-weight:800; }
        .event-study-table-title {
            margin:1.25rem 0 .65rem; color:#000; font-family:Georgia,"Times New Roman",serif;
            font-size:1.25rem; font-weight:700; letter-spacing:-.018em; line-height:1.20;
        }
        .event-study-table-wrap {
            width:100%; overflow-x:auto; -webkit-overflow-scrolling:touch;
            margin:.15rem 0 .55rem; border:1px solid #aeb7bd; background:#fff;
        }
        table.event-study-table {
            width:100%; min-width:1080px; border-collapse:collapse; table-layout:fixed;
            font-family:Arial,Helvetica,sans-serif; font-size:.72rem; line-height:1.15;
        }
        table.event-study-table col.metric-col { width:205px; }
        table.event-study-table col.horizon-col { width:87px; }
        table.event-study-table th, table.event-study-table td {
            box-sizing:border-box; border-right:1px solid #aeb7bd; border-bottom:1px solid #aeb7bd;
            padding:.42rem .32rem; text-align:center; vertical-align:middle; white-space:nowrap;
            overflow:hidden; text-overflow:clip;
        }
        table.event-study-table th:last-child, table.event-study-table td:last-child { border-right:0; }
        table.event-study-table tr:last-child td { border-bottom:0; }
        table.event-study-table thead th { background:#357f8d; color:#fff; font-weight:800; }
        table.event-study-table thead th.metric-head { text-align:left; padding-left:.72rem; }
        table.event-study-table tbody td.metric {
            background:#edf0f2; color:#111; font-weight:800; text-align:left; padding-left:.72rem;
        }
        table.event-study-table tbody td.good { background:#dce9e1; color:#111; }
        table.event-study-table tbody td.bad { background:#edc9cd; color:#111; }
        table.event-study-table tbody td.neutral { background:#e7edf1; color:#111; }
        .event-study-caption {
            color:#666; font-family:Arial,Helvetica,sans-serif; font-size:.73rem;
            line-height:1.42; margin:.18rem 0 .80rem;
        }
        @media (max-width:800px) {
            .event-study-status { gap:.38rem .70rem; font-size:.73rem; }
            .event-study-table-title { font-size:1.10rem; }
            table.event-study-table { min-width:1035px; font-size:.70rem; }
            table.event-study-table col.metric-col { width:190px; }
            table.event-study-table col.horizon-col { width:84px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(profile: str, settings: dict, crowding_source: str | None = None) -> None:
    with st.sidebar:
        render_sidebar_about("25_Commodity_Event_Study.py")
        st.subheader("Current definition")
        st.caption(
            f"{profile} · return ≥ {settings['return_pctile']:.0f}th pct · "
            f"trend ≥ {settings['trend_z']:.2f}σ · RSI ≥ {settings['rsi']:.0f} · "
            f"vol ≥ {settings['vol_pctile']:.0f}th pct"
        )
        if crowding_source:
            st.caption(f"Crowding input: {crowding_source}")


def _flatten_yfinance_columns(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame
    for level in range(frame.columns.nlevels):
        values = frame.columns.get_level_values(level)
        if symbol in values:
            try:
                return frame.xs(symbol, axis=1, level=level, drop_level=True)
            except Exception:
                pass
    out = frame.copy()
    out.columns = out.columns.get_level_values(0)
    return out


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_contract_history(symbol: str) -> pd.DataFrame:
    symbol = str(symbol).strip().upper()
    last_error = None
    for attempt in range(3):
        try:
            frame = yf.download(
                symbol,
                period="max",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if frame is None or frame.empty:
                raise ValueError("Yahoo Finance returned no rows.")
            frame = _flatten_yfinance_columns(frame, symbol)
            if "Close" not in frame.columns:
                raise ValueError("Yahoo Finance returned no Close column.")
            close = pd.to_numeric(frame["Close"], errors="coerce")
            volume = (
                pd.to_numeric(frame["Volume"], errors="coerce")
                if "Volume" in frame.columns
                else pd.Series(index=frame.index, dtype=float)
            )
            out = pd.DataFrame({"Close": close, "Volume": volume})
            out.index = pd.to_datetime(out.index)
            if getattr(out.index, "tz", None) is not None:
                out.index = out.index.tz_localize(None)
            out = out[~out.index.duplicated(keep="last")].sort_index()
            out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Close"])
            if len(out) < 260:
                raise ValueError("Insufficient daily history for an event study.")
            return out
        except Exception as exc:
            last_error = exc
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Could not load {symbol}: {last_error}")


def _reverse_cftc_proxy_map() -> Dict[str, str]:
    return {str(values[0]).upper(): str(code) for code, values in PRICE_PROXIES.items()}


@st.cache_data(ttl=21600, show_spinner=False)
def load_cftc_crowding(symbol: str) -> Tuple[pd.Series, str]:
    code = _reverse_cftc_proxy_map().get(str(symbol).upper())
    if not code:
        return pd.Series(dtype=float), ""
    try:
        raw = fetch_contract_history("Disaggregated", code)
        if raw is None or raw.empty:
            return pd.Series(dtype=float), ""
        metrics = add_metrics(raw, "Disaggregated", "Managed Money")
        weekly = metrics[["report_date", "net_pct_oi"]].dropna().sort_values("report_date")
        if len(weekly) < 52:
            return pd.Series(dtype=float), ""
        weekly["crowding_pctile"] = (
            weekly["net_pct_oi"].rolling(156, min_periods=52).rank(pct=True) * 100.0
        )
        # COT Tuesday positions are normally published Friday. Shift availability to Friday
        # so the event study does not use the Tuesday observation before it was public.
        release_date = pd.to_datetime(weekly["report_date"]) + pd.Timedelta(days=3)
        series = pd.Series(weekly["crowding_pctile"].to_numpy(), index=release_date)
        series = series[~series.index.duplicated(keep="last")].sort_index()
        return series, "CFTC Managed Money · 3Y percentile"
    except Exception:
        return pd.Series(dtype=float), ""


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    out = out.mask((avg_gain == 0.0) & (avg_loss > 0.0), 0.0)
    return out.clip(lower=0.0, upper=100.0)


def _rolling_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    return clean.rolling(window, min_periods=min_periods).rank(pct=True) * 100.0


def build_exhaustion_frame(
    data: pd.DataFrame,
    symbol: str,
    profile: str,
    settings: dict,
    return_days: int,
) -> Tuple[pd.DataFrame, pd.Series, str, str]:
    close = pd.to_numeric(data["Close"], errors="coerce").astype(float)
    volume = pd.to_numeric(data["Volume"], errors="coerce").astype(float)
    positive_close = close.where(close > 0.0)

    lag = close.shift(return_days)
    period_return = close / lag - 1.0
    period_return = period_return.mask((close <= 0.0) | (lag <= 0.0))
    return_pctile = _rolling_percentile(period_return, 1260, 252)

    ma200 = close.rolling(200, min_periods=160).mean()
    std200 = close.rolling(200, min_periods=160).std(ddof=0)
    trend_z = (close - ma200) / std200.replace(0.0, np.nan)
    rsi14 = _rsi(close, 14)

    log_return = np.log(positive_close / positive_close.shift(1))
    realized_vol = log_return.rolling(20, min_periods=15).std(ddof=0) * np.sqrt(252.0)
    vol_pctile = _rolling_percentile(realized_vol, 756, 126)
    volume_pctile = _rolling_percentile(volume.replace(0.0, np.nan), 756, 126)

    cftc_weekly, cftc_label = load_cftc_crowding(symbol)
    if not cftc_weekly.empty:
        union_index = cftc_weekly.index.union(close.index)
        cftc_daily = cftc_weekly.reindex(union_index).sort_index().ffill().reindex(close.index)
        crowding_pctile = cftc_daily
        crowding_source = cftc_label
    else:
        crowding_pctile = volume_pctile
        crowding_source = "Volume intensity fallback"

    ma10 = close.rolling(10, min_periods=8).mean()
    ret5 = close / close.shift(5) - 1.0
    prior5_low = close.shift(1).rolling(5, min_periods=5).min()
    reversal_score = (
        (ret5 < 0.0).astype(int)
        + (close < ma10).astype(int)
        + (close < prior5_low).astype(int)
    )

    ret_extreme = return_pctile >= float(settings["return_pctile"])
    trend_extreme = trend_z >= float(settings["trend_z"])
    rsi_extreme = rsi14 >= float(settings["rsi"])
    vol_extreme = vol_pctile >= float(settings["vol_pctile"])
    crowd_extreme = crowding_pctile >= float(settings["crowding_pctile"])

    core_count = trend_extreme.astype(int) + rsi_extreme.astype(int) + vol_extreme.astype(int)
    early_setup = ret_extreme & (core_count >= 2)

    if profile == "Early Warning":
        condition = early_setup
        label = (
            f"Early warning: {return_days}D return ≥ {settings['return_pctile']:.0f}th pct "
            "+ 2/3 trend, RSI, vol extremes"
        )
    elif profile == "Confirmed Exhaustion":
        recent_setup = early_setup.rolling(int(settings["memory"]), min_periods=1).max().astype(bool)
        condition = recent_setup & (reversal_score >= int(settings["reversal_components"]))
        label = (
            f"Confirmed exhaustion: recent extreme + {int(settings['reversal_components'])}/3 reversal checks"
        )
    else:
        strong_count = (
            trend_extreme.astype(int)
            + rsi_extreme.astype(int)
            + vol_extreme.astype(int)
            + crowd_extreme.astype(int)
        )
        blowoff_setup = ret_extreme & (strong_count >= 3) & crowd_extreme
        recent_setup = blowoff_setup.rolling(int(settings["memory"]), min_periods=1).max().astype(bool)
        condition = recent_setup & (reversal_score >= int(settings["reversal_components"]))
        label = (
            f"Crowded blow-off: extreme tape + {crowding_source.lower()} + reversal"
        )

    frame = pd.DataFrame(
        {
            "Close": close,
            "Return": period_return,
            "ReturnPctile": return_pctile,
            "TrendZ": trend_z,
            "RSI": rsi14,
            "RealizedVol": realized_vol,
            "VolPctile": vol_pctile,
            "VolumePctile": volume_pctile,
            "CrowdingPctile": crowding_pctile,
            "MA10": ma10,
            "Ret5": ret5,
            "Prior5Low": prior5_low,
            "ReversalScore": reversal_score,
            "EarlySetup": early_setup.astype(bool),
            "Signal": condition.fillna(False).astype(bool),
        },
        index=close.index,
    )
    return frame, condition.fillna(False).astype(bool), label, crowding_source


def detect_events(condition: pd.Series, spacing_days: int) -> pd.DatetimeIndex:
    condition = condition.fillna(False).astype(bool)
    candidates = condition & ~condition.shift(1, fill_value=False)
    positions = np.flatnonzero(candidates.to_numpy())
    kept: List[int] = []
    last_position = -10**9
    for position in positions:
        if position - last_position >= spacing_days:
            kept.append(int(position))
            last_position = int(position)
    return pd.DatetimeIndex(condition.index[kept])


def build_event_observations(
    close: pd.Series,
    events: Iterable[pd.Timestamp],
    diagnostics: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    close = close.dropna().astype(float)
    positions = {timestamp: i for i, timestamp in enumerate(close.index)}
    rows: List[dict] = []
    store: Dict[str, Dict[str, List[float]]] = {
        label: {"return": [], "signal_dd": [], "path_dd": [], "upside": []}
        for label in FORWARD_HORIZONS
    }

    for event_date in events:
        if event_date not in positions:
            continue
        start_pos = positions[event_date]
        start_price = float(close.iloc[start_pos])
        diag = diagnostics.reindex([event_date]).iloc[0]
        row = {
            "Date": event_date,
            "Price": start_price,
            "ReturnPctile": diag.get("ReturnPctile", np.nan),
            "TrendZ": diag.get("TrendZ", np.nan),
            "RSI": diag.get("RSI", np.nan),
            "VolPctile": diag.get("VolPctile", np.nan),
            "CrowdingPctile": diag.get("CrowdingPctile", np.nan),
            "ReversalScore": diag.get("ReversalScore", np.nan),
        }

        if start_pos + 21 < len(close):
            local_start = max(0, start_pos - 21)
            local_end = min(len(close), start_pos + 22)
            local_path = close.iloc[local_start:local_end]
            peak_date = local_path.idxmax()
            peak_pos = positions[peak_date]
            peak_price = float(local_path.max())
            row["DaysFromLocalPeak"] = float(start_pos - peak_pos)
            row["PeakToSignal"] = start_price / peak_price - 1.0
        else:
            row["DaysFromLocalPeak"] = np.nan
            row["PeakToSignal"] = np.nan

        for label, horizon in FORWARD_HORIZONS.items():
            end_pos = start_pos + horizon
            if end_pos >= len(close):
                row[label] = np.nan
                continue
            path = close.iloc[start_pos : end_pos + 1]
            end_return = float(path.iloc[-1] / start_price - 1.0)
            from_signal = path / start_price - 1.0
            running_peak = path.cummax()
            path_dd = path / running_peak - 1.0
            row[label] = end_return
            store[label]["return"].append(end_return)
            store[label]["signal_dd"].append(float(from_signal.min()))
            store[label]["path_dd"].append(float(path_dd.min()))
            store[label]["upside"].append(float(from_signal.max()))
        rows.append(row)

    history = pd.DataFrame(rows)
    arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for label, metrics in store.items():
        arrays[label] = {
            metric: np.asarray(values, dtype=float) for metric, values in metrics.items()
        }
    return history, arrays


def summarize_forward_performance(arrays: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    metrics = [
        "Average",
        "Median",
        "Best for Top Signal",
        "Worst for Top Signal",
        "% Negative",
        "Avg DD From Signal",
        "Worst DD From Signal",
        "Avg Peak-to-Trough DD",
        "Sample",
    ]
    summary = pd.DataFrame(index=metrics, columns=list(FORWARD_HORIZONS), dtype=float)
    for horizon in FORWARD_HORIZONS:
        returns = arrays[horizon]["return"]
        signal_dd = arrays[horizon]["signal_dd"]
        path_dd = arrays[horizon]["path_dd"]
        if returns.size == 0:
            continue
        summary.loc["Average", horizon] = np.mean(returns)
        summary.loc["Median", horizon] = np.median(returns)
        summary.loc["Best for Top Signal", horizon] = np.min(returns)
        summary.loc["Worst for Top Signal", horizon] = np.max(returns)
        summary.loc["% Negative", horizon] = np.mean(returns < 0.0)
        summary.loc["Avg DD From Signal", horizon] = np.mean(signal_dd)
        summary.loc["Worst DD From Signal", horizon] = np.min(signal_dd)
        summary.loc["Avg Peak-to-Trough DD", horizon] = np.mean(path_dd)
        summary.loc["Sample", horizon] = float(returns.size)
    return summary


def _top_hit_rate(values: np.ndarray, threshold: float, direction: str) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return np.nan
    if direction == "down":
        return float(np.mean(clean <= -abs(threshold)))
    return float(np.mean(clean >= abs(threshold)))


def top_diagnostics(history: pd.DataFrame, arrays: Dict[str, Dict[str, np.ndarray]]) -> dict:
    days = pd.to_numeric(history.get("DaysFromLocalPeak", pd.Series(dtype=float)), errors="coerce").dropna()
    peak_move = pd.to_numeric(history.get("PeakToSignal", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "days_from_peak": float(days.median()) if not days.empty else np.nan,
        "peak_to_signal": float(peak_move.median()) if not peak_move.empty else np.nan,
        "down10_3m": _top_hit_rate(arrays["3M"]["signal_dd"], 0.10, "down"),
        "down20_6m": _top_hit_rate(arrays["6M"]["signal_dd"], 0.20, "down"),
        "up10_3m": _top_hit_rate(arrays["3M"]["upside"], 0.10, "up"),
    }


def make_price_chart(
    close: pd.Series,
    event_dates: pd.DatetimeIndex,
    diagnostics: pd.DataFrame,
    signal_label: str,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=close.index,
            y=close.values,
            mode="lines",
            line={"color": PASTEL["blue"], "width": 1.65},
            hovertemplate="%{x|%b %d, %Y}<br>Price: %{y:,.2f}<extra></extra>",
            name="Price",
        )
    )
    if len(event_dates):
        event_prices = close.reindex(event_dates)
        custom = diagnostics.reindex(event_dates)[
            ["ReturnPctile", "TrendZ", "RSI", "VolPctile", "CrowdingPctile", "ReversalScore"]
        ].to_numpy()
        figure.add_trace(
            go.Scatter(
                x=event_dates,
                y=event_prices.values,
                mode="markers",
                marker={
                    "color": PASTEL["rose"],
                    "size": 8,
                    "line": {"color": "#ffffff", "width": 0.7},
                },
                customdata=custom,
                hovertemplate=(
                    "%{x|%b %d, %Y}<br>Price: %{y:,.2f}"
                    "<br>Return pctile: %{customdata[0]:.2f}"
                    "<br>Trend Z: %{customdata[1]:.2f}"
                    "<br>RSI: %{customdata[2]:.2f}"
                    "<br>Vol pctile: %{customdata[3]:.2f}"
                    "<br>Crowding pctile: %{customdata[4]:.2f}"
                    "<br>Reversal score: %{customdata[5]:.0f}/3"
                    "<extra></extra>"
                ),
                name=signal_label,
            )
        )
    figure.update_layout(
        height=500,
        margin={"l": 8, "r": 8, "t": 12, "b": 8},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=False,
        hovermode="closest",
        font={"family": "Arial, Helvetica, sans-serif", "color": "#1b1b1b", "size": 12},
    )
    figure.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor="#8d8d8d",
        linewidth=1,
        ticks="outside",
        tickcolor="#8d8d8d",
        tickformat="%Y",
        fixedrange=False,
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor="#e3e8ec",
        gridwidth=1,
        zeroline=False,
        showline=False,
        tickformat=",.2f",
        title=None,
        fixedrange=False,
    )
    return figure


def _cell_class(metric: str, value: float) -> str:
    if metric == "Sample" or pd.isna(value):
        return "neutral"
    if metric == "% Negative":
        return "good" if value >= 0.5 else "bad"
    if "DD" in metric:
        return "good"
    return "good" if value <= 0.0 else "bad"


def _format_summary_value(metric: str, value: float) -> str:
    if pd.isna(value):
        return "—"
    if metric == "Sample":
        return f"{int(round(value))}"
    if metric == "% Negative":
        return f"{value * 100.0:.2f}%"
    return f"{value * 100.0:+.2f}%"


def summary_table_html(summary: pd.DataFrame) -> str:
    columns = list(summary.columns)
    head = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body_rows = []
    for metric, row in summary.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            cells.append(
                f"<td class='{_cell_class(metric, value)}'>{_format_summary_value(metric, value)}</td>"
            )
        body_rows.append(
            "<tr>" f"<td class='metric'>{escape(str(metric))}</td>" + "".join(cells) + "</tr>"
        )
    colgroup = "<col class='metric-col'>" + "".join(
        "<col class='horizon-col'>" for _ in columns
    )
    return (
        "<div class='event-study-table-wrap'><table class='event-study-table'>"
        f"<colgroup>{colgroup}</colgroup>"
        f"<thead><tr><th class='metric-head'>Metric</th>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def _format_days_from_peak(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    days = int(round(abs(value)))
    if abs(value) < 0.5:
        return "at peak"
    return f"{days}d after" if value > 0 else f"{days}d before"


def _history_display(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    display = history.copy().sort_values("Date", ascending=False)
    display["Date"] = pd.to_datetime(display["Date"]).dt.strftime("%Y-%m-%d")
    display["Price"] = display["Price"].map(lambda x: f"{x:,.2f}")
    for column in ["ReturnPctile", "RSI", "VolPctile", "CrowdingPctile"]:
        display[column] = display[column].map(
            lambda x: "—" if pd.isna(x) else f"{float(x):.2f}"
        )
    display["TrendZ"] = display["TrendZ"].map(
        lambda x: "—" if pd.isna(x) else f"{float(x):.2f}"
    )
    display["ReversalScore"] = display["ReversalScore"].map(
        lambda x: "—" if pd.isna(x) else f"{int(x)}/3"
    )
    display["DaysFromLocalPeak"] = display["DaysFromLocalPeak"].map(
        lambda x: _format_days_from_peak(float(x)) if pd.notna(x) else "—"
    )
    display["PeakToSignal"] = display["PeakToSignal"].map(
        lambda x: "—" if pd.isna(x) else f"{float(x) * 100.0:+.2f}%"
    )
    for column in FORWARD_HORIZONS:
        display[column] = display[column].map(
            lambda x: "—" if pd.isna(x) else f"{float(x) * 100.0:+.2f}%"
        )
    return display


def _settings_controls(profile: str) -> Tuple[dict, str]:
    preset = dict(PROFILE_PRESETS[profile])
    with st.sidebar:
        with st.expander("Advanced thresholds", expanded=False):
            customize = st.checkbox("Customize preset", value=False)
            if customize:
                preset["return_pctile"] = st.slider(
                    "Return percentile", 80.0, 99.5, float(preset["return_pctile"]), 0.5
                )
                preset["trend_z"] = st.slider(
                    "Trend extension (σ)", 0.5, 4.0, float(preset["trend_z"]), 0.25
                )
                preset["rsi"] = st.slider("RSI threshold", 55.0, 90.0, float(preset["rsi"]), 1.0)
                preset["vol_pctile"] = st.slider(
                    "Realized-vol percentile", 50.0, 99.0, float(preset["vol_pctile"]), 1.0
                )
                preset["crowding_pctile"] = st.slider(
                    "Crowding percentile", 70.0, 99.0, float(preset["crowding_pctile"]), 1.0
                )
                preset["memory"] = st.slider(
                    "Setup memory (sessions)", 3, 20, int(preset["memory"]), 1
                )
                if profile != "Early Warning":
                    preset["reversal_components"] = st.slider(
                        "Reversal checks required", 1, 3, int(preset["reversal_components"]), 1
                    )
            return_window = st.selectbox("Extension return window", list(RETURN_WINDOWS), index=2)
    return preset, return_window


def render_commodity_event_study() -> None:
    _inject_page_style()
    render_page_header(
        PageHeader(
            title=TITLE,
            description=(
                "Find historically extended commodity moves, wait for exhaustion or reversal, "
                "and measure how often the signal actually identified a top."
            ),
            eyebrow="ADFM Historical Context",
            source_note="Yahoo Finance continuous futures · CFTC positioning where mapped",
        )
    )

    controls = st.columns([2.05, 1.35, 0.85, 0.85], gap="small")
    with controls[0]:
        contract_choice = st.selectbox("Commodity future", CONTRACT_OPTIONS, index=0)
    with controls[1]:
        profile = st.selectbox(
            "Top signal",
            ["Early Warning", "Confirmed Exhaustion", "Crowded Blow-Off"],
            index=1,
        )
    with controls[2]:
        spacing_label = st.selectbox("Event spacing", list(SPACING_OPTIONS), index=2)
    with controls[3]:
        lookback_label = st.selectbox("Lookback", list(LOOKBACK_OPTIONS), index=0)

    if contract_choice == "Custom Yahoo futures symbol…":
        symbol = st.text_input(
            "Yahoo futures symbol",
            value="CL=F",
            help="Use a Yahoo Finance futures ticker, usually ending in =F.",
        ).strip().upper()
        contract_name = symbol or "Custom contract"
    else:
        symbol = CONTRACT_LABEL_TO_SYMBOL[contract_choice]
        contract_name = CONTRACT_SYMBOL_TO_NAME.get(symbol, symbol)

    settings, return_window = _settings_controls(profile)
    return_days = RETURN_WINDOWS[return_window]

    if not symbol:
        st.warning("Enter a Yahoo Finance futures symbol.")
        st.stop()

    try:
        with st.spinner(f"Loading {symbol} history…"):
            data = load_contract_history(symbol)
            diagnostics, condition, signal_label, crowding_source = build_exhaustion_frame(
                data, symbol, profile, settings, return_days
            )
    except Exception as exc:
        st.error(str(exc))
        render_footer(
            data_note=(
                "Yahoo Finance continuous-futures histories are provider-supplied stitched series. "
                "Availability and roll construction vary by contract."
            )
        )
        return

    _render_sidebar(profile, settings, crowding_source)

    close_full = data["Close"].dropna().astype(float)
    all_events = detect_events(condition, SPACING_OPTIONS[spacing_label])

    lookback_years = LOOKBACK_OPTIONS[lookback_label]
    if lookback_years is None:
        chart_close = close_full.copy()
    else:
        chart_start = close_full.index.max() - pd.DateOffset(years=int(lookback_years))
        chart_close = close_full.loc[close_full.index >= chart_start].copy()
    if chart_close.empty:
        st.error("No price history is available inside the selected lookback.")
        return

    study_events = all_events[all_events >= chart_close.index.min()]
    history, arrays = build_event_observations(close_full, study_events, diagnostics)
    summary = summarize_forward_performance(arrays)
    top_stats = top_diagnostics(history, arrays)

    latest_date = close_full.index.max()
    latest = diagnostics.reindex([latest_date]).iloc[0]
    latest_event = study_events.max() if len(study_events) else None
    latest_event_text = latest_event.strftime("%b %d, %Y") if latest_event is not None else "None"
    current_signal = bool(condition.reindex([latest_date]).fillna(False).iloc[0])
    recent_setup = bool(
        diagnostics["EarlySetup"].tail(int(settings["memory"])).fillna(False).any()
    )
    current_state = "ACTIVE" if current_signal else "Setup / waiting for reversal" if recent_setup else "Normal"
    crowding_value = latest.get("CrowdingPctile", np.nan)
    crowding_text = "n/a" if pd.isna(crowding_value) else f"{float(crowding_value):.2f}th pct"

    st.markdown(
        (
            "<div class='event-study-status'>"
            f"<span><strong>{escape(contract_name)}</strong> · {escape(symbol)}</span>"
            f"<span><strong>Signal</strong> {escape(profile)}</span>"
            f"<span><strong>Current</strong> {escape(current_state)}</span>"
            f"<span><strong>63D pctile</strong> {float(latest.get('ReturnPctile', np.nan)):.2f}</span>"
            f"<span><strong>Trend Z</strong> {float(latest.get('TrendZ', np.nan)):.2f}</span>"
            f"<span><strong>RSI</strong> {float(latest.get('RSI', np.nan)):.2f}</span>"
            f"<span><strong>Crowding</strong> {escape(crowding_text)}</span>"
            f"<span><strong>Events</strong> {len(study_events)}</span>"
            f"<span><strong>Latest event</strong> {escape(latest_event_text)}</span>"
            f"<span><strong>Data through</strong> {latest_date:%b %d, %Y}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    figure = make_price_chart(chart_close, study_events, diagnostics, signal_label)
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    st.markdown("<div class='event-study-table-title'>Top-Picking Diagnostics</div>", unsafe_allow_html=True)
    diag_cols = st.columns(5, gap="small")
    diag_cols[0].metric("Median signal timing", _format_days_from_peak(top_stats["days_from_peak"]))
    diag_cols[1].metric(
        "Median peak → signal",
        "—" if not np.isfinite(top_stats["peak_to_signal"]) else f"{top_stats['peak_to_signal'] * 100.0:+.2f}%",
    )
    diag_cols[2].metric(
        "3M ≥10% decline",
        "—" if not np.isfinite(top_stats["down10_3m"]) else f"{top_stats['down10_3m'] * 100.0:.2f}%",
    )
    diag_cols[3].metric(
        "6M ≥20% decline",
        "—" if not np.isfinite(top_stats["down20_6m"]) else f"{top_stats['down20_6m'] * 100.0:.2f}%",
    )
    diag_cols[4].metric(
        "3M >10% further upside",
        "—" if not np.isfinite(top_stats["up10_3m"]) else f"{top_stats['up10_3m'] * 100.0:.2f}%",
    )
    st.markdown(
        "<div class='event-study-caption'>Local peak timing is measured against the highest close in the 21 sessions before through 21 sessions after each signal. Further-upside rate is a direct false-positive check.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='event-study-table-title'>{escape(contract_name)} Forward Performance After Top Signal</div>",
        unsafe_allow_html=True,
    )
    st.markdown(summary_table_html(summary), unsafe_allow_html=True)
    st.markdown(
        (
            "<div class='event-study-caption'>"
            "Green means the top signal worked: negative forward return or a meaningful post-signal drawdown. "
            "Best/Worst are defined from the perspective of a top signal. Sample falls at longer horizons when recent events have not matured."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    with st.expander("Historical top signals"):
        if history.empty:
            st.info("No events met the selected definition in this lookback.")
        else:
            display = _history_display(history)
            st.dataframe(display, width="stretch", hide_index=True, height=420)
            csv_bytes = history.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download signal history CSV",
                data=csv_bytes,
                file_name=f"{symbol.replace('=', '_')}_commodity_top_study.csv",
                mime="text/csv",
            )

    with st.expander("Method and data caveat"):
        st.markdown(
            """
            - Return percentiles use the selected return window ranked against up to five years of trailing daily observations. The current observation is included only with information available that day.
            - Trend extension is the close versus its 200-day moving average, scaled by the rolling 200-day price standard deviation.
            - Realized volatility uses 20-day annualized log-return volatility ranked against a trailing three-year distribution.
            - Confirmed Exhaustion requires an extreme setup within the recent setup-memory window, then reversal confirmation from negative 5-day return, close below the 10-day moving average, and close below the prior 5-session low.
            - CFTC Managed Money positioning is used only for mapped contracts and is shifted from Tuesday report date to estimated Friday public availability to reduce look-ahead bias. Unmapped contracts use volume intensity only for the Crowded Blow-Off profile.
            - Yahoo Finance futures histories are continuous provider series. Contract-roll methodology can create discontinuities, so this is historical tendency research rather than execution-grade roll attribution.
            """
        )

    render_footer(
        data_note=(
            "Primary input: Yahoo Finance daily continuous-futures history. CFTC Disaggregated Futures Only positioning is used where mapped. "
            "Historical tendency only; event studies are descriptive and not forecasts."
        )
    )
