from __future__ import annotations

import tempfile
import time
import warnings
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
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
BOOTSTRAP_DRAWS = 2000
CFTC_MAX_STALE_SESSIONS = 7
US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

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
        "memory": 8,
        "reversal_components": 0,
    },
    "Confirmed Exhaustion": {
        "return_pctile": 94.0,
        "trend_z": 1.75,
        "rsi": 72.0,
        "vol_pctile": 70.0,
        "memory": 10,
        "reversal_components": 2,
    },
    "Failed Breakout": {
        "breakout_days": 63,
        "memory": 10,
    },
}

CFTC_CONTRACT_CODES: Dict[str, str] = {
    str(values[0]).upper(): str(code) for code, values in PRICE_PROXIES.items()
}
CFTC_CONTRACT_CODES.update(
    {
        "CL=F": "067651",
        "BZ=F": "06765T",
        "NG=F": "023651",
        "HO=F": "022651",
        "RB=F": "111659",
        "GC=F": "088691",
        "MGC=F": "088691",
        "SI=F": "084691",
        "SIL=F": "084691",
        "HG=F": "085692",
        "PL=F": "076651",
        "PA=F": "075651",
        "ZC=F": "002602",
        "ZW=F": "001602",
        "KE=F": "001612",
        "ZS=F": "005602",
        "ZM=F": "026603",
        "ZL=F": "007601",
        "ZR=F": "039601",
        "LE=F": "057642",
        "GF=F": "061641",
        "HE=F": "054642",
        "CC=F": "073732",
        "KC=F": "083731",
        "SB=F": "080732",
        "CT=F": "033661",
        "OJ=F": "040701",
    }
)

CFTC_PUBLICATION_OVERRIDES: Dict[str, str] = {
    "2020-12-21": "2020-12-28",
    "2021-06-15": "2021-06-21",
    "2023-01-31": "2023-02-24",
    "2023-02-07": "2023-03-03",
    "2023-02-14": "2023-03-08",
    "2023-02-21": "2023-03-10",
    "2023-02-28": "2023-03-14",
    "2023-03-07": "2023-03-16",
    "2023-03-14": "2023-03-21",
    "2025-01-07": "2025-01-13",
    "2025-09-30": "2025-11-19",
    "2025-10-07": "2025-11-21",
    "2025-10-14": "2025-11-25",
    "2025-10-21": "2025-12-02",
    "2025-10-28": "2025-12-05",
    "2025-11-04": "2025-12-10",
    "2025-11-10": "2025-12-10",
    "2025-11-18": "2025-12-12",
    "2025-11-25": "2025-12-15",
    "2025-12-02": "2025-12-17",
    "2025-12-09": "2025-12-19",
    "2025-12-16": "2025-12-23",
    "2025-12-23": "2025-12-29",
}
CFTC_EXCLUDED_REPORT_RANGES = ((pd.Timestamp("2018-12-24"), pd.Timestamp("2019-02-26")),)


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
        if profile == "Failed Breakout":
            st.caption(
                f"Close above the prior {settings['breakout_days']}D closing high, then "
                f"close below that original level and the 10D average within "
                f"{settings['memory']} sessions. First failure per breakout."
            )
        else:
            st.caption(
                f"{profile} · return ≥ {settings['return_pctile']:.0f}th pct · "
                f"trend ≥ {settings['trend_z']:.2f} vol units · RSI ≥ {settings['rsi']:.0f} · "
                f"vol ≥ {settings['vol_pctile']:.0f}th pct"
            )
        if crowding_source == "CFTC unavailable":
            st.caption("Crowding: CFTC unavailable; price signals remain available.")
        elif crowding_source:
            st.caption(f"Crowding context only: {crowding_source}")


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


def _report_date_is_excluded(report_date: pd.Timestamp) -> bool:
    date = pd.Timestamp(report_date).normalize()
    return any(start <= date <= end for start, end in CFTC_EXCLUDED_REPORT_RANGES)


def cftc_publication_date(report_date: pd.Timestamp) -> pd.Timestamp:
    date = pd.Timestamp(report_date).normalize()
    if _report_date_is_excluded(date):
        return pd.NaT
    override = CFTC_PUBLICATION_OVERRIDES.get(date.strftime("%Y-%m-%d"))
    if override:
        return pd.Timestamp(override)
    return date + pd.Timedelta(days=3)


def cftc_availability_date(report_date: pd.Timestamp) -> pd.Timestamp:
    publication = cftc_publication_date(report_date)
    if pd.isna(publication):
        return pd.NaT
    return pd.Timestamp(publication + US_BUSINESS_DAY).normalize()


@st.cache_data(ttl=21600, show_spinner=False)
def load_cftc_crowding(symbol: str) -> Tuple[pd.Series, str]:
    code = CFTC_CONTRACT_CODES.get(str(symbol).upper())
    if not code:
        return pd.Series(dtype=float), "CFTC unavailable"
    try:
        raw = fetch_contract_history("Disaggregated", code)
        if raw is None or raw.empty:
            return pd.Series(dtype=float), "CFTC unavailable"
        metrics = add_metrics(raw, "Disaggregated", "Managed Money")
        weekly = metrics[["report_date", "net_pct_oi"]].dropna().sort_values("report_date")
        if len(weekly) < 52:
            return pd.Series(dtype=float), "CFTC unavailable"
        weekly["crowding_pctile"] = (
            weekly["net_pct_oi"].rolling(156, min_periods=52).rank(pct=True) * 100.0
        )
        weekly["availability_date"] = weekly["report_date"].map(cftc_availability_date)
        weekly = weekly.dropna(subset=["availability_date", "crowding_pctile"])
        if weekly.empty:
            return pd.Series(dtype=float), "CFTC unavailable"
        series = pd.Series(
            weekly["crowding_pctile"].to_numpy(dtype=float),
            index=pd.to_datetime(weekly["availability_date"]),
            dtype=float,
        )
        series = series[~series.index.duplicated(keep="last")].sort_index()
        return series, "CFTC Managed Money · 3Y percentile · next-session availability"
    except Exception:
        return pd.Series(dtype=float), "CFTC unavailable"


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


def _align_cftc_to_prices(cftc: pd.Series, close_index: pd.DatetimeIndex) -> pd.Series:
    if cftc.empty:
        return pd.Series(np.nan, index=close_index, dtype=float)
    union_index = cftc.index.union(close_index)
    aligned = cftc.reindex(union_index).sort_index().ffill(limit=CFTC_MAX_STALE_SESSIONS)
    return aligned.reindex(close_index).astype(float)


def failed_breakout_frame(close: pd.Series, breakout_days: int, memory: int) -> pd.DataFrame:
    """Track each breakout's fixed level, using only closes known at that session."""
    prior_high = close.shift(1).rolling(breakout_days, min_periods=breakout_days).max()
    setup = close > prior_high
    ma10 = close.rolling(10, min_periods=10).mean()
    signal = pd.Series(False, index=close.index)
    pending = pd.Series(False, index=close.index)
    level = pd.Series(np.nan, index=close.index)
    active: List[Tuple[int, float]] = []
    for position, price in enumerate(close.to_numpy(dtype=float)):
        active = [(start, threshold) for start, threshold in active if position - start <= memory]
        remaining = []
        for start, threshold in active:
            if price < threshold and price < ma10.iloc[position]:
                signal.iloc[position] = True
                level.iloc[position] = threshold
            else:
                remaining.append((start, threshold))
        active = remaining
        # A breakout cannot fail on its own setup day. New highs do not reset older levels.
        if setup.iloc[position]:
            active.append((position, float(prior_high.iloc[position])))
        # After the tenth close, an unfailed setup has no remaining confirmation window.
        active = [(start, threshold) for start, threshold in active if position - start < memory]
        pending.iloc[position] = bool(active)
        if active and not signal.iloc[position]:
            level.iloc[position] = active[-1][1]
    return pd.DataFrame({"ProfileSetup": setup, "BreakoutPending": pending,
                         "BreakoutLevel": level, "Signal": signal}, index=close.index)


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

    log_return = np.log(positive_close / positive_close.shift(1))
    daily_vol = log_return.rolling(20, min_periods=15).std(ddof=0)
    realized_vol = daily_vol * np.sqrt(252.0)
    vol_pctile = _rolling_percentile(realized_vol, 756, 126)
    volume_pctile = _rolling_percentile(volume.replace(0.0, np.nan), 756, 126)

    ma200 = close.rolling(200, min_periods=160).mean()
    trend_scale = daily_vol * np.sqrt(20.0)
    trend_z = np.log(positive_close / ma200.where(ma200 > 0.0)) / trend_scale.replace(0.0, np.nan)
    rsi14 = _rsi(close, 14)

    cftc_weekly, cftc_label = load_cftc_crowding(symbol)
    if not cftc_weekly.empty:
        crowding_pctile = _align_cftc_to_prices(cftc_weekly, close.index)
        crowding_source = cftc_label
    else:
        crowding_pctile = pd.Series(np.nan, index=close.index, dtype=float)
        crowding_source = "CFTC unavailable"

    ma10 = close.rolling(10, min_periods=8).mean()
    ret5 = close / close.shift(5) - 1.0
    prior5_low = close.shift(1).rolling(5, min_periods=5).min()
    reversal_score = (
        (ret5 < 0.0).astype(int)
        + (close < ma10).astype(int)
        + (close < prior5_low).astype(int)
    )

    breakout = None
    if profile == "Failed Breakout":
        breakout = failed_breakout_frame(close, int(settings["breakout_days"]), int(settings["memory"]))
        profile_setup = breakout["ProfileSetup"]
        condition = breakout["Signal"]
        early_setup = pd.Series(False, index=close.index)
        label = (
            f"Failed breakout: prior {settings['breakout_days']}D closing high lost within "
            f"{settings['memory']} sessions + close below 10D average"
        )
    else:
        ret_extreme = return_pctile >= float(settings["return_pctile"])
        trend_extreme = trend_z >= float(settings["trend_z"])
        rsi_extreme = rsi14 >= float(settings["rsi"])
        vol_extreme = vol_pctile >= float(settings["vol_pctile"])
        core_count = trend_extreme.astype(int) + rsi_extreme.astype(int) + vol_extreme.astype(int)
        early_setup = ret_extreme & (core_count >= 2)
        profile_setup = early_setup
        if profile == "Early Warning":
            condition = profile_setup
            label = (
                f"Early warning: {return_days}D return ≥ {settings['return_pctile']:.0f}th pct "
                "+ 2/3 trend, RSI, vol extremes"
            )
        elif profile == "Confirmed Exhaustion":
            recent_setup = profile_setup.rolling(int(settings["memory"]), min_periods=1).max().astype(bool)
            condition = recent_setup & (reversal_score >= int(settings["reversal_components"]))
            label = (
                f"Confirmed exhaustion: recent extreme + {int(settings['reversal_components'])}/3 "
                "price-reversal checks"
            )
        else:
            raise ValueError(f"Unknown signal profile: {profile}")

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
            "ProfileSetup": profile_setup.fillna(False).astype(bool),
            "Signal": condition.fillna(False).astype(bool),
        },
        index=close.index,
    )
    if breakout is not None:
        frame[["BreakoutPending", "BreakoutLevel"]] = breakout[["BreakoutPending", "BreakoutLevel"]]
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
    store: Dict[str, Dict[str, list]] = {
        label: {
            "date": [],
            "return": [],
            "signal_dd": [],
            "path_dd": [],
            "upside": [],
            "signal_dd_vol": [],
            "upside_vol": [],
        }
        for label in FORWARD_HORIZONS
    }

    for event_date in events:
        if event_date not in positions:
            continue
        start_pos = positions[event_date]
        start_price = float(close.iloc[start_pos])
        if not np.isfinite(start_price) or start_price == 0.0:
            continue
        diag = diagnostics.reindex([event_date]).iloc[0]
        realized_vol = float(diag.get("RealizedVol", np.nan))
        row = {
            "Date": event_date,
            "Price": start_price,
            "ReturnPctile": diag.get("ReturnPctile", np.nan),
            "TrendZ": diag.get("TrendZ", np.nan),
            "RSI": diag.get("RSI", np.nan),
            "VolPctile": diag.get("VolPctile", np.nan),
            "CrowdingPctile": diag.get("CrowdingPctile", np.nan),
            "ReversalScore": diag.get("ReversalScore", np.nan),
            "RealizedVol": realized_vol,
        }

        if "BreakoutLevel" in diagnostics.columns:
            row["BreakoutLevel"] = diag.get("BreakoutLevel", np.nan)

        if start_pos + 21 < len(close):
            local_start = max(0, start_pos - 21)
            local_end = min(len(close), start_pos + 22)
            local_path = close.iloc[local_start:local_end]
            peak_date = local_path.idxmax()
            peak_pos = positions[peak_date]
            peak_price = float(local_path.max())
            row["DaysFromLocalPeak"] = float(start_pos - peak_pos)
            row["PeakToSignal"] = start_price / peak_price - 1.0 if peak_price else np.nan
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
            path_dd = path / running_peak.replace(0.0, np.nan) - 1.0
            signal_dd = float(from_signal.min())
            upside = float(from_signal.max())
            horizon_sigma = (
                realized_vol * np.sqrt(horizon / 252.0)
                if np.isfinite(realized_vol) and realized_vol > 0.0
                else np.nan
            )
            row[label] = end_return
            store[label]["date"].append(pd.Timestamp(event_date))
            store[label]["return"].append(end_return)
            store[label]["signal_dd"].append(signal_dd)
            store[label]["path_dd"].append(float(path_dd.min()))
            store[label]["upside"].append(upside)
            store[label]["signal_dd_vol"].append(
                signal_dd / horizon_sigma if np.isfinite(horizon_sigma) else np.nan
            )
            store[label]["upside_vol"].append(
                upside / horizon_sigma if np.isfinite(horizon_sigma) else np.nan
            )
        rows.append(row)

    history = pd.DataFrame(rows)
    arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for label, metrics in store.items():
        arrays[label] = {}
        for metric, values in metrics.items():
            if metric == "date":
                arrays[label][metric] = np.asarray(values, dtype="datetime64[ns]")
            else:
                arrays[label][metric] = np.asarray(values, dtype=float)
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


def _independent_indices(
    close_index: pd.DatetimeIndex,
    event_dates: np.ndarray,
    horizon: int,
) -> np.ndarray:
    positions = {timestamp: i for i, timestamp in enumerate(close_index)}
    kept: List[int] = []
    last_position = -10**9
    for array_index, raw_date in enumerate(event_dates):
        date = pd.Timestamp(raw_date)
        position = positions.get(date)
        if position is None:
            continue
        if position - last_position >= max(int(horizon), 1):
            kept.append(array_index)
            last_position = position
    return np.asarray(kept, dtype=int)


def _seasonally_matched_baseline(
    close: pd.Series,
    start_date: pd.Timestamp,
    horizon: int,
    signal_dates: pd.DatetimeIndex,
) -> np.ndarray:
    close = close.dropna().astype(float)
    start_date = pd.Timestamp(start_date)
    positions = {timestamp: i for i, timestamp in enumerate(close.index)}
    signal_positions = [positions[date] for date in signal_dates if date in positions]
    month_counts = pd.Series(signal_dates.month).value_counts().to_dict() if len(signal_dates) else {}
    if not month_counts:
        month_counts = {month: 1 for month in range(1, 13)}

    weighted_returns: List[np.ndarray] = []
    for month, weight in sorted(month_counts.items()):
        candidates: List[float] = []
        last_position = -10**9
        for position, date in enumerate(close.index):
            if date < start_date or date.month != int(month):
                continue
            end_pos = position + horizon
            if end_pos >= len(close) or position - last_position < max(horizon, 1):
                continue
            start_price = float(close.iloc[position])
            end_price = float(close.iloc[end_pos])
            if not np.isfinite(start_price) or not np.isfinite(end_price) or start_price <= 0.0:
                continue
            if any(abs(position - signal_pos) < max(horizon, 1) for signal_pos in signal_positions):
                continue
            candidates.append(end_price / start_price - 1.0)
            last_position = position
        if candidates:
            values = np.asarray(candidates, dtype=float)
            weighted_returns.extend([values] * int(weight))
    if weighted_returns:
        return np.concatenate(weighted_returns)

    fallback: List[float] = []
    last_position = -10**9
    for position, date in enumerate(close.index):
        if date < start_date:
            continue
        end_pos = position + horizon
        if end_pos >= len(close) or position - last_position < max(horizon, 1):
            continue
        start_price = float(close.iloc[position])
        end_price = float(close.iloc[end_pos])
        if not np.isfinite(start_price) or not np.isfinite(end_price) or start_price <= 0.0:
            continue
        if any(abs(position - signal_pos) < max(horizon, 1) for signal_pos in signal_positions):
            continue
        fallback.append(end_price / start_price - 1.0)
        last_position = position
    return np.asarray(fallback, dtype=float)


def _bootstrap_median_edge_ci(
    signal_returns: np.ndarray,
    baseline_median: float,
    seed: int,
) -> Tuple[float, float]:
    signal_returns = np.asarray(signal_returns, dtype=float)
    signal_returns = signal_returns[np.isfinite(signal_returns)]
    if signal_returns.size < 3 or not np.isfinite(baseline_median):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    edges = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for i in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(signal_returns, size=signal_returns.size, replace=True)
        edges[i] = float(np.median(sampled) - baseline_median)
    low, high = np.quantile(edges, [0.025, 0.975])
    return float(low), float(high)


def summarize_forward_edge(
    close: pd.Series,
    history: pd.DataFrame,
    arrays: Dict[str, Dict[str, np.ndarray]],
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    metrics = [
        "Signal Median",
        "Baseline Median",
        "Median Edge",
        "% Negative",
        "Baseline % Negative",
        "Hit-Rate Lift",
        "Median Further Upside",
        "Median Subsequent Decline",
        "95% CI Low",
        "95% CI High",
        "Independent N",
    ]
    summary = pd.DataFrame(index=metrics, columns=list(FORWARD_HORIZONS), dtype=float)
    close = close.dropna().astype(float)

    for label, horizon in FORWARD_HORIZONS.items():
        store = arrays[label]
        event_dates = store.get("date", np.asarray([], dtype="datetime64[ns]"))
        independent = _independent_indices(close.index, event_dates, horizon)
        signal_returns = store["return"][independent] if independent.size else np.asarray([], dtype=float)
        signal_dates = (
            pd.DatetimeIndex(pd.to_datetime(event_dates[independent]))
            if independent.size
            else pd.DatetimeIndex([])
        )
        baseline = _seasonally_matched_baseline(close, start_date, horizon, signal_dates)

        if baseline.size:
            baseline_median = float(np.median(baseline))
            summary.loc["Baseline Median", label] = baseline_median
            summary.loc["Baseline % Negative", label] = float(np.mean(baseline < 0.0))
        else:
            baseline_median = np.nan

        summary.loc["Independent N", label] = float(signal_returns.size)
        if signal_returns.size == 0:
            continue

        signal_median = float(np.median(signal_returns))
        signal_negative = float(np.mean(signal_returns < 0.0))
        summary.loc["Signal Median", label] = signal_median
        summary.loc["% Negative", label] = signal_negative
        if np.isfinite(baseline_median):
            summary.loc["Median Edge", label] = signal_median - baseline_median
            summary.loc["Hit-Rate Lift", label] = (
                signal_negative - float(summary.loc["Baseline % Negative", label])
            )
            ci_low, ci_high = _bootstrap_median_edge_ci(
                signal_returns, baseline_median, seed=7300 + int(horizon)
            )
            summary.loc["95% CI Low", label] = ci_low
            summary.loc["95% CI High", label] = ci_high

        upside = store["upside"][independent]
        decline = store["signal_dd"][independent]
        summary.loc["Median Further Upside", label] = float(np.nanmedian(upside))
        summary.loc["Median Subsequent Decline", label] = float(np.nanmedian(decline))

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
    days = pd.to_numeric(
        history.get("DaysFromLocalPeak", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    peak_move = pd.to_numeric(
        history.get("PeakToSignal", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    return {
        "days_from_peak": float(days.median()) if not days.empty else np.nan,
        "peak_to_signal": float(peak_move.median()) if not peak_move.empty else np.nan,
        "down1sigma_3m": _top_hit_rate(arrays["3M"]["signal_dd_vol"], 1.0, "down"),
        "down15sigma_6m": _top_hit_rate(arrays["6M"]["signal_dd_vol"], 1.5, "down"),
        "up1sigma_3m": _top_hit_rate(arrays["3M"]["upside_vol"], 1.0, "up"),
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
        hover_extra = ""
        custom = diagnostics.reindex(event_dates)[
            ["ReturnPctile", "TrendZ", "RSI", "VolPctile", "CrowdingPctile", "ReversalScore"]
        ].to_numpy()
        if "BreakoutLevel" in diagnostics:
            custom = np.column_stack([custom, diagnostics.reindex(event_dates)["BreakoutLevel"],
                                      diagnostics.reindex(event_dates)["MA10"]])
            hover_extra = "<br>Original breakout level: %{customdata[6]:,.2f}<br>10D average: %{customdata[7]:,.2f}"
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
                    "<br>Trend extension: %{customdata[1]:.2f} vol units"
                    "<br>RSI: %{customdata[2]:.2f}"
                    "<br>Vol pctile: %{customdata[3]:.2f}"
                    "<br>Crowding pctile: %{customdata[4]:.2f}"
                    "<br>Reversal score: %{customdata[5]:.0f}/3"
                    + hover_extra + "<extra></extra>"
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
    if pd.isna(value):
        return "neutral"
    if metric == "Median Edge":
        return "good" if value < 0.0 else "bad" if value > 0.0 else "neutral"
    if metric == "Hit-Rate Lift":
        return "good" if value > 0.0 else "bad" if value < 0.0 else "neutral"
    return "neutral"


def _format_summary_value(metric: str, value: float) -> str:
    if pd.isna(value):
        return "—"
    if metric == "Independent N":
        return f"{int(round(value))}"
    if metric in {"% Negative", "Baseline % Negative"}:
        return f"{value * 100.0:.1f}%"
    if metric == "Hit-Rate Lift":
        return f"{value * 100.0:+.1f} pp"
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


def _format_number(value: float, fmt: str) -> str:
    return "—" if pd.isna(value) or not np.isfinite(float(value)) else format(float(value), fmt)


def _history_display(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    display = history.copy().sort_values("Date", ascending=False)
    display["Date"] = pd.to_datetime(display["Date"]).dt.strftime("%Y-%m-%d")
    display["Price"] = display["Price"].map(lambda x: _format_number(x, ",.2f"))
    if "BreakoutLevel" in display:
        display["BreakoutLevel"] = display["BreakoutLevel"].map(lambda x: _format_number(x, ",.2f"))
    for column in ["ReturnPctile", "RSI", "VolPctile", "CrowdingPctile"]:
        display[column] = display[column].map(lambda x: _format_number(x, ".2f"))
    display["TrendZ"] = display["TrendZ"].map(lambda x: _format_number(x, ".2f"))
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
    columns = [
        "Date",
        "Price",
        "BreakoutLevel",
        "ReturnPctile",
        "TrendZ",
        "RSI",
        "VolPctile",
        "CrowdingPctile",
        "ReversalScore",
        "DaysFromLocalPeak",
        "PeakToSignal",
        *FORWARD_HORIZONS,
    ]
    return display[[column for column in columns if column in display.columns]]


def _settings_controls(profile: str) -> Tuple[dict, str]:
    preset = dict(PROFILE_PRESETS[profile])
    with st.sidebar:
        if profile == "Failed Breakout":
            return preset, "3M"
        with st.expander("Advanced thresholds", expanded=False):
            customize = st.checkbox("Customize preset", value=False)
            if customize:
                preset["return_pctile"] = st.slider(
                    "Return percentile", 80.0, 99.5, float(preset["return_pctile"]), 0.5
                )
                preset["trend_z"] = st.slider(
                    "Trend extension (20D vol units)",
                    0.5,
                    4.0,
                    float(preset["trend_z"]),
                    0.25,
                )
                preset["rsi"] = st.slider("RSI threshold", 55.0, 90.0, float(preset["rsi"]), 1.0)
                preset["vol_pctile"] = st.slider(
                    "Realized-vol percentile", 50.0, 99.0, float(preset["vol_pctile"]), 1.0
                )
                preset["memory"] = st.slider(
                    "Setup memory (sessions)", 3, 20, int(preset["memory"]), 1
                )
                if profile != "Early Warning":
                    preset["reversal_components"] = st.slider(
                        "Price-reversal checks required",
                        1,
                        3,
                        int(preset["reversal_components"]),
                        1,
                    )
            return_window = st.selectbox("Extension return window", list(RETURN_WINDOWS), index=2)
    return preset, return_window


def _current_state(
    profile: str,
    current_signal: bool,
    recent_profile_setup: bool,
    crowding_source: str,
) -> str:
    if current_signal:
        return "ACTIVE"
    if profile == "Failed Breakout":
        return "Breakout / waiting for failure" if recent_profile_setup else "Normal"
    if profile == "Early Warning":
        return "Recent setup" if recent_profile_setup else "Normal"
    return "Setup / waiting for reversal" if recent_profile_setup else "Normal"


def render_commodity_event_study() -> None:
    _inject_page_style()
    render_page_header(
        PageHeader(
            title=TITLE,
            description=(
                "Find historically extended commodity moves, wait for exhaustion or reversal, "
                "and measure whether the signal changed the forward distribution versus history."
            ),
            eyebrow="ADFM Historical Context",
            source_note="Yahoo Finance continuous futures · CFTC Disaggregated COT where mapped",
        )
    )

    controls = st.columns([2.05, 1.35, 0.85, 0.85], gap="small")
    with controls[0]:
        contract_choice = st.selectbox("Commodity future", CONTRACT_OPTIONS, index=0)
    with controls[1]:
        profile = st.selectbox(
            "Top signal",
            list(PROFILE_PRESETS),
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
    summary = summarize_forward_edge(close_full, history, arrays, chart_close.index.min())
    top_stats = top_diagnostics(history, arrays)

    latest_date = close_full.index.max()
    latest = diagnostics.reindex([latest_date]).iloc[0]
    latest_event = study_events.max() if len(study_events) else None
    latest_event_text = latest_event.strftime("%b %d, %Y") if latest_event is not None else "None"
    current_signal = bool(condition.reindex([latest_date]).fillna(False).iloc[0])
    recent_profile_setup = (
        bool(latest["BreakoutPending"])
        if profile == "Failed Breakout"
        else bool(diagnostics["ProfileSetup"].tail(int(settings["memory"])).fillna(False).any())
    )
    current_state = _current_state(
        profile, current_signal, recent_profile_setup, crowding_source
    )
    crowding_value = latest.get("CrowdingPctile", np.nan)
    crowding_text = "n/a" if pd.isna(crowding_value) else f"{float(crowding_value):.2f}th pct"
    return_pctile_value = latest.get("ReturnPctile", np.nan)
    trend_value = latest.get("TrendZ", np.nan)
    rsi_value = latest.get("RSI", np.nan)

    st.markdown(
        (
            "<div class='event-study-status'>"
            f"<span><strong>{escape(contract_name)}</strong> · {escape(symbol)}</span>"
            f"<span><strong>Signal</strong> {escape(profile)}</span>"
            f"<span><strong>Current</strong> {escape(current_state)}</span>"
            f"<span><strong>{escape(return_window)} return pctile</strong> "
            f"{_format_number(return_pctile_value, '.2f')}</span>"
            f"<span><strong>Trend ext.</strong> {_format_number(trend_value, '.2f')}</span>"
            f"<span><strong>RSI</strong> {_format_number(rsi_value, '.2f')}</span>"
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

    st.markdown(
        "<div class='event-study-table-title'>Top-Picking Diagnostics</div>",
        unsafe_allow_html=True,
    )
    diag_cols = st.columns(5, gap="small")
    diag_cols[0].metric("Median signal timing", _format_days_from_peak(top_stats["days_from_peak"]))
    diag_cols[1].metric(
        "Median peak → signal",
        "—"
        if not np.isfinite(top_stats["peak_to_signal"])
        else f"{top_stats['peak_to_signal'] * 100.0:+.2f}%",
    )
    diag_cols[2].metric(
        "3M ≥1σ decline",
        "—"
        if not np.isfinite(top_stats["down1sigma_3m"])
        else f"{top_stats['down1sigma_3m'] * 100.0:.1f}%",
    )
    diag_cols[3].metric(
        "6M ≥1.5σ decline",
        "—"
        if not np.isfinite(top_stats["down15sigma_6m"])
        else f"{top_stats['down15sigma_6m'] * 100.0:.1f}%",
    )
    diag_cols[4].metric(
        "3M >1σ further upside",
        "—"
        if not np.isfinite(top_stats["up1sigma_3m"])
        else f"{top_stats['up1sigma_3m'] * 100.0:.1f}%",
    )
    st.markdown(
        (
            "<div class='event-study-caption'>Local peak timing uses the highest close in the 21 "
            "sessions before through 21 sessions after each signal. Decline and further-upside "
            "thresholds are scaled by the annualized volatility known on the signal date, so the "
            "diagnostic is comparable across commodities.</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='event-study-table-title'>{escape(contract_name)} Edge After Top Signal</div>",
        unsafe_allow_html=True,
    )
    st.markdown(summary_table_html(summary), unsafe_allow_html=True)
    st.markdown(
        (
            "<div class='event-study-caption'>"
            "Edge compares de-overlapped signal outcomes with a seasonally matched historical "
            "baseline from the same lookback. Green is reserved for favorable top-signal edge: "
            "a more-negative median than baseline or a higher negative-return hit rate. The 95% "
            "interval bootstraps the independent signal median around the historical baseline."
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
            - Return percentiles use the selected return window ranked against up to five years of trailing daily observations. Only information available by that session is used.
            - Trend extension is the log distance from the 200-day moving average divided by the current 20-session expected move. This is more stable across commodity price regimes than scaling by the standard deviation of price levels.
            - Realized volatility uses 20-day annualized log-return volatility ranked against a trailing three-year distribution.
            - Confirmed Exhaustion uses three price-reversal checks after a recent extreme: negative 5-day return, close below the 10-day moving average, and close below the prior 5-session low. These are deliberately described as related price checks, not independent evidence.
            - Failed Breakout starts with a close above the highest close of the preceding 63 sessions. Each breakout keeps its original level for the following 10 sessions; the first close below both that level and the 10-day moving average signals failure. Later highs do not reset earlier levels or extend their expiry. Multiple failures on one date count once, with the existing event-spacing rule applied afterward.
            - CFTC Managed Money positioning is context only and never gates a price signal. Volume is never substituted for positioning; missing CFTC data remains unavailable.
            - CFTC observations become usable only on the first business session after public release at 3:30 p.m. ET. Known 2020, 2021, 2023 and 2025 delays are explicitly dated; the 2018-19 shutdown backlog is excluded rather than assigned invented publication dates. Positioning also expires after seven sessions if no fresh report is available.
            - Forward statistics de-overlap signal events separately for each horizon. The baseline is conditioned on the signal sample's calendar months, excludes starts close to signal windows, and is de-overlapped before comparison. If a short history leaves no same-month control, the page falls back to an unconditional de-overlapped control. Bootstrap intervals require at least three independent signals.
            - Volatility-normalized excursion diagnostics scale each event by the annualized volatility known on the signal date and the square root of the forward horizon.
            - Yahoo Finance futures histories are provider-supplied continuous series. Roll methodology can create discontinuities; the study does not infer or back-adjust individual contract rolls and should not be treated as execution-grade roll attribution.
            """
        )

    render_footer(
        data_note=(
            "Primary input: Yahoo Finance daily continuous-futures history. CFTC Disaggregated "
            "Futures Only positioning is used where a verified contract mapping exists. Historical "
            "tendency only; event studies are descriptive and not forecasts."
        )
    )
