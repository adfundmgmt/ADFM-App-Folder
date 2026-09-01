from __future__ import annotations

import tempfile
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from adfm_core.ui import PageHeader, inject_explorer_style, render_footer, render_page_header

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

RETURN_WINDOWS = {
    "1M": 21,
    "2M": 42,
    "3M": 63,
    "6M": 126,
    "12M": 252,
}

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

SPACING_OPTIONS = {
    "1M": 21,
    "2M": 42,
    "3M": 63,
    "6M": 126,
    "12M": 252,
}

LOOKBACK_OPTIONS = {
    "Max": None,
    "10Y": 10,
    "25Y": 25,
    "50Y": 50,
}


st.set_page_config(page_title=TITLE, layout="wide")
inject_explorer_style(max_width_px=1540)

st.markdown(
    """
    <style>
    .event-study-status {
        display: flex;
        flex-wrap: wrap;
        gap: .55rem 1.2rem;
        align-items: center;
        border-top: 1px solid #d7d7d7;
        border-bottom: 1px solid #d7d7d7;
        margin: .25rem 0 .55rem;
        padding: .58rem 0;
        color: #222222;
        font-family: Arial, Helvetica, sans-serif;
        font-size: .79rem;
        line-height: 1.35;
    }
    .event-study-status strong {
        color: #000000;
        font-weight: 800;
    }
    .event-study-table-wrap {
        width: 100%;
        overflow-x: auto;
        margin: .2rem 0 .65rem;
        border: 1px solid #aeb7bd;
    }
    table.event-study-table {
        width: 100%;
        min-width: 930px;
        border-collapse: collapse;
        table-layout: fixed;
        font-family: Arial, Helvetica, sans-serif;
        font-size: .76rem;
        line-height: 1.16;
    }
    table.event-study-table th,
    table.event-study-table td {
        border-right: 1px solid #aeb7bd;
        border-bottom: 1px solid #aeb7bd;
        padding: .36rem .42rem;
        text-align: center;
        white-space: nowrap;
    }
    table.event-study-table th:last-child,
    table.event-study-table td:last-child {
        border-right: 0;
    }
    table.event-study-table tr:last-child td {
        border-bottom: 0;
    }
    table.event-study-table thead th {
        background: #137f8e;
        color: #ffffff;
        font-weight: 800;
        letter-spacing: .01em;
    }
    table.event-study-table tbody td.metric {
        background: #edf0f2;
        color: #111111;
        font-weight: 800;
        text-align: left;
        padding-left: .72rem;
    }
    table.event-study-table tbody td.pos {
        background: #dcebe1;
        color: #111111;
    }
    table.event-study-table tbody td.neg {
        background: #efc8cc;
        color: #111111;
    }
    table.event-study-table tbody td.neutral {
        background: #e8eef2;
        color: #111111;
    }
    .event-study-caption {
        color: #666666;
        font-family: Arial, Helvetica, sans-serif;
        font-size: .74rem;
        line-height: 1.4;
        margin: .2rem 0 .8rem;
    }
    @media (max-width: 800px) {
        .event-study-status {
            gap: .4rem .8rem;
            font-size: .74rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

render_page_header(
    PageHeader(
        title=TITLE,
        description=(
            "Define a repeatable commodity price event, mark every historical occurrence, "
            "and measure the forward return and drawdown distribution."
        ),
        eyebrow="ADFM Historical Context",
        source_note="Yahoo Finance continuous-futures price history",
    )
)


# -------------------------
# Data
# -------------------------


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


# -------------------------
# Signal construction
# -------------------------


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    rsi = rsi.mask((avg_gain == 0.0) & (avg_loss > 0.0), 0.0)
    return rsi.clip(lower=0.0, upper=100.0)


def build_signal(
    close: pd.Series,
    signal_type: str,
    *,
    direction: str = "Rally",
    window_days: int = 63,
    threshold: float = 25.0,
    rsi_period: int = 14,
) -> Tuple[pd.Series, pd.Series, str, str]:
    if signal_type == "Return threshold":
        metric = close.pct_change(window_days) * 100.0
        if direction == "Rally":
            condition = metric >= threshold
            label = f"{_window_label(window_days)} return ≥ +{threshold:.2f}%"
        else:
            condition = metric <= -threshold
            label = f"{_window_label(window_days)} return ≤ -{threshold:.2f}%"
        return metric, condition.fillna(False), label, "percent"

    if signal_type == "52-week breakout":
        if direction == "High":
            prior_extreme = close.shift(1).rolling(252, min_periods=200).max()
            metric = close / prior_extreme * 100.0 - 100.0
            condition = close >= prior_extreme
            label = "New 52-week high"
        else:
            prior_extreme = close.shift(1).rolling(252, min_periods=200).min()
            metric = close / prior_extreme * 100.0 - 100.0
            condition = close <= prior_extreme
            label = "New 52-week low"
        return metric, condition.fillna(False), label, "percent"

    if signal_type == "RSI extreme":
        metric = _rsi(close, rsi_period)
        if direction == "Overbought":
            condition = metric >= threshold
            label = f"RSI({rsi_period}) ≥ {threshold:.2f}"
        else:
            condition = metric <= threshold
            label = f"RSI({rsi_period}) ≤ {threshold:.2f}"
        return metric, condition.fillna(False), label, "number"

    if signal_type == "200D trend stretch":
        moving_average = close.rolling(200, min_periods=180).mean()
        metric = (close / moving_average - 1.0) * 100.0
        if direction == "Above":
            condition = metric >= threshold
            label = f"Price ≥ {threshold:.2f}% above 200D MA"
        else:
            condition = metric <= -threshold
            label = f"Price {threshold:.2f}% below 200D MA"
        return metric, condition.fillna(False), label, "percent"

    raise ValueError(f"Unknown signal type: {signal_type}")


def _window_label(window_days: int) -> str:
    for label, days in RETURN_WINDOWS.items():
        if days == window_days:
            return label
    return f"{window_days}D"


def detect_events(condition: pd.Series, spacing_days: int, *, continuous: bool = False) -> pd.DatetimeIndex:
    condition = condition.fillna(False).astype(bool)
    if continuous:
        candidates = condition
    else:
        candidates = condition & ~condition.shift(1, fill_value=False)

    candidate_positions = np.flatnonzero(candidates.to_numpy())
    kept_positions: List[int] = []
    last_position = -10**9
    for position in candidate_positions:
        if position - last_position >= spacing_days:
            kept_positions.append(int(position))
            last_position = int(position)

    return pd.DatetimeIndex(condition.index[kept_positions])


# -------------------------
# Event-study math
# -------------------------


def build_event_observations(
    close: pd.Series,
    events: Iterable[pd.Timestamp],
    signal_metric: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    close = close.dropna().astype(float)
    index_positions = {timestamp: i for i, timestamp in enumerate(close.index)}
    rows: List[dict] = []
    horizon_store: Dict[str, Dict[str, List[float]]] = {
        label: {"return": [], "signal_dd": [], "path_dd": [], "upside": []}
        for label in FORWARD_HORIZONS
    }

    for event_date in events:
        if event_date not in index_positions:
            continue
        start_pos = index_positions[event_date]
        start_price = float(close.iloc[start_pos])
        row = {
            "Date": event_date,
            "Price": start_price,
            "Signal": float(signal_metric.reindex([event_date]).iloc[0]),
        }

        for label, horizon in FORWARD_HORIZONS.items():
            end_pos = start_pos + horizon
            if end_pos >= len(close):
                row[label] = np.nan
                continue

            path = close.iloc[start_pos : end_pos + 1].astype(float)
            end_return = float(path.iloc[-1] / start_price - 1.0)
            from_signal = path / start_price - 1.0
            running_peak = path.cummax()
            path_drawdown = path / running_peak - 1.0

            row[label] = end_return
            horizon_store[label]["return"].append(end_return)
            horizon_store[label]["signal_dd"].append(float(from_signal.min()))
            horizon_store[label]["path_dd"].append(float(path_drawdown.min()))
            horizon_store[label]["upside"].append(float(from_signal.max()))

        rows.append(row)

    history = pd.DataFrame(rows)
    arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for label, metrics in horizon_store.items():
        arrays[label] = {
            metric_name: np.asarray(values, dtype=float)
            for metric_name, values in metrics.items()
        }
    return history, arrays


def summarize_forward_performance(
    arrays: Dict[str, Dict[str, np.ndarray]],
) -> pd.DataFrame:
    metric_rows = [
        "Average",
        "Median",
        "Best",
        "Worst",
        "% Positive",
        "Avg DD From Signal",
        "Worst DD From Signal",
        "Avg Peak-to-Trough DD",
        "Sample",
    ]
    summary = pd.DataFrame(index=metric_rows, columns=list(FORWARD_HORIZONS), dtype=float)

    for horizon in FORWARD_HORIZONS:
        returns = arrays[horizon]["return"]
        signal_dd = arrays[horizon]["signal_dd"]
        path_dd = arrays[horizon]["path_dd"]

        if returns.size == 0:
            continue

        summary.loc["Average", horizon] = np.mean(returns)
        summary.loc["Median", horizon] = np.median(returns)
        summary.loc["Best", horizon] = np.max(returns)
        summary.loc["Worst", horizon] = np.min(returns)
        summary.loc["% Positive", horizon] = np.mean(returns > 0.0)
        summary.loc["Avg DD From Signal", horizon] = np.mean(signal_dd)
        summary.loc["Worst DD From Signal", horizon] = np.min(signal_dd)
        summary.loc["Avg Peak-to-Trough DD", horizon] = np.mean(path_dd)
        summary.loc["Sample", horizon] = float(returns.size)

    return summary


# -------------------------
# Presentation
# -------------------------


def _format_signal_value(value: float, value_kind: str) -> str:
    if pd.isna(value):
        return "n/a"
    if value_kind == "percent":
        return f"{value:+.2f}%"
    return f"{value:.2f}"


def make_price_chart(
    close: pd.Series,
    event_dates: pd.DatetimeIndex,
    signal_metric: pd.Series,
    signal_label: str,
    signal_value_kind: str,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=close.index,
            y=close.values,
            mode="lines",
            line={"color": "#2f7fd1", "width": 1.65},
            hovertemplate="%{x|%b %d, %Y}<br>Price: %{y:,.2f}<extra></extra>",
            name="Price",
        )
    )

    if len(event_dates):
        event_prices = close.reindex(event_dates)
        hover_text = []
        for event_date in event_dates:
            signal_value = signal_metric.reindex([event_date]).iloc[0]
            hover_text.append(
                f"{event_date:%b %d, %Y}<br>"
                f"Price: {event_prices.loc[event_date]:,.2f}<br>"
                f"Signal: {_format_signal_value(signal_value, signal_value_kind)}"
            )
        figure.add_trace(
            go.Scatter(
                x=event_dates,
                y=event_prices.values,
                mode="markers",
                marker={
                    "color": "#e52822",
                    "size": 7,
                    "line": {"color": "#ffffff", "width": 0.65},
                },
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
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
    if metric == "% Positive":
        return "pos" if value >= 0.5 else "neg"
    if "Drawdown" in metric or "DD" in metric:
        return "neg"
    return "pos" if value >= 0.0 else "neg"


def _format_summary_value(metric: str, value: float) -> str:
    if pd.isna(value):
        return "—"
    if metric == "Sample":
        return f"{int(round(value))}"
    if metric == "% Positive":
        return f"{value * 100.0:.2f}%"
    return f"{value * 100.0:+.2f}%"


def summary_table_html(summary: pd.DataFrame) -> str:
    columns = list(summary.columns)
    head = "".join(f"<th>{column}</th>" for column in columns)
    body_rows = []
    for metric, row in summary.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            cells.append(
                f"<td class='{_cell_class(metric, value)}'>"
                f"{_format_summary_value(metric, value)}</td>"
            )
        body_rows.append(
            "<tr>"
            f"<td class='metric'>{metric}</td>"
            + "".join(cells)
            + "</tr>"
        )
    return (
        "<div class='event-study-table-wrap'>"
        "<table class='event-study-table'>"
        f"<thead><tr><th>Metric</th>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def _history_display(history: pd.DataFrame, signal_kind: str) -> pd.DataFrame:
    if history.empty:
        return history
    display = history.copy().sort_values("Date", ascending=False)
    display["Date"] = pd.to_datetime(display["Date"]).dt.strftime("%Y-%m-%d")
    display["Price"] = display["Price"].map(lambda value: f"{value:,.2f}")
    display["Signal"] = display["Signal"].map(
        lambda value: _format_signal_value(value, signal_kind)
    )
    for column in FORWARD_HORIZONS:
        display[column] = display[column].map(
            lambda value: "—" if pd.isna(value) else f"{value * 100.0:+.2f}%"
        )
    return display


# -------------------------
# Controls
# -------------------------

control_columns = st.columns([2.05, 1.35, 0.85, 0.85], gap="small")
with control_columns[0]:
    contract_choice = st.selectbox(
        "Commodity future",
        CONTRACT_OPTIONS,
        index=0,
    )
with control_columns[1]:
    signal_type = st.selectbox(
        "Event signal",
        ["Return threshold", "52-week breakout", "RSI extreme", "200D trend stretch"],
        index=0,
    )
with control_columns[2]:
    spacing_label = st.selectbox("Event spacing", list(SPACING_OPTIONS), index=2)
with control_columns[3]:
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

param_columns = st.columns([1.0, 1.0, 1.0, 2.0], gap="small")
continuous_signal = False

if signal_type == "Return threshold":
    with param_columns[0]:
        direction = st.selectbox("Direction", ["Rally", "Selloff"], index=0)
    with param_columns[1]:
        return_window = st.selectbox("Signal window", list(RETURN_WINDOWS), index=2)
    with param_columns[2]:
        threshold = st.number_input(
            "Threshold (%)",
            min_value=1.0,
            max_value=300.0,
            value=25.0,
            step=1.0,
            format="%.2f",
        )
    window_days = RETURN_WINDOWS[return_window]
    rsi_period = 14
elif signal_type == "52-week breakout":
    with param_columns[0]:
        direction = st.selectbox("Breakout", ["High", "Low"], index=0)
    with param_columns[1]:
        st.text_input("Window", value="52 weeks", disabled=True)
    threshold = 0.0
    window_days = 252
    rsi_period = 14
    continuous_signal = True
elif signal_type == "RSI extreme":
    with param_columns[0]:
        direction = st.selectbox("Direction", ["Overbought", "Oversold"], index=0)
    with param_columns[1]:
        rsi_period = st.number_input(
            "RSI period",
            min_value=5,
            max_value=50,
            value=14,
            step=1,
        )
    with param_columns[2]:
        default_rsi = 70.0 if direction == "Overbought" else 30.0
        threshold = st.number_input(
            "RSI threshold",
            min_value=1.0,
            max_value=99.0,
            value=default_rsi,
            step=1.0,
            format="%.2f",
        )
    window_days = int(rsi_period)
else:
    with param_columns[0]:
        direction = st.selectbox("Direction", ["Above", "Below"], index=0)
    with param_columns[1]:
        st.text_input("Reference", value="200D moving average", disabled=True)
    with param_columns[2]:
        threshold = st.number_input(
            "Stretch (%)",
            min_value=1.0,
            max_value=200.0,
            value=20.0,
            step=1.0,
            format="%.2f",
        )
    window_days = 200
    rsi_period = 14

if not symbol:
    st.warning("Enter a Yahoo Finance futures symbol.")
    st.stop()


# -------------------------
# Run study
# -------------------------

try:
    with st.spinner(f"Loading {symbol} history…"):
        data = load_contract_history(symbol)
except Exception as exc:
    st.error(str(exc))
    render_footer(
        data_note=(
            "Yahoo Finance continuous-futures histories are provider-supplied stitched series. "
            "Availability and roll construction vary by contract."
        )
    )
    st.stop()

close_full = data["Close"].dropna().astype(float)
signal_metric, condition, signal_label, signal_kind = build_signal(
    close_full,
    signal_type,
    direction=direction,
    window_days=int(window_days),
    threshold=float(threshold),
    rsi_period=int(rsi_period),
)

all_events = detect_events(
    condition,
    SPACING_OPTIONS[spacing_label],
    continuous=continuous_signal,
)

lookback_years = LOOKBACK_OPTIONS[lookback_label]
if lookback_years is None:
    chart_close = close_full.copy()
else:
    chart_start = close_full.index.max() - pd.DateOffset(years=int(lookback_years))
    chart_close = close_full.loc[close_full.index >= chart_start].copy()

if chart_close.empty:
    st.error("No price history is available inside the selected lookback.")
    st.stop()

study_events = all_events[all_events >= chart_close.index.min()]
history, arrays = build_event_observations(close_full, study_events, signal_metric)
summary = summarize_forward_performance(arrays)

latest_date = close_full.index.max()
latest_condition = bool(condition.reindex([latest_date]).fillna(False).iloc[0])
latest_event = study_events.max() if len(study_events) else None
latest_signal_value = float(signal_metric.reindex([latest_date]).iloc[0])
active_text = "ACTIVE" if latest_condition else "Inactive"
latest_event_text = latest_event.strftime("%b %d, %Y") if latest_event is not None else "None"

st.markdown(
    (
        "<div class='event-study-status'>"
        f"<span><strong>{contract_name}</strong> · {symbol}</span>"
        f"<span><strong>Signal</strong> {signal_label}</span>"
        f"<span><strong>Current</strong> {active_text} "
        f"({_format_signal_value(latest_signal_value, signal_kind)})</span>"
        f"<span><strong>Events</strong> {len(study_events)}</span>"
        f"<span><strong>Latest event</strong> {latest_event_text}</span>"
        f"<span><strong>Data through</strong> {latest_date:%b %d, %Y}</span>"
        "</div>"
    ),
    unsafe_allow_html=True,
)

figure = make_price_chart(
    chart_close,
    study_events,
    signal_metric,
    signal_label,
    signal_kind,
)
st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

st.markdown(f"### {contract_name} Forward Performance After Event")
st.markdown(summary_table_html(summary), unsafe_allow_html=True)
st.markdown(
    (
        "<div class='event-study-caption'>"
        "Forward returns are close-to-close from the signal session. "
        "Avg/Worst DD From Signal use the lowest price reached versus the signal close. "
        "Avg Peak-to-Trough DD measures the worst decline from any interim peak inside each window. "
        "Sample falls at longer horizons when recent events have not yet matured."
        "</div>"
    ),
    unsafe_allow_html=True,
)

with st.expander("Historical event dates"):
    if history.empty:
        st.info("No events met the selected definition in this lookback.")
    else:
        history_display = _history_display(history, signal_kind)
        st.dataframe(history_display, width="stretch", hide_index=True, height=420)
        csv_bytes = history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download event history CSV",
            data=csv_bytes,
            file_name=f"{symbol.replace('=', '_')}_commodity_event_study.csv",
            mime="text/csv",
        )

with st.expander("Method and data caveat"):
    st.markdown(
        """
        - A signal is recorded on the first session a threshold becomes true. New 52-week highs/lows are sampled directly, then de-duplicated by the selected event spacing.
        - Forward horizons use trading sessions: 1W = 5, 1M = 21, 3M = 63, 6M = 126, and 12M = 252 sessions.
        - Yahoo Finance futures histories are continuous provider series. Contract-roll methodology can create discontinuities, so this is best used for historical tendency and signal research rather than execution-grade roll attribution.
        """
    )

render_footer(
    data_note=(
        "Primary input: Yahoo Finance daily continuous-futures price history. "
        "Historical tendency only; event studies are descriptive and not forecasts."
    )
)
