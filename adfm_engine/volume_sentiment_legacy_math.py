"""Extracted causal volume math from the Streamlit tool. Regenerate with scripts/extract_volume_sentiment_engine.py."""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
PASTEL_GREEN = '#88bca0'
PASTEL_RED = '#e8aaaa'
PASTEL_GREY = '#a7b6c7'
AMBER = '#e3c391'
BLUE = '#a4bee6'


def rolling_percentile_previous(
    series: pd.Series,
    window: int,
    min_periods: int,
    scale: float = 1.0,
) -> pd.Series:
    """Rank each observation against prior observations only.

    Excluding the current observation avoids a small but persistent look-ahead bias
    in live percentile signals. Ties receive half credit.
    """
    clean = pd.to_numeric(series, errors="coerce")
    window = max(int(window), 2)
    min_periods = max(1, min(int(min_periods), window))

    def _rank(values: np.ndarray) -> float:
        current = values[-1]
        history = pd.Series(values[:-1]).dropna().to_numpy(dtype=float)
        if not np.isfinite(current) or len(history) < min_periods:
            return np.nan
        below = float(np.sum(history < current))
        tied = float(np.sum(history == current))
        return ((below + 0.5 * tied) / len(history)) * float(scale)

    return clean.rolling(window + 1, min_periods=min_periods + 1).apply(_rank, raw=True)

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def normalize_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")

    out = out[~out.index.isna()].copy()

    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(None)

    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")].copy()

    return out

def validate_ohlcv(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"{source_name} returned an empty DataFrame.")

    out = normalize_dt_index(df)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in out.columns]

    if missing:
        raise ValueError(f"{source_name} missing columns: {missing}")

    keep = needed + (["Raw_Close"] if "Raw_Close" in out.columns else [])
    out = out[keep].copy()

    for col in keep:
        out[col] = safe_numeric(out[col])

    out = out.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
    out = out[(out["Close"] > 0) & (out["High"] > 0) & (out["Low"] > 0)].copy()
    out = out[out["Volume"] >= 0].copy()

    if out.empty:
        raise ValueError(f"{source_name} had no valid OHLCV rows after cleaning.")

    return out

def classify_state(pctl: float, high_cutoff: float, low_cutoff: float) -> str:
    if not np.isfinite(pctl):
        return "Unavailable"
    if pctl >= high_cutoff:
        return "Heavy"
    if pctl <= low_cutoff:
        return "Quiet"
    return "Normal"

def classify_setup(row: pd.Series, high_cutoff: float, low_cutoff: float) -> str:
    pctl = row.get("Volume_Pctl", np.nan)
    ret = row.get("Ret_1D", np.nan)
    close_loc = row.get("Close_Location", np.nan)
    close = row.get("Close", np.nan)
    ma20 = row.get("Price_MA20", np.nan)
    ma50 = row.get("Price_MA50", np.nan)
    range_atr = row.get("Range_ATR20", np.nan)

    if not np.isfinite(pctl):
        return "Unavailable"

    is_above_20 = np.isfinite(close) and np.isfinite(ma20) and close >= ma20
    is_below_20 = np.isfinite(close) and np.isfinite(ma20) and close < ma20
    is_uptrend = np.isfinite(ma20) and np.isfinite(ma50) and ma20 >= ma50
    is_downtrend = np.isfinite(ma20) and np.isfinite(ma50) and ma20 < ma50

    if pctl >= high_cutoff:
        if np.isfinite(ret) and np.isfinite(close_loc):
            if ret > 0 and close_loc >= 0.65 and is_above_20:
                return "Heavy Accumulation"
            if ret < 0 and close_loc <= 0.35:
                return "Heavy Distribution"
            if abs(ret) <= 0.35 and np.isfinite(range_atr) and range_atr >= 0.90:
                return "High Effort, Low Progress"
            if ret > 0 and close_loc <= 0.45:
                return "Upside Rejection"
            if ret < 0 and close_loc >= 0.55:
                return "Downside Reversal"

        return "Heavy Participation"

    if pctl <= low_cutoff:
        if np.isfinite(ret):
            if abs(ret) <= 0.35:
                return "Quiet Compression"
            if ret > 0 and is_above_20:
                return "Quiet Drift Up"
            if ret < 0 and is_below_20:
                return "Quiet Drift Down"

        return "Quiet Session"

    if is_above_20 and is_uptrend:
        return "Normal Uptrend"

    if is_below_20 and is_downtrend:
        return "Normal Downtrend"

    return "Normal"

def setup_color(setup: str, ret_1d: Optional[float] = None) -> str:
    setup = str(setup)

    constructive = {
        "Heavy Accumulation",
        "Downside Reversal",
        "Quiet Drift Up",
        "Normal Uptrend",
    }

    negative = {
        "Heavy Distribution",
        "Upside Rejection",
        "Quiet Drift Down",
        "Normal Downtrend",
    }

    amber = {
        "High Effort, Low Progress",
        "Heavy Participation",
    }

    quiet = {
        "Quiet Compression",
        "Quiet Session",
    }

    if setup in constructive:
        return PASTEL_GREEN

    if setup in negative:
        return PASTEL_RED

    if setup in amber:
        return AMBER

    if setup in quiet:
        return PASTEL_GREY

    if ret_1d is not None and np.isfinite(ret_1d):
        if ret_1d > 0:
            return PASTEL_GREEN
        if ret_1d < 0:
            return PASTEL_RED

    return PASTEL_GREY

def compute_forward_outcomes(
    df: pd.DataFrame, horizons: Tuple[int, ...] = (5, 10, 20, 63)
) -> pd.DataFrame:
    out = df.copy()

    for h in horizons:
        out[f"Fwd_{h}D"] = (out["Close"].shift(-h) / out["Close"] - 1.0) * 100.0

    close_arr = out["Close"].to_numpy(dtype=float)
    low_arr = out["Low"].to_numpy(dtype=float)
    high_arr = out["High"].to_numpy(dtype=float)

    max_dd_20 = np.full(len(out), np.nan)
    max_up_20 = np.full(len(out), np.nan)

    for i in range(len(out)):
        if not np.isfinite(close_arr[i]) or close_arr[i] <= 0:
            continue

        forward_low = low_arr[i + 1 : i + 21]
        forward_high = high_arr[i + 1 : i + 21]

        if len(forward_low) > 0 and np.isfinite(forward_low).any():
            max_dd_20[i] = (np.nanmin(forward_low) / close_arr[i] - 1.0) * 100.0

        if len(forward_high) > 0 and np.isfinite(forward_high).any():
            max_up_20[i] = (np.nanmax(forward_high) / close_arr[i] - 1.0) * 100.0

    out["Max_DD_20D"] = max_dd_20
    out["Max_Up_20D"] = max_up_20

    return out

def compute_volume_framework(
    df: pd.DataFrame,
    volume_mode: str,
    shares_outstanding: Optional[float],
    percentile_window: int,
    smooth_window: int,
    high_cutoff: float,
    low_cutoff: float,
) -> Tuple[pd.DataFrame, str, bool]:
    out = df.copy()

    raw_close = out.get("Raw_Close", out["Close"])
    out["Dollar_Volume"] = raw_close * out["Volume"]

    turnover_fallback = False

    if volume_mode == "Turnover %" and shares_outstanding and shares_outstanding > 0:
        out["Volume_Display"] = out["Volume"] / shares_outstanding * 100.0
        vol_label = "Turnover (% of shares outstanding)"
    elif volume_mode == "Dollar volume":
        out["Volume_Display"] = out["Dollar_Volume"].astype(float)
        vol_label = "Dollar volume"
    else:
        if volume_mode == "Turnover %":
            turnover_fallback = True

        out["Volume_Display"] = out["Volume"].astype(float)
        vol_label = "Volume (shares)"

    min_smooth = max(10, smooth_window // 2)

    out["Volume_Baseline"] = (
        out["Volume_Display"]
        .shift(1)
        .rolling(
            smooth_window,
            min_periods=min_smooth,
        )
        .median()
    )

    out["Volume_Ratio"] = out["Volume_Display"] / out["Volume_Baseline"].replace(
        0, np.nan
    )

    out["RVOL_20D"] = out["Volume_Display"] / out["Volume_Display"].shift(1).rolling(
        20,
        min_periods=10,
    ).median().replace(0, np.nan)

    out["RVOL_60D"] = out["Volume_Display"] / out["Volume_Display"].shift(1).rolling(
        60,
        min_periods=30,
    ).median().replace(0, np.nan)

    out["Volume_Pctl"] = rolling_percentile_previous(
        out["Volume_Ratio"],
        window=percentile_window,
        min_periods=max(40, percentile_window // 3),
        scale=100.0,
    )

    out["Ret_1D"] = out["Close"].pct_change() * 100.0
    out["Ret_5D"] = out["Close"].pct_change(5) * 100.0
    out["Ret_20D"] = out["Close"].pct_change(20) * 100.0

    out["Price_MA20"] = out["Close"].rolling(20, min_periods=20).mean()
    out["Price_MA50"] = out["Close"].rolling(50, min_periods=50).mean()
    out["Price_MA100"] = out["Close"].rolling(100, min_periods=100).mean()

    daily_range = (out["High"] - out["Low"]).replace(0, np.nan)
    out["Close_Location"] = ((out["Close"] - out["Low"]) / daily_range).clip(0, 1)
    out["Close_Location"] = out["Close_Location"].fillna(0.50)

    prev_close = out["Close"].shift(1)

    tr_components = pd.concat(
        [
            out["High"] - out["Low"],
            (out["High"] - prev_close).abs(),
            (out["Low"] - prev_close).abs(),
        ],
        axis=1,
    )

    out["True_Range"] = tr_components.max(axis=1)
    out["ATR20"] = out["True_Range"].rolling(20, min_periods=14).mean()
    out["Range_ATR20"] = out["True_Range"] / out["ATR20"].replace(0, np.nan)

    out["State"] = out["Volume_Pctl"].apply(
        lambda x: classify_state(x, high_cutoff, low_cutoff)
    )
    out["Setup"] = out.apply(
        lambda row: classify_setup(row, high_cutoff, low_cutoff), axis=1
    )

    out["Is_Heavy"] = out["State"].eq("Heavy")
    out["Is_Quiet"] = out["State"].eq("Quiet")
    out["Is_Extreme"] = out["State"].isin(["Heavy", "Quiet"])

    out = compute_forward_outcomes(out)

    return out, vol_label, turnover_fallback

def volume_bar_color(row: pd.Series) -> str:
    state = row.get("State", "Normal")
    ret = row.get("Ret_1D", np.nan)
    close_loc = row.get("Close_Location", np.nan)

    if state == "Heavy":
        if np.isfinite(ret) and np.isfinite(close_loc):
            if ret >= 0 and close_loc >= 0.50:
                return "rgba(79,118,95,0.72)"
            if ret < 0 or close_loc < 0.45:
                return "rgba(160,100,82,0.72)"

        return "rgba(176,137,88,0.68)"

    if state == "Quiet":
        return "rgba(148,163,184,0.26)"

    return "rgba(148,163,184,0.14)"

def build_chart(
    df: pd.DataFrame,
    symbol: str,
    vol_label: str,
    show_price_mas: bool,
    holiday_values: list,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.74, 0.26],
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name=symbol,
            line=dict(width=2.0, color="#111827"),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Close: %{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if show_price_mas:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Price_MA20"],
                mode="lines",
                name="20D",
                line=dict(width=1.15, color="rgba(82,111,143,0.72)"),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>20D: %{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Price_MA50"],
                mode="lines",
                name="50D",
                line=dict(width=1.15, color="rgba(107,114,128,0.80)"),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>50D: %{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    events = df[df["Is_Extreme"]].copy()

    if not events.empty:
        marker_colors = [
            setup_color(row["Setup"], row.get("Ret_1D", np.nan))
            for _, row in events.iterrows()
        ]

        marker_sizes = np.where(events["State"].eq("Heavy"), 8.0, 6.0)

        customdata = list(
            zip(
                events["Setup"].astype(str),
                events["Volume_Pctl"],
                events["Volume_Ratio"],
                events["Ret_1D"],
                events["Close_Location"] * 100.0,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=events.index,
                y=events["Close"],
                mode="markers",
                name="Extreme sessions",
                marker=dict(
                    size=marker_sizes,
                    color=marker_colors,
                    line=dict(width=0.8, color="white"),
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{x|%b %d, %Y}</b><br>"
                    "%{customdata[0]}"
                    "<br>Volume percentile: %{customdata[1]:.0f}"
                    "<br>Vs baseline: %{customdata[2]:.2f}x"
                    "<br>1D move: %{customdata[3]:+.2f}%"
                    "<br>Close location: %{customdata[4]:.0f}%"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    bar_colors = [volume_bar_color(row) for _, row in df.iterrows()]

    bar_customdata = list(
        zip(
            df["Setup"].astype(str),
            df["Volume_Pctl"],
            df["Volume_Ratio"],
            df["Ret_1D"],
        )
    )

    if "Turnover" in vol_label:
        bar_hover = (
            "<b>%{x|%b %d, %Y}</b><br>"
            "Turnover: %{y:.2f}%"
            "<br>%{customdata[0]}"
            "<br>Percentile: %{customdata[1]:.0f}"
            "<br>Vs baseline: %{customdata[2]:.2f}x"
            "<br>1D move: %{customdata[3]:+.2f}%"
            "<extra></extra>"
        )
        baseline_hover = "<b>%{x|%b %d, %Y}</b><br>Baseline: %{y:.2f}%<extra></extra>"
    elif "Dollar" in vol_label:
        bar_hover = (
            "<b>%{x|%b %d, %Y}</b><br>"
            "Dollar volume: $%{y:,.0f}"
            "<br>%{customdata[0]}"
            "<br>Percentile: %{customdata[1]:.0f}"
            "<br>Vs baseline: %{customdata[2]:.2f}x"
            "<br>1D move: %{customdata[3]:+.2f}%"
            "<extra></extra>"
        )
        baseline_hover = "<b>%{x|%b %d, %Y}</b><br>Baseline: $%{y:,.0f}<extra></extra>"
    else:
        bar_hover = (
            "<b>%{x|%b %d, %Y}</b><br>"
            "Volume: %{y:,.0f}"
            "<br>%{customdata[0]}"
            "<br>Percentile: %{customdata[1]:.0f}"
            "<br>Vs baseline: %{customdata[2]:.2f}x"
            "<br>1D move: %{customdata[3]:+.2f}%"
            "<extra></extra>"
        )
        baseline_hover = "<b>%{x|%b %d, %Y}</b><br>Baseline: %{y:,.0f}<extra></extra>"

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume_Display"],
            name="Participation",
            marker_color=bar_colors,
            customdata=bar_customdata,
            hovertemplate=bar_hover,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Volume_Baseline"],
            mode="lines",
            name="Baseline",
            line=dict(width=1.85, color=BLUE),
            hovertemplate=baseline_hover,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=740,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        dragmode="pan",
        xaxis_rangeslider_visible=False,
        bargap=0.06,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.012,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=11, color="#374151"),
        ),
    )

    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(values=holiday_values),
        ],
        showgrid=True,
        gridcolor="#f1f5f9",
        showline=False,
        zeroline=False,
        tickfont=dict(size=11, color="#6b7280"),
    )

    fig.update_yaxes(
        row=1,
        col=1,
        title_text="Price",
        showgrid=True,
        gridcolor="#eef2f7",
        zeroline=False,
        showline=False,
        tickfont=dict(size=11, color="#6b7280"),
        title_font=dict(size=11, color="#6b7280"),
    )

    fig.update_yaxes(
        row=2,
        col=1,
        title_text=vol_label,
        showgrid=True,
        gridcolor="#f3f4f6",
        zeroline=False,
        showline=False,
        tickfont=dict(size=10, color="#6b7280"),
        title_font=dict(size=11, color="#6b7280"),
    )

    if "Dollar" in vol_label:
        fig.update_yaxes(row=2, col=1, tickformat="$.2s")
    elif "Turnover" in vol_label:
        fig.update_yaxes(row=2, col=1, ticksuffix="%")
    else:
        fig.update_yaxes(row=2, col=1, tickformat=".2s")

    return fig

def build_recent_events(
    df: pd.DataFrame, event_filter: str, max_rows: int
) -> pd.DataFrame:
    events = df[df["State"].isin(["Heavy", "Quiet"])].copy()

    if event_filter == "Heavy only":
        events = events[events["State"] == "Heavy"].copy()
    elif event_filter == "Quiet only":
        events = events[events["State"] == "Quiet"].copy()

    if events.empty:
        return pd.DataFrame()

    return events.tail(max_rows).iloc[::-1].copy()

def build_setup_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize realized outcomes by setup using only rows with known futures."""
    realized = df.dropna(subset=["Setup", "Fwd_5D", "Fwd_20D", "Max_DD_20D"]).copy()
    realized = realized[~realized["Setup"].isin(["Unavailable", "Normal"])].copy()
    if realized.empty:
        return pd.DataFrame()

    grouped = realized.groupby("Setup", sort=False)
    out = grouped.agg(
        Observations=("Fwd_20D", "count"),
        **{
            "Avg 5D": ("Fwd_5D", "mean"),
            "Avg 20D": ("Fwd_20D", "mean"),
            "20D Hit Rate": ("Fwd_20D", lambda x: float((x > 0).mean() * 100.0)),
            "Median Max DD": ("Max_DD_20D", "median"),
        },
    ).reset_index()
    out = out[out["Observations"] >= 3].sort_values(
        ["Observations", "Avg 20D"], ascending=[False, False]
    )
    return out.reset_index(drop=True)
