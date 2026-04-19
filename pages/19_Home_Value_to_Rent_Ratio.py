import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="Home Value to Rent Ratio",
    layout="wide",
)

APP_TITLE = "Home Value to Rent Ratio"
APP_SUBTITLE = "Housing valuation, affordability, and regime monitor"

CACHE_TTL_SECONDS = 60 * 60
HTTP_TIMEOUT = 25
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 1.25

LOCAL_CACHE_DIR = Path(".cache_fred_home_rent")
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOGO_PATH = Path("/mnt/data/0ea02e99-f067-4315-accc-0d2bbd3ee87d.png")

FRED_SERIES = {
    "home_index": "CSUSHPINSA",        # S&P CoreLogic Case-Shiller U.S. National Home Price Index
    "rent_index": "CUSR0000SEHA",      # CPI Rent of Primary Residence: U.S. city average
    "mortgage_30y": "MORTGAGE30US",    # 30-Year Fixed Rate Mortgage Average in the United States
    "recession": "USREC",              # NBER based recession indicators
}

USER_AGENT = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SeriesFetchResult:
    series_id: str
    data: pd.Series
    source: str
    fetched_at: pd.Timestamp
    last_observation: pd.Timestamp
    num_points: int
    frequency_hint: str

@dataclass
class DashboardData:
    monthly: pd.DataFrame
    annual: pd.DataFrame
    recession_spans: List[Tuple[pd.Timestamp, pd.Timestamp]]
    diagnostics: pd.DataFrame
    latest_snapshot: Dict[str, float]

# =============================================================================
# UTILS
# =============================================================================

def percentile_rank(series: pd.Series, value: float) -> float:
    s = series.dropna().astype(float)
    if s.empty or not np.isfinite(value):
        return np.nan
    return 100.0 * float((s <= value).sum()) / float(len(s))

def zscore_last(series: pd.Series) -> float:
    s = series.dropna().astype(float)
    if len(s) < 12:
        return np.nan
    std = float(s.std(ddof=0))
    if std == 0 or not np.isfinite(std):
        return np.nan
    return float((s.iloc[-1] - s.mean()) / std)

def format_num(x: float, digits: int = 1, suffix: str = "") -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}{suffix}"

def format_pct(x: float, digits: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}%"

def safe_first_valid(series: pd.Series) -> Optional[float]:
    s = series.dropna()
    if s.empty:
        return None
    return float(s.iloc[0])

def infer_frequency(index: pd.DatetimeIndex) -> str:
    if len(index) < 3:
        return "unknown"
    diffs = np.diff(index.sort_values().asi8) / 1e9 / 86400.0
    med_days = float(np.median(diffs))
    if med_days <= 8:
        return "weekly_or_daily"
    if med_days <= 40:
        return "monthly"
    if med_days <= 120:
        return "quarterly"
    return "low_frequency"

# =============================================================================
# FRED DATA LAYER
# =============================================================================

def _fred_csv_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

def _series_cache_path(series_id: str) -> Path:
    return LOCAL_CACHE_DIR / f"{series_id}.csv"

def _meta_cache_path(series_id: str) -> Path:
    return LOCAL_CACHE_DIR / f"{series_id}.meta.json"

def _write_local_cache(series_id: str, s: pd.Series, source: str) -> None:
    df = s.rename("value").to_frame()
    df.index.name = "date"
    df.to_csv(_series_cache_path(series_id))
    meta = {
        "series_id": series_id,
        "source": source,
        "fetched_at": pd.Timestamp.utcnow().isoformat(),
        "last_observation": str(df.index.max()),
        "num_points": int(len(df)),
    }
    _meta_cache_path(series_id).write_text(json.dumps(meta, indent=2))

def _read_local_cache(series_id: str) -> Optional[SeriesFetchResult]:
    csv_path = _series_cache_path(series_id)
    meta_path = _meta_cache_path(series_id)

    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if df.empty or "date" not in df.columns or "value" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna(subset=["date"]).set_index("date")["value"].dropna().sort_index()

    if s.empty:
        return None

    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}

    fetched_at = pd.to_datetime(meta.get("fetched_at", pd.Timestamp.utcnow()))
    last_obs = pd.to_datetime(meta.get("last_observation", s.index.max()))

    return SeriesFetchResult(
        series_id=series_id,
        data=s,
        source="local_cache",
        fetched_at=fetched_at,
        last_observation=last_obs,
        num_points=int(len(s)),
        frequency_hint=infer_frequency(s.index),
    )

def _download_fred_series(series_id: str) -> pd.Series:
    url = _fred_csv_url(series_id)
    last_err = None
    delay = RETRY_BACKOFF_SECONDS

    for _ in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=USER_AGENT, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            df = pd.read_csv(pd.io.common.BytesIO(r.content))
            if df.shape[1] < 2:
                raise ValueError(f"Unexpected FRED format for {series_id}")

            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

            s = (
                df.dropna(subset=["date"])
                .set_index("date")["value"]
                .dropna()
                .sort_index()
            )

            if s.empty:
                raise ValueError(f"Empty series returned for {series_id}")

            return s

        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay *= 2.0

    raise RuntimeError(f"Failed to fetch {series_id}: {last_err}")

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def fetch_fred_series(series_id: str) -> SeriesFetchResult:
    try:
        s = _download_fred_series(series_id)
        _write_local_cache(series_id, s, source="fred_live")
        return SeriesFetchResult(
            series_id=series_id,
            data=s,
            source="fred_live",
            fetched_at=pd.Timestamp.utcnow(),
            last_observation=s.index.max(),
            num_points=int(len(s)),
            frequency_hint=infer_frequency(s.index),
        )
    except Exception:
        cached = _read_local_cache(series_id)
        if cached is not None:
            return cached
        raise

# =============================================================================
# ANALYTICS LAYER
# =============================================================================

def to_monthly(series: pd.Series, method: str = "mean") -> pd.Series:
    s = series.sort_index()
    if method == "mean":
        return s.resample("MS").mean().dropna()
    if method == "last":
        return s.resample("MS").last().dropna()
    raise ValueError("method must be 'mean' or 'last'")

def recession_spans(usrec: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    x = usrec.copy().dropna()
    x = (x > 0.5).astype(int)

    spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    in_rec = False
    start = None

    for dt, val in x.items():
        if val == 1 and not in_rec:
            in_rec = True
            start = dt
        elif val == 0 and in_rec:
            in_rec = False
            spans.append((start, dt))
            start = None

    if in_rec and start is not None:
        spans.append((start, x.index.max() + pd.offsets.MonthBegin(1)))

    return spans

def mortgage_payment_per_100k(rate_pct: pd.Series, term_months: int = 360) -> pd.Series:
    r = (rate_pct / 100.0) / 12.0
    denom = 1.0 - (1.0 + r) ** (-term_months)
    factor = np.where(denom == 0, np.nan, r / denom)
    return pd.Series(100_000.0 * factor, index=rate_pct.index)

def build_analytics(
    normalize_ratio: bool,
    target_median: float,
    downturn_reference: float,
    affordability_baseline_year: int,
) -> DashboardData:
    fetched = {
        key: fetch_fred_series(series_id)
        for key, series_id in FRED_SERIES.items()
    }

    # Monthly aligned series
    home_m = to_monthly(fetched["home_index"].data, method="mean")
    rent_m = to_monthly(fetched["rent_index"].data, method="mean")
    mort_m = to_monthly(fetched["mortgage_30y"].data, method="mean")
    rec_m = to_monthly(fetched["recession"].data, method="mean")

    monthly = pd.concat(
        [home_m, rent_m, mort_m, rec_m],
        axis=1,
        join="inner",
    ).dropna()

    monthly.columns = ["home_index", "rent_index", "mortgage_rate", "recession"]

    if monthly.empty or len(monthly) < 24:
        raise ValueError("Merged monthly dataset is too short after alignment.")

    monthly["ratio_raw"] = monthly["home_index"] / monthly["rent_index"]

    raw_median = float(monthly["ratio_raw"].median())
    if np.isfinite(raw_median) and raw_median != 0:
        scale = float(target_median) / raw_median
    else:
        scale = 1.0

    monthly["ratio_display"] = (
        monthly["ratio_raw"] * scale if normalize_ratio else monthly["ratio_raw"]
    )

    monthly["ratio_12m_avg"] = monthly["ratio_display"].rolling(12, min_periods=6).mean()
    monthly["ratio_zscore"] = (
        (monthly["ratio_raw"] - monthly["ratio_raw"].mean()) / monthly["ratio_raw"].std(ddof=0)
    )

    monthly["home_yoy"] = monthly["home_index"].pct_change(12) * 100.0
    monthly["rent_yoy"] = monthly["rent_index"].pct_change(12) * 100.0
    monthly["spread_yoy"] = monthly["home_yoy"] - monthly["rent_yoy"]

    monthly["payment_per_100k"] = mortgage_payment_per_100k(monthly["mortgage_rate"])

    # Synthetic affordability index
    monthly["synthetic_payment_burden_raw"] = (
        monthly["payment_per_100k"] * monthly["home_index"]
    )

    baseline_year = int(affordability_baseline_year)
    baseline_slice = monthly.loc[monthly.index.year == baseline_year, "synthetic_payment_burden_raw"].dropna()

    if not baseline_slice.empty and float(baseline_slice.mean()) != 0:
        baseline_value = float(baseline_slice.mean())
    else:
        first_valid = safe_first_valid(monthly["synthetic_payment_burden_raw"])
        if first_valid is None or first_valid == 0:
            baseline_value = 1.0
        else:
            baseline_value = first_valid

    monthly["affordability_index"] = (
        monthly["synthetic_payment_burden_raw"] / baseline_value
    ) * 100.0

    annual = pd.DataFrame(index=pd.Index(sorted(monthly.index.year.unique()), name="year"))
    annual["ratio_display"] = monthly["ratio_display"].resample("YE").mean().values
    annual["ratio_raw"] = monthly["ratio_raw"].resample("YE").mean().values
    annual["mortgage_rate"] = monthly["mortgage_rate"].resample("YE").mean().values
    annual["home_yoy"] = monthly["home_yoy"].resample("YE").mean().values
    annual["rent_yoy"] = monthly["rent_yoy"].resample("YE").mean().values
    annual["spread_yoy"] = monthly["spread_yoy"].resample("YE").mean().values
    annual["affordability_index"] = monthly["affordability_index"].resample("YE").mean().values

    annual.index = annual.index.astype(int)

    rec_spans = recession_spans(monthly["recession"])

    latest = monthly.dropna().iloc[-1]
    prev_3m = monthly["ratio_raw"].shift(3).iloc[-1]
    prev_12m = monthly["ratio_raw"].shift(12).iloc[-1]

    latest_snapshot = {
        "latest_ratio_raw": float(latest["ratio_raw"]),
        "latest_ratio_display": float(latest["ratio_display"]),
        "ratio_percentile": percentile_rank(monthly["ratio_raw"], float(latest["ratio_raw"])),
        "ratio_zscore_latest": zscore_last(monthly["ratio_raw"]),
        "ratio_vs_3m": float(latest["ratio_raw"] - prev_3m) if np.isfinite(prev_3m) else np.nan,
        "ratio_vs_12m": float(latest["ratio_raw"] - prev_12m) if np.isfinite(prev_12m) else np.nan,
        "latest_home_yoy": float(latest["home_yoy"]),
        "latest_rent_yoy": float(latest["rent_yoy"]),
        "latest_spread_yoy": float(latest["spread_yoy"]),
        "latest_mortgage_rate": float(latest["mortgage_rate"]),
        "latest_affordability_index": float(latest["affordability_index"]),
        "display_downturn_reference": float(downturn_reference),
        "raw_ratio_median": float(monthly["ratio_raw"].median()),
        "display_ratio_median": float(monthly["ratio_display"].median()),
    }

    diagnostics_rows = []
    for name, result in fetched.items():
        diagnostics_rows.append(
            {
                "series_name": name,
                "series_id": result.series_id,
                "source": result.source,
                "fetched_at_utc": pd.to_datetime(result.fetched_at),
                "last_observation": pd.to_datetime(result.last_observation),
                "num_points": result.num_points,
                "frequency_hint": result.frequency_hint,
            }
        )

    diagnostics = pd.DataFrame(diagnostics_rows).sort_values("series_name").reset_index(drop=True)

    return DashboardData(
        monthly=monthly,
        annual=annual,
        recession_spans=rec_spans,
        diagnostics=diagnostics,
        latest_snapshot=latest_snapshot,
    )

# =============================================================================
# CHARTS
# =============================================================================

def add_recession_shapes(fig: go.Figure, spans: List[Tuple[pd.Timestamp, pd.Timestamp]], row: int, col: int) -> None:
    for start, end in spans:
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="lightgray",
            opacity=0.16,
            line_width=0,
            row=row,
            col=col,
        )

def build_ratio_chart(
    data: DashboardData,
    show_recessions: bool,
    show_mortgage_overlay: bool,
    normalize_ratio: bool,
    show_12m_avg: bool,
    downturn_reference: float,
) -> go.Figure:
    df = data.monthly.copy()

    ratio_col = "ratio_display"
    ratio_label = "Home / Rent Ratio"
    if normalize_ratio:
        ratio_label = "Home / Rent Ratio (Normalized)"

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if show_recessions:
        add_recession_shapes(fig, data.recession_spans, row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[ratio_col],
            mode="lines",
            name=ratio_label,
            line=dict(width=2.5),
            hovertemplate="%{x|%Y-%m}<br>Ratio: %{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )

    if show_12m_avg:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["ratio_12m_avg"],
                mode="lines",
                name="12M Avg",
                line=dict(width=2, dash="dot"),
                hovertemplate="%{x|%Y-%m}<br>12M Avg: %{y:.2f}<extra></extra>",
            ),
            secondary_y=False,
        )

    median_value = float(df[ratio_col].median())
    fig.add_hline(
        y=median_value,
        line_width=1.5,
        line_dash="dash",
        opacity=0.8,
        annotation_text=f"Median {median_value:.2f}",
        annotation_position="top left",
        secondary_y=False,
    )

    if normalize_ratio:
        fig.add_hline(
            y=downturn_reference,
            line_width=1.5,
            line_dash="dash",
            opacity=0.8,
            annotation_text=f"Downturn Ref {downturn_reference:.2f}",
            annotation_position="bottom left",
            secondary_y=False,
        )

    latest_x = df.index[-1]
    latest_y = float(df[ratio_col].iloc[-1])
    fig.add_trace(
        go.Scatter(
            x=[latest_x],
            y=[latest_y],
            mode="markers+text",
            name="Latest",
            text=[f"{latest_y:.2f}"],
            textposition="middle right",
            marker=dict(size=9),
            hovertemplate="%{x|%Y-%m}<br>Latest: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        secondary_y=False,
    )

    if show_mortgage_overlay:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["mortgage_rate"],
                mode="lines",
                name="30Y Mortgage Rate",
                line=dict(width=2, dash="dot"),
                hovertemplate="%{x|%Y-%m}<br>Mortgage: %{y:.2f}%<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title="Home Value to Rent Ratio",
        height=520,
        margin=dict(l=20, r=20, t=55, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text=ratio_label, secondary_y=False)
    fig.update_yaxes(title_text="Mortgage Rate (%)", secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="")
    return fig

def build_inflation_chart(data: DashboardData, show_recessions: bool) -> go.Figure:
    df = data.monthly.copy()

    fig = go.Figure()
    if show_recessions:
        for start, end in data.recession_spans:
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="lightgray",
                opacity=0.16,
                line_width=0,
            )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["home_yoy"],
            mode="lines",
            name="Home Price YoY",
            line=dict(width=2.4),
            hovertemplate="%{x|%Y-%m}<br>Home YoY: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["rent_yoy"],
            mode="lines",
            name="Rent YoY",
            line=dict(width=2.4),
            hovertemplate="%{x|%Y-%m}<br>Rent YoY: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["spread_yoy"],
            mode="lines",
            name="Spread",
            line=dict(width=2.0, dash="dot"),
            hovertemplate="%{x|%Y-%m}<br>Spread: %{y:.2f}%<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_width=1, opacity=0.7)

    fig.update_layout(
        title="Home vs Rent Inflation",
        height=500,
        margin=dict(l=20, r=20, t=55, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="YoY %")
    fig.update_xaxes(title_text="")
    return fig

def build_affordability_chart(data: DashboardData, show_recessions: bool, baseline_year: int) -> go.Figure:
    df = data.monthly.copy()

    fig = go.Figure()
    if show_recessions:
        for start, end in data.recession_spans:
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="lightgray",
                opacity=0.16,
                line_width=0,
            )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["affordability_index"],
            mode="lines",
            name=f"Affordability Index ({baseline_year}=100)",
            line=dict(width=2.6),
            hovertemplate="%{x|%Y-%m}<br>Index: %{y:.1f}<extra></extra>",
        )
    )

    latest_x = df.index[-1]
    latest_y = float(df["affordability_index"].iloc[-1])
    fig.add_trace(
        go.Scatter(
            x=[latest_x],
            y=[latest_y],
            mode="markers+text",
            name="Latest",
            text=[f"{latest_y:.0f}"],
            textposition="middle right",
            marker=dict(size=9),
            hovertemplate="%{x|%Y-%m}<br>Latest: %{y:.1f}<extra></extra>",
            showlegend=False,
        )
    )

    fig.add_hline(
        y=100,
        line_width=1.3,
        line_dash="dash",
        opacity=0.8,
        annotation_text="Baseline = 100",
        annotation_position="top left",
    )

    fig.update_layout(
        title="Synthetic Mortgage Payment Burden Index",
        height=500,
        margin=dict(l=20, r=20, t=55, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Index")
    fig.update_xaxes(title_text="")
    return fig

def build_distribution_chart(data: DashboardData) -> go.Figure:
    df = data.monthly.copy()

    latest_ratio = float(df["ratio_raw"].iloc[-1])
    pctl = percentile_rank(df["ratio_raw"], latest_ratio)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df["ratio_raw"],
            nbinsx=35,
            name="History",
            hovertemplate="Ratio bin: %{x}<br>Count: %{y}<extra></extra>",
        )
    )

    fig.add_vline(
        x=latest_ratio,
        line_width=2,
        line_dash="dash",
        annotation_text=f"Latest {latest_ratio:.2f} | {pctl:.0f}th pct",
        annotation_position="top",
    )

    fig.update_layout(
        title="Raw Ratio Distribution",
        height=400,
        margin=dict(l=20, r=20, t=55, b=20),
        bargap=0.05,
    )
    fig.update_xaxes(title_text="Raw Home / Rent Ratio")
    fig.update_yaxes(title_text="Count")
    return fig

# =============================================================================
# INTERPRETATION
# =============================================================================

def generate_regime_commentary(snapshot: Dict[str, float]) -> str:
    ratio_pct = snapshot["ratio_percentile"]
    spread = snapshot["latest_spread_yoy"]
    mort = snapshot["latest_mortgage_rate"]
    aff = snapshot["latest_affordability_index"]

    if np.isnan(ratio_pct):
        ratio_regime = "valuation signal unavailable"
    elif ratio_pct >= 85:
        ratio_regime = "housing valuation sits in the upper end of its own history"
    elif ratio_pct >= 65:
        ratio_regime = "housing valuation is above its long-run history"
    elif ratio_pct >= 35:
        ratio_regime = "housing valuation is near its historical middle"
    else:
        ratio_regime = "housing valuation is below its long-run history"

    if np.isnan(spread):
        spread_text = "The home-versus-rent spread is unavailable."
    elif spread > 1.5:
        spread_text = "Home-price inflation is still running ahead of rent inflation, which argues the ratio remains under upward pressure."
    elif spread > -1.5:
        spread_text = "Home-price inflation and rent inflation are moving close together, which suggests the ratio is no longer being pushed materially by relative price acceleration."
    else:
        spread_text = "Rent inflation is now outpacing home-price inflation, which usually helps mean reversion in the valuation ratio over time."

    if np.isnan(mort):
        mort_text = "Mortgage-rate context is unavailable."
    elif mort >= 7:
        mort_text = "Mortgage rates remain restrictive, so valuation resilience still depends on supply tightness, buyer mix, or eventual policy relief."
    elif mort >= 5.5:
        mort_text = "Mortgage rates remain meaningfully above the pre-2022 regime, which keeps affordability pressure elevated."
    else:
        mort_text = "Mortgage rates are no longer the same degree of headwind they were at the peak of the tightening shock."

    if np.isnan(aff):
        aff_text = "Affordability proxy is unavailable."
    elif aff >= 160:
        aff_text = "The synthetic burden index remains severely stretched versus the chosen baseline year."
    elif aff >= 125:
        aff_text = "The synthetic burden index is still materially above baseline, so affordability remains impaired."
    else:
        aff_text = "The synthetic burden index is closer to baseline than it was during the most stressed period."

    return (
        f"{ratio_regime}. {spread_text} {mort_text} {aff_text}"
    )

# =============================================================================
# UI
# =============================================================================

if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=70)

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        This dashboard tracks housing valuation through the relationship between home prices and rent,
        then layers in mortgage-rate pressure and a synthetic affordability burden proxy.

        It is built for regime monitoring rather than one-chart storytelling.

        Core design principles:
        - Raw ratio stays intact as the economic source of truth
        - Normalization is optional and clearly separated
        - Data quality is surfaced through diagnostics
        - Recession shading and overlays are optional
        """
    )

    st.markdown("---")
    st.subheader("Display")
    show_recessions = st.checkbox("Shade recessions", value=True)
    show_mortgage_overlay = st.checkbox("Overlay 30Y mortgage rate", value=True)
    show_12m_avg = st.checkbox("Show 12M moving average", value=True)

    st.markdown("---")
    st.subheader("Ratio settings")
    normalize_ratio = st.checkbox("Normalize ratio to target median", value=False)
    target_median = st.number_input("Target median", min_value=1.0, max_value=50.0, value=13.9, step=0.1)
    downturn_reference = st.number_input("Downturn reference", min_value=1.0, max_value=50.0, value=12.8, step=0.1)

    st.markdown("---")
    st.subheader("Affordability settings")
    affordability_baseline_year = st.number_input(
        "Affordability baseline year",
        min_value=1980,
        max_value=2100,
        value=2000,
        step=1,
    )

try:
    dashboard = build_analytics(
        normalize_ratio=normalize_ratio,
        target_median=float(target_median),
        downturn_reference=float(downturn_reference),
        affordability_baseline_year=int(affordability_baseline_year),
    )
except Exception as e:
    st.error(f"Failed to build dashboard: {e}")
    st.stop()

snapshot = dashboard.latest_snapshot
latest_date = dashboard.monthly.index.max()

# =============================================================================
# KPI ROW
# =============================================================================

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.metric(
        "Raw Ratio",
        format_num(snapshot["latest_ratio_raw"], 2),
        delta=format_num(snapshot["ratio_vs_12m"], 2),
    )

with c2:
    st.metric(
        "Historical Percentile",
        format_pct(snapshot["ratio_percentile"], 0),
    )

with c3:
    st.metric(
        "Z-Score",
        format_num(snapshot["ratio_zscore_latest"], 2),
    )

with c4:
    st.metric(
        "Mortgage Rate",
        format_pct(snapshot["latest_mortgage_rate"], 2),
    )

with c5:
    st.metric(
        "Home minus Rent YoY",
        format_pct(snapshot["latest_spread_yoy"], 2),
    )

with c6:
    st.metric(
        "Affordability Index",
        format_num(snapshot["latest_affordability_index"], 0),
    )

st.markdown(
    f"**Latest monthly observation:** {latest_date.strftime('%B %Y')}"
)

st.info(generate_regime_commentary(snapshot))

# =============================================================================
# CHARTS
# =============================================================================

fig_ratio = build_ratio_chart(
    data=dashboard,
    show_recessions=show_recessions,
    show_mortgage_overlay=show_mortgage_overlay,
    normalize_ratio=normalize_ratio,
    show_12m_avg=show_12m_avg,
    downturn_reference=float(downturn_reference),
)
st.plotly_chart(fig_ratio, use_container_width=True)

left, right = st.columns(2)

with left:
    fig_infl = build_inflation_chart(dashboard, show_recessions=show_recessions)
    st.plotly_chart(fig_infl, use_container_width=True)

with right:
    fig_aff = build_affordability_chart(
        dashboard,
        show_recessions=show_recessions,
        baseline_year=int(affordability_baseline_year),
    )
    st.plotly_chart(fig_aff, use_container_width=True)

fig_dist = build_distribution_chart(dashboard)
st.plotly_chart(fig_dist, use_container_width=True)

# =============================================================================
# DATA TABLES / DIAGNOSTICS
# =============================================================================

with st.expander("Current snapshot table", expanded=False):
    snapshot_table = pd.DataFrame(
        {
            "Metric": [
                "Latest raw ratio",
                "Latest display ratio",
                "Raw ratio historical percentile",
                "Raw ratio z-score",
                "Raw ratio median",
                "Display ratio median",
                "Home YoY",
                "Rent YoY",
                "Home minus Rent spread",
                "30Y mortgage rate",
                "Affordability index",
            ],
            "Value": [
                snapshot["latest_ratio_raw"],
                snapshot["latest_ratio_display"],
                snapshot["ratio_percentile"],
                snapshot["ratio_zscore_latest"],
                snapshot["raw_ratio_median"],
                snapshot["display_ratio_median"],
                snapshot["latest_home_yoy"],
                snapshot["latest_rent_yoy"],
                snapshot["latest_spread_yoy"],
                snapshot["latest_mortgage_rate"],
                snapshot["latest_affordability_index"],
            ],
        }
    )
    st.dataframe(snapshot_table, use_container_width=True, hide_index=True)

with st.expander("Diagnostics", expanded=False):
    diag = dashboard.diagnostics.copy()
    diag["fetched_at_utc"] = pd.to_datetime(diag["fetched_at_utc"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    diag["last_observation"] = pd.to_datetime(diag["last_observation"]).dt.strftime("%Y-%m-%d")
    st.dataframe(diag, use_container_width=True, hide_index=True)

    merged_info = pd.DataFrame(
        {
            "Field": [
                "Merged monthly start",
                "Merged monthly end",
                "Merged monthly rows",
                "Annual rows",
                "Normalization enabled",
                "Affordability baseline year",
            ],
            "Value": [
                str(dashboard.monthly.index.min().date()),
                str(dashboard.monthly.index.max().date()),
                int(len(dashboard.monthly)),
                int(len(dashboard.annual)),
                str(bool(normalize_ratio)),
                int(affordability_baseline_year),
            ],
        }
    )
    st.dataframe(merged_info, use_container_width=True, hide_index=True)

with st.expander("Methodology", expanded=False):
    st.markdown(
        """
        **Raw ratio**
        
        The core signal is the monthly U.S. national home price index divided by the CPI rent index.
        This is a valuation proxy, not a direct cap-rate or cash-flow measure.

        **Normalized ratio**
        
        When enabled, the raw ratio is scaled so its full-sample median equals the user-selected target.
        This is a presentation transform only. It should not be treated as the economic truth series.

        **Inflation spread**
        
        Home-price inflation and rent inflation are each measured as 12-month % change.
        The spread is home-price YoY minus rent YoY.

        **Synthetic affordability index**
        
        This uses the standard monthly payment formula for a 30-year fixed mortgage per $100k of principal,
        then multiplies that by the home price index level. The result is indexed to the chosen baseline year.
        It is a directional burden proxy, not a literal household payment series.

        **Recession shading**
        
        Recessions are shaded using the FRED USREC monthly recession indicator.

        **Data source**
        
        FRED series used:
        - CSUSHPINSA
        - CUSR0000SEHA
        - MORTGAGE30US
        - USREC
        """
    )

with st.expander("Underlying monthly dataset", expanded=False):
    export_df = dashboard.monthly.reset_index().rename(columns={"index": "date"})
    st.dataframe(export_df.tail(120), use_container_width=True, hide_index=True)

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download monthly dataset as CSV",
        data=csv_bytes,
        file_name="home_value_to_rent_ratio_monthly.csv",
        mime="text/csv",
    )

st.caption(
    "Source: FRED. This dashboard uses public macro data and a synthetic affordability framework intended for monitoring and comparative analysis."
)
st.caption("© 2026 AD Fund Management LP")
