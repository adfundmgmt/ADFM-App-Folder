from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from pandas_datareader import data as pdr
from plotly.subplots import make_subplots

from adfm_core.market_data import close_panel, configure_yfinance_cache, fetch_daily_ohlcv
from adfm_core.ui import (
    PageHeader,
    inject_explorer_style,
    render_footer,
    render_kpi_cards,
    render_page_header,
    render_section_header,
    render_selection_note,
)

configure_yfinance_cache()

TITLE = "Liquidity Conditions Monitor"
st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")
inject_explorer_style()

BLACK = "#111827"
BLUE = "#4472C4"
GREEN = "#70AD47"
RED = "#C00000"
ORANGE = "#ED7D31"
PURPLE = "#7030A0"
TEAL = "#008C95"
GRAY = "#6B7280"
GRID = "rgba(203, 213, 225, 0.62)"

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}&coed={end}"
FRED_START = "2010-01-01"
FRED_IDS = (
    "WRESBAL",
    "WALCL",
    "WTREGEN",
    "RRPONTSYD",
    "SOFR",
    "IORB",
    "EFFR",
    "BAMLH0A0HYM2",
    "BAMLC0A0CM",
    "DTWEXBGS",
    "DFII10",
)
FRED_LABELS = {
    "WRESBAL": "Reserve Balances",
    "WALCL": "Federal Reserve Total Assets",
    "WTREGEN": "Treasury General Account",
    "RRPONTSYD": "Overnight Reverse Repo",
    "SOFR": "Secured Overnight Financing Rate",
    "IORB": "Interest on Reserve Balances",
    "EFFR": "Effective Federal Funds Rate",
    "BAMLH0A0HYM2": "US High Yield OAS",
    "BAMLC0A0CM": "US Corporate OAS",
    "DTWEXBGS": "Broad US Dollar Index",
    "DFII10": "10-Year Real Yield",
}
FCIG_URLS = {
    "FCI-G Baseline": "https://www.federalreserve.gov/econres/notes/feds-notes/fci_g_public_monthly_3yr.csv",
    "FCI-G 1Y Lookback": "https://www.federalreserve.gov/econres/notes/feds-notes/fci_g_public_monthly_1yr.csv",
}

PRIMARY_SPECS: List[Dict[str, object]] = [
    dict(name="Reserve Balances", category="Balance Sheet", series="WRESBAL", orientation=1.0, weight=0.45, change_kind="diff", include_level=True, format="mm_tn", source="Federal Reserve H.4.1", description="Reserve balances are the banking system's settlement liquidity."),
    dict(name="Federal Reserve Assets", category="Balance Sheet", series="WALCL", orientation=1.0, weight=0.20, change_kind="diff", include_level=True, format="mm_tn", source="Federal Reserve H.4.1", description="Changes in Federal Reserve assets alter the supply of central-bank liabilities."),
    dict(name="Treasury General Account", category="Balance Sheet", series="WTREGEN", orientation=-1.0, weight=0.20, change_kind="diff", include_level=True, format="mm_tn", source="Federal Reserve H.4.1", description="A rising TGA drains reserves; a falling TGA adds reserves."),
    dict(name="Overnight Reverse Repo", category="Balance Sheet", series="RRPONTSYD", orientation=-1.0, weight=0.15, change_kind="diff", include_level=True, format="bn_tn", source="Federal Reserve Bank of New York", description="RRP runoff can release cash into reserves or private markets."),
    dict(name="SOFR minus IORB", category="Funding", formula="spread", inputs=("SOFR", "IORB"), orientation=-1.0, weight=0.60, change_kind="diff", include_level=True, format="pct_bp", source="Federal Reserve Bank of New York / Federal Reserve", description="A wider secured funding spread signals tighter reserve distribution."),
    dict(name="EFFR minus IORB", category="Funding", formula="spread", inputs=("EFFR", "IORB"), orientation=-1.0, weight=0.40, change_kind="diff", include_level=True, format="pct_bp", source="Federal Reserve Bank of New York / Federal Reserve", description="A wider unsecured policy spread indicates firmer overnight funding pressure."),
    dict(name="High Yield OAS", category="Transmission", series="BAMLH0A0HYM2", orientation=-1.0, weight=0.35, change_kind="diff", include_level=True, format="pct", source="ICE BofA / Federal Reserve FRED", description="Wider high-yield spreads transmit tighter financing conditions."),
    dict(name="Investment Grade OAS", category="Transmission", series="BAMLC0A0CM", orientation=-1.0, weight=0.20, change_kind="diff", include_level=True, format="pct", source="ICE BofA / Federal Reserve FRED", description="Investment-grade spreads capture broad corporate funding pressure."),
    dict(name="Broad US Dollar", category="Transmission", series="DTWEXBGS", orientation=-1.0, weight=0.25, change_kind="pct", include_level=True, format="index", source="Federal Reserve", description="A stronger broad dollar tightens global dollar liquidity."),
    dict(name="10-Year Real Yield", category="Transmission", series="DFII10", orientation=-1.0, weight=0.20, change_kind="diff", include_level=True, format="pct", source="US Treasury / Federal Reserve", description="Higher real yields tighten the economy's discount rate."),
]

MARKET_SPECS: List[Dict[str, object]] = [
    dict(name="Equal-Weight S&P / S&P 500", category="Market Confirmation", numerator="RSP", denominator="SPY", orientation=1.0, weight=0.18, change_kind="pct", include_level=False, description="Broad S&P participation."),
    dict(name="Small Caps / S&P 500", category="Market Confirmation", numerator="IWM", denominator="SPY", orientation=1.0, weight=0.18, change_kind="pct", include_level=False, description="Domestic cyclicality and financing sensitivity."),
    dict(name="Disruptive Growth / Nasdaq", category="Market Confirmation", numerator="ARKK", denominator="QQQ", orientation=1.0, weight=0.14, change_kind="pct", include_level=False, description="Speculative duration appetite."),
    dict(name="Biotech / Nasdaq", category="Market Confirmation", numerator="XBI", denominator="QQQ", orientation=1.0, weight=0.14, change_kind="pct", include_level=False, description="Financing-sensitive animal spirits."),
    dict(name="Regional Banks / S&P 500", category="Market Confirmation", numerator="KRE", denominator="SPY", orientation=1.0, weight=0.14, change_kind="pct", include_level=False, description="Bank-equity confirmation."),
    dict(name="Bitcoin / S&P 500", category="Market Confirmation", numerator="BTC-USD", denominator="SPY", orientation=1.0, weight=0.10, change_kind="pct", include_level=False, description="Crypto beta relative to equities."),
    dict(name="Emerging Markets / S&P 500", category="Market Confirmation", numerator="EEM", denominator="SPY", orientation=1.0, weight=0.07, change_kind="pct", include_level=False, description="Global dollar-liquidity confirmation."),
    dict(name="Volatility Pressure", category="Market Confirmation", ticker="^VIX", orientation=-1.0, weight=0.05, change_kind="pct", include_level=False, description="Lower volatility releases risk-budget capacity."),
]

SLEEVE_WEIGHTS = {
    "Balance Sheet": 0.35,
    "Funding": 0.25,
    "Transmission": 0.25,
    "Market Confirmation": 0.15,
}


def plot_layout(fig: go.Figure, height: int, margin: Optional[Dict[str, int]] = None, showlegend: bool = True, hovermode: str = "x unified") -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        autosize=True,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=margin or dict(l=50, r=36, t=82, b=48),
        font=dict(color="#334155", family="Arial, sans-serif"),
        hovermode=hovermode,
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.025, xanchor="left", x=0.0, font=dict(size=11), bgcolor="rgba(255,255,255,0)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(226,232,240,.48)", showline=True, linecolor="#cbd5e1", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, showline=False, zeroline=False)
    return fig


def latest(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def obs_change(series: pd.Series, periods: int) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.iloc[-1] - clean.iloc[-1 - periods]) if len(clean) > periods else np.nan


def fmt_score(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:+.2f}"


def fmt_pct(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.0f}%"


def fmt_raw(value: float, fmt: str) -> str:
    if pd.isna(value):
        return "N/A"
    if fmt == "mm_tn":
        return f"${value / 1_000_000:.2f}tn"
    if fmt == "bn_tn":
        return f"${value / 1_000:.2f}tn"
    if fmt == "pct_bp":
        return f"{value * 100:+.1f} bp"
    if fmt == "pct":
        return f"{value:.2f}%"
    return f"{value:.2f}"


def score_bucket(value: float) -> str:
    if pd.isna(value):
        return "Unavailable"
    if value >= 0.90:
        return "Strong easing"
    if value >= 0.35:
        return "Easing"
    if value > -0.35:
        return "Mixed"
    if value > -0.90:
        return "Tightening"
    return "Strong tightening"


def classify_regime(level: float, impulse: float, breadth: float) -> Tuple[str, str]:
    if pd.isna(level) or pd.isna(impulse):
        return "Unavailable", "Insufficient primary-source coverage."
    broad_up = pd.notna(breadth) and breadth >= 60
    broad_down = pd.notna(breadth) and breadth <= 40
    if level >= 0.35 and impulse >= 0.35 and broad_up:
        return "Liquidity Expansion", "Conditions are easy and improving with broad confirmation."
    if level >= 0.35 and impulse <= -0.35:
        return "Easy, Deteriorating", "Liquidity remains supportive, but the marginal impulse is rolling over."
    if level <= -0.35 and impulse >= 0.35:
        return "Tight, Improving", "Conditions remain restrictive, but the marginal impulse has turned positive."
    if level <= -0.35 and impulse <= -0.35 and broad_down:
        return "Liquidity Contraction", "Conditions are restrictive and becoming tighter across the major sleeves."
    if impulse >= 0.35:
        return "Improving", "The marginal impulse is positive, but the level is not yet easy."
    if impulse <= -0.35:
        return "Deteriorating", "The marginal impulse is negative, though the level is not yet deeply tight."
    return "Neutral / Mixed", "Level and impulse are near trailing norms or offsetting one another."


def zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    mean = clean.rolling(window, min_periods=min_periods).mean()
    std = clean.rolling(window, min_periods=min_periods).std()
    return ((clean - mean) / std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def change(series: pd.Series, periods: int, kind: str) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    return clean.pct_change(periods, fill_method=None) * 100 if kind == "pct" else clean.diff(periods)


def filter_lookback(obj: pd.Series | pd.DataFrame, lookback: str) -> pd.Series | pd.DataFrame:
    out = obj.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].sort_index()
    if out.empty or lookback == "max":
        return out
    offsets = {
        "6m": pd.DateOffset(months=6),
        "1y": pd.DateOffset(years=1),
        "2y": pd.DateOffset(years=2),
        "3y": pd.DateOffset(years=3),
        "5y": pd.DateOffset(years=5),
        "10y": pd.DateOffset(years=10),
    }
    return out.loc[out.index >= out.index.max() - offsets[lookback]]


def rebase(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    valid = clean.dropna()
    return clean / valid.iloc[0] * 100 if not valid.empty and valid.iloc[0] != 0 else pd.Series(index=clean.index, dtype=float)


def color_score(value: object) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    if pd.isna(x):
        return ""
    if x >= 0.90:
        return "background-color:#d9ead3;color:#274e13;"
    if x >= 0.35:
        return "background-color:#e2f0d9;color:#385723;"
    if x > -0.35:
        return "background-color:#f2f2f2;color:#404040;"
    if x > -0.90:
        return "background-color:#fce4d6;color:#843c0c;"
    return "background-color:#f4cccc;color:#990000;"


def _normalize_fred_frame(raw: pd.DataFrame, series_id: str) -> pd.Series:
    if raw is None or raw.empty:
        raise ValueError("empty response")
    frame = raw.copy()
    if series_id in frame.columns:
        series = frame[series_id]
    elif frame.shape[1] == 1:
        series = frame.iloc[:, 0]
    else:
        date_col = frame.columns[0]
        value_col = frame.columns[-1]
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame = frame.dropna(subset=[date_col]).set_index(date_col)
        series = frame[value_col]
    series = pd.to_numeric(series, errors="coerce")
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = series.loc[series.index.notna()].sort_index()
    try:
        if series.index.tz is not None:
            series.index = series.index.tz_convert(None)
    except Exception:
        pass
    series = series[~series.index.duplicated(keep="last")].dropna().rename(series_id)
    if series.empty:
        raise ValueError("no numeric observations")
    return series


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_fred_one(series_id: str, start: str, end: str) -> pd.Series:
    """Fetch one FRED series sequentially. Failed calls are not cached by Streamlit."""
    errors: List[str] = []

    try:
        raw = pdr.DataReader(series_id, "fred", start, end)
        return _normalize_fred_frame(raw, series_id)
    except Exception as exc:
        errors.append(f"pandas_datareader: {type(exc).__name__}: {exc}")

    try:
        response = requests.get(
            FRED_CSV_URL.format(series_id=series_id, start=start, end=end),
            headers={
                "User-Agent": "Mozilla/5.0 ADFM-Liquidity-Monitor/3.1",
                "Accept": "text/csv,application/octet-stream;q=0.9,*/*;q=0.8",
            },
            timeout=(8, 45),
        )
        response.raise_for_status()
        raw = pd.read_csv(BytesIO(response.content))
        return _normalize_fred_frame(raw, series_id)
    except Exception as exc:
        errors.append(f"direct CSV: {type(exc).__name__}: {exc}")

    raise RuntimeError(" | ".join(errors))


def load_fred(ids: Tuple[str, ...]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load series one at a time to avoid FRED throttling and retry transient failures on rerun."""
    end = pd.Timestamp.utcnow().date().isoformat()
    data: Dict[str, pd.Series] = {}
    errors: Dict[str, str] = {}

    for series_id in ids:
        try:
            data[series_id] = fetch_fred_one(series_id, FRED_START, end)
        except Exception as exc:
            errors[series_id] = str(exc)

    if not data:
        return pd.DataFrame(), errors

    panel = pd.concat(data.values(), axis=1).sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]
    business_index = pd.date_range(panel.index.min(), panel.index.max(), freq="B")
    panel = panel.reindex(business_index).ffill(limit=10)
    return panel.dropna(how="all"), errors


@st.cache_data(ttl=60 * 60 * 4, show_spinner=False)
def load_market(tickers: Tuple[str, ...], period: str) -> pd.DataFrame:
    frames, _ = fetch_daily_ohlcv(tickers, period=period)
    close = close_panel(frames, tickers, adjusted=True)
    if close.empty:
        return pd.DataFrame()
    close.index = pd.to_datetime(close.index, errors="coerce")
    close = close.loc[close.index.notna()].sort_index()
    close = close.loc[:, ~close.columns.duplicated(keep="last")]
    valid = [column for column in close.columns if pd.to_numeric(close[column], errors="coerce").notna().sum() >= 90]
    return close[valid].apply(pd.to_numeric, errors="coerce")


def market_tickers() -> List[str]:
    tickers = {"SPY", "QQQ"}
    for spec in MARKET_SPECS:
        if "ticker" in spec:
            tickers.add(str(spec["ticker"]))
        else:
            tickers.update((str(spec["numerator"]), str(spec["denominator"])))
    return sorted(tickers)


def build_primary(panel: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    series_map: Dict[str, pd.Series] = {}
    specs: List[Dict[str, object]] = []
    for raw in PRIMARY_SPECS:
        spec = dict(raw)
        if spec.get("formula") == "spread":
            left, right = tuple(spec["inputs"])
            if left not in panel or right not in panel:
                continue
            series = panel[left] - panel[right]
        else:
            series_id = str(spec["series"])
            if series_id not in panel:
                continue
            series = panel[series_id]
        if series.dropna().shape[0] >= 180:
            series_map[str(spec["name"])] = series
            specs.append(spec)
    return (pd.DataFrame(series_map).sort_index().dropna(how="all"), specs) if series_map else (pd.DataFrame(), [])


def build_market_components(prices: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    series_map: Dict[str, pd.Series] = {}
    specs: List[Dict[str, object]] = []
    for raw in MARKET_SPECS:
        spec = dict(raw)
        if "ticker" in spec:
            ticker = str(spec["ticker"])
            if ticker not in prices:
                continue
            series = prices[ticker]
            spec["display_ticker"] = ticker
        else:
            numerator, denominator = str(spec["numerator"]), str(spec["denominator"])
            if numerator not in prices or denominator not in prices:
                continue
            series = prices[numerator] / prices[denominator].replace(0, np.nan)
            spec["display_ticker"] = f"{numerator}/{denominator}"
        if series.dropna().shape[0] >= 180:
            series_map[str(spec["name"])] = series
            specs.append(spec)
    return (pd.DataFrame(series_map).sort_index().dropna(how="all"), specs) if series_map else (pd.DataFrame(), [])


def component_scores(components: pd.DataFrame, specs: Sequence[Mapping[str, object]], window: int, min_periods: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    levels = pd.DataFrame(index=components.index)
    impulses = pd.DataFrame(index=components.index)
    for spec in specs:
        name = str(spec["name"])
        raw = pd.to_numeric(components[name], errors="coerce")
        orientation = float(spec.get("orientation", 1.0))
        kind = str(spec.get("change_kind", "diff"))
        z21 = zscore(change(raw, 21, kind) * orientation, window, min_periods)
        z63 = zscore(change(raw, 63, kind) * orientation, window, min_periods)
        z126 = zscore(change(raw, 126, kind) * orientation, window, min_periods)
        impulses[name] = (0.50 * z21 + 0.35 * z63 + 0.15 * z126).clip(-3, 3)
        if bool(spec.get("include_level", True)):
            levels[name] = zscore(raw * orientation, window, min_periods).clip(-3, 3)
    return levels, impulses


def sleeve_composite(
    scores: pd.DataFrame,
    specs: Sequence[Mapping[str, object]],
    min_component_coverage: float,
    min_group_coverage: float,
    min_groups: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    groups: Dict[str, List[Dict[str, object]]] = {}
    for raw in specs:
        spec = dict(raw)
        name = str(spec["name"])
        if name in scores and scores[name].notna().any():
            groups.setdefault(str(spec["category"]), []).append(spec)

    sleeves = pd.DataFrame(index=scores.index)
    for group, members in groups.items():
        names = [str(member["name"]) for member in members]
        weights = pd.Series({str(member["name"]): float(member.get("weight", 1.0)) for member in members})
        total = float(weights.sum())

        def score_row(row: pd.Series) -> float:
            valid = row.dropna()
            if valid.empty:
                return np.nan
            active = weights.loc[valid.index]
            if float(active.sum()) / total < min_component_coverage:
                return np.nan
            return float((valid * active).sum() / active.sum())

        sleeves[group] = scores[names].apply(score_row, axis=1)

    if sleeves.empty:
        empty = pd.Series(index=scores.index, dtype=float)
        return sleeves, empty, empty, empty

    group_weights = pd.Series({group: SLEEVE_WEIGHTS[group] for group in sleeves.columns})
    total_weight = float(group_weights.sum())
    composite: List[float] = []
    breadth: List[float] = []
    coverage: List[float] = []

    for _, row in sleeves.iterrows():
        valid = row.dropna()
        if valid.empty:
            composite.append(np.nan)
            breadth.append(np.nan)
            coverage.append(np.nan)
            continue
        active = group_weights.loc[valid.index]
        active_weight = float(active.sum())
        cov = active_weight / total_weight
        coverage.append(cov * 100)
        if len(valid) < min_groups or cov < min_group_coverage:
            composite.append(np.nan)
            breadth.append(np.nan)
            continue
        composite.append(float((valid * active).sum() / active_weight))
        positive_weight = float(active.loc[valid.index[valid > 0]].sum())
        breadth.append(positive_weight / active_weight * 100)

    return (
        sleeves,
        pd.Series(composite, index=sleeves.index, dtype=float),
        pd.Series(breadth, index=sleeves.index, dtype=float),
        pd.Series(coverage, index=sleeves.index, dtype=float),
    )


def scorecard(components: pd.DataFrame, levels: pd.DataFrame, impulses: pd.DataFrame, specs: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        name = str(spec["name"])
        if name not in components:
            continue
        rows.append(
            {
                "Sleeve": str(spec["category"]),
                "Component": name,
                "Latest": fmt_raw(latest(components[name]), str(spec.get("format", "index"))),
                "Level Score": latest(levels[name]) if name in levels else np.nan,
                "Impulse Score": latest(impulses[name]) if name in impulses else np.nan,
                "Signal": score_bucket(latest(impulses[name])) if name in impulses else "Unavailable",
                "Within-Sleeve Weight": float(spec.get("weight", 1.0)),
                "Source": str(spec.get("source", "Yahoo Finance")),
                "Description": str(spec.get("description", "")),
            }
        )
    return pd.DataFrame(rows).sort_values(["Sleeve", "Impulse Score"], ascending=[True, False]).reset_index(drop=True) if rows else pd.DataFrame()


def fcig_column(frame: pd.DataFrame) -> Optional[str]:
    numeric = []
    for column in frame:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].notna().sum() >= 12:
            numeric.append(column)
    for term in ("fci-g index", "fci-g", "fcig", "fci_g", "fci g"):
        for column in numeric:
            lower = str(column).lower()
            if term in lower and "cont" not in lower:
                return column
    return numeric[0] if numeric else None


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_fcig() -> Tuple[pd.DataFrame, Dict[str, str]]:
    frames: List[pd.DataFrame] = []
    errors: Dict[str, str] = {}
    for label, url in FCIG_URLS.items():
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 ADFM-Liquidity-Monitor/3.1"}, timeout=(8, 45))
            response.raise_for_status()
            frame = pd.read_csv(BytesIO(response.content))
            date_col = next((column for column in frame if any(term in str(column).lower() for term in ("date", "month", "time"))), frame.columns[0])
            frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
            frame = frame.dropna(subset=[date_col]).set_index(date_col).sort_index()
            value_col = fcig_column(frame)
            if value_col is None:
                errors[label] = "No numeric FCI-G column."
            else:
                frames.append(frame[[value_col]].rename(columns={value_col: label}))
        except Exception as exc:
            errors[label] = str(exc)
    return (pd.concat(frames, axis=1).sort_index().dropna(how="all"), errors) if frames else (pd.DataFrame(), errors)


render_page_header(
    PageHeader(
        title=TITLE,
        description=(
            "Primary-source liquidity level and marginal impulse from Federal Reserve balance-sheet plumbing, "
            "overnight funding, credit spreads, the broad dollar, and real yields. Traded markets are retained "
            "as a separate confirmation sleeve rather than treated as the source of liquidity."
        ),
        eyebrow="ADFM Liquidity Regimes",
    )
)

with st.sidebar:
    st.markdown("## About This Tool")
    st.markdown(
        """
        **Purpose:** Separate the level of system liquidity from its marginal direction.

        - **35% Balance Sheet:** reserves, Fed assets, TGA, ON RRP.
        - **25% Funding:** SOFR and EFFR relative to IORB.
        - **25% Transmission:** HY OAS, IG OAS, broad dollar, real yields.
        - **15% Market Confirmation:** breadth, speculative beta, banks, EM, Bitcoin, volatility.

        Market prices confirm or reject the primary-source signal. They no longer define it.
        """
    )
    st.markdown("### Display Controls")
    lookback = st.selectbox("Display lookback", ["6m", "1y", "2y", "3y", "5y", "10y", "max"], index=2)
    market_period = st.selectbox("Market-data history", ["1y", "2y", "3y", "5y", "10y", "max"], index=3)
    benchmark = st.selectbox("Benchmark overlay", ["SPY", "QQQ"], index=0)
    z_window = st.number_input("Score lookback, business days", 252, 1260, 756, 21)
    min_periods = st.number_input("Minimum score observations", 126, 756, 252, 21)
    smoothing = st.number_input("Composite smoothing, business days", 1, 21, 3, 1)
    show_benchmark = st.checkbox("Show benchmark overlay", True)
    show_quadrant = st.checkbox("Show level-versus-impulse map", True)
    show_raw = st.checkbox("Show raw primary-source panels", True)
    show_scorecards = st.checkbox("Show component scorecards", True)
    show_fcig_overlay = st.checkbox("Show Fed FCI-G overlay", True)
    show_download = st.checkbox("Show download", True)
    st.caption("Regime labels use a level × impulse framework. The ±0.35 bands are transparent heuristic z-score thresholds.")

with st.spinner("Loading primary-source liquidity data..."):
    fred, fred_errors = load_fred(FRED_IDS)

if fred.empty:
    st.error("Primary-source liquidity data could not be loaded. The page will retry failed series on the next rerun rather than caching this failure.")
    if fred_errors:
        with st.expander("Primary-source diagnostics"):
            for series_id, error in fred_errors.items():
                st.write(f"**{FRED_LABELS.get(series_id, series_id)}:** {error}")
    st.stop()

primary, primary_specs = build_primary(fred)
if primary.empty:
    st.error("The primary liquidity components could not be constructed from the available series.")
    st.stop()

prices = load_market(tuple(market_tickers()), market_period)
market, market_specs = build_market_components(prices) if not prices.empty else (pd.DataFrame(), [])

primary_levels, primary_impulses = component_scores(primary, primary_specs, int(z_window), int(min_periods))
market_levels, market_impulses = component_scores(market, market_specs, int(z_window), int(min_periods)) if not market.empty else (pd.DataFrame(), pd.DataFrame())

all_impulses = pd.concat([primary_impulses, market_impulses], axis=1).sort_index()
all_specs = primary_specs + market_specs
sleeve_impulses, liquidity_impulse, easing_breadth, impulse_coverage = sleeve_composite(all_impulses, all_specs, 0.65, 0.70, 3)
sleeve_levels, liquidity_level, _, level_coverage = sleeve_composite(primary_levels, primary_specs, 0.65, 0.70, 2)

if int(smoothing) > 1:
    liquidity_impulse = liquidity_impulse.rolling(int(smoothing), min_periods=1).mean()
    liquidity_level = liquidity_level.rolling(int(smoothing), min_periods=1).mean()

market_confirmation = sleeve_impulses["Market Confirmation"] if "Market Confirmation" in sleeve_impulses else pd.Series(index=liquidity_impulse.index, dtype=float)

display_level = filter_lookback(liquidity_level, lookback)
display_impulse = filter_lookback(liquidity_impulse, lookback)
display_breadth = filter_lookback(easing_breadth, lookback)
display_coverage = filter_lookback(impulse_coverage, lookback)
display_market = filter_lookback(market_confirmation, lookback)
display_sleeve_levels = filter_lookback(sleeve_levels, lookback)
display_sleeve_impulses = filter_lookback(sleeve_impulses, lookback)
display_primary = filter_lookback(primary, lookback)
display_prices = filter_lookback(prices, lookback) if not prices.empty else pd.DataFrame()

current_level = latest(display_level)
current_impulse = latest(display_impulse)
current_breadth = latest(display_breadth)
current_coverage = latest(display_coverage)
current_market = latest(display_market)
regime, regime_description = classify_regime(current_level, current_impulse, current_breadth)
valid_latest_dates = [series.dropna().index.max() for series in (display_level, display_impulse) if not series.dropna().empty]
latest_date = max(valid_latest_dates) if valid_latest_dates else fred.index.max()

if fred_errors:
    st.warning(
        "Unavailable primary series: "
        + ", ".join(FRED_LABELS.get(series_id, series_id) for series_id in fred_errors)
        + ". Coverage tests prevent a partial sleeve from printing as a full signal."
    )

render_section_header(
    "Liquidity Regime Snapshot",
    f"Latest composite observation: {pd.Timestamp(latest_date).strftime('%b %d, %Y')}. Level is the starting point; impulse is the marginal direction.",
)
render_kpi_cards(
    [
        ("Regime", regime, regime_description),
        ("Liquidity Level", fmt_score(current_level), score_bucket(current_level)),
        ("Liquidity Impulse", fmt_score(current_impulse), f"1W {fmt_score(obs_change(display_impulse, 5))} · 1M {fmt_score(obs_change(display_impulse, 21))}"),
        ("Market Confirmation", fmt_score(current_market), score_bucket(current_market)),
        ("Weighted Breadth", fmt_pct(current_breadth), "Share of active sleeve weight easing"),
        ("Data Coverage", fmt_pct(current_coverage), f"{len(primary_specs) + len(market_specs)}/{len(PRIMARY_SPECS) + len(MARKET_SPECS)} components"),
    ]
)

latest_sleeves = pd.DataFrame(
    {
        "Level": display_sleeve_levels.apply(latest) if not display_sleeve_levels.empty else pd.Series(dtype=float),
        "Impulse": display_sleeve_impulses.apply(latest) if not display_sleeve_impulses.empty else pd.Series(dtype=float),
    }
)
if not latest_sleeves.empty and latest_sleeves["Impulse"].notna().any():
    strongest = latest_sleeves["Impulse"].idxmax()
    weakest = latest_sleeves["Impulse"].idxmin()
    sleeve_read = f"{strongest} is strongest at {latest_sleeves.loc[strongest, 'Impulse']:+.2f}; {weakest} is weakest at {latest_sleeves.loc[weakest, 'Impulse']:+.2f}."
else:
    sleeve_read = "Sleeve attribution is unavailable."
render_selection_note("Active liquidity read", f"{regime}: {regime_description} {sleeve_read}")

render_section_header(
    "Liquidity Level and Marginal Impulse",
    "The level score measures how easy or restrictive primary-source conditions are versus their trailing history. The impulse score measures whether those conditions are improving or deteriorating.",
)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.70, 0.30], specs=[[{"secondary_y": True}], [{}]], subplot_titles=("Level and impulse", "Weighted easing breadth"))
fig.add_hrect(y0=-0.35, y1=0.35, fillcolor="rgba(107,114,128,.07)", line_width=0, row=1, col=1)
fig.add_hline(y=0, line_dash="dot", line_color=GRAY, row=1, col=1)
fig.add_trace(go.Scatter(x=display_level.index, y=display_level, name="Liquidity Level", mode="lines", line=dict(color=BLUE, width=2.7)), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=display_impulse.index, y=display_impulse, name="Liquidity Impulse", mode="lines", line=dict(color=BLACK, width=2.9)), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=display_market.index, y=display_market, name="Market Confirmation", mode="lines", line=dict(color=ORANGE, width=1.8, dash="dot"), opacity=0.9), row=1, col=1, secondary_y=False)
if show_benchmark and benchmark in display_prices:
    benchmark_rebased = rebase(display_prices[benchmark])
    fig.add_trace(go.Scatter(x=benchmark_rebased.index, y=benchmark_rebased, name=f"{benchmark}, rebased", mode="lines", line=dict(color=TEAL, width=1.6), opacity=0.55), row=1, col=1, secondary_y=True)
fig.add_hline(y=50, line_dash="dot", line_color=GRAY, row=2, col=1)
fig.add_trace(go.Scatter(x=display_breadth.index, y=display_breadth, name="Weighted Breadth", mode="lines", fill="tozeroy", line=dict(color=PURPLE, width=2.0), opacity=0.75), row=2, col=1)
plot_layout(fig, 700, margin=dict(l=50, r=56, t=84, b=46))
fig.update_yaxes(title_text="Composite z-score", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text=f"{benchmark}, rebased", showgrid=False, row=1, col=1, secondary_y=True)
fig.update_yaxes(title_text="Breadth", range=[0, 100], ticksuffix="%", row=2, col=1)
st.plotly_chart(fig, width="stretch")

render_section_header("Current Sleeve Attribution", "The headline score is constructed sleeve first, so a category with more proxies cannot dominate by count.")
if not latest_sleeves.empty:
    sleeve_plot = latest_sleeves.reset_index().rename(columns={"index": "Sleeve"})
    fig_sleeves = go.Figure()
    fig_sleeves.add_vline(x=0, line_dash="dot", line_color=GRAY)
    fig_sleeves.add_trace(go.Bar(x=sleeve_plot["Impulse"], y=sleeve_plot["Sleeve"], orientation="h", name="Impulse", marker_color=BLUE))
    plot_layout(fig_sleeves, 350, margin=dict(l=135, r=30, t=45, b=42), showlegend=False, hovermode="closest")
    fig_sleeves.update_xaxes(title_text="Latest sleeve impulse")
    fig_sleeves.update_yaxes(showgrid=False)
    st.plotly_chart(fig_sleeves, width="stretch")

if show_quadrant and not latest_sleeves.empty:
    quadrant = latest_sleeves.dropna(subset=["Level", "Impulse"]).copy()
    if not quadrant.empty:
        render_section_header("Level × Impulse Map", "Upper right is easy and improving. Lower left is tight and deteriorating. Funding and transmission may sit in different quadrants during turning points.")
        fig_q = go.Figure()
        fig_q.add_hline(y=0, line_dash="dot", line_color=GRAY)
        fig_q.add_vline(x=0, line_dash="dot", line_color=GRAY)
        fig_q.add_trace(go.Scatter(x=quadrant["Level"], y=quadrant["Impulse"], mode="markers+text", text=quadrant.index, textposition="top center", marker=dict(size=14, color=[BLUE, ORANGE, PURPLE, TEAL][: len(quadrant)]), hovertemplate="%{text}<br>Level %{x:.2f}<br>Impulse %{y:.2f}<extra></extra>"))
        plot_layout(fig_q, 460, margin=dict(l=54, r=32, t=48, b=52), showlegend=False, hovermode="closest")
        fig_q.update_xaxes(title_text="Liquidity level")
        fig_q.update_yaxes(title_text="Liquidity impulse")
        st.plotly_chart(fig_q, width="stretch")

if show_raw:
    render_section_header("Primary-Source Plumbing and Transmission", "Raw series remain visible so the composite can be audited against the underlying reserve, funding, credit, dollar, and real-yield data.")
    raw_groups = {
        "Balance Sheet": ["Reserve Balances", "Federal Reserve Assets", "Treasury General Account", "Overnight Reverse Repo"],
        "Funding": ["SOFR minus IORB", "EFFR minus IORB"],
        "Transmission": ["High Yield OAS", "Investment Grade OAS", "Broad US Dollar", "10-Year Real Yield"],
    }
    for group, columns in raw_groups.items():
        available = [column for column in columns if column in display_primary]
        if not available:
            continue
        fig_raw = go.Figure()
        for column in available:
            normalized = zscore(display_primary[column], min(int(z_window), max(126, len(display_primary))), min(int(min_periods), max(63, len(display_primary) // 2)))
            fig_raw.add_trace(go.Scatter(x=normalized.index, y=normalized, name=column, mode="lines", line=dict(width=1.9)))
        fig_raw.add_hline(y=0, line_dash="dot", line_color=GRAY)
        plot_layout(fig_raw, 360, margin=dict(l=50, r=28, t=58, b=42))
        fig_raw.update_yaxes(title_text="Normalized level")
        st.plotly_chart(fig_raw, width="stretch")

primary_card = scorecard(primary, primary_levels, primary_impulses, primary_specs)
market_card = scorecard(market, market_levels, market_impulses, market_specs) if not market.empty else pd.DataFrame()
if show_scorecards:
    render_section_header("Component Audit", "Every component shows its latest raw level, level score, marginal impulse, source, weight, and interpretation.")
    tabs = st.tabs(["Primary Sources", "Market Confirmation", "Source Diagnostics"])
    with tabs[0]:
        if not primary_card.empty:
            numeric_cols = [column for column in ("Level Score", "Impulse Score") if column in primary_card]
            styled = primary_card.style.applymap(color_score, subset=numeric_cols).format({"Level Score": "{:+.2f}", "Impulse Score": "{:+.2f}", "Within-Sleeve Weight": "{:.0%}"}, na_rep="N/A")
            st.dataframe(styled, width="stretch", hide_index=True)
    with tabs[1]:
        if market_card.empty:
            st.info("Market confirmation data are unavailable.")
        else:
            styled = market_card.style.applymap(color_score, subset=["Impulse Score"]).format({"Impulse Score": "{:+.2f}", "Within-Sleeve Weight": "{:.0%}"}, na_rep="N/A")
            st.dataframe(styled, width="stretch", hide_index=True)
    with tabs[2]:
        diagnostics = pd.DataFrame(
            [
                {
                    "Series": FRED_LABELS.get(series_id, series_id),
                    "FRED ID": series_id,
                    "Status": "Unavailable" if series_id in fred_errors else "Loaded",
                    "Latest Observation": fred[series_id].dropna().index.max().date().isoformat() if series_id in fred and fred[series_id].notna().any() else "N/A",
                    "Error": fred_errors.get(series_id, ""),
                }
                for series_id in FRED_IDS
            ]
        )
        st.dataframe(diagnostics, width="stretch", hide_index=True)

if show_fcig_overlay:
    fcig, fcig_errors = load_fcig()
    render_section_header("Federal Reserve FCI-G Overlay", "FCI-G estimates the growth headwind or tailwind from financial conditions. It is kept separate from the liquidity index because it measures transmission rather than balance-sheet liquidity.")
    if fcig.empty:
        st.info("Federal Reserve FCI-G is temporarily unavailable. The primary liquidity composite above is unaffected.")
    else:
        fcig_display = filter_lookback(fcig, lookback)
        monthly_impulse = filter_lookback(display_impulse.resample("ME").last(), lookback)
        fig_fcig = make_subplots(specs=[[{"secondary_y": True}]])
        for column in fcig_display:
            fig_fcig.add_trace(go.Scatter(x=fcig_display.index, y=fcig_display[column], name=column, mode="lines", line=dict(width=2.0)), secondary_y=False)
        fig_fcig.add_trace(go.Scatter(x=monthly_impulse.index, y=monthly_impulse, name="Liquidity Impulse", mode="lines", line=dict(color=BLACK, width=2.3, dash="dot")), secondary_y=True)
        fig_fcig.add_hline(y=0, line_dash="dot", line_color=GRAY, secondary_y=False)
        plot_layout(fig_fcig, 430, margin=dict(l=52, r=58, t=68, b=44))
        fig_fcig.update_yaxes(title_text="FCI-G", secondary_y=False)
        fig_fcig.update_yaxes(title_text="Liquidity impulse", showgrid=False, secondary_y=True)
        st.plotly_chart(fig_fcig, width="stretch")

if show_download:
    render_section_header("Download", "Exports preserve numeric values for independent audit and backtesting.")
    export = pd.concat(
        {
            "Liquidity Level": liquidity_level,
            "Liquidity Impulse": liquidity_impulse,
            "Weighted Breadth": easing_breadth,
            "Coverage": impulse_coverage,
            "Market Confirmation": market_confirmation,
        },
        axis=1,
    ).reset_index(names="Date")
    st.download_button("Download liquidity history", export.to_csv(index=False).encode("utf-8"), "adfm_liquidity_conditions.csv", "text/csv")

render_footer(
    data_note=(
        "Primary inputs: Federal Reserve FRED and H.4.1 series, New York Fed overnight rates and reverse-repo usage, "
        "ICE BofA spread indexes distributed through FRED, Federal Reserve FCI-G, and Yahoo Finance market prices. "
        "Failed primary-source calls are retried on the next rerun and are never cached as successful data."
    )
)
