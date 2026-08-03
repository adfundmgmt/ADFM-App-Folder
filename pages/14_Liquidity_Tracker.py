from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
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

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
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
    dict(
        name="Reserve Balances",
        category="Balance Sheet",
        series="WRESBAL",
        orientation=1.0,
        weight=0.45,
        change_kind="diff",
        include_level=True,
        format="mm_tn",
        source="Federal Reserve H.4.1",
        description="Directly observed reserve balances held at Federal Reserve Banks.",
    ),
    dict(
        name="Fed Total Assets",
        category="Balance Sheet",
        series="WALCL",
        orientation=1.0,
        weight=0.25,
        change_kind="diff",
        include_level=True,
        format="mm_tn",
        source="Federal Reserve H.4.1",
        description="Federal Reserve balance-sheet expansion or contraction.",
    ),
    dict(
        name="Treasury General Account",
        category="Balance Sheet",
        series="WTREGEN",
        orientation=-1.0,
        weight=0.15,
        change_kind="diff",
        include_level=False,
        format="mm_tn",
        source="Federal Reserve H.4.1",
        description="A rising TGA drains reserves; a falling TGA injects reserves.",
    ),
    dict(
        name="ON RRP",
        category="Balance Sheet",
        series="RRPONTSYD",
        orientation=-1.0,
        weight=0.15,
        change_kind="diff",
        include_level=False,
        format="bn_tn",
        source="Federal Reserve Bank of New York",
        description="RRP runoff shifts Federal Reserve liabilities toward reserves.",
    ),
    dict(
        name="SOFR minus IORB",
        category="Funding",
        inputs=("SOFR", "IORB"),
        formula="spread",
        orientation=-1.0,
        weight=0.60,
        change_kind="diff",
        include_level=True,
        format="pct_bp",
        source="New York Fed / Federal Reserve Board",
        description="Secured overnight funding pressure versus the administered reserve rate.",
    ),
    dict(
        name="EFFR minus IORB",
        category="Funding",
        inputs=("EFFR", "IORB"),
        formula="spread",
        orientation=-1.0,
        weight=0.40,
        change_kind="diff",
        include_level=True,
        format="pct_bp",
        source="New York Fed / Federal Reserve Board",
        description="Unsecured overnight funding pressure versus the administered reserve rate.",
    ),
    dict(
        name="High Yield OAS",
        category="Transmission",
        series="BAMLH0A0HYM2",
        orientation=-1.0,
        weight=0.35,
        change_kind="diff",
        include_level=True,
        format="pct",
        source="ICE BofA via FRED",
        description="Direct high-yield credit-risk compensation.",
    ),
    dict(
        name="Investment Grade OAS",
        category="Transmission",
        series="BAMLC0A0CM",
        orientation=-1.0,
        weight=0.20,
        change_kind="diff",
        include_level=True,
        format="pct",
        source="ICE BofA via FRED",
        description="Broad investment-grade corporate funding conditions.",
    ),
    dict(
        name="Broad Dollar",
        category="Transmission",
        series="DTWEXBGS",
        orientation=-1.0,
        weight=0.25,
        change_kind="pct",
        include_level=True,
        format="index",
        source="Federal Reserve Board",
        description="A stronger broad dollar tightens global dollar liquidity.",
    ),
    dict(
        name="10-Year Real Yield",
        category="Transmission",
        series="DFII10",
        orientation=-1.0,
        weight=0.20,
        change_kind="diff",
        include_level=True,
        format="pct",
        source="US Treasury / Federal Reserve",
        description="Higher real yields tighten the economy's discount rate.",
    ),
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


def plot_layout(
    fig: go.Figure,
    height: int,
    margin: Optional[Dict[str, int]] = None,
    showlegend: bool = True,
    hovermode: str = "x unified",
) -> go.Figure:
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.025,
            xanchor="left",
            x=0.0,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0)",
        ),
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


def _fred_series(series_id: str) -> Tuple[str, pd.Series, Optional[str]]:
    try:
        response = requests.get(
            FRED_URL.format(series_id=series_id),
            headers={"User-Agent": "ADFM-Liquidity-Monitor/3.0"},
            timeout=(4, 18),
        )
        response.raise_for_status()
        frame = pd.read_csv(BytesIO(response.content))
        if frame.empty or frame.shape[1] < 2:
            return series_id, pd.Series(dtype=float), "Empty response."
        date_col = frame.columns[0]
        value_col = series_id if series_id in frame.columns else frame.columns[-1]
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
        series = frame.dropna(subset=[date_col]).set_index(date_col)[value_col].sort_index().rename(series_id)
        series = series[~series.index.duplicated(keep="last")]
        return (series_id, series, None) if series.notna().any() else (series_id, pd.Series(dtype=float), "No numeric observations.")
    except Exception as exc:
        return series_id, pd.Series(dtype=float), str(exc)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_fred(ids: Tuple[str, ...]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    data: Dict[str, pd.Series] = {}
    errors: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(6, len(ids))) as executor:
        futures = [executor.submit(_fred_series, series_id) for series_id in ids]
        for future in as_completed(futures):
            series_id, series, error = future.result()
            if error:
                errors[series_id] = error
            else:
                data[series_id] = series
    if not data:
        return pd.DataFrame(), errors
    panel = pd.concat(data.values(), axis=1).sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]
    panel = panel.reindex(pd.date_range(panel.index.min(), panel.index.max(), freq="B")).ffill(limit=10)
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
    return close[[column for column in close.columns if pd.to_numeric(close[column], errors="coerce").notna().sum() >= 90]].apply(pd.to_numeric, errors="coerce")


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


def component_scores(
    components: pd.DataFrame,
    specs: Sequence[Mapping[str, object]],
    window: int,
    min_periods: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    composite, breadth, coverage = [], [], []
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
        breadth.append(float(active.loc[valid.index[valid > 0]].sum()) / active_weight * 100)
    return (
        sleeves,
        pd.Series(composite, index=sleeves.index, dtype=float),
        pd.Series(breadth, index=sleeves.index, dtype=float),
        pd.Series(coverage, index=sleeves.index, dtype=float),
    )


def scorecard(
    components: pd.DataFrame,
    levels: pd.DataFrame,
    impulses: pd.DataFrame,
    specs: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
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
    frames, errors = [], {}
    for label, url in FCIG_URLS.items():
        try:
            response = requests.get(url, headers={"User-Agent": "ADFM-Liquidity-Monitor/3.0"}, timeout=(4, 20))
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
    st.error("Primary-source liquidity data could not be loaded from FRED.")
    st.stop()

primary, primary_specs = build_primary(fred)
if primary.empty:
    st.error("The primary liquidity components could not be constructed.")
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
latest_date = max(series.dropna().index.max() for series in (display_level, display_impulse) if not series.dropna().empty)

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
    "The old risk-appetite composite is retained only as market confirmation and capped at 15% of the headline impulse.",
)
benchmark_series = rebase(display_prices[benchmark]).dropna() if show_benchmark and benchmark in display_prices else None
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09, row_heights=[0.74, 0.26], specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
for y in (0, 0.35, -0.35):
    fig.add_hline(y=y, line_width=1, line_dash="dot", line_color=GRAY if y == 0 else "#aab7c4", row=1, col=1)
fig.add_trace(go.Scatter(x=display_level.index, y=display_level, name="Liquidity Level", mode="lines", line=dict(color=BLUE, width=2.8), hovertemplate="%{x|%Y-%m-%d}<br>Level: %{y:.2f}<extra></extra>"), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=display_impulse.index, y=display_impulse, name="Liquidity Impulse", mode="lines", line=dict(color=GREEN, width=2.8), hovertemplate="%{x|%Y-%m-%d}<br>Impulse: %{y:.2f}<extra></extra>"), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=display_market.index, y=display_market, name="Market Confirmation", mode="lines", line=dict(color=ORANGE, width=1.8, dash="dash"), opacity=0.82, hovertemplate="%{x|%Y-%m-%d}<br>Confirmation: %{y:.2f}<extra></extra>"), row=1, col=1, secondary_y=False)
if benchmark_series is not None and not benchmark_series.empty:
    fig.add_trace(go.Scatter(x=benchmark_series.index, y=benchmark_series, name=f"{benchmark}, rebased", mode="lines", line=dict(color=PURPLE, width=1.6), opacity=0.45, hovertemplate=f"%{{x|%Y-%m-%d}}<br>{benchmark}: %{{y:.1f}}<extra></extra>"), row=1, col=1, secondary_y=True)
fig.add_hline(y=50, line_width=1, line_dash="dot", line_color=GRAY, row=2, col=1)
fig.add_trace(go.Scatter(x=display_breadth.index, y=display_breadth, name="Weighted Easing Breadth", mode="lines", line=dict(color=TEAL, width=2.1), fill="tozeroy", opacity=0.78, hovertemplate="%{x|%Y-%m-%d}<br>Breadth: %{y:.0f}%<extra></extra>"), row=2, col=1)
plot_layout(fig, 700, dict(l=52, r=62, t=88, b=48))
fig.update_yaxes(title_text="Trailing z-score", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text=f"{benchmark}, rebased", showgrid=False, row=1, col=1, secondary_y=True)
fig.update_yaxes(title_text="Weighted breadth", ticksuffix="%", range=[0, 100], row=2, col=1)
fig.update_xaxes(title_text="Date", row=2, col=1)
st.plotly_chart(fig, width="stretch")

if show_quadrant:
    render_section_header(
        "Liquidity Regime Map",
        "Upper-right is easy and improving. Lower-left is tight and deteriorating. The path shows the trailing six months.",
    )
    regime_frame = pd.concat([display_level.rename("Level"), display_impulse.rename("Impulse")], axis=1).dropna().tail(126)
    quadrant = go.Figure()
    for x0, x1, y0, y1, fill in [
        (0, 4, 0, 4, "rgba(112,173,71,.10)"),
        (0, 4, -4, 0, "rgba(237,125,49,.09)"),
        (-4, 0, 0, 4, "rgba(68,114,196,.09)"),
        (-4, 0, -4, 0, "rgba(192,0,0,.08)"),
    ]:
        quadrant.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=fill, line_width=0, layer="below")
    if not regime_frame.empty:
        quadrant.add_trace(go.Scatter(x=regime_frame["Level"], y=regime_frame["Impulse"], mode="lines+markers", name="Trailing 6M path", line=dict(color="#94a3b8", width=1.6), marker=dict(size=4), customdata=regime_frame.index.strftime("%Y-%m-%d"), hovertemplate="%{customdata}<br>Level: %{x:.2f}<br>Impulse: %{y:.2f}<extra></extra>"))
        quadrant.add_trace(go.Scatter(x=[regime_frame["Level"].iloc[-1]], y=[regime_frame["Impulse"].iloc[-1]], mode="markers+text", name="Current", marker=dict(size=14, color=BLACK, line=dict(color="#fff", width=1.5)), text=["Current"], textposition="top center"))
    quadrant.add_hline(y=0, line_color=GRAY)
    quadrant.add_vline(x=0, line_color=GRAY)
    quadrant.add_annotation(x=2.6, y=3.4, text="Easy + Improving", showarrow=False)
    quadrant.add_annotation(x=2.6, y=-3.4, text="Easy + Deteriorating", showarrow=False)
    quadrant.add_annotation(x=-2.6, y=3.4, text="Tight + Improving", showarrow=False)
    quadrant.add_annotation(x=-2.6, y=-3.4, text="Tight + Deteriorating", showarrow=False)
    plot_layout(quadrant, 560, dict(l=58, r=36, t=70, b=55), hovermode="closest")
    quadrant.update_xaxes(title_text="Liquidity Level", range=[-4, 4])
    quadrant.update_yaxes(title_text="Liquidity Impulse", range=[-4, 4], scaleanchor="x", scaleratio=1)
    st.plotly_chart(quadrant, width="stretch")

render_section_header(
    "Liquidity Attribution by Sleeve",
    "The page shows the four sleeves separately so a market rally cannot conceal tightening in funding or balance-sheet liquidity.",
)
attribution = latest_sleeves.reset_index().rename(columns={"index": "Sleeve"})
if not attribution.empty:
    order = [sleeve for sleeve in SLEEVE_WEIGHTS if sleeve in attribution["Sleeve"].tolist()]
    attribution["Sleeve"] = pd.Categorical(attribution["Sleeve"], categories=order, ordered=True)
    attribution = attribution.sort_values("Sleeve")
    bars = go.Figure()
    bars.add_hline(y=0, line_width=1, line_dash="dot", line_color=GRAY)
    bars.add_trace(go.Bar(x=attribution["Sleeve"], y=attribution["Level"], name="Level", marker_color=BLUE))
    bars.add_trace(go.Bar(x=attribution["Sleeve"], y=attribution["Impulse"], name="Impulse", marker_color=GREEN))
    plot_layout(bars, 450, dict(l=48, r=30, t=78, b=70), hovermode="closest")
    bars.update_layout(barmode="group")
    bars.update_yaxes(title_text="Latest sleeve z-score")
    st.plotly_chart(bars, width="stretch")

if show_raw:
    render_section_header(
        "Primary-Source Liquidity Plumbing",
        "Raw quantities and funding spreads are shown separately from the standardized composite.",
    )
    tab_balance, tab_funding, tab_transmission = st.tabs(["Balance Sheet", "Funding", "Transmission"])
    with tab_balance:
        balance = pd.DataFrame(index=display_primary.index)
        conversions = {
            "Reserve Balances": 1_000_000,
            "Fed Total Assets": 1_000_000,
            "Treasury General Account": 1_000_000,
            "ON RRP": 1_000,
        }
        colors = {
            "Reserve Balances": BLUE,
            "Fed Total Assets": BLACK,
            "Treasury General Account": ORANGE,
            "ON RRP": PURPLE,
        }
        for name, divisor in conversions.items():
            if name in display_primary:
                balance[name] = display_primary[name] / divisor
        chart = go.Figure()
        for name in balance:
            chart.add_trace(go.Scatter(x=balance.index, y=balance[name], name=name, mode="lines", line=dict(color=colors[name], width=2.1), hovertemplate=f"%{{x|%Y-%m-%d}}<br>{name}: $%{{y:.2f}}tn<extra></extra>"))
        plot_layout(chart, 500)
        chart.update_yaxes(title_text="$ trillions")
        st.plotly_chart(chart, width="stretch")
    with tab_funding:
        funding = pd.DataFrame(index=display_primary.index)
        for name in ("SOFR minus IORB", "EFFR minus IORB"):
            if name in display_primary:
                funding[name] = display_primary[name] * 100
        chart = go.Figure()
        for name, color in zip(funding, (RED, ORANGE)):
            chart.add_trace(go.Scatter(x=funding.index, y=funding[name], name=name, mode="lines", line=dict(color=color, width=2.2), hovertemplate=f"%{{x|%Y-%m-%d}}<br>{name}: %{{y:.1f}} bp<extra></extra>"))
        chart.add_hline(y=0, line_width=1, line_dash="dot", line_color=GRAY)
        plot_layout(chart, 470)
        chart.update_yaxes(title_text="Basis points")
        st.plotly_chart(chart, width="stretch")
    with tab_transmission:
        chart = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10, specs=[[{"secondary_y": False}], [{"secondary_y": True}]])
        if "High Yield OAS" in display_primary:
            chart.add_trace(go.Scatter(x=display_primary.index, y=display_primary["High Yield OAS"], name="High Yield OAS", mode="lines", line=dict(color=RED, width=2.2)), row=1, col=1)
        if "Investment Grade OAS" in display_primary:
            chart.add_trace(go.Scatter(x=display_primary.index, y=display_primary["Investment Grade OAS"], name="Investment Grade OAS", mode="lines", line=dict(color=ORANGE, width=2.0)), row=1, col=1)
        if "Broad Dollar" in display_primary:
            chart.add_trace(go.Scatter(x=display_primary.index, y=display_primary["Broad Dollar"], name="Broad Dollar", mode="lines", line=dict(color=BLUE, width=2.0)), row=2, col=1, secondary_y=False)
        if "10-Year Real Yield" in display_primary:
            chart.add_trace(go.Scatter(x=display_primary.index, y=display_primary["10-Year Real Yield"], name="10-Year Real Yield", mode="lines", line=dict(color=PURPLE, width=2.0)), row=2, col=1, secondary_y=True)
        plot_layout(chart, 620, dict(l=54, r=58, t=88, b=48))
        chart.update_yaxes(title_text="Credit OAS", row=1, col=1)
        chart.update_yaxes(title_text="Broad dollar", row=2, col=1, secondary_y=False)
        chart.update_yaxes(title_text="Real yield", row=2, col=1, secondary_y=True)
        st.plotly_chart(chart, width="stretch")

primary_card = scorecard(primary, primary_levels, primary_impulses, primary_specs)
market_card = scorecard(market, market_levels, market_impulses, market_specs) if not market.empty else pd.DataFrame()

if show_scorecards:
    render_section_header(
        "Primary-Source Component Scorecard",
        "Positive scores indicate easier conditions after direction adjustment. Level and impulse remain separate.",
    )
    if not primary_card.empty:
        styled = primary_card.style.map(color_score, subset=["Level Score", "Impulse Score"]).format({"Level Score": "{:+.2f}", "Impulse Score": "{:+.2f}", "Within-Sleeve Weight": "{:.0%}"}, na_rep="N/A")
        st.dataframe(styled, width="stretch", hide_index=True, column_config={"Description": st.column_config.TextColumn(width="large"), "Source": st.column_config.TextColumn(width="medium")})
    render_section_header(
        "Market Confirmation Scorecard",
        "This sleeve is capped at 15% and excludes the duplicated credit, dollar, duration, and AI-leadership trades in the old formula.",
    )
    if market_card.empty:
        st.info("Market confirmation is unavailable from the current Yahoo Finance download.")
    else:
        columns = ["Component", "Latest", "Impulse Score", "Signal", "Within-Sleeve Weight", "Description"]
        styled = market_card[columns].style.map(color_score, subset=["Impulse Score"]).format({"Impulse Score": "{:+.2f}", "Within-Sleeve Weight": "{:.0%}"}, na_rep="N/A")
        st.dataframe(styled, width="stretch", hide_index=True, column_config={"Description": st.column_config.TextColumn(width="large")})

if show_fcig_overlay:
    render_section_header(
        "Federal Reserve FCI-G Overlay",
        "FCI-G measures the estimated growth headwind or tailwind from financial conditions. It remains an external macro overlay.",
    )
    fcig, fcig_errors = load_fcig()
    if fcig.empty:
        st.warning("Federal Reserve FCI-G data could not be loaded.")
    else:
        fcig = filter_lookback(fcig, lookback)
        monthly_level = filter_lookback(liquidity_level.resample("ME").last(), lookback)
        monthly_impulse = filter_lookback(liquidity_impulse.resample("ME").last(), lookback)
        chart = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10)
        for index, column in enumerate(fcig):
            chart.add_trace(go.Scatter(x=fcig.index, y=fcig[column], name=column, mode="lines", line=dict(color=(RED, ORANGE)[index % 2], width=2.0)), row=1, col=1)
        chart.add_hline(y=0, line_width=1, line_dash="dot", line_color=GRAY, row=1, col=1)
        chart.add_trace(go.Scatter(x=monthly_level.index, y=monthly_level, name="Liquidity Level", mode="lines", line=dict(color=BLUE, width=2.2)), row=2, col=1)
        chart.add_trace(go.Scatter(x=monthly_impulse.index, y=monthly_impulse, name="Liquidity Impulse", mode="lines", line=dict(color=GREEN, width=2.2)), row=2, col=1)
        chart.add_hline(y=0, line_width=1, line_dash="dot", line_color=GRAY, row=2, col=1)
        plot_layout(chart, 650)
        chart.update_yaxes(title_text="FCI-G", row=1, col=1)
        chart.update_yaxes(title_text="Liquidity z-score", row=2, col=1)
        st.plotly_chart(chart, width="stretch")

if show_download:
    render_section_header("Download Underlying Data", "The export preserves numeric values for independent audit and backtesting.")
    export = pd.concat(
        [
            liquidity_level.rename("Liquidity Level"),
            liquidity_impulse.rename("Liquidity Impulse"),
            market_confirmation.rename("Market Confirmation"),
            easing_breadth.rename("Weighted Easing Breadth"),
            impulse_coverage.rename("Weighted Coverage"),
            sleeve_levels.add_prefix("Level | "),
            sleeve_impulses.add_prefix("Impulse | "),
            primary.add_prefix("Raw | "),
        ],
        axis=1,
    )
    export.index.name = "Date"
    st.download_button(
        "Download liquidity history",
        data=export.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="adfm_liquidity_conditions_history.csv",
        mime="text/csv",
    )

render_footer(
    data_note=(
        "Primary inputs: Federal Reserve H.4.1, New York Fed overnight rates and reverse repo, "
        "ICE BofA option-adjusted spreads via FRED, the Federal Reserve broad dollar index, "
        "10-year real yields, Federal Reserve FCI-G, and Yahoo Finance market confirmation proxies."
    )
)
