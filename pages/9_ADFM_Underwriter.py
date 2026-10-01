from __future__ import annotations

from html import escape
from typing import Any, Mapping, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from adfm_core.market_data import configure_yfinance_cache, fetch_daily_ohlcv
from adfm_core.palette import PASTEL
from adfm_core.sec_fundamentals import (
    SecClient,
    SecDataError,
    ValuationSnapshot,
    annual_cagr,
    balance_sheet_table,
    build_valuation_snapshot,
    extract_metrics,
    financial_table,
    latest_quarter_growth,
    maturity_table,
    period_label,
    recent_filings,
    resolve_company,
    source_audit_table,
)
from adfm_core.ui import (
    PageHeader,
    dataframe_download,
    inject_explorer_style,
    metric_table,
    render_footer,
    render_page_header,
    render_section_header,
    render_sidebar_about,
    render_selection_note,
)

TITLE = "Equity Underwriter"
DESCRIPTION = (
    "Filing-driven company fundamentals, current valuation, capital structure, "
    "issuer-credit ratios, debt maturities, market context, and recent SEC events."
)

st.set_page_config(
    layout="wide",
    page_title=TITLE,
    initial_sidebar_state="expanded",
)
inject_explorer_style(max_width_px=1580)
configure_yfinance_cache()

st.markdown(
    """
    <style>
    .uw-issuer {
        display:flex;
        justify-content:space-between;
        align-items:flex-end;
        gap:1.25rem;
        border-top:3px solid #111;
        border-bottom:1px solid #b9b9b9;
        padding:.78rem .05rem .72rem;
        margin:.05rem 0 .75rem;
    }
    .uw-symbol-line {
        display:flex;
        align-items:baseline;
        gap:.6rem;
        min-width:0;
    }
    .uw-symbol {
        font-family:Arial,Helvetica,sans-serif;
        font-size:1.55rem;
        line-height:1;
        font-weight:800;
        color:#111;
        letter-spacing:.015em;
    }
    .uw-company {
        font-family:Arial,Helvetica,sans-serif;
        font-size:1rem;
        line-height:1.15;
        font-weight:700;
        color:#2f5597;
        overflow:hidden;
        text-overflow:ellipsis;
        white-space:nowrap;
    }
    .uw-meta {
        margin-top:.38rem;
        color:#666;
        font-family:Arial,Helvetica,sans-serif;
        font-size:.68rem;
        line-height:1.35;
    }
    .uw-quote {
        text-align:right;
        flex:0 0 auto;
    }
    .uw-price {
        font-family:Arial,Helvetica,sans-serif;
        font-size:1.75rem;
        line-height:1;
        font-weight:800;
        color:#111;
        font-variant-numeric:tabular-nums;
    }
    .uw-price-date {
        margin-top:.3rem;
        color:#666;
        font-family:Arial,Helvetica,sans-serif;
        font-size:.66rem;
    }
    .uw-day-change {
        margin-top:.22rem;
        font-family:Arial,Helvetica,sans-serif;
        font-size:.72rem;
        font-weight:800;
        font-variant-numeric:tabular-nums;
    }
    .uw-day-change-positive { color:#237a3b; }
    .uw-day-change-negative { color:#b13030; }
    .uw-day-change-neutral { color:#666; }
    .uw-overview {
        display:grid;
        grid-template-columns:repeat(5,minmax(0,1fr));
        border-top:1px solid #c8c8c8;
        border-left:1px solid #d7d7d7;
        margin:.25rem 0 .55rem;
        background:#fff;
    }
    .uw-overview-cell {
        min-width:0;
        display:grid;
        grid-template-columns:minmax(0,1fr) auto;
        align-items:baseline;
        gap:.45rem;
        padding:.42rem .52rem;
        border-right:1px solid #d7d7d7;
        border-bottom:1px solid #e0e0e0;
        font-family:Arial,Helvetica,sans-serif;
        font-size:.69rem;
        line-height:1.2;
        cursor:help;
    }
    .uw-overview-label {
        color:#6b7280;
        white-space:nowrap;
        overflow:hidden;
        text-overflow:ellipsis;
    }
    .uw-overview-value {
        color:#171717;
        font-weight:800;
        text-align:right;
        white-space:nowrap;
        font-variant-numeric:tabular-nums;
    }
    .uw-tone-positive .uw-overview-value { color:#237a3b; }
    .uw-tone-caution .uw-overview-value { color:#9a6700; }
    .uw-tone-negative .uw-overview-value { color:#b13030; }
    .uw-tone-neutral .uw-overview-value { color:#171717; }
    .uw-overview-value.uw-unavailable { color:#8b8b8b; font-weight:600; }
    .uw-overview-note {
        color:#6b6b6b;
        font-family:Arial,Helvetica,sans-serif;
        font-size:.64rem;
        line-height:1.35;
        margin:.1rem 0 .85rem;
    }
    .uw-legend {
        display:flex;
        flex-wrap:wrap;
        gap:.55rem 1rem;
        margin:.1rem 0 .7rem;
        color:#666;
        font-family:Arial,Helvetica,sans-serif;
        font-size:.66rem;
    }
    .uw-legend-item { display:inline-flex; align-items:center; gap:.3rem; }
    .uw-legend-dot { width:.5rem; height:.5rem; border-radius:50%; }
    div[data-testid="stPlotlyChart"] { margin-top:-.1rem; }
    @media (max-width: 1250px) {
        .uw-overview { grid-template-columns:repeat(4,minmax(0,1fr)); }
    }
    @media (max-width: 950px) {
        .uw-overview { grid-template-columns:repeat(3,minmax(0,1fr)); }
        .uw-issuer { align-items:flex-start; }
    }
    @media (max-width: 700px) {
        .uw-overview { grid-template-columns:repeat(2,minmax(0,1fr)); }
        .uw-issuer { display:block; }
        .uw-quote { text-align:left; margin-top:.65rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=86_400, show_spinner=False)
def load_ticker_map() -> Mapping[str, Any]:
    return SecClient().company_tickers()


@st.cache_data(ttl=900, show_spinner=False)
def load_company_facts(cik: int) -> Mapping[str, Any]:
    return SecClient().company_facts(cik)


@st.cache_data(ttl=900, show_spinner=False)
def load_submissions(cik: int) -> Mapping[str, Any]:
    return SecClient().submissions(cik)


def market_history(
    ticker: str,
) -> tuple[pd.Series, Optional[float], Optional[pd.Timestamp]]:
    frames, _ = fetch_daily_ohlcv((ticker,), period="2y")
    frame = frames.get(ticker)
    if frame is None or frame.empty or "Close" not in frame:
        return pd.Series(dtype="float64"), None, None
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if close.empty:
        return close, None, None
    return close, float(close.iloc[-1]), pd.Timestamp(close.index[-1]).normalize()


CURRENCY_SYMBOLS: Mapping[str, str] = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CNY": "¥",
    "HKD": "HK$",
    "CAD": "C$",
    "AUD": "A$",
    "CHF": "CHF ",
    "INR": "₹",
    "KRW": "₩",
}


def currency_prefix(currency: str) -> str:
    code = str(currency or "").upper()
    return CURRENCY_SYMBOLS.get(code, f"{code} " if code else "")


def _signed_currency(value: float, currency: str, decimals: int = 0) -> str:
    sign = "-" if float(value) < 0 else ""
    return f"{sign}{currency_prefix(currency)}{abs(float(value)):,.{decimals}f}"


def format_money(value: Optional[float], *, currency: str = "USD") -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    magnitude = abs(float(value))
    if magnitude >= 1_000_000_000:
        scaled, suffix = value / 1_000_000_000, "B"
    elif magnitude >= 1_000_000:
        scaled, suffix = value / 1_000_000, "M"
    else:
        scaled, suffix = value, ""
    return f"{_signed_currency(scaled, currency, 2)}{suffix}"


def format_multiple(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{value:,.2f}x"


def format_percent(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{value * 100:,.1f}%"


def statement_currency(metrics: Mapping[str, Any]) -> str:
    for key in ("revenue", "cash", "debt_total", "equity"):
        metric = metrics.get(key)
        if metric is not None and metric.unit:
            return str(metric.unit)
    return "USD"


def scale_financial_table(frame: pd.DataFrame, currency: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in out.columns:
        if column != "Period End":
            numeric = pd.to_numeric(out[column], errors="coerce") / 1_000_000
            out[column] = numeric.map(
                lambda value: (
                    _signed_currency(value, currency)
                    if pd.notna(value)
                    else "Unavailable"
                )
            )
    out["Period End"] = pd.to_datetime(out["Period End"]).dt.date
    return out


def first_recent_value(submissions: Mapping[str, Any], field: str) -> str:
    recent = submissions.get("filings", {}).get("recent", {})
    values = recent.get(field, []) if isinstance(recent, Mapping) else []
    if not isinstance(values, list) or not values:
        return "Unavailable"
    return str(values[0] or "Unavailable")


def quarterly_chart(frame: pd.DataFrame, unit: str) -> go.Figure:
    indexed = frame.set_index("Period End").copy()
    revenue = pd.to_numeric(indexed.get("Revenue"), errors="coerce")
    operating_income = pd.to_numeric(indexed.get("Operating Income"), errors="coerce")
    margin = operating_income.div(revenue.replace(0, pd.NA)) * 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=indexed.index,
            y=revenue / 1_000_000_000,
            name=f"Revenue ({currency_prefix(unit)}bn)",
            marker_color=PASTEL["blue"],
            hovertemplate=(
                f"%{{x|%Y-%m-%d}}<br>Revenue: {currency_prefix(unit)}%{{y:,.2f}}bn<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    if margin.notna().any():
        fig.add_trace(
            go.Scatter(
                x=indexed.index,
                y=margin,
                name="Operating Margin",
                line={"color": PASTEL["coral"], "width": 2.4},
                marker={"size": 6},
                hovertemplate="%{x|%Y-%m-%d}<br>Margin: %{y:,.1f}%<extra></extra>",
            ),
            secondary_y=True,
        )
    fig.update_yaxes(
        title_text=f"Revenue ({currency_prefix(unit)}bn)",
        tickprefix=currency_prefix(unit),
        secondary_y=False,
        gridcolor="#e5e5e5",
    )
    fig.update_yaxes(
        title_text="Operating margin", ticksuffix="%", secondary_y=True, showgrid=False
    )
    fig.update_layout(
        height=410,
        margin={"l": 25, "r": 25, "t": 30, "b": 25},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#171717", "family": "Arial"},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hovermode="x unified",
        bargap=0.28,
    )
    return fig


def price_history_chart(close: pd.Series, ticker: str, currency: str) -> go.Figure:
    clean = pd.to_numeric(close, errors="coerce").dropna()
    visible_start = pd.Timestamp(clean.index[-1]) - pd.DateOffset(years=1)
    visible = clean.loc[pd.to_datetime(clean.index) >= visible_start]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=visible.index,
            y=visible,
            name=ticker,
            mode="lines",
            line={"color": PASTEL["blue"], "width": 2.5},
            hovertemplate=(
                f"%{{x|%Y-%m-%d}}<br>{currency_prefix(currency)}%{{y:,.2f}}<extra></extra>"
            ),
        )
    )
    for window, color in ((20, PASTEL["lavender"]), (50, PASTEL["coral"]), (200, PASTEL["sage"])):
        average = (
            clean.rolling(window, min_periods=window).mean().reindex(visible.index)
        )
        if average.notna().any():
            fig.add_trace(
                go.Scatter(
                    x=average.index,
                    y=average,
                    name=f"{window}D average",
                    mode="lines",
                    line={"color": color, "width": 1.4},
                    hovertemplate=(
                        f"%{{x|%Y-%m-%d}}<br>{currency_prefix(currency)}%{{y:,.2f}}<extra></extra>"
                    ),
                )
            )
    fig.update_layout(
        height=330,
        margin={"l": 25, "r": 25, "t": 18, "b": 25},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#171717", "family": "Arial"},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(
        tickprefix=currency_prefix(currency),
        tickformat=",.0f",
        gridcolor="#e5e5e5",
        title_text="Price",
    )
    return fig


def _lower_is_better(
    value: Optional[float], favorable: float, caution: float
) -> tuple[str, str]:
    if value is None or pd.isna(value):
        return "neutral", "Unavailable"
    if value < 0:
        return "negative", "Negative denominator"
    if value <= favorable:
        return "positive", "Favorable band"
    if value <= caution:
        return "caution", "Watch band"
    return "negative", "Adverse band"


def _higher_is_better(
    value: Optional[float], favorable: float, caution: float
) -> tuple[str, str]:
    if value is None or pd.isna(value):
        return "neutral", "Unavailable"
    if value >= favorable:
        return "positive", "Favorable band"
    if value >= caution:
        return "caution", "Watch band"
    return "negative", "Adverse band"


def _context_only(value: Optional[float]) -> tuple[str, str]:
    if value is None or pd.isna(value):
        return "neutral", "Unavailable"
    return "neutral", "Context only"


def _card(
    section: str,
    metric: str,
    display: str,
    formula: str,
    assessment: tuple[str, str],
) -> dict[str, str]:
    tone, context = assessment
    return {
        "Section": section,
        "Metric": metric,
        "Value": display,
        "Formula": formula,
        "Tone": tone,
        "Context": context,
    }


def render_underwriter_legend() -> None:
    st.markdown(
        """
        <div class="uw-legend">
          <span class="uw-legend-item"><span class="uw-legend-dot" style="background:#237a3b"></span>Favorable / improving</span>
          <span class="uw-legend-item"><span class="uw-legend-dot" style="background:#9a6700"></span>Watch</span>
          <span class="uw-legend-item"><span class="uw-legend-dot" style="background:#b13030"></span>Adverse / deteriorating</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_underwriter_cards(rows: list[dict[str, str]]) -> None:
    """Render a dense Finviz-style metric matrix; formulas remain available on hover."""
    cells: list[str] = []
    for row in rows:
        metric = escape(str(row.get("Metric", "")))
        value = escape(str(row.get("Value", "Unavailable")))
        section = str(row.get("Section", ""))
        formula = str(row.get("Formula", ""))
        context = str(row.get("Context", ""))
        tone = str(row.get("Tone", "neutral"))
        title = escape(
            " · ".join(part for part in (section, formula, context) if part),
            quote=True,
        )
        unavailable = value.lower() == "unavailable"
        value_class = "uw-overview-value uw-unavailable" if unavailable else "uw-overview-value"
        cells.append(
            f'<div class="uw-overview-cell uw-tone-{escape(tone)}" title="{title}">'
            f'<span class="uw-overview-label">{metric}</span>'
            f'<span class="{value_class}">{value}</span>'
            "</div>"
        )
    st.markdown(
        '<div class="uw-overview">' + "".join(cells) + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="uw-overview-note">Hover any metric for the SEC-derived formula and underwriting context. '
        "Green/red on growth and price metrics indicates direction; valuation, profitability, liquidity and leverage use the existing transparent underwriting bands.</div>",
        unsafe_allow_html=True,
    )


def render_underwriter_audit(rows: list[dict[str, str]]) -> None:
    frame = pd.DataFrame(
        rows,
        columns=["Section", "Metric", "Value", "Formula", "Context", "Tone"],
    )
    colors = {
        "positive": "#237a3b",
        "caution": "#9a6700",
        "negative": "#b13030",
        "neutral": "#6b6b6b",
    }
    tones = frame.pop("Tone")
    styled = frame.style.apply(
        lambda column: [
            f"color: {colors.get(tone, '#6b6b6b')}" for tone in tones
        ],
        subset=["Value", "Context"],
    )
    st.dataframe(styled, hide_index=True, width="stretch", height="auto")


def _directional_assessment(value: Optional[float]) -> tuple[str, str]:
    if value is None or pd.isna(value):
        return "neutral", "Unavailable"
    if value > 0:
        return "positive", "Improving / positive"
    if value < 0:
        return "negative", "Deteriorating / negative"
    return "neutral", "Flat"


def _trailing_price_change(close: pd.Series, periods: int) -> Optional[float]:
    clean = pd.to_numeric(close, errors="coerce").dropna()
    if len(clean) <= periods:
        return None
    base = float(clean.iloc[-1 - periods])
    return float(clean.iloc[-1] / base - 1.0) if base else None


def _ytd_price_change(close: pd.Series) -> Optional[float]:
    clean = pd.to_numeric(close, errors="coerce").dropna()
    if clean.empty:
        return None
    dated = clean.copy()
    dated.index = pd.to_datetime(dated.index)
    latest_year = int(dated.index[-1].year)
    year = dated[dated.index.year == latest_year]
    if len(year) < 2 or float(year.iloc[0]) == 0:
        return None
    return float(year.iloc[-1] / year.iloc[0] - 1.0)


def price_context_cards(close: pd.Series, *, currency: str = "USD") -> list[dict[str, str]]:
    clean = pd.to_numeric(close, errors="coerce").dropna()
    if clean.empty:
        return []
    last = float(clean.iloc[-1])
    rows = [
        _card("Market", "Price", _signed_currency(last, currency, 2), "Latest completed-session close", _context_only(last)),
    ]
    for label, periods in (
        ("Perf 1W", 5),
        ("Perf 1M", 21),
        ("Perf 3M", 63),
        ("Perf 6M", 126),
        ("Perf 1Y", 252),
    ):
        value = _trailing_price_change(clean, periods)
        rows.append(_card("Market", label, format_percent(value), f"Price change over {periods} trading sessions", _directional_assessment(value)))
    ytd = _ytd_price_change(clean)
    rows.append(_card("Market", "Perf YTD", format_percent(ytd), "Price change from first completed session of the calendar year", _directional_assessment(ytd)))

    year = clean.tail(252)
    high = float(year.max()) if not year.empty else None
    low = float(year.min()) if not year.empty else None
    from_high = last / high - 1.0 if high else None
    from_low = last / low - 1.0 if low else None
    rows.extend([
        _card("Market", "From 52W High", format_percent(from_high), "Latest close ÷ trailing 252-session high − 1", _context_only(from_high)),
        _card("Market", "From 52W Low", format_percent(from_low), "Latest close ÷ trailing 252-session low − 1", _context_only(from_low)),
    ])
    for window in (20, 50, 200):
        average = clean.rolling(window, min_periods=window).mean().iloc[-1]
        distance = last / float(average) - 1.0 if pd.notna(average) and float(average) != 0 else None
        rows.append(
            _card(
                "Market",
                f"SMA{window}",
                format_percent(distance),
                f"Latest close ÷ {window}-session simple moving average − 1",
                _directional_assessment(distance),
            )
        )
    return rows


def growth_cards(metrics: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label, key in (("Sales", "revenue"), ("EPS", "eps_diluted")):
        metric = metrics.get(key)
        values = (
            (f"{label} Q/Q", _quarter_change(metric, 1), "Latest reported quarter versus immediately prior quarter"),
            (f"{label} Y/Y", _quarter_change(metric, 4), "Latest reported quarter versus year-ago quarter"),
            (f"{label} 3Y CAGR", annual_cagr(metric, 3), "Compound annual growth across the latest three fiscal years"),
            (f"{label} 5Y CAGR", annual_cagr(metric, 5), "Compound annual growth across the latest five fiscal years"),
        )
        for metric_label, value, formula in values:
            rows.append(
                _card(
                    "Growth",
                    metric_label,
                    format_percent(value),
                    formula,
                    _directional_assessment(value),
                )
            )
    return rows


def _net_leverage_assessment(value: Optional[float]) -> tuple[str, str]:
    if value is None or pd.isna(value):
        return "neutral", "Unavailable"
    if value < 0:
        return "positive", "Net cash"
    if value <= 1.5:
        return "positive", "Favorable band"
    if value <= 3.5:
        return "caution", "Watch band"
    return "negative", "Adverse band"


def credit_overview_cards(
    snapshot: ValuationSnapshot, *, currency: str = "USD"
) -> list[dict[str, str]]:
    return [
        _card(
            "Credit",
            "Debt / EBITDA",
            format_multiple(snapshot.debt_ebitda),
            "Funded debt ÷ LTM calculated EBITDA",
            _lower_is_better(snapshot.debt_ebitda, 2.0, 4.0),
        ),
        _card(
            "Credit",
            "Net Debt / EBITDA",
            format_multiple(snapshot.net_debt_ebitda),
            "(Funded debt − cash and short-term investments) ÷ LTM calculated EBITDA",
            _net_leverage_assessment(snapshot.net_debt_ebitda),
        ),
        _card(
            "Credit",
            "Interest Coverage",
            format_multiple(snapshot.interest_coverage),
            "LTM calculated EBITDA ÷ LTM reported interest expense",
            _higher_is_better(snapshot.interest_coverage, 6.0, 3.0),
        ),
    ]


def overview_cards(
    metrics: Mapping[str, Any],
    snapshot: ValuationSnapshot,
    close: pd.Series,
    *,
    currency: str = "USD",
) -> list[dict[str, str]]:
    combined = (
        price_context_cards(close, currency="USD")
        + valuation_cards(snapshot, currency=currency)
        + growth_cards(metrics)
        + sec_snapshot_cards(snapshot, currency=currency)
        + credit_overview_cards(snapshot, currency=currency)
    )
    ordered_metrics = [
        "Price", "Market Capitalization", "Enterprise Value", "Shares Outstanding", "LTM Diluted EPS",
        "Sales / Share", "Book / Share", "Cash / Share", "P / E", "P / Sales", "P / Book", "P / FCF",
        "EV / Revenue", "EV / EBITDA", "FCF Yield", "Sales Q/Q", "Sales Y/Y", "Sales 3Y CAGR", "Sales 5Y CAGR",
        "EPS Q/Q", "EPS Y/Y", "EPS 3Y CAGR", "EPS 5Y CAGR", "Gross Margin", "Operating Margin", "Profit Margin",
        "FCF Margin", "ROA", "ROE", "ROIC", "Current Ratio", "Quick Ratio", "Debt / Equity", "Debt / EBITDA",
        "Net Debt / EBITDA", "Interest Coverage", "Dividend Yield", "Payout Ratio", "Perf 1W", "Perf 1M",
        "Perf 3M", "Perf 6M", "Perf YTD", "Perf 1Y", "From 52W High", "From 52W Low", "SMA20", "SMA50", "SMA200",
    ]
    first_by_metric: dict[str, dict[str, str]] = {}
    for row in combined:
        first_by_metric.setdefault(str(row["Metric"]), row)
    rows = [first_by_metric[name] for name in ordered_metrics if name in first_by_metric]
    rows.extend(
        row for name, row in first_by_metric.items() if name not in set(ordered_metrics)
    )
    return rows


def render_issuer_masthead(
    *,
    ticker: str,
    name: str,
    price: Optional[float],
    price_date: Optional[pd.Timestamp],
    sic_description: str,
    fiscal_year_end: str,
    latest_form: str,
    latest_filed: str,
    price_currency: str,
    filing_currency: str,
    close_history: pd.Series,
) -> None:
    price_text = (
        _signed_currency(float(price), price_currency, 2)
        if price is not None and pd.notna(price)
        else "Unavailable"
    )
    date_text = period_label(price_date)
    clean_close = pd.to_numeric(close_history, errors="coerce").dropna()
    day_change_text = ""
    day_change_class = "uw-day-change-neutral"
    if len(clean_close) >= 2:
        prior = float(clean_close.iloc[-2])
        current = float(clean_close.iloc[-1])
        day_change = current - prior
        day_change_pct = day_change / prior if prior else 0.0
        day_change_text = f"{day_change:+.2f} ({day_change_pct:+.2%})"
        day_change_class = (
            "uw-day-change-positive"
            if day_change > 0
            else "uw-day-change-negative"
            if day_change < 0
            else "uw-day-change-neutral"
        )
    st.markdown(
        f"""
        <div class="uw-issuer">
          <div>
            <div class="uw-symbol-line">
              <span class="uw-symbol">{escape(ticker)}</span>
              <span class="uw-company">{escape(name)}</span>
            </div>
            <div class="uw-meta">{escape(sic_description)} · FY end {escape(fiscal_year_end)} · Filing currency {escape(filing_currency)} · Latest filing {escape(latest_form)} on {escape(latest_filed)}</div>
          </div>
          <div class="uw-quote">
            <div class="uw-price">{escape(price_text)}</div>
            <div class="uw-day-change {day_change_class}">{escape(day_change_text)}</div>
            <div class="uw-price-date">Completed-session close · {escape(date_text)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _annual_series(metrics: Mapping[str, Any], key: str, *, instant: bool = False) -> pd.Series:
    metric = metrics.get(key)
    if metric is None:
        return pd.Series(dtype="float64")
    observations = metric.instant if instant else metric.annual
    if not observations:
        return pd.Series(dtype="float64")
    series = pd.Series(
        {pd.Timestamp(item.end): float(item.value) for item in observations},
        dtype="float64",
    ).sort_index()
    if instant and not series.empty:
        frame = series.to_frame("value")
        frame["year"] = frame.index.year
        series = frame.groupby("year")["value"].last()
        series.index = pd.to_datetime([f"{int(year)}-12-31" for year in series.index])
    return series.tail(8)


def _mini_bar_chart(
    series: pd.Series,
    *,
    title: str,
    color: str,
    scale: float = 1.0,
    suffix: str = "",
    decimals: int = 2,
) -> go.Figure:
    values = pd.to_numeric(series, errors="coerce").dropna() / scale
    labels = [str(pd.Timestamp(index).year) for index in values.index]
    text_values = [f"{value:,.{decimals}f}{suffix}" for value in values.values]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values.values,
            marker_color=color,
            text=text_values,
            textposition="inside",
            insidetextanchor="end",
            hovertemplate=f"%{{x}}<br>{title}: %{{y:,.{decimals}f}}{suffix}<extra></extra>",
        )
    )
    fig.update_layout(
        height=245,
        margin={"l": 12, "r": 12, "t": 38, "b": 22},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#4b5563", "family": "Arial", "size": 10},
        title={"text": title, "x": 0.01, "xanchor": "left", "font": {"size": 12, "color": "#4b5563"}},
        showlegend=False,
        bargap=0.12,
    )
    fig.update_xaxes(showgrid=False, fixedrange=True)
    fig.update_yaxes(gridcolor="#ececec", zeroline=False, fixedrange=True)
    return fig


def render_annual_snapshot(metrics: Mapping[str, Any], currency: str) -> None:
    eps = _annual_series(metrics, "eps_diluted")
    revenue = _annual_series(metrics, "revenue")
    shares = _annual_series(metrics, "shares_outstanding", instant=True)
    if eps.empty and revenue.empty and shares.empty:
        return
    render_section_header(
        "Annual trajectory",
        "Reported fiscal-year history. Shares use the last reported point-in-time balance for each year.",
    )
    columns = st.columns(3)
    series_specs = (
        (columns[0], eps, "GAAP EPS", PASTEL["blue"], 1.0, "", 2),
        (columns[1], revenue, f"Sales ({currency_prefix(currency)}bn)", PASTEL["lavender"], 1_000_000_000, "", 1),
        (columns[2], shares, "Shares outstanding (bn)", PASTEL["teal"], 1_000_000_000, "", 2),
    )
    for column, series, title, color, scale, suffix, decimals in series_specs:
        with column:
            if series.empty:
                st.caption(f"{title}: unavailable")
            else:
                st.plotly_chart(
                    _mini_bar_chart(
                        series,
                        title=title,
                        color=color,
                        scale=scale,
                        suffix=suffix,
                        decimals=decimals,
                    ),
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True},
                )


def _latest_quarter_margin(metrics: Mapping[str, Any]) -> Optional[float]:
    revenue = metrics.get("revenue")
    operating_income = metrics.get("operating_income")
    if revenue is None or operating_income is None:
        return None
    revenue_by_end = {item.end: item.value for item in revenue.quarterly}
    operating_by_end = {item.end: item.value for item in operating_income.quarterly}
    common_dates = sorted(set(revenue_by_end) & set(operating_by_end))
    if not common_dates:
        return None
    end = common_dates[-1]
    denominator = revenue_by_end[end]
    return operating_by_end[end] / denominator if denominator else None


def underwrite_read(
    metrics: Mapping[str, Any], valuation: ValuationSnapshot, *, currency: str = "USD"
) -> list[tuple[str, str]]:
    revenue_growth = latest_quarter_growth(metrics.get("revenue"))
    margin = _latest_quarter_margin(metrics)
    reads: list[tuple[str, str]] = []

    if revenue_growth is not None:
        direction = "expanded" if revenue_growth >= 0 else "contracted"
        reads.append(
            (
                "Top line",
                f"Latest reported quarterly revenue {direction} {abs(revenue_growth) * 100:.1f}% year over year.",
            )
        )
    if margin is not None:
        reads.append(
            (
                "Earnings power",
                f"Latest reported-quarter operating margin was {margin * 100:.1f}%.",
            )
        )
    if valuation.ltm_fcf is not None:
        cash_text = (
            f"LTM free cash flow was {format_money(valuation.ltm_fcf, currency=currency)} with a "
            f"{format_percent(valuation.fcf_margin)} conversion margin."
        )
        reads.append(("Cash conversion", cash_text))
    if valuation.net_debt_ebitda is not None:
        balance = "net cash" if valuation.net_debt_ebitda < 0 else "net debt"
        reads.append(
            (
                "Balance sheet",
                f"The issuer carries {balance} equal to {abs(valuation.net_debt_ebitda):.2f}x LTM calculated EBITDA.",
            )
        )
    if valuation.interest_coverage is not None:
        reads.append(
            (
                "Debt service",
                f"Calculated EBITDA covers LTM reported interest expense {valuation.interest_coverage:.1f}x.",
            )
        )
    return reads


def valuation_cards(
    snapshot: ValuationSnapshot, *, currency: str = "USD"
) -> list[dict[str, str]]:
    return [
        _card(
            "Scale",
            "Market Capitalization",
            format_money(snapshot.market_cap, currency=currency),
            "Latest completed-session close × latest SEC shares outstanding",
            _context_only(snapshot.market_cap),
        ),
        _card(
            "Scale",
            "Enterprise Value",
            format_money(snapshot.enterprise_value, currency=currency),
            "Market cap + funded debt + preferred + minority interest − cash and short-term investments",
            _context_only(snapshot.enterprise_value),
        ),
        _card(
            "Valuation",
            "P / E",
            format_multiple(snapshot.pe),
            "Market capitalization ÷ LTM net income available to common",
            _lower_is_better(snapshot.pe, 20.0, 35.0),
        ),
        _card(
            "Valuation",
            "P / Sales",
            format_multiple(snapshot.price_sales),
            "Market capitalization ÷ LTM revenue",
            _lower_is_better(snapshot.price_sales, 3.0, 8.0),
        ),
        _card(
            "Valuation",
            "P / Book",
            format_multiple(snapshot.price_book),
            "Market capitalization ÷ latest SEC stockholders' equity",
            _lower_is_better(snapshot.price_book, 3.0, 8.0),
        ),
        _card(
            "Valuation",
            "P / Cash",
            format_multiple(snapshot.price_cash),
            "Market capitalization ÷ cash and short-term investments",
            _lower_is_better(snapshot.price_cash, 10.0, 25.0),
        ),
        _card(
            "Valuation",
            "P / FCF",
            format_multiple(snapshot.price_fcf),
            "Market capitalization ÷ LTM free cash flow",
            _lower_is_better(snapshot.price_fcf, 25.0, 50.0),
        ),
        _card(
            "Valuation",
            "EV / Revenue",
            format_multiple(snapshot.ev_revenue),
            "Enterprise value ÷ LTM revenue",
            _lower_is_better(snapshot.ev_revenue, 4.0, 10.0),
        ),
        _card(
            "Valuation",
            "EV / EBITDA",
            format_multiple(snapshot.ev_ebitda),
            "Enterprise value ÷ (LTM operating income + LTM D&A)",
            _lower_is_better(snapshot.ev_ebitda, 15.0, 25.0),
        ),
        _card(
            "Cash Yield",
            "FCF Yield",
            format_percent(snapshot.fcf_yield),
            "(LTM operating cash flow − LTM capex) ÷ market capitalization",
            _higher_is_better(snapshot.fcf_yield, 0.05, 0.02),
        ),
        _card(
            "Margins",
            "Operating Margin",
            format_percent(snapshot.operating_margin),
            "LTM operating income ÷ LTM revenue",
            _higher_is_better(snapshot.operating_margin, 0.20, 0.05),
        ),
        _card(
            "Margins",
            "FCF Margin",
            format_percent(snapshot.fcf_margin),
            "LTM free cash flow ÷ LTM revenue",
            _higher_is_better(snapshot.fcf_margin, 0.15, 0.05),
        ),
    ]


def valuation_table(
    snapshot: ValuationSnapshot, *, currency: str = "USD"
) -> pd.DataFrame:
    return pd.DataFrame(valuation_cards(snapshot, currency=currency))[
        ["Metric", "Value", "Formula", "Context"]
    ]


def sec_snapshot_cards(
    snapshot: ValuationSnapshot, *, currency: str = "USD"
) -> list[dict[str, str]]:
    return [
        _card(
            "Per Share",
            "LTM Diluted EPS",
            _signed_currency(snapshot.eps, currency, 2)
            if snapshot.eps is not None
            else "Unavailable",
            "LTM reported diluted EPS; net income ÷ diluted shares if unavailable",
            _context_only(snapshot.eps),
        ),
        _card(
            "Per Share",
            "Sales / Share",
            _signed_currency(snapshot.sales_per_share, currency, 2)
            if snapshot.sales_per_share is not None
            else "Unavailable",
            "LTM revenue ÷ diluted weighted-average shares",
            _context_only(snapshot.sales_per_share),
        ),
        _card(
            "Per Share",
            "Book / Share",
            _signed_currency(snapshot.book_per_share, currency, 2)
            if snapshot.book_per_share is not None
            else "Unavailable",
            "Latest equity ÷ shares outstanding",
            _context_only(snapshot.book_per_share),
        ),
        _card(
            "Per Share",
            "Cash / Share",
            _signed_currency(snapshot.cash_per_share, currency, 2)
            if snapshot.cash_per_share is not None
            else "Unavailable",
            "Cash and short-term investments ÷ shares outstanding",
            _context_only(snapshot.cash_per_share),
        ),
        _card(
            "Per Share",
            "Shares Outstanding",
            f"{snapshot.shares:,.0f}" if snapshot.shares is not None else "Unavailable",
            "Latest SEC shares outstanding",
            _context_only(snapshot.shares),
        ),
        _card(
            "Margins",
            "Gross Margin",
            format_percent(snapshot.gross_margin),
            "LTM gross profit ÷ LTM revenue",
            _higher_is_better(snapshot.gross_margin, 0.40, 0.20),
        ),
        _card(
            "Margins",
            "Operating Margin",
            format_percent(snapshot.operating_margin),
            "LTM operating income ÷ LTM revenue",
            _higher_is_better(snapshot.operating_margin, 0.20, 0.05),
        ),
        _card(
            "Margins",
            "Profit Margin",
            format_percent(snapshot.profit_margin),
            "LTM net income ÷ LTM revenue",
            _higher_is_better(snapshot.profit_margin, 0.10, 0.0),
        ),
        _card(
            "Margins",
            "FCF Margin",
            format_percent(snapshot.fcf_margin),
            "LTM free cash flow ÷ LTM revenue",
            _higher_is_better(snapshot.fcf_margin, 0.15, 0.05),
        ),
        _card(
            "Returns",
            "ROA",
            format_percent(snapshot.roa),
            "LTM net income ÷ average current/prior-year assets",
            _higher_is_better(snapshot.roa, 0.10, 0.03),
        ),
        _card(
            "Returns",
            "ROE",
            format_percent(snapshot.roe),
            "LTM net income ÷ average current/prior-year equity",
            _higher_is_better(snapshot.roe, 0.15, 0.08),
        ),
        _card(
            "Returns",
            "ROIC",
            format_percent(snapshot.roic),
            "After-tax operating income ÷ equity plus debt less liquid assets",
            _higher_is_better(snapshot.roic, 0.15, 0.08),
        ),
        _card(
            "Liquidity",
            "Current Ratio",
            format_multiple(snapshot.current_ratio),
            "Current assets ÷ current liabilities",
            _higher_is_better(snapshot.current_ratio, 1.50, 1.0),
        ),
        _card(
            "Liquidity",
            "Quick Ratio",
            format_multiple(snapshot.quick_ratio),
            "Cash, short-term investments, and receivables ÷ current liabilities",
            _higher_is_better(snapshot.quick_ratio, 1.0, 0.75),
        ),
        _card(
            "Capital",
            "Debt / Equity",
            format_multiple(snapshot.debt_equity),
            "Funded debt ÷ stockholders' equity",
            _lower_is_better(snapshot.debt_equity, 0.50, 1.50),
        ),
        _card(
            "Capital",
            "Dividend Yield",
            format_percent(snapshot.dividend_yield),
            "LTM common dividends paid ÷ market capitalization",
            _context_only(snapshot.dividend_yield),
        ),
        _card(
            "Capital",
            "Payout Ratio",
            format_percent(snapshot.payout_ratio),
            "LTM common dividends paid ÷ LTM net income",
            _lower_is_better(snapshot.payout_ratio, 0.60, 1.0),
        ),
    ]


def sec_snapshot_table(
    snapshot: ValuationSnapshot, *, currency: str = "USD"
) -> pd.DataFrame:
    return pd.DataFrame(sec_snapshot_cards(snapshot, currency=currency))[
        ["Section", "Metric", "Value", "Formula", "Context"]
    ]


def _quarter_change(metric: Any, periods: int) -> Optional[float]:
    if metric is None or len(metric.quarterly) <= periods:
        return None
    current = metric.quarterly[-1].value
    prior = metric.quarterly[-1 - periods].value
    if prior == 0:
        return None
    return (current - prior) / abs(prior)


def growth_table(metrics: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for label, key in (("Revenue", "revenue"), ("Diluted EPS", "eps_diluted")):
        metric = metrics.get(key)
        rows.append(
            {
                "Metric": label,
                "Q / Q": format_percent(_quarter_change(metric, 1)),
                "Y / Y": format_percent(_quarter_change(metric, 4)),
                "3Y CAGR": format_percent(annual_cagr(metric, 3)),
                "5Y CAGR": format_percent(annual_cagr(metric, 5)),
            }
        )
    return pd.DataFrame(rows)


def credit_table(snapshot: ValuationSnapshot, *, currency: str = "USD") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Metric": "Cash & Short-Term Investments",
                "Value": format_money(snapshot.liquid_assets, currency=currency),
                "Formula": "Latest SEC cash plus short-term investments",
            },
            {
                "Metric": "Funded Debt",
                "Value": format_money(snapshot.debt, currency=currency),
                "Formula": "Latest disclosed debt plus separately reported short-term borrowings",
            },
            {
                "Metric": "Debt / EBITDA",
                "Value": format_multiple(snapshot.debt_ebitda),
                "Formula": "Funded debt ÷ LTM calculated EBITDA",
            },
            {
                "Metric": "Net Debt / EBITDA",
                "Value": format_multiple(snapshot.net_debt_ebitda),
                "Formula": "Debt less liquid assets ÷ LTM calculated EBITDA",
            },
            {
                "Metric": "Interest Coverage",
                "Value": format_multiple(snapshot.interest_coverage),
                "Formula": "LTM calculated EBITDA ÷ LTM reported interest expense",
            },
            {
                "Metric": "LTM Interest Expense",
                "Value": format_money(snapshot.ltm_interest_expense, currency=currency),
                "Formula": "Latest four stand-alone quarters",
            },
        ]
    )


def format_source_audit(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return audit
    out = audit.copy()

    def display_value(row: pd.Series) -> str:
        value = pd.to_numeric(row.get("Latest Reported"), errors="coerce")
        if pd.isna(value):
            return "Unavailable"
        unit = str(row.get("Unit", ""))
        if unit == "shares":
            return f"{value:,.0f}"
        if "/shares" in unit.replace(" ", ""):
            return _signed_currency(float(value), unit.split("/")[0].strip(), 2)
        return _signed_currency(float(value), unit, 0)

    out["Latest Reported"] = out.apply(display_value, axis=1)
    return out


with st.sidebar:
    render_sidebar_about("9_ADFM_Underwriter.py")


render_page_header(
    PageHeader(
        title=TITLE,
        description=DESCRIPTION,
        eyebrow="ADFM Fundamental Research",
        source_note="SEC EDGAR Company Facts and submissions; Yahoo Finance completed-session price history",
    )
)

with st.form("issuer_search"):
    search_col, button_col = st.columns([5, 1])
    with search_col:
        query = st.text_input(
            "Ticker, CIK, or company name",
            value=st.session_state.get("underwriter_query", "AAPL"),
            placeholder="Examples: AAPL, 320193, Apple Inc.",
        )
    with button_col:
        st.markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run Underwrite", use_container_width=True)

if submitted:
    st.session_state["underwriter_query"] = query.strip()
    st.session_state["underwriter_active"] = True

if not st.session_state.get("underwriter_active", False):
    render_selection_note(
        "Start here",
        "Enter a ticker and run the underwrite. The page will retrieve the issuer's SEC filing history, normalize reported financials, and calculate current equity and credit ratios.",
    )
    render_footer()
    st.stop()

active_query = str(st.session_state.get("underwriter_query", query)).strip()

try:
    with st.spinner("Reading SEC filings and rebuilding the issuer model..."):
        identity = resolve_company(active_query, load_ticker_map())
        company_facts = load_company_facts(identity.cik)
        submissions = load_submissions(identity.cik)
        metrics = extract_metrics(company_facts)
        close_history, price, price_date = market_history(identity.ticker)
        filing_currency = statement_currency(metrics)
        valuation = build_valuation_snapshot(
            metrics,
            price=price if filing_currency == "USD" else None,
            price_date=price_date,
        )
except SecDataError as exc:
    st.error(str(exc))
    render_footer()
    st.stop()
except Exception as exc:
    st.error(f"The issuer model could not be built: {exc}")
    render_footer()
    st.stop()

currency = filing_currency
if currency != "USD":
    st.warning(
        f"This issuer reports primarily in {currency}. Current US-dollar market multiples are suppressed until a filing-currency FX conversion is available."
    )

latest_form = first_recent_value(submissions, "form")
latest_filed = first_recent_value(submissions, "filingDate")
sic_description = str(submissions.get("sicDescription", "Unavailable"))
fiscal_year_end = str(submissions.get("fiscalYearEnd", "Unavailable"))
render_issuer_masthead(
    ticker=identity.ticker,
    name=identity.name,
    price=price,
    price_date=price_date,
    sic_description=sic_description,
    fiscal_year_end=fiscal_year_end,
    latest_form=latest_form,
    latest_filed=latest_filed,
    price_currency="USD",
    filing_currency=currency,
    close_history=close_history,
)

if not close_history.empty:
    st.plotly_chart(
        price_history_chart(close_history, identity.ticker, "USD"),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )

rows = overview_cards(
    metrics,
    valuation,
    close_history,
    currency=currency,
)
render_underwriter_cards(rows)

st.caption(
    f"SEC Company Facts through the latest accepted filing; market price through {period_label(price_date)}. "
    "The overview uses reported historical data and current completed-session price only. "
    "Forward estimates, analyst targets, ownership and short-interest data are intentionally excluded."
)

render_annual_snapshot(metrics, currency)

with st.expander("Metric definitions & methodology", expanded=False):
    render_underwriter_legend()
    st.caption(
        "Valuation, leverage, profitability and liquidity colors use transparent absolute underwriting bands, "
        "not sector-relative rankings or investment recommendations. Growth and price-performance colors indicate direction only. "
        "Banks, insurers, REITs, pre-revenue companies and other sector-specific structures can require different thresholds."
    )
    render_underwriter_audit(rows)

with st.expander("Operating trajectory and issuer read-through", expanded=False, on_change="rerun") as trajectory_detail:
    if trajectory_detail.open:
        render_section_header(
            "Reported growth",
            "Quarterly comparisons and annual compound growth calculated from SEC filing periods. Non-positive CAGR bases remain unavailable.",
        )
        metric_table(growth_table(metrics))

        render_section_header(
            "Issuer read-through",
            "A deterministic first pass from the latest reported operating trajectory, cash conversion, leverage, and debt service.",
        )
        reads = underwrite_read(metrics, valuation, currency=currency)
        if reads:
            for label, text in reads:
                st.markdown(f"**{label}.** {text}")
        else:
            st.info(
                "The filing does not contain enough standardized data for an automated issuer read-through."
            )

        events = recent_filings(submissions, forms=("8-K", "6-K"), limit=8)
        render_section_header(
            "Recent SEC events",
            "Material current reports and foreign-issuer updates. These are filing events, not a general news feed.",
        )
        if events.empty:
            st.caption("No recent 8-K or 6-K filings were returned.")
        else:
            metric_table(
                events[["Filed", "Period", "Form", "Description", "Document"]],
                column_config={
                    "Document": st.column_config.LinkColumn(
                        "SEC Document", display_text="Open"
                    )
                },
            )

        st.caption(
            "Enterprise value includes separately tagged debt, preferred equity, and minority interest when available, and subtracts tagged cash and short-term investments. It does not infer missing pension, lease, derivative, or unconsolidated obligations. Forward estimates, analyst targets, short interest, and aggregated ownership are not calculated because they are not 10-K/10-Q Company Facts."
        )

with st.expander("Financials", expanded=False, on_change="rerun") as financial_detail:
    if financial_detail.open:
        quarterly = financial_table(
            metrics,
            ("revenue", "gross_profit", "operating_income", "net_income", "cfo", "capex"),
            frequency="quarterly",
            periods=12,
        )
        annual = financial_table(
            metrics,
            ("revenue", "gross_profit", "operating_income", "net_income", "cfo", "capex"),
            frequency="annual",
            periods=8,
        )
        balance_sheet = balance_sheet_table(
            metrics,
            (
                "cash",
                "short_term_investments",
                "receivables",
                "current_assets",
                "current_liabilities",
                "debt_current",
                "debt_noncurrent",
                "short_term_borrowings",
                "equity",
                "assets",
            ),
            periods=12,
        )

        render_section_header(
            "Quarterly operating record",
            f"Stand-alone quarters in {currency_prefix(currency)} millions. Cash-flow quarters can be mechanically derived from issuer-reported YTD values.",
        )
        if quarterly.empty:
            st.info("No standardized quarterly financial series were available.")
        else:
            if {"Revenue", "Operating Income"}.issubset(quarterly.columns):
                st.plotly_chart(
                    quarterly_chart(quarterly, currency),
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True},
                )
            quarterly_display = scale_financial_table(quarterly, currency)
            metric_table(quarterly_display)
            dataframe_download(
                "Download quarterly data",
                quarterly,
                f"{identity.ticker}_sec_quarterly.csv",
            )

        render_section_header(
            "Annual operating record",
            f"Full fiscal years in {currency_prefix(currency)} millions, using the latest-filed observation for each period.",
        )
        if not annual.empty:
            metric_table(scale_financial_table(annual, currency))
        else:
            st.caption("Unavailable")

        render_section_header(
            "Balance-sheet history",
            f"Point-in-time reported values in {currency_prefix(currency)} millions. No observations are forward-filled.",
        )
        if not balance_sheet.empty:
            metric_table(scale_financial_table(balance_sheet, currency))
        else:
            st.caption("Unavailable")

with st.expander("Credit", expanded=False, on_change="rerun") as credit_detail:
    if credit_detail.open:
        render_section_header(
            "Issuer credit profile",
            "Capital structure and debt-service measures from current market value and the latest SEC-reported balance sheet and income statement.",
        )
        metric_table(credit_table(valuation, currency=currency))

        maturities = maturity_table(company_facts)
        render_section_header(
            "Debt maturity ladder",
            "Standardized principal maturities from the latest filing. Many issuers place issue-level detail in custom tags or debt-footnote text, so missing buckets remain blank.",
        )
        if maturities.empty:
            st.caption(
                "The issuer did not expose a standardized debt maturity ladder through SEC Company Facts."
            )
        else:
            maturity_display = maturities.copy()
            maturity_display["Principal"] = (
                pd.to_numeric(maturity_display["Principal"], errors="coerce")
                .div(1_000_000)
                .map(
                    lambda value: (
                        _signed_currency(value, currency)
                        if pd.notna(value)
                        else "Unavailable"
                    )
                )
            )
            maturity_display = maturity_display.rename(
                columns={"Principal": f"Principal ({currency_prefix(currency)} millions)"}
            )
            metric_table(
                maturity_display,
                column_config={
                    "Source": st.column_config.LinkColumn("SEC Source", display_text="Open")
                },
            )

with st.expander("Filings & Sources", expanded=False, on_change="rerun") as filing_detail:
    if filing_detail.open:
        filings = recent_filings(submissions, limit=35)
        render_section_header(
            "Recent filings",
            "Direct links to the issuer's recent annual, quarterly, and current reports.",
        )
        if filings.empty:
            st.caption("No matching filing metadata was returned.")
        else:
            metric_table(
                filings,
                column_config={
                    "Document": st.column_config.LinkColumn(
                        "Primary Document", display_text="Open"
                    ),
                    "Filing Index": st.column_config.LinkColumn(
                        "Filing Index", display_text="Index"
                    ),
                },
            )

        audit = source_audit_table(metrics)
        render_section_header(
            "Source audit",
            "The exact taxonomy concept selected for each normalized metric, including reporting period, filing date, form, and source filing.",
        )
        if audit.empty:
            st.caption("No standardized source observations were available.")
        else:
            audit_display = format_source_audit(audit)
            metric_table(
                audit_display,
                column_config={
                    "Source": st.column_config.LinkColumn("SEC Source", display_text="Open")
                },
            )
            dataframe_download(
                "Download source audit",
                audit.drop(columns=["Source"]),
                f"{identity.ticker}_sec_source_audit.csv",
            )

render_footer(
    data_note=(
        "Primary inputs: SEC EDGAR Company Facts, SEC submissions, and Yahoo Finance completed-session price history. "
        "Calculated values disclose their formulas; missing filing concepts remain unavailable."
    )
)
