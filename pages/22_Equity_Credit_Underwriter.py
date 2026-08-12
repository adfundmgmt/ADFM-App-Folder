from __future__ import annotations

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
    render_kpi_cards,
    render_page_header,
    render_section_header,
    render_selection_note,
)

TITLE = "Equity & Credit Underwriter"
DESCRIPTION = (
    "Filing-driven company fundamentals, current valuation, capital structure, "
    "issuer-credit ratios, debt maturities, and recent SEC events."
)

st.set_page_config(
    layout="wide",
    page_title=TITLE,
    initial_sidebar_state="expanded",
)
inject_explorer_style(max_width_px=1580)
configure_yfinance_cache()


@st.cache_data(ttl=86_400, show_spinner=False)
def load_ticker_map() -> Mapping[str, Any]:
    return SecClient().company_tickers()


@st.cache_data(ttl=900, show_spinner=False)
def load_company_facts(cik: int) -> Mapping[str, Any]:
    return SecClient().company_facts(cik)


@st.cache_data(ttl=900, show_spinner=False)
def load_submissions(cik: int) -> Mapping[str, Any]:
    return SecClient().submissions(cik)


def latest_market_close(ticker: str) -> tuple[Optional[float], Optional[pd.Timestamp]]:
    frames, _ = fetch_daily_ohlcv((ticker,), period="1mo")
    frame = frames.get(ticker)
    if frame is None or frame.empty or "Close" not in frame:
        return None, None
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if close.empty:
        return None, None
    return float(close.iloc[-1]), pd.Timestamp(close.index[-1]).normalize()


def format_money(value: Optional[float], *, currency: str = "USD") -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    magnitude = abs(float(value))
    if magnitude >= 1_000_000_000_000:
        scaled, suffix = value / 1_000_000_000_000, "T"
    elif magnitude >= 1_000_000_000:
        scaled, suffix = value / 1_000_000_000, "B"
    elif magnitude >= 1_000_000:
        scaled, suffix = value / 1_000_000, "M"
    else:
        scaled, suffix = value, ""
    prefix = "$" if currency == "USD" else f"{currency} "
    return f"{prefix}{scaled:,.2f}{suffix}"


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


def scale_financial_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in out.columns:
        if column != "Period End":
            out[column] = pd.to_numeric(out[column], errors="coerce") / 1_000_000
    out["Period End"] = pd.to_datetime(out["Period End"]).dt.date
    return out.rename(columns={column: f"{column}" for column in out.columns})


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
            name=f"Revenue ({unit} bn)",
            marker_color=PASTEL["blue"],
            hovertemplate="%{x|%Y-%m-%d}<br>Revenue: %{y:,.2f}bn<extra></extra>",
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
    fig.update_yaxes(title_text=f"Revenue ({unit} bn)", secondary_y=False, gridcolor="#e5e5e5")
    fig.update_yaxes(title_text="Operating margin", ticksuffix="%", secondary_y=True, showgrid=False)
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
        reads.append(("Earnings power", f"Latest reported-quarter operating margin was {margin * 100:.1f}%."))
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


def valuation_table(snapshot: ValuationSnapshot) -> pd.DataFrame:
    rows = (
        ("Market Capitalization", snapshot.market_cap, "Latest completed-session close × latest SEC shares outstanding"),
        ("Enterprise Value", snapshot.enterprise_value, "Market cap + funded debt + preferred + minority interest − cash and short-term investments"),
        ("P / E", snapshot.pe, "Market capitalization ÷ LTM net income available to common"),
        ("EV / Revenue", snapshot.ev_revenue, "Enterprise value ÷ LTM revenue"),
        ("EV / EBITDA", snapshot.ev_ebitda, "Enterprise value ÷ (LTM operating income + LTM D&A)"),
        ("FCF Yield", snapshot.fcf_yield, "(LTM operating cash flow − LTM capex) ÷ market capitalization"),
        ("Operating Margin", snapshot.operating_margin, "LTM operating income ÷ LTM revenue"),
        ("FCF Margin", snapshot.fcf_margin, "LTM free cash flow ÷ LTM revenue"),
    )
    output: list[dict[str, Any]] = []
    for metric, value, formula in rows:
        if metric in {"P / E", "EV / Revenue", "EV / EBITDA"}:
            display = format_multiple(value)
        elif metric in {"FCF Yield", "Operating Margin", "FCF Margin"}:
            display = format_percent(value)
        else:
            display = format_money(value)
        output.append({"Metric": metric, "Value": display, "Formula": formula})
    return pd.DataFrame(output)


def credit_table(snapshot: ValuationSnapshot, *, currency: str = "USD") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Metric": "Cash & Short-Term Investments", "Value": format_money(snapshot.liquid_assets, currency=currency), "Formula": "Latest SEC cash plus short-term investments"},
            {"Metric": "Funded Debt", "Value": format_money(snapshot.debt, currency=currency), "Formula": "Latest disclosed debt plus separately reported short-term borrowings"},
            {"Metric": "Debt / EBITDA", "Value": format_multiple(snapshot.debt_ebitda), "Formula": "Funded debt ÷ LTM calculated EBITDA"},
            {"Metric": "Net Debt / EBITDA", "Value": format_multiple(snapshot.net_debt_ebitda), "Formula": "Debt less liquid assets ÷ LTM calculated EBITDA"},
            {"Metric": "Interest Coverage", "Value": format_multiple(snapshot.interest_coverage), "Formula": "LTM calculated EBITDA ÷ LTM reported interest expense"},
            {"Metric": "LTM Interest Expense", "Value": format_money(snapshot.ltm_interest_expense, currency=currency), "Formula": "Latest four stand-alone quarters"},
        ]
    )


with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Underwrite an SEC-reporting company from primary filings, then connect the equity valuation to the issuer's balance sheet and debt-service capacity.

        **How to read it**
        - Reported values come from SEC Company Facts or filing metadata.
        - LTM values sum the latest four stand-alone quarters.
        - Cash-flow quarters may be derived from cumulative YTD disclosures.
        - Market value uses the latest completed Yahoo Finance close.
        - Missing or incompatible fields remain unavailable.
        """
    )
    st.markdown("---")
    st.header("Coverage")
    st.caption(
        "Best coverage is for US issuers with standardized XBRL. Banks, insurers, foreign private issuers, partnerships, and custom-tag-heavy filers may require issuer-specific adjustments."
    )


render_page_header(
    PageHeader(
        title=TITLE,
        description=DESCRIPTION,
        eyebrow="ADFM Fundamental Research",
        source_note="SEC EDGAR Company Facts and submissions; Yahoo Finance completed-session close",
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
        price, price_date = latest_market_close(identity.ticker)
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
render_selection_note(
    f"{identity.ticker} · {identity.name}",
    f"CIK {identity.padded_cik} · {sic_description} · Fiscal year end {fiscal_year_end} · Latest filing {latest_form} on {latest_filed}",
)

render_kpi_cards(
    [
        ("Price", format_money(price), f"Close through {period_label(price_date)}"),
        ("Market Cap", format_money(valuation.market_cap), "Price × SEC shares"),
        ("Enterprise Value", format_money(valuation.enterprise_value), "Calculated capital value"),
        ("LTM Revenue", format_money(valuation.ltm_revenue, currency=currency), "Latest four quarters"),
        ("LTM Free Cash Flow", format_money(valuation.ltm_fcf, currency=currency), "CFO less capex"),
        ("Net Debt / EBITDA", format_multiple(valuation.net_debt_ebitda), "Calculated issuer leverage"),
    ]
)
render_kpi_cards(
    [
        ("P / E", format_multiple(valuation.pe), "Current ÷ LTM"),
        ("EV / Revenue", format_multiple(valuation.ev_revenue), "Current ÷ LTM"),
        ("EV / EBITDA", format_multiple(valuation.ev_ebitda), "Calculated EBITDA"),
        ("FCF Yield", format_percent(valuation.fcf_yield), "LTM FCF ÷ market cap"),
        ("Operating Margin", format_percent(valuation.operating_margin), "LTM operating income"),
        ("Interest Coverage", format_multiple(valuation.interest_coverage), "EBITDA ÷ interest"),
    ]
)

underwrite_tab, financials_tab, valuation_tab, credit_tab, filings_tab = st.tabs(
    ["Underwrite", "Financials", "Valuation", "Credit", "Filings & Sources"]
)

with underwrite_tab:
    render_section_header(
        "Issuer read-through",
        "A deterministic first pass from the latest reported operating trajectory, cash conversion, leverage, and debt service.",
    )
    reads = underwrite_read(metrics, valuation, currency=currency)
    if reads:
        for label, text in reads:
            st.markdown(f"**{label}.** {text}")
    else:
        st.info("The filing does not contain enough standardized data for an automated issuer read-through.")

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
            column_config={"Document": st.column_config.LinkColumn("SEC Document", display_text="Open")},
        )

with financials_tab:
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
        ("cash", "short_term_investments", "debt_current", "debt_noncurrent", "short_term_borrowings", "equity"),
        periods=12,
    )

    render_section_header(
        "Quarterly operating record",
        f"Stand-alone quarters in {currency} millions. Cash-flow quarters can be mechanically derived from issuer-reported YTD values.",
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
        quarterly_display = scale_financial_table(quarterly)
        metric_table(quarterly_display)
        dataframe_download(
            "Download quarterly data",
            quarterly,
            f"{identity.ticker}_sec_quarterly.csv",
        )

    render_section_header(
        "Annual operating record",
        f"Full fiscal years in {currency} millions, using the latest-filed observation for each period.",
    )
    metric_table(scale_financial_table(annual)) if not annual.empty else st.caption("Unavailable")

    render_section_header(
        "Balance-sheet history",
        f"Point-in-time reported values in {currency} millions. No observations are forward-filled.",
    )
    metric_table(scale_financial_table(balance_sheet)) if not balance_sheet.empty else st.caption("Unavailable")

with valuation_tab:
    render_section_header(
        "Current valuation",
        "Every multiple is calculated in the app. SEC supplies the denominator inputs; Yahoo Finance supplies the latest completed-session close.",
    )
    if currency == "USD":
        metric_table(valuation_table(valuation))
    else:
        st.info("Valuation multiples are unavailable because the filing currency is not USD.")

    st.caption(
        "Enterprise value includes separately tagged debt, preferred equity, and minority interest when available, and subtracts tagged cash and short-term investments. It does not infer missing pension, lease, derivative, or unconsolidated obligations."
    )

with credit_tab:
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
        st.caption("The issuer did not expose a standardized debt maturity ladder through SEC Company Facts.")
    else:
        maturity_display = maturities.copy()
        maturity_display["Principal"] = pd.to_numeric(maturity_display["Principal"], errors="coerce") / 1_000_000
        maturity_display = maturity_display.rename(columns={"Principal": f"Principal ({currency} mm)"})
        metric_table(
            maturity_display,
            column_config={"Source": st.column_config.LinkColumn("SEC Source", display_text="Open")},
        )

with filings_tab:
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
                "Document": st.column_config.LinkColumn("Primary Document", display_text="Open"),
                "Filing Index": st.column_config.LinkColumn("Filing Index", display_text="Index"),
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
        metric_table(
            audit,
            column_config={"Source": st.column_config.LinkColumn("SEC Source", display_text="Open")},
        )
        dataframe_download(
            "Download source audit",
            audit.drop(columns=["Source"]),
            f"{identity.ticker}_sec_source_audit.csv",
        )

render_footer(
    data_note=(
        "Primary inputs: SEC EDGAR Company Facts, SEC submissions, and Yahoo Finance completed-session close. "
        "Calculated values disclose their formulas; missing filing concepts remain unavailable."
    )
)
