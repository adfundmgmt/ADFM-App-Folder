from __future__ import annotations
from typing import Any,Mapping,Optional
import pandas as pd
from adfm_engine.analytics.sec_fundamentals import *
TITLE = "ADFM Underwriter"
DESCRIPTION = (
    "Filing-driven company fundamentals, current valuation, capital structure, "
    "issuer-credit ratios, debt maturities, market context, and recent SEC events."
)
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

