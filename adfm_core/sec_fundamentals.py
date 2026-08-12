"""SEC EDGAR fundamentals and issuer-credit normalization.

The module keeps official filing observations separate from market prices.  It
normalizes common US-GAAP/IFRS concepts, reconstructs stand-alone quarters from
year-to-date disclosures, and exposes every calculated field with its filing
provenance.  Missing concepts remain missing rather than being estimated.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd
import requests

SEC_DATA_BASE = "https://data.sec.gov"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FORMS = ("10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A", "40-F", "40-F/A")
DEFAULT_SEC_USER_AGENT = "ADFM Analytics research@adfundmgmt.com"


class SecDataError(RuntimeError):
    """Raised when EDGAR cannot return a usable response."""


@dataclass(frozen=True)
class CompanyIdentity:
    """Resolved EDGAR issuer identity."""

    cik: int
    ticker: str
    name: str

    @property
    def padded_cik(self) -> str:
        return f"{self.cik:010d}"


@dataclass(frozen=True)
class ConceptSpec:
    """One normalized metric and its ordered taxonomy candidates."""

    label: str
    statement: str
    unit_kind: str
    candidates: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Observation:
    """One reported or mechanically derived filing observation."""

    value: float
    unit: str
    start: Optional[pd.Timestamp]
    end: pd.Timestamp
    filed: Optional[pd.Timestamp]
    form: str
    accession: str
    fiscal_year: Optional[int]
    fiscal_period: str
    taxonomy: str
    concept: str
    derived: bool = False
    derivation: str = "Reported"

    @property
    def source_url(self) -> str:
        return filing_index_url(self.accession)


@dataclass(frozen=True)
class MetricSeries:
    """Normalized observations for one financial metric."""

    key: str
    spec: ConceptSpec
    taxonomy: Optional[str]
    concept: Optional[str]
    unit: Optional[str]
    quarterly: tuple[Observation, ...] = ()
    annual: tuple[Observation, ...] = ()
    instant: tuple[Observation, ...] = ()

    @property
    def available(self) -> bool:
        return bool(self.quarterly or self.annual or self.instant)


@dataclass(frozen=True)
class ValuationSnapshot:
    """Market-value and credit calculations built from EDGAR plus one price."""

    price: Optional[float]
    price_date: Optional[pd.Timestamp]
    shares: Optional[float]
    diluted_shares: Optional[float]
    market_cap: Optional[float]
    cash: Optional[float]
    short_term_investments: Optional[float]
    liquid_assets: Optional[float]
    receivables: Optional[float]
    assets: Optional[float]
    current_assets: Optional[float]
    current_liabilities: Optional[float]
    debt: Optional[float]
    equity: Optional[float]
    preferred_equity: Optional[float]
    minority_interest: Optional[float]
    enterprise_value: Optional[float]
    ltm_revenue: Optional[float]
    ltm_gross_profit: Optional[float]
    ltm_operating_income: Optional[float]
    ltm_net_income: Optional[float]
    ltm_cfo: Optional[float]
    ltm_capex: Optional[float]
    ltm_fcf: Optional[float]
    ltm_da: Optional[float]
    ltm_ebitda: Optional[float]
    ltm_interest_expense: Optional[float]
    ltm_dividends: Optional[float]
    eps: Optional[float]
    sales_per_share: Optional[float]
    book_per_share: Optional[float]
    cash_per_share: Optional[float]
    pe: Optional[float]
    price_sales: Optional[float]
    price_book: Optional[float]
    price_cash: Optional[float]
    price_fcf: Optional[float]
    ev_revenue: Optional[float]
    ev_ebitda: Optional[float]
    fcf_yield: Optional[float]
    gross_margin: Optional[float]
    operating_margin: Optional[float]
    profit_margin: Optional[float]
    fcf_margin: Optional[float]
    current_ratio: Optional[float]
    quick_ratio: Optional[float]
    debt_equity: Optional[float]
    roa: Optional[float]
    roe: Optional[float]
    roic: Optional[float]
    dividend_yield: Optional[float]
    payout_ratio: Optional[float]
    debt_ebitda: Optional[float]
    net_debt_ebitda: Optional[float]
    interest_coverage: Optional[float]


def _candidate(*items: tuple[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(items)


CONCEPT_SPECS: Mapping[str, ConceptSpec] = {
    "revenue": ConceptSpec(
        "Revenue",
        "duration",
        "currency",
        _candidate(
            ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
            ("us-gaap", "Revenues"),
            ("us-gaap", "SalesRevenueNet"),
            ("ifrs-full", "Revenue"),
        ),
    ),
    "gross_profit": ConceptSpec(
        "Gross Profit",
        "duration",
        "currency",
        _candidate(("us-gaap", "GrossProfit"), ("ifrs-full", "GrossProfit")),
    ),
    "operating_income": ConceptSpec(
        "Operating Income",
        "duration",
        "currency",
        _candidate(
            ("us-gaap", "OperatingIncomeLoss"),
            ("ifrs-full", "ProfitLossFromOperatingActivities"),
        ),
    ),
    "net_income": ConceptSpec(
        "Net Income",
        "duration",
        "currency",
        _candidate(
            ("us-gaap", "NetIncomeLossAvailableToCommonStockholdersBasic"),
            ("us-gaap", "NetIncomeLoss"),
            ("us-gaap", "ProfitLoss"),
            ("ifrs-full", "ProfitLossAttributableToOwnersOfParent"),
            ("ifrs-full", "ProfitLoss"),
        ),
    ),
    "cfo": ConceptSpec(
        "Cash From Operations",
        "duration",
        "currency",
        _candidate(
            ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
            ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
            ("ifrs-full", "CashFlowsFromUsedInOperatingActivities"),
        ),
    ),
    "capex": ConceptSpec(
        "Capital Expenditures",
        "duration",
        "currency",
        _candidate(
            ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
            ("us-gaap", "PaymentsForAdditionsToPropertyPlantAndEquipment"),
            ("ifrs-full", "PurchaseOfPropertyPlantAndEquipment"),
        ),
    ),
    "da": ConceptSpec(
        "Depreciation & Amortization",
        "duration",
        "currency",
        _candidate(
            ("us-gaap", "DepreciationDepletionAndAmortization"),
            ("us-gaap", "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment"),
            ("us-gaap", "Depreciation"),
            ("ifrs-full", "DepreciationAndAmortisationExpense"),
        ),
    ),
    "interest_expense": ConceptSpec(
        "Interest Expense",
        "duration",
        "currency",
        _candidate(
            ("us-gaap", "InterestExpenseNonOperating"),
            ("us-gaap", "InterestAndDebtExpense"),
            ("us-gaap", "InterestExpense"),
            ("ifrs-full", "FinanceCosts"),
        ),
    ),
    "pretax_income": ConceptSpec(
        "Pre-Tax Income",
        "duration",
        "currency",
        _candidate(
            (
                "us-gaap",
                "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            ),
            ("us-gaap", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"),
            ("ifrs-full", "ProfitLossBeforeTax"),
        ),
    ),
    "income_tax_expense": ConceptSpec(
        "Income Tax Expense",
        "duration",
        "currency",
        _candidate(
            ("us-gaap", "IncomeTaxExpenseBenefit"),
            ("ifrs-full", "IncomeTaxExpenseContinuingOperations"),
            ("ifrs-full", "IncomeTaxExpenseBenefit"),
        ),
    ),
    "dividends_paid": ConceptSpec(
        "Common Dividends Paid",
        "duration",
        "currency",
        _candidate(
            ("us-gaap", "PaymentsOfDividendsCommonStock"),
            ("us-gaap", "PaymentsOfDividends"),
            ("ifrs-full", "DividendsPaid"),
        ),
    ),
    "eps_diluted": ConceptSpec(
        "Diluted EPS",
        "duration",
        "per_share",
        _candidate(
            ("us-gaap", "EarningsPerShareDiluted"),
            ("us-gaap", "EarningsPerShareBasicAndDiluted"),
            ("ifrs-full", "DilutedEarningsLossPerShare"),
        ),
    ),
    "diluted_shares": ConceptSpec(
        "Diluted Weighted-Average Shares",
        "duration",
        "shares",
        _candidate(
            ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
            ("us-gaap", "WeightedAverageNumberOfShareOutstandingBasicAndDiluted"),
            ("ifrs-full", "AdjustedWeightedAverageSharesDiluted"),
        ),
    ),
    "shares_outstanding": ConceptSpec(
        "Common Shares Outstanding",
        "instant",
        "shares",
        _candidate(
            ("dei", "EntityCommonStockSharesOutstanding"),
            ("us-gaap", "CommonStockSharesOutstanding"),
        ),
    ),
    "cash": ConceptSpec(
        "Cash & Equivalents",
        "instant",
        "currency",
        _candidate(
            ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
            ("ifrs-full", "CashAndCashEquivalents"),
        ),
    ),
    "short_term_investments": ConceptSpec(
        "Short-Term Investments",
        "instant",
        "currency",
        _candidate(
            ("us-gaap", "ShortTermInvestments"),
            ("us-gaap", "MarketableSecuritiesCurrent"),
            ("us-gaap", "AvailableForSaleSecuritiesCurrent"),
            ("ifrs-full", "CurrentFinancialAssetsAtFairValueThroughProfitOrLoss"),
        ),
    ),
    "receivables": ConceptSpec(
        "Accounts Receivable",
        "instant",
        "currency",
        _candidate(
            ("us-gaap", "AccountsReceivableNetCurrent"),
            ("us-gaap", "AccountsNotesAndLoansReceivableNetCurrent"),
            ("ifrs-full", "TradeAndOtherCurrentReceivables"),
        ),
    ),
    "assets": ConceptSpec(
        "Total Assets",
        "instant",
        "currency",
        _candidate(("us-gaap", "Assets"), ("ifrs-full", "Assets")),
    ),
    "current_assets": ConceptSpec(
        "Current Assets",
        "instant",
        "currency",
        _candidate(("us-gaap", "AssetsCurrent"), ("ifrs-full", "CurrentAssets")),
    ),
    "current_liabilities": ConceptSpec(
        "Current Liabilities",
        "instant",
        "currency",
        _candidate(
            ("us-gaap", "LiabilitiesCurrent"),
            ("ifrs-full", "CurrentLiabilities"),
        ),
    ),
    "debt_total": ConceptSpec(
        "Long-Term Debt & Finance Leases",
        "instant",
        "currency",
        _candidate(
            ("us-gaap", "LongTermDebtAndFinanceLeaseObligations"),
            ("us-gaap", "LongTermDebtAndCapitalLeaseObligations"),
            ("us-gaap", "LongTermDebt"),
            ("ifrs-full", "Borrowings"),
        ),
    ),
    "debt_current": ConceptSpec(
        "Current Debt",
        "instant",
        "currency",
        _candidate(
            ("us-gaap", "LongTermDebtAndFinanceLeaseObligationsCurrent"),
            ("us-gaap", "LongTermDebtAndCapitalLeaseObligationsCurrent"),
            ("us-gaap", "LongTermDebtCurrent"),
            ("us-gaap", "DebtCurrent"),
            ("ifrs-full", "CurrentBorrowings"),
        ),
    ),
    "debt_noncurrent": ConceptSpec(
        "Noncurrent Debt",
        "instant",
        "currency",
        _candidate(
            ("us-gaap", "LongTermDebtAndFinanceLeaseObligationsNoncurrent"),
            ("us-gaap", "LongTermDebtAndCapitalLeaseObligationsNoncurrent"),
            ("us-gaap", "LongTermDebtNoncurrent"),
            ("ifrs-full", "NoncurrentBorrowings"),
        ),
    ),
    "short_term_borrowings": ConceptSpec(
        "Short-Term Borrowings",
        "instant",
        "currency",
        _candidate(
            ("us-gaap", "ShortTermBorrowings"),
            ("us-gaap", "CommercialPaper"),
        ),
    ),
    "equity": ConceptSpec(
        "Stockholders' Equity",
        "instant",
        "currency",
        _candidate(
            ("us-gaap", "StockholdersEquity"),
            ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
            ("ifrs-full", "EquityAttributableToOwnersOfParent"),
            ("ifrs-full", "Equity"),
        ),
    ),
    "preferred_equity": ConceptSpec(
        "Preferred Equity",
        "instant",
        "currency",
        _candidate(
            ("us-gaap", "PreferredStocksIncludingAdditionalPaidInCapital"),
            ("us-gaap", "PreferredStockValue"),
        ),
    ),
    "minority_interest": ConceptSpec(
        "Minority Interest",
        "instant",
        "currency",
        _candidate(
            ("us-gaap", "MinorityInterest"),
            ("us-gaap", "NoncontrollingInterestInConsolidatedEntity"),
        ),
    ),
}


MATURITY_SPECS: Mapping[str, ConceptSpec] = {
    "Within 1 year": ConceptSpec(
        "Within 1 year",
        "instant",
        "currency",
        _candidate(
            ("us-gaap", "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths"),
            ("us-gaap", "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearOne"),
        ),
    ),
    "Year 2": ConceptSpec(
        "Year 2",
        "instant",
        "currency",
        _candidate(("us-gaap", "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo")),
    ),
    "Year 3": ConceptSpec(
        "Year 3",
        "instant",
        "currency",
        _candidate(("us-gaap", "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree")),
    ),
    "Year 4": ConceptSpec(
        "Year 4",
        "instant",
        "currency",
        _candidate(("us-gaap", "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour")),
    ),
    "Year 5": ConceptSpec(
        "Year 5",
        "instant",
        "currency",
        _candidate(("us-gaap", "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive")),
    ),
    "After year 5": ConceptSpec(
        "After year 5",
        "instant",
        "currency",
        _candidate(("us-gaap", "LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive")),
    ),
}


class SecClient:
    """Small fair-access EDGAR client with retries and an explicit user agent."""

    def __init__(
        self,
        user_agent: Optional[str] = None,
        *,
        timeout_seconds: float = 20.0,
        retries: int = 3,
        retry_pause_seconds: float = 0.7,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.user_agent = (
            user_agent
            or os.getenv("SEC_USER_AGENT")
            or DEFAULT_SEC_USER_AGENT
        ).strip()
        self.timeout_seconds = timeout_seconds
        self.retries = max(1, int(retries))
        self.retry_pause_seconds = max(0.0, float(retry_pause_seconds))
        self.session = session or requests.Session()

    @property
    def headers(self) -> dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
        }

    def get_json(self, url: str) -> Mapping[str, Any]:
        headers = dict(self.headers)
        if url.startswith(SEC_DATA_BASE):
            headers["Host"] = "data.sec.gov"
        last_error: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(
                    url,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, Mapping):
                    raise SecDataError(f"SEC returned a non-object payload for {url}")
                return payload
            except (requests.RequestException, ValueError, SecDataError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.retry_pause_seconds * (attempt + 1))
        raise SecDataError(f"SEC request failed for {url}: {last_error}")

    def company_tickers(self) -> Mapping[str, Any]:
        return self.get_json(SEC_TICKER_URL)

    def company_facts(self, cik: int) -> Mapping[str, Any]:
        return self.get_json(
            f"{SEC_DATA_BASE}/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
        )

    def submissions(self, cik: int) -> Mapping[str, Any]:
        return self.get_json(f"{SEC_DATA_BASE}/submissions/CIK{int(cik):010d}.json")


def ticker_records(payload: Mapping[str, Any]) -> list[CompanyIdentity]:
    """Normalize the SEC ticker map into stable issuer identities."""
    records: list[CompanyIdentity] = []
    for raw in payload.values():
        if not isinstance(raw, Mapping):
            continue
        try:
            records.append(
                CompanyIdentity(
                    cik=int(raw["cik_str"]),
                    ticker=str(raw["ticker"]).upper().strip(),
                    name=str(raw["title"]).strip(),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(records, key=lambda item: (item.ticker, item.name))


def resolve_company(
    query: str, ticker_payload: Mapping[str, Any]
) -> CompanyIdentity:
    """Resolve ticker, CIK, or an unambiguous company name."""
    clean = query.strip()
    if not clean:
        raise SecDataError("Enter a ticker, CIK, or company name.")
    records = ticker_records(ticker_payload)
    if clean.isdigit():
        cik = int(clean)
        match = next((record for record in records if record.cik == cik), None)
        if match is not None:
            return match
        raise SecDataError(f"CIK {clean} was not found in the SEC ticker map.")

    upper = clean.upper()
    match = next((record for record in records if record.ticker == upper), None)
    if match is not None:
        return match

    exact_names = [record for record in records if record.name.upper() == upper]
    if len(exact_names) == 1:
        return exact_names[0]
    partial = [record for record in records if upper in record.name.upper()]
    if len(partial) == 1:
        return partial[0]
    if partial:
        suggestions = ", ".join(record.ticker for record in partial[:6])
        raise SecDataError(
            f"Company name is ambiguous. Use a ticker or CIK. Matches include: {suggestions}."
        )
    raise SecDataError(f"No SEC reporting company matched {clean!r}.")


def _as_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _as_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if pd.notna(parsed) else None


def _preferred_units(units: Mapping[str, Any], unit_kind: str) -> list[str]:
    if unit_kind == "shares":
        preferred = ["shares"]
    elif unit_kind == "per_share":
        preferred = ["USD/shares", "USD / shares"]
    else:
        preferred = ["USD"]
    available = [str(unit) for unit in units]
    return [unit for unit in preferred if unit in units] + [
        unit for unit in available if unit not in preferred
    ]


def _raw_concept_facts(
    company_facts: Mapping[str, Any],
    taxonomy: str,
    concept: str,
    unit_kind: str,
) -> tuple[str, list[Mapping[str, Any]]]:
    concept_payload = (
        company_facts.get("facts", {})
        .get(taxonomy, {})
        .get(concept, {})
    )
    units = concept_payload.get("units", {}) if isinstance(concept_payload, Mapping) else {}
    if not isinstance(units, Mapping):
        return "", []
    for unit in _preferred_units(units, unit_kind):
        facts = units.get(unit, [])
        if isinstance(facts, list) and facts:
            return unit, [fact for fact in facts if isinstance(fact, Mapping)]
    return "", []


def _fact_latest_end(facts: Sequence[Mapping[str, Any]]) -> pd.Timestamp:
    dates = [_as_timestamp(fact.get("end")) for fact in facts]
    valid = [value for value in dates if value is not None]
    return max(valid) if valid else pd.Timestamp.min


def _select_concept(
    company_facts: Mapping[str, Any], spec: ConceptSpec
) -> tuple[Optional[str], Optional[str], Optional[str], list[Mapping[str, Any]]]:
    choices: list[tuple[pd.Timestamp, int, int, str, str, str, list[Mapping[str, Any]]]] = []
    for priority, (taxonomy, concept) in enumerate(spec.candidates):
        unit, facts = _raw_concept_facts(
            company_facts, taxonomy, concept, spec.unit_kind
        )
        eligible = [fact for fact in facts if str(fact.get("form", "")) in SEC_FORMS]
        if eligible:
            choices.append(
                (
                    _fact_latest_end(eligible),
                    len(eligible),
                    -priority,
                    taxonomy,
                    concept,
                    unit,
                    eligible,
                )
            )
    if not choices:
        return None, None, None, []
    _, _, _, taxonomy, concept, unit, facts = max(choices, key=lambda item: item[:3])
    return taxonomy, concept, unit, facts


def _observation(
    fact: Mapping[str, Any],
    *,
    taxonomy: str,
    concept: str,
    unit: str,
) -> Optional[Observation]:
    value = _as_float(fact.get("val"))
    end = _as_timestamp(fact.get("end"))
    if value is None or end is None:
        return None
    fiscal_year = fact.get("fy")
    try:
        fy = int(fiscal_year) if fiscal_year is not None else None
    except (TypeError, ValueError):
        fy = None
    return Observation(
        value=value,
        unit=unit,
        start=_as_timestamp(fact.get("start")),
        end=end,
        filed=_as_timestamp(fact.get("filed")),
        form=str(fact.get("form", "")),
        accession=str(fact.get("accn", "")),
        fiscal_year=fy,
        fiscal_period=str(fact.get("fp", "")),
        taxonomy=taxonomy,
        concept=concept,
    )


def _filed_sort_value(observation: Observation) -> pd.Timestamp:
    return observation.filed or pd.Timestamp.min


def _dedupe_intervals(observations: Iterable[Observation]) -> list[Observation]:
    latest: dict[tuple[Optional[pd.Timestamp], pd.Timestamp], Observation] = {}
    for observation in observations:
        key = (observation.start, observation.end)
        current = latest.get(key)
        if current is None or _filed_sort_value(observation) >= _filed_sort_value(current):
            latest[key] = observation
    return sorted(latest.values(), key=lambda item: (item.end, _filed_sort_value(item)))


def _duration_days(observation: Observation) -> Optional[int]:
    if observation.start is None:
        return None
    return int((observation.end - observation.start).days)


def quarterly_observations(observations: Iterable[Observation]) -> tuple[Observation, ...]:
    """Return stand-alone quarters, deriving deltas from YTD disclosures when needed."""
    intervals = _dedupe_intervals(
        observation for observation in observations if observation.start is not None
    )
    explicit: dict[pd.Timestamp, Observation] = {}
    for observation in intervals:
        days = _duration_days(observation)
        if days is not None and 60 <= days <= 130:
            current = explicit.get(observation.end)
            if current is None or _filed_sort_value(observation) >= _filed_sort_value(current):
                explicit[observation.end] = observation

    derived: dict[pd.Timestamp, Observation] = {}
    by_start: dict[pd.Timestamp, list[Observation]] = {}
    for observation in intervals:
        if observation.start is not None:
            by_start.setdefault(observation.start, []).append(observation)
    for same_start in by_start.values():
        ordered = sorted(same_start, key=lambda item: item.end)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            delta_days = int((current.end - previous.end).days)
            current_days = _duration_days(current)
            if not (60 <= delta_days <= 130 and current_days is not None and current_days >= 130):
                continue
            candidate = Observation(
                value=current.value - previous.value,
                unit=current.unit,
                start=previous.end + pd.Timedelta(days=1),
                end=current.end,
                filed=current.filed,
                form=current.form,
                accession=current.accession,
                fiscal_year=current.fiscal_year,
                fiscal_period=current.fiscal_period,
                taxonomy=current.taxonomy,
                concept=current.concept,
                derived=True,
                derivation="Current YTD less prior YTD",
            )
            existing = derived.get(candidate.end)
            if existing is None or _filed_sort_value(candidate) >= _filed_sort_value(existing):
                derived[candidate.end] = candidate

    combined = dict(derived)
    combined.update(explicit)
    return tuple(sorted(combined.values(), key=lambda item: item.end))


def annual_observations(observations: Iterable[Observation]) -> tuple[Observation, ...]:
    """Return full-year duration facts with latest-filed restatements retained."""
    annuals: dict[pd.Timestamp, Observation] = {}
    for observation in _dedupe_intervals(observations):
        days = _duration_days(observation)
        if days is None or not 300 <= days <= 430:
            continue
        current = annuals.get(observation.end)
        if current is None or _filed_sort_value(observation) >= _filed_sort_value(current):
            annuals[observation.end] = observation
    return tuple(sorted(annuals.values(), key=lambda item: item.end))


def instant_observations(observations: Iterable[Observation]) -> tuple[Observation, ...]:
    """Return point-in-time observations with one latest-filed value per date."""
    latest: dict[pd.Timestamp, Observation] = {}
    for observation in observations:
        current = latest.get(observation.end)
        if current is None or _filed_sort_value(observation) >= _filed_sort_value(current):
            latest[observation.end] = observation
    return tuple(sorted(latest.values(), key=lambda item: item.end))


def extract_metric(
    company_facts: Mapping[str, Any], key: str, spec: Optional[ConceptSpec] = None
) -> MetricSeries:
    """Extract one normalized metric from a Company Facts payload."""
    selected = spec or CONCEPT_SPECS[key]
    taxonomy, concept, unit, raw_facts = _select_concept(company_facts, selected)
    if taxonomy is None or concept is None or unit is None:
        return MetricSeries(key=key, spec=selected, taxonomy=None, concept=None, unit=None)
    observations = tuple(
        observation
        for fact in raw_facts
        if (
            observation := _observation(
                fact,
                taxonomy=taxonomy,
                concept=concept,
                unit=unit,
            )
        )
        is not None
    )
    if selected.statement == "instant":
        return MetricSeries(
            key=key,
            spec=selected,
            taxonomy=taxonomy,
            concept=concept,
            unit=unit,
            instant=instant_observations(observations),
        )
    return MetricSeries(
        key=key,
        spec=selected,
        taxonomy=taxonomy,
        concept=concept,
        unit=unit,
        quarterly=quarterly_observations(observations),
        annual=annual_observations(observations),
    )


def extract_metrics(company_facts: Mapping[str, Any]) -> dict[str, MetricSeries]:
    """Extract the supported issuer-level financial statement concepts."""
    return {key: extract_metric(company_facts, key) for key in CONCEPT_SPECS}


def latest_observation(metric: Optional[MetricSeries]) -> Optional[Observation]:
    """Latest point-in-time observation for a normalized metric."""
    if metric is None:
        return None
    candidates = metric.instant or metric.quarterly or metric.annual
    return candidates[-1] if candidates else None


def ltm_value(metric: Optional[MetricSeries]) -> Optional[float]:
    """Sum the latest four consecutive stand-alone quarters."""
    if metric is None or len(metric.quarterly) < 4:
        return None
    latest = list(metric.quarterly[-4:])
    if int((latest[-1].end - latest[0].end).days) > 410:
        return None
    gaps = [
        int((right.end - left.end).days)
        for left, right in zip(latest, latest[1:], strict=False)
    ]
    if any(gap < 55 or gap > 140 for gap in gaps):
        return None
    return float(sum(observation.value for observation in latest))


def average_quarter_value(metric: Optional[MetricSeries]) -> Optional[float]:
    """Average the latest four consecutive stand-alone quarterly observations."""
    if metric is None or len(metric.quarterly) < 4:
        return None
    latest = list(metric.quarterly[-4:])
    gaps = [
        int((right.end - left.end).days)
        for left, right in zip(latest, latest[1:], strict=False)
    ]
    if int((latest[-1].end - latest[0].end).days) > 410:
        return None
    if any(gap < 55 or gap > 140 for gap in gaps):
        return None
    return float(sum(observation.value for observation in latest) / len(latest))


def average_balance_value(metric: Optional[MetricSeries]) -> Optional[float]:
    """Average the latest balance with its closest prior-year observation."""
    if metric is None or len(metric.instant) < 2:
        return None
    latest = metric.instant[-1]
    prior_candidates = [
        observation
        for observation in metric.instant[:-1]
        if 330 <= int((latest.end - observation.end).days) <= 400
    ]
    if not prior_candidates:
        return None
    return float((latest.value + prior_candidates[-1].value) / 2.0)


def annual_cagr(metric: Optional[MetricSeries], years: int) -> Optional[float]:
    """Calculate an annual filing CAGR over approximately ``years`` fiscal years."""
    if metric is None or len(metric.annual) < 2 or int(years) <= 0:
        return None
    latest = metric.annual[-1]
    candidates = [
        observation
        for observation in metric.annual[:-1]
        if 300 * int(years)
        <= int((latest.end - observation.end).days)
        <= 430 * int(years)
    ]
    if not candidates or latest.value <= 0:
        return None
    prior = min(
        candidates,
        key=lambda observation: abs(
            int((latest.end - observation.end).days) - round(365.25 * int(years))
        ),
    )
    if prior.value <= 0:
        return None
    elapsed_years = float((latest.end - prior.end).days) / 365.25
    if elapsed_years <= 0:
        return None
    return float((latest.value / prior.value) ** (1.0 / elapsed_years) - 1.0)


def _latest_value(metrics: Mapping[str, MetricSeries], key: str) -> Optional[float]:
    observation = latest_observation(metrics.get(key))
    return observation.value if observation is not None else None


def _sum_available(*values: Optional[float]) -> Optional[float]:
    available = [float(value) for value in values if value is not None]
    return sum(available) if available else None


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    value = float(numerator) / float(denominator)
    return value if pd.notna(value) else None


def debt_value(metrics: Mapping[str, MetricSeries]) -> Optional[float]:
    """Return disclosed funded debt without double-counting current maturities."""
    total = _latest_value(metrics, "debt_total")
    short_term = _latest_value(metrics, "short_term_borrowings")
    if total is not None:
        return _sum_available(total, short_term)
    return _sum_available(
        _latest_value(metrics, "debt_current"),
        _latest_value(metrics, "debt_noncurrent"),
        short_term,
    )


def build_valuation_snapshot(
    metrics: Mapping[str, MetricSeries],
    *,
    price: Optional[float],
    price_date: Optional[pd.Timestamp] = None,
) -> ValuationSnapshot:
    """Calculate transparent current valuation and issuer-credit ratios."""
    shares = _latest_value(metrics, "shares_outstanding")
    diluted_shares = average_quarter_value(metrics.get("diluted_shares"))
    market_cap = (
        float(price) * shares if price is not None and shares is not None else None
    )
    cash = _latest_value(metrics, "cash")
    investments = _latest_value(metrics, "short_term_investments")
    liquid_assets = _sum_available(cash, investments)
    receivables = _latest_value(metrics, "receivables")
    assets = _latest_value(metrics, "assets")
    current_assets = _latest_value(metrics, "current_assets")
    current_liabilities = _latest_value(metrics, "current_liabilities")
    debt = debt_value(metrics)
    equity = _latest_value(metrics, "equity")
    preferred = _latest_value(metrics, "preferred_equity")
    minority = _latest_value(metrics, "minority_interest")
    enterprise_value = None
    if market_cap is not None:
        enterprise_value = (
            market_cap
            + (debt or 0.0)
            + (preferred or 0.0)
            + (minority or 0.0)
            - (liquid_assets or 0.0)
        )

    revenue = ltm_value(metrics.get("revenue"))
    gross_profit = ltm_value(metrics.get("gross_profit"))
    operating_income = ltm_value(metrics.get("operating_income"))
    net_income = ltm_value(metrics.get("net_income"))
    cfo = ltm_value(metrics.get("cfo"))
    capex = ltm_value(metrics.get("capex"))
    fcf = cfo - capex if cfo is not None and capex is not None else None
    da = ltm_value(metrics.get("da"))
    ebitda = (
        operating_income + da
        if operating_income is not None and da is not None
        else None
    )
    interest_expense = ltm_value(metrics.get("interest_expense"))
    dividends_raw = ltm_value(metrics.get("dividends_paid"))
    dividends = abs(dividends_raw) if dividends_raw is not None else None
    eps = ltm_value(metrics.get("eps_diluted"))
    if eps is None:
        eps = _safe_ratio(net_income, diluted_shares)
    pretax_income = ltm_value(metrics.get("pretax_income"))
    income_tax_expense = ltm_value(metrics.get("income_tax_expense"))
    effective_tax_rate = _safe_ratio(income_tax_expense, pretax_income)
    if effective_tax_rate is not None:
        effective_tax_rate = min(0.50, max(0.0, effective_tax_rate))
    nopat = (
        operating_income * (1.0 - effective_tax_rate)
        if operating_income is not None and effective_tax_rate is not None
        else None
    )
    invested_capital = (
        equity + (debt or 0.0) - (liquid_assets or 0.0)
        if equity is not None
        else None
    )
    average_assets = average_balance_value(metrics.get("assets"))
    average_equity = average_balance_value(metrics.get("equity"))
    quick_assets = (
        liquid_assets + receivables
        if liquid_assets is not None and receivables is not None
        else None
    )
    net_debt = debt - (liquid_assets or 0.0) if debt is not None else None

    return ValuationSnapshot(
        price=price,
        price_date=price_date,
        shares=shares,
        diluted_shares=diluted_shares,
        market_cap=market_cap,
        cash=cash,
        short_term_investments=investments,
        liquid_assets=liquid_assets,
        receivables=receivables,
        assets=assets,
        current_assets=current_assets,
        current_liabilities=current_liabilities,
        debt=debt,
        equity=equity,
        preferred_equity=preferred,
        minority_interest=minority,
        enterprise_value=enterprise_value,
        ltm_revenue=revenue,
        ltm_gross_profit=gross_profit,
        ltm_operating_income=operating_income,
        ltm_net_income=net_income,
        ltm_cfo=cfo,
        ltm_capex=capex,
        ltm_fcf=fcf,
        ltm_da=da,
        ltm_ebitda=ebitda,
        ltm_interest_expense=interest_expense,
        ltm_dividends=dividends,
        eps=eps,
        sales_per_share=_safe_ratio(revenue, diluted_shares),
        book_per_share=_safe_ratio(equity, shares),
        cash_per_share=_safe_ratio(liquid_assets, shares),
        pe=_safe_ratio(market_cap, net_income),
        price_sales=_safe_ratio(market_cap, revenue),
        price_book=_safe_ratio(market_cap, equity),
        price_cash=_safe_ratio(market_cap, liquid_assets),
        price_fcf=_safe_ratio(market_cap, fcf),
        ev_revenue=_safe_ratio(enterprise_value, revenue),
        ev_ebitda=_safe_ratio(enterprise_value, ebitda),
        fcf_yield=_safe_ratio(fcf, market_cap),
        gross_margin=_safe_ratio(gross_profit, revenue),
        operating_margin=_safe_ratio(operating_income, revenue),
        profit_margin=_safe_ratio(net_income, revenue),
        fcf_margin=_safe_ratio(fcf, revenue),
        current_ratio=_safe_ratio(current_assets, current_liabilities),
        quick_ratio=_safe_ratio(quick_assets, current_liabilities),
        debt_equity=_safe_ratio(debt, equity),
        roa=_safe_ratio(net_income, average_assets),
        roe=_safe_ratio(net_income, average_equity),
        roic=_safe_ratio(nopat, invested_capital),
        dividend_yield=_safe_ratio(dividends, market_cap),
        payout_ratio=_safe_ratio(dividends, net_income),
        debt_ebitda=_safe_ratio(debt, ebitda),
        net_debt_ebitda=_safe_ratio(net_debt, ebitda),
        interest_coverage=_safe_ratio(ebitda, interest_expense),
    )


def financial_table(
    metrics: Mapping[str, MetricSeries],
    keys: Sequence[str],
    *,
    frequency: str = "quarterly",
    periods: int = 8,
) -> pd.DataFrame:
    """Build an aligned filing table without forward filling observations."""
    series: dict[str, pd.Series] = {}
    for key in keys:
        metric = metrics.get(key)
        if metric is None:
            continue
        observations = metric.quarterly if frequency == "quarterly" else metric.annual
        if observations:
            series[metric.spec.label] = pd.Series(
                {observation.end: observation.value for observation in observations},
                dtype="float64",
            )
    if not series:
        return pd.DataFrame()
    frame = pd.DataFrame(series).sort_index().tail(periods)
    frame.index.name = "Period End"
    return frame.reset_index()


def balance_sheet_table(
    metrics: Mapping[str, MetricSeries], keys: Sequence[str], *, periods: int = 8
) -> pd.DataFrame:
    """Build an aligned point-in-time balance-sheet table."""
    series: dict[str, pd.Series] = {}
    for key in keys:
        metric = metrics.get(key)
        if metric is None or not metric.instant:
            continue
        series[metric.spec.label] = pd.Series(
            {observation.end: observation.value for observation in metric.instant},
            dtype="float64",
        )
    if not series:
        return pd.DataFrame()
    frame = pd.DataFrame(series).sort_index().tail(periods)
    frame.index.name = "Period End"
    return frame.reset_index()


def maturity_table(company_facts: Mapping[str, Any]) -> pd.DataFrame:
    """Extract a standardized debt maturity ladder when the issuer tags it."""
    rows: list[dict[str, Any]] = []
    for bucket, spec in MATURITY_SPECS.items():
        metric = extract_metric(company_facts, f"maturity_{bucket}", spec=spec)
        observation = latest_observation(metric)
        if observation is None:
            continue
        rows.append(
            {
                "Maturity Bucket": bucket,
                "Principal": observation.value,
                "Unit": observation.unit,
                "As Of": observation.end,
                "Filed": observation.filed,
                "Concept": observation.concept,
                "Source": observation.source_url,
            }
        )
    return pd.DataFrame(rows)


def filing_index_url(accession: str) -> str:
    """Build the SEC filing index URL from an accession number."""
    clean = str(accession).strip()
    if not clean:
        return ""
    digits = clean.replace("-", "")
    try:
        cik = int(digits[:10])
    except (TypeError, ValueError):
        return ""
    return f"{SEC_ARCHIVES_BASE}/{cik}/{digits}/{clean}-index.html"


def recent_filings(
    submissions: Mapping[str, Any],
    *,
    forms: Sequence[str] = ("10-K", "10-Q", "8-K", "20-F", "6-K"),
    limit: int = 30,
) -> pd.DataFrame:
    """Return recent filing metadata with canonical SEC document links."""
    recent = submissions.get("filings", {}).get("recent", {})
    if not isinstance(recent, Mapping):
        return pd.DataFrame()
    columns = {
        name: recent.get(name, [])
        for name in (
            "accessionNumber",
            "filingDate",
            "reportDate",
            "form",
            "primaryDocument",
            "primaryDocDescription",
        )
    }
    lengths = [len(value) for value in columns.values() if isinstance(value, list)]
    if not lengths:
        return pd.DataFrame()
    row_count = min(lengths)
    cik = int(submissions.get("cik", 0) or 0)
    rows: list[dict[str, Any]] = []
    accepted_forms = set(forms)
    for index in range(row_count):
        form = str(columns["form"][index])
        if form not in accepted_forms and form.removesuffix("/A") not in accepted_forms:
            continue
        accession = str(columns["accessionNumber"][index])
        primary_document = str(columns["primaryDocument"][index])
        digits = accession.replace("-", "")
        document_url = (
            f"{SEC_ARCHIVES_BASE}/{cik}/{digits}/{primary_document}"
            if cik and accession and primary_document
            else filing_index_url(accession)
        )
        rows.append(
            {
                "Filed": _as_timestamp(columns["filingDate"][index]),
                "Period": _as_timestamp(columns["reportDate"][index]),
                "Form": form,
                "Description": str(columns["primaryDocDescription"][index] or "").strip(),
                "Accession": accession,
                "Document": document_url,
                "Filing Index": filing_index_url(accession),
            }
        )
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def source_audit_table(metrics: Mapping[str, MetricSeries]) -> pd.DataFrame:
    """Show the latest reported source selected for every normalized metric."""
    rows: list[dict[str, Any]] = []
    for metric in metrics.values():
        observation = latest_observation(metric)
        if observation is None:
            continue
        rows.append(
            {
                "Metric": metric.spec.label,
                "Latest Reported": observation.value,
                "Unit": observation.unit,
                "Period End": observation.end,
                "Filed": observation.filed,
                "Form": observation.form,
                "Taxonomy": observation.taxonomy,
                "Concept": observation.concept,
                "Derived Quarter": observation.derived,
                "Source": observation.source_url,
            }
        )
    return pd.DataFrame(rows).sort_values("Metric").reset_index(drop=True)


def latest_quarter_growth(metric: Optional[MetricSeries]) -> Optional[float]:
    """Latest reported-quarter year-over-year growth."""
    if metric is None or len(metric.quarterly) < 5:
        return None
    latest = metric.quarterly[-1]
    prior_candidates = [
        observation
        for observation in metric.quarterly[:-1]
        if 330 <= int((latest.end - observation.end).days) <= 400
    ]
    if not prior_candidates:
        return None
    prior = prior_candidates[-1]
    return _safe_ratio(latest.value - prior.value, abs(prior.value))


def period_label(value: Optional[pd.Timestamp]) -> str:
    """Stable date label for Streamlit tables and source notes."""
    return value.date().isoformat() if value is not None else "Unavailable"


def today_label() -> str:
    """Current date label isolated for deterministic UI composition."""
    return date.today().isoformat()
