"""Preserved from the original Cross-Asset Ratio Chartbook; no UI runtime."""
import re
from dataclasses import dataclass
from typing import Dict, List, Iterable
from adfm_engine.palette import PASTEL

DEFAULT_LOOKBACK = "3 Years"


DEFAULT_RSI_WINDOW = 14


DEFAULT_STALE_DAYS = 7


MA_DEFAULTS = {
    8: False,
    21: True,
    50: True,
    100: False,
    200: True,
}


MA_COLORS = {
    8: "#6c757d",
    21: PASTEL["lavender"],
    50: PASTEL["blue"],
    100: PASTEL["coral"],
    200: PASTEL["rose"],
}


@dataclass(frozen=True)
class RatioSpec:
    ticker_1: str
    ticker_2: str
    label: str
    note: str = ""


CORE_RATIO_SPECS: List[RatioSpec] = [
    RatioSpec("TLT", "SHY", "Long Treasuries / Short Treasuries", "Long-duration demand versus the front of the Treasury curve."),
    RatioSpec("IEF", "SHY", "Intermediate Treasuries / Short Treasuries", "Intermediate-duration demand versus short Treasuries."),
    RatioSpec("EDV", "SHY", "Extended Duration Treasuries / Short Treasuries", "Maximum-duration Treasury convexity versus the short end."),
    RatioSpec("TLT", "SPY", "Long Treasuries / S&P 500", "Duration hedge performance versus equity risk."),
    RatioSpec("GLD", "SPY", "Gold / S&P 500", "Monetary and geopolitical hedge performance versus equity risk."),
    RatioSpec("GLD", "TLT", "Gold / Long Treasuries", "Monetary and fiscal defense versus sovereign duration."),
    RatioSpec("UUP", "SPY", "Dollar / S&P 500", "Dollar defensiveness and liquidity pressure versus equity risk."),
    RatioSpec("FXY", "SPY", "Yen / S&P 500", "Yen strength and carry-unwind pressure versus equity risk."),
    RatioSpec("BIL", "SPY", "Treasury Bills / S&P 500", "Cash-like Treasury safety versus equity risk."),
    RatioSpec("^VIX", "^VIX3M", "Spot VIX / 3-Month VIX", "Volatility-term-structure stress; readings above one indicate inversion."),

    RatioSpec("DBC", "SPY", "Broad Commodities / S&P 500", "Broad commodity leadership versus U.S. equities."),
    RatioSpec("DBC", "QQQ", "Broad Commodities / Nasdaq 100", "Broad commodity leadership versus long-duration growth equities."),
    RatioSpec("GLD", "QQQ", "Gold / Nasdaq 100", "Gold and monetary defense versus long-duration growth equities."),
    RatioSpec("SLV", "SPY", "Silver / S&P 500", "Silver's monetary and industrial beta versus U.S. equities."),
    RatioSpec("CPER", "SPY", "Copper / S&P 500", "Copper's global-growth signal versus U.S. equities."),
    RatioSpec("DBB", "SPY", "Industrial Metals / S&P 500", "Industrial-metals leadership versus U.S. equities."),
    RatioSpec("USO", "SPY", "WTI Oil / S&P 500", "Crude-oil inflation and scarcity pressure versus U.S. equities."),
    RatioSpec("BNO", "SPY", "Brent Oil / S&P 500", "Global crude-oil leadership versus U.S. equities."),
    RatioSpec("UNG", "SPY", "Natural Gas / S&P 500", "Natural-gas scarcity and weather beta versus U.S. equities."),
    RatioSpec("DBA", "SPY", "Agriculture / S&P 500", "Agricultural inflation and crop stress versus U.S. equities."),

    RatioSpec("HYG", "IEF", "High Yield / Intermediate Treasuries", "Credit-risk appetite versus intermediate-duration government bonds."),
    RatioSpec("HYG", "LQD", "High Yield / Investment Grade", "Lower-quality credit performance versus investment-grade credit."),
    RatioSpec("LQD", "IEF", "Investment Grade / Intermediate Treasuries", "Corporate credit performance versus duration-matched government bonds."),
    RatioSpec("BKLN", "IEF", "Senior Loans / Intermediate Treasuries", "Floating-rate credit risk versus intermediate duration."),
    RatioSpec("JAAA", "SHY", "AAA CLOs / Short Treasuries", "Top-of-stack structured credit versus short government bonds."),
    RatioSpec("JBBB", "SHY", "BBB CLOs / Short Treasuries", "Lower-rated structured credit and funding sensitivity versus short government bonds."),
    RatioSpec("EMB", "IEF", "Emerging-Market Dollar Debt / Intermediate Treasuries", "External sovereign credit and dollar-liquidity risk versus Treasuries."),
    RatioSpec("PFF", "IEF", "Preferred Securities / Intermediate Treasuries", "Bank-capital and hybrid-credit performance versus government duration."),
    RatioSpec("MBB", "IEF", "Agency MBS / Intermediate Treasuries", "Mortgage-basis and convexity performance versus intermediate Treasuries."),

    RatioSpec("KRE", "SPY", "Regional Banks / S&P 500", "Regional-bank funding and credit-cycle performance versus the broad market."),
    RatioSpec("KBE", "SPY", "U.S. Banks / S&P 500", "Broad bank equity performance versus the broad market."),
    RatioSpec("KBWB", "SPY", "Money-Center Banks / S&P 500", "Large-bank balance-sheet and capital-markets performance versus the broad market."),
    RatioSpec("BIZD", "SPY", "Business Development Companies / S&P 500", "BDC and private-credit equity performance versus the broad market."),
    RatioSpec("PBDC", "SPY", "Active BDC Basket / S&P 500", "Actively selected BDC exposure versus the broad market."),
    RatioSpec("PSP", "SPY", "Listed Private Equity / S&P 500", "Alternative-asset-manager and listed private-equity performance versus the broad market."),
    RatioSpec("REM", "SPY", "Mortgage REITs / S&P 500", "Levered mortgage-finance and funding sensitivity versus the broad market."),
    RatioSpec("KIE", "SPY", "Insurance / S&P 500", "Insurance underwriting and investment-income performance versus the broad market."),
    RatioSpec("KCE", "SPY", "Capital Markets / S&P 500", "Brokerage, exchange, and capital-markets activity versus the broad market."),
]


RATIO_FAMILIES: Dict[str, List[RatioSpec]] = {
    "Duration / Crisis Hedges": CORE_RATIO_SPECS[0:10],
    "Commodities / Equity Indices": CORE_RATIO_SPECS[10:20],
    "Credit / Funding": CORE_RATIO_SPECS[20:29],
    "Financial Intermediaries": CORE_RATIO_SPECS[29:38],
}


def clean_ticker(ticker: str) -> str:
    return str(ticker).strip().upper()


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []

    for item in items:
        item = clean_ticker(item)

        if item and item not in seen:
            seen.add(item)
            out.append(item)

    return out


def parse_custom_ratio_text(text: str) -> List[RatioSpec]:
    if not text or not text.strip():
        return []

    raw_parts = re.split(r"[\n,;]+", text)
    specs: List[RatioSpec] = []

    for part in raw_parts:
        part = part.strip().upper()

        if not part:
            continue

        if "/" in part:
            pieces = [p.strip() for p in part.split("/") if p.strip()]
        else:
            pieces = [p.strip() for p in re.split(r"\s+", part) if p.strip()]

        if len(pieces) < 2:
            continue

        a = clean_ticker(pieces[0])
        b = clean_ticker(pieces[1])

        if a and b and a != b:
            specs.append(RatioSpec(a, b, f"{a} / {b}"))

    deduped = []
    seen = set()

    for spec in specs:
        key = (spec.ticker_1, spec.ticker_2)

        if key not in seen:
            seen.add(key)
            deduped.append(spec)

    return deduped


def make_display_title(spec: RatioSpec) -> str:
    return f"{spec.label} ({spec.ticker_1}/{spec.ticker_2})"


