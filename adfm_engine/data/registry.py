"""Typed registry for market proxies and primary macro series.

The registry gives every analytical input one stable key, source description,
freshness policy, and constructive direction. Pages can therefore disclose
what they are using without duplicating provider metadata or silently changing
the meaning of a signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class SeriesDefinition:
    """Metadata for one market or macro input."""

    key: str
    label: str
    symbol: str
    provider: str
    group: str
    description: str
    max_stale_sessions: int = 3


@dataclass(frozen=True)
class ProxyDefinition:
    """A constructive market signal derived from one price or a ratio."""

    key: str
    label: str
    group: str
    numerator: str
    denominator: str | None
    direction: int
    description: str


MARKET_SERIES: Final[tuple[SeriesDefinition, ...]] = (
    SeriesDefinition(
        "spy",
        "S&P 500",
        "SPY",
        "Yahoo Finance",
        "Equity Tape",
        "US large-cap equity beta.",
    ),
    SeriesDefinition(
        "rsp",
        "S&P 500 Equal Weight",
        "RSP",
        "Yahoo Finance",
        "Equity Tape",
        "Equal-weight participation and breadth.",
    ),
    SeriesDefinition(
        "xly",
        "Consumer Discretionary",
        "XLY",
        "Yahoo Finance",
        "Equity Tape",
        "Cyclical consumer equity leadership.",
    ),
    SeriesDefinition(
        "xlp",
        "Consumer Staples",
        "XLP",
        "Yahoo Finance",
        "Equity Tape",
        "Defensive consumer equity leadership.",
    ),
    SeriesDefinition(
        "smh",
        "Semiconductors",
        "SMH",
        "Yahoo Finance",
        "Equity Tape",
        "Semiconductor and AI-cycle leadership.",
    ),
    SeriesDefinition(
        "hyg",
        "High Yield Credit",
        "HYG",
        "Yahoo Finance",
        "Credit",
        "Liquid high-yield credit proxy.",
    ),
    SeriesDefinition(
        "lqd",
        "Investment Grade Credit",
        "LQD",
        "Yahoo Finance",
        "Credit",
        "Liquid investment-grade credit proxy.",
    ),
    SeriesDefinition(
        "tlt",
        "Long Treasuries",
        "TLT",
        "Yahoo Finance",
        "Rates",
        "Long-duration Treasury price proxy.",
    ),
    SeriesDefinition(
        "shy",
        "Treasury Bills",
        "SHY",
        "Yahoo Finance",
        "Rates",
        "Short-duration Treasury price proxy.",
    ),
    SeriesDefinition(
        "uup",
        "US Dollar",
        "UUP",
        "Yahoo Finance",
        "Dollar Liquidity",
        "Broad dollar price proxy.",
    ),
    SeriesDefinition(
        "dbc",
        "Broad Commodities",
        "DBC",
        "Yahoo Finance",
        "Commodities",
        "Broad commodity price proxy.",
    ),
    SeriesDefinition(
        "gld", "Gold", "GLD", "Yahoo Finance", "Commodities", "Gold price proxy."
    ),
    SeriesDefinition(
        "uso",
        "Crude Oil",
        "USO",
        "Yahoo Finance",
        "Commodities",
        "Crude-oil price proxy.",
    ),
    SeriesDefinition(
        "vix",
        "VIX",
        "^VIX",
        "Yahoo Finance",
        "Volatility",
        "S&P 500 implied-volatility index.",
    ),
    SeriesDefinition(
        "eem",
        "Emerging Markets",
        "EEM",
        "Yahoo Finance",
        "Global Risk",
        "Emerging-market equity beta.",
    ),
    SeriesDefinition(
        "fxi",
        "China Large Caps",
        "FXI",
        "Yahoo Finance",
        "Global Risk",
        "China large-cap equity proxy.",
    ),
)


PRIMARY_MACRO_SERIES: Final[tuple[SeriesDefinition, ...]] = (
    SeriesDefinition(
        "dgs2",
        "US 2-Year Treasury",
        "DGS2",
        "Federal Reserve FRED",
        "Rates",
        "Constant-maturity 2-year Treasury yield.",
        5,
    ),
    SeriesDefinition(
        "dgs10",
        "US 10-Year Treasury",
        "DGS10",
        "Federal Reserve FRED",
        "Rates",
        "Constant-maturity 10-year Treasury yield.",
        5,
    ),
    SeriesDefinition(
        "dgs30",
        "US 30-Year Treasury",
        "DGS30",
        "Federal Reserve FRED",
        "Rates",
        "Constant-maturity 30-year Treasury yield.",
        5,
    ),
    SeriesDefinition(
        "dfii10",
        "US 10-Year Real Yield",
        "DFII10",
        "Federal Reserve FRED",
        "Rates",
        "Constant-maturity 10-year TIPS yield.",
        5,
    ),
    SeriesDefinition(
        "t10yie",
        "US 10-Year Breakeven",
        "T10YIE",
        "Federal Reserve FRED",
        "Inflation",
        "Market-implied 10-year inflation compensation.",
        5,
    ),
    SeriesDefinition(
        "hy_oas",
        "US High Yield OAS",
        "BAMLH0A0HYM2",
        "Federal Reserve FRED",
        "Credit",
        "ICE BofA US High Yield option-adjusted spread.",
        5,
    ),
    SeriesDefinition(
        "walcl",
        "Federal Reserve Assets",
        "WALCL",
        "Federal Reserve FRED",
        "Liquidity",
        "Federal Reserve total assets.",
        10,
    ),
    SeriesDefinition(
        "tga",
        "Treasury General Account",
        "WTREGEN",
        "Federal Reserve FRED",
        "Liquidity",
        "US Treasury General Account balance.",
        10,
    ),
    SeriesDefinition(
        "rrp",
        "Overnight Reverse Repo",
        "RRPONTSYD",
        "Federal Reserve FRED",
        "Liquidity",
        "Overnight reverse-repurchase facility usage.",
        5,
    ),
)


COCKPIT_PROXIES: Final[tuple[ProxyDefinition, ...]] = (
    ProxyDefinition(
        "broad_equity",
        "Broad equity tape",
        "Equity Tape",
        "SPY",
        None,
        1,
        "Absolute US equity trend.",
    ),
    ProxyDefinition(
        "breadth",
        "Equal-weight breadth",
        "Equity Tape",
        "RSP",
        "SPY",
        1,
        "Equal-weight leadership versus cap-weighted equities.",
    ),
    ProxyDefinition(
        "cyclicals",
        "Cyclical leadership",
        "Equity Tape",
        "XLY",
        "XLP",
        1,
        "Discretionary leadership versus staples.",
    ),
    ProxyDefinition(
        "semis",
        "Semiconductor leadership",
        "Equity Tape",
        "SMH",
        "SPY",
        1,
        "Semiconductor leadership versus the S&P 500.",
    ),
    ProxyDefinition(
        "credit",
        "Credit sponsorship",
        "Credit",
        "HYG",
        "LQD",
        1,
        "High-yield performance versus investment grade.",
    ),
    ProxyDefinition(
        "duration",
        "Duration bid",
        "Rates",
        "TLT",
        "SHY",
        1,
        "Long-duration Treasuries versus short duration.",
    ),
    ProxyDefinition(
        "dollar",
        "Dollar liquidity",
        "Dollar Liquidity",
        "UUP",
        None,
        -1,
        "A softer dollar is treated as easier global liquidity.",
    ),
    ProxyDefinition(
        "commodities",
        "Commodity impulse",
        "Commodities",
        "DBC",
        None,
        1,
        "Broad commodity price impulse.",
    ),
    ProxyDefinition(
        "crude_relief",
        "Crude disinflation relief",
        "Commodities",
        "USO",
        None,
        -1,
        "Lower crude prices are treated as disinflation relief.",
    ),
    ProxyDefinition(
        "volatility",
        "Volatility relief",
        "Volatility",
        "^VIX",
        None,
        -1,
        "Lower implied volatility is constructive.",
    ),
    ProxyDefinition(
        "em_risk",
        "Emerging-market risk",
        "Global Risk",
        "EEM",
        "SPY",
        1,
        "Emerging-market equities versus US equities.",
    ),
    ProxyDefinition(
        "china_risk",
        "China risk appetite",
        "Global Risk",
        "FXI",
        "SPY",
        1,
        "China large caps versus US equities.",
    ),
)


def market_symbols() -> tuple[str, ...]:
    """Return the de-duplicated ticker set required by the PM cockpit."""

    return tuple(dict.fromkeys(item.symbol for item in MARKET_SERIES))


def registry_by_key() -> dict[str, SeriesDefinition]:
    """Return all registered series keyed by their stable identifier."""

    return {item.key: item for item in MARKET_SERIES + PRIMARY_MACRO_SERIES}
