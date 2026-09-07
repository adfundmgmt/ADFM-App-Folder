from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from adfm_engine.cache import ttl_cache
from adfm_engine.palette import PASTEL
@dataclass(frozen=True)
class RatioSpec:
    ticker_1: str
    ticker_2: str
    label: str
    note: str

    @property
    def key(self) -> str:
        return f"{self.ticker_1}/{self.ticker_2}"

LEADERSHIP_FAMILIES: Dict[str, List[RatioSpec]] = {
    "S&P 500 Sector Leadership": [
        RatioSpec("XLK", "SPY", "Technology / S&P 500", "Technology sector leadership versus the broad market."),
        RatioSpec("XLC", "SPY", "Communication Services / S&P 500", "Communication-services leadership versus the broad market."),
        RatioSpec("XLY", "SPY", "Consumer Discretionary / S&P 500", "Consumer-cycle leadership versus the broad market."),
        RatioSpec("XLF", "SPY", "Financials / S&P 500", "Financial-sector leadership and balance-sheet sensitivity versus the broad market."),
        RatioSpec("XLI", "SPY", "Industrials / S&P 500", "Industrial, capex, and cyclical-growth leadership versus the broad market."),
        RatioSpec("XLV", "SPY", "Health Care / S&P 500", "Health-care leadership versus the broad market."),
        RatioSpec("XLP", "SPY", "Consumer Staples / S&P 500", "Defensive-consumption leadership versus the broad market."),
        RatioSpec("XLE", "SPY", "Energy / S&P 500", "Energy and inflation-beta leadership versus the broad market."),
        RatioSpec("XLB", "SPY", "Materials / S&P 500", "Materials and upstream cyclicality versus the broad market."),
        RatioSpec("XLU", "SPY", "Utilities / S&P 500", "Defensive-duration and power-sector leadership versus the broad market."),
        RatioSpec("XLRE", "SPY", "Real Estate / S&P 500", "Rate-sensitive real-estate leadership versus the broad market."),
    ],
    "China / U.S. Leadership": [
        RatioSpec("FXI", "SPY", "China Large Caps / S&P 500", "China policy and growth beta versus U.S. equity leadership."),
        RatioSpec("MCHI", "SPY", "Broad China / S&P 500", "Broad investable Chinese equities versus U.S. equity leadership."),
        RatioSpec("KWEB", "QQQ", "China Internet / Nasdaq 100", "Chinese internet and platform companies versus U.S. large-cap technology."),
        RatioSpec("CQQQ", "QQQ", "China Technology / Nasdaq 100", "Chinese technology leadership versus the U.S. technology benchmark."),
        RatioSpec("ASHR", "SPY", "China A-Shares / S&P 500", "Mainland-listed China exposure and domestic policy transmission versus U.S. equities."),
    ],
    "Breadth / Alternative Weighting": [
        RatioSpec("QQQE", "QQQ", "Equal Weight Nasdaq 100 / Nasdaq 100", "Nasdaq breadth versus mega-cap concentration."),
        RatioSpec("RSP", "SPY", "Equal Weight S&P 500 / S&P 500", "Median-stock participation versus capitalization-weighted leadership."),
        RatioSpec("RWJ", "IJR", "Revenue Weight Small Caps / Cap Weight Small Caps", "A live small-cap alternative-weighting proxy using the same S&P 600 universe."),
    ],
    "Inter-Sector Leadership": [
        RatioSpec("SMH", "IGV", "Semiconductors / Software", "AI hardware and compute leadership versus software and the application layer."),
        RatioSpec("XLF", "XLK", "Financials / Technology", "Nominal-growth and curve-sensitive leadership versus long-duration technology."),
        RatioSpec("XLI", "XLU", "Industrials / Utilities", "Cyclical growth and capex leadership versus defensive duration."),
        RatioSpec("XLY", "XLP", "Consumer Discretionary / Staples", "Consumer-cycle leadership versus defensive consumption."),
        RatioSpec("XLE", "XLK", "Energy / Technology", "Hard-asset and inflation beta versus long-duration growth."),
        RatioSpec("KRE", "XLF", "Regional Banks / Financials", "Regional-bank funding and credit sensitivity versus diversified financials."),
    ],
}

ALL_SPECS = [spec for specs in LEADERSHIP_FAMILIES.values() for spec in specs]

SPEC_BY_KEY = {spec.key: spec for spec in ALL_SPECS}

FAMILY_BY_KEY = {spec.key: family for family, specs in LEADERSHIP_FAMILIES.items() for spec in specs}

def raw_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    first, second = numerator.align(denominator, join="inner")
    return (first / second).replace([np.inf, -np.inf], np.nan).dropna()
