"""Original credit and sovereign universes and metadata."""
from __future__ import annotations
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from adfm_engine.palette import PASTEL, PASTEL_20
from adfm_engine.data.registry import PRIMARY_MACRO_SERIES, SeriesDefinition

pd.options.mode.chained_assignment = None


TITLE = "Credit Conditions Monitor"


SUBTITLE = (
    "Separates credit-spread stress from outright funding-cost pressure, then "
    "checks banks, loans, volatility, and global sovereign yields for confirmation."
)


COLORS = {
    "green": PASTEL["sage"],
    "red": PASTEL["rose"],
    "orange": PASTEL["coral"],
    "amber": PASTEL["amber"],
    "blue": PASTEL["blue"],
    "purple": PASTEL["lavender"],
    "teal": PASTEL["teal"],
    "grey": "#A8ADB5",
    "slate": "#334155",
    "muted": "#64748b",
    "grid": "#e5e7eb",
    "dark": "#111111",
}


LINE_COLORS = list(PASTEL_20)


SOVEREIGN_UP = "#F28E2B"


SOVEREIGN_DOWN = "#4E79A7"


SOVEREIGN_FLAT = "#9CA3AF"


SOVEREIGN_MEDIAN = "#111827"


SOVEREIGN_ZERO = "#CBD5E1"


MARKET_TICKERS: Tuple[str, ...] = (
    "HYG",
    "JNK",
    "LQD",
    "BKLN",
    "SRLN",
    "EMB",
    "KRE",
    "XLF",
    "SPY",
    "IWM",
    "TLT",
    "IEF",
    "UUP",
    "^VIX",
)


DISPLAY_NAMES: Dict[str, str] = {
    "HYG": "High Yield",
    "JNK": "High Yield 2",
    "LQD": "Investment Grade",
    "BKLN": "Leveraged Loans",
    "SRLN": "Senior Loans",
    "EMB": "EM USD Debt",
    "KRE": "Regional Banks",
    "XLF": "Financials",
    "SPY": "S&P 500",
    "IWM": "Russell 2000",
    "TLT": "20Y+ Treasuries",
    "IEF": "7-10Y Treasuries",
    "UUP": "U.S. Dollar",
    "^VIX": "VIX",
}


FOCUS_WINDOWS = ["5D", "1M", "3M", "YTD", "1Y"]


GLOBAL_WINDOWS = ["5D", "1M", "YTD", "1Y", "3Y", "5Y"]


PRIMARY_BY_KEY = {definition.key: definition for definition in PRIMARY_MACRO_SERIES}


CREDIT_FRED_DEFINITIONS: Tuple[SeriesDefinition, ...] = (
    PRIMARY_BY_KEY["hy_oas"],
    PRIMARY_BY_KEY["dgs10"],
    PRIMARY_BY_KEY["dgs30"],
    SeriesDefinition(
        "ig_oas",
        "US Corporate OAS",
        "BAMLC0A0CM",
        "Federal Reserve FRED",
        "Credit",
        "ICE BofA US Corporate option-adjusted spread.",
        5,
    ),
    SeriesDefinition(
        "bbb_oas",
        "US BBB OAS",
        "BAMLC0A4CBBB",
        "Federal Reserve FRED",
        "Credit",
        "ICE BofA BBB US Corporate option-adjusted spread.",
        5,
    ),
)


SOVEREIGN_UNIVERSE: Tuple[dict, ...] = (
    {"country": "United States", "label": "U.S.", "group": "Developed", "stooq": "10YUSY.B", "fred": "IRLTLT01USM156N"},
    {"country": "Japan", "label": "Japan", "group": "Developed", "stooq": "10YJPY.B", "fred": "IRLTLT01JPM156N"},
    {"country": "Australia", "label": "Australia", "group": "Developed", "stooq": "10YAUY.B", "fred": "IRLTLT01AUM156N"},
    {"country": "Canada", "label": "Canada", "group": "Developed", "stooq": "10YCAY.B", "fred": "IRLTLT01CAM156N"},
    {"country": "Germany", "label": "Germany", "group": "Developed", "stooq": "10YDEY.B", "fred": "IRLTLT01DEM156N"},
    {"country": "France", "label": "France", "group": "Developed", "stooq": "10YFRY.B", "fred": "IRLTLT01FRM156N"},
    {"country": "Italy", "label": "Italy", "group": "Developed", "stooq": "10YITY.B", "fred": "IRLTLT01ITM156N"},
    {"country": "Spain", "label": "Spain", "group": "Developed", "stooq": "10YESY.B", "fred": "IRLTLT01ESM156N"},
    {"country": "United Kingdom", "label": "U.K.", "group": "Developed", "stooq": "10YGBY.B", "fred": "IRLTLT01GBM156N"},
    {"country": "Switzerland", "label": "Switzerland", "group": "Developed", "stooq": "10YCHY.B", "fred": "IRLTLT01CHM156N"},
    {"country": "Netherlands", "label": "Netherlands", "group": "Developed", "stooq": "10YNLY.B", "fred": "IRLTLT01NLM156N"},
    {"country": "Belgium", "label": "Belgium", "group": "Developed", "stooq": "10YBEY.B", "fred": "IRLTLT01BEM156N"},
    {"country": "Portugal", "label": "Portugal", "group": "Developed", "stooq": "10YPTY.B", "fred": "IRLTLT01PTM156N"},
    {"country": "Sweden", "label": "Sweden", "group": "Developed", "stooq": "10YSEY.B", "fred": "IRLTLT01SEM156N"},
    {"country": "Norway", "label": "Norway", "group": "Developed", "stooq": "10YNOY.B", "fred": "IRLTLT01NOM156N"},
    {"country": "New Zealand", "label": "New Zealand", "group": "Developed", "stooq": "10YNZY.B", "fred": "IRLTLT01NZM156N"},
    {"country": "Turkey", "label": "Turkey", "group": "Emerging", "stooq": "10YTRY.B", "fred": "IRLTLT01TRM156N"},
    {"country": "South Korea", "label": "South Korea", "group": "Emerging", "stooq": "10YKRY.B", "fred": "IRLTLT01KRM156N"},
    {"country": "Poland", "label": "Poland", "group": "Emerging", "stooq": "10YPLY.B", "fred": "IRLTLT01PLM156N"},
    {"country": "Czechia", "label": "Czechia", "group": "Emerging", "stooq": "10YCZY.B", "fred": "IRLTLT01CZM156N"},
    {"country": "Hungary", "label": "Hungary", "group": "Emerging", "stooq": "10YHUY.B", "fred": "IRLTLT01HUM156N"},
    {"country": "Brazil", "label": "Brazil", "group": "Emerging", "stooq": "10YBRY.B", "fred": "IRLTLT01BRM156N"},
    {"country": "Mexico", "label": "Mexico", "group": "Emerging", "stooq": "10YMXY.B", "fred": "IRLTLT01MXM156N"},
    {"country": "India", "label": "India", "group": "Emerging", "stooq": "10YINY.B", "fred": "IRLTLT01INM156N"},
    {"country": "China", "label": "China", "group": "Emerging", "stooq": "10YCNY.B", "fred": "IRLTLT01CNM156N"},
    {"country": "Indonesia", "label": "Indonesia", "group": "Emerging", "stooq": "10YIDY.B", "fred": "IRLTLT01IDM156N"},
    {"country": "Malaysia", "label": "Malaysia", "group": "Emerging", "stooq": "10YMYY.B", "fred": "IRLTLT01MYM156N"},
    {"country": "South Africa", "label": "South Africa", "group": "Emerging", "stooq": "10YZAY.B", "fred": "IRLTLT01ZAM156N"},
    {"country": "Colombia", "label": "Colombia", "group": "Emerging", "stooq": "10YCOY.B", "fred": "IRLTLT01COM156N"},
    {"country": "Chile", "label": "Chile", "group": "Emerging", "stooq": "10YCLY.B", "fred": "IRLTLT01CLM156N"},
)


TE_SLUG_OVERRIDES = {
    "United States": "united-states",
    "United Kingdom": "united-kingdom",
    "South Korea": "south-korea",
    "New Zealand": "new-zealand",
    "South Africa": "south-africa",
    "Czechia": "czech-republic",
}


HISTORY_DAYS = {
    "1 Year": 365,
    "3 Years": 365 * 3,
    "5 Years": 365 * 5,
    "10 Years": 365 * 10,
}


