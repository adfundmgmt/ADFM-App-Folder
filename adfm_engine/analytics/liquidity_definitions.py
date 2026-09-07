"""Original liquidity source definitions, weights and thresholds."""
from __future__ import annotations
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
TITLE = "Liquidity Conditions Monitor"


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


