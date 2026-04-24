import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time


# ============================================================
# Page and theme
# ============================================================
st.set_page_config(page_title="Basket Panels", layout="wide")

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
    h1, h2, h3 {font-weight: 600; letter-spacing: 0.15px;}
    .stPlotlyChart {background: #ffffff;}
    .sidebar-content {padding-top: 0.5rem;}
    .js-plotly-plot .table .cell {font-size: 12px;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

TITLE = "ADFM Basket Panels"
SUBTITLE = "Consolidated panel for all baskets, plus per-category panels with aligned business-day dates."

PASTEL = [
    "#AEC6CF", "#FFB347", "#B39EB5", "#77DD77", "#F49AC2",
    "#CFCFC4", "#DEA5A4", "#C6E2FF", "#FFDAC1", "#E2F0CB",
    "#C7CEEA", "#FFB3BA", "#FFD1DC", "#B5EAD7", "#E7E6F7",
    "#F1E3DD", "#B0E0E6", "#E0BBE4", "#F3E5AB", "#D5E8D4"
]

MIN_MARKET_CAP = 1_000_000_000
INDICATOR_WARMUP_DAYS = 520
BENCH = "SPY"

CACHE_DIR = Path(".adfm_cache")
LEVELS_CACHE = CACHE_DIR / "basket_levels_last_good.pkl"
LEVELS_META_CACHE = CACHE_DIR / "basket_levels_last_good_meta.json"

MARKET_CAP_EXCEPTIONS: set[str] = set()

# If Yahoo does not return a market cap, the ticker is kept.
# ETFs, ADRs, and foreign listings often have missing market-cap fields.
EXCLUDE_MISSING_MARKET_CAP = False


# ============================================================
# Category -> Baskets -> Tickers
# ============================================================
CATEGORIES: Dict[str, Dict[str, List[str]]] = {
    "Growth & Innovation": {
        "Semis ETFs": ["SMH", "SOXX", "XSD"],
        "Semis Compute and Accelerators": ["NVDA", "AMD", "AVGO", "MRVL", "ARM", "INTC"],
        "Semis Analog and Power": ["TXN", "ADI", "MCHP", "NXPI", "MPWR", "ON", "STM", "IFNNY"],
        "Semis RF and Connectivity": ["QCOM", "SWKS", "QRVO", "MTSI", "AVNW"],
        "Semis Memory and Storage": ["MU", "WDC", "STX", "SNDK", "SKM"],
        "Foundry Packaging and OSAT": ["TSM", "UMC", "GFS", "ASX", "AMKR"],
        "Semis Equipment": ["ASML", "AMAT", "LRCX", "KLAC", "TER", "ONTO", "AEIS", "ACMR"],
        "Semis EDA and IP": ["SNPS", "CDNS", "ARM"],
        "Semis Automotive and Industrial End-Markets": ["NXPI", "ON", "MCHP", "STM", "QCOM", "ADI"],
        "Semis Test Assembly and Packaging": ["AMKR", "TER", "ONTO", "AEIS", "KLAC"],
        "AI Infrastructure Supply Chain": [
            "NVDA", "AMD", "AVGO", "TSM", "ASML",
            "ANET", "MRVL", "MU",
            "AMAT", "LRCX", "KLAC", "TER",
            "VRT", "ETN", "SMCI", "DELL"
        ],
        "Hyperscalers": ["MSFT", "AMZN", "GOOGL", "META", "ORCL"],
        "Enterprise Software and Cloud": [
            "NOW", "CRM", "DDOG", "SNOW", "MDB", "NET", "ZS", "OKTA", "TEAM", "HUBS", "INTU", "ADBE"
        ],
        "Usage-Based and Consumption Software": ["SNOW", "DDOG", "MDB", "NET", "CRWD"],
        "Developer Tools and DevOps": ["DT", "GTLB", "MDB", "DDOG", "NET", "TEAM"],
        "Observability and Data Tooling": ["DDOG", "ESTC", "MDB", "DT"],
        "Quality SaaS": ["ADBE", "CRM", "NOW", "INTU", "TEAM", "HUBS", "DDOG", "NET", "MDB", "SNOW"],
        "Cybersecurity": ["PANW", "FTNT", "CRWD", "ZS", "OKTA", "TENB", "S", "CYBR", "CHKP", "NET"],
        "Digital Payments": ["V", "MA", "PYPL", "XYZ", "FI", "FIS", "GPN", "AXP", "COF", "DFS", "MELI"],
        "E-Commerce Platforms": ["AMZN", "SHOP", "MELI", "ETSY", "PDD", "BABA", "JD", "SE"],
        "Social and Consumer Internet": ["META", "SNAP", "PINS", "MTCH", "GOOGL", "BILI", "BIDU", "RBLX"],
        "Streaming and Media": ["NFLX", "DIS", "WBD", "PARA", "ROKU", "SPOT", "LYV", "CHTR", "CMCSA"],
        "Fintech and Neobanks": ["XYZ", "PYPL", "AFRM", "HOOD", "SOFI", "UPST"],
    },

    "AI and Data Center Stack": {
        "Data Center Networking": ["ANET", "CSCO", "JNPR", "MRVL", "AVGO", "CIEN", "LITE", "INFN"],
        "Optical and Coherent": ["CIEN", "LITE", "INFN", "COHR", "AAOI"],
        "Data Center Power and Cooling": ["VRT", "ETN", "TT", "JCI", "CARR", "ABB", "POWL", "GEV"],
        "Data Center REITs": ["EQIX", "DLR", "AMT", "SBAC", "CCI"],
        "Servers": ["SMCI", "DELL", "HPE"],
        "Storage and Data Infrastructure": ["NTAP", "WDC", "STX", "IBM", "SNDK"],
        "IT Services and Systems Integrators": ["ACN", "IBM", "CTSH", "EPAM", "GIB"],
        "Rack Power Delivery and Electrical Gear": ["ETN", "VRT", "ABB", "POWL", "JCI", "GEV"],
        "AI Data Center Capex Beneficiaries": ["ANET", "AVGO", "MRVL", "ETN", "VRT", "GEV", "PWR", "SMCI", "DELL", "EQIX"],
    },

    "Connectivity and Industrial Tech": {
        "5G and Networking Infra": ["AMT", "CCI", "SBAC", "ANET", "CSCO", "JNPR", "ERIC", "NOK", "FFIV"],
        "Industrial Automation": ["ROK", "ETN", "EMR", "AME", "PH", "ABB", "KEYS", "TRMB", "CGNX", "IEX", "ITW", "GWW", "SYM"],
        "Aerospace Tech and Space": ["RKLB", "IRDM", "ASTS", "LHX", "LMT", "NOC", "RTX"],
        "Defense Software and ISR": ["PLTR", "KTOS", "AVAV", "LHX", "LDOS", "BAH"],
        "Aerospace Aftermarket and MRO Exposure": ["GE", "RTX", "HEI", "TDG"],
        "Electrification and Grid Modernization": ["ETN", "ABB", "PWR", "MYRG", "POWL", "VRT", "GEV"],
    },

    "Energy and Hard Assets": {
        "Energy Majors": ["XOM", "CVX", "COP", "SHEL", "BP", "TTE", "EQNR", "ENB", "PBR"],
        "US Shale and E&Ps": ["EOG", "DVN", "FANG", "MRO", "OXY", "APA", "AR", "RRC", "EXE", "CTRA"],
        "Natural Gas and LNG": ["LNG", "EQNR", "KMI", "WMB", "EPD", "ET", "WDS"],
        "Oilfield Services": ["SLB", "HAL", "BKR", "NOV", "FTI", "PTEN", "HP", "NBR", "OII"],
        "Uranium and Fuel Cycle": ["CCJ", "UUUU", "UEC", "URG", "UROY", "DNN", "NXE", "LEU", "URA", "URNM"],
        "Nuclear and Grid Buildout": ["VST", "CEG", "BWXT", "LEU", "ETN", "VRT", "GEV"],
        "Metals and Mining": ["BHP", "RIO", "VALE", "FCX", "NEM", "TECK", "SCCO", "AA"],
        "Gold and Silver Miners": ["GDX", "GDXJ", "NEM", "AEM", "GOLD", "KGC", "AG", "PAAS", "WPM"],
        "Energy Midstream and Storage": ["KMI", "WMB", "EPD", "ET", "ENB", "MPLX"],
        "Refining and Downstream": ["MPC", "VLO", "PSX", "DK", "PBF"],
        "Oil-Weighted E&Ps": ["EOG", "FANG", "OXY", "DVN", "MRO"],
        "Gas-Weighted E&Ps": ["AR", "RRC", "CTRA", "EXE", "EQT"],
        "LNG Export Levered": ["LNG", "KMI", "WMB", "ET", "EPD", "WDS"],
    },

    "Clean Energy Transition": {
        "Solar and Inverters": ["TAN", "FSLR", "ENPH", "SEDG", "RUN", "CSIQ", "JKS"],
        "Wind and Renewables": ["ICLN", "FAN", "NEP", "FSLR", "CWEN"],
        "Distributed Power and Fuel Cells": ["BE", "BLDP", "FCEL", "PLUG"],
        "Utilities and Power": ["VST", "CEG", "NEE", "DUK", "SO", "AEP", "XEL", "EXC", "PCG", "EIX"],
        "Regulated Utilities Core": ["DUK", "SO", "AEP", "XEL", "EXC", "ED"],
        "Merchant Power and Volatility": ["VST", "CEG", "NRG", "AES", "CWEN"],
        "Grid Equipment and Power Conversion": ["GEV", "ETN", "VRT", "ABB", "POWL", "HUBB"],
    },

    "Health and Longevity": {
        "Large-Cap Biotech": ["AMGN", "GILD", "REGN", "BIIB", "VRTX"],
        "GLP-1 and Metabolic": ["NVO", "LLY", "AZN", "MRK", "PFE"],
        "MedTech Devices": ["MDT", "SYK", "ISRG", "BSX", "ZBH", "EW", "PEN"],
        "Diagnostics and Tools": ["TMO", "DHR", "A", "RGEN", "ILMN"],
        "Healthcare Payers": ["UNH", "HUM", "CI", "ELV", "CNC", "MOH"],
        "Healthcare Providers and Hospitals": ["HCA", "THC", "UHS", "CYH"],
        "Healthcare Services and Outsourcing": ["LH", "DGX", "AMN", "EHC"],
        "Drug Channel and PBM Exposure": ["CVS", "UNH", "CI", "ELV"],
        "Life Science Tools and Supply Chain": ["TMO", "DHR", "A", "RGEN", "WAT"],
        "CRO and Clinical Services": ["IQV", "LH", "DGX", "MEDP", "ICLR"],
    },

    "Financials and Credit": {
        "Money-Center and IBs": ["JPM", "BAC", "C", "WFC", "GS", "MS"],
        "Regional Banks": ["KRE", "TFC", "FITB", "CFG", "RF", "KEY", "PNC", "USB", "MTB"],
        "Brokers and Exchanges": ["IBKR", "SCHW", "HOOD", "CME", "ICE", "NDAQ", "CBOE", "MKTX"],
        "Alt Managers and PE": ["BX", "KKR", "APO", "CG", "ARES", "OWL", "TPG"],
        "Private Credit and BDCs": ["ARES", "MAIN", "ARCC"],
        "Mortgage Finance": ["RKT", "UWMC", "COOP", "FNF", "NMIH", "ESNT"],
        "Insurers": ["BRK-B", "CB", "TRV", "PGR", "AIG", "MET"],
        "Insurance P&C": ["PGR", "TRV", "CB", "ALL", "CINF"],
        "Insurance Life and Retirement": ["MET", "PRU", "LNC", "AIG"],
        "Reinsurers": ["RNR", "RE", "EG"],
        "Insurance Brokers": ["AJG", "BRO", "MMC", "AON", "WTW"],
        "Deposit Beta and Funding Sensitivity": ["WFC", "BAC", "USB", "PNC", "KEY"],
        "CRE Sensitivity Banks": ["KEY", "CFG", "NYCB", "ZION", "WAL"],
        "Card and Consumer Lenders": ["AXP", "COF", "DFS", "SYF", "ALLY"],
        "Capital Markets Activity Proxies": ["GS", "MS", "CME", "ICE", "NDAQ"],
    },

    "Real Assets and Inflation Beneficiaries": {
        "Homebuilders": ["ITB", "DHI", "LEN", "NVR", "PHM", "TOL", "KBH", "MTH"],
        "REITs Core": ["VNQ", "PLD", "EQIX", "SPG", "O", "PSA", "DLR", "ARE", "VTR", "WELL"],
        "Industrials and Infrastructure": ["CAT", "DE", "URI", "PWR", "VMC", "MLM", "NUE"],
        "Shipping and Logistics": ["FDX", "UPS", "GXO", "XPO", "MATX", "DAC"],
        "Agriculture and Machinery": ["MOS", "NTR", "DE", "CNHI", "ADM", "BG", "CF", "AGCO"],
        "Housing Home Improvement and Repair": ["HD", "LOW", "TSCO", "POOL"],
        "Housing Building Products and Materials": ["BLDR", "TREX", "MAS", "VMC", "MLM", "SUM"],
        "Housing Mortgage and Title": ["COOP", "RKT", "UWMC", "FNF", "FAF"],
        "Housing Residential Transaction Proxies": ["RDFN", "ZG", "OPEN"],
        "Net Lease and Long Lease Duration REITs": ["O", "NNN", "ADC", "WPC"],
        "Multifamily REITs": ["AVB", "EQR", "UDR", "ESS", "MAA"],
        "Office and Refi Wall Sensitive REITs": ["BXP", "VNO", "SLG", "KRC"],
        "Construction Materials and Aggregates": ["VMC", "MLM", "SUM", "NUE", "EXP"],
    },

    "Consumer Cyclicals": {
        "Retail Discretionary": ["HD", "LOW", "M", "GPS", "BBY", "TJX", "TGT", "ROST"],
        "Restaurants": ["MCD", "SBUX", "YUM", "CMG", "DRI", "DPZ", "WING", "QSR"],
        "Travel and Booking": ["BKNG", "EXPE", "ABNB", "TRIP"],
        "Hotels and Casinos": ["MAR", "HLT", "IHG", "MGM", "LVS", "WYNN", "MLCO", "CZR", "PENN"],
        "Airlines": ["AAL", "DAL", "UAL", "LUV", "JBLU", "ALK"],
        "Autos Legacy OEMs": ["TM", "HMC", "F", "GM", "STLA"],
        "Electric Vehicles": ["TSLA", "RIVN", "LCID", "NIO", "LI", "XPEV"],
        "Luxury and Apparel": ["TPR", "RL", "CPRI", "LVMUY"],
        "Retail Asset-Heavy Inventory Risk": ["TGT", "BBY", "M", "GPS", "KSS", "BBWI"],
        "Retail Asset-Light Platforms and Marketplaces": ["AMZN", "EBAY", "ETSY", "SHOP", "PDD", "MELI"],
        "Prime Consumer Discretionary": ["HD", "LOW", "LULU", "NKE", "RH"],
        "Subprime and Credit-Sensitive Consumer": ["AFRM", "UPST", "COF", "DFS", "SYF"],
        "Big Ticket Durables": ["WHR", "SGI", "BBY", "RH", "LOW"],
        "Auto Parts and Service Tailwind": ["AZO", "ORLY", "LKQ", "AAP"],
    },

    "Defensives and Staples": {
        "Retail Staples": ["WMT", "TGT", "DG", "KR", "WBA"],
        "Staples and Beverages": ["PG", "KO", "PEP", "PM", "MO", "MDLZ"],
        "Telecom and Cable": ["T", "VZ", "TMUS", "CHTR", "CMCSA"],
        "Aerospace and Defense": ["LMT", "NOC", "RTX", "GD", "HII", "TDG", "HEI"],
        "Utilities Defensive": ["DUK", "SO", "AEP", "XEL", "EXC", "ED"],
        "Beverage Bottlers and Distributors": ["KOF", "COKE", "FIZZ", "KO"],
        "Pricing Power Staples": ["PG", "KO", "PEP", "COST", "MDLZ"],
        "Volume Risk and Trade-Down Staples": ["WMT", "DG", "DLTR", "KR"],
    },

    "Materials and Chemicals": {
        "Fertilizers": ["CF", "MOS", "NTR"],
        "Commodity Chemicals": ["DOW", "LYB", "CE"],
        "Specialty Chemicals": ["SHW", "EMN", "IFF", "PPG", "RPM"],
        "Industrial Gases": ["LIN", "APD", "AIQUY"],
        "Packaging and Containers": ["IP", "PKG", "WRK"],
        "Construction Chemicals and Adhesives": ["EMN", "RPM", "SHW"],
    },

    "Country and Region ETFs": {
        "Developed ex-US": ["VEA"],
        "Europe Broad": ["VGK"],
        "Eurozone": ["EZU"],
        "UK": ["EWU"],
        "Japan": ["EWJ"],
        "Canada": ["EWC"],
        "Australia": ["EWA"],
        "China": ["MCHI"],
        "Hong Kong": ["EWH"],
        "Taiwan": ["EWT"],
        "South Korea": ["EWY"],
        "India": ["INDA"],
        "Singapore": ["EWS"],
        "Emerging Markets Broad": ["IEMG"],
        "Frontier Markets": ["FM"],
        "Latin America Broad": ["ILF"],
        "Brazil": ["EWZ"],
        "Mexico": ["EWW"],
        "Chile": ["ECH"],
        "Peru": ["EPU"],
        "Israel": ["EIS"],
        "Saudi Arabia": ["KSA"],
        "UAE": ["UAE"],
        "Qatar": ["QAT"],
        "South Africa": ["EZA"],
    },

    "Alt and Global Risk": {
        "Crypto Proxies": ["COIN", "MSTR", "MARA", "RIOT", "BITO"],
        "China Tech ADRs": ["BABA", "BIDU", "JD", "PDD", "BILI", "NTES", "TCEHY"],
        "EM Internet and Commerce": ["MELI", "SE", "NU", "STNE"],
    },

    "Regime Diagnostics": {
        "Long-Duration Equities": ["ARKK", "IPO", "IGV", "SNOW", "NET", "DDOG", "MDB", "SHOP"],
        "Short-Duration Cash Flow": ["BRK-B", "PGR", "CB", "ICE", "CME", "NDAQ", "SPGI", "MSCI"],
        "Yield Proxies": ["XLU", "VZ", "T", "KMI", "EPD", "ENB"],
        "Rate-Sensitive Cyclicals": ["ITB", "XHB", "CVNA", "COF", "DFS", "AXP", "SYF"],
        "Labor-Intensive Services": ["SBUX", "CMG", "DRI", "MAR", "HLT", "RCL", "CCL"],
        "Automation and Productivity Winners": ["ROK", "ABB", "ETN", "PH", "CGNX", "ISRG", "SYM", "TER"],
        "IT Services and Outsourcing": ["ACN", "IBM", "CTSH", "EPAM", "GIB"],
        "Staffing and Wage-Sensitive Names": ["RHI", "MAN", "KFY", "ASGN"],
        "Leveraged Cyclicals": ["CCL", "RCL", "NCLH", "AAL", "UAL", "DAL", "MGM", "LVS"],
        "Net-Cash Compounders": ["AAPL", "MSFT", "GOOGL", "META", "ORCL", "ADBE", "INTU", "V"],
        "Equity Credit Stress Proxies": ["HYG", "JNK", "LQD"],
        "Financial Conditions Sensitive": ["IWM", "XLY", "KRE", "HYG", "ARKK"],
        "Commodity FX Equities": ["EWC", "EWA", "EWZ", "EWW"],
        "EM Domestic Demand": ["EEM", "INDA", "EWW", "EWZ", "EIDO"],
        "High Multiple Duration Risk": ["SNOW", "NET", "DDOG", "MDB", "SHOP"],
        "Buyback and Cash Yield Winners": ["AAPL", "MSFT", "BRK-B", "XOM", "META"],
        "High Short Interest and Retail Flow": ["CVNA", "UPST", "RIVN", "BYND"],
        "Spread Beta Equities": ["COF", "DFS", "SYF", "OMF", "ENVA"],
        "Inflation Pass-Through Winners": ["COST", "HD", "SHW", "NUE", "VMC"],
        "Input Cost and Margin Pressure Risk": ["DAL", "UAL", "SBUX", "DPZ", "WHR"],
    },

    "Sector Expansions": {
        "Transportation Rails and Trucking": ["UNP", "CSX", "NSC", "CNI", "CP", "JBHT", "KNX"],
        "Transportation Parcel and Last-Mile": ["FDX", "UPS", "GXO", "XPO"],
        "Air Cargo and Leasing": ["AL", "FTAI", "AER"],
        "Digital Advertising Platforms": ["GOOGL", "META", "TTD", "PINS", "SNAP"],
        "Traditional Media and Content": ["DIS", "WBD", "PARA", "CHTR", "CMCSA"],
        "Marketing Services and Agencies": ["IPG", "OMC", "WPP"],
        "Gov IT and Services": ["SAIC", "CACI", "LDOS", "BAH", "PSN", "G"],
        "Defense and Security Spending": ["LMT", "NOC", "RTX", "GD", "LHX", "HII", "TDG"],
        "Infrastructure and Grid Spend": ["PWR", "ETN", "VRT", "ABB", "GEV", "CAT", "URI", "VMC", "MLM"],
        "Healthcare Policy Sensitive": ["UNH", "HUM", "CI", "ELV", "CNC", "CVS"],
        "Energy Subsidy and Transition Plays": ["FSLR", "ENPH", "BE", "PLUG", "NEE", "VST", "ICLN"],
    },

    "Everyday Economy": {
        "Recreation and Experiences": ["YETI", "FOXF", "ASO", "DOO", "PLAY", "LYV", "FUN", "RICK"],
        "Big Ticket and Home Replacement Cycle": ["SGI", "SNBR", "WHR", "POOL", "LOW", "TTC", "LAD"],
        "Deferred Healthcare": ["ALGN", "EYE", "WRBY", "HSIC"],
        "Debt and Credit Paydown": ["OMF", "CACC", "SYF", "COF", "OPFI", "ENVA"],
        "Trade-Down Retail and Off-Price": ["TJX", "ROST", "BURL", "FIVE", "OLLI"],
        "Discount and Dollar": ["DG", "DLTR", "BURL"],
        "Staple Volume and Clubs": ["WMT", "COST", "KR"],
        "Value QSR": ["MCD", "YUM", "QSR", "WEN", "DPZ"],
        "Auto Parts and Repair": ["AZO", "ORLY", "AAP", "LKQ"],
        "Home Repair and Maintenance": ["HD", "LOW", "POOL", "TSCO"],
        "Used Auto and Affordability": ["KMX", "CVNA", "LAD"],
        "Shelter and Rent Economy": ["INVH", "AMH", "AVB", "EQR", "UDR", "MAA", "ESS"],
        "Manufactured Housing Affordability": ["ELS", "SUI", "CUBE"],
        "Storage and Mobility Stress": ["PSA", "EXR", "CUBE"],
        "Freight and Parcels": ["UNP", "CSX", "NSC", "JBHT", "KNX", "SAIA", "ODFL", "FDX", "UPS", "CHRW", "XPO", "GXO"],
        "Consumer Credit Stress": ["SYF", "COF", "DFS", "ALLY", "OMF", "ENVA"],
        "Payroll and Staffing": ["ADP", "PAYX", "RHI", "MAN", "KFY", "ASGN"],
        "Budget Hotels and Value Travel": ["CHH", "WH", "RYAAY"],
        "Chemicals Feedstock Sensitivity": ["DOW", "LYB", "WLK", "OLN", "CF", "NTR", "MOS"],
    },
}

ALL_BASKETS = {basket: tickers for cat in CATEGORIES.values() for basket, tickers in cat.items()}


# ============================================================
# Data helpers
# ============================================================
def _chunk(lst: List[str], n: int) -> List[List[str]]:
    n = max(1, n)
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out.index = pd.to_datetime(out.index)

    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)

    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def _to_float_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.apply(pd.to_numeric, errors="coerce")


def save_last_good_levels(levels: pd.DataFrame, meta: Dict[str, Any]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        levels.to_pickle(LEVELS_CACHE)
        with LEVELS_META_CACHE.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
    except Exception:
        pass


def load_last_good_levels() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    try:
        if not LEVELS_CACHE.exists():
            return pd.DataFrame(), {}

        levels = pd.read_pickle(LEVELS_CACHE)
        meta: Dict[str, Any] = {}

        if LEVELS_META_CACHE.exists():
            with LEVELS_META_CACHE.open("r", encoding="utf-8") as f:
                meta = json.load(f)

        return _clean_index(levels), meta
    except Exception:
        return pd.DataFrame(), {}


def _download_close_once(batch: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not batch:
        return pd.DataFrame()

    df = yf.download(
        tickers=batch,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
        ignore_tz=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            return pd.DataFrame()

        close = df["Close"].copy()
        close.columns = [str(c).upper() for c in close.columns]
        return _to_float_frame(_clean_index(close))

    if "Close" in df.columns and len(batch) == 1:
        sym = str(batch[0]).upper()
        close = df[["Close"]].rename(columns={"Close": sym}).copy()
        return _to_float_frame(_clean_index(close))

    return pd.DataFrame()


def _download_close(batch: List[str], start: pd.Timestamp, end: pd.Timestamp, retries: int = 2) -> pd.DataFrame:
    for attempt in range(retries + 1):
        try:
            close = _download_close_once(batch, start=start, end=end)
            if not close.empty:
                return close
        except Exception:
            pass

        if attempt < retries:
            time.sleep(0.5 * (attempt + 1))

    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_daily_levels(
    tickers: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_size: int = 45,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    uniq = sorted({str(t).upper().strip() for t in tickers if str(t).strip()})
    frames: List[pd.DataFrame] = []
    failed_batches: List[List[str]] = []

    for batch in _chunk(uniq, chunk_size):
        close = _download_close(batch, start=start, end=end)
        if close.empty:
            failed_batches.append(batch)
        else:
            frames.append(close)

    if not frames:
        cached, cached_meta = load_last_good_levels()
        if not cached.empty:
            return cached, {
                "source": "last_good_cache",
                "requested_tickers": len(uniq),
                "returned_tickers": int(cached.shape[1]),
                "cache_meta": cached_meta,
                "failed_batches": failed_batches,
            }

        return pd.DataFrame(), {
            "source": "yahoo",
            "requested_tickers": len(uniq),
            "returned_tickers": 0,
            "failed_batches": failed_batches,
        }

    wide = pd.concat(frames, axis=1)
    wide = wide.loc[:, ~wide.columns.duplicated()].sort_index()
    wide = _to_float_frame(_clean_index(wide))

    if not wide.empty:
        bidx = pd.bdate_range(wide.index.min(), wide.index.max(), name=wide.index.name)
        wide = wide.reindex(bidx).ffill()

    if BENCH not in wide.columns or wide[BENCH].dropna().empty:
        spy_only = _download_close([BENCH], start=start, end=end)
        if not spy_only.empty:
            spy_only = spy_only.reindex(wide.index if not wide.empty else spy_only.index).ffill()
            wide = pd.concat([wide.drop(columns=[BENCH], errors="ignore"), spy_only], axis=1)
            wide = wide.loc[:, ~wide.columns.duplicated()].sort_index()

    meta = {
        "source": "yahoo",
        "requested_tickers": len(uniq),
        "returned_tickers": int(wide.shape[1]),
        "missing_tickers": sorted(set(uniq) - set(wide.columns)),
        "failed_batches": failed_batches,
        "start": str(start.date()),
        "end": str(end.date()),
        "last_observation": str(wide.index.max().date()) if not wide.empty else None,
    }

    coverage = wide.shape[1] / max(len(uniq), 1)
    if not wide.empty and BENCH in wide.columns and coverage >= 0.70:
        save_last_good_levels(wide, meta)

    return wide, meta


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def fetch_market_caps(tickers: List[str]) -> Dict[str, float]:
    uniq = sorted({str(t).upper().strip() for t in tickers if str(t).strip()})
    if not uniq:
        return {}

    caps: Dict[str, float] = {}

    for batch in _chunk(uniq, 75):
        try:
            tk_obj = yf.Tickers(" ".join(batch))

            for sym, tk in tk_obj.tickers.items():
                try:
                    mc_val = None

                    fast_info = getattr(tk, "fast_info", None)
                    if fast_info is not None:
                        if isinstance(fast_info, dict):
                            mc_val = fast_info.get("market_cap")
                        else:
                            mc_val = getattr(fast_info, "market_cap", None)

                    if mc_val is None:
                        info = getattr(tk, "info", {}) or {}
                        mc_val = info.get("marketCap")

                    if mc_val is not None and pd.notna(mc_val):
                        caps[str(sym).upper()] = float(mc_val)

                except Exception:
                    continue

        except Exception:
            continue

    return caps


def normalize_basket_members(
    levels: pd.DataFrame,
    basket_tickers: List[str],
    market_caps: Optional[Dict[str, float]] = None,
    min_market_cap: Optional[float] = None,
    stale_days: int = 30,
) -> List[str]:
    valid: List[str] = []

    if levels.empty:
        return valid

    last_idx = pd.Timestamp(levels.index.max())

    for ticker in basket_tickers:
        sym = str(ticker).upper().strip()
        if not sym:
            continue

        if sym not in levels.columns:
            continue

        s = levels[sym].dropna()
        if s.empty:
            continue

        if s.index.max() < last_idx - pd.Timedelta(days=stale_days):
            continue

        if min_market_cap is not None and market_caps is not None and sym not in MARKET_CAP_EXCEPTIONS:
            mc = market_caps.get(sym)

            if mc is None and EXCLUDE_MISSING_MARKET_CAP:
                continue

            if mc is not None and mc < min_market_cap:
                continue

        valid.append(sym)

    return valid


def build_live_baskets(
    levels: pd.DataFrame,
    categories: Dict[str, Dict[str, List[str]]],
    market_caps: Dict[str, float],
    min_market_cap: Optional[float],
    stale_days: int,
) -> Dict[str, Dict[str, List[str]]]:
    live_categories: Dict[str, Dict[str, List[str]]] = {}

    for category, baskets in categories.items():
        live_baskets: Dict[str, List[str]] = {}

        for basket_name, tickers in baskets.items():
            live_members = normalize_basket_members(
                levels=levels,
                basket_tickers=tickers,
                market_caps=market_caps,
                min_market_cap=min_market_cap,
                stale_days=stale_days,
            )

            if live_members:
                live_baskets[basket_name] = live_members

        if live_baskets:
            live_categories[category] = live_baskets

    return live_categories


def flatten_baskets(categories: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:
    return {basket: tickers for cat in categories.values() for basket, tickers in cat.items()}


def ew_rets_from_levels(
    levels: pd.DataFrame,
    baskets: Dict[str, List[str]],
) -> pd.DataFrame:
    if levels.empty:
        return pd.DataFrame()

    rets = levels.pct_change(fill_method=None)
    out: Dict[str, pd.Series] = {}

    for basket_name, basket_tickers in baskets.items():
        cols = [str(t).upper() for t in basket_tickers if str(t).upper() in rets.columns]
        if not cols:
            continue

        if len(cols) == 1:
            out[basket_name] = rets[cols[0]]
        else:
            out[basket_name] = rets[cols].mean(axis=1, skipna=True)

    return pd.DataFrame(out).dropna(how="all")


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out


def macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def ema_regime(series: pd.Series, e1: int = 4, e2: int = 9, e3: int = 18) -> str:
    clean = series.dropna()
    if clean.shape[0] < max(e1, e2, e3):
        return "Neutral"

    ema1 = clean.ewm(span=e1, adjust=False).mean().iloc[-1]
    ema2 = clean.ewm(span=e2, adjust=False).mean().iloc[-1]
    ema3 = clean.ewm(span=e3, adjust=False).mean().iloc[-1]

    if ema1 > ema2 > ema3:
        return "Up"

    if ema1 < ema2 < ema3:
        return "Down"

    return "Neutral"


def momentum_label(hist: pd.Series, lookback: int = 5, z_window: int = 63) -> str:
    h = hist.dropna()
    if h.shape[0] < max(lookback + 1, z_window):
        return "Neutral"

    latest = h.iloc[-1]
    ref = h.iloc[-(lookback + 1)]

    base = "Positive" if latest > 0 else ("Negative" if latest < 0 else "Neutral")
    if base == "Neutral":
        return "Neutral"

    window = h.iloc[-z_window:]
    std = window.std(ddof=0)
    z = (latest - window.mean()) / std if pd.notna(std) and std != 0 else 0.0

    accel = "Accelerating" if (latest - ref) > 0 else "Decelerating"
    strength = "Strong" if abs(z) > 1 else "Weak"

    return f"{base} | {accel} | {strength}"


def realized_vol(returns: pd.Series, days: int = 63, ann: int = 252) -> float:
    sub = returns.dropna().iloc[-days:]
    if sub.shape[0] < 20:
        return np.nan

    return float(sub.std(ddof=0) * np.sqrt(ann) * 100.0)


def pct_since(levels: pd.Series, start_ts: pd.Timestamp) -> float:
    sub = levels[levels.index >= start_ts].dropna()
    if sub.shape[0] < 2:
        return np.nan

    return float((sub.iloc[-1] / sub.iloc[0]) - 1.0)


def first_valid_on_or_after(index: pd.Index, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    if len(index) == 0:
        return None

    loc = index.searchsorted(ts)
    if loc >= len(index):
        return None

    return pd.Timestamp(index[loc])


def compute_display_start(preset: str, today: date) -> date:
    presets = {
        "YTD": lambda t: date(t.year, 1, 1),
        "1W": lambda t: t - timedelta(days=7),
        "1M": lambda t: t - timedelta(days=30),
        "3M": lambda t: t - timedelta(days=90),
        "6M": lambda t: t - timedelta(days=182),
        "1Y": lambda t: t - timedelta(days=365),
        "3Y": lambda t: t - timedelta(days=365 * 3),
        "5Y": lambda t: t - timedelta(days=365 * 5),
    }
    return presets[preset](today)


def compute_fetch_start(display_start: date) -> date:
    return display_start - timedelta(days=INDICATOR_WARMUP_DAYS)


def build_panel_df(
    basket_returns_full: pd.DataFrame,
    display_start: pd.Timestamp,
    dynamic_label: str,
    benchmark_series_full: Optional[pd.Series] = None,
) -> pd.DataFrame:
    cols = [
        "Basket", "%5D", "%1M", f"%{dynamic_label}",
        "RSI(14D)", "MACD Momentum", "EMA 4/9/18", "RSI(14W)",
        "3M RVOL", "RVOL / SPY", "Corr(63D)"
    ]

    if basket_returns_full.empty:
        return pd.DataFrame(columns=cols).set_index("Basket")

    levels_full = 100.0 * (1.0 + basket_returns_full.fillna(0.0)).cumprod()
    rows: List[Dict[str, Any]] = []

    spy_rv = np.nan
    if benchmark_series_full is not None:
        spy_rv = realized_vol(benchmark_series_full, 63, 252)

    for basket in levels_full.columns:
        s_full = levels_full[basket].dropna()
        r_full = basket_returns_full[basket].dropna()

        if s_full.shape[0] < 15:
            continue

        start_anchor = first_valid_on_or_after(s_full.index, display_start)
        if start_anchor is None:
            continue

        s_display = s_full[s_full.index >= start_anchor]
        if s_display.shape[0] < 2:
            continue

        r5d = np.nan
        if s_full.shape[0] >= 6:
            r5d = (s_full.iloc[-1] / s_full.iloc[-6]) - 1.0

        r1m = pct_since(s_full, s_full.index.max() - pd.DateOffset(months=1))
        r_dyn = pct_since(s_full, start_anchor)

        rsi_d = rsi(s_full, 14)
        rsi_14d = rsi_d.dropna().iloc[-1] if rsi_d.dropna().shape[0] else np.nan

        weekly = s_full.resample("W-FRI").last().dropna()
        rsi_14w = np.nan
        if weekly.shape[0] >= 14:
            rsi_w = rsi(weekly, 14)
            if rsi_w.dropna().shape[0]:
                rsi_14w = rsi_w.dropna().iloc[-1]

        hist = macd_hist(s_full, 12, 26, 9)
        macd_m = momentum_label(hist, lookback=5, z_window=63)
        ema_tag = ema_regime(s_full, 4, 9, 18)

        rv = realized_vol(r_full, 63, 252)
        rv_rel = np.nan
        if pd.notna(rv) and pd.notna(spy_rv) and spy_rv != 0:
            rv_rel = rv / spy_rv

        corr_spy = np.nan
        if benchmark_series_full is not None:
            merged = pd.concat(
                [basket_returns_full[basket], benchmark_series_full],
                axis=1,
                join="inner"
            ).dropna()

            if merged.shape[0] >= 63:
                rolling_corr = merged.iloc[:, 0].rolling(63).corr(merged.iloc[:, 1])
                if rolling_corr.dropna().shape[0]:
                    corr_spy = rolling_corr.dropna().iloc[-1]

        rows.append({
            "Basket": basket,
            "%5D": round(r5d * 100, 1) if pd.notna(r5d) else np.nan,
            "%1M": round(r1m * 100, 1) if pd.notna(r1m) else np.nan,
            f"%{dynamic_label}": round(r_dyn * 100, 1) if pd.notna(r_dyn) else np.nan,
            "RSI(14D)": round(rsi_14d, 2) if pd.notna(rsi_14d) else np.nan,
            "MACD Momentum": macd_m,
            "EMA 4/9/18": ema_tag,
            "RSI(14W)": round(rsi_14w, 2) if pd.notna(rsi_14w) else np.nan,
            "3M RVOL": round(rv, 1) if pd.notna(rv) else np.nan,
            "RVOL / SPY": round(rv_rel, 2) if pd.notna(rv_rel) else np.nan,
            "Corr(63D)": round(corr_spy, 2) if pd.notna(corr_spy) else np.nan
        })

    if not rows:
        return pd.DataFrame(columns=cols).set_index("Basket")

    df = pd.DataFrame(rows).set_index("Basket")
    dyn_col = f"%{dynamic_label}"

    if dyn_col in df.columns:
        df = df.sort_values(by=dyn_col, ascending=False)

    return df


def slice_returns_for_display(returns_full: pd.DataFrame, display_start: pd.Timestamp) -> pd.DataFrame:
    if returns_full.empty:
        return returns_full

    out = returns_full[returns_full.index >= display_start].copy()
    return out.dropna(how="all")


# ============================================================
# Rendering helpers
# ============================================================
def color_ret(x):
    if pd.isna(x):
        return "white"

    if x >= 0:
        s = min(abs(x) / 20.0, 1.0)
        g = int(255 - 90 * s)
        return f"rgb({int(240 - 120 * s)},{g},{int(240 - 120 * s)})"

    s = min(abs(x) / 20.0, 1.0)
    r = int(255 - 90 * s)
    return f"rgb({r},{int(240 - 120 * s)},{int(240 - 120 * s)})"


def color_rsi(x):
    if pd.isna(x):
        return "white"

    if x >= 70:
        return "rgb(255,237,170)"

    if x <= 30:
        return "rgb(255,210,210)"

    return "rgb(230,236,245)"


def color_macd(tag):
    if not isinstance(tag, str):
        return "white"

    if tag.startswith("Positive"):
        if "Accelerating" in tag and "Strong" in tag:
            return "rgb(190,235,190)"
        if "Accelerating" in tag:
            return "rgb(204,238,204)"
        return "rgb(225,246,225)"

    if tag.startswith("Negative"):
        if "Accelerating" in tag and "Strong" in tag:
            return "rgb(255,190,190)"
        if "Accelerating" in tag:
            return "rgb(255,210,210)"
        return "rgb(255,228,228)"

    return "rgb(230,236,245)"


def color_ema(tag):
    if tag == "Up":
        return "rgb(204,238,204)"

    if tag == "Down":
        return "rgb(255,210,210)"

    return "rgb(230,236,245)"


def color_vol(x):
    if pd.isna(x):
        return "white"

    return "rgb(220,232,255)" if x < 60 else ("rgb(200,220,255)" if x < 90 else "rgb(180,205,255)")


def color_vol_rel(x):
    if pd.isna(x):
        return "white"

    if x < 0.85:
        return "rgb(225,246,225)"

    if x < 1.15:
        return "rgb(230,236,245)"

    return "rgb(255,228,228)"


def color_corr(x):
    if pd.isna(x):
        return "white"

    v = abs(x)

    if v >= 0.8:
        return "rgb(210,230,255)"

    if v >= 0.5:
        return "rgb(220,235,255)"

    return "rgb(230,240,255)"


def plot_panel_table(panel_df: pd.DataFrame):
    if panel_df.empty:
        st.info("No baskets passed the data-quality checks for this window.")
        return

    dynamic_col = [c for c in panel_df.columns if c.startswith("%") and c not in ["%5D", "%1M"]]
    dynamic_col = dynamic_col[0] if dynamic_col else None

    headers = [
        "Basket", "%5D", "%1M", dynamic_col,
        "RSI(14D)", "MACD Momentum", "EMA 4/9/18", "RSI(14W)",
        "3M RVOL", "RVOL / SPY", "Corr(63D)"
    ]

    values = [panel_df.index.tolist()]
    fill_colors = [["white"] * len(panel_df)]

    for col in ["%5D", "%1M", dynamic_col]:
        vals = panel_df[col].tolist()
        values.append(vals)
        fill_colors.append([color_ret(v) for v in vals])

    vals = panel_df["RSI(14D)"].tolist()
    values.append(vals)
    fill_colors.append([color_rsi(v) for v in vals])

    vals = panel_df["MACD Momentum"].tolist()
    values.append(vals)
    fill_colors.append([color_macd(v) for v in vals])

    vals = panel_df["EMA 4/9/18"].tolist()
    values.append(vals)
    fill_colors.append([color_ema(v) for v in vals])

    vals = panel_df["RSI(14W)"].tolist()
    values.append(vals)
    fill_colors.append([color_rsi(v) for v in vals])

    vals = panel_df["3M RVOL"].tolist()
    values.append(vals)
    fill_colors.append([color_vol(v) for v in vals])

    vals = panel_df["RVOL / SPY"].tolist()
    values.append(vals)
    fill_colors.append([color_vol_rel(v) for v in vals])

    vals = panel_df["Corr(63D)"].tolist()
    values.append(vals)
    fill_colors.append([color_corr(v) for v in vals])

    col_widths = [0.22, 0.06, 0.06, 0.085, 0.085, 0.145, 0.105, 0.075, 0.075, 0.075, 0.05]

    fig_tbl = go.Figure(data=[go.Table(
        columnwidth=[int(w * 1000) for w in col_widths],
        header=dict(
            values=headers,
            fill_color="white",
            line_color="rgb(230,230,230)",
            font=dict(color="black", size=13),
            align="left",
            height=32
        ),
        cells=dict(
            values=values,
            fill_color=fill_colors,
            line_color="rgb(240,240,240)",
            font=dict(color="black", size=12),
            align="left",
            height=26,
            format=[None, ".1f", ".1f", ".1f", ".2f", None, None, ".2f", ".1f", ".2f", ".2f"]
        )
    )])

    fig_tbl.update_layout(
        margin=dict(l=0, r=0, t=6, b=0),
        height=min(900, 64 + 26 * max(3, len(panel_df)))
    )

    st.plotly_chart(fig_tbl, use_container_width=True)


def plot_cumulative_chart(
    basket_returns_display: pd.DataFrame,
    title: str,
    benchmark_series_display: pd.Series,
):
    if basket_returns_display.empty or benchmark_series_display.dropna().empty:
        st.info("Insufficient data to render chart for this window.")
        return

    common_index = basket_returns_display.index.intersection(benchmark_series_display.index)
    if common_index.empty:
        st.info("No overlapping dates between series and benchmark.")
        return

    cum_pct = ((1 + basket_returns_display.loc[common_index].fillna(0.0)).cumprod() - 1.0) * 100.0
    bm_cum = ((1 + benchmark_series_display.loc[common_index].fillna(0.0)).cumprod() - 1.0) * 100.0

    fig = go.Figure()

    for i, basket in enumerate(cum_pct.columns):
        fig.add_trace(go.Scatter(
            x=cum_pct.index,
            y=cum_pct[basket],
            mode="lines",
            line=dict(width=2, color=PASTEL[i % len(PASTEL)]),
            name=basket,
            hovertemplate=f"{basket}<br>% Cum: %{{y:.1f}}%<extra></extra>"
        ))

    fig.add_trace(go.Scatter(
        x=bm_cum.index,
        y=bm_cum.values,
        mode="lines",
        line=dict(width=2, dash="dash", color="#888"),
        name="SPY",
        hovertemplate="SPY<br>% Cum: %{y:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        showlegend=True,
        hovermode="x unified",
        yaxis_title="Cumulative return, %",
        title=dict(text=title, x=0, xanchor="left", y=0.95),
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor", showgrid=True),
        yaxis=dict(zeroline=False, showgrid=True)
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Sidebar
# ============================================================
st.title(TITLE)
st.caption(SUBTITLE)

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Internal basket monitor for equal-weight performance, trend, and SPY-relative context.

        **What this tab shows**
        - Daily basket snapshots and relative performance versus SPY.
        - Trend, momentum, volatility, and correlation diagnostics by basket.
        - Constituents are cleaned automatically, so unavailable or excluded tickers are removed from calculations.

        **Data source**
        - Internal basket definitions and Yahoo Finance market data.
        """
    )
    st.divider()
    st.markdown("### Controls")

    today = date.today()

    preset = st.selectbox(
        "Date Range Preset",
        ["YTD", "1W", "1M", "3M", "6M", "1Y", "3Y", "5Y"],
        index=0
    )

    apply_market_cap_filter = st.checkbox("Apply $1B market-cap filter", value=True)
    stale_days = st.slider("Stale Price Threshold, Days", min_value=10, max_value=90, value=30, step=5)

    selected_categories = st.multiselect(
        "Categories",
        options=list(CATEGORIES.keys()),
        default=list(CATEGORIES.keys())
    )

display_start_date = compute_display_start(preset, today)
fetch_start_date = compute_fetch_start(display_start_date)
end_date = today

DYNAMIC_LABEL = preset
min_market_cap = MIN_MARKET_CAP if apply_market_cap_filter else None


# ============================================================
# Fetch data
# ============================================================
need = {BENCH}
for tickers in ALL_BASKETS.values():
    need.update(str(t).upper() for t in tickers)

levels, fetch_meta = fetch_daily_levels(
    sorted(list(need)),
    start=pd.to_datetime(fetch_start_date),
    end=pd.to_datetime(end_date) + pd.Timedelta(days=1)
)

if levels.empty:
    st.error("No data returned for the selected range.")
    st.stop()

if fetch_meta.get("source") == "last_good_cache":
    st.warning("Yahoo returned no usable data. Showing the last-good local cache.")

if BENCH not in levels.columns or levels[BENCH].dropna().empty:
    st.error("SPY data missing or empty for the selected range.")
    st.stop()

market_caps = fetch_market_caps(list(levels.columns))

live_categories = build_live_baskets(
    levels=levels,
    categories=CATEGORIES,
    market_caps=market_caps,
    min_market_cap=min_market_cap,
    stale_days=stale_days,
)

if selected_categories:
    live_categories = {
        cat: baskets
        for cat, baskets in live_categories.items()
        if cat in selected_categories
    }

live_all_baskets = flatten_baskets(live_categories)

all_basket_rets_full = ew_rets_from_levels(
    levels=levels,
    baskets=live_all_baskets,
)

if all_basket_rets_full.empty:
    st.error("No baskets passed the current filters.")
    st.stop()

bench_rets_full = levels[BENCH].pct_change(fill_method=None).dropna()

display_start_ts = pd.Timestamp(display_start_date)
all_basket_rets_display = slice_returns_for_display(all_basket_rets_full, display_start_ts)
bench_rets_display = bench_rets_full[bench_rets_full.index >= display_start_ts].copy()


# ============================================================
# Consolidated panel + chart
# ============================================================
st.subheader("All Baskets | Consolidated Panel")

all_panel_df = build_panel_df(
    basket_returns_full=all_basket_rets_full,
    display_start=display_start_ts,
    dynamic_label=DYNAMIC_LABEL,
    benchmark_series_full=bench_rets_full
)

plot_panel_table(all_panel_df)

st.subheader("All Baskets | Cumulative Performance vs SPY")

all_display_chart_rets = all_basket_rets_display[all_panel_df.index] if not all_panel_df.empty else all_basket_rets_display

plot_cumulative_chart(
    basket_returns_display=all_display_chart_rets,
    title="All Baskets vs SPY",
    benchmark_series_display=bench_rets_display
)


# ============================================================
# Per-category sections
# ============================================================
for category, baskets in live_categories.items():
    st.markdown(f"## {category}")

    cat_names = [basket for basket in baskets.keys() if basket in all_basket_rets_full.columns]

    if not cat_names:
        st.info("No data for this group in the selected range.")
        continue

    cat_rets_full = all_basket_rets_full[cat_names].dropna(how="all")
    if cat_rets_full.empty:
        st.info("No data for this group in the selected range.")
        continue

    cat_rets_display = slice_returns_for_display(cat_rets_full, display_start_ts)

    cat_panel = build_panel_df(
        basket_returns_full=cat_rets_full,
        display_start=display_start_ts,
        dynamic_label=DYNAMIC_LABEL,
        benchmark_series_full=bench_rets_full
    )

    plot_panel_table(cat_panel)

    chart_rets = cat_rets_display[cat_panel.index] if not cat_panel.empty else cat_rets_display

    plot_cumulative_chart(
        basket_returns_display=chart_rets,
        title=f"{category} | Cumulative Performance vs SPY",
        benchmark_series_display=bench_rets_display
    )


# ============================================================
# Basket constituents
# ============================================================
with st.expander("Basket Constituents"):
    st.caption("Shows only live members used in the calculations after price, stale-data, and market-cap filters.")

    for category, groups in live_categories.items():
        st.markdown(f"**{category}**")
        for name, tickers in groups.items():
            st.write(f"- {name}: {', '.join(sorted(set(str(t).upper() for t in tickers)))}")


with st.expander("Data Notes"):
    last_obs = fetch_meta.get("last_observation")
    requested = fetch_meta.get("requested_tickers")
    returned = fetch_meta.get("returned_tickers")

    if last_obs:
        st.write(f"Last observation: {last_obs}")

    if requested is not None and returned is not None:
        st.write(f"Yahoo price coverage: {returned}/{requested} tickers returned.")

    missing = fetch_meta.get("missing_tickers", [])
    if missing:
        st.write("Tickers missing from Yahoo result:")
        st.write(", ".join(missing[:300]))

st.caption("© 2026 AD Fund Management LP")
