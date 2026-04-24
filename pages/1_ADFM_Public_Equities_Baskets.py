import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time


# ============================================================
# Page and theme
# ============================================================
st.set_page_config(page_title="ADFM Basket Panels", layout="wide")

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.15rem; padding-bottom: 2rem; max-width: 1600px;}
    h1, h2, h3 {font-weight: 650; letter-spacing: 0.1px;}
    .stPlotlyChart {background: #ffffff;}
    div[data-testid="stMetricValue"] {font-size: 1.15rem;}
    div[data-testid="stMetricLabel"] {font-size: 0.80rem;}
    .small-note {font-size: 0.82rem; color: #555;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

TITLE = "ADFM Basket Panels"
SUBTITLE = (
    "Decision-grade basket monitor for equal-weight performance, relative strength, "
    "breadth, risk, and data-quality diagnostics."
)

PASTEL = [
    "#AEC6CF", "#FFB347", "#B39EB5", "#77DD77", "#F49AC2",
    "#CFCFC4", "#DEA5A4", "#C6E2FF", "#FFDAC1", "#E2F0CB",
    "#C7CEEA", "#FFB3BA", "#FFD1DC", "#B5EAD7", "#E7E6F7",
    "#F1E3DD", "#B0E0E6", "#E0BBE4", "#F3E5AB", "#D5E8D4"
]

INDICATOR_WARMUP_DAYS = 520
DEFAULT_MIN_MARKET_CAP = 1_000_000_000
CACHE_DIR = Path(".adfm_cache")
LEVELS_CACHE = CACHE_DIR / "basket_levels_last_good.pkl"
LEVELS_META_CACHE = CACHE_DIR / "basket_levels_last_good_meta.json"
OPTIONAL_BASKET_FILE = Path("adfm_baskets.json")

BENCHMARK = "SPY"
SECONDARY_BENCHMARKS = ["QQQ", "IWM"]

# Keep this as a set. Add exceptions only when you intentionally want smaller caps to pass.
MARKET_CAP_EXCEPTIONS: set[str] = set()


DEFAULT_CATEGORIES: Dict[str, Dict[str, List[str]]] = {
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



# ============================================================
# Basket loading
# ============================================================
def validate_basket_config(obj: Any) -> Dict[str, Dict[str, List[str]]]:
    if not isinstance(obj, dict):
        raise ValueError("Basket config must be a dictionary of categories.")

    cleaned: Dict[str, Dict[str, List[str]]] = {}

    for category, baskets in obj.items():
        if not isinstance(category, str) or not category.strip():
            continue
        if not isinstance(baskets, dict):
            continue

        clean_baskets: Dict[str, List[str]] = {}
        for basket_name, tickers in baskets.items():
            if not isinstance(basket_name, str) or not basket_name.strip():
                continue
            if not isinstance(tickers, list):
                continue

            clean_tickers = sorted({
                str(t).upper().strip()
                for t in tickers
                if str(t).strip()
            })

            if clean_tickers:
                clean_baskets[basket_name.strip()] = clean_tickers

        if clean_baskets:
            cleaned[category.strip()] = clean_baskets

    if not cleaned:
        raise ValueError("Basket config did not contain any valid categories.")

    return cleaned


def load_basket_definitions(default_categories: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, List[str]]]:
    """
    The app preserves the full embedded ADFM taxonomy by default.
    If adfm_baskets.json exists beside the app, it will be loaded instead.
    Expected JSON shape:
    {
      "Category Name": {
        "Basket Name": ["AAPL", "MSFT"]
      }
    }
    """
    if OPTIONAL_BASKET_FILE.exists():
        try:
            with OPTIONAL_BASKET_FILE.open("r", encoding="utf-8") as f:
                external = json.load(f)
            return validate_basket_config(external)
        except Exception as exc:
            st.warning(f"Could not load adfm_baskets.json, using embedded baskets instead. Error: {exc}")

    return validate_basket_config(default_categories)


CATEGORIES = load_basket_definitions(DEFAULT_CATEGORIES)
ALL_BASKETS = {basket: tickers for category in CATEGORIES.values() for basket, tickers in category.items()}


# ============================================================
# Date helpers
# ============================================================
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


def first_valid_on_or_after(index: pd.Index, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    if len(index) == 0:
        return None
    loc = index.searchsorted(ts)
    if loc >= len(index):
        return None
    return pd.Timestamp(index[loc])


# ============================================================
# Generic helpers
# ============================================================
def _chunk(lst: List[str], n: int) -> List[List[str]]:
    n = max(1, int(n))
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


# ============================================================
# Yahoo data fetchers
# ============================================================
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
    last = pd.DataFrame()
    for attempt in range(retries + 1):
        try:
            last = _download_close_once(batch, start, end)
            if not last.empty:
                return last
        except Exception:
            pass
        if attempt < retries:
            time.sleep(0.6 * (attempt + 1))
    return last


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_daily_levels(
    tickers: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_size: int = 45
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
        cached, cache_meta = load_last_good_levels()
        if not cached.empty:
            meta = {
                "source": "last_good_cache",
                "cache_meta": cache_meta,
                "requested_tickers": len(uniq),
                "returned_tickers": int(cached.shape[1]),
                "failed_batches": failed_batches,
            }
            return cached, meta
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

    # Ensure the primary benchmark exists even if it was lost in a failed chunk.
    if BENCHMARK not in wide.columns or wide[BENCHMARK].dropna().empty:
        spy_only = _download_close([BENCHMARK], start=start, end=end)
        if not spy_only.empty:
            spy_only = spy_only.reindex(wide.index if not wide.empty else spy_only.index).ffill()
            wide = pd.concat([wide.drop(columns=[BENCHMARK], errors="ignore"), spy_only], axis=1)
            wide = wide.loc[:, ~wide.columns.duplicated()].sort_index()

    returned = set(wide.columns)
    missing = sorted(set(uniq) - returned)

    meta = {
        "source": "yahoo",
        "requested_tickers": len(uniq),
        "returned_tickers": int(wide.shape[1]),
        "missing_tickers": missing,
        "failed_batches": failed_batches,
        "start": str(start.date()),
        "end": str(end.date()),
        "last_observation": str(wide.index.max().date()) if not wide.empty else None,
    }

    # Save as last-good only if the result contains the benchmark and has broad coverage.
    coverage = wide.shape[1] / max(len(uniq), 1)
    if not wide.empty and BENCHMARK in wide.columns and coverage >= 0.70:
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


# ============================================================
# Basket construction and diagnostics
# ============================================================
def build_member_diagnostics(
    levels: pd.DataFrame,
    categories: Dict[str, Dict[str, List[str]]],
    market_caps: Dict[str, float],
    min_market_cap: Optional[float],
    cap_policy: str,
    stale_days: int,
    min_members: int,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    members_by_basket: Dict[str, List[str]] = {}

    if levels.empty:
        return members_by_basket, pd.DataFrame()

    last_idx = pd.Timestamp(levels.index.max())
    cap_filter_on = min_market_cap is not None and cap_policy != "Do not filter by market cap"
    exclude_missing_caps = cap_policy == "Exclude missing market caps"

    for category, baskets in categories.items():
        for basket_name, basket_tickers in baskets.items():
            intended = sorted({str(t).upper().strip() for t in basket_tickers if str(t).strip()})

            valid: List[str] = []
            missing_price: List[str] = []
            stale: List[str] = []
            missing_cap: List[str] = []
            below_cap: List[str] = []

            for sym in intended:
                if sym not in levels.columns or levels[sym].dropna().empty:
                    missing_price.append(sym)
                    continue

                s = levels[sym].dropna()
                if s.index.max() < last_idx - pd.Timedelta(days=stale_days):
                    stale.append(sym)
                    continue

                if cap_filter_on and sym not in MARKET_CAP_EXCEPTIONS:
                    mc = market_caps.get(sym)
                    if mc is None:
                        missing_cap.append(sym)
                        if exclude_missing_caps:
                            continue
                    elif mc < float(min_market_cap):
                        below_cap.append(sym)
                        continue

                valid.append(sym)

            required_members = 1 if len(intended) <= 1 else max(1, int(min_members))
            valid_for_calc = valid if len(valid) >= required_members else []

            if valid_for_calc:
                members_by_basket[basket_name] = valid_for_calc

            excluded = sorted(set(intended) - set(valid_for_calc))

            rows.append({
                "Category": category,
                "Basket": basket_name,
                "Defined Members": len(intended),
                "Members Used": len(valid_for_calc),
                "Coverage %": round(100 * len(valid_for_calc) / max(len(intended), 1), 1),
                "Missing Price": len(missing_price),
                "Stale Price": len(stale),
                "Missing Market Cap": len(missing_cap),
                "Below Market Cap": len(below_cap),
                "Used Tickers": ", ".join(valid_for_calc),
                "Excluded Tickers": ", ".join(excluded),
                "Missing Price Tickers": ", ".join(missing_price),
                "Stale Tickers": ", ".join(stale),
                "Missing Cap Tickers": ", ".join(missing_cap),
                "Below Cap Tickers": ", ".join(below_cap),
            })

    diag = pd.DataFrame(rows)
    return members_by_basket, diag


def ew_rets_from_members(levels: pd.DataFrame, members_by_basket: Dict[str, List[str]]) -> pd.DataFrame:
    if levels.empty or not members_by_basket:
        return pd.DataFrame()

    rets = levels.pct_change()
    out: Dict[str, pd.Series] = {}

    for basket_name, cols in members_by_basket.items():
        usable = [c for c in cols if c in rets.columns]
        if not usable:
            continue
        if len(usable) == 1:
            out[basket_name] = rets[usable[0]]
        else:
            out[basket_name] = rets[usable].mean(axis=1, skipna=True)

    return pd.DataFrame(out).dropna(how="all")


def slice_returns_for_display(returns_full: pd.DataFrame, display_start: pd.Timestamp) -> pd.DataFrame:
    if returns_full.empty:
        return returns_full
    return returns_full[returns_full.index >= display_start].dropna(how="all").copy()


# ============================================================
# Indicators and risk
# ============================================================
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


def drawdown_stats(levels: pd.Series) -> Tuple[float, float]:
    clean = levels.dropna()
    if clean.shape[0] < 2:
        return np.nan, np.nan
    dd = clean / clean.cummax() - 1.0
    return float(dd.iloc[-1] * 100.0), float(dd.min() * 100.0)


def rolling_beta(asset: pd.Series, benchmark: pd.Series, days: int = 63) -> float:
    merged = pd.concat([asset, benchmark], axis=1, join="inner").dropna().iloc[-days:]
    if merged.shape[0] < 20:
        return np.nan
    var_b = merged.iloc[:, 1].var(ddof=0)
    if pd.isna(var_b) or var_b == 0:
        return np.nan
    return float(merged.iloc[:, 0].cov(merged.iloc[:, 1]) / var_b)


def rolling_corr(asset: pd.Series, benchmark: pd.Series, days: int = 63) -> float:
    merged = pd.concat([asset, benchmark], axis=1, join="inner").dropna().iloc[-days:]
    if merged.shape[0] < 20:
        return np.nan
    return float(merged.iloc[:, 0].corr(merged.iloc[:, 1]))


def upside_downside_capture(asset: pd.Series, benchmark: pd.Series, days: int = 63) -> Tuple[float, float]:
    merged = pd.concat([asset, benchmark], axis=1, join="inner").dropna().iloc[-days:]
    if merged.shape[0] < 20:
        return np.nan, np.nan

    a = merged.iloc[:, 0]
    b = merged.iloc[:, 1]

    up = merged[b > 0]
    down = merged[b < 0]

    upside = np.nan
    downside = np.nan

    if not up.empty and up.iloc[:, 1].mean() != 0:
        upside = float(up.iloc[:, 0].mean() / up.iloc[:, 1].mean() * 100.0)

    if not down.empty and down.iloc[:, 1].mean() != 0:
        downside = float(down.iloc[:, 0].mean() / down.iloc[:, 1].mean() * 100.0)

    return upside, downside


def relative_strength_percentile(asset_returns: pd.Series, benchmark_returns: pd.Series, window: int = 252) -> float:
    merged = pd.concat([asset_returns, benchmark_returns], axis=1, join="inner").dropna()
    if merged.shape[0] < 63:
        return np.nan

    asset_level = (1.0 + merged.iloc[:, 0]).cumprod()
    bench_level = (1.0 + merged.iloc[:, 1]).cumprod()
    rs = asset_level / bench_level.replace(0, np.nan)
    sub = rs.dropna().iloc[-window:]

    if sub.shape[0] < 63:
        return np.nan

    latest = sub.iloc[-1]
    return float((sub <= latest).mean() * 100.0)


def basket_breadth(levels: pd.DataFrame, members: List[str]) -> Dict[str, float]:
    cols = [c for c in members if c in levels.columns]
    if not cols:
        return {
            "% > 20D": np.nan,
            "% > 50D": np.nan,
            "% > 200D": np.nan,
            "% 20D High": np.nan,
            "% 20D Low": np.nan,
            "Median 1M %": np.nan,
        }

    px = levels[cols].dropna(how="all")
    if px.empty:
        return {
            "% > 20D": np.nan,
            "% > 50D": np.nan,
            "% > 200D": np.nan,
            "% 20D High": np.nan,
            "% 20D Low": np.nan,
            "Median 1M %": np.nan,
        }

    latest = px.iloc[-1]

    def pct_above_ma(window: int) -> float:
        ma = px.rolling(window).mean().iloc[-1]
        valid = latest.notna() & ma.notna()
        if valid.sum() == 0:
            return np.nan
        return float((latest[valid] > ma[valid]).mean() * 100.0)

    rolling_high = px.rolling(20).max().iloc[-1]
    rolling_low = px.rolling(20).min().iloc[-1]
    valid_high = latest.notna() & rolling_high.notna()
    valid_low = latest.notna() & rolling_low.notna()

    pct_high = float((latest[valid_high] >= rolling_high[valid_high] * 0.999).mean() * 100.0) if valid_high.sum() else np.nan
    pct_low = float((latest[valid_low] <= rolling_low[valid_low] * 1.001).mean() * 100.0) if valid_low.sum() else np.nan

    ret_1m = px.pct_change(21).iloc[-1] * 100.0
    median_1m = float(ret_1m.dropna().median()) if ret_1m.dropna().shape[0] else np.nan

    return {
        "% > 20D": pct_above_ma(20),
        "% > 50D": pct_above_ma(50),
        "% > 200D": pct_above_ma(200),
        "% 20D High": pct_high,
        "% 20D Low": pct_low,
        "Median 1M %": median_1m,
    }


# ============================================================
# Panel construction
# ============================================================
def build_panel_df(
    basket_returns_full: pd.DataFrame,
    display_start: pd.Timestamp,
    dynamic_label: str,
    members_by_basket: Dict[str, List[str]],
    member_diag: pd.DataFrame,
    levels: pd.DataFrame,
    benchmark_returns_full: pd.Series,
) -> pd.DataFrame:
    if basket_returns_full.empty:
        return pd.DataFrame()

    category_map = {}
    if not member_diag.empty:
        category_map = member_diag.set_index("Basket")["Category"].to_dict()
        member_count_map = member_diag.set_index("Basket")["Members Used"].to_dict()
        defined_count_map = member_diag.set_index("Basket")["Defined Members"].to_dict()
        coverage_map = member_diag.set_index("Basket")["Coverage %"].to_dict()
    else:
        member_count_map = {}
        defined_count_map = {}
        coverage_map = {}

    levels_full = 100.0 * (1.0 + basket_returns_full.fillna(0.0)).cumprod()
    rows: List[Dict[str, Any]] = []

    spy_level = 100.0 * (1.0 + benchmark_returns_full.fillna(0.0)).cumprod()
    spy_rv = realized_vol(benchmark_returns_full, 63, 252)

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

        r5d = (s_full.iloc[-1] / s_full.iloc[-6] - 1.0) if s_full.shape[0] >= 6 else np.nan
        r1m = pct_since(s_full, s_full.index.max() - pd.DateOffset(months=1))
        r_dyn = pct_since(s_full, start_anchor)

        bm_display_start = first_valid_on_or_after(spy_level.index, start_anchor)
        bm_dyn = pct_since(spy_level, bm_display_start) if bm_display_start is not None else np.nan
        rel_dyn = r_dyn - bm_dyn if pd.notna(r_dyn) and pd.notna(bm_dyn) else np.nan

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
        rv_rel = rv / spy_rv if pd.notna(rv) and pd.notna(spy_rv) and spy_rv != 0 else np.nan

        beta = rolling_beta(r_full, benchmark_returns_full, 63)
        corr = rolling_corr(r_full, benchmark_returns_full, 63)
        upside, downside = upside_downside_capture(r_full, benchmark_returns_full, 63)
        rs_pctile = relative_strength_percentile(r_full, benchmark_returns_full, 252)
        curr_dd, max_dd = drawdown_stats(s_full[s_full.index >= start_anchor])

        breadth = basket_breadth(levels, members_by_basket.get(basket, []))

        rows.append({
            "Category": category_map.get(basket, ""),
            "Basket": basket,
            "Members": int(member_count_map.get(basket, len(members_by_basket.get(basket, [])))),
            "Defined": int(defined_count_map.get(basket, len(members_by_basket.get(basket, [])))),
            "Coverage %": round(float(coverage_map.get(basket, np.nan)), 1) if pd.notna(coverage_map.get(basket, np.nan)) else np.nan,
            "%5D": round(r5d * 100, 1) if pd.notna(r5d) else np.nan,
            "%1M": round(r1m * 100, 1) if pd.notna(r1m) else np.nan,
            f"%{dynamic_label}": round(r_dyn * 100, 1) if pd.notna(r_dyn) else np.nan,
            f"Rel %{dynamic_label} vs SPY": round(rel_dyn * 100, 1) if pd.notna(rel_dyn) else np.nan,
            "RS %ile 1Y": round(rs_pctile, 1) if pd.notna(rs_pctile) else np.nan,
            "RSI(14D)": round(rsi_14d, 1) if pd.notna(rsi_14d) else np.nan,
            "MACD Momentum": macd_m,
            "EMA 4/9/18": ema_tag,
            "RSI(14W)": round(rsi_14w, 1) if pd.notna(rsi_14w) else np.nan,
            "% > 20D": round(breadth["% > 20D"], 1) if pd.notna(breadth["% > 20D"]) else np.nan,
            "% > 50D": round(breadth["% > 50D"], 1) if pd.notna(breadth["% > 50D"]) else np.nan,
            "% > 200D": round(breadth["% > 200D"], 1) if pd.notna(breadth["% > 200D"]) else np.nan,
            "% 20D High": round(breadth["% 20D High"], 1) if pd.notna(breadth["% 20D High"]) else np.nan,
            "% 20D Low": round(breadth["% 20D Low"], 1) if pd.notna(breadth["% 20D Low"]) else np.nan,
            "Median 1M %": round(breadth["Median 1M %"], 1) if pd.notna(breadth["Median 1M %"]) else np.nan,
            "3M RVOL": round(rv, 1) if pd.notna(rv) else np.nan,
            "RVOL / SPY": round(rv_rel, 2) if pd.notna(rv_rel) else np.nan,
            "Beta 63D": round(beta, 2) if pd.notna(beta) else np.nan,
            "Corr 63D": round(corr, 2) if pd.notna(corr) else np.nan,
            "Upside Capture": round(upside, 0) if pd.notna(upside) else np.nan,
            "Downside Capture": round(downside, 0) if pd.notna(downside) else np.nan,
            "Current DD %": round(curr_dd, 1) if pd.notna(curr_dd) else np.nan,
            "Max DD %": round(max_dd, 1) if pd.notna(max_dd) else np.nan,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("Basket")
    sort_col = f"%{dynamic_label}"
    if sort_col in df.columns:
        df = df.sort_values(by=sort_col, ascending=False)
    return df


# ============================================================
# Styling and rendering
# ============================================================
def style_panel(df: pd.DataFrame, dynamic_label: str) -> pd.io.formats.style.Styler:
    ret_cols = ["%5D", "%1M", f"%{dynamic_label}", f"Rel %{dynamic_label} vs SPY", "Median 1M %"]
    breadth_cols = ["% > 20D", "% > 50D", "% > 200D", "% 20D High", "% 20D Low", "RS %ile 1Y"]
    risk_cols = ["3M RVOL", "RVOL / SPY", "Beta 63D", "Corr 63D", "Upside Capture", "Downside Capture", "Current DD %", "Max DD %"]
    rsi_cols = ["RSI(14D)", "RSI(14W)"]

    def color_return(v: Any) -> str:
        if pd.isna(v):
            return ""
        return "background-color: #e8f4ea;" if float(v) >= 0 else "background-color: #fdeaea;"

    def color_breadth(v: Any) -> str:
        if pd.isna(v):
            return ""
        x = float(v)
        if x >= 70:
            return "background-color: #e2f2e5;"
        if x <= 30:
            return "background-color: #fdecec;"
        return "background-color: #f2f4f7;"

    def color_rsi(v: Any) -> str:
        if pd.isna(v):
            return ""
        x = float(v)
        if x >= 70:
            return "background-color: #fff2cc;"
        if x <= 30:
            return "background-color: #fdecec;"
        return "background-color: #f2f4f7;"

    def color_risk(v: Any) -> str:
        if pd.isna(v):
            return ""
        return "background-color: #eef4ff;"

    styler = df.style.format(precision=1)

    for col in ret_cols:
        if col in df.columns:
            styler = styler.map(color_return, subset=[col])

    for col in breadth_cols:
        if col in df.columns:
            styler = styler.map(color_breadth, subset=[col])

    for col in rsi_cols:
        if col in df.columns:
            styler = styler.map(color_rsi, subset=[col])

    for col in risk_cols:
        if col in df.columns:
            styler = styler.map(color_risk, subset=[col])

    return styler


def render_panel(df: pd.DataFrame, dynamic_label: str, title: Optional[str] = None) -> None:
    if title:
        st.subheader(title)

    if df.empty:
        st.info("No baskets passed the data-quality checks for this window.")
        return

    st.dataframe(
        style_panel(df, dynamic_label),
        use_container_width=True,
        height=min(900, 38 + 29 * max(4, len(df))),
    )


def select_chart_columns(
    panel_df: pd.DataFrame,
    dynamic_label: str,
    mode: str,
    max_lines: int,
    selected_baskets: Optional[List[str]] = None,
) -> List[str]:
    if panel_df.empty:
        return []

    dyn_col = f"%{dynamic_label}"
    rel_col = f"Rel %{dynamic_label} vs SPY"

    if selected_baskets:
        return [b for b in selected_baskets if b in panel_df.index]

    if mode == "Leaders":
        return panel_df.sort_values(dyn_col, ascending=False).head(max_lines).index.tolist()

    if mode == "Laggards":
        return panel_df.sort_values(dyn_col, ascending=True).head(max_lines).index.tolist()

    if mode == "Relative leaders vs SPY" and rel_col in panel_df.columns:
        return panel_df.sort_values(rel_col, ascending=False).head(max_lines).index.tolist()

    if mode == "Relative laggards vs SPY" and rel_col in panel_df.columns:
        return panel_df.sort_values(rel_col, ascending=True).head(max_lines).index.tolist()

    if mode == "Leaders and laggards":
        half = max(1, max_lines // 2)
        leaders = panel_df.sort_values(dyn_col, ascending=False).head(half).index.tolist()
        laggards = panel_df.sort_values(dyn_col, ascending=True).head(max_lines - len(leaders)).index.tolist()
        return list(dict.fromkeys(leaders + laggards))

    return panel_df.head(max_lines).index.tolist()


def plot_cumulative_chart(
    basket_returns_display: pd.DataFrame,
    title: str,
    benchmark_series_display: pd.Series,
    selected_columns: List[str],
) -> None:
    if basket_returns_display.empty or benchmark_series_display.dropna().empty:
        st.info("Insufficient data to render chart for this window.")
        return

    selected_columns = [c for c in selected_columns if c in basket_returns_display.columns]
    if not selected_columns:
        st.info("No selected baskets available for chart.")
        return

    chart_rets = basket_returns_display[selected_columns].dropna(how="all")
    common_index = chart_rets.index.intersection(benchmark_series_display.index)

    if common_index.empty:
        st.info("No overlapping dates between basket series and benchmark.")
        return

    chart_rets = chart_rets.loc[common_index]
    benchmark_rets = benchmark_series_display.loc[common_index]

    cum_pct = ((1 + chart_rets.fillna(0.0)).cumprod() - 1.0) * 100.0
    bm_cum = ((1 + benchmark_rets.fillna(0.0)).cumprod() - 1.0) * 100.0

    fig = go.Figure()

    for i, basket in enumerate(cum_pct.columns):
        fig.add_trace(go.Scatter(
            x=cum_pct.index,
            y=cum_pct[basket],
            mode="lines",
            line=dict(width=2, color=PASTEL[i % len(PASTEL)]),
            name=basket,
            hovertemplate=f"{basket}<br>Cumulative: %{{y:.1f}}%<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=bm_cum.index,
        y=bm_cum.values,
        mode="lines",
        line=dict(width=2.2, dash="dash", color="#777777"),
        name=BENCHMARK,
        hovertemplate=f"{BENCHMARK}<br>Cumulative: %{{y:.1f}}%<extra></extra>",
    ))

    fig.update_layout(
        showlegend=True,
        hovermode="x unified",
        yaxis_title="Cumulative return, %",
        title=dict(text=title, x=0, xanchor="left", y=0.96),
        margin=dict(l=10, r=10, t=42, b=10),
        xaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor", showgrid=True),
        yaxis=dict(zeroline=True, showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="left", x=0),
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_diagnostics(fetch_meta: Dict[str, Any], member_diag: pd.DataFrame, market_caps: Dict[str, float]) -> None:
    if member_diag.empty:
        return

    total_baskets = member_diag.shape[0]
    live_baskets = int((member_diag["Members Used"] > 0).sum())
    requested = int(fetch_meta.get("requested_tickers", 0))
    returned = int(fetch_meta.get("returned_tickers", 0))
    missing_px_count = int(member_diag["Missing Price"].sum())
    stale_count = int(member_diag["Stale Price"].sum())
    missing_cap_count = int(member_diag["Missing Market Cap"].sum())
    below_cap_count = int(member_diag["Below Market Cap"].sum())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Live Baskets", f"{live_baskets}/{total_baskets}")
    c2.metric("Price Coverage", f"{returned}/{requested}")
    c3.metric("Missing Price Flags", f"{missing_px_count}")
    c4.metric("Stale Price Flags", f"{stale_count}")
    c5.metric("Missing Cap Flags", f"{missing_cap_count}")
    c6.metric("Below Cap Exclusions", f"{below_cap_count}")

    source = fetch_meta.get("source", "unknown")
    last_obs = fetch_meta.get("last_observation")
    if source == "last_good_cache":
        st.warning("Yahoo returned no usable data. The app is using the last-good local cache.")
    elif last_obs:
        st.caption(f"Data source: Yahoo Finance. Last observation in price panel: {last_obs}. Market caps fetched: {len(market_caps)}.")

    with st.expander("Data Quality Diagnostics", expanded=False):
        show_cols = [
            "Category", "Defined Members", "Members Used", "Coverage %",
            "Missing Price", "Stale Price", "Missing Market Cap", "Below Market Cap",
            "Used Tickers", "Excluded Tickers"
        ]
        diag = member_diag.set_index("Basket")
        st.dataframe(diag[show_cols], use_container_width=True, height=500)

        missing_tickers = fetch_meta.get("missing_tickers", [])
        if missing_tickers:
            st.markdown("**Tickers missing from Yahoo result:**")
            st.write(", ".join(missing_tickers[:300]))


# ============================================================
# Sidebar
# ============================================================
st.title(TITLE)
st.caption(SUBTITLE)

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Internal basket monitor for leadership, breakdowns, breadth, and SPY-relative context.

        **What changed in this version**
        - Preserves every embedded ADFM basket while allowing optional external JSON overrides.
        - Shows data-quality diagnostics instead of silently dropping names.
        - Adds breadth, relative strength, beta, drawdown, and capture metrics.
        - Limits noisy charts to leaders, laggards, or selected baskets.
        - Uses a last-good local cache when Yahoo fails.
        """
    )

    st.divider()
    st.markdown("### Controls")

    today = date.today()
    preset = st.selectbox(
        "Date Range Preset",
        ["YTD", "1W", "1M", "3M", "6M", "1Y", "3Y", "5Y"],
        index=0,
    )

    cap_policy = st.selectbox(
        "Market-Cap Policy",
        [
            "Include missing market caps but flag",
            "Exclude missing market caps",
            "Do not filter by market cap",
        ],
        index=0,
    )

    min_market_cap_b = st.number_input(
        "Minimum Market Cap, $B",
        min_value=0.0,
        max_value=500.0,
        value=DEFAULT_MIN_MARKET_CAP / 1_000_000_000,
        step=0.5,
    )

    min_market_cap = None if cap_policy == "Do not filter by market cap" else min_market_cap_b * 1_000_000_000

    stale_days = st.slider("Stale Price Threshold, Days", min_value=5, max_value=90, value=30, step=5)
    min_members = st.slider("Minimum Members for Multi-Name Baskets", min_value=1, max_value=10, value=2, step=1)

    st.divider()
    st.markdown("### Chart Controls")

    chart_mode = st.selectbox(
        "Consolidated Chart Mode",
        [
            "Leaders",
            "Laggards",
            "Leaders and laggards",
            "Relative leaders vs SPY",
            "Relative laggards vs SPY",
        ],
        index=2,
    )

    max_chart_lines = st.slider("Max Chart Lines", min_value=5, max_value=40, value=16, step=1)

    render_categories = st.checkbox("Render Per-Category Panels", value=True)
    selected_categories = st.multiselect(
        "Categories to Render",
        options=list(CATEGORIES.keys()),
        default=list(CATEGORIES.keys()),
    )

    st.divider()
    with st.expander("Optional External Basket File"):
        st.markdown(
            """
            To maintain baskets outside this script, place **adfm_baskets.json** beside the app.
            The embedded taxonomy remains the fallback, so the existing baskets are preserved.
            """
        )


display_start_date = compute_display_start(preset, today)
fetch_start_date = compute_fetch_start(display_start_date)
end_date = today

DYNAMIC_LABEL = preset
display_start_ts = pd.Timestamp(display_start_date)


# ============================================================
# Fetch and compute
# ============================================================
need = {BENCHMARK, *SECONDARY_BENCHMARKS}
for tickers in ALL_BASKETS.values():
    need.update(str(t).upper() for t in tickers)

with st.spinner("Fetching price and market-cap data..."):
    levels, fetch_meta = fetch_daily_levels(
        sorted(list(need)),
        start=pd.to_datetime(fetch_start_date),
        end=pd.to_datetime(end_date) + pd.Timedelta(days=1),
    )

if levels.empty:
    st.error("No price data returned and no last-good cache was available.")
    st.stop()

if BENCHMARK not in levels.columns or levels[BENCHMARK].dropna().empty:
    st.error(f"{BENCHMARK} data is missing or empty for the selected range.")
    st.stop()

market_caps = fetch_market_caps(list(levels.columns))

members_by_basket, member_diag = build_member_diagnostics(
    levels=levels,
    categories=CATEGORIES,
    market_caps=market_caps,
    min_market_cap=min_market_cap,
    cap_policy=cap_policy,
    stale_days=stale_days,
    min_members=min_members,
)

all_basket_rets_full = ew_rets_from_members(levels=levels, members_by_basket=members_by_basket)

if all_basket_rets_full.empty:
    st.error("No baskets passed the current data-quality filters. Lower the market-cap threshold or member-count requirement.")
    render_diagnostics(fetch_meta, member_diag, market_caps)
    st.stop()

bench_rets_full = levels[BENCHMARK].pct_change().dropna()
bench_rets_display = bench_rets_full[bench_rets_full.index >= display_start_ts].copy()
all_basket_rets_display = slice_returns_for_display(all_basket_rets_full, display_start_ts)

all_panel_df = build_panel_df(
    basket_returns_full=all_basket_rets_full,
    display_start=display_start_ts,
    dynamic_label=DYNAMIC_LABEL,
    members_by_basket=members_by_basket,
    member_diag=member_diag,
    levels=levels,
    benchmark_returns_full=bench_rets_full,
)

# Keep selected categories in the table if the user narrows the render list.
if selected_categories:
    all_panel_display = all_panel_df[all_panel_df["Category"].isin(selected_categories)].copy()
else:
    all_panel_display = all_panel_df.copy()


# ============================================================
# App body
# ============================================================
render_diagnostics(fetch_meta, member_diag, market_caps)

st.subheader("All Baskets | Consolidated Panel")
render_panel(all_panel_display, DYNAMIC_LABEL)

st.subheader("All Baskets | Cumulative Performance vs SPY")

manual_selected = st.multiselect(
    "Optional: manually select baskets for the consolidated chart",
    options=all_panel_display.index.tolist(),
    default=[],
)

selected_cols = select_chart_columns(
    panel_df=all_panel_display,
    dynamic_label=DYNAMIC_LABEL,
    mode=chart_mode,
    max_lines=max_chart_lines,
    selected_baskets=manual_selected,
)

plot_cumulative_chart(
    basket_returns_display=all_basket_rets_display,
    title=f"All Baskets vs {BENCHMARK} | {chart_mode}",
    benchmark_series_display=bench_rets_display,
    selected_columns=selected_cols,
)


# ============================================================
# Per-category sections
# ============================================================
if render_categories:
    for category, baskets in CATEGORIES.items():
        if selected_categories and category not in selected_categories:
            continue

        st.markdown(f"## {category}")

        cat_names = [
            basket for basket in baskets.keys()
            if basket in all_basket_rets_full.columns and basket in all_panel_df.index
        ]

        if not cat_names:
            st.info("No data for this group under the current filters.")
            continue

        cat_panel = all_panel_df.loc[cat_names].copy()
        dyn_col = f"%{DYNAMIC_LABEL}"
        if dyn_col in cat_panel.columns:
            cat_panel = cat_panel.sort_values(dyn_col, ascending=False)

        render_panel(cat_panel, DYNAMIC_LABEL)

        cat_rets_display = slice_returns_for_display(all_basket_rets_full[cat_names], display_start_ts)
        cat_selected = select_chart_columns(
            panel_df=cat_panel,
            dynamic_label=DYNAMIC_LABEL,
            mode="Leaders and laggards",
            max_lines=min(max_chart_lines, max(5, len(cat_panel))),
            selected_baskets=None,
        )

        plot_cumulative_chart(
            basket_returns_display=cat_rets_display,
            title=f"{category} | Cumulative Performance vs {BENCHMARK}",
            benchmark_series_display=bench_rets_display,
            selected_columns=cat_selected,
        )


# ============================================================
# Basket constituents
# ============================================================
with st.expander("Basket Constituents and Live Members", expanded=False):
    if member_diag.empty:
        st.info("No member diagnostics available.")
    else:
        diag_idx = member_diag.set_index("Basket")
        for category, groups in CATEGORIES.items():
            st.markdown(f"### {category}")
            for name, tickers in groups.items():
                used = ""
                excluded = ""
                if name in diag_idx.index:
                    used = diag_idx.loc[name, "Used Tickers"]
                    excluded = diag_idx.loc[name, "Excluded Tickers"]

                st.markdown(f"**{name}**")
                st.write(f"Defined: {', '.join(sorted(set(str(t).upper() for t in tickers)))}")
                if used:
                    st.caption(f"Used: {used}")
                if excluded:
                    st.caption(f"Excluded/flagged: {excluded}")

st.caption("© 2026 AD Fund Management LP")
