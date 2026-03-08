import json
import math
from pathlib import Path
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="Basket Panels", layout="wide")

CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1550px;
    }
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: 0.1px;
    }
    .stPlotlyChart {
        background: #ffffff;
    }
    .stCaption {
        color: #6b7280;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

TITLE = "ADFM Basket Panels"
SUBTITLE = "Consolidated panel for all baskets, plus per-category panels with real trading-session alignment, stricter data quality checks, and graceful cache fallback."

PASTEL = [
    "#AEC6CF", "#FFB347", "#B39EB5", "#77DD77", "#F49AC2",
    "#CFCFC4", "#DEA5A4", "#C6E2FF", "#FFDAC1", "#E2F0CB",
    "#C7CEEA", "#FFB3BA", "#FFD1DC", "#B5EAD7", "#E7E6F7",
    "#F1E3DD", "#B0E0E6", "#E0BBE4", "#F3E5AB", "#D5E8D4"
]

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MIN_MARKET_CAP = 1_000_000_000
STRICT_MARKET_CAP_FILTER = True
STALE_DAYS = 30
PRICE_CACHE_TTL_HOURS = 6
MARKET_CAP_CACHE_TTL_HOURS = 24 * 7

MARKET_CAP_EXCEPTIONS = {
    # "UUUU", "UEC", "URG", "UROY", "DNN", "NXE", "LEU"
}

# =========================================================
# Basket config
# =========================================================
CATEGORIES: Dict[str, Dict[str, List[str]]] = {
    "Growth & Innovation": {
        "Semis ETFs": ["SMH","SOXX","XSD"],
        "Semis Compute and Accelerators": ["NVDA","AMD","INTC","ARM","AVGO","MRVL"],
        "Semis Analog and Power": ["TXN","ADI","MCHP","NXPI","MPWR","ON","STM","IFNNY","WOLF"],
        "Semis RF and Connectivity": ["QCOM","SWKS","QRVO","MTSI","AVNW"],
        "Semis Memory and Storage": ["MU","WDC","STX","SKM"],
        "Semis Foundry and OSAT": ["TSM","UMC","GFS","ASX"],
        "Semis Equipment": ["ASML","AMAT","LRCX","KLAC","TER","ONTO","AEIS","ACMR"],
        "Semis EDA and IP": ["SNPS","CDNS","ARM"],
        "Semis China and HK ADRs": ["HIMX","TSM","UMC"],
        "Semis Automotive and Industrial End-Markets": ["NXPI","ON","MCHP","STM","QCOM","ADI"],
        "Semis Test, Assembly, and Packaging": ["AMKR","TER","ONTO","AEIS","KLAC"],
        "AI Infrastructure Leaders": [
            "NVDA","AMD","AVGO","TSM","ASML","ANET","SMCI","DELL","HPE",
            "AMAT","LRCX","KLAC","TER","MRVL","MU","WDC","STX","NTAP",
            "ORCL","MSFT","AMZN","GOOGL"
        ],
        "Hyperscalers and Cloud": [
            "MSFT","AMZN","GOOGL","META","ORCL","IBM",
            "NOW","CRM","DDOG","SNOW","MDB","NET","ZS","OKTA"
        ],
        "Usage-Based and Consumption Software": ["SNOW","DDOG","MDB","NET","CRWD"],
        "Developer Tools and DevOps": ["DT","GTLB","MDB","DDOG","NET","TEAM"],
        "Observability and Data Tooling": ["DDOG","ESTC","SPLK","NEWR","MDB"],
        "Quality SaaS": ["ADBE","CRM","NOW","INTU","TEAM","HUBS","DDOG","NET","MDB","SNOW"],
        "Cybersecurity": ["PANW","FTNT","CRWD","ZS","OKTA","TENB","S","CYBR","CHKP","NET"],
        "Digital Payments": ["V","MA","PYPL","SQ","FI","FIS","GPN","AXP","COF","DFS","ADYEY","MELI"],
        "E-Commerce Platforms": ["AMZN","SHOP","MELI","ETSY","PDD","BABA","JD","SE"],
        "Social and Consumer Internet": ["META","SNAP","PINS","MTCH","GOOGL","BILI","BIDU","RBLX"],
        "Streaming and Media": ["NFLX","DIS","WBD","PARA","ROKU","SPOT","LYV","CHTR","CMCSA"],
        "Fintech and Neobanks": ["SQ","PYPL","AFRM","HOOD","SOFI","UPST","LC","ALLY","COF","DFS"],
    },

    "AI and Data Center Stack": {
        "Data Center Networking": ["ANET","CSCO","JNPR","MRVL","AVGO","CIEN","LITE","INFN"],
        "Optical and Coherent": ["CIEN","LITE","INFN","COHR","AAOI"],
        "Data Center Power and Cooling": ["VRT","ETN","TT","JCI","CARR","ABB","POWL"],
        "Data Center REITs": ["EQIX","DLR","AMT","SBAC","CCI"],
        "Servers and Storage": ["SMCI","DELL","HPE","NTAP","WDC","STX","IBM"],
        "IT Services and Systems Integrators": ["ACN","IBM","CTSH","EPAM","GIB"],
        "Rack Power Delivery and Electrical Gear": ["ETN","VRT","ABB","POWL","JCI"],
        "AI Data Center Capex Beneficiaries": ["ANET","AVGO","MRVL","ETN","VRT","SMCI","DELL","EQIX"],
    },

    "Connectivity and Industrial Tech": {
        "5G and Networking Infra": ["AMT","CCI","SBAC","ANET","CSCO","JNPR","HPE","ERIC","NOK","FFIV"],
        "Industrial Automation": ["ROK","ETN","EMR","AME","PH","ABB","FANUY","KEYS","TRMB","CGNX","IEX","ITW","GWW","SYM"],
        "Aerospace Tech and Space": ["RKLB","IRDM","ASTS","LHX","LMT","NOC","RTX"],
        "Defense Software and ISR": ["PLTR","KTOS","AVAV","LHX","LDOS","BAH"],
        "Aerospace Aftermarket and MRO Exposure": ["GE","RTX","LMT","HEI","TDG"],
        "Electrification and Grid Modernization": ["ETN","ABB","PWR","MYRG","POWL","VRT"],
    },

    "Energy and Hard Assets": {
        "Energy Majors": ["XOM","CVX","COP","SHEL","BP","TTE","EQNR","ENB","PBR"],
        "US Shale and E&Ps": ["EOG","DVN","FANG","MRO","OXY","APA","AR","RRC","SWN","CHK","CTRA"],
        "Natural Gas and LNG": ["LNG","EQNR","KMI","WMB","EPD","ET","TELL"],
        "Oilfield Services": ["SLB","HAL","BKR","NOV","CHX","FTI","PTEN","HP","NBR","OII"],
        "Uranium and Fuel Cycle": ["CCJ","UUUU","UEC","URG","UROY","DNN","NXE","LEU","URA","URNM"],
        "Nuclear and Grid Buildout": ["VST","CEG","BWXT","LEU","ETN","VRT"],
        "Metals and Mining": ["BHP","RIO","VALE","FCX","NEM","TECK","SCCO","AA"],
        "Gold and Silver Miners": ["GDX","GDXJ","NEM","AEM","GOLD","KGC","AG","PAAS","WPM"],
        "Energy Midstream and Storage": ["KMI","WMB","EPD","ET","ENB","MPLX"],
        "Refining and Downstream": ["MPC","VLO","PSX","DK","PBF"],
        "Oil-Weighted E&Ps": ["EOG","FANG","OXY","DVN","MRO"],
        "Gas-Weighted E&Ps": ["AR","RRC","SWN","CTRA","CHK"],
        "LNG Export Levered": ["LNG","KMI","WMB","ET","EPD"],
    },

    "Clean Energy Transition": {
        "Solar and Inverters": ["TAN","FSLR","ENPH","SEDG","RUN","CSIQ","JKS"],
        "Wind and Renewables": ["ICLN","FAN","AY","NEP","FSLR"],
        "Hydrogen": ["PLUG","BE","BLDP"],
        "Utilities and Power": ["VST","CEG","NEE","DUK","SO","AEP","XEL","EXC","PCG","EIX","ED"],
        "Regulated Utilities Core": ["DUK","SO","AEP","XEL","EXC","ED"],
        "Merchant Power and Volatility": ["VST","CEG","NRG","AES","CWEN"],
    },

    "Health and Longevity": {
        "Large-Cap Biotech": ["AMGN","GILD","REGN","BIIB","VRTX","ILMN"],
        "GLP-1 and Metabolic": ["NVO","LLY","AZN","MRK","PFE"],
        "MedTech Devices": ["MDT","SYK","ISRG","BSX","ZBH","EW","PEN"],
        "Diagnostics and Tools": ["TMO","DHR","A","RGEN","ILMN"],
        "Healthcare Payers": ["UNH","HUM","CI","ELV","CNC","MOH"],
        "Healthcare Providers and Hospitals": ["HCA","THC","UHS","CYH"],
        "Healthcare Services and Outsourcing": ["LH","DGX","AMN","EHC"],
        "Drug Channel and PBM Exposure": ["CVS","UNH","CI","ELV"],
        "Life Science Tools and Supply Chain": ["TMO","DHR","A","RGEN","WAT"],
        "CRO and Clinical Services": ["IQV","SYK","TMO","LH","DGX"],
    },

    "Financials and Credit": {
        "Money-Center and IBs": ["JPM","BAC","C","WFC","GS","MS"],
        "Regional Banks": ["KRE","TFC","FITB","CFG","RF","KEY","PNC","USB","MTB"],
        "Brokers and Exchanges": ["IBKR","SCHW","HOOD","CME","ICE","NDAQ","CBOE","MKTX"],
        "Alt Managers and PE": ["BX","KKR","APO","CG","ARES","OWL","TPG"],
        "Credit and Specialty Finance": ["MS","GS","BX","KKR","ARES","MAIN","ARCC"],
        "Mortgage Finance": ["RKT","UWMC","COOP","FNF","NMIH","ESNT"],
        "Insurers": ["BRK-B","CB","TRV","PGR","AIG","MET"],
        "Insurance P&C": ["PGR","TRV","CB","ALL","CINF"],
        "Insurance Life and Retirement": ["MET","PRU","LNC","AIG"],
        "Reinsurers": ["RNR","RE","EG"],
        "Insurance Brokers": ["AJG","BRO","MMC","AON","WTW"],
        "Deposit Beta and Funding Sensitivity": ["WFC","BAC","USB","PNC","KEY"],
        "CRE Sensitivity Banks": ["KEY","CFG","NYCB","ZION","WAL"],
        "Card and Consumer Lenders": ["AXP","COF","DFS","SYF","ALLY"],
        "Capital Markets Activity Proxies": ["GS","MS","CME","ICE","NDAQ"],
    },

    "Real Assets and Inflation Beneficiaries": {
        "Homebuilders": ["ITB","DHI","LEN","NVR","PHM","TOL","KBH","MTH"],
        "REITs Core": ["VNQ","PLD","EQIX","SPG","O","PSA","DLR","ARE","VTR","WELL"],
        "Industrials and Infrastructure": ["CAT","DE","URI","PWR","VMC","MLM","NUE"],
        "Shipping and Logistics": ["FDX","UPS","GXO","XPO","ZIM","MATX","DAC"],
        "Agriculture and Machinery": ["MOS","NTR","DE","CNHI","ADM","BG","CF","AGCO"],
        "Housing Home Improvement and Repair": ["HD","LOW","TSCO","POOL"],
        "Housing Building Products and Materials": ["BLDR","TREX","MAS","VMC","MLM","SUM"],
        "Housing Mortgage and Title": ["COOP","RKT","UWMC","FNF","FAF"],
        "Housing Residential Transaction Proxies": ["RDFN","ZG","OPEN"],
        "Net Lease and Long Lease Duration REITs": ["O","NNN","ADC","WPC"],
        "Multifamily REITs": ["AVB","EQR","UDR","ESS","MAA"],
        "Office and Refi Wall Sensitive REITs": ["BXP","VNO","SLG","KRC"],
        "Construction Materials and Aggregates": ["VMC","MLM","SUM","NUE","EXP"],
    },

    "Consumer Cyclicals": {
        "Retail Discretionary": ["HD","LOW","M","GPS","BBY","TJX","TGT","ROST"],
        "Restaurants": ["MCD","SBUX","YUM","CMG","DRI","DPZ","WING","QSR"],
        "Travel and Booking": ["BKNG","EXPE","ABNB","TRIP"],
        "Hotels and Casinos": ["MAR","HLT","IHG","MGM","LVS","WYNN","MLCO","CZR","PENN"],
        "Airlines": ["AAL","DAL","UAL","LUV","JBLU","ALK"],
        "Autos Legacy OEMs": ["TM","HMC","F","GM","STLA"],
        "Electric Vehicles": ["TSLA","RIVN","LCID","NIO","LI","XPEV"],
        "Luxury and Apparel": ["TPR","RL","CPRI","LVMUY"],
        "Retail Asset-Heavy Inventory Risk": ["WMT","TGT","COST","BBY","M","GPS","KSS","BBWI"],
        "Retail Asset-Light Platforms and Marketplaces": ["AMZN","EBAY","ETSY","SHOP","PDD","MELI"],
        "Prime Consumer Discretionary": ["COST","HD","LOW","LULU","NKE"],
        "Subprime and Credit-Sensitive Consumer": ["AFRM","UPST","COF","DFS","SYF"],
        "Big Ticket Durables": ["WHR","TPX","BBY","RH","LOW"],
        "Auto Parts and Service Tailwind": ["AZO","ORLY","LKQ","AAP"],
    },

    "Defensives and Staples": {
        "Retail Staples": ["WMT","COST","TGT","DG","KR","WBA"],
        "Staples and Beverages": ["PG","KO","PEP","PM","MO","MDLZ"],
        "Telecom and Cable": ["T","VZ","TMUS","CHTR","CMCSA"],
        "Aerospace and Defense": ["LMT","NOC","RTX","GD","HII","TDG","HEI"],
        "Utilities Defensive": ["DUK","SO","AEP","XEL","EXC","ED"],
        "Beverage Bottlers and Distributors": ["KOF","COKE","FIZZ","KO"],
        "Pricing Power Staples": ["PG","KO","PEP","COST","MDLZ"],
        "Volume Risk and Trade-Down Staples": ["WMT","DG","DLTR","KR"],
    },

    "Materials and Chemicals": {
        "Fertilizers": ["CF","MOS","NTR"],
        "Commodity Chemicals": ["DOW","LYB","CE"],
        "Specialty Chemicals": ["SHW","EMN","IFF","PPG","RPM"],
        "Industrial Gases": ["LIN","APD","AIQUY"],
        "Packaging and Containers": ["IP","PKG","WRK"],
        "Construction Chemicals and Adhesives": ["EMN","RPM","SHW"],
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
        "Crypto Proxies": ["COIN","MSTR","MARA","RIOT","BITO"],
        "China Tech ADRs": ["BABA","BIDU","JD","PDD","BILI","NTES","TCEHY"],
        "EM Internet and Commerce": ["MELI","SE","NU","STNE"],
    },

    "Regime Diagnostics": {
        "Long-Duration Equities": ["ARKK","IPO","IGV","SNOW","NET","DDOG","MDB","SHOP"],
        "Short-Duration Cash Flow": ["BRK-B","PGR","CB","ICE","CME","NDAQ","SPGI","MSCI"],
        "Yield Proxies": ["XLU","VZ","T","KMI","EPD","ENB"],
        "Rate-Sensitive Cyclicals": ["ITB","XHB","CVNA","COF","DFS","AXP","SYF"],
        "Labor-Intensive Services": ["SBUX","CMG","DRI","MAR","HLT","RCL","CCL"],
        "Automation and Productivity Winners": ["ROK","ABB","ETN","PH","CGNX","ISRG","SYM","TER"],
        "IT Services and Outsourcing": ["ACN","IBM","CTSH","EPAM","GIB"],
        "Staffing and Wage-Sensitive Names": ["RHI","MAN","KFY","ASGN"],
        "Leveraged Cyclicals": ["CCL","RCL","NCLH","AAL","UAL","DAL","MGM","LVS"],
        "Net-Cash Compounders": ["AAPL","MSFT","GOOGL","META","ORCL","ADBE","INTU","V"],
        "Equity Credit Stress Proxies": ["HYG","JNK","LQD"],
        "Financial Conditions Sensitive": ["IWM","XLY","KRE","HYG","ARKK"],
        "Dollar-Up Winners": ["XLK","XLC","XLY","IYT"],
        "Dollar-Down Beneficiaries": ["XME","GDX","EEM","EWZ"],
        "Commodity FX Equities": ["EWC","EWA","EWZ","EWW"],
        "EM Domestic Demand": ["EEM","INDA","EWW","EWZ","EIDO"],
        "High Multiple Duration Risk": ["SNOW","NET","DDOG","MDB","SHOP"],
        "Buyback and Cash Yield Winners": ["AAPL","MSFT","BRK-B","XOM","META"],
        "High Short Interest and Retail Flow": ["CVNA","UPST","RIVN","BYND"],
        "Spread Beta Equities": ["COF","DFS","SYF","OMF","ENVA"],
        "Inflation Pass-Through Winners": ["COST","HD","SHW","NUE","VMC"],
        "Input Cost and Margin Pressure Risk": ["DAL","UAL","SBUX","DPZ","WHR"],
    },

    "Sector Expansions": {
        "Transportation Rails and Trucking": ["UNP","CSX","NSC","CNI","CP","JBHT","KNX"],
        "Transportation Parcel and Last-Mile": ["FDX","UPS","GXO","XPO"],
        "Air Cargo and Leasing": ["AL","FTAI","AER"],
        "Commodity Chemicals": ["DOW","LYB","CE"],
        "Specialty Chemicals": ["SHW","EMN","IFF","PPG"],
        "Industrial Gases": ["LIN","APD","AIQUY"],
        "Digital Advertising Platforms": ["GOOGL","META","TTD","PINS","SNAP"],
        "Traditional Media and Content": ["DIS","WBD","PARA","CHTR","CMCSA"],
        "Marketing Services and Agencies": ["IPG","OMC","WPP"],
        "Gov IT and Services": ["SAIC","CACI","LDOS","BAH","PSN","G"],
        "Public-Sector Systems Integration": ["ACN","IBM","CTSH","EPAM"],
        "Defense and Security Spending": ["LMT","NOC","RTX","GD","LHX","HII","TDG"],
        "Infrastructure and Grid Spend": ["PWR","ETN","VRT","ABB","CAT","URI","VMC","MLM"],
        "Healthcare Policy Sensitive": ["UNH","HUM","CI","ELV","CNC","CVS"],
        "Energy Subsidy and Transition Plays": ["FSLR","ENPH","BE","PLUG","NEE","VST","ICLN"],
    },

    "Everyday Economy": {
        "Recreation and Experiences": ["YETI","FOXF","ASO","DOO","PLAY","LYV","FUN","RICK"],
        "Deferred Durables and Home": ["SGI","SNBR","WHR","POOL","LOW","TTC","LAD"],
        "Deferred Healthcare": ["ALGN","EYE","WRBY","HSIC"],
        "Debt and Credit Paydown": ["OMF","CACC","SYF","COF","OPFI","ENVA"],
        "Trade-Down Retail and Off-Price": ["TJX","ROST","BURL","FIVE","OLLI"],
        "Discount and Dollar": ["DG","DLTR","BURL"],
        "Staple Volume and Clubs": ["WMT","COST","KR"],
        "Value QSR": ["MCD","YUM","QSR","WEN","DPZ"],
        "Auto Parts and Repair": ["AZO","ORLY","AAP","LKQ"],
        "Home Repair and Maintenance": ["HD","LOW","POOL","TSCO"],
        "Used Auto and Affordability": ["KMX","CVNA","LAD"],
        "Shelter and Rent Economy": ["INVH","AMH","AVB","EQR","UDR","MAA","ESS"],
        "Manufactured Housing Affordability": ["ELS","SUI","CUBE"],
        "Storage and Mobility Stress": ["PSA","EXR","CUBE"],
        "Freight and Parcels": ["UNP","CSX","NSC","JBHT","KNX","SAIA","ODFL","FDX","UPS","CHRW","XPO","GXO"],
        "Consumer Credit Stress": ["SYF","COF","DFS","ALLY","OMF","ENVA"],
        "Payroll and Staffing": ["ADP","PAYX","RHI","MAN","KFY","ASGN"],
        "Budget Hotels and Value Travel": ["CHH","WH","RYAAY"],
        "Chemicals Feedstock Sensitivity": ["DOW","LYB","WLK","OLN","CF","NTR","MOS"],
    },
}

ALL_BASKETS = {basket: tickers for category in CATEGORIES.values() for basket, tickers in category.items()}


# =========================================================
# Utility helpers
# =========================================================
def safe_ticker(t: str) -> str:
    return str(t).strip().upper()


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = safe_ticker(item)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def chunked(items: List[str], n: int) -> List[List[str]]:
    n = max(1, int(n))
    return [items[i:i + n] for i in range(0, len(items), n)]


def cache_is_fresh(path: Path, ttl_hours: int) -> bool:
    if not path.exists():
        return False
    age_hours = (pd.Timestamp.utcnow().timestamp() - path.stat().st_mtime) / 3600
    return age_hours <= ttl_hours


def save_dataframe_cache(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_pickle(path)
    except Exception:
        pass


def load_dataframe_cache(path: Path) -> pd.DataFrame:
    try:
        return pd.read_pickle(path)
    except Exception:
        return pd.DataFrame()


def save_json_cache(obj: dict, path: Path) -> None:
    try:
        path.write_text(json.dumps(obj), encoding="utf-8")
    except Exception:
        pass


def load_json_cache(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# =========================================================
# Data fetch
# =========================================================
def normalize_close_frame(raw: pd.DataFrame, batch: List[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        target_col = "Close"
        if target_col not in set(level0):
            return pd.DataFrame()
        out = raw[target_col].copy()
        out.columns = [safe_ticker(c) for c in out.columns]
        out.index = pd.to_datetime(out.index).tz_localize(None)
        return out.sort_index()

    if "Close" in raw.columns:
        sym = safe_ticker(batch[0])
        out = raw[["Close"]].rename(columns={"Close": sym}).copy()
        out.index = pd.to_datetime(out.index).tz_localize(None)
        return out.sort_index()

    return pd.DataFrame()


def download_close_batch(batch: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        raw = yf.download(
            tickers=batch,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column"
        )
        return normalize_close_frame(raw, batch)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=PRICE_CACHE_TTL_HOURS * 3600)
def fetch_daily_levels_live(tickers: List[str], start: pd.Timestamp, end: pd.Timestamp, chunk_size: int = 40) -> pd.DataFrame:
    tickers = dedupe_keep_order(tickers)
    frames = []
    for batch in chunked(tickers, chunk_size):
        df = download_close_batch(batch, start, end)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out = out.sort_index()
    return out


def fetch_daily_levels_with_fallback(tickers: List[str], start: pd.Timestamp, end: pd.Timestamp, cache_key: str) -> Tuple[pd.DataFrame, str]:
    cache_path = CACHE_DIR / f"{cache_key}_levels.pkl"

    live = fetch_daily_levels_live(tickers, start, end)
    if not live.empty:
        save_dataframe_cache(live, cache_path)
        return live, "live"

    cached = load_dataframe_cache(cache_path)
    if not cached.empty:
        cached = cached[(cached.index >= start) & (cached.index <= end)]
        if not cached.empty:
            return cached, "cache"

    return pd.DataFrame(), "none"


@st.cache_data(show_spinner=False, ttl=MARKET_CAP_CACHE_TTL_HOURS * 3600)
def fetch_market_caps_live(tickers: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    tickers = dedupe_keep_order(tickers)
    for sym in tickers:
        try:
            tk = yf.Ticker(sym)
            fi = getattr(tk, "fast_info", None)
            mc = None

            if fi is not None:
                if isinstance(fi, dict):
                    mc = fi.get("market_cap")
                else:
                    mc = getattr(fi, "market_cap", None)

            if mc is None:
                info = tk.info
                mc = info.get("marketCap")

            if mc is not None and not pd.isna(mc):
                out[sym] = float(mc)
        except Exception:
            continue

    return out


def fetch_market_caps_with_fallback(tickers: List[str], cache_key: str) -> Tuple[Dict[str, float], str]:
    cache_path = CACHE_DIR / f"{cache_key}_market_caps.json"

    live = fetch_market_caps_live(tickers)
    if live:
        save_json_cache(live, cache_path)
        return live, "live"

    cached = load_json_cache(cache_path)
    if cached:
        return {safe_ticker(k): float(v) for k, v in cached.items()}, "cache"

    return {}, "none"


# =========================================================
# Analytics
# =========================================================
def build_master_calendar(levels: pd.DataFrame, benchmark: str = "SPY") -> pd.DatetimeIndex:
    benchmark = safe_ticker(benchmark)
    if benchmark in levels.columns:
        idx = levels[benchmark].dropna().index
        if len(idx) > 0:
            return pd.DatetimeIndex(idx)
    return pd.DatetimeIndex(levels.dropna(how="all").index)


def align_levels_to_master(levels: pd.DataFrame, master_index: pd.DatetimeIndex) -> pd.DataFrame:
    if levels.empty or len(master_index) == 0:
        return pd.DataFrame()
    aligned = levels.reindex(master_index)
    return aligned


def compute_raw_returns_from_levels(levels: pd.DataFrame) -> pd.DataFrame:
    if levels.empty:
        return pd.DataFrame()
    return levels.sort_index().pct_change()


def valid_constituents_for_basket(
    levels: pd.DataFrame,
    tickers: List[str],
    market_caps: Optional[Dict[str, float]],
    min_market_cap: Optional[float],
    strict_market_cap: bool,
    stale_days: int
) -> List[str]:
    cols = []
    if levels.empty:
        return cols

    last_dt = levels.dropna(how="all").index.max()
    for ticker in dedupe_keep_order(tickers):
        if ticker not in levels.columns:
            continue

        s = levels[ticker].dropna()
        if s.empty:
            continue

        if last_dt - s.index.max() > pd.Timedelta(days=stale_days):
            continue

        if min_market_cap is not None and ticker not in MARKET_CAP_EXCEPTIONS:
            mc = None if market_caps is None else market_caps.get(ticker)

            if strict_market_cap:
                if mc is None or pd.isna(mc) or float(mc) < min_market_cap:
                    continue
            else:
                if mc is not None and not pd.isna(mc) and float(mc) < min_market_cap:
                    continue

        cols.append(ticker)

    return cols


def compute_basket_returns(
    levels: pd.DataFrame,
    baskets: Dict[str, List[str]],
    market_caps: Optional[Dict[str, float]],
    min_market_cap: Optional[float],
    strict_market_cap: bool,
    stale_days: int,
    master_index: pd.DatetimeIndex
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    if levels.empty:
        return pd.DataFrame(), {}

    raw_rets = compute_raw_returns_from_levels(levels)
    raw_rets = raw_rets.reindex(master_index)

    basket_ret_map: Dict[str, pd.Series] = {}
    basket_members: Dict[str, List[str]] = {}

    for basket_name, tickers in baskets.items():
        members = valid_constituents_for_basket(
            levels=levels,
            tickers=tickers,
            market_caps=market_caps,
            min_market_cap=min_market_cap,
            strict_market_cap=strict_market_cap,
            stale_days=stale_days
        )

        if len(members) == 0:
            continue

        member_rets = raw_rets[members].copy()

        if member_rets.empty:
            continue

        basket_series = member_rets.mean(axis=1, skipna=True, numeric_only=True)
        basket_series = basket_series.where(member_rets.notna().sum(axis=1) > 0)

        if basket_series.dropna().shape[0] == 0:
            continue

        basket_ret_map[basket_name] = basket_series
        basket_members[basket_name] = members

    if not basket_ret_map:
        return pd.DataFrame(), {}

    basket_rets = pd.DataFrame(basket_ret_map).sort_index()
    return basket_rets, basket_members


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return pd.Series(index=series.index, dtype=float)

    delta = s.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.reindex(series.index)


def macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return pd.Series(index=series.index, dtype=float)

    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return hist.reindex(series.index)


def ema_regime(series: pd.Series, e1: int = 4, e2: int = 9, e3: int = 18) -> str:
    s = series.dropna()
    if len(s) < max(e1, e2, e3):
        return "Neutral"

    a = s.ewm(span=e1, adjust=False).mean().iloc[-1]
    b = s.ewm(span=e2, adjust=False).mean().iloc[-1]
    c = s.ewm(span=e3, adjust=False).mean().iloc[-1]

    if a > b > c:
        return "Up"
    if a < b < c:
        return "Down"
    return "Neutral"


def momentum_label(hist: pd.Series, lookback: int = 5, z_window: int = 63) -> str:
    h = hist.dropna()
    if len(h) < max(lookback + 1, z_window):
        return "Neutral"

    latest = h.iloc[-1]
    prior = h.iloc[-(lookback + 1)]

    if latest > 0:
        base = "Positive"
    elif latest < 0:
        base = "Negative"
    else:
        return "Neutral"

    window = h.iloc[-z_window:]
    std = window.std(ddof=0)
    z = 0.0 if std in (0, np.nan) or pd.isna(std) else (latest - window.mean()) / std

    accel = "Accelerating" if latest > prior else "Decelerating"
    strength = "Strong" if abs(z) >= 1 else "Weak"
    return f"{base} | {accel} | {strength}"


def realized_vol(returns: pd.Series, days: int = 63, ann: int = 252) -> float:
    r = returns.dropna().iloc[-days:]
    if r.empty:
        return np.nan
    return float(r.std(ddof=0) * math.sqrt(ann) * 100.0)


def pct_over_trading_days(levels: pd.Series, trading_days: int) -> float:
    s = levels.dropna()
    if len(s) <= trading_days:
        return np.nan
    return float((s.iloc[-1] / s.iloc[-(trading_days + 1)]) - 1.0)


def pct_since_date(levels: pd.Series, start_ts: pd.Timestamp) -> float:
    s = levels.dropna()
    if s.empty:
        return np.nan
    sub = s[s.index >= start_ts]
    if len(sub) < 2:
        return np.nan
    return float((sub.iloc[-1] / sub.iloc[0]) - 1.0)


def first_valid_on_or_after(index: pd.DatetimeIndex, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    arr = index[index >= ts]
    return arr[0] if len(arr) else None


def build_panel_df(
    basket_returns: pd.DataFrame,
    ref_start: pd.Timestamp,
    dynamic_label: str,
    benchmark_series: Optional[pd.Series],
    basket_members: Dict[str, List[str]]
) -> pd.DataFrame:
    cols = [
        "Basket", "Members", "%5D", "%21D", f"%{dynamic_label}",
        "RSI(14D)", "MACD Momentum", "EMA 4/9/18", "RSI(14W)",
        "3M RVOL", "RVOL / SPY", "Corr(63D)"
    ]

    if basket_returns.empty:
        return pd.DataFrame(columns=cols).set_index("Basket")

    levels = 100 * (1 + basket_returns.fillna(0)).cumprod()

    spy_rv = realized_vol(benchmark_series, 63, 252) if benchmark_series is not None else np.nan
    out = []

    for basket in levels.columns:
        s = levels[basket].dropna()
        if len(s) < 15:
            continue

        r5d = pct_over_trading_days(s, 5)
        r21d = pct_over_trading_days(s, 21)

        start_valid = first_valid_on_or_after(s.index, ref_start)
        r_dyn = pct_since_date(s, start_valid) if start_valid is not None else np.nan

        rsi_14d = rsi(s, 14).dropna()
        rsi_14d_last = rsi_14d.iloc[-1] if not rsi_14d.empty else np.nan

        weekly = s.resample("W-FRI").last().dropna()
        rsi_14w = rsi(weekly, 14).dropna()
        rsi_14w_last = rsi_14w.iloc[-1] if not rsi_14w.empty else np.nan

        hist = macd_hist(s)
        macd_m = momentum_label(hist)

        ema_tag = ema_regime(s, 4, 9, 18)

        rv = realized_vol(basket_returns[basket], 63, 252)
        rv_rel = rv / spy_rv if pd.notna(rv) and pd.notna(spy_rv) and spy_rv != 0 else np.nan

        corr_spy = np.nan
        if benchmark_series is not None:
            merged = pd.concat([basket_returns[basket], benchmark_series], axis=1).dropna()
            if len(merged) >= 80:
                corr_spy = float(merged.iloc[:, 0].rolling(63).corr(merged.iloc[:, 1]).iloc[-1])

        out.append({
            "Basket": basket,
            "Members": len(basket_members.get(basket, [])),
            "%5D": round(r5d * 100, 1) if pd.notna(r5d) else np.nan,
            "%21D": round(r21d * 100, 1) if pd.notna(r21d) else np.nan,
            f"%{dynamic_label}": round(r_dyn * 100, 1) if pd.notna(r_dyn) else np.nan,
            "RSI(14D)": round(rsi_14d_last, 1) if pd.notna(rsi_14d_last) else np.nan,
            "MACD Momentum": macd_m,
            "EMA 4/9/18": ema_tag,
            "RSI(14W)": round(rsi_14w_last, 1) if pd.notna(rsi_14w_last) else np.nan,
            "3M RVOL": round(rv, 1) if pd.notna(rv) else np.nan,
            "RVOL / SPY": round(rv_rel, 2) if pd.notna(rv_rel) else np.nan,
            "Corr(63D)": round(corr_spy, 2) if pd.notna(corr_spy) else np.nan,
        })

    if not out:
        return pd.DataFrame(columns=cols).set_index("Basket")

    df = pd.DataFrame(out).set_index("Basket")
    sort_col = f"%{dynamic_label}"
    if sort_col in df.columns:
        df = df.sort_values(by=sort_col, ascending=False)
    return df


def build_overlap_table(basket_members: Dict[str, List[str]]) -> pd.DataFrame:
    counts: Dict[str, int] = {}
    for members in basket_members.values():
        for m in members:
            counts[m] = counts.get(m, 0) + 1

    if not counts:
        return pd.DataFrame(columns=["Ticker", "Basket Count"])

    df = pd.DataFrame({"Ticker": list(counts.keys()), "Basket Count": list(counts.values())})
    return df.sort_values(["Basket Count", "Ticker"], ascending=[False, True]).reset_index(drop=True)


def build_category_summary(panel_df: pd.DataFrame, dynamic_col: str) -> str:
    if panel_df.empty or dynamic_col not in panel_df.columns:
        return "No valid baskets for this category."

    leaders = panel_df.head(2)
    laggards = panel_df.tail(2).sort_values(dynamic_col, ascending=True)

    up_count = int((panel_df["EMA 4/9/18"] == "Up").sum()) if "EMA 4/9/18" in panel_df.columns else 0
    down_count = int((panel_df["EMA 4/9/18"] == "Down").sum()) if "EMA 4/9/18" in panel_df.columns else 0

    leader_text = ", ".join([f"{idx} ({row[dynamic_col]:.1f}%)" for idx, row in leaders.iterrows()])
    laggard_text = ", ".join([f"{idx} ({row[dynamic_col]:.1f}%)" for idx, row in laggards.iterrows()])

    return (
        f"Leadership is concentrated in {leader_text}. "
        f"Pressure is most visible in {laggard_text}. "
        f"EMA regime breadth stands at {up_count} up-trending baskets versus {down_count} down-trending baskets."
    )


# =========================================================
# Styling
# =========================================================
def ret_bg(v):
    if pd.isna(v):
        return ""
    if v >= 0:
        alpha = min(abs(v) / 20.0, 1.0)
        return f"background-color: rgba(34, 197, 94, {0.12 + 0.23 * alpha});"
    alpha = min(abs(v) / 20.0, 1.0)
    return f"background-color: rgba(239, 68, 68, {0.12 + 0.23 * alpha});"


def rsi_bg(v):
    if pd.isna(v):
        return ""
    if v >= 70:
        return "background-color: rgba(245, 158, 11, 0.22);"
    if v <= 30:
        return "background-color: rgba(239, 68, 68, 0.18);"
    return "background-color: rgba(59, 130, 246, 0.08);"


def label_bg(v):
    if not isinstance(v, str):
        return ""
    if v == "Up":
        return "background-color: rgba(34, 197, 94, 0.20);"
    if v == "Down":
        return "background-color: rgba(239, 68, 68, 0.18);"
    if v.startswith("Positive"):
        return "background-color: rgba(34, 197, 94, 0.14);"
    if v.startswith("Negative"):
        return "background-color: rgba(239, 68, 68, 0.12);"
    return "background-color: rgba(59, 130, 246, 0.08);"


def vol_bg(v):
    if pd.isna(v):
        return ""
    if v < 20:
        return "background-color: rgba(59, 130, 246, 0.08);"
    if v < 40:
        return "background-color: rgba(59, 130, 246, 0.14);"
    if v < 60:
        return "background-color: rgba(59, 130, 246, 0.20);"
    return "background-color: rgba(37, 99, 235, 0.26);"


def corr_bg(v):
    if pd.isna(v):
        return ""
    alpha = min(abs(v), 1.0) * 0.25
    return f"background-color: rgba(99, 102, 241, {alpha});"


def style_panel_df(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    if df.empty:
        return df.style

    styler = df.style
    for col in [c for c in df.columns if c.startswith("%")]:
        styler = styler.map(ret_bg, subset=[col])

    for col in ["RSI(14D)", "RSI(14W)"]:
        if col in df.columns:
            styler = styler.map(rsi_bg, subset=[col])

    for col in ["MACD Momentum", "EMA 4/9/18"]:
        if col in df.columns:
            styler = styler.map(label_bg, subset=[col])

    for col in ["3M RVOL", "RVOL / SPY"]:
        if col in df.columns:
            styler = styler.map(vol_bg, subset=[col])

    if "Corr(63D)" in df.columns:
        styler = styler.map(corr_bg, subset=["Corr(63D)"])

    format_map = {
        "Members": "{:.0f}",
        "%5D": "{:.1f}",
        "%21D": "{:.1f}",
        "RSI(14D)": "{:.1f}",
        "RSI(14W)": "{:.1f}",
        "3M RVOL": "{:.1f}",
        "RVOL / SPY": "{:.2f}",
        "Corr(63D)": "{:.2f}",
    }
    for col in [c for c in df.columns if c.startswith("%") and c not in ("%5D", "%21D")]:
        format_map[col] = "{:.1f}"

    styler = styler.format(format_map).set_properties(
        **{"font-size": "12px", "white-space": "nowrap"}
    )
    return styler


# =========================================================
# Charts
# =========================================================
def plot_cumulative_chart(
    basket_returns: pd.DataFrame,
    title: str,
    benchmark_series: pd.Series,
    default_show_count: int = 12
) -> None:
    if basket_returns.empty or benchmark_series.dropna().empty:
        st.info("Insufficient data to render chart.")
        return

    common_index = basket_returns.index.intersection(benchmark_series.index)
    if len(common_index) == 0:
        st.info("No overlapping dates between baskets and benchmark.")
        return

    basket_returns = basket_returns.loc[common_index]
    benchmark_series = benchmark_series.loc[common_index]

    cum_pct = ((1 + basket_returns.fillna(0)).cumprod() - 1.0) * 100
    bm_cum = ((1 + benchmark_series.fillna(0)).cumprod() - 1.0) * 100

    final_rank = cum_pct.iloc[-1].sort_values(ascending=False)
    visible_names = set(final_rank.head(default_show_count).index)

    fig = go.Figure()

    for i, basket in enumerate(final_rank.index):
        fig.add_trace(go.Scatter(
            x=cum_pct.index,
            y=cum_pct[basket],
            mode="lines",
            name=basket,
            visible=True if basket in visible_names else "legendonly",
            line=dict(width=2, color=PASTEL[i % len(PASTEL)]),
            hovertemplate=f"{basket}<br>Cum Return: %{{y:.1f}}%<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=bm_cum.index,
        y=bm_cum.values,
        mode="lines",
        name="SPY",
        line=dict(width=2.25, dash="dash", color="#6b7280"),
        hovertemplate="SPY<br>Cum Return: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left"),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=42, b=8),
        yaxis_title="Cumulative Return %",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        xaxis=dict(showgrid=True, rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, zeroline=True, zerolinecolor="rgba(120,120,120,0.25)"),
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# Sidebar controls
# =========================================================
st.title(TITLE)
st.caption(SUBTITLE)

with st.sidebar:
    st.markdown("### About This Tool")
    st.write(
        "Equal-weight basket monitor built on daily closes, with stricter ticker eligibility, real benchmark-session alignment, "
        "and cache fallback so transient Yahoo issues do not take the whole app down."
    )
    st.write(
        "Metrics shown: %5D, %21D, preset return, RSI(14D), RSI(14W), MACD momentum, EMA 4/9/18 regime, "
        "63D realized vol, RVOL relative to SPY, and rolling 63D correlation to SPY."
    )
    st.divider()
    st.markdown("### Controls")

    today = date.today()
    PRESETS = {
        "YTD": lambda t: date(t.year, 1, 1),
        "1W": lambda t: t - timedelta(days=7),
        "1M": lambda t: t - timedelta(days=30),
        "3M": lambda t: t - timedelta(days=90),
        "1Y": lambda t: t - timedelta(days=365),
        "3Y": lambda t: t - timedelta(days=365 * 3),
        "5Y": lambda t: t - timedelta(days=365 * 5),
    }

    preset = st.selectbox("Date Range Preset", list(PRESETS.keys()), index=0)
    start_date = PRESETS[preset](today)
    end_date = today

    strict_cap_filter_ui = st.checkbox("Strict market cap filter", value=STRICT_MARKET_CAP_FILTER)
    show_overlap = st.checkbox("Show overlap diagnostic", value=True)
    show_constituents = st.checkbox("Show basket constituents", value=False)
    default_chart_count = st.slider("Default visible lines per chart", min_value=6, max_value=25, value=12)

LABEL_MAP = {
    "YTD": "YTD",
    "1W": "1W",
    "1M": "1M",
    "3M": "3M",
    "1Y": "1Y",
    "3Y": "3Y",
    "5Y": "5Y",
}
DYNAMIC_LABEL = LABEL_MAP.get(preset, "YTD")


# =========================================================
# Data load
# =========================================================
BENCHMARK = "SPY"
all_needed = {BENCHMARK}
for tickers in ALL_BASKETS.values():
    all_needed.update(safe_ticker(t) for t in tickers)

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)

cache_key = f"{preset}_{start_date.isoformat()}_{end_date.isoformat()}"

levels_raw, levels_source = fetch_daily_levels_with_fallback(
    tickers=sorted(all_needed),
    start=start_ts,
    end=end_ts,
    cache_key=cache_key
)

if levels_raw.empty:
    st.error("No price data available. Live fetch failed and no usable last-good cache was found.")
    st.stop()

market_caps, cap_source = fetch_market_caps_with_fallback(
    tickers=list(levels_raw.columns),
    cache_key="global_universe"
)

master_index = build_master_calendar(levels_raw, benchmark=BENCHMARK)
if len(master_index) == 0:
    st.error("Unable to construct a valid trading-session calendar from benchmark data.")
    st.stop()

levels = align_levels_to_master(levels_raw, master_index)

if BENCHMARK not in levels.columns or levels[BENCHMARK].dropna().empty:
    st.error("SPY data missing or empty after alignment.")
    st.stop()

bench_rets = compute_raw_returns_from_levels(levels[[BENCHMARK]])[BENCHMARK].dropna()

all_basket_rets, basket_members = compute_basket_returns(
    levels=levels,
    baskets=ALL_BASKETS,
    market_caps=market_caps,
    min_market_cap=MIN_MARKET_CAP,
    strict_market_cap=strict_cap_filter_ui,
    stale_days=STALE_DAYS,
    master_index=master_index
)

if all_basket_rets.empty:
    st.error("No baskets passed the data quality filters for the selected range.")
    st.stop()


# =========================================================
# Status strip
# =========================================================
src_text = f"Prices: {levels_source.upper()} | Market caps: {cap_source.upper()} | Trading sessions: {len(master_index):,}"
cap_text = (
    f"Strict market cap filter ON at ${MIN_MARKET_CAP:,.0f}"
    if strict_cap_filter_ui
    else f"Loose market cap filter ON at ${MIN_MARKET_CAP:,.0f}"
)
st.caption(f"{src_text} | {cap_text}")


# =========================================================
# Consolidated panel
# =========================================================
st.subheader("All Baskets | Consolidated Panel")

all_panel_df = build_panel_df(
    basket_returns=all_basket_rets,
    ref_start=pd.Timestamp(start_date),
    dynamic_label=DYNAMIC_LABEL,
    benchmark_series=bench_rets,
    basket_members=basket_members
)

dynamic_col = f"%{DYNAMIC_LABEL}"

if all_panel_df.empty:
    st.info("No valid baskets available after applying the current filters.")
else:
    c1, c2, c3 = st.columns(3)

    with c1:
        leader_name = all_panel_df.index[0]
        leader_val = all_panel_df.iloc[0][dynamic_col]
        st.metric(f"Top {DYNAMIC_LABEL}", leader_name, f"{leader_val:.1f}%")

    with c2:
        laggard_name = all_panel_df.index[-1]
        laggard_val = all_panel_df.iloc[-1][dynamic_col]
        st.metric(f"Bottom {DYNAMIC_LABEL}", laggard_name, f"{laggard_val:.1f}%")

    with c3:
        up_breadth = int((all_panel_df["EMA 4/9/18"] == "Up").sum())
        total_baskets = int(len(all_panel_df))
        st.metric("EMA Up Breadth", f"{up_breadth}/{total_baskets}", f"{(100 * up_breadth / total_baskets):.1f}%")

    top3 = ", ".join(
        [f"{idx} ({row[dynamic_col]:.1f}%)" for idx, row in all_panel_df.head(3).iterrows()]
    )
    bottom3 = ", ".join(
        [f"{idx} ({row[dynamic_col]:.1f}%)" for idx, row in all_panel_df.tail(3).sort_values(dynamic_col).iterrows()]
    )
    high_rvol = all_panel_df.sort_values("RVOL / SPY", ascending=False).head(3)
    high_rvol_text = ", ".join([f"{idx} ({row['RVOL / SPY']:.2f}x)" for idx, row in high_rvol.iterrows()])

    st.write(
        f"Leadership is sitting in {top3}, while the weakest tape is concentrated in {bottom3}. "
        f"On a volatility-adjusted basis the most active baskets relative to SPY are {high_rvol_text}. "
        f"This helps separate clean trend leadership from noisy beta."
    )

    st.dataframe(
        style_panel_df(all_panel_df),
        use_container_width=True,
        height=min(1200, 42 + 31 * len(all_panel_df))
    )

st.subheader("All Baskets | Cumulative Performance vs SPY")
ordered_cols = all_panel_df.index.tolist() if not all_panel_df.empty else all_basket_rets.columns.tolist()
plot_cumulative_chart(
    basket_returns=all_basket_rets[ordered_cols],
    title="All Baskets vs SPY",
    benchmark_series=bench_rets,
    default_show_count=default_chart_count
)


# =========================================================
# Per-category sections
# =========================================================
for category, baskets in CATEGORIES.items():
    st.markdown(f"## {category}")

    category_names = [b for b in baskets.keys() if b in all_basket_rets.columns]
    if not category_names:
        st.info("No valid baskets in this category for the current filter set.")
        continue

    category_rets = all_basket_rets[category_names].dropna(how="all")
    if category_rets.empty:
        st.info("No usable data in this category for the current range.")
        continue

    category_member_map = {k: v for k, v in basket_members.items() if k in category_names}
    category_panel = build_panel_df(
        basket_returns=category_rets,
        ref_start=pd.Timestamp(start_date),
        dynamic_label=DYNAMIC_LABEL,
        benchmark_series=bench_rets,
        basket_members=category_member_map
    )

    if category_panel.empty:
        st.info("No baskets passed the final metric checks in this category.")
        continue

    st.write(build_category_summary(category_panel, dynamic_col))

    st.dataframe(
        style_panel_df(category_panel),
        use_container_width=True,
        height=min(900, 42 + 31 * len(category_panel))
    )

    plot_cumulative_chart(
        basket_returns=category_rets[category_panel.index.tolist()],
        title=f"{category} vs SPY",
        benchmark_series=bench_rets,
        default_show_count=min(default_chart_count, max(6, len(category_panel)))
    )


# =========================================================
# Diagnostics
# =========================================================
if show_overlap:
    st.markdown("## Overlap Diagnostic")
    overlap_df = build_overlap_table(basket_members)
    if overlap_df.empty:
        st.info("No overlap data available.")
    else:
        st.write(
            "This is the quickest way to see whether apparent breadth is truly broad or just the same leaders showing up under different labels."
        )
        st.dataframe(overlap_df.head(50), use_container_width=True, height=500)

        repeated = overlap_df[overlap_df["Basket Count"] >= 5].head(15)
        if not repeated.empty:
            repeated_text = ", ".join(
                [f"{row['Ticker']} ({int(row['Basket Count'])})" for _, row in repeated.iterrows()]
            )
            st.write(f"Most repeated constituents across baskets: {repeated_text}.")


# =========================================================
# Constituents
# =========================================================
if show_constituents:
    with st.expander("Basket Constituents", expanded=False):
        for category, groups in CATEGORIES.items():
            st.markdown(f"**{category}**")
            for name, tickers in groups.items():
                eligible = basket_members.get(name, [])
                raw = dedupe_keep_order(tickers)
                st.write(
                    f"- {name}: {', '.join(raw)}"
                    + (f" | Eligible now: {', '.join(eligible)}" if eligible else " | Eligible now: none")
                )

st.caption("© 2026 AD Fund Management LP")
