import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import time


# ============================================================
# Page and theme
# ============================================================
st.set_page_config(page_title="ADFM Public Equities Baskets", layout="wide")

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

TITLE = "ADFM Public Equities Baskets"
SUBTITLE = "Sector, thematic, country, and macro dislocation baskets."


PASTEL = [
    "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
    "#b279a2", "#ff9da6", "#9d755d", "#bab0ac", "#59a14f",
    "#edc948", "#af7aa1", "#ff9da7", "#76b7b2", "#8cd17d",
    "#b6992d", "#499894", "#d37295", "#fabfd2", "#79706e",
]

MIN_MARKET_CAP = 1_000_000_000
INDICATOR_WARMUP_DAYS = 520
BENCH = "SPY"

MAGNIFICENT_SEVEN = "Magnificent Seven"
MAGNIFICENT_SEVEN_TICKERS = ["MAGS"]

CACHE_DIR = Path(".adfm_cache")
LEVELS_CACHE = CACHE_DIR / "sector_thematic_basket_levels_last_good.pkl"
LEVELS_META_CACHE = CACHE_DIR / "sector_thematic_basket_levels_last_good_meta.json"


# If Yahoo does not return a market cap, the ticker is kept.
# ETFs, ADRs, and foreign listings often have missing market-cap fields.
EXCLUDE_MISSING_MARKET_CAP = False


# ============================================================
# Category -> Baskets -> Tickers
# Consolidated basket map. Near-duplicate baskets were removed so each remaining basket has a cleaner signal.
# ============================================================
CATEGORIES: Dict[str, Dict[str, List[str]]] = {'Tech & AI': {'AI Accelerators': ['NVDA', 'AMD', 'AVGO', 'MRVL', 'INTC'],
               'Custom Silicon': ['TSM', 'QCOM', 'ARM'],
               'Semis ETFs': ['SMH', 'SOXX', 'XSD'],
               'Analog Chips': ['TXN', 'ADI', 'MCHP', 'NXPI', 'MPWR', 'ON', 'STM', 'IFNNY'],
               'Memory': ['MU', 'SNDK', '000660.KS', '005930.KS'],
               'Foundry/OSAT': ['UMC', 'GFS', 'ASX', 'AMKR', 'BESIY'],
               'Semicap Equipment': ['ASML', 'AMAT', 'LRCX', 'KLAC', 'TER', 'ONTO', 'ACMR', 'COHU'],
               'Semicap Components': ['MKSI', 'ENTG', 'AEIS', 'UCTT', 'ICHR', 'VECO', 'CAMT'],
               'EDA/IP': ['SNPS', 'CDNS'],
               'RF/Wireless': ['SWKS', 'QRVO', 'MTSI', 'AVNW'],
               'Optical': ['CIEN', 'LITE', 'COHR', 'AAOI', 'INFN', 'NOK'],
               'DC Networking': ['ANET', 'CSCO', 'JNPR'],
               'Servers/Storage': ['SMCI', 'DELL', 'HPE', 'NTAP', 'WDC', 'STX', 'IBM'],
               'Hyperscalers': ['MSFT', 'AMZN', 'GOOGL', 'META', 'ORCL'],
               'Neocloud': ['NBIS', 'IREN', 'CORZ', 'APLD'],
               'SaaS': ['NOW', 'CRM', 'ADBE', 'INTU', 'HUBS', 'WDAY', 'DOCU'],
               'Vertical Software': ['VEEV', 'TYL', 'APPF', 'MNDY', 'PAYC', 'GWRE', 'DAY'],
               'Data Software': ['PLTR', 'SNOW', 'MDB', 'DDOG', 'ESTC', 'CFLT', 'S', 'AI'],
               'DevOps/Edge': ['DT', 'GTLB', 'NET', 'TEAM'],
               'Cybersecurity': ['PANW', 'CRWD', 'FTNT', 'ZS', 'OKTA', 'CYBR', 'CHKP', 'TENB'],
               'AdTech': ['TTD', 'PINS', 'SNAP', 'APP'],
               'Internet': ['BIDU', 'BILI', 'RBLX', 'MTCH'],
               'E-Commerce': ['SHOP', 'MELI', 'ETSY', 'PDD', 'BABA', 'JD', 'SE', 'EBAY'],
               'Streaming': ['NFLX', 'DIS', 'WBD', 'PARA', 'ROKU', 'SPOT'],
               'Fintech Apps': ['HOOD', 'SOFI', 'AFRM', 'UPST', 'NU'],
               'IT Services': ['ACN', 'CTSH', 'EPAM', 'GIB', 'INFY', 'WIT'],
               'Robotics': ['ISRG', 'SYM', 'ROK', 'FANUY', 'CGNX'],
               'Magnificent Seven': ['MAGS']},
 'Power & Materials': {'Oil Majors': ['XOM', 'CVX', 'COP', 'SHEL', 'BP', 'TTE', 'EQNR', 'PBR'],
                       'Oil E&Ps': ['EOG', 'FANG', 'OXY', 'DVN', 'PR', 'MTDR', 'APA', 'MRO'],
                       'Gas E&Ps': ['EQT', 'AR', 'RRC', 'CTRA', 'EXE', 'CRK', 'CNX'],
                       'LNG Infra': ['LNG', 'WDS', 'NEXT'],
                       'Midstream': ['KMI', 'WMB', 'ET', 'EPD', 'ENB', 'MPLX', 'OKE', 'TRP'],
                       'Oil Services': ['SLB', 'HAL', 'BKR', 'NOV', 'FTI', 'PTEN', 'HP', 'NBR', 'OII'],
                       'Refiners': ['MPC', 'VLO', 'PSX', 'DK', 'PBF', 'SUN'],
                       'Uranium Miners': ['CCJ', 'UUUU', 'UEC', 'URG', 'UROY', 'DNN', 'NXE', 'URA', 'URNM'],
                       'Nuclear Services': ['BWXT', 'LEU'],
                       'Merchant Power': ['VST', 'CEG', 'NRG', 'TLN', 'AES', 'CWEN'],
                       'Utilities': ['NEE', 'DUK', 'SO', 'AEP', 'XEL', 'EXC', 'PCG', 'EIX', 'ED'],
                       'Grid Equipment': ['ETN', 'GEV', 'ABB', 'HUBB', 'POWL', 'MYRG', 'RRX'],
                       'Thermal/Cooling': ['VRT', 'TT', 'CARR', 'JCI', 'MOD', 'AOS'],
                       'Backup Power': ['GNRC'],
                       'Solar': ['TAN', 'FSLR', 'ENPH', 'SEDG', 'RUN', 'CSIQ', 'JKS', 'ARRY', 'NXT'],
                       'Wind/Renewables': ['FAN', 'ICLN', 'BEP', 'AY', 'VWSYF'],
                       'Hydrogen': ['BE', 'BLDP', 'FCEL', 'PLUG', 'LIN', 'APD'],
                       'Copper': ['FCX', 'SCCO', 'ERO', 'IVPAF', 'COPX'],
                       'Diversified Mining': ['BHP', 'RIO', 'VALE', 'GLNCY', 'TECK', 'XME'],
                       'Lithium/Battery Metals': ['ALB', 'SQM', 'LAC', 'PLL', 'LIT', 'MP', 'REMX', 'LYSDY'],
                       'Gold/Silver Miners': ['GDX', 'GDXJ', 'NEM', 'AEM', 'GOLD', 'KGC', 'AG', 'PAAS'],
                       'Steel': ['NUE', 'STLD', 'X', 'CLF'],
                       'Fertilizers': ['CF', 'MOS', 'NTR', 'IPI'],
                       'Specialty Chemicals': ['SHW', 'PPG', 'EMN', 'RPM', 'IFF'],
                       'Aggregates': ['VMC', 'MLM', 'SUM', 'EXP', 'CRH', 'CX'],
                       'Water/AI Permits': ['XYL', 'WTS', 'AWK', 'ECL', 'DHR', 'PNR', 'CWCO'],
                       'Power-Cost Losers': ['DOW', 'LYB', 'CE', 'WLK', 'OLN', 'AA', 'CENX', 'IP', 'PKG']},
 'Industrials & Defense': {'Defense Primes': ['LMT', 'NOC', 'RTX', 'GD', 'HII', 'BAESY', 'RNMBY'],
                           'Defense Supply Chain': ['LHX', 'TDG', 'HEI', 'HWM', 'ATI', 'CW'],
                           'Europe Defense': ['RHM.DE',
                                              'SAAB-B.ST',
                                              'BA.L',
                                              'HO.PA',
                                              'SAF.PA',
                                              'LDO.MI',
                                              'HAG.DE',
                                              'KOG.OL',
                                              'AM.PA',
                                              'ESLT'],
                           'Defense Tech': ['LDOS', 'BAH', 'CACI', 'KTOS', 'AVAV', 'PSN', 'G'],
                           'Space': ['RKLB', 'IRDM', 'ASTS', 'VSAT', 'GSAT'],
                           'Aero OEMs': ['BA', 'GE', 'SPR', 'TXT'],
                           'Aero Leasing': ['FTAI', 'AER', 'AL', 'ATSG', 'CPA'],
                           'Automation': ['EMR', 'AME', 'PH', 'KEYS', 'TRMB'],
                           'Sensors/Test': ['TDY', 'FTV', 'A'],
                           'Industrial Software': ['ADSK', 'PTC', 'DASTY'],
                           'Machinery': ['CAT', 'DE', 'CNHI', 'AGCO', 'PCAR', 'CMI'],
                           'Rental/Distribution': ['URI', 'HRI', 'GWW', 'FAST', 'AIT', 'WCC', 'SITE'],
                           'EPC/Construction': ['J', 'ACM', 'FLR', 'PWR', 'MTZ', 'FIX', 'EME', 'DY'],
                           'Waste': ['WM', 'RSG', 'WCN', 'CLH', 'SRCL'],
                           'Rails': ['UNP', 'CSX', 'NSC', 'CNI', 'CP'],
                           'Logistics': ['FDX', 'UPS', 'GXO', 'XPO', 'CHRW'],
                           'Trucking': ['ODFL', 'SAIA', 'JBHT', 'KNX', 'ARCB']},
 'Healthcare': {'Pharma': ['LLY', 'JNJ', 'MRK', 'PFE', 'BMY', 'ABBV', 'AZN', 'NVO', 'NVS', 'GSK'],
                'Biotech Large': ['AMGN', 'GILD', 'REGN', 'BIIB', 'VRTX', 'ALNY'],
                'GLP-1 Drugs': ['VKTX'],
                'Oncology/Immunology': ['RHHBY', 'INCY', 'EXEL'],
                'Rare Disease': ['BMRN', 'RARE', 'IONS', 'HALO'],
                'MedTech': ['MDT', 'SYK', 'BSX', 'ZBH', 'EW', 'PEN', 'ABT'],
                'Surgical Robotics': ['TMDX'],
                'Diabetes Devices': ['DXCM', 'PODD', 'TNDM'],
                'Diagnostics/Tools': ['TMO', 'RGEN', 'ILMN', 'WAT', 'BRKR'],
                'CROs/Clinical': ['IQV', 'LH', 'DGX', 'MEDP', 'ICLR', 'CRL'],
                'Managed Care': ['UNH', 'HUM', 'CI', 'ELV', 'CNC', 'MOH', 'OSCR'],
                'Drug Distribution': ['MCK', 'COR', 'CAH', 'CVS', 'OMI'],
                'Hospitals': ['HCA', 'THC', 'UHS', 'CYH', 'EHC'],
                'Dental/Elective': ['ALGN', 'EYE', 'WRBY', 'XRAY'],
                'Staffing': ['AMN', 'RHI', 'ASGN', 'KFY'],
                'Animal Health': ['ZTS', 'ELAN', 'IDXX', 'PETS', 'HSIC']},
 'Financials & RE': {'Money Banks': ['JPM', 'BAC', 'C', 'WFC'],
                     'IBanks': ['GS', 'MS', 'RJF', 'LAZ', 'PJT', 'EVR'],
                     'Regional Banks': ['KRE', 'TFC', 'FITB', 'CFG', 'RF', 'KEY', 'PNC', 'USB', 'MTB', 'WAL', 'ZION'],
                     'Alts': ['BX', 'KKR', 'APO', 'CG', 'ARES', 'OWL', 'TPG', 'BLK', 'TROW'],
                     'BDCs': ['ARCC', 'MAIN', 'BXSL', 'OBDC', 'FSK', 'PSEC'],
                     'Exchanges/Data': ['CME', 'ICE', 'NDAQ', 'CBOE', 'MKTX', 'SPGI', 'MSCI', 'MCO'],
                     'Brokers': ['IBKR', 'SCHW'],
                     'Market Makers': ['VIRT', 'TW'],
                     'Card Networks': ['V', 'MA', 'AXP'],
                     'Consumer Lenders': ['COF', 'DFS', 'SYF', 'OMF', 'ENVA'],
                     'Legacy Payments': ['FI', 'FIS', 'GPN', 'PYPL', 'XYZ'],
                     'Tokenization': ['COIN', 'MSTR', 'IBIT', 'ETHA'],
                     'Bitcoin Miners': ['MARA', 'RIOT', 'CLSK', 'CIFR', 'BITF'],
                     'P&C Insurance': ['PGR', 'TRV', 'CB', 'ALL', 'CINF', 'WRB', 'HIG'],
                     'Life Insurance': ['MET', 'PRU', 'LNC', 'AIG', 'EQH', 'RGA'],
                     'Insurance Brokers': ['AJG', 'BRO', 'MMC', 'AON', 'WTW'],
                     'Mortgage Finance': ['RKT', 'UWMC', 'COOP', 'FNF', 'FAF', 'NMIH', 'ESNT'],
                     'Data Center REITs': ['EQIX', 'DLR'],
                     'Industrial REITs': ['PLD', 'REXR', 'EGP', 'STAG', 'TRNO'],
                     'Residential REITs': ['AVB', 'EQR', 'UDR', 'ESS', 'MAA', 'INVH', 'AMH'],
                     'Storage REITs': ['PSA', 'EXR', 'CUBE', 'NSA'],
                     'Tower REITs': ['AMT', 'SBAC', 'CCI'],
                     'Office REITs': ['BXP', 'VNO', 'SLG', 'KRC', 'DEI'],
                     'Senior Housing REITs': ['WELL', 'VTR', 'OHI', 'SBRA'],
                     'CRE Credit Stress': ['BXMT', 'STWD', 'KREF', 'LADR', 'ARI', 'ABR', 'RITM']},
 'Consumer': {'Grocery/Clubs': ['WMT', 'COST', 'KR', 'BJ', 'ACI'],
              'Household Care': ['PG', 'CL', 'KMB', 'CHD', 'EL', 'KVUE'],
              'Beverages/Tobacco': ['KO', 'PEP', 'KDP', 'MNST', 'PM', 'MO', 'BTI'],
              'Packaged Food': ['GIS', 'K', 'CPB', 'CAG', 'HSY', 'MDLZ', 'SJM', 'KHC'],
              'Discount Retail': ['DG', 'DLTR', 'FIVE', 'OLLI'],
              'Off-Price': ['TJX', 'ROST', 'BURL'],
              'Home Improvement': ['HD', 'LOW', 'TSCO', 'POOL', 'BLDR'],
              'Apparel': ['NKE', 'LULU', 'DECK', 'ONON', 'UAA', 'RL', 'TPR'],
              'Luxury': ['LVMUY', 'CPRI', 'CFRUY', 'PPRUY', 'RACE'],
              'Beauty': ['ULTA', 'COTY', 'LRLCY', 'ELF'],
              'Restaurants': ['MCD', 'YUM', 'QSR', 'WEN', 'DPZ', 'CMG'],
              'Casual Dining': ['DRI', 'TXRH', 'EAT', 'BLMN', 'CAKE', 'WING'],
              'Travel Booking': ['BKNG', 'EXPE', 'ABNB', 'TRIP'],
              'Hotels/Casinos': ['MAR', 'HLT', 'IHG', 'MGM', 'LVS', 'WYNN', 'MLCO', 'CZR', 'PENN'],
              'Cruise Lines': ['RCL', 'CCL', 'NCLH'],
              'Airlines': ['DAL', 'UAL', 'AAL', 'LUV', 'ALK', 'JBLU'],
              'Sports Betting': ['DKNG', 'FLUT', 'RSI'],
              'Live Events': ['LYV', 'TKO', 'MSGS', 'SPHR', 'BATRA', 'FWONA'],
              'Gaming': ['EA', 'TTWO', 'NTDOY', 'SONY'],
              'Autos': ['TM', 'HMC', 'F', 'GM', 'STLA', 'VWAGY'],
              'EVs': ['TSLA', 'RIVN', 'LCID'],
              'Auto Parts': ['AZO', 'ORLY', 'AAP', 'LKQ', 'GPC'],
              'Used Autos': ['KMX', 'CVNA', 'LAD', 'AN', 'PAG'],
              'Homebuilders': ['ITB', 'DHI', 'LEN', 'NVR', 'PHM', 'TOL', 'KBH', 'MTH'],
              'Building Products': ['TREX', 'MAS', 'OC', 'JELD', 'FBIN']},
 'Countries': {'Developed ex-US': ['VEA'],
               'Europe': ['VGK'],
               'Eurozone': ['EZU'],
               'Germany': ['EWG'],
               'France': ['EWQ'],
               'Italy': ['EWI'],
               'Spain': ['EWP'],
               'UK': ['EWU'],
               'Switzerland': ['EWL'],
               'Nordics': ['EWD', 'EDEN', 'NORW'],
               'Japan': ['EWJ', 'DXJ'],
               'Canada': ['EWC'],
               'Australia': ['EWA'],
               'China': ['MCHI', 'FXI', 'KWEB'],
               'Taiwan': ['EWT'],
               'South Korea': ['EWY'],
               'India': ['INDA', 'EPI', 'INDY'],
               'Indonesia': ['EIDO', 'IDX'],
               'Vietnam': ['VNM'],
               'EM Broad': ['IEMG', 'EEM'],
               'Frontier': ['FM'],
               'LatAm': ['ILF'],
               'Brazil': ['EWZ'],
               'Mexico': ['EWW'],
               'Argentina': ['ARGT'],
               'Chile': ['ECH'],
               'Peru': ['EPU'],
               'Poland': ['EPOL'],
               'Turkey': ['TUR'],
               'Middle East': ['KSA', 'UAE', 'QAT', 'EIS'],
               'South Africa': ['EZA'],
               'Japan Banks': ['MUFG', 'SMFG', 'MFG', 'IX', 'NMR', 'MKTAY', 'ORIX', 'JPXN'],
               'Europe Banks': ['EUFN', 'DB', 'ING', 'SAN', 'BBVA', 'UBS', 'BNPQY', 'CRARY', 'UNCFF']},
 'Macro & Special': {'Tankers': ['FRO', 'STNG', 'INSW', 'TNK', 'DHT', 'NAT', 'TRMD', 'HAFN', 'CMBT', 'ASC'],
                     'Offshore Drilling': ['RIG', 'VAL', 'NE', 'BORR', 'DO', 'SDRL'],
                     'Critical Minerals': ['ALM', 'PPTA', 'USAR', 'UAMY'],
                     'eVTOL': ['JOBY', 'ACHR', 'EH', 'EVTL', 'BLDE'],
                     'Cannabis/MSOs': ['MSOS', 'CURLF', 'GTBIF', 'TCNNF', 'CRLBF'],
                     'Container Shipping': ['ZIM', 'MATX', 'DAC', 'GSL', 'CMRE', 'SFL'],
                     'Dry Bulk': ['SBLK', 'GNK', 'GOGL', 'EGLE', 'SB', 'DSX', 'PANL'],
                     'LNG/LPG Shipping': ['FLNG', 'GLNG', 'LPG', 'BWLP', 'NVGS'],
                     'Coal': ['BTU', 'AMR', 'HCC', 'METC'],
                     'Clean Firm Power': ['ORA', 'OKLO', 'SMR'],
                     'Gold Royalties': ['FNV', 'WPM', 'RGLD', 'OR', 'SAND', 'TFPM'],
                     'Silver Pure Play': ['SLV', 'SIL', 'HL', 'CDE', 'FSM', 'MAG'],
                     'Volatility Products': ['VIXY', 'VXX', 'UVXY', 'SVIX', 'VXZ'],
                     'Credit Data': ['FICO', 'EFX', 'TRU', 'EXPGY'],
                     'Argentina Reform': ['YPF', 'GGAL', 'BMA', 'BBAR', 'PAM', 'TGS', 'CEPU', 'LOMA'],
                     'Lower Oil Winners': ['JETS', 'CRUZ', 'IYT', 'XTN', 'AWAY'],
                     'Oil Shock': ['XLE', 'XOP', 'OIH', 'XES'],
                     'Liquidity Beta': ['TQQQ', 'SOXL', 'NVDL', 'TSLL', 'MSTU', 'CONL', 'BITX', 'DPST', 'LABU'],
                     'Biotech Funding': ['XBI',
                                         'IBB',
                                         'ARKG',
                                         'RXRX',
                                         'SDGR',
                                         'BEAM',
                                         'CRSP',
                                         'NTLA',
                                         'EDIT',
                                         'PACB',
                                         'TWST'],
                     'Medical AI': ['GEHC', 'HOLX', 'BFLY', 'EXAS', 'NTRA', 'GH', 'TEM'],
                     'GLP-1 Losers': ['INSP', 'RMD'],
                     'Food Inflation': ['ADM', 'BG', 'TSN', 'PPC', 'CALM', 'HRL'],
                     'Border Security': ['AXON', 'MSI', 'GEO', 'CXW', 'OSK'],
                     'Fiber/Broadband': ['GLW', 'FYBR', 'CABO', 'ATUS', 'COMM', 'CHTR', 'CMCSA'],
                     'China EV Export': ['KARS', 'DRIV', 'BYDDY', 'LI', 'NIO', 'XPEV'],
                     'Private Credit Stress': ['BIZD', 'PBDC']}}


# Keep this basket explicit so it survives future edits to the raw basket map.
CATEGORIES.setdefault("Tech & AI", {})[MAGNIFICENT_SEVEN] = MAGNIFICENT_SEVEN_TICKERS



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

        if min_market_cap is not None and market_caps is not None:
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
    return {basket: tickers for groups in categories.values() for basket, tickers in groups.items()}


def unique_tickers_from_baskets(baskets: Dict[str, List[str]], extra: Optional[Iterable[str]] = None) -> List[str]:
    tickers = {str(t).upper().strip() for members in baskets.values() for t in members if str(t).strip()}

    if extra:
        tickers.update(str(t).upper().strip() for t in extra if str(t).strip())

    return sorted(tickers)


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


def basket_vs_dma_pct(
    series: pd.Series,
    window: int,
) -> float:
    """
    Current basket level versus the basket's own moving average.

    The basket series is already equal-weighted by ew_rets_from_levels().
    This returns the percent distance between today's basket level and
    today's rolling DMA:

        current basket level / current basket DMA - 1

    A positive value means the basket is trading above that DMA.
    A negative value means the basket is trading below that DMA.
    """
    clean = series.dropna()

    if clean.shape[0] < window:
        return np.nan

    dma = clean.rolling(window=window, min_periods=window).mean()
    latest_px = clean.iloc[-1]
    latest_dma = dma.dropna().iloc[-1] if dma.dropna().shape[0] else np.nan

    if pd.isna(latest_px) or pd.isna(latest_dma) or latest_dma == 0:
        return np.nan

    return float((latest_px / latest_dma - 1.0) * 100.0)


def build_panel_df(
    basket_returns_full: pd.DataFrame,
    display_start: pd.Timestamp,
    dynamic_label: str,
    levels: Optional[pd.DataFrame] = None,
    basket_members: Optional[Dict[str, List[str]]] = None,
    benchmark_series_full: Optional[pd.Series] = None,
) -> pd.DataFrame:
    cols = [
        "Basket", "%5D", "%1M", f"%{dynamic_label}",
        "MACD Momentum", "EMA 4/9/18", "RSI(14W)", "Corr(63D)",
        "vs 21DMA %", "vs 50DMA %"
    ]

    if basket_returns_full.empty:
        return pd.DataFrame(columns=cols).set_index("Basket")

    levels_full = 100.0 * (1.0 + basket_returns_full.fillna(0.0)).cumprod()
    rows: List[Dict[str, Any]] = []

    basket_members = basket_members or {}

    for basket in levels_full.columns:
        s_full = levels_full[basket].dropna()

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

        dma_21_pct = basket_vs_dma_pct(s_full, window=21)
        dma_50_pct = basket_vs_dma_pct(s_full, window=50)

        weekly = s_full.resample("W-FRI").last().dropna()
        rsi_14w = np.nan
        if weekly.shape[0] >= 14:
            rsi_w = rsi(weekly, 14)
            if rsi_w.dropna().shape[0]:
                rsi_14w = rsi_w.dropna().iloc[-1]

        hist = macd_hist(s_full, 12, 26, 9)
        macd_m = momentum_label(hist, lookback=5, z_window=63)
        ema_tag = ema_regime(s_full, 4, 9, 18)

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
            "MACD Momentum": macd_m,
            "EMA 4/9/18": ema_tag,
            "RSI(14W)": round(rsi_14w, 2) if pd.notna(rsi_14w) else np.nan,
            "Corr(63D)": round(corr_spy, 2) if pd.notna(corr_spy) else np.nan,
            "vs 21DMA %": round(dma_21_pct, 1) if pd.notna(dma_21_pct) else np.nan,
            "vs 50DMA %": round(dma_50_pct, 1) if pd.notna(dma_50_pct) else np.nan
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
        r1, g1, b1 = 240, 255, 245
        r2, g2, b2 = 82, 183, 136
        r = int(r1 + (r2 - r1) * s)
        g = int(g1 + (g2 - g1) * s)
        b = int(b1 + (b2 - b1) * s)
        return f"rgb({r},{g},{b})"

    s = min(abs(x) / 20.0, 1.0)
    r1, g1, b1 = 255, 245, 245
    r2, g2, b2 = 232, 93, 93
    r = int(r1 + (r2 - r1) * s)
    g = int(g1 + (g2 - g1) * s)
    b = int(b1 + (b2 - b1) * s)
    return f"rgb({r},{g},{b})"


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

    dynamic_cols = [c for c in panel_df.columns if c.startswith("%") and c not in ["%5D", "%1M"]]
    dynamic_col = dynamic_cols[0] if dynamic_cols else None

    if dynamic_col is None:
        st.info("No dynamic return column available.")
        return

    headers = [
        "Basket", "%5D", "%1M", dynamic_col,
        "MACD Momentum", "EMA 4/9/18", "RSI(14W)", "Corr(63D)",
        "vs 21DMA %", "vs 50DMA %"
    ]

    values = [panel_df.index.tolist()]
    fill_colors = [["white"] * len(panel_df)]

    for col in ["%5D", "%1M", dynamic_col]:
        vals = panel_df[col].tolist()
        values.append(vals)
        fill_colors.append([color_ret(v) for v in vals])

    vals = panel_df["MACD Momentum"].tolist()
    values.append(vals)
    fill_colors.append([color_macd(v) for v in vals])

    vals = panel_df["EMA 4/9/18"].tolist()
    values.append(vals)
    fill_colors.append([color_ema(v) for v in vals])

    vals = panel_df["RSI(14W)"].tolist()
    values.append(vals)
    fill_colors.append([color_rsi(v) for v in vals])

    vals = panel_df["Corr(63D)"].tolist()
    values.append(vals)
    fill_colors.append([color_corr(v) for v in vals])

    for col in ["vs 21DMA %", "vs 50DMA %"]:
        vals = panel_df[col].tolist()
        values.append(vals)
        fill_colors.append([color_ret(v) for v in vals])

    col_widths = [0.25, 0.06, 0.06, 0.085, 0.155, 0.105, 0.075, 0.055, 0.085, 0.085]

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
            format=[None, ".1f", ".1f", ".1f", None, None, ".2f", ".2f", ".1f", ".1f"]
        )
    )])

    fig_tbl.update_layout(
        margin=dict(l=0, r=0, t=6, b=0),
        height=min(920, 64 + 26 * max(3, len(panel_df)))
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

    chart_rets = basket_returns_display.loc[common_index].copy()
    chart_rets = chart_rets.dropna(axis=1, how="all")

    if chart_rets.empty:
        st.info("No basket return series available for this chart.")
        return

    cum_pct = ((1 + chart_rets.fillna(0.0)).cumprod() - 1.0) * 100.0
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




def render_basket_section(
    heading: str,
    basket_returns_full: pd.DataFrame,
    basket_returns_display: pd.DataFrame,
    benchmark_returns_full: pd.Series,
    benchmark_returns_display: pd.Series,
    display_start: pd.Timestamp,
    dynamic_label: str,
    show_chart: bool,
    levels: Optional[pd.DataFrame] = None,
    basket_members: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    st.subheader(heading)

    panel_df = build_panel_df(
        basket_returns_full=basket_returns_full,
        display_start=display_start,
        dynamic_label=dynamic_label,
        levels=levels,
        basket_members=basket_members,
        benchmark_series_full=benchmark_returns_full,
    )

    plot_panel_table(panel_df)

    if show_chart:
        ordered_cols = [c for c in panel_df.index if c in basket_returns_display.columns]
        chart_rets = basket_returns_display[ordered_cols] if ordered_cols else basket_returns_display

        plot_cumulative_chart(
            basket_returns_display=chart_rets,
            title=f"{heading} | Cumulative Performance vs SPY",
            benchmark_series_display=benchmark_returns_display,
        )

    return panel_df


# ============================================================
# Sidebar
# ============================================================
st.title(TITLE)
st.caption(f"{SUBTITLE} Current map: {sum(len(v) for v in CATEGORIES.values())} baskets across {len(CATEGORIES)} groups.")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Internal sector, thematic, and country basket monitor.

        **What this page includes**
        - Sector and subsector baskets.
        - Cross-sector thematic baskets.
        - Country and regional baskets.
        - Macro dislocation and special-situation baskets.

        **What this page excludes**
        - Pure factor baskets.
        - Redundant duplicate baskets.

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

    apply_market_cap_filter = st.checkbox("Apply $1B market-cap filter", value=False)
    stale_days = st.slider("Stale Price Threshold, Days", min_value=10, max_value=90, value=30, step=5)

    selected_categories = st.multiselect(
        "Categories",
        options=list(CATEGORIES.keys()),
        default=list(CATEGORIES.keys())
    )

    st.markdown("### Optional Sections")
    show_all_chart = st.checkbox("Show consolidated cumulative chart", value=False)
    show_category_sections = st.checkbox("Show per-category panels and charts", value=False)
    show_constituents = st.checkbox("Show live basket constituents", value=False)
    show_full_map = st.checkbox("Show raw basket map", value=True)
    show_data_notes = st.checkbox("Show data notes", value=False)

display_start_date = compute_display_start(preset, today)
fetch_start_date = compute_fetch_start(display_start_date)
end_date = today

DYNAMIC_LABEL = preset
min_market_cap = MIN_MARKET_CAP if apply_market_cap_filter else None


# ============================================================
# Fetch data
# ============================================================
need = unique_tickers_from_baskets(flatten_baskets(CATEGORIES), extra=[BENCH])

with st.spinner("Fetching basket price history..."):
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
# Consolidated panel and optional sections
# ============================================================
all_panel_df = render_basket_section(
    heading="All Baskets | Consolidated Panel",
    basket_returns_full=all_basket_rets_full,
    basket_returns_display=all_basket_rets_display,
    benchmark_returns_full=bench_rets_full,
    benchmark_returns_display=bench_rets_display,
    display_start=display_start_ts,
    dynamic_label=DYNAMIC_LABEL,
    show_chart=show_all_chart,
    levels=levels,
    basket_members=live_all_baskets,
)

if show_category_sections:
    for category, baskets in live_categories.items():
        cat_names = [basket for basket in baskets if basket in all_basket_rets_full.columns]

        if not cat_names:
            st.info(f"{category}: no data for this group in the selected range.")
            continue

        cat_rets_full = all_basket_rets_full[cat_names].dropna(how="all")
        if cat_rets_full.empty:
            st.info(f"{category}: no data for this group in the selected range.")
            continue

        render_basket_section(
            heading=category,
            basket_returns_full=cat_rets_full,
            basket_returns_display=slice_returns_for_display(cat_rets_full, display_start_ts),
            benchmark_returns_full=bench_rets_full,
            benchmark_returns_display=bench_rets_display,
            display_start=display_start_ts,
            dynamic_label=DYNAMIC_LABEL,
            show_chart=True,
            levels=levels,
            basket_members=baskets,
        )

if show_constituents:
    with st.expander("Basket Constituents", expanded=True):
        st.caption("Shows only live members used in the calculations after price, stale-data, and optional market-cap filters.")

        for category, groups in live_categories.items():
            st.markdown(f"**{category}**")
            for name, tickers in groups.items():
                members = sorted({str(t).upper() for t in tickers})
                st.write(f"- {name}: {', '.join(members)}")

if show_full_map:
    with st.expander("Full Basket Map", expanded=True):
        st.caption("Raw basket definitions before data-quality filtering.")
        for category, groups in CATEGORIES.items():
            st.markdown(f"**{category}**")
            for name, tickers in groups.items():
                st.write(f"- {name}: {', '.join(tickers)}")

if show_data_notes:
    with st.expander("Data Notes", expanded=True):
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
