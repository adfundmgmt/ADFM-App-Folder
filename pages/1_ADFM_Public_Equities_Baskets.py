import streamlit as st

from adfm_core.ui import render_footer

import pandas as pd

import numpy as np

import yfinance as yf

from datetime import date, datetime, time as dt_time, timedelta

import plotly.graph_objects as go

from pathlib import Path

from typing import Any, Dict, Iterable, List, Optional, Tuple

from zoneinfo import ZoneInfo

import hashlib

import json

import math

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

NY_TZ = ZoneInfo("America/New_York")

COMPLETED_SESSION_TIME = dt_time(16, 15)

CACHE_VERSION = 2

CACHE_MAX_AGE_DAYS = 7

MIN_LIVE_MEMBER_COVERAGE = 0.50

MIN_DAILY_MEMBER_COVERAGE = 0.60

MAX_FORWARD_FILL_SESSIONS = 5

BASKET_KEY_SEPARATOR = " :: "



MAGNIFICENT_SEVEN = "Magnificent Seven"

MAGNIFICENT_SEVEN_TICKERS = ["NVDA", "MSFT", "GOOGL", "AMZN", "AAPL", "META", "TSLA"]



CACHE_DIR = Path(".adfm_cache")

FX_CONVERSIONS: Dict[str, Tuple[str, str]] = {

    "000660.KS": ("KRW=X", "divide"),

    "005930.KS": ("KRW=X", "divide"),

    "EUROB.AT": ("EURUSD=X", "multiply"),

    "MYTIL.AT": ("EURUSD=X", "multiply"),

    "OPAP.AT": ("EURUSD=X", "multiply"),

    "TRUL.CN": ("CAD=X", "divide"),

    "TUN.L": ("GBPUSD=X", "multiply"),

    "WHC.AX": ("AUDUSD=X", "multiply"),

}

FUND_QUOTE_TYPES = {"ETF", "MUTUALFUND", "MONEYMARKET"}





# ============================================================

# Category -> Baskets -> Tickers

# Revised sector-internal map with restored thematic, country, and macro-dislocation groups.

# Structure preserved: Dict[str, Dict[str, List[str]]]

# 273 basket definitions across 14 groups. Internal keys include category names,
# so repeated display names cannot overwrite one another.

# ============================================================

CATEGORIES: Dict[str, Dict[str, List[str]]] = {



'Information Technology': {

		'Semiconductors Broad':['NVDA','AMD','AVGO','QCOM','TXN','ADI','MU','INTC','MRVL','NXPI','MCHP','ON','MPWR','ARM'],

		'AI Compute and Accelerators': ['NVDA', 'AMD', 'AVGO', 'MRVL', 'ARM', 'INTC'],

                'Custom Silicon and ASICs': ['AVGO', 'MRVL', 'AMD', 'TSM', 'ARM', 'SNPS', 'CDNS'],

                'Semiconductor Equipment': ['ASML','AMAT','LRCX','KLAC','TER','ONTO','AEIS','ACMR','COHU'],

                'Semicap Subsystems and Components': ['MKSI','ENTG','AEIS','UCTT','ICHR','COHU','VECO','CAMT'],

                'Foundries and OSAT': ['TSM', 'UMC', 'GFS', 'ASX', 'AMKR', 'BESIY'],

                'Advanced Packaging': ['AMKR', 'ASX', 'TSM', 'BESIY', 'AMAT', 'LRCX', 'KLAC', 'ONTO'],

                'Memory and Storage Semis': ['MU', 'WDC', 'STX', 'SNDK', '000660.KS', '005930.KS'],

                'Analog and Mixed Signal': ['TXN', 'ADI', 'MCHP', 'NXPI', 'MPWR', 'STM', 'IFNNY'],

                'Power Semiconductors': ['ON', 'WOLF', 'STM', 'IFNNY', 'NXPI', 'MCHP', 'MPWR', 'ADI'],

                'RF and Wireless Chips': ['QCOM', 'SWKS', 'QRVO', 'MTSI', 'AVGO', 'MRVL'],

                'EDA and Chip IP': ['SNPS', 'CDNS', 'ARM'],

                'Data Center Networking': ['ANET', 'CSCO', 'JNPR', 'AVGO', 'MRVL', 'CIEN', 'LITE'],

                'Optical Networking and Interconnect': ['CIEN', 'LITE', 'COHR', 'AAOI', 'INFN', 'NOK'],

                'Servers and AI Hardware': ['SMCI', 'DELL', 'HPE', 'IBM', 'NTAP', 'WDC', 'STX'],

                'Enterprise Storage': ['NTAP', 'WDC', 'STX', 'PSTG', 'DELL', 'HPE', 'IBM'],

                'IT Hardware and PCs': ['AAPL', 'DELL', 'HPQ', 'HPE', 'LOGI', 'NTAP', 'PSTG'],

                'Consumer Hardware': ['AAPL', 'SONY', 'LOGI', 'GRMN', 'GPRO'],

                'Application Software': ['MSFT','ADBE','CRM','NOW','INTU','TEAM','WDAY','DOCU','HUBS','APP'],

		'Systems and Infrastructure Software': ['MSFT','ORCL','IBM','PLTR','DDOG','NET','ESTC','MDB','SNOW','GTLB'],

                'Database and Data Platforms': ['SNOW', 'MDB', 'ESTC', 'CFLT', 'DDOG', 'ORCL', 'IBM'],

                'Observability and DevOps': ['DDOG', 'DT', 'GTLB', 'NET', 'ESTC', 'TEAM', 'MDB'],

                'Cybersecurity': ['PANW', 'CRWD', 'FTNT', 'ZS', 'OKTA', 'CYBR', 'CHKP', 'NET', 'TENB', 'S'],

                'Identity and Access Management': ['OKTA', 'CYBR', 'MSFT', 'PANW', 'FTNT', 'CHKP'],

                'Vertical Software': ['VEEV', 'TYL', 'APPF', 'MNDY', 'PAYC', 'GWRE', 'DAY'],

                'Financial Software': ['FICO', 'BR', 'FI', 'FIS', 'GPN', 'INTU', 'GWRE'],

                'Design and Engineering Software': ['ADSK', 'PTC', 'DASTY', 'SNPS', 'CDNS', 'TRMB'],

                'IT Consulting and Services': ['ACN', 'IBM', 'CTSH', 'EPAM', 'GIB', 'INFY', 'WIT'],

                'Government IT Services': ['SAIC', 'CACI', 'LDOS', 'BAH', 'PSN', 'G']},



 'Financials': {

		'Money Center Banks': ['JPM', 'BAC', 'C', 'WFC'],

                'Super-Regional Banks': ['PNC', 'USB', 'TFC', 'MTB', 'FITB', 'CFG', 'RF', 'KEY'],

                'Regional Banks': ['WAL', 'ZION', 'CMA', 'FHN', 'NYCB', 'EWBC', 'HBAN', 'SNV', 'CBSH'],

                'Investment Banks': ['GS', 'MS', 'RJF', 'LAZ', 'PJT', 'EVR', 'MC'],

                'Brokers and Trading Platforms': ['IBKR', 'SCHW', 'HOOD', 'RJF', 'MS', 'GS'],

                'Exchanges and Market Data': ['CME', 'ICE', 'NDAQ', 'CBOE', 'MKTX', 'SPGI', 'MSCI'],

                'Electronic Trading and Market Makers': ['VIRT', 'TW', 'IBKR', 'CBOE', 'NDAQ'],

                'Asset Managers': ['BLK', 'TROW', 'BEN', 'AMG', 'IVZ', 'JHG'],

                'Alternative Asset Managers': ['BX', 'KKR', 'APO', 'CG', 'ARES', 'OWL', 'TPG'],

                'Private Credit and BDCs': ['ARES', 'ARCC', 'MAIN', 'BXSL', 'OBDC', 'FSK'],

                'P&C Insurance': ['PGR', 'TRV', 'CB', 'ALL', 'CINF', 'WRB', 'HIG'],

                'Life and Retirement Insurance': ['MET', 'PRU', 'LNC', 'AIG', 'EQH', 'RGA'],

                'Reinsurance': ['ACGL', 'RNR', 'EG', 'HIG', 'CB'],

                'Insurance Brokers': ['AJG', 'BRO', 'MMC', 'AON', 'WTW'],

                'Mortgage Finance and Title': ['RKT', 'UWMC', 'COOP', 'FNF', 'FAF', 'NMIH', 'ESNT'],

                'Consumer Finance': ['AXP', 'COF', 'DFS', 'SYF', 'ALLY', 'OMF', 'ENVA', 'SLM'],

                'Payments Networks': ['V', 'MA', 'AXP', 'DFS'],

                'Payments Processors': ['FI', 'FIS', 'GPN', 'PYPL', 'XYZ'],

                'Fintech Lenders and Platforms': ['HOOD', 'SOFI', 'AFRM', 'UPST', 'XYZ', 'PYPL', 'NU'],

                'Crypto and Tokenization Proxies': ['COIN', 'HOOD', 'MSTR', 'PYPL', 'XYZ', 'IBKR']},



 'Health Care': {

		'Large-Cap Pharma': ['LLY', 'JNJ', 'MRK', 'PFE', 'BMY', 'ABBV', 'AZN', 'NVO', 'NVS', 'GSK'],

                 'Specialty Pharma': ['VRTX', 'ALNY', 'BMRN', 'RARE', 'IONS', 'HALO', 'INCY'],

                 'Large-Cap Biotech': ['AMGN', 'GILD', 'REGN', 'BIIB', 'VRTX', 'ALNY'],

                 'Oncology and Immunology': ['MRK', 'BMY', 'RHHBY', 'REGN', 'VRTX', 'INCY', 'EXEL'],

                 'GLP-1 and Metabolic': ['LLY', 'NVO', 'AZN', 'MRK', 'PFE', 'VKTX', 'AMGN'],

                 'Life Science Tools': ['TMO', 'DHR', 'A', 'RGEN', 'ILMN', 'WAT', 'BRKR'],

                 'Diagnostics': ['LH', 'DGX', 'TMO', 'DHR', 'A', 'ILMN', 'EXAS'],

                 'Medical Devices': ['MDT', 'SYK', 'ISRG', 'BSX', 'ZBH', 'EW', 'PEN', 'ABT'],

                 'Surgical Robotics and Advanced Devices': ['ISRG', 'SYK', 'MDT', 'BSX', 'DXCM', 'TMDX'],

                 'Orthopedics and Spine': ['SYK', 'ZBH', 'GMED', 'MDT', 'SNN', 'OFIX'],

                 'Cardiovascular Devices': ['BSX', 'MDT', 'ABT', 'EW', 'PEN', 'TMDX'],

                 'Diabetes Devices': ['DXCM', 'PODD', 'ABT', 'MDT', 'TNDM'],

                 'Managed Care': ['UNH', 'HUM', 'CI', 'ELV', 'CNC', 'MOH', 'OSCR'],

                 'Hospitals and Providers': ['HCA', 'THC', 'UHS', 'CYH', 'EHC'],

                 'Healthcare Services': ['MCK', 'COR', 'CAH', 'CVS', 'CI', 'HCA', 'THC'],

                 'Drug Distributors': ['MCK', 'COR', 'CAH', 'CVS', 'CI'],

                 'CRO and Clinical Services': ['IQV', 'MEDP', 'ICLR', 'CRL', 'LH', 'DGX'],

                 'CDMO and Bioprocessing': ['RGEN', 'DHR', 'TMO', 'WST', 'CTLT', 'BIO'],

                 'Healthcare IT': ['VEEV', 'TDOC', 'DOCS', 'OMCL', 'EVH', 'HIMS'],

                 'Dental Vision and Elective Care': ['ALGN', 'HSIC', 'EYE', 'WRBY', 'XRAY'],

                 'Animal Health': ['ZTS', 'ELAN', 'IDXX', 'PETS', 'HSIC']},



 'Consumer Discretionary': {

		'Autos Legacy OEMs': ['TM', 'HMC', 'F', 'GM', 'STLA', 'VWAGY'],

                'Electric Vehicles': ['TSLA', 'RIVN', 'LCID', 'NIO', 'LI', 'XPEV', 'RACE'],

                'Auto Parts and Suppliers': ['APTV', 'BWA', 'LEA', 'MGA', 'ALV', 'GPC', 'LKQ'],

                'Auto Dealers': ['LAD', 'AN', 'PAG', 'ABG', 'KMX', 'CVNA'],

                'Auto Repair and Aftermarket Retail': ['AZO', 'ORLY', 'AAP', 'GPC', 'LKQ'],

                'Homebuilders': ['DHI', 'LEN', 'NVR', 'PHM', 'TOL', 'KBH', 'MTH'],

                'Home Improvement Retail': ['HD', 'LOW', 'TSCO', 'POOL', 'BLDR'],

                'Building Products': ['BLDR', 'TREX', 'MAS', 'OC', 'JELD', 'FBIN'],

                'Apparel and Footwear': ['NKE', 'LULU', 'DECK', 'ONON', 'UAA', 'RL', 'TPR'],

                'Luxury Goods': ['LVMUY', 'RACE', 'TPR', 'RL', 'CPRI', 'CFRUY', 'PPRUY'],

                'Beauty Retail and Cosmetics': ['EL', 'ULTA', 'COTY', 'LRLCY', 'ELF'],

                'Department and Specialty Retail': ['M', 'KSS', 'JWN', 'BBY', 'DKS', 'WSM', 'ANF'],

                'Off-Price Retail': ['TJX', 'ROST', 'BURL'],

                'E-Commerce': ['AMZN', 'SHOP', 'MELI', 'ETSY', 'EBAY', 'PDD', 'BABA', 'JD', 'SE'],

                'Restaurants and QSR': ['MCD', 'YUM', 'QSR', 'WEN', 'DPZ', 'CMG'],

                'Casual Dining': ['DRI', 'TXRH', 'EAT', 'BLMN', 'CAKE', 'WING'],

                'Hotels': ['MAR', 'HLT', 'H', 'IHG', 'WH'],

                'Casinos and Gaming': ['MGM', 'LVS', 'WYNN', 'MLCO', 'CZR', 'PENN'],

                'Cruise Lines': ['RCL', 'CCL', 'NCLH'],

                'Airlines': ['DAL', 'UAL', 'AAL', 'LUV', 'ALK', 'JBLU'],

                'Travel Booking': ['BKNG', 'EXPE', 'ABNB', 'TRIP'],

                'Leisure Products': ['BC', 'PII', 'HOG', 'HAS', 'MAT', 'YETI'],

                'Consumer Electronics Retail': ['BBY', 'AAPL', 'GME', 'WMT', 'COST']},



 'Communication Services': {

		'Telecom Carriers': ['T', 'VZ', 'TMUS', 'CHTR', 'CMCSA', 'LUMN'],

                'Wireless Infrastructure': ['AMT', 'SBAC', 'CCI', 'T', 'VZ', 'TMUS'],

                'Cable and Broadband': ['CMCSA', 'CHTR', 'CABO', 'LBRDA', 'ATUS'],

                'Search and Digital Advertising': ['GOOGL', 'META', 'TTD', 'PINS', 'SNAP', 'APP'],

                'Social Media and Platforms': ['META', 'SNAP', 'PINS', 'RDDT', 'MTCH', 'BMBL'],

                'Traditional Media': ['DIS', 'WBD', 'PARA', 'FOX', 'FOXA', 'NYT'],

                'Streaming Video': ['NFLX', 'DIS', 'WBD', 'PARA', 'ROKU', 'CMCSA'],

                'Music and Audio': ['SPOT', 'SIRI', 'LYV', 'WMG'],

                'Video Games': ['EA', 'TTWO', 'RBLX', 'NTDOY', 'SONY', 'MSFT'],

                'Interactive Entertainment': ['RBLX', 'EA', 'TTWO', 'U', 'NTDOY', 'SONY'],

                'Live Entertainment and Sports': ['LYV', 'TKO', 'MSGS', 'SPHR', 'BATRA', 'FWONA'],

                'Advertising Agencies': ['OMC', 'IPG', 'WPP', 'PUBGY'],

                'China Internet and Gaming': ['BABA', 'JD', 'PDD', 'BIDU', 'BILI', 'NTES', 'TCEHY']},



 'Industrials': {

		'Aerospace and Defense Primes': ['LMT', 'NOC', 'RTX', 'GD', 'HII', 'BAESY', 'RNMBY'],

                'Commercial Aerospace OEMs': ['BA', 'EADSY', 'TXT', 'GE'],

                'Aerospace Suppliers and Aftermarket': ['GE', 'RTX', 'HEI', 'TDG', 'FTAI', 'HWM', 'SPR'],

                'Defense Electronics and ISR': ['LHX', 'LDOS', 'BAH', 'CACI', 'KTOS', 'AVAV', 'PLTR'],

                'Drones and Autonomous Defense': ['AVAV', 'KTOS', 'LMT', 'NOC', 'PLTR', 'TXT'],


                'Space and Satellites': ['RKLB', 'IRDM', 'ASTS', 'LHX', 'LMT', 'NOC', 'VSAT', 'GSAT'],

                'Machinery and Heavy Equipment': ['CAT', 'DE', 'CNHI', 'AGCO', 'PCAR', 'CMI'],

                'Electrical Equipment': ['ETN', 'GEV', 'HUBB', 'POWL', 'ABB', 'RRX', 'AME'],

                'Industrial Automation': ['ROK', 'ETN', 'EMR', 'AME', 'PH', 'ABB', 'KEYS', 'TRMB', 'CGNX', 'SYM'],

                'Sensors and Measurement': ['KEYS', 'TRMB', 'AME', 'TDY', 'FTV', 'ROK'],

                'Test and Measurement': ['KEYS', 'TER', 'FTV', 'TDY', 'A', 'COHU'],

                'Engineering and Construction': ['J', 'ACM', 'FLR', 'PWR', 'MTZ', 'FIX', 'EME'],

                'Building and Construction Services': ['PWR', 'MTZ', 'EME', 'FIX', 'J', 'ACM', 'URI'],

                'Rental Equipment and Tools': ['URI', 'HRI', 'FTAI', 'GWW', 'FAST'],

                'Industrial Distribution': ['GWW', 'FAST', 'AIT', 'WCC', 'SITE'],

                'Waste and Environmental Services': ['WM', 'RSG', 'WCN', 'CLH', 'SRCL'],

                'Railroads': ['UNP', 'CSX', 'NSC', 'CNI', 'CP'],

                'Parcel and Logistics': ['FDX', 'UPS', 'GXO', 'XPO', 'CHRW'],

                'Trucking and LTL': ['ODFL', 'SAIA', 'JBHT', 'KNX', 'ARCB', 'XPO'],

                'Air Cargo and Aircraft Leasing': ['AL', 'FTAI', 'AER', 'ATSG', 'CPA'],

                'Marine Transportation': ['MATX', 'ZIM', 'DAC', 'GSL', 'CMRE', 'SFL']},



 'Consumer Staples': {

		'Grocery and Clubs': ['WMT', 'COST', 'KR', 'BJ', 'ACI'],

                'Discount Staples Retail': ['WMT', 'COST', 'DG', 'DLTR', 'BJ'],

                'Packaged Food': ['GIS', 'K', 'CPB', 'CAG', 'HSY', 'MDLZ', 'SJM', 'KHC'],

                'Food Distribution': ['SYY', 'USFD', 'PFGC', 'CHEF'],

                'Protein and Meat Producers': ['TSN', 'HRL', 'PPC', 'CALM'],

                'Beverages Non-Alcoholic': ['KO', 'PEP', 'KDP', 'MNST', 'CELH'],

                'Alcoholic Beverages': ['STZ', 'TAP', 'BUD', 'DEO', 'BF-B'],

                'Tobacco': ['PM', 'MO', 'BTI', 'UVV'],

                'Household Products': ['PG', 'CL', 'KMB', 'CHD', 'CLX'],

                'Personal Care': ['EL', 'KVUE', 'COTY', 'ELF', 'LRLCY'],

                'Agricultural Products': ['ADM', 'BG', 'DAR', 'INGR']},



 'Energy': {

		'Integrated Energy Majors': ['XOM', 'CVX', 'COP', 'SHEL', 'BP', 'TTE', 'EQNR', 'PBR'],

            	'Oil-Weighted E&Ps': ['EOG', 'FANG', 'OXY', 'DVN', 'COP', 'PR', 'MTDR', 'APA'],

            	'Gas-Weighted E&Ps': ['EQT', 'AR', 'RRC', 'CTRA', 'EXE', 'CRK', 'CNX'],

            	'Canadian Oil Sands': ['CNQ', 'SU', 'CVE', 'IMO', 'MEGEF'],

            	'Oilfield Services': ['SLB', 'HAL', 'BKR', 'NOV', 'FTI', 'PTEN', 'HP', 'NBR', 'OII'],

            	'Offshore Drilling': ['RIG', 'VAL', 'NE', 'BORR', 'DO', 'SDRL'],

            	'Refiners and Downstream': ['MPC', 'VLO', 'PSX', 'DK', 'PBF', 'SUN'],

            	'Midstream Pipelines': ['KMI', 'WMB', 'EPD', 'ET', 'ENB', 'MPLX', 'OKE', 'TRP'],

            	'LNG Export and Gas Infrastructure': ['LNG', 'KMI', 'WMB', 'ET', 'EPD', 'WDS', 'NEXT'],

            	'Coal': ['BTU', 'ARCH', 'CEIX', 'AMR', 'WHC.AX'],

            	'Uranium Miners and Fuel Cycle': ['CCJ', 'UUUU', 'UEC', 'URG', 'UROY', 'DNN', 'NXE', 'LEU'],

            	'Renewable Fuels': ['DAR', 'GPRE', 'GEVO', 'AMTX']},



 'Utilities': {



		'Regulated Electric Utilities': ['NEE', 'DUK', 'SO', 'AEP', 'XEL', 'EXC', 'PCG', 'EIX', 'ED'],

               	'Gas Utilities': ['ATO', 'NI', 'OGS', 'SR', 'UGI', 'NWN'],

               	'Water Utilities': ['AWK', 'WTRG', 'AWR', 'CWT', 'MSEX', 'SJW'],

               	'Independent Power Producers': ['VST', 'CEG', 'NRG', 'TLN', 'AES', 'CWEN'],

               	'Merchant Power': ['VST', 'NRG', 'TLN', 'CEG'],

               	'Nuclear Utilities': ['CEG', 'VST', 'DUK', 'SO', 'EXC', 'NEE'],

               	'Renewable Power Utilities': ['NEE', 'BEP', 'CWEN', 'AY', 'AES', 'ORA'],

               	'Grid and Transmission': ['NEE', 'AEP', 'XEL', 'ETR', 'PWR', 'MYRG', 'ETN'],

               	'Data Center Power Beneficiaries': ['VST', 'CEG', 'NRG', 'TLN', 'ETN', 'VRT', 'GEV', 'PWR'],

               	'Utility Equipment and Services': ['ETN', 'GEV', 'HUBB', 'POWL', 'PWR', 'MYRG', 'VRT']},



 'Real Estate': {

		'Data Center REITs': ['EQIX', 'DLR'],

                'Tower REITs': ['AMT', 'SBAC', 'CCI'],

                'Industrial REITs': ['PLD', 'REXR', 'EGP', 'STAG', 'TRNO'],

                'Apartment REITs': ['AVB', 'EQR', 'UDR', 'ESS', 'MAA', 'CPT'],

                'Single-Family Rental REITs': ['INVH', 'AMH'],

                'Office REITs': ['BXP', 'VNO', 'SLG', 'KRC', 'DEI'],

                'Retail REITs': ['SPG', 'REG', 'KIM', 'FRT', 'BRX', 'KRG'],

                'Mall REITs': ['SPG', 'MAC', 'SKT', 'PECO'],

                'Net Lease REITs': ['O', 'NNN', 'ADC', 'WPC', 'EPRT'],

                'Self-Storage REITs': ['PSA', 'EXR', 'CUBE', 'NSA'],

                'Hotel REITs': ['HST', 'RHP', 'APLE', 'PK', 'SHO'],

                'Healthcare REITs': ['WELL', 'VTR', 'OHI', 'SBRA', 'DOC'],

                'Real Estate Services': ['CBRE', 'JLL', 'CWK', 'Z', 'RDFN']},



 'Materials': {

		'Commodity Chemicals': ['DOW', 'LYB', 'CE', 'WLK', 'OLN'],

               	'Specialty Chemicals': ['SHW', 'PPG', 'EMN', 'RPM', 'IFF', 'ALB'],

               	'Industrial Gases': ['LIN', 'APD', 'AIQUY'],

               	'Fertilizers': ['CF', 'MOS', 'NTR', 'IPI'],

               	'Agricultural Chemicals': ['CTVA', 'FMC', 'NTR', 'MOS', 'CF'],

               	'Steel': ['NUE', 'STLD', 'CLF', 'RS', 'CMC', 'GGB'],

               	'Aluminum': ['AA', 'CENX', 'ACH'],

               	'Copper Miners': ['FCX', 'SCCO', 'TECK', 'ERO', 'IVPAF'],

               	'Diversified Metals and Mining': ['BHP', 'RIO', 'VALE', 'GLNCY', 'TECK', 'AA'],

               	'Gold Miners': ['NEM', 'AEM', 'GOLD', 'KGC', 'AU', 'EGO'],

               	'Silver Miners': ['AG', 'PAAS', 'WPM', 'HL', 'MAG'],

               	'Lithium and Battery Metals': ['ALB', 'SQM', 'LAC', 'MP', 'VALE', 'BHP'],

               	'Rare Earths and Magnets': ['MP', 'LYSDY', 'UUUU', 'NEM'],


               	'Aggregates and Cement': ['VMC', 'MLM', 'SUM', 'EXP', 'CRH', 'CX'],

               	'Packaging': ['BALL', 'CCK', 'AMCR', 'SEE', 'PKG', 'IP'],

               	'Paper and Forest Products': ['IP', 'PKG', 'SUZ', 'RYAM']},



 'Thematic Cross-Sector Baskets': {'AI Data Center Capex': ['NVDA','AMD','AVGO','MRVL','ANET','VRT','ETN','GEV','SMCI','DELL','TSM','ASML'],

                                   'AI Power Demand': ['VST', 'CEG', 'NRG', 'TLN', 'ETN', 'VRT', 'GEV', 'PWR', 'NEE'],

                                   'AI Margin Expansion Beneficiaries': ['MCK','COR','CAH','UNH','CI','ACN','IBM','FIS','FI','ADP','PAYX'],

                                   'AI Application Layer': ['MSFT','NOW','CRM','ADBE','INTU','PLTR','SNOW','MDB','DDOG'],

                                   'AI Hardware Supply Chain': ['NVDA','AMD','AVGO','TSM','ASML','AMAT','LRCX','KLAC','MU','ANET','VRT','DELL'],

                                   'Sovereign AI Infrastructure': ['NVDA','AMD','AVGO','TSM','ASML','ANET','VRT','ETN','ORCL','HPE'],

                                   'Data Center Construction and EPC': ['PWR','MYRG','EME','FIX','J','ACM','VRT','ETN','GEV'],

                                   'Reindustrialization': ['CAT','DE','ETN','PWR','URI','VMC','MLM','NUE','STLD','GEV'],

                                   'Reshoring and Factory Buildout': ['PWR','MTZ','EME','FIX','URI','CAT','VMC','MLM','ROK','ABB'],

                                   'North American Onshoring Materials': ['VMC','MLM','SUM','NUE','STLD','X','CLF','FCX','EXP'],

                                   'Nearshoring Mexico': ['EWW', 'KOF', 'FMX', 'CX', 'AMX', 'PAC', 'OMAB', 'ASR'],

                                   'Defense Modernization': ['LMT','NOC','RTX','GD','LHX','PLTR','KTOS','AVAV','LDOS','BAH'],

                                   'NATO Re-Armament': ['BAESY','RNMBY','SAABY','THLLY','ESLT'],

                                   'GLP-1 Winners': ['LLY', 'NVO', 'VKTX', 'AMGN', 'TMO', 'DHR', 'MCK', 'COR'],

                                   'GLP-1 Consumer Losers': ['HSY', 'MDLZ', 'PEP', 'KO', 'MCD', 'YUM', 'DPZ', 'KDP'],

                                   'Longevity and Aging Population': ['LLY','NVO','ISRG','SYK','MDT','MCK','COR','WELL','VTR','HCA'],

                                   'Healthcare Supply Chain Automation': ['MCK','COR','CAH','UNH','CI','ACN','CTSH','PLTR'],

                                   'Crypto and Tokenization Proxies': ['COIN','MSTR','HOOD','MARA','RIOT','CLSK','IBIT','ETHA'],

                                   'Speculative Liquidity and Retail Beta': ['CVNA','UPST','RIVN','COIN','MSTR','HOOD','SOFI','MARA','RIOT','PLTR'],

                                   'Sports Live Events and Experiences': ['LYV','TKO','MSGS','SPHR'],

                                   'Premium Consumer': ['LULU', 'NKE', 'ONON', 'RACE', 'LVMUY', 'COST', 'CMG', 'MAR'],

                                   'Trade-Down Consumer': ['WMT', 'COST', 'TJX', 'ROST', 'BURL', 'DG', 'DLTR', 'OLLI'],

                                   'Housing Affordability Stress': ['RKT','UWMC','COOP','INVH','AMH','AVB'],

                                   'Mortgage Rate Sensitive Housing': ['ITB','DHI','LEN','PHM','TOL','RKT','UWMC','FNF'],

                                   'Tariff Beneficiaries': ['NUE','STLD','CLF','X','CAT','DE','GEV','ETN','PWR'],

                                   'Japan Reflation Beneficiaries': ['EWJ', 'DXJ', 'TM', 'HMC', 'MUFG', 'SMFG', 'SONY'],

                                   'Europe Fiscal Expansion': ['VGK', 'EZU', 'EWG', 'BAESY', 'RNMBY', 'EADSY', 'SIEGY'],

                                   'Humanoid Robotics': ['TSLA', 'NVDA', 'ISRG','BOT', 'SYM', 'ROK', 'TER', 'FANUY', 'ABB'],

                                   'Quantum Computing': ['IONQ', 'RGTI', 'QBTS', 'QUBT', 'IBM', 'GOOGL', 'MSFT'],

                                   'Autonomous Vehicles and Robotaxis': ['TSLA','GOOGL','GM','UBER','MBLY']},

 'Countries and Regions': {'Developed ex-US': ['VEA'],

                           'Europe Broad': ['VGK'],

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

                           'China Broad': ['MCHI', 'FXI'],

                           'China Internet': ['KWEB', 'BABA', 'JD', 'PDD', 'BIDU', 'BILI', 'NTES'],

                           'Taiwan': ['EWT', 'TSM'],

                           'South Korea': ['EWY'],

                           'India': ['INDA', 'EPI', 'INDY'],

                           'Indonesia': ['EIDO', 'IDX'],

                           'Vietnam': ['VNM'],

                           'Emerging Markets Broad': ['IEMG', 'EEM'],

                           'Latin America Broad': ['ILF'],

                           'Brazil': ['EWZ'],

                           'Mexico': ['EWW'],

                           'Argentina': ['ARGT'],

                           'Chile': ['ECH'],

                           'Peru': ['EPU'],

                           'Poland': ['EPOL'],

                           'Turkey': ['TUR'],

                           'Middle East Broad': ['KSA', 'UAE', 'QAT', 'EIS', 'TUR'],

                           'Saudi Arabia': ['KSA'],

                           'UAE': ['UAE'],

                           'Israel': ['EIS'],

                           'South Africa': ['EZA']},

 'Macro Dislocation and Special Situations': {'Tanker Shipping': ['FRO','STNG','INSW','TNK','DHT','NAT','TRMD','HAFN','CMBT','ASC'],


                                              'Critical Minerals: Tungsten and Antimony': ['ALM','PPTA','TUN.L','USAR','UAMY'],

                                              'eVTOL and Urban Air Mobility': ['JOBY', 'ACHR', 'EH', 'EVTL', 'BLDE'],

                                              'Greece Reclassification': ['GREK','NBG','EUROB.AT','OPAP.AT','MYTIL.AT'],

                                              'Cannabis and MSOs': ['MSOS','CURLF','GTBIF','TCNNF','VRNOF','CRLBF','TRUL.CN'],

                                              'Shipping and Geopolitical Tonne-Mile Risk': ['FRO','DHT','INSW','TNK','STNG','TRMD','ASC','FLNG','LPG','NVGS'],

                                              'Energy Services: Offshore and Deepwater': ['RIG','VAL','NE','BORR','DO','SLB','HAL','BKR','FTI'],

                                              'Strategic Minerals and Defense Metals': ['USAR','UAMY','MP','PPTA','ALM','TUN.L','UUUU','CRML'],

                                              'Speculative Aviation and Autonomy': ['JOBY','ACHR','EH','EVTL','BLDE','RKLB','ASTS','IRDM'],

                                              'Dry Bulk Shipping': ['SBLK', 'GNK', 'GOGL', 'EGLE', 'SB', 'DSX', 'PANL'],

                                              'LNG and LPG Shipping': ['FLNG', 'GLNG', 'LPG', 'BWLP', 'NVGS', 'SFL'],

                                              'Marine Insurance and Reinsurance': ['ACGL','RNR','EG','AXS','CB','TRV','WRB'],

                                              'Coal and Baseload Scarcity': ['CNR','BTU','AMR','HCC','METC','TECK','BHP','RIO'],

                                              'Nuclear Fuel, Enrichment and Services': ['LEU','BWXT','CCJ','UUUU','UEC','NXE','DNN','URNM'],

                                              'Geothermal and Clean Firm Power': ['ORA','BE','OKLO','SMR','CEG','VST','TLN','NEE'],

                                              'Gold Royalty and Streamers': ['FNV','WPM','RGLD','OR','SAND','TFPM'],

                                              'Silver Pure Play': ['SLV','SIL','PAAS','AG','HL','CDE','FSM','MAG'],

                                              'Volatility and Market Plumbing': ['CBOE','CME','ICE','NDAQ','VIRT','TW','IBKR','SCHW','MKTX'],

                                              'Credit Data and Underwriting': ['FICO','EFX','TRU','SPGI','MCO','EXPGY'],

                                              'Argentina Reform': ['ARGT','YPF','GGAL','BMA','BBAR','PAM','TGS','CEPU','LOMA']}}

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





def _cache_key(tickers: List[str], start: pd.Timestamp) -> str:

    payload = {

        "version": CACHE_VERSION,

        "tickers": sorted(tickers),

        "start": str(pd.Timestamp(start).date()),

    }

    raw = json.dumps(payload, sort_keys=True).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()[:20]





def _cache_paths(cache_key: str) -> Tuple[Path, Path]:

    stem = CACHE_DIR / f"basket_levels_{cache_key}"

    return Path(f"{stem}.pkl"), Path(f"{stem}.json")





def save_last_good_levels(

    levels: pd.DataFrame,

    meta: Dict[str, Any],

    cache_key: str,

) -> Optional[str]:

    levels_path, meta_path = _cache_paths(cache_key)

    tmp_levels = Path(f"{levels_path}.tmp")

    tmp_meta = Path(f"{meta_path}.tmp")

    try:

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        levels.to_pickle(tmp_levels)

        with tmp_meta.open("w", encoding="utf-8") as f:

            json.dump(meta, f, indent=2, default=str)

        tmp_levels.replace(levels_path)

        tmp_meta.replace(meta_path)

        return None

    except Exception as exc:

        for path in (tmp_levels, tmp_meta):

            try:

                path.unlink(missing_ok=True)

            except Exception:

                pass

        return str(exc)





def load_last_good_levels(cache_key: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    levels_path, meta_path = _cache_paths(cache_key)

    try:

        if not levels_path.exists():

            return pd.DataFrame(), {}

        levels = pd.read_pickle(levels_path)

        meta: Dict[str, Any] = {}

        if meta_path.exists():

            with meta_path.open("r", encoding="utf-8") as f:

                meta = json.load(f)

        return _clean_index(levels), meta

    except Exception as exc:

        return pd.DataFrame(), {"cache_load_error": str(exc)}





def _cache_is_usable(

    levels: pd.DataFrame,

    requested_tickers: List[str],

    start: pd.Timestamp,

    end: pd.Timestamp,

) -> bool:

    if levels.empty or BENCH not in levels.columns:

        return False

    bench = levels[BENCH].dropna()

    if bench.empty:

        return False

    if bench.index.min() > pd.Timestamp(start) + pd.Timedelta(days=10):

        return False

    latest_acceptable = pd.Timestamp(end) - pd.Timedelta(days=CACHE_MAX_AGE_DAYS)

    if bench.index.max() < latest_acceptable:

        return False

    available = sum(

        1 for ticker in requested_tickers

        if ticker in levels.columns and not levels[ticker].dropna().empty

    )

    return available / max(len(requested_tickers), 1) >= 0.70





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

    start = pd.Timestamp(start).tz_localize(None) if pd.Timestamp(start).tzinfo else pd.Timestamp(start)

    end = pd.Timestamp(end).tz_localize(None) if pd.Timestamp(end).tzinfo else pd.Timestamp(end)

    cache_key = _cache_key(uniq, start)

    cached, cached_meta = load_last_good_levels(cache_key)

    cache_usable = _cache_is_usable(cached, uniq, start, end)

    frames: List[pd.DataFrame] = []

    failed_batches: List[List[str]] = []

    consecutive_failed_batches = 0



    for batch in _chunk(uniq, chunk_size):

        close = _download_close(batch, start=start, end=end)

        if close.empty:

            failed_batches.append(batch)

            consecutive_failed_batches += 1

            if cache_usable and consecutive_failed_batches >= 3:

                break

        else:

            frames.append(close)

            consecutive_failed_batches = 0



    wide = pd.concat(frames, axis=1) if frames else pd.DataFrame()

    if not wide.empty:

        wide = wide.loc[:, ~wide.columns.duplicated()].sort_index()

        wide = _to_float_frame(_clean_index(wide))



    missing_after_batches = sorted(set(uniq) - set(wide.columns))

    retry_frames: List[pd.DataFrame] = []

    retry_batches = _chunk(missing_after_batches, 8) if frames else []

    for batch in retry_batches:

        retry = _download_close(batch, start=start, end=end, retries=1)

        if not retry.empty:

            retry_frames.append(retry)

    if retry_frames:

        wide = pd.concat([wide, *retry_frames], axis=1)

        wide = wide.loc[:, ~wide.columns.duplicated(keep="last")]

        wide = _to_float_frame(_clean_index(wide))



    if BENCH not in wide.columns or wide[BENCH].dropna().empty:

        spy_only = _download_close([BENCH], start=start, end=end)

        if not spy_only.empty:

            wide = pd.concat([wide.drop(columns=[BENCH], errors="ignore"), spy_only], axis=1)

            wide = wide.loc[:, ~wide.columns.duplicated()].sort_index()

    cache_used = False

    if cache_usable:

        cached = cached.loc[(cached.index >= start) & (cached.index < end)]

        if wide.empty:

            wide = cached.copy()

            cache_used = True

        else:

            before_columns = set(wide.columns)

            wide = wide.combine_first(cached)

            cache_used = bool(set(uniq) - before_columns)



    if wide.empty:

        return pd.DataFrame(), {

            "source": "yahoo",

            "requested_tickers": len(uniq),

            "returned_tickers": 0,

            "failed_batches": failed_batches,

            "cache_meta": cached_meta,

        }



    wide = wide.loc[(wide.index >= start) & (wide.index < end)]

    wide = wide.loc[:, ~wide.columns.duplicated()]

    wide = _to_float_frame(_clean_index(wide)).dropna(axis=1, how="all")

    source = "last_good_cache" if not frames and cache_used else ("yahoo+cache" if cache_used else "yahoo")

    last_observation = None

    if BENCH in wide.columns and not wide[BENCH].dropna().empty:

        last_observation = str(wide[BENCH].dropna().index.max().date())

    meta = {

        "source": source,

        "requested_tickers": len(uniq),

        "returned_tickers": int(wide.shape[1]),

        "missing_tickers": sorted(set(uniq) - set(wide.columns)),

        "failed_batches": failed_batches,

        "start": str(start.date()),

        "end": str(end.date()),

        "last_observation": last_observation,

        "cache_meta": cached_meta if cache_used else {},

    }



    coverage = wide.shape[1] / max(len(uniq), 1)

    if frames and BENCH in wide.columns and coverage >= 0.70:

        cache_error = save_last_good_levels(wide, meta, cache_key)

        if cache_error:

            meta["cache_save_error"] = cache_error



    return wide, meta





@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)

def fetch_market_metadata(tickers: List[str]) -> Dict[str, Dict[str, Any]]:

    uniq = sorted({str(t).upper().strip() for t in tickers if str(t).strip()})

    if not uniq:

        return {}



    metadata: Dict[str, Dict[str, Any]] = {}



    for batch in _chunk(uniq, 75):

        try:

            tk_obj = yf.Tickers(" ".join(batch))



            for sym, tk in tk_obj.tickers.items():

                try:

                    sym_upper = str(sym).upper()

                    mc_val = None

                    quote_type = None

                    currency = None



                    fast_info = getattr(tk, "fast_info", None)

                    if fast_info is not None:

                        if isinstance(fast_info, dict):

                            mc_val = fast_info.get("market_cap")

                            currency = fast_info.get("currency")

                        else:

                            mc_val = getattr(fast_info, "market_cap", None)

                            currency = getattr(fast_info, "currency", None)



                    info = getattr(tk, "info", {}) or {}

                    if mc_val is None:

                        mc_val = info.get("marketCap")

                    quote_type = info.get("quoteType")

                    currency = currency or info.get("currency")



                    metadata[sym_upper] = {

                        "market_cap": float(mc_val) if mc_val is not None and pd.notna(mc_val) else None,

                        "quote_type": str(quote_type).upper() if quote_type else None,

                        "currency": str(currency) if currency else None,

                    }



                except Exception:

                    continue



        except Exception:

            continue



    return metadata



def normalize_basket_members(

    levels: pd.DataFrame,

    basket_tickers: List[str],

    market_metadata: Optional[Dict[str, Dict[str, Any]]] = None,

    min_market_cap: Optional[float] = None,

    stale_days: int = 30,

    reference_date: Optional[pd.Timestamp] = None,

) -> List[str]:

    valid: List[str] = []



    if levels.empty:

        return valid



    last_idx = pd.Timestamp(reference_date) if reference_date is not None else pd.Timestamp(levels.index.max())



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



        if min_market_cap is not None:

            record = (market_metadata or {}).get(sym, {})

            quote_type = str(record.get("quote_type") or "").upper()

            mc = record.get("market_cap")

            if quote_type not in FUND_QUOTE_TYPES:

                if mc is None or pd.isna(mc) or float(mc) < min_market_cap:

                    continue



        valid.append(sym)



    return valid





def build_live_baskets(

    levels: pd.DataFrame,

    categories: Dict[str, Dict[str, List[str]]],

    market_metadata: Dict[str, Dict[str, Any]],

    min_market_cap: Optional[float],

    stale_days: int,

    reference_date: pd.Timestamp,

) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, Any]], List[str]]:

    live_categories: Dict[str, Dict[str, List[str]]] = {}

    basket_metadata: Dict[str, Dict[str, Any]] = {}

    dropped_baskets: List[str] = []



    for category, baskets in categories.items():

        live_baskets: Dict[str, List[str]] = {}



        for basket_name, tickers in baskets.items():

            live_members = normalize_basket_members(

                levels=levels,

                basket_tickers=tickers,

                market_metadata=market_metadata,

                min_market_cap=min_market_cap,

                stale_days=stale_days,

                reference_date=reference_date,

            )

            defined_count = len({str(t).upper().strip() for t in tickers if str(t).strip()})

            required_count = 1 if defined_count <= 1 else max(

                2,

                int(math.ceil(defined_count * MIN_LIVE_MEMBER_COVERAGE)),

            )

            key = basket_key(category, basket_name)

            if len(live_members) >= required_count:

                live_baskets[basket_name] = live_members

                basket_metadata[key] = {

                    "Category": category,

                    "Basket": basket_name,

                    "Live Members": len(live_members),

                    "Defined Members": defined_count,

                    "Members": f"{len(live_members)}/{defined_count}",

                    "Coverage %": round((len(live_members) / defined_count) * 100.0, 1) if defined_count else np.nan,

                }

            else:

                dropped_baskets.append(key)



        if live_baskets:

            live_categories[category] = live_baskets



    return live_categories, basket_metadata, dropped_baskets





def basket_key(category: str, basket_name: str) -> str:

    return f"{category}{BASKET_KEY_SEPARATOR}{basket_name}"





def flatten_baskets(categories: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:

    return {

        basket_key(category, basket_name): tickers

        for category, groups in categories.items()

        for basket_name, tickers in groups.items()

    }





def unique_tickers_from_baskets(baskets: Dict[str, List[str]], extra: Optional[Iterable[str]] = None) -> List[str]:

    tickers = {str(t).upper().strip() for members in baskets.values() for t in members if str(t).strip()}



    if extra:

        tickers.update(str(t).upper().strip() for t in extra if str(t).strip())



    return sorted(tickers)





def required_fx_tickers(tickers: Iterable[str]) -> List[str]:

    return sorted({FX_CONVERSIONS[ticker][0] for ticker in tickers if ticker in FX_CONVERSIONS})





def convert_foreign_levels_to_usd(levels: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:

    if levels.empty:

        return levels, []

    out = levels.copy()

    issues: List[str] = []

    for ticker, (fx_ticker, operation) in FX_CONVERSIONS.items():

        if ticker not in out.columns:

            continue

        if fx_ticker not in out.columns or out[fx_ticker].dropna().empty:

            out[ticker] = np.nan

            issues.append(f"{ticker}: missing {fx_ticker}")

            continue

        local = out[ticker]

        fx = out[fx_ticker]

        union_index = local.index.union(fx.index).sort_values()

        fx_aligned = fx.reindex(union_index).ffill(limit=MAX_FORWARD_FILL_SESSIONS).reindex(local.index)

        if operation == "multiply":

            out[ticker] = local * fx_aligned

        elif operation == "divide":

            out[ticker] = local / fx_aligned.replace(0, np.nan)

        else:

            out[ticker] = np.nan

            issues.append(f"{ticker}: unsupported FX operation {operation}")

    return out.drop(columns=required_fx_tickers(out.columns), errors="ignore"), issues





def convert_market_metadata_to_usd(

    metadata: Dict[str, Dict[str, Any]],

    raw_levels: pd.DataFrame,

) -> Dict[str, Dict[str, Any]]:

    out = {ticker: dict(record) for ticker, record in metadata.items()}

    for ticker, (fx_ticker, operation) in FX_CONVERSIONS.items():

        record = out.get(ticker)

        if not record or record.get("market_cap") is None:

            continue

        if fx_ticker not in raw_levels.columns or raw_levels[fx_ticker].dropna().empty:

            record["market_cap"] = None

            continue

        market_cap = float(record["market_cap"])

        currency = str(record.get("currency") or "")

        if currency in {"GBp", "GBX", "GBpence", "GBpenny"} and ticker.endswith(".L"):

            market_cap /= 100.0

        fx_value = float(raw_levels[fx_ticker].dropna().iloc[-1])

        if operation == "multiply":

            market_cap *= fx_value

        elif operation == "divide" and fx_value != 0:

            market_cap /= fx_value

        else:

            market_cap = np.nan

        record["market_cap"] = market_cap if pd.notna(market_cap) else None

    return out





def align_levels_to_calendar(

    levels: pd.DataFrame,

    calendar_index: pd.Index,

) -> pd.DataFrame:

    if levels.empty or len(calendar_index) == 0:

        return pd.DataFrame()

    calendar = pd.DatetimeIndex(calendar_index).sort_values().unique()

    return levels.reindex(calendar)



def ew_rets_from_levels(

    levels: pd.DataFrame,

    baskets: Dict[str, List[str]],

    min_daily_coverage: float = MIN_DAILY_MEMBER_COVERAGE,

) -> pd.DataFrame:

    if levels.empty:

        return pd.DataFrame()



    rets = pd.DataFrame(

        {

            column: levels[column].dropna().pct_change(fill_method=None).reindex(levels.index)

            for column in levels.columns

        },

        index=levels.index,

    )

    out: Dict[str, pd.Series] = {}



    for basket_name, basket_tickers in baskets.items():

        cols = [str(t).upper() for t in basket_tickers if str(t).upper() in rets.columns]

        if not cols:

            continue



        member_rets = rets[cols]

        required_count = max(1, int(math.ceil(len(cols) * min_daily_coverage)))

        valid_count = member_rets.notna().sum(axis=1)

        basket_ret = member_rets.mean(axis=1, skipna=True)

        basket_ret[valid_count < required_count] = np.nan



        price_count = levels[cols].notna().sum(axis=1)

        inception_dates = price_count[price_count >= required_count].index

        if len(inception_dates):

            inception_date = inception_dates[0]

            if pd.isna(basket_ret.loc[inception_date]):

                basket_ret.loc[inception_date] = 0.0



        out[basket_name] = basket_ret



    return pd.DataFrame(out).dropna(how="all")





def rsi(series: pd.Series, period: int = 14) -> pd.Series:

    delta = series.diff()

    gain = delta.clip(lower=0.0)

    loss = -delta.clip(upper=0.0)



    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()



    rs = avg_gain / avg_loss.replace(0, np.nan)

    out = 100 - (100 / (1 + rs))

    rising_only = (avg_loss == 0) & (avg_gain > 0)

    falling_only = (avg_gain == 0) & (avg_loss > 0)

    unchanged = (avg_gain == 0) & (avg_loss == 0)

    out = out.mask(rising_only, 100.0)

    out = out.mask(falling_only, 0.0)

    out = out.mask(unchanged, 50.0)

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



    directional_change = (latest - ref) * np.sign(latest)

    accel = "Accelerating" if directional_change > 0 else "Decelerating"

    strength = "Strong" if abs(z) > 1 else "Weak"



    return f"{base} | {accel} | {strength}"





def pct_since(levels: pd.Series, start_ts: pd.Timestamp) -> float:

    clean = levels.dropna()

    if clean.empty:

        return np.nan

    anchors = clean[clean.index <= pd.Timestamp(start_ts)]

    if anchors.empty:

        return np.nan

    anchor = anchors.iloc[-1]

    latest = clean.iloc[-1]

    if pd.isna(anchor) or anchor == 0 or pd.isna(latest):

        return np.nan

    return float((latest / anchor) - 1.0)





def compute_display_start(preset: str, today: date) -> date:

    presets = {

        "YTD": lambda t: date(t.year, 1, 1),

        "1W": lambda t: t - timedelta(days=7),

        "1M": lambda t: (pd.Timestamp(t) - pd.DateOffset(months=1)).date(),

        "3M": lambda t: (pd.Timestamp(t) - pd.DateOffset(months=3)).date(),

        "6M": lambda t: (pd.Timestamp(t) - pd.DateOffset(months=6)).date(),

        "1Y": lambda t: (pd.Timestamp(t) - pd.DateOffset(years=1)).date(),

        "3Y": lambda t: (pd.Timestamp(t) - pd.DateOffset(years=3)).date(),

        "5Y": lambda t: (pd.Timestamp(t) - pd.DateOffset(years=5)).date(),

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

    basket_metadata: Dict[str, Dict[str, Any]],

    benchmark_series_full: pd.Series,

) -> pd.DataFrame:

    dynamic_col = f"%{dynamic_label}"

    return_cols = ["%5D", "%1M"]

    if dynamic_col not in return_cols:

        return_cols.append(dynamic_col)

    relative_col = f"vs SPY {dynamic_label}"

    cols = [

        "Basket", *return_cols,

        "MACD Momentum", "EMA 4/9/18", "RSI(14W)", relative_col,

        "vs 21DMA %", "vs 50DMA %",

    ]



    if basket_returns_full.empty:

        return pd.DataFrame(columns=cols)



    levels_full = 100.0 * (1.0 + basket_returns_full).cumprod(skipna=True)

    rows: List[Dict[str, Any]] = []



    bench_levels = 100.0 * (1.0 + benchmark_series_full.dropna()).cumprod()

    bench_dynamic = pct_since(bench_levels, display_start)



    basket_name_counts: Dict[str, int] = {}

    for basket_id in levels_full.columns:

        meta = basket_metadata.get(basket_id, {})

        if BASKET_KEY_SEPARATOR in basket_id:

            _, fallback_basket = basket_id.split(BASKET_KEY_SEPARATOR, 1)

        else:

            fallback_basket = basket_id

        basket_name = str(meta.get("Basket", fallback_basket))

        basket_name_counts[basket_name] = basket_name_counts.get(basket_name, 0) + 1



    for basket_id in levels_full.columns:

        s_full = levels_full[basket_id].dropna()

        if s_full.shape[0] < 15:

            continue



        r5d = np.nan

        if s_full.shape[0] >= 6:

            r5d = (s_full.iloc[-1] / s_full.iloc[-6]) - 1.0



        r1m = pct_since(s_full, s_full.index.max() - pd.DateOffset(months=1))

        r_dyn = pct_since(s_full, display_start)

        relative_return = np.nan

        if pd.notna(r_dyn) and pd.notna(bench_dynamic) and (1.0 + bench_dynamic) != 0:

            relative_return = ((1.0 + r_dyn) / (1.0 + bench_dynamic)) - 1.0



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



        meta = basket_metadata.get(basket_id, {})

        if BASKET_KEY_SEPARATOR in basket_id:

            fallback_category, fallback_basket = basket_id.split(BASKET_KEY_SEPARATOR, 1)

        else:

            fallback_category, fallback_basket = "", basket_id

        category = str(meta.get("Category", fallback_category))

        basket_name = str(meta.get("Basket", fallback_basket))

        display_name = (

            f"{category} | {basket_name}"

            if basket_name_counts.get(basket_name, 0) > 1 and category

            else basket_name

        )



        row: Dict[str, Any] = {

            "_BasketKey": basket_id,

            "Basket": display_name,

            "%5D": round(r5d * 100, 1) if pd.notna(r5d) else np.nan,

            "%1M": round(r1m * 100, 1) if pd.notna(r1m) else np.nan,

            relative_col: round(relative_return * 100, 1) if pd.notna(relative_return) else np.nan,

            "MACD Momentum": macd_m,

            "EMA 4/9/18": ema_tag,

            "RSI(14W)": round(rsi_14w, 2) if pd.notna(rsi_14w) else np.nan,

            "vs 21DMA %": round(dma_21_pct, 1) if pd.notna(dma_21_pct) else np.nan,

            "vs 50DMA %": round(dma_50_pct, 1) if pd.notna(dma_50_pct) else np.nan,

        }

        row[dynamic_col] = round(r_dyn * 100, 1) if pd.notna(r_dyn) else np.nan

        rows.append(row)



    if not rows:

        return pd.DataFrame(columns=cols)



    df = pd.DataFrame(rows).set_index("_BasketKey")

    if dynamic_col in df.columns:

        df = df.sort_values(by=dynamic_col, ascending=False, na_position="last")

    return df[[column for column in cols if column in df.columns]]



def cumulative_return_since(returns: pd.Series, display_start: pd.Timestamp) -> pd.Series:

    clean = returns.dropna()

    if clean.empty:

        return pd.Series(dtype=float)

    levels = 100.0 * (1.0 + clean).cumprod()

    anchors = levels[levels.index <= pd.Timestamp(display_start)]

    if anchors.empty:

        return pd.Series(dtype=float)

    anchor = anchors.iloc[-1]

    if pd.isna(anchor) or anchor == 0:

        return pd.Series(dtype=float)

    display = levels[levels.index >= pd.Timestamp(display_start)]

    return ((display / anchor) - 1.0) * 100.0





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



    if x >= 80:

        return "rgb(255,237,170)"



    if x <= 20:

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




def plot_panel_table(panel_df: pd.DataFrame, dynamic_label: str):

    if panel_df.empty:

        st.info("No baskets passed the data-quality checks for this window.")

        return



    dynamic_col = f"%{dynamic_label}"

    relative_col = f"vs SPY {dynamic_label}"

    return_cols = ["%5D", "%1M"]

    if dynamic_col not in return_cols:

        return_cols.append(dynamic_col)



    headers = [

        "Basket", *return_cols,

        "MACD Momentum", "EMA 4/9/18", "RSI(14W)", relative_col,

        "vs 21DMA %", "vs 50DMA %",

    ]



    values: List[List[Any]] = [panel_df["Basket"].tolist()]

    fill_colors: List[List[str]] = [["white"] * len(panel_df)]



    for col in return_cols:

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



    vals = panel_df[relative_col].tolist()

    values.append(vals)

    fill_colors.append([color_ret(v) for v in vals])



    for col in ["vs 21DMA %", "vs 50DMA %"]:

        vals = panel_df[col].tolist()

        values.append(vals)

        fill_colors.append([color_ret(v) for v in vals])



    if dynamic_col in ["%5D", "%1M"]:

        col_widths = [0.27, 0.065, 0.065, 0.16, 0.11, 0.08, 0.08, 0.085, 0.085]

    else:

        col_widths = [0.25, 0.06, 0.06, 0.085, 0.155, 0.105, 0.075, 0.075, 0.085, 0.085]



    formats = []

    for header in headers:

        if header in {"Basket", "MACD Momentum", "EMA 4/9/18"}:

            formats.append(None)

        elif header == "RSI(14W)":

            formats.append(".2f")

        else:

            formats.append(".1f")



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

            format=formats

        )

    )])



    fig_tbl.update_layout(

        margin=dict(l=0, r=0, t=6, b=0),

        height=min(920, 64 + 26 * max(3, len(panel_df)))

    )



    st.plotly_chart(fig_tbl, use_container_width=True)




def _chart_display_name(

    basket_id: str,

    basket_metadata: Dict[str, Dict[str, Any]],

    duplicate_names: Dict[str, int],

) -> str:

    meta = basket_metadata.get(basket_id, {})

    if BASKET_KEY_SEPARATOR in basket_id:

        fallback_category, fallback_basket = basket_id.split(BASKET_KEY_SEPARATOR, 1)

    else:

        fallback_category, fallback_basket = "", basket_id

    category = str(meta.get("Category", fallback_category))

    basket_name = str(meta.get("Basket", fallback_basket))

    if duplicate_names.get(basket_name, 0) > 1 and category:

        return f"{category} | {basket_name}"

    return basket_name




def plot_cumulative_chart(

    basket_returns_full: pd.DataFrame,

    title: str,

    benchmark_returns_full: pd.Series,

    display_start: pd.Timestamp,

    basket_metadata: Dict[str, Dict[str, Any]],

):

    if basket_returns_full.empty or benchmark_returns_full.dropna().empty:

        st.info("Insufficient data to render chart for this window.")

        return



    cumulative: Dict[str, pd.Series] = {}

    for basket_id in basket_returns_full.columns:

        series = cumulative_return_since(basket_returns_full[basket_id], display_start)

        if not series.empty:

            cumulative[basket_id] = series



    if not cumulative:

        st.info("No basket return series available for this chart.")

        return



    cum_pct = pd.DataFrame(cumulative)

    bm_cum = cumulative_return_since(benchmark_returns_full, display_start)

    if bm_cum.empty:

        st.info("Insufficient benchmark history for this chart window.")

        return



    duplicate_names: Dict[str, int] = {}

    for basket_id in cum_pct.columns:

        meta = basket_metadata.get(basket_id, {})

        fallback = basket_id.split(BASKET_KEY_SEPARATOR, 1)[-1]

        basket_name = str(meta.get("Basket", fallback))

        duplicate_names[basket_name] = duplicate_names.get(basket_name, 0) + 1



    fig = go.Figure()



    for i, basket_id in enumerate(cum_pct.columns):

        trace_name = _chart_display_name(basket_id, basket_metadata, duplicate_names)

        fig.add_trace(go.Scatter(

            x=cum_pct.index,

            y=cum_pct[basket_id],

            mode="lines",

            line=dict(width=2, color=PASTEL[i % len(PASTEL)]),

            name=trace_name,

            hovertemplate=f"{trace_name}<br>% Cum: %{{y:.1f}}%<extra></extra>"

        ))



    fig.add_trace(go.Scatter(

        x=bm_cum.index,

        y=bm_cum.values,

        mode="lines",

        line=dict(width=2, dash="dash", color="#888"),

        name="SPY",

        hovertemplate="SPY<br>% Cum: %{y:.1f}%<extra></extra>"

    ))



    rangebreaks: List[Dict[str, Any]] = [dict(bounds=["sat", "mon"])]

    business_days = pd.date_range(bm_cum.index.min(), bm_cum.index.max(), freq="B")

    missing_sessions = business_days.difference(bm_cum.index)

    if len(missing_sessions):

        rangebreaks.append(dict(values=missing_sessions))



    fig.update_layout(

        showlegend=True,

        hovermode="x unified",

        yaxis_title="Cumulative return, %",

        title=dict(text=title, x=0, xanchor="left", y=0.95),

        margin=dict(l=10, r=10, t=35, b=10),

        xaxis=dict(

            showspikes=True,

            spikemode="across",

            spikesnap="cursor",

            showgrid=True,

            rangebreaks=rangebreaks,

        ),

        yaxis=dict(zeroline=False, showgrid=True)

    )



    st.plotly_chart(fig, use_container_width=True)




def render_basket_section(

    heading: str,

    basket_returns_full: pd.DataFrame,

    benchmark_returns_full: pd.Series,

    display_start: pd.Timestamp,

    dynamic_label: str,

    show_chart: bool,

    basket_metadata: Dict[str, Dict[str, Any]],

) -> pd.DataFrame:

    st.subheader(heading)



    panel_df = build_panel_df(

        basket_returns_full=basket_returns_full,

        display_start=display_start,

        dynamic_label=dynamic_label,

        basket_metadata=basket_metadata,

        benchmark_series_full=benchmark_returns_full,

    )



    plot_panel_table(panel_df, dynamic_label=dynamic_label)



    if show_chart:

        dynamic_col = f"%{dynamic_label}"

        ordered_cols = [

            basket_id for basket_id in panel_df.index

            if basket_id in basket_returns_full.columns

            and dynamic_col in panel_df.columns

            and pd.notna(panel_df.loc[basket_id, dynamic_col])

        ]

        chart_rets = basket_returns_full[ordered_cols] if ordered_cols else basket_returns_full



        plot_cumulative_chart(

            basket_returns_full=chart_rets,

            title=f"{heading} | Cumulative Performance vs SPY",

            benchmark_returns_full=benchmark_returns_full,

            display_start=display_start,

            basket_metadata=basket_metadata,

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



    ny_now = datetime.now(NY_TZ)

    today = ny_now.date()



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



if not selected_categories:

    st.warning("Select at least one category.")

    st.stop()



selected_definitions = {

    category: CATEGORIES[category]

    for category in selected_categories

}



display_start_for_fetch = compute_display_start(preset, today)

fetch_start_date = compute_fetch_start(display_start_for_fetch)

include_today = ny_now.weekday() >= 5 or ny_now.time() >= COMPLETED_SESSION_TIME

download_end_exclusive = today + timedelta(days=1) if include_today else today

DYNAMIC_LABEL = preset

min_market_cap = MIN_MARKET_CAP if apply_market_cap_filter else None



# ============================================================

# Fetch data

# ============================================================

raw_selected_baskets = flatten_baskets(selected_definitions)

equity_tickers = unique_tickers_from_baskets(raw_selected_baskets)

fx_tickers = required_fx_tickers(equity_tickers)

need = sorted(set(equity_tickers + fx_tickers + [BENCH]))



with st.spinner("Fetching basket price history..."):

    levels, fetch_meta = fetch_daily_levels(

        need,

        start=pd.to_datetime(fetch_start_date),

        end=pd.to_datetime(download_end_exclusive),

    )



if levels.empty:

    st.error("No data returned for the selected range.")

    st.stop()



if fetch_meta.get("source") == "last_good_cache":

    st.warning("Yahoo returned no usable data. Showing the last-good local cache.")



levels_usd, fx_issues = convert_foreign_levels_to_usd(levels)



if BENCH not in levels_usd.columns or levels_usd[BENCH].dropna().empty:

    st.error("SPY data missing or empty for the selected range.")

    st.stop()



reference_date = pd.Timestamp(levels_usd[BENCH].dropna().index.max())

display_start_date = compute_display_start(preset, reference_date.date())

display_start_ts = pd.Timestamp(display_start_date)



market_metadata: Dict[str, Dict[str, Any]] = {}

if apply_market_cap_filter:

    with st.spinner("Fetching current market-cap metadata..."):

        market_metadata = fetch_market_metadata(equity_tickers)

    market_metadata = convert_market_metadata_to_usd(market_metadata, levels)



live_categories, basket_metadata, dropped_baskets = build_live_baskets(

    levels=levels_usd,

    categories=selected_definitions,

    market_metadata=market_metadata,

    min_market_cap=min_market_cap,

    stale_days=stale_days,

    reference_date=reference_date,

)



live_all_baskets = flatten_baskets(live_categories)



if not live_all_baskets:

    st.error("No baskets passed the current filters.")

    st.stop()



live_tickers = unique_tickers_from_baskets(live_all_baskets, extra=[BENCH])

available_tickers = [ticker for ticker in live_tickers if ticker in levels_usd.columns]

benchmark_calendar = levels_usd[BENCH].dropna().index

aligned_levels = align_levels_to_calendar(

    levels_usd[available_tickers],

    calendar_index=benchmark_calendar,

)



all_basket_rets_full = ew_rets_from_levels(

    levels=aligned_levels,

    baskets=live_all_baskets,

)



if all_basket_rets_full.empty:

    st.error("No baskets passed the current filters.")

    st.stop()



bench_rets_full = aligned_levels[BENCH].dropna().pct_change(fill_method=None).dropna()



# ============================================================

# Consolidated panel and optional sections

# ============================================================

all_panel_df = render_basket_section(

    heading="All Baskets | Consolidated Panel",

    basket_returns_full=all_basket_rets_full,

    benchmark_returns_full=bench_rets_full,

    display_start=display_start_ts,

    dynamic_label=DYNAMIC_LABEL,

    show_chart=show_all_chart,

    basket_metadata=basket_metadata,

)



if show_category_sections:

    for category, baskets in live_categories.items():

        cat_names = [

            basket_key(category, basket_name)

            for basket_name in baskets

            if basket_key(category, basket_name) in all_basket_rets_full.columns

        ]



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

            benchmark_returns_full=bench_rets_full,

            display_start=display_start_ts,

            dynamic_label=DYNAMIC_LABEL,

            show_chart=True,

            basket_metadata=basket_metadata,

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



render_footer()
