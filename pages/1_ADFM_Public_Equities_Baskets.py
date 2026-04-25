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
st.set_page_config(page_title="Sector and Thematic Basket Panels", layout="wide")

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

TITLE = "ADFM Sector and Thematic Basket Panels"
SUBTITLE = "Sector, thematic, and country baskets only. No pure regime diagnostics or duplicate factor baskets."

PASTEL_GREEN = "#52b788"
PASTEL_RED = "#e85d5d"
PASTEL_GREY = "#8b949e"

PASTEL = [
    "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
    "#b279a2", "#ff9da6", "#9d755d", "#bab0ac", "#59a14f",
    "#edc948", "#af7aa1", "#ff9da7", "#76b7b2", "#8cd17d",
    "#b6992d", "#499894", "#d37295", "#fabfd2", "#79706e",
]

MIN_MARKET_CAP = 1_000_000_000
INDICATOR_WARMUP_DAYS = 520
BENCH = "SPY"

CACHE_DIR = Path(".adfm_cache")
LEVELS_CACHE = CACHE_DIR / "sector_thematic_basket_levels_last_good.pkl"
LEVELS_META_CACHE = CACHE_DIR / "sector_thematic_basket_levels_last_good_meta.json"

MARKET_CAP_EXCEPTIONS: set[str] = set()

# If Yahoo does not return a market cap, the ticker is kept.
# ETFs, ADRs, and foreign listings often have missing market-cap fields.
EXCLUDE_MISSING_MARKET_CAP = False


# ============================================================
# Category -> Baskets -> Tickers
# 250 baskets total:
# Technology/AI/Internet: 36
# Energy/Power/Infrastructure/Materials: 36
# Industrials/Defense/Transport: 27
# Healthcare: 25
# Financials/Real Estate: 28
# Consumer/Housing/Travel: 28
# Thematic Cross-Sector: 35
# Countries/Regions: 35
# ============================================================
CATEGORIES: Dict[str, Dict[str, List[str]]] = {'Technology, AI and Internet': {'AI Compute and Accelerators': ['NVDA', 'AMD', 'AVGO', 'MRVL', 'ARM', 'INTC'],
                                 'AI ASICs and Custom Silicon': ['AVGO', 'MRVL', 'AMD', 'TSM', 'ARM', 'SNPS', 'CDNS'],
                                 'Semiconductor Broad ETFs': ['SMH', 'SOXX', 'XSD'],
                                 'Analog Power and Mixed Signal': ['TXN',
                                                                   'ADI',
                                                                   'MCHP',
                                                                   'NXPI',
                                                                   'MPWR',
                                                                   'ON',
                                                                   'STM',
                                                                   'IFNNY'],
                                 'Memory and Storage': ['MU', 'WDC', 'STX', 'SNDK', 'HXSCF', 'SSNLF'],
                                 'HBM and Advanced Memory': ['MU',
                                                             'HXSCF',
                                                             'SSNLF',
                                                             'NVDA',
                                                             'AMD',
                                                             'TSM',
                                                             'ASML',
                                                             'AMAT',
                                                             'LRCX'],
                                 'Foundry and OSAT': ['TSM', 'UMC', 'GFS', 'ASX', 'AMKR', 'BESIY'],
                                 'Advanced Packaging': ['AMKR', 'ASX', 'TSM', 'BESIY', 'AMAT', 'LRCX', 'KLAC', 'ONTO'],
                                 'Semiconductor Equipment': ['ASML',
                                                             'AMAT',
                                                             'LRCX',
                                                             'KLAC',
                                                             'TER',
                                                             'ONTO',
                                                             'AEIS',
                                                             'ACMR',
                                                             'COHU'],
                                 'Semicap Subsystems and Components': ['MKSI',
                                                                       'ENTG',
                                                                       'AEIS',
                                                                       'UCTT',
                                                                       'ICHR',
                                                                       'COHU',
                                                                       'VECO',
                                                                       'CAMT'],
                                 'EDA and Chip IP': ['SNPS', 'CDNS', 'ARM'],
                                 'RF and Wireless Connectivity': ['QCOM',
                                                                  'SWKS',
                                                                  'QRVO',
                                                                  'MTSI',
                                                                  'AVNW',
                                                                  'CIEN',
                                                                  'LITE'],
                                 'Optical Networking and Interconnect': ['CIEN', 'LITE', 'COHR', 'AAOI', 'INFN', 'NOK'],
                                 'Data Center Networking': ['ANET', 'CSCO', 'JNPR', 'AVGO', 'MRVL', 'CIEN', 'LITE'],
                                 'Servers and AI Hardware': ['SMCI', 'DELL', 'HPE', 'NTAP', 'WDC', 'STX', 'IBM'],
                                 'Enterprise Storage and Data Infrastructure': ['NTAP',
                                                                                'WDC',
                                                                                'STX',
                                                                                'IBM',
                                                                                'DDOG',
                                                                                'SNOW',
                                                                                'MDB'],
                                 'Hyperscalers': ['MSFT', 'AMZN', 'GOOGL', 'META', 'ORCL'],
                                 'AI Cloud Challengers and Neocloud': ['ORCL',
                                                                       'IBM',
                                                                       'HPE',
                                                                       'SMCI',
                                                                       'DELL',
                                                                       'NBIS',
                                                                       'IREN',
                                                                       'CORZ',
                                                                       'APLD'],
                                 'Enterprise SaaS': ['NOW', 'CRM', 'ADBE', 'INTU', 'TEAM', 'HUBS', 'WDAY', 'DOCU'],
                                 'Vertical Software': ['VEEV', 'TYL', 'APPF', 'MNDY', 'PAYC', 'GWRE', 'DAY', 'APP'],
                                 'Data Analytics and AI Software': ['PLTR',
                                                                    'SNOW',
                                                                    'MDB',
                                                                    'DDOG',
                                                                    'ESTC',
                                                                    'CFLT',
                                                                    'S',
                                                                    'AI'],
                                 'Database and Data Platforms': ['SNOW', 'MDB', 'ESTC', 'CFLT', 'DDOG', 'ORCL', 'IBM'],
                                 'Observability and DevOps': ['DDOG', 'DT', 'GTLB', 'NET', 'ESTC', 'MDB', 'TEAM'],
                                 'Developer Productivity and Code AI': ['MSFT', 'GTLB', 'TEAM', 'DDOG', 'MDB', 'NOW'],
                                 'Cybersecurity Platforms': ['PANW',
                                                             'CRWD',
                                                             'FTNT',
                                                             'ZS',
                                                             'OKTA',
                                                             'CYBR',
                                                             'CHKP',
                                                             'NET',
                                                             'TENB',
                                                             'S'],
                                 'Identity and Access Management': ['OKTA', 'CYBR', 'MSFT', 'PANW', 'FTNT', 'CHKP'],
                                 'AI Security and Model Governance': ['CRWD',
                                                                      'PANW',
                                                                      'ZS',
                                                                      'DDOG',
                                                                      'PLTR',
                                                                      'SNOW',
                                                                      'NET',
                                                                      'S'],
                                 'Digital Advertising Platforms': ['GOOGL', 'META', 'TTD', 'PINS', 'SNAP', 'APP'],
                                 'Retail Media Networks': ['AMZN', 'WMT', 'CART', 'GOOGL', 'META', 'TTD'],
                                 'Consumer Internet': ['META', 'SNAP', 'PINS', 'MTCH', 'GOOGL', 'BIDU', 'BILI', 'RBLX'],
                                 'E-Commerce Marketplaces': ['AMZN',
                                                             'SHOP',
                                                             'MELI',
                                                             'ETSY',
                                                             'PDD',
                                                             'BABA',
                                                             'JD',
                                                             'SE',
                                                             'EBAY'],
                                 'Streaming and Audio': ['NFLX', 'DIS', 'WBD', 'PARA', 'ROKU', 'SPOT', 'LYV', 'CMCSA'],
                                 'Digital Payments and Networks': ['V',
                                                                   'MA',
                                                                   'AXP',
                                                                   'PYPL',
                                                                   'XYZ',
                                                                   'FI',
                                                                   'FIS',
                                                                   'GPN',
                                                                   'MELI'],
                                 'Fintech and Brokerage Platforms': ['HOOD',
                                                                     'SOFI',
                                                                     'AFRM',
                                                                     'UPST',
                                                                     'XYZ',
                                                                     'PYPL',
                                                                     'NU'],
                                 'IT Services and AI Implementation': ['ACN',
                                                                       'IBM',
                                                                       'CTSH',
                                                                       'EPAM',
                                                                       'GIB',
                                                                       'INFY',
                                                                       'WIT'],
                                 'Robotics and Embodied AI': ['ISRG',
                                                              'SYM',
                                                              'ROK',
                                                              'TER',
                                                              'ABB',
                                                              'FANUY',
                                                              'TSLA',
                                                              'CGNX']},
 'Energy, Power, Infrastructure and Materials': {'Integrated Energy Majors': ['XOM',
                                                                              'CVX',
                                                                              'COP',
                                                                              'SHEL',
                                                                              'BP',
                                                                              'TTE',
                                                                              'EQNR',
                                                                              'PBR'],
                                                 'Oil-Weighted E&Ps': ['EOG',
                                                                       'FANG',
                                                                       'OXY',
                                                                       'DVN',
                                                                       'COP',
                                                                       'PR',
                                                                       'MTDR',
                                                                       'APA',
                                                                       'MRO'],
                                                 'Gas-Weighted E&Ps': ['EQT', 'AR', 'RRC', 'CTRA', 'EXE', 'CRK', 'CNX'],
                                                 'LNG Export and Gas Infrastructure': ['LNG',
                                                                                       'KMI',
                                                                                       'WMB',
                                                                                       'ET',
                                                                                       'EPD',
                                                                                       'WDS',
                                                                                       'NEXT'],
                                                 'Midstream Pipelines': ['KMI',
                                                                         'WMB',
                                                                         'EPD',
                                                                         'ET',
                                                                         'ENB',
                                                                         'MPLX',
                                                                         'OKE',
                                                                         'TRP'],
                                                 'Oilfield Services': ['SLB',
                                                                       'HAL',
                                                                       'BKR',
                                                                       'NOV',
                                                                       'FTI',
                                                                       'PTEN',
                                                                       'HP',
                                                                       'NBR',
                                                                       'OII'],
                                                 'Refiners and Downstream': ['MPC', 'VLO', 'PSX', 'DK', 'PBF', 'SUN'],
                                                 'Uranium Miners and Fuel Cycle': ['CCJ',
                                                                                   'UUUU',
                                                                                   'UEC',
                                                                                   'URG',
                                                                                   'UROY',
                                                                                   'DNN',
                                                                                   'NXE',
                                                                                   'LEU',
                                                                                   'URA',
                                                                                   'URNM'],
                                                 'Nuclear Power and Services': ['VST',
                                                                                'CEG',
                                                                                'BWXT',
                                                                                'LEU',
                                                                                'SMR',
                                                                                'OKLO',
                                                                                'CCJ'],
                                                 'Nuclear SMR Developers': ['SMR', 'OKLO', 'BWXT', 'FLR', 'GEV'],
                                                 'Merchant Power Producers': ['VST',
                                                                              'CEG',
                                                                              'NRG',
                                                                              'TLN',
                                                                              'AES',
                                                                              'CWEN'],
                                                 'Regulated Electric Utilities': ['NEE',
                                                                                  'DUK',
                                                                                  'SO',
                                                                                  'AEP',
                                                                                  'XEL',
                                                                                  'EXC',
                                                                                  'PCG',
                                                                                  'EIX',
                                                                                  'ED'],
                                                 'Grid Equipment and Transformers': ['ETN',
                                                                                     'GEV',
                                                                                     'ABB',
                                                                                     'HUBB',
                                                                                     'POWL',
                                                                                     'MYRG',
                                                                                     'PWR',
                                                                                     'RRX'],
                                                 'Transformer Bottleneck': ['ETN', 'GEV', 'HUBB', 'POWL', 'ABB', 'RRX'],
                                                 'Switchgear and Electrical Distribution': ['ETN',
                                                                                            'ABB',
                                                                                            'GEV',
                                                                                            'HUBB',
                                                                                            'POWL',
                                                                                            'GNRC'],
                                                 'Data Center Power and Thermal': ['VRT',
                                                                                   'ETN',
                                                                                   'TT',
                                                                                   'CARR',
                                                                                   'JCI',
                                                                                   'POWL',
                                                                                   'GEV',
                                                                                   'HUBB'],
                                                 'Liquid Cooling and Thermal Management': ['VRT',
                                                                                           'TT',
                                                                                           'CARR',
                                                                                           'JCI',
                                                                                           'MOD',
                                                                                           'AOS',
                                                                                           'ETN'],
                                                 'Backup Power and Generators': ['GNRC',
                                                                                 'CAT',
                                                                                 'CMI',
                                                                                 'GEV',
                                                                                 'ETN',
                                                                                 'VRT'],
                                                 'Gas Turbines and Peakers': ['GEV', 'VST', 'NRG', 'TLN', 'CEG', 'CMI'],
                                                 'Solar and Inverters': ['TAN',
                                                                         'FSLR',
                                                                         'ENPH',
                                                                         'SEDG',
                                                                         'RUN',
                                                                         'CSIQ',
                                                                         'JKS',
                                                                         'ARRY'],
                                                 'Wind and Renewables': ['FAN',
                                                                         'ICLN',
                                                                         'NEE',
                                                                         'BEP',
                                                                         'AY',
                                                                         'GEV',
                                                                         'VWSYF'],
                                                 'Hydrogen and Fuel Cells': ['BE',
                                                                             'BLDP',
                                                                             'FCEL',
                                                                             'PLUG',
                                                                             'LIN',
                                                                             'APD'],
                                                 'Battery Storage and BESS': ['FLNC',
                                                                              'STEM',
                                                                              'TSLA',
                                                                              'ENPH',
                                                                              'SEDG',
                                                                              'NXT',
                                                                              'NEE',
                                                                              'AES'],
                                                 'Copper Miners Pure Play': ['FCX',
                                                                             'SCCO',
                                                                             'TECK',
                                                                             'ERO',
                                                                             'IVPAF',
                                                                             'COPX'],
                                                 'Diversified Metals and Mining': ['BHP',
                                                                                   'RIO',
                                                                                   'VALE',
                                                                                   'GLNCY',
                                                                                   'TECK',
                                                                                   'AA',
                                                                                   'XME'],
                                                 'Lithium Miners': ['ALB', 'SQM', 'LAC', 'LTHM', 'LIT', 'PLL'],
                                                 'Battery Metals Broad': ['ALB',
                                                                          'SQM',
                                                                          'LAC',
                                                                          'MP',
                                                                          'VALE',
                                                                          'BHP',
                                                                          'LIT'],
                                                 'Rare Earths and Magnets': ['MP', 'LYSDY', 'REMX', 'UUUU', 'NEM'],
                                                 'Gold and Silver Miners': ['GDX',
                                                                            'GDXJ',
                                                                            'NEM',
                                                                            'AEM',
                                                                            'GOLD',
                                                                            'KGC',
                                                                            'AG',
                                                                            'PAAS',
                                                                            'WPM'],
                                                 'Steel and Aluminum': ['NUE', 'STLD', 'X', 'CLF', 'AA', 'CENX'],
                                                 'Fertilizers': ['CF', 'MOS', 'NTR', 'IPI'],
                                                 'Industrial Gases': ['LIN', 'APD', 'AIQUY'],
                                                 'Commodity Chemicals': ['DOW', 'LYB', 'CE', 'WLK', 'OLN'],
                                                 'Specialty Chemicals': ['SHW', 'PPG', 'EMN', 'RPM', 'IFF', 'ALB'],
                                                 'Aggregates and Construction Materials': ['VMC',
                                                                                           'MLM',
                                                                                           'SUM',
                                                                                           'EXP',
                                                                                           'CRH',
                                                                                           'CX'],
                                                 'Water Infrastructure and Treatment': ['XYL',
                                                                                        'WTS',
                                                                                        'AWK',
                                                                                        'ECL',
                                                                                        'DHR',
                                                                                        'PNR',
                                                                                        'CWCO']},
 'Industrials, Defense and Transport': {'Aerospace and Defense Primes': ['LMT',
                                                                         'NOC',
                                                                         'RTX',
                                                                         'GD',
                                                                         'HII',
                                                                         'BAESY',
                                                                         'RNMBY'],
                                        'Defense Electronics and ISR': ['LHX',
                                                                        'LDOS',
                                                                        'BAH',
                                                                        'CACI',
                                                                        'KTOS',
                                                                        'AVAV',
                                                                        'PLTR'],
                                        'Missiles and Munitions': ['LMT', 'RTX', 'NOC', 'GD', 'LHX', 'BAESY', 'RNMBY'],
                                        'Drones and Autonomous Defense': ['AVAV', 'KTOS', 'LMT', 'NOC', 'PLTR', 'TXT'],
                                        'Naval Shipbuilding and Undersea': ['HII', 'GD', 'LMT', 'NOC', 'RTX'],
                                        'Air Defense and Radar': ['RTX', 'LMT', 'NOC', 'LHX', 'BAESY', 'ESLT'],
                                        'Defense Cybersecurity': ['PLTR', 'CRWD', 'PANW', 'ZS', 'LDOS', 'BAH', 'CACI'],
                                        'Government IT and Mission Services': ['SAIC',
                                                                               'CACI',
                                                                               'LDOS',
                                                                               'BAH',
                                                                               'PSN',
                                                                               'G'],
                                        'Space and Satellite': ['RKLB',
                                                                'IRDM',
                                                                'ASTS',
                                                                'LHX',
                                                                'LMT',
                                                                'NOC',
                                                                'VSAT',
                                                                'GSAT'],
                                        'Satellite Communications': ['IRDM', 'VSAT', 'GSAT', 'ASTS', 'LHX', 'NOC'],
                                        'Aerospace Aftermarket': ['GE', 'RTX', 'HEI', 'TDG', 'FTAI', 'HWM'],
                                        'Commercial Aerospace OEM and Suppliers': ['BA',
                                                                                   'GE',
                                                                                   'HWM',
                                                                                   'SPR',
                                                                                   'TDG',
                                                                                   'HEI',
                                                                                   'TXT'],
                                        'Industrial Automation': ['ROK',
                                                                  'ETN',
                                                                  'EMR',
                                                                  'AME',
                                                                  'PH',
                                                                  'ABB',
                                                                  'KEYS',
                                                                  'TRMB',
                                                                  'CGNX',
                                                                  'SYM'],
                                        'Factory Automation': ['ROK', 'ABB', 'FANUY', 'CGNX', 'TER', 'KEYS', 'AME'],
                                        'Warehouse Automation': ['SYM', 'ZBRA', 'ROK', 'TER', 'CGNX', 'AME'],
                                        'Sensors and Measurement': ['KEYS', 'TRMB', 'AME', 'TDY', 'FTV', 'ROK'],
                                        'Test and Measurement': ['KEYS', 'TER', 'FTV', 'TDY', 'A', 'COHU'],
                                        'Industrial Software and PLM': ['ADSK', 'PTC', 'DASTY', 'SNPS', 'CDNS', 'TRMB'],
                                        'Machinery and Heavy Equipment': ['CAT', 'DE', 'CNHI', 'AGCO', 'PCAR', 'CMI'],
                                        'Rental Equipment and Tools': ['URI', 'HRI', 'FTAI', 'GWW', 'FAST'],
                                        'Industrial Distribution': ['GWW', 'FAST', 'AIT', 'WCC', 'SITE'],
                                        'Waste and Environmental Services': ['WM', 'RSG', 'WCN', 'CLH', 'SRCL'],
                                        'Engineering and Consulting': ['J', 'ACM', 'FLR', 'PWR', 'MTZ', 'FIX'],
                                        'Railroads': ['UNP', 'CSX', 'NSC', 'CNI', 'CP'],
                                        'Parcel and Logistics': ['FDX', 'UPS', 'GXO', 'XPO', 'CHRW'],
                                        'Trucking and LTL': ['ODFL', 'SAIA', 'JBHT', 'KNX', 'ARCB', 'XPO'],
                                        'Air Cargo and Aircraft Leasing': ['AL', 'FTAI', 'AER', 'ATSG', 'CPA']},
 'Healthcare': {'Large-Cap Pharma': ['LLY', 'JNJ', 'MRK', 'PFE', 'BMY', 'ABBV', 'AZN', 'NVO', 'NVS', 'GSK'],
                'Large-Cap Biotech': ['AMGN', 'GILD', 'REGN', 'BIIB', 'VRTX', 'ALNY'],
                'GLP-1 and Metabolic': ['LLY', 'NVO', 'AZN', 'MRK', 'PFE', 'VKTX', 'AMGN'],
                'Obesity Drug Ecosystem': ['LLY', 'NVO', 'VKTX', 'AMGN', 'MRK', 'PFE', 'TMO', 'DHR'],
                'Oncology and Immunology': ['MRK', 'BMY', 'RHHBY', 'REGN', 'VRTX', 'INCY', 'EXEL'],
                'Rare Disease and Specialty Pharma': ['VRTX', 'ALNY', 'BMRN', 'RARE', 'IONS', 'HALO'],
                'MedTech Devices': ['MDT', 'SYK', 'ISRG', 'BSX', 'ZBH', 'EW', 'PEN', 'ABT'],
                'Surgical Robotics and Advanced Devices': ['ISRG', 'SYK', 'MDT', 'BSX', 'DXCM', 'TMDX'],
                'Orthopedics and Spine': ['SYK', 'ZBH', 'GMED', 'MDT', 'SNN', 'OFIX'],
                'Cardiovascular Devices': ['BSX', 'MDT', 'ABT', 'EW', 'PEN', 'TMDX'],
                'Diabetes Devices': ['DXCM', 'PODD', 'ABT', 'MDT', 'TNDM'],
                'Diagnostics and Life Science Tools': ['TMO', 'DHR', 'A', 'RGEN', 'ILMN', 'WAT', 'BRKR'],
                'CRO and Clinical Services': ['IQV', 'LH', 'DGX', 'MEDP', 'ICLR', 'CRL'],
                'Healthcare Payers and Managed Care': ['UNH', 'HUM', 'CI', 'ELV', 'CNC', 'MOH', 'OSCR'],
                'Medicare Advantage Risk': ['HUM', 'UNH', 'ELV', 'CI', 'CNC', 'MOH'],
                'Medicaid Managed Care': ['CNC', 'MOH', 'ELV', 'UNH'],
                'Drug Distributors and Healthcare Supply Chain': ['MCK', 'COR', 'CAH', 'CVS', 'CI', 'HSIC', 'OMI'],
                'Specialty Pharmacy and PBM': ['CVS', 'CI', 'UNH', 'ELV', 'MCK', 'COR'],
                'Hospitals and Providers': ['HCA', 'THC', 'UHS', 'CYH', 'EHC'],
                'Hospital Utilization Winners': ['HCA', 'THC', 'UHS', 'SYK', 'BSX', 'ISRG', 'EW'],
                'Dental Vision and Elective Care': ['ALGN', 'HSIC', 'EYE', 'WRBY', 'XRAY'],
                'Senior Housing and Aging Care': ['WELL', 'VTR', 'OHI', 'SBRA', 'HCA', 'EHC'],
                'Healthcare AI and Automation': ['MCK', 'COR', 'UNH', 'CI', 'HCA', 'TMO', 'DHR', 'ISRG', 'PLTR'],
                'Healthcare Staffing': ['AMN', 'RHI', 'ASGN', 'KFY'],
                'Animal Health': ['ZTS', 'ELAN', 'IDXX', 'PETS', 'HSIC']},
 'Financials and Real Estate': {'Money Center Banks': ['JPM', 'BAC', 'C', 'WFC'],
                                'Investment Banks': ['GS', 'MS', 'RJF', 'LAZ', 'PJT', 'EVR'],
                                'Regional Banks': ['KRE',
                                                   'TFC',
                                                   'FITB',
                                                   'CFG',
                                                   'RF',
                                                   'KEY',
                                                   'PNC',
                                                   'USB',
                                                   'MTB',
                                                   'WAL',
                                                   'ZION'],
                                'Asset Managers and Alts': ['BX',
                                                            'KKR',
                                                            'APO',
                                                            'CG',
                                                            'ARES',
                                                            'OWL',
                                                            'TPG',
                                                            'BLK',
                                                            'TROW'],
                                'Private Credit and BDCs': ['ARES', 'ARCC', 'MAIN', 'BXSL', 'OBDC', 'FSK'],
                                'Exchanges and Market Data': ['CME', 'ICE', 'NDAQ', 'CBOE', 'MKTX', 'SPGI', 'MSCI'],
                                'Derivatives Exchanges': ['CME', 'ICE', 'CBOE', 'NDAQ'],
                                'Market Data and Index Providers': ['SPGI', 'MSCI', 'NDAQ', 'ICE', 'MCO'],
                                'Brokers and Trading Platforms': ['IBKR', 'SCHW', 'HOOD', 'RJF', 'MS', 'GS'],
                                'Electronic Trading and Market Makers': ['VIRT', 'TW', 'IBKR', 'CBOE', 'NDAQ'],
                                'Card Networks and Consumer Lenders': ['V',
                                                                       'MA',
                                                                       'AXP',
                                                                       'COF',
                                                                       'DFS',
                                                                       'SYF',
                                                                       'ALLY',
                                                                       'OMF',
                                                                       'ENVA'],
                                'Payments Processors Legacy': ['FI', 'FIS', 'GPN', 'PYPL', 'XYZ'],
                                'Cross-Border Payments': ['V', 'MA', 'PYPL', 'WU', 'EEFT', 'GPN'],
                                'Stablecoin and Tokenization Proxies': ['COIN', 'HOOD', 'PYPL', 'XYZ', 'MSTR', 'IBKR'],
                                'Crypto Exchanges and Custody': ['COIN', 'HOOD', 'MSTR', 'IBKR'],
                                'Bitcoin Miners': ['MARA', 'RIOT', 'CLSK', 'IREN', 'CORZ', 'CIFR', 'BITF'],
                                'P&C Insurance': ['PGR', 'TRV', 'CB', 'ALL', 'CINF', 'WRB', 'HIG'],
                                'Life and Retirement Insurance': ['MET', 'PRU', 'LNC', 'AIG', 'EQH', 'RGA'],
                                'Insurance Brokers': ['AJG', 'BRO', 'MMC', 'AON', 'WTW'],
                                'Insurance Software and Data': ['BR', 'GWRE', 'VRSK', 'MSCI', 'SPGI'],
                                'Mortgage Finance and Title': ['RKT', 'UWMC', 'COOP', 'FNF', 'FAF', 'NMIH', 'ESNT'],
                                'Data Center REITs': ['EQIX', 'DLR'],
                                'Industrial REITs': ['PLD', 'REXR', 'EGP', 'STAG', 'TRNO'],
                                'Residential REITs': ['AVB', 'EQR', 'UDR', 'ESS', 'MAA', 'INVH', 'AMH'],
                                'Self-Storage REITs': ['PSA', 'EXR', 'CUBE', 'NSA'],
                                'Tower REITs': ['AMT', 'SBAC', 'CCI'],
                                'Office REITs': ['BXP', 'VNO', 'SLG', 'KRC', 'DEI'],
                                'Senior Housing REITs': ['WELL', 'VTR', 'OHI', 'SBRA']},
 'Consumer, Housing and Travel': {'Grocery and Clubs': ['WMT', 'COST', 'KR', 'BJ', 'ACI'],
                                  'Household and Personal Care': ['PG', 'CL', 'KMB', 'CHD', 'EL', 'KVUE'],
                                  'Beverages and Tobacco': ['KO', 'PEP', 'KDP', 'MNST', 'PM', 'MO', 'BTI'],
                                  'Food Producers and Packaged Food': ['GIS',
                                                                       'K',
                                                                       'CPB',
                                                                       'CAG',
                                                                       'HSY',
                                                                       'MDLZ',
                                                                       'SJM',
                                                                       'KHC'],
                                  'Discount and Dollar Stores': ['DG', 'DLTR', 'WMT', 'BURL', 'FIVE', 'OLLI'],
                                  'Off-Price Retail': ['TJX', 'ROST', 'BURL'],
                                  'Home Improvement': ['HD', 'LOW', 'TSCO', 'POOL', 'BLDR'],
                                  'Apparel and Footwear': ['NKE', 'LULU', 'DECK', 'ONON', 'UAA', 'RL', 'TPR'],
                                  'Luxury Goods': ['LVMUY', 'TPR', 'RL', 'CPRI', 'CFRUY', 'PPRUY'],
                                  'Beauty and Cosmetics': ['EL', 'ULTA', 'COTY', 'LRLCY', 'ELF'],
                                  'Restaurants and QSR': ['MCD', 'YUM', 'QSR', 'WEN', 'DPZ', 'CMG'],
                                  'Casual Dining': ['DRI', 'TXRH', 'EAT', 'BLMN', 'CAKE', 'WING'],
                                  'Travel Booking': ['BKNG', 'EXPE', 'ABNB', 'TRIP'],
                                  'Hotels and Casinos': ['MAR',
                                                         'HLT',
                                                         'IHG',
                                                         'MGM',
                                                         'LVS',
                                                         'WYNN',
                                                         'MLCO',
                                                         'CZR',
                                                         'PENN'],
                                  'Cruise Lines': ['RCL', 'CCL', 'NCLH'],
                                  'Airlines': ['DAL', 'UAL', 'AAL', 'LUV', 'ALK', 'JBLU'],
                                  'Sports Betting and iGaming': ['DKNG', 'FLUT', 'MGM', 'PENN', 'CZR', 'RSI'],
                                  'Live Sports and Venue Economics': ['LYV', 'TKO', 'MSGS', 'SPHR', 'BATRA', 'FWONA'],
                                  'Gaming Publishers': ['EA', 'TTWO', 'RBLX', 'NTDOY', 'SONY', 'MSFT'],
                                  'Autos Legacy OEMs': ['TM', 'HMC', 'F', 'GM', 'STLA', 'VWAGY'],
                                  'Electric Vehicles': ['TSLA', 'RIVN', 'LCID', 'NIO', 'LI', 'XPEV', 'RACE'],
                                  'Auto Parts and Repair': ['AZO', 'ORLY', 'AAP', 'LKQ', 'GPC'],
                                  'Used Auto and Affordability': ['KMX', 'CVNA', 'LAD', 'AN', 'PAG'],
                                  'Auto Finance and Subprime Credit': ['ALLY', 'COF', 'DFS', 'SYF', 'OMF', 'ENVA'],
                                  'Homebuilders': ['ITB', 'DHI', 'LEN', 'NVR', 'PHM', 'TOL', 'KBH', 'MTH'],
                                  'Building Products': ['BLDR', 'TREX', 'MAS', 'OC', 'JELD', 'FBIN'],
                                  'Single-Family Rental': ['INVH', 'AMH'],
                                  'Apartment Rent Pressure': ['AVB', 'EQR', 'UDR', 'ESS', 'MAA', 'CPT']},
 'Thematic Cross-Sector Baskets': {'AI Data Center Capex': ['NVDA',
                                                            'AMD',
                                                            'AVGO',
                                                            'MRVL',
                                                            'ANET',
                                                            'VRT',
                                                            'ETN',
                                                            'GEV',
                                                            'SMCI',
                                                            'DELL',
                                                            'TSM',
                                                            'ASML'],
                                   'AI Power Demand': ['VST', 'CEG', 'NRG', 'TLN', 'ETN', 'VRT', 'GEV', 'PWR', 'NEE'],
                                   'AI Margin Expansion Beneficiaries': ['MCK',
                                                                         'COR',
                                                                         'CAH',
                                                                         'UNH',
                                                                         'CI',
                                                                         'ACN',
                                                                         'IBM',
                                                                         'FIS',
                                                                         'FI',
                                                                         'ADP',
                                                                         'PAYX'],
                                   'AI Application Layer': ['MSFT',
                                                            'NOW',
                                                            'CRM',
                                                            'ADBE',
                                                            'INTU',
                                                            'PLTR',
                                                            'SNOW',
                                                            'MDB',
                                                            'DDOG'],
                                   'AI Hardware Supply Chain': ['NVDA',
                                                                'AMD',
                                                                'AVGO',
                                                                'TSM',
                                                                'ASML',
                                                                'AMAT',
                                                                'LRCX',
                                                                'KLAC',
                                                                'MU',
                                                                'ANET',
                                                                'VRT',
                                                                'DELL'],
                                   'Sovereign AI Infrastructure': ['NVDA',
                                                                   'AMD',
                                                                   'AVGO',
                                                                   'TSM',
                                                                   'ASML',
                                                                   'ANET',
                                                                   'VRT',
                                                                   'ETN',
                                                                   'ORCL',
                                                                   'HPE'],
                                   'Data Center Construction and EPC': ['PWR',
                                                                        'MYRG',
                                                                        'EME',
                                                                        'FIX',
                                                                        'J',
                                                                        'ACM',
                                                                        'VRT',
                                                                        'ETN',
                                                                        'GEV'],
                                   'Grid Bottleneck': ['ETN', 'GEV', 'HUBB', 'POWL', 'PWR', 'MYRG', 'ABB', 'VRT'],
                                   'Electrification': ['ETN', 'GEV', 'ABB', 'HUBB', 'POWL', 'VRT', 'ON', 'MPWR'],
                                   'Reindustrialization': ['CAT',
                                                           'DE',
                                                           'ETN',
                                                           'PWR',
                                                           'URI',
                                                           'VMC',
                                                           'MLM',
                                                           'NUE',
                                                           'STLD',
                                                           'GEV'],
                                   'Reshoring and Factory Buildout': ['PWR',
                                                                      'MTZ',
                                                                      'EME',
                                                                      'FIX',
                                                                      'URI',
                                                                      'CAT',
                                                                      'VMC',
                                                                      'MLM',
                                                                      'ROK',
                                                                      'ABB'],
                                   'North American Onshoring Materials': ['VMC',
                                                                          'MLM',
                                                                          'SUM',
                                                                          'NUE',
                                                                          'STLD',
                                                                          'X',
                                                                          'CLF',
                                                                          'FCX',
                                                                          'EXP'],
                                   'Nearshoring Mexico': ['EWW', 'KOF', 'FMX', 'CX', 'AMX', 'PAC', 'OMAB', 'ASR'],
                                   'Defense Modernization': ['LMT',
                                                             'NOC',
                                                             'RTX',
                                                             'GD',
                                                             'LHX',
                                                             'PLTR',
                                                             'KTOS',
                                                             'AVAV',
                                                             'LDOS',
                                                             'BAH'],
                                   'NATO Re-Armament': ['LMT',
                                                        'NOC',
                                                        'RTX',
                                                        'GD',
                                                        'BAESY',
                                                        'RNMBY',
                                                        'SAABY',
                                                        'THLLY',
                                                        'ESLT'],
                                   'Space Economy': ['RKLB', 'IRDM', 'ASTS', 'LHX', 'NOC', 'LMT', 'VSAT', 'GSAT'],
                                   'GLP-1 Winners': ['LLY', 'NVO', 'VKTX', 'AMGN', 'TMO', 'DHR', 'MCK', 'COR'],
                                   'GLP-1 Consumer Losers': ['HSY', 'MDLZ', 'PEP', 'KO', 'MCD', 'YUM', 'DPZ', 'KDP'],
                                   'Longevity and Aging Population': ['LLY',
                                                                      'NVO',
                                                                      'ISRG',
                                                                      'SYK',
                                                                      'MDT',
                                                                      'MCK',
                                                                      'COR',
                                                                      'WELL',
                                                                      'VTR',
                                                                      'HCA'],
                                   'Healthcare Supply Chain Automation': ['MCK',
                                                                          'COR',
                                                                          'CAH',
                                                                          'UNH',
                                                                          'CI',
                                                                          'ACN',
                                                                          'CTSH',
                                                                          'PLTR'],
                                   'Crypto and Tokenization Proxies': ['COIN',
                                                                       'MSTR',
                                                                       'HOOD',
                                                                       'MARA',
                                                                       'RIOT',
                                                                       'CLSK',
                                                                       'IBIT',
                                                                       'ETHA'],
                                   'Speculative Liquidity and Retail Beta': ['CVNA',
                                                                             'UPST',
                                                                             'RIVN',
                                                                             'COIN',
                                                                             'MSTR',
                                                                             'HOOD',
                                                                             'SOFI',
                                                                             'MARA',
                                                                             'RIOT',
                                                                             'PLTR'],
                                   'Sports Live Events and Experiences': ['LYV',
                                                                          'TKO',
                                                                          'MSGS',
                                                                          'SPHR',
                                                                          'RCL',
                                                                          'CCL',
                                                                          'BKNG',
                                                                          'ABNB'],
                                   'Premium Consumer': ['LULU', 'NKE', 'ONON', 'RACE', 'LVMUY', 'COST', 'CMG', 'MAR'],
                                   'Trade-Down Consumer': ['WMT', 'COST', 'TJX', 'ROST', 'BURL', 'DG', 'DLTR', 'OLLI'],
                                   'Housing Affordability Stress': ['KMX',
                                                                    'CVNA',
                                                                    'RKT',
                                                                    'UWMC',
                                                                    'COOP',
                                                                    'INVH',
                                                                    'AMH',
                                                                    'AVB'],
                                   'Mortgage Rate Sensitive Housing': ['ITB',
                                                                       'DHI',
                                                                       'LEN',
                                                                       'PHM',
                                                                       'TOL',
                                                                       'RKT',
                                                                       'UWMC',
                                                                       'FNF'],
                                   'Tariff Beneficiaries': ['NUE',
                                                            'STLD',
                                                            'CLF',
                                                            'X',
                                                            'CAT',
                                                            'DE',
                                                            'GEV',
                                                            'ETN',
                                                            'PWR'],
                                   'Supply Chain Diversification': ['EWW',
                                                                    'INDA',
                                                                    'VNM',
                                                                    'EWT',
                                                                    'EWY',
                                                                    'EWM',
                                                                    'TSM',
                                                                    'FMX'],
                                   'Japan Reflation Beneficiaries': ['EWJ', 'DXJ', 'TM', 'HMC', 'MUFG', 'SMFG', 'SONY'],
                                   'Europe Fiscal Expansion': ['VGK', 'EZU', 'EWG', 'BAESY', 'RNMBY', 'EADSY', 'SIEGY'],
                                   'Climate Adaptation': ['URI',
                                                          'PWR',
                                                          'VMC',
                                                          'MLM',
                                                          'GNRC',
                                                          'AWK',
                                                          'XYL',
                                                          'WM',
                                                          'RSG'],
                                   'Humanoid Robotics': ['TSLA', 'NVDA', 'ISRG', 'SYM', 'ROK', 'TER', 'FANUY', 'ABB'],
                                   'Quantum Computing': ['IONQ', 'RGTI', 'QBTS', 'QUBT', 'IBM', 'GOOGL', 'MSFT'],
                                   'Autonomous Vehicles and Robotaxis': ['TSLA',
                                                                         'GOOGL',
                                                                         'GM',
                                                                         'UBER',
                                                                         'MBLY',
                                                                         'QCOM',
                                                                         'NVDA']},
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
                           'Frontier Markets': ['FM'],
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
                           'South Africa': ['EZA']}}

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
            hovertemplate=f"{basket}<br>% Cum: %{y:.1f}%<extra></extra>"
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

        **What this page excludes**
        - Pure factor baskets.
        - Pure regime diagnostics.
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

ordered_cols = [c for c in all_panel_df.index if c in all_basket_rets_display.columns]
all_display_chart_rets = all_basket_rets_display[ordered_cols] if ordered_cols else all_basket_rets_display

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

    cat_ordered_cols = [c for c in cat_panel.index if c in cat_rets_display.columns]
    chart_rets = cat_rets_display[cat_ordered_cols] if cat_ordered_cols else cat_rets_display

    plot_cumulative_chart(
        basket_returns_display=chart_rets,
        title=f"{category} | Cumulative Performance vs SPY",
        benchmark_series_display=bench_rets_display
    )


# ============================================================
# Basket constituents and data notes
# ============================================================
with st.expander("Basket Constituents"):
    st.caption("Shows only live members used in the calculations after price, stale-data, and optional market-cap filters.")

    for category, groups in live_categories.items():
        st.markdown(f"**{category}**")
        for name, tickers in groups.items():
            st.write(f"- {name}: {', '.join(sorted(set(str(t).upper() for t in tickers)))}")


with st.expander("Full Basket Map"):
    st.caption("Raw basket definitions before data-quality filtering.")
    for category, groups in CATEGORIES.items():
        st.markdown(f"**{category}**")
        for name, tickers in groups.items():
            st.write(f"- {name}: {', '.join(tickers)}")


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
