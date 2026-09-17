"""Expanded catalog inputs for Sector Breadth and Rotation."""

from typing import Sequence

import pandas as pd

from adfm_sector_rotation_config import MAJOR_SECTORS, SUBSECTOR_ROWS

LOOKBACK_PERIOD = "3y"
NEUTRAL_MAP_THRESHOLD = 0.01
STATE_CONFIRM_DAYS = 3
MOVEMENT_LOOKBACK = 5
BREADTH_MIN_COVERAGE = 0.60
BASKET_MIN_COVERAGE = 0.60
UNIVERSE_SCOPES = ["Sectors", "Industries", "Themes", "Countries", "All"]
LABEL_MODES = ["Selected only", "Top ranked only", "All tickers", "No labels"]
STATE_STREET_SECTOR_ETFS = set(MAJOR_SECTORS)
STATE_STREET_HOLDINGS_URL = (
    "https://www.ssga.com/us/en/intermediary/library-content/products/"
    "fund-data/etfs/us/holdings-daily-us-en-{ticker}.xlsx"
)
PARENT_SECTOR_ETF = {name: ticker for ticker, name in MAJOR_SECTORS.items()}

ADDITIONAL_SUBSECTORS = [
    {"Ticker":"IPAY","Name":"Payments","Sector Group":"Financials","Tier":"Core"},
    {"Ticker":"XSW","Name":"Software & services equal-weight","Sector Group":"Technology","Tier":"Core"},
    {"Ticker":"XNTK","Name":"Next-generation technology","Sector Group":"Technology","Tier":"Thematic"},
    {"Ticker":"PKB","Name":"Building & construction","Sector Group":"Industrials","Tier":"Core"},
    {"Ticker":"SRVR","Name":"Data centers / digital infrastructure REITs","Sector Group":"Real Estate","Tier":"Core"},
    {"Ticker":"CRAK","Name":"Oil refiners","Sector Group":"Energy","Tier":"Core"},
    {"Ticker":"XTL","Name":"Telecom equal-weight","Sector Group":"Communication Services","Tier":"Core"},
]

_STOCK_BASKET_SPECS = [
    ("BASKET_CUSTODY","Custody and Trust Banks","Financials","BNY STT NTRS"),
    ("BASKET_EXCHANGES","Exchanges and Market Data","Financials","CME ICE NDAQ CBOE MKTX SPGI MSCI"),
    ("BASKET_ASSET_MGRS","Traditional Asset Managers","Financials","BLK TROW BEN AMG IVZ JHG"),
    ("BASKET_ALT_MGRS","Alternative Asset Managers","Financials","BX KKR APO CG ARES OWL"),
    ("BASKET_PAYMENTS","Payments Networks and Processors","Financials","V MA FI FIS GPN"),
    ("BASKET_CONSUMER_FIN","Consumer Finance","Financials","COF SYF AXP OMF SOFI"),
    ("BASKET_MORTGAGE_FIN","Mortgage Finance","Financials","RKT UWMC COOP PFSI"),
    ("BASKET_SEMICAP","Semiconductor Equipment","Technology","ASML AMAT LRCX KLAC TER ONTO"),
    ("BASKET_DC_NETWORK","Data Center Networking","Technology","ANET CSCO CIEN LITE COHR NOK"),
    ("BASKET_OPTICAL","Optical Networking and Interconnect","Technology","CIEN LITE COHR AAOI"),
    ("BASKET_ENTERPRISE_SW","Enterprise Software","Technology","MSFT ORCL CRM NOW ADBE INTU"),
    ("BASKET_AI_HARDWARE","Servers and AI Hardware","Technology","DELL HPE SMCI IBM NTAP PSTG"),
    ("BASKET_ELECTRICAL","Electrical Equipment","Industrials","ETN HUBB VRT EMR ROK NVT"),
    ("BASKET_MACHINERY","Machinery","Industrials","CAT DE CMI PCAR PH TEX"),
    ("BASKET_EC","Engineering and Construction","Industrials","PWR EME FIX MTZ ACM J"),
    ("BASKET_LOGISTICS","Logistics and Freight","Industrials","UPS FDX JBHT CHRW EXPD XPO"),
    ("BASKET_RESTAURANTS","Restaurants","Consumer Discretionary","MCD SBUX CMG YUM DRI DPZ CAVA"),
    ("BASKET_HOTELS","Hotels and Lodging","Consumer Discretionary","MAR HLT H WH ABNB"),
    ("BASKET_CRUISES","Cruise Lines","Consumer Discretionary","CCL RCL NCLH"),
    ("BASKET_LUXURY","Luxury and Premium Apparel","Consumer Discretionary","TPR RL CPRI DECK"),
    ("BASKET_HOME_IMPROVEMENT","Home Improvement and Building Products","Consumer Discretionary","HD LOW FND BLDR TREX"),
    ("BASKET_APARTMENTS","Apartment REITs","Real Estate","AVB EQR ESS MAA CPT UDR"),
    ("BASKET_INDUSTRIAL_REIT","Industrial and Warehouse REITs","Real Estate","PLD REXR FR STAG TRNO"),
    ("BASKET_OFFICE_REIT","Office REITs","Real Estate","BXP VNO KRC HIW SLG"),
    ("BASKET_RETAIL_REIT","Retail REITs","Real Estate","SPG O REG KIM FRT BRX"),
    ("BASKET_HEALTHCARE_REIT","Healthcare REITs","Real Estate","WELL VTR DOC CTRE HR"),
    ("BASKET_TOWERS","Cell Towers","Real Estate","AMT CCI SBAC"),
    ("BASKET_DATA_CENTER_REIT","Data Center REITs","Real Estate","EQIX DLR IRM"),
    ("BASKET_REFINERS","Refiners","Energy","VLO MPC PSX DK PBF"),
    ("BASKET_INTEGRATED_OIL","Integrated Oil","Energy","XOM CVX SHEL BP TTE"),
    ("BASKET_LNG","LNG Exporters and Infrastructure","Energy","LNG KMI WMB TRGP ET"),
    ("BASKET_COAL","Coal Producers","Energy","BTU CNR AMR HCC METC"),
    ("BASKET_FERTILIZER","Fertilizers","Materials","NTR MOS CF ICL"),
    ("BASKET_MANAGED_CARE","Managed Care","Health Care","UNH ELV CI CVS HUM CNC MOH"),
    ("BASKET_HOSPITALS","Hospitals","Health Care","HCA THC UHS CYH"),
    ("BASKET_LIFE_SCIENCE","Life Science Tools","Health Care","TMO DHR A IQV WAT MTD RVTY"),
]
STOCK_BASKETS = [
    {"Key": key, "Name": name, "Sector Group": group, "Members": members.split()}
    for key, name, group, members in _STOCK_BASKET_SPECS
]

_COUNTRY_SPECS = [
    ("EWC","Canada"),("EWW","Mexico"),("EWZ","Brazil"),("ARGT","Argentina"),("ECH","Chile"),
    ("EPU","Peru"),("GXG","Colombia"),("EWU","United Kingdom"),("EWG","Germany"),("EWQ","France"),
    ("EWI","Italy"),("EWP","Spain"),("EWL","Switzerland"),("EWD","Sweden"),("EDEN","Denmark"),
    ("NORW","Norway"),("EWN","Netherlands"),("EWO","Austria"),("EWK","Belgium"),("EPOL","Poland"),
    ("TUR","Turkey"),("GREK","Greece"),("EWJ","Japan"),("EWY","South Korea"),("EWT","Taiwan"),
    ("EWH","Hong Kong"),("EWS","Singapore"),("EWA","Australia"),("ENZL","New Zealand"),("MCHI","China"),
    ("INDA","India"),("EIDO","Indonesia"),("EWM","Malaysia"),("THD","Thailand"),("EPHE","Philippines"),
    ("VNM","Vietnam"),("KSA","Saudi Arabia"),("UAE","United Arab Emirates"),("EIS","Israel"),("EZA","South Africa"),
]
COUNTRY_ROWS = [
    {"Ticker": ticker, "Name": name, "Sector Group": "Countries & Regions", "Tier": "Country"}
    for ticker, name in _COUNTRY_SPECS
]


def build_catalog() -> pd.DataFrame:
    rows = []
    for ticker, name in MAJOR_SECTORS.items():
        rows.append(
            {
                "Key": ticker,
                "Ticker": ticker,
                "Name": name,
                "Sector Group": name,
                "Tier": "Major Sector",
                "Universe": "Sectors",
                "Kind": "ETF",
                "Members": None,
                "Broad Benchmark": "SPY",
                "Parent Benchmark": "",
            }
        )

    subsector_rows = list(SUBSECTOR_ROWS)
    existing_subsector_tickers = {row["Ticker"] for row in subsector_rows}
    subsector_rows.extend(
        row for row in ADDITIONAL_SUBSECTORS if row["Ticker"] not in existing_subsector_tickers
    )
    for row in subsector_rows:
        tier = row["Tier"]
        rows.append(
            {
                "Key": row["Ticker"],
                "Ticker": row["Ticker"],
                "Name": row["Name"],
                "Sector Group": row["Sector Group"],
                "Tier": tier,
                "Universe": "Themes" if tier == "Thematic" else "Industries",
                "Kind": "ETF",
                "Members": None,
                "Broad Benchmark": "SPY",
                "Parent Benchmark": PARENT_SECTOR_ETF.get(row["Sector Group"], ""),
            }
        )

    for basket in STOCK_BASKETS:
        rows.append(
            {
                "Key": basket["Key"],
                "Ticker": "",
                "Name": basket["Name"],
                "Sector Group": basket["Sector Group"],
                "Tier": "Basket",
                "Universe": "Industries",
                "Kind": "Stock Basket",
                "Members": list(basket["Members"]),
                "Broad Benchmark": "SPY",
                "Parent Benchmark": PARENT_SECTOR_ETF.get(basket["Sector Group"], ""),
            }
        )

    for row in COUNTRY_ROWS:
        rows.append(
            {
                "Key": row["Ticker"],
                "Ticker": row["Ticker"],
                "Name": row["Name"],
                "Sector Group": row["Sector Group"],
                "Tier": row["Tier"],
                "Universe": "Countries",
                "Kind": "ETF",
                "Members": None,
                "Broad Benchmark": "ACWI",
                "Parent Benchmark": "",
            }
        )

    catalog = pd.DataFrame(rows)
    duplicates = catalog.loc[catalog["Key"].duplicated(keep=False), "Key"].tolist()
    if duplicates:
        raise ValueError(f"Duplicate sector-rotation keys: {sorted(set(duplicates))}")
    return catalog.reset_index(drop=True)


def select_catalog(
    catalog: pd.DataFrame,
    universe: str,
    sector_groups: Sequence[str] | None = None,
) -> pd.DataFrame:
    if universe == "All":
        selected = catalog.copy()
    else:
        selected = catalog[catalog["Universe"] == universe].copy()
    if sector_groups is not None:
        selected = selected[selected["Sector Group"].isin(list(sector_groups))]
    return selected.reset_index(drop=True)
