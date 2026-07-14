"""Shared sector and subsector definitions for ADFM equity pages."""

from typing import Dict, List


MAJOR_SECTORS: Dict[str, str] = {
    "XLC": "Communication Services", "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples", "XLE": "Energy", "XLF": "Financials",
    "XLV": "Health Care", "XLI": "Industrials", "XLB": "Materials",
    "XLRE": "Real Estate", "XLK": "Technology", "XLU": "Utilities",
}

SUBSECTOR_ROWS: List[Dict[str, str]] = [
    {"Ticker":"IYZ","Name":"Telecom","Sector Group":"Communication Services","Tier":"Core"},
    {"Ticker":"FDN","Name":"Internet / platform growth","Sector Group":"Communication Services","Tier":"Core"},
    {"Ticker":"PBS","Name":"Media / entertainment","Sector Group":"Communication Services","Tier":"Core"},
    {"Ticker":"SOCL","Name":"Social media / online networks","Sector Group":"Communication Services","Tier":"Thematic"},
    {"Ticker":"ESPO","Name":"Video games / esports","Sector Group":"Communication Services","Tier":"Thematic"},
    {"Ticker":"XRT","Name":"Retail","Sector Group":"Consumer Discretionary","Tier":"Core"},
    {"Ticker":"RTH","Name":"Large-cap retail","Sector Group":"Consumer Discretionary","Tier":"Core"},
    {"Ticker":"XHB","Name":"Homebuilders","Sector Group":"Consumer Discretionary","Tier":"Core"},
    {"Ticker":"ITB","Name":"Home construction","Sector Group":"Consumer Discretionary","Tier":"Core"},
    {"Ticker":"JETS","Name":"Airlines","Sector Group":"Consumer Discretionary","Tier":"Core"},
    {"Ticker":"PEJ","Name":"Travel / leisure","Sector Group":"Consumer Discretionary","Tier":"Core"},
    {"Ticker":"CARZ","Name":"Autos","Sector Group":"Consumer Discretionary","Tier":"Thematic"},
    {"Ticker":"IBUY","Name":"Online retail / e-commerce","Sector Group":"Consumer Discretionary","Tier":"Thematic"},
    {"Ticker":"BJK","Name":"Gaming / casinos","Sector Group":"Consumer Discretionary","Tier":"Thematic"},
    {"Ticker":"PBJ","Name":"Food & beverage","Sector Group":"Consumer Staples","Tier":"Core"},
    {"Ticker":"RHS","Name":"Equal-weight staples","Sector Group":"Consumer Staples","Tier":"Core"},
    {"Ticker":"IYK","Name":"Broad staples confirmation","Sector Group":"Consumer Staples","Tier":"Core"},
    {"Ticker":"MOO","Name":"Agribusiness / food supply chain","Sector Group":"Consumer Staples","Tier":"Thematic"},
    {"Ticker":"XOP","Name":"E&P","Sector Group":"Energy","Tier":"Core"},
    {"Ticker":"OIH","Name":"Oil services","Sector Group":"Energy","Tier":"Core"},
    {"Ticker":"XES","Name":"Oil equipment & services equal-weight","Sector Group":"Energy","Tier":"Core"},
    {"Ticker":"FCG","Name":"Natural gas equities","Sector Group":"Energy","Tier":"Core"},
    {"Ticker":"AMLP","Name":"Midstream / MLPs","Sector Group":"Energy","Tier":"Core"},
    {"Ticker":"ICLN","Name":"Clean energy","Sector Group":"Energy","Tier":"Thematic"},
    {"Ticker":"TAN","Name":"Solar","Sector Group":"Energy","Tier":"Thematic"},
    {"Ticker":"URA","Name":"Uranium / nuclear fuel cycle","Sector Group":"Energy","Tier":"Thematic"},
    {"Ticker":"KBE","Name":"Banks","Sector Group":"Financials","Tier":"Core"},
    {"Ticker":"KRE","Name":"Regional banks","Sector Group":"Financials","Tier":"Core"},
    {"Ticker":"KBWB","Name":"Money-center banks","Sector Group":"Financials","Tier":"Core"},
    {"Ticker":"KIE","Name":"Insurance","Sector Group":"Financials","Tier":"Core"},
    {"Ticker":"KCE","Name":"Capital markets / brokers","Sector Group":"Financials","Tier":"Core"},
    {"Ticker":"IAI","Name":"Broker-dealers / investment banks","Sector Group":"Financials","Tier":"Core"},
    {"Ticker":"BIZD","Name":"BDCs / private credit beta","Sector Group":"Financials","Tier":"Thematic"},
    {"Ticker":"PSP","Name":"Private equity / alt managers","Sector Group":"Financials","Tier":"Thematic"},
    {"Ticker":"FINX","Name":"Fintech","Sector Group":"Financials","Tier":"Thematic"},
    {"Ticker":"XBI","Name":"Biotech equal-weight","Sector Group":"Health Care","Tier":"Core"},
    {"Ticker":"IBB","Name":"Biotech cap-weight","Sector Group":"Health Care","Tier":"Core"},
    {"Ticker":"XPH","Name":"Pharmaceuticals","Sector Group":"Health Care","Tier":"Core"},
    {"Ticker":"IHE","Name":"Large pharma","Sector Group":"Health Care","Tier":"Core"},
    {"Ticker":"IHI","Name":"Medical devices","Sector Group":"Health Care","Tier":"Core"},
    {"Ticker":"XHE","Name":"Health care equipment equal-weight","Sector Group":"Health Care","Tier":"Core"},
    {"Ticker":"IHF","Name":"Health care providers","Sector Group":"Health Care","Tier":"Core"},
    {"Ticker":"XHS","Name":"Health care services","Sector Group":"Health Care","Tier":"Core"},
    {"Ticker":"ARKG","Name":"Genomics / speculative health innovation","Sector Group":"Health Care","Tier":"Thematic"},
    {"Ticker":"ITA","Name":"Aerospace & defense","Sector Group":"Industrials","Tier":"Core"},
    {"Ticker":"XAR","Name":"Aerospace & defense equal-weight","Sector Group":"Industrials","Tier":"Core"},
    {"Ticker":"PPA","Name":"Defense primes / contractor basket","Sector Group":"Industrials","Tier":"Core"},
    {"Ticker":"IYT","Name":"Transportation","Sector Group":"Industrials","Tier":"Core"},
    {"Ticker":"XTN","Name":"Transportation equal-weight","Sector Group":"Industrials","Tier":"Core"},
    {"Ticker":"PAVE","Name":"Infrastructure","Sector Group":"Industrials","Tier":"Core"},
    {"Ticker":"AIRR","Name":"Small/mid industrial cyclicals","Sector Group":"Industrials","Tier":"Core"},
    {"Ticker":"PHO","Name":"Water infrastructure","Sector Group":"Industrials","Tier":"Thematic"},
    {"Ticker":"BOTZ","Name":"Robotics / automation","Sector Group":"Industrials","Tier":"Thematic"},
    {"Ticker":"XME","Name":"Metals & mining","Sector Group":"Materials","Tier":"Core"},
    {"Ticker":"SLX","Name":"Steel","Sector Group":"Materials","Tier":"Core"},
    {"Ticker":"COPX","Name":"Copper miners","Sector Group":"Materials","Tier":"Core"},
    {"Ticker":"PICK","Name":"Broad global mining","Sector Group":"Materials","Tier":"Core"},
    {"Ticker":"GDX","Name":"Gold miners","Sector Group":"Materials","Tier":"Core"},
    {"Ticker":"GDXJ","Name":"Junior gold miners","Sector Group":"Materials","Tier":"Core"},
    {"Ticker":"SIL","Name":"Silver miners","Sector Group":"Materials","Tier":"Core"},
    {"Ticker":"LIT","Name":"Lithium / battery materials","Sector Group":"Materials","Tier":"Thematic"},
    {"Ticker":"REMX","Name":"Rare earths / strategic materials","Sector Group":"Materials","Tier":"Thematic"},
    {"Ticker":"WOOD","Name":"Timber / forest products","Sector Group":"Materials","Tier":"Thematic"},
    {"Ticker":"VNQ","Name":"Broad REITs","Sector Group":"Real Estate","Tier":"Core"},
    {"Ticker":"IYR","Name":"U.S. real estate","Sector Group":"Real Estate","Tier":"Core"},
    {"Ticker":"REZ","Name":"Residential / multisector REITs","Sector Group":"Real Estate","Tier":"Core"},
    {"Ticker":"REM","Name":"Mortgage REITs","Sector Group":"Real Estate","Tier":"Core"},
    {"Ticker":"KBWY","Name":"High-yield REITs","Sector Group":"Real Estate","Tier":"Thematic"},
    {"Ticker":"HOMZ","Name":"Housing ecosystem","Sector Group":"Real Estate","Tier":"Thematic"},
    {"Ticker":"SMH","Name":"Semiconductors","Sector Group":"Technology","Tier":"Core"},
    {"Ticker":"SOXX","Name":"Semiconductors broader","Sector Group":"Technology","Tier":"Core"},
    {"Ticker":"XSD","Name":"Semiconductors equal-weight","Sector Group":"Technology","Tier":"Core"},
    {"Ticker":"IGV","Name":"Software","Sector Group":"Technology","Tier":"Core"},
    {"Ticker":"SKYY","Name":"Cloud infrastructure","Sector Group":"Technology","Tier":"Core"},
    {"Ticker":"CIBR","Name":"Cybersecurity","Sector Group":"Technology","Tier":"Core"},
    {"Ticker":"HACK","Name":"Cybersecurity high beta","Sector Group":"Technology","Tier":"Core"},
    {"Ticker":"IGM","Name":"Expanded technology","Sector Group":"Technology","Tier":"Core"},
    {"Ticker":"WCLD","Name":"Cloud software","Sector Group":"Technology","Tier":"Thematic"},
    {"Ticker":"AIQ","Name":"AI / automation","Sector Group":"Technology","Tier":"Thematic"},
    {"Ticker":"ROBO","Name":"Robotics","Sector Group":"Technology","Tier":"Thematic"},
    {"Ticker":"RYU","Name":"Equal-weight utilities","Sector Group":"Utilities","Tier":"Core"},
    {"Ticker":"PUI","Name":"Utilities momentum / defensive confirmation","Sector Group":"Utilities","Tier":"Core"},
    {"Ticker":"GRID","Name":"Grid / electrification","Sector Group":"Utilities","Tier":"Thematic"},
    {"Ticker":"NLR","Name":"Nuclear energy","Sector Group":"Utilities","Tier":"Thematic"},
]

BENCHMARKS: Dict[str, str] = {
    "SPY": "SPDR S&P 500", "RSP": "Invesco S&P 500 Equal Weight", "QQQ": "Invesco QQQ",
    "IWM": "iShares Russell 2000", "DIA": "SPDR Dow Jones Industrial Average",
    "TLT": "iShares 20+ Year Treasury Bond", "IEF": "iShares 7-10 Year Treasury Bond",
    "UUP": "Invesco DB US Dollar Index Bullish Fund",
}

SECTOR_GROUP_COLORS = {
    "Communication Services":"#4E79A7", "Consumer Discretionary":"#F28E2B", "Consumer Staples":"#59A14F",
    "Energy":"#E15759", "Financials":"#76B7B2", "Health Care":"#EDC948", "Industrials":"#B07AA1",
    "Materials":"#FF9DA7", "Real Estate":"#9C755F", "Technology":"#2F5597", "Utilities":"#BAB0AC",
}

LOOKBACK_PERIOD = "3y"
INTERVAL = "1d"
DOWNLOAD_CHUNK_SIZE = 40
DOWNLOAD_RETRIES = 3
FORWARD_FILL_LIMIT = 3
MAX_STALE_SESSIONS = 3
NEUTRAL_MAP_THRESHOLD = 0.10
MIN_SCORE_COMPONENTS = 5

WINDOW_PRESETS = {
    "Fast (1M vs 3M)": {"short": 21, "long": 63, "label_short": "1M", "label_long": "3M"},
    "Intermediate (3M vs 6M)": {"short": 63, "long": 126, "label_short": "3M", "label_long": "6M"},
    "Trend (6M vs 12M)": {"short": 126, "long": 252, "label_short": "6M", "label_long": "12M"},
}
TRAIL_OPTIONS = {"None": 0, "4 weeks": 4, "8 weeks": 8, "12 weeks": 12}
ROTATION_MODES = {"Benchmark-relative rotation": "relative", "Absolute sector/subsector rotation": "absolute"}
UNIVERSE_SCOPES = ["Major sectors only", "Core subsectors", "Core + thematic subsectors"]
LABEL_MODES = ["Top ranked only", "All tickers", "No labels"]
QUADRANT_LABELS = {"Q1": "Leading", "Q2": "Improving", "Q3": "Lagging", "Q4": "Weakening", "Neutral": "Neutral"}
STATE_COLORS = {"Leading": "#16a34a", "Improving": "#2563eb", "Lagging": "#dc2626", "Weakening": "#f59e0b", "Neutral": "#6b7280"}
