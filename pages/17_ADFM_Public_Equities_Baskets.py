import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go
from typing import Dict, List, Optional

# -----------------------------
# Page and theme
# -----------------------------
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
    "#AEC6CF","#FFB347","#B39EB5","#77DD77","#F49AC2",
    "#CFCFC4","#DEA5A4","#C6E2FF","#FFDAC1","#E2F0CB",
    "#C7CEEA","#FFB3BA","#FFD1DC","#B5EAD7","#E7E6F7",
    "#F1E3DD","#B0E0E6","#E0BBE4","#F3E5AB","#D5E8D4"
]

MIN_MARKET_CAP = 1_000_000_000

MARKET_CAP_EXCEPTIONS = set([
    # Add exceptions here if you want smaller-cap baskets to pass the filter
    # Examples: "UUUU","UEC","URG","UROY","DNN","NXE","LEU"
])

# -----------------------------
# CATEGORY -> BASKETS -> TICKERS
# Rule: every basket has at least 3 names
# -----------------------------
CATEGORIES: Dict[str, Dict[str, List[str]]] = {
    "Growth & Innovation": {
        "Semis ETFs": ["SMH","SOXX","XSD"],
        "Semis Compute and Accelerators": ["NVDA","AMD","INTC","ARM","AVGO","MRVL"],
        "Semis Analog and Power": ["TXN","ADI","MCHP","NXPI","MPWR","ON","STM","IFNNY","WOLF"],
        "Semis RF and Connectivity": ["QCOM","SWKS","QRVO","MTSI","AVNW"],
        "Semis Memory and Storage": ["MU","WDC","STX","SKM"],
        "Semis Foundry and OSAT": ["TSM","UMC","GFS","ASX"],
        "Semis Equipment": ["ASML","AMAT","LRCX","KLAC","TER","ONTO","AEIS","ACMR"],
        "Semis EDA and IP": ["SNPS","CDNS","ANSS","ARM"],
        "Semis China and HK ADRs": ["HIMX","TSM","UMC"],

        "AI Infrastructure Leaders": [
            "NVDA","AMD","AVGO","TSM","ASML",
            "ANET","SMCI","DELL","HPE",
            "AMAT","LRCX","KLAC","TER",
            "MRVL","MU","WDC","STX","NTAP",
            "ORCL","MSFT","AMZN","GOOGL"
        ],
        "Hyperscalers and Cloud": [
            "MSFT","AMZN","GOOGL","META","ORCL","IBM",
            "NOW","CRM","DDOG","SNOW","MDB","NET","ZS","OKTA"
        ],
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
    },

    "Connectivity and Industrial Tech": {
        "5G and Networking Infra": ["AMT","CCI","SBAC","ANET","CSCO","JNPR","HPE","ERIC","NOK","FFIV"],
        "Industrial Automation": ["ROK","ETN","EMR","AME","PH","ABB","FANUY","KEYS","TRMB","CGNX","IEX","ITW","GWW","SYM"],
        "Aerospace Tech and Space": ["RKLB","IRDM","ASTS","LHX","LMT","NOC","RTX"],
        "Defense Software and ISR": ["PLTR","KTOS","AVAV","LHX","LDOS","BAH"],
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
    },

    "Clean Energy Transition": {
        "Solar and Inverters": ["TAN","FSLR","ENPH","SEDG","RUN","CSIQ","JKS","SPWR"],
        "Wind and Renewables": ["ICLN","FAN","AY","NEP","FSLR"],
        "Hydrogen": ["PLUG","BE","BLDP"],
        "Utilities and Power": ["VST","CEG","NEE","DUK","SO","AEP","XEL","EXC","PCG","EIX","ED"],
        "Grid Equipment": ["ETN","VRT","ABB","PWR","MYRG","POWL"],
    },

    "Health and Longevity": {
        "Large-Cap Biotech": ["AMGN","GILD","REGN","BIIB","VRTX","ILMN"],
        "GLP-1 and Metabolic": ["NVO","LLY","AZN","MRK","PFE"],
        "MedTech Devices": ["MDT","SYK","ISRG","BSX","ZBH","EW","PEN"],
        "Diagnostics and Tools": ["TMO","DHR","A","RGEN","ILMN"],
        "Healthcare Payers": ["UNH","HUM","CI","ELV","CNC","MOH"],
        "Healthcare Providers and Hospitals": ["HCA","THC","UHS","CYH"],
        "Healthcare Services and Outsourcing": ["LH","DGX","AMN","EHC"],
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
    },

    "Defensives and Staples": {
        "Retail Staples": ["WMT","COST","TGT","DG","KR","WBA"],
        "Staples and Beverages": ["PG","KO","PEP","PM","MO","MDLZ"],
        "Telecom and Cable": ["T","VZ","TMUS","CHTR","CMCSA"],
        "Aerospace and Defense": ["LMT","NOC","RTX","GD","HII","TDG","HEI"],
        "Utilities Defensive": ["DUK","SO","AEP","XEL","EXC","ED"],
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
    },

    "Sector Expansions": {
        "Transportation Rails and Trucking": ["UNP","CSX","NSC","CNI","CP","JBHT","KNX"],
        "Transportation Parcel and Last-Mile": ["FDX","UPS","GXO","XPO"],
        "Air Cargo and Leasing": ["ATSG","AL","FTAI"],
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

    # New: behavior-first economy baskets (like your attached image)
    "Everyday Economy": {
        "Recreation and Experiences": ["YETI","FOXF","ASO","DOO","PLAY","LYV","SIX","FUN","RICK"],
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

ALL_BASKETS = {bk: tks for cat in CATEGORIES.values() for bk, tks in cat.items()}

# -----------------------------
# Data helpers
# -----------------------------
def _chunk(lst: List[str], n: int) -> List[List[str]]:
    n = max(1, n)
    return [lst[i:i+n] for i in range(0, len(lst), n)]

@st.cache_data(show_spinner=False)
def fetch_daily_levels(tickers, start, end, chunk_size: int = 40) -> pd.DataFrame:
    uniq = sorted(list(set(tickers)))
    frames = []
    for batch in _chunk(uniq, chunk_size):
        df = yf.download(batch, start=start, end=end, auto_adjust=True, progress=False)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame()
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    wide = pd.concat(frames, axis=1)
    wide = wide.loc[:, ~wide.columns.duplicated()].sort_index()
    if wide.empty:
        return wide
    bidx = pd.bdate_range(wide.index.min(), wide.index.max(), name=wide.index.name)
    return wide.reindex(bidx).ffill()

@st.cache_data(show_spinner=False)
def fetch_market_caps(tickers: List[str]) -> Dict[str, float]:
    uniq = sorted(list({t.upper() for t in tickers}))
    if not uniq:
        return {}
    caps: Dict[str, float] = {}
    tk_obj = yf.Tickers(" ".join(uniq))
    for sym, tk in tk_obj.tickers.items():
        try:
            mc_val = None
            fi = getattr(tk, "fast_info", None)
            if fi is not None:
                if isinstance(fi, dict):
                    mc_val = fi.get("market_cap")
                else:
                    mc_val = getattr(fi, "market_cap", None)
            if mc_val is not None:
                caps[sym.upper()] = float(mc_val)
        except Exception:
            continue
    return caps

def ew_rets_from_levels(
    levels: pd.DataFrame,
    baskets: dict,
    market_caps: Optional[Dict[str, float]] = None,
    min_market_cap: Optional[float] = None,
    stale_days: int = 30
) -> pd.DataFrame:
    rets = levels.pct_change()
    out = {}
    if levels.empty:
        return pd.DataFrame()
    last_idx = levels.index.max()
    for b, tks in baskets.items():
        cols = []
        for c in tks:
            c_u = str(c).upper()
            if c_u not in rets.columns:
                continue
            s = levels[c_u].dropna()
            if s.empty:
                continue
            if s.index.max() < last_idx - pd.Timedelta(days=stale_days):
                continue
            if min_market_cap is not None and market_caps is not None:
                if c_u not in MARKET_CAP_EXCEPTIONS:
                    mc = market_caps.get(c_u)
                    if mc is not None and mc < min_market_cap:
                        continue
            cols.append(c_u)
        if not cols:
            continue
        if len(cols) > 1:
            out[b] = rets[cols].mean(axis=1, skipna=True)
        else:
            out[b] = rets[cols[0]]
    return pd.DataFrame(out).dropna(how="all")

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    ag = gain.ewm(alpha=1/period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)
    rs = ag / al
    return 100 - (100 / (1 + rs))

def macd_hist(series: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig

def ema_regime(series: pd.Series, e1=4, e2=9, e3=18) -> str:
    e_1 = series.ewm(span=e1, adjust=False).mean()
    e_2 = series.ewm(span=e2, adjust=False).mean()
    e_3 = series.ewm(span=e3, adjust=False).mean()
    last = series.index[-1]
    if e_1.loc[last] > e_2.loc[last] > e_3.loc[last]:
        return "Up"
    if e_1.loc[last] < e_2.loc[last] < e_3.loc[last]:
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
    z = (latest - window.mean()) / std if std and not np.isnan(std) else 0.0

    accel = "Accelerating" if (latest - ref) > 0 else "Decelerating"
    strength = "Strong" if abs(z) > 1 else "Weak"
    return f"{base} | {accel} | {strength}"

def realized_vol(returns: pd.Series, days: int = 63, ann: int = 252) -> float:
    sub = returns.dropna().iloc[-days:]
    return float(sub.std(ddof=0) * np.sqrt(ann) * 100.0) if sub.shape[0] else np.nan

def pct_since(levels: pd.Series, start_ts: pd.Timestamp) -> float:
    sub = levels[levels.index >= start_ts]
    return float((sub.iloc[-1] / sub.iloc[0]) - 1.0) if sub.shape[0] else np.nan

# -----------------------------
# Rendering helpers
# -----------------------------
def color_ret(x):
    if pd.isna(x):
        return "white"
    if x >= 0:
        s = min(abs(x)/20.0, 1.0)
        g = int(255 - 90*s)
        return f"rgb({int(240-120*s)},{g},{int(240-120*s)})"
    s = min(abs(x)/20.0, 1.0)
    r = int(255 - 90*s)
    return f"rgb({r},{int(240-120*s)},{int(240-120*s)})"

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

def build_panel_df(
    basket_returns: pd.DataFrame,
    ref_start: pd.Timestamp,
    dynamic_label: str,
    benchmark_series: Optional[pd.Series] = None
) -> pd.DataFrame:
    if basket_returns.empty:
        cols = [
            "Basket","%5D","%1M",f"↓ %{dynamic_label}",
            "RSI(14D)","MACD Momentum","EMA 4/9/18","RSI(14W)",
            "3M RVOL","RVOL / SPY","Corr(63D)"
        ]
        return pd.DataFrame(columns=cols).set_index("Basket")

    levels_100 = 100 * (1 + basket_returns).cumprod()
    rows = []

    spy_rv = np.nan
    if benchmark_series is not None:
        spy_rv = realized_vol(benchmark_series, 63, 252)

    for b in levels_100.columns:
        s = levels_100[b].dropna()
        if s.shape[0] < 10:
            continue

        r5d = (s.iloc[-1] / s.iloc[-6]) - 1.0 if len(s) > 6 else np.nan
        r1m = pct_since(s, s.index.max() - pd.DateOffset(months=1))

        start_idx = s.index[s.index.get_indexer([pd.Timestamp(ref_start)], method="backfill")]
        r_dyn = pct_since(s, start_idx[0]) if len(start_idx) and start_idx[0] in s.index else np.nan

        rsi_d = rsi(s, 14)
        rsi_14d = rsi_d.iloc[-1] if rsi_d.dropna().shape[0] > 0 else np.nan

        weekly = s.resample("W-FRI").last().dropna()
        rsi_14w = np.nan
        if weekly.shape[0] > 20:
            rsi_w = rsi(weekly, 14)
            rsi_14w = rsi_w.iloc[-1] if rsi_w.dropna().shape[0] > 0 else np.nan

        hist = macd_hist(s, 12, 26, 9)
        macd_m = momentum_label(hist, lookback=5, z_window=63)

        ema_tag = ema_regime(s, 4, 9, 18)

        rv = realized_vol(basket_returns[b], 63, 252)
        rv_rel = np.nan
        if pd.notna(rv) and pd.notna(spy_rv) and spy_rv != 0:
            rv_rel = rv / spy_rv

        corr_spy = np.nan
        if benchmark_series is not None:
            merged = pd.concat(
                [basket_returns[b].dropna(), benchmark_series.dropna()],
                axis=1, join="inner"
            ).dropna()
            if merged.shape[0] >= 80:
                rolling_corr = merged.iloc[:, 0].rolling(63).corr(merged.iloc[:, 1])
                corr_spy = rolling_corr.iloc[-1]

        rows.append({
            "Basket": b,
            "%5D": round(r5d*100, 1) if pd.notna(r5d) else np.nan,
            "%1M": round(r1m*100, 1) if pd.notna(r1m) else np.nan,
            f"↓ %{dynamic_label}": round(r_dyn*100, 1) if pd.notna(r_dyn) else np.nan,
            "RSI(14D)": round(rsi_14d, 2) if pd.notna(rsi_14d) else np.nan,
            "MACD Momentum": macd_m,
            "EMA 4/9/18": ema_tag,
            "RSI(14W)": round(rsi_14w, 2) if pd.notna(rsi_14w) else np.nan,
            "3M RVOL": round(rv, 1) if pd.notna(rv) else np.nan,
            "RVOL / SPY": round(rv_rel, 2) if pd.notna(rv_rel) else np.nan,
            "Corr(63D)": round(corr_spy, 2) if pd.notna(corr_spy) else np.nan
        })

    if not rows:
        cols = [
            "Basket","%5D","%1M",f"↓ %{dynamic_label}",
            "RSI(14D)","MACD Momentum","EMA 4/9/18","RSI(14W)",
            "3M RVOL","RVOL / SPY","Corr(63D)"
        ]
        return pd.DataFrame(columns=cols).set_index("Basket")

    df = pd.DataFrame(rows).set_index("Basket")
    dyn_col = f"↓ %{dynamic_label}"
    if dyn_col in df.columns:
        df = df.sort_values(by=dyn_col, ascending=False)
    return df

def plot_panel_table(panel_df: pd.DataFrame):
    if panel_df.empty:
        st.info("No baskets passed the data quality checks for this window.")
        return

    dynamic_col = [c for c in panel_df.columns if c.startswith("↓ %")]
    dynamic_col = dynamic_col[0] if dynamic_col else None

    headers = [
        "Basket","%5D","%1M",dynamic_col,
        "RSI(14D)","MACD Momentum","EMA 4/9/18","RSI(14W)",
        "3M RVOL","RVOL / SPY","Corr(63D)"
    ]

    values = [panel_df.index.tolist()]
    fill_colors = [["white"] * len(panel_df)]

    for col in ["%5D","%1M",dynamic_col]:
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
        columnwidth=[int(w*1000) for w in col_widths],
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

def plot_cumulative_chart(basket_returns: pd.DataFrame, title: str, benchmark_series: pd.Series):
    if basket_returns.empty or benchmark_series.dropna().empty:
        st.info("Insufficient data to render chart for this window.")
        return
    common_index = basket_returns.index.intersection(benchmark_series.index)
    if common_index.empty:
        st.info("No overlapping dates between series and benchmark.")
        return

    cum_pct = ((1 + basket_returns.loc[common_index]).cumprod() - 1.0) * 100.0
    bm_cum = ((1 + benchmark_series.loc[common_index]).cumprod() - 1.0) * 100.0

    fig = go.Figure()
    for i, b in enumerate(cum_pct.columns):
        fig.add_trace(go.Scatter(
            x=cum_pct.index,
            y=cum_pct[b],
            mode="lines",
            line=dict(width=2, color=PASTEL[i % len(PASTEL)]),
            name=b,
            hovertemplate=f"{b}<br>% Cum: %{{y:.1f}}%<extra></extra>"
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

# -----------------------------
# Sidebar - presets
# -----------------------------
st.title(TITLE)
st.caption(SUBTITLE)

with st.sidebar:
    st.markdown("### About This Tool")
    st.write(
        "Daily metrics: %5D, %1M, preset-matched %, RSI(14D)/(14W), MACD momentum, EMA 4/9/18, "
        "3M RVOL, RVOL/SPY, and rolling 63D correlation to SPY. Equal-weight inside each basket. "
        "Business-day alignment for consistent panels and hover."
    )
    st.write(f"Constituents are filtered for stale data and market cap ≥ {MIN_MARKET_CAP:,.0f} USD, with optional exceptions.")
    st.divider()
    st.markdown("### Controls")

    today = date.today()

    PRESETS = {
        "YTD": lambda t: date(t.year, 1, 1),
        "1W": lambda t: t - timedelta(days=7),
        "1M": lambda t: t - timedelta(days=30),
        "3M": lambda t: t - timedelta(days=90),
        "1Y": lambda t: t - timedelta(days=365),
        "3Y": lambda t: t - timedelta(days=365*3),
        "5Y": lambda t: t - timedelta(days=365*5),
    }

    preset = st.selectbox("Date Range Preset", list(PRESETS.keys()), index=0)
    start_date = PRESETS[preset](today)
    end_date = today

LABEL_MAP = {"YTD":"YTD","1W":"1W","1M":"1M","3M":"3M","1Y":"1Y","3Y":"3Y","5Y":"5Y"}
DYNAMIC_LABEL = LABEL_MAP.get(preset, "YTD")

# -----------------------------
# Data fetch once
# -----------------------------
bench = "SPY"
need = {bench}
for tks in ALL_BASKETS.values():
    need.update([str(t).upper() for t in tks])

levels = fetch_daily_levels(
    sorted(list(need)),
    start=pd.to_datetime(start_date),
    end=pd.to_datetime(end_date) + pd.Timedelta(days=1)
)

if levels.empty:
    st.error("No data returned for the selected range.")
    st.stop()

market_caps = fetch_market_caps(list(levels.columns))

all_basket_rets = ew_rets_from_levels(
    levels,
    ALL_BASKETS,
    market_caps=market_caps,
    min_market_cap=MIN_MARKET_CAP,
    stale_days=30
)

if bench not in levels.columns or levels[bench].dropna().empty:
    st.error("SPY data missing or empty for the selected range.")
    st.stop()

bench_rets = levels[bench].pct_change().dropna()

# -----------------------------
# Consolidated top panel + chart
# -----------------------------
st.subheader("All Baskets - Consolidated Panel")
all_panel_df = build_panel_df(
    all_basket_rets,
    ref_start=pd.Timestamp(start_date),
    dynamic_label=DYNAMIC_LABEL,
    benchmark_series=bench_rets
)
plot_panel_table(all_panel_df)

st.subheader("All Baskets - Cumulative Performance vs SPY")
plot_cumulative_chart(
    all_basket_rets[all_panel_df.index] if not all_panel_df.empty else all_basket_rets,
    title="All Baskets vs SPY",
    benchmark_series=bench_rets
)

# -----------------------------
# Per-category sections
# -----------------------------
for category, baskets in CATEGORIES.items():
    st.markdown(f"## {category}")
    cat_names = [bk for bk in baskets.keys() if bk in all_basket_rets.columns]
    cat_rets = all_basket_rets[cat_names].dropna(how="all")
    if cat_rets.empty:
        st.info("No data for this group in the selected range.")
        continue

    cat_panel = build_panel_df(
        cat_rets,
        ref_start=pd.Timestamp(start_date),
        dynamic_label=DYNAMIC_LABEL,
        benchmark_series=bench_rets
    )

    plot_panel_table(cat_panel)

    plot_cumulative_chart(
        cat_rets[cat_panel.index] if not cat_panel.empty else cat_rets,
        title=f"{category} - Cumulative Performance vs SPY",
        benchmark_series=bench_rets
    )

# -----------------------------
# Basket constituents
# -----------------------------
with st.expander("Basket Constituents"):
    for cat, groups in CATEGORIES.items():
        st.markdown(f"**{cat}**")
        for name, tks in groups.items():
            st.write(f"- {name}: {', '.join(sorted(set(str(t).upper() for t in tks)))}")
