import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

# ---------------- Config ----------------
st.set_page_config(page_title="Factor Momentum and Basket Rotation", layout="wide")
plt.style.use("default")

PASTELS = [
    "#A8DADC", "#F4A261", "#90BE6D", "#FFD6A5", "#BDE0FE",
    "#CDB4DB", "#E2F0CB", "#F1C0E8", "#B9FBC0", "#F7EDE2"
]
TEXT = "#222222"
GRID = "#e6e6e6"

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
    h1, h2, h3 {font-weight: 600; letter-spacing: 0.15px;}
    .stPlotlyChart {background: #ffffff;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def card_box(inner_html: str):
    st.markdown(
        f"""
        <div style="border:1px solid #e0e0e0; border-radius:10px;
                    padding:14px; background:#fafafa; color:{TEXT};
                    font-size:14px; line-height:1.35;">
          {inner_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- Helpers: prices / factors ----------------
def rs(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return aligned.iloc[:, 0] / aligned.iloc[:, 1]

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def pct_change_window(series: pd.Series, days: int) -> float:
    if len(series) <= 1:
        return np.nan
    days = int(min(days, len(series) - 1))
    return float(series.iloc[-1] / series.iloc[-days] - 1.0)

def momentum(series: pd.Series, win: int = 20) -> float:
    r = series.pct_change().dropna()
    if len(r) < 2:
        return np.nan
    win = int(min(win, len(r)))
    return float(r.rolling(win).mean().iloc[-1])

def trend_class(series: pd.Series) -> str:
    if len(series) < 50:
        return "Neutral"
    e1 = ema(series, 10).iloc[-1]
    e2 = ema(series, 20).iloc[-1]
    e3 = ema(series, 40).iloc[-1]
    if e1 > e2 > e3:
        return "Up"
    if e1 < e2 < e3:
        return "Down"
    return "Neutral"

def inflection(short_mom: float, long_mom: float) -> str:
    if pd.isna(short_mom) or pd.isna(long_mom):
        return "Neutral"
    if short_mom > 0 and long_mom < 0:
        return "Turning Up"
    if short_mom < 0 and long_mom > 0:
        return "Turning Down"
    if abs(short_mom) > abs(long_mom):
        return "Strengthening"
    return "Weakening"

def bucket_breadth(breadth: float) -> str:
    if breadth < 10:
        return "extremely narrow"
    if breadth < 25:
        return "narrow and selective"
    if breadth < 40:
        return "tilted to a small group of styles"
    if breadth < 60:
        return "balanced across factors"
    if breadth < 75:
        return "broadening out across styles"
    return "very broad and inclusive"

def bucket_regime(regime_score: float) -> str:
    if regime_score < 25:
        return "deeply defensive and stress driven"
    if regime_score < 40:
        return "defensive and risk averse"
    if regime_score < 55:
        return "roughly neutral with a mild defensive lean"
    if regime_score < 70:
        return "constructive and risk friendly"
    return "high beta, late-cycle risk on"

def build_commentary(
    mom_df: pd.DataFrame,
    breadth: float,
    regime_score: float,
    corr: Optional[pd.DataFrame] = None,
) -> str:
    trend_counts = mom_df["Trend"].value_counts()
    up_count = int(trend_counts.get("Up", 0))
    down_count = int(trend_counts.get("Down", 0))

    established_leaders = mom_df[(mom_df["Short"] > 0) & (mom_df["Long"] > 0)].sort_values("Short", ascending=False).index.tolist()
    new_rotations = mom_df[mom_df["Inflection"] == "Turning Up"].index.tolist()
    fading_leaders = mom_df[mom_df["Inflection"] == "Turning Down"].index.tolist()

    leaders_text = ", ".join(established_leaders[:4]) if established_leaders else "no factor pair in a clean dual-horizon uptrend"
    rotations_text = ", ".join(new_rotations[:4]) if new_rotations else "no factor is clearly turning up yet"
    fading_text = ", ".join(fading_leaders[:4]) if fading_leaders else "no obvious factor is rolling over from strength"

    breadth_desc = bucket_breadth(breadth)
    regime_desc = bucket_regime(regime_score)

    crowd_line = "Correlation picture is mixed and does not add a strong crowding signal."
    if corr is not None and not corr.empty:
        avg_abs = {}
        for f in corr.columns:
            vals = corr.loc[f].drop(f)
            if vals.empty:
                continue
            avg_abs[f] = float(vals.abs().mean())
        if avg_abs:
            crowded_factor = max(avg_abs, key=avg_abs.get)
            diversifier = min(avg_abs, key=avg_abs.get)
            crowd_line = (
                f"Most crowded style on this window is {crowded_factor} with average |corr| around "
                f"{avg_abs[crowded_factor]:.2f} to peers, while the cleanest diversifier is "
                f"{diversifier} with average |corr| near {avg_abs[diversifier]:.2f}."
            )

    conclusion = (
        f"Factor tape is {breadth_desc} and currently {regime_desc}. "
        f"Leadership is anchored in {leaders_text}, with rotations starting to show up in "
        f"{rotations_text}, and pressure building in {fading_text}."
    )

    why_matters = (
        "This grid is the style map for the equity tape. It tells you which buckets the market is "
        "paying for right now, how persistent that preference is across short and long windows, "
        "and whether you should lean into existing trends or hunt for rotations."
    )

    drivers = []
    drivers.append(
        f"{up_count} factors are in up trends and {down_count} are in down trends based on the "
        "10/20/40-day moving average stack, with the rest stuck in noisy ranges."
    )

    drivers.append(
        f"Short horizon strength is concentrated in "
        f"{', '.join(mom_df.sort_values('Short', ascending=False).index.tolist()[:5])}, "
        "while the weakest short-term tape sits in "
        f"{', '.join(mom_df.sort_values('Short', ascending=True).index.tolist()[:3])}."
    )

    if new_rotations:
        drivers.append(
            f"Inflection signals flag {', '.join(new_rotations)} as turning up from weaker long-term trends, "
            "which is where new leaders usually emerge if the regime stays constructive."
        )
    if fading_leaders:
        drivers.append(
            f"On the other side, {', '.join(fading_leaders)} are turning down against still-positive long windows, "
            "a typical pattern near the end of a leadership run."
        )

    drivers.append(crowd_line)

    key_stats = (
        f"Breadth index {breadth:.1f}%. "
        f"Regime score {regime_score:.1f} on a 0-100 scale, where 50 is neutral."
    )

    body = (
        '<div style="font-weight:700; margin-bottom:6px;">Conclusion</div>'
        f'<div>{conclusion}</div>'
        '<div style="font-weight:700; margin:10px 0 6px;">Why it matters</div>'
        f'<div>{why_matters}</div>'
        '<div style="font-weight:700; margin:10px 0 6px;">Key drivers</div>'
        '<ul style="margin-top:4px; margin-bottom:4px;">'
        + "".join(f"<li>{d}</li>" for d in drivers)
        + "</ul>"
        '<div style="font-weight:700; margin:10px 0 6px;">Key stats</div>'
        f'<div>{key_stats}</div>'
    )
    return body

# ---------------- Factors ----------------
FACTOR_ETFS = {
    "Growth vs Value": ("VUG", "VTV"),
    "Quality vs Junk": ("QUAL", "JNK"),
    "High Beta vs Low Vol": ("SPHB", "SPLV"),
    "Small vs Large": ("IWM", "SPY"),
    "Tech vs Broad": ("XLK", "SPY"),
    "Cyclicals vs Defensives": ("XLY", "XLP"),
    "US vs World": ("SPY", "VEA"),
    "Momentum": ("MTUM", None),
    "Equal Weight vs Cap": ("RSP", "SPY"),
}

# ---------------- Basket universe (your basket engine) ----------------
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

ALL_BASKETS: Dict[str, List[str]] = {bk: tks for cat in CATEGORIES.values() for bk, tks in cat.items()}

# ---------------- Basket engine helpers ----------------
def _chunk(lst: List[str], n: int) -> List[List[str]]:
    n = max(1, int(n))
    return [lst[i:i+n] for i in range(0, len(lst), n)]

@st.cache_data(show_spinner=False)
def fetch_daily_levels(tickers: List[str], start: pd.Timestamp, end: pd.Timestamp, chunk_size: int = 40) -> pd.DataFrame:
    uniq = sorted(list({str(t).upper() for t in tickers if t}))
    frames = []
    for batch in _chunk(uniq, chunk_size):
        df = yf.download(batch, start=start, end=end, auto_adjust=True, progress=False)
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            if ("Close" in df.columns.get_level_values(0)):
                df = df["Close"]
            else:
                df = df.xs("Close", axis=1, level=0, drop_level=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(-1)
        else:
            # sometimes yfinance returns a single series for one ticker
            if isinstance(df, pd.Series):
                df = df.to_frame()
        if not df.empty:
            df.columns = [str(c).upper() for c in df.columns]
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    wide = pd.concat(frames, axis=1)
    wide = wide.loc[:, ~wide.columns.duplicated()].sort_index()
    if wide.empty:
        return wide

    bidx = pd.bdate_range(wide.index.min(), wide.index.max(), name=wide.index.name)
    return wide.reindex(bidx).ffill()

def ew_rets_from_levels(
    levels: pd.DataFrame,
    baskets: Dict[str, List[str]],
    stale_days: int = 30,
) -> pd.DataFrame:
    if levels.empty:
        return pd.DataFrame()

    rets = levels.pct_change()
    out = {}
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
            if s.index.max() < last_idx - pd.Timedelta(days=int(stale_days)):
                continue
            cols.append(c_u)

        if not cols:
            continue

        if len(cols) > 1:
            out[b] = rets[cols].mean(axis=1, skipna=True)
        else:
            out[b] = rets[cols[0]]

    return pd.DataFrame(out).dropna(how="all")

# ---------------- Indicators used in rotation table ----------------
def macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig

def momentum_label(hist: pd.Series, lookback: int = 5, z_window: int = 63) -> str:
    h = hist.dropna()
    if h.shape[0] < max(lookback + 1, z_window):
        return "Neutral"

    latest = float(h.iloc[-1])
    ref = float(h.iloc[-(lookback + 1)])

    base = "Positive" if latest > 0 else ("Negative" if latest < 0 else "Neutral")
    if base == "Neutral":
        return "Neutral"

    window = h.iloc[-z_window:]
    std = float(window.std(ddof=0))
    z = (latest - float(window.mean())) / std if std and not np.isnan(std) else 0.0

    accel = "Accelerating" if (latest - ref) > 0 else "Decelerating"
    strength = "Strong" if abs(z) > 1 else "Weak"
    return f"{base} | {accel} | {strength}"

def pct_since(levels: pd.Series, start_ts: pd.Timestamp) -> float:
    sub = levels[levels.index >= start_ts]
    if sub.shape[0] < 2:
        return np.nan
    return float((sub.iloc[-1] / sub.iloc[0]) - 1.0)

def _safe_round(x: float, nd: int = 2):
    if pd.isna(x):
        return None
    return round(float(x), nd)

def _ret_color(x):
    if x is None:
        return "white"
    v = float(x)
    if v >= 0:
        s = min(abs(v) / 20.0, 1.0)
        g = int(255 - 90 * s)
        return f"rgb({int(240-120*s)},{g},{int(240-120*s)})"
    s = min(abs(v) / 20.0, 1.0)
    r = int(255 - 90 * s)
    return f"rgb({r},{int(240-120*s)},{int(240-120*s)})"

def _macd_color(tag: Optional[str]):
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

def _corr_color(x):
    if x is None:
        return "white"
    v = abs(float(x))
    if v >= 0.8:
        return "rgb(210,230,255)"
    if v >= 0.5:
        return "rgb(220,235,255)"
    return "rgb(230,240,255)"

def rolling_corr_last(a: pd.Series, b: pd.Series, window: int) -> float:
    merged = pd.concat([a, b], axis=1, join="inner").dropna()
    if merged.shape[0] < max(30, int(window * 0.7)):
        return np.nan
    rc = merged.iloc[:, 0].rolling(int(window)).corr(merged.iloc[:, 1])
    return float(rc.iloc[-1]) if rc.dropna().shape[0] else np.nan

def beta_last(a: pd.Series, b: pd.Series, window: int) -> float:
    # beta of a vs b, on last window points
    merged = pd.concat([a, b], axis=1, join="inner").dropna()
    if merged.shape[0] < max(30, int(window * 0.7)):
        return np.nan
    sub = merged.iloc[-int(window):]
    x = sub.iloc[:, 1].values  # factor
    y = sub.iloc[:, 0].values  # basket
    vx = np.var(x)
    if vx == 0 or np.isnan(vx):
        return np.nan
    cov = np.mean((x - np.mean(x)) * (y - np.mean(y)))
    return float(cov / vx)

def build_rotation_panel_df(
    basket_returns: pd.DataFrame,
    ref_start: pd.Timestamp,
    dynamic_label: str,
    factor_ret: pd.Series,
    corr_window: int,
) -> pd.DataFrame:
    if basket_returns.empty:
        cols = ["Basket", "%5D", "%1M", f"↓ %{dynamic_label}", "MACD Momentum", "Corr"]
        return pd.DataFrame(columns=cols).set_index("Basket")

    levels_100 = 100 * (1 + basket_returns).cumprod()
    rows = []
    dyn_col = f"↓ %{dynamic_label}"

    for b in levels_100.columns:
        s = levels_100[b].dropna()
        if s.shape[0] < 30:
            continue

        r5d = (s.iloc[-1] / s.iloc[-6]) - 1.0 if s.shape[0] > 6 else np.nan
        # 1M approx: use last 21 business days if available
        r1m = (s.iloc[-1] / s.iloc[-22]) - 1.0 if s.shape[0] > 22 else np.nan

        # dynamic since ref_start (backfill to next available bday)
        start_idx = s.index[s.index.get_indexer([pd.Timestamp(ref_start)], method="backfill")]
        r_dyn = pct_since(s, start_idx[0]) if len(start_idx) and start_idx[0] in s.index else np.nan

        hist = macd_hist(s, 12, 26, 9)
        macd_m = momentum_label(hist, lookback=5, z_window=max(63, corr_window))

        corr_val = rolling_corr_last(basket_returns[b], factor_ret, corr_window)

        rows.append({
            "Basket": b,
            "%5D": _safe_round(r5d * 100.0, 1),
            "%1M": _safe_round(r1m * 100.0, 1),
            dyn_col: _safe_round(r_dyn * 100.0, 1),
            "MACD Momentum": macd_m,
            "Corr": _safe_round(corr_val, 2),
        })

    if not rows:
        cols = ["Basket", "%5D", "%1M", f"↓ %{dynamic_label}", "MACD Momentum", "Corr"]
        return pd.DataFrame(columns=cols).set_index("Basket")

    df = pd.DataFrame(rows).set_index("Basket")
    if dyn_col in df.columns:
        df = df.sort_values(by=dyn_col, ascending=False)
    return df

def plot_rotation_table(panel_df: pd.DataFrame, title: str):
    st.markdown(f"**{title}**")
    if panel_df.empty:
        st.info("No baskets passed the data quality checks for this window.")
        return

    dyn_col = [c for c in panel_df.columns if c.startswith("↓ %")]
    dyn_col = dyn_col[0] if dyn_col else None

    headers = ["Basket", "%5D", "%1M", dyn_col, "MACD Momentum", "Corr"]
    values = [panel_df.index.tolist()]

    fill_colors = [["white"] * len(panel_df)]

    for col in ["%5D", "%1M", dyn_col]:
        vals = panel_df[col].tolist()
        values.append(vals)
        fill_colors.append([_ret_color(v) for v in vals])

    macd_vals = panel_df["MACD Momentum"].tolist()
    values.append(macd_vals)
    fill_colors.append([_macd_color(v) for v in macd_vals])

    corr_vals = panel_df["Corr"].tolist()
    values.append(corr_vals)
    fill_colors.append([_corr_color(v) for v in corr_vals])

    # widths tuned to look clean with fewer columns
    col_widths = [0.42, 0.10, 0.10, 0.14, 0.18, 0.06]

    fig_tbl = go.Figure(
        data=[
            go.Table(
                columnwidth=[int(w * 1000) for w in col_widths],
                header=dict(
                    values=headers,
                    fill_color="white",
                    line_color="rgb(230,230,230)",
                    font=dict(color="black", size=13),
                    align="left",
                    height=34,
                ),
                cells=dict(
                    values=values,
                    fill_color=fill_colors,
                    line_color="rgb(240,240,240)",
                    font=dict(color="black", size=12),
                    align="left",
                    height=28,
                    format=[None, ".1f", ".1f", ".1f", None, ".2f"],
                ),
            )
        ]
    )
    fig_tbl.update_layout(
        margin=dict(l=0, r=0, t=6, b=0),
        height=min(900, 74 + 28 * max(3, len(panel_df))),
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
        fig.add_trace(
            go.Scatter(
                x=cum_pct.index,
                y=cum_pct[b],
                mode="lines",
                line=dict(width=2, color=PASTELS[i % len(PASTELS)]),
                name=b,
                hovertemplate=f"{b}<br>% Cum: %{{y:.1f}}%<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=bm_cum.index,
            y=bm_cum.values,
            mode="lines",
            line=dict(width=2, dash="dash", color="#888"),
            name="SPY",
            hovertemplate="SPY<br>% Cum: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        showlegend=True,
        hovermode="x unified",
        yaxis_title="Cumulative return, %",
        title=dict(text=title, x=0, xanchor="left", y=0.95),
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor", showgrid=True),
        yaxis=dict(zeroline=False, showgrid=True),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Sidebar ----------------
st.title("Factor Momentum and Basket Rotation")

WINDOW_CHOICES = ["1W", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "10Y"]

def window_to_ref_start(choice: str, today_: date) -> date:
    if choice == "YTD":
        return date(today_.year, 1, 1)
    cal_days = {
        "1W": 10,
        "1M": 45,
        "3M": 140,
        "6M": 280,
        "1Y": 430,
        "3Y": 365 * 3 + 220,
        "5Y": 365 * 5 + 300,
        "10Y": 365 * 10 + 500,
    }
    return today_ - timedelta(days=int(cal_days[choice]))

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Tracks factor leadership (ETF pairs) and maps your ADFM baskets onto each factor as
        Pro vs Anti exposures using rolling beta sign and rolling correlation.
        """
    )
    st.divider()
    st.header("Settings")

    window_choice = st.selectbox("Analysis window", WINDOW_CHOICES, index=4)  # YTD default
    corr_window = st.slider("Factor mapping window (days)", 21, 252, 63)
    max_per_side = st.slider("Max baskets per side", 6, 25, 12)
    stale_days = st.slider("Stale data cutoff (days)", 7, 60, 30)

    # optional basket universe filter
    category_list = list(CATEGORIES.keys())
    selected_categories = st.multiselect(
        "Basket categories to include",
        options=category_list,
        default=category_list,
    )
    st.caption("Data source: Yahoo Finance. Internal use only.")

# ---------------- Build basket subset ----------------
if not selected_categories:
    st.error("No basket categories selected.")
    st.stop()

BASKETS_ACTIVE: Dict[str, List[str]] = {}
for cat in selected_categories:
    BASKETS_ACTIVE.update(CATEGORIES.get(cat, {}))

if not BASKETS_ACTIVE:
    st.error("No baskets found for the selected categories.")
    st.stop()

# ---------------- Determine dates and tickers ----------------
today = date.today()
ref_start_date = window_to_ref_start(window_choice, today)
ref_start_ts = pd.Timestamp(ref_start_date)

# download start includes buffer for MACD and correlation stability
download_start = pd.Timestamp(ref_start_date - timedelta(days=420))
download_end = pd.Timestamp(today) + pd.Timedelta(days=1)

bench = "SPY"

factor_tickers = sorted({t for pair in FACTOR_ETFS.values() for t in pair if t is not None})
basket_tickers = sorted({str(t).upper() for tks in BASKETS_ACTIVE.values() for t in tks})
need = sorted(set([bench] + factor_tickers + basket_tickers))

# ---------------- Fetch levels ----------------
levels = fetch_daily_levels(need, start=download_start, end=download_end, chunk_size=40)
if levels.empty:
    st.error("No price data returned. Yahoo Finance request returned empty.")
    st.stop()

# ensure benchmark exists
if bench not in levels.columns or levels[bench].dropna().empty:
    st.error("SPY data missing or empty.")
    st.stop()

bench_rets = levels[bench].pct_change().dropna()

# ---------------- Construct factor series ----------------
factor_levels_full = {}
for name, (up, down) in FACTOR_ETFS.items():
    if down is None:
        if up in levels.columns:
            factor_levels_full[name] = levels[up].dropna()
        continue
    if up in levels.columns and down in levels.columns:
        series = rs(levels[up], levels[down])
        if not series.empty:
            factor_levels_full[name] = series

factor_df_full = pd.DataFrame(factor_levels_full).dropna(how="all")
if factor_df_full.empty:
    st.error("No factor series could be constructed from available data.")
    st.stop()

factor_df = factor_df_full[factor_df_full.index >= ref_start_ts].copy()
if factor_df.empty:
    # fall back to last 60 trading days if window is too tight or missing
    factor_df = factor_df_full.iloc[-60:].copy()

# ---------------- Momentum snapshot data ----------------
rows = []
lookback_short = 20
lookback_long = 60

for f in factor_df.columns:
    s = factor_df[f].dropna()
    if len(s) < 10:
        continue
    r5 = pct_change_window(s, 5)
    r_short = pct_change_window(s, lookback_short)
    r_long = pct_change_window(s, lookback_long)
    mom_val = momentum(s, win=lookback_short)
    tclass = trend_class(s)
    infl = inflection(r_short, r_long)
    rows.append([f, r5, r_short, r_long, mom_val, tclass, infl])

mom_df = pd.DataFrame(rows, columns=["Factor", "%5D", "Short", "Long", "Momentum", "Trend", "Inflection"]).set_index("Factor")
if mom_df.empty:
    st.error("No factors passed data checks for this window.")
    st.stop()

mom_df = mom_df.sort_values("Short", ascending=False)

# ---------------- Breadth & regime + correlation for commentary ----------------
trend_counts = mom_df["Trend"].value_counts()
num_up = int(trend_counts.get("Up", 0))
breadth = num_up / len(mom_df) * 100.0

raw_score = (
    0.4 * mom_df["Short"].mean()
    + 0.3 * ((mom_df["Inflection"] == "Turning Up").mean() - (mom_df["Inflection"] == "Turning Down").mean())
    + 0.3 * ((mom_df["Trend"] == "Up").mean() - (mom_df["Trend"] == "Down").mean())
)
regime_score = max(0.0, min(100.0, 50.0 + 50.0 * (raw_score / 5.0)))

corr_matrix = factor_df.pct_change().dropna(how="all").corr()

# ---------------- Factor tape summary ----------------
st.subheader(f"Factor Tape Summary ({window_choice})")
summary_html = build_commentary(mom_df, breadth, regime_score, corr_matrix)
card_box(summary_html)

# ---------------- Factor time series ----------------
st.subheader(f"Factor Time Series ({window_choice})")

n_factors = len(factor_df.columns)
ncols = 3
nrows = int(np.ceil(n_factors / ncols))

fig_ts, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), squeeze=False)
axes = axes.ravel()

if len(factor_df.index) > 1:
    span_days = (factor_df.index[-1] - factor_df.index[0]).days
else:
    span_days = 0

if span_days <= 370:
    locator = mdates.MonthLocator()
    formatter = mdates.DateFormatter("%b")
else:
    locator = mdates.YearLocator()
    formatter = mdates.DateFormatter("%Y")

for i, f in enumerate(factor_df.columns):
    ax = axes[i]
    s = factor_df[f].dropna()
    ax.plot(s.index, s.values, color=PASTELS[i % len(PASTELS)], linewidth=2)
    ax.set_title(f, color=TEXT)
    ax.grid(color=GRID, linewidth=0.5)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_fontsize(8)

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

fig_ts.tight_layout()
st.pyplot(fig_ts, clear_figure=True)

# ---------------- Factor momentum snapshot (table) ----------------
st.subheader("Factor Momentum Snapshot")

display_df = mom_df.copy()
for col in ["%5D", "Short", "Long", "Momentum"]:
    display_df[col] = display_df[col] * 100.0

display_df = display_df[["%5D", "Short", "Long", "Momentum", "Trend", "Inflection"]]

st.dataframe(
    display_df.style.format(
        {
            "%5D": "{:.1f}%",
            "Short": "{:.1f}%",
            "Long": "{:.1f}%",
            "Momentum": "{:.2f}%",
        }
    ),
    use_container_width=True,
)

# ---------------- Leadership map (Short vs Long scatter) ----------------
st.subheader("Leadership Map (Short vs Long Momentum)")

fig_lead, ax_lead = plt.subplots(figsize=(8, 6))

short_vals = mom_df["Short"] * 100.0
long_vals = mom_df["Long"] * 100.0

x_max = max(abs(short_vals.min()), abs(short_vals.max()))
y_max = max(abs(long_vals.min()), abs(long_vals.max()))
pad_x = x_max * 0.15 if x_max > 0 else 1.0
pad_y = y_max * 0.15 if y_max > 0 else 1.0

ax_lead.set_xlim(-x_max - pad_x, x_max + pad_x)
ax_lead.set_ylim(-y_max - pad_y, y_max + pad_y)

x_min, x_max_lim = ax_lead.get_xlim()
y_min, y_max_lim = ax_lead.get_ylim()

ax_lead.fill_between([0, x_max_lim], 0, y_max_lim, color="#e1f5e0", alpha=0.55)
ax_lead.fill_between([x_min, 0], 0, y_max_lim, color="#fff9c4", alpha=0.55)
ax_lead.fill_between([x_min, 0], y_min, 0, color="#fde0dc", alpha=0.55)
ax_lead.fill_between([0, x_max_lim], y_min, 0, color="#ffe9b3", alpha=0.55)

ax_lead.axvline(0, color="#888888", linewidth=1)
ax_lead.axhline(0, color="#888888", linewidth=1)

ax_lead.text(x_max_lim * 0.65, y_max_lim * 0.75, "Short ↑ / Long ↑\nEstablished leaders", fontsize=9, ha="center", va="center", color="#333333")
ax_lead.text(x_min * 0.65, y_max_lim * 0.75, "Short ↓ / Long ↑\nMean reversion", fontsize=9, ha="center", va="center", color="#333333")
ax_lead.text(x_min * 0.65, y_min * 0.75, "Short ↓ / Long ↓\nPersistent laggards", fontsize=9, ha="center", va="center", color="#333333")
ax_lead.text(x_max_lim * 0.65, y_min * 0.75, "Short ↑ / Long ↓\nNew rotations", fontsize=9, ha="center", va="center", color="#333333")

for i, factor in enumerate(mom_df.index):
    x = float(short_vals.loc[factor])
    y = float(long_vals.loc[factor])
    ax_lead.scatter(x, y, s=70, color=PASTELS[i % len(PASTELS)], edgecolor="#444444", linewidth=0.6, zorder=3)
    ax_lead.annotate(factor, xy=(x, y), xytext=(4, 3), textcoords="offset points", fontsize=9, va="center", color="#111111")

ax_lead.set_xlabel("Short window return %", color=TEXT)
ax_lead.set_ylabel("Long window return %", color=TEXT)
ax_lead.set_title("Factors by Short vs Long Momentum", color=TEXT, pad=10)
ax_lead.grid(color=GRID, linewidth=0.6, alpha=0.6)
fig_lead.tight_layout()
st.pyplot(fig_lead, clear_figure=True)

# ---------------- Basket returns (engine wired) ----------------
st.subheader("Factor Rotation Using ADFM Baskets")

all_basket_rets = ew_rets_from_levels(levels, BASKETS_ACTIVE, stale_days=int(stale_days))
if all_basket_rets.empty:
    # show a useful debug message instead of a fake placeholder
    missing = []
    for b, tks in list(BASKETS_ACTIVE.items())[:10]:
        present = [str(t).upper() for t in tks if str(t).upper() in levels.columns]
        if len(present) < 1:
            missing.append(b)
    st.error(
        "Basket return series failed to build from current price data. "
        "This usually means Yahoo did not return enough constituent prices or your stale filter removed everything."
    )
    if missing:
        st.caption(f"Examples of baskets with zero constituents present in the download: {', '.join(missing[:10])}")
    st.stop()

# factor returns for mapping (use full range)
factor_rets_full = factor_df_full.pct_change().dropna(how="all")

def split_baskets_for_factor(
    baskets_ret: pd.DataFrame,
    factor_ret: pd.Series,
    window: int,
    top_n: int
) -> Tuple[List[str], List[str], pd.DataFrame]:
    stats = []
    for b in baskets_ret.columns:
        s = baskets_ret[b].dropna()
        if s.shape[0] < max(40, int(window * 0.7)):
            continue
        c = rolling_corr_last(s, factor_ret, window)
        bt = beta_last(s, factor_ret, window)
        if pd.isna(c) or pd.isna(bt):
            continue
        stats.append({"Basket": b, "Corr": float(c), "Beta": float(bt), "AbsCorr": abs(float(c))})

    if not stats:
        return [], [], pd.DataFrame()

    stats_df = pd.DataFrame(stats).set_index("Basket").sort_values("AbsCorr", ascending=False)

    pro = stats_df[stats_df["Beta"] >= 0].head(max(2 * top_n, top_n)).index.tolist()
    anti = stats_df[stats_df["Beta"] < 0].head(max(2 * top_n, top_n)).index.tolist()

    # trim to top_n after we later sort by performance in the panel df
    return pro, anti, stats_df

# render all factor sections fully expanded (no expanders)
for factor_name in FACTOR_ETFS.keys():
    if factor_name not in factor_rets_full.columns:
        continue

    st.markdown(f"### {factor_name}")

    f_ret = factor_rets_full[factor_name].dropna()
    if f_ret.empty:
        st.info("Factor return series missing for this factor on the selected date range.")
        continue

    pro_baskets, anti_baskets, stats_df = split_baskets_for_factor(
        all_basket_rets, f_ret, window=int(corr_window), top_n=int(max_per_side)
    )

    if not pro_baskets and not anti_baskets:
        st.info("No baskets had sufficient overlap to compute beta/corr on this factor window.")
        continue

    # Build panel dfs, then clip to top_n on each side after sorting by dynamic return
    dyn_label = window_choice

    pro_df = build_rotation_panel_df(
        all_basket_rets[pro_baskets].dropna(how="all") if pro_baskets else pd.DataFrame(),
        ref_start=ref_start_ts,
        dynamic_label=dyn_label,
        factor_ret=f_ret,
        corr_window=int(corr_window),
    )
    anti_df = build_rotation_panel_df(
        all_basket_rets[anti_baskets].dropna(how="all") if anti_baskets else pd.DataFrame(),
        ref_start=ref_start_ts,
        dynamic_label=dyn_label,
        factor_ret=f_ret,
        corr_window=int(corr_window),
    )

    if not pro_df.empty:
        pro_df = pro_df.head(int(max_per_side))
    if not anti_df.empty:
        anti_df = anti_df.head(int(max_per_side))

    # tables side by side
    c1, c2 = st.columns(2, gap="large")
    with c1:
        plot_rotation_table(pro_df, "Pro side baskets")
    with c2:
        plot_rotation_table(anti_df, "Anti side baskets")

    # charts side by side
    c3, c4 = st.columns(2, gap="large")
    with c3:
        plot_cumulative_chart(
            all_basket_rets[pro_df.index].dropna(how="all") if not pro_df.empty else pd.DataFrame(),
            title=f"{factor_name} (Pro) vs SPY",
            benchmark_series=bench_rets,
        )
    with c4:
        plot_cumulative_chart(
            all_basket_rets[anti_df.index].dropna(how="all") if not anti_df.empty else pd.DataFrame(),
            title=f"{factor_name} (Anti) vs SPY",
            benchmark_series=bench_rets,
        )

    st.divider()

st.caption("© 2026 AD Fund Management LP")
