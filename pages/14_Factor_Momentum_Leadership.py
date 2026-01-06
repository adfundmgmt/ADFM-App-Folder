import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from datetime import datetime, date

# ---------------- Config ----------------
st.set_page_config(page_title="Factor Momentum and Leadership", layout="wide")
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

# ---------------- Helpers (Factor tool) ----------------
def load_prices(tickers, start):
    data = yf.download(tickers, start=start, progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    return data.dropna(how="all")

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

def rs(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return aligned.iloc[:, 0] / aligned.iloc[:, 1]

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

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

def build_commentary(mom_df: pd.DataFrame, breadth: float, regime_score: float) -> str:
    trend_counts = mom_df["Trend"].value_counts()
    up_count = int(trend_counts.get("Up", 0))
    down_count = int(trend_counts.get("Down", 0))

    established_leaders = mom_df[
        (mom_df["Short"] > 0) & (mom_df["Long"] > 0)
    ].sort_values("Short", ascending=False).index.tolist()

    new_rotations = mom_df[mom_df["Inflection"] == "Turning Up"].index.tolist()
    fading_leaders = mom_df[mom_df["Inflection"] == "Turning Down"].index.tolist()

    leaders_text = ", ".join(established_leaders[:4]) if established_leaders else "no factor pair in a clean dual-horizon uptrend"
    rotations_text = ", ".join(new_rotations[:4]) if new_rotations else "no factor is clearly turning up yet"
    fading_text = ", ".join(fading_leaders[:4]) if fading_leaders else "no obvious factor is rolling over from strength"

    breadth_desc = bucket_breadth(breadth)
    regime_desc = bucket_regime(regime_score)

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
            f"Inflection signals flag {', '.join(new_rotations)} as turning up from weaker long-term trends."
        )
    if fading_leaders:
        drivers.append(
            f"On the other side, {', '.join(fading_leaders)} are turning down against still-positive long windows."
        )
    drivers.append("Basket rotation diagnostics replace correlation crowding and sit below.")

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
ALL_TICKERS = sorted({t for pair in FACTOR_ETFS.values() for t in pair if t is not None})

WINDOW_MAP_DAYS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
    "3Y": 252 * 3,
    "5Y": 252 * 5,
    "10Y": 252 * 10,
}

# ---------------- Sidebar ----------------
st.title("Factor Momentum and Leadership Dashboard")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Tracks style and macro factor leadership using ETF pairs and scores each factor on
        short and long momentum, trend structure, and inflection. Basket rotation diagnostics
        map factors into your execution baskets to show capital flow at the book level.
        """
    )
    st.divider()
    st.header("Settings")
    history_start = st.date_input("History start", datetime(2015, 1, 1))
    window_choice = st.selectbox(
        "Analysis window",
        ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "10Y"],
        index=3,
    )
    lookback_short = st.slider("Short momentum window (days)", 10, 60, 20)
    lookback_long = st.slider("Long momentum window (days)", 30, 180, 60)

    st.divider()
    st.header("Basket Rotation Settings")
    basket_universe = st.selectbox(
        "Rotation basket universe",
        ["Mapped baskets only (faster)", "All baskets (heavier)"],
        index=0,
    )
    stale_days = st.slider("Stale data cutoff (days)", 10, 90, 30)
    apply_mktcap_filter = st.checkbox("Apply market cap filter (≥ $1B)", value=False)
    min_mktcap = 1_000_000_000

    st.caption("Data source: Yahoo Finance. Internal use only.")

# ---------------- Load factor data ----------------
prices_full = load_prices(ALL_TICKERS, start=str(history_start))
if prices_full.empty:
    st.error("No data returned.")
    st.stop()

# ---------------- Build factor series (full history) ----------------
factor_levels_full = {}
for name, (up, down) in FACTOR_ETFS.items():
    if down is None:
        if up in prices_full:
            factor_levels_full[name] = prices_full[up]
        continue
    if up in prices_full and down in prices_full:
        factor_levels_full[name] = rs(prices_full[up], prices_full[down])

factor_df_full = pd.DataFrame(factor_levels_full).dropna(how="all")
if factor_df_full.empty:
    st.error("No factor series could be constructed.")
    st.stop()

# ---------------- Window selection ----------------
if window_choice == "YTD":
    current_year_start = date(datetime.now().year, 1, 1)
    factor_df = factor_df_full[factor_df_full.index >= pd.to_datetime(current_year_start)]
else:
    days = WINDOW_MAP_DAYS[window_choice]
    factor_df = factor_df_full.copy() if len(factor_df_full) <= days else factor_df_full.iloc[-days:].copy()

if factor_df.empty:
    st.error("No data available for the selected window.")
    st.stop()

# ---------------- Momentum snapshot data ----------------
rows = []
for f in factor_df.columns:
    s = factor_df[f].dropna()
    if len(s) < 5:
        continue
    r5 = pct_change_window(s, 5)
    r_short = pct_change_window(s, lookback_short)
    r_long = pct_change_window(s, lookback_long)
    mom_val = momentum(s, win=lookback_short)
    tclass = trend_class(s)
    infl = inflection(r_short, r_long)
    rows.append([f, r5, r_short, r_long, mom_val, tclass, infl])

mom_df = pd.DataFrame(
    rows,
    columns=["Factor", "%5D", "Short", "Long", "Momentum", "Trend", "Inflection"],
).set_index("Factor")

if mom_df.empty:
    st.error("No factors passed data checks for this window.")
    st.stop()

mom_df = mom_df.sort_values("Short", ascending=False)

# ---------------- Breadth & regime ----------------
trend_counts = mom_df["Trend"].value_counts()
num_up = int(trend_counts.get("Up", 0))
breadth = num_up / len(mom_df) * 100.0

raw_score = (
    0.4 * mom_df["Short"].mean()
    + 0.3 * ((mom_df["Inflection"] == "Turning Up").mean() - (mom_df["Inflection"] == "Turning Down").mean())
    + 0.3 * ((mom_df["Trend"] == "Up").mean() - (mom_df["Trend"] == "Down").mean())
)
regime_score = max(0.0, min(100.0, 50.0 + 50.0 * (raw_score / 5.0)))

# ---------------- Factor tape summary ----------------
st.subheader(f"Factor Tape Summary ({window_choice})")
summary_html = build_commentary(mom_df, breadth, regime_score)
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
    x = short_vals.loc[factor]
    y = long_vals.loc[factor]
    ax_lead.scatter(
        x,
        y,
        s=70,
        color=PASTELS[i % len(PASTELS)],
        edgecolor="#444444",
        linewidth=0.6,
        zorder=3,
    )
    ax_lead.annotate(
        factor,
        xy=(x, y),
        xytext=(4, 3),
        textcoords="offset points",
        fontsize=9,
        va="center",
        color="#111111",
    )

ax_lead.set_xlabel("Short window return %", color=TEXT)
ax_lead.set_ylabel("Long window return %", color=TEXT)
ax_lead.set_title("Factors by Short vs Long Momentum", color=TEXT, pad=10)
ax_lead.grid(color=GRID, linewidth=0.6, alpha=0.6)
fig_lead.tight_layout()
st.pyplot(fig_lead, clear_figure=True)

# ======================================================================
# ================== REPLACEMENT: Basket-Driven Rotation Engine =========
# ======================================================================

# ---------------- Basket universe (your definitions) ----------------
PASTEL_PLOTLY = [
    "#AEC6CF","#FFB347","#B39EB5","#77DD77","#F49AC2",
    "#CFCFC4","#DEA5A4","#C6E2FF","#FFDAC1","#E2F0CB",
    "#C7CEEA","#FFB3BA","#FFD1DC","#B5EAD7","#E7E6F7",
    "#F1E3DD","#B0E0E6","#E0BBE4","#F3E5AB","#D5E8D4"
]

MARKET_CAP_EXCEPTIONS = set([
    # You can add exceptions here later if desired
])

CATEGORIES = {
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

# ---------------- Basket data helpers ----------------
def _chunk(lst, n):
    n = max(1, int(n))
    return [lst[i:i+n] for i in range(0, len(lst), n)]

@st.cache_data(show_spinner=False)
def fetch_daily_levels(tickers, start, end, chunk_size: int = 40) -> pd.DataFrame:
    uniq = sorted(list(set([str(t).upper() for t in tickers])))
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
def fetch_market_caps(tickers):
    uniq = sorted(list({str(t).upper() for t in tickers}))
    if not uniq:
        return {}
    caps = {}
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
                caps[str(sym).upper()] = float(mc_val)
        except Exception:
            continue
    return caps

def ew_rets_from_levels(levels, baskets, market_caps=None, min_market_cap=None, stale_days=30) -> pd.DataFrame:
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
        out[b] = rets[cols].mean(axis=1, skipna=True) if len(cols) > 1 else rets[cols[0]]
    return pd.DataFrame(out).dropna(how="all")

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    ag = gain.ewm(alpha=1/period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)
    rs_ = ag / al
    return 100 - (100 / (1 + rs_))

def macd_hist(series: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd_ = ema_f - ema_s
    sig_ = macd_.ewm(span=signal, adjust=False).mean()
    return macd_ - sig_

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

# ---- Table coloring (Plotly) ----
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

def build_panel_df(basket_returns: pd.DataFrame, ref_start: pd.Timestamp, dynamic_label: str, benchmark_series: pd.Series | None = None) -> pd.DataFrame:
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
    if benchmark_series is not None and not benchmark_series.dropna().empty:
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
        if benchmark_series is not None and not benchmark_series.dropna().empty:
            merged = pd.concat([basket_returns[b].dropna(), benchmark_series.dropna()], axis=1, join="inner").dropna()
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
            line=dict(width=2, color=PASTEL_PLOTLY[i % len(PASTEL_PLOTLY)]),
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

# ---------------- Factor -> Basket mapping (explicit) ----------------
# Basket names here must match keys in ALL_BASKETS exactly.
FACTOR_TO_BASKETNAMES = {
    "Growth vs Value": {
        "pro": [
            "Semis Compute and Accelerators",
            "AI Infrastructure Leaders",
            "Hyperscalers and Cloud",
            "Quality SaaS",
            "Cybersecurity",
            "Long-Duration Equities",
        ],
        "anti": [
            "Energy Majors",
            "Money-Center and IBs",
            "Homebuilders",
            "Industrials and Infrastructure",
            "Staples and Beverages",
            "Utilities Defensive",
        ],
    },
    "Quality vs Junk": {
        "pro": [
            "Net-Cash Compounders",
            "Short-Duration Cash Flow",
            "Insurance P&C",
            "Insurance Brokers",
            "Diagnostics and Tools",
            "Large-Cap Biotech",
        ],
        "anti": [
            "Equity Credit Stress Proxies",
            "Leveraged Cyclicals",
            "Financial Conditions Sensitive",
            "Retail Asset-Heavy Inventory Risk",
            "Consumer Credit Stress",
            "Rate-Sensitive Cyclicals",
        ],
    },
    "High Beta vs Low Vol": {
        "pro": [
            "Long-Duration Equities",
            "Leveraged Cyclicals",
            "Crypto Proxies",
            "China Tech ADRs",
            "Fintech and Neobanks",
            "Electric Vehicles",
        ],
        "anti": [
            "Utilities Defensive",
            "Yield Proxies",
            "Insurance P&C",
            "Staples and Beverages",
            "Retail Staples",
            "Short-Duration Cash Flow",
        ],
    },
    "Small vs Large": {
        "pro": [
            "Recreation and Experiences",
            "Deferred Durables and Home",
            "Trade-Down Retail and Off-Price",
            "Consumer Credit Stress",
            "Budget Hotels and Value Travel",
            "Regional Banks",
        ],
        "anti": [
            "Hyperscalers and Cloud",
            "AI Infrastructure Leaders",
            "Net-Cash Compounders",
            "Energy Majors",
            "Money-Center and IBs",
            "REITs Core",
        ],
    },
    "Cyclicals vs Defensives": {
        "pro": [
            "Restaurants",
            "Travel and Booking",
            "Hotels and Casinos",
            "Airlines",
            "Homebuilders",
            "US Shale and E&Ps",
        ],
        "anti": [
            "Retail Staples",
            "Staples and Beverages",
            "Utilities Defensive",
            "Telecom and Cable",
            "Healthcare Payers",
            "Large-Cap Biotech",
        ],
    },
    "Tech vs Broad": {
        "pro": [
            "Semis ETFs",
            "Semis Equipment",
            "AI Infrastructure Leaders",
            "Hyperscalers and Cloud",
            "Cybersecurity",
            "Data Center Networking",
        ],
        "anti": "BENCH:SPY",
    },
    "US vs World": {
        "pro": [
            "Net-Cash Compounders",
            "Hyperscalers and Cloud",
            "Semis Compute and Accelerators",
            "Cybersecurity",
        ],
        "anti": [
            "China Tech ADRs",
            "EM Domestic Demand",
            "Commodity FX Equities",
            "Dollar-Down Beneficiaries",
        ],
    },
    "Momentum": {
        "pro": "TOP_QUARTILE",
        "anti": "BOTTOM_QUARTILE",
    },
}

def compound_return_from_rets(rets: pd.Series, days: int) -> float:
    r = rets.dropna()
    if r.shape[0] < 2:
        return np.nan
    days = int(min(days, r.shape[0] - 1))
    sub = r.iloc[-days:]
    return float((1.0 + sub).prod() - 1.0)

# ---------------- Build basket universe for rotation ----------------
st.subheader("Basket-Driven Factor Rotation")

window_start = factor_df.index.min()
window_end = factor_df.index.max()

# Fetch some extra history for RSI/MACD, without pulling 10+ years unless needed
extra_days = 450
basket_fetch_start = (window_start - pd.Timedelta(days=extra_days)).date()
basket_fetch_end = (window_end + pd.Timedelta(days=2)).date()  # buffer

# Universe selection
mapped_baskets = set()
for v in FACTOR_TO_BASKETNAMES.values():
    if isinstance(v.get("pro"), list):
        mapped_baskets.update(v["pro"])
    if isinstance(v.get("anti"), list):
        mapped_baskets.update(v["anti"])

if basket_universe.startswith("Mapped"):
    universe_baskets = {bk: ALL_BASKETS[bk] for bk in sorted(mapped_baskets) if bk in ALL_BASKETS}
else:
    universe_baskets = ALL_BASKETS.copy()

# Need tickers = all constituents + SPY
bench = "SPY"
need = {bench}
for tks in universe_baskets.values():
    need.update([str(t).upper() for t in tks])

levels = fetch_daily_levels(
    sorted(list(need)),
    start=pd.to_datetime(basket_fetch_start),
    end=pd.to_datetime(basket_fetch_end) + pd.Timedelta(days=1)
)

if levels.empty or bench not in levels.columns or levels[bench].dropna().empty:
    st.error("Basket engine could not load data. If you are on Streamlit Cloud, this can be Yahoo throttling.")
    st.stop()

market_caps = None
if apply_mktcap_filter:
    market_caps = fetch_market_caps(list(levels.columns))

basket_rets_all = ew_rets_from_levels(
    levels,
    universe_baskets,
    market_caps=market_caps,
    min_market_cap=min_mktcap if apply_mktcap_filter else None,
    stale_days=stale_days
)

if basket_rets_all.empty:
    st.error("No basket returns could be constructed with current filters. Turn off market cap filter or widen universe.")
    st.stop()

# Restrict to factor window dates for rotation calculations, but keep full for indicators
basket_rets_win = basket_rets_all[(basket_rets_all.index >= window_start) & (basket_rets_all.index <= window_end)].copy()
bench_rets_full = levels[bench].pct_change().dropna()
bench_rets_win = bench_rets_full[(bench_rets_full.index >= window_start) & (bench_rets_full.index <= window_end)].copy()

if basket_rets_win.empty or bench_rets_win.empty:
    st.error("No overlapping window data between baskets and benchmark.")
    st.stop()

# Basket momentum table (for momentum factor and for diagnostics)
basket_mom = pd.DataFrame(index=basket_rets_win.columns)
basket_mom["Short"] = [compound_return_from_rets(basket_rets_win[c], lookback_short) for c in basket_rets_win.columns]
basket_mom["Long"] = [compound_return_from_rets(basket_rets_win[c], lookback_long) for c in basket_rets_win.columns]
basket_mom = basket_mom.dropna()

if basket_mom.empty:
    st.error("Basket momentum table is empty for this window.")
    st.stop()

basket_mom["Rank"] = basket_mom["Short"].rank(ascending=False, method="average")
q25 = basket_mom["Rank"].quantile(0.25)
q75 = basket_mom["Rank"].quantile(0.75)
top_quart = basket_mom[basket_mom["Rank"] <= q25].index.tolist()
bot_quart = basket_mom[basket_mom["Rank"] >= q75].index.tolist()

def resolve_side(side):
    if side == "TOP_QUARTILE":
        return top_quart
    if side == "BOTTOM_QUARTILE":
        return bot_quart
    if isinstance(side, str) and side.startswith("BENCH:"):
        return side
    if isinstance(side, list):
        return [b for b in side if b in basket_rets_win.columns]
    return []

def side_series(side):
    if isinstance(side, str) and side.startswith("BENCH:"):
        return bench_rets_win.rename(bench)
    cols = resolve_side(side)
    cols = [c for c in cols if c in basket_rets_win.columns]
    if not cols:
        return pd.Series(dtype=float)
    return basket_rets_win[cols].mean(axis=1, skipna=True).rename("EW")

rotation_rows = []
factor_panels = {}

for factor_name, mapping in FACTOR_TO_BASKETNAMES.items():
    pro = mapping.get("pro")
    anti = mapping.get("anti")

    pro_cols = resolve_side(pro)
    anti_cols = resolve_side(anti)

    pro_rets = side_series(pro)
    anti_rets = side_series(anti)

    if pro_rets.empty or anti_rets.empty:
        continue

    pro_s = compound_return_from_rets(pro_rets, lookback_short)
    pro_l = compound_return_from_rets(pro_rets, lookback_long)
    anti_s = compound_return_from_rets(anti_rets, lookback_short)
    anti_l = compound_return_from_rets(anti_rets, lookback_long)

    spread_s = pro_s - anti_s
    spread_l = pro_l - anti_l
    accel = spread_s - spread_l

    # Breadth and dispersion inside the pro side (if it's baskets, not BENCH)
    pro_breadth = np.nan
    pro_disp = np.nan
    if isinstance(pro_cols, list) and len(pro_cols) >= 2:
        pro_breadth = float((basket_mom.loc[pro_cols, "Short"] > 0).mean() * 100.0)
        pro_disp = float(basket_mom.loc[pro_cols, "Short"].std(ddof=0) * 100.0)

    anti_breadth = np.nan
    if isinstance(anti_cols, list) and len(anti_cols) >= 2:
        anti_breadth = float((basket_mom.loc[anti_cols, "Short"] > 0).mean() * 100.0)

    rotation_rows.append({
        "Factor": factor_name,
        "Pro (Short)": pro_s * 100.0,
        "Anti (Short)": anti_s * 100.0,
        "Spread (Short)": spread_s * 100.0,
        "Pro (Long)": pro_l * 100.0,
        "Anti (Long)": anti_l * 100.0,
        "Spread (Long)": spread_l * 100.0,
        "Acceleration": accel * 100.0,
        "Pro breadth %": pro_breadth,
        "Pro dispersion %": pro_disp,
        "Anti breadth %": anti_breadth,
    })

    # Store panels for drilldown
    factor_panels[factor_name] = {
        "pro_cols": pro_cols if isinstance(pro_cols, list) else [],
        "anti_cols": anti_cols if isinstance(anti_cols, list) else [],
        "anti_is_bench": isinstance(anti_cols, str) and anti_cols.startswith("BENCH:"),
    }

rot_df = pd.DataFrame(rotation_rows).set_index("Factor")
if rot_df.empty:
    st.error("Rotation table is empty. The mapping likely does not overlap with your current basket universe filters.")
    st.stop()

rot_df = rot_df.sort_values("Acceleration", ascending=False)

st.markdown("### Rotation Scoreboard")
st.dataframe(
    rot_df.style.format(
        {
            "Pro (Short)": "{:.1f}%",
            "Anti (Short)": "{:.1f}%",
            "Spread (Short)": "{:.1f}%",
            "Pro (Long)": "{:.1f}%",
            "Anti (Long)": "{:.1f}%",
            "Spread (Long)": "{:.1f}%",
            "Acceleration": "{:.1f}%",
            "Pro breadth %": "{:.0f}%",
            "Pro dispersion %": "{:.1f}%",
            "Anti breadth %": "{:.0f}%",
        }
    ),
    use_container_width=True,
)

leaders = rot_df[rot_df["Acceleration"] > 0].index.tolist()
laggards = rot_df[rot_df["Acceleration"] < 0].index.tolist()

rot_line = (
    f"Rotation is accelerating into {', '.join(leaders[:3]) if leaders else 'no clear factor'}, "
    f"while it is decelerating most in {', '.join(laggards[:3]) if laggards else 'no clear factor'}."
)
card_box(f"<div style='font-weight:700; margin-bottom:6px;'>Rotation read</div><div>{rot_line}</div>")

st.markdown("### Factor Drilldown Panels")

# Dynamic label consistent with your basket tool style, but tied to window_choice
DYNAMIC_LABEL = window_choice

for factor_name in rot_df.index.tolist():
    meta = factor_panels.get(factor_name, {})
    pro_cols = meta.get("pro_cols", [])
    anti_cols = meta.get("anti_cols", [])
    anti_is_bench = meta.get("anti_is_bench", False)

    with st.expander(f"{factor_name}"):
        left, right = st.columns(2)

        with left:
            st.markdown("**Pro side baskets**")
            if pro_cols:
                pro_rets_df = basket_rets_all[pro_cols].dropna(how="all")
                pro_rets_df_win = pro_rets_df[(pro_rets_df.index >= window_start) & (pro_rets_df.index <= window_end)]
                pro_panel = build_panel_df(
                    pro_rets_df_win,
                    ref_start=pd.Timestamp(window_start.date()),
                    dynamic_label=DYNAMIC_LABEL,
                    benchmark_series=bench_rets_win
                )
                plot_panel_table(pro_panel)

                st.markdown("**Pro side cumulative vs SPY**")
                plot_cumulative_chart(
                    pro_rets_df_win[pro_panel.index] if not pro_panel.empty else pro_rets_df_win,
                    title=f"{factor_name} (Pro) vs SPY",
                    benchmark_series=bench_rets_win
                )
            else:
                st.info("No pro baskets available for this factor under current universe/filters.")

        with right:
            st.markdown("**Anti side baskets**")
            if anti_is_bench:
                anti_series = bench_rets_win.rename("SPY").to_frame()
                anti_panel = build_panel_df(
                    anti_series,
                    ref_start=pd.Timestamp(window_start.date()),
                    dynamic_label=DYNAMIC_LABEL,
                    benchmark_series=bench_rets_win
                )
                plot_panel_table(anti_panel)

                st.markdown("**Anti side cumulative vs SPY**")
                plot_cumulative_chart(
                    anti_series,
                    title=f"{factor_name} (Anti = SPY) vs SPY",
                    benchmark_series=bench_rets_win
                )
            elif anti_cols:
                anti_rets_df = basket_rets_all[anti_cols].dropna(how="all")
                anti_rets_df_win = anti_rets_df[(anti_rets_df.index >= window_start) & (anti_rets_df.index <= window_end)]
                anti_panel = build_panel_df(
                    anti_rets_df_win,
                    ref_start=pd.Timestamp(window_start.date()),
                    dynamic_label=DYNAMIC_LABEL,
                    benchmark_series=bench_rets_win
                )
                plot_panel_table(anti_panel)

                st.markdown("**Anti side cumulative vs SPY**")
                plot_cumulative_chart(
                    anti_rets_df_win[anti_panel.index] if not anti_panel.empty else anti_rets_df_win,
                    title=f"{factor_name} (Anti) vs SPY",
                    benchmark_series=bench_rets_win
                )
            else:
                st.info("No anti baskets available for this factor under current universe/filters.")

st.caption("© 2026 AD Fund Management LP")
