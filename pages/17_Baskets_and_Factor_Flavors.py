# streamlit run adfm_basket_panel.py
# ADFM Baskets — Bloomberg-style white panel, no flavors

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="ADFM Basket Panel", layout="wide")

TITLE = "ADFM Basket Panel"
SUBTITLE = "Bloomberg-style white panel for baskets only. Indicators use DAILY data regardless of view."

# Pastel palette used for chart lines, not table fills
PASTEL = [
    "#AEC6CF","#FFB347","#B39EB5","#77DD77","#F49AC2",
    "#CFCFC4","#DEA5A4","#C6E2FF","#FFDAC1","#E2F0CB",
    "#C7CEEA","#FFB3BA","#FFD1DC","#B5EAD7","#E7E6F7",
    "#F1E3DD","#B0E0E6","#E0BBE4","#F3E5AB","#D5E8D4"
]

# -----------------------------
# Basket universe (flattened)
# -----------------------------
BASKETS = {
    # Growth & Innovation
    "Semiconductors": ["SMH"],  # proxy ok
    "AI Infrastructure Leaders": ["NVDA","AMD","AVGO","TSM","ASML","ANET","MU"],
    "Hyperscalers & Cloud": ["MSFT","AMZN","GOOGL","META","ORCL"],
    "Quality SaaS": ["ADBE","CRM","NOW","INTU","SNOW"],
    "Cybersecurity": ["PANW","FTNT","CRWD","ZS","OKTA"],
    "Digital Payments": ["V","MA","PYPL","SQ","FI","FIS"],
    "E-Commerce Platforms": ["AMZN","SHOP","MELI","ETSY"],
    "Social & Consumer Internet": ["META","SNAP","PINS","MTCH","GOOGL"],
    "Streaming & Media": ["NFLX","DIS","WBD","PARA","ROKU"],
    "Fintech & Neobanks": ["SQ","PYPL","AFRM","HOOD","SOFI"],

    # AI & Next-Gen Compute
    "5G & Networking Infra": ["AMT","CCI","SBAC","ANET","CSCO"],
    "Industrial Automation": ["ROK","ETN","EMR","AME","PH"],
    "Space Economy": ["ARKX","RKLB","IRDM","ASTS"],

    # Energy & Hard Assets
    "Energy Majors": ["XOM","CVX","COP","SHEL","BP"],
    "US Shale & E&Ps": ["EOG","DVN","FANG","MRO","OXY"],
    "Oilfield Services": ["SLB","HAL","BKR","NOV","CHX"],
    "Uranium & Fuel Cycle": ["CCJ","UUUU","UEC","URG","UROY"],
    "Battery & Materials": ["ALB","SQM","LTHM","PLL","LAC"],
    "Metals & Mining": ["BHP","RIO","VALE","FCX","NEM"],
    "Gold & Silver Miners": ["GDX","GDXJ","NEM","AEM","PAAS"],

    # Clean Energy
    "Solar & Inverters": ["TAN","FSLR","ENPH","SEDG","RUN"],
    "Wind & Renewables": ["ICLN","FAN","FSLR","ENPH","SEDG"],
    "Hydrogen": ["PLUG","BE","BLDP"],
    "Utilities & Power": ["VST","CEG","NEE","DUK","SO"],

    # Health & Longevity
    "Large-Cap Biotech": ["AMGN","GILD","REGN","BIIB"],
    "GLP-1 & Metabolic": ["NVO","LLY","PFE","AZN"],
    "MedTech Devices": ["MDT","SYK","ISRG","BSX","ZBH"],
    "Healthcare Payers": ["UNH","HUM","CI","ELV"],

    # Financials & Credit
    "Money-Center & IBs": ["JPM","BAC","C","WFC","GS","MS"],
    "Regional Banks": ["KRE","CFG","FITB","TFC","RF"],
    "Brokers & Exchanges": ["IBKR","SCHW","CME","ICE","NDAQ","CBOE"],
    "Alt Managers & PE": ["BX","KKR","APO","CG","ARES"],
    "Mortgage Finance": ["RKT","UWMC","COOP","FNF"],

    # Real Assets & Inflation
    "Homebuilders": ["ITB","DHI","LEN","NVR","PHM","TOL"],
    "REITs Core": ["VNQ","PLD","AMT","EQIX","SPG","O"],
    "Shipping & Logistics": ["FDX","UPS","GXO","XPO","ZIM"],
    "Agriculture & Machinery": ["MOS","NTR","DE","CNHI","ADM","BG"],

    # Consumer
    "Retail Discretionary": ["HD","LOW","M","GPS","BBY","TJX"],
    "Restaurants": ["MCD","SBUX","YUM","CMG","DRI"],
    "Travel & Booking": ["BKNG","EXPE","ABNB","TRIP"],
    "Hotels & Casinos": ["MAR","HLT","IHG","MGM","LVS","WYNN"],
    "Airlines": ["AAL","DAL","UAL","LUV","JBLU"],
    "Autos Legacy OEMs": ["TM","HMC","F","GM","STLA"],
    "Electric Vehicles": ["TSLA","RIVN","LCID","NIO","LI","XPEV"],

    # Defensives & Alt
    "Retail Staples": ["WMT","COST","TGT","DG","KR"],
    "Telecom & Cable": ["T","VZ","TMUS","CHTR","CMCSA"],
    "Aerospace & Defense": ["LMT","NOC","RTX","GD","HII"],
    "Crypto Proxies": ["COIN","MSTR","MARA","RIOT","BITO"],
    "China Tech ADRs": ["BABA","BIDU","JD","PDD","BILI","TCEHY"]
}

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_daily_levels(tickers, start, end):
    df = yf.download(list(set(tickers)), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series): df = df.to_frame()
    return df.sort_index()

def ew_rets_from_levels(levels: pd.DataFrame, baskets: dict) -> pd.DataFrame:
    rets = levels.pct_change()
    out = {}
    for b, tks in baskets.items():
        cols = [c for c in tks if c in rets.columns]
        if not cols: 
            continue
        if len(cols) == 1:
            out[b] = rets[cols[0]]
        else:
            out[b] = rets[cols].mean(axis=1, skipna=True)
    return pd.DataFrame(out).dropna(how="all")

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd_hist(series: pd.Series, fast=12, slow=26, signal=9):
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

def momentum_label(hist: pd.Series, lookback: int = 5) -> str:
    if hist.empty: return "Neutral"
    latest = hist.iloc[-1]
    ref = hist.iloc[-lookback] if len(hist) > lookback else hist.iloc[0]
    base = "Positive" if latest > 0 else ("Negative" if latest < 0 else "Neutral")
    if base == "Neutral": return "Neutral"
    return f"{base} {'Strengthening' if latest - ref > 0 else 'Weakening'}"

def realized_vol(returns: pd.Series, days: int = 63, ann: int = 252) -> float:
    sub = returns.dropna().iloc[-days:]
    return float(sub.std(ddof=0) * np.sqrt(ann) * 100.0) if sub.shape[0] else np.nan

def pct_since(levels: pd.Series, start_ts: pd.Timestamp) -> float:
    sub = levels[levels.index >= start_ts]
    return float((sub.iloc[-1] / sub.iloc[0]) - 1.0) if sub.shape[0] else np.nan

# -----------------------------
# Sidebar
# -----------------------------
st.title(TITLE)
st.caption(SUBTITLE)

with st.sidebar:
    st.markdown("### About This Tool")
    st.write(
        "Daily indicators on equal-weight baskets: **%5D, %1M, %YTD**, **RSI-14D**, "
        "**MACD momentum**, **EMA 4/9/18 regime**, **RSI-14W**, **3M realized vol**."
    )
    st.divider()
    st.markdown("### Controls")
    today = date.today()
    start_of_year = date(today.year, 1, 1)
    preset = st.selectbox("Date Range Preset", ["YTD","1W","1M","3M","1Y","3Y","5Y"], index=0)
    if preset == "YTD":
        start_date = start_of_year
    elif preset == "1W":
        start_date = today - timedelta(days=7)
    elif preset == "1M":
        start_date = today - timedelta(days=30)
    elif preset == "3M":
        start_date = today - timedelta(days=90)
    elif preset == "1Y":
        start_date = today - timedelta(days=365)
    elif preset == "3Y":
        start_date = today - timedelta(days=365*3)
    else:
        start_date = today - timedelta(days=365*5)
    end_date = today

    # choose baskets
    all_names = list(BASKETS.keys())
    default_sel = all_names[:15]
    selected_baskets = st.multiselect("Baskets", all_names, default=default_sel)

    show_chart = st.checkbox("Show interactive cumulative chart", value=True)

if not selected_baskets:
    st.warning("Select at least one basket.")
    st.stop()

# -----------------------------
# Data
# -----------------------------
need = set(["SPY"])
for b in selected_baskets: need.update(BASKETS[b])
levels = fetch_daily_levels(list(need), pd.to_datetime(start_date), pd.to_datetime(end_date) + pd.Timedelta(days=1))
if levels.empty:
    st.error("No data returned. Expand the range.")
    st.stop()

basket_rets_full = ew_rets_from_levels(levels, BASKETS)
basket_rets = basket_rets_full[selected_baskets].dropna(how="all")
basket_lvls_100 = 100 * (1 + basket_rets).cumprod()

# -----------------------------
# Build Bloomberg-style panel rows
# -----------------------------
rows = []
for b in basket_lvls_100.columns:
    lvl = basket_lvls_100[b].dropna()
    if lvl.shape[0] < 30: 
        continue
    r5d = (lvl.iloc[-1] / lvl.iloc[-6]) - 1.0 if len(lvl) > 6 else np.nan
    r1m = pct_since(lvl, lvl.index.max() - pd.DateOffset(months=1))
    y_start = pd.Timestamp(year=lvl.index.max().year, month=1, day=1, tz=getattr(lvl.index.max(), "tz", None))
    rytd = pct_since(lvl, y_start)
    rsi_14d = rsi(lvl, 14).iloc[-1] if len(lvl) > 20 else np.nan
    weekly = lvl.resample("W-FRI").last().dropna()
    rsi_14w = rsi(weekly, 14).iloc[-1] if len(weekly) > 20 else np.nan
    hist = macd_hist(lvl, 12, 26, 9)
    macd_m = momentum_label(hist, 5)
    ema_tag = ema_regime(lvl, 4, 9, 18)
    rv = realized_vol(basket_rets[b], 63, 252)

    rows.append({
        "Basket": b,
        "%5D": round(r5d*100,1) if pd.notna(r5d) else np.nan,
        "%1M": round(r1m*100,1) if pd.notna(r1m) else np.nan,
        "↓ %YTD": round(rytd*100,1) if pd.notna(rytd) else np.nan,
        "RSI 14D": round(rsi_14d,2) if pd.notna(rsi_14d) else np.nan,
        "MACD Momentum": macd_m,
        "EMA 4/9/18": ema_tag,
        "RSI 14W": round(rsi_14w,2) if pd.notna(rsi_14w) else np.nan,
        "3M RVOL": round(rv,1) if pd.notna(rv) else np.nan
    })

panel_df = pd.DataFrame(rows).set_index("Basket")
if "↓ %YTD" in panel_df.columns:
    panel_df = panel_df.sort_values(by="↓ %YTD", ascending=False)

# -----------------------------
# Color functions for the table
# -----------------------------
def color_ret(x):
    # green positive, red negative, pale intensity
    if pd.isna(x): return "white"
    if x >= 0:
        # scale 0 to +20 -> light to medium green
        s = min(abs(x)/20.0, 1.0)
        g = int(255 - 90*s)
        return f"rgb({int(240-120*s)},{g},{int(240-120*s)})"
    else:
        s = min(abs(x)/20.0, 1.0)
        r = int(255 - 90*s)
        return f"rgb({r},{int(240-120*s)},{int(240-120*s)})"

def color_rsi(x):
    if pd.isna(x): return "white"
    if x >= 70: return "rgb(255,237,170)"     # overbought, soft amber
    if x <= 30: return "rgb(255,200,200)"     # oversold, soft red
    return "rgb(230,236,245)"                  # neutral blue-gray

def color_macd(tag):
    if not isinstance(tag, str): return "white"
    if tag.startswith("Positive"):
        return "rgb(204,238,204)" if "Strengthening" in tag else "rgb(225,246,225)"
    if tag.startswith("Negative"):
        return "rgb(255,210,210)" if "Strengthening" in tag else "rgb(255,228,228)"
    return "rgb(230,236,245)"

def color_ema(tag):
    if tag == "Up": return "rgb(204,238,204)"
    if tag == "Down": return "rgb(255,210,210)"
    return "rgb(230,236,245)"

def color_vol(x):
    if pd.isna(x): return "white"
    # blue gradient by percentile
    return "rgb(220,232,255)" if x < 60 else ("rgb(200,220,255)" if x < 90 else "rgb(180,205,255)")

# Build cell color matrix for Plotly Table
headers = ["Basket","%5D","%1M","↓ %YTD","RSI 14D","MACD Momentum","EMA 4/9/18","RSI 14W","3M RVOL"]
values = [panel_df.index.tolist()]
fill_colors = [["white"] * len(panel_df)]  # Basket column

for col in ["%5D","%1M","↓ %YTD"]:
    vals = panel_df[col].tolist()
    values.append(vals)
    fill_colors.append([color_ret(v) for v in vals])

vals = panel_df["RSI 14D"].tolist()
values.append(vals)
fill_colors.append([color_rsi(v) for v in vals])

vals = panel_df["MACD Momentum"].tolist()
values.append(vals)
fill_colors.append([color_macd(v) for v in vals])

vals = panel_df["EMA 4/9/18"].tolist()
values.append(vals)
fill_colors.append([color_ema(v) for v in vals])

vals = panel_df["RSI 14W"].tolist()
values.append(vals)
fill_colors.append([color_rsi(v) for v in vals])

vals = panel_df["3M RVOL"].tolist()
values.append(vals)
fill_colors.append([color_vol(v) for v in vals])

# Render Plotly table
st.subheader("Bloomberg-Style Basket Panel")
fig_tbl = go.Figure(data=[go.Table(
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
        height=28,
        format=[None, ".1f", ".1f", ".1f", ".2f", None, None, ".2f", ".1f"]
    )
)])
fig_tbl.update_layout(margin=dict(l=0,r=0,t=0,b=0))
st.plotly_chart(fig_tbl, use_container_width=True)

# -----------------------------
# Optional interactive chart
# -----------------------------
if show_chart:
    st.subheader("Cumulative Performance (interactive)")
    cum_pct = ((1 + basket_rets[selected_baskets]).cumprod() - 1.0) * 100.0
    spy_cum = ((1 + levels["SPY"].pct_change().dropna()).cumprod() - 1.0) * 100.0
    fig = go.Figure()
    for i, b in enumerate(cum_pct.columns):
        fig.add_trace(go.Scatter(
            x=cum_pct.index, y=cum_pct[b],
            mode="lines",
            line=dict(width=2, color=PASTEL[i % len(PASTEL)]),
            name=b,
            hovertemplate=f"{b}<br>% Cum: %{ '{y:.1f}' }%<extra></extra>"
        ))
    fig.add_trace(go.Scatter(
        x=spy_cum.index, y=spy_cum.values,
        mode="lines",
        line=dict(width=2, dash="dash", color="#888"),
        name="SPY",
        hovertemplate="SPY<br>% Cum: %{y:.1f}%<extra></extra>"
    ))
    fig.update_layout(showlegend=False, hovermode="x unified", yaxis_title="Cumulative return, %")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Basket constituents
# -----------------------------
with st.expander("Basket Constituents"):
    for name, tks in BASKETS.items():
        st.write(f"**{name}**: {', '.join(tks)}")
