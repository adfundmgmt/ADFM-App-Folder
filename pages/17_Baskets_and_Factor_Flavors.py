# streamlit run this_file.py
# ADFM Basket & Flavor Dashboard — Bloomberg-style panel included
# Pastel visuals, basket filtering, flavor aggregates, frequency control,
# rolling corr and z-score vs SPY, plus a Bloomberg-like table for each basket.
# Charts use matplotlib only (no seaborn). Pastel color accents.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="ADFM Baskets & Flavors", layout="wide")

TITLE = "ADFM Basket Performance with Factor-Flavor Aggregates"
SUBTITLE = "Equal-weight returns via Yahoo Finance. Bloomberg-style panel adds 5D, 1M, YTD, RSI-14D/W, MACD momentum, 4/9/18 EMA regime, and 3M realized vol."

PASTEL = [
    "#AEC6CF", "#FFB347", "#B39EB5", "#77DD77", "#F49AC2",
    "#CFCFC4", "#DEA5A4", "#C6E2FF", "#FFDAC1", "#E2F0CB",
    "#C7CEEA", "#FFB3BA", "#FFD1DC", "#B5EAD7", "#E7E6F7",
    "#F1E3DD", "#B0E0E6", "#E0BBE4", "#F3E5AB", "#D5E8D4"
]

# ---------------------------
# Factor Flavors and Baskets
# ---------------------------
FACTOR_FLAVORS = {
    "Growth & Innovation": {
        "Semiconductors": ["SMH"],
        "AI Infrastructure Leaders": ["NVDA","AMD","AVGO","TSM","ASML","ANET","MU"],
        "Hyperscalers & Cloud": ["MSFT","AMZN","GOOGL","META","ORCL"],
        "Quality SaaS": ["ADBE","CRM","NOW","INTU","SNOW"],
        "Cybersecurity": ["PANW","FTNT","CRWD","ZS","OKTA"],
        "Digital Payments": ["V","MA","PYPL","SQ","FI","FIS"],
        "E-Commerce Platforms": ["AMZN","SHOP","MELI","ETSY"],
        "Social & Consumer Internet": ["META","SNAP","PINS","MTCH","GOOGL"],
        "Streaming & Media": ["NFLX","DIS","WBD","PARA","ROKU"],
        "Fintech & Neobanks": ["SQ","PYPL","AFRM","HOOD","SOFI"]
    },
    "AI & Next-Gen Compute": {
        "AI Infrastructure Leaders": ["NVDA","AMD","AVGO","TSM","ASML","ANET","MU"],
        "Hyperscalers & Cloud": ["MSFT","AMZN","GOOGL","META","ORCL"],
        "5G & Networking Infra": ["AMT","CCI","SBAC","ANET","CSCO"],
        "Industrial Automation": ["ROK","ETN","EMR","AME","PH"],
        "Space Economy": ["ARKX","RKLB","IRDM","ASTS"]
    },
    "Energy & Hard Assets": {
        "Energy Majors": ["XOM","CVX","COP","SHEL","BP"],
        "US Shale & E&Ps": ["EOG","DVN","FANG","MRO","OXY"],
        "Oilfield Services": ["SLB","HAL","BKR","NOV","CHX"],
        "Uranium & Fuel Cycle": ["CCJ","UUUU","UEC","URG","UROY"],
        "Battery & Materials": ["ALB","SQM","LTHM","PLL","LAC"],
        "Metals & Mining": ["BHP","RIO","VALE","FCX","NEM"],
        "Gold & Silver Miners": ["GDX","GDXJ","NEM","AEM","PAAS"]
    },
    "Clean Energy Transition": {
        "Solar & Inverters": ["TAN","FSLR","ENPH","SEDG","RUN"],
        "Wind & Renewables": ["ICLN","FAN","FSLR","ENPH","SEDG"],
        "Hydrogen": ["PLUG","BE","BLDP"],
        "Utilities & Power": ["VST","CEG","NEE","DUK","SO"]
    },
    "Health & Longevity": {
        "Large-Cap Biotech": ["AMGN","GILD","REGN","BIIB"],
        "GLP-1 & Metabolic": ["NVO","LLY","PFE","AZN"],
        "MedTech Devices": ["MDT","SYK","ISRG","BSX","ZBH"],
        "Healthcare Payers": ["UNH","HUM","CI","ELV"]
    },
    "Financials & Credit": {
        "Money-Center & IBs": ["JPM","BAC","C","WFC","GS","MS"],
        "Regional Banks": ["KRE","CFG","FITB","TFC","RF"],
        "Brokers & Exchanges": ["IBKR","SCHW","CME","ICE","NDAQ","CBOE"],
        "Alt Managers & PE": ["BX","KKR","APO","CG","ARES"],
        "Mortgage Finance": ["RKT","UWMC","COOP","FNF"]
    },
    "Real Assets & Inflation Beneficiaries": {
        "Homebuilders": ["ITB","DHI","LEN","NVR","PHM","TOL"],
        "REITs Core": ["VNQ","PLD","AMT","EQIX","SPG","O"],
        "Shipping & Logistics": ["FDX","UPS","GXO","XPO","ZIM"],
        "Agriculture & Machinery": ["MOS","NTR","DE","CNHI","ADM","BG"]
    },
    "Consumer Cyclicals": {
        "Retail Discretionary": ["HD","LOW","M","GPS","BBY","TJX"],
        "Restaurants": ["MCD","SBUX","YUM","CMG","DRI"],
        "Travel & Booking": ["BKNG","EXPE","ABNB","TRIP"],
        "Hotels & Casinos": ["MAR","HLT","IHG","MGM","LVS","WYNN"],
        "Airlines": ["AAL","DAL","UAL","LUV","JBLU"],
        "Autos Legacy OEMs": ["TM","HMC","F","GM","STLA"],
        "Electric Vehicles": ["TSLA","RIVN","LCID","NIO","LI","XPEV"]
    },
    "Defensives & Staples": {
        "Retail Staples": ["WMT","COST","TGT","DG","KR"],
        "Telecom & Cable": ["T","VZ","TMUS","CHTR","CMCSA"],
        "Aerospace & Defense": ["LMT","NOC","RTX","GD","HII"]
    },
    "Alternative Assets & Reflexivity Plays": {
        "Crypto Proxies": ["COIN","MSTR","MARA","RIOT","BITO"],
        "China Tech ADRs": ["BABA","BIDU","JD","PDD","BILI","TCEHY"]
    }
}

USE_SELECTED_FOR_FLAVOR = False  # set True to aggregate flavors only from the selected baskets

def build_basket_universe(flavor_dict: dict) -> dict:
    baskets = {}
    for _, groups in flavor_dict.items():
        for name, tks in groups.items():
            baskets[name] = list(dict.fromkeys(tks))
    return baskets

BASKETS = build_basket_universe(FACTOR_FLAVORS)

# ---------------------------
# Data helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def fetch_prices(tickers, start, end):
    df = yf.download(list(set(tickers)), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.sort_index()

def resample_levels(levels: pd.DataFrame, freq: str) -> pd.DataFrame:
    rule = {"Daily":"1D","Weekly":"W-FRI","Monthly":"M","Annual":"Y"}[freq]
    return levels.resample(rule).last().dropna(how="all")

def pct_returns(levels: pd.DataFrame) -> pd.DataFrame:
    return levels.pct_change().dropna(how="all")

def ew_basket_returns(levels: pd.DataFrame, baskets: dict) -> pd.DataFrame:
    rets = levels.pct_change()
    out = {}
    for b, tks in baskets.items():
        cols = [c for c in tks if c in rets.columns]
        if len(cols) == 1:
            out[b] = rets[cols[0]]
        elif len(cols) > 1:
            out[b] = rets[cols].mean(axis=1, skipna=True)
    return pd.DataFrame(out).dropna(how="all")

def flavor_returns(basket_ret_df: pd.DataFrame, selected=None) -> pd.DataFrame:
    out = {}
    for flavor, groups in FACTOR_FLAVORS.items():
        names = list(groups.keys())
        if USE_SELECTED_FOR_FLAVOR and selected is not None:
            names = [n for n in names if n in selected]
        cols = [c for c in names if c in basket_ret_df.columns]
        if len(cols) == 1:
            out[flavor] = basket_ret_df[cols[0]]
        elif len(cols) > 1:
            out[flavor] = basket_ret_df[cols].mean(axis=1, skipna=True)
    return pd.DataFrame(out).dropna(how="all")

# ---------------------------
# Indicator helpers
# ---------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def macd_hist(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

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
    # Positive/Negative from sign, Strengthening/Weakening from slope vs lookback
    if hist.empty:
        return "Neutral"
    latest = hist.iloc[-1]
    ref = hist.iloc[-lookback] if len(hist) > lookback else hist.iloc[0]
    slope = latest - ref
    base = "Positive" if latest > 0 else ("Negative" if latest < 0 else "Neutral")
    if base == "Neutral":
        return "Neutral"
    trend = "Strengthening" if slope > 0 else "Weakening"
    return f"{base} {trend}"

def realized_vol(returns: pd.Series, days: int = 63, ann: int = 252) -> float:
    sub = returns.dropna().iloc[-days:]
    if sub.empty:
        return np.nan
    return float(sub.std(ddof=0) * np.sqrt(ann) * 100.0)

def period_return(levels: pd.Series, periods_back: int) -> float:
    if len(levels) <= periods_back: 
        return np.nan
    return float((levels.iloc[-1] / levels.iloc[-1 - periods_back]) - 1.0)

def date_return(levels: pd.Series, start_ts: pd.Timestamp) -> float:
    sub = levels[levels.index >= start_ts]
    if sub.empty:
        return np.nan
    return float((sub.iloc[-1] / sub.iloc[0]) - 1.0)

# ---------------------------
# UI
# ---------------------------
st.title(TITLE)
st.caption(SUBTITLE)

with st.sidebar:
    st.header("Controls")
    today = date.today()
    default_start = today - timedelta(days=365*3)
    start_date = st.date_input("Start date", default_start)
    end_date = st.date_input("End date", today)
    freq = st.selectbox("Frequency", ["Daily","Weekly","Monthly","Annual"], index=2)
    corr_window = st.number_input("Rolling correlation window (periods)", 20, 252, 63, 1)
    show_flavor_panel = st.checkbox("Show factor-flavor aggregates", value=True)
    show_bloomberg_panel = st.checkbox("Show Bloomberg-style basket panel", value=True)

    fltrs = ["All"] + list(FACTOR_FLAVORS.keys())
    chosen_flavor = st.selectbox("Filter baskets by flavor", fltrs, index=0)
    possible_baskets = list(BASKETS.keys()) if chosen_flavor == "All" else list(FACTOR_FLAVORS[chosen_flavor].keys())
    default_sel = possible_baskets[:12] if len(possible_baskets) >= 12 else possible_baskets
    selected_baskets = st.multiselect("Baskets", possible_baskets, default=default_sel)
    st.markdown("---")
    st.markdown("Equal-weight baskets by period. ETF proxies where appropriate. Pastel accents, minimal clutter.")

if not selected_baskets:
    st.warning("Select at least one basket.")
    st.stop()

# ---------------------------
# Data fetch
# ---------------------------
need = set(["SPY"])
for b in selected_baskets:
    need.update(BASKETS[b])
if show_flavor_panel and not USE_SELECTED_FOR_FLAVOR:
    for flv in FACTOR_FLAVORS:
        for b in FACTOR_FLAVORS[flv]:
            need.update(BASKETS[b])

levels = fetch_prices(list(need), start=pd.to_datetime(start_date), end=pd.to_datetime(end_date) + pd.Timedelta(days=1))
if levels.empty:
    st.error("No price data returned. Expand the range.")
    st.stop()

levels_res = resample_levels(levels, freq)
rets_res = pct_returns(levels_res)

basket_rets_full = ew_basket_returns(levels_res, BASKETS)
basket_rets = basket_rets_full[selected_baskets].dropna(how="all")
basket_cum = (1 + basket_rets).cumprod()
spy_ret = rets_res["SPY"].rename("SPY")
spy_cum = (1 + spy_ret).cumprod()

# ---------------------------
# Bloomberg-style basket panel
# ---------------------------
if show_bloomberg_panel:
    st.subheader("Bloomberg-style Basket Panel")

    panel_rows = []
    for b in basket_rets.columns:
        # rebuild a synthetic "basket level" to compute indicators from levels not returns
        # proxy price series = index the cumulative return to 100
        lvl = 100 * (1 + basket_rets[b].dropna()).cumprod()

        # returns
        r_5d = period_return(lvl, periods_back=5)
        r_1m = date_return(lvl, lvl.index.max() - pd.DateOffset(months=1))
        y_start = pd.Timestamp(year=lvl.index.max().year, month=1, day=1, tz=getattr(lvl.index.max(), "tz", None))
        r_ytd = date_return(lvl, y_start)

        # RSI daily and weekly
        rsi_14d = rsi(lvl, 14).iloc[-1] if len(lvl) > 20 else np.nan
        weekly_lvl = lvl.resample("W-FRI").last().dropna()
        rsi_14w = rsi(weekly_lvl, 14).iloc[-1] if len(weekly_lvl) > 20 else np.nan

        # MACD histogram and label
        _, _, hist = macd_hist(lvl, 12, 26, 9)
        macd_mom = momentum_label(hist, lookback=5)

        # 4/9/18 EMA regime on daily levels
        ema_reg = ema_regime(lvl, 4, 9, 18)

        # 3M realized vol from daily returns
        rv_3m = realized_vol(basket_rets[b], days=63, ann=252)

        panel_rows.append({
            "Basket": b,
            "%5D": round(r_5d*100, 1) if pd.notna(r_5d) else np.nan,
            "%1M": round(r_1m*100, 1) if pd.notna(r_1m) else np.nan,
            "%YTD": round(r_ytd*100, 1) if pd.notna(r_ytd) else np.nan,
            "RSI 14D": round(rsi_14d, 2) if pd.notna(rsi_14d) else np.nan,
            "MACD Momentum": macd_mom,
            "EMA 4/9/18": ema_reg,
            "RSI 14W": round(rsi_14w, 2) if pd.notna(rsi_14w) else np.nan,
            "3M Realized Vol": round(rv_3m, 1) if pd.notna(rv_3m) else np.nan
        })

    panel = pd.DataFrame(panel_rows).set_index("Basket")
    # Sort by YTD descending like Bloomberg often does
    panel = panel.sort_values(by="%YTD", ascending=False)

    # Display clean table
    st.dataframe(
        panel,
        use_container_width=True
    )

    # Optional: compact legend
    with st.expander("Legend"):
        st.markdown(
            "- **MACD Momentum**: Positive/Negative based on MACD histogram sign, Strengthening/Weakening vs 5-day change\n"
            "- **EMA 4/9/18**: Up if EMA4>EMA9>EMA18, Down if EMA4<EMA9<EMA18, else Neutral\n"
            "- **3M Realized Vol**: Annualized vol from last 63 daily returns\n"
            "- For true RVOL using exchange volume, we can compute it for ETF proxies only on request"
        )

# ---------------------------
# Basket overview and charts (kept from prior build)
# ---------------------------
st.subheader("Basket Overview")
cols = st.columns(min(4, len(basket_rets.columns)))
groups = [list(basket_rets.columns)[i::4] for i in range(4)]
def perf_summary(returns: pd.Series):
    if returns is None or returns.empty:
        return {"YTD": np.nan, "1M": np.nan, "3M": np.nan, "6M": np.nan, "1Y": np.nan}
    last = returns.index.max()
    def sub(start): 
        subr = returns[returns.index >= start]
        return (1 + subr).prod() - 1 if subr.shape[0] else np.nan
    s = {}
    s["YTD"] = sub(pd.Timestamp(year=last.year, month=1, day=1, tz=getattr(last, "tz", None)))
    s["1M"] = sub(last - pd.DateOffset(months=1))
    s["3M"] = sub(last - pd.DateOffset(months=3))
    s["6M"] = sub(last - pd.DateOffset(months=6))
    s["1Y"] = sub(last - pd.DateOffset(years=1))
    return s

def rolling_corr_zscore(a: pd.Series, b: pd.Series, window=63):
    aligned = pd.concat([a, b], axis=1).dropna()
    if aligned.shape[0] < window * 2:
        return pd.Series(dtype=float), np.nan, np.nan
    corr = aligned.iloc[:,0].rolling(window).corr(aligned.iloc[:,1])
    mu, sd = corr.mean(), corr.std(ddof=0)
    z = (corr - mu) / sd if sd and sd > 0 else pd.Series(np.nan, index=corr.index)
    return z, corr.dropna().iloc[-1] if corr.dropna().shape[0] else np.nan, z.dropna().iloc[-1] if z.dropna().shape[0] else np.nan

def max_dd(cum: pd.Series):
    peak = cum.cummax()
    dd = cum/peak - 1
    return float(dd.min()) if len(dd) else np.nan

for c, g in zip(cols, groups):
    with c:
        for b in g:
            sr = basket_rets[b].dropna()
            ps = perf_summary(sr)
            dd = max_dd((1 + sr).cumprod())
            _, lc, lz = rolling_corr_zscore(sr, spy_ret, window=int(corr_window))
            c.metric(
                label=b,
                value=f"YTD {ps['YTD']*100:0.1f}%" if pd.notna(ps['YTD']) else "YTD n/a",
                help=(f"1M {ps['1M']*100:0.1f}%, 3M {ps['3M']*100:0.1f}%, "
                      f"6M {ps['6M']*100:0.1f}%, 1Y {ps['1Y']*100:0.1f}%, "
                      f"Max DD {dd*100:0.1f}%, Corr {lc:0.2f}, Corr z {lz:0.2f}")
            )

st.subheader("Cumulative Performance")
fig1, ax1 = plt.subplots(figsize=(10,5))
for i, b in enumerate(basket_rets.columns):
    ax1.plot((1 + basket_rets[b]).cumprod().index, (1 + basket_rets[b]).cumprod().values,
             label=b, linewidth=2, color=PASTEL[i % len(PASTEL)])
ax1.plot(spy_cum.index, spy_cum.values, label="SPY", linewidth=2, linestyle="--", color="#888888")
ax1.set_ylabel("Growth of $1"); ax1.grid(alpha=0.2); ax1.legend(ncol=2, fontsize=9)
st.pyplot(fig1, clear_figure=True)

st.subheader("Rolling Correlation vs SPY and z-score")
tab1, tab2 = st.tabs(["Correlation", "Correlation z-score"])
with tab1:
    fig2, ax2 = plt.subplots(figsize=(10,5))
    for i, b in enumerate(basket_rets.columns):
        corr = pd.concat([basket_rets[b], spy_ret], axis=1).dropna().iloc[:,0].rolling(int(corr_window)).corr(
            pd.concat([basket_rets[b], spy_ret], axis=1).dropna().iloc[:,1]
        )
        ax2.plot(corr.index, corr, label=b, linewidth=2, color=PASTEL[i % len(PASTEL)])
    ax2.axhline(0, color="#999999", linestyle=":"); ax2.grid(alpha=0.2); ax2.legend(ncol=2, fontsize=9)
    st.pyplot(fig2, clear_figure=True)
with tab2:
    fig3, ax3 = plt.subplots(figsize=(10,5))
    for i, b in enumerate(basket_rets.columns):
        aligned = pd.concat([basket_rets[b], spy_ret], axis=1).dropna()
        if aligned.shape[0] >= int(corr_window)*2:
            corr = aligned.iloc[:,0].rolling(int(corr_window)).corr(aligned.iloc[:,1])
            mu, sd = corr.mean(), corr.std(ddof=0)
            z = (corr - mu) / sd if sd and sd > 0 else pd.Series(np.nan, index=corr.index)
            ax3.plot(z.index, z, label=b, linewidth=2, color=PASTEL[i % len(PASTEL)])
    ax3.axhline(0, color="#999999", linestyle=":"); ax3.grid(alpha=0.2); ax3.legend(ncol=2, fontsize=9)
    st.pyplot(fig3, clear_figure=True)

# ---------------------------
# Flavor aggregates (optional)
# ---------------------------
if show_flavor_panel:
    st.markdown("### Factor-Flavor Aggregates")
    flv_rets = flavor_returns(basket_rets_full, selected=basket_rets.columns.tolist())
    flv_rets = flv_rets.loc[basket_rets.index]
    flv_cum = (1 + flv_rets).cumprod()

    # KPIs
    st.subheader("Flavor Overview")
    fcols = st.columns(min(4, len(flv_rets.columns)))
    fgps = [list(flv_rets.columns)[i::4] for i in range(4)]
    for c, g in zip(fcols, fgps):
        with c:
            for f in g:
                sr = flv_rets[f].dropna()
                last = sr.index.max()
                def sub(start):
                    subr = sr[sr.index >= start]
                    return (1 + subr).prod() - 1 if subr.shape[0] else np.nan
                ytd = sub(pd.Timestamp(year=last.year, month=1, day=1, tz=getattr(last,"tz",None)))
                one_m = sub(last - pd.DateOffset(months=1))
                dd = (1 + sr).cumprod()
                ddv = (dd / dd.cummax() - 1).min()
                aligned = pd.concat([sr, spy_ret], axis=1).dropna()
                corr = aligned.iloc[:,0].rolling(int(corr_window)).corr(aligned.iloc[:,1])
                c.metric(
                    label=f,
                    value=f"YTD {ytd*100:0.1f}%" if pd.notna(ytd) else "YTD n/a",
                    help=f"1M {one_m*100:0.1f}%, Max DD {ddv*100:0.1f}%"
                )

    st.subheader("Flavor Cumulative Performance")
    fig4, ax4 = plt.subplots(figsize=(10,5))
    for i, f in enumerate(flv_rets.columns):
        ax4.plot(flv_cum.index, flv_cum[f], label=f, linewidth=2, color=PASTEL[i % len(PASTEL)])
    ax4.plot(spy_cum.index, spy_cum.values, label="SPY", linewidth=2, linestyle="--", color="#888888")
    ax4.grid(alpha=0.2); ax4.legend(ncol=2, fontsize=9)
    st.pyplot(fig4, clear_figure=True)

with st.expander("About this dashboard"):
    st.markdown("""
**Methodology**
- Basket return = equal-weight average of member tickers' period returns.
- Flavor return = equal-weight average of member baskets' returns.
- Resample uses period-end closes, then percent change.
- MACD regime from histogram sign, momentum from change over 5 sessions.
- EMA regime uses 4/9/18 EMAs: Up if stacked rising, Down if stacked falling, else Neutral.
- 3M Realized Vol is annualized from the last 63 daily returns.

**Notes**
- For true RVOL you need exchange volume. I can compute that for ETF proxies on request.
- Yahoo adjusted close used as a total-return proxy.
""")
