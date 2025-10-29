# streamlit run this_file.py
# ADFM Basket & Factor-Flavor Dashboard
# Pastel visuals, basket filtering, flavor aggregates, frequency control, rolling corr and z-score vs SPY.
# Charts use matplotlib only (no seaborn). Pastel colors per request.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import matplotlib.pyplot as plt

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="Baskets & Factor Flavors", layout="wide")

TITLE = "ADFM Basket Performance with Factor-Flavor Aggregates"
SUBTITLE = "Equal-weight returns from Yahoo Finance. Frequency selectable. Rolling correlation and z-score vs SPY for baskets and flavors."

# Pastel palette
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

# Use only currently selected baskets when computing flavor aggregates?
USE_SELECTED_FOR_FLAVOR = False  # set True if you want flavor aggregates tied to selection

def build_basket_universe(factor_flavors: dict) -> dict:
    baskets = {}
    for _, groups in factor_flavors.items():
        for basket_name, tks in groups.items():
            baskets[basket_name] = list(dict.fromkeys(tks))  # de-dup while keeping order
    return baskets

BASKETS = build_basket_universe(FACTOR_FLAVORS)

def baskets_by_flavor(flavor_name: str) -> list:
    return list(FACTOR_FLAVORS[flavor_name].keys())

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def fetch_prices(tickers, start, end):
    df = yf.download(
        tickers=list(set(tickers)),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.sort_index()
    return df

def resample_prices(prices: pd.DataFrame, freq: str) -> pd.DataFrame:
    rule = {"Daily":"1D", "Weekly":"W-FRI", "Monthly":"M", "Annual":"Y"}[freq]
    out = prices.resample(rule).last().dropna(how="all")
    return out

def pct_returns(levels: pd.DataFrame) -> pd.DataFrame:
    return levels.pct_change().dropna(how="all")

def build_basket_returns(levels: pd.DataFrame, baskets: dict) -> pd.DataFrame:
    simple_ret = levels.pct_change()
    basket_ret = {}
    for bname, tks in baskets.items():
        cols = [c for c in tks if c in simple_ret.columns]
        if len(cols) == 0:
            continue
        if len(cols) == 1:
            basket_ret[bname] = simple_ret[cols[0]]
        else:
            basket_ret[bname] = simple_ret[cols].mean(axis=1, skipna=True)
    out = pd.DataFrame(basket_ret).dropna(how="all")
    return out

def build_flavor_returns(basket_returns: pd.DataFrame, selected_baskets: list | None = None) -> pd.DataFrame:
    # Equal-weight of baskets within each flavor
    flavor_ret = {}
    for flavor in FACTOR_FLAVORS.keys():
        baskets_list = baskets_by_flavor(flavor)
        if USE_SELECTED_FOR_FLAVOR and selected_baskets is not None:
            baskets_list = [b for b in baskets_list if b in selected_baskets]
        cols = [b for b in baskets_list if b in basket_returns.columns]
        if len(cols) == 0:
            continue
        if len(cols) == 1:
            flavor_ret[flavor] = basket_returns[cols[0]]
        else:
            flavor_ret[flavor] = basket_returns[cols].mean(axis=1, skipna=True)
    return pd.DataFrame(flavor_ret).dropna(how="all")

def max_drawdown(cum_series: pd.Series) -> float:
    roll_max = cum_series.cummax()
    dd = (cum_series / roll_max) - 1.0
    return dd.min() if len(dd) else np.nan

def rolling_corr_zscore(series_a: pd.Series, series_b: pd.Series, window: int = 63):
    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    if aligned.shape[0] < window * 2:
        return pd.Series(dtype=float), np.nan, np.nan
    corr = aligned.iloc[:,0].rolling(window).corr(aligned.iloc[:,1])
    mu, sd = corr.mean(), corr.std(ddof=0)
    z = (corr - mu) / sd if sd and sd > 0 else pd.Series(np.nan, index=corr.index)
    latest_corr = corr.dropna().iloc[-1] if corr.dropna().shape[0] else np.nan
    latest_z = z.dropna().iloc[-1] if z.dropna().shape[0] else np.nan
    return z, latest_corr, latest_z

def perf_summary(returns: pd.Series):
    if returns is None or returns.empty:
        return {"YTD": np.nan, "1M": np.nan, "3M": np.nan, "6M": np.nan, "1Y": np.nan}
    s = {}
    last = returns.index.max()
    year_start = pd.Timestamp(year=last.year, month=1, day=1, tz=getattr(last, "tz", None))
    month_ago = last - pd.DateOffset(months=1)
    three_mo = last - pd.DateOffset(months=3)
    six_mo = last - pd.DateOffset(months=6)
    one_year = last - pd.DateOffset(years=1)

    def subperiod_ret(start):
        sub = returns.loc[returns.index >= start]
        return (1 + sub).prod() - 1 if sub.shape[0] else np.nan

    s["YTD"] = subperiod_ret(year_start)
    s["1M"] = subperiod_ret(month_ago)
    s["3M"] = subperiod_ret(three_mo)
    s["6M"] = subperiod_ret(six_mo)
    s["1Y"] = subperiod_ret(one_year)
    return s

# ---------------------------
# UI
# ---------------------------
st.title(TITLE)
st.caption(SUBTITLE)

with st.sidebar:
    st.header("Controls")
    today = date.today()
    default_start = today - timedelta(days=365*3)
    start_date = st.date_input("Start date", default_start, help="Fetch from this date onward")
    end_date = st.date_input("End date", today, help="Up to this date")
    freq = st.selectbox("Frequency", ["Daily","Weekly","Monthly","Annual"], index=2, help="Resample frequency for returns")
    corr_window = st.number_input("Rolling correlation window (periods)", min_value=20, max_value=252, value=63, step=1)
    show_flavor_panel = st.checkbox("Show factor-flavor aggregates", value=True)

    flavors = ["All"] + list(FACTOR_FLAVORS.keys())
    chosen_flavor = st.selectbox("Filter baskets by flavor", flavors, index=0)

    if chosen_flavor == "All":
        possible_baskets = list(BASKETS.keys())
    else:
        possible_baskets = list(FACTOR_FLAVORS[chosen_flavor].keys())

    default_selection = possible_baskets[:6] if len(possible_baskets) >= 6 else possible_baskets
    selected_baskets = st.multiselect("Baskets", possible_baskets, default=default_selection)

    st.markdown("---")
    st.markdown("Equal-weight baskets by period. Pastel lines, SPY reference, minimal clutter.")

if len(selected_baskets) == 0:
    st.warning("Select at least one basket to display.")
    st.stop()

# ---------------------------
# Data Pipeline: Universe selection
# ---------------------------
# Always include SPY
tickers_needed = set(["SPY"])
# Baskets shown
for b in selected_baskets:
    tickers_needed.update(BASKETS[b])
# Flavor aggregates may require all baskets
if show_flavor_panel and not USE_SELECTED_FOR_FLAVOR:
    for flv in FACTOR_FLAVORS:
        for b in FACTOR_FLAVORS[flv]:
            tickers_needed.update(BASKETS[b])

# Fetch levels and compute returns
prices = fetch_prices(
    list(tickers_needed),
    start=pd.to_datetime(start_date),
    end=pd.to_datetime(end_date) + pd.Timedelta(days=1)
)
if prices.empty:
    st.error("No price data returned. Try expanding the date range.")
    st.stop()

prices_res = resample_prices(prices, freq=freq)
rets_res = pct_returns(prices_res)

# Basket returns on the resampled series
basket_rets_full = build_basket_returns(prices_res, BASKETS)
basket_rets = basket_rets_full[selected_baskets].dropna(how="all")
basket_cum = (1 + basket_rets).cumprod()

spy_ret = rets_res["SPY"].rename("SPY")
spy_cum = (1 + spy_ret).cumprod()

# ---------------------------
# Basket Overview KPIs
# ---------------------------
st.subheader("Basket Overview")
kpi_cols = st.columns(min(4, len(selected_baskets)))
kpi_groups = [selected_baskets[i::4] for i in range(4)]

for col, group in zip(kpi_cols, kpi_groups):
    with col:
        for b in group:
            sr = basket_rets[b].dropna()
            ps = perf_summary(sr)
            dd = max_drawdown((1 + sr).cumprod())
            _, latest_corr, latest_z = rolling_corr_zscore(sr, spy_ret, window=int(corr_window))
            col.metric(
                label=f"{b}",
                value=f"YTD {ps['YTD']*100:0.1f}%" if pd.notna(ps['YTD']) else "YTD n/a",
                help=(
                    f"1M {ps['1M']*100:0.1f}%, 3M {ps['3M']*100:0.1f}%, 6M {ps['6M']*100:0.1f}%, "
                    f"1Y {ps['1Y']*100:0.1f}%, Max DD {dd*100:0.1f}%\n"
                    f"Corr(SPY) {latest_corr:0.2f}, Corr z-score {latest_z:0.2f} (window {corr_window})"
                )
            )

# ---------------------------
# Basket Charts
# ---------------------------
st.subheader("Basket Cumulative Performance")
fig1, ax1 = plt.subplots(figsize=(10, 5))
for i, b in enumerate(basket_rets.columns):
    clr = PASTEL[i % len(PASTEL)]
    ax1.plot(basket_cum.index, basket_cum[b], label=b, linewidth=2, color=clr)
ax1.plot(spy_cum.index, spy_cum, label="SPY", linewidth=2, linestyle="--", color="#888888")
ax1.set_title(f"Cumulative Return ({freq} resample)")
ax1.set_ylabel("Growth of $1")
ax1.grid(alpha=0.2)
ax1.legend(loc="best", ncol=2, fontsize=9)
st.pyplot(fig1, clear_figure=True)

st.subheader("Basket Rolling Correlation vs SPY and z-score")
tab_corr, tab_z = st.tabs(["Correlation", "Correlation z-score"])

with tab_corr:
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for i, b in enumerate(basket_rets.columns):
        clr = PASTEL[i % len(PASTEL)]
        aligned = pd.concat([basket_rets[b], spy_ret], axis=1).dropna()
        if aligned.shape[0] >= int(corr_window):
            corr = aligned.iloc[:,0].rolling(int(corr_window)).corr(aligned.iloc[:,1])
            ax2.plot(corr.index, corr, label=b, linewidth=2, color=clr)
    ax2.axhline(0, linewidth=1, color="#999999", linestyle=":")
    ax2.set_title(f"{corr_window}-period rolling correlation")
    ax2.grid(alpha=0.2)
    ax2.legend(loc="best", ncol=2, fontsize=9)
    st.pyplot(fig2, clear_figure=True)

with tab_z:
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    for i, b in enumerate(basket_rets.columns):
        clr = PASTEL[i % len(PASTEL)]
        z, _, _ = rolling_corr_zscore(basket_rets[b], spy_ret, window=int(corr_window))
        if not z.empty:
            ax3.plot(z.index, z, label=b, linewidth=2, color=clr)
    ax3.axhline(0, linewidth=1, color="#999999", linestyle=":")
    ax3.set_title(f"Z-score of rolling correlation (window {corr_window})")
    ax3.grid(alpha=0.2)
    ax3.legend(loc="best", ncol=2, fontsize=9)
    st.pyplot(fig3, clear_figure=True)

# ---------------------------
# Basket Snapshot Table
# ---------------------------
st.subheader("Basket Snapshot Table")
rows = []
for b in basket_rets.columns:
    sr = basket_rets[b].dropna()
    ps = perf_summary(sr)
    dd = max_drawdown((1 + sr).cumprod())
    _, latest_corr, latest_z = rolling_corr_zscore(sr, spy_ret, window=int(corr_window))
    rows.append({
        "Basket": b,
        "YTD %": np.round(ps["YTD"]*100 if pd.notna(ps["YTD"]) else np.nan, 2),
        "1M %": np.round(ps["1M"]*100 if pd.notna(ps["1M"]) else np.nan, 2),
        "3M %": np.round(ps["3M"]*100 if pd.notna(ps["3M"]) else np.nan, 2),
        "6M %": np.round(ps["6M"]*100 if pd.notna(ps["6M"]) else np.nan, 2),
        "1Y %": np.round(ps["1Y"]*100 if pd.notna(ps["1Y"]) else np.nan, 2),
        "Max Drawdown %": np.round(dd*100 if pd.notna(dd) else np.nan, 2),
        "Corr(SPY)": np.round(latest_corr, 3) if pd.notna(latest_corr) else np.nan,
        "Corr z-score": np.round(latest_z, 2) if pd.notna(latest_z) else np.nan
    })
st.dataframe(pd.DataFrame(rows).set_index("Basket"), use_container_width=True)

# ---------------------------
# Factor-Flavor Aggregates
# ---------------------------
if show_flavor_panel:
    st.markdown("### Factor-Flavor Aggregates")
    # Build flavor returns from either all baskets in each flavor or the selected subset
    flavor_rets = build_flavor_returns(
        basket_returns=basket_rets_full,
        selected_baskets=basket_rets.columns.tolist()
    )
    flavor_rets = flavor_rets.loc[basket_rets.index]  # align frequencies
    flavor_cum = (1 + flavor_rets).cumprod()

    # KPIs
    st.subheader("Flavor Overview")
    flv_names = list(flavor_rets.columns)
    flv_cols = st.columns(min(4, len(flv_names)))
    flv_groups = [flv_names[i::4] for i in range(4)]
    for col, group in zip(flv_cols, flv_groups):
        with col:
            for f in group:
                sr = flavor_rets[f].dropna()
                ps = perf_summary(sr)
                dd = max_drawdown((1 + sr).cumprod())
                _, latest_corr, latest_z = rolling_corr_zscore(sr, spy_ret, window=int(corr_window))
                col.metric(
                    label=f,
                    value=f"YTD {ps['YTD']*100:0.1f}%" if pd.notna(ps['YTD']) else "YTD n/a",
                    help=(
                        f"1M {ps['1M']*100:0.1f}%, 3M {ps['3M']*100:0.1f}%, 6M {ps['6M']*100:0.1f}%, "
                        f"1Y {ps['1Y']*100:0.1f}%, Max DD {dd*100:0.1f}%\n"
                        f"Corr(SPY) {latest_corr:0.2f}, Corr z-score {latest_z:0.2f} (window {corr_window})"
                    )
                )

    # Charts
    st.subheader("Flavor Cumulative Performance")
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    for i, f in enumerate(flavor_rets.columns):
        clr = PASTEL[i % len(PASTEL)]
        ax4.plot(flavor_cum.index, flavor_cum[f], label=f, linewidth=2, color=clr)
    ax4.plot(spy_cum.index, spy_cum, label="SPY", linewidth=2, linestyle="--", color="#888888")
    ax4.set_title(f"Cumulative Return by Flavor ({freq} resample)")
    ax4.set_ylabel("Growth of $1")
    ax4.grid(alpha=0.2)
    ax4.legend(loc="best", ncol=2, fontsize=9)
    st.pyplot(fig4, clear_figure=True)

    st.subheader("Flavor Rolling Correlation vs SPY and z-score")
    ftab_corr, ftab_z = st.tabs(["Flavor Correlation", "Flavor Correlation z-score"])

    with ftab_corr:
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        for i, f in enumerate(flavor_rets.columns):
            clr = PASTEL[i % len(PASTEL)]
            aligned = pd.concat([flavor_rets[f], spy_ret], axis=1).dropna()
            if aligned.shape[0] >= int(corr_window):
                corr = aligned.iloc[:,0].rolling(int(corr_window)).corr(aligned.iloc[:,1])
                ax5.plot(corr.index, corr, label=f, linewidth=2, color=clr)
        ax5.axhline(0, linewidth=1, color="#999999", linestyle=":")
        ax5.set_title(f"{corr_window}-period rolling correlation (flavors)")
        ax5.grid(alpha=0.2)
        ax5.legend(loc="best", ncol=2, fontsize=9)
        st.pyplot(fig5, clear_figure=True)

    with ftab_z:
        fig6, ax6 = plt.subplots(figsize=(10, 5))
        for i, f in enumerate(flavor_rets.columns):
            clr = PASTEL[i % len(PASTEL)]
            z, _, _ = rolling_corr_zscore(flavor_rets[f], spy_ret, window=int(corr_window))
            if not z.empty:
                ax6.plot(z.index, z, label=f, linewidth=2, color=clr)
        ax6.axhline(0, linewidth=1, color="#999999", linestyle=":")
        ax6.set_title(f"Z-score of rolling correlation (flavors, window {corr_window})")
        ax6.grid(alpha=0.2)
        ax6.legend(loc="best", ncol=2, fontsize=9)
        st.pyplot(fig6, clear_figure=True)

    # Snapshot table
    st.subheader("Flavor Snapshot Table")
    frows = []
    for f in flavor_rets.columns:
        sr = flavor_rets[f].dropna()
        ps = perf_summary(sr)
        dd = max_drawdown((1 + sr).cumprod())
        _, latest_corr, latest_z = rolling_corr_zscore(sr, spy_ret, window=int(corr_window))
        frows.append({
            "Flavor": f,
            "YTD %": np.round(ps["YTD"]*100 if pd.notna(ps["YTD"]) else np.nan, 2),
            "1M %": np.round(ps["1M"]*100 if pd.notna(ps["1M"]) else np.nan, 2),
            "3M %": np.round(ps["3M"]*100 if pd.notna(ps["3M"]) else np.nan, 2),
            "6M %": np.round(ps["6M"]*100 if pd.notna(ps["6M"]) else np.nan, 2),
            "1Y %": np.round(ps["1Y"]*100 if pd.notna(ps["1Y"]) else np.nan, 2),
            "Max Drawdown %": np.round(dd*100 if pd.notna(dd) else np.nan, 2),
            "Corr(SPY)": np.round(latest_corr, 3) if pd.notna(latest_corr) else np.nan,
            "Corr z-score": np.round(latest_z, 2) if pd.notna(latest_z) else np.nan
        })
    st.dataframe(pd.DataFrame(frows).set_index("Flavor"), use_container_width=True)

# ---------------------------
# About
# ---------------------------
with st.expander("About this dashboard"):
    st.markdown('''
**Methodology**
- Basket return is the equal-weight average of constituent returns each period.
- Flavor return is the equal-weight average of the member baskets' returns.
- Frequency resample uses period-end prices, then percent change.
- Correlation is rolling Pearson correlation with SPY; z-score is normalized over the displayed window.

**Notes**
- Yahoo Finance data via `yfinance` with adjusted close as a proxy for total return.
- ETF proxies appear where appropriate for tradability.
- Toggle `USE_SELECTED_FOR_FLAVOR` to True if you want flavor aggregates to reflect only selected baskets.
''')
