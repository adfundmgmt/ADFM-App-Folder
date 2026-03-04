import datetime as dt
import io
import time
import warnings
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter, MaxNLocator
from matplotlib import gridspec

plt.style.use("default")
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ------------------------------------------------
# Streamlit UI
# ------------------------------------------------

st.set_page_config(page_title="Seasonality Dashboard", layout="wide")
st.title("Monthly Seasonality Explorer")

# ------------------------------------------------
# Data Download
# ------------------------------------------------

def _yf_download(symbol: str, start: str, end: str, retries: int = 3) -> Optional[pd.Series]:

    for n in range(retries):

        df = yf.download(symbol,start=start,end=end,auto_adjust=True,progress=False)

        if not df.empty and "Close" in df:
            s = df["Close"].dropna()
            s = s[~s.index.duplicated()]
            return s

        time.sleep(2*(n+1))

    return None


@st.cache_data(ttl=3600)
def fetch_prices(symbol,start,end):

    series = _yf_download(symbol,start,end)

    if series is None:
        return None

    return series


# ------------------------------------------------
# Monthly return decomposition
# ------------------------------------------------

def _intra_month_halves(prices: pd.Series) -> pd.DataFrame:

    periods = prices.index.to_period("M")

    grouped = prices.groupby(periods)

    rows = []

    for m,data in grouped:

        if len(data)<3:
            continue

        prev = prices[periods==(m-1)]

        if prev.empty:
            continue

        prev_eom = prev.iloc[-1]

        mid = data.iloc[len(data)//2-1]

        last = data.iloc[-1]

        total = (last/prev_eom-1)*100
        h1 = (mid/prev_eom-1)*100
        h2 = total-h1

        rows.append({
            "period":m,
            "year":m.year,
            "month":m.month,
            "total_ret":float(total),
            "h1_ret":float(h1),
            "h2_ret":float(h2)
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df.set_index("period",inplace=True)

    return df


# ------------------------------------------------
# Seasonality stats
# ------------------------------------------------

@st.cache_data(ttl=3600)
def seasonal_stats(prices,start_year,end_year):

    halves = _intra_month_halves(prices)

    if halves.empty:
        stats = pd.DataFrame(index=range(1,13))
        stats["label"] = MONTH_LABELS
        return stats

    halves = halves[(halves.year>=start_year)&(halves.year<=end_year)]

    halves["total_ret"] = pd.to_numeric(halves["total_ret"],errors="coerce")
    halves["h1_ret"] = pd.to_numeric(halves["h1_ret"],errors="coerce")
    halves["h2_ret"] = pd.to_numeric(halves["h2_ret"],errors="coerce")

    stats = pd.DataFrame(index=range(1,13))
    stats["label"] = MONTH_LABELS

    g = halves.groupby("month")

    stats["mean_total"] = g["total_ret"].mean()
    stats["mean_h1"] = g["h1_ret"].mean()
    stats["mean_h2"] = g["h2_ret"].mean()

    stats["hit_rate"] = ((g["total_ret"].apply(lambda x:(x>0).sum()) /
                         g["total_ret"].count()) * 100)

    stats["p05"] = g["total_ret"].quantile(.05)
    stats["p95"] = g["total_ret"].quantile(.95)

    stats["years_observed"] = g["year"].nunique()

    return stats


# ------------------------------------------------
# Plot Seasonality
# ------------------------------------------------

def plot_seasonality(stats,title):

    df = stats.dropna()

    x = np.arange(len(df))

    fig = plt.figure(figsize=(12.5,7.5),dpi=200)

    gs = gridspec.GridSpec(2,1,height_ratios=[3,1])

    ax = fig.add_subplot(gs[0])

    ax.bar(x,df.mean_h1,color="#62c38e")

    ax.bar(x,df.mean_h2,bottom=df.mean_h1,hatch="///",color="#62c38e")

    yerr = np.vstack([df.mean_total-df.p05,df.p95-df.mean_total])

    ax.errorbar(x,df.mean_total,yerr=yerr,fmt="none",ecolor="gray")

    ax.set_xticks(x,df.label)

    ax.set_ylabel("Return (%)")

    ax.yaxis.set_major_formatter(PercentFormatter())

    ax.grid(axis="y",linestyle="--",alpha=.5)

    ax2 = ax.twinx()

    ax2.scatter(x,df.hit_rate,color="black",marker="D")

    ax2.set_ylim(0,100)

    ax2.set_ylabel("Hit Rate")

    ax2.yaxis.set_major_formatter(PercentFormatter())

    # table

    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")

    table = ax_tbl.table(
        cellText=[
            [f"{v:+.1f}%" for v in df.mean_h1],
            [f"{v:+.1f}%" for v in df.mean_h2],
            [f"{v:+.1f}%" for v in df.mean_total]
        ],
        rowLabels=["1H","2H","Total"],
        colLabels=df.label,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    fig.suptitle(title)

    buf = io.BytesIO()

    fig.savefig(buf,bbox_inches="tight")

    plt.close(fig)

    buf.seek(0)

    return buf


# ------------------------------------------------
# Intra-month paths
# ------------------------------------------------

def build_month_paths(prices,month,start,end):

    px = prices[(prices.index.year>=start)&(prices.index.year<=end)]

    paths = {}

    for y in sorted(px.index.year.unique()):

        m = px[(px.index.year==y)&(px.index.month==month)]

        if len(m)<3:
            continue

        prev = px[(px.index.year==y if month>1 else y-1)&
                  (px.index.month==(month-1 if month>1 else 12))]

        if prev.empty:
            continue

        prev_eom = prev.iloc[-1]

        r = (m/prev_eom-1)*100

        r.index = range(1,len(r)+1)

        r.loc[0]=0

        r = r.sort_index()

        paths[y]=r

    if not paths:
        return pd.DataFrame(),pd.Series()

    max_days=max(s.index.max() for s in paths.values())

    idx=range(0,max_days+1)

    df=pd.DataFrame(index=idx)

    for y,s in paths.items():
        df[y]=s.reindex(idx).ffill()

    avg=df.mean(axis=1)

    return df,avg


# ------------------------------------------------
# Forward distribution
# ------------------------------------------------

def forward_distribution(df_paths,day):

    if day not in df_paths.index:
        return None

    end = df_paths.index.max()

    fwd = df_paths.loc[end] - df_paths.loc[day]

    fwd = fwd.dropna()

    if fwd.empty:
        return None

    return {
        "mean":fwd.mean(),
        "median":fwd.median(),
        "hit":(fwd>0).mean()*100,
        "p05":np.percentile(fwd,5),
        "p95":np.percentile(fwd,95),
        "n":len(fwd)
    }


# ------------------------------------------------
# Controls
# ------------------------------------------------

col1,col2,col3 = st.columns([2,1,1])

with col1:
    symbol = st.text_input("Ticker","^SPX")

with col2:
    start_year = st.number_input("Start year",2020)

with col3:
    end_year = st.number_input("End year",dt.datetime.today().year)

prices = fetch_prices(symbol,f"{start_year}-01-01",f"{end_year}-12-31")

if prices is None:
    st.error("No data found")
    st.stop()

stats = seasonal_stats(prices,start_year,end_year)

buf = plot_seasonality(stats,f"{symbol} Seasonality")

st.image(buf,use_container_width=True)


# ------------------------------------------------
# Intra-month curve
# ------------------------------------------------

st.subheader("Intra-Month Seasonality Curve")

month = st.selectbox(
    "Month",
    list(range(1,13)),
    format_func=lambda m: MONTH_LABELS[m-1]
)

df_paths,avg = build_month_paths(prices,month,start_year,end_year)

fig = plt.figure(figsize=(12,7),dpi=200)

ax = fig.add_subplot(111)

std = df_paths.std(axis=1)

ax.fill_between(avg.index,(avg-std),(avg+std),alpha=.25)

ax.plot(avg.index,avg,color="black",linewidth=2)

today = pd.Timestamp.today()

if today.month == month:

    trading_day = min(len(avg)-1,today.day)

    stats_fwd = forward_distribution(df_paths,trading_day)

    if stats_fwd:

        txt = f"""Forward return distribution (today → month end)

Mean {stats_fwd["mean"]:+.2f}%
Median {stats_fwd["median"]:+.2f}%
Hit Rate {stats_fwd["hit"]:.0f}%
5th pct {stats_fwd["p05"]:+.2f}%
95th pct {stats_fwd["p95"]:+.2f}%
Observations {stats_fwd["n"]}"""

        ax.text(.02,.05,txt,transform=ax.transAxes,
                bbox=dict(boxstyle="round",fc="white"))

buf = io.BytesIO()

fig.savefig(buf,bbox_inches="tight")

plt.close(fig)

buf.seek(0)

st.image(buf,use_container_width=True)

st.caption("© 2026 AD Fund Management LP")
