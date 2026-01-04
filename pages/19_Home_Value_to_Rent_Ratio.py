import io
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import streamlit as st
from matplotlib.ticker import MultipleLocator

plt.style.use("default")

CACHE_TTL_SECONDS = 3600

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Home Value to Rent Ratio", layout="wide")

LOGO_PATH = Path("/mnt/data/0ea02e99-f067-4315-accc-0d2bbd3ee87d.png")
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=70)

st.title("Home Value to Rent Ratio")
st.subheader("A valuation regime lens for U.S. housing")

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        This tool recreates the long-run **Home Value to Rent Ratio** using only
        publicly available data and transparent proxy stitching.

        What it shows:

        • A valuation multiple for housing analogous to P/E for equities  
        • Historical bubbles where prices decoupled from rent fundamentals  
        • Downturn floors where mean reversion historically stabilized  

        Data construction:

        • Home prices: Shiller national index pre-2000, Zillow ZHVI post-2000  
        • Rents: CPI Rent of Primary Residence pre-2015, Zillow ZORI post-2015  
        • Series are level-matched in overlap windows to preserve continuity  

        This is a **regime indicator**, not a timing signal.
        """
    )
    st.markdown("---")
    downturn_floor = st.number_input("Downturn reference (x)", value=12.8, step=0.1)
    show_median = st.checkbox("Show long-run median", value=True)

st.markdown("<hr style='margin-top:2px; margin-bottom:15px;'>", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────
def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.content

def fred_series(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(io.BytesIO(fetch_bytes(url)))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"].dropna()

def zillow_national_series(urls) -> pd.Series:
    last_err = None
    for u in urls:
        try:
            df = pd.read_csv(io.BytesIO(fetch_bytes(u)))
            row = df.loc[df["RegionName"] == "United States"].iloc[0]
            cols = [c for c in df.columns if c[:4].isdigit()]
            s = row[cols].astype(float)
            s.index = pd.to_datetime(cols)
            return s.sort_index().dropna()
        except Exception as e:
            last_err = e
    raise RuntimeError(last_err)

def splice(left: pd.Series, right: pd.Series, start: str, end: str) -> pd.Series:
    overlap = pd.concat([left, right], axis=1).dropna()
    overlap = overlap.loc[start:end]
    scale = overlap.iloc[:, 1].median() / overlap.iloc[:, 0].median()
    left_adj = left * scale
    return pd.concat([left_adj[left_adj.index < right.index.min()], right]).sort_index()

# ── Data build ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def build_ratio():
    # Shiller home price index
    shiller_url = "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/42ecd6a6-687c-4846-914c-a0ab392b648c/Fig3-1%20(1).xls"
    sh = pd.read_excel(fetch_bytes(shiller_url))
    sh.columns = [str(c).strip() for c in sh.columns]
    sh["date"] = pd.to_datetime(sh.iloc[:, 0].astype(int).astype(str) + "-" +
                                sh.iloc[:, 1].astype(int).astype(str) + "-01")
    sh_home = pd.to_numeric(sh.iloc[:, 2], errors="coerce")
    sh_home.index = sh["date"]
    sh_home = sh_home.dropna()

    # Zillow ZHVI
    zhvi_urls = [
        "https://files.zillowstatic.com/research/public_csvs/zhvi/Nation_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
    ]
    zhvi = zillow_national_series(zhvi_urls)

    home_level = splice(sh_home, zhvi, "2000-01-01", "2002-12-01")

    # Rent series
    cpi_rent = fred_series("CUSR0000SEHA")

    zori_urls = [
        "https://files.zillowstatic.com/research/public_csvs/zori/Nation_zori_uc_sfrcondomfr_sm_sa_month.csv"
    ]
    zori = zillow_national_series(zori_urls)

    rent_level = splice(cpi_rent, zori, "2015-01-01", "2017-12-01")

    df = pd.concat([home_level, rent_level], axis=1).dropna()
    df.columns = ["home", "rent"]
    ratio = (df["home"] / df["rent"]).resample("Y").mean()
    ratio.index = ratio.index.year
    return ratio

ratio = build_ratio()

median_val = float(ratio.median())
latest_year = int(ratio.index.max())
latest_val = float(ratio.loc[latest_year])

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(ratio.index, ratio.values, lw=3.0)

if show_median:
    ax.axhline(median_val, ls="--", lw=1.5, alpha=0.8)
    ax.text(ratio.index.min() + 1, median_val + 0.05, f"Median: {median_val:.1f}x")

ax.axhline(downturn_floor, ls="--", lw=1.5, alpha=0.9)
ax.text(ratio.index.min() + 1, downturn_floor - 0.35, f"Downturn floor: {downturn_floor:.1f}x")

ax.scatter([latest_year], [latest_val], s=90, zorder=5)
ax.text(latest_year + 0.4, latest_val, "Now", weight="bold")

ax.set_title("Home Value to Rent Ratio (Public Proxy Recreation)", fontsize=16, weight="bold")
ax.set_ylabel("Home Value / Rent (x)", fontsize=13)
ax.set_xlabel("")
ax.set_ylim(12.3, 20.2)

ax.grid(axis="y", ls=":", lw=0.7, alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.yaxis.set_major_locator(MultipleLocator(1.0))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))

plt.tight_layout()
st.pyplot(fig)

# ── Footer ────────────────────────────────────────────────────────────────
st.caption("© 2026 AD Fund Management LP")
