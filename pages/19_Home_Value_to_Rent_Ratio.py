import io
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
        This tool recreates the **Home Value to Rent Ratio** using transparent,
        publicly available housing data and robust proxy stitching.

        What it shows:

        • A valuation multiple for housing analogous to a P/E ratio  
        • Historical housing bubbles and post-crisis resets  
        • Long-run mean reversion behavior across cycles  

        Data construction:

        • Home prices: Case-Shiller National Index pre-2000, Zillow ZHVI post-2000  
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
def fetch_bytes(url: str, headers=None) -> bytes:
    r = requests.get(
        url,
        headers=headers or {"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    r.raise_for_status()
    return r.content

def fred_series(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(io.BytesIO(fetch_bytes(url)))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"].dropna()

def zillow_zhvi_with_fallback() -> pd.Series:
    urls = [
        "https://files.zillowstatic.com/research/public_csvs/zhvi/"
        "Nation_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
        "https://files.zillowstatic.com/research/public_csvs/zhvi/"
        "Nation_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv?t=1",
    ]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0 Safari/537.36"
        )
    }

    for url in urls:
        try:
            df = pd.read_csv(io.BytesIO(fetch_bytes(url, headers=headers)))
            row = df.loc[df["RegionName"] == "United States"].iloc[0]
            cols = [c for c in df.columns if c[:4].isdigit()]
            s = row[cols].astype(float)
            s.index = pd.to_datetime(cols)
            return s.sort_index().dropna()
        except Exception:
            continue

    st.warning("Zillow ZHVI unavailable. Falling back to FRED ZHVI proxy.")
    return fred_series("USAUCSFRCONDOSMSAMID")

def zillow_zori_with_fallback() -> pd.Series:
    urls = [
        "https://files.zillowstatic.com/research/public_csvs/zori/"
        "Nation_zori_uc_sfrcondomfr_sm_sa_month.csv",
        "https://files.zillowstatic.com/research/public_csvs/zori/"
        "Nation_zori_uc_sfrcondomfr_sm_sa_month.csv?t=1",
    ]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0 Safari/537.36"
        )
    }

    for url in urls:
        try:
            df = pd.read_csv(io.BytesIO(fetch_bytes(url, headers=headers)))
            row = df.loc[df["RegionName"] == "United States"].iloc[0]
            cols = [c for c in df.columns if c[:4].isdigit()]
            s = row[cols].astype(float)
            s.index = pd.to_datetime(cols)
            return s.sort_index().dropna()
        except Exception:
            continue

    st.warning("Zillow ZORI unavailable. Falling back to CPI Rent proxy only.")
    return None

def splice(left: pd.Series, right: pd.Series, start: str, end: str) -> pd.Series:
    overlap = pd.concat([left, right], axis=1).dropna().loc[start:end]
    scale = overlap.iloc[:, 1].median() / overlap.iloc[:, 0].median()
    left_adj = left * scale
    return pd.concat([left_adj[left_adj.index < right.index.min()], right]).sort_index()

# ── Data build ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def build_ratio():
    # Home prices
    case_shiller = fred_series("CSUSHPINSA")
    zhvi = zillow_zhvi_with_fallback()
    home_level = splice(case_shiller, zhvi, "2000-01-01", "2002-12-01")

    # Rents
    cpi_rent = fred_series("CUSR0000SEHA")
    zori = zillow_zori_with_fallback()

    if zori is not None:
        rent_level = splice(cpi_rent, zori, "2015-01-01", "2017-12-01")
    else:
        rent_level = cpi_rent.copy()

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
ax.text(
    ratio.index.min() + 1,
    downturn_floor - 0.35,
    f"Downturn floor: {downturn_floor:.1f}x",
)

ax.scatter([latest_year], [latest_val], s=90, zorder=5)
ax.text(latest_year + 0.4, latest_val, "Now", weight="bold")

ax.set_title(
    "Home Value to Rent Ratio (Public Proxy Recreation)",
    fontsize=16,
    weight="bold",
)
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

st.caption("© 2026 AD Fund Management LP")
