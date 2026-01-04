import io
import datetime as dt
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Home Value to Rent Ratio", layout="wide")
st.title("Home Value to Rent Ratio (Public Proxies, Recreated)")

st.caption(
    "Data sources: Zillow Research CSVs (ZHVI/ZORI), BLS CPI Rent via FRED, Shiller long-run home price series. "
    "Series are stitched via level-matching in overlap windows."
)

# -----------------------------
# Helpers
# -----------------------------
def _get_bytes(url: str, timeout: int = 30) -> bytes:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.content

def try_urls(urls, kind: str) -> bytes:
    last_err = None
    for u in urls:
        try:
            return _get_bytes(u)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to download {kind}. Last error: {last_err}")

def fred_csv(series_id: str) -> pd.DataFrame:
    # No API key needed for fredgraph CSV
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    b = _get_bytes(url)
    df = pd.read_csv(io.BytesIO(b))
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    df = df.replace(".", np.nan)
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df.dropna()

def zillow_wide_to_long(csv_bytes: bytes, region_name: str = "United States") -> pd.Series:
    # Zillow Research time-series CSV format:
    # RegionID, SizeRank, RegionName, RegionType, StateName, ... , YYYY-MM-31, YYYY-MM-31, ...
    df = pd.read_csv(io.BytesIO(csv_bytes))
    if "RegionName" not in df.columns:
        raise ValueError("Unexpected Zillow file format: missing RegionName")

    row = df.loc[df["RegionName"] == region_name]
    if row.empty:
        # Some files use "United States" or "United States of America" or similar, try relaxed match
        row = df.loc[df["RegionName"].str.contains("United States", na=False)]
    if row.empty:
        raise ValueError(f"Could not find {region_name} row in Zillow file")

    row = row.iloc[0]
    date_cols = [c for c in df.columns if c[:4].isdigit() and "-" in c]
    s = row[date_cols].astype(float)
    s.index = pd.to_datetime(date_cols)
    s = s.sort_index()
    s.name = "value"
    return s.dropna()

def splice_levels(left: pd.Series, right: pd.Series, anchor_start: str, anchor_end: str) -> pd.Series:
    """
    Create a continuous series where left covers earlier history and right covers later history.
    We scale left to match right over the anchor window via median ratio.
    """
    anchor_start = pd.to_datetime(anchor_start)
    anchor_end = pd.to_datetime(anchor_end)

    overlap = pd.concat([left, right], axis=1, join="inner")
    overlap.columns = ["left", "right"]
    overlap = overlap[(overlap.index >= anchor_start) & (overlap.index <= anchor_end)].dropna()

    if overlap.empty:
        raise ValueError("No overlap in anchor window for splicing")

    # Robust level match
    scale = (overlap["right"].median() / overlap["left"].median())
    left_scaled = left * scale

    combined = pd.concat([left_scaled[left_scaled.index < right.index.min()], right], axis=0).sort_index()
    return combined

# -----------------------------
# Download + build series
# -----------------------------
@st.cache_data(ttl=24 * 3600)
def build_series():
    # Zillow download URLs are published on Zillow Research data page.
    # We try Nation files first. If Zillow changes naming, the app still runs using FRED fallbacks.
    zhvi_urls = [
        # Nation ZHVI (smoothed, seasonally adjusted) mid-tier all homes
        "https://files.zillowstatic.com/research/public_csvs/zhvi/Nation_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
        "https://files.zillowstatic.com/research/public_csvs/zhvi/Nation_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv?t=1",
        # As a backup, pull Metro file and still try to find a national row if present (often it is not)
        "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    ]

    zori_urls = [
        # Nation ZORI (smoothed, seasonally adjusted) all homes + multifamily
        "https://files.zillowstatic.com/research/public_csvs/zori/Nation_zori_uc_sfrcondomfr_sm_sa_month.csv",
        "https://files.zillowstatic.com/research/public_csvs/zori/Nation_zori_uc_sfrcondomfr_sm_sa_month.csv?t=1",
        # Metro fallback
        "https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfrcondomfr_sm_sa_month.csv",
        "https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfrcondomfr_sm_month.csv",
    ]

    # Shiller Fig3-1 xls link hub is on shillerdata.com; direct file is hosted on img1.wsimg.com.
    # We use the direct host that ShillerData links to.
    shiller_urls = [
        "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/42ecd6a6-687c-4846-914c-a0ab392b648c/Fig3-1%20%281%29.xls",
        "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/42ecd6a6-687c-4846-914c-a0ab392b648c/Fig3-1%20(1).xls",
    ]

    # Pull CPI Rent (BLS) via FRED
    cpi_rent = fred_csv("CUSR0000SEHA").set_index("date")["CUSR0000SEHA"].dropna()
    cpi_rent.name = "cpi_rent_index"

    # Pull Case-Shiller national (optional diagnostic, not required)
    # cs = fred_csv("CSUSHPINSA").set_index("date")["CSUSHPINSA"].dropna()

    # Pull Zillow ZHVI dollars
    zhvi = None
    try:
        zhvi_bytes = try_urls(zhvi_urls, "Zillow ZHVI CSV")
        zhvi = zillow_wide_to_long(zhvi_bytes, region_name="United States")
        zhvi.name = "zhvi_dollars"
    except Exception:
        # Fallback: FRED has a Zillow ZHVI series for the U.S. (dollars, monthly, smoothed SA)
        zhvi_fred = fred_csv("USAUCSFRCONDOSMSAMID").set_index("date")["USAUCSFRCONDOSMSAMID"].dropna()
        zhvi = zhvi_fred
        zhvi.name = "zhvi_dollars_fred"

    # Pull Zillow ZORI dollars
    zori = None
    try:
        zori_bytes = try_urls(zori_urls, "Zillow ZORI CSV")
        zori = zillow_wide_to_long(zori_bytes, region_name="United States")
        zori.name = "zori_dollars"
    except Exception:
        zori = None

    # Pull Shiller long-run home price series (monthly, index), then splice into ZHVI dollars at 2000
    sh_b = try_urls(shiller_urls, "Shiller Fig3-1 XLS")

    # Shiller xls structure can vary slightly by version. We parse flexibly:
    sh = pd.read_excel(io.BytesIO(sh_b), sheet_name=0)
    sh.columns = [str(c).strip() for c in sh.columns]

    # He typically includes Year, Month, and a nominal home price index.
    # We detect year/month columns and a "Home Price" column by fuzzy match.
    col_year = next((c for c in sh.columns if c.lower() in ["year", "yr"]), None)
    col_month = next((c for c in sh.columns if c.lower() in ["month", "mo"]), None)

    if col_year is None or col_month is None:
        # Sometimes first two columns are Year, Month without labels
        sh2 = sh.copy()
        sh2.columns = [f"c{i}" for i in range(len(sh2.columns))]
        col_year, col_month = "c0", "c1"
        sh = sh2

    # Pick a plausible home price index column
    home_candidates = [c for c in sh.columns if "home" in c.lower() and "price" in c.lower()]
    if not home_candidates:
        # fallback: common label used in Fig3-1 files is "Home Price Index" or similar,
        # else use the first numeric column after year/month as last resort
        numeric_cols = [c for c in sh.columns if c not in [col_year, col_month]]
        home_col = numeric_cols[0]
    else:
        home_col = home_candidates[0]

    sh = sh[[col_year, col_month, home_col]].dropna()
    sh[col_year] = pd.to_numeric(sh[col_year], errors="coerce")
    sh[col_month] = pd.to_numeric(sh[col_month], errors="coerce")
    sh[home_col] = pd.to_numeric(sh[home_col], errors="coerce")
    sh = sh.dropna()

    sh["date"] = pd.to_datetime(
        sh[col_year].astype(int).astype(str) + "-" + sh[col_month].astype(int).astype(str) + "-01"
    )
    sh = sh.set_index("date").sort_index()
    sh_home_index = sh[home_col].copy()
    sh_home_index.name = "shiller_home_index"

    # Convert Shiller index to a "dollar-like" level by splicing to ZHVI in 2000-2002 overlap
    # This gives you a continuous home-value level series from 1980s onward.
    home_level = splice_levels(
        left=sh_home_index,
        right=zhvi,
        anchor_start="2000-01-01",
        anchor_end="2002-12-01",
    )

    # Rent level series
    # Long history proxy: CPI rent index. From mid-2010s onward, ZORI is more “market rent”.
    # If ZORI is available, splice CPI index into ZORI dollars using 2015-2017 anchor.
    if zori is not None:
        rent_level = splice_levels(
            left=cpi_rent,
            right=zori,
            anchor_start="2015-01-01",
            anchor_end="2017-12-01",
        )
    else:
        # If ZORI download fails, we keep CPI rent for entire history (still a usable proxy).
        rent_level = cpi_rent.copy()

    # Align and compute ratio
    combined = pd.concat([home_level, rent_level], axis=1, join="inner").dropna()
    combined.columns = ["home_level", "rent_level"]
    combined["ratio"] = combined["home_level"] / combined["rent_level"]

    # Annualize to match the style of your chart
    annual = combined["ratio"].resample("Y").mean()
    annual.index = annual.index.year
    annual.name = "home_value_to_rent_ratio"

    return annual, combined

annual_ratio, monthly = build_series()

# -----------------------------
# Controls
# -----------------------------
colA, colB, colC = st.columns([1, 1, 2])
with colA:
    downturn_low = st.number_input("Downturn low reference (x)", value=12.8, step=0.1)
with colB:
    use_median = st.checkbox("Show median line", value=True)
with colC:
    st.write("")

median_ratio = float(annual_ratio.median())
latest_year = int(annual_ratio.index.max())
latest_ratio = float(annual_ratio.loc[latest_year])

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(annual_ratio.index, annual_ratio.values, linewidth=2.5)

if use_median:
    ax.axhline(median_ratio, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(annual_ratio.index.min() + 1, median_ratio + 0.05, f"Median: {median_ratio:.1f}x", alpha=0.8)

ax.axhline(downturn_low, linestyle="--", linewidth=1.5, alpha=0.9)
ax.text(annual_ratio.index.min() + 1, downturn_low - 0.35, f"Low in downturns: {downturn_low:.1f}x", alpha=0.9)

ax.scatter([latest_year], [latest_ratio], s=80, zorder=5)
ax.text(latest_year + 0.3, latest_ratio, "Now", fontsize=11, weight="bold")

ax.set_title("Home Value to Rent Ratio (proxy recreation)", fontsize=14)
ax.set_ylabel("Home Value / Rent (x)")
ax.set_xlabel("")
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

st.pyplot(fig)

# -----------------------------
# Diagnostics
# -----------------------------
with st.expander("Data diagnostics"):
    st.write(f"Annual series range: {annual_ratio.index.min()} to {annual_ratio.index.max()}")
    st.write(f"Latest: {latest_year} = {latest_ratio:.2f}x")
    st.write("Monthly stitched series (tail):")
    st.dataframe(monthly.tail(24))

st.markdown(
    """
Interpretation-wise, treat this as a valuation regime indicator, not a precision instrument. The joins matter.
If ZORI downloads cleanly, the post-2015 rent leg becomes much closer to market asking rents; if ZORI is unavailable, CPI Rent stays a slower-moving proxy and the 2020–2022 spike will usually look larger.
"""
)
