import io
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import streamlit as st
from matplotlib.ticker import MultipleLocator
from PIL import Image

# We generate the image ourselves; keep Pillow from blocking trusted output
Image.MAX_IMAGE_PIXELS = None

plt.style.use("default")

CACHE_TTL_SECONDS = 3600
MAX_RETRIES = 4

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Home Value to Rent Ratio", layout="wide")

LOGO_PATH = Path("/mnt/data/0ea02e99-f067-4315-accc-0d2bbd3ee87d.png")
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=70)

st.title("Home Value to Rent Ratio")
st.subheader("Public proxy recreation (FRED)")

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        This panel recreates the **Home Value to Rent Ratio** using stable, public FRED series.

        What it is:

        • A valuation multiple for housing analogous to a P/E style lens  
        • Elevated readings flag price outpacing rent fundamentals across cycles  
        • Mean reversion can happen through prices, rents, or time  

        Sources (FRED):

        • Home prices: Case-Shiller U.S. National HPI (CSUSHPINSA)  
        • Rents: CPI Rent of Primary Residence (CUSR0000SEHA)  
        • Recessions: NBER recession indicator (USREC)  
        • Mortgage rates: 30Y fixed (MORTGAGE30US)
        """
    )
    st.markdown("---")
    show_recessions = st.checkbox("Shade recessions (NBER)", value=True)
    show_mortgage = st.checkbox("Overlay 30Y mortgage rate", value=True)

    st.markdown("---")
    downturn_floor = st.number_input("Downturn reference (x)", value=12.8, step=0.1)
    show_median = st.checkbox("Show long-run median", value=True)

st.markdown("<hr style='margin-top:2px; margin-bottom:15px;'>", unsafe_allow_html=True)

# ── Data helpers ─────────────────────────────────────────────────────────
def fetch_bytes(url: str, timeout: int = 30) -> bytes:
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    r.raise_for_status()
    return r.content

def fred_series(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    last_err = None
    delay = 1.0
    for _ in range(MAX_RETRIES):
        try:
            df = pd.read_csv(io.BytesIO(fetch_bytes(url)))
            if df.shape[1] < 2:
                raise ValueError("FRED returned an unexpected format.")
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            s = df.set_index("date")["value"].dropna()
            if s.empty:
                raise ValueError("FRED returned an empty series.")
            return s
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"Failed to fetch FRED series {series_id}: {last_err}")

def to_monthly(s: pd.Series, how: str = "mean") -> pd.Series:
    if how == "mean":
        return s.resample("MS").mean().dropna()
    if how == "last":
        return s.resample("MS").last().dropna()
    raise ValueError("how must be 'mean' or 'last'")

def recession_spans(usrec_monthly: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    x = usrec_monthly.copy().dropna()
    x = (x > 0.5).astype(int)

    spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    in_rec = False
    start: Optional[pd.Timestamp] = None

    for t, val in x.items():
        if val == 1 and not in_rec:
            in_rec = True
            start = t
        elif val == 0 and in_rec:
            in_rec = False
            spans.append((start, t))
            start = None

    if in_rec and start is not None:
        spans.append((start, x.index.max() + pd.offsets.MonthBegin(1)))

    return spans

# ── Build series ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def build_all():
    # Monthly series
    cs_m = to_monthly(fred_series("CSUSHPINSA"), how="mean")
    rent_m = to_monthly(fred_series("CUSR0000SEHA"), how="mean")

    # Ratio (monthly), then annual mean (Reventure-like annual points)
    df = pd.concat([cs_m, rent_m], axis=1, join="inner").dropna()
    df.columns = ["home_index", "rent_index"]
    df["ratio"] = df["home_index"] / df["rent_index"]

    ratio_y = df["ratio"].resample("Y").mean()
    ratio_y.index = ratio_y.index.year

    # Recessions (monthly)
    usrec_m = to_monthly(fred_series("USREC"), how="mean")
    usrec_m = usrec_m.reindex(df.index).ffill().dropna()
    spans = recession_spans(usrec_m)

    # Mortgage (weekly -> monthly mean -> annual mean)
    mort = fred_series("MORTGAGE30US")
    mort_m = to_monthly(mort, how="mean")
    mort_y = mort_m.resample("Y").mean()
    mort_y.index = mort_y.index.year

    return ratio_y, spans, mort_y, usrec_m

ratio_y, rec_spans, mort_y, usrec_m = build_all()

# ── Metrics ──────────────────────────────────────────────────────────────
median_val = float(ratio_y.median())
latest_year = int(ratio_y.index.max())
latest_ratio = float(ratio_y.loc[latest_year])
latest_mort = float(mort_y.loc[latest_year]) if latest_year in mort_y.index else np.nan
in_recession = bool(usrec_m.iloc[-1] > 0.5) if not usrec_m.empty else False

m1, m2, m3 = st.columns(3)
m1.metric(f"{latest_year} Ratio", f"{latest_ratio:.2f}x")
m2.metric(f"{latest_year} 30Y Mortgage", "N/A" if np.isnan(latest_mort) else f"{latest_mort:.2f}%")
m3.metric("Recession (USREC)", "Yes" if in_recession else "No")

st.markdown("<hr style='margin-top:0; margin-bottom:6px;'>", unsafe_allow_html=True)

# ── Plot ─────────────────────────────────────────────────────────────────
years = ratio_y.index.to_numpy(dtype=float)
vals = ratio_y.values.astype(float)

# Keep pixel dimensions sane to prevent Streamlit/PIL issues
fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=110)

# Shade recessions first (so points/line sit on top)
if show_recessions and rec_spans:
    for start, end in rec_spans:
        # annual axis, approximate span
        x0 = start.year + (start.month - 1) / 12.0
        x1 = end.year + (end.month - 1) / 12.0
        ax.axvspan(x0, x1, alpha=0.10, zorder=0)

# Line + dots (your spec)
ax.plot(years, vals, lw=2.4, alpha=0.95, zorder=2)
ax.scatter(years, vals, s=28, zorder=3)

# Highlight latest point
ax.scatter([latest_year], [latest_ratio], s=85, zorder=4)
ax.text(latest_year + 0.35, latest_ratio, "Now", weight="bold", fontsize=10)

# Reference lines
if show_median:
    ax.axhline(median_val, ls="--", lw=1.2, alpha=0.75, zorder=1)
    ax.text(years.min() + 1, median_val + 0.05, f"Median: {median_val:.1f}x", alpha=0.85)

ax.axhline(downturn_floor, ls="--", lw=1.2, alpha=0.85, zorder=1)
ax.text(years.min() + 1, downturn_floor - 0.35, f"Low in downturns: {downturn_floor:.1f}x", alpha=0.90)

# Mortgage overlay (rhs)
ax2 = None
if show_mortgage:
    mort_common = mort_y.reindex(ratio_y.index).dropna()
    if not mort_common.empty:
        ax2 = ax.twinx()
        ax2.plot(
            mort_common.index.to_numpy(dtype=float),
            mort_common.values.astype(float),
            ls=":",
            lw=2.0,
            alpha=0.9,
        )
        ax2.set_ylabel("30Y Mortgage Rate (%)", fontsize=12)
        ax2.yaxis.set_major_locator(MultipleLocator(1.0))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax2.spines["top"].set_visible(False)

# Styling
ax.set_title("Home Value to Rent Ratio (FRED proxy)", fontsize=16, weight="bold")
ax.set_ylabel("Home Value / Rent (x)", fontsize=12)
ax.set_xlabel("")
ax.grid(axis="y", ls=":", lw=0.7, alpha=0.55)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Limits with padding
ymin, ymax = float(np.nanmin(vals)), float(np.nanmax(vals))
pad = 0.06 * (ymax - ymin) if ymax > ymin else 0.5
ax.set_ylim(ymin - pad, ymax + pad)

# Tight layout without bbox hacks that can blow up canvas
fig.tight_layout(pad=1.0)

# Render to PNG ourselves (stable sizing), then st.image
buf = io.BytesIO()
fig.savefig(buf, format="png")  # no bbox_inches="tight"
plt.close(fig)
buf.seek(0)

st.image(buf.getvalue(), use_container_width=True)
st.caption("© 2026 AD Fund Management LP")

with st.expander("Diagnostics (data tail)"):
    st.write("Annual ratio (tail):")
    st.dataframe(ratio_y.tail(15))
    st.write("Annual mortgage rate (tail):")
    st.dataframe(mort_y.tail(15))
