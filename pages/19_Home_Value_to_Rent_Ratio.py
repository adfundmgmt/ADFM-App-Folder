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

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        This panel recreates the long-run **Home Value to Rent Ratio** using stable, public FRED series.

        Sources (FRED):

        • Home prices: Case-Shiller U.S. National HPI (CSUSHPINSA)  
        • Rents: CPI Rent of Primary Residence (CUSR0000SEHA)  
        • Recessions: NBER recession indicator (USREC)  
        • Mortgage rates: 30Y fixed (MORTGAGE30US)

        Proxy note:

        The ratio is built from indices. To make the output comparable to the “x multiple” framing,
        the series is scaled so its long-run median equals 13.9x.
        """
    )
    st.markdown("---")
    show_recessions = st.checkbox("Shade recessions (NBER)", value=True)
    show_mortgage = st.checkbox("Overlay 30Y mortgage rate", value=True)

    st.markdown("---")
    target_median = st.number_input("Target median (x)", value=13.9, step=0.1)
    downturn_floor = st.number_input("Downturn reference (x)", value=12.8, step=0.1)
    show_median = st.checkbox("Show median line", value=True)

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
def build_all(target_median_local: float):
    cs_m = to_monthly(fred_series("CSUSHPINSA"), how="mean")
    rent_m = to_monthly(fred_series("CUSR0000SEHA"), how="mean")

    df = pd.concat([cs_m, rent_m], axis=1, join="inner").dropna()
    df.columns = ["home_index", "rent_index"]
    df["ratio_raw"] = df["home_index"] / df["rent_index"]

    ratio_y = df["ratio_raw"].resample("Y").mean()
    ratio_y.index = ratio_y.index.year

    raw_median = float(ratio_y.median())
    scale = float(target_median_local) / raw_median if np.isfinite(raw_median) and raw_median != 0 else 1.0
    ratio_y_scaled = ratio_y * scale

    usrec_m = to_monthly(fred_series("USREC"), how="mean")
    usrec_m = usrec_m.reindex(df.index).ffill().dropna()
    spans = recession_spans(usrec_m)

    mort = fred_series("MORTGAGE30US")
    mort_m = to_monthly(mort, how="mean")
    mort_y = mort_m.resample("Y").mean()
    mort_y.index = mort_y.index.year

    # YoY inflation chart inputs (annualized YoY from monthly indices)
    home_yoy_m = df["home_index"].pct_change(12) * 100.0
    rent_yoy_m = df["rent_index"].pct_change(12) * 100.0

    home_yoy_y = home_yoy_m.resample("Y").mean()
    home_yoy_y.index = home_yoy_y.index.year

    rent_yoy_y = rent_yoy_m.resample("Y").mean()
    rent_yoy_y.index = rent_yoy_y.index.year

    spread_y = (home_yoy_y - rent_yoy_y).rename("Home YoY minus Rent YoY")

    return ratio_y_scaled, spans, mort_y, home_yoy_y, rent_yoy_y, spread_y

ratio_y, rec_spans, mort_y, home_yoy_y, rent_yoy_y, spread_y = build_all(float(target_median))

# ── Chart 1: Ratio (dots + connecting line) ───────────────────────────────
years = ratio_y.index.to_numpy(dtype=float)
vals = ratio_y.values.astype(float)

latest_year = int(ratio_y.index.max())
latest_ratio = float(ratio_y.loc[latest_year])
median_val = float(ratio_y.median())

fig1, ax = plt.subplots(figsize=(13.5, 6.2), dpi=110)

if show_recessions and rec_spans:
    for start, end in rec_spans:
        x0 = start.year + (start.month - 1) / 12.0
        x1 = end.year + (end.month - 1) / 12.0
        ax.axvspan(x0, x1, alpha=0.10, zorder=0)

# Ratio line and dots
ax.plot(years, vals, lw=2.4, alpha=0.95, label="Home Value / Rent (lhs)", zorder=2)
ax.scatter(years, vals, s=30, zorder=3)

# Highlight latest
ax.scatter([latest_year], [latest_ratio], s=90, zorder=4, label="Now")
ax.text(latest_year + 0.35, latest_ratio, "Now", weight="bold", fontsize=10)

# Reference lines
if show_median:
    ax.axhline(median_val, ls="--", lw=1.2, alpha=0.75, label=f"Median ({median_val:.1f}x)", zorder=1)

ax.axhline(downturn_floor, ls="--", lw=1.2, alpha=0.85, label=f"Downturn low ({downturn_floor:.1f}x)", zorder=1)

# Mortgage overlay
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
            label="30Y Mortgage (rhs)",
        )
        ax2.set_ylabel("30Y Mortgage Rate (%)", fontsize=12)
        ax2.yaxis.set_major_locator(MultipleLocator(1.0))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax2.spines["top"].set_visible(False)

ax.set_title("Home Value to Rent Ratio (FRED proxy)", fontsize=16, weight="bold")
ax.set_ylabel("Home Value / Rent (x)", fontsize=12)
ax.set_xlabel("")
ax.grid(axis="y", ls=":", lw=0.7, alpha=0.55)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ymin, ymax = float(np.nanmin(vals)), float(np.nanmax(vals))
pad = 0.06 * (ymax - ymin) if ymax > ymin else 0.5
ax.set_ylim(ymin - pad, ymax + pad)

# Legend combining both axes
handles1, labels1 = ax.get_legend_handles_labels()
handles, labels = handles1, labels1
if ax2 is not None:
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2

if handles:
    ax.legend(handles, labels, loc="best", frameon=False, fontsize=10)

fig1.tight_layout(pad=1.0)

buf1 = io.BytesIO()
fig1.savefig(buf1, format="png")  # no bbox_inches="tight" to avoid whitespace explosions
plt.close(fig1)
buf1.seek(0)
st.image(buf1.getvalue(), use_container_width=True)

st.markdown("<hr style='margin-top:10px; margin-bottom:12px;'>", unsafe_allow_html=True)

# ── Chart 2: What drives the ratio (home inflation vs rent inflation) ─────
# This is the cleanest decomposition of why the multiple expands or compresses.
common_years = sorted(set(home_yoy_y.dropna().index).intersection(rent_yoy_y.dropna().index))
home_y = home_yoy_y.reindex(common_years)
rent_y = rent_yoy_y.reindex(common_years)
spr_y = spread_y.reindex(common_years)

fig2, axb = plt.subplots(figsize=(13.5, 5.8), dpi=110)

if show_recessions and rec_spans:
    for start, end in rec_spans:
        x0 = start.year + (start.month - 1) / 12.0
        x1 = end.year + (end.month - 1) / 12.0
        axb.axvspan(x0, x1, alpha=0.10, zorder=0)

axb.plot(home_y.index.to_numpy(dtype=float), home_y.values.astype(float), lw=2.2, label="Home price YoY (%)", zorder=2)
axb.plot(rent_y.index.to_numpy(dtype=float), rent_y.values.astype(float), lw=2.2, label="Rent inflation YoY (%)", zorder=2)

# Spread as a dotted line
axb.plot(spr_y.index.to_numpy(dtype=float), spr_y.values.astype(float), ls=":", lw=2.2, label="Spread: Home YoY minus Rent YoY", zorder=2)

axb.axhline(0.0, lw=1.0, alpha=0.5)

axb.set_title("Home vs Rent Inflation (YoY) and the Spread", fontsize=16, weight="bold")
axb.set_ylabel("YoY %", fontsize=12)
axb.set_xlabel("")
axb.grid(axis="y", ls=":", lw=0.7, alpha=0.55)
axb.spines["top"].set_visible(False)
axb.spines["right"].set_visible(False)

axb.legend(loc="best", frameon=False, fontsize=10)

fig2.tight_layout(pad=1.0)

buf2 = io.BytesIO()
fig2.savefig(buf2, format="png")
plt.close(fig2)
buf2.seek(0)
st.image(buf2.getvalue(), use_container_width=True)

st.caption("© 2026 AD Fund Management LP")
