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
        This tool recreates a Reventure-style **Home Value to Rent Ratio** using stable, public FRED series.

        Sources (FRED):
        • Case-Shiller U.S. National HPI (CSUSHPINSA)  
        • CPI Rent of Primary Residence (CUSR0000SEHA)  
        • NBER recession indicator (USREC)  
        • 30Y fixed mortgage rate (MORTGAGE30US)

        Build notes:
        • Ratio is built from indices; it is scaled so the long-run median matches the target median (default 13.9x).  
        • All charts are plotted **per annum** as dots connected by lines.  
        """
    )
    st.markdown("---")
    show_recessions = st.checkbox("Shade recessions (NBER)", value=True)
    show_mortgage = st.checkbox("Overlay 30Y mortgage rate (Chart 1)", value=True)

    st.markdown("---")
    target_median = st.number_input("Target median (x)", value=13.9, step=0.1)
    downturn_floor = st.number_input("Downturn reference (x)", value=12.8, step=0.1)
    show_median = st.checkbox("Show median line", value=True)

    st.markdown("---")
    baseline_year = st.number_input("Affordability baseline year", value=2000, step=1)

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

def mortgage_payment_per_100k(rate_pct: pd.Series, term_months: int = 360) -> pd.Series:
    """
    Standard fixed-rate monthly payment per $100k principal.
    rate_pct is % annual nominal rate.
    """
    r = (rate_pct / 100.0) / 12.0
    denom = 1.0 - (1.0 + r) ** (-term_months)
    factor = np.where(denom == 0, np.nan, r / denom)
    return pd.Series(100_000.0 * factor, index=rate_pct.index)

def add_recession_shading(ax, spans: List[Tuple[pd.Timestamp, pd.Timestamp]]):
    # Neutral light gray so it does not compete with data
    for start, end in spans:
        x0 = start.year + (start.month - 1) / 12.0
        x1 = end.year + (end.month - 1) / 12.0
        ax.axvspan(x0, x1, alpha=0.10, color="#c7c7c7", zorder=0)

def set_year_ticks(ax, years: np.ndarray):
    if years.size == 0:
        return
    xmin = float(np.nanmin(years))
    xmax = float(np.nanmax(years))
    ax.set_xlim(xmin - 0.5, xmax + 0.8)
    start = int(np.floor(xmin / 5.0) * 5)
    end = int(np.ceil(xmax / 5.0) * 5)
    ax.set_xticks(np.arange(start, end + 1, 5))
    ax.tick_params(axis="x", labelsize=10)

def label_last_value(ax, x: float, y: float, text: str):
    # Small end label, avoids "Now" clutter and adds information density
    ax.text(x + 0.35, y, text, fontsize=10, va="center")

# ── Build series ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def build_all(target_median_local: float, baseline_year_local: int):
    cs_m = to_monthly(fred_series("CSUSHPINSA"), how="mean")
    rent_m = to_monthly(fred_series("CUSR0000SEHA"), how="mean")

    mort_w = fred_series("MORTGAGE30US")
    mort_m = to_monthly(mort_w, how="mean")

    df = pd.concat([cs_m, rent_m, mort_m], axis=1, join="inner").dropna()
    df.columns = ["home_index", "rent_index", "mortgage_rate"]

    df["ratio_raw"] = df["home_index"] / df["rent_index"]
    ratio_y_raw = df["ratio_raw"].resample("YE").mean()
    ratio_y_raw.index = ratio_y_raw.index.year

    raw_median = float(ratio_y_raw.median())
    scale = float(target_median_local) / raw_median if np.isfinite(raw_median) and raw_median != 0 else 1.0
    ratio_y = ratio_y_raw * scale

    usrec_m = to_monthly(fred_series("USREC"), how="mean")
    usrec_m = usrec_m.reindex(df.index).ffill().dropna()
    spans = recession_spans(usrec_m)

    mort_y = df["mortgage_rate"].resample("YE").mean()
    mort_y.index = mort_y.index.year

    home_yoy_y = (df["home_index"].pct_change(12) * 100.0).resample("YE").mean()
    home_yoy_y.index = home_yoy_y.index.year

    rent_yoy_y = (df["rent_index"].pct_change(12) * 100.0).resample("YE").mean()
    rent_yoy_y.index = rent_yoy_y.index.year

    spread_y = home_yoy_y - rent_yoy_y

    pay_100k_m = mortgage_payment_per_100k(df["mortgage_rate"])
    pay_100k_y = pay_100k_m.resample("YE").mean()
    pay_100k_y.index = pay_100k_y.index.year

    home_y = df["home_index"].resample("YE").mean()
    home_y.index = home_y.index.year

    burden_raw = pay_100k_y * home_y
    by = int(baseline_year_local)
    if by in burden_raw.index and np.isfinite(burden_raw.loc[by]) and float(burden_raw.loc[by]) != 0:
        burden_idx = (burden_raw / float(burden_raw.loc[by])) * 100.0
    else:
        first = int(burden_raw.dropna().index.min())
        burden_idx = (burden_raw / float(burden_raw.loc[first])) * 100.0

    return ratio_y, spans, mort_y, home_yoy_y, rent_yoy_y, spread_y, burden_idx

ratio_y, rec_spans, mort_y, home_yoy_y, rent_yoy_y, spread_y, burden_idx = build_all(
    float(target_median), int(baseline_year)
)

# ── Chart 1: Ratio ────────────────────────────────────────────────────────
years1 = ratio_y.index.to_numpy(dtype=float)
vals1 = ratio_y.values.astype(float)

latest_year = int(ratio_y.index.max())
latest_ratio = float(ratio_y.loc[latest_year])
median_val = float(ratio_y.median())

fig1, ax = plt.subplots(figsize=(13.5, 6.2), dpi=110)

if show_recessions and rec_spans:
    add_recession_shading(ax, rec_spans)

ax.plot(years1, vals1, lw=2.8, alpha=0.95, color="#1f77b4", label="Home value / rent (x)", zorder=3)
ax.scatter(years1, vals1, s=34, color="#1f77b4", zorder=4)

# Latest point: emphasized, but excluded from legend and no "Now" label
ax.scatter([latest_year], [latest_ratio], s=120, color="#ff7f0e", zorder=5, label="_nolegend_")
label_last_value(ax, latest_year, latest_ratio, f"{latest_ratio:.1f}x")

if show_median:
    ax.axhline(median_val, ls="--", lw=1.6, alpha=0.90, color="#6e6e6e", label=f"Median ({median_val:.1f}x)", zorder=2)

ax.axhline(downturn_floor, ls="--", lw=1.8, alpha=0.95, color="#d62728", label=f"Downturn ref ({downturn_floor:.1f}x)", zorder=2)

ax2 = None
if show_mortgage:
    mort_common = mort_y.reindex(ratio_y.index).dropna()
    if not mort_common.empty:
        ax2 = ax.twinx()
        ax2.plot(
            mort_common.index.to_numpy(dtype=float),
            mort_common.values.astype(float),
            ls=":",
            lw=2.4,
            alpha=0.95,
            color="#2ca02c",
            label="30Y mortgage rate (%)",
            zorder=2,
        )
        ax2.set_ylabel("30Y Mortgage Rate (%)", fontsize=12)
        ax2.yaxis.set_major_locator(MultipleLocator(1.0))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax2.spines["top"].set_visible(False)

ax.set_title("Home Value to Rent Ratio (FRED proxy)", fontsize=16, weight="bold")
ax.set_ylabel("Home value / rent (x)", fontsize=12)
ax.set_xlabel("")
ax.grid(axis="y", ls=":", lw=0.7, alpha=0.45)
ax.grid(axis="x", ls=":", lw=0.5, alpha=0.20)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ymin, ymax = float(np.nanmin(vals1)), float(np.nanmax(vals1))
pad = 0.06 * (ymax - ymin) if ymax > ymin else 0.5
ax.set_ylim(ymin - pad, ymax + pad)
set_year_ticks(ax, years1)

handles1, labels1 = ax.get_legend_handles_labels()
handles, labels = handles1, labels1
if ax2 is not None:
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2

ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=10)

fig1.tight_layout(pad=1.0)
buf1 = io.BytesIO()
fig1.savefig(buf1, format="png", bbox_inches="tight")
plt.close(fig1)
buf1.seek(0)
st.image(buf1.getvalue(), use_container_width=True)

st.markdown("<hr style='margin-top:10px; margin-bottom:12px;'>", unsafe_allow_html=True)

# ── Chart 2: YoY home vs rent inflation and spread ───────────────────────
common_years = sorted(set(home_yoy_y.dropna().index).intersection(rent_yoy_y.dropna().index))
home_y = home_yoy_y.reindex(common_years)
rent_y = rent_yoy_y.reindex(common_years)
spr_y = spread_y.reindex(common_years)
x2 = np.array(common_years, dtype=float)

fig2, axb = plt.subplots(figsize=(13.5, 5.8), dpi=110)

if show_recessions and rec_spans:
    add_recession_shading(axb, rec_spans)

axb.plot(x2, home_y.values.astype(float), lw=2.3, alpha=0.95, color="#1f77b4", label="Home price YoY (%)", zorder=3)
axb.scatter(x2, home_y.values.astype(float), s=30, color="#1f77b4", zorder=4)

axb.plot(x2, rent_y.values.astype(float), lw=2.3, alpha=0.95, color="#ff7f0e", label="Rent inflation YoY (%)", zorder=3)
axb.scatter(x2, rent_y.values.astype(float), s=30, color="#ff7f0e", zorder=4)

axb.plot(x2, spr_y.values.astype(float), ls=":", lw=2.3, alpha=0.95, color="#2ca02c", label="Spread (home minus rent)", zorder=3)
axb.scatter(x2, spr_y.values.astype(float), s=30, color="#2ca02c", zorder=4)

axb.axhline(0.0, lw=1.2, alpha=0.70, color="#6e6e6e")

axb.set_title("Home vs Rent Inflation (YoY) and the Spread", fontsize=16, weight="bold")
axb.set_ylabel("YoY %", fontsize=12)
axb.set_xlabel("")
axb.grid(axis="y", ls=":", lw=0.7, alpha=0.45)
axb.grid(axis="x", ls=":", lw=0.5, alpha=0.20)
axb.spines["top"].set_visible(False)
axb.spines["right"].set_visible(False)

set_year_ticks(axb, x2)
axb.legend(loc="upper left", frameon=False, fontsize=10)

fig2.tight_layout(pad=1.0)
buf2 = io.BytesIO()
fig2.savefig(buf2, format="png", bbox_inches="tight")
plt.close(fig2)
buf2.seek(0)
st.image(buf2.getvalue(), use_container_width=True)

st.markdown("<hr style='margin-top:10px; margin-bottom:12px;'>", unsafe_allow_html=True)

# ── Chart 3: Mortgage payment affordability proxy ─────────────────────────
years3 = burden_idx.dropna().index.to_numpy(dtype=float)
vals3 = burden_idx.dropna().values.astype(float)

latest_aff_year = int(burden_idx.dropna().index.max())
latest_aff = float(burden_idx.dropna().loc[latest_aff_year])

fig3, axc = plt.subplots(figsize=(13.5, 5.8), dpi=110)

if show_recessions and rec_spans:
    add_recession_shading(axc, rec_spans)

axc.plot(
    years3,
    vals3,
    lw=2.6,
    alpha=0.95,
    color="#9467bd",
    label=f"Payment burden index (baseline {int(baseline_year)}=100)",
    zorder=3,
)
axc.scatter(years3, vals3, s=32, color="#9467bd", zorder=4)

# Latest point: emphasized, excluded from legend and no "Now" label
axc.scatter([latest_aff_year], [latest_aff], s=120, color="#d62728", zorder=5, label="_nolegend_")
label_last_value(axc, latest_aff_year, latest_aff, f"{latest_aff:.0f}")

axc.axhline(100.0, lw=1.2, alpha=0.70, color="#6e6e6e", ls="--", label="Baseline = 100")

axc.set_title("Mortgage Payment Affordability Proxy", fontsize=16, weight="bold")
axc.set_ylabel("Index", fontsize=12)
axc.set_xlabel("")
axc.grid(axis="y", ls=":", lw=0.7, alpha=0.45)
axc.grid(axis="x", ls=":", lw=0.5, alpha=0.20)
axc.spines["top"].set_visible(False)
axc.spines["right"].set_visible(False)

ymin3, ymax3 = float(np.nanmin(vals3)), float(np.nanmax(vals3))
pad3 = 0.06 * (ymax3 - ymin3) if ymax3 > ymin3 else 5.0
axc.set_ylim(ymin3 - pad3, ymax3 + pad3)

set_year_ticks(axc, years3)
axc.legend(loc="upper left", frameon=False, fontsize=10)

fig3.tight_layout(pad=1.0)
buf3 = io.BytesIO()
fig3.savefig(buf3, format="png", bbox_inches="tight")
plt.close(fig3)
buf3.seek(0)
st.image(buf3.getvalue(), use_container_width=True)

st.caption(
    "Sources: FRED (CSUSHPINSA, CUSR0000SEHA, USREC, MORTGAGE30US). "
    "Ratio is scaled to the chosen median target for comparability. "
    "Affordability proxy uses the standard fixed-rate payment formula (30Y, payment per $100k) combined with home price level, indexed to the baseline year."
)
st.caption("© 2026 AD Fund Management LP")
