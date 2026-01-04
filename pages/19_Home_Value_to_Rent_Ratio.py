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

# ── PIL safety (we are generating trusted images) ─────────────────────────
Image.MAX_IMAGE_PIXELS = None  # disable decompression-bomb check for generated figures

plt.style.use("default")
plt.rcParams["figure.dpi"] = 90
plt.rcParams["savefig.dpi"] = 90

CACHE_TTL_SECONDS = 3600
MAX_RETRIES = 4

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Home Value to Rent Ratio", layout="wide")

LOGO_PATH = Path("/mnt/data/0ea02e99-f067-4315-accc-0d2bbd3ee87d.png")
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=70)

st.title("Home Value to Rent Ratio")
st.subheader("FRED-only recreation with recession shading, mortgage overlay, and z-score regimes")

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        This panel recreates the long-run **Home Value to Rent Ratio** using only public, stable sources.

        Why it matters:

        • Treat it like a valuation multiple for housing  
        • Bubbles show up when prices run far ahead of rent fundamentals  
        • Mean reversion tends to occur through price weakness, rent catch-up, or both  

        Data sources (FRED):

        • Home prices: S&P/Case-Shiller U.S. National Home Price Index (CSUSHPINSA)  
        • Rents: CPI Rent of Primary Residence (CUSR0000SEHA)  
        • Recessions: NBER recession indicator (USREC)  
        • Rates: 30-Year Fixed Mortgage Rate (MORTGAGE30US)

        This is a regime lens. Use it to frame asymmetry and timing windows, not to call tops and bottoms by itself.
        """
    )
    st.markdown("---")
    show_recessions = st.checkbox("Shade recessions (NBER)", value=True)
    show_mortgage = st.checkbox("Overlay 30Y mortgage rate", value=True)
    show_zbands = st.checkbox("Show z-score regime bands", value=True)

    st.markdown("---")
    downturn_floor = st.number_input("Downturn reference (x)", value=12.8, step=0.1)
    show_median = st.checkbox("Show long-run median", value=True)

    st.markdown("---")
    z_window_years = st.slider("Z-score window (years)", 5, 25, 15, 1)
    band_sigma = st.slider("Band width (σ)", 1.0, 3.0, 2.0, 0.5)

st.markdown("<hr style='margin-top:2px; margin-bottom:15px;'>", unsafe_allow_html=True)

# ── Data helpers ─────────────────────────────────────────────────────────
def fetch_bytes(url: str, timeout: int = 30) -> bytes:
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    r.raise_for_status()
    return r.content

def fred_series(series_id: str) -> pd.Series:
    """
    Pull a FRED series via fredgraph CSV. No API key needed.
    Returns a float Series indexed by datetime.
    """
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

def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window=window, min_periods=max(5, window // 3)).mean()
    sig = s.rolling(window=window, min_periods=max(5, window // 3)).std()
    return (s - mu) / sig

# ── Build series ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def build_all(z_window_years_local: int):
    cs = to_monthly(fred_series("CSUSHPINSA"), how="mean")
    rent = to_monthly(fred_series("CUSR0000SEHA"), how="mean")

    df = pd.concat([cs, rent], axis=1, join="inner").dropna()
    df.columns = ["home_index", "rent_index"]
    df["ratio"] = df["home_index"] / df["rent_index"]

    ratio_y = df["ratio"].resample("Y").mean()
    ratio_y.index = ratio_y.index.year

    usrec_m = to_monthly(fred_series("USREC"), how="mean")
    usrec_m = usrec_m.reindex(df.index).ffill().dropna()
    spans = recession_spans(usrec_m)

    mort = fred_series("MORTGAGE30US")
    mort_m = to_monthly(mort, how="mean")
    mort_y = mort_m.resample("Y").mean()
    mort_y.index = mort_y.index.year

    z = rolling_zscore(ratio_y, window=int(z_window_years_local))
    return df, ratio_y, usrec_m, spans, mort_y, z

monthly_df, ratio_y, usrec_m, rec_spans, mort_y, zscore_y = build_all(z_window_years)

# ── Metrics ──────────────────────────────────────────────────────────────
median_val = float(ratio_y.median())
latest_year = int(ratio_y.index.max())
latest_ratio = float(ratio_y.loc[latest_year])

latest_z = float(zscore_y.loc[latest_year]) if latest_year in zscore_y.index and np.isfinite(zscore_y.loc[latest_year]) else np.nan
latest_mort = float(mort_y.loc[latest_year]) if latest_year in mort_y.index else np.nan
in_recession = bool(usrec_m.iloc[-1] > 0.5) if not usrec_m.empty else False

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"{latest_year} Ratio", f"{latest_ratio:.2f}x")
m2.metric(f"{latest_year} Z-score", "N/A" if np.isnan(latest_z) else f"{latest_z:.2f}")
m3.metric(f"{latest_year} 30Y Mortgage", "N/A" if np.isnan(latest_mort) else f"{latest_mort:.2f}%")
m4.metric("Recession (USREC)", "Yes" if in_recession else "No")

st.markdown("<hr style='margin-top:0; margin-bottom:6px;'>", unsafe_allow_html=True)

# ── Plot ─────────────────────────────────────────────────────────────────
# Controlled rendering to avoid Streamlit/PIL huge intermediate images
fig, ax = plt.subplots(figsize=(11.5, 5.8))  # keep dimensions modest

ax.plot(ratio_y.index, ratio_y.values, lw=3.0, label="Home Value / Rent")

if show_median:
    ax.axhline(median_val, ls="--", lw=1.5, alpha=0.8)
    ax.text(ratio_y.index.min() + 1, median_val + 0.05, f"Median: {median_val:.1f}x", alpha=0.85)

ax.axhline(downturn_floor, ls="--", lw=1.5, alpha=0.9)
ax.text(ratio_y.index.min() + 1, downturn_floor - 0.35, f"Downturn floor: {downturn_floor:.1f}x", alpha=0.9)

ax.scatter([latest_year], [latest_ratio], s=80, zorder=5)
ax.text(latest_year + 0.4, latest_ratio, "Now", weight="bold")

if show_zbands:
    window = int(z_window_years)
    mu = ratio_y.rolling(window=window, min_periods=max(5, window // 3)).mean()
    sig = ratio_y.rolling(window=window, min_periods=max(5, window // 3)).std()
    upper = mu + band_sigma * sig
    lower = mu - band_sigma * sig
    ax.fill_between(
        ratio_y.index,
        lower.values,
        upper.values,
        alpha=0.12,
        label=f"±{band_sigma:.1f}σ band ({window}y)",
    )

if show_recessions and rec_spans:
    for start, end in rec_spans:
        y0 = start.year + (start.month - 1) / 12.0
        y1 = end.year + (end.month - 1) / 12.0
        ax.axvspan(y0, y1, alpha=0.08)

ax2 = None
if show_mortgage:
    ax2 = ax.twinx()
    mort_common = mort_y.reindex(ratio_y.index).dropna()
    if not mort_common.empty:
        ax2.plot(mort_common.index, mort_common.values, lw=2.0, ls=":", label="30Y Mortgage (rhs)")
        ax2.set_ylabel("30Y Mortgage Rate (%)", fontsize=12)
        ax2.yaxis.set_major_locator(MultipleLocator(1.0))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax2.spines["top"].set_visible(False)

ax.set_title("Home Value to Rent Ratio (FRED)", fontsize=16, weight="bold")
ax.set_ylabel("Home Value / Rent (x)", fontsize=13)
ax.set_xlabel("")
ax.grid(axis="y", ls=":", lw=0.7, alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ymin, ymax = float(np.nanmin(ratio_y.values)), float(np.nanmax(ratio_y.values))
pad = 0.06 * (ymax - ymin) if ymax > ymin else 0.5
ax.set_ylim(ymin - pad, ymax + pad)

handles1, labels1 = ax.get_legend_handles_labels()
handles, labels = handles1, labels1
if ax2 is not None:
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
if handles:
    ax.legend(handles, labels, loc="best", frameon=False, fontsize=10)

plt.tight_layout()

# Render ourselves to PNG at controlled DPI, then show via st.image
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
plt.close(fig)
buf.seek(0)
st.image(buf.getvalue(), use_container_width=True)

st.caption("© 2026 AD Fund Management LP")

with st.expander("Diagnostics (data tail)"):
    st.write("Annual ratio (tail):")
    st.dataframe(ratio_y.tail(12))
    st.write("Annual mortgage rate (tail):")
    st.dataframe(mort_y.tail(12))
    st.write("Annual z-score (tail):")
    st.dataframe(zscore_y.tail(12))
