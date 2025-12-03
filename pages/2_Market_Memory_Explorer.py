# ──────────────────────────────────────────────────────────────────────────
#  Market Memory Explorer  –  AD Fund Management LP
#  ------------------------------------------------
#  v1.9  ·  analog envelope, outcome histogram, stats table
# ──────────────────────────────────────────────────────────────────────────
import datetime as dt
import time
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.ticker import FuncFormatter, MultipleLocator

plt.style.use("default")

TRADING_DAYS_FULL_YEAR  = 253
MIN_DAYS_REQUIRED       = 30
CACHE_TTL_SECONDS       = 3600

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Market Memory Explorer", layout="wide")

LOGO_PATH = Path("/mnt/data/0ea02e99-f067-4315-accc-0d2bbd3ee87d.png")
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=70)

st.title("Market Memory Explorer")
st.subheader("Compare the current year's return path with history")

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Explore how this year's cumulative return path compares to history.

        • Pulls adjusted daily closes from Yahoo Finance  
        • Aligns each calendar year by trading day to build YTD paths  
        • Computes correlations (ρ) between the current year and all past years  
        • Overlays the highest-correlation analogue paths for visual comparison  

        Use the filters below to exclude extreme years or large jump days, and export CSV snapshots for further work.
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.subheader("Filters (optional)")
    f_outliers = st.checkbox("Exclude analogs with extreme YTD returns", value=False)
    f_jumps    = st.checkbox("Exclude analogs with large daily jumps",  value=False)
    if f_outliers:
        lo, hi = st.slider("Allowed YTD Return Range (%)", -100, 1000, (-95, 300), 1)
    if f_jumps:
        max_jump = st.slider("Max Single-Day Move (%)", 5, 100, 25, 1)
    st.markdown("---")

col1, col2, col3 = st.columns([2, 1, 1])
ticker   = col1.text_input("Ticker", "^GSPC").upper()
top_n    = col2.slider("Top Analogs", 1, 10, 5)
min_corr = col3.slider("Min ρ", min_value=0.00, max_value=1.00,
                       value=0.00, step=0.05, format="%.2f")

st.markdown("<hr style='margin-top:2px; margin-bottom:15px;'>", unsafe_allow_html=True)

# ── Data fetch helper ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def load_history(symbol: str, auto_adjust: bool = True) -> pd.DataFrame:
    attempts, delay = 0, 1
    df = pd.DataFrame()
    while attempts < 4:
        try:
            df = yf.Ticker(symbol).history(period="max", auto_adjust=auto_adjust)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            break
        attempts += 1
        time.sleep(delay)
        delay *= 2
    if df.empty:
        raise ValueError("Yahoo returned no data after 4 attempts.")
    if "Close" not in df.columns:
        raise ValueError("'Close' column missing.")
    df = df.loc[df["Close"].notna(), ["Close"]].copy()
    df["Year"] = df.index.year
    return df

def cumret(series: pd.Series) -> pd.Series:
    return series / series.iloc[0] - 1

def max_drawdown(cum_series: pd.Series) -> float:
    """Max drawdown on a cumulative return series (starts at 0)."""
    if cum_series.empty:
        return np.nan
    equity = 1.0 + cum_series
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())

# ── Build YTD paths ──────────────────────────────────────────────────────
try:
    raw = load_history(ticker, auto_adjust=True)
except Exception as e:
    st.error(f"Download failed – {e}")
    st.stop()

paths = {}
for yr, grp in raw.groupby("Year"):
    closes = grp["Close"].dropna()
    if len(closes) < MIN_DAYS_REQUIRED:
        continue
    ytd = cumret(closes)
    ytd.index = np.arange(1, len(closes) + 1)
    paths[yr] = ytd

if not paths:
    st.error("No usable yearly data found.")
    st.stop()

ytd_df    = pd.DataFrame(paths)
this_year = dt.datetime.now().year
if this_year not in ytd_df.columns:
    st.warning(f"No YTD data for {this_year}")
    st.stop()

current = ytd_df[this_year].dropna()
n_days  = len(current)

# ── Correlations ─────────────────────────────────────────────────────────
corrs = {}
for yr, series in ytd_df.items():
    if yr == this_year:
        continue
    clean = series.dropna()
    if len(clean) < n_days:
        continue
    x = current.values
    y = clean.iloc[:n_days].values
    rho = np.corrcoef(x, y)[0, 1]
    if rho >= min_corr:
        corrs[yr] = rho

if not corrs:
    st.warning("No historical years meet the correlation cutoff.")
    st.stop()

# ── Filters then Top-N ───────────────────────────────────────────────────
def keep_year(yr: int) -> bool:
    ser = ytd_df[yr].dropna()
    if len(ser) < n_days:
        return False
    ret_n = ser.iloc[n_days - 1]
    daily_ret = (1.0 + ser).pct_change()
    max_d = daily_ret.abs().max()
    if f_outliers and not (lo/100 <= ret_n <= hi/100):
        return False
    if f_jumps and max_d > max_jump/100:
        return False
    return True

eligible = {yr: rho for yr, rho in corrs.items() if keep_year(yr)}
if not eligible:
    st.info("All candidates excluded by your filters.")
    st.stop()

top = sorted(eligible.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

# ── Metrics ──────────────────────────────────────────────────────────────
current_ret = float(current.iloc[-1])
finals = [float(ytd_df[yr].dropna().iloc[-1]) for yr, _ in top if not ytd_df[yr].dropna().empty]
median_final = float(np.nanmedian(finals)) if finals else np.nan
sigma_final  = float(np.nanstd(finals))    if finals else np.nan
fmt = lambda x: "N/A" if np.isnan(x) else f"{x:.2%}"

m1, m2, m3 = st.columns(3)
m1.metric(f"{this_year} YTD",         fmt(current_ret))
m2.metric("Median Analog Final Return",      fmt(median_final))
m3.metric("Analog Dispersion (σ)",    fmt(sigma_final))

st.markdown("<hr style='margin-top:0; margin-bottom:6px;'>", unsafe_allow_html=True)

# ── Plot: top panel paths + envelope, bottom panel histogram ─────────────
palette = plt.cm.get_cmap("tab10" if len(top) <= 10 else "tab20")(np.linspace(0, 1, len(top)))
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]}
)

# Analog paths on top axis
analog_paths = []
for idx, (yr, rho) in enumerate(top):
    ser = ytd_df[yr].dropna()
    analog_paths.append(ser.iloc[:n_days].values)
    ax_top.plot(
        ser.index,
        ser.values,
        "--",
        lw=1.8,
        alpha=0.55,
        color=palette[idx],
        label=f"{yr} (ρ={rho:.2f})",
    )

# Analog envelope (median ±1σ) on top axis
if analog_paths:
    analog_arr = np.vstack(analog_paths)  # shape: [n_analogs, n_days]
    median_path = np.nanmedian(analog_arr, axis=0)
    std_path    = np.nanstd(analog_arr, axis=0)
    days_idx    = np.arange(1, n_days + 1)

    ax_top.fill_between(
        days_idx,
        median_path - std_path,
        median_path + std_path,
        color="#d0d0d0",
        alpha=0.35,
        label="Analog band (median ±1σ)",
        zorder=1,
    )
    ax_top.plot(
        days_idx,
        median_path,
        color="#555555",
        lw=2.0,
        linestyle="-",
        label="Analog median path",
        zorder=2,
    )

# Current year path on top axis
ax_top.plot(
    current.index,
    current.values,
    color="black",
    lw=3.0,
    label=f"{this_year} (YTD)",
    zorder=3,
)

# Today marker and vertical line
today_day = current.index[-1]
today_val = current.values[-1]
ax_top.axvline(today_day, color="gray", ls=":", lw=1.3, alpha=0.8, zorder=2.5)
ax_top.scatter(today_day, today_val, color="black", s=45, zorder=4)

ax_top.annotate(
    f"Today (Day {today_day})\n{current_ret:.2%}",
    xy=(today_day, today_val),
    xytext=(12, 12),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.8),
)

ax_top.set_title(f"{ticker} - {this_year} vs historical analogs", fontsize=16, weight="bold")
ax_top.set_xlabel("Trading day of year", fontsize=13)
ax_top.set_ylabel("Cumulative return (%)", fontsize=13)
ax_top.axhline(0, color="gray", ls="--", lw=1)

# x-limit from data
xmax = max(len(ytd_df[c].dropna()) for c in ytd_df.columns)
ax_top.set_xlim(1, xmax)

# y-limits from full range of all plotted paths (capture entire move)
all_y = np.hstack([current.values] + [ytd_df[yr].dropna().values for yr, _ in top])
all_y = all_y[np.isfinite(all_y)]
if all_y.size == 0:
    ymin, ymax = -0.02, 0.02
else:
    ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
pad = 0.06 * (ymax - ymin) if ymax > ymin else 0.02
ax_top.set_ylim(ymin - pad, ymax + pad)

# adaptive percent ticks on top axis
span = ax_top.get_ylim()[1] - ax_top.get_ylim()[0]
target_ticks = 12
raw_step = max(span / target_ticks, 0.0025)  # min 0.25%
candidates = np.array([0.0025, 0.005, 0.01, 0.02, 0.025, 0.05, 0.10, 0.20, 0.25, 0.50, 1.00])
step = float(candidates[np.argmin(np.abs(candidates - raw_step))])
ax_top.yaxis.set_major_locator(MultipleLocator(step))
ax_top.yaxis.set_minor_locator(MultipleLocator(step / 2))
ax_top.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))

ax_top.grid(True, ls=":", lw=0.7, color="#888")
ax_top.legend(loc="upper left", frameon=False, ncol=2, fontsize=11)

# Analog summary box on top axis
if finals:
    q25 = float(np.percentile(finals, 25))
    q75 = float(np.percentile(finals, 75))
    prob_up = float(np.mean(np.array(finals) > current_ret))
    box_text = (
        f"Top {len(top)} analogs:\n"
        f"Median final: {median_final:.2%}\n"
        f"25-75% range: {q25:.2%} to {q75:.2%}\n"
        f"P(final > current YTD): {prob_up*100:.0f}%"
    )
    ax_top.text(
        0.02,
        0.96,
        box_text,
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=0.8),
    )

# Bottom panel: histogram of analog full-year outcomes
if finals:
    ax_bottom.hist(
        finals,
        bins=min(15, max(5, len(finals))),
        alpha=0.7,
        edgecolor="black",
    )

    # Vertical lines for current YTD and median analog final
    ax_bottom.axvline(
        current_ret,
        color="black",
        lw=2.0,
        linestyle="-",
        label=f"Current YTD ({this_year})",
    )
    if not np.isnan(median_final):
        ax_bottom.axvline(
            median_final,
            color="gray",
            lw=1.6,
            linestyle="--",
            label="Median analog final",
        )

    ax_bottom.set_xlabel("Analog full-year return (%)", fontsize=12)
    ax_bottom.set_ylabel("Frequency", fontsize=11)
    ax_bottom.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax_bottom.grid(True, ls=":", lw=0.6, color="#aaaaaa")
    ax_bottom.legend(loc="upper right", frameon=False, fontsize=10)
else:
    ax_bottom.text(
        0.5,
        0.5,
        "No analog final-return data available.",
        transform=ax_bottom.transAxes,
        ha="center",
        va="center",
    )
    ax_bottom.axis("off")

plt.tight_layout()
st.pyplot(fig, use_container_width=True)

# ── Analog stats table ───────────────────────────────────────────────────
st.subheader("Analog year statistics")

rows = []
for yr, rho in top:
    ser_full = ytd_df[yr].dropna()
    if len(ser_full) < n_days:
        continue
    ytd_n = float(ser_full.iloc[n_days - 1])
    full_ret = float(ser_full.iloc[-1])
    mdd_full = max_drawdown(ser_full)

    # Max drawdown from "today" to year end
    seg = ser_full.iloc[n_days - 1 :]
    if len(seg) >= 2:
        # rebase at today's level
        equity_seg = 1.0 + seg
        base = equity_seg.iloc[0]
        rel = equity_seg / base
        roll_max_seg = rel.cummax()
        dd_future = (rel / roll_max_seg - 1.0).min()
    else:
        dd_future = np.nan

    rows.append(
        {
            "Year": yr,
            "Corr": rho,
            f"YTD @ day {n_days}": ytd_n,
            "Full-year": full_ret,
            "MaxDD full": mdd_full,
            "MaxDD from today": float(dd_future),
        }
    )

if rows:
    table_df = pd.DataFrame(rows).sort_values("Corr", ascending=False)
    display_df = table_df.copy()
    pct_cols = [c for c in display_df.columns if c not in ["Year", "Corr"]]
    for c in pct_cols:
        display_df[c] = display_df[c] * 100.0
    st.dataframe(
        display_df.style.format(
            {
                "Corr": "{:.2f}",
                **{c: "{:.2f}%" for c in pct_cols},
            }
        ),
        use_container_width=True,
    )
else:
    st.info("No analog stats available for the selected configuration.")

# ── Downloads ────────────────────────────────────────────────────────────
st.subheader("Downloads")
paths_trunc = ytd_df.apply(lambda s: s.dropna().iloc[:n_days])
st.download_button(
    "Download YTD Paths (first n days)",
    data=paths_trunc.to_csv(index_label="TradingDay"),
    file_name=f"{ticker}_ytd_paths_first_{n_days}_days.csv",
    mime="text/csv",
)
corr_df = pd.DataFrame(sorted(corrs.items(), key=lambda kv: kv[1], reverse=True), columns=["Year", "Corr"])
top_df  = pd.DataFrame(top, columns=["Year", "Corr"])
c1, c2 = st.columns(2)
with c1:
    st.download_button("Download Correlations (all eligible)",
                       data=corr_df.to_csv(index=False),
                       file_name=f"{ticker}_correlations_all.csv",
                       mime="text/csv")
with c2:
    st.download_button("Download Correlations (top shown)",
                       data=top_df.to_csv(index=False),
                       file_name=f"{ticker}_correlations_top.csv",
                       mime="text/csv")

st.caption("© 2025 AD Fund Management LP")
