import datetime as dt
import time
from pathlib import Path
import colorsys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.ticker import FuncFormatter, MultipleLocator

plt.style.use("default")

MIN_DAYS_REQUIRED = 30               # historical years
MIN_DAYS_CURRENT_YEAR = 5            # allow early-year operation
MIN_DAYS_FOR_CORR = 10               # avoid noisy correlations on tiny samples
CACHE_TTL_SECONDS = 3600

TRAILING_DAYS = 252
ROLLING_MIN_CORR = 0.90

st.set_page_config(page_title="Market Memory Explorer", layout="wide")

LOGO_PATH = Path("/mnt/data/0ea02e99-f067-4315-accc-0d2bbd3ee87d.png")
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=70)

st.title("Market Memory Explorer")
st.subheader("Compare the current year's return path with history")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Explore how this year's cumulative return path compares to history.

        • Pulls adjusted daily closes from Yahoo Finance  
        • Aligns each calendar year by trading day to build YTD paths  
        • Selects historical analogs using correlation to the current year so far  
        • Displays full-year paths for selected analogs to show how similar setups resolved  

        Additional output below:
        • Scans every rolling 252-day historical window  
        • Finds the highest correlating windows versus the current trailing 252 trading days  
        • Keeps years whose best rolling match clears 0.90 correlation  
        • Displays a second overlay chart and a match table  
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.subheader("Filters (optional)")
    f_outliers = st.checkbox("Exclude analogs with extreme YTD returns", value=False)
    f_jumps = st.checkbox("Exclude analogs with large daily jumps", value=False)
    if f_outliers:
        lo, hi = st.slider("Allowed YTD Return Range (%)", -100, 1000, (-95, 300), 1)
    if f_jumps:
        max_jump = st.slider("Max Single-Day Move (%)", 5, 100, 25, 1)
    st.markdown("---")

col1, col2, col3 = st.columns([2, 1, 1])
ticker_in = col1.text_input("Ticker", "^SPX").upper()
top_n = col2.slider("Top Analogs", 1, 10, 5)
min_corr = col3.slider("Min ρ", 0.00, 1.00, 0.00, 0.05, format="%.2f")

# Normalize a couple of common Yahoo quirks
TICKER_ALIASES = {
    "^SPX": "^GSPC",   # S&P 500
}
ticker = TICKER_ALIASES.get(ticker_in, ticker_in)

st.markdown("<hr style='margin-top:2px; margin-bottom:15px;'>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def load_history(symbol: str) -> pd.DataFrame:
    attempts, delay = 0, 1
    df = pd.DataFrame()
    while attempts < 4:
        try:
            df = yf.Ticker(symbol).history(period="max", auto_adjust=True)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            break
        attempts += 1
        time.sleep(delay)
        delay *= 2

    if df.empty or "Close" not in df.columns:
        raise ValueError("Yahoo returned no usable data.")

    df = df.loc[df["Close"].notna(), ["Close"]].copy()
    df["Year"] = df.index.year
    return df

def cumret(series: pd.Series) -> pd.Series:
    return series / series.iloc[0] - 1

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def distinct_palette(n: int):
    if n <= 0:
        return []

    cmap_names = ["tab10", "Dark2", "Set1", "tab20", "tab20b", "tab20c", "Paired", "Accent"]

    candidates = []
    for name in cmap_names:
        cmap = plt.cm.get_cmap(name)
        if hasattr(cmap, "colors"):
            candidates.extend([tuple(c[:3]) for c in cmap.colors])
        else:
            candidates.extend([tuple(cmap(x)[:3]) for x in np.linspace(0, 1, 24)])

    def key_255(rgb):
        return tuple(int(round(x * 255)) for x in rgb)

    uniq = []
    seen = set()
    for c in candidates:
        k = key_255(c)
        if k not in seen:
            seen.add(k)
            uniq.append(c)

    strong = []
    for rgb in uniq:
        h, s, v = colorsys.rgb_to_hsv(*rgb)
        if s >= 0.75 and 0.25 <= v <= 0.85:
            strong.append(rgb)

    pool = strong if len(strong) >= n else uniq

    def dist(a, b):
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        return float(np.sqrt(np.sum((a - b) ** 2)))

    def sat(rgb):
        return colorsys.rgb_to_hsv(*rgb)[1]

    chosen = [max(pool, key=sat)]
    if n == 1:
        return chosen

    remaining = [c for c in pool if c != chosen[0]]
    while len(chosen) < n and remaining:
        best_i, best_min = None, -1.0
        for i, c in enumerate(remaining):
            dmin = min(dist(c, ch) for ch in chosen)
            if dmin > best_min:
                best_min = dmin
                best_i = i
        chosen.append(remaining.pop(best_i))

    if len(chosen) < n:
        extra = list(plt.cm.get_cmap("tab20")(np.linspace(0, 1, n)))
        for c in extra:
            rgb = tuple(c[:3])
            if rgb not in chosen:
                chosen.append(rgb)
            if len(chosen) == n:
                break

    return chosen[:n]

def forward_ret_from_last_day(series: pd.Series, last_day_loc: int, horizon: int) -> float:
    future_loc = last_day_loc + horizon
    if future_loc >= len(series):
        return np.nan
    start_px = float(series.iloc[last_day_loc])
    end_px = float(series.iloc[future_loc])
    if start_px == 0:
        return np.nan
    return end_px / start_px - 1

try:
    raw = load_history(ticker)
except Exception as e:
    st.error(f"Download failed: {e}")
    st.stop()

this_year = dt.datetime.now().year

paths = {}
for yr, grp in raw.groupby("Year"):
    closes = grp["Close"].dropna()
    if yr == this_year:
        if len(closes) < MIN_DAYS_CURRENT_YEAR:
            continue
    else:
        if len(closes) < MIN_DAYS_REQUIRED:
            continue

    ytd = cumret(closes)
    ytd.index = np.arange(1, len(ytd) + 1)
    paths[int(yr)] = ytd

if not paths:
    st.error("No usable yearly data found.")
    st.stop()

ytd_df = pd.DataFrame(paths)

if this_year not in ytd_df.columns:
    st.warning(f"No YTD data for {this_year}. Try lowering MIN_DAYS_CURRENT_YEAR or verify the ticker.")
    st.stop()

current = ytd_df[this_year].dropna()
n_days = len(current)

if n_days < MIN_DAYS_FOR_CORR:
    st.info(
        f"{this_year} has only {n_days} trading days so far. "
        f"Correlations get noisy. Come back after {MIN_DAYS_FOR_CORR} days, or lower MIN_DAYS_FOR_CORR."
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(current.index, current.values, color="black", lw=3.2, label=f"{this_year} (YTD)")
    ax.axvline(n_days, color="gray", ls=":", lw=1.3, alpha=0.8)
    ax.set_title(f"{ticker} - {this_year} YTD", fontsize=16, weight="bold")
    ax.set_xlabel("Trading Day of Year", fontsize=13)
    ax.set_ylabel("Cumulative Return", fontsize=13)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, ls=":", lw=0.7, color="#888")
    ax.legend(loc="best", frameon=False, fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("© 2026 AD Fund Management LP")
    st.stop()

corrs = {}
for yr, ser in ytd_df.items():
    if yr == this_year:
        continue
    ser = ser.dropna()
    if len(ser) < n_days:
        continue
    rho = safe_corr(current.values, ser.iloc[:n_days].values)
    if np.isfinite(rho) and rho >= min_corr:
        corrs[int(yr)] = rho

if not corrs:
    st.warning("No historical years meet the correlation cutoff.")
    st.stop()

def keep_year(yr: int) -> bool:
    ser = ytd_df[yr].dropna()
    if len(ser) < n_days:
        return False

    ret_n = ser.iloc[n_days - 1]
    daily_ret = (1 + ser).div((1 + ser).shift(1)).sub(1)
    max_d = daily_ret.iloc[:n_days].abs().max()

    if f_outliers and not (lo / 100 <= ret_n <= hi / 100):
        return False
    if f_jumps and max_d > max_jump / 100:
        return False
    return True

eligible = {yr: rho for yr, rho in corrs.items() if keep_year(yr)}
if not eligible:
    st.info("All candidates excluded by your filters.")
    st.stop()

top = sorted(eligible.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

current_ret = float(current.iloc[-1])
finals = [float(ytd_df[yr].dropna().iloc[-1]) for yr, _ in top]
median_final = float(np.nanmedian(finals)) if finals else np.nan
sigma_final = float(np.nanstd(finals)) if finals else np.nan

fmt = lambda x: "N/A" if np.isnan(x) else f"{x:.2%}"

m1, m2, m3 = st.columns(3)
m1.metric(f"{this_year} YTD", fmt(current_ret))
m2.metric("Median Full-Year Return (Analogs)", fmt(median_final))
m3.metric("Analog Dispersion (σ)", fmt(sigma_final))

st.markdown("<hr style='margin-top:0; margin-bottom:6px;'>", unsafe_allow_html=True)

palette = distinct_palette(len(top))
fig, ax = plt.subplots(figsize=(14, 7))

for idx, (yr, rho) in enumerate(top):
    ser_full = ytd_df[yr].dropna()
    ax.plot(
        ser_full.index,
        ser_full.values,
        "--",
        lw=2.3,
        alpha=0.95,
        color=palette[idx],
        label=f"{yr} (ρ={rho:.2f})",
    )

ax.plot(current.index, current.values, color="black", lw=3.2, label=f"{this_year} (YTD)")
ax.axvline(n_days, color="gray", ls=":", lw=1.3, alpha=0.8)

ax.set_title(f"{ticker} - {this_year} vs Historical Analogs", fontsize=16, weight="bold")
ax.set_xlabel("Trading Day of Year", fontsize=13)
ax.set_ylabel("Cumulative Return", fontsize=13)
ax.axhline(0, color="gray", ls="--", lw=1)

xmax = max(len(ytd_df[c].dropna()) for c in ytd_df.columns)
ax.set_xlim(1, xmax)

all_y = np.hstack([current.values] + [ytd_df[yr].dropna().values for yr, _ in top])
ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
pad = 0.06 * (ymax - ymin) if ymax > ymin else 0.02
ax.set_ylim(ymin - pad, ymax + pad)

span = ax.get_ylim()[1] - ax.get_ylim()[0]
raw_step = max(span / 12, 0.0025)
candidates = np.array([0.0025, 0.005, 0.01, 0.02, 0.025, 0.05, 0.10, 0.20, 0.25, 0.50, 1.00])
step = float(candidates[np.argmin(np.abs(candidates - raw_step))])
ax.yaxis.set_major_locator(MultipleLocator(step))
ax.yaxis.set_minor_locator(MultipleLocator(step / 2))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))

ax.grid(True, ls=":", lw=0.7, color="#888")
ax.legend(loc="best", frameon=False, ncol=2, fontsize=11)
plt.tight_layout()
st.pyplot(fig)

st.caption("© 2026 AD Fund Management LP")

# =========================
# SECOND OUTPUT
# Trailing 252-day rolling analogs
# =========================

st.markdown("<hr style='margin-top:18px; margin-bottom:12px;'>", unsafe_allow_html=True)
st.subheader(f"Trailing {TRAILING_DAYS}-Day Rolling Analog Overlay")

close_px = raw["Close"].dropna().copy()

if len(close_px) < TRAILING_DAYS * 2:
    st.info(
        f"Need at least {TRAILING_DAYS * 2} trading days of history "
        f"to compare the current trailing {TRAILING_DAYS}-day window against prior rolling windows."
    )
else:
    current_trailing = close_px.iloc[-TRAILING_DAYS:].copy()
    current_trailing_path = cumret(current_trailing)
    current_trailing_path.index = np.arange(1, len(current_trailing_path) + 1)

    current_start_ts = current_trailing.index[0]

    rolling_matches = []

    for start_loc in range(0, len(close_px) - TRAILING_DAYS + 1):
        end_loc = start_loc + TRAILING_DAYS
        hist_window = close_px.iloc[start_loc:end_loc].copy()

        if len(hist_window) != TRAILING_DAYS:
            continue

        if hist_window.index[-1] >= current_start_ts:
            continue

        hist_path = cumret(hist_window)
        hist_path.index = np.arange(1, len(hist_path) + 1)

        rho = safe_corr(current_trailing_path.values, hist_path.values)
        if not np.isfinite(rho) or rho < ROLLING_MIN_CORR:
            continue

        rolling_matches.append(
            {
                "Year": int(hist_window.index[-1].year),
                "Start Date": pd.Timestamp(hist_window.index[0]),
                "End Date": pd.Timestamp(hist_window.index[-1]),
                "Correlation": float(rho),
                "252D Return": float(hist_path.iloc[-1]),
                "Next 21D": forward_ret_from_last_day(close_px, end_loc - 1, 21),
                "Next 63D": forward_ret_from_last_day(close_px, end_loc - 1, 63),
                "Next 126D": forward_ret_from_last_day(close_px, end_loc - 1, 126),
                "_start_loc": int(start_loc),
                "_end_loc": int(end_loc),
            }
        )

    if not rolling_matches:
        st.warning(f"No rolling {TRAILING_DAYS}-day windows found with correlation above {ROLLING_MIN_CORR:.2f}.")
    else:
        rolling_df = pd.DataFrame(rolling_matches)

        best_by_year = (
            rolling_df.sort_values(["Year", "Correlation", "End Date"], ascending=[True, False, True])
            .drop_duplicates(subset=["Year"], keep="first")
            .sort_values("Correlation", ascending=False)
            .reset_index(drop=True)
        )

        overlay_df = best_by_year.head(top_n).copy()

        m4, m5, m6 = st.columns(3)
        m4.metric("Years Above 0.90", f"{len(best_by_year)}")
        m5.metric("Best Rolling ρ", f"{best_by_year['Correlation'].max():.2f}")
        m6.metric(f"Current {TRAILING_DAYS}D Return", f"{current_trailing_path.iloc[-1]:.2%}")

        palette2 = distinct_palette(len(overlay_df))
        fig2, ax2 = plt.subplots(figsize=(14, 7))

        for idx, row in overlay_df.reset_index(drop=True).iterrows():
            hist_window = close_px.iloc[int(row["_start_loc"]):int(row["_end_loc"])].copy()
            hist_path = cumret(hist_window)
            hist_path.index = np.arange(1, len(hist_path) + 1)

            ax2.plot(
                hist_path.index,
                hist_path.values,
                "--",
                lw=2.3,
                alpha=0.95,
                color=palette2[idx],
                label=f"{int(row['Year'])} (ρ={row['Correlation']:.2f})",
            )

        ax2.plot(
            current_trailing_path.index,
            current_trailing_path.values,
            color="black",
            lw=3.2,
            label=f"Current Trailing {TRAILING_DAYS}D",
        )

        ax2.set_title(
            f"{ticker} - Current Trailing {TRAILING_DAYS}D vs Highest Correlating Historical Windows",
            fontsize=16,
            weight="bold",
        )
        ax2.set_xlabel(f"Trading Day in {TRAILING_DAYS}-Day Window", fontsize=13)
        ax2.set_ylabel("Cumulative Return", fontsize=13)
        ax2.axhline(0, color="gray", ls="--", lw=1)
        ax2.set_xlim(1, TRAILING_DAYS)

        all_y2 = [current_trailing_path.values]
        for _, row in overlay_df.iterrows():
            hist_window = close_px.iloc[int(row["_start_loc"]):int(row["_end_loc"])].copy()
            hist_path = cumret(hist_window)
            all_y2.append(hist_path.values)

        all_y2 = np.hstack(all_y2)
        ymin2, ymax2 = float(np.min(all_y2)), float(np.max(all_y2))
        pad2 = 0.06 * (ymax2 - ymin2) if ymax2 > ymin2 else 0.02
        ax2.set_ylim(ymin2 - pad2, ymax2 + pad2)

        span2 = ax2.get_ylim()[1] - ax2.get_ylim()[0]
        raw_step2 = max(span2 / 12, 0.0025)
        candidates2 = np.array([0.0025, 0.005, 0.01, 0.02, 0.025, 0.05, 0.10, 0.20, 0.25, 0.50, 1.00])
        step2 = float(candidates2[np.argmin(np.abs(candidates2 - raw_step2))])

        ax2.yaxis.set_major_locator(MultipleLocator(step2))
        ax2.yaxis.set_minor_locator(MultipleLocator(step2 / 2))
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))

        ax2.grid(True, ls=":", lw=0.7, color="#888")
        ax2.legend(loc="best", frameon=False, ncol=2, fontsize=11)
        plt.tight_layout()
        st.pyplot(fig2)

        table_df = best_by_year.copy()
        table_df["Start Date"] = pd.to_datetime(table_df["Start Date"]).dt.strftime("%Y-%m-%d")
        table_df["End Date"] = pd.to_datetime(table_df["End Date"]).dt.strftime("%Y-%m-%d")
        table_df["Correlation"] = table_df["Correlation"].map(lambda x: f"{x:.3f}")
        table_df["252D Return"] = table_df["252D Return"].map(lambda x: f"{x:.2%}")
        table_df["Next 21D"] = table_df["Next 21D"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2%}")
        table_df["Next 63D"] = table_df["Next 63D"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2%}")
        table_df["Next 126D"] = table_df["Next 126D"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2%}")

        table_df = table_df[
            [
                "Year",
                "Start Date",
                "End Date",
                "Correlation",
                "252D Return",
                "Next 21D",
                "Next 63D",
                "Next 126D",
            ]
        ]

        st.markdown(f"**Best rolling {TRAILING_DAYS}-day match per year with ρ >= {ROLLING_MIN_CORR:.2f}**")
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        st.caption("Second output: trailing 252-day analog scan")
