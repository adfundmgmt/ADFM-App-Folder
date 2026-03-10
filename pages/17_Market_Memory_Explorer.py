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

MIN_DAYS_REQUIRED = 30
MIN_DAYS_CURRENT_YEAR = 5
MIN_DAYS_FOR_CORR = 10
CACHE_TTL_SECONDS = 3600

TRAILING_DAYS = 252
ROLLING_MIN_CORR = 0.75
ROLLING_STEP = 5
VOL_RATIO_LOW = 0.5
VOL_RATIO_HIGH = 1.25

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
        Purpose: Historical analog explorer comparing the current market path versus prior periods.

        What the first chart does  
        • Compares the current year-to-date cumulative return path against prior calendar years  
        • Ranks analog years using correlation over the portion of the year completed so far  
        • Helps frame where the current tape sits relative to past calendar-year paths

        What the second chart does  
        • Compares the most recent trailing 252 trading days against all prior rolling 252-day windows  
        • Matches on standardized daily return sequences rather than cumulative paths  
        • Uses the matched historical windows to show what happened over the following 252 trading days  
        • Adds a realized volatility screen so the setup shape and risk profile are closer

        Why this matters  
        • Matching on cumulative paths can overstate similarity because smooth compounding boosts correlation  
        • Matching on normalized daily returns is tougher and usually more honest  
        • The second chart should be treated as scenario framing, not prediction

        Data source  
        • Yahoo Finance adjusted close history
        """,
        unsafe_allow_html=False,
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

TICKER_ALIASES = {
    "^SPX": "^GSPC",
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
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df["Year"] = df.index.year
    return df


def cumret(series: pd.Series) -> pd.Series:
    return series / series.iloc[0] - 1


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) < 2:
        return np.nan
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


def forward_path_from_signal(series: pd.Series, signal_loc: int, horizon: int) -> pd.Series | None:
    end_loc = signal_loc + horizon
    if end_loc >= len(series):
        return None

    window = series.iloc[signal_loc:end_loc + 1].copy()
    if len(window) != horizon + 1 or window.isna().any():
        return None

    out = window / window.iloc[0] - 1
    out.index = np.arange(0, len(out))
    return out


def forward_ret_from_signal(series: pd.Series, signal_loc: int, horizon: int) -> float:
    path = forward_path_from_signal(series, signal_loc, horizon)
    if path is None:
        return np.nan
    return float(path.iloc[-1])


def max_drawdown_from_path(path: pd.Series) -> float:
    wealth = 1 + path
    dd = wealth / wealth.cummax() - 1
    return float(dd.min())


def pct_returns(series: pd.Series) -> pd.Series:
    return series.pct_change().dropna()


def standardized_return_feature(series: pd.Series) -> np.ndarray | None:
    r = pct_returns(series)
    if len(r) != len(series) - 1 or len(r) < 20:
        return None

    mu = float(r.mean())
    sigma = float(r.std(ddof=0))
    if not np.isfinite(sigma) or sigma == 0:
        return None

    z = (r - mu) / sigma
    return z.to_numpy(dtype=float)


def realized_vol(series: pd.Series) -> float:
    r = pct_returns(series)
    if len(r) == 0:
        return np.nan
    return float(r.std(ddof=0))


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
# Rolling analogs used as a signal, then show next 252 trading days
# =========================

st.markdown("<hr style='margin-top:18px; margin-bottom:12px;'>", unsafe_allow_html=True)
st.subheader(f"Forward {TRAILING_DAYS}-Day Signal from Rolling Historical Analogs")

close_px = raw["Close"].dropna().copy()

if len(close_px) < TRAILING_DAYS * 3:
    st.info(
        f"Need at least {TRAILING_DAYS * 3} trading days of history to run a clean "
        f"{TRAILING_DAYS}-day match plus a full {TRAILING_DAYS}-day forward signal window."
    )
else:
    current_start_loc = len(close_px) - TRAILING_DAYS
    current_trailing = close_px.iloc[-TRAILING_DAYS:].copy()

    current_feature = standardized_return_feature(current_trailing)
    current_vol = realized_vol(current_trailing)

    if current_feature is None or not np.isfinite(current_vol):
        st.warning("Current trailing window is not usable for the rolling analog signal.")
    else:
        rolling_matches = []

        max_start = len(close_px) - TRAILING_DAYS + 1
        for start_loc in range(0, max_start, ROLLING_STEP):
            end_loc = start_loc + TRAILING_DAYS
            signal_loc = end_loc - 1

            hist_window = close_px.iloc[start_loc:end_loc].copy()
            if len(hist_window) != TRAILING_DAYS or hist_window.isna().any():
                continue

            # Keep the full forward outcome window outside the current live period
            if end_loc + TRAILING_DAYS > current_start_loc:
                continue

            hist_feature = standardized_return_feature(hist_window)
            hist_vol = realized_vol(hist_window)

            if hist_feature is None or not np.isfinite(hist_vol):
                continue

            if not (VOL_RATIO_LOW * current_vol <= hist_vol <= VOL_RATIO_HIGH * current_vol):
                continue

            rho = safe_corr(current_feature, hist_feature)
            if not np.isfinite(rho) or rho < ROLLING_MIN_CORR:
                continue

            fwd_path = forward_path_from_signal(close_px, signal_loc, TRAILING_DAYS)
            if fwd_path is None:
                continue

            rolling_matches.append(
                {
                    "Year": int(hist_window.index[-1].year),
                    "Match Start": pd.Timestamp(hist_window.index[0]),
                    "Match End": pd.Timestamp(hist_window.index[-1]),
                    "Signal Date": pd.Timestamp(hist_window.index[-1]),
                    "Correlation": float(rho),
                    "Window Vol": float(hist_vol),
                    "Current Vol": float(current_vol),
                    "Vol Ratio": float(hist_vol / current_vol) if current_vol != 0 else np.nan,
                    "Next 21D": float(fwd_path.iloc[min(21, len(fwd_path) - 1)]),
                    "Next 63D": float(fwd_path.iloc[min(63, len(fwd_path) - 1)]),
                    "Next 126D": float(fwd_path.iloc[min(126, len(fwd_path) - 1)]),
                    "Next 252D": float(fwd_path.iloc[-1]),
                    "Max DD Next 252D": max_drawdown_from_path(fwd_path),
                    "_start_loc": int(start_loc),
                    "_end_loc": int(end_loc),
                    "_signal_loc": int(signal_loc),
                }
            )

        if not rolling_matches:
            st.warning(
                f"No rolling {TRAILING_DAYS}-day windows found with correlation above {ROLLING_MIN_CORR:.2f} "
                f"after matching on standardized daily returns and applying the volatility screen."
            )
        else:
            rolling_df = pd.DataFrame(rolling_matches)

            best_by_year = (
                rolling_df.sort_values(
                    ["Year", "Correlation", "Signal Date"],
                    ascending=[True, False, True]
                )
                .drop_duplicates(subset=["Year"], keep="first")
                .sort_values("Correlation", ascending=False)
                .reset_index(drop=True)
            )

            overlay_df = best_by_year.head(top_n).copy()

            forward_paths = []
            for _, row in best_by_year.iterrows():
                fwd_path = forward_path_from_signal(close_px, int(row["_signal_loc"]), TRAILING_DAYS)
                if fwd_path is not None and len(fwd_path) == TRAILING_DAYS + 1:
                    forward_paths.append(fwd_path.values)

            if not forward_paths:
                st.warning("No forward signal paths were available after filtering.")
            else:
                forward_matrix = np.vstack(forward_paths)
                median_path = np.nanmedian(forward_matrix, axis=0)
                p25_path = np.nanpercentile(forward_matrix, 25, axis=0)
                p75_path = np.nanpercentile(forward_matrix, 75, axis=0)
                p10_path = np.nanpercentile(forward_matrix, 10, axis=0)
                p90_path = np.nanpercentile(forward_matrix, 90, axis=0)

                hit_rate_252 = float(np.mean(best_by_year["Next 252D"] > 0)) if len(best_by_year) else np.nan
                median_252 = float(np.nanmedian(best_by_year["Next 252D"])) if len(best_by_year) else np.nan
                median_dd = float(np.nanmedian(best_by_year["Max DD Next 252D"])) if len(best_by_year) else np.nan

                s1, s2, s3 = st.columns(3)
                s1.metric("Qualifying Years", f"{len(best_by_year)}")
                s2.metric("Median Next 252D", fmt(median_252))
                s3.metric("Positive 252D Hit Rate", fmt(hit_rate_252))

                fig2, ax2 = plt.subplots(figsize=(14, 7))
                palette2 = distinct_palette(len(overlay_df))

                x_fwd = np.arange(0, TRAILING_DAYS + 1)

                ax2.fill_between(x_fwd, p10_path, p90_path, color="lightgray", alpha=0.35, label="10th-90th %ile")
                ax2.fill_between(x_fwd, p25_path, p75_path, color="silver", alpha=0.45, label="25th-75th %ile")
                ax2.plot(x_fwd, median_path, color="black", lw=3.0, label="Median Forecast")

                for idx, row in overlay_df.reset_index(drop=True).iterrows():
                    fwd_path = forward_path_from_signal(close_px, int(row["_signal_loc"]), TRAILING_DAYS)
                    if fwd_path is None:
                        continue

                    ax2.plot(
                        fwd_path.index,
                        fwd_path.values,
                        "--",
                        lw=2.2,
                        alpha=0.95,
                        color=palette2[idx],
                        label=f"{int(row['Year'])} (ρ={row['Correlation']:.2f})",
                    )

                ax2.axhline(0, color="gray", ls="--", lw=1)
                ax2.axvline(0, color="gray", ls=":", lw=1.1, alpha=0.9)

                ax2.set_title(
                    f"{ticker} - What Happened Next After Similar {TRAILING_DAYS}-Day Setups",
                    fontsize=16,
                    weight="bold",
                )
                ax2.set_xlabel("Trading Days After Signal", fontsize=13)
                ax2.set_ylabel("Forward Cumulative Return", fontsize=13)
                ax2.set_xlim(0, TRAILING_DAYS)

                all_y2 = [median_path, p10_path, p90_path]
                for _, row in overlay_df.iterrows():
                    fwd_path = forward_path_from_signal(close_px, int(row["_signal_loc"]), TRAILING_DAYS)
                    if fwd_path is not None:
                        all_y2.append(fwd_path.values)

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
                ax2.legend(loc="best", frameon=False, ncol=2, fontsize=10)

                plt.tight_layout()
                st.pyplot(fig2)

                st.caption(
                    f"Signal logic: the current trailing {TRAILING_DAYS}-day window is matched against prior rolling "
                    f"{TRAILING_DAYS}-day windows using standardized daily return sequences, with a realized volatility "
                    f"screen and a {ROLLING_STEP}-day search step. The chart then plots the next {TRAILING_DAYS} trading "
                    f"days after those matched historical windows."
                )

                table_df = best_by_year.copy()
                table_df["Match Start"] = pd.to_datetime(table_df["Match Start"]).dt.strftime("%Y-%m-%d")
                table_df["Match End"] = pd.to_datetime(table_df["Match End"]).dt.strftime("%Y-%m-%d")
                table_df["Signal Date"] = pd.to_datetime(table_df["Signal Date"]).dt.strftime("%Y-%m-%d")
                table_df["Correlation"] = table_df["Correlation"].map(lambda x: f"{x:.3f}")
                table_df["Vol Ratio"] = table_df["Vol Ratio"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}x")
                table_df["Next 21D"] = table_df["Next 21D"].map(lambda x: f"{x:.2%}")
                table_df["Next 63D"] = table_df["Next 63D"].map(lambda x: f"{x:.2%}")
                table_df["Next 126D"] = table_df["Next 126D"].map(lambda x: f"{x:.2%}")
                table_df["Next 252D"] = table_df["Next 252D"].map(lambda x: f"{x:.2%}")
                table_df["Max DD Next 252D"] = table_df["Max DD Next 252D"].map(lambda x: f"{x:.2%}")

                table_df = table_df[
                    [
                        "Year",
                        "Match Start",
                        "Match End",
                        "Signal Date",
                        "Correlation",
                        "Vol Ratio",
                        "Next 21D",
                        "Next 63D",
                        "Next 126D",
                        "Next 252D",
                        "Max DD Next 252D",
                    ]
                ]

                st.markdown(
                    f"**Best forward signal per year with rolling ρ >= {ROLLING_MIN_CORR:.2f} after standardized return matching**"
                )
                st.dataframe(table_df, use_container_width=True, hide_index=True)

                d1, d2, d3 = st.columns(3)
                d1.metric("Median Max Drawdown", fmt(median_dd))
                d2.metric("Median Next 63D", fmt(float(np.nanmedian(best_by_year["Next 63D"]))))
                d3.metric("Median Next 126D", fmt(float(np.nanmedian(best_by_year["Next 126D"]))))

                st.caption("Second output: forward 252-day signal from rolling analogs")
