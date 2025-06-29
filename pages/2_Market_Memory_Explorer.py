"""
Market Memory Explorer — AD Fund Management LP
----------------------------------------------
Always show analogs; filtering is *optional* and user-controlled.
"""

import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.ticker import FuncFormatter, MultipleLocator

plt.style.use("default")

START_YEAR = 1980
TRADING_DAYS_FULL_YEAR = 253  # US average, NYSE/NASDAQ; 2025 = 250 days

st.set_page_config(page_title="Market Memory Explorer", layout="wide")

LOGO_PATH = Path("/mnt/data/0ea02e99-f067-4315-accc-0d2bbd3ee87d.png")
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=70)

st.title("Market Memory Explorer")
st.subheader("Compare the current year's return path with history")

# ---- Sidebar ----
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        f"""
        Compare **this year’s YTD performance** for any ticker (index, ETF, or stock) to prior years with the most similar path.

        - **Black = this year**
        - **Dashed = top-correlated analog years**
        - **Legend shows correlation coefficients (ρ)**
        """
    )
    st.markdown("---")
    st.subheader("Analog Outlier Filters (optional)")
    filter_outliers = st.checkbox("Exclude analogs with extreme returns", value=False)
    filter_jumps = st.checkbox("Exclude analogs with large daily jumps", value=False)
    if filter_outliers:
        min_return_sidebar, max_return_sidebar = st.slider(
            "Total Return (%) Range",
            min_value=-100, max_value=1000, value=(-95, 300), step=1,
            help="Exclude analogs with YTD returns outside this range."
        )
    if filter_jumps:
        max_jump_sidebar = st.slider(
            "Allowed Max Daily Jump (%)",
            min_value=5, max_value=100, value=25, step=1,
            help="Exclude analogs where any single day move exceeds this value."
        )
    st.markdown("---")

input_col1, input_col2, input_col3 = st.columns([2, 1, 1])
with input_col1:
    ticker = st.text_input("Ticker", value="^GSPC", help="Index, ETF, or equity.").upper()
with input_col2:
    top_n = st.slider("Top Analogs", 1, 10, 5, help="Number of most correlated years to overlay.")
with input_col3:
    min_corr = st.slider("Min ρ", 0.00, 1.00, 0.00, 0.05, format="%.2f", help="Minimum correlation cutoff.")

st.markdown("<hr style='margin-top: 2px; margin-bottom: 15px;'>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def fetch_price_history(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, start=f"{START_YEAR}-01-01", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, axis=1, level=1, drop_level=True)
    df = df[["Close"]].dropna().copy()
    df["Year"] = df.index.year
    return df

def cumulative_returns(prices: pd.Series) -> pd.Series:
    return prices / prices.iloc[0] - 1

try:
    raw = fetch_price_history(ticker)
except Exception as err:
    st.error(f"Download failed – check ticker. ({err})")
    st.stop()

returns_by_year: dict[int, pd.Series] = {}
for year, grp in raw.groupby("Year"):
    if len(grp) < 30:
        continue
    ytd = cumulative_returns(grp["Close"])
    ytd.index = np.arange(1, len(grp) + 1)  # <<--- FIX: Use trading-day index, NOT dayofyear
    if ytd.isnull().any() or len(ytd) < 30:
        continue
    returns_by_year[year] = ytd

ytd_df = pd.DataFrame(returns_by_year)
current_year = dt.datetime.now().year

if current_year not in ytd_df.columns:
    st.warning(f"No valid YTD data for {current_year}")
    st.stop()

current_ytd = ytd_df[current_year].dropna()
n_days = len(current_ytd)
correlations = {}

for year in ytd_df.columns:
    if year == current_year:
        continue
    past_ytd = ytd_df[year].dropna()
    overlap = min(n_days, len(past_ytd))
    if overlap < 30:
        continue
    rho = np.corrcoef(current_ytd[:overlap], past_ytd[:overlap])[0, 1]
    correlations[year] = rho

filtered_corr = {
    yr: rho for yr, rho in correlations.items()
    if rho >= min_corr
}
top_matches = sorted(filtered_corr.items(), key=lambda kv: kv[1], reverse=True)[: top_n]

if not top_matches:
    st.warning("No historical years meet the correlation cutoff.")
    st.stop()

# --- Filtering and final analog return computation ---
valid_top_matches = []
excluded_analogs = []
analog_ytd_returns = []
outlier_count = 0
jump_count = 0

if filter_outliers or filter_jumps:
    for yr, rho in top_matches:
        analog = ytd_df[yr].dropna()
        if len(analog) >= n_days and not np.isnan(analog.iloc[n_days - 1]):
            valid = True
            ytd_return = analog.iloc[n_days - 1]
            max_jump = analog.pct_change().abs().max()
            if filter_outliers:
                min_return = min_return_sidebar / 100.0
                max_return = max_return_sidebar / 100.0
                if not (min_return < ytd_return < max_return):
                    valid = False
                    outlier_count += 1
            if filter_jumps:
                jump_limit = max_jump_sidebar / 100.0
                if max_jump > jump_limit:
                    valid = False
                    jump_count += 1
            if valid:
                valid_top_matches.append((yr, rho))
                analog_ytd_returns.append(ytd_return)
            else:
                excluded_analogs.append((yr, ytd_return, max_jump))
else:
    for yr, rho in top_matches:
        analog = ytd_df[yr].dropna()
        if len(analog) >= n_days and not np.isnan(analog.iloc[n_days - 1]):
            valid_top_matches.append((yr, rho))
            analog_ytd_returns.append(analog.iloc[n_days - 1])

if excluded_analogs:
    msg = f"{len(excluded_analogs)} analog(s) excluded"
    if outlier_count:
        msg += f" due to extreme returns (outside slider)"
    if jump_count:
        msg += f" due to large daily jump"
    st.info(msg + ".")

current_ytd_return = current_ytd.iloc[-1] if len(current_ytd) > 0 else float('nan')

# --- Compute final (full-year) return for each analog ---
final_analog_returns = []
for yr, rho in valid_top_matches:
    analog = ytd_df[yr].dropna()
    if len(analog) > 0:
        final_analog_returns.append(analog.iloc[-1])

if final_analog_returns:
    median_final = np.median(final_analog_returns)
    std_final = np.std(final_analog_returns)
else:
    median_final = std_final = float("nan")

# --- Metrics display ---
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
with metrics_col1:
    st.metric(
        f"{current_year} YTD Return",
        f"{current_ytd_return:.2%}" if not np.isnan(current_ytd_return) else "N/A",
        help=f"{ticker} YTD cumulative return."
    )
with metrics_col2:
    st.metric(
        "Median Final Analog Return",
        f"{median_final:.2%}" if not np.isnan(median_final) else "N/A",
        help="Median full-year return among top analogs."
    )
with metrics_col3:
    st.metric(
        "Analog Return Dispersion",
        f"{std_final:.2%}" if not np.isnan(std_final) else "N/A",
        help="Standard deviation of full-year returns among top analogs."
    )

st.markdown("<hr style='margin-top: 0; margin-bottom: 6px;'>", unsafe_allow_html=True)

with st.sidebar:
    if valid_top_matches:
        st.markdown(
            f"""
            <div style='background-color:#f8fafb; border-radius:10px; padding:14px 18px 10px 18px; margin-top:4px; margin-bottom:8px; box-shadow:0 1px 3px 0 rgba(0,0,0,0.03);'>
                <span style='font-size:1.1rem; font-weight:600; color:#1761a0;'>
                    Top {len(valid_top_matches)} Analogs to {current_year}
                </span>
                <ul style='margin-top:6px; margin-bottom:0;'>
                    {''.join(f"<li><b style='color:#1f77b4'>{yr}</b>: <span style='color:#555'>ρ = {rho:.4f}</span></li>" for yr, rho in valid_top_matches)}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.download_button(
        label="Download Returns CSV",
        data=ytd_df.to_csv(index=True),
        file_name=f"{ticker}_ytd_paths.csv",
        mime="text/csv"
    )
    st.download_button(
        label="Download Correlation Table",
        data=pd.DataFrame(valid_top_matches, columns=["Year", "Correlation"]).to_csv(index=False),
        file_name=f"{ticker}_top_analogs.csv",
        mime="text/csv"
    )

if len(valid_top_matches) > 0:
    if len(valid_top_matches) <= 10:
        base_cmap = plt.cm.get_cmap("tab10")
    else:
        base_cmap = plt.cm.get_cmap("tab20")
    palette = base_cmap(np.linspace(0, 1, len(valid_top_matches)))

    fig, ax = plt.subplots(figsize=(14, 7))
    for idx, (yr, rho) in enumerate(valid_top_matches):
        analog = ytd_df[yr].dropna()
        ax.plot(
            analog.index,
            analog.values,
            linestyle="--",
            linewidth=2,
            color=palette[idx],
            alpha=0.7,
            label=f"{yr} (ρ={rho:.2f})"
        )
    ax.plot(
        current_ytd.index,
        current_ytd.values,
        color="black",
        linewidth=3.2,
        label=f"{current_year} (YTD)"
    )
    ax.axvline(current_ytd.index[-1], color="gray", linestyle=":", linewidth=1.3, alpha=0.7)
    ax.set_title(f"{ticker} YTD {current_year} vs Historical Analogs", fontsize=16, fontweight="bold")
    ax.set_xlabel("Trading Day of Year", fontsize=13)
    ax.set_ylabel("Cumulative Return", fontsize=13)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(1, TRADING_DAYS_FULL_YEAR)
    ax.grid(True, linestyle=":", linewidth=0.7, color="#888")
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    all_y = [current_ytd.values]
    for yr, _ in valid_top_matches:
        analog = ytd_df[yr].dropna()
        all_y.append(analog.values)
    min_y = min(np.nanmin(arr) for arr in all_y)
    max_y = max(np.nanmax(arr) for arr in all_y)
    span = max_y - min_y
    pad = max(0.02, span * 0.08)
    ax.set_ylim(min_y - pad, max_y + pad)
    ax.legend(loc="upper left", fontsize=11, frameon=False, ncol=2)
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("No valid analog years with complete YTD data and passing all filters for this ticker and selection.")

st.caption("© 2025 AD Fund Management LP")
