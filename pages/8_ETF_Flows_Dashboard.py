import time
from datetime import datetime, date
from typing import Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from ui_helpers import render_page_header

st.set_page_config(page_title="ETF Net Flows", layout="wide")
render_page_header(
    "ETF Net Flows",
    subtitle="Follow creations and redemptions with a cleaner layout.",
    description="Standardized headers and compact dividers keep the flow tables and charts aligned across the suite.",
)

# --------------------------- SIDEBAR ---------------------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Estimate primary market money moving in and out of key ETFs over a chosen window.

        Methodology
        • First pass: primary flows from Δ shares outstanding × daily close (close as NAV proxy)
        • Fallback: CMF-style turnover proxy when historical shares data are missing or incomplete
        • Sign convention: positive = net creations (inflows), negative = net redemptions (outflows)

        Notes
        • `get_shares_full` from Yahoo Finance is not available for every ETF or every date
        • Close is used as a proxy for daily NAV where needed
        • All flows are aggregated over the selected lookback window and shown in USD
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.subheader("Lookback")

# Timezone-aware now
TZ = pytz.timezone("US/Eastern")


def now_et() -> datetime:
    return datetime.now(TZ)


def as_naive_ts(dt: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return pd.Timestamp(ts.replace(tzinfo=None))


def ytd_days(as_of: datetime) -> int:
    start_ytd = TZ.localize(datetime(as_of.year, 1, 1))
    return (as_of - start_ytd).days


as_of_dt = now_et()
as_of_date = as_of_dt.date()
as_of_dt_naive = as_naive_ts(as_of_dt)

lookback_dict = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "12 Months": 365,
    "YTD": ytd_days(as_of_dt),
}
period_label = st.sidebar.radio("Select Lookback Period", list(lookback_dict.keys()), index=0)
period_days = int(lookback_dict[period_label])

# --------------------------- ETF COVERAGE ---------------------------
etf_info = {
    # Sectors (SPDR + comms)
    "XLB": ("US Materials", "S&P 500 materials sector"),
    "XLC": ("US Communication Services", "S&P 500 communication services sector"),
    "XLE": ("US Energy", "S&P 500 energy sector"),
    "XLF": ("US Financials", "S&P 500 financial sector"),
    "XLI": ("US Industrials", "S&P 500 industrial sector"),
    "XLK": ("US Technology", "S&P 500 technology sector"),
    "XLP": ("US Staples", "S&P 500 consumer staples sector"),
    "XLRE": ("US Real Estate", "S&P 500 real estate sector"),
    "XLU": ("US Utilities", "S&P 500 utilities sector"),
    "XLV": ("US Healthcare", "S&P 500 healthcare sector"),
    "XLY": ("US Discretionary", "S&P 500 consumer discretionary sector"),

    # Semis / software / AI plumbing
    "SMH": ("Semiconductors", "VanEck Semiconductor ETF"),
    "SOXX": ("Semiconductors", "iShares Semiconductor ETF"),
    "IGV": ("Software", "iShares Expanded Tech-Software Sector ETF"),
    "SKYY": ("Cloud", "First Trust Cloud Computing ETF"),

    # Factors / styles
    "VTV": ("US Value", "Vanguard Value ETF"),
    "VUG": ("US Growth", "Vanguard Growth ETF"),
    "IWD": ("US Value", "iShares Russell 1000 Value ETF"),
    "IWF": ("US Growth", "iShares Russell 1000 Growth ETF"),
    "MTUM": ("US Momentum", "Momentum factor US equities"),
    "USMV": ("US Min Volatility", "Low volatility US equities"),
    "QUAL": ("US Quality", "iShares MSCI USA Quality Factor ETF"),
    "VLUE": ("US Value Factor", "Value factor US equities"),
    "SCHD": ("US Dividends", "Schwab U.S. Dividend Equity ETF"),

    # International equity
    "ACWI": ("Global Equity", "MSCI ACWI"),
    "VEA": ("Developed ex-US", "Vanguard FTSE Developed Markets ETF"),
    "VXUS": ("Total ex-US", "Vanguard Total International Stock ETF"),
    "VWO": ("Emerging Markets", "Vanguard FTSE Emerging Markets ETF"),
    "EFA": ("Developed ex-US", "iShares MSCI EAFE ETF"),
    "EEM": ("Emerging Markets", "iShares MSCI Emerging Markets ETF"),
    "VGK": ("Europe Large-Cap", "Developed Europe equities"),
    "EWJ": ("Japan", "Japanese equities"),
    "FXI": ("China Large-Cap", "China mega-cap equities"),
    "KWEB": ("China Internet", "China internet equities"),
    "INDA": ("India", "Indian equities"),
    "EWZ": ("Brazil", "Brazilian equities"),
    "EWG": ("Germany", "German equities"),
    "EWU": ("UK", "UK equities"),

    # Fixed income (rates)
    "BIL": ("T-Bills", "1-3 month T-Bills"),
    "SGOV": ("T-Bills", "0-3 month T-Bills"),
    "SHY": ("UST 1-3y", "Short-term US Treasuries"),
    "IEI": ("UST 3-7y", "Intermediate US Treasuries"),
    "IEF": ("UST 7-10y", "Intermediate US Treasuries"),
    "TLT": ("UST 20y+", "Long-term US Treasuries"),
    "TIP": ("TIPS", "Inflation-protected Treasuries"),
    "LQD": ("US IG Credit", "Investment-grade corporates"),
    "HYG": ("US High Yield", "High-yield corporate bonds"),
    "JNK": ("US High Yield", "High-yield corporate bonds"),
    "EMB": ("EM Debt", "USD-denominated EM debt"),
    "BND": ("US Agg", "US Total Bond Market"),

    # Commodities
    "GLD": ("Gold", "SPDR Gold Trust"),
    "IAU": ("Gold", "iShares Gold Trust"),
    "SLV": ("Silver", "iShares Silver Trust"),
    "USO": ("Crude Oil", "US Oil Fund"),
    "DBO": ("Crude Oil", "Invesco DB Oil Fund"),
    "DBC": ("Broad Commodities", "Invesco DB Commodity Index ETF"),

    # Volatility / hedging
    "VXX": ("Equity Volatility", "Short-term VIX futures ETN/ETF wrapper"),
    "SVXY": ("Equity Volatility", "Short VIX futures strategy wrapper"),

    # Currency wrappers
    "UUP": ("USD", "US Dollar Index bullish wrapper"),
    "FXE": ("EURUSD", "Euro trust"),
    "FXY": ("USDJPY", "Japanese yen trust"),
    "FXB": ("GBPUSD", "British pound trust"),

    # Crypto ETFs
    "ETH": ("Spot ETH", "Grayscale Ethereum Mini Trust ETF"),
    "IBIT": ("Spot BTC", "BlackRock Spot Bitcoin ETF"),
}
etf_tickers: List[str] = list(etf_info.keys())

# --------------------------- HELPERS ---------------------------
def fmt_compact_cur(x) -> str:
    if x is None or pd.isna(x):
        return ""
    x = float(x)
    ax = abs(x)
    if ax >= 1e12:
        return f"${x/1e12:,.0f}T"
    if ax >= 1e9:
        return f"${x/1e9:,.0f}B"
    if ax >= 1e6:
        return f"${x/1e6:,.0f}M"
    if ax >= 1e3:
        return f"${x/1e3:,.0f}K"
    return f"${x:,.0f}"


def axis_fmt(x, _pos=None) -> str:
    return fmt_compact_cur(x)


def retry(n=2, delay=0.5):
    def deco(fn):
        def wrap(*args, **kwargs):
            last = None
            for i in range(n):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last = e
                    time.sleep(delay * (i + 1))
            if last:
                raise last
        return wrap
    return deco


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    need = ["Open", "High", "Low", "Close", "Volume"]
    for c in need:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    idx = pd.to_datetime(df.index, errors="coerce")
    try:
        idx = idx.tz_localize(None)
    except Exception:
        pass
    df.index = idx
    return df.dropna(subset=["Close"])


def _calc_start_date(days: int, as_of: datetime) -> date:
    padding = 7
    return (as_of - pd.Timedelta(days=days + padding)).date()


def _quality_flag(method: str, shares_obs_window: int, nonzero_delta_days: int) -> str:
    if method == "cmf_proxy":
        return "Proxy"
    if shares_obs_window >= 6 and nonzero_delta_days >= 2:
        return "High"
    if shares_obs_window >= 2:
        return "Medium"
    return "Low"


# --------------------------- DATA ---------------------------
@retry()
@st.cache_data(show_spinner=False, ttl=300)
def fetch_prices(tickers: Tuple[str, ...], start_date: date, as_of_date: date) -> Dict[str, pd.DataFrame]:
    data = yf.download(
        tickers=list(tickers),
        start=start_date,
        end=as_of_date + pd.Timedelta(days=1),
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    out: Dict[str, pd.DataFrame] = {}
    for tk in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data[tk].copy()
            else:
                df = data.copy()
        except Exception:
            df = pd.DataFrame()
        out[tk] = _normalize_ohlcv(df)
    return out


@retry()
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_one_shares_series_cached(ticker: str, start_date: date) -> pd.Series:
    t = yf.Ticker(ticker)
    try:
        s = t.get_shares_full(start=start_date)
    except Exception:
        s = None

    if s is None or (isinstance(s, pd.DataFrame) and s.empty):
        return pd.Series(dtype="float64")

    if isinstance(s, pd.DataFrame):
        if s.shape[1] >= 1:
            s = s.iloc[:, 0]
        else:
            return pd.Series(dtype="float64")

    s = pd.to_numeric(s, errors="coerce")
    idx = pd.to_datetime(s.index, errors="coerce")
    try:
        idx = idx.tz_localize(None)
    except Exception:
        pass
    s.index = idx
    return s.dropna()


@st.cache_data(show_spinner=False, ttl=900)
def fetch_shares_series_parallel(
    tickers: Tuple[str, ...],
    start_date: date,
    timeout_sec: float = 10.0,
) -> Dict[str, pd.Series]:
    results: Dict[str, pd.Series] = {tk: pd.Series(dtype="float64") for tk in tickers}

    max_workers = min(10, max(1, len(tickers)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_one_shares_series_cached, tk, start_date): tk for tk in tickers}

        start_time = time.time()
        for fut in as_completed(futures):
            tk = futures[fut]
            remaining = max(0.01, timeout_sec - (time.time() - start_time))
            try:
                results[tk] = fut.result(timeout=remaining)
            except FuturesTimeoutError:
                results[tk] = pd.Series(dtype="float64")
            except Exception:
                results[tk] = pd.Series(dtype="float64")

            if (time.time() - start_time) >= timeout_sec:
                break

    return results


def compute_daily_flows_with_stats(
    close: pd.Series,
    shares: pd.Series,
    window_index: pd.DatetimeIndex
) -> Tuple[pd.Series, Dict[str, float]]:
    stats = {"shares_obs_window": 0, "nonzero_delta_days": 0}

    if close is None or shares is None or close.empty or shares.empty or window_index is None or len(window_index) == 0:
        return pd.Series(dtype="float64"), stats

    close = pd.to_numeric(close, errors="coerce").dropna()
    close.index = pd.to_datetime(close.index, errors="coerce")
    try:
        close.index = close.index.tz_localize(None)
    except Exception:
        pass

    sh = pd.to_numeric(shares, errors="coerce").dropna()
    sh.index = pd.to_datetime(sh.index, errors="coerce")
    try:
        sh.index = sh.index.tz_localize(None)
    except Exception:
        pass

    sh_in = sh.loc[(sh.index >= window_index.min()) & (sh.index <= window_index.max())]
    stats["shares_obs_window"] = int(sh_in.shape[0])

    idx = pd.DatetimeIndex(window_index)
    sh_daily = sh.sort_index().resample("B").ffill().reindex(idx).ffill()
    delta_shares = sh_daily.diff().fillna(0.0)
    stats["nonzero_delta_days"] = int((delta_shares != 0).sum())

    close_aligned = close.reindex(idx).ffill()
    flows = (delta_shares * close_aligned).astype(float)
    flows = flows.replace([np.inf, -np.inf], np.nan).dropna()
    return flows, stats


def compute_cmf_turnover_proxy(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return np.nan

    hl = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl
    mfm = mfm.fillna(0.0)

    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0).clip(lower=0.0)

    return float((mfm * typical * vol).sum())


# --------------------------- BUILD TABLE ---------------------------
@st.cache_data(show_spinner=True, ttl=300)
def build_table(tickers: Tuple[str, ...], period_days: int, as_of_date: date, as_of_dt_naive: pd.Timestamp) -> pd.DataFrame:
    as_of_dt_local = as_of_dt_naive.to_pydatetime().replace(tzinfo=None)
    as_of_dt_et = TZ.localize(as_of_dt_local)

    start_date = _calc_start_date(period_days, as_of_dt_et)

    price_map = fetch_prices(tuple(tickers), start_date, as_of_date)
    shares_map = fetch_shares_series_parallel(tuple(tickers), start_date, timeout_sec=10.0)

    cutoff = as_of_dt_naive - pd.Timedelta(days=period_days)

    rows = []
    for tk in tickers:
        px = price_map.get(tk, pd.DataFrame())
        if not px.empty:
            px = px.loc[px.index >= cutoff]

        flow_usd_sum = np.nan
        method = "flows"
        quality = ""
        shares_obs_window = 0
        nonzero_delta_days = 0

        if not px.empty:
            window_index = pd.DatetimeIndex(px.index)
            shares = shares_map.get(tk, pd.Series(dtype="float64"))

            if shares is not None and not shares.empty:
                daily_flows, stats = compute_daily_flows_with_stats(px["Close"], shares, window_index)
                shares_obs_window = int(stats.get("shares_obs_window", 0))
                nonzero_delta_days = int(stats.get("nonzero_delta_days", 0))

                if not daily_flows.empty and daily_flows.replace(0.0, np.nan).dropna().size > 0:
                    flow_usd_sum = float(daily_flows.sum())
                    method = "flows"
                else:
                    flow_usd_sum = compute_cmf_turnover_proxy(px)
                    method = "cmf_proxy"
            else:
                flow_usd_sum = compute_cmf_turnover_proxy(px)
                method = "cmf_proxy"

            quality = _quality_flag(method, shares_obs_window, nonzero_delta_days)

        cat, desc = etf_info.get(tk, ("", ""))
        rows.append(
            {
                "Ticker": tk,
                "Category": cat,
                "Flow ($)": flow_usd_sum,
                "Method": method,
                "Quality": quality,
                "Shares Obs (window)": shares_obs_window,
                "Nonzero Δ days": nonzero_delta_days,
                "Description": desc,
            }
        )

    df = pd.DataFrame(rows)

    def label_for(t: str) -> str:
        cat = etf_info.get(t, ("", ""))[0]
        return f"{cat} ({t})"

    df["Label"] = [label_for(t) for t in df["Ticker"]]
    return df


# --------------------------- MAIN ---------------------------
st.title("ETF Net Flows")
st.caption(
    f"Estimated net creations/redemptions over: {period_label}. "
    f"Quality uses shares-observations and nonzero share-change days when shares data exist."
)

df = build_table(tuple(etf_tickers), period_days, as_of_date, as_of_dt_naive)

chart_df = df.dropna(subset=["Flow ($)"]).sort_values("Flow ($)", ascending=False).copy()
vals = pd.to_numeric(chart_df["Flow ($)"], errors="coerce").fillna(0.0)

if vals.empty:
    st.info("No values computed. Try a different period.")
else:
    # --- Dynamic x scaling: use actual min/max with a modest pad (less wasted horizontal space)
    x_min = float(vals.min())
    x_max = float(vals.max())
    x_min = min(x_min, 0.0)
    x_max = max(x_max, 0.0)
    span = (x_max - x_min) if (x_max - x_min) > 0 else 1.0
    pad = 0.08 * span

    # --- Dynamic height: tighter baseline, still scales with rows
    n = len(chart_df)
    fig_h = max(6.0, min(20.0, 0.30 * n + 2.0))

    colors = ["green" if v > 0 else "red" if v < 0 else "gray" for v in vals]

    fig, ax = plt.subplots(figsize=(15, fig_h))
    bars = ax.barh(chart_df["Label"], vals, color=colors, alpha=0.9, height=0.82)

    ax.set_xlabel("Estimated Net Flow ($)")
    ax.set_title(f"ETF Net Flows - {period_label}")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(axis_fmt))

    # --- Remove grid completely (covers rcParams environments too)
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    # --- Tighten vertical spacing: remove categorical margins and any top/bottom gap
    ax.invert_yaxis()
    ax.set_ylim(n - 0.5, -0.5)  # this is the key fix for the big gaps
    ax.margins(y=0.0)

    # --- Cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(x_min - pad, x_max + pad)

    # Text labels with padding based on axis span (more stable than abs_max-based padding)
    text_pad = 0.012 * (span if span > 0 else 1.0)
    for bar, raw in zip(bars, vals):
        label = fmt_compact_cur(raw) if pd.notna(raw) else "$0"
        x = bar.get_width()

        if raw > 0:
            x_text, ha = x + text_pad, "left"
        elif raw < 0:
            x_text, ha = x - text_pad, "right"
        else:
            x_text, ha = 0.0, "center"

        ax.text(
            x_text,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha=ha,
            fontsize=10,
            color="black",
            clip_on=True,
        )

    # Tight layout without adding extra whitespace
    fig.tight_layout(pad=0.6)
    st.pyplot(fig)
    plt.close(fig)

# --------------------------- QUALITY TABLE ---------------------------
st.markdown("#### Coverage and Data Quality")
view = df.copy()
view["Flow"] = pd.to_numeric(view["Flow ($)"], errors="coerce")
view["Flow (fmt)"] = view["Flow"].apply(fmt_compact_cur)
view = view.sort_values(["Quality", "Flow"], ascending=[True, False])

show_cols = [
    "Ticker",
    "Category",
    "Flow (fmt)",
    "Method",
    "Quality",
    "Shares Obs (window)",
    "Nonzero Δ days",
    "Description",
]
st.dataframe(
    view[show_cols],
    use_container_width=True,
    hide_index=True,
)

# --------------------------- TOP LISTS ---------------------------
st.markdown("#### Top Inflows and Outflows")
valid = df.dropna(subset=["Flow ($)"]).copy()

if valid.empty:
    st.write("No ETFs with computable values in the selected period.")
else:
    valid["Value"] = valid["Flow ($)"].apply(fmt_compact_cur)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Top Inflows**")
        st.table(valid[valid["Flow ($)"] > 0].nlargest(5, "Flow ($)").set_index("Label")[["Value", "Quality"]])
    with col2:
        st.write("**Top Outflows**")
        st.table(valid[valid["Flow ($)"] < 0].nsmallest(5, "Flow ($)").set_index("Label")[["Value", "Quality"]])

st.caption(f"Last refresh: {as_of_dt_naive}  |  © 2025 AD Fund Management LP")
