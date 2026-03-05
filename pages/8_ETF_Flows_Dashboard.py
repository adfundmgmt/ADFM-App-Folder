import time
from datetime import datetime, date
from typing import Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

st.set_page_config(page_title="ETF Net Flows", layout="wide")

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
    "VTI": ("US Total Market", "Total US equity market"),
    "VUG": ("US Growth", "Large-cap growth"),
    "VTV": ("US Value", "Large-cap value"),
    "MTUM": ("US Momentum", "Momentum factor"),
    "QUAL": ("US Quality", "Quality factor"),
    "USMV": ("US Min Vol", "Minimum volatility"),
    "SCHD": ("US Dividends", "Dividend growth and yield"),
    "XLB": ("US Materials", "S&P 500 Materials"),
    "XLC": ("US Communication Services", "S&P 500 Communication Services"),
    "XLE": ("US Energy", "S&P 500 Energy"),
    "XLF": ("US Financials", "S&P 500 Financials"),
    "XLI": ("US Industrials", "S&P 500 Industrials"),
    "XLK": ("US Technology", "S&P 500 Technology"),
    "XLP": ("US Staples", "S&P 500 Consumer Staples"),
    "XLRE": ("US Real Estate", "S&P 500 Real Estate"),
    "XLU": ("US Utilities", "S&P 500 Utilities"),
    "XLV": ("US Healthcare", "S&P 500 Healthcare"),
    "XLY": ("US Discretionary", "S&P 500 Consumer Discretionary"),
    "SMH": ("Semiconductors", "Global semiconductor equities"),
    "IGV": ("Software", "US application software"),
    "SKYY": ("Cloud", "Cloud infrastructure and services"),
    "VT": ("Global Equity", "Total world equity market"),
    "ACWI": ("Global Equity ex-Frontier", "All-country world equity"),
    "VEA": ("Developed ex-US", "Developed markets ex-US"),
    "VWO": ("Emerging Markets", "Emerging markets equity"),
    "EXUS": ("Non-US Equity", "Global equity ex-US"),
    "EWG": ("Germany", "Germany equities"),
    "EWQ": ("France", "France equities"),
    "EWU": ("United Kingdom", "UK equities"),
    "EWI": ("Italy", "Italy equities"),
    "EWP": ("Spain", "Spain equities"),
    "EWL": ("Switzerland", "Switzerland equities"),
    "EWN": ("Netherlands", "Netherlands equities"),
    "EWD": ("Sweden", "Sweden equities"),
    "EWO": ("Austria", "Austria equities"),
    "EWK": ("Belgium", "Belgium equities"),
    "EWJ": ("Japan", "Japan equities"),
    "EWY": ("South Korea", "Korea equities"),
    "ASHR": ("China A-Shares", "Onshore China equities"),
    "FXI": ("China Large-Cap", "China offshore large caps"),
    "EWT": ("Taiwan", "Taiwan equities"),
    "INDA": ("India", "India equities"),
    "EWS": ("Singapore", "Singapore equities"),
    "EWA": ("Australia", "Australia equities"),
    "EWH": ("Hong Kong", "Hong Kong equities"),
    "EPHE": ("Philippines", "Philippines equities"),
    "EWM": ("Malaysia", "Malaysia equities"),
    "IDX": ("Indonesia", "Indonesia equities"),
    "THD": ("Thailand", "Thailand equities"),
    "VNM": ("Vietnam", "Vietnam equities"),
    "EWZ": ("Brazil", "Brazil equities"),
    "EWW": ("Mexico", "Mexico equities"),
    "EWC": ("Canada", "Canada equities"),
    "EPU": ("Peru", "Peru equities"),
    "ECH": ("Chile", "Chile equities"),
    "ARGT": ("Argentina", "Argentina equities"),
    "GXG": ("Colombia", "Colombia equities"),
    "SGOV": ("UST Bills", "0-3 month Treasuries"),
    "SHY": ("UST 1-3y", "Short-term Treasuries"),
    "IEF": ("UST 7-10y", "Intermediate Treasuries"),
    "TLT": ("UST 20y+", "Long-duration Treasuries"),
    "TIP": ("TIPS", "Inflation-linked Treasuries"),
    "LQD": ("IG Credit", "Investment-grade corporates"),
    "VCIT": ("IG Credit Duration", "Intermediate IG corporates"),
    "HYG": ("High Yield", "High-yield credit"),
    "BKLN": ("Floating-Rate Credit", "Senior loans"),
    "EMB": ("EM Debt", "USD EM sovereign debt"),
    "BND": ("US Aggregate", "Total US bond market"),
    "GLD": ("Gold", "Gold bullion"),
    "SLV": ("Silver", "Silver bullion"),
    "CPER": ("Copper", "Industrial copper"),
    "USO": ("Crude Oil", "WTI crude oil"),
    "DBC": ("Broad Commodities", "Commodity basket"),
    "PDBC": ("Broad Commodities Alt", "Rules-based commodities"),
    "URA": ("Uranium", "Nuclear fuel cycle"),
    "VXX": ("Equity Volatility", "Front-end VIX futures"),
    "UUP": ("USD", "US Dollar Index"),
    "FXE": ("EURUSD", "Euro vs USD"),
    "FXY": ("USDJPY", "Japanese Yen"),
    "FXF": ("CHFUSD", "Swiss franc"),
    "CEW": ("EM FX", "Emerging market currencies"),
    "IBIT": ("Bitcoin", "Spot Bitcoin ETF"),
    "ETH": ("Ethereum", "Spot Ethereum ETF"),
}

etf_tickers = list(etf_info.keys())

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
    padding = 12
    return (as_of - pd.Timedelta(days=days + padding)).date()


def _last_friday(on_or_before: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(on_or_before).normalize()
    wd = ts.weekday()  # Mon=0 ... Sun=6
    days_back = wd - 4 if wd >= 4 else wd + 3
    return ts - pd.Timedelta(days=int(days_back))


def _week_start_monday(week_ending_friday: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(week_ending_friday).normalize() - pd.tseries.offsets.BDay(4)).normalize()


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


def compute_daily_flows(close: pd.Series, shares: pd.Series, window_index: pd.DatetimeIndex) -> pd.Series:
    if close is None or shares is None or close.empty or shares.empty or window_index is None or len(window_index) == 0:
        return pd.Series(dtype="float64")

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

    idx = pd.DatetimeIndex(window_index)
    sh_daily = sh.sort_index().resample("B").ffill().reindex(idx).ffill()
    delta_shares = sh_daily.diff().fillna(0.0)

    close_aligned = close.reindex(idx).ffill()
    flows = (delta_shares * close_aligned).astype(float)
    flows = flows.replace([np.inf, -np.inf], np.nan).dropna()
    return flows


def compute_cmf_turnover_proxy(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return np.nan
    hl = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl
    mfm = mfm.fillna(0.0)
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return float((mfm * typical * vol).sum())


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
        daily_flows_series = pd.Series(dtype="float64")

        if not px.empty:
            window_index = pd.DatetimeIndex(px.index)
            shares = shares_map.get(tk, pd.Series(dtype="float64"))

            if shares is not None and not shares.empty:
                daily_flows = compute_daily_flows(px["Close"], shares, window_index)
                if not daily_flows.empty and daily_flows.replace(0.0, np.nan).dropna().size > 0:
                    flow_usd_sum = float(daily_flows.sum())
                    method = "flows"
                    daily_flows_series = daily_flows
                else:
                    flow_usd_sum = compute_cmf_turnover_proxy(px)
                    method = "cmf_proxy"
            else:
                flow_usd_sum = compute_cmf_turnover_proxy(px)
                method = "cmf_proxy"

        cat, desc = etf_info.get(tk, ("", ""))
        rows.append({
            "Ticker": tk,
            "Category": cat,
            "Flow ($)": flow_usd_sum,
            "Method": method,
            "Description": desc,
            "_daily_flows": daily_flows_series,
        })

    df = pd.DataFrame(rows)

    def label_for(t: str) -> str:
        cat = etf_info.get(t, ("", ""))[0]
        return f"{cat} ({t})"

    df["Label"] = [label_for(t) for t in df["Ticker"]]
    return df


def compute_week_view_value(daily: pd.Series, view: str, as_of_ts: pd.Timestamp) -> float:
    daily = daily.copy()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    this_week_end = _last_friday(as_of_ts)
    this_week_start = _week_start_monday(this_week_end)

    if view == "Week to date":
        start = this_week_start
        end = pd.Timestamp(as_of_ts).normalize()
        return float(daily.loc[(daily.index >= start) & (daily.index <= end)].sum())

    # Last complete week
    last_week_end = this_week_end - pd.Timedelta(days=7)
    last_week_start = _week_start_monday(last_week_end)
    return float(daily.loc[(daily.index >= last_week_start) & (daily.index <= last_week_end)].sum())


# --------------------------- MAIN ---------------------------
st.title("ETF Net Flows")
st.caption(f"Estimated net creations/redemptions over: {period_label}.")

df = build_table(tuple(etf_tickers), period_days, as_of_date, as_of_dt_naive)

# ================================================================
# SECTION 1: HEADLINE BAR CHART WITH WEEKLY TOGGLE
# ================================================================
st.markdown("### Net Flows Snapshot")

bar_view = st.radio(
    "Bar chart view",
    ["Lookback total", "Last complete week", "Week to date"],
    horizontal=True,
    key="bar_view",
)

as_of_ts = pd.Timestamp(as_of_dt_naive).normalize()

chart_rows = []
for _, row in df.iterrows():
    tk = row["Ticker"]
    label = row["Label"]
    method = row["Method"]
    daily = row.get("_daily_flows", pd.Series(dtype="float64"))

    if bar_view == "Lookback total":
        v = row["Flow ($)"]
        chart_rows.append({"Ticker": tk, "Label": label, "Value": v, "Method": method})
    else:
        # Weekly views require shares-based flow series
        if method != "flows" or not isinstance(daily, pd.Series) or daily.empty:
            continue
        v = compute_week_view_value(daily, bar_view, as_of_ts)
        chart_rows.append({"Ticker": tk, "Label": label, "Value": float(v), "Method": method})

# Force schema so empty results never drop columns (prevents KeyError)
chart_df = pd.DataFrame(chart_rows, columns=["Ticker", "Label", "Value", "Method"])
chart_df["Value"] = pd.to_numeric(chart_df["Value"], errors="coerce")
chart_df = chart_df.dropna(subset=["Value"]).sort_values("Value", ascending=False)

if bar_view != "Lookback total":
    wk_end = _last_friday(as_of_ts)
    wk_start = _week_start_monday(wk_end)
    if bar_view == "Last complete week":
        wk_end = wk_end - pd.Timedelta(days=7)
        wk_start = _week_start_monday(wk_end)
    st.caption(f"{bar_view} window: {wk_start.date()} to {wk_end.date()} (week ends Friday)")

vals = chart_df["Value"].fillna(0.0)

if chart_df.empty:
    st.info("No eligible tickers for this view. Weekly views require shares-based flow series (Method = flows).")
else:
    x_min = float(vals.min())
    x_max = float(vals.max())
    x_min = min(x_min, 0.0)
    x_max = max(x_max, 0.0)
    span = (x_max - x_min) if (x_max - x_min) > 0 else 1.0
    pad = 0.08 * span

    n = len(chart_df)
    fig_h = max(6.0, min(20.0, 0.30 * n + 2.0))
    colors = ["green" if v > 0 else "red" if v < 0 else "gray" for v in vals]

    fig, ax = plt.subplots(figsize=(15, fig_h))
    bars = ax.barh(chart_df["Label"], vals, color=colors, alpha=0.9, height=0.82)

    ax.set_xlabel("Estimated Net Flow ($)")
    ax.set_title(f"ETF Net Flows, {bar_view} ({period_label} context)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(axis_fmt))
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.invert_yaxis()
    ax.set_ylim(n - 0.5, -0.5)
    ax.margins(y=0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(x_min - pad, x_max + pad)

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
        ax.text(x_text, bar.get_y() + bar.get_height() / 2, label,
                va="center", ha=ha, fontsize=10, color="black", clip_on=True)

    fig.tight_layout(pad=0.6)
    st.pyplot(fig)
    plt.close(fig)

# ================================================================
# SECTION 2: TOP INFLOWS / OUTFLOWS (kept from your original)
# ================================================================
st.markdown("---")
st.markdown("#### Top Inflows and Outflows")
valid = df.dropna(subset=["Flow ($)"]).copy()

if valid.empty:
    st.write("No ETFs with computable values in the selected period.")
else:
    valid["Value"] = valid["Flow ($)"].apply(fmt_compact_cur)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Top Inflows**")
        st.table(
            valid[valid["Flow ($)"] > 0]
            .nlargest(5, "Flow ($)")
            .set_index("Label")[["Value"]]
        )
    with col2:
        st.write("**Top Outflows**")
        st.table(
            valid[valid["Flow ($)"] < 0]
            .nsmallest(5, "Flow ($)")
            .set_index("Label")[["Value"]]
        )

st.caption(f"Last refresh: {as_of_dt_naive}  |  © 2026 AD Fund Management LP")
