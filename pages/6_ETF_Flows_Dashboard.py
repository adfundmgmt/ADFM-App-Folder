import time
from datetime import datetime, date
from functools import wraps
from io import BytesIO
from typing import Dict, Tuple, List, Sequence

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="ETF Flow Pressure Proxy", layout="wide")

CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 0.85rem;
        padding-bottom: 1.65rem;
        max-width: 1750px;
    }

    h1, h2, h3 {
        letter-spacing: 0.1px;
        font-weight: 650;
        margin-bottom: 0.35rem;
    }

    div[data-testid="stMetric"] {
        background: #fafafa;
        border: 1px solid #ececec;
        border-radius: 12px;
        padding: 10px 12px;
    }

    .section-card {
        background: #ffffff;
        border: 1px solid #ececec;
        border-radius: 14px;
        padding: 15px 17px;
        margin-top: 12px;
        margin-bottom: 12px;
        color: #242833;
        line-height: 1.55;
    }

    .muted-note {
        color: #6b7280;
        font-size: 0.88rem;
        line-height: 1.4;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 260
plt.rcParams["text.antialiased"] = True
plt.rcParams["font.size"] = 9


# =========================================================
# GLOBALS
# =========================================================
TZ = pytz.timezone("America/New_York")

PASTEL_GREEN = "#b7e4c7"
PASTEL_RED = "#f4a6a6"
PASTEL_GREY = "#d1d5db"
AXIS_GREY = "#6b7280"
TEXT_DARK = "#1f2937"


# =========================================================
# TIME HELPERS
# =========================================================
def now_et() -> datetime:
    return datetime.now(TZ)


def floor_time_to_bucket(dt: datetime, minutes: int = 15) -> datetime:
    minute = (dt.minute // minutes) * minutes
    return dt.replace(minute=minute, second=0, microsecond=0)


def strip_tz_from_index(idx: pd.Index) -> pd.DatetimeIndex:
    out = pd.to_datetime(idx, errors="coerce")
    try:
        if out.tz is not None:
            out = out.tz_convert(None)
    except Exception:
        try:
            out = out.tz_localize(None)
        except Exception:
            pass
    return pd.DatetimeIndex(out)


def ytd_days(as_of: datetime) -> int:
    start_ytd = TZ.localize(datetime(as_of.year, 1, 1))
    return max((as_of - start_ytd).days, 1)


def calc_start_date(days: int, as_of: date) -> date:
    padding = 30
    return (pd.Timestamp(as_of) - pd.Timedelta(days=int(days) + padding)).date()


def last_friday(on_or_before: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(on_or_before).normalize()
    wd = ts.weekday()
    days_back = wd - 4 if wd >= 4 else wd + 3
    return ts - pd.Timedelta(days=int(days_back))


def monday_of_week(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts).normalize()
    return ts - pd.Timedelta(days=int(ts.weekday()))


def week_start_monday(week_ending_friday: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(week_ending_friday).normalize() - pd.tseries.offsets.BDay(4)).normalize()


def business_day_gap(last_data_date: date, as_of: date) -> int:
    if last_data_date is None or pd.isna(last_data_date):
        return 999

    last_ts = pd.Timestamp(last_data_date).normalize()
    as_of_ts = pd.Timestamp(as_of).normalize()

    if last_ts >= as_of_ts:
        return 0

    start = last_ts + pd.tseries.offsets.BDay(1)
    rng = pd.bdate_range(start=start, end=as_of_ts)
    return int(len(rng))


# =========================================================
# RUNTIME
# =========================================================
as_of_dt_et = now_et()
as_of_bucket = floor_time_to_bucket(as_of_dt_et, minutes=15)
as_of_bucket_key = as_of_bucket.strftime("%Y-%m-%d %H:%M %Z")
as_of_date = as_of_dt_et.date()

lookback_dict = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "12 Months": 365,
    "YTD": ytd_days(as_of_dt_et),
}


# =========================================================
# ETF COVERAGE
# =========================================================
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
    "FXY": ("JPYUSD", "Japanese yen vs USD"),
    "FXF": ("CHFUSD", "Swiss franc vs USD"),
    "CEW": ("EM FX", "Emerging market currencies"),
    "IBIT": ("Bitcoin", "Spot Bitcoin ETF"),
    "ETH": ("Ethereum", "Ethereum proxy ticker as originally listed"),
}

etf_tickers = tuple(etf_info.keys())

US_EQUITY_TICKERS = {
    "VTI", "VUG", "VTV", "MTUM", "QUAL", "USMV", "SCHD",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
    "SMH", "IGV", "SKYY",
}

INTERNATIONAL_EQUITY_TICKERS = {
    "VT", "ACWI", "VEA", "VWO", "EXUS",
    "EWG", "EWQ", "EWU", "EWI", "EWP", "EWL", "EWN", "EWD", "EWO", "EWK",
    "EWJ", "EWY", "ASHR", "FXI", "EWT", "INDA", "EWS", "EWA", "EWH", "EPHE",
    "EWM", "IDX", "THD", "VNM", "EWZ", "EWW", "EWC", "EPU", "ECH", "ARGT", "GXG",
}

RATES_CREDIT_TICKERS = {
    "SGOV", "SHY", "IEF", "TLT", "TIP", "LQD", "VCIT", "HYG", "BKLN", "EMB", "BND",
}

COMMODITY_TICKERS = {
    "GLD", "SLV", "CPER", "USO", "DBC", "PDBC", "URA",
}

FX_TICKERS = {
    "UUP", "FXE", "FXY", "FXF", "CEW",
}

CRYPTO_VOL_TICKERS = {
    "IBIT", "ETH", "VXX",
}


def infer_asset_class(ticker: str) -> str:
    if ticker in US_EQUITY_TICKERS:
        return "US Equity"
    if ticker in INTERNATIONAL_EQUITY_TICKERS:
        return "International Equity"
    if ticker in RATES_CREDIT_TICKERS:
        return "Rates and Credit"
    if ticker in COMMODITY_TICKERS:
        return "Commodities"
    if ticker in FX_TICKERS:
        return "FX"
    if ticker in CRYPTO_VOL_TICKERS:
        return "Crypto and Volatility"
    return "Other"


# =========================================================
# FORMATTERS
# =========================================================
def fmt_compact_cur(x) -> str:
    if x is None or pd.isna(x):
        return ""
    x = float(x)
    ax = abs(x)
    sign = "-" if x < 0 else ""

    if ax >= 1e12:
        return f"{sign}${ax / 1e12:,.2f}T"
    if ax >= 1e9:
        return f"{sign}${ax / 1e9:,.2f}B"
    if ax >= 1e6:
        return f"{sign}${ax / 1e6:,.2f}M"
    if ax >= 1e3:
        return f"{sign}${ax / 1e3:,.0f}K"
    return f"{sign}${ax:,.0f}"


def fmt_pct(x) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):,.2f}%"


def fmt_price(x) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"${float(x):,.2f}"


def fmt_score(x) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.4f}"


def fmt_int(x) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{int(x):,}"


def axis_fmt_currency(x, _pos=None) -> str:
    return fmt_compact_cur(x)


# =========================================================
# RETRY AND DOWNLOAD HELPERS
# =========================================================
def retry(n: int = 3, delay: float = 0.8):
    def deco(fn):
        @wraps(fn)
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


def chunked(seq: Sequence[str], size: int) -> List[Tuple[str, ...]]:
    return [tuple(seq[i:i + size]) for i in range(0, len(seq), size)]


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]

    needed = ["Open", "High", "Low", "Close", "Volume"]

    for c in needed:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out.index = strip_tz_from_index(out.index)
    out = out[~out.index.isna()].dropna(subset=["Close"]).sort_index()

    return out[needed]


def extract_ticker_frame(raw: pd.DataFrame, ticker: str, batch_len: int) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        lvl1 = raw.columns.get_level_values(1)

        if ticker in lvl0:
            return raw[ticker].copy()

        if ticker in lvl1:
            return raw.xs(ticker, axis=1, level=1).copy()

        return pd.DataFrame()

    if batch_len == 1:
        return raw.copy()

    return pd.DataFrame()


def safe_yf_download(
    tickers: Tuple[str, ...],
    start_date: date,
    end_date: date,
    threads: bool = True,
    attempts: int = 3,
    delay: float = 0.9,
) -> pd.DataFrame:
    for i in range(attempts):
        try:
            raw = yf.download(
                tickers=list(tickers),
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                threads=threads,
                progress=False,
            )

            if raw is not None and not raw.empty:
                return raw

        except Exception:
            pass

        time.sleep(delay * (i + 1))

    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=900)
def fetch_prices(
    tickers: Tuple[str, ...],
    start_date: date,
    end_date: date,
    cache_key: str,
    batch_size: int = 35,
) -> Dict[str, pd.DataFrame]:
    _ = cache_key

    out: Dict[str, pd.DataFrame] = {}

    for batch in chunked(tickers, batch_size):
        raw = safe_yf_download(batch, start_date, end_date, threads=True)

        for tk in batch:
            df = normalize_ohlcv(extract_ticker_frame(raw, tk, len(batch)))

            if df.empty and len(batch) > 1:
                raw_single = safe_yf_download((tk,), start_date, end_date, threads=False, attempts=2)
                df = normalize_ohlcv(extract_ticker_frame(raw_single, tk, 1))

            out[tk] = df

    for tk in tickers:
        if tk not in out:
            out[tk] = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    return out


# =========================================================
# FLOW PRESSURE ENGINE
# =========================================================
def compute_money_flow_proxy(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0).clip(lower=0.0)

    hl = (high - low).replace(0, np.nan)

    mfm = ((close - low) - (high - close)) / hl
    mfm = mfm.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)

    typical = ((high + low + close) / 3.0).fillna(close)
    money_flow_value = (mfm * vol * typical).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    money_flow_value.index = strip_tz_from_index(money_flow_value.index)

    return money_flow_value.sort_index()


def compute_traded_value(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    close = pd.to_numeric(df["Close"], errors="coerce")
    vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0).clip(lower=0.0)

    tv = (close * vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    tv.index = strip_tz_from_index(tv.index)

    return tv.sort_index()


def compute_pressure_score(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return np.nan

    mfv = compute_money_flow_proxy(df)
    traded_value = compute_traded_value(df)

    if mfv.empty or traded_value.empty:
        return np.nan

    denom = float(traded_value.sum())

    if denom <= 0:
        return np.nan

    return float(mfv.sum() / denom)


def compute_window_sum(daily: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if daily is None or daily.empty:
        return np.nan

    s = daily.copy()
    s.index = strip_tz_from_index(s.index)
    s = s.sort_index()

    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()

    window = s.loc[(s.index >= start) & (s.index <= end)]

    if window.empty:
        return np.nan

    return float(window.sum())


def classify_data_status(
    px: pd.DataFrame,
    lookback_px: pd.DataFrame,
    period_days: int,
    as_of: date,
    adv: float,
) -> Tuple[str, int, float, int, str]:
    if px is None or px.empty:
        return "Missing", 0, np.nan, 999, ""

    last_date = px.index.max().date()
    stale_gap = business_day_gap(last_date, as_of)

    cutoff = pd.Timestamp(as_of).normalize() - pd.Timedelta(days=int(period_days))
    expected_obs = max(len(pd.bdate_range(start=cutoff, end=pd.Timestamp(as_of))), 1)
    obs_count = int(len(lookback_px))
    coverage_ratio = float(obs_count / expected_obs) if expected_obs > 0 else np.nan

    last_date_str = str(last_date)

    if stale_gap >= 4:
        return "Stale", obs_count, coverage_ratio, stale_gap, last_date_str

    if coverage_ratio < 0.60:
        return "Partial History", obs_count, coverage_ratio, stale_gap, last_date_str

    if pd.notna(adv) and adv > 0 and adv < 2_000_000:
        return "Low Volume", obs_count, coverage_ratio, stale_gap, last_date_str

    return "OK", obs_count, coverage_ratio, stale_gap, last_date_str


# =========================================================
# TABLE BUILD
# =========================================================
@st.cache_data(show_spinner=True, ttl=900)
def build_table(
    tickers: Tuple[str, ...],
    period_label: str,
    period_days: int,
    as_of: date,
    cache_key: str,
) -> pd.DataFrame:
    start_date = calc_start_date(period_days, as_of)
    end_date = (pd.Timestamp(as_of) + pd.Timedelta(days=1)).date()

    price_map = fetch_prices(tickers, start_date, end_date, cache_key)

    cutoff = pd.Timestamp(as_of).normalize() - pd.Timedelta(days=int(period_days))

    as_of_ts_local = pd.Timestamp(as_of).normalize()
    latest_complete_week_end = last_friday(as_of_ts_local - pd.Timedelta(days=1))
    latest_complete_week_start = week_start_monday(latest_complete_week_end)

    current_week_start = monday_of_week(as_of_ts_local)
    current_day_end = as_of_ts_local

    rows: List[Dict] = []

    for tk in tickers:
        cat, desc = etf_info.get(tk, ("", ""))
        asset_class = infer_asset_class(tk)
        px = price_map.get(tk, pd.DataFrame()).copy()

        lookback_px = pd.DataFrame()
        if px is not None and not px.empty:
            lookback_px = px.loc[px.index >= cutoff].copy()

        full_daily_proxy = compute_money_flow_proxy(px) if px is not None and not px.empty else pd.Series(dtype="float64")

        lookback_proxy = np.nan
        pressure_score = np.nan
        ret = np.nan
        last_price = np.nan
        adv = np.nan

        if not lookback_px.empty:
            lookback_daily_proxy = compute_money_flow_proxy(lookback_px)

            if not lookback_daily_proxy.empty:
                lookback_proxy = float(lookback_daily_proxy.sum())

            pressure_score = compute_pressure_score(lookback_px)

            close = pd.to_numeric(lookback_px["Close"], errors="coerce").dropna()
            if len(close) >= 2 and close.iloc[0] != 0:
                ret = float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0)

            if len(close) >= 1:
                last_price = float(close.iloc[-1])

            adv_series = compute_traded_value(lookback_px)
            if adv_series.notna().any():
                adv = float(adv_series.mean())

        latest_complete_week_proxy = compute_window_sum(
            full_daily_proxy,
            latest_complete_week_start,
            latest_complete_week_end,
        )

        week_to_date_proxy = compute_window_sum(
            full_daily_proxy,
            current_week_start,
            current_day_end,
        )

        data_status, obs_count, coverage_ratio, stale_gap, last_data_date = classify_data_status(
            px=px,
            lookback_px=lookback_px,
            period_days=period_days,
            as_of=as_of,
            adv=adv,
        )

        rows.append(
            {
                "Ticker": tk,
                "Label": f"{cat} ({tk})",
                "Asset Class": asset_class,
                "Category": cat,
                "Description": desc,
                "Last Price": last_price,
                f"{period_label} Flow Pressure Proxy": lookback_proxy,
                "Latest Complete Week": latest_complete_week_proxy,
                "Week to Date": week_to_date_proxy,
                "Pressure Score": pressure_score,
                f"{period_label} Return %": ret,
                "Avg Daily Dollar Vol": adv,
                "Obs": obs_count,
                "Coverage %": coverage_ratio * 100.0 if pd.notna(coverage_ratio) else np.nan,
                "Business Days Since Last Bar": stale_gap,
                "Last Data Date": last_data_date,
                "Data Status": data_status,
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# DISPLAY HELPERS
# =========================================================
def render_matplotlib_high_res(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=285,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        edgecolor="white",
    )
    buf.seek(0)
    return buf.getvalue()


def color_for_value(v: float, was_missing: bool = False) -> str:
    if was_missing:
        return PASTEL_GREY
    if pd.isna(v) or abs(float(v)) < 1e-12:
        return PASTEL_GREY
    if float(v) > 0:
        return PASTEL_GREEN
    return PASTEL_RED


def metric_is_currency(metric_name: str) -> bool:
    return metric_name in {
        "Latest Complete Week",
        "Week to Date",
    } or "Flow Pressure Proxy" in metric_name


def format_metric_value(metric_name: str, value) -> str:
    if pd.isna(value):
        return ""

    if metric_is_currency(metric_name):
        return fmt_compact_cur(value)

    if metric_name.endswith("Return %") or metric_name == "Coverage %":
        return fmt_pct(value)

    if metric_name == "Pressure Score":
        return fmt_score(value)

    return f"{float(value):,.2f}"


def top_names(df: pd.DataFrame, metric: str, n: int = 3, positive: bool = True) -> str:
    x = df.copy()
    x[metric] = pd.to_numeric(x[metric], errors="coerce")
    x = x.dropna(subset=[metric])

    if x.empty:
        return "None"

    x = x[x[metric] > 0] if positive else x[x[metric] < 0]

    if x.empty:
        return "None"

    x = x.nlargest(n, metric) if positive else x.nsmallest(n, metric)

    parts = [
        f"{row['Ticker']} {format_metric_value(metric, row[metric])}"
        for _, row in x.iterrows()
    ]

    return ", ".join(parts)


def build_tape_read(df: pd.DataFrame, flow_col: str) -> str:
    valid = df[df["Data Status"] != "Missing"].copy()
    valid[flow_col] = pd.to_numeric(valid[flow_col], errors="coerce")

    if valid.empty or valid[flow_col].dropna().empty:
        return "Tape read unavailable because no usable flow-pressure values were returned."

    net = valid[flow_col].sum(min_count=1)

    group = (
        valid.groupby("Asset Class", dropna=False)[flow_col]
        .sum(min_count=1)
        .dropna()
        .sort_values(ascending=False)
    )

    if group.empty:
        group_leader = "None"
        group_lagger = "None"
    else:
        group_leader = f"{group.index[0]} {fmt_compact_cur(group.iloc[0])}"
        group_lagger = f"{group.index[-1]} {fmt_compact_cur(group.iloc[-1])}"

    top_pos = top_names(valid, flow_col, n=3, positive=True)
    top_neg = top_names(valid, flow_col, n=3, positive=False)

    if pd.isna(net):
        net_text = "unavailable"
    elif net > 0:
        net_text = f"positive at {fmt_compact_cur(net)}"
    elif net < 0:
        net_text = f"negative at {fmt_compact_cur(net)}"
    else:
        net_text = "flat"

    return (
        f"<strong>Tape read:</strong> Aggregate {flow_col.lower()} is {net_text}. "
        f"The strongest positive pressure is in {top_pos}. "
        f"The weakest pressure is in {top_neg}. "
        f"By asset class, the leader is {group_leader}, while the weakest bucket is {group_lagger}. "
        f"This is a price-volume pressure proxy from public OHLCV data. Treat it as a directional tape signal, "
        f"separate from official ETF creation and redemption flow data."
    )


def build_chart_dataframe(
    source_df: pd.DataFrame,
    metric: str,
    sort_mode: str,
) -> pd.DataFrame:
    chart_df = source_df[
        ["Ticker", "Label", "Asset Class", metric, "Data Status"]
    ].copy()

    chart_df["Raw Value"] = pd.to_numeric(chart_df[metric], errors="coerce")
    chart_df["Missing Chart Value"] = chart_df["Raw Value"].isna()
    chart_df["Chart Value"] = chart_df["Raw Value"].fillna(0.0)

    def label_with_status(row):
        label = row["Label"]
        status = row["Data Status"]
        if status in {"Missing", "Stale", "Partial History", "Low Volume"}:
            return f"{label} [{status}]"
        return label

    chart_df["Chart Label"] = chart_df.apply(label_with_status, axis=1)

    if sort_mode == "Positive to Negative":
        chart_df = chart_df.sort_values("Chart Value", ascending=False)
    elif sort_mode == "Negative to Positive":
        chart_df = chart_df.sort_values("Chart Value", ascending=True)
    elif sort_mode == "Asset Class":
        chart_df = chart_df.sort_values(["Asset Class", "Chart Value"], ascending=[True, False])
    else:
        chart_df["Original Order"] = range(len(chart_df))
        chart_df = chart_df.sort_values("Original Order")

    return chart_df.reset_index(drop=True)


# =========================================================
# SIDEBAR CONTROLS
# =========================================================
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** ETF flow-pressure monitor using a stable price-volume proxy workflow.

        **What the chart shows**
        - Full ETF coverage by default.
        - Missing chart values are retained and shown as zero.
        - Green means positive pressure.
        - Red means negative pressure.
        - Grey means zero, missing, or unchartable.

        **Data source**
        - Yahoo Finance OHLCV data through yfinance.

        **Important**
        - This is a directional price-volume pressure signal.
        - It is separate from official ETF creation and redemption flow data.
        """
    )

    st.markdown("---")

    period_label = st.radio(
        "Lookback Window",
        list(lookback_dict.keys()),
        index=0,
    )

    chart_metric_choice = st.radio(
        "Ranking Metric",
        options=[
            "Flow Pressure Proxy",
            "Latest Complete Week",
            "Week to Date",
            "Pressure Score",
            "Return %",
        ],
        index=0,
    )

    sort_mode = st.radio(
        "Chart Sort",
        options=[
            "Positive to Negative",
            "Negative to Positive",
            "Asset Class",
            "Original Coverage Order",
        ],
        index=0,
    )

    st.markdown("---")
    st.caption(f"As of: {as_of_dt_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    st.caption(f"Cache bucket: {as_of_bucket_key}")


period_days = int(lookback_dict[period_label])
flow_col = f"{period_label} Flow Pressure Proxy"
return_col = f"{period_label} Return %"

metric_map = {
    "Flow Pressure Proxy": flow_col,
    "Latest Complete Week": "Latest Complete Week",
    "Week to Date": "Week to Date",
    "Pressure Score": "Pressure Score",
    "Return %": return_col,
}

bar_view = metric_map[chart_metric_choice]


# =========================================================
# HEADER
# =========================================================
st.title("ETF Flow Pressure Proxy")
st.caption(
    "Directional ETF pressure monitor using public OHLCV data. "
    "Positive values suggest accumulation pressure. Negative values suggest distribution pressure."
)


# =========================================================
# DATA BUILD
# =========================================================
df = build_table(
    tickers=etf_tickers,
    period_label=period_label,
    period_days=period_days,
    as_of=as_of_date,
    cache_key=as_of_bucket_key,
)


# =========================================================
# RANKING VIEW FIRST
# =========================================================
st.markdown("---")
st.subheader("Ranking View")

chart_df = build_chart_dataframe(
    source_df=df,
    metric=bar_view,
    sort_mode=sort_mode,
)

shown_count = int(len(chart_df))
total_count = int(len(df))

st.caption(
    f"Showing {shown_count} of {total_count} ETFs. "
    f"Rows with missing or unchartable values stay in the ranking as grey zero-value bars."
)

if chart_df.empty:
    st.info("No chartable values available.")
else:
    vals = chart_df["Chart Value"].astype(float)
    raw_vals = chart_df["Raw Value"]

    x_min = float(vals.min())
    x_max = float(vals.max())
    x_min = min(x_min, 0.0)
    x_max = max(x_max, 0.0)

    span = (x_max - x_min) if (x_max - x_min) > 0 else 1.0
    pad = 0.075 * span

    n = len(chart_df)

    fig_h = max(13.5, min(28.0, 0.255 * n + 1.8))
    bar_height = 0.82 if n >= 70 else 0.78
    y_font = 6.8 if n >= 80 else 7.3
    value_font = 6.7 if n >= 80 else 7.2

    colors = [
        color_for_value(v, was_missing=missing)
        for v, missing in zip(chart_df["Chart Value"], chart_df["Missing Chart Value"])
    ]

    fig, ax = plt.subplots(figsize=(16.8, fig_h), dpi=220)

    bars = ax.barh(
        chart_df["Chart Label"],
        vals,
        color=colors,
        alpha=0.98,
        height=bar_height,
        linewidth=0,
    )

    ax.axvline(0, color="#9ca3af", linewidth=0.85, alpha=0.85)

    ax.set_xlabel(
        "Estimated Flow Pressure ($)" if metric_is_currency(bar_view) else bar_view,
        fontsize=9,
        color=TEXT_DARK,
    )
    ax.set_title(f"{bar_view} | Full Coverage", fontsize=10.8, pad=7, color=TEXT_DARK)

    if metric_is_currency(bar_view):
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(axis_fmt_currency))
    elif bar_view.endswith("Return %"):
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{x:,.1f}%"))
    else:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{x:,.3f}"))

    ax.tick_params(axis="y", labelsize=y_font, pad=0.6, length=0, colors=TEXT_DARK)
    ax.tick_params(axis="x", labelsize=8.2, colors=AXIS_GREY)

    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    ax.invert_yaxis()
    ax.set_ylim(n - 0.5, -0.5)
    ax.margins(y=0.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e5e7eb")
    ax.spines["bottom"].set_color("#e5e7eb")

    ax.set_xlim(x_min - pad, x_max + pad)

    text_pad = 0.0075 * span

    for bar, raw, chart_value, was_missing in zip(
        bars,
        raw_vals,
        vals,
        chart_df["Missing Chart Value"],
    ):
        if was_missing:
            label_txt = "NA"
        else:
            label_txt = format_metric_value(bar_view, raw)

        x = chart_value

        if chart_value > 0:
            x_text, ha = x + text_pad, "left"
        elif chart_value < 0:
            x_text, ha = x - text_pad, "right"
        else:
            x_text, ha = 0.0, "center"

        ax.text(
            x_text,
            bar.get_y() + bar.get_height() / 2,
            label_txt,
            va="center",
            ha=ha,
            fontsize=value_font,
            color="#374151",
            clip_on=False,
        )

    fig.subplots_adjust(left=0.285, right=0.975, top=0.955, bottom=0.055)

    chart_png = render_matplotlib_high_res(fig)
    st.image(chart_png, use_container_width=True)
    plt.close(fig)


# =========================================================
# EVERYTHING ELSE BELOW THE RANKING IMAGE
# =========================================================
st.markdown("---")

all_asset_classes = sorted(df["Asset Class"].dropna().unique().tolist())
all_statuses = ["OK", "Low Volume", "Partial History", "Stale", "Missing"]

with st.expander("Filters and Display Controls", expanded=False):
    c1, c2 = st.columns(2)

    with c1:
        selected_assets = st.multiselect(
            "Asset Class Filter",
            options=all_asset_classes,
            default=all_asset_classes,
        )

    with c2:
        selected_statuses = st.multiselect(
            "Data Status Filter",
            options=all_statuses,
            default=all_statuses,
        )

filtered_df = df[
    df["Asset Class"].isin(selected_assets)
    & df["Data Status"].isin(selected_statuses)
].copy()


# =========================================================
# SUMMARY METRICS
# =========================================================
missing_count = int((df["Data Status"] == "Missing").sum())
stale_count = int((df["Data Status"] == "Stale").sum())
partial_count = int((df["Data Status"] == "Partial History").sum())
low_vol_count = int((df["Data Status"] == "Low Volume").sum())
ok_count = int((df["Data Status"] == "OK").sum())

net_lookback = pd.to_numeric(df[flow_col], errors="coerce").sum(min_count=1)
net_lcw = pd.to_numeric(df["Latest Complete Week"], errors="coerce").sum(min_count=1)
net_wtd = pd.to_numeric(df["Week to Date"], errors="coerce").sum(min_count=1)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Tickers Listed", f"{len(df)}")
c2.metric("Clean Data", f"{ok_count}")
c3.metric("Stale or Partial", f"{stale_count + partial_count}")
c4.metric("Missing", f"{missing_count}")
c5.metric(f"{period_label} Aggregate", fmt_compact_cur(net_lookback) if pd.notna(net_lookback) else "")
c6.metric("WTD Aggregate", fmt_compact_cur(net_wtd) if pd.notna(net_wtd) else "")

st.caption(
    f"Diagnostics: {low_vol_count} low-volume, {partial_count} partial-history, "
    f"{stale_count} stale, {missing_count} missing."
)

st.markdown(
    f"""
<div class="section-card">
{build_tape_read(df, flow_col)}
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# ASSET CLASS ROLLUP
# =========================================================
with st.expander("Asset-Class Rollup", expanded=False):
    group_df = (
        df.groupby("Asset Class", dropna=False)
        .agg(
            Tickers=("Ticker", "count"),
            Clean_Data=("Data Status", lambda x: int((x == "OK").sum())),
            Lookback_Flow=(flow_col, lambda x: pd.to_numeric(x, errors="coerce").sum(min_count=1)),
            Latest_Complete_Week=("Latest Complete Week", lambda x: pd.to_numeric(x, errors="coerce").sum(min_count=1)),
            Week_to_Date=("Week to Date", lambda x: pd.to_numeric(x, errors="coerce").sum(min_count=1)),
            Avg_Pressure_Score=("Pressure Score", lambda x: pd.to_numeric(x, errors="coerce").mean()),
            Avg_Return=(return_col, lambda x: pd.to_numeric(x, errors="coerce").mean()),
        )
        .reset_index()
        .sort_values("Lookback_Flow", ascending=False)
    )

    group_display = group_df.copy()
    group_display["Lookback_Flow"] = group_display["Lookback_Flow"].apply(fmt_compact_cur)
    group_display["Latest_Complete_Week"] = group_display["Latest_Complete_Week"].apply(fmt_compact_cur)
    group_display["Week_to_Date"] = group_display["Week_to_Date"].apply(fmt_compact_cur)
    group_display["Avg_Pressure_Score"] = group_display["Avg_Pressure_Score"].apply(fmt_score)
    group_display["Avg_Return"] = group_display["Avg_Return"].apply(fmt_pct)

    group_display = group_display.rename(
        columns={
            "Clean_Data": "Clean Data",
            "Lookback_Flow": f"{period_label} Flow Pressure Proxy",
            "Latest_Complete_Week": "Latest Complete Week",
            "Week_to_Date": "Week to Date",
            "Avg_Pressure_Score": "Avg Pressure Score",
            "Avg_Return": f"Avg {period_label} Return",
        }
    )

    st.dataframe(
        group_display,
        use_container_width=True,
        hide_index=True,
        height=260,
    )


# =========================================================
# TABLE PREP
# =========================================================
display_cols = [
    "Ticker",
    "Asset Class",
    "Category",
    "Description",
    "Last Price",
    flow_col,
    "Latest Complete Week",
    "Week to Date",
    "Pressure Score",
    return_col,
    "Avg Daily Dollar Vol",
    "Obs",
    "Coverage %",
    "Business Days Since Last Bar",
    "Last Data Date",
    "Data Status",
]

display_df = filtered_df[display_cols].copy()

display_df["Last Price"] = display_df["Last Price"].apply(fmt_price)
display_df[flow_col] = display_df[flow_col].apply(fmt_compact_cur)
display_df["Latest Complete Week"] = display_df["Latest Complete Week"].apply(fmt_compact_cur)
display_df["Week to Date"] = display_df["Week to Date"].apply(fmt_compact_cur)
display_df["Pressure Score"] = display_df["Pressure Score"].apply(fmt_score)
display_df[return_col] = display_df[return_col].apply(fmt_pct)
display_df["Avg Daily Dollar Vol"] = display_df["Avg Daily Dollar Vol"].apply(fmt_compact_cur)
display_df["Obs"] = display_df["Obs"].apply(fmt_int)
display_df["Coverage %"] = display_df["Coverage %"].apply(fmt_pct)
display_df["Business Days Since Last Bar"] = display_df["Business Days Since Last Bar"].apply(fmt_int)


# =========================================================
# FULL COVERAGE TABLE
# =========================================================
st.markdown("---")
st.subheader("Full Coverage Table")

st.caption(
    f"Table shows {len(display_df)} of {len(df)} ETFs after filters. "
    f"The ranking image above always starts from full coverage."
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    height=900,
)


# =========================================================
# EXPORT
# =========================================================
raw_export = df.copy()
raw_export.insert(0, "Run Timestamp ET", as_of_dt_et.strftime("%Y-%m-%d %H:%M:%S %Z"))
raw_export.insert(1, "Cache Bucket ET", as_of_bucket_key)

csv_bytes = raw_export.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download raw table as CSV",
    data=csv_bytes,
    file_name=f"etf_flow_pressure_proxy_{as_of_date}.csv",
    mime="text/csv",
)


# =========================================================
# FOOTNOTE
# =========================================================
st.caption(
    f"Last refresh: {as_of_dt_et.strftime('%Y-%m-%d %H:%M:%S %Z')} | "
    f"Source: Yahoo Finance OHLCV via yfinance | "
    f"Method: dollarized price-volume flow-pressure proxy, normalized pressure score, and data diagnostics | "
    f"Every ticker in the original coverage list remains in the ranking image and table | "
    f"© 2026 AD Fund Management LP"
)
