import time
from datetime import datetime, date
from functools import wraps
from io import BytesIO
from typing import Dict, Tuple, List, Sequence

import numpy as np
import pandas as pd
import pytz
import streamlit as st

from adfm_core.palette import PASTEL
from adfm_core.ui import PageHeader, render_footer, render_page_header, render_sidebar_about
from adfm_core.market_data import completed_daily_observations, download_market_data
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="ETF Trading Pressure", layout="wide")

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

PASTEL_GREEN = PASTEL["sage"]
PASTEL_RED = PASTEL["rose"]
PASTEL_GREY = "#8b949e"
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
    # Keep enough pre-window history to compare the current pressure regime
    # with like-for-like rolling windows over the prior year.
    padding = 420
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
# 99-name ETF signal universe.
# Broad allocator parking-lot ETFs such as QQQ, SPY, IVV, VOO,
# SPLG, VTI, VEA, IEFA, IEMG, VWO, BND, and AGG are excluded.
# =========================================================
etf_info = {
    # -------------------------
    # US EQUITY: TACTICAL INDEX, STYLE, FACTOR
    # -------------------------
    "DIA": ("Dow Industrials", "Old-economy large-cap cyclicals"),
    "IWM": ("Russell 2000", "Small-cap equity risk appetite"),
    "IJR": ("S&P SmallCap 600", "Higher-quality small-cap proxy"),
    "MDY": ("S&P MidCap 400", "Mid-cap equity risk appetite"),
    "RSP": ("S&P 500 Equal Weight", "Equal-weight breadth versus cap-weight leadership"),
    "IWF": ("Russell 1000 Growth", "Large-cap growth factor"),
    "IWD": ("Russell 1000 Value", "Large-cap value factor"),
    "IWO": ("Russell 2000 Growth", "Small-cap growth factor"),
    "IWN": ("Russell 2000 Value", "Small-cap value factor"),
    "MTUM": ("US Momentum", "Momentum factor"),
    "QUAL": ("US Quality", "Quality factor"),
    "USMV": ("US Minimum Volatility", "Defensive low-volatility factor"),
    "VLUE": ("US Value Factor", "Value factor"),
    "SPHB": ("S&P 500 High Beta", "High-beta equity risk appetite"),
    "SPLV": ("S&P 500 Low Volatility", "Low-volatility defensive equity"),
    "SCHD": ("US Dividend Quality", "Dividend / quality / defensive income equity"),

    # -------------------------
    # US SECTORS, INDUSTRIES, AND CYCLICAL SIGNALS
    # -------------------------
    "XLK": ("US Technology", "S&P 500 technology sector"),
    "SMH": ("Semiconductors", "Large-cap semiconductor leadership"),
    "SOXX": ("Semiconductor Breadth", "Semiconductor industry breadth"),
    "XSD": ("Equal-Weight Semis", "Equal-weight semiconductor risk appetite"),
    "XLC": ("US Communication Services", "Communication services / platform equities"),
    "XLY": ("US Discretionary", "Consumer discretionary sector"),
    "XLP": ("US Staples", "Consumer staples defensive sector"),
    "XLF": ("US Financials", "Large-cap financials"),
    "KRE": ("Regional Banks", "Regional banking and credit sensitivity"),
    "XLE": ("US Energy", "Integrated energy and energy beta"),
    "XOP": ("Oil & Gas E&P", "Exploration and production equities"),
    "OIH": ("Oil Services", "Oilfield services / energy capex cycle"),
    "XLI": ("US Industrials", "Industrial cyclicals"),
    "IYT": ("Transports", "Transport cyclicality and goods movement"),
    "XLU": ("US Utilities", "Defensive rate-sensitive equity"),
    "XLV": ("US Healthcare", "Healthcare defensives"),
    "IBB": ("Biotech Large-Cap", "Large-cap biotech"),
    "XBI": ("Biotech Equal-Weight", "Speculative biotech risk appetite"),
    "IHF": ("Healthcare Providers", "Managed care and healthcare services"),
    "XLB": ("US Materials", "Materials cyclicality"),
    "XME": ("Metals & Mining", "Mining and reflation sensitivity"),
    "XRT": ("Retail", "Retail / consumer impulse"),
    "XHB": ("Homebuilders", "Housing cycle proxy"),
    "ITB": ("Home Construction", "Home construction and housing beta"),
    "IYR": ("US Real Estate", "REITs and rate-sensitive real estate"),
    "IGV": ("Software", "Software and application tech"),
    "ARKK": ("Speculative Innovation", "Long-duration speculative growth"),

    # -------------------------
    # INTERNATIONAL AND EM EQUITY SIGNALS
    # -------------------------
    "EEM": ("Emerging Markets", "Liquid EM equity trading proxy"),
    "VGK": ("Europe", "Developed Europe equities"),
    "FEZ": ("Eurozone Large-Cap", "Eurozone blue-chip equity"),
    "EWU": ("United Kingdom", "UK equities"),
    "EWG": ("Germany", "Germany equities"),
    "EWQ": ("France", "France equities"),
    "EWI": ("Italy", "Italy equities"),
    "EWP": ("Spain", "Spain equities"),
    "EWJ": ("Japan", "Japan equities"),
    "EWY": ("South Korea", "Korea equities"),
    "EWT": ("Taiwan", "Taiwan equities / semis supply chain"),
    "FXI": ("China Large-Cap", "China offshore large caps"),
    "MCHI": ("China Broad Equity", "Broad China equity"),
    "KWEB": ("China Internet", "China internet and platform equities"),
    "ASHR": ("China A-Shares", "Onshore China equities"),
    "INDA": ("India", "India equities"),
    "EWZ": ("Brazil", "Brazil equities"),
    "EWW": ("Mexico", "Mexico equities"),
    "ECH": ("Chile", "Chile equities / copper sensitivity"),
    "ARGT": ("Argentina", "Argentina equities"),

    # -------------------------
    # RATES, CREDIT, CASH, INFLATION
    # -------------------------
    "SGOV": ("UST Bills", "0-3 month Treasury bills"),
    "BIL": ("UST Bills Alt", "1-3 month Treasury bills"),
    "SHY": ("UST 1-3y", "Short-duration Treasuries"),
    "IEF": ("UST 7-10y", "Intermediate-duration Treasuries"),
    "TLT": ("UST 20y+", "Long-duration Treasuries"),
    "EDV": ("Extended Duration UST", "Long-duration zero-coupon Treasury exposure"),
    "TIP": ("TIPS", "Inflation-linked Treasuries"),
    "STIP": ("Short TIPS", "Short-duration inflation-linked Treasuries"),
    "LQD": ("IG Credit", "Investment-grade corporate credit"),
    "VCIT": ("Intermediate IG Credit", "Intermediate-duration investment-grade credit"),
    "HYG": ("High Yield", "High-yield corporate credit"),
    "JNK": ("High Yield Alt", "High-yield corporate credit alternative"),
    "BKLN": ("Senior Loans", "Floating-rate senior loans"),
    "MBB": ("Agency MBS", "Mortgage-backed securities"),
    "EMB": ("EM Debt", "USD emerging-market sovereign debt"),
    "MUB": ("Municipal Bonds", "Investment-grade municipal bonds"),
    "GOVT": ("UST Aggregate", "Broad Treasury curve exposure"),

    # -------------------------
    # COMMODITIES, CRYPTO, VOLATILITY, FX
    # -------------------------
    "GLD": ("Gold", "Gold bullion"),
    "SLV": ("Silver", "Silver bullion"),
    "GDX": ("Gold Miners", "Large-cap gold miners"),
    "GDXJ": ("Junior Gold Miners", "Junior gold miners"),
    "USO": ("Crude Oil", "WTI crude oil"),
    "UNG": ("Natural Gas", "Natural gas futures"),
    "DBC": ("Broad Commodities", "Diversified commodity basket"),
    "DBA": ("Agriculture", "Agricultural commodities"),
    "CPER": ("Copper", "Copper futures"),
    "URA": ("Uranium", "Uranium and nuclear fuel cycle equities"),
    "REMX": ("Rare Earths", "Rare earth and strategic metals equities"),
    "IBIT": ("Bitcoin", "Spot Bitcoin ETF"),
    "ETHA": ("Ethereum", "Spot Ethereum ETF"),
    "VIXY": ("Equity Volatility", "Front-end VIX futures ETF"),
    "UUP": ("US Dollar", "US Dollar Index bullish exposure"),
    "FXE": ("Euro", "Euro versus US dollar"),
    "FXY": ("Japanese Yen", "Japanese yen versus US dollar"),
    "FXF": ("Swiss Franc", "Swiss franc versus US dollar"),
    "CEW": ("EM FX", "Emerging-market currency basket"),
}

etf_tickers = tuple(etf_info.keys())

US_EQUITY_TICKERS = {
    "DIA", "IWM", "IJR", "MDY", "RSP",
    "IWF", "IWD", "IWO", "IWN",
    "MTUM", "QUAL", "USMV", "VLUE", "SPHB", "SPLV", "SCHD",
    "XLK", "SMH", "SOXX", "XSD", "XLC", "XLY", "XLP", "XLF", "KRE",
    "XLE", "XOP", "OIH", "XLI", "IYT", "XLU", "XLV", "IBB", "XBI", "IHF",
    "XLB", "XME", "XRT", "XHB", "ITB", "IYR", "IGV", "ARKK",
}

INTERNATIONAL_EQUITY_TICKERS = {
    "EEM", "VGK", "FEZ", "EWU", "EWG", "EWQ", "EWI", "EWP",
    "EWJ", "EWY", "EWT",
    "FXI", "MCHI", "KWEB", "ASHR", "INDA",
    "EWZ", "EWW", "ECH", "ARGT",
}

RATES_CREDIT_TICKERS = {
    "SGOV", "BIL", "SHY", "IEF", "TLT", "EDV",
    "TIP", "STIP",
    "LQD", "VCIT", "HYG", "JNK", "BKLN", "MBB", "EMB", "MUB", "GOVT",
}

COMMODITY_TICKERS = {
    "GLD", "SLV", "GDX", "GDXJ",
    "USO", "UNG", "DBC", "DBA", "CPER",
    "URA", "REMX",
}

FX_TICKERS = {
    "UUP", "FXE", "FXY", "FXF", "CEW",
}

CRYPTO_VOL_TICKERS = {
    "IBIT", "ETHA", "VIXY",
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

    needed = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

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
    deadline: float = None,
) -> pd.DataFrame:
    deadline = time.monotonic() + 25.0 if deadline is None else deadline
    for i in range(attempts):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            raw = download_market_data(
                tickers=list(tickers),
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                threads=threads,
                progress=False,
                recovery_budget_seconds=remaining,
                timeout=min(10.0, remaining),
            )

            if raw is not None and not raw.empty:
                return raw

        except Exception:
            pass

        if i < attempts - 1:
            time.sleep(min(delay * (i + 1), max(0.0, deadline - time.monotonic())))

    return pd.DataFrame()


def fetch_prices(
    tickers: Tuple[str, ...],
    start_date: date,
    end_date: date,
    cache_key: str,
    batch_size: int = 35,
) -> Dict[str, pd.DataFrame]:
    _ = cache_key
    deadline = time.monotonic() + 25.0

    out: Dict[str, pd.DataFrame] = {}

    for batch in chunked(tickers, batch_size):
        if time.monotonic() >= deadline:
            break
        raw = safe_yf_download(batch, start_date, end_date, threads=True, deadline=deadline)

        for tk in batch:
            df = normalize_ohlcv(extract_ticker_frame(raw, tk, len(batch)))

            if df.empty and len(batch) > 1 and time.monotonic() < deadline:
                raw_single = safe_yf_download((tk,), start_date, end_date, threads=False, attempts=2, deadline=deadline)
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


def compute_pressure_percentile(df: pd.DataFrame, window_obs: int, history_days: int = 365) -> float:
    """Percentile of the current normalized pressure versus like-for-like rolling windows."""
    if df is None or df.empty or int(window_obs) < 2:
        return np.nan

    mfv = compute_money_flow_proxy(df)
    traded_value = compute_traded_value(df)
    if mfv.empty or traded_value.empty:
        return np.nan

    window_obs = int(window_obs)
    rolling_flow = mfv.rolling(window_obs, min_periods=window_obs).sum()
    rolling_turnover = traded_value.rolling(window_obs, min_periods=window_obs).sum()
    rolling_score = (rolling_flow / rolling_turnover.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    if rolling_score.empty:
        return np.nan

    current = float(rolling_score.iloc[-1])
    cutoff = rolling_score.index[-1] - pd.Timedelta(days=int(history_days))
    history = rolling_score.loc[rolling_score.index >= cutoff].dropna()
    if len(history) < 20:
        return np.nan

    return float((history <= current).mean() * 100.0)


def classify_price_pressure(return_pct: float, pressure_score: float) -> str:
    if pd.isna(return_pct) or pd.isna(pressure_score):
        return "N/A"
    if return_pct > 0 and pressure_score > 0:
        return "Confirmed accumulation"
    if return_pct > 0 and pressure_score < 0:
        return "Distribution into strength"
    if return_pct < 0 and pressure_score > 0:
        return "Accumulation into weakness"
    if return_pct < 0 and pressure_score < 0:
        return "Confirmed distribution"
    return "Mixed"


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
@st.cache_data(show_spinner=True, ttl=900, max_entries=64)
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
        px = completed_daily_observations(price_map.get(tk, pd.DataFrame()).copy(), tk)

        lookback_px = pd.DataFrame()
        if px is not None and not px.empty:
            lookback_px = px.loc[px.index >= cutoff].copy()

        full_daily_proxy = compute_money_flow_proxy(px) if px is not None and not px.empty else pd.Series(dtype="float64")

        lookback_proxy = np.nan
        pressure_score = np.nan
        pressure_adv = np.nan
        pressure_percentile = np.nan
        ret = np.nan
        last_price = np.nan
        adv = np.nan

        if not lookback_px.empty:
            lookback_daily_proxy = compute_money_flow_proxy(lookback_px)

            if not lookback_daily_proxy.empty:
                lookback_proxy = float(lookback_daily_proxy.sum())

            pressure_score = compute_pressure_score(lookback_px)

            close = pd.to_numeric(lookback_px["Close"], errors="coerce").dropna()
            adjusted = pd.to_numeric(lookback_px.get("Adj Close", lookback_px["Close"]), errors="coerce").dropna()
            if len(adjusted) >= 2 and adjusted.iloc[0] != 0:
                ret = float((adjusted.iloc[-1] / adjusted.iloc[0] - 1.0) * 100.0)

            if len(close) >= 1:
                last_price = float(close.iloc[-1])

            adv_series = compute_traded_value(lookback_px)
            if adv_series.notna().any():
                adv = float(adv_series.mean())

            if pd.notna(lookback_proxy) and pd.notna(adv) and adv > 0:
                pressure_adv = float(lookback_proxy / adv)

            pressure_percentile = compute_pressure_percentile(px, len(lookback_px))

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
                "Pressure / ADV (x)": pressure_adv,
                "1Y Pressure Percentile": pressure_percentile,
                "Price / Pressure": classify_price_pressure(ret, pressure_score),
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
    render_sidebar_about("14_ETF_Flow_Pressure_Proxy.py")
    period_label = st.radio(
        "Lookback Window",
        list(lookback_dict.keys()),
        index=0,
    )
    asset_filter = st.selectbox(
        "Asset Class",
        ["All"] + sorted({infer_asset_class(ticker) for ticker in etf_tickers}),
        index=0,
    )
    st.caption(f"As of: {as_of_dt_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")


period_days = int(lookback_dict[period_label])
flow_col = f"{period_label} Flow Pressure Proxy"
return_col = f"{period_label} Return %"


# =========================================================
# HEADER
# =========================================================
render_page_header(
    PageHeader(
        title="ETF Trading Pressure",
        description=(
            "Dollar-weighted price-volume pressure across the full 99-name tactical ETF universe. "
            "Positive values indicate trading concentrated toward session highs; negative values "
            "indicate trading concentrated toward session lows."
        ),
        eyebrow="ADFM Flows and Sentiment",
    )
)


# =========================================================
# DATA
# =========================================================
df = build_table(
    tickers=etf_tickers,
    period_label=period_label,
    period_days=period_days,
    as_of=as_of_date,
    cache_key=as_of_bucket_key,
)

view_df = df if asset_filter == "All" else df.loc[df["Asset Class"] == asset_filter].copy()


# =========================================================
# FULL-UNIVERSE DOLLAR PRESSURE RANKING
# =========================================================
st.subheader("Dollar-Weighted Trading Pressure")

chart_df = build_chart_dataframe(
    source_df=view_df,
    metric=flow_col,
    sort_mode="Positive to Negative",
)

st.caption(
    f"{period_label} · showing {len(chart_df)} of {len(df)} ETFs"
    + ("" if asset_filter == "All" else f" · {asset_filter}")
    + ". Every ETF remains in the ranking; unavailable observations are shown in grey."
)

if chart_df.empty:
    st.info("No ETF observations are available for this selection.")
else:
    vals = chart_df["Chart Value"].astype(float)
    raw_vals = chart_df["Raw Value"]

    x_min = min(float(vals.min()), 0.0)
    x_max = max(float(vals.max()), 0.0)
    span = (x_max - x_min) if (x_max - x_min) > 0 else 1.0
    pad = 0.075 * span

    n = len(chart_df)
    fig_h = max(7.5, min(29.0, 0.255 * n + 1.8))
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
    ax.set_xlabel("Estimated dollar trading pressure", fontsize=9, color=TEXT_DARK)
    ax.set_title(
        f"{period_label} Dollar-Weighted Trading Pressure | {len(chart_df)} ETFs",
        fontsize=10.8,
        pad=7,
        color=TEXT_DARK,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(axis_fmt_currency))
    ax.tick_params(axis="y", labelsize=y_font, pad=0.6, length=0, colors=TEXT_DARK)
    ax.tick_params(axis="x", labelsize=8.2, colors=AXIS_GREY)
    ax.grid(False)
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
        label_txt = "NA" if was_missing else fmt_compact_cur(raw)
        if chart_value > 0:
            x_text, ha = chart_value + text_pad, "left"
        elif chart_value < 0:
            x_text, ha = chart_value - text_pad, "right"
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
# DENSE UNDERLYING TABLE
# =========================================================
st.subheader("ETF Pressure Detail")

display_df = view_df.copy()

display_df = display_df.rename(
    columns={
        "Category": "Exposure",
        flow_col: f"{period_label} Dollar Pressure",
        "Week to Date": "WTD $ Pressure",
        return_col: f"{period_label} Return (%)",
        "Avg Daily Dollar Vol": "Avg Daily $ Volume",
    }
)

display_cols = [
    "Ticker",
    "Exposure",
    "Asset Class",
    f"{period_label} Dollar Pressure",
    "WTD $ Pressure",
    "Pressure / ADV (x)",
    "1Y Pressure Percentile",
    f"{period_label} Return (%)",
    "Price / Pressure",
    "Avg Daily $ Volume",
]
if display_df["Data Status"].ne("OK").any():
    display_cols.append("Data Status")

display_df = display_df[display_cols].copy()
display_df = display_df.sort_values(f"{period_label} Dollar Pressure", ascending=False, na_position="last")

signed_cols = [
    f"{period_label} Dollar Pressure",
    "WTD $ Pressure",
    "Pressure / ADV (x)",
    f"{period_label} Return (%)",
]

def tint_signed(value):
    if value is None or pd.isna(value):
        return ""
    return f"color: {'#28734c' if float(value) > 0 else '#a33d4b' if float(value) < 0 else '#64748b'}"

styled = display_df.style.format(
    {
        f"{period_label} Dollar Pressure": fmt_compact_cur,
        "WTD $ Pressure": fmt_compact_cur,
        "Pressure / ADV (x)": lambda x: "" if pd.isna(x) else f"{float(x):+.2f}x",
        "1Y Pressure Percentile": lambda x: "" if pd.isna(x) else f"{float(x):.0f}th",
        f"{period_label} Return (%)": lambda x: "" if pd.isna(x) else f"{float(x):+.2f}",
        "Avg Daily $ Volume": fmt_compact_cur,
    },
    na_rep="",
).map(tint_signed, subset=signed_cols)

st.dataframe(
    styled,
    hide_index=True,
    width="stretch",
    height=820,
)

st.caption(
    "Dollar pressure = close-location multiplier × typical price × shares traded, summed across the selected window. "
    "Pressure / ADV expresses that cumulative signal in average daily turnover equivalents. "
    "The 1Y percentile compares normalized pressure with like-for-like rolling windows over the prior year. "
    "Price / Pressure highlights confirmation or divergence between return direction and trading pressure. "
    "This is a public-market trading-pressure proxy, not reported ETF creations or redemptions."
)

raw_export = df.copy()
raw_export.insert(0, "Run Timestamp ET", as_of_dt_et.strftime("%Y-%m-%d %H:%M:%S %Z"))
raw_export.insert(1, "Cache Bucket ET", as_of_bucket_key)

st.download_button(
    "Download ETF pressure data",
    raw_export.to_csv(index=False).encode("utf-8"),
    file_name=f"etf_trading_pressure_{as_of_date}.csv",
    mime="text/csv",
)

render_footer(
    data_note=(
        "ETF trading pressure and adjusted returns: Yahoo Finance OHLCV. "
        "The dollar-weighted pressure measure is derived from price location and traded value."
    )
)
