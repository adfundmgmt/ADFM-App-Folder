"""Extracted ETF coverage and price-volume flow mathematics from the Streamlit page."""
from datetime import date
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from adfm_engine.data.market import fetch_daily_ohlcv

def fetch_prices(tickers, start_date, end_date, cache_key):
    # The native market loader batches, retries, caches and retains missing symbols.
    days = (end_date - start_date).days
    period = "2y" if days > 370 else "1y"
    frames, _ = fetch_daily_ohlcv(tickers, period=period)
    return {ticker: normalize_ohlcv(frame.loc[frame.index >= pd.Timestamp(start_date)]
            if not frame.empty else frame) for ticker in tickers
            for frame in (frames.get(ticker, pd.DataFrame()),)}


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
