"""G20 cross-country market and macro comparisons.

The module intentionally keeps slow official macro series separate from the fast
market layer. Missing observations are never imputed and no opaque composite
score is created.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from .market_data import configure_yfinance_cache
from .primary_data import fetch_fred_symbols


@dataclass(frozen=True)
class Country:
    name: str
    iso: str
    ticker: str
    index: str
    currency: str
    yield_id: str = ""
    fx_ticker: str = ""
    fx_inverse: bool = False


COUNTRIES = (
    Country("Argentina", "ARG", "^MERV", "S&P MERVAL", "ARS", fx_ticker="ARS=X", fx_inverse=True),
    Country("Australia", "AUS", "^AXJO", "S&P/ASX 200", "AUD", "IRLTLT01AUM156N", "AUDUSD=X", False),
    Country("Brazil", "BRA", "^BVSP", "Ibovespa (total return)", "BRL", fx_ticker="BRL=X", fx_inverse=True),
    Country("Canada", "CAN", "^GSPTSE", "S&P/TSX Composite", "CAD", "IRLTLT01CAM156N", "CAD=X", True),
    Country("China", "CHN", "000001.SS", "Shanghai Composite", "CNY", fx_ticker="CNY=X", fx_inverse=True),
    Country("France", "FRA", "^FCHI", "CAC 40", "EUR", "IRLTLT01FRM156N", "EURUSD=X", False),
    Country("Germany", "DEU", "^GDAXI", "DAX (total return)", "EUR", "IRLTLT01DEM156N", "EURUSD=X", False),
    Country("India", "IND", "^NSEI", "Nifty 50", "INR", fx_ticker="INR=X", fx_inverse=True),
    Country("Indonesia", "IDN", "^JKSE", "Jakarta Composite", "IDR", fx_ticker="IDR=X", fx_inverse=True),
    Country("Italy", "ITA", "FTSEMIB.MI", "FTSE MIB", "EUR", "IRLTLT01ITM156N", "EURUSD=X", False),
    Country("Japan", "JPN", "^N225", "Nikkei 225", "JPY", "IRLTLT01JPM156N", "JPY=X", True),
    Country("Mexico", "MEX", "^MXX", "S&P/BMV IPC", "MXN", "IRLTLT01MXM156N", "MXN=X", True),
    Country("Russia", "RUS", "IMOEX.ME", "MOEX Russia", "RUB", fx_ticker="RUB=X", fx_inverse=True),
    Country("Saudi Arabia", "SAU", "^TASI.SR", "Tadawul All Share", "SAR", fx_ticker="SAR=X", fx_inverse=True),
    Country("South Africa", "ZAF", "^J203.JO", "FTSE/JSE All Share", "ZAR", "IRLTLT01ZAM156N", "ZAR=X", True),
    Country("South Korea", "KOR", "^KS11", "KOSPI", "KRW", "IRLTLT01KRM156N", "KRW=X", True),
    Country("Türkiye", "TUR", "XU100.IS", "BIST 100", "TRY", fx_ticker="TRY=X", fx_inverse=True),
    Country("United Kingdom", "GBR", "^FTSE", "FTSE 100", "GBP", "IRLTLT01GBM156N", "GBPUSD=X", False),
    Country("United States", "USA", "^GSPC", "S&P 500", "USD", "IRLTLT01USM156N"),
)

INDICATORS = {
    "GDP growth": "NY.GDP.MKTP.KD.ZG",
    "Unemployment": "SL.UEM.TOTL.ZS",
    "Inflation": "FP.CPI.TOTL.ZG",
}
MARKET_HORIZONS = ("1D", "1W", "1M", "3M", "6M", "YTD", "1Y")
HORIZONS = {
    "1W": pd.DateOffset(weeks=1),
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
}


def clean(series):
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    dates = pd.to_datetime(values.index, errors="coerce")
    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_localize(None)
    out = pd.Series(values.to_numpy(), index=dates).dropna()
    out = out.loc[out.index.notna()]
    return out.loc[~out.index.duplicated(keep="last")].sort_index()


def empty_snapshot():
    return dict(Level=np.nan, Value=np.nan, Period="", Baseline="", Status="Unavailable")


def _latest_market_series(series, reference_date):
    """Keep plausible current-session observations, including Asia's next date."""
    s = clean(series)
    s = s.loc[s > 0]
    if s.empty:
        return s
    ceiling = pd.Timestamp(reference_date).normalize() + pd.Timedelta(days=1)
    return s.loc[s.index.normalize() <= ceiling]


def market_return_snapshot(series, today, horizon):
    """Return a point-to-point market move using the latest available observation."""
    s = _latest_market_series(series, today)
    result = empty_snapshot()
    if s.empty:
        return result

    end = s.index[-1]
    end_day = end.normalize()
    result.update(Level=float(s.iloc[-1]), Period=end_day.strftime("%Y-%m-%d"))
    age = (pd.Timestamp(today).normalize() - end_day).days
    if age > 7:
        result["Status"] = "Stale quote"
        return result

    if horizon == "1D":
        prior = s.loc[s.index < end]
        if prior.empty:
            result["Status"] = "Missing baseline"
            return result
        start = prior.index[-1]
    else:
        anchor = (
            pd.Timestamp(end_day.year, 1, 1) - pd.Timedelta(days=1)
            if horizon == "YTD"
            else end_day - HORIZONS[horizon]
        )
        prior = s.loc[s.index.normalize() <= anchor]
        if prior.empty:
            result["Status"] = "Missing baseline"
            return result
        start = prior.index[-1]
        if (anchor - start.normalize()).days > 7:
            result["Status"] = "Missing baseline"
            return result

    if start >= end:
        result["Status"] = "Missing baseline"
        return result
    result.update(
        Value=float((s.iloc[-1] / s.loc[start] - 1) * 100),
        Baseline=start.normalize().strftime("%Y-%m-%d"),
        Status="Current session" if age <= 0 else "Latest close",
    )
    return result


def equity_snapshot(series, today, horizon):
    return market_return_snapshot(series, today, horizon)


def currency_snapshot(series, today, horizon, inverse_quote=False):
    """Return local-currency performance versus USD.

    inverse_quote=False means Yahoo is USD per local currency (e.g. EURUSD=X).
    inverse_quote=True means Yahoo is local currency per USD (e.g. JPY=X).
    """
    raw = market_return_snapshot(series, today, horizon)
    if not np.isfinite(raw["Value"]):
        return raw
    if inverse_quote:
        start_value = raw["Level"] / (1 + raw["Value"] / 100)
        raw["Value"] = float((start_value / raw["Level"] - 1) * 100)
    return raw


def _merge_intraday(daily, intraday):
    daily = clean(daily)
    intraday = clean(intraday)
    if intraday.empty:
        return daily
    latest_time = intraday.index[-1]
    latest_day = latest_time.normalize()
    daily.loc[latest_day] = float(intraday.iloc[-1])
    return daily.sort_index()


def _close_frame(frame):
    if frame is None or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        try:
            return frame["Close"]
        except KeyError:
            return pd.DataFrame()
    return frame[["Close"]].rename(columns={"Close": "__single__"}) if "Close" in frame else pd.DataFrame()


@st.cache_data(ttl=120, max_entries=3, show_spinner=False)
def load_equities():
    """Load two years of daily history and overlay the freshest 5-minute bar."""
    configure_yfinance_cache()
    index_tickers = [c.ticker for c in COUNTRIES]
    fx_tickers = sorted({c.fx_ticker for c in COUNTRIES if c.fx_ticker})
    tickers = sorted(set(index_tickers + fx_tickers))
    try:
        daily_raw = yf.download(
            tickers,
            period="2y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            timeout=12,
            group_by="column",
        )
        daily = _close_frame(daily_raw)
        intraday = pd.DataFrame()
        try:
            intraday_raw = yf.download(
                tickers,
                period="5d",
                interval="5m",
                auto_adjust=False,
                progress=False,
                threads=True,
                timeout=10,
                group_by="column",
            )
            intraday = _close_frame(intraday_raw)
        except Exception:
            intraday = pd.DataFrame()

        by_ticker = {}
        errors = {}
        for ticker in tickers:
            d = clean(daily[ticker]) if ticker in daily else pd.Series(dtype=float)
            i = clean(intraday[ticker]) if ticker in intraday else pd.Series(dtype=float)
            merged = _merge_intraday(d, i)
            if merged.empty:
                errors[ticker] = "No market observations returned"
            else:
                by_ticker[ticker] = merged

        equities = {c.iso: by_ticker[c.ticker] for c in COUNTRIES if c.ticker in by_ticker}
        fx = {c.iso: by_ticker[c.fx_ticker] for c in COUNTRIES if c.fx_ticker and c.fx_ticker in by_ticker}
        return equities, fx, errors
    except Exception:
        return {}, {}, {"Yahoo Finance": "Market data download unavailable"}


def equity_matrix(equities, fx, today):
    rows = []
    for c in COUNTRIES:
        row = {
            "Country": c.name,
            "ISO": c.iso,
            "Index": c.index,
            "Currency": c.currency,
        }
        local = {}
        for horizon in MARKET_HORIZONS:
            eq = equity_snapshot(equities.get(c.iso, pd.Series(dtype=float)), today, horizon)
            local[horizon] = eq["Value"]
            row[horizon] = eq["Value"]
            row[f"Local {horizon}"] = eq["Value"]
            if c.currency == "USD":
                fx_move = 0.0
            elif c.fx_ticker:
                fx_snap = currency_snapshot(
                    fx.get(c.iso, pd.Series(dtype=float)),
                    today,
                    horizon,
                    inverse_quote=c.fx_inverse,
                )
                fx_move = fx_snap["Value"]
            else:
                fx_move = np.nan
            row[f"FX {horizon}"] = fx_move
            usd_move = (
                ((1 + eq["Value"] / 100) * (1 + fx_move / 100) - 1) * 100
                if np.isfinite(eq["Value"]) and np.isfinite(fx_move)
                else np.nan
            )
            row[f"USD {horizon}"] = usd_move

        latest = equity_snapshot(equities.get(c.iso, pd.Series(dtype=float)), today, "1D")
        row["Level"] = latest["Level"]
        row["Observation"] = latest["Period"]
        row["Status"] = latest["Status"]

        one_month, three_month = local["1M"], local["3M"]
        if np.isfinite(one_month) and np.isfinite(three_month):
            if one_month > 0 and three_month > 0:
                tape = "Leading"
            elif one_month > 0 and three_month <= 0:
                tape = "Rebounding"
            elif one_month <= 0 and three_month > 0:
                tape = "Fading"
            else:
                tape = "Struggling"
        else:
            tape = "Unavailable"
        row["Tape"] = tape
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(ttl=21600, max_entries=4, show_spinner=False)
def load_yields():
    countries = [c for c in COUNTRIES if c.yield_id]
    panel, status = fetch_fred_symbols(tuple(c.yield_id for c in countries), start="2015-01-01")
    series = {c.iso: clean(panel[c.yield_id]) for c in countries if c.yield_id in panel}
    errors = {}
    if not status.empty:
        for _, row in status.iterrows():
            if row.get("error"):
                errors[str(row.get("symbol", "FRED"))] = str(row["error"])
    return series, errors


def parse_world_bank(payload, indicator):
    if not isinstance(payload, list) or len(payload) != 2 or not isinstance(payload[1], list):
        raise ValueError("Invalid World Bank response")
    if int(payload[0].get("pages", 1)) > 1:
        raise ValueError("Incomplete World Bank response")
    observations = {}
    for row in payload[1]:
        iso = row.get("countryiso3code")
        if iso not in {c.iso for c in COUNTRIES} or row.get("indicator", {}).get("id") != indicator:
            continue
        value, year = row.get("value"), str(row.get("date", ""))
        if value is None or not year.isdigit():
            continue
        value = float(value)
        if np.isfinite(value):
            observations.setdefault(iso, {})[pd.Timestamp(int(year), 12, 31)] = value
    return {iso: clean(pd.Series(values)) for iso, values in observations.items()}


@st.cache_data(ttl=21600, max_entries=6, show_spinner=False)
def load_economics(metric):
    indicator = INDICATORS[metric]
    codes = ";".join(c.iso for c in COUNTRIES)
    try:
        response = requests.get(
            f"https://api.worldbank.org/v2/country/{codes}/indicator/{indicator}",
            params={"format": "json", "date": "2010:2030", "per_page": 1000},
            timeout=(4, 15),
        )
        response.raise_for_status()
        return parse_world_bank(response.json(), indicator), {}
    except (requests.RequestException, ValueError, TypeError, KeyError):
        return {}, {"World Bank": "Economic data download unavailable"}


def period_snapshot(series, today, frequency, view, steps, period=None):
    """Require exact period matches; missing months or years are never bridged."""
    s = clean(series)
    s = pd.Series(s.to_numpy(), index=s.index.to_period(frequency))
    s = s.loc[~s.index.duplicated(keep="last")]
    current = pd.Timestamp(today).to_period(frequency)
    s = s.loc[s.index < current]
    result = empty_snapshot()
    if s.empty:
        return result
    end = pd.Period(period, freq=frequency) if period else s.index[-1]
    result["Period"] = str(end)
    if end not in s.index:
        result["Status"] = "Missing period"
        return result
    result["Level"] = float(s.loc[end])
    if current.ordinal - end.ordinal > (4 if frequency == "M" else 2):
        result["Status"] = "Stale observation"
        return result
    if view == "Level":
        result.update(Value=float(s.loc[end]), Status="Available")
        return result
    start = end - steps
    if start not in s.index:
        result["Status"] = "Missing baseline"
        return result
    result.update(
        Value=float(s.loc[end] - s.loc[start]) * (100 if frequency == "M" else 1),
        Baseline=str(start),
        Status="Available",
    )
    return result


def comparison_period(series, today, frequency):
    """Latest completed period shared by 80% of fresh reporting countries."""
    coverage, reporters = {}, 0
    current = pd.Timestamp(today).to_period(frequency)
    lag = 4 if frequency == "M" else 2
    for raw in series.values():
        periods = set(clean(raw).index.to_period(frequency))
        recent = [p for p in periods if 0 < current.ordinal - p.ordinal <= lag]
        if not recent:
            continue
        reporters += 1
        for p in recent:
            coverage[p] = coverage.get(p, 0) + 1
    eligible = [p for p, count in coverage.items() if count >= max(1, np.ceil(.8 * reporters))]
    return str(max(eligible)) if eligible else None


def country_rows(series, today, metric, view="Level", horizon="1M", period=None):
    rows = []
    for c in COUNTRIES:
        s = series.get(c.iso, pd.Series(dtype=float))
        if metric == "Equities":
            snap = equity_snapshot(s, today, horizon)
            label, source = c.index, f"https://finance.yahoo.com/quote/{c.ticker}/history/"
        elif metric == "10Y yields":
            snap = period_snapshot(
                s,
                today,
                "M",
                view,
                {"1M": 1, "3M": 3, "6M": 6, "1Y": 12}[horizon],
                period,
            )
            label = "10Y government yield · monthly average"
            source = f"https://fred.stlouisfed.org/series/{c.yield_id}" if c.yield_id else ""
            if not c.yield_id:
                snap["Status"] = "No comparable series"
        else:
            snap = period_snapshot(s, today, "Y", view, 1, period)
            label = metric + (" · ILO modeled estimate" if metric == "Unemployment" else " · annual")
            source = f"https://data.worldbank.org/indicator/{INDICATORS[metric]}?locations={c.iso}"
        rows.append(
            {
                "Country": c.name,
                "ISO": c.iso,
                **snap,
                "Series": label,
                "Currency": c.currency if metric == "Equities" else "",
                "Source": source,
            }
        )
    return pd.DataFrame(rows)
