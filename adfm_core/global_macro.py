"""G20 comparisons without imputation or composite scores."""
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

COUNTRIES = (
    Country("Argentina", "ARG", "^MERV", "S&P MERVAL", "ARS"),
    Country("Australia", "AUS", "^AXJO", "S&P/ASX 200", "AUD", "IRLTLT01AUM156N"),
    Country("Brazil", "BRA", "^BVSP", "Ibovespa (total return)", "BRL"),
    Country("Canada", "CAN", "^GSPTSE", "S&P/TSX Composite", "CAD", "IRLTLT01CAM156N"),
    Country("China", "CHN", "000001.SS", "Shanghai Composite", "CNY"),
    Country("France", "FRA", "^FCHI", "CAC 40", "EUR", "IRLTLT01FRM156N"),
    Country("Germany", "DEU", "^GDAXI", "DAX (total return)", "EUR", "IRLTLT01DEM156N"),
    Country("India", "IND", "^NSEI", "Nifty 50", "INR"),
    Country("Indonesia", "IDN", "^JKSE", "Jakarta Composite", "IDR"),
    Country("Italy", "ITA", "FTSEMIB.MI", "FTSE MIB", "EUR", "IRLTLT01ITM156N"),
    Country("Japan", "JPN", "^N225", "Nikkei 225", "JPY", "IRLTLT01JPM156N"),
    Country("Mexico", "MEX", "^MXX", "S&P/BMV IPC", "MXN", "IRLTLT01MXM156N"),
    Country("Russia", "RUS", "IMOEX.ME", "MOEX Russia", "RUB"),
    Country("Saudi Arabia", "SAU", "^TASI.SR", "Tadawul All Share", "SAR"),
    Country("South Africa", "ZAF", "^J203.JO", "FTSE/JSE All Share", "ZAR", "IRLTLT01ZAM156N"),
    Country("South Korea", "KOR", "^KS11", "KOSPI", "KRW", "IRLTLT01KRM156N"),
    Country("Türkiye", "TUR", "XU100.IS", "BIST 100", "TRY"),
    Country("United Kingdom", "GBR", "^FTSE", "FTSE 100", "GBP", "IRLTLT01GBM156N"),
    Country("United States", "USA", "^GSPC", "S&P 500", "USD", "IRLTLT01USM156N"),
)
INDICATORS = {"GDP growth": "NY.GDP.MKTP.KD.ZG", "Unemployment": "SL.UEM.TOTL.ZS", "Inflation": "FP.CPI.TOTL.ZG"}
HORIZONS = {"1W": pd.DateOffset(weeks=1), "1M": pd.DateOffset(months=1), "3M": pd.DateOffset(months=3), "6M": pd.DateOffset(months=6), "1Y": pd.DateOffset(years=1)}

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

def equity_snapshot(series, today, horizon):
    series = clean(series)
    series = series.loc[(series.index < today.normalize()) & (series > 0)]
    result = empty_snapshot()
    if series.empty:
        return result
    end = series.index[-1]
    result.update(Level=float(series.iloc[-1]), Period=end.strftime("%Y-%m-%d"))
    if (today.normalize() - end).days > 7:
        result["Status"] = "Stale quote"
        return result
    anchor = (pd.Timestamp(today.year, 1, 1) - pd.Timedelta(days=1) if horizon == "YTD" else today.normalize() - HORIZONS[horizon])
    prior = series.loc[series.index <= anchor]
    if prior.empty or (anchor - prior.index[-1]).days > 7 or prior.index[-1] >= end:
        result["Status"] = "Missing baseline"
        return result
    result.update(Value=float((series.iloc[-1] / prior.iloc[-1] - 1) * 100), Baseline=prior.index[-1].strftime("%Y-%m-%d"), Status="Available")
    return result

@st.cache_data(ttl=900, max_entries=4, show_spinner=False)
def load_equities():
    configure_yfinance_cache()
    try:
        frame = yf.download([c.ticker for c in COUNTRIES], period="2y", interval="1d", auto_adjust=False, progress=False, threads=True, timeout=10, group_by="column")
        closes = frame["Close"] if not frame.empty else pd.DataFrame()
        series = {c.iso: clean(closes[c.ticker]) for c in COUNTRIES if c.ticker in closes}
        errors = {c.ticker: "No index observations returned" for c in COUNTRIES if c.iso not in series or series[c.iso].empty}
        return series, errors
    except Exception:
        return {}, {"Yahoo Finance": "Index download unavailable"}

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
        response = requests.get(f"https://api.worldbank.org/v2/country/{codes}/indicator/{indicator}", params={"format": "json", "date": "2010:2030", "per_page": 1000}, timeout=(4, 15))
        response.raise_for_status()
        return parse_world_bank(response.json(), indicator), {}
    except (requests.RequestException, ValueError, TypeError, KeyError):
        return {}, {"World Bank": "Economic data download unavailable"}

def period_snapshot(series, today, frequency, view, steps, period=None):
    """Require exact period matches; missing months or years are never bridged."""
    s = clean(series)
    s = pd.Series(s.to_numpy(), index=s.index.to_period(frequency))
    s = s.loc[~s.index.duplicated(keep="last")]
    current = today.to_period(frequency)
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
    result.update(Value=float(s.loc[end] - s.loc[start]) * (100 if frequency == "M" else 1), Baseline=str(start), Status="Available")
    return result

def comparison_period(series, today, frequency):
    """Latest completed period shared by 80% of fresh reporting countries."""
    coverage, reporters = {}, 0
    current = today.to_period(frequency)
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
            snap = period_snapshot(s, today, "M", view, {"1M": 1, "3M": 3, "6M": 6, "1Y": 12}[horizon], period)
            label = "10Y government yield · monthly average"
            source = f"https://fred.stlouisfed.org/series/{c.yield_id}" if c.yield_id else ""
            if not c.yield_id:
                snap["Status"] = "No comparable series"
        else:
            snap = period_snapshot(s, today, "Y", view, 1, period)
            label = metric + (" · ILO modeled estimate" if metric == "Unemployment" else " · annual")
            source = f"https://data.worldbank.org/indicator/{INDICATORS[metric]}?locations={c.iso}"
        rows.append({"Country": c.name, "ISO": c.iso, **snap, "Series": label, "Currency": c.currency if metric == "Equities" else "", "Source": source})
    return pd.DataFrame(rows)
