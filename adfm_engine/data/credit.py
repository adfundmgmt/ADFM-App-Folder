"""Credit market and sovereign providers with explicit source selection."""
from __future__ import annotations
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import os,re
from html import unescape
from io import StringIO
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests
import yfinance as yf
from adfm_engine.cache import ttl_cache
from adfm_engine.data.primary import read_fred, read_fred_panel
from adfm_engine.analytics.credit_definitions import *
from adfm_engine.analytics.credit import clean_series, sovereign_move_rows, public_snapshot_rows, _adequate_global_coverage

@ttl_cache(seconds=1800)
def fetch_market_prices(tickers: Tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    try:
        raw = yf.download(
            list(tickers),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()

    out = pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        lvl1 = raw.columns.get_level_values(1)
        if "Close" in lvl0:
            block = raw["Close"]
            for ticker in tickers:
                if ticker in block.columns:
                    out[ticker] = pd.to_numeric(block[ticker], errors="coerce")
        elif "Close" in lvl1:
            for ticker in tickers:
                try:
                    out[ticker] = pd.to_numeric(raw[(ticker, "Close")], errors="coerce")
                except Exception:
                    try:
                        out[ticker] = pd.to_numeric(raw[("Close", ticker)], errors="coerce")
                    except Exception:
                        pass
    elif "Close" in raw.columns and len(tickers) == 1:
        out[tickers[0]] = pd.to_numeric(raw["Close"], errors="coerce")

    if out.empty:
        return out
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].sort_index().ffill().dropna(how="all")
    return out


def _get_secret(name):
    return os.getenv(name, "").strip() or None

def _parse_stooq_csv(text: str) -> pd.Series:
    if not text or "Date" not in text[:120]:
        return pd.Series(dtype=float)
    try:
        frame = pd.read_csv(StringIO(text))
    except Exception:
        return pd.Series(dtype=float)
    if "Date" not in frame.columns or "Close" not in frame.columns:
        return pd.Series(dtype=float)
    dates = pd.to_datetime(frame["Date"], errors="coerce")
    values = pd.to_numeric(frame["Close"], errors="coerce")
    series = pd.Series(values.values, index=dates).dropna().sort_index()
    return series[(series > -5.0) & (series < 100.0)]


def _fetch_stooq_symbol(symbol: str, start_date: date, end_date: date) -> pd.Series:
    d1 = start_date.strftime("%Y%m%d")
    d2 = end_date.strftime("%Y%m%d")
    symbol_q = symbol.lower()
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ADFM-Analytics/1.0)",
        "Accept": "text/csv,text/plain,*/*",
    }
    params = {"s": symbol_q, "d1": d1, "d2": d2, "i": "d"}
    stooq_key = _get_secret("STOOQ_API_KEY")
    if stooq_key:
        params["apikey"] = stooq_key
    for base_url in ("https://stooq.com/q/d/l/", "https://stooq.pl/q/d/l/"):
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=6)
            if response.status_code != 200:
                continue
            series = _parse_stooq_csv(response.text)
            if len(series) >= 2:
                return series
        except Exception:
            continue
    return pd.Series(dtype=float)


@ttl_cache(seconds=3600)
def fetch_stooq_sovereigns(start_date: date, end_date: date) -> Tuple[Dict[str, pd.Series], List[str]]:
    probe = _fetch_stooq_symbol("10YUSY.B", start_date, end_date)
    if probe.empty:
        key_note = "" if _get_secret("STOOQ_API_KEY") else " Configure STOOQ_API_KEY for authenticated historical access."
        return {}, ["Stooq daily sovereign-yield endpoint unavailable." + key_note]

    results: Dict[str, pd.Series] = {"United States": probe}
    errors: List[str] = []
    rows = [row for row in SOVEREIGN_UNIVERSE if row["country"] != "United States"]
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(_fetch_stooq_symbol, row["stooq"], start_date, end_date): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                series = future.result()
            except Exception:
                series = pd.Series(dtype=float)
            if series.empty:
                errors.append(str(row["country"]))
            else:
                results[str(row["country"])] = series
    return results, errors


@ttl_cache(seconds=21600)
def fetch_oecd_sovereigns(start_date: date, end_date: date) -> Dict[str, pd.Series]:
    id_to_country = {
        str(row["fred"]): str(row["country"])
        for row in SOVEREIGN_UNIVERSE
        if row.get("fred")
    }
    results: Dict[str, pd.Series] = {}
    series_ids = list(id_to_country.keys())

    try:
        raw = read_fred_panel(series_ids, start_date, end_date)
        if isinstance(raw, pd.DataFrame) and not raw.empty:
            for series_id, country in id_to_country.items():
                if series_id not in raw.columns:
                    continue
                series = clean_series(raw[series_id])
                if len(series) >= 2:
                    results[country] = series
    except Exception:
        pass

    missing = [series_id for series_id in series_ids if id_to_country[series_id] not in results]
    if missing:
        def fetch_one(series_id: str) -> Tuple[str, pd.Series]:
            try:
                raw = read_fred(series_id, str(start_date), str(end_date))
                if raw is not None and not raw.empty and series_id in raw.columns:
                    return series_id, clean_series(raw[series_id])
            except Exception:
                pass
            return series_id, pd.Series(dtype=float)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(fetch_one, series_id): series_id for series_id in missing}
            for future in as_completed(futures):
                series_id = futures[future]
                try:
                    _, series = future.result()
                except Exception:
                    series = pd.Series(dtype=float)
                if len(series) >= 2:
                    results[id_to_country[series_id]] = series
    return results


def _te_slug(country: str) -> str:
    return TE_SLUG_OVERRIDES.get(country, country.lower().replace(" ", "-"))


def _html_to_text(raw_html: str) -> str:
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", raw_html, flags=re.I | re.S)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _signed_points(fragment: str) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s+(?:percentage\s+)?points?", fragment, flags=re.I)
    if not match:
        return np.nan
    value = float(match.group(1))
    negative_words = ("lower", "down", "fell", "fallen", "declin", "decreas", "lost", "eased", "drop", "slid", "slipped")
    if any(word in fragment.lower() for word in negative_words):
        value = -value
    return value


def _parse_te_public_page(country: str, raw_html: str) -> Optional[dict]:
    text = _html_to_text(raw_html)
    if not text or "Bond Yield" not in text:
        return None

    actual_match = re.search(r"\bActual\s+(-?[0-9]+(?:\.[0-9]+)?)\s+Daily Change\b", text, flags=re.I)
    if not actual_match:
        actual_match = re.search(
            r"10Y Bond Yield[^.]{0,120}?(?:to|at|around)\s+(-?[0-9]+(?:\.[0-9]+)?)%",
            text,
            flags=re.I,
        )
    if not actual_match:
        return None
    end_yield = float(actual_match.group(1))
    if not (-5.0 < end_yield < 100.0):
        return None

    date_match = re.search(
        r"10Y Bond Yield[^.]{0,180}?\bon\s+([A-Z][a-z]+\s+\d{1,2},\s+20\d{2})",
        text,
        flags=re.I,
    )
    if not date_match:
        date_match = re.search(r"last updated on\s+([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?\s+of\s+20\d{2})", text, flags=re.I)
    if not date_match:
        return None
    end_date = pd.to_datetime(date_match.group(1), errors="coerce")
    if pd.isna(end_date):
        return None

    month_move = np.nan
    month_match = re.search(r"Over the past month,\s*(.{0,220}?)(?:and is|and stands|while it is|\. )", text, flags=re.I)
    if month_match:
        month_move = _signed_points(month_match.group(1)) * 100.0

    year_move = np.nan
    year_match = re.search(
        r"(?:is|stands)\s+([0-9]+(?:\.[0-9]+)?)\s+(?:percentage\s+)?points?\s+(higher|lower)\s+than\s+(?:a|one)\s+year\s+ago",
        text,
        flags=re.I,
    )
    if year_match:
        year_move = float(year_match.group(1)) * 100.0
        if year_match.group(2).lower() == "lower":
            year_move = -year_move

    return {
        "Country": country,
        "End Yield": end_yield,
        "End Date": pd.Timestamp(end_date).normalize(),
        "1M Move bp": month_move,
        "1Y Move bp": year_move,
    }


@ttl_cache(seconds=1800)
def fetch_te_public_snapshots() -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/151 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    def fetch_one(row: dict) -> Optional[dict]:
        country = str(row["country"])
        url = f"https://tradingeconomics.com/{_te_slug(country)}/government-bond-yield"
        try:
            response = requests.get(url, headers=headers, timeout=8)
            if response.status_code != 200:
                return None
            return _parse_te_public_page(country, response.text)
        except Exception:
            return None

    records: List[dict] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_one, row): row for row in SOVEREIGN_UNIVERSE}
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception:
                record = None
            if record:
                records.append(record)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame["End Date"] = pd.to_datetime(frame["End Date"], errors="coerce")
    return frame.dropna(subset=["Country", "End Yield", "End Date"])


def _fetch_te_sovereigns(api_key: str, start_date: date, end_date: date) -> Tuple[Dict[str, pd.Series], str]:
    try:
        snap = requests.get(
            "https://api.tradingeconomics.com/markets/bond",
            params={"c": api_key, "type": "10Y", "f": "json"},
            timeout=10,
        )
        snap.raise_for_status()
        payload = snap.json()
        if not isinstance(payload, list):
            return {}, "Trading Economics returned no 10Y snapshot."
    except Exception as exc:
        return {}, f"Trading Economics snapshot failed: {type(exc).__name__}"

    wanted = {str(row["country"]) for row in SOVEREIGN_UNIVERSE}
    aliases = {"Czech Republic": "Czechia", "Korea": "South Korea"}
    symbol_to_country: Dict[str, str] = {}
    for item in payload:
        raw_country = str(item.get("Country", "")).strip()
        country = aliases.get(raw_country, raw_country)
        symbol = str(item.get("Symbol", "")).strip()
        if country in wanted and symbol:
            symbol_to_country[symbol] = country
    if not symbol_to_country:
        return {}, "Trading Economics returned no matching sovereign symbols."

    out: Dict[str, pd.Series] = {}
    symbols = list(symbol_to_country.keys())
    for i in range(0, len(symbols), 12):
        batch = symbols[i : i + 12]
        try:
            response = requests.get(
                "https://api.tradingeconomics.com/markets/historical/" + ",".join(batch),
                params={
                    "c": api_key,
                    "d1": start_date.isoformat(),
                    "d2": end_date.isoformat(),
                    "f": "json",
                },
                timeout=15,
            )
            response.raise_for_status()
            history = response.json()
            if not isinstance(history, list):
                continue
        except Exception:
            continue

        grouped: Dict[str, List[tuple[pd.Timestamp, float]]] = {}
        for item in history:
            symbol = str(item.get("Symbol", "")).strip()
            if symbol not in symbol_to_country:
                continue
            dt = pd.to_datetime(item.get("Date"), dayfirst=True, errors="coerce")
            close = pd.to_numeric(item.get("Close"), errors="coerce")
            if pd.isna(dt) or pd.isna(close):
                continue
            grouped.setdefault(symbol, []).append((pd.Timestamp(dt), float(close)))

        for symbol, pairs in grouped.items():
            series = pd.Series(
                [value for _, value in pairs],
                index=[dt for dt, _ in pairs],
                dtype=float,
            ).sort_index()
            if len(series) >= 2:
                out[symbol_to_country[symbol]] = series
    return out, "" if out else "Trading Economics historical data unavailable."


def load_global_sovereign_moves(horizon: str) -> Tuple[pd.DataFrame, str, str]:
    end_date = date.today()
    years_needed = 6 if horizon == "5Y" else 4 if horizon == "3Y" else 2
    start_date = end_date - timedelta(days=365 * years_needed + 45)

    te_key = _get_secret("TRADING_ECONOMICS_API_KEY")
    if te_key:
        te_map, te_error = _fetch_te_sovereigns(te_key, start_date, end_date)
        te_rows = sovereign_move_rows(te_map, horizon, "Trading Economics API")
        if _adequate_global_coverage(te_rows):
            return te_rows, "Trading Economics API", te_error

    stooq_map, stooq_errors = fetch_stooq_sovereigns(start_date, end_date)
    stooq_rows = sovereign_move_rows(stooq_map, horizon, "Stooq daily")
    if _adequate_global_coverage(stooq_rows):
        note = f"{len(stooq_rows)} fresh countries loaded"
        if stooq_errors:
            note += f"; {len(stooq_errors)} unavailable"
        return stooq_rows, "Stooq daily", note

    public_snapshots = fetch_te_public_snapshots()
    if horizon in {"1M", "1Y"}:
        public_rows = public_snapshot_rows(public_snapshots, horizon)
        if _adequate_global_coverage(public_rows):
            return (
                public_rows,
                "Trading Economics public market pages",
                "Fresh public 10Y benchmark pages; provider-reported monthly/yearly change used directly.",
            )

    oecd_map: Dict[str, pd.Series] = {}
    if horizon in {"YTD", "1Y", "3Y", "5Y"}:
        oecd_map = fetch_oecd_sovereigns(start_date, end_date)

    if horizon in {"YTD", "3Y", "5Y"} and not public_snapshots.empty and oecd_map:
        hybrid_rows = public_snapshot_rows(public_snapshots, horizon, oecd_map)
        if _adequate_global_coverage(hybrid_rows):
            return (
                hybrid_rows,
                "Trading Economics current + OECD/FRED historical anchor",
                "Current benchmark yield is fresh; the comparison anchor is the nearest available OECD monthly 10Y observation on or before the target date.",
            )

    if horizon in {"1Y", "3Y", "5Y"} and oecd_map:
        oecd_rows = sovereign_move_rows(oecd_map, horizon, "OECD/FRED monthly")
        if _adequate_global_coverage(oecd_rows):
            return (
                oecd_rows,
                "OECD/FRED monthly",
                "Daily providers unavailable; structural monthly benchmark-yield data are shown with the observation dates disclosed.",
            )

    if horizon == "5D":
        note = (
            "Five-day moves require fresh daily benchmark history. Stooq now requires authenticated historical access in many hosted environments; "
            "configure STOOQ_API_KEY or TRADING_ECONOMICS_API_KEY to unlock the daily panel."
        )
    else:
        note = (
            "No source passed the minimum coverage/freshness test for this horizon. The page does not substitute bond ETFs or stale values for benchmark yields."
        )
    return pd.DataFrame(), "Unavailable", note


