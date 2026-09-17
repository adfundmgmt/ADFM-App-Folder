"""Market-data and holdings IO for sector rotation."""

from __future__ import annotations

from datetime import datetime, time as dt_time
from io import BytesIO
import time
from typing import Dict, Iterable, List, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    yf = None

from adfm_sector_rotation_config import DOWNLOAD_CHUNK_SIZE, DOWNLOAD_RETRIES, INTERVAL
from adfm_core.sector_rotation_catalog import LOOKBACK_PERIOD, STATE_STREET_HOLDINGS_URL


def _normalize_download(data: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    out: Dict[str, pd.Series] = {}
    fields = ("Adj Close", "Close")
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            for field in fields:
                for key in ((field, ticker), (ticker, field)):
                    if key in data.columns:
                        s = pd.to_numeric(data[key], errors="coerce")
                        if s.notna().any():
                            out[ticker] = s
                        break
                if ticker in out:
                    break
    elif len(tickers) == 1:
        ticker = tickers[0]
        for field in fields:
            if field in data.columns:
                s = pd.to_numeric(data[field], errors="coerce")
                if s.notna().any():
                    out[ticker] = s
                    break
    frame = pd.DataFrame(out)
    if frame.empty:
        return frame
    frame.index = pd.to_datetime(frame.index)
    try:
        if frame.index.tz is not None:
            frame.index = frame.index.tz_convert(None)
    except Exception:
        pass
    return frame.sort_index()


def _download_batch(tickers: Sequence[str], period: str, interval: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is required to download market data")
    if not tickers:
        return pd.DataFrame()
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            raw = yf.download(
                tickers=list(tickers), period=period, interval=interval,
                auto_adjust=False, progress=False, group_by="column", threads=True,
            )
            normalized = _normalize_download(raw, tickers)
            if not normalized.empty:
                return normalized
        except Exception:
            pass
        if attempt + 1 < DOWNLOAD_RETRIES:
            time.sleep(0.5 * (attempt + 1))
    return pd.DataFrame()


def drop_incomplete_us_session(prices: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    if prices.empty:
        return prices
    clock = now or datetime.now(ZoneInfo("America/New_York"))
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=ZoneInfo("America/New_York"))
    else:
        clock = clock.astimezone(ZoneInfo("America/New_York"))
    last_date = pd.to_datetime(prices.index.max()).date()
    current_date = clock.date()
    if last_date == current_date and clock.weekday() < 5 and clock.time() < dt_time(16, 15):
        return prices.loc[pd.to_datetime(prices.index).date < current_date].copy()
    return prices


def download_prices(tickers: Sequence[str], period: str = LOOKBACK_PERIOD, interval: str = INTERVAL) -> pd.DataFrame:
    unique = list(dict.fromkeys(t for t in tickers if t))
    pieces: List[pd.DataFrame] = []
    for start in range(0, len(unique), DOWNLOAD_CHUNK_SIZE):
        piece = _download_batch(unique[start : start + DOWNLOAD_CHUNK_SIZE], period, interval)
        if not piece.empty:
            pieces.append(piece)
    prices = pd.concat(pieces, axis=1) if pieces else pd.DataFrame()
    if not prices.empty:
        prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]
    for ticker in ("SPY", "ACWI"):
        if ticker in unique and (prices.empty or ticker not in prices or not prices[ticker].notna().any()):
            piece = _download_batch([ticker], period, interval)
            if not piece.empty:
                prices = pd.concat([prices, piece], axis=1)
                prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]
    if prices.empty:
        return prices
    prices.index = pd.to_datetime(prices.index)
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")
    return drop_incomplete_us_session(prices)


def required_tickers(catalog: pd.DataFrame, breadth_members: Mapping[str, Sequence[str]] | None = None) -> List[str]:
    tickers: List[str] = []
    tickers.extend(catalog["Ticker"].dropna().astype(str).tolist())
    tickers.extend(catalog["Broad Benchmark"].dropna().astype(str).tolist())
    tickers.extend(catalog["Parent Benchmark"].dropna().astype(str).tolist())
    for members in catalog["Members"].dropna():
        tickers.extend(list(members))
    if breadth_members:
        for members in breadth_members.values():
            tickers.extend(list(members))
    return list(dict.fromkeys(t for t in tickers if t))


def parse_state_street_holdings_frame(frame: pd.DataFrame) -> List[str]:
    header_row = None
    ticker_col = None
    for i in range(min(len(frame), 40)):
        values = [str(v).strip() if pd.notna(v) else "" for v in frame.iloc[i].tolist()]
        for j, value in enumerate(values):
            if value.lower() == "ticker":
                header_row, ticker_col = i, j
                break
        if header_row is not None:
            break
    if header_row is None or ticker_col is None:
        return []
    excluded = {"", "USD", "CASH", "N/A", "NA", "-", "--"}
    tickers: List[str] = []
    for value in frame.iloc[header_row + 1 :, ticker_col]:
        if pd.isna(value):
            continue
        ticker = str(value).strip().upper().replace(".", "-")
        if ticker in excluded or len(ticker) > 12 or " " in ticker:
            continue
        if not any(ch.isalpha() for ch in ticker):
            continue
        tickers.append(ticker)
    return list(dict.fromkeys(tickers))


def parse_state_street_holdings_bytes(content: bytes) -> List[str]:
    frame = pd.read_excel(BytesIO(content), header=None, engine="openpyxl")
    return parse_state_street_holdings_frame(frame)


def fetch_state_street_holdings(ticker: str, timeout: int = 15) -> List[str]:
    url = STATE_STREET_HOLDINGS_URL.format(ticker=ticker.lower())
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return parse_state_street_holdings_bytes(response.content)


def price_diagnostics(prices: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    rows = []
    latest_index = pd.to_datetime(prices.index.max()) if not prices.empty else pd.NaT
    for ticker in tickers:
        if ticker not in prices:
            rows.append({"Ticker": ticker, "Status": "Missing", "Latest Date": pd.NaT, "Valid Rows": 0})
            continue
        s = pd.to_numeric(prices[ticker], errors="coerce").dropna()
        latest = pd.to_datetime(s.index.max()) if not s.empty else pd.NaT
        stale = (latest_index - latest).days if pd.notna(latest_index) and pd.notna(latest) else np.nan
        rows.append({
            "Ticker": ticker,
            "Status": "OK" if len(s) else "Missing",
            "Latest Date": latest.date() if pd.notna(latest) else pd.NaT,
            "Valid Rows": int(len(s)),
            "Stale Days": stale,
        })
    return pd.DataFrame(rows)
