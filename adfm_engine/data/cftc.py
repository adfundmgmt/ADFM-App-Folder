"""CFTC public reporting data retrieval and cached market overlays."""
from __future__ import annotations
from datetime import date,timedelta
from typing import Final,Mapping
import numpy as np
import pandas as pd
import requests
from adfm_engine.analytics.cftc import *
from adfm_engine.analytics.cftc import _all_position_fields
from adfm_engine.cache import ttl_cache
from adfm_engine.data.market import fetch_daily_ohlcv, adjusted_ohlcv

def _request(report_type: str, params: Mapping[str, object], timeout: int = 30) -> pd.DataFrame:
    url = f"{CFTC_HOST}/resource/{DATASETS[report_type]}.json"
    headers = {"User-Agent": "ADFM-Analytics/1.0"}
    response = requests.get(url, params=dict(params), headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected CFTC API payload")
    return pd.DataFrame(payload)


def fetch_recent(report_type: str, years: int = 5, timeout: int = 30) -> pd.DataFrame:
    start = date.today() - timedelta(days=max(years, 1) * 366)
    select = ",".join((*BASE_FIELDS, *_all_position_fields(report_type)))
    params = {
        "$select": select,
        "$where": f"report_date_as_yyyy_mm_dd >= '{start.isoformat()}T00:00:00.000'",
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": 50000,
    }
    return normalize(_request(report_type, params, timeout), report_type)


def fetch_contract_history(
    report_type: str, contract_code: str, timeout: int = 30
) -> pd.DataFrame:
    code = str(contract_code).replace("'", "''").strip()
    select = ",".join((*BASE_FIELDS, *_all_position_fields(report_type)))
    params = {
        "$select": select,
        "$where": f"cftc_contract_market_code = '{code}'",
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": 10000,
    }
    return normalize(_request(report_type, params, timeout), report_type)


@ttl_cache(seconds=21600)
def load_report(report_type: str) -> tuple[pd.DataFrame, str]:
    try:
        return fetch_recent(report_type, years=5), ""
    except Exception as exc:
        return pd.DataFrame(), str(exc)


@ttl_cache(seconds=21600)
def load_history(report_type: str, contract_code: str) -> tuple[pd.DataFrame, str]:
    try:
        return fetch_contract_history(report_type, contract_code), ""
    except Exception as exc:
        return pd.DataFrame(), str(exc)


@ttl_cache(seconds=3600)
def load_price(ticker: str) -> tuple[pd.Series, str]:
    frames, failures = fetch_daily_ohlcv((ticker,), period="max")
    frame = frames.get(ticker)
    warning = ""
    if failures is not None and not failures.empty:
        matched = failures.loc[failures["Ticker"].eq(ticker), "Reason"]
        if not matched.empty:
            warning = str(matched.iloc[0])
    if frame is None or frame.empty:
        return pd.Series(dtype=float), warning or "No price history returned"
    close = pd.to_numeric(adjusted_ohlcv(frame).get("Close"), errors="coerce").dropna()
    return close, warning


