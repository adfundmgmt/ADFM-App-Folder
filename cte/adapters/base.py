"""
Adapter base layer: a retrying HTTP session, a parquet cache helper, and the
common return contract every adapter conforms to.

Return contract
---------------
Adapters return tidy long-form pandas DataFrames so the normalizer downstream
never has to special-case a source. Two shapes:

  yields :  [date, ccy, tenor, value, source, fetched_at]
  fx     :  [date, ccy, value, source, fetched_at]   # value = USD per 1 ccy
  reer   :  [date, ccy, value, source, fetched_at]
  tff    :  [date, ccy, metric, value, source, fetched_at]

`value` is always float in natural units (yields in %, fx as USD-per-unit).
`fetched_at` is a UTC timestamp stamped at pull time for cache provenance.
"""
from __future__ import annotations

import datetime as dt
import io
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from cte.config import CACHE_DIR, HTTP_TIMEOUT, HTTP_UA


def make_session(total_retries: int = 3, backoff: float = 0.6) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": HTTP_UA})
    return s


_SESSION = make_session()


def http_get(url: str, *, timeout: int = HTTP_TIMEOUT, **kw) -> requests.Response:
    r = _SESSION.get(url, timeout=timeout, **kw)
    r.raise_for_status()
    return r


def utcnow() -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.now(dt.timezone.utc)).tz_localize(None)


def read_csv_bytes(content: bytes, **kw) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content), **kw)


def cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.parquet"


def write_cache(df: pd.DataFrame, name: str) -> Path:
    p = cache_path(name)
    df.to_parquet(p, index=False)
    return p


def read_cache(name: str) -> pd.DataFrame | None:
    p = cache_path(name)
    if p.exists():
        return pd.read_parquet(p)
    return None


def tidy_yields(rows: list[dict], source: str) -> pd.DataFrame:
    """rows: list of {date, ccy, tenor, value} -> contract DataFrame."""
    if not rows:
        return pd.DataFrame(columns=["date", "ccy", "tenor", "value", "source", "fetched_at"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).copy()
    df["source"] = source
    df["fetched_at"] = utcnow()
    return df[["date", "ccy", "tenor", "value", "source", "fetched_at"]].sort_values(
        ["ccy", "tenor", "date"]
    ).reset_index(drop=True)
