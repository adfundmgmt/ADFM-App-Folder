"""Validated FRED transport and atomic last-good storage, independent of the UI.

Known macro series ship with scheduled snapshots. Runtime cache writes never
alter those committed snapshots. A failed refresh cannot replace good data.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import BoundedSemaphore, Lock

import numpy as np
import pandas as pd
import requests

from .fred_registry import policy_for

ROOT = Path(__file__).resolve().parents[1]
API_ROOT = "https://api.stlouisfed.org/fred"
CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_FAILURES: OrderedDict[str, tuple[float, str]] = OrderedDict()
_FAILURE_LOCK = Lock()
_DOWNLOAD_SLOTS = BoundedSemaphore(3)
PAGE_SIZE = 100000


class FredError(RuntimeError):
    """A safe error message that never contains a request URL or API credential."""


def utcnow() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def api_key() -> str:
    key = os.getenv("FRED_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st

        return str(st.secrets.get("FRED_API_KEY", "")).strip()
    except Exception:
        return ""


def normalize(symbol: str, frame: pd.DataFrame) -> pd.Series:
    if symbol not in frame:
        raise FredError("Response is missing the requested series")
    values = frame[symbol].replace({".": np.nan, "": np.nan})
    numeric = pd.to_numeric(values, errors="coerce")
    if (values.notna() & numeric.isna()).any():
        raise FredError("Response contains nonnumeric observations")
    dates = pd.to_datetime(frame.index, errors="coerce", utc=True)
    if dates.isna().any() or dates.duplicated().any():
        raise FredError("Response contains invalid or duplicate observation dates")
    out = pd.Series(numeric.to_numpy(dtype=float), index=dates.tz_localize(None).normalize(), name=symbol).sort_index()
    out.index.name = None
    if out.dropna().empty or np.isinf(out.to_numpy()).any():
        raise FredError("Response is empty or contains infinite observations")
    policy = policy_for(symbol)
    if policy.minimum is not None and out.lt(policy.minimum).any():
        raise FredError("Observation is below the registered unit range")
    if policy.maximum is not None and out.gt(policy.maximum).any():
        raise FredError("Observation is above the registered unit range")
    if symbol == "USREC" and not out.dropna().isin([0, 1]).all():
        raise FredError("Recession indicator must be zero, one, or missing")
    return out


def request(session, url: str, params: dict, *, attempts: int = 3):
    """Retry transient failures only; cap Retry-After and connection/read waits."""
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=(4, 12))
        except requests.RequestException:
            response = None
        if response is not None and response.status_code == 200:
            return response
        status = response.status_code if response is not None else None
        if status is not None and status not in {429, 500, 502, 503, 504}:
            raise FredError(f"FRED rejected the request (HTTP {status})")
        if attempt + 1 < attempts:
            delay = 0.5 * 2**attempt
            if response is not None:
                try:
                    delay = max(delay, min(float(response.headers.get("Retry-After", 0)), 5))
                except ValueError:
                    pass
            time.sleep(delay)
    raise FredError("FRED temporarily unavailable after bounded retries")


def download(symbol: str, start: str, end: str, *, key: str = "", vintage: str | None = None):
    """Return raw natural-unit observations and provider metadata.

Vintage requests never fall back to today's CSV data. Metadata is validated
on the API path; CSV responses use the versioned source contract.
"""
    policy = policy_for(symbol)
    if vintage and not key:
        raise FredError("Historical vintages require a configured FRED API key")
    with requests.Session() as session:
        if key:
            session.headers.update({"Accept": "application/json"})
            params = {"api_key": key, "file_type": "json", "series_id": symbol}
            try:
                metadata = request(session, API_ROOT + "/series", params).json()["seriess"][0]
                units = str(metadata["units"])
                frequency = str(metadata["frequency"])
                if policy.units != "provider units" and not units.lower().startswith(policy.units.lower()):
                    raise FredError("Provider units differ from the registered series definition")
                if policy.frequency != "unknown" and not frequency.startswith(policy.frequency):
                    raise FredError("Provider frequency differs from the registered series definition")
                params.update({"observation_start": start, "observation_end": end, "units": "lin", "limit": PAGE_SIZE})
                if vintage:
                    params.update({"realtime_start": vintage, "realtime_end": vintage})
                rows = []
                for offset in range(0, PAGE_SIZE * 10, PAGE_SIZE):
                    payload = request(session, API_ROOT + "/series/observations", {**params, "offset": offset}).json()
                    block = payload["observations"]
                    rows.extend(block)
                    if len(rows) >= int(payload["count"]):
                        break
                    if not block:
                        raise FredError("FRED returned an incomplete observation page")
                else:
                    raise FredError("FRED pagination exceeded the safety limit")
                frame = pd.DataFrame(rows)
                if frame.empty:
                    raise FredError("FRED returned no observations")
                frame = frame.set_index("date")[["value"]].rename(columns={"value": symbol})
            except (KeyError, IndexError, ValueError) as exc:
                raise FredError("FRED returned an invalid API response") from exc
            source = "FRED API / ALFRED" if vintage else "FRED API"
        else:
            response = request(session, CSV_URL, {"id": symbol, "cosd": start, "coed": end})
            try:
                frame = pd.read_csv(io.BytesIO(response.content), index_col=0)
            except Exception as exc:
                raise FredError("FRED returned an invalid CSV response") from exc
            units, frequency, source = policy.units, policy.frequency, "FRED CSV (API key not configured)"
        series = normalize(symbol, frame)
        return series.loc[start:end], {"source": source, "units": units, "frequency": frequency, "vintage": vintage}


def path_for(directory: Path, symbol: str, vintage: str | None = None) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_]+", symbol):
        raise FredError("Invalid FRED series identifier")
    if vintage and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", vintage):
        raise FredError("Invalid vintage date")
    return directory / f"{symbol}{'_' + vintage if vintage else ''}.json.gz"


def read_record(path: Path, symbol: str):
    try:
        payload = json.loads(gzip.decompress(path.read_bytes()))
        if payload["schema"] != 1 or payload["symbol"] != symbol:
            return None
        frame = pd.DataFrame(payload["observations"], columns=["date", symbol]).set_index("date")
        series = normalize(symbol, frame)
        if pd.Timestamp(payload["fetched_at"]).tzinfo is None:
            return None
        pd.Timestamp(payload["requested_start"])
        pd.Timestamp(payload["requested_end"])
        return series, payload
    except (OSError, EOFError, ValueError, KeyError, TypeError, FredError):
        return None


def write_record(path: Path, series: pd.Series, metadata: dict):
    payload = {**metadata, "schema": 1, "symbol": series.name,
               "observations": [[d.strftime("%Y-%m-%d"), None if pd.isna(v) else float(v)] for d, v in series.items()]}
    data = gzip.compress(json.dumps(payload, allow_nan=False, sort_keys=True).encode(), mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(data)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


@dataclass
class FredResult:
    series: pd.Series
    metadata: dict


class FredStore:
    def __init__(self, cache_dir: Path | None = None, snapshot_dir: Path | None = None):
        self.cache_dir = cache_dir if cache_dir is not None else Path(os.getenv("ADFM_FRED_CACHE", ROOT / "data/last_good/fred"))
        self.snapshot_dir = snapshot_dir if snapshot_dir is not None else ROOT / "data/fred"

    def get(self, symbol: str, start: str, end: str, *, refresh: bool = False,
            offline: bool = False, key: str | None = None, vintage: str | None = None) -> FredResult:
        today = utcnow()
        end = min(pd.Timestamp(end), today.tz_localize(None).normalize()).date().isoformat()
        start = pd.Timestamp(start).date().isoformat()
        if start > end:
            raise FredError("Observation start is after end")
        runtime = path_for(self.cache_dir, symbol, vintage)
        snapshot = read_record(path_for(self.snapshot_dir, symbol, vintage), symbol)
        candidates = [read_record(runtime, symbol), snapshot]
        candidates = [r for r in candidates if r is not None]
        saved = max(candidates, key=lambda r: pd.Timestamp(r[1]["fetched_at"])) if candidates else None
        failure = ""
        mode = "saved snapshot"
        covered = saved and saved[1]["requested_start"] <= start
        recent = saved and (today - pd.Timestamp(saved[1]["fetched_at"])).total_seconds() < 6 * 3600
        # Registered snapshots keep page rendering independent of provider
        # availability. The scheduled writer performs refreshes separately.
        shipped = snapshot is not None
        should_fetch = not offline and (refresh or (not (covered and (recent or shipped))))
        failure_key = str(runtime)
        with _FAILURE_LOCK:
            prior_failure = _FAILURES.get(failure_key)
        if not refresh and prior_failure and time.monotonic() - prior_failure[0] < 300:
            should_fetch = False
            failure = prior_failure[1]
            mode = "last-good fallback"
        if should_fetch:
            try:
                fetch_start = min(start, saved[1]["requested_start"]) if saved else start
                with _DOWNLOAD_SLOTS:
                    series, metadata = download(symbol, fetch_start, end, key=api_key() if key is None else key, vintage=vintage)
                observed = series.dropna()
                if observed.empty:
                    raise FredError("FRED returned no usable observations")
                if saved:
                    old = saved[0].dropna()
                    if not old.empty and observed.index.max() < old.index.max():
                        raise FredError("Refresh is older than the saved dataset")
                    overlap = old.index.intersection(series.index)
                    if len(overlap) and series.reindex(overlap).isna().sum() > max(2, len(overlap) * .05):
                        raise FredError("Refresh unexpectedly removed previously observed values")
                    if policy_for(symbol).publish_snapshot and len(old) > 100 and len(observed) < len(old) * .9:
                        raise FredError("Refresh unexpectedly truncated the saved history")
                metadata.update({"fetched_at": today.isoformat(), "requested_start": fetch_start, "requested_end": end})
                # A fresh authoritative response replaces its full requested
                # window, including revisions. Never splice older vintages in.
                write_record(runtime, series, metadata)
                saved = series, metadata
                mode = "downloaded"
                with _FAILURE_LOCK:
                    _FAILURES.pop(failure_key, None)
            except Exception as exc:
                failure = str(exc) if isinstance(exc, FredError) else f"Refresh failed ({type(exc).__name__})"
                mode = "last-good fallback"
                with _FAILURE_LOCK:
                    _FAILURES[failure_key] = time.monotonic(), failure
                    _FAILURES.move_to_end(failure_key)
                    while len(_FAILURES) > 128:
                        _FAILURES.popitem(last=False)
        if saved is None:
            return FredResult(pd.Series(dtype=float, name=symbol), {"status": "FAILED", "error": failure or "No saved data available", "symbol": symbol})
        series, metadata = saved
        series = series.loc[start:end]
        valid = series.dropna()
        latest = valid.index.max() if len(valid) else None
        reference = pd.Timestamp(end)
        age_limit = policy_for(symbol).max_age_days
        if policy_for(symbol).frequency == "unknown":
            frequency = str(metadata.get("frequency", ""))
            median_gap = valid.index.to_series().diff().dt.days.median() if len(valid) > 1 else 0
            if frequency.startswith("Quarterly") or median_gap >= 70:
                age_limit = 170
            elif frequency.startswith("Monthly") or median_gap >= 20:
                age_limit = 100
        stale = latest is not None and (reference - latest).days > age_limit
        status = "EMPTY" if valid.empty else "STALE" if stale else "CACHED" if mode != "downloaded" else "OK"
        details = {**{k: v for k, v in metadata.items() if k != "observations"},
                   "symbol": symbol, "status": status, "delivery": mode, "error": failure or None,
                   "data_from": valid.index.min().date().isoformat() if len(valid) else None,
                   "data_through": latest.date().isoformat() if latest is not None else None,
                   "observations": len(valid), "requested_start": start, "requested_end": end,
                   "history_years": (latest - valid.index.min()).days / 365.25 if len(valid) else 0}
        return FredResult(series, details)
