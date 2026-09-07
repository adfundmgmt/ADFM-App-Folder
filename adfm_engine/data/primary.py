"""Primary-source macro loading with explicit source and freshness metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable
from io import StringIO
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from .registry import PRIMARY_MACRO_SERIES, SeriesDefinition


@dataclass(frozen=True)
class PrimarySeriesStatus:
    """Provider status for one primary-source series."""

    key: str
    symbol: str
    provider: str
    data_through: str | None
    observations: int
    status: str
    error: str | None = None


def read_fred(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Read the same FRED CSV observations, with bounded network waits."""
    response = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": symbol, "cosd": start, "coed": end}, timeout=(5, 20))
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text), index_col=0, parse_dates=True, na_values=".")
    if symbol not in frame:
        raise ValueError(f"FRED did not return the requested series {symbol}.")
    return frame.loc[start:end]


def fetch_fred_series(
    definitions: Iterable[SeriesDefinition] = PRIMARY_MACRO_SERIES,
    *,
    start: str = "2000-01-01",
    end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch registered FRED series and return values plus status diagnostics.

    Each series is requested independently. One provider failure cannot erase
    the remaining macro panel. Missing values remain missing.
    """

    values: dict[str, pd.Series] = {}
    statuses: list[PrimarySeriesStatus] = []
    end_date = end or datetime.now(timezone.utc).date().isoformat()

    def fetch_one(definition):
        try:
            raw = read_fred(definition.symbol, start, end_date)
            series = pd.to_numeric(raw[definition.symbol], errors="coerce")
            series.index = pd.to_datetime(series.index).tz_localize(None)
            observed = series.dropna()
            return series, (
                PrimarySeriesStatus(
                    key=definition.key,
                    symbol=definition.symbol,
                    provider=definition.provider,
                    data_through=observed.index.max().date().isoformat()
                    if not observed.empty
                    else None,
                    observations=len(observed),
                    status="OK" if not observed.empty else "EMPTY",
                )
            )
        except Exception as exc:
            return None, (
                PrimarySeriesStatus(
                    key=definition.key,
                    symbol=definition.symbol,
                    provider=definition.provider,
                    data_through=None,
                    observations=0,
                    status="FAILED",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    definitions = tuple(definitions)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for definition, (series, status) in zip(definitions, pool.map(fetch_one, definitions)):
            if series is not None:
                values[definition.key] = series
            statuses.append(status)
    panel = pd.DataFrame(values).sort_index() if values else pd.DataFrame()
    diagnostics = pd.DataFrame(asdict(item) for item in statuses)
    return panel, diagnostics


def read_fred_panel(symbols, start, end):
    """Multi-series FRED transport for the original sovereign fallback."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        frames = list(pool.map(lambda symbol: read_fred(symbol, str(start), str(end)), symbols))
    return pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
