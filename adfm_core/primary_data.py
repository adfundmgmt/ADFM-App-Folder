"""Primary-source macro loading with explicit source and freshness metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
from pandas_datareader import data as web

from .data_registry import PRIMARY_MACRO_SERIES, SeriesDefinition


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

    for definition in definitions:
        try:
            raw = web.DataReader(definition.symbol, "fred", start, end_date)
            series = pd.to_numeric(raw[definition.symbol], errors="coerce")
            series.index = pd.to_datetime(series.index).tz_localize(None)
            values[definition.key] = series
            observed = series.dropna()
            statuses.append(
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
            statuses.append(
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

    panel = pd.DataFrame(values).sort_index() if values else pd.DataFrame()
    diagnostics = pd.DataFrame(asdict(item) for item in statuses)
    return panel, diagnostics
