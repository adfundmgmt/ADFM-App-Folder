"""Shared macro loading: snapshots first, bounded provider recovery, diagnostics."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from .data_registry import PRIMARY_MACRO_SERIES, SeriesDefinition
from .fred_store import FredStore


def fetch_fred_series(
    definitions: Iterable[SeriesDefinition] = PRIMARY_MACRO_SERIES,
    *, start: str = "2000-01-01", end: str | None = None,
    refresh: bool = False, offline: bool = False,
    store: FredStore | None = None, vintage: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    definitions = tuple(definitions)
    if not definitions:
        return pd.DataFrame(), pd.DataFrame()
    end_date = end or datetime.now(timezone.utc).date().isoformat()
    source = store or FredStore()

    def fetch(definition):
        result = source.get(definition.symbol, start, end_date, refresh=refresh, offline=offline, vintage=vintage)
        status = {**result.metadata, "key": definition.key, "provider": definition.provider}
        return definition.key, result.series, status

    with ThreadPoolExecutor(max_workers=min(3, len(definitions))) as pool:
        rows = list(pool.map(fetch, definitions))
    values = {key: series for key, series, _ in rows if not series.empty}
    return pd.DataFrame(values).sort_index(), pd.DataFrame([status for _, _, status in rows])


def fetch_fred_symbols(symbols: Iterable[str], *, start: str, end: str | None = None):
    definitions = tuple(SeriesDefinition(s, s, s, "FRED", "Macro", "Natural-unit provider observations") for s in dict.fromkeys(symbols))
    return fetch_fred_series(definitions, start=start, end=end)


def render_fred_status(status: pd.DataFrame, *, expanded: bool = False):
    """Every consuming page can disclose failures, staleness and provenance."""
    import streamlit as st

    if status.empty:
        return
    problems = status[status["status"].isin(["FAILED", "STALE", "EMPTY"])]
    if not problems.empty:
        st.warning("Some macro observations are unavailable or older than their expected publication window: " + ", ".join(problems["symbol"].astype(str)))
    recovered = status[status.get("delivery", pd.Series(index=status.index, dtype=str)).eq("last-good fallback")]
    if not recovered.empty:
        st.info("A provider refresh failed. The last validated observations are retained with their original dates.")
    with st.expander("Macro data freshness and sources", expanded=expanded):
        columns = [c for c in ("symbol", "status", "data_from", "data_through", "fetched_at", "history_years", "units", "frequency", "source", "delivery", "error") if c in status]
        st.dataframe(status[columns], width="stretch", hide_index=True)
        st.caption("Observation dates describe the measured period; download dates do not make old observations current. Historical macro values use the latest revisions unless a vintage is explicitly selected.")
