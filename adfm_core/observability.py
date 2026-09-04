"""Non-sensitive data-load observability shared by legacy and native runtimes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping

import pandas as pd

try:
    import streamlit as st
except ImportError:  # Native API runtime intentionally does not install Streamlit.
    st = None  # type: ignore[assignment]

SESSION_KEY = "adfm_data_health"
_fallback_data_health: DataLoadEvent | None = None


@dataclass(frozen=True)
class DataLoadEvent:
    """One provider request outcome safe to show in the application."""

    provider: str
    requested_symbols: int
    returned_symbols: int
    failed_symbols: int
    data_through: str | None
    recorded_at_utc: str


def record_data_load(
    provider: str,
    frames: Mapping[str, pd.DataFrame],
    requested_symbols: Iterable[str],
) -> DataLoadEvent:
    """Record the latest compact provider status for the active runtime."""
    global _fallback_data_health

    requested = tuple(dict.fromkeys(symbol for symbol in requested_symbols if symbol))
    dates = [frame.index.max() for frame in frames.values() if not frame.empty]
    event = DataLoadEvent(
        provider=provider,
        requested_symbols=len(requested),
        returned_symbols=len(frames),
        failed_symbols=max(0, len(requested) - len(frames)),
        data_through=pd.Timestamp(max(dates)).date().isoformat() if dates else None,
        recorded_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
    if st is not None:
        st.session_state[SESSION_KEY] = asdict(event)
    else:
        _fallback_data_health = event
    return event


def current_data_health() -> DataLoadEvent | None:
    """Return the latest data-load event for the active runtime."""
    if st is not None:
        raw = st.session_state.get(SESSION_KEY)
        return DataLoadEvent(**raw) if isinstance(raw, dict) else None
    return _fallback_data_health
