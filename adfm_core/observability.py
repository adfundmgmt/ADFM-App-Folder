"""Session-scoped, non-sensitive data-load observability for Streamlit pages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping

import pandas as pd
import streamlit as st

SESSION_KEY = "adfm_data_health"


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
    """Record a compact latest-provider status in the current Streamlit session."""
    requested = tuple(dict.fromkeys(symbol for symbol in requested_symbols if symbol))
    dates = []
    for frame in frames.values():
        observed = frame.dropna(subset=["Close"]) if "Close" in frame else frame.dropna(how="all")
        if not observed.empty:
            dates.append(observed.index.max())
    event = DataLoadEvent(
        provider=provider,
        requested_symbols=len(requested),
        returned_symbols=len(frames),
        failed_symbols=max(0, len(requested) - len(frames)),
        data_through=pd.Timestamp(max(dates)).date().isoformat() if dates else None,
        recorded_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
    st.session_state[SESSION_KEY] = asdict(event)
    return event


def current_data_health() -> DataLoadEvent | None:
    """Return the latest data-load event recorded in this Streamlit session."""
    raw = st.session_state.get(SESSION_KEY)
    return DataLoadEvent(**raw) if isinstance(raw, dict) else None


PERFORMANCE_KEY = "adfm_performance_events"
MAX_PERFORMANCE_EVENTS = 100


def record_performance(
    operation: str,
    elapsed_seconds: float,
    *,
    cache_hit: bool = False,
    requested_count: int = 0,
    failed_count: int = 0,
) -> dict:
    """Record bounded aggregate timings; never include symbols or account data.

    Operation is restricted to public infrastructure labels. Memory is process
    peak RSS (not uploaded file size or holdings), in MiB on Linux deployments.
    """
    import resource

    label = operation if operation in {"provider", "delivery", "page"} else "page"
    event = {
        "operation": label,
        "elapsed_seconds": round(max(0.0, elapsed_seconds), 6),
        "cache_hit": bool(cache_hit),
        "requested_count": max(0, int(requested_count)),
        "failed_count": max(0, int(failed_count)),
        "peak_memory_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
    }
    events = list(st.session_state.get(PERFORMANCE_KEY, []))
    st.session_state[PERFORMANCE_KEY] = (events + [event])[-MAX_PERFORMANCE_EVENTS:]
    return event


def performance_events() -> list[dict]:
    """Return only aggregate performance records in the current session."""
    return list(st.session_state.get(PERFORMANCE_KEY, []))


def render_performance_diagnostics() -> None:
    """Keep developer telemetry out of the primary analytical table."""
    with st.expander("Data delivery diagnostics", expanded=False, on_change="rerun") as details:
        if details is None or not details.open:
            return
        events = performance_events()
        if events:
            st.dataframe(pd.DataFrame(events), hide_index=True, use_container_width=True)
        else:
            st.caption("No provider deliveries recorded in this session.")


class page_timer:
    """Context manager for bounded page/calculation timing without page names."""

    def __enter__(self):
        import time
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        import time
        record_performance("page", time.perf_counter() - self.started)
