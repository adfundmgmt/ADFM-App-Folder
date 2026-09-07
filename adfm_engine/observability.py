"""Non-sensitive request diagnostics usable from APIs and scheduled jobs."""
import logging
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger("adfm.data")


def record_data_load(provider, frames, requested_symbols):
    dates = [frame.index.max() for frame in frames.values() if not frame.empty]
    event = {
        "provider": provider,
        "requested_symbols": len(set(requested_symbols)),
        "returned_symbols": len(frames),
        "data_through": pd.Timestamp(max(dates)).date().isoformat() if dates else None,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("market_data_load", extra={"data_load": event})
    return event
