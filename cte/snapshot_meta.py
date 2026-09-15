"""Freshness metadata for the Currency Tension Engine snapshot."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from cte.config import CACHE_DIR


def snapshot_generated_at(cache_dir: Path = CACHE_DIR) -> Optional[pd.Timestamp]:
    """Return the validated snapshot timestamp, falling back to snapshot history."""
    manifest_path = cache_dir / "snapshot_manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw = payload.get("validated_at_utc")
            if raw:
                stamp = pd.Timestamp(raw)
                if stamp.tzinfo is None:
                    stamp = stamp.tz_localize("UTC")
                return stamp
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    history_path = cache_dir / "snapshot_history.parquet"
    if history_path.exists():
        try:
            history = pd.read_parquet(history_path)
            if not history.empty and "date" in history.columns:
                dates = pd.to_datetime(history["date"], errors="coerce", utc=True).dropna()
                if not dates.empty:
                    return dates.max()
        except Exception:
            pass

    return None
