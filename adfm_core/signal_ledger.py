"""Durable, point-in-time history for PM command-center signals."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_LEDGER_PATH = Path("data/last_good/pm_signal_ledger.parquet")
LEDGER_COLUMNS = (
    "Captured At UTC",
    "Data Through",
    "Signal",
    "Key",
    "Group",
    "Composite",
    "Impulse",
    "Confidence",
)


def load_signal_history(path: Path = DEFAULT_LEDGER_PATH) -> pd.DataFrame:
    """Load prior snapshots, returning an empty stable schema when unavailable."""

    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    for column in LEDGER_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[list(LEDGER_COLUMNS)].copy()


def record_signal_snapshot(
    snapshot: pd.DataFrame,
    path: Path = DEFAULT_LEDGER_PATH,
    *,
    captured_at: datetime | None = None,
) -> pd.DataFrame:
    """Upsert one point-in-time snapshot and write it atomically."""

    if snapshot.empty:
        return load_signal_history(path)
    required = {
        "Data Through",
        "Signal",
        "Key",
        "Group",
        "Composite",
        "Impulse",
        "Confidence",
    }
    missing = required.difference(snapshot.columns)
    if missing:
        raise ValueError(
            f"Signal snapshot is missing required columns: {sorted(missing)}"
        )

    captured = captured_at or datetime.now(timezone.utc)
    current = snapshot[list(required)].copy()
    current["Captured At UTC"] = captured.replace(microsecond=0).isoformat()
    current = current[list(LEDGER_COLUMNS)]

    history = load_signal_history(path)
    dates = set(current["Data Through"].dropna().astype(str))
    keys = set(current["Key"].dropna().astype(str))
    if not history.empty and dates and keys:
        keep = ~(
            history["Data Through"].astype(str).isin(dates)
            & history["Key"].astype(str).isin(keys)
        )
        history = history.loc[keep]
    combined = pd.concat([history, current], ignore_index=True)
    combined = combined.sort_values(["Data Through", "Key", "Captured At UTC"])

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}-", suffix=".parquet", delete=False
    ) as handle:
        temp_path = Path(handle.name)
    try:
        combined.to_parquet(temp_path, index=False)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return combined.reset_index(drop=True)


def latest_score_changes(history: pd.DataFrame) -> pd.DataFrame:
    """Compare the two latest available dates for each signal."""

    if history.empty:
        return pd.DataFrame(columns=["Key", "Previous Composite", "Change Since Prior"])
    frame = history.copy()
    frame["Data Through"] = pd.to_datetime(frame["Data Through"], errors="coerce")
    frame["Composite"] = pd.to_numeric(frame["Composite"], errors="coerce")
    frame = frame.dropna(subset=["Data Through", "Composite", "Key"])
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby("Key"):
        daily = (
            group.sort_values(["Data Through", "Captured At UTC"])
            .drop_duplicates("Data Through", keep="last")
            .tail(2)
        )
        if len(daily) < 2:
            continue
        rows.append(
            {
                "Key": key,
                "Previous Composite": float(daily["Composite"].iloc[-2]),
                "Change Since Prior": float(
                    daily["Composite"].iloc[-1] - daily["Composite"].iloc[-2]
                ),
            }
        )
    return pd.DataFrame(rows)
