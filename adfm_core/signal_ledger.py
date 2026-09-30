"""Durable, point-in-time history for PM command-center signals."""

from __future__ import annotations

import fcntl
import os
import tempfile
from contextlib import contextmanager
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



def _ledger_path(path: Path | None) -> Path:
    return Path(path) if path is not None else Path(os.getenv("ADFM_SIGNAL_LEDGER_PATH", str(DEFAULT_LEDGER_PATH)))


@contextmanager
def _write_lock(path: Path):
    # flock coordinates both page threads and separate scheduled processes on Linux.
    with path.with_suffix(path.suffix + ".lock").open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_signal_history(path: Path | None = None) -> pd.DataFrame:
    """Load prior snapshots, returning an empty stable schema when unavailable."""

    path = _ledger_path(path)
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
    path: Path | None = None,
    *,
    captured_at: datetime | None = None,
) -> pd.DataFrame:
    """Append immutable versions under a process lock; unchanged reruns are no-ops.

    Set ADFM_SIGNAL_LEDGER_PATH to a persistent mounted location in deployment.
    Explicit captured_at identifies a scheduled capture; implicit page reruns compare
    each key with its latest version, retaining the original capture timestamp.
    """
    path = _ledger_path(path)
    required = list(LEDGER_COLUMNS[1:])
    missing = set(required).difference(snapshot.columns)
    if missing and not snapshot.empty:
        raise ValueError(f"Signal snapshot is missing required columns: {sorted(missing)}")
    captured = captured_at or datetime.now(timezone.utc)
    if captured.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    captured = captured.astimezone(timezone.utc)
    current = snapshot.reindex(columns=required).copy()
    current["Captured At UTC"] = captured.isoformat()
    current = current[list(LEDGER_COLUMNS)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock(path):
        if path.exists():
            try:
                history = pd.read_parquet(path)
            except Exception as exc:
                raise ValueError("Signal ledger cannot be read; original file retained") from exc
            if set(LEDGER_COLUMNS).difference(history.columns):
                raise ValueError("Signal ledger schema is invalid; original file retained")
            history = history[list(LEDGER_COLUMNS)]
        else:
            history = pd.DataFrame(columns=LEDGER_COLUMNS)
        pending = []
        for _, row in current.iterrows():
            prior = history[history["Key"].astype(str).eq(str(row["Key"]))]
            columns = list(LEDGER_COLUMNS) if captured_at is not None else required
            candidates = prior if captured_at is not None else prior.sort_values("Captured At UTC").tail(1)
            matches = candidates[columns].fillna("<missing>").astype(str).eq(row[columns].fillna("<missing>").astype(str), axis=1).all(axis=1)
            if not matches.any():
                pending.append(row.to_dict())
        if not pending:
            return history.reset_index(drop=True)
        combined = pd.concat([history, pd.DataFrame(pending)], ignore_index=True)
        combined = combined.sort_values(["Data Through", "Key", "Captured At UTC"], kind="stable")
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}-", suffix=".parquet", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            combined.to_parquet(temp_path, index=False)
            with temp_path.open("rb") as handle:
                os.fsync(handle.fileno())
            temp_path.replace(path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temp_path.unlink(missing_ok=True)
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
