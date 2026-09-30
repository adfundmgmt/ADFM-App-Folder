"""Public commodity computations for immutable scheduled and page signal capture."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .signal_ledger import LEDGER_COLUMNS, load_signal_history, record_signal_snapshot

PUBLIC_SYMBOLS = ("CL=F", "NG=F", "GC=F", "SI=F", "HG=F", "ZC=F", "ZS=F")


def commodity_snapshot(symbol, profile, settings, return_days, diagnostics, condition) -> pd.DataFrame:
    """Stable key includes the entire signal definition, keeping presets distinct."""
    if diagnostics.empty or condition.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS[1:])
    definition = json.dumps({"profile": profile, "settings": settings, "return_days": return_days}, sort_keys=True)
    digest = hashlib.sha256(definition.encode()).hexdigest()[:12]
    latest = diagnostics.index.max()
    signals = condition.reindex(diagnostics.index).fillna(False).astype(float)
    return pd.DataFrame([{
        "Data Through": pd.Timestamp(latest).date().isoformat(), "Signal": f"{symbol} · {profile}",
        "Key": f"commodity:{symbol}:{digest}", "Group": "Commodity price signal",
        "Composite": float(signals.iloc[-1]),
        "Impulse": float(signals.iloc[-1] - signals.iloc[-2]) if len(signals) > 1 else np.nan,
        "Confidence": 1.0 if len(diagnostics) >= 260 else np.nan,
    }])


def capture_public_signals(symbols=PUBLIC_SYMBOLS, *, path: Path | None = None, captured_at=None, history_loader=None, compute=None) -> tuple[pd.DataFrame, dict[str, str]]:
    """Capture actual fresh price signals; missing providers never fabricate rows."""
    from .commodity_top_exhaustion_page import (
        PROFILE_PRESETS,
        build_exhaustion_frame,
        load_contract_history,
    )
    history_loader = history_loader or load_contract_history
    compute = compute or build_exhaustion_frame
    captured = captured_at or datetime.now(timezone.utc)
    now = pd.Timestamp(captured).tz_convert("America/New_York").tz_localize(None).normalize()
    snapshots, failures = [], {}
    for symbol in symbols:
        try:
            history = history_loader(symbol)
            if history.empty:
                raise ValueError("No public observations")
            latest = pd.Timestamp(history.index.max()).tz_localize(None).normalize()
            if latest > now or len(pd.bdate_range(latest, now)) - 1 > 5:
                raise ValueError("Observed public prices are future-dated or stale")
            settings = PROFILE_PRESETS["Confirmed Exhaustion"]
            diagnostics, condition, _, _ = compute(history, symbol, "Confirmed Exhaustion", settings, 21)
            row = commodity_snapshot(symbol, "Confirmed Exhaustion", settings, 21, diagnostics, condition)
            if row.empty:
                raise ValueError("Signal computation unavailable")
            snapshots.append(row)
        except Exception as exc:
            failures[symbol] = f"Capture unavailable ({type(exc).__name__}): {exc}"
    result = pd.concat(snapshots, ignore_index=True) if snapshots else pd.DataFrame(columns=LEDGER_COLUMNS[1:])
    if not result.empty:
        record_signal_snapshot(result, path, captured_at=captured)
    return result, failures


def write_capture_audit(snapshot: pd.DataFrame, path: Path) -> None:
    """Exclusive creation of a public audit copy; conflicts cannot overwrite it."""
    payload = snapshot.to_json(orient="records", date_format="iso", indent=2).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError("Capture audit already exists with different content") from None


def load_captured_commodity_history(*, shipped_path: Path | None = None, local_path: Path | None = None) -> pd.DataFrame:
    """Display scheduled audit history together with local immutable captures."""
    shipped_path = shipped_path or Path(__file__).resolve().parents[1] / "data/signals/public_signal_ledger.parquet"
    frames = [load_signal_history(shipped_path), load_signal_history(local_path)]
    result = pd.concat(frames, ignore_index=True).drop_duplicates(list(LEDGER_COLUMNS))
    return result.sort_values(["Data Through", "Key", "Captured At UTC"], kind="stable").reset_index(drop=True)
