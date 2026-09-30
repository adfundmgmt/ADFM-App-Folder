"""Scheduled, public-data-only commodity signal capture (never account uploads)."""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from adfm_core.public_signal_capture import capture_public_signals, write_capture_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=Path("data/signals/captures"))
    args = parser.parse_args()
    captured_at = datetime.now(timezone.utc)
    snapshot, failures = capture_public_signals(captured_at=captured_at)
    if not snapshot.empty:
        snapshot.insert(0, "Captured At UTC", captured_at.isoformat())
        digest = hashlib.sha256(snapshot.to_json(orient="records").encode()).hexdigest()[:16]
        path = args.audit_dir / f"{captured_at:%Y%m%dT%H%M%S%fZ}-{digest}.json"
        write_capture_audit(snapshot, path)
        print(f"Captured {len(snapshot)} public signal versions; audit {path.name}")
    for symbol, error in failures.items():
        print(f"{symbol}: {error}")
    return 1 if failures or snapshot.empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
