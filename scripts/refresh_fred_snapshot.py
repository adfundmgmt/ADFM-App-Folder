"""Refresh redistributable macro series; retain last-good files on any failure."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from adfm_core.fred_registry import SNAPSHOT_SYMBOLS
from adfm_core.fred_store import (
    ROOT,
    FredStore,
    api_key,
    path_for,
    read_record,
    utcnow,
    write_record,
)


def refresh_snapshot(output: Path, symbols=SNAPSHOT_SYMBOLS) -> dict:
    results = []
    with tempfile.TemporaryDirectory(prefix="adfm-fred-") as temporary:
        store = FredStore(Path(temporary), output)
        key_present = bool(api_key())
        for symbol in symbols:
            result = store.get(symbol, "1900-01-01", utcnow().date().isoformat(), refresh=True)
            status = result.metadata.copy()
            if result.metadata.get("delivery") == "downloaded":
                record = read_record(path_for(Path(temporary), symbol), symbol)
                if record is not None:
                    write_record(path_for(output, symbol), *record)
            results.append(status)
            print(f"{symbol}: {status['status']} through {status.get('data_through')}", flush=True)
    report = {"checked_at": utcnow().isoformat(), "api_configured": key_present, "series": results}
    output.mkdir(parents=True, exist_ok=True)
    (output / "refresh_status.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/fred")
    args = parser.parse_args()
    report = refresh_snapshot(args.output)
    failures = [r for r in report["series"] if r.get("error") or r["status"] in {"FAILED", "STALE", "EMPTY"}]
    if failures:
        print(f"{len(failures)} series require attention; retained previous validated data.")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
