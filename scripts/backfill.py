"""Refresh Currency Tension Engine source caches from primary providers.

This is the local ingestion entrypoint for ADFM's scheduled Currency Tension build.
It writes only to the repository's cache directory and has no dependency on another
GitHub repository.

Usage:
    python -m scripts.backfill          # cold-start/full-history refresh
    python -m scripts.backfill --daily  # scheduled incremental refresh

Required environment variables: FRED_API_KEY, ESTAT_APP_ID.
"""
from __future__ import annotations

import sys
from collections.abc import Callable

import pandas as pd

from cte.store import depth_audit, merge_cache


class IngestionFailure(RuntimeError):
    """Raised when one or more required source blocks fail to refresh."""


def _fetch(label: str, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame | None:
    try:
        frame = fn()
    except Exception as exc:
        print(f"  [FAIL] {label:16} {type(exc).__name__}: {str(exc)[:160]}")
        return None
    if frame is None or frame.empty:
        print(f"  [FAIL] {label:16} returned no rows")
        return None
    print(f"  [ok]   {label:16} rows={len(frame):>7}")
    return frame


def run(full_history: bool = True) -> None:
    mode = "cold-start (full history)" if full_history else "daily incremental"
    print(f"Refreshing CTE cache — {mode}...\n")
    failures: list[str] = []

    from cte.adapters.macro import build_macro_backbone

    macro = _fetch("macro_backbone", build_macro_backbone)
    if macro is None:
        failures.append("macro_backbone")
    else:
        merge_cache("macro_backbone", macro, keys=["ccy", "metric", "date"])

    from cte.adapters import bis_reer, cftc_tff, sovereign_yields, yahoo

    fx = _fetch("fx_spot", lambda: yahoo.fetch_fx_spot(period="10y"))
    if fx is None:
        failures.append("fx_spot")
    else:
        merge_cache("fx_spot", fx, keys=["ccy", "date"])

    yields = _fetch(
        "yields",
        lambda: sovereign_yields.fetch_all_yields(full_history=full_history),
    )
    if yields is None:
        failures.append("yields")
    else:
        merge_cache("yields", yields, keys=["ccy", "tenor", "date"])

    reer = _fetch("reer", bis_reer.fetch_reer)
    if reer is None:
        failures.append("reer")
    else:
        merge_cache("reer", reer, keys=["ccy", "date"])

    tff = _fetch("tff", lambda: cftc_tff.fetch_tff(weeks_back=850))
    if tff is None:
        failures.append("tff")
    else:
        merge_cache("tff", tff, keys=["ccy", "date"])

    if failures:
        raise IngestionFailure(
            "Required Currency Tension sources failed: " + ", ".join(failures)
        )

    print("\nDepth audit (macro backbone):")
    audit = depth_audit(macro, ["ccy", "metric"])
    short = audit[audit.struct_10y == "SHORT"]
    print(audit.to_string(index=False))
    print(f"\n  {len(audit)} series | {len(short)} short of the 10y structural window")

    from cte.adapters.base import read_cache

    for name, group_cols in (
        ("reer", ["ccy"]),
        ("tff", ["ccy"]),
        ("yields", ["ccy", "tenor"]),
    ):
        frame = read_cache(name)
        if frame is not None and "date" in frame.columns:
            source_audit = depth_audit(frame, group_cols)
            print(
                f"\nDepth audit ({name}): median span "
                f"{source_audit.years.median():.1f}y, "
                f"{(source_audit.struct_10y == 'SHORT').sum()}/{len(source_audit)} "
                "short of 10y"
            )


if __name__ == "__main__":
    run(full_history="--daily" not in sys.argv)
