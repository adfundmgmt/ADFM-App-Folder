"""Seed the tension-map history (one-time monthly backfill).

Usage:
  python -m scripts.backfill_history               # always (re)build
  python -m scripts.backfill_history --if-missing  # no-op when the committed
                                                   # snapshot_history.parquet exists
                                                   # (the daily engine appends to it)
Run AFTER scripts.backfill (needs the raw feature cache).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cte.adapters.base import CACHE_DIR                      # noqa: E402
from cte.scoring.history import (CARRY_HISTORY_NAME, HISTORY_NAME,   # noqa: E402
                                 OVERLAY_HISTORY_NAME, PILLAR_HISTORY_NAME,
                                 backfill)

if __name__ == "__main__":
    targets = [CACHE_DIR / f"{n}.parquet"
               for n in (HISTORY_NAME, PILLAR_HISTORY_NAME, CARRY_HISTORY_NAME,
                         OVERLAY_HISTORY_NAME)]
    def _schema_current() -> bool:
        try:
            import pandas as pd
            sh = pd.read_parquet(CACHE_DIR / f"{HISTORY_NAME}.parquet")
            return "axis1_fundamental_secular" in sh.columns
        except Exception:
            return False

    if "--if-missing" in sys.argv and all(t.exists() for t in targets) \
            and _schema_current():
        print("history files present and schema current — skipping; "
              "the engine appends daily.")
        sys.exit(0)
    h = backfill()
    print(f"backfilled {h.date.nunique()} month-ends, "
          f"{h.ccy.nunique()} currencies, {len(h)} rows "
          f"({h.date.min().date()} -> {h.date.max().date()})")
