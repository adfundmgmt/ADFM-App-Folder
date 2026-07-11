"""Persistent store — cold-start backfill + incremental merge for the CTE cache.

The design is a one-time full-history backfill that seeds the z-score baselines,
followed by daily incremental merges that append only new observations. Both write
to the same parquet cache (cte.adapters.base.CACHE_DIR); the z-score layer reads
the accumulated history from there rather than re-downloading it.

merge_cache() is idempotent: re-running an adapter and merging never duplicates
rows — it keeps the most recently fetched value per key.
"""
from __future__ import annotations

import pandas as pd

from cte.adapters.base import cache_path, read_cache, write_cache


def merge_cache(name: str, new_df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Merge new rows into the cached frame `name`, de-duplicating on `keys` and
    keeping the most recently fetched value. Returns the merged frame."""
    if new_df is None or new_df.empty:
        cached = read_cache(name)
        return cached if cached is not None else new_df
    existing = read_cache(name)
    combined = (pd.concat([existing, new_df], ignore_index=True)
                if existing is not None else new_df.copy())
    # If a (ccy, metric)'s source changed (e.g. a CPI cadence switch), drop the old
    # source's cached rows so no mixed-cadence orphans survive. Vectorized: anti-join
    # cached rows against the (ccy, metric, source) whitelist from the incoming pull,
    # but only for groups the incoming pull actually touched.
    if existing is not None and {"ccy", "metric", "source"}.issubset(new_df.columns):
        wl = new_df[["ccy", "metric", "source"]].drop_duplicates().assign(_keep=True)
        touched = set(map(tuple, new_df[["ccy", "metric"]].drop_duplicates().values))
        merged = combined.merge(wl, on=["ccy", "metric", "source"], how="left")
        in_touched = [tuple(x) in touched
                      for x in combined[["ccy", "metric"]].values]
        stale = pd.Series(in_touched, index=combined.index) & merged["_keep"].isna().values
        if stale.any():
            dropped = (combined[stale].groupby(["ccy", "metric", "source"]).size())
            print(f"[store] pruned {int(stale.sum())} stale-source rows in '{name}': "
                  f"{dict(dropped)}")
        combined = combined[~stale.values].copy()
    sort_col = "fetched_at" if "fetched_at" in combined.columns else None
    if sort_col:
        combined = combined.sort_values(sort_col)
    combined = (combined.drop_duplicates(subset=keys, keep="last")
                .sort_values(keys).reset_index(drop=True))
    write_cache(combined, name)
    return combined


def depth_audit(df: pd.DataFrame, group_cols: list[str],
                date_col: str = "date") -> pd.DataFrame:
    """Per-series history span, for judging dual-horizon z-score feasibility.
    Flags series shorter than the ~10y structural window and ~2y regime window."""
    rows = []
    for key, g in df.groupby(group_cols):
        gd = pd.to_datetime(g[date_col])
        span_yrs = (gd.max() - gd.min()).days / 365.25
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(group_cols, key)),
                     "start": gd.min().date(), "end": gd.max().date(),
                     "n": len(g), "years": round(span_yrs, 1),
                     "struct_10y": "ok" if span_yrs >= 10 else "SHORT",
                     "regime_2y": "ok" if span_yrs >= 2 else "SHORT"})
    return pd.DataFrame(rows).sort_values("years").reset_index(drop=True)
