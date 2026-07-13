"""Multi-horizon z-score engine (spec §Normalization).

Every trajectory/stretch metric is z-scored against *its own* history on
calendar lookback windows, shown side by side:

  secular    (~15y) — "stretched vs. a full generation of cycles"
  structural (~10y) — "stretched vs. its whole cycle"
  regime     (~2y)  — "stretched vs. recent normal"

A persistent short-vs-long detachment is itself the regime-change tell. Windows are
calendar-based (not fixed observation counts) so the same code works for daily,
monthly, and quarterly series. Each currency is scored against its own history —
USD included; there is no cross-sectional demeaning here.
"""
from __future__ import annotations

import pandas as pd

STRUCT_YEARS = 10
REGIME_YEARS = 2
SECULAR_YEARS = 15
# ordered horizon registry — the compositor, engine, history, and app all key off
# this so adding a horizon is a one-line change here plus a history reseed
HORIZONS: list[tuple[str, str, int]] = [
    ("struct", "struct_z", STRUCT_YEARS),
    ("regime", "regime_z", REGIME_YEARS),
    ("secular", "secular_z", SECULAR_YEARS),
]
_DAYS = 365.25


def _roll_z(s: pd.Series, years: float, min_frac: float = 0.5) -> pd.Series:
    """Calendar-window rolling z-score of a date-indexed series. min_frac sets the
    minimum fraction of the window (in observations) required before emitting a z."""
    s = s.sort_index()
    win = f"{int(years * _DAYS)}D"
    # infer typical obs spacing to set a sane min_periods
    gap = s.index.to_series().diff().dt.days.median()
    if pd.isna(gap) or gap == 0:        # NaN- and zero-safe
        gap = 30
    min_obs = max(6, int((years * _DAYS / gap) * min_frac))
    roll = s.rolling(win, min_periods=min_obs)
    return (s - roll.mean()) / roll.std(ddof=0)


def dual_horizon_z(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Add one z column per registered horizon (struct_z / regime_z / secular_z) to
    a tidy [date, ccy, metric, value] frame, computed per (ccy, metric) group.
    (Name kept for API stability — it is now multi-horizon.)"""
    out = []
    for (ccy, metric), g in df.groupby(["ccy", "metric"], sort=False):
        g = g.sort_values("date").copy()
        s = g.set_index("date")[value_col]
        for _, zcol, years in HORIZONS:
            g[zcol] = _roll_z(s, years).values
        out.append(g)
    cols = ["date", "ccy", "metric", value_col] + [z for _, z, _ in HORIZONS]
    res = pd.concat(out, ignore_index=True)
    keep = [c for c in cols if c in res.columns] + \
           [c for c in res.columns if c not in cols]
    return res[keep].sort_values(["ccy", "metric", "date"]).reset_index(drop=True)


def latest_z(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """The current-snapshot multi-horizon z per (ccy, metric): the last
    observation's z on every registered horizon, plus its date and raw value."""
    z = dual_horizon_z(df, value_col=value_col)
    last = z.sort_values("date").groupby(["ccy", "metric"]).tail(1)
    return last.reset_index(drop=True)
