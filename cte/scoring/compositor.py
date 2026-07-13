"""Compositor — roll signed dual-horizon z-scores up into pillar, axis, and the
two-axis tension map (spec §4-5).

Flow: feature z-scores -> apply sign -> weighted mean within pillar -> weighted mean
of pillars within axis. Produces, per currency, a structural and regime score on:
  axis1_fundamental  (deteriorating <-> improving)
  axis2_stretch      (cheap <-> maxed-out)

Signs/weights come from config (FEATURE_SIGN, FEATURE_WEIGHT, PILLAR_WEIGHT) so they
are tunable; the carry grid (pairwise) is produced separately in transform.pairwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cte.config import (FEATURE_PILLAR, FEATURE_SIGN, FEATURE_WEIGHT,
                        PILLAR_AXIS, PILLAR_WEIGHT)
from cte.flags.overlays import overlay_snapshot, warnings
from cte.transform.features import build_features
from cte.transform.zscore import latest_z


def _wmean(vals: pd.Series, weights: pd.Series) -> float:
    m = vals.notna()
    if not m.any():
        return np.nan
    return np.average(vals[m], weights=weights[m])


def _apply_overlays(d: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """Bend the two conditional signals by the objective overlay multipliers:
      real_10y  scaled by the yield-reward/stress multiplier (tanh of FX-yield corr);
      inflation scaled by hike-feasibility, but ONLY where inflation is above target
                (the currency's infl_gap > 0) — below target the trap logic doesn't
                apply. Both inflation rows (gap and momentum) share that one gate.
    Shrinking toward 0 (decoupled / at inflection) mutes the signal; going negative
    (punished / trapped) flips tailwind to headwind."""
    r10 = snap.set_index("ccy")["real10y_mult"]
    imult = snap.set_index("ccy")["infl_mult"]
    # the trap gate is the currency's inflation GAP (above/below target), looked up
    # once per currency — not each row's own value (infl_momentum's value is a change).
    gap = (d[d.metric == "infl_gap"].set_index("ccy")["value"])
    d = d.copy()
    for i, row in d.iterrows():
        c, met = row["ccy"], row["metric"]
        if met == "real_10y" and pd.notna(r10.get(c, np.nan)):
            d.at[i, "signed"] *= r10.get(c)
        elif met in ("infl_gap", "infl_momentum") and gap.get(c, 0) > 0 \
                and pd.notna(imult.get(c, np.nan)):
            d.at[i, "signed"] *= imult.get(c)
    return d


def score(zcol: str, latest: pd.DataFrame,
          snap: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Composite one horizon (zcol) into pillar and axis scores per currency, after
    applying the objective inflection multipliers. Returns (pillar, axis) frames."""
    d = latest.copy()
    d["pillar"] = d["metric"].map(FEATURE_PILLAR)
    d["axis"] = d["pillar"].map(PILLAR_AXIS)
    d["sign"] = d["metric"].map(FEATURE_SIGN).fillna(1)
    d["w"] = d["metric"].map(FEATURE_WEIGHT).fillna(1.0)
    d["signed"] = d[zcol] * d["sign"]
    if snap is not None:
        d = _apply_overlays(d, snap)
    d = d.dropna(subset=["pillar", "signed"])

    pill = (d.groupby(["ccy", "pillar"])
            .apply(lambda g: _wmean(g["signed"], g["w"]), include_groups=False)
            .rename("pscore").reset_index())
    pill["axis"] = pill["pillar"].map(PILLAR_AXIS)
    pill["pw"] = pill["pillar"].map(PILLAR_WEIGHT).fillna(1.0)

    axis = (pill.groupby(["ccy", "axis"])
            .apply(lambda g: _wmean(g["pscore"], g["pw"]), include_groups=False)
            .rename("ascore").reset_index())
    return (pill, axis)


def axes_from_pillars(pill: pd.DataFrame, value_col: str,
                      weights: dict | None = None,
                      keys: tuple = ("ccy",)) -> pd.DataFrame:
    """Recompose axis scores from persisted pillar scores under (optionally
    custom) pillar weights — the exact aggregation score() applies, factored out
    so the app can re-weight client-side without recomputing pillars or overlays.
    A weight of 0 excludes the pillar. Returns a wide frame keyed by `keys` with
    columns axis1_fundamental_<value_col> / axis2_stretch_<value_col>."""
    w = dict(PILLAR_WEIGHT)
    w.update(weights or {})
    d = pill.dropna(subset=[value_col]).copy()
    d["axis"] = d["pillar"].map(PILLAR_AXIS)
    d = d.dropna(subset=["axis"])
    d["pw"] = d["pillar"].map(w).fillna(1.0)
    d = d[d["pw"] > 0]
    if d.empty:
        return pd.DataFrame(columns=[*keys])
    g = (d.groupby([*keys, "axis"])
         .apply(lambda x: _wmean(x[value_col], x["pw"]), include_groups=False)
         .rename("ascore").reset_index())
    wide = g.pivot_table(index=list(keys), columns="axis",
                         values="ascore").reset_index()
    wide.columns.name = None
    return wide.rename(columns={a: f"{a}_{value_col}"
                                for a in ("axis1_fundamental", "axis2_stretch")})


def tension_map() -> tuple[pd.DataFrame, dict]:
    """Current snapshot: each currency's structural & regime score on both axes, plus
    the per-currency inflection warnings that explain any bent contributions."""
    latest = build_features().rename(columns={"feature": "metric"})
    lz = latest_z(latest)
    snap = overlay_snapshot()
    rows = {}
    from cte.transform.zscore import HORIZONS
    for horizon, zcol, _ in HORIZONS:
        _, axis = score(zcol, lz, snap)
        for _, r in axis.iterrows():
            rows.setdefault(r["ccy"], {})[f"{r['axis']}_{horizon}"] = r["ascore"]
    tm = pd.DataFrame(rows).T
    tm.index.name = "ccy"
    return tm.reset_index(), warnings(snap, lz)


if __name__ == "__main__":
    latest = build_features().rename(columns={"feature": "metric"})
    lz = latest_z(latest)
    snap = overlay_snapshot()
    pill, axis = score("struct_z", lz, snap)
    print("=== Pillar scores (structural, signed, overlay-adjusted) ===")
    print(pill.pivot_table(index="ccy", columns="pillar", values="pscore")
          .round(2).to_string())
    tm, warns = tension_map()
    print("\n=== Tension map (axis scores, both horizons) ===")
    cols = ["ccy", "axis1_fundamental_struct", "axis2_stretch_struct",
            "axis1_fundamental_regime", "axis2_stretch_regime"]
    print(tm[[c for c in cols if c in tm.columns]].round(2).to_string(index=False))
    print(f"\n{sum(len(v) for v in warns.values())} inflection warnings across "
          f"{len(warns)} currencies (see cte.flags.overlays.warnings).")
