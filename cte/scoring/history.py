"""Snapshot history — the tension map through time (trails, time dial, backtests).

Two entry points:

  backfill()      one-time: recompute the map as-of every month-end back through the
                  data (~2007+), using AS-OF overlay multipliers, not today's — the
                  yield-reward correlation and hike-feasibility are themselves rolling
                  series, so each historical point is bent by what the overlays said
                  THEN. Writes/overwrites data/cache/snapshot_history.parquet.

  append_today()  daily: called by the engine after build_snapshot; appends today's
                  live axis scores (idempotent per calendar date).

Honesty note (docs/DATA_SOURCES.md carries the long form): the backfill is computed
from TODAY'S vintage of the data — revised macro, and CPI aligned to its reference
month. Historical positions are "as we'd compute them now", not "as you'd have seen
them then". Fine for trails and the dial; a point-in-time backtest must additionally
lag inflation-derived features by one month and treat revised macro with suspicion.

Grain: month-end rows from the backfill (kind='month_end'), daily rows from the
engine (kind='daily'). Trails resample to month-end; the dial exposes every row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cte.adapters.base import read_cache, utcnow, write_cache
from cte.config import CURRENCIES
from cte.flags.overlays import _fx_wide, _y10_wide, _WIN
from cte.scoring.compositor import score
from cte.transform.features import build_features
from cte.transform.zscore import dual_horizon_z

HISTORY_NAME = "snapshot_history"
PILLAR_HISTORY_NAME = "pillar_history"
CARRY_HISTORY_NAME = "carry_history"
OVERLAY_HISTORY_NAME = "overlay_history"
_STALE_LIMIT = 12          # month-ends a feature may be carried forward as-of
from cte.transform.zscore import HORIZONS

_AXIS_COLS = [f"{a}_{h}" for h, _, _ in HORIZONS
              for a in ("axis1_fundamental", "axis2_stretch")]


# ------------------------------------------------------------------ as-of panels

def _monthly_panel(z: pd.DataFrame) -> tuple[pd.DatetimeIndex, dict]:
    """Reindex each (ccy, metric) z-series onto a common month-end grid with a
    bounded forward-fill, so 'as-of date D' is one slice, not N searches."""
    grid = pd.date_range(z.date.min() + pd.offsets.MonthEnd(0),
                         z.date.max() + pd.offsets.MonthEnd(0), freq="ME")
    panels = {}
    for (ccy, metric), g in z.groupby(["ccy", "metric"]):
        g = g.sort_values("date").set_index("date")
        g = g[~g.index.duplicated(keep="last")]
        g.index = g.index + pd.offsets.MonthEnd(0)
        g = g[~g.index.duplicated(keep="last")]
        zc = [z for _, z, _ in HORIZONS]
        panels[(ccy, metric)] = (g[["value"] + zc]
                                 .reindex(grid).ffill(limit=_STALE_LIMIT))
    return grid, panels


def _overlay_history(z_grid: pd.DatetimeIndex,
                     panels: dict) -> dict[pd.Timestamp, pd.DataFrame]:
    """As-of overlay multipliers per month-end: real10y_mult from the rolling
    FX/yield correlation series (month-end sampled), infl_mult from as-of
    growth/policy z (same construction as overlays.hike_feasibility, applied
    through time)."""
    # real10y_mult: full corr series per ccy, month-end sampled
    fx, y10 = _fx_wide(), _y10_wide()
    r10 = {}
    for c in CURRENCIES:
        if c not in fx or c not in y10:
            continue
        j = pd.concat({"fx": fx[c], "y": y10[c]}, axis=1, join="inner").dropna()
        corr = j["fx"].pct_change().rolling(_WIN).corr(j["y"].diff())
        corr.index = corr.index + pd.offsets.MonthEnd(0)
        corr = corr[~corr.index.duplicated(keep="last")]
        r10[c] = np.tanh(2.0 * corr.reindex(z_grid).ffill(limit=2))

    # keep the raw month-end corr too — the historical Overlays tab shows it
    corr_me = {c: (np.arctanh(np.clip(v, -0.999999, 0.999999)) / 2.0)
               for c, v in r10.items()}

    # infl_mult: growth composite minus real-policy restrictiveness, as-of
    g_feats = {"bcicp_slope": 1, "gdp_yoy": 1, "unemp_3m_chg": -1}
    out = {}
    for d in z_grid:
        rows = []
        for c in CURRENCIES:
            gz = [panels[(c, m)].loc[d, "struct_z"] * s
                  for m, s in g_feats.items() if (c, m) in panels]
            gz = [v for v in gz if pd.notna(v)]
            growth = float(np.mean(gz)) if gz else np.nan
            rp = (panels[(c, "real_policy")].loc[d, "struct_z"]
                  if (c, "real_policy") in panels else np.nan)
            feas = (0 if pd.isna(growth) else growth) - \
                   (0 if pd.isna(rp) else rp)
            imult = np.tanh(feas) if (pd.notna(growth) or pd.notna(rp)) else np.nan
            rmult = float(r10[c].loc[d]) if c in r10 and pd.notna(r10[c].loc[d]) \
                    else np.nan
            corr = float(corr_me[c].loc[d]) if c in corr_me and \
                pd.notna(corr_me[c].loc[d]) else np.nan
            label = (np.nan if pd.isna(corr) else
                     "rewarded" if corr > 0.15 else
                     "PUNISHED (stress)" if corr < -0.15 else "decoupled")
            rows.append({"ccy": c, "yld_fx_corr": round(corr, 2) if pd.notna(corr)
                         else np.nan, "yld_regime": label,
                         "real10y_mult": round(rmult, 2) if pd.notna(rmult)
                         else np.nan,
                         "growth_z": round(growth, 2) if pd.notna(growth)
                         else np.nan,
                         "real_policy_z": round(rp, 2) if pd.notna(rp) else np.nan,
                         "feasibility": round(feas, 2),
                         "infl_mult": round(imult, 2) if pd.notna(imult)
                         else np.nan})
        out[d] = pd.DataFrame(rows)
    return out


def _asof_frame(panels: dict, d: pd.Timestamp) -> pd.DataFrame:
    zcols = [z for _, z, _ in HORIZONS]
    rows = []
    for (ccy, metric), p in panels.items():
        r = p.loc[d]
        if r[["value"] + zcols].isna().all():
            continue
        rows.append({"ccy": ccy, "metric": metric, "value": r["value"],
                     **{z: r[z] for z in zcols}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ entry points

def backfill(persist: bool = True) -> pd.DataFrame:
    """Recompute the tension map as-of every month-end. Minutes, not hours; run
    once (the Action seeds it when the committed file is absent)."""
    raw_feats = build_features()
    feats = raw_feats.rename(columns={"feature": "metric"})
    z = dual_horizon_z(feats)
    grid, panels = _monthly_panel(z)
    overlays = _overlay_history(grid, panels)

    out, pill_out, carry_out, ovl_out = [], [], [], []
    for d in grid:
        asof = _asof_frame(panels, d)
        if asof.empty:
            continue
        snap = overlays[d]
        row, pill_acc = {}, {}
        for horizon, zcol, _ in HORIZONS:
            # early dates have no z on this horizon yet (window not filled) —
            # score() can't composite an empty frame, so skip the horizon
            if not asof[zcol].notna().any():
                continue
            pill, axis = score(zcol, asof, snap)
            for _, r in axis.iterrows():
                row.setdefault(r["ccy"], {})[f"{r['axis']}_{horizon}"] = r["ascore"]
            for _, r in pill.iterrows():
                pill_acc.setdefault((r["ccy"], r["pillar"]), {})[horizon] = r["pscore"]
        for ccy, vals in row.items():
            if any(pd.notna(v) for v in vals.values()):
                out.append({"date": d, "ccy": ccy, "kind": "month_end", **vals})
        for (ccy, pillar), h in pill_acc.items():
            pill_out.append({"date": d, "ccy": ccy, "pillar": pillar,
                             "kind": "month_end",
                             "struct": h.get("struct"), "regime": h.get("regime"),
                             "secular": h.get("secular")})
        ov = snap.copy()
        ov["date"], ov["kind"] = d, "month_end"
        ovl_out.append(ov)
        cv = asof[asof.metric.isin(["real_2y", "nominal_2y"])]
        for ccy, g in cv.groupby("ccy"):
            vals2 = g.set_index("metric")["value"]
            carry_out.append({"date": d, "ccy": ccy, "kind": "month_end",
                              "real_2y": vals2.get("real_2y"),
                              "nominal_2y": vals2.get("nominal_2y")})

    hist = pd.DataFrame(out)
    for c in _AXIS_COLS:
        if c not in hist.columns:
            hist[c] = np.nan
    hist = hist[["date", "ccy", "kind"] + _AXIS_COLS].sort_values(["date", "ccy"])
    if persist:
        write_cache(hist.reset_index(drop=True), HISTORY_NAME)
        def _frame(rows, cols, sort):
            d = pd.DataFrame(rows, columns=None if rows else cols)
            return d.sort_values(sort).reset_index(drop=True)
        write_cache(_frame(pill_out,
                           ["date", "ccy", "pillar", "kind", "struct", "regime", "secular"],
                           ["date", "ccy", "pillar"]), PILLAR_HISTORY_NAME)
        write_cache(_frame(carry_out,
                           ["date", "ccy", "kind", "real_2y", "nominal_2y"],
                           ["date", "ccy"]), CARRY_HISTORY_NAME)
        from cte.flags.overlays import carry_to_vol_history
        ovl = pd.concat(ovl_out, ignore_index=True)
        ctv = carry_to_vol_history(raw_feats)
        if len(ctv):
            ctv = ctv.copy()
            ctv["date"] = ctv["date"] + pd.offsets.MonthEnd(0)
            ovl = ovl.merge(ctv, on=["date", "ccy"], how="left")
        else:
            ovl["carry_to_vol"], ovl["ctv_pctile"] = np.nan, np.nan
        write_cache(ovl.sort_values(["date", "ccy"]).reset_index(drop=True),
                    OVERLAY_HISTORY_NAME)
    return hist


def append_today(tm: pd.DataFrame, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Append the live snapshot's axis scores under today's date. Idempotent:
    re-running on the same date replaces that date's daily rows."""
    asof = (asof or utcnow()).normalize()
    hist = read_cache(HISTORY_NAME)
    add = tm.copy()
    add["date"], add["kind"] = asof, "daily"
    for c in _AXIS_COLS:
        if c not in add.columns:
            add[c] = np.nan
    add = add[["date", "ccy", "kind"] + _AXIS_COLS]
    if hist is None:
        out = add
    else:
        keep = hist[~((hist.date == asof) & (hist.kind == "daily"))]
        out = pd.concat([keep, add], ignore_index=True)
    out = out.sort_values(["date", "ccy"]).reset_index(drop=True)
    write_cache(out, HISTORY_NAME)
    return out


def _append_rows(name: str, add: pd.DataFrame, asof: pd.Timestamp,
                 sort_cols: list[str]) -> None:
    """Idempotent daily append shared by the pillar/carry histories."""
    hist = read_cache(name)
    if hist is None:
        out = add
    else:
        keep = hist[~((hist.date == asof) & (hist.kind == "daily"))]
        out = pd.concat([keep, add], ignore_index=True)
    write_cache(out.sort_values(sort_cols).reset_index(drop=True), name)


_OVL_COLS = ["ccy", "yld_fx_corr", "yld_regime", "real10y_mult", "growth_z",
             "real_policy_z", "feasibility", "infl_mult", "carry_to_vol",
             "ctv_pctile"]


def append_today_details(pill_struct: pd.DataFrame, pill_regime: pd.DataFrame,
                         lz: pd.DataFrame, snap: pd.DataFrame | None = None,
                         asof: pd.Timestamp | None = None,
                         pill_secular: pd.DataFrame | None = None) -> None:
    """Daily rows for the pillar, carry, and overlay histories (mirrors
    append_today). snap = the live overlay_snapshot; its positioning columns are
    excluded here (pos_history carries those at weekly grain)."""
    asof = (asof or utcnow()).normalize()
    pill = pill_struct.rename(columns={"pscore": "struct"})         .merge(pill_regime.rename(columns={"pscore": "regime"})[
            ["ccy", "pillar", "regime"]], on=["ccy", "pillar"], how="outer")
    pill = pill[["ccy", "pillar", "struct", "regime"]]
    pill["date"], pill["kind"] = asof, "daily"
    _append_rows(PILLAR_HISTORY_NAME, pill, asof, ["date", "ccy", "pillar"])

    cv = lz[lz.metric.isin(["real_2y", "nominal_2y"])]         .pivot_table(index="ccy", columns="metric", values="value").reset_index()
    for c in ("real_2y", "nominal_2y"):
        if c not in cv.columns:
            cv[c] = np.nan
    cv = cv[["ccy", "real_2y", "nominal_2y"]]
    cv["date"], cv["kind"] = asof, "daily"
    _append_rows(CARRY_HISTORY_NAME, cv, asof, ["date", "ccy"])

    if snap is not None:
        ov = snap[[c for c in _OVL_COLS if c in snap.columns]].copy()
        ov["date"], ov["kind"] = asof, "daily"
        _append_rows(OVERLAY_HISTORY_NAME, ov, asof, ["date", "ccy"])


def load_history() -> pd.DataFrame | None:
    return read_cache(HISTORY_NAME)


def dial_options(hist: pd.DataFrame, horizon: str,
                 min_ccys: int = 4) -> list[pd.Timestamp]:
    """Month-ends the time dial may offer for a horizon: completed month-end rows
    only (kind == 'month_end'; daily appends are the LIVE terminal point, never a
    historical option — otherwise the current partial month appears as a phantom
    future-dated month-end), where at least min_ccys currencies have both axis
    scores (struct z's start ~8y after regime z's)."""
    if hist is None or not len(hist):
        return []
    h = hist[hist.kind == "month_end"] if "kind" in hist.columns else hist
    cols = [f"axis1_fundamental_{horizon}", f"axis2_stretch_{horizon}"]
    if not all(c in h.columns for c in cols):
        return []                     # horizon predates this history file's schema
    v = h.dropna(subset=cols)
    if v.empty:
        return []
    cnt = v.groupby(v.date + pd.offsets.MonthEnd(0))["ccy"].nunique()
    return sorted(cnt[cnt >= min_ccys].index)


if __name__ == "__main__":
    h = backfill()
    print(f"backfilled {h.date.nunique()} month-ends x "
          f"{h.ccy.nunique()} currencies -> {len(h)} rows "
          f"({h.date.min().date()} -> {h.date.max().date()})")
