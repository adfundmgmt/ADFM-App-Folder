"""Native read-only view of the scheduled Currency Tension Engine snapshot.

The upstream repository is the same source used by the original Streamlit
snapshot sync. This module never recalculates macro inputs or silently serves
the old snapshot committed in this application's checkout.
"""
from __future__ import annotations

from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import time

import numpy as np
import pandas as pd
import requests

from adfm_engine.serialization import records
from adfm_engine.services import DataUnavailable

SOURCE = "https://raw.githubusercontent.com/smileys21/currency_tension_tool-main/main/data/cache"
CURRENCIES = ("USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD")
FILES = ("tension_map", "pillar_scores", "overlays", "snapshot_history",
         "pillar_history", "carry_history", "overlay_history", "pos_history",
         "carry_grid_real", "carry_grid_nominal")
PILLAR_AXIS = {"A_growth": "axis1_fundamental", "B_inflation": "axis1_fundamental",
               "C_external": "axis1_fundamental", "D_fiscal": "axis1_fundamental",
               "E_policy": "axis2_stretch", "G_valuation": "axis2_stretch"}
DEFAULT_WEIGHTS = {name: (2.0 if name == "G_valuation" else 1.0) for name in PILLAR_AXIS}


def _download(name: str, suffix: str) -> bytes:
    try:
        response = requests.get(f"{SOURCE}/{name}.{suffix}", timeout=18)
        response.raise_for_status()
        if not response.content or len(response.content) > 6_000_000:
            raise ValueError("Snapshot file is empty or unexpectedly large.")
        return response.content
    except (requests.RequestException, ValueError) as exc:
        raise DataUnavailable("The current currency snapshot could not be loaded.") from exc


@lru_cache(maxsize=2)
def _snapshot(slot: int):
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            blobs = list(pool.map(lambda name: _download(name, "parquet"), FILES))
            warning_blob = _download("warnings", "json")
        frames = {name: pd.read_parquet(BytesIO(blob)) for name, blob in zip(FILES, blobs)}
        warnings = json.loads(warning_blob)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        raise DataUnavailable("The currency snapshot is incomplete or invalid.") from exc
    tm, history = frames["tension_map"], frames["snapshot_history"]
    if set(tm.ccy) != set(CURRENCIES) or history.empty or "date" not in history:
        raise DataUnavailable("The currency snapshot failed its coverage check.")
    asof = pd.Timestamp(history.date.max()).normalize()
    if len(pd.bdate_range(asof + pd.Timedelta(days=1), pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize())) > 3:
        raise DataUnavailable(f"The currency snapshot is stale (last updated {asof.date()}).")
    if any(not set(PILLAR_AXIS).issubset(frames[name].columns) for name in ("pillar_scores",)):
        raise DataUnavailable("The currency snapshot is missing pillar scores.")
    return frames, warnings, asof


def _months(history: pd.DataFrame, horizon: str):
    # A daily snapshot on the final business day is a completed month-end too.
    h = history.loc[history.kind.eq("month_end") | history.date.eq(history.date + pd.offsets.BMonthEnd(0))] if "kind" in history else history
    cols = [f"axis1_fundamental_{horizon}", f"axis2_stretch_{horizon}"]
    if not all(col in h for col in cols):
        return []
    valid = h.dropna(subset=cols)
    if valid.empty:
        return []
    counts = valid.groupby(valid.date + pd.offsets.MonthEnd(0))["ccy"].nunique()
    return [d.date().isoformat() for d in sorted(counts[counts >= 4].index)]


def _weighted_axes(pillars: pd.DataFrame, horizon: str, weights: dict):
    d = pillars.loc[pillars[horizon].notna(), ["ccy", "pillar", horizon]].copy()
    d["axis"] = d.pillar.map(PILLAR_AXIS)
    d["weight"] = d.pillar.map(weights)
    d = d.loc[d.weight.gt(0) & d.axis.notna()]
    if d.empty:
        return pd.DataFrame(columns=["ccy"])
    result = d.groupby(["ccy", "axis"]).apply(
        lambda group: np.average(group[horizon], weights=group.weight), include_groups=False
    ).rename("score").reset_index()
    return result.pivot(index="ccy", columns="axis", values="score").reset_index().rename(
        columns={axis: f"{axis}_{horizon}" for axis in ("axis1_fundamental", "axis2_stretch")}
    )


def _asof_rows(frame: pd.DataFrame, asof: pd.Timestamp, keys: list[str]):
    rows = frame.loc[(frame.date + pd.offsets.MonthEnd(0)).eq(asof)].sort_values("date")
    return rows.groupby(keys).tail(1)


def _carry_grid(frame: pd.DataFrame, basis: str):
    values = frame.set_index("ccy")[basis]
    names = [c for c in CURRENCIES if c in values and pd.notna(values[c])]
    return [{"base": base, **{quote: round(float(values[base] - values[quote]), 2) for quote in names}}
            for base in names]


def _position_label(lev_z, am_z):
    if pd.isna(lev_z):
        return "no data"
    if abs(lev_z) >= 1.5:
        side = "long" if lev_z > 0 else "short"
        opposed = pd.notna(am_z) and abs(am_z) >= 1 and np.sign(am_z) != np.sign(lev_z)
        return f"CROWDED {side}" + (" / real-money opposed" if opposed else "")
    if pd.notna(am_z) and abs(am_z) >= 1 and abs(lev_z) >= 1 and np.sign(am_z) != np.sign(lev_z):
        return "spec vs real-money split"
    return "normal"


def _trails(history: pd.DataFrame, tm: pd.DataFrame, horizon: str, trail: int, asof: pd.Timestamp | None):
    if not trail:
        return {}
    x, y = f"axis1_fundamental_{horizon}", f"axis2_stretch_{horizon}"
    h = history.loc[history.kind.eq("month_end") | history.date.eq(history.date + pd.offsets.BMonthEnd(0))] if "kind" in history else history
    if asof is not None:
        h = h[h.date.lt(asof)]
    h = h.dropna(subset=[x, y]).copy()
    h["month_end"] = h.date + pd.offsets.MonthEnd(0)
    h = h.sort_values("date").groupby(["ccy", "month_end"]).tail(1)
    out = {}
    for row in tm.itertuples(index=False):
        ccy = row.ccy
        past = h[h.ccy.eq(ccy)].sort_values("date").tail(trail)
        terminal = tm[tm.ccy.eq(ccy)].iloc[0]
        points = [{"date": r.date.date().isoformat(), "x": float(r[x]), "y": float(r[y])}
                  for _, r in past.iterrows()]
        if pd.notna(terminal.get(x)) and pd.notna(terminal.get(y)):
            points.append({"date": asof.date().isoformat() if asof is not None else "Live",
                           "x": float(terminal[x]), "y": float(terminal[y])})
        if len(points) > 1:
            out[ccy] = points
    return out


def load_currency_tension(*, horizon="struct", trail=6, asof=None, weights=None, session_slot=None):
    frames, warnings, latest_date = _snapshot(int(time.time() // 600) if session_slot is None else session_slot)
    hist = frames["snapshot_history"]
    months = _months(hist, horizon)
    historical = asof is not None
    if historical and asof not in months:
        raise ValueError("Choose an available completed month-end for this horizon.")
    selected = pd.Timestamp(asof) if historical else None
    tm = (_asof_rows(hist, selected, ["ccy"]) if historical else frames["tension_map"].copy())
    pillars = frames["pillar_scores"].copy()
    if historical or horizon != "struct":
        ph = frames["pillar_history"]
        phase = _asof_rows(ph, selected, ["ccy", "pillar"]) if historical else ph.loc[ph.date.eq(ph.date.max())]
        if not phase.empty:
            pillars = phase.pivot_table(index="ccy", columns="pillar", values=horizon).reset_index()
            pillars.columns.name = None
    weights = weights or DEFAULT_WEIGHTS
    if weights != DEFAULT_WEIGHTS:
        ph = frames["pillar_history"]
        phase = _asof_rows(ph, selected, ["ccy", "pillar"]) if historical else ph.loc[ph.date.eq(ph.date.max())]
        weighted = _weighted_axes(phase, horizon, weights)
        tm = tm.drop(columns=[f"axis1_fundamental_{horizon}", f"axis2_stretch_{horizon}"], errors="ignore").merge(weighted, on="ccy", how="left")
    tm = tm.loc[tm.ccy.isin(CURRENCIES)].copy()
    x, y = f"axis1_fundamental_{horizon}", f"axis2_stretch_{horizon}"
    tm["quadrant"] = np.select(
        [(tm[x] >= 0) & (tm[y] < 0), (tm[x] >= 0) & (tm[y] >= 0),
         (tm[x] < 0) & (tm[y] < 0), (tm[x] < 0) & (tm[y] >= 0)],
        ["Improving / Cheap", "Improving / Stretched", "Deteriorating / Cheap", "Deteriorating / Stretched"], default="Insufficient data")
    tm = tm.set_index("ccy").reindex(CURRENCIES).reset_index()
    if historical:
        carry = _asof_rows(frames["carry_history"], selected, ["ccy"])
        real, nominal = (_carry_grid(carry, basis) if basis in carry else [] for basis in ("real_2y", "nominal_2y"))
        overlay = _asof_rows(frames["overlay_history"], selected, ["ccy"])
        pos = frames["pos_history"].loc[frames["pos_history"].date.le(selected)].sort_values("date").groupby("ccy").tail(1)
        pos = pos.rename(columns={"date": "pos_date"})
        pos["pos_label"] = [_position_label(r.lev_z, r.am_z) for r in pos.itertuples()]
        hist_pos = frames["pos_history"].sort_values("date").groupby("ccy")
        for col in ("lev_z", "am_z"):
            previous = hist_pos.apply(lambda g: g.loc[g.date.le(selected), col].iloc[-14] if len(g.loc[g.date.le(selected)]) >= 14 else np.nan, include_groups=False).rename(f"{col}_13w")
            pos = pos.merge(previous.reset_index(), on="ccy", how="left")
        overlay = overlay.drop(columns=["kind", "date", "pos_label"], errors="ignore").merge(pos, on="ccy", how="left", suffixes=("", "_pos"))
    else:
        real, nominal = (records(frames[f"carry_grid_{basis}"]) for basis in ("real", "nominal"))
        overlay = frames["overlays"].copy()
        pos = overlay.copy()
    columns = ["ccy", "pos_label", "lev_z", "lev_z_13w", "am_z", "am_z_13w", "lev_pct_oi", "am_pct_oi", "pos_date"]
    position_rows = pos[[c for c in columns if c in pos]].dropna(subset=["lev_z"]) if "lev_z" in pos else pd.DataFrame()
    trail_rows = _trails(hist, tm, horizon, trail, selected)
    if weights != DEFAULT_WEIGHTS and trail_rows:
        # Historical trails under custom weights must use the same composition as the map.
        ph = frames["pillar_history"]
        for ccy, points in trail_rows.items():
            for point in points[:-1]:
                d = ph[ph.ccy.eq(ccy) & ph.date.eq(pd.Timestamp(point["date"]))]
                if not d.empty:
                    v = _weighted_axes(d, horizon, weights)
                    if not v.empty:
                        point["x"] = float(v.iloc[0].get(x, np.nan))
                        point["y"] = float(v.iloc[0].get(y, np.nan))
    return {
        "source": "Scheduled Currency Tension Engine snapshot",
        "snapshot_date": latest_date.date().isoformat(), "asof": asof or "Live", "horizon": horizon,
        "available_months": months, "weights": weights,
        "map": records(tm), "trails": trail_rows,
        "pillars": records(pillars.set_index("ccy").reindex(CURRENCIES).reset_index()),
        "carry_real": real, "carry_nominal": nominal,
        "positioning": records(position_rows), "overlays": records(overlay),
        "warnings": warnings, "flagged": list(warnings),
        "crowded": (overlay.loc[overlay.pos_label.astype(str).str.startswith("CROWDED"), "ccy"].tolist()
                    if not historical and "pos_label" in overlay else []),
        "historical_note": "Historical values use today's revised macro-data vintage and are not a point-in-time backtest." if historical else None,
    }
