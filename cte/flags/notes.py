"""Narrative context notes (spec §6 companion) — objective, plain-language reads that
add breadth beyond the mechanical overlay flags. Kept out of the engine so each trigger
is independently unit-testable and its thresholds live next to their siblings.

context_notes(tm, pillars, real_grid) -> dict[ccy, list[str]]
"""
from __future__ import annotations

import pandas as pd

from cte.config import PILLAR_AXIS, PILLAR_DISPLAY

# named thresholds (were magic numbers inline)
FUND_HI, FUND_LO = 0.25, -0.25          # fundamental "improving" / "deteriorating"
STRETCH_HI, STRETCH_LO = 0.25, -0.25    # valuation "expensive" / "cheap"
REGIME_RICH, REGIME_CHEAP = 0.4, -0.4   # decade-level extremes for regime divergence
REGIME_FLIP = 0.1                        # opposite-sign regime reading
CARRY_HI = 0.4                           # avg real carry vs the field (%)
CARRY_FRAGILE_STRETCH = 0.5              # valuation level that makes carry fragile
DOMINANCE = 3.0                          # a pillar this many x the rest = "one-legged"
DOMINANCE_MIN = 1.0                      # ...and at least this large in absolute terms


def completeness_warnings(pillars: pd.DataFrame) -> dict[str, str]:
    """Flag any currency missing a whole pillar. Reindexes to the FULL expected pillar
    set first: the pivot only has columns for pillars that scored for >=1 currency, so a
    pillar missing for *every* currency (a whole-source outage — the case this exists to
    catch) would otherwise leave no column, no NaN, and no warning."""
    expected = [p for p in PILLAR_AXIS if p != "F_carry"]
    chk = pillars.reindex(columns=expected)
    out: dict[str, str] = {}
    for ccy in chk.index:
        missing = [PILLAR_DISPLAY.get(p, p) for p in expected if pd.isna(chk.loc[ccy, p])]
        if missing:
            out[ccy] = (f"Incomplete data — missing the {', '.join(missing)} pillar(s); "
                        f"a source may be unavailable, so this position is provisional.")
    return out


def _get(row, col):
    """KeyError-safe access — a thin/degraded cache may lack the regime columns."""
    try:
        return row[col]
    except (KeyError, IndexError):
        return float("nan")


def context_notes(tm: pd.DataFrame, pillars: pd.DataFrame,
                  real_grid: pd.DataFrame) -> dict[str, list[str]]:
    q = tm.set_index("ccy")
    carry_avg = {c: real_grid.loc[c].drop(c, errors="ignore").mean()
                 for c in real_grid.index}
    out: dict[str, list[str]] = {}
    for c in q.index:
        f = _get(q.loc[c], "axis1_fundamental_struct")
        v = _get(q.loc[c], "axis2_stretch_struct")
        vr = _get(q.loc[c], "axis2_stretch_regime")
        notes: list[str] = []

        # twin-signal alignment — the headline read on where it sits
        if pd.notna(f) and pd.notna(v):
            if f > FUND_HI and v < STRETCH_LO:
                notes.append(f"Cheap and improving — the mean-reversion setup: valuation "
                             f"is below its own norm ({v:+.1f}) while fundamentals are "
                             f"turning up ({f:+.1f}).")
            elif f < FUND_LO and v > STRETCH_HI:
                notes.append(f"Expensive and deteriorating — the vulnerable combination: "
                             f"rich valuation ({v:+.1f}) on a softening fundamental "
                             f"trajectory ({f:+.1f}).")

        # regime divergence — the "off its highs, not cheap" case
        if pd.notna(v) and pd.notna(vr):
            if v > REGIME_RICH and vr < -REGIME_FLIP:
                notes.append(f"Rich versus its decade ({v:+.1f}) but easing versus its "
                             f"recent 2-year range ({vr:+.1f}) — read as 'off its highs,' "
                             f"not absolute cheap.")
            elif v < REGIME_CHEAP and vr > REGIME_FLIP:
                notes.append(f"Cheap versus its decade ({v:+.1f}) but firmer versus its "
                             f"recent range ({vr:+.1f}) — bouncing off lows rather than "
                             f"richly valued.")

        # carry on a fragile base
        ca = carry_avg.get(c, float("nan"))
        if pd.notna(ca) and ca > CARRY_HI and pd.notna(v) and pd.notna(f) \
                and (v > CARRY_FRAGILE_STRETCH or f < FUND_LO):
            base = ("a stretched valuation" if v > CARRY_FRAGILE_STRETCH
                    else "a deteriorating fundamental trajectory")
            notes.append(f"Fat carry on a fragile base: out-yields the field in real "
                         f"terms (avg {ca:+.1f}%) but sits on {base} — the classic "
                         f"carry-unwind setup.")

        # one-legged read — is a single pillar driving the whole position?
        if c in pillars.index:
            prow = pillars.loc[c].dropna()
            if len(prow) >= 3:
                dom = prow.abs().idxmax(); mag = prow[dom]
                rest = prow.drop(dom).abs().mean() or 0.01
                if abs(mag) > DOMINANCE_MIN and abs(mag) > DOMINANCE * rest:
                    notes.append(f"One-legged read: {PILLAR_DISPLAY.get(dom, dom)} "
                                 f"({mag:+.1f}) is doing almost all the work; the other "
                                 f"pillars are near neutral, so treat this as narrow, "
                                 f"not broad-based.")
        if notes:
            out[c] = notes
    return out
