"""Engine snapshot — the Stage-2 capstone that assembles the full current read and
persists it for the dashboard (Stage 6) and the daily Action.

Bundles: the two-axis tension map (structural + regime), the pillar scores behind it,
the pairwise carry grid, the objective overlays, and the inflection warnings. The
warnings make the balance visible wherever a conditional signal was bent.
"""
from __future__ import annotations

import json

import pandas as pd

from cte.adapters.base import CACHE_DIR, write_cache
from cte.flags.notes import completeness_warnings, context_notes
from cte.flags.overlays import overlay_snapshot, warnings
from cte.scoring.compositor import score, tension_map
from cte.transform.features import build_features
from cte.transform.pairwise import carry_grid, carry_ranking
from cte.transform.zscore import latest_z


def build_snapshot(persist: bool = True) -> dict:
    latest = build_features().rename(columns={"feature": "metric"})
    lz = latest_z(latest)
    snap = overlay_snapshot()

    tm, warns = tension_map()
    pill_struct, _ = score("struct_z", lz, snap)
    pillars = pill_struct.pivot_table(index="ccy", columns="pillar",
                                      values="pscore").round(2)

    real_grid = carry_grid("real_2y")
    nom_grid = carry_grid("nominal_2y")
    ranking = carry_ranking("real_2y")

    # narrative context notes (twin-signal, regime divergence, carry fragility, one-legged)
    for c, notes in context_notes(tm, pillars, real_grid).items():
        warns[c] = notes + warns.get(c, [])

    # Completeness guard runs LAST so it lands first (see notes.completeness_warnings:
    # it reindexes to the full pillar set so a whole-source outage is caught, not just
    # per-currency gaps).
    for ccy, msg in completeness_warnings(pillars).items():
        warns.setdefault(ccy, []).insert(0, msg)

    if persist:
        write_cache(tm, "tension_map")
        write_cache(pillars.reset_index(), "pillar_scores")
        write_cache(snap, "overlays")
        write_cache(real_grid.reset_index(), "carry_grid_real")
        write_cache(nom_grid.reset_index(), "carry_grid_nominal")
        (CACHE_DIR / "warnings.json").write_text(json.dumps(warns, indent=2))

    return {"tension_map": tm, "pillars": pillars, "overlays": snap,
            "carry_real": real_grid, "carry_nominal": nom_grid,
            "carry_ranking": ranking, "warnings": warns}


if __name__ == "__main__":
    s = build_snapshot()
    tm = s["tension_map"]
    print("=== TENSION MAP (structural) ===")
    q = tm.set_index("ccy")
    for ccy in q.index:
        f = q.loc[ccy, "axis1_fundamental_struct"]
        v = q.loc[ccy, "axis2_stretch_struct"]
        quad = (("improving" if f > 0 else "deteriorating") + " / " +
                ("stretched" if v > 0 else "cheap"))
        print(f"  {ccy}: fundamental {f:+.2f}, stretch {v:+.2f}   [{quad}]")
    print(f"\n{sum(len(v) for v in s['warnings'].values())} inflection warnings; "
          f"snapshot persisted to cache (tension_map, pillar_scores, overlays, "
          f"carry_grid_real, warnings.json).")
