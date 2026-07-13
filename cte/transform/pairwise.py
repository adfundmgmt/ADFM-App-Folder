"""Pairwise carry & positioning grid (spec §Pillar F — pure pairwise).

Carry and positioning are the one part of the engine that is *not* scored against a
currency's own history but differenced across the grid. The dollar is one leg of
eight here, not a hub. Produces the 8x8 matrix of 2Y real-rate differentials (base
minus quote) and the nominal-2Y differential, from the latest month-end features.

carry[base, quote] = real_2y(base) - real_2y(quote)  (positive = base out-yields quote)
"""
from __future__ import annotations

import pandas as pd

from cte.config import CURRENCIES
from cte.transform.features import build_features


def _latest(feature: str, feats: pd.DataFrame) -> pd.Series:
    d = feats[feats.feature == feature]
    last = d.sort_values("date").groupby("ccy").tail(1)
    return last.set_index("ccy")["value"]


def grid_from_values(vals: pd.Series) -> pd.DataFrame:
    """Antisymmetric differential grid from per-ccy values (row=base, col=quote,
    cell = base - quote). Shared by the live grid and the historical tabs."""
    ccys = [c for c in CURRENCIES if c in vals.index and pd.notna(vals[c])]
    grid = pd.DataFrame({b: {q: vals[b] - vals[q] for q in ccys} for b in ccys}).T
    grid.index.name = "base"
    return grid.loc[ccys, ccys].round(2)


def carry_grid(feature: str = "real_2y") -> pd.DataFrame:
    """8x8 differential matrix for a per-leg rate feature (real_2y or nominal_2y).
    Row = base leg, col = quote leg; cell = base - quote."""
    feats = build_features()
    return grid_from_values(_latest(feature, feats))


def carry_ranking(feature: str = "real_2y") -> pd.DataFrame:
    """Each leg's average carry vs the other seven — a quick 'who's the funder /
    who's the target' read distilled from the grid."""
    feats = build_features()
    vals = _latest(feature, feats).reindex(
        [c for c in CURRENCIES if c in _latest(feature, feats).index])
    avg = {c: (vals[c] - vals.drop(c)).mean() for c in vals.index}
    out = pd.Series(avg).sort_values(ascending=False)
    return out.rename("avg_carry_vs_rest").round(2).reset_index().rename(
        columns={"index": "ccy"})


if __name__ == "__main__":
    print("=== 2Y real-rate carry grid (base - quote, %) ===")
    print(carry_grid("real_2y").to_string())
    print("\n=== Carry ranking (avg 2Y real vs the other 7) ===")
    print(carry_ranking("real_2y").to_string(index=False))
