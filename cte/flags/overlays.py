"""Flag / overlay layer (spec §6) — objective, data-driven, and transparent.

Everything here is computed from price and the z-scored features; no subjective sign
calls. These overlays are NOT in the axis composites — they modulate the two
genuinely-conditional signals (inflation, real-10Y) and surface warnings so the
balance is visible rather than buried in a number.

  yield_fx_regime   rolling corr(FX return, 10Y yield change) — is a currency being
                    REWARDED or PUNISHED for its yield? (the GBP-2022 stress tell).
                    Resolves "why is the real yield high" empirically. Drives the
                    real_10y multiplier.
  hike_feasibility  growth health minus policy restrictiveness — can the central bank
                    keep tightening on inflation, or is it trapped? Drives the
                    inflation multiplier (tailwind fades to headwind past the flip).
  carry_to_vol      2Y real carry / realized FX vol, percentiled — crowding/fragility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cte.adapters.base import read_cache
from cte.config import CURRENCIES
from cte.transform.features import build_features
from cte.transform.zscore import latest_z

_WIN = 63  # ~3 trading months


def _fx_wide() -> pd.DataFrame:
    fx = read_cache("fx_spot")
    w = fx.pivot_table(index="date", columns="ccy", values="value").sort_index()
    return w


def _y10_wide() -> pd.DataFrame:
    y = read_cache("yields")
    y = y[y.tenor == "10Y"]
    return y.pivot_table(index="date", columns="ccy", values="value").sort_index()


def yield_fx_regime(window: int = _WIN) -> pd.DataFrame:
    """Per ccy: latest rolling corr between FX daily return and 10Y yield daily change,
    a regime label, and a real_10y multiplier in [-1, +1] (tanh of the correlation).
    Positive corr = market rewards the yield (support holds); negative = risk-premium
    / stress regime, so the high real yield flips to a drag."""
    fx, y10 = _fx_wide(), _y10_wide()
    rows = []
    for c in CURRENCIES:
        if c not in fx or c not in y10:
            continue
        # inner-join the two calendars for THIS leg — no union+ffill, which would inject
        # zero-change days on one-sided holidays and bias the correlation toward 0.
        j = pd.concat({"fx": fx[c], "y": y10[c]}, axis=1, join="inner").dropna()
        ret = j["fx"].pct_change()        # value = USD per unit (USD row = DXY); up = stronger
        dY = j["y"].diff()
        corr = ret.rolling(window).corr(dY).dropna()
        if corr.empty:
            continue
        val = float(corr.iloc[-1])
        mult = float(np.tanh(2.0 * val))  # ~+1 rewarded, ~-1 punished, ~0 decoupled
        label = ("rewarded" if val > 0.15 else
                 "PUNISHED (stress)" if val < -0.15 else "decoupled")
        rows.append({"ccy": c, "yld_fx_corr": round(val, 2),
                     "yld_regime": label, "real10y_mult": round(mult, 2)})
    return pd.DataFrame(rows)


def _growth_and_policy(lz: pd.DataFrame) -> pd.DataFrame:
    """Growth-momentum z and real-policy z per ccy from the latest feature z-scores."""
    g_feats = {"bcicp_slope": 1, "gdp_yoy": 1, "unemp_3m_chg": -1}
    d = lz[lz.metric.isin(g_feats)].copy()
    d["signed"] = d["struct_z"] * d["metric"].map(g_feats)
    growth = d.groupby("ccy")["signed"].mean().rename("growth_z")
    rp = (lz[lz.metric == "real_policy"].set_index("ccy")["struct_z"]
          .rename("real_policy_z"))
    return pd.concat([growth, rp], axis=1).reset_index()


def hike_feasibility(lz: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per ccy: feasibility = growth health − policy restrictiveness, and an inflation
    multiplier in [-1, +1]. High feasibility → inflation is a hawkish tailwind (+1);
    low/negative (weak growth + already-tight policy) → the trap, tailwind → headwind."""
    if lz is None:
        lz = latest_z(build_features().rename(columns={"feature": "metric"}))
    gp = _growth_and_policy(lz)
    gp["feasibility"] = gp["growth_z"].fillna(0) - gp["real_policy_z"].fillna(0)
    gp["infl_mult"] = np.tanh(gp["feasibility"]).round(2)
    return gp[["ccy", "growth_z", "real_policy_z", "feasibility", "infl_mult"]].round(2)


def carry_to_vol(window: int = _WIN) -> pd.DataFrame:
    """Per ccy: 2Y real carry / realized FX vol (annualized), and its percentile vs
    own history. A high percentile = a lot of carry riding on quiet vol = crowded /
    fragile (prone to violent unwind when vol wakes up)."""
    fx = _fx_wide()
    feats = build_features()
    r2 = feats[feats.feature == "real_2y"]
    rows = []
    for c in CURRENCIES:
        if c not in fx:
            continue
        vol = (fx[c].pct_change().rolling(window).std() * np.sqrt(252) * 100)
        vol.index = vol.index + pd.offsets.MonthEnd(0)
        vol = vol[~vol.index.duplicated(keep="last")]
        carry = r2[r2.ccy == c].set_index("date")["value"]
        j = pd.concat({"carry": carry, "vol": vol}, axis=1, sort=True).dropna()
        if len(j) < 24:
            continue
        ratio = (j["carry"] / j["vol"]).replace([np.inf, -np.inf], np.nan).dropna()
        pctile = (ratio.rank(pct=True).iloc[-1]) * 100
        rows.append({"ccy": c, "carry_to_vol": round(float(ratio.iloc[-1]), 2),
                     "ctv_pctile": round(float(pctile))})
    return pd.DataFrame(rows)


def warnings(snapshot: pd.DataFrame | None = None,
             lz: pd.DataFrame | None = None) -> dict[str, list[str]]:
    """Per-currency sidenotes that surface the balance when a conditional signal is
    near/through an inflection — so the ambiguity is visible, not hidden in a score.
    Thresholds are objective; confounds are flagged explicitly."""
    if lz is None:
        lz = latest_z(build_features().rename(columns={"feature": "metric"}))
    if snapshot is None:
        snapshot = overlay_snapshot()
    infl = lz[lz.metric == "infl_gap"].set_index("ccy")["value"]
    r10 = lz[lz.metric == "real_10y"].set_index("ccy")["struct_z"]
    out: dict[str, list[str]] = {}
    for _, r in snapshot.iterrows():
        c = r["ccy"]; notes = []
        gap = infl.get(c, np.nan)
        # policy-trap: above-target inflation but can't tighten
        if pd.notna(gap) and gap > 0 and pd.notna(r["feasibility"]) and r["feasibility"] < -0.5:
            notes.append(
                f"Policy trap: inflation above target but growth soft and policy "
                f"already tight (feasibility {r['feasibility']:+.1f}) — the hawkish "
                f"tailwind is fading toward a stagflation drag; inflation's axis-1 "
                f"contribution is dampened (mult {r['infl_mult']:+.2f}).")
        # yield stress: high real 10Y but currency punished for it
        if (pd.notna(r10.get(c, np.nan)) and r10.get(c) > 0.5
                and str(r["yld_regime"]).startswith("PUNISHED")):
            notes.append(
                f"Yield stress: real 10Y is high (z {r10.get(c):+.1f}) but the "
                f"currency weakens as yields rise ({r['yld_fx_corr']:+.2f} corr) — "
                f"risk-premium regime, not reward; real-yield support downweighted "
                f"(mult {r['real10y_mult']:+.2f}). Note: cross vs USD partly confounds "
                f"this in a strong-dollar tape.")
        elif str(r["yld_regime"]) == "decoupled":
            notes.append(
                "FX and yields are decoupled — the real-yield reward/stress read is "
                "ambiguous this window; treat Pillar D lightly.")
        # carry crowding
        if pd.notna(r["ctv_pctile"]) and r["ctv_pctile"] >= 85:
            notes.append(
                f"Carry crowded: carry-to-vol in the {int(r['ctv_pctile'])}th "
                f"percentile — a lot of carry riding on quiet vol, fragile to a vol spike.")
        if notes:
            out[c] = notes
    return out


def overlay_snapshot() -> pd.DataFrame:
    lz = latest_z(build_features().rename(columns={"feature": "metric"}))
    reg = yield_fx_regime()
    feas = hike_feasibility(lz)
    ctv = carry_to_vol()
    out = reg.merge(feas, on="ccy", how="outer").merge(ctv, on="ccy", how="outer")
    return out.set_index("ccy").reindex(CURRENCIES).reset_index()


if __name__ == "__main__":
    snap = overlay_snapshot()
    print(snap.to_string(index=False))
    print("\n=== Warnings / sidenotes ===")
    for c, notes in warnings(snap).items():
        print(f"\n{c}:")
        for n in notes:
            print(f"  - {n}")
