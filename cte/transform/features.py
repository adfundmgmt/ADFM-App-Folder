"""Feature normalization — turn the raw cached panel into the pillar inputs the
scoring layer z-scores (spec §4-5).

Produces a tidy [date, ccy, feature, value] long frame. Fundamentals and rates are
aligned at month-end; genuinely quarterly series (gdp, niip, current_account) keep
their quarterly cadence and are z-scored on their own frequency downstream.

Features by pillar:
  A growth   bcicp, bcicp_slope (3m), gdp_yoy, unemp_3m_chg
  B inflation infl_gap (cpi-target), infl_momentum (3m change in YoY)
  C external  current_account, niip
  D fiscal    real_10y  (US: TIPS; others: 10Y nominal - CPI)
  E policy    real_policy (policy - CPI), priced_path (2Y nominal - policy)
  F carry     nominal_2y, real_2y            (absolute; differenced pairwise later)
  G valuation reer
"""
from __future__ import annotations

import pandas as pd

from cte.adapters.base import read_cache
from cte.config import CB_INFLATION_TARGET


def _month_end(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s) + pd.offsets.MonthEnd(0)


def _cal_ref(s: pd.Series, months: int) -> pd.Series:
    """The value `months` CALENDAR months before each date (period-end aligned), via
    exact date lookup — so a missing month or a mixed monthly/quarterly cadence can't
    silently shift the window. Missing reference period -> NaN (honest, not a wrong lag)."""
    ref_dates = (s.index - pd.DateOffset(months=months)) + pd.offsets.MonthEnd(0)
    prev = s.reindex(ref_dates)
    prev.index = s.index
    return prev


def _chg(series: pd.Series, months: int) -> pd.Series:
    """Change over `months` calendar months (period-end aligned), on the clean
    (non-null) series so mixed-frequency pivots and gaps don't corrupt the window."""
    s = series.dropna().sort_index()
    if len(s) < 2:
        return s.iloc[0:0]
    return s - _cal_ref(s, months)


def _pct(series: pd.Series, months: int) -> pd.Series:
    s = series.dropna().sort_index()
    if len(s) < 2:
        return s.iloc[0:0]
    return (s / _cal_ref(s, months) - 1) * 100


def _yields_me(yields: pd.DataFrame) -> pd.DataFrame:
    """Month-end nominal 2Y and 10Y per currency (last obs in each month)."""
    y = yields[yields.tenor.isin(["2Y", "10Y"])].copy()
    y["m"] = _month_end(y["date"])
    y = (y.sort_values("date").groupby(["ccy", "tenor", "m"]).tail(1))
    piv = y.pivot_table(index=["ccy", "m"], columns="tenor", values="value")
    return piv.rename(columns={"2Y": "nominal_2y", "10Y": "nominal_10y"}).reset_index()


def build_features() -> pd.DataFrame:
    mb = read_cache("macro_backbone")
    yields = read_cache("yields")
    reer = read_cache("reer")
    if mb is None or yields is None:
        raise RuntimeError("cache not seeded — run scripts.backfill first")

    # wide monthly-ish macro per (ccy, date) for the monthly/derived features
    mac = mb.copy()
    mac["date"] = pd.to_datetime(mac["date"])
    ym = _yields_me(yields)

    feats = []

    def emit(df, ccy, feature):
        d = df.dropna().copy()
        feats.append(pd.DataFrame({"date": d["date"].values, "ccy": ccy,
                                   "feature": feature, "value": d["value"].values}))

    for ccy in mac["ccy"].unique():
        m = mac[mac.ccy == ccy]
        wide = m.pivot_table(index="date", columns="metric",
                             values="value").sort_index()
        yc = ym[ym.ccy == ccy].set_index("m").sort_index()

        # ---- Pillar A: growth
        if "bcicp" in wide:
            emit(wide[["bcicp"]].rename(columns={"bcicp": "value"}).reset_index(),
                 ccy, "bcicp")
            emit(_chg(wide["bcicp"], 3).rename("value").reset_index(),
                 ccy, "bcicp_slope")
        if "gdp" in wide:
            emit(_pct(wide["gdp"], 12).rename("value").reset_index(), ccy, "gdp_yoy")
        if "unemp" in wide:
            emit(_chg(wide["unemp"], 3).rename("value").reset_index(),
                 ccy, "unemp_3m_chg")

        # ---- Pillar B: inflation
        if "cpi" in wide:
            tgt = CB_INFLATION_TARGET.get(ccy, 2.0)
            emit((wide["cpi"].dropna() - tgt).rename("value").reset_index(),
                 ccy, "infl_gap")
            emit(_chg(wide["cpi"], 3).rename("value").reset_index(),
                 ccy, "infl_momentum")

        # ---- Pillar C: external
        for f in ("current_account", "niip"):
            if f in wide:
                emit(wide[f].rename("value").reset_index(), ccy, f)

        # ---- Pillar G: valuation
        r = reer[reer.ccy == ccy][["date", "value"]] if reer is not None else None
        if r is not None and len(r):
            emit(r, ccy, "reer")

        # ---- rate-derived features (align CPI onto month-end grid)
        cpi_me = None
        if "cpi" in wide:
            cpi_me = wide["cpi"].copy()
            cpi_me.index = _month_end(cpi_me.index)
            cpi_me = cpi_me[~cpi_me.index.duplicated(keep="last")]

        if len(yc):
            grid = yc.index
            cpi_al = cpi_me.reindex(grid).ffill(limit=4) if cpi_me is not None else None
            # policy onto grid
            pol = None
            if "policy" in wide:
                pol = wide["policy"].copy(); pol.index = _month_end(pol.index)
                pol = pol[~pol.index.duplicated(keep="last")].reindex(grid).ffill(limit=4)

            # Pillar F: carry legs
            if "nominal_2y" in yc:
                emit(yc["nominal_2y"].rename_axis("date").rename("value").reset_index(),
                     ccy, "nominal_2y")
                if cpi_al is not None:
                    real2 = yc["nominal_2y"] - cpi_al
                    emit(real2.rename_axis("date").rename("value").reset_index(),
                         ccy, "real_2y")
                if pol is not None:
                    pp = yc["nominal_2y"] - pol
                    emit(pp.rename_axis("date").rename("value").reset_index(),
                         ccy, "priced_path")

            # Pillar D: 10Y real. US uses TIPS (backbone real_10y) EXCLUSIVELY — never
            # the nominal-CPI derivation, which is a different (breakeven vs realized)
            # basis; if TIPS is missing we emit nothing for USD rather than splice bases.
            if ccy == "USD":
                if "real_10y" in wide:
                    r10 = wide["real_10y"].copy(); r10.index = _month_end(r10.index)
                    r10 = r10[~r10.index.duplicated(keep="last")]
                    emit(r10.rename_axis("date").rename("value").reset_index(),
                         ccy, "real_10y")
            elif "nominal_10y" in yc and cpi_al is not None:
                emit((yc["nominal_10y"] - cpi_al).rename_axis("date")
                     .rename("value").reset_index(), ccy, "real_10y")

            # Pillar E: real policy rate
            if pol is not None and cpi_al is not None:
                emit((pol - cpi_al).rename_axis("date").rename("value").reset_index(),
                     ccy, "real_policy")

    out = pd.concat(feats, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return (out.dropna(subset=["value"]).sort_values(["ccy", "feature", "date"])
            .reset_index(drop=True))


if __name__ == "__main__":
    f = build_features()
    print(f"features: {len(f)} rows | {f.feature.nunique()} features | "
          f"{f.ccy.nunique()} ccys")
    latest = f.sort_values("date").groupby(["ccy", "feature"]).tail(1)
    piv = latest.pivot_table(index="ccy", columns="feature", values="value")
    order = ["bcicp", "bcicp_slope", "gdp_yoy", "unemp_3m_chg", "infl_gap",
             "infl_momentum", "current_account", "niip", "real_10y", "real_policy",
             "priced_path", "nominal_2y", "real_2y", "reer"]
    print(piv[[c for c in order if c in piv.columns]].round(2).to_string())
