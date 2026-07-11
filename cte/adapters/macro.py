"""Unified macro backbone — assembles the fundamental panel the scoring layer uses.

Combines the macro sources into one tidy long frame, with a clear source of
truth per (currency, metric):

  bcicp            OECD amplitude-adjusted business confidence for all 8 (standalone
                   Pillar A leading indicator; replaced the composite CLI entirely).
  cpi              Headline CPI YoY for all 8: FRED index (US/EUR/CHF, YoY'd here);
                   OECD headline YoY (GBP/CAD/AUD/NZD); Japan e-Stat (JPY).
  unemp            FRED for USD/JPN/GBR/CHF/CAD/AUS/NZD; Eurostat for EUR.
  gdp              Real GDP level (chain-linked vol): OECD QNA for 7, FRED for EUR.
  current_account  OECD BOP, % of GDP (Pillar C flow), all 8.
  niip             Net IIP % of GDP (Pillar C stock): OECD IIP/QNA for 7, Eurostat EUR.
  policy           BIS CBPOL actual central-bank policy rate, all 8 (uniform source).
  real_10y         FRED US TIPS (US only; other legs derived in transform).

FRED headline CPI arrives as an index and is converted to YoY here so every leg's
inflation input is a comparable year-on-year rate under the unified metric 'cpi'.
Contract: [date, ccy, metric, value, source, fetched_at].
"""
from __future__ import annotations

import pandas as pd

from cte.adapters import bis_reer, estat, eurostat, external, fred, oecd, ons


def _cpi_index_to_yoy(df: pd.DataFrame) -> pd.DataFrame:
    """Convert FRED headline-CPI index series (metric 'cpi_index') to YoY % ('cpi'),
    using a calendar 12-month lookup (period-end aligned) so a dropped month can't
    silently turn the YoY into a 13-month change."""
    out = []
    for ccy, g in df[df.metric == "cpi_index"].groupby("ccy"):
        g = g.sort_values("date").copy()
        s = g.set_index("date")["value"]
        ref = (s.index - pd.offsets.DateOffset(months=12)) + pd.offsets.MonthEnd(0)
        prev = s.reindex(ref); prev.index = s.index
        g["value"] = ((s / prev - 1) * 100).values
        g["metric"] = "cpi"
        g["source"] = g["source"] + ":yoy"
        out.append(g.dropna(subset=["value"]))
    return pd.concat(out, ignore_index=True) if out else df.head(0)


def build_macro_backbone() -> pd.DataFrame:
    fred_df = fred.fetch_fred_macro()
    oecd_df = oecd.fetch_oecd_all()
    ea_df = eurostat.fetch_ea_unemployment()
    jp_cpi = estat.fetch_estat_cpi()
    ca_df = external.fetch_current_account()
    niip_df = external.fetch_niip()
    policy_df = bis_reer.fetch_policy_rates()
    gb_unemp = ons.fetch_uk_unemployment()

    # FRED headline CPI (index) -> unified YoY 'cpi'; OECD/e-Stat already provide YoY
    cpi_yoy = _cpi_index_to_yoy(fred_df)
    oecd_cpi = oecd_df[oecd_df.metric == "cpi_yoy"].copy()
    oecd_cpi["metric"] = "cpi"
    jp_cpi = jp_cpi.copy()
    jp_cpi["metric"] = "cpi"
    # CLI replaced by OECD business confidence (BCICP) for all 8, standalone
    oecd_bcicp = oecd_df[oecd_df.metric == "bcicp"].copy()
    # Real GDP levels: OECD QNA for 7 legs, FRED (EUR) via fred_rest — uniform basis
    oecd_gdp = oecd_df[oecd_df.metric == "gdp"].copy()
    # Unemployment: FRED (USD) + Eurostat (EUR) + ONS (GBP) + OECD (JPY/CHF/CAD/AUD/NZD)
    oecd_unemp = oecd_df[oecd_df.metric == "unemp"].copy()

    fred_rest = fred_df[~fred_df.metric.isin(["cpi_index"])].copy()

    panel = pd.concat([fred_rest, cpi_yoy, oecd_bcicp, oecd_cpi, oecd_gdp,
                       oecd_unemp, gb_unemp, jp_cpi, ea_df, ca_df, niip_df,
                       policy_df], ignore_index=True)
    return panel.sort_values(["ccy", "metric", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    from cte.config import CURRENCIES
    p = build_macro_backbone()
    print(f"rows: {len(p)} | groups: {p.groupby(['ccy','metric']).ngroups}")
    latest = p.sort_values("date").groupby(["ccy", "metric"]).tail(1)
    metrics = ["bcicp", "cpi", "unemp", "gdp", "current_account", "niip", "policy", "real_10y"]
    print(f"\n{'ccy':<5}" + "".join(f"{m:<10}" for m in metrics))
    for ccy in CURRENCIES:
        row = f"{ccy:<5}"
        for m in metrics:
            sub = latest[(latest.ccy == ccy) & (latest.metric == m)]
            row += f"{sub.iloc[0]['value']:<10.2f}" if len(sub) else f"{'—':<10}"
        print(row)
