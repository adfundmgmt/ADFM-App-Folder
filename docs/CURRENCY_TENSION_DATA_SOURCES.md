# CTE — Data Sources & Metrics vs. Spec v4

Status as of 2026-07-07. Everything below is verified against live pulls and persisted
to the cache by `scripts/backfill.py`. Maps what we ingest onto the spec's two axes and
seven pillars, with the source of truth and freshness for each input.

Currencies (8): USD · EUR · JPY · GBP · CHF · CAD · AUD · NZD

Two design principles drove the source choices: **apples-to-apples** (one definition per
metric across all legs) and **freshness** (each series as current as its primary source
allows). Where those conflicted we favored consistency, except where the freshest source
*is* the consistent one (UK unemployment).

---

## 1. Source inventory

| Source | Adapter | Provides | Cadence | Auth |
|---|---|---|---|---|
| Yahoo Finance | `yahoo` | 8 FX legs (USD/x), DXY | daily | none |
| US Treasury / ECB / MoF / BoE / BoC / RBA / RBNZ / SNB | `sovereign_yields` | sovereign 2/3/10Y curves | daily | none |
| Bank of England archive | `sovereign_yields.fetch_boe(history=True)` | GBP GLC spot curve back to 1979 | daily | none |
| CFTC | `cftc_tff` | TFF positioning, 8 FX futures | weekly | none |
| BIS | `bis_reer` | REER (8) + **central-bank policy rates (8)** | monthly | none |
| FRED | `fred` | US CPI/unemp/real-yield, EUR CPI/GDP, CHF CPI | daily–qtrly | key |
| OECD SDMX | `oecd` | BCICP (8), headline CPI (4), real GDP (7), unemployment (5) | monthly/qtrly | none |
| OECD SDMX | `external` | current account %GDP (8), net IIP (7) | quarterly | none |
| Eurostat | `eurostat` / `external` | EA unemployment, EA NIIP %GDP | monthly/qtrly | none |
| Japan e-Stat | `estat` | JP headline CPI | monthly | key |
| UK ONS | `ons` | UK unemployment (LFS, = OECD harmonised) | monthly | none |

`macro.build_macro_backbone()` unifies the macro rows into one panel (57 series). The
one-time cold-start backfill seeds full history; the daily Action merges the tail.

---

## 2. Axis 1 — Fundamental trajectory

### Pillar A · Growth
| Metric | Source | Coverage | Notes |
|---|---|---|---|
| Business confidence (BCICP) | OECD, amplitude-adjusted | all 8 | standalone leading indicator; replaced the composite CLI (which was discontinued for EUR/CHF/NZD) so the leading signal is uniform |
| Real GDP (level → YoY) | OECD QNA (7) + FRED (EUR) | all 8 | chain-linked volume levels, uniform basis; YoY computed in transform |
| Unemployment | FRED (USD) · ONS (GBP) · Eurostat (EUR) · OECD (JPY/CHF/CAD/AUD/NZD) | all 8 | harmonised LFS rate everywhere; GBP on ONS (identical to OECD harmonised, 0.016pp mean diff, but 1–2mo fresher) |

### Pillar B · Inflation
| Metric | Source | Coverage | Basis |
|---|---|---|---|
| Headline CPI YoY − CB target | FRED (USD/EUR/CHF index) · OECD (GBP/CAD/AUD/NZD) · e-Stat (JPY) | all 8 | **headline all-items**, uniform — more apples-to-apples than core (core definitions differ by country) and matches the CB target basis |

### Pillar C · External
| Metric | Source | Coverage | Notes |
|---|---|---|---|
| Current account % GDP (flow) | OECD BOP | all 8 | primary FX-pressure signal |
| Net IIP % GDP (stock) | OECD IIP/QNA (7) + Eurostat (EUR) | all 8 | accumulated-imbalance / creditor-status signal |
| — combine CA + NIIP into **one** pillar-level input in scoring so external isn't double-weighted vs single-metric pillars. |

### Pillar D · Fiscal
| Metric | Source | Coverage | Notes |
|---|---|---|---|
| 10Y real yield | FRED TIPS (USD) + derived (other 7) | all 8 | non-US = 10Y nominal − CPI YoY, in transform |

---

## 3. Axis 2 — Valuation & policy stretch

### Pillar E · Policy room
| Metric | Source | Coverage | Notes |
|---|---|---|---|
| Policy rate | **BIS CBPOL** | all 8 | actual central-bank rate, one uniform source/definition (replaced FRED-actual + 3M-interbank mix) |
| 2Y real | derived in transform | all 8 | 2Y nominal − CPI YoY |

### Pillar F · Carry (pairwise)
| Metric | Source | Coverage | Notes |
|---|---|---|---|
| 2Y real differential | sovereign 2Y + CPI | all 28 crosses | pure pairwise; USD one leg of 8 |
| Positioning | CFTC TFF | all 8 | leveraged & asset-mgr net, % OI |

### Pillar G · REER
| Metric | Source | Coverage |
|---|---|---|
| Real effective exchange rate | BIS | all 8 |

---

## 4. Overlay flags

| Flag | Data | Status |
|---|---|---|
| TFF positioning | CFTC | ✅ |
| Technical / correlation-flip | FX spot | ✅ derivable |
| Carry-to-vol | realized vol from FX spot | ✅ (realized, not implied — deliberate) |
| Catalyst proximity | CB calendar / news | ⏳ deferred (per your call) |

---

## 5. Data-quality posture

- **History depth** — the backfill's depth audit confirms every scored series clears the
  ~10y structural z-score window. (One exception: NZD sovereign yields reach back to 2018
  / 8.5y; the current RBNZ file starts there. Usable; an archive would close it.)
- **Freshness** — a lag audit flags any series stale beyond its natural publication
  cadence. Unemployment was migrated off FRED's lagging mirror to fresher primaries.
  Remaining known lags: EUR current account (OECD compiles the euro-area aggregate ~2
  quarters late) and GBP current account (one quarter behind) — both inherently laggy and
  Pillar C is lightly weighted.
- **Date alignment** — all quarterly/monthly observations are snapped to period-end so
  cross-source dates line up (FRED labels periods at the start; OECD at the end).
- **Yield basis** — curves are kept on each source's published convention (mostly
  zero-coupon; a few par/YTM). Standardizing all to zero-coupon would require lossy
  bootstrapping of the YTM-only curves — a net quality loss for the ~1-2bp it would fix at
  the 2Y point — so it's deliberately not done.

## 6. Divergences from the original vision — resolved

1. **Pillar C external** — now sourced (current account + NIIP). ✅
2. **CPI** — uniform headline (was a core/headline mix). ✅
3. **CLI** — uniform BCICP (was composite-where-available + proxy). ✅
4. **GDP** — uniform real-GDP levels (was a level/growth-rate mix). ✅
5. **Policy rates** — uniform BIS actual (was FRED-actual + interbank proxy). ✅
6. **Unemployment freshness** — migrated to fresher primaries. ✅

Open: catalyst-proximity flag (deferred); the two current-account lags above; NZD yield
depth. Nothing blocks the transform layer.
