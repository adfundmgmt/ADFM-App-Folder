"""
Central configuration for the Currency Tension Engine.

Single source of truth for: the 8-currency universe, the spot-quoting
convention (USD crosses + triangulation, DXY for the dollar leg), the
sovereign-yield source map, and target bond tenors. Per the spec, USD is one
leg among eight — there is no hub currency; "USD-based" here is a *quoting*
convention only.
"""
from __future__ import annotations

from pathlib import Path

# ------------------------------------------------------------------ universe
# Spec §1 universe. Order is canonical for grids/tables.
CURRENCIES: tuple[str, ...] = ("USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD")

# ------------------------------------------------------------------ paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ FX spot
# Spec §1: pairs displayed as USD crosses, cross rates triangulated; the
# dollar's own leg uses DXY. yfinance tickers for the 7 non-USD legs vs USD,
# plus DXY for the dollar leg. Every cross in the 28-grid is triangulated from
# these eight price legs at transform time — we never pull 28 pairs.
YF_USD_PAIRS: dict[str, str] = {
    "EUR": "EURUSD=X",   # USD per EUR
    "GBP": "GBPUSD=X",   # USD per GBP
    "JPY": "USDJPY=X",   # JPY per USD  (inverted at transform)
    "CHF": "USDCHF=X",   # CHF per USD  (inverted)
    "CAD": "USDCAD=X",   # CAD per USD  (inverted)
    "AUD": "AUDUSD=X",   # USD per AUD
    "NZD": "NZDUSD=X",   # USD per NZD
}
YF_DXY = "DX-Y.NYB"      # dollar's own price leg (broad-ish DXY proxy)

# True for tickers quoted as (foreign per USD), i.e. need inverting to get
# "USD value of 1 unit of foreign" on a common basis.
YF_PAIR_INVERTED: dict[str, bool] = {
    "EUR": False, "GBP": False, "JPY": True,
    "CHF": True, "CAD": True, "AUD": False, "NZD": False,
}

# ------------------------------------------------------------------ tenors
# Carry / priced-path leg = 2Y; fiscal / pillar-D leg = 10Y. Every currency in
# the universe exposes a live 2Y today, so no tenor fallback is wired. We still
# pull 3Y where a source publishes it (cheap, and useful for later curve work).
TENORS_WANTED: tuple[str, ...] = ("2Y", "3Y", "10Y")
CARRY_TENOR = "2Y"
FISCAL_TENOR = "10Y"

# ------------------------------------------------------------------ yield sources
# Spec §7. Nine endpoints for eight currencies: EUR carries two legs — the ECB
# euro-area composite curve (canonical EUR yield for scoring) and the German
# Bund (cleanest single-name benchmark, kept as secondary cross-reference).
# `role`: "primary" feeds the currency's pillar; "secondary" is available but
# not wired into scoring unless promoted.
YIELD_SOURCES: dict[str, dict] = {
    "USD": {"adapter": "us_treasury", "role": "primary"},
    "EUR": {"adapter": "ecb_yc",      "role": "primary"},
    "EUR_DE": {"adapter": "bundesbank", "role": "secondary", "ccy": "EUR"},
    "JPY": {"adapter": "jp_mof",      "role": "primary"},
    "GBP": {"adapter": "boe_iadb",    "role": "primary"},
    "CHF": {"adapter": "snb",         "role": "primary"},
    "CAD": {"adapter": "boc_valet",   "role": "primary"},
    "AUD": {"adapter": "rba_f2",      "role": "primary"},
    "NZD": {"adapter": "rbnz_b2",     "role": "primary"},
}

# BIS REER reference-area codes per currency (verified at BIS adapter stage).
BIS_EER_AREA: dict[str, str] = {
    "USD": "US", "EUR": "XM", "JPY": "JP", "GBP": "GB",
    "CHF": "CH", "CAD": "CA", "AUD": "AU", "NZD": "NZ",
}

# CFTC Traders-in-Financial-Futures contract-market codes per currency.
# CME majors plus the ICE U.S. Dollar Index. Verified live at the TFF stage.
# Cross-rate contracts (EUR/GBP, EUR/JPY, …) are deliberately excluded; we want
# only the USD-quoted legs so positioning stays on the same axis as the FX block.
CFTC_TFF_CODES: dict[str, str] = {
    "EUR": "099741",  # EURO FX
    "JPY": "097741",  # JAPANESE YEN
    "GBP": "096742",  # BRITISH POUND
    "CHF": "092741",  # SWISS FRANC
    "CAD": "090741",  # CANADIAN DOLLAR
    "AUD": "232741",  # AUSTRALIAN DOLLAR
    "NZD": "112741",  # NZ DOLLAR
    "USD": "098662",  # USD INDEX (ICE Futures U.S.)
}
# ------------------------------------------------------------------ OECD SDMX (gap filler)
# Closes the cross-country inputs FRED can no longer serve (see FRED_GAPS). Uses
# the new OECD Data Explorer SDMX API (sdmx.oecd.org) with SDMX-CSV negotiation.
OECD_SDMX_BASE = "https://sdmx.oecd.org/public/rest/data"
OECD_CSV_ACCEPT = "application/vnd.sdmx.data+csv"

# OECD ref-area codes per currency (euro area = EA20 aggregate).
OECD_REF_AREA: dict[str, str] = {
    "USD": "USA", "EUR": "EA20", "JPY": "JPN", "GBP": "GBR",
    "CHF": "CHE", "CAD": "CAN", "AUD": "AUS", "NZD": "NZL",
}

# Pillar A leading indicator: amplitude-adjusted business confidence (BCICP) for
# ALL 8 currencies — a PMI-equivalent business-tendency survey, used standalone and
# uniformly (replacing the composite CLI entirely) so the leading signal is apples-
# to-apples across legs. Dataflow DSD_STES@DF_CLI, MEASURE=BCICP, ADJUSTMENT=AA,
# TRANSFORMATION=IX.
OECD_BCICP_DATAFLOW = "OECD.SDD.STES,DSD_STES@DF_CLI,"  # DF_CLI hosts BCICP
OECD_BCICP_CCYS = ("USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD")

# CPI gap (GBP/CAD/AUS/NZL): headline CPI YoY from DSD_PRICES@DF_PRICES_ALL, filter
# EXPENDITURE=_T, MEASURE=CPI, TRANSFORMATION=GY, UNIT_MEASURE=PA. Monthly for
# GBP/CAD/AUS, quarterly for NZL (national release cadence). Headline (not core)
# because a consistent monthly core across these does not exist free; US/EUR/CHF
# keep FRED's core, so the inflation input is core for those three and headline
# here — a definitional split documented in the README.
OECD_CPI_DATAFLOW = "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,"
OECD_CPI_CCYS = ("GBP", "CAD", "AUD", "NZD")

# ------------------------------------------------------------------ Japan e-Stat (JPY CPI)
# Closes the last macro gap: Japan CPI (absent from FRED, OECD prices, Eurostat).
# Japan's Statistics Bureau portal (e-Stat) serves it via a free appId, read from
# env (never committed). Table 0003427113 = CPI 2020-base; tab=3 is YoY %, cat01
# 0001 is all-items headline (総合) — matching the OECD-headline basis used for the
# other non-FRED legs — and area 00000 is nationwide. YoY history runs to 1971.
ESTAT_BASE = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"
ESTAT_APP_ID_ENV = "ESTAT_APP_ID"
ESTAT_CPI_TABLE = "0003427113"
ESTAT_CPI_TAB_YOY = "3"          # 前年同月比 (year-on-year %)
ESTAT_CPI_ITEM_HEADLINE = "0001"  # 総合 (all items); 0161 = ex-fresh-food "core"
ESTAT_CPI_AREA_NATIONWIDE = "00000"

# UK unemployment from ONS directly (series MGSX, 16+, SA) — the primary source
# for what is, empirically, the same number as the OECD harmonised UK rate, but
# 1-2 months fresher. GBP is therefore dropped from the OECD unemployment set below.
ONS_UNEMP_URL = ("https://www.ons.gov.uk/employmentandlabourmarket/peoplenotinwork/"
                 "unemployment/timeseries/mgsx/lms/data")

# Unemployment for the non-US/EUR/GBP legs from OECD infra-annual labour stats —
# uniformly ~1-2 months fresher than FRED's harmonised mirror. Monthly where it has
# depth (JPY/CAD/AUD), quarterly for CHF/NZD. Filter MEASURE=UNE_LF_M,
# UNIT_MEASURE=PT_LF_SUB, SEX=_T, ADJUSTMENT=Y, AGE=Y_GE15. US stays FRED (UNRATE);
# EUR stays Eurostat; GBP is ONS (above).
OECD_UNEMP_DATAFLOW = "OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,"
OECD_UNEMP_CCYS = ("JPY", "CHF", "CAD", "AUD", "NZD")

# Real GDP levels (chain-linked volume, SA) from OECD Quarterly National Accounts,
# for all legs except EUR (euro-area aggregate isn't in this dataflow; it stays on
# FRED's chain-linked-volume level). Uniform real-GDP-level basis -> YoY in transform.
# Filter TRANSACTION=B1GQ, PRICE_BASE=L, ADJUSTMENT=Y, EXPENDITURE=_Z, FREQ=Q.
OECD_QNA_DATAFLOW = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_NATIO_CURR,"
OECD_GDP_CCYS = ("USD", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD")

# ------------------------------------------------------------------ OECD BOP (Pillar C external)
# Current account as % of GDP — the primary external-pressure signal. Direct from
# the OECD balance-of-payments dataflow (UNIT_MEASURE PT_B1GQ = percent of GDP,
# ACCOUNTING_ENTRY B = balance, quarterly, counterpart = World). All 8 covered.
# NIIP (stock) is a planned light structural companion; it lives in DF_IIP but only
# in currency units (needs a nominal-GDP denominator) and lacks the euro-area
# aggregate, so it is deferred to a follow-up.
OECD_BOP_DATAFLOW = "OECD.SDD.TPS,DSD_BOP@DF_BOP,"
OECD_IIP_DATAFLOW = "OECD.SDD.TPS,DSD_BOP@DF_IIP,"

CFTC_TFF_DATASET = "gpe5-46if"  # TFF, Futures-Only

# ------------------------------------------------------------------ FRED (macro backbone)
# Spec §3/§14: FRED is the fundamental backbone + US real rates. It serves the
# cross-country macro inputs (CLI, GDP, unemployment, core CPI, policy rate) for
# all eight legs, plus US TIPS real yields.
#
# The API key is read from the environment (never hard-coded / committed):
FRED_API_KEY_ENV = "FRED_API_KEY"
FRED_BASE = "https://api.stlouisfed.org/fred"

# Central-bank inflation targets (Pillar B/E use *distance from target*, which is
# more comparable across countries than raw CPI). Point targets; RBA/RBNZ midpoints.
CB_INFLATION_TARGET: dict[str, float] = {
    "USD": 2.0, "EUR": 2.0, "JPY": 2.0, "GBP": 2.0,
    "CHF": 1.0,   # SNB: price stability defined as <2%; ~1% used as operating midpoint
    "CAD": 2.0, "AUD": 2.5, "NZD": 2.0,   # RBA target midpoint 2.5; RBNZ 2.0 midpoint
}

# ---------------------------------------------------------------- scoring / compositor
# Which pillar each feature belongs to, and which axis each pillar rolls up into.
# Axis 1 = fundamental trajectory (deteriorating <-> improving).
# Axis 2 = valuation & policy stretch (cheap <-> maxed-out).
FEATURE_PILLAR: dict[str, str] = {
    "bcicp": "A_growth", "bcicp_slope": "A_growth", "gdp_yoy": "A_growth",
    "unemp_3m_chg": "A_growth",
    "infl_gap": "B_inflation", "infl_momentum": "B_inflation",
    "current_account": "C_external", "niip": "C_external",
    "real_10y": "D_fiscal",   # market-priced long-end real yield (term-premium read,
                              # NOT the budget balance); reward/stress via the overlay
    "real_policy": "E_policy", "priced_path": "E_policy",
    # real_2y (carry) is PAIRWISE only — it lives in the carry grid, not the
    # per-currency composite (including it double-counted and re-imported the
    # post-ZIRP common component that lifts every leg's real-rate z-score).
    "reer": "G_valuation",
}
PILLAR_AXIS: dict[str, str] = {
    "A_growth": "axis1_fundamental", "B_inflation": "axis1_fundamental",
    "C_external": "axis1_fundamental", "D_fiscal": "axis1_fundamental",
    "E_policy": "axis2_stretch", "F_carry": "axis2_stretch",
    "G_valuation": "axis2_stretch",
}
# Short display names — single source of truth (used by the completeness guard, the
# narrative notes, and the dashboard's pillar labels/methodology).
PILLAR_DISPLAY: dict[str, str] = {
    "A_growth": "Growth", "B_inflation": "Inflation", "C_external": "External",
    "D_fiscal": "Real 10Y", "E_policy": "Policy", "F_carry": "Carry",
    "G_valuation": "Valuation",
}
# Sign so a POSITIVE contribution points to the axis's positive pole (Axis 1: improving/
# supportive; Axis 2: more stretched/maxed). Several are genuine design choices — see
# README "scoring signs". Tunable.
FEATURE_SIGN: dict[str, int] = {
    "bcicp": +1, "bcicp_slope": +1, "gdp_yoy": +1, "unemp_3m_chg": -1,
    "infl_gap": +1, "infl_momentum": +1,   # above-target/accelerating = hawkish support (debatable)
    "current_account": +1, "niip": +1,
    "real_10y": +1,                        # higher real 10y = support (vs fiscal-stress reading; debatable)
    "real_policy": +1, "priced_path": +1,  # tighter / more hikes priced = less room = more maxed
    "real_2y": +1,                         # higher real carry = more stretched/attractive
    "reer": +1,                            # richer REER = more expensive
}
# Within-pillar feature weights (gdp low-weight per spec); pillar weights within axis.
# All tunable via user sliders later.
FEATURE_WEIGHT: dict[str, float] = {
    "bcicp": 1.0, "bcicp_slope": 1.0, "gdp_yoy": 0.5, "unemp_3m_chg": 1.0,
    "infl_gap": 1.0, "infl_momentum": 1.0,
    "current_account": 1.0, "niip": 1.0,
    "real_10y": 1.0, "real_policy": 1.0, "priced_path": 1.0,
    "real_2y": 1.0, "reer": 1.0,
}
PILLAR_WEIGHT: dict[str, float] = {
    "A_growth": 1.0, "B_inflation": 1.0, "C_external": 1.0, "D_fiscal": 1.0,
    # Axis 2: valuation (REER) leads 2:1 over policy. Policy's own-history z-score
    # misreads early-cycle currencies off an anomalous baseline as "stretched" (e.g.
    # JPY, with the BoJ hiking off NIRP), so REER — the direct cheap/expensive read —
    # carries the axis, with policy as a modifier.
    "E_policy": 1.0, "F_carry": 1.0, "G_valuation": 2.0,
}

# FRED series per (currency, metric) — *** all verified live against actual
# observations via scripts/verify_fred.py (2026-07-01) ***. FRED mirrors of OECD
# families froze at different times (CLI/CPI/harmonised-unemployment/immediate-
# rates largely 2024–25; GDP-growth still updates), so IDs were chosen empirically
# by real freshness, not by search metadata (which lags actual data). metric keys:
#   gdp        Real GDP level (quarterly) — YoY computed in transform
#   unemp      Unemployment rate (monthly; CH/NZ quarterly, held flat)
#   cpi_index  Headline CPI index (monthly) — YoY computed in transform. US/EUR/CHF
#              only; GBP/CAD/AUD/NZD come as YoY from OECD, JPY from e-Stat.
#   policy     Policy rate (US/EUR actual; others being moved to BIS CBPOL)
# Gaps that FRED can no longer serve are tracked in FRED_GAPS below.
FRED_SERIES: dict[str, dict[str, str]] = {
    "USD": {"unemp": "UNRATE", "cpi_index": "CPIAUCSL"},
    "EUR": {"gdp": "CLVMNACSCAB1GQEA19", "cpi_index": "CP0000EZ19M086NEST"},
    "JPY": {},
    "GBP": {},
    "CHF": {"cpi_index": "CP0000CHM086NEST"},
    "CAD": {},
    "AUD": {},
    "NZD": {},
}

# Inputs FRED can no longer serve, now closed by the OECD-SDMX / Eurostat / e-Stat
# adapters. Pillar A no longer uses FRED at all (BCICP is OECD for all 8).
#   cpi  GBP/CAD/AUD/NZD -> OECD headline CPI YoY (cte.adapters.oecd)
#   cpi  JPY             -> Japan e-Stat headline CPI (cte.adapters.estat)
#   unemp EUR            -> Eurostat une_rt_m (cte.adapters.eurostat)
FRED_GAPS: dict[str, list[str]] = {
    "cpi":   ["GBP", "JPY", "CAD", "AUD", "NZD"],
    "unemp": ["EUR"],
}
MACRO_GAPS_REMAINING: dict[str, list[str]] = {}  # all macro inputs now sourced

# US TIPS constant-maturity real yields. Only 10Y is a clean series (there is no
# DFII2 — the shortest TIPS CM is 5Y). US 2Y real is derived in the transform layer
# as 2Y nominal − core CPI, the same construction used for the other legs.
FRED_US_REAL: dict[str, str] = {"real_10y": "DFII10"}

HTTP_UA = "Mozilla/5.0 (Currency Tension Engine; macro research)"
HTTP_TIMEOUT = 45
