from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List

import pandas as pd

from adfm_core import catalyst_calendar_exact_page as exact
from adfm_core import catalyst_calendar_page as base

OVERRIDDEN_EVENTS = {
    "Payrolls / Employment Situation",
    "CPI Inflation Window",
    "PPI Inflation Window",
    "PCE Inflation Window",
    "JOLTS Job Openings Window",
    "ISM Manufacturing Window",
    "ISM Services Window",
    "Retail Sales Window",
    "FOMC Decision Window",
    "GDP Release Window",
    "Quarterly Treasury Refunding Window",
    "Initial Jobless Claims",
    "Earnings Season Ramp",
}


def _event(d: str, name: str, event_type: str, why: str, source: str) -> Dict[str, object]:
    return {
        "Date": pd.Timestamp(d).date(),
        "Event": name,
        "Type": event_type,
        "Region": "U.S.",
        "Why It Matters": why,
        "Precision": "Official",
        "Source": source,
    }


OFFICIAL_EVENTS: List[Dict[str, object]] = [
    _event("2026-09-01", "JOLTS Job Openings", "Labor", "Labor demand, wage pressure, and Fed pricing.", "BLS"),
    _event("2026-09-01", "ISM Manufacturing", "Growth", "Cyclical growth, rates, commodities, and small-cap beta.", "ISM"),
    _event("2026-09-03", "ISM Services", "Growth", "Services inflation, labor demand, and broad growth.", "ISM"),
    _event("2026-09-03", "Initial Jobless Claims", "Labor", "High-frequency labor deterioration or resilience ahead of payrolls.", "DOL"),
    _event("2026-09-04", "Payrolls / Employment Situation", "Labor", "Growth, wages, Fed pricing, USD, and equity beta.", "BLS"),
    _event("2026-09-10", "PPI Inflation", "Inflation", "Pipeline inflation, margins, and input costs.", "BLS"),
    _event("2026-09-10", "Initial Jobless Claims", "Labor", "High-frequency labor deterioration or resilience ahead of payrolls.", "DOL"),
    _event("2026-09-11", "CPI Inflation", "Inflation", "Rates, USD, duration, Nasdaq duration factor, and real income.", "BLS"),
    _event("2026-09-16", "Retail Sales", "Growth", "Consumption impulse, cyclicals, rates, and the growth narrative.", "Census"),
    _event("2026-09-16", "FOMC Decision", "Fed", "Policy path, financial conditions, USD, curve, and growth duration.", "Federal Reserve"),
    _event("2026-09-17", "Initial Jobless Claims", "Labor", "High-frequency labor deterioration or resilience ahead of payrolls.", "DOL"),
    _event("2026-09-24", "Initial Jobless Claims", "Labor", "High-frequency labor deterioration or resilience ahead of payrolls.", "DOL"),
    _event("2026-09-29", "JOLTS Job Openings", "Labor", "Labor demand, wage pressure, and Fed pricing.", "BLS"),
    _event("2026-09-30", "PCE Inflation", "Inflation", "Fed-preferred inflation gauge, real rates, USD, and duration.", "BEA"),
    _event("2026-09-30", "GDP — Q2 Third Estimate", "Growth", "Growth regime, real-rate pricing, cyclicals, USD, and earnings expectations.", "BEA"),
    _event("2026-10-01", "ISM Manufacturing", "Growth", "Cyclical growth, rates, commodities, and small-cap beta.", "ISM"),
    _event("2026-10-01", "Initial Jobless Claims", "Labor", "High-frequency labor deterioration or resilience ahead of payrolls.", "DOL"),
    _event("2026-10-02", "Payrolls / Employment Situation", "Labor", "Growth, wages, Fed pricing, USD, and equity beta.", "BLS"),
    _event("2026-10-05", "ISM Services", "Growth", "Services inflation, labor demand, and broad growth.", "ISM"),
    _event("2026-10-08", "Initial Jobless Claims", "Labor", "High-frequency labor deterioration or resilience ahead of payrolls.", "DOL"),
    _event("2026-10-14", "CPI Inflation", "Inflation", "Rates, USD, duration, Nasdaq duration factor, and real income.", "BLS"),
    _event("2026-10-15", "PPI Inflation", "Inflation", "Pipeline inflation, margins, and input costs.", "BLS"),
    _event("2026-10-15", "Retail Sales", "Growth", "Consumption impulse, cyclicals, rates, and the growth narrative.", "Census"),
    _event("2026-10-15", "Initial Jobless Claims", "Labor", "High-frequency labor deterioration or resilience ahead of payrolls.", "DOL"),
    _event("2026-10-22", "Initial Jobless Claims", "Labor", "High-frequency labor deterioration or resilience ahead of payrolls.", "DOL"),
    _event("2026-10-28", "FOMC Decision", "Fed", "Policy path, financial conditions, USD, curve, and growth duration.", "Federal Reserve"),
    _event("2026-10-29", "PCE Inflation", "Inflation", "Fed-preferred inflation gauge, real rates, USD, and duration.", "BEA"),
    _event("2026-10-29", "GDP — Q3 Advance Estimate", "Growth", "Growth regime, real-rate pricing, cyclicals, USD, and earnings expectations.", "BEA"),
    _event("2026-10-29", "Initial Jobless Claims", "Labor", "High-frequency labor deterioration or resilience ahead of payrolls.", "DOL"),
    _event("2026-11-02", "ISM Manufacturing", "Growth", "Cyclical growth, rates, commodities, and small-cap beta.", "ISM"),
    _event("2026-11-03", "JOLTS Job Openings", "Labor", "Labor demand, wage pressure, and Fed pricing.", "BLS"),
    _event("2026-11-03", "U.S. Midterm Election", "Custom", "Fiscal expectations, regulation, sector leadership, rates, USD, and index beta.", "FEC"),
    _event("2026-11-04", "Quarterly Treasury Refunding", "Treasury", "Coupon supply, term premium, curve pressure, and duration risk.", "U.S. Treasury"),
    _event("2026-11-04", "ISM Services", "Growth", "Services inflation, labor demand, and broad growth.", "ISM"),
    _event("2026-11-05", "Initial Jobless Claims", "Labor", "High-frequency labor deterioration or resilience ahead of payrolls.", "DOL"),
    _event("2026-11-06", "Payrolls / Employment Situation", "Labor", "Growth, wages, Fed pricing, USD, and equity beta.", "BLS"),
    _event("2026-11-10", "CPI Inflation", "Inflation", "Rates, USD, duration, Nasdaq duration factor, and real income.", "BLS"),
    _event("2026-11-12", "Initial Jobless Claims", "Labor", "High-frequency labor deterioration or resilience ahead of payrolls.", "DOL"),
    _event("2026-11-13", "PPI Inflation", "Inflation", "Pipeline inflation, margins, and input costs.", "BLS"),
    _event("2026-11-17", "Retail Sales", "Growth", "Consumption impulse, cyclicals, rates, and the growth narrative.", "Census"),
    _event("2026-11-19", "Initial Jobless Claims", "Labor", "High-frequency labor deterioration or resilience ahead of payrolls.", "DOL"),
    _event("2026-11-25", "Initial Jobless Claims", "Labor", "Thanksgiving-adjusted weekly claims release; high-frequency labor signal ahead of payrolls.", "DOL"),
    _event("2026-11-25", "PCE Inflation", "Inflation", "Fed-preferred inflation gauge, real rates, USD, and duration.", "BEA"),
    _event("2026-11-25", "GDP — Q3 Second Estimate", "Growth", "Growth regime, real-rate pricing, cyclicals, USD, and earnings expectations.", "BEA"),
    _event("2026-12-01", "JOLTS Job Openings", "Labor", "Labor demand, wage pressure, and Fed pricing.", "BLS"),
    _event("2026-12-01", "ISM Manufacturing", "Growth", "Cyclical growth, rates, commodities, and small-cap beta.", "ISM"),
    _event("2026-12-03", "ISM Services", "Growth", "Services inflation, labor demand, and broad growth.", "ISM"),
    _event("2026-12-03", "Initial Jobless Claims", "Labor", "High-frequency labor deterioration or resilience ahead of payrolls.", "DOL"),
    _event("2026-12-04", "Payrolls / Employment Situation", "Labor", "Growth, wages, Fed pricing, USD, and equity beta.", "BLS"),
    _event("2026-12-09", "FOMC Decision", "Fed", "Policy path, financial conditions, USD, curve, and growth duration.", "Federal Reserve"),
    _event("2026-12-10", "CPI Inflation", "Inflation", "Rates, USD, duration, Nasdaq duration factor, and real income.", "BLS"),
    _event("2026-12-10", "Initial Jobless Claims", "Labor", "High-frequency labor deterioration or resilience ahead of payrolls.", "DOL"),
    _event("2026-12-15", "PPI Inflation", "Inflation", "Pipeline inflation, margins, and input costs.", "BLS"),
    _event("2026-12-16", "Retail Sales", "Growth", "Consumption impulse, cyclicals, rates, and the growth narrative.", "Census"),
    _event("2026-12-17", "Initial Jobless Claims", "Labor", "High-frequency labor deterioration or resilience ahead of payrolls.", "DOL"),
    _event("2026-12-23", "PCE Inflation", "Inflation", "Fed-preferred inflation gauge, real rates, USD, and duration.", "BEA"),
    _event("2026-12-23", "GDP — Q3 Third Estimate", "Growth", "Growth regime, real-rate pricing, cyclicals, USD, and earnings expectations.", "BEA"),
    _event("2026-12-24", "Initial Jobless Claims", "Labor", "High-frequency labor deterioration or resilience ahead of payrolls.", "DOL"),
    _event("2026-12-31", "Initial Jobless Claims", "Labor", "High-frequency labor deterioration or resilience ahead of payrolls.", "DOL"),
    _event("2027-01-27", "FOMC Decision", "Fed", "Policy path, financial conditions, USD, curve, and growth duration.", "Federal Reserve"),
]


LABEL_REPLACEMENTS = {
    "CPI Inflation Window": "CPI Inflation",
    "PPI Inflation Window": "PPI Inflation",
    "PCE Inflation Window": "PCE Inflation",
    "JOLTS Job Openings Window": "JOLTS Job Openings",
    "ISM Manufacturing Window": "ISM Manufacturing",
    "ISM Services Window": "ISM Services",
    "Retail Sales Window": "Retail Sales",
    "FOMC Decision Window": "FOMC Decision",
    "GDP Release Window": "GDP Release",
    "Quarterly Treasury Refunding Window": "Quarterly Treasury Refunding",
}


def _official_dated_calendar(start: date, horizon_days: int, include_fed: bool) -> pd.DataFrame:
    end = start + timedelta(days=horizon_days)

    # Keep only genuinely deterministic rule-based market-calendar events such
    # as monthly options expiration and quarter-end. Scheduled macro releases
    # are supplied exclusively by OFFICIAL_EVENTS so the app never manufactures
    # an "exact" date from a heuristic when an agency has not published one.
    recurring = base._build_rule_calendar(start, horizon_days, include_fed=False)
    if not recurring.empty:
        recurring = recurring.reset_index(drop=True)
        original_names = recurring["Event"].copy()
        recurring["Source"] = "Calendar rule"
        recurring["Event"] = recurring["Event"].replace(LABEL_REPLACEMENTS)
        recurring = recurring[~original_names.isin(OVERRIDDEN_EVENTS)].copy()

    official = pd.DataFrame(OFFICIAL_EVENTS)
    official = official[(official["Date"] >= start) & (official["Date"] <= end)].copy()
    if not include_fed and not official.empty:
        official = official[official["Type"] != "Fed"].copy()

    frames = [df for df in (recurring, official) if not df.empty]
    if not frames:
        return pd.DataFrame()

    calendar = pd.concat(frames, ignore_index=True, sort=False)
    calendar["Type"] = calendar["Type"].map(base._normalize_event_type)
    return calendar.sort_values(["Date", "Event"]).reset_index(drop=True)


# The exact-date renderer calls this module-level function. Replace it with the
# official-date-aware builder while preserving the existing scoring and layout.
exact._dated_calendar = _official_dated_calendar
render_catalyst_calendar = exact.render_catalyst_calendar
