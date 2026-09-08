"""FRED source contracts: natural units, release cadence, and cache policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FredPolicy:
    units: str = "provider units"
    frequency: str = "unknown"
    max_age_days: int = 14
    minimum: float | None = None
    maximum: float | None = None
    publish_snapshot: bool = False


POLICIES = {
    **{s: FredPolicy("Percent", "Daily", 7, -10, 100, True) for s in (
        "DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30", "DFII5", "DFII10",
        "T5YIE", "T10YIE", "SOFR", "IORB", "EFFR",
    )},
    **{s: FredPolicy("Millions of U.S. Dollars", "Weekly", 14, 0, None, True)
       for s in ("WALCL", "WRESBAL", "WTREGEN")},
    "RRPONTSYD": FredPolicy("Billions of U.S. Dollars", "Daily", 7, 0, None, True),
    # Daily observations are distributed on a weekly publication schedule.
    "DTWEXBGS": FredPolicy("Index", "Daily", 14, 0, None, True),
    "NFCI": FredPolicy("Index", "Weekly", 14, -100, 100, True),
    "STLFSI4": FredPolicy("Index", "Weekly", 14, -100, 100, True),
    "FEDFUNDS": FredPolicy("Percent", "Monthly", 65, -10, 100, True),
    "UNRATE": FredPolicy("Percent", "Monthly", 65, 0, 100, True),
    # Monthly FRED dates label period starts; the next release can arrive more
    # than two months after the last period-start label, especially on holidays.
    "CPIAUCSL": FredPolicy("Index", "Monthly", 80, 0, None, True),
    "PCEPILFE": FredPolicy("Index", "Monthly", 75, 0, None, True),
    "PAYEMS": FredPolicy("Thousands of Persons", "Monthly", 65, 0, None, True),
    "INDPRO": FredPolicy("Index", "Monthly", 80, 0, None, True),
    "ICSA": FredPolicy("Number", "Weekly", 14, 0, None, True),
    "USREC": FredPolicy("+1 or 0", "Monthly", 75, 0, 1, True),
    # Vendor observations remain in the application's local cache, not in the
    # public repository snapshot. Their history can be limited by the provider.
    **{s: FredPolicy("Percent", "Daily", 7, 0, 100) for s in (
        "BAMLH0A0HYM2", "BAMLC0A0CM", "BAMLC0A4CBBB",
    )},
}


def policy_for(symbol: str) -> FredPolicy:
    if symbol.startswith("IRLTLT01"):
        return FredPolicy("Percent", "Monthly", 100, -10, 100)
    return POLICIES.get(symbol, FredPolicy())


SNAPSHOT_SYMBOLS = tuple(s for s, policy in POLICIES.items() if policy.publish_snapshot)
