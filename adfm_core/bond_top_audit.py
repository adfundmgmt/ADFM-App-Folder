"""Retrospective reference-episode audit, isolated from signal generation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bond_event_study import event_dates, monthly_history, signal_frame
from .global_macro import clean

# These windows identify the user's calibration episodes. Yield levels and
# exact peak dates are derived from the selected official series, not entered
# as fabricated observations. 1974-75 reports the higher peak in that window.
REFERENCE_TOPS = (
    ("Jan 1960", "1960-01-01", "1960-01-31"),
    ("Aug 1966", "1966-08-01", "1966-08-31"),
    ("May 1970", "1970-05-01", "1970-05-31"),
    ("1974–75", "1974-01-01", "1975-12-31"),
    ("Sep 1981", "1981-09-01", "1981-09-30"),
    ("Jun 1984", "1984-05-01", "1984-07-31"),
    ("Nov 1994", "1994-11-01", "1994-11-30"),
    ("Jan 2000", "2000-01-01", "2000-01-31"),
    ("Jun 2007", "2007-06-01", "2007-06-30"),
    ("Oct 2023", "2023-10-01", "2023-10-31"),
)


def event_spacing(frequency: str, profile: str) -> int:
    return 1 if profile == "Cycle Top" else 6 if frequency == "monthly" else 63


def audit_reference_tops(rates: pd.Series, frequency: str, profile: str) -> pd.DataFrame:
    """Match only actual plotted alerts after each observed reference peak.

    A capture is <=10 observed sessions after a daily peak or <=3 months after
    a monthly peak. Later alerts remain late; missing data never become hits.
    Monthly dates are period ends with unknown actual publication timing.
    """
    frame = signal_frame(rates, frequency, profile)
    history = monthly_history(rates) if frequency == "monthly" else clean(rates)
    dates = event_dates(frame, event_spacing(frequency, profile))
    rows = []
    for label, start, end in REFERENCE_TOPS:
        row = {"Episode": label, "Basis": "GS10 monthly average" if frequency == "monthly" else "DGS10 daily close",
               "Peak date": pd.NaT, "Peak yield (%)": np.nan, "Alert date": pd.NaT,
               "Alert yield (%)": np.nan, "Delay": "", "Yield fall at alert (bp)": np.nan,
               "3M after alert (bp)": np.nan, "Status": "History unavailable"}
        window = history.loc[start:end].dropna()
        if window.empty:
            rows.append(row)
            continue
        peak_date = window.idxmax()
        peak = float(window.loc[peak_date])
        row.update({"Peak date": peak_date, "Peak yield (%)": peak})
        pos = history.index.get_loc(peak_date)
        if frequency == "monthly":
            deadline = peak_date + pd.offsets.MonthEnd(3)
        else:
            deadline = history.index[min(pos + 10, len(history) - 1)]
        if history.index[-1] < deadline or (frequency == "daily" and pos + 10 >= len(history)):
            row["Status"] = "Incomplete follow-up"
        else:
            candidates = dates[(dates > peak_date) & (dates <= peak_date + pd.DateOffset(months=12))]
            row["Status"] = "Missed"
            if len(candidates):
                alert = candidates[0]
                value = float(history.loc[alert])
                delay = history.index.get_loc(alert) - pos
                row.update({"Alert date": alert, "Alert yield (%)": value,
                            "Delay": f"{delay} " + ("month" if delay == 1 else "months") if frequency == "monthly"
                                     else f"{delay} " + ("session" if delay == 1 else "sessions"),
                            "Yield fall at alert (bp)": (peak - value) * 100,
                            "Status": "Captured" if alert <= deadline else "Late"})
                alert_pos = history.index.get_loc(alert)
                steps = 3 if frequency == "monthly" else 63
                path = history.iloc[alert_pos:alert_pos + steps + 1]
                if len(path) == steps + 1 and path.notna().all():
                    row["3M after alert (bp)"] = (float(path.iloc[-1]) - value) * 100
        rows.append(row)
    return pd.DataFrame(rows)
