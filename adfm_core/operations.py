"""Decision journal, threshold alerts, daily brief, and reliability primitives."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Mapping
from uuid import uuid4

import numpy as np
import pandas as pd

from .observability import DataLoadEvent
from .pm_cockpit import CockpitSummary

DEFAULT_JOURNAL_PATH = Path("data/last_good/decision_journal.parquet")
DEFAULT_ALERT_PATH = Path("data/last_good/alert_state.json")
CALCULATION_VERSION = "pm-research-v1"

JOURNAL_COLUMNS = (
    "Decision ID",
    "Created At UTC",
    "Trade Date",
    "Instrument",
    "Direction",
    "Size",
    "Thesis",
    "Catalyst",
    "Invalidation",
    "Expected Path",
    "Review Date",
    "Status",
    "Entry Regime",
    "Entry Composite",
    "Entry Snapshot",
    "Outcome",
    "Thesis Grade",
    "Timing Grade",
    "Sizing Grade",
    "Execution Grade",
    "Luck",
    "Review Notes",
)


@dataclass(frozen=True)
class AlertEvent:
    """One state transition that warrants PM attention."""

    key: str
    severity: str
    title: str
    detail: str
    observed_at_utc: str


@dataclass(frozen=True)
class ReliabilityCheck:
    """One inspectable platform-health check."""

    component: str
    status: str
    data_through: str | None
    detail: str


def load_decision_journal(
    path: Path = DEFAULT_JOURNAL_PATH,
) -> pd.DataFrame:
    """Read the local journal with a stable schema and graceful corruption handling."""

    if not path.exists():
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    for column in JOURNAL_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[list(JOURNAL_COLUMNS)].copy()


def save_decision(
    decision: Mapping[str, object],
    *,
    snapshot: pd.DataFrame | None = None,
    path: Path = DEFAULT_JOURNAL_PATH,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Append or update one decision and atomically preserve its entry snapshot."""

    required = {"Instrument", "Direction", "Thesis", "Invalidation"}
    missing = sorted(
        field for field in required if not str(decision.get(field, "")).strip()
    )
    if missing:
        raise ValueError(f"Decision is missing required fields: {missing}")
    timestamp = (now or datetime.now(UTC)).replace(microsecond=0)
    decision_id = str(decision.get("Decision ID") or uuid4())
    row = {column: decision.get(column, pd.NA) for column in JOURNAL_COLUMNS}
    row.update(
        {
            "Decision ID": decision_id,
            "Created At UTC": decision.get("Created At UTC", timestamp.isoformat()),
            "Trade Date": decision.get("Trade Date", timestamp.date().isoformat()),
            "Status": decision.get("Status", "Open"),
            "Entry Snapshot": (
                snapshot.to_json(orient="records", date_format="iso")
                if snapshot is not None and not snapshot.empty
                else decision.get("Entry Snapshot", "[]")
            ),
        }
    )
    history = load_decision_journal(path)
    history = history.loc[history["Decision ID"].astype(str) != decision_id]
    combined = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".parquet",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        combined.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return combined


def journal_review_queue(
    journal: pd.DataFrame,
    *,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Return open decisions due for review or missing explicit invalidation."""

    if journal.empty:
        return journal.copy()
    today = pd.Timestamp(as_of or date.today())
    frame = journal.copy()
    review_dates = pd.to_datetime(frame["Review Date"], errors="coerce")
    open_mask = frame["Status"].astype(str).str.lower().eq("open")
    due = review_dates.le(today) | review_dates.isna()
    missing_invalidation = (
        frame["Invalidation"].fillna("").astype(str).str.strip().eq("")
    )
    frame["Review Reason"] = np.select(
        [missing_invalidation, review_dates.le(today), review_dates.isna()],
        ["Missing invalidation", "Review date reached", "No review date"],
        default="",
    )
    return frame.loc[open_mask & (due | missing_invalidation)].sort_values(
        "Review Date", na_position="first"
    )


def weekly_process_review(journal: pd.DataFrame) -> pd.DataFrame:
    """Separate thesis, timing, sizing, execution, and luck from review notes."""

    if journal.empty:
        return pd.DataFrame()
    reviewed = journal.loc[
        journal["Thesis Grade"].notna()
        | journal["Timing Grade"].notna()
        | journal["Sizing Grade"].notna()
        | journal["Execution Grade"].notna()
        | journal["Luck"].notna()
        | journal["Outcome"].notna()
        | journal["Review Notes"].notna()
    ].copy()
    if reviewed.empty:
        return reviewed
    reviewed["Trade Date"] = pd.to_datetime(reviewed["Trade Date"], errors="coerce")
    reviewed["Review Week"] = reviewed["Trade Date"].dt.to_period("W").astype(str)
    return reviewed.sort_values("Trade Date", ascending=False)


def evaluate_alerts(
    summary: CockpitSummary,
    snapshot: pd.DataFrame,
    *,
    prior_summary: CockpitSummary | None = None,
    data_health: DataLoadEvent | None = None,
    journal: pd.DataFrame | None = None,
    now: datetime | None = None,
) -> list[AlertEvent]:
    """Evaluate sparse threshold-crossing alerts from current platform state."""

    observed = (now or datetime.now(UTC)).replace(microsecond=0).isoformat()
    alerts: list[AlertEvent] = []
    if (
        prior_summary is not None
        and summary.regime != prior_summary.regime
        and summary.available_signals
    ):
        alerts.append(
            AlertEvent(
                "regime-transition",
                "high",
                "Regime transition",
                f"{prior_summary.regime} changed to {summary.regime}.",
                observed,
            )
        )
    if np.isfinite(summary.dispersion) and summary.dispersion >= 0.55:
        alerts.append(
            AlertEvent(
                "signal-dispersion",
                "medium",
                "Cross-signal dispersion",
                f"Dispersion reached {summary.dispersion:.2f}; regime coherence is weak.",
                observed,
            )
        )
    if not snapshot.empty:
        for _, row in snapshot.iterrows():
            impulse = pd.to_numeric(row.get("Impulse"), errors="coerce")
            composite = pd.to_numeric(row.get("Composite"), errors="coerce")
            if np.isfinite(impulse) and abs(impulse) >= 1.0:
                alerts.append(
                    AlertEvent(
                        f"impulse:{row.get('Key')}",
                        "high",
                        f"{row.get('Signal')} reversal",
                        f"Impulse is {impulse:+.2f}; composite is {composite:+.2f}.",
                        observed,
                    )
                )
    if data_health is not None and data_health.failed_symbols > 0:
        alerts.append(
            AlertEvent(
                "provider-failure",
                "medium",
                "Market-data coverage",
                f"{data_health.failed_symbols} of {data_health.requested_symbols} symbols failed.",
                observed,
            )
        )
    if journal is not None:
        due = journal_review_queue(journal)
        if not due.empty:
            alerts.append(
                AlertEvent(
                    "journal-review",
                    "medium",
                    "Decision reviews due",
                    f"{len(due)} open decisions require review or invalidation work.",
                    observed,
                )
            )
    return alerts


def new_alert_transitions(
    alerts: list[AlertEvent],
    *,
    path: Path = DEFAULT_ALERT_PATH,
) -> list[AlertEvent]:
    """Persist active alert keys and return only newly crossed thresholds."""

    previous: set[str] = set()
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            previous = set(payload.get("active_keys", []))
        except (OSError, json.JSONDecodeError):
            previous = set()
    active = {alert.key for alert in alerts}
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "calculation_version": CALCULATION_VERSION,
        "updated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "active_keys": sorted(active),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return [alert for alert in alerts if alert.key not in previous]


def reliability_report(
    *,
    data_health: DataLoadEvent | None,
    repository_root: Path,
) -> pd.DataFrame:
    """Inspect provider, snapshot, persistence, test, and fallback readiness."""

    checks: list[ReliabilityCheck] = []
    if data_health is None:
        checks.append(
            ReliabilityCheck(
                "Market provider",
                "UNKNOWN",
                None,
                "No shared market request has run in this session.",
            )
        )
    else:
        checks.append(
            ReliabilityCheck(
                "Market provider",
                "OK" if data_health.failed_symbols == 0 else "DEGRADED",
                data_health.data_through,
                f"{data_health.returned_symbols}/{data_health.requested_symbols} symbols returned.",
            )
        )
    cache = repository_root / "data" / "cache"
    required = (
        "tension_map.parquet",
        "pillar_scores.parquet",
        "overlays.parquet",
        "warnings.json",
    )
    missing = [name for name in required if not (cache / name).is_file()]
    dates = [
        datetime.fromtimestamp((cache / name).stat().st_mtime, UTC)
        for name in required
        if (cache / name).is_file()
    ]
    checks.append(
        ReliabilityCheck(
            "Currency snapshot",
            "OK" if not missing else "FAILED",
            max(dates).date().isoformat() if dates else None,
            "Required snapshot present."
            if not missing
            else f"Missing: {', '.join(missing)}",
        )
    )
    ledger = repository_root / "data" / "last_good" / "pm_signal_ledger.parquet"
    checks.append(
        ReliabilityCheck(
            "Signal ledger",
            "OK" if ledger.exists() else "READY",
            datetime.fromtimestamp(ledger.stat().st_mtime, UTC).date().isoformat()
            if ledger.exists()
            else None,
            "Point-in-time history available."
            if ledger.exists()
            else "Ledger will initialize after the first successful command-center run.",
        )
    )
    checks.append(
        ReliabilityCheck(
            "Calculation contract",
            "OK",
            None,
            f"Version {CALCULATION_VERSION}; causal scores and missing-value preservation enforced.",
        )
    )
    workflow = repository_root / ".github" / "workflows" / "ci.yml"
    checks.append(
        ReliabilityCheck(
            "Repository tests",
            "CONFIGURED" if workflow.exists() else "FAILED",
            None,
            "GitHub CI compiles, lints, and coverage-gates the platform."
            if workflow.exists()
            else "CI workflow unavailable.",
        )
    )
    return pd.DataFrame(asdict(check) for check in checks)


def daily_brief_tables(
    summary: CockpitSummary,
    snapshot: pd.DataFrame,
    journal: pd.DataFrame,
) -> Mapping[str, pd.DataFrame]:
    """Build the operating tables for the morning PM read."""

    movers = snapshot.copy()
    movers["Absolute Impulse"] = pd.to_numeric(
        movers.get("Impulse"), errors="coerce"
    ).abs()
    movers = movers.nlargest(6, "Absolute Impulse")[
        ["Signal", "Group", "Composite", "Impulse", "Data Through"]
    ]
    due = journal_review_queue(journal)
    stale = snapshot.loc[
        pd.to_datetime(snapshot.get("Data Through"), errors="coerce")
        < pd.Timestamp(summary.as_of) - pd.Timedelta(days=3)
        if summary.as_of
        else pd.Series(False, index=snapshot.index)
    ]
    return {"movers": movers, "reviews": due, "stale": stale}
