from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from adfm_core.operations import (
    evaluate_alerts,
    journal_review_queue,
    load_decision_journal,
    new_alert_transitions,
    reliability_report,
    save_decision,
)
from adfm_core.pm_cockpit import CockpitSummary


def summary(regime: str, dispersion: float = 0.20) -> CockpitSummary:
    return CockpitSummary(
        regime=regime,
        composite=0.20,
        confidence=0.90,
        breadth=0.60,
        dispersion=dispersion,
        impulse=0.10,
        as_of="2026-07-30",
        available_signals=10,
        total_signals=12,
    )


class OperationsTests(unittest.TestCase):
    def test_decision_journal_preserves_entry_snapshot_and_review_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.parquet"
            snapshot = pd.DataFrame([{"Key": "credit_risk", "Composite": -0.4}])
            journal = save_decision(
                {
                    "Instrument": "HYG",
                    "Direction": "Short",
                    "Thesis": "Credit deterioration",
                    "Invalidation": "HYG/LQD recovers its prior high",
                    "Review Date": "2026-07-29",
                },
                snapshot=snapshot,
                path=path,
                now=datetime(2026, 7, 30, tzinfo=timezone.utc),
            )

            reloaded = load_decision_journal(path)
            queue = journal_review_queue(reloaded, as_of=date(2026, 7, 30))

            self.assertEqual(len(journal), 1)
            self.assertIn("credit_risk", reloaded["Entry Snapshot"].iloc[0])
            self.assertEqual(queue["Review Reason"].iloc[0], "Review date reached")

    def test_alerts_emit_only_new_threshold_transitions(self) -> None:
        snapshot = pd.DataFrame(
            [
                {
                    "Key": "credit_risk",
                    "Signal": "Credit risk",
                    "Impulse": -1.2,
                    "Composite": -0.5,
                }
            ]
        )
        alerts = evaluate_alerts(
            summary("Defensive", dispersion=0.60),
            snapshot,
            prior_summary=summary("Constructive"),
            now=datetime(2026, 7, 30, tzinfo=timezone.utc),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alerts.json"
            first = new_alert_transitions(alerts, path=path)
            second = new_alert_transitions(alerts, path=path)

            self.assertEqual(len(first), 3)
            self.assertEqual(second, [])

    def test_reliability_report_is_explicit_when_session_has_not_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".github" / "workflows").mkdir(parents=True)
            (root / ".github" / "workflows" / "ci.yml").write_text(
                "name: CI", encoding="utf-8"
            )

            report = reliability_report(data_health=None, repository_root=root)

            self.assertEqual(
                report.loc[report["component"] == "Market provider", "status"].iloc[0],
                "UNKNOWN",
            )
            self.assertIn("Repository tests", report["component"].tolist())


if __name__ == "__main__":
    unittest.main()
