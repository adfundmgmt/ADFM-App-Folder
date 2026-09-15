from __future__ import annotations

import unittest
from pathlib import Path

from cte.adapters.oecd import CPI_CCYS


class CurrencyTensionWorkflowTests(unittest.TestCase):
    def test_workflow_builds_snapshot_locally(self) -> None:
        workflow = Path(".github/workflows/sync_currency_tension_snapshot.yml").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("smileys21", workflow)
        self.assertNotIn("raw.githubusercontent.com", workflow)
        self.assertIn("python -m scripts.backfill --daily", workflow)
        self.assertIn("python -m scripts.backfill_history --if-missing", workflow)
        self.assertIn("python -m cte.scoring.engine", workflow)
        self.assertIn(
            "python scripts/validate_currency_snapshot.py data/cache", workflow
        )
        self.assertIn("FRED_API_KEY", workflow)
        self.assertNotIn("ESTAT_APP_ID", workflow)

    def test_japan_cpi_uses_the_secretless_oecd_headline_series(self) -> None:
        self.assertIn("JPY", CPI_CCYS)


if __name__ == "__main__":
    unittest.main()
