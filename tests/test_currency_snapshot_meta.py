from __future__ import annotations

import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class CurrencySnapshotMetaTests(unittest.TestCase):
    def test_snapshot_generated_at_prefers_manifest_validation_time(self) -> None:
        spec = importlib.util.find_spec("cte.snapshot_meta")
        self.assertIsNotNone(spec, "cte.snapshot_meta must provide snapshot freshness metadata")
        if spec is None:
            return
        module = importlib.import_module("cte.snapshot_meta")

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "snapshot_manifest.json").write_text(
                json.dumps({"validated_at_utc": "2026-09-15T12:34:56+00:00"}),
                encoding="utf-8",
            )

            stamp = module.snapshot_generated_at(cache_dir)

            self.assertEqual(stamp, pd.Timestamp("2026-09-15T12:34:56+00:00"))

    def test_currency_page_uses_snapshot_manifest_freshness(self) -> None:
        page = Path("pages/6_Currency_Tension_Engine.py").read_text(encoding="utf-8")

        self.assertIn("from cte.snapshot_meta import snapshot_generated_at", page)
        self.assertIn("return snapshot_generated_at()", page)
        self.assertNotIn('CACHE_DIR / "commentary_meta.json"', page)


if __name__ == "__main__":
    unittest.main()
