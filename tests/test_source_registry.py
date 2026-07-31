from __future__ import annotations

import unittest

from adfm_core.source_registry import (
    SOURCE_SYSTEMS,
    source_by_key,
    source_capability_table,
)


class SourceRegistryTests(unittest.TestCase):
    def test_required_institutions_are_registered(self) -> None:
        self.assertEqual(
            {source.key for source in SOURCE_SYSTEMS},
            {
                "treasury",
                "bls",
                "bea",
                "fed",
                "ecb",
                "boj",
                "boe",
                "cftc",
                "eia",
                "sec",
            },
        )
        self.assertEqual(source_by_key("sec").authentication, "User-Agent required")
        self.assertIsNone(source_by_key("unknown"))

    def test_capability_table_has_auditable_metadata(self) -> None:
        frame = source_capability_table()

        self.assertEqual(len(frame), len(SOURCE_SYSTEMS))
        self.assertTrue(frame["endpoint"].str.startswith("https://").all())
        self.assertTrue(frame["revision_policy"].str.len().gt(20).all())


if __name__ == "__main__":
    unittest.main()
