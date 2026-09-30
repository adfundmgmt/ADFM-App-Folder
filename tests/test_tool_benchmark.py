"""The page benchmark must retain failures and contain stalled providers."""
from __future__ import annotations

import unittest
from subprocess import CompletedProcess, TimeoutExpired
from unittest.mock import patch

from scripts.benchmark_tools import benchmark_page, parse_worker_output


class ToolBenchmarkTests(unittest.TestCase):
    def test_parsing_retains_runtime_failure_after_noisy_stdout(self):
        result = parse_worker_output('provider log\nADFM_BENCHMARK={"page":"x.py","status":"error","errors":["bad input"]}\n')
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["errors"], ["bad input"])

    @patch("scripts.benchmark_tools.subprocess.run")
    def test_stalled_page_is_reported_and_does_not_stop_other_pages(self, run):
        run.side_effect = TimeoutExpired(["python"], 5)
        result = benchmark_page("2_Global_Macro_Regime.py", 5)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["page"], "2_Global_Macro_Regime.py")

    @patch("scripts.benchmark_tools.subprocess.run")
    def test_worker_crash_is_not_reported_as_a_success(self, run):
        run.return_value = CompletedProcess([], 1, stdout="", stderr="import failed")
        result = benchmark_page("2_Global_Macro_Regime.py", 5)
        self.assertEqual(result["status"], "error")


if __name__ == "__main__":
    unittest.main()
