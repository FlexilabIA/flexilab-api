from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aslr_engine import make_aslr_thresholds  # noqa: E402


class ASLRThresholdPolicyTests(unittest.TestCase):
    def test_boundary_policy(self):
        self.assertEqual(make_aslr_thresholds(59.9)["rating"], "red")
        self.assertEqual(make_aslr_thresholds(60.0)["rating"], "yellow")
        self.assertEqual(make_aslr_thresholds(75.0)["rating"], "yellow")
        self.assertEqual(make_aslr_thresholds(75.1)["rating"], "green")

    def test_band_values_and_equal_visual_contract(self):
        threshold = make_aslr_thresholds(52.6)
        self.assertEqual(
            [(band["min"], band["max"]) for band in threshold["bands"]],
            [(0, 60), (60, 75), (75, 90)],
        )
        self.assertEqual(threshold["visual_band_layout"], "equal_thirds")
        self.assertEqual(threshold["boundary_policy"]["green"], ">75")

    def test_report_reclassifies_historical_aslr_rows(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("ASLR thresholds were recalibrated in V101.28.4", app)
        self.assertIn("make_aslr_thresholds(aslr_value)", app)


if __name__ == "__main__":
    unittest.main(verbosity=2)
