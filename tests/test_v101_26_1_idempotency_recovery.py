from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source() -> str:
    return (ROOT / "app.py").read_text(encoding="utf-8")


class FlexiLabV101261IdempotencyRecoveryTests(unittest.TestCase):
    def test_existing_screening_is_recovered(self) -> None:
        text = source()
        self.assertIn("def _find_existing_screening", text)
        self.assertIn("result_json = _analysis_result_from_screening", text)
        self.assertIn('"recovered": True', text)

    def test_duplicate_insert_is_reconciled(self) -> None:
        text = source()
        self.assertIn("def _is_duplicate_screening_error", text)
        self.assertIn("screenings_idempotency_unique_idx", text)
        self.assertGreaterEqual(text.count("_is_duplicate_screening_error(exc)"), 3)

    def test_session_score_cache_failure_does_not_fail_analysis(self) -> None:
        text = source()
        self.assertIn("def _update_session_score_best_effort", text)
        self.assertIn("authoritative result remains in", text)
        self.assertIn("analysis_session_score_update_failed", text)

    def test_raw_database_errors_are_not_returned_to_clients(self) -> None:
        text = source()
        self.assertIn("def _public_analysis_error", text)
        self.assertIn("analysis_job_failed", text)
        self.assertIn('"error_message": _public_analysis_error(exc)', text)

    def test_idempotency_recovery_remains_present_in_later_versions(self) -> None:
        text = source()
        self.assertIn("idempotency_key", text)
        self.assertIn("_complete_analysis_job", text)
        self.assertIn("existing_screening = _find_existing_screening", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
