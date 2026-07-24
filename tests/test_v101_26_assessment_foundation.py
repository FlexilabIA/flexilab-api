from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


class FlexiLabV10126AssessmentFoundationTests(unittest.TestCase):
    def test_pose_runtime_is_thread_safe_and_recoverable(self) -> None:
        source = text("app.py")
        self.assertIn("POSE_MODEL_INFERENCE_LOCK = threading.RLock()", source)
        self.assertIn("with POSE_MODEL_INFERENCE_LOCK", source)
        self.assertIn("POSE_MODEL_RELOAD_COUNT", source)
        self.assertIn("known_fused_conv_error", source)
        self.assertIn("imgsz=POSE_INFERENCE_IMGSZ", source)

    def test_capture_metadata_is_accepted_without_schema_change(self) -> None:
        source = text("app.py")
        self.assertGreaterEqual(source.count("capture_metadata_json: str = Form(None)"), 2)
        self.assertIn("_flexilab_capture_metadata", source)
        self.assertIn('result["metrics"]["capture_metadata"]', source)
        self.assertIn("_split_job_intake_and_capture_metadata", source)

    def test_image_diagnostics_are_non_blocking(self) -> None:
        source = text("app.py")
        self.assertIn("decode_and_normalize_analysis_image", source)
        self.assertIn("brightness_mean", source)
        self.assertIn("blur_laplacian_variance", source)
        self.assertIn('result["metrics"]["image_quality_diagnostics"]', source)

    def test_diagnostic_retention_defaults_to_off(self) -> None:
        source = text("app.py")
        env = text("env.example")
        self.assertIn('FLEXILAB_DIAGNOSTIC_RETENTION_HOURS", "0"', source)
        self.assertIn("FLEXILAB_DIAGNOSTIC_RETENTION_HOURS=0", env)
        self.assertIn("keep_diagnostic_image", source)

    def test_health_exposes_reproducibility_information(self) -> None:
        source = text("app.py")
        self.assertIn("RUNTIME_PACKAGE_VERSIONS", source)
        self.assertIn('"pose_inference_imgsz"', source)
        self.assertIn('"analysis_max_edge"', source)
        self.assertIn('"pose_model_reload_count"', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
