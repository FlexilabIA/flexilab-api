from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")
ENGINE_SOURCE = (ROOT / "aslr_engine.py").read_text(encoding="utf-8")
VISION_SOURCE = (ROOT / "vision_qa.py").read_text(encoding="utf-8")


class ASLRDedicatedYOLO11mTests(unittest.TestCase):
    def test_patch_and_model_contract(self):
        self.assertIn('"patch_version": "V101.28.4-aslr-thresholds-60-75"', APP_SOURCE)
        self.assertIn('"FLEXILAB_ASLR_POSE_MODEL", "yolo11m-pose.pt"', APP_SOURCE)
        self.assertIn('ASLR_POSE_MODEL_INFERENCE_LOCK = threading.RLock()', APP_SOURCE)
        self.assertIn('aslr_model = _load_aslr_pose_model()', APP_SOURCE)
        self.assertIn('def detect_aslr_pose_with_fallback', APP_SOURCE)
        self.assertIn('pose_detector = detect_aslr_pose_with_fallback if is_aslr else detect_pose_with_fallback', APP_SOURCE)
        self.assertIn('rotated_prediction, rotated_threshold, rotated_imgsz = detect_aslr_pose_with_fallback', APP_SOURCE)
        self.assertIn('"general_model_fallback": False', APP_SOURCE)

    def test_aslr_model_is_reported_in_health_and_metrics(self):
        self.assertIn('"aslr_pose_model_loaded": aslr_model is not None', APP_SOURCE)
        self.assertIn('"aslr_pose_model_name": ASLR_POSE_MODEL_NAME', APP_SOURCE)
        self.assertIn('"dedicated_pose_model": ASLR_POSE_MODEL_NAME', APP_SOURCE)
        self.assertIn('"model_role": "dedicated_aslr_pose" if is_aslr else "general_pose"', APP_SOURCE)

    def test_general_tests_keep_existing_model(self):
        self.assertIn('POSE_MODEL_NAME = os.environ.get("FLEXILAB_POSE_MODEL", "yolov8n-pose.pt")', APP_SOURCE)
        self.assertIn('def detect_pose_with_fallback', APP_SOURCE)
        self.assertIn('ASLR_POSE_MODEL_NAME if is_aslr else POSE_MODEL_NAME', APP_SOURCE)

    def test_engine_version(self):
        self.assertIn('ASLR_ENGINE_VERSION = "aslr-dedicated-yolo11m-thresholds-60-75-v6"', ENGINE_SOURCE)

    def test_vision_qa_labels_dedicated_model(self):
        self.assertIn('VISION_QA_VERSION = "vision-qa-overlay-v1.2-aslr-model-label"', VISION_SOURCE)
        self.assertIn('model {model_name}', VISION_SOURCE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
