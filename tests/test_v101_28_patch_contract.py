from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PatchContractTests(unittest.TestCase):
    def test_backend_contract(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        engine = (ROOT / "aslr_engine.py").read_text(encoding="utf-8")
        self.assertIn("V101.28.3-aslr-dedicated-yolo11m", app)
        self.assertIn("cv2.ROTATE_90_CLOCKWISE", app)
        self.assertIn("aslr_dual_orientation", app)
        self.assertIn("vision_qa_requested", app)
        self.assertIn("_without_ephemeral_vision_qa", app)
        self.assertIn("single_pelvic_anchor", engine)
        self.assertIn("raised_ankle_not_detected", engine)


if __name__ == "__main__":
    unittest.main()
