from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class PatchContractTests(unittest.TestCase):
    def test_backend_contract(self):
        app = (ROOT / "backend" / "app.py").read_text(encoding="utf-8")
        self.assertIn("V101.28-native-camera-vision-qa", app)
        self.assertIn("adaptive_person_crop", app)
        self.assertIn("vision_qa_requested", app)
        self.assertIn("_without_ephemeral_vision_qa", app)
        self.assertIn("AnalysisWithDiagnosticsError", app)
        self.assertIn('failure_update["result_json"]', app)

    def test_frontend_uses_native_file_capture(self):
        screening = (ROOT / "frontend" / "src" / "routes" / "screening.tsx").read_text(encoding="utf-8")
        capture = (ROOT / "frontend" / "src" / "lib" / "assessment-capture.ts").read_text(encoding="utf-8")
        self.assertNotIn("getUserMedia", screening)
        self.assertNotIn("applyMinimumCameraZoom", screening)
        self.assertIn("capture", screening)
        self.assertIn("View what the AI measured", screening)
        self.assertIn("View why the AI rejected the photo", screening)
        api = (ROOT / "frontend" / "src" / "lib" / "api.ts").read_text(encoding="utf-8")
        self.assertIn("AnalysisRejectedError", api)
        self.assertIn("source_orientation_requirement: \"none\"", capture)


if __name__ == "__main__":
    unittest.main()
