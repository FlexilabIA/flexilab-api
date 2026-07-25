from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aslr_engine import ASLRQualityError, analyze_aslr_v2  # noqa: E402


def blank_pose():
    xy = [[0.0, 0.0] for _ in range(17)]
    conf = [0.0 for _ in range(17)]
    return xy, conf


def set_point(xy, conf, index, x, y, confidence=0.85):
    xy[index] = [float(x), float(y)]
    conf[index] = float(confidence)


class FlexiLabV10127ASLRIntegrityTests(unittest.TestCase):
    def make_valid_pose(self):
        xy, conf = blank_pose()
        # COCO left limb: raised and straight, approximately 80 degrees.
        set_point(xy, conf, 11, 100, 200)
        set_point(xy, conf, 13, 112, 130)
        set_point(xy, conf, 15, 122, 62)
        # COCO right limb: straight and on floor.
        set_point(xy, conf, 12, 105, 202)
        set_point(xy, conf, 14, 180, 204)
        set_point(xy, conf, 16, 255, 205)
        return xy, conf

    def test_same_orientation_workflow_side_does_not_require_coco_match(self):
        xy, conf = self.make_valid_pose()
        result = analyze_aslr_v2(xy, conf, side="RIGHT")
        self.assertEqual(result["metrics"]["requested_side"], "RIGHT")
        self.assertEqual(result["metrics"]["detected_coco_side"], "COCO_LEFT")
        self.assertIn("coco_side_differs_from_workflow_side", result["metrics"]["diagnostic_flags"])
        self.assertGreater(result["metrics"]["aslr_angle"], 70)

    def test_visual_thresholds_use_60_75_policy(self):
        xy, conf = self.make_valid_pose()
        result = analyze_aslr_v2(xy, conf, side="LEFT")
        threshold = result["thresholds"]["aslr_angle"]
        bands = threshold["bands"]
        self.assertEqual([(b["min"], b["max"]) for b in bands], [(0, 60), (60, 75), (75, 90)])
        self.assertIn("pointer_value", threshold)
        self.assertNotIn("value", threshold)

    def test_bent_raised_knee_is_rejected(self):
        xy, conf = self.make_valid_pose()
        # Move raised knee laterally to create a pronounced flexion angle.
        set_point(xy, conf, 13, 165, 135)
        with self.assertRaises(ASLRQualityError) as context:
            analyze_aslr_v2(xy, conf, side="LEFT")
        self.assertEqual(context.exception.code, "raised_knee_bent")

    def test_lifted_resting_leg_is_rejected(self):
        xy, conf = self.make_valid_pose()
        set_point(xy, conf, 14, 180, 165)
        set_point(xy, conf, 16, 250, 125)
        with self.assertRaises(ASLRQualityError) as context:
            analyze_aslr_v2(xy, conf, side="LEFT")
        self.assertIn(context.exception.code, {"resting_leg_lifted", "raised_limb_ambiguous"})

    def test_low_confidence_required_landmarks_are_rejected(self):
        xy, conf = self.make_valid_pose()
        conf[15] = 0.08
        conf[16] = 0.08
        with self.assertRaises(ASLRQualityError) as context:
            analyze_aslr_v2(xy, conf, side="LEFT")
        self.assertIn(context.exception.code, {"required_landmarks_low_confidence", "raised_limb_low_confidence"})

    def test_skin_colour_fallback_is_absent(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("estimate_aslr_from_image_skin", source)
        self.assertNotIn("image_skin_fallback_used", source)

    def test_patch_version_and_health_configuration(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('"patch_version": "V101.29-production-efficiency"', source)
        self.assertIn('"visual_thresholds_preserved": True', source)
        self.assertIn("ASLR_ENGINE_VERSION", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
