from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np

from aslr_engine import ASLR_ENGINE_VERSION, ASLRQualityError, analyze_aslr_v2


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")


def _load_app_functions(*names):
    tree = ast.parse(APP_SOURCE)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"np": np}
    exec(compile(module, str(ROOT / "app.py"), "exec"), namespace)
    return [namespace[name] for name in names]


def blank_pose():
    return [[0.0, 0.0] for _ in range(17)], [0.0 for _ in range(17)]


def set_point(xy, conf, index, x, y, confidence=0.92):
    xy[index] = [float(x), float(y)]
    conf[index] = float(confidence)


def valid_pose():
    xy, conf = blank_pose()
    set_point(xy, conf, 5, 20, 100)
    set_point(xy, conf, 6, 20, 104)
    set_point(xy, conf, 11, 100, 100)
    set_point(xy, conf, 12, 104, 104)
    set_point(xy, conf, 13, 102, 50)
    set_point(xy, conf, 15, 102, 0)
    set_point(xy, conf, 14, 180, 103)
    set_point(xy, conf, 16, 260, 103)
    return xy, conf


class ASLRDualOrientationTests(unittest.TestCase):
    def test_patch_contract(self):
        self.assertIn('"patch_version": "V101.28.4-aslr-thresholds-60-75"', APP_SOURCE)
        self.assertIn('cv2.ROTATE_90_CLOCKWISE', APP_SOURCE)
        self.assertIn('"mode": "aslr_dual_orientation"', APP_SOURCE)
        self.assertIn('inference_imgsz=max(POSE_INFERENCE_IMGSZ, 960)', APP_SOURCE)
        self.assertIn('not is_aslr', APP_SOURCE)
        self.assertEqual(ASLR_ENGINE_VERSION, "aslr-dedicated-yolo11m-thresholds-60-75-v6")

    def test_clockwise_pose_mapping_returns_source_coordinates(self):
        mapper, = _load_app_functions("_map_rotated_cw_pose_to_original")
        original_height = 100
        source_point = np.array([[30.0, 40.0]])
        rotated_point = np.array([[original_height - 1 - 40.0, 30.0]])
        rotated_box = np.array([[55.0, 25.0, 63.0, 35.0]])
        mapped_xy, mapped_boxes = mapper(
            rotated_point,
            rotated_box,
            (original_height, 200, 3),
        )
        self.assertAlmostEqual(mapped_xy[0, 0], source_point[0, 0], places=5)
        self.assertAlmostEqual(mapped_xy[0, 1], source_point[0, 1], places=5)
        self.assertEqual(mapped_boxes.shape, (1, 4))

    def test_pass_quality_penalizes_lone_near_baseline_ankle(self):
        scorer, = _load_app_functions("_aslr_pose_pass_quality")
        false_low = {
            "confidence": 0.75,
            "metrics": {
                "selected_chain_score": 0.75,
                "selected_limb_mean_confidence": 0.8,
                "selected_limb_min_confidence": 0.7,
                "aslr_angle": 9.0,
                "detected_ankle_endpoint_count": 1,
                "ankle_endpoints_are_distinct": False,
                "resting_leg_verified": False,
                "diagnostic_flags": ["resting_ankle_not_independently_resolved"],
            },
        }
        reliable = {
            "confidence": 0.62,
            "metrics": {
                "selected_chain_score": 0.68,
                "selected_limb_mean_confidence": 0.65,
                "selected_limb_min_confidence": 0.52,
                "aslr_angle": 48.0,
                "detected_ankle_endpoint_count": 2,
                "ankle_endpoints_are_distinct": True,
                "resting_leg_verified": True,
                "diagnostic_flags": [],
            },
        }
        self.assertGreater(scorer(reliable), scorer(false_low))

    def test_engine_uses_single_pelvic_anchor(self):
        xy, conf = valid_pose()
        result = analyze_aslr_v2(xy, conf, side="RIGHT")
        metrics = result["metrics"]
        pelvis = metrics["selected_limb_points"]["pelvis"]
        self.assertAlmostEqual(pelvis["x"], 102.0, places=1)
        self.assertAlmostEqual(pelvis["y"], 102.0, places=1)
        self.assertEqual(metrics["pelvic_anchor_method"], "confidence_weighted_visible_hip_region")
        self.assertTrue(metrics["ankle_endpoints_are_distinct"])
        self.assertGreater(metrics["aslr_angle"], 80.0)

    def test_lone_near_baseline_ankle_is_rejected_not_scored(self):
        xy, conf = valid_pose()
        conf[15] = 0.01  # raised ankle absent; only resting ankle remains
        with self.assertRaises(ASLRQualityError) as context:
            analyze_aslr_v2(xy, conf, side="RIGHT")
        self.assertEqual(context.exception.code, "raised_ankle_not_detected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
