from __future__ import annotations

import math
import unittest

from aslr_engine import ASLR_ENGINE_VERSION, ASLRQualityError, analyze_aslr_v2


def blank_pose():
    return [[0.0, 0.0] for _ in range(17)], [0.0 for _ in range(17)]


def set_point(xy, conf, index, x, y, confidence=0.92):
    xy[index] = [float(x), float(y)]
    conf[index] = float(confidence)


def crossed_label_pose():
    xy, conf = blank_pose()
    # Body axis: shoulders -> pelvis along +x.
    set_point(xy, conf, 5, 20, 100)
    set_point(xy, conf, 6, 20, 104)
    set_point(xy, conf, 11, 100, 100)
    set_point(xy, conf, 12, 102, 104)

    # Ankle endpoints are correct and distinct.
    set_point(xy, conf, 15, 101, 0)      # visibly raised endpoint
    set_point(xy, conf, 16, 260, 103)    # resting endpoint

    # COCO knee labels are crossed: index 14 belongs to the raised ankle,
    # while index 13 belongs to the resting ankle.
    set_point(xy, conf, 13, 180, 103)
    set_point(xy, conf, 14, 101, 50)
    return xy, conf


def rotate(points, degrees):
    radians = math.radians(degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return [
        [
            point[0] * cosine - point[1] * sine,
            point[0] * sine + point[1] * cosine,
        ]
        for point in points
    ]


class ASLREndpointFirstTests(unittest.TestCase):
    def test_crossed_coco_knee_labels_are_reconstructed(self):
        xy, conf = crossed_label_pose()
        result = analyze_aslr_v2(xy, conf, side="LEFT")
        self.assertEqual(result["metrics"]["measurement_engine_version"], ASLR_ENGINE_VERSION)
        self.assertEqual(result["metrics"]["selected_source_indices"]["ankle"], 15)
        self.assertEqual(result["metrics"]["selected_source_indices"]["knee"], 14)
        self.assertGreater(result["metrics"]["aslr_angle"], 85.0)
        self.assertGreater(result["metrics"]["raised_knee_extension_angle"], 170.0)
        self.assertEqual(
            result["metrics"]["chain_reconstruction_method"],
            "raised_ankle_first_then_best_knee_and_hip_combination",
        )

    def test_geometry_is_independent_from_workflow_side(self):
        xy, conf = crossed_label_pose()
        left = analyze_aslr_v2(xy, conf, side="LEFT")
        right = analyze_aslr_v2(xy, conf, side="RIGHT")
        self.assertAlmostEqual(left["metrics"]["aslr_angle"], right["metrics"]["aslr_angle"], places=2)
        self.assertEqual(left["metrics"]["requested_side"], "LEFT")
        self.assertEqual(right["metrics"]["requested_side"], "RIGHT")

    def test_endpoint_reconstruction_is_rotation_invariant(self):
        xy, conf = crossed_label_pose()
        original = analyze_aslr_v2(xy, conf, side="LEFT")
        rotated = analyze_aslr_v2(rotate(xy, 29.0), conf, side="LEFT")
        self.assertAlmostEqual(
            original["metrics"]["aslr_angle"],
            rotated["metrics"]["aslr_angle"],
            places=1,
        )

    def test_candidate_overlay_labels_endpoint_chains(self):
        xy, conf = crossed_label_pose()
        result = analyze_aslr_v2(xy, conf, side="LEFT")
        labels = [candidate["label"] for candidate in result["metrics"]["candidate_limbs"]]
        self.assertIn("RAISED_ENDPOINT_CHAIN", labels)
        self.assertIn("RESTING_ENDPOINT_CHAIN", labels)
        self.assertEqual(result["metrics"]["selected_limb"], "RAISED_ENDPOINT_CHAIN")

    def test_no_clear_ankle_is_rejected(self):
        xy, conf = crossed_label_pose()
        conf[15] = 0.05
        conf[16] = 0.05
        with self.assertRaises(ASLRQualityError) as context:
            analyze_aslr_v2(xy, conf, side="LEFT")
        self.assertEqual(context.exception.code, "required_landmarks_low_confidence")


if __name__ == "__main__":
    unittest.main(verbosity=2)
