import math
import unittest

from aslr_engine import ASLR_ENGINE_VERSION, analyze_aslr_v2


def base_pose():
    xy = [[0.0, 0.0] for _ in range(17)]
    conf = [0.95 for _ in range(17)]

    # Torso axis points from shoulders toward the shared pelvic region.
    xy[5] = [100.0, 100.0]
    xy[6] = [100.0, 105.0]
    xy[11] = [200.0, 100.0]
    xy[12] = [200.0, 105.0]

    # COCO left leg raised approximately 90 degrees relative to torso.
    xy[13] = [200.0, 40.0]
    xy[15] = [200.0, -20.0]

    # COCO right leg resting along torso axis.
    xy[14] = [260.0, 105.0]
    xy[16] = [320.0, 105.0]
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


class ASLRBodyRelativeTests(unittest.TestCase):
    def test_angle_is_invariant_to_source_rotation(self):
        xy, conf = base_pose()
        original = analyze_aslr_v2(xy, conf, side="LEFT")
        rotated = analyze_aslr_v2(rotate(xy, 37.0), conf, side="LEFT")
        # A single confidence-weighted pelvic anchor can shift the synthetic
        # exact-right-angle example slightly; invariance is the key contract.
        self.assertGreater(original["metrics"]["aslr_angle"], 88.0)
        self.assertLess(original["metrics"]["aslr_angle"], 90.1)
        self.assertAlmostEqual(
            original["metrics"]["aslr_angle"],
            rotated["metrics"]["aslr_angle"],
            places=1,
        )
        self.assertEqual(
            rotated["metrics"]["measurement_engine_version"],
            ASLR_ENGINE_VERSION,
        )
        self.assertEqual(rotated["metrics"]["source_orientation_requirement"], "none")
        self.assertEqual(
            rotated["metrics"]["pelvic_anchor_method"],
            "confidence_weighted_visible_hip_region",
        )

    def test_duplicate_coco_leg_chains_do_not_force_false_retake(self):
        xy, conf = base_pose()
        xy[12] = list(xy[11])
        xy[14] = list(xy[13])
        xy[16] = list(xy[15])
        result = analyze_aslr_v2(xy, conf, side="RIGHT")
        self.assertGreater(result["metrics"]["aslr_angle"], 80.0)
        self.assertFalse(result["metrics"]["resting_leg_verified"])
        self.assertIn(
            "resting_leg_not_independently_resolved_by_pose_model",
            result["metrics"]["diagnostic_flags"],
        )


if __name__ == "__main__":
    unittest.main()
