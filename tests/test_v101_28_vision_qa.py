import unittest

import numpy as np

from vision_qa import VISION_QA_VERSION, build_vision_qa_payload


class VisionQATests(unittest.TestCase):
    def test_composite_is_generated(self):
        image = np.zeros((480, 320, 3), dtype=np.uint8)
        xy = np.array([[160.0, 120.0] for _ in range(17)], dtype=float)
        conf = np.array([0.9 for _ in range(17)], dtype=float)
        result = {
            "metrics": {
                "side": "RIGHT",
                "shoulder_flexion_angle": 165.0,
                "arm_point_used": "WRIST",
                "measurement_points": {
                    "points": {
                        "shoulder": {"x": 160, "y": 180},
                        "elbow": {"x": 160, "y": 120},
                        "wrist": {"x": 160, "y": 60},
                        "hip": {"x": 160, "y": 300},
                    }
                },
            }
        }
        payload = build_vision_qa_payload(
            image,
            xy,
            conf,
            [80, 40, 240, 420],
            result,
            "shoulder_right",
            analysis_pass={"mode": "full_image"},
        )
        self.assertEqual(payload["version"], VISION_QA_VERSION)
        self.assertTrue(payload["composite_data_url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(len(payload["keypoints"]), 17)


if __name__ == "__main__":
    unittest.main()
