from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")
VISION_SOURCE = (ROOT / "vision_qa.py").read_text(encoding="utf-8")


def test_validation_mode_is_environment_controlled():
    assert 'FLEXILAB_VISION_QA_MODE' in APP_SOURCE
    assert 'VISION_QA_VALIDATION_ENABLED' in APP_SOURCE
    assert 'V101.35.18-validation-overlay' in APP_SOURCE


def test_validation_mode_generates_overlay_for_all_tests():
    assert 'if VISION_QA_VALIDATION_ENABLED or vision_qa_requested:' in APP_SOURCE
    assert 'vision_qa_disabled_for_performance' not in APP_SOURCE
    assert 'all_tests_enabled_in_validation_mode' in APP_SOURCE


def test_aslr_overlay_exposes_both_hips_and_selected_anchor():
    assert 'COCO left hip' in VISION_SOURCE
    assert 'COCO right hip' in VISION_SOURCE
    assert 'SELECTED shared pelvic anchor' in VISION_SOURCE
    assert 'Measurement vertex: shared pelvis' in VISION_SOURCE
    assert 'detected chain' in VISION_SOURCE


def test_overlay_remains_ephemeral():
    assert 'metrics.pop("vision_qa", None)' in APP_SOURCE
    assert 'persisted_to_screenings": False' in APP_SOURCE


def test_aslr_validation_payload_renders():
    import importlib.util

    spec = importlib.util.spec_from_file_location("vision_qa_under_test", ROOT / "vision_qa.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    image = np.zeros((480, 320, 3), dtype=np.uint8)
    xy = np.array([[100.0, 100.0] for _ in range(17)], dtype=float)
    conf = np.full(17, 0.9, dtype=float)
    xy[11] = [140.0, 300.0]
    xy[12] = [160.0, 300.0]
    xy[13] = [150.0, 220.0]
    xy[15] = [150.0, 100.0]

    result = {
        "metrics": {
            "aslr_angle": 85.0,
            "requested_side": "LEFT",
            "detected_coco_side": "LEFT",
            "raised_knee_extension_angle": 170.0,
            "measurement_vertex_policy": "single_shared_pelvic_anchor_for_image_horizontal_and_raised_leg_vectors",
            "selected_limb_points": {
                "hip": {"x": 150.0, "y": 300.0},
                "knee": {"x": 150.0, "y": 220.0},
                "ankle": {"x": 150.0, "y": 100.0},
            },
            "resting_limb_points": {
                "hip": {"x": 150.0, "y": 300.0},
                "reference_endpoint": {"x": 290.0, "y": 300.0},
            },
            "body_baseline": {
                "line_start": {"x": 150.0, "y": 300.0},
                "line_end": {"x": 290.0, "y": 300.0},
                "reference_source": "image_horizontal_primary",
            },
            "reference_source": "image_horizontal_primary",
            "measurement_engine_version": "test-engine",
            "model_runtime": {"model": "yolo11m-pose.pt"},
        }
    }

    payload = module.build_vision_qa_payload(
        image, xy, conf, [0, 0, 319, 479], result, "aslr_left"
    )
    assert payload["test_type"] == "aslr_left"
    assert payload["measurement_engine_version"] == "test-engine"
    assert payload["composite_data_url"].startswith("data:image/jpeg;base64,")
    assert len(payload["keypoints"]) == 17
