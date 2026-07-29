import ast
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
ASLR = ROOT / "aslr_engine.py"


def _source() -> str:
    return APP.read_text(encoding="utf-8")


def _load_quality_helper():
    tree = ast.parse(_source())
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_aslr_same_side_chain_quality"
    )
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {
        "np": np,
        "math": __import__("math"),
        "ASLR_KEYPOINT_MIN_CONF": 0.20,
        "ASLR_REQUIRED_MEAN_CONF": 0.35,
    }
    exec(compile(module, str(APP), "exec"), namespace)
    return namespace["_aslr_same_side_chain_quality"]


def test_patch_is_conditional_and_aslr_only():
    source = _source()
    assert 'V101.35.20-aslr-conditional-detection-recovery' in source
    assert 'if not bool(first_chain_quality.get("valid_detection"))' in source
    assert 'inference_imgsz=1280' in source
    assert 'rotated_90_clockwise_focused_crop_recovery' in source
    assert 'conditional_focused_crop_used' in source


def test_selector_requires_complete_same_side_chain():
    quality = _load_quality_helper()
    xy = np.zeros((17, 2), dtype=float)
    conf = np.zeros(17, dtype=float)
    xy[11], xy[13], xy[15] = (0, 0), (1, 0), (2, 0)
    conf[11], conf[13], conf[15] = 0.90, 0.92, 0.91
    best, candidates = quality(xy, conf)
    assert best["label"] == "H11-K13-A15"
    assert best["valid_detection"] is True
    assert len(candidates) == 2


def test_missing_hip_does_not_count_as_valid_detection():
    quality = _load_quality_helper()
    xy = np.zeros((17, 2), dtype=float)
    conf = np.zeros(17, dtype=float)
    xy[11], xy[13], xy[15] = (0, 0), (1, 0), (2, 0)
    conf[11], conf[13], conf[15] = 0.05, 0.95, 0.96
    best, _ = quality(xy, conf)
    assert best["valid_detection"] is False


def test_measurement_engine_is_not_replaced_by_detection_patch():
    source = ASLR.read_text(encoding="utf-8")
    assert 'aslr-dedicated-yolo11m-coherent-hip-knee-ankle-one-call-v27' in source
    assert 'coherent_same_side_raised_hip_knee_ankle' in source
