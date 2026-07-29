from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np

from aslr_engine import analyze_aslr_rotated_fullbody

ROOT = Path(__file__).resolve().parents[1]


def _pose():
    return np.zeros((17, 2), dtype=float), np.zeros(17, dtype=float)


def _set(xy, conf, index, x, y, confidence=0.95):
    xy[index] = [x, y]
    conf[index] = confidence


def _load_shoulder_function():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "analyze_shoulder"
    )

    def make_thresholds(unit, scale_min, scale_max, bands, pointer_value):
        value = max(float(scale_min), min(float(scale_max), float(pointer_value)))
        rating = "unknown"
        for band in bands:
            if value >= float(band["min"]) and value < float(band["max"]):
                rating = str(band["color"]).lower()
                break
        if value == float(scale_max):
            rating = str(bands[-1]["color"]).lower()
        return {
            "unit": unit,
            "scale_min": scale_min,
            "scale_max": scale_max,
            "bands": bands,
            "pointer_value": round(value, 2),
            "rating": rating,
        }

    namespace = {
        "np": np,
        "math": math,
        "make_thresholds": make_thresholds,
        "SHOULDER_RED_MAX_DEG": 160.0,
        "SHOULDER_GREEN_MIN_DEG": 175.0,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "app.py", "exec"), namespace)
    return namespace["analyze_shoulder"]


def test_aslr_uses_selected_raised_leg_hip_not_shared_pelvis_midpoint():
    xy, conf = _pose()
    # The resting hip shifts the pelvis midpoint 40 px left. The true raised hip
    # is almost vertically below the raised ankle.
    _set(xy, conf, 11, 200, 200)
    _set(xy, conf, 12, 120, 200)
    _set(xy, conf, 13, 202, 125)
    _set(xy, conf, 15, 205, 50)
    _set(xy, conf, 14, 260, 202)
    _set(xy, conf, 16, 360, 205)

    result = analyze_aslr_rotated_fullbody(xy, conf, side="LEFT")

    assert result["metrics"]["aslr_angle"] > 85.0
    assert result["metrics"]["selected_source_indices"] == {
        "hip": 11,
        "knee": 13,
        "ankle": 15,
    }
    assert result["metrics"]["measurement_vertex_policy"].startswith("selected_raised_leg_hip")


def test_aslr_chain_selection_ignores_swapped_coco_labels_but_keeps_geometry():
    xy, conf = _pose()
    _set(xy, conf, 12, 300, 250)
    _set(xy, conf, 11, 240, 250)
    # Cross-label chain is geometrically coherent: hip 12, knee 13, ankle 15.
    _set(xy, conf, 13, 304, 165)
    _set(xy, conf, 15, 307, 75)
    _set(xy, conf, 14, 380, 252)
    _set(xy, conf, 16, 470, 255)

    result = analyze_aslr_rotated_fullbody(xy, conf, side="RIGHT")

    assert result["metrics"]["aslr_angle"] > 85.0
    assert result["metrics"]["selected_source_indices"]["hip"] == 12
    assert result["metrics"]["selected_source_indices"]["ankle"] == 15
    assert "coco_left_right_labels_ignored_for_raised_chain" in result["metrics"]["diagnostic_flags"]


def test_shoulder_170_degrees_is_yellow_and_elbow_is_primary():
    analyze_shoulder = _load_shoulder_function()
    xy, conf = _pose()
    _set(xy, conf, 11, 100, 300)
    _set(xy, conf, 5, 100, 200)
    # Shoulder->elbow is 10 degrees forward of the trunk continuation: 170 flexion.
    _set(xy, conf, 7, 117.365, 101.519)
    _set(xy, conf, 9, 134.73, 3.038)

    result = analyze_shoulder(xy, conf, side="LEFT")

    assert 169.9 <= result["metrics"]["shoulder_flexion_angle"] <= 170.1
    assert result["metrics"]["arm_point_used"] == "ELBOW_PRIMARY"
    assert result["thresholds"]["shoulder_flexion"]["rating"] == "yellow"
    assert result["score"] == 90.0


def test_shoulder_green_requires_175_degrees():
    analyze_shoulder = _load_shoulder_function()
    xy, conf = _pose()
    _set(xy, conf, 11, 100, 300)
    _set(xy, conf, 5, 100, 200)
    _set(xy, conf, 7, 106.976, 100.244)  # about 176 degrees flexion
    _set(xy, conf, 9, 113.952, 0.488)

    result = analyze_shoulder(xy, conf, side="LEFT")

    assert result["metrics"]["shoulder_flexion_angle"] >= 175.5
    assert result["thresholds"]["shoulder_flexion"]["rating"] == "green"


def test_submit_analysis_is_image_fingerprint_aware():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "hashlib.sha256(img_bytes).hexdigest()" in source
    assert '"_flexilab_capture_fingerprint"' in source
    assert '"_flexilab_force_reanalysis"' in source
    assert '"ANALYSIS_ALREADY_RUNNING"' in source
    assert "if existing_screening and not force_reanalysis" in source
    assert "if force_reanalysis and existing_screening" in source


def test_aslr_qa_remains_ephemeral():
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'metrics.pop("vision_qa", None)' in app_source
    assert "if vision_qa_requested:" in app_source


def _load_app_function(name):
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )

    def make_thresholds(unit, scale_min, scale_max, bands, pointer_value):
        value = max(float(scale_min), min(float(scale_max), float(pointer_value)))
        rating = "unknown"
        for band in bands:
            if value >= float(band["min"]) and value < float(band["max"]):
                rating = str(band["color"]).lower()
                break
        if value == float(scale_max):
            rating = str(bands[-1]["color"]).lower()
        return {
            "unit": unit,
            "scale_min": scale_min,
            "scale_max": scale_max,
            "bands": bands,
            "pointer_value": round(value, 2),
            "rating": rating,
        }

    namespace = {
        "np": np,
        "math": math,
        "make_thresholds": make_thresholds,
        "angle_to_vertical": lambda p1, p2: abs(math.degrees(math.atan2(float(p2[0]-p1[0]), float(p1[1]-p2[1])))),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "app.py", "exec"), namespace)
    return namespace[name]


def test_posture_rejects_low_confidence_instead_of_fabricating_minimum_60_percent():
    analyze_posture = _load_app_function("analyze_posture")
    xy, conf = _pose()
    _set(xy, conf, 4, 100, 100, 0.12)
    _set(xy, conf, 6, 100, 200, 0.12)
    _set(xy, conf, 12, 100, 350, 0.12)
    try:
        analyze_posture(xy, conf)
    except ValueError as exc:
        assert "not detected reliably" in str(exc)
    else:
        raise AssertionError("Low-confidence posture must be rejected")


def test_squat_uses_one_coherent_visible_side_not_bilateral_midpoints():
    analyze_squat = _load_app_function("analyze_squat")
    xy, conf = _pose()
    # Reliable right side.
    _set(xy, conf, 6, 180, 80)
    _set(xy, conf, 12, 200, 180)
    _set(xy, conf, 14, 220, 270)
    _set(xy, conf, 16, 290, 280)
    # Left side is deliberately poor and spatially inconsistent.
    _set(xy, conf, 5, 20, 20, 0.10)
    _set(xy, conf, 11, 30, 300, 0.10)
    _set(xy, conf, 13, 400, 40, 0.10)
    _set(xy, conf, 15, 10, 400, 0.10)

    result = analyze_squat(xy, conf)

    assert result["metrics"]["side_used"] == "RIGHT"
    assert result["metrics"]["selected_source_indices"] == {
        "shoulder": 6,
        "hip": 12,
        "knee": 14,
        "ankle": 16,
    }


def test_image_aware_job_action_reuses_only_identical_capture():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {"_analysis_job_private_metadata", "_analysis_job_action"}
    nodes = [
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name in names
    ]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), namespace)
    action = namespace["_analysis_job_action"]

    same = [{
        "id": "job-1",
        "status": "completed",
        "intake_json": {"_flexilab_capture_fingerprint": "abc"},
    }]
    assert action(same, "abc")["action"] == "reuse"
    assert action(same, "different")["action"] == "retake"

    active_other = [{
        "id": "job-2",
        "status": "processing",
        "intake_json": {"_flexilab_capture_fingerprint": "abc"},
    }]
    assert action(active_other, "different")["action"] == "conflict"
