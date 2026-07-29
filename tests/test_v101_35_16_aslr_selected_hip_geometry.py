import math

from aslr_engine import analyze_aslr_rotated_fullbody


def _blank_pose():
    return [[0.0, 0.0] for _ in range(17)], [0.0 for _ in range(17)]


def test_near_vertical_raised_leg_is_measured_from_selected_hip_not_pelvis_midpoint():
    xy, conf = _blank_pose()

    # Deliberately separated hip detections. A midpoint anchor would be far to
    # the left and would materially under-estimate this near-vertical leg.
    xy[11], conf[11] = [120.0, 500.0], 0.95
    xy[12], conf[12] = [400.0, 500.0], 0.96

    # Coherent raised chain from right hip -> right knee -> right ankle.
    xy[14], conf[14] = [415.0, 300.0], 0.97
    xy[16], conf[16] = [430.0, 100.0], 0.98

    # Plausible resting-leg points, lower confidence and horizontal.
    xy[13], conf[13] = [250.0, 505.0], 0.70
    xy[15], conf[15] = [520.0, 510.0], 0.72

    result = analyze_aslr_rotated_fullbody(xy, conf, side="LEFT")
    angle = result["metrics"]["aslr_angle"]

    assert angle > 84.0
    assert result["metrics"]["selected_source_indices"]["hip"] == 12
    assert result["metrics"]["selected_limb_points"]["hip"] == {"x": 400.0, "y": 500.0}
    assert "selected_raised_hip" in result["metrics"]["angle_method"]


def test_selected_chain_uses_one_real_hip_knee_ankle_chain():
    xy, conf = _blank_pose()
    xy[11], conf[11] = [200.0, 500.0], 0.93
    xy[12], conf[12] = [390.0, 500.0], 0.95
    xy[14], conf[14] = [400.0, 305.0], 0.96
    xy[16], conf[16] = [410.0, 110.0], 0.97
    xy[13], conf[13] = [280.0, 500.0], 0.65
    xy[15], conf[15] = [530.0, 500.0], 0.68

    result = analyze_aslr_rotated_fullbody(xy, conf, side="LEFT")
    indices = result["metrics"]["selected_source_indices"]

    assert indices == {"hip": 12, "knee": 14, "ankle": 16}
    assert result["metrics"]["measurement_vertex_policy"] == (
        "selected_raised_leg_hip_for_image_horizontal_and_raised_leg_vectors"
    )
