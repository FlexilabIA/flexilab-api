import pytest

from aslr_engine import ASLRQualityError, analyze_aslr_rotated_fullbody


def _blank_pose():
    return [[0.0, 0.0] for _ in range(17)], [0.0 for _ in range(17)]


def test_left_chain_uses_h11_k13_a15_and_near_vertical_angle():
    xy, conf = _blank_pose()
    # Both hips are detected but deliberately far apart. The old midpoint anchor
    # would underestimate the raised-leg angle.
    xy[11], conf[11] = [400.0, 500.0], 0.97
    xy[12], conf[12] = [120.0, 500.0], 0.95
    xy[13], conf[13] = [410.0, 300.0], 0.98
    xy[15], conf[15] = [420.0, 100.0], 0.99
    # Resting right chain.
    xy[14], conf[14] = [280.0, 505.0], 0.80
    xy[16], conf[16] = [520.0, 510.0], 0.82

    result = analyze_aslr_rotated_fullbody(xy, conf, side="LEFT")
    metrics = result["metrics"]

    assert metrics["aslr_angle"] > 84.0
    assert metrics["selected_source_indices"] == {"hip": 11, "knee": 13, "ankle": 15}
    assert metrics["selected_limb_points"]["hip"] == {"x": 400.0, "y": 500.0}
    assert metrics["measurement_vertex_policy"] == (
        "selected_raised_leg_hip_for_image_horizontal_and_raised_leg_vectors"
    )


def test_right_chain_uses_h12_k14_a16_and_near_vertical_angle():
    xy, conf = _blank_pose()
    xy[11], conf[11] = [120.0, 500.0], 0.95
    xy[12], conf[12] = [390.0, 500.0], 0.97
    xy[14], conf[14] = [400.0, 300.0], 0.98
    xy[16], conf[16] = [410.0, 100.0], 0.99
    xy[13], conf[13] = [260.0, 505.0], 0.80
    xy[15], conf[15] = [520.0, 510.0], 0.82

    result = analyze_aslr_rotated_fullbody(xy, conf, side="RIGHT")
    assert result["metrics"]["aslr_angle"] > 84.0
    assert result["metrics"]["selected_source_indices"] == {"hip": 12, "knee": 14, "ankle": 16}


def test_missing_same_side_hip_is_rejected_not_replaced_by_pelvis_midpoint():
    xy, conf = _blank_pose()
    # Raised left knee/ankle are clear but H11 is absent. H12 must not be used as
    # a synthetic/shared substitute for the left chain.
    xy[12], conf[12] = [120.0, 500.0], 0.96
    xy[13], conf[13] = [410.0, 300.0], 0.98
    xy[15], conf[15] = [420.0, 100.0], 0.99

    with pytest.raises(ASLRQualityError) as exc:
        analyze_aslr_rotated_fullbody(xy, conf, side="LEFT")

    assert exc.value.code in {"raised_ankle_or_knee_not_detected", "raised_chain_ambiguous"}
