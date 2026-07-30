from aslr_engine import analyze_aslr_rotated_fullbody


def _blank_pose():
    return [[0.0, 0.0] for _ in range(17)], [0.0 for _ in range(17)]


def test_left_aslr_reference_endpoint_points_left_of_selected_hip():
    xy, conf = _blank_pose()
    xy[11], conf[11] = [400.0, 500.0], 0.97
    xy[12], conf[12] = [120.0, 500.0], 0.95
    xy[13], conf[13] = [410.0, 300.0], 0.98
    xy[15], conf[15] = [420.0, 100.0], 0.99
    xy[14], conf[14] = [280.0, 505.0], 0.80
    xy[16], conf[16] = [520.0, 510.0], 0.82

    result = analyze_aslr_rotated_fullbody(xy, conf, side="LEFT")
    metrics = result["metrics"]
    hip_x = metrics["selected_limb_points"]["hip"]["x"]
    ref_x = metrics["body_baseline"]["line_end"]["x"]

    assert ref_x < hip_x
    assert metrics["reference_direction_x"] == -1.0
    assert metrics["aslr_angle"] > 84.0


def test_right_aslr_reference_endpoint_points_right_of_selected_hip():
    xy, conf = _blank_pose()
    xy[11], conf[11] = [120.0, 500.0], 0.95
    xy[12], conf[12] = [390.0, 500.0], 0.97
    xy[14], conf[14] = [400.0, 300.0], 0.98
    xy[16], conf[16] = [410.0, 100.0], 0.99
    xy[13], conf[13] = [260.0, 505.0], 0.80
    xy[15], conf[15] = [520.0, 510.0], 0.82

    result = analyze_aslr_rotated_fullbody(xy, conf, side="RIGHT")
    metrics = result["metrics"]
    hip_x = metrics["selected_limb_points"]["hip"]["x"]
    ref_x = metrics["body_baseline"]["line_end"]["x"]

    assert ref_x > hip_x
    assert metrics["reference_direction_x"] == 1.0
    assert metrics["aslr_angle"] > 84.0
