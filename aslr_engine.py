"""FlexiLab ASLR rotated full-body body-axis engine.

V101.35.11 rotates a private full-body copy 90 degrees clockwise for YOLO pose
inference, inverse-maps the landmarks to the original horizontal photo, and
measures the raised leg from a shared pelvic anchor. The raised endpoint must
always be a true YOLO ankle keypoint. The reference axis is always the
shoulder-midpoint-to-pelvis body axis extended through the pelvis.

Floor-leg knee and ankle landmarks are optional validation signals only. They
can never alter the reported ASLR angle. The displayed pink and yellow lines use
the exact same vectors as the reported angle. A toe, skin contour, shoe edge,
or invented foot endpoint is never used.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

ASLR_ENGINE_VERSION = "aslr-dedicated-yolo11m-body-axis-primary-v22"
ASLR_THRESHOLD_EVIDENCE_STATUS = (
    "provisional_flexilab_reference_bands_not_diagnostic_cutoffs"
)

ASLR_RED_MAX_DEG = 60.0
ASLR_YELLOW_MAX_DEG = 75.0
ASLR_SCALE_MAX_DEG = 90.0


class ASLRQualityError(ValueError):
    def __init__(self, code: str, message: str, details: Mapping[str, Any] | None = None):
        self.code = code
        self.details = dict(details or {})
        super().__init__(message)


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite coordinate")
    return number


def _point(xy: Sequence[Sequence[float]], index: int) -> Tuple[float, float]:
    return (_finite(xy[index][0]), _finite(xy[index][1]))


def _confidence(conf: Sequence[float], index: int) -> float:
    return max(0.0, min(1.0, _finite(conf[index])))


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _joint_angle(
    a: Tuple[float, float],
    vertex: Tuple[float, float],
    c: Tuple[float, float],
) -> float:
    v1 = (a[0] - vertex[0], a[1] - vertex[1])
    v2 = (c[0] - vertex[0], c[1] - vertex[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 <= 1e-6 or n2 <= 1e-6:
        return 0.0
    cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosine))


def _rounded_point(point: Tuple[float, float] | None) -> Dict[str, float] | None:
    if point is None:
        return None
    return {"x": round(float(point[0]), 2), "y": round(float(point[1]), 2)}


def _acute_angle_between_vectors(
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    na = math.hypot(a[0], a[1])
    nb = math.hypot(b[0], b[1])
    if na <= 1e-6 or nb <= 1e-6:
        return 0.0
    cosine = abs((a[0] * b[0] + a[1] * b[1]) / (na * nb))
    cosine = max(-1.0, min(1.0, cosine))
    return max(0.0, min(90.0, math.degrees(math.acos(cosine))))


def _horizontal_angle(anchor: Tuple[float, float], endpoint: Tuple[float, float]) -> float:
    dx = abs(float(endpoint[0] - anchor[0]))
    dy_up = max(0.0, float(anchor[1] - endpoint[1]))
    return max(0.0, min(90.0, math.degrees(math.atan2(dy_up, dx + 1e-6))))


def _point_to_segment_ratio(
    point: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> Tuple[float, float]:
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    length_sq = vx * vx + vy * vy
    length = math.sqrt(length_sq)
    if length <= 1e-6:
        return 1.0, 0.0
    wx = point[0] - start[0]
    wy = point[1] - start[1]
    t = (wx * vx + wy * vy) / length_sq
    projected = (start[0] + t * vx, start[1] + t * vy)
    return _distance(point, projected) / length, t


def _score_from_reference_bands(angle: float) -> float:
    if angle < 60.0:
        return max(0.0, 40.0 + (angle / 60.0) * 19.0)
    if angle <= 75.0:
        return 60.0 + ((angle - 60.0) / 15.0) * 19.0
    return 85.0 + ((min(angle, 90.0) - 75.0) / 15.0) * 15.0


def make_aslr_thresholds(value: float) -> Dict[str, Any]:
    value = max(0.0, min(90.0, float(value)))
    if value < ASLR_RED_MAX_DEG:
        rating = "red"
    elif value <= ASLR_YELLOW_MAX_DEG:
        rating = "yellow"
    else:
        rating = "green"
    return {
        "unit": "deg",
        "scale_min": 0,
        "scale_max": 90,
        "bands": [
            {"label": "Red", "min": 0, "max": 60, "color": "red"},
            {"label": "Yellow", "min": 60, "max": 75, "color": "yellow"},
            {"label": "Green", "min": 75, "max": 90, "color": "green"},
        ],
        "pointer_value": round(value, 2),
        "rating": rating,
        "boundary_policy": {"red": "<60", "yellow": "60-75_inclusive", "green": ">75"},
        "visual_band_layout": "equal_thirds",
    }


def _fit_body_chain(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    label: str,
    ear_index: int,
    shoulder_index: int,
    hip_index: int,
) -> Dict[str, Any] | None:
    """Build one subject reference axis from the same-side ear/shoulder to the hip.

    Primary rule: ear -> hip.
    Fallback rule: shoulder -> hip when the ear is not reliable enough.
    The shoulder is still used as a validation anchor when the ear is available.
    """
    shoulder_conf = _confidence(conf, shoulder_index)
    hip_conf = _confidence(conf, hip_index)
    ear_conf = _confidence(conf, ear_index)
    if hip_conf < 0.08 or max(ear_conf, shoulder_conf) < 0.08:
        return None

    hip = _point(xy, hip_index)
    shoulder = _point(xy, shoulder_index) if shoulder_conf >= 0.08 else None
    ear = _point(xy, ear_index) if ear_conf >= 0.08 else None

    if ear is not None:
        origin = ear
        origin_label = "ear"
        origin_conf = ear_conf
    elif shoulder is not None:
        origin = shoulder
        origin_label = "shoulder"
        origin_conf = shoulder_conf
    else:
        return None

    direction_arr = np.asarray([hip[0] - origin[0], hip[1] - origin[1]], dtype=float)
    body_length = float(np.linalg.norm(direction_arr))
    if body_length <= 1e-6:
        return None
    direction = direction_arr / body_length

    validation_points = []
    validation_weights = []
    if origin_label == "ear" and shoulder is not None:
        validation_points.append(shoulder)
        validation_weights.append(max(0.20, shoulder_conf))
    elif origin_label == "shoulder" and ear is not None:
        validation_points.append(ear)
        validation_weights.append(max(0.10, ear_conf * 0.5))

    if validation_points:
        normal = np.asarray([-direction[1], direction[0]], dtype=float)
        centered = np.asarray(validation_points, dtype=float) - np.asarray(origin, dtype=float)
        residuals = np.abs(centered @ normal)
        residual_ratio = float(np.average(residuals, weights=np.asarray(validation_weights, dtype=float)) / max(body_length, 1.0))
    else:
        residual_ratio = 0.0
    collinearity = max(0.0, min(1.0, 1.0 - residual_ratio * 6.0))

    mean_confidence = float(np.mean([hip_conf, origin_conf] + ([shoulder_conf] if shoulder is not None else [])))
    score = mean_confidence * 0.74 + collinearity * 0.26

    return {
        "side": label,
        "ear": ear,
        "shoulder": shoulder,
        "hip": hip,
        "ear_index": ear_index if ear is not None else None,
        "shoulder_index": shoulder_index if shoulder is not None else None,
        "hip_index": hip_index,
        "reference_origin": origin,
        "reference_origin_label": origin_label,
        "line_start": origin,
        "line_end": hip,
        "direction": (float(direction[0]), float(direction[1])),
        "image_angle_deg": float(math.degrees(math.atan2(direction[1], direction[0]))),
        "confidence": max(0.0, min(1.0, score)),
        "collinearity": collinearity,
        "residual_ratio": residual_ratio,
        "anchors_used": [origin_label, "hip"] + (["shoulder_validation"] if shoulder is not None and origin_label == "ear" else []),
    }


def _build_selected_hip_reference_axis(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    selected_hip: Tuple[float, float],
    selected_hip_idx: int,
) -> Dict[str, Any] | None:
    del selected_hip
    if int(selected_hip_idx) == 11:
        return _fit_body_chain(xy, conf, "COCO_LEFT_BODY", 3, 5, 11)
    if int(selected_hip_idx) == 12:
        return _fit_body_chain(xy, conf, "COCO_RIGHT_BODY", 4, 6, 12)
    return None


def _body_reference_axis(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    *,
    selected_hip: Tuple[float, float] | None = None,
    selected_hip_idx: int | None = None,
) -> Dict[str, Any]:
    if selected_hip is not None and selected_hip_idx is not None:
        selected_axis = _build_selected_hip_reference_axis(xy, conf, selected_hip, selected_hip_idx)
        if selected_axis is not None:
            return selected_axis

    candidates = [
        _fit_body_chain(xy, conf, "COCO_LEFT_BODY", 3, 5, 11),
        _fit_body_chain(xy, conf, "COCO_RIGHT_BODY", 4, 6, 12),
    ]
    candidates = [candidate for candidate in candidates if candidate is not None]
    if not candidates:
        raise ASLRQualityError(
            "body_axis_not_detected",
            "Keep one ear, one shoulder, the pelvis and the raised foot visible, then retake the photo.",
        )
    return max(candidates, key=lambda candidate: candidate["confidence"])


def _pelvis_center(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
) -> Tuple[Tuple[float, float], float]:
    hips = [
        (idx, _point(xy, idx), _confidence(conf, idx))
        for idx in (11, 12)
        if _confidence(conf, idx) >= 0.08
    ]
    if not hips:
        raise ASLRQualityError(
            "pelvis_not_detected",
            "Keep the pelvis and both legs visible, then retake the photo.",
        )
    weight_sum = sum(c for _, _, c in hips)
    center = (
        sum(p[0] * c for _, p, c in hips) / max(weight_sum, 1e-6),
        sum(p[1] * c for _, p, c in hips) / max(weight_sum, 1e-6),
    )
    return center, weight_sum / len(hips)


def _vector(
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> Tuple[float, float]:
    return (end[0] - start[0], end[1] - start[1])


def _dot(
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    return float(a[0] * b[0] + a[1] * b[1])


def _reference_line_through_selected_hip(
    body_baseline: Mapping[str, Any],
    selected_hip: Tuple[float, float],
    reference_xy: Sequence[Sequence[float]],
    reference_conf: Sequence[float],
) -> Dict[str, Any]:
    """Translate the fitted body direction through the selected raised hip.

    The angle between two vectors is translation invariant, so the visual line
    can be extended beyond the hip without changing the formula. The preferred
    construction is: ear -> selected hip, then continue in the same direction
    toward the resting foot. If the ear is unreliable, shoulder -> selected hip
    becomes the fallback reference construction.
    """
    direction = (
        float(body_baseline["direction"][0]),
        float(body_baseline["direction"][1]),
    )
    upper = body_baseline.get("reference_origin") or body_baseline.get("ear") or body_baseline.get("shoulder")
    source_hip = body_baseline.get("hip") or selected_hip
    body_length = max(80.0, _distance(upper, source_hip) if upper is not None else 180.0)

    def projection(point: Tuple[float, float]) -> float:
        return _dot(_vector(selected_hip, point), direction)

    start_projection = projection(upper) if upper is not None else -body_length
    # Ensure the line visibly reaches the head side even if the selected rotated
    # hip differs by a few pixels from the original-photo hip landmark.
    start_projection = min(start_projection, -0.72 * body_length)

    distal_candidates = []
    for index in (15, 16, 13, 14):
        if index >= len(reference_xy) or index >= len(reference_conf):
            continue
        confidence = _confidence(reference_conf, index)
        if confidence < 0.08:
            continue
        point = _point(reference_xy, index)
        scalar = projection(point)
        if scalar > 0:
            distal_candidates.append((scalar, point, index, confidence))

    if distal_candidates:
        distal_scalar, distal_point, distal_index, distal_confidence = max(
            distal_candidates, key=lambda item: item[0]
        )
        end_projection = max(distal_scalar, body_length * 1.05)
    else:
        distal_point = None
        distal_index = None
        distal_confidence = None
        end_projection = body_length * 1.35

    line_start = (
        selected_hip[0] + start_projection * direction[0],
        selected_hip[1] + start_projection * direction[1],
    )
    line_end = (
        selected_hip[0] + end_projection * direction[0],
        selected_hip[1] + end_projection * direction[1],
    )

    return {
        "line_start": line_start,
        "line_end": line_end,
        "measurement_vertex": selected_hip,
        "start_projection_px": start_projection,
        "end_projection_px": end_projection,
        "distal_reference_point": distal_point,
        "distal_reference_index": distal_index,
        "distal_reference_confidence": distal_confidence,
        "policy": "ear_or_shoulder_to_selected_hip_then_extend_toward_resting_foot",
    }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _projection_geometry(
    hip: Tuple[float, float],
    knee: Tuple[float, float],
    ankle: Tuple[float, float],
) -> Dict[str, float]:
    """Return knee position relative to the straight hip-to-ankle segment."""
    leg = _vector(hip, ankle)
    length_sq = leg[0] * leg[0] + leg[1] * leg[1]
    leg_length = math.sqrt(max(length_sq, 0.0))
    if leg_length <= 1e-6:
        return {
            "projection": 0.0,
            "perpendicular_ratio": 99.0,
            "leg_length": 0.0,
        }
    knee_relative = _vector(hip, knee)
    projection = (
        knee_relative[0] * leg[0] + knee_relative[1] * leg[1]
    ) / length_sq
    closest = (
        hip[0] + projection * leg[0],
        hip[1] + projection * leg[1],
    )
    perpendicular_ratio = _distance(knee, closest) / leg_length
    return {
        "projection": projection,
        "perpendicular_ratio": perpendicular_ratio,
        "leg_length": leg_length,
    }


def _chain_candidate(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    hip_index: int,
    knee_index: int,
    ankle_index: int,
    pelvis: Tuple[float, float],
    *,
    keypoint_min_conf: float,
) -> Dict[str, Any] | None:
    """Score one hip/knee combination around a genuine YOLO ankle endpoint.

    COCO left/right labels may cross in a supine side view, so the raised ankle
    is fixed first and both hips and both knees are evaluated around it. This is
    the stable V101.28 endpoint-first behaviour, with the endpoint restricted to
    keypoint 15 or 16 only.
    """
    hip_conf = _confidence(conf, hip_index)
    knee_conf = _confidence(conf, knee_index)
    ankle_conf = _confidence(conf, ankle_index)

    if ankle_conf < keypoint_min_conf:
        return None
    if knee_conf < max(0.10, keypoint_min_conf * 0.60):
        return None
    if hip_conf < max(0.08, keypoint_min_conf * 0.50):
        return None

    hip = _point(xy, hip_index)
    knee = _point(xy, knee_index)
    ankle = _point(xy, ankle_index)
    geometry = _projection_geometry(hip, knee, ankle)
    leg_length = geometry["leg_length"]
    if leg_length < 50.0:
        return None

    projection = geometry["projection"]
    perpendicular_ratio = geometry["perpendicular_ratio"]
    if projection < -0.10 or projection > 1.10:
        return None

    thigh_length = _distance(hip, knee)
    shank_length = _distance(knee, ankle)
    if thigh_length <= 2.0 or shank_length <= 2.0:
        return None

    knee_extension = _joint_angle(hip, knee, ankle)
    segment_ratio = thigh_length / max(shank_length, 1e-6)
    mean_confidence = (hip_conf + knee_conf + ankle_conf) / 3.0
    minimum_confidence = min(hip_conf, knee_conf, ankle_conf)

    straightness = _clamp((knee_extension - 115.0) / 65.0)
    line_alignment = _clamp(1.0 - perpendicular_ratio / 0.24)
    projection_quality = _clamp(1.0 - abs(projection - 0.50) / 0.55)
    ratio_quality = _clamp(
        1.0 - abs(math.log(max(segment_ratio, 1e-6))) / math.log(3.5)
    )
    hip_distance = _distance(hip, pelvis)
    hip_quality = _clamp(1.0 - hip_distance / max(20.0, leg_length * 0.30))

    same_coco_side = (
        (hip_index, knee_index, ankle_index) == (11, 13, 15)
        or (hip_index, knee_index, ankle_index) == (12, 14, 16)
    )

    chain_score = (
        mean_confidence * 0.30
        + straightness * 0.25
        + line_alignment * 0.20
        + projection_quality * 0.10
        + ratio_quality * 0.09
        + hip_quality * 0.06
        + (0.015 if same_coco_side else 0.0)
    )

    available = (
        knee_extension >= 125.0
        and perpendicular_ratio <= 0.24
        and -0.05 <= projection <= 1.05
        and 0.28 <= segment_ratio <= 3.50
    )

    return {
        "hip_idx": hip_index,
        "knee_idx": knee_index,
        "ankle_idx": ankle_index,
        "hip": hip,
        "knee": knee,
        "ankle": ankle,
        "keypoint_confidence": {
            "hip": hip_conf,
            "knee": knee_conf,
            "ankle": ankle_conf,
        },
        "minimum_confidence": minimum_confidence,
        "mean_confidence": mean_confidence,
        "knee_extension_angle": knee_extension,
        "thigh_length_px": thigh_length,
        "shank_length_px": shank_length,
        "thigh_to_shank_ratio": segment_ratio,
        "projection": projection,
        "perpendicular_ratio": perpendicular_ratio,
        "chain_score": chain_score,
        "same_coco_side": same_coco_side,
        "available": available,
    }


def _best_chain_for_ankle(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    ankle_index: int,
    pelvis: Tuple[float, float],
    *,
    keypoint_min_conf: float,
) -> Dict[str, Any] | None:
    candidates = []
    for hip_index in (11, 12):
        for knee_index in (13, 14):
            candidate = _chain_candidate(
                xy,
                conf,
                hip_index,
                knee_index,
                ankle_index,
                pelvis,
                keypoint_min_conf=keypoint_min_conf,
            )
            if candidate is not None:
                candidates.append(candidate)
    if not candidates:
        return None
    available = [candidate for candidate in candidates if candidate["available"]]
    pool = available if available else candidates
    return max(pool, key=lambda candidate: candidate["chain_score"])


def analyze_aslr_v2(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    side: str = "RIGHT",
    *,
    img: np.ndarray | None = None,
    body_xy: Sequence[Sequence[float]] | None = None,
    body_conf: Sequence[float] | None = None,
    keypoint_min_conf: float = 0.20,
    required_mean_conf: float = 0.35,
    raised_knee_extension_min: float = 150.0,
    resting_knee_extension_min: float = 145.0,
    resting_leg_max_angle: float = 15.0,
    **_: Any,
) -> Dict[str, Any]:
    """Measure ASLR from a subject reference line to a genuine YOLO ankle.

    Detection strategy:
    1. Provisional body direction is inferred from ear/shoulder/hip landmarks.
    2. The raised endpoint is selected exactly as in the stable V101.28 model:
       evaluate genuine YOLO ankle keypoints 15 and 16 first.
    3. After the raised chain is selected, the final reference line is rebuilt
       from the selected raised hip to the same-side ear. If the ear is weak,
       the same-side shoulder becomes the fallback origin.
    4. The pink line is then extended through the hip toward the resting foot,
       while the yellow line remains the selected hip -> true YOLO ankle line.
    2. The raised endpoint is selected exactly as in the stable V101.28 model:
       evaluate genuine YOLO ankle keypoints 15 and 16 first.
    3. For each ankle, reconstruct the most plausible hip/knee chain from all
       bilateral combinations because COCO left/right labels can swap while the
       person is supine.
    4. The measurement line always stops at the selected ankle keypoint. Skin,
       shoe contour, toe tip and highest-visible pixels are never endpoints.
    """
    del img, resting_knee_extension_min, resting_leg_max_angle
    if len(xy) < 17 or len(conf) < 17:
        raise ASLRQualityError(
            "missing_pose",
            "Keep the head, pelvis and both complete legs visible, then retake the photo.",
        )

    requested_side = str(side or "RIGHT").upper()
    if requested_side not in {"LEFT", "RIGHT"}:
        requested_side = "RIGHT"

    reference_xy = body_xy if body_xy is not None else xy
    reference_conf = body_conf if body_conf is not None else conf
    if len(reference_xy) < 17 or len(reference_conf) < 17:
        reference_xy = xy
        reference_conf = conf

    provisional_body_baseline = _body_reference_axis(reference_xy, reference_conf)
    pelvis, pelvis_confidence = _pelvis_center(reference_xy, reference_conf)

    endpoint_candidates = []
    for ankle_index, ankle_label in (
        (15, "COCO_LEFT_ANKLE"),
        (16, "COCO_RIGHT_ANKLE"),
    ):
        ankle_confidence = _confidence(conf, ankle_index)
        if ankle_confidence < keypoint_min_conf:
            continue
        chain = _best_chain_for_ankle(
            xy,
            conf,
            ankle_index,
            pelvis,
            keypoint_min_conf=keypoint_min_conf,
        )
        if chain is None:
            continue
        leg_vector = _vector(chain["hip"], chain["ankle"])
        body_relative_angle = _acute_angle_between_vectors(
            provisional_body_baseline["direction"],
            leg_vector,
        )
        endpoint_candidates.append({
            "ankle_idx": ankle_index,
            "ankle_label": ankle_label,
            "ankle_confidence": ankle_confidence,
            "chain": chain,
            "body_relative_angle": body_relative_angle,
            "horizontal_angle": _horizontal_angle(chain["hip"], chain["ankle"]),
        })

    if not endpoint_candidates:
        raise ASLRQualityError(
            "raised_ankle_or_knee_not_detected",
            "The raised ankle and knee were not identified reliably. Keep the complete raised leg and foot visible, then retake the photo.",
            {"endpoint_policy": "true_yolo_ankle_15_or_16_only"},
        )

    # V101.28 behaviour: the ankle with the greatest body-relative elevation is
    # the raised endpoint. Chain quality and ankle confidence resolve close ties.
    endpoint_candidates.sort(
        key=lambda item: (
            item["body_relative_angle"],
            item["chain"]["chain_score"],
            item["ankle_confidence"],
        ),
        reverse=True,
    )
    selected_endpoint = endpoint_candidates[0]
    selected = selected_endpoint["chain"]

    # Rebuild the final body reference from the selected raised hip so the
    # displayed and computed reference line passes through the visible ear/hip
    # pair whenever possible.
    body_baseline = provisional_body_baseline
    for _ in range(2):
        refined_baseline = _body_reference_axis(
            reference_xy,
            reference_conf,
            selected_hip=selected["hip"],
            selected_hip_idx=int(selected["hip_idx"]),
        )
        reranked = []
        for endpoint in endpoint_candidates:
            endpoint_copy = dict(endpoint)
            leg_vector = _vector(endpoint_copy["chain"]["hip"], endpoint_copy["chain"]["ankle"])
            endpoint_copy["body_relative_angle"] = _acute_angle_between_vectors(
                refined_baseline["direction"],
                leg_vector,
            )
            reranked.append(endpoint_copy)
        reranked.sort(
            key=lambda item: (
                item["body_relative_angle"],
                item["chain"]["chain_score"],
                item["ankle_confidence"],
            ),
            reverse=True,
        )
        body_baseline = refined_baseline
        endpoint_candidates = reranked
        new_selected_endpoint = endpoint_candidates[0]
        new_selected = new_selected_endpoint["chain"]
        if (
            int(new_selected_endpoint["ankle_idx"]) == int(selected_endpoint["ankle_idx"])
            and int(new_selected["hip_idx"]) == int(selected["hip_idx"])
        ):
            selected_endpoint = new_selected_endpoint
            selected = new_selected
            break
        selected_endpoint = new_selected_endpoint
        selected = new_selected

    final_angle = float(selected_endpoint["body_relative_angle"])

    if final_angle < 18.0:
        raise ASLRQualityError(
            "raised_ankle_not_detected",
            "The detected ankle is aligned with the resting leg. Keep the raised foot fully visible and retake the photo.",
            {
                "candidate_angle": round(final_angle, 2),
                "selected_ankle_index": int(selected_endpoint["ankle_idx"]),
            },
        )

    if not selected["available"]:
        raise ASLRQualityError(
            "raised_chain_geometry_uncertain",
            "The ankle was detected, but it could not be connected reliably to the raised knee and hip. Keep the full raised leg visible and retake the photo.",
            {
                "selected_ankle_index": int(selected_endpoint["ankle_idx"]),
                "knee_extension_angle": round(selected["knee_extension_angle"], 2),
                "knee_line_distance_ratio": round(selected["perpendicular_ratio"], 4),
            },
        )

    if selected["mean_confidence"] < required_mean_conf:
        raise ASLRQualityError(
            "raised_limb_low_confidence",
            "The raised hip, knee or ankle is not clear enough. Improve the lighting and retake the photo.",
            {
                "mean_confidence": round(selected["mean_confidence"], 3),
                "minimum_confidence": round(selected["minimum_confidence"], 3),
            },
        )

    flags = []
    knee_extension = float(selected["knee_extension_angle"])
    if knee_extension < raised_knee_extension_min:
        flags.append("raised_knee_below_preferred_extension")
    if not selected["same_coco_side"]:
        flags.append("coco_labels_crossed_endpoint_first_chain_reconstruction_used")

    other_endpoint = endpoint_candidates[1] if len(endpoint_candidates) > 1 else None
    ankle_separation = None
    ankles_distinct = False
    if other_endpoint is not None:
        ankle_separation = _distance(selected["ankle"], other_endpoint["chain"]["ankle"])
        body_scale = max(80.0, _distance(selected["hip"], selected["ankle"]))
        ankles_distinct = ankle_separation >= max(22.0, body_scale * 0.12)
        if not ankles_distinct:
            flags.append("ankle_endpoints_duplicated_by_pose_model")
    else:
        flags.append("only_one_yolo_ankle_detected")

    reliability = (
        body_baseline["confidence"] * 0.28
        + selected["mean_confidence"] * 0.30
        + selected["chain_score"] * 0.30
        + _clamp(knee_extension / 180.0) * 0.12
    )
    if not ankles_distinct and len(endpoint_candidates) > 1:
        reliability *= 0.93
    reliability = _clamp(reliability)

    common_reference = _reference_line_through_selected_hip(
        body_baseline,
        selected["hip"],
        reference_xy,
        reference_conf,
    )

    body_baseline_payload = {
        "method": "ear_to_selected_hip_with_shoulder_fallback_then_extended_toward_resting_foot",
        "side": body_baseline["side"],
        "ear": _rounded_point(body_baseline["ear"]),
        "shoulder": _rounded_point(body_baseline["shoulder"]),
        "pelvis": _rounded_point(body_baseline["hip"]),
        "reference_origin": _rounded_point(body_baseline.get("reference_origin")),
        "reference_origin_label": body_baseline.get("reference_origin_label"),
        "source_fit_line_start": _rounded_point(body_baseline["line_start"]),
        "source_fit_line_end": _rounded_point(body_baseline["line_end"]),
        "line_start": _rounded_point(common_reference["line_start"]),
        "line_end": _rounded_point(common_reference["line_end"]),
        "measurement_vertex": _rounded_point(common_reference["measurement_vertex"]),
        "common_vertex_policy": common_reference["policy"],
        "distal_reference_point": _rounded_point(common_reference["distal_reference_point"]),
        "distal_reference_index": common_reference["distal_reference_index"],
        "distal_reference_confidence": round(float(common_reference["distal_reference_confidence"]), 3) if common_reference["distal_reference_confidence"] is not None else None,
        "direction": {
            "x": round(float(body_baseline["direction"][0]), 6),
            "y": round(float(body_baseline["direction"][1]), 6),
        },
        "image_angle_deg": round(float(body_baseline["image_angle_deg"]), 2),
        "confidence": round(float(body_baseline["confidence"]), 3),
        "collinearity": round(float(body_baseline["collinearity"]), 3),
        "anchors_used": body_baseline["anchors_used"],
        "source_indices": {
            "ear": body_baseline["ear_index"],
            "shoulder": body_baseline["shoulder_index"],
            "hip": body_baseline["hip_index"],
        },
    }

    selected_points = {
        "pelvis": _rounded_point(pelvis),
        "hip": _rounded_point(selected["hip"]),
        "knee": _rounded_point(selected["knee"]),
        "ankle": _rounded_point(selected["ankle"]),
    }

    public_candidates = []
    for endpoint in endpoint_candidates:
        chain = endpoint["chain"]
        public_candidates.append({
            "label": endpoint["ankle_label"],
            "points": {
                "hip": _rounded_point(chain["hip"]),
                "knee": _rounded_point(chain["knee"]),
                "ankle": _rounded_point(chain["ankle"]),
            },
            "available": bool(chain["available"]),
            "body_relative_angle": round(endpoint["body_relative_angle"], 2),
            "horizontal_angle": round(endpoint["horizontal_angle"], 2),
            "chain_score": round(chain["chain_score"], 3),
            "mean_confidence": round(chain["mean_confidence"], 3),
            "minimum_confidence": round(chain["minimum_confidence"], 3),
            "knee_extension_angle": round(chain["knee_extension_angle"], 2),
            "knee_line_distance_ratio": round(chain["perpendicular_ratio"], 4),
            "knee_projection": round(chain["projection"], 4),
            "same_coco_side": bool(chain["same_coco_side"]),
            "source_indices": {
                "hip": int(chain["hip_idx"]),
                "knee": int(chain["knee_idx"]),
                "ankle": int(chain["ankle_idx"]),
            },
        })

    return {
        "score": round(_score_from_reference_bands(final_angle), 1),
        "confidence": round(reliability, 3),
        "metrics": {
            "aslr_angle": round(final_angle, 2),
            "requested_side": requested_side,
            "side": requested_side,
            "detected_coco_side": selected_endpoint["ankle_label"].replace("_ANKLE", ""),
            "side_identity_method": "workflow_label_plus_ankle_first_geometry",
            "measurement_engine_version": ASLR_ENGINE_VERSION,
            "angle_method": "common_raised_hip_vertex_ear_or_shoulder_reference_to_true_yolo_ankle",
            "source_orientation_requirement": "none_dual_orientation_pose_detection",
            "reference_axis": "ear_to_selected_hip_or_shoulder_fallback_extended_toward_resting_foot",
            "measurement_vertex_policy": "pink_reference_uses_same_selected_hip_vertex_as_yellow_leg_line",
            "body_baseline": body_baseline_payload,
            "endpoint_source": "true_yolo_ankle_keypoint",
            "endpoint_policy": "ankle_indices_15_or_16_only_no_toe_no_skin_endpoint",
            "heel_keypoint_available": False,
            "heel_note": "The COCO pose model provides ankle, not heel, keypoints.",
            "chain_reconstruction_method": "raised_ankle_first_then_best_hip_knee_combination",
            "measurement_reliability": round(reliability, 3),
            "quality_label": "good" if reliability >= 0.72 and not flags else "moderate",
            "diagnostic_flags": flags,
            "selected_limb": selected_endpoint["ankle_label"],
            "selected_endpoint_type": "ankle",
            "selected_endpoint_confidence": round(selected_endpoint["ankle_confidence"], 3),
            "selected_chain_score": round(selected["chain_score"], 3),
            "selected_limb_mean_confidence": round(selected["mean_confidence"], 3),
            "selected_limb_min_confidence": round(selected["minimum_confidence"], 3),
            "selected_limb_points": selected_points,
            "selected_source_indices": {
                "hip": int(selected["hip_idx"]),
                "knee": int(selected["knee_idx"]),
                "ankle": int(selected["ankle_idx"]),
            },
            "raised_knee_extension_angle": round(knee_extension, 2),
            "raised_knee_line_distance_ratio": round(selected["perpendicular_ratio"], 4),
            "pelvis_center": _rounded_point(pelvis),
            "pelvis_confidence": round(pelvis_confidence, 3),
            "candidate_limbs": public_candidates,
            "endpoint_candidates": [
                {
                    "source_label": endpoint["ankle_label"],
                    "endpoint_index": int(endpoint["ankle_idx"]),
                    "point_type": "ankle",
                    "point": _rounded_point(endpoint["chain"]["ankle"]),
                    "confidence": round(endpoint["ankle_confidence"], 3),
                    "body_relative_angle": round(endpoint["body_relative_angle"], 2),
                }
                for endpoint in endpoint_candidates
            ],
            "detected_ankle_endpoint_count": len(endpoint_candidates),
            "ankle_endpoints_are_distinct": ankles_distinct,
            "ankle_endpoint_separation_px": round(ankle_separation, 2) if ankle_separation is not None else None,
            "skin_shape_endpoint": None,
            "angle_estimators": {
                "body_axis_to_true_ankle": round(final_angle, 2),
                "image_horizontal_to_true_ankle": round(selected_endpoint["horizontal_angle"], 2),
                "spread": round(abs(final_angle - selected_endpoint["horizontal_angle"]), 2),
            },
            "angle_estimators_deg": {
                "body_axis_to_true_ankle": round(final_angle, 2),
                "image_horizontal_to_true_ankle": round(selected_endpoint["horizontal_angle"], 2),
            },
            "quality_gate_config": {
                "keypoint_min_conf": keypoint_min_conf,
                "required_mean_conf": required_mean_conf,
                "raised_knee_extension_min": raised_knee_extension_min,
                "true_ankle_required": True,
                "visual_endpoint_allowed": False,
                "cross_label_chain_reconstruction": True,
            },
            "threshold_evidence_status": ASLR_THRESHOLD_EVIDENCE_STATUS,
        },
        "thresholds": {"aslr_angle": make_aslr_thresholds(final_angle)},
    }






def _angle_between_vectors(
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """Directional angle in degrees from 0 to 180 without folding opposites."""
    na = math.hypot(a[0], a[1])
    nb = math.hypot(b[0], b[1])
    if na <= 1e-6 or nb <= 1e-6:
        return 180.0
    cosine = (a[0] * b[0] + a[1] * b[1]) / (na * nb)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _unit_vector(vector: Tuple[float, float]) -> Tuple[float, float]:
    length = math.hypot(vector[0], vector[1])
    if length <= 1e-6:
        raise ASLRQualityError(
            "reference_axis_uncertain",
            "The body reference axis could not be identified reliably. Retake the photo.",
        )
    return (vector[0] / length, vector[1] / length)


def _raised_leg_chain_candidates(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    pelvis: Tuple[float, float],
    pelvis_confidence: float,
    torso_vector: Tuple[float, float],
    *,
    keypoint_min_conf: float,
) -> list[Dict[str, Any]]:
    """Build plausible pelvis-knee-ankle chains without trusting COCO side labels."""
    torso_length = math.hypot(*torso_vector)

    # Guard against a classic pose-model failure mode: when the resting-leg
    # ankle is occluded, the model still emits a high-confidence keypoint for
    # it, and that keypoint often collapses onto (or very near) the visible
    # raised leg instead of the real resting foot. Two "ankles" that sit
    # implausibly close together are almost certainly one real physical point
    # plus one hallucinated duplicate. In that case we only let the
    # higher-confidence ankle index compete; the low-confidence duplicate is
    # dropped instead of being scored as an independent candidate.
    ankle_min_separation_px = max(20.0, 0.18 * torso_length)
    excluded_ankle_idx: int | None = None
    ankle_conf_15 = _confidence(conf, 15)
    ankle_conf_16 = _confidence(conf, 16)
    if ankle_conf_15 >= keypoint_min_conf and ankle_conf_16 >= keypoint_min_conf:
        ankle_gap = _distance(_point(xy, 15), _point(xy, 16))
        if ankle_gap < ankle_min_separation_px:
            excluded_ankle_idx = 16 if ankle_conf_15 >= ankle_conf_16 else 15

    candidates: list[Dict[str, Any]] = []
    for knee_idx in (13, 14):
        for ankle_idx in (15, 16):
            if ankle_idx == excluded_ankle_idx:
                continue
            knee_conf = _confidence(conf, knee_idx)
            ankle_conf = _confidence(conf, ankle_idx)
            if ankle_conf < keypoint_min_conf:
                continue
            if knee_conf < max(0.10, keypoint_min_conf * 0.60):
                continue

            knee = _point(xy, knee_idx)
            ankle = _point(xy, ankle_idx)
            geometry = _projection_geometry(pelvis, knee, ankle)
            leg_length = float(geometry["leg_length"])
            projection = float(geometry["projection"])
            perpendicular_ratio = float(geometry["perpendicular_ratio"])
            if leg_length < 50.0 or projection < -0.12 or projection > 1.12:
                continue

            thigh_length = _distance(pelvis, knee)
            shank_length = _distance(knee, ankle)
            if thigh_length <= 2.0 or shank_length <= 2.0:
                continue

            # A true straight-leg chain has pelvis->ankle roughly equal to
            # thigh + shank. If the "ankle" actually belongs to a different,
            # nearby landmark (e.g. a hallucinated duplicate near the knee),
            # this ratio collapses well below 1 even when the perpendicular
            # and projection checks above happen to pass. Reject those chains
            # outright rather than letting them compete on score.
            segment_sum = thigh_length + shank_length
            length_consistency_ratio = leg_length / max(segment_sum, 1e-6)
            if length_consistency_ratio < 0.80:
                continue

            knee_extension = _joint_angle(pelvis, knee, ankle)
            segment_ratio = thigh_length / max(shank_length, 1e-6)
            mean_confidence = (pelvis_confidence + knee_conf + ankle_conf) / 3.0
            minimum_confidence = min(pelvis_confidence, knee_conf, ankle_conf)
            straightness = _clamp((knee_extension - 112.0) / 68.0)
            line_alignment = _clamp(1.0 - perpendicular_ratio / 0.27)
            projection_quality = _clamp(1.0 - abs(projection - 0.50) / 0.60)
            ratio_quality = _clamp(
                1.0 - abs(math.log(max(segment_ratio, 1e-6))) / math.log(4.2)
            )
            chain_score = (
                mean_confidence * 0.34
                + straightness * 0.27
                + line_alignment * 0.22
                + projection_quality * 0.09
                + ratio_quality * 0.08
            )
            available = (
                knee_extension >= 120.0
                and perpendicular_ratio <= 0.27
                and -0.08 <= projection <= 1.08
                and 0.22 <= segment_ratio <= 4.20
            )
            leg_vector = _vector(pelvis, ankle)
            torso_separation = _acute_angle_between_vectors(torso_vector, leg_vector)
            directional_torso_alignment = _angle_between_vectors(torso_vector, leg_vector)
            same_coco_side = (
                (knee_idx, ankle_idx) == (13, 15)
                or (knee_idx, ankle_idx) == (14, 16)
            )
            raised_score = (
                chain_score * 0.60
                + _clamp((torso_separation - 10.0) / 75.0) * 0.36
                + (0.04 if same_coco_side else 0.0)
            )
            candidates.append({
                "hip_idx": None,
                "knee_idx": knee_idx,
                "ankle_idx": ankle_idx,
                "hip": pelvis,
                "knee": knee,
                "ankle": ankle,
                "keypoint_confidence": {
                    "hip": pelvis_confidence,
                    "knee": knee_conf,
                    "ankle": ankle_conf,
                },
                "minimum_confidence": minimum_confidence,
                "mean_confidence": mean_confidence,
                "knee_extension_angle": knee_extension,
                "thigh_length_px": thigh_length,
                "shank_length_px": shank_length,
                "thigh_to_shank_ratio": segment_ratio,
                "projection": projection,
                "perpendicular_ratio": perpendicular_ratio,
                "chain_score": chain_score,
                "unbiased_chain_score": chain_score,
                "same_coco_side": same_coco_side,
                "length_consistency_ratio": length_consistency_ratio,
                "available": available,
                "source_label": f"PELVIS-K{knee_idx}-A{ankle_idx}",
                "torso_separation_angle": torso_separation,
                "torso_alignment_angle": directional_torso_alignment,
                "raised_selection_score": raised_score,
            })

    # Keep only the best knee reconstruction for each true YOLO ankle endpoint.
    best_by_ankle: Dict[int, Dict[str, Any]] = {}
    for candidate in candidates:
        if not candidate["available"]:
            continue
        ankle_idx = int(candidate["ankle_idx"])
        previous = best_by_ankle.get(ankle_idx)
        if previous is None or candidate["raised_selection_score"] > previous["raised_selection_score"]:
            best_by_ankle[ankle_idx] = candidate
    return list(best_by_ankle.values())


def _build_body_axis_reference(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    pelvis: Tuple[float, float],
    pelvis_confidence: float,
    shoulder_center: Tuple[float, float],
    torso_vector: Tuple[float, float],
    raised: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the single primary ASLR reference from shoulders to pelvis.

    The body axis is always used for the final angle. Floor-leg landmarks never
    replace it. The vector points from the shoulder midpoint toward the pelvis,
    which is toward the resting-leg/floor direction after inverse mapping.
    """
    torso_unit = _unit_vector(torso_vector)
    torso_length = math.hypot(*torso_vector)
    raised_length = max(80.0, _distance(pelvis, raised["ankle"]))
    display_length = max(135.0, raised_length * 0.96, torso_length * 1.70)
    line_end = (
        pelvis[0] + torso_unit[0] * display_length,
        pelvis[1] + torso_unit[1] * display_length,
    )
    shoulder_conf_values = [
        _confidence(conf, idx) for idx in (5, 6) if _confidence(conf, idx) >= 0.10
    ]
    shoulder_confidence = (
        sum(shoulder_conf_values) / len(shoulder_conf_values)
        if shoulder_conf_values
        else 0.0
    )
    reference_confidence = min(pelvis_confidence, shoulder_confidence)
    length_quality = _clamp((torso_length - 35.0) / 100.0)
    quality = reference_confidence * 0.82 + length_quality * 0.18
    return {
        "source": "body_axis_primary",
        "reference_vector": torso_vector,
        "line_start": pelvis,
        "line_end": line_end,
        "reference_point": shoulder_center,
        "reference_index": None,
        "reference_confidence": reference_confidence,
        "alignment_angle": 0.0,
        "quality": quality,
        "anchors_used": [
            "shoulder_midpoint",
            "shared_pelvic_anchor",
            "body_axis_extension",
        ],
        "source_fit_line_start": shoulder_center,
        "source_fit_line_end": pelvis,
    }


def _observe_optional_resting_leg(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    pelvis: Tuple[float, float],
    torso_vector: Tuple[float, float],
    raised: Mapping[str, Any],
    *,
    keypoint_min_conf: float,
) -> Dict[str, Any]:
    """Observe the floor leg without allowing it to influence the angle.

    Only a strict, anatomically coherent remaining knee/ankle pair is exposed as
    a validation signal. Wrong or missing floor landmarks are ignored rather
    than being drawn or used for measurement.
    """
    raised_ankle_idx = int(raised["ankle_idx"])
    raised_knee_idx = int(raised["knee_idx"])
    remaining_ankles = [idx for idx in (15, 16) if idx != raised_ankle_idx]
    remaining_knees = [idx for idx in (13, 14) if idx != raised_knee_idx]

    best: Dict[str, Any] | None = None
    for knee_idx in remaining_knees:
        knee_conf = _confidence(conf, knee_idx)
        if knee_conf < max(0.16, keypoint_min_conf * 0.80):
            continue
        knee = _point(xy, knee_idx)
        knee_vector = _vector(pelvis, knee)
        knee_length = math.hypot(*knee_vector)
        knee_alignment = _angle_between_vectors(torso_vector, knee_vector)

        # Knee-only validation is accepted only when it closely follows the
        # body axis. It is never used as the measurement reference.
        if knee_length >= 35.0 and knee_alignment <= 11.0:
            knee_quality = (
                knee_conf * 0.62
                + _clamp(1.0 - knee_alignment / 11.0) * 0.38
            )
            candidate = {
                "verified": True,
                "source": "resting_knee_optional_validation",
                "knee": knee,
                "knee_idx": knee_idx,
                "knee_confidence": knee_conf,
                "ankle": None,
                "ankle_idx": None,
                "ankle_confidence": None,
                "knee_extension_angle": None,
                "alignment_angle": knee_alignment,
                "quality": knee_quality,
            }
            if best is None or candidate["quality"] > best["quality"]:
                best = candidate

        for ankle_idx in remaining_ankles:
            ankle_conf = _confidence(conf, ankle_idx)
            if ankle_conf < keypoint_min_conf:
                continue
            ankle = _point(xy, ankle_idx)
            geometry = _projection_geometry(pelvis, knee, ankle)
            leg_length = float(geometry["leg_length"])
            projection = float(geometry["projection"])
            perpendicular_ratio = float(geometry["perpendicular_ratio"])
            if leg_length < 55.0:
                continue
            knee_extension = _joint_angle(pelvis, knee, ankle)
            ankle_alignment = _angle_between_vectors(
                torso_vector,
                _vector(pelvis, ankle),
            )
            thigh_length = _distance(pelvis, knee)
            shank_length = _distance(knee, ankle)
            segment_ratio = thigh_length / max(shank_length, 1e-6)
            coherent = (
                ankle_alignment <= 12.0
                and knee_alignment <= 14.0
                and knee_extension >= 142.0
                and perpendicular_ratio <= 0.14
                and 0.02 <= projection <= 1.06
                and 0.28 <= segment_ratio <= 3.50
            )
            if not coherent:
                continue
            mean_confidence = (knee_conf + ankle_conf) / 2.0
            quality = (
                mean_confidence * 0.42
                + _clamp(1.0 - ankle_alignment / 12.0) * 0.24
                + _clamp((knee_extension - 142.0) / 38.0) * 0.20
                + _clamp(1.0 - perpendicular_ratio / 0.14) * 0.14
            )
            candidate = {
                "verified": True,
                "source": "resting_ankle_optional_validation",
                "knee": knee,
                "knee_idx": knee_idx,
                "knee_confidence": knee_conf,
                "ankle": ankle,
                "ankle_idx": ankle_idx,
                "ankle_confidence": ankle_conf,
                "knee_extension_angle": knee_extension,
                "alignment_angle": ankle_alignment,
                "perpendicular_ratio": perpendicular_ratio,
                "projection": projection,
                "quality": quality,
            }
            if best is None or candidate["quality"] > best["quality"]:
                best = candidate

    if best is not None:
        return best
    return {
        "verified": False,
        "source": "unavailable",
        "knee": None,
        "knee_idx": None,
        "knee_confidence": None,
        "ankle": None,
        "ankle_idx": None,
        "ankle_confidence": None,
        "knee_extension_angle": None,
        "alignment_angle": None,
        "quality": 0.0,
    }


def analyze_aslr_rotated_fullbody(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    side: str = "RIGHT",
    *,
    keypoint_min_conf: float = 0.20,
    required_mean_conf: float = 0.35,
    raised_knee_extension_min: float = 145.0,
    resting_knee_extension_min: float = 145.0,
    **_: Any,
) -> Dict[str, Any]:
    """ASLR measured against one invariant shoulder-to-pelvis body axis.

    The full image is rotated clockwise only for YOLO inference, then keypoints
    are inverse-mapped to the original horizontal photo. The final angle always
    uses the shoulder-midpoint-to-pelvis axis and a true raised YOLO ankle. The
    floor leg is optional validation only and can never change the score.
    """
    if len(xy) < 17 or len(conf) < 17:
        raise ASLRQualityError(
            "missing_pose",
            "Keep the shoulders, pelvis and complete raised leg visible, then retake the photo.",
        )

    requested_side = str(side or "RIGHT").upper()
    if requested_side not in {"LEFT", "RIGHT"}:
        requested_side = "RIGHT"

    shoulder_ids = [i for i in (5, 6) if _confidence(conf, i) >= 0.10]
    hip_ids = [i for i in (11, 12) if _confidence(conf, i) >= 0.08]
    if not shoulder_ids or not hip_ids:
        raise ASLRQualityError(
            "full_body_not_detected",
            "The rotated full-body skeleton was incomplete. Keep the shoulders, pelvis and raised leg visible, then retake the photo.",
        )

    shoulder_weight = sum(_confidence(conf, i) for i in shoulder_ids)
    shoulder_center = (
        sum(_point(xy, i)[0] * _confidence(conf, i) for i in shoulder_ids)
        / shoulder_weight,
        sum(_point(xy, i)[1] * _confidence(conf, i) for i in shoulder_ids)
        / shoulder_weight,
    )
    pelvis, pelvis_confidence = _pelvis_center(xy, conf)
    torso_vector = _vector(shoulder_center, pelvis)
    torso_length = math.hypot(*torso_vector)
    if torso_length < 35.0:
        raise ASLRQualityError(
            "torso_axis_uncertain",
            "The torso axis could not be identified reliably. Retake the photo with the shoulders and pelvis visible.",
        )

    raised_candidates = _raised_leg_chain_candidates(
        xy,
        conf,
        pelvis,
        pelvis_confidence,
        torso_vector,
        keypoint_min_conf=keypoint_min_conf,
    )
    if not raised_candidates:
        raise ASLRQualityError(
            "raised_ankle_or_knee_not_detected",
            "The raised ankle and knee were not identified reliably. Retake the photo with the complete raised foot visible.",
        )

    raised_candidates.sort(key=lambda item: item["raised_selection_score"], reverse=True)
    raised = raised_candidates[0]
    if float(raised["torso_separation_angle"]) < 15.0:
        raise ASLRQualityError(
            "raised_leg_not_separated",
            "The raised leg was not separated clearly from the body axis. Retake the photo.",
        )
    if float(raised["knee_extension_angle"]) < max(125.0, raised_knee_extension_min - 20.0):
        raise ASLRQualityError(
            "raised_knee_geometry_uncertain",
            "The raised pelvis, knee and ankle do not form a reliable leg chain. Straighten the knee and retake the photo.",
            {"raised_knee_extension_angle": round(float(raised["knee_extension_angle"]), 2)},
        )

    raised_margin = None
    if len(raised_candidates) > 1:
        raised_margin = float(
            raised_candidates[0]["raised_selection_score"]
            - raised_candidates[1]["raised_selection_score"]
        )
        separation_gap = abs(
            float(raised_candidates[0]["torso_separation_angle"])
            - float(raised_candidates[1]["torso_separation_angle"])
        )
        if raised_margin < 0.012 and separation_gap < 6.0:
            raise ASLRQualityError(
                "raised_chain_ambiguous",
                "The model detected more than one plausible raised-leg chain. Retake the photo with the raised knee and ankle clearly visible.",
                {
                    "raised_chain_margin": round(raised_margin, 4),
                    "separation_gap_deg": round(separation_gap, 2),
                },
            )

    reference = _build_body_axis_reference(
        xy,
        conf,
        pelvis,
        pelvis_confidence,
        shoulder_center,
        torso_vector,
        raised,
    )
    reference_conf = float(reference["reference_confidence"])
    if reference_conf < max(0.15, keypoint_min_conf * 0.75):
        raise ASLRQualityError(
            "body_axis_low_confidence",
            "The shoulder-to-pelvis body axis is not clear enough. Keep the shoulders and pelvis visible and retake the photo.",
        )

    raised_vector = _vector(pelvis, raised["ankle"])
    final_angle = _acute_angle_between_vectors(
        reference["reference_vector"],
        raised_vector,
    )
    if final_angle < 15.0:
        raise ASLRQualityError(
            "raised_leg_not_separated",
            "The raised leg was not separated clearly from the body axis. Retake the photo.",
        )

    raised_mean_conf = float(raised["mean_confidence"])
    mean_conf = (raised_mean_conf + reference_conf) / 2.0
    if mean_conf < required_mean_conf:
        raise ASLRQualityError(
            "lower_body_low_confidence",
            "The raised-leg and body-axis landmarks are not clear enough. Improve the lighting and retake the photo.",
        )

    overlay_angle = _acute_angle_between_vectors(
        _vector(reference["line_start"], reference["line_end"]),
        raised_vector,
    )
    consistency_error = abs(final_angle - overlay_angle)
    if consistency_error > 0.05:
        raise ASLRQualityError(
            "angle_overlay_geometry_mismatch",
            "The selected landmarks were internally inconsistent. Retake the photo.",
            {"consistency_error_deg": round(consistency_error, 3)},
        )

    resting_observation = _observe_optional_resting_leg(
        xy,
        conf,
        pelvis,
        torso_vector,
        raised,
        keypoint_min_conf=keypoint_min_conf,
    )

    reliability = _clamp(
        raised_mean_conf * 0.43
        + float(raised["unbiased_chain_score"]) * 0.31
        + reference_conf * 0.20
        + float(reference["quality"]) * 0.06
    )

    resting_knee_angle = resting_observation.get("knee_extension_angle")
    flags: list[str] = ["body_axis_primary_reference_used"]
    if bool(resting_observation["verified"]):
        flags.append("resting_leg_optional_validation_available")
    else:
        flags.append("resting_leg_optional_validation_unavailable")
    if resting_knee_angle is not None and float(resting_knee_angle) < resting_knee_extension_min:
        flags.append("resting_knee_below_preferred_extension_advisory")
    if float(raised["knee_extension_angle"]) < raised_knee_extension_min:
        flags.append("raised_knee_below_preferred_extension")
    if not bool(raised["same_coco_side"]):
        flags.append("coco_left_right_labels_ignored_for_raised_chain")

    body_baseline_payload = {
        "method": "shoulder_midpoint_to_shared_pelvic_anchor_body_axis_primary",
        "side": "BODY_AXIS_PRIMARY",
        "ear": None,
        "shoulder": _rounded_point(shoulder_center),
        "pelvis": _rounded_point(pelvis),
        "reference_origin": _rounded_point(shoulder_center),
        "reference_origin_label": "shoulder midpoint",
        "source_fit_line_start": _rounded_point(reference["source_fit_line_start"]),
        "source_fit_line_end": _rounded_point(reference["source_fit_line_end"]),
        "line_start": _rounded_point(reference["line_start"]),
        "line_end": _rounded_point(reference["line_end"]),
        "measurement_vertex": _rounded_point(pelvis),
        "common_vertex_policy": "single_shared_side_view_pelvic_anchor",
        "reference_source": reference["source"],
        "distal_reference_point": _rounded_point(shoulder_center),
        "distal_reference_index": None,
        "distal_reference_confidence": round(reference_conf, 3),
        "direction": {
            "x": round(float(_unit_vector(reference["reference_vector"])[0]), 6),
            "y": round(float(_unit_vector(reference["reference_vector"])[1]), 6),
        },
        "image_angle_deg": round(
            float(math.degrees(math.atan2(
                reference["reference_vector"][1],
                reference["reference_vector"][0],
            ))),
            2,
        ),
        "confidence": round(reference_conf, 3),
        "collinearity": 1.0,
        "anchors_used": list(reference["anchors_used"]),
        "source_indices": {
            "ear": None,
            "shoulder": list(shoulder_ids),
            "hip": list(hip_ids),
            "knee": None,
            "ankle": None,
        },
    }

    raised_diagnostics = [
        {
            "rank": rank + 1,
            "selection_score": round(float(candidate["raised_selection_score"]), 4),
            "torso_separation_angle": round(float(candidate["torso_separation_angle"]), 2),
            "source_indices": {
                "hip": None,
                "knee": int(candidate["knee_idx"]),
                "ankle": int(candidate["ankle_idx"]),
            },
        }
        for rank, candidate in enumerate(raised_candidates[:2])
    ]

    detected_ankles = [idx for idx in (15, 16) if _confidence(conf, idx) >= keypoint_min_conf]
    ankle_separation = None
    if len(detected_ankles) == 2:
        ankle_separation = _distance(_point(xy, detected_ankles[0]), _point(xy, detected_ankles[1]))

    resting_points = {
        "hip": _rounded_point(pelvis),
        "knee": _rounded_point(resting_observation.get("knee")),
        "ankle": _rounded_point(resting_observation.get("ankle")),
        "reference_endpoint": None,
    }

    return {
        "score": round(_score_from_reference_bands(final_angle), 1),
        "confidence": round(reliability, 3),
        "metrics": {
            "aslr_angle": round(final_angle, 2),
            "requested_side": requested_side,
            "side": requested_side,
            "detected_coco_side": raised["source_label"],
            "side_identity_method": "workflow_label_for_reporting_geometry_for_raised_chain_selection",
            "measurement_engine_version": ASLR_ENGINE_VERSION,
            "angle_method": "angle_between_shoulder_to_pelvis_body_axis_and_shared_pelvic_anchor_to_true_raised_ankle",
            "source_orientation_requirement": "full_image_rotated_90_clockwise_for_inference_then_landmarks_mapped_back",
            "display_orientation": "original_normal_horizontal_image",
            "display_rotation_applied": "90_degrees_anticlockwise_equivalent_via_coordinate_inverse_mapping",
            "reference_axis": "shoulder_midpoint_to_pelvis_body_axis_always_primary",
            "reference_source": reference["source"],
            "resting_reference_fallback_used": False,
            "resting_leg_used_for_measurement": False,
            "resting_leg_verified": bool(resting_observation["verified"]),
            "resting_leg_validation_source": resting_observation["source"],
            "measurement_vertex_policy": "single_shared_pelvic_anchor_for_body_axis_and_raised_leg_vectors",
            "body_baseline": body_baseline_payload,
            "endpoint_source": "true_yolo_raised_ankle_with_body_axis_primary_reference",
            "endpoint_policy": "raised_ankle_index_15_or_16_required_floor_leg_landmarks_optional_validation_only_no_toe_no_skin_endpoint",
            "heel_keypoint_available": False,
            "chain_reconstruction_method": "raised_ankle_first_cross_label_chain_plus_body_axis_primary_reference",
            "coco_side_labels_used_for_pairing": False,
            "requested_side_used_for_pairing": False,
            "measurement_reliability": round(reliability, 3),
            "quality_label": "good" if reliability >= 0.72 else "moderate",
            "diagnostic_flags": flags,
            "selected_limb": "GEOMETRIC_RAISED_LEG",
            "selected_endpoint_type": "ankle",
            "selected_endpoint_confidence": round(float(raised["keypoint_confidence"]["ankle"]), 3),
            "selected_chain_score": round(float(raised["unbiased_chain_score"]), 3),
            "selected_limb_mean_confidence": round(raised_mean_conf, 3),
            "selected_limb_min_confidence": round(float(raised["minimum_confidence"]), 3),
            "selected_limb_points": {
                "pelvis": _rounded_point(pelvis),
                "hip": _rounded_point(pelvis),
                "knee": _rounded_point(raised["knee"]),
                "ankle": _rounded_point(raised["ankle"]),
            },
            "resting_limb_points": resting_points,
            "selected_source_indices": {
                "hip": None,
                "knee": int(raised["knee_idx"]),
                "ankle": int(raised["ankle_idx"]),
            },
            "resting_source_indices": {
                "hip": None,
                "knee": resting_observation.get("knee_idx"),
                "ankle": resting_observation.get("ankle_idx"),
            },
            "raised_knee_extension_angle": round(float(raised["knee_extension_angle"]), 2),
            "resting_knee_extension_angle": (
                round(float(resting_knee_angle), 2) if resting_knee_angle is not None else None
            ),
            "raised_knee_line_distance_ratio": round(float(raised["perpendicular_ratio"]), 4),
            "resting_knee_line_distance_ratio": (
                round(float(resting_observation["perpendicular_ratio"]), 4)
                if resting_observation.get("perpendicular_ratio") is not None
                else None
            ),
            "pelvis_center": _rounded_point(pelvis),
            "pelvis_confidence": round(pelvis_confidence, 3),
            "shoulder_center": _rounded_point(shoulder_center),
            "body_axis_length_px": round(torso_length, 2),
            "detected_ankle_endpoint_count": len(detected_ankles),
            "ankle_endpoints_are_distinct": (
                bool(ankle_separation and ankle_separation >= 20.0)
                if ankle_separation is not None
                else None
            ),
            "ankle_endpoint_separation_px": round(ankle_separation, 2) if ankle_separation is not None else None,
            "skin_shape_endpoint": None,
            "raised_chain_selection_score": round(float(raised["raised_selection_score"]), 4),
            "raised_chain_margin": round(raised_margin, 4) if raised_margin is not None else None,
            "raised_chain_candidates": raised_diagnostics,
            "reference_confidence": round(reference_conf, 3),
            "reference_alignment_angle": 0.0,
            "angle_estimators": {
                "body_axis_to_raised_ankle": round(final_angle, 2),
                "overlay_geometry": round(overlay_angle, 2),
                "spread": round(consistency_error, 3),
            },
            "angle_estimators_deg": {
                "body_axis_to_raised_ankle": round(final_angle, 2),
                "overlay_geometry": round(overlay_angle, 2),
            },
            "quality_gate_config": {
                "keypoint_min_conf": keypoint_min_conf,
                "required_mean_conf": required_mean_conf,
                "raised_knee_extension_min": raised_knee_extension_min,
                "resting_knee_extension_min": resting_knee_extension_min,
                "raised_true_ankle_required": True,
                "resting_true_ankle_required": False,
                "body_axis_primary_for_all_measurements": True,
                "resting_landmarks_used_for_angle": False,
                "resting_landmarks_optional_validation_only": True,
                "full_body_rotated_inference_required": True,
                "shared_pelvic_anchor_required": True,
                "coco_side_labels_ignored": True,
                "overlay_angle_consistency_max_error_deg": 0.05,
            },
            "threshold_evidence_status": ASLR_THRESHOLD_EVIDENCE_STATUS,
        },
        "thresholds": {"aslr_angle": make_aslr_thresholds(final_angle)},
    }

