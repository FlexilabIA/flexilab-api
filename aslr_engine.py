"""FlexiLab ASLR endpoint-first measurement engine.

V101.28.3 keeps endpoint-first chain reconstruction but measures from one
stable pelvic anchor and supports dual-orientation pose selection in the caller.
of trusting COCO left/right chains. This is designed for side-view supine ASLR
where bilateral hip/knee labels can cross or swap while both ankles remain
visually distinct.

Source orientation is irrelevant. Portrait, landscape, square and slightly
tilted desktop-camera images are valid when the required anatomy is visible.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple


ASLR_ENGINE_VERSION = "aslr-dedicated-yolo11m-geometry-side-agnostic-v8"
ASLR_THRESHOLD_EVIDENCE_STATUS = (
    "provisional_flexilab_reference_bands_not_diagnostic_cutoffs"
)


class ASLRQualityError(ValueError):
    """Controlled capture rejection that is safe to show to the client."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ):
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


def _vector(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (b[0] - a[0], b[1] - a[1])


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _acute_angle_between(
    first: Tuple[float, float],
    second: Tuple[float, float],
) -> float:
    """Return the unsigned acute angle between two axes, in [0°, 90°]."""
    n1 = math.hypot(*first)
    n2 = math.hypot(*second)
    if n1 <= 1e-6 or n2 <= 1e-6:
        return 0.0
    cosine = max(
        -1.0,
        min(1.0, (first[0] * second[0] + first[1] * second[1]) / (n1 * n2)),
    )
    angle = math.degrees(math.acos(cosine))
    return min(angle, 180.0 - angle)


def _joint_angle(
    a: Tuple[float, float],
    vertex: Tuple[float, float],
    c: Tuple[float, float],
) -> float:
    """Included angle at vertex; a straight knee is approximately 180°."""
    v1 = (a[0] - vertex[0], a[1] - vertex[1])
    v2 = (c[0] - vertex[0], c[1] - vertex[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 <= 1e-6 or n2 <= 1e-6:
        return 0.0
    cosine = max(
        -1.0,
        min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)),
    )
    return math.degrees(math.acos(cosine))


def _rounded_point(point: Tuple[float, float]) -> Dict[str, float]:
    return {"x": round(float(point[0]), 2), "y": round(float(point[1]), 2)}


ASLR_RED_MAX_DEG = 60.0
ASLR_YELLOW_MAX_DEG = 75.0
ASLR_SCALE_MAX_DEG = 90.0


def make_aslr_thresholds(value: float) -> Dict[str, Any]:
    """Return the current FlexiLab ASLR reference bands.

    Boundary policy:
    - red: angle < 60°
    - yellow: 60° <= angle <= 75°
    - green: angle > 75°

    The three zones are intentionally rendered as equal visual thirds by the
    frontend, even though their numeric spans differ.
    """
    value = max(0.0, min(ASLR_SCALE_MAX_DEG, float(value)))
    bands = [
        {"label": "Red", "min": 0, "max": 60, "color": "red"},
        {"label": "Yellow", "min": 60, "max": 75, "color": "yellow"},
        {"label": "Green", "min": 75, "max": 90, "color": "green"},
    ]
    rating = "red" if value < ASLR_RED_MAX_DEG else "yellow" if value <= ASLR_YELLOW_MAX_DEG else "green"
    return {
        "unit": "deg",
        "scale_min": 0,
        "scale_max": 90,
        "bands": bands,
        "pointer_value": round(value, 2),
        "rating": rating,
        "visual_band_layout": "equal_thirds",
        "boundary_policy": {
            "red": "<60",
            "yellow": "60-75_inclusive",
            "green": ">75",
        },
    }


def _score_from_reference_bands(angle: float) -> float:
    """Map the updated red/yellow/green bands onto the existing score tiers."""
    angle = max(0.0, min(ASLR_SCALE_MAX_DEG, float(angle)))
    if angle < ASLR_RED_MAX_DEG:
        score = 40.0 + (angle / ASLR_RED_MAX_DEG) * 19.0
    elif angle <= ASLR_YELLOW_MAX_DEG:
        score = 60.0 + ((angle - ASLR_RED_MAX_DEG) / (ASLR_YELLOW_MAX_DEG - ASLR_RED_MAX_DEG)) * 19.0
    else:
        score = 85.0 + ((angle - ASLR_YELLOW_MAX_DEG) / (ASLR_SCALE_MAX_DEG - ASLR_YELLOW_MAX_DEG)) * 15.0
    return max(0.0, min(100.0, score))


def _weighted_center(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    indices: Sequence[int],
    minimum_confidence: float,
) -> Tuple[Tuple[float, float] | None, float]:
    values = []
    for index in indices:
        confidence = _confidence(conf, index)
        if confidence >= minimum_confidence:
            values.append((_point(xy, index), confidence))
    if not values:
        return None, 0.0
    weight_sum = sum(weight for _, weight in values)
    center = (
        sum(point[0] * weight for point, weight in values) / weight_sum,
        sum(point[1] * weight for point, weight in values) / weight_sum,
    )
    return center, sum(weight for _, weight in values) / len(values)


def _body_axis(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    minimum_confidence: float,
) -> Dict[str, Any] | None:
    shoulder_center, shoulder_conf = _weighted_center(
        xy, conf, (5, 6), minimum_confidence
    )
    pelvis_center, pelvis_conf = _weighted_center(
        xy, conf, (11, 12), minimum_confidence
    )
    if shoulder_center is None or pelvis_center is None:
        return None
    vector = _vector(shoulder_center, pelvis_center)
    length = math.hypot(*vector)
    if length <= 5.0:
        return None
    return {
        "shoulder": shoulder_center,
        "pelvis": pelvis_center,
        "vector": vector,
        "confidence": (shoulder_conf + pelvis_conf) / 2.0,
        "length_px": length,
    }


def _projection_geometry(
    hip: Tuple[float, float],
    knee: Tuple[float, float],
    ankle: Tuple[float, float],
) -> Dict[str, float]:
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
    projection = (knee_relative[0] * leg[0] + knee_relative[1] * leg[1]) / length_sq
    closest = (hip[0] + projection * leg[0], hip[1] + projection * leg[1])
    perpendicular_ratio = _distance(knee, closest) / leg_length
    return {
        "projection": projection,
        "perpendicular_ratio": perpendicular_ratio,
        "leg_length": leg_length,
    }


def _chain_candidate(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    hip_idx: int,
    knee_idx: int,
    ankle_idx: int,
    pelvis_center: Tuple[float, float],
    *,
    keypoint_min_conf: float,
    reused_knee_idx: int | None = None,
    reused_hip_idx: int | None = None,
) -> Dict[str, Any] | None:
    hip = _point(xy, hip_idx)
    knee = _point(xy, knee_idx)
    ankle = _point(xy, ankle_idx)
    confidences = {
        "hip": _confidence(conf, hip_idx),
        "knee": _confidence(conf, knee_idx),
        "ankle": _confidence(conf, ankle_idx),
    }
    if confidences["ankle"] < keypoint_min_conf:
        return None
    if confidences["knee"] < max(0.12, keypoint_min_conf * 0.75):
        return None
    if confidences["hip"] < max(0.10, keypoint_min_conf * 0.60):
        return None

    thigh_length = _distance(hip, knee)
    shank_length = _distance(knee, ankle)
    if thigh_length <= 2.0 or shank_length <= 2.0:
        return None

    geometry = _projection_geometry(hip, knee, ankle)
    projection = geometry["projection"]
    perpendicular_ratio = geometry["perpendicular_ratio"]
    if projection < -0.10 or projection > 1.10:
        return None

    knee_extension = _joint_angle(hip, knee, ankle)
    segment_ratio = thigh_length / max(shank_length, 1e-6)
    confidence_score = sum(confidences.values()) / 3.0
    straightness_score = _clamp((knee_extension - 115.0) / 65.0)
    projection_score = _clamp(1.0 - abs(projection - 0.5) / 0.55)
    line_score = _clamp(1.0 - perpendicular_ratio / 0.24)
    ratio_score = _clamp(1.0 - abs(math.log(max(segment_ratio, 1e-6))) / math.log(3.5))
    hip_distance = _distance(hip, pelvis_center)
    hip_score = _clamp(1.0 - hip_distance / max(20.0, geometry["leg_length"] * 0.30))

    score = (
        confidence_score * 0.30
        + straightness_score * 0.25
        + line_score * 0.20
        + projection_score * 0.10
        + ratio_score * 0.09
        + hip_score * 0.06
    )

    same_coco_side = (
        (hip_idx, knee_idx, ankle_idx) == (11, 13, 15)
        or (hip_idx, knee_idx, ankle_idx) == (12, 14, 16)
    )
    if same_coco_side:
        score += 0.015
    if reused_knee_idx is not None and knee_idx == reused_knee_idx:
        score -= 0.22
    if reused_hip_idx is not None and hip_idx == reused_hip_idx:
        # Overlapping side-view hips are common, so this is only a mild penalty.
        score -= 0.035

    return {
        "hip_idx": hip_idx,
        "knee_idx": knee_idx,
        "ankle_idx": ankle_idx,
        "hip": hip,
        "knee": knee,
        "ankle": ankle,
        "keypoint_confidence": confidences,
        "minimum_confidence": min(confidences.values()),
        "mean_confidence": confidence_score,
        "knee_extension_angle": knee_extension,
        "thigh_length_px": thigh_length,
        "shank_length_px": shank_length,
        "thigh_to_shank_ratio": segment_ratio,
        "projection": projection,
        "perpendicular_ratio": perpendicular_ratio,
        "chain_score": score,
        "same_coco_side": same_coco_side,
    }


def _best_chain_for_ankle(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    ankle_idx: int,
    pelvis_center: Tuple[float, float],
    *,
    keypoint_min_conf: float,
    reused_knee_idx: int | None = None,
    reused_hip_idx: int | None = None,
) -> Dict[str, Any] | None:
    candidates = []
    for hip_idx in (11, 12):
        for knee_idx in (13, 14):
            candidate = _chain_candidate(
                xy,
                conf,
                hip_idx,
                knee_idx,
                ankle_idx,
                pelvis_center,
                keypoint_min_conf=keypoint_min_conf,
                reused_knee_idx=reused_knee_idx,
                reused_hip_idx=reused_hip_idx,
            )
            if candidate is not None:
                candidates.append(candidate)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["chain_score"])


def _public_chain(label: str, chain: Mapping[str, Any], angle: float) -> Dict[str, Any]:
    return {
        "label": label,
        "available": True,
        "body_relative_angle": round(float(angle), 2),
        "knee_extension_angle": round(float(chain["knee_extension_angle"]), 2),
        "minimum_confidence": round(float(chain["minimum_confidence"]), 3),
        "mean_confidence": round(float(chain["mean_confidence"]), 3),
        "chain_score": round(float(chain["chain_score"]), 3),
        "thigh_to_shank_ratio": round(float(chain["thigh_to_shank_ratio"]), 3),
        "projection": round(float(chain["projection"]), 3),
        "perpendicular_ratio": round(float(chain["perpendicular_ratio"]), 3),
        "keypoint_confidence": {
            key: round(float(value), 3)
            for key, value in chain["keypoint_confidence"].items()
        },
        "source_indices": {
            "hip": int(chain["hip_idx"]),
            "knee": int(chain["knee_idx"]),
            "ankle": int(chain["ankle_idx"]),
        },
        "points": {
            "hip": _rounded_point(chain["hip"]),
            "knee": _rounded_point(chain["knee"]),
            "ankle": _rounded_point(chain["ankle"]),
        },
    }


def analyze_aslr_v2(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    side: str = "RIGHT",
    *,
    keypoint_min_conf: float = 0.20,
    required_mean_conf: float = 0.35,
    raised_knee_extension_min: float = 155.0,
    resting_knee_extension_min: float = 150.0,
    resting_leg_max_angle: float = 20.0,
    ambiguous_angle_gap: float = 6.0,
    minimum_detectable_raise: float = 8.0,
    torso_min_conf: float = 0.15,
) -> Dict[str, Any]:
    """Measure ASLR by selecting the raised ankle first, then rebuilding its chain.

    COCO left/right labels are not trusted to define the limb. Each visible ankle
    is evaluated relative to the body axis, then connected to the most anatomically
    plausible knee and hip from all lower-limb keypoints.
    """

    if len(xy) < 17 or len(conf) < 17:
        raise ASLRQualityError(
            "missing_pose",
            "We couldn’t measure this position clearly. Keep the pelvis, both legs and both feet visible, then retake the photo.",
        )

    requested_side = str(side or "RIGHT").strip().upper()
    if requested_side not in {"LEFT", "RIGHT"}:
        requested_side = "RIGHT"

    body_axis = _body_axis(xy, conf, torso_min_conf)
    pelvis_center, pelvis_confidence = _weighted_center(
        xy, conf, (11, 12), max(0.10, torso_min_conf * 0.75)
    )
    if pelvis_center is None:
        raise ASLRQualityError(
            "pelvis_not_detected",
            "We couldn’t measure this position clearly. Keep the pelvis and both legs visible, then retake the photo.",
        )

    flags = []
    if body_axis is not None:
        torso_vector = body_axis["vector"]
        baseline_method = "shoulder_center_to_pelvis_center_axis"
        baseline_confidence = float(body_axis["confidence"])
    else:
        torso_vector = (1.0, 0.0)
        baseline_method = "image_horizontal_fallback"
        baseline_confidence = 0.0
        flags.append("body_axis_unavailable_image_horizontal_used")

    # Raised ankles are sometimes assigned lower confidence than the resting
    # ankle in strict side-view images, particularly after a COCO left/right
    # label swap. Use a relaxed endpoint threshold, then require the complete
    # reconstructed chain to satisfy the normal mean-confidence and geometry
    # gates. This prevents a valid raised leg from disappearing before chain
    # reconstruction while still rejecting isolated noisy ankle points.
    endpoint_min_conf = max(0.08, keypoint_min_conf * 0.45)
    endpoints = []
    for ankle_idx, coco_label in ((15, "COCO_LEFT_ANKLE"), (16, "COCO_RIGHT_ANKLE")):
        ankle_confidence = _confidence(conf, ankle_idx)
        if ankle_confidence < endpoint_min_conf:
            continue
        ankle = _point(xy, ankle_idx)
        vector = _vector(pelvis_center, ankle)
        length = math.hypot(*vector)
        if length <= 5.0:
            continue
        chain = _best_chain_for_ankle(
            xy,
            conf,
            ankle_idx,
            pelvis_center,
            keypoint_min_conf=endpoint_min_conf,
        )
        endpoints.append(
            {
                "ankle_idx": ankle_idx,
                "coco_label": coco_label,
                "ankle": ankle,
                "ankle_confidence": ankle_confidence,
                "vector": vector,
                "length_px": length,
                "torso_relative_angle": _acute_angle_between(torso_vector, vector),
                "chain": chain,
            }
        )

    if not endpoints:
        raise ASLRQualityError(
            "required_landmarks_low_confidence",
            "We couldn’t measure the raised leg clearly. Keep both feet visible and retake the photo in good lighting.",
            {"missing_region": "ankles"},
        )

    endpoints.sort(
        key=lambda item: (
            item["torso_relative_angle"],
            item["chain"]["chain_score"] if item["chain"] else -1.0,
            item["ankle_confidence"],
        ),
        reverse=True,
    )
    raised_endpoint = endpoints[0]
    resting_endpoint = endpoints[1] if len(endpoints) > 1 else None

    if raised_endpoint["chain"] is None:
        raise ASLRQualityError(
            "raised_chain_not_reconstructed",
            "We couldn’t measure the full raised leg clearly. Keep the hip, knee and foot visible, then retake the photo.",
            {
                "raised_ankle": _rounded_point(raised_endpoint["ankle"]),
                "raised_ankle_confidence": round(raised_endpoint["ankle_confidence"], 3),
            },
        )

    raised_chain = raised_endpoint["chain"]
    if raised_chain["mean_confidence"] < required_mean_conf:
        raise ASLRQualityError(
            "raised_limb_low_confidence",
            "The raised leg is not clear enough for a reliable measurement. Improve the lighting and retake the photo.",
            {"raised_chain": _public_chain("RAISED_ENDPOINT_CHAIN", raised_chain, raised_endpoint["torso_relative_angle"])},
        )

    if raised_chain["chain_score"] < 0.46:
        raise ASLRQualityError(
            "raised_chain_geometry_uncertain",
            "We found the raised foot, but the knee-to-hip connection is uncertain. Keep the complete raised leg visible and avoid overlapping objects.",
            {"raised_chain": _public_chain("RAISED_ENDPOINT_CHAIN", raised_chain, raised_endpoint["torso_relative_angle"])},
        )

    resting_chain = None
    endpoints_are_distinct = False
    endpoint_separation = None
    endpoint_separation_ratio = None
    distinct_threshold = None
    if resting_endpoint is not None:
        endpoint_separation = _distance(raised_endpoint["ankle"], resting_endpoint["ankle"])
        reference_length = max(raised_endpoint["length_px"], resting_endpoint["length_px"], 1.0)
        endpoint_separation_ratio = endpoint_separation / reference_length
        distinct_threshold = max(8.0, 0.08 * reference_length)
        endpoints_are_distinct = endpoint_separation >= distinct_threshold
        if endpoints_are_distinct:
            resting_chain = _best_chain_for_ankle(
                xy,
                conf,
                resting_endpoint["ankle_idx"],
                pelvis_center,
                keypoint_min_conf=keypoint_min_conf,
                reused_knee_idx=int(raised_chain["knee_idx"]),
                reused_hip_idx=int(raised_chain["hip_idx"]),
            )
        else:
            flags.append("ankle_endpoints_duplicated_by_pose_model")
            flags.append("resting_leg_not_independently_resolved_by_pose_model")

    # A strict side-view ASLR does not provide two visually separable hips.
    # Measurement therefore starts from one confidence-weighted pelvic anchor.
    # V7 never lets a lone ankle define the floor/resting baseline. The complete
    # resting chain must be anatomically verified first; otherwise the stable
    # shoulder-to-pelvis body axis is used.
    raised_measurement_vector = _vector(pelvis_center, raised_chain["ankle"])

    resting_angle = None
    resting_knee_extension = None
    resting_verified = False
    if resting_endpoint is not None and endpoints_are_distinct and resting_chain is not None:
        resting_angle = _acute_angle_between(torso_vector, resting_endpoint["vector"])
        resting_knee_extension = float(resting_chain["knee_extension_angle"])
        resting_verified = (
            float(resting_chain["chain_score"]) >= 0.42
            and float(resting_chain["mean_confidence"]) >= max(0.28, required_mean_conf * 0.80)
            and resting_knee_extension >= resting_knee_extension_min
            and resting_angle <= resting_leg_max_angle
        )

    if resting_verified:
        final_baseline_vector = _vector(pelvis_center, resting_chain["ankle"])
        final_baseline_method = "verified_resting_chain_axis"
    else:
        final_baseline_vector = torso_vector
        final_baseline_method = baseline_method
        flags.append("unverified_resting_chain_body_axis_used")

    raised_knee_extension = float(raised_chain["knee_extension_angle"])
    if raised_knee_extension < raised_knee_extension_min:
        raise ASLRQualityError(
            "raised_knee_bent",
            "The raised knee appears bent. Retake the photo while keeping the raised knee straight.",
            {
                "raised_knee_extension_angle": round(raised_knee_extension, 2),
                "required_minimum": raised_knee_extension_min,
            },
        )

    # Robust consensus from independent anatomical estimators. These remain
    # rotation-invariant because all are measured relative to the body axis.
    body_axis_angle = _acute_angle_between(torso_vector, raised_measurement_vector)
    hip_axis_angle = _acute_angle_between(torso_vector, _vector(raised_chain["hip"], raised_chain["ankle"]))
    thigh_axis_angle = _acute_angle_between(torso_vector, _vector(raised_chain["hip"], raised_chain["knee"]))
    body_estimators = [body_axis_angle, hip_axis_angle, thigh_axis_angle]
    ordered_estimators = sorted(body_estimators)
    body_consensus_angle = ordered_estimators[len(ordered_estimators) // 2]
    body_estimator_spread = max(body_estimators) - min(body_estimators)
    if body_estimator_spread > 12.0:
        raise ASLRQualityError(
            "aslr_geometry_disagreement",
            "We detected the full leg, but the landmarks do not agree on one reliable angle. Please retake the photo without changing your position.",
            {
                "body_axis_angle": round(body_axis_angle, 2),
                "hip_axis_angle": round(hip_axis_angle, 2),
                "thigh_axis_angle": round(thigh_axis_angle, 2),
                "estimator_spread": round(body_estimator_spread, 2),
            },
        )

    resting_reference_angle = None
    if resting_verified:
        resting_reference_angle = _acute_angle_between(final_baseline_vector, raised_measurement_vector)
        if abs(resting_reference_angle - body_consensus_angle) <= 12.0:
            raised_angle = (resting_reference_angle + body_consensus_angle) / 2.0
        else:
            # A verified-looking resting chain can still contain a swapped hip or
            # diagonal ankle. Prefer the internally consistent body estimators.
            raised_angle = body_consensus_angle
            flags.append("resting_reference_disagreed_body_consensus")
    else:
        raised_angle = body_consensus_angle
    # A lone ankle close to the resting/body baseline is almost always the
    # resting ankle, not a severe-mobility result. Reject instead of saving a
    # false low score. A clearly elevated single endpoint remains measurable.
    if not endpoints_are_distinct and raised_angle < 18.0:
        raise ASLRQualityError(
            "raised_ankle_not_detected",
            "We couldn’t measure the raised leg clearly. Keep both feet fully visible, then retake the photo.",
            {
                "detected_endpoint_count": len(endpoints),
                "candidate_angle": round(raised_angle, 2),
                "selected_ankle": _rounded_point(raised_chain["ankle"]),
            },
        )
    if raised_angle < minimum_detectable_raise:
        raise ASLRQualityError(
            "raised_leg_not_detected",
            "Raise one leg to your highest comfortable position, keep the knee straight and retake the photo.",
        )

    candidate_limbs = [
        _public_chain("RAISED_ENDPOINT_CHAIN", raised_chain, raised_angle)
    ]

    if resting_endpoint is not None and endpoints_are_distinct:
        if resting_chain is not None:
            candidate_limbs.append(
                _public_chain("RESTING_ENDPOINT_CHAIN", resting_chain, resting_angle or 0.0)
            )
            if not resting_verified:
                if resting_angle is not None and resting_angle > resting_leg_max_angle:
                    flags.append("resting_leg_angle_above_verification_limit")
                if resting_knee_extension is not None and resting_knee_extension < resting_knee_extension_min:
                    flags.append("resting_knee_below_verification_limit")
                if float(resting_chain["chain_score"]) < 0.42:
                    flags.append("resting_chain_geometry_uncertain")
        else:
            flags.append("resting_chain_not_reconstructed")
    else:
        flags.append("resting_leg_not_fully_verified")

    detected_coco_side = raised_endpoint["coco_label"].replace("_ANKLE", "")
    expected_coco_side = f"COCO_{requested_side}"
    if detected_coco_side != expected_coco_side:
        flags.append("coco_ankle_label_differs_from_workflow_side")
        flags.append("coco_side_differs_from_workflow_side")

    minimum_confidence = float(raised_chain["minimum_confidence"])
    mean_confidence = float(raised_chain["mean_confidence"])
    straightness_quality = _clamp(
        (raised_knee_extension - raised_knee_extension_min)
        / max(1.0, 180.0 - raised_knee_extension_min)
    )
    chain_quality = _clamp((float(raised_chain["chain_score"]) - 0.40) / 0.60)
    confidence_out = (
        minimum_confidence * 0.30
        + mean_confidence * 0.28
        + straightness_quality * 0.14
        + chain_quality * 0.20
        + baseline_confidence * 0.08
    )
    if not resting_verified:
        confidence_out *= 0.86
    if body_axis is None:
        confidence_out *= 0.84
    confidence_out = _clamp(confidence_out)

    quality_label = "good" if confidence_out >= 0.65 and not flags else "moderate"
    score = _score_from_reference_bands(raised_angle)

    baseline_public = {
        "method": final_baseline_method,
        "confidence": round(baseline_confidence, 3),
        "vector": {
            "x": round(float(final_baseline_vector[0]), 2),
            "y": round(float(final_baseline_vector[1]), 2),
        },
    }
    if body_axis is not None:
        baseline_public.update(
            {
                "shoulder": _rounded_point(body_axis["shoulder"]),
                "hip": _rounded_point(body_axis["pelvis"]),
            }
        )

    endpoint_diagnostics = []
    for endpoint in endpoints:
        endpoint_diagnostics.append(
            {
                "source_label": endpoint["coco_label"],
                "ankle_index": endpoint["ankle_idx"],
                "ankle": _rounded_point(endpoint["ankle"]),
                "ankle_confidence": round(endpoint["ankle_confidence"], 3),
                "torso_relative_angle": round(endpoint["torso_relative_angle"], 2),
                "best_chain_score": (
                    round(float(endpoint["chain"]["chain_score"]), 3)
                    if endpoint["chain"] is not None
                    else None
                ),
            }
        )

    return {
        "score": round(float(score), 1),
        "confidence": round(float(confidence_out), 3),
        "metrics": {
            "aslr_angle": round(raised_angle, 2),
            "requested_side": requested_side,
            "side": requested_side,
            "detected_coco_side": detected_coco_side,
            "side_identity_method": "workflow_label_with_geometry_selected_raised_chain",
            "measurement_engine_version": ASLR_ENGINE_VERSION,
            "angle_method": "robust_body_axis_consensus_with_verified_resting_chain_only",
            "chain_reconstruction_method": "raised_ankle_first_then_best_knee_and_hip_combination",
            "endpoint_min_confidence": round(endpoint_min_conf, 3),
            "pelvic_anchor_method": "confidence_weighted_visible_hip_region",
            "source_orientation_requirement": "none",
            "body_baseline": baseline_public,
            "angle_estimators": {
                "body_axis": round(body_axis_angle, 2),
                "hip_to_ankle": round(hip_axis_angle, 2),
                "thigh_axis": round(thigh_axis_angle, 2),
                "body_consensus": round(body_consensus_angle, 2),
                "verified_resting_reference": round(resting_reference_angle, 2) if resting_reference_angle is not None else None,
                "spread": round(body_estimator_spread, 2),
            },
            "measurement_reliability": round(confidence_out, 3),
            "raised_knee_extension_angle": round(raised_knee_extension, 2),
            "resting_leg_angle": round(resting_angle, 2) if resting_angle is not None else None,
            "resting_knee_extension_angle": (
                round(resting_knee_extension, 2)
                if resting_knee_extension is not None
                else None
            ),
            "resting_leg_verified": resting_verified,
            "quality_label": quality_label,
            "diagnostic_flags": flags,
            "selected_limb": "RAISED_ENDPOINT_CHAIN",
            "selected_limb_min_confidence": round(minimum_confidence, 3),
            "selected_limb_mean_confidence": round(mean_confidence, 3),
            "selected_chain_score": round(float(raised_chain["chain_score"]), 3),
            "candidate_limbs": candidate_limbs,
            "endpoint_candidates": endpoint_diagnostics,
            "detected_ankle_endpoint_count": len(endpoints),
            "ankle_endpoints_are_distinct": endpoints_are_distinct,
            "ankle_endpoint_separation_px": round(endpoint_separation, 2) if endpoint_separation is not None else None,
            "ankle_endpoint_separation_ratio": round(endpoint_separation_ratio, 4) if endpoint_separation_ratio is not None else None,
            "ankle_distinct_threshold_px": round(distinct_threshold, 2) if distinct_threshold is not None else None,
            "selected_limb_points": {
                "pelvis": _rounded_point(pelvis_center),
                "hip": _rounded_point(raised_chain["hip"]),
                "knee": _rounded_point(raised_chain["knee"]),
                "ankle": _rounded_point(raised_chain["ankle"]),
            },
            "selected_source_indices": {
                "hip": int(raised_chain["hip_idx"]),
                "knee": int(raised_chain["knee_idx"]),
                "ankle": int(raised_chain["ankle_idx"]),
            },
            "pelvis_center": _rounded_point(pelvis_center),
            "pelvis_confidence": round(float(pelvis_confidence), 3),
            "quality_gate_config": {
                "keypoint_min_conf": keypoint_min_conf,
                "required_mean_conf": required_mean_conf,
                "raised_knee_extension_min": raised_knee_extension_min,
                "resting_knee_extension_min": resting_knee_extension_min,
                "resting_leg_max_angle": resting_leg_max_angle,
            },
            "threshold_evidence_status": ASLR_THRESHOLD_EVIDENCE_STATUS,
        },
        "thresholds": {"aslr_angle": make_aslr_thresholds(raised_angle)},
    }
