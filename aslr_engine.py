"""FlexiLab ASLR geometric measurement engine.

V101.28 makes the measurement independent from source orientation. Portrait,
landscape, square and slightly tilted desktop-camera images are valid when the
required anatomy is visible. The raised-leg angle is measured relative to the
subject's torso/resting-body axis instead of the image border.

The workflow assigns the requested anatomical side. COCO left/right labels are
kept only as diagnostics because bilateral labels are unstable in a side-view
supine pose with overlapping limbs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple


ASLR_ENGINE_VERSION = "aslr-body-relative-geometry-v2"
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


def _image_horizontal_elevation(
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """Diagnostic only: elevation above image horizontal."""
    dx = abs(b[0] - a[0])
    dy = abs(b[1] - a[1])
    return max(0.0, min(90.0, math.degrees(math.atan2(dy, dx + 1e-9))))


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


def _make_thresholds(value: float) -> Dict[str, Any]:
    bands = [
        {"label": "Red", "min": 0, "max": 45, "color": "red"},
        {"label": "Yellow", "min": 45, "max": 70, "color": "yellow"},
        {"label": "Green", "min": 70, "max": 90, "color": "green"},
    ]
    rating = "red" if value < 45 else "yellow" if value < 70 else "green"
    return {
        "unit": "deg",
        "scale_min": 0,
        "scale_max": 90,
        "bands": bands,
        "pointer_value": round(float(value), 2),
        "rating": rating,
    }


def _score_from_existing_bands(angle: float) -> float:
    """Preserve the deployed score mapping while measurement is validated."""
    if angle < 45:
        score = 40.0
    elif angle < 70:
        score = 60.0 + ((angle - 45.0) / 25.0) * 19.0
    else:
        score = 85.0 + ((min(angle, 90.0) - 70.0) / 20.0) * 15.0
    return max(0.0, min(100.0, score))


def _torso_baseline(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    minimum_confidence: float,
) -> Dict[str, Any] | None:
    """Choose the clearest same-side shoulder→hip body axis."""
    candidates = []
    for label, shoulder_idx, hip_idx in (
        ("COCO_LEFT_TORSO", 5, 11),
        ("COCO_RIGHT_TORSO", 6, 12),
    ):
        shoulder = _point(xy, shoulder_idx)
        hip = _point(xy, hip_idx)
        shoulder_conf = _confidence(conf, shoulder_idx)
        hip_conf = _confidence(conf, hip_idx)
        length = _distance(shoulder, hip)
        mean_conf = (shoulder_conf + hip_conf) / 2.0
        candidates.append(
            {
                "label": label,
                "shoulder": shoulder,
                "hip": hip,
                "vector": _vector(shoulder, hip),
                "minimum_confidence": min(shoulder_conf, hip_conf),
                "mean_confidence": mean_conf,
                "length_px": length,
                "available": (
                    min(shoulder_conf, hip_conf) >= minimum_confidence
                    and length > 5.0
                ),
            }
        )

    available = [candidate for candidate in candidates if candidate["available"]]
    if not available:
        return None
    return max(available, key=lambda item: (item["mean_confidence"], item["length_px"]))


def _candidate(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    label: str,
    indices: Tuple[int, int, int],
    keypoint_min_conf: float,
    baseline_vector: Tuple[float, float],
) -> Dict[str, Any]:
    hip_idx, knee_idx, ankle_idx = indices
    hip = _point(xy, hip_idx)
    knee = _point(xy, knee_idx)
    ankle = _point(xy, ankle_idx)
    confs = {
        "hip": _confidence(conf, hip_idx),
        "knee": _confidence(conf, knee_idx),
        "ankle": _confidence(conf, ankle_idx),
    }
    minimum_conf = min(confs.values())
    mean_conf = sum(confs.values()) / 3.0
    thigh_length = _distance(hip, knee)
    shank_length = _distance(knee, ankle)
    segment_ratio = thigh_length / max(shank_length, 1e-6)
    available = (
        minimum_conf >= keypoint_min_conf
        and thigh_length > 2.0
        and shank_length > 2.0
    )
    leg_vector = _vector(hip, ankle)
    return {
        "label": label,
        "indices": indices,
        "hip": hip,
        "knee": knee,
        "ankle": ankle,
        "keypoint_confidence": confs,
        "minimum_confidence": minimum_conf,
        "mean_confidence": mean_conf,
        "body_relative_angle": _acute_angle_between(baseline_vector, leg_vector),
        "image_horizontal_angle": _image_horizontal_elevation(hip, ankle),
        "knee_extension_angle": _joint_angle(hip, knee, ankle),
        "thigh_length_px": thigh_length,
        "shank_length_px": shank_length,
        "thigh_to_shank_ratio": segment_ratio,
        "available": available,
    }


def _public_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "label": candidate["label"],
        "available": bool(candidate["available"]),
        "body_relative_angle": round(float(candidate["body_relative_angle"]), 2),
        "image_horizontal_angle": round(float(candidate["image_horizontal_angle"]), 2),
        "knee_extension_angle": round(float(candidate["knee_extension_angle"]), 2),
        "minimum_confidence": round(float(candidate["minimum_confidence"]), 3),
        "mean_confidence": round(float(candidate["mean_confidence"]), 3),
        "thigh_to_shank_ratio": round(float(candidate["thigh_to_shank_ratio"]), 3),
        "keypoint_confidence": {
            key: round(float(value), 3)
            for key, value in candidate["keypoint_confidence"].items()
        },
        "points": {
            "hip": _rounded_point(candidate["hip"]),
            "knee": _rounded_point(candidate["knee"]),
            "ankle": _rounded_point(candidate["ankle"]),
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
    """Measure ASLR from one COCO-17 pose with orientation-independent geometry.

    The source can be portrait, landscape, square or tilted. The raised leg is
    measured relative to the clearest shoulder→hip body axis. The second leg is
    validated when YOLO identifies it independently; overlapping/duplicated
    COCO chains lower confidence but no longer create a false mandatory retake.
    """

    if len(xy) < 17 or len(conf) < 17:
        raise ASLRQualityError(
            "missing_pose",
            "We could not identify the full leg. Retake the photo with the hips, both knees and both ankles visible.",
        )

    requested_side = str(side or "RIGHT").strip().upper()
    if requested_side not in {"LEFT", "RIGHT"}:
        requested_side = "RIGHT"

    torso = _torso_baseline(xy, conf, torso_min_conf)
    if torso is not None:
        baseline_vector = torso["vector"]
        baseline_method = "clearest_same_side_shoulder_to_hip_axis"
        baseline_confidence = float(torso["mean_confidence"])
    else:
        # Fail-soft fallback for a clear, level desktop or uploaded photo.
        baseline_vector = (1.0, 0.0)
        baseline_method = "image_horizontal_fallback"
        baseline_confidence = 0.0

    candidates = [
        _candidate(
            xy,
            conf,
            "COCO_LEFT",
            (11, 13, 15),
            keypoint_min_conf,
            baseline_vector,
        ),
        _candidate(
            xy,
            conf,
            "COCO_RIGHT",
            (12, 14, 16),
            keypoint_min_conf,
            baseline_vector,
        ),
    ]
    available = [candidate for candidate in candidates if candidate["available"]]
    public_candidates = [_public_candidate(candidate) for candidate in candidates]

    if not available:
        raise ASLRQualityError(
            "required_landmarks_low_confidence",
            "The raised hip, knee or ankle is not clear enough. Improve the lighting, keep the full body visible and retake the photo.",
            {"candidates": public_candidates},
        )

    raised = max(
        available,
        key=lambda item: (item["body_relative_angle"], item["mean_confidence"]),
    )
    resting_options = [candidate for candidate in available if candidate is not raised]
    resting = resting_options[0] if resting_options else None

    if raised["mean_confidence"] < required_mean_conf:
        raise ASLRQualityError(
            "raised_limb_low_confidence",
            "The raised leg is not clear enough for a reliable measurement. Improve the lighting and retake the photo.",
            {"raised": _public_candidate(raised)},
        )

    ratio = float(raised["thigh_to_shank_ratio"])
    if ratio < 0.35 or ratio > 2.80:
        raise ASLRQualityError(
            "inconsistent_limb_geometry",
            "We could not track one complete leg reliably. Retake with the full raised leg visible and avoid overlapping objects.",
            {"raised": _public_candidate(raised)},
        )

    raised_angle = float(raised["body_relative_angle"])
    if raised_angle < minimum_detectable_raise:
        raise ASLRQualityError(
            "raised_leg_not_detected",
            "We could not confirm that one leg was raised. Retake at the highest comfortable position with the knee straight.",
            {"raised": _public_candidate(raised)},
        )

    raised_knee_extension = float(raised["knee_extension_angle"])
    if raised_knee_extension < raised_knee_extension_min:
        raise ASLRQualityError(
            "raised_knee_bent",
            "The raised knee appears bent. Retake the photo while keeping the raised knee straight.",
            {
                "raised_knee_extension_angle": round(raised_knee_extension, 2),
                "required_minimum": raised_knee_extension_min,
            },
        )

    flags = []
    if torso is None:
        flags.append("torso_baseline_unavailable_image_horizontal_used")

    resting_angle = None
    resting_knee_extension = None
    resting_verified = False

    if resting is not None:
        resting_angle = float(resting["body_relative_angle"])
        resting_knee_extension = float(resting["knee_extension_angle"])
        angle_gap = raised_angle - resting_angle

        # Side-view supine photos often make YOLO duplicate both COCO leg chains
        # onto the visible raised leg. Do not blame the client for that model
        # ambiguity. Keep the raised measurement, lower confidence, and expose it
        # in Vision QA for operator review.
        if angle_gap >= ambiguous_angle_gap:
            resting_verified = True
            if resting_angle > resting_leg_max_angle:
                raise ASLRQualityError(
                    "resting_leg_lifted",
                    "The resting leg appears lifted. Keep it straight and flat on the floor, then retake the photo.",
                    {
                        "resting_leg_angle": round(resting_angle, 2),
                        "maximum_allowed": resting_leg_max_angle,
                    },
                )
            if resting_knee_extension < resting_knee_extension_min:
                raise ASLRQualityError(
                    "resting_knee_bent",
                    "The resting knee appears bent. Keep the resting leg straight and flat on the floor, then retake the photo.",
                    {
                        "resting_knee_extension_angle": round(resting_knee_extension, 2),
                        "required_minimum": resting_knee_extension_min,
                    },
                )
        else:
            flags.append("resting_leg_not_independently_resolved_by_pose_model")
    else:
        flags.append("resting_leg_not_fully_verified")

    detected_coco_side = str(raised["label"])
    expected_coco_side = f"COCO_{requested_side}"
    if detected_coco_side != expected_coco_side:
        flags.append("coco_side_differs_from_workflow_side")

    minimum_confidence = float(raised["minimum_confidence"])
    mean_confidence = float(raised["mean_confidence"])
    straightness_quality = max(
        0.0,
        min(
            1.0,
            (raised_knee_extension - raised_knee_extension_min)
            / max(1.0, 180.0 - raised_knee_extension_min),
        ),
    )
    confidence_out = (
        minimum_confidence * 0.42
        + mean_confidence * 0.38
        + straightness_quality * 0.12
        + baseline_confidence * 0.08
    )
    if not resting_verified:
        confidence_out *= 0.82
    if torso is None:
        confidence_out *= 0.82
    confidence_out = max(0.0, min(1.0, confidence_out))

    quality_label = "good" if confidence_out >= 0.65 and not flags else "moderate"
    score = _score_from_existing_bands(raised_angle)

    baseline_public = {
        "method": baseline_method,
        "confidence": round(baseline_confidence, 3),
        "vector": {
            "x": round(float(baseline_vector[0]), 2),
            "y": round(float(baseline_vector[1]), 2),
        },
    }
    if torso is not None:
        baseline_public.update(
            {
                "side": torso["label"],
                "shoulder": _rounded_point(torso["shoulder"]),
                "hip": _rounded_point(torso["hip"]),
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
            "side_identity_method": "workflow_side_with_geometric_raised_limb_selection",
            "measurement_engine_version": ASLR_ENGINE_VERSION,
            "angle_method": "hip_to_ankle_relative_to_body_axis",
            "source_orientation_requirement": "none",
            "body_baseline": baseline_public,
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
            "selected_limb": detected_coco_side,
            "selected_limb_min_confidence": round(minimum_confidence, 3),
            "selected_limb_mean_confidence": round(mean_confidence, 3),
            "candidate_limbs": public_candidates,
            "selected_limb_points": {
                "hip": _rounded_point(raised["hip"]),
                "knee": _rounded_point(raised["knee"]),
                "ankle": _rounded_point(raised["ankle"]),
            },
            "quality_gate_config": {
                "keypoint_min_conf": keypoint_min_conf,
                "required_mean_conf": required_mean_conf,
                "raised_knee_extension_min": raised_knee_extension_min,
                "resting_knee_extension_min": resting_knee_extension_min,
                "resting_leg_max_angle": resting_leg_max_angle,
            },
            "threshold_evidence_status": ASLR_THRESHOLD_EVIDENCE_STATUS,
        },
        "thresholds": {"aslr_angle": _make_thresholds(raised_angle)},
    }
