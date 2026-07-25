"""FlexiLab ASLR horizontal-reference measurement engine.

V101.34 simplifies ASLR to the clinically transparent convention used by the
older stable implementation:

* 0° = raised limb parallel to the image floor/horizontal
* 90° = raised limb vertical above the pelvic anchor

The workflow label determines RIGHT/LEFT. COCO left/right labels are diagnostic
only. The final angle is never measured from the torso axis and the caller does
not rotate the source image.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import cv2
import numpy as np

ASLR_ENGINE_VERSION = "aslr-dedicated-yolo11m-horizontal-endpoint-v10"
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


def _joint_angle(a: Tuple[float, float], vertex: Tuple[float, float], c: Tuple[float, float]) -> float:
    v1 = (a[0] - vertex[0], a[1] - vertex[1])
    v2 = (c[0] - vertex[0], c[1] - vertex[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 <= 1e-6 or n2 <= 1e-6:
        return 0.0
    cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosine))


def _rounded_point(point: Tuple[float, float]) -> Dict[str, float]:
    return {"x": round(float(point[0]), 2), "y": round(float(point[1]), 2)}


def _horizontal_angle(anchor: Tuple[float, float], endpoint: Tuple[float, float]) -> float:
    """Angle above image horizontal, constrained to [0, 90]."""
    dx = abs(float(endpoint[0] - anchor[0]))
    dy_up = max(0.0, float(anchor[1] - endpoint[1]))
    return max(0.0, min(90.0, math.degrees(math.atan2(dy_up, dx + 1e-6))))


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


def _skin_shape_endpoint(img: np.ndarray, pelvis: Tuple[float, float]) -> Dict[str, Any] | None:
    """Find a tall skin component above the pelvis as a rescue for supine ASLR.

    This is deliberately limited to a central band above the pelvis. It is used
    only when pose endpoints are duplicated or produce an implausibly low angle.
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return None
    h, w = img.shape[:2]
    pelvis_x, pelvis_y = pelvis

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    mask_y = cv2.inRange(ycrcb, np.array([0, 128, 70], np.uint8), np.array([255, 190, 145], np.uint8))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_h = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 15, 35], np.uint8), np.array([38, 255, 255], np.uint8)),
        cv2.inRange(hsv, np.array([158, 15, 35], np.uint8), np.array([180, 255, 255], np.uint8)),
    )
    mask = cv2.bitwise_and(mask_y, mask_h)

    roi = np.zeros_like(mask)
    x_margin = int(w * 0.30)
    x1 = max(0, int(pelvis_x - x_margin))
    x2 = min(w, int(pelvis_x + x_margin))
    y1 = max(0, int(h * 0.03))
    y2 = min(h, int(pelvis_y + h * 0.02))
    roi[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

    kernel = np.ones((5, 5), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=2)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(roi, 8)
    best = None
    for idx in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[idx]]
        if area < max(180, int(h * w * 0.0002)) or bh < h * 0.13:
            continue
        cx, cy = [float(v) for v in centroids[idx]]
        if cy >= pelvis_y or abs(cx - pelvis_x) > w * 0.30:
            continue
        elongation = bh / max(1.0, bw)
        vertical_gain = pelvis_y - y
        # Prefer tall, central components extending well above the pelvis.
        score = vertical_gain * 1.8 + elongation * 24.0 + area * 0.002 - abs(cx - pelvis_x) * 0.35
        if best is None or score > best["score"]:
            ys, xs = np.where(labels == idx)
            if len(xs) == 0:
                continue
            # Robust endpoint: median x among the top 3% pixels, rather than one noisy top pixel.
            cutoff = np.percentile(ys, 3)
            top_x = xs[ys <= cutoff]
            top_y = ys[ys <= cutoff]
            endpoint = (float(np.median(top_x)), float(np.median(top_y)))
            best = {
                "score": score,
                "endpoint": endpoint,
                "component_area": area,
                "component_height": bh,
                "component_width": bw,
            }
    return best


def _best_chain_for_endpoint(
    endpoint_index: int,
    endpoint: Tuple[float, float],
    endpoint_conf: float,
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    pelvis: Tuple[float, float],
    minimum_conf: float,
) -> Dict[str, Any]:
    """Attach the endpoint to the most plausible hip/knee for diagnostics only."""
    hip_indices = (11, 12)
    knee_indices = (13, 14)
    best = None
    for hi in hip_indices:
        hc = _confidence(conf, hi)
        if hc < minimum_conf * 0.5:
            continue
        hip = _point(xy, hi)
        for ki in knee_indices:
            kc = _confidence(conf, ki)
            if kc < minimum_conf * 0.5:
                continue
            knee = _point(xy, ki)
            knee_angle = _joint_angle(hip, knee, endpoint)
            thigh = _distance(hip, knee)
            shank = _distance(knee, endpoint)
            ratio = min(thigh, shank) / max(thigh, shank, 1e-6)
            anchor_distance = _distance(hip, pelvis)
            straightness = max(0.0, min(1.0, (knee_angle - 130.0) / 50.0))
            score = (hc + kc + endpoint_conf) / 3.0 * 0.55 + straightness * 0.30 + ratio * 0.10 - min(1.0, anchor_distance / 180.0) * 0.05
            candidate = {
                "hip_idx": hi,
                "knee_idx": ki,
                "ankle_idx": endpoint_index,
                "hip": hip,
                "knee": knee,
                "ankle": endpoint,
                "knee_extension_angle": knee_angle,
                "mean_confidence": (hc + kc + endpoint_conf) / 3.0,
                "minimum_confidence": min(hc, kc, endpoint_conf),
                "chain_score": score,
            }
            if best is None or score > best["chain_score"]:
                best = candidate
    if best is None:
        best = {
            "hip_idx": 11,
            "knee_idx": 13,
            "ankle_idx": endpoint_index,
            "hip": pelvis,
            "knee": endpoint,
            "ankle": endpoint,
            "knee_extension_angle": None,
            "mean_confidence": endpoint_conf,
            "minimum_confidence": endpoint_conf,
            "chain_score": endpoint_conf * 0.5,
        }
    return best


def analyze_aslr_v2(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    side: str = "RIGHT",
    *,
    img: np.ndarray | None = None,
    keypoint_min_conf: float = 0.20,
    required_mean_conf: float = 0.35,
    raised_knee_extension_min: float = 150.0,
    resting_knee_extension_min: float = 145.0,
    resting_leg_max_angle: float = 15.0,
    **_: Any,
) -> Dict[str, Any]:
    if len(xy) < 17 or len(conf) < 17:
        raise ASLRQualityError("missing_pose", "Keep the pelvis and both legs visible, then retake the photo.")

    requested_side = str(side or "RIGHT").upper()
    if requested_side not in {"LEFT", "RIGHT"}:
        requested_side = "RIGHT"

    visible_hips = [(idx, _point(xy, idx), _confidence(conf, idx)) for idx in (11, 12) if _confidence(conf, idx) >= 0.08]
    if not visible_hips:
        raise ASLRQualityError("pelvis_not_detected", "Keep the pelvis and both legs visible, then retake the photo.")
    weight_sum = sum(c for _, _, c in visible_hips)
    pelvis = (
        sum(p[0] * c for _, p, c in visible_hips) / max(weight_sum, 1e-6),
        sum(p[1] * c for _, p, c in visible_hips) / max(weight_sum, 1e-6),
    )
    pelvis_conf = weight_sum / len(visible_hips)

    endpoint_min_conf = min(0.09, keypoint_min_conf)
    raw_endpoints = []
    for idx, label, point_type, weight in (
        (15, "COCO_LEFT_ANKLE", "ankle", 1.15),
        (16, "COCO_RIGHT_ANKLE", "ankle", 1.15),
        (13, "COCO_LEFT_KNEE", "knee", 0.82),
        (14, "COCO_RIGHT_KNEE", "knee", 0.82),
    ):
        c = _confidence(conf, idx)
        if c < endpoint_min_conf:
            continue
        p = _point(xy, idx)
        angle = _horizontal_angle(pelvis, p)
        elevation = pelvis[1] - p[1]
        if elevation < -20:
            continue
        # Angle is primary. Elevation and confidence only break ties.
        selection_score = angle * 2.2 + max(0.0, elevation) * 0.03 + c * 8.0 + weight
        raw_endpoints.append({
            "index": idx,
            "label": label,
            "point_type": point_type,
            "point": p,
            "confidence": c,
            "angle": angle,
            "elevation": elevation,
            "selection_score": selection_score,
        })

    if not raw_endpoints:
        raise ASLRQualityError("raised_endpoint_not_detected", "Keep the raised foot visible, then retake the photo.")

    ankle_entries = [e for e in raw_endpoints if e["point_type"] == "ankle"]
    ankle_separation = None
    ankles_distinct = False
    if len(ankle_entries) >= 2:
        ankle_separation = _distance(ankle_entries[0]["point"], ankle_entries[1]["point"])
        body_scale = max(80.0, _distance(pelvis, max(raw_endpoints, key=lambda e: e["elevation"])["point"]))
        ankles_distinct = ankle_separation >= max(22.0, body_scale * 0.12)

    # Prefer a real ankle whenever YOLO resolved at least one ankle. Knees are
    # validation/rescue points only; selecting the knee can overestimate a nearly
    # vertical leg and makes the displayed endpoint clinically confusing.
    selected_pool = ankle_entries if ankle_entries else raw_endpoints
    selected = max(selected_pool, key=lambda e: e["selection_score"])
    pose_angle = selected["angle"]
    flags = []
    if len(ankle_entries) >= 2 and not ankles_distinct:
        flags.append("ankle_endpoints_duplicated_by_pose_model")

    skin = None
    # Rescue when pose gives a low angle, duplicates both ankles, or selects only a knee.
    if img is not None and (pose_angle < 58.0 or not ankles_distinct or selected["point_type"] == "knee"):
        skin = _skin_shape_endpoint(img, pelvis)

    skin_angle = None
    if skin is not None:
        skin_angle = _horizontal_angle(pelvis, skin["endpoint"])

    angle_method = "pelvis_to_highest_pose_endpoint_against_image_horizontal"
    endpoint = selected["point"]
    final_angle = pose_angle
    if skin_angle is not None and skin_angle >= pose_angle + 8.0:
        final_angle = skin_angle
        endpoint = skin["endpoint"]
        angle_method = "pelvis_to_skin_shape_endpoint_against_image_horizontal"
        flags.append("skin_shape_endpoint_rescue_used")

    if final_angle < 8.0:
        raise ASLRQualityError("raised_leg_not_detected", "Raise one leg and keep the foot fully visible, then retake the photo.")

    chain = _best_chain_for_endpoint(
        selected["index"], selected["point"], selected["confidence"], xy, conf, pelvis, keypoint_min_conf
    )
    knee_extension = chain.get("knee_extension_angle")
    if knee_extension is not None and knee_extension < raised_knee_extension_min:
        flags.append("raised_knee_may_be_bent_or_pose_chain_crossed")

    # Reliability reflects landmark quality and agreement with the visual rescue.
    selected_conf = float(selected["confidence"])
    reliability = max(0.0, min(1.0, pelvis_conf * 0.35 + selected_conf * 0.45 + float(chain["chain_score"]) * 0.20))
    if skin_angle is not None and abs(skin_angle - pose_angle) <= 8.0:
        reliability = min(1.0, reliability + 0.08)
    elif skin_angle is not None and abs(skin_angle - pose_angle) > 18.0:
        reliability *= 0.88

    endpoint_candidates = [
        {
            "source_label": e["label"],
            "endpoint_index": e["index"],
            "point_type": e["point_type"],
            "point": _rounded_point(e["point"]),
            "confidence": round(e["confidence"], 3),
            "horizontal_angle": round(e["angle"], 2),
            "elevation_px": round(e["elevation"], 2),
        }
        for e in sorted(raw_endpoints, key=lambda e: e["selection_score"], reverse=True)
    ]

    selected_points = {
        "pelvis": _rounded_point(pelvis),
        "hip": _rounded_point(chain["hip"]),
        "knee": _rounded_point(chain["knee"]),
        "ankle": _rounded_point(endpoint),
    }

    return {
        "score": round(_score_from_reference_bands(final_angle), 1),
        "confidence": round(reliability, 3),
        "metrics": {
            "aslr_angle": round(final_angle, 2),
            "requested_side": requested_side,
            "side": requested_side,
            "detected_coco_side": selected["label"].replace("_ANKLE", "").replace("_KNEE", ""),
            "side_identity_method": "workflow_label_geometry_only",
            "measurement_engine_version": ASLR_ENGINE_VERSION,
            "angle_method": angle_method,
            "source_orientation_requirement": "original_normalized_photo_only",
            "reference_axis": "image_horizontal",
            "pose_endpoint_angle": round(pose_angle, 2),
            "skin_shape_endpoint_angle": round(skin_angle, 2) if skin_angle is not None else None,
            "measurement_reliability": round(reliability, 3),
            "quality_label": "good" if reliability >= 0.72 and not flags else "moderate",
            "diagnostic_flags": flags,
            "selected_limb": "SELECTED_HORIZONTAL_ENDPOINT",
            "selected_endpoint_type": selected["point_type"],
            "selected_endpoint_confidence": round(selected_conf, 3),
            "selected_limb_points": selected_points,
            "selected_source_indices": {
                "hip": int(chain["hip_idx"]),
                "knee": int(chain["knee_idx"]),
                "ankle": int(chain["ankle_idx"]),
            },
            "raised_knee_extension_angle": round(knee_extension, 2) if knee_extension is not None else None,
            "pelvis_center": _rounded_point(pelvis),
            "pelvis_confidence": round(pelvis_conf, 3),
            "candidate_limbs": [{
                "label": "SELECTED_HORIZONTAL_ENDPOINT",
                "points": selected_points,
                "available": True,
                "body_relative_angle": round(final_angle, 2),
                "horizontal_angle": round(final_angle, 2),
                "chain_score": round(float(chain["chain_score"]), 3),
                "mean_confidence": round(float(chain["mean_confidence"]), 3),
                "minimum_confidence": round(float(chain["minimum_confidence"]), 3),
                "knee_extension_angle": round(knee_extension, 2) if knee_extension is not None else None,
                "source_indices": {
                    "hip": int(chain["hip_idx"]),
                    "knee": int(chain["knee_idx"]),
                    "ankle": int(chain["ankle_idx"]),
                },
            }],
            "endpoint_candidates": endpoint_candidates,
            "detected_ankle_endpoint_count": len(ankle_entries),
            "ankle_endpoints_are_distinct": ankles_distinct,
            "ankle_endpoint_separation_px": round(ankle_separation, 2) if ankle_separation is not None else None,
            "skin_shape_endpoint": _rounded_point(skin["endpoint"]) if skin is not None else None,
            "quality_gate_config": {
                "keypoint_min_conf": keypoint_min_conf,
                "required_mean_conf": required_mean_conf,
                "raised_knee_extension_min": raised_knee_extension_min,
                "resting_knee_extension_min": resting_knee_extension_min,
                "resting_leg_max_angle": resting_leg_max_angle,
            },
            "threshold_evidence_status": ASLR_THRESHOLD_EVIDENCE_STATUS,
        },
        "thresholds": {"aslr_angle": make_aslr_thresholds(final_angle)},
    }
