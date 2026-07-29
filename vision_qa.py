"""Private, ephemeral Vision QA rendering for FlexiLab validation accounts.

The generated composite is returned only when explicitly requested by the
validation UI. It is ephemeral and is never written into the authoritative
`screenings` row. ASLR overlays show the exact inverse-mapped YOLO landmarks
and measurement geometry used by the scoring engine.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, Mapping, Sequence, Tuple

import cv2
import numpy as np


VISION_QA_VERSION = "vision-qa-overlay-v2.7-aslr-enabled-validation"

COCO_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]


def _point(value: Mapping[str, Any] | Sequence[float]) -> Tuple[int, int]:
    if isinstance(value, Mapping):
        return (int(round(float(value.get("x", 0)))), int(round(float(value.get("y", 0)))))
    return (int(round(float(value[0]))), int(round(float(value[1]))))


def _put_label(image: np.ndarray, text: str, origin: Tuple[int, int], scale: float = 0.48) -> None:
    x, y = origin
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = max(2, min(x, max(2, image.shape[1] - width - 4)))
    y = max(height + 4, min(y, max(height + 4, image.shape[0] - baseline - 3)))
    cv2.rectangle(image, (x - 2, y - height - 3), (x + width + 2, y + baseline + 2), (10, 15, 25), -1)
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def _draw_header(image: np.ndarray, title: str) -> np.ndarray:
    bar_height = 44
    output = cv2.copyMakeBorder(image, bar_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(8, 13, 22))
    cv2.putText(
        output,
        title,
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (90, 235, 225),
        2,
        cv2.LINE_AA,
    )
    return output


def _fit_panel(image: np.ndarray, target_height: int = 620) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((target_height, 360, 3), dtype=np.uint8)
    scale = target_height / height
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), target_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    return resized


def _draw_skeleton(
    image: np.ndarray,
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    box: Sequence[float] | None,
    minimum_confidence: float = 0.12,
) -> np.ndarray:
    output = image.copy()
    if box is not None and len(box) >= 4:
        x1, y1, x2, y2 = [int(round(float(value))) for value in box[:4]]
        cv2.rectangle(output, (x1, y1), (x2, y2), (80, 220, 245), 2)

    for start, end in COCO_SKELETON:
        if start >= len(xy) or end >= len(xy) or start >= len(conf) or end >= len(conf):
            continue
        if float(conf[start]) < minimum_confidence or float(conf[end]) < minimum_confidence:
            continue
        p1 = _point(xy[start])
        p2 = _point(xy[end])
        cv2.line(output, p1, p2, (85, 225, 125), 3, cv2.LINE_AA)

    for index, name in enumerate(COCO_NAMES):
        if index >= len(xy) or index >= len(conf):
            continue
        confidence = float(conf[index])
        if confidence < minimum_confidence:
            continue
        point = _point(xy[index])
        color = (80, 235, 120) if confidence >= 0.50 else (40, 180, 250)
        cv2.circle(output, point, 6, color, -1, cv2.LINE_AA)
        _put_label(output, f"{index}:{name} {confidence:.2f}", (point[0] + 8, point[1] - 8), 0.36)
    return output


def _draw_shoulder_measurement(image: np.ndarray, metrics: Mapping[str, Any]) -> None:
    side = str(metrics.get("side") or "RIGHT").upper()
    indices = (6, 8, 10, 12) if side == "RIGHT" else (5, 7, 9, 11)
    shoulder_idx, elbow_idx, wrist_idx, hip_idx = indices
    geometry = metrics.get("measurement_points") or {}
    if not isinstance(geometry, Mapping):
        return
    points = geometry.get("points") or {}
    if not isinstance(points, Mapping):
        return
    shoulder = _point(points.get("shoulder", (0, 0)))
    elbow = _point(points.get("elbow", (0, 0)))
    wrist = _point(points.get("wrist", (0, 0)))
    hip = _point(points.get("hip", (0, 0)))
    arm_used = str(metrics.get("arm_point_used") or "WRIST")
    arm_point = wrist if arm_used == "WRIST" else elbow
    cv2.line(image, hip, shoulder, (255, 190, 60), 5, cv2.LINE_AA)
    cv2.line(image, shoulder, arm_point, (90, 235, 225), 5, cv2.LINE_AA)
    cv2.circle(image, shoulder, 9, (255, 255, 255), -1, cv2.LINE_AA)
    value = metrics.get("shoulder_flexion_angle")
    _put_label(
        image,
        f"Shoulder {side}: {float(value):.1f} deg | points {hip_idx}-{shoulder_idx}-{wrist_idx if arm_used == 'WRIST' else elbow_idx}",
        (18, 34),
        0.58,
    )


def _draw_posture_measurement(image: np.ndarray, metrics: Mapping[str, Any]) -> None:
    geometry = metrics.get("measurement_points") or {}
    points = geometry.get("points") if isinstance(geometry, Mapping) else None
    if not isinstance(points, Mapping):
        return
    ear = _point(points.get("ear", (0, 0)))
    shoulder = _point(points.get("shoulder", (0, 0)))
    hip = _point(points.get("hip", (0, 0)))
    cv2.line(image, hip, shoulder, (255, 190, 60), 5, cv2.LINE_AA)
    cv2.line(image, shoulder, ear, (90, 235, 225), 5, cv2.LINE_AA)
    _put_label(
        image,
        f"Neck {float(metrics.get('neck_angle', 0)):.1f} deg | Trunk {float(metrics.get('thoracic_angle', 0)):.1f} deg",
        (18, 34),
        0.58,
    )


def _draw_squat_measurement(image: np.ndarray, metrics: Mapping[str, Any]) -> None:
    geometry = metrics.get("measurement_points") or {}
    points = geometry.get("points") if isinstance(geometry, Mapping) else None
    if not isinstance(points, Mapping):
        return
    shoulder = _point(points.get("shoulder", (0, 0)))
    hip = _point(points.get("hip", (0, 0)))
    knee = _point(points.get("knee", (0, 0)))
    ankle = _point(points.get("ankle", (0, 0)))
    cv2.line(image, shoulder, hip, (255, 190, 60), 5, cv2.LINE_AA)
    cv2.line(image, hip, knee, (90, 235, 225), 5, cv2.LINE_AA)
    cv2.line(image, knee, ankle, (90, 235, 225), 5, cv2.LINE_AA)
    _put_label(
        image,
        f"Knee {float(metrics.get('knee_angle', 0)):.1f} deg | Trunk {float(metrics.get('trunk_lean', 0)):.1f} deg",
        (18, 34),
        0.58,
    )


def _draw_aslr_measurement(image: np.ndarray, metrics: Mapping[str, Any]) -> None:
    """Draw the deterministic image-horizontal reference and raised leg.

    Pink: original-image horizontal through the shared pelvic anchor.
    Yellow: the same pelvic anchor to the true raised YOLO ankle.
    Shoulder and floor-leg landmarks are excluded from the angle.
    """
    selected_points = metrics.get("selected_limb_points") or {}
    resting_points = metrics.get("resting_limb_points") or {}
    baseline = metrics.get("body_baseline") or {}
    if not isinstance(selected_points, Mapping):
        selected_points = {}
    if not isinstance(resting_points, Mapping):
        resting_points = {}
    if not isinstance(baseline, Mapping):
        baseline = {}

    raised_hip_value = selected_points.get("hip")
    raised_ankle_value = selected_points.get("ankle")
    raised_knee_value = selected_points.get("knee")
    resting_knee_value = resting_points.get("knee")
    resting_ankle_value = resting_points.get("ankle")

    raised_hip = _point(raised_hip_value) if raised_hip_value else None
    raised_ankle = _point(raised_ankle_value) if raised_ankle_value else None
    raised_knee = _point(raised_knee_value) if raised_knee_value else None
    resting_knee = _point(resting_knee_value) if resting_knee_value else None
    resting_ankle = _point(resting_ankle_value) if resting_ankle_value else None

    reference_start_value = baseline.get("line_start") or resting_points.get("hip")
    reference_end_value = baseline.get("line_end") or resting_points.get("reference_endpoint")
    reference_start = _point(reference_start_value) if reference_start_value else None
    reference_end = _point(reference_end_value) if reference_end_value else None
    reference_source = str(metrics.get("reference_source") or baseline.get("reference_source") or "unknown")

    reference_labels = {
        "image_horizontal_primary": "Image-horizontal reference",
        "body_axis_primary": "Body-axis reference (legacy)",
    }
    reference_label = reference_labels.get(reference_source, "ASLR reference")

    if reference_start is not None and reference_end is not None:
        cv2.line(image, reference_start, reference_end, (230, 100, 240), 6, cv2.LINE_AA)
        cv2.circle(image, reference_start, 9, (230, 100, 240), -1, cv2.LINE_AA)
        cv2.circle(image, reference_end, 7, (230, 100, 240), -1, cv2.LINE_AA)
        _put_label(image, "Pelvic anchor", (reference_start[0] + 10, reference_start[1] - 10), 0.42)
        _put_label(image, reference_label, (reference_end[0] + 10, reference_end[1] - 10), 0.40)

    if raised_hip is not None and raised_ankle is not None:
        cv2.line(image, raised_hip, raised_ankle, (40, 235, 250), 7, cv2.LINE_AA)
        cv2.circle(image, raised_hip, 10, (40, 235, 250), -1, cv2.LINE_AA)
        cv2.circle(image, raised_ankle, 10, (40, 235, 250), -1, cv2.LINE_AA)
        _put_label(image, "Pelvic anchor", (raised_hip[0] + 10, raised_hip[1] - 10), 0.45)
        _put_label(image, "Raised ankle (YOLO)", (raised_ankle[0] + 10, raised_ankle[1] - 10), 0.45)

    if resting_knee is not None:
        cv2.circle(image, resting_knee, 7, (180, 145, 220), -1, cv2.LINE_AA)
        _put_label(image, "Resting knee check (optional)", (resting_knee[0] + 10, resting_knee[1] - 10), 0.34)
    if resting_ankle is not None:
        cv2.circle(image, resting_ankle, 7, (180, 145, 220), -1, cv2.LINE_AA)
        _put_label(image, "Resting ankle check (optional)", (resting_ankle[0] + 10, resting_ankle[1] - 10), 0.34)

    if raised_knee is not None:
        cv2.circle(image, raised_knee, 9, (85, 225, 125), -1, cv2.LINE_AA)
        _put_label(image, "Raised knee check", (raised_knee[0] + 10, raised_knee[1] - 10), 0.40)

    analysis_pass = metrics.get("analysis_pass") or {}
    selected_pass = analysis_pass.get("selected_pass") or analysis_pass.get("mode") or "unknown"
    model_runtime = metrics.get("model_runtime") or {}
    model_name = str(model_runtime.get("model") or "unknown-model")
    knee_angle = metrics.get("raised_knee_extension_angle")
    knee_text = f"{float(knee_angle):.1f}" if knee_angle is not None else "n/a"
    reference_text = reference_source.replace("_", " ")
    _put_label(
        image,
        f"ASLR {metrics.get('requested_side', '')}: {float(metrics.get('aslr_angle', 0)):.1f} deg | ref image horizontal | knee {knee_text} deg | pass {selected_pass} | model {model_name}",
        (18, 34),
        0.52,
    )

def _draw_measurement(
    image: np.ndarray,
    test_type: str,
    result: Mapping[str, Any],
) -> np.ndarray:
    output = image.copy()
    metrics = result.get("metrics") or {}
    if test_type in {"shoulder_left", "shoulder_right"}:
        _draw_shoulder_measurement(output, metrics)
    elif test_type == "posture_side":
        _draw_posture_measurement(output, metrics)
    elif test_type == "squat":
        _draw_squat_measurement(output, metrics)
    elif str(test_type).startswith("aslr"):
        _draw_aslr_measurement(output, metrics)
    return output


def _encode_jpeg_data_url(image: np.ndarray, quality: int = 78) -> str:
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Unable to render Vision QA image")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def build_vision_qa_payload(
    image: np.ndarray,
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    box: Sequence[float] | None,
    result: Mapping[str, Any],
    test_type: str,
    *,
    analysis_pass: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create a compact three-panel original/skeleton/measurement composite."""
    original_panel = _draw_header(_fit_panel(image), "NORMALIZED INPUT")
    skeleton_panel = _draw_header(
        _fit_panel(_draw_skeleton(image, xy, conf, box)),
        "YOLO SKELETON + CONFIDENCE",
    )
    measurement_panel = _draw_header(
        _fit_panel(_draw_measurement(image, test_type, result)),
        "FLEXILAB ANGLE SELECTION",
    )

    separator = np.full((original_panel.shape[0], 8, 3), (24, 32, 44), dtype=np.uint8)
    composite = np.concatenate(
        [original_panel, separator, skeleton_panel, separator, measurement_panel],
        axis=1,
    )

    keypoints = []
    for index, name in enumerate(COCO_NAMES):
        if index >= len(xy) or index >= len(conf):
            continue
        keypoints.append(
            {
                "index": index,
                "name": name,
                "x": round(float(xy[index][0]), 2),
                "y": round(float(xy[index][1]), 2),
                "confidence": round(float(conf[index]), 3),
            }
        )

    return {
        "version": VISION_QA_VERSION,
        "composite_data_url": _encode_jpeg_data_url(composite),
        "normalized_width": int(image.shape[1]),
        "normalized_height": int(image.shape[0]),
        "analysis_pass": dict(analysis_pass or {}),
        "keypoints": keypoints,
        "notice": "Validation-only visualization. Landmark placement must be reviewed before clinical interpretation.",
    }
