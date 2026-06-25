from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import math
import os
import json
import base64
from datetime import datetime, timezone
from program_engine import generate_program_from_report

os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

from ultralytics import YOLO
from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

supabase = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolov8n-pose.pt")


@app.get("/health")
def health():
    return {"ok": True}


def safe_json_loads(raw):
    try:
        return json.loads(raw) if raw else None
    except Exception:
        return None


def make_thresholds(unit, scale_min, scale_max, bands, pointer_value):
    v = float(pointer_value)
    v = max(float(scale_min), min(float(scale_max), v))

    rating = "unknown"
    for b in bands:
        if v >= float(b["min"]) and v < float(b["max"]):
            rating = b.get("color", b.get("label", "unknown")).lower()
            break

    if v == float(scale_max) and bands:
        rating = bands[-1].get("color", bands[-1].get("label", "unknown")).lower()

    return {
        "unit": unit,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "bands": bands,
        "pointer_value": round(v, 2),
        "rating": rating
    }


def angle_to_vertical(p1, p2):
    dx = float(p2[0] - p1[0])
    dy = float(p1[1] - p2[1])
    ang = abs(math.degrees(math.atan2(dx, dy)))
    if ang > 90:
        ang = 180 - ang
    return ang


def analyze_posture(xy, conf):
    L_EAR, R_EAR = 3, 4
    L_SH, R_SH = 5, 6
    L_HIP, R_HIP = 11, 12

    left_score = float(conf[L_EAR] + conf[L_SH] + conf[L_HIP])
    right_score = float(conf[R_EAR] + conf[R_SH] + conf[R_HIP])

    if right_score >= left_score:
        ear, shoulder, hip = xy[R_EAR], xy[R_SH], xy[R_HIP]
        side = "RIGHT"
        quality = right_score / 3.0
    else:
        ear, shoulder, hip = xy[L_EAR], xy[L_SH], xy[L_HIP]
        side = "LEFT"
        quality = left_score / 3.0

    neck_angle = angle_to_vertical(shoulder, ear)
    thoracic_angle = angle_to_vertical(hip, shoulder)
    pelvic_proxy_angle = thoracic_angle

    def penalty(angle, optimal, severe, w=1.0):
        if angle <= optimal:
            return 0.0
        a = min(float(angle), float(severe))
        t = (a - float(optimal)) / (float(severe) - float(optimal))
        return float(w) * 30.0 * (t ** 2)

    total_pen = (
        penalty(neck_angle, 10, 55, 1.2) +
        penalty(thoracic_angle, 5, 45, 1.0) +
        penalty(pelvic_proxy_angle, 5, 40, 0.8)
    )

    score = max(0.0, 100.0 - total_pen)
    conf_out = max(0.6, min(1.0, float(quality)))

    neck_thr = make_thresholds(
        "deg", 0, 60,
        [
            {"label": "Green", "min": 0, "max": 10, "color": "green"},
            {"label": "Yellow", "min": 10, "max": 20, "color": "yellow"},
            {"label": "Red", "min": 20, "max": 60, "color": "red"},
        ],
        neck_angle
    )

    thor_thr = make_thresholds(
        "deg", 0, 45,
        [
            {"label": "Green", "min": 0, "max": 5, "color": "green"},
            {"label": "Yellow", "min": 5, "max": 15, "color": "yellow"},
            {"label": "Red", "min": 15, "max": 45, "color": "red"},
        ],
        thoracic_angle
    )

    pelvis_thr = make_thresholds(
        "deg", 0, 45,
        [
            {"label": "Green", "min": 0, "max": 5, "color": "green"},
            {"label": "Yellow", "min": 5, "max": 15, "color": "yellow"},
            {"label": "Red", "min": 15, "max": 45, "color": "red"},
        ],
        pelvic_proxy_angle
    )

    return {
        "score": round(score, 1),
        "confidence": round(conf_out, 3),
        "metrics": {
            "neck_angle": round(neck_angle, 2),
            "thoracic_angle": round(thoracic_angle, 2),
            "pelvic_proxy_angle": round(pelvic_proxy_angle, 2),
            "side_used": side,
        },
        "thresholds": {
            "neck_angle": neck_thr,
            "thoracic_angle": thor_thr,
            "pelvic_proxy_angle": pelvis_thr
        }
    }


def analyze_shoulder(xy, conf, side="RIGHT"):
    """
    Robust overhead shoulder mobility analysis — V16 complementary angle fix.

    Main change vs previous version:
    - Uses shoulder -> wrist as the primary arm vector instead of shoulder -> elbow.
      This is more stable for overhead mobility, because a slightly bent elbow can
      artificially reduce the angle if we only use the elbow.
    - Falls back to shoulder -> elbow only if wrist confidence is poor.
    - Uses hip -> shoulder as trunk reference and arm direction shoulder -> wrist/elbow.
    - Keeps the same FlexiLab thresholds:
        <160° = red
        160–170° = yellow
        >=170° = green
    """

    L_SH, R_SH = 5, 6
    L_EL, R_EL = 7, 8
    L_WR, R_WR = 9, 10
    L_HIP, R_HIP = 11, 12

    MIN_KP_CONF = 0.25

    if side == "RIGHT":
        sh_i, el_i, wr_i, hip_i = R_SH, R_EL, R_WR, R_HIP
    else:
        sh_i, el_i, wr_i, hip_i = L_SH, L_EL, L_WR, L_HIP

    sh = xy[sh_i]
    el = xy[el_i]
    wr = xy[wr_i]
    hip = xy[hip_i]

    sh_c = float(conf[sh_i])
    el_c = float(conf[el_i])
    wr_c = float(conf[wr_i])
    hip_c = float(conf[hip_i])

    # Use wrist when possible. If wrist is unreliable, fallback to elbow.
    if wr_c >= MIN_KP_CONF:
        arm_point = wr
        arm_point_used = "WRIST"
        arm_c = wr_c
    else:
        arm_point = el
        arm_point_used = "ELBOW_FALLBACK"
        arm_c = el_c

    c = float(sh_c + arm_c + hip_c) / 3.0

    # If shoulder or hip are not detected well, keep output but mark lower confidence.
    # The frontend can later display "photo quality low" if confidence is low.
    v_trunk = sh - hip          # hip -> shoulder
    v_arm = arm_point - sh      # shoulder -> wrist/elbow

    denom = np.linalg.norm(v_trunk) * np.linalg.norm(v_arm)

    if denom < 1e-6:
        shoulder_flexion = 0.0
    else:
        cosang = float(np.dot(v_trunk, v_arm) / denom)
        cosang = max(-1.0, min(1.0, cosang))
        raw_angle = float(math.degrees(math.acos(cosang)))

        # raw_angle is the small geometric angle between trunk direction and arm direction.
        # For overhead mobility we need the complementary value:
        # example raw_angle 10° => shoulder flexion 170°.
        shoulder_flexion = 180.0 - raw_angle

    shoulder_flexion = max(0.0, min(180.0, shoulder_flexion))

    deficit = max(0.0, 170.0 - shoulder_flexion)
    score = max(0.0, 100.0 - deficit * 2.0)
    conf_out = max(0.0, min(1.0, float(c)))

    shoulder_thr = make_thresholds(
        "deg", 0, 180,
        [
            {"label": "Red", "min": 0, "max": 160, "color": "red"},
            {"label": "Yellow", "min": 160, "max": 170, "color": "yellow"},
            {"label": "Green", "min": 170, "max": 180, "color": "green"},
        ],
        shoulder_flexion
    )

    return {
        "score": round(score, 1),
        "confidence": round(conf_out, 3),
        "metrics": {
            "shoulder_flexion_angle": round(shoulder_flexion, 2),
            "side": side,
            "arm_point_used": arm_point_used,
            "keypoint_confidence": {
                "shoulder": round(sh_c, 3),
                "elbow": round(el_c, 3),
                "wrist": round(wr_c, 3),
                "hip": round(hip_c, 3)
            }
        },
        "thresholds": {
            "shoulder_flexion": shoulder_thr
        }
    }


def analyze_squat(xy, conf):
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANK, R_ANK = 15, 16
    L_SH, R_SH = 5, 6

    hip = (xy[L_HIP] + xy[R_HIP]) / 2
    knee = (xy[L_KNEE] + xy[R_KNEE]) / 2
    ankle = (xy[L_ANK] + xy[R_ANK]) / 2
    shoulder = (xy[L_SH] + xy[R_SH]) / 2

    v1 = hip - knee
    v2 = ankle - knee

    knee_angle = abs(
        math.degrees(math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0]))
    )
    if knee_angle > 180:
        knee_angle = 360 - knee_angle

    trunk_dx = float(shoulder[0] - hip[0])
    trunk_dy = float(hip[1] - shoulder[1])
    trunk_angle = abs(math.degrees(math.atan2(trunk_dx, trunk_dy)))

    depth_pen = 0.0
    if knee_angle > 120:
        depth_pen = 35
    elif knee_angle > 100:
        depth_pen = 20
    elif knee_angle > 85:
        depth_pen = 10

    trunk_pen = 0.0
    if trunk_angle > 25:
        trunk_pen = 20
    elif trunk_angle > 15:
        trunk_pen = 10

    score = max(0.0, 100.0 - depth_pen - trunk_pen)
    c = float(np.mean(conf))
    conf_out = max(0.6, min(1.0, c))

    trunk_thr = make_thresholds(
        "deg", 0, 60,
        [
            {"label": "Green", "min": 0, "max": 15, "color": "green"},
            {"label": "Yellow", "min": 15, "max": 25, "color": "yellow"},
            {"label": "Red", "min": 25, "max": 60, "color": "red"},
        ],
        trunk_angle
    )

    knee_thr = make_thresholds(
        "deg", 60, 180,
        [
            {"label": "Green", "min": 60, "max": 95, "color": "green"},
            {"label": "Yellow", "min": 95, "max": 110, "color": "yellow"},
            {"label": "Red", "min": 110, "max": 180, "color": "red"},
        ],
        knee_angle
    )

    return {
        "score": round(score, 1),
        "confidence": round(conf_out, 3),
        "metrics": {
            "knee_angle": round(float(knee_angle), 2),
            "trunk_lean": round(float(trunk_angle), 2),
        },
        "thresholds": {
            "knee_angle": knee_thr,
            "trunk_lean": trunk_thr
        }
    }



def analyze_aslr(xy, conf, side="RIGHT"):
    """
    Active Straight Leg Raise (ASLR) analysis — V15.1 raised-leg auto-selection.

    Main fix vs V15:
    - In supine / lying photos, YOLO often swaps left and right legs.
    - Therefore, for ASLR we DO NOT blindly trust COCO left/right labels.
    - We calculate both leg angles and automatically select the raised leg.
    - The selected ASLR angle is the leg with the higher elevation angle.

    This fixes the common error:
    - user raises the leg vertically
    - YOLO labels the other/floor leg as the requested side
    - old code reads 20–40° instead of ~90°
    """

    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANK, R_ANK = 15, 16

    MIN_REQUIRED_CONF = 0.10
    MIN_GOOD_CONF = 0.25

    def angle_from_horizontal(p1, p2):
        dx = float(p2[0] - p1[0])
        dy = float(p1[1] - p2[1])  # invert image y-axis
        ang = abs(math.degrees(math.atan2(dy, abs(dx) + 1e-6)))
        return max(0.0, min(180.0, ang))

    def knee_angle_deg(hip, knee, ankle):
        try:
            v1 = hip - knee
            v2 = ankle - knee
            raw = abs(math.degrees(math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])))
            if raw > 180:
                raw = 360 - raw
            return float(raw)
        except Exception:
            return None

    def leg_candidate(label, hip_i, knee_i, ankle_i, other_hip_i, other_ankle_i):
        hip = xy[hip_i]
        knee = xy[knee_i]
        ankle = xy[ankle_i]
        other_hip = xy[other_hip_i]
        other_ankle = xy[other_ankle_i]

        hip_c = float(conf[hip_i])
        knee_c = float(conf[knee_i])
        ankle_c = float(conf[ankle_i])
        other_hip_c = float(conf[other_hip_i])
        other_ankle_c = float(conf[other_ankle_i])

        hip_to_ankle_angle = angle_from_horizontal(hip, ankle)
        hip_to_knee_angle = angle_from_horizontal(hip, knee)
        knee_to_ankle_angle = angle_from_horizontal(knee, ankle)

        estimates = []
        if hip_c >= MIN_REQUIRED_CONF and ankle_c >= MIN_REQUIRED_CONF:
            estimates.append((hip_to_ankle_angle, max(0.05, min(1.0, (hip_c + ankle_c) / 2.0)) * 1.2, "hip_to_ankle"))
        if hip_c >= MIN_REQUIRED_CONF and knee_c >= MIN_REQUIRED_CONF:
            estimates.append((hip_to_knee_angle, max(0.05, min(1.0, (hip_c + knee_c) / 2.0)) * 1.0, "hip_to_knee"))
        if knee_c >= MIN_REQUIRED_CONF and ankle_c >= MIN_REQUIRED_CONF:
            estimates.append((knee_to_ankle_angle, max(0.05, min(1.0, (knee_c + ankle_c) / 2.0)) * 0.45, "knee_to_ankle"))

        diagnostic_flags = []
        if hip_c < MIN_REQUIRED_CONF:
            diagnostic_flags.append("low_hip_confidence")
        if knee_c < MIN_REQUIRED_CONF:
            diagnostic_flags.append("low_knee_confidence")
        if ankle_c < MIN_REQUIRED_CONF:
            diagnostic_flags.append("low_ankle_confidence")

        if not estimates:
            angle = 0.0
            method = "insufficient_keypoints"
            diagnostic_flags.append("insufficient_required_keypoints")
        else:
            total_w = sum(w for _, w, _ in estimates)
            angle = sum(a * w for a, w, _ in estimates) / max(total_w, 1e-6)
            method = "+".join(m for _, _, m in estimates)

        raised_knee_angle = knee_angle_deg(hip, knee, ankle)
        if raised_knee_angle is not None and raised_knee_angle < 145:
            diagnostic_flags.append("raised_knee_bent")

        opposite_leg_angle = None
        if other_hip_c >= MIN_REQUIRED_CONF and other_ankle_c >= MIN_REQUIRED_CONF:
            opposite_leg_angle = angle_from_horizontal(other_hip, other_ankle)

        conf_out = max(0.0, min(1.0, float(hip_c + knee_c + ankle_c) / 3.0))

        # Selection score favors a high elevation angle and acceptable confidence.
        # It also slightly penalizes a very bent knee, because ASLR requires a straight leg.
        selection_score = angle * (0.65 + 0.35 * conf_out)
        if raised_knee_angle is not None and raised_knee_angle < 145:
            selection_score -= 10

        return {
            "label": label,
            "angle": max(0.0, min(180.0, float(angle))),
            "selection_score": float(selection_score),
            "confidence": conf_out,
            "angle_method": method,
            "hip_to_ankle_angle": hip_to_ankle_angle,
            "hip_to_knee_angle": hip_to_knee_angle,
            "knee_to_ankle_angle": knee_to_ankle_angle,
            "raised_knee_angle": raised_knee_angle,
            "opposite_leg_angle": opposite_leg_angle,
            "diagnostic_flags": diagnostic_flags,
            "keypoint_confidence": {
                "hip": round(hip_c, 3),
                "knee": round(knee_c, 3),
                "ankle": round(ankle_c, 3),
                "opposite_hip": round(other_hip_c, 3),
                "opposite_ankle": round(other_ankle_c, 3),
            }
        }

    left_candidate = leg_candidate("COCO_LEFT", L_HIP, L_KNEE, L_ANK, R_HIP, R_ANK)
    right_candidate = leg_candidate("COCO_RIGHT", R_HIP, R_KNEE, R_ANK, L_HIP, L_ANK)

    # For ASLR, the raised leg is the leg with the higher elevation angle/selection score.
    candidates = [left_candidate, right_candidate]
    selected = max(candidates, key=lambda c: c["selection_score"])

    aslr_angle = selected["angle"]
    diagnostic_flags = list(selected["diagnostic_flags"])

    requested_side = side.upper()
    detected_coco_side = selected["label"]

    # This flag is expected sometimes in lying photos and is the reason V15.1 exists.
    if (requested_side == "RIGHT" and detected_coco_side == "COCO_LEFT") or (requested_side == "LEFT" and detected_coco_side == "COCO_RIGHT"):
        diagnostic_flags.append("yolo_left_right_swap_possible")

    # Opposite/floor leg should stay relatively close to horizontal.
    if selected.get("opposite_leg_angle") is not None and selected["opposite_leg_angle"] > 25:
        diagnostic_flags.append("opposite_leg_lifted")

    if aslr_angle < 45:
        score = 40.0
    elif aslr_angle < 70:
        score = 60.0 + ((aslr_angle - 45.0) / 25.0) * 19.0
    else:
        score = 85.0 + (min(aslr_angle, 110.0) - 70.0) / 40.0 * 15.0

    # Small validity penalties only.
    if "raised_knee_bent" in diagnostic_flags:
        score -= 6.0
    if "opposite_leg_lifted" in diagnostic_flags:
        score -= 4.0
    if selected["keypoint_confidence"]["ankle"] < MIN_GOOD_CONF:
        score -= 3.0

    score = max(0.0, min(100.0, score))

    aslr_thr = make_thresholds(
        "deg",
        0,
        180,
        [
            {"label": "Red", "min": 0, "max": 45, "color": "red"},
            {"label": "Yellow", "min": 45, "max": 70, "color": "yellow"},
            {"label": "Green", "min": 70, "max": 180, "color": "green"},
        ],
        aslr_angle
    )

    conf_out = max(0.0, min(1.0, float(selected["confidence"])))

    quality_label = "good"
    if conf_out < 0.35 or "insufficient_required_keypoints" in diagnostic_flags:
        quality_label = "low"
    elif conf_out < 0.55 or diagnostic_flags:
        quality_label = "moderate"

    return {
        "score": round(float(score), 1),
        "confidence": round(conf_out, 3),
        "metrics": {
            "aslr_angle": round(float(aslr_angle), 2),
            "requested_side": requested_side,
            "detected_coco_side": detected_coco_side,
            "side": requested_side,
            "angle_method": selected["angle_method"],
            "quality_label": quality_label,
            "diagnostic_flags": diagnostic_flags,
            "left_candidate": {
                "angle": round(float(left_candidate["angle"]), 2),
                "selection_score": round(float(left_candidate["selection_score"]), 2),
                "confidence": round(float(left_candidate["confidence"]), 3),
                "keypoint_confidence": left_candidate["keypoint_confidence"],
            },
            "right_candidate": {
                "angle": round(float(right_candidate["angle"]), 2),
                "selection_score": round(float(right_candidate["selection_score"]), 2),
                "confidence": round(float(right_candidate["confidence"]), 3),
                "keypoint_confidence": right_candidate["keypoint_confidence"],
            },
            "hip_to_ankle_angle": round(float(selected["hip_to_ankle_angle"]), 2),
            "hip_to_knee_angle": round(float(selected["hip_to_knee_angle"]), 2),
            "knee_to_ankle_angle": round(float(selected["knee_to_ankle_angle"]), 2),
            "raised_knee_angle": round(float(selected["raised_knee_angle"]), 2) if selected["raised_knee_angle"] is not None else None,
            "opposite_leg_angle": round(float(selected["opposite_leg_angle"]), 2) if selected["opposite_leg_angle"] is not None else None,
            "keypoint_confidence": selected["keypoint_confidence"]
        },
        "thresholds": {
            "aslr_angle": aslr_thr
        }
    }


def compute_composite(posture, shoulder_r, shoulder_l, squat, aslr_r=None, aslr_l=None):
    shoulder = None
    if shoulder_r is not None and shoulder_l is not None:
        shoulder = min(float(shoulder_r), float(shoulder_l))
    elif shoulder_r is not None:
        shoulder = float(shoulder_r)
    elif shoulder_l is not None:
        shoulder = float(shoulder_l)

    aslr = None
    if aslr_r is not None and aslr_l is not None:
        aslr = min(float(aslr_r), float(aslr_l))
    elif aslr_r is not None:
        aslr = float(aslr_r)
    elif aslr_l is not None:
        aslr = float(aslr_l)

    parts = []
    if posture is not None:
        parts.append((float(posture), 0.30))
    if shoulder is not None:
        parts.append((float(shoulder), 0.25))
    if squat is not None:
        parts.append((float(squat), 0.25))
    if aslr is not None:
        parts.append((float(aslr), 0.20))

    if not parts:
        return None

    wsum = sum(w for _, w in parts)
    composite = sum(val * w for val, w in parts) / wsum
    return round(float(composite), 1)


@app.post("/start_session")
def start_session(user_email: str = Form(...), intake_json: str = Form(None)):
    if supabase is None:
        return {"error": "Supabase is not configured on server."}

    intake_data = safe_json_loads(intake_json)

    session_row = {
        "user_email": user_email,
        "status": "in_progress"
    }

    # Only saved in screenings for now.
    # If you add intake_json to sessions later, uncomment this:
    # session_row["intake_json"] = intake_data

    resp = supabase.table("sessions").insert(session_row).execute()

    return {
        "session_id": resp.data[0]["id"],
        "intake_json": intake_data
    }


@app.post("/finalize_session")
def finalize_session(session_id: str = Form(...)):
    if supabase is None:
        return {"error": "Supabase is not configured on server."}

    s = supabase.table("sessions").select("*").eq("id", session_id).limit(1).execute()
    if not s.data:
        return {"error": "Session not found"}

    row = s.data[0]
    composite = compute_composite(
        row.get("posture_score"),
        row.get("shoulder_right_score"),
        row.get("shoulder_left_score"),
        row.get("squat_score"),
        row.get("aslr_right_score"),
        row.get("aslr_left_score"),
    )

    supabase.table("sessions").update({
        "composite_score": composite,
        "status": "completed"
    }).eq("id", session_id).execute()

    return {
        "session_id": session_id,
        "status": "completed",
        "posture_score": row.get("posture_score"),
        "shoulder_right_score": row.get("shoulder_right_score"),
        "shoulder_left_score": row.get("shoulder_left_score"),
        "squat_score": row.get("squat_score"),
        "aslr_right_score": row.get("aslr_right_score"),
        "aslr_left_score": row.get("aslr_left_score"),
        "composite_score": composite
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    user_email: str = Form(...),
    test_type: str = Form(...),
    session_id: str = Form(...),
    intake_json: str = Form(None)
):
    if supabase is None:
        return {"error": "Supabase is not configured on server."}

    intake_data = safe_json_loads(intake_json)

    img_bytes = await image.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image"}

    h, w = img.shape[:2]
    max_side = 960
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    yolo_conf = 0.20 if str(test_type).startswith("aslr") else 0.50
    res = model(img, conf=yolo_conf, classes=[0])
    if res[0].keypoints is None or len(res[0].keypoints.xy) == 0:
        return {"error": "No person detected"}

    boxes = res[0].boxes.xyxy.cpu().numpy()
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    main_idx = int(np.argmax(areas))

    xy = res[0].keypoints.xy[main_idx].cpu().numpy()
    conf = res[0].keypoints.conf[main_idx].cpu().numpy()

    if test_type == "posture_side":
        result = analyze_posture(xy, conf)
        session_update = {"posture_score": result["score"]}
    elif test_type == "shoulder_right":
        result = analyze_shoulder(xy, conf, "RIGHT")
        session_update = {"shoulder_right_score": result["score"]}
    elif test_type == "shoulder_left":
        result = analyze_shoulder(xy, conf, "LEFT")
        session_update = {"shoulder_left_score": result["score"]}
    elif test_type == "squat":
        result = analyze_squat(xy, conf)
        session_update = {"squat_score": result["score"]}
    elif test_type == "aslr_right":
        result = analyze_aslr(xy, conf, "RIGHT")
        session_update = {"aslr_right_score": result["score"]}
    elif test_type == "aslr_left":
        result = analyze_aslr(xy, conf, "LEFT")
        session_update = {"aslr_left_score": result["score"]}
    else:
        return {"error": "Invalid test_type"}

    row = {
        "user_email": user_email,
        "session_id": session_id,
        "test_type": test_type,
        "score": float(result["score"]),
        "confidence": float(result["confidence"]),
        "metrics": result["metrics"],
        "thresholds": result.get("thresholds"),
        "intake_json": intake_data,
        "annotated_image_url": None
    }

    if test_type == "posture_side":
        row["neck_angle_deg"] = result["metrics"].get("neck_angle")
        row["thoracic_angle_deg"] = result["metrics"].get("thoracic_angle")
        row["pelvic_proxy_angle_deg"] = result["metrics"].get("pelvic_proxy_angle")
        row["side_used"] = result["metrics"].get("side_used")

    elif test_type in ["shoulder_right", "shoulder_left"]:
        row["shoulder_flexion_angle_deg"] = result["metrics"].get("shoulder_flexion_angle")
        row["shoulder_side"] = result["metrics"].get("side")

    elif test_type == "squat":
        row["squat_knee_angle_deg"] = result["metrics"].get("knee_angle")
        row["squat_trunk_lean_deg"] = result["metrics"].get("trunk_lean")

    supabase.table("screenings").insert(row).execute()
    supabase.table("sessions").update(session_update).eq("id", session_id).execute()

    return {
        "user_email": user_email,
        "session_id": session_id,
        "test_type": test_type,
        "score": result["score"],
        "confidence": result["confidence"],
        "metrics": result["metrics"],
        "thresholds": result.get("thresholds"),
        "intake_json": intake_data,
        "annotated_image_url": None
    }



def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def build_screening_row(user_email, session_id, test_type, result, intake_data):
    row = {
        "user_email": user_email,
        "session_id": session_id,
        "test_type": test_type,
        "score": float(result["score"]),
        "confidence": float(result["confidence"]),
        "metrics": result["metrics"],
        "thresholds": result.get("thresholds"),
        "intake_json": intake_data,
        "annotated_image_url": None
    }

    if test_type == "posture_side":
        row["neck_angle_deg"] = result["metrics"].get("neck_angle")
        row["thoracic_angle_deg"] = result["metrics"].get("thoracic_angle")
        row["pelvic_proxy_angle_deg"] = result["metrics"].get("pelvic_proxy_angle")
        row["side_used"] = result["metrics"].get("side_used")

    elif test_type in ["shoulder_right", "shoulder_left"]:
        row["shoulder_flexion_angle_deg"] = result["metrics"].get("shoulder_flexion_angle")
        row["shoulder_side"] = result["metrics"].get("side")

    elif test_type == "squat":
        row["squat_knee_angle_deg"] = result["metrics"].get("knee_angle")
        row["squat_trunk_lean_deg"] = result["metrics"].get("trunk_lean")

    return row


def run_yolo_analysis_from_bytes(img_bytes, test_type):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    h, w = img.shape[:2]
    max_side = 960
    scale = max_side / max(h, w)

    if scale < 1.0:
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    yolo_conf = 0.20 if str(test_type).startswith("aslr") else 0.50
    res = model(img, conf=yolo_conf, classes=[0])

    if res[0].keypoints is None or len(res[0].keypoints.xy) == 0:
        raise ValueError("No person detected")

    boxes = res[0].boxes.xyxy.cpu().numpy()
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    main_idx = int(np.argmax(areas))

    xy = res[0].keypoints.xy[main_idx].cpu().numpy()
    conf = res[0].keypoints.conf[main_idx].cpu().numpy()

    if test_type == "posture_side":
        result = analyze_posture(xy, conf)
        session_update = {"posture_score": result["score"]}

    elif test_type == "shoulder_right":
        result = analyze_shoulder(xy, conf, "RIGHT")
        session_update = {"shoulder_right_score": result["score"]}

    elif test_type == "shoulder_left":
        result = analyze_shoulder(xy, conf, "LEFT")
        session_update = {"shoulder_left_score": result["score"]}

    elif test_type == "squat":
        result = analyze_squat(xy, conf)
        session_update = {"squat_score": result["score"]}

    elif test_type == "aslr_right":
        result = analyze_aslr(xy, conf, "RIGHT")
        session_update = {"aslr_right_score": result["score"]}

    elif test_type == "aslr_left":
        result = analyze_aslr(xy, conf, "LEFT")
        session_update = {"aslr_left_score": result["score"]}

    else:
        raise ValueError("Invalid test_type")

    return result, session_update


def process_analysis_job(job_id: str):
    if supabase is None:
        return

    try:
        resp = (
            supabase.table("analysis_jobs")
            .select("*")
            .eq("id", job_id)
            .limit(1)
            .execute()
        )

        if not resp.data:
            return

        job = resp.data[0]

        if job.get("status") == "completed":
            return

        supabase.table("analysis_jobs").update({
            "status": "processing",
            "started_at": utc_now_iso(),
            "error_message": None
        }).eq("id", job_id).execute()

        img_b64 = job.get("image_base64")
        if not img_b64:
            raise ValueError("Missing image_base64")

        img_bytes = base64.b64decode(img_b64)

        test_type = job.get("test_type")
        user_email = job.get("user_email")
        session_id = job.get("session_id")
        intake_data = job.get("intake_json")

        result, session_update = run_yolo_analysis_from_bytes(img_bytes, test_type)

        screening_row = build_screening_row(
            user_email=user_email,
            session_id=session_id,
            test_type=test_type,
            result=result,
            intake_data=intake_data
        )

        supabase.table("screenings").insert(screening_row).execute()
        supabase.table("sessions").update(session_update).eq("id", session_id).execute()

        supabase.table("analysis_jobs").update({
            "status": "completed",
            "completed_at": utc_now_iso(),
            "result_json": {
                "user_email": user_email,
                "session_id": session_id,
                "test_type": test_type,
                "score": result["score"],
                "confidence": result["confidence"],
                "metrics": result["metrics"],
                "thresholds": result.get("thresholds"),
                "intake_json": intake_data,
                "annotated_image_url": None
            },
            "error_message": None
        }).eq("id", job_id).execute()

    except Exception as e:
        supabase.table("analysis_jobs").update({
            "status": "failed",
            "completed_at": utc_now_iso(),
            "error_message": str(e)
        }).eq("id", job_id).execute()


@app.post("/submit_analysis")
async def submit_analysis(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    user_email: str = Form(...),
    test_type: str = Form(...),
    session_id: str = Form(...),
    intake_json: str = Form(None)
):
    if supabase is None:
        return {"error": "Supabase is not configured on server."}

    img_bytes = await image.read()

    job = {
        "session_id": session_id,
        "user_email": user_email,
        "test_type": test_type,
        "status": "queued",
        "image_base64": base64.b64encode(img_bytes).decode("utf-8"),
        "intake_json": safe_json_loads(intake_json)
    }

    resp = supabase.table("analysis_jobs").insert(job).execute()
    job_id = resp.data[0]["id"]

    background_tasks.add_task(process_analysis_job, job_id)

    return {
        "job_id": job_id,
        "status": "queued"
    }


@app.get("/job_status/{job_id}")
def job_status(job_id: str, background_tasks: BackgroundTasks):
    if supabase is None:
        return {"error": "Supabase is not configured on server."}

    resp = (
        supabase.table("analysis_jobs")
        .select("*")
        .eq("id", job_id)
        .limit(1)
        .execute()
    )

    if not resp.data:
        return {"error": "Job not found"}

    job = resp.data[0]

    if job.get("status") == "queued":
        background_tasks.add_task(process_analysis_job, job_id)

    return {
        "job_id": job.get("id"),
        "session_id": job.get("session_id"),
        "user_email": job.get("user_email"),
        "test_type": job.get("test_type"),
        "status": job.get("status"),
        "result": job.get("result_json"),
        "error_message": job.get("error_message"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at")
    }



@app.get("/report")
def report(session_id: str, lang: str = "fr"):
    """
    Bilingual report endpoint.
    Use /report?session_id=...&lang=fr or /report?session_id=...&lang=en
    Keeps FR keys for backward compatibility and adds EN equivalents.
    """
    lang = "en" if str(lang).lower().startswith("en") else "fr"

    if supabase is None:
        return {"error": "Supabase not configured"}

    s_resp = supabase.table("sessions").select("*").eq("id", session_id).limit(1).execute()
    if not s_resp.data:
        return {"error": "Session not found"}

    session = s_resp.data[0]

    scr_resp = supabase.table("screenings").select("*").eq("session_id", session_id).execute()
    screenings = scr_resp.data or []

    tests_found = [x.get("test_type") for x in screenings if x.get("test_type")]

    intake_context = None
    for r in screenings:
        if r.get("intake_json"):
            intake_context = r.get("intake_json")
            break

    def txt(fr, en):
        return en if lang == "en" else fr

    LABELS = {
        "neck_angle": ("Angle cervical", "Cervical alignment"),
        "thoracic_angle": ("Angle thoracique", "Thoracic alignment"),
        "pelvic_proxy_angle": ("Alignement tronc-bassin", "Trunk-pelvis alignment"),
        "shoulder_right_flexion": ("Flexion épaule droite", "Right shoulder flexion"),
        "shoulder_left_flexion": ("Flexion épaule gauche", "Left shoulder flexion"),
        "squat_knee_angle": ("Angle du genou", "Knee angle"),
        "squat_trunk_lean": ("Inclinaison du tronc", "Trunk lean"),
        "aslr_right_angle": ("ASLR jambe droite", "Right ASLR"),
        "aslr_left_angle": ("ASLR jambe gauche", "Left ASLR"),
        "aslr_title": ("Active Straight Leg Raise", "Active Straight Leg Raise"),
        "posture_title": ("Posture (vue de profil)", "Posture (side view)"),
        "shoulders_title": ("Mobilité des épaules", "Shoulder mobility"),
        "squat_title": ("Squat (contrôle et mobilité)", "Squat (control and mobility)"),
    }

    def label_pair(key):
        fr, en = LABELS.get(key, (key, key))
        return fr, en, txt(fr, en)

    def get_test(tt):
        for r in screenings:
            if r.get("test_type") == tt:
                return r
        return None

    posture = get_test("posture_side")
    sh_r = get_test("shoulder_right")
    sh_l = get_test("shoulder_left")
    squat = get_test("squat")
    aslr_r = get_test("aslr_right")
    aslr_l = get_test("aslr_left")

    flexilab_score = session.get("composite_score", None)

    if flexilab_score is None:
        posture_score = session.get("posture_score", None) or (posture.get("score") if posture else None)
        sh_r_score = session.get("shoulder_right_score", None) or (sh_r.get("score") if sh_r else None)
        sh_l_score = session.get("shoulder_left_score", None) or (sh_l.get("score") if sh_l else None)
        squat_score = session.get("squat_score", None) or (squat.get("score") if squat else None)
        aslr_r_score = session.get("aslr_right_score", None) or (aslr_r.get("score") if aslr_r else None)
        aslr_l_score = session.get("aslr_left_score", None) or (aslr_l.get("score") if aslr_l else None)

        shoulder = None
        if sh_r_score is not None and sh_l_score is not None:
            shoulder = min(float(sh_r_score), float(sh_l_score))
        elif sh_r_score is not None:
            shoulder = float(sh_r_score)
        elif sh_l_score is not None:
            shoulder = float(sh_l_score)

        flexilab_score = compute_composite(
            posture_score,
            sh_r_score,
            sh_l_score,
            squat_score,
            aslr_r_score,
            aslr_l_score
        )

    def risk_from_score(score):
        if score is None:
            return {
                "label": "Unknown",
                "color": "grey",
                "description_fr": "Session incomplète : termine tous les tests pour un score global.",
                "description_en": "Incomplete session: complete all tests to obtain a global score.",
                "description": txt("Session incomplète : termine tous les tests pour un score global.", "Incomplete session: complete all tests to obtain a global score.")
            }

        score = float(score)
        if score >= 85:
            return {
                "label": "Low",
                "color": "green",
                "description_fr": "Bon équilibre global. Quelques ajustements possibles.",
                "description_en": "Good overall balance. Minor adjustments may be useful.",
                "description": txt("Bon équilibre global. Quelques ajustements possibles.", "Good overall balance. Minor adjustments may be useful.")
            }

        if score >= 70:
            return {
                "label": "Moderate",
                "color": "yellow",
                "description_fr": "Profil intermédiaire : plusieurs axes d’amélioration.",
                "description_en": "Intermediate profile: several areas can be improved.",
                "description": txt("Profil intermédiaire : plusieurs axes d’amélioration.", "Intermediate profile: several areas can be improved.")
            }

        return {
            "label": "High",
            "color": "red",
            "description_fr": "Priorité d’amélioration : plusieurs indicateurs hors zone cible.",
            "description_en": "Improvement priority: several indicators are outside the target zone.",
            "description": txt("Priorité d’amélioration : plusieurs indicateurs hors zone cible.", "Improvement priority: several indicators are outside the target zone.")
        }

    risk_category = risk_from_score(flexilab_score)

    def thr_item(thresholds, key):
        if not thresholds:
            return None
        v = thresholds.get(key)
        return v if isinstance(v, dict) else None

    def rating_word(rating):
        if rating == "green":
            return txt("Bon", "Good")
        if rating == "yellow":
            return txt("À améliorer", "Improve")
        if rating == "red":
            return txt("Priorité", "Priority")
        return txt("Info", "Info")

    def insight_posture(label_fr, label_en, rating):
        if rating == "green":
            fr = f"{label_fr} satisfaisant."
            en = f"{label_en} is satisfactory."
        elif rating == "yellow":
            fr = f"{label_fr} à améliorer légèrement."
            en = f"{label_en} can be slightly improved."
        elif rating == "red":
            fr = f"{label_fr} prioritaire à corriger."
            en = f"{label_en} is a priority to correct."
        else:
            fr = f"{label_fr} : données insuffisantes."
            en = f"{label_en}: insufficient data."
        return fr, en, txt(fr, en)

    def insight_shoulder(rating):
        pairs = {
            "green": ("Mobilité au-dessus de la tête très bonne.", "Overhead mobility is very good."),
            "yellow": ("Légère limitation par rapport à l'objectif.", "Slight limitation compared with the target."),
            "red": ("Limitation marquée : priorité mobilité.", "Marked limitation: mobility is a priority."),
        }
        fr, en = pairs.get(rating, ("Données insuffisantes.", "Insufficient data."))
        return fr, en, txt(fr, en)

    def insight_squat(label_fr, label_en, rating):
        if rating == "green":
            fr = f"{label_fr} satisfaisant."
            en = f"{label_en} is satisfactory."
        elif rating == "yellow":
            fr = f"{label_fr} à améliorer."
            en = f"{label_en} can be improved."
        elif rating == "red":
            fr = f"{label_fr} prioritaire à améliorer."
            en = f"{label_en} is a priority to improve."
        else:
            fr = f"{label_fr} : données insuffisantes."
            en = f"{label_en}: insufficient data."
        return fr, en, txt(fr, en)

    def item_obj(item_id, value, unit, rating, thresholds):
        label_fr, label_en, label = label_pair(item_id)
        if item_id in ["neck_angle", "thoracic_angle", "pelvic_proxy_angle"]:
            ins_fr, ins_en, ins = insight_posture(label_fr, label_en, rating)
        elif item_id in ["shoulder_right_flexion", "shoulder_left_flexion"]:
            ins_fr, ins_en, ins = insight_shoulder(rating)
        elif item_id == "squat_knee_angle":
            ins_fr, ins_en, ins = insight_squat("Profondeur", "Depth", rating)
        elif item_id == "squat_trunk_lean":
            ins_fr, ins_en, ins = insight_squat("Contrôle du tronc", "Trunk control", rating)
        elif item_id in ["aslr_right_angle", "aslr_left_angle"]:
            if rating == "green":
                ins_fr, ins_en = "Mobilité active de hanche satisfaisante.", "Active hip mobility is satisfactory."
            elif rating == "yellow":
                ins_fr, ins_en = "Mobilité active de hanche à améliorer.", "Active hip mobility can be improved."
            elif rating == "red":
                ins_fr, ins_en = "Restriction importante : priorité mobilité hanche/ischio-jambiers.", "Important restriction: hip/hamstring mobility is a priority."
            else:
                ins_fr, ins_en = "Données insuffisantes.", "Insufficient data."
            ins = txt(ins_fr, ins_en)
        else:
            ins_fr, ins_en, ins = ("", "", "")
        return {
            "id": item_id,
            "label_fr": label_fr,
            "label_en": label_en,
            "label": label,
            "value": value,
            "unit": unit,
            "rating": rating,
            "rating_label": rating_word(rating),
            "thresholds": thresholds,
            "short_insight_fr": ins_fr,
            "short_insight_en": ins_en,
            "short_insight": ins,
        }

    sections = []

    if posture:
        m = posture.get("metrics") or {}
        t = posture.get("thresholds") or {}
        neck_thr = thr_item(t, "neck_angle")
        thor_thr = thr_item(t, "thoracic_angle")
        pelv_thr = thr_item(t, "pelvic_proxy_angle")
        fr, en, title = label_pair("posture_title")
        sections.append({
            "id": "posture",
            "title_fr": fr,
            "title_en": en,
            "title": title,
            "items": [
                item_obj("neck_angle", m.get("neck_angle"), "°", (neck_thr or {}).get("rating"), neck_thr),
                item_obj("thoracic_angle", m.get("thoracic_angle"), "°", (thor_thr or {}).get("rating"), thor_thr),
                item_obj("pelvic_proxy_angle", m.get("pelvic_proxy_angle"), "°", (pelv_thr or {}).get("rating"), pelv_thr),
            ]
        })

    if sh_r or sh_l:
        items = []
        asym = None
        if sh_r:
            mr = sh_r.get("metrics") or {}
            tr = sh_r.get("thresholds") or {}
            thr = thr_item(tr, "shoulder_flexion")
            items.append(item_obj("shoulder_right_flexion", mr.get("shoulder_flexion_angle"), "°", (thr or {}).get("rating"), thr))
        if sh_l:
            ml = sh_l.get("metrics") or {}
            tl = sh_l.get("thresholds") or {}
            thr = thr_item(tl, "shoulder_flexion")
            items.append(item_obj("shoulder_left_flexion", ml.get("shoulder_flexion_angle"), "°", (thr or {}).get("rating"), thr))
        if sh_r and sh_l:
            vr = (sh_r.get("metrics") or {}).get("shoulder_flexion_angle")
            vl = (sh_l.get("metrics") or {}).get("shoulder_flexion_angle")
            if vr is not None and vl is not None:
                asym_deg = abs(float(vr) - float(vl))
                if asym_deg <= 5:
                    a_rating = "green"
                    fr_txt = "Symétrie satisfaisante."
                    en_txt = "Satisfactory symmetry."
                elif asym_deg <= 12:
                    a_rating = "yellow"
                    fr_txt = "Asymétrie légère entre droite et gauche."
                    en_txt = "Slight asymmetry between right and left."
                else:
                    a_rating = "red"
                    fr_txt = "Asymétrie importante : priorité équilibre D/G."
                    en_txt = "Important asymmetry: right/left balance is a priority."
                asym = {
                    "value_deg": round(asym_deg, 2),
                    "rating": a_rating,
                    "short_insight_fr": fr_txt,
                    "short_insight_en": en_txt,
                    "short_insight": txt(fr_txt, en_txt)
                }
        fr, en, title = label_pair("shoulders_title")
        sections.append({"id": "shoulders", "title_fr": fr, "title_en": en, "title": title, "items": items, "asymmetry": asym})

    if squat:
        ms = squat.get("metrics") or {}
        ts = squat.get("thresholds") or {}
        knee_thr = thr_item(ts, "knee_angle")
        trunk_thr = thr_item(ts, "trunk_lean")
        fr, en, title = label_pair("squat_title")
        sections.append({
            "id": "squat",
            "title_fr": fr,
            "title_en": en,
            "title": title,
            "items": [
                item_obj("squat_knee_angle", ms.get("knee_angle"), "°", (knee_thr or {}).get("rating"), knee_thr),
                item_obj("squat_trunk_lean", ms.get("trunk_lean"), "°", (trunk_thr or {}).get("rating"), trunk_thr),
            ]
        })


    if aslr_r or aslr_l:
        items = []
        asym = None

        if aslr_r:
            mr = aslr_r.get("metrics") or {}
            tr = aslr_r.get("thresholds") or {}
            thr = thr_item(tr, "aslr_angle")
            items.append(item_obj("aslr_right_angle", mr.get("aslr_angle"), "°", (thr or {}).get("rating"), thr))

        if aslr_l:
            ml = aslr_l.get("metrics") or {}
            tl = aslr_l.get("thresholds") or {}
            thr = thr_item(tl, "aslr_angle")
            items.append(item_obj("aslr_left_angle", ml.get("aslr_angle"), "°", (thr or {}).get("rating"), thr))

        if aslr_r and aslr_l:
            vr = (aslr_r.get("metrics") or {}).get("aslr_angle")
            vl = (aslr_l.get("metrics") or {}).get("aslr_angle")
            if vr is not None and vl is not None:
                asym_deg = abs(float(vr) - float(vl))
                if asym_deg <= 5:
                    a_rating = "green"
                    fr_txt = "Symétrie ASLR satisfaisante."
                    en_txt = "Satisfactory ASLR symmetry."
                elif asym_deg <= 12:
                    a_rating = "yellow"
                    fr_txt = "Asymétrie ASLR légère entre droite et gauche."
                    en_txt = "Slight ASLR asymmetry between right and left."
                else:
                    a_rating = "red"
                    fr_txt = "Asymétrie ASLR importante : priorité mobilité D/G."
                    en_txt = "Important ASLR asymmetry: right/left mobility is a priority."
                asym = {
                    "value_deg": round(asym_deg, 2),
                    "rating": a_rating,
                    "short_insight_fr": fr_txt,
                    "short_insight_en": en_txt,
                    "short_insight": txt(fr_txt, en_txt)
                }

        fr, en, title = label_pair("aslr_title")
        sections.append({
            "id": "aslr",
            "title_fr": fr,
            "title_en": en,
            "title": title,
            "items": items,
            "asymmetry": asym
        })

    candidates = []

    def add_candidate(sev, title_fr, title_en, why_fr, why_en):
        candidates.append({
            "severity": sev,
            "title_fr": title_fr,
            "title_en": title_en,
            "title": txt(title_fr, title_en),
            "why_fr": why_fr,
            "why_en": why_en,
            "why": txt(why_fr, why_en),
        })

    if posture:
        t = posture.get("thresholds") or {}
        na = thr_item(t, "neck_angle")
        ta = thr_item(t, "thoracic_angle")
        pa = thr_item(t, "pelvic_proxy_angle")
        if (na or {}).get("rating") in ["red", "yellow"]:
            add_candidate((na or {}).get("rating"), "Alignement cervical", "Cervical alignment", "L’angle cervical est hors de la zone optimale.", "Cervical alignment is outside the optimal zone.")
        if (ta or {}).get("rating") in ["red", "yellow"]:
            add_candidate((ta or {}).get("rating"), "Alignement thoracique", "Thoracic alignment", "L’angle thoracique est hors de la zone optimale.", "Thoracic alignment is outside the optimal zone.")
        if (pa or {}).get("rating") in ["red", "yellow"]:
            add_candidate((pa or {}).get("rating"), "Alignement tronc-bassin", "Trunk-pelvis alignment", "Le contrôle tronc-bassin est hors de la zone optimale.", "Trunk-pelvis control is outside the optimal zone.")

    if sh_r:
        thr = thr_item((sh_r.get("thresholds") or {}), "shoulder_flexion")
        if (thr or {}).get("rating") in ["red", "yellow"]:
            add_candidate((thr or {}).get("rating"), "Mobilité épaule droite", "Right shoulder mobility", "La flexion de l’épaule droite est sous l’objectif.", "Right shoulder flexion is below the target.")

    if sh_l:
        thr = thr_item((sh_l.get("thresholds") or {}), "shoulder_flexion")
        if (thr or {}).get("rating") in ["red", "yellow"]:
            add_candidate((thr or {}).get("rating"), "Mobilité épaule gauche", "Left shoulder mobility", "La flexion de l’épaule gauche est sous l’objectif.", "Left shoulder flexion is below the target.")

    if aslr_r:
        thr = thr_item((aslr_r.get("thresholds") or {}), "aslr_angle")
        if (thr or {}).get("rating") in ["red", "yellow"]:
            add_candidate((thr or {}).get("rating"), "Mobilité ASLR droite", "Right ASLR mobility", "L’élévation active de la jambe droite est sous l’objectif.", "Right active straight leg raise is below the target.")

    if aslr_l:
        thr = thr_item((aslr_l.get("thresholds") or {}), "aslr_angle")
        if (thr or {}).get("rating") in ["red", "yellow"]:
            add_candidate((thr or {}).get("rating"), "Mobilité ASLR gauche", "Left ASLR mobility", "L’élévation active de la jambe gauche est sous l’objectif.", "Left active straight leg raise is below the target.")

    if squat:
        ts = squat.get("thresholds") or {}
        tr = thr_item(ts, "trunk_lean")
        kn = thr_item(ts, "knee_angle")
        if (tr or {}).get("rating") in ["red", "yellow"]:
            add_candidate((tr or {}).get("rating"), "Inclinaison du tronc en squat", "Trunk lean during squat", "L’inclinaison du tronc est hors zone cible.", "Trunk lean is outside the target zone.")
        if (kn or {}).get("rating") in ["red", "yellow"]:
            add_candidate((kn or {}).get("rating"), "Profondeur du squat", "Squat depth", "L’angle du genou est hors zone cible.", "Knee angle is outside the target zone.")

    sev_order = {"red": 0, "yellow": 1, "green": 2, "unknown": 3, None: 4}
    candidates.sort(key=lambda x: sev_order.get(x["severity"], 9))

    top_priorities = []
    for i, c in enumerate(candidates[:3], start=1):
        top_priorities.append({
            "id": f"priority_{i}",
            "title_fr": c["title_fr"],
            "title_en": c["title_en"],
            "title": c["title"],
            "severity": c["severity"],
            "why_fr": c["why_fr"],
            "why_en": c["why_en"],
            "why": c["why"],
        })

    return {
        "session_id": session_id,
        "language": lang,
        "user_email": session.get("user_email"),
        "created_at": session.get("created_at"),
        "intake_context": intake_context,
        "flexilab_score": flexilab_score,
        "risk_category": risk_category,
        "sections": sections,
        "top_priorities": top_priorities,
        "next_step_fr": "Refais le screening dans 14 jours pour vérifier l'évolution.",
        "next_step_en": "Repeat the screening in 14 days to check progress.",
        "next_step": txt("Refais le screening dans 14 jours pour vérifier l'évolution.", "Repeat the screening in 14 days to check progress."),
        "debug": {"tests_found": tests_found}
    }

@app.get("/program")
def program(session_id: str, lang: str = "fr"):
    """
    Generate a FlexiLab 4-week corrective program from an existing screening session.

    Flow:
    1. Reuse the existing /report logic.
    2. Convert screening findings into movement-system priorities.
    3. Read exercise_library.json through program_engine.py.
    4. Return a structured 4-week corrective program.
    """
    report_data = report(session_id=session_id, lang=lang)

    if isinstance(report_data, dict) and report_data.get("error"):
        return report_data

    program_data = generate_program_from_report(
        report=report_data,
        lang=lang
    )

    return {
        "session_id": session_id,
        "language": lang,
        "report": report_data,
        "program": program_data
    }
