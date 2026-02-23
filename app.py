from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import math
import os

# Must be set BEFORE importing YOLO (helps reduce Ultralytics warnings)
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

from ultralytics import YOLO
from supabase import create_client

# Supabase connection (backend only)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

supabase = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP (later restrict)
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
model = YOLO("yolov8n-pose.pt")


@app.get("/health")
def health():
    return {"ok": True}


# ----------------------------
# Threshold helpers
# ----------------------------
def make_thresholds(unit, scale_min, scale_max, bands, pointer_value):
    """
    bands: list of dicts like:
      [{"label":"Green","min":0,"max":10,"color":"green"}, ...]
    pointer_value: numeric
    returns: dict with rating and pointer_value clamped
    """
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


# ----------------------------
# Geometry helpers
# ----------------------------
def angle_to_vertical(p1, p2):
    """Angle (0..90) of segment p1->p2 relative to vertical axis."""
    dx = float(p2[0] - p1[0])
    dy = float(p1[1] - p2[1])
    ang = abs(math.degrees(math.atan2(dx, dy)))
    if ang > 90:
        ang = 180 - ang
    return ang


# ----------------------------
# Analysis functions
# ----------------------------
def analyze_posture(xy, conf):
    # COCO indices for YOLOv8 pose
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

    # Smooth penalty score (kept)
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

    # Thresholds (MVP)
    neck_thr = make_thresholds(
        unit="deg",
        scale_min=0, scale_max=60,
        bands=[
            {"label": "Green", "min": 0, "max": 10, "color": "green"},
            {"label": "Yellow", "min": 10, "max": 20, "color": "yellow"},
            {"label": "Red", "min": 20, "max": 60, "color": "red"},
        ],
        pointer_value=neck_angle
    )

    thor_thr = make_thresholds(
        unit="deg",
        scale_min=0, scale_max=45,
        bands=[
            {"label": "Green", "min": 0, "max": 5, "color": "green"},
            {"label": "Yellow", "min": 5, "max": 15, "color": "yellow"},
            {"label": "Red", "min": 15, "max": 45, "color": "red"},
        ],
        pointer_value=thoracic_angle
    )

    pelvis_thr = make_thresholds(
        unit="deg",
        scale_min=0, scale_max=45,
        bands=[
            {"label": "Green", "min": 0, "max": 5, "color": "green"},
            {"label": "Yellow", "min": 5, "max": 15, "color": "yellow"},
            {"label": "Red", "min": 15, "max": 45, "color": "red"},
        ],
        pointer_value=pelvic_proxy_angle
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
    Biomechanical: angle between trunk vector (shoulder->hip) and upper-arm vector (shoulder->elbow).
    ~180 = overhead flexion, ~90 = arm horizontal, ~0 = arm down along trunk
    """
    L_SH, R_SH = 5, 6
    L_EL, R_EL = 7, 8
    L_HIP, R_HIP = 11, 12

    if side == "RIGHT":
        sh, el, hip = xy[R_SH], xy[R_EL], xy[R_HIP]
        c = float(conf[R_SH] + conf[R_EL] + conf[R_HIP]) / 3.0
    else:
        sh, el, hip = xy[L_SH], xy[L_EL], xy[L_HIP]
        c = float(conf[L_SH] + conf[L_EL] + conf[L_HIP]) / 3.0

    v_trunk = hip - sh
    v_arm = el - sh

    denom = (np.linalg.norm(v_trunk) * np.linalg.norm(v_arm))
    if denom < 1e-6:
        shoulder_flexion = 0.0
    else:
        cosang = float(np.dot(v_trunk, v_arm) / denom)
        cosang = max(-1.0, min(1.0, cosang))
        shoulder_flexion = float(math.degrees(math.acos(cosang)))

    # Score per test (kept /100)
    deficit = max(0.0, 170.0 - shoulder_flexion)
    score = max(0.0, 100.0 - deficit * 2.0)

    conf_out = max(0.6, min(1.0, float(c)))

    shoulder_thr = make_thresholds(
        unit="deg",
        scale_min=0, scale_max=180,
        bands=[
            {"label": "Red", "min": 0, "max": 160, "color": "red"},
            {"label": "Yellow", "min": 160, "max": 170, "color": "yellow"},
            {"label": "Green", "min": 170, "max": 180, "color": "green"},
        ],
        pointer_value=shoulder_flexion
    )

    return {
        "score": round(score, 1),
        "confidence": round(conf_out, 3),
        "metrics": {
            "shoulder_flexion_angle": round(shoulder_flexion, 2),
            "side": side
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

    # v1 scoring
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
        unit="deg",
        scale_min=0, scale_max=60,
        bands=[
            {"label": "Green", "min": 0, "max": 15, "color": "green"},
            {"label": "Yellow", "min": 15, "max": 25, "color": "yellow"},
            {"label": "Red", "min": 25, "max": 60, "color": "red"},
        ],
        pointer_value=trunk_angle
    )

    # NOTE: knee thresholds depend on angle definition; this is MVP and we can tune
    knee_thr = make_thresholds(
        unit="deg",
        scale_min=60, scale_max=180,
        bands=[
            {"label": "Green", "min": 60, "max": 95, "color": "green"},
            {"label": "Yellow", "min": 95, "max": 110, "color": "yellow"},
            {"label": "Red", "min": 110, "max": 180, "color": "red"},
        ],
        pointer_value=knee_angle
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


def compute_composite(posture, shoulder_r, shoulder_l, squat):
    # Use worst shoulder side
    shoulder = None
    if shoulder_r is not None and shoulder_l is not None:
        shoulder = min(float(shoulder_r), float(shoulder_l))
    elif shoulder_r is not None:
        shoulder = float(shoulder_r)
    elif shoulder_l is not None:
        shoulder = float(shoulder_l)

    parts = []
    if posture is not None:
        parts.append((float(posture), 0.4))
    if shoulder is not None:
        parts.append((float(shoulder), 0.3))
    if squat is not None:
        parts.append((float(squat), 0.3))

    if not parts:
        return None

    wsum = sum(w for _, w in parts)
    composite = sum(val * w for val, w in parts) / wsum
    return round(float(composite), 1)


# ----------------------------
# API endpoints
# ----------------------------
@app.post("/start_session")
def start_session(user_email: str = Form(...)):
    if supabase is None:
        return {"error": "Supabase is not configured on server."}

    resp = supabase.table("sessions").insert({
        "user_email": user_email,
        "status": "in_progress"
    }).execute()

    return {"session_id": resp.data[0]["id"]}


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
        "composite_score": composite
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    user_email: str = Form(...),
    test_type: str = Form(...),
    session_id: str = Form(...)
):
    if supabase is None:
        return {"error": "Supabase is not configured on server."}

    # Decode image
    img_bytes = await image.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image"}

    # Resize for Render Free stability
    h, w = img.shape[:2]
    max_side = 960  # reduce to 720 if you still see memory issues
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    # Inference
    res = model(img, conf=0.5, classes=[0])
    if res[0].keypoints is None or len(res[0].keypoints.xy) == 0:
        return {"error": "No person detected"}

    # Select largest person
    boxes = res[0].boxes.xyxy.cpu().numpy()
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    main_idx = int(np.argmax(areas))

    xy = res[0].keypoints.xy[main_idx].cpu().numpy()
    conf = res[0].keypoints.conf[main_idx].cpu().numpy()

    # Route analysis
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
    else:
        return {"error": "Invalid test_type"}

    # Build DB row (angles in columns + JSON metrics + JSON thresholds)
    row = {
        "user_email": user_email,
        "session_id": session_id,
        "test_type": test_type,
        "score": float(result["score"]),
        "confidence": float(result["confidence"]),
        "metrics": result["metrics"],
        "thresholds": result.get("thresholds"),  # <- saved to DB
        "annotated_image_url": None  # Free mode
    }

    # Map metrics -> dedicated columns
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

    # Insert screening row
    supabase.table("screenings").insert(row).execute()

    # Update session partial score
    supabase.table("sessions").update(session_update).eq("id", session_id).execute()

    # Return response (with thresholds)
    return {
        "user_email": user_email,
        "session_id": session_id,
        "test_type": test_type,
        "score": result["score"],
        "confidence": result["confidence"],
        "metrics": result["metrics"],
        "thresholds": result.get("thresholds"),
        "annotated_image_url": None
    }
