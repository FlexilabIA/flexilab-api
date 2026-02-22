from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import math
import os
import uuid

# Set env BEFORE importing YOLO to reduce warnings
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
    allow_origins=["*"],  # MVP. Later restrict to LearnWorlds domain(s)
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
model = YOLO("yolov8n-pose.pt")


@app.get("/health")
def health():
    return {"ok": True}


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

    left_score = conf[L_EAR] + conf[L_SH] + conf[L_HIP]
    right_score = conf[R_EAR] + conf[R_SH] + conf[R_HIP]

    if right_score >= left_score:
        ear, shoulder, hip = xy[R_EAR], xy[R_SH], xy[R_HIP]
        side = "RIGHT"
        quality = right_score / 3
    else:
        ear, shoulder, hip = xy[L_EAR], xy[L_SH], xy[L_HIP]
        side = "LEFT"
        quality = left_score / 3

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
        penalty(neck_angle, 10, 55, 1.2)
        + penalty(thoracic_angle, 5, 45, 1.0)
        + penalty(pelvic_proxy_angle, 5, 40, 0.8)
    )

    score = max(0.0, 100.0 - total_pen)
    conf_out = max(0.6, min(1.0, float(quality)))

    return {
        "score": round(score, 1),
        "confidence": round(conf_out, 3),
        "metrics": {
            "neck_angle": round(neck_angle, 2),
            "thoracic_angle": round(thoracic_angle, 2),
            "pelvic_proxy_angle": round(pelvic_proxy_angle, 2),
            "side_used": side,
        },
    }


def analyze_shoulder(xy, conf, side="RIGHT"):
    L_SH, R_SH = 5, 6
    L_EL, R_EL = 7, 8
    L_WR, R_WR = 9, 10

    if side == "RIGHT":
        sh, el, wr = xy[R_SH], xy[R_EL], xy[R_WR]
        c = (conf[R_SH] + conf[R_EL] + conf[R_WR]) / 3
    else:
        sh, el, wr = xy[L_SH], xy[L_EL], xy[L_WR]
        c = (conf[L_SH] + conf[L_EL] + conf[L_WR]) / 3

    dx = wr[0] - sh[0]
    dy = sh[1] - wr[1]
    angle = abs(math.degrees(math.atan2(dx, dy)))

    deficit = max(0.0, 170.0 - float(angle))
    score = max(0.0, 100.0 - deficit * 2.0)

    return {
        "score": round(score, 1),
        "confidence": round(max(0.6, min(1.0, float(c))), 3),
        "metrics": {"shoulder_flexion_angle": round(angle, 2), "side": side},
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

    trunk_dx = shoulder[0] - hip[0]
    trunk_dy = hip[1] - shoulder[1]
    trunk_angle = abs(math.degrees(math.atan2(trunk_dx, trunk_dy)))

    # Simple v1 scoring: depth good if knee_angle ~80-110, trunk penalty if lean >20
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

    return {
        "score": round(score, 1),
        "confidence": round(conf_out, 3),
        "metrics": {"knee_angle": round(knee_angle, 2), "trunk_lean": round(trunk_angle, 2)},
    }


def compute_composite(posture, shoulder_r, shoulder_l, squat):
    # Use worst shoulder side for screening
    shoulder = None
    if shoulder_r is not None and shoulder_l is not None:
        shoulder = min(shoulder_r, shoulder_l)
    elif shoulder_r is not None:
        shoulder = shoulder_r
    elif shoulder_l is not None:
        shoulder = shoulder_l

    # If some tests missing, compute from what exists (normalized weights)
    parts = []
    if posture is not None:
        parts.append(("posture", posture, 0.4))
    if shoulder is not None:
        parts.append(("shoulder", shoulder, 0.3))
    if squat is not None:
        parts.append(("squat", squat, 0.3))

    if not parts:
        return None

    wsum = sum(w for _, _, w in parts)
    composite = sum(val * w for _, val, w in parts) / wsum
    return round(float(composite), 1)


@app.post("/start_session")
def start_session(user_email: str = Form(...)):
    if supabase is None:
        return {"error": "Supabase is not configured on server."}

    resp = supabase.table("sessions").insert({
        "user_email": user_email,
        "status": "in_progress"
    }).execute()

    # Supabase returns inserted rows in data
    session_id = resp.data[0]["id"]
    return {"session_id": session_id}


@app.post("/finalize_session")
def finalize_session(session_id: str = Form(...)):
    if supabase is None:
        return {"error": "Supabase is not configured on server."}

    # Read session row
    s = supabase.table("sessions").select("*").eq("id", session_id).limit(1).execute()
    if not s.data:
        return {"error": "Session not found"}

    row = s.data[0]
    posture = row.get("posture_score")
    sr = row.get("shoulder_right_score")
    sl = row.get("shoulder_left_score")
    squat = row.get("squat_score")

    composite = compute_composite(posture, sr, sl, squat)

    supabase.table("sessions").update({
        "composite_score": composite,
        "status": "completed"
    }).eq("id", session_id).execute()

    return {
        "session_id": session_id,
        "status": "completed",
        "posture_score": posture,
        "shoulder_right_score": sr,
        "shoulder_left_score": sl,
        "squat_score": squat,
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

    # Run pose
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

    # Annotated image upload
    annotated = res[0].plot()
    ok, png = cv2.imencode(".png", annotated)
    if not ok:
        return {"error": "Failed to encode annotated image"}

    path = f"{user_email}/{session_id}/{test_type}/{uuid.uuid4()}.png"
    supabase.storage.from_("screening").upload(
        path,
        png.tobytes(),
        {"content-type": "image/png"}
    )
    annotated_url = supabase.storage.from_("screening").get_public_url(path)

    # Insert screening row
    supabase.table("screenings").insert({
        "user_email": user_email,
        "session_id": session_id,
        "test_type": test_type,
        "score": result["score"],
        "confidence": result["confidence"],
        "metrics": result["metrics"],
        "annotated_image_url": annotated_url
    }).execute()

    # Update session partial scores
    supabase.table("sessions").update(session_update).eq("id", session_id).execute()

    return {
        "user_email": user_email,
        "session_id": session_id,
        "test_type": test_type,
        "score": result["score"],
        "confidence": result["confidence"],
        "metrics": result["metrics"],
        "annotated_image_url": annotated_url
    }
