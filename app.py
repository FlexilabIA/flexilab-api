from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import math
import os
import uuid

from ultralytics import YOLO
from supabase import create_client

# Avoid Ultralytics config warning
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# Supabase connection
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
            return 0
        a = min(angle, severe)
        t = (a - optimal) / (severe - optimal)
        return w * 30 * (t ** 2)

    total_pen = (
        penalty(neck_angle, 10, 55, 1.2) +
        penalty(thoracic_angle, 5, 45, 1.0) +
        penalty(pelvic_proxy_angle, 5, 40, 0.8)
    )

    score = max(0, 100 - total_pen)

    return {
        "score": round(score, 1),
        "confidence": round(float(quality), 3),
        "metrics": {
            "neck_angle": round(neck_angle, 2),
            "thoracic_angle": round(thoracic_angle, 2),
            "pelvic_proxy_angle": round(pelvic_proxy_angle, 2),
            "side_used": side
        }
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

    deficit = max(0, 170 - angle)
    score = max(0, 100 - deficit * 2)

    return {
        "score": round(score, 1),
        "confidence": round(float(c), 3),
        "metrics": {
            "shoulder_flexion_angle": round(angle, 2),
            "side": side
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
        math.degrees(
            math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
        )
    )
    if knee_angle > 180:
        knee_angle = 360 - knee_angle

    trunk_dx = shoulder[0] - hip[0]
    trunk_dy = hip[1] - shoulder[1]
    trunk_angle = abs(math.degrees(math.atan2(trunk_dx, trunk_dy)))

    depth_score = min(100, knee_angle)
    trunk_penalty = max(0, trunk_angle - 20) * 1.5
    score = max(0, depth_score - trunk_penalty)

    return {
        "score": round(score, 1),
        "confidence": round(float(np.mean(conf)), 3),
        "metrics": {
            "knee_angle": round(knee_angle, 2),
            "trunk_lean": round(trunk_angle, 2)
        }
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    user_email: str = Form(...),
    test_type: str = Form(...)
):
    img_bytes = await image.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    res = model(img, conf=0.5, classes=[0])

    if res[0].keypoints is None or len(res[0].keypoints.xy) == 0:
        return {"error": "No person detected"}

    boxes = res[0].boxes.xyxy.cpu().numpy()
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    main_idx = int(np.argmax(areas))

    xy = res[0].keypoints.xy[main_idx].cpu().numpy()
    conf = res[0].keypoints.conf[main_idx].cpu().numpy()

    if test_type == "posture_side":
        result = analyze_posture(xy, conf)
    elif test_type == "shoulder_right":
        result = analyze_shoulder(xy, conf, "RIGHT")
    elif test_type == "shoulder_left":
        result = analyze_shoulder(xy, conf, "LEFT")
    elif test_type == "squat":
        result = analyze_squat(xy, conf)
    else:
        return {"error": "Invalid test_type"}

    # Annotated image
    annotated = res[0].plot()
    ok, png = cv2.imencode(".png", annotated)
    annotated_url = None

    if supabase:
        path = f"{user_email}/{test_type}/{uuid.uuid4()}.png"
        supabase.storage.from_("screening").upload(
            path,
            png.tobytes(),
            {"content-type": "image/png"}
        )
        annotated_url = supabase.storage.from_("screening").get_public_url(path)

        supabase.table("screenings").insert({
            "user_email": user_email,
            "test_type": test_type,
            "score": result["score"],
            "confidence": result["confidence"],
            "metrics": result["metrics"],
            "annotated_image_url": annotated_url
        }).execute()

    return {
        "user_email": user_email,
        "test_type": test_type,
        "score": result["score"],
        "confidence": result["confidence"],
        "metrics": result["metrics"],
        "annotated_image_url": annotated_url
    }
