from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import math
import os

from ultralytics import YOLO
from supabase import create_client

# Avoid Ultralytics config warning on Render
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# Supabase client (backend only)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

supabase = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP - later restrict to your LearnWorlds domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup (important)
model = YOLO("yolov8n-pose.pt")


@app.get("/health")
def health():
    return {"ok": True}


def angle_to_vertical(p1, p2):
    dx = float(p2[0] - p1[0])
    dy = float(p1[1] - p2[1])  # invert y axis
    ang = abs(math.degrees(math.atan2(dx, dy)))
    if ang > 90:
        ang = 180 - ang
    return ang


def smooth_penalty(angle, optimal_max, severe_max, w=1.0):
    """Continuous penalty (0 -> 30) using a smooth quadratic curve."""
    if angle <= optimal_max:
        return 0.0
    a = min(float(angle), float(severe_max))
    t = (a - float(optimal_max)) / (float(severe_max) - float(optimal_max))
    return float(w) * 30.0 * (t ** 2)


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    user_email: str = Form(...),
    test_type: str = Form(...),
):
    # 1) Decode image
    img_bytes = await image.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "user_email": user_email,
            "test_type": test_type,
            "error": "Invalid image file. Please upload a valid JPG/PNG.",
        }

    # 2) Run pose model
    res = model(img, conf=0.5, classes=[0])

    if res[0].keypoints is None or len(res[0].keypoints.xy) == 0:
        return {
            "user_email": user_email,
            "test_type": test_type,
            "error": "No person detected. Retake photo with full body visible.",
        }

    # 3) Select largest detected person (robust against jackets/background)
    boxes = res[0].boxes.xyxy.cpu().numpy()
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    main_idx = int(np.argmax(areas))

    xy = res[0].keypoints.xy[main_idx].cpu().numpy()
    conf = res[0].keypoints.conf[main_idx].cpu().numpy()

    # 4) Posture-side metrics (for now we compute the same metrics regardless of test_type)
    # Choose best side by confidence: ear + shoulder + hip
    L_EAR, R_EAR = 3, 4
    L_SH, R_SH = 5, 6
    L_HIP, R_HIP = 11, 12

    left_score = float(conf[L_EAR] + conf[L_SH] + conf[L_HIP])
    right_score = float(conf[R_EAR] + conf[R_SH] + conf[R_HIP])

    if right_score >= left_score:
        ear = xy[R_EAR]
        shoulder = xy[R_SH]
        hip = xy[R_HIP]
        side = "RIGHT"
        quality = right_score / 3.0
    else:
        ear = xy[L_EAR]
        shoulder = xy[L_SH]
        hip = xy[L_HIP]
        side = "LEFT"
        quality = left_score / 3.0

    neck_angle = angle_to_vertical(shoulder, ear)
    thoracic_angle = angle_to_vertical(hip, shoulder)
    pelvic_proxy_angle = thoracic_angle

    # 5) Continuous posture score (scientific-ish screening)
    neck_pen = smooth_penalty(neck_angle, optimal_max=10, severe_max=55, w=1.2)
    thor_pen = smooth_penalty(thoracic_angle, optimal_max=5, severe_max=45, w=1.0)
    pelv_pen = smooth_penalty(pelvic_proxy_angle, optimal_max=5, severe_max=40, w=0.8)

    quality_weight = max(0.6, min(1.0, float(quality)))
    total_pen = (neck_pen + thor_pen + pelv_pen) / quality_weight
    posture_score = max(0.0, 100.0 - total_pen)

    metrics = {
        "neck_angle": round(neck_angle, 2),
        "thoracic_angle": round(thoracic_angle, 2),
        "pelvic_proxy_angle": round(pelvic_proxy_angle, 2),
        "side_used": side,
    }

    # 6) Save to Supabase (DB)
    saved_to_supabase = False
    supabase_error = None

    if supabase is not None:
        try:
            supabase.table("screenings").insert({
                "user_email": user_email,
                "test_type": test_type,
                "score": float(round(posture_score, 1)),
                "confidence": float(round(quality_weight, 3)),
                "metrics": metrics,
            }).execute()
            saved_to_supabase = True
        except Exception as e:
            # Don't fail the API if DB insert fails
            supabase_error = str(e)

    # 7) Return response
    resp = {
        "user_email": user_email,
        "test_type": test_type,
        "confidence": float(round(quality_weight, 3)),
        "metrics": metrics,
        "score": float(round(posture_score, 1)),
        "annotated_image_url": None,  # next step (Storage + overlay)
        "saved_to_supabase": saved_to_supabase,
    }

    if supabase_error:
        resp["supabase_error"] = supabase_error

    return resp
