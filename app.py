from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import math

from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP
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


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    user_email: str = Form(...),
    test_type: str = Form(...),
):
    img_bytes = await image.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run pose model
    res = model(img, conf=0.5, classes=[0])

    if res[0].keypoints is None or len(res[0].keypoints.xy) == 0:
        return {
            "user_email": user_email,
            "test_type": test_type,
            "error": "No person detected. Retake photo with full body visible.",
        }

    # Select largest person box (robust)
    boxes = res[0].boxes.xyxy.cpu().numpy()
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    main_idx = int(np.argmax(areas))

    xy = res[0].keypoints.xy[main_idx].cpu().numpy()
    conf = res[0].keypoints.conf[main_idx].cpu().numpy()

    # For posture_side: compute neck + thoracic (proxy)
    # Use best side by confidence (ear+shoulder+hip)
    L_EAR, R_EAR = 3, 4
    L_SH, R_SH = 5, 6
    L_HIP, R_HIP = 11, 12

    left_score = conf[L_EAR] + conf[L_SH] + conf[L_HIP]
    right_score = conf[R_EAR] + conf[R_SH] + conf[R_HIP]

    if right_score >= left_score:
        ear = xy[R_EAR]; shoulder = xy[R_SH]; hip = xy[R_HIP]
        side = "RIGHT"
        quality = float(right_score/3)
    else:
        ear = xy[L_EAR]; shoulder = xy[L_SH]; hip = xy[L_HIP]
        side = "LEFT"
        quality = float(left_score/3)

    neck_angle = angle_to_vertical(shoulder, ear)
    thoracic_angle = angle_to_vertical(hip, shoulder)
    pelvic_proxy_angle = thoracic_angle

    # Simple continuous posture score (v1)
    def smooth_penalty(angle, optimal_max, severe_max, w=1.0):
        if angle <= optimal_max:
            return 0.0
        a = min(angle, severe_max)
        t = (a - optimal_max) / (severe_max - optimal_max)
        return w * 30.0 * (t ** 2)

    neck_pen = smooth_penalty(neck_angle, 10, 55, 1.2)
    thor_pen = smooth_penalty(thoracic_angle, 5, 45, 1.0)
    pelv_pen = smooth_penalty(pelvic_proxy_angle, 5, 40, 0.8)

    quality_weight = max(0.6, min(1.0, quality))
    total_pen = (neck_pen + thor_pen + pelv_pen) / quality_weight
    posture_score = max(0.0, 100.0 - total_pen)

    return {
        "user_email": user_email,
        "test_type": test_type,
        "side_used": side,
        "confidence": quality_weight,
        "metrics": {
            "neck_angle": round(neck_angle, 2),
            "thoracic_angle": round(thoracic_angle, 2),
            "pelvic_proxy_angle": round(pelvic_proxy_angle, 2),
        },
        "score": round(posture_score, 1),
        "annotated_image_url": None,  # next step
    }
