from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import math
import os
import json
import base64
from datetime import datetime, timezone

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
    Robust overhead shoulder mobility analysis.

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

        # raw_angle is the angle between trunk direction and arm direction.
        # In overhead flexion, this maps naturally toward 180° when the arm is overhead.
        shoulder_flexion = raw_angle

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


def compute_composite(posture, shoulder_r, shoulder_l, squat):
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

    res = model(img, conf=0.5, classes=[0])
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

    res = model(img, conf=0.5, classes=[0])

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

    flexilab_score = session.get("composite_score", None)

    if flexilab_score is None:
        posture_score = session.get("posture_score", None) or (posture.get("score") if posture else None)
        sh_r_score = session.get("shoulder_right_score", None) or (sh_r.get("score") if sh_r else None)
        sh_l_score = session.get("shoulder_left_score", None) or (sh_l.get("score") if sh_l else None)
        squat_score = session.get("squat_score", None) or (squat.get("score") if squat else None)

        shoulder = None
        if sh_r_score is not None and sh_l_score is not None:
            shoulder = min(float(sh_r_score), float(sh_l_score))
        elif sh_r_score is not None:
            shoulder = float(sh_r_score)
        elif sh_l_score is not None:
            shoulder = float(sh_l_score)

        parts = []
        if posture_score is not None:
            parts.append((float(posture_score), 0.4))
        if shoulder is not None:
            parts.append((float(shoulder), 0.3))
        if squat_score is not None:
            parts.append((float(squat_score), 0.3))

        if parts:
            wsum = sum(w for _, w in parts)
            flexilab_score = round(sum(v * w for v, w in parts) / wsum, 1)
        else:
            flexilab_score = None

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
