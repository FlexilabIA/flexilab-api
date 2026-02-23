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
@app.get("/report")
def report(session_id: str):
    """
    Build a 'product-ready' report JSON for a given session_id.
    Option B: works even if finalize_session wasn't called.
    It reads screenings, computes flexilab_score if missing, and returns a structured FR report.
    """
    if supabase is None:
        return {"error": "Supabase not configured"}

    # 1) Load session
    s_resp = supabase.table("sessions").select("*").eq("id", session_id).limit(1).execute()
    if not s_resp.data:
        return {"error": "Session not found"}
    session = s_resp.data[0]

    # 2) Load screenings for that session
    scr_resp = supabase.table("screenings").select("*").eq("session_id", session_id).execute()
    screenings = scr_resp.data or []

    tests_found = [x.get("test_type") for x in screenings if x.get("test_type")]

    # Helper to find a screening by test_type
    def get_test(tt):
        for r in screenings:
            if r.get("test_type") == tt:
                return r
        return None

    posture = get_test("posture_side")
    sh_r = get_test("shoulder_right")
    sh_l = get_test("shoulder_left")
    squat = get_test("squat")

    # 3) Compute flexilab_score if not already computed
    flexilab_score = session.get("composite_score", None)
    if flexilab_score is None:
        # Use per-test scores saved in sessions if present, otherwise fallback to screenings
        posture_score = session.get("posture_score", None) or (posture.get("score") if posture else None)
        sh_r_score = session.get("shoulder_right_score", None) or (sh_r.get("score") if sh_r else None)
        sh_l_score = session.get("shoulder_left_score", None) or (sh_l.get("score") if sh_l else None)
        squat_score = session.get("squat_score", None) or (squat.get("score") if squat else None)

        # Mimic compute_composite logic (worst shoulder)
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

    # 4) Risk category
    def risk_from_score(score):
        if score is None:
            return {
                "label": "Unknown",
                "color": "grey",
                "description_fr": "Session incomplète : termine tous les tests pour un score global."
            }
        score = float(score)
        if score >= 85:
            return {"label": "Low", "color": "green", "description_fr": "Bon équilibre global. Quelques ajustements possibles."}
        if score >= 70:
            return {"label": "Moderate", "color": "yellow", "description_fr": "Profil intermédiaire : plusieurs axes d’amélioration."}
        return {"label": "High", "color": "red", "description_fr": "Priorité d’amélioration : plusieurs indicateurs hors zone cible."}

    risk_category = risk_from_score(flexilab_score)

    # 5) Utility: extract item from a screening thresholds dict
    def thr_item(thresholds, key):
        if not thresholds:
            return None
        v = thresholds.get(key)
        return v if isinstance(v, dict) else None

    # 6) Build sections with FR labels + short insights
    sections = []

    # --- Posture section ---
    if posture:
        m = posture.get("metrics") or {}
        t = posture.get("thresholds") or {}
        neck_val = m.get("neck_angle")
        thor_val = m.get("thoracic_angle")
        pelv_val = m.get("pelvic_proxy_angle")

        neck_thr = thr_item(t, "neck_angle")
        thor_thr = thr_item(t, "thoracic_angle")
        pelv_thr = thr_item(t, "pelvic_proxy_angle")

        def insight_posture(label, rating):
            if rating == "green":
                return f"{label} satisfaisant."
            if rating == "yellow":
                return f"{label} à améliorer légèrement."
            if rating == "red":
                return f"{label} prioritaire à corriger."
            return f"{label} : données insuffisantes."

        sections.append({
            "id": "posture",
            "title_fr": "Posture (vue de profil)",
            "items": [
                {
                    "id": "neck_angle",
                    "label_fr": "Angle cervical",
                    "value": neck_val,
                    "unit": "°",
                    "rating": (neck_thr or {}).get("rating"),
                    "thresholds": neck_thr,
                    "short_insight_fr": insight_posture("Alignement cervical", (neck_thr or {}).get("rating")),
                },
                {
                    "id": "thoracic_angle",
                    "label_fr": "Angle thoracique",
                    "value": thor_val,
                    "unit": "°",
                    "rating": (thor_thr or {}).get("rating"),
                    "thresholds": thor_thr,
                    "short_insight_fr": insight_posture("Alignement thoracique", (thor_thr or {}).get("rating")),
                },
                {
                    "id": "pelvic_proxy_angle",
                    "label_fr": "Bassin (proxy)",
                    "value": pelv_val,
                    "unit": "°",
                    "rating": (pelv_thr or {}).get("rating"),
                    "thresholds": pelv_thr,
                    "short_insight_fr": insight_posture("Position du bassin", (pelv_thr or {}).get("rating")),
                },
            ]
        })

    # --- Shoulders section ---
    if sh_r or sh_l:
        items = []
        asym = None

        def insight_shoulder(rating):
            if rating == "green":
                return "Mobilité overhead très bonne."
            if rating == "yellow":
                return "Légère limitation par rapport à l'objectif."
            if rating == "red":
                return "Limitation marquée : priorité mobilité."
            return "Données insuffisantes."

        # right
        if sh_r:
            mr = sh_r.get("metrics") or {}
            tr = sh_r.get("thresholds") or {}
            thr = thr_item(tr, "shoulder_flexion")
            val = mr.get("shoulder_flexion_angle")
            items.append({
                "id": "shoulder_right_flexion",
                "label_fr": "Flexion épaule droite",
                "value": val,
                "unit": "°",
                "rating": (thr or {}).get("rating"),
                "thresholds": thr,
                "short_insight_fr": insight_shoulder((thr or {}).get("rating")),
            })

        # left
        if sh_l:
            ml = sh_l.get("metrics") or {}
            tl = sh_l.get("thresholds") or {}
            thr = thr_item(tl, "shoulder_flexion")
            val = ml.get("shoulder_flexion_angle")
            items.append({
                "id": "shoulder_left_flexion",
                "label_fr": "Flexion épaule gauche",
                "value": val,
                "unit": "°",
                "rating": (thr or {}).get("rating"),
                "thresholds": thr,
                "short_insight_fr": insight_shoulder((thr or {}).get("rating")),
            })

        # asymmetry (if both present)
        if sh_r and sh_l:
            vr = (sh_r.get("metrics") or {}).get("shoulder_flexion_angle")
            vl = (sh_l.get("metrics") or {}).get("shoulder_flexion_angle")
            if vr is not None and vl is not None:
                asym_deg = abs(float(vr) - float(vl))
                # simple asymmetry rating
                if asym_deg <= 5:
                    a_rating = "green"
                    a_txt = "Symétrie satisfaisante."
                elif asym_deg <= 12:
                    a_rating = "yellow"
                    a_txt = "Asymétrie légère entre droite et gauche."
                else:
                    a_rating = "red"
                    a_txt = "Asymétrie importante : priorité équilibre D/G."
                asym = {"value_deg": round(asym_deg, 2), "rating": a_rating, "short_insight_fr": a_txt}

        sections.append({
            "id": "shoulders",
            "title_fr": "Mobilité des épaules",
            "items": items,
            "asymmetry": asym
        })

    # --- Squat section ---
    if squat:
        ms = squat.get("metrics") or {}
        ts = squat.get("thresholds") or {}

        knee_val = ms.get("knee_angle")
        trunk_val = ms.get("trunk_lean")

        knee_thr = thr_item(ts, "knee_angle")
        trunk_thr = thr_item(ts, "trunk_lean")

        def insight_squat(label, rating):
            if rating == "green":
                return f"{label} satisfaisant."
            if rating == "yellow":
                return f"{label} à améliorer."
            if rating == "red":
                return f"{label} prioritaire à améliorer."
            return f"{label} : données insuffisantes."

        sections.append({
            "id": "squat",
            "title_fr": "Squat (contrôle et mobilité)",
            "items": [
                {
                    "id": "squat_knee_angle",
                    "label_fr": "Angle du genou",
                    "value": knee_val,
                    "unit": "°",
                    "rating": (knee_thr or {}).get("rating"),
                    "thresholds": knee_thr,
                    "short_insight_fr": insight_squat("Profondeur", (knee_thr or {}).get("rating")),
                },
                {
                    "id": "squat_trunk_lean",
                    "label_fr": "Inclinaison du tronc",
                    "value": trunk_val,
                    "unit": "°",
                    "rating": (trunk_thr or {}).get("rating"),
                    "thresholds": trunk_thr,
                    "short_insight_fr": insight_squat("Contrôle du tronc", (trunk_thr or {}).get("rating")),
                }
            ]
        })

    # 7) Priorities (max 3): collect red then yellow across key measures
    candidates = []

    def add_candidate(sev, title_fr, why_fr):
        candidates.append({"severity": sev, "title_fr": title_fr, "why_fr": why_fr})

    # posture
    if posture:
        t = posture.get("thresholds") or {}
        na = thr_item(t, "neck_angle")
        ta = thr_item(t, "thoracic_angle")
        if (na or {}).get("rating") in ["red", "yellow"]:
            add_candidate((na or {}).get("rating"), "Alignement cervical", "L’angle cervical est hors de la zone optimale.")
        if (ta or {}).get("rating") in ["red", "yellow"]:
            add_candidate((ta or {}).get("rating"), "Alignement thoracique", "L’angle thoracique est hors de la zone optimale.")

    # shoulders
    if sh_r:
        thr = thr_item((sh_r.get("thresholds") or {}), "shoulder_flexion")
        if (thr or {}).get("rating") in ["red", "yellow"]:
            add_candidate((thr or {}).get("rating"), "Mobilité épaule droite", "Flexion overhead sous l’objectif.")
    if sh_l:
        thr = thr_item((sh_l.get("thresholds") or {}), "shoulder_flexion")
        if (thr or {}).get("rating") in ["red", "yellow"]:
            add_candidate((thr or {}).get("rating"), "Mobilité épaule gauche", "Flexion overhead sous l’objectif.")

    # squat
    if squat:
        ts = squat.get("thresholds") or {}
        tr = thr_item(ts, "trunk_lean")
        kn = thr_item(ts, "knee_angle")
        if (tr or {}).get("rating") in ["red", "yellow"]:
            add_candidate((tr or {}).get("rating"), "Inclinaison du tronc en squat", "Inclinaison du tronc hors zone cible.")
        if (kn or {}).get("rating") in ["red", "yellow"]:
            add_candidate((kn or {}).get("rating"), "Profondeur du squat", "Angle du genou hors zone cible.")

    # Sort: red first then yellow, limit 3
    sev_order = {"red": 0, "yellow": 1, "green": 2, "unknown": 3, None: 4}
    candidates.sort(key=lambda x: sev_order.get(x["severity"], 9))

    top_priorities = []
    for i, c in enumerate(candidates[:3], start=1):
        top_priorities.append({
            "id": f"priority_{i}",
            "title_fr": c["title_fr"],
            "severity": c["severity"],
            "why_fr": c["why_fr"]
        })

    # 8) Created_at
    created_at = session.get("created_at")

    # 9) Final report object
    return {
        "session_id": session_id,
        "user_email": session.get("user_email"),
        "created_at": created_at,

        "flexilab_score": flexilab_score,
        "risk_category": risk_category,

        "sections": sections,
        "top_priorities": top_priorities,
        "next_step_fr": "Refais le screening dans 14 jours pour vérifier l'évolution.",
        "debug": {"tests_found": tests_found}
    }
