from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
    Header,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from account_api import create_account_router
from stripe_api import create_stripe_router
from trainer_api import create_trainer_router
from operator_api import create_operator_router
from screening_access import (
    authenticated_user,
    ensure_email_matches,
    reserve_credit,
    consume_credit,
    release_credit,
    effective_entitlement,
)

import numpy as np
import cv2
import math
import os
import json
import base64
import hashlib
import logging
import time
import uuid
import threading
import copy
import httpx
from importlib.metadata import PackageNotFoundError, version as package_version
from datetime import datetime, timedelta, timezone
from aslr_engine import (
    ASLRQualityError,
    ASLR_ENGINE_VERSION,
    ASLR_RED_MAX_DEG,
    ASLR_YELLOW_MAX_DEG,
    analyze_aslr_v2,
    analyze_aslr_rotated_fullbody,
    make_aslr_thresholds,
)
from vision_qa import VISION_QA_VERSION, build_vision_qa_payload
# FlexiLab V2 backend architecture imports.
# Old engines remain in the repository for rollback, but /program now uses:
# score_engine_v2 -> Movement DNA / CKB -> Clinical Prescription Engine v2.1.
try:
    from engines.score_engine_v2 import attach_score_v2
except Exception:
    attach_score_v2 = None

try:
    from engines.clinical_report_engine_v1 import attach_expert_report
except Exception:
    attach_expert_report = None

try:
    from engines.flexilab_ckb_engine_v1 import generate_movement_dna, load_json as load_ckb_json
except Exception:
    generate_movement_dna = None
    load_ckb_json = None

try:
    from engines.clinical_prescription_engine_v21 import (
        generate_clinical_prescription_v21,
        load_exercise_library,
        load_json as load_prescription_json,
    )
except Exception:
    generate_clinical_prescription_v21 = None
    load_exercise_library = None
    load_prescription_json = None

os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

from ultralytics import YOLO
from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

supabase = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ANALYSIS_STORAGE_BUCKET = os.environ.get("FLEXILAB_ANALYSIS_BUCKET", "screening-private").strip() or "screening-private"
PROCESS_ROLE = os.environ.get("FLEXILAB_PROCESS_ROLE", "api").strip().lower() or "api"
# Kept for health/backward compatibility only. The web API is queue-only and can
# never execute analysis jobs, even if an old Render variable is misconfigured.
ANALYSIS_INLINE_ENABLED = False
ANALYSIS_MAX_EDGE = max(640, min(1920, int(os.environ.get("FLEXILAB_ANALYSIS_MAX_EDGE", "960"))))
POSE_INFERENCE_IMGSZ = max(320, min(1280, int(os.environ.get("FLEXILAB_POSE_IMGSZ", "640"))))
DIAGNOSTIC_RETENTION_HOURS = max(0, min(168, int(os.environ.get("FLEXILAB_DIAGNOSTIC_RETENTION_HOURS", "0"))))
ANALYSIS_IMAGE_TTL_MINUTES = max(15, min(120, int(os.environ.get("FLEXILAB_ANALYSIS_IMAGE_TTL_MINUTES", "60"))))
ANALYSIS_IMAGE_DELETE_RETRIES = max(1, min(5, int(os.environ.get("FLEXILAB_ANALYSIS_IMAGE_DELETE_RETRIES", "3"))))
VISION_QA_MODE = os.environ.get("FLEXILAB_VISION_QA_MODE", "shoulder").strip().lower()
VISION_QA_ONE_TIME_DELIVERY = os.environ.get("FLEXILAB_VISION_QA_ONE_TIME_DELIVERY", "true").strip().lower() not in {"0", "false", "no", "off"}

def _vision_qa_enabled_for_test(test_type):
    """Enable ephemeral YOLO overlays only for the configured diagnostic scope."""
    normalized_mode = VISION_QA_MODE.replace("-", "_")
    normalized_test = str(test_type or "").strip().lower()
    if normalized_mode in {"all", "on", "validation", "true", "1"}:
        return True
    if normalized_mode in {"shoulder", "shoulders", "shoulder_only"}:
        return normalized_test in {"shoulder_right", "shoulder_left"}
    if normalized_mode in {"aslr", "aslr_only"}:
        return normalized_test.startswith("aslr_")
    return False

VISION_QA_VALIDATION_ENABLED = VISION_QA_MODE not in {"", "off", "false", "0", "none"}
ASLR_KEYPOINT_MIN_CONF = max(0.05, min(0.80, float(os.environ.get("FLEXILAB_ASLR_KEYPOINT_MIN_CONF", "0.20"))))
ASLR_REQUIRED_MEAN_CONF = max(ASLR_KEYPOINT_MIN_CONF, min(0.90, float(os.environ.get("FLEXILAB_ASLR_REQUIRED_MEAN_CONF", "0.35"))))
ASLR_RAISED_KNEE_EXTENSION_MIN = max(135.0, min(175.0, float(os.environ.get("FLEXILAB_ASLR_RAISED_KNEE_EXTENSION_MIN", "155"))))
ASLR_RESTING_KNEE_EXTENSION_MIN = max(135.0, min(175.0, float(os.environ.get("FLEXILAB_ASLR_RESTING_KNEE_EXTENSION_MIN", "150"))))
ASLR_RESTING_LEG_MAX_ANGLE = max(8.0, min(30.0, float(os.environ.get("FLEXILAB_ASLR_RESTING_LEG_MAX_ANGLE", "20"))))

MOVEMENT_PATTERNS_PATH = os.path.join(DATA_DIR, "movement_patterns_v1.json")
PRESCRIPTION_RULES_PATH = os.path.join(DATA_DIR, "prescription_rules_v1.json")
# V68: exercise library can run in filmed-demo mode or full-library mode.
# Default is "demo" because the current client demo should use only filmed exercises.
# Set FLEXILAB_LIBRARY_MODE=full in Render when you want to return to the full library.
EXERCISE_LIBRARY_MODE = os.environ.get("FLEXILAB_LIBRARY_MODE", "demo").strip().lower()
EXERCISE_LIBRARY_PATH_FULL = os.path.join(DATA_DIR, "flexilab_exercise_library_full_v1.json")
EXERCISE_LIBRARY_PATH_DEMO = os.path.join(DATA_DIR, "flexilab_exercise_library_demo_v1.json")

if EXERCISE_LIBRARY_MODE in ["full", "production", "prod"]:
    EXERCISE_LIBRARY_PATH = EXERCISE_LIBRARY_PATH_FULL
elif EXERCISE_LIBRARY_MODE in ["demo", "filmed", "filmed_demo", "client_demo"]:
    EXERCISE_LIBRARY_PATH = EXERCISE_LIBRARY_PATH_DEMO
else:
    # Optional custom JSON path for testing. Falls back to demo if the custom path is invalid.
    custom_path = os.environ.get("FLEXILAB_EXERCISE_LIBRARY_PATH")
    EXERCISE_LIBRARY_PATH = custom_path if custom_path else EXERCISE_LIBRARY_PATH_DEMO

MOVEMENT_PATTERNS = None
PRESCRIPTION_RULES = None
EXERCISE_LIBRARY = None
RESOURCE_LOAD_ERRORS = {}


def load_clinical_resources():
    """
    Load FlexiLab V2 clinical resources once at startup.

    The server must stay alive even if one resource is temporarily missing,
    so errors are stored in RESOURCE_LOAD_ERRORS and returned inside /program.
    """
    global MOVEMENT_PATTERNS, PRESCRIPTION_RULES, EXERCISE_LIBRARY, RESOURCE_LOAD_ERRORS

    RESOURCE_LOAD_ERRORS = {}

    if load_ckb_json:
        try:
            MOVEMENT_PATTERNS = load_ckb_json(MOVEMENT_PATTERNS_PATH)
        except Exception as e:
            MOVEMENT_PATTERNS = None
            RESOURCE_LOAD_ERRORS["movement_patterns"] = str(e)
    else:
        RESOURCE_LOAD_ERRORS["movement_patterns"] = "flexilab_ckb_engine_v1 import failed"

    if load_prescription_json:
        try:
            PRESCRIPTION_RULES = load_prescription_json(PRESCRIPTION_RULES_PATH)
        except Exception as e:
            PRESCRIPTION_RULES = None
            RESOURCE_LOAD_ERRORS["prescription_rules"] = str(e)
    else:
        RESOURCE_LOAD_ERRORS["prescription_rules"] = "clinical_prescription_engine_v21 import failed"

    if load_exercise_library:
        try:
            EXERCISE_LIBRARY = load_exercise_library(EXERCISE_LIBRARY_PATH)
        except Exception as e:
            # V68 safety fallback: if selected library fails, try the original v1 file before failing.
            fallback_path = os.path.join(DATA_DIR, "flexilab_exercise_library_v1.json")
            try:
                EXERCISE_LIBRARY = load_exercise_library(fallback_path)
                RESOURCE_LOAD_ERRORS["exercise_library_selected_path_failed"] = str(e)
                RESOURCE_LOAD_ERRORS["exercise_library_fallback_path"] = fallback_path
            except Exception:
                EXERCISE_LIBRARY = None
                RESOURCE_LOAD_ERRORS["exercise_library"] = str(e)
    else:
        RESOURCE_LOAD_ERRORS["exercise_library"] = "load_exercise_library import failed"


load_clinical_resources()

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "FLEXILAB_ALLOWED_ORIGINS",
        "https://flexilab.fr,https://www.flexilab.fr,https://flexi-move-lab.lovable.app,http://localhost:3000,http://localhost:5173",
    ).split(",")
    if origin.strip()
]

POSE_MODEL_NAME = os.environ.get("FLEXILAB_POSE_MODEL", "yolov8n-pose.pt")
POSE_MODEL_LOAD_ERROR = None
POSE_MODEL_INFERENCE_LOCK = threading.RLock()
POSE_MODEL_RELOAD_COUNT = 0

# Dedicated higher-accuracy model used only for the two ASLR tests.
# YOLO11m-pose has materially higher COCO pose mAP than nano variants while
# remaining a reasonable fit for the current 2 GB Render instance.
ASLR_POSE_MODEL_NAME = os.environ.get(
    "FLEXILAB_ASLR_POSE_MODEL", "yolo11m-pose.pt"
).strip() or "yolo11m-pose.pt"
ASLR_POSE_MODEL_LOAD_ERROR = None
ASLR_POSE_MODEL_INFERENCE_LOCK = threading.RLock()

# In-process terminal-state failsafe. Supabase can occasionally reject the final
# status update with a transient httpx/httpcore ReadError. Without this cache the
# database row can remain ``processing`` and the browser polls forever even though
# the worker has already ended. This cache is not the source of truth for completed
# screenings; it only guarantees a terminal response for the current Render process.
ANALYSIS_RUNTIME_STATE = {}
ANALYSIS_RUNTIME_STATE_LOCK = threading.RLock()
ANALYSIS_RUNTIME_TIMEOUT_SECONDS = max(60, min(180, int(os.environ.get("FLEXILAB_ANALYSIS_RUNTIME_TIMEOUT_SECONDS", "120"))))

def _delete_analysis_image(image_path: str, *, job_id: str | None = None) -> bool:
    """Delete one queued image with bounded retries and visible observability."""
    path = str(image_path or "").strip()
    if not path or supabase is None:
        return not path
    last_error: Exception | None = None
    for attempt in range(1, ANALYSIS_IMAGE_DELETE_RETRIES + 1):
        try:
            supabase.storage.from_(ANALYSIS_STORAGE_BUCKET).remove([path])
            logger.info(
                "analysis_image_deleted job_id=%s path=%s attempt=%s",
                job_id, path, attempt,
            )
            return True
        except Exception as exc:
            last_error = exc
            logger.warning(
                "analysis_image_delete_retry job_id=%s path=%s attempt=%s/%s error=%r",
                job_id, path, attempt, ANALYSIS_IMAGE_DELETE_RETRIES, exc,
            )
            if attempt < ANALYSIS_IMAGE_DELETE_RETRIES:
                time.sleep(0.25 * attempt)
    logger.error(
        "analysis_image_delete_failed job_id=%s path=%s retries=%s error=%r",
        job_id, path, ANALYSIS_IMAGE_DELETE_RETRIES, last_error,
    )
    return False


def _clear_analysis_image_reference(job_id: str, image_path: str) -> bool:
    removed = _delete_analysis_image(image_path, job_id=job_id)
    changes = {"image_base64": None}
    if removed:
        changes.update({"image_path": None, "image_expires_at": None})
    else:
        changes["image_expires_at"] = utc_now_iso()
    try:
        supabase.table("analysis_jobs").update(changes).eq("id", job_id).execute()
    except Exception:
        logger.exception("analysis_image_reference_update_failed job_id=%s", job_id)
    return removed


def _set_analysis_runtime_state(job_id, status, *, result=None, error_message=None):
    with ANALYSIS_RUNTIME_STATE_LOCK:
        ANALYSIS_RUNTIME_STATE[str(job_id)] = {
            "status": str(status),
            "result": result,
            "error_message": error_message,
            "updated_at_monotonic": time.monotonic(),
        }

def _get_analysis_runtime_state(job_id):
    with ANALYSIS_RUNTIME_STATE_LOCK:
        value = ANALYSIS_RUNTIME_STATE.get(str(job_id))
        return dict(value) if isinstance(value, dict) else None
ASLR_POSE_MODEL_RELOAD_COUNT = 0


def _load_pose_model():
    global POSE_MODEL_LOAD_ERROR
    try:
        loaded_model = YOLO(POSE_MODEL_NAME)
        POSE_MODEL_LOAD_ERROR = None
        return loaded_model
    except Exception as exc:
        POSE_MODEL_LOAD_ERROR = str(exc)
        return None


def _load_aslr_pose_model():
    global ASLR_POSE_MODEL_LOAD_ERROR
    try:
        loaded_model = YOLO(ASLR_POSE_MODEL_NAME)
        ASLR_POSE_MODEL_LOAD_ERROR = None
        return loaded_model
    except Exception as exc:
        ASLR_POSE_MODEL_LOAD_ERROR = str(exc)
        return None


def _installed_version(distribution_name):
    try:
        return package_version(distribution_name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


RUNTIME_PACKAGE_VERSIONS = {
    "fastapi": _installed_version("fastapi"),
    "uvicorn": _installed_version("uvicorn"),
    "python-multipart": _installed_version("python-multipart"),
    "opencv-python-headless": _installed_version("opencv-python-headless"),
    "numpy": _installed_version("numpy"),
    "supabase": _installed_version("supabase"),
    "ultralytics": _installed_version("ultralytics"),
    "torch": _installed_version("torch"),
    "stripe": _installed_version("stripe"),
}


model = _load_pose_model()
aslr_model = _load_aslr_pose_model()

app = FastAPI(
    title="FlexiLab Movement Intelligence API",
    version="101.35.17",
)
app.include_router(create_account_router(supabase))
app.include_router(create_stripe_router(supabase))
app.include_router(create_trainer_router(supabase))
app.include_router(
    create_operator_router(
        supabase,
        health_provider=lambda: {
            "clinical_resources": {
                "exercise_library_mode": EXERCISE_LIBRARY_MODE,
                "exercise_library_count": len(EXERCISE_LIBRARY or []),
                "resource_load_errors": RESOURCE_LOAD_ERRORS,
            },
            "pose_model": {
                "configured_model": POSE_MODEL_NAME,
                "loaded": model is not None,
                "load_error": POSE_MODEL_LOAD_ERROR,
                "reload_count": POSE_MODEL_RELOAD_COUNT,
                "inference_imgsz": POSE_INFERENCE_IMGSZ,
                "analysis_max_edge": ANALYSIS_MAX_EDGE,
                "runtime_versions": RUNTIME_PACKAGE_VERSIONS,
            },
            "aslr_pose_model": {
                "configured_model": ASLR_POSE_MODEL_NAME,
                "loaded": aslr_model is not None,
                "load_error": ASLR_POSE_MODEL_LOAD_ERROR,
                "reload_count": ASLR_POSE_MODEL_RELOAD_COUNT,
                "inference_imgsz": 960,
            },
        },
    )
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    # The frontend and browser may add request-id, tracing, or Supabase headers.
    # Restricting this list caused legitimate OPTIONS requests to return 400.
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "Server-Timing"],
    max_age=600,
)

logger = logging.getLogger("flexilab.performance")

def _attach_cors_headers_for_error(request, response):
    """Preserve CORS visibility when an exception is converted to JSON here."""
    origin = request.headers.get("origin")
    if origin and origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Vary"] = "Origin"
    return response


@app.middleware("http")
async def request_timing_middleware(request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except (
        httpx.ReadError,
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.PoolTimeout,
    ):
        logger.exception(
            "upstream_temporarily_unavailable method=%s path=%s request_id=%s",
            request.method,
            request.url.path,
            request_id,
        )
        response = JSONResponse(
            status_code=503,
            content={
                "detail": "A required service is temporarily unavailable. Please retry.",
                "code": "UPSTREAM_TEMPORARILY_UNAVAILABLE",
                "request_id": request_id,
            },
        )
        _attach_cors_headers_for_error(request, response)
    except Exception:
        logger.exception(
            "request_failed method=%s path=%s request_id=%s",
            request.method,
            request.url.path,
            request_id,
        )
        response = JSONResponse(
            status_code=500,
            content={
                "detail": "An unexpected server error occurred.",
                "code": "INTERNAL_SERVER_ERROR",
                "request_id": request_id,
            },
        )
        _attach_cors_headers_for_error(request, response)

    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    response.headers["Server-Timing"] = f"total;dur={duration_ms}"
    logger.info(
        "request method=%s path=%s status=%s duration_ms=%s request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request_id,
    )
    return response



@app.get("/health")
def health():
    return {
        "ok": True,
        "patch_version": "V101.39.1-shoulder-protocol-directed-angle",
        "base_patch": "V101.35.31-aslr-left-image-mirror-before-yolo",
        "release_policy": "minimal_shoulder_protocol_directed_angle_change_with_existing_stability_preserved",
        "production_formula_changes_allowed": True,
        "process_role": PROCESS_ROLE,
        "analysis_execution_policy": "worker_only",
        "inline_analysis_enabled": False,
        "exercise_library_mode": EXERCISE_LIBRARY_MODE,
        "exercise_library_path": EXERCISE_LIBRARY_PATH,
        "exercise_library_count": len(EXERCISE_LIBRARY or []),
        "resource_load_errors": RESOURCE_LOAD_ERRORS,
        "pose_model_loaded": model is not None,
        "pose_model_error": POSE_MODEL_LOAD_ERROR,
        "pose_model_reload_count": POSE_MODEL_RELOAD_COUNT,
        "pose_inference_imgsz": POSE_INFERENCE_IMGSZ,
        "aslr_pose_model_loaded": aslr_model is not None,
        "aslr_pose_model_name": ASLR_POSE_MODEL_NAME,
        "aslr_pose_model_error": ASLR_POSE_MODEL_LOAD_ERROR,
        "aslr_pose_model_reload_count": ASLR_POSE_MODEL_RELOAD_COUNT,
        "analysis_max_edge": ANALYSIS_MAX_EDGE,
        "diagnostic_retention_hours": DIAGNOSTIC_RETENTION_HOURS,
        "aslr_engine": {
            "version": ASLR_ENGINE_VERSION,
            "keypoint_min_conf": ASLR_KEYPOINT_MIN_CONF,
            "required_mean_conf": ASLR_REQUIRED_MEAN_CONF,
            "raised_knee_extension_min": ASLR_RAISED_KNEE_EXTENSION_MIN,
            "resting_knee_extension_min": ASLR_RESTING_KNEE_EXTENSION_MIN,
            "resting_leg_max_angle": ASLR_RESTING_LEG_MAX_ANGLE,
            "visual_thresholds_preserved": True,
            "visual_band_layout": "equal_thirds",
            "classification_bands_deg": {
                "red": "<60",
                "yellow": "60-75_inclusive",
                "green": ">75",
            },
            "source_orientation_requirement": "head_left_capture_protocol_internal_90_clockwise_inference",
            "chain_strategy": "right_clockwise_left_counterclockwise_then_conditional_focused_crop",
            "pose_passes": ["right_90_clockwise_or_left_90_counterclockwise_full_image", "conditional_same_direction_focused_crop_recovery"],
            "pose_model_inference_count": "1 normally; 2 only when coherent chain recovery is required",
            "detection_attempt_count": "1 normally; 2 conditionally",
            "tracked_image_processing": False,
            "aslr_inference_imgsz": 960,
            "measurement_anchor": "single_shared_pelvic_anchor_image_horizontal_to_true_raised_ankle",
            "dedicated_pose_model": ASLR_POSE_MODEL_NAME,
            "general_model_fallback": False,
        },
        "vision_qa": {
            "version": VISION_QA_VERSION,
            "delivery": "ephemeral_job_result_only",
            "mode": VISION_QA_MODE,
            "validation_enabled": VISION_QA_VALIDATION_ENABLED,
            "enabled_scope": VISION_QA_MODE,
            "shoulder_enabled": _vision_qa_enabled_for_test("shoulder_right"),
            "aslr_enabled": _vision_qa_enabled_for_test("aslr_right"),
            "one_time_delivery": VISION_QA_ONE_TIME_DELIVERY,
            "persisted_to_screenings": False,
        },
        "runtime_versions": RUNTIME_PACKAGE_VERSIONS,
    }


@app.get("/library_status")
def library_status():
    return {
        "patch_version": "V101.35.31-aslr-left-image-mirror-before-yolo",
        "exercise_library_mode": EXERCISE_LIBRARY_MODE,
        "exercise_library_path": EXERCISE_LIBRARY_PATH,
        "exercise_library_count": len(EXERCISE_LIBRARY or []),
        "resource_load_errors": RESOURCE_LOAD_ERRORS,
        "demo_ready_count": sum(1 for e in (EXERCISE_LIBRARY or []) if e.get("demo_ready") is True),
        "video_ready_count": sum(1 for e in (EXERCISE_LIBRARY or []) if e.get("video_ready") is True or e.get("video_url") or e.get("vimeo_url")),
    }


def safe_json_loads(raw):
    try:
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _sanitize_capture_metadata(value, depth=0):
    if depth > 4:
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:240]
    if isinstance(value, list):
        return [
            _sanitize_capture_metadata(item, depth + 1)
            for item in value[:40]
        ]
    if isinstance(value, dict):
        sanitized = {}
        for key, item in list(value.items())[:80]:
            sanitized[str(key)[:80]] = _sanitize_capture_metadata(item, depth + 1)
        return sanitized
    return str(value)[:240]


def parse_capture_metadata(capture_metadata_json=None):
    if not capture_metadata_json:
        return {}
    if len(capture_metadata_json) > 20000:
        raise ValueError("Capture metadata is too large.")
    parsed = safe_json_loads(capture_metadata_json)
    if not isinstance(parsed, dict):
        return {}
    sanitized = _sanitize_capture_metadata(parsed)
    return sanitized if isinstance(sanitized, dict) else {}


def _split_job_intake_and_capture_metadata(value):
    intake = dict(value or {}) if isinstance(value, dict) else {}
    capture_metadata = intake.pop("_flexilab_capture_metadata", {})
    if not isinstance(capture_metadata, dict):
        capture_metadata = {}
    return intake, capture_metadata


def parse_intake_payload(intake_json=None, questionnaire_json=None):
    """
    V64 compatibility helper.
    Accepts both the older frontend field name `intake_json` and the V63+
    questionnaire field name `questionnaire_json`.

    If both are present and both are dictionaries, merge them, with
    questionnaire_json taking priority.
    """
    intake_data = safe_json_loads(intake_json)
    questionnaire_data = safe_json_loads(questionnaire_json)

    if isinstance(intake_data, dict) and isinstance(questionnaire_data, dict):
        merged = dict(intake_data)
        merged.update(questionnaire_data)
        return merged

    if isinstance(questionnaire_data, dict):
        return questionnaire_data

    return intake_data


def try_save_session_intake(session_id, intake_data):
    """
    Best-effort save of questionnaire/intake context on the sessions table.
    This will silently skip if the Supabase column does not exist yet.
    The authoritative fallback remains the screenings.intake_json field.
    """
    try:
        if supabase is not None and session_id and intake_data:
            supabase.table("sessions").update({"intake_json": intake_data}).eq("id", session_id).execute()
            return True
    except Exception:
        return False
    return False

def require_owned_session(user, session_id: str):
    """Return a session only when it belongs to the authenticated account."""
    if supabase is None:
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured on server.",
        )

    response = (
        supabase.table("sessions")
        .select("*")
        .eq("id", session_id)
        .limit(1)
        .execute()
    )
    if not response.data:
        raise HTTPException(status_code=404, detail="Session not found.")

    row = response.data[0]
    row_user_id = str(row.get("user_id") or "")
    row_trainer_id = str(row.get("trainer_id") or "")
    row_performed_by = str(row.get("performed_by_user_id") or "")
    row_email = str(row.get("user_email") or "").strip().lower()

    if row_user_id:
        if user["id"] not in {row_user_id, row_trainer_id, row_performed_by}:
            raise HTTPException(
                status_code=403,
                detail="This screening session belongs to another account.",
            )
        return row

    # Temporary compatibility for legacy rows created before user_id was stored.
    if row_email and row_email == user["email"]:
        try:
            supabase.table("sessions").update(
                {"user_id": user["id"]}
            ).eq("id", session_id).execute()
            row["user_id"] = user["id"]
        except Exception:
            pass
        return row

    raise HTTPException(
        status_code=403,
        detail="This screening session belongs to another account.",
    )


def require_owned_program(user, program_id: str):
    """Resolve a program using the authenticated Supabase user UUID."""
    program_row = resolve_corrective_program(
        program_id,
        user_id=user["id"],
        user_email=user["email"],
    )
    if not program_row:
        raise HTTPException(status_code=404, detail="Program not found.")

    row_user_id = str(program_row.get("user_id") or "")
    row_email = str(program_row.get("user_email") or "").strip().lower()
    if row_user_id and row_user_id != user["id"]:
        raise HTTPException(status_code=403, detail="This corrective program belongs to another account.")
    if not row_user_id and row_email != user["email"]:
        raise HTTPException(status_code=403, detail="This corrective program belongs to another account.")
    return program_row


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
            "keypoint_confidence": {
                "ear": round(float(conf[R_EAR] if side == "RIGHT" else conf[L_EAR]), 3),
                "shoulder": round(float(conf[R_SH] if side == "RIGHT" else conf[L_SH]), 3),
                "hip": round(float(conf[R_HIP] if side == "RIGHT" else conf[L_HIP]), 3),
            },
        },
        "thresholds": {
            "neck_angle": neck_thr,
            "thoracic_angle": thor_thr,
            "pelvic_proxy_angle": pelvis_thr
        }
    }


def _shoulder_chain_candidates(xy, conf):
    """Rank both COCO arm chains by confidence and anatomical coherence.

    Side-view photographs frequently swap the visible anatomical side. The
    requested test side remains the reporting label, while the actual landmarks
    are selected from the arm chain that best follows the raised arm.
    """
    candidates = []
    for label, sh_i, el_i, wr_i, hip_i in (
        ("COCO_LEFT_H5_E7_W9", 5, 7, 9, 11),
        ("COCO_RIGHT_H6_E8_W10", 6, 8, 10, 12),
    ):
        sh = np.asarray(xy[sh_i], dtype=float)
        el = np.asarray(xy[el_i], dtype=float)
        wr = np.asarray(xy[wr_i], dtype=float)
        hip = np.asarray(xy[hip_i], dtype=float)
        values = [float(conf[i]) for i in (sh_i, el_i, wr_i, hip_i)]
        mean_conf = float(np.mean(values))
        min_conf = float(min(values))

        upper = sh - el
        lower = wr - el
        elbow_denom = float(np.linalg.norm(upper) * np.linalg.norm(lower))
        elbow_extension = 0.0
        if elbow_denom > 1e-6:
            cosine = max(-1.0, min(1.0, float(np.dot(upper, lower) / elbow_denom)))
            elbow_extension = float(math.degrees(math.acos(cosine)))

        upper_len = float(np.linalg.norm(el - sh))
        fore_len = float(np.linalg.norm(wr - el))
        ratio = upper_len / max(fore_len, 1e-6)
        ratio_quality = max(0.0, 1.0 - abs(math.log(max(ratio, 1e-6))) / math.log(3.5))
        raised_height = max(0.0, float(sh[1] - wr[1]))
        trunk_length = max(30.0, float(np.linalg.norm(sh - hip)))
        overhead_quality = max(0.0, min(1.0, raised_height / trunk_length))
        straightness = max(0.0, min(1.0, (elbow_extension - 115.0) / 65.0))
        score = mean_conf * 0.42 + min_conf * 0.18 + overhead_quality * 0.20 + straightness * 0.12 + ratio_quality * 0.08
        candidates.append({
            "label": label,
            "indices": {"shoulder": sh_i, "elbow": el_i, "wrist": wr_i, "hip": hip_i},
            "score": round(score, 6),
            "mean_confidence": round(mean_conf, 6),
            "minimum_confidence": round(min_conf, 6),
            "elbow_extension_angle": round(elbow_extension, 3),
            "overhead_quality": round(overhead_quality, 6),
            "segment_ratio": round(ratio, 4),
            "valid_detection": min_conf >= 0.20 and mean_conf >= 0.35 and overhead_quality >= 0.20,
        })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def _shoulder_chain_quality(xy, conf):
    candidates = _shoulder_chain_candidates(xy, conf)
    best = candidates[0]
    margin = best["score"] - candidates[1]["score"] if len(candidates) > 1 else best["score"]
    return {
        **best,
        "selection_margin": round(float(margin), 6),
        "uncertain": (not bool(best["valid_detection"])) or float(margin) < 0.035 or float(best["minimum_confidence"]) < 0.32,
    }, candidates


def analyze_shoulder(xy, conf, side="RIGHT"):
    """Estimate overhead shoulder mobility using the most coherent visible arm chain."""
    candidates = _shoulder_chain_candidates(xy, conf)
    selected = candidates[0]
    indices = selected["indices"]
    sh_i, el_i, wr_i, hip_i = indices["shoulder"], indices["elbow"], indices["wrist"], indices["hip"]

    sh = np.asarray(xy[sh_i], dtype=float)
    el = np.asarray(xy[el_i], dtype=float)
    wr = np.asarray(xy[wr_i], dtype=float)
    hip = np.asarray(xy[hip_i], dtype=float)
    sh_c, el_c, wr_c, hip_c = [float(conf[i]) for i in (sh_i, el_i, wr_i, hip_i)]

    arm_point = wr if wr_c >= 0.25 else el
    arm_point_used = "WRIST" if wr_c >= 0.25 else "ELBOW_FALLBACK"
    arm_c = wr_c if wr_c >= 0.25 else el_c

    # Preserve the validated shoulder geometry and use the screening protocol to
    # disambiguate only the small over-vertical range. The right and left tests
    # are captured as mirrored views, so their expected cross-product signs are
    # opposite. Facial landmarks are deliberately not required: the raised arm
    # may hide the nose or ear in users with limited mobility or larger bodies.
    v_trunk_up = sh - hip
    v_trunk_down = hip - sh
    v_arm = arm_point - sh
    denom = float(np.linalg.norm(v_trunk_up) * np.linalg.norm(v_arm))
    shoulder_flexion_base = 0.0
    shoulder_flexion = 0.0
    protocol_branch = "DIRECT_0_180"
    signed_cross = 0.0
    over_vertical_candidate = False

    if denom > 1e-6:
        cosang = max(-1.0, min(1.0, float(np.dot(v_trunk_up, v_arm) / denom)))
        shoulder_flexion_base = 180.0 - float(math.degrees(math.acos(cosang)))

        # 2-D cross product between the downward trunk reference and the arm.
        # In image coordinates, the expected sign is mirrored between tests.
        signed_cross = float(v_trunk_down[0] * v_arm[1] - v_trunk_down[1] * v_arm[0])
        expected_over_vertical_sign = -1.0 if str(side).upper() == "RIGHT" else 1.0
        reflex_candidate = 360.0 - shoulder_flexion_base
        over_vertical_candidate = (
            signed_cross * expected_over_vertical_sign > 0.0
            and 180.0 < reflex_candidate <= 210.0
        )
        if over_vertical_candidate:
            shoulder_flexion = reflex_candidate
            protocol_branch = f"{str(side).upper()}_PROTOCOL_OVER_VERTICAL"
        else:
            shoulder_flexion = shoulder_flexion_base

    shoulder_flexion_base = max(0.0, min(180.0, shoulder_flexion_base))
    shoulder_flexion = max(0.0, min(210.0, shoulder_flexion))

    elbow_extension = float(selected["elbow_extension_angle"])
    confidence = max(0.0, min(1.0, float((sh_c + arm_c + hip_c) / 3.0)))
    deficit = max(0.0, 175.0 - shoulder_flexion)
    score = max(0.0, 100.0 - deficit * 2.0)
    margin = float(selected["score"] - candidates[1]["score"]) if len(candidates) > 1 else float(selected["score"])

    shoulder_thr = make_thresholds(
        "deg", 0, 210,
        [
            {"label": "Red", "min": 0, "max": 160, "color": "red"},
            {"label": "Yellow", "min": 160, "max": 175, "color": "yellow"},
            {"label": "Green", "min": 175, "max": 210, "color": "green"},
        ],
        shoulder_flexion,
    )

    return {
        "score": round(score, 1),
        "confidence": round(confidence, 3),
        "metrics": {
            "shoulder_flexion_angle": round(shoulder_flexion, 2),
            "shoulder_flexion_base_angle": round(shoulder_flexion_base, 2),
            "angle_protocol_branch": protocol_branch,
            "protocol_over_vertical_applied": bool(over_vertical_candidate),
            "protocol_cross_product": round(signed_cross, 4),
            "angle_method": "existing_trunk_to_arm_geometry_plus_test_specific_over_vertical_disambiguation",
            "side": side,
            "requested_side": side,
            "detected_coco_chain": selected["label"],
            "chain_selection_method": "highest_confidence_anatomically_coherent_overhead_arm_chain",
            "chain_selection_margin": round(margin, 4),
            "selected_source_indices": indices,
            "arm_point_used": arm_point_used,
            "elbow_extension_angle": round(elbow_extension, 2),
            "keypoint_confidence": {
                "shoulder": round(sh_c, 3), "elbow": round(el_c, 3),
                "wrist": round(wr_c, 3), "hip": round(hip_c, 3),
            },
            "shoulder_chain_candidates": candidates,
        },
        "thresholds": {"shoulder_flexion": shoulder_thr},
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

    # `knee_angle` is the included hip-knee-ankle angle used by the UI.
    # Conventional knee flexion is approximately 180 - knee_angle, so a
    # smaller displayed value means a deeper squat. Launch screening bands:
    #   <=55 deg included angle  ~= >=125 deg flexion (deep/full capability)
    #   55-75 deg included angle ~= 105-125 deg flexion (moderate/parallel)
    #   >75 deg included angle   ~= <105 deg flexion (shallow/limited depth)
    depth_pen = 0.0
    if knee_angle > 90:
        depth_pen = 40
    elif knee_angle > 75:
        depth_pen = 30
    elif knee_angle > 65:
        depth_pen = 18
    elif knee_angle > 55:
        depth_pen = 8

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
        "deg", 0, 120,
        [
            {"label": "Green", "min": 0, "max": 55, "color": "green"},
            {"label": "Yellow", "min": 55, "max": 75, "color": "yellow"},
            {"label": "Red", "min": 75, "max": 120, "color": "red"},
        ],
        knee_angle
    )

    return {
        "score": round(score, 1),
        "confidence": round(conf_out, 3),
        "metrics": {
            "knee_angle": round(float(knee_angle), 2),
            "estimated_knee_flexion_angle": round(float(180.0 - knee_angle), 2),
            "knee_angle_convention": "included_hip_knee_ankle_angle_smaller_is_deeper",
            "trunk_lean": round(float(trunk_angle), 2),
            "keypoint_confidence": {
                "shoulders_mean": round(float(np.mean([conf[L_SH], conf[R_SH]])), 3),
                "hips_mean": round(float(np.mean([conf[L_HIP], conf[R_HIP]])), 3),
                "knees_mean": round(float(np.mean([conf[L_KNEE], conf[R_KNEE]])), 3),
                "ankles_mean": round(float(np.mean([conf[L_ANK], conf[R_ANK]])), 3),
            },
            "heel_elevation_assessment": "not_available_from_coco_ankle_keypoints",
        },
        "thresholds": {
            "knee_angle": knee_thr,
            "trunk_lean": trunk_thr
        }
    }




def _mirror_pose_xy_horizontally(xy, frame_width=None, img=None):
    """Return a horizontally mirrored copy of pose xy coordinates.

    This is used only for ASLR LEFT so that the left workflow is processed
    through the exact same geometric path as ASLR RIGHT.
    """
    mirrored = np.array(xy, dtype=float, copy=True)
    width = None
    try:
        if frame_width is not None:
            width = float(frame_width)
        elif img is not None and hasattr(img, "shape") and len(img.shape) >= 2:
            width = float(img.shape[1])
        elif mirrored.size:
            finite_x = mirrored[:, 0][np.isfinite(mirrored[:, 0])]
            if finite_x.size:
                width = float(np.max(finite_x)) + 1.0
    except Exception:
        width = None

    if width is None or width <= 1.0:
        return mirrored

    mirrored[:, 0] = (width - 1.0) - mirrored[:, 0]
    return mirrored



def _relabel_aslr_result_for_left(result):
    """Convert a mirrored-right ASLR result back to LEFT reporting labels."""
    if not isinstance(result, dict):
        return result

    relabelled = copy.deepcopy(result)
    metrics = relabelled.setdefault("metrics", {})
    metrics["requested_side"] = "LEFT"
    metrics["side"] = "LEFT"
    metrics["side_identity_method"] = (
        "left_capture_horizontally_mirrored_then_processed_through_right_pipeline"
    )
    metrics["source_orientation_requirement"] = (
        "left_capture_horizontally_mirrored_then_processed_with_right_aslr_pipeline"
    )
    metrics["display_rotation_applied"] = "horizontal_image_mirror_before_yolo_then_right_clockwise_pipeline"
    metrics["mirror_processing_applied"] = True
    metrics["mirror_processing_rule"] = (
        "if_test_left_mirror_image_before_yolo_then_run_exact_right_pipeline_then_relabel_left"
    )
    metrics["expected_resting_side_after_reporting"] = "RIGHT"

    body_baseline = metrics.get("body_baseline")
    if isinstance(body_baseline, dict):
        body_baseline["side"] = "LEFT"

    resting_limb_points = metrics.get("resting_limb_points")
    if isinstance(resting_limb_points, dict):
        resting_limb_points["expected_resting_side"] = "RIGHT"

    flags = metrics.get("diagnostic_flags")
    if isinstance(flags, list):
        if "left_capture_mirrored_into_right_pipeline" not in flags:
            flags.append("left_capture_mirrored_into_right_pipeline")

    return relabelled



def analyze_aslr(xy, conf, side="RIGHT", img=None, body_xy=None, body_conf=None):
    """ASLR analysis using one canonical RIGHT pipeline.

    The image-level normalization happens before YOLO in
    ``run_yolo_analysis_from_bytes``:
      * RIGHT: original normalized image.
      * LEFT: horizontally mirrored normalized image.

    Both then enter the exact same clockwise-rotation, YOLO, chain-selection,
    quality-gate and angle pipeline. This function only relabels the canonical
    RIGHT result back to LEFT for reporting.
    """
    requested_side = str(side or "RIGHT").upper()
    canonical_result = analyze_aslr_rotated_fullbody(
        xy,
        conf,
        side="RIGHT",
        keypoint_min_conf=ASLR_KEYPOINT_MIN_CONF,
        required_mean_conf=ASLR_REQUIRED_MEAN_CONF,
        raised_knee_extension_min=ASLR_RAISED_KNEE_EXTENSION_MIN,
        resting_knee_extension_min=ASLR_RESTING_KNEE_EXTENSION_MIN,
        resting_leg_max_angle=ASLR_RESTING_LEG_MAX_ANGLE,
        img=img,
        body_xy=body_xy,
        body_conf=body_conf,
    )
    if requested_side == "LEFT":
        return _relabel_aslr_result_for_left(canonical_result)
    return canonical_result



def _attach_screening_soft_warnings(result, test_type):
    """Attach non-blocking technique/quality warnings to an estimated screening result."""
    metrics = result.setdefault("metrics", {})
    warnings = []

    def add(code, message_fr, message_en, possible_effect_fr, possible_effect_en):
        warnings.append({
            "code": code,
            "severity": "warning",
            "blocking": False,
            "message_fr": message_fr,
            "message_en": message_en,
            "possible_effect_fr": possible_effect_fr,
            "possible_effect_en": possible_effect_en,
        })

    if test_type == "posture_side":
        kp = metrics.get("keypoint_confidence") or {}
        if min(float(kp.get("ear", 1)), float(kp.get("shoulder", 1)), float(kp.get("hip", 1))) < 0.45:
            add(
                "posture_landmark_moderate_confidence",
                "Un repère de profil est partiellement masqué, mais les angles restent estimables.",
                "A side-view landmark is partly obscured, but the angles remain estimable.",
                "La précision peut être légèrement réduite.",
                "Precision may be slightly reduced.",
            )
    elif test_type in {"shoulder_right", "shoulder_left"}:
        margin = float(metrics.get("chain_selection_margin", 1.0) or 0.0)
        if margin < 0.05:
            add(
                "shoulder_chain_selection_uncertain",
                "Le repère d’épaule est estimé avec une incertitude modérée.",
                "The shoulder landmark has moderate estimation uncertainty.",
                "L’amplitude peut varier légèrement selon le repère sélectionné.",
                "The range may vary slightly depending on the selected landmark.",
            )
        elbow = metrics.get("elbow_extension_angle")
        if elbow is not None and float(elbow) < 160.0:
            add(
                "shoulder_elbow_flexion",
                "Le coude est légèrement fléchi.",
                "The elbow is slightly bent.",
                "L’amplitude d’épaule peut être légèrement surestimée ou sous-estimée.",
                "Shoulder range may be slightly over- or underestimated.",
            )
    elif test_type == "squat":
        if float(metrics.get("trunk_lean", 0) or 0) > 25.0:
            add(
                "squat_trunk_inclination",
                "Le tronc est fortement incliné pendant le squat.",
                "The trunk is substantially inclined during the squat.",
                "Cela peut refléter votre stratégie de mouvement et influencer la profondeur estimée.",
                "This may reflect your movement strategy and influence estimated depth.",
            )
        kp = metrics.get("keypoint_confidence") or {}
        if float(kp.get("ankles_mean", 1)) < 0.40:
            add(
                "squat_ankle_landmark_moderate_confidence",
                "Les repères de cheville sont moins nets.",
                "The ankle landmarks are less clear.",
                "La mesure du genou peut être légèrement moins précise.",
                "The knee measurement may be slightly less precise.",
            )
    elif test_type in {"aslr_right", "aslr_left"}:
        knee = metrics.get("raised_knee_extension_angle")
        knee_line_ratio = metrics.get("raised_knee_line_distance_ratio")
        preferred = float((metrics.get("quality_gate_config") or {}).get("raised_knee_extension_min", 145.0))
        # Launch-stable rule: the hip-to-ankle ASLR angle is primary. A knee
        # warning is shown only when both the joint angle and the knee's
        # displacement from the hip-to-ankle line indicate meaningful flexion.
        # This avoids false warnings caused by a slightly misplaced YOLO knee.
        knee_is_low = knee is not None and float(knee) < preferred
        knee_is_off_line = knee_line_ratio is not None and float(knee_line_ratio) > 0.06
        if knee_is_low and knee_is_off_line:
            add(
                "aslr_raised_knee_flexion",
                "Le repère détecté du genou suggère une possible légère flexion.",
                "The detected knee landmark suggests possible slight flexion.",
                "Le résultat reste une estimation; vous pouvez vérifier les repères, reprendre la photo ou l’accepter.",
                "The result remains an estimate; review the landmarks, retake the photo, or accept it.",
            )
        elif knee_is_low:
            metrics.setdefault("diagnostic_flags", []).append("knee_landmark_secondary_uncertainty_no_user_warning")

    metrics["screening_validation"] = {
        "status": "measurable_with_warning" if warnings else "measurable",
        "requires_user_acknowledgement": bool(warnings),
        "warnings": warnings,
        "screening_estimate": True,
        "diagnostic_claim": False,
    }
    if warnings:
        metrics["quality_label"] = "moderate"
    return result

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



@app.get("/latest_session")
def latest_session(authorization: str = Header(None)):
    """Return the newest usable screening session for the authenticated account."""
    if supabase is None:
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured on server.",
        )

    user = authenticated_user(supabase, authorization)

    completed = (
        supabase.table("sessions")
        .select("*")
        .eq("user_id", user["id"])
        .eq("status", "completed")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    session = completed.data[0] if completed.data else None

    # Compatibility fallback for legacy rows without user_id.
    if session is None:
        legacy = (
            supabase.table("sessions")
            .select("*")
            .ilike("user_email", user["email"])
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )
        for candidate in legacy.data or []:
            candidate_id = candidate.get("id")
            if not candidate_id:
                continue
            screenings = (
                supabase.table("screenings")
                .select("id,test_type")
                .eq("session_id", candidate_id)
                .limit(1)
                .execute()
            )
            if screenings.data:
                session = require_owned_session(user, candidate_id)
                break

    if session is None:
        return {
            "found": False,
            "user_email": user["email"],
            "session_id": None,
        }

    return {
        "found": True,
        "user_email": user["email"],
        "session_id": session.get("id"),
        "status": session.get("status"),
        "created_at": session.get("created_at"),
        "composite_score": session.get("composite_score"),
    }

def _safe_number(value):
    try:
        if value is None:
            return None
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except Exception:
        return None


def _mean_available(values):
    clean = [_safe_number(value) for value in values]
    clean = [value for value in clean if value is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 1)


def _symmetry_score(left_value, right_value):
    """
    Convert a bilateral score difference into a 0–100 symmetry score.

    A difference of 0 gives 100. The score decreases progressively as the
    left/right gap grows. Missing bilateral data returns None rather than
    inventing a value.
    """
    left = _safe_number(left_value)
    right = _safe_number(right_value)
    if left is None or right is None:
        return None
    return round(max(0.0, 100.0 - abs(left - right) * 2.0), 1)


def _score_lower_is_better(value, green_max, yellow_max, red_max):
    """
    Convert an angle where lower values are better into a 0–100 domain score.

    Green range -> 100 to 85
    Yellow range -> 85 to 60
    Red range -> 60 to 0
    """
    number = _safe_number(value)
    if number is None:
        return None

    number = max(0.0, min(float(red_max), number))

    if number <= float(green_max):
        if green_max <= 0:
            return 100.0
        return round(100.0 - (number / float(green_max)) * 15.0, 1)

    if number <= float(yellow_max):
        span = max(1e-6, float(yellow_max) - float(green_max))
        ratio = (number - float(green_max)) / span
        return round(85.0 - ratio * 25.0, 1)

    span = max(1e-6, float(red_max) - float(yellow_max))
    ratio = (number - float(yellow_max)) / span
    return round(max(0.0, 60.0 - ratio * 60.0), 1)



REQUIRED_SCREENING_TESTS = {
    "posture_side",
    "shoulder_right",
    "shoulder_left",
    "squat",
    "aslr_right",
    "aslr_left",
}

V3_BAND_SCORES = {
    "green": 88.0,
    "yellow": 64.0,
    "orange": 52.0,
    "red": 36.0,
}


def _as_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _screening_created_sort_key(screening):
    return str(screening.get("created_at") or "")


def deduplicate_screenings(screenings):
    """Keep the newest row for each test type and ignore duplicate inserts."""
    latest_by_test = {}
    for screening in sorted(screenings or [], key=_screening_created_sort_key):
        test_type = screening.get("test_type")
        if test_type in REQUIRED_SCREENING_TESTS:
            latest_by_test[test_type] = screening
    return latest_by_test


def is_complete_screening_set(screenings):
    return REQUIRED_SCREENING_TESTS.issubset(set(deduplicate_screenings(screenings).keys()))


def _metric_value(screening, metric_key, column_key=None):
    if column_key and screening.get(column_key) is not None:
        return _safe_number(screening.get(column_key))
    metrics = _as_dict(screening.get("metrics"))
    return _safe_number(metrics.get(metric_key))


def _fallback_metric_rating(test_type, metric_key, value):
    number = _safe_number(value)
    if number is None:
        return None

    if test_type == "posture_side" and metric_key == "neck_angle":
        return "green" if number < 10 else "yellow" if number < 20 else "red"
    if test_type == "posture_side" and metric_key == "thoracic_angle":
        return "green" if number < 5 else "yellow" if number < 15 else "red"
    if test_type in {"shoulder_right", "shoulder_left"}:
        return "red" if number < 160 else "yellow" if number < 170 else "green"
    if test_type in {"aslr_right", "aslr_left"}:
        return "red" if number < ASLR_RED_MAX_DEG else "yellow" if number <= ASLR_YELLOW_MAX_DEG else "green"
    if test_type == "squat" and metric_key == "knee_angle":
        return "green" if number < 95 else "yellow" if number < 110 else "red"
    if test_type == "squat" and metric_key == "trunk_lean":
        return "green" if number < 15 else "yellow" if number < 25 else "red"
    return None


def _screening_metric_rating(screening, threshold_key, metric_key, column_key=None):
    value = _metric_value(screening, metric_key, column_key)

    # ASLR thresholds were recalibrated in V101.28.4. Recompute the rating from
    # the measured angle so historical rows with the previous 45°/70° payload
    # are displayed consistently under the current 60°/75° policy.
    if screening.get("test_type") in {"aslr_right", "aslr_left"}:
        return _fallback_metric_rating(screening.get("test_type"), metric_key, value)

    thresholds = _as_dict(screening.get("thresholds"))
    threshold = _as_dict(thresholds.get(threshold_key))
    rating = str(threshold.get("rating") or "").strip().lower()
    if rating in V3_BAND_SCORES:
        return rating
    return _fallback_metric_rating(screening.get("test_type"), metric_key, value)


def _v3_band_score(rating):
    return V3_BAND_SCORES.get(str(rating or "").lower())


def calculate_v3_score_from_screenings(screenings):
    """
    Rebuild the Evidence-Aware V3 score from the newest unique row for each of
    the six required tests. This mirrors engines/score_engine_v2.py (V3).
    """
    by_test = deduplicate_screenings(screenings)
    if not REQUIRED_SCREENING_TESTS.issubset(set(by_test.keys())):
        return None

    posture = by_test["posture_side"]
    shoulder_r = by_test["shoulder_right"]
    shoulder_l = by_test["shoulder_left"]
    squat = by_test["squat"]
    aslr_r = by_test["aslr_right"]
    aslr_l = by_test["aslr_left"]

    domains = []

    neck = _v3_band_score(_screening_metric_rating(posture, "neck_angle", "neck_angle", "neck_angle_deg"))
    thoracic = _v3_band_score(_screening_metric_rating(posture, "thoracic_angle", "thoracic_angle", "thoracic_angle_deg"))
    shoulder_scores = [
        _v3_band_score(_screening_metric_rating(shoulder_r, "shoulder_flexion", "shoulder_flexion_angle", "shoulder_flexion_angle_deg")),
        _v3_band_score(_screening_metric_rating(shoulder_l, "shoulder_flexion", "shoulder_flexion_angle", "shoulder_flexion_angle_deg")),
    ]
    aslr_scores = [
        _v3_band_score(_screening_metric_rating(aslr_r, "aslr_angle", "aslr_angle")),
        _v3_band_score(_screening_metric_rating(aslr_l, "aslr_angle", "aslr_angle")),
    ]
    squat_scores = [
        _v3_band_score(_screening_metric_rating(squat, "knee_angle", "knee_angle", "squat_knee_angle_deg")),
        _v3_band_score(_screening_metric_rating(squat, "trunk_lean", "trunk_lean", "squat_trunk_lean_deg")),
    ]

    if neck is not None:
        domains.append((neck, 15.0))
    if thoracic is not None:
        domains.append((thoracic, 15.0))

    clean_shoulders = [v for v in shoulder_scores if v is not None]
    if clean_shoulders:
        domains.append((sum(clean_shoulders) / len(clean_shoulders), 25.0))

    clean_aslr = [v for v in aslr_scores if v is not None]
    if clean_aslr:
        domains.append((sum(clean_aslr) / len(clean_aslr), 20.0))

    clean_squat = [v for v in squat_scores if v is not None]
    if clean_squat:
        domains.append((sum(clean_squat) / len(clean_squat), 25.0))

    if not domains:
        return None

    weight_sum = sum(weight for _, weight in domains)
    return round(sum(score * weight for score, weight in domains) / weight_sum, 1)


def session_with_screening_scores(session, screenings, v3_score=None):
    """Create a non-persistent session view from deduplicated screening rows."""
    by_test = deduplicate_screenings(screenings)
    result = dict(session or {})
    field_map = {
        "posture_side": "posture_score",
        "shoulder_right": "shoulder_right_score",
        "shoulder_left": "shoulder_left_score",
        "squat": "squat_score",
        "aslr_right": "aslr_right_score",
        "aslr_left": "aslr_left_score",
    }
    for test_type, field in field_map.items():
        screening = by_test.get(test_type)
        if screening is not None:
            result[field] = _safe_number(screening.get("score"))
    if v3_score is not None:
        result["composite_score"] = v3_score
    return result


def build_movement_profile_from_session(session, screenings=None):
    """
    Build a six-domain descriptive movement profile from measurements that
    actually exist in the completed session.

    Trunk control is derived from direct trunk-related measurements when
    available:
    - thoracic alignment from the side-posture screening
    - trunk lean during the squat screening

    Missing tests remain None and are never converted to zero.
    """
    screenings = screenings or []

    posture = _safe_number(session.get("posture_score"))

    shoulder_right = _safe_number(session.get("shoulder_right_score"))
    shoulder_left = _safe_number(session.get("shoulder_left_score"))
    shoulder_mobility = _mean_available([shoulder_right, shoulder_left])

    squat = _safe_number(session.get("squat_score"))
    aslr_right = _safe_number(session.get("aslr_right_score"))
    aslr_left = _safe_number(session.get("aslr_left_score"))
    aslr_mobility = _mean_available([aslr_right, aslr_left])
    lower_body_mobility = _mean_available([squat, aslr_mobility])

    thoracic_angle = None
    squat_trunk_lean = None

    for screening in screenings:
        test_type = screening.get("test_type")
        metrics = screening.get("metrics") or {}

        if test_type == "posture_side":
            thoracic_angle = (
                screening.get("thoracic_angle_deg")
                if screening.get("thoracic_angle_deg") is not None
                else metrics.get("thoracic_angle")
            )

        elif test_type == "squat":
            squat_trunk_lean = (
                screening.get("squat_trunk_lean_deg")
                if screening.get("squat_trunk_lean_deg") is not None
                else metrics.get("trunk_lean")
            )

    thoracic_alignment_score = _score_lower_is_better(
        thoracic_angle,
        green_max=5,
        yellow_max=15,
        red_max=45,
    )
    squat_trunk_control_score = _score_lower_is_better(
        squat_trunk_lean,
        green_max=15,
        yellow_max=25,
        red_max=60,
    )

    trunk_control = _mean_available([
        thoracic_alignment_score,
        squat_trunk_control_score,
    ])

    # Compatibility fallback for older sessions where detailed screening
    # metrics are unavailable.
    if trunk_control is None:
        trunk_control = _mean_available([posture, squat])

    shoulder_symmetry = _symmetry_score(shoulder_left, shoulder_right)
    aslr_symmetry = _symmetry_score(aslr_left, aslr_right)
    symmetry = _mean_available([shoulder_symmetry, aslr_symmetry])

    overall = _safe_number(session.get("composite_score"))
    if overall is None:
        overall = compute_composite(
            posture,
            shoulder_right,
            shoulder_left,
            squat,
            aslr_right,
            aslr_left,
        )

    return {
        "posture": posture,
        "shoulder_mobility": shoulder_mobility,
        "lower_body_mobility": lower_body_mobility,
        "trunk_control": trunk_control,

        # Temporary compatibility alias so the currently deployed frontend
        # does not break before progress.tsx is updated.
        "movement_control": trunk_control,

        "symmetry": symmetry,
        "overall_movement_capacity": overall,
    }


@app.get("/screening_history")
def screening_history(
    limit: int = 6,
    authorization: str = Header(None),
):
    """
    Rebuild the latest valid assessments from their individual screening rows.

    A history point is included only when all six required test types exist.
    Duplicate inserts are collapsed by test type, keeping the newest row.
    Every score is recalculated with Evidence-Aware Score Engine V3; stored
    legacy composite_score values are never trusted.
    """
    if supabase is None:
        return {"error": "Supabase is not configured on server."}

    user = authenticated_user(supabase, authorization)
    normalized_email = user["email"]

    safe_limit = max(1, min(int(limit or 6), 12))

    session_columns = (
        "id,user_id,user_email,status,created_at,composite_score,"
        "posture_score,shoulder_right_score,shoulder_left_score,"
        "squat_score,aslr_right_score,aslr_left_score"
    )
    sessions_resp = (
        supabase.table("sessions")
        .select(session_columns)
        .eq("user_id", user["id"])
        .order("created_at", desc=True)
        .limit(max(safe_limit * 8, 48))
        .execute()
    )

    if not sessions_resp.data:
        sessions_resp = (
            supabase.table("sessions")
            .select(session_columns)
            .ilike("user_email", normalized_email)
            .order("created_at", desc=True)
            .limit(max(safe_limit * 8, 48))
            .execute()
        )

    usable = []

    for session in sessions_resp.data or []:
        session_id = session.get("id")
        if not session_id:
            continue

        # A modern assessment is finalized with sessions.status=completed.
        # Legacy production sessions were not always finalized even though all
        # six required tests were successfully persisted. Validate the actual
        # six-test evidence below so those complete historical assessments are
        # retained while partial/abandoned sessions remain excluded.
        screenings_resp = (
            supabase.table("screenings")
            .select(
                "id,created_at,test_type,score,confidence,metrics,thresholds,"
                "neck_angle_deg,thoracic_angle_deg,shoulder_flexion_angle_deg,"
                "squat_knee_angle_deg,squat_trunk_lean_deg"
            )
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .execute()
        )
        session_screenings = screenings_resp.data or []

        if not is_complete_screening_set(session_screenings):
            continue

        v3_score = calculate_v3_score_from_screenings(session_screenings)
        if v3_score is None:
            continue

        effective_session = session_with_screening_scores(
            session,
            session_screenings,
            v3_score=v3_score,
        )

        usable.append({
            "session_id": session_id,
            "created_at": session.get("created_at"),
            "status": "completed",
            "stored_status": session.get("status"),
            "completion_basis": (
                "finalized_session"
                if str(session.get("status") or "").strip().lower() == "completed"
                else "legacy_complete_six_test_set"
            ),
            "score": v3_score,
            "score_version": "FlexiLab Evidence-Aware Score Engine V3",
            "movement_profile": build_movement_profile_from_session(
                effective_session,
                list(deduplicate_screenings(session_screenings).values()),
            ),
        })

        if len(usable) >= safe_limit:
            break

    usable.reverse()

    return {
        "found": bool(usable),
        "user_email": normalized_email,
        "count": len(usable),
        "screenings": usable,
        "latest": usable[-1] if usable else None,
        "profile_method": "v3_rebuilt_unique_six_test_history_v2",
        "profile_disclaimer": (
            "History is rebuilt from the newest unique row for each of the six "
            "required tests. Scores use Evidence-Aware Score Engine V3."
        ),
    }



def _release_previous_unfinished_credit_reservations(
    credit_owner_user_id: str,
) -> dict:
    """Release only genuine active reservations from screening_usage.

    Historical sessions with status=in_progress are not evidence of a reserved
    credit and must never reduce availability. The screening_usage ledger is
    the authoritative source.
    """
    released_session_ids = []
    failed_session_ids = []

    try:
        response = (
            supabase.table("screening_usage")
            .select("id,session_id,usage_status,created_at")
            .eq("user_id", credit_owner_user_id)
            .eq("usage_status", "reserved")
            .order("created_at", desc=False)
            .execute()
        )
    except Exception:
        response = None

    for row in (response.data if response else []) or []:
        session_id = str(row.get("session_id") or "").strip()
        if not session_id:
            continue

        released = release_credit(
            supabase,
            credit_owner_user_id,
            session_id,
            reason="superseded_by_new_screening",
        )

        if released:
            released_session_ids.append(session_id)
            try:
                supabase.table("sessions").update({
                    "status": "abandoned",
                }).eq("id", session_id).neq("status", "completed").execute()
            except Exception:
                pass
        else:
            failed_session_ids.append(session_id)

    return {
        "released_session_ids": released_session_ids,
        "failed_session_ids": failed_session_ids,
    }


@app.post("/start_session")
def start_session(
    user_email: str = Form(...),
    intake_json: str = Form(None),
    questionnaire_json: str = Form(None),
    trainer_client_link_id: str = Form(None),
    authorization: str = Header(None),
):
    if supabase is None:
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured on server.",
        )

    user = authenticated_user(supabase, authorization)

    intake_data = parse_intake_payload(
        intake_json=intake_json,
        questionnaire_json=questionnaire_json,
    )

    credit_owner_user_id = user["id"]
    session_owner_user_id = user["id"]

    # The bearer token is the authoritative identity for self-assessments.
    # Never trust a browser/localStorage email over the authenticated account.
    # For Trainer-client assessments, the validated client link below replaces
    # both the owner email and owner user ID.
    session_owner_email = str(user["email"] or "").strip().lower()
    trainer_id = None
    trainer_link_id = None

    if trainer_client_link_id:
        link_response = (
            supabase.table("trainer_clients")
            .select("*")
            .eq("id", trainer_client_link_id)
            .eq("trainer_id", user["id"])
            .limit(1)
            .execute()
        )
        if not link_response.data:
            raise HTTPException(status_code=404, detail="Trainer client not found.")
        link = link_response.data[0]
        link_status = str(link.get("status") or "").strip().lower()
        if link_status not in {"pending", "active"}:
            raise HTTPException(
                status_code=409,
                detail="This Trainer-client link is not available for screening.",
            )

        # The invitation email can fail to create a Supabase Auth UUID immediately.
        # In that case, create the screening under the invited email and link ID.
        # The session user_id is backfilled automatically when the client first signs in.
        client_user_id = str(link.get("client_user_id") or "").strip()
        session_owner_user_id = client_user_id or None
        session_owner_email = str(link.get("invited_email") or "").strip().lower()
        trainer_id = user["id"]
        trainer_link_id = str(link["id"])

    reservation_cleanup = _release_previous_unfinished_credit_reservations(
        credit_owner_user_id,
    )

    if reservation_cleanup.get("failed_session_ids"):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "SCREENING_RESERVATION_RELEASE_FAILED",
                "message": (
                    "Your previous unfinished screening is still reserving a "
                    "credit. Please retry once; no credit has been consumed."
                ),
                "reserved_session_count": len(
                    reservation_cleanup.get("failed_session_ids") or []
                ),
            },
        )

    session_row = {
        "user_email": session_owner_email,
        "user_id": session_owner_user_id,
        "status": "in_progress",
        "trainer_id": trainer_id,
        "performed_by_user_id": user["id"],
        "trainer_client_link_id": trainer_link_id,
        "credit_owner_user_id": credit_owner_user_id,
    }

    resp = supabase.table("sessions").insert(session_row).execute()
    if not resp.data:
        raise HTTPException(
            status_code=500,
            detail="Unable to create the screening session.",
        )

    session_id = resp.data[0]["id"]

    try:
        reservation = reserve_credit(
            supabase,
            credit_owner_user_id,
            session_id,
        )
    except Exception:
        # Do not retain an unusable session when no credit was reserved.
        try:
            supabase.table("sessions").delete().eq(
                "id",
                session_id,
            ).execute()
        except Exception:
            pass
        raise

    try_save_session_intake(session_id, intake_data)

    return {
        "session_id": session_id,
        "intake_json": intake_data,
        "questionnaire_json": intake_data,
        "screening_credits_remaining": reservation.get(
            "credits_remaining",
            0,
        ),
        "released_previous_unfinished_sessions": len(
            reservation_cleanup.get("released_session_ids") or []
        ),
    }



@app.post("/abandon_session")
def abandon_session(
    session_id: str = Form(...),
    authorization: str = Header(None),
):
    """Explicitly abandon an unfinished screening and release its reservation."""
    if supabase is None:
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured on server.",
        )

    user = authenticated_user(supabase, authorization)

    response = (
        supabase.table("sessions")
        .select(
            "id,status,user_id,trainer_id,performed_by_user_id,"
            "credit_owner_user_id"
        )
        .eq("id", session_id)
        .limit(1)
        .execute()
    )
    if not response.data:
        raise HTTPException(status_code=404, detail="Session not found.")

    row = response.data[0]
    allowed_user_ids = {
        str(row.get("user_id") or ""),
        str(row.get("trainer_id") or ""),
        str(row.get("performed_by_user_id") or ""),
    }
    if user["id"] not in allowed_user_ids:
        raise HTTPException(
            status_code=403,
            detail="This screening session belongs to another account.",
        )

    if str(row.get("status") or "").lower() == "completed":
        return {
            "session_id": session_id,
            "status": "completed",
            "released": False,
            "reason": "completed_session_is_immutable",
        }

    credit_owner_user_id = str(
        row.get("credit_owner_user_id")
        or row.get("trainer_id")
        or row.get("user_id")
        or user["id"]
    )

    released = release_credit(
        supabase,
        credit_owner_user_id,
        session_id,
        reason="user_quit_screening",
    )

    # Mark abandoned even when the reservation had already been released.
    supabase.table("sessions").update({
        "status": "abandoned",
    }).eq("id", session_id).neq("status", "completed").execute()


    return {
        "session_id": session_id,
        "status": "abandoned",
        "released": bool(released),
    }


@app.post("/finalize_session")
def finalize_session(
    session_id: str = Form(...),
    authorization: str = Header(None),
):
    if supabase is None:
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured on server.",
        )

    user = authenticated_user(supabase, authorization)

    s = (
        supabase.table("sessions")
        .select("*")
        .eq("id", session_id)
        .limit(1)
        .execute()
    )
    if not s.data:
        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        )

    row = s.data[0]

    allowed_user_ids = {
        str(row.get("user_id") or ""),
        str(row.get("trainer_id") or ""),
        str(row.get("performed_by_user_id") or ""),
    }
    if user["id"] not in allowed_user_ids:
        raise HTTPException(
            status_code=403,
            detail="This screening session belongs to another account.",
        )

    screenings_resp = (
        supabase.table("screenings")
        .select(
            "id,created_at,test_type,score,confidence,metrics,thresholds,"
            "neck_angle_deg,thoracic_angle_deg,shoulder_flexion_angle_deg,"
            "squat_knee_angle_deg,squat_trunk_lean_deg"
        )
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .execute()
    )
    screenings = screenings_resp.data or []

    if not is_complete_screening_set(screenings):
        found = sorted(deduplicate_screenings(screenings).keys())
        missing = sorted(REQUIRED_SCREENING_TESTS.difference(found))
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INCOMPLETE_SCREENING",
                "message": (
                    "Session cannot be finalized because required tests "
                    "are missing."
                ),
                "missing_tests": missing,
            },
        )

    v3_score = calculate_v3_score_from_screenings(screenings)
    if v3_score is None:
        raise HTTPException(
            status_code=422,
            detail="Unable to calculate Evidence-Aware Score Engine V3 score.",
        )

    effective = session_with_screening_scores(
        row,
        screenings,
        v3_score=v3_score,
    )
    update_payload = {
        "posture_score": effective.get("posture_score"),
        "shoulder_right_score": effective.get("shoulder_right_score"),
        "shoulder_left_score": effective.get("shoulder_left_score"),
        "squat_score": effective.get("squat_score"),
        "aslr_right_score": effective.get("aslr_right_score"),
        "aslr_left_score": effective.get("aslr_left_score"),
        "composite_score": v3_score,
        "status": "completed",
    }

    supabase.table("sessions").update(update_payload).eq(
        "id",
        session_id,
    ).execute()

    credit_owner_user_id = str(
        row.get("credit_owner_user_id")
        or row.get("trainer_id")
        or row.get("user_id")
        or user["id"]
    )
    credit_result = consume_credit(
        supabase,
        credit_owner_user_id,
        session_id,
    )

    # Client-app lifecycle: when the user's previous program is already
    # complete, a newly completed self-assessment immediately receives its new
    # program. An unfinished program is never replaced automatically.
    automatic_program = {
        "generated": False,
        "reason": "not_eligible",
        "program_id": None,
    }
    client_owner_id = str(row.get("user_id") or "")
    if client_owner_id and user["id"] == client_owner_id:
        try:
            existing_for_session = _program_for_screening(user["id"], session_id)
            if existing_for_session:
                automatic_program = {
                    "generated": False,
                    "reason": "already_generated_for_assessment",
                    "program_id": str(existing_for_session.get("id")),
                }
            else:
                previous_program = _current_program_row(user["id"])
                previous_completion = (
                    _program_completion_summary(previous_program)
                    if previous_program
                    else None
                )
                if previous_program and previous_completion["is_completed"]:
                    generated_payload = _generate_program_for_session(
                        session_id=session_id,
                        lang="en",
                        authorization=authorization,
                    )
                    automatic_program = {
                        "generated": True,
                        "reason": "previous_program_completed",
                        "program_id": generated_payload.get("program_id"),
                    }
                elif previous_program:
                    automatic_program = {
                        "generated": False,
                        "reason": "active_program_unfinished",
                        "program_id": str(previous_program.get("id")),
                        "remaining_sessions": previous_completion["remaining_sessions"],
                    }
                else:
                    automatic_program = {
                        "generated": False,
                        "reason": "first_program_available_on_program_page",
                        "program_id": None,
                    }
        except Exception as exc:
            # Assessment completion and credit consumption remain authoritative;
            # a program-generation failure must never invalidate the screening.
            logger.exception("automatic_program_generation_failed session_id=%s", session_id)
            automatic_program = {
                "generated": False,
                "reason": "generation_failed",
                "error": str(exc),
                "program_id": None,
            }

    return {
        "session_id": session_id,
        "status": "completed",
        **update_payload,
        "score_version": "FlexiLab Evidence-Aware Score Engine V3",
        "screening_credits_remaining": credit_result.get(
            "credits_remaining",
            0,
        ),
        "automatic_program": automatic_program,
    }



class AnalysisWithDiagnosticsError(ValueError):
    """Controlled analysis rejection carrying an ephemeral Vision QA payload."""

    def __init__(self, message, diagnostic_result):
        self.diagnostic_result = diagnostic_result
        super().__init__(message)


def _pose_arrays(prediction):
    result = prediction[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        raise ValueError("No person detected.")
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    main_idx = int(np.argmax(areas))
    box_confidences = (
        result.boxes.conf.cpu().numpy()
        if getattr(result.boxes, "conf", None) is not None
        else np.zeros(len(boxes), dtype=float)
    )
    xy = result.keypoints.xy[main_idx].cpu().numpy()
    conf = result.keypoints.conf[main_idx].cpu().numpy()
    return boxes, areas, box_confidences, main_idx, xy, conf


def _expanded_person_crop(img, box, *, aslr=False):
    height, width = img.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    pad_x = box_width * (0.20 if aslr else 0.12)
    pad_y = box_height * (0.18 if aslr else 0.12)
    left = max(0, int(math.floor(x1 - pad_x)))
    top = max(0, int(math.floor(y1 - pad_y)))
    right = min(width, int(math.ceil(x2 + pad_x)))
    bottom = min(height, int(math.ceil(y2 + pad_y)))
    if right - left < 160 or bottom - top < 160:
        return None, None
    return img[top:bottom, left:right].copy(), {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": right - left,
        "height": bottom - top,
    }


def _map_crop_pose_to_full(xy, boxes, crop_bounds):
    mapped_xy = np.array(xy, dtype=float, copy=True)
    mapped_xy[:, 0] += float(crop_bounds["left"])
    mapped_xy[:, 1] += float(crop_bounds["top"])
    mapped_boxes = np.array(boxes, dtype=float, copy=True)
    mapped_boxes[:, [0, 2]] += float(crop_bounds["left"])
    mapped_boxes[:, [1, 3]] += float(crop_bounds["top"])
    return mapped_xy, mapped_boxes


def _map_rotated_cw_pose_to_original(xy, boxes, original_shape):
    """Map 90-degree-clockwise inference coordinates back to the source image."""
    original_height, original_width = original_shape[:2]
    mapped_xy = np.array(xy, dtype=float, copy=True)
    rotated_x = mapped_xy[:, 0].copy()
    rotated_y = mapped_xy[:, 1].copy()
    mapped_xy[:, 0] = rotated_y
    mapped_xy[:, 1] = float(original_height - 1) - rotated_x

    mapped_boxes = []
    for box in np.array(boxes, dtype=float):
        x1, y1, x2, y2 = [float(value) for value in box[:4]]
        corners = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=float,
        )
        source_x = corners[:, 1]
        source_y = float(original_height - 1) - corners[:, 0]
        mapped_boxes.append(
            [
                float(np.min(source_x)),
                float(np.min(source_y)),
                float(np.max(source_x)),
                float(np.max(source_y)),
            ]
        )
    return mapped_xy, np.array(mapped_boxes, dtype=float)


def _map_rotated_ccw_pose_to_original(xy, boxes, original_shape):
    """Map 90-degree-counterclockwise inference coordinates back to source image."""
    original_height, original_width = original_shape[:2]
    mapped_xy = np.array(xy, dtype=float, copy=True)
    rotated_x = mapped_xy[:, 0].copy()
    rotated_y = mapped_xy[:, 1].copy()
    mapped_xy[:, 0] = float(original_width - 1) - rotated_y
    mapped_xy[:, 1] = rotated_x

    mapped_boxes = []
    for box in np.array(boxes, dtype=float):
        x1, y1, x2, y2 = [float(value) for value in box[:4]]
        corners = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=float,
        )
        source_x = float(original_width - 1) - corners[:, 1]
        source_y = corners[:, 0]
        mapped_boxes.append(
            [
                float(np.min(source_x)),
                float(np.min(source_y)),
                float(np.max(source_x)),
                float(np.max(source_y)),
            ]
        )
    return mapped_xy, np.array(mapped_boxes, dtype=float)


def _aslr_pose_pass_quality(result):
    """Score pose-pass reliability without selecting the largest clinical angle."""
    metrics = result.get("metrics") or {}
    confidence = float(result.get("confidence") or 0.0)
    chain_score = float(metrics.get("selected_chain_score") or 0.0)
    mean_conf = float(metrics.get("selected_limb_mean_confidence") or 0.0)
    min_conf = float(metrics.get("selected_limb_min_confidence") or 0.0)
    angle = float(metrics.get("aslr_angle") or 0.0)
    endpoint_count = int(metrics.get("detected_ankle_endpoint_count") or 0)
    endpoints_distinct = bool(metrics.get("ankle_endpoints_are_distinct"))
    resting_verified = bool(metrics.get("resting_leg_verified"))
    estimator_spread = float((metrics.get("angle_estimators") or {}).get("spread") or 0.0)
    flags = set(metrics.get("diagnostic_flags") or [])

    quality = (
        confidence * 0.36
        + chain_score * 0.24
        + mean_conf * 0.18
        + min_conf * 0.12
    )
    if endpoints_distinct:
        quality += 0.12
    elif endpoint_count < 2:
        quality -= 0.08
    if resting_verified:
        quality += 0.08
    if estimator_spread <= 5.0:
        quality += 0.08
    elif estimator_spread > 10.0:
        quality -= 0.12
    if "resting_ankle_not_independently_resolved" in flags:
        quality -= 0.06
    if "ankle_endpoints_duplicated_by_pose_model" in flags:
        quality -= 0.10
    # A lone near-baseline endpoint is likely the resting ankle. This is a
    # detection-quality penalty, not a preference for a larger clinical result.
    if not endpoints_distinct and angle < 18.0:
        quality -= 0.50
    return round(float(quality), 6)



def _aslr_same_side_chain_quality(xy, conf):
    """Return the strongest coherent COCO hip-knee-ankle detection quality.

    This is a detection selector only. It never computes or changes the ASLR
    clinical angle. The measurement engine remains authoritative.
    """
    candidates = []
    for label, hip_i, knee_i, ankle_i in (
        ("H11-K13-A15", 11, 13, 15),
        ("H12-K14-A16", 12, 14, 16),
    ):
        values = [float(conf[i]) if i < len(conf) else 0.0 for i in (hip_i, knee_i, ankle_i)]
        minimum = min(values)
        mean = sum(values) / 3.0
        knee_extension = 0.0
        try:
            hip = np.asarray(xy[hip_i], dtype=float)
            knee = np.asarray(xy[knee_i], dtype=float)
            ankle = np.asarray(xy[ankle_i], dtype=float)
            a = hip - knee
            b = ankle - knee
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom > 1e-6:
                cosine = max(-1.0, min(1.0, float(np.dot(a, b) / denom)))
                knee_extension = float(math.degrees(math.acos(cosine)))
        except Exception:
            knee_extension = 0.0

        geometry_factor = max(0.0, min(1.0, (knee_extension - 100.0) / 60.0))
        score = mean * 0.55 + minimum * 0.35 + geometry_factor * 0.10
        candidates.append({
            "label": label,
            "score": round(score, 6),
            "minimum_confidence": round(minimum, 6),
            "mean_confidence": round(mean, 6),
            "knee_extension_deg": round(knee_extension, 3),
            "valid_detection": minimum >= ASLR_KEYPOINT_MIN_CONF and mean >= ASLR_REQUIRED_MEAN_CONF,
        })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[0], candidates

def _relevant_pose_confidence(conf, test_type):
    test_type = str(test_type or "")
    if test_type.startswith("aslr"):
        indices = [5, 6, 11, 12, 13, 14, 15, 16]
    elif test_type.startswith("shoulder"):
        indices = [5, 6, 7, 8, 9, 10, 11, 12]
    elif test_type == "posture_side":
        indices = [3, 4, 5, 6, 11, 12]
    else:
        indices = [5, 6, 11, 12, 13, 14, 15, 16]
    values = [float(conf[index]) for index in indices if index < len(conf)]
    return float(np.mean(values)) if values else 0.0


def _attach_measurement_points(result, test_type, xy):
    metrics = result.setdefault("metrics", {})
    if test_type == "posture_side":
        side = str(metrics.get("side_used") or "RIGHT").upper()
        indices = (4, 6, 12) if side == "RIGHT" else (3, 5, 11)
        ear_i, shoulder_i, hip_i = indices
        metrics["measurement_points"] = {
            "indices": {"ear": ear_i, "shoulder": shoulder_i, "hip": hip_i},
            "points": {
                "ear": {"x": round(float(xy[ear_i][0]), 2), "y": round(float(xy[ear_i][1]), 2)},
                "shoulder": {"x": round(float(xy[shoulder_i][0]), 2), "y": round(float(xy[shoulder_i][1]), 2)},
                "hip": {"x": round(float(xy[hip_i][0]), 2), "y": round(float(xy[hip_i][1]), 2)},
            },
        }
    elif test_type in {"shoulder_right", "shoulder_left"}:
        selected = metrics.get("selected_source_indices") or {}
        if selected:
            shoulder_i = int(selected.get("shoulder"))
            elbow_i = int(selected.get("elbow"))
            wrist_i = int(selected.get("wrist"))
            hip_i = int(selected.get("hip"))
        else:
            side = str(metrics.get("side") or "RIGHT").upper()
            shoulder_i, elbow_i, wrist_i, hip_i = (6, 8, 10, 12) if side == "RIGHT" else (5, 7, 9, 11)
        metrics["measurement_points"] = {
            "indices": {
                "shoulder": shoulder_i,
                "elbow": elbow_i,
                "wrist": wrist_i,
                "hip": hip_i,
            },
            "points": {
                "shoulder": {"x": round(float(xy[shoulder_i][0]), 2), "y": round(float(xy[shoulder_i][1]), 2)},
                "elbow": {"x": round(float(xy[elbow_i][0]), 2), "y": round(float(xy[elbow_i][1]), 2)},
                "wrist": {"x": round(float(xy[wrist_i][0]), 2), "y": round(float(xy[wrist_i][1]), 2)},
                "hip": {"x": round(float(xy[hip_i][0]), 2), "y": round(float(xy[hip_i][1]), 2)},
            },
        }
    elif test_type == "squat":
        shoulder = (xy[5] + xy[6]) / 2
        hip = (xy[11] + xy[12]) / 2
        knee = (xy[13] + xy[14]) / 2
        ankle = (xy[15] + xy[16]) / 2
        metrics["measurement_points"] = {
            "method": "bilateral_midpoints_current_engine",
            "points": {
                "shoulder": {"x": round(float(shoulder[0]), 2), "y": round(float(shoulder[1]), 2)},
                "hip": {"x": round(float(hip[0]), 2), "y": round(float(hip[1]), 2)},
                "knee": {"x": round(float(knee[0]), 2), "y": round(float(knee[1]), 2)},
                "ankle": {"x": round(float(ankle[0]), 2), "y": round(float(ankle[1]), 2)},
            },
        }


def _without_ephemeral_vision_qa(result):
    persistent = copy.deepcopy(result)
    metrics = persistent.get("metrics")
    if isinstance(metrics, dict):
        metrics.pop("vision_qa", None)
    return persistent


def _deliver_and_scrub_vision_qa(job_id, result_json):
    """Return the overlay once, then remove the composite from the queued job row.

    The authoritative screening row never contains the overlay. This additional
    scrub prevents the diagnostic composite from remaining in analysis_jobs after
    the frontend receives it. Failures are logged and never block screening UX.
    """
    if not VISION_QA_ONE_TIME_DELIVERY or not isinstance(result_json, dict):
        return result_json
    metrics = result_json.get("metrics")
    if not isinstance(metrics, dict) or "vision_qa" not in metrics:
        return result_json

    scrubbed = _without_ephemeral_vision_qa(result_json)
    try:
        _execute_with_transient_retry(
            lambda: (
                supabase.table("analysis_jobs")
                .update({"result_json": scrubbed})
                .eq("id", job_id)
                .execute()
            ),
            label="vision_qa_one_time_scrub",
        )
        logger.info("vision_qa_delivered_and_scrubbed job_id=%s", job_id)
    except Exception:
        logger.exception("vision_qa_scrub_failed job_id=%s", job_id)
    return result_json


def detect_pose_with_fallback(img, test_type, inference_imgsz=None):
    """Run YOLO pose inference under one model lock with one safe reload."""
    global model, POSE_MODEL_LOAD_ERROR, POSE_MODEL_RELOAD_COUNT

    is_aslr = str(test_type).startswith("aslr")
    thresholds = [0.20, 0.14, 0.10] if is_aslr else [0.50, 0.35, 0.25]
    requested_imgsz = int(inference_imgsz or POSE_INFERENCE_IMGSZ)
    requested_imgsz = max(320, min(1280, requested_imgsz))

    with POSE_MODEL_INFERENCE_LOCK:
        if model is None:
            model = _load_pose_model()
        if model is None:
            raise ValueError("Pose model is temporarily unavailable. Please retry shortly.")

        recovered_once = False
        for threshold in thresholds:
            while True:
                try:
                    prediction = model(
                        img,
                        conf=threshold,
                        classes=[0],
                        imgsz=requested_imgsz,
                        verbose=False,
                    )
                    break
                except AttributeError as exc:
                    error_text = str(exc)
                    known_fused_conv_error = (
                        "Conv" in error_text
                        and "has no attribute" in error_text
                        and "bn" in error_text
                    )
                    if not known_fused_conv_error or recovered_once:
                        raise ValueError(
                            "Pose analysis is temporarily unavailable. Please retry."
                        ) from exc

                    logger.warning(
                        "Reloading pose model after fused Conv state error: %s",
                        error_text,
                    )
                    recovered_once = True
                    POSE_MODEL_RELOAD_COUNT += 1
                    model = _load_pose_model()
                    if model is None:
                        raise ValueError(
                            "Pose model could not be reloaded. Please retry shortly."
                        ) from exc
                except Exception as exc:
                    logger.exception("YOLO pose inference failed")
                    raise ValueError(
                        "Pose analysis is temporarily unavailable. Please retry."
                    ) from exc

            if (
                prediction
                and prediction[0].keypoints is not None
                and len(prediction[0].keypoints.xy) > 0
            ):
                return prediction, threshold, requested_imgsz

    raise ValueError(
        "No person detected. Keep the required body area visible, improve lighting, "
        "and avoid cropping the active joints."
    )


def detect_aslr_pose_with_fallback(img, inference_imgsz=None):
    """Run exactly one dedicated ASLR YOLO inference.

    V101.35.14 removes the confidence-threshold retry loop. A single model call
    is made at a permissive person-detection threshold; anatomical confidence
    and geometry gates in the ASLR engine decide whether the result is usable.
    """
    global aslr_model, ASLR_POSE_MODEL_LOAD_ERROR, ASLR_POSE_MODEL_RELOAD_COUNT

    threshold = 0.18
    requested_imgsz = int(inference_imgsz or 960)
    requested_imgsz = max(640, min(1280, requested_imgsz))

    with ASLR_POSE_MODEL_INFERENCE_LOCK:
        if aslr_model is None:
            aslr_model = _load_aslr_pose_model()
        if aslr_model is None:
            raise ValueError(
                "The dedicated ASLR pose model is temporarily unavailable. Please retry shortly."
            )

        try:
            prediction = aslr_model(
                img,
                conf=threshold,
                classes=[0],
                imgsz=requested_imgsz,
                verbose=False,
            )
        except AttributeError as exc:
            error_text = str(exc)
            known_fused_conv_error = (
                "Conv" in error_text
                and "has no attribute" in error_text
                and "bn" in error_text
            )
            if not known_fused_conv_error:
                raise ValueError(
                    "ASLR pose analysis is temporarily unavailable. Please retry."
                ) from exc
            # Reloading repairs the model object but deliberately does not run a
            # second inference in the same request. The user can retry once.
            logger.warning(
                "Reloading dedicated ASLR pose model after fused Conv error: %s",
                error_text,
            )
            ASLR_POSE_MODEL_RELOAD_COUNT += 1
            aslr_model = _load_aslr_pose_model()
            raise ValueError(
                "ASLR pose analysis was reset. Please retry the photo once."
            ) from exc
        except Exception as exc:
            logger.exception("Dedicated ASLR YOLO inference failed")
            raise ValueError(
                "ASLR pose analysis is temporarily unavailable. Please retry."
            ) from exc

        if (
            prediction
            and prediction[0].keypoints is not None
            and len(prediction[0].keypoints.xy) > 0
        ):
            return prediction, threshold, requested_imgsz

    raise ValueError(
        "The ASLR pose could not be detected reliably. Keep the pelvis, raised knee, "
        "and complete raised foot visible, then retake the photo."
    )


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    user_email: str = Form(...),
    test_type: str = Form(...),
    session_id: str = Form(...),
    intake_json: str = Form(None),
    questionnaire_json: str = Form(None),
    capture_metadata_json: str = Form(None),
    authorization: str = Header(None),
):
    raise HTTPException(
        status_code=410,
        detail="Synchronous analysis is disabled. Use /submit_analysis and the dedicated worker.",
    )

    if supabase is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured on server.")

    total_started = time.perf_counter()
    phases = {}

    phase_started = time.perf_counter()
    user = authenticated_user(supabase, authorization)
    session = require_owned_session(user, session_id)
    phases["auth_session_ms"] = round((time.perf_counter() - phase_started) * 1000, 1)
    authoritative_email = str(session.get("user_email") or user["email"]).strip().lower()
    intake_data = parse_intake_payload(intake_json=intake_json, questionnaire_json=questionnaire_json)
    try:
        capture_metadata = parse_capture_metadata(capture_metadata_json)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try_save_session_intake(session_id, intake_data)

    phase_started = time.perf_counter()
    img_bytes = await image.read()
    phases["image_read_ms"] = round((time.perf_counter() - phase_started) * 1000, 1)
    phases["image_bytes"] = len(img_bytes)
    if not img_bytes:
        raise HTTPException(status_code=422, detail="The uploaded image is empty.")

    try:
        result, session_update = run_yolo_analysis_from_bytes(
            img_bytes,
            test_type,
            capture_metadata=capture_metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    persistent_result = _without_ephemeral_vision_qa(result)
    row = build_screening_row(
        user_email=authoritative_email,
        user_id=session.get("user_id"),
        session_id=session_id,
        test_type=test_type,
        result=persistent_result,
        intake_data=intake_data,
    )

    try:
        supabase.table("screenings").insert(row).execute()
        result_json = _analysis_result_from_runtime(
            authoritative_email, session_id, test_type, result, intake_data
        )
    except Exception as exc:
        if not _is_duplicate_screening_error(exc):
            raise
        existing_screening = _find_existing_screening(session_id, test_type)
        if not existing_screening:
            raise
        result_json = _analysis_result_from_screening(existing_screening, intake_data)

    _update_session_score_best_effort(
        session_id,
        _session_score_update_for_test(test_type, result_json.get("score")),
        test_type=test_type,
    )
    return result_json


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def build_screening_row(user_email, user_id, session_id, test_type, result, intake_data):
    capture_fingerprint = ""
    if isinstance(intake_data, dict):
        capture_fingerprint = str(
            intake_data.get("_flexilab_capture_fingerprint") or ""
        ).strip().lower()
    idempotency_key = (
        f"{session_id}:{test_type}:{capture_fingerprint}"
        if capture_fingerprint
        else f"{session_id}:{test_type}"
    )
    row = {
        "user_email": user_email,
        "user_id": user_id,
        "session_id": session_id,
        "idempotency_key": idempotency_key,
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


def _is_transient_upstream_error(exc):
    """Return True for temporary HTTP/socket failures from Supabase/PostgREST."""
    if isinstance(exc, (
        httpx.ReadError,
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.PoolTimeout,
    )):
        return True
    text = str(exc or "").lower()
    return (
        "resource temporarily unavailable" in text
        or "errno 11" in text
        or "server disconnected" in text
        or "connection reset" in text
    )


def _execute_with_transient_retry(operation, *, label, attempts=3):
    """Retry a small Supabase operation without rerunning YOLO inference."""
    last_exc = None
    for attempt in range(1, max(1, int(attempts)) + 1):
        try:
            return operation()
        except Exception as exc:
            last_exc = exc
            if not _is_transient_upstream_error(exc) or attempt >= attempts:
                raise
            delay = 0.20 * attempt
            logger.warning(
                "transient_upstream_retry label=%s attempt=%s/%s delay_s=%.2f error=%s",
                label,
                attempt,
                attempts,
                delay,
                str(exc)[:240],
            )
            time.sleep(delay)
    raise last_exc


def _is_duplicate_screening_error(exc):
    text = str(exc or "").lower()
    return (
        "23505" in text
        or "duplicate key" in text
        or "screenings_idempotency_unique_idx" in text
    )


def _find_existing_screening(session_id, test_type):
    if supabase is None:
        return None
    response = _execute_with_transient_retry(
        lambda: (
            supabase.table("screenings")
            .select("*")
            .eq("session_id", session_id)
            .eq("test_type", test_type)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        ),
        label="find_existing_screening",
    )
    return response.data[0] if response.data else None


def _session_score_update_for_test(test_type, score):
    column_by_test = {
        "posture_side": "posture_score",
        "shoulder_right": "shoulder_right_score",
        "shoulder_left": "shoulder_left_score",
        "squat": "squat_score",
        "aslr_right": "aslr_right_score",
        "aslr_left": "aslr_left_score",
    }
    column = column_by_test.get(str(test_type or ""))
    return {column: float(score)} if column and score is not None else {}


def _analysis_result_from_screening(screening, fallback_intake=None):
    return {
        "user_email": screening.get("user_email"),
        "session_id": screening.get("session_id"),
        "test_type": screening.get("test_type"),
        "score": screening.get("score"),
        "confidence": screening.get("confidence"),
        "metrics": screening.get("metrics") or {},
        "thresholds": screening.get("thresholds"),
        "intake_json": screening.get("intake_json") or fallback_intake or {},
        "annotated_image_url": screening.get("annotated_image_url"),
    }


def _analysis_result_from_runtime(user_email, session_id, test_type, result, intake_data):
    return {
        "user_email": user_email,
        "session_id": session_id,
        "test_type": test_type,
        "score": result["score"],
        "confidence": result["confidence"],
        "metrics": result["metrics"],
        "thresholds": result.get("thresholds"),
        "intake_json": intake_data,
        "annotated_image_url": None,
    }


def _update_session_score_best_effort(session_id, session_update, *, job_id=None, test_type=None):
    if not session_update:
        return
    try:
        _execute_with_transient_retry(
            lambda: supabase.table("sessions").update(session_update).eq("id", session_id).execute(),
            label="update_session_score",
        )
    except Exception:
        # Session score columns are a cache. The authoritative result remains in
        # screenings, so a cache/schema mismatch must never invalidate a test.
        logger.exception(
            "analysis_session_score_update_failed job_id=%s session_id=%s test_type=%s fields=%s",
            job_id,
            session_id,
            test_type,
            sorted(session_update.keys()),
        )


def _complete_analysis_job(job_id, result_json, *, image_expires_at=None):
    payload = {
        "status": "completed",
        "completed_at": utc_now_iso(),
        "result_json": result_json,
        "error_message": None,
        "image_base64": None,
        "image_expires_at": image_expires_at,
    }
    _execute_with_transient_retry(
        lambda: supabase.table("analysis_jobs").update(payload).eq("id", job_id).execute(),
        label="complete_analysis_job",
    )


def _public_analysis_error(exc):
    text = str(exc or "")
    lowered = text.lower()
    if _is_duplicate_screening_error(exc):
        return "This test was already saved. Please wait while the saved result is recovered."
    if isinstance(exc, ValueError):
        return text
    if "column" in lowered or "schema cache" in lowered or "database" in lowered:
        return "The analysis was completed but could not be finalized. Please retry once."
    return "The analysis could not be completed. Please retake the photo and try again."


def decode_and_normalize_analysis_image(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")

    original_h, original_w = img.shape[:2]
    scale = min(1.0, ANALYSIS_MAX_EDGE / max(original_h, original_w))
    if scale < 1.0:
        img = cv2.resize(
            img,
            (max(1, int(round(original_w * scale))), max(1, int(round(original_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    normalized_h, normalized_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    quality = {
        "original_width": int(original_w),
        "original_height": int(original_h),
        "normalized_width": int(normalized_w),
        "normalized_height": int(normalized_h),
        "resize_scale": round(float(scale), 6),
        "brightness_mean": round(float(np.mean(gray)), 2),
        "blur_laplacian_variance": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
    }
    return img, quality


def run_yolo_analysis_from_bytes(img_bytes, test_type, capture_metadata=None):
    img, image_quality = decode_and_normalize_analysis_image(img_bytes)
    is_aslr = str(test_type).startswith("aslr")

    first_requested_imgsz = max(POSE_INFERENCE_IMGSZ, 960) if is_aslr else None
    aslr_single_rotated_pass = None

    if is_aslr:
        # V101.35.31 canonical parity rule:
        #   RIGHT -> original image -> clockwise RIGHT pipeline
        #   LEFT  -> horizontal image mirror BEFORE YOLO -> same clockwise RIGHT pipeline
        # The result is relabelled to LEFT only after the canonical analysis.
        left_mirror_applied = str(test_type) == "aslr_left"
        aslr_analysis_img = cv2.flip(img, 1) if left_mirror_applied else img
        rotation_code = cv2.ROTATE_90_CLOCKWISE
        rotation_name = "rotated_90_clockwise"
        map_rotated_pose_to_original = _map_rotated_cw_pose_to_original
        rotated_pose_image = cv2.rotate(aslr_analysis_img, rotation_code)
        first_prediction, first_threshold, first_imgsz = detect_aslr_pose_with_fallback(
            rotated_pose_image,
            inference_imgsz=first_requested_imgsz,
        )
        (
            first_boxes_rotated,
            first_areas,
            first_box_confidences,
            first_main_idx,
            first_xy_rotated,
            first_conf,
        ) = _pose_arrays(first_prediction)
        first_xy, first_boxes = map_rotated_pose_to_original(
            first_xy_rotated,
            first_boxes_rotated,
            aslr_analysis_img.shape,
        )
        first_areas = np.array([
            max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
            for box in first_boxes
        ], dtype=float)
        first_chain_quality, first_chain_candidates = _aslr_same_side_chain_quality(
            first_xy, first_conf
        )
        aslr_single_rotated_pass = {
            "name": f"{rotation_name}_full_image",
            "xy": first_xy,
            "conf": first_conf,
            "boxes": first_boxes,
            "box_confidences": first_box_confidences,
            "main_idx": first_main_idx,
            "threshold": first_threshold,
            "imgsz": first_imgsz,
            "chain_quality": first_chain_quality,
            "chain_candidates": first_chain_candidates,
            "pose_model_inference_count": 1,
            "detection_attempt_count": 1,
            "rotation_direction": "clockwise",
            "capture_protocol": "left_image_mirrored_to_head_left" if left_mirror_applied else "head_left",
            "horizontal_mirror_before_yolo": left_mirror_applied,
        }

        # V101.35.20: keep the fast one-call path when a coherent same-side
        # hip-knee-ankle chain is reliable. Only weak/incomplete detections get
        # one focused, higher-detail recovery pass.
        first_knee_uncertain = (
            float(first_chain_quality.get("knee_extension_deg", 0.0)) < 155.0
            or float(first_chain_quality.get("minimum_confidence", 0.0)) < 0.45
            or float(first_chain_quality.get("score", 0.0)) < 0.72
        )
        if (not bool(first_chain_quality.get("valid_detection"))) or first_knee_uncertain:
            crop_img, crop_bounds = _expanded_person_crop(
                rotated_pose_image,
                first_boxes_rotated[first_main_idx],
                aslr=True,
            )
            rotated_area = float(rotated_pose_image.shape[0] * rotated_pose_image.shape[1])
            crop_is_useful = bool(
                crop_img is not None
                and crop_bounds is not None
                and float(crop_bounds["width"] * crop_bounds["height"]) < rotated_area * 0.97
            )
            if crop_is_useful:
                crop_prediction, crop_threshold, crop_imgsz = detect_aslr_pose_with_fallback(
                    crop_img,
                    inference_imgsz=1280,
                )
                (
                    crop_boxes_local,
                    _crop_areas,
                    crop_box_confidences,
                    crop_main_idx,
                    crop_xy_local,
                    crop_conf,
                ) = _pose_arrays(crop_prediction)
                crop_xy_rotated, crop_boxes_rotated = _map_crop_pose_to_full(
                    crop_xy_local, crop_boxes_local, crop_bounds
                )
                crop_xy, crop_boxes = map_rotated_pose_to_original(
                    crop_xy_rotated, crop_boxes_rotated, aslr_analysis_img.shape
                )
                crop_chain_quality, crop_chain_candidates = _aslr_same_side_chain_quality(
                    crop_xy, crop_conf
                )
                crop_pass = {
                    "name": f"{rotation_name}_focused_crop_recovery",
                    "xy": crop_xy,
                    "conf": crop_conf,
                    "boxes": crop_boxes,
                    "box_confidences": crop_box_confidences,
                    "main_idx": crop_main_idx,
                    "threshold": crop_threshold,
                    "imgsz": crop_imgsz,
                    "chain_quality": crop_chain_quality,
                    "chain_candidates": crop_chain_candidates,
                    "crop_bounds_rotated": crop_bounds,
                    "pose_model_inference_count": 2,
                    "detection_attempt_count": 2,
                    "rotation_direction": "clockwise",
                    "capture_protocol": "left_image_mirrored_to_head_left" if left_mirror_applied else "head_left",
                    "horizontal_mirror_before_yolo": left_mirror_applied,
                }
                crop_score = float(crop_chain_quality.get("score", 0.0))
                first_score = float(first_chain_quality.get("score", 0.0))
                crop_knee = float(crop_chain_quality.get("knee_extension_deg", 0.0))
                first_knee = float(first_chain_quality.get("knee_extension_deg", 0.0))
                crop_min_conf = float(crop_chain_quality.get("minimum_confidence", 0.0))
                first_min_conf = float(first_chain_quality.get("minimum_confidence", 0.0))
                crop_improves_uncertain_knee = (
                    bool(crop_chain_quality.get("valid_detection"))
                    and crop_knee >= first_knee + 3.0
                    and crop_min_conf >= first_min_conf - 0.05
                    and crop_score >= first_score - 0.015
                )
                if crop_score > first_score + 0.02 or crop_improves_uncertain_knee:
                    aslr_single_rotated_pass = crop_pass
    else:
        first_prediction, first_threshold, first_imgsz = detect_pose_with_fallback(
            img,
            test_type,
            inference_imgsz=first_requested_imgsz,
        )
        (
            first_boxes,
            first_areas,
            first_box_confidences,
            first_main_idx,
            first_xy,
            first_conf,
        ) = _pose_arrays(first_prediction)
    if is_aslr and len(first_boxes) > 1:
        main_area = max(float(first_areas[first_main_idx]), 1.0)
        significant_others = [
            index
            for index, area in enumerate(first_areas)
            if index != first_main_idx
            and float(area) >= main_area * 0.35
            and float(first_box_confidences[index]) >= 0.25
        ]
        if significant_others:
            raise ASLRQualityError(
                "multiple_people",
                "More than one person is visible. Retake the photo with only the person being assessed in the frame.",
                {"significant_other_person_count": len(significant_others)},
            )

    image_area = max(1.0, float(img.shape[0] * img.shape[1]))
    person_coverage = float(first_areas[first_main_idx]) / image_area
    first_relevant_conf = _relevant_pose_confidence(first_conf, test_type)
    if is_aslr:
        # No crop copy or second-pass preparation for ASLR.
        crop_img, crop_bounds = None, None
    else:
        crop_img, crop_bounds = _expanded_person_crop(
            img,
            first_boxes[first_main_idx],
            aslr=False,
        )

    # A body-focused pass is used only when the subject is small or lower-body
    # landmarks are weak. Portrait ASLR and desktop webcam frames are therefore
    # supported without imposing any source orientation.
    shoulder_first_quality = None
    if str(test_type).startswith("shoulder"):
        shoulder_first_quality, _ = _shoulder_chain_quality(first_xy, first_conf)

    should_use_crop_pass = bool(
        not is_aslr
        and crop_img is not None
        and crop_bounds is not None
        and (crop_bounds["width"] * crop_bounds["height"]) < image_area * 0.97
        and (
            person_coverage < 0.24
            or (shoulder_first_quality is not None and bool(shoulder_first_quality.get("uncertain")))
        )
    )

    final_boxes = first_boxes
    final_box_confidences = first_box_confidences
    final_main_idx = first_main_idx
    final_xy = first_xy
    final_conf = first_conf
    detection_threshold = first_threshold
    inference_imgsz = first_imgsz
    analysis_pass = {
        "mode": "full_image",
        "adaptive_crop_used": False,
        "source_orientation_required": False,
        "person_coverage": round(person_coverage, 4),
        "relevant_keypoint_mean_confidence_before_crop": round(first_relevant_conf, 4),
    }

    if should_use_crop_pass:
        crop_imgsz = 1280 if str(test_type).startswith("shoulder") else max(POSE_INFERENCE_IMGSZ, 768 if is_aslr else POSE_INFERENCE_IMGSZ)
        crop_prediction, crop_threshold, crop_imgsz = detect_pose_with_fallback(
            crop_img,
            test_type,
            inference_imgsz=crop_imgsz,
        )
        (
            crop_boxes,
            _crop_areas,
            crop_box_confidences,
            crop_main_idx,
            crop_xy,
            crop_conf,
        ) = _pose_arrays(crop_prediction)
        mapped_xy, mapped_boxes = _map_crop_pose_to_full(crop_xy, crop_boxes, crop_bounds)
        crop_relevant_conf = _relevant_pose_confidence(crop_conf, test_type)

        crop_shoulder_quality = None
        if str(test_type).startswith("shoulder"):
            crop_shoulder_quality, _ = _shoulder_chain_quality(mapped_xy, crop_conf)
        prefer_crop = crop_relevant_conf >= first_relevant_conf - 0.03
        if shoulder_first_quality is not None and crop_shoulder_quality is not None:
            prefer_crop = (
                bool(crop_shoulder_quality.get("valid_detection"))
                and float(crop_shoulder_quality.get("score", 0.0))
                >= float(shoulder_first_quality.get("score", 0.0)) + 0.015
            ) or (
                bool(shoulder_first_quality.get("uncertain"))
                and float(crop_shoulder_quality.get("score", 0.0))
                >= float(shoulder_first_quality.get("score", 0.0)) - 0.01
            )
        if prefer_crop:
            final_boxes = mapped_boxes
            final_box_confidences = crop_box_confidences
            final_main_idx = crop_main_idx
            final_xy = mapped_xy
            final_conf = crop_conf
            detection_threshold = crop_threshold
            inference_imgsz = crop_imgsz
            analysis_pass = {
                "mode": "adaptive_person_crop",
                "adaptive_crop_used": True,
                "source_orientation_required": False,
                "person_coverage": round(person_coverage, 4),
                "crop_bounds": crop_bounds,
                "relevant_keypoint_mean_confidence_before_crop": round(first_relevant_conf, 4),
                "relevant_keypoint_mean_confidence_after_crop": round(crop_relevant_conf, 4),
                "shoulder_chain_quality_before_crop": shoulder_first_quality,
                "shoulder_chain_quality_after_crop": crop_shoulder_quality,
            }

    aslr_precomputed_result = None
    aslr_precomputed_error = None
    if is_aslr:
        requested_side = "RIGHT" if test_type == "aslr_right" else "LEFT"

        # The private rotated pass supplies the pelvis, raised knee and true YOLO
        # ankle. After inverse mapping, the final ASLR reference is the original
        # image horizontal through the pelvis. Shoulder and floor-leg landmarks
        # are excluded from the final angle.
        body_reference_xy = np.array(final_xy, dtype=float, copy=True)
        body_reference_conf = np.array(final_conf, dtype=float, copy=True)

        # A true YOLO ankle is mandatory. Toe, shoe-contour and skin endpoints
        # are never permitted. Reuse the one already completed rotated pass;
        # no original-orientation ASLR inference and no second YOLO call occur.
        if aslr_single_rotated_pass is None:
            raise ASLRQualityError(
                "aslr_single_inference_unavailable",
                "The ASLR pose inference could not be completed. Please retake the photo.",
            )
        selected_pass = aslr_single_rotated_pass
        selected_pass_name = selected_pass["name"]
        try:
            aslr_precomputed_result = analyze_aslr(
                selected_pass["xy"],
                selected_pass["conf"],
                requested_side,
                aslr_analysis_img,
                body_xy=body_reference_xy,
                body_conf=body_reference_conf,
            )
        except ASLRQualityError as exc:
            aslr_precomputed_error = exc

        final_xy = selected_pass["xy"]
        final_conf = selected_pass["conf"]
        final_boxes = selected_pass["boxes"]
        final_box_confidences = selected_pass["box_confidences"]
        final_main_idx = selected_pass["main_idx"]
        detection_threshold = selected_pass["threshold"]
        inference_imgsz = selected_pass["imgsz"]

        analysis_pass = {
            "mode": "aslr_canonical_right_pipeline_image_mirror_before_yolo",
            "selected_pass": selected_pass_name,
            "rotation_direction": selected_pass.get("rotation_direction"),
            "capture_protocol": selected_pass.get("capture_protocol"),
            "source_orientation_required": True,
            "pose_passes": [selected_pass_name],
            "pose_pass_count": int(selected_pass.get("detection_attempt_count", 1)),
            "pose_model_inference_count": int(selected_pass.get("pose_model_inference_count", 1)),
            "detection_attempt_count": int(selected_pass.get("detection_attempt_count", 1)),
            "conditional_focused_crop_used": selected_pass_name.endswith("focused_crop_recovery"),
            "selected_chain_detection_quality": selected_pass.get("chain_quality"),
            "candidate_chain_detection_quality": selected_pass.get("chain_candidates"),
            "tracked_image_processing_enabled": False,
            "original_orientation_pose_inference_used": False,
            "rotated_pass_failure": None,
            "fallback_used": False,
            "visual_endpoint_allowed": False,
            "endpoint_policy": "raised_true_yolo_ankle_required_floor_leg_optional_validation_only",
            "chain_policy": "left_mirror_before_yolo_then_exact_right_chain_pipeline",
            "body_reference_policy": "original_image_horizontal_through_pelvis_always_primary_shoulders_and_floor_leg_excluded",
            "person_coverage": round(person_coverage, 4),
            "adaptive_crop_used": False,
        }

    xy = final_xy
    conf = final_conf

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
    elif test_type in {"aslr_right", "aslr_left"}:
        requested_side = "RIGHT" if test_type == "aslr_right" else "LEFT"
        exc = aslr_precomputed_error
        if exc is None:
            result = aslr_precomputed_result
            if result is None:
                try:
                    result = analyze_aslr(xy, conf, requested_side, aslr_analysis_img)
                except ASLRQualityError as runtime_exc:
                    exc = runtime_exc
        if exc is not None:
            rejection_metrics = {
                "requested_side": requested_side,
                "capture_rejection": {
                    "code": exc.code,
                    "message": str(exc),
                    "details": exc.details,
                },
                "candidate_limbs": exc.details.get("candidates", [])
                if isinstance(exc.details, dict)
                else [],
                "diagnostic_flags": ["capture_rejected_before_scoring"],
                "analysis_pass": analysis_pass,
                "image_quality_diagnostics": image_quality,
                "capture_metadata": capture_metadata or {},
                "model_runtime": {
                    "model": ASLR_POSE_MODEL_NAME,
                    "model_role": "dedicated_aslr_pose",
                    "inference_imgsz": inference_imgsz,
                    "analysis_max_edge": ANALYSIS_MAX_EDGE,
                    "reload_count": ASLR_POSE_MODEL_RELOAD_COUNT,
                },
            }
            rejected_result = {
                "score": 0.0,
                "confidence": 0.0,
                "metrics": rejection_metrics,
                "thresholds": {},
            }
            if _vision_qa_enabled_for_test(test_type):
                rejected_result["metrics"]["vision_qa"] = build_vision_qa_payload(
                    img,
                    xy,
                    conf,
                    final_boxes[final_main_idx],
                    rejected_result,
                    test_type,
                    analysis_pass=analysis_pass,
                )
            raise AnalysisWithDiagnosticsError(str(exc), rejected_result) from exc
        session_update = {
            "aslr_right_score" if requested_side == "RIGHT" else "aslr_left_score": result["score"]
        }
    else:
        raise ValueError("Invalid test_type")

    _attach_measurement_points(result, test_type, xy)
    _attach_screening_soft_warnings(result, test_type)
    result.setdefault("metrics", {})["detection_confidence_threshold"] = detection_threshold
    result["metrics"]["person_detection"] = {
        "person_count": int(len(first_boxes)),
        "selected_index": int(final_main_idx),
        "selected_box_confidence": round(float(final_box_confidences[final_main_idx]), 4),
    }
    result["metrics"]["image_quality_diagnostics"] = image_quality
    result["metrics"]["capture_metadata"] = capture_metadata or {}
    result["metrics"]["analysis_pass"] = analysis_pass
    result["metrics"]["model_runtime"] = {
        "model": ASLR_POSE_MODEL_NAME if is_aslr else POSE_MODEL_NAME,
        "model_role": "dedicated_aslr_pose" if is_aslr else "general_pose",
        "inference_imgsz": inference_imgsz,
        "analysis_max_edge": ANALYSIS_MAX_EDGE,
        "reload_count": ASLR_POSE_MODEL_RELOAD_COUNT if is_aslr else POSE_MODEL_RELOAD_COUNT,
    }

    if _vision_qa_enabled_for_test(test_type):
        result["metrics"]["vision_qa"] = build_vision_qa_payload(
            img,
            final_xy,
            final_conf,
            final_boxes[final_main_idx],
            result,
            test_type,
            analysis_pass=analysis_pass,
        )
        result["metrics"]["vision_qa_mode"] = VISION_QA_MODE
    else:
        result["metrics"]["vision_qa_mode"] = "off"

    return result, session_update


def process_analysis_job(job_id: str):
    if PROCESS_ROLE != "worker":
        logger.error(
            "analysis_execution_blocked process_role=%s job_id=%s",
            PROCESS_ROLE,
            job_id,
        )
        return
    if supabase is None:
        return

    _set_analysis_runtime_state(job_id, "starting")
    total_started = time.perf_counter()
    phases = {}
    job = None
    try:
        phase_started = time.perf_counter()
        resp = _execute_with_transient_retry(
            lambda: (
                supabase.table("analysis_jobs")
                .select("*")
                .eq("id", job_id)
                .limit(1)
                .execute()
            ),
            label="analysis_job_fetch",
        )

        phases["job_fetch_ms"] = round((time.perf_counter() - phase_started) * 1000, 1)
        if not resp.data:
            return

        job = resp.data[0]
        job_status_value = str(job.get("status") or "").lower()

        if job_status_value == "completed":
            return

        if job_status_value == "processing":
            started_raw = job.get("started_at")
            try:
                started_at = datetime.fromisoformat(
                    str(started_raw).replace("Z", "+00:00")
                )
            except Exception:
                started_at = None

            if (
                started_at is not None
                and (datetime.now(timezone.utc) - started_at).total_seconds() < 120
            ):
                return

        phase_started = time.perf_counter()
        claim = _execute_with_transient_retry(
            lambda: (
                supabase.table("analysis_jobs")
                .update({
                    "status": "processing",
                    "started_at": utc_now_iso(),
                    "error_message": None,
                })
                .eq("id", job_id)
                .eq("status", "queued")
                .execute()
            ),
            label="analysis_job_claim",
        )
        phases["job_claim_ms"] = round((time.perf_counter() - phase_started) * 1000, 1)
        if not claim.data:
            # Another web process or worker claimed this job first.
            return

        _set_analysis_runtime_state(job_id, "processing")
        test_type = job.get("test_type")
        user_email = job.get("user_email")
        session_id = job.get("session_id")
        intake_data, capture_metadata = _split_job_intake_and_capture_metadata(
            job.get("intake_json")
        )

        # Recovery path: a previous attempt may have inserted the authoritative
        # screening row and then failed while updating a non-authoritative cache.
        existing_screening = _find_existing_screening(session_id, test_type)
        if existing_screening:
            result_json = _analysis_result_from_screening(existing_screening, intake_data)
            _update_session_score_best_effort(
                session_id,
                _session_score_update_for_test(test_type, result_json.get("score")),
                job_id=job_id,
                test_type=test_type,
            )
            _complete_analysis_job(job_id, result_json, image_expires_at=None)
            _set_analysis_runtime_state(job_id, "completed", result=result_json)
            image_path = str(job.get("image_path") or "").strip()
            if image_path:
                _clear_analysis_image_reference(job_id, image_path)
            return

        phase_started = time.perf_counter()
        image_path = str(job.get("image_path") or "").strip()
        img_b64 = job.get("image_base64")
        if image_path:
            try:
                img_bytes = supabase.storage.from_(ANALYSIS_STORAGE_BUCKET).download(image_path)
            except Exception as exc:
                if not img_b64:
                    raise ValueError(f"Unable to load queued image: {exc}")
                img_bytes = base64.b64decode(img_b64)
        elif img_b64:
            img_bytes = base64.b64decode(img_b64)
        else:
            raise ValueError("Missing queued image")
        phases["image_load_ms"] = round((time.perf_counter() - phase_started) * 1000, 1)
        phases["image_bytes"] = len(img_bytes)

        phase_started = time.perf_counter()
        result, session_update = run_yolo_analysis_from_bytes(
            img_bytes,
            test_type,
            capture_metadata=capture_metadata,
        )
        phases["analysis_engine_ms"] = round((time.perf_counter() - phase_started) * 1000, 1)

        persistent_result = _without_ephemeral_vision_qa(result)
        screening_row = build_screening_row(
            user_email=user_email,
            user_id=job.get("user_id"),
            session_id=session_id,
            test_type=test_type,
            result=persistent_result,
            intake_data=intake_data
        )

        phase_started = time.perf_counter()
        try:
            _execute_with_transient_retry(
                lambda: supabase.table("screenings").insert(screening_row).execute(),
                label="screening_insert",
            )
            result_json = _analysis_result_from_runtime(
                user_email, session_id, test_type, result, intake_data
            )
        except Exception as exc:
            if not _is_duplicate_screening_error(exc):
                raise
            existing_screening = _find_existing_screening(session_id, test_type)
            if not existing_screening:
                raise
            result_json = _analysis_result_from_screening(existing_screening, intake_data)
        phases["screening_save_ms"] = round((time.perf_counter() - phase_started) * 1000, 1)

        # Failure to update denormalized session score columns must not convert a
        # successfully saved screening into a failed job.
        _update_session_score_best_effort(
            session_id,
            _session_score_update_for_test(test_type, result_json.get("score")),
            job_id=job_id,
            test_type=test_type,
        )

        keep_diagnostic_image = bool(image_path and DIAGNOSTIC_RETENTION_HOURS > 0)
        diagnostic_expiry = (
            datetime.now(timezone.utc) + timedelta(hours=DIAGNOSTIC_RETENTION_HOURS)
        ).isoformat() if keep_diagnostic_image else None

        phase_started = time.perf_counter()
        _complete_analysis_job(
            job_id,
            result_json,
            image_expires_at=diagnostic_expiry,
        )
        _set_analysis_runtime_state(job_id, "completed", result=result_json)
        phases["job_complete_ms"] = round((time.perf_counter() - phase_started) * 1000, 1)
        phases["total_ms"] = round((time.perf_counter() - total_started) * 1000, 1)
        logger.info(
            "analysis_perf job_id=%s session_id=%s test_type=%s phases=%s",
            job_id,
            session_id,
            test_type,
            json.dumps(phases, sort_keys=True),
        )

        if image_path and not keep_diagnostic_image:
            _clear_analysis_image_reference(job_id, image_path)

    except Exception as exc:
        public_error = _public_analysis_error(exc)
        runtime_result = exc.diagnostic_result if isinstance(exc, AnalysisWithDiagnosticsError) else None
        _set_analysis_runtime_state(job_id, "failed", result=runtime_result, error_message=public_error)
        logger.exception(
            "analysis_job_failed job_id=%s session_id=%s test_type=%s",
            job_id,
            (job or {}).get("session_id"),
            (job or {}).get("test_type"),
        )
        try:
            failure_update = {
                "status": "failed",
                "completed_at": utc_now_iso(),
                "error_message": public_error,
            }
            if isinstance(exc, AnalysisWithDiagnosticsError):
                failure_update["result_json"] = exc.diagnostic_result
            _execute_with_transient_retry(
                lambda: supabase.table("analysis_jobs").update(failure_update).eq("id", job_id).execute(),
                label="analysis_job_failure_update",
            )
        except Exception:
            logger.exception("analysis_job_failure_status_update_failed job_id=%s", job_id)
        failed_image_path = str((job or {}).get("image_path") or "").strip()
        if failed_image_path and DIAGNOSTIC_RETENTION_HOURS <= 0:
            _clear_analysis_image_reference(job_id, failed_image_path)


@app.post("/submit_analysis")
async def submit_analysis(
    image: UploadFile = File(...),
    user_email: str = Form(...),
    test_type: str = Form(...),
    session_id: str = Form(...),
    intake_json: str = Form(None),
    questionnaire_json: str = Form(None),
    capture_metadata_json: str = Form(None),
    authorization: str = Header(None),
):
    total_started = time.perf_counter()
    phases = {}

    if supabase is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured on server.")

    user = authenticated_user(supabase, authorization)
    session = require_owned_session(user, session_id)
    authoritative_email = str(session.get("user_email") or user_email or "").strip().lower()

    intake_data = parse_intake_payload(
        intake_json=intake_json,
        questionnaire_json=questionnaire_json,
    )
    if not isinstance(intake_data, dict):
        intake_data = {}
    try:
        capture_metadata = parse_capture_metadata(capture_metadata_json)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Read and fingerprint the actual upload before any reuse decision. Reusing
    # by session_id + test_type alone can return a measurement produced from a
    # different photo, which is unacceptable for assessment integrity.
    phase_started = time.perf_counter()
    img_bytes = await image.read()
    phases["image_read_ms"] = round((time.perf_counter() - phase_started) * 1000, 1)
    phases["image_bytes"] = len(img_bytes)
    if not img_bytes:
        raise HTTPException(status_code=422, detail="The uploaded image is empty.")
    if len(img_bytes) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="The uploaded image is too large. Maximum size is 12 MB.")

    image_fingerprint = hashlib.sha256(img_bytes).hexdigest()
    intake_data["_flexilab_capture_fingerprint"] = image_fingerprint
    try_save_session_intake(session_id, intake_data)
    job_idempotency_key = f"{session_id}:{test_type}:{image_fingerprint}"

    existing_jobs = _execute_with_transient_retry(
        lambda: (
            supabase.table("analysis_jobs")
            .select("id,status,created_at,result_json,error_message,intake_json,idempotency_key")
            .eq("session_id", session_id)
            .eq("test_type", test_type)
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        ),
        label="analysis_existing_jobs_fetch",
    )
    matching_jobs = []
    for existing in existing_jobs.data or []:
        existing_intake = existing.get("intake_json")
        existing_fingerprint = ""
        if isinstance(existing_intake, dict):
            existing_fingerprint = str(
                existing_intake.get("_flexilab_capture_fingerprint") or ""
            ).strip().lower()
        if not existing_fingerprint:
            key = str(existing.get("idempotency_key") or "")
            prefix = f"{session_id}:{test_type}:"
            if key.startswith(prefix):
                existing_fingerprint = key[len(prefix):].strip().lower()
        if existing_fingerprint == image_fingerprint:
            matching_jobs.append(existing)

    for existing in matching_jobs:
        existing_status = str(existing.get("status") or "").lower()
        if existing_status in {"queued", "processing", "completed"}:
            response = {
                "job_id": existing.get("id"),
                "status": existing_status,
                "reused": True,
                "reuse_reason": "exact_image_match",
                "image_fingerprint": image_fingerprint,
            }
            if existing_status == "completed" and existing.get("result_json"):
                response["result"] = existing.get("result_json")
            phases["total_ms"] = round((time.perf_counter() - total_started) * 1000, 1)
            logger.info(
                "analysis_submit_perf session_id=%s test_type=%s reused=true fingerprint=%s phases=%s",
                session_id,
                test_type,
                image_fingerprint[:12],
                json.dumps(phases, sort_keys=True),
            )
            return response

    # Recover only a failed job for the exact same image. A screening generated
    # from another image in the same session/test must never be substituted.
    if matching_jobs:
        existing_screening = _find_existing_screening(session_id, test_type)
        screening_fingerprint = ""
        if existing_screening and isinstance(existing_screening.get("intake_json"), dict):
            screening_fingerprint = str(
                existing_screening["intake_json"].get("_flexilab_capture_fingerprint") or ""
            ).strip().lower()
        if existing_screening and screening_fingerprint == image_fingerprint:
            recovered_job_id = matching_jobs[0].get("id")
            result_json = _analysis_result_from_screening(existing_screening, intake_data)
            _update_session_score_best_effort(
                session_id,
                _session_score_update_for_test(test_type, result_json.get("score")),
                job_id=recovered_job_id,
                test_type=test_type,
            )
            _complete_analysis_job(recovered_job_id, result_json, image_expires_at=None)
            return {
                "job_id": recovered_job_id,
                "status": "completed",
                "reused": True,
                "recovered": True,
                "reuse_reason": "exact_image_match_recovery",
                "image_fingerprint": image_fingerprint,
            }

    image_path = f"{session_id}/{test_type}/{utc_now_iso().replace(':', '-')}-{image_fingerprint[:12]}.jpg"
    image_base64_fallback = None
    phase_started = time.perf_counter()
    try:
        supabase.storage.from_(ANALYSIS_STORAGE_BUCKET).upload(
            image_path,
            img_bytes,
            {
                "content-type": image.content_type or "image/jpeg",
                "upsert": "false",
            },
        )
    except Exception as exc:
        logger.exception(
            "analysis_image_upload_failed session_id=%s test_type=%s path=%s",
            session_id, test_type, image_path,
        )
        raise HTTPException(
            status_code=503,
            detail="The screening image could not be queued securely. Please retry.",
        ) from exc
    phases["storage_upload_ms"] = round((time.perf_counter() - phase_started) * 1000, 1)

    job_intake_data = dict(intake_data)
    if capture_metadata:
        job_intake_data["_flexilab_capture_metadata"] = capture_metadata

    job = {
        "session_id": session_id,
        "session_uuid": session_id,
        "user_id": session.get("user_id"),
        "idempotency_key": job_idempotency_key,
        "user_email": authoritative_email,
        "test_type": test_type,
        "status": "queued",
        "image_path": image_path,
        "image_base64": image_base64_fallback,
        "image_expires_at": (
            datetime.now(timezone.utc)
            + timedelta(minutes=ANALYSIS_IMAGE_TTL_MINUTES)
        ).isoformat(),
        "intake_json": job_intake_data,
    }

    phase_started = time.perf_counter()
    try:
        resp = _execute_with_transient_retry(
            lambda: supabase.table("analysis_jobs").insert(job).execute(),
            label="analysis_job_insert",
        )
    except Exception as exc:
        # Concurrent double-submission of the exact same image can race between
        # the initial lookup and insert. The unique idempotency key safely
        # collapses that race back to the existing exact-image job.
        if not _is_duplicate_screening_error(exc):
            raise
        resp = _execute_with_transient_retry(
            lambda: (
                supabase.table("analysis_jobs")
                .select("id,status,result_json")
                .eq("idempotency_key", job_idempotency_key)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            ),
            label="analysis_duplicate_job_fetch",
        )
        if not resp.data:
            raise
        existing = resp.data[0]
        response = {
            "job_id": existing.get("id"),
            "status": str(existing.get("status") or "queued").lower(),
            "reused": True,
            "reuse_reason": "exact_image_concurrent_submission",
            "image_fingerprint": image_fingerprint,
        }
        if response["status"] == "completed" and existing.get("result_json"):
            response["result"] = existing.get("result_json")
        _delete_analysis_image(image_path, job_id=str(existing.get("id") or ""))
        return response

    phases["job_insert_ms"] = round((time.perf_counter() - phase_started) * 1000, 1)
    if not resp.data:
        _delete_analysis_image(image_path)
        raise HTTPException(status_code=500, detail="Unable to queue the photo analysis.")

    job_id = resp.data[0]["id"]
    phases["total_ms"] = round((time.perf_counter() - total_started) * 1000, 1)
    logger.info(
        "analysis_submit_perf session_id=%s test_type=%s reused=false fingerprint=%s phases=%s",
        session_id,
        test_type,
        image_fingerprint[:12],
        json.dumps(phases, sort_keys=True),
    )
    return {
        "job_id": job_id,
        "status": "queued",
        "reused": False,
        "reuse_reason": "fresh_analysis",
        "image_fingerprint": image_fingerprint,
    }


@app.get("/job_status/{job_id}")
def job_status(
    job_id: str,
    authorization: str = Header(None),
):
    if supabase is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured on server.")

    user = authenticated_user(supabase, authorization)
    resp = _execute_with_transient_retry(
        lambda: (
            supabase.table("analysis_jobs")
            .select("*")
            .eq("id", job_id)
            .limit(1)
            .execute()
        ),
        label="job_status_fetch",
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="Analysis job not found.")

    job = resp.data[0]
    require_owned_session(user, str(job.get("session_id") or ""))
    current_status = str(job.get("status") or "").lower()

    runtime_state = _get_analysis_runtime_state(job_id)
    if current_status in {"queued", "processing"} and runtime_state:
        runtime_status = str(runtime_state.get("status") or "").lower()
        if runtime_status in {"completed", "failed"}:
            return {
                "job_id": job.get("id"),
                "session_id": job.get("session_id"),
                "user_email": job.get("user_email"),
                "test_type": job.get("test_type"),
                "status": runtime_status,
                "result": _deliver_and_scrub_vision_qa(
                    job_id, runtime_state.get("result") or job.get("result_json")
                ),
                "error_message": runtime_state.get("error_message") or job.get("error_message"),
                "created_at": job.get("created_at"),
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at") or utc_now_iso(),
                "status_source": "runtime_terminal_failsafe",
            }

    # A processing job must not remain immortal. We never requeue it because
    # that can create competing YOLO runs. Once it is clearly stale, mark it
    # failed so the client stops polling and a deliberate retry can create a
    # fresh fingerprint-matched job.
    if current_status == "processing":
        started_raw = job.get("started_at")
        try:
            started_at = datetime.fromisoformat(str(started_raw).replace("Z", "+00:00"))
        except Exception:
            started_at = None
        if started_at is not None and (datetime.now(timezone.utc) - started_at).total_seconds() >= ANALYSIS_RUNTIME_TIMEOUT_SECONDS:
            stale_message = "Analysis timed out before completion. Please retry the photo once."
            _set_analysis_runtime_state(job_id, "failed", error_message=stale_message)
            try:
                _execute_with_transient_retry(
                    lambda: (
                        supabase.table("analysis_jobs")
                        .update({
                            "status": "failed",
                            "completed_at": utc_now_iso(),
                            "error_message": stale_message,
                        })
                        .eq("id", job_id)
                        .eq("status", "processing")
                        .execute()
                    ),
                    label="analysis_job_stale_fail",
                )
                current_status = "failed"
                job["status"] = "failed"
                job["completed_at"] = utc_now_iso()
                job["error_message"] = stale_message
            except Exception:
                logger.exception("analysis_job_stale_fail_update_failed job_id=%s", job_id)

    # Queue consumption is exclusively owned by the dedicated worker.
    # Polling is read-only and must never execute or schedule YOLO.

    return {
        "job_id": job.get("id"),
        "session_id": job.get("session_id"),
        "user_email": job.get("user_email"),
        "test_type": job.get("test_type"),
        "status": current_status,
        "result": _deliver_and_scrub_vision_qa(job_id, job.get("result_json")),
        "error_message": job.get("error_message"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
    }



@app.get("/report")
def report(
    session_id: str,
    lang: str = "fr",
    authorization: str = Header(None),
):
    """
    Bilingual report endpoint.
    Use /report?session_id=...&lang=fr or /report?session_id=...&lang=en
    Keeps FR keys for backward compatibility and adds EN equivalents.
    """
    lang = "en" if str(lang).lower().startswith("en") else "fr"

    if supabase is None:
        return {"error": "Supabase not configured"}

    user = authenticated_user(supabase, authorization)
    session = require_owned_session(user, session_id)

    scr_resp = supabase.table("screenings").select("*").eq("session_id", session_id).execute()
    screenings = scr_resp.data or []

    tests_found = [x.get("test_type") for x in screenings if x.get("test_type")]

    # Prefer session-level questionnaire when available, then fallback to screenings.
    intake_context = session.get("intake_json") or session.get("questionnaire_json")
    if not intake_context:
        for r in screenings:
            if r.get("intake_json"):
                intake_context = r.get("intake_json")
                break

    def txt(fr, en):
        return en if lang == "en" else fr

    LABELS = {
        "neck_angle": ("Angle cervical", "Cervical angle"),
        "thoracic_angle": ("Angle thoracique", "Thoracic angle"),
        "pelvic_proxy_angle": ("Alignement tronc-bassin", "Trunk-pelvis alignment"),
        "shoulder_right_flexion": ("Flexion épaule droite", "Right shoulder flexion"),
        "shoulder_left_flexion": ("Flexion épaule gauche", "Left shoulder flexion"),
        "squat_knee_angle": ("Angle du genou", "Knee angle"),
        "squat_trunk_lean": ("Inclinaison du tronc", "Trunk lean"),
        "aslr_right_angle": ("ASLR jambe droite", "Right ASLR"),
        "aslr_left_angle": ("ASLR jambe gauche", "Left ASLR"),
        "aslr_title": ("Élévation active de jambe", "Active Straight Leg Raise"),
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
            return txt("Dans la cible", "On target")
        if rating == "yellow":
            return txt("À développer", "Develop")
        if rating == "red":
            return txt("Priorité", "Priority")
        return txt("Mesure", "Measure")

    def insight_posture(label_fr, label_en, rating):
        if rating == "green":
            fr = f"{label_fr} dans la zone cible."
            en = f"{label_en} is within the target zone."
        elif rating == "yellow":
            fr = f"{label_fr} constitue un axe de progression modéré."
            en = f"{label_en} is a moderate development area."
        elif rating == "red":
            fr = f"{label_fr} constitue un axe prioritaire du programme."
            en = f"{label_en} is a priority focus for the program."
        else:
            fr = f"{label_fr} sera suivi lors du prochain screening."
            en = f"{label_en} will be tracked at the next screening."
        return fr, en, txt(fr, en)

    def insight_shoulder(rating):
        pairs = {
            "green": ("Élévation active fluide et dans la zone cible.", "Active elevation is fluid and within the target zone."),
            "yellow": ("L’élévation active peut gagner en amplitude et en contrôle.", "Active elevation can gain range and control."),
            "red": ("L’élévation active est un axe prioritaire pour la mobilité et le contrôle.", "Active elevation is a priority for mobility and control."),
        }
        fr, en = pairs.get(rating, ("Mesure à suivre au prochain screening.", "Measure to track at the next screening."))
        return fr, en, txt(fr, en)

    def insight_squat(label_fr, label_en, rating):
        if rating == "green":
            fr = f"{label_fr} dans la zone cible."
            en = f"{label_en} is within the target zone."
        elif rating == "yellow":
            fr = f"{label_fr} constitue un axe de progression."
            en = f"{label_en} is a development area."
        elif rating == "red":
            fr = f"{label_fr} constitue un axe prioritaire du programme."
            en = f"{label_en} is a priority focus for the program."
        else:
            fr = f"{label_fr} sera suivi lors du prochain screening."
            en = f"{label_en} will be tracked at the next screening."
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
                ins_fr, ins_en = "La mobilité active de hanche constitue un axe de progression.", "Active hip mobility is a development area."
            elif rating == "red":
                ins_fr, ins_en = "La mobilité active de hanche et le contrôle du bassin constituent un axe prioritaire.", "Active hip mobility and pelvic control are a priority focus."
            else:
                ins_fr, ins_en = "Mesure à suivre au prochain screening.", "Measure to track at the next screening."
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
            aslr_value = _safe_number(mr.get("aslr_angle"))
            thr = make_aslr_thresholds(aslr_value) if aslr_value is not None else thr_item(tr, "aslr_angle")
            items.append(item_obj("aslr_right_angle", mr.get("aslr_angle"), "°", (thr or {}).get("rating"), thr))

        if aslr_l:
            ml = aslr_l.get("metrics") or {}
            tl = aslr_l.get("thresholds") or {}
            aslr_value = _safe_number(ml.get("aslr_angle"))
            thr = make_aslr_thresholds(aslr_value) if aslr_value is not None else thr_item(tl, "aslr_angle")
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

    def add_candidate(sev, metric_id, title_fr, title_en, why_fr, why_en):
        candidates.append({
            "severity": sev,
            "metric_id": metric_id,
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
            add_candidate((na or {}).get("rating"), "neck_angle", "Alignement cervical", "Cervical alignment", "Cette mesure guidera le travail d’alignement cervical actif.", "This measure will guide active cervical-alignment work.")
        if (ta or {}).get("rating") in ["red", "yellow"]:
            add_candidate((ta or {}).get("rating"), "thoracic_angle", "Alignement thoracique", "Thoracic alignment", "Cette mesure guidera le travail de mobilité et de contrôle thoracique.", "This measure will guide thoracic mobility and control work.")
        if (pa or {}).get("rating") in ["red", "yellow"]:
            add_candidate((pa or {}).get("rating"), "pelvic_proxy_angle", "Alignement tronc-bassin", "Trunk-pelvis alignment", "Cette mesure guidera le travail de coordination entre le tronc et le bassin.", "This measure will guide trunk-pelvis coordination work.")

    if sh_r:
        thr = thr_item((sh_r.get("thresholds") or {}), "shoulder_flexion")
        if (thr or {}).get("rating") in ["red", "yellow"]:
            add_candidate((thr or {}).get("rating"), "shoulder_right_flexion", "Mobilité épaule droite", "Right shoulder mobility", "Cette mesure guidera la progression de l’élévation active de l’épaule droite.", "This measure will guide progression of active right-shoulder elevation.")

    if sh_l:
        thr = thr_item((sh_l.get("thresholds") or {}), "shoulder_flexion")
        if (thr or {}).get("rating") in ["red", "yellow"]:
            add_candidate((thr or {}).get("rating"), "shoulder_left_flexion", "Mobilité épaule gauche", "Left shoulder mobility", "Cette mesure guidera la progression de l’élévation active de l’épaule gauche.", "This measure will guide progression of active left-shoulder elevation.")

    if aslr_r:
        aslr_value = _safe_number((aslr_r.get("metrics") or {}).get("aslr_angle"))
        thr = make_aslr_thresholds(aslr_value) if aslr_value is not None else thr_item((aslr_r.get("thresholds") or {}), "aslr_angle")
        if (thr or {}).get("rating") in ["red", "yellow"]:
            add_candidate((thr or {}).get("rating"), "aslr_right_angle", "Mobilité ASLR droite", "Right ASLR mobility", "Cette mesure guidera la progression de la mobilité active de hanche droite.", "This measure will guide progression of active right-hip mobility.")

    if aslr_l:
        aslr_value = _safe_number((aslr_l.get("metrics") or {}).get("aslr_angle"))
        thr = make_aslr_thresholds(aslr_value) if aslr_value is not None else thr_item((aslr_l.get("thresholds") or {}), "aslr_angle")
        if (thr or {}).get("rating") in ["red", "yellow"]:
            add_candidate((thr or {}).get("rating"), "aslr_left_angle", "Mobilité ASLR gauche", "Left ASLR mobility", "Cette mesure guidera la progression de la mobilité active de hanche gauche.", "This measure will guide progression of active left-hip mobility.")

    if squat:
        ts = squat.get("thresholds") or {}
        tr = thr_item(ts, "trunk_lean")
        kn = thr_item(ts, "knee_angle")
        if (tr or {}).get("rating") in ["red", "yellow"]:
            add_candidate((tr or {}).get("rating"), "squat_trunk_lean", "Inclinaison du tronc en squat", "Trunk lean during squat", "Cette mesure guidera le travail de contrôle du tronc pendant le squat.", "This measure will guide trunk-control work during the squat.")
        if (kn or {}).get("rating") in ["red", "yellow"]:
            add_candidate((kn or {}).get("rating"), "squat_knee_angle", "Profondeur du squat", "Squat depth", "Cette mesure guidera la progression de la profondeur et de la coordination du squat.", "This measure will guide progression of squat depth and coordination.")

    sev_order = {"red": 0, "yellow": 1, "green": 2, "unknown": 3, None: 4}
    candidates.sort(key=lambda x: sev_order.get(x["severity"], 9))

    top_priorities = []
    for i, c in enumerate(candidates[:3], start=1):
        top_priorities.append({
            "id": f"priority_{i}",
            "metric_id": c.get("metric_id"),
            "title_fr": c["title_fr"],
            "title_en": c["title_en"],
            "title": c["title"],
            "severity": c["severity"],
            "why_fr": c["why_fr"],
            "why_en": c["why_en"],
            "why": c["why"],
        })

    report_payload = {
        "session_id": session_id,
        "language": lang,
        "user_email": session.get("user_email"),
        "created_at": session.get("created_at"),
        "intake_context": intake_context,
        "flexilab_score": flexilab_score,
        "risk_category": risk_category,
        "sections": sections,
        "top_priorities": top_priorities,
        "next_step_fr": "Refais le screening après 4 semaines dans des conditions comparables.",
        "next_step_en": "Repeat the screening after 4 weeks under comparable conditions.",
        "next_step": txt(
            "Refais le screening après 4 semaines dans des conditions comparables.",
            "Repeat the screening after 4 weeks under comparable conditions."
        ),
        "debug": {"tests_found": tests_found}
    }

    # Evidence-aware movement score. The compatibility function name remains
    # attach_score_v2 even though the replacement engine is V3.
    try:
        if attach_score_v2:
            report_payload = attach_score_v2(report_payload, lang=lang)
        else:
            report_payload["score_v2_error"] = "attach_score_v2 not loaded"
    except Exception as e:
        report_payload["score_v2_error"] = str(e)

    # Expert clinical and biomechanical interpretation. This also attaches
    # backend-defined symmetry thresholds to each asymmetry object.
    try:
        if attach_expert_report:
            report_payload = attach_expert_report(report_payload, lang=lang)
        else:
            report_payload["expert_report_error"] = "attach_expert_report not loaded"
    except Exception as e:
        report_payload["expert_report_error"] = str(e)

    return report_payload



def save_screening_history(user_email: str, session_id: str, result: dict):
    """
    Save exactly one history snapshot per screening session.

    Opening /program must never create a new screening event. If a snapshot for
    this session already exists, update that row without touching created_at.
    """
    try:
        if not supabase:
            return {"saved": False, "reason": "supabase_not_configured"}

        report_data = result.get("report", {}) if isinstance(result, dict) else {}
        score = report_data.get("flexilab_score")
        risk = report_data.get("risk_category", {})
        risk_level = risk.get("label") if isinstance(risk, dict) else None
        payload = {
            "user_email": str(user_email or report_data.get("user_email") or "anonymous").strip() or "anonymous",
            "session_id": str(session_id),
            "flexilab_score": score,
            "risk_level": risk_level,
            "result": result,
        }

        existing = (
            supabase.table("screening_history")
            .select("id")
            .eq("session_id", str(session_id))
            .order("created_at", desc=False)
            .limit(1)
            .execute()
        )
        rows = getattr(existing, "data", None) or []
        if rows:
            supabase.table("screening_history").update(payload).eq("id", rows[0]["id"]).execute()
            return {"saved": True, "action": "updated_existing"}

        supabase.table("screening_history").insert(payload).execute()
        return {"saved": True, "action": "inserted_once"}
    except Exception as e:
        return {"saved": False, "reason": str(e)}


def get_session_user_email(session_id: str, report_data: dict | None = None):
    """
    Best-effort user email/name lookup for history saving.
    """
    try:
        if isinstance(report_data, dict) and report_data.get("user_email"):
            return report_data.get("user_email")

        if supabase:
            res = supabase.table("sessions").select("user_email").eq("id", session_id).limit(1).execute()
            rows = getattr(res, "data", None) or []
            if rows and rows[0].get("user_email"):
                return rows[0].get("user_email")
    except Exception:
        pass

    return "anonymous"


def extract_pain_score_from_report(report_data: dict) -> float:
    """
    Best-effort pain extraction from intake_context.
    Pain changes clinical readiness, not movement score.
    """
    try:
        intake = report_data.get("intake_context") or {}
        if not isinstance(intake, dict):
            return 0.0
        for key in ["pain_score", "pain_level", "pain", "painIntensity"]:
            if key in intake and intake.get(key) is not None:
                return float(intake.get(key) or 0)
    except Exception:
        pass
    return 0.0


def extract_symmetry_status_from_report(report_data: dict) -> tuple[float, bool]:
    """
    Estimate symmetry index from shoulder and ASLR asymmetry objects when present.
    Returns (symmetry_index, asymmetry_significant).
    """
    asym_diffs = []
    try:
        for section in report_data.get("sections", []) or []:
            asym = section.get("asymmetry")
            if isinstance(asym, dict) and asym.get("value_deg") is not None:
                asym_diffs.append(float(asym.get("value_deg") or 0))
    except Exception:
        pass

    if not asym_diffs:
        return 100.0, False

    max_diff = max(asym_diffs)
    # Simple readable index for now. Can be replaced by a validated formula later.
    symmetry_index = max(0.0, min(100.0, 100.0 - max_diff * 2.0))
    return round(symmetry_index, 1), bool(max_diff >= 10.0)


def attach_movement_dna_to_report(report_data: dict, lang: str = "fr") -> dict:
    """
    Attach Movement DNA and clinical pattern recognition to the report.
    Non-blocking: if CKB fails, the report still works.
    """
    if not isinstance(report_data, dict):
        return report_data

    pain_score = extract_pain_score_from_report(report_data)
    symmetry_index, asymmetry_significant = extract_symmetry_status_from_report(report_data)

    if generate_movement_dna and MOVEMENT_PATTERNS:
        try:
            movement_dna = generate_movement_dna(
                {"report": report_data},
                MOVEMENT_PATTERNS,
                language=lang,
                pain_score=pain_score,
                symmetry_index=symmetry_index,
                asymmetry_significant=asymmetry_significant,
            )
            report_data["movement_dna"] = movement_dna
            report_data["clinical_patterns"] = movement_dna.get("matched_patterns", [])
            report_data["clinical_priority"] = movement_dna.get("clinical_priority")
        except Exception as e:
            report_data["movement_dna_error"] = str(e)
    else:
        report_data["movement_dna_error"] = "CKB engine or movement_patterns not loaded"

    return report_data



# -----------------------------------------------------------------------------
# V45 frontend-contract i18n normalizer
# Ensures /program returns selected-language strings for all visible program fields.
# This is dictionary/string cleanup only; it does not run AI and has negligible CPU cost.
# -----------------------------------------------------------------------------
def _v45_lang(lang: str) -> str:
    return "en" if str(lang).lower().startswith("en") else "fr"

def _v45_walk_program_i18n(program_data: dict, lang: str = "fr") -> dict:
    lang = _v45_lang(lang)
    if not isinstance(program_data, dict):
        return program_data
    program_data["language"] = lang
    for week in program_data.get("weeks", []) or []:
        for session in week.get("sessions", []) or week.get("days", []) or []:
            for e in session.get("exercises", []) or []:
                if not isinstance(e, dict):
                    continue
                # Prefer backend-engine bilingual fields. These are filled by the V45 engine patch.
                e["name"] = e.get(f"name_{lang}") or e.get("name") or e.get("id") or e.get("exercise_id") or ""
                e["target"] = e.get(f"target_{lang}") or e.get("target") or ""
                e["equipment"] = e.get(f"equipment_{lang}") or e.get(f"material_{lang}") or e.get("equipment") or e.get("material") or ""
                e["material"] = e.get(f"material_{lang}") or e.get(f"equipment_{lang}") or e.get("material") or e.get("equipment") or ""
                e["coaching_cues"] = e.get(f"coaching_cues_{lang}") or e.get(f"tips_{lang}") or e.get("coaching_cues") or e.get("tips") or ""
                e["tips"] = e.get(f"tips_{lang}") or e.get(f"coaching_cues_{lang}") or e.get("tips") or e.get("coaching_cues") or ""
                e["clinical_rationale"] = e.get(f"clinical_rationale_{lang}") or e.get("clinical_rationale") or ""
                e["why_in_this_program"] = e.get(f"why_in_this_program_{lang}") or e.get("why_in_this_program") or e.get("clinical_rationale") or ""
                e["reps_time"] = e.get(f"reps_time_{lang}") or e.get("reps_time") or ""
                e["tempo"] = e.get(f"tempo_{lang}") or e.get("tempo") or ""
                e["rest"] = e.get(f"rest_{lang}") or e.get("rest") or ""
    return program_data

def normalize_clinical_program_for_frontend(program_data: dict, report_data: dict, lang: str = "fr") -> dict:
    """
    Add a few legacy-compatible fields so the current frontend can read the new engine output.
    """
    if not isinstance(program_data, dict):
        return {
            "engine_version": "FlexiLab Clinical Prescription Engine unavailable",
            "weeks": [],
            "error": "program_data_not_dict",
        }

    priorities = program_data.get("main_priorities", []) or []
    movement_dna = report_data.get("movement_dna", {}) if isinstance(report_data, dict) else {}

    headline_fr = "Programme correctif basé sur le Movement DNA."
    headline_en = "Corrective program based on Movement DNA."
    if movement_dna.get("primary_profile"):
        headline_fr = f"Profil principal : {movement_dna.get('primary_profile')}."
        headline_en = f"Primary profile: {movement_dna.get('primary_profile')}."

    program_data.setdefault("report_ready_summary", {
        "headline": headline_en if lang == "en" else headline_fr,
        "total_sessions": sum(len(w.get("sessions", [])) for w in program_data.get("weeks", []) or []),
        "average_session_duration_minutes": 20,
        "next_action": "Follow the 4-week corrective plan, then repeat the same screening." if lang == "en" else "Suivre le plan correctif 4 semaines, puis refaire le même screening.",
        "top_systems": [
            {
                "system": p.get("id"),
                "label": p.get("label_en") if lang == "en" else p.get("label"),
                "priority_score": round(100 - float(p.get("score", 100)), 1)
            }
            for p in priorities[:3]
        ]
    })

    program_data.setdefault("root_cause_analysis", [
        {
            "fault": p.get("id"),
            "label": p.get("label_en") if lang == "en" else p.get("label"),
            "priority_score": round(100 - float(p.get("score", 100)), 1),
            "contributors": [],
            "evidence": [f"{p.get('id')} domain score: {p.get('score')}"]
        }
        for p in priorities[:3]
    ])

    program_data.setdefault("top_priority_systems", [
        {
            "system": p.get("id"),
            "system_label": p.get("label_en") if lang == "en" else p.get("label"),
            "priority_score": round(100 - float(p.get("score", 100)), 1),
            "pain_status": program_data.get("clinical_readiness", {}).get("label"),
            "pain_limited": program_data.get("clinical_readiness", {}).get("readiness") in ["limited", "medical_clearance_recommended"]
        }
        for p in priorities[:5]
    ])

    return _v45_walk_program_i18n(program_data, lang)



def persist_corrective_program(
    user_email: str,
    screening_session_id: str,
    language: str,
    program_data: dict,
    user_id: str | None = None,
):
    """Create or update the durable program generated from one screening."""
    if not supabase:
        return {"saved": False, "program_id": str(screening_session_id), "reason": "supabase_not_configured"}

    email = str(user_email or "anonymous").strip().lower() or "anonymous"
    session_uuid = str(screening_session_id)
    lang = "en" if str(language).lower().startswith("en") else "fr"

    try:
        existing = (
            supabase.table("corrective_programs")
            .select("id,program_version,status")
            .eq("screening_session_id", session_uuid)
            .limit(1)
            .execute()
        )
        rows = getattr(existing, "data", None) or []

        payload = {
            "user_id": user_id,
            "user_email": email,
            "screening_session_id": session_uuid,
            "language": lang,
            "status": "active",
            "program_data": program_data,
            "generated_at": utc_now_iso(),
        }

        if rows:
            program_id = str(rows[0]["id"])
            supabase.table("corrective_programs").update(payload).eq("id", program_id).execute()
        else:
            versions = supabase.table("corrective_programs").select("program_version")
            versions = versions.eq("user_id", user_id) if user_id else versions.eq("user_email", email)
            version_response = versions.order("program_version", desc=True).limit(1).execute()
            version_rows = getattr(version_response, "data", None) or []
            payload["program_version"] = int(version_rows[0].get("program_version") or 0) + 1 if version_rows else 1
            inserted = supabase.table("corrective_programs").insert(payload).execute()
            inserted_rows = getattr(inserted, "data", None) or []
            if not inserted_rows:
                raise RuntimeError("corrective_program_insert_returned_no_row")
            program_id = str(inserted_rows[0]["id"])

        try:
            supersede = supabase.table("corrective_programs").update({"status": "superseded"})
            supersede = supersede.eq("user_id", user_id) if user_id else supersede.eq("user_email", email)
            supersede.eq("status", "active").neq("id", program_id).execute()
        except Exception:
            pass

        return {"saved": True, "program_id": program_id}
    except Exception as exc:
        return {"saved": False, "program_id": str(screening_session_id), "reason": str(exc)}


def resolve_corrective_program(
    program_id: str,
    user_id: str | None = None,
    user_email: str | None = None,
):
    """Resolve a durable program UUID or source screening UUID by owner UUID."""
    if not supabase or not program_id:
        return None

    pid = str(program_id)

    def owned_query():
        query = supabase.table("corrective_programs").select("*")
        if user_id:
            query = query.eq("user_id", user_id)
        elif user_email:
            query = query.eq("user_email", str(user_email).strip().lower())
        return query

    try:
        by_id = owned_query().eq("id", pid).limit(1).execute()
        rows = getattr(by_id, "data", None) or []
        if rows:
            return rows[0]
    except Exception:
        pass

    try:
        by_screening = owned_query().eq("screening_session_id", pid).limit(1).execute()
        rows = getattr(by_screening, "data", None) or []
        return rows[0] if rows else None
    except Exception:
        return None



def _program_for_screening(user_id: str, screening_session_id: str):
    if not supabase or not user_id or not screening_session_id:
        return None
    response = (
        supabase.table("corrective_programs")
        .select(
            "id,user_id,user_email,screening_session_id,program_version,language,"
            "status,program_data,generated_at,completed_at,created_at"
        )
        .eq("user_id", user_id)
        .eq("screening_session_id", str(screening_session_id))
        .order("generated_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = getattr(response, "data", None) or []
    return rows[0] if rows else None


def _program_progress_rows(program_id: str):
    if not supabase or not program_id:
        return []
    response = (
        supabase.table("program_session_progress")
        .select(
            "id,program_id,user_email,week_number,day_number,status,"
            "started_at,completed_at,updated_at,completion_data"
        )
        .eq("program_id", str(program_id))
        .order("week_number")
        .order("day_number")
        .execute()
    )
    return getattr(response, "data", None) or []


def _program_total_sessions(program_row: dict) -> int:
    data = program_row.get("program_data") if isinstance(program_row, dict) else {}
    program_data = data.get("program") if isinstance(data, dict) and isinstance(data.get("program"), dict) else data
    if not isinstance(program_data, dict):
        return 12
    return sum(
        len(week.get("sessions") or [])
        for week in (program_data.get("weeks") or [])
        if isinstance(week, dict)
    ) or 12


def _program_completion_summary(program_row: dict, progress_rows=None) -> dict:
    rows = progress_rows if progress_rows is not None else _program_progress_rows(program_row.get("id"))
    completed_count = len([row for row in rows if row.get("status") == "completed"])
    total_sessions = _program_total_sessions(program_row)
    stored_completed = str(program_row.get("status") or "").lower() == "completed"
    is_completed = stored_completed or completed_count >= total_sessions
    return {
        "is_completed": is_completed,
        "completed_sessions": completed_count,
        "total_sessions": total_sessions,
        "remaining_sessions": max(0, total_sessions - completed_count),
    }


def _stored_program_response(program_row: dict, lang: str = "en") -> dict:
    """Rebuild the public /program contract from a durable stored program."""
    language = "en" if str(lang).lower().startswith("en") else "fr"
    program_data = copy.deepcopy(program_row.get("program_data") or {})
    if isinstance(program_data, dict):
        program_data["program_id"] = str(program_row.get("id"))
        program_data["generated_from_screening_id"] = str(program_row.get("screening_session_id"))
        program_data.setdefault("generated_at", program_row.get("generated_at") or program_row.get("created_at"))
        program_data = _v45_walk_program_i18n(program_data, language)
    return {
        "session_id": str(program_row.get("screening_session_id")),
        "program_id": str(program_row.get("id")),
        "generated_from_screening_id": str(program_row.get("screening_session_id")),
        "program_generated_at": program_row.get("generated_at") or program_row.get("created_at"),
        "language": language,
        "program": program_data,
        "prescription": program_data,
        "resource_load_errors": RESOURCE_LOAD_ERRORS,
        "storage_status": str(program_row.get("status") or "active"),
        "api_contract_note": {
            "program_is_canonical": True,
            "prescription_is_legacy_alias": True,
            "stored_program_reused": True,
        },
    }


def _parse_datetime(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _session_created_at(session_id: str):
    if not supabase or not session_id:
        return None
    response = (
        supabase.table("sessions")
        .select("id,created_at,status")
        .eq("id", str(session_id))
        .limit(1)
        .execute()
    )
    rows = getattr(response, "data", None) or []
    return _parse_datetime(rows[0].get("created_at")) if rows else None


def _assessment_is_newer(latest_session: dict, program_row: dict | None) -> bool:
    if not latest_session:
        return False
    if not program_row:
        return True
    if str(latest_session.get("id")) == str(program_row.get("screening_session_id")):
        return False
    latest_created = _parse_datetime(latest_session.get("created_at"))
    source_created = _session_created_at(program_row.get("screening_session_id"))
    if latest_created and source_created:
        return latest_created > source_created
    # The latest session query is authoritative and ordered newest first. If its
    # ID differs from the program source, treat it as newer when timestamps are
    # unavailable on legacy rows.
    return True


def _current_program_row(user_id: str):
    """Return one current row without downloading the user's full history."""
    if not supabase or not user_id:
        return None

    fields = (
        "id,user_id,user_email,screening_session_id,program_version,language,"
        "status,program_data,generated_at,completed_at,created_at"
    )
    active_response = (
        supabase.table("corrective_programs")
        .select(fields)
        .eq("user_id", user_id)
        .eq("status", "active")
        .order("generated_at", desc=True)
        .limit(1)
        .execute()
    )
    active_rows = getattr(active_response, "data", None) or []
    if active_rows:
        return active_rows[0]

    latest_response = (
        supabase.table("corrective_programs")
        .select(fields)
        .eq("user_id", user_id)
        .order("generated_at", desc=True)
        .limit(1)
        .execute()
    )
    latest_rows = getattr(latest_response, "data", None) or []
    return latest_rows[0] if latest_rows else None


def _program_generation_state(
    user_id: str,
    latest_session: dict,
    current_row: dict | None,
    progress_rows=None,
) -> dict:
    completion = _program_completion_summary(current_row, progress_rows) if current_row else {
        "is_completed": False,
        "completed_sessions": 0,
        "total_sessions": 0,
        "remaining_sessions": 0,
    }
    newer = _assessment_is_newer(latest_session, current_row)
    current_status = str(current_row.get("status") or "") if current_row else None
    can_generate = bool(latest_session and newer and current_row and not completion["is_completed"])
    return {
        "latest_assessment_id": latest_session.get("id") if latest_session else None,
        "current_program_id": current_row.get("id") if current_row else None,
        "current_program_status": current_status,
        "current_program_source_assessment_id": current_row.get("screening_session_id") if current_row else None,
        "has_newer_completed_assessment": newer,
        "can_generate_from_latest": can_generate,
        "automatic_generation_eligible": bool(latest_session and newer and (current_row is None or completion["is_completed"])),
        **completion,
    }


def _latest_owned_session_for_user(user):
    completed = (
        supabase.table("sessions")
        .select("id,status,created_at,composite_score")
        .eq("user_id", user["id"])
        .eq("status", "completed")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return completed.data[0] if completed.data else None


def _compact_program_summary(program_data, progress_rows):
    program = (program_data or {}).get("program") if isinstance(program_data, dict) else None
    if not isinstance(program, dict):
        program = program_data if isinstance(program_data, dict) else {}
    weeks = program.get("weeks") or []
    completed = {
        f"w{row.get('week_number')}-d{row.get('day_number')}"
        for row in (progress_rows or []) if row.get("status") == "completed"
    }
    total = 0
    next_session = None
    for week in weeks:
        for session in week.get("sessions") or []:
            total += 1
            rid = f"w{week.get('week')}-d{session.get('day')}"
            if next_session is None and rid not in completed:
                next_session = {
                    "route_id": rid,
                    "week": week.get("week"),
                    "day": session.get("day"),
                    "focus": session.get("focus"),
                    "estimated_duration_minutes": session.get("estimated_duration_minutes") or 20,
                }
    return {
        "total_sessions": total,
        "completed_sessions": len(completed),
        "progress_percent": round((len(completed) / total) * 100) if total else 0,
        "next_session": next_session,
        "program_summary": program.get("program_summary") or {},
    }



def _screening_credit_summary(user_id: str) -> dict:
    """Return the live, non-expired screening-credit balance for one client.

    This mirrors /me/entitlements so /me/bootstrap remains the single source
    used by Home. It is read-only and does not reserve or consume a credit.
    """
    cycles_response = (
        supabase.table("screening_credit_cycles")
        .select(
            "id,source,cycle_start,cycle_end,grace_expires_at,"
            "credits_granted,credits_used"
        )
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )

    now = datetime.now(timezone.utc)
    remaining = 0
    active_cycle = None
    next_future_cycle = None

    def parse_utc(value):
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None

    for cycle in cycles_response.data or []:
        cycle_start = parse_utc(cycle.get("cycle_start"))
        expiry = parse_utc(cycle.get("grace_expires_at") or cycle.get("cycle_end"))
        available = max(
            0,
            int(cycle.get("credits_granted") or 0)
            - int(cycle.get("credits_used") or 0),
        )

        if not cycle_start or available <= 0:
            continue

        if cycle_start <= now and (expiry is None or expiry >= now):
            remaining += available
            if active_cycle is None:
                active_cycle = cycle
        elif cycle_start > now:
            if next_future_cycle is None:
                next_future_cycle = cycle
            else:
                current_next = parse_utc(next_future_cycle.get("cycle_start"))
                if current_next is None or cycle_start < current_next:
                    next_future_cycle = cycle

    # Only screening_usage rows with usage_status=reserved hold a credit.
    # Historical sessions marked in_progress are not reservations.
    active_reservations = 0
    try:
        reservation_usage = (
            supabase.table("screening_usage")
            .select("id")
            .eq("user_id", user_id)
            .eq("usage_status", "reserved")
            .execute()
        )
        active_reservations = len(reservation_usage.data or [])
    except Exception:
        active_reservations = 0

    available_remaining = max(0, remaining - active_reservations)

    return {
        "screening_credits_remaining": available_remaining,
        "screening_credits_unused": remaining,
        "screening_credits_reserved": active_reservations,
        "screening_credit_expires_at": (
            active_cycle.get("grace_expires_at") or active_cycle.get("cycle_end")
            if active_cycle
            else None
        ),
        "next_credit_cycle_at": (
            next_future_cycle.get("cycle_start") if next_future_cycle else None
        ),
    }


@app.get("/me/bootstrap")
def me_bootstrap(lang: str = "en", authorization: str = Header(None)):
    started = time.perf_counter()
    user = authenticated_user(supabase, authorization)
    entitlement = effective_entitlement(supabase, user["id"])
    # Home must receive the same live credit balance as /me/entitlements.
    # Reading this value never reserves or consumes a screening credit.
    entitlement = {
        **entitlement,
        **_screening_credit_summary(user["id"]),
    }
    session = _latest_owned_session_for_user(user)
    program_summary = None
    current_program = _current_program_row(user["id"])
    progress_rows = _program_progress_rows(current_program["id"]) if current_program else []
    generation_state = _program_generation_state(
        user["id"], session, current_program, progress_rows
    )
    if current_program:
        program_summary = _compact_program_summary(current_program.get("program_data"), progress_rows)
        program_summary["program_id"] = current_program.get("id")
        program_summary["screening_session_id"] = current_program.get("screening_session_id")
        program_summary["status"] = current_program.get("status")
    return {
        "account": {"id": user["id"], "email": user["email"]},
        "account_mode": {"is_trainer": False, "mode": "client"},
        "entitlements": entitlement,
        "latest_session": session,
        "dashboard": {
            "score": session.get("composite_score") if session else None,
            "created_at": session.get("created_at") if session else None,
            "program": program_summary,
            "program_generation": generation_state,
        },
        "server_ms": round((time.perf_counter() - started) * 1000, 1),
    }


@app.get("/me/program-overview")
def me_program_overview(lang: str = "en", authorization: str = Header(None)):
    """Return the correct active program without replacing unfinished progress.

    Lifecycle rules:
    - No program yet: generate from the latest completed assessment.
    - Current program completed + newer assessment: generate automatically.
    - Current program unfinished + newer assessment: keep it active and expose
      a user-controlled generation action.
    - Same assessment already used: reuse the stored program.
    """
    total_started = time.perf_counter()
    phases = {}

    phase = time.perf_counter()
    user = authenticated_user(supabase, authorization)
    phases["auth"] = round((time.perf_counter() - phase) * 1000, 1)

    phase = time.perf_counter()
    entitlement = effective_entitlement(supabase, user["id"])
    phases["entitlements"] = round((time.perf_counter() - phase) * 1000, 1)
    if not entitlement.get("program_access"):
        raise HTTPException(status_code=402, detail={"code": "PROGRAM_ACCESS_REQUIRED"})

    phase = time.perf_counter()
    latest_session = _latest_owned_session_for_user(user)
    phases["latest_session"] = round((time.perf_counter() - phase) * 1000, 1)
    if not latest_session:
        raise HTTPException(status_code=404, detail={"code": "NO_COMPLETED_SCREENING"})

    phase = time.perf_counter()
    current_row = _current_program_row(user["id"])
    initial_current_id = str(current_row.get("id")) if current_row else None
    initial_progress_rows = _program_progress_rows(initial_current_id) if initial_current_id else []
    latest_program_row = _program_for_screening(user["id"], latest_session["id"])
    phases["program_state"] = round((time.perf_counter() - phase) * 1000, 1)

    auto_generated = False
    generation_reason = None

    # If a program already exists for the newest assessment, it is authoritative.
    if latest_program_row:
        current_row = latest_program_row
    else:
        lifecycle = _program_generation_state(
            user["id"], latest_session, current_row, initial_progress_rows
        )
        if lifecycle["automatic_generation_eligible"]:
            phase = time.perf_counter()
            generated = _generate_program_for_session(
                session_id=latest_session["id"],
                lang=lang,
                authorization=authorization,
            )
            phases["automatic_generation"] = round((time.perf_counter() - phase) * 1000, 1)
            current_row = _program_for_screening(user["id"], latest_session["id"])
            if not current_row:
                # Supabase may be unavailable in local tests; keep the generated
                # payload usable even when durable persistence is unavailable.
                program_payload = generated
                progress_payload = {
                    "program_id": generated.get("program_id") or latest_session["id"],
                    "screening_session_id": latest_session["id"],
                    "progress": [],
                }
                state = {
                    **lifecycle,
                    "auto_generated": True,
                    "generation_reason": "first_program" if lifecycle["current_program_id"] is None else "completed_program_and_new_assessment",
                    "can_generate_from_latest": False,
                    "has_newer_completed_assessment": False,
                }
                result = {
                    "access": {"program_access": True},
                    "latest_session": latest_session,
                    "program": program_payload,
                    "progress": progress_payload,
                    "generation": state,
                }
                phases["total"] = round((time.perf_counter() - total_started) * 1000, 1)
                result["timings_ms"] = phases
                return result
            auto_generated = True
            generation_reason = (
                "first_program"
                if lifecycle["current_program_id"] is None
                else "completed_program_and_new_assessment"
            )

    if not current_row:
        raise HTTPException(status_code=404, detail={"code": "PROGRAM_NOT_AVAILABLE"})

    phase = time.perf_counter()
    if initial_current_id and str(current_row.get("id")) == initial_current_id:
        progress_rows = initial_progress_rows
    else:
        progress_rows = _program_progress_rows(current_row["id"])
    phases["progress"] = round((time.perf_counter() - phase) * 1000, 1)

    state = _program_generation_state(
        user["id"], latest_session, current_row, progress_rows
    )
    state["auto_generated"] = auto_generated
    state["generation_reason"] = generation_reason

    result = {
        "access": {"program_access": True},
        "latest_session": latest_session,
        "program": _stored_program_response(current_row, lang=lang),
        "progress": {
            "program_id": str(current_row["id"]),
            "screening_session_id": current_row.get("screening_session_id"),
            "progress": progress_rows,
        },
        "generation": state,
    }
    phases["total"] = round((time.perf_counter() - total_started) * 1000, 1)
    logger.info("program_overview user_id=%s phases=%s generation=%s", user["id"], phases, state)
    result["timings_ms"] = phases
    return result


@app.post("/programs/generate-latest")
def generate_program_from_latest_assessment(
    replace_current: bool = Form(False),
    lang: str = Form("en"),
    authorization: str = Header(None),
):
    """Generate once from the newest completed assessment.

    The same assessment can never be used to create a second random program.
    An unfinished active program requires explicit replacement confirmation.
    """
    language = "en" if str(lang).lower().startswith("en") else "fr"
    user = authenticated_user(supabase, authorization)
    entitlement = effective_entitlement(supabase, user["id"])
    if not (entitlement.get("program_access") and entitlement.get("can_generate_program")):
        raise HTTPException(status_code=402, detail={"code": "PROGRAM_ACCESS_REQUIRED"})

    latest_session = _latest_owned_session_for_user(user)
    if not latest_session:
        raise HTTPException(status_code=404, detail={"code": "NO_COMPLETED_SCREENING"})

    existing_latest = _program_for_screening(user["id"], latest_session["id"])
    if existing_latest:
        return {
            "generated": False,
            "reason": "LATEST_ASSESSMENT_ALREADY_USED",
            "program": _stored_program_response(existing_latest, lang=language),
        }

    current_row = _current_program_row(user["id"])
    if current_row and not _assessment_is_newer(latest_session, current_row):
        raise HTTPException(
            status_code=409,
            detail={"code": "NO_NEWER_COMPLETED_ASSESSMENT"},
        )

    completion = _program_completion_summary(current_row) if current_row else None
    if current_row and not completion["is_completed"] and not replace_current:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "ACTIVE_PROGRAM_REPLACEMENT_CONFIRMATION_REQUIRED",
                "remaining_sessions": completion["remaining_sessions"],
            },
        )

    generated = _generate_program_for_session(
        session_id=latest_session["id"],
        lang=language,
        authorization=authorization,
    )
    return {
        "generated": True,
        "reason": "USER_REPLACED_ACTIVE_PROGRAM" if current_row and not completion["is_completed"] else "NEWER_ASSESSMENT",
        "program": generated,
    }


@app.get("/programs/{user_email}")
def programs_for_user(user_email: str, limit: int = 20, authorization: str = Header(None)):
    """Return durable program history, newest first, without the large JSON payload."""
    try:
        if not supabase:
            raise HTTPException(status_code=503, detail="Supabase is not configured on server.")
        user = authenticated_user(supabase, authorization)
        normalized_email = ensure_email_matches(user["email"], user_email)
        res = (
            supabase.table("corrective_programs")
            .select("id, user_email, screening_session_id, program_version, language, status, generated_at, completed_at, created_at")
            .eq("user_id", user["id"])
            .order("generated_at", desc=True)
            .limit(max(1, min(int(limit), 100)))
            .execute()
        )
        rows = getattr(res, "data", None) or []
        return {"user_email": normalized_email, "count": len(rows), "items": rows}
    except Exception as e:
        return {"user_email": user_email, "items": [], "error": str(e)}


@app.get("/program_progress")
def get_program_progress(
    program_id: str,
    authorization: str = Header(None),
):
    """Load completion states for a program owned by the authenticated account."""
    if not supabase:
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured on server.",
        )

    user = authenticated_user(supabase, authorization)
    program_row = require_owned_program(user, program_id)
    durable_id = str(program_row["id"])

    res = (
        supabase.table("program_session_progress")
        .select(
            "id, program_id, user_email, week_number, day_number, status, "
            "started_at, completed_at, updated_at, completion_data"
        )
        .eq("program_id", durable_id)
        .order("week_number")
        .order("day_number")
        .execute()
    )
    rows = getattr(res, "data", None) or []
    return {
        "program_id": durable_id,
        "screening_session_id": program_row.get("screening_session_id"),
        "progress": rows,
    }


@app.post("/program_progress")
def save_program_progress(
    program_id: str = Form(...),
    week_number: int = Form(...),
    day_number: int = Form(...),
    status: str = Form(...),
    completion_data: str = Form(None),
    authorization: str = Header(None),
):
    """Upsert one program day for the authenticated program owner."""
    allowed = {"not_started", "in_progress", "completed"}
    state = str(status or "").strip().lower()
    if state not in allowed:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_status", "allowed": sorted(allowed)},
        )
    if not 1 <= int(week_number) <= 4 or not 1 <= int(day_number) <= 3:
        raise HTTPException(
            status_code=422,
            detail="Invalid week or day.",
        )
    if not supabase:
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured on server.",
        )

    user = authenticated_user(supabase, authorization)
    program_row = require_owned_program(user, program_id)
    durable_id = str(program_row["id"])
    now = utc_now_iso()
    extra = safe_json_loads(completion_data) if completion_data else None

    payload = {
        "program_id": durable_id,
        "user_id": user["id"],
        "user_email": user["email"],
        "week_number": int(week_number),
        "day_number": int(day_number),
        "status": state,
        "updated_at": now,
        "completion_data": extra,
    }
    if state == "in_progress":
        payload["started_at"] = now
    elif state == "completed":
        payload["completed_at"] = now
        payload["started_at"] = now

    existing = (
        supabase.table("program_session_progress")
        .select("id, status, started_at, completed_at, completion_data")
        .eq("program_id", durable_id)
        .eq("week_number", int(week_number))
        .eq("day_number", int(day_number))
        .limit(1)
        .execute()
    )
    rows = getattr(existing, "data", None) or []
    if rows:
        existing_row = rows[0]

        # A completed day is immutable within the same program. Users may open
        # it again for review, but an automatic "in_progress" call must never
        # downgrade completion. A new assessment receives a new program_id and
        # therefore starts with a clean progress set.
        if existing_row.get("status") == "completed" and state != "completed":
            saved = existing
        else:
            if existing_row.get("started_at") and state == "completed":
                payload.pop("started_at", None)
            saved = (
                supabase.table("program_session_progress")
                .update(payload)
                .eq("id", existing_row["id"])
                .execute()
            )
    else:
        saved = (
            supabase.table("program_session_progress")
            .insert(payload)
            .execute()
        )

    completed_rows = (
        supabase.table("program_session_progress")
        .select("id")
        .eq("program_id", durable_id)
        .eq("status", "completed")
        .execute()
    )
    completed_count = len(getattr(completed_rows, "data", None) or [])
    if completed_count >= 12:
        (
            supabase.table("corrective_programs")
            .update({"status": "completed", "completed_at": now})
            .eq("id", durable_id)
            .execute()
        )

    saved_rows = getattr(saved, "data", None) or []
    return {
        "saved": True,
        "program_id": durable_id,
        "week_number": int(week_number),
        "day_number": int(day_number),
        "status": state,
        "completed_days": completed_count,
        "progress": saved_rows[0] if saved_rows else payload,
    }


def _generate_program_for_session(
    session_id: str,
    lang: str = "fr",
    intake_json: str = None,
    questionnaire_json: str = None,
    authorization: str = Header(None),
):
    """
    FlexiLab V2.1 clinical program endpoint.

    Flow:
    1) Build report from existing session.
    2) Attach Score V2.
    3) Attach Movement DNA + clinical pattern recognition.
    4) Generate the 4-week balanced block-based clinical prescription program.
    5) Return legacy-compatible keys: report, program, prescription.
    """
    lang = "en" if str(lang).lower().startswith("en") else "fr"
    program_user = authenticated_user(supabase, authorization)
    program_session = require_owned_session(program_user, session_id)
    if str(program_session.get("user_id") or "") != program_user["id"]:
        raise HTTPException(
            status_code=403,
            detail="Only the client who owns this screening can access its corrective program.",
        )
    entitlement = effective_entitlement(supabase, program_user["id"])
    if not (entitlement.get("program_access") and entitlement.get("can_generate_program")):
        raise HTTPException(
            status_code=402,
            detail={
                "code": "PROGRAM_ACCESS_REQUIRED",
                "message": "An active FlexiLab plan with corrective-program access is required.",
            },
        )

    # A program is immutable for a given completed assessment. Reopening the
    # Program page must reuse the durable program instead of generating a new
    # exercise selection from the same results.
    existing_program = _program_for_screening(program_user["id"], session_id)
    if existing_program:
        return _stored_program_response(existing_program, lang=lang)

    try:
        report_data = report(
            session_id=session_id,
            lang=lang,
            authorization=authorization,
        )
    except HTTPException:
        raise
    except Exception as e:
        report_data = {
            "session_id": session_id,
            "flexilab_score": 66,
            "sections": [],
            "fallback_reason": f"report_failed: {str(e)}"
        }

    if not isinstance(report_data, dict):
        report_data = {"session_id": session_id, "flexilab_score": 66, "sections": []}

    if report_data.get("error"):
        report_data = {
            "session_id": session_id,
            "flexilab_score": 66,
            "sections": [],
            "fallback_reason": report_data.get("error")
        }

    # Optional query-level compatibility for tests: /program?...&questionnaire_json={...}
    query_intake = parse_intake_payload(intake_json=intake_json, questionnaire_json=questionnaire_json)
    if query_intake:
        existing_intake = report_data.get("intake_context")
        if isinstance(existing_intake, dict) and isinstance(query_intake, dict):
            merged_intake = dict(existing_intake)
            merged_intake.update(query_intake)
            report_data["intake_context"] = merged_intake
        else:
            report_data["intake_context"] = query_intake

    # /report already attaches the evidence-aware score. Keep this fallback
    # only for legacy/fallback report payloads.
    if not report_data.get("score_v2"):
        try:
            if attach_score_v2:
                report_data = attach_score_v2(report_data, lang=lang)
            else:
                report_data["score_v2_error"] = "attach_score_v2 not loaded"
        except Exception as e:
            report_data["score_v2_error"] = str(e)

    # Movement DNA + clinical pattern recognition.
    report_data = attach_movement_dna_to_report(report_data, lang=lang)

    # Clinical Prescription Engine v2.1.1: balanced block-based program generation.
    try:
        if not generate_clinical_prescription_v21:
            raise RuntimeError("generate_clinical_prescription_v21 not loaded")
        if not EXERCISE_LIBRARY:
            raise RuntimeError("exercise library not loaded")
        if not PRESCRIPTION_RULES:
            raise RuntimeError("prescription rules not loaded")

        clinical_program = generate_clinical_prescription_v21(
            {"report": report_data, "intake_context": report_data.get("intake_context")},
            exercise_library=EXERCISE_LIBRARY,
            rules=PRESCRIPTION_RULES,
            movement_dna=report_data.get("movement_dna"),
            language=lang,
        )
        clinical_program = normalize_clinical_program_for_frontend(clinical_program, report_data, lang=lang)

        # V85: bind each generated program to the screening that created it.
        # This gives the frontend a stable namespace for session-completion state.
        generated_at = datetime.now(timezone.utc).isoformat()
        clinical_program["program_id"] = str(session_id)
        clinical_program["generated_from_screening_id"] = str(session_id)
        clinical_program["generated_at"] = generated_at
        clinical_program["program_version"] = "V85-" + str(session_id)

    except Exception as e:
        clinical_program = {
            "engine_version": "FlexiLab Clinical Prescription Engine unavailable",
            "error": str(e),
            "resource_load_errors": RESOURCE_LOAD_ERRORS,
            "weeks": [],
            "report_ready_summary": {
                "headline": "Clinical prescription engine unavailable." if lang == "en" else "Moteur de prescription clinique indisponible.",
                "total_sessions": 0,
                "average_session_duration_minutes": 0,
                "next_action": "Check server logs and resource paths." if lang == "en" else "Vérifier les logs serveur et les chemins des fichiers."
            }
        }

    # V87: persist the program and expose its durable database UUID.
    user_email = get_session_user_email(session_id, report_data)
    durable_program = persist_corrective_program(
        user_email=user_email,
        user_id=program_user["id"],
        screening_session_id=session_id,
        language=lang,
        program_data=clinical_program if isinstance(clinical_program, dict) else {},
    )
    durable_program_id = durable_program.get("program_id") or str(session_id)

    if isinstance(clinical_program, dict):
        clinical_program["program_id"] = durable_program_id
        clinical_program["generated_from_screening_id"] = str(session_id)
        clinical_program.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
        clinical_program["program_storage_status"] = durable_program

    result_payload = {
        "session_id": session_id,
        "program_id": durable_program_id,
        "generated_from_screening_id": str(session_id),
        "program_generated_at": clinical_program.get("generated_at") if isinstance(clinical_program, dict) else datetime.now(timezone.utc).isoformat(),
        "language": lang,
        "report": report_data,
        "movement_dna": report_data.get("movement_dna"),
        "clinical_patterns": report_data.get("clinical_patterns", []),
        "program": clinical_program,
        # Legacy alias: current frontend may still read response["prescription"].
        "prescription": clinical_program,
        "resource_load_errors": RESOURCE_LOAD_ERRORS,
        "api_contract_note": {
            "program_is_canonical": True,
            "prescription_is_legacy_alias": True,
            "clinical_engine_expected": "FlexiLab Clinical Prescription Engine v2.1.1",
            "i18n_contract": "v85 program identity + completion-state namespace; v68 filmed demo library; v64 questionnaire compatibility"
        }
    }

    history_status = save_screening_history(user_email=user_email, session_id=session_id, result=result_payload)
    result_payload["history_status"] = history_status

    return result_payload



@app.get("/program")
def program(
    session_id: str,
    lang: str = "fr",
    intake_json: str = None,
    questionnaire_json: str = None,
    authorization: str = Header(None),
):
    """Return the correct durable program under the lifecycle rules.

    This compatibility endpoint no longer replaces an unfinished program merely
    because a newer assessment exists. Explicit replacement uses the dedicated
    POST endpoint after user confirmation.
    """
    language = "en" if str(lang).lower().startswith("en") else "fr"
    user = authenticated_user(supabase, authorization)
    entitlement = effective_entitlement(supabase, user["id"])
    if not (entitlement.get("program_access") and entitlement.get("can_generate_program")):
        raise HTTPException(status_code=402, detail={"code": "PROGRAM_ACCESS_REQUIRED"})

    # Only the client who owns this screening can access its corrective program.
    requested_session = require_owned_session(user, session_id)
    if str(requested_session.get("user_id") or "") != user["id"]:
        raise HTTPException(
            status_code=403,
            detail="Only the client who owns this screening can access its corrective program.",
        )
    if str(requested_session.get("status") or "").lower() != "completed":
        raise HTTPException(status_code=409, detail={"code": "COMPLETED_ASSESSMENT_REQUIRED"})

    existing = _program_for_screening(user["id"], session_id)
    if existing:
        return _stored_program_response(existing, lang=language)

    latest_session = _latest_owned_session_for_user(user)
    if not latest_session or str(latest_session.get("id")) != str(session_id):
        raise HTTPException(
            status_code=409,
            detail={"code": "PROGRAM_GENERATION_REQUIRES_LATEST_ASSESSMENT"},
        )

    current_row = _current_program_row(user["id"])
    if current_row and _assessment_is_newer(latest_session, current_row):
        completion = _program_completion_summary(current_row)
        if not completion["is_completed"]:
            response = _stored_program_response(current_row, lang=language)
            response["generation"] = {
                "has_newer_completed_assessment": True,
                "can_generate_from_latest": True,
                "remaining_sessions": completion["remaining_sessions"],
                "latest_assessment_id": latest_session.get("id"),
            }
            return response

    return _generate_program_for_session(
        session_id=session_id,
        lang=language,
        intake_json=intake_json,
        questionnaire_json=questionnaire_json,
        authorization=authorization,
    )


@app.get("/history/{user_email}")
def history(user_email: str, limit: int = 20, authorization: str = Header(None)):
    """
    Return one real history item per screening session.

    New records use sessions.created_at and sessions.composite_score.
    Legacy records fall back to the oldest screening_history snapshot for the
    same session, so existing screenings remain visible without creating fake dates.
    """
    try:
        if not supabase:
            raise HTTPException(status_code=503, detail="Supabase is not configured on server.")
        user = authenticated_user(supabase, authorization)
        normalized_email = ensure_email_matches(user["email"], user_email)

        s_resp = (
            supabase.table("sessions")
            .select("id, user_email, created_at, composite_score, status")
            .eq("user_id", user["id"] )
            .order("created_at", desc=True)
            .limit(max(limit * 4, 50))
            .execute()
        )
        sessions = getattr(s_resp, "data", None) or []
        session_map = {str(r.get("id")): r for r in sessions if r.get("id")}

        h_resp = (
            supabase.table("screening_history")
            .select("id, user_email, session_id, created_at, flexilab_score, risk_level")
            .eq("user_email", normalized_email)
            .order("created_at", desc=False)
            .limit(max(limit * 10, 200))
            .execute()
        )
        snapshots = getattr(h_resp, "data", None) or []
        oldest_snapshot = {}
        for row in snapshots:
            sid = str(row.get("session_id") or "")
            if sid and sid not in oldest_snapshot:
                oldest_snapshot[sid] = row

        candidate_ids = set(session_map.keys()) | set(oldest_snapshot.keys())
        rows = []
        for sid in candidate_ids:
            session = session_map.get(sid) or {}
            snap = oldest_snapshot.get(sid) or {}

            score = session.get("composite_score")
            if score is None:
                score = snap.get("flexilab_score")
            if score is None:
                continue

            created_at = session.get("created_at") or snap.get("created_at")
            if not created_at:
                continue

            rows.append({
                "id": sid,
                "user_email": session.get("user_email") or snap.get("user_email") or normalized_email,
                "session_id": sid,
                "created_at": created_at,
                "flexilab_score": score,
                "risk_level": snap.get("risk_level"),
                "session_status": session.get("status"),
            })

        rows.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)
        rows = rows[:limit]
        return {
            "user_email": normalized_email,
            "count": len(rows),
            "items": rows,
            "source": "sessions_with_legacy_snapshot_fallback",
        }
    except Exception as e:
        return {"user_email": normalized_email, "items": [], "error": str(e)}


@app.get("/history/{user_email}/latest")
def latest_history(user_email: str, authorization: str = Header(None)):
    """Return the latest two items using the same compatibility source."""
    payload = history(user_email=user_email, limit=2, authorization=authorization)
    rows = payload.get("items") or []
    latest = rows[0] if len(rows) >= 1 else None
    previous = rows[1] if len(rows) >= 2 else None
    delta = None
    if latest and previous:
        try:
            delta = round(float(latest["flexilab_score"]) - float(previous["flexilab_score"]), 1)
        except Exception:
            delta = None
    return {
        "user_email": user_email,
        "latest": latest,
        "previous": previous,
        "score_delta": delta,
        "source": payload.get("source"),
        "error": payload.get("error"),
    }
