import time
from datetime import datetime
import base64
import numpy as np
import cv2

from app import (
    supabase,
    analyze_posture,
    analyze_shoulder,
    analyze_squat,
    model
)

print("FlexiLab worker started...")

while True:

    try:

        jobs = (
            supabase.table("analysis_jobs")
            .select("*")
            .eq("status", "queued")
            .order("created_at")
            .limit(1)
            .execute()
        )

        if not jobs.data:
            time.sleep(2)
            continue

        job = jobs.data[0]
        job_id = job["id"]

        print(f"Processing job {job_id}")

        supabase.table("analysis_jobs").update(
            {
                "status": "processing",
                "started_at": datetime.utcnow().isoformat()
            }
        ).eq("id", job_id).execute()

        img_bytes = base64.b64decode(job["image_base64"])

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            supabase.table("analysis_jobs").update(
                {
                    "status": "failed",
                    "error_message": "Invalid image"
                }
            ).eq("id", job_id).execute()
            continue

        # Resize large images to avoid memory crashes
        h, w = img.shape[:2]
        max_side = 480
        scale = max_side / max(h, w)

        if scale < 1.0:
            img = cv2.resize(
                img,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA
            )

        res = model(img, conf=0.5, classes=[0])

        if res[0].keypoints is None or len(res[0].keypoints.xy) == 0:
            supabase.table("analysis_jobs").update(
                {
                    "status": "failed",
                    "error_message": "No person detected"
                }
            ).eq("id", job_id).execute()
            continue

        boxes = res[0].boxes.xyxy.cpu().numpy()
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        main_idx = int(np.argmax(areas))

        xy = res[0].keypoints.xy[main_idx].cpu().numpy()
        conf = res[0].keypoints.conf[main_idx].cpu().numpy()

        test_type = job["test_type"]

        if test_type == "posture_side":
            result = analyze_posture(xy, conf)

        elif test_type == "shoulder_right":
            result = analyze_shoulder(xy, conf, "RIGHT")

        elif test_type == "shoulder_left":
            result = analyze_shoulder(xy, conf, "LEFT")

        elif test_type == "squat":
            result = analyze_squat(xy, conf)

        else:
            supabase.table("analysis_jobs").update(
                {
                    "status": "failed",
                    "error_message": "Invalid test_type"
                }
            ).eq("id", job_id).execute()
            continue

        supabase.table("analysis_jobs").update(
            {
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "result_json": result
            }
        ).eq("id", job_id).execute()

        print(f"Completed job {job_id}")

    except Exception as e:

        print("Worker error:", str(e))

    time.sleep(1)
