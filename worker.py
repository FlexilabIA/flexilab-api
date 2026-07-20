"""FlexiLab analysis worker.

Run as a separate Render Background Worker:
    python worker.py

For production, set FLEXILAB_INLINE_ANALYSIS=false on the web service and run one
or more workers. Database claims are atomic, so multiple workers can safely poll
the same queue without processing one job twice.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from app import ANALYSIS_STORAGE_BUCKET, process_analysis_job, supabase

POLL_SECONDS = max(1, int(os.environ.get("ANALYSIS_WORKER_POLL_SECONDS", "2")))
CLEANUP_INTERVAL_SECONDS = max(
    60,
    int(os.environ.get("ANALYSIS_WORKER_CLEANUP_SECONDS", "300")),
)

if supabase is None:
    raise RuntimeError("Supabase is not configured")


def cleanup_expired_images() -> None:
    now = datetime.now(timezone.utc).isoformat()
    expired = (
        supabase.table("analysis_jobs")
        .select("id,status,image_path")
        .lt("image_expires_at", now)
        .limit(100)
        .execute()
    )
    for row in expired.data or []:
        path = str(row.get("image_path") or "").strip()
        storage_removed = not path
        if path:
            try:
                supabase.storage.from_(ANALYSIS_STORAGE_BUCKET).remove([path])
                storage_removed = True
            except Exception:
                # Leave the path and expiry in place so a later cleanup pass can retry.
                storage_removed = False
        status = str(row.get("status") or "").lower()
        changes: dict[str, object] = {
            "image_base64": None,
        }
        if storage_removed:
            changes.update({"image_path": None, "image_expires_at": None})
        if status in {"queued", "processing"}:
            changes.update({
                "status": "failed",
                "completed_at": now,
                "error_message": "Queued image expired before analysis completed.",
            })
        supabase.table("analysis_jobs").update(changes).eq("id", row["id"]).execute()


print("FlexiLab analysis worker started", flush=True)
last_cleanup = 0.0
while True:
    try:
        current = time.monotonic()
        if current - last_cleanup >= CLEANUP_INTERVAL_SECONDS:
            cleanup_expired_images()
            last_cleanup = current

        jobs = (
            supabase.table("analysis_jobs")
            .select("id")
            .eq("status", "queued")
            .order("created_at")
            .limit(1)
            .execute()
        )
        if jobs.data:
            process_analysis_job(str(jobs.data[0]["id"]))
        else:
            time.sleep(POLL_SECONDS)
    except Exception as exc:
        print(f"Analysis worker error: {exc}", flush=True)
        time.sleep(POLL_SECONDS)
