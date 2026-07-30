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
import traceback
from datetime import datetime, timezone

# Must be set before importing app so analysis execution is cryptographically
# separated by process role rather than relying on a boolean Render variable.
os.environ["FLEXILAB_PROCESS_ROLE"] = "worker"

from app import ANALYSIS_STORAGE_BUCKET, PROCESS_ROLE, process_analysis_job, supabase

POLL_SECONDS = max(1, int(os.environ.get("ANALYSIS_WORKER_POLL_SECONDS", "2")))
CLEANUP_INTERVAL_SECONDS = max(
    60,
    int(os.environ.get("ANALYSIS_WORKER_CLEANUP_SECONDS", "300")),
)

if supabase is None:
    raise RuntimeError("Supabase is not configured")


def _log(event: str, **fields: object) -> None:
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"{event}{(' ' + payload) if payload else ''}", flush=True)


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


_log(
    "worker_started",
    role=PROCESS_ROLE,
    bucket=ANALYSIS_STORAGE_BUCKET,
    poll_seconds=POLL_SECONDS,
    cleanup_seconds=CLEANUP_INTERVAL_SECONDS,
)
last_cleanup = 0.0
while True:
    try:
        current = time.monotonic()
        if current - last_cleanup >= CLEANUP_INTERVAL_SECONDS:
            cleanup_started = time.perf_counter()
            cleanup_expired_images()
            _log("worker_cleanup_completed", duration_ms=round((time.perf_counter() - cleanup_started) * 1000, 1))
            last_cleanup = current

        jobs = (
            supabase.table("analysis_jobs")
            .select("id,session_id,test_type,created_at")
            .eq("status", "queued")
            .order("created_at")
            .limit(1)
            .execute()
        )
        if not jobs.data:
            time.sleep(POLL_SECONDS)
            continue

        job = jobs.data[0]
        job_id = str(job["id"])
        session_id = str(job.get("session_id") or "")
        test_type = str(job.get("test_type") or "")
        started = time.perf_counter()
        _log(
            "worker_job_found",
            job_id=job_id,
            session_id=session_id,
            test_type=test_type,
            created_at=job.get("created_at"),
        )
        _log("worker_processing_started", job_id=job_id, test_type=test_type)

        process_analysis_job(job_id)

        final = (
            supabase.table("analysis_jobs")
            .select("status,started_at,completed_at,error_message")
            .eq("id", job_id)
            .limit(1)
            .execute()
        )
        final_row = (final.data or [{}])[0]
        final_status = str(final_row.get("status") or "unknown")
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        if final_status == "completed":
            _log(
                "worker_completed",
                job_id=job_id,
                session_id=session_id,
                test_type=test_type,
                duration_ms=duration_ms,
                started_at=final_row.get("started_at"),
                completed_at=final_row.get("completed_at"),
            )
        elif final_status == "failed":
            _log(
                "worker_failed",
                job_id=job_id,
                session_id=session_id,
                test_type=test_type,
                duration_ms=duration_ms,
                error=repr(final_row.get("error_message")),
            )
        else:
            _log(
                "worker_processing_returned",
                job_id=job_id,
                session_id=session_id,
                test_type=test_type,
                status=final_status,
                duration_ms=duration_ms,
            )
    except Exception as exc:
        _log("worker_loop_error", error=repr(exc))
        traceback.print_exc()
        time.sleep(POLL_SECONDS)
