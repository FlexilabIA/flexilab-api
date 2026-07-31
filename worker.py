"""FlexiLab adaptive analysis worker pool.

Run as a Render Background Worker:
    python worker.py

The web API only uploads and queues. Every image-analysis operation is executed
inside one of these isolated worker processes. Each process owns its YOLO model
instances, avoiding cross-thread model access.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from supabase import create_client

MAX_CONCURRENCY = max(1, min(3, int(os.environ.get("ANALYSIS_WORKER_MAX_CONCURRENCY", "3"))))
MIN_CONCURRENCY = max(1, min(MAX_CONCURRENCY, int(os.environ.get("ANALYSIS_WORKER_MIN_CONCURRENCY", "1"))))
POLL_SECONDS = max(0.25, float(os.environ.get("ANALYSIS_WORKER_POLL_SECONDS", "1")))
SUPERVISOR_SECONDS = max(1.0, float(os.environ.get("ANALYSIS_WORKER_SUPERVISOR_SECONDS", "2")))
IDLE_EXIT_SECONDS = max(30, int(os.environ.get("ANALYSIS_WORKER_IDLE_EXIT_SECONDS", "90")))
CLEANUP_INTERVAL_SECONDS = max(60, int(os.environ.get("ANALYSIS_WORKER_CLEANUP_SECONDS", "300")))
MIN_FREE_MEMORY_MB = max(256, int(os.environ.get("ANALYSIS_WORKER_MIN_FREE_MEMORY_MB", "700")))
MAX_MEMORY_PERCENT = min(95.0, max(50.0, float(os.environ.get("ANALYSIS_WORKER_MAX_MEMORY_PERCENT", "82"))))
MAX_CPU_PERCENT = min(100.0, max(50.0, float(os.environ.get("ANALYSIS_WORKER_MAX_CPU_PERCENT", "90"))))
STALE_PROCESSING_SECONDS = max(180, int(os.environ.get("ANALYSIS_WORKER_STALE_PROCESSING_SECONDS", "300")))
IMAGE_DELETE_RETRIES = max(1, min(5, int(os.environ.get("FLEXILAB_ANALYSIS_IMAGE_DELETE_RETRIES", "3"))))
ASLR_WEIGHT = 2
LIGHT_WEIGHT = 1
CAPACITY_UNITS = MAX_CONCURRENCY

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = (
    os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
)
ANALYSIS_STORAGE_BUCKET = os.environ.get("FLEXILAB_ANALYSIS_BUCKET", "screening-private").strip()

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")


def _log(event: str, **fields: object) -> None:
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"{event}{(' ' + payload) if payload else ''}", flush=True)


def _delete_storage_object(client: Any, path: str, *, job_id: str) -> bool:
    clean_path = str(path or "").strip()
    if not clean_path:
        return True
    last_error: Exception | None = None
    for attempt in range(1, IMAGE_DELETE_RETRIES + 1):
        try:
            client.storage.from_(ANALYSIS_STORAGE_BUCKET).remove([clean_path])
            _log(
                "worker_image_deleted",
                job_id=job_id,
                path=clean_path,
                attempt=attempt,
            )
            return True
        except Exception as exc:
            last_error = exc
            _log(
                "worker_image_delete_retry",
                job_id=job_id,
                path=clean_path,
                attempt=f"{attempt}/{IMAGE_DELETE_RETRIES}",
                error=repr(exc),
            )
            if attempt < IMAGE_DELETE_RETRIES:
                time.sleep(0.25 * attempt)
    _log(
        "worker_image_delete_failed",
        job_id=job_id,
        path=clean_path,
        retries=IMAGE_DELETE_RETRIES,
        error=repr(last_error),
    )
    return False


def _memory_snapshot() -> tuple[float, float]:
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, raw = line.split(":", 1)
                values[key] = int(raw.strip().split()[0])
        total_kb = values.get("MemTotal", 0)
        available_kb = values.get("MemAvailable", values.get("MemFree", 0))
        used_percent = 100.0 * (1.0 - (available_kb / total_kb)) if total_kb else 0.0
        return available_kb / 1024.0, used_percent
    except Exception:
        return 99999.0, 0.0


def _cpu_pressure_percent() -> float:
    try:
        one_minute_load = os.getloadavg()[0]
        cpus = max(1, os.cpu_count() or 1)
        return min(100.0, (one_minute_load / cpus) * 100.0)
    except Exception:
        return 0.0


def _resources_allow_scale_up() -> tuple[bool, dict[str, float]]:
    free_mb, memory_percent = _memory_snapshot()
    cpu_percent = _cpu_pressure_percent()
    safe = (
        free_mb >= MIN_FREE_MEMORY_MB
        and memory_percent <= MAX_MEMORY_PERCENT
        and cpu_percent <= MAX_CPU_PERCENT
    )
    return safe, {
        "free_memory_mb": round(free_mb, 1),
        "memory_percent": round(memory_percent, 1),
        "cpu_pressure_percent": round(cpu_percent, 1),
    }


def _job_weight(test_type: str) -> int:
    return ASLR_WEIGHT if str(test_type).lower().startswith("aslr_") else LIGHT_WEIGHT


def _acquire_units(semaphore: Any, units: int, timeout: float = 0.1) -> bool:
    acquired = 0
    try:
        for _ in range(units):
            if not semaphore.acquire(timeout=timeout):
                return False
            acquired += 1
        return True
    finally:
        if acquired != units:
            for _ in range(acquired):
                semaphore.release()


def _release_units(semaphore: Any, units: int) -> None:
    for _ in range(units):
        semaphore.release()


def _child_loop(slot: int, stop_event: Any, capacity: Any) -> None:
    os.environ["FLEXILAB_PROCESS_ROLE"] = "worker"
    worker_id = f"{os.uname().nodename}-{slot}-{os.getpid()}-{uuid.uuid4().hex[:6]}"

    # Importing app here gives every process its own preloaded model instances.
    from app import process_analysis_job, supabase  # pylint: disable=import-outside-toplevel

    if supabase is None:
        raise RuntimeError("Supabase is not configured in worker child")

    _log("worker_slot_started", worker_id=worker_id, slot=slot, pid=os.getpid())
    last_work_at = time.monotonic()

    while not stop_event.is_set():
        try:
            jobs = (
                supabase.table("analysis_jobs")
                .select("id,session_id,test_type,created_at")
                .eq("status", "queued")
                .order("created_at")
                .limit(5)
                .execute()
            )
            rows = jobs.data or []
            if not rows:
                if slot >= MIN_CONCURRENCY and time.monotonic() - last_work_at >= IDLE_EXIT_SECONDS:
                    _log("worker_slot_idle_exit", worker_id=worker_id, slot=slot)
                    return
                time.sleep(POLL_SECONDS)
                continue

            selected = None
            selected_weight = 0
            for candidate in rows:
                weight = _job_weight(str(candidate.get("test_type") or ""))
                if _acquire_units(capacity, weight):
                    selected = candidate
                    selected_weight = weight
                    break
            if selected is None:
                time.sleep(POLL_SECONDS)
                continue

            job_id = str(selected["id"])
            session_id = str(selected.get("session_id") or "")
            test_type = str(selected.get("test_type") or "")
            started = time.perf_counter()
            try:
                _log(
                    "worker_job_found",
                    worker_id=worker_id,
                    slot=slot,
                    job_id=job_id,
                    session_id=session_id,
                    test_type=test_type,
                    weight=selected_weight,
                    created_at=selected.get("created_at"),
                )
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
                _log(
                    "worker_job_terminal",
                    worker_id=worker_id,
                    slot=slot,
                    job_id=job_id,
                    test_type=test_type,
                    status=final_status,
                    duration_ms=duration_ms,
                    error=repr(final_row.get("error_message")) if final_status == "failed" else None,
                )
                last_work_at = time.monotonic()
            finally:
                _release_units(capacity, selected_weight)
        except Exception as exc:
            _log("worker_slot_error", worker_id=worker_id, slot=slot, error=repr(exc))
            traceback.print_exc()
            time.sleep(POLL_SECONDS)

    _log("worker_slot_stopped", worker_id=worker_id, slot=slot)


def _queue_depth(client: Any) -> int:
    response = (
        client.table("analysis_jobs")
        .select("id", count="exact")
        .eq("status", "queued")
        .limit(1)
        .execute()
    )
    return int(getattr(response, "count", None) or len(response.data or []))


def _cleanup_and_recover(client: Any) -> None:
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    stale_before = (now - timedelta(seconds=STALE_PROCESSING_SECONDS)).isoformat()

    # A process killed during inference must not leave a job permanently stuck.
    stale = (
        client.table("analysis_jobs")
        .select("id")
        .eq("status", "processing")
        .lt("started_at", stale_before)
        .limit(100)
        .execute()
    )
    for row in stale.data or []:
        client.table("analysis_jobs").update({
            "status": "queued",
            "started_at": None,
            "error_message": "Recovered after an interrupted worker process.",
        }).eq("id", row["id"]).eq("status", "processing").execute()
        _log("worker_stale_job_requeued", job_id=row["id"])

    expired = (
        client.table("analysis_jobs")
        .select("id,status,image_path")
        .lt("image_expires_at", now_iso)
        .limit(100)
        .execute()
    )
    for row in expired.data or []:
        path = str(row.get("image_path") or "").strip()
        storage_removed = _delete_storage_object(
            client, path, job_id=str(row.get("id") or "")
        )
        changes: dict[str, object] = {"image_base64": None}
        if storage_removed:
            changes.update({"image_path": None, "image_expires_at": None})
        if str(row.get("status") or "").lower() in {"queued", "processing"}:
            changes.update({
                "status": "failed",
                "completed_at": now_iso,
                "error_message": "Queued image expired before analysis completed.",
            })
        client.table("analysis_jobs").update(changes).eq("id", row["id"]).execute()


def main() -> None:
    mp.set_start_method("spawn", force=True)
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    stop_event = mp.Event()
    capacity = mp.BoundedSemaphore(CAPACITY_UNITS)
    children: dict[int, mp.Process] = {}
    last_cleanup = 0.0

    def request_stop(*_: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    _log(
        "worker_pool_started",
        max_concurrency=MAX_CONCURRENCY,
        min_concurrency=MIN_CONCURRENCY,
        capacity_units=CAPACITY_UNITS,
        poll_seconds=POLL_SECONDS,
        min_free_memory_mb=MIN_FREE_MEMORY_MB,
        max_memory_percent=MAX_MEMORY_PERCENT,
        max_cpu_percent=MAX_CPU_PERCENT,
    )

    while not stop_event.is_set():
        try:
            for slot, process in list(children.items()):
                if not process.is_alive():
                    process.join(timeout=0.1)
                    children.pop(slot, None)
                    _log("worker_slot_reaped", slot=slot, exitcode=process.exitcode)

            current = time.monotonic()
            if current - last_cleanup >= CLEANUP_INTERVAL_SECONDS:
                _cleanup_and_recover(client)
                last_cleanup = current

            queued = _queue_depth(client)
            alive = len(children)
            desired = min(MAX_CONCURRENCY, max(MIN_CONCURRENCY, queued))

            while alive < desired:
                safe, resources = _resources_allow_scale_up()
                if not safe and alive >= MIN_CONCURRENCY:
                    _log("worker_backpressure", queued=queued, active_processes=alive, **resources)
                    break
                slot = next(index for index in range(MAX_CONCURRENCY) if index not in children)
                process = mp.Process(
                    target=_child_loop,
                    args=(slot, stop_event, capacity),
                    name=f"flexilab-analysis-{slot}",
                    daemon=False,
                )
                process.start()
                children[slot] = process
                alive += 1
                _log("worker_slot_spawned", slot=slot, pid=process.pid, queued=queued, **resources)
                # Let memory settle before deciding whether another model process is safe.
                time.sleep(1.5)

            time.sleep(SUPERVISOR_SECONDS)
        except Exception as exc:
            _log("worker_supervisor_error", error=repr(exc))
            traceback.print_exc()
            time.sleep(SUPERVISOR_SECONDS)

    _log("worker_pool_stopping", active_processes=len(children))
    for process in children.values():
        process.join(timeout=30)
    for process in children.values():
        if process.is_alive():
            process.terminate()
    _log("worker_pool_stopped")


if __name__ == "__main__":
    main()
