"""Durable FlexiLab background worker.

Run as a separate Render Background Worker:
    python operator_worker.py

It processes queued corporate CSV imports in small idempotent batches. The API only
creates jobs and rows; this worker owns invitation/account creation so a browser
request can end without interrupting a large import.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from supabase import create_client


SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_KEY = (
    os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
)
FRONTEND_URL = os.environ.get(
    "FRONTEND_URL",
    "https://flexi-move-lab.lovable.app",
).rstrip("/")
POLL_SECONDS = max(1, int(os.environ.get("OPERATOR_WORKER_POLL_SECONDS", "3")))
BATCH_SIZE = max(1, min(int(os.environ.get("OPERATOR_IMPORT_BATCH_SIZE", "20")), 100))

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_user(user_obj: Any) -> tuple[Optional[str], Optional[str]]:
    user = getattr(user_obj, "user", None)
    if user is None and isinstance(user_obj, dict):
        user = user_obj.get("user")
    user_id = getattr(user, "id", None)
    email = getattr(user, "email", None)
    if isinstance(user, dict):
        user_id = user_id or user.get("id")
        email = email or user.get("email")
    return (str(user_id) if user_id else None, str(email or "").strip().lower() or None)


def find_profile(email: str) -> Optional[dict[str, Any]]:
    response = supabase.table("profiles").select("id,email,full_name").ilike("email", email).limit(1).execute()
    return response.data[0] if response.data else None


def invite_or_find(email: str, full_name: str) -> tuple[Optional[str], str]:
    profile = find_profile(email)
    if profile:
        return str(profile["id"]), "existing_user"

    invited = supabase.auth.admin.invite_user_by_email(
        email,
        options={
            "redirect_to": f"{FRONTEND_URL}/reset-password",
            "data": {"full_name": full_name},
        },
    )
    user_id, _ = read_user(invited)
    if not user_id:
        raise RuntimeError("Supabase invitation did not return a user ID")

    supabase.table("profiles").upsert({
        "id": user_id,
        "email": email,
        "full_name": full_name or None,
        "language": "en",
        "account_status": "active",
        "updated_at": now_iso(),
    }, on_conflict="id").execute()
    return user_id, "invited"


def apply_organization_entitlement(user_id: str, organization: dict[str, Any]) -> None:
    plan_code = organization.get("default_plan_code")
    access_ends_at = organization.get("access_ends_at")
    if not plan_code:
        return

    plan_response = supabase.table("plans").select("*").eq("code", plan_code).limit(1).execute()
    if not plan_response.data:
        raise RuntimeError(f"Unknown organization plan: {plan_code}")
    plan = plan_response.data[0]

    # Corporate access is stored in a dedicated organization entitlement row.
    # The account's personal Stripe entitlement remains untouched.
    supabase.table("organization_entitlements").upsert({
        "organization_id": organization["id"],
        "user_id": user_id,
        "plan_code": plan_code,
        "status": "active",
        "program_access": bool(plan.get("program_access")),
        "workout_access": bool(plan.get("workout_access")),
        "history_access": bool(plan.get("history_access", True)),
        "report_access": bool(plan.get("report_access", True)),
        "valid_from": now_iso(),
        "valid_until": access_ends_at,
        "updated_at": now_iso(),
    }, on_conflict="organization_id,user_id").execute()

    credits = max(0, int(plan.get("screenings_per_cycle") or 0))
    if credits:
        source = f"organization:{organization['id']}"
        cycle_end = access_ends_at or (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
        existing_cycle = (
            supabase.table("screening_credit_cycles")
            .select("id,credits_granted,credits_used")
            .eq("user_id", user_id)
            .eq("source", source)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if existing_cycle.data:
            cycle = existing_cycle.data[0]
            supabase.table("screening_credit_cycles").update({
                "cycle_end": cycle_end,
                "grace_expires_at": cycle_end,
                "credits_granted": max(credits, int(cycle.get("credits_granted") or 0), int(cycle.get("credits_used") or 0)),
                "updated_at": now_iso(),
            }).eq("id", cycle["id"]).execute()
        else:
            supabase.table("screening_credit_cycles").insert({
                "user_id": user_id,
                "subscription_id": None,
                "source": source,
                "cycle_start": now_iso(),
                "cycle_end": cycle_end,
                "grace_expires_at": cycle_end,
                "credits_granted": credits,
                "credits_used": 0,
                "updated_at": now_iso(),
            }).execute()


def process_row(row: dict[str, Any], organization: dict[str, Any]) -> str:
    row_id = str(row["id"])
    email = str(row.get("email") or "").strip().lower()
    full_name = str(row.get("full_name") or "").strip()

    existing_member = (
        supabase.table("organization_members")
        .select("id,user_id,status")
        .eq("organization_id", organization["id"])
        .ilike("invited_email", email)
        .limit(1)
        .execute()
    )
    if existing_member.data:
        member = existing_member.data[0]
        supabase.table("bulk_import_rows").update({
            "status": "duplicate",
            "error_message": "Already belongs to this organization",
            "user_id": member.get("user_id"),
            "processed_at": now_iso(),
        }).eq("id", row_id).execute()
        return "duplicate"

    user_id, invite_state = invite_or_find(email, full_name)
    member = supabase.table("organization_members").insert({
        "organization_id": organization["id"],
        "user_id": user_id,
        "invited_email": email,
        "full_name": full_name or None,
        "department": row.get("department"),
        "cohort": row.get("cohort"),
        "status": "active" if invite_state == "existing_user" else "invited",
        "invited_at": now_iso(),
        "accepted_at": now_iso() if invite_state == "existing_user" else None,
        "metadata": row.get("metadata") or {},
        "updated_at": now_iso(),
    }).execute()
    if not member.data:
        raise RuntimeError("Unable to create organization membership")

    apply_organization_entitlement(user_id, organization)

    supabase.table("bulk_import_rows").update({
        "status": "success",
        "error_message": None,
        "user_id": user_id,
        "processed_at": now_iso(),
    }).eq("id", row_id).execute()
    return "success"


def recalculate_job(job_id: str) -> None:
    rows = supabase.table("bulk_import_rows").select("status").eq("job_id", job_id).execute()
    statuses = [str(row.get("status") or "") for row in rows.data or []]
    pending = sum(1 for status in statuses if status == "pending")
    processing = sum(1 for status in statuses if status == "processing")
    success = sum(1 for status in statuses if status == "success")
    failed = sum(1 for status in statuses if status in {"failed", "invalid"})
    duplicates = sum(1 for status in statuses if status == "duplicate")
    processed = len(statuses) - pending - processing
    completed = pending == 0 and processing == 0
    supabase.table("bulk_import_jobs").update({
        "status": "completed_with_errors" if completed and failed else ("completed" if completed else "processing"),
        "processed_rows": processed,
        "success_rows": success,
        "failed_rows": failed,
        "duplicate_rows": duplicates,
        "started_at": now_iso(),
        "completed_at": now_iso() if completed else None,
        "updated_at": now_iso(),
    }).eq("id", job_id).execute()


def process_next_job() -> bool:
    jobs = (
        supabase.table("bulk_import_jobs")
        .select("*")
        .in_("status", ["queued", "processing"])
        .order("created_at")
        .limit(1)
        .execute()
    )
    if not jobs.data:
        return False

    job = jobs.data[0]
    job_id = str(job["id"])
    organization_response = supabase.table("organizations").select("*").eq("id", job["organization_id"]).limit(1).execute()
    if not organization_response.data:
        supabase.table("bulk_import_jobs").update({
            "status": "failed", "errors_json": [{"error": "Organization not found"}],
            "completed_at": now_iso(), "updated_at": now_iso(),
        }).eq("id", job_id).execute()
        return True
    organization = organization_response.data[0]

    supabase.table("bulk_import_jobs").update({
        "status": "processing", "started_at": job.get("started_at") or now_iso(), "updated_at": now_iso(),
    }).eq("id", job_id).execute()

    rows = (
        supabase.table("bulk_import_rows")
        .select("*")
        .eq("job_id", job_id)
        .eq("status", "pending")
        .order("row_number")
        .limit(BATCH_SIZE)
        .execute()
    )
    for row in rows.data or []:
        row_id = str(row["id"])
        claimed = (
            supabase.table("bulk_import_rows")
            .update({"status": "processing"})
            .eq("id", row_id)
            .eq("status", "pending")
            .execute()
        )
        if not claimed.data:
            continue
        try:
            process_row(row, organization)
        except Exception as exc:
            supabase.table("bulk_import_rows").update({
                "status": "failed", "error_message": str(exc)[:1000], "processed_at": now_iso(),
            }).eq("id", row_id).execute()

    recalculate_job(job_id)
    return True


if __name__ == "__main__":
    print(f"FlexiLab Operator Worker started (batch={BATCH_SIZE}, poll={POLL_SECONDS}s)", flush=True)
    while True:
        try:
            worked = process_next_job()
            if not worked:
                time.sleep(POLL_SECONDS)
        except Exception as exc:
            print(f"Operator worker error: {exc}", flush=True)
            time.sleep(POLL_SECONDS)
