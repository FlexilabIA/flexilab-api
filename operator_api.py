from __future__ import annotations

import csv
import io
import json
import os
import hashlib
import hmac
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import stripe
from fastapi import APIRouter, BackgroundTasks, Depends, File, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field


APP_VERSION = os.environ.get("FLEXILAB_APP_VERSION", "V100.0").strip() or "V100.0"
FRONTEND_URL = os.environ.get(
    "FRONTEND_URL",
    "https://flexi-move-lab.lovable.app",
).rstrip("/")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "").strip()
stripe.api_key = STRIPE_SECRET_KEY


PLAN_INCLUDED_SCREENING_CREDITS: dict[str, int] = {
    "pro_monthly": 1,
    "pro_three_month": 3,
    "pro_annual": 12,
    "corporate": 0,
}


def _plan_label(plan_code: str) -> str:
    return {
        "pro_monthly": "FlexiLab Pro — monthly access",
        "pro_three_month": "FlexiLab Pro — 3 months",
        "pro_annual": "FlexiLab Pro — annual access",
        "corporate": "FlexiLab Corporate",
    }.get(plan_code, plan_code.replace("_", " ").title())


def _send_access_grant_email(*, email: str, full_name: str, plan_code: str, expires_at: str, credits: int) -> bool:
    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    if not api_key:
        return False
    sender = os.environ.get("FLEXILAB_EMAIL_FROM", "FlexiLab <support@flexilab.app>").strip()
    name = (full_name or "there").strip()
    plan = _plan_label(plan_code)
    subject = "Your FlexiLab access has been activated"
    html = f"""
    <div style="font-family:Arial,sans-serif;line-height:1.6;color:#0f172a">
      <h2>Your FlexiLab access is active</h2>
      <p>Hello {name},</p>
      <p>Your <strong>{plan}</strong> access has been granted by the FlexiLab team.</p>
      <p><strong>Access valid until:</strong> {expires_at[:10]}<br>
      <strong>Screening credits included:</strong> {credits}</p>
      <p>You can sign in and begin using your access now.</p>
      <p><a href="{FRONTEND_URL}/login">Open FlexiLab</a></p>
      <p>FlexiLab Support<br>support@flexilab.app</p>
    </div>
    """
    body = json.dumps({"from": sender, "to": [email], "subject": subject, "html": html}).encode("utf-8")
    request = urllib.request.Request(
        "https://api.resend.com/emails",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            return 200 <= int(response.status) < 300
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return False

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "super_admin": {"*"},
    "admin": {
        "dashboard.read", "users.read", "users.write", "trainers.read", "trainers.write",
        "screenings.read", "screenings.write", "finance.read", "vouchers.read", "vouchers.write",
        "organizations.read", "organizations.write", "imports.read", "imports.write",
        "health.read", "audit.read", "roles.write",
    },
    "operations": {
        "dashboard.read", "users.read", "users.write", "trainers.read", "trainers.write",
        "screenings.read", "screenings.write", "organizations.read", "organizations.write",
        "imports.read", "imports.write", "health.read", "audit.read",
    },
    "support": {
        "dashboard.read", "users.read", "users.write", "trainers.read", "screenings.read",
        "organizations.read", "imports.read", "health.read", "audit.read",
    },
    "finance": {
        "dashboard.read", "users.read", "finance.read", "vouchers.read", "vouchers.write",
        "audit.read", "health.read",
    },
    "clinical_reviewer": {
        "dashboard.read", "users.read", "trainers.read", "screenings.read", "audit.read",
    },
    "organization_manager": {
        "dashboard.read", "users.read", "organizations.read", "organizations.write",
        "imports.read", "imports.write", "health.read",
    },
}



def _normalize_email_domains(values: Optional[list[str]]) -> list[str]:
    domains: list[str] = []
    for raw in values or []:
        domain = str(raw or "").strip().lower()
        if domain.startswith("@"):
            domain = domain[1:]
        if not domain or "@" in domain or "." not in domain:
            raise ValueError(f"Invalid email domain: {raw}")
        if not re.fullmatch(r"[a-z0-9.-]+", domain):
            raise ValueError(f"Invalid email domain: {raw}")
        if domain not in domains:
            domains.append(domain)
    return domains


def _hash_qvct_access_code(code: str) -> tuple[str, str]:
    normalized = str(code or "").strip()
    if len(normalized) < 6 or len(normalized) > 64:
        raise ValueError("QVCT access code must contain between 6 and 64 characters.")
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", normalized.encode("utf-8"), salt, 200_000)
    return salt.hex(), digest.hex()


def _safe_organization_row(row: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not row:
        return row
    safe = {
        key: value for key, value in row.items()
        if key not in {"qvct_access_code_hash", "qvct_access_code_salt"}
    }
    safe["qvct_access_code_configured"] = bool(row.get("qvct_access_code_hash"))
    return safe

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime] = None) -> str:
    return (value or _now()).isoformat()


def _safe_search(value: Optional[str]) -> str:
    return re.sub(r"[^a-zA-Z0-9@._+\- ]", "", str(value or "").strip())[:120]


def _page_values(page: int, page_size: int) -> tuple[int, int, int, int]:
    safe_page = max(1, int(page or 1))
    safe_size = max(1, min(int(page_size or 25), 100))
    start = (safe_page - 1) * safe_size
    return safe_page, safe_size, start, start + safe_size - 1


def _count(response: Any, fallback_rows: list[dict[str, Any]]) -> int:
    raw = getattr(response, "count", None)
    try:
        return int(raw) if raw is not None else len(fallback_rows)
    except Exception:
        return len(fallback_rows)


def _user_from_response(response: Any) -> dict[str, str]:
    user = getattr(response, "user", None)
    if user is None and isinstance(response, dict):
        user = response.get("user")
    user_id = getattr(user, "id", None)
    email = getattr(user, "email", None)
    if isinstance(user, dict):
        user_id = user_id or user.get("id")
        email = email or user.get("email")
    if not user_id:
        raise ValueError("Authenticated user is incomplete")
    return {"id": str(user_id), "email": str(email or "").strip().lower()}


class UserStatusUpdate(BaseModel):
    account_status: str
    reason: str = Field(min_length=3, max_length=500)


class TrainerStatusUpdate(BaseModel):
    status: str
    reason: str = Field(min_length=3, max_length=500)


class VoucherCreate(BaseModel):
    code: str = Field(min_length=3, max_length=40)
    name: Optional[str] = Field(default=None, max_length=120)
    description: Optional[str] = Field(default=None, max_length=500)
    promotion_type: str
    percent_off: Optional[float] = None
    amount_off_cents: Optional[int] = None
    currency: str = "eur"
    granted_plan_code: Optional[str] = None
    granted_access_days: Optional[int] = None
    screening_credits: Optional[int] = None
    new_users_only: bool = False
    max_redemptions: Optional[int] = None
    expires_at: Optional[str] = None


class VoucherStatusUpdate(BaseModel):
    is_active: bool
    reason: str = Field(min_length=3, max_length=500)


class OrganizationCreate(BaseModel):
    name: str = Field(min_length=2, max_length=180)
    slug: Optional[str] = Field(default=None, max_length=100)
    default_plan_code: Optional[str] = None
    access_ends_at: Optional[str] = None
    enrollment_limit: Optional[int] = Field(default=None, ge=1, le=100000)
    qvct_access_code: Optional[str] = Field(default=None, min_length=6, max_length=64)
    qvct_allowed_email_domains: list[str] = Field(default_factory=list, max_length=20)


class OrganizationUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=180)
    status: Optional[str] = None
    default_plan_code: Optional[str] = None
    access_ends_at: Optional[str] = None
    enrollment_limit: Optional[int] = Field(default=None, ge=1, le=100000)
    qvct_access_code: Optional[str] = Field(default=None, min_length=6, max_length=64)
    qvct_allowed_email_domains: Optional[list[str]] = Field(default=None, max_length=20)


class GrantCreate(BaseModel):
    grant_type: str
    quantity: Optional[int] = None
    granted_plan_code: Optional[str] = None
    expires_at: Optional[str] = None
    reason: str = Field(min_length=3, max_length=500)


class RoleUpdate(BaseModel):
    role: str
    reason: str = Field(min_length=3, max_length=500)


class RetryJobRequest(BaseModel):
    reason: str = Field(min_length=3, max_length=500)


class UserInviteCreate(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    full_name: Optional[str] = Field(default=None, max_length=160)
    language: str = "en"
    account_type: str = "user"
    plan_code: Optional[str] = None
    screening_credits: int = Field(default=0, ge=0, le=1000)
    access_ends_at: Optional[str] = None
    reason: str = Field(min_length=3, max_length=500)


class CreditAdjustment(BaseModel):
    quantity: int = Field(ge=-1000, le=1000)
    expires_at: Optional[str] = None
    reason: str = Field(min_length=3, max_length=500)


class AccessRevoke(BaseModel):
    reason: str = Field(min_length=3, max_length=500)


def create_operator_router(
    supabase_client: Any,
    health_provider: Optional[Callable[[], dict[str, Any]]] = None,
) -> APIRouter:
    router = APIRouter(prefix="/operator", tags=["operator"])

    def require_user(authorization: Optional[str] = Header(default=None)) -> dict[str, str]:
        if supabase_client is None:
            raise HTTPException(status_code=503, detail="Supabase is not configured.")
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing authentication token.")
        token = authorization.split(" ", 1)[1].strip()
        try:
            return _user_from_response(supabase_client.auth.get_user(token))
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid or expired authentication token.")

    def operator_context(user: dict[str, str]) -> dict[str, Any]:
        response = (
            supabase_client.table("admin_roles")
            .select("user_id,role,granted_at,revoked_at")
            .eq("user_id", user["id"])
            .is_("revoked_at", "null")
            .limit(1)
            .execute()
        )
        if not response.data:
            raise HTTPException(status_code=403, detail="FlexiLab Operator access required.")
        role = str(response.data[0].get("role") or "").strip()
        permissions = ROLE_PERMISSIONS.get(role, set())
        if not permissions:
            raise HTTPException(status_code=403, detail="This Operator role is not recognized.")
        return {**user, "role": role, "permissions": sorted(permissions)}

    def permission(name: str):
        def dependency(user: dict[str, str] = Depends(require_user)) -> dict[str, Any]:
            operator = operator_context(user)
            allowed = set(operator["permissions"])
            if "*" not in allowed and name not in allowed:
                raise HTTPException(status_code=403, detail=f"Missing Operator permission: {name}")
            return operator
        return dependency

    def audit(
        operator: dict[str, Any],
        action: str,
        *,
        target_user_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        before_data: Optional[dict[str, Any]] = None,
        after_data: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        try:
            supabase_client.table("audit_logs").insert({
                "actor_user_id": operator["id"],
                "target_user_id": target_user_id,
                "action": action,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "before_data": before_data,
                "after_data": after_data,
                "metadata": metadata or {},
            }).execute()
        except Exception:
            # An Operator action must not fail solely because secondary audit storage is unavailable.
            pass

    def table_count(table: str, filters: Optional[list[tuple[str, str, Any]]] = None) -> int:
        query = supabase_client.table(table).select("*", count="exact")
        for method, column, value in filters or []:
            query = getattr(query, method)(column, value)
        response = query.limit(1).execute()
        return int(getattr(response, "count", 0) or 0)

    @router.get("/me")
    def me(user: dict[str, str] = Depends(require_user)):
        operator = operator_context(user)
        return {
            "user_id": operator["id"],
            "email": operator["email"],
            "role": operator["role"],
            "permissions": operator["permissions"],
            "app_version": APP_VERSION,
        }

    @router.get("/dashboard")
    def dashboard(operator: dict[str, Any] = Depends(permission("dashboard.read"))):
        failures = (
            supabase_client.table("analysis_jobs")
            .select("id,session_id,test_type,error_message,created_at,completed_at")
            .eq("status", "failed")
            .order("created_at", desc=True)
            .limit(8)
            .execute()
        )
        webhook_failures = (
            supabase_client.table("stripe_webhook_events")
            .select("event_id,event_type,error_message,received_at,updated_at")
            .eq("status", "failed")
            .order("received_at", desc=True)
            .limit(8)
            .execute()
        )
        return {
            "generated_at": _iso(),
            "counts": {
                "users": table_count("profiles"),
                "trainers": table_count("trainer_profiles", [("eq", "status", "active")]),
                "trainer_clients": table_count("trainer_clients", [("neq", "status", "archived")]),
                "completed_screenings": table_count("sessions", [("eq", "status", "completed")]),
                "screenings_today": table_count("sessions", [("gte", "created_at", _now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat())]),
                "queued_analysis": table_count("analysis_jobs", [("eq", "status", "queued")]),
                "processing_analysis": table_count("analysis_jobs", [("eq", "status", "processing")]),
                "failed_analysis": table_count("analysis_jobs", [("eq", "status", "failed")]),
                "active_subscriptions": table_count("subscriptions", [("eq", "status", "active")]),
                "active_organizations": table_count("organizations", [("eq", "status", "active")]),
                "pending_imports": table_count("bulk_import_jobs", [("in_", "status", ["queued", "processing"])]),
            },
            "recent_analysis_failures": failures.data or [],
            "recent_webhook_failures": webhook_failures.data or [],
        }

    @router.get("/users")
    def users(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=25, ge=1, le=100),
        q: Optional[str] = None,
        account_status: Optional[str] = None,
        operator: dict[str, Any] = Depends(permission("users.read")),
    ):
        safe_page, safe_size, start, end = _page_values(page, page_size)
        query = supabase_client.table("profiles").select(
            "id,email,full_name,language,account_status,onboarding_completed,created_at,updated_at",
            count="exact",
        )
        search = _safe_search(q)
        if search:
            query = query.or_(f"email.ilike.%{search}%,full_name.ilike.%{search}%")
        if account_status:
            query = query.eq("account_status", account_status)
        response = query.order("created_at", desc=True).range(start, end).execute()
        rows = response.data or []
        user_ids = [str(row.get("id")) for row in rows if row.get("id")]
        entitlements: dict[str, dict[str, Any]] = {}
        trainer_ids: set[str] = set()
        operator_roles: dict[str, str] = {}
        if user_ids:
            ent = supabase_client.table("entitlements").select(
                "user_id,plan_code,status,program_access,workout_access,valid_until"
            ).in_("user_id", user_ids).execute()
            entitlements = {str(row.get("user_id")): row for row in ent.data or []}
            trainers = supabase_client.table("trainer_profiles").select("user_id,status").in_("user_id", user_ids).execute()
            trainer_ids = {str(row.get("user_id")) for row in trainers.data or [] if row.get("status") == "active"}
            roles = (
                supabase_client.table("admin_roles")
                .select("user_id,role,revoked_at")
                .in_("user_id", user_ids)
                .is_("revoked_at", "null")
                .execute()
            )
            operator_roles = {
                str(row.get("user_id")): str(row.get("role") or "")
                for row in roles.data or []
                if row.get("user_id") and row.get("role")
            }
        items = [
            {
                **row,
                "entitlement": entitlements.get(str(row.get("id"))),
                "is_trainer": str(row.get("id")) in trainer_ids,
                "operator_role": operator_roles.get(str(row.get("id"))),
            }
            for row in rows
        ]
        return {"items": items, "page": safe_page, "page_size": safe_size, "total": _count(response, rows)}

    @router.get("/users/{user_id}")
    def user_detail(
        user_id: str,
        operator: dict[str, Any] = Depends(permission("users.read")),
    ):
        profile = supabase_client.table("profiles").select(
            "id,email,full_name,language,account_status,onboarding_completed,created_at,updated_at"
        ).eq("id", user_id).limit(1).execute()
        if not profile.data:
            raise HTTPException(status_code=404, detail="User profile not found.")
        entitlement = supabase_client.table("entitlements").select(
            "user_id,plan_code,source,status,program_access,workout_access,history_access,report_access,valid_from,valid_until,updated_at"
        ).eq("user_id", user_id).limit(1).execute()
        cycles = supabase_client.table("screening_credit_cycles").select(
            "id,source,cycle_start,cycle_end,grace_expires_at,credits_granted,credits_used,created_at"
        ).eq("user_id", user_id).order("created_at", desc=True).limit(100).execute()
        cycle_rows = cycles.data or []
        available = sum(max(0, int(r.get("credits_granted") or 0) - int(r.get("credits_used") or 0)) for r in cycle_rows)
        grants = supabase_client.table("admin_grants").select(
            "id,grant_type,granted_plan_code,quantity,starts_at,expires_at,reason,granted_by,revoked_at,created_at"
        ).eq("user_id", user_id).order("created_at", desc=True).limit(50).execute()
        latest = supabase_client.table("sessions").select(
            "id,status,composite_score,created_at"
        ).eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        trainer = supabase_client.table("trainer_profiles").select(
            "user_id,status,company_name,created_at"
        ).eq("user_id", user_id).limit(1).execute()
        return {
            "profile": profile.data[0],
            "entitlement": entitlement.data[0] if entitlement.data else None,
            "screening_credits": {"available": available, "cycles": cycle_rows},
            "grants": grants.data or [],
            "latest_screening": latest.data[0] if latest.data else None,
            "trainer": trainer.data[0] if trainer.data else None,
        }

    @router.post("/users/invite")
    def invite_user(
        payload: UserInviteCreate,
        operator: dict[str, Any] = Depends(permission("users.write")),
    ):
        email = payload.email.strip().lower()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            raise HTTPException(status_code=422, detail="Enter a valid email address.")
        existing = supabase_client.table("profiles").select("id,email").ilike("email", email).limit(1).execute()
        if existing.data:
            raise HTTPException(status_code=409, detail="A FlexiLab account already exists for this email.")
        language = "fr" if payload.language.strip().lower() == "fr" else "en"
        account_type = payload.account_type.strip().lower()
        if account_type not in {"user", "trainer"}:
            raise HTTPException(status_code=422, detail="Account type must be user or trainer.")
        try:
            invited = supabase_client.auth.admin.invite_user_by_email(
                email,
                options={
                    "redirect_to": f"{FRONTEND_URL}/reset-password",
                    "data": {"full_name": payload.full_name or "", "language": language},
                },
            )
            invited_user = getattr(invited, "user", None)
            user_id = str(getattr(invited_user, "id", "") or "")
            if not user_id and isinstance(invited, dict):
                raw_user = invited.get("user") or {}
                user_id = str(raw_user.get("id") or "")
            if not user_id:
                raise RuntimeError("Supabase did not return an invited user ID")
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Unable to send invitation: {exc}")
        supabase_client.table("profiles").upsert({
            "id": user_id, "email": email, "full_name": payload.full_name,
            "language": language, "account_status": "active", "updated_at": _iso(),
        }, on_conflict="id").execute()
        if account_type == "trainer":
            supabase_client.table("trainer_profiles").upsert({
                "user_id": user_id, "status": "active", "updated_at": _iso(),
            }, on_conflict="user_id").execute()
        access_end = payload.access_ends_at or (_now() + timedelta(days=30)).isoformat()
        if payload.plan_code:
            supabase_client.table("admin_grants").insert({
                "user_id": user_id, "grant_type": "pro_access",
                "granted_plan_code": payload.plan_code, "starts_at": _iso(),
                "expires_at": access_end, "reason": payload.reason, "granted_by": operator["id"],
            }).execute()
            supabase_client.table("entitlements").upsert({
                "user_id": user_id, "plan_code": payload.plan_code, "source": "admin", "status": "active",
                "program_access": True, "workout_access": True, "history_access": True,
                "report_access": True, "can_generate_program": True, "valid_from": _iso(),
                "valid_until": access_end, "updated_at": _iso(),
            }, on_conflict="user_id").execute()
        if payload.screening_credits > 0:
            supabase_client.table("admin_grants").insert({
                "user_id": user_id, "grant_type": "screening_credit", "quantity": payload.screening_credits,
                "starts_at": _iso(), "expires_at": access_end, "reason": payload.reason, "granted_by": operator["id"],
            }).execute()
            supabase_client.table("screening_credit_cycles").insert({
                "user_id": user_id, "subscription_id": None, "source": "operator_grant",
                "cycle_start": _iso(), "cycle_end": access_end, "grace_expires_at": access_end,
                "credits_granted": payload.screening_credits, "credits_used": 0,
            }).execute()
        audit(operator, "operator.user_invited", target_user_id=user_id, entity_type="profile", entity_id=user_id, after_data={
            "email": email, "account_type": account_type, "plan_code": payload.plan_code,
            "screening_credits": payload.screening_credits, "access_ends_at": access_end,
        }, metadata={"reason": payload.reason})
        return {"user_id": user_id, "email": email, "invite_sent": True}

    @router.post("/users/{user_id}/credits/adjust")
    def adjust_credits(
        user_id: str,
        payload: CreditAdjustment,
        operator: dict[str, Any] = Depends(permission("users.write")),
    ):
        if payload.quantity == 0:
            raise HTTPException(status_code=422, detail="Quantity cannot be zero.")
        profile = supabase_client.table("profiles").select("id,email").eq("id", user_id).limit(1).execute()
        if not profile.data:
            raise HTTPException(status_code=404, detail="User profile not found.")
        expires_at = payload.expires_at or (_now() + timedelta(days=3650)).isoformat()
        if payload.quantity > 0:
            supabase_client.table("screening_credit_cycles").insert({
                "user_id": user_id, "subscription_id": None, "source": "operator_grant",
                "cycle_start": _iso(), "cycle_end": expires_at, "grace_expires_at": expires_at,
                "credits_granted": payload.quantity, "credits_used": 0,
            }).execute()
        else:
            remaining = abs(payload.quantity)
            cycles = supabase_client.table("screening_credit_cycles").select(
                "id,credits_granted,credits_used,cycle_end,source"
            ).eq("user_id", user_id).gt("cycle_end", _iso()).order("cycle_end", desc=False).execute()
            total_available = sum(max(0, int(r.get("credits_granted") or 0)-int(r.get("credits_used") or 0)) for r in cycles.data or [])
            if total_available < remaining:
                raise HTTPException(status_code=409, detail=f"Only {total_available} unused credits are available.")
            for row in cycles.data or []:
                available = max(0, int(row.get("credits_granted") or 0)-int(row.get("credits_used") or 0))
                take = min(available, remaining)
                if take:
                    supabase_client.table("screening_credit_cycles").update({
                        "credits_used": int(row.get("credits_used") or 0) + take, "updated_at": _iso(),
                    }).eq("id", row["id"]).execute()
                    remaining -= take
                if remaining == 0:
                    break
        grant = supabase_client.table("admin_grants").insert({
            "user_id": user_id, "grant_type": "screening_credit", "quantity": payload.quantity,
            "starts_at": _iso(), "expires_at": expires_at, "reason": payload.reason, "granted_by": operator["id"],
        }).execute()
        audit(operator, "operator.credits_adjusted", target_user_id=user_id, entity_type="screening_credit",
              entity_id=str((grant.data or [{}])[0].get("id") or ""), after_data={"quantity": payload.quantity}, metadata={"reason": payload.reason})
        return {"quantity": payload.quantity}

    @router.post("/users/{user_id}/access/revoke")
    def revoke_manual_access(
        user_id: str,
        payload: AccessRevoke,
        operator: dict[str, Any] = Depends(permission("users.write")),
    ):
        entitlement = supabase_client.table("entitlements").select("*").eq("user_id", user_id).limit(1).execute()
        current = entitlement.data[0] if entitlement.data else None
        if not current or str(current.get("source") or "") != "admin":
            raise HTTPException(status_code=409, detail="Only manually granted access can be revoked here. Paid access is protected.")
        updated = supabase_client.table("entitlements").update({
            "plan_code": "free", "source": "admin", "status": "active",
            "program_access": False, "workout_access": False, "can_generate_program": False,
            "valid_until": _iso(), "updated_at": _iso(),
        }).eq("user_id", user_id).execute()
        supabase_client.table("admin_grants").update({"revoked_at": _iso()}).eq("user_id", user_id).is_("revoked_at", "null").neq("grant_type", "screening_credit").execute()
        audit(operator, "operator.manual_access_revoked", target_user_id=user_id, entity_type="entitlement", entity_id=user_id, before_data=current, after_data=(updated.data or [None])[0], metadata={"reason": payload.reason})
        return {"entitlement": (updated.data or [None])[0]}

    @router.patch("/users/{user_id}/status")
    def update_user_status(
        user_id: str,
        payload: UserStatusUpdate,
        operator: dict[str, Any] = Depends(permission("users.write")),
    ):
        status = payload.account_status.strip().lower()
        if status not in {"active", "suspended", "deleted_pending"}:
            raise HTTPException(status_code=422, detail="Invalid account status.")
        existing = supabase_client.table("profiles").select("*").eq("id", user_id).limit(1).execute()
        if not existing.data:
            raise HTTPException(status_code=404, detail="User profile not found.")
        updated = supabase_client.table("profiles").update({"account_status": status, "updated_at": _iso()}).eq("id", user_id).execute()
        audit(operator, "operator.user_status_updated", target_user_id=user_id, entity_type="profile", entity_id=user_id, before_data=existing.data[0], after_data=(updated.data or [None])[0], metadata={"reason": payload.reason})
        return {"profile": (updated.data or [None])[0]}

    @router.post("/users/{user_id}/grants")
    def create_grant(
        user_id: str,
        payload: GrantCreate,
        background_tasks: BackgroundTasks,
        operator: dict[str, Any] = Depends(permission("users.write")),
    ):
        grant_type = payload.grant_type.strip().lower()
        if grant_type not in {"pro_access", "screening_credit", "program_access", "workout_access"}:
            raise HTTPException(status_code=422, detail="Invalid grant type.")
        expires_at = payload.expires_at or (_now() + timedelta(days=30)).isoformat()
        grant_row = {
            "user_id": user_id,
            "grant_type": grant_type,
            "granted_plan_code": payload.granted_plan_code,
            "quantity": payload.quantity,
            "starts_at": _iso(),
            "expires_at": expires_at,
            "reason": payload.reason,
            "granted_by": operator["id"],
        }
        result = supabase_client.table("admin_grants").insert(grant_row).execute()
        if grant_type == "screening_credit":
            quantity = max(1, min(int(payload.quantity or 1), 1000))
            supabase_client.table("screening_credit_cycles").insert({
                "user_id": user_id,
                "subscription_id": None,
                "source": "operator_grant",
                "cycle_start": _iso(),
                "cycle_end": expires_at,
                "grace_expires_at": expires_at,
                "credits_granted": quantity,
                "credits_used": 0,
            }).execute()
        else:
            profile = supabase_client.table("profiles").select("id,email,full_name").eq("id", user_id).limit(1).execute()
            if not profile.data:
                raise HTTPException(status_code=404, detail="User profile not found.")
            existing_ent = supabase_client.table("entitlements").select("*").eq("user_id", user_id).limit(1).execute()
            current = existing_ent.data[0] if existing_ent.data else {}
            plan_code = payload.granted_plan_code or current.get("plan_code") or "free"
            next_ent = {
                "user_id": user_id,
                "plan_code": plan_code,
                "source": "admin",
                "status": "active",
                "program_access": grant_type in {"pro_access", "program_access"} or bool(current.get("program_access")),
                "workout_access": grant_type in {"pro_access", "workout_access"} or bool(current.get("workout_access")),
                "history_access": True,
                "report_access": True,
                "can_generate_program": grant_type in {"pro_access", "program_access"} or bool(current.get("can_generate_program")),
                "valid_from": _iso(),
                "valid_until": expires_at,
                "updated_at": _iso(),
            }
            supabase_client.table("entitlements").upsert(next_ent, on_conflict="user_id").execute()

            included_credits = PLAN_INCLUDED_SCREENING_CREDITS.get(str(plan_code), 0) if grant_type == "pro_access" else 0
            if included_credits > 0:
                supabase_client.table("screening_credit_cycles").insert({
                    "user_id": user_id,
                    "subscription_id": None,
                    "source": "operator_plan_grant",
                    "cycle_start": _iso(),
                    "cycle_end": expires_at,
                    "grace_expires_at": expires_at,
                    "credits_granted": included_credits,
                    "credits_used": 0,
                }).execute()
                supabase_client.table("admin_grants").insert({
                    "user_id": user_id,
                    "grant_type": "screening_credit",
                    "quantity": included_credits,
                    "starts_at": _iso(),
                    "expires_at": expires_at,
                    "reason": f"Included with {plan_code} access: {payload.reason}",
                    "granted_by": operator["id"],
                }).execute()

            recipient = profile.data[0]
            if grant_type == "pro_access" and recipient.get("email"):
                background_tasks.add_task(
                    _send_access_grant_email,
                    email=str(recipient.get("email")),
                    full_name=str(recipient.get("full_name") or ""),
                    plan_code=str(plan_code),
                    expires_at=expires_at,
                    credits=included_credits,
                )
        audit(operator, "operator.grant_created", target_user_id=user_id, entity_type="admin_grant", entity_id=str((result.data or [{}])[0].get("id") or ""), after_data={**grant_row, "included_screening_credits": locals().get("included_credits", 0)})
        return {"grant": (result.data or [grant_row])[0], "included_screening_credits": locals().get("included_credits", 0), "email_queued": bool(grant_type == "pro_access" and os.environ.get("RESEND_API_KEY"))}

    @router.put("/users/{user_id}/role")
    def update_role(
        user_id: str,
        payload: RoleUpdate,
        operator: dict[str, Any] = Depends(permission("roles.write")),
    ):
        role = payload.role.strip().lower()
        if role not in ROLE_PERMISSIONS:
            raise HTTPException(status_code=422, detail="Invalid Operator role.")
        previous = supabase_client.table("admin_roles").select("*").eq("user_id", user_id).limit(1).execute()
        result = supabase_client.table("admin_roles").upsert({
            "user_id": user_id,
            "role": role,
            "granted_by": operator["id"],
            "granted_at": _iso(),
            "revoked_at": None,
        }, on_conflict="user_id").execute()
        audit(operator, "operator.role_updated", target_user_id=user_id, entity_type="admin_role", entity_id=user_id, before_data=(previous.data or [None])[0], after_data=(result.data or [None])[0], metadata={"reason": payload.reason})
        return {"role": (result.data or [None])[0]}

    @router.delete("/users/{user_id}/role")
    def revoke_role(
        user_id: str,
        reason: str = Query(min_length=3, max_length=500),
        operator: dict[str, Any] = Depends(permission("roles.write")),
    ):
        previous = (
            supabase_client.table("admin_roles")
            .select("*")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if not previous.data:
            raise HTTPException(status_code=404, detail="Operator role not found.")
        result = (
            supabase_client.table("admin_roles")
            .update({"revoked_at": _iso()})
            .eq("user_id", user_id)
            .execute()
        )
        audit(
            operator,
            "operator.role_revoked",
            target_user_id=user_id,
            entity_type="admin_role",
            entity_id=user_id,
            before_data=previous.data[0],
            after_data=(result.data or [None])[0],
            metadata={"reason": reason},
        )
        return {"role": (result.data or [None])[0]}

    @router.get("/trainers")
    def trainers(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=25, ge=1, le=100),
        q: Optional[str] = None,
        status: Optional[str] = None,
        operator: dict[str, Any] = Depends(permission("trainers.read")),
    ):
        safe_page, safe_size, start, end = _page_values(page, page_size)
        query = supabase_client.table("trainer_profiles").select("*", count="exact")
        search = _safe_search(q)
        if search:
            query = query.or_(f"full_name.ilike.%{search}%,company_name.ilike.%{search}%")
        if status:
            query = query.eq("status", status)
        response = query.order("created_at", desc=True).range(start, end).execute()
        rows = response.data or []
        trainer_ids = [str(row.get("user_id")) for row in rows if row.get("user_id")]
        emails: dict[str, str] = {}
        if trainer_ids:
            profiles = supabase_client.table("profiles").select("id,email").in_("id", trainer_ids).execute()
            emails = {str(row.get("id")): str(row.get("email") or "") for row in profiles.data or []}
        return {"items": [{**row, "email": emails.get(str(row.get("user_id")), "")} for row in rows], "page": safe_page, "page_size": safe_size, "total": _count(response, rows)}

    @router.patch("/trainers/{trainer_id}/status")
    def update_trainer_status(
        trainer_id: str,
        payload: TrainerStatusUpdate,
        operator: dict[str, Any] = Depends(permission("trainers.write")),
    ):
        status = payload.status.strip().lower()
        if status not in {"active", "suspended"}:
            raise HTTPException(status_code=422, detail="Invalid Trainer status.")
        before = supabase_client.table("trainer_profiles").select("*").eq("user_id", trainer_id).limit(1).execute()
        if not before.data:
            raise HTTPException(status_code=404, detail="Trainer not found.")
        result = supabase_client.table("trainer_profiles").update({"status": status, "updated_at": _iso()}).eq("user_id", trainer_id).execute()
        audit(operator, "operator.trainer_status_updated", target_user_id=trainer_id, entity_type="trainer_profile", entity_id=trainer_id, before_data=before.data[0], after_data=(result.data or [None])[0], metadata={"reason": payload.reason})
        return {"trainer": (result.data or [None])[0]}

    @router.get("/screenings")
    def screenings(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=25, ge=1, le=100),
        status: Optional[str] = None,
        trainer_id: Optional[str] = None,
        user_id: Optional[str] = None,
        q: Optional[str] = None,
        operator: dict[str, Any] = Depends(permission("screenings.read")),
    ):
        safe_page, safe_size, start, end = _page_values(page, page_size)
        query = supabase_client.table("sessions").select(
            "id,user_id,user_email,status,created_at,composite_score,trainer_id,performed_by_user_id,trainer_client_link_id,credit_owner_user_id",
            count="exact",
        )
        if status:
            query = query.eq("status", status)
        if trainer_id:
            query = query.eq("trainer_id", trainer_id)
        if user_id:
            query = query.eq("user_id", user_id)
        search = _safe_search(q)
        if search:
            query = query.ilike("user_email", f"%{search}%")
        response = query.order("created_at", desc=True).range(start, end).execute()
        rows = response.data or []
        session_ids = [str(row.get("id")) for row in rows if row.get("id")]
        test_counts: dict[str, int] = {}
        if session_ids:
            screening_rows = supabase_client.table("screenings").select("session_id,test_type").in_("session_id", session_ids).execute()
            unique: dict[str, set[str]] = {}
            for item in screening_rows.data or []:
                unique.setdefault(str(item.get("session_id")), set()).add(str(item.get("test_type")))
            test_counts = {sid: len(values) for sid, values in unique.items()}
        items = [{**row, "completed_test_count": test_counts.get(str(row.get("id")), 0)} for row in rows]
        return {"items": items, "page": safe_page, "page_size": safe_size, "total": _count(response, rows)}

    @router.post("/analysis-jobs/{job_id}/retry")
    def retry_analysis_job(
        job_id: str,
        payload: RetryJobRequest,
        operator: dict[str, Any] = Depends(permission("screenings.write")),
    ):
        before = supabase_client.table("analysis_jobs").select("*").eq("id", job_id).limit(1).execute()
        if not before.data:
            raise HTTPException(status_code=404, detail="Analysis job not found.")
        result = supabase_client.table("analysis_jobs").update({
            "status": "queued", "started_at": None, "completed_at": None, "error_message": None,
        }).eq("id", job_id).execute()
        audit(operator, "operator.analysis_job_retried", entity_type="analysis_job", entity_id=job_id, before_data=before.data[0], after_data=(result.data or [None])[0], metadata={"reason": payload.reason})
        return {"job": (result.data or [None])[0]}

    @router.get("/finance")
    def finance(operator: dict[str, Any] = Depends(permission("finance.read"))):
        subscriptions = supabase_client.table("subscriptions").select(
            "id,user_id,plan_code,status,current_period_start,current_period_end,cancel_at_period_end,created_at"
        ).order("created_at", desc=True).limit(50).execute()
        webhooks = supabase_client.table("stripe_webhook_events").select(
            "event_id,event_type,status,error_message,received_at,processed_at"
        ).order("received_at", desc=True).limit(50).execute()
        return {
            "summary": {
                "active": table_count("subscriptions", [("eq", "status", "active")]),
                "past_due": table_count("subscriptions", [("eq", "status", "past_due")]),
                "canceled": table_count("subscriptions", [("eq", "status", "canceled")]),
                "failed_webhooks": table_count("stripe_webhook_events", [("eq", "status", "failed")]),
            },
            "subscriptions": subscriptions.data or [],
            "webhooks": webhooks.data or [],
        }

    @router.get("/vouchers")
    def vouchers(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=25, ge=1, le=100),
        operator: dict[str, Any] = Depends(permission("vouchers.read")),
    ):
        safe_page, safe_size, start, end = _page_values(page, page_size)
        response = supabase_client.table("promotion_codes").select("*", count="exact").order("created_at", desc=True).range(start, end).execute()
        rows = response.data or []
        return {"items": rows, "page": safe_page, "page_size": safe_size, "total": _count(response, rows)}

    @router.post("/vouchers")
    def create_voucher(
        payload: VoucherCreate,
        operator: dict[str, Any] = Depends(permission("vouchers.write")),
    ):
        promotion_type = payload.promotion_type.strip().lower()
        if promotion_type not in {"percent_discount", "fixed_discount", "free_access", "screening_credit"}:
            raise HTTPException(status_code=422, detail="Invalid promotion type.")
        code = re.sub(r"[^A-Z0-9_-]", "", payload.code.upper())
        if len(code) < 3:
            raise HTTPException(status_code=422, detail="Voucher code must contain at least 3 valid characters.")
        existing = supabase_client.table("promotion_codes").select("id").ilike("code", code).limit(1).execute()
        if existing.data:
            raise HTTPException(status_code=409, detail="This voucher code already exists.")

        provider_id = None
        provider = "flexilab"
        if promotion_type in {"percent_discount", "fixed_discount"}:
            if not STRIPE_SECRET_KEY:
                raise HTTPException(status_code=503, detail="Stripe is not configured.")
            coupon_params: dict[str, Any] = {"duration": "once", "name": payload.name or code}
            if promotion_type == "percent_discount":
                if payload.percent_off is None or not 0 < float(payload.percent_off) <= 100:
                    raise HTTPException(status_code=422, detail="percent_off must be between 0 and 100.")
                coupon_params["percent_off"] = float(payload.percent_off)
            else:
                if payload.amount_off_cents is None or int(payload.amount_off_cents) <= 0:
                    raise HTTPException(status_code=422, detail="amount_off_cents must be positive.")
                coupon_params["amount_off"] = int(payload.amount_off_cents)
                coupon_params["currency"] = payload.currency.lower()
            try:
                coupon = stripe.Coupon.create(**coupon_params)
                promo_args: dict[str, Any] = {"code": code, "active": True}
                if payload.max_redemptions:
                    promo_args["max_redemptions"] = int(payload.max_redemptions)
                if payload.expires_at:
                    promo_args["expires_at"] = int(datetime.fromisoformat(payload.expires_at.replace("Z", "+00:00")).timestamp())
                try:
                    promotion = stripe.PromotionCode.create(
                        promotion={"type": "coupon", "coupon": coupon.id},
                        **promo_args,
                    )
                except Exception:
                    promotion = stripe.PromotionCode.create(coupon=coupon.id, **promo_args)
                provider_id = str(promotion.id)
                provider = "stripe"
            except stripe.StripeError as exc:
                raise HTTPException(status_code=502, detail=getattr(exc, "user_message", None) or str(exc))

        row = {
            "code": code,
            "name": payload.name,
            "description": payload.description,
            "promotion_type": promotion_type,
            "percent_off": payload.percent_off,
            "amount_off_cents": payload.amount_off_cents,
            "currency": payload.currency.lower(),
            "granted_plan_code": payload.granted_plan_code,
            "granted_access_days": payload.granted_access_days,
            "screening_credits": payload.screening_credits,
            "new_users_only": payload.new_users_only,
            "max_redemptions": payload.max_redemptions,
            "starts_at": _iso(),
            "expires_at": payload.expires_at,
            "is_active": True,
            "provider": provider,
            "provider_promotion_id": provider_id,
            "metadata": {"created_by": operator["id"], "app_version": APP_VERSION},
        }
        result = supabase_client.table("promotion_codes").insert(row).execute()
        audit(operator, "operator.voucher_created", entity_type="promotion_code", entity_id=str((result.data or [{}])[0].get("id") or ""), after_data=row)
        return {"voucher": (result.data or [row])[0]}

    @router.patch("/vouchers/{voucher_id}")
    def update_voucher_status(
        voucher_id: str,
        payload: VoucherStatusUpdate,
        operator: dict[str, Any] = Depends(permission("vouchers.write")),
    ):
        before = supabase_client.table("promotion_codes").select("*").eq("id", voucher_id).limit(1).execute()
        if not before.data:
            raise HTTPException(status_code=404, detail="Voucher not found.")
        voucher = before.data[0]
        provider_id = voucher.get("provider_promotion_id")
        if voucher.get("provider") == "stripe" and provider_id and STRIPE_SECRET_KEY:
            try:
                stripe.PromotionCode.modify(str(provider_id), active=payload.is_active)
            except stripe.StripeError as exc:
                raise HTTPException(status_code=502, detail=getattr(exc, "user_message", None) or str(exc))
        result = supabase_client.table("promotion_codes").update({"is_active": payload.is_active, "updated_at": _iso()}).eq("id", voucher_id).execute()
        audit(operator, "operator.voucher_status_updated", entity_type="promotion_code", entity_id=voucher_id, before_data=voucher, after_data=(result.data or [None])[0], metadata={"reason": payload.reason})
        return {"voucher": (result.data or [None])[0]}

    @router.get("/organizations")
    def organizations(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=25, ge=1, le=100),
        q: Optional[str] = None,
        operator: dict[str, Any] = Depends(permission("organizations.read")),
    ):
        safe_page, safe_size, start, end = _page_values(page, page_size)
        query = supabase_client.table("organizations").select("*", count="exact")
        search = _safe_search(q)
        if search:
            query = query.or_(f"name.ilike.%{search}%,slug.ilike.%{search}%")
        response = query.order("created_at", desc=True).range(start, end).execute()
        rows = response.data or []
        org_ids = [str(row.get("id")) for row in rows if row.get("id")]
        member_counts: dict[str, int] = {}
        enrolled_counts: dict[str, int] = {}
        if org_ids:
            members = supabase_client.table("organization_members").select("organization_id,status").in_("organization_id", org_ids).execute()
            for row in members.data or []:
                oid = str(row.get("organization_id"))
                member_counts[oid] = member_counts.get(oid, 0) + 1
                if str(row.get("status") or "").lower() == "active":
                    enrolled_counts[oid] = enrolled_counts.get(oid, 0) + 1
        items = []
        for row in rows:
            oid = str(row.get("id"))
            enrolled = enrolled_counts.get(oid, 0)
            limit = row.get("enrollment_limit")
            remaining = max(int(limit) - enrolled, 0) if limit is not None else None
            safe_row = {
                key: value for key, value in row.items()
                if key not in {"qvct_access_code_hash", "qvct_access_code_salt"}
            }
            items.append({
                **safe_row,
                "qvct_access_code_configured": bool(row.get("qvct_access_code_hash")),
                "member_count": member_counts.get(oid, 0),
                "enrolled_count": enrolled,
                "remaining_enrollments": remaining,
            })
        return {"items": items, "page": safe_page, "page_size": safe_size, "total": _count(response, rows)}

    @router.get("/organizations/{organization_id}/qvct-report")
    def organization_qvct_report(
        organization_id: str,
        language: str = Query(default="en", pattern="^(fr|en)$"),
        operator: dict[str, Any] = Depends(permission("organizations.read")),
    ):
        fr = language == "fr"
        organization_response = (
            supabase_client.table("organizations")
            .select("id,name,slug,status,default_plan_code,access_ends_at,enrollment_limit,created_at")
            .eq("id", organization_id).limit(1).execute()
        )
        if not organization_response.data:
            raise HTTPException(status_code=404, detail="Organization not found.")
        organization = organization_response.data[0]
        if organization.get("default_plan_code") != "corporate":
            raise HTTPException(status_code=422, detail="QVCT reporting is available only for QVCT organizations.")

        members = (supabase_client.table("organization_members")
            .select("id,user_id,status").eq("organization_id", organization_id)
            .eq("status", "active").execute()).data or []
        user_ids = [str(r.get("user_id")) for r in members if r.get("user_id")]
        user_id_set = set(user_ids)

        qrows = (supabase_client.table("corporate_qvct_responses").select("*")
            .eq("organization_id", organization_id).order("submitted_at", desc=True).execute()).data or []
        latest_qvct: dict[str, dict[str, Any]] = {}
        for row in qrows:
            uid = str(row.get("user_id") or "")
            if uid and uid in user_id_set and uid not in latest_qvct:
                latest_qvct[uid] = row

        session_rows: list[dict[str, Any]] = []
        if user_ids:
            session_rows = (supabase_client.table("sessions")
                .select("id,user_id,status,created_at,composite_score,posture_score,shoulder_right_score,shoulder_left_score,squat_score,aslr_right_score,aslr_left_score")
                .in_("user_id", user_ids).order("created_at", desc=True).limit(5000).execute()).data or []
        session_ids = [str(r.get("id")) for r in session_rows if r.get("id")]
        tests_by_session: dict[str, set[str]] = {}
        if session_ids:
            srows = (supabase_client.table("screenings").select("session_id,test_type")
                .in_("session_id", session_ids).limit(20000).execute()).data or []
            for row in srows:
                sid, tt = str(row.get("session_id") or ""), str(row.get("test_type") or "")
                if sid and tt: tests_by_session.setdefault(sid, set()).add(tt)
        latest_screening: dict[str, dict[str, Any]] = {}
        for row in session_rows:
            uid, sid = str(row.get("user_id") or ""), str(row.get("id") or "")
            if not uid or uid in latest_screening: continue
            if str(row.get("status") or "").lower() == "completed" or len(tests_by_session.get(sid, set())) >= 6:
                latest_screening[uid] = row

        def num(v: Any) -> Optional[float]:
            try: return None if v is None else float(v)
            except Exception: return None
        def avg(vals: list[Optional[float]]) -> Optional[float]:
            clean=[float(v) for v in vals if v is not None]
            return round(sum(clean)/len(clean),1) if clean else None
        def median(vals: list[Optional[float]]) -> Optional[float]:
            clean=sorted(float(v) for v in vals if v is not None)
            if not clean: return None
            n=len(clean); mid=n//2
            return round(clean[mid] if n%2 else (clean[mid-1]+clean[mid])/2,1)
        def pct(c:int,t:int)->float: return round(c/t*100,1) if t else 0.0
        def label(en:str, fr_text:str)->str: return fr_text if fr else en

        completed=list(latest_screening.values())
        def row_domain(row: dict[str,Any], key:str)->Optional[float]:
            if key=="overall": return num(row.get("composite_score"))
            if key=="posture": return num(row.get("posture_score"))
            if key=="shoulder_right": return num(row.get("shoulder_right_score"))
            if key=="shoulder_left": return num(row.get("shoulder_left_score"))
            if key=="shoulders": return avg([num(row.get("shoulder_right_score")),num(row.get("shoulder_left_score"))])
            if key=="lower_body": return num(row.get("squat_score"))
            if key=="aslr_right": return num(row.get("aslr_right_score"))
            if key=="aslr_left": return num(row.get("aslr_left_score"))
            if key=="posterior_chain": return avg([num(row.get("aslr_right_score")),num(row.get("aslr_left_score"))])
            return None

        specs=[
            ("overall",label("Overall movement","Mouvement global")),
            ("posture",label("Posture / cervical-thoracic","Posture / cervical-thoracique")),
            ("shoulder_right",label("Right shoulder","Épaule droite")),
            ("shoulder_left",label("Left shoulder","Épaule gauche")),
            ("shoulders",label("Shoulder mobility","Mobilité des épaules")),
            ("lower_body",label("Lower-body functional mobility","Mobilité fonctionnelle du bas du corps")),
            ("aslr_right",label("Right posterior chain","Chaîne postérieure droite")),
            ("aslr_left",label("Left posterior chain","Chaîne postérieure gauche")),
            ("posterior_chain",label("Posterior-chain mobility","Mobilité de la chaîne postérieure")),
        ]
        movement_domains=[]
        for key, lbl in specs:
            vals=[row_domain(r,key) for r in completed]; clean=[v for v in vals if v is not None]
            preferred=sum(1 for v in clean if v>=80); attention=sum(1 for v in clean if 70<=v<80); priority=sum(1 for v in clean if v<70)
            movement_domains.append({"key":key,"label":lbl,"average":avg(clean),"median":median(clean),"count":len(clean),
                "preferred_percent":pct(preferred,len(clean)),"attention_percent":pct(attention,len(clean)),"priority_percent":pct(priority,len(clean))})
        md={d["key"]:d["average"] for d in movement_domains}
        movement_profile={"overall":md.get("overall"),"posture":md.get("posture"),"shoulders":md.get("shoulders"),"lower_body":md.get("lower_body"),"posterior_chain":md.get("posterior_chain")}
        priorities=sorted([{"key":d["key"],"label":d["label"],"count":round(d["priority_percent"]*d["count"]/100),"total":d["count"],"percent":d["priority_percent"],"average_score":d["average"]}
            for d in movement_domains if d["key"] in {"posture","shoulders","lower_body","posterior_chain"}], key=lambda x:x["percent"], reverse=True)

        asym_users=0; asym_total=0
        for r in completed:
            sr,sl,ar,al=[num(r.get(k)) for k in ("shoulder_right_score","shoulder_left_score","aslr_right_score","aslr_left_score")]
            checks=[]
            if sr is not None and sl is not None: checks.append(abs(sr-sl)>=15)
            if ar is not None and al is not None: checks.append(abs(ar-al)>=15)
            if checks:
                asym_total+=1
                if any(checks): asym_users+=1
        asymmetry_percent=pct(asym_users,asym_total)

        questionnaire_rows=list(latest_qvct.values()); qt=len(questionnaire_rows)
        def resp_pct(field:str, accepted:set[str])->float: return pct(sum(1 for r in questionnaire_rows if str(r.get(field) or "") in accepted),qt)
        high_sitting=resp_pct("sitting_hours",{"6to8","gt8"}); high_screen=resp_pct("screen_hours",{"6to8","gt8"})
        infrequent=resp_pct("movement_breaks",{"60to90","gt90"}); laptop=resp_pct("screen_setup",{"laptop"}); below=resp_pct("screen_height",{"below"})
        limited_chair=resp_pct("chair_support",{"partial","little"}); reaching=resp_pct("keyboard_mouse",{"reach","laptop"}); low_walk=resp_pct("daily_walking",{"lt30"})
        discomfort=resp_pct("desk_discomfort",{"occasional","frequent"}); frequent_discomfort=resp_pct("desk_discomfort",{"frequent"})
        long_sitting=resp_pct("longest_sitting",{"60to90","gt90"}); low_activity=resp_pct("physical_activity",{"0","1to2"}); no_sitstand=resp_pct("sit_stand_desk",{"never"})
        low_comfort=resp_pct("workstation_comfort",{"1","2"}); low_energy=resp_pct("end_day_energy",{"1","2"}); low_movement_support=resp_pct("setup_encourages_movement",{"no","partly"})
        work_modes={}; discomfort_areas={}
        for r in questionnaire_rows:
            k=str(r.get("work_mode") or "");
            if k: work_modes[k]=work_modes.get(k,0)+1
            for a in r.get("discomfort_areas") or []:
                a=str(a or "");
                if a: discomfort_areas[a]=discomfort_areas.get(a,0)+1
        workplace={
            "high_sitting_percent":high_sitting,"high_screen_percent":high_screen,"infrequent_breaks_percent":infrequent,"laptop_only_percent":laptop,
            "screen_below_eye_level_percent":below,"limited_chair_support_percent":limited_chair,"input_reach_or_laptop_percent":reaching,
            "low_walking_percent":low_walk,"desk_discomfort_percent":discomfort,"frequent_discomfort_percent":frequent_discomfort,
            "long_uninterrupted_sitting_percent":long_sitting,"low_physical_activity_percent":low_activity,"no_sit_stand_desk_percent":no_sitstand,
            "low_workstation_comfort_percent":low_comfort,"low_end_day_energy_percent":low_energy,"low_movement_support_percent":low_movement_support,
            "work_mode_distribution":[{"key":k,"count":c,"percent":pct(c,qt)} for k,c in sorted(work_modes.items(),key=lambda x:x[1],reverse=True)],
            "discomfort_areas":[{"key":k,"count":c,"percent":pct(c,qt)} for k,c in sorted(discomfort_areas.items(),key=lambda x:x[1],reverse=True)],
        }

        hybrid=[r for r in questionnaire_rows if str(r.get("work_mode") or "") in {"hybrid","remote"}]
        ht=len(hybrid)
        def hp(field:str, accepted:set[str])->float: return pct(sum(1 for r in hybrid if str(r.get(field) or "") in accepted),ht)
        office_home={"eligible_count":ht,"available":ht>=3,
            "office":[
                {"key":"laptop_only","label":label("Laptop only","Portable uniquement"),"value":laptop},
                {"key":"screen_below","label":label("Screen below eye level","Écran sous le niveau des yeux"),"value":below},
                {"key":"limited_chair","label":label("Limited chair support","Soutien de chaise limité"),"value":limited_chair},
                {"key":"infrequent_breaks","label":label("Infrequent movement","Mouvements peu fréquents"),"value":infrequent},
                {"key":"long_sitting","label":label("60+ min uninterrupted sitting","60+ min assis sans interruption"),"value":long_sitting},
            ],
            "home":[
                {"key":"laptop_only","label":label("Laptop only","Portable uniquement"),"value":hp("home_screen_setup",{"laptop"})},
                {"key":"screen_below","label":label("Screen below eye level","Écran sous le niveau des yeux"),"value":hp("home_screen_height",{"below"})},
                {"key":"limited_chair","label":label("Limited chair support","Soutien de chaise limité"),"value":hp("home_chair_support",{"partial","little"})},
                {"key":"infrequent_breaks","label":label("Infrequent movement","Mouvements peu fréquents"),"value":hp("home_movement_breaks",{"60to90","gt90"})},
                {"key":"long_sitting","label":label("60+ min uninterrupted sitting","60+ min assis sans interruption"),"value":hp("home_longest_sitting",{"60to90","gt90"})},
            ],
            "less_active_at_home_percent":hp("home_activity_vs_office",{"less"}),
            "dedicated_home_workstation_percent":hp("home_work_location",{"dedicated_desk"}),
        }

        matched=[uid for uid in user_ids if uid in latest_qvct and uid in latest_screening]
        linkages=[]
        def add_linkage(field:str, exposed:set[str], domain:str, title_en:str,title_fr:str, exposed_en:str,exposed_fr:str, ref_en:str,ref_fr:str, action_en:str,action_fr:str):
            ev=[]; rv=[]
            for uid in matched:
                score=row_domain(latest_screening[uid],domain)
                if score is None: continue
                (ev if str(latest_qvct[uid].get(field) or "") in exposed else rv).append(score)
            if len(ev)>=3 and len(rv)>=3:
                ea,ra=avg(ev),avg(rv)
                linkages.append({"title":label(title_en,title_fr),"domain":domain,"exposed_label":label(exposed_en,exposed_fr),"reference_label":label(ref_en,ref_fr),
                    "exposed_count":len(ev),"reference_count":len(rv),"exposed_average":ea,"reference_average":ra,"difference":round((ea or 0)-(ra or 0),1),
                    "action":label(action_en,action_fr),"disclaimer":label("Observed association within this cohort; this does not establish causation.","Association observée dans cette cohorte ; elle n’établit pas de lien de causalité.")})
        add_linkage("movement_breaks",{"60to90","gt90"},"shoulders","Movement breaks and shoulder mobility","Pauses actives et mobilité des épaules","Infrequent breaks","Pauses peu fréquentes","Regularer breaks","Pauses plus régulières","Introduce short movement opportunities and guided upper-body mobility during screen-based work.","Introduire de courtes occasions de mouvement et des routines guidées du haut du corps pendant le travail sur écran.")
        add_linkage("screen_height",{"below"},"posture","Screen height and posture score","Hauteur d’écran et score postural","Below eye level","Sous le niveau des yeux","Other screen positions","Autres positions d’écran","Review screen height and prolonged laptop use; encourage adjustable setups where practical.","Revoir la hauteur d’écran et l’usage prolongé du portable ; favoriser des installations réglables lorsque possible.")
        add_linkage("sitting_hours",{"6to8","gt8"},"posterior_chain","Sitting exposure and posterior-chain mobility","Temps assis et mobilité de la chaîne postérieure","6+ hours sitting","6 h ou plus assis","Under 6 hours","Moins de 6 h","Increase position changes, walking opportunities and hip/posterior-chain mobility routines.","Augmenter les changements de position, les occasions de marche et les routines de mobilité des hanches/chaîne postérieure.")
        add_linkage("longest_sitting",{"60to90","gt90"},"overall","Uninterrupted sitting and overall movement","Assise ininterrompue et mouvement global","60+ min uninterrupted","60+ min sans interruption","Under 60 min","Moins de 60 min","Reduce long uninterrupted seated periods and create practical movement prompts.","Réduire les longues périodes assises sans interruption et créer des rappels de mouvement adaptés.")
        add_linkage("physical_activity",{"0","1to2"},"lower_body","Physical activity and lower-body mobility","Activité physique et mobilité du bas du corps","0–2 days/week","0–2 jours/semaine","3+ days/week","3+ jours/semaine","Support regular activity opportunities alongside workplace mobility actions.","Soutenir des occasions régulières d’activité en complément des actions de mobilité au travail.")
        add_linkage("home_activity_vs_office",{"less"},"overall","Home-working activity and movement score","Activité en télétravail et score de mouvement","Less active at home","Moins actif à domicile","Same or more active","Aussi ou plus actif","For hybrid teams, strengthen home-working movement prompts and workstation guidance.","Pour les équipes hybrides, renforcer les rappels de mouvement et les conseils d’installation à domicile.")

        recs=[]
        def rec(priority,title_en,title_fr,evidence_en,evidence_fr,action_en,action_fr): recs.append({"priority":priority,"title":label(title_en,title_fr),"evidence":label(evidence_en,evidence_fr),"action":label(action_en,action_fr)})
        if (md.get("shoulders") or 100)<72 or (priorities and priorities[0]["key"]=="shoulders"):
            rec("high","Upper-body mobility & workstation variation","Mobilité du haut du corps et variation du poste",f"Group shoulder mobility averages {md.get('shoulders') or 0:.0f}/100.",f"La mobilité moyenne des épaules est de {md.get('shoulders') or 0:.0f}/100.","Deploy 3–5 minute shoulder/thoracic mobility routines and vary arm position during the workday.","Déployer des routines de 3 à 5 minutes pour les épaules/le thorax et varier la position des bras au cours de la journée.")
        if below>=35 or (md.get("posture") or 100)<72:
            rec("high","Screen setup & postural variety","Installation écran et variété posturale",f"{below:.0f}% report a main screen below eye level; posture score averages {md.get('posture') or 0:.0f}/100.",f"{below:.0f}% déclarent un écran principal sous le niveau des yeux ; le score postural moyen est de {md.get('posture') or 0:.0f}/100.","Promote screen-height adjustment, external peripherals and frequent changes of sitting position.","Favoriser l’ajustement de la hauteur d’écran, les périphériques externes et les changements fréquents de position.")
        if high_sitting>=45 or infrequent>=35 or long_sitting>=35:
            rec("high","Reduce uninterrupted sedentary periods","Réduire les périodes sédentaires ininterrompues",f"{high_sitting:.0f}% report 6+ hours sitting and {long_sitting:.0f}% report 60+ minute uninterrupted sitting periods.",f"{high_sitting:.0f}% déclarent 6 h ou plus assis et {long_sitting:.0f}% des périodes assises ininterrompues de 60 min ou plus.","Introduce brief movement opportunities approximately every 60–90 minutes, adapted to operational constraints.","Introduire de brèves occasions de mouvement environ toutes les 60 à 90 minutes, adaptées aux contraintes opérationnelles.")
        if ht>=3 and (office_home["home"][0]["value"]>office_home["office"][0]["value"]+10 or office_home["less_active_at_home_percent"]>=35):
            rec("high","Hybrid-work home setup","Installation en télétravail",f"Home-working indicators show a less favorable setup or lower activity for a meaningful share of hybrid/remote respondents.",f"Les indicateurs de télétravail montrent une installation moins favorable ou une activité réduite pour une part significative des répondants hybrides/à distance.","Provide a home-workstation checklist, external-screen/peripheral guidance and remote-day movement prompts.","Fournir une checklist du poste à domicile, des conseils sur écran/périphériques externes et des rappels de mouvement les jours de télétravail.")
        if discomfort>=30:
            rec("medium","Targeted ergonomic follow-up","Suivi ergonomique ciblé",f"{discomfort:.0f}% report occasional or frequent workday-related discomfort.",f"{discomfort:.0f}% déclarent un inconfort occasionnel ou fréquent lié à la journée de travail.","Offer workstation self-check guidance and refer persistent/significant discomfort to qualified occupational-health or ergonomic professionals.","Proposer une auto-évaluation du poste et orienter les inconforts persistants/importants vers des professionnels qualifiés de santé au travail ou d’ergonomie.")
        if not recs:
            rec("maintain","Maintain healthy movement habits","Maintenir de bonnes habitudes de mouvement","Current group indicators do not show a dominant workplace movement priority.","Les indicateurs actuels du groupe ne montrent pas de priorité dominante.","Maintain movement variety, periodic workstation checks and reassess the cohort over time.","Maintenir la variété de mouvement, des vérifications périodiques du poste et réévaluer la cohorte dans le temps.")

        qc,sc,mc,ec=len(latest_qvct),len(latest_screening),len(matched),len(members); min_group=10
        return {"generated_at":_iso(),"language":language,"organization":organization,
            "sample":{"enrolled":ec,"questionnaire_completed":qc,"screening_completed":sc,"matched_completed":mc,
                "questionnaire_completion_percent":pct(qc,ec),"screening_completion_percent":pct(sc,ec),"full_completion_percent":pct(mc,ec),
                "minimum_group_size":min_group,"privacy_ready":mc>=min_group,"analysis_level":"full" if mc>=20 else "limited" if mc>=10 else "preview"},
            "movement_profile":movement_profile,"movement_domains":movement_domains,"priorities":priorities,"asymmetry_percent":asymmetry_percent,
            "workplace":workplace,"office_home":office_home,"linkages":linkages,"recommendations":recs,
            "method":{"screening_basis":label("Latest complete six-test FlexiLab screening per enrolled employee.","Dernier screening FlexiLab complet à six tests par salarié inscrit."),
                "questionnaire_basis":label("Latest QVCT questionnaire response per enrolled employee.","Dernière réponse au questionnaire QVCT par salarié inscrit."),
                "matching_basis":label("Authenticated Supabase user UUID within the organization.","UUID utilisateur Supabase authentifié au sein de l’organisation."),
                "privacy_note":label("Employer-facing reporting is aggregate. Individual names, emails and individual screening results are not included.","Le rapport destiné à l’employeur est agrégé. Les noms, e-mails et résultats individuels ne sont pas inclus."),
                "disclaimer":label("FlexiLab provides movement and workplace-wellbeing insights for educational and preventive purposes. It does not diagnose medical conditions or replace medical, occupational-health, physiotherapy or ergonomic assessment.","FlexiLab fournit des indicateurs de mouvement et de bien-être au travail à visée éducative et préventive. Il ne pose aucun diagnostic et ne remplace pas une évaluation médicale, de santé au travail, de physiothérapie ou ergonomique.")}}

    @router.post("/organizations")
    def create_organization(
        payload: OrganizationCreate,
        operator: dict[str, Any] = Depends(permission("organizations.write")),
    ):
        slug = payload.slug or re.sub(r"[^a-z0-9]+", "-", payload.name.lower()).strip("-")
        slug = slug[:100]
        row = {
            "name": payload.name.strip(), "slug": slug, "status": "active",
            "default_plan_code": payload.default_plan_code, "access_ends_at": payload.access_ends_at,
            "enrollment_limit": payload.enrollment_limit,
            "created_by": operator["id"], "updated_at": _iso(),
        }
        if payload.default_plan_code == "corporate":
            if not payload.qvct_access_code:
                raise HTTPException(status_code=422, detail="A QVCT employee access code is required.")
            try:
                salt, digest = _hash_qvct_access_code(payload.qvct_access_code)
                domains = _normalize_email_domains(payload.qvct_allowed_email_domains)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            row.update({
                "qvct_access_code_salt": salt,
                "qvct_access_code_hash": digest,
                "qvct_allowed_email_domains": domains,
            })
        try:
            result = supabase_client.table("organizations").insert(row).execute()
        except Exception as exc:
            raise HTTPException(status_code=409, detail=f"Unable to create organization: {exc}")
        created = (result.data or [row])[0]
        audit(operator, "operator.organization_created", entity_type="organization", entity_id=str(created.get("id") or ""), after_data={**row, "qvct_access_code_hash": "[stored]" if row.get("qvct_access_code_hash") else None, "qvct_access_code_salt": "[stored]" if row.get("qvct_access_code_salt") else None})
        return {"organization": _safe_organization_row(created)}

    @router.patch("/organizations/{organization_id}")
    def update_organization(
        organization_id: str,
        payload: OrganizationUpdate,
        operator: dict[str, Any] = Depends(permission("organizations.write")),
    ):
        before = supabase_client.table("organizations").select("*").eq("id", organization_id).limit(1).execute()
        if not before.data:
            raise HTTPException(status_code=404, detail="Organization not found.")
        raw_changes = {k: v for k, v in payload.model_dump().items() if v is not None}
        access_code = raw_changes.pop("qvct_access_code", None)
        domains_value = raw_changes.pop("qvct_allowed_email_domains", None)
        changes = raw_changes
        if "status" in changes and changes["status"] not in {"active", "suspended", "archived"}:
            raise HTTPException(status_code=422, detail="Invalid organization status.")
        try:
            if access_code:
                salt, digest = _hash_qvct_access_code(access_code)
                changes["qvct_access_code_salt"] = salt
                changes["qvct_access_code_hash"] = digest
            if domains_value is not None:
                changes["qvct_allowed_email_domains"] = _normalize_email_domains(domains_value)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        changes["updated_at"] = _iso()
        result = supabase_client.table("organizations").update(changes).eq("id", organization_id).execute()
        updated = (result.data or [None])[0]
        audit(operator, "operator.organization_updated", entity_type="organization", entity_id=organization_id, before_data=before.data[0], after_data=updated)
        return {"organization": _safe_organization_row(updated)}

    @router.post("/organizations/{organization_id}/imports")
    async def create_import(
        organization_id: str,
        file: UploadFile = File(...),
        operator: dict[str, Any] = Depends(permission("imports.write")),
    ):
        organization = supabase_client.table("organizations").select("id,status").eq("id", organization_id).limit(1).execute()
        if not organization.data:
            raise HTTPException(status_code=404, detail="Organization not found.")
        raw = await file.read()
        if len(raw) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="CSV file is too large. Maximum size is 5 MB.")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(status_code=422, detail="CSV must be UTF-8 encoded.")
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames or "email" not in {str(x).strip().lower() for x in reader.fieldnames}:
            raise HTTPException(status_code=422, detail="CSV requires an email column.")
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row_number, source in enumerate(reader, start=2):
            normalized = {str(k).strip().lower(): str(v or "").strip() for k, v in source.items()}
            email = normalized.get("email", "").lower()
            full_name = normalized.get("full_name") or normalized.get("name") or ""
            row_status = "pending"
            error_message = None
            if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
                row_status = "invalid"
                error_message = "Invalid email"
            elif email in seen:
                row_status = "duplicate"
                error_message = "Duplicate email inside CSV"
            seen.add(email)
            rows.append({
                "row_number": row_number, "email": email, "full_name": full_name[:180],
                "department": normalized.get("department", "")[:180] or None,
                "cohort": normalized.get("cohort", "")[:180] or None,
                "status": row_status, "error_message": error_message,
                "metadata": {k: v for k, v in normalized.items() if k not in {"email", "full_name", "name", "department", "cohort"} and v},
            })
            if len(rows) > 10000:
                raise HTTPException(status_code=413, detail="CSV cannot exceed 10,000 rows.")
        if not rows:
            raise HTTPException(status_code=422, detail="CSV contains no data rows.")
        invalid = sum(1 for row in rows if row["status"] == "invalid")
        duplicate = sum(1 for row in rows if row["status"] == "duplicate")
        job = supabase_client.table("bulk_import_jobs").insert({
            "organization_id": organization_id,
            "uploaded_by": operator["id"],
            "filename": file.filename or "corporate-import.csv",
            "status": "queued",
            "total_rows": len(rows),
            "processed_rows": invalid + duplicate,
            "success_rows": 0,
            "failed_rows": invalid,
            "duplicate_rows": duplicate,
            "errors_json": [],
            "updated_at": _iso(),
        }).execute()
        if not job.data:
            raise HTTPException(status_code=500, detail="Unable to create import job.")
        job_id = str(job.data[0]["id"])
        payload_rows = [{**row, "job_id": job_id, "organization_id": organization_id} for row in rows]
        for offset in range(0, len(payload_rows), 500):
            supabase_client.table("bulk_import_rows").insert(payload_rows[offset:offset + 500]).execute()
        audit(operator, "operator.bulk_import_queued", entity_type="bulk_import_job", entity_id=job_id, after_data={"organization_id": organization_id, "total_rows": len(rows), "filename": file.filename})
        return {"job": job.data[0], "preview": rows[:20], "invalid_rows": invalid, "duplicate_rows": duplicate}

    @router.get("/imports")
    def imports(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=25, ge=1, le=100),
        organization_id: Optional[str] = None,
        operator: dict[str, Any] = Depends(permission("imports.read")),
    ):
        safe_page, safe_size, start, end = _page_values(page, page_size)
        query = supabase_client.table("bulk_import_jobs").select("*", count="exact")
        if organization_id:
            query = query.eq("organization_id", organization_id)
        response = query.order("created_at", desc=True).range(start, end).execute()
        rows = response.data or []
        return {"items": rows, "page": safe_page, "page_size": safe_size, "total": _count(response, rows)}

    @router.get("/imports/{job_id}")
    def import_detail(
        job_id: str,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=100),
        operator: dict[str, Any] = Depends(permission("imports.read")),
    ):
        job = supabase_client.table("bulk_import_jobs").select("*").eq("id", job_id).limit(1).execute()
        if not job.data:
            raise HTTPException(status_code=404, detail="Import job not found.")
        safe_page, safe_size, start, end = _page_values(page, page_size)
        rows = supabase_client.table("bulk_import_rows").select("*", count="exact").eq("job_id", job_id).order("row_number").range(start, end).execute()
        return {"job": job.data[0], "rows": rows.data or [], "page": safe_page, "page_size": safe_size, "total": _count(rows, rows.data or [])}

    @router.post("/imports/{job_id}/retry")
    def retry_import(
        job_id: str,
        payload: RetryJobRequest,
        operator: dict[str, Any] = Depends(permission("imports.write")),
    ):
        job = supabase_client.table("bulk_import_jobs").select("*").eq("id", job_id).limit(1).execute()
        if not job.data:
            raise HTTPException(status_code=404, detail="Import job not found.")
        supabase_client.table("bulk_import_rows").update({"status": "pending", "error_message": None, "processed_at": None}).eq("job_id", job_id).eq("status", "failed").execute()
        result = supabase_client.table("bulk_import_jobs").update({"status": "queued", "updated_at": _iso(), "completed_at": None}).eq("id", job_id).execute()
        audit(operator, "operator.bulk_import_retried", entity_type="bulk_import_job", entity_id=job_id, metadata={"reason": payload.reason})
        return {"job": (result.data or [None])[0]}

    @router.get("/health")
    def health(operator: dict[str, Any] = Depends(permission("health.read"))):
        started = time.perf_counter()
        db_ok = True
        db_error = None
        try:
            supabase_client.table("profiles").select("id").limit(1).execute()
        except Exception as exc:
            db_ok = False
            db_error = str(exc)
        db_latency_ms = round((time.perf_counter() - started) * 1000, 1)
        extra = health_provider() if health_provider else {}
        now = _now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        last_24h_start = now - timedelta(hours=24)

        # Queue values are live. Completion/failure values use rolling windows so
        # historical development tests do not permanently degrade platform health.
        queued = table_count("analysis_jobs", [("eq", "status", "queued")])
        processing = table_count("analysis_jobs", [("eq", "status", "processing")])
        completed_today = table_count(
            "analysis_jobs",
            [("eq", "status", "completed"), ("gte", "created_at", _iso(today_start))],
        )
        failed_today = table_count(
            "analysis_jobs",
            [("eq", "status", "failed"), ("gte", "created_at", _iso(today_start))],
        )
        completed_last_24h = table_count(
            "analysis_jobs",
            [("eq", "status", "completed"), ("gte", "created_at", _iso(last_24h_start))],
        )
        failed_last_24h = table_count(
            "analysis_jobs",
            [("eq", "status", "failed"), ("gte", "created_at", _iso(last_24h_start))],
        )
        analysed_last_24h = completed_last_24h + failed_last_24h
        success_rate_last_24h = (
            round((completed_last_24h / analysed_last_24h) * 100, 1)
            if analysed_last_24h
            else 100.0
        )

        import_queued = table_count("bulk_import_jobs", [("in_", "status", ["queued", "processing"])])
        if not db_ok:
            status = "down"
        elif failed_last_24h >= 3:
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "checked_at": _iso(now),
            "app_version": APP_VERSION,
            "database": {"ok": db_ok, "latency_ms": db_latency_ms, "error": db_error},
            "analysis_queue": {
                "queued": queued,
                "processing": processing,
                "completed_today": completed_today,
                "failed_today": failed_today,
                "completed_last_24h": completed_last_24h,
                "failed_last_24h": failed_last_24h,
                "success_rate_last_24h": success_rate_last_24h,
            },
            "corporate_import_queue": {"queued_or_processing": import_queued},
            "integrations": {
                "stripe_secret_configured": bool(STRIPE_SECRET_KEY),
                "stripe_webhook_configured": bool(os.environ.get("STRIPE_WEBHOOK_SECRET")),
                "email_configured": bool(os.environ.get("RESEND_API_KEY") or os.environ.get("SMTP_PASSWORD")),
                "supabase_configured": bool(os.environ.get("SUPABASE_URL") and (os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY"))),
            },
            **extra,
        }

    @router.get("/audit")
    def audit_log(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=100),
        action: Optional[str] = None,
        actor_user_id: Optional[str] = None,
        target_user_id: Optional[str] = None,
        operator: dict[str, Any] = Depends(permission("audit.read")),
    ):
        safe_page, safe_size, start, end = _page_values(page, page_size)
        query = supabase_client.table("audit_logs").select("*", count="exact")
        if action:
            query = query.ilike("action", f"%{_safe_search(action)}%")
        if actor_user_id:
            query = query.eq("actor_user_id", actor_user_id)
        if target_user_id:
            query = query.eq("target_user_id", target_user_id)
        response = query.order("created_at", desc=True).range(start, end).execute()
        rows = response.data or []
        return {"items": rows, "page": safe_page, "page_size": safe_size, "total": _count(response, rows)}

    return router
