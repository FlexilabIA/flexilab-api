from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

from screening_access import authenticated_user


class CorporateEnroll(BaseModel):
    organization_id: str = Field(min_length=36, max_length=36)
    full_name: Optional[str] = Field(default=None, max_length=160)
    access_code: str = Field(min_length=6, max_length=64)


class CorporateAccessValidate(BaseModel):
    organization_id: str = Field(min_length=36, max_length=36)
    email: str = Field(min_length=5, max_length=254)
    access_code: str = Field(min_length=6, max_length=64)


class QVCTResponseCreate(BaseModel):
    organization_id: str = Field(min_length=36, max_length=36)
    language: str = Field(default="en", pattern="^(fr|en)$")
    work_mode: str
    sitting_hours: str
    screen_hours: str
    movement_breaks: str
    screen_setup: str
    screen_height: str
    chair_support: str
    keyboard_mouse: str
    daily_walking: str
    desk_discomfort: str
    discomfort_areas: list[str] = Field(default_factory=list, max_length=8)
    physical_activity: str
    longest_sitting: str
    sit_stand_desk: str
    workstation_comfort: str
    end_day_energy: str
    setup_encourages_movement: str
    wfh_days: Optional[str] = None
    home_work_location: Optional[str] = None
    home_screen_setup: Optional[str] = None
    home_screen_height: Optional[str] = None
    home_chair_support: Optional[str] = None
    home_movement_breaks: Optional[str] = None
    home_activity_vs_office: Optional[str] = None
    home_longest_sitting: Optional[str] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _active_organization(supabase_client, organization_id: str) -> dict[str, Any]:
    response = (
        supabase_client.table("organizations")
        .select("id,name,slug,status,default_plan_code,access_ends_at,enrollment_limit,qvct_access_code_hash,qvct_access_code_salt,qvct_allowed_email_domains")
        .eq("id", organization_id)
        .eq("status", "active")
        .limit(1)
        .execute()
    )
    if not response.data:
        raise HTTPException(status_code=404, detail="Organization is not available.")

    organization = response.data[0]
    if organization.get("default_plan_code") != "corporate":
        raise HTTPException(status_code=404, detail="Organization is not available for QVCT assessment.")
    access_ends_at = organization.get("access_ends_at")
    if access_ends_at:
        try:
            end = datetime.fromisoformat(str(access_ends_at).replace("Z", "+00:00"))
            if end < datetime.now(timezone.utc):
                raise HTTPException(status_code=404, detail="Organization is not available.")
        except HTTPException:
            raise
        except Exception:
            pass
    return organization



def _verify_access_code(organization: dict[str, Any], access_code: str) -> None:
    expected = str(organization.get("qvct_access_code_hash") or "")
    salt_hex = str(organization.get("qvct_access_code_salt") or "")
    if not expected or not salt_hex:
        raise HTTPException(status_code=403, detail="This organization is not configured for employee access yet.")
    try:
        salt = bytes.fromhex(salt_hex)
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            str(access_code or "").strip().encode("utf-8"),
            salt,
            200_000,
        ).hex()
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid organization access code.")
    if not hmac.compare_digest(actual, expected):
        raise HTTPException(status_code=403, detail="Invalid organization access code.")


def _verify_email_domain(organization: dict[str, Any], email: str) -> None:
    domains = [str(value or "").strip().lower().lstrip("@") for value in (organization.get("qvct_allowed_email_domains") or [])]
    domains = [value for value in domains if value]
    if not domains:
        return
    normalized_email = str(email or "").strip().lower()
    domain = normalized_email.rsplit("@", 1)[-1] if "@" in normalized_email else ""
    if domain not in domains:
        raise HTTPException(status_code=403, detail="Please use an authorized organization email address.")

def _validated_choice(value: str, allowed: set[str], field: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in allowed:
        raise HTTPException(status_code=422, detail=f"Invalid {field} value.")
    return normalized


def create_corporate_qvct_router(supabase_client) -> APIRouter:
    router = APIRouter(prefix="/corporate", tags=["corporate-qvct"])

    @router.get("/organizations/search")
    def search_organizations(
        q: str = Query(min_length=3, max_length=80),
        limit: int = Query(default=8, ge=1, le=12),
    ):
        if supabase_client is None:
            raise HTTPException(status_code=503, detail="Corporate service is unavailable.")
        search = "".join(ch for ch in q.strip() if ch.isalnum() or ch in " .&'-")[:80]
        if len(search) < 3:
            return {"items": []}
        response = (
            supabase_client.table("organizations")
            .select("id,name,slug,default_plan_code,access_ends_at,qvct_access_code_hash,qvct_allowed_email_domains")
            .eq("status", "active")
            .eq("default_plan_code", "corporate")
            .ilike("name", f"%{search}%")
            .order("name")
            .limit(limit)
            .execute()
        )
        now = datetime.now(timezone.utc)
        items: list[dict[str, Any]] = []
        for row in response.data or []:
            access_ends_at = row.get("access_ends_at")
            if access_ends_at:
                try:
                    if datetime.fromisoformat(str(access_ends_at).replace("Z", "+00:00")) < now:
                        continue
                except Exception:
                    pass
            items.append({"id": row.get("id"), "name": row.get("name"), "slug": row.get("slug"), "requires_access_code": bool(row.get("qvct_access_code_hash")), "email_domain_restricted": bool(row.get("qvct_allowed_email_domains"))})
        return {"items": items}

    @router.post("/access/validate")
    def validate_access(payload: CorporateAccessValidate):
        organization = _active_organization(supabase_client, payload.organization_id)
        _verify_access_code(organization, payload.access_code)
        _verify_email_domain(organization, payload.email)
        return {"ok": True}

    @router.post("/enroll")
    def enroll(
        payload: CorporateEnroll,
        authorization: Optional[str] = Header(default=None),
    ):
        user = authenticated_user(supabase_client, authorization)
        organization = _active_organization(supabase_client, payload.organization_id)
        email = user["email"]
        now = _now_iso()

        _verify_access_code(organization, payload.access_code)
        _verify_email_domain(organization, email)

        existing = (
            supabase_client.table("organization_members")
            .select("*")
            .eq("organization_id", organization["id"])
            .ilike("invited_email", email)
            .limit(1)
            .execute()
        )
        # Existing enrolled employees may always return. The cap is only checked
        # when this request would consume a new active enrollment.
        existing_member = (existing.data or [None])[0]
        already_enrolled = bool(existing_member and str(existing_member.get("status") or "").lower() == "active")
        if not already_enrolled:
            enrollment_limit = organization.get("enrollment_limit")
            if enrollment_limit is not None:
                active_members = (
                    supabase_client.table("organization_members")
                    .select("id", count="exact")
                    .eq("organization_id", organization["id"])
                    .eq("status", "active")
                    .limit(1)
                    .execute()
                )
                enrolled_count = int(getattr(active_members, "count", 0) or 0)
                if enrolled_count >= int(enrollment_limit):
                    raise HTTPException(
                        status_code=409,
                        detail="This organization's employee enrollment limit has been reached.",
                    )

        changes = {
            "user_id": user["id"],
            "status": "active",
            "accepted_at": now,
            "updated_at": now,
        }
        if payload.full_name and payload.full_name.strip():
            changes["full_name"] = payload.full_name.strip()

        if existing.data:
            result = (
                supabase_client.table("organization_members")
                .update(changes)
                .eq("id", existing.data[0]["id"])
                .execute()
            )
        else:
            result = supabase_client.table("organization_members").insert({
                "organization_id": organization["id"],
                "user_id": user["id"],
                "invited_email": email,
                "full_name": payload.full_name.strip() if payload.full_name else None,
                "status": "active",
                "accepted_at": now,
                "metadata": {"source": "corporate_qvct_web"},
                "updated_at": now,
            }).execute()

        return {
            "organization": {"id": organization["id"], "name": organization["name"]},
            "membership": (result.data or [{}])[0],
        }

    @router.post("/qvct-responses")
    def save_qvct_response(
        payload: QVCTResponseCreate,
        authorization: Optional[str] = Header(default=None),
    ):
        user = authenticated_user(supabase_client, authorization)
        organization = _active_organization(supabase_client, payload.organization_id)

        membership = (
            supabase_client.table("organization_members")
            .select("id,status")
            .eq("organization_id", organization["id"])
            .eq("user_id", user["id"])
            .eq("status", "active")
            .limit(1)
            .execute()
        )
        if not membership.data:
            raise HTTPException(status_code=403, detail="Corporate enrollment is required first.")

        allowed = {
            "work_mode": {"office", "hybrid", "remote", "mobile", "other"},
            "sitting_hours": {"lt4", "4to6", "6to8", "gt8"},
            "screen_hours": {"lt4", "4to6", "6to8", "gt8"},
            "movement_breaks": {"lt30", "30to60", "60to90", "gt90"},
            "screen_setup": {"laptop", "single_monitor", "dual_monitor", "other"},
            "screen_height": {"below", "eye_level", "above", "unsure"},
            "chair_support": {"good", "partial", "little", "unsure"},
            "keyboard_mouse": {"comfortable", "reach", "laptop", "unsure"},
            "daily_walking": {"lt30", "30to60", "60to90", "gt90"},
            "desk_discomfort": {"none", "occasional", "frequent"},
            "physical_activity": {"0", "1to2", "3to4", "5plus"},
            "longest_sitting": {"lt30", "30to60", "60to90", "gt90"},
            "sit_stand_desk": {"regular", "sometimes", "never"},
            "workstation_comfort": {"1", "2", "3", "4", "5"},
            "end_day_energy": {"1", "2", "3", "4", "5"},
            "setup_encourages_movement": {"yes", "partly", "no"},
        }
        values = {
            key: _validated_choice(getattr(payload, key), choices, key)
            for key, choices in allowed.items()
        }
        allowed_areas = {"neck", "shoulders", "upper_back", "lower_back", "hips", "legs", "wrists_hands", "other"}
        areas = sorted({str(area).strip().lower() for area in payload.discomfort_areas if str(area).strip().lower() in allowed_areas})

        remote_values: dict[str, Any] = {
            "wfh_days": None,
            "home_work_location": None,
            "home_screen_setup": None,
            "home_screen_height": None,
            "home_chair_support": None,
            "home_movement_breaks": None,
            "home_activity_vs_office": None,
            "home_longest_sitting": None,
        }
        if values["work_mode"] in {"hybrid", "remote"}:
            remote_allowed = {
                "wfh_days": {"1", "2", "3", "4", "5"},
                "home_work_location": {"dedicated_desk", "dining_table", "sofa", "bed", "other"},
                "home_screen_setup": {"laptop", "single_monitor", "dual_monitor", "other"},
                "home_screen_height": {"below", "eye_level", "above", "unsure"},
                "home_chair_support": {"good", "partial", "little", "unsure"},
                "home_movement_breaks": {"lt30", "30to60", "60to90", "gt90"},
                "home_activity_vs_office": {"more", "same", "less"},
                "home_longest_sitting": {"lt30", "30to60", "60to90", "gt90"},
            }
            for key, choices in remote_allowed.items():
                raw = getattr(payload, key)
                if raw is None:
                    raise HTTPException(status_code=422, detail=f"Missing {key} value.")
                remote_values[key] = _validated_choice(raw, choices, key)

        row = {
            "organization_id": organization["id"],
            "user_id": user["id"],
            "member_id": membership.data[0]["id"],
            "language": payload.language,
            **values,
            **remote_values,
            "discomfort_areas": areas,
            "submitted_at": _now_iso(),
        }
        result = supabase_client.table("corporate_qvct_responses").insert(row).execute()

        # The QVCT package includes exactly one AI movement screening credit.
        # Grant it once per user + organization, even if the questionnaire is resubmitted.
        credit_source = f"qvct_assessment:{organization['id']}"
        existing_credit = (
            supabase_client.table("screening_credit_cycles")
            .select("id")
            .eq("user_id", user["id"])
            .eq("source", credit_source)
            .limit(1)
            .execute()
        )
        screening_credit_granted = False
        if not existing_credit.data:
            now = datetime.now(timezone.utc)
            credit_end = organization.get("access_ends_at") or (now + timedelta(days=365)).isoformat()
            supabase_client.table("screening_credit_cycles").insert({
                "user_id": user["id"],
                "subscription_id": None,
                "source": credit_source,
                "cycle_start": now.isoformat(),
                "cycle_end": credit_end,
                "grace_expires_at": credit_end,
                "credits_granted": 1,
                "credits_used": 0,
            }).execute()
            screening_credit_granted = True

        return {
            "response": (result.data or [row])[0],
            "organization": {"id": organization["id"], "name": organization["name"]},
            "screening_credit_granted": screening_credit_granted,
        }

    return router
