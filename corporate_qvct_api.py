from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

from screening_access import authenticated_user


class CorporateEnroll(BaseModel):
    organization_id: str = Field(min_length=36, max_length=36)
    full_name: Optional[str] = Field(default=None, max_length=160)


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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _active_organization(supabase_client, organization_id: str) -> dict[str, Any]:
    response = (
        supabase_client.table("organizations")
        .select("id,name,slug,status,access_ends_at")
        .eq("id", organization_id)
        .eq("status", "active")
        .limit(1)
        .execute()
    )
    if not response.data:
        raise HTTPException(status_code=404, detail="Organization is not available.")

    organization = response.data[0]
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
            .select("id,name,slug,access_ends_at")
            .eq("status", "active")
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
            items.append({"id": row.get("id"), "name": row.get("name"), "slug": row.get("slug")})
        return {"items": items}

    @router.post("/enroll")
    def enroll(
        payload: CorporateEnroll,
        authorization: Optional[str] = Header(default=None),
    ):
        user = authenticated_user(supabase_client, authorization)
        organization = _active_organization(supabase_client, payload.organization_id)
        email = user["email"]
        now = _now_iso()

        existing = (
            supabase_client.table("organization_members")
            .select("*")
            .eq("organization_id", organization["id"])
            .ilike("invited_email", email)
            .limit(1)
            .execute()
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
        }
        values = {
            key: _validated_choice(getattr(payload, key), choices, key)
            for key, choices in allowed.items()
        }
        allowed_areas = {"neck", "shoulders", "upper_back", "lower_back", "hips", "legs", "wrists_hands", "other"}
        areas = sorted({str(area).strip().lower() for area in payload.discomfort_areas if str(area).strip().lower() in allowed_areas})

        row = {
            "organization_id": organization["id"],
            "user_id": user["id"],
            "member_id": membership.data[0]["id"],
            "language": payload.language,
            **values,
            "discomfort_areas": areas,
            "submitted_at": _now_iso(),
        }
        result = supabase_client.table("corporate_qvct_responses").insert(row).execute()
        return {
            "response": (result.data or [row])[0],
            "organization": {"id": organization["id"], "name": organization["name"]},
        }

    return router
