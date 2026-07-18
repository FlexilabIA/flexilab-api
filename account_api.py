from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field


class ProfileUpdate(BaseModel):
    full_name: Optional[str] = Field(default=None, max_length=120)
    language: Optional[str] = None
    program_notifications: Optional[bool] = None
    reassessment_notifications: Optional[bool] = None
    update_notifications: Optional[bool] = None


def create_account_router(supabase_client) -> APIRouter:
    router = APIRouter(tags=["account"])

    def require_user(
        authorization: Optional[str] = Header(default=None),
    ) -> dict[str, Any]:
        if supabase_client is None:
            raise HTTPException(
                status_code=503,
                detail="Supabase is not configured on the server.",
            )

        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(
                status_code=401,
                detail="Missing authentication token.",
            )

        access_token = authorization.split(" ", 1)[1].strip()
        if not access_token:
            raise HTTPException(
                status_code=401,
                detail="Missing authentication token.",
            )

        try:
            response = supabase_client.auth.get_user(access_token)
            user = getattr(response, "user", None)

            if user is None and isinstance(response, dict):
                user = response.get("user")

            user_id = getattr(user, "id", None)
            email = getattr(user, "email", None)

            if isinstance(user, dict):
                user_id = user_id or user.get("id")
                email = email or user.get("email")

            if not user_id:
                raise ValueError("Authenticated user has no id.")

            return {
                "id": str(user_id),
                "email": str(email or "").strip().lower(),
            }
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired authentication token.",
            )

    def get_profile_row(user: dict[str, Any]) -> dict[str, Any]:
        response = (
            supabase_client.table("profiles")
            .select("*")
            .eq("id", user["id"])
            .limit(1)
            .execute()
        )

        if response.data:
            return response.data[0]

        row = {
            "id": user["id"],
            "email": user["email"],
            "full_name": None,
            "language": "en",
            "account_status": "active",
            "program_notifications": True,
            "reassessment_notifications": True,
            "update_notifications": True,
        }

        created = supabase_client.table("profiles").insert(row).execute()
        if not created.data:
            raise HTTPException(
                status_code=500,
                detail="Unable to create the user profile.",
            )
        return created.data[0]

    @router.get("/me/profile")
    def get_my_profile(
        user: dict[str, Any] = Depends(require_user),
    ):
        profile = get_profile_row(user)

        return {
            "id": profile.get("id"),
            "email": user["email"] or profile.get("email"),
            "full_name": profile.get("full_name") or "",
            "language": profile.get("language") or "en",
            "account_status": profile.get("account_status") or "active",
            "onboarding_completed": bool(
                profile.get("onboarding_completed", False)
            ),
            "program_notifications": bool(
                profile.get("program_notifications", True)
            ),
            "reassessment_notifications": bool(
                profile.get("reassessment_notifications", True)
            ),
            "update_notifications": bool(
                profile.get("update_notifications", True)
            ),
            "created_at": profile.get("created_at"),
            "updated_at": profile.get("updated_at"),
        }

    @router.patch("/me/profile")
    def update_my_profile(
        payload: ProfileUpdate,
        user: dict[str, Any] = Depends(require_user),
    ):
        get_profile_row(user)

        changes: dict[str, Any] = {}

        if payload.full_name is not None:
            clean_name = payload.full_name.strip()
            if not clean_name:
                raise HTTPException(
                    status_code=422,
                    detail="Full name cannot be empty.",
                )
            changes["full_name"] = clean_name

        if payload.language is not None:
            clean_language = payload.language.strip().lower()
            if clean_language not in {"en", "fr"}:
                raise HTTPException(
                    status_code=422,
                    detail="Language must be 'en' or 'fr'.",
                )
            changes["language"] = clean_language

        for field_name in (
            "program_notifications",
            "reassessment_notifications",
            "update_notifications",
        ):
            value = getattr(payload, field_name)
            if value is not None:
                changes[field_name] = value

        if changes:
            response = (
                supabase_client.table("profiles")
                .update(changes)
                .eq("id", user["id"])
                .execute()
            )
            if not response.data:
                raise HTTPException(
                    status_code=500,
                    detail="Unable to update the profile.",
                )

        return get_my_profile(user)

    @router.get("/me/entitlements")
    def get_my_entitlements(
        user: dict[str, Any] = Depends(require_user),
    ):
        entitlement_response = (
            supabase_client.table("entitlements")
            .select("*")
            .eq("user_id", user["id"])
            .limit(1)
            .execute()
        )

        entitlement = (
            entitlement_response.data[0]
            if entitlement_response.data
            else {
                "plan_code": "free",
                "source": "free_signup",
                "status": "active",
                "program_access": False,
                "workout_access": False,
                "history_access": True,
                "report_access": True,
                "can_generate_program": False,
                "valid_until": None,
            }
        )

        cycles_response = (
            supabase_client.table("screening_credit_cycles")
            .select(
                "id,source,cycle_start,cycle_end,grace_expires_at,"
                "credits_granted,credits_used"
            )
            .eq("user_id", user["id"])
            .order("created_at", desc=True)
            .execute()
        )

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        remaining = 0
        active_cycle = None

        for cycle in cycles_response.data or []:
            cycle_start_raw = cycle.get("cycle_start")
            expiry_raw = (
                cycle.get("grace_expires_at")
                or cycle.get("cycle_end")
            )

            try:
                cycle_start = datetime.fromisoformat(
                    str(cycle_start_raw).replace("Z", "+00:00")
                )
                started = cycle_start <= now
            except Exception:
                started = False

            valid = True
            if expiry_raw:
                try:
                    expiry = datetime.fromisoformat(
                        str(expiry_raw).replace("Z", "+00:00")
                    )
                    valid = expiry >= now
                except Exception:
                    valid = False

            available = max(
                0,
                int(cycle.get("credits_granted") or 0)
                - int(cycle.get("credits_used") or 0),
            )

            if started and valid and available > 0:
                remaining += available
                if active_cycle is None:
                    active_cycle = cycle

        return {
            "plan_code": entitlement.get("plan_code") or "free",
            "source": entitlement.get("source") or "free",
            "subscription_status": entitlement.get("status") or "active",
            "program_access": bool(
                entitlement.get("program_access", False)
            ),
            "workout_access": bool(
                entitlement.get("workout_access", False)
            ),
            "history_access": bool(
                entitlement.get("history_access", True)
            ),
            "report_access": bool(
                entitlement.get("report_access", True)
            ),
            "can_generate_program": bool(
                entitlement.get("can_generate_program", False)
            ),
            "valid_until": entitlement.get("valid_until"),
            "screening_credits_remaining": remaining,
            "screening_credit_expires_at": (
                active_cycle.get("grace_expires_at")
                or active_cycle.get("cycle_end")
                if active_cycle
                else None
            ),
        }

    return router
