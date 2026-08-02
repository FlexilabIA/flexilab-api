from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from screening_access import effective_entitlement


class ProfileUpdate(BaseModel):
    full_name: Optional[str] = Field(default=None, max_length=120)
    language: Optional[str] = None
    program_notifications: Optional[bool] = None
    reassessment_notifications: Optional[bool] = None
    update_notifications: Optional[bool] = None


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


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

    def latest_subscription(user_id: str) -> Optional[dict[str, Any]]:
        response = (
            supabase_client.table("subscriptions")
            .select("*")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    def subscription_summary(row: Optional[dict[str, Any]]) -> dict[str, Any]:
        if not row:
            return {
                "billing_model": "free",
                "provider": None,
                "provider_customer_id": None,
                "provider_subscription_id": None,
                "current_period_start": None,
                "current_period_end": None,
                "next_billing_date": None,
                "cancel_at_period_end": False,
                "cancellation_requested": False,
                "cancellation_effective_at": None,
                "auto_renew": False,
                "can_manage_billing": False,
                "can_cancel": False,
                "can_reactivate": False,
            }

        plan_code = str(row.get("plan_code") or "free")
        status = str(row.get("status") or "active")
        is_monthly = plan_code == "pro_monthly"
        is_prepaid = plan_code in {"pro_three_month", "pro_annual", "trainer_pack_30"}
        cancel_at_period_end = bool(row.get("cancel_at_period_end", False))
        period_end = row.get("current_period_end")

        if is_monthly:
            billing_model = "monthly"
        elif is_prepaid:
            billing_model = "prepaid"
        else:
            billing_model = "free"

        return {
            "billing_model": billing_model,
            "provider": row.get("provider"),
            "provider_customer_id": row.get("provider_customer_id"),
            "provider_subscription_id": row.get("provider_subscription_id"),
            "current_period_start": row.get("current_period_start"),
            "current_period_end": period_end,
            "next_billing_date": (
                period_end
                if is_monthly
                and status in {"active", "trialing"}
                and not cancel_at_period_end
                else None
            ),
            "cancel_at_period_end": cancel_at_period_end,
            "cancellation_requested": cancel_at_period_end,
            "cancellation_effective_at": period_end if cancel_at_period_end else None,
            "auto_renew": is_monthly and not cancel_at_period_end,
            "can_manage_billing": bool(row.get("provider_customer_id")),
            "can_cancel": (
                is_monthly
                and status in {"active", "trialing"}
                and not cancel_at_period_end
            ),
            "can_reactivate": (
                is_monthly
                and status in {"active", "trialing"}
                and cancel_at_period_end
            ),
        }

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

            trainer_changes = {
                key: changes[key]
                for key in ("full_name", "language")
                if key in changes
            }
            if trainer_changes:
                try:
                    (
                        supabase_client.table("trainer_profiles")
                        .update(trainer_changes)
                        .eq("user_id", user["id"])
                        .execute()
                    )
                except Exception:
                    pass

        return get_my_profile(user)

    @router.get("/me/entitlements")
    def get_my_entitlements(
        user: dict[str, Any] = Depends(require_user),
    ):
        entitlement = effective_entitlement(supabase_client, user["id"])

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

        now = datetime.now(timezone.utc)
        remaining = 0
        active_cycle = None
        next_future_cycle = None

        for cycle in cycles_response.data or []:
            cycle_start = _parse_datetime(cycle.get("cycle_start"))
            expiry = _parse_datetime(
                cycle.get("grace_expires_at") or cycle.get("cycle_end")
            )

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
                    current_next = _parse_datetime(
                        next_future_cycle.get("cycle_start")
                    )
                    if current_next is None or cycle_start < current_next:
                        next_future_cycle = cycle

        subscription = subscription_summary(latest_subscription(user["id"]))

        upgrade_response = (
            supabase_client.table("assessment_upgrade_credits")
            .select("amount_cents,status,expires_at")
            .eq("user_id", user["id"])
            .limit(1)
            .execute()
        )
        upgrade_row = upgrade_response.data[0] if upgrade_response.data else None
        upgrade_expires = _parse_datetime((upgrade_row or {}).get("expires_at"))
        upgrade_available = bool(
            upgrade_row
            and upgrade_row.get("status") == "available"
            and (upgrade_expires is None or upgrade_expires >= now)
        )

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
            "organization_sources": entitlement.get("organization_sources", []),
            "screening_credits_remaining": remaining,
            "screening_credit_expires_at": (
                active_cycle.get("grace_expires_at")
                or active_cycle.get("cycle_end")
                if active_cycle
                else None
            ),
            "next_credit_cycle_at": (
                next_future_cycle.get("cycle_start")
                if next_future_cycle
                else None
            ),
            "assessment_upgrade_credit_available": upgrade_available,
            "assessment_upgrade_credit_amount_cents": (
                int(upgrade_row.get("amount_cents") or 0) if upgrade_available else 0
            ),
            "assessment_upgrade_credit_expires_at": (
                upgrade_row.get("expires_at") if upgrade_available else None
            ),
            **subscription,
        }

    @router.delete("/me/account")
    def delete_my_account(
        user: dict[str, Any] = Depends(require_user),
    ):
        subscription = latest_subscription(user["id"])
        if subscription:
            plan_code = str(subscription.get("plan_code") or "")
            status = str(subscription.get("status") or "")
            cancel_at_period_end = bool(subscription.get("cancel_at_period_end", False))
            if plan_code == "pro_monthly" and status in {"active", "trialing"} and not cancel_at_period_end:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Cancel automatic renewal before deleting your account. "
                        "Your access remains active until the end of the paid period."
                    ),
                )

        try:
            supabase_client.rpc(
                "delete_flexilab_user_data",
                {"p_user_id": user["id"], "p_user_email": user.get("email") or ""},
            ).execute()
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Unable to delete account data: {exc}",
            )

        try:
            # Explicitly request a hard deletion. A soft-deleted or retained Auth
            # identity would prevent the same email address from being registered again.
            supabase_client.auth.admin.delete_user(
                user["id"],
                should_soft_delete=False,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Account data was deleted, but authentication removal failed: {exc}",
            )

        # Do not report success merely because the SDK call returned without raising.
        # Verify that Supabase Auth can no longer retrieve the identity.
        auth_user_still_exists = True
        verification_error: Optional[Exception] = None

        for attempt in range(3):
            try:
                response = supabase_client.auth.admin.get_user_by_id(user["id"])
                auth_user = getattr(response, "user", None)
                if auth_user is None and isinstance(response, dict):
                    auth_user = response.get("user")

                if auth_user is None:
                    auth_user_still_exists = False
                    break
            except Exception as exc:
                message = str(exc).lower()
                if any(
                    marker in message
                    for marker in (
                        "user not found",
                        "not found",
                        "404",
                    )
                ):
                    auth_user_still_exists = False
                    break
                verification_error = exc

            if attempt < 2:
                time.sleep(0.25)

        if auth_user_still_exists:
            detail = (
                "Authentication removal could not be verified. "
                "The account data was deleted, but the Supabase Auth identity may still exist."
            )
            if verification_error is not None:
                detail += f" Verification error: {verification_error}"
            raise HTTPException(status_code=500, detail=detail)

        return {"ok": True, "message": "Account permanently deleted."}

    return router
