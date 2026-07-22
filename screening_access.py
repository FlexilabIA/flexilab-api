from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import logging

from fastapi import HTTPException

logger = logging.getLogger("flexilab.screening_access")


def authenticated_user(
    supabase_client,
    authorization: Optional[str],
) -> dict[str, str]:
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

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication token.",
        )

    try:
        response = supabase_client.auth.get_user(token)
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
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in {401, 403}:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired authentication token.",
            )
        logger.exception("supabase_auth_http_error status=%s", exc.response.status_code)
        raise HTTPException(
            status_code=503,
            detail="Authentication service is temporarily unavailable.",
        )
    except (
        httpx.ReadError,
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.PoolTimeout,
    ):
        logger.exception("supabase_auth_transport_error")
        raise HTTPException(
            status_code=503,
            detail="Authentication service is temporarily unavailable.",
        )
    except Exception:
        logger.exception("supabase_auth_unexpected_error")
        raise HTTPException(
            status_code=503,
            detail="Authentication verification failed temporarily.",
        )


def ensure_email_matches(
    authenticated_email: str,
    submitted_email: str,
) -> str:
    normalized = str(submitted_email or "").strip().lower()

    if not normalized or normalized != authenticated_email:
        raise HTTPException(
            status_code=403,
            detail="The submitted email does not match the authenticated account.",
        )

    return normalized


def reserve_credit(
    supabase_client,
    user_id: str,
    session_id: str,
) -> dict[str, Any]:
    response = supabase_client.rpc(
        "reserve_screening_credit",
        {
            "p_user_id": user_id,
            "p_session_id": session_id,
        },
    ).execute()

    rows = response.data or []
    result = rows[0] if rows else {}

    if not result.get("allowed"):
        reason = result.get("reason") or "no_screening_credit"
        raise HTTPException(
            status_code=402,
            detail={
                "code": "SCREENING_CREDIT_REQUIRED",
                "message": (
                    "No screening credit is available. "
                    "Upgrade your plan to continue."
                ),
                "reason": reason,
                "screening_credits_remaining": int(
                    result.get("credits_remaining") or 0
                ),
            },
        )

    return result


def consume_credit(
    supabase_client,
    user_id: str,
    session_id: str,
) -> dict[str, Any]:
    response = supabase_client.rpc(
        "consume_screening_credit",
        {
            "p_user_id": user_id,
            "p_session_id": session_id,
        },
    ).execute()

    rows = response.data or []
    result = rows[0] if rows else {}

    reason = str(result.get("reason") or "").strip().lower()
    already_consumed = bool(
        result.get("already_consumed")
        or reason in {
            "already_consumed",
            "session_already_consumed",
            "credit_already_consumed",
        }
    )

    if not result.get("consumed") and not already_consumed:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "SCREENING_CREDIT_CONSUMPTION_FAILED",
                "message": "The screening was completed, but its credit could not be confirmed.",
                "reason": result.get("reason") or "unknown",
            },
        )

    if already_consumed:
        result["consumed"] = True
        result["idempotent_replay"] = True

    return result


def release_credit(
    supabase_client,
    user_id: str,
    session_id: str,
    reason: str = "screening_abandoned",
) -> bool:
    try:
        response = supabase_client.rpc(
            "release_screening_credit",
            {
                "p_user_id": user_id,
                "p_session_id": session_id,
                "p_reason": reason,
            },
        ).execute()

        data = response.data
        if isinstance(data, bool):
            return data
        if isinstance(data, dict):
            return bool(
                data.get("released")
                or data.get("success")
                or data.get("allowed")
            )
        if isinstance(data, list):
            if not data:
                return False
            first = data[0]
            if isinstance(first, bool):
                return first
            if isinstance(first, dict):
                return bool(
                    first.get("released")
                    or first.get("success")
                    or first.get("allowed")
                )
        return bool(data)
    except (
        httpx.ReadError,
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.PoolTimeout,
    ):
        logger.exception(
            "release_screening_credit_transport_error user_id=%s session_id=%s",
            user_id,
            session_id,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "SCREENING_CREDIT_SERVICE_UNAVAILABLE",
                "message": (
                    "The screening credit service is temporarily unavailable. "
                    "No credit has been consumed."
                ),
            },
        )
    except Exception:
        logger.exception(
            "release_screening_credit_failed user_id=%s session_id=%s",
            user_id,
            session_id,
        )
        return False


def _parse_utc(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def effective_entitlement(supabase_client, user_id: str) -> dict[str, Any]:
    """Return the effective personal + organization entitlement for one Auth UUID.

    Personal Stripe/admin access is never overwritten by a corporate membership.
    Access flags are combined only while their source is active and unexpired.
    """
    now = datetime.now(timezone.utc)
    base: dict[str, Any] = {
        "plan_code": "free",
        "source": "free_signup",
        "status": "active",
        "program_access": False,
        "workout_access": False,
        "history_access": True,
        "report_access": True,
        "can_generate_program": False,
        "valid_until": None,
        "organization_sources": [],
    }

    personal_response = (
        supabase_client.table("entitlements")
        .select("*")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    personal = personal_response.data[0] if personal_response.data else None
    if personal:
        personal_expiry = _parse_utc(personal.get("valid_until"))
        personal_active = (
            str(personal.get("status") or "active") in {"active", "grace"}
            and (personal_expiry is None or personal_expiry >= now)
        )
        base.update(personal)
        if not personal_active:
            base.update({
                "program_access": False,
                "workout_access": False,
                "can_generate_program": False,
            })

    try:
        organization_response = (
            supabase_client.table("organization_entitlements")
            .select("organization_id,plan_code,status,program_access,workout_access,history_access,report_access,valid_from,valid_until")
            .eq("user_id", user_id)
            .in_("status", ["active", "grace"])
            .execute()
        )
    except Exception:
        organization_response = None

    organization_sources: list[dict[str, Any]] = []
    valid_until_values: list[Optional[datetime]] = []
    base_expiry = _parse_utc(base.get("valid_until"))
    if base.get("program_access") or base.get("workout_access"):
        valid_until_values.append(base_expiry)

    for row in (organization_response.data if organization_response else []) or []:
        starts = _parse_utc(row.get("valid_from"))
        expiry = _parse_utc(row.get("valid_until"))
        if starts and starts > now:
            continue
        if expiry and expiry < now:
            continue
        organization_sources.append({
            "organization_id": row.get("organization_id"),
            "plan_code": row.get("plan_code"),
            "valid_until": row.get("valid_until"),
        })
        base["program_access"] = bool(base.get("program_access")) or bool(row.get("program_access"))
        base["workout_access"] = bool(base.get("workout_access")) or bool(row.get("workout_access"))
        base["history_access"] = bool(base.get("history_access", True)) or bool(row.get("history_access", True))
        base["report_access"] = bool(base.get("report_access", True)) or bool(row.get("report_access", True))
        base["can_generate_program"] = bool(base.get("can_generate_program")) or bool(row.get("program_access"))
        valid_until_values.append(expiry)

    if organization_sources:
        if not personal or str(base.get("plan_code") or "free") == "free":
            base["plan_code"] = organization_sources[0].get("plan_code") or "organization"
            base["source"] = "organization"
        base["status"] = "active"

    # Any active source without an expiry means effective access has no fixed end.
    if valid_until_values:
        if any(value is None for value in valid_until_values):
            base["valid_until"] = None
        else:
            base["valid_until"] = max(value for value in valid_until_values if value).isoformat()

    base["organization_sources"] = organization_sources
    return base
