from typing import Any, Optional

from fastapi import Header, HTTPException


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
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token.",
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

    if not result.get("consumed"):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "SCREENING_CREDIT_CONSUMPTION_FAILED",
                "message": "The screening was completed, but its credit could not be confirmed.",
                "reason": result.get("reason") or "unknown",
            },
        )

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
        return bool(response.data)
    except Exception:
        return False
