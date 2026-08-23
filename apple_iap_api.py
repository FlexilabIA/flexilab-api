from __future__ import annotations

import calendar
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import requests
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from appstoreserverlibrary.models.Environment import Environment
from appstoreserverlibrary.signed_data_verifier import SignedDataVerifier, VerificationException

from screening_access import effective_entitlement


APPLE_BUNDLE_ID = os.environ.get("APPLE_BUNDLE_ID", "app.flexilab.mobile").strip()
APPLE_APP_ID = os.environ.get("APPLE_APP_ID", "").strip()

APPLE_PRODUCTS: dict[str, str] = {
    "app.flexilab.screening.single": "standalone_assessment_4",
    "app.flexilab.trainer.credits15": "trainer_pack_30",
    "app.flexilab.pro.monthly": "pro_monthly",
    "app.flexilab.pro.3months": "pro_three_month",
    "app.flexilab.pro.annual": "pro_annual",
}

APPLE_ROOT_URLS = (
    "https://www.apple.com/certificateauthority/AppleRootCA-G2.cer",
    "https://www.apple.com/certificateauthority/AppleRootCA-G3.cer",
)


class VerifyApplePurchaseRequest(BaseModel):
    signed_transaction: str
    expected_product_id: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _add_months(value: datetime, months: int) -> datetime:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _apple_datetime(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        numeric = float(value)
        # App Store Server JWS date fields are milliseconds since the Unix epoch.
        if numeric > 10_000_000_000:
            numeric /= 1000.0
        return datetime.fromtimestamp(numeric, tz=timezone.utc)
    except Exception:
        pass
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _value(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj.get(name)
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


@lru_cache(maxsize=1)
def _apple_root_certificates() -> tuple[bytes, ...]:
    local_names = ("apple-root-ca-g2.cer", "apple-root-ca-g3.cer")
    base = Path(__file__).resolve().parent
    roots: list[bytes] = []

    for name in local_names:
        path = base / name
        if path.exists():
            roots.append(path.read_bytes())

    if roots:
        return tuple(roots)

    # Render has outbound HTTPS access. Cache the official Apple roots for the
    # lifetime of the process so purchase verification does not repeatedly fetch them.
    for url in APPLE_ROOT_URLS:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        roots.append(response.content)

    if not roots:
        raise RuntimeError("Apple root certificates could not be loaded.")
    return tuple(roots)


def _verifier(environment: Environment) -> SignedDataVerifier:
    app_apple_id: Optional[int] = None
    if environment == Environment.PRODUCTION:
        if not APPLE_APP_ID:
            raise RuntimeError(
                "APPLE_APP_ID must be configured before production App Store purchases can be verified."
            )
        app_apple_id = int(APPLE_APP_ID)

    return SignedDataVerifier(
        list(_apple_root_certificates()),
        True,
        environment,
        APPLE_BUNDLE_ID,
        app_apple_id,
    )


def _verify_transaction(signed_transaction: str) -> Any:
    last_error: Optional[Exception] = None
    for environment in (Environment.SANDBOX, Environment.PRODUCTION):
        try:
            return _verifier(environment).verify_and_decode_signed_transaction(
                signed_transaction
            )
        except (VerificationException, RuntimeError, ValueError) as exc:
            last_error = exc
    raise HTTPException(
        status_code=400,
        detail=f"Apple transaction verification failed: {last_error or 'unknown error'}",
    )


def create_apple_iap_router(supabase_client) -> APIRouter:
    router = APIRouter(prefix="/apple-iap", tags=["apple-iap"])

    def require_user(
        authorization: Optional[str] = Header(default=None),
    ) -> dict[str, Any]:
        if supabase_client is None:
            raise HTTPException(status_code=503, detail="Supabase is not configured.")
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing authentication token.")
        access_token = authorization.split(" ", 1)[1].strip()
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
                raise ValueError("Authenticated user is incomplete.")
            return {"id": str(user_id), "email": str(email or "").lower()}
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid or expired authentication token.")

    def ensure_credit_cycle(
        *, user_id: str, subscription_id: Optional[str], source: str,
        cycle_start: datetime, cycle_end: datetime, credits: int = 1,
    ) -> None:
        existing = (
            supabase_client.table("screening_credit_cycles")
            .select("id")
            .eq("user_id", user_id)
            .eq("source", source)
            .limit(1)
            .execute()
        )
        if existing.data:
            return
        response = supabase_client.table("screening_credit_cycles").insert({
            "user_id": user_id,
            "subscription_id": subscription_id,
            "source": source,
            "cycle_start": _iso(cycle_start),
            "cycle_end": _iso(cycle_end),
            "grace_expires_at": _iso(cycle_end + timedelta(days=14) if credits == 1 else cycle_end),
            "credits_granted": credits,
            "credits_used": 0,
        }).execute()
        if not response.data:
            raise RuntimeError("Unable to grant Apple purchase credits.")

    def save_subscription(
        *, user_id: str, plan_code: str, original_transaction_id: str,
        transaction_id: str, period_start: datetime, period_end: datetime,
        metadata: dict[str, Any],
    ) -> dict:
        existing = (
            supabase_client.table("subscriptions")
            .select("*")
            .eq("provider", "apple")
            .eq("provider_subscription_id", original_transaction_id)
            .limit(1)
            .execute()
        )
        row = {
            "user_id": user_id,
            "plan_code": plan_code,
            "price_id": None,
            "provider": "apple",
            "provider_customer_id": None,
            "provider_subscription_id": original_transaction_id,
            "status": "active",
            "current_period_start": _iso(period_start),
            "current_period_end": _iso(period_end),
            "cancel_at_period_end": False,
            "metadata": {**metadata, "latest_transaction_id": transaction_id},
            "updated_at": _iso(_utc_now()),
        }
        if existing.data:
            result = (
                supabase_client.table("subscriptions")
                .update(row)
                .eq("id", existing.data[0]["id"])
                .execute()
            )
        else:
            result = supabase_client.table("subscriptions").insert(row).execute()
        if not result.data:
            raise RuntimeError("Unable to save Apple subscription.")
        return result.data[0]

    def activate_entitlement(
        *, user_id: str, plan_code: str, source_id: str,
        valid_from: datetime, valid_until: datetime,
    ) -> None:
        supabase_client.table("entitlements").upsert({
            "user_id": user_id,
            "plan_code": plan_code,
            "source": f"apple:{source_id}",
            "status": "active",
            "program_access": True,
            "workout_access": True,
            "history_access": True,
            "report_access": True,
            "can_generate_program": True,
            "valid_from": _iso(valid_from),
            "valid_until": _iso(valid_until),
            "updated_at": _iso(_utc_now()),
        }, on_conflict="user_id").execute()

    def grant_referral_reward(client_user_id: str, source_payment_id: str) -> None:
        links = (
            supabase_client.table("trainer_clients")
            .select("id,trainer_id")
            .eq("client_user_id", client_user_id)
            .eq("status", "active")
            .order("created_at", desc=False)
            .limit(1)
            .execute()
        )
        if not links.data:
            return
        link = links.data[0]
        trainer_id = str(link.get("trainer_id") or "")
        if not trainer_id:
            return
        existing = (
            supabase_client.table("trainer_referral_rewards")
            .select("id")
            .eq("trainer_id", trainer_id)
            .eq("client_user_id", client_user_id)
            .limit(1)
            .execute()
        )
        if existing.data:
            return
        now = _utc_now()
        end = _add_months(now, 12)
        ensure_credit_cycle(
            user_id=trainer_id,
            subscription_id=None,
            source=f"trainer_referral:{trainer_id}:{client_user_id}",
            cycle_start=now,
            cycle_end=end,
            credits=2,
        )
        supabase_client.table("trainer_referral_rewards").insert({
            "trainer_id": trainer_id,
            "client_user_id": client_user_id,
            "trainer_client_link_id": link.get("id"),
            "source_payment_id": source_payment_id,
            "tokens_granted": 2,
        }).execute()

    @router.post("/verify")
    def verify_apple_purchase(
        payload: VerifyApplePurchaseRequest,
        user: dict[str, Any] = Depends(require_user),
    ) -> dict[str, Any]:
        expected_product = payload.expected_product_id.strip()
        if expected_product not in APPLE_PRODUCTS:
            raise HTTPException(status_code=400, detail="Unknown FlexiLab Apple product.")

        transaction = _verify_transaction(payload.signed_transaction.strip())
        product_id = str(_value(transaction, "productId", "product_id", default="") or "")
        transaction_id = str(_value(transaction, "transactionId", "transaction_id", default="") or "")
        original_transaction_id = str(
            _value(transaction, "originalTransactionId", "original_transaction_id", default="")
            or transaction_id
        )
        app_account_token = str(
            _value(transaction, "appAccountToken", "app_account_token", default="") or ""
        )
        revocation_date = _apple_datetime(
            _value(transaction, "revocationDate", "revocation_date")
        )

        if product_id != expected_product:
            raise HTTPException(status_code=400, detail="Apple product identifier mismatch.")
        if not transaction_id:
            raise HTTPException(status_code=400, detail="Apple transaction has no transaction id.")
        if app_account_token and app_account_token.lower() != user["id"].lower():
            raise HTTPException(status_code=403, detail="This Apple purchase belongs to another FlexiLab account.")
        if revocation_date:
            raise HTTPException(status_code=409, detail="This Apple purchase has been revoked.")

        plan_code = APPLE_PRODUCTS[product_id]
        now = _utc_now()
        purchase_date = _apple_datetime(
            _value(transaction, "purchaseDate", "purchase_date")
        ) or now
        expires_date = _apple_datetime(
            _value(transaction, "expiresDate", "expires_date")
        )

        if product_id == "app.flexilab.screening.single":
            entitlement = effective_entitlement(supabase_client, user["id"])
            if bool(entitlement.get("program_access") or entitlement.get("workout_access")):
                plan_code = "extra_screening_4"
            else:
                plan_code = "standalone_assessment_4"
            ensure_credit_cycle(
                user_id=user["id"],
                subscription_id=None,
                source=f"apple:{transaction_id}",
                cycle_start=purchase_date,
                cycle_end=_add_months(purchase_date, 12),
                credits=1,
            )
            grant_referral_reward(user["id"], transaction_id)

        elif plan_code == "trainer_pack_30":
            supabase_client.table("trainer_profiles").upsert({
                "user_id": user["id"], "status": "active", "updated_at": _iso(now)
            }, on_conflict="user_id").execute()
            ensure_credit_cycle(
                user_id=user["id"],
                subscription_id=None,
                source=f"trainer_pack_30:apple:{transaction_id}",
                cycle_start=purchase_date,
                cycle_end=_add_months(purchase_date, 12),
                credits=15,
            )

        else:
            if not expires_date or expires_date <= now:
                raise HTTPException(status_code=409, detail="Apple subscription is not active.")
            saved = save_subscription(
                user_id=user["id"],
                plan_code=plan_code,
                original_transaction_id=original_transaction_id,
                transaction_id=transaction_id,
                period_start=purchase_date,
                period_end=expires_date,
                metadata={
                    "product_id": product_id,
                    "environment": str(_value(transaction, "environment", default="") or ""),
                    "app_account_token": app_account_token,
                },
            )
            activate_entitlement(
                user_id=user["id"],
                plan_code=plan_code,
                source_id=original_transaction_id,
                valid_from=purchase_date,
                valid_until=expires_date,
            )

            # FlexiLab grants one screening credit per month. Create only the
            # cycles covered by this verified StoreKit subscription period.
            cycle_start = purchase_date
            index = 0
            while cycle_start < expires_date and index < 12:
                candidate_end = _add_months(cycle_start, 1)
                cycle_end = min(candidate_end, expires_date)
                ensure_credit_cycle(
                    user_id=user["id"],
                    subscription_id=saved["id"],
                    source=f"{plan_code}:apple:{original_transaction_id}:{_iso(cycle_start)}",
                    cycle_start=cycle_start,
                    cycle_end=cycle_end,
                    credits=1,
                )
                cycle_start = candidate_end
                index += 1
            grant_referral_reward(user["id"], transaction_id)

        return {
            "ok": True,
            "provider": "apple",
            "product_id": product_id,
            "plan_code": plan_code,
            "transaction_id": transaction_id,
        }

    return router
