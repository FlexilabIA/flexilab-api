from __future__ import annotations

import calendar
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import uuid

import stripe
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

from screening_access import effective_entitlement


STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "").strip()
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()
STRIPE_PRICE_MONTHLY = os.environ.get("STRIPE_PRICE_MONTHLY", "").strip()
STRIPE_PRICE_THREE_MONTH = os.environ.get("STRIPE_PRICE_THREE_MONTH", "").strip()
STRIPE_PRICE_ANNUAL = os.environ.get("STRIPE_PRICE_ANNUAL", "").strip()
STRIPE_PRICE_TRAINER_PACK = os.environ.get("STRIPE_PRICE_TRAINER_PACK", "").strip()
STRIPE_PRICE_SINGLE_ASSESSMENT = os.environ.get("STRIPE_PRICE_SINGLE_ASSESSMENT", "").strip()
STRIPE_UPGRADE_COUPON_ID = os.environ.get("STRIPE_UPGRADE_COUPON_ID", "").strip()
FRONTEND_URL = os.environ.get(
    "FRONTEND_URL",
    "https://flexi-move-lab.lovable.app",
).rstrip("/")

stripe.api_key = STRIPE_SECRET_KEY


class CheckoutRequest(BaseModel):
    plan_code: str


PLAN_CONFIG = {
    "pro_monthly": {
        "price_id": STRIPE_PRICE_MONTHLY,
        "mode": "subscription",
        "months": 1,
    },
    "pro_three_month": {
        "price_id": STRIPE_PRICE_THREE_MONTH,
        "mode": "payment",
        "months": 3,
    },
    "pro_annual": {
        "price_id": STRIPE_PRICE_ANNUAL,
        "mode": "payment",
        "months": 12,
    },
    "standalone_assessment_4": {
        "price_id": STRIPE_PRICE_SINGLE_ASSESSMENT,
        "mode": "payment",
        "months": 0,
        "screening_credits": 1,
        "creates_upgrade_credit": True,
    },
    "extra_screening_4": {
        "price_id": STRIPE_PRICE_SINGLE_ASSESSMENT,
        "mode": "payment",
        "months": 0,
        "screening_credits": 1,
        "creates_upgrade_credit": False,
    },
    "trainer_pack_30": {
        "price_id": STRIPE_PRICE_TRAINER_PACK,
        "mode": "payment",
        "months": 12,
        "trainer_tokens": 30,
    },
}


def _object_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _metadata(obj: Any) -> dict[str, str]:
    raw = _object_value(obj, "metadata", {}) or {}
    if hasattr(raw, "to_dict"):
        raw = raw.to_dict()
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if v is not None}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _from_unix(value: Any) -> Optional[datetime]:
    try:
        if value is None:
            return None
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except Exception:
        return None


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _add_months(value: datetime, months: int) -> datetime:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _subscription_period(subscription: Any) -> tuple[datetime, datetime]:
    start = _from_unix(_object_value(subscription, "current_period_start"))
    end = _from_unix(_object_value(subscription, "current_period_end"))

    if not start or not end:
        items = _object_value(subscription, "items", {})
        data = _object_value(items, "data", []) or []
        first_item = data[0] if data else None
        start = start or _from_unix(
            _object_value(first_item, "current_period_start")
        )
        end = end or _from_unix(
            _object_value(first_item, "current_period_end")
        )

    now = _utc_now()
    return start or now, end or _add_months(start or now, 1)


def _invoice_subscription_id(invoice: Any) -> Optional[str]:
    direct = _object_value(invoice, "subscription")
    if direct:
        return str(direct)

    parent = _object_value(invoice, "parent", {})
    details = _object_value(parent, "subscription_details", {})
    nested = _object_value(details, "subscription")
    return str(nested) if nested else None


def create_stripe_router(supabase_client) -> APIRouter:
    router = APIRouter(prefix="/stripe", tags=["stripe"])

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

            if not user_id or not email:
                raise ValueError("Authenticated user is incomplete.")

            return {
                "id": str(user_id),
                "email": str(email).strip().lower(),
            }
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired authentication token.",
            )

    def price_row_id(plan_code: str) -> Optional[str]:
        response = (
            supabase_client.table("prices")
            .select("id")
            .eq("plan_code", plan_code)
            .eq("provider", "stripe")
            .limit(1)
            .execute()
        )
        return str(response.data[0]["id"]) if response.data else None

    def find_subscription(provider_subscription_id: str) -> Optional[dict]:
        response = (
            supabase_client.table("subscriptions")
            .select("*")
            .eq("provider", "stripe")
            .eq("provider_subscription_id", provider_subscription_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None


    def latest_user_subscription(user_id: str) -> Optional[dict]:
        response = (
            supabase_client.table("subscriptions")
            .select("*")
            .eq("user_id", user_id)
            .eq("provider", "stripe")
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    def require_monthly_subscription(user_id: str) -> dict:
        saved = latest_user_subscription(user_id)
        if not saved:
            raise HTTPException(
                status_code=404,
                detail="No Stripe subscription was found for this account.",
            )
        if saved.get("plan_code") != "pro_monthly":
            raise HTTPException(
                status_code=409,
                detail="This prepaid plan does not renew automatically.",
            )
        provider_subscription_id = str(
            saved.get("provider_subscription_id") or ""
        )
        if not provider_subscription_id:
            raise HTTPException(
                status_code=409,
                detail="This subscription cannot be managed automatically.",
            )
        return saved

    def save_subscription(
        *,
        user_id: str,
        plan_code: str,
        provider_customer_id: Optional[str],
        provider_subscription_id: str,
        status: str,
        period_start: datetime,
        period_end: datetime,
        cancel_at_period_end: bool = False,
        metadata: Optional[dict] = None,
    ) -> dict:
        row = {
            "user_id": user_id,
            "plan_code": plan_code,
            "price_id": price_row_id(plan_code),
            "provider": "stripe",
            "provider_customer_id": provider_customer_id,
            "provider_subscription_id": provider_subscription_id,
            "status": status,
            "current_period_start": _iso(period_start),
            "current_period_end": _iso(period_end),
            "cancel_at_period_end": cancel_at_period_end,
            "metadata": metadata or {},
            "updated_at": _iso(_utc_now()),
        }

        existing = find_subscription(provider_subscription_id)
        if existing:
            response = (
                supabase_client.table("subscriptions")
                .update(row)
                .eq("id", existing["id"])
                .execute()
            )
        else:
            response = supabase_client.table("subscriptions").insert(row).execute()

        if not response.data:
            raise RuntimeError("Unable to save Stripe subscription.")
        return response.data[0]

    def activate_entitlement(
        *,
        user_id: str,
        plan_code: str,
        source_id: str,
        valid_from: datetime,
        valid_until: datetime,
    ) -> None:
        row = {
            "user_id": user_id,
            "plan_code": plan_code,
            "source": f"stripe:{source_id}",
            "status": "active",
            "program_access": True,
            "workout_access": True,
            "history_access": True,
            "report_access": True,
            "can_generate_program": True,
            "valid_from": _iso(valid_from),
            "valid_until": _iso(valid_until),
            "updated_at": _iso(_utc_now()),
        }
        supabase_client.table("entitlements").upsert(
            row,
            on_conflict="user_id",
        ).execute()

    def deactivate_entitlement(
        *,
        user_id: str,
        source_id: str,
        status: str = "expired",
    ) -> None:
        response = (
            supabase_client.table("entitlements")
            .select("source")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if not response.data:
            return

        if response.data[0].get("source") != f"stripe:{source_id}":
            return

        supabase_client.table("entitlements").update(
            {
                "status": status,
                "program_access": False,
                "workout_access": False,
                "history_access": True,
                "report_access": True,
                "can_generate_program": False,
                "updated_at": _iso(_utc_now()),
            }
        ).eq("user_id", user_id).execute()

    def ensure_credit_cycle(
        *,
        user_id: str,
        subscription_id: str,
        source: str,
        cycle_start: datetime,
        cycle_end: datetime,
    ) -> None:
        existing = (
            supabase_client.table("screening_credit_cycles")
            .select("id")
            .eq("subscription_id", subscription_id)
            .eq("cycle_start", _iso(cycle_start))
            .limit(1)
            .execute()
        )
        if existing.data:
            return

        supabase_client.table("screening_credit_cycles").insert(
            {
                "user_id": user_id,
                "subscription_id": subscription_id,
                "source": source,
                "cycle_start": _iso(cycle_start),
                "cycle_end": _iso(cycle_end),
                "grace_expires_at": _iso(cycle_end + timedelta(days=14)),
                "credits_granted": 2,
                "credits_used": 0,
            }
        ).execute()

    def create_prepaid_cycles(
        *,
        user_id: str,
        subscription_id: str,
        plan_code: str,
        start: datetime,
        months: int,
    ) -> None:
        for index in range(months):
            cycle_start = _add_months(start, index)
            cycle_end = _add_months(start, index + 1)
            ensure_credit_cycle(
                user_id=user_id,
                subscription_id=subscription_id,
                source=plan_code,
                cycle_start=cycle_start,
                cycle_end=cycle_end,
            )

    def activate_trainer_pack(
        *,
        user_id: str,
        subscription_id: str,
        plan_code: str,
        start: datetime,
        end: datetime,
    ) -> None:
        # Trainer profiles are unique by user_id. Explicitly target that key so
        # an existing Trainer profile does not abort Stripe fulfillment.
        supabase_client.table("trainer_profiles").upsert(
            {
                "user_id": user_id,
                "status": "active",
                "updated_at": _iso(_utc_now()),
            },
            on_conflict="user_id",
        ).execute()

        # Idempotency: one token cycle per saved Stripe purchase.
        existing = (
            supabase_client.table("screening_credit_cycles")
            .select("id,credits_granted,credits_used")
            .eq("subscription_id", subscription_id)
            .eq("source", plan_code)
            .limit(1)
            .execute()
        )
        if existing.data:
            return

        cycle_response = (
            supabase_client.table("screening_credit_cycles")
            .insert({
                "user_id": user_id,
                "subscription_id": subscription_id,
                "source": plan_code,
                "cycle_start": _iso(start),
                "cycle_end": _iso(end),
                "grace_expires_at": _iso(end),
                "credits_granted": int(
                    PLAN_CONFIG[plan_code].get("trainer_tokens", 30)
                ),
                "credits_used": 0,
            })
            .execute()
        )
        if not cycle_response.data:
            raise RuntimeError("Unable to grant purchased Trainer tokens.")

    def grant_referral_reward(*, client_user_id: str, source_payment_id: str) -> None:
        """
        Grant two screening tokens to the original Trainer after the referred
        client completes their first successful paid FlexiLab purchase.

        The reward is intentionally granted once per Trainer/client pair, not
        once per invoice or renewal. This prevents duplicate rewards from
        recurring monthly invoices and repeated Stripe webhook delivery.
        """
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
        trainer_client_link_id = str(link.get("id") or "")
        if not trainer_id:
            return

        existing_reward = (
            supabase_client.table("trainer_referral_rewards")
            .select("id")
            .eq("trainer_id", trainer_id)
            .eq("client_user_id", client_user_id)
            .limit(1)
            .execute()
        )
        if existing_reward.data:
            return

        now = _utc_now()
        reward_end = _add_months(now, 12)
        reward_source = f"trainer_referral:{trainer_id}:{client_user_id}"

        existing_cycle = (
            supabase_client.table("screening_credit_cycles")
            .select("id")
            .eq("user_id", trainer_id)
            .eq("source", reward_source)
            .limit(1)
            .execute()
        )

        if not existing_cycle.data:
            cycle_response = (
                supabase_client.table("screening_credit_cycles")
                .insert({
                    "user_id": trainer_id,
                    "subscription_id": None,
                    "source": reward_source,
                    "cycle_start": _iso(now),
                    "cycle_end": _iso(reward_end),
                    "grace_expires_at": _iso(reward_end),
                    "credits_granted": 2,
                    "credits_used": 0,
                })
                .execute()
            )
            if not cycle_response.data:
                raise RuntimeError("Unable to grant Trainer referral tokens.")

        reward_response = (
            supabase_client.table("trainer_referral_rewards")
            .insert({
                "trainer_id": trainer_id,
                "client_user_id": client_user_id,
                "trainer_client_link_id": trainer_client_link_id or None,
                "source_payment_id": source_payment_id,
                "tokens_granted": 2,
            })
            .execute()
        )
        if not reward_response.data:
            raise RuntimeError("Unable to record the Trainer referral reward.")


    def grant_single_assessment_credit(*, user_id: str, checkout_id: str, plan_code: str) -> None:
        source = f"{plan_code}:{checkout_id}"
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

        now = _utc_now()
        end = now + timedelta(days=365)
        response = supabase_client.table("screening_credit_cycles").insert({
            "user_id": user_id,
            "subscription_id": None,
            "source": source,
            "cycle_start": _iso(now),
            "cycle_end": _iso(end),
            "grace_expires_at": _iso(end),
            "credits_granted": 1,
            "credits_used": 0,
        }).execute()
        if not response.data:
            raise RuntimeError("Unable to grant the purchased screening credit.")

        if plan_code == "standalone_assessment_4":
            # One lifetime conversion credit per account. Replayed webhooks are harmless.
            supabase_client.table("assessment_upgrade_credits").upsert({
                "user_id": user_id,
                "source_checkout_id": checkout_id,
                "amount_cents": 400,
                "currency": "eur",
                "status": "available",
                "expires_at": _iso(now + timedelta(days=30)),
                "updated_at": _iso(now),
            }, on_conflict="user_id", ignore_duplicates=True).execute()

    def available_upgrade_credit(user_id: str) -> Optional[dict]:
        now = _utc_now()
        response = (
            supabase_client.table("assessment_upgrade_credits")
            .select("*")
            .eq("user_id", user_id)
            .eq("status", "available")
            .limit(1)
            .execute()
        )
        if not response.data:
            return None
        row = response.data[0]
        expires = row.get("expires_at")
        if expires and datetime.fromisoformat(str(expires).replace("Z", "+00:00")) < now:
            supabase_client.table("assessment_upgrade_credits").update({
                "status": "expired", "updated_at": _iso(now)
            }).eq("id", row["id"]).execute()
            return None
        return row

    def mark_upgrade_credit_used(user_id: str, credit_id: str, payment_id: str) -> None:
        supabase_client.table("assessment_upgrade_credits").update({
            "status": "used",
            "used_at": _iso(_utc_now()),
            "used_payment_id": payment_id,
            "updated_at": _iso(_utc_now()),
        }).eq("id", credit_id).eq("user_id", user_id).eq("status", "available").execute()

    def process_checkout_completed(session: Any) -> None:
        payment_status = str(
            _object_value(session, "payment_status", "")
        ).lower()
        if payment_status not in {"paid", "no_payment_required"}:
            return

        meta = _metadata(session)
        user_id = meta.get("user_id") or str(
            _object_value(session, "client_reference_id", "")
        )
        plan_code = meta.get("plan_code", "")
        if not user_id or plan_code not in PLAN_CONFIG:
            raise RuntimeError("Stripe Checkout metadata is incomplete.")

        customer_id = _object_value(session, "customer")
        mode = str(_object_value(session, "mode", ""))
        now = _utc_now()

        if plan_code in {"standalone_assessment_4", "extra_screening_4"}:
            checkout_id = str(_object_value(session, "id"))
            grant_single_assessment_credit(
                user_id=user_id, checkout_id=checkout_id, plan_code=plan_code
            )
            grant_referral_reward(
                client_user_id=user_id, source_payment_id=checkout_id
            )
            return

        if mode == "subscription":
            stripe_subscription_id = str(
                _object_value(session, "subscription", "")
            )
            if not stripe_subscription_id:
                raise RuntimeError("Checkout has no Stripe subscription id.")

            subscription = stripe.Subscription.retrieve(
                stripe_subscription_id
            )
            period_start, period_end = _subscription_period(subscription)
            subscription_meta = _metadata(subscription)
            subscription_meta.update(meta)

            saved = save_subscription(
                user_id=user_id,
                plan_code=plan_code,
                provider_customer_id=str(customer_id) if customer_id else None,
                provider_subscription_id=stripe_subscription_id,
                status="active",
                period_start=period_start,
                period_end=period_end,
                cancel_at_period_end=bool(
                    _object_value(subscription, "cancel_at_period_end", False)
                ),
                metadata=subscription_meta,
            )
            activate_entitlement(
                user_id=user_id,
                plan_code=plan_code,
                source_id=stripe_subscription_id,
                valid_from=period_start,
                valid_until=period_end,
            )
            ensure_credit_cycle(
                user_id=user_id,
                subscription_id=saved["id"],
                source=plan_code,
                cycle_start=period_start,
                cycle_end=period_end,
            )
            grant_referral_reward(
                client_user_id=user_id,
                source_payment_id=stripe_subscription_id,
            )
            if meta.get("upgrade_credit_id"):
                mark_upgrade_credit_used(user_id, meta["upgrade_credit_id"], stripe_subscription_id)
            return

        months = int(PLAN_CONFIG[plan_code]["months"])
        checkout_id = str(_object_value(session, "id"))
        plan_end = _add_months(now, months)

        saved = save_subscription(
            user_id=user_id,
            plan_code=plan_code,
            provider_customer_id=str(customer_id) if customer_id else None,
            provider_subscription_id=checkout_id,
            status="active",
            period_start=now,
            period_end=plan_end,
            metadata=meta,
        )
        if plan_code == "trainer_pack_30":
            activate_trainer_pack(
                user_id=user_id,
                subscription_id=saved["id"],
                plan_code=plan_code,
                start=now,
                end=plan_end,
            )
        else:
            activate_entitlement(
                user_id=user_id,
                plan_code=plan_code,
                source_id=checkout_id,
                valid_from=now,
                valid_until=plan_end,
            )
            create_prepaid_cycles(
                user_id=user_id,
                subscription_id=saved["id"],
                plan_code=plan_code,
                start=now,
                months=months,
            )
            grant_referral_reward(
                client_user_id=user_id,
                source_payment_id=checkout_id,
            )
            if meta.get("upgrade_credit_id"):
                mark_upgrade_credit_used(user_id, meta["upgrade_credit_id"], checkout_id)

    def process_invoice_paid(invoice: Any) -> None:
        stripe_subscription_id = _invoice_subscription_id(invoice)
        if not stripe_subscription_id:
            return

        subscription = stripe.Subscription.retrieve(stripe_subscription_id)
        meta = _metadata(subscription)
        user_id = meta.get("user_id", "")
        plan_code = meta.get("plan_code", "")
        if not user_id or plan_code != "pro_monthly":
            return

        period_start, period_end = _subscription_period(subscription)
        customer_id = _object_value(subscription, "customer")

        saved = save_subscription(
            user_id=user_id,
            plan_code=plan_code,
            provider_customer_id=str(customer_id) if customer_id else None,
            provider_subscription_id=stripe_subscription_id,
            status="active",
            period_start=period_start,
            period_end=period_end,
            cancel_at_period_end=bool(
                _object_value(subscription, "cancel_at_period_end", False)
            ),
            metadata=meta,
        )
        activate_entitlement(
            user_id=user_id,
            plan_code=plan_code,
            source_id=stripe_subscription_id,
            valid_from=period_start,
            valid_until=period_end,
        )
        ensure_credit_cycle(
            user_id=user_id,
            subscription_id=saved["id"],
            source=plan_code,
            cycle_start=period_start,
            cycle_end=period_end,
        )

    def process_invoice_failed(invoice: Any) -> None:
        stripe_subscription_id = _invoice_subscription_id(invoice)
        if not stripe_subscription_id:
            return

        saved = find_subscription(stripe_subscription_id)
        if not saved:
            return

        supabase_client.table("subscriptions").update(
            {
                "status": "past_due",
                "updated_at": _iso(_utc_now()),
            }
        ).eq("id", saved["id"]).execute()

        deactivate_entitlement(
            user_id=saved["user_id"],
            source_id=stripe_subscription_id,
            status="grace",
        )

    def process_subscription_updated(subscription: Any) -> None:
        stripe_subscription_id = str(_object_value(subscription, "id", ""))
        meta = _metadata(subscription)
        existing = find_subscription(stripe_subscription_id)

        user_id = meta.get("user_id") or (
            str(existing.get("user_id")) if existing else ""
        )
        plan_code = meta.get("plan_code") or (
            str(existing.get("plan_code")) if existing else ""
        )
        if not user_id or not plan_code:
            return

        period_start, period_end = _subscription_period(subscription)
        stripe_status = str(_object_value(subscription, "status", "active"))
        supported_status = (
            stripe_status
            if stripe_status
            in {
                "trialing",
                "active",
                "past_due",
                "paused",
                "canceled",
                "incomplete",
            }
            else "active"
        )
        customer_id = _object_value(subscription, "customer")

        save_subscription(
            user_id=user_id,
            plan_code=plan_code,
            provider_customer_id=str(customer_id) if customer_id else None,
            provider_subscription_id=stripe_subscription_id,
            status=supported_status,
            period_start=period_start,
            period_end=period_end,
            cancel_at_period_end=bool(
                _object_value(subscription, "cancel_at_period_end", False)
            ),
            metadata=meta,
        )

        if supported_status in {"active", "trialing"}:
            activate_entitlement(
                user_id=user_id,
                plan_code=plan_code,
                source_id=stripe_subscription_id,
                valid_from=period_start,
                valid_until=period_end,
            )

    def process_subscription_deleted(subscription: Any) -> None:
        stripe_subscription_id = str(_object_value(subscription, "id", ""))
        existing = find_subscription(stripe_subscription_id)
        if not existing:
            return

        now = _utc_now()
        supabase_client.table("subscriptions").update(
            {
                "status": "canceled",
                "canceled_at": _iso(now),
                "ended_at": _iso(now),
                "updated_at": _iso(now),
            }
        ).eq("id", existing["id"]).execute()

        deactivate_entitlement(
            user_id=existing["user_id"],
            source_id=stripe_subscription_id,
            status="expired",
        )

    def begin_event(event_id: str, event_type: str, payload: dict) -> bool:
        existing = (
            supabase_client.table("stripe_webhook_events")
            .select("status")
            .eq("event_id", event_id)
            .limit(1)
            .execute()
        )
        if existing.data and existing.data[0].get("status") in {
            "processed",
            "ignored",
        }:
            return False

        row = {
            "event_id": event_id,
            "event_type": event_type,
            "status": "processing",
            "payload": payload,
            "error_message": None,
            "updated_at": _iso(_utc_now()),
        }
        supabase_client.table("stripe_webhook_events").upsert(
            row,
            on_conflict="event_id",
        ).execute()
        return True

    def finish_event(
        event_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        now = _utc_now()
        supabase_client.table("stripe_webhook_events").update(
            {
                "status": status,
                "error_message": error_message,
                "processed_at": _iso(now)
                if status in {"processed", "ignored"}
                else None,
                "updated_at": _iso(now),
            }
        ).eq("event_id", event_id).execute()

    @router.get("/status")
    def stripe_status():
        return {
            "configured": bool(
                STRIPE_SECRET_KEY
                and STRIPE_PRICE_MONTHLY
                and STRIPE_PRICE_THREE_MONTH
                and STRIPE_PRICE_ANNUAL
                and STRIPE_PRICE_TRAINER_PACK
            ),
            "webhook_configured": bool(STRIPE_WEBHOOK_SECRET),
            "monthly_price_configured": bool(STRIPE_PRICE_MONTHLY),
            "three_month_price_configured": bool(STRIPE_PRICE_THREE_MONTH),
            "annual_price_configured": bool(STRIPE_PRICE_ANNUAL),
            "trainer_pack_price_configured": bool(STRIPE_PRICE_TRAINER_PACK),
        }

    @router.post("/create-checkout-session")
    def create_checkout_session(
        payload: CheckoutRequest,
        authorization: Optional[str] = Header(default=None),
    ):
        if not STRIPE_SECRET_KEY:
            raise HTTPException(
                status_code=503,
                detail="Stripe is not configured on the server.",
            )

        user = require_user(authorization)
        plan_code = payload.plan_code.strip().lower()
        plan = PLAN_CONFIG.get(plan_code)

        if plan is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Invalid plan_code. Use pro_monthly, pro_three_month, "
                    "pro_annual, standalone_assessment_4, extra_screening_4, "
                    "or trainer_pack_30."
                ),
            )

        price_id = str(plan["price_id"] or "").strip()
        if not price_id:
            raise HTTPException(
                status_code=503,
                detail=f"Stripe price is not configured for {plan_code}.",
            )

        entitlement = effective_entitlement(supabase_client, user["id"])
        has_pro_access = bool(entitlement.get("program_access") or entitlement.get("workout_access"))
        if plan_code == "standalone_assessment_4" and has_pro_access:
            raise HTTPException(status_code=409, detail="Active Pro accounts must use extra_screening_4.")
        if plan_code == "extra_screening_4" and not has_pro_access:
            raise HTTPException(status_code=409, detail="Free accounts must use standalone_assessment_4.")

        metadata = {
            "user_id": user["id"],
            "user_email": user["email"],
            "plan_code": plan_code,
        }

        upgrade_credit = None
        if plan_code in {"pro_monthly", "pro_three_month", "pro_annual"}:
            upgrade_credit = available_upgrade_credit(user["id"])
            if upgrade_credit:
                if not STRIPE_UPGRADE_COUPON_ID:
                    raise HTTPException(status_code=503, detail="The assessment upgrade discount is not configured.")
                metadata["upgrade_credit_id"] = str(upgrade_credit["id"])

        session_params: dict[str, Any] = {
            "mode": plan["mode"],
            "line_items": [{"price": price_id, "quantity": 1}],
            "customer_email": user["email"],
            "client_reference_id": user["id"],
            "metadata": metadata,
            "success_url": (
                f"{FRONTEND_URL}"
                + ("/trainer" if plan_code == "trainer_pack_30" else "/home")
                + "?payment=success"
                + "&session_id={CHECKOUT_SESSION_ID}"
            ),
            "cancel_url": (
                f"{FRONTEND_URL}"
                + ("/trainer" if plan_code == "trainer_pack_30" else "/paywall")
                + "?payment=cancelled"
            ),
            # Display Stripe's secure promotion-code field for every
            # client plan and the Trainer 30-token pack.
            "allow_promotion_codes": not bool(upgrade_credit),
        }
        if upgrade_credit:
            session_params["discounts"] = [{"coupon": STRIPE_UPGRADE_COUPON_ID}]

        if plan["mode"] == "subscription":
            session_params["subscription_data"] = {
                "metadata": metadata,
            }
        else:
            session_params["customer_creation"] = "always"
            session_params["payment_intent_data"] = {
                "metadata": metadata,
            }

        try:
            checkout_session = stripe.checkout.Session.create(
                **session_params
            )
        except stripe.StripeError as exc:
            message = getattr(exc, "user_message", None) or str(exc)
            raise HTTPException(
                status_code=502,
                detail=f"Stripe checkout creation failed: {message}",
            )
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="Unable to create the Stripe Checkout session.",
            )

        if not checkout_session.url:
            raise HTTPException(
                status_code=500,
                detail="Stripe did not return a Checkout URL.",
            )

        return {
            "checkout_url": checkout_session.url,
            "session_id": checkout_session.id,
            "plan_code": plan_code,
        }


    @router.post("/checkout/reconcile/{checkout_session_id}")
    def reconcile_checkout_session(
        checkout_session_id: str,
        authorization: Optional[str] = Header(default=None),
    ):
        """Reconcile a paid Checkout session if webhook delivery was delayed."""
        if not STRIPE_SECRET_KEY:
            raise HTTPException(
                status_code=503,
                detail="Stripe is not configured on the server.",
            )

        user = require_user(authorization)

        try:
            checkout_session = stripe.checkout.Session.retrieve(
                checkout_session_id
            )
        except stripe.StripeError as exc:
            message = getattr(exc, "user_message", None) or str(exc)
            raise HTTPException(
                status_code=502,
                detail=f"Unable to verify Stripe Checkout: {message}",
            )

        meta = _metadata(checkout_session)
        checkout_user_id = meta.get("user_id") or str(
            _object_value(checkout_session, "client_reference_id", "")
        )
        if checkout_user_id != user["id"]:
            raise HTTPException(
                status_code=403,
                detail="This Stripe Checkout session belongs to another account.",
            )

        payment_status = str(
            _object_value(checkout_session, "payment_status", "")
        ).lower()
        if payment_status not in {"paid", "no_payment_required"}:
            raise HTTPException(
                status_code=409,
                detail="Stripe payment is not confirmed yet.",
            )

        process_checkout_completed(checkout_session)

        return {
            "ok": True,
            "session_id": checkout_session_id,
            "plan_code": meta.get("plan_code"),
            "payment_status": payment_status,
            "fulfilled": True,
        }


    @router.post("/billing-portal")
    def create_billing_portal(
        authorization: Optional[str] = Header(default=None),
    ):
        if not STRIPE_SECRET_KEY:
            raise HTTPException(
                status_code=503,
                detail="Stripe is not configured on the server.",
            )

        user = require_user(authorization)
        saved = latest_user_subscription(user["id"])
        customer_id = str(
            (saved or {}).get("provider_customer_id") or ""
        ).strip()

        if not customer_id:
            raise HTTPException(
                status_code=409,
                detail="No Stripe billing profile is available for this account.",
            )

        try:
            portal = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=f"{FRONTEND_URL}/profile",
            )
        except stripe.StripeError as exc:
            message = getattr(exc, "user_message", None) or str(exc)
            raise HTTPException(
                status_code=502,
                detail=f"Unable to open Stripe billing portal: {message}",
            )

        return {"portal_url": portal.url}

    @router.post("/subscription/cancel")
    def cancel_subscription(
        authorization: Optional[str] = Header(default=None),
    ):
        if not STRIPE_SECRET_KEY:
            raise HTTPException(
                status_code=503,
                detail="Stripe is not configured on the server.",
            )

        user = require_user(authorization)
        saved = require_monthly_subscription(user["id"])
        provider_subscription_id = str(saved["provider_subscription_id"])

        try:
            subscription = stripe.Subscription.modify(
                provider_subscription_id,
                cancel_at_period_end=True,
            )
        except stripe.StripeError as exc:
            message = getattr(exc, "user_message", None) or str(exc)
            raise HTTPException(
                status_code=502,
                detail=f"Unable to cancel renewal: {message}",
            )

        period_start, period_end = _subscription_period(subscription)
        meta = _metadata(subscription)
        meta.update(saved.get("metadata") or {})

        save_subscription(
            user_id=user["id"],
            plan_code="pro_monthly",
            provider_customer_id=str(
                _object_value(subscription, "customer", "")
            ) or saved.get("provider_customer_id"),
            provider_subscription_id=provider_subscription_id,
            status=str(_object_value(subscription, "status", "active")),
            period_start=period_start,
            period_end=period_end,
            cancel_at_period_end=True,
            metadata=meta,
        )

        return {
            "ok": True,
            "status": str(_object_value(subscription, "status", "active")),
            "cancel_at_period_end": True,
            "current_period_end": _iso(period_end),
            "message": "Automatic renewal has been canceled.",
        }

    @router.post("/subscription/reactivate")
    def reactivate_subscription(
        authorization: Optional[str] = Header(default=None),
    ):
        if not STRIPE_SECRET_KEY:
            raise HTTPException(
                status_code=503,
                detail="Stripe is not configured on the server.",
            )

        user = require_user(authorization)
        saved = require_monthly_subscription(user["id"])
        provider_subscription_id = str(saved["provider_subscription_id"])

        try:
            subscription = stripe.Subscription.modify(
                provider_subscription_id,
                cancel_at_period_end=False,
            )
        except stripe.StripeError as exc:
            message = getattr(exc, "user_message", None) or str(exc)
            raise HTTPException(
                status_code=502,
                detail=f"Unable to reactivate renewal: {message}",
            )

        period_start, period_end = _subscription_period(subscription)
        meta = _metadata(subscription)
        meta.update(saved.get("metadata") or {})

        save_subscription(
            user_id=user["id"],
            plan_code="pro_monthly",
            provider_customer_id=str(
                _object_value(subscription, "customer", "")
            ) or saved.get("provider_customer_id"),
            provider_subscription_id=provider_subscription_id,
            status=str(_object_value(subscription, "status", "active")),
            period_start=period_start,
            period_end=period_end,
            cancel_at_period_end=False,
            metadata=meta,
        )

        return {
            "ok": True,
            "status": str(_object_value(subscription, "status", "active")),
            "cancel_at_period_end": False,
            "current_period_end": _iso(period_end),
            "message": "Automatic renewal has been reactivated.",
        }

    @router.post("/webhook")
    async def stripe_webhook(
        request: Request,
        stripe_signature: Optional[str] = Header(
            default=None,
            alias="stripe-signature",
        ),
    ):
        if not STRIPE_WEBHOOK_SECRET:
            raise HTTPException(
                status_code=503,
                detail="Stripe webhook is not configured.",
            )
        if not stripe_signature:
            raise HTTPException(
                status_code=400,
                detail="Missing Stripe signature.",
            )

        raw_body = await request.body()

        try:
            event = stripe.Webhook.construct_event(
                payload=raw_body,
                sig_header=stripe_signature,
                secret=STRIPE_WEBHOOK_SECRET,
            )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid webhook body.")
        except stripe.SignatureVerificationError:
            raise HTTPException(
                status_code=400,
                detail="Invalid Stripe webhook signature.",
            )

        event_id = str(_object_value(event, "id"))
        event_type = str(_object_value(event, "type"))
        data = _object_value(event, "data", {})
        data_object = _object_value(data, "object", {})

        event_payload = json.loads(raw_body.decode("utf-8"))

        if not begin_event(event_id, event_type, event_payload):
            return {"received": True, "duplicate": True}

        handlers = {
            "checkout.session.completed": process_checkout_completed,
            "checkout.session.async_payment_succeeded": process_checkout_completed,
            "invoice.paid": process_invoice_paid,
            "invoice.payment_failed": process_invoice_failed,
            "customer.subscription.updated": process_subscription_updated,
            "customer.subscription.deleted": process_subscription_deleted,
        }

        handler = handlers.get(event_type)
        if handler is None:
            finish_event(event_id, "ignored")
            return {"received": True, "ignored": True}

        try:
            handler(data_object)
            finish_event(event_id, "processed")
        except Exception as exc:
            finish_event(event_id, "failed", str(exc))
            raise HTTPException(
                status_code=500,
                detail="Stripe event processing failed.",
            )

        return {"received": True, "processed": event_type}

    return router
