from __future__ import annotations

import os
from typing import Any, Optional

import stripe
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel


STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "").strip()
STRIPE_PRICE_MONTHLY = os.environ.get("STRIPE_PRICE_MONTHLY", "").strip()
STRIPE_PRICE_THREE_MONTH = os.environ.get("STRIPE_PRICE_THREE_MONTH", "").strip()
STRIPE_PRICE_ANNUAL = os.environ.get("STRIPE_PRICE_ANNUAL", "").strip()
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
    },
    "pro_three_month": {
        "price_id": STRIPE_PRICE_THREE_MONTH,
        "mode": "payment",
    },
    "pro_annual": {
        "price_id": STRIPE_PRICE_ANNUAL,
        "mode": "payment",
    },
}


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

    @router.get("/status")
    def stripe_status():
        return {
            "configured": bool(
                STRIPE_SECRET_KEY
                and STRIPE_PRICE_MONTHLY
                and STRIPE_PRICE_THREE_MONTH
                and STRIPE_PRICE_ANNUAL
            ),
            "monthly_price_configured": bool(STRIPE_PRICE_MONTHLY),
            "three_month_price_configured": bool(STRIPE_PRICE_THREE_MONTH),
            "annual_price_configured": bool(STRIPE_PRICE_ANNUAL),
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
                    "Invalid plan_code. Use pro_monthly, "
                    "pro_three_month, or pro_annual."
                ),
            )

        price_id = str(plan["price_id"] or "").strip()
        if not price_id:
            raise HTTPException(
                status_code=503,
                detail=f"Stripe price is not configured for {plan_code}.",
            )

        metadata = {
            "user_id": user["id"],
            "user_email": user["email"],
            "plan_code": plan_code,
        }

        session_params: dict[str, Any] = {
            "mode": plan["mode"],
            "line_items": [{"price": price_id, "quantity": 1}],
            "customer_email": user["email"],
            "client_reference_id": user["id"],
            "metadata": metadata,
            "success_url": (
                f"{FRONTEND_URL}/paywall"
                "?payment=success"
                "&session_id={CHECKOUT_SESSION_ID}"
            ),
            "cancel_url": f"{FRONTEND_URL}/paywall?payment=cancelled",
            "allow_promotion_codes": False,
        }

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

    return router
