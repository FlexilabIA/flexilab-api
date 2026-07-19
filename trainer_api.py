from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field


FRONTEND_URL = os.environ.get(
    "FRONTEND_URL",
    "https://flexi-move-lab.lovable.app",
).rstrip("/")


class TrainerRegister(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    company_name: str = Field(min_length=2, max_length=160)


class TrainerClientCreate(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    email: str = Field(min_length=5, max_length=254)


class TrainerClientUpdate(BaseModel):
    full_name: Optional[str] = Field(default=None, min_length=2, max_length=120)
    status: Optional[str] = None


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_trainer_router(supabase_client) -> APIRouter:
    router = APIRouter(tags=["trainer"])

    def require_user(
        authorization: Optional[str] = Header(default=None),
    ) -> dict[str, str]:
        if supabase_client is None:
            raise HTTPException(status_code=503, detail="Supabase is not configured.")
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing authentication token.")
        token = authorization.split(" ", 1)[1].strip()
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
                raise ValueError("Missing user id")
            return {
                "id": str(user_id),
                "email": str(email or "").strip().lower(),
            }
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid authentication token.")

    def trainer_profile(user_id: str) -> Optional[dict[str, Any]]:
        response = (
            supabase_client.table("trainer_profiles")
            .select("*")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    def require_trainer(user: dict[str, str]) -> dict[str, Any]:
        profile = trainer_profile(user["id"])
        if not profile or profile.get("status") != "active":
            raise HTTPException(status_code=403, detail="Trainer account required.")
        return profile

    def remaining_tokens(user_id: str) -> tuple[int, Optional[str]]:
        now = datetime.now(timezone.utc)
        response = (
            supabase_client.table("screening_credit_cycles")
            .select("cycle_end,grace_expires_at,credits_granted,credits_used,cycle_start")
            .eq("user_id", user_id)
            .order("cycle_end", desc=False)
            .execute()
        )
        total = 0
        expiry: Optional[str] = None
        for row in response.data or []:
            try:
                start = datetime.fromisoformat(str(row.get("cycle_start")).replace("Z", "+00:00"))
                end_raw = row.get("grace_expires_at") or row.get("cycle_end")
                end = datetime.fromisoformat(str(end_raw).replace("Z", "+00:00"))
            except Exception:
                continue
            if start <= now <= end:
                available = max(0, int(row.get("credits_granted") or 0) - int(row.get("credits_used") or 0))
                total += available
                if available and (expiry is None or str(end_raw) < expiry):
                    expiry = str(end_raw)
        return total, expiry

    def client_link(trainer_id: str, link_id: str) -> dict[str, Any]:
        response = (
            supabase_client.table("trainer_clients")
            .select("*")
            .eq("id", link_id)
            .eq("trainer_id", trainer_id)
            .limit(1)
            .execute()
        )
        if not response.data:
            raise HTTPException(status_code=404, detail="Client not found.")
        return response.data[0]

    def latest_client_session(client_user_id: Optional[str], trainer_id: str) -> Optional[dict[str, Any]]:
        if not client_user_id:
            return None
        query = (
            supabase_client.table("sessions")
            .select("id,created_at,status,composite_score")
            .eq("trainer_id", trainer_id)
            .order("created_at", desc=True)
            .limit(1)
        )
        if client_user_id:
            query = query.eq("user_id", client_user_id)
        response = query.execute()
        return response.data[0] if response.data else None

    @router.post("/trainer/register")
    def register_trainer(
        payload: TrainerRegister,
        user: dict[str, str] = Depends(require_user),
    ):
        now = datetime.now(timezone.utc)
        trial_end = now + timedelta(days=30)

        profile_row = {
            "user_id": user["id"],
            "status": "active",
            "full_name": payload.full_name.strip(),
            "company_name": payload.company_name.strip(),
            "updated_at": _iso_now(),
        }
        profile_response = (
            supabase_client.table("trainer_profiles")
            .upsert(profile_row, on_conflict="user_id")
            .execute()
        )
        if not profile_response.data:
            raise HTTPException(
                status_code=500,
                detail="Unable to create the Trainer profile.",
            )

        # One free screening token, granted once per Trainer account.
        existing_trial = (
            supabase_client.table("screening_credit_cycles")
            .select("id")
            .eq("user_id", user["id"])
            .eq("source", "trainer_free_trial")
            .limit(1)
            .execute()
        )
        if not existing_trial.data:
            supabase_client.table("screening_credit_cycles").insert({
                "user_id": user["id"],
                "subscription_id": None,
                "source": "trainer_free_trial",
                "cycle_start": now.isoformat(),
                "cycle_end": trial_end.isoformat(),
                "grace_expires_at": trial_end.isoformat(),
                "credits_granted": 1,
                "credits_used": 0,
            }).execute()

        try:
            supabase_client.table("profiles").update({
                "full_name": payload.full_name.strip(),
                "updated_at": _iso_now(),
            }).eq("id", user["id"]).execute()
        except Exception:
            pass

        tokens, expires_at = remaining_tokens(user["id"])
        return {
            "trainer": profile_response.data[0],
            "tokens_remaining": tokens,
            "tokens_expires_at": expires_at,
            "free_token_granted": not bool(existing_trial.data),
        }

    @router.get("/me/account-mode")
    def account_mode(user: dict[str, str] = Depends(require_user)):
        # An invited client becomes active the first time they authenticate.
        try:
            supabase_client.table("trainer_clients").update({
                "client_user_id": user["id"],
                "status": "active",
                "accepted_at": _iso_now(),
                "updated_at": _iso_now(),
            }).ilike("invited_email", user["email"]).eq("status", "pending").execute()
        except Exception:
            pass
        profile = trainer_profile(user["id"])
        return {
            "mode": "trainer" if profile and profile.get("status") == "active" else "client",
            "is_trainer": bool(profile and profile.get("status") == "active"),
        }

    @router.get("/trainer/overview")
    def overview(user: dict[str, str] = Depends(require_user)):
        profile = require_trainer(user)
        clients_response = (
            supabase_client.table("trainer_clients")
            .select("id,status")
            .eq("trainer_id", user["id"])
            .neq("status", "archived")
            .execute()
        )
        sessions_response = (
            supabase_client.table("sessions")
            .select("id,status,created_at")
            .eq("trainer_id", user["id"])
            .eq("status", "completed")
            .execute()
        )
        reward_response = (
            supabase_client.table("trainer_referral_rewards")
            .select("tokens_granted")
            .eq("trainer_id", user["id"])
            .execute()
        )
        tokens, expires_at = remaining_tokens(user["id"])
        return {
            "trainer": profile,
            "tokens_remaining": tokens,
            "tokens_expires_at": expires_at,
            "total_clients": len(clients_response.data or []),
            "active_clients": sum(
                1 for row in (clients_response.data or [])
                if row.get("status") == "active"
            ),
            "completed_screenings": len(sessions_response.data or []),
            "referral_tokens_earned": sum(int(x.get("tokens_granted") or 0) for x in reward_response.data or []),
        }

    @router.get("/trainer/clients")
    def list_clients(user: dict[str, str] = Depends(require_user)):
        require_trainer(user)
        response = (
            supabase_client.table("trainer_clients")
            .select("*")
            .eq("trainer_id", user["id"])
            .neq("status", "archived")
            .order("created_at", desc=True)
            .execute()
        )
        result = []
        for row in response.data or []:
            latest = latest_client_session(row.get("client_user_id"), user["id"])
            result.append({**row, "latest_screening": latest})
        return {"clients": result}

    @router.post("/trainer/clients")
    def create_client(payload: TrainerClientCreate, user: dict[str, str] = Depends(require_user)):
        require_trainer(user)
        email = str(payload.email).strip().lower()
        name = payload.full_name.strip()

        existing_link = (
            supabase_client.table("trainer_clients")
            .select("*")
            .eq("trainer_id", user["id"])
            .eq("invited_email", email)
            .limit(1)
            .execute()
        )
        if existing_link.data:
            return {"client": existing_link.data[0], "already_exists": True}

        profile_response = (
            supabase_client.table("profiles")
            .select("id,email,full_name")
            .ilike("email", email)
            .limit(1)
            .execute()
        )
        client_user_id = None
        status = "pending"
        invite_sent = False

        if profile_response.data:
            client_user_id = str(profile_response.data[0].get("id") or "") or None
            status = "pending"
        else:
            try:
                invited = supabase_client.auth.admin.invite_user_by_email(
                    email,
                    options={"redirect_to": f"{FRONTEND_URL}/reset-password"},
                )
                invited_user = getattr(invited, "user", None)
                client_user_id = str(getattr(invited_user, "id", "") or "") or None
                invite_sent = True
                if client_user_id:
                    supabase_client.table("profiles").upsert({
                        "id": client_user_id,
                        "email": email,
                        "full_name": name,
                        "language": "en",
                        "account_status": "active",
                    }).execute()
            except Exception:
                # Keep the pending client record even if email delivery is not configured.
                invite_sent = False

        row = {
            "trainer_id": user["id"],
            "client_user_id": client_user_id,
            "invited_email": email,
            "client_name": name,
            "status": status,
            "accepted_at": _iso_now() if status == "active" else None,
            "updated_at": _iso_now(),
        }
        created = supabase_client.table("trainer_clients").insert(row).execute()
        if not created.data:
            raise HTTPException(status_code=500, detail="Unable to create client.")
        return {"client": created.data[0], "invite_sent": invite_sent}

    @router.get("/trainer/clients/{link_id}")
    def get_client(link_id: str, user: dict[str, str] = Depends(require_user)):
        require_trainer(user)
        link = client_link(user["id"], link_id)
        sessions = (
            supabase_client.table("sessions")
            .select("id,created_at,status,composite_score,user_id,user_email")
            .eq("trainer_client_link_id", link_id)
            .order("created_at", desc=True)
            .execute()
        )
        return {"client": link, "screenings": sessions.data or []}

    @router.patch("/trainer/clients/{link_id}")
    def update_client(link_id: str, payload: TrainerClientUpdate, user: dict[str, str] = Depends(require_user)):
        require_trainer(user)
        client_link(user["id"], link_id)
        changes: dict[str, Any] = {"updated_at": _iso_now()}
        if payload.full_name is not None:
            changes["client_name"] = payload.full_name.strip()
        if payload.status is not None:
            status = payload.status.strip().lower()
            if status not in {"pending", "active", "archived", "revoked"}:
                raise HTTPException(status_code=422, detail="Invalid client status.")
            changes["status"] = status
            if status == "archived":
                changes["archived_at"] = _iso_now()
        response = (
            supabase_client.table("trainer_clients")
            .update(changes)
            .eq("id", link_id)
            .eq("trainer_id", user["id"])
            .execute()
        )
        return {"client": response.data[0] if response.data else None}

    @router.get("/trainer/token-history")
    def token_history(user: dict[str, str] = Depends(require_user)):
        require_trainer(user)
        cycles = (
            supabase_client.table("screening_credit_cycles")
            .select("id,source,cycle_start,cycle_end,grace_expires_at,credits_granted,credits_used,created_at")
            .eq("user_id", user["id"])
            .order("created_at", desc=True)
            .execute()
        )
        rewards = (
            supabase_client.table("trainer_referral_rewards")
            .select("id,client_user_id,source_payment_id,tokens_granted,created_at")
            .eq("trainer_id", user["id"])
            .order("created_at", desc=True)
            .execute()
        )
        return {"cycles": cycles.data or [], "referral_rewards": rewards.data or []}

    return router
