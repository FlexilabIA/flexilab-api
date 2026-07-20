from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
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


class TrainerNoteCreate(BaseModel):
    note: str = Field(min_length=1, max_length=5000)
    session_id: Optional[str] = None


class TrainerInviteResend(BaseModel):
    language: str = Field(default="en", max_length=2)


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

    def latest_client_session(
        link_id: str,
        client_user_id: Optional[str],
        trainer_id: str,
    ) -> Optional[dict[str, Any]]:
        query = (
            supabase_client.table("sessions")
            .select("id,created_at,status,composite_score")
            .eq("trainer_id", trainer_id)
            .eq("trainer_client_link_id", link_id)
            .order("created_at", desc=True)
            .limit(1)
        )
        response = query.execute()
        if response.data:
            return response.data[0]

        # Compatibility for old active-client sessions created before link IDs were stored.
        if client_user_id:
            fallback = (
                supabase_client.table("sessions")
                .select("id,created_at,status,composite_score")
                .eq("trainer_id", trainer_id)
                .eq("user_id", client_user_id)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            return fallback.data[0] if fallback.data else None
        return None

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

        # Two free screening tokens, granted once per Trainer account.
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
                "credits_granted": 2,
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
            "free_tokens_granted": 2 if not existing_trial.data else 0,
        }

    @router.get("/me/account-mode")
    def account_mode(user: dict[str, str] = Depends(require_user)):
        # An invited client becomes active the first time they authenticate.
        # Screenings created before activation may have user_id=NULL; attach them
        # to the newly authenticated client account using the secure link ID.
        try:
            matching_links = (
                supabase_client.table("trainer_clients")
                .select("id,status,client_user_id")
                .ilike("invited_email", user["email"])
                .in_("status", ["pending", "active"])
                .execute()
            )

            supabase_client.table("trainer_clients").update({
                "client_user_id": user["id"],
                "status": "active",
                "accepted_at": _iso_now(),
                "updated_at": _iso_now(),
            }).ilike("invited_email", user["email"]).in_("status", ["pending", "active"]).execute()

            for link in matching_links.data or []:
                link_id = str(link.get("id") or "").strip()
                if not link_id:
                    continue
                linked_sessions = (
                    supabase_client.table("sessions")
                    .select("id")
                    .eq("trainer_client_link_id", link_id)
                    .execute()
                )
                session_ids = [str(row.get("id")) for row in (linked_sessions.data or []) if row.get("id")]
                supabase_client.table("sessions").update({
                    "user_id": user["id"],
                    "user_email": user["email"],
                }).eq("trainer_client_link_id", link_id).execute()

                if session_ids:
                    supabase_client.table("screenings").update({
                        "user_id": user["id"],
                        "user_email": user["email"],
                    }).in_("session_id", session_ids).execute()
                    supabase_client.table("analysis_jobs").update({
                        "user_id": user["id"],
                        "user_email": user["email"],
                    }).in_("session_id", session_ids).execute()
                    supabase_client.table("screening_history").update({
                        "user_id": user["id"],
                        "user_email": user["email"],
                    }).in_("session_id", session_ids).execute()
                    supabase_client.table("corrective_programs").update({
                        "user_id": user["id"],
                        "user_email": user["email"],
                    }).in_("screening_session_id", session_ids).execute()
        except Exception:
            # Activation should still succeed if one legacy compatibility table is absent.
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
            .eq("performed_by_user_id", user["id"])
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
    def list_clients(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=100),
        q: Optional[str] = None,
        user: dict[str, str] = Depends(require_user),
    ):
        require_trainer(user)
        safe_page = max(1, int(page))
        safe_size = max(1, min(int(page_size), 100))
        start = (safe_page - 1) * safe_size
        query = (
            supabase_client.table("trainer_clients")
            .select("*", count="exact")
            .eq("trainer_id", user["id"])
            .neq("status", "archived")
        )
        search = str(q or "").strip().replace(",", " ")[:120]
        if search:
            query = query.or_(
                f"client_name.ilike.%{search}%,invited_email.ilike.%{search}%"
            )
        response = (
            query.order("created_at", desc=True)
            .range(start, start + safe_size - 1)
            .execute()
        )
        result = []
        for row in response.data or []:
            latest = latest_client_session(
                str(row.get("id") or ""),
                row.get("client_user_id"),
                user["id"],
            )
            result.append({**row, "latest_screening": latest})
        return {
            "clients": result,
            "page": safe_page,
            "page_size": safe_size,
            "total": int(getattr(response, "count", len(result)) or len(result)),
        }

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
        notes = (
            supabase_client.table("trainer_client_notes")
            .select("id,session_id,note,created_at,updated_at")
            .eq("trainer_id", user["id"])
            .eq("trainer_client_link_id", link_id)
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )
        return {
            "client": link,
            "screenings": sessions.data or [],
            "notes": notes.data or [],
        }

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


    @router.get("/trainer/clients/{link_id}/screenings/{session_id}")
    def get_client_screening(
        link_id: str,
        session_id: str,
        user: dict[str, str] = Depends(require_user),
    ):
        require_trainer(user)
        link = client_link(user["id"], link_id)
        response = (
            supabase_client.table("sessions")
            .select("id,created_at,status,composite_score,user_id,user_email,trainer_id,performed_by_user_id,trainer_client_link_id")
            .eq("id", session_id)
            .eq("trainer_id", user["id"])
            .eq("trainer_client_link_id", link_id)
            .limit(1)
            .execute()
        )
        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="This screening is not attached to the selected Trainer-client profile.",
            )
        return {"client": link, "screening": response.data[0]}

    @router.post("/trainer/clients/{link_id}/notes")
    def add_client_note(
        link_id: str,
        payload: TrainerNoteCreate,
        user: dict[str, str] = Depends(require_user),
    ):
        require_trainer(user)
        client_link(user["id"], link_id)
        if payload.session_id:
            session = (
                supabase_client.table("sessions")
                .select("id")
                .eq("id", payload.session_id)
                .eq("trainer_id", user["id"])
                .eq("trainer_client_link_id", link_id)
                .limit(1)
                .execute()
            )
            if not session.data:
                raise HTTPException(status_code=404, detail="Screening not found for this client.")
        result = supabase_client.table("trainer_client_notes").insert({
            "trainer_id": user["id"],
            "trainer_client_link_id": link_id,
            "session_id": payload.session_id,
            "note": payload.note.strip(),
            "updated_at": _iso_now(),
        }).execute()
        return {"note": (result.data or [None])[0]}

    @router.post("/trainer/clients/{link_id}/resend-invite")
    def resend_client_invite(
        link_id: str,
        payload: TrainerInviteResend,
        user: dict[str, str] = Depends(require_user),
    ):
        require_trainer(user)
        link = client_link(user["id"], link_id)
        if link.get("status") not in {"pending", "active"}:
            raise HTTPException(status_code=409, detail="This invitation cannot be resent.")
        email = str(link.get("invited_email") or "").strip().lower()
        try:
            invited = supabase_client.auth.admin.invite_user_by_email(
                email,
                options={
                    "redirect_to": f"{FRONTEND_URL}/reset-password",
                    "data": {
                        "full_name": link.get("client_name"),
                        "language": "fr" if payload.language == "fr" else "en",
                    },
                },
            )
            invited_user = getattr(invited, "user", None)
            invited_user_id = str(getattr(invited_user, "id", "") or "") or None
            if invited_user_id and not link.get("client_user_id"):
                supabase_client.table("trainer_clients").update({
                    "client_user_id": invited_user_id,
                    "updated_at": _iso_now(),
                }).eq("id", link_id).eq("trainer_id", user["id"]).execute()
            return {"sent": True, "email": email}
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Unable to resend invitation: {exc}")


    @router.get("/trainer/self-screenings")
    def self_screenings(user: dict[str, str] = Depends(require_user)):
        """
        Return only assessments performed by the Trainer on their own account.

        Client assessments remain available exclusively inside the corresponding
        Trainer-client profile through /trainer/clients/{link_id}.
        """
        require_trainer(user)
        response = (
            supabase_client.table("sessions")
            .select("id,created_at,status,composite_score,user_id,user_email")
            .eq("user_id", user["id"])
            .eq("performed_by_user_id", user["id"])
            .is_("trainer_client_link_id", "null")
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )
        return {"screenings": response.data or []}

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
