from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field


logger = logging.getLogger("flexilab.trainer")

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
    language: str = Field(default="en", max_length=2)


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

    def _read_with_retry(operation, label: str):
        """Retry one transient read without repeating writes or user actions."""
        last_error = None
        for attempt in range(2):
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(0.12)
                    continue
                logger.warning("trainer_read_failed label=%s error=%s", label, exc)
        raise last_error

    REQUIRED_HISTORY_TESTS = {
        "posture_side",
        "shoulder_right",
        "shoulder_left",
        "squat",
        "aslr_right",
        "aslr_left",
    }

    def _latest_complete_assessment_sessions(
        sessions: list[dict[str, Any]],
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        """Keep finalized sessions plus legacy sessions with all six tests.

        New sessions should have status=completed. Some valid historical
        sessions predate reliable finalization; they are accepted only when the
        complete six-test evidence exists. Partial sessions are always removed.
        """
        if not sessions:
            return []
        session_ids = [str(row.get("id") or "") for row in sessions if row.get("id")]
        tests_by_session: dict[str, set[str]] = {sid: set() for sid in session_ids}
        if session_ids:
            response = _read_with_retry(
                lambda: (
                    supabase_client.table("screenings")
                    .select("session_id,test_type")
                    .in_("session_id", session_ids)
                    .execute()
                ),
                "trainer_complete_assessment_evidence",
            )
            for row in response.data or []:
                sid = str(row.get("session_id") or "")
                test_type = str(row.get("test_type") or "")
                if sid in tests_by_session and test_type:
                    tests_by_session[sid].add(test_type)

        eligible = []
        for row in sessions:
            sid = str(row.get("id") or "")
            finalized = str(row.get("status") or "").strip().lower() == "completed"
            has_complete_evidence = REQUIRED_HISTORY_TESTS.issubset(tests_by_session.get(sid, set()))
            if finalized or has_complete_evidence:
                normalized = dict(row)
                normalized["status"] = "completed"
                normalized["stored_status"] = row.get("status")
                eligible.append(normalized)
            if len(eligible) >= max(1, limit):
                break
        return eligible

    def latest_client_sessions_bulk(
        rows: list[dict[str, Any]],
        trainer_id: str,
    ) -> dict[str, dict[str, Any]]:
        """Return one latest session per client link without N+1 queries."""
        link_ids = [str(row.get("id") or "") for row in rows if row.get("id")]
        if not link_ids:
            return {}

        response = _read_with_retry(
            lambda: (
                supabase_client.table("sessions")
                .select("id,created_at,status,composite_score,trainer_client_link_id,user_id")
                .eq("trainer_id", trainer_id)
                .in_("trainer_client_link_id", link_ids)
                .eq("status", "completed")
                .order("created_at", desc=True)
                .execute()
            ),
            "latest_client_sessions",
        )
        latest: dict[str, dict[str, Any]] = {}
        for session in response.data or []:
            link_id = str(session.get("trainer_client_link_id") or "")
            if link_id and link_id not in latest:
                latest[link_id] = session

        # Legacy compatibility in one bulk query for links created before the
        # trainer_client_link_id field was persisted on sessions.
        missing_rows = [row for row in rows if str(row.get("id") or "") not in latest]
        user_to_link = {
            str(row.get("client_user_id")): str(row.get("id"))
            for row in missing_rows
            if row.get("client_user_id") and row.get("id")
        }
        if user_to_link:
            fallback = _read_with_retry(
                lambda: (
                    supabase_client.table("sessions")
                    .select("id,created_at,status,composite_score,user_id")
                    .eq("trainer_id", trainer_id)
                    .in_("user_id", list(user_to_link))
                    .eq("status", "completed")
                    .order("created_at", desc=True)
                    .execute()
                ),
                "legacy_latest_client_sessions",
            )
            seen_users: set[str] = set()
            for session in fallback.data or []:
                user_id = str(session.get("user_id") or "")
                if not user_id or user_id in seen_users:
                    continue
                link_id = user_to_link.get(user_id)
                if link_id:
                    latest[link_id] = session
                    seen_users.add(user_id)
        return latest

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
        try:
            account_profile_response = (
                supabase_client.table("profiles")
                .select("full_name,language,email")
                .eq("id", user["id"])
                .limit(1)
                .execute()
            )
            account_profile = (
                account_profile_response.data[0]
                if account_profile_response.data
                else {}
            )
            current_name = str(account_profile.get("full_name") or "").strip()
            current_language = str(account_profile.get("language") or "").strip()
            if current_name:
                profile["full_name"] = current_name
            if current_language in {"en", "fr"}:
                profile["language"] = current_language
            if not profile.get("email"):
                profile["email"] = account_profile.get("email") or user.get("email")
        except Exception:
            pass

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
        page_size: int = Query(default=20, ge=1, le=20),
        q: Optional[str] = None,
        user: dict[str, str] = Depends(require_user),
    ):
        require_trainer(user)
        started = time.perf_counter()
        safe_page = max(1, int(page))
        search = str(q or "").strip().replace(",", " ")[:120]
        safe_size = 20 if search else min(max(1, int(page_size)), 5)
        start = (safe_page - 1) * safe_size
        query = (
            supabase_client.table("trainer_clients")
            .select("*", count="exact")
            .eq("trainer_id", user["id"])
            .neq("status", "archived")
        )
        if search:
            query = query.or_(
                f"client_name.ilike.%{search}%,invited_email.ilike.%{search}%"
            )
        response = _read_with_retry(
            lambda: (
                query.order("updated_at", desc=True)
                .order("created_at", desc=True)
                .range(start, start + safe_size - 1)
                .execute()
            ),
            "trainer_clients",
        )
        rows = response.data or []
        latest_by_link = latest_client_sessions_bulk(rows, user["id"])
        result = [
            {**row, "latest_screening": latest_by_link.get(str(row.get("id") or ""))}
            for row in rows
        ]
        logger.info(
            "PERF path=/trainer/clients duration_ms=%d rows=%d search=%s",
            round((time.perf_counter() - started) * 1000),
            len(result),
            bool(search),
        )
        return {
            "clients": result,
            "page": safe_page,
            "page_size": safe_size,
            "total": int(getattr(response, "count", len(result)) or len(result)),
        }


    @router.get("/trainer/bootstrap")
    def trainer_bootstrap(user: dict[str, str] = Depends(require_user)):
        """Return one compact, authoritative Trainer workspace snapshot."""
        started = time.perf_counter()
        profile = require_trainer(user)

        account_profile_response = _read_with_retry(
            lambda: (
                supabase_client.table("profiles")
                .select("full_name,language,email")
                .eq("id", user["id"])
                .limit(1)
                .execute()
            ),
            "bootstrap_account_profile",
        )
        account_profile = (
            account_profile_response.data[0]
            if account_profile_response.data
            else {}
        )
        current_name = str(account_profile.get("full_name") or "").strip()
        current_language = str(account_profile.get("language") or "").strip()

        if current_name:
            profile["full_name"] = current_name
        if current_language in {"en", "fr"}:
            profile["language"] = current_language
        if not profile.get("email"):
            profile["email"] = account_profile.get("email") or user.get("email")

        clients_response = _read_with_retry(
            lambda: (
                supabase_client.table("trainer_clients")
                .select("*", count="exact")
                .eq("trainer_id", user["id"])
                .neq("status", "archived")
                .order("updated_at", desc=True)
                .order("created_at", desc=True)
                .limit(20)
                .execute()
            ),
            "bootstrap_recent_clients",
        )
        recent_rows = clients_response.data or []
        latest_by_link = latest_client_sessions_bulk(recent_rows, user["id"])
        recent_clients = [
            {**row, "latest_screening": latest_by_link.get(str(row.get("id") or ""))}
            for row in recent_rows
        ]
        recent_clients.sort(
            key=lambda row: str(
                (row.get("latest_screening") or {}).get("created_at")
                or row.get("updated_at")
                or row.get("created_at")
                or ""
            ),
            reverse=True,
        )

        client_counts = _read_with_retry(
            lambda: (
                supabase_client.table("trainer_clients")
                .select("id,status")
                .eq("trainer_id", user["id"])
                .neq("status", "archived")
                .execute()
            ),
            "bootstrap_client_counts",
        )
        completed = _read_with_retry(
            lambda: (
                supabase_client.table("sessions")
                .select("id")
                .eq("performed_by_user_id", user["id"])
                .eq("status", "completed")
                .execute()
            ),
            "bootstrap_completed_screenings",
        )
        rewards = _read_with_retry(
            lambda: (
                supabase_client.table("trainer_referral_rewards")
                .select("tokens_granted")
                .eq("trainer_id", user["id"])
                .execute()
            ),
            "bootstrap_referral_rewards",
        )
        personal = _read_with_retry(
            lambda: (
                supabase_client.table("sessions")
                .select("id,created_at,status,composite_score,user_id,user_email")
                .eq("user_id", user["id"])
                .eq("performed_by_user_id", user["id"])
                .is_("trainer_client_link_id", "null")
                .eq("status", "completed")
                .order("created_at", desc=True)
                .limit(6)
                .execute()
            ),
            "bootstrap_self_screenings",
        )
        tokens, expires_at = remaining_tokens(user["id"])
        all_clients = client_counts.data or []
        overview_payload = {
            "trainer": profile,
            "tokens_remaining": tokens,
            "tokens_expires_at": expires_at,
            "total_clients": len(all_clients),
            "active_clients": sum(1 for row in all_clients if row.get("status") == "active"),
            "completed_screenings": len(completed.data or []),
            "referral_tokens_earned": sum(
                int(row.get("tokens_granted") or 0) for row in rewards.data or []
            ),
        }

        payload = {
            "overview": overview_payload,
            "clients": recent_clients[:5],
            "clients_page": 1,
            "clients_page_size": 5,
            "clients_total": len(all_clients),
            "self_screenings": personal.data or [],
            # Full history is requested only when the Tokens page is opened.
            "token_history": {"cycles": [], "referral_rewards": []},
            "loaded_at": _iso_now(),
            "bootstrap_version": "trainer-bootstrap-v3-completed-client-history",
        }
        logger.info(
            "PERF path=/trainer/bootstrap duration_ms=%d recent_clients=%d",
            round((time.perf_counter() - started) * 1000),
            len(payload["clients"]),
        )
        return payload

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
                invite_language = "fr" if payload.language == "fr" else "en"
                invited = supabase_client.auth.admin.invite_user_by_email(
                    email,
                    options={
                        "redirect_to": f"{FRONTEND_URL}/reset-password",
                        "data": {
                            "full_name": name,
                            "language": invite_language,
                            "invited_client": True,
                            "welcome_demo_required": True,
                            "welcome_demo_completed": False,
                        },
                    },
                )
                invited_user = getattr(invited, "user", None)
                client_user_id = str(getattr(invited_user, "id", "") or "") or None
                invite_sent = True
                if client_user_id:
                    supabase_client.table("profiles").upsert({
                        "id": client_user_id,
                        "email": email,
                        "full_name": name,
                        "language": invite_language,
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
            .limit(48)
            .execute()
        )
        completed_sessions = _latest_complete_assessment_sessions(sessions.data or [], limit=6)
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
            "screenings": completed_sessions,
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
                        "invited_client": True,
                        "welcome_demo_required": True,
                        "welcome_demo_completed": False,
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
            .limit(48)
            .execute()
        )
        return {"screenings": _latest_complete_assessment_sessions(response.data or [], limit=6)}

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
