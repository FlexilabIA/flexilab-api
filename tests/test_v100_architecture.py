from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


class FlexiLabV100ArchitectureTests(unittest.TestCase):
    def test_operator_api_has_role_permissions_and_core_routes(self) -> None:
        source = text("operator_api.py")
        for role in (
            "super_admin",
            "admin",
            "operations",
            "support",
            "finance",
            "clinical_reviewer",
            "organization_manager",
        ):
            self.assertIn(f'"{role}"', source)
        for route in (
            "/dashboard",
            "/users",
            "/trainers",
            "/screenings",
            "/finance",
            "/vouchers",
            "/organizations",
            "/imports",
            "/health",
            "/audit",
        ):
            self.assertIn(f'"{route}"', source)
        self.assertIn("Missing Operator permission", source)

    def test_operator_mutations_are_audited(self) -> None:
        source = text("operator_api.py")
        self.assertIn('supabase_client.table("audit_logs").insert', source)
        for action in (
            "operator.user_status_updated",
            "operator.grant_created",
            "operator.role_updated",
            "operator.trainer_status_updated",
            "operator.analysis_job_retried",
            "operator.voucher_created",
            "operator.organization_created",
            "operator.bulk_import_queued",
        ):
            self.assertIn(action, source)

    def test_program_is_owner_only_and_entitlement_guarded(self) -> None:
        source = text("app.py")
        program_section = source[source.index('@app.get("/program")') :]
        self.assertIn("require_owned_session(program_user, session_id)", program_section)
        self.assertIn("Only the client who owns this screening", program_section)
        self.assertIn("effective_entitlement(supabase, program_user", program_section)
        self.assertIn("PROGRAM_ACCESS_REQUIRED", program_section)

    def test_analysis_job_claim_is_atomic_and_storage_is_private(self) -> None:
        source = text("app.py")
        job_section = source[source.index("def process_analysis_job") : source.index("@app.post", source.index("def process_analysis_job"))]
        self.assertIn('.eq("status", "queued")', job_section)
        self.assertIn("Another web process or worker claimed this job first", job_section)
        self.assertIn("ANALYSIS_STORAGE_BUCKET", job_section)
        self.assertIn("image_path", job_section)
        self.assertIn("image_base64", job_section)  # temporary compatibility fallback

    def test_screening_writes_include_uuid_and_idempotency(self) -> None:
        source = text("app.py")
        self.assertRegex(source, r"def build_screening_row\([\s\S]*?user_id")
        self.assertIn('"idempotency_key":', source)
        self.assertIn('"user_id": session.get("user_id")', source)

    def test_trainer_activation_backfills_all_linked_data(self) -> None:
        source = text("trainer_api.py")
        for table in (
            "sessions",
            "screenings",
            "analysis_jobs",
            "screening_history",
            "corrective_programs",
        ):
            self.assertIn(f'table("{table}")', source)
        self.assertIn('"client_user_id": user["id"]', source)
        self.assertIn('"status": "active"', source)

    def test_organization_entitlement_is_merged_not_destructive(self) -> None:
        source = text("screening_access.py")
        self.assertIn("def effective_entitlement", source)
        self.assertIn('table("organization_entitlements")', source)
        self.assertIn("organization_sources", source)
        self.assertIn("personal", source.lower())
        account = text("account_api.py")
        self.assertIn("effective_entitlement", account)
        self.assertIn("organization_sources", account)

    def test_workers_are_separate_and_claim_rows_safely(self) -> None:
        analysis_worker = text("worker.py")
        import_worker = text("operator_worker.py")
        self.assertIn("process_analysis_job", analysis_worker)
        self.assertIn("ANALYSIS_WORKER_POLL_SECONDS", analysis_worker)
        self.assertIn('status", "pending"', import_worker)
        self.assertIn('"status": "processing"', import_worker)
        self.assertIn("OPERATOR_IMPORT_BATCH_SIZE", import_worker)
        self.assertIn("screening_credit_cycles", import_worker)

    def test_migration_hardens_rls_and_removes_public_history(self) -> None:
        migration = text("migrations/20260720_v100_operator_security.sql").lower()
        for table in (
            "sessions",
            "screenings",
            "analysis_jobs",
            "screening_history",
            "corrective_programs",
            "program_session_progress",
            "trainer_clients",
            "trainer_profiles",
        ):
            self.assertIn(f"alter table public.{table} enable row level security", migration)
        self.assertIn('drop policy if exists "allow read screening history"', migration)
        self.assertIn("screening_history_select_own", migration)
        self.assertNotIn("using (true)", migration)

    def test_migration_adds_relational_integrity_and_indexes(self) -> None:
        migration = text("migrations/20260720_v100_operator_security.sql")
        for marker in (
            "screenings_session_id_fkey",
            "analysis_jobs_session_uuid_fkey",
            "screenings_idempotency_unique_idx",
            "analysis_jobs_active_idempotency_unique_idx",
            "sessions_owner_created_idx",
            "sessions_performer_created_idx",
        ):
            self.assertIn(marker, migration)

    def test_migration_creates_private_storage_and_operator_tables(self) -> None:
        migration = text("migrations/20260720_v100_operator_security.sql")
        for table in (
            "organizations",
            "organization_members",
            "organization_entitlements",
            "bulk_import_jobs",
            "bulk_import_rows",
            "trainer_client_notes",
            "system_health_events",
            "feature_flags",
            "operator_action_approvals",
        ):
            self.assertIn(f"public.{table}", migration)
        self.assertIn("'screening-private'", migration)
        self.assertRegex(migration, r"values \([\s\S]*?'screening-private'[\s\S]*?false")

    def test_bootstrap_requires_explicit_auth_uuid(self) -> None:
        bootstrap = text("migrations/BOOTSTRAP_FIRST_OPERATOR.sql")
        self.assertIn("REPLACE_WITH_YOUR_AUTH_USER_UUID", bootstrap)
        self.assertIn("super_admin", bootstrap)
        self.assertIn("on conflict (user_id) do update", bootstrap.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
