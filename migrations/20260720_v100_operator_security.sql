-- FlexiLab V100 — Operator Console, account isolation, security and scale
-- Apply before deploying the V100 backend.
-- Designed to be rerunnable. It does not delete legacy tables or client data.

begin;

create extension if not exists pgcrypto;

-- ---------------------------------------------------------------------------
-- 1. Operator roles and production administration tables
-- ---------------------------------------------------------------------------

do $$
begin
  if exists (
    select 1 from pg_constraint
    where conrelid = 'public.admin_roles'::regclass
      and conname = 'admin_roles_role_check'
  ) then
    alter table public.admin_roles drop constraint admin_roles_role_check;
  end if;
end $$;

alter table public.admin_roles
  add constraint admin_roles_role_check
  check (role = any (array[
    'support'::text,
    'admin'::text,
    'super_admin'::text,
    'operations'::text,
    'finance'::text,
    'clinical_reviewer'::text,
    'organization_manager'::text
  ])) not valid;
alter table public.admin_roles validate constraint admin_roles_role_check;

create table if not exists public.organizations (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  slug text not null,
  status text not null default 'active'
    check (status = any (array['active'::text, 'suspended'::text, 'archived'::text])),
  default_plan_code text references public.plans(code) on delete set null,
  access_ends_at timestamptz,
  created_by uuid references auth.users(id) on delete set null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create unique index if not exists organizations_slug_lower_unique_idx
  on public.organizations (lower(slug));
create index if not exists organizations_status_created_idx
  on public.organizations (status, created_at desc);

create table if not exists public.organization_members (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  user_id uuid references auth.users(id) on delete set null,
  invited_email text not null,
  full_name text,
  department text,
  cohort text,
  status text not null default 'invited'
    check (status = any (array['invited'::text, 'active'::text, 'suspended'::text, 'removed'::text])),
  invited_at timestamptz not null default now(),
  accepted_at timestamptz,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create unique index if not exists organization_members_org_email_unique_idx
  on public.organization_members (organization_id, lower(invited_email));
create index if not exists organization_members_user_idx
  on public.organization_members (user_id, status);
create index if not exists organization_members_org_status_idx
  on public.organization_members (organization_id, status, created_at desc);

create table if not exists public.organization_entitlements (
  organization_id uuid not null references public.organizations(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  plan_code text not null references public.plans(code),
  status text not null default 'active'
    check (status = any (array['active'::text, 'grace'::text, 'expired'::text, 'suspended'::text])),
  program_access boolean not null default false,
  workout_access boolean not null default false,
  history_access boolean not null default true,
  report_access boolean not null default true,
  valid_from timestamptz not null default now(),
  valid_until timestamptz,
  updated_at timestamptz not null default now(),
  primary key (organization_id, user_id)
);
create index if not exists organization_entitlements_user_status_idx
  on public.organization_entitlements (user_id, status, valid_until);

create table if not exists public.bulk_import_jobs (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  uploaded_by uuid references auth.users(id) on delete set null,
  filename text not null,
  status text not null default 'queued'
    check (status = any (array[
      'queued'::text, 'processing'::text, 'completed'::text,
      'completed_with_errors'::text, 'failed'::text, 'cancelled'::text
    ])),
  total_rows integer not null default 0 check (total_rows >= 0),
  processed_rows integer not null default 0 check (processed_rows >= 0),
  success_rows integer not null default 0 check (success_rows >= 0),
  failed_rows integer not null default 0 check (failed_rows >= 0),
  duplicate_rows integer not null default 0 check (duplicate_rows >= 0),
  errors_json jsonb not null default '[]'::jsonb,
  started_at timestamptz,
  completed_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists bulk_import_jobs_queue_idx
  on public.bulk_import_jobs (status, created_at)
  where status in ('queued', 'processing');
create index if not exists bulk_import_jobs_org_idx
  on public.bulk_import_jobs (organization_id, created_at desc);

create table if not exists public.bulk_import_rows (
  id uuid primary key default gen_random_uuid(),
  job_id uuid not null references public.bulk_import_jobs(id) on delete cascade,
  organization_id uuid not null references public.organizations(id) on delete cascade,
  row_number integer not null,
  email text not null,
  full_name text,
  department text,
  cohort text,
  status text not null default 'pending'
    check (status = any (array[
      'pending'::text, 'processing'::text, 'success'::text,
      'failed'::text, 'invalid'::text, 'duplicate'::text
    ])),
  error_message text,
  user_id uuid references auth.users(id) on delete set null,
  metadata jsonb not null default '{}'::jsonb,
  processed_at timestamptz,
  created_at timestamptz not null default now(),
  unique (job_id, row_number)
);
create index if not exists bulk_import_rows_queue_idx
  on public.bulk_import_rows (job_id, status, row_number);
create index if not exists bulk_import_rows_email_idx
  on public.bulk_import_rows (lower(email));

create table if not exists public.trainer_client_notes (
  id uuid primary key default gen_random_uuid(),
  trainer_id uuid not null references auth.users(id) on delete cascade,
  trainer_client_link_id uuid not null references public.trainer_clients(id) on delete cascade,
  session_id uuid references public.sessions(id) on delete set null,
  note text not null check (char_length(note) between 1 and 5000),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists trainer_client_notes_link_created_idx
  on public.trainer_client_notes (trainer_client_link_id, created_at desc);
create index if not exists trainer_client_notes_session_idx
  on public.trainer_client_notes (session_id, created_at desc);

create table if not exists public.system_health_events (
  id bigint generated always as identity primary key,
  component text not null,
  status text not null check (status = any (array['healthy'::text, 'degraded'::text, 'down'::text, 'recovered'::text])),
  details jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);
create index if not exists system_health_events_component_created_idx
  on public.system_health_events (component, created_at desc);

create table if not exists public.feature_flags (
  key text primary key,
  enabled boolean not null default false,
  description text,
  configuration jsonb not null default '{}'::jsonb,
  updated_by uuid references auth.users(id) on delete set null,
  updated_at timestamptz not null default now()
);

create table if not exists public.operator_action_approvals (
  id uuid primary key default gen_random_uuid(),
  requested_by uuid not null references auth.users(id) on delete cascade,
  approved_by uuid references auth.users(id) on delete set null,
  action text not null,
  payload jsonb not null default '{}'::jsonb,
  status text not null default 'pending'
    check (status = any (array['pending'::text, 'approved'::text, 'rejected'::text, 'executed'::text, 'expired'::text])),
  requested_at timestamptz not null default now(),
  decided_at timestamptz,
  executed_at timestamptz,
  expires_at timestamptz
);
create index if not exists operator_action_approvals_status_idx
  on public.operator_action_approvals (status, requested_at desc);

-- ---------------------------------------------------------------------------
-- 2. UUID ownership and idempotency upgrades
-- ---------------------------------------------------------------------------
alter table public.corrective_programs
  add column if not exists user_id uuid references auth.users(id) on delete set null;
alter table public.program_session_progress
  add column if not exists user_id uuid references auth.users(id) on delete set null;
alter table public.screening_history
  add column if not exists user_id uuid references auth.users(id) on delete set null;
alter table public.analysis_jobs
  add column if not exists session_uuid uuid,
  add column if not exists idempotency_key text,
  add column if not exists image_path text,
  add column if not exists image_expires_at timestamptz;
alter table public.screenings
  add column if not exists idempotency_key text;

update public.screenings sc
set user_id = s.user_id,
    user_email = s.user_email
from public.sessions s
where sc.session_id = s.id
  and s.user_id is not null
  and (sc.user_id is null or lower(sc.user_email) <> lower(s.user_email));

update public.corrective_programs cp
set user_id = s.user_id
from public.sessions s
where cp.user_id is null
  and cp.screening_session_id = s.id
  and s.user_id is not null;

update public.program_session_progress psp
set user_id = cp.user_id
from public.corrective_programs cp
where psp.user_id is null
  and psp.program_id = cp.id
  and cp.user_id is not null;

update public.screening_history sh
set user_id = s.user_id
from public.sessions s
where sh.user_id is null
  and sh.session_id = s.id
  and s.user_id is not null;

update public.analysis_jobs
set session_uuid = session_id::uuid
where session_uuid is null
  and session_id ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$';

update public.analysis_jobs aj
set user_id = s.user_id,
    user_email = s.user_email
from public.sessions s
where aj.session_uuid = s.id
  and s.user_id is not null
  and (aj.user_id is null or lower(coalesce(aj.user_email, '')) <> lower(s.user_email));

-- Completed images are temporary processing artifacts and should not remain in
-- PostgreSQL. Failed images are retained briefly to allow an Operator retry.
update public.analysis_jobs
set image_base64 = null,
    image_expires_at = null
where status = 'completed'
  and image_base64 is not null;

update public.analysis_jobs
set image_base64 = null,
    image_expires_at = null
where status = 'failed'
  and created_at < now() - interval '7 days'
  and image_base64 is not null;

-- New rows are protected; historical duplicate rows remain readable and untouched.
create unique index if not exists screenings_idempotency_unique_idx
  on public.screenings (idempotency_key)
  where idempotency_key is not null;
create unique index if not exists analysis_jobs_active_idempotency_unique_idx
  on public.analysis_jobs (idempotency_key)
  where idempotency_key is not null and status in ('queued', 'processing', 'completed');
create index if not exists analysis_jobs_session_status_idx
  on public.analysis_jobs (session_uuid, status, created_at desc);
create index if not exists analysis_jobs_queue_idx
  on public.analysis_jobs (status, created_at)
  where status in ('queued', 'processing');
create index if not exists corrective_programs_user_generated_uuid_idx
  on public.corrective_programs (user_id, generated_at desc);
create index if not exists program_session_progress_user_idx
  on public.program_session_progress (user_id, updated_at desc);
create index if not exists screening_history_user_created_idx
  on public.screening_history (user_id, created_at desc);
create index if not exists sessions_owner_created_idx
  on public.sessions (user_id, status, created_at desc);
create index if not exists sessions_performer_created_idx
  on public.sessions (performed_by_user_id, status, created_at desc);
create index if not exists screenings_session_test_created_idx
  on public.screenings (session_id, test_type, created_at desc);

-- Add referential integrity for new rows without invalidating historical orphan checks.
do $$
begin
  if not exists (select 1 from pg_constraint where conname = 'screenings_session_id_fkey') then
    alter table public.screenings
      add constraint screenings_session_id_fkey
      foreign key (session_id) references public.sessions(id) on delete cascade not valid;
  end if;
  if not exists (select 1 from pg_constraint where conname = 'analysis_jobs_session_uuid_fkey') then
    alter table public.analysis_jobs
      add constraint analysis_jobs_session_uuid_fkey
      foreign key (session_uuid) references public.sessions(id) on delete cascade not valid;
  end if;
end $$;

-- ---------------------------------------------------------------------------
-- 3. Private temporary image bucket
-- ---------------------------------------------------------------------------
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
  'screening-private',
  'screening-private',
  false,
  12582912,
  array['image/jpeg', 'image/png', 'image/webp']
)
on conflict (id) do update
set public = false,
    file_size_limit = excluded.file_size_limit,
    allowed_mime_types = excluded.allowed_mime_types;

-- No anon/authenticated Storage policy is created. The backend service role owns
-- upload/download/delete and removes images after analysis.

-- ---------------------------------------------------------------------------
-- 4. RLS hardening for movement, Trainer and Operator data
-- ---------------------------------------------------------------------------
alter table public.sessions enable row level security;
alter table public.screenings enable row level security;
alter table public.analysis_jobs enable row level security;
alter table public.screening_history enable row level security;
alter table public.corrective_programs enable row level security;
alter table public.program_session_progress enable row level security;
alter table public.trainer_clients enable row level security;
alter table public.trainer_profiles enable row level security;
alter table public.trainer_client_notes enable row level security;
alter table public.organizations enable row level security;
alter table public.organization_members enable row level security;
alter table public.organization_entitlements enable row level security;
alter table public.bulk_import_jobs enable row level security;
alter table public.bulk_import_rows enable row level security;
alter table public.system_health_events enable row level security;
alter table public.feature_flags enable row level security;
alter table public.operator_action_approvals enable row level security;

-- Remove the dangerous legacy public policy.
drop policy if exists "Allow read screening history" on public.screening_history;

-- Idempotent policy replacement.
drop policy if exists sessions_select_owner_or_performer on public.sessions;
create policy sessions_select_owner_or_performer
on public.sessions for select to authenticated
using (
  user_id = auth.uid()
  or trainer_id = auth.uid()
  or performed_by_user_id = auth.uid()
);

drop policy if exists screenings_select_via_session on public.screenings;
create policy screenings_select_via_session
on public.screenings for select to authenticated
using (
  exists (
    select 1 from public.sessions s
    where s.id = screenings.session_id
      and (s.user_id = auth.uid() or s.trainer_id = auth.uid() or s.performed_by_user_id = auth.uid())
  )
);

drop policy if exists analysis_jobs_select_via_session on public.analysis_jobs;
create policy analysis_jobs_select_via_session
on public.analysis_jobs for select to authenticated
using (
  user_id = auth.uid()
  or exists (
    select 1 from public.sessions s
    where s.id = analysis_jobs.session_uuid
      and (s.user_id = auth.uid() or s.trainer_id = auth.uid() or s.performed_by_user_id = auth.uid())
  )
);

drop policy if exists screening_history_select_own on public.screening_history;
create policy screening_history_select_own
on public.screening_history for select to authenticated
using (user_id = auth.uid());

drop policy if exists corrective_programs_select_own on public.corrective_programs;
create policy corrective_programs_select_own
on public.corrective_programs for select to authenticated
using (user_id = auth.uid());

drop policy if exists program_progress_select_own on public.program_session_progress;
create policy program_progress_select_own
on public.program_session_progress for select to authenticated
using (user_id = auth.uid());

drop policy if exists trainer_clients_select_participant on public.trainer_clients;
create policy trainer_clients_select_participant
on public.trainer_clients for select to authenticated
using (trainer_id = auth.uid() or client_user_id = auth.uid());

drop policy if exists trainer_profiles_select_own on public.trainer_profiles;
create policy trainer_profiles_select_own
on public.trainer_profiles for select to authenticated
using (user_id = auth.uid());

drop policy if exists trainer_notes_select_own on public.trainer_client_notes;
create policy trainer_notes_select_own
on public.trainer_client_notes for select to authenticated
using (trainer_id = auth.uid());

drop policy if exists organization_members_select_own on public.organization_members;
create policy organization_members_select_own
on public.organization_members for select to authenticated
using (user_id = auth.uid());

drop policy if exists organization_entitlements_select_own on public.organization_entitlements;
create policy organization_entitlements_select_own
on public.organization_entitlements for select to authenticated
using (user_id = auth.uid());

-- Operator tables intentionally have no client-side policies. The Operator API uses
-- the service role and performs role checks plus audit logging on every action.

-- ---------------------------------------------------------------------------
-- 5. Updated-at triggers
-- ---------------------------------------------------------------------------

do $$
declare
  table_name text;
begin
  foreach table_name in array array[
    'organizations', 'organization_members', 'organization_entitlements',
    'bulk_import_jobs', 'trainer_client_notes'
  ] loop
    execute format('drop trigger if exists %I_set_updated_at on public.%I', table_name, table_name);
    execute format(
      'create trigger %I_set_updated_at before update on public.%I for each row execute function public.set_updated_at()',
      table_name,
      table_name
    );
  end loop;
end $$;

commit;
