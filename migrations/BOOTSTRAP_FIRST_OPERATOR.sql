-- Run once after replacing the placeholder with your real Supabase Auth UUID.
-- Find it in Supabase Dashboard -> Authentication -> Users.

insert into public.admin_roles (user_id, role, granted_by, granted_at, revoked_at)
values (
  'REPLACE_WITH_YOUR_AUTH_USER_UUID'::uuid,
  'super_admin',
  'REPLACE_WITH_YOUR_AUTH_USER_UUID'::uuid,
  now(),
  null
)
on conflict (user_id) do update
set role = excluded.role,
    granted_by = excluded.granted_by,
    granted_at = now(),
    revoked_at = null;
