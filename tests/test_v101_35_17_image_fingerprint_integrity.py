from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
SOURCE = APP_PATH.read_text(encoding="utf-8")


def _submit_section() -> str:
    start = SOURCE.index('@app.post("/submit_analysis")')
    end = SOURCE.index('@app.get("/job_status/{job_id}")', start)
    return SOURCE[start:end]


def test_upload_is_read_and_hashed_before_existing_job_reuse_lookup():
    section = _submit_section()
    read_pos = section.index("img_bytes = await image.read()")
    hash_pos = section.index("hashlib.sha256(img_bytes).hexdigest()")
    lookup_pos = section.index('supabase.table("analysis_jobs")')
    assert read_pos < hash_pos < lookup_pos


def test_reuse_requires_exact_image_fingerprint():
    section = _submit_section()
    assert 'existing_fingerprint == image_fingerprint' in section
    assert '"reuse_reason": "exact_image_match"' in section
    assert '"reuse_reason": "fresh_analysis"' in section
    assert '"image_fingerprint": image_fingerprint' in section


def test_job_idempotency_key_contains_image_fingerprint():
    section = _submit_section()
    assert 'job_idempotency_key = f"{session_id}:{test_type}:{image_fingerprint}"' in section
    assert '"idempotency_key": job_idempotency_key' in section


def test_screening_idempotency_key_uses_capture_fingerprint():
    assert 'intake_data.get("_flexilab_capture_fingerprint")' in SOURCE
    assert 'f"{session_id}:{test_type}:{capture_fingerprint}"' in SOURCE


def test_legacy_jobs_without_fingerprint_are_not_silently_reused():
    section = _submit_section()
    assert 'if existing_fingerprint == image_fingerprint:' in section
    assert 'matching_jobs.append(existing)' in section
    # The former broad reuse loop over every existing session/test job is gone.
    assert 'for existing in existing_jobs.data or []:\n        existing_status' not in section
