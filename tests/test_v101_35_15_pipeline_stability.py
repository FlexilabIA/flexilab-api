from pathlib import Path


APP_SOURCE = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")


def test_aslr_vision_qa_is_disabled_in_production_path():
    assert 'if vision_qa_requested and not is_aslr:' in APP_SOURCE
    assert 'elif vision_qa_requested and is_aslr:' in APP_SOURCE
    assert 'result["metrics"]["vision_qa_disabled_for_performance"] = True' in APP_SOURCE


def test_job_status_does_not_requeue_processing_jobs_after_120_seconds():
    endpoint = APP_SOURCE.split('@app.get("/job_status/{job_id}")', 1)[1].split('@app.get("/report")', 1)[0]
    assert 'current_status == "processing"' not in endpoint
    assert 'total_seconds() >= 120' not in endpoint
    assert 'current_status == "queued" and ANALYSIS_INLINE_ENABLED' in endpoint


def test_transient_supabase_read_failures_are_retried_without_rerunning_inference():
    assert 'def _is_transient_upstream_error(exc):' in APP_SOURCE
    assert 'def _execute_with_transient_retry(operation, *, label, attempts=3):' in APP_SOURCE
    assert 'httpx.ReadError' in APP_SOURCE
    assert '"resource temporarily unavailable" in text' in APP_SOURCE
    assert 'label="analysis_job_fetch"' in APP_SOURCE
    assert 'label="analysis_job_claim"' in APP_SOURCE
    assert 'label="screening_insert"' in APP_SOURCE
    assert 'label="complete_analysis_job"' in APP_SOURCE


def test_stabilization_does_not_change_aslr_model_or_inference_contract():
    assert '"yolo11m-pose.pt"' in APP_SOURCE
    assert 'pose_model_inference_count": 1' in APP_SOURCE
    assert '"aslr_inference_imgsz": 960' in APP_SOURCE
    assert 'V101.35.14-aslr-one-call-no-tracked-images' in APP_SOURCE
