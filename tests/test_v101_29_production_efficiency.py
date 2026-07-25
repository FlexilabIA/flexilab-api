from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_aslr_uses_rotated_fast_path_and_lazy_fallback():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert '"patch_version": "V101.29-production-efficiency"' in source
    assert 'first_pose_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) if is_aslr else img' in source
    assert 'fallback_used = primary_quality is None or primary_quality < 0.72' in source
    assert '"pose_pass_count": len(evaluated_passes)' in source


def test_trainer_bootstrap_is_compact_and_client_queries_are_bulk():
    source = (ROOT / "trainer_api.py").read_text(encoding="utf-8")
    assert 'latest_client_sessions_bulk' in source
    assert '.in_("trainer_client_link_id", link_ids)' in source
    assert '.limit(20)' in source and 'recent_clients[:5]' in source
    assert '"bootstrap_version": "trainer-bootstrap-v2-recent-five"' in source
    assert '"token_history": {"cycles": [], "referral_rewards": []}' in source


def test_report_copy_keeps_one_concise_wellness_notice():
    source = (ROOT / "engines" / "clinical_report_engine_v1.py").read_text(encoding="utf-8")
    assert "FlexiLab fournit un screening du mouvement destiné au bien-être" in source
    assert "It does not replace medical advice" in source
    assert "Dorsiflexion de cheville non mesurée directement" not in source
