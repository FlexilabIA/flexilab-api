from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")
ASLR = (ROOT / "aslr_engine.py").read_text(encoding="utf-8")


def test_patch_version_and_warning_contract_present():
    assert "V101.35.23-screening-soft-warnings-all-landmarks" in APP
    assert "screening_validation" in APP
    assert "requires_user_acknowledgement" in APP
    assert "measurable_with_warning" in APP


def test_all_screening_warning_codes_present():
    for code in (
        "posture_landmark_moderate_confidence",
        "shoulder_elbow_flexion",
        "squat_trunk_inclination",
        "aslr_raised_knee_flexion",
    ):
        assert code in APP


def test_aslr_moderate_knee_flexion_is_warning_not_hard_rejection():
    assert 'if float(raised["knee_extension_angle"]) < 105.0:' in ASLR
    assert "raised_knee_below_preferred_extension" in ASLR
