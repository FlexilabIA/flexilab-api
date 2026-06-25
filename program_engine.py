"""
FlexiLab Program Engine V1
Generate a 4-week corrective exercise program from FlexiLab screening results
using exercise_library.json. Independent from app.py.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

LIBRARY_PATH = Path(__file__).with_name("exercise_library.json")

SEVERITY_POINTS = {"red": 3, "yellow": 2, "green": 0, "unknown": 0, None: 0}
PHASE_ORDER = ["1_release_recovery", "2_mobility", "3_motor_control", "4_stability", "5_strength", "6_integration"]
SYSTEM_LABELS = {
    "CERV": {"fr": "Contrôle cervical", "en": "Cervical control"},
    "THOR": {"fr": "Mobilité thoracique", "en": "Thoracic mobility"},
    "SHLD": {"fr": "Mobilité d'épaule", "en": "Shoulder mobility"},
    "SCAP": {"fr": "Stabilité scapulaire", "en": "Scapular stability"},
    "CORE": {"fr": "Stabilité du tronc", "en": "Core stability"},
    "PELV": {"fr": "Contrôle lombo-pelvien", "en": "Lumbopelvic control"},
    "HIP": {"fr": "Mobilité de hanche", "en": "Hip mobility"},
    "POST": {"fr": "Chaîne postérieure", "en": "Posterior chain"},
    "ANKL": {"fr": "Mobilité de cheville", "en": "Ankle mobility"},
    "KNEE": {"fr": "Contrôle du genou", "en": "Knee control"},
    "FUNC": {"fr": "Intégration fonctionnelle", "en": "Functional integration"},
    "REC": {"fr": "Reset / récupération", "en": "Reset / recovery"},
}


def load_exercise_library(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or LIBRARY_PATH
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_lang(lang: str = "fr") -> str:
    return "en" if str(lang).lower().startswith("en") else "fr"


def label_system(system_code: str, lang: str = "fr") -> str:
    lang = normalize_lang(lang)
    return SYSTEM_LABELS.get(system_code, {}).get(lang, system_code)


def extract_rating_map(report: Dict[str, Any]) -> Dict[str, str]:
    rating_map: Dict[str, str] = {}
    for section in report.get("sections", []) or []:
        for item in section.get("items", []) or []:
            item_id = item.get("id")
            rating = item.get("rating")
            if item_id:
                rating_map[item_id] = rating
        asym = section.get("asymmetry")
        if isinstance(asym, dict) and asym.get("rating"):
            section_id = section.get("id", "unknown")
            rating_map[f"{section_id}_asymmetry"] = asym.get("rating")
    for key, value in report.items():
        if key.endswith("_rating") and isinstance(value, str):
            rating_map[key.replace("_rating", "")] = value
    return rating_map


def compute_movement_system_scores(rating_map: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    scores = {k: 0.0 for k in ["CERV", "THOR", "SHLD", "SCAP", "CORE", "PELV", "HIP", "POST", "ANKL", "KNEE", "FUNC", "REC"]}
    reasons = {k: [] for k in scores}

    def add(system: str, points: float, reason: str) -> None:
        if points > 0:
            scores[system] += points
            reasons[system].append(reason)

    def pts(key: str) -> int:
        return SEVERITY_POINTS.get(rating_map.get(key), 0)

    p = pts("neck_angle")
    add("CERV", p * 12, "Cervical alignment finding")
    add("THOR", p * 4, "Cervical alignment may be influenced by thoracic posture")

    p = pts("thoracic_angle")
    add("THOR", p * 14, "Thoracic alignment finding")
    add("SHLD", p * 5, "Thoracic posture can influence overhead mobility")

    p = pts("pelvic_proxy_angle")
    add("PELV", p * 10, "Trunk-pelvis alignment finding")
    add("CORE", p * 6, "Trunk-pelvis alignment can reflect core control")

    shoulder_p = max(pts("shoulder_right_flexion"), pts("shoulder_left_flexion"))
    add("SHLD", shoulder_p * 14, "Shoulder flexion limitation")
    add("THOR", shoulder_p * 8, "Overhead limitation may involve thoracic mobility")
    add("SCAP", shoulder_p * 8, "Overhead limitation may involve scapular control")
    add("CERV", shoulder_p * 2, "Shoulder limitation can increase neck compensation")

    p = pts("shoulders_asymmetry")
    add("SHLD", p * 8, "Right-left shoulder asymmetry")
    add("SCAP", p * 5, "Right-left shoulder asymmetry may involve scapular control")

    p = pts("squat_trunk_lean")
    add("CORE", p * 12, "Trunk lean during squat")
    add("HIP", p * 8, "Trunk lean may reflect hip mobility limitation")
    add("ANKL", p * 6, "Trunk lean may reflect ankle mobility limitation")
    add("THOR", p * 4, "Trunk lean may include thoracic contribution")
    add("FUNC", p * 5, "Squat pattern integration finding")

    p = pts("squat_knee_angle")
    add("KNEE", p * 8, "Squat depth / knee angle finding")
    add("HIP", p * 8, "Squat depth may reflect hip mobility")
    add("ANKL", p * 8, "Squat depth may reflect ankle dorsiflexion")
    add("FUNC", p * 5, "Squat pattern integration finding")

    aslr_p = max(pts("aslr_right_angle"), pts("aslr_left_angle"))
    add("POST", aslr_p * 14, "Active Straight Leg Raise limitation")
    add("HIP", aslr_p * 8, "ASLR limitation may reflect hip mobility")
    add("PELV", aslr_p * 8, "ASLR limitation may reflect pelvic control")
    add("CORE", aslr_p * 5, "ASLR requires trunk-pelvis control")

    p = pts("aslr_asymmetry")
    add("POST", p * 8, "Right-left ASLR asymmetry")
    add("HIP", p * 5, "Right-left ASLR asymmetry may involve hip mobility")
    add("PELV", p * 5, "Right-left ASLR asymmetry may involve pelvic control")

    max_score = max(scores.values()) if scores else 0
    output = {}
    for system, raw in scores.items():
        if raw <= 0:
            continue
        priority = round((raw / max_score) * 100, 1) if max_score > 0 else 0.0
        output[system] = {"system": system, "raw_score": round(raw, 2), "priority_score": priority, "reasons": reasons[system]}
    return dict(sorted(output.items(), key=lambda kv: kv[1]["priority_score"], reverse=True))


def normalize_pain_clearance(pain_clearance: Optional[Dict[str, str]]) -> Dict[str, str]:
    if not pain_clearance:
        return {}
    normalized = {}
    for key, value in pain_clearance.items():
        v = str(value).lower().strip()
        if v in ["none", "no", "no pain", "no_pain"]:
            normalized[key] = "no_pain"
        elif v in ["discomfort", "tension", "mild", "mild_discomfort"]:
            normalized[key] = "discomfort"
        elif v in ["pain", "yes", "true"]:
            normalized[key] = "pain"
        else:
            normalized[key] = v
    return normalized


def system_has_pain(system: str, pain: Dict[str, str]) -> bool:
    painful_tests = {k for k, v in pain.items() if v == "pain"}
    system_to_tests = {
        "CERV": {"posture_side"},
        "THOR": {"posture_side", "shoulder_right", "shoulder_left", "squat"},
        "SHLD": {"shoulder_right", "shoulder_left"},
        "SCAP": {"shoulder_right", "shoulder_left"},
        "CORE": {"squat", "aslr_right", "aslr_left"},
        "PELV": {"squat", "aslr_right", "aslr_left"},
        "HIP": {"squat", "aslr_right", "aslr_left"},
        "POST": {"aslr_right", "aslr_left"},
        "ANKL": {"squat"},
        "KNEE": {"squat"},
        "FUNC": {"squat"},
    }
    return bool(system_to_tests.get(system, set()) & painful_tests)


def allowed_phases_for_week(week: int, painful: bool = False) -> List[str]:
    if painful:
        return ["1_release_recovery", "2_mobility"]
    if week == 1:
        return ["1_release_recovery", "2_mobility"]
    if week == 2:
        return ["2_mobility", "3_motor_control"]
    if week == 3:
        return ["3_motor_control", "4_stability"]
    return ["4_stability", "5_strength", "6_integration"]


def exercise_matches_system(exercise: Dict[str, Any], system: str) -> bool:
    return exercise.get("primary_system") == system or system in (exercise.get("secondary_systems") or [])


def select_exercises_for_system(exercises: List[Dict[str, Any]], system: str, week: int, painful: bool = False, max_items: int = 2) -> List[Dict[str, Any]]:
    allowed_phases = allowed_phases_for_week(week, painful=painful)
    candidates = []
    for ex in exercises:
        if not exercise_matches_system(ex, system):
            continue
        if ex.get("phase") not in allowed_phases:
            continue
        if int(ex.get("week_start", 1)) > week or int(ex.get("week_end", 4)) < week:
            continue
        if painful and ex.get("phase") not in ["1_release_recovery", "2_mobility"]:
            continue
        phase_rank = PHASE_ORDER.index(ex.get("phase")) if ex.get("phase") in PHASE_ORDER else 99
        difficulty = int(ex.get("difficulty", 3))
        candidates.append((phase_rank, difficulty, ex.get("id", ""), ex))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    selected, seen_ids = [], set()
    for _, _, _, ex in candidates:
        if ex["id"] in seen_ids:
            continue
        selected.append(ex)
        seen_ids.add(ex["id"])
        if len(selected) >= max_items:
            break
    return selected


def format_exercise_for_program(ex: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": ex.get("id"), "name": ex.get("name"), "primary_system": ex.get("primary_system"),
        "secondary_systems": ex.get("secondary_systems", []), "phase": ex.get("phase"),
        "difficulty": ex.get("difficulty"), "equipment": ex.get("equipment", []), "sets": ex.get("sets"),
        "reps": ex.get("reps"), "hold": ex.get("hold"), "tempo": ex.get("tempo"),
        "frequency_per_week": ex.get("frequency_per_week"), "duration_minutes": ex.get("duration_minutes"),
        "instructions": ex.get("instructions"), "common_errors": ex.get("common_errors"),
        "video_url": ex.get("video_url"), "thumbnail_url": ex.get("thumbnail_url"), "tags": ex.get("tags", []),
    }


def generate_program_from_report(report: Dict[str, Any], exercise_library: Optional[Dict[str, Any]] = None, lang: str = "fr", pain_clearance: Optional[Dict[str, str]] = None, max_priority_systems: int = 4) -> Dict[str, Any]:
    lang = normalize_lang(lang)
    library = exercise_library or load_exercise_library()
    exercises = library.get("exercises", [])
    rating_map = extract_rating_map(report)
    movement_scores = compute_movement_system_scores(rating_map)
    pain = normalize_pain_clearance(pain_clearance)

    if not movement_scores:
        movement_scores = {"FUNC": {"system": "FUNC", "raw_score": 1.0, "priority_score": 100.0, "reasons": ["No major deficit detected; maintenance program selected."]}}

    top_systems = list(movement_scores.keys())[:max_priority_systems]
    weeks = []
    themes_fr = ["Reset + mobilité", "Mobilité + contrôle moteur", "Contrôle + stabilité", "Stabilité + intégration fonctionnelle"]
    themes_en = ["Reset + mobility", "Mobility + motor control", "Control + stability", "Stability + functional integration"]
    for week in range(1, 5):
        week_items = []
        for system in top_systems:
            painful = system_has_pain(system, pain)
            selected = select_exercises_for_system(exercises, system, week, painful=painful, max_items=2 if week <= 2 else 1)
            if not selected:
                continue
            week_items.append({
                "system": system,
                "system_label": label_system(system, lang),
                "priority_score": movement_scores[system]["priority_score"],
                "pain_limited": painful,
                "exercises": [format_exercise_for_program(x) for x in selected],
            })
        weeks.append({"week": week, "theme_fr": themes_fr[week-1], "theme_en": themes_en[week-1], "theme": themes_en[week-1] if lang == "en" else themes_fr[week-1], "items": week_items})

    safety_notes_fr = [
        "Ne pas faire un exercice qui provoque une douleur.",
        "Garder une respiration calme et un mouvement contrôlé.",
        "Progresser uniquement si l’alignement est maîtrisé sans compensation.",
        "En cas de douleur, engourdissement, faiblesse inexpliquée ou symptôme inhabituel : arrêter et demander un avis médical."
    ]
    safety_notes_en = [
        "Do not perform any exercise that causes pain.",
        "Keep breathing calm and movement controlled.",
        "Progress only when alignment is controlled without compensation.",
        "In case of pain, numbness, unexplained weakness or unusual symptoms: stop and seek medical advice."
    ]

    return {
        "engine_version": "FlexiLab Program Engine V1",
        "library_version": library.get("version"),
        "language": lang,
        "movement_system_scores": movement_scores,
        "top_priority_systems": [{"system": s, "system_label": label_system(s, lang), "priority_score": movement_scores[s]["priority_score"], "reasons": movement_scores[s]["reasons"], "pain_limited": system_has_pain(s, pain)} for s in top_systems],
        "pain_clearance": pain,
        "weeks": weeks,
        "safety_notes_fr": safety_notes_fr,
        "safety_notes_en": safety_notes_en,
        "safety_notes": safety_notes_en if lang == "en" else safety_notes_fr,
        "progression_rule_fr": "Semaine 1 mobilité/reset, semaine 2 contrôle, semaine 3 stabilité, semaine 4 intégration. Re-tester après 4 semaines.",
        "progression_rule_en": "Week 1 mobility/reset, week 2 control, week 3 stability, week 4 integration. Re-test after 4 weeks.",
        "progression_rule": "Week 1 mobility/reset, week 2 control, week 3 stability, week 4 integration. Re-test after 4 weeks." if lang == "en" else "Semaine 1 mobilité/reset, semaine 2 contrôle, semaine 3 stabilité, semaine 4 intégration. Re-tester après 4 semaines."
    }


def generate_program_from_ratings(ratings: Dict[str, str], lang: str = "fr", pain_clearance: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    fake_report = {"sections": [{"id": "manual", "items": [{"id": k, "rating": v} for k, v in ratings.items()]}]}
    return generate_program_from_report(report=fake_report, lang=lang, pain_clearance=pain_clearance)
