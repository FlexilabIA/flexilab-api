"""
FlexiLab Program Engine V2

Upgrades vs V1
--------------
1. Avoids duplicate exercises inside the same week.
2. Builds balanced weekly sessions instead of separate repeated blocks.
3. Caps total exercises per week.
4. Adds session order, purpose, estimated duration, and prescription fields.
5. Adds pain-clearance logic:
   - pain => release/recovery + gentle mobility only
   - discomfort => no high difficulty / no strength loading
6. Uses the exercise_library.json file as the knowledge base.

This module stays independent from app.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


LIBRARY_PATH = Path(__file__).with_name("exercise_library.json")

SEVERITY_POINTS = {
    "red": 3,
    "yellow": 2,
    "green": 0,
    "unknown": 0,
    None: 0,
}

PHASE_ORDER = [
    "1_release_recovery",
    "2_mobility",
    "3_motor_control",
    "4_stability",
    "5_strength",
    "6_integration",
]

PHASE_LABELS = {
    "1_release_recovery": {"fr": "Reset / récupération", "en": "Reset / recovery"},
    "2_mobility": {"fr": "Mobilité", "en": "Mobility"},
    "3_motor_control": {"fr": "Contrôle moteur", "en": "Motor control"},
    "4_stability": {"fr": "Stabilité", "en": "Stability"},
    "5_strength": {"fr": "Renforcement", "en": "Strength"},
    "6_integration": {"fr": "Intégration", "en": "Integration"},
}

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

# How many total exercises per week.
WEEK_CAPS = {
    1: 6,
    2: 6,
    3: 6,
    4: 6,
}

# Desired session structure by week.
WEEK_PHASE_PLAN = {
    1: ["1_release_recovery", "2_mobility", "2_mobility", "3_motor_control", "3_motor_control", "6_integration"],
    2: ["2_mobility", "2_mobility", "3_motor_control", "3_motor_control", "4_stability", "6_integration"],
    3: ["2_mobility", "3_motor_control", "4_stability", "4_stability", "5_strength", "6_integration"],
    4: ["2_mobility", "4_stability", "5_strength", "5_strength", "6_integration", "6_integration"],
}

PAIN_ALLOWED_PHASES = ["1_release_recovery", "2_mobility"]
DISCOMFORT_BLOCKED_PHASES = ["5_strength"]


def load_exercise_library(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or LIBRARY_PATH
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_lang(lang: str = "fr") -> str:
    return "en" if str(lang).lower().startswith("en") else "fr"


def label_system(system_code: str, lang: str = "fr") -> str:
    lang = normalize_lang(lang)
    return SYSTEM_LABELS.get(system_code, {}).get(lang, system_code)


def label_phase(phase: str, lang: str = "fr") -> str:
    lang = normalize_lang(lang)
    return PHASE_LABELS.get(phase, {}).get(lang, phase)


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
    scores: Dict[str, float] = {
        "CERV": 0, "THOR": 0, "SHLD": 0, "SCAP": 0, "CORE": 0, "PELV": 0,
        "HIP": 0, "POST": 0, "ANKL": 0, "KNEE": 0, "FUNC": 0, "REC": 0,
    }
    reasons: Dict[str, List[str]] = {k: [] for k in scores}

    def add(system: str, points: float, reason: str) -> None:
        if points <= 0:
            return
        scores[system] += points
        if reason not in reasons[system]:
            reasons[system].append(reason)

    def pts(key: str) -> int:
        return SEVERITY_POINTS.get(rating_map.get(key), 0)

    # Posture
    p = pts("neck_angle")
    add("CERV", p * 12, "Cervical alignment finding")
    add("THOR", p * 4, "Cervical alignment may be influenced by thoracic posture")

    p = pts("thoracic_angle")
    add("THOR", p * 14, "Thoracic alignment finding")
    add("SHLD", p * 5, "Thoracic posture can influence overhead mobility")

    p = pts("pelvic_proxy_angle")
    add("PELV", p * 10, "Trunk-pelvis alignment finding")
    add("CORE", p * 6, "Trunk-pelvis alignment can reflect core control")

    # Shoulders
    p = max(pts("shoulder_right_flexion"), pts("shoulder_left_flexion"))
    add("SHLD", p * 14, "Shoulder flexion limitation")
    add("THOR", p * 8, "Overhead limitation may involve thoracic mobility")
    add("SCAP", p * 8, "Overhead limitation may involve scapular control")
    add("CERV", p * 2, "Shoulder limitation can increase neck compensation")

    p = pts("shoulders_asymmetry")
    add("SHLD", p * 8, "Right-left shoulder asymmetry")
    add("SCAP", p * 5, "Right-left shoulder asymmetry may involve scapular control")

    # Squat
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

    # ASLR
    p = max(pts("aslr_right_angle"), pts("aslr_left_angle"))
    add("POST", p * 14, "Active Straight Leg Raise limitation")
    add("HIP", p * 8, "ASLR limitation may reflect hip mobility")
    add("PELV", p * 8, "ASLR limitation may reflect pelvic control")
    add("CORE", p * 5, "ASLR requires trunk-pelvis control")

    p = pts("aslr_asymmetry")
    add("POST", p * 8, "Right-left ASLR asymmetry")
    add("HIP", p * 5, "Right-left ASLR asymmetry may involve hip mobility")
    add("PELV", p * 5, "Right-left ASLR asymmetry may involve pelvic control")

    max_score = max(scores.values()) if scores else 0
    output: Dict[str, Dict[str, Any]] = {}

    for system, raw in scores.items():
        if raw <= 0:
            continue
        priority = round((raw / max_score) * 100, 1) if max_score > 0 else 0.0
        output[system] = {
            "system": system,
            "raw_score": round(raw, 2),
            "priority_score": priority,
            "reasons": reasons[system],
        }

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


def system_related_tests(system: str) -> Set[str]:
    mapping = {
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
        "REC": {"posture_side", "shoulder_right", "shoulder_left", "squat", "aslr_right", "aslr_left"},
    }
    return mapping.get(system, set())


def system_pain_status(system: str, pain: Dict[str, str]) -> str:
    tests = system_related_tests(system)
    statuses = [pain.get(t) for t in tests if pain.get(t)]

    if "pain" in statuses:
        return "pain"
    if "discomfort" in statuses:
        return "discomfort"
    return "no_pain"


def exercise_matches_system(exercise: Dict[str, Any], system: str) -> bool:
    if exercise.get("primary_system") == system:
        return True
    return system in (exercise.get("secondary_systems") or [])


def phase_rank(phase: str) -> int:
    return PHASE_ORDER.index(phase) if phase in PHASE_ORDER else 99


def exercise_score(
    ex: Dict[str, Any],
    target_system: str,
    desired_phase: str,
    week: int,
) -> float:
    score = 0.0

    if ex.get("primary_system") == target_system:
        score += 100
    elif target_system in (ex.get("secondary_systems") or []):
        score += 65

    if ex.get("phase") == desired_phase:
        score += 40
    else:
        diff = abs(phase_rank(ex.get("phase")) - phase_rank(desired_phase))
        score += max(0, 25 - diff * 8)

    difficulty = int(ex.get("difficulty", 3))
    # prefer easier in early weeks, more moderate in later weeks
    ideal_difficulty = 1 if week == 1 else 2 if week == 2 else 3
    score += max(0, 20 - abs(difficulty - ideal_difficulty) * 7)

    if int(ex.get("week_start", 1)) <= week <= int(ex.get("week_end", 4)):
        score += 20

    # Encourage exercises with clear prescription
    if ex.get("reps") or ex.get("hold"):
        score += 5

    return score


def is_exercise_safe_for_status(ex: Dict[str, Any], pain_status: str) -> bool:
    phase = ex.get("phase")
    difficulty = int(ex.get("difficulty", 3))

    if pain_status == "pain":
        return phase in PAIN_ALLOWED_PHASES and difficulty <= 2

    if pain_status == "discomfort":
        if phase in DISCOMFORT_BLOCKED_PHASES:
            return False
        if difficulty >= 4:
            return False

    return True


def find_best_exercise(
    exercises: List[Dict[str, Any]],
    target_system: str,
    desired_phase: str,
    week: int,
    pain_status: str,
    used_ids: Set[str],
    used_names: Set[str],
) -> Optional[Dict[str, Any]]:
    candidates = []

    for ex in exercises:
        if ex.get("id") in used_ids:
            continue
        if ex.get("name") in used_names:
            continue
        if not exercise_matches_system(ex, target_system):
            continue
        if not is_exercise_safe_for_status(ex, pain_status):
            continue
        if int(ex.get("week_start", 1)) > week:
            continue
        if int(ex.get("week_end", 4)) < week:
            continue

        score = exercise_score(ex, target_system, desired_phase, week)
        candidates.append((score, ex.get("id", ""), ex))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][2]


def format_exercise_for_program(ex: Dict[str, Any], order: int, system: str, lang: str) -> Dict[str, Any]:
    return {
        "order": order,
        "id": ex.get("id"),
        "name": ex.get("name"),
        "purpose": label_system(system, lang),
        "phase": ex.get("phase"),
        "phase_label": label_phase(ex.get("phase"), lang),
        "primary_system": ex.get("primary_system"),
        "secondary_systems": ex.get("secondary_systems", []),
        "difficulty": ex.get("difficulty"),
        "equipment": ex.get("equipment", []),
        "sets": ex.get("sets"),
        "reps": ex.get("reps"),
        "hold": ex.get("hold"),
        "tempo": ex.get("tempo"),
        "frequency_per_week": ex.get("frequency_per_week"),
        "duration_minutes": ex.get("duration_minutes"),
        "instructions": ex.get("instructions"),
        "common_errors": ex.get("common_errors"),
        "video_url": ex.get("video_url"),
        "thumbnail_url": ex.get("thumbnail_url"),
        "tags": ex.get("tags", []),
    }


def build_week_session(
    week: int,
    top_systems: List[str],
    movement_scores: Dict[str, Dict[str, Any]],
    exercises: List[Dict[str, Any]],
    pain: Dict[str, str],
    lang: str,
) -> Dict[str, Any]:
    cap = WEEK_CAPS.get(week, 6)
    desired_phases = WEEK_PHASE_PLAN[week]

    used_ids: Set[str] = set()
    used_names: Set[str] = set()
    selected: List[Dict[str, Any]] = []

    # Weighted system rotation: highest priority gets more chances.
    system_rotation: List[str] = []
    if top_systems:
        system_rotation.append(top_systems[0])
    if len(top_systems) > 1:
        system_rotation.append(top_systems[1])
    if len(top_systems) > 2:
        system_rotation.append(top_systems[2])
    if top_systems:
        system_rotation.append(top_systems[0])
    if len(top_systems) > 3:
        system_rotation.append(top_systems[3])
    if len(top_systems) > 1:
        system_rotation.append(top_systems[1])

    # Ensure length matches phases
    while len(system_rotation) < len(desired_phases):
        system_rotation.append(top_systems[0] if top_systems else "FUNC")

    for desired_phase, system in zip(desired_phases, system_rotation):
        if len(selected) >= cap:
            break

        pain_status = system_pain_status(system, pain)
        chosen = find_best_exercise(
            exercises=exercises,
            target_system=system,
            desired_phase=desired_phase,
            week=week,
            pain_status=pain_status,
            used_ids=used_ids,
            used_names=used_names,
        )

        # Fallback: if target system has no exercise, try functional integration.
        if chosen is None and system != "FUNC":
            chosen = find_best_exercise(
                exercises=exercises,
                target_system="FUNC",
                desired_phase=desired_phase,
                week=week,
                pain_status=pain_status,
                used_ids=used_ids,
                used_names=used_names,
            )

        if chosen is None:
            continue

        used_ids.add(chosen["id"])
        used_names.add(chosen["name"])
        selected.append(format_exercise_for_program(chosen, len(selected) + 1, system, lang))

    estimated_duration = sum(int(x.get("duration_minutes") or 2) for x in selected)

    theme_fr = [
        "Reset + mobilité",
        "Mobilité + contrôle moteur",
        "Contrôle + stabilité",
        "Stabilité + intégration fonctionnelle",
    ][week - 1]
    theme_en = [
        "Reset + mobility",
        "Mobility + motor control",
        "Control + stability",
        "Stability + functional integration",
    ][week - 1]

    return {
        "week": week,
        "theme_fr": theme_fr,
        "theme_en": theme_en,
        "theme": theme_en if lang == "en" else theme_fr,
        "estimated_duration_minutes": estimated_duration,
        "recommended_frequency_per_week": 3 if week >= 2 else 4,
        "session_structure": [
            {
                "order": x["order"],
                "exercise_id": x["id"],
                "exercise": x["name"],
                "purpose": x["purpose"],
                "phase": x["phase"],
                "phase_label": x["phase_label"],
            }
            for x in selected
        ],
        "exercises": selected,
    }


def generate_program_from_report(
    report: Dict[str, Any],
    exercise_library: Optional[Dict[str, Any]] = None,
    lang: str = "fr",
    pain_clearance: Optional[Dict[str, str]] = None,
    max_priority_systems: int = 4,
) -> Dict[str, Any]:
    lang = normalize_lang(lang)
    library = exercise_library or load_exercise_library()
    exercises = library.get("exercises", [])

    rating_map = extract_rating_map(report)
    movement_scores = compute_movement_system_scores(rating_map)
    pain = normalize_pain_clearance(pain_clearance)

    if not movement_scores:
        movement_scores = {
            "FUNC": {
                "system": "FUNC",
                "raw_score": 1.0,
                "priority_score": 100.0,
                "reasons": ["No major deficit detected; maintenance program selected."],
            }
        }

    top_systems = list(movement_scores.keys())[:max_priority_systems]

    weeks = [
        build_week_session(
            week=week,
            top_systems=top_systems,
            movement_scores=movement_scores,
            exercises=exercises,
            pain=pain,
            lang=lang,
        )
        for week in range(1, 5)
    ]

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
        "engine_version": "FlexiLab Program Engine V2",
        "library_version": library.get("version"),
        "language": lang,
        "movement_system_scores": movement_scores,
        "top_priority_systems": [
            {
                "system": s,
                "system_label": label_system(s, lang),
                "priority_score": movement_scores[s]["priority_score"],
                "reasons": movement_scores[s]["reasons"],
                "pain_status": system_pain_status(s, pain),
                "pain_limited": system_pain_status(s, pain) == "pain",
            }
            for s in top_systems
        ],
        "pain_clearance": pain,
        "weeks": weeks,
        "safety_notes_fr": safety_notes_fr,
        "safety_notes_en": safety_notes_en,
        "safety_notes": safety_notes_en if lang == "en" else safety_notes_fr,
        "progression_rule_fr": "Semaine 1 mobilité/reset, semaine 2 contrôle, semaine 3 stabilité, semaine 4 intégration. Re-tester après 4 semaines.",
        "progression_rule_en": "Week 1 mobility/reset, week 2 control, week 3 stability, week 4 integration. Re-test after 4 weeks.",
        "progression_rule": (
            "Week 1 mobility/reset, week 2 control, week 3 stability, week 4 integration. Re-test after 4 weeks."
            if lang == "en"
            else "Semaine 1 mobilité/reset, semaine 2 contrôle, semaine 3 stabilité, semaine 4 intégration. Re-tester après 4 semaines."
        )
    }


def generate_program_from_ratings(
    ratings: Dict[str, str],
    lang: str = "fr",
    pain_clearance: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    fake_report = {
        "sections": [
            {
                "id": "manual",
                "items": [
                    {"id": k, "rating": v}
                    for k, v in ratings.items()
                ]
            }
        ]
    }
    return generate_program_from_report(
        report=fake_report,
        lang=lang,
        pain_clearance=pain_clearance,
    )
