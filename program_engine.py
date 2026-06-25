"""
FlexiLab Program Engine V4

Adds to V3:
1. Keeps root-cause analysis and weekly progression logic.
2. Adds professional report-ready output.
3. Adds Session A/B/C for each week.
4. Adds exercise priority: essential / recommended / optional.
5. Adds "why prescribed" explanations.
6. Adds FlexiLab Priority Index.
7. Adds expected reassessment guidance.
8. Supports bilingual exercise fields when present in exercise_library.json.

Compatible with app.py endpoint:
    from program_engine import generate_program_from_report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


LIBRARY_PATH = Path(__file__).with_name("exercise_library.json")

SEVERITY_POINTS = {"red": 3, "yellow": 2, "green": 0, "unknown": 0, None: 0}

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

FAULT_LABELS = {
    "forward_head": {"fr": "Projection cervicale / tête en avant", "en": "Forward head posture"},
    "thoracic_posture": {"fr": "Mobilité / posture thoracique", "en": "Thoracic mobility/posture"},
    "overhead_limitation": {"fr": "Limitation overhead", "en": "Overhead limitation"},
    "squat_trunk_lean": {"fr": "Inclinaison du tronc en squat", "en": "Squat trunk lean"},
    "squat_depth_control": {"fr": "Contrôle / profondeur du squat", "en": "Squat depth/control"},
    "aslr_limitation": {"fr": "Limitation ASLR / chaîne postérieure", "en": "ASLR/posterior-chain limitation"},
    "asymmetry": {"fr": "Asymétrie droite/gauche", "en": "Right-left asymmetry"},
}

PRIORITY_LABELS = {
    "essential": {"fr": "Essentiel", "en": "Essential"},
    "recommended": {"fr": "Recommandé", "en": "Recommended"},
    "optional": {"fr": "Optionnel", "en": "Optional"},
}

WEEK_BLUEPRINTS = {
    1: [
        {"slot": "reset_breathing", "phase": "1_release_recovery", "systems": ["REC", "CORE", "THOR"]},
        {"slot": "soft_tissue_or_reset", "phase": "1_release_recovery", "systems": ["REC", "THOR", "HIP", "POST"]},
        {"slot": "mobility_primary", "phase": "2_mobility", "systems": []},
        {"slot": "mobility_secondary", "phase": "2_mobility", "systems": []},
        {"slot": "easy_control", "phase": "3_motor_control", "systems": []},
    ],
    2: [
        {"slot": "mobility_primary", "phase": "2_mobility", "systems": []},
        {"slot": "mobility_secondary", "phase": "2_mobility", "systems": []},
        {"slot": "control_primary", "phase": "3_motor_control", "systems": []},
        {"slot": "control_secondary", "phase": "3_motor_control", "systems": []},
        {"slot": "easy_stability", "phase": "4_stability", "systems": []},
        {"slot": "pattern_rebuild", "phase": "6_integration", "systems": ["FUNC", "CORE", "HIP"]},
    ],
    3: [
        {"slot": "mobility_maintenance", "phase": "2_mobility", "systems": []},
        {"slot": "control_primary", "phase": "3_motor_control", "systems": []},
        {"slot": "stability_primary", "phase": "4_stability", "systems": []},
        {"slot": "stability_secondary", "phase": "4_stability", "systems": []},
        {"slot": "strength_intro", "phase": "5_strength", "systems": ["CORE", "HIP", "FUNC"]},
        {"slot": "integration", "phase": "6_integration", "systems": ["FUNC", "THOR", "HIP", "CORE"]},
    ],
    4: [
        {"slot": "mobility_maintenance", "phase": "2_mobility", "systems": []},
        {"slot": "stability_primary", "phase": "4_stability", "systems": []},
        {"slot": "strength_primary", "phase": "5_strength", "systems": ["CORE", "HIP", "FUNC"]},
        {"slot": "integration_primary", "phase": "6_integration", "systems": ["FUNC", "CORE", "HIP"]},
        {"slot": "integration_secondary", "phase": "6_integration", "systems": ["THOR", "SHLD", "SCAP", "CORE"]},
        {"slot": "retest_prep", "phase": "3_motor_control", "systems": []},
    ],
}

SESSION_SPLITS = {
    "A": [1, 2, 3, 4],
    "B": [1, 3, 4, 5],
    "C": [2, 3, 5, 6],
}


def load_exercise_library(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or LIBRARY_PATH
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_lang(lang: str = "fr") -> str:
    return "en" if str(lang).lower().startswith("en") else "fr"


def localized(ex: Dict[str, Any], field: str, lang: str) -> str:
    lang = normalize_lang(lang)
    return ex.get(f"{field}_{lang}") or ex.get(field) or ex.get(f"{field}_en") or ""


def label_system(system_code: str, lang: str = "fr") -> str:
    lang = normalize_lang(lang)
    return SYSTEM_LABELS.get(system_code, {}).get(lang, system_code)


def label_phase(phase: str, lang: str = "fr") -> str:
    lang = normalize_lang(lang)
    return PHASE_LABELS.get(phase, {}).get(lang, phase)


def label_fault(fault: str, lang: str = "fr") -> str:
    lang = normalize_lang(lang)
    return FAULT_LABELS.get(fault, {}).get(lang, fault)


def label_priority(priority: str, lang: str = "fr") -> str:
    lang = normalize_lang(lang)
    return PRIORITY_LABELS.get(priority, {}).get(lang, priority)


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


def points(rating_map: Dict[str, str], key: str) -> int:
    return SEVERITY_POINTS.get(rating_map.get(key), 0)


def add_score(scores: Dict[str, float], reasons: Dict[str, List[str]], system: str, pts: float, reason: str) -> None:
    if pts <= 0:
        return
    scores[system] = scores.get(system, 0) + pts
    if reason not in reasons.setdefault(system, []):
        reasons[system].append(reason)


def compute_root_cause_analysis(rating_map: Dict[str, str], lang: str = "fr") -> List[Dict[str, Any]]:
    lang = normalize_lang(lang)
    faults: Dict[str, Dict[str, Any]] = {}

    def add_fault(fault: str, severity_points: int, contributors: Dict[str, float], evidence: str) -> None:
        if severity_points <= 0:
            return
        if fault not in faults:
            faults[fault] = {"fault": fault, "label": label_fault(fault, lang), "raw_score": 0.0, "contributors": {}, "evidence": []}
        faults[fault]["raw_score"] += severity_points * 10
        faults[fault]["evidence"].append(evidence)
        for system, weight in contributors.items():
            faults[fault]["contributors"][system] = faults[fault]["contributors"].get(system, 0) + severity_points * weight

    add_fault("forward_head", points(rating_map, "neck_angle"), {"CERV": 12, "THOR": 5, "SCAP": 2}, "neck_angle outside target zone")
    add_fault("thoracic_posture", points(rating_map, "thoracic_angle"), {"THOR": 14, "CERV": 3, "SHLD": 4}, "thoracic_angle outside target zone")
    shoulder_sev = max(points(rating_map, "shoulder_right_flexion"), points(rating_map, "shoulder_left_flexion"))
    add_fault("overhead_limitation", shoulder_sev, {"SHLD": 12, "THOR": 8, "SCAP": 8, "CERV": 2}, "shoulder flexion below target")
    add_fault("squat_trunk_lean", points(rating_map, "squat_trunk_lean"), {"CORE": 12, "HIP": 8, "ANKL": 6, "THOR": 4, "FUNC": 5}, "trunk lean during squat outside target zone")
    add_fault("squat_depth_control", points(rating_map, "squat_knee_angle"), {"KNEE": 8, "HIP": 8, "ANKL": 8, "FUNC": 5}, "squat knee angle/depth outside target zone")
    aslr_sev = max(points(rating_map, "aslr_right_angle"), points(rating_map, "aslr_left_angle"))
    add_fault("aslr_limitation", aslr_sev, {"POST": 14, "HIP": 8, "PELV": 8, "CORE": 5}, "ASLR below target zone")
    asym_sev = max(points(rating_map, "shoulders_asymmetry"), points(rating_map, "aslr_asymmetry"))
    add_fault("asymmetry", asym_sev, {"SHLD": 5, "SCAP": 4, "POST": 5, "HIP": 4, "PELV": 4}, "right-left asymmetry detected")

    if not faults:
        return []
    max_raw = max(v["raw_score"] for v in faults.values())
    output = []
    for fault, obj in faults.items():
        contributors = obj["contributors"]
        total = sum(contributors.values()) or 1
        contributor_list = [{"system": s, "system_label": label_system(s, lang), "confidence": round((v / total) * 100, 1)} for s, v in sorted(contributors.items(), key=lambda kv: kv[1], reverse=True)]
        output.append({"fault": fault, "label": obj["label"], "priority_score": round((obj["raw_score"] / max_raw) * 100, 1), "contributors": contributor_list, "evidence": obj["evidence"]})
    return sorted(output, key=lambda x: x["priority_score"], reverse=True)


def compute_movement_system_scores(rating_map: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    root_causes = compute_root_cause_analysis(rating_map, lang="en")
    scores: Dict[str, float] = {}
    reasons: Dict[str, List[str]] = {}
    for fault in root_causes:
        fault_weight = fault["priority_score"] / 100.0
        for contributor in fault["contributors"]:
            system = contributor["system"]
            confidence = contributor["confidence"] / 100.0
            add_score(scores, reasons, system, fault_weight * confidence * 100, f"{fault['fault']}: {', '.join(fault.get('evidence', []))}")
    if not scores:
        return {}
    max_score = max(scores.values())
    output = {}
    for system, raw in scores.items():
        output[system] = {"system": system, "raw_score": round(raw, 2), "priority_score": round((raw / max_score) * 100, 1), "reasons": reasons.get(system, [])}
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
        "CERV": {"posture_side"}, "THOR": {"posture_side", "shoulder_right", "shoulder_left", "squat"},
        "SHLD": {"shoulder_right", "shoulder_left"}, "SCAP": {"shoulder_right", "shoulder_left"},
        "CORE": {"squat", "aslr_right", "aslr_left"}, "PELV": {"squat", "aslr_right", "aslr_left"},
        "HIP": {"squat", "aslr_right", "aslr_left"}, "POST": {"aslr_right", "aslr_left"},
        "ANKL": {"squat"}, "KNEE": {"squat"}, "FUNC": {"squat"},
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
    return exercise.get("primary_system") == system or system in (exercise.get("secondary_systems") or [])


def phase_rank(phase: str) -> int:
    return PHASE_ORDER.index(phase) if phase in PHASE_ORDER else 99


def is_exercise_safe_for_status(ex: Dict[str, Any], pain_status: str) -> bool:
    phase = ex.get("phase")
    difficulty = int(ex.get("difficulty", 3))
    if pain_status == "pain":
        return phase in ["1_release_recovery", "2_mobility"] and difficulty <= 2
    if pain_status == "discomfort":
        return phase != "5_strength" and difficulty <= 3
    return True


def exercise_score(ex: Dict[str, Any], target_system: str, desired_phase: str, week: int) -> float:
    score = 0.0
    if ex.get("primary_system") == target_system:
        score += 100
    elif target_system in (ex.get("secondary_systems") or []):
        score += 65
    phase = ex.get("phase")
    if phase == desired_phase:
        score += 50
    else:
        score += max(0, 25 - abs(phase_rank(phase) - phase_rank(desired_phase)) * 10)
    difficulty = int(ex.get("difficulty", 3))
    ideal_difficulty = 1 if week == 1 else 2 if week == 2 else 3
    score += max(0, 20 - abs(difficulty - ideal_difficulty) * 8)
    if int(ex.get("week_start", 1)) <= week <= int(ex.get("week_end", 4)):
        score += 20
    if ex.get("reps") or ex.get("hold"):
        score += 5
    if week == 1 and phase == "1_release_recovery":
        score += 18
    if week == 1 and phase in ["5_strength", "6_integration"]:
        score -= 30
    return score


def find_best_exercise(exercises: List[Dict[str, Any]], target_systems: List[str], desired_phase: str, week: int, pain: Dict[str, str], used_ids: Set[str], used_names: Set[str]) -> Optional[Dict[str, Any]]:
    candidates = []
    for ex in exercises:
        if ex.get("id") in used_ids or ex.get("name") in used_names:
            continue
        if int(ex.get("week_start", 1)) > week or int(ex.get("week_end", 4)) < week:
            continue
        matching_systems = [s for s in target_systems if exercise_matches_system(ex, s)]
        if not matching_systems:
            continue
        worst_status = "no_pain"
        for s in matching_systems:
            status = system_pain_status(s, pain)
            if status == "pain":
                worst_status = "pain"
                break
            if status == "discomfort":
                worst_status = "discomfort"
        if not is_exercise_safe_for_status(ex, worst_status):
            continue
        best_system_score = max(exercise_score(ex, s, desired_phase, week) for s in matching_systems)
        candidates.append((best_system_score, ex.get("id", ""), ex, matching_systems[0]))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    chosen = candidates[0][2].copy()
    chosen["_selected_for_system"] = candidates[0][3]
    return chosen


def slot_systems_from_priorities(slot: Dict[str, Any], top_systems: List[str]) -> List[str]:
    explicit = slot.get("systems") or []
    return list(dict.fromkeys(explicit + top_systems)) if explicit else top_systems


def infer_priority(ex: Dict[str, Any], order: int) -> str:
    if ex.get("exercise_priority"):
        return ex["exercise_priority"]
    if order <= 3:
        return "essential"
    if order <= 5:
        return "recommended"
    return "optional"


def format_exercise_for_program(ex: Dict[str, Any], order: int, lang: str) -> Dict[str, Any]:
    selected_system = ex.get("_selected_for_system") or ex.get("primary_system")
    priority = infer_priority(ex, order)
    return {
        "order": order,
        "id": ex.get("id"),
        "name": localized(ex, "name", lang),
        "name_en": ex.get("name_en") or ex.get("name"),
        "name_fr": ex.get("name_fr") or ex.get("name"),
        "purpose": label_system(selected_system, lang),
        "selected_for_system": selected_system,
        "phase": ex.get("phase"),
        "phase_label": label_phase(ex.get("phase"), lang),
        "priority": priority,
        "priority_label": label_priority(priority, lang),
        "why_prescribed": localized(ex, "why_prescribed", lang) or build_why_prescribed(ex, selected_system, lang),
        "expected_improvement": localized(ex, "expected_improvement", lang) or ex.get("expected_improvement", ""),
        "expected_improvement_window": ex.get("expected_improvement_window", "2-4 weeks"),
        "primary_system": ex.get("primary_system"),
        "secondary_systems": ex.get("secondary_systems", []),
        "movement_goals": ex.get("movement_goals", []),
        "movement_pattern": ex.get("movement_pattern"),
        "movement_plane": ex.get("movement_plane"),
        "difficulty": ex.get("difficulty"),
        "equipment": ex.get("equipment", []),
        "sets": ex.get("sets"),
        "reps": ex.get("reps"),
        "hold": ex.get("hold"),
        "tempo": ex.get("tempo"),
        "frequency_per_week": ex.get("frequency_per_week"),
        "duration_minutes": ex.get("duration_minutes"),
        "instructions": localized(ex, "instructions", lang),
        "coaching_cues": ex.get("coaching_cues", []),
        "common_errors": ex.get("common_errors"),
        "video_url": ex.get("video_url"),
        "thumbnail_url": ex.get("thumbnail_url"),
        "tags": ex.get("tags", []),
    }


def build_why_prescribed(ex: Dict[str, Any], selected_system: str, lang: str) -> str:
    goals = ", ".join((ex.get("movement_goals") or [])[:3])
    if normalize_lang(lang) == "en":
        return f"Selected to support {label_system(selected_system, lang)} by targeting {goals}."
    return f"Sélectionné pour soutenir {label_system(selected_system, lang)} en ciblant {goals}."


def build_week_session(week: int, top_systems: List[str], exercises: List[Dict[str, Any]], pain: Dict[str, str], lang: str) -> Dict[str, Any]:
    blueprint = WEEK_BLUEPRINTS[week]
    used_ids: Set[str] = set()
    used_names: Set[str] = set()
    selected: List[Dict[str, Any]] = []
    for slot in blueprint:
        target_systems = slot_systems_from_priorities(slot, top_systems)
        chosen = find_best_exercise(exercises, target_systems, slot["phase"], week, pain, used_ids, used_names)
        if chosen is None:
            continue
        used_ids.add(chosen["id"])
        used_names.add(chosen["name"])
        selected.append(format_exercise_for_program(chosen, len(selected) + 1, lang))
    estimated_duration = sum(int(x.get("duration_minutes") or 2) for x in selected)
    theme_fr = ["Restaurer la respiration, la mobilité et les amplitudes de base", "Contrôler la mobilité et stabiliser les positions", "Renforcer le contrôle et stabiliser sous contrainte légère", "Intégrer les acquis dans des mouvements fonctionnels"][week - 1]
    theme_en = ["Restore breathing, mobility and basic ranges", "Control mobility and stabilize positions", "Strengthen control and stabilize under light demand", "Integrate gains into functional movement"][week - 1]
    sessions = build_sessions_from_week(selected, week, lang)
    return {
        "week": week,
        "theme_fr": theme_fr,
        "theme_en": theme_en,
        "theme": theme_en if lang == "en" else theme_fr,
        "estimated_duration_minutes": estimated_duration,
        "recommended_frequency_per_week": 4 if week == 1 else 3,
        "session_structure": [{"order": x["order"], "exercise_id": x["id"], "exercise": x["name"], "purpose": x["purpose"], "phase": x["phase"], "phase_label": x["phase_label"], "priority": x["priority"], "priority_label": x["priority_label"]} for x in selected],
        "exercises": selected,
        "sessions": sessions,
    }


def build_sessions_from_week(exercises: List[Dict[str, Any]], week: int, lang: str) -> List[Dict[str, Any]]:
    output = []
    day_names = {"A": {"fr": "Session A", "en": "Session A"}, "B": {"fr": "Session B", "en": "Session B"}, "C": {"fr": "Session C", "en": "Session C"}}
    days = {"A": {"fr": "Lundi", "en": "Monday"}, "B": {"fr": "Mercredi", "en": "Wednesday"}, "C": {"fr": "Vendredi", "en": "Friday"}}
    for key, orders in SESSION_SPLITS.items():
        selected = [e for e in exercises if e["order"] in orders]
        if not selected:
            continue
        output.append({
            "session_id": f"W{week}{key}",
            "title": day_names[key][normalize_lang(lang)],
            "suggested_day": days[key][normalize_lang(lang)],
            "estimated_duration_minutes": sum(int(e.get("duration_minutes") or 2) for e in selected),
            "focus": build_session_focus(selected, lang),
            "exercises": selected,
        })
    return output


def build_session_focus(exercises: List[Dict[str, Any]], lang: str) -> str:
    phases = list(dict.fromkeys([e["phase_label"] for e in exercises]))
    purposes = list(dict.fromkeys([e["purpose"] for e in exercises]))
    if normalize_lang(lang) == "en":
        return f"{', '.join(phases)} for {', '.join(purposes[:3])}"
    return f"{', '.join(phases)} pour {', '.join(purposes[:3])}"


def build_priority_index(movement_scores: Dict[str, Dict[str, Any]], lang: str) -> List[Dict[str, Any]]:
    index = []
    for system, data in movement_scores.items():
        score = data["priority_score"]
        if score >= 80:
            color = "red"
        elif score >= 55:
            color = "orange"
        elif score >= 30:
            color = "yellow"
        else:
            color = "green"
        index.append({"system": system, "label": label_system(system, lang), "priority_score": score, "color": color})
    return index


def build_reassessment_plan(movement_scores: Dict[str, Dict[str, Any]], lang: str) -> Dict[str, Any]:
    top = list(movement_scores.keys())[:3]
    if normalize_lang(lang) == "en":
        return {
            "retest_after": "4 weeks",
            "objective": "Repeat the same screening and compare FlexiLab score, root-cause priorities and movement-system scores.",
            "progress_rule": "If a priority system improves below 55, progress to the next exercise level. If it remains above 80, repeat or regress the current phase.",
            "systems_to_monitor": [{"system": s, "label": label_system(s, lang)} for s in top],
        }
    return {
        "retest_after": "4 semaines",
        "objective": "Refaire le même screening et comparer le score FlexiLab, les causes probables et les scores par système.",
        "progress_rule": "Si un système prioritaire passe sous 55, progresser vers le niveau supérieur. S’il reste au-dessus de 80, répéter ou régresser la phase actuelle.",
        "systems_to_monitor": [{"system": s, "label": label_system(s, lang)} for s in top],
    }


def build_report_ready_summary(report: Dict[str, Any], root_causes: List[Dict[str, Any]], movement_scores: Dict[str, Dict[str, Any]], weeks: List[Dict[str, Any]], lang: str) -> Dict[str, Any]:
    score = report.get("flexilab_score")
    top_fault = root_causes[0]["label"] if root_causes else None
    top_systems = list(movement_scores.keys())[:3]
    if normalize_lang(lang) == "en":
        headline = f"FlexiLab score: {score}. Main focus: {top_fault or 'maintenance'}."
        next_action = "Follow the 4-week corrective plan, then repeat the same screening."
    else:
        headline = f"Score FlexiLab : {score}. Focus principal : {top_fault or 'entretien'}."
        next_action = "Suivre le plan correctif 4 semaines, puis refaire le même screening."
    return {
        "headline": headline,
        "top_systems": [{"system": s, "label": label_system(s, lang), "priority_score": movement_scores[s]["priority_score"]} for s in top_systems],
        "total_sessions": sum(len(w.get("sessions", [])) for w in weeks),
        "average_session_duration_minutes": round(sum(s["estimated_duration_minutes"] for w in weeks for s in w.get("sessions", [])) / max(1, sum(len(w.get("sessions", [])) for w in weeks)), 1),
        "next_action": next_action,
    }


def generate_program_from_report(report: Dict[str, Any], exercise_library: Optional[Dict[str, Any]] = None, lang: str = "fr", pain_clearance: Optional[Dict[str, str]] = None, max_priority_systems: int = 5) -> Dict[str, Any]:
    lang = normalize_lang(lang)
    library = exercise_library or load_exercise_library()
    exercises = library.get("exercises", [])
    rating_map = extract_rating_map(report)
    root_causes = compute_root_cause_analysis(rating_map, lang=lang)
    movement_scores = compute_movement_system_scores(rating_map)
    pain = normalize_pain_clearance(pain_clearance)

    if not movement_scores:
        movement_scores = {"FUNC": {"system": "FUNC", "raw_score": 1.0, "priority_score": 100.0, "reasons": ["No major deficit detected; maintenance program selected."]}}

    top_systems = list(movement_scores.keys())[:max_priority_systems]
    weeks = [build_week_session(week, top_systems, exercises, pain, lang) for week in range(1, 5)]
    priority_index = build_priority_index(movement_scores, lang)
    reassessment_plan = build_reassessment_plan(movement_scores, lang)
    report_ready_summary = build_report_ready_summary(report, root_causes, movement_scores, weeks, lang)

    safety_notes_fr = ["Ne pas faire un exercice qui provoque une douleur.", "Garder une respiration calme et un mouvement contrôlé.", "Progresser uniquement si l’alignement est maîtrisé sans compensation.", "En cas de douleur, engourdissement, faiblesse inexpliquée ou symptôme inhabituel : arrêter et demander un avis médical."]
    safety_notes_en = ["Do not perform any exercise that causes pain.", "Keep breathing calm and movement controlled.", "Progress only when alignment is controlled without compensation.", "In case of pain, numbness, unexplained weakness or unusual symptoms: stop and seek medical advice."]

    return {
        "engine_version": "FlexiLab Program Engine V4",
        "library_version": library.get("version"),
        "language": lang,
        "report_ready_summary": report_ready_summary,
        "root_cause_analysis": root_causes,
        "movement_system_scores": movement_scores,
        "flexilab_priority_index": priority_index,
        "top_priority_systems": [{"system": s, "system_label": label_system(s, lang), "priority_score": movement_scores[s]["priority_score"], "reasons": movement_scores[s]["reasons"], "pain_status": system_pain_status(s, pain), "pain_limited": system_pain_status(s, pain) == "pain"} for s in top_systems],
        "pain_clearance": pain,
        "weeks": weeks,
        "reassessment_plan": reassessment_plan,
        "safety_notes_fr": safety_notes_fr,
        "safety_notes_en": safety_notes_en,
        "safety_notes": safety_notes_en if lang == "en" else safety_notes_fr,
        "progression_rule_fr": "S1 restaurer, S2 contrôler, S3 stabiliser, S4 intégrer. Re-tester après 4 semaines.",
        "progression_rule_en": "W1 restore, W2 control, W3 stabilize, W4 integrate. Re-test after 4 weeks.",
        "progression_rule": "W1 restore, W2 control, W3 stabilize, W4 integrate. Re-test after 4 weeks." if lang == "en" else "S1 restaurer, S2 contrôler, S3 stabiliser, S4 intégrer. Re-tester après 4 semaines.",
    }


def generate_program_from_ratings(ratings: Dict[str, str], lang: str = "fr", pain_clearance: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    fake_report = {"sections": [{"id": "manual", "items": [{"id": k, "rating": v} for k, v in ratings.items()]}]}
    return generate_program_from_report(report=fake_report, lang=lang, pain_clearance=pain_clearance)
