"""
FlexiLab Clinical Prescription Engine v1.0

Purpose
-------
Generate a clinically coherent 4-week corrective exercise program from:
- FlexiLab postural screening results
- movement domain scores
- pain/readiness information
- asymmetry data
- exercise library metadata

Important clinical principle
----------------------------
Pain does not automatically lower the movement score.
Pain creates a separate clinical readiness status and modifies exercise selection.

Expected files
--------------
- flexilab_exercise_library_v1.json
- prescription_rules_v1.json

Main function
-------------
generate_clinical_prescription(screening_payload, exercise_library=None, rules=None, language="fr")

Integration target
------------------
Use this function inside your `/program` endpoint or after `/final-report`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import math
from pathlib import Path
from datetime import datetime, timezone


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------

def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_exercise_library(path: str | Path) -> List[Dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Invalid exercise library format")


# ---------------------------------------------------------------------
# Core score/readiness helpers
# ---------------------------------------------------------------------

def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def score_band(score: float, language: str = "fr") -> Dict[str, str]:
    score = _to_float(score)
    if score >= 80:
        return {"color": "green", "label": "Bon" if language == "fr" else "Good"}
    if score >= 70:
        return {"color": "yellow", "label": "Correct" if language == "fr" else "Fair"}
    if score >= 60:
        return {"color": "orange", "label": "À améliorer" if language == "fr" else "Needs improvement"}
    return {"color": "red", "label": "Limité" if language == "fr" else "Limited"}


def pain_readiness_from_intake(intake: Optional[Dict[str, Any]], language: str = "fr") -> Dict[str, Any]:
    intake = intake or {}
    pain_score = _to_float(
        intake.get("pain_score",
        intake.get("pain_level",
        intake.get("pain", 0))),
        0.0
    )
    red_flags = intake.get("red_flags", []) or []
    pain_region = intake.get("pain_region", intake.get("pain_area", ""))

    if red_flags:
        status = "medical_clearance_recommended"
        mode = "recovery_only"
        label_fr = "Avis médical recommandé"
        label_en = "Medical clearance recommended"
        advice_fr = "Des signaux d'alerte ont été indiqués. Éviter le programme correctif complet avant avis médical."
        advice_en = "Red flags were reported. Avoid the full corrective program before medical advice."
    elif pain_score >= 7:
        status = "medical_clearance_recommended"
        mode = "recovery_only"
        label_fr = "Douleur élevée"
        label_en = "High pain"
        advice_fr = "Douleur élevée : privilégier les mouvements doux sans douleur et demander un avis médical."
        advice_en = "High pain: use gentle pain-free movement only and seek medical advice."
    elif pain_score >= 4:
        status = "limited"
        mode = "recovery_control"
        label_fr = "Douleur modérée"
        label_en = "Moderate pain"
        advice_fr = "Douleur modérée : réduire l'amplitude, éviter les exercices provocateurs et demander conseil si la douleur persiste."
        advice_en = "Moderate pain: reduce range, avoid provocative exercises, and seek advice if pain persists."
    elif pain_score >= 1:
        status = "caution"
        mode = "pain_free_corrective"
        label_fr = "Douleur légère"
        label_en = "Mild pain"
        advice_fr = "Douleur légère : travailler uniquement dans une amplitude confortable et sans aggravation."
        advice_en = "Mild pain: use comfortable pain-free range only and avoid symptom aggravation."
    else:
        status = "normal"
        mode = "corrective"
        label_fr = "Aucune douleur"
        label_en = "No pain"
        advice_fr = "Programme correctif standard, en gardant une exécution contrôlée et sans douleur."
        advice_en = "Standard corrective program, keeping movement controlled and pain-free."

    return {
        "pain_score": pain_score,
        "pain_region": pain_region,
        "red_flags": red_flags,
        "readiness": status,
        "program_mode": mode,
        "label": label_fr if language == "fr" else label_en,
        "advice": advice_fr if language == "fr" else advice_en,
        "medical_advice_recommended": bool(red_flags or pain_score >= 4),
    }


# ---------------------------------------------------------------------
# Screening parsing
# ---------------------------------------------------------------------

def extract_domain_scores(screening_payload: Dict[str, Any], rules: Dict[str, Any], language: str = "fr") -> List[Dict[str, Any]]:
    """
    Reads domain scores from modern `report.score_v2.domain_scores`.
    Falls back to program movement_system_scores if needed.
    """
    report = screening_payload.get("report", screening_payload) or {}
    score_v2 = report.get("score_v2") or {}

    domains = score_v2.get("domain_scores") or []
    if domains:
        normalized = []
        for d in domains:
            did = d.get("id") or d.get("domain") or d.get("system")
            score = _to_float(d.get("score", d.get("priority_score", 0)))
            normalized.append({
                "id": did,
                "label": d.get("label") or d.get("label_fr") or did,
                "label_en": d.get("label_en") or d.get("label") or did,
                "score": score,
                "band": score_band(score, language),
                "weight": d.get("weight", None),
            })
        return sorted(normalized, key=lambda x: x["score"])

    # Fallback for older program engine system scores
    program = screening_payload.get("program") or {}
    system_scores = program.get("movement_system_scores") or {}
    fallback_map = {
        "CERV": ("cervical_control", "Contrôle cervical", "Cervical Control"),
        "THOR": ("thoracic_mobility", "Mobilité thoracique", "Thoracic Mobility"),
        "SCAP": ("shoulder_mobility", "Mobilité des épaules", "Shoulder Mobility"),
        "CORE": ("core_stability", "Stabilité du tronc", "Core Stability"),
        "HIP": ("hip_mobility", "Mobilité de hanche", "Hip Mobility"),
        "ANKL": ("ankle_mobility", "Mobilité de cheville", "Ankle Mobility"),
        "FUNC": ("functional_integration", "Intégration fonctionnelle", "Functional Integration"),
    }
    out = []
    for sys, vals in system_scores.items():
        if sys in fallback_map:
            did, fr, en = fallback_map[sys]
            # Older movement_system_scores may be priority scores where higher means worse.
            # Convert priority to domain score.
            priority = _to_float(vals.get("priority_score", 0))
            score = max(0, min(100, 100 - priority))
            out.append({"id": did, "label": fr, "label_en": en, "score": score, "band": score_band(score, language), "weight": None})
    return sorted(out, key=lambda x: x["score"])


def extract_asymmetries(screening_payload: Dict[str, Any], language: str = "fr") -> List[Dict[str, Any]]:
    report = screening_payload.get("report", screening_payload) or {}
    items = {}
    for sec in report.get("sections", []) or []:
        for it in sec.get("items", []) or []:
            if it.get("id"):
                items[it["id"]] = it

    asymmetries = []

    # Shoulder flexion symmetry
    sr = items.get("shoulder_right_flexion", {}).get("value")
    sl = items.get("shoulder_left_flexion", {}).get("value")
    if sr is not None and sl is not None:
        sr, sl = _to_float(sr), _to_float(sl)
        diff = abs(sr - sl)
        restricted = "right" if sr < sl else "left" if sl < sr else "none"
        asymmetries.append({
            "id": "shoulder_symmetry",
            "label": "Symétrie mobilité épaules" if language == "fr" else "Shoulder mobility symmetry",
            "left_value": sl,
            "right_value": sr,
            "difference": diff,
            "restricted_side": restricted,
            "significant": diff >= 10,
        })

    # ASLR symmetry
    ar = items.get("aslr_right_angle", {}).get("value", items.get("aslr_right", {}).get("value"))
    al = items.get("aslr_left_angle", {}).get("value", items.get("aslr_left", {}).get("value"))
    if ar is not None and al is not None:
        ar, al = _to_float(ar), _to_float(al)
        diff = abs(ar - al)
        restricted = "right" if ar < al else "left" if al < ar else "none"
        asymmetries.append({
            "id": "aslr_symmetry",
            "label": "Symétrie ASLR / ischio-jambiers" if language == "fr" else "ASLR / hamstring symmetry",
            "left_value": al,
            "right_value": ar,
            "difference": diff,
            "restricted_side": restricted,
            "significant": diff >= 10,
        })

    return asymmetries


# ---------------------------------------------------------------------
# Exercise filtering/scoring
# ---------------------------------------------------------------------

def exercise_domains(exercise: Dict[str, Any]) -> List[str]:
    raw = exercise.get("screening_domains_improved", "")
    return [x.strip() for x in raw.split(",") if x.strip()]


def exercise_objective(exercise: Dict[str, Any]) -> str:
    return str(exercise.get("primary_objective", "")).lower()


def objective_matches(exercise: Dict[str, Any], wanted: List[str]) -> bool:
    obj = exercise_objective(exercise)
    return any(w in obj for w in wanted)


def is_pain_compatible(exercise: Dict[str, Any], readiness: Dict[str, Any]) -> bool:
    pain_mode = readiness.get("program_mode", "corrective")
    pain_rule = str(exercise.get("pain_compatibility", "")).lower()
    objective = exercise_objective(exercise)

    if pain_mode == "corrective":
        return True

    if pain_mode == "pain_free_corrective":
        return "contra" not in pain_rule

    if pain_mode == "recovery_control":
        # Keep recovery, breathing, gentle mobility, low-level motor control
        if "recovery" in objective:
            return True
        if "mobility" in objective and int(exercise.get("difficulty_1_5", 3)) <= 2:
            return True
        if "motor_control" in objective and int(exercise.get("difficulty_1_5", 3)) <= 2:
            return True
        return False

    if pain_mode == "recovery_only":
        return "recovery" in objective or exercise.get("category_code") == "RB"

    return True


def exercise_priority_score(
    exercise: Dict[str, Any],
    priorities: List[Dict[str, Any]],
    week_rule: Dict[str, Any],
    day_bias: List[str],
    readiness: Dict[str, Any],
    asymmetries: List[Dict[str, Any]],
) -> float:
    if not is_pain_compatible(exercise, readiness):
        return -9999

    difficulty = int(exercise.get("difficulty_1_5", 3))
    if difficulty > int(week_rule.get("difficulty_max", 5)):
        return -1000 + (5 - difficulty)

    ex_domains = exercise_domains(exercise)
    score = 0.0

    # Priority weighting based on lowest domains
    for rank, domain in enumerate(priorities[:6], start=1):
        if domain["id"] in ex_domains:
            base = 100 - _to_float(domain["score"])
            multiplier = {1: 1.6, 2: 1.3, 3: 1.1}.get(rank, 0.75)
            score += base * multiplier

    # Day objective bias
    if objective_matches(exercise, day_bias):
        score += 12

    # Week objectives
    if objective_matches(exercise, week_rule.get("objectives", [])):
        score += 10

    # Pain mode boost for recovery/control
    if readiness.get("program_mode") in ("recovery_control", "recovery_only"):
        if exercise.get("category_code") == "RB":
            score += 30
        if int(exercise.get("difficulty_1_5", 3)) <= 2:
            score += 8

    # Asymmetry boosts
    for a in asymmetries:
        if not a.get("significant"):
            continue
        if a["id"] == "shoulder_symmetry" and any(d in ex_domains for d in ["shoulder_mobility", "thoracic_mobility"]):
            score += 8
        if a["id"] == "aslr_symmetry" and any(d in ex_domains for d in ["hamstring_mobility", "hip_mobility"]):
            score += 8

    # Prefer lower difficulty early if close
    score += max(0, 5 - difficulty) * 1.5

    return score


def choose_session_exercises(
    library: List[Dict[str, Any]],
    priorities: List[Dict[str, Any]],
    week_rule: Dict[str, Any],
    day_key: str,
    day_rule: Dict[str, Any],
    readiness: Dict[str, Any],
    asymmetries: List[Dict[str, Any]],
    already_core_ids: List[str],
    language: str = "fr",
) -> List[Dict[str, Any]]:
    target_min, target_max = week_rule.get("session_target_exercises", [6, 8])
    bias = day_rule.get("bias", [])

    scored = []
    for ex in library:
        s = exercise_priority_score(ex, priorities, week_rule, bias, readiness, asymmetries)
        if s > -900:
            scored.append((s, ex))
    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    category_counts = {}

    # Always try to include 1 recovery/breathing or reset drill
    recovery_candidates = [ex for score, ex in scored if ex.get("category_code") == "RB"]
    if recovery_candidates:
        selected.append(recovery_candidates[0])
        category_counts["RB"] = 1

    # Maintain core high-value drills across weeks when still relevant
    for eid in already_core_ids:
        ex = next((e for e in library if e.get("exercise_id") == eid), None)
        if ex and ex not in selected and is_pain_compatible(ex, readiness):
            selected.append(ex)
            category_counts[ex["category_code"]] = category_counts.get(ex["category_code"], 0) + 1
            if len(selected) >= target_min:
                break

    # Add best candidates with category cap
    for score, ex in scored:
        if len(selected) >= target_max:
            break
        if ex in selected:
            continue
        cat = ex.get("category_code", "")
        if category_counts.get(cat, 0) >= 3:
            continue
        selected.append(ex)
        category_counts[cat] = category_counts.get(cat, 0) + 1

    return selected[:target_max]


# ---------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------

def localize_exercise(ex: Dict[str, Any], language: str = "fr") -> Dict[str, Any]:
    name = ex.get("name_fr") if language == "fr" else ex.get("name_en")
    target = ex.get("category_fr") if language == "fr" else ex.get("category_en")
    return {
        "id": ex.get("exercise_id"),
        "name": name or ex.get("name_en"),
        "name_en": ex.get("name_en"),
        "name_fr": ex.get("name_fr"),
        "category_code": ex.get("category_code"),
        "target": target,
        "primary_objective": ex.get("primary_objective"),
        "difficulty": ex.get("difficulty_1_5"),
        "phase": ex.get("phase"),
        "equipment": ex.get("equipment", ""),
        "sets": ex.get("sets", ""),
        "reps_time": ex.get("reps_time", ""),
        "tempo": ex.get("tempo", ""),
        "rest": ex.get("rest", ""),
        "frequency_per_week": ex.get("frequency_per_week", ""),
        "coaching_cues": ex.get("coaching_cues", ""),
        "common_errors": ex.get("common_errors", ""),
        "regression_id": ex.get("regression_id", ""),
        "progression_id": ex.get("progression_id", ""),
        "clinical_rationale": ex.get("clinical_rationale", ""),
        "pain_rule": ex.get("pain_rule", ""),
        "asymmetry_rule": ex.get("asymmetry_rule", ""),
        "video_url": ex.get("video_url", ""),
        "vimeo_url": ex.get("vimeo_url", ""),
        "mp4_url": ex.get("mp4_url", ""),
        "thumbnail_url": ex.get("thumbnail_url", ""),
    }


def week_objective_text(week: int, language: str = "fr") -> str:
    fr = {
        1: "Restaurer la respiration, la mobilité et les amplitudes de base.",
        2: "Améliorer le contrôle actif et stabiliser les positions.",
        3: "Renforcer le contrôle et stabiliser sous contrainte légère.",
        4: "Intégrer les corrections dans des mouvements fonctionnels.",
    }
    en = {
        1: "Restore breathing, mobility, and basic ranges.",
        2: "Improve active control and stabilize positions.",
        3: "Build control and stabilize under light demand.",
        4: "Integrate corrections into functional movement.",
    }
    return (fr if language == "fr" else en).get(week, "")


def generate_clinical_prescription(
    screening_payload: Dict[str, Any],
    exercise_library: Optional[List[Dict[str, Any]]] = None,
    rules: Optional[Dict[str, Any]] = None,
    language: str = "fr",
) -> Dict[str, Any]:
    """
    Main engine entry point.
    """
    if exercise_library is None:
        raise ValueError("exercise_library is required")
    if rules is None:
        raise ValueError("rules is required")

    report = screening_payload.get("report", screening_payload) or {}
    movement_score = _to_float(report.get("flexilab_score", report.get("score", 0)))
    intake = screening_payload.get("intake_context") or report.get("intake_context") or screening_payload.get("intake") or {}

    readiness = pain_readiness_from_intake(intake, language)
    domains = extract_domain_scores(screening_payload, rules, language)
    asymmetries = extract_asymmetries(screening_payload, language)

    if not domains:
        # fallback neutral priorities
        domains = [
            {"id": "thoracic_mobility", "label": "Mobilité thoracique", "label_en": "Thoracic Mobility", "score": 65, "band": score_band(65, language)},
            {"id": "core_stability", "label": "Stabilité du tronc", "label_en": "Core Stability", "score": 65, "band": score_band(65, language)},
            {"id": "cervical_control", "label": "Contrôle cervical", "label_en": "Cervical Control", "score": 70, "band": score_band(70, language)},
        ]

    # Priority domains: lowest scores first
    priorities = sorted(domains, key=lambda x: x["score"])
    main_priorities = priorities[:3]
    monitor_domains = priorities[3:]

    weeks_out = []
    persistent_core_ids = []

    for week_rule in rules.get("weekly_progression", []):
        week_no = int(week_rule["week"])
        sessions = []

        for day_no, (day_key, day_rule) in enumerate(rules.get("session_structure", {}).items(), start=1):
            selected = choose_session_exercises(
                exercise_library,
                priorities,
                week_rule,
                day_key,
                day_rule,
                readiness,
                asymmetries,
                persistent_core_ids,
                language,
            )

            # Update persistent foundational exercises: keep first 2-3 exercises linked to top priorities
            if week_no == 1 and day_no == 1:
                persistent_core_ids = [
                    ex["exercise_id"] for ex in selected
                    if ex.get("category_code") in ("TM", "SH", "CS", "HM", "AM", "CC")
                ][:3]

            sessions.append({
                "day": day_no,
                "focus": day_rule.get("focus_fr" if language == "fr" else "focus_en"),
                "estimated_duration_minutes": estimate_session_duration(selected),
                "exercises": [localize_exercise(ex, language) for ex in selected],
            })

        weeks_out.append({
            "week": week_no,
            "phase": week_rule.get("phase_fr" if language == "fr" else "phase"),
            "objective": week_objective_text(week_no, language),
            "difficulty_max": week_rule.get("difficulty_max"),
            "progression_logic": progression_logic_text(week_no, language),
            "sessions": sessions,
        })

    return {
        "engine_version": rules.get("engine_version", "FlexiLab Clinical Prescription Engine v1.0"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "language": language,
        "movement_score": movement_score,
        "movement_score_band": score_band(movement_score, language),
        "clinical_readiness": readiness,
        "main_priorities": main_priorities,
        "monitor_domains": monitor_domains,
        "asymmetries": asymmetries,
        "program_summary": {
            "duration": "4 weeks" if language != "fr" else "4 semaines",
            "frequency": "3 sessions/week" if language != "fr" else "3 séances/semaine",
            "session_duration": "15-30 min",
            "model": "adaptive maintenance + replacement",
            "medical_advice_recommended": readiness.get("medical_advice_recommended", False),
        },
        "weeks": weeks_out,
        "safety_notes": safety_notes(language, readiness),
        "reassessment_plan": {
            "when": "after 4 weeks" if language != "fr" else "après 4 semaines",
            "what": "Repeat the same screening and compare domain scores, symmetry, and pain status." if language != "fr" else "Refaire le même screening et comparer les scores par domaine, la symétrie et la douleur.",
        },
        "integration_notes": {
            "frontend": "Program page should render weeks/sessions/exercises and open video modal using vimeo_url/mp4_url.",
            "backend": "Use this output as the /program response. Do not regenerate separate PDFs; print the rendered program page.",
        }
    }


def estimate_session_duration(exercises: List[Dict[str, Any]]) -> int:
    # Conservative estimate based on count + holds/rests.
    if not exercises:
        return 0
    return max(12, min(35, 4 + len(exercises) * 3))


def progression_logic_text(week_no: int, language: str = "fr") -> str:
    fr = {
        1: "Installer les bases : respiration, mobilité douce, contrôle sans douleur.",
        2: "Conserver les exercices clés et ajouter du contrôle actif.",
        3: "Remplacer progressivement les exercices faciles par de la stabilité et du renforcement léger.",
        4: "Intégrer les acquis dans des mouvements fonctionnels et préparer le re-test.",
    }
    en = {
        1: "Build the foundation: breathing, gentle mobility, pain-free control.",
        2: "Keep high-value drills and add active control.",
        3: "Gradually replace easy drills with stability and light strengthening.",
        4: "Integrate gains into functional movement and prepare reassessment.",
    }
    return (fr if language == "fr" else en).get(week_no, "")


def safety_notes(language: str, readiness: Dict[str, Any]) -> List[str]:
    if language == "fr":
        notes = [
            "Aucun exercice ne doit provoquer ou augmenter la douleur.",
            "Réduire l’amplitude ou régresser si la qualité du mouvement diminue.",
            "Prioriser la respiration calme et le contrôle avant l’intensité.",
        ]
        if readiness.get("medical_advice_recommended"):
            notes.insert(0, "Un avis médical est recommandé si la douleur persiste, augmente ou s’accompagne de symptômes inhabituels.")
        return notes

    notes = [
        "No exercise should provoke or increase pain.",
        "Reduce range or regress if movement quality decreases.",
        "Prioritize calm breathing and control before intensity.",
    ]
    if readiness.get("medical_advice_recommended"):
        notes.insert(0, "Medical advice is recommended if pain persists, increases, or comes with unusual symptoms.")
    return notes


# ---------------------------------------------------------------------
# Example CLI usage
# ---------------------------------------------------------------------

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    library_path = here / "flexilab_exercise_library_v1.json"
    rules_path = here / "prescription_rules_v1.json"

    library = load_exercise_library(library_path)
    rules = load_json(rules_path)

    sample = {
        "report": {
            "flexilab_score": 71.2,
            "score_v2": {
                "domain_scores": [
                    {"id": "thoracic_mobility", "label": "Mobilité thoracique", "label_en": "Thoracic Mobility", "score": 64.2},
                    {"id": "core_stability", "label": "Stabilité du tronc", "label_en": "Core Stability", "score": 64.2},
                    {"id": "cervical_control", "label": "Contrôle cervical", "label_en": "Cervical Control", "score": 70.7},
                    {"id": "shoulder_mobility", "label": "Mobilité des épaules", "label_en": "Shoulder Mobility", "score": 73.6},
                    {"id": "hip_mobility", "label": "Mobilité de hanche", "label_en": "Hip Mobility", "score": 78},
                    {"id": "ankle_mobility", "label": "Mobilité de cheville", "label_en": "Ankle Mobility", "score": 78},
                ]
            },
            "sections": []
        },
        "intake_context": {"pain_score": 0, "red_flags": []}
    }

    program = generate_clinical_prescription(sample, library, rules, language="fr")
    print(json.dumps(program, ensure_ascii=False, indent=2))
