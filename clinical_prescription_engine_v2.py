"""
FlexiLab Clinical Prescription Engine v2.0

Main upgrade vs v1:
- Builds programs from clinical session blocks, not random exercise lists.
- Uses Movement DNA / clinical patterns as strategy input.
- Maintains foundational drills while adding progressions.
- Limits repetition and avoids sessions dominated by one category.
- Separates reset, mobility, motor control, stability, integration, and recovery.
- Keeps pain as a readiness gatekeeper, not a movement-score penalty.

Primary function:
generate_clinical_prescription_v2(screening_payload, exercise_library, rules, movement_dna=None, language="fr")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import json
from pathlib import Path


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


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _csv(value: str) -> List[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


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
    pain_score = _to_float(intake.get("pain_score", intake.get("pain_level", intake.get("pain", 0))), 0.0)
    red_flags = intake.get("red_flags", []) or []
    pain_region = intake.get("pain_region", intake.get("pain_area", ""))

    if red_flags or pain_score >= 7:
        return {
            "pain_score": pain_score,
            "pain_region": pain_region,
            "red_flags": red_flags,
            "readiness": "medical_clearance_recommended",
            "program_mode": "recovery_only",
            "label": "Avis médical recommandé" if language == "fr" else "Medical clearance recommended",
            "advice": "Privilégier uniquement les mouvements doux sans douleur et demander un avis médical." if language == "fr" else "Use gentle pain-free movement only and seek medical advice.",
            "medical_advice_recommended": True,
        }
    if pain_score >= 4:
        return {
            "pain_score": pain_score,
            "pain_region": pain_region,
            "red_flags": red_flags,
            "readiness": "limited",
            "program_mode": "recovery_control",
            "label": "Douleur modérée" if language == "fr" else "Moderate pain",
            "advice": "Réduire l’amplitude, éviter les exercices provocateurs et demander conseil si la douleur persiste." if language == "fr" else "Reduce range, avoid provocative exercises, and seek advice if pain persists.",
            "medical_advice_recommended": True,
        }
    if pain_score >= 1:
        return {
            "pain_score": pain_score,
            "pain_region": pain_region,
            "red_flags": red_flags,
            "readiness": "caution",
            "program_mode": "pain_free_corrective",
            "label": "Douleur légère" if language == "fr" else "Mild pain",
            "advice": "Travailler uniquement dans une amplitude confortable et sans aggravation." if language == "fr" else "Work only in a comfortable pain-free range.",
            "medical_advice_recommended": False,
        }
    return {
        "pain_score": 0.0,
        "pain_region": pain_region,
        "red_flags": red_flags,
        "readiness": "normal",
        "program_mode": "corrective",
        "label": "Aucune douleur" if language == "fr" else "No pain",
        "advice": "Programme correctif standard, exécution contrôlée et sans douleur." if language == "fr" else "Standard corrective program, controlled and pain-free.",
        "medical_advice_recommended": False,
    }


def extract_domain_scores(screening_payload: Dict[str, Any], language: str = "fr") -> List[Dict[str, Any]]:
    report = screening_payload.get("report", screening_payload) or {}
    score_v2 = report.get("score_v2") or {}
    domains = score_v2.get("domain_scores") or []
    out = []
    for d in domains:
        did = d.get("id") or d.get("domain") or d.get("system")
        if not did:
            continue
        score = _to_float(d.get("score", 0))
        out.append({
            "id": did,
            "label": d.get("label") or d.get("label_fr") or did,
            "label_en": d.get("label_en") or d.get("label") or did,
            "score": score,
            "band": score_band(score, language),
            "weight": d.get("weight", None),
        })
    return sorted(out, key=lambda x: x["score"])


def extract_asymmetries(screening_payload: Dict[str, Any], language: str = "fr") -> List[Dict[str, Any]]:
    report = screening_payload.get("report", screening_payload) or {}
    items = {}
    for sec in report.get("sections", []) or []:
        for it in sec.get("items", []) or []:
            if it.get("id"):
                items[it["id"]] = it

    asym = []
    sr = items.get("shoulder_right_flexion", {}).get("value")
    sl = items.get("shoulder_left_flexion", {}).get("value")
    if sr is not None and sl is not None:
        sr, sl = _to_float(sr), _to_float(sl)
        diff = abs(sr - sl)
        asym.append({
            "id": "shoulder_symmetry",
            "label": "Symétrie mobilité épaules" if language == "fr" else "Shoulder mobility symmetry",
            "left_value": sl,
            "right_value": sr,
            "difference": round(diff, 2),
            "restricted_side": "right" if sr < sl else "left" if sl < sr else "none",
            "significant": diff >= 10,
        })

    ar = items.get("aslr_right_angle", {}).get("value", items.get("aslr_right", {}).get("value"))
    al = items.get("aslr_left_angle", {}).get("value", items.get("aslr_left", {}).get("value"))
    if ar is not None and al is not None:
        ar, al = _to_float(ar), _to_float(al)
        diff = abs(ar - al)
        asym.append({
            "id": "aslr_symmetry",
            "label": "Symétrie ASLR / ischio-jambiers" if language == "fr" else "ASLR / hamstring symmetry",
            "left_value": al,
            "right_value": ar,
            "difference": round(diff, 2),
            "restricted_side": "right" if ar < al else "left" if al < ar else "none",
            "significant": diff >= 10,
        })
    return asym


DOMAIN_TO_CATEGORIES = {
    "cervical_control": ["CC", "TM", "RB"],
    "thoracic_mobility": ["TM", "RB", "SH"],
    "shoulder_mobility": ["SH", "TM", "CS"],
    "core_stability": ["CS", "RB", "FI"],
    "hip_mobility": ["HM", "HS", "CS"],
    "hamstring_mobility": ["HS", "HM"],
    "ankle_mobility": ["AM", "BP", "FI"],
    "functional_integration": ["FI", "CS", "BP"],
    "balance_proprioception": ["BP", "AM", "FI"],
    "recovery_breathing": ["RB"],
}


BLOCKS = {
    "reset": {
        "label_fr": "Reset / respiration",
        "label_en": "Reset / breathing",
        "objectives": ["recovery"],
        "categories": ["RB"],
        "target_count": 1,
    },
    "mobility": {
        "label_fr": "Mobilité prioritaire",
        "label_en": "Priority mobility",
        "objectives": ["mobility", "mobility_activation", "mobility_control", "mobility_stability", "mobility_neural"],
        "categories": ["TM", "SH", "HM", "HS", "AM"],
        "target_count": 2,
    },
    "motor_control": {
        "label_fr": "Contrôle moteur",
        "label_en": "Motor control",
        "objectives": ["motor_control", "activation", "activation_stability"],
        "categories": ["CC", "CS", "SH", "HM"],
        "target_count": 1,
    },
    "stability": {
        "label_fr": "Stabilité",
        "label_en": "Stability",
        "objectives": ["stability", "activation_stability"],
        "categories": ["CS", "SH", "BP"],
        "target_count": 1,
    },
    "integration": {
        "label_fr": "Intégration fonctionnelle",
        "label_en": "Functional integration",
        "objectives": ["integration", "functional_strength", "dynamic_balance"],
        "categories": ["FI", "BP", "CS"],
        "target_count": 1,
    },
    "recovery": {
        "label_fr": "Récupération",
        "label_en": "Recovery",
        "objectives": ["recovery"],
        "categories": ["RB"],
        "target_count": 1,
    },
}


WEEK_BLUEPRINTS = {
    1: {
        "phase_fr": "Restaurer",
        "phase_en": "Restore",
        "difficulty_max": 2,
        "sessions": [
            ["reset", "mobility", "mobility", "motor_control", "recovery"],
            ["reset", "mobility", "motor_control", "stability", "recovery"],
            ["reset", "mobility", "mobility", "motor_control", "integration"],
        ],
    },
    2: {
        "phase_fr": "Contrôler",
        "phase_en": "Control",
        "difficulty_max": 3,
        "sessions": [
            ["reset", "mobility", "mobility", "motor_control", "stability", "recovery"],
            ["reset", "mobility", "motor_control", "stability", "stability", "integration"],
            ["reset", "mobility", "motor_control", "stability", "integration", "recovery"],
        ],
    },
    3: {
        "phase_fr": "Stabiliser",
        "phase_en": "Stabilize",
        "difficulty_max": 4,
        "sessions": [
            ["reset", "mobility", "mobility", "motor_control", "stability", "integration"],
            ["reset", "mobility", "motor_control", "stability", "stability", "integration"],
            ["reset", "mobility", "motor_control", "stability", "integration", "integration"],
        ],
    },
    4: {
        "phase_fr": "Intégrer",
        "phase_en": "Integrate",
        "difficulty_max": 5,
        "sessions": [
            ["reset", "mobility", "motor_control", "stability", "integration", "integration"],
            ["reset", "mobility", "stability", "integration", "integration", "recovery"],
            ["reset", "mobility", "motor_control", "stability", "integration", "recovery"],
        ],
    },
}


def objective_of(ex: Dict[str, Any]) -> str:
    return str(ex.get("primary_objective", "")).lower()


def domains_of(ex: Dict[str, Any]) -> List[str]:
    return _csv(ex.get("screening_domains_improved", ""))


def category_of(ex: Dict[str, Any]) -> str:
    return str(ex.get("category_code", ""))


def is_pain_compatible(ex: Dict[str, Any], readiness: Dict[str, Any]) -> bool:
    mode = readiness.get("program_mode", "corrective")
    obj = objective_of(ex)
    diff = int(_to_float(ex.get("difficulty_1_5", 3), 3))
    if mode == "corrective":
        return True
    if mode == "pain_free_corrective":
        return True
    if mode == "recovery_control":
        return category_of(ex) == "RB" or ("mobility" in obj and diff <= 2) or ("motor_control" in obj and diff <= 2)
    if mode == "recovery_only":
        return category_of(ex) == "RB" or "recovery" in obj
    return True


def infer_strategy(priorities: List[Dict[str, Any]], movement_dna: Optional[Dict[str, Any]], language: str = "fr") -> Dict[str, Any]:
    primary_profile = (movement_dna or {}).get("primary_profile") if movement_dna else None
    matched = (movement_dna or {}).get("matched_patterns", []) if movement_dna else []
    domains = [p["id"] for p in priorities[:4]]

    # Strategy bias from priorities
    category_bias = []
    for d in domains:
        category_bias += DOMAIN_TO_CATEGORIES.get(d, [])

    # Strategy bias from Movement DNA matched patterns can be added later.
    profile_text = primary_profile or ("Profil correctif général" if language == "fr" else "General corrective profile")

    if language == "fr":
        strategy_text = "Programme construit par blocs : reset, mobilité prioritaire, contrôle moteur, stabilité puis intégration fonctionnelle."
    else:
        strategy_text = "Program built by blocks: reset, priority mobility, motor control, stability, then functional integration."

    return {
        "primary_profile": profile_text,
        "matched_patterns": matched,
        "priority_domains": domains,
        "category_bias": list(dict.fromkeys(category_bias)),
        "strategy_text": strategy_text,
    }


def block_label(block: str, language: str) -> str:
    b = BLOCKS.get(block, {})
    return b.get("label_fr" if language == "fr" else "label_en", block)


def exercise_score(
    ex: Dict[str, Any],
    block: str,
    priorities: List[Dict[str, Any]],
    week: int,
    readiness: Dict[str, Any],
    strategy: Dict[str, Any],
    used_ids: set,
    session_category_counts: Dict[str, int],
    global_week_counts: Dict[str, int],
) -> float:
    if ex.get("exercise_id") in used_ids:
        return -9999
    if not is_pain_compatible(ex, readiness):
        return -9999

    diff = int(_to_float(ex.get("difficulty_1_5", 3), 3))
    diff_max = WEEK_BLUEPRINTS[week]["difficulty_max"]
    if diff > diff_max:
        return -999

    cat = category_of(ex)
    obj = objective_of(ex)
    ex_domains = domains_of(ex)
    block_cfg = BLOCKS[block]

    score = 0.0

    # Must loosely match block objective or category.
    objective_match = any(o in obj for o in block_cfg["objectives"])
    category_match = cat in block_cfg["categories"]
    if objective_match:
        score += 25
    if category_match:
        score += 20
    if not objective_match and not category_match:
        return -100

    # Priorities drive relevance.
    for rank, d in enumerate(priorities[:6], start=1):
        domain_id = d["id"]
        deficit = max(0.0, 100.0 - _to_float(d["score"], 100))
        if domain_id in ex_domains:
            score += deficit * ({1: 1.7, 2: 1.35, 3: 1.1}.get(rank, 0.7))

        # Also score by domain-to-category map.
        if cat in DOMAIN_TO_CATEGORIES.get(domain_id, []):
            score += deficit * ({1: 0.75, 2: 0.55, 3: 0.35}.get(rank, 0.2))

    # Strategy category bias
    if cat in strategy.get("category_bias", []):
        score += 8

    # Avoid one category dominating a session.
    if session_category_counts.get(cat, 0) >= 2:
        score -= 25
    if session_category_counts.get(cat, 0) >= 3:
        score -= 80

    # Avoid same exercise repeated across all week sessions.
    if global_week_counts.get(ex.get("exercise_id"), 0) >= 2:
        score -= 20

    # Week progression: don't overuse very easy exercises in weeks 3-4 unless they are reset/foundational mobility.
    if week >= 3 and diff == 1 and block not in ["reset", "mobility", "recovery"]:
        score -= 12

    # Prefer not too hard early.
    if week == 1:
        score += max(0, 3 - diff) * 3
    else:
        score += min(diff, WEEK_BLUEPRINTS[week]["difficulty_max"]) * 1.5

    return score


def select_exercise_for_block(
    library: List[Dict[str, Any]],
    block: str,
    priorities: List[Dict[str, Any]],
    week: int,
    readiness: Dict[str, Any],
    strategy: Dict[str, Any],
    used_ids: set,
    session_category_counts: Dict[str, int],
    global_week_counts: Dict[str, int],
) -> Optional[Dict[str, Any]]:
    scored = []
    for ex in library:
        s = exercise_score(ex, block, priorities, week, readiness, strategy, used_ids, session_category_counts, global_week_counts)
        if s > -900:
            scored.append((s, ex))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else None


def localize_exercise(ex: Dict[str, Any], block: str, language: str) -> Dict[str, Any]:
    return {
        "id": ex.get("exercise_id"),
        "block": block,
        "block_label": block_label(block, language),
        "name": ex.get("name_fr") if language == "fr" else ex.get("name_en"),
        "name_fr": ex.get("name_fr"),
        "name_en": ex.get("name_en"),
        "category_code": ex.get("category_code"),
        "target": ex.get("category_fr") if language == "fr" else ex.get("category_en"),
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
        "clinical_rationale": ex.get("clinical_rationale", ""),
        "regression_id": ex.get("regression_id", ""),
        "progression_id": ex.get("progression_id", ""),
        "pain_rule": ex.get("pain_rule", ""),
        "asymmetry_rule": ex.get("asymmetry_rule", ""),
        "video_url": ex.get("video_url", ""),
        "vimeo_url": ex.get("vimeo_url", ""),
        "mp4_url": ex.get("mp4_url", ""),
        "thumbnail_url": ex.get("thumbnail_url", ""),
    }


def week_objective(week: int, language: str = "fr") -> str:
    fr = {
        1: "Restaurer les amplitudes de base et installer un contrôle sans douleur.",
        2: "Conserver les exercices clés et ajouter du contrôle actif.",
        3: "Stabiliser les amplitudes sous contrainte légère.",
        4: "Intégrer les corrections dans les mouvements fonctionnels.",
    }
    en = {
        1: "Restore basic ranges and establish pain-free control.",
        2: "Keep key drills and add active control.",
        3: "Stabilize ranges under light demand.",
        4: "Integrate corrections into functional movement.",
    }
    return (fr if language == "fr" else en).get(week, "")


def safety_notes(language: str, readiness: Dict[str, Any]) -> List[str]:
    if language == "fr":
        notes = [
            "Aucun exercice ne doit provoquer ou augmenter la douleur.",
            "La qualité du mouvement prime sur le nombre de répétitions.",
            "Réduire l’amplitude si une compensation apparaît.",
        ]
        if readiness.get("medical_advice_recommended"):
            notes.insert(0, "Un avis médical est recommandé si la douleur persiste, augmente ou s’accompagne de symptômes inhabituels.")
        return notes
    notes = [
        "No exercise should provoke or increase pain.",
        "Movement quality is more important than the number of repetitions.",
        "Reduce range if compensation appears.",
    ]
    if readiness.get("medical_advice_recommended"):
        notes.insert(0, "Medical advice is recommended if pain persists, increases, or comes with unusual symptoms.")
    return notes


def estimate_duration(exercises: List[Dict[str, Any]]) -> int:
    return max(12, min(32, 5 + len(exercises) * 3))


def generate_clinical_prescription_v2(
    screening_payload: Dict[str, Any],
    exercise_library: List[Dict[str, Any]],
    rules: Optional[Dict[str, Any]] = None,
    movement_dna: Optional[Dict[str, Any]] = None,
    language: str = "fr",
) -> Dict[str, Any]:
    report = screening_payload.get("report", screening_payload) or {}
    movement_score = _to_float(report.get("flexilab_score", report.get("score", 0)), 0)
    intake = screening_payload.get("intake_context") or report.get("intake_context") or screening_payload.get("intake") or {}

    readiness = pain_readiness_from_intake(intake, language)
    domains = extract_domain_scores(screening_payload, language)

    if not domains:
        domains = [
            {"id": "thoracic_mobility", "label": "Mobilité thoracique", "label_en": "Thoracic Mobility", "score": 70, "band": score_band(70, language), "weight": 20},
            {"id": "hip_mobility", "label": "Mobilité de hanche", "label_en": "Hip Mobility", "score": 70, "band": score_band(70, language), "weight": 20},
            {"id": "core_stability", "label": "Stabilité du tronc", "label_en": "Core Stability", "score": 70, "band": score_band(70, language), "weight": 15},
        ]

    priorities = sorted(domains, key=lambda x: (x["score"], -_to_float(x.get("weight"), 0)))
    asymmetries = extract_asymmetries(screening_payload, language)
    strategy = infer_strategy(priorities, movement_dna, language)

    weeks = []
    for week_no in [1, 2, 3, 4]:
        week_cfg = WEEK_BLUEPRINTS[week_no]
        sessions = []
        global_week_counts = {}

        for day_index, block_sequence in enumerate(week_cfg["sessions"], start=1):
            selected = []
            used_ids = set()
            session_category_counts = {}

            for block in block_sequence:
                ex = select_exercise_for_block(
                    exercise_library,
                    block,
                    priorities,
                    week_no,
                    readiness,
                    strategy,
                    used_ids,
                    session_category_counts,
                    global_week_counts,
                )
                if ex is None:
                    continue

                eid = ex.get("exercise_id")
                cat = category_of(ex)
                selected.append(localize_exercise(ex, block, language))
                used_ids.add(eid)
                session_category_counts[cat] = session_category_counts.get(cat, 0) + 1
                global_week_counts[eid] = global_week_counts.get(eid, 0) + 1

            focus_fr = ["Mobilité / restauration", "Contrôle / stabilité", "Intégration / mouvement"][day_index - 1]
            focus_en = ["Mobility / restore", "Control / stability", "Integration / movement"][day_index - 1]

            sessions.append({
                "day": day_index,
                "focus": focus_fr if language == "fr" else focus_en,
                "session_model": "block_based",
                "blocks": block_sequence,
                "estimated_duration_minutes": estimate_duration(selected),
                "exercises": selected,
            })

        weeks.append({
            "week": week_no,
            "phase": week_cfg["phase_fr"] if language == "fr" else week_cfg["phase_en"],
            "objective": week_objective(week_no, language),
            "difficulty_max": week_cfg["difficulty_max"],
            "progression_logic": {
                "fr": {
                    1: "Installer les bases et identifier les exercices clés à maintenir.",
                    2: "Maintenir les exercices prioritaires et ajouter contrôle/stabilité.",
                    3: "Remplacer une partie des exercices faciles par des exercices plus intégrés.",
                    4: "Conserver l’essentiel et préparer le re-test.",
                },
                "en": {
                    1: "Build foundations and identify key drills to maintain.",
                    2: "Maintain priority drills and add control/stability.",
                    3: "Replace some easy drills with more integrated exercises.",
                    4: "Keep essentials and prepare reassessment.",
                },
            }["fr" if language == "fr" else "en"][week_no],
            "sessions": sessions,
        })

    return {
        "engine_version": "FlexiLab Clinical Prescription Engine v2.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "language": language,
        "movement_score": movement_score,
        "movement_score_band": score_band(movement_score, language),
        "clinical_readiness": readiness,
        "movement_dna_summary": movement_dna or {},
        "clinical_strategy": strategy,
        "main_priorities": priorities[:3],
        "monitor_domains": priorities[3:],
        "asymmetries": asymmetries,
        "program_summary": {
            "duration": "4 semaines" if language == "fr" else "4 weeks",
            "frequency": "3 séances/semaine" if language == "fr" else "3 sessions/week",
            "session_duration": "15-30 min",
            "model": "block-based adaptive maintenance + replacement",
            "medical_advice_recommended": readiness.get("medical_advice_recommended", False),
        },
        "weeks": weeks,
        "safety_notes": safety_notes(language, readiness),
        "reassessment_plan": {
            "when": "après 4 semaines" if language == "fr" else "after 4 weeks",
            "what": "Refaire le même screening et comparer les domaines, la symétrie et la douleur." if language == "fr" else "Repeat the same screening and compare domains, symmetry, and pain.",
        },
        "integration_notes": {
            "replace_v1_import": "from engines.clinical_prescription_engine_v2 import generate_clinical_prescription_v2",
            "frontend": "Render weeks > sessions > block-based exercises. Each exercise can open a Vimeo/MP4 modal later.",
        },
    }


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    library = load_exercise_library(here / "flexilab_exercise_library_v1.json")
    rules = load_json(here / "prescription_rules_v1.json")
    sample = {
        "report": {
            "flexilab_score": 71.2,
            "score_v2": {
                "domain_scores": [
                    {"id": "thoracic_mobility", "label": "Mobilité thoracique", "label_en": "Thoracic Mobility", "score": 64.2, "weight": 20},
                    {"id": "core_stability", "label": "Stabilité du tronc", "label_en": "Core Stability", "score": 64.2, "weight": 15},
                    {"id": "cervical_control", "label": "Contrôle cervical", "label_en": "Cervical Control", "score": 70.7, "weight": 15},
                    {"id": "shoulder_mobility", "label": "Mobilité des épaules", "label_en": "Shoulder Mobility", "score": 73.6, "weight": 20},
                    {"id": "hip_mobility", "label": "Mobilité de hanche", "label_en": "Hip Mobility", "score": 78, "weight": 20},
                    {"id": "ankle_mobility", "label": "Mobilité de cheville", "label_en": "Ankle Mobility", "score": 78, "weight": 10},
                ]
            },
            "sections": []
        },
        "intake_context": {"pain_score": 0, "red_flags": []}
    }
    print(json.dumps(generate_clinical_prescription_v2(sample, library, rules, language="fr"), ensure_ascii=False, indent=2))
