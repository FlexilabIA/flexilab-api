"""
FlexiLab Evidence-Aware Score Engine V3

Principles
----------
- Score only domains supported by the current screening.
- Never assign a positive default score to an unassessed domain.
- Keep measured evidence and inferred support separate.
- Reweight the global score across assessed domains only.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

DOMAIN_CONFIG = {
    "cervical_alignment_control": {
        "weight": 15,
        "label_fr": "Alignement et contrôle cervical",
        "label_en": "Cervical alignment and control",
        "metrics": ["neck_angle"],
        "evidence_level": "direct_screening_measure",
    },
    "upper_trunk_alignment": {
        "weight": 15,
        "label_fr": "Alignement du haut du tronc",
        "label_en": "Upper-trunk alignment",
        "metrics": ["thoracic_angle"],
        "evidence_level": "direct_screening_measure",
    },
    "shoulder_overhead_mobility": {
        "weight": 25,
        "label_fr": "Mobilité active des épaules",
        "label_en": "Active shoulder mobility",
        "metrics": ["shoulder_right_flexion", "shoulder_left_flexion"],
        "evidence_level": "direct_screening_measure",
    },
    "posterior_chain_active_mobility": {
        "weight": 20,
        "label_fr": "Mobilité active de la chaîne postérieure",
        "label_en": "Active posterior-chain mobility",
        "metrics": ["aslr_right_angle", "aslr_left_angle"],
        "evidence_level": "direct_screening_measure",
    },
    "squat_movement_strategy": {
        "weight": 25,
        "label_fr": "Stratégie de mouvement au squat",
        "label_en": "Squat movement strategy",
        "metrics": ["squat_knee_angle", "squat_trunk_lean"],
        "evidence_level": "multi_joint_screening_measure",
    },
}

UNASSESSED_DOMAINS = {
    "ankle_dorsiflexion": {
        "label_fr": "Dorsiflexion de cheville",
        "label_en": "Ankle dorsiflexion",
        "reason_fr": "Aucun test direct de dorsiflexion n’a été réalisé.",
        "reason_en": "No direct dorsiflexion test was performed.",
    },
    "scapular_control": {
        "label_fr": "Contrôle scapulaire",
        "label_en": "Scapular control",
        "reason_fr": "La rotation scapulaire n’a pas été mesurée directement.",
        "reason_en": "Scapular rotation was not measured directly.",
    },
    "thoracic_mobility": {
        "label_fr": "Mobilité thoracique",
        "label_en": "Thoracic mobility",
        "reason_fr": "La vue de profil mesure un alignement statique, pas une amplitude thoracique.",
        "reason_en": "The side view measures static alignment, not thoracic range of motion.",
    },
    "isolated_core_stability": {
        "label_fr": "Stabilité isolée du tronc",
        "label_en": "Isolated trunk stability",
        "reason_fr": "Le squat renseigne sur une stratégie globale et ne constitue pas un test isolé du tronc.",
        "reason_en": "The squat reflects a global strategy and is not an isolated trunk test.",
    },
}

def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(value)))

def get_item(report: Dict[str, Any], item_id: str) -> Optional[Dict[str, Any]]:
    for section in report.get("sections", []) or []:
        for item in section.get("items", []) or []:
            if item.get("id") == item_id:
                return item
    return None

def band_score(item: Dict[str, Any]) -> float:
    rating = str(item.get("rating") or "").lower()
    base = {"green": 88.0, "yellow": 64.0, "orange": 52.0, "red": 36.0}.get(rating, 70.0)
    return clamp(base)

def confidence_from_items(items: List[Dict[str, Any]]) -> str:
    values = []
    for item in items:
        try:
            values.append(float(item.get("confidence")))
        except Exception:
            pass
    if not values:
        return "moderate"
    mean = sum(values) / len(values)
    if mean >= 0.75:
        return "high"
    if mean >= 0.45:
        return "moderate"
    return "low"

def derive_domain_scores(report: Dict[str, Any], lang: str = "fr") -> List[Dict[str, Any]]:
    domains = []
    for domain_id, cfg in DOMAIN_CONFIG.items():
        items = [get_item(report, metric_id) for metric_id in cfg["metrics"]]
        items = [item for item in items if item and item.get("value") is not None]
        if not items:
            continue

        score = round(sum(band_score(item) for item in items) / len(items), 1)
        domains.append({
            "id": domain_id,
            "label_fr": cfg["label_fr"],
            "label_en": cfg["label_en"],
            "label": cfg["label_fr"] if lang == "fr" else cfg["label_en"],
            "score": score,
            "weight": cfg["weight"],
            "assessment_status": "assessed",
            "evidence_level": cfg["evidence_level"],
            "evidence": [
                {
                    "metric_id": item.get("id"),
                    "value": item.get("value"),
                    "unit": item.get("unit"),
                    "rating": item.get("rating"),
                }
                for item in items
            ],
            "confidence": confidence_from_items(items),
        })
    return domains

def movement_quality_label(score: float, lang: str = "fr") -> Dict[str, str]:
    if score >= 85:
        fr, en, color = "Profil global favorable", "Favourable overall profile", "green"
    elif score >= 70:
        fr, en, color = "Axes d’amélioration modérés", "Moderate improvement opportunities", "yellow"
    else:
        fr, en, color = "Priorités correctives identifiées", "Corrective priorities identified", "red"
    return {"label_fr": fr, "label_en": en, "label": fr if lang == "fr" else en, "color": color}

def compute_flexilab_score_v3(report: Dict[str, Any], lang: str = "fr") -> Dict[str, Any]:
    domains = derive_domain_scores(report, lang)
    if not domains:
        return {
            "version": "FlexiLab Evidence-Aware Score Engine V3",
            "score": None,
            "movement_quality": movement_quality_label(0, lang),
            "domain_scores": [],
            "main_areas_to_improve": [],
            "strengths": [],
            "not_assessed_domains": [],
        }

    weight_sum = sum(d["weight"] for d in domains)
    score = round(sum(d["score"] * d["weight"] for d in domains) / weight_sum, 1)
    priorities = [d for d in sorted(domains, key=lambda d: d["score"]) if d["score"] < 75][:3]
    strengths = [d for d in sorted(domains, key=lambda d: d["score"], reverse=True) if d["score"] >= 75][:3]

    not_assessed = [
        {
            "id": did,
            "label_fr": cfg["label_fr"],
            "label_en": cfg["label_en"],
            "label": cfg["label_fr"] if lang == "fr" else cfg["label_en"],
            "assessment_status": "not_assessed",
            "score": None,
            "reason_fr": cfg["reason_fr"],
            "reason_en": cfg["reason_en"],
            "reason": cfg["reason_fr"] if lang == "fr" else cfg["reason_en"],
        }
        for did, cfg in UNASSESSED_DOMAINS.items()
    ]

    return {
        "version": "FlexiLab Evidence-Aware Score Engine V3",
        "score": score,
        "movement_quality": movement_quality_label(score, lang),
        "domain_scores": domains,
        "main_areas_to_improve": priorities,
        "strengths": strengths,
        "not_assessed_domains": not_assessed,
        "scoring_note_fr": "Le score global est recalculé uniquement à partir des domaines effectivement évalués.",
        "scoring_note_en": "The global score is reweighted across assessed domains only.",
    }

def attach_score_v2(report: Dict[str, Any], lang: str = "fr") -> Dict[str, Any]:
    """Compatibility name retained for app.py."""
    if not isinstance(report, dict):
        return report
    result = compute_flexilab_score_v3(report, lang)
    report["flexilab_score_v1"] = report.get("flexilab_score")
    report["flexilab_score"] = result["score"]
    report["score_v2"] = result
    report["movement_quality"] = result["movement_quality"]
    report["risk_category"] = {
        "label": result["movement_quality"]["label"],
        "color": result["movement_quality"]["color"],
        "description_fr": result.get("scoring_note_fr"),
        "description_en": result.get("scoring_note_en"),
        "description": result.get("scoring_note_fr") if lang == "fr" else result.get("scoring_note_en"),
    }
    report["top_priorities_v2"] = result["main_areas_to_improve"]
    report["not_assessed_domains"] = result["not_assessed_domains"]
    return report
