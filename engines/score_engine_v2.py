
"""
FlexiLab Score Engine V2

Goal
----
Replace the old "deduct from 100" score with a movement-quality score.

The final score is built from 6 domains:
- cervical_control: 15%
- thoracic_mobility: 20%
- shoulder_mobility: 20%
- hip_mobility: 20%
- core_stability: 15%
- ankle_mobility: 10%

This avoids unrealistic 90+ scores when several movement areas need improvement.
"""

from __future__ import annotations
from typing import Dict, Any, List


DOMAIN_WEIGHTS = {
    "cervical_control": 15,
    "thoracic_mobility": 20,
    "shoulder_mobility": 20,
    "hip_mobility": 20,
    "core_stability": 15,
    "ankle_mobility": 10,
}

DOMAIN_LABELS = {
    "cervical_control": {"fr": "Contrôle cervical", "en": "Cervical Control"},
    "thoracic_mobility": {"fr": "Mobilité thoracique", "en": "Thoracic Mobility"},
    "shoulder_mobility": {"fr": "Mobilité des épaules", "en": "Shoulder Mobility"},
    "hip_mobility": {"fr": "Mobilité de hanche", "en": "Hip Mobility"},
    "core_stability": {"fr": "Stabilité du tronc", "en": "Core Stability"},
    "ankle_mobility": {"fr": "Mobilité de cheville", "en": "Ankle Mobility"},
}


def clamp(x: float, lo: float = 0, hi: float = 100) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo


def score_from_band(rating: str) -> float:
    """
    Base score from traffic-light rating.
    Green is not always 100 because 'green' may still contain improvement potential.
    """
    rating = str(rating or "").lower()
    if rating == "green":
        return 85
    if rating == "yellow":
        return 62
    if rating == "red":
        return 38
    return 70


def get_item(report: Dict[str, Any], item_id: str) -> Dict[str, Any] | None:
    for sec in report.get("sections", []) or []:
        for item in sec.get("items", []) or []:
            if item.get("id") == item_id:
                return item
    return None


def score_metric(report: Dict[str, Any], item_id: str, default: float = 70) -> float:
    item = get_item(report, item_id)
    if not item:
        return default

    base = score_from_band(item.get("rating"))

    # Fine adjustment inside rating band using pointer when available.
    th = item.get("thresholds", {}) or {}
    scale_min = th.get("scale_min")
    scale_max = th.get("scale_max")
    pointer = th.get("pointer_value", item.get("value"))
    rating = str(item.get("rating", "")).lower()

    try:
        if scale_min is not None and scale_max is not None and scale_max != scale_min:
            normalized = (float(pointer) - float(scale_min)) / (float(scale_max) - float(scale_min))
            normalized = clamp(normalized, 0, 1)
            # For metrics where higher is better, rating bands usually have green at high values.
            higher_is_better = item_id in [
                "shoulder_right_flexion",
                "shoulder_left_flexion",
                "aslr_right_angle",
                "aslr_left_angle",
                "aslr_angle",
            ]
            if higher_is_better:
                fine = normalized
            else:
                fine = 1 - normalized

            if rating == "green":
                base = 82 + 18 * fine
            elif rating == "yellow":
                base = 52 + 23 * fine
            elif rating == "red":
                base = 18 + 27 * fine
    except Exception:
        pass

    return clamp(base)


def avg(values: List[float], default: float = 70) -> float:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else default


def derive_domain_scores(report: Dict[str, Any]) -> Dict[str, float]:
    """
    Build domain scores from available screening metrics.
    Missing domains receive a conservative default score.
    """
    neck = score_metric(report, "neck_angle", 70)
    thoracic_posture = score_metric(report, "thoracic_angle", 70)
    pelvis = score_metric(report, "pelvic_proxy_angle", 70)
    sh_r = score_metric(report, "shoulder_right_flexion", 70)
    sh_l = score_metric(report, "shoulder_left_flexion", 70)
    squat_knee = score_metric(report, "squat_knee_angle", 70)
    squat_trunk = score_metric(report, "squat_trunk_lean", 70)
    aslr_r = score_metric(report, "aslr_right_angle", None)
    aslr_l = score_metric(report, "aslr_left_angle", None)

    # Shoulder asymmetry penalty
    shoulder_asym_penalty = 0
    try:
        for sec in report.get("sections", []) or []:
            if sec.get("id") == "shoulders":
                asym = sec.get("asymmetry", {}) or {}
                if asym.get("rating") == "yellow":
                    shoulder_asym_penalty = 8
                elif asym.get("rating") == "red":
                    shoulder_asym_penalty = 18
    except Exception:
        pass

    shoulder = clamp(avg([sh_r, sh_l], 70) - shoulder_asym_penalty)

    # ASLR contributes strongly to hip/posterior chain when available.
    aslr = avg([aslr_r, aslr_l], None) if (aslr_r is not None or aslr_l is not None) else None

    domains = {
        "cervical_control": neck,
        "thoracic_mobility": avg([thoracic_posture, squat_trunk], 70),
        "shoulder_mobility": shoulder,
        "hip_mobility": avg([squat_knee, aslr if aslr is not None else 65], 65),
        "core_stability": avg([squat_trunk, pelvis], 70),
        "ankle_mobility": avg([squat_knee], 70),
    }

    # If several domains are flagged as priorities, prevent inflated score.
    priority_count = 0
    for sec in report.get("sections", []) or []:
        for item in sec.get("items", []) or []:
            if item.get("rating") in ["yellow", "red"]:
                priority_count += 1

    if priority_count >= 4:
        for k in domains:
            domains[k] = min(domains[k], 78)
    if priority_count >= 6:
        for k in domains:
            domains[k] = min(domains[k], 72)

    return {k: round(clamp(v), 1) for k, v in domains.items()}


def movement_quality_label(score: float, lang: str = "fr") -> Dict[str, str]:
    s = float(score)
    if s >= 90:
        key, fr, en, color = "excellent", "Excellent", "Excellent Movement", "green"
    elif s >= 80:
        key, fr, en, color = "good", "Bon", "Good Movement", "green"
    elif s >= 70:
        key, fr, en, color = "fair", "Correct", "Fair Movement", "yellow"
    elif s >= 60:
        key, fr, en, color = "needs_improvement", "À améliorer", "Needs Improvement", "orange"
    elif s >= 40:
        key, fr, en, color = "limited", "Mobilité limitée", "Limited Mobility", "red"
    else:
        key, fr, en, color = "priority_program", "Programme prioritaire", "Corrective Program Recommended", "red"
    return {"key": key, "label_fr": fr, "label_en": en, "label": fr if lang == "fr" else en, "color": color}


def compute_flexilab_score_v2(report: Dict[str, Any], lang: str = "fr") -> Dict[str, Any]:
    domains = derive_domain_scores(report)
    weighted = 0
    for domain, weight in DOMAIN_WEIGHTS.items():
        weighted += domains.get(domain, 70) * weight / 100

    score = round(clamp(weighted), 1)
    label = movement_quality_label(score, lang=lang)

    domain_list = []
    for k, v in domains.items():
        domain_list.append({
            "id": k,
            "label_fr": DOMAIN_LABELS[k]["fr"],
            "label_en": DOMAIN_LABELS[k]["en"],
            "label": DOMAIN_LABELS[k]["fr"] if lang == "fr" else DOMAIN_LABELS[k]["en"],
            "score": v,
            "weight": DOMAIN_WEIGHTS[k],
            "status": movement_quality_label(v, lang=lang)["key"],
            "color": movement_quality_label(v, lang=lang)["color"],
        })

    priorities = sorted(domain_list, key=lambda x: x["score"])[:6]
    strengths = sorted(domain_list, key=lambda x: x["score"], reverse=True)[:3]

    improvement_potential = round(clamp(100 - score), 1)

    return {
        "version": "FlexiLab Score Engine V2",
        "score": score,
        "movement_quality": label,
        "improvement_potential": improvement_potential,
        "domain_scores": domain_list,
        "main_areas_to_improve": priorities,
        "strengths": strengths,
        "weights": DOMAIN_WEIGHTS,
    }


def attach_score_v2(report: Dict[str, Any], lang: str = "fr") -> Dict[str, Any]:
    """
    Mutates and returns report with V2 score fields.
    Keeps old score under flexilab_score_v1 for backward compatibility.
    """
    if not isinstance(report, dict):
        return report

    score_v2 = compute_flexilab_score_v2(report, lang=lang)
    old_score = report.get("flexilab_score")
    report["flexilab_score_v1"] = old_score
    report["flexilab_score"] = score_v2["score"]
    report["score_v2"] = score_v2
    report["risk_category"] = {
        "label": score_v2["movement_quality"]["label"],
        "color": score_v2["movement_quality"]["color"],
        "description_fr": "Score basé sur la qualité globale du mouvement et les domaines prioritaires.",
        "description_en": "Score based on overall movement quality and priority domains.",
        "description": "Score basé sur la qualité globale du mouvement et les domaines prioritaires." if lang == "fr" else "Score based on overall movement quality and priority domains.",
    }
    report["top_priorities_v2"] = score_v2["main_areas_to_improve"]
    return report
