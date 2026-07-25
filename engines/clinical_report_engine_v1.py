"""Expert bilingual clinical and biomechanical report builder."""

from __future__ import annotations
from typing import Any, Dict, List, Optional

SYMMETRY_THRESHOLDS = {
    "shoulders": {
        "unit": "deg",
        "scale_min": 0,
        "scale_max": 20,
        "bands": [
            {"min": 0, "max": 5, "color": "green"},
            {"min": 5, "max": 10, "color": "yellow"},
            {"min": 10, "max": 20, "color": "red"},
        ],
    },
    "aslr": {
        "unit": "deg",
        "scale_min": 0,
        "scale_max": 20,
        "bands": [
            {"min": 0, "max": 5, "color": "green"},
            {"min": 5, "max": 10, "color": "yellow"},
            {"min": 10, "max": 20, "color": "red"},
        ],
    },
}

def _txt(lang: str, fr: str, en: str) -> str:
    return en if str(lang).lower().startswith("en") else fr

def _items(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        item.get("id"): item
        for section in report.get("sections", []) or []
        for item in section.get("items", []) or []
        if item.get("id")
    }

def _value(items, key):
    value = (items.get(key) or {}).get("value")
    try:
        return float(value)
    except Exception:
        return None

def _rating(items, key):
    return str((items.get(key) or {}).get("rating") or "").lower()

def _fmt(value):
    return "—" if value is None else f"{value:.1f}°"

def _symmetry_threshold(section_id: str, difference: float) -> Dict[str, Any]:
    cfg = dict(SYMMETRY_THRESHOLDS[section_id])
    cfg["pointer_value"] = round(max(cfg["scale_min"], min(cfg["scale_max"], difference)), 2)
    cfg["rating"] = "green" if difference <= 5 else "yellow" if difference <= 10 else "red"
    return cfg

def attach_symmetry_thresholds(report: Dict[str, Any], lang: str = "fr") -> Dict[str, Any]:
    for section in report.get("sections", []) or []:
        if section.get("id") not in SYMMETRY_THRESHOLDS:
            continue
        asym = section.get("asymmetry")
        if not asym or asym.get("value_deg") is None:
            continue
        diff = float(asym["value_deg"])
        asym["thresholds"] = _symmetry_threshold(section["id"], diff)
        asym["rating"] = asym["thresholds"]["rating"]
        if diff <= 5:
            asym["short_insight_fr"] = "La différence droite–gauche est faible et compatible avec une bonne symétrie de screening."
            asym["short_insight_en"] = "The side-to-side difference is small and consistent with good screening symmetry."
        elif diff <= 10:
            asym["short_insight_fr"] = "Une différence modérée est présente; elle mérite un travail bilatéral contrôlé et un suivi au re-test."
            asym["short_insight_en"] = "A moderate difference is present; controlled bilateral work and reassessment are appropriate."
        else:
            asym["short_insight_fr"] = "Une asymétrie notable est observée; le côté le plus limité doit être suivi sans conclure à une lésion."
            asym["short_insight_en"] = "A relevant asymmetry is observed; the more limited side should be monitored without inferring injury."
        asym["short_insight"] = asym["short_insight_fr"] if lang == "fr" else asym["short_insight_en"]
    return report

def build_expert_report(report: Dict[str, Any], lang: str = "fr") -> Dict[str, Any]:
    items = _items(report)
    neck = _value(items, "neck_angle")
    thoracic = _value(items, "thoracic_angle")
    shoulder_r = _value(items, "shoulder_right_flexion")
    shoulder_l = _value(items, "shoulder_left_flexion")
    aslr_r = _value(items, "aslr_right_angle")
    aslr_l = _value(items, "aslr_left_angle")
    knee = _value(items, "squat_knee_angle")
    trunk = _value(items, "squat_trunk_lean")

    shoulder_limitation = any(_rating(items, k) in {"yellow","red"} for k in ["shoulder_right_flexion","shoulder_left_flexion"])
    trunk_priority = _rating(items, "squat_trunk_lean") == "red"
    aslr_limitation = any(_rating(items, k) in {"yellow","red"} for k in ["aslr_right_angle","aslr_left_angle"])

    measured = []
    if neck is not None:
        measured.append(_txt(lang,
            f"L’angle cervical mesuré est de {_fmt(neck)}.",
            f"The measured cervical angle is {_fmt(neck)}."))
    if thoracic is not None:
        measured.append(_txt(lang,
            f"L’alignement du haut du tronc en vue de profil est de {_fmt(thoracic)}.",
            f"Upper-trunk alignment in the side view is {_fmt(thoracic)}."))
    if shoulder_r is not None or shoulder_l is not None:
        measured.append(_txt(lang,
            f"La flexion active d’épaule est de {_fmt(shoulder_r)} à droite et {_fmt(shoulder_l)} à gauche.",
            f"Active shoulder flexion is {_fmt(shoulder_r)} on the right and {_fmt(shoulder_l)} on the left."))
    if knee is not None or trunk is not None:
        measured.append(_txt(lang,
            f"Au squat, l’angle du genou est de {_fmt(knee)} et l’inclinaison du tronc de {_fmt(trunk)}.",
            f"During the squat, knee angle is {_fmt(knee)} and trunk inclination is {_fmt(trunk)}."))
    if aslr_r is not None or aslr_l is not None:
        measured.append(_txt(lang,
            f"L’ASLR est de {_fmt(aslr_r)} à droite et {_fmt(aslr_l)} à gauche.",
            f"ASLR is {_fmt(aslr_r)} on the right and {_fmt(aslr_l)} on the left."))

    correlations = []
    if shoulder_limitation:
        correlations.append(_txt(lang,
            "L’élévation active de l’épaule met en jeu la mobilité de l’épaule, le contrôle de l’omoplate et la contribution du haut du tronc. Le programme ciblera ces composantes de façon progressive.",
            "Active shoulder elevation combines shoulder mobility, scapular control, and upper-trunk contribution. The program will address these components progressively."))
    if trunk_priority:
        correlations.append(_txt(lang,
            "L’inclinaison du tronc au squat reflète la stratégie utilisée pour conserver l’équilibre et la profondeur. Le travail privilégiera le contrôle du tronc, la mobilité disponible et la qualité du mouvement.",
            "Squat trunk inclination reflects the strategy used to maintain balance and depth. Training will focus on trunk control, available mobility, and movement quality."))
    if aslr_limitation:
        correlations.append(_txt(lang,
            "Une ASLR réduite indique une priorité de progression pour la mobilité active de hanche, la chaîne postérieure et le contrôle du bassin.",
            "A reduced ASLR indicates a progression priority for active hip mobility, posterior-chain mobility, and pelvic control."))

    if shoulder_limitation and trunk_priority:
        summary = _txt(lang,
            "Le profil met principalement en évidence une limitation de la mobilité active du quadrant supérieur associée à une stratégie de squat avec compensation du tronc. La priorité est d’améliorer le contrôle actif et la qualité du mouvement plutôt que de corriger une supposée déformation posturale.",
            "The profile primarily shows reduced active upper-quarter mobility together with a squat strategy involving trunk compensation. The priority is to improve active control and movement quality rather than correct an assumed postural deformity.")
    elif shoulder_limitation:
        summary = _txt(lang,
            "Le principal axe d’amélioration concerne la mobilité active et le contrôle au-dessus de la tête. Le programme associera mobilité, contrôle scapulaire et progression de l’élévation.",
            "The main improvement area is active overhead mobility and control. The program will combine mobility, scapular control, and progressive elevation.")
    elif trunk_priority:
        summary = _txt(lang,
            "Le principal axe d’amélioration concerne la stratégie de mouvement au squat, notamment le contrôle du tronc sous contrainte fonctionnelle.",
            "The main improvement area concerns squat movement strategy, particularly trunk control under functional demand.")
    else:
        summary = _txt(lang,
            "Les mesures disponibles ne montrent pas de déficit dominant. Le programme peut cibler l’entretien, la qualité du contrôle et une progression graduelle.",
            "Available measurements do not show a dominant deficit. The program can focus on maintenance, movement control and gradual progression.")

    hypotheses = []
    if shoulder_limitation:
        hypotheses.append(_txt(lang,
            "Hypothèse prioritaire : limitation de l’élévation active et/ou coordination thoraco-scapulaire insuffisante.",
            "Primary hypothesis: reduced active elevation and/or insufficient thoracic-scapular coordination."))
    if trunk_priority:
        hypotheses.append(_txt(lang,
            "Priorité secondaire : améliorer la stratégie de squat par un travail de contrôle du tronc, de mobilité et de coordination.",
            "Secondary priority: improve squat strategy through trunk-control, mobility, and coordination work."))

    not_assessed = [
        _txt(
            lang,
            "FlexiLab analyse les mouvements visibles sur les photos afin de guider l’entraînement et le suivi.",
            "FlexiLab analyses visible movement patterns to guide training and progress tracking.",
        )
    ]


    plan = {
        "primary_objective": _txt(lang,
            "Améliorer le contrôle actif des domaines limités sans provoquer de douleur.",
            "Improve active control in limited domains without provoking pain."),
        "secondary_objective": _txt(lang,
            "Réduire les compensations observées lors du squat et des mouvements au-dessus de la tête.",
            "Reduce observed compensations during squat and overhead movement."),
        "monitor": _txt(lang,
            "Symétrie droite-gauche, compensation du tronc, ouverture costale, tension cervicale et réponse douloureuse.",
            "Side-to-side symmetry, trunk compensation, rib flare, neck tension and pain response."),
        "reassessment": _txt(lang,
            "Répéter les mêmes tests après quatre semaines dans des conditions comparables.",
            "Repeat the same tests after four weeks under comparable conditions."),
    }

    return {
        "version": "FlexiLab Expert Clinical Report v1",
        "expert_summary": summary,
        "measured_findings": measured,
        "biomechanical_correlations": correlations,
        "ranked_movement_hypotheses": hypotheses,
        "functional_implications": _txt(lang,
            "Les priorités observées orientent le programme vers une meilleure aisance, un meilleur contrôle et une progression plus régulière dans les gestes fonctionnels.",
            "The observed priorities guide the program toward greater ease, better control, and more consistent progress in functional movements."),
        "not_assessed_and_limitations": not_assessed,
        "reassessment_plan": plan,
        "disclaimer": _txt(lang,
            "FlexiLab fournit un screening du mouvement destiné au bien-être et à l’entraînement. Il ne remplace pas un avis médical.",
            "FlexiLab provides movement screening for wellness and training guidance. It does not replace medical advice."),
    }

def attach_expert_report(report: Dict[str, Any], lang: str = "fr") -> Dict[str, Any]:
    attach_symmetry_thresholds(report, lang)
    report["expert_report"] = build_expert_report(report, lang)
    return report
