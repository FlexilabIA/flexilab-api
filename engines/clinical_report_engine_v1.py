"""Premium bilingual movement-report builder for the trainer experience."""

from __future__ import annotations

from typing import Any, Dict, List


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

_PRIORITY_RATINGS = {"yellow", "red"}

_METRIC_ORDER = [
    "neck_angle",
    "thoracic_angle",
    "pelvic_proxy_angle",
    "shoulder_right_flexion",
    "shoulder_left_flexion",
    "aslr_right_angle",
    "aslr_left_angle",
    "squat_trunk_lean",
    "squat_knee_angle",
]

_DOMAIN_BY_METRIC = {
    "neck_angle": "posture",
    "thoracic_angle": "posture",
    "pelvic_proxy_angle": "posture",
    "shoulder_right_flexion": "shoulder",
    "shoulder_left_flexion": "shoulder",
    "aslr_right_angle": "aslr",
    "aslr_left_angle": "aslr",
    "squat_trunk_lean": "squat",
    "squat_knee_angle": "squat",
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


def _value(items: Dict[str, Dict[str, Any]], key: str) -> float | None:
    value = (items.get(key) or {}).get("value")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rating(items: Dict[str, Dict[str, Any]], key: str) -> str:
    return str((items.get(key) or {}).get("rating") or "").lower()


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}°"


def _symmetry_threshold(section_id: str, difference: float) -> Dict[str, Any]:
    cfg = dict(SYMMETRY_THRESHOLDS[section_id])
    cfg["pointer_value"] = round(
        max(cfg["scale_min"], min(cfg["scale_max"], difference)),
        2,
    )
    cfg["rating"] = (
        "green" if difference <= 5 else "yellow" if difference <= 10 else "red"
    )
    return cfg


def attach_symmetry_thresholds(
    report: Dict[str, Any],
    lang: str = "fr",
) -> Dict[str, Any]:
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
            asym["short_insight_fr"] = (
                "La différence droite-gauche est faible et traduit une symétrie "
                "de screening bien maîtrisée."
            )
            asym["short_insight_en"] = (
                "The side-to-side difference is small and reflects well-balanced "
                "screening symmetry."
            )
        elif diff <= 10:
            asym["short_insight_fr"] = (
                "Une différence modérée est présente. Le programme privilégiera un "
                "travail bilatéral contrôlé et vérifiera son évolution au re-test."
            )
            asym["short_insight_en"] = (
                "A moderate difference is present. The program will use controlled "
                "bilateral work and track its evolution at reassessment."
            )
        else:
            asym["short_insight_fr"] = (
                "Une asymétrie notable est observée. Le programme donnera davantage "
                "d’attention au côté présentant l’amplitude la plus faible et suivra "
                "sa progression au re-test."
            )
            asym["short_insight_en"] = (
                "A relevant asymmetry is present. The program will give additional "
                "attention to the side with the lower range and track its progression "
                "at reassessment."
            )
        asym["short_insight"] = (
            asym["short_insight_fr"] if lang == "fr" else asym["short_insight_en"]
        )
    return report


def _priority_metric_ids(
    report: Dict[str, Any],
    items: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Use the same ranked metrics as the priority cards whenever available."""
    ranked: List[str] = []
    for priority in report.get("top_priorities", []) or []:
        metric_id = priority.get("metric_id")
        if metric_id in items and metric_id not in ranked:
            ranked.append(metric_id)

    # Backward-compatible fallback for reports created before metric_id existed.
    for metric_id in _METRIC_ORDER:
        if _rating(items, metric_id) in _PRIORITY_RATINGS and metric_id not in ranked:
            ranked.append(metric_id)
    return ranked


def _observation(metric_id: str, lang: str) -> str:
    observations = {
        "neck_angle": (
            "L’alignement cervical constitue un axe de progression. Le programme "
            "renforcera le contrôle de la tête et du haut du tronc dans des positions "
            "fonctionnelles.",
            "Cervical alignment is a development focus. The program will reinforce "
            "head and upper-trunk control in functional positions.",
        ),
        "thoracic_angle": (
            "L’organisation du haut du tronc mérite une attention spécifique. Le "
            "travail associera mobilité thoracique, respiration et contrôle actif.",
            "Upper-trunk organisation deserves specific attention. Training will "
            "combine thoracic mobility, breathing, and active control.",
        ),
        "pelvic_proxy_angle": (
            "La coordination entre le tronc et le bassin est un axe prioritaire. Le "
            "programme cherchera une position plus stable et plus facilement "
            "reproductible.",
            "Trunk-pelvis coordination is a priority area. The program will develop a "
            "more stable and repeatable movement position.",
        ),
        "shoulder_right_flexion": (
            "L’élévation active de l’épaule droite est un axe de progression. Le "
            "programme combinera amplitude, contrôle scapulaire et coordination du "
            "haut du tronc.",
            "Active right-shoulder elevation is a development focus. The program will "
            "combine range, scapular control, and upper-trunk coordination.",
        ),
        "shoulder_left_flexion": (
            "L’élévation active de l’épaule gauche est un axe de progression. Le "
            "programme combinera amplitude, contrôle scapulaire et coordination du "
            "haut du tronc.",
            "Active left-shoulder elevation is a development focus. The program will "
            "combine range, scapular control, and upper-trunk coordination.",
        ),
        "aslr_right_angle": (
            "L’élévation active de la jambe droite est un axe prioritaire. Le travail "
            "associera mobilité de hanche, aisance de la chaîne postérieure et contrôle "
            "du bassin.",
            "Active right-leg raise is a priority area. Training will combine hip "
            "mobility, posterior-chain ease, and pelvic control.",
        ),
        "aslr_left_angle": (
            "L’élévation active de la jambe gauche est un axe prioritaire. Le travail "
            "associera mobilité de hanche, aisance de la chaîne postérieure et contrôle "
            "du bassin.",
            "Active left-leg raise is a priority area. Training will combine hip "
            "mobility, posterior-chain ease, and pelvic control.",
        ),
        "squat_trunk_lean": (
            "Le contrôle du tronc pendant le squat est un axe prioritaire. Le programme "
            "développera stabilité, coordination et gestion progressive de la profondeur.",
            "Trunk control during the squat is a priority area. The program will develop "
            "stability, coordination, and progressive depth management.",
        ),
        "squat_knee_angle": (
            "La profondeur du squat constitue un axe de progression. Le programme "
            "associera mobilité disponible, contrôle du bassin et trajectoire des membres "
            "inférieurs.",
            "Squat depth is a development focus. The program will combine available "
            "mobility, pelvic control, and lower-limb movement trajectory.",
        ),
    }
    fr, en = observations.get(
        metric_id,
        (
            "Cette mesure constitue un axe de progression du programme.",
            "This measurement is a development focus for the program.",
        ),
    )
    return _txt(lang, fr, en)


def _domain_label(domain: str, lang: str) -> str:
    labels = {
        "posture": ("l’alignement actif", "active alignment"),
        "shoulder": ("la mobilité active des épaules", "active shoulder mobility"),
        "aslr": ("la mobilité active de hanche", "active hip mobility"),
        "squat": ("la stratégie de squat", "squat strategy"),
    }
    fr, en = labels[domain]
    return _txt(lang, fr, en)


def _natural_list(values: List[str], lang: str) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    conjunction = " et " if not str(lang).lower().startswith("en") else " and "
    if len(values) == 2:
        return conjunction.join(values)
    return ", ".join(values[:-1]) + conjunction + values[-1]


def _needs_attention(items: Dict[str, Dict[str, Any]], metric_id: str) -> bool:
    return _rating(items, metric_id) in _PRIORITY_RATINGS


def _bilateral_difference(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs(a - b)


def _whole_profile_narrative(
    items: Dict[str, Dict[str, Any]],
    lang: str,
) -> List[str]:
    """Create 2–3 integrated observations covering the whole measured profile.

    Bilateral measures are narrated together and symmetry is interpreted rather than
    repeating one sentence per side. Priority ranking still guides emphasis, but every
    assessed yellow/red domain can contribute to the narrative.
    """
    neck = _value(items, "neck_angle")
    thoracic = _value(items, "thoracic_angle")
    pelvis = _value(items, "pelvic_proxy_angle")
    shoulder_r = _value(items, "shoulder_right_flexion")
    shoulder_l = _value(items, "shoulder_left_flexion")
    aslr_r = _value(items, "aslr_right_angle")
    aslr_l = _value(items, "aslr_left_angle")
    knee = _value(items, "squat_knee_angle")
    trunk = _value(items, "squat_trunk_lean")

    bullets: List[str] = []

    posture_attention = any(
        _needs_attention(items, metric_id)
        for metric_id in ("neck_angle", "thoracic_angle", "pelvic_proxy_angle")
    )
    shoulder_attention = any(
        _needs_attention(items, metric_id)
        for metric_id in ("shoulder_right_flexion", "shoulder_left_flexion")
    )
    aslr_attention = any(
        _needs_attention(items, metric_id)
        for metric_id in ("aslr_right_angle", "aslr_left_angle")
    )
    squat_attention = any(
        _needs_attention(items, metric_id)
        for metric_id in ("squat_trunk_lean", "squat_knee_angle")
    )

    # Upper-quarter narrative: combine cervical, thoracic and bilateral shoulder findings.
    if posture_attention or shoulder_attention:
        shoulder_diff = _bilateral_difference(shoulder_r, shoulder_l)
        symmetry_clause = ""
        if shoulder_diff is not None:
            if shoulder_diff <= 5:
                symmetry_clause = _txt(
                    lang,
                    " Les amplitudes d’épaule restent toutefois bien équilibrées entre les deux côtés.",
                    " Shoulder range remains well balanced between sides.",
                )
            elif shoulder_diff <= 10:
                symmetry_clause = _txt(
                    lang,
                    " Une différence droite-gauche modérée suggère qu’un côté peut davantage solliciter le thorax ou l’omoplate pour compléter l’élévation du bras.",
                    " A moderate side-to-side difference suggests that one side may rely more on trunk or scapular contribution to complete arm elevation.",
                )
            else:
                symmetry_clause = _txt(
                    lang,
                    " L’écart droite-gauche est notable et peut favoriser des stratégies compensatoires répétées lors des gestes au-dessus de la tête.",
                    " The side-to-side gap is notable and may encourage repeated compensatory strategies during overhead movement.",
                )
        bullets.append(
            _txt(
                lang,
                "Le haut du corps montre une combinaison d’alignement cervical perfectible, de mobilité thoracique limitée et d’élévation active des épaules encore incomplète. Ces éléments sont liés : lorsque le thorax s’étend ou tourne moins facilement, le cou et les omoplates peuvent participer davantage pour amener les bras au-dessus de la tête.",
                "The upper body shows a combination of cervical alignment that can improve, limited thoracic mobility, and active shoulder elevation that is not yet fully available. These findings are connected: when the upper trunk extends or rotates less freely, the neck and shoulder blades may contribute more to bring the arms overhead.",
            ) + symmetry_clause
        )

    # Lower-quarter narrative: combine bilateral ASLR, posterior chain and pelvic control.
    if aslr_attention:
        aslr_diff = _bilateral_difference(aslr_r, aslr_l)
        symmetry_clause = ""
        if aslr_diff is not None:
            if aslr_diff <= 5:
                symmetry_clause = _txt(
                    lang,
                    " Les deux côtés sont proches, ce qui oriente surtout le travail vers une limitation globale d’amplitude plutôt que vers une asymétrie dominante.",
                    " The two sides are close, indicating a mainly global range limitation rather than a dominant asymmetry.",
                )
            elif aslr_diff <= 10:
                lower_side = "droite" if (aslr_r or 0) < (aslr_l or 0) else "gauche"
                lower_side_en = "right" if (aslr_r or 0) < (aslr_l or 0) else "left"
                symmetry_clause = _txt(
                    lang,
                    f" La différence modérée, avec une amplitude plus faible à {lower_side}, peut entraîner une organisation légèrement différente du bassin entre les deux côtés et mérite d’être suivie au re-test.",
                    f" The moderate difference, with lower range on the {lower_side_en}, may lead to a slightly different pelvic strategy between sides and should be tracked at reassessment.",
                )
            else:
                lower_side = "droite" if (aslr_r or 0) < (aslr_l or 0) else "gauche"
                lower_side_en = "right" if (aslr_r or 0) < (aslr_l or 0) else "left"
                symmetry_clause = _txt(
                    lang,
                    f" L’écart est marqué au détriment du côté {lower_side}; cette asymétrie peut augmenter le recours à des compensations du bassin ou du tronc lors des mouvements unilatéraux.",
                    f" The gap is marked on the {lower_side_en}; this asymmetry may increase reliance on pelvic or trunk compensation during unilateral movement.",
                )
        bullets.append(
            _txt(
                lang,
                "L’élévation active des jambes met en évidence une disponibilité réduite de la chaîne postérieure et de la mobilité active de hanche. L’objectif n’est pas seulement de gagner de l’amplitude, mais de le faire sans bascule du bassin ni perte de contrôle lombo-pelvien.",
                "Active leg raise indicates reduced posterior-chain availability and active hip mobility. The goal is not only to gain range, but to do so without pelvic rotation or loss of lumbopelvic control.",
            ) + symmetry_clause
        )

    # Functional narrative: connect squat depth and trunk strategy to the other findings.
    if squat_attention:
        bullets.append(
            _txt(
                lang,
                "Au squat, la profondeur disponible et l’inclinaison du tronc traduisent une stratégie globale d’équilibre plutôt qu’un défaut isolé. La mobilité des hanches et de la chaîne postérieure, le contrôle du tronc et la coordination des membres inférieurs influencent ensemble la capacité à descendre tout en restant stable.",
                "During the squat, available depth and trunk inclination reflect an overall balance strategy rather than one isolated deficit. Hip and posterior-chain mobility, trunk control, and lower-limb coordination jointly influence the ability to descend while remaining stable.",
            )
        )

    if not bullets:
        bullets.append(
            _txt(
                lang,
                "Le profil est globalement équilibré. Les amplitudes, la symétrie et le contrôle observés constituent une base solide; le programme cherchera surtout à consolider cette qualité et à la rendre plus constante dans des mouvements plus exigeants.",
                "The profile is globally balanced. The observed range, symmetry, and control provide a strong base; the program will mainly consolidate this quality and make it more consistent in more demanding movement.",
            )
        )

    # Keep the report concise: maximum three narrative bullets.
    return bullets[:3]


def _integrated_correlations(
    items: Dict[str, Dict[str, Any]],
    lang: str,
) -> List[str]:
    """Explain relationships and practical meaning without repeating measurements."""
    shoulder_attention = any(
        _needs_attention(items, metric_id)
        for metric_id in ("shoulder_right_flexion", "shoulder_left_flexion")
    )
    posture_attention = any(
        _needs_attention(items, metric_id)
        for metric_id in ("neck_angle", "thoracic_angle", "pelvic_proxy_angle")
    )
    aslr_attention = any(
        _needs_attention(items, metric_id)
        for metric_id in ("aslr_right_angle", "aslr_left_angle")
    )
    squat_attention = any(
        _needs_attention(items, metric_id)
        for metric_id in ("squat_trunk_lean", "squat_knee_angle")
    )

    out: List[str] = []
    if posture_attention or shoulder_attention:
        out.append(
            _txt(
                lang,
                "La posture et la mobilité d’épaule doivent être lues ensemble : une meilleure mobilité thoracique et un meilleur contrôle cervical peuvent réduire le besoin de compenser par la nuque, les côtes ou le bas du dos pendant l’élévation des bras.",
                "Posture and shoulder mobility should be interpreted together: improved thoracic mobility and cervical control may reduce the need to compensate through the neck, ribs, or lower back during arm elevation.",
            )
        )
    if aslr_attention or squat_attention:
        out.append(
            _txt(
                lang,
                "Les résultats de l’ASLR et du squat se complètent. Une chaîne postérieure moins disponible peut limiter la liberté du bassin et modifier l’inclinaison du tronc; le programme associera donc mobilité, contrôle lombo-pelvien et intégration en charge.",
                "The ASLR and squat findings complement one another. Reduced posterior-chain availability can limit pelvic freedom and alter trunk inclination, so the program will combine mobility, lumbopelvic control, and weight-bearing integration.",
            )
        )
    out.append(
        _txt(
            lang,
            "Les différences droite-gauche ne signifient pas automatiquement un problème, mais elles indiquent que le corps peut organiser le même geste différemment selon le côté. Le suivi cherchera à réduire les écarts importants tout en améliorant la qualité globale du mouvement.",
            "Side-to-side differences do not automatically indicate a problem, but they show that the body may organise the same task differently on each side. Follow-up will aim to reduce meaningful gaps while improving overall movement quality.",
        )
    )
    return out[:3]


def build_expert_report(report: Dict[str, Any], lang: str = "fr") -> Dict[str, Any]:
    items = _items(report)
    priority_ids = _priority_metric_ids(report, items)
    top_priority_ids = priority_ids[:3]

    neck = _value(items, "neck_angle")
    thoracic = _value(items, "thoracic_angle")
    pelvis = _value(items, "pelvic_proxy_angle")
    shoulder_r = _value(items, "shoulder_right_flexion")
    shoulder_l = _value(items, "shoulder_left_flexion")
    aslr_r = _value(items, "aslr_right_angle")
    aslr_l = _value(items, "aslr_left_angle")
    knee = _value(items, "squat_knee_angle")
    trunk = _value(items, "squat_trunk_lean")

    measured: List[str] = []
    if neck is not None or thoracic is not None or pelvis is not None:
        measured.append(_txt(lang, f"Posture : cervical {_fmt(neck)}, thoracique {_fmt(thoracic)}, relation tronc-bassin {_fmt(pelvis)}.", f"Posture: cervical {_fmt(neck)}, thoracic {_fmt(thoracic)}, trunk-pelvis relationship {_fmt(pelvis)}."))
    if shoulder_r is not None or shoulder_l is not None:
        measured.append(_txt(lang, f"Épaules : {_fmt(shoulder_r)} à droite et {_fmt(shoulder_l)} à gauche.", f"Shoulders: {_fmt(shoulder_r)} right and {_fmt(shoulder_l)} left."))
    if aslr_r is not None or aslr_l is not None:
        measured.append(_txt(lang, f"ASLR : {_fmt(aslr_r)} à droite et {_fmt(aslr_l)} à gauche.", f"ASLR: {_fmt(aslr_r)} right and {_fmt(aslr_l)} left."))
    if knee is not None or trunk is not None:
        measured.append(_txt(lang, f"Squat : angle du genou {_fmt(knee)}, inclinaison du tronc {_fmt(trunk)}.", f"Squat: knee angle {_fmt(knee)}, trunk inclination {_fmt(trunk)}."))

    priority_domains: List[str] = []
    for metric_id in top_priority_ids:
        domain = _DOMAIN_BY_METRIC.get(metric_id)
        if domain and domain not in priority_domains:
            priority_domains.append(domain)
    domain_labels = [_domain_label(domain, lang) for domain in priority_domains]

    summary = _txt(
        lang,
        "L’analyse ci-dessous relie l’ensemble des mesures afin d’expliquer comment mobilité, posture, symétrie et contrôle peuvent influencer la stratégie de mouvement. Les priorités restent hiérarchisées, mais les autres limitations mesurées sont également intégrées à l’interprétation.",
        "The analysis below connects the full set of measurements to explain how mobility, posture, symmetry, and control may influence movement strategy. Priorities remain ranked, while other measured limitations are also included in the interpretation.",
    )

    observations = _whole_profile_narrative(items, lang)
    correlations = _integrated_correlations(items, lang)

    focus_text = _natural_list(domain_labels, lang)
    primary_objective = _txt(
        lang,
        f"Développer {focus_text} tout en améliorant les limitations associées du profil global." if focus_text else "Consolider la qualité globale du mouvement.",
        f"Develop {focus_text} while improving the associated limitations across the whole profile." if focus_text else "Consolidate overall movement quality.",
    )

    plan = {
        "primary_objective": primary_objective,
        "secondary_objective": _txt(lang, "Améliorer la coordination entre les segments et limiter les compensations inutiles.", "Improve coordination between body segments and reduce unnecessary compensation."),
        "monitor": _txt(lang, "Suivre les amplitudes, la symétrie, le contrôle et le confort lors du re-test.", "Track range, symmetry, control, and comfort at reassessment."),
        "reassessment": _txt(lang, "Répéter les mêmes tests après quatre semaines dans des conditions comparables.", "Repeat the same tests after four weeks under comparable conditions."),
    }

    return {
        "version": "FlexiLab Professional Movement Report v2.2-whole-profile-narrative",
        "expert_summary": summary,
        "measured_findings": measured,
        "biomechanical_correlations": correlations,
        "ranked_movement_hypotheses": observations,
        "functional_implications": _txt(lang, "Le programme transforme ces relations en un parcours progressif de mobilité, de contrôle et d’intégration fonctionnelle.", "The program translates these relationships into a progressive journey of mobility, control, and functional integration."),
        "not_assessed_and_limitations": [_txt(lang, "FlexiLab analyse les mouvements visibles sur les photos afin de guider l’entraînement et le suivi de la progression.", "FlexiLab analyses visible movement patterns to guide training and progress tracking.")],
        "reassessment_plan": plan,
        "disclaimer": _txt(lang, "FlexiLab fournit un screening du mouvement destiné au bien-être et à l’entraînement. Il ne remplace pas un avis médical.", "FlexiLab provides movement screening for wellness and training guidance. It does not replace medical advice."),
    }


def attach_expert_report(report: Dict[str, Any], lang: str = "fr") -> Dict[str, Any]:
    attach_symmetry_thresholds(report, lang)
    report["expert_report"] = build_expert_report(report, lang)
    return report
