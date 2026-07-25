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
        measured.append(
            _txt(
                lang,
                f"Le profil de posture mesure {_fmt(neck)} au niveau cervical, "
                f"{_fmt(thoracic)} pour le haut du tronc et {_fmt(pelvis)} pour la "
                "relation tronc-bassin.",
                f"The posture profile measures {_fmt(neck)} at the cervical level, "
                f"{_fmt(thoracic)} for the upper trunk, and {_fmt(pelvis)} for the "
                "trunk-pelvis relationship.",
            )
        )
    if shoulder_r is not None or shoulder_l is not None:
        measured.append(
            _txt(
                lang,
                f"La flexion active d’épaule est de {_fmt(shoulder_r)} à droite et "
                f"{_fmt(shoulder_l)} à gauche.",
                f"Active shoulder flexion is {_fmt(shoulder_r)} on the right and "
                f"{_fmt(shoulder_l)} on the left.",
            )
        )
    if knee is not None or trunk is not None:
        measured.append(
            _txt(
                lang,
                f"Au squat, l’angle du genou est de {_fmt(knee)} et l’inclinaison du "
                f"tronc de {_fmt(trunk)}.",
                f"During the squat, knee angle is {_fmt(knee)} and trunk inclination "
                f"is {_fmt(trunk)}.",
            )
        )
    if aslr_r is not None or aslr_l is not None:
        measured.append(
            _txt(
                lang,
                f"L’ASLR est de {_fmt(aslr_r)} à droite et {_fmt(aslr_l)} à gauche.",
                f"ASLR is {_fmt(aslr_r)} on the right and {_fmt(aslr_l)} on the left.",
            )
        )

    priority_domains: List[str] = []
    for metric_id in top_priority_ids:
        domain = _DOMAIN_BY_METRIC.get(metric_id)
        if domain and domain not in priority_domains:
            priority_domains.append(domain)

    domain_labels = [_domain_label(domain, lang) for domain in priority_domains]
    if top_priority_ids:
        summary = _txt(
            lang,
            "Le profil met en évidence des axes de progression clairement hiérarchisés "
            f"autour de {_natural_list(domain_labels, lang)}. Le programme traduira ces "
            "mesures en un travail progressif de mobilité, de coordination et de contrôle "
            "afin d’améliorer la qualité et la reproductibilité des gestes.",
            "The profile highlights clearly ranked development areas centred on "
            f"{_natural_list(domain_labels, lang)}. The program will translate these "
            "measurements into progressive mobility, coordination, and control work to "
            "improve movement quality and repeatability.",
        )
    else:
        summary = _txt(
            lang,
            "Le profil est globalement équilibré, sans priorité dominante. Le programme "
            "visera à consolider les amplitudes disponibles, affiner le contrôle et "
            "soutenir une progression régulière.",
            "The profile is globally balanced, with no dominant priority. The program "
            "will consolidate available range, refine movement control, and support "
            "steady progression.",
        )

    observations = [_observation(metric_id, lang) for metric_id in top_priority_ids]

    correlations: List[str] = []
    if "posture" in priority_domains:
        correlations.append(
            _txt(
                lang,
                "L’alignement observé résulte d’une coordination active entre la tête, "
                "le thorax et le bassin. Le programme travaillera cette organisation dans "
                "des gestes simples avant de l’intégrer à des tâches plus dynamiques.",
                "Observed alignment reflects active coordination between the head, trunk, "
                "and pelvis. The program will develop this organisation in simple movements "
                "before integrating it into more dynamic tasks.",
            )
        )
    if "shoulder" in priority_domains:
        correlations.append(
            _txt(
                lang,
                "L’élévation du bras dépend de l’amplitude de l’épaule, du mouvement de "
                "l’omoplate et de la contribution du haut du tronc. Leur coordination sera "
                "progressée ensemble plutôt que travaillée isolément.",
                "Arm elevation depends on shoulder range, scapular motion, and upper-trunk "
                "contribution. Their coordination will be progressed together rather than "
                "trained in isolation.",
            )
        )
    if "aslr" in priority_domains:
        correlations.append(
            _txt(
                lang,
                "L’ASLR combine mobilité active de hanche, disponibilité de la chaîne "
                "postérieure et stabilité du bassin. La progression cherchera à augmenter "
                "l’amplitude sans perdre le contrôle lombo-pelvien.",
                "ASLR combines active hip mobility, posterior-chain availability, and pelvic "
                "stability. Progression will aim to increase range while maintaining "
                "lumbopelvic control.",
            )
        )
    if "squat" in priority_domains:
        correlations.append(
            _txt(
                lang,
                "La profondeur et l’inclinaison du tronc au squat reflètent une stratégie "
                "globale d’équilibre. Le travail associera mobilité disponible, stabilité "
                "du tronc et coordination des membres inférieurs.",
                "Squat depth and trunk inclination reflect an overall balance strategy. "
                "Training will combine available mobility, trunk stability, and lower-limb "
                "coordination.",
            )
        )
    if not correlations:
        correlations.append(
            _txt(
                lang,
                "Les différentes mesures sont cohérentes entre elles et offrent une base "
                "solide pour consolider le contrôle global et poursuivre la progression.",
                "The measurements are coherent with one another and provide a strong base "
                "for consolidating overall control and continuing progression.",
            )
        )

    focus_text = _natural_list(domain_labels, lang)
    if focus_text:
        primary_objective = _txt(
            lang,
            f"Développer {focus_text} par une progression confortable, précise et "
            "mesurable.",
            f"Develop {focus_text} through comfortable, precise, and measurable progression.",
        )
    else:
        primary_objective = _txt(
            lang,
            "Consolider la qualité globale du mouvement et maintenir les amplitudes "
            "disponibles.",
            "Consolidate overall movement quality and maintain available range.",
        )

    monitor_labels = {
        "posture": ("organisation tête-thorax-bassin", "head-trunk-pelvis organisation"),
        "shoulder": ("aisance au-dessus de la tête", "overhead ease"),
        "aslr": ("contrôle lombo-pelvien pendant l’ASLR", "lumbopelvic control during ASLR"),
        "squat": ("profondeur et contrôle du tronc au squat", "squat depth and trunk control"),
    }
    monitor_values = [
        _txt(lang, *monitor_labels[domain]) for domain in priority_domains
    ]
    monitor = _natural_list(monitor_values, lang) or _txt(
        lang,
        "qualité du mouvement, symétrie et confort",
        "movement quality, symmetry, and comfort",
    )

    plan = {
        "primary_objective": primary_objective,
        "secondary_objective": _txt(
            lang,
            "Renforcer la coordination entre les segments afin de rendre les gains "
            "transférables aux gestes quotidiens et sportifs.",
            "Strengthen coordination between body segments so gains transfer to daily "
            "and sporting movements.",
        ),
        "monitor": _txt(lang, f"Suivre : {monitor}.", f"Track: {monitor}."),
        "reassessment": _txt(
            lang,
            "Répéter les mêmes tests après quatre semaines dans des conditions comparables.",
            "Repeat the same tests after four weeks under comparable conditions.",
        ),
    }

    return {
        "version": "FlexiLab Professional Movement Report v2.1",
        "expert_summary": summary,
        "measured_findings": measured,
        "biomechanical_correlations": correlations,
        # Field retained for API compatibility; content is now commercial movement guidance.
        "ranked_movement_hypotheses": observations,
        "functional_implications": _txt(
            lang,
            "Le programme est organisé pour transformer les priorités mesurées en gains "
            "concrets de mobilité active, de contrôle et d’efficacité dans les gestes "
            "fonctionnels.",
            "The program is structured to convert measured priorities into practical gains "
            "in active mobility, control, and efficiency during functional movement.",
        ),
        # Field retained for API compatibility; content explains scope without defensive wording.
        "not_assessed_and_limitations": [
            _txt(
                lang,
                "FlexiLab analyse les mouvements visibles sur les photos afin de guider "
                "l’entraînement et le suivi de la progression.",
                "FlexiLab analyses visible movement patterns to guide training and progress "
                "tracking.",
            )
        ],
        "reassessment_plan": plan,
        "disclaimer": _txt(
            lang,
            "FlexiLab fournit un screening du mouvement destiné au bien-être et à "
            "l’entraînement. Il ne remplace pas un avis médical.",
            "FlexiLab provides movement screening for wellness and training guidance. It does not replace medical advice.",
        ),
    }


def attach_expert_report(report: Dict[str, Any], lang: str = "fr") -> Dict[str, Any]:
    attach_symmetry_thresholds(report, lang)
    report["expert_report"] = build_expert_report(report, lang)
    return report
