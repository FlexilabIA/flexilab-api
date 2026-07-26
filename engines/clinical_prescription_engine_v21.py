from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import json, math, re, unicodedata

ENGINE_VERSION = "FlexiLab Clinical Prescription Engine v3.6-direct-priority-quota-symptom-aware-recovery-compat"

# Direct movement families are the only exercises allowed to satisfy minimum
# priority quotas. Supporting and integration exercises remain available for
# variety, but they cannot replace the required direct stimulus.
DIRECT_PRIORITY_CATS = {
    "cervical_control": {"CC"},
    "thoracic_mobility": {"TM"},
    "shoulder_mobility": {"SH"},
    "scapular_control": {"SH"},
    "core_stability": {"CS"},
    "trunk_core_control": {"CS"},
    "hip_mobility": {"HM"},
    "hamstring_mobility": {"HS"},
    "aslr": {"HS"},
    "ankle_mobility": {"AM"},
    "squat_pattern": {"FI"},
    "functional_integration": {"FI"},
    "balance_proprioception": {"BP"},
}

# Filmed demo exercises that can be used as area-specific recovery support.
# These mappings target surrounding soft tissue; they never instruct the user
# to roll directly over a joint, bone, or the spine.
RECOVERY_AREA_EXERCISE_IDS = {
    "neck": ["DMCC002"],
    "shoulders": ["DMSH008", "DMSH009"],
    "upper_back": ["DMTM004", "DMSH009", "DMTM005"],
    "lower_back": ["DMRB002"],
    "hips": ["DMRB003", "DMRB004"],
    "hamstrings": ["DMRB005"],
    "knees": ["DMRB007"],
    "ankles_feet": ["DMRB006"],
}

AREA_ALIASES = {
    "neck": "neck", "nuque": "neck", "cervical": "neck",
    "shoulder": "shoulders", "shoulders": "shoulders", "epaule": "shoulders",
    "épaules": "shoulders", "epaules": "shoulders",
    "upper_back": "upper_back", "upper back": "upper_back", "haut_du_dos": "upper_back",
    "haut du dos": "upper_back", "thoracic": "upper_back",
    "lower_back": "lower_back", "lower back": "lower_back", "bas_du_dos": "lower_back",
    "bas du dos": "lower_back", "lumbar": "lower_back",
    "hip": "hips", "hips": "hips", "hanche": "hips", "hanches": "hips",
    "hamstring": "hamstrings", "hamstrings": "hamstrings",
    "ischio-jambiers": "hamstrings", "ischios": "hamstrings",
    "knee": "knees", "knees": "knees", "genou": "knees", "genoux": "knees",
    "ankle": "ankles_feet", "ankles": "ankles_feet", "foot": "ankles_feet",
    "feet": "ankles_feet", "ankles_feet": "ankles_feet",
    "chevilles": "ankles_feet", "pieds": "ankles_feet",
}

MOVEMENT_DISCOMFORT_AREAS = {
    "posture_side": ["neck", "upper_back"],
    "shoulder_right": ["shoulders"],
    "shoulder_left": ["shoulders"],
    "squat": ["hips", "knees", "ankles_feet"],
    "aslr_right": ["hamstrings"],
    "aslr_left": ["hamstrings"],
}

DOMAIN_TO_RECOVERY_AREAS = {
    "cervical_control": ["neck"],
    "thoracic_mobility": ["upper_back"],
    "shoulder_mobility": ["shoulders"],
    "scapular_control": ["shoulders", "upper_back"],
    "core_stability": ["lower_back"],
    "trunk_core_control": ["lower_back", "hips"],
    "hip_mobility": ["hips"],
    "hamstring_mobility": ["hamstrings"],
    "aslr": ["hamstrings"],
    "ankle_mobility": ["ankles_feet"],
    "squat_pattern": ["hips", "knees", "ankles_feet"],
    "functional_integration": ["hips"],
    "balance_proprioception": ["ankles_feet"],
}

SEVERITY_RANK = {
    "no_pain": 0,
    "mild_discomfort": 1,
    "moderate_non_sharp_pain": 2,
    "high_risk_pain": 3,
}

DOMAIN_TO_CATS = {
    "cervical_control": ["CC", "SH"],
    "thoracic_mobility": ["TM", "SH", "CC"],
    # Thoracic work is no longer treated as a universal shoulder correction.
    # It is selected when thoracic mobility is itself measured as a priority.
    "shoulder_mobility": ["SH", "CS"],
    "scapular_control": ["SH", "CS"],
    "core_stability": ["CS", "FI", "CC"],
    "trunk_core_control": ["CS", "FI", "HM"],
    "hip_mobility": ["HM", "HS", "CS"],
    "hamstring_mobility": ["HS", "HM", "CS"],
    "aslr": ["HS", "CS", "HM"],
    "ankle_mobility": ["AM", "FI", "HM"],
    "squat_pattern": ["FI", "HM", "AM", "CS"],
    "functional_integration": ["FI", "CS", "BP"],
    "balance_proprioception": ["BP", "FI", "AM"],
}

def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_exercise_library(path: str | Path):
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Invalid exercise library format")

def _num(v, default=0.0):
    try: return float(v)
    except Exception: return default

def _csv(v):
    if isinstance(v, list): return [str(x).strip() for x in v if str(x).strip()]
    return [x.strip() for x in str(v or "").split(",") if x.strip()]

def _lang(language):
    return "en" if str(language or "").lower().startswith("en") else "fr"

def _band(score, lang):
    score = _num(score)
    if score >= 80: return {"color":"green","label":"Good" if lang=="en" else "Bon"}
    if score >= 70: return {"color":"yellow","label":"Fair" if lang=="en" else "Correct"}
    if score >= 60: return {"color":"orange","label":"Needs improvement" if lang=="en" else "À améliorer"}
    return {"color":"red","label":"Limited" if lang=="en" else "Limité"}

def _safe_dict(value) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _safe_json_value(value) -> Any:
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw or raw[0] not in "[{":
        return value
    try:
        return json.loads(raw)
    except Exception:
        return value


def _deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for key, value in (incoming or {}).items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _questionnaire_fragment(value) -> Dict[str, Any]:
    """Unwrap common API/storage envelopes without discarding direct fields."""
    obj = _safe_dict(value)
    if not obj:
        return {}
    wrappers = (
        "data", "answers", "responses", "questionnaire", "questionnaire_json",
        "pre_screening_questionnaire", "intake", "intake_json", "intake_context",
    )
    merged: Dict[str, Any] = {}
    for wrapper in wrappers:
        nested = _safe_dict(obj.get(wrapper))
        if nested:
            merged = _deep_merge(merged, _questionnaire_fragment(nested))
    for key, item in obj.items():
        if key not in wrappers:
            merged[key] = item
    return merged


def _apply_questionnaire_aliases(q: Dict[str, Any]) -> Dict[str, Any]:
    aliases = {
        "painLevel": "pain_level",
        "painStatus": "pain_status",
        "painScore": "pain_score",
        "painIntensity": "pain_intensity",
        "symptomLevel": "symptom_level",
        "discomfortLevel": "discomfort_level",
        "primaryTensionArea": "primary_tension_area",
        "tensionAreas": "tension_areas",
        "medicalRestriction": "medical_restriction",
        "medicalRestrictions": "medical_restriction",
        "recentInjury": "recent_injury",
        "availableEquipment": "available_equipment",
        "painClearance": "pain_clearance",
        "movementPainClearance": "movement_pain_clearance",
        "movementClearance": "movement_clearance",
        "painContext": "pain_context",
        "activityLevel": "activity_level",
        "trainingLevel": "training_level",
        "seatedHoursPerDay": "seated_hours_per_day",
        "redFlags": "red_flags",
        "safetyFlags": "safety_flags",
        "painQuality": "pain_quality",
        "painDescription": "pain_description",
        "medicalNotes": "medical_notes",
        "sharpPain": "sharp_pain",
        "radiatingSymptoms": "radiating_symptoms",
        "neurologicalSymptoms": "neurological_symptoms",
        "recentTrauma": "recent_trauma",
        "acuteInjury": "acute_injury",
        "unexplainedWeakness": "unexplained_weakness",
    }
    out = dict(q or {})
    for source, target in aliases.items():
        if target not in out and source in out:
            out[target] = out[source]
    return out


def _questionnaire(payload):
    """Merge every supported questionnaire/intake representation.

    Report/session values are loaded first and explicit top-level request values
    override them. This preserves the base questionnaire while accepting a later
    pain-clearance object supplied under a different compatibility field.
    """
    payload = payload or {}
    report = _safe_dict(payload.get("report"))
    # Low-to-high precedence: legacy intake first, questionnaire_json last.
    candidate_keys = (
        "pre_screening_questionnaire", "intake", "intake_json",
        "intake_context", "questionnaire", "questionnaire_json",
    )
    direct_shape_keys = (
        "pain_level", "painLevel", "pain_score", "painScore", "painIntensity",
        "tension_areas", "tensionAreas", "pain_clearance", "painClearance",
        "medical_restriction", "medicalRestriction",
    )
    merged: Dict[str, Any] = {}
    for container in (report, payload):
        for key in candidate_keys:
            fragment = _questionnaire_fragment(container.get(key))
            if fragment:
                merged = _deep_merge(merged, fragment)
        # Also accept questionnaire-shaped fields attached directly to report or payload.
        if any(key in container for key in direct_shape_keys):
            direct = {key: value for key, value in container.items() if key != "report"}
            merged = _deep_merge(merged, _questionnaire_fragment(direct))
    return _apply_questionnaire_aliases(merged)


def _normalize_token(value) -> str:
    raw = str(value or "").strip().lower()
    raw = unicodedata.normalize("NFKD", raw)
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    raw = raw.replace("’", "'")
    raw = re.sub(r"[\\/|]+", "_", raw)
    raw = re.sub(r"[^a-z0-9]+", "_", raw)
    return raw.strip("_")


_NEGATIVE_ANSWERS = {
    "", "0", "false", "no", "none", "non", "n", "not_applicable", "na",
    "n_a", "aucun", "aucune", "sans", "negative", "absent", "no_pain",
    "none_reported", "nothing_reported", "no_issue", "no_issues",
    "no_restriction", "no_restrictions", "no_medical_restriction",
    "no_medical_restrictions", "no_recent_injury", "no_injury",
    "aucune_restriction", "aucune_restriction_medicale", "pas_de_restriction",
    "pas_de_restriction_medicale", "aucune_blessure", "pas_de_blessure",
    "aucun_symptome", "aucun_drapeau_rouge",
}


def _truthy(value) -> bool:
    value = _safe_json_value(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, dict):
        for key in ("value", "answer", "status", "selected", "checked", "enabled", "present", "yes"):
            if key in value:
                return _truthy(value.get(key))
        return any(_truthy(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_truthy(item) for item in value)
    token = _normalize_token(value)
    if token in _NEGATIVE_ANSWERS:
        return False
    negative_prefixes = ("no_", "not_", "pas_de_", "aucune_", "aucun_", "sans_")
    negative_subjects = ("restriction", "injury", "blessure", "symptom", "pain", "douleur", "red_flag", "drapeau_rouge")
    if token.startswith(negative_prefixes) and any(subject in token for subject in negative_subjects):
        return False
    return True


def _normalize_area(value) -> Optional[str]:
    raw = _normalize_token(value)
    if raw in _NEGATIVE_ANSWERS or raw in {
        "no_specific_tension", "aucune_zone_specifique", "nothing", "rien",
    }:
        return None
    normalized_aliases = {_normalize_token(alias): area for alias, area in AREA_ALIASES.items()}
    normalized_aliases.update({
        "epaule": "shoulders", "epaules": "shoulders",
        "haut_du_dos": "upper_back", "dos_superieur": "upper_back",
        "bas_du_dos": "lower_back", "lombaires": "lower_back",
        "ischio_jambier": "hamstrings", "ischio_jambiers": "hamstrings",
        "ischios_jambiers": "hamstrings", "arriere_des_cuisses": "hamstrings",
        "cheville_pied": "ankles_feet", "chevilles_pieds": "ankles_feet",
        "ankles_and_feet": "ankles_feet", "ankle_foot": "ankles_feet",
    })
    return normalized_aliases.get(raw)


def _normalize_symptom_status(value) -> str:
    """Normalize raw labels, scores, localized text and structured answers."""
    value = _safe_json_value(value)
    if isinstance(value, dict):
        risk_keys = (
            "sharp", "is_sharp", "sharp_pain", "radiating", "radiation",
            "radiating_symptoms", "numbness", "tingling", "instability",
            "significant", "severe", "neurological", "neurological_symptoms",
        )
        if any(_truthy(value.get(key)) for key in risk_keys if key in value):
            return "high_risk_pain"
        for key in (
            "severity", "status", "level", "pain_level", "painLevel",
            "pain_score", "painScore", "pain_intensity", "painIntensity",
            "pain", "value", "answer",
        ):
            if key in value and value.get(key) not in (None, ""):
                return _normalize_symptom_status(value.get(key))
        return "no_pain"
    if isinstance(value, (int, float)):
        score = float(value)
        if score >= 7:
            return "high_risk_pain"
        if score >= 4:
            return "moderate_non_sharp_pain"
        if score > 0:
            return "mild_discomfort"
        return "no_pain"

    raw_string = str(value or "").strip()
    numeric_match = re.fullmatch(r"([0-9]+(?:[.,][0-9]+)?)\s*(?:/\s*10)?", raw_string)
    if numeric_match:
        return _normalize_symptom_status(float(numeric_match.group(1).replace(",", ".")))

    raw = _normalize_token(raw_string)
    negative_pain_prefixes = ("no_", "not_", "pas_de_", "aucune_", "aucun_", "sans_")
    if raw.startswith(negative_pain_prefixes) and any(
        token in raw for token in ("pain", "douleur", "discomfort", "inconfort", "gene", "sharp", "vive", "significant", "important")
    ):
        return "no_pain"
    if raw in _NEGATIVE_ANSWERS or raw in {
        "no_pain", "sans_douleur", "aucune_douleur", "pas_de_douleur",
        "no_discomfort", "aucun_inconfort", "clear", "comfortable",
    }:
        return "no_pain"
    high_tokens = (
        "significant", "important", "importante", "sharp", "vive", "aigu",
        "severe", "high", "radiat", "irradi", "numb", "engourdi", "tingl",
        "fourmil", "instabil", "acute_injury", "trauma", "neurolog",
    )
    if any(token in raw for token in high_tokens):
        return "high_risk_pain"
    if raw in {
        "moderate", "modere", "moderee", "moderate_pain", "douleur_moderee",
        "pain", "douleur", "non_sharp_pain", "moderate_non_sharp_pain",
    } or "moderate" in raw or "modere" in raw:
        return "moderate_non_sharp_pain"
    mild_tokens = (
        "mild", "leger", "legere", "discomfort", "inconfort", "gene",
        "tension", "caution", "ache", "courbature", "sensitive",
    )
    if any(token in raw for token in mild_tokens):
        return "mild_discomfort"
    return "no_pain"


def _movement_key(value) -> str:
    raw = _normalize_token(value)
    aliases = {
        "deep_squat": "squat", "squat_test": "squat", "deep_squat_test": "squat",
        "active_straight_leg_raise_right": "aslr_right",
        "active_straight_leg_raise_left": "aslr_left",
        "active_straight_leg_raise_r": "aslr_right",
        "active_straight_leg_raise_l": "aslr_left",
        "aslr_r": "aslr_right", "aslr_l": "aslr_left",
        "right_aslr": "aslr_right", "left_aslr": "aslr_left",
        "shoulder_mobility_right": "shoulder_right",
        "shoulder_mobility_left": "shoulder_left",
        "right_shoulder": "shoulder_right", "left_shoulder": "shoulder_left",
        "shoulder_right_test": "shoulder_right", "shoulder_left_test": "shoulder_left",
        "side_posture": "posture_side", "posture_side_test": "posture_side",
    }
    if raw in aliases:
        return aliases[raw]
    if "aslr" in raw or "straight_leg" in raw:
        return "aslr_left" if "left" in raw or "gauche" in raw or raw.endswith("_l") else "aslr_right"
    if "shoulder" in raw or "epaule" in raw:
        return "shoulder_left" if "left" in raw or "gauche" in raw or raw.endswith("_l") else "shoulder_right"
    if "squat" in raw:
        return "squat"
    if "posture" in raw:
        return "posture_side"
    return raw


def _movement_side(movement: str) -> Optional[str]:
    key = _movement_key(movement)
    if "right" in key:
        return "right"
    if "left" in key:
        return "left"
    return None


def _medical_restriction_present(q: Dict[str, Any]) -> bool:
    for key in (
        "medical_restriction", "medical_restrictions", "recent_injury",
        "medicalRestriction", "medicalRestrictions", "recentInjury",
    ):
        if key in q and q.get(key) not in (None, ""):
            return _truthy(q.get(key))
    labels = _safe_dict(q.get("labels"))
    for key in ("medical_restriction", "medicalRestriction", "medical"):
        if key in labels:
            return _truthy(labels.get(key))
    return False


def _red_flags(q: Dict[str, Any]) -> List[str]:
    q = q or {}
    found: List[str] = []
    if _medical_restriction_present(q):
        found.append("medical_restriction")
    direct_flags = {
        "sharp_pain": ("sharp_pain", "sharp", "sharpPain"),
        "radiating_symptoms": ("radiating_symptoms", "radiating", "radiation", "radiatingSymptoms"),
        "numbness": ("numbness", "numb"),
        "tingling": ("tingling",),
        "instability": ("instability", "giving_way", "givingWay"),
        "recent_trauma": ("recent_trauma", "trauma", "recentTrauma"),
        "acute_injury": ("acute_injury", "injury", "acuteInjury"),
        "neurological_symptoms": ("neurological_symptoms", "neurological", "neurologicalSymptoms"),
        "dizziness": ("dizziness", "dizzy"),
        "unexplained_weakness": ("unexplained_weakness", "unexplainedWeakness"),
    }
    for label, keys in direct_flags.items():
        if any(_truthy(q.get(key)) for key in keys if key in q):
            found.append(label)

    raw_flags = q.get("red_flags") or q.get("safety_flags") or q.get("redFlags") or q.get("safetyFlags") or []
    raw_flags = _safe_json_value(raw_flags)
    if isinstance(raw_flags, str):
        raw_flags = _csv(raw_flags)
    if isinstance(raw_flags, dict):
        raw_flags = [key for key, value in raw_flags.items() if _truthy(value)]
    for value in raw_flags if isinstance(raw_flags, list) else []:
        label = _normalize_token(value)
        if label and label not in _NEGATIVE_ANSWERS and label not in found:
            found.append(label)

    free_text = " ".join(str(q.get(key) or "") for key in (
        "pain_quality", "painQuality", "pain_description", "painDescription",
        "symptoms", "notes", "medical_notes", "medicalNotes",
    )).lower()
    text_terms = {
        "sharp_pain": ("sharp pain", "douleur vive", "douleur aigue", "douleur aiguë"),
        "radiating_symptoms": ("radiat", "irradi"),
        "numbness": ("numb", "engourdi"),
        "tingling": ("tingl", "fourmil"),
        "instability": ("instab", "giving way", "derob"),
        "recent_trauma": ("trauma", "chute recente", "chute récente", "recent fall"),
        "dizziness": ("dizz", "vertige"),
        "neurological_symptoms": ("neurolog",),
    }
    for label, terms in text_terms.items():
        if any(term in free_text for term in terms) and label not in found:
            found.append(label)
    return found


def _priority_area_order(priorities: List[Dict[str, Any]]) -> List[str]:
    ordered: List[str] = []
    for priority in priorities:
        for area in DOMAIN_TO_RECOVERY_AREAS.get(priority.get("id"), []):
            if area not in ordered:
                ordered.append(area)
    return ordered


def _choose_movement_area(
    movement: str,
    explicit_areas: List[str],
    priorities: List[Dict[str, Any]],
    explicit_area: Optional[str] = None,
) -> Optional[str]:
    if explicit_area:
        normalized = _normalize_area(explicit_area)
        if normalized:
            return normalized
    candidates = MOVEMENT_DISCOMFORT_AREAS.get(_movement_key(movement), [])
    for area in explicit_areas:
        if area in candidates:
            return area
    for area in _priority_area_order(priorities):
        if area in candidates:
            return area
    return candidates[0] if candidates else None


def _as_values(value) -> List[Any]:
    value = _safe_json_value(value)
    if value is None:
        return []
    if isinstance(value, dict):
        # Checkbox-style storage: {"hamstrings": true, "hips": false}
        return [key for key, selected in value.items() if _truthy(selected)]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        return _csv(value)
    return [value]


def _questionnaire_severity(q: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    fields = (
        "pain_level", "pain_status", "pain", "pain_score", "pain_intensity",
        "painLevel", "painStatus", "painScore", "painIntensity",
        "symptom_level", "discomfort_level",
    )
    observed: List[Dict[str, Any]] = []
    strongest = "no_pain"
    for field in fields:
        if field not in q or q.get(field) in (None, ""):
            continue
        status = _normalize_symptom_status(q.get(field))
        observed.append({"field": field, "value": q.get(field), "normalized": status})
        if SEVERITY_RANK[status] > SEVERITY_RANK[strongest]:
            strongest = status
    labels = _safe_dict(q.get("labels"))
    if not observed and labels.get("pain_level"):
        status = _normalize_symptom_status(labels.get("pain_level"))
        observed.append({"field": "labels.pain_level", "value": labels.get("pain_level"), "normalized": status})
        strongest = status
    return strongest, observed


def _iter_movement_clearance(q: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Accept dict, nested dict, list and legacy flat movement-pain fields."""
    entries: List[Tuple[str, Any]] = []
    wrapper_keys = {"tests", "movements", "answers", "responses", "results", "items", "pain_clearance"}

    def consume(value):
        value = _safe_json_value(value)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    movement = (
                        item.get("movement") or item.get("movement_key") or item.get("movementKey")
                        or item.get("test_type") or item.get("testType") or item.get("test")
                        or item.get("key") or item.get("id") or item.get("name")
                    )
                    if movement:
                        entries.append((str(movement), item))
                    else:
                        consume(item)
            return
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            if _normalize_token(key) in wrapper_keys:
                consume(item)
            else:
                entries.append((str(key), item))

    for key in (
        "pain_clearance", "movement_pain_clearance", "movement_clearance",
        "painClearance", "movementPainClearance", "movementClearance",
    ):
        if key in q and q.get(key) not in (None, ""):
            consume(q.get(key))

    flat_patterns = (
        re.compile(r"^(?:pain_clearance|movement_pain|movement_clearance)_(.+)$"),
        re.compile(r"^(.+?)_(?:pain_clearance|pain_status|pain|discomfort)$"),
    )
    for key, value in q.items():
        normalized_key = _normalize_token(key)
        if normalized_key in {
            "pain", "pain_status", "pain_level", "pain_clearance",
            "movement_pain_clearance", "movement_clearance",
        }:
            continue
        for pattern in flat_patterns:
            match = pattern.match(normalized_key)
            if match:
                entries.append((match.group(1), value))
                break

    deduped: Dict[str, Any] = {}
    for movement, value in entries:
        key = _movement_key(movement)
        if not key:
            continue
        current = deduped.get(key)
        if current is None or SEVERITY_RANK[_normalize_symptom_status(value)] >= SEVERITY_RANK[_normalize_symptom_status(current)]:
            deduped[key] = value
    return list(deduped.items())


def _build_symptom_profile(q: Dict[str, Any], priorities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge questionnaire and movement-level symptom answers into one profile."""
    q = _apply_questionnaire_aliases(q or {})
    signals: List[Dict[str, Any]] = []
    flags = _red_flags(q)
    medical_restriction = "medical_restriction" in flags

    explicit_areas: List[str] = []
    labels = _safe_dict(q.get("labels"))
    primary = _normalize_area(
        q.get("primary_tension_area") or labels.get("primary_tension_area")
    )
    if primary:
        explicit_areas.append(primary)
    tension_values = q.get("tension_areas")
    if tension_values in (None, "", []):
        tension_values = labels.get("tension_areas")
    for value in _as_values(tension_values):
        area = _normalize_area(value)
        if area and area not in explicit_areas:
            explicit_areas.append(area)

    questionnaire_severity, pain_fields = _questionnaire_severity(q)
    questionnaire_raw = [entry["value"] for entry in pain_fields]
    if explicit_areas:
        area_severity = questionnaire_severity
        if area_severity == "no_pain":
            # Selecting a tension area is itself a valid mild-discomfort signal.
            area_severity = "mild_discomfort"
        for area in explicit_areas:
            signals.append({
                "area": area,
                "severity": area_severity,
                "source": "questionnaire",
                "related_movement": None,
                "side": None,
                "raw_status": questionnaire_raw,
            })

    clearance_entries = _iter_movement_clearance(q)
    for movement, raw_value in clearance_entries:
        detail = raw_value if isinstance(raw_value, dict) else {}
        severity = _normalize_symptom_status(raw_value)
        if severity == "no_pain":
            continue
        explicit_area = None
        if isinstance(detail, dict):
            explicit_area = (
                detail.get("area") or detail.get("target_area") or detail.get("targetArea")
                or detail.get("body_area") or detail.get("bodyArea") or detail.get("region")
            )
        area = _choose_movement_area(movement, explicit_areas, priorities, explicit_area)
        side_value = None
        if isinstance(detail, dict):
            side_value = detail.get("side") or detail.get("body_side") or detail.get("bodySide")
        signals.append({
            "area": area,
            "severity": severity,
            "source": "movement_pain_clearance",
            "related_movement": _movement_key(movement),
            "side": side_value or _movement_side(movement),
            "raw_status": raw_value,
        })

    high_risk_areas = {
        signal["area"] for signal in signals
        if signal.get("area") and signal.get("severity") == "high_risk_pain"
    }
    global_high_risk = medical_restriction or any(
        flag in flags for flag in {
            "sharp_pain", "radiating_symptoms", "numbness", "tingling", "instability",
            "recent_trauma", "acute_injury", "neurological_symptoms",
            "dizziness", "unexplained_weakness",
        }
    )

    merged: Dict[str, Dict[str, Any]] = {}
    for signal in signals:
        area = signal.get("area")
        if not area:
            continue
        current = merged.get(area)
        if current is None:
            merged[area] = {
                "area": area,
                "severity": signal["severity"],
                "sources": [signal["source"]],
                "related_movements": [signal["related_movement"]] if signal.get("related_movement") else [],
                "sides": [signal["side"]] if signal.get("side") else [],
            }
            continue
        if SEVERITY_RANK[signal["severity"]] > SEVERITY_RANK[current["severity"]]:
            current["severity"] = signal["severity"]
        if signal["source"] not in current["sources"]:
            current["sources"].append(signal["source"])
        if signal.get("related_movement") and signal["related_movement"] not in current["related_movements"]:
            current["related_movements"].append(signal["related_movement"])
        if signal.get("side") and signal["side"] not in current["sides"]:
            current["sides"].append(signal["side"])

    eligible_targets = [
        value for area, value in merged.items()
        if value["severity"] in {"mild_discomfort", "moderate_non_sharp_pain"}
        and area not in high_risk_areas
    ]
    area_order = explicit_areas + [
        signal.get("area") for signal in signals if signal.get("area") not in explicit_areas
    ]
    order_index = {area: idx for idx, area in enumerate(area_order)}
    eligible_targets.sort(key=lambda item: (
        -SEVERITY_RANK[item["severity"]],
        order_index.get(item["area"], 999),
    ))

    overall_state = questionnaire_severity
    for signal in signals:
        if SEVERITY_RANK[signal["severity"]] > SEVERITY_RANK[overall_state]:
            overall_state = signal["severity"]
    if global_high_risk:
        overall_state = "high_risk_pain"

    duration_token = _normalize_token(q.get("duration"))
    is_acute = duration_token in {
        "less_than_1_week", "less_than_one_week", "moins_d_une_semaine",
        "moins_de_1_semaine", "under_1_week",
    }

    return {
        "overall_state": overall_state,
        "legacy_pain_state": (
            "no_pain" if overall_state == "no_pain"
            else "discomfort" if overall_state == "mild_discomfort"
            else "pain"
        ),
        "questionnaire_severity": questionnaire_severity,
        "signals": signals,
        "eligible_targets": eligible_targets,
        "high_risk_areas": sorted(high_risk_areas),
        "red_flags": flags,
        "global_high_risk": global_high_risk,
        "medical_restriction": medical_restriction,
        "is_acute": is_acute,
        "input_compatibility": {
            "pain_fields_detected": [entry["field"] for entry in pain_fields],
            "movement_clearance_entries": len(clearance_entries),
            "questionnaire_keys": sorted(str(key) for key in q.keys()),
        },
    }


def _pain_state(q):
    return _build_symptom_profile(q or {}, [])["overall_state"]

def _experience(q):
    raw = str(q.get("activity_level") or q.get("training_level") or q.get("experience") or "moderate").lower()
    if raw in {"low","sedentary","beginner","inactive"}: return "beginner"
    if raw in {"high","advanced","athlete","very_active"}: return "advanced"
    return "intermediate"

def _available_equipment(q):
    raw = q.get("available_equipment") or q.get("availableEquipment") or q.get("equipment") or q.get("materials")
    if not raw:
        return {"none","bodyweight","foam_roller","massage_ball","elastic_band","light_weight","balance_pad","stick_or_pvc","trx"}
    aliases = {
        "none":"none", "no_equipment":"none", "aucun":"none",
        "bodyweight":"bodyweight", "body_weight":"bodyweight", "poids_du_corps":"bodyweight",
        "foam_roller":"foam_roller", "foamroller":"foam_roller", "rouleau":"foam_roller",
        "massage_ball":"massage_ball", "massageball":"massage_ball", "balle_de_massage":"massage_ball",
        "elastic_band":"elastic_band", "resistance_band":"elastic_band", "bande_elastique":"elastic_band",
        "light_weight":"light_weight", "light_weights":"light_weight", "poids_legers":"light_weight",
        "balance_pad":"balance_pad", "coussin_d_equilibre":"balance_pad",
        "stick_or_pvc":"stick_or_pvc", "stick":"stick_or_pvc", "pvc":"stick_or_pvc",
        "trx":"trx",
    }
    values = _as_values(raw)
    vals = {aliases.get(_normalize_token(value), _normalize_token(value)) for value in values if _normalize_token(value)}
    vals |= {"none","bodyweight"}
    return vals

def _domains(payload, lang):
    report = payload.get("report", payload) or {}
    rows = ((report.get("score_v2") or {}).get("domain_scores") or [])
    out = []
    for d in rows:
        did = d.get("id")
        if not did: continue
        score = _num(d.get("score"), 0)
        assessed = d.get("assessed", True)
        if assessed is False: continue
        out.append({
            "id": did,
            "score": score,
            "label": d.get("label_en") if lang=="en" else d.get("label_fr") or d.get("label") or did,
            "weight": _num(d.get("weight"), 1),
            "band": _band(score, lang),
        })
    if not out:
        # Safe fallback: do not invent multiple deficits.
        score = _num(report.get("flexilab_score", report.get("score", 70)), 70)
        out = [{"id":"functional_integration","score":score,"label":"Functional Integration" if lang=="en" else "Intégration fonctionnelle","weight":1,"band":_band(score,lang)}]
    return sorted(out, key=lambda d:(d["score"], -d["weight"]))

def _reported_categories(q):
    mapping = {
        "neck":["CC","TM","SH"], "shoulders":["SH","TM","CS"], "upper_back":["TM","SH","RB"],
        "lower_back":["CS","HM","RB"], "hips":["HM","CS","FI"], "hamstrings":["HS","HM"],
        "knees":["FI","HM","CS"], "ankles_feet":["AM","FI","HM"]
    }
    labels = _safe_dict(q.get("labels"))
    raw_areas = q.get("tension_areas") or q.get("tensionAreas") or labels.get("tension_areas")
    areas = [_normalize_area(value) for value in _as_values(raw_areas)]
    primary = _normalize_area(
        q.get("primary_tension_area") or q.get("primaryTensionArea") or labels.get("primary_tension_area")
    )
    cats = []
    for area in ([primary] if primary else []) + [area for area in areas if area]:
        for category in mapping.get(area, []):
            if category not in cats:
                cats.append(category)
    return cats


def _weekly_recovery_plan(profile: Dict[str, Any], week: int) -> List[Dict[str, Any]]:
    """Return the area-specific recovery targets for one week.

    Mild discomfort receives one exposure for one area or two exposures when
    several relevant areas are reported. Moderate non-sharp pain receives only
    one gentle exposure per week, rotating areas when several are present.
    """
    if profile.get("global_high_risk"):
        return []
    eligible = list(profile.get("eligible_targets") or [])
    moderate = [x for x in eligible if x.get("severity") == "moderate_non_sharp_pain"]
    mild = [x for x in eligible if x.get("severity") == "mild_discomfort"]
    if moderate:
        return [moderate[(week - 1) % len(moderate)]]
    if not mild:
        return []
    if len(mild) == 1:
        return [mild[0]]
    start = ((week - 1) * 2) % len(mild)
    return [mild[start], mild[(start + 1) % len(mild)]]


def _targeted_recovery_candidates(area: str, exercise_library: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    wanted = RECOVERY_AREA_EXERCISE_IDS.get(area, [])
    by_id = {str(ex.get("exercise_id")): ex for ex in exercise_library}
    return [by_id[eid] for eid in wanted if eid in by_id]


def _recovery_candidate_allowed(
    ex: Dict[str, Any],
    signal: Dict[str, Any],
    profile: Dict[str, Any],
    week: int,
    pain: str,
    experience: str,
    equipment: set,
) -> Tuple[bool, Optional[str]]:
    if ex.get("intervention_role") != "recovery":
        return False, "not_recovery"
    if str(ex.get("equipment") or "") not in {"foam_roller", "massage_ball"}:
        return False, "unsupported_equipment"
    if ex.get("demo_ready") is False or ex.get("video_ready") is False:
        return False, "video_not_ready"
    if profile.get("global_high_risk") or signal.get("severity") == "high_risk_pain":
        return False, "high_risk_symptom"
    if signal.get("area") in set(profile.get("high_risk_areas") or []):
        return False, "high_risk_area"
    if not _stage_allowed(ex, week) or not _load_allowed(ex, week, pain, experience, equipment):
        return False, "stage_load_or_equipment"

    contraindications = str(ex.get("contraindications") or "").lower()
    flags = set(profile.get("red_flags") or [])
    contraindication_map = {
        "dizziness": ("dizziness", "dizzy"),
        "neurological_symptoms": ("neurological",),
        "recent_trauma": ("trauma",),
        "acute_injury": ("acute", "injury"),
        "radiating_symptoms": ("radiat",),
        "numbness": ("numb",),
        "instability": ("unstable", "instability"),
    }
    for flag, terms in contraindication_map.items():
        if flag in flags and any(term in contraindications for term in terms):
            return False, f"contraindication_{flag}"
    if profile.get("is_acute") and "acute" in contraindications and signal.get("severity") == "moderate_non_sharp_pain":
        return False, "acute_pain_contraindication"
    return True, None


def _targeted_recovery_copy(
    ex: Dict[str, Any],
    signal: Dict[str, Any],
    week: int,
    day: int,
    priorities: List[Dict[str, Any]],
    pain: str,
    experience: str,
    lang: str,
) -> Dict[str, Any]:
    localized = _loc(ex, week, day, priorities, pain, experience, lang)
    severity = signal.get("severity") or "mild_discomfort"
    area = signal.get("area")
    moderate = severity == "moderate_non_sharp_pain"
    side_values = signal.get("sides") or []
    side = side_values[0] if len(side_values) == 1 else None
    movements = signal.get("related_movements") or []
    sources = signal.get("sources") or []

    localized["targeted_discomfort_recovery"] = True
    localized["targeted_symptom_recovery"] = True
    localized["gentle_recovery"] = True
    localized["symptom_severity"] = severity
    localized["target_area"] = area
    localized["targeted_recovery_area"] = area  # backward compatibility
    localized["target_side"] = side
    localized["symptom_source"] = sources[0] if len(sources) == 1 else sources
    localized["related_movement"] = movements[0] if len(movements) == 1 else movements
    localized["pressure_level"] = "light" if moderate else "light_to_moderate"
    localized["sets"] = "1"

    if moderate:
        localized["reps_time"] = "30 sec"
        stop_message = (
            "Use light pressure and stay within a comfortable range. Stop immediately if pain increases, becomes sharp, radiates, or causes numbness or tingling."
            if lang == "en" else
            "Utilisez une pression légère et restez dans une zone confortable. Arrêtez immédiatement si la douleur augmente, devient vive, irradie ou provoque un engourdissement ou des fourmillements."
        )
        rationale = (
            f"Included once this week as gentle, area-specific recovery support for reported moderate non-sharp pain in the {area.replace('_', ' ')}."
            if lang == "en" else
            f"Inclus une fois cette semaine comme récupération douce et ciblée pour une douleur modérée non vive signalée au niveau de {area.replace('_', ' ')}."
        )
    else:
        # Keep filmed-exercise dosage, bounded to a short recovery exposure.
        duration = int(_num(ex.get("default_duration_seconds_v3"), 30))
        duration = max(30, min(60, duration))
        localized["reps_time"] = f"{duration} sec"
        stop_message = (
            "Use comfortable pressure. Stop if discomfort increases, becomes sharp, radiates, or causes numbness or tingling."
            if lang == "en" else
            "Utilisez une pression confortable. Arrêtez si la gêne augmente, devient vive, irradie ou provoque un engourdissement ou des fourmillements."
        )
        rationale = (
            f"Included as targeted self-massage for reported tension or mild discomfort in the {area.replace('_', ' ')}."
            if lang == "en" else
            f"Inclus comme auto-massage ciblé pour une tension ou une gêne légère signalée au niveau de {area.replace('_', ' ')}."
        )

    joint_warning = (
        " Do not roll directly over a joint, bone, or the spine."
        if lang == "en" else
        " Ne passez pas directement sur une articulation, un os ou la colonne vertébrale."
    )
    localized["safety_stop_message"] = stop_message
    localized["why_in_this_program"] = rationale
    localized["instructions"] = ((localized.get("instructions") or "").strip() + " " + stop_message + joint_warning).strip()
    localized["tips"] = ((localized.get("tips") or "").strip() + joint_warning).strip()
    return localized

def _exercise_domains(ex):
    return _csv(ex.get("screening_domains_improved"))


def _load_allowed(ex, week, pain, experience, equipment):
    eq = str(ex.get("equipment") or "none").lower()
    if eq not in equipment and eq not in {"none","bodyweight"}:
        return False
    load = int(_num(ex.get("load_level_v3"), 0))
    if pain == "high_risk_pain" and load >= 2:
        return False
    if pain == "moderate_non_sharp_pain" and load >= 3:
        return False
    if pain == "mild_discomfort" and load >= 5:
        return False

    earliest = 1
    if eq == "elastic_band":
        earliest = {
            "no_pain": 2,
            "mild_discomfort": 3,
            "moderate_non_sharp_pain": 4,
            "high_risk_pain": 99,
        }.get(pain, 3)
    if eq == "trx" and pain == "high_risk_pain":
        earliest = 99
    if eq == "light_weight":
        earliest = {"beginner":4,"intermediate":3,"advanced":2}[experience]
        if pain != "no_pain":
            earliest = 99
    return week >= earliest

def _stage_allowed(ex, week):
    return int(_num(ex.get("progression_stage_v3", ex.get("min_week",1)),1)) <= week and int(_num(ex.get("difficulty_1_5",3),3)) <= {1:2,2:3,3:4,4:5}[week]

def _matches_priority(ex, domain_id: str) -> Tuple[bool, bool]:
    """Return (exact_domain_match, mapped_category_match)."""
    domains = _exercise_domains(ex)
    exact = domain_id in domains
    mapped = str(ex.get("category_code") or "") in DOMAIN_TO_CATS.get(domain_id, [])
    return exact, mapped

def _is_direct_priority_exercise(ex: Dict[str, Any], domain_id: str) -> bool:
    """Strict direct match used for dosage quotas.

    An exercise must explicitly improve the measured domain and belong to that
    domain's direct library family. Supporting or integration exercises remain
    selectable, but never count as direct exposure.
    """
    domains = set(_exercise_domains(ex))
    category = str(ex.get("category_code") or "")
    return domain_id in domains and category in DIRECT_PRIORITY_CATS.get(domain_id, set())

def _direct_priority_exposure_counts(sessions: List[Dict[str, Any]], priorities: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    for session in sessions:
        for ex in session.get("exercises", []):
            for p in priorities[:3]:
                if _is_direct_priority_exercise(ex, p["id"]):
                    counts[p["id"]] += 1
    return dict(counts)

def _priority_relationship(ex: Dict[str, Any], domain_id: str) -> str:
    if _is_direct_priority_exercise(ex, domain_id):
        return "direct"
    exact, mapped = _matches_priority(ex, domain_id)
    if exact:
        return "supporting"
    if mapped:
        return "integration"
    return "unrelated"

def _priority_weekly_quota(priority: Dict[str, Any], rank: int) -> int:
    """Minimum direct exposures. Rank and deficit determine dose, not library size."""
    score = _num(priority.get("score"), 100)
    if rank == 0:
        return 4 if score < 60 else 3 if score < 70 else 2 if score < 80 else 1
    if rank == 1:
        return 3 if score < 60 else 2 if score < 80 else 1
    return 2 if score < 70 else 1

def _feasible_direct_quota(
    priority: Dict[str, Any],
    rank: int,
    exercise_library: List[Dict[str, Any]],
    week: int,
    pain: str,
    experience: str,
    equipment: set,
) -> int:
    """Cap the dosage target only when the current demo library cannot deliver it.

    Feasibility respects progression stage, equipment, pain/load rules and each
    exercise's weekly repeat cap. This prevents a false failed audit for a domain
    that has only one safe Week-1 direct exercise, while retaining the full quota
    whenever the library can support it.
    """
    base = _priority_weekly_quota(priority, rank)
    capacity = 0
    for ex in exercise_library:
        if not _is_direct_priority_exercise(ex, priority["id"]):
            continue
        if not _stage_allowed(ex, week) or not _load_allowed(ex, week, pain, experience, equipment):
            continue
        capacity += min(3, int(_num(ex.get("repeat_limit_per_week_v3"), 2)))
    return min(base, capacity)

def _build_priority_slots(priorities: List[Dict[str, Any]]) -> Dict[int, List[str]]:
    """Distribute priority exposures across three sessions while preserving variety."""
    slots = {1: [], 2: [], 3: []}
    for rank, priority in enumerate(priorities[:3]):
        quota = _priority_weekly_quota(priority, rank)
        # Stagger the starting day so secondary priorities do not always appear in Day 1.
        start = rank % 3
        for i in range(quota):
            day = ((start + i) % 3) + 1
            slots[day].append(priority["id"])
    return slots

def _priority_exposure_counts(sessions: List[Dict[str, Any]], priorities: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    for session in sessions:
        for ex in session.get("exercises", []):
            domains = set(_csv(ex.get("screening_domains_improved")))
            for p in priorities[:3]:
                if p["id"] in domains:
                    counts[p["id"]] += 1
    return dict(counts)

def _score_ex(ex, week, day, priorities, qcats, pain, experience, equipment,
              session_used, week_counts, program_counts, previous_week_ids, target_domain=None):
    eid = ex.get("exercise_id")
    if not eid or eid in session_used: return -1e9
    if not _stage_allowed(ex, week) or not _load_allowed(ex, week, pain, experience, equipment): return -1e9

    role = ex.get("intervention_role") or "mobility"
    day_roles = {
        1:["mobility","activation","stability","recovery"],
        2:["stability","integration","activation","mobility"],
        3:["integration","stability","mobility","activation"],
    }[day]
    if role not in day_roles: return -1e9

    weekly_cap = int(_num(ex.get("repeat_limit_per_week_v3"), 2))
    program_cap = int(_num(ex.get("repeat_limit_program_v3"), 5))
    exact_target = bool(target_domain and target_domain in _exercise_domains(ex))
    # A scarce demo-library priority must not disappear in weeks 3-4 merely
    # because a generic program-wide variety cap was reached. Weekly safety caps
    # remain intact; direct-priority anchors receive a larger program allowance.
    effective_program_cap = max(program_cap, 12) if exact_target else program_cap
    if week_counts[eid] >= weekly_cap or program_counts[eid] >= effective_program_cap: return -1e9

    c = str(ex.get("category_code") or "")
    domains = _exercise_domains(ex)
    score = 20.0 + (len(day_roles)-day_roles.index(role))*8

    for rank,p in enumerate(priorities[:5]):
        deficit = max(0, 100-p["score"])
        if p["id"] in domains: score += deficit * [1.7,1.35,1.0,.65,.35][rank]
        if c in DOMAIN_TO_CATS.get(p["id"],[]): score += deficit * [0.8,.6,.4,.25,.15][rank]

    if c in qcats: score += 12

    # Prefer exercises whose phase/stage fits the current week instead of merely
    # allowing every earlier-stage exercise forever.
    stage = int(_num(ex.get("progression_stage_v3", 1), 1))
    if stage == week:
        score += 24
    elif stage == week - 1:
        score += 10
    elif stage < week - 1:
        score -= 10 * (week - stage - 1)

    # Criteria-based chain continuity: reward the named next exercise when its
    # regression was used in the previous week. This does not force progression;
    # pain, load, equipment and movement-priority filters still take precedence.
    regression_id = str(ex.get("regression_id") or "")
    if regression_id and regression_id in previous_week_ids:
        score += 38

    if role == "recovery": score -= 18
    if day == 2 and int(_num(ex.get("load_level_v3"),0)) >= 3: score += 24 if week >= 2 else -30
    if day == 3 and role == "integration": score += 26
    if week >= 3 and int(_num(ex.get("load_level_v3"),0)) >= 3: score += 22
    if week == 4 and role == "integration": score += 16
    if eid in previous_week_ids: score -= 4 if exact_target else 14
    if week_counts[eid] > 0: score -= 10 if exact_target else 22
    if program_counts[eid] > 1: score -= (2 if exact_target else 8) * program_counts[eid]
    return score

def _dose(ex, week, pain, experience, lang):
    dtype = ex.get("dosage_type") or "dynamic_mobility_reps"
    min_sets = int(_num(ex.get("min_sets_v3"),1))
    default_sets = int(_num(ex.get("default_sets_v3"),min_sets))
    max_sets = int(_num(ex.get("max_sets_v3"),default_sets))

    if dtype in {"recovery_hold","soft_tissue_time"}:
        sets = 1
    elif pain in {"moderate_non_sharp_pain", "high_risk_pain"}:
        sets = min_sets
    elif week == 1:
        sets = default_sets
    elif week >= 3 and dtype in {"motor_control_reps","strength_reps"}:
        sets = min(max_sets, default_sets + 1)
    else:
        sets = default_sets

    duration = ex.get("default_duration_seconds_v3")
    reps = ex.get("default_reps_v3")
    if duration:
        duration = int(duration)
        max_total = int(_num(ex.get("max_total_exposure_seconds_v3"), duration*sets))
        if sets * duration > max_total:
            sets = max(1, max_total // duration)
        reps_time = f"{duration} sec" if duration < 60 else f"{duration//60} min"
    elif reps:
        reps = int(reps)
        if week == 2 and dtype != "strength_reps": reps = min(reps+2, 15)
        if week >= 3 and dtype == "strength_reps": reps = max(6, min(reps, 12))
        reps_time = f"{reps} repetitions" if lang=="en" else f"{reps} répétitions"
    else:
        reps_time = ex.get("reps_time") or ("8 repetitions" if lang=="en" else "8 répétitions")
    return str(sets), reps_time

def _loc(ex, week, day, priorities, pain, experience, lang):
    sets, reps_time = _dose(ex, week, pain, experience, lang)
    name = ex.get("name_en") if lang=="en" else ex.get("name_fr")
    role = ex.get("intervention_role") or "mobility"
    priority_names = [p["label"] for p in priorities[:3] if p["id"] in _exercise_domains(ex) or ex.get("category_code") in DOMAIN_TO_CATS.get(p["id"],[])]
    why = (
        "Selected to improve " + ", ".join(priority_names[:2]) + " through a progressive " + role + " stimulus."
        if lang=="en" else
        "Sélectionné pour améliorer " + ", ".join(priority_names[:2]) + " grâce à un stimulus progressif de " + role + "."
    )
    return {
        "id":ex.get("exercise_id"), "exercise_id":ex.get("exercise_id"),
        "name":name, "name_en":ex.get("name_en"), "name_fr":ex.get("name_fr"),
        "category_code":ex.get("category_code"), "target":ex.get("category_en") if lang=="en" else ex.get("category_fr"),
        "screening_domains_improved": _exercise_domains(ex),
        "priority_relationships": {p["id"]: _priority_relationship(ex, p["id"]) for p in priorities[:3]},
        "primary_objective":ex.get("primary_objective"), "intervention_role":role,
        "clinical_subject":ex.get("clinical_subject_v4") or ex.get("category_en"),
        "clinical_intervention_role":ex.get("clinical_intervention_role_v4") or role,
        "progression_id":ex.get("progression_id") or None,
        "regression_id":ex.get("regression_id") or None,
        "progression_criteria":ex.get("progression_criteria_v4") or [],
        "stop_criteria":ex.get("stop_criteria_v4") or [],
        "difficulty":ex.get("difficulty_1_5"), "phase":ex.get("phase"),
        "equipment":ex.get("equipment","none"),
        "equipment_label":ex.get("equipment_label_en") if lang=="en" else ex.get("equipment_label_fr"),
        "sets":sets, "reps_time":reps_time, "tempo":ex.get("tempo","controlled"), "rest":ex.get("rest",""),
        "coaching_cues":ex.get("coaching_cues_en") if lang=="en" else ex.get("coaching_cues_fr"),
        "common_errors":ex.get("common_errors_en") if lang=="en" else ex.get("common_errors_fr"),
        "clinical_rationale":ex.get("clinical_rationale_en") if lang=="en" else ex.get("clinical_rationale_fr"),
        "why_in_this_program":why,
        "instructions":ex.get("instructions_en") if lang=="en" else ex.get("instructions_fr"),
        "tips":ex.get("tips_en") if lang=="en" else ex.get("tips_fr"),
        "video_url":ex.get("video_url",""), "vimeo_url":ex.get("vimeo_url",""),
        "thumbnail_url":ex.get("thumbnail_url",""), "mp4_url":ex.get("mp4_url",""),
        "progression_stage":week,
    }

def _enforce_direct_priority_quotas(
    sessions: List[Dict[str, Any]],
    priorities: List[Dict[str, Any]],
    exercise_library: List[Dict[str, Any]],
    week: int,
    qcats: List[str],
    pain: str,
    experience: str,
    equipment: set,
    week_counts: Counter,
    program_counts: Counter,
    previous_week_ids: set,
    lang: str,
) -> Dict[str, Any]:
    """Repair a generated week until strict direct-priority quotas are met.

    The repair runs after normal selection, so progression and variety still
    shape the program. It replaces the least valuable non-direct slot only when
    a measured priority is underdosed.
    """
    targets = {
        p["id"]: _feasible_direct_quota(p, rank, exercise_library, week, pain, experience, equipment)
        for rank, p in enumerate(priorities[:3])
    }
    replacements = []

    def counts_now():
        return _direct_priority_exposure_counts(sessions, priorities)

    for rank, priority in enumerate(priorities[:3]):
        domain = priority["id"]
        safety = 0
        while counts_now().get(domain, 0) < targets[domain] and safety < 12:
            safety += 1
            # Put the next exposure into the session with the fewest direct
            # exercises for this priority, preserving day-to-day variety.
            session_order = sorted(
                sessions,
                key=lambda s: (
                    sum(_is_direct_priority_exercise(e, domain) for e in s.get("exercises", [])),
                    s.get("day", 0),
                ),
            )
            placed = False
            for session in session_order:
                day = int(session.get("day", 1))
                used = {e.get("exercise_id") or e.get("id") for e in session.get("exercises", [])}
                candidates = []
                for ex in exercise_library:
                    eid = ex.get("exercise_id")
                    if not eid or eid in used or not _is_direct_priority_exercise(ex, domain):
                        continue
                    sc = _score_ex(
                        ex, week, day, priorities, qcats, pain, experience, equipment,
                        used, week_counts, program_counts, previous_week_ids,
                        target_domain=domain,
                    )
                    if sc <= -1e8:
                        continue
                    # Direct quota delivery outranks stage novelty, while normal
                    # scoring still chooses the safest and most varied option.
                    candidates.append((sc + 500 - week_counts[eid] * 20, ex))
                candidates.sort(key=lambda x: (x[0], str(x[1].get("exercise_id"))), reverse=True)
                if not candidates:
                    continue
                new_ex = candidates[0][1]

                # Protect all direct work for the top three priorities. Prefer
                # replacing unrelated/supporting exercises from overrepresented
                # categories, never another scarce direct-priority anchor.
                category_counts = Counter(e.get("category_code") for e in session.get("exercises", []))
                replaceable = []
                for idx, old in enumerate(session.get("exercises", [])):
                    if old.get("targeted_discomfort_recovery"):
                        continue
                    if any(_is_direct_priority_exercise(old, p["id"]) for p in priorities[:3]):
                        continue
                    relationships = [_priority_relationship(old, p["id"]) for p in priorities[:3]]
                    importance = sum({"direct": 100, "supporting": 20, "integration": 8, "unrelated": 0}[r] for r in relationships)
                    # Repeated categories and late-session novelty are easiest to replace.
                    replace_score = importance - category_counts.get(old.get("category_code"), 0) * 5 - idx * 0.1
                    replaceable.append((replace_score, idx, old))
                if not replaceable:
                    continue
                replaceable.sort(key=lambda x: (x[0], x[1]))
                _, idx, old = replaceable[0]
                old_id = old.get("exercise_id") or old.get("id")
                if old_id:
                    week_counts[old_id] = max(0, week_counts[old_id] - 1)
                    program_counts[old_id] = max(0, program_counts[old_id] - 1)
                localized = _loc(new_ex, week, day, priorities, pain, experience, lang)
                session["exercises"][idx] = localized
                new_id = new_ex.get("exercise_id")
                week_counts[new_id] += 1
                program_counts[new_id] += 1
                replacements.append({
                    "priority": domain,
                    "day": day,
                    "removed": old_id,
                    "added": new_id,
                })
                placed = True
                break
            if not placed:
                break

    final_counts = counts_now()
    return {
        "targets": targets,
        "counts": final_counts,
        "passed": all(final_counts.get(k, 0) >= v for k, v in targets.items()),
        "replacements": replacements,
    }


def _enforce_targeted_discomfort_recovery(
    sessions: List[Dict[str, Any]],
    priorities: List[Dict[str, Any]],
    exercise_library: List[Dict[str, Any]],
    week: int,
    profile: Dict[str, Any],
    pain: str,
    experience: str,
    equipment: set,
    week_counts: Counter,
    program_counts: Counter,
    lang: str,
) -> Dict[str, Any]:
    """Insert symptom-specific massage/rolling without displacing direct work."""
    plan = _weekly_recovery_plan(profile, week)
    requested_quota = len(plan)
    if requested_quota == 0:
        return {
            "planned_targets": [],
            "requested_quota": 0,
            "quota": 0,
            "count": 0,
            "maximum_per_week": 0,
            "passed": True,
            "replacements": [],
            "skipped_targets": [],
        }

    replacements: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    deliverable_plan: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []

    for signal in plan:
        allowed_candidates = []
        rejection_reasons = Counter()
        for ex in _targeted_recovery_candidates(signal["area"], exercise_library):
            allowed, reason = _recovery_candidate_allowed(
                ex, signal, profile, week, pain, experience, equipment,
            )
            if not allowed:
                rejection_reasons[reason or "unknown"] += 1
                continue
            eid = ex.get("exercise_id")
            weekly_cap = int(_num(ex.get("repeat_limit_per_week_v3"), 2))
            program_cap = max(8, int(_num(ex.get("repeat_limit_program_v3"), 5)))
            if week_counts[eid] >= weekly_cap:
                rejection_reasons["weekly_cap"] += 1
                continue
            if program_counts[eid] >= program_cap:
                rejection_reasons["program_cap"] += 1
                continue
            allowed_candidates.append(ex)
        if allowed_candidates:
            deliverable_plan.append((signal, allowed_candidates))
        else:
            skipped.append({
                "area": signal.get("area"),
                "severity": signal.get("severity"),
                "reason": rejection_reasons.most_common(1)[0][0] if rejection_reasons else "no_filmed_candidate",
            })

    required_quota = len(deliverable_plan)
    if required_quota == 0:
        return {
            "planned_targets": plan,
            "requested_quota": requested_quota,
            "quota": 0,
            "count": 0,
            "maximum_per_week": 1 if any(x.get("severity") == "moderate_non_sharp_pain" for x in plan) else 2,
            "passed": True,
            "replacements": [],
            "skipped_targets": skipped,
        }

    moderate_plan = any(signal.get("severity") == "moderate_non_sharp_pain" for signal, _ in deliverable_plan)
    desired_days = [3] if moderate_plan or required_quota == 1 else [1, 3]

    for slot, (signal, candidates) in enumerate(deliverable_plan):
        # Prefer the least-used filmed option to preserve variety.
        candidates = sorted(
            candidates,
            key=lambda ex: (
                week_counts[ex.get("exercise_id")],
                program_counts[ex.get("exercise_id")],
                str(ex.get("exercise_id")),
            ),
        )
        preferred_day = desired_days[min(slot, len(desired_days) - 1)]
        session_order = sorted(
            sessions,
            key=lambda s: (
                0 if s.get("day") == preferred_day else 1,
                sum(1 for e in s.get("exercises", []) if e.get("targeted_symptom_recovery")),
                s.get("day", 0),
            ),
        )
        placed = False

        for session in session_order:
            if any(e.get("targeted_symptom_recovery") for e in session.get("exercises", [])):
                continue  # maximum one targeted recovery exercise per session
            used = {e.get("exercise_id") or e.get("id") for e in session.get("exercises", [])}
            new_ex = next((ex for ex in candidates if ex.get("exercise_id") not in used), None)
            if new_ex is None:
                continue

            category_counts = Counter(e.get("category_code") for e in session.get("exercises", []))
            replaceable = []
            for idx, old in enumerate(session.get("exercises", [])):
                if old.get("targeted_symptom_recovery"):
                    continue
                if any(_is_direct_priority_exercise(old, p["id"]) for p in priorities[:3]):
                    continue
                relationships = [_priority_relationship(old, p["id"]) for p in priorities[:3]]
                importance = sum(
                    {"direct": 100, "supporting": 22, "integration": 10, "unrelated": 0}[r]
                    for r in relationships
                )
                # Generic recovery and repeated/unrelated novelty are replaced first.
                generic_recovery_bonus = -50 if old.get("intervention_role") == "recovery" else 0
                repetition_bonus = -category_counts.get(old.get("category_code"), 0) * 6
                replace_score = importance + generic_recovery_bonus + repetition_bonus - idx * 0.1
                replaceable.append((replace_score, idx, old))
            if not replaceable:
                continue

            replaceable.sort(key=lambda item: (item[0], item[1]))
            _, idx, old = replaceable[0]
            old_id = old.get("exercise_id") or old.get("id")
            if old_id:
                week_counts[old_id] = max(0, week_counts[old_id] - 1)
                program_counts[old_id] = max(0, program_counts[old_id] - 1)

            localized = _targeted_recovery_copy(
                new_ex, signal, week, int(session.get("day", 1)),
                priorities, pain, experience, lang,
            )
            # Keep targeted recovery at the end of the session.
            remaining = [
                exercise for exercise_index, exercise in enumerate(session["exercises"])
                if exercise_index != idx
            ]
            session["exercises"] = remaining + [localized]

            new_id = new_ex.get("exercise_id")
            week_counts[new_id] += 1
            program_counts[new_id] += 1
            replacements.append({
                "area": signal.get("area"),
                "severity": signal.get("severity"),
                "day": session.get("day"),
                "removed": old_id,
                "added": new_id,
            })
            placed = True
            break

        if not placed:
            skipped.append({
                "area": signal.get("area"),
                "severity": signal.get("severity"),
                "reason": "no_replaceable_session_slot",
            })

    targeted = [
        e for session in sessions for e in session.get("exercises", [])
        if e.get("targeted_symptom_recovery")
    ]
    count = len(targeted)
    return {
        "planned_targets": plan,
        "requested_quota": requested_quota,
        "quota": required_quota,
        "count": count,
        "maximum_per_week": 1 if moderate_plan else 2,
        "passed": count >= required_quota,
        "replacements": replacements,
        "skipped_targets": skipped,
    }

def _session_similarity(a, b):
    A={e["id"] for e in a}; B={e["id"] for e in b}
    return len(A&B)/max(1,len(A|B))


def _quality(program, pain):
    sessions=[s for w in program["weeks"] for s in w["sessions"]]
    similarities=[]
    for i,s in enumerate(sessions):
        for t in sessions[i+1:]:
            if s["week"]==t["week"]:
                similarities.append(_session_similarity(s["exercises"],t["exercises"]))
    recovery=sum(1 for s in sessions for e in s["exercises"] if e.get("intervention_role")=="recovery")
    loaded_by_week=defaultdict(int)
    for s in sessions:
        for e in s["exercises"]:
            if e.get("equipment") in {"elastic_band","light_weight","trx"}:
                loaded_by_week[s["week"]]+=1

    failures=[]
    if similarities and max(similarities) > .55:
        failures.append("within_week_session_similarity")
    if recovery > 12:
        failures.append("recovery_overuse")
    if pain=="no_pain" and loaded_by_week[3]+loaded_by_week[4] == 0:
        failures.append("missing_loaded_progression")

    underexposed_weeks=[]
    targeted_failed_weeks=[]
    targeted_cap_violations=[]
    multiple_targeted_in_session=[]
    unsafe_targeted=[]
    equipment_mismatches=[]
    available_equipment=set(program.get("selection_strategy",{}).get("available_equipment") or [])
    profile=program.get("symptom_profile") or {}
    high_risk_areas=set(profile.get("high_risk_areas") or [])
    global_high_risk=bool(profile.get("global_high_risk"))

    for week in program.get("weeks", []):
        if not week.get("priority_coverage_passed", True):
            underexposed_weeks.append(week.get("week"))
        audit=week.get("targeted_symptom_recovery") or week.get("targeted_discomfort_recovery") or {}
        if not audit.get("passed", True):
            targeted_failed_weeks.append(week.get("week"))
        if audit.get("count",0) > audit.get("maximum_per_week",2):
            targeted_cap_violations.append(week.get("week"))
        for session in week.get("sessions", []):
            targeted=[e for e in session.get("exercises",[]) if e.get("targeted_symptom_recovery")]
            if len(targeted)>1:
                multiple_targeted_in_session.append({"week":week.get("week"),"day":session.get("day")})
            for ex in targeted:
                if global_high_risk or ex.get("target_area") in high_risk_areas:
                    unsafe_targeted.append(ex.get("exercise_id"))
                if ex.get("equipment") not in available_equipment:
                    equipment_mismatches.append(ex.get("exercise_id"))

    if underexposed_weeks:
        failures.append("priority_underexposure")
    if targeted_failed_weeks:
        failures.append("targeted_symptom_recovery_missing")
    if targeted_cap_violations:
        failures.append("targeted_symptom_recovery_overuse")
    if multiple_targeted_in_session:
        failures.append("multiple_targeted_recovery_in_session")
    if unsafe_targeted:
        failures.append("unsafe_targeted_recovery")
    if equipment_mismatches:
        failures.append("targeted_recovery_equipment_unavailable")

    return {
        "passed": not failures,
        "failures": failures,
        "max_within_week_similarity": round(max(similarities) if similarities else 0,2),
        "recovery_exposure_total": recovery,
        "loaded_exposures_by_week": dict(loaded_by_week),
        "priority_underexposed_weeks": underexposed_weeks,
        "targeted_recovery_failed_weeks": targeted_failed_weeks,
        "targeted_recovery_cap_violations": targeted_cap_violations,
        "multiple_targeted_recovery_sessions": multiple_targeted_in_session,
        "unsafe_targeted_recovery_exercises": unsafe_targeted,
        "targeted_recovery_equipment_mismatches": equipment_mismatches,
    }


def generate_clinical_prescription_v21(screening_payload, exercise_library, rules=None, movement_dna=None, language="fr"):
    lang=_lang(language)
    q=_questionnaire(screening_payload)
    experience=_experience(q)
    equipment=_available_equipment(q)
    priorities=_domains(screening_payload,lang)
    symptom_profile=_build_symptom_profile(q, priorities)
    pain=symptom_profile["overall_state"]
    qcats=_reported_categories(q)

    program_counts=Counter()
    previous_week_ids=set()
    weeks=[]
    phases={
        1:("Restore & Learn","Restaurer et apprendre"),
        2:("Control & Build Capacity","Contrôler et développer"),
        3:("Strengthen & Stabilize","Renforcer et stabiliser"),
        4:("Integrate & Challenge","Intégrer et progresser"),
    }
    focuses={
        1:("Mobility & Movement Quality","Mobilité et qualité du mouvement"),
        2:("Stability & Strength","Stabilité et renforcement"),
        3:("Functional Integration","Intégration fonctionnelle"),
    }

    for week in range(1,5):
        week_counts=Counter()
        sessions=[]
        priority_slots = _build_priority_slots(priorities)
        for day in range(1,4):
            selected=[]
            used=set()

            # Reserve measured movement priorities before general variety.
            for target_domain in priority_slots.get(day, []):
                candidates=[]
                for ex in exercise_library:
                    exact, mapped = _matches_priority(ex, target_domain)
                    if not (exact or mapped):
                        continue
                    sc=_score_ex(
                        ex,week,day,priorities,qcats,pain,experience,equipment,
                        used,week_counts,program_counts,previous_week_ids,
                        target_domain=target_domain,
                    )
                    if sc <= -1e8:
                        continue
                    sc += 180 if exact else 45
                    same_role = sum(
                        1 for e in selected
                        if e.get("intervention_role") == (ex.get("intervention_role") or "mobility")
                    )
                    sc -= same_role * 10
                    if ex.get("exercise_id") in previous_week_ids:
                        sc += 8 if target_domain == priorities[0]["id"] else 0
                    candidates.append((1 if exact else 0,sc,ex))
                candidates.sort(
                    key=lambda x:(x[0], x[1], str(x[2].get("exercise_id"))),
                    reverse=True,
                )
                ex=next((x for exact_rank,sc,x in candidates if sc > -1e8),None)
                if ex:
                    selected.append(_loc(ex,week,day,priorities,pain,experience,lang))
                    eid=ex.get("exercise_id")
                    used.add(eid); week_counts[eid]+=1; program_counts[eid]+=1

            role_targets = {
                1:["mobility","mobility","activation","stability","integration","recovery"],
                2:["activation","stability","stability","integration","mobility","integration"],
                3:["integration","integration","stability","mobility","activation","recovery"],
            }[day]
            for desired_role in role_targets:
                if len(selected) >= 6:
                    break
                candidates=[]
                for ex in exercise_library:
                    if (ex.get("intervention_role") or "mobility") != desired_role:
                        continue
                    sc=_score_ex(
                        ex,week,day,priorities,qcats,pain,experience,equipment,
                        used,week_counts,program_counts,previous_week_ids,
                    )
                    candidates.append((sc,ex))
                candidates.sort(key=lambda x:(x[0], str(x[1].get("exercise_id"))), reverse=True)
                ex=next((x for sc,x in candidates if sc > -1e8),None)
                if ex:
                    selected.append(_loc(ex,week,day,priorities,pain,experience,lang))
                    eid=ex.get("exercise_id")
                    used.add(eid); week_counts[eid]+=1; program_counts[eid]+=1

            # Generic recovery remains limited to one exercise in a session.
            # The targeted symptom strategy runs later and replaces this slot
            # whenever possible rather than adding random recovery volume.
            recovery_cap = 1
            if sum(1 for e in selected if e["intervention_role"]=="recovery") > recovery_cap:
                kept=[]; recovery_seen=0
                for e in selected:
                    if e["intervention_role"]=="recovery":
                        recovery_seen += 1
                        if recovery_seen > recovery_cap:
                            eid=e.get("exercise_id") or e.get("id")
                            if eid:
                                week_counts[eid]=max(0,week_counts[eid]-1)
                                program_counts[eid]=max(0,program_counts[eid]-1)
                            continue
                    kept.append(e)
                selected=kept
                used={e.get("exercise_id") or e.get("id") for e in selected}

            # Complete the session without filling it with random recovery drills.
            while len(selected) < 6:
                category_counts = Counter(e.get("category_code") for e in selected)
                fallback=[]
                for ex in exercise_library:
                    role = ex.get("intervention_role") or "mobility"
                    if role == "recovery" and sum(
                        1 for e in selected if e.get("intervention_role")=="recovery"
                    ) >= recovery_cap:
                        continue
                    if category_counts.get(ex.get("category_code"), 0) >= 2:
                        continue
                    sc=_score_ex(
                        ex,week,day,priorities,qcats,pain,experience,equipment,
                        used,week_counts,program_counts,previous_week_ids,
                    )
                    if role in {"stability","integration"} and day in {2,3}:
                        sc += 12
                    fallback.append((sc,ex))
                fallback.sort(key=lambda x:(x[0], str(x[1].get("exercise_id"))), reverse=True)
                ex=next((x for sc,x in fallback if sc > -1e8),None)
                if not ex:
                    break
                selected.append(_loc(ex,week,day,priorities,pain,experience,lang))
                eid=ex.get("exercise_id")
                used.add(eid); week_counts[eid]+=1; program_counts[eid]+=1

            sessions.append({
                "day":day, "week":week,
                "focus":focuses[day][0] if lang=="en" else focuses[day][1],
                "session_model":"priority_constrained_symptom_aware_v3_6_compat",
                "estimated_duration_minutes":min(30, 5+len(selected)*3),
                "exercises":selected,
                "clinical_balance":{
                    "exercise_count":len(selected),
                    "categories":sorted({e["category_code"] for e in selected}),
                },
            })

        # First repair strict direct-priority quotas, then add symptom-specific
        # recovery only by replacing an unprotected complementary slot.
        direct_audit = _enforce_direct_priority_quotas(
            sessions, priorities, exercise_library, week, qcats, pain, experience,
            equipment, week_counts, program_counts, previous_week_ids, lang,
        )
        recovery_audit = _enforce_targeted_discomfort_recovery(
            sessions, priorities, exercise_library, week, symptom_profile, pain,
            experience, equipment, week_counts, program_counts, lang,
        )

        # Re-audit after recovery insertion and refresh session metadata.
        exposure_counts = _priority_exposure_counts(sessions, priorities)
        direct_exposure_counts = _direct_priority_exposure_counts(sessions, priorities)
        exposure_targets = direct_audit["targets"]
        priority_coverage_passed = all(
            direct_exposure_counts.get(domain,0) >= target
            for domain,target in exposure_targets.items()
        )
        for session in sessions:
            session["clinical_balance"]={
                "exercise_count":len(session.get("exercises",[])),
                "categories":sorted({
                    e.get("category_code") for e in session.get("exercises",[])
                    if e.get("category_code")
                }),
                "targeted_recovery_count":sum(
                    1 for e in session.get("exercises",[])
                    if e.get("targeted_symptom_recovery")
                ),
            }
            session["estimated_duration_minutes"]=min(
                30, 5+len(session.get("exercises",[]))*3
            )

        weeks.append({
            "week":week,
            "priority_exposure_targets": exposure_targets,
            "priority_exposure_counts": exposure_counts,
            "direct_priority_exposure_counts": direct_exposure_counts,
            "direct_priority_quota_replacements": direct_audit["replacements"],
            "priority_coverage_passed": priority_coverage_passed,
            "targeted_discomfort_recovery": recovery_audit,
            "targeted_symptom_recovery": recovery_audit,
            "phase":phases[week][0] if lang=="en" else phases[week][1],
            "objective":(
                [
                    "Learn comfortable movement foundations.",
                    "Develop active control and capacity.",
                    "Introduce resistance and stronger stability demands when appropriate.",
                    "Integrate movement priorities into more challenging functional movement.",
                ][week-1]
                if lang=="en" else
                [
                    "Apprendre les bases du mouvement dans une zone confortable.",
                    "Développer le contrôle actif et la capacité.",
                    "Introduire la résistance et renforcer la stabilité lorsque cela est adapté.",
                    "Intégrer les priorités de mouvement dans des tâches fonctionnelles plus exigeantes.",
                ][week-1]
            ),
            "progression_logic":"Learn → control → load → integrate" if lang=="en" else "Apprendre → contrôler → charger → intégrer",
            "sessions":sessions,
        })
        previous_week_ids={
            e.get("exercise_id") or e.get("id")
            for session in sessions for e in session.get("exercises",[])
            if e.get("exercise_id") or e.get("id")
        }

    report=screening_payload.get("report",screening_payload) or {}
    movement_score=_num(report.get("flexilab_score", report.get("score",0)),0)
    program_mode={
        "no_pain":"movement_training",
        "mild_discomfort":"discomfort_aware_training",
        "moderate_non_sharp_pain":"gentle_pain_aware_training",
        "high_risk_pain":"safety_limited_training",
    }.get(pain,"movement_training")

    safety_notes=[
        "No exercise should increase symptoms.",
        "Movement quality is more important than repetitions.",
        "Reduce range or load if compensation appears.",
    ] if lang=="en" else [
        "Aucun exercice ne doit augmenter les symptômes.",
        "La qualité du mouvement prime sur le nombre de répétitions.",
        "Réduisez l’amplitude ou la charge si une compensation apparaît.",
    ]
    if pain=="mild_discomfort":
        safety_notes.append(
            "Targeted recovery should remain comfortable; stop if discomfort increases or becomes sharp."
            if lang=="en" else
            "La récupération ciblée doit rester confortable ; arrêtez si la gêne augmente ou devient vive."
        )
    elif pain=="moderate_non_sharp_pain":
        safety_notes.append(
            "Use only light pressure for targeted recovery and stop if pain increases, becomes sharp, radiates, or causes numbness or tingling."
            if lang=="en" else
            "Utilisez uniquement une pression légère pour la récupération ciblée et arrêtez si la douleur augmente, devient vive, irradie ou provoque un engourdissement ou des fourmillements."
        )
    elif pain=="high_risk_pain":
        safety_notes.append(
            "Targeted self-massage is excluded for high-risk symptoms. Seek qualified professional assessment when symptoms are significant, sharp, radiating, neurological, unstable, or linked to recent trauma."
            if lang=="en" else
            "L’auto-massage ciblé est exclu en présence de symptômes à risque. Demandez l’avis d’un professionnel qualifié si les symptômes sont importants, vifs, irradiants, neurologiques, instables ou liés à un traumatisme récent."
        )

    program={
        "engine_version":ENGINE_VERSION,
        "created_at":datetime.now(timezone.utc).isoformat(),
        "language":lang,
        "movement_score":movement_score,
        "movement_score_band":_band(movement_score,lang),
        "clinical_readiness":{
            # Keep the legacy field stable for existing frontend consumers.
            "pain_state":symptom_profile["legacy_pain_state"],
            "symptom_state":pain,
            "training_experience":experience,
            "program_mode":program_mode,
            "medical_advice_recommended": pain=="high_risk_pain",
            "moderate_non_sharp_recovery_enabled": (
                pain=="moderate_non_sharp_pain"
                and not symptom_profile.get("global_high_risk")
            ),
        },
        "symptom_profile":symptom_profile,
        "pre_screening_questionnaire":q,
        "movement_dna_summary":movement_dna or {},
        "clinical_priorities":priorities[:3],
        "main_priorities":priorities[:3],
        "monitor_domains":priorities[3:],
        "program_summary":{
            "duration":"4 weeks" if lang=="en" else "4 semaines",
            "frequency":"3 sessions/week" if lang=="en" else "3 séances/semaine",
            "session_duration":"18-30 min",
            "model":"stable movement priorities + varied progression + symptom-aware recovery",
        },
        "selection_strategy":{
            "principles":[
                "stable movement priorities",
                "anatomical subject separated from intervention role",
                "explicit regression and progression links where coherent",
                "criteria-based progression rather than week number alone",
                "priority exposure guaranteed before general session filling",
                "strict direct-priority quotas audited after final generation",
                "supporting and integration exercises excluded from direct quota counts",
                "mild discomfort receives 1-2 area-specific filmed recovery exposures per week",
                "moderate non-sharp pain receives a maximum of one gentle targeted recovery exposure per week",
                "significant, sharp, radiating or neurological symptoms exclude targeted self-massage",
                "targeted recovery never substitutes for direct priority work",
                "maximum one targeted recovery exercise in a session",
                "targeted recovery is placed at the end of the session",
                "variety inside the same movement-priority family",
                "strategic repetition when the filmed demo library is scarce",
                "questionnaire and movement-clearance driven safety",
                "compatibility aliases for intake_json, questionnaire_json, camelCase fields, numeric pain scores and localized negative answers",
                "exercise-specific dosage",
            ],
            "progression_model":"direct_priority_quota_symptom_aware_recovery_restore_control_stabilize_integrate_v9_compat",
            "available_equipment":sorted(equipment),
            "reported_category_preferences":qcats,
            "symptom_recovery_strategy":{
                "mild_discomfort_per_week":"1-2",
                "moderate_non_sharp_pain_per_week":"maximum 1",
                "high_risk_pain_per_week":"0",
                "eligible_equipment":["foam_roller","massage_ball"],
            },
        },
        "weeks":weeks,
        "safety_notes":safety_notes,
    }
    program["validation_flags"]=_quality(program,pain)
    return program

