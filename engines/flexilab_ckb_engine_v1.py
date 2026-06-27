"""
FlexiLab Clinical Knowledge Base Engine v1.0

This module adds clinical reasoning on top of:
- screening results
- domain scores
- clinical prescription engine

It generates:
- Movement DNA
- matched movement patterns
- clinical observations
- biomechanical interpretation
- prescription strategy
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _score(domain_scores: Dict[str, float], key: str, default: float = 100.0) -> float:
    try:
        return float(domain_scores.get(key, default))
    except Exception:
        return default


def classify_band(score: float) -> str:
    if score >= 80:
        return "good"
    if score >= 70:
        return "fair"
    if score >= 60:
        return "needs_improvement"
    return "limited"


def extract_domain_score_map(screening_payload: Dict[str, Any]) -> Dict[str, float]:
    report = screening_payload.get("report", screening_payload) or {}
    domains = ((report.get("score_v2") or {}).get("domain_scores") or [])
    out = {}
    for d in domains:
        did = d.get("id") or d.get("domain")
        if did:
            out[did] = float(d.get("score", 0))
    return out


def calculate_indices(domain_scores: Dict[str, float], symmetry_index: float = 100, pain_score: float = 0) -> Dict[str, Any]:
    mobility_keys = ["thoracic_mobility", "shoulder_mobility", "hip_mobility", "hamstring_mobility", "ankle_mobility"]
    control_keys = ["cervical_control", "core_stability", "functional_integration", "balance_proprioception"]

    mobility_values = [domain_scores[k] for k in mobility_keys if k in domain_scores]
    control_values = [domain_scores[k] for k in control_keys if k in domain_scores]

    mobility_index = round(sum(mobility_values) / len(mobility_values), 1) if mobility_values else None
    control_index = round(sum(control_values) / len(control_values), 1) if control_values else None
    movement_efficiency = round(sum(domain_scores.values()) / len(domain_scores), 1) if domain_scores else None

    if pain_score >= 7:
        readiness = "medical_clearance_recommended"
    elif pain_score >= 4:
        readiness = "limited"
    elif pain_score >= 1:
        readiness = "caution"
    else:
        readiness = "good"

    return {
        "movement_efficiency": movement_efficiency,
        "mobility_index": mobility_index,
        "control_index": control_index,
        "symmetry_index": symmetry_index,
        "recovery_readiness": readiness,
        "mobility_band": classify_band(mobility_index or 100),
        "control_band": classify_band(control_index or 100),
    }


def pattern_score(pattern: Dict[str, Any], domain_scores: Dict[str, float], pain_score: float = 0, asymmetry_significant: bool = False) -> float:
    score = 0.0

    for d in pattern.get("primary_domains", []):
        if d in domain_scores:
            score += max(0, 100 - domain_scores[d]) * 1.4

    for d in pattern.get("secondary_domains", []):
        if d in domain_scores:
            score += max(0, 100 - domain_scores[d]) * 0.8

    pid = pattern.get("pattern_id")
    if pid == "P07" and asymmetry_significant:
        score += 35
    if pid == "P12" and pain_score >= 4:
        score += 60
    if pid == "P10":
        if domain_scores and all(v >= 80 for v in domain_scores.values()) and not asymmetry_significant and pain_score <= 3:
            score += 80

    return round(score, 1)


def match_patterns(
    screening_payload: Dict[str, Any],
    movement_patterns: Dict[str, Any],
    pain_score: float = 0,
    asymmetry_significant: bool = False,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    domain_scores = extract_domain_score_map(screening_payload)
    scored = []
    for p in movement_patterns.get("patterns", []):
        s = pattern_score(p, domain_scores, pain_score, asymmetry_significant)
        if s > 0:
            item = dict(p)
            item["match_score"] = s
            scored.append(item)
    scored.sort(key=lambda x: x["match_score"], reverse=True)
    return scored[:top_n]


def generate_movement_dna(
    screening_payload: Dict[str, Any],
    movement_patterns: Dict[str, Any],
    language: str = "fr",
    pain_score: float = 0,
    symmetry_index: float = 100,
    asymmetry_significant: bool = False,
) -> Dict[str, Any]:
    domain_scores = extract_domain_score_map(screening_payload)
    indices = calculate_indices(domain_scores, symmetry_index=symmetry_index, pain_score=pain_score)
    patterns = match_patterns(screening_payload, movement_patterns, pain_score, asymmetry_significant, top_n=3)

    primary = patterns[0] if patterns else None
    secondary = patterns[1] if len(patterns) > 1 else None

    profile_name = None
    secondary_name = None
    clinical_priority = None

    if primary:
        profile_name = primary["name_fr"] if language == "fr" else primary["name_en"]
        clinical_priority = primary["clinical_priority_fr"] if language == "fr" else primary["clinical_priority_en"]
    if secondary:
        secondary_name = secondary["name_fr"] if language == "fr" else secondary["name_en"]

    return {
        "movement_dna_version": "FlexiLab Movement DNA v1.0",
        "primary_profile": profile_name,
        "secondary_profile": secondary_name,
        "matched_patterns": [
            {
                "pattern_id": p["pattern_id"],
                "name": p["name_fr"] if language == "fr" else p["name_en"],
                "match_score": p["match_score"],
                "clinical_priority": p["clinical_priority_fr"] if language == "fr" else p["clinical_priority_en"],
            }
            for p in patterns
        ],
        "indices": indices,
        "clinical_priority": clinical_priority,
        "domain_scores": domain_scores,
    }


def clinical_observation_text(movement_dna: Dict[str, Any], language: str = "fr") -> List[str]:
    primary = movement_dna.get("primary_profile") or ("Profil non classé" if language == "fr" else "Unclassified profile")
    priority = movement_dna.get("clinical_priority") or ""
    indices = movement_dna.get("indices", {})

    if language == "fr":
        return [
            f"Profil principal : {primary}.",
            f"Efficacité du mouvement : {indices.get('movement_efficiency', '--')}%.",
            f"Priorité clinique : {priority}",
        ]

    return [
        f"Primary profile: {primary}.",
        f"Movement efficiency: {indices.get('movement_efficiency', '--')}%.",
        f"Clinical priority: {priority}",
    ]


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    patterns = load_json(here / "movement_patterns_v1.json")
    sample = {
        "report": {
            "score_v2": {
                "domain_scores": [
                    {"id": "thoracic_mobility", "score": 64},
                    {"id": "core_stability", "score": 64},
                    {"id": "cervical_control", "score": 71},
                    {"id": "shoulder_mobility", "score": 74},
                    {"id": "hip_mobility", "score": 78},
                    {"id": "ankle_mobility", "score": 78},
                ]
            }
        }
    }
    dna = generate_movement_dna(sample, patterns, language="fr")
    print(json.dumps(dna, ensure_ascii=False, indent=2))
