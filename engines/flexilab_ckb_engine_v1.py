"""FlexiLab evidence-aware Movement DNA engine."""

from __future__ import annotations
from typing import Any, Dict, List
import json

AUTOMATIC_PATTERN_IDS = {
    "P01","P02","P03","P04","P05","P06","P07","P08","P09","P10","P11","P12"
}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_domains(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    report = payload.get("report", payload) or {}
    rows = ((report.get("score_v2") or {}).get("domain_scores") or [])
    return {
        row["id"]: row
        for row in rows
        if row.get("id") and row.get("assessment_status", "assessed") == "assessed"
    }

def _deficit(row: Dict[str, Any]) -> float:
    try:
        return max(0.0, 100.0 - float(row.get("score")))
    except Exception:
        return 0.0

def _pattern_score(pattern: Dict[str, Any], domains: Dict[str, Dict[str, Any]], pain_score: float, asymmetry_significant: bool) -> Dict[str, Any]:
    primary = [d for d in pattern.get("primary_domains", []) if d in domains]
    secondary = [d for d in pattern.get("secondary_domains", []) if d in domains]
    required = pattern.get("primary_domains", [])

    if pattern.get("pattern_id") not in AUTOMATIC_PATTERN_IDS:
        return {"score": 0, "coverage": 0}

    if required and not primary:
        return {"score": 0, "coverage": 0}

    raw = sum(_deficit(domains[d]) * 1.4 for d in primary)
    raw += sum(_deficit(domains[d]) * 0.7 for d in secondary)

    pid = pattern.get("pattern_id")
    if pid == "P07" and asymmetry_significant:
        raw += 35
    if pid == "P12" and pain_score >= 4:
        raw += 60
    if pid == "P10" and domains and all(float(v.get("score", 0)) >= 80 for v in domains.values()) and not asymmetry_significant and pain_score <= 3:
        raw += 80

    coverage = len(primary) / max(1, len(required))
    normalized = min(100.0, raw / max(1.0, 1.4 * 100 * max(1, len(required))) * 100)
    return {"score": round(normalized, 1), "coverage": round(coverage, 2)}

def match_patterns(payload, movement_patterns, pain_score=0, asymmetry_significant=False, top_n=3):
    domains = extract_domains(payload)
    matches = []
    for pattern in movement_patterns.get("patterns", []):
        result = _pattern_score(pattern, domains, pain_score, asymmetry_significant)
        if result["score"] < 15 or result["coverage"] < 0.5:
            continue
        item = dict(pattern)
        item["match_score"] = result["score"]
        item["evidence_coverage"] = result["coverage"]
        matches.append(item)
    matches.sort(key=lambda x: (x["match_score"], x["evidence_coverage"]), reverse=True)
    return matches[:top_n]

def generate_movement_dna(
    screening_payload,
    movement_patterns,
    language="fr",
    pain_score=0,
    symmetry_index=100,
    asymmetry_significant=False,
):
    domains = extract_domains(screening_payload)
    matches = match_patterns(
        screening_payload,
        movement_patterns,
        pain_score=pain_score,
        asymmetry_significant=asymmetry_significant,
        top_n=3,
    )

    primary = matches[0] if matches else None
    second = matches[1] if len(matches) > 1 else None
    confidence = "low"
    if primary:
        gap = primary["match_score"] - (second["match_score"] if second else 0)
        if primary["match_score"] >= 45 and primary["evidence_coverage"] >= 0.75 and gap >= 10:
            confidence = "moderate"
        if primary["match_score"] >= 65 and primary["evidence_coverage"] == 1 and gap >= 15:
            confidence = "high"

    profile = (
        primary["name_fr" if language == "fr" else "name_en"]
        if primary and confidence != "low"
        else ("Aucun profil dominant identifié" if language == "fr" else "No dominant profile identified")
    )

    return {
        "movement_dna_version": "FlexiLab Evidence-Aware Movement DNA v2",
        "primary_profile": profile,
        "profile_confidence": confidence,
        "matched_patterns": [
            {
                "pattern_id": p["pattern_id"],
                "name": p["name_fr"] if language == "fr" else p["name_en"],
                "match_score": p["match_score"],
                "evidence_coverage": p["evidence_coverage"],
                "clinical_priority": p["clinical_priority_fr"] if language == "fr" else p["clinical_priority_en"],
                "screening_disclaimer": p["disclaimer_fr"] if language == "fr" else p["disclaimer_en"],
            }
            for p in matches
        ],
        "domain_scores": {k: v.get("score") for k, v in domains.items()},
        "symmetry_index": symmetry_index,
        "clinical_priority": (
            primary["clinical_priority_fr"] if language == "fr" else primary["clinical_priority_en"]
        ) if primary and confidence != "low" else "",
    }

def clinical_observation_text(movement_dna, language="fr"):
    if language == "fr":
        return [
            f"Profil principal : {movement_dna.get('primary_profile', 'Aucun profil dominant identifié')}.",
            f"Niveau de confiance : {movement_dna.get('profile_confidence', 'faible')}.",
            "Cette classification décrit un profil de screening et ne constitue pas un diagnostic.",
        ]
    return [
        f"Primary profile: {movement_dna.get('primary_profile', 'No dominant profile identified')}.",
        f"Confidence level: {movement_dna.get('profile_confidence', 'low')}.",
        "This classification describes a screening profile and is not a diagnosis.",
    ]
