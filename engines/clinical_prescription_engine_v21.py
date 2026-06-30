
from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from pathlib import Path
import json

def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_exercise_library(path: str | Path):
    payload = load_json(path)
    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Invalid exercise library format")

def fnum(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def csv(x):
    return [v.strip() for v in str(x or "").split(",") if v.strip()]

def band(score, lang="fr"):
    score = fnum(score)
    if score >= 80: return {"color":"green","label":"Bon" if lang=="fr" else "Good"}
    if score >= 70: return {"color":"yellow","label":"Correct" if lang=="fr" else "Fair"}
    if score >= 60: return {"color":"orange","label":"À améliorer" if lang=="fr" else "Needs improvement"}
    return {"color":"red","label":"Limité" if lang=="fr" else "Limited"}

DOMAIN_TO_CATS = {
    "thoracic_mobility":["TM","RB","SH"],
    "core_stability":["CS","FI","RB"],
    "cervical_control":["CC","TM","RB"],
    "shoulder_mobility":["SH","TM","CS"],
    "hip_mobility":["HM","HS","CS"],
    "ankle_mobility":["AM","BP","FI"],
    "hamstring_mobility":["HS","HM"],
    "functional_integration":["FI","CS","BP"],
    "balance_proprioception":["BP","AM","FI"],
}
BLOCK_LABELS = {
    "reset":("Reset / respiration","Reset / breathing"),
    "mobility_primary":("Mobilité principale","Primary mobility"),
    "mobility_secondary":("Mobilité secondaire","Secondary mobility"),
    "activation":("Activation / contrôle","Activation / control"),
    "stability":("Stabilité","Stability"),
    "integration":("Intégration fonctionnelle","Functional integration"),
    "recovery":("Retour au calme","Cool-down"),
}
WEEK_TEMPLATES = {
    1:[["reset","mobility_primary","mobility_secondary","activation","stability","recovery"],
       ["reset","mobility_primary","activation","stability","integration","recovery"],
       ["reset","mobility_primary","mobility_secondary","activation","integration","recovery"]],
    2:[["reset","mobility_primary","mobility_secondary","activation","stability","integration"],
       ["reset","mobility_primary","activation","stability","stability","integration"],
       ["reset","mobility_primary","mobility_secondary","activation","stability","recovery"]],
    3:[["reset","mobility_primary","mobility_secondary","activation","stability","integration"],
       ["reset","mobility_primary","activation","stability","stability","integration"],
       ["reset","mobility_primary","mobility_secondary","stability","integration","recovery"]],
    4:[["reset","mobility_primary","activation","stability","integration","integration"],
       ["reset","mobility_primary","mobility_secondary","stability","integration","recovery"],
       ["reset","mobility_primary","activation","stability","integration","recovery"]],
}
WEEK_META = {
    1:("Restaurer","Restore",2,"Restaurer les amplitudes de base et installer un contrôle sans douleur.","Restore basic ranges and establish pain-free control."),
    2:("Contrôler","Control",3,"Maintenir les mobilités clés et ajouter du contrôle actif.","Maintain key mobility drills and add active control."),
    3:("Stabiliser","Stabilize",4,"Stabiliser les amplitudes gagnées sous contrainte légère.","Stabilize gained ranges under light demand."),
    4:("Intégrer","Integrate",5,"Intégrer les corrections dans des mouvements fonctionnels et préparer le re-test.","Integrate corrections into functional movement and prepare reassessment."),
}

def readiness(intake, lang="fr"):
    intake = intake or {}
    pain = fnum(intake.get("pain_score", intake.get("pain", 0)), 0)
    red_flags = intake.get("red_flags") or []
    if red_flags or pain >= 7:
        return {"pain_score":pain,"red_flags":red_flags,"program_mode":"recovery_only","readiness":"medical_clearance_recommended","label":"Avis médical recommandé" if lang=="fr" else "Medical clearance recommended","medical_advice_recommended":True}
    if pain >= 4:
        return {"pain_score":pain,"red_flags":red_flags,"program_mode":"recovery_control","readiness":"limited","label":"Douleur modérée" if lang=="fr" else "Moderate pain","medical_advice_recommended":True}
    if pain >= 1:
        return {"pain_score":pain,"red_flags":red_flags,"program_mode":"pain_free_corrective","readiness":"caution","label":"Douleur légère" if lang=="fr" else "Mild pain","medical_advice_recommended":False}
    return {"pain_score":0,"red_flags":[],"program_mode":"corrective","readiness":"normal","label":"Aucune douleur" if lang=="fr" else "No pain","medical_advice_recommended":False}

def domains(payload, lang="fr"):
    report = payload.get("report", payload) or {}
    ds = ((report.get("score_v2") or {}).get("domain_scores") or [])
    out = []
    for d in ds:
        did = d.get("id")
        if did:
            sc = fnum(d.get("score"), 0)
            out.append({"id":did,"label":d.get("label") or d.get("label_fr") or did,"label_en":d.get("label_en") or did,"score":sc,"weight":d.get("weight"),"band":band(sc,lang)})
    if not out:
        out = [{"id":"thoracic_mobility","label":"Mobilité thoracique","label_en":"Thoracic Mobility","score":70,"weight":20,"band":band(70,lang)},
               {"id":"core_stability","label":"Stabilité du tronc","label_en":"Core Stability","score":70,"weight":15,"band":band(70,lang)}]
    return sorted(out, key=lambda x:(x["score"], -fnum(x.get("weight"),0)))

def asymmetries(payload, lang="fr"):
    report = payload.get("report", payload) or {}
    items = {}
    for sec in report.get("sections", []) or []:
        for it in sec.get("items", []) or []:
            items[it.get("id")] = it
    out = []
    pairs = [
        ("shoulder_symmetry","Symétrie mobilité épaules","Shoulder mobility symmetry","shoulder_right_flexion","shoulder_left_flexion"),
        ("aslr_symmetry","Symétrie ASLR / ischio-jambiers","ASLR / hamstring symmetry","aslr_right_angle","aslr_left_angle"),
    ]
    for aid, fr, en, rkey, lkey in pairs:
        rv = items.get(rkey,{}).get("value"); lv = items.get(lkey,{}).get("value")
        if rv is not None and lv is not None:
            diff = abs(fnum(rv)-fnum(lv))
            out.append({"id":aid,"label":fr if lang=="fr" else en,"left_value":fnum(lv),"right_value":fnum(rv),"difference":round(diff,2),"restricted_side":"right" if fnum(rv)<fnum(lv) else "left" if fnum(lv)<fnum(rv) else "none","significant":diff>=10})
    return out

def cat(ex): return str(ex.get("category_code",""))
def obj(ex): return str(ex.get("primary_objective","")).lower()
def diff(ex): return int(fnum(ex.get("difficulty_1_5",3),3))
def exdomains(ex): return csv(ex.get("screening_domains_improved",""))

def pain_ok(ex, r):
    mode = r.get("program_mode","corrective")
    if mode == "corrective": return True
    if mode == "pain_free_corrective": return diff(ex) <= 4
    if mode == "recovery_control": return cat(ex)=="RB" or (diff(ex)<=2 and ("mobility" in obj(ex) or "control" in obj(ex)))
    if mode == "recovery_only": return cat(ex)=="RB" or "recovery" in obj(ex)
    return True

def build_strategy(priorities, movement_dna=None, lang="fr"):
    rank = []
    for p in priorities[:4]:
        for c in DOMAIN_TO_CATS.get(p["id"], []):
            if c not in rank: rank.append(c)
    profile = (movement_dna or {}).get("primary_profile","")
    low = str(profile).lower()
    if "bureau" in low or "desk" in low:
        for c in ["TM","CC","SH","CS","RB","FI"]:
            if c not in rank: rank.append(c)
    if len(rank) < 4:
        for c in ["TM","CS","SH","CC","FI","RB"]:
            if c not in rank: rank.append(c)
    return {
        "primary_profile": profile or ("Profil correctif général" if lang=="fr" else "General corrective profile"),
        "matched_patterns": (movement_dna or {}).get("matched_patterns", []),
        "priority_domains":[p["id"] for p in priorities[:4]],
        "primary_categories":rank[:2],
        "secondary_categories":rank[2:5],
        "support_categories":rank[5:] + [c for c in ["FI","RB"] if c not in rank],
        "strategy_text": "Stratégie v2.1 : équilibrer chaque séance entre mobilité prioritaire, contrôle cervical/scapulaire, stabilité du tronc et intégration fonctionnelle." if lang=="fr" else "v2.1 strategy: balance each session across priority mobility, cervical/scapular control, core stability, and functional integration."
    }

def slot_cats(slot, s):
    primary, secondary, support = s["primary_categories"], s["secondary_categories"], s["support_categories"]
    if slot=="reset": return ["RB"]
    if slot=="recovery": return ["RB"]
    if slot=="mobility_primary": return [c for c in primary if c in ["TM","SH","HM","HS","AM"]] or ["TM","HM"]
    if slot=="mobility_secondary": return [c for c in secondary if c in ["TM","SH","HM","HS","AM"]] or ["SH","TM"]
    if slot=="activation": return [c for c in primary+secondary+support if c in ["SH","CC","CS","HM"]] or ["CC","SH","CS"]
    if slot=="stability": return [c for c in primary+secondary+support if c in ["CS","SH","BP","CC"]] or ["CS","SH"]
    if slot=="integration": return ["FI","CS","BP"]
    return primary+secondary+support

def slot_keywords(slot):
    if slot in ["reset","recovery"]: return ["recovery","breathing","release"]
    if slot.startswith("mobility"): return ["mobility"]
    if slot=="activation": return ["activation","motor_control","control","mobility_control"]
    if slot=="stability": return ["stability","activation_stability","control"]
    if slot=="integration": return ["integration","functional_strength","dynamic_balance"]
    return []

def anchors(library, strategy, r):
    out = {}
    for c in (strategy["primary_categories"]+strategy["secondary_categories"])[:4]:
        cand = [e for e in library if cat(e)==c and pain_ok(e,r) and diff(e)<=2 and any(k in obj(e) for k in ["mobility","control","activation","stability","recovery"])]
        cand.sort(key=lambda e:(diff(e), e.get("exercise_id","")))
        if cand:
            chain=[]; byid={e.get("exercise_id"):e for e in library}; cur=cand[0].get("exercise_id"); seen=set()
            while cur and cur in byid and cur not in seen and len(chain)<4:
                chain.append(cur); seen.add(cur); cur=byid[cur].get("progression_id","")
            out[c]=chain
    return out

def score_ex(ex, slot, week, priorities, strategy, r, anch, used, sess_counts, week_counts):
    eid = ex.get("exercise_id")
    if not eid or eid in used or not pain_ok(ex,r): return -9999
    if diff(ex) > WEEK_META[week][2]: return -999
    c = cat(ex); o = obj(ex); domains = exdomains(ex)
    allowed = slot_cats(slot, strategy)
    if c not in allowed:
        if slot=="integration" and c in ["FI","CS","BP"]: pass
        else: return -200
    sc = 35
    if any(k in o for k in slot_keywords(slot)): sc += 25
    elif slot=="stability" and ("activation_stability" in o or c=="CS"): sc += 15
    for i,p in enumerate(priorities[:6],1):
        deficit=max(0,100-fnum(p["score"],100))
        if p["id"] in domains: sc += deficit * {1:1.45,2:1.20,3:.9,4:.55}.get(i,.25)
        if c in DOMAIN_TO_CATS.get(p["id"],[]): sc += deficit * {1:.55,2:.45,3:.32,4:.18}.get(i,.08)
    chain = anch.get(c,[])
    if eid in chain:
        pos=chain.index(eid); target=min(max(week-1,0),len(chain)-1)
        sc += 30 - abs(pos-target)*8
        if pos==0 and week<=3: sc += 10
    # balance: avoid dominance
    if c!="RB" and sess_counts.get(c,0)>=1: sc -= 45
    if sess_counts.get(c,0)>=2: sc -= 120
    if c=="RB" and sess_counts.get(c,0)>=1 and slot!="recovery": sc -= 30
    if week_counts.get(c,0)>=4: sc -= 18
    if week_counts.get(c,0)>=6: sc -= 55
    # difficulty progression
    preferred = {1:[1,2],2:[2,3],3:[3,4],4:[3,4,5]}[week]
    if diff(ex) in preferred: sc += 12
    if slot=="stability" and "stability" not in o and "activation_stability" not in o and c!="CS": sc -= 18
    return sc

def select(library, slot, week, priorities, strategy, r, anch, used, sess_counts, week_counts):
    scored=[(score_ex(e,slot,week,priorities,strategy,r,anch,used,sess_counts,week_counts),e) for e in library]
    scored=[x for x in scored if x[0]>-900]
    scored.sort(key=lambda x:x[0], reverse=True)
    return scored[0][1] if scored else None

def loc(ex, slot, priorities, lang):
    fr,en = BLOCK_LABELS.get(slot,(slot,slot))
    related=[]
    for p in priorities[:4]:
        if cat(ex) in DOMAIN_TO_CATS.get(p["id"],[]) or p["id"] in exdomains(ex):
            related.append(p["label"] if lang=="fr" else p.get("label_en",p["label"]))
    why = ("Sélectionné pour soutenir : " if lang=="fr" else "Selected to support: ") + ", ".join(related[:3] or [ex.get("category_fr") or ex.get("category_en") or cat(ex)])
    return {
        "id":ex.get("exercise_id"),"block":slot,"block_label":fr if lang=="fr" else en,
        "name":ex.get("name_fr") if lang=="fr" else ex.get("name_en"),
        "name_fr":ex.get("name_fr"),"name_en":ex.get("name_en"),
        "category_code":cat(ex),"target":ex.get("category_fr") if lang=="fr" else ex.get("category_en"),
        "primary_objective":ex.get("primary_objective"),"difficulty":ex.get("difficulty_1_5"),"phase":ex.get("phase"),
        "equipment":ex.get("equipment",""),"sets":ex.get("sets",""),"reps_time":ex.get("reps_time",""),
        "tempo":ex.get("tempo",""),"rest":ex.get("rest",""),"frequency_per_week":ex.get("frequency_per_week",""),
        "coaching_cues":ex.get("coaching_cues",""),"common_errors":ex.get("common_errors",""),
        "clinical_rationale":ex.get("clinical_rationale",""),"why_in_this_program":why,
        "regression_id":ex.get("regression_id",""),"progression_id":ex.get("progression_id",""),
        "pain_rule":ex.get("pain_rule",""),"asymmetry_rule":ex.get("asymmetry_rule",""),
        "video_url":ex.get("video_url",""),"vimeo_url":ex.get("vimeo_url",""),"mp4_url":ex.get("mp4_url",""),"thumbnail_url":ex.get("thumbnail_url","")
    }


ENGINE_VERSION = "FlexiLab Clinical Prescription Engine v2.1.1"

def build_measured_alerts(screening_payload, lang="fr", limit=5):
    report = screening_payload.get("report", screening_payload) or {}
    alerts = []
    severity_rank = {"red": 0, "orange": 1, "yellow": 2, "green": 3}

    for section in report.get("sections", []) or []:
        for item in section.get("items", []) or []:
            rating = str(item.get("rating", "")).lower()
            if rating not in ["red", "orange", "yellow"]:
                continue
            alerts.append({
                "id": item.get("id"),
                "label": item.get("label") or item.get("label_fr") or item.get("label_en"),
                "label_fr": item.get("label_fr"),
                "label_en": item.get("label_en"),
                "value": item.get("value"),
                "unit": item.get("unit"),
                "severity": rating,
                "rating_label": item.get("rating_label"),
                "interpretation": item.get("short_insight") or item.get("short_insight_fr") or item.get("short_insight_en"),
                "thresholds": item.get("thresholds"),
            })

    alerts.sort(key=lambda x: severity_rank.get(x.get("severity"), 99))
    return alerts[:limit]


def recompute_clinical_balance(session):
    exercises = session.get("exercises", []) or []
    return {
        "categories": sorted(list({e.get("category_code") for e in exercises if e.get("category_code")})),
        "exercise_count": len(exercises),
    }


def _candidate_cc_for_week(exercise_library, week, readiness_payload):
    max_diff = WEEK_META.get(week, WEEK_META[1])[2]
    candidates = [
        e for e in exercise_library
        if cat(e) == "CC"
        and pain_ok(e, readiness_payload)
        and diff(e) <= max_diff
    ]
    candidates.sort(key=lambda e: (diff(e), e.get("exercise_id", "")))
    if not candidates:
        return None
    # Gradual exposure: earlier weeks use easiest options; later weeks can move further in the chain.
    idx = min(max(week - 1, 0), len(candidates) - 1)
    return candidates[idx]


def _replace_with_cervical(session, cc_exercise, priorities, lang, week):
    exercises = session.get("exercises", []) or []
    if any(e.get("category_code") == "CC" for e in exercises):
        return False

    # Preserve primary thoracic and core exposure. Replace secondary/redundant SH/RB first.
    candidate_indices = []

    for idx, ex in enumerate(exercises):
        c = ex.get("category_code")
        b = ex.get("block")
        if b == "reset":
            continue
        if c == "RB" and b == "recovery":
            candidate_indices.append((0, idx))
        elif c == "SH" and b in ["mobility_secondary", "activation", "stability"]:
            candidate_indices.append((1, idx))
        elif c == "FI" and week <= 2:
            candidate_indices.append((2, idx))
        elif c == "RB":
            candidate_indices.append((3, idx))

    if not candidate_indices:
        return False

    candidate_indices.sort(key=lambda x: x[0])
    _, idx = candidate_indices[0]
    old_block = exercises[idx].get("block", "activation")
    new_block = old_block if old_block in ["activation", "stability"] else "activation"
    exercises[idx] = loc(cc_exercise, new_block, priorities, lang)
    session["exercises"] = exercises
    session["clinical_balance"] = recompute_clinical_balance(session)
    session["blocks"] = [e.get("block") for e in exercises]
    session["estimated_duration_minutes"] = max(15, min(35, 5 + len(exercises) * 3))
    return True


def enforce_cervical_priority_protection(program, exercise_library, priorities, readiness_payload, lang="fr"):
    priority_ids = [p.get("id") for p in priorities[:3]]
    detected = "cervical_control" in priority_ids
    if not detected:
        return program

    min_per_week = 2
    total_added_or_present = 0
    weeks_failed = []

    for week in program.get("weeks", []) or []:
        week_number = int(week.get("week", 1))
        sessions = week.get("sessions", []) or []
        current = sum(
            1
            for s in sessions
            for e in s.get("exercises", []) or []
            if e.get("category_code") == "CC"
        )

        while current < min_per_week:
            cc_exercise = _candidate_cc_for_week(exercise_library, week_number, readiness_payload)
            if not cc_exercise:
                weeks_failed.append(week_number)
                break

            replaced = False
            for session in sessions:
                if current >= min_per_week:
                    break
                if _replace_with_cervical(session, cc_exercise, priorities, lang, week_number):
                    current += 1
                    replaced = True
                    total_added_or_present += 1

            if not replaced:
                weeks_failed.append(week_number)
                break

        week["sessions"] = sessions

    program.setdefault("validation_flags", {})
    program["validation_flags"]["cervical_priority_protection"] = {
        "detected": detected,
        "min_exposures_per_week": min_per_week,
        "weeks_failed": weeks_failed,
        "passed": len(weeks_failed) == 0,
        "note": "Direct CC exposure is required when cervical_control is a top-three clinical priority."
    }
    return program


def personalize_exercise_rationale(exercise, clinical_priorities, movement_dna, measured_alerts, lang="fr"):
    c = exercise.get("category_code")
    priority_ids = [p.get("id") for p in clinical_priorities or []]
    profile = (movement_dna or {}).get("primary_profile", "")

    if lang == "fr":
        if c == "TM":
            return "Inclus car la mobilité thoracique fait partie des priorités cliniques. L’objectif est d’améliorer la mobilité du haut du dos pour réduire les compensations cervicales, scapulaires et le contrôle excessif du tronc."
        if c == "CS":
            return "Inclus car la stabilité du tronc fait partie des priorités cliniques. Il vise le contrôle lombo-pelvien et la capacité à stabiliser le tronc pendant les mouvements fonctionnels."
        if c == "CC":
            return "Inclus car le contrôle cervical est une priorité de votre profil. L’objectif est un travail léger de positionnement cervical, d’endurance posturale et de réduction des tensions inutiles."
        if c == "SH":
            return "Inclus pour soutenir la mobilité des épaules et la mécanique scapulaire, en complément du travail thoracique et cervical."
        if c == "FI":
            return "Inclus pour transférer les gains de mobilité et de contrôle vers un mouvement fonctionnel proche des gestes quotidiens."
        if c == "RB":
            return "Inclus pour améliorer la respiration, diminuer les tensions inutiles et préparer un meilleur contrôle postural."
    else:
        if c == "TM":
            return "Included because thoracic mobility is one of the clinical priorities. The goal is to improve upper-back mobility and reduce cervical, scapular, and trunk compensations."
        if c == "CS":
            return "Included because core stability is one of the clinical priorities. It targets lumbopelvic control and trunk stability during functional movement."
        if c == "CC":
            return "Included because cervical control is a priority in this profile. The goal is low-load cervical positioning, postural endurance, and unnecessary tension reduction."
        if c == "SH":
            return "Included to support shoulder mobility and scapular mechanics alongside thoracic and cervical work."
        if c == "FI":
            return "Included to transfer mobility and control gains into functional movement."
        if c == "RB":
            return "Included to improve breathing mechanics, reduce unnecessary tension, and prepare better postural control."

    return exercise.get("why_in_this_program") or exercise.get("clinical_rationale") or ""


def enrich_program_rationales(program, lang="fr"):
    clinical_priorities = program.get("clinical_priorities") or program.get("main_priorities") or []
    movement_dna = program.get("movement_dna_summary") or {}
    measured_alerts = program.get("measured_alerts") or []
    for week in program.get("weeks", []) or []:
        for session in week.get("sessions", []) or []:
            for exercise in session.get("exercises", []) or []:
                exercise["why_in_this_program"] = personalize_exercise_rationale(
                    exercise, clinical_priorities, movement_dna, measured_alerts, lang
                )
    return program


def build_validation_flags(program):
    weeks = program.get("weeks", []) or []
    sessions = [s for w in weeks for s in (w.get("sessions", []) or [])]
    exercises = [e for s in sessions for e in (s.get("exercises", []) or [])]

    category_counts = {}
    for e in exercises:
        c = e.get("category_code")
        if c:
            category_counts[c] = category_counts.get(c, 0) + 1

    priority_ids = [p.get("id") for p in (program.get("clinical_priorities") or program.get("main_priorities") or [])]

    flags = {
        "engine_version": program.get("engine_version"),
        "has_weeks": bool(weeks),
        "week_count": len(weeks),
        "sessions_per_week": [len(w.get("sessions", []) or []) for w in weeks],
        "total_exercise_exposures": len(exercises),
        "category_counts": category_counts,
        "has_clinical_balance": all("clinical_balance" in s for s in sessions),
        "has_why_in_this_program": all(bool(e.get("why_in_this_program")) for e in exercises),
        "cervical_priority_detected": "cervical_control" in priority_ids[:3],
        "cervical_exposure_count_total": category_counts.get("CC", 0),
        "cervical_priority_protection_passed": (
            "cervical_control" not in priority_ids[:3] or category_counts.get("CC", 0) >= 6
        ),
        "core_priority_detected": "core_stability" in priority_ids[:3],
        "core_exposure_count_total": category_counts.get("CS", 0),
        "functional_integration_count_total": category_counts.get("FI", 0),
        "measured_alerts_present": bool(program.get("measured_alerts")),
        "clinical_priorities_present": bool(program.get("clinical_priorities")),
    }

    # Preserve any earlier validation details, such as protection-rule notes.
    existing = program.get("validation_flags") or {}
    existing.update(flags)
    return existing

def generate_clinical_prescription_v21(screening_payload, exercise_library, rules=None, movement_dna=None, language="fr"):
    report=screening_payload.get("report",screening_payload) or {}
    movement_score=fnum(report.get("flexilab_score", report.get("score",0)),0)
    intake=screening_payload.get("intake_context") or report.get("intake_context") or screening_payload.get("intake") or {}
    r=readiness(intake,language)
    pri=domains(screening_payload,language)
    strategy=build_strategy(pri,movement_dna,language)
    anch=anchors(exercise_library,strategy,r)
    weeks=[]
    for week in [1,2,3,4]:
        meta=WEEK_META[week]; sessions=[]; week_counts={}
        for day, template in enumerate(WEEK_TEMPLATES[week],1):
            selected=[]; used=set(); sess_counts={}
            for slot in template:
                ex=select(exercise_library,slot,week,pri,strategy,r,anch,used,sess_counts,week_counts)
                if not ex: continue
                used.add(ex.get("exercise_id")); sess_counts[cat(ex)]=sess_counts.get(cat(ex),0)+1; week_counts[cat(ex)]=week_counts.get(cat(ex),0)+1
                selected.append(loc(ex,slot,pri,language))
            cats={e["category_code"] for e in selected}
            # safeguard: ensure CS/CC included when priority is core/cervical
            if any(p["id"] in ["core_stability","cervical_control"] for p in pri[:3]) and not (cats & {"CS","CC"}) and len(selected)<8:
                for extra in ["activation","stability"]:
                    ex=select(exercise_library,extra,week,pri,strategy,r,anch,used,sess_counts,week_counts)
                    if ex and cat(ex) in ["CS","CC"]:
                        used.add(ex.get("exercise_id")); sess_counts[cat(ex)]=sess_counts.get(cat(ex),0)+1; week_counts[cat(ex)]=week_counts.get(cat(ex),0)+1
                        selected.append(loc(ex,extra,pri,language)); break
            focus_fr=["Mobilité / restauration","Contrôle / stabilité","Intégration / mouvement"][day-1]
            focus_en=["Mobility / restore","Control / stability","Integration / movement"][day-1]
            sessions.append({"day":day,"focus":focus_fr if language=="fr" else focus_en,"session_model":"balanced_block_based_v2_1","clinical_balance":{"categories":sorted(list({e["category_code"] for e in selected})),"exercise_count":len(selected)},"blocks":template,"estimated_duration_minutes":max(15,min(35,5+len(selected)*3)),"exercises":selected})
        weeks.append({"week":week,"phase":meta[0] if language=="fr" else meta[1],"objective":meta[3] if language=="fr" else meta[4],"difficulty_max":meta[2],"progression_logic":("Maintenir les fondations, ajouter progressivement contrôle, stabilité et intégration." if language=="fr" else "Maintain foundations while progressively adding control, stability, and integration."),"clinical_goal":strategy["strategy_text"],"sessions":sessions})
    clinical_priorities = pri[:3]
    measured_alerts = build_measured_alerts(screening_payload, language)
    program = {
        "engine_version": ENGINE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "language": language,
        "movement_score": movement_score,
        "movement_score_band": band(movement_score, language),
        "clinical_readiness": r,
        "movement_dna_summary": movement_dna or {},
        "clinical_strategy": strategy,
        "foundation_carryover": anch,
        "measured_alerts": measured_alerts,
        "clinical_priorities": clinical_priorities,
        # Backward-compatible alias. Frontend can migrate to clinical_priorities later.
        "main_priorities": clinical_priorities,
        "monitor_domains": pri[3:],
        "asymmetries": asymmetries(screening_payload, language),
        "program_summary": {
            "duration": "4 semaines" if language == "fr" else "4 weeks",
            "frequency": "3 séances/semaine" if language == "fr" else "3 sessions/week",
            "session_duration": "15-35 min",
            "model": "balanced block-based carryover + progression",
            "medical_advice_recommended": r.get("medical_advice_recommended", False),
        },
        "weeks": weeks,
        "safety_notes": [
            "Aucun exercice ne doit provoquer ou augmenter la douleur.",
            "La qualité du mouvement prime sur le nombre de répétitions.",
            "Réduire l’amplitude si une compensation apparaît.",
        ] if language == "fr" else [
            "No exercise should provoke or increase pain.",
            "Movement quality is more important than reps.",
            "Reduce range if compensation appears.",
        ],
        "reassessment_plan": {
            "when": "après 4 semaines" if language == "fr" else "after 4 weeks",
            "what": "Refaire le même screening et comparer les domaines, la symétrie et la douleur." if language == "fr" else "Repeat the same screening and compare domains, symmetry and pain.",
        },
        "integration_notes": {
            "import": "from engines.clinical_prescription_engine_v21 import generate_clinical_prescription_v21, load_exercise_library",
            "call": "generate_clinical_prescription_v21(screening_payload, EXERCISE_LIBRARY, PRESCRIPTION_RULES, movement_dna=movement_dna, language=lang)",
        },
    }

    program = enforce_cervical_priority_protection(
        program=program,
        exercise_library=exercise_library,
        priorities=clinical_priorities,
        readiness_payload=r,
        lang=language,
    )
    program = enrich_program_rationales(program, language)
    program["validation_flags"] = build_validation_flags(program)
    return program

generate_clinical_prescription = generate_clinical_prescription_v21

# -----------------------------------------------------------------------------
# FlexiLab v2.2 Clinical Reasoning Patch
# Purpose:
# - Do not present untested domains as measured strengths.
# - Add squat-compensation reasoning when trunk lean is red.
# - Add ankle support when trunk lean is red and ankle was not directly assessed.
# - Reduce breathing overuse.
# - Keep the existing public function name for app.py compatibility.
# -----------------------------------------------------------------------------
_generate_clinical_prescription_v211_base = generate_clinical_prescription_v21
ENGINE_VERSION = "FlexiLab Clinical Prescription Engine v2.2 Clinical Reasoning"

DIRECT_DOMAIN_TESTS = {
    "cervical_control": ["neck_angle"],
    "thoracic_mobility": ["thoracic_angle", "squat_trunk_lean"],
    "shoulder_mobility": ["shoulder_right_flexion", "shoulder_left_flexion"],
    "hip_mobility": ["aslr_right_angle", "aslr_left_angle", "squat_knee_angle"],
    "hamstring_mobility": ["aslr_right_angle", "aslr_left_angle"],
    "core_stability": ["squat_trunk_lean"],
    "ankle_mobility": ["ankle_dorsiflexion", "knee_to_wall", "ankle_right_dorsiflexion", "ankle_left_dorsiflexion"],
    "balance_proprioception": ["single_leg_balance", "balance_right", "balance_left"],
}

SUPPORT_DOMAIN_LABELS = {
    "ankle_mobility": {"fr": "Mobilité de cheville", "en": "Ankle Mobility"},
    "balance_proprioception": {"fr": "Équilibre & proprioception", "en": "Balance & Proprioception"},
}

TM_PROGRESSIONS = {1: ["TM001", "TM002"], 2: ["TM004", "TM006", "TM005"], 3: ["TM003", "TM006", "TM009"], 4: ["TM004", "TM008", "TM010"]}
CS_PROGRESSIONS = {1: ["CS001"], 2: ["CS002", "CS001"], 3: ["CS003", "CS004"], 4: ["CS004", "CS005", "CS006"]}
CC_PROGRESSIONS = {1: ["CC001"], 2: ["CC002"], 3: ["CC003", "CC009"], 4: ["CC004", "CC006"]}
AM_PROGRESSIONS = {1: ["AM001"], 2: ["AM002"], 3: ["AM003"], 4: ["AM004", "AM005"]}
RB_PROGRESSIONS = {1: ["RB001"], 2: ["RB002"], 3: ["RB003"], 4: ["RB004"]}


def _report_items(screening_payload):
    report = screening_payload.get("report", screening_payload) or {}
    out = {}
    for section in report.get("sections", []) or []:
        for item in section.get("items", []) or []:
            if item.get("id"):
                out[item.get("id")] = item
    return out


def _domain_is_directly_assessed(domain_id, screening_payload):
    items = _report_items(screening_payload)
    return any(test_id in items for test_id in DIRECT_DOMAIN_TESTS.get(domain_id, []))


def _split_assessed_domains(domains_list, screening_payload, lang="fr"):
    assessed, not_assessed = [], []
    for domain in domains_list or []:
        domain_id = domain.get("id")
        if _domain_is_directly_assessed(domain_id, screening_payload):
            assessed.append(domain)
        else:
            if domain_id in ["ankle_mobility", "balance_proprioception"]:
                labels = SUPPORT_DOMAIN_LABELS.get(domain_id, {})
                not_assessed.append({
                    "id": domain_id,
                    "label": labels.get("fr") if lang == "fr" else labels.get("en"),
                    "label_fr": labels.get("fr"),
                    "label_en": labels.get("en"),
                    "assessment_status": "not_assessed",
                    "score": None,
                    "reason": "Aucun test direct de ce domaine n’a été réalisé." if lang == "fr" else "No direct test for this domain was performed.",
                })
            else:
                assessed.append(domain)
    return assessed, not_assessed


def _metric_rating(screening_payload, metric_id):
    item = _report_items(screening_payload).get(metric_id) or {}
    return str(item.get("rating", "")).lower(), fnum(item.get("value"), 0)


def _squat_trunk_lean_red(screening_payload):
    rating, _ = _metric_rating(screening_payload, "squat_trunk_lean")
    return rating == "red"


def _extract_movement_pain_flags(screening_payload, lang="fr"):
    report = screening_payload.get("report", screening_payload) or {}
    flags = []
    for section in report.get("sections", []) or []:
        for item in section.get("items", []) or []:
            pain_value = None
            for key in ["pain_score", "pain", "discomfort_score", "discomfort"]:
                if key in item:
                    pain_value = item.get(key)
                    break
            pain_value = fnum(pain_value, 0)
            pain_flag = item.get("pain_flag") or item.get("discomfort_flag")
            if pain_value > 0 or pain_flag:
                level = "mild" if pain_value <= 3 else "moderate" if pain_value <= 6 else "severe"
                flags.append({
                    "test_id": item.get("id"),
                    "label": item.get("label") or item.get("label_fr") or item.get("label_en"),
                    "pain_score": pain_value,
                    "level": level,
                    "interpretation": (
                        "La douleur peut influencer la stratégie de mouvement; l’interprétation doit rester prudente."
                        if lang == "fr" else
                        "Pain can influence movement strategy; interpretation should remain cautious."
                    ),
                })
    return flags


def _build_reasoning(screening_payload, not_assessed_domains, lang="fr"):
    trunk_red = _squat_trunk_lean_red(screening_payload)
    pain_flags = _extract_movement_pain_flags(screening_payload, lang)
    reasoning = {
        "version": "FlexiLab Clinical Reasoning v2.2",
        "pain_is_gatekeeper": True,
        "movement_pain_flags": pain_flags,
        "not_assessed_domains": not_assessed_domains,
        "compensation_rules_applied": [],
    }
    if trunk_red:
        reasoning["compensation_rules_applied"].append({
            "id": "squat_trunk_lean_red",
            "measured_finding": "Inclinaison du tronc excessive en squat" if lang == "fr" else "Excessive trunk lean during squat",
            "possible_contributors": [
                "stabilité du tronc" if lang == "fr" else "trunk/core stability",
                "mobilité thoracique" if lang == "fr" else "thoracic mobility",
                "mobilité de hanche" if lang == "fr" else "hip mobility",
                "mobilité de cheville non confirmée" if lang == "fr" else "unconfirmed ankle mobility",
                "stratégie motrice / équilibre" if lang == "fr" else "motor strategy / balance",
            ],
            "prescription_logic": (
                "Renforcer le contrôle du tronc, maintenir la mobilité thoracique, ajouter un soutien cheville non diagnostique, puis intégrer dans des mouvements fonctionnels."
                if lang == "fr" else
                "Prioritize trunk control, maintain thoracic mobility, add non-diagnostic ankle support, then integrate into functional movement."
            ),
        })
    if pain_flags:
        reasoning["compensation_rules_applied"].append({
            "id": "pain_aware_interpretation",
            "prescription_logic": (
                "Les mouvements douloureux ne sont pas interprétés comme de simples déficits de mobilité ou stabilité; le programme reste sans douleur et à faible charge."
                if lang == "fr" else
                "Painful movements are not interpreted as simple mobility or stability deficits; the program remains pain-free and low-load."
            ),
        })
    return reasoning


def _by_id(exercise_library):
    return {e.get("exercise_id"): e for e in exercise_library or [] if e.get("exercise_id")}


def _candidate_from_ids(exercise_library, ids, fallback_cat, week, readiness_payload):
    byid = _by_id(exercise_library)
    max_diff = WEEK_META.get(int(week), WEEK_META[1])[2]
    for eid in ids:
        ex = byid.get(eid)
        if ex and pain_ok(ex, readiness_payload) and diff(ex) <= max_diff:
            return ex
    fallback = [e for e in exercise_library or [] if cat(e) == fallback_cat and pain_ok(e, readiness_payload) and diff(e) <= max_diff]
    fallback.sort(key=lambda e: (diff(e), e.get("exercise_id", "")))
    return fallback[0] if fallback else None


def _pick_progression_exercise(exercise_library, category_code, week, readiness_payload):
    table = {"TM": TM_PROGRESSIONS, "CS": CS_PROGRESSIONS, "CC": CC_PROGRESSIONS, "AM": AM_PROGRESSIONS, "RB": RB_PROGRESSIONS}.get(category_code, {})
    ids = table.get(int(week), [])
    return _candidate_from_ids(exercise_library, ids, category_code, week, readiness_payload)


def _replace_low_value_exercise(session, new_exercise, priorities, lang, preferred_block="activation", protected=("TM", "CS"), allow_replace=("RB", "SH", "FI")):
    exercises = session.get("exercises", []) or []
    if not exercises or not new_exercise:
        return False
    new_cat = cat(new_exercise)
    if any(e.get("category_code") == new_cat and e.get("id") == new_exercise.get("exercise_id") for e in exercises):
        return True
    candidates = []
    rb_count = sum(1 for e in exercises if e.get("category_code") == "RB")
    sh_count = sum(1 for e in exercises if e.get("category_code") == "SH")
    for idx, ex in enumerate(exercises):
        c = ex.get("category_code")
        b = ex.get("block")
        if c in protected:
            continue
        if c == "RB" and rb_count > 1:
            candidates.append((0, idx))
        elif c == "SH" and sh_count > 1:
            candidates.append((1, idx))
        elif c in allow_replace and b in ["recovery", "mobility_secondary", "activation", "stability"]:
            candidates.append((2, idx))
        elif c in allow_replace:
            candidates.append((3, idx))
    if not candidates:
        return False
    candidates.sort(key=lambda x: x[0])
    _, idx = candidates[0]
    old_block = exercises[idx].get("block") or preferred_block
    block = preferred_block if new_cat in ["AM", "CC", "CS"] else old_block
    exercises[idx] = loc(new_exercise, block, priorities, lang)
    session["exercises"] = exercises
    session["clinical_balance"] = recompute_clinical_balance(session)
    session["blocks"] = [e.get("block") for e in exercises]
    session["estimated_duration_minutes"] = max(15, min(35, 5 + len(exercises) * 3))
    return True


def _limit_breathing_volume(program, exercise_library, priorities, readiness_payload, lang="fr"):
    for week in program.get("weeks", []) or []:
        week_number = int(week.get("week", 1))
        for session in week.get("sessions", []) or []:
            exercises = session.get("exercises", []) or []
            rb_indices = [i for i, e in enumerate(exercises) if e.get("category_code") == "RB"]
            if len(rb_indices) <= 1:
                continue
            # Keep the first breathing/reset drill; replace additional breathing exposure.
            for idx in rb_indices[1:]:
                replacement = None
                if _squat_trunk_lean_red({"report": {}}):
                    replacement = None
                # Prefer trunk/ankle/functional replacement over another breathing drill.
                for cat_code in ["CS", "AM", "FI", "SH"]:
                    candidate = _pick_progression_exercise(exercise_library, cat_code, week_number, readiness_payload)
                    if candidate and not any(e.get("id") == candidate.get("exercise_id") for e in exercises):
                        replacement = candidate
                        break
                if replacement:
                    block = "activation" if cat(replacement) in ["AM", "CS"] else "integration" if cat(replacement) == "FI" else "mobility_secondary"
                    exercises[idx] = loc(replacement, block, priorities, lang)
            session["exercises"] = exercises
            session["clinical_balance"] = recompute_clinical_balance(session)
            session["blocks"] = [e.get("block") for e in exercises]
    return program


def _enforce_weekly_category_minimum(program, exercise_library, priorities, readiness_payload, lang, category_code, min_per_week, preferred_block="activation", protected=("TM", "CS")):
    failed = []
    for week in program.get("weeks", []) or []:
        week_number = int(week.get("week", 1))
        sessions = week.get("sessions", []) or []
        current = sum(1 for s in sessions for e in (s.get("exercises", []) or []) if e.get("category_code") == category_code)
        while current < min_per_week:
            candidate = _pick_progression_exercise(exercise_library, category_code, week_number, readiness_payload)
            if not candidate:
                failed.append(week_number)
                break
            replaced = False
            for session in sessions:
                if current >= min_per_week:
                    break
                if any(e.get("id") == candidate.get("exercise_id") for e in session.get("exercises", []) or []):
                    continue
                if _replace_low_value_exercise(session, candidate, priorities, lang, preferred_block=preferred_block, protected=protected):
                    current += 1
                    replaced = True
            if not replaced:
                failed.append(week_number)
                break
    return failed


def _enforce_progression_variety(program, exercise_library, priorities, readiness_payload, lang="fr"):
    # Replace excessive early repetitions with week-appropriate progressions for major domains.
    for week in program.get("weeks", []) or []:
        week_number = int(week.get("week", 1))
        for session in week.get("sessions", []) or []:
            exercises = session.get("exercises", []) or []
            for idx, ex in enumerate(exercises):
                c = ex.get("category_code")
                if c not in ["TM", "CS", "CC", "AM", "RB"]:
                    continue
                candidate = _pick_progression_exercise(exercise_library, c, week_number, readiness_payload)
                if not candidate:
                    continue
                # Keep variety inside same week by not duplicating the exact same candidate in every session when alternatives exist.
                if ex.get("id") == candidate.get("exercise_id"):
                    continue
                if c == "TM" and week_number >= 2:
                    exercises[idx] = loc(candidate, ex.get("block") or "mobility_primary", priorities, lang)
                elif c in ["CS", "CC", "AM"]:
                    exercises[idx] = loc(candidate, ex.get("block") or "activation", priorities, lang)
                elif c == "RB" and week_number >= 2:
                    exercises[idx] = loc(candidate, ex.get("block") or "reset", priorities, lang)
            session["exercises"] = exercises
            session["clinical_balance"] = recompute_clinical_balance(session)
            session["blocks"] = [e.get("block") for e in exercises]
    return program


def _update_v22_rationales(program, lang="fr"):
    reasoning = program.get("clinical_reasoning", {})
    trunk_rule = any(r.get("id") == "squat_trunk_lean_red" for r in reasoning.get("compensation_rules_applied", []) or [])
    for week in program.get("weeks", []) or []:
        for session in week.get("sessions", []) or []:
            for exercise in session.get("exercises", []) or []:
                c = exercise.get("category_code")
                if c == "AM" and trunk_rule:
                    exercise["why_in_this_program"] = (
                        "Ajouté comme soutien non diagnostique: une mobilité de cheville limitée peut contribuer à une inclinaison excessive du tronc en squat. Ce travail aide à explorer l’amplitude de cheville sans conclure qu’elle est déficitaire."
                        if lang == "fr" else
                        "Added as non-diagnostic support: limited ankle mobility can contribute to excessive trunk lean during squats. This explores ankle range without claiming an ankle deficit."
                    )
                elif c == "CS" and trunk_rule:
                    exercise["why_in_this_program"] = (
                        "Inclus car le squat montre une inclinaison excessive du tronc. L’objectif est d’améliorer le contrôle du tronc et du bassin avant d’augmenter la complexité du mouvement."
                        if lang == "fr" else
                        "Included because the squat shows excessive trunk lean. The goal is to improve trunk and pelvic control before increasing movement complexity."
                    )
                elif c == "RB":
                    exercise["why_in_this_program"] = (
                        "Utilisé brièvement comme reset pour diminuer les tensions inutiles et préparer un mouvement contrôlé."
                        if lang == "fr" else
                        "Used briefly as a reset to reduce unnecessary tension and prepare controlled movement."
                    )
    return program


def _build_program_quality_flags_v22(program):
    flags = build_validation_flags(program)
    category_counts = flags.get("category_counts", {}) or {}
    reasoning = program.get("clinical_reasoning", {}) or {}
    trunk_rule = any(r.get("id") == "squat_trunk_lean_red" for r in reasoning.get("compensation_rules_applied", []) or [])
    flags.update({
        "engine_version": program.get("engine_version"),
        "not_assessed_domains_present": bool(program.get("not_assessed_domains")),
        "ankle_not_assessed_handled": any(d.get("id") == "ankle_mobility" for d in program.get("not_assessed_domains", []) or []),
        "squat_compensation_reasoning_present": trunk_rule,
        "ankle_support_exposure_count_total": category_counts.get("AM", 0),
        "breathing_exposure_count_total": category_counts.get("RB", 0),
        "breathing_volume_reduced": category_counts.get("RB", 0) <= 10,
        "core_trunk_exposure_count_total": category_counts.get("CS", 0),
    })
    return flags


def generate_clinical_prescription_v21(screening_payload, exercise_library, rules=None, movement_dna=None, language="fr"):
    base_program = _generate_clinical_prescription_v211_base(screening_payload, exercise_library, rules=rules, movement_dna=movement_dna, language=language)
    original_domains = domains(screening_payload, language)
    assessed_domains, not_assessed_domains = _split_assessed_domains(original_domains, screening_payload, language)

    # Re-rank using only assessed domains. Untested domains remain visible as not assessed, never as strengths.
    base_program["engine_version"] = ENGINE_VERSION
    base_program["clinical_priorities"] = assessed_domains[:3]
    base_program["main_priorities"] = assessed_domains[:3]
    base_program["monitor_domains"] = assessed_domains[3:]
    base_program["not_assessed_domains"] = not_assessed_domains

    trunk_red = _squat_trunk_lean_red(screening_payload)
    base_program["clinical_reasoning"] = _build_reasoning(screening_payload, not_assessed_domains, language)

    # Update strategy labels.
    if base_program.get("clinical_strategy"):
        if language == "fr":
            base_program["clinical_strategy"]["strategy_text"] = "Stratégie v2.2 : raisonner à partir des compensations observées, distinguer les domaines mesurés des domaines non évalués, renforcer le contrôle du tronc, maintenir la mobilité thoracique, ajouter un soutien cheville si le squat montre une inclinaison excessive du tronc, puis intégrer progressivement."
        else:
            base_program["clinical_strategy"]["strategy_text"] = "v2.2 strategy: reason from observed compensations, separate measured from unassessed domains, prioritize trunk control, maintain thoracic mobility, add ankle support when squat trunk lean is excessive, then progressively integrate."

    readiness_payload = base_program.get("clinical_readiness") or readiness((screening_payload.get("report", screening_payload) or {}).get("intake_context") or {}, language)
    priorities = base_program.get("clinical_priorities") or []

    # Reduce excessive breathing before adding support work.
    base_program = _limit_breathing_volume(base_program, exercise_library, priorities, readiness_payload, language)

    failed_rules = {}
    if trunk_red:
        # Trunk lean red requires trunk stability, thoracic mobility, ankle support if ankle was not directly assessed.
        failed_rules["core_minimum"] = _enforce_weekly_category_minimum(base_program, exercise_library, priorities, readiness_payload, language, "CS", 2, preferred_block="stability", protected=("TM", "CC"))
        failed_rules["ankle_support_minimum"] = _enforce_weekly_category_minimum(base_program, exercise_library, priorities, readiness_payload, language, "AM", 1, preferred_block="activation", protected=("TM", "CS", "CC"))

    # Progress exercises by week rather than repeating the same entry drill too often.
    base_program = _enforce_progression_variety(base_program, exercise_library, priorities, readiness_payload, language)
    base_program = _update_v22_rationales(base_program, language)

    if failed_rules:
        base_program.setdefault("clinical_reasoning", {})["rule_failures"] = failed_rules

    base_program["program_summary"]["model"] = "compensation-aware clinical reasoning + progressive load model"
    base_program["program_summary"]["session_duration"] = "15-30 min"
    base_program["safety_notes"] = [
        "Aucun exercice ne doit provoquer ou augmenter la douleur.",
        "Si un mouvement est douloureux, réduire l’amplitude ou passer à l’option plus facile.",
        "Les domaines non testés ne sont pas présentés comme déficits confirmés.",
        "La qualité du mouvement prime sur le nombre de répétitions.",
    ] if language == "fr" else [
        "No exercise should provoke or increase pain.",
        "If a movement is painful, reduce range or switch to the easier option.",
        "Untested domains are not presented as confirmed deficits.",
        "Movement quality is more important than reps.",
    ]
    base_program["validation_flags"] = _build_program_quality_flags_v22(base_program)
    return base_program

generate_clinical_prescription = generate_clinical_prescription_v21

# Final v2.2 wrapper: reduce breathing dose by week after all protection rules.
_generate_clinical_prescription_v22_before_breathing_budget = generate_clinical_prescription_v21


def _reduce_breathing_by_week_budget(program, exercise_library, priorities, readiness_payload, lang="fr"):
    weekly_budget = {1: 3, 2: 2, 3: 2, 4: 1}
    for week in program.get("weeks", []) or []:
        week_number = int(week.get("week", 1))
        max_rb = weekly_budget.get(week_number, 1)
        rb_locations = []
        for si, session in enumerate(week.get("sessions", []) or []):
            for ei, exercise in enumerate(session.get("exercises", []) or []):
                if exercise.get("category_code") == "RB":
                    rb_locations.append((si, ei))
        # Keep earlier RBs only; replace later breathing drills with active support work.
        for si, ei in rb_locations[max_rb:]:
            session = week.get("sessions", [])[si]
            exercises = session.get("exercises", []) or []
            replacement = None
            for cat_code in ["AM", "CS", "FI", "SH"]:
                candidate = _pick_progression_exercise(exercise_library, cat_code, week_number, readiness_payload)
                if candidate and not any(e.get("id") == candidate.get("exercise_id") for e in exercises):
                    replacement = candidate
                    break
            if replacement:
                block = "activation" if cat(replacement) in ["AM", "CS"] else "integration" if cat(replacement) == "FI" else "mobility_secondary"
                exercises[ei] = loc(replacement, block, priorities, lang)
                session["exercises"] = exercises
                session["clinical_balance"] = recompute_clinical_balance(session)
                session["blocks"] = [e.get("block") for e in exercises]
                session["estimated_duration_minutes"] = max(15, min(30, 5 + len(exercises) * 3))
    return program


def generate_clinical_prescription_v21(screening_payload, exercise_library, rules=None, movement_dna=None, language="fr"):
    program = _generate_clinical_prescription_v22_before_breathing_budget(screening_payload, exercise_library, rules=rules, movement_dna=movement_dna, language=language)
    readiness_payload = program.get("clinical_readiness") or readiness((screening_payload.get("report", screening_payload) or {}).get("intake_context") or {}, language)
    priorities = program.get("clinical_priorities") or []
    program = _reduce_breathing_by_week_budget(program, exercise_library, priorities, readiness_payload, language)
    program = _update_v22_rationales(program, language)
    program["validation_flags"] = _build_program_quality_flags_v22(program)
    return program

generate_clinical_prescription = generate_clinical_prescription_v21

# Final v2.2 wrapper: keep shoulder support proportional when shoulder limitation is mild.
_generate_clinical_prescription_v22_before_shoulder_budget = generate_clinical_prescription_v21


def _reduce_category_by_week_budget(program, exercise_library, priorities, readiness_payload, lang="fr", category_code="SH", weekly_budget=None, replacement_order=None):
    weekly_budget = weekly_budget or {1: 3, 2: 2, 3: 2, 4: 1}
    replacement_order = replacement_order or ["AM", "CS", "FI"]
    for week in program.get("weeks", []) or []:
        week_number = int(week.get("week", 1))
        max_cat = weekly_budget.get(week_number, 1)
        locations = []
        for si, session in enumerate(week.get("sessions", []) or []):
            for ei, exercise in enumerate(session.get("exercises", []) or []):
                if exercise.get("category_code") == category_code:
                    locations.append((si, ei))
        for si, ei in locations[max_cat:]:
            session = week.get("sessions", [])[si]
            exercises = session.get("exercises", []) or []
            replacement = None
            for cat_code in replacement_order:
                candidate = _pick_progression_exercise(exercise_library, cat_code, week_number, readiness_payload)
                if candidate and not any(e.get("id") == candidate.get("exercise_id") for e in exercises):
                    replacement = candidate
                    break
            if replacement:
                block = "activation" if cat(replacement) in ["AM", "CS"] else "integration" if cat(replacement) == "FI" else "mobility_secondary"
                exercises[ei] = loc(replacement, block, priorities, lang)
                session["exercises"] = exercises
                session["clinical_balance"] = recompute_clinical_balance(session)
                session["blocks"] = [e.get("block") for e in exercises]
                session["estimated_duration_minutes"] = max(15, min(30, 5 + len(exercises) * 3))
    return program


def generate_clinical_prescription_v21(screening_payload, exercise_library, rules=None, movement_dna=None, language="fr"):
    program = _generate_clinical_prescription_v22_before_shoulder_budget(screening_payload, exercise_library, rules=rules, movement_dna=movement_dna, language=language)
    readiness_payload = program.get("clinical_readiness") or readiness((screening_payload.get("report", screening_payload) or {}).get("intake_context") or {}, language)
    priorities = program.get("clinical_priorities") or []
    # Mild shoulder findings should support the plan, not dominate it.
    program = _reduce_category_by_week_budget(program, exercise_library, priorities, readiness_payload, language, category_code="SH", weekly_budget={1: 3, 2: 2, 3: 2, 4: 1}, replacement_order=["AM", "CS", "FI"])
    program = _update_v22_rationales(program, language)
    program["validation_flags"] = _build_program_quality_flags_v22(program)
    program["validation_flags"]["shoulder_support_exposure_count_total"] = program["validation_flags"].get("category_counts", {}).get("SH", 0)
    program["validation_flags"]["shoulder_support_volume_controlled"] = program["validation_flags"]["shoulder_support_exposure_count_total"] <= 8
    return program

generate_clinical_prescription = generate_clinical_prescription_v21

# Final v2.2 wrapper: robust category budget using alternative exercises within a category.
_generate_clinical_prescription_v22_before_robust_budget = generate_clinical_prescription_v21


def _candidate_any_category(exercise_library, category_code, week, readiness_payload, session_exercises):
    max_diff = WEEK_META.get(int(week), WEEK_META[1])[2]
    used_ids = {e.get("id") for e in session_exercises or []}
    candidates = [e for e in exercise_library or [] if cat(e) == category_code and pain_ok(e, readiness_payload) and diff(e) <= max_diff and e.get("exercise_id") not in used_ids]
    candidates.sort(key=lambda e: (diff(e), e.get("exercise_id", "")))
    return candidates[0] if candidates else None


def _reduce_shoulder_robust(program, exercise_library, priorities, readiness_payload, lang="fr"):
    weekly_budget = {1: 3, 2: 2, 3: 2, 4: 1}
    for week in program.get("weeks", []) or []:
        week_number = int(week.get("week", 1))
        locations = []
        for si, session in enumerate(week.get("sessions", []) or []):
            for ei, exercise in enumerate(session.get("exercises", []) or []):
                if exercise.get("category_code") == "SH":
                    locations.append((si, ei))
        for si, ei in locations[weekly_budget.get(week_number, 1):]:
            session = week.get("sessions", [])[si]
            exercises = session.get("exercises", []) or []
            replacement = None
            for cat_code in ["AM", "CS", "FI"]:
                replacement = _candidate_any_category(exercise_library, cat_code, week_number, readiness_payload, exercises)
                if replacement:
                    break
            if replacement:
                block = "activation" if cat(replacement) in ["AM", "CS"] else "integration"
                exercises[ei] = loc(replacement, block, priorities, lang)
                session["exercises"] = exercises
                session["clinical_balance"] = recompute_clinical_balance(session)
                session["blocks"] = [e.get("block") for e in exercises]
                session["estimated_duration_minutes"] = max(15, min(30, 5 + len(exercises) * 3))
    return program


def generate_clinical_prescription_v21(screening_payload, exercise_library, rules=None, movement_dna=None, language="fr"):
    program = _generate_clinical_prescription_v22_before_robust_budget(screening_payload, exercise_library, rules=rules, movement_dna=movement_dna, language=language)
    readiness_payload = program.get("clinical_readiness") or readiness((screening_payload.get("report", screening_payload) or {}).get("intake_context") or {}, language)
    priorities = program.get("clinical_priorities") or []
    program = _reduce_shoulder_robust(program, exercise_library, priorities, readiness_payload, language)
    program = _update_v22_rationales(program, language)
    program["validation_flags"] = _build_program_quality_flags_v22(program)
    program["validation_flags"]["shoulder_support_exposure_count_total"] = program["validation_flags"].get("category_counts", {}).get("SH", 0)
    program["validation_flags"]["shoulder_support_volume_controlled"] = program["validation_flags"]["shoulder_support_exposure_count_total"] <= 8
    return program

generate_clinical_prescription = generate_clinical_prescription_v21

# Final v2.2 wrapper: ankle is support, not dominant.
_generate_clinical_prescription_v22_before_ankle_budget = generate_clinical_prescription_v21


def _reduce_ankle_support_budget(program, exercise_library, priorities, readiness_payload, lang="fr"):
    weekly_budget = {1: 1, 2: 1, 3: 1, 4: 2}
    for week in program.get("weeks", []) or []:
        week_number = int(week.get("week", 1))
        locations = []
        for si, session in enumerate(week.get("sessions", []) or []):
            for ei, exercise in enumerate(session.get("exercises", []) or []):
                if exercise.get("category_code") == "AM":
                    locations.append((si, ei))
        for si, ei in locations[weekly_budget.get(week_number, 1):]:
            session = week.get("sessions", [])[si]
            exercises = session.get("exercises", []) or []
            replacement = None
            for cat_code in ["CS", "FI", "SH"]:
                replacement = _candidate_any_category(exercise_library, cat_code, week_number, readiness_payload, exercises)
                if replacement:
                    break
            if replacement:
                block = "stability" if cat(replacement) == "CS" else "integration" if cat(replacement) == "FI" else "mobility_secondary"
                exercises[ei] = loc(replacement, block, priorities, lang)
                session["exercises"] = exercises
                session["clinical_balance"] = recompute_clinical_balance(session)
                session["blocks"] = [e.get("block") for e in exercises]
                session["estimated_duration_minutes"] = max(15, min(30, 5 + len(exercises) * 3))
    return program


def generate_clinical_prescription_v21(screening_payload, exercise_library, rules=None, movement_dna=None, language="fr"):
    program = _generate_clinical_prescription_v22_before_ankle_budget(screening_payload, exercise_library, rules=rules, movement_dna=movement_dna, language=language)
    readiness_payload = program.get("clinical_readiness") or readiness((screening_payload.get("report", screening_payload) or {}).get("intake_context") or {}, language)
    priorities = program.get("clinical_priorities") or []
    program = _reduce_ankle_support_budget(program, exercise_library, priorities, readiness_payload, language)
    program = _update_v22_rationales(program, language)
    program["validation_flags"] = _build_program_quality_flags_v22(program)
    program["validation_flags"]["shoulder_support_exposure_count_total"] = program["validation_flags"].get("category_counts", {}).get("SH", 0)
    program["validation_flags"]["shoulder_support_volume_controlled"] = program["validation_flags"]["shoulder_support_exposure_count_total"] <= 8
    program["validation_flags"]["ankle_support_volume_controlled"] = 4 <= program["validation_flags"].get("category_counts", {}).get("AM", 0) <= 6
    return program

generate_clinical_prescription = generate_clinical_prescription_v21
# -----------------------------------------------------------------------------
# FlexiLab V54 Clinical Reasoning Engine
# Backend-only wrapper around the existing v2.2 engine.
# Goals:
# - Keep the current API contract used by the HTML.
# - Add severity-aware and pain-aware loading rules.
# - Use bands/weights only when clinically appropriate by week.
# - Reduce repeated exercises within the same week.
# - Keep internal reasoning out of user-facing exercise text.
# -----------------------------------------------------------------------------
_generate_clinical_prescription_v54_base = generate_clinical_prescription_v21
ENGINE_VERSION = "FlexiLab Clinical Prescription Engine v5.4 Clinical Reasoning"

V54_CATEGORY_DOMAIN = {
    "CC": "cervical_control",
    "TM": "thoracic_mobility",
    "SH": "shoulder_mobility",
    "CS": "core_stability",
    "HM": "hip_mobility",
    "HS": "hamstring_mobility",
    "AM": "ankle_mobility",
    "BP": "balance_proprioception",
    "FI": "functional_integration",
    "RB": "recovery_breathing",
}

V54_ALLOWED_LOADING = {"none", "bodyweight", "foam_roller", "stick_or_pvc", "balance_pad", "elastic_band", "light_weight", "trx"}

V54_LOAD_ORDER = {
    "none": 0,
    "bodyweight": 0,
    "foam_roller": 0,
    "stick_or_pvc": 1,
    "balance_pad": 2,
    "elastic_band": 3,
    "trx": 4,
    "light_weight": 5,
}


def _v54_normalize_lang(language):
    return "en" if str(language or "").lower().startswith("en") else "fr"


def _v54_text(lang, fr, en):
    return en if _v54_normalize_lang(lang) == "en" else fr


def _v54_equipment(exercise):
    eq = str((exercise or {}).get("equipment") or "none").strip().lower()
    return eq if eq in V54_ALLOWED_LOADING else "none"


def _v54_pain_status(screening_payload):
    """
    User-facing pain states are: no_pain, discomfort, pain.
    This helper accepts future frontend payloads but remains safe with the
    current frontend, where pain may still be absent from stored intake_json.
    """
    report = (screening_payload or {}).get("report", screening_payload) or {}
    intake = (
        (screening_payload or {}).get("intake_context")
        or report.get("intake_context")
        or (screening_payload or {}).get("intake")
        or {}
    )
    values = []

    def collect(v):
        if v is None:
            return
        if isinstance(v, dict):
            for vv in v.values():
                collect(vv)
        elif isinstance(v, (list, tuple)):
            for vv in v:
                collect(vv)
        else:
            values.append(str(v).strip().lower())

    if isinstance(intake, dict):
        for key in ["pain_status", "pain", "pain_level", "movement_pain", "pain_clearance", "painClearance", "pain_by_test", "painByTest"]:
            collect(intake.get(key))
        for key in ["pain_score", "painIntensity"]:
            try:
                score = float(intake.get(key))
                if score >= 4:
                    values.append("pain")
                elif score > 0:
                    values.append("discomfort")
            except Exception:
                pass

    # Also accept future report item-level flags.
    for item in _report_items({"report": report}).values():
        for key in ["pain_status", "pain", "pain_flag", "discomfort", "discomfort_flag"]:
            collect(item.get(key))

    joined = " ".join(values)
    if any(x in joined for x in ["pain", "douleur", "douloureux", "true"]):
        # A literal 'no_pain' should not become pain.
        if "no_pain" in joined or "no pain" in joined or "aucune douleur" in joined:
            if "discomfort" not in joined and "inconfort" not in joined and "gêne" not in joined and "gene" not in joined:
                return "no_pain"
        return "pain"
    if any(x in joined for x in ["discomfort", "inconfort", "gêne", "gene", "mild"]):
        return "discomfort"
    return "no_pain"


def _v54_domain_scores(program, screening_payload):
    out = {}
    for d in (program or {}).get("clinical_priorities", []) or []:
        if d.get("id"):
            out[d.get("id")] = fnum(d.get("score"), 100)
    report = (screening_payload or {}).get("report", screening_payload) or {}
    for d in ((report.get("score_v2") or {}).get("domain_scores") or []):
        if d.get("id") and d.get("id") not in out:
            out[d.get("id")] = fnum(d.get("score"), 100)
    return out


def _v54_domain_severity(domain_id, domain_scores):
    if domain_id == "recovery_breathing":
        return "support"
    score = domain_scores.get(domain_id)
    if score is None:
        return "unassessed"
    score = fnum(score, 100)
    if score < 60:
        return "red"
    if score < 80:
        return "yellow"
    return "green"


def _v54_allowed_load_level(severity, pain_status, week):
    """
    Returns max loading score allowed this week.
    none/bodyweight/foam_roller=0, stick=1, balance=2, band=3, trx=4, light_weight=5.
    """
    week = int(week or 1)
    if pain_status == "pain":
        return 0
    if pain_status == "discomfort":
        if severity == "red":
            return 0 if week <= 3 else 1
        if severity == "yellow":
            return 0 if week <= 2 else 2 if week == 3 else 3
        return 1 if week <= 2 else 2 if week == 3 else 3
    # no pain
    if severity == "red":
        return 0 if week <= 2 else 2 if week == 3 else 3
    if severity == "yellow":
        return 0 if week == 1 else 2 if week == 2 else 3 if week == 3 else 5
    if severity == "green":
        return 1 if week == 1 else 3 if week == 2 else 5
    return 0 if week <= 2 else 2


def _v54_exercise_domain(exercise):
    return V54_CATEGORY_DOMAIN.get(cat(exercise), "functional_integration")


def _v54_is_load_allowed(exercise, week, pain_status, domain_scores):
    domain_id = _v54_exercise_domain(exercise)
    severity = _v54_domain_severity(domain_id, domain_scores)
    max_level = _v54_allowed_load_level(severity, pain_status, week)
    eq_level = V54_LOAD_ORDER.get(_v54_equipment(exercise), 0)
    if pain_status == "pain" and _v54_equipment(exercise) in ["elastic_band", "light_weight", "trx", "balance_pad"]:
        return False
    return eq_level <= max_level


def _v54_candidate_for_slot(exercise_library, category_code, week, pain_status, domain_scores, session_exercises, prefer_loaded=False):
    used_ids = {e.get("id") for e in session_exercises or []}
    max_diff = WEEK_META.get(int(week), WEEK_META[1])[2]
    candidates = []
    for e in exercise_library or []:
        if cat(e) != category_code:
            continue
        if e.get("exercise_id") in used_ids:
            continue
        if diff(e) > max_diff:
            continue
        if not _v54_is_load_allowed(e, week, pain_status, domain_scores):
            continue
        candidates.append(e)
    if not candidates:
        return None

    def score(e):
        eq = _v54_equipment(e)
        eq_level = V54_LOAD_ORDER.get(eq, 0)
        sc = 0
        if prefer_loaded:
            sc += eq_level * 10
        else:
            sc -= eq_level * 6
        # Prefer stability exercises in stability blocks and clinically simple options early.
        o = obj(e)
        if "stability" in o or "activation_stability" in o:
            sc += 18
        if "mobility" in o and int(week) <= 2:
            sc += 8
        sc -= abs(diff(e) - min(max(int(week), 1), 4)) * 3
        return sc

    candidates.sort(key=lambda e: (score(e), -diff(e), e.get("exercise_id", "")), reverse=True)
    return candidates[0]


def _v54_block_for_category(category_code, fallback="stability"):
    if category_code == "RB":
        return "reset"
    if category_code in ["TM", "SH", "HM", "HS", "AM"]:
        return "mobility_primary" if fallback.startswith("mobility") else fallback
    if category_code in ["CS", "CC", "BP"]:
        return "stability" if fallback in ["stability", "integration"] else "activation"
    if category_code == "FI":
        return "integration"
    return fallback


def _v54_recompute_session(session):
    session["clinical_balance"] = recompute_clinical_balance(session)
    session["blocks"] = [e.get("block") for e in session.get("exercises", []) or []]
    session["estimated_duration_minutes"] = max(15, min(35, 5 + len(session.get("exercises", []) or []) * 3))


def _v54_apply_load_gating(program, exercise_library, priorities, pain_status, domain_scores, lang="fr"):
    """Replace exercises whose equipment/load is too advanced for severity + pain + week."""
    for week in program.get("weeks", []) or []:
        week_number = int(week.get("week", 1))
        for session in week.get("sessions", []) or []:
            exercises = session.get("exercises", []) or []
            for idx, exercise in enumerate(list(exercises)):
                ex_id = exercise.get("id")
                source = next((e for e in exercise_library or [] if e.get("exercise_id") == ex_id), None)
                if not source:
                    continue
                if _v54_is_load_allowed(source, week_number, pain_status, domain_scores):
                    continue
                replacement = _v54_candidate_for_slot(
                    exercise_library,
                    exercise.get("category_code") or cat(source),
                    week_number,
                    pain_status,
                    domain_scores,
                    exercises,
                    prefer_loaded=False,
                )
                if replacement:
                    block = exercise.get("block") or _v54_block_for_category(cat(replacement))
                    exercises[idx] = loc(replacement, block, priorities, lang)
            session["exercises"] = exercises
            _v54_recompute_session(session)
    return program


def _v54_add_equipment_progression(program, exercise_library, priorities, pain_status, domain_scores, lang="fr"):
    """
    When allowed, introduce band/weight options in later weeks without forcing them too early.
    This upgrades one suitable stability/integration slot per session at most.
    """
    if pain_status == "pain":
        return program

    priority_domains = [p.get("id") for p in priorities[:4] if p.get("id")]
    priority_categories = []
    for domain_id in priority_domains:
        for c in DOMAIN_TO_CATS.get(domain_id, []):
            if c not in priority_categories:
                priority_categories.append(c)
    # Core and shoulder are the most relevant categories for band/weight stability progression.
    for c in ["CS", "SH", "FI", "HM", "AM"]:
        if c not in priority_categories:
            priority_categories.append(c)

    for week in program.get("weeks", []) or []:
        week_number = int(week.get("week", 1))
        for session in week.get("sessions", []) or []:
            exercises = session.get("exercises", []) or []
            has_loaded = any(_v54_equipment({"equipment": e.get("equipment")}) in ["elastic_band", "light_weight", "trx"] for e in exercises)
            if has_loaded:
                continue
            # Do not add loading before the policy allows it for the relevant domain.
            replacement = None
            replace_idx = None
            for cat_code in priority_categories:
                domain_id = V54_CATEGORY_DOMAIN.get(cat_code)
                severity = _v54_domain_severity(domain_id, domain_scores)
                max_level = _v54_allowed_load_level(severity, pain_status, week_number)
                if max_level < 3:
                    continue
                candidate = _v54_candidate_for_slot(
                    exercise_library,
                    cat_code,
                    week_number,
                    pain_status,
                    domain_scores,
                    exercises,
                    prefer_loaded=True,
                )
                if candidate and _v54_equipment(candidate) in ["elastic_band", "light_weight", "trx"]:
                    # Replace a same-category non-loaded stability/integration exercise when possible.
                    for i, ex in enumerate(exercises):
                        if ex.get("category_code") == cat_code and _v54_equipment({"equipment": ex.get("equipment")}) not in ["elastic_band", "light_weight", "trx"]:
                            replace_idx = i
                            replacement = candidate
                            break
                    if replacement:
                        break
            if replacement is not None and replace_idx is not None:
                old_block = exercises[replace_idx].get("block") or _v54_block_for_category(cat(replacement), "stability")
                exercises[replace_idx] = loc(replacement, old_block, priorities, lang)
                session["exercises"] = exercises
                _v54_recompute_session(session)
    return program


def _v54_reduce_week_duplicates(program, exercise_library, priorities, pain_status, domain_scores, lang="fr"):
    """Avoid exact same exercise appearing too many times in the same week."""
    protected_categories = {"RB"} if pain_status != "pain" else {"RB", "TM", "CC"}
    for week in program.get("weeks", []) or []:
        week_number = int(week.get("week", 1))
        seen = {}
        for session in week.get("sessions", []) or []:
            exercises = session.get("exercises", []) or []
            for idx, exercise in enumerate(list(exercises)):
                ex_id = exercise.get("id")
                category_code = exercise.get("category_code")
                seen[ex_id] = seen.get(ex_id, 0) + 1
                allowed_repeats = 2 if category_code in protected_categories else 1
                if seen.get(ex_id, 0) <= allowed_repeats:
                    continue
                replacement = _v54_candidate_for_slot(
                    exercise_library,
                    category_code,
                    week_number,
                    pain_status,
                    domain_scores,
                    exercises,
                    prefer_loaded=False,
                )
                if replacement and replacement.get("exercise_id") != ex_id:
                    block = exercise.get("block") or _v54_block_for_category(cat(replacement))
                    exercises[idx] = loc(replacement, block, priorities, lang)
                    seen[replacement.get("exercise_id")] = seen.get(replacement.get("exercise_id"), 0) + 1
            session["exercises"] = exercises
            _v54_recompute_session(session)
    return program


def _v54_sets_for_exercise(exercise, week, pain_status, domain_scores):
    category_code = exercise.get("category_code")
    domain_id = V54_CATEGORY_DOMAIN.get(category_code)
    severity = _v54_domain_severity(domain_id, domain_scores)
    block = str(exercise.get("block") or "")
    if pain_status == "pain":
        return "1 set" if block in ["reset", "recovery"] else "2 sets"
    if severity == "red":
        return "2 sets" if int(week) <= 2 else "3 sets"
    if severity == "yellow":
        return "2 sets" if int(week) == 1 else "3 sets"
    if block in ["integration", "stability"] and int(week) >= 3:
        return "3 sets"
    return "2 sets"


def _v54_clean_reps_time(raw, lang="fr"):
    raw = str(raw or "").strip()
    if not raw:
        return "6–8 repetitions" if lang == "en" else "6 à 8 répétitions"
    text = raw.replace("reps", "repetitions" if lang == "en" else "répétitions")
    text = text.replace("sec", "seconds" if lang == "en" else "secondes")
    text = text.replace("holds", "holds" if lang == "en" else "maintien")
    # Never let old combined prescriptions leak into this field.
    for marker in ["<br>", "Breathe", "Respirez", "sets", "séries", "series"]:
        if marker in text:
            text = text.split(marker)[0].strip()
    if not text:
        text = "6–8 repetitions" if lang == "en" else "6 à 8 répétitions"
    return text


def _v54_equipment_label(eq, lang="fr"):
    labels = {
        "none": ("Aucun", "None"),
        "bodyweight": ("Poids du corps", "Bodyweight"),
        "foam_roller": ("Foam roller", "Foam roller"),
        "stick_or_pvc": ("Bâton ou PVC", "Stick or PVC"),
        "balance_pad": ("Coussin d’équilibre", "Balance pad"),
        "elastic_band": ("Élastique", "Elastic band"),
        "light_weight": ("Charge légère", "Light weight"),
        "trx": ("Sangles/TRX", "TRX/suspension straps"),
    }
    fr, en = labels.get(str(eq or "none"), labels["none"])
    return en if lang == "en" else fr


def _v54_user_instruction(exercise, week, pain_status, domain_scores, lang="fr"):
    category_code = exercise.get("category_code")
    domain_id = V54_CATEGORY_DOMAIN.get(category_code)
    severity = _v54_domain_severity(domain_id, domain_scores)
    eq = exercise.get("equipment") or "none"
    loaded = eq in ["elastic_band", "light_weight", "trx"]
    if lang == "fr":
        if pain_status == "pain":
            return "Travaillez dans une amplitude confortable, sans provoquer la douleur. Le but est de relâcher les tensions et de restaurer un mouvement facile."
        if loaded:
            return "Utilisez une résistance légère et contrôlée. Gardez la qualité du mouvement prioritaire sur l’intensité."
        if severity == "red":
            return "Réalisez le mouvement lentement pour restaurer le contrôle et réduire les compensations."
        if severity == "yellow":
            return "Gardez un rythme contrôlé et cherchez une exécution fluide et symétrique."
        return "Conservez une exécution propre et contrôlée pour soutenir la progression globale."
    else:
        if pain_status == "pain":
            return "Work in a comfortable range without provoking pain. The goal is to reduce tension and restore easy movement."
        if loaded:
            return "Use light, controlled resistance. Movement quality is more important than intensity."
        if severity == "red":
            return "Move slowly to restore control and reduce compensations."
        if severity == "yellow":
            return "Keep a controlled rhythm and aim for smooth, symmetrical execution."
        return "Maintain clean, controlled execution to support the overall progression."


def _v54_user_tip(exercise, pain_status, lang="fr"):
    cues = str(exercise.get("coaching_cues") or "").strip()
    if cues:
        # Keep the first cue only for a cleaner table.
        first = cues.split(";")[0].strip()
        if first:
            return first[0].upper() + first[1:]
    return _v54_text(lang, "Respirez calmement et arrêtez si une douleur apparaît.", "Breathe calmly and stop if pain appears.")


def _v54_clean_user_fields(program, pain_status, domain_scores, lang="fr"):
    for week in program.get("weeks", []) or []:
        week_number = int(week.get("week", 1))
        # Never show internal strategy text as the client-facing weekly goal.
        week["clinical_goal"] = _v54_text(
            lang,
            "Améliorer le contrôle du mouvement, restaurer la mobilité et intégrer progressivement les corrections dans des exercices fonctionnels.",
            "Improve movement control, restore mobility, and progressively integrate corrections into functional exercises.",
        )
        for session in week.get("sessions", []) or []:
            for exercise in session.get("exercises", []) or []:
                eq = _v54_equipment(exercise)
                exercise["sets"] = _v54_sets_for_exercise(exercise, week_number, pain_status, domain_scores)
                exercise["reps_time"] = _v54_clean_reps_time(exercise.get("reps_time"), lang)
                exercise["equipment"] = eq
                exercise["equipment_label"] = _v54_equipment_label(eq, lang)
                exercise["instructions"] = _v54_user_instruction(exercise, week_number, pain_status, domain_scores, lang)
                exercise["tips"] = _v54_user_tip(exercise, pain_status, lang)
                # Replace internal selection text with concise client-facing rationale.
                exercise["why_in_this_program"] = exercise["instructions"]
    return program


def _v54_update_readiness(program, pain_status, lang="fr"):
    labels = {
        "no_pain": ("Aucune douleur", "No pain"),
        "discomfort": ("Gêne / inconfort", "Discomfort"),
        "pain": ("Douleur", "Pain"),
    }
    fr, en = labels.get(pain_status, labels["no_pain"])
    program.setdefault("clinical_readiness", {})
    program["clinical_readiness"].update({
        "pain_status": pain_status,
        "pain_status_label": en if lang == "en" else fr,
        "loading_allowed": pain_status != "pain",
        "loading_rule": _v54_text(
            lang,
            "La charge est introduite progressivement selon la sévérité du score et uniquement si le mouvement reste confortable.",
            "Loading is introduced progressively based on score severity and only when movement remains comfortable.",
        ),
    })
    return program


def _v54_quality_flags(program, pain_status, domain_scores):
    flags = _build_program_quality_flags_v22(program) if "_build_program_quality_flags_v22" in globals() else {}
    equipment_counts = {}
    duplicate_by_week = {}
    for week in program.get("weeks", []) or []:
        seen = {}
        for session in week.get("sessions", []) or []:
            for ex in session.get("exercises", []) or []:
                eq = ex.get("equipment") or "none"
                equipment_counts[eq] = equipment_counts.get(eq, 0) + 1
                eid = ex.get("id")
                seen[eid] = seen.get(eid, 0) + 1
        duplicate_by_week[str(week.get("week"))] = {k: v for k, v in seen.items() if v > 1}
    flags.update({
        "v54_clinical_reasoning_enabled": True,
        "v54_pain_status": pain_status,
        "v54_domain_scores_used": domain_scores,
        "v54_equipment_counts": equipment_counts,
        "v54_duplicate_exercises_by_week": duplicate_by_week,
    })
    return flags


def generate_clinical_prescription_v21(screening_payload, exercise_library, rules=None, movement_dna=None, language="fr"):
    lang = _v54_normalize_lang(language)
    program = _generate_clinical_prescription_v54_base(
        screening_payload,
        exercise_library,
        rules=rules,
        movement_dna=movement_dna,
        language=lang,
    )
    priorities = program.get("clinical_priorities") or program.get("main_priorities") or []
    domain_scores = _v54_domain_scores(program, screening_payload)
    pain_status = _v54_pain_status(screening_payload)

    program = _v54_apply_load_gating(program, exercise_library, priorities, pain_status, domain_scores, lang)
    program = _v54_reduce_week_duplicates(program, exercise_library, priorities, pain_status, domain_scores, lang)
    program = _v54_add_equipment_progression(program, exercise_library, priorities, pain_status, domain_scores, lang)
    program = _v54_clean_user_fields(program, pain_status, domain_scores, lang)
    program = _v54_update_readiness(program, pain_status, lang)

    program["engine_version"] = ENGINE_VERSION
    program["clinical_reasoning_version"] = "v54"
    program["selection_strategy"] = {
        "type": "severity_pain_equipment_progression",
        "pain_states": ["no_pain", "discomfort", "pain"],
        "principles": [
            "score severity controls progression timing",
            "pain prevents loaded stability work",
            "discomfort delays loading and reduces intensity",
            "red domains progress later than yellow domains",
            "bands and weights are introduced only when clinically appropriate",
            "duplicate exercises are reduced while preserving clinical anchors",
        ],
    }
    program["validation_flags"] = _v54_quality_flags(program, pain_status, domain_scores)
    return program


generate_clinical_prescription = generate_clinical_prescription_v21

# V54.1 refinement: week-level duplicate avoidance and safer pain parsing.
def _v54_pain_status(screening_payload):
    report = (screening_payload or {}).get("report", screening_payload) or {}
    intake = (
        (screening_payload or {}).get("intake_context")
        or report.get("intake_context")
        or (screening_payload or {}).get("intake")
        or {}
    )
    tokens = []

    def collect(v):
        if v is None:
            return
        if isinstance(v, dict):
            for vv in v.values():
                collect(vv)
        elif isinstance(v, (list, tuple)):
            for vv in v:
                collect(vv)
        else:
            tokens.append(str(v).strip().lower())

    if isinstance(intake, dict):
        for key in ["pain_status", "pain", "pain_level", "movement_pain", "pain_clearance", "painClearance", "pain_by_test", "painByTest"]:
            collect(intake.get(key))
        for key in ["pain_score", "painIntensity"]:
            try:
                score = float(intake.get(key))
                if score >= 4:
                    tokens.append("pain")
                elif score > 0:
                    tokens.append("discomfort")
            except Exception:
                pass
    for item in _report_items({"report": report}).values():
        for key in ["pain_status", "pain", "pain_flag", "discomfort", "discomfort_flag"]:
            collect(item.get(key))

    normalized = set()
    for t in tokens:
        if t in ["pain", "douleur", "true", "yes"] or ("douleur" in t and "aucune" not in t) or ("pain" in t and "no pain" not in t and "no_pain" not in t):
            normalized.add("pain")
        elif any(x in t for x in ["discomfort", "inconfort", "gêne", "gene", "mild"]):
            normalized.add("discomfort")
        elif t in ["no_pain", "no pain", "none", "aucune douleur", "false", "0"]:
            normalized.add("no_pain")
    if "pain" in normalized:
        return "pain"
    if "discomfort" in normalized:
        return "discomfort"
    return "no_pain"


def _v54_candidate_for_slot(exercise_library, category_code, week, pain_status, domain_scores, session_exercises, prefer_loaded=False, exclude_ids=None):
    used_ids = {e.get("id") for e in session_exercises or []}
    if exclude_ids:
        used_ids |= set(exclude_ids)
    max_diff = WEEK_META.get(int(week), WEEK_META[1])[2]
    candidates = []
    for e in exercise_library or []:
        if cat(e) != category_code:
            continue
        if e.get("exercise_id") in used_ids:
            continue
        if diff(e) > max_diff:
            continue
        if not _v54_is_load_allowed(e, week, pain_status, domain_scores):
            continue
        candidates.append(e)
    if not candidates:
        return None

    def score(e):
        eq = _v54_equipment(e)
        eq_level = V54_LOAD_ORDER.get(eq, 0)
        sc = eq_level * 10 if prefer_loaded else -eq_level * 6
        o = obj(e)
        if "stability" in o or "activation_stability" in o:
            sc += 18
        if "mobility" in o and int(week) <= 2:
            sc += 8
        sc -= abs(diff(e) - min(max(int(week), 1), 4)) * 3
        return sc

    candidates.sort(key=lambda e: (score(e), -diff(e), e.get("exercise_id", "")), reverse=True)
    return candidates[0]


def _v54_reduce_week_duplicates(program, exercise_library, priorities, pain_status, domain_scores, lang="fr"):
    protected_categories = {"RB"} if pain_status != "pain" else {"RB", "TM", "CC"}
    for week in program.get("weeks", []) or []:
        week_number = int(week.get("week", 1))
        seen_counts = {}
        week_used = set()
        for session in week.get("sessions", []) or []:
            exercises = session.get("exercises", []) or []
            for idx, exercise in enumerate(list(exercises)):
                ex_id = exercise.get("id")
                category_code = exercise.get("category_code")
                allowed_repeats = 2 if category_code in protected_categories else 1
                if seen_counts.get(ex_id, 0) >= allowed_repeats:
                    replacement = _v54_candidate_for_slot(
                        exercise_library,
                        category_code,
                        week_number,
                        pain_status,
                        domain_scores,
                        exercises,
                        prefer_loaded=False,
                        exclude_ids=week_used,
                    )
                    if replacement and replacement.get("exercise_id") != ex_id:
                        block = exercise.get("block") or _v54_block_for_category(cat(replacement))
                        exercise = loc(replacement, block, priorities, lang)
                        exercises[idx] = exercise
                        ex_id = replacement.get("exercise_id")
                seen_counts[ex_id] = seen_counts.get(ex_id, 0) + 1
                week_used.add(ex_id)
            session["exercises"] = exercises
            _v54_recompute_session(session)
    return program

# V54.2 final public wrapper: run duplicate control again after equipment progression.
def generate_clinical_prescription_v21(screening_payload, exercise_library, rules=None, movement_dna=None, language="fr"):
    lang = _v54_normalize_lang(language)
    program = _generate_clinical_prescription_v54_base(
        screening_payload,
        exercise_library,
        rules=rules,
        movement_dna=movement_dna,
        language=lang,
    )
    priorities = program.get("clinical_priorities") or program.get("main_priorities") or []
    domain_scores = _v54_domain_scores(program, screening_payload)
    pain_status = _v54_pain_status(screening_payload)

    program = _v54_apply_load_gating(program, exercise_library, priorities, pain_status, domain_scores, lang)
    program = _v54_reduce_week_duplicates(program, exercise_library, priorities, pain_status, domain_scores, lang)
    program = _v54_add_equipment_progression(program, exercise_library, priorities, pain_status, domain_scores, lang)
    program = _v54_reduce_week_duplicates(program, exercise_library, priorities, pain_status, domain_scores, lang)
    program = _v54_clean_user_fields(program, pain_status, domain_scores, lang)
    program = _v54_update_readiness(program, pain_status, lang)

    program["engine_version"] = ENGINE_VERSION
    program["clinical_reasoning_version"] = "v54.2"
    program["selection_strategy"] = {
        "type": "severity_pain_equipment_progression",
        "pain_states": ["no_pain", "discomfort", "pain"],
        "principles": [
            "score severity controls progression timing",
            "pain prevents loaded stability work",
            "discomfort delays loading and reduces intensity",
            "red domains progress later than yellow domains",
            "bands and weights are introduced only when clinically appropriate",
            "duplicate exercises are reduced while preserving clinical anchors",
        ],
    }
    program["validation_flags"] = _v54_quality_flags(program, pain_status, domain_scores)
    return program


generate_clinical_prescription = generate_clinical_prescription_v21

# -----------------------------------------------------------------------------
# FlexiLab V55 Clinical Reasoning Engine refinement
# Stronger material/equipment progression layer.
# This patch intentionally keeps the frontend stable. It changes only backend
# program generation so the existing program table can display meaningful
# material choices when clinically appropriate.
# -----------------------------------------------------------------------------
_generate_clinical_prescription_v55_base = generate_clinical_prescription_v21
ENGINE_VERSION = "FlexiLab Clinical Prescription Engine v5.5 Material-Aware Clinical Reasoning"

V55_LOADED_EQUIPMENT = {"elastic_band", "light_weight", "trx", "balance_pad"}
V55_VISIBLE_MATERIALS = {"elastic_band", "light_weight", "trx", "balance_pad", "foam_roller", "stick_or_pvc"}


def _v55_equipment_allowed_for_domain(domain_id, week_number, pain_status, domain_scores, eq):
    """Score/pain-based loading permission, using the user's 3 pain states."""
    severity = _v54_domain_severity(domain_id, domain_scores)
    week_number = int(week_number or 1)
    eq = _v54_equipment({"equipment": eq})

    if eq in ["none", "bodyweight"]:
        return True

    # Pain: no external load. Recovery tools only.
    if pain_status == "pain":
        return eq in ["foam_roller", "stick_or_pvc"]

    # Discomfort: no weights/TRX. Band/balance only late and only if not red.
    if pain_status == "discomfort":
        if eq in ["light_weight", "trx"]:
            return False
        if eq in ["elastic_band", "balance_pad"]:
            return week_number >= 4 and severity in ["yellow", "green", "support", "unassessed"]
        return True

    # No pain: controlled progression by severity.
    if severity == "red":
        # Red domains earn external resistance only at the end, and usually band/balance first.
        if eq in ["elastic_band", "balance_pad"]:
            return week_number >= 4
        if eq in ["light_weight", "trx"]:
            return week_number >= 4 and domain_id in ["core_stability", "functional_integration"]
        return True

    if severity == "yellow":
        if eq in ["elastic_band", "balance_pad"]:
            return week_number >= 3
        if eq in ["light_weight", "trx"]:
            return week_number >= 4
        return True

    # Green / support / unassessed supportive domains.
    if eq in ["elastic_band", "balance_pad"]:
        return week_number >= 2
    if eq in ["light_weight", "trx"]:
        return week_number >= 3
    return True


def _v55_candidate_pool(exercise_library, week_number, pain_status, domain_scores, preferred_categories=None, preferred_equipment=None):
    preferred_categories = preferred_categories or []
    preferred_equipment = preferred_equipment or []
    max_diff = WEEK_META.get(int(week_number), WEEK_META[1])[2]
    pool = []
    for e in exercise_library or []:
        eq = _v54_equipment(e)
        if preferred_equipment and eq not in preferred_equipment:
            continue
        if preferred_categories and cat(e) not in preferred_categories:
            continue
        if diff(e) > max_diff:
            continue
        domain_id = _v54_exercise_domain(e)
        if not _v55_equipment_allowed_for_domain(domain_id, week_number, pain_status, domain_scores, eq):
            continue
        if not pain_ok(e, {"program_mode": "corrective" if pain_status == "no_pain" else "recovery_control" if pain_status == "pain" else "pain_free_corrective"}):
            continue
        pool.append(e)
    return pool


def _v55_priority_categories(priorities):
    cats = []
    for p in (priorities or [])[:5]:
        for c in DOMAIN_TO_CATS.get(p.get("id"), []):
            if c not in cats:
                cats.append(c)
    # Add categories where materials exist and are useful for stability/integration.
    for c in ["CS", "SH", "FI", "HM", "AM", "BP", "TM", "RB"]:
        if c not in cats:
            cats.append(c)
    return cats


def _v55_replace_index_for_material(session_exercises):
    """Prefer replacing a stability/integration/activation slot, not reset/recovery."""
    preferred_blocks = ["stability", "integration", "activation", "mobility_secondary", "mobility_primary"]
    for block in preferred_blocks:
        for i, ex in enumerate(session_exercises or []):
            if ex.get("block") == block and _v54_equipment({"equipment": ex.get("equipment")}) not in V55_VISIBLE_MATERIALS:
                return i
    for i, ex in enumerate(session_exercises or []):
        if ex.get("block") not in ["reset", "recovery"] and _v54_equipment({"equipment": ex.get("equipment")}) not in V55_VISIBLE_MATERIALS:
            return i
    return None


def _v55_select_material_candidate(exercise_library, week_number, pain_status, domain_scores, priorities, existing_ids):
    priority_categories = _v55_priority_categories(priorities)

    if pain_status == "pain":
        equipment_order = ["foam_roller", "stick_or_pvc"]
        category_order = [c for c in priority_categories if c in ["RB", "TM", "HM", "HS", "AM", "CC"]] or priority_categories
    elif pain_status == "discomfort":
        equipment_order = ["elastic_band", "balance_pad", "foam_roller", "stick_or_pvc"]
        category_order = [c for c in priority_categories if c in ["CS", "SH", "AM", "BP", "TM", "HM", "FI", "RB"]]
    else:
        if int(week_number) <= 1:
            equipment_order = ["foam_roller", "stick_or_pvc"]
        elif int(week_number) == 2:
            equipment_order = ["elastic_band", "balance_pad", "foam_roller", "stick_or_pvc"]
        elif int(week_number) == 3:
            equipment_order = ["elastic_band", "balance_pad", "light_weight", "trx"]
        else:
            equipment_order = ["elastic_band", "light_weight", "trx", "balance_pad"]
        category_order = [c for c in priority_categories if c in ["CS", "SH", "FI", "HM", "AM", "BP", "TM", "RB"]]

    pool = _v55_candidate_pool(
        exercise_library,
        week_number,
        pain_status,
        domain_scores,
        preferred_categories=category_order,
        preferred_equipment=equipment_order,
    )
    if not pool:
        return None

    existing_ids = set(existing_ids or [])

    def score(e):
        eq = _v54_equipment(e)
        c = cat(e)
        o = obj(e)
        sc = 0
        # Prefer first available equipment type by week/pain policy.
        try:
            sc += (len(equipment_order) - equipment_order.index(eq)) * 30
        except ValueError:
            pass
        try:
            sc += (len(category_order) - category_order.index(c)) * 6
        except ValueError:
            pass
        if e.get("exercise_id") in existing_ids:
            sc -= 100
        if "stability" in o or "activation_stability" in o:
            sc += 22
        if "functional" in o or "integration" in o:
            sc += 10
        if int(week_number) <= 2 and ("mobility" in o or "recovery" in o):
            sc += 8
        sc -= abs(diff(e) - min(int(week_number), 4)) * 2
        return sc

    pool.sort(key=lambda e: (score(e), -diff(e), e.get("exercise_id", "")), reverse=True)
    return pool[0]


def _v55_force_visible_materials(program, exercise_library, priorities, pain_status, domain_scores, lang="fr"):
    """
    Ensure at least some clinically appropriate materials appear in the 4-week plan.
    This does not add material blindly; it follows severity + pain + week rules.
    """
    if not program or not exercise_library:
        return program

    global_ids = set()
    material_sessions = 0
    for week in program.get("weeks", []) or []:
        week_number = int(week.get("week", 1))
        # Desired material exposure by week. Week 1 can still be no-equipment.
        if pain_status == "pain":
            desired_sessions = 1 if week_number >= 1 else 0  # recovery tools only, when available
        elif pain_status == "discomfort":
            desired_sessions = 1 if week_number >= 4 else 0
        else:
            desired_sessions = 0 if week_number == 1 else 1 if week_number == 2 else 2

        week_material_count = 0
        for session in week.get("sessions", []) or []:
            exercises = session.get("exercises", []) or []
            for ex in exercises:
                global_ids.add(ex.get("id"))
            if any(_v54_equipment({"equipment": ex.get("equipment")}) in V55_VISIBLE_MATERIALS for ex in exercises):
                week_material_count += 1

        if week_material_count >= desired_sessions:
            continue

        for session in week.get("sessions", []) or []:
            if week_material_count >= desired_sessions:
                break
            exercises = session.get("exercises", []) or []
            if any(_v54_equipment({"equipment": ex.get("equipment")}) in V55_VISIBLE_MATERIALS for ex in exercises):
                continue
            replace_idx = _v55_replace_index_for_material(exercises)
            if replace_idx is None:
                continue
            candidate = _v55_select_material_candidate(
                exercise_library,
                week_number,
                pain_status,
                domain_scores,
                priorities,
                global_ids | {ex.get("id") for ex in exercises},
            )
            if not candidate:
                continue
            old_block = exercises[replace_idx].get("block") or _v54_block_for_category(cat(candidate), "stability")
            exercises[replace_idx] = loc(candidate, old_block, priorities, lang)
            global_ids.add(candidate.get("exercise_id"))
            session["exercises"] = exercises
            _v54_recompute_session(session)
            week_material_count += 1
            material_sessions += 1

    program["v55_material_sessions_forced"] = material_sessions
    return program


def _v55_enrich_material_fields(program, lang="fr"):
    for week in program.get("weeks", []) or []:
        for session in week.get("sessions", []) or []:
            for exercise in session.get("exercises", []) or []:
                eq = _v54_equipment(exercise)
                exercise["equipment"] = eq
                exercise["equipment_label"] = _v54_equipment_label(eq, lang)
                # Extra aliases for frontend display compatibility.
                exercise["material"] = exercise["equipment_label"]
                exercise["material_label"] = exercise["equipment_label"]
    return program


def generate_clinical_prescription_v21(screening_payload, exercise_library, rules=None, movement_dna=None, language="fr"):
    lang = _v54_normalize_lang(language)
    program = _generate_clinical_prescription_v55_base(
        screening_payload,
        exercise_library,
        rules=rules,
        movement_dna=movement_dna,
        language=lang,
    )
    priorities = program.get("clinical_priorities") or program.get("main_priorities") or []
    domain_scores = _v54_domain_scores(program, screening_payload)
    pain_status = _v54_pain_status(screening_payload)

    program = _v55_force_visible_materials(program, exercise_library, priorities, pain_status, domain_scores, lang)
    program = _v54_reduce_week_duplicates(program, exercise_library, priorities, pain_status, domain_scores, lang)
    program = _v54_clean_user_fields(program, pain_status, domain_scores, lang)
    program = _v55_enrich_material_fields(program, lang)
    program = _v54_update_readiness(program, pain_status, lang)

    equipment_counts = {}
    for week in program.get("weeks", []) or []:
        for session in week.get("sessions", []) or []:
            for exercise in session.get("exercises", []) or []:
                eq = _v54_equipment(exercise)
                equipment_counts[eq] = equipment_counts.get(eq, 0) + 1

    program["engine_version"] = ENGINE_VERSION
    program["clinical_reasoning_version"] = "v55_material_aware"
    program["validation_flags"] = _v54_quality_flags(program, pain_status, domain_scores)
    program["validation_flags"]["v55_equipment_counts"] = equipment_counts
    program["validation_flags"]["v55_material_sessions_forced"] = program.get("v55_material_sessions_forced", 0)
    program["selection_strategy"] = {
        "type": "v55_material_aware_clinical_reasoning",
        "pain_states": ["no_pain", "discomfort", "pain"],
        "principles": [
            "pain blocks bands, weights, TRX and loaded stability",
            "discomfort delays loading and avoids weights",
            "red domains receive external resistance late",
            "yellow domains can receive band work from week 3 when pain-free",
            "materials are intentionally introduced when clinically appropriate",
        ],
        "user_facing_summary": _v54_text(
            lang,
            "Votre programme progresse de la mobilité vers le contrôle, puis vers la stabilité et l’intégration fonctionnelle avec du matériel uniquement lorsque c’est approprié.",
            "Your program progresses from mobility to control, then to stability and functional integration, using equipment only when appropriate."
        ),
    }
    return program


generate_clinical_prescription = generate_clinical_prescription_v21
