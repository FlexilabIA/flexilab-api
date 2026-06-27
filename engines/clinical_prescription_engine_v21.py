
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
