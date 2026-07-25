from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import json, math, re

ENGINE_VERSION = "FlexiLab Clinical Prescription Engine v3.1-explicit-progression"

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

def _questionnaire(payload):
    payload = payload or {}
    report = payload.get("report", payload) or {}
    candidates = [
        payload.get("intake_context"), payload.get("questionnaire"),
        payload.get("questionnaire_json"), payload.get("pre_screening_questionnaire"),
        payload.get("intake"), report.get("intake_context"), report.get("questionnaire")
    ]
    for q in candidates:
        if isinstance(q, str):
            try: q = json.loads(q)
            except Exception: q = {}
        if isinstance(q, dict) and q: return q
    return {}

def _pain_state(q):
    raw = str(q.get("pain_level") or q.get("pain_status") or q.get("pain") or "no_pain").lower()
    score = _num(q.get("pain_score", q.get("painIntensity", 0)))
    restriction = str(q.get("medical_restriction") or "no").lower()
    if restriction not in {"", "no", "none", "false", "0"}:
        return "pain"
    if score >= 4 or raw in {"pain","moderate","severe","high"}: return "pain"
    if score > 0 or raw in {"discomfort","mild","caution"}: return "discomfort"
    return "no_pain"

def _experience(q):
    raw = str(q.get("activity_level") or q.get("training_level") or q.get("experience") or "moderate").lower()
    if raw in {"low","sedentary","beginner","inactive"}: return "beginner"
    if raw in {"high","advanced","athlete","very_active"}: return "advanced"
    return "intermediate"

def _available_equipment(q):
    raw = q.get("available_equipment") or q.get("equipment") or q.get("materials")
    if not raw:
        return {"none","bodyweight","foam_roller","massage_ball","elastic_band","light_weight","balance_pad","stick_or_pvc","trx"}
    vals = set(_csv(raw))
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
    areas = q.get("tension_areas") or []
    if isinstance(areas, str): areas = _csv(areas)
    primary = q.get("primary_tension_area")
    cats = []
    for area in ([primary] if primary else []) + list(areas):
        for c in mapping.get(str(area), []):
            if c not in cats: cats.append(c)
    return cats

def _exercise_domains(ex):
    return _csv(ex.get("screening_domains_improved"))

def _load_allowed(ex, week, pain, experience, equipment):
    eq = str(ex.get("equipment") or "none").lower()
    if eq not in equipment and eq not in {"none","bodyweight"}: return False
    load = int(_num(ex.get("load_level_v3"), 0))
    if pain == "pain" and load >= 3: return False
    if pain == "discomfort" and load >= 5: return False
    earliest = 1
    if eq == "elastic_band": earliest = 2 if pain=="no_pain" else 3
    if eq == "light_weight":
        earliest = {"beginner":4,"intermediate":3,"advanced":2}[experience]
        if pain != "no_pain": earliest = 99
    return week >= earliest

def _stage_allowed(ex, week):
    return int(_num(ex.get("progression_stage_v3", ex.get("min_week",1)),1)) <= week and int(_num(ex.get("difficulty_1_5",3),3)) <= {1:2,2:3,3:4,4:5}[week]

def _score_ex(ex, week, day, priorities, qcats, pain, experience, equipment,
              session_used, week_counts, program_counts, previous_week_ids):
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
    if week_counts[eid] >= weekly_cap or program_counts[eid] >= program_cap: return -1e9

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
    if eid in previous_week_ids: score -= 14
    if week_counts[eid] > 0: score -= 22
    if program_counts[eid] > 1: score -= 8 * program_counts[eid]
    return score

def _dose(ex, week, pain, experience, lang):
    dtype = ex.get("dosage_type") or "dynamic_mobility_reps"
    min_sets = int(_num(ex.get("min_sets_v3"),1))
    default_sets = int(_num(ex.get("default_sets_v3"),min_sets))
    max_sets = int(_num(ex.get("max_sets_v3"),default_sets))

    if dtype in {"recovery_hold","soft_tissue_time"}:
        sets = 1
    elif pain == "pain":
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
    if similarities and max(similarities) > .55: failures.append("within_week_session_similarity")
    if recovery > 8: failures.append("recovery_overuse")
    if pain=="no_pain" and loaded_by_week[3]+loaded_by_week[4] == 0: failures.append("missing_loaded_progression")
    return {
        "passed": not failures,
        "failures": failures,
        "max_within_week_similarity": round(max(similarities) if similarities else 0,2),
        "recovery_exposure_total": recovery,
        "loaded_exposures_by_week": dict(loaded_by_week),
    }

def generate_clinical_prescription_v21(screening_payload, exercise_library, rules=None, movement_dna=None, language="fr"):
    lang=_lang(language)
    q=_questionnaire(screening_payload)
    pain=_pain_state(q)
    experience=_experience(q)
    equipment=_available_equipment(q)
    priorities=_domains(screening_payload,lang)
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
        week_ids=set()
        for day in range(1,4):
            selected=[]
            used=set()
            role_targets = {
                1:["mobility","mobility","activation","stability","integration","recovery"],
                2:["activation","stability","stability","integration","mobility","integration"],
                3:["integration","integration","stability","mobility","activation","recovery"],
            }[day]
            for desired_role in role_targets:
                candidates=[]
                for ex in exercise_library:
                    if (ex.get("intervention_role") or "mobility") != desired_role: continue
                    sc=_score_ex(ex,week,day,priorities,qcats,pain,experience,equipment,used,week_counts,program_counts,previous_week_ids)
                    candidates.append((sc,ex))
                candidates.sort(key=lambda x:(x[0], str(x[1].get("exercise_id"))), reverse=True)
                ex=next((x for sc,x in candidates if sc > -1e8),None)
                if ex:
                    selected.append(_loc(ex,week,day,priorities,pain,experience,lang))
                    eid=ex.get("exercise_id"); used.add(eid); week_counts[eid]+=1; program_counts[eid]+=1; week_ids.add(eid)

            # Recovery exposure adapts to the reported pain state instead of
            # occupying corrective slots indiscriminately.
            recovery_cap = {"no_pain": 1, "discomfort": 2, "pain": 3}[pain]
            if sum(1 for e in selected if e["intervention_role"]=="recovery") > recovery_cap:
                kept=[]; recovery_seen=0
                for e in selected:
                    if e["intervention_role"]=="recovery":
                        recovery_seen += 1
                        if recovery_seen > recovery_cap: continue
                    kept.append(e)
                selected=kept

            # Guarantee a complete 5-7 exercise session without filling it with random recovery drills.
            while len(selected) < 6:
                category_counts = Counter(e.get("category_code") for e in selected)
                fallback=[]
                for ex in exercise_library:
                    role = ex.get("intervention_role") or "mobility"
                    recovery_cap = {"no_pain": 1, "discomfort": 2, "pain": 3}[pain]
                    if role == "recovery" and sum(1 for e in selected if e.get("intervention_role")=="recovery") >= recovery_cap:
                        continue
                    if category_counts.get(ex.get("category_code"), 0) >= 2:
                        continue
                    sc=_score_ex(ex,week,day,priorities,qcats,pain,experience,equipment,used,week_counts,program_counts,previous_week_ids)
                    if role in {"stability","integration"} and day in {2,3}: sc += 12
                    fallback.append((sc,ex))
                fallback.sort(key=lambda x:(x[0], str(x[1].get("exercise_id"))), reverse=True)
                ex=next((x for sc,x in fallback if sc > -1e8),None)
                if not ex: break
                selected.append(_loc(ex,week,day,priorities,pain,experience,lang))
                eid=ex.get("exercise_id"); used.add(eid); week_counts[eid]+=1; program_counts[eid]+=1; week_ids.add(eid)

            sessions.append({
                "day":day, "week":week,
                "focus":focuses[day][0] if lang=="en" else focuses[day][1],
                "session_model":"clinical_progression_engagement_v3",
                "estimated_duration_minutes":min(30, 5+len(selected)*3),
                "exercises":selected,
                "clinical_balance":{"exercise_count":len(selected),"categories":sorted({e["category_code"] for e in selected})},
            })
        weeks.append({
            "week":week,
            "phase":phases[week][0] if lang=="en" else phases[week][1],
            "objective":(
                ["Learn pain-free movement foundations.","Develop active control and capacity.","Introduce resistance and stronger stability demands.","Integrate corrections into challenging functional movement."][week-1]
                if lang=="en" else
                ["Apprendre les bases du mouvement sans douleur.","Développer le contrôle actif et la capacité.","Introduire la résistance et renforcer la stabilité.","Intégrer les corrections dans des mouvements fonctionnels plus exigeants."][week-1]
            ),
            "progression_logic":"Learn → control → load → integrate" if lang=="en" else "Apprendre → contrôler → charger → intégrer",
            "sessions":sessions,
        })
        previous_week_ids=week_ids

    report=screening_payload.get("report",screening_payload) or {}
    movement_score=_num(report.get("flexilab_score", report.get("score",0)),0)
    program={
        "engine_version":ENGINE_VERSION,
        "created_at":datetime.now(timezone.utc).isoformat(),
        "language":lang,
        "movement_score":movement_score,
        "movement_score_band":_band(movement_score,lang),
        "clinical_readiness":{
            "pain_state":pain,
            "training_experience":experience,
            "program_mode":"corrective_training" if pain=="no_pain" else "pain_free_corrective",
            "medical_advice_recommended": pain=="pain",
        },
        "pre_screening_questionnaire":q,
        "movement_dna_summary":movement_dna or {},
        "clinical_priorities":priorities[:3],
        "main_priorities":priorities[:3],
        "monitor_domains":priorities[3:],
        "program_summary":{
            "duration":"4 weeks" if lang=="en" else "4 semaines",
            "frequency":"3 sessions/week" if lang=="en" else "3 séances/semaine",
            "session_duration":"18-30 min",
            "model":"stable clinical targets + varied progressive exercise journey",
        },
        "selection_strategy":{
            "principles":[
                "stable clinical targets",
                "anatomical subject separated from intervention role",
                "explicit regression and progression links where clinically coherent",
                "criteria-based progression rather than week number alone",
                "exercise variation without randomness",
                "questionnaire-driven loading and safety",
                "exercise-specific dosage",
            ],
            "progression_model":"restore_control_stabilize_integrate_v4",
            "available_equipment":sorted(equipment),
            "reported_category_preferences":qcats,
        },
        "weeks":weeks,
        "safety_notes":[
            "No exercise should provoke or increase pain.",
            "Movement quality is more important than repetitions.",
            "Reduce range or load if compensation appears.",
        ] if lang=="en" else [
            "Aucun exercice ne doit provoquer ou augmenter la douleur.",
            "La qualité du mouvement prime sur le nombre de répétitions.",
            "Réduisez l’amplitude ou la charge si une compensation apparaît.",
        ],
    }
    program["validation_flags"]=_quality(program,pain)
    return program

generate_clinical_prescription = generate_clinical_prescription_v21
