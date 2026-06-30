
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


# -----------------------------------------------------------------------------
# FlexiLab V45 i18n language lock
# Lightweight dictionary-based localization. This is CPU-cheap: simple dict lookup
# and string replacement only. No ML translation, no external calls.
# -----------------------------------------------------------------------------
CATEGORY_LABELS_I18N = {
    "RB": {"fr": "Respiration / récupération", "en": "Breathing / recovery"},
    "TM": {"fr": "Mobilité thoracique", "en": "Thoracic mobility"},
    "SH": {"fr": "Mobilité des épaules", "en": "Shoulder mobility"},
    "CC": {"fr": "Contrôle cervical", "en": "Cervical control"},
    "CS": {"fr": "Stabilité du tronc", "en": "Core stability"},
    "HM": {"fr": "Mobilité de hanche", "en": "Hip mobility"},
    "HS": {"fr": "Mobilité ischio-jambiers", "en": "Hamstring mobility"},
    "AM": {"fr": "Mobilité de cheville", "en": "Ankle mobility"},
    "BP": {"fr": "Équilibre / proprioception", "en": "Balance / proprioception"},
    "FI": {"fr": "Intégration fonctionnelle", "en": "Functional integration"},
}

OBJECTIVE_LABELS_I18N = {
    "recovery": {"fr": "Récupération", "en": "Recovery"},
    "breathing": {"fr": "Respiration", "en": "Breathing"},
    "mobility": {"fr": "Mobilité", "en": "Mobility"},
    "mobility_control": {"fr": "Mobilité contrôlée", "en": "Mobility control"},
    "motor_control": {"fr": "Contrôle moteur", "en": "Motor control"},
    "control": {"fr": "Contrôle", "en": "Control"},
    "activation": {"fr": "Activation", "en": "Activation"},
    "activation_stability": {"fr": "Activation / stabilité", "en": "Activation / stability"},
    "stability": {"fr": "Stabilité", "en": "Stability"},
    "functional_strength": {"fr": "Force fonctionnelle", "en": "Functional strength"},
    "dynamic_balance": {"fr": "Équilibre dynamique", "en": "Dynamic balance"},
    "integration": {"fr": "Intégration", "en": "Integration"},
}

PHASE_LABELS_I18N = {
    "0_recovery": {"fr": "Récupération", "en": "Recovery"},
    "1_restore": {"fr": "Restaurer", "en": "Restore"},
    "2_control": {"fr": "Contrôler", "en": "Control"},
    "3_stabilize": {"fr": "Stabiliser", "en": "Stabilize"},
    "4_integrate": {"fr": "Intégrer", "en": "Integrate"},
}

EQUIPMENT_LABELS_I18N = {
    "": {"fr": "Aucun matériel", "en": "No equipment"},
    "none": {"fr": "Aucun matériel", "en": "No equipment"},
    "no equipment": {"fr": "Aucun matériel", "en": "No equipment"},
    "mat": {"fr": "Tapis", "en": "Mat"},
    "exercise mat": {"fr": "Tapis", "en": "Exercise mat"},
    "wall": {"fr": "Mur", "en": "Wall"},
    "chair": {"fr": "Chaise", "en": "Chair"},
    "box": {"fr": "Box / chaise", "en": "Box / chair"},
    "stick_or_pvc": {"fr": "Bâton ou PVC", "en": "Stick or PVC"},
    "pvc": {"fr": "Bâton PVC", "en": "PVC dowel"},
    "pvc dowel": {"fr": "Bâton PVC", "en": "PVC dowel"},
    "foam roller": {"fr": "Foam roller", "en": "Foam roller"},
    "mini band": {"fr": "Mini-bande élastique", "en": "Mini band"},
    "resistance band": {"fr": "Élastique", "en": "Resistance band"},
    "light dumbbell": {"fr": "Haltère léger", "en": "Light dumbbell"},
    "dumbbell": {"fr": "Haltère", "en": "Dumbbell"},
    "kettlebell": {"fr": "Kettlebell", "en": "Kettlebell"},
}

EXERCISE_NAME_I18N = {
    "RB001": {"fr": "Respiration crocodile", "en": "Crocodile breathing"},
    "RB002": {"fr": "Respiration allongée", "en": "Supine breathing"},
    "RB003": {"fr": "Respiration diaphragmatique", "en": "Diaphragmatic breathing"},
    "RB004": {"fr": "Respiration 90/90", "en": "90/90 breathing"},
    "TM001": {"fr": "Ouverture du livre", "en": "Open book rotation"},
    "TM002": {"fr": "Dos rond / dos creux", "en": "Cat-camel"},
    "TM003": {"fr": "Extension thoracique sur foam roller", "en": "Foam roller thoracic extension"},
    "TM004": {"fr": "Thread the needle", "en": "Thread the needle"},
    "SH001": {"fr": "Pass-through avec bâton PVC", "en": "PVC pass-through"},
    "SH002": {"fr": "Glissement mural", "en": "Wall slide"},
    "SH011": {"fr": "Glissement mural serratus", "en": "Serratus wall slide"},
    "CC001": {"fr": "Double menton allongé", "en": "Supine chin tuck"},
    "CC002": {"fr": "Maintien des fléchisseurs profonds du cou", "en": "Deep neck flexor hold"},
    "CC003": {"fr": "Hochement cervical contrôlé", "en": "Chin nod"},
    "CC004": {"fr": "Rétraction cervicale au mur", "en": "Wall chin retraction"},
    "CS001": {"fr": "Dead bug", "en": "Dead bug"},
    "CS002": {"fr": "Bird dog", "en": "Bird dog"},
    "CS003": {"fr": "Curl-up de McGill", "en": "McGill curl-up"},
    "CS004": {"fr": "Planche latérale", "en": "Side plank"},
    "HM001": {"fr": "Étirement fléchisseur de hanche", "en": "Hip flexor stretch"},
    "HM002": {"fr": "Mobilité 90/90 de hanche", "en": "90/90 hip mobility"},
    "AM001": {"fr": "Mobilisation cheville genou-au-mur", "en": "Knee-to-wall ankle mobilization"},
    "AM002": {"fr": "Mobilisation cheville en demi-genou", "en": "Half-kneeling ankle mobilization"},
    "AM003": {"fr": "Mobilisation cheville en squat profond", "en": "Deep squat ankle rock"},
    "AM004": {"fr": "Élévation des talons", "en": "Heel raise"},
    "FI001": {"fr": "Goblet squat vers box", "en": "Goblet squat to box"},
    "FI002": {"fr": "Squat vers box", "en": "Squat to box"},
    "FI006": {"fr": "Soulevé de terre unipodal avec reach", "en": "Single-leg RDL reach"},
    "FI011": {"fr": "Fente arrière", "en": "Reverse lunge"},
    "FI012": {"fr": "Assis-debout", "en": "Sit to stand"},
}

TIP_I18N_BY_CATEGORY = {
    "RB": {"fr": "Respirez lentement par le nez, relâchez les épaules et allongez l’expiration.", "en": "Breathe slowly through the nose, relax the shoulders and lengthen the exhale."},
    "TM": {"fr": "Bougez depuis le haut du dos, gardez le bassin stable et restez dans une amplitude confortable.", "en": "Move from the upper back, keep the pelvis stable and stay in a comfortable range."},
    "SH": {"fr": "Gardez les côtes abaissées, les épaules détendues et évitez de cambrer le dos.", "en": "Keep the ribs down, shoulders relaxed and avoid arching the back."},
    "CC": {"fr": "Gardez la nuque longue, la mâchoire détendue et évitez de lever la tête.", "en": "Keep the neck long, jaw relaxed and avoid lifting the head."},
    "CS": {"fr": "Gardez le bassin stable, respirez et privilégiez la qualité du contrôle.", "en": "Keep the pelvis stable, breathe and prioritize control quality."},
    "HM": {"fr": "Gardez une amplitude sans douleur et évitez de compenser avec le bas du dos.", "en": "Use a pain-free range and avoid compensating with the lower back."},
    "HS": {"fr": "Gardez le genou tendu sans forcer et respirez dans l’amplitude.", "en": "Keep the knee straight without forcing and breathe into the range."},
    "AM": {"fr": "Gardez le talon au sol et avancez le genou progressivement sans douleur.", "en": "Keep the heel down and move the knee forward progressively without pain."},
    "BP": {"fr": "Bougez lentement, gardez l’alignement et recherchez la stabilité avant l’amplitude.", "en": "Move slowly, keep alignment and prioritize stability before range."},
    "FI": {"fr": "Contrôlez l’alignement, bougez lentement et gardez une respiration régulière.", "en": "Control alignment, move slowly and keep steady breathing."},
}

RATIONALE_I18N_BY_CATEGORY = {
    "RB": {"fr": "Inclus pour améliorer la respiration, réduire les tensions inutiles et préparer un meilleur contrôle postural.", "en": "Included to improve breathing mechanics, reduce unnecessary tension and prepare better postural control."},
    "TM": {"fr": "Inclus pour améliorer la mobilité du haut du dos et réduire les compensations cervicales, scapulaires et du tronc.", "en": "Included to improve upper-back mobility and reduce cervical, scapular and trunk compensations."},
    "SH": {"fr": "Inclus pour soutenir la mobilité des épaules et la mécanique scapulaire en complément du travail thoracique et cervical.", "en": "Included to support shoulder mobility and scapular mechanics alongside thoracic and cervical work."},
    "CC": {"fr": "Inclus pour renforcer le contrôle cervical, l’endurance posturale et limiter les tensions inutiles au niveau du cou.", "en": "Included to reinforce cervical control, postural endurance and reduce unnecessary neck tension."},
    "CS": {"fr": "Inclus pour améliorer le contrôle lombo-pelvien et la stabilité du tronc pendant les mouvements fonctionnels.", "en": "Included to improve lumbopelvic control and trunk stability during functional movement."},
    "HM": {"fr": "Inclus pour soutenir la mobilité de hanche et réduire les compensations du bassin et du bas du dos.", "en": "Included to support hip mobility and reduce pelvic and lower-back compensation."},
    "HS": {"fr": "Inclus pour améliorer la mobilité active hanche/ischio-jambiers et l’équilibre gauche-droite.", "en": "Included to improve active hip/hamstring mobility and left-right balance."},
    "AM": {"fr": "Inclus comme soutien au squat car la mobilité de cheville peut influencer l’inclinaison du tronc.", "en": "Included as squat support because ankle mobility can influence trunk lean."},
    "BP": {"fr": "Inclus pour améliorer le contrôle postural, l’équilibre et la coordination du mouvement.", "en": "Included to improve postural control, balance and movement coordination."},
    "FI": {"fr": "Inclus pour transférer les gains de mobilité et de contrôle vers des mouvements fonctionnels du quotidien.", "en": "Included to transfer mobility and control gains into daily functional movement."},
}

DYNAMIC_TEXT_REPLACEMENTS = {
    "fr": {
        "Reset / breathing": "Reset / respiration",
        "Primary mobility": "Mobilité principale",
        "Secondary mobility": "Mobilité secondaire",
        "Activation / control": "Activation / contrôle",
        "Functional integration": "Intégration fonctionnelle",
        "Cool-down": "Retour au calme",
        "Restore": "Restaurer", "Control": "Contrôler", "Stabilize": "Stabiliser", "Integrate": "Intégrer",
        "controlled": "contrôlé", "slow breathing": "respiration lente", "as needed": "selon besoin",
        "reps / side": "répétitions de chaque côté", "reps": "répétitions", "sec holds": "secondes de maintien", "sec": "secondes",
    },
    "en": {
        "Reset / respiration": "Reset / breathing",
        "Mobilité principale": "Primary mobility",
        "Mobilité secondaire": "Secondary mobility",
        "Activation / contrôle": "Activation / control",
        "Intégration fonctionnelle": "Functional integration",
        "Retour au calme": "Cool-down",
        "Restaurer": "Restore", "Contrôler": "Control", "Stabiliser": "Stabilize", "Intégrer": "Integrate",
        "contrôlé": "controlled", "respiration lente": "slow breathing", "selon besoin": "as needed",
        "répétitions de chaque côté": "reps / side", "répétitions": "reps", "secondes de maintien": "sec holds", "secondes": "sec",
    }
}

def _lng(lang: str) -> str:
    return "en" if str(lang).lower().startswith("en") else "fr"

def _replace_dynamic_text(value: Any, lang: str) -> str:
    lang = _lng(lang)
    s = str(value or "")
    for src, dst in DYNAMIC_TEXT_REPLACEMENTS.get(lang, {}).items():
        s = s.replace(src, dst)
    return s

def _equipment_i18n(value: Any, lang: str) -> str:
    lang = _lng(lang)
    raw = str(value or "").strip()
    if not raw:
        return EQUIPMENT_LABELS_I18N[""][lang]
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    out = []
    for p in parts:
        key = p.lower().strip()
        out.append(EQUIPMENT_LABELS_I18N.get(key, {"fr": p, "en": p}).get(lang, p))
    return ", ".join(out) if out else EQUIPMENT_LABELS_I18N[""][lang]

def _exercise_name_i18n(ex: dict, lang: str) -> str:
    lang = _lng(lang)
    eid = str(ex.get("exercise_id") or ex.get("id") or "").upper()
    if eid in EXERCISE_NAME_I18N:
        return EXERCISE_NAME_I18N[eid][lang]
    return ex.get(f"name_{lang}") or ex.get("name_fr") or ex.get("name_en") or ex.get("name") or eid

def _category_i18n(code: str, lang: str) -> str:
    lang = _lng(lang)
    code = str(code or "").upper()
    return CATEGORY_LABELS_I18N.get(code, {"fr": code, "en": code}).get(lang, code)

def _objective_i18n(value: Any, lang: str) -> str:
    lang = _lng(lang)
    key = str(value or "").lower().strip()
    return OBJECTIVE_LABELS_I18N.get(key, {"fr": _replace_dynamic_text(value, "fr"), "en": _replace_dynamic_text(value, "en")}).get(lang, str(value or ""))

def _phase_i18n(value: Any, lang: str) -> str:
    lang = _lng(lang)
    key = str(value or "").lower().strip()
    return PHASE_LABELS_I18N.get(key, {"fr": _replace_dynamic_text(value, "fr"), "en": _replace_dynamic_text(value, "en")}).get(lang, str(value or ""))

def _tip_i18n(ex: dict, lang: str) -> str:
    lang = _lng(lang)
    c = cat(ex)
    return TIP_I18N_BY_CATEGORY.get(c, TIP_I18N_BY_CATEGORY["FI"])[lang]

def _rationale_i18n(ex: dict, lang: str) -> str:
    lang = _lng(lang)
    c = cat(ex)
    return RATIONALE_I18N_BY_CATEGORY.get(c, RATIONALE_I18N_BY_CATEGORY["FI"])[lang]


def loc(ex, slot, priorities, lang):
    lang = _lng(lang)
    fr, en = BLOCK_LABELS.get(slot, (slot, slot))
    related = []
    for p in priorities[:4]:
        if cat(ex) in DOMAIN_TO_CATS.get(p.get("id"), []) or p.get("id") in exdomains(ex):
            related.append(p.get("label") if lang == "fr" else p.get("label_en", p.get("label", "")))
    related = [r for r in related if r]
    why_prefix = "Sélectionné pour soutenir : " if lang == "fr" else "Selected to support: "
    why = why_prefix + ", ".join(related[:3] or [_category_i18n(cat(ex), lang)])
    eid = ex.get("exercise_id")
    return {
        "id": eid,
        "exercise_id": eid,
        "block": slot,
        "block_label": _replace_dynamic_text(fr if lang == "fr" else en, lang),
        "block_label_fr": _replace_dynamic_text(fr, "fr"),
        "block_label_en": _replace_dynamic_text(en, "en"),
        "name": _exercise_name_i18n(ex, lang),
        "name_fr": _exercise_name_i18n(ex, "fr"),
        "name_en": _exercise_name_i18n(ex, "en"),
        "category_code": cat(ex),
        "target": _category_i18n(cat(ex), lang),
        "target_fr": _category_i18n(cat(ex), "fr"),
        "target_en": _category_i18n(cat(ex), "en"),
        "primary_objective": ex.get("primary_objective"),
        "primary_objective_label": _objective_i18n(ex.get("primary_objective"), lang),
        "primary_objective_label_fr": _objective_i18n(ex.get("primary_objective"), "fr"),
        "primary_objective_label_en": _objective_i18n(ex.get("primary_objective"), "en"),
        "difficulty": ex.get("difficulty_1_5"),
        "phase": ex.get("phase"),
        "phase_label": _phase_i18n(ex.get("phase"), lang),
        "phase_label_fr": _phase_i18n(ex.get("phase"), "fr"),
        "phase_label_en": _phase_i18n(ex.get("phase"), "en"),
        "equipment": _equipment_i18n(ex.get("equipment", ""), lang),
        "equipment_fr": _equipment_i18n(ex.get("equipment", ""), "fr"),
        "equipment_en": _equipment_i18n(ex.get("equipment", ""), "en"),
        "material": _equipment_i18n(ex.get("equipment", ""), lang),
        "material_fr": _equipment_i18n(ex.get("equipment", ""), "fr"),
        "material_en": _equipment_i18n(ex.get("equipment", ""), "en"),
        "sets": ex.get("sets", ""),
        "reps_time": _replace_dynamic_text(ex.get("reps_time", ""), lang),
        "reps_time_fr": _replace_dynamic_text(ex.get("reps_time", ""), "fr"),
        "reps_time_en": _replace_dynamic_text(ex.get("reps_time", ""), "en"),
        "tempo": _replace_dynamic_text(ex.get("tempo", ""), lang),
        "tempo_fr": _replace_dynamic_text(ex.get("tempo", ""), "fr"),
        "tempo_en": _replace_dynamic_text(ex.get("tempo", ""), "en"),
        "rest": _replace_dynamic_text(ex.get("rest", ""), lang),
        "rest_fr": _replace_dynamic_text(ex.get("rest", ""), "fr"),
        "rest_en": _replace_dynamic_text(ex.get("rest", ""), "en"),
        "frequency_per_week": ex.get("frequency_per_week", ""),
        "coaching_cues": _tip_i18n(ex, lang),
        "coaching_cues_fr": _tip_i18n(ex, "fr"),
        "coaching_cues_en": _tip_i18n(ex, "en"),
        "tips": _tip_i18n(ex, lang),
        "tips_fr": _tip_i18n(ex, "fr"),
        "tips_en": _tip_i18n(ex, "en"),
        "common_errors": _replace_dynamic_text(ex.get("common_errors", ""), lang),
        "clinical_rationale": _rationale_i18n(ex, lang),
        "clinical_rationale_fr": _rationale_i18n(ex, "fr"),
        "clinical_rationale_en": _rationale_i18n(ex, "en"),
        "why_in_this_program": why,
        "why_in_this_program_fr": why_prefix.replace("Selected to support: ", "Sélectionné pour soutenir : ") + ", ".join(related[:3] or [_category_i18n(cat(ex), "fr")]),
        "why_in_this_program_en": why_prefix.replace("Sélectionné pour soutenir : ", "Selected to support: ") + ", ".join(related[:3] or [_category_i18n(cat(ex), "en")]),
        "regression_id": ex.get("regression_id", ""),
        "progression_id": ex.get("progression_id", ""),
        "pain_rule": ex.get("pain_rule", ""),
        "asymmetry_rule": ex.get("asymmetry_rule", ""),
        "video_url": ex.get("video_url", ""),
        "vimeo_url": ex.get("vimeo_url", ""),
        "mp4_url": ex.get("mp4_url", ""),
        "thumbnail_url": ex.get("thumbnail_url", ""),
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

# V45 final wrapper: language-lock every exercise after all clinical rules.
_generate_clinical_prescription_v45_before_i18n = generate_clinical_prescription_v21

def _language_lock_program(program: dict, lang: str = "fr") -> dict:
    lang = _lng(lang)
    if not isinstance(program, dict):
        return program
    program["language"] = lang
    for week in program.get("weeks", []) or []:
        if week.get("phase"):
            week["phase"] = _replace_dynamic_text(week.get("phase"), lang)
        if week.get("objective"):
            week["objective"] = _replace_dynamic_text(week.get("objective"), lang)
        for session in week.get("sessions", []) or week.get("days", []) or []:
            if session.get("focus"):
                session["focus"] = _replace_dynamic_text(session.get("focus"), lang)
            for exercise in session.get("exercises", []) or []:
                eid = exercise.get("exercise_id") or exercise.get("id")
                code = exercise.get("category_code")
                exercise["exercise_id"] = eid
                exercise["name"] = EXERCISE_NAME_I18N.get(str(eid or "").upper(), {"fr": exercise.get("name_fr") or exercise.get("name"), "en": exercise.get("name_en") or exercise.get("name")})[lang]
                exercise["target"] = _category_i18n(code, lang)
                exercise["equipment"] = exercise.get(f"equipment_{lang}") or _equipment_i18n(exercise.get("equipment", ""), lang)
                exercise["material"] = exercise.get(f"material_{lang}") or exercise["equipment"]
                exercise["tips"] = exercise.get(f"tips_{lang}") or TIP_I18N_BY_CATEGORY.get(str(code or "").upper(), TIP_I18N_BY_CATEGORY["FI"])[lang]
                exercise["coaching_cues"] = exercise.get(f"coaching_cues_{lang}") or exercise["tips"]
                exercise["clinical_rationale"] = exercise.get(f"clinical_rationale_{lang}") or RATIONALE_I18N_BY_CATEGORY.get(str(code or "").upper(), RATIONALE_I18N_BY_CATEGORY["FI"])[lang]
                exercise["why_in_this_program"] = exercise.get(f"why_in_this_program_{lang}") or _replace_dynamic_text(exercise.get("why_in_this_program", ""), lang)
                exercise["reps_time"] = exercise.get(f"reps_time_{lang}") or _replace_dynamic_text(exercise.get("reps_time", ""), lang)
                exercise["tempo"] = exercise.get(f"tempo_{lang}") or _replace_dynamic_text(exercise.get("tempo", ""), lang)
                exercise["rest"] = exercise.get(f"rest_{lang}") or _replace_dynamic_text(exercise.get("rest", ""), lang)
    return program

def generate_clinical_prescription_v21(screening_payload, exercise_library, rules=None, movement_dna=None, language="fr"):
    program = _generate_clinical_prescription_v45_before_i18n(screening_payload, exercise_library, rules=rules, movement_dna=movement_dna, language=language)
    return _language_lock_program(program, language)

generate_clinical_prescription = generate_clinical_prescription_v21
