
import json
from pathlib import Path

WEEK_PHASES = {
    1: ("Restore", "Restore breathing, joint mobility, and basic motor control."),
    2: ("Control", "Improve active control, coordination, and controlled range of motion."),
    3: ("Stabilize", "Build stability and light strength while maintaining movement quality."),
    4: ("Integrate", "Integrate corrections into functional movement patterns."),
}

DAY_FOCUS = {
    1: "Mobility / Restore",
    2: "Control / Stability",
    3: "Integration / Movement Pattern",
}

FAULT_TO_REGIONS = {
    "forward_head": ["cervical", "thoracic", "shoulder"],
    "thoracic_posture": ["thoracic", "shoulder"],
    "overhead_limitation": ["shoulder", "thoracic"],
    "squat_trunk_lean": ["core", "hip", "ankle", "functional"],
    "squat_depth_control": ["ankle", "hip", "core", "functional"],
    "aslr_limitation": ["posterior_chain", "hip", "core"],
    "asymmetry": ["hip", "posterior_chain", "shoulder", "core"],
}

DEFAULT_REGIONS = ["core", "hip", "thoracic", "shoulder", "posterior_chain", "ankle"]

def load_library(path="exercise_knowledge_base_v1.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def regions_from_findings(findings):
    scores = {}
    for f in findings or []:
        fault = f.get("fault") or f.get("id") or ""
        priority = float(f.get("priority_score", 70))
        for region in FAULT_TO_REGIONS.get(fault, []):
            scores[region] = scores.get(region, 0) + priority
        for c in f.get("contributors", []) or []:
            label = str(c.get("system", "") or c.get("system_label", "")).lower()
            if "cerv" in label or "neck" in label: scores["cervical"] = scores.get("cervical", 0) + priority*.4
            if "thor" in label: scores["thoracic"] = scores.get("thoracic", 0) + priority*.4
            if "shoulder" in label or "scap" in label: scores["shoulder"] = scores.get("shoulder", 0) + priority*.4
            if "core" in label or "tronc" in label: scores["core"] = scores.get("core", 0) + priority*.4
            if "hip" in label or "hanche" in label: scores["hip"] = scores.get("hip", 0) + priority*.4
            if "posterior" in label or "aslr" in label or "postérieure" in label: scores["posterior_chain"] = scores.get("posterior_chain", 0) + priority*.4
            if "ankle" in label or "cheville" in label: scores["ankle"] = scores.get("ankle", 0) + priority*.4
    ranked = [r for r,_ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
    for r in DEFAULT_REGIONS:
        if r not in ranked:
            ranked.append(r)
    return ranked[:6]

def normalize_pain(pain_clearance):
    pain_clearance = pain_clearance or {}
    out = {}
    for k, v in pain_clearance.items():
        s = str(v).lower()
        out[k] = "pain" if s in ["pain","yes","true"] else "discomfort" if "discomfort" in s or "tension" in s else "no_pain"
    return out

def candidates(library, region, week):
    return [e for e in library["exercises"] if e["region"] == region and int(e["week"]) == int(week)]

def pick_week_exercises(library, priority_regions, week):
    selected, used = [], set()
    for region in priority_regions:
        for ex in candidates(library, region, week):
            if ex["id"] not in used:
                selected.append(ex); used.add(ex["id"]); break
        if len(selected) >= 6: break
    if len(selected) < 6:
        for region in DEFAULT_REGIONS + ["functional"]:
            for ex in candidates(library, region, week):
                if ex["id"] not in used:
                    selected.append(ex); used.add(ex["id"])
                if len(selected) >= 6: break
            if len(selected) >= 6: break
    return selected[:6]

def format_exercise(e):
    return {
        "id": e["id"], "name": e["name"], "region": e["region"], "phase": e["phase"],
        "goal": e["goal"], "why": e["why"], "equipment": e.get("equipment", []),
        "sets": e["sets"], "reps": e.get("reps",""), "hold": e.get("hold",""),
        "rest": e["rest"], "tempo": e["tempo"], "coaching_cues": e["coaching_cues"],
        "common_mistakes": e["common_mistakes"], "pain_modification": e["pain_modification"],
        "progression": e.get("progression",""), "regression": e.get("regression","")
    }

def split_days(exercises):
    day1 = exercises[:4]
    day2 = (exercises[1:3] + exercises[4:6])[:4] or day1
    day3 = (exercises[2:6])[:4] or day1
    return [
        {"day": 1, "focus": DAY_FOCUS[1], "estimated_duration": "12–15 min", "exercises": [format_exercise(e) for e in day1]},
        {"day": 2, "focus": DAY_FOCUS[2], "estimated_duration": "12–15 min", "exercises": [format_exercise(e) for e in day2]},
        {"day": 3, "focus": DAY_FOCUS[3], "estimated_duration": "12–15 min", "exercises": [format_exercise(e) for e in day3]},
    ]

def generate_prescription(findings=None, pain_clearance=None, library_path="exercise_knowledge_base_v1.json", client_name="", score=None):
    library = load_library(library_path)
    pain = normalize_pain(pain_clearance)
    priority_regions = regions_from_findings(findings or [])
    weeks = []
    for week in range(1,5):
        phase, objective = WEEK_PHASES[week]
        exs = pick_week_exercises(library, priority_regions, week)
        weeks.append({
            "week": week,
            "phase": phase,
            "objective": objective,
            "progression_rule": [
                "Learn the positions, restore mobility, and move without pain.",
                "Increase active control while keeping the same quality standards.",
                "Add stability demand and light resistance when movement is controlled.",
                "Integrate corrections into functional patterns and prepare for reassessment."
            ][week-1],
            "days": split_days(exs)
        })
    return {
        "engine_version": "FlexiLab Prescription Engine V1",
        "client_name": client_name,
        "initial_score": score,
        "priority_regions": priority_regions,
        "pain_clearance": pain,
        "frequency": "3 days/week",
        "duration": "4 weeks",
        "safety_rule": "No exercise should provoke pain. If pain appears, stop, regress, or seek professional guidance.",
        "weeks": weeks
    }

if __name__ == "__main__":
    sample = generate_prescription(
        findings=[
            {"fault":"squat_trunk_lean","priority_score":100},
            {"fault":"aslr_limitation","priority_score":85},
            {"fault":"overhead_limitation","priority_score":72},
            {"fault":"forward_head","priority_score":65}
        ],
        pain_clearance={"squat":"discomfort"},
        library_path=Path(__file__).with_name("exercise_knowledge_base_v1.json"),
        client_name="Sample Client",
        score=76
    )
    print(json.dumps(sample, indent=2, ensure_ascii=False))
