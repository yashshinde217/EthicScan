from typing import Any

# Full 7 NITI Aayog principles
DEFAULT_CHECKLIST = [
    {"id": "safety", "principle": "Safety & Reliability", "question": "Are safety tests and failure monitoring documented?"},
    {"id": "equality", "principle": "Equality & Non-Discrimination", "question": "Have relevant demographic groups been considered in evaluation?"},
    {"id": "privacy", "principle": "Privacy & Security", "question": "Are personal data handling and access controls documented?"},
    {"id": "transparency", "principle": "Transparency", "question": "Can users understand the model's purpose and limitations?"},
    {"id": "accountability", "principle": "Accountability", "question": "Is an accountable owner or reviewer assigned?"},
    {"id": "protection", "principle": "Protection & Reinforcement of Positive Human Values", "question": "Does the system reinforce positive societal values and avoid harm?"},
    {"id": "inclusion", "principle": "Inclusion & Accessibility", "question": "Is the system accessible and inclusive across diverse user groups?"},
]

def auto_check_checklist(audit_result: dict) -> dict[str, Any]:
    """Infer checklist responses from audit data. Returns {id: bool} responses and {id: reason} evidence."""
    fairness   = audit_result.get("fairness", {})
    flags      = audit_result.get("risk_flags", [])
    global_exp = audit_result.get("global_explainability", {})
    accuracy   = audit_result.get("accuracy", 0.0)
    validation = audit_result.get("validation", {})

    high_flags   = [f for f in flags if f.get("severity") == "high"]
    any_breach   = len(flags) > 0
    shap_avail   = bool(global_exp.get("summary"))
    groups       = fairness.get("groups", [])
    sensitive_in_features = validation.get("sensitive_in_features", False)

    responses = {
        "safety":        accuracy >= 0.60 and not high_flags,
        "equality":      not any_breach,
        "privacy":       not sensitive_in_features,
        "transparency":  shap_avail,
        "accountability": False,   # always requires a human owner
        "protection":    not high_flags,
        "inclusion":     len(groups) > 1,
    }
    evidence = {
        "safety":        f"Model accuracy {accuracy:.1%}" + ("; high-severity flags present" if high_flags else "; no high-severity flags"),
        "equality":      "No fairness threshold breaches" if not any_breach else f"{len(flags)} fairness flag(s) detected",
        "privacy":       "Sensitive attribute not used as model feature" if not sensitive_in_features else "Sensitive attribute appears in model features",
        "transparency":  "SHAP global explanations available" if shap_avail else "SHAP unavailable for this model type",
        "accountability": "Requires manual assignment of an accountable owner",
        "protection":    "No high-severity flags" if not high_flags else f"{len(high_flags)} high-severity flag(s) detected",
        "inclusion":     f"{len(groups)} group(s) found in sensitive attribute: {', '.join(str(g) for g in groups)}",
    }
    return responses, evidence


def score_checklist(responses: dict[str, Any], evidence: dict[str, str] | None = None, checklist=DEFAULT_CHECKLIST) -> dict[str, Any]:
    evidence = evidence or {}
    items = []
    earned = 0
    for item in checklist:
        value = responses.get(item["id"], False)
        yes = value is True or str(value).lower() in {"yes", "true", "1", "complete"}
        earned += int(yes)
        items.append({
            **item,
            "response": "Yes" if yes else "No",
            "status": "pass" if yes else "gap",
            "evidence": evidence.get(item["id"], ""),
        })
    score = round(100 * earned / len(checklist), 1) if checklist else 0.0
    return {"score": score, "earned": earned, "total": len(checklist), "items": items, "status": "good" if score >= 80 else "needs_review"}
