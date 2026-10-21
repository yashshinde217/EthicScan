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
