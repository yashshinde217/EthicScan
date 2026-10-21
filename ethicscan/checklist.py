from typing import Any

DEFAULT_CHECKLIST = [
    {"id": "safety", "principle": "Safety & Reliability", "question": "Are safety tests and failure monitoring documented?"},
    {"id": "equality", "principle": "Equality & Non-Discrimination", "question": "Have relevant demographic groups been considered in evaluation?"},
    {"id": "privacy", "principle": "Privacy & Security", "question": "Are personal data handling and access controls documented?"},
    {"id": "transparency", "principle": "Transparency", "question": "Can users understand the model's purpose and limitations?"},
    {"id": "accountability", "principle": "Accountability", "question": "Is an accountable owner or reviewer assigned?"},
]

def score_checklist(responses: dict[str, Any], checklist=DEFAULT_CHECKLIST) -> dict[str, Any]:
    items = []
    earned = 0
    for item in checklist:
        value = responses.get(item["id"], False)
        yes = value is True or str(value).lower() in {"yes", "true", "1", "complete"}
        earned += int(yes)
        items.append({**item, "response": "Yes" if yes else "No", "status": "pass" if yes else "gap"})
    score = round(100 * earned / len(checklist), 1) if checklist else 0.0
    return {"score": score, "earned": earned, "total": len(checklist), "items": items, "status": "good" if score >= 80 else "needs_review"}
