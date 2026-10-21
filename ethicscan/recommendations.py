from typing import Any

_MITIGATIONS: dict[str, dict[str, Any]] = {
    "disparate_impact_ratio": {
        "title": "Address Adverse Impact (Disparate Impact Ratio)",
        "actions": [
            "Re-sample training data to balance representation across groups (e.g., oversampling with SMOTE or undersampling the majority group).",
            "Apply re-weighting: assign higher sample weights to the under-represented group during model training.",
            "Use a fairness-aware algorithm (e.g., Fairlearn's ExponentiatedGradient with DemographicParity constraint).",
            "Audit the feature set for proxies of the sensitive attribute and consider removing or transforming them.",
            "Adjust the decision threshold per group to equalise positive-rate outcomes.",
        ],
        "references": ["Fairlearn documentation — Mitigation algorithms", "NIST AI RMF — GOVERN 1.2"],
    },
    "demographic_parity_difference": {
        "title": "Reduce Demographic Parity Gap",
        "actions": [
            "Investigate whether the positive-rate gap reflects a genuine base-rate difference or a model artefact.",
            "Apply post-processing threshold calibration to align group-level positive rates.",
            "Collect additional representative data for the lower-rate group.",
            "Review labelling process for potential annotator bias.",
        ],
        "references": ["Barocas et al. — Fairness and Machine Learning (fairmlbook.org)"],
    },
    "equal_opportunity_difference": {
        "title": "Improve Equal Opportunity (TPR Parity)",
        "actions": [
            "Apply Fairlearn's EqualizedOdds post-processing to align true-positive rates across groups.",
            "Investigate whether training data under-represents positive examples for the lower-TPR group.",
            "Consider separate threshold optimisation per group to equalise recall.",
        ],
        "references": ["Hardt et al. (2016) — Equality of Opportunity in Supervised Learning"],
    },
    "equalized_odds_gap": {
        "title": "Close Equalized Odds Gap (TPR + FPR Parity)",
        "actions": [
            "Use Fairlearn's ThresholdOptimizer with equalized_odds constraint.",
            "Audit false-positive costs per group — high FPR for one group may indicate systematic over-prediction.",
            "Collect more balanced ground-truth labels across groups.",
        ],
        "references": ["Fairlearn — ThresholdOptimizer"],
    },
    "checklist_gap": {
        "title": "Close Governance Checklist Gaps",
        "actions": [
            "Assign a named accountable owner for each unanswered principle.",
            "Document evidence artefacts (test reports, privacy impact assessments, monitoring dashboards) and attach them to the checklist.",
            "Schedule a governance review before deployment.",
        ],
        "references": ["NITI Aayog Responsible AI Principles (2021)"],
    },
}

def generate_recommendations(risk_flags: list[dict], checklist: dict) -> list[dict[str, Any]]:
    recs = []
    seen = set()
    for flag in risk_flags:
        metric = flag.get("metric", "")
        if metric in _MITIGATIONS and metric not in seen:
            seen.add(metric)
            recs.append({"severity": flag["severity"], **_MITIGATIONS[metric]})
    if checklist.get("status") == "needs_review" and "checklist_gap" not in seen:
        recs.append({"severity": "medium", **_MITIGATIONS["checklist_gap"]})
    return recs
