from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def _positive_rate(y_pred: pd.Series | np.ndarray) -> float:
    values = np.asarray(y_pred)
    return float(np.mean(values == 1)) if len(values) else 0.0

def audit_fairness(y_true, y_pred, sensitive: pd.Series, favorable_label=1) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = pd.Series(sensitive).astype(str)
    unique_groups = list(pd.unique(groups))
    if len(unique_groups) < 2:
        return {"error": "At least two sensitive groups are required."}
    rates = {g: float(np.mean(y_pred[groups.to_numpy() == g] == favorable_label)) for g in unique_groups}
    tprs, fprs, subgroup_perf = {}, {}, {}
    for g in unique_groups:
        mask = groups.to_numpy() == g
        yt_g, yp_g = y_true[mask], y_pred[mask]
        positives = yt_g == favorable_label
        negatives = yt_g != favorable_label
        tprs[g] = float(np.mean(yp_g[positives] == favorable_label)) if positives.any() else None
        fprs[g] = float(np.mean(yp_g[negatives] == favorable_label)) if negatives.any() else None
        subgroup_perf[g] = {
            "count": int(mask.sum()),
            "accuracy": float(accuracy_score(yt_g, yp_g)),
            "precision": float(precision_score(yt_g, yp_g, zero_division=0)),
            "recall": float(recall_score(yt_g, yp_g, zero_division=0)),
            "f1": float(f1_score(yt_g, yp_g, zero_division=0)),
            "positive_rate": rates[g],
        }
    reference = max(rates, key=rates.get)
    protected = min(rates, key=rates.get)
    dpd = rates[protected] - rates[reference]
    di = rates[protected] / rates[reference] if rates[reference] > 0 else 0.0
    valid_tprs = {g: v for g, v in tprs.items() if v is not None}
    valid_fprs = {g: v for g, v in fprs.items() if v is not None}
    eo_diff = (min(valid_tprs.values()) - max(valid_tprs.values())) if len(valid_tprs) >= 2 else None
    # Equalized odds: max gap across both TPR and FPR
    tpr_gap = (max(valid_tprs.values()) - min(valid_tprs.values())) if len(valid_tprs) >= 2 else None
    fpr_gap = (max(valid_fprs.values()) - min(valid_fprs.values())) if len(valid_fprs) >= 2 else None
    equalized_odds_gap = max(tpr_gap or 0.0, fpr_gap or 0.0) if (tpr_gap is not None or fpr_gap is not None) else None
    return {
        "groups": unique_groups,
        "group_counts": {g: int((groups == g).sum()) for g in unique_groups},
        "positive_rates": rates,
        "true_positive_rates": tprs,
        "false_positive_rates": fprs,
        "reference_group": reference,
        "lowest_positive_rate_group": protected,
        "demographic_parity_difference": float(dpd),
        "disparate_impact_ratio": float(di),
        "equal_opportunity_difference": float(eo_diff) if eo_diff is not None else None,
        "equalized_odds_gap": float(equalized_odds_gap) if equalized_odds_gap is not None else None,
        "subgroup_performance": subgroup_perf,
    }

def risk_flags(fairness: dict[str, Any], disparate_impact_threshold=0.8, dpd_threshold=0.1) -> list[dict[str, str]]:
    if "error" in fairness:
        return [{"severity": "high", "metric": "input", "message": fairness["error"]}]
    flags = []
    if fairness["disparate_impact_ratio"] < disparate_impact_threshold:
        flags.append({"severity": "high", "metric": "disparate_impact_ratio", "message": f"Disparate impact ratio is {fairness['disparate_impact_ratio']:.3f}, below {disparate_impact_threshold:.2f}. Review possible adverse impact."})
    if abs(fairness["demographic_parity_difference"]) > dpd_threshold:
        flags.append({"severity": "medium", "metric": "demographic_parity_difference", "message": f"Demographic parity difference is {fairness['demographic_parity_difference']:.3f}, exceeding ±{dpd_threshold:.2f}."})
    if fairness.get("equal_opportunity_difference") is not None and abs(fairness["equal_opportunity_difference"]) > dpd_threshold:
        flags.append({"severity": "medium", "metric": "equal_opportunity_difference", "message": f"Equal opportunity difference is {fairness['equal_opportunity_difference']:.3f}, exceeding ±{dpd_threshold:.2f}."})
    if fairness.get("equalized_odds_gap") is not None and fairness["equalized_odds_gap"] > dpd_threshold:
        flags.append({"severity": "medium", "metric": "equalized_odds_gap", "message": f"Equalized odds gap is {fairness['equalized_odds_gap']:.3f}, exceeding {dpd_threshold:.2f}."})
    return flags
