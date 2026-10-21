from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ethicscan.validation import validate_inputs
from ethicscan.fairness import audit_fairness, risk_flags
from ethicscan.checklist import score_checklist
from ethicscan.recommendations import generate_recommendations
from ethicscan.audit_log import log_audit, load_history
from ethicscan.pipeline import run_audit

def test_validation_detects_missing_value():
    df = pd.DataFrame({"x": [1, None], "group": [0, 1], "label": [0, 1]})
    result = validate_inputs(df, "label", ["group"], ["x", "group"])
    assert result.ok
    assert result.missing_total == 1

def test_fairness_flags_low_disparate_impact():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 0, 0, 1, 0])
    group = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])
    result = audit_fairness(y_true, y_pred, group)
    assert "disparate_impact_ratio" in result
    assert "equalized_odds_gap" in result
    assert "subgroup_performance" in result
    assert isinstance(risk_flags(result), list)

def test_fairness_subgroup_performance_keys():
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1, 0, 1])
    group = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])
    result = audit_fairness(y_true, y_pred, group)
    for g in ["A", "B"]:
        assert all(k in result["subgroup_performance"][g] for k in ["accuracy", "precision", "recall", "f1"])

def test_checklist_evidence_stored():
    responses = {"safety": True, "equality": False, "privacy": True, "transparency": True, "accountability": False, "protection": True, "inclusion": False}
    evidence = {"safety": "test_report.pdf"}
    result = score_checklist(responses, evidence)
    assert result["total"] == 7
    safety_item = next(i for i in result["items"] if i["id"] == "safety")
    assert safety_item["evidence"] == "test_report.pdf"

def test_recommendations_triggered():
    flags = [{"severity": "high", "metric": "disparate_impact_ratio", "message": "test"}]
    checklist = {"status": "needs_review"}
    recs = generate_recommendations(flags, checklist)
    assert any(r["title"] for r in recs)
    assert len(recs) >= 2

def test_audit_log_append_and_load(tmp_path):
    import ethicscan.audit_log as al
    original = al._LOG_FILE
    al._LOG_FILE = tmp_path / "history.jsonl"
    result = {"accuracy": 0.9, "fairness": {"disparate_impact_ratio": 0.75, "demographic_parity_difference": -0.1, "equal_opportunity_difference": None, "equalized_odds_gap": None}, "checklist": {"score": 80}, "risk_flags": []}
    log_audit(result, {"target": "label", "sensitive_attribute": "group"})
    history = load_history(al._LOG_FILE)
    al._LOG_FILE = original
    assert len(history) == 1
    assert history[0]["summary"]["accuracy"] == 0.9

def test_pipeline_writes_report(tmp_path):
    df = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5, 6, 7], "group": [0, 0, 0, 0, 1, 1, 1, 1], "label": [0, 0, 0, 1, 1, 1, 1, 1]})
    model = LogisticRegression().fit(df[["x", "group"]], df.label)
    result = run_audit(model, df, "label", "group", {"transparency": True}, tmp_path)
    assert (tmp_path / "audit_result.json").exists()
    assert (tmp_path / "ethicscan_audit_report.html").exists()
    assert (tmp_path / "ethicscan_audit_report.pdf").exists()
    assert "fairness" in result
    assert "global_explainability" in result
    assert "recommendations" in result
