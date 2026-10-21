from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ethicscan.validation import validate_inputs
from ethicscan.fairness import audit_fairness, risk_flags
from ethicscan.checklist import score_checklist
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
    assert isinstance(risk_flags(result), list)

def test_pipeline_writes_report(tmp_path):
    df = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5, 6, 7], "group": [0, 0, 0, 0, 1, 1, 1, 1], "label": [0, 0, 0, 1, 1, 1, 1, 1]})
    model = LogisticRegression().fit(df[["x", "group"]], df.label)
    result = run_audit(model, df, "label", "group", {"transparency": True}, tmp_path)
    assert (tmp_path / "audit_result.json").exists()
    assert (tmp_path / "ethicscan_audit_report.html").exists()
    assert "fairness" in result
