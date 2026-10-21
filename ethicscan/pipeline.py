from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from .validation import validate_inputs, impute_for_model
from .fairness import audit_fairness, risk_flags
from .explainability import explain_local
from .checklist import score_checklist
from .reporting import generate_html

def run_audit(model, df: pd.DataFrame, target: str, sensitive_attribute: str, checklist_responses: dict, output_dir: str | Path = "reports", sample_index: int = 0) -> dict:
    model_features = list(getattr(model, "feature_names_in_", [])) or [c for c in df.columns if c != target]
    validation = validate_inputs(df, target, [sensitive_attribute], model_features)
    clean = impute_for_model(df)
    X = clean[model_features]
    y = clean[target]
    predictions = model.predict(X)
    result = {"validation": validation.to_dict(), "accuracy": float(accuracy_score(y, predictions)), "fairness": audit_fairness(y, predictions, clean[sensitive_attribute]), "checklist": score_checklist(checklist_responses)}
    result["risk_flags"] = risk_flags(result["fairness"])
    result["explainability"] = explain_local(model, X, sample_index=sample_index)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    generate_html(result, out / "ethicscan_audit_report.html")
    return result

def load_model(path: str | Path):
    return joblib.load(path)
