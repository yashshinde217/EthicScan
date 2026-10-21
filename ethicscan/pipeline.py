from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from .validation import validate_inputs, impute_for_model
from .fairness import audit_fairness, risk_flags
from .explainability import explain_local, explain_global
from .checklist import score_checklist
from .recommendations import generate_recommendations
from .audit_log import log_audit
from .reporting import generate_html, generate_pdf

def run_audit(
    model,
    df: pd.DataFrame,
    target: str,
    sensitive_attribute: str,
    checklist_responses: dict,
    output_dir: str | Path = "reports",
    sample_index: int = 0,
    evidence: dict | None = None,
    reviewer_name: str = "",
    reviewer_comments: str = "",
) -> dict:
    model_features = list(getattr(model, "feature_names_in_", [])) or [c for c in df.columns if c != target]
    validation = validate_inputs(df, target, [sensitive_attribute], model_features)
    clean = impute_for_model(df)
    X = clean[model_features]
    y = clean[target]
    predictions = model.predict(X)
    fairness = audit_fairness(y, predictions, clean[sensitive_attribute])
    checklist = score_checklist(checklist_responses, evidence)
    flags = risk_flags(fairness)
    result = {
        "validation": validation.to_dict(),
        "accuracy": float(accuracy_score(y, predictions)),
        "fairness": fairness,
        "checklist": checklist,
        "risk_flags": flags,
        "explainability": explain_local(model, X, sample_index=sample_index),
        "global_explainability": explain_global(model, X),
        "recommendations": generate_recommendations(flags, checklist),
        "reviewer": {"name": reviewer_name, "comments": reviewer_comments, "approved": False},
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    generate_html(result, out / "ethicscan_audit_report.html")
    generate_pdf(result, out / "ethicscan_audit_report.pdf")
    config = {"target": target, "sensitive_attribute": sensitive_attribute, "sample_index": sample_index, "rows": len(df), "features": model_features}
    log_audit(result, config)
    return result

def load_model(path: str | Path):
    return joblib.load(path)
