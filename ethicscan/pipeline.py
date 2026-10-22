from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from .validation import validate_inputs, impute_for_model
from .fairness import audit_fairness, risk_flags
from .explainability import explain_local, explain_global
from .checklist import score_checklist, auto_check_checklist
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
) -> dict:
    model_features = list(getattr(model, "feature_names_in_", [])) or [c for c in df.columns if c != target]
    validation = validate_inputs(df, target, [sensitive_attribute], model_features)
    clean = impute_for_model(df)
    X = clean[model_features].copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = pd.factorize(X[col])[0]
    y = clean[target]
    predictions = model.predict(X)
    fairness = audit_fairness(y, predictions, clean[sensitive_attribute])
    flags = risk_flags(fairness)
    # Build partial result so auto_check can inspect it
    sensitive_in_features = sensitive_attribute in model_features
    partial = {
        "validation": {**validation.to_dict(), "sensitive_in_features": sensitive_in_features},
        "accuracy": float(accuracy_score(y, predictions)),
        "fairness": fairness,
        "risk_flags": flags,
        "global_explainability": explain_global(model, X),
    }
    # Auto-infer checklist responses if none provided
    if not checklist_responses:
        checklist_responses, evidence = auto_check_checklist(partial)
    checklist = score_checklist(checklist_responses, evidence)
    result = {
        **partial,
        "checklist": checklist,
        "explainability": explain_local(model, X, sample_index=sample_index),
        "recommendations": generate_recommendations(flags, checklist),
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
