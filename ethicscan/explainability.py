from typing import Any
import numpy as np
import pandas as pd

def _unwrap(model, X: pd.DataFrame):
    """If model is a sklearn Pipeline, return (final_estimator, transformed_X)."""
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            X_t = pd.DataFrame(
                model[:-1].transform(X),
                columns=X.columns,
                index=X.index,
            )
            return model[-1], X_t
    except Exception:
        pass
    return model, X

def _get_shap_values(model, X: pd.DataFrame):
    import shap
    estimator, X_t = _unwrap(model, X)
    # Try TreeExplainer first (works for GBM, RF, XGBoost — no background sample needed)
    try:
        te = shap.TreeExplainer(estimator)
        raw = te.shap_values(X_t)
        # binary classifiers return list [neg_class, pos_class] or 3-D array
        if isinstance(raw, list):
            raw = raw[1] if len(raw) > 1 else raw[0]
        raw = np.asarray(raw)
        if raw.ndim == 3:
            raw = raw[:, :, 1 if raw.shape[2] > 1 else 0]
        return te, None, raw
    except Exception:
        pass
    # Generic fallback with a capped background sample
    bg = X_t.sample(min(100, len(X_t)), random_state=42)
    explainer = shap.Explainer(estimator, bg)
    values = explainer(X_t)
    raw = values.values
    if raw.ndim == 3:
        raw = raw[:, :, 1 if raw.shape[2] > 1 else 0]
    return explainer, values, np.asarray(raw)

def explain_local(model, X: pd.DataFrame, sample_index: int = 0, max_features: int = 12) -> dict[str, Any]:
    sample_index = min(max(sample_index, 0), len(X) - 1)
    try:
        import shap
        estimator, X_t = _unwrap(model, X)
        row_t = X_t.iloc[[sample_index]]
        try:
            te = shap.TreeExplainer(estimator)
            raw = te.shap_values(row_t)
            if isinstance(raw, list):
                raw = raw[1] if len(raw) > 1 else raw[0]
            contribution = np.asarray(raw).reshape(-1)
            base_value = float(te.expected_value[1] if hasattr(te.expected_value, '__len__') else te.expected_value)
        except Exception:
            bg = X_t.sample(min(100, len(X_t)), random_state=42)
            explainer = shap.Explainer(estimator, bg)
            values = explainer(row_t)
            raw = values.values
            if raw.ndim == 3:
                raw = raw[:, :, 1 if raw.shape[2] > 1 else 0]
            contribution = np.asarray(raw).reshape(-1)
            base = np.asarray(values.base_values).reshape(-1)
            base_value = float(base[0]) if len(base) else 0.0
        method = "SHAP"
    except Exception as exc:
        contribution = np.zeros(X.shape[1], dtype=float)
        base_value = 0.0
        method = f"Fallback (SHAP unavailable for this model: {type(exc).__name__})"
    pairs = sorted(zip(X.columns, contribution), key=lambda x: abs(float(x[1])), reverse=True)[:max_features]
    return {"sample_index": int(sample_index), "prediction": str(model.predict(X.iloc[[sample_index]])[0]), "base_value": base_value, "method": method, "attributions": [{"feature": f, "value": float(v)} for f, v in pairs]}

def explain_global(model, X: pd.DataFrame, max_features: int = 12) -> dict[str, Any]:
    """Global SHAP feature importance and dependency data."""
    try:
        _, _, shap_matrix = _get_shap_values(model, X)
        mean_abs = np.abs(shap_matrix).mean(axis=0)
        ranked = sorted(zip(X.columns, mean_abs), key=lambda x: float(x[1]), reverse=True)[:max_features]
        # Dependency data: for each top feature, store feature values + shap values for scatter
        dependency = []
        for feat, _ in ranked:
            idx = list(X.columns).index(feat)
            dependency.append({
                "feature": feat,
                "feature_values": X[feat].tolist(),
                "shap_values": shap_matrix[:, idx].tolist(),
            })
        return {
            "method": "SHAP",
            "summary": [{"feature": f, "mean_abs_shap": float(v)} for f, v in ranked],
            "dependency": dependency,
        }
    except Exception as exc:
        return {"method": f"Fallback (SHAP unavailable: {type(exc).__name__})", "summary": [], "dependency": []}
