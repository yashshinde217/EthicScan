from typing import Any
import numpy as np
import pandas as pd

def explain_local(model, X: pd.DataFrame, sample_index: int = 0, max_features: int = 12) -> dict[str, Any]:
    sample_index = min(max(sample_index, 0), len(X) - 1)
    row = X.iloc[[sample_index]]
    try:
        import shap
        explainer = shap.Explainer(model, X)
        values = explainer(row)
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
    return {"sample_index": int(sample_index), "prediction": str(model.predict(row)[0]), "base_value": base_value, "method": method, "attributions": [{"feature": f, "value": float(v)} for f, v in pairs]}
