from dataclasses import dataclass, asdict
from typing import Any
import pandas as pd

@dataclass
class ValidationResult:
    ok: bool
    rows: int
    columns: int
    missing_values: dict[str, int]
    missing_total: int
    target_present: bool
    sensitive_present: dict[str, bool]
    missing_columns: list[str]
    messages: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def validate_inputs(df: pd.DataFrame, target: str, sensitive: list[str], model_features: list[str] | None = None) -> ValidationResult:
    messages: list[str] = []
    missing_values = df.isna().sum()
    missing_values = {k: int(v) for k, v in missing_values[missing_values > 0].items()}
    missing_columns = []
    if model_features:
        missing_columns = [c for c in model_features if c not in df.columns]
        if missing_columns:
            messages.append(f"Dataset is missing model features: {', '.join(missing_columns)}")
    target_present = target in df.columns
    if not target_present:
        messages.append(f"Target column '{target}' was not found.")
    sensitive_present = {col: col in df.columns for col in sensitive}
    absent_sensitive = [c for c, present in sensitive_present.items() if not present]
    if absent_sensitive:
        messages.append(f"Sensitive columns not found: {', '.join(absent_sensitive)}")
    if missing_values:
        messages.append("Missing values detected; numeric values will be median-imputed and categorical values mode-imputed.")
    if not messages:
        messages.append("Input schema and required columns look valid.")
    ok = target_present and not missing_columns and all(sensitive_present.values())
    return ValidationResult(ok, len(df), len(df.columns), missing_values, int(sum(missing_values.values())), target_present, sensitive_present, missing_columns, messages)

def impute_for_model(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in result.columns:
        if result[col].isna().any():
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(result[col].median())
            else:
                mode = result[col].mode(dropna=True)
                result[col] = result[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    return result
