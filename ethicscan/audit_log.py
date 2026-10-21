from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

_LOG_FILE = Path("reports/audit_history.jsonl")

def log_audit(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Append one audit run entry to the JSONL history log and return the entry."""
    entry = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "summary": {
            "accuracy": result.get("accuracy"),
            "disparate_impact_ratio": result.get("fairness", {}).get("disparate_impact_ratio"),
            "demographic_parity_difference": result.get("fairness", {}).get("demographic_parity_difference"),
            "equal_opportunity_difference": result.get("fairness", {}).get("equal_opportunity_difference"),
            "equalized_odds_gap": result.get("fairness", {}).get("equalized_odds_gap"),
            "checklist_score": result.get("checklist", {}).get("score"),
            "risk_flag_count": len(result.get("risk_flags", [])),
        },
    }
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry

def load_history(log_file: str | Path = _LOG_FILE) -> list[dict[str, Any]]:
    path = Path(log_file)
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries
