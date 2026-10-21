# EthicScan Phase 1 — MVP

EthicScan is a local Streamlit prototype for auditing a trained machine-learning model across **basic fairness**, **local SHAP explainability**, and a **static Responsible AI checklist**. It is an audit aid, not a legal compliance decision.

## What is included

- `app.py` — Streamlit interface
- `ethicscan/validation.py` — schema, missing value, target, and sensitive-column checks
- `ethicscan/fairness.py` — demographic parity difference, disparate impact ratio, equal opportunity difference, and threshold flags
- `ethicscan/explainability.py` — local SHAP attributions with a clear fallback message if a model is unsupported
- `ethicscan/checklist.py` — Phase 1 static checklist
- `ethicscan/pipeline.py` — end-to-end audit orchestration
- `ethicscan/reporting.py` — static HTML report generation
- `generate_dummy_data.py` — reproducible demo CSV and Logistic Regression model
- `tests/` — smoke tests
- `GUIDE.md` — detailed VS Code setup, testing, demo, and presentation workflow

## Quick start (VS Code terminal)

### Windows PowerShell
```powershell
cd path\to\ethicscan_phase1
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_dummy_data.py
pytest -q
streamlit run app.py
```

### macOS/Linux
```bash
cd /path/to/ethicscan_phase1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_dummy_data.py
pytest -q
streamlit run app.py
```

Open the URL shown by Streamlit, normally `http://localhost:8501`. Upload:

- `sample_data/ethicscan_demo_model.joblib`
- `sample_data/ethicscan_demo.csv`

Choose `approved` as target and `gender` as sensitive attribute, answer the checklist, then click **Run EthicScan audit**.

## Thresholds used in Phase 1

- Disparate impact ratio `< 0.80`: high-severity potential adverse-impact flag
- Absolute demographic parity difference `> 0.10`: medium-severity flag
- Absolute equal opportunity difference `> 0.10`: medium-severity flag

These are screening thresholds and must be interpreted in context.
