# EthicScan Phase 1 — Step-by-step guide

## A. Open the project in VS Code

1. Download and unzip the project.
2. Open VS Code.
3. Select **File → Open Folder** and choose the `EthicScan` folder.
4. Open **Terminal → New Terminal**.
5. Confirm the terminal is inside the project folder. You should see files such as `app.py`, `requirements.txt`, and `ethicscan/`.

## B. Create the Python environment

Use the command for your operating system. Python 3.11 or later is required (tested on Python 3.13).

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

If you previously created a `.venv` with a Python version that has since been removed, delete it and recreate:
```powershell
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

In VS Code, press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and select the interpreter inside `.venv`.

## C. Install dependencies and create the demo inputs

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_dummy_data.py
```

The generator creates:

- `sample_data/ethicscan_demo.csv`: 240-row evaluation dataset
- `sample_data/ethicscan_demo_model.joblib`: trained Logistic Regression model

The dummy data includes `gender`, `approved`, and a few missing values so you can demonstrate validation and fairness auditing.

## D. Test the project without the UI

```bash
pytest -q
```

Expected result: all tests pass. The tests cover missing-value detection, fairness metrics/risk flags, and report generation.

## E. Run the Streamlit application

```bash
streamlit run app.py
```

1. Open the local URL printed in the terminal (normally `http://localhost:8501`).
2. Upload `sample_data/ethicscan_demo_model.joblib`.
3. Upload `sample_data/ethicscan_demo.csv`.
4. Select target column `approved`.
5. Select sensitive attribute `gender`.
6. Leave sample index at `0` for the first local SHAP explanation.
7. Answer the Responsible AI checklist.
8. Click **Run EthicScan audit**.
9. Review accuracy, disparate impact, checklist score, risk flags, SHAP attributions, and fairness JSON.
10. Download `audit_result.json` and the HTML audit report.

Generated files also appear in the local `reports/` folder.

## F. How to demonstrate the full workflow to your group

### Suggested 7-minute demonstration

1. **Problem (45 seconds):** "A model can be accurate but still produce unequal outcomes or be difficult to explain. EthicScan is an audit aid that checks these risks."
2. **Architecture (60 seconds):** Show `User inputs → validation → fairness engine + SHAP engine + checklist → risk flags → report`.
3. **Inputs (60 seconds):** Show the model file, CSV, target column, sensitive attribute, and checklist.
4. **Validation (45 seconds):** Point out the generated missing values in the demo CSV and explain that the pipeline reports them and imputes values for prediction.
5. **Fairness (75 seconds):** Explain demographic parity difference and disparate impact ratio. Highlight the `< 0.8` screening rule.
6. **Explainability (60 seconds):** Show local SHAP features. Explain that positive attribution pushes toward the predicted class and negative attribution pushes away from it.
7. **Governance/report (60 seconds):** Show checklist score, risk flags, and download the HTML report. Emphasize that the result supports human review; it does not certify legal compliance.

### What each group member can explain

- **Data/model member:** input files, schema validation, missing values, baseline model.
- **Fairness member:** group rates, demographic parity difference, disparate impact.
- **Explainability member:** SHAP local attribution and its limitations.
- **Governance/report member:** checklist, thresholds, report output, human-in-the-loop limitations.

## G. Suggested live test cases

### Test 1 — Normal complete run
Use the generated model and CSV. Show all outputs and download the report.

### Test 2 — Missing target column
Make a copy of the CSV and remove `approved`. Upload it. The system should show an audit error or validation issue. Explain that target alignment is required.

### Test 3 — Change the sensitive attribute
Select `age` instead of `gender`. Explain that an audit conclusion depends on the protected attribute being evaluated.

### Test 4 — Governance gap
Untick all checklist items and rerun. The checklist score should fall to 0%, demonstrating qualitative governance risk.

## H. How to explain the code

- `app.py` is the front end. Calls `data_file.seek(0)` before the main CSV read to reset the Streamlit upload cursor after the sidebar preview.
- `pipeline.py` coordinates the engines.
- `validation.py` checks the input contract.
- `fairness.py` computes group metrics and flags threshold breaches.
- `explainability.py` calls SHAP for one sample.
- `checklist.py` converts yes/no answers into a score.
- `reporting.py` creates the HTML evidence artifact.

## I. Phase 1 limitations to state honestly

- Only local SHAP is implemented in this MVP; global plots are Phase 2.
- The checklist is a simplified static checklist; evidence uploads are Phase 2.
- Thresholds are configurable screening rules, not universal legal standards.
- Results depend on the evaluation data and selected sensitive attribute.
- The tool does not prove a model is ethical or legally compliant.
- Human/domain-expert review remains necessary before deployment.
