import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from ethicscan.pipeline import run_audit
from ethicscan.checklist import DEFAULT_CHECKLIST

st.set_page_config(page_title="EthicScan", page_icon="⚖️", layout="wide")
st.title("EthicScan")
st.caption("Phase 1 MVP · Fairness + local SHAP explainability + Responsible AI checklist")
with st.sidebar:
    st.header("1. Upload inputs")
    model_file = st.file_uploader("Trained model (.pkl/.joblib)", type=["pkl", "joblib"])
    data_file = st.file_uploader("Evaluation dataset (.csv)", type=["csv"])
    st.header("2. Map columns")
    if data_file:
        preview = pd.read_csv(data_file)
        target = st.selectbox("Target column", preview.columns)
        sensitive = st.selectbox("Sensitive attribute", preview.columns)
        sample_index = st.number_input("Sample index for local SHAP", min_value=0, max_value=max(0, len(preview)-1), value=0)
    else:
        target = sensitive = None
        sample_index = 0
    st.header("3. Checklist")
    responses = {item["id"]: st.checkbox(item["principle"], value=True if item["id"] == "transparency" else False) for item in DEFAULT_CHECKLIST}
    run = st.button("Run EthicScan audit", type="primary", use_container_width=True)

if not model_file or not data_file:
    st.info("Upload the included sample_data/ethicscan_demo_model.joblib and sample_data/ethicscan_demo.csv to run the demo.")
    st.markdown("### Workflow\n1. Upload model and evaluation CSV.\n2. Map target and sensitive columns.\n3. Answer the checklist.\n4. Run audit and download the JSON/HTML evidence.")
else:
    data_file.seek(0)
    df = pd.read_csv(data_file)
    if run:
        temp_model = Path(".streamlit_uploaded_model.joblib")
        temp_model.write_bytes(model_file.getvalue())
        try:
            model = joblib.load(temp_model)
            result = run_audit(model, df, target, sensitive, responses, output_dir="reports", sample_index=int(sample_index))
            st.session_state["result"] = result
        except Exception as exc:
            st.error(f"Audit failed: {type(exc).__name__}: {exc}")

result = st.session_state.get("result")
if result:
    st.success("Audit completed. Review the evidence below and download the static report.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{result['accuracy']:.1%}")
    c2.metric("Disparate impact", f"{result['fairness'].get('disparate_impact_ratio', 0):.3f}")
    c3.metric("Checklist score", f"{result['checklist']['score']:.1f}%")
    st.subheader("Risk flags")
    if result["risk_flags"]:
        for flag in result["risk_flags"]:
            st.warning(flag["message"])
    else:
        st.success("No Phase 1 threshold breach detected.")
    st.subheader("Local SHAP attributions")
    st.dataframe(pd.DataFrame(result["explainability"]["attributions"]))
    st.subheader("Fairness details")
    st.json(result["fairness"])
    st.download_button("Download audit_result.json", json.dumps(result, indent=2, default=str), "audit_result.json", "application/json")
    report_path = Path("reports/ethicscan_audit_report.html")
    if report_path.exists():
        st.download_button("Download HTML audit report", report_path.read_bytes(), "ethicscan_audit_report.html", "text/html")
