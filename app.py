import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from ethicscan.pipeline import run_audit
from ethicscan.checklist import DEFAULT_CHECKLIST
from ethicscan.audit_log import load_history

st.set_page_config(page_title="EthicScan", page_icon="⚖️", layout="wide")
st.title("⚖️ EthicScan")
st.caption("Phase 2 · Fairness · Global & Local SHAP · NITI Aayog Checklist · Recommendations · Audit History")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1. Upload inputs")
    model_file = st.file_uploader("Trained model (.pkl/.joblib)", type=["pkl", "joblib"])
    data_file = st.file_uploader("Evaluation dataset (.csv)", type=["csv"])

    st.header("2. Map columns")
    if data_file:
        data_file.seek(0)
        preview = pd.read_csv(data_file)
        target = st.selectbox("Target column", preview.columns)
        sensitive = st.selectbox("Sensitive attribute", preview.columns)
        sample_index = st.number_input("Sample index for local SHAP", min_value=0, max_value=max(0, len(preview) - 1), value=0)
    else:
        preview = target = sensitive = None
        sample_index = 0

    st.header("3. NITI Aayog Checklist (7 Principles)")
    responses, evidence = {}, {}
    for item in DEFAULT_CHECKLIST:
        responses[item["id"]] = st.checkbox(item["principle"], value=item["id"] == "transparency")
        evidence[item["id"]] = st.text_input(f"Evidence — {item['principle']}", key=f"ev_{item['id']}", placeholder="Link or description")

    st.header("4. Reviewer sign-off")
    reviewer_name = st.text_input("Reviewer name")
    reviewer_comments = st.text_area("Reviewer comments", height=80)
    reviewer_approved = st.checkbox("Approve for deployment")

    run = st.button("▶ Run EthicScan audit", type="primary", use_container_width=True)

# ── Run audit ─────────────────────────────────────────────────────────────────
if not model_file or not data_file:
    st.info("Upload `sample_data/ethicscan_demo_model.joblib` and `sample_data/ethicscan_demo.csv` to run the demo.")
    st.stop()

data_file.seek(0)
df = pd.read_csv(data_file)

if run:
    temp_model = Path(".streamlit_uploaded_model.joblib")
    temp_model.write_bytes(model_file.getvalue())
    try:
        model = joblib.load(temp_model)
        result = run_audit(
            model, df, target, sensitive, responses,
            output_dir="reports", sample_index=int(sample_index),
            evidence=evidence, reviewer_name=reviewer_name,
            reviewer_comments=reviewer_comments,
        )
        result["reviewer"]["approved"] = reviewer_approved
        st.session_state["result"] = result
        st.session_state["model"] = model
    except Exception as exc:
        st.error(f"Audit failed: {type(exc).__name__}: {exc}")
        st.stop()

result = st.session_state.get("result")
if not result:
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["📊 Overview", "⚖️ Fairness", "🔍 Explainability", "📋 Checklist & Recommendations", "📜 Audit History", "📥 Export"])

# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tabs[0]:
    st.success("Audit completed.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{result['accuracy']:.1%}")
    c2.metric("Disparate Impact", f"{result['fairness'].get('disparate_impact_ratio', 0):.3f}", delta_color="inverse")
    c3.metric("Checklist Score", f"{result['checklist']['score']:.1f}%")
    c4.metric("Risk Flags", len(result["risk_flags"]))

    st.subheader("Risk Flags")
    if result["risk_flags"]:
        for flag in result["risk_flags"]:
            icon = "🔴" if flag["severity"] == "high" else "🟡"
            st.warning(f"{icon} **{flag['severity'].upper()}** — {flag['message']}")
    else:
        st.success("No threshold breaches detected.")

    # Risk heatmap
    st.subheader("Risk Heatmap")
    metrics = ["disparate_impact_ratio", "demographic_parity_difference", "equal_opportunity_difference", "equalized_odds_gap"]
    thresholds = {"disparate_impact_ratio": 0.8, "demographic_parity_difference": 0.1, "equal_opportunity_difference": 0.1, "equalized_odds_gap": 0.1}
    fairness = result["fairness"]
    heat_vals, heat_labels = [], []
    for m in metrics:
        v = fairness.get(m)
        if v is not None:
            thresh = thresholds[m]
            ratio = (thresh - v) / thresh if m == "disparate_impact_ratio" else abs(v) / thresh
            heat_vals.append(min(ratio, 2.0))
            heat_labels.append(f"{m.replace('_',' ').title()}<br>{v:.3f}")
    if heat_vals:
        fig = go.Figure(go.Heatmap(
            z=[heat_vals], x=heat_labels, y=["Risk level"],
            colorscale=[[0, "#147a4b"], [0.5, "#f59e0b"], [1.0, "#a61b1b"]],
            zmin=0, zmax=2, showscale=True,
        ))
        fig.update_layout(height=160, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Fairness ───────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Group Positive & True-Positive Rates")
    groups = fairness.get("groups", [])
    pos_rates = [fairness["positive_rates"].get(g, 0) for g in groups]
    tpr_rates = [fairness.get("true_positive_rates", {}).get(g) or 0 for g in groups]
    fpr_rates = [fairness.get("false_positive_rates", {}).get(g) or 0 for g in groups]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="Positive Rate", x=groups, y=pos_rates, marker_color="#0b5563"))
    fig_bar.add_trace(go.Bar(name="True Positive Rate", x=groups, y=tpr_rates, marker_color="#f59e0b"))
    fig_bar.add_trace(go.Bar(name="False Positive Rate", x=groups, y=fpr_rates, marker_color="#a61b1b"))
    fig_bar.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="DI threshold 0.8")
    fig_bar.update_layout(barmode="group", yaxis_range=[0, 1], height=350)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Subgroup Performance Breakdown")
    sp = fairness.get("subgroup_performance", {})
    if sp:
        sp_df = pd.DataFrame(sp).T.reset_index().rename(columns={"index": "Group"})
        st.dataframe(sp_df, use_container_width=True)

    st.subheader("Fairness Metrics Detail")
    fm = {k: v for k, v in fairness.items() if k not in ("groups", "group_counts", "positive_rates", "true_positive_rates", "false_positive_rates", "subgroup_performance")}
    st.json(fm)

# ── Tab 3: Explainability ─────────────────────────────────────────────────────
with tabs[2]:
    global_exp = result.get("global_explainability", {})
    local_exp = result.get("explainability", {})

    st.subheader("Global SHAP Feature Importance")
    summary = global_exp.get("summary", [])
    if summary:
        feat_names = [s["feature"] for s in summary]
        feat_vals = [s["mean_abs_shap"] for s in summary]
        fig_shap = px.bar(x=feat_vals, y=feat_names, orientation="h", labels={"x": "Mean |SHAP|", "y": "Feature"},
                          color=feat_vals, color_continuous_scale="Blues")
        fig_shap.update_layout(height=max(300, len(feat_names) * 35), showlegend=False)
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info(f"Global SHAP unavailable: {global_exp.get('method','')}")

    st.subheader("SHAP Dependency Plots (Top 3 Features)")
    dep_data = global_exp.get("dependency", [])[:3]
    if dep_data:
        cols = st.columns(min(len(dep_data), 3))
        for i, dep in enumerate(dep_data):
            with cols[i]:
                fig_dep = px.scatter(x=dep["feature_values"], y=dep["shap_values"],
                                     labels={"x": dep["feature"], "y": "SHAP value"},
                                     title=f"Dependency: {dep['feature']}", opacity=0.5)
                fig_dep.add_hline(y=0, line_dash="dash", line_color="grey")
                fig_dep.update_layout(height=280)
                st.plotly_chart(fig_dep, use_container_width=True)

    st.subheader(f"Local SHAP — Sample {local_exp.get('sample_index', '-')}")
    st.write(f"Prediction: **{local_exp.get('prediction', '-')}** | Base value: `{local_exp.get('base_value', 0):.4f}` | Method: {local_exp.get('method', '-')}")
    attrs = local_exp.get("attributions", [])
    if attrs:
        attr_df = pd.DataFrame(attrs)
        colors_list = ["#a61b1b" if v > 0 else "#0b5563" for v in attr_df["value"]]
        fig_local = go.Figure(go.Bar(x=attr_df["value"], y=attr_df["feature"], orientation="h",
                                     marker_color=colors_list))
        fig_local.add_vline(x=0, line_color="grey")
        fig_local.update_layout(height=max(250, len(attrs) * 30), xaxis_title="SHAP attribution")
        st.plotly_chart(fig_local, use_container_width=True)

# ── Tab 4: Checklist & Recommendations ───────────────────────────────────────
with tabs[3]:
    checklist = result["checklist"]
    st.subheader(f"NITI Aayog Checklist — {checklist['score']:.1f}% ({checklist['earned']}/{checklist['total']})")
    cl_df = pd.DataFrame([{
        "Principle": i["principle"], "Question": i["question"],
        "Response": i["response"], "Status": i["status"],
        "Evidence": i.get("evidence", "") or ""
    } for i in checklist["items"]])
    st.dataframe(cl_df, use_container_width=True)

    # Checklist score gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=checklist["score"],
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#0b5563"},
               "steps": [{"range": [0, 50], "color": "#fee2e2"}, {"range": [50, 80], "color": "#fef9c3"}, {"range": [80, 100], "color": "#dcfce7"}],
               "threshold": {"line": {"color": "red", "width": 2}, "thickness": 0.75, "value": 80}},
        title={"text": "Checklist Score (%)"},
    ))
    fig_gauge.update_layout(height=250)
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.subheader("Actionable Recommendations")
    recs = result.get("recommendations", [])
    if recs:
        for r in recs:
            icon = "🔴" if r["severity"] == "high" else "🟡"
            with st.expander(f"{icon} {r['title']}"):
                for a in r["actions"]:
                    st.write(f"• {a}")
                if r.get("references"):
                    st.caption("References: " + " | ".join(r["references"]))
    else:
        st.success("No recommendations triggered.")

    st.subheader("Reviewer Sign-off")
    rev = result.get("reviewer", {})
    st.write(f"**Reviewer:** {rev.get('name') or 'Not assigned'}")
    st.write(f"**Comments:** {rev.get('comments') or 'None provided'}")
    st.write(f"**Approved:** {'✅ Yes' if rev.get('approved') else '❌ Pending'}")

# ── Tab 5: Audit History ──────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Audit Run History")
    history = load_history()
    if history:
        hist_df = pd.DataFrame([{
            "Run ID": h["run_id"],
            "Timestamp": h["timestamp"],
            "Target": h["config"].get("target", ""),
            "Sensitive": h["config"].get("sensitive_attribute", ""),
            "Rows": h["config"].get("rows", ""),
            "Accuracy": h["summary"].get("accuracy"),
            "Disparate Impact": h["summary"].get("disparate_impact_ratio"),
            "Checklist %": h["summary"].get("checklist_score"),
            "Flags": h["summary"].get("risk_flag_count"),
        } for h in history])
        st.dataframe(hist_df, use_container_width=True)

        if len(hist_df) > 1:
            st.subheader("Disparate Impact Trend")
            fig_trend = px.line(hist_df, x="Timestamp", y="Disparate Impact", markers=True,
                                title="Disparate Impact Ratio Across Runs")
            fig_trend.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="Threshold 0.8")
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No audit history yet. Run an audit to start logging.")

# ── Tab 6: Export ─────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Download Reports")
    st.download_button("📄 Download audit_result.json", json.dumps(result, indent=2, default=str), "audit_result.json", "application/json")
    html_path = Path("reports/ethicscan_audit_report.html")
    if html_path.exists():
        st.download_button("🌐 Download HTML audit report", html_path.read_bytes(), "ethicscan_audit_report.html", "text/html")
    pdf_path = Path("reports/ethicscan_audit_report.pdf")
    if pdf_path.exists():
        st.download_button("📑 Download PDF audit report", pdf_path.read_bytes(), "ethicscan_audit_report.pdf", "application/pdf")
    log_path = Path("reports/audit_history.jsonl")
    if log_path.exists():
        st.download_button("📜 Download audit history log", log_path.read_bytes(), "audit_history.jsonl", "application/json")
