import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from ethicscan.pipeline import run_audit
from ethicscan.checklist import DEFAULT_CHECKLIST
from ethicscan.audit_log import load_history

st.set_page_config(page_title="EthicScan", page_icon="⚖️", layout="wide", initial_sidebar_state="expanded")

# ── Dark mode palette ─────────────────────────────────────────────────────────
# bg:      #0d1117   surface: #161b22   border: #30363d
# accent:  #38bdf8   text:    #e6edf3   muted:  #8b949e
# good:    #3fb950   warn:    #d29922   danger: #f85149

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
}
.block-container { padding: 1.8rem 2.5rem; background: #0d1117; }
.stApp { background: #0d1117; }

/* hide streamlit default header chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #30363d !important;
}
section[data-testid="stSidebar"] > div { background: #0d1117 !important; }
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #38bdf8 !important; font-size: 0.72rem !important; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    border-bottom: 1px solid #30363d; padding-bottom: 5px; margin-top: 1.4rem;
}
section[data-testid="stSidebar"] label { color: #8b949e !important; font-size: 0.8rem !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: #38bdf8 !important; color: #0d1117 !important;
    font-weight: 700; border-radius: 8px; border: none;
    width: 100%; padding: 0.55rem; margin-top: 0.6rem;
    transition: opacity 0.15s;
}
section[data-testid="stSidebar"] .stButton > button:hover { opacity: 0.85; }
section[data-testid="stSidebar"] hr { border-color: #30363d !important; }

/* inputs dark */
.stTextInput input, .stTextArea textarea, .stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background: #161b22 !important; border: 1px solid #30363d !important;
    color: #e6edf3 !important; border-radius: 6px !important;
}
.stFileUploader { background: #161b22 !important; border: 1px dashed #30363d !important; border-radius: 8px !important; }
.stCheckbox label { color: #c9d1d9 !important; font-size: 0.83rem !important; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0f2942 0%, #0c3a52 50%, #0e4d63 100%);
    border: 1px solid #30363d; border-radius: 14px;
    padding: 1.6rem 2rem; margin-bottom: 1.4rem;
    display: flex; align-items: center; gap: 1.2rem;
}
.hero-icon { font-size: 2.6rem; line-height: 1; }
.hero h1 { color: #e6edf3; font-size: 1.9rem; font-weight: 800; margin: 0; letter-spacing: -0.3px; }
.hero-sub { color: #38bdf8; font-size: 0.78rem; margin: 3px 0 0 0; letter-spacing: 0.02em; }

/* ── KPI cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin: 0 0 1.4rem 0; }
.kpi-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 1.1rem 1.3rem; border-top: 3px solid #38bdf8;
}
.kpi-card.warn  { border-top-color: #d29922; }
.kpi-card.danger{ border-top-color: #f85149; }
.kpi-card.good  { border-top-color: #3fb950; }
.kpi-label { font-size: 0.7rem; font-weight: 600; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-value { font-size: 1.9rem; font-weight: 800; color: #e6edf3; line-height: 1.15; margin-top: 3px; }
.kpi-sub   { font-size: 0.7rem; color: #6e7681; margin-top: 2px; }

/* ── Section title ── */
.section-title {
    font-size: 0.78rem; font-weight: 700; color: #8b949e;
    text-transform: uppercase; letter-spacing: 0.1em;
    border-bottom: 1px solid #30363d; padding-bottom: 5px; margin: 1.4rem 0 0.8rem 0;
}

/* ── Flag cards ── */
.flag-high   { background: #1a0a0a; border-left: 3px solid #f85149; border-radius: 8px; padding: 0.75rem 1rem; margin: 5px 0; display:flex; gap:0.7rem; align-items:flex-start; }
.flag-medium { background: #1a1400; border-left: 3px solid #d29922; border-radius: 8px; padding: 0.75rem 1rem; margin: 5px 0; display:flex; gap:0.7rem; align-items:flex-start; }
.flag-badge-high   { background:#f85149; color:#fff; font-size:0.6rem; font-weight:700; padding:2px 6px; border-radius:4px; text-transform:uppercase; white-space:nowrap; margin-top:2px; }
.flag-badge-medium { background:#d29922; color:#fff; font-size:0.6rem; font-weight:700; padding:2px 6px; border-radius:4px; text-transform:uppercase; white-space:nowrap; margin-top:2px; }
.flag-msg { font-size: 0.85rem; color: #c9d1d9; line-height: 1.5; }

/* ── Rec cards ── */
.rec-high   { background:#1a0a0a; border:1px solid #3d1a1a; border-radius:10px; padding:1rem 1.2rem; margin:6px 0; }
.rec-medium { background:#1a1400; border:1px solid #3d3000; border-radius:10px; padding:1rem 1.2rem; margin:6px 0; }
.rec-title  { font-weight:700; font-size:0.88rem; color:#e6edf3; margin-bottom:0.4rem; }
.rec-action { font-size:0.81rem; color:#8b949e; margin:3px 0; }
.rec-ref    { font-size:0.72rem; color:#6e7681; margin-top:6px; }

/* ── Reviewer box ── */
.reviewer-box { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:1.2rem 1.5rem; margin-top:0.8rem; }
.rev-label { font-size:0.68rem; font-weight:700; color:#38bdf8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:2px; }
.rev-value { font-size:0.88rem; color:#c9d1d9; margin-bottom:0.7rem; }

/* ── Landing ── */
.landing { text-align:center; padding:5rem 2rem; }
.landing h2 { color:#e6edf3; font-size:1.5rem; font-weight:700; margin-bottom:0.5rem; }
.landing p  { color:#8b949e; font-size:0.9rem; max-width:460px; margin:0 auto 0.3rem; }
.step-grid  { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:2rem 0; }
.step-card  { background:#161b22; border:1px solid #30363d; border-radius:12px; padding:1.2rem 1rem; }
.step-num   { width:30px; height:30px; background:#38bdf8; color:#0d1117; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:0.85rem; margin:0 auto 0.6rem; }
.step-card p { font-size:0.8rem; color:#8b949e; margin:0; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background:#161b22; border-radius:10px; padding:4px; gap:3px; border:1px solid #30363d; }
.stTabs [data-baseweb="tab"] { border-radius:7px; padding:0.35rem 0.9rem; font-size:0.82rem; font-weight:600; color:#8b949e; }
.stTabs [aria-selected="true"] { background:#0d1117 !important; color:#38bdf8 !important; }

/* ── Dataframe ── */
.stDataFrame { border-radius:10px; overflow:hidden; }
[data-testid="stDataFrame"] { background:#161b22 !important; }

/* ── Metrics ── */
[data-testid="stMetric"] { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:0.8rem 1rem; }
[data-testid="stMetricLabel"] { color:#8b949e !important; font-size:0.75rem !important; }
[data-testid="stMetricValue"] { color:#e6edf3 !important; }

/* ── Download buttons ── */
.stDownloadButton > button {
    background:#161b22 !important; border:1px solid #30363d !important;
    color:#e6edf3 !important; border-radius:8px !important; font-weight:600 !important;
    transition: border-color 0.15s;
}
.stDownloadButton > button:hover { border-color:#38bdf8 !important; color:#38bdf8 !important; }

/* ── Alerts ── */
.stSuccess { background:#0d2818 !important; border:1px solid #3fb950 !important; color:#3fb950 !important; border-radius:8px !important; }
.stInfo    { background:#0d1f2e !important; border:1px solid #38bdf8 !important; color:#38bdf8 !important; border-radius:8px !important; }
.stError   { background:#1a0a0a !important; border:1px solid #f85149 !important; color:#f85149 !important; border-radius:8px !important; }

/* scrollbar */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:#0d1117; }
::-webkit-scrollbar-thumb { background:#30363d; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark template ──────────────────────────────────────────────────────
_DARK = dict(
    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
    font=dict(family="Inter, sans-serif", size=12, color="#c9d1d9"),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    margin=dict(l=0, r=0, t=30, b=0),
)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-icon">⚖️</div>
  <div>
    <h1>EthicScan</h1>
    <div class="hero-sub">Fairness &nbsp;·&nbsp; Explainability &nbsp;·&nbsp; Governance</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Upload")
    model_file = st.file_uploader("Model (.pkl / .joblib)", type=["pkl", "joblib"])
    data_file  = st.file_uploader("Dataset (.csv)", type=["csv"])

    st.markdown("### Columns")
    if data_file:
        data_file.seek(0)
        preview   = pd.read_csv(data_file)
        target    = st.selectbox("Target", preview.columns)
        sensitive = st.selectbox("Sensitive attribute", preview.columns)
        sample_index = st.number_input("SHAP sample index", min_value=0,
                                       max_value=max(0, len(preview) - 1), value=0)
    else:
        preview = target = sensitive = None
        sample_index = 0

    st.markdown("### Checklist")
    responses, evidence = {}, {}
    for item in DEFAULT_CHECKLIST:
        responses[item["id"]] = st.checkbox(item["principle"], value=item["id"] == "transparency")
        evidence[item["id"]]  = st.text_input(item["principle"],
                                               key=f"ev_{item['id']}", placeholder="Evidence link or note",
                                               label_visibility="collapsed")

    st.markdown("### Reviewer")
    reviewer_name     = st.text_input("Name", placeholder="Reviewer name")
    reviewer_comments = st.text_area("Comments", height=70, placeholder="Notes...")
    reviewer_approved = st.checkbox("Approve for deployment")

    st.markdown("---")
    run = st.button("Run Audit", type="primary", use_container_width=True)

# ── Landing ───────────────────────────────────────────────────────────────────
if not model_file or not data_file:
    st.markdown("""
    <div class="landing">
      <h2>AI Model Audit Platform</h2>
      <p>Upload your model and dataset to run a fairness, explainability, and governance audit.</p>
      <div class="step-grid">
        <div class="step-card"><div class="step-num">1</div><p>Upload model</p></div>
        <div class="step-card"><div class="step-num">2</div><p>Upload dataset</p></div>
        <div class="step-card"><div class="step-num">3</div><p>Fill checklist</p></div>
        <div class="step-card"><div class="step-num">4</div><p>Run &amp; review</p></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

data_file.seek(0)
df = pd.read_csv(data_file)

# ── Run audit ─────────────────────────────────────────────────────────────────
if run:
    temp_model = Path(".streamlit_uploaded_model.joblib")
    temp_model.write_bytes(model_file.getvalue())
    with st.spinner("Running audit..."):
        try:
            model  = joblib.load(temp_model)
            result = run_audit(
                model, df, target, sensitive, responses,
                output_dir="reports", sample_index=int(sample_index),
                evidence=evidence, reviewer_name=reviewer_name,
                reviewer_comments=reviewer_comments,
            )
            result["reviewer"]["approved"] = reviewer_approved
            st.session_state["result"] = result
        except Exception as exc:
            st.error(f"Audit failed: {type(exc).__name__}: {exc}")
            st.stop()

result = st.session_state.get("result")
if not result:
    st.info("Configure inputs in the sidebar and click **Run Audit**.")
    st.stop()

fairness  = result["fairness"]
checklist = result["checklist"]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["Overview", "Fairness", "Explainability", "Checklist", "History", "Export"])

# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tabs[0]:
    flags      = result["risk_flags"]
    flag_count = len(flags)
    di_val     = fairness.get("disparate_impact_ratio", 0)
    di_class   = "danger" if di_val < 0.8 else ("warn" if di_val < 0.9 else "good")
    cl_score   = checklist["score"]
    cl_class   = "good" if cl_score >= 80 else ("warn" if cl_score >= 50 else "danger")
    fc_class   = "danger" if flag_count >= 3 else ("warn" if flag_count >= 1 else "good")

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card good">
        <div class="kpi-label">Accuracy</div>
        <div class="kpi-value">{result['accuracy']:.1%}</div>
        <div class="kpi-sub">Overall model accuracy</div>
      </div>
      <div class="kpi-card {di_class}">
        <div class="kpi-label">Disparate Impact</div>
        <div class="kpi-value">{di_val:.3f}</div>
        <div class="kpi-sub">Threshold &lt; 0.80</div>
      </div>
      <div class="kpi-card {cl_class}">
        <div class="kpi-label">Checklist</div>
        <div class="kpi-value">{cl_score:.0f}%</div>
        <div class="kpi-sub">{checklist['earned']} / {checklist['total']} principles</div>
      </div>
      <div class="kpi-card {fc_class}">
        <div class="kpi-label">Risk Flags</div>
        <div class="kpi-value">{flag_count}</div>
        <div class="kpi-sub">Active breaches</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Risk Flags</div>', unsafe_allow_html=True)
    if flags:
        for flag in flags:
            sev = flag["severity"]
            st.markdown(
                f'<div class="flag-{sev}">'
                f'<span class="flag-badge-{sev}">{sev}</span>'
                f'<span class="flag-msg">{flag["message"]}</span></div>',
                unsafe_allow_html=True)
    else:
        st.success("No threshold breaches detected.")

    st.markdown('<div class="section-title">Risk Heatmap</div>', unsafe_allow_html=True)
    import numpy as np
    _hm_defs = [
        ("Disparate Impact",        "disparate_impact_ratio",        0.8,  True),
        ("Demographic Parity Diff", "demographic_parity_difference", 0.1,  False),
        ("Equal Opportunity Diff",  "equal_opportunity_difference",  0.1,  False),
        ("Equalized Odds Gap",      "equalized_odds_gap",            0.1,  False),
    ]
    hm_labels, hm_scores, hm_raw = [], [], []
    for label, key, thresh, invert in _hm_defs:
        v = fairness.get(key)
        if v is None:
            continue
        score = (thresh - v) / thresh if invert else abs(v) / thresh
        hm_scores.append(float(min(max(score, 0.0), 1.0)))
        hm_labels.append(label)
        hm_raw.append(v)
    if hm_scores:
        n  = len(hm_scores)
        W  = 120   # columns per metric
        H  = 80    # rows
        # X axis: smooth interpolation across all metrics
        x_scores = np.interp(np.linspace(0, n - 1, n * W), range(n), hm_scores)
        # Y axis: vertical gradient — top row = score value, bottom row = 0
        # so color fades from full intensity at top to dark at bottom
        y_fade   = np.linspace(1.0, 0.15, H)
        # 2D matrix: each row scales x_scores by the y fade factor
        z_2d     = np.outer(y_fade, x_scores)
        fig_heat = go.Figure(go.Heatmap(
            z=z_2d,
            colorscale=[
                [0.00, "#0d1117"],
                [0.15, "#0d2818"],
                [0.35, "#3fb950"],
                [0.55, "#d29922"],
                [0.75, "#e05c2a"],
                [1.00, "#f85149"],
            ],
            zmin=0, zmax=1,
            showscale=True,
            colorbar=dict(
                title=dict(text="Risk", font=dict(color="#8b949e", size=11)),
                thickness=12, len=1.0,
                tickvals=[0, 0.35, 0.55, 0.75, 1.0],
                ticktext=["Safe", "Low", "Medium", "High", "Critical"],
                tickfont=dict(color="#8b949e", size=10),
                outlinewidth=0, bgcolor="#0d1117",
            ),
            hoverinfo="skip",
        ))
        # thin dark dividers between metric zones
        for i in range(1, n):
            fig_heat.add_vline(x=i * W - 0.5, line_color="#0d1117", line_width=2)
        # metric label + value at top of each zone
        for i in range(n):
            cx     = i * W + W / 2 - 0.5
            s      = hm_scores[i]
            status = "PASS" if s < 0.35 else ("WARN" if s < 0.75 else "FAIL")
            t_color = "#3fb950" if s < 0.35 else ("#d29922" if s < 0.75 else "#f85149")
            fig_heat.add_annotation(
                x=cx, y=4,
                text=f"<b>{hm_labels[i]}</b>",
                showarrow=False,
                font=dict(color="#ffffff", size=12, family="Inter, sans-serif"),
                align="center",
            )
            fig_heat.add_annotation(
                x=cx, y=H - 8,
                text=f"<b style='color:{t_color}'>{hm_raw[i]:.3f}</b>  <span style='font-size:10px'>{status}</span>",
                showarrow=False,
                font=dict(color=t_color, size=12, family="Inter, sans-serif"),
                align="center",
            )
        fig_heat.update_layout(
            height=260,
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            margin=dict(l=0, r=75, t=10, b=10),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False),
            font=dict(family="Inter, sans-serif", color="#c9d1d9"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ── Tab 2: Fairness ───────────────────────────────────────────────────────────
with tabs[1]:
    groups    = fairness.get("groups", [])
    pos_rates = [fairness["positive_rates"].get(g, 0) for g in groups]
    tpr_rates = [fairness.get("true_positive_rates", {}).get(g) or 0 for g in groups]
    fpr_rates = [fairness.get("false_positive_rates", {}).get(g) or 0 for g in groups]

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown('<div class="section-title">Group Rates</div>', unsafe_allow_html=True)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Positive Rate",       x=groups, y=pos_rates, marker_color="#38bdf8", marker_line_width=0))
        fig_bar.add_trace(go.Bar(name="True Positive Rate",  x=groups, y=tpr_rates, marker_color="#3fb950", marker_line_width=0))
        fig_bar.add_trace(go.Bar(name="False Positive Rate", x=groups, y=fpr_rates, marker_color="#f85149", marker_line_width=0))
        fig_bar.add_hline(y=0.8, line_dash="dot", line_color="#f85149",
                          annotation_text="0.80 threshold", annotation_font_size=10,
                          annotation_font_color="#f85149")
        fig_bar.update_layout(
            barmode="group", yaxis_range=[0, 1], height=340,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            **_DARK,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Metrics</div>', unsafe_allow_html=True)
        for label, key in [
            ("Disparate Impact Ratio",      "disparate_impact_ratio"),
            ("Demographic Parity Diff",     "demographic_parity_difference"),
            ("Equal Opportunity Diff",      "equal_opportunity_difference"),
            ("Equalized Odds Gap",          "equalized_odds_gap"),
        ]:
            v = fairness.get(key)
            if v is not None:
                st.metric(label, f"{v:.4f}")

    sp = fairness.get("subgroup_performance", {})
    if sp:
        st.markdown('<div class="section-title">Subgroup Performance</div>', unsafe_allow_html=True)
        sp_df = pd.DataFrame(sp).T.reset_index().rename(columns={"index": "Group"}).round(3)
        st.dataframe(sp_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Performance Radar</div>', unsafe_allow_html=True)
        radar_cats = ["accuracy","precision","recall","f1","positive_rate"]
        fig_radar  = go.Figure()
        palette    = ["#38bdf8","#3fb950","#d29922","#a855f7"]
        for i, (g, v) in enumerate(sp.items()):
            vals = [v.get(c, 0) for c in radar_cats] + [v.get(radar_cats[0], 0)]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=radar_cats + [radar_cats[0]],
                fill="toself", name=f"Group {g}",
                line_color=palette[i % len(palette)], opacity=0.75,
            ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0,1], gridcolor="#30363d", color="#8b949e"),
                angularaxis=dict(color="#8b949e"),
                bgcolor="#161b22",
            ),
            height=360, paper_bgcolor="#161b22",
            legend=dict(orientation="h", y=-0.15, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            font=dict(family="Inter, sans-serif", size=12, color="#c9d1d9"),
            margin=dict(l=40, r=40, t=20, b=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ── Tab 3: Explainability ─────────────────────────────────────────────────────
with tabs[2]:
    global_exp = result.get("global_explainability", {})
    local_exp  = result.get("explainability", {})

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Global Feature Importance</div>', unsafe_allow_html=True)
        summary = global_exp.get("summary", [])
        if summary:
            feat_names = [s["feature"] for s in summary]
            feat_vals  = [s["mean_abs_shap"] for s in summary]
            fig_shap = go.Figure(go.Bar(
                x=feat_vals, y=feat_names, orientation="h",
                marker=dict(color=feat_vals, colorscale=[[0,"#1e3a5f"],[1,"#38bdf8"]], showscale=False),
                hovertemplate="%{y}: %{x:.5f}<extra></extra>",
            ))
            fig_shap.update_layout(
                height=max(280, len(feat_names) * 36),
                xaxis_title="Mean |SHAP|",
                **_DARK,
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("Global SHAP unavailable for this model type.")

    with col2:
        st.markdown(f'<div class="section-title">Local Explanation — Sample {local_exp.get("sample_index","-")}</div>',
                    unsafe_allow_html=True)
        pred_val = local_exp.get("prediction", "-")
        base_val = local_exp.get("base_value", 0)
        st.markdown(f"Prediction: **`{pred_val}`** &nbsp;·&nbsp; Base value: `{base_val:.4f}`")
        attrs = local_exp.get("attributions", [])
        if attrs:
            attr_df    = pd.DataFrame(attrs)
            bar_colors = ["#f85149" if v > 0 else "#38bdf8" for v in attr_df["value"]]
            fig_local  = go.Figure(go.Bar(
                x=attr_df["value"], y=attr_df["feature"], orientation="h",
                marker_color=bar_colors, marker_line_width=0,
                hovertemplate="%{y}: %{x:+.5f}<extra></extra>",
            ))
            fig_local.add_vline(x=0, line_color="#30363d", line_width=1)
            fig_local.update_layout(
                height=max(260, len(attrs) * 34),
                xaxis_title="SHAP attribution",
                **_DARK,
            )
            st.plotly_chart(fig_local, use_container_width=True)

    dep_data = global_exp.get("dependency", [])[:3]
    if dep_data:
        st.markdown('<div class="section-title">Dependency Plots</div>', unsafe_allow_html=True)
        dcols = st.columns(len(dep_data))
        for i, dep in enumerate(dep_data):
            with dcols[i]:
                fig_dep = px.scatter(
                    x=dep["feature_values"], y=dep["shap_values"],
                    labels={"x": dep["feature"], "y": "SHAP"},
                    opacity=0.5, color_discrete_sequence=["#38bdf8"],
                )
                fig_dep.add_hline(y=0, line_dash="dot", line_color="#30363d")
                fig_dep.update_layout(
                    title=dict(text=dep["feature"], font=dict(size=12, color="#8b949e")),
                    height=260, **_DARK,
                )
                st.plotly_chart(fig_dep, use_container_width=True)

# ── Tab 4: Checklist ──────────────────────────────────────────────────────────
with tabs[3]:
    col_g, col_t = st.columns([1, 2])
    with col_g:
        st.markdown('<div class="section-title">Governance Score</div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=checklist["score"],
            number={"suffix": "%", "font": {"size": 34, "color": "#e6edf3"}},
            gauge={
                "axis": {"range": [0,100], "tickcolor": "#6e7681", "tickfont": {"color": "#6e7681"}},
                "bar":  {"color": "#38bdf8", "thickness": 0.22},
                "bgcolor": "#161b22",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  50], "color": "#1a0a0a"},
                    {"range": [50, 80], "color": "#1a1400"},
                    {"range": [80,100], "color": "#0d2818"},
                ],
                "threshold": {"line": {"color": "#f85149", "width": 2}, "thickness": 0.75, "value": 80},
            },
        ))
        fig_gauge.update_layout(
            height=230, margin=dict(l=20, r=20, t=20, b=0),
            paper_bgcolor="#161b22",
            font=dict(family="Inter, sans-serif", color="#c9d1d9"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        sc = checklist["score"]
        sc_color = "#3fb950" if sc >= 80 else ("#d29922" if sc >= 50 else "#f85149")
        st.markdown(f"<p style='text-align:center;font-weight:700;color:{sc_color};font-size:0.85rem'>"
                    f"{checklist['earned']} / {checklist['total']} principles met</p>", unsafe_allow_html=True)

    with col_t:
        st.markdown('<div class="section-title">Checklist</div>', unsafe_allow_html=True)
        cl_df = pd.DataFrame([{
            "Principle": i["principle"],
            "Response":  i["response"],
            "Status":    i["status"],
            "Evidence":  i.get("evidence","") or "",
        } for i in checklist["items"]])
        st.dataframe(cl_df, use_container_width=True, hide_index=True)

    recs = result.get("recommendations", [])
    if recs:
        st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)
        for r in recs:
            sev_css      = "rec-high" if r["severity"] == "high" else "rec-medium"
            icon         = "🔴" if r["severity"] == "high" else "🟡"
            actions_html = "".join(f'<div class="rec-action">· {a}</div>' for a in r["actions"])
            refs_html    = f'<div class="rec-ref">{" · ".join(r["references"])}</div>' if r.get("references") else ""
            st.markdown(f'<div class="{sev_css}"><div class="rec-title">{icon} {r["title"]}</div>'
                        f'{actions_html}{refs_html}</div>', unsafe_allow_html=True)

    rev = result.get("reviewer", {})
    if rev.get("name") or rev.get("comments"):
        st.markdown('<div class="section-title">Reviewer</div>', unsafe_allow_html=True)
        approved_color = "#3fb950" if rev.get("approved") else "#d29922"
        approved_text  = "Approved" if rev.get("approved") else "Pending"
        st.markdown(f"""
        <div class="reviewer-box">
          <div class="rev-label">Name</div><div class="rev-value">{rev.get('name') or '—'}</div>
          <div class="rev-label">Comments</div><div class="rev-value">{rev.get('comments') or '—'}</div>
          <div class="rev-label">Status</div>
          <div class="rev-value" style="color:{approved_color};font-weight:700">{approved_text}</div>
        </div>""", unsafe_allow_html=True)

# ── Tab 5: History ────────────────────────────────────────────────────────────
with tabs[4]:
    history = load_history()
    if history:
        st.markdown('<div class="section-title">Run History</div>', unsafe_allow_html=True)
        hist_df = pd.DataFrame([{
            "Run ID":          h["run_id"],
            "Timestamp":       h["timestamp"],
            "Target":          h["config"].get("target",""),
            "Sensitive":       h["config"].get("sensitive_attribute",""),
            "Rows":            h["config"].get("rows",""),
            "Accuracy":        round(h["summary"].get("accuracy") or 0, 3),
            "Disparate Impact":round(h["summary"].get("disparate_impact_ratio") or 0, 3),
            "Checklist %":     h["summary"].get("checklist_score"),
            "Flags":           h["summary"].get("risk_flag_count"),
        } for h in history])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        if len(hist_df) > 1:
            st.markdown('<div class="section-title">Disparate Impact Trend</div>', unsafe_allow_html=True)
            fig_trend = px.line(hist_df, x="Timestamp", y="Disparate Impact",
                                markers=True, color_discrete_sequence=["#38bdf8"])
            fig_trend.add_hline(y=0.8, line_dash="dot", line_color="#f85149",
                                annotation_text="0.80", annotation_font_size=10,
                                annotation_font_color="#f85149")
            fig_trend.update_layout(height=300, **_DARK)
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No history yet. Run an audit to start logging.")

# ── Tab 6: Export ─────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="section-title">Downloads</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("JSON Report", json.dumps(result, indent=2, default=str),
                           "audit_result.json", "application/json", use_container_width=True)
        pdf_path = Path("reports/ethicscan_audit_report.pdf")
        if pdf_path.exists():
            st.download_button("PDF Report", pdf_path.read_bytes(),
                               "ethicscan_audit_report.pdf", "application/pdf", use_container_width=True)
    with c2:
        html_path = Path("reports/ethicscan_audit_report.html")
        if html_path.exists():
            st.download_button("HTML Report", html_path.read_bytes(),
                               "ethicscan_audit_report.html", "text/html", use_container_width=True)
        log_path = Path("reports/audit_history.jsonl")
        if log_path.exists():
            st.download_button("Audit Log", log_path.read_bytes(),
                               "audit_history.jsonl", "application/json", use_container_width=True)
