import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from ethicscan.pipeline import run_audit
from ethicscan.checklist import DEFAULT_CHECKLIST, score_checklist, auto_check_checklist
from ethicscan.audit_log import load_history

st.set_page_config(page_title="EthicScan", page_icon="⚖️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; background-color: #0d1117 !important; color: #e6edf3 !important; }
.block-container { padding: 1.5rem 2.5rem; background: #0d1117; max-width: 1100px; }
.stApp { background: #0d1117; }
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #30363d !important; }
section[data-testid="stSidebar"] > div { background: #0d1117 !important; }
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
.stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox div[data-baseweb="select"] { background: #161b22 !important; border: 1px solid #30363d !important; color: #e6edf3 !important; border-radius: 6px !important; }
.stFileUploader { background: #161b22 !important; border: 1px dashed #30363d !important; border-radius: 8px !important; }
.stCheckbox label { color: #c9d1d9 !important; font-size: 0.83rem !important; }
.stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 10px; padding: 4px; gap: 3px; border: 1px solid #30363d; }
.stTabs [data-baseweb="tab"] { border-radius: 7px; padding: 0.35rem 0.9rem; font-size: 0.82rem; font-weight: 600; color: #8b949e; }
.stTabs [aria-selected="true"] { background: #0d1117 !important; color: #38bdf8 !important; }
.stDataFrame { border-radius: 10px; overflow: hidden; }
[data-testid="stDataFrame"] { background: #161b22 !important; }
[data-testid="stMetric"] { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 0.8rem 1rem; }
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; }
.stDownloadButton > button { background: #161b22 !important; border: 1px solid #30363d !important; color: #e6edf3 !important; border-radius: 8px !important; font-weight: 600 !important; }
.stDownloadButton > button:hover { border-color: #38bdf8 !important; color: #38bdf8 !important; }
.stSuccess { background: #0d2818 !important; border: 1px solid #3fb950 !important; color: #3fb950 !important; border-radius: 8px !important; }
.stInfo    { background: #0d1f2e !important; border: 1px solid #38bdf8 !important; color: #38bdf8 !important; border-radius: 8px !important; }
.stError   { background: #1a0a0a !important; border: 1px solid #f85149 !important; color: #f85149 !important; border-radius: 8px !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
.kpi-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin: 0 0 1.4rem 0; }
.kpi-card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 1.1rem 1.3rem; border-top: 3px solid #38bdf8; }
.kpi-card.warn   { border-top-color: #d29922; }
.kpi-card.danger { border-top-color: #f85149; }
.kpi-card.good   { border-top-color: #3fb950; }
.kpi-label { font-size: 0.7rem; font-weight: 600; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-value { font-size: 1.9rem; font-weight: 800; color: #e6edf3; line-height: 1.15; margin-top: 3px; }
.kpi-sub   { font-size: 0.7rem; color: #6e7681; margin-top: 2px; }
.section-title { font-size: 0.78rem; font-weight: 700; color: #8b949e; text-transform: uppercase; letter-spacing: 0.1em; border-bottom: 1px solid #30363d; padding-bottom: 5px; margin: 1.4rem 0 0.8rem 0; }
.flag-high   { background: #1a0a0a; border-left: 3px solid #f85149; border-radius: 8px; padding: 0.75rem 1rem; margin: 5px 0; display:flex; gap:0.7rem; align-items:flex-start; }
.flag-medium { background: #1a1400; border-left: 3px solid #d29922; border-radius: 8px; padding: 0.75rem 1rem; margin: 5px 0; display:flex; gap:0.7rem; align-items:flex-start; }
.flag-badge-high   { background:#f85149; color:#fff; font-size:0.6rem; font-weight:700; padding:2px 6px; border-radius:4px; text-transform:uppercase; white-space:nowrap; margin-top:2px; }
.flag-badge-medium { background:#d29922; color:#fff; font-size:0.6rem; font-weight:700; padding:2px 6px; border-radius:4px; text-transform:uppercase; white-space:nowrap; margin-top:2px; }
.flag-msg { font-size: 0.85rem; color: #c9d1d9; line-height: 1.5; }
.rec-high   { background:#1a0a0a; border:1px solid #3d1a1a; border-radius:10px; padding:1rem 1.2rem; margin:6px 0; }
.rec-medium { background:#1a1400; border:1px solid #3d3000; border-radius:10px; padding:1rem 1.2rem; margin:6px 0; }
.rec-title  { font-weight:700; font-size:0.88rem; color:#e6edf3; margin-bottom:0.4rem; }
.rec-action { font-size:0.81rem; color:#8b949e; margin:3px 0; }
.rec-ref    { font-size:0.72rem; color:#6e7681; margin-top:6px; }
.stepper { display:flex; align-items:center; gap:0; margin-bottom:1.8rem; }
.step-item { display:flex; align-items:center; gap:0.5rem; }
.step-dot { width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:0.8rem; flex-shrink:0; }
.step-dot.active { background:#38bdf8; color:#0d1117; }
.step-dot.done   { background:#3fb950; color:#0d1117; }
.step-dot.idle   { background:#21262d; color:#8b949e; border:1px solid #30363d; }
.step-label { font-size:0.8rem; font-weight:600; }
.step-label.active { color:#38bdf8; }
.step-label.done   { color:#3fb950; }
.step-label.idle   { color:#8b949e; }
.step-line { flex:1; height:1px; background:#30363d; margin:0 0.6rem; min-width:30px; }
.upload-zone { background:#161b22; border:1px dashed #30363d; border-radius:14px; padding:2rem; text-align:center; }
.upload-zone h3 { color:#e6edf3; font-size:1.1rem; margin:0 0 0.4rem; }
.upload-zone p  { color:#8b949e; font-size:0.85rem; margin:0; }
.cl-card { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:1rem 1.2rem; margin:8px 0; }
.cl-question { font-size:0.9rem; color:#e6edf3; font-weight:600; margin-bottom:0.5rem; }
.cl-principle { font-size:0.72rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.6rem; }
.verdict-high   { background:#1a0a0a; border:2px solid #f85149; border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:1rem; }
.verdict-medium { background:#1a1400; border:2px solid #d29922; border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:1rem; }
.verdict-good   { background:#0d2818; border:2px solid #3fb950; border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:1rem; }
.verdict-title  { font-size:1.2rem; font-weight:800; margin-bottom:0.3rem; }
.verdict-sub    { font-size:0.85rem; color:#8b949e; }
.issue-card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:0.8rem 1rem; margin:5px 0; display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; }
.issue-text { font-size:0.85rem; color:#c9d1d9; }
.issue-fix  { font-size:0.78rem; color:#8b949e; margin-top:3px; }
.plain-note { font-size:0.8rem; color:#8b949e; margin-top:4px; font-style:italic; }
</style>
""", unsafe_allow_html=True)

_DARK = dict(
    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
    font=dict(family="Inter, sans-serif", size=12, color="#c9d1d9"),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    margin=dict(l=0, r=0, t=30, b=0),
)

SAMPLE_MODEL = Path("sample_data/ethicscan_demo_model.joblib")
SAMPLE_CSV   = Path("sample_data/ethicscan_demo.csv")

# ── Session state defaults ────────────────────────────────────────────────────
for k, v in {"step": 1, "result": None, "responses": {}, "evidence": {}, "sample_row": 0}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar: past runs only ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚖️ EthicScan")
    st.markdown("---")
    if st.button("🏠 Start over", use_container_width=True):
        for k in ["step", "result", "responses", "evidence", "sample_row"]:
            st.session_state[k] = {"step": 1, "result": None, "responses": {}, "evidence": {}, "sample_row": 0}[k]
        st.rerun()
    st.markdown("### Past runs")
    history = load_history()
    if history:
        for h in history[-5:][::-1]:
            ts  = h["timestamp"][:16].replace("T", " ")
            acc = h["summary"].get("accuracy", 0)
            st.markdown(f"<div style='font-size:0.78rem;color:#8b949e;padding:4px 0;border-bottom:1px solid #21262d'>"
                        f"<b style='color:#c9d1d9'>{ts}</b><br>{h['config'].get('target','')} · "
                        f"acc {acc:.1%}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='font-size:0.8rem;color:#6e7681'>No runs yet.</span>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([5, 1])
with hcol1:
    st.markdown("<h1 style='font-size:1.6rem;font-weight:800;margin:0'>⚖️ EthicScan</h1>"
                "<p style='color:#8b949e;font-size:0.8rem;margin:2px 0 1rem'>Fairness · Explainability · Governance</p>",
                unsafe_allow_html=True)
with hcol2:
    result_hdr = st.session_state.get("result")
    if result_hdr:
        st.download_button("⬇ Report", json.dumps(result_hdr, indent=2, default=str),
                           "audit_result.json", "application/json", use_container_width=True)

# ── Stepper ───────────────────────────────────────────────────────────────────
step = st.session_state["step"]

def _dot(n):
    if n < step:  return "done"
    if n == step: return "active"
    return "idle"

st.markdown(f"""
<div class="stepper">
  <div class="step-item">
    <div class="step-dot {_dot(1)}">{"✓" if step > 1 else "1"}</div>
    <span class="step-label {_dot(1)}">Upload</span>
  </div>
  <div class="step-line"></div>
  <div class="step-item">
    <div class="step-dot {_dot(2)}">{"✓" if step > 2 else "2"}</div>
    <span class="step-label {_dot(2)}">Configure</span>
  </div>
  <div class="step-line"></div>
  <div class="step-item">
    <div class="step-dot {_dot(3)}">3</div>
    <span class="step-label {_dot(3)}">Results</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Step 1: Upload ────────────────────────────────────────────────────────────
if step == 1:
    st.markdown("### Step 1 — Upload your files")
    col_m, col_d = st.columns(2)
    with col_m:
        st.markdown("**Trained model** (.pkl / .joblib)")
        model_file = st.file_uploader("Model", type=["pkl", "joblib"], label_visibility="collapsed", key="uf_model")
    with col_d:
        st.markdown("**Evaluation dataset** (.csv)")
        data_file = st.file_uploader("Dataset", type=["csv"], label_visibility="collapsed", key="uf_data")

    st.markdown("---")
    sc1, sc2 = st.columns([1, 4])
    with sc1:
        if st.button("▶ Try sample data", use_container_width=True):
            st.session_state["use_sample"] = True
            st.session_state["step"] = 2
            st.rerun()

    if model_file and data_file:
        st.session_state["model_bytes"] = model_file.getvalue()
        st.session_state["csv_bytes"]   = data_file.getvalue()
        st.session_state["use_sample"]  = False
        nc1, nc2 = st.columns([4, 1])
        with nc2:
            if st.button("Next: Configure →", type="primary", use_container_width=True):
                st.session_state["step"] = 2
                st.rerun()
    st.stop()

# ── Step 2: Configure ─────────────────────────────────────────────────────────
if step == 2:
    use_sample = st.session_state.get("use_sample", False)
    if use_sample:
        df_preview = pd.read_csv(SAMPLE_CSV)
    else:
        df_preview = pd.read_csv(pd.io.common.BytesIO(st.session_state["csv_bytes"]))

    st.markdown("### Step 2 — Configure the audit")
    c1, c2 = st.columns(2)
    with c1:
        target = st.selectbox("Target column (what the model predicts)",
                              df_preview.columns,
                              index=list(df_preview.columns).index("approved") if "approved" in df_preview.columns else 0)
    with c2:
        sensitive = st.selectbox("Sensitive attribute (group to check for fairness)",
                                 [c for c in df_preview.columns if c != target],
                                 index=0)

    with st.expander("⚙ Advanced options"):
        thresh_di  = st.slider("Selection rate ratio threshold (flag below)", 0.5, 1.0, 0.80, 0.01,
                               help="Disparate impact ratio — flag if minority group's selection rate is below this fraction of the majority group's rate.")
        thresh_dp  = st.slider("Outcome gap threshold (flag above)", 0.0, 0.5, 0.10, 0.01,
                               help="Demographic parity difference — absolute difference in positive prediction rates between groups.")
        thresh_eod = st.slider("Qualified-people gap threshold (flag above)", 0.0, 0.5, 0.10, 0.01,
                               help="Equal opportunity difference — difference in true positive rates between groups.")

    bc1, bc2 = st.columns([1, 1])
    with bc1:
        if st.button("← Back"):
            st.session_state["step"] = 1
            st.rerun()
    with bc2:
        if st.button("▶ Check my model", type="primary", use_container_width=True):
            prog = st.progress(0, text="Checking data…")
            try:
                if use_sample:
                    model = joblib.load(SAMPLE_MODEL)
                    df    = pd.read_csv(SAMPLE_CSV)
                else:
                    tmp = Path(".streamlit_uploaded_model.joblib")
                    tmp.write_bytes(st.session_state["model_bytes"])
                    model = joblib.load(tmp)
                    df    = pd.read_csv(pd.io.common.BytesIO(st.session_state["csv_bytes"]))

                prog.progress(25, text="Running fairness checks…")
                prog.progress(50, text="Computing explanations…")
                result = run_audit(
                    model, df, target, sensitive,
                    st.session_state["responses"],
                    output_dir="reports",
                    sample_index=st.session_state["sample_row"],
                    evidence=st.session_state["evidence"],
                )
                prog.progress(90, text="Generating report…")
                st.session_state["result"]    = result
                st.session_state["target"]    = target
                st.session_state["sensitive"]  = sensitive
                # Seed checkboxes from auto-detected responses
                auto_resp, auto_ev = auto_check_checklist(result)
                st.session_state["responses"] = auto_resp
                st.session_state["evidence"]  = auto_ev
                prog.progress(100, text="Done!")
                st.session_state["step"] = 3
                st.rerun()
            except Exception as exc:
                st.error(f"Audit failed: {type(exc).__name__}: {exc}")
    st.stop()

# ── Step 3: Results ───────────────────────────────────────────────────────────
if step == 3:
    result   = st.session_state.get("result")
    if not result:
        st.warning("No results yet. Go back to Step 1.")
        st.stop()

    fairness  = result["fairness"]
    checklist = result["checklist"]
    flags     = result["risk_flags"]

    # Plain-English flag messages
    _PLAIN = {
        "disparate_impact_ratio":        lambda v: f"The minority group is selected only {v:.0%} as often as the majority group.",
        "demographic_parity_difference": lambda v: f"There is a {abs(v):.0%} outcome gap between groups.",
        "equal_opportunity_difference":  lambda v: f"Among qualified people, one group is approved {abs(v):.0%} less often.",
        "equalized_odds_gap":            lambda v: f"Equalized odds gap of {abs(v):.3f} detected.",
    }
    _FIXES = {
        "disparate_impact_ratio":        "Re-sample training data or apply post-processing to equalise selection rates.",
        "demographic_parity_difference": "Review feature selection — remove or transform features correlated with the sensitive attribute.",
        "equal_opportunity_difference":  "Adjust decision threshold per group or retrain with fairness constraints.",
        "equalized_odds_gap":            "Apply equalized odds post-processing or use an in-processing fairness algorithm.",
    }

    tabs = st.tabs(["📋 Summary", "⚖️ Fairness", "🔍 Explain", "📝 Governance"])

    # ── Summary tab ──────────────────────────────────────────────────────────
    with tabs[0]:
        flag_count = len(flags)
        di_val     = fairness.get("disparate_impact_ratio", 1.0)
        cl_score   = checklist["score"]

        if flag_count >= 2 or di_val < 0.8:
            verdict_cls, verdict_icon, verdict_text = "verdict-high",   "🔴", f"High risk — {flag_count} issue{'s' if flag_count != 1 else ''} found"
        elif flag_count == 1:
            verdict_cls, verdict_icon, verdict_text = "verdict-medium", "🟡", "Medium risk — 1 issue found"
        else:
            verdict_cls, verdict_icon, verdict_text = "verdict-good",   "🟢", "Low risk — no threshold breaches"

        st.markdown(f"""
        <div class="{verdict_cls}">
          <div class="verdict-title">{verdict_icon} {verdict_text}</div>
          <div class="verdict-sub">Accuracy {result['accuracy']:.1%} · Governance {cl_score:.0f}% · {flag_count} flag{'s' if flag_count != 1 else ''}</div>
        </div>
        """, unsafe_allow_html=True)

        di_class = "danger" if di_val < 0.8 else ("warn" if di_val < 0.9 else "good")
        cl_class = "good" if cl_score >= 80 else ("warn" if cl_score >= 50 else "danger")
        st.markdown(f"""
        <div class="kpi-grid">
          <div class="kpi-card good">
            <div class="kpi-label">Accuracy</div>
            <div class="kpi-value">{result['accuracy']:.1%}</div>
            <div class="kpi-sub">Overall model accuracy</div>
          </div>
          <div class="kpi-card {di_class}">
            <div class="kpi-label">Fairness status</div>
            <div class="kpi-value">{"Pass" if di_val >= 0.8 else "Gap"}</div>
            <div class="kpi-sub">Selection rate ratio {di_val:.3f}</div>
          </div>
          <div class="kpi-card {cl_class}">
            <div class="kpi-label">Governance score</div>
            <div class="kpi-value">{cl_score:.0f}%</div>
            <div class="kpi-sub">{checklist['earned']} / {checklist['total']} principles</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if flags:
            st.markdown('<div class="section-title">Top issues</div>', unsafe_allow_html=True)
            for flag in flags[:3]:
                sev  = flag["severity"]
                key  = flag.get("metric", "")
                plain = _PLAIN.get(key, lambda v: flag["message"])(fairness.get(key, 0))
                fix   = _FIXES.get(key, "Review the Fairness tab for details.")
                st.markdown(
                    f'<div class="issue-card">'
                    f'<div><span class="flag-badge-{sev}">{sev}</span>'
                    f'<div class="issue-text" style="margin-top:4px">{plain}</div>'
                    f'<div class="issue-fix">💡 {fix}</div></div>'
                    f'</div>',
                    unsafe_allow_html=True)
        else:
            st.success("✅ No threshold breaches detected.")

        if st.button("← Back to configure"):
            st.session_state["step"] = 2
            st.rerun()

    # ── Fairness tab ──────────────────────────────────────────────────────────
    with tabs[1]:
        groups    = fairness.get("groups", [])
        pos_rates = [fairness["positive_rates"].get(g, 0) for g in groups]

        st.markdown('<div class="section-title">Selection rate by group</div>', unsafe_allow_html=True)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Selection rate", x=groups, y=pos_rates,
                                 marker_color="#38bdf8", marker_line_width=0))
        fig_bar.add_hline(y=0.8, line_dash="dot", line_color="#f85149",
                          annotation_text="0.80 threshold", annotation_font_size=10,
                          annotation_font_color="#f85149")
        fig_bar.update_layout(yaxis_range=[0, 1], height=300,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                         bgcolor="rgba(0,0,0,0)", font=dict(size=11)), **_DARK)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('<div class="section-title">Fairness metrics</div>', unsafe_allow_html=True)
        _METRIC_ROWS = [
            ("Selection rate ratio",   "disparate_impact_ratio",        0.8,  True,
             lambda v: f"The minority group is selected {v:.0%} as often as the majority group. (Threshold: ≥ 0.80)"),
            ("Outcome gap",            "demographic_parity_difference",  0.1,  False,
             lambda v: f"There is a {abs(v):.0%} difference in approval rates between groups. (Threshold: ≤ 0.10)"),
            ("Qualified-people gap",   "equal_opportunity_difference",   0.1,  False,
             lambda v: f"Among qualified people, approval rates differ by {abs(v):.0%}. (Threshold: ≤ 0.10)"),
            ("Equalized odds gap",     "equalized_odds_gap",             0.1,  False,
             lambda v: f"Combined TPR/FPR gap is {abs(v):.3f}. (Threshold: ≤ 0.10)"),
        ]
        mc = st.columns(2)
        for i, (label, key, thresh, invert, note_fn) in enumerate(_METRIC_ROWS):
            v = fairness.get(key)
            if v is None:
                continue
            breach = (v < thresh) if invert else (abs(v) > thresh)
            color  = "#f85149" if breach else "#3fb950"
            status = "Gap" if breach else "Pass"
            with mc[i % 2]:
                st.metric(label, f"{v:.4f}", delta=status,
                          delta_color="inverse" if breach else "normal",
                          help=f"Technical name: {key}")
                st.markdown(f"<div class='plain-note'>{note_fn(v)}</div>", unsafe_allow_html=True)

        sp = fairness.get("subgroup_performance", {})
        if sp:
            with st.expander("Detailed metrics"):
                sp_df = pd.DataFrame(sp).T.reset_index().rename(columns={"index": "Group"}).round(3)
                st.dataframe(sp_df, use_container_width=True, hide_index=True)

                radar_cats = ["accuracy", "precision", "recall", "f1", "positive_rate"]
                fig_radar  = go.Figure()
                palette    = ["#38bdf8", "#3fb950", "#d29922", "#a855f7"]
                for i, (g, v) in enumerate(sp.items()):
                    vals = [v.get(c, 0) for c in radar_cats] + [v.get(radar_cats[0], 0)]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals, theta=radar_cats + [radar_cats[0]],
                        fill="toself", name=f"Group {g}",
                        line_color=palette[i % len(palette)], opacity=0.75,
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="#30363d", color="#8b949e"),
                               angularaxis=dict(color="#8b949e"), bgcolor="#161b22"),
                    height=340, paper_bgcolor="#161b22",
                    legend=dict(orientation="h", y=-0.15, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
                    font=dict(family="Inter, sans-serif", size=12, color="#c9d1d9"),
                    margin=dict(l=40, r=40, t=20, b=40),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

    # ── Explain tab ───────────────────────────────────────────────────────────
    with tabs[2]:
        global_exp = result.get("global_explainability", {})
        local_exp  = result.get("explainability", {})

        summary = global_exp.get("summary", [])
        st.markdown('<div class="section-title">Top factors the model uses</div>', unsafe_allow_html=True)
        if summary:
            top3 = ", ".join(s["feature"] for s in summary[:3])
            st.markdown(f"<div style='font-size:0.85rem;color:#8b949e;margin-bottom:0.8rem'>Most influential: <b style='color:#e6edf3'>{top3}</b></div>",
                        unsafe_allow_html=True)
            feat_names = [s["feature"] for s in summary]
            feat_vals  = [s["mean_abs_shap"] for s in summary]
            fig_shap = go.Figure(go.Bar(
                x=feat_vals, y=feat_names, orientation="h",
                marker=dict(color=feat_vals, colorscale=[[0, "#1e3a5f"], [1, "#38bdf8"]], showscale=False),
                hovertemplate="%{y}: %{x:.5f}<extra></extra>",
            ))
            fig_shap.update_layout(height=max(260, len(feat_names) * 34), xaxis_title="Mean |SHAP|", **_DARK)
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("Global SHAP unavailable for this model type.")

        st.markdown('<div class="section-title">Explain one prediction</div>', unsafe_allow_html=True)
        use_sample2 = st.session_state.get("use_sample", False)
        if use_sample2:
            df_exp = pd.read_csv(SAMPLE_CSV)
        else:
            df_exp = pd.read_csv(pd.io.common.BytesIO(st.session_state.get("csv_bytes", b"")))

        row_idx = st.number_input("Pick a row to explain", min_value=0,
                                  max_value=max(0, len(df_exp) - 1),
                                  value=st.session_state["sample_row"],
                                  key="row_picker")
        if row_idx != st.session_state["sample_row"]:
            st.session_state["sample_row"] = int(row_idx)

        attrs = local_exp.get("attributions", [])
        pred_val = local_exp.get("prediction", "-")
        base_val = local_exp.get("base_value", 0)
        st.markdown(f"Prediction: **`{pred_val}`** &nbsp;·&nbsp; Base value: `{base_val:.4f}`")
        if attrs:
            bar_colors = ["#f85149" if v > 0 else "#38bdf8" for v in [a["value"] for a in attrs]]
            attr_df    = pd.DataFrame(attrs)
            fig_local  = go.Figure(go.Bar(
                x=attr_df["value"], y=attr_df["feature"], orientation="h",
                marker_color=bar_colors, marker_line_width=0,
                hovertemplate="%{y}: %{x:+.5f}<extra></extra>",
            ))
            fig_local.add_vline(x=0, line_color="#30363d", line_width=1)
            fig_local.update_layout(height=max(240, len(attrs) * 32), xaxis_title="SHAP attribution", **_DARK)
            st.plotly_chart(fig_local, use_container_width=True)

        dep_data = global_exp.get("dependency", [])[:3]
        if dep_data:
            with st.expander("Dependency plots (advanced)"):
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
                            height=240, **_DARK,
                        )
                        st.plotly_chart(fig_dep, use_container_width=True)

    # ── Governance tab ────────────────────────────────────────────────────────
    with tabs[3]:
        # Live checklist — checkboxes update session state and re-score immediately
        st.markdown('<div class="section-title">Responsible AI checklist</div>', unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.82rem;color:#8b949e;margin-bottom:1rem'>Automatically checked from audit data. You can override any item.</p>",
                    unsafe_allow_html=True)
        for item in DEFAULT_CHECKLIST:
            checked = st.session_state["responses"].get(item["id"], False)
            reason  = st.session_state["evidence"].get(item["id"], "")
            auto_icon = "🤖" if reason else ""
            st.markdown(f'<div class="cl-card"><div class="cl-principle">{item["principle"]}</div>'
                        f'<div class="cl-question">{item["question"]}</div>'
                        + (f'<div style="font-size:0.75rem;color:#8b949e;margin-top:4px">{auto_icon} {reason}</div>' if reason else "")
                        + '</div>',
                        unsafe_allow_html=True)
            new_val = st.checkbox("Met", value=checked, key=f"cl_{item['id']}")
            if new_val != checked:
                st.session_state["responses"][item["id"]] = new_val
                # re-score and patch result in session state
                updated_cl = score_checklist(st.session_state["responses"], st.session_state["evidence"])
                st.session_state["result"]["checklist"] = updated_cl
                result["checklist"] = updated_cl
                checklist = updated_cl
                st.rerun()

        sc = checklist["score"]
        sc_color = "#3fb950" if sc >= 80 else ("#d29922" if sc >= 50 else "#f85149")
        st.markdown(f"<p style='font-size:1.1rem;font-weight:700;color:{sc_color};margin-top:1rem'>"
                    f"Governance score: {sc:.0f}% &nbsp;({checklist['earned']} / {checklist['total']} principles met)</p>",
                    unsafe_allow_html=True)

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

        st.markdown('<div class="section-title">Run history</div>', unsafe_allow_html=True)
        history = load_history()
        if history:
            hist_df = pd.DataFrame([{
                "Timestamp":       h["timestamp"][:16].replace("T", " "),
                "Target":          h["config"].get("target", ""),
                "Sensitive":       h["config"].get("sensitive_attribute", ""),
                "Accuracy":        round(h["summary"].get("accuracy") or 0, 3),
                "Selection ratio": round(h["summary"].get("disparate_impact_ratio") or 0, 3),
                "Governance %":    h["summary"].get("checklist_score"),
                "Flags":           h["summary"].get("risk_flag_count"),
            } for h in history])
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
        else:
            st.info("No history yet.")

        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            st.download_button("⬇ JSON report", json.dumps(result, indent=2, default=str),
                               "audit_result.json", "application/json", use_container_width=True)
        with c_dl2:
            html_path = Path("reports/ethicscan_audit_report.html")
            if html_path.exists():
                st.download_button("⬇ HTML report", html_path.read_bytes(),
                                   "ethicscan_audit_report.html", "text/html", use_container_width=True)
