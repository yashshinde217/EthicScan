from datetime import datetime
from html import escape
from pathlib import Path
import json
import base64
import io

def _b64_plot(fig) -> str:
    """Encode a matplotlib figure as a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _fairness_bar_b64(fairness: dict) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups = list(fairness.get("positive_rates", {}).keys())
        rates = [fairness["positive_rates"][g] for g in groups]
        tprs = [fairness.get("true_positive_rates", {}).get(g) or 0 for g in groups]
        x = range(len(groups))
        fig, ax = plt.subplots(figsize=(6, 3))
        w = 0.35
        ax.bar([i - w/2 for i in x], rates, w, label="Positive Rate", color="#0b5563")
        ax.bar([i + w/2 for i in x], tprs, w, label="True Positive Rate", color="#f59e0b")
        ax.set_xticks(list(x)); ax.set_xticklabels(groups)
        ax.set_ylim(0, 1); ax.set_ylabel("Rate"); ax.set_title("Group Rates by Sensitive Attribute")
        ax.legend(); ax.axhline(0.8, color="red", linestyle="--", linewidth=0.8, label="DI threshold")
        fig.tight_layout()
        enc = _b64_plot(fig)
        plt.close(fig)
        return enc
    except Exception:
        return ""

def _shap_bar_b64(global_exp: dict) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        summary = global_exp.get("summary", [])
        if not summary:
            return ""
        features = [s["feature"] for s in reversed(summary)]
        values = [s["mean_abs_shap"] for s in reversed(summary)]
        fig, ax = plt.subplots(figsize=(6, max(3, len(features) * 0.4)))
        ax.barh(features, values, color="#0b5563")
        ax.set_xlabel("Mean |SHAP value|"); ax.set_title("Global Feature Importance (SHAP)")
        fig.tight_layout()
        enc = _b64_plot(fig)
        plt.close(fig)
        return enc
    except Exception:
        return ""

def _dependency_b64(dep: dict) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        feat = dep["feature"]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(dep["feature_values"], dep["shap_values"], alpha=0.4, s=12, color="#0b5563")
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.set_xlabel(feat); ax.set_ylabel("SHAP value"); ax.set_title(f"Dependency: {feat}")
        fig.tight_layout()
        enc = _b64_plot(fig)
        plt.close(fig)
        return enc
    except Exception:
        return ""

_CSS = """
body{font-family:Arial,sans-serif;max-width:1100px;margin:40px auto;color:#17212b;line-height:1.5}
h1{color:#0b5563;border-bottom:2px solid #0b5563;padding-bottom:6px}
h2{color:#0b5563;margin-top:32px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:16px 0}
.card{border:1px solid #d9e2e8;border-radius:10px;padding:16px;background:#f7fafb;text-align:center}
.card h2{margin:4px 0;font-size:1.6rem;color:#0b5563}
table{width:100%;border-collapse:collapse;margin:12px 0}
td,th{border-bottom:1px solid #e5e7eb;padding:8px;text-align:left}
th{background:#f0f4f6}
li{margin:6px 0}
.high{color:#a61b1b}.medium{color:#9a5b00}.low{color:#147a4b}.good{color:#147a4b}
.muted{color:#63727d;font-size:.9rem}
.rec{background:#fffbeb;border-left:4px solid #f59e0b;padding:10px 14px;margin:10px 0;border-radius:4px}
.rec.high{border-color:#a61b1b;background:#fff5f5}
.reviewer{background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;padding:16px;margin:16px 0}
img{max-width:100%;border-radius:6px;margin:8px 0}
.plots{display:grid;grid-template-columns:1fr 1fr;gap:16px}
pre{background:#f4f4f4;padding:12px;border-radius:6px;overflow-x:auto;font-size:.82rem}
"""

def generate_html(result: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fairness = result.get("fairness", {})
    checklist = result.get("checklist", {})
    flags = result.get("risk_flags", [])
    recs = result.get("recommendations", [])
    reviewer = result.get("reviewer", {})
    global_exp = result.get("global_explainability", {})
    local_exp = result.get("explainability", {})

    # Plots
    fairness_img = _fairness_bar_b64(fairness)
    shap_img = _shap_bar_b64(global_exp)
    dep_imgs = [_dependency_b64(d) for d in global_exp.get("dependency", [])[:3]]

    def img_tag(b64): return f'<img src="data:image/png;base64,{b64}" alt="chart"/>' if b64 else ""

    # Executive summary
    di = fairness.get("disparate_impact_ratio", "-")
    dpd = fairness.get("demographic_parity_difference", "-")
    eo = fairness.get("equal_opportunity_difference", "-")
    eog = fairness.get("equalized_odds_gap", "-")
    acc = result.get("accuracy", "-")
    cs = checklist.get("score", "-")
    flag_count = len(flags)
    exec_status = "⚠️ Issues detected — human review required." if flags else "✅ No threshold breaches detected."

    flag_rows = "".join(f"<li class='{escape(f['severity'])}'><b>{escape(f['severity'].upper())}</b>: {escape(f['message'])}</li>" for f in flags) or "<li class='good'>No threshold breach detected.</li>"
    rec_blocks = "".join(f"<div class='rec {escape(r['severity'])}'><b>{escape(r['title'])}</b><ul>{''.join(f'<li>{escape(a)}</li>' for a in r['actions'])}</ul></div>" for r in recs) or "<p class='good'>No recommendations triggered.</p>"
    attr_rows = "".join(f"<tr><td>{escape(str(a['feature']))}</td><td>{a['value']:+.5f}</td></tr>" for a in local_exp.get("attributions", []))
    checklist_rows = "".join(f"<tr><td>{escape(i['principle'])}</td><td>{escape(i['question'])}</td><td>{i['response']}</td><td class='{'good' if i['status']=='pass' else 'high'}'>{i['status']}</td><td>{escape(i.get('evidence','') or '')}</td></tr>" for i in checklist.get("items", []))
    subgroup_rows = "".join(f"<tr><td>{escape(g)}</td><td>{v['count']}</td><td>{v['accuracy']:.3f}</td><td>{v['precision']:.3f}</td><td>{v['recall']:.3f}</td><td>{v['f1']:.3f}</td><td>{v['positive_rate']:.3f}</td></tr>" for g, v in fairness.get("subgroup_performance", {}).items())
    dep_plot_html = "<div class='plots'>" + "".join(img_tag(i) for i in dep_imgs if i) + "</div>" if any(dep_imgs) else ""

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>EthicScan Phase 2 Audit Report</title><style>{_CSS}</style></head><body>
<h1>⚖️ EthicScan Phase 2 Audit Report</h1>
<p class="muted">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Audit aid only — not a legal compliance decision.</p>

<h2>Executive Summary</h2>
<p><b>Overall status:</b> {exec_status} &nbsp;|&nbsp; <b>Risk flags:</b> {flag_count} &nbsp;|&nbsp; <b>Checklist:</b> {cs}% ({checklist.get('status','')})</p>
<div class="grid">
  <div class="card"><b>Accuracy</b><h2>{f'{acc:.1%}' if isinstance(acc, float) else acc}</h2></div>
  <div class="card"><b>Disparate Impact</b><h2>{f'{di:.3f}' if isinstance(di, float) else di}</h2></div>
  <div class="card"><b>Checklist Score</b><h2>{cs}%</h2></div>
  <div class="card"><b>Risk Flags</b><h2>{flag_count}</h2></div>
</div>

<h2>Risk Flags</h2><ul>{flag_rows}</ul>

<h2>Actionable Recommendations</h2>{rec_blocks}

<h2>Fairness Metrics</h2>
<table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>Disparate Impact Ratio</td><td>{f'{di:.4f}' if isinstance(di, float) else di}</td></tr>
<tr><td>Demographic Parity Difference</td><td>{f'{dpd:.4f}' if isinstance(dpd, float) else dpd}</td></tr>
<tr><td>Equal Opportunity Difference</td><td>{f'{eo:.4f}' if isinstance(eo, float) else eo}</td></tr>
<tr><td>Equalized Odds Gap</td><td>{f'{eog:.4f}' if isinstance(eog, float) else eog}</td></tr>
</table>
{img_tag(fairness_img)}

<h2>Subgroup Performance Breakdown</h2>
<table><tr><th>Group</th><th>Count</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Positive Rate</th></tr>{subgroup_rows}</table>

<h2>Global SHAP Feature Importance</h2>
{img_tag(shap_img)}
<h2>SHAP Dependency Plots (Top 3 Features)</h2>
{dep_plot_html}

<h2>Local SHAP Explanation (Sample {local_exp.get('sample_index','-')})</h2>
<p>Prediction: <b>{escape(str(local_exp.get('prediction','-')))}</b> &nbsp;|&nbsp; Base value: {local_exp.get('base_value',0):.4f} &nbsp;|&nbsp; Method: {escape(str(local_exp.get('method','-')))}</p>
<table><tr><th>Feature</th><th>Attribution</th></tr>{attr_rows}</table>

<h2>Responsible AI Checklist (NITI Aayog — 7 Principles)</h2>
<p>Score: <b>{cs}%</b> ({checklist.get('earned','-')}/{checklist.get('total','-')})</p>
<table><tr><th>Principle</th><th>Question</th><th>Response</th><th>Status</th><th>Evidence</th></tr>{checklist_rows}</table>

<h2>Human-in-the-Loop Reviewer Sign-off</h2>
<div class="reviewer">
  <p><b>Reviewer:</b> {escape(reviewer.get('name','') or 'Not assigned')}</p>
  <p><b>Comments:</b> {escape(reviewer.get('comments','') or 'None provided')}</p>
  <p><b>Approved for deployment:</b> {'✅ Yes' if reviewer.get('approved') else '❌ Pending approval'}</p>
</div>

<h2>Limitations</h2>
<p>Metrics depend on the selected evaluation data, sensitive attributes, labels, thresholds, and model context. Thresholds are screening rules, not universal legal standards. Human and domain-expert review is required before deployment. This report does not certify legal or regulatory compliance.</p>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    return path


def generate_pdf(result: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.units import cm

        doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=colors.HexColor("#0b5563"), fontSize=16)
        h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=colors.HexColor("#0b5563"), fontSize=12)
        normal = styles["Normal"]
        small = ParagraphStyle("small", parent=normal, fontSize=8, textColor=colors.HexColor("#63727d"))
        bold = ParagraphStyle("bold", parent=normal, fontName="Helvetica-Bold")

        fairness = result.get("fairness", {})
        checklist = result.get("checklist", {})
        flags = result.get("risk_flags", [])
        recs = result.get("recommendations", [])
        reviewer = result.get("reviewer", {})
        local_exp = result.get("explainability", {})
        global_exp = result.get("global_explainability", {})

        story = [
            Paragraph("EthicScan Phase 2 Audit Report", h1),
            Paragraph(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Audit aid only — not a legal compliance decision.", small),
            Spacer(1, 0.3*cm),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0b5563")),
            Spacer(1, 0.3*cm),
            Paragraph("Executive Summary", h2),
            Paragraph(f"Accuracy: {result.get('accuracy', '-'):.1%} | Disparate Impact: {fairness.get('disparate_impact_ratio', '-')} | Checklist: {checklist.get('score', '-')}% | Risk Flags: {len(flags)}", normal),
            Spacer(1, 0.3*cm),
            Paragraph("Risk Flags", h2),
        ]
        if flags:
            for f in flags:
                story.append(Paragraph(f"[{f['severity'].upper()}] {f['message']}", normal))
        else:
            story.append(Paragraph("No threshold breaches detected.", normal))

        story += [Spacer(1, 0.3*cm), Paragraph("Actionable Recommendations", h2)]
        for r in recs:
            story.append(Paragraph(f"<b>{r['title']}</b>", normal))
            for a in r["actions"]:
                story.append(Paragraph(f"• {a}", normal))
            story.append(Spacer(1, 0.2*cm))

        story += [Spacer(1, 0.3*cm), Paragraph("Fairness Metrics", h2)]
        fm_data = [["Metric", "Value"],
                   ["Disparate Impact Ratio", str(round(fairness.get("disparate_impact_ratio", 0), 4))],
                   ["Demographic Parity Difference", str(round(fairness.get("demographic_parity_difference", 0), 4))],
                   ["Equal Opportunity Difference", str(round(fairness.get("equal_opportunity_difference") or 0, 4))],
                   ["Equalized Odds Gap", str(round(fairness.get("equalized_odds_gap") or 0, 4))]]
        fm_table = Table(fm_data, colWidths=[10*cm, 5*cm])
        fm_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0b5563")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#d9e2e8")), ("FONTSIZE", (0,0), (-1,-1), 9)]))
        story.append(fm_table)

        story += [Spacer(1, 0.3*cm), Paragraph("Subgroup Performance", h2)]
        sp = fairness.get("subgroup_performance", {})
        if sp:
            sp_data = [["Group", "Count", "Accuracy", "Precision", "Recall", "F1", "Pos. Rate"]]
            for g, v in sp.items():
                sp_data.append([g, str(v["count"]), f"{v['accuracy']:.3f}", f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['f1']:.3f}", f"{v['positive_rate']:.3f}"])
            sp_table = Table(sp_data)
            sp_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0b5563")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#d9e2e8")), ("FONTSIZE", (0,0), (-1,-1), 8)]))
            story.append(sp_table)

        story += [Spacer(1, 0.3*cm), Paragraph("Global SHAP Feature Importance", h2)]
        for s in global_exp.get("summary", []):
            story.append(Paragraph(f"• {s['feature']}: {s['mean_abs_shap']:.5f}", normal))

        story += [Spacer(1, 0.3*cm), Paragraph(f"Local SHAP (Sample {local_exp.get('sample_index', '-')})", h2),
                  Paragraph(f"Prediction: {local_exp.get('prediction', '-')} | Base value: {local_exp.get('base_value', 0):.4f}", normal)]
        for a in local_exp.get("attributions", []):
            story.append(Paragraph(f"• {a['feature']}: {a['value']:+.5f}", normal))

        story += [Spacer(1, 0.3*cm), Paragraph("Responsible AI Checklist (NITI Aayog — 7 Principles)", h2)]
        cl_data = [["Principle", "Response", "Status", "Evidence"]]
        for i in checklist.get("items", []):
            cl_data.append([i["principle"], i["response"], i["status"], i.get("evidence", "") or ""])
        cl_table = Table(cl_data, colWidths=[6*cm, 2.5*cm, 2.5*cm, 5*cm])
        cl_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0b5563")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#d9e2e8")), ("FONTSIZE", (0,0), (-1,-1), 8), ("WORDWRAP", (0,0), (-1,-1), True)]))
        story.append(cl_table)

        story += [Spacer(1, 0.3*cm), Paragraph("Human-in-the-Loop Reviewer Sign-off", h2),
                  Paragraph(f"Reviewer: {reviewer.get('name','Not assigned') or 'Not assigned'}", normal),
                  Paragraph(f"Comments: {reviewer.get('comments','None provided') or 'None provided'}", normal),
                  Paragraph(f"Approved: {'Yes' if reviewer.get('approved') else 'Pending'}", normal),
                  Spacer(1, 0.3*cm),
                  Paragraph("Limitations", h2),
                  Paragraph("Metrics depend on evaluation data, sensitive attributes, labels, thresholds, and model context. Thresholds are screening rules, not legal standards. Human review is required before deployment.", small)]

        doc.build(story)
    except Exception as exc:
        path.write_text(f"PDF generation failed: {exc}", encoding="utf-8")
    return path
