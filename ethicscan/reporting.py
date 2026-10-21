from datetime import datetime
from html import escape
from pathlib import Path
import json

def generate_html(result: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fairness = result.get("fairness", {})
    checklist = result.get("checklist", {})
    flags = result.get("risk_flags", [])
    attr_rows = "".join(f"<tr><td>{escape(str(a['feature']))}</td><td>{a['value']:+.5f}</td></tr>" for a in result.get("explainability", {}).get("attributions", []))
    flag_rows = "".join(f"<li class='{escape(f['severity'])}'><b>{escape(f['severity'].upper())}</b>: {escape(f['message'])}</li>" for f in flags) or "<li class='good'>No threshold breach detected in Phase 1 checks.</li>"
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>EthicScan Audit Report</title><style>body{{font-family:Arial,sans-serif;max-width:980px;margin:40px auto;color:#17212b}}h1{{color:#0b5563}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card{{border:1px solid #d9e2e8;border-radius:10px;padding:16px;background:#f7fafb}}table{{width:100%;border-collapse:collapse}}td,th{{border-bottom:1px solid #e5e7eb;padding:8px;text-align:left}}li{{margin:8px 0}}.high{{color:#a61b1b}}.medium{{color:#9a5b00}}.good{{color:#147a4b}}.muted{{color:#63727d}}</style></head><body><h1>EthicScan Phase 1 Audit Report</h1><p class="muted">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · This is an audit aid, not a legal compliance decision.</p><div class="grid"><div class="card"><b>Rows audited</b><h2>{result.get('validation',{}).get('rows','-')}</h2></div><div class="card"><b>Disparate impact</b><h2>{fairness.get('disparate_impact_ratio','-')}</h2></div><div class="card"><b>Checklist score</b><h2>{checklist.get('score','-')}%</h2></div></div><h2>Risk flags</h2><ul>{flag_rows}</ul><h2>Fairness metrics</h2><pre>{escape(json.dumps(fairness, indent=2, default=str))}</pre><h2>Local SHAP explanation</h2><p>Method: {escape(str(result.get('explainability',{}).get('method','-')))}</p><table><tr><th>Feature</th><th>Attribution</th></tr>{attr_rows}</table><h2>Responsible AI checklist</h2><p>Score: {checklist.get('score','-')}% ({checklist.get('earned','-')}/{checklist.get('total','-')})</p><table><tr><th>Principle</th><th>Response</th><th>Status</th></tr>{''.join(f"<tr><td>{escape(i['principle'])}</td><td>{i['response']}</td><td>{i['status']}</td></tr>" for i in checklist.get('items',[]))}</table><h2>Limitations</h2><p>Metrics depend on the selected evaluation data, sensitive attributes, labels, thresholds, and model context. Review results with domain experts before deployment.</p></body></html>'''
    path.write_text(html, encoding="utf-8")
    return path
