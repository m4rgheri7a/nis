#!/usr/bin/env python3
"""FIMI-Cyber PoC — Web Dashboard.

Usage
-----
    python scripts/serve.py              # http://localhost:5000
    python scripts/serve.py --port 8080
"""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent          # fimi-cyber-poc/
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
EVIDENCE_DIR = RESULTS / "evidence_paths"

sys.path.insert(0, str(ROOT / "src"))

try:
    from flask import (Flask, Response, render_template_string,
                       send_from_directory, stream_with_context)
except ImportError:
    sys.exit("Flask 없음. 설치: pip install flask --break-system-packages")

app = Flask(__name__, static_folder=None)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# ── Pipeline steps ────────────────────────────────────────────────────────────
STEPS = [
    ("M1-load",       "DISINFOX + EUvsDisinfo 통합 로드  →  events.jsonl"),
    ("M3-ioc",        "IOC 추출·분류·합성  →  iocs.jsonl"),
    ("M2-embed",      "텍스트 임베딩 계산"),
    ("M2-narrative",  "내러티브 유사도 행렬  N(i,j)"),
    ("M4-graph",      "증거 그래프 구축  →  graph.json"),
    ("M4-ioc-score",  "IOC 연계 점수  I(i,j)"),
    ("M5-components", "D / C / T / A 성분 계산"),
    ("M5-fcls",       "FCLS 통합 점수  →  pairwise_scores.csv"),
    ("M5-priority",   "Priority(i)  →  priority_table.csv"),
    ("M6-eval",       "통합 E1/E2/E3 평가  →  metrics_summary.csv"),
    ("M7-ablation",   "Ablation study  →  ablation.csv"),
    ("M7-grid",       "Grid search  →  gridsearch.csv"),
    ("M7-robust",     "강건성 실험  →  robustness.csv"),
    ("M8-viz",        "증거 경로 HTML 생성"),
    ("M8-charts",     "차트 생성  →  figures/"),
    ("M8-report",     "결과 보고서  →  report.md"),
]

CHART_INFO = [
    ("metrics_bar.png",        "E1 / E2 / E3 성능 비교"),
    ("ablation_bar.png",       "Ablation: 성분별 MAP 영향"),
    ("score_distribution.png", "FCLS 점수 분포"),
    ("component_radar.png",    "상위 쌍 성분 레이더 차트"),
    ("grid_heatmap.png",       "Grid Search: α × β MAP"),
    ("robustness_lines.png",   "강건성: 노이즈 비율 vs MAP"),
    ("pairwise_heatmap.png",   "Pairwise FCLS_E3 히트맵"),
    ("event_cluster.png",      "이벤트 클러스터 (스프링 레이아웃)"),
]

# ── SSE run helper ────────────────────────────────────────────────────────────
_running: dict[str, bool] = {}


def _sse_stream(cmd: list[str]):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(ROOT), env=env, bufsize=1,
    )
    try:
        for line in proc.stdout:
            yield f"data: {json.dumps({'line': line.rstrip()})}\n\n"
    finally:
        proc.wait()
        yield f"data: {json.dumps({'done': True, 'rc': proc.returncode})}\n\n"


# ── CSS + JS (shared) ─────────────────────────────────────────────────────────
_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --sb:#1e2330;--sb-h:#2a3040;--sb-a:#3b4262;
  --ac:#6366f1;--ac2:#818cf8;
  --bg:#f0f2f7;--card:#fff;--bdr:#e1e4ed;
  --tx:#1a1d2e;--mu:#6b7280;
  --ok:#16a34a;--er:#dc2626;--wn:#d97706;--inf:#0369a1;
  --code:#0d1117;
}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:var(--bg);color:var(--tx);display:flex;min-height:100vh}

/* sidebar */
.sb{width:230px;background:var(--sb);display:flex;flex-direction:column;
  flex-shrink:0;position:fixed;top:0;left:0;bottom:0;overflow-y:auto;z-index:10}
.sb-brand{padding:18px 16px 14px;border-bottom:1px solid #2d3350}
.sb-brand h1{color:#fff;font-size:13px;font-weight:700;letter-spacing:.5px;text-transform:uppercase}
.sb-brand p{color:#7c8799;font-size:11px;margin-top:3px}
.sb nav{padding:8px 8px;flex:1}
.sb nav a{display:flex;align-items:center;gap:9px;padding:9px 10px;border-radius:6px;
  color:#9ba5b8;text-decoration:none;font-size:13px;font-weight:500;margin-bottom:2px;
  transition:background .12s,color .12s}
.sb nav a:hover{background:var(--sb-h);color:#e2e8f0}
.sb nav a.active{background:var(--sb-a);color:#fff}
.sb nav a .ico{font-size:14px;width:18px;text-align:center}
.sb-sec{color:#4b5563;font-size:10px;text-transform:uppercase;letter-spacing:1px;
  padding:14px 10px 4px;font-weight:600}

/* main */
.main{margin-left:230px;flex:1;display:flex;flex-direction:column;min-height:100vh}
.topbar{background:var(--card);border-bottom:1px solid var(--bdr);padding:13px 26px;
  display:flex;align-items:center;gap:10px;position:sticky;top:0;z-index:5}
.topbar h2{font-size:15px;font-weight:600}
.content{padding:22px 26px;flex:1}

/* cards */
.card{background:var(--card);border:1px solid var(--bdr);border-radius:10px;padding:18px 20px}
.card+.card{margin-top:14px}
.card-title{font-size:11px;font-weight:600;color:var(--mu);text-transform:uppercase;
  letter-spacing:.5px;margin-bottom:10px}

/* metric grid */
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:12px;margin-bottom:20px}
.mc{background:var(--card);border:1px solid var(--bdr);border-radius:10px;
  padding:16px 18px}
.mc-label{font-size:10px;color:var(--mu);font-weight:600;text-transform:uppercase;letter-spacing:.4px}
.mc-val{font-size:28px;font-weight:700;margin:6px 0 3px}
.mc-sub{font-size:11px;color:var(--mu)}
.c-ok{color:var(--ok)}.c-er{color:var(--er)}.c-bl{color:var(--inf)}.c-or{color:var(--wn)}
.c-pu{color:var(--ac)}

/* charts */
.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:14px}
.cc{background:var(--card);border:1px solid var(--bdr);border-radius:10px;overflow:hidden}
.cc h3{padding:11px 15px;font-size:12px;font-weight:600;
  border-bottom:1px solid var(--bdr);color:var(--mu);text-transform:uppercase;letter-spacing:.4px}
.cc img{width:100%;display:block;cursor:zoom-in}

/* lightbox */
#lb{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:999;
  align-items:center;justify-content:center;cursor:zoom-out}
#lb.show{display:flex}
#lb img{max-width:90vw;max-height:90vh;border-radius:8px;box-shadow:0 20px 60px #000}

/* buttons */
.btn{display:inline-flex;align-items:center;gap:6px;padding:8px 16px;border-radius:7px;
  font-size:13px;font-weight:600;cursor:pointer;border:none;
  transition:opacity .12s,transform .1s;text-decoration:none;color:#fff}
.btn:hover{opacity:.88}.btn:active{transform:scale(.97)}
.btn-pr{background:var(--ac)}.btn-ok{background:var(--ok)}
.btn-er{background:var(--er)}.btn-wn{background:var(--wn)}
.btn-out{background:transparent;border:1px solid var(--bdr);color:var(--tx)}
.btn-sm{padding:5px 11px;font-size:12px}
.btn-group{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;align-items:center}

/* terminal */
.term{background:var(--code);border-radius:8px;padding:14px 16px;
  font-family:'Cascadia Code','Fira Code','Courier New',monospace;font-size:11.5px;
  color:#c9d1d9;min-height:180px;max-height:480px;overflow-y:auto;
  white-space:pre-wrap;word-break:break-all;line-height:1.55;margin-top:14px}
.lo{color:#7ee787}.le{color:#ff7b72}.li{color:#79c0ff}.lw{color:#e3b341}

/* steps */
.step-list{display:flex;flex-direction:column;gap:5px;margin-bottom:16px}
.step{display:flex;align-items:center;gap:10px;padding:9px 13px;
  background:var(--card);border:1px solid var(--bdr);border-radius:8px;font-size:12.5px;
  transition:background .2s,border-color .2s}
.step .sn{font-weight:600;width:130px;flex-shrink:0;color:var(--ac);font-size:12px}
.step .sd{color:var(--mu);flex:1;font-size:12px}
.step .ss{font-size:15px;width:22px;text-align:center}
.step.s-run{border-color:var(--ac);background:#eef2ff}
.step.s-ok{border-color:#86efac;background:#f0fdf4}
.step.s-ok .ss{color:var(--ok)}
.step.s-er{border-color:#fca5a5;background:#fff1f1}
.step.s-er .ss{color:var(--er)}

/* table */
.tbl-wrap{overflow-x:auto;margin-top:10px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{background:var(--bg);padding:8px 12px;text-align:left;font-size:11px;
  font-weight:600;color:var(--mu);text-transform:uppercase;letter-spacing:.4px;
  border-bottom:2px solid var(--bdr)}
td{padding:7px 12px;border-bottom:1px solid var(--bdr)}
tr:hover td{background:#f7f8fd}

/* tabs */
.tabs{display:flex;gap:0;border-bottom:2px solid var(--bdr);margin-bottom:14px;flex-wrap:wrap}
.tab{padding:8px 15px;font-size:12.5px;font-weight:500;cursor:pointer;
  border:none;background:none;color:var(--mu);border-bottom:3px solid transparent;
  margin-bottom:-2px;transition:color .12s}
.tab.act{color:var(--ac);border-bottom-color:var(--ac);font-weight:600}
.tp{display:none}.tp.act{display:block}

/* pill */
.pill{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;
  border-radius:99px;font-size:11px;font-weight:600}
.pill-ok{background:#dcfce7;color:#166534}.pill-er{background:#fee2e2;color:#991b1b}
.pill-run{background:#eef2ff;color:#3730a3}.pill-idle{background:#f3f4f6;color:var(--mu)}

/* evidence */
.ev-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:14px}
.ev-card{background:var(--card);border:1px solid var(--bdr);border-radius:10px;overflow:hidden}
.ev-card h3{padding:10px 15px;font-size:12.5px;font-weight:600;
  border-bottom:1px solid var(--bdr);background:var(--bg)}
.ev-card iframe{width:100%;height:480px;border:none}

/* report */
.rpt h1{font-size:19px;font-weight:700;margin:0 0 18px;padding-bottom:10px;border-bottom:2px solid var(--bdr)}
.rpt h2{font-size:14px;font-weight:700;margin:22px 0 9px;color:var(--inf)}
.rpt p,.rpt li{font-size:13.5px;line-height:1.7;margin-bottom:6px}
.rpt ul{padding-left:20px;margin-bottom:10px}
.rpt table{margin:8px 0}
.rpt code{background:var(--bg);padding:1px 5px;border-radius:4px;
  font-size:12px;font-family:monospace}
.rpt strong{font-weight:600}

/* spinner */
@keyframes spin{to{transform:rotate(360deg)}}
.spin{display:inline-block;width:13px;height:13px;border:2px solid #c7d2fe;
  border-top-color:var(--ac);border-radius:50%;animation:spin .7s linear infinite}

/* utils */
.row{display:flex;align-items:center;gap:10px}
.mb-3{margin-bottom:12px}.mb-4{margin-bottom:16px}.mb-5{margin-bottom:20px}
.text-mu{color:var(--mu);font-size:12px}
hr.div{border:none;border-top:1px solid var(--bdr);margin:18px 0}
"""

_JS_COMMON = """
// lightbox
document.addEventListener('click', e => {
  if (e.target.matches('.cc img')) {
    document.getElementById('lb').classList.add('show');
    document.getElementById('lb-img').src = e.target.src;
  } else if (e.target.closest('#lb')) {
    document.getElementById('lb').classList.remove('show');
  }
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') document.getElementById('lb').classList.remove('show');
});

// tabs
function switchTab(el, panelId, groupCls) {
  el.closest('.tabs').querySelectorAll('.tab').forEach(t => t.classList.remove('act'));
  document.querySelectorAll('.' + groupCls).forEach(p => p.classList.remove('act'));
  el.classList.add('act');
  document.getElementById(panelId).classList.add('act');
}

// SSE runner
function runSSE(url, termId, statusId, stepPrefix, onDone) {
  const term = document.getElementById(termId);
  const status = document.getElementById(statusId);
  if (!term) return;
  term.innerHTML = '';
  if (status) { status.className = 'pill pill-run'; status.innerHTML = '<span class="spin"></span> 실행 중'; }

  // reset all steps if stepPrefix given
  if (stepPrefix) {
    document.querySelectorAll('.step').forEach(s => {
      s.classList.remove('s-run','s-ok','s-er');
      s.querySelector('.ss').textContent = '○';
    });
  }

  const es = new EventSource(url);
  let currentStep = null;

  es.onmessage = e => {
    const d = JSON.parse(e.data);
    if (d.done) {
      es.close();
      // mark last running step as ok/err
      if (currentStep) {
        const sel = document.getElementById('step-' + currentStep);
        if (sel) {
          sel.classList.remove('s-run');
          sel.classList.add(d.rc === 0 ? 's-ok' : 's-er');
          sel.querySelector('.ss').textContent = d.rc === 0 ? '✓' : '✗';
        }
      }
      const ok = d.rc === 0;
      if (status) {
        status.className = 'pill ' + (ok ? 'pill-ok' : 'pill-er');
        status.textContent = ok ? '완료 ✓' : '실패 ✗ (rc=' + d.rc + ')';
      }
      if (onDone) onDone(ok);
      return;
    }
    const line = d.line || '';

    // detect step marker  [step-name]
    if (stepPrefix) {
      const m = line.match(/^\\[([A-Za-z0-9-]+)\\]/);
      if (m) {
        const name = m[1];
        if (currentStep && currentStep !== name) {
          const prev = document.getElementById('step-' + currentStep);
          if (prev && prev.classList.contains('s-run')) {
            prev.classList.remove('s-run');
            prev.classList.add('s-ok');
            prev.querySelector('.ss').textContent = '✓';
          }
        }
        currentStep = name;
        const el = document.getElementById('step-' + name);
        if (el) { el.classList.add('s-run'); el.querySelector('.ss').textContent = '…'; }
      }
    }

    // color lines
    const div = document.createElement('span');
    let cls = '';
    if (/ERROR|Traceback|FAILED/.test(line)) cls = 'le';
    else if (/^\\[|^===|→/.test(line)) cls = 'li';
    else if (/WARNING|warn/i.test(line)) cls = 'lw';
    else if (/PASSED|passed|✓|Done/.test(line)) cls = 'lo';
    if (cls) div.className = cls;
    div.textContent = line;
    term.appendChild(div);
    term.appendChild(document.createTextNode('\\n'));
    term.scrollTop = term.scrollHeight;
  };
  es.onerror = () => {
    es.close();
    if (status) { status.className = 'pill pill-er'; status.textContent = '연결 오류'; }
  };
}
"""

_BASE_TMPL = (
    """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FIMI-Cyber — {{ title }}</title>
<style>"""
    + _CSS
    + """</style>
</head>
<body>
<div class="sb">
  <div class="sb-brand">
    <h1>FIMI-Cyber PoC</h1>
    <p>Link Score Dashboard</p>
  </div>
  <nav>
    <div class="sb-sec">Overview</div>
    <a href="/" class="{{ 'active' if page=='dash' else '' }}"><span class="ico">📊</span> 대시보드</a>
    <div class="sb-sec">실행</div>
    <a href="/run"   class="{{ 'active' if page=='run'   else '' }}"><span class="ico">▶</span> 파이프라인</a>
    <a href="/tests" class="{{ 'active' if page=='tests' else '' }}"><span class="ico">🧪</span> 테스트</a>
    <div class="sb-sec">결과</div>
    <a href="/charts"   class="{{ 'active' if page=='charts'   else '' }}"><span class="ico">📈</span> 차트</a>
    <a href="/tables"   class="{{ 'active' if page=='tables'   else '' }}"><span class="ico">📋</span> 테이블</a>
    <a href="/report"   class="{{ 'active' if page=='report'   else '' }}"><span class="ico">📄</span> 보고서</a>
    <a href="/evidence" class="{{ 'active' if page=='evidence' else '' }}"><span class="ico">🕸</span> 증거 경로</a>
    <a href="/cluster"  class="{{ 'active' if page=='cluster'  else '' }}"><span class="ico">🔵</span> 클러스터 그래프</a>
  </nav>
</div>

<div class="main">
  <div class="topbar">
    <h2>{{ title }}</h2>
    {{ topbar | safe }}
  </div>
  <div class="content">{{ content | safe }}</div>
</div>

<!-- lightbox -->
<div id="lb"><img id="lb-img" src=""></div>

<script>"""
    + _JS_COMMON
    + """
{{ js | safe }}
</script>
</body>
</html>"""
)


def _render(title, page, content, topbar="", js=""):
    return render_template_string(
        _BASE_TMPL, title=title, page=page,
        content=content, topbar=topbar, js=js,
    )


# ── Static file routes ────────────────────────────────────────────────────────

@app.route("/fig/<name>")
def fig(name):
    return send_from_directory(FIGURES, name)


@app.route("/evidence/file/<name>")
def evidence_file(name):
    return send_from_directory(EVIDENCE_DIR, name)


@app.route("/evidence/file/lib/<path:filename>")
def evidence_lib(filename):
    lib_dir = ROOT / "lib"
    if lib_dir.exists():
        return send_from_directory(lib_dir, filename)
    return "", 404


# ── SSE API ───────────────────────────────────────────────────────────────────

@app.route("/api/run/pipeline")
def api_run_pipeline():
    if _running.get("pipeline"):
        return Response("data: {\"line\": \"이미 실행 중\"}\n\ndata: {\"done\":true,\"rc\":1}\n\n",
                        mimetype="text/event-stream")
    _running["pipeline"] = True

    def gen():
        try:
            yield from _sse_stream([sys.executable, str(ROOT / "scripts" / "run_all.py")])
        finally:
            _running["pipeline"] = False

    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/run/tests")
def api_run_tests():
    if _running.get("tests"):
        return Response("data: {\"line\": \"이미 실행 중\"}\n\ndata: {\"done\":true,\"rc\":1}\n\n",
                        mimetype="text/event-stream")
    _running["tests"] = True

    def gen():
        try:
            yield from _sse_stream(
                [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "--no-header"],
            )
        finally:
            _running["tests"] = False

    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    # Load metrics if available
    metrics: dict[str, dict] = {}
    mp = RESULTS / "metrics_summary.csv"
    if mp.exists():
        try:
            import csv
            with open(mp) as f:
                for row in csv.DictReader(f):
                    metrics[row["condition"]] = row
        except Exception:
            pass

    def _mc(label, val, sub, color):
        return (f'<div class="mc"><div class="mc-label">{label}</div>'
                f'<div class="mc-val {color}">{val}</div>'
                f'<div class="mc-sub">{sub}</div></div>')

    cards = ""
    if metrics:
        for cond, color, sub in [
            ("E1", "c-bl",  "내러티브 only"),
            ("E2", "c-or",  "IOC only"),
            ("E3", "c-ok",  "통합 (α·N + β·I + …)"),
        ]:
            m = metrics.get(cond, {})
            map_v = f"{float(m['MAP']):.3f}" if "MAP" in m else "—"
            ndcg  = f"nDCG@10: {float(m['nDCG@10']):.3f}" if "nDCG@10" in m else ""
            cards += _mc(f"MAP — {cond}", map_v, ndcg, color)
    else:
        cards = '<div class="text-mu">결과 없음. 파이프라인을 먼저 실행하세요.</div>'

    # key chart
    chart_html = ""
    bar = FIGURES / "metrics_bar.png"
    if bar.exists():
        chart_html = (
            '<div class="cc" style="max-width:620px">'
            '<h3>E1 / E2 / E3 성능</h3>'
            '<img src="/fig/metrics_bar.png" alt="metrics"></div>'
        )

    # recent files
    recent = ""
    csv_files = ["metrics_summary.csv", "ablation.csv", "pairwise_scores.csv"]
    rows = ""
    for fn in csv_files:
        fp = RESULTS / fn
        if fp.exists():
            import time
            mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(fp.stat().st_mtime))
            rows += f"<tr><td>{fn}</td><td>{mtime}</td></tr>"
    if rows:
        recent = (
            '<div class="card" style="margin-top:18px">'
            '<div class="card-title">결과 파일</div>'
            '<div class="tbl-wrap"><table><thead><tr><th>파일</th><th>최종 수정</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div></div>"
        )

    content = (
        f'<div class="metric-grid">{cards}</div>'
        f'{chart_html}'
        f'{recent}'
        '<div style="margin-top:18px" class="btn-group">'
        '<a href="/run"   class="btn btn-pr">▶ 파이프라인 실행</a>'
        '<a href="/tests" class="btn btn-ok">🧪 테스트 실행</a>'
        '<a href="/charts" class="btn btn-out">📈 차트 보기</a>'
        '</div>'
    )
    return _render("대시보드", "dash", content)


# ── Pipeline runner ───────────────────────────────────────────────────────────

@app.route("/run")
def run_page():
    step_items = ""
    for name, desc in STEPS:
        step_items += (
            f'<div class="step" id="step-{name}">'
            f'<span class="sn">{name}</span>'
            f'<span class="sd">{desc}</span>'
            f'<span class="ss">○</span>'
            f'</div>'
        )

    content = (
        '<div class="btn-group mb-4">'
        '<button class="btn btn-pr" onclick="startPipeline()">▶ 전체 파이프라인 실행</button>'
        '<span id="pipe-status" class="pill pill-idle">대기</span>'
        '</div>'
        f'<div class="step-list">{step_items}</div>'
        '<div class="card-title">실행 로그</div>'
        '<div class="term" id="pipe-term"></div>'
    )

    js = """
function startPipeline() {
  runSSE('/api/run/pipeline', 'pipe-term', 'pipe-status', true, rc => {
    if (rc) {
      // mark remaining idle steps as ok if pipeline succeeded
    }
  });
}
"""
    return _render("파이프라인 실행", "run", content, js=js)


# ── Test runner ───────────────────────────────────────────────────────────────

@app.route("/tests")
def tests_page():
    content = (
        '<div class="btn-group mb-4">'
        '<button class="btn btn-ok" onclick="startTests()">🧪 pytest 실행</button>'
        '<span id="test-status" class="pill pill-idle">대기</span>'
        '</div>'
        '<div class="card mb-4" style="background:#f0fdf4;border-color:#86efac">'
        '<div class="card-title">테스트 목록</div>'
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:6px;font-size:12.5px">'
        + "".join(
            f'<div style="padding:5px 8px;background:#fff;border:1px solid #e1e4ed;border-radius:6px">'
            f'<code style="color:#6366f1">{t}</code> — <span style="color:#6b7280">{d}</span></div>'
            for t, d in [
                ("T1",  "refang"),
                ("T2",  "IPv4 추출"),
                ("T3",  "IOC 분류"),
                ("T4",  "Confidence=0.83"),
                ("T5",  "N(i,j) 골든"),
                ("T6",  "I_direct=0.3182"),
                ("T7",  "temporal_overlap"),
                ("T8",  "Jaccard/Overlap/mix"),
                ("T9",  "FCLS=0.6375"),
                ("T10", "ζ=0 가드"),
                ("T11", "예약 대역 IOC"),
                ("T12", "결정론적 manifest"),
                ("T13", "MAP=1.0 완벽 랭킹"),
                ("T14", "Event 경유 금지"),
            ]
        )
        + "</div></div>"
        '<div class="card-title">실행 로그</div>'
        '<div class="term" id="test-term"></div>'
    )

    js = """
function startTests() {
  runSSE('/api/run/tests', 'test-term', 'test-status', false, ok => {});
}
"""
    return _render("테스트", "tests", content, js=js)


# ── Charts ────────────────────────────────────────────────────────────────────

@app.route("/charts")
def charts_page():
    items = ""
    for fname, label in CHART_INFO:
        if (FIGURES / fname).exists():
            items += (
                f'<div class="cc">'
                f'<h3>{label}</h3>'
                f'<img src="/fig/{fname}" alt="{label}" title="클릭하면 크게 보기">'
                f'</div>'
            )
        else:
            items += (
                f'<div class="cc"><h3>{label}</h3>'
                f'<div style="padding:40px;text-align:center;color:var(--mu);font-size:12px">'
                f'{fname} 없음 — 파이프라인을 실행하세요</div></div>'
            )

    topbar = '<a href="/run" class="btn btn-pr btn-sm">▶ 파이프라인 실행</a>'
    content = (
        '<p class="text-mu mb-4">클릭하면 크게 볼 수 있습니다.</p>'
        f'<div class="chart-grid">{items}</div>'
    )
    return _render("차트", "charts", content, topbar=topbar)


# ── Tables ────────────────────────────────────────────────────────────────────

CSV_TABS = [
    ("metrics", "성능 지표",   "metrics_summary.csv"),
    ("abl",     "Ablation",   "ablation.csv"),
    ("grid",    "Grid Search","gridsearch.csv"),
    ("rob",     "강건성",      "robustness.csv"),
    ("prio",    "Priority",   "priority_table.csv"),
    ("pairs",   "쌍별 점수",   "pairwise_scores.csv"),
]


def _csv_to_html(path: Path) -> str:
    if not path.exists():
        return f'<div class="text-mu" style="padding:20px">파일 없음: {path.name}</div>'
    try:
        import csv
        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return '<div class="text-mu">빈 파일</div>'
        header, body = rows[0], rows[1:]
        ths = "".join(f"<th>{html.escape(h)}</th>" for h in header)
        trs = ""
        for row in body:
            tds = "".join(f"<td>{html.escape(c)}</td>" for c in row)
            trs += f"<tr>{tds}</tr>"
        return (
            f'<div class="tbl-wrap"><table>'
            f'<thead><tr>{ths}</tr></thead>'
            f'<tbody>{trs}</tbody>'
            f'</table></div>'
        )
    except Exception as e:
        return f'<div class="text-mu">로드 오류: {e}</div>'


@app.route("/tables")
def tables_page():
    tab_btns = ""
    panels = ""
    for i, (tid, label, fname) in enumerate(CSV_TABS):
        act = "act" if i == 0 else ""
        tab_btns += (
            f'<button class="tab {act}" '
            f'onclick="switchTab(this,\'tp-{tid}\',\'tp\')">{label}</button>'
        )
        panels += (
            f'<div class="tp {act}" id="tp-{tid}">'
            + _csv_to_html(RESULTS / fname)
            + "</div>"
        )

    content = (
        f'<div class="tabs">{tab_btns}</div>'
        f'{panels}'
    )
    return _render("테이블", "tables", content)


# ── Report ────────────────────────────────────────────────────────────────────

@app.route("/report")
def report_page():
    rp = RESULTS / "report.md"
    if not rp.exists():
        content = '<div class="text-mu">report.md 없음. 파이프라인을 실행하세요.</div>'
    else:
        md_text = rp.read_text(encoding="utf-8")
        try:
            import markdown
            body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
        except Exception:
            body = f"<pre>{html.escape(md_text)}</pre>"
        content = f'<div class="card rpt">{body}</div>'

    return _render("보고서", "report", content)


# ── Evidence paths ────────────────────────────────────────────────────────────

@app.route("/evidence")
def evidence_page():
    pairs = list(EVIDENCE_DIR.glob("pair_*.html")) if EVIDENCE_DIR.exists() else []

    if not pairs:
        content = (
            '<div class="text-mu">증거 경로 HTML 없음. 파이프라인을 실행하세요.</div>'
            '<div style="margin-top:12px">'
            '<a href="/run" class="btn btn-pr btn-sm">▶ 파이프라인 실행</a>'
            '</div>'
        )
    else:
        cards = ""
        for p in sorted(pairs):
            name = p.name
            pair_label = name.replace("pair_", "").replace(".html", "").replace("_", " ↔ ")
            cards += (
                f'<div class="ev-card">'
                f'<h3>증거 경로: {pair_label}</h3>'
                f'<iframe src="/evidence/file/{name}" loading="lazy"></iframe>'
                f'</div>'
            )
        content = f'<div class="ev-grid">{cards}</div>'

    return _render("증거 경로", "evidence", content)


# ── Cluster graph ─────────────────────────────────────────────────────────────

@app.route("/cluster/file")
def cluster_file():
    p = RESULTS / "event_cluster.html"
    if p.exists():
        return send_from_directory(RESULTS, "event_cluster.html")
    return "No cluster graph yet", 404


@app.route("/cluster/file/lib/<path:filename>")
def cluster_lib(filename):
    return send_from_directory(ROOT / "lib", filename)


@app.route("/api/run/cluster")
def api_run_cluster():
    if _running.get("cluster"):
        return Response(
            "data: {\"line\":\"이미 실행 중\"}\n\ndata: {\"done\":true,\"rc\":1}\n\n",
            mimetype="text/event-stream",
        )
    _running["cluster"] = True

    def gen():
        try:
            yield from _sse_stream([
                sys.executable, "-c",
                "import sys; sys.path.insert(0,'src');"
                "from fimicyber.config import load_config;"
                "from fimicyber.schema import Event;"
                "import json, pandas as pd;"
                "cfg = load_config();"
                "events = [Event.model_validate_json(l) for l in open('data/interim/events.jsonl')];"
                "scores_df = pd.read_csv('results/pairwise_scores.csv');"
                "from fimicyber.viz.cluster import build_event_cluster, plot_event_cluster_static;"
                "build_event_cluster(events, scores_df, cfg);"
                "plot_event_cluster_static(events, scores_df, cfg);"
                "print('클러스터 그래프 생성 완료')",
            ])
        finally:
            _running["cluster"] = False

    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/cluster")
def cluster_page():
    html_exists = (RESULTS / "event_cluster.html").exists()
    png_exists  = (RESULTS / "figures" / "event_cluster.png").exists()

    build_btn = (
        '<button class="btn btn-pr" onclick="rebuildCluster()">🔄 재생성</button>'
        '<span id="cl-status" class="pill pill-idle">대기</span>'
    )

    if html_exists:
        graph_area = (
            '<div style="border:1px solid var(--bdr);border-radius:10px;overflow:hidden;'
            'height:calc(100vh - 160px);margin-bottom:16px">'
            '<iframe src="/cluster/file" style="width:100%;height:100%;border:none" '
            'id="cluster-iframe"></iframe>'
            '</div>'
        )
    else:
        graph_area = (
            '<div class="card" style="text-align:center;padding:60px;color:var(--mu)">'
            '클러스터 그래프가 없습니다. 재생성 버튼을 눌러 생성하세요.'
            '</div>'
        )

    static_area = ""
    if png_exists:
        static_area = (
            '<div class="cc" style="margin-top:14px">'
            '<h3>정적 스프링 레이아웃 (PNG)</h3>'
            '<img src="/fig/event_cluster.png" style="width:100%;cursor:zoom-in">'
            '</div>'
        )

    term = '<div class="card-title" style="margin-top:14px">재생성 로그</div><div class="term" id="cl-term" style="min-height:80px"></div>'

    content = (
        f'<div class="btn-group mb-4">{build_btn}</div>'
        f'{graph_area}'
        f'{static_area}'
        f'{term}'
    )

    js = """
function rebuildCluster() {
  runSSE('/api/run/cluster', 'cl-term', 'cl-status', false, ok => {
    if (ok) {
      document.getElementById('cluster-iframe') &&
        (document.getElementById('cluster-iframe').src = '/cluster/file?' + Date.now());
    }
  });
}
"""
    return _render("클러스터 그래프", "cluster", content, js=js)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FIMI-Cyber PoC Dashboard")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"\n  FIMI-Cyber PoC Dashboard")
    print(f"  http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
