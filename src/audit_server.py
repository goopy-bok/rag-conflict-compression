"""
Interactive Unified Audit Server for CCD-G1 Experiments.
Supports Sheets 1 through 6 in a single web interface:
  1. G0 저자 라벨 (author_labels.csv)
  2. C2 Distractor FCR 검토 (sheet2_c2_distractor_fcr.csv)
  3. C2 Test-ID CSR·D 검토 (sheet3_c2_test_id_csr_d.csv)
  4. C3-G Test-ID CSR·D 검토 (sheet4_c3g_test_id_csr_d.csv)
  5. C1 OOD2 CSR 검토 (sheet5_c1_test_ood2_csr.csv)
  6. C3-G OOD2 CSR 검토 (sheet6_c3g_test_ood2_csr.csv)

Usage:
  uv run python src/audit_server.py
  uv run python src/audit_server.py --port 8768 --no-browser
  uv run python src/audit_server.py --blind      # judge 판정·근거를 가리고 라벨 (권장)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
AUDIT_DIR = RESULTS_DIR / "audit_sheets"
G0_DIR = RESULTS_DIR / "g0_local"

SHEETS = [
    {
        "id": "sheet1",
        "title": "1. G0 저자 라벨 (C0/C1)",
        "path": G0_DIR / "author_labels.csv",
        "type": "g0",
        "key_col": "item_key",
        "flags": ["a_supported", "a_distorted", "b_supported", "b_distorted"],
    },
    {
        "id": "sheet2",
        "title": "2. C2 Distractor FCR 검토",
        "path": AUDIT_DIR / "sheet2_c2_distractor_fcr.csv",
        "type": "fcr",
        "key_col": "id",
        "flags": ["human_is_conflict", "human_category"],
    },
    {
        "id": "sheet3",
        "title": "3. C2 Test-ID CSR·D",
        "path": AUDIT_DIR / "sheet3_c2_test_id_csr_d.csv",
        "type": "csr_d",
        "key_col": "id",
        "flags": ["human_csr", "human_d"],
    },
    {
        "id": "sheet4",
        "title": "4. C3-G Test-ID CSR·D",
        "path": AUDIT_DIR / "sheet4_c3g_test_id_csr_d.csv",
        "type": "csr_d",
        "key_col": "id",
        "flags": ["human_csr", "human_d"],
    },
    {
        "id": "sheet5",
        "title": "5. C1 OOD2 CSR",
        "path": AUDIT_DIR / "sheet5_c1_test_ood2_csr.csv",
        "type": "csr_only",
        "key_col": "id",
        "flags": ["human_csr"],
    },
    {
        "id": "sheet6",
        "title": "6. C3-G OOD2 CSR",
        "path": AUDIT_DIR / "sheet6_c3g_test_ood2_csr.csv",
        "type": "csr_only",
        "key_col": "id",
        "flags": ["human_csr"],
    },
    {
        # 2026-10-12 압축기 탐색: gpt-4o-mini C2 dev, judge CSR=1 20건 + CSR=0 5건 섞음(키는 sheet7_label_key.json)
        "id": "sheet7",
        "title": "7. gpt-4o-mini C2 dev 블라인드 (25건)",
        "path": AUDIT_DIR / "sheet7_4omini_c2_dev_csr.csv",
        "type": "g0",
        "key_col": "item_key",
        "flags": ["a_supported", "a_distorted", "b_supported", "b_distorted"],
    },
]


def load_sheet_rows(path: Path) -> tuple[list[str], list[dict]]:
    if not path.exists():
        return [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def write_sheet_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


PAGE_HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CCD-G1 실험 오류·타당성 종합 검토</title>
<style>
:root {
  --bg: #f8fafc;
  --card: #ffffff;
  --ink: #0f172a;
  --mut: #64748b;
  --line: #e2e8f0;
  --pri: #2563eb;
  --pri-bg: #eff6ff;
  --a: #0284c7;
  --b: #d97706;
  --ok: #16a34a;
  --err: #dc2626;
  --warn-bg: #fffbeb;
  --warn-border: #fef3c7;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f172a;
    --card: #1e293b;
    --ink: #f8fafc;
    --mut: #94a3b8;
    --line: #334155;
    --pri: #3b82f6;
    --pri-bg: #1e3a8a;
    --a: #38bdf8;
    --b: #fbbf24;
    --ok: #4ade80;
    --err: #f87171;
    --warn-bg: #451a03;
    --warn-border: #78350f;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 14.5px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
header { background: var(--card); border-bottom: 1px solid var(--line); padding: 10px 20px; position: sticky; top: 0; z-index: 10; }
.tabs { display: flex; gap: 8px; overflow-x: auto; padding-bottom: 4px; }
.tab { padding: 6px 14px; border: 1px solid var(--line); border-radius: 8px; background: transparent; color: var(--mut); cursor: pointer; white-space: nowrap; font-size: 13.5px; font-weight: 500; }
.tab.active { background: var(--pri); color: #fff; border-color: var(--pri); }
.tab .badge { margin-left: 6px; font-size: 11px; padding: 2px 6px; border-radius: 10px; background: rgba(0,0,0,0.1); }
main { max-width: 1020px; margin: 16px auto; padding: 0 16px 40px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px 20px; margin-bottom: 14px; }
.header-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
h1 { font-size: 17px; margin: 0; }
.mut { color: var(--mut); font-size: 13px; }
.ctx { white-space: pre-wrap; font-family: ui-monospace, Menlo, Monaco, Consolas, monospace; font-size: 13px; line-height: 1.5; background: var(--bg); padding: 12px; border-radius: 8px; border: 1px solid var(--line); max-height: 38vh; overflow-y: auto; }
.tag-bar { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin: 8px 0; }
.cand { font-weight: 600; }
.A { color: var(--a); }
.B { color: var(--b); }
.btn { font: inherit; padding: 6px 14px; border-radius: 6px; border: 1px solid var(--line); background: var(--card); color: var(--ink); cursor: pointer; font-size: 13.5px; }
.btn:hover { border-color: var(--mut); }
.btn.on { background: var(--pri); color: #fff; border-color: var(--pri); font-weight: 600; }
.btn.on-err { background: var(--err); color: #fff; border-color: var(--err); font-weight: 600; }
.btn.on-ok { background: var(--ok); color: #fff; border-color: var(--ok); font-weight: 600; }
.category-btn { font-size: 12px; padding: 4px 10px; border-radius: 6px; border: 1px solid var(--line); background: var(--card); color: var(--ink); cursor: pointer; }
.category-btn.on { background: var(--ink); color: var(--card); }
textarea { width: 100%; font: inherit; border: 1px solid var(--line); border-radius: 6px; background: var(--card); color: var(--ink); padding: 8px; resize: vertical; }
.eval-box { background: var(--pri-bg); border: 1px solid var(--pri); border-radius: 8px; padding: 10px 14px; margin: 10px 0; font-size: 13px; }
.eval-box summary { cursor: pointer; font-weight: 600; color: var(--pri); }
.nav-row { display: flex; justify-content: space-between; align-items: center; margin-top: 14px; }
mark.ma { background: #bfdbfe; color: #000; border-radius: 2px; }
mark.mb { background: #fde68a; color: #000; border-radius: 2px; }
.rubric { font-size: 12.5px; line-height: 1.5; color: var(--mut); border-left: 3px solid var(--pri); padding-left: 10px; margin: 8px 0; }
</style>
</head>
<body>
<header>
  <div class="tabs" id="tabBar"></div>
</header>
<main>
  <div class="header-row">
    <div>
      <h1 id="sheetTitle">로딩 중…</h1>
      <span class="mut" id="sheetDesc"></span>
    </div>
    <div>
      <span id="posIndicator" class="mut" style="font-weight:600; margin-right: 12px;"></span>
      <button class="btn" id="btnNextEmpty">미완료 항목으로</button>
    </div>
  </div>

  <section class="card" id="contentCard">
    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
      <div style="font-size:13px; font-weight:600; color:var(--mut);" id="itemId">ID: -</div>
      <label class="mut" style="font-size:12px; cursor:pointer;">
        <input type="checkbox" id="chkHl" checked> 키워드 강조
      </label>
    </div>
    <div style="margin: 8px 0;">
      <div style="font-size:15px; font-weight:600; margin-bottom: 6px;" id="qText"></div>
      <div id="goldSection" style="margin-bottom: 6px;"></div>
    </div>
    
    <div id="judgeBox" class="eval-box" style="display:none;"></div>

    <div style="margin-top: 10px;">
      <div class="mut" style="font-weight:600; font-size:12px; margin-bottom:4px;">압축 모델 출력 (COMPRESSED CONTEXT)</div>
      <div class="ctx" id="ctxText"></div>
    </div>

    <div id="downstreamSection" style="margin-top: 12px; display:none;">
      <div class="mut" style="font-weight:600; font-size:12px; margin-bottom:4px;">하류 생성 모델 답변 (DOWNSTREAM ANSWER)</div>
      <div class="ctx" id="downstreamText" style="max-height: 18vh; background: var(--warn-bg); border-color: var(--warn-border);"></div>
    </div>
  </section>

  <section class="card" id="labelCard">
    <div style="font-size:14px; font-weight:600; margin-bottom: 10px;" id="labelHeading">사람 검토 판정 입력</div>
    <div id="labelControls"></div>
    
    <div style="margin-top: 12px;">
      <div class="mut" style="font-size:12.5px; margin-bottom: 4px;">판정 메모 (특이사항, 모순 경계선, 오판 근거 등)</div>
      <textarea id="txtNote" rows="2" placeholder="판정 사유 또는 특이사항 기록"></textarea>
    </div>
  </section>

  <div class="nav-row">
    <div>
      <button class="btn" id="btnPrev">← 이전 [←]</button>
      <button class="btn" id="btnNext">다음 [→] →</button>
    </div>
    <span id="saveStatus" class="mut" style="font-weight:500;"></span>
  </div>
</main>

<script>
let sheets = [];
let curSheetId = "sheet1";
let curRows = [];
let curIdx = 0;

const $ = id => document.getElementById(id);

async function init() {
  const res = await fetch("/api/sheets");
  sheets = await res.parse ? await res.parse() : await res.json();
  renderTabs();
  loadSheet(sheets[0].id);
}

function renderTabs() {
  const bar = $("tabBar");
  bar.innerHTML = "";
  sheets.forEach(s => {
    const btn = document.createElement("button");
    btn.className = "tab" + (s.id === curSheetId ? " active" : "");
    btn.innerHTML = `${s.title} <span class="badge" id="badge_${s.id}">${s.done}/${s.total}</span>`;
    btn.onclick = () => loadSheet(s.id);
    bar.appendChild(btn);
  });
}

async function loadSheet(sheetId) {
  curSheetId = sheetId;
  renderTabs();
  const res = await fetch(`/api/sheet?id=${sheetId}`);
  const data = await res.json();
  curRows = data.rows;
  curIdx = 0;
  
  // Find first unlabelled item if any
  const sheetMeta = sheets.find(s => s.id === curSheetId);
  const firstEmpty = curRows.findIndex(r => !isRowDone(r, sheetMeta));
  if (firstEmpty >= 0) curIdx = firstEmpty;
  
  renderItem();
}

function isRowDone(row, meta) {
  if (meta.type === "g0") {
    return ["a_supported","a_distorted","b_supported","b_distorted"].every(f => row[f] === "1" || row[f] === "0");
  } else if (meta.type === "fcr") {
    return row.human_is_conflict === "1" || row.human_is_conflict === "0";
  } else if (meta.type === "csr_d") {
    return (row.human_csr === "1" || row.human_csr === "0") && (row.human_d === "1" || row.human_d === "0");
  } else if (meta.type === "csr_only") {
    return row.human_csr === "1" || row.human_csr === "0";
  }
  return false;
}

function updateBadge() {
  const meta = sheets.find(s => s.id === curSheetId);
  if (!meta) return;
  meta.done = curRows.filter(r => isRowDone(r, meta)).length;
  meta.total = curRows.length;
  const b = $(`badge_${curSheetId}`);
  if (b) b.textContent = `${meta.done}/${meta.total}`;
}

function highlightText(el, text, terms) {
  el.textContent = "";
  if (!$("chkHl").checked || !terms || terms.length === 0) {
    el.textContent = text;
    return;
  }
  const cleanTerms = terms.filter(t => t && t.trim().length > 1).map((t, idx) => ({ text: t.toLowerCase(), orig: t, cls: idx === 0 ? "ma" : "mb" }));
  let lower = text.toLowerCase();
  let p = 0;
  while (p < text.length) {
    let best = null;
    for (const item of cleanTerms) {
      const pos = lower.indexOf(item.text, p);
      if (pos >= 0 && (!best || pos < best.pos)) {
        best = { pos, len: item.text.length, cls: item.cls };
      }
    }
    if (!best) {
      el.append(text.slice(p));
      break;
    }
    el.append(text.slice(p, best.pos));
    const mark = document.createElement("mark");
    mark.className = best.cls;
    mark.textContent = text.slice(best.pos, best.pos + best.len);
    el.append(mark);
    p = best.pos + best.len;
  }
}

function renderItem() {
  if (curRows.length === 0) {
    $("sheetTitle").textContent = "데이터가 비어있습니다.";
    return;
  }
  const meta = sheets.find(s => s.id === curSheetId);
  const r = curRows[curIdx];
  const done = isRowDone(r, meta);

  $("sheetTitle").textContent = meta.title;
  $("posIndicator").textContent = `${curIdx + 1} / ${curRows.length} ${done ? "✓ 완료" : "· 미완료"}`;
  $("itemId").textContent = `Item ID: ${r.item_key || r.id}`;
  $("qText").textContent = `Q: ${r.question || ""}`;

  // Gold Answers Section
  const gBox = $("goldSection");
  gBox.innerHTML = "";
  let hlTerms = [];
  if (r.answer_a || r.gold_a) {
    const ga = r.answer_a || r.gold_a;
    hlTerms.push(ga);
    gBox.innerHTML += `<div style="font-size:13.5px;"><span class="cand A">기준 정답 A</span>: <b>${ga}</b></div>`;
  }
  if (r.answer_b || r.gold_b) {
    const gb = r.answer_b || r.gold_b;
    hlTerms.push(gb);
    gBox.innerHTML += `<div style="font-size:13.5px;"><span class="cand B">기준 정답 B</span>: <b>${gb}</b></div>`;
  }

  // Judge Box
  const jBox = $("judgeBox");
  if (r.judge_rationale || r.judge_false_conflict !== undefined || r.auto_csr !== undefined) {
    jBox.style.display = "block";
    let flagsText = [];
    if (r.judge_false_conflict !== undefined) {
      flagsText.push(`가짜충돌 판정: <b>${r.judge_false_conflict === "1" ? "TRUE (충돌 날조)" : "FALSE"}</b>`);
    }
    if (r.auto_csr !== undefined) {
      flagsText.push(`자동 CSR: <b>${r.auto_csr}</b>`);
    }
    if (r.auto_d !== undefined) {
      flagsText.push(`자동 D (충돌인정): <b>${r.auto_d}</b>`);
    }
    jBox.innerHTML = `
      <details>
        <summary>평가기 판정 근거 (Judge Rationale) — [${flagsText.join(" | ")}]</summary>
        <div style="margin-top:6px; font-size:12.5px; line-height:1.45; color:var(--ink);">${r.judge_rationale || "Rationale 없음"}</div>
      </details>
    `;
  } else {
    jBox.style.display = "none";
  }

  // Compressed Context
  highlightText($("ctxText"), r.compressed || "", hlTerms);

  // Downstream Answer
  const dsSection = $("downstreamSection");
  if (r.downstream_answer) {
    dsSection.style.display = "block";
    highlightText($("downstreamText"), r.downstream_answer, hlTerms);
  } else {
    dsSection.style.display = "none";
  }

  // Note
  $("txtNote").value = r.note || r.human_note || "";

  // Dynamic Controls based on sheet type
  renderControls(r, meta);
}

function renderControls(r, meta) {
  const box = $("labelControls");
  box.innerHTML = "";

  if (meta.type === "g0") {
    // Sheet 1: 4 flags
    const flags = [
      ["a_supported", "A 지지 (질문에 대한 유효한 답)", "A"],
      ["a_distorted", "A 왜곡 (값 변경·잘못된 귀속·병합)", "A"],
      ["b_supported", "B 지지 (질문에 대한 유효한 답)", "B"],
      ["b_distorted", "B 왜곡 (값 변경·잘못된 귀속·병합)", "B"]
    ];
    flags.forEach(([f, label, side]) => {
      const row = document.createElement("div");
      row.className = "tag-bar";
      row.innerHTML = `<span class="cand ${side}" style="min-width:240px;">${label}:</span>`;
      ["1", "0"].forEach(val => {
        const btn = document.createElement("button");
        btn.className = "btn" + (r[f] === val ? (val === "1" ? " on-ok" : " on") : "");
        btn.textContent = val === "1" ? "1 (Yes)" : "0 (No)";
        btn.onclick = () => saveField(f, val);
        row.appendChild(btn);
      });
      box.appendChild(row);
    });
  } else if (meta.type === "fcr") {
    // Sheet 2: C2 Distractor FCR
    const row1 = document.createElement("div");
    row1.className = "tag-bar";
    row1.innerHTML = `<span style="font-weight:600; min-width:220px;">실제 가짜 충돌 여부 (Human FCR):</span>`;
    [
      ["1", "1: 참된 가짜 충돌 (무충돌 입력에서 모순 생성)", "on-err"],
      ["0", "0: 충돌 아님 (평가기 오탐 / 단순 무관 정보)", "on-ok"]
    ].forEach(([val, txt, cls]) => {
      const btn = document.createElement("button");
      btn.className = "btn" + (r.human_is_conflict === val ? ` ${cls}` : "");
      btn.textContent = txt;
      btn.onclick = () => saveField("human_is_conflict", val);
      row1.appendChild(btn);
    });
    box.appendChild(row1);

    const row2 = document.createElement("div");
    row2.className = "tag-bar";
    row2.style.marginTop = "8px";
    row2.innerHTML = `<span class="mut" style="font-size:13px; min-width:220px;">평가기 결함 분류 태그:</span>`;
    const categories = ["주체 혼동", "외부 정답 주입", "동어 반복 오판", "양립 가능 속성", "압축기 결함", "정상 판정"];
    categories.forEach(cat => {
      const btn = document.createElement("button");
      btn.className = "category-btn" + (r.human_category === cat ? " on" : "");
      btn.textContent = cat;
      btn.onclick = () => saveField("human_category", cat);
      row2.appendChild(btn);
    });
    box.appendChild(row2);
  } else if (meta.type === "csr_d") {
    // Sheet 3 & 4: CSR & D
    const row1 = document.createElement("div");
    row1.className = "tag-bar";
    row1.innerHTML = `<span style="font-weight:600; min-width:220px;">실제 CSR 보존 성공 (A·B 양립):</span>`;
    [
      ["1", "1: 양쪽 답 질문에 맞게 유효 보존", "on-ok"],
      ["0", "0: 유실 또는 범주 왜곡 (직장/학교 혼동 등)", "on-err"]
    ].forEach(([val, txt, cls]) => {
      const btn = document.createElement("button");
      btn.className = "btn" + (r.human_csr === val ? ` ${cls}` : "");
      btn.textContent = txt;
      btn.onclick = () => saveField("human_csr", val);
      row1.appendChild(btn);
    });
    box.appendChild(row1);

    const row2 = document.createElement("div");
    row2.className = "tag-bar";
    row2.style.marginTop = "8px";
    row2.innerHTML = `<span style="font-weight:600; min-width:220px;">실제 하류 충돌 인지 (Human D):</span>`;
    [
      ["1", "1: 하류 답변이 상충 인정/병기함", "on-ok"],
      ["0", "0: 단정 답안 또는 불일치 무시", "on-err"]
    ].forEach(([val, txt, cls]) => {
      const btn = document.createElement("button");
      btn.className = "btn" + (r.human_d === val ? ` ${cls}` : "");
      btn.textContent = txt;
      btn.onclick = () => saveField("human_d", val);
      row2.appendChild(btn);
    });
    box.appendChild(row2);
  } else if (meta.type === "csr_only") {
    // Sheet 5 & 6: OOD2 CSR
    const row1 = document.createElement("div");
    row1.className = "tag-bar";
    row1.innerHTML = `<span style="font-weight:600; min-width:220px;">실제 CSR 보존 성공:</span>`;
    [
      ["1", "1: 양쪽 답 모두 질문에 대해 보존됨", "on-ok"],
      ["0", "0: 보존 실패 또는 범주 왜곡", "on-err"]
    ].forEach(([val, txt, cls]) => {
      const btn = document.createElement("button");
      btn.className = "btn" + (r.human_csr === val ? ` ${cls}` : "");
      btn.textContent = txt;
      btn.onclick = () => saveField("human_csr", val);
      row1.appendChild(btn);
    });
    box.appendChild(row1);
  }
}

async function saveField(fieldName, value) {
  const r = curRows[curIdx];
  r[fieldName] = value;
  updateBadge();
  renderItem();

  const keyCol = sheets.find(s => s.id === curSheetId).key_col;
  const updatePayload = {
    sheet_id: curSheetId,
    key_val: r[keyCol],
    updates: { [fieldName]: value }
  };

  $("saveStatus").textContent = "저장 중…";
  try {
    const res = await fetch("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updatePayload)
    });
    if (res.ok) {
      $("saveStatus").textContent = "✓ 저장됨";
    } else {
      $("saveStatus").textContent = "저장 실패: " + (await res.text());
    }
  } catch (err) {
    $("saveStatus").textContent = "네트워크 오류: " + err;
  }
}

$("txtNote").onchange = () => {
  const meta = sheets.find(s => s.id === curSheetId);
  const noteField = meta.type === "g0" ? "note" : "human_note";
  saveField(noteField, $("txtNote").value);
};

$("chkHl").onchange = renderItem;
$("btnPrev").onclick = () => { if (curIdx > 0) { curIdx--; renderItem(); } };
$("btnNext").onclick = () => { if (curIdx < curRows.length - 1) { curIdx++; renderItem(); } };
$("btnNextEmpty").onclick = () => {
  const meta = sheets.find(s => s.id === curSheetId);
  const nextEmpty = curRows.findIndex((r, idx) => idx > curIdx && !isRowDone(r, meta));
  if (nextEmpty >= 0) {
    curIdx = nextEmpty;
    renderItem();
  } else {
    const anyEmpty = curRows.findIndex(r => !isRowDone(r, meta));
    if (anyEmpty >= 0) {
      curIdx = anyEmpty;
      renderItem();
    } else {
      alert("모든 항목의 검토가 완료되었습니다.");
    }
  }
};

document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
  if (e.key === "ArrowRight") $("btnNext").click();
  if (e.key === "ArrowLeft") $("btnPrev").click();
});

window.onload = init;
</script>
</body>
</html>
"""


BLIND_PREFIXES = ("judge_", "auto_")


def make_unified_handler(lock: threading.Lock, blind: bool = False):
    class UnifiedHandler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/sheets":
                with lock:
                    meta_list = []
                    for s in SHEETS:
                        _, rows = load_sheet_rows(s["path"])
                        total = len(rows)
                        if s["type"] == "g0":
                            done = sum(1 for r in rows if all(r.get(fl) in ("1", "0") for fl in s["flags"]))
                        elif s["type"] == "fcr":
                            done = sum(1 for r in rows if r.get("human_is_conflict") in ("1", "0"))
                        elif s["type"] == "csr_d":
                            done = sum(1 for r in rows if r.get("human_csr") in ("1", "0") and r.get("human_d") in ("1", "0"))
                        elif s["type"] == "csr_only":
                            done = sum(1 for r in rows if r.get("human_csr") in ("1", "0"))
                        else:
                            done = 0
                        meta_list.append({
                            "id": s["id"],
                            "title": s["title"],
                            "type": s["type"],
                            "key_col": s["key_col"],
                            "total": total,
                            "done": done
                        })
                self._send(200, json.dumps(meta_list, ensure_ascii=False).encode("utf-8"), "application/json")
            elif self.path.startswith("/api/sheet?"):
                sheet_id = self.path.split("id=")[-1].split("&")[0]
                sheet_cfg = next((s for s in SHEETS if s["id"] == sheet_id), None)
                if not sheet_cfg:
                    self._send(404, b"sheet not found", "text/plain")
                    return
                with lock:
                    fields, rows = load_sheet_rows(sheet_cfg["path"])
                if blind:  # judge 판정·근거를 보기 전에 라벨하도록 가림 (파일은 그대로)
                    fields = [f for f in fields if not f.startswith(BLIND_PREFIXES)]
                    rows = [{k: v for k, v in r.items() if not k.startswith(BLIND_PREFIXES)} for r in rows]
                self._send(200, json.dumps({"fields": fields, "rows": rows}, ensure_ascii=False).encode("utf-8"), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/save":
                return self._send(404, b"not found", "text/plain")
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                sheet_id = payload.get("sheet_id")
                key_val = payload.get("key_val")
                updates = payload.get("updates", {})

                sheet_cfg = next((s for s in SHEETS if s["id"] == sheet_id), None)
                if not sheet_cfg:
                    return self._send(400, b"unknown sheet_id", "text/plain")

                with lock:
                    fields, rows = load_sheet_rows(sheet_cfg["path"])
                    key_col = sheet_cfg["key_col"]
                    target = next((r for r in rows if r.get(key_col) == key_val), None)
                    if not target:
                        return self._send(400, f"item {key_val} not found".encode("utf-8"), "text/plain")

                    for k, v in updates.items():
                        target[k] = str(v)
                    write_sheet_rows(sheet_cfg["path"], fields, rows)

                self._send(200, b"ok", "text/plain")
            except Exception as e:
                self._send(500, str(e).encode("utf-8"), "text/plain")

        def log_message(self, *args):
            pass

    return UnifiedHandler


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8768)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--blind", action="store_true", help="judge 판정·근거(judge_*, auto_*) 컬럼을 화면에 보내지 않음")
    a = ap.parse_args(argv)

    lock = threading.Lock()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_unified_handler(lock, blind=a.blind))
    url = f"http://127.0.0.1:{a.port}"
    print(f"=== Unified Audit Server running at {url} ===" + (" [BLIND: judge 컬럼 숨김]" if a.blind else ""))
    print("Supports Sheets 1 ~ 6 with atomic auto-save. Press Ctrl+C to stop.")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nUnified Audit Server stopped.")


if __name__ == "__main__":
    main()
