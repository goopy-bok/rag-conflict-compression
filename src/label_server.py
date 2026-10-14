"""
Local labelling page for the G0 author labels (amendment 2.2-d; the author's step, run last).

  uv run python src/label_server.py              # http://127.0.0.1:8765 (opens the browser)
  uv run python src/label_server.py --no-browser --port 8766

Reads results/g0_local/author_labels.csv (written by `validate_judge.py label-sheet`), shows one item at a time
(question, candidates A and B, compressed context) and saves every click back to the same CSV (1/0 per flag,
UTF-8 with BOM, same columns), so `scripts/run_lite.sh g0 stats` can run as soon as all items are labelled.
Bound to 127.0.0.1 only. It never opens label_key.json (conditions stay hidden; the order is already shuffled).
Standard library only.
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
CSV_PATH = ROOT_DIR / "results" / "g0_local" / "author_labels.csv"
KO_PATH = ROOT_DIR / "results" / "g0_local" / "author_labels_ko.json"
FLAGS = ("a_supported", "a_distorted", "b_supported", "b_distorted")  # same as validate_judge.FLAGS
VALID = {"1", "0", ""}


def load_rows(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    """Atomic write (temp file + replace) so an interrupted save never leaves a half-written CSV."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def apply_update(rows: list[dict], update: dict) -> dict:
    """Validated in-place update of one row: flags must be '1', '0' or ''; note is free text."""
    key = update.get("item_key")
    row = next((r for r in rows if r["item_key"] == key), None)
    if row is None:
        raise KeyError(f"unknown item_key {key!r}")
    for fl in FLAGS:
        if fl in update:
            v = str(update[fl])
            if v not in VALID:
                raise ValueError(f"{fl} must be 1, 0 or empty, got {v!r}")
            row[fl] = v
    if "note" in update:
        row["note"] = str(update["note"])
    return row


def labelled(rows: list[dict]) -> int:
    return sum(all(r.get(fl) in ("1", "0") for fl in FLAGS) for r in rows)


def _load_ko_translations() -> dict:
    if KO_PATH.exists():
        try:
            with open(KO_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G0 저자 라벨</title><style>
:root{--bg:#fafaf9;--card:#fff;--ink:#1c1917;--mut:#57534e;--line:#e7e5e4;--a:#1d4ed8;--b:#b45309;--ok:#15803d;--sub-bg:#f5f5f4}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--card:#292524;--ink:#f5f5f4;--mut:#a8a29e;--line:#44403c;--a:#93c5fd;--b:#fcd34d;--ok:#86efac;--sub-bg:#1f1d1b}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,system-ui,sans-serif}
main{max-width:920px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0}
h1{font-size:18px;margin:4px 0} .mut{color:var(--mut);font-size:13px}
.ctx{white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;font-size:13.5px;max-height:46vh;overflow:auto;background:var(--sub-bg);padding:10px 12px;border-radius:8px;border:1px solid var(--line)}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:6px 0}
.cand{font-weight:600} .A{color:var(--a)} .B{color:var(--b)}
.flag{display:flex;gap:6px;align-items:center;min-width:210px}
button{font:inherit;padding:6px 12px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button.on{background:var(--ink);color:var(--card);font-weight:600}
button.mode-btn{padding:3px 10px;font-size:12.5px}
mark.ma{background:#bfdbfe;color:#000;border-radius:2px;padding:0 2px}
mark.mb{background:#fde68a;color:#000;border-radius:2px;padding:0 2px}
textarea{width:100%;box-sizing:border-box;font:inherit;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);padding:6px 8px}
.done{color:var(--ok);font-weight:600} details summary{cursor:pointer}
.orig-sub{font-size:12.5px;color:var(--mut);margin-top:2px;display:block}
</style></head><body><main>
<h1>G0 저자 라벨 <span id="prog" class="mut"></span></h1>
<div class="mut">각 버튼을 누르면 즉시 저장됩니다. 조건(C0/C1)은 가려져 있고, <b>label_key.json은 열지 마세요</b>. ←/→ 이전/다음.</div>
<details class="card"><summary><b>판정 기준</b> (judge 루브릭과 같음)</summary>
<p><b>supported = 1</b>: 압축문(CONTEXT)에 후보의 값·개체가 명시적으로 있고, 질문에 대한 답으로 서술되어 압축문만 읽고 그 후보에 도달할 수 있다. 추론·짐작 금지. 다른 후보가 함께 있거나 출처가 엇갈린다고 써 있어도 supported다.</p>
<p><b>distorted = 1</b>은 세 경우뿐: (a) 값 변경(다른 수치·날짜·개체), (b) 잘못된 귀속(값은 있으나 다른 출처·인물·대상에 붙음), (c) 병합(두 후보가 한 진술로 섞여 따로 복원 불가). 다른 후보와 충돌, 충돌 언급, 후보 부재(supported=0, distorted=0), 사실이 틀렸다는 판단은 distorted가 아니다.</p>
<p class="mut">경계 예: 반올림·단위 생략·날짜 형식 변경은 값 유지(supported=1, distorted=0). "1919 or 1918"처럼 두 값이 따로 나열되면 둘 다 supported=1.</p></details>
<div class="row" style="justify-content:space-between;">
  <div class="row">
    <label><input type="checkbox" id="hl" checked> 후보 값 강조</label>
    <label style="margin-left:10px;"><input type="checkbox" id="chkKo" checked> 한글 번역 보기</label>
  </div>
  <button id="nextEmpty">미완료 항목으로</button>
</div>
<section class="card">
  <div class="mut" id="pos"></div>
  <p><b>질문</b>: <span id="q"></span><span id="q_en" class="orig-sub"></span></p>
  <p>
    <span class="cand A">후보 A</span>: <span id="a"></span> <span id="a_en" class="orig-sub" style="display:inline;"></span><br>
    <span class="cand B">후보 B</span>: <span id="b"></span> <span id="b_en" class="orig-sub" style="display:inline;"></span>
  </p>
  <div class="row" style="justify-content:space-between; margin-bottom:4px;">
    <div class="mut" style="font-weight:600;">압축문 (CONTEXT)</div>
    <div class="row" id="viewModeGroup" style="margin:0;">
      <button class="mode-btn on" id="btnModeKo">한글 번역</button>
      <button class="mode-btn" id="btnModeEn">영문 원문</button>
      <button class="mode-btn" id="btnModeBoth">동시 비교</button>
    </div>
  </div>
  <div id="ctxContainer">
    <div class="ctx" id="ctxMain"></div>
    <div class="ctx" id="ctxSub" style="margin-top:8px; border-top:1px dashed var(--line); display:none;"></div>
  </div>
</section>
<section class="card" id="flags"></section>
<section class="card"><div class="mut">메모(선택)</div><textarea id="note" rows="2" placeholder="경계선 사례, 애매한 이유 메모"></textarea></section>
<div class="row"><button id="prev">← 이전 [←]</button><button id="next">다음 [→] →</button><span id="saved" class="mut"></span></div>
</main><script>
const FLAGS=[["a_supported","A supported (지지)","A"],["a_distorted","A distorted (왜곡)","A"],["b_supported","B supported (지지)","B"],["b_distorted","B distorted (왜곡)","B"]];
let items=[],i=0,ctxMode="ko";const $=id=>document.getElementById(id);
const done=r=>FLAGS.every(([f])=>r[f]==="1"||r[f]==="0");

function markInEl(el, text, terms) {
  el.textContent = "";
  if (!terms || terms.length === 0 || !$("hl").checked) {
    el.textContent = text || "";
    return;
  }
  const clean = terms.filter(t => t && t.trim().length > 1).map((t, idx) => [t.toLowerCase(), idx === 0 ? "ma" : "mb"]);
  let low = (text || "").toLowerCase(), p = 0;
  while (p < text.length) {
    let best = null;
    for (const [t, cls] of clean) {
      const pos = low.indexOf(t, p);
      if (pos >= 0 && (!best || pos < best[0])) best = [pos, t.length, cls];
    }
    if (!best) {
      el.append(text.slice(p));
      break;
    }
    el.append(text.slice(p, best[0]));
    const m = document.createElement("mark");
    m.className = best[2];
    m.textContent = text.slice(best[0], best[0] + best[1]);
    el.append(m);
    p = best[0] + best[1];
  }
}

function render(){
  if (!items.length) return;
  const r = items[i];
  const useKo = $("chkKo").checked;
  $("pos").textContent = `${i+1} / ${items.length}  ·  ${r.item_key}` + (done(r) ? "  ·  ✓ 완료" : "  ·  미완료");

  // Question & Answers
  if (useKo && r.question_ko) {
    $("q").textContent = r.question_ko;
    $("q_en").textContent = "원문: " + r.question;
    $("q_en").style.display = "block";
  } else {
    $("q").textContent = r.question;
    $("q_en").style.display = "none";
  }

  if (useKo && r.answer_a_ko) {
    $("a").textContent = r.answer_a_ko;
    $("a_en").textContent = `(원문: ${r.answer_a})`;
  } else {
    $("a").textContent = r.answer_a;
    $("a_en").textContent = "";
  }

  if (useKo && r.answer_b_ko) {
    $("b").textContent = r.answer_b_ko;
    $("b_en").textContent = `(원문: ${r.answer_b})`;
  } else {
    $("b").textContent = r.answer_b;
    $("b_en").textContent = "";
  }

  // Highlight terms
  const termsEn = [r.answer_a, r.answer_b].filter(Boolean);
  const termsKo = [r.answer_a_ko || r.answer_a, r.answer_b_ko || r.answer_b].filter(Boolean);

  const mainEl = $("ctxMain");
  const subEl = $("ctxSub");

  if (ctxMode === "ko" && r.compressed_ko && useKo) {
    mainEl.style.display = "block";
    subEl.style.display = "none";
    markInEl(mainEl, r.compressed_ko, termsKo);
  } else if (ctxMode === "en" || !r.compressed_ko || !useKo) {
    mainEl.style.display = "block";
    subEl.style.display = "none";
    markInEl(mainEl, r.compressed, termsEn);
  } else if (ctxMode === "both") {
    mainEl.style.display = "block";
    subEl.style.display = "block";
    markInEl(mainEl, "【한글 번역】\n" + (r.compressed_ko || ""), termsKo);
    markInEl(subEl, "【영문 원문】\n" + r.compressed, termsEn);
  }

  $("note").value = r.note || "";
  const box = $("flags");
  box.textContent = "";
  for (const [f, label, side] of FLAGS) {
    const d = document.createElement("div");
    d.className = "flag";
    const s = document.createElement("span");
    s.className = "cand " + side;
    s.textContent = label;
    d.append(s);
    for (const v of ["1", "0"]) {
      const bt = document.createElement("button");
      bt.textContent = v === "1" ? "1 (Yes)" : "0 (No)";
      if (r[f] === v) bt.className = "on";
      bt.onclick = () => save({[f]: v});
      d.append(bt);
    }
    box.append(d);
  }

  const n = items.filter(done).length;
  $("prog").innerHTML = `· 완료 ${n}/${items.length}` + (n === items.length ? ' <span class="done">전부 완료: 터미널에서 scripts/run_lite.sh g0 stats</span>' : "");
}

async function save(upd){
  const r = items[i];
  Object.assign(r, upd);
  render();
  $("saved").textContent = "저장 중…";
  const res = await fetch("/save", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({item_key: r.item_key, ...upd})
  });
  $("saved").textContent = res.ok ? "✓ 저장됨" : "저장 실패: " + (await res.text());
}

$("note").onchange = () => save({note: $("note").value});
$("hl").onchange = render;
$("chkKo").onchange = render;

$("btnModeKo").onclick = () => { ctxMode = "ko"; updateModeBtns(); render(); };
$("btnModeEn").onclick = () => { ctxMode = "en"; updateModeBtns(); render(); };
$("btnModeBoth").onclick = () => { ctxMode = "both"; updateModeBtns(); render(); };

function updateModeBtns() {
  $("btnModeKo").className = "mode-btn" + (ctxMode === "ko" ? " on" : "");
  $("btnModeEn").className = "mode-btn" + (ctxMode === "en" ? " on" : "");
  $("btnModeBoth").className = "mode-btn" + (ctxMode === "both" ? " on" : "");
}

$("prev").onclick = () => { if (i > 0) { i--; render(); } };
$("next").onclick = () => { if (i < items.length - 1) { i++; render(); } };
$("nextEmpty").onclick = () => {
  const k = items.findIndex((r, idx) => idx > i && !done(r));
  if (k >= 0) { i = k; render(); }
  else {
    const any = items.findIndex(r => !done(r));
    if (any >= 0) { i = any; render(); }
    else { alert("50건의 모든 항목이 라벨링 완료되었습니다!"); }
  }
};

document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
  if (e.key === "ArrowRight") $("next").click();
  if (e.key === "ArrowLeft") $("prev").click();
});

fetch("/items").then(r => r.json()).then(d => {
  items = d;
  const k = items.findIndex(r => !done(r));
  i = k >= 0 ? k : 0;
  render();
});
</script></body></html>"""


def make_handler(path: Path, fields: list[str], rows: list[dict], lock: threading.Lock):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/items":
                ko_map = _load_ko_translations()
                with lock:
                    data = []
                    for r in rows:
                        item_dict = {k: r.get(k, "") for k in ("item_key", "question", "answer_a", "answer_b", "compressed",
                                                               *FLAGS, "note")}
                        k = r.get("item_key", "")
                        ko_item = ko_map.get(k, {})
                        item_dict["question_ko"] = ko_item.get("question_ko", "")
                        item_dict["answer_a_ko"] = ko_item.get("answer_a_ko", "")
                        item_dict["answer_b_ko"] = ko_item.get("answer_b_ko", "")
                        item_dict["compressed_ko"] = ko_item.get("compressed_ko", "")
                        data.append(item_dict)
                self._send(200, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/save":
                return self._send(404, b"not found", "text/plain")
            try:
                update = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                with lock:
                    apply_update(rows, update)
                    write_rows(path, fields, rows)
                self._send(200, b"ok", "text/plain")
            except (KeyError, ValueError) as e:
                self._send(400, str(e).encode("utf-8"), "text/plain")

        def log_message(self, *args):  # quiet
            pass
    return Handler


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    if not a.csv.exists():
        raise SystemExit(f"{a.csv} not found: run `scripts/run_lite.sh g0auto` first (it writes the label sheet)")
    fields, rows = load_rows(a.csv)
    missing = [c for c in ("item_key", "question", "compressed", "answer_a", "answer_b", *FLAGS) if c not in fields]
    if missing:
        raise SystemExit(f"{a.csv} lacks columns {missing}")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(a.csv, fields, rows, threading.Lock()))
    url = f"http://127.0.0.1:{a.port}"
    print(f"{labelled(rows)}/{len(rows)} labelled. Open {url}  (Ctrl+C to stop; every click is saved to {a.csv})")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped. {labelled(rows)}/{len(rows)} labelled. When all are done: scripts/run_lite.sh g0 stats")


if __name__ == "__main__":
    main()
