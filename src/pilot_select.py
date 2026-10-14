"""
Main-compressor selection pilot (preregistration 2026-10-09, section 2.2; rules implemented as written).

  uv run python src/pilot_select.py                                  # plan only: call count, time estimate
  uv run python src/pilot_select.py --mode execute                    # 5 candidates x C0,C1 x dev[:50], r=0.3
  uv run python src/pilot_select.py --models qwen2.5:7b --limit 2 --mode execute   # smoke
  uv run python src/pilot_select.py --mode mock                       # offline, under results/mock/pilot_select/
  uv run python src/pilot_select.py --h2h --mode execute              # amendment 2.2-b: phi3.5 vs gemma4:e4b, dev[:100],
                                                                      # M2' + output normalization N, results/pilot_h2h/

Compression only (no judge, no downstream). Same PromptCompressor as the harness (1-retry budget rule),
temperature 0, seed 0 (configs/models.yaml ollama.seed), num_ctx 8192. Data: run.load_split("dev")
(few-shot ids removed, file order), first 50.
Outputs: results/pilot_select/{model}_{C0|C1}.jsonl (':' in the model id written as '-'), one line per
instance, appended as it completes; a rerun skips ids already in the file. Report:
reports/2026-10-09_compressor_selection.md.

Metrics per model over its 100 outputs (C0 and C1 pooled):
  M1 budget compliance  : out_tokens (cl100k, after the retry) <= budget
  M2 format compliance  : non-empty, no refusal/clarification prefix, no preamble prefix (lists below, fixed)
  M3 context retention  : out_tokens >= 0.1 * budget
  processing time       : per output, sum of Ollama total_duration over its calls (incl. retry; wall clock
                          when Ollama reports none, e.g. mock); the median per model
Selection: (1) keep models with M1 >= 0.95 and M2 >= 0.95; (2) among them the highest M3; models within
0.02 of the highest M3 -> shortest median processing time; (3) if none passes (1), the highest (M1+M2)/2,
reported as such.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from budget import compute_budget  # noqa: E402
from configure_roles import assign_roles  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from run import build_compressor, load_split  # noqa: E402

CANDIDATES = ["gemma2:9b", "llama3.1:8b", "phi3.5", "qwen2.5:7b", "qwen2.5:14b"]  # 2.2 order
CONDITIONS = ["C0", "C1"]
R = 0.3
N_DEV = 50
SEED = 0
SECONDS_PER_CALL = (10, 22)  # measured 2026-10-09: 7-9B 10-18 s, 14B 22 s per compression
M1_MIN, M2_MIN, M3_FRAC, M3_TIE = 0.95, 0.95, 0.10, 0.02
EPS = 1e-9  # metrics are k/100; guards float error in the >= and "within 0.02" comparisons

# Amendment 2.2-b (head-to-head, `--h2h`): incumbent phi3.5 vs challenger gemma4:e4b. Values fixed before the run;
# the original 5-candidate rule above stays reproducible (format_flags / select_compressor are unchanged).
H2H_CANDIDATES = ["phi3.5", "gemma4:e4b"]  # incumbent first: exact-tie fallback
H2H_N, H2H_N_MAX = 100, 100  # dev instances per condition
G_MAX, M3_TIE_H2H = 0.05, 0.10  # condition gap |M2'(C0)-M2'(C1)| gate; M3 differences within 0.10 count as a tie
H2H_OUT_DIR = ROOT_DIR / "results" / "pilot_h2h"
H2H_MOCK_DIR = ROOT_DIR / "results" / "mock" / "pilot_h2h"
H2H_REPORT = ROOT_DIR / "reports" / "2026-10-09_compressor_h2h.md"
PREREG_AMEND = "paper/rag/연구계획서/미팅준비/1005/2026-10-09_압축기 선정 사전등록 개정 2.2-b (phi3.5 대 gemma4-e4b).md"

# M2 lists (fixed before the pilot). Matched case-insensitively at the start of the output, after
# stripping whitespace and leading markdown/quote characters, as whole words.
PREAMBLE_PREFIXES = (
    "here is", "here's", "here are", "here you go", "sure", "certainly", "of course", "okay", "absolutely",
    "below is", "below are", "the following is", "the following are", "as requested",
    "i have compressed", "i've compressed", "this is the compressed",
)
REFUSAL_PREFIXES = (
    "i'm sorry", "i am sorry", "sorry", "i apologize", "i cannot", "i can't", "i can not", "i am unable",
    "i'm unable", "i am not able", "i'm not able", "i won't", "i will not", "as an ai", "unfortunately, i",
    "could you", "can you", "would you", "please provide", "please clarify", "please specify",
    "do you want", "what do you mean", "i need more",
)

OUT_DIR = ROOT_DIR / "results" / "pilot_select"
MOCK_DIR = ROOT_DIR / "results" / "mock" / "pilot_select"
REPORT = ROOT_DIR / "reports" / "2026-10-09_compressor_selection.md"
PREREG = "paper/rag/연구계획서/미팅준비/1005/2026-10-09_G1 로컬 모델 전환 변경안 및 압축기 선정 사전등록.md"


def _prefix_re(phrases: tuple[str, ...]) -> re.Pattern:
    return re.compile(r"^(" + "|".join(re.escape(p) for p in phrases) + r")(?!\w)", re.I)


_PREAMBLE, _REFUSAL = _prefix_re(PREAMBLE_PREFIXES), _prefix_re(REFUSAL_PREFIXES)


def format_flags(text: str) -> dict:
    head = text.replace("’", "'").strip().lstrip("*_#>\"'` \t\n").lower()
    refusal, preamble = _REFUSAL.match(head), _PREAMBLE.match(head)
    flags = {"empty": not text.strip(), "refusal": refusal.group(1) if refusal else None,
             "preamble": preamble.group(1) if preamble else None}
    flags["ok"] = not (flags["empty"] or flags["refusal"] or flags["preamble"])
    return flags


_QA_QUESTION = re.compile(r"^\s*(?:\*\*)?question(?:\*\*)?\s*[:：]", re.I)
_QA_ANSWER = re.compile(r"^\s*(?:\*\*)?answer(?:\*\*)?\s*[:：]", re.I | re.M)
# C2's prescribed no-conflict line is "ANSWER: [dX] <value>" (prompts/c2_best.txt rule 4); the pre-test gate
# (amendment 2.2-c) accepts a document-tagged ANSWER line, an untagged "Answer: ..." stays QA mode.
_QA_ANSWER_UNTAGGED = re.compile(r"^\s*(?:\*\*)?answer(?:\*\*)?\s*[:：](?!\s*\[d\d+\])", re.I | re.M)


def _json_left(text: str) -> bool:
    s = text.strip()
    if s.startswith("{"):
        return True
    if s.startswith("["):  # '[d3] ...' is plain text, only a parseable array counts
        try:
            return isinstance(json.loads(s), (list, dict))
        except ValueError:
            return False
    return False


def format_flags_v2(text: str, tagged_answer_ok: bool = False) -> dict:
    """M2' (amendment 2.2-b), applied to the already normalized text: format_flags plus
    qa_mode (starts with 'Question:' or has a line starting with 'Answer:') and json_left (JSON that N could not
    unwrap). ok = none of empty, refusal, preamble, qa_mode, json_left.
    tagged_answer_ok=True (pre-test gate, amendment 2.2-c): a line 'ANSWER: [dX] ...' is not QA mode."""
    flags = format_flags(text)
    answer = _QA_ANSWER_UNTAGGED if tagged_answer_ok else _QA_ANSWER
    flags["qa_mode"] = bool(_QA_QUESTION.match(text) or answer.search(text))
    flags["json_left"] = _json_left(text)
    flags["ok"] = flags["ok"] and not (flags["qa_mode"] or flags["json_left"])
    return flags


def out_file(out_dir: Path, model: str, condition: str) -> Path:
    return Path(out_dir) / f"{model.replace(':', '-')}_{condition}.jsonl"


def _read_done(path: Path) -> list[dict]:
    """Complete records of an earlier (possibly interrupted) run; a cut-off last line is dropped."""
    if not path.exists():
        return []
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return recs


def run_model_condition(model: str, condition: str, instances: list[dict], client, out_dir: Path,
                        flags_fn=format_flags) -> list[dict]:
    path = out_file(out_dir, model, condition)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = _read_done(path)
    done = {r["id"] for r in old}
    todo = [inst for inst in instances if inst["id"] not in done]
    print(f"[{model} {condition}] {len(done)} done, {len(todo)} to run")
    compressor = build_compressor(condition, SEED, client, model)
    new = []
    with path.open("w", encoding="utf-8") as f:
        for rec in old:  # rewrite without a cut-off line
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for i, inst in enumerate(todo, start=1):
            budget = compute_budget(inst["total_tokens"], R)
            t0 = time.time()
            comp = compressor.compress(inst, budget)
            wall = time.time() - t0
            stats = comp["meta"]["ollama"]
            durations = [s["total_duration_ns"] for s in stats if s.get("total_duration_ns") is not None]
            rec = {"id": inst["id"], "model": model, "condition": condition, "r": R, "seed": SEED,
                   "budget": budget, "out_tokens": comp["tokens"], "over_budget": comp["over_budget"],
                   "retried": comp["retried"], "text": comp["text"], "format": flags_fn(comp["text"]),
                   "raw_text": comp["raw_text"] if comp.get("raw_text") != comp["text"] else None,
                   "seconds": sum(durations) / 1e9 if durations else wall, "wall_seconds": wall,
                   "cached": bool(comp["meta"]["call1_cached"]), "model_digest": comp["meta"]["model_digest"],
                   "ollama": stats, "mode": client.mode,
                   "prompt_sha256": comp["meta"]["instruction_sha256"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            new.append(rec)
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")
    by_id = {r["id"]: r for r in old + new}
    return [by_id[inst["id"]] for inst in instances if inst["id"] in by_id]


def compute_metrics(records: list[dict], flags_fn=format_flags) -> dict:
    n = len(records)
    if not n:
        return {"n": 0}
    fl = [flags_fn(r["text"]) for r in records]
    m1 = sum(r["out_tokens"] <= r["budget"] for r in records) / n
    m2 = sum(f["ok"] for f in fl) / n
    m3 = sum(r["out_tokens"] >= M3_FRAC * r["budget"] for r in records) / n
    stats = [s for r in records for s in r.get("ollama") or []]
    pe = [s["prompt_eval_count"] for s in stats if s.get("prompt_eval_count") is not None]
    return {
        "n": n, "M1": m1, "M2": m2, "M3": m3,
        "median_s": statistics.median(r["seconds"] for r in records),
        "retried": sum(r["retried"] for r in records),
        "empty": sum(f["empty"] for f in fl),
        "refusal": sum(bool(f["refusal"]) for f in fl),
        "preamble": sum(bool(f["preamble"]) for f in fl),
        "qa_mode": sum(bool(f.get("qa_mode")) for f in fl),
        "json_left": sum(bool(f.get("json_left")) for f in fl),
        "normalized": sum(bool(r.get("raw_text")) for r in records),  # outputs that N changed
        "util_median": statistics.median(r["out_tokens"] / r["budget"] for r in records if r["budget"]),
        "thinking_chars": sum(s.get("thinking_chars") or 0 for s in stats),
        "cached": sum(bool(r.get("cached")) for r in records),
        "max_prompt_eval_count": max(pe) if pe else None,
        "num_ctx": max((s["num_ctx"] for s in stats), default=None),
        "done_reason_length": sum(s.get("done_reason") == "length" for s in stats),
        "digests": sorted({r["model_digest"] for r in records if r.get("model_digest")}),
    }


def select_compressor(metrics: dict[str, dict], order: list[str] = CANDIDATES) -> dict:
    models = [m for m in order if m in metrics] + [m for m in metrics if m not in order]
    eligible = [m for m in models if metrics[m]["M1"] >= M1_MIN - EPS and metrics[m]["M2"] >= M2_MIN - EPS]
    notes = []
    if eligible:
        top = max(metrics[m]["M3"] for m in eligible)
        tie = [m for m in eligible if top - metrics[m]["M3"] <= M3_TIE + EPS]
        fastest = min(metrics[m]["median_s"] for m in tie)
        best = [m for m in tie if metrics[m]["median_s"] == fastest]
        if len(best) > 1:
            notes.append(f"동일한 처리 시간 중앙값 {best}: 사전 등록에 규정이 없어 후보 목록 순서로 결정함")
        return {"rule": 2, "selected": best[0], "eligible": eligible, "tie_set": tie, "notes": notes}
    score = {m: (metrics[m]["M1"] + metrics[m]["M2"]) / 2 for m in models}
    top = max(score.values())
    best = [m for m in models if top - score[m] <= EPS]
    notes.append("규칙 1을 통과한 모델이 없어 규칙 3((M1+M2)/2 최대)으로 선정함")
    if len(best) > 1:
        notes.append(f"(M1+M2)/2 동률 {best}: 사전 등록에 규정이 없어 후보 목록 순서로 결정함")
    return {"rule": 3, "selected": best[0], "eligible": [], "tie_set": best, "score": score, "notes": notes}


def _f(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def write_report(path: Path, by_model: dict[str, list[dict]], *, n_instances: int, mode: str,
                 complete: bool, ollama_cfg: dict) -> dict:
    metrics = {m: compute_metrics(recs) for m, recs in by_model.items()}
    sel = select_compressor(metrics)
    roles = assign_roles(sel["selected"])
    L = ["# 주 압축기 선정 파일럿 결과", "",
         f"작성: {time.strftime('%Y-%m-%d %H:%M')} · 모드: {mode} · 근거: `{PREREG}` 2.2절", ""]
    if not complete:
        L += ["> **부분 실행.** 사전 등록된 전체 범위(후보 5개 × C0·C1 × dev 50건)가 아니므로 아래 선정 결과는 "
              "사전 등록 판정이 아니다.", ""]
    if mode != "execute":
        L += [f"> 모드 {mode}: 실제 모델 출력이 아니다.", ""]
    L += ["## 설정", "",
          f"- 데이터: dev 앞 {n_instances}건(few-shot 제외 후 파일 순서), 조건 {', '.join(CONDITIONS)}, r = {R}",
          f"- temperature 0, seed {ollama_cfg.get('seed')}, num_ctx {ollama_cfg.get('num_ctx')}, 재시도 1회 규칙"
          " (PromptCompressor)", "- judge·하류 호출 없음. 토큰 수는 cl100k.",
          "- 처리 시간: 출력 1건당 Ollama total_duration 합(재시도 포함, 모델 적재 시간 포함). Ollama 값이 없으면 벽시계 시간.",
          "", "## 지표 정의와 고정 목록", "",
          f"- M1 예산 준수율: 재시도 후 출력 토큰 ≤ 예산인 비율",
          f"- M2 형식 준수율: 출력이 비어 있지 않고, 아래 거절·반문 문구로 시작하지 않고, 아래 서두 문구로 시작하지 "
          "않는 비율 (대소문자 무시, 앞쪽 공백·마크다운·따옴표 제거 후 단어 단위 접두 일치)",
          f"- M3 문맥 보존 행동률: 출력 토큰 ≥ 예산 × {M3_FRAC}인 비율",
          "- 서두 문구 목록: " + ", ".join(f'"{p}"' for p in PREAMBLE_PREFIXES),
          "- 거절·반문 문구 목록: " + ", ".join(f'"{p}"' for p in REFUSAL_PREFIXES), "",
          "## 모델별 결과 (C0·C1 합산)", "",
          "| 모델 | n | M1 | M2 | M3 | (M1+M2)/2 | 처리 시간 중앙값(s) | 재시도 | 빈 출력 | 거절·반문 | 서두 | 캐시 적중 "
          "| 최대 prompt_eval_count / num_ctx | num_predict 도달 | 다이제스트 |",
          "| --- | " + "--- | " * 14]
    for m, x in metrics.items():
        if not x["n"]:
            L.append(f"| `{m}` | 0 |" + " - |" * 13)
            continue
        L.append(f"| `{m}` | {x['n']} | {_f(x['M1'])} | {_f(x['M2'])} | {_f(x['M3'])} | "
                 f"{_f((x['M1'] + x['M2']) / 2)} | {_f(x['median_s'], 1)} | {x['retried']} | {x['empty']} | "
                 f"{x['refusal']} | {x['preamble']} | {x['cached']} | {_f(x['max_prompt_eval_count'])} / "
                 f"{_f(x['num_ctx'])} | {x['done_reason_length']} | {', '.join(d[:12] for d in x['digests']) or '-'} |")
    L += ["", "## 조건별 내역", "", "| 모델 | 조건 | n | M1 | M2 | M3 | 처리 시간 중앙값(s) |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for m, recs in by_model.items():
        for c in CONDITIONS:
            x = compute_metrics([r for r in recs if r["condition"] == c])
            if x["n"]:
                L.append(f"| `{m}` | {c} | {x['n']} | {_f(x['M1'])} | {_f(x['M2'])} | {_f(x['M3'])} | {_f(x['median_s'], 1)} |")
    L += ["", "## 선정", "",
          f"- 규칙 1 통과(M1 ≥ {M1_MIN}, M2 ≥ {M2_MIN}): {', '.join(f'`{m}`' for m in sel['eligible']) or '없음'}",
          f"- 적용 규칙: {sel['rule']}",
          f"- 동률 집합(규칙 2: 최고 M3와의 차이 ≤ {M3_TIE} / 규칙 3: (M1+M2)/2 최대): "
          f"{', '.join(f'`{m}`' for m in sel['tie_set'])}",
          f"- **선정 결과: `{sel['selected']}`**"]
    L += [f"- 비고: {n}" for n in sel["notes"]]
    L += ["", "## 2.3 역할 배정 (선정 결과 적용 시)", "", "| 역할 | 모델 |", "| --- | --- |"]
    L += [f"| {k} | {v} |" for k, v in roles.items()]
    L += ["", f"적용 명령: `uv run python src/configure_roles.py --compressor {sel['selected']}`", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return {"metrics": metrics, "selection": sel, "roles": roles}


def select_h2h(pooled: dict[str, dict], by_cond: dict[str, dict[str, dict]], order: list[str] = H2H_CANDIDATES) -> dict:
    """Amendment 2.2-b decision rule (M2 here is M2', computed with format_flags_v2).
    pooled: model -> compute_metrics over C0+C1; by_cond: model -> {condition: compute_metrics}.
    Gate: M1 >= 0.95, M2' >= 0.95 and G = |M2'(C0) - M2'(C1)| <= 0.05. One model passes -> it. Several pass ->
    the higher M3 if the M3 gap is > 0.10, else the smaller G, else the shorter median time, else the earlier
    model in `order` (the incumbent). None passes -> the higher M2' (incumbent on a tie), reported as outside the gate."""
    models = [m for m in order if m in pooled]
    gap = {m: abs(by_cond[m]["C0"]["M2"] - by_cond[m]["C1"]["M2"]) for m in models}
    gate = {m: pooled[m]["M1"] >= M1_MIN - EPS and pooled[m]["M2"] >= M2_MIN - EPS and gap[m] <= G_MAX + EPS
            for m in models}
    passed = [m for m in models if gate[m]]
    if not passed:
        top = max(pooled[m]["M2"] for m in models)
        pick = next(m for m in models if top - pooled[m]["M2"] <= EPS)
        return {"step": "none", "selected": pick, "passed": [], "gap": gap, "gate": gate,
                "notes": ["어느 모델도 게이트를 통과하지 못함: 사전 등록 규칙대로 M2'가 높은 쪽을 고르되 지도교수와 상의한다"]}
    top = max(pooled[m]["M3"] for m in passed)
    left = [m for m in passed if top - pooled[m]["M3"] <= M3_TIE_H2H + EPS]
    step = "gate" if len(passed) == 1 else "M3"
    if len(left) > 1:
        lo = min(gap[m] for m in left)
        left, step = [m for m in left if gap[m] - lo <= EPS], "G"
        if len(left) > 1:
            fast = min(pooled[m]["median_s"] for m in left)
            left, step = [m for m in left if pooled[m]["median_s"] - fast <= EPS], "speed"
    return {"step": step, "selected": left[0], "passed": passed, "gap": gap, "gate": gate, "notes": []}


_H2H_STEP = {"gate": "게이트 통과 모델이 하나", "M3": "M3 차이 > 0.10", "G": "M3 차이 ≤ 0.10 → 조건 간 격차 G가 작은 쪽",
             "speed": "M3·G 동률 → 처리 시간이 짧은 쪽", "none": "게이트 통과 모델 없음"}


def write_h2h_report(path: Path, by_model: dict[str, list[dict]], *, n_instances: int, mode: str,
                     ollama_cfg: dict, run_info: dict) -> dict:
    pooled = {m: compute_metrics(recs, format_flags_v2) for m, recs in by_model.items()}
    by_cond = {m: {c: compute_metrics([r for r in recs if r["condition"] == c], format_flags_v2)
                   for c in CONDITIONS} for m, recs in by_model.items()}
    sel = select_h2h(pooled, by_cond)
    roles = assign_roles(sel["selected"])
    L = ["# 압축기 1:1 비교 결과 (phi3.5 대 gemma4:e4b)", "",
         f"작성: {time.strftime('%Y-%m-%d %H:%M')} · 모드: {mode} · 근거: `{PREREG_AMEND}`", ""]
    if mode != "execute":
        L += [f"> 모드 {mode}: 실제 모델 출력이 아니다.", ""]
    L += ["## 설정", "",
          f"- 데이터: dev 앞 {n_instances}건(few-shot 제외 후 파일 순서) × 조건 {', '.join(CONDITIONS)}, r = {R}"
          f" → 모델당 {n_instances * len(CONDITIONS)}건",
          f"- temperature 0, seed {ollama_cfg.get('seed')}, num_ctx {ollama_cfg.get('num_ctx')}, 재시도 1회 규칙, "
          "출력 정규화 N 적용(`compressors/llm.py::normalize_output`), judge·하류·CSR 계산 없음",
          f"- Ollama 버전: {run_info.get('ollama_version', '-')} · think 설정: {run_info.get('think', '-')} · "
          f"모델 새로 생성(별도 캐시 `results/pilot_h2h/cache`)", "",
          "## 결과 (C0·C1 합산)", "",
          "| 모델 | n | M1 | M2' | G | M3 | 예산 활용도(중앙값) | 처리 시간 중앙값(s) | 재시도 | 빈 출력 | 거절·반문 | 서두 "
          "| QA 모드 | JSON 잔존 | N 적용 건수 | thinking 문자 수 | 게이트 | 다이제스트 |",
          "| --- | " + "--- | " * 17]
    for m, x in pooled.items():
        L.append(f"| `{m}` | {x['n']} | {_f(x['M1'])} | {_f(x['M2'])} | {_f(sel['gap'][m])} | {_f(x['M3'])} | "
                 f"{_f(x['util_median'])} | {_f(x['median_s'], 1)} | {x['retried']} | {x['empty']} | {x['refusal']} | "
                 f"{x['preamble']} | {x['qa_mode']} | {x['json_left']} | {x['normalized']} | {x['thinking_chars']} | "
                 f"{'통과' if sel['gate'][m] else '미통과'} | {', '.join(d[:12] for d in x['digests']) or '-'} |")
    L += ["", "## 조건별 내역", "", "| 모델 | 조건 | n | M1 | M2' | M3 | QA 모드 | JSON 잔존 | 빈 출력 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m, d in by_cond.items():
        for c, x in d.items():
            L.append(f"| `{m}` | {c} | {x['n']} | {_f(x['M1'])} | {_f(x['M2'])} | {_f(x['M3'])} | {x['qa_mode']} | "
                     f"{x['json_left']} | {x['empty']} |")
    L += ["", "## 선정", "",
          f"- 게이트(M1 ≥ {M1_MIN}, M2' ≥ {M2_MIN}, G ≤ {G_MAX}) 통과: "
          f"{', '.join(f'`{m}`' for m in sel['passed']) or '없음'}",
          f"- 결정 단계: {_H2H_STEP[sel['step']]}",
          f"- **선정 결과: `{sel['selected']}`**"]
    L += [f"- 비고: {n}" for n in sel["notes"]]
    L += ["", "## 2.3 역할 배정 (선정 결과 적용 시)", "", "| 역할 | 모델 |", "| --- | --- |"]
    L += [f"| {k} | {v} |" for k, v in roles.items()]
    L += ["", f"적용 명령: `uv run python src/configure_roles.py --compressor {sel['selected']}`", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return {"pooled": pooled, "by_cond": by_cond, "selection": sel, "roles": roles}


def run_h2h(a, ap) -> None:
    n = a.limit or H2H_N
    if not 1 <= n <= H2H_N_MAX:
        ap.error(f"--limit must be 1..{H2H_N_MAX} with --h2h")
    if a.mode == "plan":
        print_plan(H2H_CANDIDATES, n)
        return
    out_dir, report = (H2H_OUT_DIR, H2H_REPORT) if a.mode == "execute" else (H2H_MOCK_DIR, H2H_MOCK_DIR / H2H_REPORT.name)
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)  # fresh cache: both models are generated on the same Ollama
    client = LLMClient(mode=a.mode, max_cost_usd=0.0 if a.mode == "execute" else None, cache_dir=out_dir / "cache")
    not_local = [m for m in H2H_CANDIDATES if client.provider(m) != "ollama"]
    if not_local:
        raise SystemExit(f"not Ollama models in configs/models.yaml: {not_local}; refusing to run")
    run_info = {"limit": n, "think": {m: client.provider_opts(m).get("think") for m in H2H_CANDIDATES}}
    if a.mode == "execute":
        run_info["digests"] = {m: client.model_digest(m) for m in H2H_CANDIDATES}  # fails fast on a missing model
        try:
            run_info["ollama_version"] = client.ollama_transport("/api/version", None).get("version")
        except Exception as e:  # noqa: BLE001 - recorded, not fatal
            run_info["ollama_version"] = f"unavailable: {e}"
        run_info["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(run_info, ensure_ascii=False))
    instances = load_split("dev")[:n]
    by_model = {m: [rec for c in CONDITIONS
                    for rec in run_model_condition(m, c, instances, client, out_dir, flags_fn=format_flags_v2)]
                for m in H2H_CANDIDATES}
    res = write_h2h_report(report, by_model, n_instances=n, mode=a.mode, ollama_cfg=client.ollama, run_info=run_info)
    print(f"report: {report}\nselected: {res['selection']['selected']} (step {res['selection']['step']})")


def print_plan(models: list[str], n: int) -> None:
    calls = len(models) * len(CONDITIONS) * n
    lo, hi = (calls * s / 3600 for s in SECONDS_PER_CALL)
    print(f"plan: {len(models)} models x {len(CONDITIONS)} conditions x {n} instances = {calls} compressions "
          f"(+ 1 retry per over-budget output, not predictable)")
    print(f"models: {', '.join(models)}")
    print(f"estimated time at {SECONDS_PER_CALL[0]}-{SECONDS_PER_CALL[1]} s per compression: {lo:.1f}-{hi:.1f} h "
          f"(local Ollama, cost 0). Nothing was run; add --mode execute (or mock).")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=CANDIDATES, default=CANDIDATES)
    ap.add_argument("--limit", type=int, default=None,
                    help=f"instances per condition (default {N_DEV}; {H2H_N} with --h2h; max {N_DEV} / {H2H_N_MAX})")
    ap.add_argument("--mode", choices=["plan", "mock", "execute"], default="plan")
    ap.add_argument("--h2h", action="store_true",
                    help=f"amendment 2.2-b head-to-head {' vs '.join(H2H_CANDIDATES)} (M2', output normalization N)")
    a = ap.parse_args(argv)
    if a.h2h:
        return run_h2h(a, ap)
    a.limit = a.limit or N_DEV
    if not 1 <= a.limit <= N_DEV:
        ap.error(f"--limit must be 1..{N_DEV}")
    models = [m for m in CANDIDATES if m in a.models]
    if a.mode == "plan":
        print_plan(models, a.limit)
        return
    client = LLMClient(mode=a.mode, max_cost_usd=0.0 if a.mode == "execute" else None)
    not_local = [m for m in models if client.provider(m) != "ollama"]
    if not_local:
        raise SystemExit(f"not Ollama models in configs/models.yaml: {not_local}; refusing to run")
    if a.mode == "execute":
        for m in models:  # fail fast on a missing model; records digests in results/models_manifest.json
            print(f"{m}: digest {client.model_digest(m)}")
    out_dir, report = (OUT_DIR, REPORT) if a.mode == "execute" else (MOCK_DIR, MOCK_DIR / REPORT.name)
    instances = load_split("dev")[:a.limit]
    by_model = {m: [rec for c in CONDITIONS for rec in run_model_condition(m, c, instances, client, out_dir)]
                for m in models}
    complete = models == CANDIDATES and a.limit == N_DEV
    res = write_report(report, by_model, n_instances=a.limit, mode=a.mode, complete=complete,
                       ollama_cfg=client.ollama)
    print(f"report: {report}\nselected: {res['selection']['selected']} (rule {res['selection']['rule']})"
          + ("" if complete else "  [partial run: not the preregistered decision]"))


if __name__ == "__main__":
    main()
