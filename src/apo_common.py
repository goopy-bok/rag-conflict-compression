"""
Shared pieces of the C3 optimizers (execution guide v2 section 7.1). One definition of
the APO split, the score, and the feedback text, used by GEPA, MIPROv2 and ProTeGi.

  build_apo_split / load_apo_split : dev(297) -> train 197 / val 100, dev_distractor(100) -> train 50 / val 50,
                                     r in {0.3, 0.5} assigned per instance (balanced, deterministic).
  describe_loss                    : natural-language feedback (no CCD type names).
  judge_and_score                  : judge one compressed output -> score + feedback (the metric).
  score_instruction                : compress with an instruction (retry rule) + judge_and_score, per item.
  record_examples / examples_seen  : sha256 of every examples block that reached a compressor call
                                     (read by fewshot_check.py through the APO logs).

Score: conflict instance -> CSR (judge), distractor -> 1 if no false conflict and the single answer is kept
(answer_kept, amendment 2.2-g; before it any output without a false conflict scored 1, so garbage scored 1/3 of
val), over budget after retry -> 0.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from budget import compute_budget  # noqa: E402
from compressors.llm import PromptCompressor, render_examples  # noqa: E402

SPLIT_PATH = ROOT_DIR / "data" / "processed" / "apo_split.json"
R_LEVELS = (0.3, 0.5)
SIZES = {"dev": (197, 100), "dev_distractor": (50, 50)}  # (train, val)
C2_TUNING_LOG = ROOT_DIR / "prompts" / "2026-10-09_c2_tuning_log.md"


# ------------------------------------------------------------------ split

# run.load_split으로 split("dev" 또는 "dev_distractor")의 문항 행 목록을 읽어 돌려준다.
# run.load_split은 기본 dev 경로에서 고정 3-shot의 출처 문항을 빼고 돌려준다(아래 import 줄 주석 참조).
def _load_rows(split: str) -> list[dict]:
    from run import load_split  # dev drops the 3 few-shot source instances
    return load_split(split)


# APO용 train/val 분할을 seed로 한 번만 만들어 data/processed/apo_split.json에 저장하고 그 dict를 돌려준다.
# split마다 문항 id를 정렬한 뒤 섞어 SIZES대로 자르므로 train 247건(dev 197 + dev_distractor 50),
# val 150건(충돌 100 + dev_distractor 50)이 된다. r은 섞은 순서대로 0.3/0.5를 번갈아 배정한다.
# 반환 형식: {"seed", "train": [{"id", "r"}, ...], "val": [...]}.
# 파일이 이미 있으면 다시 쓰지 않고 읽기만 하며, 저장된 seed가 인자와 다르거나 문항 수가 모자라면 ValueError를 낸다.
def build_apo_split(seed: int = 0, out_path: Path = SPLIT_PATH,
                    rows: dict[str, list[dict]] | None = None) -> dict:
    """Create the split once; if out_path exists it is read, never rewritten.
    rows: optional {"dev": [...], "dev_distractor": [...]} (tests)."""
    out_path = Path(out_path)
    if out_path.exists():
        split = json.loads(out_path.read_text(encoding="utf-8"))
        if split.get("seed") != seed:
            raise ValueError(f"{out_path} was built with seed {split.get('seed')}, not {seed}")
        return split
    rows = rows or {name: _load_rows(name) for name in SIZES}
    rng = random.Random(seed)
    split = {"seed": seed, "train": [], "val": []}
    for name, (n_train, n_val) in SIZES.items():
        ids = sorted(r["id"] for r in rows[name])
        if len(ids) < n_train + n_val:
            raise ValueError(f"{name}: {len(ids)} instances < {n_train}+{n_val}")
        rng.shuffle(ids)
        for part, chunk in (("train", ids[:n_train]), ("val", ids[n_train:n_train + n_val])):
            # alternate r after the shuffle -> both levels equally often within every part
            split[part] += [{"id": i, "r": R_LEVELS[k % 2]} for k, i in enumerate(chunk)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split, indent=1), encoding="utf-8")
    return split


# build_apo_split의 id 목록을 실제 문항 dict로 바꿔 {"train": [(문항, r)], "val": [(문항, r)]}를 돌려준다.
# C3-P(apo_protegi)·C3-M(apo_mipro)·C3-G(apo_gepa)가 공통으로 쓰는 입력이며, 분할 파일이 없으면 이 호출에서 처음 생성된다.
def load_apo_split(seed: int = 0, path: Path = SPLIT_PATH,
                   rows: dict[str, list[dict]] | None = None) -> dict[str, list[tuple[dict, float]]]:
    rows = rows or {name: _load_rows(name) for name in SIZES}
    split = build_apo_split(seed, path, rows)
    by_id = {r["id"]: r for name in SIZES for r in rows[name]}
    return {part: [(by_id[e["id"]], float(e["r"])) for e in split[part]] for part in ("train", "val")}


# ------------------------------------------------------------------ feedback and score

# answer_kept용 정규화: 소문자로 바꾸고 구두점을 공백으로 치환한 뒤 관사(a/an/the)를 뺀 단어열을 돌려준다.
def _norm_answer(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


# 정규화한 정답이 정규화한 출력 안에 단어 경계 단위로 통째로 들어 있는지 문자열 비교로만 판정한다(LLM 호출 없음).
# 정답이 비면 False다. 무충돌(dev_distractor) 문항 점수에 쓰이고 pretest_gate.py도 가져다 쓴다.
def answer_kept(answer: str, text: str) -> bool:
    """Lexical: the normalized answer (lowercase, punctuation -> space, articles dropped) is a whole-word span of
    the normalized text. No LLM call, so it adds no unvalidated judge use (amendment 2.2-g)."""
    a = _norm_answer(answer or "")
    return bool(a) and f" {a} " in f" {_norm_answer(text or '')} "


# describe_loss용: role이 일치하는 첫 문서의 doc_id를 찾고, 없으면 conflict_positions[pos] 위치 문서의
# doc_id를, 그것도 없으면 "?"를 돌려준다.
def _doc_of(instance: dict, role: str, pos: int) -> str:
    for d in instance["docs"]:
        if d.get("role") == role:
            return d["doc_id"]
    cp = instance.get("conflict_positions") or []
    return instance["docs"][cp[pos]]["doc_id"] if len(cp) > pos else "?"


# 채점 결과를 최적화기에 줄 자연어 피드백 문장으로 바꾼다. 경우는 셋이다.
# 예산 초과(재시도 후) → 0점 안내 / 무충돌 문항 → 거짓 충돌 생성 또는 단일 정답 누락 / 충돌 문항 → 답 A·B 각각의
# 탈락(supported 아님)·변형(distorted)을 값과 출처 doc_id와 함께 적는다. CCD 유형 이름은 넣지 않는다.
# 이 문장이 GEPA 지표의 feedback이자 ProTeGi 오류 예시의 "Problem:" 줄이 된다.
# 주의: judge 파싱 실패 안내는 탈락·변형 문장이 하나라도 있을 때만 붙고, 둘 다 보존이면 "OK" 문장만 나간다.
def describe_loss(instance: dict, judge: dict, over_budget: bool, out_tokens: int, budget: int) -> str:
    """Which answer (value, source document) was dropped or changed; budget overrun; invented conflict."""
    g = instance["gold"]
    if over_budget:
        return (f"Over budget: the output has {out_tokens} tokens but the limit is {budget}, even after one "
                f"retry, so it scores 0. Remove content that does not bear on the question first.")
    if instance["conflict_type"] == "none":
        if judge.get("false_conflict"):
            return (f"The documents agree on one answer ('{g['answer_a']}'), but the output presents "
                    f"a conflict that is not in the documents.")
        if not judge.get("answer_kept", True):
            return (f"The documents agree on one answer ('{g['answer_a']}'), but the output does not contain it, "
                    f"so a reader of the output cannot answer the question.")
        return f"OK: the single answer ('{g['answer_a']}') is kept and no conflict is invented."
    parts = []
    for side, label in (("a", "A"), ("b", "B")):
        src = _doc_of(instance, f"conflict_{side}", 0 if side == "a" else 1)
        ans = g[f"answer_{side}"]
        if not judge.get(f"{side}_supported"):
            parts.append(f"Answer {label} ('{ans}', from {src}) was dropped: a reader of the output cannot recover it.")
        elif judge.get(f"{side}_distorted"):
            parts.append(f"Answer {label} ('{ans}', from {src}) was changed or merged with another value "
                         f"instead of being kept as written.")
    if not parts:
        return (f"OK: both answers ('{g['answer_a']}' from {_doc_of(instance, 'conflict_a', 0)} and "
                f"'{g['answer_b']}' from {_doc_of(instance, 'conflict_b', 1)}) are preserved.")
    if judge.get("parse_ok") is False:
        parts.append("(The judge reply could not be parsed; missing flags were read as false.)")
    return " ".join(parts)


# C3 최적화 지표의 단일 정의. 압축 출력 하나를 judge_model(호출부에서 opt_judge 역할)로 채점해
# {"score", "feedback", "judge"}를 돌려준다.
# 무충돌 문항: FCRJudge + answer_kept → 예산 초과·거짓 충돌·정답 누락 중 하나라도 있으면 0, 아니면 1.
# 충돌 문항: CSRJudge의 csr 값, 예산 초과면 0.
# score_instruction(ProTeGi)과 dspy_adapter.make_metric(GEPA·MIPROv2)이 이 함수를 부른다.
def judge_and_score(instance: dict, compressed: str, out_tokens: int, budget: int, over_budget: bool,
                    client, judge_model: str) -> dict:
    """The C3 metric for one compressed output: {"score", "feedback", "judge"}."""
    from judge import CSRJudge, FCRJudge
    if instance["conflict_type"] == "none":
        j = FCRJudge(client, judge_model).evaluate(instance, compressed, over_budget)
        j["answer_kept"] = answer_kept(instance["gold"]["answer_a"], compressed)
        score = 0.0 if over_budget or j["false_conflict"] or not j["answer_kept"] else 1.0
    else:
        j = CSRJudge(client, judge_model).evaluate(instance, compressed, over_budget)
        score = 0.0 if over_budget else float(j["csr"])
    return {"score": score, "feedback": describe_loss(instance, j, over_budget, out_tokens, budget), "judge": j}


# ------------------------------------------------------------------ examples that reached the compressor

_SEEN: set[str] = set()
_SEEN_LOCK = threading.Lock()


# 문자열의 UTF-8 SHA-256 16진 해시를 돌려준다. 예시 블록·시작 지시문의 해시 기록에 쓴다.
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# 압축기 호출에 실제로 들어간 예시 블록의 해시를 모듈 전역 집합 _SEEN에 잠금을 잡고 추가한다.
# score_instruction과 dspy_adapter.HarnessAdapter.format이 호출하며, 3-shot 고정 검사(fewshot_check.py)의 근거가 된다.
def record_examples(examples_block: str) -> None:
    with _SEEN_LOCK:
        _SEEN.add(sha256_text(examples_block))


# 지금까지 기록된 예시 블록 해시를 정렬한 목록으로 돌려준다. APO 로그에 저장된다.
def examples_seen() -> list[str]:
    with _SEEN_LOCK:
        return sorted(_SEEN)


# 기록된 예시 블록 해시를 비운다. apo_gepa·apo_mipro가 compile 직전에 호출한다(apo_protegi는 호출하지 않는다).
def reset_examples_seen() -> None:
    with _SEEN_LOCK:
        _SEEN.clear()


# use_fewshot이 참이면 prompts/c2_fewshot.json을 render_examples로 렌더링한 예시 블록을, 거짓이면 빈 문자열을 돌려준다.
# DSPy 프로그램에 넣을 예시 블록과 로그의 기대 해시(examples_sha256_expected)를 만드는 데 쓴다.
def expected_examples_block(use_fewshot: bool, fewshot_path: Path = ROOT_DIR / "prompts" / "c2_fewshot.json") -> str:
    return render_examples(json.loads(Path(fewshot_path).read_text(encoding="utf-8"))) if use_fewshot else ""


# 지시문 하나로 (문항, r) 목록을 압축·채점해 문항별 결과 dict 목록을 입력 순서대로 돌려준다.
# 본 시험 하네스와 같은 PromptCompressor(이름 "APO")를 쓰므로 예시 블록 조립과 예산 초과 시 1회 재시도 규칙이 같다.
# 결과 dict: id, r, score, feedback(describe_loss), compressed, over_budget, out_tokens, budget.
# apo_protegi의 Scorer가 score_fn으로 받아 쓰며, 문항은 client.concurrency(속성이 없으면 8)개 스레드로 병렬 처리한다.
def score_instruction(instruction: str, items: list[tuple[dict, float]], client, *,
                      examples: list[dict] | None, task_model: str, judge_model: str) -> list[dict]:
    """Score one instruction on (instance, r) items. Same compressor assembly and retry rule as the
    harness (PromptCompressor). Items run in parallel threads (client.concurrency); each item's two
    compressor calls are sequential because the retry depends on the first output."""
    comp = PromptCompressor("APO", None, task_model, client, instruction_text=instruction, examples=examples or [])

    # 문항 하나: compute_budget(total_tokens, r)로 예산을 정하고, 예시 블록 해시를 기록한 뒤 압축하고 judge_and_score로 채점한다.
    def one(item):
        inst, r = item
        budget = compute_budget(inst["total_tokens"], r)
        record_examples(comp.examples_block)
        c = comp.compress(inst, budget)
        s = judge_and_score(inst, c["text"], c["tokens"], budget, c["over_budget"], client, judge_model)
        return {"id": inst["id"], "r": r, "score": s["score"], "feedback": s["feedback"], "compressed": c["text"],
                "over_budget": c["over_budget"], "out_tokens": c["tokens"], "budget": budget}

    with ThreadPoolExecutor(max_workers=max(1, getattr(client, "concurrency", 8))) as ex:
        return list(ex.map(one, items))


# 시작 지시문이 prompts/c2_best.txt인데 C2 튜닝 로그 표의 '종료' 행 두 번째 칸이 비어 있으면 경고 문자열을, 아니면 None을 돌려준다.
# C2 확정 전 초안에서 C3 최적화를 시작한다는 사실을 알릴 뿐 실행을 막지는 않는다.
# 로그 파일이 없으면 미종료로 보고, '종료' 행이 여러 개면 마지막 행으로 판정한다.
def c2_freeze_warning(start_path: Path, log_path: Path = C2_TUNING_LOG) -> str | None:
    """Warning text when a run starts from prompts/c2_best.txt while the C2 tuning log has no end time."""
    if Path(start_path).resolve() != (ROOT_DIR / "prompts" / "c2_best.txt").resolve():
        return None
    ended = False
    if Path(log_path).exists():
        for line in Path(log_path).read_text(encoding="utf-8").splitlines():
            cells = [x.strip() for x in line.strip().strip("|").split("|")]
            if cells and cells[0] == "종료":
                ended = len(cells) > 1 and bool(cells[1])
    return None if ended else ("WARNING: C2 is not frozen (the '종료' row of the C2 tuning log is empty). "
                               "This run starts from the current prompts/c2_best.txt draft.")
