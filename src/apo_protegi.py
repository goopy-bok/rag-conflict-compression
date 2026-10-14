"""
C3-P: ProTeGi (Pryzant et al., 2023) ported to budgeted conflict-preserving compression
(execution guide v2, section 7.4). Loop structure follows microsoft/LMOps prompt_optimization
(main.py rounds loop, optimizers.ProTeGi.expand_candidates, evaluators.UCBBandits); nothing from
LMOps is imported.

  uv run python src/apo_protegi.py --estimate
  uv run python src/apo_protegi.py --mode mock --seed 0
  uv run python src/apo_protegi.py --mode execute --max-cost 30 --seed 0 [--trial]

Algorithm (paper Alg. 1-3, section 3.2 defaults; none are tuned, they are only exposed as flags)
  B0 = {start instruction}; for each of `steps` steps:
  expand   one train minibatch (64) per step, shared by every beam parent (LMOps). Per parent:
           errors = items with score < 1 (conflict: CSR=0 / over budget; distractor: false conflict).
           `error_groups` (4) groups of `errors_per_group` (4) errors sampled at random ->
           one protegi_grad call per group asking for `num_feedbacks` (4) reasons ->
           one protegi_edit call per reason (`steps_per_gradient` = 1) ->
           `mc_samples` (2) protegi_para calls per edited prompt.
           Successors (edits + paraphrases) deduplicated, parent copies removed, then at most
           `max_expansion` (8) kept by random sample.
  select   candidate pool = beam parents + successors (LMOps adds the originals back).
           UCB bandit (c = 2.0): each round draws `ucb_samples` train items and scores the
           `ucb_arms_per_round` arms with the highest Q + c*sqrt(ln t / N) (all arms random in round 1).
           Rounds = ceil(budget_per_candidate (25) * n_candidates / (arms * samples)).
           Next beam = top `beam` (4) arms by mean reward Q.
  val      (step_val, default on) every beam is scored on the full val split -> curve + peak step.
  final    best val score of the last beam -> prompts/c3p.txt.

Score of one item = apo_common.score_instruction (compressor role, retry rule, opt_judge role,
fixed c2_fewshot.json examples). Gradient / edit / paraphrase calls: reflection role,
temperature 1.0, empty system prompt (LMOps sends the template as the only user message).

Parse rules
  grad / edit : every complete <START>...<END> pair, stripped, empties dropped (LMOps
                parse_tagged_text). A reply with no complete pair yields nothing and is counted
                in parse_failures[kind]; no retry.
  para        : whole reply stripped (untagged in the paper and LMOps); empty reply -> failure.
  A parent with no errors on its minibatch, or whose calls yield no successors, adds no
  candidates; it still competes in selection.

Differences from paper / LMOps (reasons in the completion report)
  - error string = guide 7.4 format (Question / Expected / Compressed output / Problem), no source docs.
  - n samples = n calls (LLMClient has no n); max_tokens is offset by the request's position in its batch
    so identical prompts (e.g. 2 paraphrases of one edit) are not deduplicated/cached into one reply.
  - UCB uses 1-indexed rounds (paper Alg. 3); LMOps passes 0-indexed round numbers.
  - the original instruction is not paraphrased (paper Alg. 2; LMOps also paraphrases it).
  - final pick = best val score of the final beam (guide 7.1), not test maxpooling.
  - mock mode: LLMClient's mock has no reflection rule, so reflection replies are replaced by
    deterministic tagged text (`_mock_reflection`). The calls are still counted by the client.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from llm_client import LLMClient, load_models_config  # noqa: E402

PROMPTS_DIR = Path(os.environ.get("CCD_PROMPTS_DIR") or ROOT_DIR / "prompts")
REPORTS_DIR = Path(os.environ.get("CCD_REPORTS_DIR") or ROOT_DIR / "reports")
TEMPLATE_PATHS = {k: ROOT_DIR / "prompts" / f"protegi_{k}.txt" for k in ("grad", "edit", "para")}
DEFAULT_START = ROOT_DIR / "prompts" / "c2_best.txt"
FEWSHOT_PATH = ROOT_DIR / "prompts" / "c2_fewshot.json"
CONDITION = "C3-P"
REFLECTION_MAX_TOKENS = 2048
TRIAL_N = 50
# Split sizes in the apo_common contract (dev 197 + distractor 50 / dev 100 + distractor 50),
# used by --estimate only when apo_common is not importable.
CONTRACT_SIZES = (247, 150)

DEFAULTS = dict(
    steps=6, beam=4, minibatch=64,
    error_groups=4, errors_per_group=4, num_feedbacks=4, steps_per_gradient=1, mc_samples=2,
    max_expansion=8,
    ucb_c=2.0, budget_per_candidate=25, ucb_samples=5, ucb_arms_per_round=8,
    step_val=True, reflection_temperature=1.0,
)


# ------------------------------------------------------------------ templates and strings

# prompts/protegi_{kind}.txt(kind = grad/edit/para) 반성 프롬프트 템플릿을 읽어 앞뒤 공백을 지워 돌려준다.
def load_template(kind: str) -> str:
    return TEMPLATE_PATHS[kind].read_text(encoding="utf-8").strip()


# 반성 모델 응답에서 완결된 <START>...<END> 쌍의 내용을 순서대로 꺼내 공백을 지운 목록으로 돌려준다.
# 빈 항목은 버리고, <END>로 닫히지 않은 마지막 <START>는 무시한다. grad·edit 응답 파싱에 쓴다.
def parse_tagged(text: str, start: str = "<START>", end: str = "<END>") -> list[str]:
    """Port of LMOps ProTeGi.parse_tagged_text; empty items dropped."""
    out = []
    while True:
        i = text.find(start)
        if i == -1:
            break
        j = text.find(end, i)
        if j == -1:
            break
        item = text[i + len(start):j].strip()
        if item:
            out.append(item)
        text = text[j + len(end):]
    return out


# error_entry용: role이 일치하는 문서의 doc_id를, 없으면 conflict_positions[pos] 위치 문서의 doc_id를 돌려준다.
# apo_common._doc_of와 같은 역할이지만 "?" 대체값이 없어 conflict_positions가 없거나 짧으면 예외가 난다.
def _doc_id(inst: dict, role: str, pos: int) -> str:
    for d in inst["docs"]:
        if d.get("role") == role:
            return d["doc_id"]
    return inst["docs"][inst["conflict_positions"][pos]]["doc_id"]


# 실패 문항 하나를 grad/edit 템플릿에 넣을 "## Example k" 블록(Question/Expected/Compressed output/Problem)으로 만든다.
# Expected는 무충돌 문항이면 "충돌을 말하지 말 것", 충돌 문항이면 두 답과 각 출처 doc_id다. Problem은 describe_loss 피드백이다.
# 원문 문서는 넣지 않는다. 무충돌 문항의 Expected에는 단일 정답 보존 요구가 없지만, 점수는 answer_kept도 반영한다.
def error_entry(k: int, inst: dict, res: dict) -> str:
    """One error in the guide 7.4 format. Problem = describe_loss text (res["feedback"])."""
    if inst["conflict_type"] == "none":
        expected = "no conflict should be stated"
    else:
        g = inst["gold"]
        expected = (f'a reader of the context can recover both "{g["answer_a"]}" (from {_doc_id(inst, "conflict_a", 0)})'
                    f' and "{g["answer_b"]}" (from {_doc_id(inst, "conflict_b", 1)})')
    return (f"## Example {k}\nQuestion: {inst['question']}\nExpected: {expected}\n"
            f"Compressed output: {res['compressed']}\nProblem: {res['feedback']}")


# 오류 목록에 1부터 번호를 매겨 error_entry 블록들을 빈 줄로 이어 붙인 문자열을 돌려준다(템플릿의 error_string 자리).
def error_string(errors: list[tuple[dict, dict]]) -> str:
    return "\n\n".join(error_entry(k, inst, res) for k, (inst, res) in enumerate(errors, start=1))


# ------------------------------------------------------------------ UCB bandit (paper Alg. 3)

# UCB 라운드 수 = ceil(budget_per_candidate × 후보 수 ÷ (라운드당 팔 수 × 라운드당 표본 수))를 돌려준다.
# 팔 수는 후보 수로, 표본 수는 train 크기로 상한을 둔다. run과 estimate_calls가 같은 식을 쓴다.
def ucb_rounds(n_candidates: int, cfg: dict, n_train: int) -> int:
    k = min(cfg["ucb_arms_per_round"], n_candidates)
    s = min(cfg["ucb_samples"], n_train)
    return math.ceil(cfg["budget_per_candidate"] * n_candidates / (k * s))


# 논문 Alg. 3의 UCB 밴딧으로 n개 후보(팔)를 rounds 라운드 동안 평가한다.
# 매 라운드 k개 팔을 골라 pull(chosen)로 (팔별 평균 보상, 표본 수 m)을 받고, 보상을 m으로 가중해 누적한다.
# 누적 표본이 0인 동안(첫 라운드)은 rng로 무작위 k개를, 이후는 Q + c·sqrt(ln t / (N + 1e-3)) 상위 k개(동점은 낮은 인덱스)를 고른다.
# 반환값: (팔별 평균 보상 Q, 팔별 누적 표본 수, 라운드 기록). 한 번도 뽑히지 않은 팔의 Q는 0이다.
def ucb_bandit(n: int, pull, rounds: int, k: int, c: float, rng: random.Random):
    """pull(chosen_arm_indices) -> (mean reward per chosen arm, n samples).
    Returns (Q per arm, pulls per arm, per-round history). Arms never pulled have Q = 0."""
    k = min(k, n)
    counts, sums, hist = [0] * n, [0.0] * n, []
    for t in range(1, rounds + 1):
        q = [s / m if m else 0.0 for s, m in zip(sums, counts)]
        if sum(counts) == 0:
            chosen = rng.sample(range(n), k)
            ucb = None
        else:
            ucb = [q[i] + c * math.sqrt(math.log(t) / (counts[i] + 1e-3)) for i in range(n)]
            chosen = sorted(range(n), key=lambda i: (-ucb[i], i))[:k]
        rewards, m = pull(chosen)
        for i, r in zip(chosen, rewards):
            counts[i] += m
            sums[i] += r * m
        hist.append({"t": t, "chosen": chosen, "rewards": rewards,
                     "ucb": [round(ucb[i], 4) for i in chosen] if ucb else None})
    q = [s / m if m else 0.0 for s, m in zip(sums, counts)]
    return q, counts, hist


# ------------------------------------------------------------------ scoring with memo

# score_instruction을 감싸 (지시문 SHA-256, 문항 id, r) 키로 결과를 메모해, 같은 조합을 다시 압축·채점하지 않게 한다.
# minibatch·UCB·val 평가가 이 객체 하나를 공유하므로, step_val이 켜져 있으면 최종 선택 때 val 재채점이 추가 호출 없이 끝난다.
# requested(요청 문항 수, 메모 적중 포함)와 len(memo)(실제 평가 문항 수)가 로그에 남는다.
class Scorer:
    """Wraps apo_common.score_instruction with a (instruction, id, r) memo, as LMOps Cached01Scorer."""

    # 클라이언트, 채점 함수(score_fn), 고정 예시, 압축기·채점 모델 이름, 지시문 단위 병렬도(parallel, 기본 4)를 보관하고 빈 메모와 잠금을 만든다.
    def __init__(self, client, score_fn, examples, task_model, judge_model, parallel: int = 4):
        self.client, self.score_fn, self.examples = client, score_fn, examples
        self.task_model, self.judge_model, self.parallel = task_model, judge_model, parallel
        self.memo: dict = {}
        self.requested = 0
        self._lock = threading.Lock()

    # 메모 키 (지시문 SHA-256 16진 해시, 문항 id, r)를 만든다. item은 (문항, r) 튜플이다.
    @staticmethod
    def _key(instruction, item):
        return hashlib.sha256(instruction.encode("utf-8")).hexdigest(), item[0]["id"], item[1]

    # 지시문 하나로 items를 채점해 입력 순서대로 결과 dict 목록을 돌려준다. 메모에 없는 키만 모아(중복 제거) score_fn을 한 번 부른다.
    # 메모 확인과 score_fn 호출은 잠금 밖에서 이뤄지므로, 같은 키를 동시에 요청한 두 스레드는 각자 평가한다(메모는 나중 값으로 덮인다).
    def score(self, instruction: str, items: list) -> list[dict]:
        todo = {}
        for it in items:
            key = self._key(instruction, it)
            if key not in self.memo:
                todo.setdefault(key, it)
        if todo:
            res = self.score_fn(instruction, list(todo.values()), self.client, examples=self.examples,
                                task_model=self.task_model, judge_model=self.judge_model)
            with self._lock:
                self.memo.update(zip(todo, res))
        with self._lock:
            self.requested += len(items)
        return [self.memo[self._key(instruction, it)] for it in items]

    # (지시문, 문항 목록) 작업들을 최대 parallel개 스레드로 동시에 채점해 작업별 평균 점수 목록을 돌려준다(문항이 없으면 0.0).
    # UCB의 pull과 val 평가가 이 메서드를 쓴다.
    def mean_many(self, jobs: list[tuple[str, list]]) -> list[float]:
        # ponytail: instructions scored in a small thread pool, each one fans out via call_many
        with ThreadPoolExecutor(max_workers=max(1, min(self.parallel, len(jobs)))) as ex:
            results = list(ex.map(lambda j: self.score(*j), jobs))
        return [sum(r["score"] for r in rs) / len(rs) if rs else 0.0 for rs in results]


# ------------------------------------------------------------------ reflection calls

# mock 모드 전용 가짜 반성 응답. 프롬프트 해시로 결정적인 텍스트를 만든다.
# grad는 <START>/<END>로 감싼 이유 4개, edit는 태그된 지시문 1개, para는 태그 없는 문장 하나다.
# LLMClient의 mock에는 반성 응답 규칙이 없어 Reflector.run이 meta["mock"]일 때 응답을 이 값으로 바꾼다.
def _mock_reflection(kind: str, prompt: str) -> str:
    h = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
    if kind == "grad":
        return "".join(f"<START>Mock reason {h % 1000}-{i}: the prompt does not say how to keep both answers.<END>\n"
                       for i in range(4))
    if kind == "edit":
        return (f"<START>Mock instruction {h % 100000}. Compress the documents for the question and keep "
                f"every conflicting answer with its document id.<END>")
    return f"Mock paraphrase {h % 100000}. Compress the documents and keep all conflicting answers."


# reflection 역할 모델(models 설정 roles["reflection"])에 grad/edit/para 반성 호출을 보내는 래퍼다.
# 종류별 호출 수(calls)와 파싱 실패 수(parse_failures)를 세어 APO 로그에 남긴다. parse_failures는 expand가 올린다.
class Reflector:
    # 클라이언트·모델 이름·온도(run에서는 reflection_temperature, 기본 1.0)를 보관하고 종류별 카운터를 0으로 둔다.
    def __init__(self, client, model: str, temperature: float):
        self.client, self.model, self.temperature = client, model, temperature
        self.calls = {"grad": 0, "edit": 0, "para": 0}
        self.parse_failures = {"grad": 0, "edit": 0, "para": 0}

    # 프롬프트마다 독립 요청 하나를 만들어 client.call_many로 한꺼번에 보내고, 응답 텍스트를 같은 순서의 목록으로 돌려준다.
    # 시스템 프롬프트는 비우고 템플릿 전체를 user 메시지로 보낸다. max_tokens를 2048 + 위치로 달리해 같은 프롬프트가
    # LLMClient의 중복 제거·캐시 키에서 한 응답으로 합쳐지지 않게 한다. 빈 목록이면 호출 없이 []를 돌려준다.
    def run(self, kind: str, prompts: list[str]) -> list[str]:
        """One independent sample per prompt. max_tokens = base + position, so repeated prompts in a batch
        (two paraphrases of one edit, two identical error groups) are distinct requests for LLMClient's
        dedup/cache key instead of collapsing into one reply."""
        reqs = [{"model": self.model, "system": "", "prompt": p, "temperature": self.temperature,
                 "max_tokens": REFLECTION_MAX_TOKENS + j} for j, p in enumerate(prompts)]
        if not reqs:
            return []
        out = []
        for p, (text, meta) in zip(prompts, self.client.call_many(reqs)):
            out.append(_mock_reflection(kind, p) if meta.get("mock") else text)
        self.calls[kind] += len(prompts)
        return out


# 논문 Alg. 2: 부모 지시문 하나를 확장한다. minibatch를 채점해 score < 1인 문항을 오류로 모으고,
# error_groups개 그룹(그룹당 errors_per_group개 비복원 표본)마다 grad 1회 → 이유마다 edit 1회 → 편집 지시문마다 para mc_samples회를 호출한다.
# 편집문+패러프레이즈를 중복 제거하고 부모와 같은 것을 뺀 뒤, max_expansion개를 넘으면 rng로 무작위로 줄인다.
# 반환값: (후계 후보 목록, trace 기록 dict). 오류가 없으면 반성 호출 없이 빈 목록을 돌려준다.
# 그룹끼리는 독립 표본이라 같은 오류가 여러 그룹에 들어갈 수 있다.
def expand(parent: str, minibatch: list, scorer: Scorer, refl: Reflector, cfg: dict,
           rng: random.Random, tpl: dict) -> tuple[list[str], dict]:
    """Paper Alg. 2 for one parent. Returns (successors, trace record)."""
    res = scorer.score(parent, minibatch)
    errors = [(inst, r) for (inst, _), r in zip(minibatch, res) if r["score"] < 1]
    rec = {"parent": parent, "minibatch_ids": [[i["id"], r] for i, r in minibatch],
           "minibatch_score": sum(r["score"] for r in res) / len(res), "n_errors": len(errors),
           "groups": [], "candidates_all": [], "candidates_kept": []}
    if not errors:
        return [], rec

    groups = [error_string(rng.sample(errors, min(len(errors), cfg["errors_per_group"])))
              for _ in range(cfg["error_groups"])]
    grad_raw = refl.run("grad", [tpl["grad"].format(prompt=parent, error_string=es,
                                                    num_feedbacks=cfg["num_feedbacks"]) for es in groups])
    pairs = []  # (group index, gradient)
    for gi, (es, raw) in enumerate(zip(groups, grad_raw)):
        grads = parse_tagged(raw)
        refl.parse_failures["grad"] += not grads
        rec["groups"].append({"error_string": es, "gradient_raw": raw, "gradients": grads, "edits": []})
        pairs += [(gi, g) for g in grads]

    edit_raw = refl.run("edit", [tpl["edit"].format(prompt=parent, error_str=groups[gi], gradient=g,
                                                    steps_per_gradient=cfg["steps_per_gradient"])
                                 for gi, g in pairs])
    edits = []
    for (gi, g), raw in zip(pairs, edit_raw):
        new = parse_tagged(raw)
        refl.parse_failures["edit"] += not new
        rec["groups"][gi]["edits"].append({"gradient": g, "edit_raw": raw, "prompts": new})
        edits += new

    m = cfg["mc_samples"]
    para_raw = refl.run("para", [tpl["para"].format(prompt_instruction=e) for e in edits for _ in range(m)])
    paras = []
    for raw in para_raw:
        text = raw.strip()
        refl.parse_failures["para"] += not text
        if text:
            paras.append(text)
    rec["paraphrases"] = paras

    cands = [c for c in dict.fromkeys(edits + paras) if c != parent]
    rec["candidates_all"] = cands
    if len(cands) > cfg["max_expansion"]:
        cands = rng.sample(cands, cfg["max_expansion"])
    rec["candidates_kept"] = cands
    return cands, rec


# ------------------------------------------------------------------ estimate

# 설정과 train/val 크기로 한 번 실행의 역할별 호출 수를 계산한다(--estimate와 apo_estimate용, LLM 호출 없음).
# 모든 부모가 오류를 갖고 모든 응답이 요청 개수대로 파싱되며 메모·캐시 적중이 없다고 가정한 명목 상한이다.
# 채점 문항 하나 = 압축기 1회(재시도 시 2회, calls_max) + opt_judge 1회로 센다. step_val이 꺼져 있으면 마지막 beam의 val 평가만 센다.
# 반환 dict: by_role(compressor/opt_judge/reflection), score_items(minibatch/ucb/val/total), ucb_rounds_per_step, 병합된 config.
def estimate_calls(config: dict, n_train: int, n_val: int) -> dict:
    """Nominal call counts for one run (upper bound: every parent has errors, every reply parses into
    the requested number of items, no memo/cache hits between minibatch, UCB and val items).
    One scored item = 1 compressor call (2 when the budget retry fires) + 1 opt_judge call
    (0 when still over budget)."""
    c = {**DEFAULTS, **config}
    per_parent_refl = (c["error_groups"] + c["error_groups"] * c["num_feedbacks"]
                       + c["error_groups"] * c["num_feedbacks"] * c["steps_per_gradient"] * c["mc_samples"])
    succ = c["error_groups"] * c["num_feedbacks"] * c["steps_per_gradient"] * (1 + c["mc_samples"])
    items = {"minibatch": 0, "ucb": 0, "val": n_val if c["step_val"] else 0}
    refl, beam, ucb_r = 0, 1, []
    for _ in range(c["steps"]):
        items["minibatch"] += beam * min(c["minibatch"], n_train)
        refl += beam * per_parent_refl
        n = beam + beam * min(c["max_expansion"], succ)
        rounds = ucb_rounds(n, c, n_train)
        ucb_r.append(rounds)
        items["ucb"] += rounds * min(c["ucb_arms_per_round"], n) * min(c["ucb_samples"], n_train)
        beam = min(c["beam"], n)
        if c["step_val"]:
            items["val"] += beam * n_val
    if not c["step_val"]:
        items["val"] += beam * n_val
    total = sum(items.values())
    return {
        "by_role": {
            "compressor": {"calls": total, "calls_max": 2 * total},
            "opt_judge": {"calls": total, "calls_max": total},
            "reflection": {"calls": refl, "calls_max": refl,
                           "per_parent": {"grad": c["error_groups"],
                                          "edit": c["error_groups"] * c["num_feedbacks"],
                                          "para": per_parent_refl - c["error_groups"] * (1 + c["num_feedbacks"])}},
        },
        "score_items": {**items, "total": total},
        "ucb_rounds_per_step": ucb_r,
        "n_train": n_train, "n_val": n_val,
        "config": c,
    }


# ------------------------------------------------------------------ run

# ProTeGi 메인 루프. beam = [시작 지시문]에서 출발해 steps번 반복한다. 각 스텝은
# (1) train에서 minibatch 하나를 뽑아 beam의 모든 부모가 공유하며 expand, (2) 부모+후계 후보를 UCB 밴딧으로 평가(후보가 1개면 평가 없이 Q=1.0),
# (3) 평균 보상 Q 상위 beam개를 다음 beam으로, (4) step_val이면 beam 전체를 val로 채점한다.
# 최종 지시문은 마지막 스텝 beam 안에서 val 최고(동점은 train Q, 다음은 앞 순서)로 골라 prompts_dir/c3p.txt에 쓴다.
# 앞 스텝의 val 최고(peak_step)는 로그에만 남고 선택에 쓰이지 않는다. reports/openrouter 로그(steps=3)에서는
# step 2의 val 0.767 후보가 step 3 beam에 남지 못해 val 0.74 지시문이 저장됐다.
# reports_dir에 protegi_trace.jsonl(실행 시작 때 비움)과 apo_log_C3-P_seed{seed}.json을 쓰고 로그 dict를 돌려준다.
# 예외가 나도 status를 aborted로 적어 로그를 남긴 뒤 다시 던진다. val이 비어 있으면(--trial) train Q로 고른다.
# seed 하나로 만든 rng가 minibatch·오류 그룹·후보 축소·UCB 첫 라운드·UCB 표본을 모두 뽑는다.
def run(cfg: dict, client, split: dict, start_instruction: str, *, seed: int, score_fn,
        examples, task_model: str, judge_model: str, reflection_model: str,
        prompts_dir: Path, reports_dir: Path, extra_log: dict | None = None) -> dict:
    cfg = {**DEFAULTS, **cfg}
    rng = random.Random(seed)
    tpl = {k: load_template(k) for k in TEMPLATE_PATHS}
    train, val = split["train"], split["val"]
    scorer = Scorer(client, score_fn, examples, task_model, judge_model)
    refl = Reflector(client, reflection_model, cfg["reflection_temperature"])
    prompts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    trace_path = reports_dir / "protegi_trace.jsonl"
    log_path = reports_dir / f"apo_log_{CONDITION}_seed{seed}.json"
    trace_path.write_text("", encoding="utf-8")
    t0 = time.time()

    # 기록 하나를 JSON 한 줄로 protegi_trace.jsonl 끝에 덧붙인다.
    def trace(rec):
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # beam의 각 지시문을 val 전체로 채점한 평균 목록을 돌려준다. val이 비어 있으면 None 목록이다.
    def val_scores(beam):
        return scorer.mean_many([(b, val) for b in beam]) if val else [None] * len(beam)

    log = {"condition": CONDITION, "seed": seed, "config": cfg, "n_train": len(train), "n_val": len(val),
           "models": {"task": task_model, "judge": judge_model, "reflection": reflection_model},
           **(extra_log or {}), "steps": [], "status": "running"}

    # peak_step(val 최고 스텝, 동점은 이른 스텝, val이 없는 스텝이 있으면 None)을 다시 계산하고, 메모·반성 호출·파싱 실패·
    # 경과 시간·비용 요약을 붙여 로그 파일을 덮어쓴다. apo_common이 import돼 있으면 examples_seen도 저장한다.
    # 스텝마다, 그리고 종료·중단 시(finally) 호출된다.
    def write_log():
        best = [s["best_val"] for s in log["steps"]]
        log["peak_step"] = (max(range(len(best)), key=lambda i: (best[i], -i))
                            if best and None not in best else None)
        log.update(metric_items_requested=scorer.requested, metric_items_evaluated=len(scorer.memo),
                   reflection_calls=dict(refl.calls), parse_failures=dict(refl.parse_failures),
                   elapsed_s=round(time.time() - t0, 1), client_summary=client.summary())
        ac = sys.modules.get("apo_common")  # examples blocks that reached the compressor (fewshot_check)
        if hasattr(ac, "examples_seen"):
            log["examples_seen"] = ac.examples_seen()
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")

    beam, beam_q, beam_n = [start_instruction], [None], [0]
    try:
        v = val_scores(beam) if cfg["step_val"] else [None]
        log["steps"].append({"step": 0, "beam": [{"text": start_instruction, "train_q": None, "train_n": 0,
                                                  "val": v[0]}], "best_val": v[0], "best_train_q": None})
        trace({"step": 0, "type": "val", "beam": beam, "val": v})
        write_log()

        for step in range(1, cfg["steps"] + 1):
            minibatch = rng.sample(train, min(cfg["minibatch"], len(train)))
            succ = []
            for pi, parent in enumerate(beam):
                new, rec = expand(parent, minibatch, scorer, refl, cfg, rng, tpl)
                trace({"step": step, "type": "expand", "parent_idx": pi, **rec})
                succ += new
            n_parents = len(dict.fromkeys(beam))
            cands = list(dict.fromkeys(beam + succ))

            if len(cands) == 1:  # LMOps score_candidates: a single prompt is not evaluated
                q, counts, hist, rounds = [1.0], [0], [], 0
            else:
                rounds = ucb_rounds(len(cands), cfg, len(train))

                # UCB 한 라운드의 보상 함수: train에서 ucb_samples개 문항을 새로 뽑아 chosen 후보들을 채점하고 (평균 점수 목록, 표본 수)를 돌려준다.
                def pull(chosen):
                    sample = rng.sample(train, min(cfg["ucb_samples"], len(train)))
                    return scorer.mean_many([(cands[i], sample) for i in chosen]), len(sample)

                q, counts, hist = ucb_bandit(len(cands), pull, rounds, cfg["ucb_arms_per_round"],
                                             cfg["ucb_c"], rng)
            order = sorted(range(len(cands)), key=lambda i: (-q[i], i))[:cfg["beam"]]
            beam, beam_q, beam_n = [cands[i] for i in order], [q[i] for i in order], [counts[i] for i in order]
            trace({"step": step, "type": "select", "ucb_rounds": rounds,
                   "candidates": [{"text": c, "q": q[i], "n": counts[i], "from_beam": i < n_parents}
                                  for i, c in enumerate(cands)],
                   "ucb_history": hist, "beam": beam})

            v = val_scores(beam) if cfg["step_val"] else [None] * len(beam)
            trace({"step": step, "type": "val", "beam": beam, "val": v})
            log["steps"].append({
                "step": step, "n_candidates": len(cands), "ucb_rounds": rounds,
                "beam": [{"text": b, "train_q": bq, "train_n": bn, "val": bv}
                         for b, bq, bn, bv in zip(beam, beam_q, beam_n, v)],
                "best_val": max(v) if None not in v else None, "best_train_q": max(beam_q)})
            write_log()

        # final: best val score of the last beam (memo makes this free when step_val is on)
        v = val_scores(beam)
        key = (lambda i: (v[i], beam_q[i] or 0, -i)) if val else (lambda i: (beam_q[i] or 0, -i))
        best = max(range(len(beam)), key=key)
        final = beam[best]
        (prompts_dir / "c3p.txt").write_text(final + "\n", encoding="utf-8")
        log["final"] = {"instruction": final, "beam_index": best, "val": v[best], "train_q": beam_q[best],
                        "final_beam_val": v, "selected_by": "val" if val else "train_q (no val split)",
                        "path": str(prompts_dir / "c3p.txt")}
        log["status"] = "done"
    except BaseException as e:
        log["status"] = f"aborted: {type(e).__name__}: {e}"
        raise
    finally:
        write_log()
    return log


# ------------------------------------------------------------------ CLI

# CLI 인자 파서를 만든다. --seed/--mode/--max-cost/--start-instruction/--estimate/--trial/출력 디렉터리 외에
# DEFAULTS의 모든 키를 --steps, --beam, --minibatch처럼 플래그로 노출한다(bool은 --step-val/--no-step-val).
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["mock", "execute"], default="mock")
    p.add_argument("--max-cost", type=float, default=None)
    p.add_argument("--start-instruction", type=Path, default=DEFAULT_START)
    p.add_argument("--estimate", action="store_true", help="print call counts; no LLM calls")
    p.add_argument("--trial", action="store_true",
                   help=f"week-2 trial: {TRIAL_N} train items, 1 step, no val; outputs under */trial/")
    p.add_argument("--prompts-dir", type=Path, default=None,
                   help="directory for prompt outputs (default: $CCD_PROMPTS_DIR or prompts/)")
    p.add_argument("--reports-dir", type=Path, default=None,
                   help="directory for report outputs (default: $CCD_REPORTS_DIR or reports/)")
    for k, v in DEFAULTS.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=v)
        else:
            p.add_argument(flag, type=type(v), default=v)
    return p


# CLI 진입점. --estimate면 apo_common.SIZES(가져올 수 없으면 CONTRACT_SIZES)로 호출 수만 출력하고 끝낸다(분할 파일을 만들지 않음).
# 그 외에는 execute 모드에 --max-cost를 요구하고, C2 확정 경고를 stderr로 알린 뒤, seed로 APO 분할을 읽어
# models 설정의 compressor·opt_judge·reflection 역할 모델과 고정 3-shot(c2_fewshot.json)으로 run을 부른다.
# --trial은 train 50건 표본·steps=1·step_val=False·val 없음으로 돈다. 출력 하위 폴더는 mock 모드면 mock/, execute+--trial이면 trial/이다.
# 참고: DEFAULTS는 steps=6이지만 reports/openrouter/apo_log_C3-P_seed0.json의 config는 steps=3(beam 4, minibatch 64)이다.
def main(argv=None) -> None:
    a = _parser().parse_args(argv)
    cfg = {k: getattr(a, k) for k in DEFAULTS}
    if a.trial:
        cfg.update(steps=1, step_val=False)
    roles = load_models_config()["roles"]

    if a.estimate:
        try:  # sizes only: load_apo_split would write data/processed/apo_split.json on first use
            from apo_common import SIZES
            n_train, n_val = (sum(v[i] for v in SIZES.values()) for i in (0, 1))
            src = "apo_common.SIZES"
        except ImportError:
            (n_train, n_val), src = CONTRACT_SIZES, "contract sizes (apo_common not importable)"
        if a.trial:
            n_train, n_val = min(TRIAL_N, n_train), 0
        est = estimate_calls(cfg, n_train, n_val)
        for role in est["by_role"]:
            est["by_role"][role]["model"] = roles[role]
        est["split_source"] = src
        print(json.dumps(est, ensure_ascii=False, indent=1))
        return

    if a.mode == "execute" and a.max_cost is None:
        raise SystemExit("--mode execute requires --max-cost")
    from apo_common import c2_freeze_warning, load_apo_split, score_instruction
    start_path = a.start_instruction.resolve()
    warning = c2_freeze_warning(start_path)
    if warning:
        print(warning, file=sys.stderr)
    split = load_apo_split(a.seed)
    if a.trial:
        split = {"train": random.Random(a.seed).sample(split["train"], min(TRIAL_N, len(split["train"]))),
                 "val": []}
    sub = "mock" if a.mode == "mock" else ("trial" if a.trial else "")
    client = LLMClient(mode=a.mode, max_cost_usd=a.max_cost)
    prompts_dir = Path(a.prompts_dir or os.environ.get("CCD_PROMPTS_DIR") or PROMPTS_DIR) / sub
    reports_dir = Path(a.reports_dir or os.environ.get("CCD_REPORTS_DIR") or REPORTS_DIR) / sub
    log = run(cfg, client, split, start_path.read_text(encoding="utf-8").strip(), seed=a.seed,
              score_fn=score_instruction, examples=json.loads(FEWSHOT_PATH.read_text(encoding="utf-8")),
              task_model=roles["compressor"], judge_model=roles["opt_judge"],
              reflection_model=roles["reflection"],
              prompts_dir=prompts_dir, reports_dir=reports_dir,
              extra_log={"start_instruction": str(start_path), "c2_freeze_warning": warning, "trial": a.trial,
                         "mode": a.mode, "max_cost": a.max_cost})
    print(json.dumps({k: log[k] for k in ("status", "peak_step", "final", "metric_items_evaluated",
                                          "reflection_calls", "parse_failures")}, ensure_ascii=False, indent=1))
    print(json.dumps(log["client_summary"], indent=1))


if __name__ == "__main__":
    main()
