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

def load_template(kind: str) -> str:
    return TEMPLATE_PATHS[kind].read_text(encoding="utf-8").strip()


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


def _doc_id(inst: dict, role: str, pos: int) -> str:
    for d in inst["docs"]:
        if d.get("role") == role:
            return d["doc_id"]
    return inst["docs"][inst["conflict_positions"][pos]]["doc_id"]


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


def error_string(errors: list[tuple[dict, dict]]) -> str:
    return "\n\n".join(error_entry(k, inst, res) for k, (inst, res) in enumerate(errors, start=1))


# ------------------------------------------------------------------ UCB bandit (paper Alg. 3)

def ucb_rounds(n_candidates: int, cfg: dict, n_train: int) -> int:
    k = min(cfg["ucb_arms_per_round"], n_candidates)
    s = min(cfg["ucb_samples"], n_train)
    return math.ceil(cfg["budget_per_candidate"] * n_candidates / (k * s))


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

class Scorer:
    """Wraps apo_common.score_instruction with a (instruction, id, r) memo, as LMOps Cached01Scorer."""

    def __init__(self, client, score_fn, examples, task_model, judge_model, parallel: int = 4):
        self.client, self.score_fn, self.examples = client, score_fn, examples
        self.task_model, self.judge_model, self.parallel = task_model, judge_model, parallel
        self.memo: dict = {}
        self.requested = 0
        self._lock = threading.Lock()

    @staticmethod
    def _key(instruction, item):
        return hashlib.sha256(instruction.encode("utf-8")).hexdigest(), item[0]["id"], item[1]

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

    def mean_many(self, jobs: list[tuple[str, list]]) -> list[float]:
        # ponytail: instructions scored in a small thread pool, each one fans out via call_many
        with ThreadPoolExecutor(max_workers=max(1, min(self.parallel, len(jobs)))) as ex:
            results = list(ex.map(lambda j: self.score(*j), jobs))
        return [sum(r["score"] for r in rs) / len(rs) if rs else 0.0 for rs in results]


# ------------------------------------------------------------------ reflection calls

def _mock_reflection(kind: str, prompt: str) -> str:
    h = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
    if kind == "grad":
        return "".join(f"<START>Mock reason {h % 1000}-{i}: the prompt does not say how to keep both answers.<END>\n"
                       for i in range(4))
    if kind == "edit":
        return (f"<START>Mock instruction {h % 100000}. Compress the documents for the question and keep "
                f"every conflicting answer with its document id.<END>")
    return f"Mock paraphrase {h % 100000}. Compress the documents and keep all conflicting answers."


class Reflector:
    def __init__(self, client, model: str, temperature: float):
        self.client, self.model, self.temperature = client, model, temperature
        self.calls = {"grad": 0, "edit": 0, "para": 0}
        self.parse_failures = {"grad": 0, "edit": 0, "para": 0}

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

    def trace(rec):
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def val_scores(beam):
        return scorer.mean_many([(b, val) for b in beam]) if val else [None] * len(beam)

    log = {"condition": CONDITION, "seed": seed, "config": cfg, "n_train": len(train), "n_val": len(val),
           "models": {"task": task_model, "judge": judge_model, "reflection": reflection_model},
           **(extra_log or {}), "steps": [], "status": "running"}

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
