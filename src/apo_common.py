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

def _load_rows(split: str) -> list[dict]:
    from run import load_split  # dev drops the 3 few-shot source instances
    return load_split(split)


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


def load_apo_split(seed: int = 0, path: Path = SPLIT_PATH,
                   rows: dict[str, list[dict]] | None = None) -> dict[str, list[tuple[dict, float]]]:
    rows = rows or {name: _load_rows(name) for name in SIZES}
    split = build_apo_split(seed, path, rows)
    by_id = {r["id"]: r for name in SIZES for r in rows[name]}
    return {part: [(by_id[e["id"]], float(e["r"])) for e in split[part]] for part in ("train", "val")}


# ------------------------------------------------------------------ feedback and score

def _norm_answer(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


def answer_kept(answer: str, text: str) -> bool:
    """Lexical: the normalized answer (lowercase, punctuation -> space, articles dropped) is a whole-word span of
    the normalized text. No LLM call, so it adds no unvalidated judge use (amendment 2.2-g)."""
    a = _norm_answer(answer or "")
    return bool(a) and f" {a} " in f" {_norm_answer(text or '')} "


def _doc_of(instance: dict, role: str, pos: int) -> str:
    for d in instance["docs"]:
        if d.get("role") == role:
            return d["doc_id"]
    cp = instance.get("conflict_positions") or []
    return instance["docs"][cp[pos]]["doc_id"] if len(cp) > pos else "?"


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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_examples(examples_block: str) -> None:
    with _SEEN_LOCK:
        _SEEN.add(sha256_text(examples_block))


def examples_seen() -> list[str]:
    with _SEEN_LOCK:
        return sorted(_SEEN)


def reset_examples_seen() -> None:
    with _SEEN_LOCK:
        _SEEN.clear()


def expected_examples_block(use_fewshot: bool, fewshot_path: Path = ROOT_DIR / "prompts" / "c2_fewshot.json") -> str:
    return render_examples(json.loads(Path(fewshot_path).read_text(encoding="utf-8"))) if use_fewshot else ""


def score_instruction(instruction: str, items: list[tuple[dict, float]], client, *,
                      examples: list[dict] | None, task_model: str, judge_model: str) -> list[dict]:
    """Score one instruction on (instance, r) items. Same compressor assembly and retry rule as the
    harness (PromptCompressor). Items run in parallel threads (client.concurrency); each item's two
    compressor calls are sequential because the retry depends on the first output."""
    comp = PromptCompressor("APO", None, task_model, client, instruction_text=instruction, examples=examples or [])

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
