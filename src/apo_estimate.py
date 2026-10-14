"""
Analytic call / token / cost estimate for the C3 optimizers. No LLM calls.

  uv run python src/apo_estimate.py

G3 (trial runs): GEPA C3-G small (train 20 / val 20, max_metric_calls 200), MIPROv2 reduced
                 (auto="light", train 20 / val 20), ProTeGi 1 step on 50 train items without val.
G4 (main runs) : GEPA x6 (C3-G seeds 0-2, C3-G1 seeds 0-2; max_metric_calls 6000, train 247 / val 150),
                 MIPROv2 auto="medium" x1, ProTeGi x1 (6 steps, beam 4, 8 successors, 25 items per
                 candidate, minibatch 64), Llama-3.1-8B auxiliary C3-G seed 0 (task LM = aux_compressor).

Token sizes come from the real APO split (apo_common.load_apo_split(0): instance lengths and budgets)
and the current prompt files. One "program call" = one scored item = compressor call (+ retry) + one
opt_judge call. Every assumption is printed with the table. Prices: configs/models.yaml (TO_CONFIRM ->
usd None); usd_assumed uses ASSUMED_PRICES, which are unconfirmed placeholders.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from budget import compute_budget, count_tokens  # noqa: E402
from compressors.llm import format_input  # noqa: E402
from llm_client import LLMClient, load_models_config  # noqa: E402

PROMPTS = ROOT_DIR / "prompts"
# USD per 1M tokens (input, output). Unconfirmed assumptions given in the task brief.
ASSUMED_PRICES = {
    "openai/gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "openai/gpt-4o-2024-08-06": (2.50, 10.00),
    "anthropic/claude-haiku-5.5": (1.00, 5.00),
    "meta-llama/llama-3.1-8b-instruct": (0.02, 0.05),
}
A = {
    "compress_out_ratio": 0.9,      # compressor output = 0.9 x budget
    "retry_rate": 0.10,             # share of items whose first output is over budget (one retry each)
    "retry_note_tokens": 30,
    "judge_out_ratio": None,        # filled from models.yaml dry_run_output_ratio (x judge max_tokens 400 / 200)
    "feedback_tokens": 45,          # describe_loss sentence
    "gepa_accept_rate": 0.10,       # share of GEPA iterations whose child beats the parent on the minibatch
    "gepa_reflect_out": "max(2 x instruction, 500) tokens",
    "mipro_bootstrap_success": 0.5,  # metric pass rate while MIPROv2 bootstraps informational demos
    "protegi_out": "grad: 60/reason, edit: 1.2 x instruction, para: 1.1 x instruction",
}
CSR_MAX, FCR_MAX = 400, 200


# ------------------------------------------------------------------ inputs

def _text(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def pool_stats(items: list[tuple[dict, float]]) -> dict:
    n = len(items)
    budgets = [compute_budget(i["total_tokens"], r) for i, r in items]
    return {
        "n": n,
        "body": sum(count_tokens(format_input(i, b)) for (i, _), b in zip(items, budgets)) / n,
        "budget": sum(budgets) / n,
        "question": sum(count_tokens(i["question"]) for i, _ in items) / n,
        "distractor_share": sum(i["conflict_type"] == "none" for i, _ in items) / n,
    }


def program_call(st: dict, instr: int, ex: int) -> dict[str, list[float]]:
    """Expected [calls, in, out] per role for one scored item."""
    from judge import load_prompt
    rr, out = A["retry_rate"], A["compress_out_ratio"] * st["budget"]
    comp_in = instr + ex + st["body"]
    d = st["distractor_share"]
    jsys = (1 - d) * count_tokens(load_prompt("judge_csr.txt")) + d * count_tokens(load_prompt("judge_fcr.txt"))
    return {
        "compressor": [1 + rr, comp_in * (1 + rr) + A["retry_note_tokens"] * rr, out * (1 + rr)],
        "opt_judge": [1, jsys + st["question"] + out + 60, A["judge_out_ratio"] * ((1 - d) * CSR_MAX + d * FCR_MAX)],
    }


class Ctx:
    def __init__(self):
        from apo_common import expected_examples_block, load_apo_split
        cfg = load_models_config()
        A["judge_out_ratio"] = cfg.get("dry_run_output_ratio", 0.5)
        self.roles = cfg["roles"]
        split = load_apo_split(0)
        self.train, self.val = pool_stats(split["train"]), pool_stats(split["val"])
        self.n_train, self.n_val = len(split["train"]), len(split["val"])
        self.ex = count_tokens(expected_examples_block(True))
        self.instr = {"C3-G": count_tokens(_text("c2_best.txt")), "C3-G1": count_tokens(_text("c1_plain.txt"))}


_CTX: Ctx | None = None


def ctx() -> Ctx:
    global _CTX
    _CTX = _CTX or Ctx()
    return _CTX


def _add(rows, job, role_model, calls, tin, tout):
    rows.append({"job": job, "model": role_model, "calls": calls, "in": tin, "out": tout})


def _program_rows(rows, job, n_calls, st, instr, ex, task_model, judge_model):
    pc = program_call(st, instr, ex)
    for key, model in (("compressor", task_model), ("opt_judge", judge_model)):
        c, i, o = pc[key]
        _add(rows, job, model, n_calls * c, n_calls * i, n_calls * o)


# ------------------------------------------------------------------ optimizers

def gepa_rows(condition: str, max_metric_calls: int = 6000, task_role: str = "compressor", *,
              n_train: int | None = None, n_val: int | None = None, label: str | None = None,
              minibatch: int = 3) -> list[dict]:
    """GEPA loop (gepa/core/engine.py run, reflective_mutation.py propose): seed full-val eval (V calls),
    then per iteration: parent on a train minibatch (m), 1 reflection call, child on the same minibatch (m);
    a child that beats the parent gets a full-val eval (V). iterations = (M - V) / (2m + a V)."""
    c = ctx()
    V, M, a = n_val or c.n_val, max_metric_calls, A["gepa_accept_rate"]
    it = math.floor((M - V) / (2 * minibatch + a * V))
    instr, ex = c.instr[condition], (c.ex if condition == "C3-G" else 0)
    job = label or f"GEPA {condition}"
    rows: list[dict] = []
    train_calls, val_calls = 2 * minibatch * it, V + a * it * V
    _program_rows(rows, job, train_calls, c.train, instr, ex, c.roles[task_role], c.roles["opt_judge"])
    _program_rows(rows, job, val_calls, c.val, instr, ex, c.roles[task_role], c.roles["opt_judge"])
    st = c.train
    refl_in = 250 + instr + minibatch * (ex + st["body"] + A["compress_out_ratio"] * st["budget"] + A["feedback_tokens"] + 20)
    _add(rows, job, c.roles["reflection"], it, it * refl_in, it * max(2 * instr, 500))
    for r in rows:
        r["note"] = f"iterations={it} (upper bound with accept rate 0: {math.floor((M - V) / (2 * minibatch))})"
    return rows


def mipro_rows(auto: str = "medium", *, n_train: int | None = None, n_val: int | None = None,
               label: str | None = None) -> list[dict]:
    """dspy/teleprompt/mipro_optimizer_v2.py, zero-shot: N = 12 (medium) / 6 (light) instruction
    candidates, trials = max(2 log2 N, 1.5 N); valset capped at 300 / 100; minibatch 35 when valset > 50
    with full evals = trials//5 + 1 (+1 if trials % 5); bootstrapping informational demos (seeds -1..N-1,
    3 + 2N demos at pass rate p); prompt model: <=10 dataset-summary calls + 1 summary + 2 program-aware
    + N proposals."""
    c = ctx()
    N = {"light": 6, "medium": 12, "heavy": 18}[auto]
    T = int(max(2 * math.log2(N), 1.5 * N))
    Vs = min(n_val or c.n_val, {"light": 100, "medium": 300, "heavy": 1000}[auto])
    Tn = n_train or c.n_train
    if Vs > 50:
        val_calls = 35 * T + Vs * (T // 5 + 1 + (1 if T % 5 else 0))
    else:
        val_calls = Vs * (T + 1)
    boot = (3 + 2 * N) / A["mipro_bootstrap_success"]
    instr, ex, job, rows = c.instr["C3-G"], c.ex, label or f"MIPROv2 {auto}", []
    _program_rows(rows, job, val_calls, c.val, instr, ex, c.roles["compressor"], c.roles["opt_judge"])
    _program_rows(rows, job, boot, c.train, instr, ex, c.roles["compressor"], c.roles["opt_judge"])
    st, pm = c.train, c.roles["reflection"]
    summ = min(10, math.ceil(Tn / 10))
    item = st["body"] + 40
    demo = ex + st["body"] + A["compress_out_ratio"] * st["budget"] + 40
    _add(rows, job, pm, summ + 1, summ * (300 + 10 * item) + 600, (summ + 1) * 150)
    _add(rows, job, pm, 2, 2 * (1500 + item), 2 * 200)
    _add(rows, job, pm, N, N * (900 + instr + 3 * demo), N * 1.2 * instr)
    for r in rows:
        r["note"] = f"N={N} trials={T} valset={Vs} program_calls={val_calls}+bootstrap~{boot:.0f}"
    return rows


def protegi_rows(config: dict | None = None, *, n_train: int | None = None, n_val: int | None = None,
                 label: str | None = None) -> list[dict]:
    """Call counts from Worker E's apo_protegi.estimate_calls(config, n_train, n_val) (guide 7.4)."""
    from apo_protegi import DEFAULTS, estimate_calls
    c = ctx()
    cfg = {**DEFAULTS, **(config or {})}
    est = estimate_calls(cfg, n_train if n_train is not None else c.n_train, n_val if n_val is not None else c.n_val)
    items = est["score_items"]
    instr, ex, job, rows = c.instr["C3-G"], c.ex, label or "ProTeGi", []
    tr_items = items["minibatch"] + items["ucb"]
    _program_rows(rows, job, tr_items, c.train, instr, ex, c.roles["compressor"], c.roles["opt_judge"])
    _program_rows(rows, job, items["val"], c.val, instr, ex, c.roles["compressor"], c.roles["opt_judge"])
    pp = est["by_role"]["reflection"]["per_parent"]
    parents = est["by_role"]["reflection"]["calls"] / max(1, sum(pp.values()))
    st = c.train
    err = 30 + st["question"] + A["compress_out_ratio"] * st["budget"] + A["feedback_tokens"] + 30
    errs = cfg["errors_per_group"] * err
    calls = {"grad": parents * pp["grad"], "edit": parents * pp["edit"], "para": parents * pp["para"]}
    tin = calls["grad"] * (120 + instr + errs) + calls["edit"] * (150 + instr + errs + 60) + calls["para"] * (40 + instr)
    tout = calls["grad"] * cfg["num_feedbacks"] * 60 + calls["edit"] * 1.2 * instr + calls["para"] * 1.1 * instr
    _add(rows, job, c.roles["reflection"], sum(calls.values()), tin, tout)
    for r in rows:
        r["note"] = (f"score_items={items['total']} (minibatch {items['minibatch']}, ucb {items['ucb']}, "
                     f"val {items['val']}), parents={parents:.0f}")
    return rows


# ------------------------------------------------------------------ output

def _usd(model: str, tin: float, tout: float, client: LLMClient) -> tuple[float | None, float | None]:
    p = client.price(model)
    q = ASSUMED_PRICES.get(model)
    real = None if p is None else (tin * p[0] + tout * p[1]) / 1e6
    return real, None if q is None else (tin * q[0] + tout * q[1]) / 1e6


def aggregate(rows: list[dict]) -> list[dict]:
    agg: dict[tuple, dict] = {}
    for r in rows:
        a = agg.setdefault((r["job"], r["model"]), {"job": r["job"], "model": r["model"], "calls": 0, "in": 0, "out": 0,
                                                     "note": r.get("note", "")})
        for k in ("calls", "in", "out"):
            a[k] += r[k]
    return list(agg.values())


def print_table(rows: list[dict], title: str = "") -> dict[str, float]:
    client = LLMClient(mode="dry_run", cache_dir=ROOT_DIR / "cache")
    rows = aggregate(rows)
    if title:
        print(f"\n== {title} ==")
    print(f"{'job':26} {'model':34} {'calls':>8} {'in_tok':>12} {'out_tok':>11} {'usd':>8} {'usd_assumed':>12}")
    tot_model: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0, 0.0])
    for r in rows:
        real, assumed = _usd(r["model"], r["in"], r["out"], client)
        print(f"{r['job'][:26]:26} {r['model'][:34]:34} {r['calls']:>8.0f} {r['in']:>12,.0f} {r['out']:>11,.0f} "
              f"{'None' if real is None else f'{real:.2f}':>8} {'-' if assumed is None else f'{assumed:.2f}':>12}")
        t = tot_model[r["model"]]
        t[0] += r["calls"]; t[1] += r["in"]; t[2] += r["out"]; t[3] += assumed or 0.0  # noqa: E702
    print("-- by model")
    for m, (c, i, o, u) in tot_model.items():
        print(f"{'':26} {m[:34]:34} {c:>8.0f} {i:>12,.0f} {o:>11,.0f} {'None':>8} {u:>12.2f}")
    total = sum(t[3] for t in tot_model.values())
    print(f"{'TOTAL (usd_assumed)':61} {'':>8} {'':>12} {'':>11} {'':>8} {total:>12.2f}")
    for job, note in dict.fromkeys((r["job"], r["note"]) for r in rows if r.get("note")):
        print(f"  note [{job}]: {note}")
    return {m: t[3] for m, t in tot_model.items()}


def g3_rows() -> list[dict]:
    return (gepa_rows("C3-G", 200, n_val=20, label="G3 GEPA C3-G trial")
            + mipro_rows("light", n_train=20, n_val=20, label="G3 MIPROv2 light")
            + protegi_rows({"steps": 1, "step_val": False}, n_train=50, n_val=0, label="G3 ProTeGi 1 step"))


def g4_rows() -> list[dict]:
    rows = []
    for cond in ("C3-G", "C3-G1"):
        for s in range(3):
            rows += gepa_rows(cond, 6000, label=f"G4 GEPA {cond} s{s}")
    rows += mipro_rows("medium", label="G4 MIPROv2 medium")
    rows += protegi_rows(label="G4 ProTeGi")
    rows += gepa_rows("C3-G", 6000, "aux_compressor", label="G4 GEPA C3-G Llama s0")
    return rows


def main() -> None:
    c = ctx()
    print("APO cost estimate (analytic; no LLM calls). Prices in configs/models.yaml are TO_CONFIRM -> usd None.")
    print("usd_assumed: UNCONFIRMED placeholder prices, USD per 1M in/out: "
          + ", ".join(f"{m.split('/')[-1]} {p[0]}/{p[1]}" for m, p in ASSUMED_PRICES.items()))
    print(f"split: train {c.n_train} / val {c.n_val}; mean budget train {c.train['budget']:.0f}, "
          f"val {c.val['budget']:.0f}; mean compressor input body {c.train['body']:.0f} tokens; "
          f"instruction C2 {c.instr['C3-G']} / C1 {c.instr['C3-G1']} tokens; few-shot block {c.ex} tokens")
    print("assumptions: " + "; ".join(f"{k}={v}" for k, v in A.items()))
    print("GEPA loop (installed gepa 0.1.4, core/engine.py + proposer/reflective_mutation): 1 reflection call per "
          "iteration (one predictor), minibatch 3 evaluated twice, full val only for accepted children; merge "
          "(<=5, no LM call) and metric re-calls while building the reflective dataset (judge cache hits) not "
          "counted. Early stop (5 val evals without improvement) can only lower these numbers.")
    g3 = print_table(g3_rows(), "G3 trial runs")
    g4 = print_table(g4_rows(), "G4 main optimization")
    print(f"\nusd_assumed totals: G3 {sum(g3.values()):.2f}, G4 {sum(g4.values()):.2f} (unconfirmed prices)")


if __name__ == "__main__":
    main()
