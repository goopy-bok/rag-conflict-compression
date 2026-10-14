"""
G0 judge-validation tools (execution guide v2, section 6 "judge 검증 순서" + "추가 검증").
Default mode is dry_run: no paid API call is made unless --mode execute is given.

  rejudge          re-score existing compressed texts (no re-compression) with a new judge
  negative-control build 100 control items (another instance's answer_b in slot B); --run judges them
  label-sheet      blind author label sheet (conditions hidden, order shuffled)
  agreement        author labels vs two rejudge files -> accuracy, kappa, confusion, FP rate
  audit-pilot      count pilot cases where 'distorted' drove CSR = 0 (no API call)

Inputs: harness records (with question/gold/compressed) or pilot_select records (id, model, condition,
text, ...; joined with data/processed/dev.jsonl by id, text -> compressed).
--n N: per-condition equal, seed-fixed sample (--sample-seed). The chosen (condition, id) list is saved to
<out-dir>/sample_ids.json on first use and reused by label-sheet, rejudge and negative-control, so all three
see the same items. Outputs go to --out-dir (default results/g0_local; the OpenRouter-era results/g0 is kept).
agreement stops with exit code 3 while author_labels.csv has empty flags; otherwise it also writes
agreement.json with "pass" (G0 criteria, all checks) for scripts/run_lite.sh.

Examples:
  uv run python src/validate_judge.py label-sheet
  uv run python src/validate_judge.py negative-control --seed 7
  uv run python src/validate_judge.py rejudge --judge opt --mode dry_run
  uv run python src/validate_judge.py agreement
  uv run python src/validate_judge.py label-sheet --n 50 \
      --results results/pilot_select/qwen2.5-7b_C0.jsonl results/pilot_select/qwen2.5-7b_C1.jsonl
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from judge import CSRJudge, value_in_text, value_kind  # noqa: E402
from stats import cohen_kappa  # noqa: E402

G0_DIR = ROOT_DIR / "results" / "g0_local"
LEGACY_G0_DIR = ROOT_DIR / "results" / "g0"   # OpenRouter-era outputs, kept as is
DEV_FILE = ROOT_DIR / "data" / "processed" / "dev.jsonl"
UNLABELLED_EXIT = 3
DEFAULT_RESULTS = [ROOT_DIR / "results" / "c0_dev50_r0.3.jsonl", ROOT_DIR / "results" / "c1_dev50_r0.3.jsonl"]
JUDGE_KEYS = {"opt": "opt_judge", "eval": "eval_judge"}
FLAGS = ("a_supported", "a_distorted", "b_supported", "b_distorted")
THRESH = {"accuracy": 0.85, "kappa": 0.7, "neg_fp": 0.10}


# ------------------------------------------------------------------ helpers

def load_records(paths) -> list[dict]:
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out += [json.loads(l) for l in f if l.strip()]
    return out


def join_dev(recs: list[dict], dev_file: Path | None = None) -> list[dict]:
    """pilot_select records (no question/gold, compressed text under "text") -> harness-like records."""
    if all("compressed" in r for r in recs):
        return recs
    dev_file = dev_file or DEV_FILE
    dev = {r["id"]: r for r in load_records([dev_file])}
    out = []
    for r in recs:
        if "compressed" in r:
            out.append(r)
            continue
        if r["id"] not in dev:
            raise KeyError(f"pilot record id {r['id']!r} not in {dev_file}")
        d = dev[r["id"]]
        out.append({**r, "question": d["question"], "gold": d["gold"], "conflict_type": d["conflict_type"],
                    "compressed": r["text"]})
    return out


def sample_records(recs: list[dict], n: int, seed: int, ids_file: Path) -> list[dict]:
    """n records, n/k per condition (k conditions), distinct instance ids across conditions, seed-fixed.
    The first call writes ids_file; later calls reuse it (same items for every command)."""
    by_key = {}
    for r in recs:
        k = (r["condition"], r["id"])
        if k in by_key:
            raise ValueError(f"duplicate record {k}: pass one compressor's files only")
        by_key[k] = r
    if ids_file.exists():
        saved = json.loads(ids_file.read_text(encoding="utf-8"))
        if (saved["n"], saved["seed"]) != (n, seed):
            raise SystemExit(f"{ids_file} has n={saved['n']} seed={saved['seed']}, asked n={n} seed={seed}; "
                             "use the saved sample or another --out-dir")
        keys = [tuple(k) for k in saved["items"]]
        missing = [k for k in keys if k not in by_key]
        if missing:
            raise SystemExit(f"{len(missing)} sampled items not in --results (e.g. {missing[0]}); wrong input files?")
        return [by_key[k] for k in keys]
    conds = sorted({c for c, _ in by_key})
    if n % len(conds):
        raise SystemExit(f"--n {n} is not divisible by the {len(conds)} conditions {conds}")
    rng, taken, keys = random.Random(seed), set(), []
    for c in conds:
        pool = sorted(i for cc, i in by_key if cc == c and i not in taken)
        if len(pool) < n // len(conds):
            raise SystemExit(f"condition {c}: only {len(pool)} unused ids, need {n // len(conds)}")
        pick = rng.sample(pool, n // len(conds))
        taken.update(pick)
        keys += [(c, i) for i in pick]
    ids_file.parent.mkdir(parents=True, exist_ok=True)
    ids_file.write_text(json.dumps({"n": n, "seed": seed, "sources": sorted({r.get("model", "") for r in recs}),
                                    "items": keys}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {ids_file} ({len(keys)} items)")
    return [by_key[k] for k in keys]


def load_inputs(args) -> list[dict]:
    """--results (+ dev join for pilot records), sampled when --n is given."""
    recs = join_dev(load_records(args.results))
    if args.n:
        recs = sample_records(recs, args.n, args.sample_seed, Path(args.out_dir) / "sample_ids.json")
    return recs


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def resolve_model(judge: str) -> str:
    """Model id for 'opt' / 'eval' from configs/models.yaml (single source of truth)."""
    import yaml
    cfg = yaml.safe_load((ROOT_DIR / "configs" / "models.yaml").read_text(encoding="utf-8")) or {}
    key = JUDGE_KEYS[judge]
    v = (cfg.get("roles") or {}).get(key) or cfg.get(key)
    if isinstance(v, dict):
        v = v.get("id") or v.get("model")
    if not v:
        raise KeyError(f"{key} not found in configs/models.yaml")
    return v


def make_client(mode: str, max_cost: float | None):
    from llm_client import LLMClient  # lazy: Worker A's module, contract in the common brief
    return LLMClient(mode=mode, max_cost_usd=max_cost)


def item_key(seed: int, cond: str, rid: str) -> str:
    return hashlib.sha256(f"g0|{seed}|{cond}|{rid}".encode()).hexdigest()[:10]


def _yn(v: str) -> bool:
    """Yes/no answer, including 'Yes as if ...' style answers."""
    w = v.strip().lower().split()
    return bool(w) and w[0].strip(".,;:") in {"yes", "no"}


def _bool_cell(v: str):
    v = (v or "").strip().lower()
    if v in {"1", "true", "t", "y", "yes", "o"}:
        return True
    if v in {"0", "false", "f", "n", "no", "x"}:
        return False
    return None


def csr_of(f: dict) -> int:
    return int(f["a_supported"] and f["b_supported"] and not f["a_distorted"] and not f["b_distorted"])


# ------------------------------------------------------------------ rejudge

def cmd_rejudge(args) -> None:
    recs = load_inputs(args)
    model = resolve_model(args.judge)
    client = make_client(args.mode, args.max_cost)
    judge = CSRJudge(client, model)
    rows = []
    for r in recs:
        # over_budget=False on purpose: G0 validates the judge on content; the budget rule is applied
        # downstream (csr_with_budget) and the original flag is kept.
        res = judge.evaluate(r, r["compressed"], over_budget=False)
        rows.append({"id": r["id"], "condition": r["condition"], "over_budget": r["over_budget"],
                     "judge": res, "csr_content": res["csr"],
                     "csr_with_budget": 0 if r["over_budget"] else res["csr"]})
    summary = client.summary()
    print(json.dumps({"judge": args.judge, "model": model, "mode": args.mode, "items": len(recs),
                      "client_summary": summary}, ensure_ascii=False, indent=1))
    if args.mode == "dry_run":
        print("dry_run: nothing written (uncached replies are empty).")
        return
    out = Path(args.out_dir) / f"rejudge_{args.judge}.jsonl"
    write_jsonl(out, rows)
    print(f"wrote {out}  parse_fail={sum(not x['judge']['parse_ok'] for x in rows)}/{len(rows)}")


# ------------------------------------------------------------ negative control

def build_negative_items(recs: list[dict], seed: int = 7) -> list[dict]:
    """For every record, replace answer_b with another instance's answer_b.

    Donor rules (so that a 'supported' verdict is a false positive):
    different instance id; donor answer is not yes/no; donor value does not occur in this
    compressed text; differs from this instance's own answers. Same value kind
    (DATE/NUMBER/NAME) as the own answer_b is preferred when available.
    """
    rng = random.Random(seed)
    donors = {}
    for r in recs:
        if not _yn(r["gold"]["answer_b"]):
            donors.setdefault(r["id"], r["gold"]["answer_b"])
    items = []
    for r in recs:
        own = {r["gold"]["answer_a"].strip().lower(), r["gold"]["answer_b"].strip().lower()}
        pool = [(i, v) for i, v in sorted(donors.items())
                if i != r["id"] and v.strip().lower() not in own and not value_in_text(v, r["compressed"])]
        same = [d for d in pool if value_kind(d[1]) == value_kind(r["gold"]["answer_b"])]
        donor_id, donor_b = rng.choice(same or pool)
        items.append({"id": r["id"], "condition": r["condition"], "question": r["question"],
                      "compressed": r["compressed"], "donor_id": donor_id,
                      "gold": {"answer_a": r["gold"]["answer_a"], "answer_b": donor_b,
                               "span_a": r["gold"].get("span_a", r["gold"]["answer_a"]), "span_b": donor_b},
                      "original_answer_b": r["gold"]["answer_b"]})
    return items


def cmd_negative(args) -> None:
    path = Path(args.out_dir) / "negative_control_items.jsonl"
    if not args.run:
        items = build_negative_items(load_inputs(args), args.seed)
        write_jsonl(path, items)
        print(f"wrote {path} ({len(items)} items, seed={args.seed})")
        return
    items = load_records([path])
    client = make_client(args.mode, args.max_cost)
    judge = CSRJudge(client, resolve_model(args.judge))
    rows = [{**it, "judge": judge.evaluate(it, it["compressed"], over_budget=False)} for it in items]
    print(json.dumps(client.summary(), ensure_ascii=False, indent=1))
    if args.mode == "dry_run":
        return
    out = Path(args.out_dir) / f"negative_control_{args.judge}.jsonl"
    write_jsonl(out, rows)
    fp = sum(r["judge"]["b_supported"] for r in rows)
    print(f"wrote {out}  b_supported false-positive rate = {fp}/{len(rows)} = {fp / len(rows):.3f}")


# -------------------------------------------------------------- label sheet

def build_label_sheet(recs: list[dict], seed: int = 7) -> tuple[list[dict], dict]:
    rows, key = [], {}
    for r in recs:
        k = item_key(seed, r["condition"], r["id"])
        key[k] = {"condition": r["condition"], "id": r["id"]}
        rows.append({"item_key": k, "question": r["question"], "compressed": r["compressed"],
                     "answer_a": r["gold"]["answer_a"], "answer_b": r["gold"]["answer_b"],
                     **{f: "" for f in FLAGS}, "note": ""})
    random.Random(seed).shuffle(rows)
    return rows, key


def cmd_label_sheet(args) -> None:
    out_dir = Path(args.out_dir)
    csv_path, key_path = out_dir / "author_labels.csv", out_dir / "label_key.json"
    if csv_path.exists() and count_labels(csv_path)[0]:
        raise SystemExit(f"{csv_path} already has author labels; not overwritten")
    rows, key = build_label_sheet(load_inputs(args), args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    key_path.write_text(json.dumps({
        "_comment": "DO NOT OPEN BEFORE LABELLING author_labels.csv. Maps item_key -> condition/id. "
                    "Label flags with 1/0. Opening this file breaks the blind.",
        "seed": args.seed, "map": key}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {csv_path} ({len(rows)} rows) and {key_path}")


# --------------------------------------------------------------- agreement

def count_labels(labels_csv: Path) -> tuple[int, int]:
    """(rows with all four flags filled, rows with at least one empty/invalid flag)."""
    with open(labels_csv, encoding="utf-8-sig") as f:
        done = [all(_bool_cell(row[k]) is not None for k in FLAGS) for row in csv.DictReader(f)]
    return sum(done), len(done) - sum(done)


def g0_checks(rep: dict) -> dict:
    """G0 criteria (THRESH) for both judges; a missing judge / inter-judge / negative-control result fails."""
    checks = {}
    for name in JUDGE_KEYS:
        j = rep["judges"].get(name)
        checks[f"accuracy_{name}"] = bool(j and j["n"] and j["csr_accuracy"] >= THRESH["accuracy"])
        v = rep["negative_control"].get(name)
        checks[f"neg_fp_{name}"] = bool(v and v["n"] and v["fp_rate"] <= THRESH["neg_fp"])
    ij = rep["inter_judge"]
    checks["inter_judge_kappa"] = bool(ij and ij["n"] and ij["kappa"] >= THRESH["kappa"])
    return checks


def _confusion(human: list[int], judge: list[int]) -> dict:
    c = {"h1_j1": 0, "h1_j0": 0, "h0_j1": 0, "h0_j0": 0}
    for h, j in zip(human, judge):
        c[f"h{h}_j{j}"] += 1
    return c


def compute_agreement(labels_csv: Path, key_json: Path, rejudge_files: dict[str, Path],
                      neg_files: dict[str, Path] | None = None) -> dict:
    keymap = json.loads(key_json.read_text(encoding="utf-8"))["map"]
    human = {}
    with open(labels_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            vals = {k: _bool_cell(row[k]) for k in FLAGS}
            if all(v is not None for v in vals.values()):
                m = keymap[row["item_key"]]
                human[(m["condition"], m["id"])] = vals
    judges = {}
    for name, p in rejudge_files.items():
        judges[name] = {(r["condition"], r["id"]): r["judge"] for r in load_records([p])}
    rep = {"n_labelled": len(human), "judges": {}, "inter_judge": None, "negative_control": {}}
    for name, J in judges.items():
        ks = [k for k in human if k in J]
        h = [csr_of(human[k]) for k in ks]
        j = [csr_of(J[k]) for k in ks]
        acc = sum(a == b for a, b in zip(h, j)) / len(ks) if ks else float("nan")
        per_flag = {}
        for fl in FLAGS:
            hf = [bool(human[k][fl]) for k in ks]
            jf = [bool(J[k][fl]) for k in ks]
            per_flag[fl] = {"accuracy": sum(a == b for a, b in zip(hf, jf)) / len(ks) if ks else float("nan"),
                            "kappa": cohen_kappa(hf, jf, [True, False])}
        rep["judges"][name] = {"n": len(ks), "csr_accuracy": acc, "csr_kappa_vs_human": cohen_kappa(h, j, [0, 1]),
                               "confusion": _confusion(h, j), "per_flag": per_flag,
                               "parse_fail": sum(not J[k].get("parse_ok", True) for k in J)}
    if len(judges) == 2:
        (n1, J1), (n2, J2) = judges.items()
        ks = sorted(set(J1) & set(J2))
        a, b = [csr_of(J1[k]) for k in ks], [csr_of(J2[k]) for k in ks]
        rep["inter_judge"] = {"pair": f"{n1} vs {n2}", "n": len(ks), "kappa": cohen_kappa(a, b, [0, 1]),
                              "confusion": _confusion(a, b)}
    for name, p in (neg_files or {}).items():
        if Path(p).exists():
            rows = load_records([p])
            fp = sum(r["judge"]["b_supported"] for r in rows)
            rep["negative_control"][name] = {"n": len(rows), "fp": fp, "fp_rate": fp / len(rows) if rows else float("nan")}
    return rep


def _pass(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def render_agreement(rep: dict) -> str:
    L = [f"# G0 judge agreement report ({dt.date.today().isoformat()})", "",
         f"Author-labelled items: {rep['n_labelled']}. CSR is computed from flags on content only "
         "(the over-budget rule is not applied, so human and judge see the same object).",
         f"Criteria: accuracy vs author >= {THRESH['accuracy']}, inter-judge kappa >= {THRESH['kappa']}, "
         f"negative-control FP <= {THRESH['neg_fp']:.0%}.", "",
         "## CSR vs author", "", "| judge | n | accuracy | kappa vs author | h1/j1 | h1/j0 | h0/j1 | h0/j0 | parse fail | criterion |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, j in rep["judges"].items():
        c = j["confusion"]
        L.append(f"| {name} | {j['n']} | {j['csr_accuracy']:.3f} | {j['csr_kappa_vs_human']:.3f} | {c['h1_j1']} | {c['h1_j0']} "
                 f"| {c['h0_j1']} | {c['h0_j0']} | {j['parse_fail']} | {_pass(j['csr_accuracy'] >= THRESH['accuracy'])} |")
    L += ["", "## Per-flag agreement with author", "", "| judge | flag | accuracy | kappa |", "|---|---|---|---|"]
    for name, j in rep["judges"].items():
        for fl, v in j["per_flag"].items():
            L.append(f"| {name} | {fl} | {v['accuracy']:.3f} | {v['kappa']:.3f} |")
    if rep["inter_judge"]:
        ij = rep["inter_judge"]
        c = ij["confusion"]
        L += ["", "## Inter-judge (CSR)", "", f"{ij['pair']}: n={ij['n']}, kappa={ij['kappa']:.3f} "
              f"({_pass(ij['kappa'] >= THRESH['kappa'])}); confusion (first=1/second=1, 1/0, 0/1, 0/0) = "
              f"{c['h1_j1']}, {c['h1_j0']}, {c['h0_j1']}, {c['h0_j0']}"]
    L += ["", "## Negative control (b_supported on a foreign answer_b)", ""]
    if rep["negative_control"]:
        for name, v in rep["negative_control"].items():
            L.append(f"- {name}: {v['fp']}/{v['n']} = {v['fp_rate']:.3f} ({_pass(v['fp_rate'] <= THRESH['neg_fp'])})")
    else:
        L.append("- not run yet")
    return "\n".join(L) + "\n"


def cmd_agreement(args) -> None:
    d = Path(args.dir)
    labels = Path(args.labels or d / "author_labels.csv")
    done, empty = count_labels(labels)
    if empty:
        print(f"{labels}: {empty} of {done + empty} rows still have empty flags (a_supported, a_distorted, "
              "b_supported, b_distorted; 1/0). Fill them in, then rerun agreement.")
        raise SystemExit(UNLABELLED_EXIT)
    rej = {n: d / f"rejudge_{n}.jsonl" for n in ("opt", "eval")}
    rej = {n: p for n, p in rej.items() if p.exists()}
    neg = {n: d / f"negative_control_{n}.jsonl" for n in ("opt", "eval")}
    rep = compute_agreement(labels, Path(args.key or d / "label_key.json"), rej, neg)
    out = d / f"{dt.date.today().isoformat()}_agreement_report.md"
    out.write_text(render_agreement(rep), encoding="utf-8")
    checks = g0_checks(rep)
    (d / "agreement.json").write_text(json.dumps({"pass": all(checks.values()), "checks": checks,
                                                  "thresholds": THRESH, "report": rep}, indent=1), encoding="utf-8")
    print(f"wrote {out} and {d / 'agreement.json'}; G0 {'PASS' if all(checks.values()) else 'FAIL'} {checks}")


# ------------------------------------------------- judge-only checks (amendment 2.2-d)

def judge_only_checks(rejudge_files: dict[str, Path], neg_files: dict[str, Path]) -> dict:
    """The G0 criteria that need no author labels: inter-judge kappa on CSR and the negative-control FP rate of
    each judge (same formulas and thresholds as compute_agreement / g0_checks). Accuracy vs the author is the
    part left for the end (amendment 2.2-d)."""
    judges = {n: {(r["condition"], r["id"]): r["judge"] for r in load_records([p])}
              for n, p in rejudge_files.items() if Path(p).exists()}
    rep = {"inter_judge": None, "negative_control": {}}
    if len(judges) == 2:
        (n1, J1), (n2, J2) = judges.items()
        ks = sorted(set(J1) & set(J2))
        a, b = [csr_of(J1[k]) for k in ks], [csr_of(J2[k]) for k in ks]
        rep["inter_judge"] = {"pair": f"{n1} vs {n2}", "n": len(ks), "kappa": cohen_kappa(a, b, [0, 1]),
                              "confusion": _confusion(a, b)}
    for name, p in neg_files.items():
        if Path(p).exists():
            rows = load_records([p])
            fp = sum(r["judge"]["b_supported"] for r in rows)
            rep["negative_control"][name] = {"n": len(rows), "fp": fp, "fp_rate": fp / len(rows) if rows else float("nan")}
    ij = rep["inter_judge"]
    checks = {"inter_judge_kappa": bool(ij and ij["n"] and ij["kappa"] >= THRESH["kappa"])}
    for name in JUDGE_KEYS:
        v = rep["negative_control"].get(name)
        checks[f"neg_fp_{name}"] = bool(v and v["n"] and v["fp_rate"] <= THRESH["neg_fp"])
    return {"pass": all(checks.values()), "checks": checks, "thresholds": THRESH, "report": rep,
            "note": "accuracy vs author labels is checked at the end (agreement); amendment 2.2-d"}


def cmd_judge_checks(args) -> None:
    d = Path(args.dir)
    res = judge_only_checks({n: d / f"rejudge_{n}.jsonl" for n in JUDGE_KEYS},
                            {n: d / f"negative_control_{n}.jsonl" for n in JUDGE_KEYS})
    (d / "g0_auto.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {d / 'g0_auto.json'}; G0-auto {'PASS' if res['pass'] else 'FAIL'} {res['checks']}")
    if not res["pass"]:
        raise SystemExit(1)


# -------------------------------------------------------------- audit pilot

def audit_pilot(recs: list[dict]) -> dict:
    rows = []
    for r in recs:
        j = r["judge"]
        g = r["gold"]
        both_vals = (not _yn(g["answer_a"]) and value_in_text(g["answer_a"], r["compressed"], g["answer_b"])
                     and value_in_text(g["answer_b"], r["compressed"], g["answer_a"]))
        flags = []
        if r["csr"] == 0 and j["a_supported"] and j["b_supported"]:
            flags.append("both_supported_csr0" + ("(over_budget)" if r["over_budget"] else "(distorted)"))
        if r["csr"] == 0 and any(w in j["rationale"].lower() for w in ("both", "conflict", "contradict")):
            flags.append("rationale_both/conflict")
        if (j["a_distorted"] and not j["a_supported"]) or (j["b_distorted"] and not j["b_supported"]):
            flags.append("distorted_and_unsupported")
        if r["csr"] == 0 and both_vals:
            flags.append("both_values_in_text")
        rows.append({"cond": r["condition"], "id": r["id"], "over_budget": r["over_budget"], "csr": r["csr"],
                     "aS": int(j["a_supported"]), "aD": int(j["a_distorted"]), "bS": int(j["b_supported"]),
                     "bD": int(j["b_distorted"]), "flags": flags, "rationale": j["rationale"]})
    return {"rows": rows}


def cmd_audit(args) -> None:
    rows = audit_pilot(load_records(args.results))["rows"]
    count = lambda pred: sum(1 for r in rows if pred(r))
    has = lambda tag: (lambda r: any(f.startswith(tag) for f in r["flags"]))
    lines = [f"# Pilot judge audit: use of 'distorted' (dev_50, r=0.3, C0 and C1)", "",
             f"Generated {dt.date.today().isoformat()} by `src/validate_judge.py audit-pilot` from the existing "
             "result files; no API call. Judge = gpt-4o-mini with the pre-G0 prompt.", "",
             "## Counts (C0 + C1, 100 records)", "",
             "| measure | count |", "|---|---|",
             f"| CSR = 0 | {count(lambda r: r['csr'] == 0)} |",
             f"| CSR = 0 and over budget | {count(lambda r: r['csr'] == 0 and r['over_budget'])} |",
             f"| a_supported and b_supported but CSR = 0 because of a distorted flag | {count(lambda r: 'both_supported_csr0(distorted)' in r['flags'])} |",
             f"| a_supported and b_supported but CSR = 0 because of over budget | {count(lambda r: 'both_supported_csr0(over_budget)' in r['flags'])} |",
             f"| CSR = 0 and rationale contains both/conflict/contradict | {count(has('rationale_both'))} |",
             f"| a candidate marked distorted=true and supported=false | {count(has('distorted_and_unsupported'))} |",
             f"| CSR = 0 although both non-yes/no values occur in the compressed text (value_in_text) | {count(has('both_values_in_text'))} |",
             "", "## Flagged records", "", "| cond | id | over_budget | aS aD bS bD | flags | rationale |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["flags"] and (r["csr"] == 0):
            rat = r["rationale"].replace("|", "/")
            lines.append(f"| {r['cond']} | {r['id']} | {r['over_budget']} | {r['aS']} {r['aD']} {r['bS']} {r['bD']} "
                         f"| {', '.join(r['flags'])} | {rat} |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


# --------------------------------------------------------------------- main

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, judge=False):
        p.add_argument("--results", nargs="+", default=[str(x) for x in DEFAULT_RESULTS])
        p.add_argument("--n", type=int, help="per-condition equal sample size in total (e.g. 50 = 25 per condition)")
        p.add_argument("--sample-seed", type=int, default=0)
        p.add_argument("--out-dir", default=str(G0_DIR))
        if judge:
            p.add_argument("--judge", choices=list(JUDGE_KEYS), default="opt")
            p.add_argument("--mode", choices=["dry_run", "mock", "execute"], default="dry_run")
            p.add_argument("--max-cost", type=float, default=None)

    p = sub.add_parser("rejudge"); common(p, judge=True); p.set_defaults(fn=cmd_rejudge)
    p = sub.add_parser("negative-control"); common(p, judge=True)
    p.add_argument("--seed", type=int, default=7); p.add_argument("--run", action="store_true")
    p.set_defaults(fn=cmd_negative)
    p = sub.add_parser("label-sheet"); common(p); p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_label_sheet)
    p = sub.add_parser("agreement")
    p.add_argument("--dir", default=str(G0_DIR), help="folder with labels, key, rejudge and negative files")
    p.add_argument("--labels", help="default <dir>/author_labels.csv")
    p.add_argument("--key", help="default <dir>/label_key.json")
    p.set_defaults(fn=cmd_agreement)
    p = sub.add_parser("judge-checks", help="G0 criteria without author labels (kappa, negative control)")
    p.add_argument("--dir", default=str(G0_DIR))
    p.set_defaults(fn=cmd_judge_checks)
    p = sub.add_parser("audit-pilot"); common(p)
    p.add_argument("--out", default=str(LEGACY_G0_DIR / f"{dt.date.today().isoformat()}_pilot_distorted_audit.md"))
    p.set_defaults(fn=cmd_audit)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
