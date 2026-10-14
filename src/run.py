"""
CCD-G1 harness CLI (execution guide v2 section 6).

  uv run python src/run.py run --condition C0 --split dev50 --r 0.3 --mode dry_run --limit 5
  uv run python src/run.py run --condition C1 --split dev --r 0.5 --mode execute --max-cost 5.0
  uv run python src/run.py estimate --conditions C0 C1 --splits dev --rs 0.3 0.5
  uv run python src/run.py freeze --conditions C0 C1 C2 C3-G C3-G1 C3-M C3-P C3-V --seeds 0 1 2

Per instance: budget -> compress (1 retry) -> CSR judge (FCR judge for conflict_type "none";
the judge itself returns CSR=0 without a call when over budget) -> downstream answer -> D judge.
--no-downstream skips the last two steps; the record then has downstream = {answer/D/category/
committed_value/model: null, skipped: true} (G1-lite: OOD and distractor splits).
Modes: mock (offline, results under results/mock/), dry_run (default; no calls, no result file),
execute (real calls, needs --max-cost). Test-type splits go through seal.check_test_access first.
Local Ollama roles cost 0, so `--mode execute --max-cost 0` runs them (OpenRouter calls are then refused).
An interrupted execute/mock run leaves <out>.jsonl.partial; the next run of the same condition/split/r/seed
keeps its complete records and processes only the remaining ids. Records carry model_digests (Ollama).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from budget import compute_budget  # noqa: E402
from compressors import PromptCompressor  # noqa: E402
from llm_client import LLMClient, load_models_config  # noqa: E402

DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = Path(os.environ.get("CCD_RESULTS_DIR") or ROOT_DIR / "results")
CONDITIONS_CONFIG = Path(os.environ.get("CCD_CONDITIONS_CONFIG") or ROOT_DIR / "configs" / "conditions.yaml")
SPLITS = {
    "dev": DATA_DIR / "processed" / "dev.jsonl",
    "dev50": DATA_DIR / "processed" / "dev_50.jsonl",
    "dev_distractor": DATA_DIR / "processed" / "dev_distractor.jsonl",
    "test_id": DATA_DIR / "sealed" / "test_id.jsonl",
    "test_ood1": DATA_DIR / "sealed" / "test_ood1.jsonl",
    "test_ood2": DATA_DIR / "sealed" / "test_ood2.jsonl",
    "distractor": DATA_DIR / "sealed" / "distractor.jsonl",
}
TEST_SPLITS = {"test_id", "test_ood1", "test_ood2", "distractor"}
FEWSHOT_FILE = ROOT_DIR / "prompts" / "c2_fewshot.json"


# prompts/c2_fewshot.json에 실린 C2 few-shot 예시들의 source_id 집합을 돌려준다(파일이 없으면 빈 집합).
# load_split이 dev 분할에서 이 id들을 조건과 무관하게 빼므로, 조건 간 짝 비교가 같은 문항 집합 위에서 이뤄진다.
def fewshot_source_ids(path: Path = FEWSHOT_FILE) -> set[str]:
    """dev ids used as C2 few-shot examples. Dropped from dev for every condition so that
    paired comparisons run on the same instances (guide v2 section 5)."""
    if not path.exists():
        return set()
    return {ex["source_id"] for ex in json.loads(path.read_text(encoding="utf-8")) if "source_id" in ex}


# 분할 이름으로 문항 목록(JSONL 한 줄 = dict 하나)을 읽어 돌려준다. 시험형 분할(TEST_SPLITS)은 파일을 열기 전에
# seal.check_test_access로 접근 허용 여부를 확인하며, seal 모듈이 없으면 ImportError로 멈춘다.
# few-shot 예시 문항 제외는 split == "dev"이고 data_path를 주지 않았을 때만 한다(dev50·dev_distractor는 제외하지 않음).
# limit은 앞에서부터 자른다.
def load_split(split: str, data_path: Path | None = None, limit: int | None = None) -> list[dict]:
    if split not in SPLITS:
        raise SystemExit(f"unknown split {split!r}; choose from {sorted(SPLITS)}")
    if split in TEST_SPLITS:
        from seal import check_test_access  # no seal module -> ImportError -> test stays closed
        check_test_access(split)
    path = Path(data_path) if data_path else SPLITS[split]
    if not path.exists():
        raise FileNotFoundError(f"data file for split {split!r} not found: {path}")
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if split == "dev" and data_path is None:
        held_out = fewshot_source_ids()
        rows = [r for r in rows if r["id"] not in held_out]
    return rows[:limit] if limit else rows


# conditions.yaml에서 조건의 instruction(필수)·examples(선택) 경로를 읽어 {seed}를 채운 뒤 PromptCompressor를 만든다.
# 등록되지 않은 조건이면 SystemExit. run_condition 외에 pilot_select.py·pretest_gate.py도 이 함수를 쓴다.
def build_compressor(condition: str, seed: int, client, model: str,
                     conditions_config: Path = CONDITIONS_CONFIG) -> PromptCompressor:
    conds = yaml.safe_load(conditions_config.read_text(encoding="utf-8"))
    if condition not in conds:
        raise SystemExit(f"unknown condition {condition!r}; choose from {list(conds)}")
    c = conds[condition]
    ex = c.get("examples")
    return PromptCompressor(condition, ROOT_DIR / c["instruction"].format(seed=seed), model, client,
                            examples_path=ROOT_DIR / ex.format(seed=seed) if ex else None)


# freeze 명령용: 주어진 조건들의 instruction·examples 파일 경로를 중복 없이(처음 등장 순서 유지) 돌려준다.
# 경로 템플릿에 {seed}가 있으면 seed마다 한 번씩 펼친다. 하나라도 없으면 누락 목록 전체를 담아 FileNotFoundError를 낸다.
def condition_prompt_files(conditions: list[str], seeds: list[int],
                           conditions_config: Path = CONDITIONS_CONFIG) -> list[Path]:
    """Instruction and example files of the given conditions; seed-templated paths once per seed.
    Missing files raise FileNotFoundError (all of them listed)."""
    conds = yaml.safe_load(conditions_config.read_text(encoding="utf-8"))
    unknown = [c for c in conditions if c not in conds]
    if unknown:
        raise SystemExit(f"unknown condition(s) {unknown}; choose from {list(conds)}")
    paths = []
    for c in conditions:
        for key in ("instruction", "examples"):
            tpl = conds[c].get(key)
            if tpl:
                paths += [ROOT_DIR / tpl.format(seed=s) for s in (seeds if "{seed}" in tpl else [None])]
    paths = list(dict.fromkeys(paths))
    missing = [str(p.relative_to(ROOT_DIR)) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"cannot freeze, missing prompt files: {missing}")
    return paths


# --eval-judge/--opt-judge를 주지 않았을 때의 judge 역할: 시험형 분할이면 eval_judge, 그 외(dev 계열)는 opt_judge.
# 실제 모델 id는 models.yaml(또는 CCD_MODELS_CONFIG가 가리키는 파일)의 roles에서 정해진다.
def default_judge_role(split: str) -> str:
    return "eval_judge" if split in TEST_SPLITS else "opt_judge"


# 조건 1개 × 분할 1개 × 압축비 r 1개 × seed 1개를 실행해 문항별 레코드 목록을 돌려준다. CLI run·estimate 외에
# explore_compressor.py와 scripts/run_openrouter_expanded.py도 호출한다. 압축기는 roles["compressor"], 하류 모델은
# roles["downstream"], judge는 judge_role(기본 default_judge_role)의 모델이며 CSR·FCR·D judge가 같은 모델을 쓴다.
# dry_run이 아니면 <결과폴더>/<condition>_<split>_r<r>_s<seed>.jsonl.partial에 한 줄씩 쓰고 끝나면 os.replace로 정식
# 이름으로 바꾼다(mock은 results/mock/ 아래). .partial이 남아 있으면 읽히는 레코드는 유지하고 남은 id만 처리한다.
def run_condition(condition: str, split: str, r: float, client: LLMClient, seed: int = 0,
                  limit: int | None = None, judge_role: str | None = None,
                  results_dir: Path | None = None, conditions_config: Path | None = None,
                  data_path: Path | None = None, downstream: bool = True) -> list[dict]:
    from judge import CSRJudge, DJudge, DownstreamGenerator, FCRJudge

    roles = load_models_config()["roles"]
    judge_role = judge_role or default_judge_role(split)
    judge_model, down_model = roles[judge_role], roles["downstream"]
    instances = load_split(split, data_path, limit)
    conds_cfg = conditions_config or CONDITIONS_CONFIG
    compressor = build_compressor(condition, seed, client, roles["compressor"], conditions_config=conds_cfg)
    csr_j, fcr_j = CSRJudge(client, judge_model), FCRJudge(client, judge_model)
    gen, d_j = DownstreamGenerator(client, down_model), DJudge(client, judge_model)

    # 문항 1건 처리(ThreadPoolExecutor에서 병렬 실행): compute_budget → compressor.compress → 충돌 문항은 CSRJudge,
    # conflict_type == "none"(distractor)은 FCRJudge → downstream=True면 DownstreamGenerator.answer와 DJudge → 결과 JSONL
    # 한 줄이 될 dict. csr·csr_lenient·csr_strict·signal(judge의 conflict_signaled)은 distractor면 None, false_conflict는 반대다.
    # dry_run에서 빈 응답이면 judge·하류에는 자리표시 텍스트를 넘기지만, 레코드의 compressed에는 comp["text"]를 그대로 저장한다.
    def process(inst: dict) -> dict:
        budget = compute_budget(inst["total_tokens"], r)
        comp = compressor.compress(inst, budget)
        text = comp["text"]
        # dry_run: an uncached call returns "". Later prompts then use a placeholder that is unique per
        # run/instance (so unrelated runs are not deduplicated as one call) and, for the compressed text,
        # budget-sized (so judge input tokens are not underestimated).
        tag = f"[dry_run placeholder {condition} {split} r{r} s{seed} {inst['id']}]"
        if comp["meta"]["dry_run"] and not text:
            text = tag + " x" * budget
        distractor = inst["conflict_type"] == "none"
        if distractor:
            j = fcr_j.evaluate(inst, text, comp["over_budget"])
        else:
            j = csr_j.evaluate(inst, text, comp["over_budget"])
        if downstream:
            answer = gen.answer(inst, text)
            if client.mode == "dry_run" and not answer:
                answer = tag
            d = d_j.evaluate(inst, answer)
            down = {"answer": answer, "D": d["d"], "category": d["category"],
                    "committed_value": d["committed_value"], "model": down_model}
        else:
            down = {"answer": None, "D": None, "category": None, "committed_value": None, "model": None,
                    "skipped": True}
        return {
            "id": inst["id"], "condition": condition, "split": split, "r": r, "seed": seed,
            "source": inst.get("source"), "conflict_type": inst["conflict_type"],
            "question": inst["question"], "gold": inst["gold"],
            "conflict_positions": inst.get("conflict_positions"),
            "budget": budget, "out_tokens": comp["tokens"], "over_budget": comp["over_budget"],
            "retried": comp["retried"], "compressed": comp["text"],
            "judge": j, "judge_model": judge_model,
            "csr": None if distractor else j["csr"],
            "csr_lenient": None if distractor else j["csr_lenient"],
            "csr_strict": None if distractor else j["csr_strict"],
            "signal": None if distractor else j.get("conflict_signaled"),
            "false_conflict": j["false_conflict"] if distractor else None,
            "downstream": down,
            "model_digests": {m: client.model_digest(m)
                              for m in {compressor.model, judge_model} | ({down_model} if downstream else set())},
            "prompt_sha256": {"instruction": comp["meta"]["instruction_sha256"],
                              "examples": comp["meta"]["examples_sha256"]},
            "meta": {**comp["meta"], "mode": client.mode},
        }

    write = client.mode != "dry_run"
    out_path = None
    if write:
        results_dir = Path(results_dir or (RESULTS_DIR / "mock" if client.mode == "mock" else RESULTS_DIR))
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"{condition}_{split}_r{r}_s{seed}.jsonl"
    print(f"[{condition} {split} r={r} seed={seed} mode={client.mode}] {len(instances)} instances, "
          f"judge={judge_model}")

    records = []
    partial = out_path.with_name(out_path.name + ".partial") if out_path else None
    if partial and partial.exists():  # resume an interrupted run: keep complete records, skip their ids
        for line in partial.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut off by the interruption
        done = {rec["id"] for rec in records}
        instances = [inst for inst in instances if inst["id"] not in done]
        print(f"  resuming {partial.name}: {len(done)} done, {len(instances)} left")
    f = partial.open("w", encoding="utf-8") if partial else None
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    try:
        with ThreadPoolExecutor(max_workers=client.concurrency) as ex:
            for i, rec in enumerate(ex.map(process, instances), start=1):
                records.append(rec)
                if f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                if i % 10 == 0 or i == len(instances):
                    print(f"  {i}/{len(instances)}")
    finally:
        if f:
            f.close()
    if partial:
        os.replace(partial, out_path)
        print(f"  saved {out_path}")
    return records


# client.summary()가 돌려준 호출 통계(전체·캐시·미캐시 호출 수, 모델별 호출·토큰·추정 비용)를 표로 출력한다.
# 단가 미정 모델은 TO_CONFIRM으로 표시하고 합계에서 뺀다. dry_run이면 압축기 재시도 호출이 추정에 빠진다고 알린다.
def print_summary(s: dict) -> None:
    print(f"\n== LLM call summary (mode={s['mode']}) ==")
    print(f"calls total {s['calls_total']} | cached {s['calls_cached']} "
          f"(legacy sqlite {s['calls_legacy_cached']}) | uncached {s['calls_uncached']}")
    print(f"{'model':45} {'calls':>7} {'in_tok':>10} {'out_tok':>10} {'usd':>10}")
    for m, v in sorted(s["by_model"].items()):
        usd = "TO_CONFIRM" if v["usd_est"] is None else f"{v['usd_est']:.4f}"
        print(f"{m:45} {v['calls']:>7} {v['in_tokens_est']:>10} {v['out_tokens_est']:>10} {usd:>10}")
    print(f"usd_est_total (priced models only): {s['usd_est_total']:.4f} | usd_spent: {s['usd_spent']:.4f}")
    if s["unpriced_models"]:
        print(f"unpriced_models (price TO_CONFIRM, not in total): {', '.join(s['unpriced_models'])}")
    if s["mode"] == "dry_run":
        print("dry_run: NO API calls were made. Uncached counts/tokens are estimates "
              "(output = max_tokens x ratio; compressor retry calls cannot be predicted and are not counted).")


# run·estimate 서브커맨드가 공유하는 옵션(--limit, --eval-judge/--opt-judge 택일, --no-downstream, --results-dir,
# --conditions-config)을 붙인다.
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--limit", type=int)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--eval-judge", dest="judge_role", action="store_const", const="eval_judge")
    g.add_argument("--opt-judge", dest="judge_role", action="store_const", const="opt_judge")
    p.add_argument("--no-downstream", dest="downstream", action="store_false",
                   help="skip downstream answer + D judge (record downstream.skipped = true)")
    p.add_argument("--results-dir", type=Path, default=None,
                   help="directory for result jsonl files (default: $CCD_RESULTS_DIR or results/)")
    p.add_argument("--conditions-config", type=Path, default=None,
                   help="path to conditions yaml (default: $CCD_CONDITIONS_CONFIG or configs/conditions.yaml)")


# CLI 진입점. freeze는 condition_prompt_files 결과를 seal.freeze_prompts에 넘긴다. run은 --mode execute일 때
# --max-cost를 요구하고 run_condition을 한 번 돌린다. estimate는 dry_run 클라이언트 하나로 조건×분할×r×seed 조합을
# 모두 돌려 조합별 미캐시 호출 수를 표로 찍는다. run·estimate 모두 마지막에 print_summary를 부른다.
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run one condition x split x r")
    pr.add_argument("--condition", required=True)
    pr.add_argument("--split", required=True, choices=sorted(SPLITS))
    pr.add_argument("--r", type=float, required=True)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--mode", choices=["mock", "dry_run", "execute"], default="dry_run")
    pr.add_argument("--max-cost", type=float)
    _add_common(pr)

    pe = sub.add_parser("estimate", help="dry_run totals over condition x split x r combinations")
    pe.add_argument("--conditions", nargs="+", required=True)
    pe.add_argument("--splits", nargs="+", required=True, choices=sorted(SPLITS))
    pe.add_argument("--rs", nargs="+", type=float, required=True)
    pe.add_argument("--seeds", nargs="+", type=int, default=[0])
    _add_common(pe)

    pf = sub.add_parser("freeze", help="record final prompt hashes (seal.freeze_prompts)")
    pf.add_argument("--conditions", nargs="+", required=True)
    pf.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])

    a = ap.parse_args(argv)
    if a.cmd == "freeze":
        from seal import freeze_prompts
        paths = condition_prompt_files(a.conditions, a.seeds)
        print(f"frozen {len(paths)} files -> {freeze_prompts(paths)}")
        return
    if a.cmd == "run":
        if a.mode == "execute" and a.max_cost is None:
            ap.error("--mode execute requires --max-cost")
        client = LLMClient(mode=a.mode, max_cost_usd=a.max_cost)
        try:
            run_condition(a.condition, a.split, a.r, client, seed=a.seed, limit=a.limit, judge_role=a.judge_role,
                          results_dir=a.results_dir, conditions_config=a.conditions_config,
                          downstream=a.downstream)
        finally:
            print_summary(client.summary())
        return

    client = LLMClient(mode="dry_run")
    rows = []
    for c in a.conditions:
        for sp in a.splits:
            for r in a.rs:
                for seed in a.seeds:
                    before = client.summary()["calls_uncached"]
                    n = len(run_condition(c, sp, r, client, seed=seed, limit=a.limit, judge_role=a.judge_role,
                                          results_dir=a.results_dir, conditions_config=a.conditions_config,
                                          downstream=a.downstream))
                    rows.append((c, sp, r, seed, n, client.summary()["calls_uncached"] - before))
    print(f"\n{'condition':10} {'split':15} {'r':>5} {'seed':>5} {'n':>6} {'uncached':>9}")
    for c, sp, r, seed, n, u in rows:
        print(f"{c:10} {sp:15} {r:>5} {seed:>5} {n:>6} {u:>9}")
    print_summary(client.summary())


if __name__ == "__main__":
    main()
