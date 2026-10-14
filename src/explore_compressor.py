"""
Exploratory compressor-capacity check (decision record §9, 2026-10-12). dev only; not used for any
formal G1 verdict and never opens sealed splits.

  uv run python src/explore_compressor.py --mode dry_run              # cost estimate, no calls
  uv run python src/explore_compressor.py --mode execute --max-cost 1  # gpt-4o-mini + phi3.5 baseline
  uv run python src/explore_compressor.py --report                    # tables + pre-registered verdict

Arms (r = 0.3, judge = opt_judge gemma2:9b for both compressors, no downstream):
  gpt-4o-mini : dev[:100] x C0 C1 C2  +  dev_distractor[:50] x C2      (OpenRouter, provider OpenAI pinned, seed 0)
  phi3.5      : dev[:100] x C0 C1 C2 = K5 gate texts (results/pretest_gate_format.jsonl), judged here
                + dev_distractor[:50] x C2 compressed fresh
Everything (results, cache, cost log, raw OpenRouter metadata) goes to results/explore_2026-10-12_4omini/.
configs/models.yaml is not modified: roles and the gpt-4o-mini price are overridden in memory only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import run  # noqa: E402
from judge import CSRJudge  # noqa: E402
from llm_client import LLMClient, load_models_config, openrouter_transport  # noqa: E402
from stats import mcnemar_p, wilson_score_interval  # noqa: E402

OUT = ROOT_DIR / "results" / "explore_2026-10-12_4omini"
GPT = "openai/gpt-4o-mini-2024-07-18"
GPT_PRICE = {"in": 0.15, "out": 0.60}  # OpenRouter list price, checked 2026-10-12
CONDS = ["C0", "C1", "C2"]
N_DEV, N_DIS, R = 100, 50, 0.3
GATE_TEXTS = ROOT_DIR / "results" / "pretest_gate_format.jsonl"
_log_lock = threading.Lock()


def pinned_transport(payload: dict, api_key: str) -> dict:
    """OpenRouter call pinned to the OpenAI provider with seed 0; logs served model/provider/finish_reason."""
    payload = {**payload, "seed": 0, "provider": {"order": ["OpenAI"], "allow_fallbacks": False}}
    data = openrouter_transport(payload, api_key)
    choice = (data.get("choices") or [{}])[0]
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "requested": payload["model"], "served_model": data.get("model"),
           "provider": data.get("provider"), "finish_reason": choice.get("finish_reason"),
           "native_finish_reason": choice.get("native_finish_reason"),
           "system_fingerprint": data.get("system_fingerprint"), "max_tokens": payload.get("max_tokens"),
           "system_sha256": hashlib.sha256(payload["messages"][0]["content"].encode()).hexdigest()
           if payload["messages"] and payload["messages"][0]["role"] == "system" else None,
           "usage": data.get("usage")}
    with _log_lock, (OUT / "openrouter_responses.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return data


def config_with_compressor(model: str) -> dict:
    cfg = copy.deepcopy(load_models_config())
    cfg["roles"]["compressor"] = model
    cfg["roles"]["downstream"] = model
    cfg.setdefault("prices", {})[GPT] = GPT_PRICE
    cfg.setdefault("models", {})[GPT] = {"provider": "openrouter", "family": "openai"}
    return cfg


def make_client(mode: str, max_cost: float | None, cfg: dict) -> LLMClient:
    OUT.mkdir(parents=True, exist_ok=True)
    return LLMClient(mode=mode, max_cost_usd=max_cost, cache_dir=OUT / "cache", cost_log=OUT / "api_cost_log.jsonl",
                     models_config=cfg, transport=pinned_transport, concurrency=8)


def run_arm(model: str, mode: str, max_cost: float | None, jobs: list[tuple[str, str, int]], subdir: str):
    cfg = config_with_compressor(model)
    run.load_models_config = lambda *a, **k: cfg  # run_condition reads roles through this name
    client = make_client(mode, max_cost, cfg)
    try:
        for cond, split, n in jobs:
            run.run_condition(cond, split, R, client, limit=n, judge_role="opt_judge",
                              results_dir=OUT / subdir, downstream=False)
    finally:
        run.print_summary(client.summary())


def judge_gate_texts(mode: str):
    """phi3.5 dev[:100] x C0-C2 from the K5 gate (same ids as run.load_split('dev', limit=100)), CSR-judged here."""
    cfg = config_with_compressor("phi3.5")
    client = make_client(mode, 0, cfg)
    judge = CSRJudge(client, cfg["roles"]["opt_judge"])
    inst = {r["id"]: r for r in run.load_split("dev", limit=N_DEV)}
    gate = [json.loads(l) for l in GATE_TEXTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    (OUT / "phi3.5").mkdir(parents=True, exist_ok=True)
    for cond in CONDS:
        rows = [g for g in gate if g["condition"] == cond]
        assert [g["id"] for g in rows] == list(inst), f"gate ids differ from dev[:{N_DEV}] for {cond}"
        out = []
        for i, g in enumerate(rows, 1):
            j = judge.evaluate(inst[g["id"]], g["text"], g["over_budget"])
            out.append({"id": g["id"], "condition": cond, "split": "dev", "r": R, "compressed": g["text"],
                        "over_budget": g["over_budget"], "budget": g["budget"], "out_tokens": g["out_tokens"],
                        "judge": j, "csr": j["csr"], "signal": j.get("conflict_signaled"),
                        "source": "results/pretest_gate_format.jsonl (K5 gate, 2026-10-10)"})
            if i % 20 == 0:
                print(f"  phi3.5 {cond} judged {i}/{len(rows)}")
        if mode != "dry_run":
            (OUT / "phi3.5" / f"{cond}_dev_r{R}_s0.jsonl").write_text(
                "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out), encoding="utf-8")
    run.print_summary(client.summary())


# ---------------------------------------------------------------- report
def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def _row(recs: list[dict]) -> dict:
    n = len(recs)
    j = [r["judge"] for r in recs]
    k = sum(r["csr"] for r in recs)
    lo, hi = wilson_score_interval(k, n)  # already in percent
    return {"n": n, "csr": k, "ci": (lo, hi), "a": sum(x["a_supported"] for x in j), "b": sum(x["b_supported"] for x in j),
            "none": sum((not x["a_supported"]) and (not x["b_supported"]) for x in j),
            "signal": sum(bool(r.get("signal")) for r in recs), "over": sum(bool(r["over_budget"]) for r in recs),
            "conflict_prefix": sum(r["compressed"].lstrip().upper().startswith("CONFLICT") for r in recs),
            "out_tokens": sum(r["out_tokens"] for r in recs) / max(n, 1)}


def _length_cutoffs() -> dict:
    """gpt-4o-mini finish_reason == length per condition (by system-prompt hash; counts calls incl. retries)."""
    import yaml
    conds = yaml.safe_load(run.CONDITIONS_CONFIG.read_text(encoding="utf-8"))
    sha = {hashlib.sha256((ROOT_DIR / conds[c]["instruction"]).read_text(encoding="utf-8").strip().encode()).hexdigest(): c
           for c in CONDS}
    out: dict[str, list[int]] = {}
    for r in _load(OUT / "openrouter_responses.jsonl"):
        c = sha.get(r.get("system_sha256"))
        if c:
            tot, ln = out.get(c, [0, 0])
            out[c] = [tot + 1, ln + (r.get("finish_reason") == "length")]
    return out


def report() -> str:
    L = ["# 압축기 용량 탐색 결과 (gpt-4o-mini vs phi3.5, dev)", "",
         f"- 생성: {time.strftime('%Y-%m-%d %H:%M')} · 규칙: 「G1-lite 설명·해석·결정 기록」 §9 (실행 전 고정)",
         f"- r = {R}, judge = gemma2:9b(두 압축기 동일), 하류 생략. **dev 탐색, 정식 판정 미사용.**", ""]
    data = {}
    for model, sub in (("gpt-4o-mini", "gpt-4o-mini"), ("phi3.5", "phi3.5")):
        for c in CONDS:
            data[(model, c)] = _load(OUT / sub / f"{c}_dev_r{R}_s0.jsonl")
        data[(model, "C2-dis")] = _load(OUT / sub / f"C2_dev_distractor_r{R}_s0.jsonl")
    L += ["## 1. dev 100건 (충돌 문항)", "",
          "| 압축기 | 조건 | CSR k/n | % [Wilson 95%] | A 보존 | B 보존 | 둘 다 없음 | 충돌 표시 | `CONFLICT:` 시작 | 예산 초과 | 평균 출력 tok |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = {}
    for (model, c), recs in data.items():
        if c == "C2-dis" or not recs:
            continue
        s = rows[(model, c)] = _row(recs)
        L.append(f"| {model} | {c} | {s['csr']}/{s['n']} | {100*s['csr']/s['n']:.1f} [{s['ci'][0]:.1f}, {s['ci'][1]:.1f}] | "
                 f"{s['a']} | {s['b']} | {s['none']} | {s['signal']} | {s['conflict_prefix']} | {s['over']} | {s['out_tokens']:.0f} |")
    L += ["", "## 2. 같은 문항 짝 비교 (exact McNemar)", "", "| 비교 | 차이 %p | 앞만 1 / 뒤만 1 | p |", "|---|---|---|---|"]

    def pair(x, y, label):
        a, b = data.get(x) or [], data.get(y) or []
        if not a or not b:
            return
        bi = {r["id"]: r["csr"] for r in b}
        ids = [r["id"] for r in a if r["id"] in bi]
        va, vb = [next(r["csr"] for r in a if r["id"] == i) for i in ids], [bi[i] for i in ids]
        only_a = sum(1 for p, q in zip(va, vb) if p and not q)
        only_b = sum(1 for p, q in zip(va, vb) if q and not p)
        L.append(f"| {label} | {100*(sum(va)-sum(vb))/len(ids):+.1f} | {only_a} / {only_b} | {mcnemar_p(va, vb):.3g} |")

    for m in ("gpt-4o-mini", "phi3.5"):
        pair((m, "C1"), (m, "C0"), f"{m}: C1 − C0")
        pair((m, "C2"), (m, "C1"), f"{m}: C2 − C1")
    for c in CONDS:
        pair(("gpt-4o-mini", c), ("phi3.5", c), f"{c}: gpt-4o-mini − phi3.5")
    L += ["", f"## 3. dev_distractor {N_DIS}건 × C2 v0 (무충돌 문서)", "",
          "| 압축기 | n | `CONFLICT:` 시작 | FCR judge true (잠정, 정답 노출 judge) | 예산 초과 |", "|---|---|---|---|---|"]
    dis = {}
    for m in ("gpt-4o-mini", "phi3.5"):
        recs = data[(m, "C2-dis")]
        if recs:
            pre = sum(r["compressed"].lstrip().upper().startswith("CONFLICT") for r in recs)
            dis[m] = pre / len(recs)
            L.append(f"| {m} | {len(recs)} | {pre} ({100*pre/len(recs):.0f}%) | "
                     f"{sum(bool(r['false_conflict']) for r in recs)} | {sum(bool(r['over_budget']) for r in recs)} |")
    cut = _length_cutoffs()
    if cut:
        L += ["", "## 4. gpt-4o-mini 생성 길이 한도 종료 (OpenRouter finish_reason, 재시도 포함 호출 기준)", "",
              "| 조건 | 호출 | length 종료 |", "|---|---|---|"] + [f"| {c} | {t} | {l} |" for c, (t, l) in sorted(cut.items())]
    L += ["", "## 5. 사전 규칙 대입 (§9)", ""]
    g1, g2, p2 = rows.get(("gpt-4o-mini", "C1")), rows.get(("gpt-4o-mini", "C2")), rows.get(("phi3.5", "C2"))
    if g1 and g2 and p2 and "gpt-4o-mini" in dis:
        d = 100 * (g2["csr"] / g2["n"] - g1["csr"] / g1["n"])
        c1 = d >= -5
        c2 = g2["none"] <= p2["none"] / 2
        c3 = dis["gpt-4o-mini"] < 0.5
        r2 = (100 * g2["csr"] / g2["n"] < 10) or dis["gpt-4o-mini"] >= 0.8
        L += [f"- ① gpt-4o-mini C2 − C1 = {d:+.1f}%p (기준 ≥ −5): {'충족' if c1 else '미충족'}",
              f"- ② dev 둘 다 없음 gpt-4o-mini C2 {g2['none']} vs phi3.5 C2 {p2['none']} (기준 ≤ 절반): {'충족' if c2 else '미충족'}",
              f"- ③ dev_distractor `CONFLICT:` 시작 {100*dis['gpt-4o-mini']:.0f}% (기준 < 50%): {'충족' if c3 else '미충족'}",
              f"- 규칙 2 조건(C2 CSR < 10% 또는 `CONFLICT:` 시작 ≥ 80%): {'해당' if r2 else '해당 없음'}", ""]
        verdict = ("규칙 1 → 원인 = 압축기 용량" if (c1 and c2 and c3) else
                   "규칙 2 → 원인 = C2 v0 설계" if r2 else "규칙 3 → 두 원인 혼재")
        L.append(f"**판정: {verdict}**")
    else:
        L.append("(데이터 부족 — 판정 보류)")
    L += ["", "> 한계: n = 100(약 15%p 미만 차이는 탐색적), seed 1, judge는 phi3.5 C0·C1 출력으로만 검증(G0)."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["dry_run", "execute"], default="dry_run")
    ap.add_argument("--max-cost", type=float)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--only", choices=["gpt", "phi-judge", "phi-dis"], help="run one arm only")
    a = ap.parse_args(argv)
    if a.report:
        text = report()
        (OUT / "2026-10-12_explore_report.md").write_text(text, encoding="utf-8")
        print(text)
        return
    if a.mode == "execute" and a.max_cost is None:
        ap.error("--mode execute requires --max-cost")
    if a.only in (None, "gpt"):
        run_arm(GPT, a.mode, a.max_cost, [(c, "dev", N_DEV) for c in CONDS] + [("C2", "dev_distractor", N_DIS)],
                "gpt-4o-mini")
    if a.only in (None, "phi-judge"):
        judge_gate_texts(a.mode)
    if a.only in (None, "phi-dis"):
        run_arm("phi3.5", a.mode, 0 if a.mode == "execute" else None, [("C2", "dev_distractor", N_DIS)], "phi3.5")


if __name__ == "__main__":
    main()
