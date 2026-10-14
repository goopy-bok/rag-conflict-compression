"""
Statistical Analysis & Checkpoint 0 Audit for CCD-G1:
Computes Wilson 95% CI, McNemar exact test, Paired Transitions,
and extracts 10 CSR=0 cases for human verification.
"""
from __future__ import annotations
import json
import math
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = Path(os.environ.get("CCD_RESULTS_DIR") or ROOT_DIR / "results")


# 비율 k/n의 95% Wilson 신뢰구간을 % 단위(0~100)로 돌려준다. 조건별 CSR % [CI] 계산에 쓴다.
# confidence 인자는 받지만 쓰지 않는다(z = 1.96 고정).
def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # 95%
    p = k / n
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_std_error = math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower_bound = (centre_adjusted_probability - z * adjusted_std_error) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_std_error) / denominator
    return (max(0.0, lower_bound * 100), min(100.0, upper_bound * 100))


# 짝 비교의 엇갈린 칸 두 개(한쪽만 성공한 문항 수)로 정확 McNemar 양측 p값을 직접 계산한다.
# 초기 파일럿(analyze_pilot) 전용이며, 본 보고서는 statsmodels 판 mcnemar_p를 쓴다.
def mcnemar_exact_test(table_01: int, table_10: int) -> float:
    b, c = table_01, table_10
    total = b + c
    if total == 0:
        return 1.0
    k = min(b, c)
    prob_le_k = 0.0
    for i in range(k + 1):
        prob_le_k += math.comb(total, i) * (0.5 ** total)
    p_val = min(1.0, 2.0 * prob_le_k)
    return p_val


# 결과 .jsonl 파일 하나를 읽어 {문항 id: 레코드} 사전으로 돌려준다.
def load_results(path: Path) -> dict[str, dict]:
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data[item["id"]] = item
    return data


# 초기 dev_50 파일럿(C0 대 C1)을 분석해 화면에 출력한다: CSR과 Wilson CI, 2x2 짝 표와 McNemar p,
# 충돌 유형(explicit/implicit)별 비교, 사람 검토용 CSR=0 사례 10건. G1-lite·Phi-4 보고서에는 쓰지 않는다.
def analyze_pilot():
    c0_path = RESULTS_DIR / "c0_dev50_r0.3.jsonl"
    c1_path = RESULTS_DIR / "c1_dev50_r0.3.jsonl"

    if not c0_path.exists() or not c1_path.exists():
        print(f"Error: Results not found at {c0_path} or {c1_path}")
        return

    c0_data = load_results(c0_path)
    c1_data = load_results(c1_path)

    common_ids = sorted(list(set(c0_data.keys()) & set(c1_data.keys())))
    n = len(common_ids)
    print(f"\n=======================================================")
    print(f" CCD-G1 Step 0 Pilot Statistical Analysis (N={n})")
    print(f"=======================================================\n")

    # 조건 하나의 공통 문항에 대해 CSR·strict·lenient·예산 초과·재시도·Signal 건수와 CSR의 Wilson CI를 모은다.
    def get_stats(data_dict, name):
        csr_cnt = sum(data_dict[k]["csr"] for k in common_ids)
        strict_cnt = sum(data_dict[k]["csr_strict"] for k in common_ids)
        lenient_cnt = sum(data_dict[k]["csr_lenient"] for k in common_ids)
        over_cnt = sum(1 for k in common_ids if data_dict[k]["over_budget"])
        retry_cnt = sum(1 for k in common_ids if data_dict[k]["retried"])
        sig_cnt = sum(1 for k in common_ids if data_dict[k]["signal"])

        csr_rate = csr_cnt / n * 100
        csr_ci = wilson_score_interval(csr_cnt, n)

        return {
            "name": name,
            "csr_cnt": csr_cnt,
            "csr_rate": csr_rate,
            "csr_ci": csr_ci,
            "strict_cnt": strict_cnt,
            "lenient_cnt": lenient_cnt,
            "over_cnt": over_cnt,
            "retry_cnt": retry_cnt,
            "sig_cnt": sig_cnt,
        }

    c0_s = get_stats(c0_data, "C0 Naive")
    c1_s = get_stats(c1_data, "C1 Manual-Plain")

    print(f"1. Overall Comparison Table (Budget r=0.3, N={n}):")
    print(f"   - C0 Naive:        CSR {c0_s['csr_rate']:.1f}% ({c0_s['csr_cnt']}/{n}) [95% CI: {c0_s['csr_ci'][0]:.1f}, {c0_s['csr_ci'][1]:.1f}]")
    print(f"     Strict: {c0_s['strict_cnt']}/{n} ({c0_s['strict_cnt']/n*100:.1f}%), Lenient: {c0_s['lenient_cnt']}/{n} ({c0_s['lenient_cnt']/n*100:.1f}%), Over-budget: {c0_s['over_cnt']}/{n}, Retried: {c0_s['retry_cnt']}/{n}")
    print(f"   - C1 Manual-Plain: CSR {c1_s['csr_rate']:.1f}% ({c1_s['csr_cnt']}/{n}) [95% CI: {c1_s['csr_ci'][0]:.1f}, {c1_s['csr_ci'][1]:.1f}]")
    print(f"     Strict: {c1_s['strict_cnt']}/{n} ({c1_s['strict_cnt']/n*100:.1f}%), Lenient: {c1_s['lenient_cnt']}/{n} ({c1_s['lenient_cnt']/n*100:.1f}%), Over-budget: {c1_s['over_cnt']}/{n}, Retried: {c1_s['retry_cnt']}/{n}")

    diff = c1_s["csr_rate"] - c0_s["csr_rate"]
    print(f"   => Effect Size (Delta): {diff:+.1f}%p")

    # Contingency table: C0 vs C1
    # C0=1, C1=1 (a); C0=1, C1=0 (b); C0=0, C1=1 (c); C0=0, C1=0 (d)
    a = sum(1 for k in common_ids if c0_data[k]["csr"] == 1 and c1_data[k]["csr"] == 1)
    b = sum(1 for k in common_ids if c0_data[k]["csr"] == 1 and c1_data[k]["csr"] == 0)
    c = sum(1 for k in common_ids if c0_data[k]["csr"] == 0 and c1_data[k]["csr"] == 1)
    d = sum(1 for k in common_ids if c0_data[k]["csr"] == 0 and c1_data[k]["csr"] == 0)

    p_mcnemar = mcnemar_exact_test(b, c)
    print(f"\n2. Paired 2x2 Contingency Table:")
    print(f"               C1=1 (Survive)   C1=0 (Loss)")
    print(f"   C0=1 (Survive)    {a:2d} (Both 1)       {b:2d} (C0 only)")
    print(f"   C0=0 (Loss)       {c:2d} (C1 only)      {d:2d} (Both 0)")
    print(f"   Exact McNemar Test p-value: {p_mcnemar:.5f} ({'p < 0.05' if p_mcnemar < 0.05 else 'not significant'})\n")

    # Breakdown by conflict type
    print("3. Breakdown by Conflict Type:")
    for ctype in ["explicit", "implicit"]:
        sub_ids = [k for k in common_ids if c0_data[k]["conflict_type"] == ctype]
        sub_n = len(sub_ids)
        c0_sub = sum(c0_data[k]["csr"] for k in sub_ids)
        c1_sub = sum(c1_data[k]["csr"] for k in sub_ids)
        c0_r = c0_sub / sub_n * 100 if sub_n else 0
        c1_r = c1_sub / sub_n * 100 if sub_n else 0
        sub_b = sum(1 for k in sub_ids if c0_data[k]["csr"] == 1 and c1_data[k]["csr"] == 0)
        sub_c = sum(1 for k in sub_ids if c0_data[k]["csr"] == 0 and c1_data[k]["csr"] == 1)
        sub_p = mcnemar_exact_test(sub_b, sub_c)
        print(f"   - {ctype.upper()} (n={sub_n}):")
        print(f"     C0: {c0_r:.1f}% ({c0_sub}/{sub_n}) -> C1: {c1_r:.1f}% ({c1_sub}/{sub_n}) [Delta: {c1_r - c0_r:+.1f}%p, McNemar p={sub_p:.4f}]")

    # Checkpoint 0: Extract 10 cases of loss (CSR=0) for human inspection
    loss_cases = [k for k in common_ids if c0_data[k]["csr"] == 0 or c1_data[k]["csr"] == 0]
    print(f"\n=======================================================")
    print(f" 4. Checkpoint 0: Human Audit on 10 Loss Cases (CSR=0)")
    print(f"    Total cases with at least one CSR=0: {len(loss_cases)}")
    print(f"=======================================================\n")

    sample_cases = loss_cases[:10]
    audit_records = []
    for idx, cid in enumerate(sample_cases, start=1):
        item_c0 = c0_data[cid]
        item_c1 = c1_data[cid]
        print(f"--- [Audit Case #{idx}: {cid}] ({item_c0['conflict_type'].upper()}) ---")
        print(f"Question: {item_c0['question']}")
        print(f"Gold A:   {item_c0['gold']['answer_a']}")
        print(f"Gold B:   {item_c0['gold']['answer_b']}")
        print(f"C0 Comp:  {item_c0['compressed']}")
        print(f"C0 Judge: CSR={item_c0['csr']} | a_sup={item_c0['judge']['a_supported']}, b_sup={item_c0['judge']['b_supported']} | {item_c0['judge']['rationale']}")
        print(f"C1 Comp:  {item_c1['compressed']}")
        print(f"C1 Judge: CSR={item_c1['csr']} | a_sup={item_c1['judge']['a_supported']}, b_sup={item_c1['judge']['b_supported']} | {item_c1['judge']['rationale']}\n")

        audit_records.append({
            "id": cid,
            "conflict_type": item_c0["conflict_type"],
            "question": item_c0["question"],
            "gold_a": item_c0["gold"]["answer_a"],
            "gold_b": item_c0["gold"]["answer_b"],
            "c0_csr": item_c0["csr"],
            "c0_rationale": item_c0["judge"]["rationale"],
            "c1_csr": item_c1["csr"],
            "c1_rationale": item_c1["judge"]["rationale"],
        })

    return {
        "n": n,
        "c0": c0_s,
        "c1": c1_s,
        "delta": diff,
        "mcnemar_p": p_mcnemar,
        "contingency": {"a": a, "b": b, "c": c, "d": d},
        "audit_sample": audit_records,
    }


# ------------------------------------------------- G1 main analysis (guide v2 s9)
# References: CAR-exp analyze_rq1.py (boot_ci_diff, holm, mcnemar_rows),
# run_judge.py (kappa), analyze_agent.py (mcnemar).


# 같은 문항에 대한 두 조건 점수(a, b)의 차이 평균과, 문항을 복원추출로 n번 다시 뽑아 얻은
# 95% 백분위 부트스트랩 신뢰구간을 돌려준다. H1·H2 짝 비교의 "차이 %p [95% CI]"가 이 값이다.
def paired_bootstrap(a, b, n: int = 10_000, seed: int = 0) -> tuple[float, tuple[float, float]]:
    """Mean of per-instance differences a - b and its 95% percentile bootstrap CI."""
    import numpy as np
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), (float(lo), float(hi))


# 같은 문항의 두 조건 0/1 결과로 2x2 표를 만들고 정확(이항) McNemar p값을 돌려준다. H1·H2의 p값이다.
def mcnemar_p(a, b) -> float:
    """Exact (binomial) McNemar p-value on paired binary outcomes."""
    from statsmodels.stats.contingency_tables import mcnemar
    import numpy as np
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    table = [[int((a & b).sum()), int((a & ~b).sum())], [int((~a & b).sum()), int((~a & ~b).sum())]]
    return float(mcnemar(table, exact=True).pvalue)


# 여러 가설 p값을 Holm 방식으로 보정해 입력 순서 그대로 돌려준다(다중비교 보정, 보고서의 "Holm p").
def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in input order."""
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda i: pvals[i])):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


# 두 비율 p1, p2 사이의 효과 크기 Cohen's h(아크사인 변환 차이)를 돌려준다. lite_report에서는 쓰지 않는다.
def cohens_h(p1: float, p2: float) -> float:
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


# 두 채점자(judge 또는 사람)가 같은 문항에 매긴 라벨의 일치도 Cohen's kappa를 돌려준다. judge 검증(G0)용.
def cohen_kappa(x, y, labels=None) -> float:
    """Cohen's kappa for two raters over the same items."""
    x, y = list(x), list(y)
    labels = list(labels) if labels is not None else sorted(set(x) | set(y), key=str)
    n = len(x)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(x, y)) / n
    pe = sum((x.count(l) / n) * (y.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


# 여러 seed 결과를 문항별로 평균 낸다(3 seed면 0, 1/3, 2/3, 1). 모든 seed에 있는 문항만 남긴다.
# 정식 G1의 3 seed 분석용이며, seed 1개인 현재 보고서에서는 쓰지 않는다.
def seed_mean_per_instance(records_by_seed: dict, key: str = "csr") -> dict[str, float]:
    """{seed: {id: record or 0/1}} -> {id: mean over seeds} (0, 1/3, 2/3, 1 for 3 seeds).
    Only ids present in every seed are kept."""
    seeds = list(records_by_seed.values())
    common = set.intersection(*(set(s) for s in seeds)) if seeds else set()
    val = lambda v: v[key] if isinstance(v, dict) else v
    return {i: sum(val(s[i]) for s in seeds) / len(seeds) for i in sorted(common)}


# McNemar 짝 비교에서 효과 delta를 검정력 power로 검출하려면 문항이 몇 개 필요한지 계산한다(표본 크기 설계용).
def required_n_mcnemar(p_discordant: float, delta: float = 0.05, alpha: float = 0.05, power: float = 0.8) -> int:
    """Guide v2 s4: n = (z_{1-alpha/2} + z_{power})^2 * p_d / delta^2, rounded up."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return math.ceil(z * z * p_discordant / delta ** 2)


# ------------------------------------------------- G1-lite report (preregistration section 7)

LITE_CONDS = ["C0", "C1", "C2", "C3-G1", "C3-G"]
LITE_SPLITS = ["test_id", "test_ood1", "test_ood2", "distractor"]
LITE_PAIRED = [("H1", "C1", "C0", 0.10), ("H2a", "C3-G1", "C1", 0.05), ("H2b", "C3-G", "C2", 0.05)]


# 서로 다른 문항 집합 a, b(예: test_id 대 OOD)를 각각 따로 복원추출해 평균 차이의 95% 신뢰구간과
# 양측 부트스트랩 p값을 돌려준다. H4의 "하락 %p [95% CI]"와 bootstrap p가 이 값이다.
def unpaired_bootstrap(a, b, n: int = 10_000, seed: int = 0) -> tuple[float, tuple[float, float], float]:
    """mean(a) - mean(b) for independent samples: point, 95% percentile CI, two-sided bootstrap p."""
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = a[rng.integers(0, len(a), (n, len(a)))].mean(1) - b[rng.integers(0, len(b), (n, len(b)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return float(a.mean() - b.mean()), (float(lo), float(hi)), float(p)


# n번 중 k번 성공이 귀무가설 비율 p0에서 나올 정확 이항검정 p값을 돌려준다. H3(CSR < 0.9)의 p값이다.
def binom_p(k: int, n: int, p0: float, alternative: str = "less") -> float:
    """Exact binomial test p-value for k successes in n trials under H0 rate p0 (amendment 2.2-i, H3)."""
    from scipy.stats import binomtest
    return float(binomtest(k, n, p0, alternative=alternative).pvalue)


# 독립된 두 표본(k1/n1, k2/n2)의 비율 차이에 대한 양측 Fisher 정확검정 p값을 돌려준다. H4 민감도 분석용.
def fisher_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p for two independent binomial samples (amendment 2.2-i, H4 sensitivity)."""
    from scipy.stats import fisher_exact
    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


# 결과 폴더에서 조건 × 시험지 조합의 결과 파일을 읽어 {(조건, 시험지): {id: 레코드}}로 돌려준다.
# 파일이 없는 조합은 건너뛴다.
def _lite_load(results_dir: Path, r: float, seed: int, conds=LITE_CONDS) -> dict:
    out = {}
    for c in conds:
        for sp in LITE_SPLITS:
            p = Path(results_dir) / f"{c}_{sp}_r{r}_s{seed}.jsonl"
            if p.exists():
                out[(c, sp)] = load_results(p)
    return out


# 0~1 비율을 소수 첫째 자리 % 문자열로 바꾼다(None이면 "-").
def _pct(x) -> str:
    return "-" if x is None else f"{x * 100:.1f}"


# k/n의 Wilson 95% 신뢰구간을 "[하한, 상한]" 문자열로 만든다(보고서 표 출력용).
def _ci(k: int, n: int) -> str:
    lo, hi = wilson_score_interval(k, n)
    return f"[{lo:.1f}, {hi:.1f}]"


# g0_dir의 G0 결과 파일(g0_auto.json, agreement.json)을 보고 judge 검증 상태 문장을 만든다.
# 파일이 없으면 "미실행"이며, 보고서 머리말의 "judge 검증(G0)" 줄이 이 값이다.
def g0_status_line(g0_dir: Path) -> str:
    """Judge-validation status for the report header (amendment 2.2-d: the author part of G0 runs last)."""
    # G0 결과 파일 하나의 상태: 없으면 "미실행", pass가 참이면 "통과", 아니면 "실패".
    def state(name: str) -> str:
        p = g0_dir / name
        return "미실행" if not p.exists() else ("통과" if json.loads(p.read_text(encoding="utf-8")).get("pass") else "실패")
    auto, human = state("g0_auto.json"), state("agreement.json")
    line = f"**judge 검증(G0)**: 라벨 없는 기준(κ·음성 대조) {auto} · 저자 라벨 대비 정확도 {human}."
    if human != "통과":
        line += (" 저자 라벨 대비 정확도가 확인되기 전까지 CSR·FCR·D는 judge 타당성이 미확정인 값이다"
                 " (개정 2.2-d).")
    return line


# 결과 파일들로 예비 결과 보고서(마크다운 문자열)를 만든다. FINAL_BENCHMARK_REPORT.md가 이 출력이다.
# 1절 조건별 기술 통계, 2절 짝 비교(H1·H2), 3절 H3, 4절 H4, 5절 유의 사항 순이며,
# 계산한 가설 p값 전체에 Holm 보정을 적용한다.
def lite_report(results_dir: Path = RESULTS_DIR, r: float = 0.3, seed: int = 0, n_boot: int = 10_000,
                conds=LITE_CONDS, paired=LITE_PAIRED, opt_conds=("C3-G",), label: str = "G1-lite") -> str:
    """G1-lite markdown report: descriptive table + H1, H2a, H2b (paired), H3 (one-sided), H4 (unpaired),
    Holm over every computed hypothesis p-value. conds/paired/opt_conds (H3/H4 targets) default to G1-lite;
    with several opt_conds the H3/H4 names carry the condition (H3-C3-P, H4-C3-P-test_ood1)."""
    import datetime as dt
    import numpy as np
    data = _lite_load(results_dir, r, seed, conds)
    g1lite = list(opt_conds) == ["C3-G"]  # G1-lite-specific amendment notes only apply to that run
    multi = len(opt_conds) > 1
    L = [f"# {label} 예비 결과 ({dt.date.today().isoformat()})", "",
         f"**예비 결과({label})이며 정식 G1 판정에 쓰지 않는다** (사전등록 7절). "
         f"r = {r}, seed {seed} 한 개이므로 seed 간 분산은 보고하지 않는다. Test-ID가 200건이라 검정력이 낮고 "
         "신뢰구간이 넓다. 결과는 효과의 방향과 크기를 짐작하는 근거로만 쓴다.", "",
         "CSR은 예산 규칙 적용 값(예산 초과 = 0)이다. CI는 비율은 Wilson 95%, 차이는 bootstrap 95%"
         f"({n_boot}회, seed 0)이다. 보정은 계산된 가설 p값 전체에 대한 Holm이다.", "",
         g0_status_line(results_dir / "g0_local"), "",
         "## 1. 조건별 기술 통계", "",
         "| split | 조건 | n | CSR k/n | CSR % [95% CI] | 예산 초과 | FCR k/n (distractor) | D k/n (Test-ID) |",
         "|---|---|---|---|---|---|---|---|"]
    for sp in LITE_SPLITS:
        for c in conds:
            recs = data.get((c, sp))
            if recs is None:
                L.append(f"| {sp} | {c} | 결과 없음 | | | | | |")
                continue
            vals = list(recs.values())
            n = len(vals)
            over = sum(bool(v.get("over_budget")) for v in vals)
            conf = [v for v in vals if v.get("csr") is not None]
            k = sum(v["csr"] for v in conf)
            csr = f"{k}/{len(conf)} | {_pct(k / len(conf))} {_ci(k, len(conf))}" if conf else "- | -"
            fc = [v for v in vals if v.get("false_conflict") is not None]
            fcr = f"{sum(v['false_conflict'] for v in fc)}/{len(fc)}" if fc else "-"
            ds = [v["downstream"]["D"] for v in vals if (v.get("downstream") or {}).get("D") is not None]
            dtxt = f"{sum(ds)}/{len(ds)}" if ds else "-"
            L.append(f"| {sp} | {c} | {n} | {csr} | {over} | {fcr} | {dtxt} |")

    # (조건, 시험지)의 {문항 id: CSR 0/1}. CSR이 없는 레코드(distractor 등)는 뺀다.
    def csr_map(c, sp):
        recs = data.get((c, sp)) or {}
        return {i: v["csr"] for i, v in recs.items() if v.get("csr") is not None}

    hyp = []  # (hypothesis, raw p) for Holm
    L += ["", f"## 2. 짝 비교 (Test-ID, 같은 인스턴스): {', '.join(n for n, *_ in paired)}", "",
          "| 가설 | 비교 | n | k(앞) | k(뒤) | 앞만 1 | 뒤만 1 | 차이 %p [95% CI] | McNemar p | Holm p | 사전 기준 | 기준 충족 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    paired_rows = []
    for name, a, b, thr in paired:
        ma, mb = csr_map(a, "test_id"), csr_map(b, "test_id")
        ids = sorted(set(ma) & set(mb))
        if not ids:
            paired_rows.append((name, a, b, thr, None))
            continue
        xa, xb = [ma[i] for i in ids], [mb[i] for i in ids]
        diff, (lo, hi) = paired_bootstrap(xa, xb, n=n_boot)
        p = mcnemar_p(xa, xb)
        only_a = sum(1 for x, y in zip(xa, xb) if x and not y)
        only_b = sum(1 for x, y in zip(xa, xb) if y and not x)
        paired_rows.append((name, a, b, thr, dict(n=len(ids), ka=sum(xa), kb=sum(xb), oa=only_a, ob=only_b,
                                                    diff=diff, lo=lo, hi=hi, p=p)))
        hyp.append((name, p))

    # H3: C3-G Test-ID CSR < 0.9. Amendment 2.2-i (post hoc, 2026-10-12): the p-value is the exact one-sided
    # binomial test under H0 p = 0.9. The bootstrap of the observed sample only gives the descriptive 95% upper
    # bound; its tail mass at 0.9 is not a null-hypothesis p-value (it was reported as one before 2.2-i).
    h3s = []  # (name, condition, stats or None)
    for oc in opt_conds:
        name = f"H3-{oc}" if multi else "H3"
        m = csr_map(oc, "test_id")
        if not m:
            h3s.append((name, oc, None))
            continue
        x = np.asarray(list(m.values()), float)
        boots = x[np.random.default_rng(0).integers(0, len(x), (n_boot, len(x)))].mean(1)
        h3 = dict(n=len(x), k=int(x.sum()), mean=float(x.mean()), up=float(np.percentile(boots, 95)),
                  p_boot=float((boots >= 0.9).mean()), p=binom_p(int(x.sum()), len(x), 0.9))
        h3s.append((name, oc, h3))
        hyp.append((name, h3["p"]))

    # H4: optimized condition Test-ID vs OOD-1 / OOD-2, unpaired
    h4 = []
    for oc in opt_conds:
        for sp in ("test_ood1", "test_ood2"):
            name = f"H4-{oc}-{sp}" if multi else f"H4-{sp}"
            mi, mo = csr_map(oc, "test_id"), csr_map(oc, sp)
            if mi and mo:
                diff, (lo, hi), p = unpaired_bootstrap(list(mi.values()), list(mo.values()), n=n_boot)
                h4.append((name, sp, dict(ni=len(mi), ki=sum(mi.values()), no=len(mo), ko=sum(mo.values()),
                                          diff=diff, lo=lo, hi=hi, p=p,
                                          p_fisher=fisher_p(sum(mi.values()), len(mi), sum(mo.values()), len(mo)))))
                hyp.append((name, p))
            else:
                h4.append((name, sp, None))

    adj = dict(zip([h for h, _ in hyp], holm([p for _, p in hyp])))
    # Amendment 2.2-i sensitivity: the same Holm family with Fisher's exact p in place of the H4 bootstrap p
    fisher_by_name = {name: v["p_fisher"] for name, _, v in h4 if v is not None}
    hyp_f = [(h, fisher_by_name.get(h, p)) for h, p in hyp]
    adj_f = dict(zip([h for h, _ in hyp_f], holm([p for _, p in hyp_f])))
    for name, a, b, thr, v in paired_rows:
        if v is None:
            L.append(f"| {name} | {a} vs {b} | 결과 없음 | | | | | | | | | |")
            continue
        ok = v["diff"] >= thr and v["lo"] > 0
        L.append(f"| {name} | {a} − {b} | {v['n']} | {v['ka']} | {v['kb']} | {v['oa']} | {v['ob']} | "
                 f"{v['diff'] * 100:+.1f} [{v['lo'] * 100:+.1f}, {v['hi'] * 100:+.1f}] | {v['p']:.4f} | "
                 f"{adj[name]:.4f} | 차이 ≥ {thr * 100:.0f}%p, CI 하한 > 0 | {'예' if ok else '아니오'} |")

    L += ["", f"## 3. H3: {'·'.join(opt_conds)}의 Test-ID CSR < 90% (단측 정확 이항검정, 개정 2.2-i)", ""]
    for name, oc, h3 in h3s:
        if h3 is None:
            L.append(f"- 결과 없음 ({oc} Test-ID).")
            continue
        ok = h3["up"] < 0.9
        L.append(f"- {oc + ': ' if multi else ''}n = {h3['n']}, k = {h3['k']}, CSR = {h3['mean'] * 100:.1f}% "
                 f"{_ci(h3['k'], h3['n'])}, bootstrap 단측 95% 상한 = {h3['up'] * 100:.1f}%, "
                 f"정확 이항검정 p(H0: CSR = 0.9) = {h3['p']:.3g}, Holm p = {adj[name]:.3g}. "
                 f"기준(상한 < 90%) 충족: {'예' if ok else '아니오'}.")
        if g1lite:
            L.append(f"- 개정 2.2-i(사후): 이전 보고서는 관측 표본 bootstrap의 꼬리 비율({h3['p_boot']:.4f})을 p값으로 "
                     "적었다. 이는 귀무가설 분포가 아니라 잘못된 값이며, 정확 이항검정으로 바꿨다. 방향·판정은 같다.")
            L.append("- 정식 계획의 H3는 C3-G·C3-M·C3-P 세 최적화기 모두를 요구한다. 축소판은 GEPA 하나로만 판정하므로 "
                     "최적화기 의존성은 다루지 못한다.")

    L += ["", f"## 4. H4: {'·'.join(opt_conds)}의 Test-ID 대 OOD (비짝 bootstrap; Fisher 정확검정은 민감도 분석, 개정 2.2-i)", "",
          "| 가설 | 비교 | n(ID) | k(ID) | n(OOD) | k(OOD) | 하락 %p [95% CI] | bootstrap p | Holm p | Fisher p | Holm p (Fisher) | 기준 충족(하락 ≥ 10%p) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, sp, v in h4:
        if v is None:
            L.append(f"| {name} | test_id vs {sp} | 결과 없음 | | | | | | | | | |")
            continue
        L.append(f"| {name} | test_id − {sp} | {v['ni']} | {v['ki']} | {v['no']} | {v['ko']} | "
                 f"{v['diff'] * 100:+.1f} [{v['lo'] * 100:+.1f}, {v['hi'] * 100:+.1f}] | {v['p']:.4f} | "
                 f"{adj[name]:.4f} | {v['p_fisher']:.4f} | {adj_f[name]:.4f} | {'예' if v['diff'] >= 0.10 else '아니오'} |")
    if any(v for _, _, v in h4):
        L.append("")
        L.append("- 개정 2.2-i(사후 민감도): 희소 성공 건수에서는 bootstrap p와 정확검정 p가 다를 수 있다. Fisher p로 "
                 "같은 Holm 보정을 적용한 값을 함께 적는다. OOD2는 이전 파일럿(dev_50)과 핵심 문서가 겹치는 문항이 "
                 "45/154건 있고 87개 기사에서 나와(기사당 최대 11문항) 문항 독립 가정이 약하다. H4-OOD2의 유의성은 "
                 "확정하지 않는다.")

    L += ["", "## 5. 해석 시 유의 사항", "",
          f"- {'H2b' if g1lite else '짝 비교'}의 C2는 튜닝하지 않은 v0이다. '수동 튜닝 최선'과의 비교가 아니다.",
          "- 기준 충족 열은 점추정·CI를 사전 기준에 기계적으로 대응시킨 것이며, 예비 결과이므로 가설 판정이 아니다.",
          "- OOD-1은 앞 100건, distractor는 앞 100건, OOD-2는 전량이다. OOD·distractor에서는 하류 생성과 D를 측정하지 "
          "않았다(Test-ID만 측정).",
          f"- Holm 보정 대상: {', '.join(adj) or '없음'}."]
    if g1lite:
        L.append("- 개정 2.2-i(사후, 2026-10-12): H3 p값을 정확 이항검정으로 바로잡았고, H4에 Fisher 민감도 열을 더했다. "
                 "FCR(distractor)은 judge 오판 사례가 확인돼(C2 28건 중 최소 6건이 지침과 어긋남) 잠정치로 다룬다. "
                 "C2의 저장 압축문은 정규화(예시 틀 절단) 뒤의 텍스트이며, 원생성은 200건 중 86건이 생성 길이 한도에서 "
                 "끊겼다(C0·C1·C3-G1은 0건, C3-G 3건). 보조 지표 csr_strict의 문자열 매칭은 느슨하다(주 지표 csr과 무관).")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="pilot analysis (default) or the G1-lite report")
    ap.add_argument("cmd", nargs="?", choices=["pilot", "lite"], default="pilot")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--out", help="lite: output markdown (default reports/<date>_g1_lite_results.md)")
    ap.add_argument("--conds", nargs="+", default=LITE_CONDS, help="lite: conditions in the descriptive table")
    ap.add_argument("--opt-conds", nargs="+", default=["C3-G"], help="lite: H3/H4 target conditions")
    ap.add_argument("--paired", nargs="+", default=None, metavar="NAME:A:B:THR",
                    help="lite: paired Test-ID comparisons, e.g. H1:C1:C0:0.10 (default: G1-lite H1/H2a/H2b)")
    ap.add_argument("--label", default="G1-lite", help="lite: run label in the title")
    a = ap.parse_args()
    if a.cmd == "pilot":
        analyze_pilot()
    else:
        import datetime as dt
        paired = LITE_PAIRED if a.paired is None else [
            (n, x, y, float(t)) for n, x, y, t in (s.split(":") for s in a.paired)]
        out = Path(a.out or ROOT_DIR / "reports" / f"{dt.date.today().isoformat()}_g1_lite_results.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(lite_report(Path(a.results_dir), conds=a.conds, paired=paired, opt_conds=a.opt_conds,
                                   label=a.label), encoding="utf-8")
        print(f"wrote {out}")
