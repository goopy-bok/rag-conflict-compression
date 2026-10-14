"""
Dataset Builder for CCD-G1:
Constructs common schema instances from WikiContradict benchmark.
Attaches 3 non-overlapping distractors per instance, shuffles document positions,
and computes total tokens using cl100k_base.
"""
from __future__ import annotations
import csv
import json
import os
import random
import sys
from pathlib import Path

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from budget import count_tokens

WIKI_CSV = os.environ.get("CCD_WIKI_CSV", str(ROOT_DIR / "data" / "raw" / "wikicontradict_all.csv"))
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
DEV_50_PATH = OUTPUT_DIR / "dev_50.jsonl"


# v1 파일럿용 dev_50.jsonl(WikiContradict 50건)을 만든다. 재현을 위해 그대로 둔 함수이며 v2 분할과는 별개다.
# Explicit 25건 + Implicit 25건을 seed로 섞어 고르고, 다른 article의 context 3개를 무작위로 방해 문서로 붙인 뒤 순서를 섞는다.
# 방해 문서는 BM25가 아니라 무작위로 고르며, 정답 문자열 포함 여부는 거르지 않는다. 입력 행 사전에 _ctype 키를 덧붙인다.
# 반환: 출력 경로 DEV_50_PATH. 이 파일의 행은 build_wikicontradict에서 test_ood2 제외 키로 다시 쓰인다.
def build_dev_50(seed: int = 42) -> Path:
    rng = random.Random(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(WIKI_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Pool of candidate distractors (all contexts with distinct article titles)
    article_contexts = {}
    for r in rows:
        title = r.get("WikipediaArticleTitle", "").strip()
        c1 = r.get("context1", "").strip()
        c2 = r.get("context2", "").strip()
        if title:
            if title not in article_contexts:
                article_contexts[title] = []
            if c1:
                article_contexts[title].append(c1)
            if c2:
                article_contexts[title].append(c2)

    # Filter valid instances
    explicit = [
        r for r in rows
        if r.get("contradictType") == "Explicit"
        and r.get("answer1") and r.get("answer2")
        and r.get("context1") and r.get("context2")
        and r["answer1"].strip().lower() != r["answer2"].strip().lower()
    ]
    implicit = [
        r for r in rows
        if r.get("contradictType") == "Implicit (reasoning required)"
        and r.get("answer1") and r.get("answer2")
        and r.get("context1") and r.get("context2")
        and r["answer1"].strip().lower() != r["answer2"].strip().lower()
    ]

    rng.shuffle(explicit)
    rng.shuffle(implicit)

    # Select 25 explicit + 25 implicit
    selected_rows = []
    for r in explicit[:25]:
        r["_ctype"] = "explicit"
        selected_rows.append(r)
    for r in implicit[:25]:
        r["_ctype"] = "implicit"
        selected_rows.append(r)

    # Build 50 instances
    instances = []
    all_titles = list(article_contexts.keys())

    for idx, r in enumerate(selected_rows, start=1):
        inst_id = f"dev_{idx:03d}"
        inst_title = r.get("WikipediaArticleTitle", "").strip()
        question = r["question"].strip()
        ans_a = r["answer1"].strip()
        ans_b = r["answer2"].strip()
        ctx_a = r["context1"].strip()
        ctx_b = r["context2"].strip()
        ctype = r["_ctype"]

        # Select 3 distractors from titles different from inst_title
        other_titles = [t for t in all_titles if t.lower() != inst_title.lower()]
        distractor_titles = rng.sample(other_titles, 3)
        distractors = [rng.choice(article_contexts[t]) for t in distractor_titles]

        # Combine docs: conflict_a, conflict_b, and 3 distractors
        raw_docs = [
            {"text": ctx_a, "role": "conflict_a"},
            {"text": ctx_b, "role": "conflict_b"},
            {"text": distractors[0], "role": "distractor"},
            {"text": distractors[1], "role": "distractor"},
            {"text": distractors[2], "role": "distractor"},
        ]

        # Shuffle document order with fixed seed progression
        rng.shuffle(raw_docs)

        # Assign doc_id [d0]..[d4] and find conflict positions
        docs = []
        conflict_positions = []
        formatted_docs_text = ""
        for d_idx, doc in enumerate(raw_docs):
            doc_id = f"d{d_idx}"
            docs.append({
                "doc_id": doc_id,
                "text": doc["text"],
                "role": doc["role"]
            })
            if doc["role"] in ("conflict_a", "conflict_b"):
                conflict_positions.append(d_idx)
            formatted_docs_text += f"[{doc_id}] {doc['text']}\n\n"

        total_tokens = count_tokens(formatted_docs_text.strip())

        instance = {
            "id": inst_id,
            "source": "wikicontradict",
            "conflict_type": ctype,
            "article_title": inst_title,
            "question": question,
            "docs": docs,
            "gold": {
                "answer_a": ans_a,
                "answer_b": ans_b,
                "span_a": ans_a,
                "span_b": ans_b
            },
            "conflict_positions": sorted(conflict_positions),
            "total_tokens": total_tokens
        }
        instances.append(instance)

    with open(DEV_50_PATH, "w", encoding="utf-8") as f:
        for inst in instances:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(instances)} instances to {DEV_50_PATH}")
    print(f"   Avg tokens per instance: {sum(x['total_tokens'] for x in instances)/len(instances):.1f}")
    return DEV_50_PATH


# =============================================================================
# v2 builders: ConflictBank dev / test_id_pool / test_ood1 / distractor sets,
# WikiContradict test_ood2, sealing. build_dev_50 above is kept as-is for reproduction.
# =============================================================================
import collections
import datetime as dt
import hashlib

import numpy as np

from distractors import DistractorPool
from seal import seal_files
from sources import conflictbank as cb
from sources import wikicontradict as wc

CB_RAW = ROOT_DIR / "data" / "raw" / "conflictbank" / "QA_dataset.json"
CB_INDEX_CACHE = ROOT_DIR / "data" / "raw" / "conflictbank" / "cb_index.json"
SEALED_DIR = ROOT_DIR / "data" / "sealed"

# split -> (size, conflict type); filled in this order from one shuffled subject stream
SPLITS = {
    "dev": (300, "misinformation"),
    "dev_distractor": (100, "none"),
    "test_id_pool": (2000, "misinformation"),
    "test_ood1": (300, "temporal"),
    "distractor": (300, "none"),
}
DEV_SIDE = {"dev", "dev_distractor"}  # these draw distractor docs from the dev-side pool only
# Per-evidence cl100k token window. Plan v2 said 100-400, but on a 20k-record sample only 1.33% of
# misinformation pairs fit (~7.4k of 553k < 3 x 2,700 needed); evidence median is ~485 tokens.
# 300-600 keeps 95.8% of pairs and only drops the short tail (upper bound not binding: max 560).
TOKEN_WINDOW = (300, 600)
N_DISTRACTORS = (3, 6)


# 문서 목록을 '[d0] text\n\n...' 형태로 이어 붙여 양끝 공백을 지운 뒤 cl100k 토큰 수를 센다.
# compressors/llm.py의 docs_block과 같은 렌더링이므로, 이 값이 압축 예산 계산의 total_tokens가 된다.
def docs_total_tokens(docs: list[dict]) -> int:
    """Same document rendering as compressors.format_input: '[d0] text\\n\\n...' stripped."""
    return count_tokens("".join(f"[{d['doc_id']}] {d['text']}\n\n" for d in docs).strip())


# 핵심 문서(conflict_a/b 또는 default)와 방해 문서를 rng로 섞고 d0부터 doc_id를 붙여 공통 스키마 인스턴스 하나를 만든다.
# conflict_positions에는 방해 문서가 아닌 문서의 위치를 오름차순으로 담는다(none 유형은 1개). **extra는 그대로 필드로 붙는다.
# build_conflictbank와 build_wikicontradict가 모든 v2 인스턴스를 이 함수로 만든다.
def assemble(inst_id: str, source: str, ctype: str, question: str, core_docs: list[dict],
             distractor_texts: list[str], gold: dict, entity: str, rng: random.Random, **extra) -> dict:
    """Shuffle core + distractor docs, assign d0.., record positions of non-distractor docs (sorted)."""
    raw = core_docs + [{"text": t, "role": "distractor"} for t in distractor_texts]
    rng.shuffle(raw)
    docs = [{"doc_id": f"d{i}", "text": d["text"], "role": d["role"]} for i, d in enumerate(raw)]
    return {
        "id": inst_id, "source": source, "conflict_type": ctype, "question": question,
        "docs": docs, "gold": gold,
        "conflict_positions": [i for i, d in enumerate(docs) if d["role"] != "distractor"],
        "entity": entity, "total_tokens": docs_total_tokens(docs), **extra,
    }


# n이 window의 양끝을 포함한 범위 안에 있는지 판정한다. 문서당 토큰 길이 창 검사에 쓴다.
def _in_window(n: int, window: tuple[int, int]) -> bool:
    return window[0] <= n <= window[1]


# subject 문자열의 md5 첫 바이트가 짝수면 "dev", 홀수면 "test"를 돌려준다.
# 방해 문서 풀을 dev 쪽과 test 쪽으로 나눠, dev 계열과 test 계열이 같은 방해 문서를 공유하지 않게 한다.
def _side(subject: str) -> str:
    return "dev" if hashlib.md5(subject.encode()).digest()[0] % 2 == 0 else "test"


# 숫자 목록의 개수·평균과 최소·p5·p25·p50·p75·p95·최대(정수로 내림)를 사전으로 돌려준다. 빈 목록이면 빈 사전이다.
def _pcts(xs) -> dict:
    if not len(xs):
        return {}
    q = np.percentile(xs, [0, 5, 25, 50, 75, 95, 100])
    return {"n": len(xs), "mean": round(float(np.mean(xs)), 1),
            **{k: int(v) for k, v in zip(["min", "p5", "p25", "p50", "p75", "p95", "max"], q)}}


# ConflictBank 무작위 n개 레코드에서 default·misinformation·temporal evidence의 cl100k 길이 분포를 잰다.
# 각 windows 창에 대해 (default, 상대 evidence) 두 문서가 모두 창 안에 드는 쌍의 비율도 구한다.
# TOKEN_WINDOW(300–600)를 고르는 근거 자료이며, build_report.md 1절과 `stats` 명령 출력이 이 값이다.
def cb_length_stats(path: Path, index: list, n: int = 20000, seed: int = 42,
                    windows=((100, 400),)) -> dict:
    """cl100k token lengths of default/misinformation/temporal evidence on n random records."""
    rng = random.Random(f"{seed}-lenstats")
    sample = sorted(rng.sample(index, min(n, len(index))), key=lambda x: x[1])
    lens = {k: [] for k in cb.EVIDENCE_KEY}
    pair_in = {(c, tuple(w)): 0 for c in ("misinformation", "temporal") for w in windows}
    with open(path, "rb") as f:
        for _, off, _ in sample:
            rec = cb.read_record(f, off)
            t = {k: count_tokens(rec.get(key, "")) for k, key in cb.EVIDENCE_KEY.items()}
            for k in lens:
                lens[k].append(t[k])
            for c, w in pair_in:
                pair_in[(c, w)] += _in_window(t["default"], w) and _in_window(t[c], w)
    return {
        "n": len(sample),
        "per_type": {k: _pcts(v) for k, v in lens.items()},
        "pair_in_window_frac": {f"{c} {w[0]}-{w[1]}": round(v / len(sample), 4) for (c, w), v in pair_in.items()},
    }


# ConflictBank에서 subject가 서로 겹치지 않는 v2 분할(dev, dev_distractor, test_id_pool, test_ood1, distractor)을 만든다.
# seed로 섞은 subject를 SPLITS 순서대로 소비하며, subject마다 레코드를 섞어 cb.check_pair와 길이 창을 처음 통과한 1건만 쓴다.
# none 유형도 misinformation 쌍 필터로 거르고 default 문서 1개만 담는다. 방해 문서는 쓰이지 않은 subject에서 pool_size개를
# 뽑아 _side로 dev/test 풀을 나누고 질문 BM25로 randint(3, 6)개를 고른다. 반환: ({분할: 인스턴스 목록}, 거절 사유·풀 통계 보고).
def build_conflictbank(path: Path, index: list, sizes: dict = None, window: tuple[int, int] = TOKEN_WINDOW,
                       pool_size: int = 50000, seed: int = 42,
                       n_distractors: tuple[int, int] = N_DISTRACTORS) -> tuple[dict, dict]:
    """Subject-disjoint ConflictBank splits. Returns ({split: [instances]}, report)."""
    sizes = sizes or SPLITS
    rng = random.Random(seed)
    by_subject = collections.defaultdict(list)
    for ln, off, subj in index:
        by_subject[subj].append((ln, off))
    subjects = sorted(by_subject)
    rng.shuffle(subjects)

    chosen = {s: [] for s in sizes}       # split -> [(subject, line_no, record)]
    rejects = {s: collections.Counter() for s in sizes}
    used_subjects, cursor = set(), 0
    with open(path, "rb") as f:
        for split, (size, ctype) in sizes.items():
            pair_type = "misinformation" if ctype == "none" else ctype
            while len(chosen[split]) < size and cursor < len(subjects):
                subj = subjects[cursor]
                cursor += 1
                recs = by_subject[subj][:]
                rng.shuffle(recs)
                for ln, off in recs:
                    rec = cb.read_record(f, off)
                    reason = cb.check_pair(rec, pair_type)
                    if reason is None and not (
                            _in_window(count_tokens(rec[cb.EVIDENCE_KEY["default"]]), window)
                            and _in_window(count_tokens(rec[cb.EVIDENCE_KEY[pair_type]]), window)):
                        reason = "length_window"
                    rejects[split]["examined"] += 1
                    if reason:
                        rejects[split][reason] += 1
                        continue
                    chosen[split].append((subj, ln, rec))
                    used_subjects.add(subj)
                    break
                else:
                    rejects[split]["subject_without_valid_record"] += 1

        # distractor pool: records of subjects never used by any instance, split dev/test side by subject hash
        cand = [x for x in index if x[2] not in used_subjects]
        pool_rows = sorted(random.Random(f"{seed}-pool").sample(cand, min(pool_size, len(cand))), key=lambda x: x[1])
        pool_docs = {"dev": [], "test": []}
        pool_window_drop = 0
        for ln, off, subj in pool_rows:
            text = cb.read_record(f, off).get(cb.EVIDENCE_KEY["default"], "")
            if not text or not _in_window(count_tokens(text), window):
                pool_window_drop += 1
                continue
            pool_docs[_side(subj)].append({"key": f"cb_line{ln}", "text": text, "group": subj})
    pools = {side: DistractorPool(docs) for side, docs in pool_docs.items() if docs}

    out, short = {}, collections.Counter()
    for split, (size, ctype) in sizes.items():
        drng = random.Random(f"{seed}-{split}")
        pool = pools["dev" if split in DEV_SIDE else "test"]
        insts = []
        for i, (subj, ln, rec) in enumerate(chosen[split], start=1):
            k = drng.randint(*n_distractors)
            a, b = rec["object"], rec["replaced_object"]
            picked = pool.pick(rec["question"], k, exclude_strings=[subj, a, b], exclude_groups={subj})
            short[split] += len(picked) < k
            core = [{"text": rec[cb.EVIDENCE_KEY["default"]], "role": "default" if ctype == "none" else "conflict_a"}]
            gold = {"answer_a": a, "span_a": a}
            if ctype != "none":
                core.append({"text": rec[cb.EVIDENCE_KEY[ctype]], "role": "conflict_b"})
                gold.update({"answer_b": b, "span_b": b})
            if ctype == "temporal":
                gold["time_span"] = rec.get("temporal_conflict_time_span")
            insts.append(assemble(
                f"{split}_{i:04d}", "conflictbank", ctype, rec["question"], core,
                [d["text"] for d in picked], gold, subj, drng,
                relation=rec.get("relation"), source_line=ln,
                distractor_keys=[d["key"] for d in picked]))
        out[split] = insts

    report = {
        "n_records": len(index), "n_subjects": len(by_subject), "window": list(window),
        "subjects_consumed": cursor, "rejects": {s: dict(c) for s, c in rejects.items()},
        "pool": {"sampled": len(pool_rows), "dropped_window": pool_window_drop,
                 **{f"{side}_docs": len(d) for side, d in pool_docs.items()}},
        "short_distractor_instances": dict(short),
    }
    return out, report


# test_ood2를 만든다: wc.clean으로 정제한 WikiContradict 행 전부를 wiki_natural 유형 인스턴스로 바꾼다(exclude_keys=dev_50 행).
# 방해 문서 풀은 CSV 전체 context(중복 제거)이며 article 제목이 group이다. 같은 article, 제목·두 답 문자열 포함 문서는 제외한다.
# 주의: 풀 문서 key가 wc_<question_ID>라 한 행의 context1·context2가 같은 key를 가지며, ConflictBank와 달리 길이 창은 적용하지 않는다.
def build_wikicontradict(rows: list[dict], exclude_keys: set, seed: int = 42,
                         n_distractors: tuple[int, int] = N_DISTRACTORS) -> tuple[list[dict], dict]:
    """Test-OOD-2: all cleaned WikiContradict rows; BM25 distractors from other articles' contexts."""
    kept, steps = wc.clean(rows, exclude_keys)
    seen, docs = set(), []
    for r in rows:  # pool = every context in the CSV, grouped by article
        for c in (r["context1"].strip(), r["context2"].strip()):
            if c and c not in seen:
                seen.add(c)
                docs.append({"key": f"wc_{r['question_ID']}", "text": c, "group": r["WikipediaArticleTitle"].strip()})
    pool = DistractorPool(docs)
    rng = random.Random(f"{seed}-test_ood2")
    insts, short = [], 0
    for i, r in enumerate(kept, start=1):
        title = r["WikipediaArticleTitle"].strip()
        a1, a2 = r["answer1"].strip(), r["answer2"].strip()
        k = rng.randint(*n_distractors)
        picked = pool.pick(r["question"], k, exclude_strings=[title, a1, a2], exclude_groups={title})
        short += len(picked) < k
        insts.append(assemble(
            f"test_ood2_{i:04d}", "wikicontradict", "wiki_natural", r["question"].strip(),
            [{"text": r["context1"].strip(), "role": "conflict_a"}, {"text": r["context2"].strip(), "role": "conflict_b"}],
            [d["text"] for d in picked], {"answer_a": a1, "answer_b": a2, "span_a": a1, "span_b": a2},
            title, rng, question_ID=r["question_ID"], wiki_contradict_type=r["contradictType"],
            distractor_keys=[d["key"] for d in picked]))
    return insts, {"steps": steps, "pool_docs": len(docs), "short_distractor_instances": short}


# 인스턴스 목록을 한 줄에 JSON 하나씩(ensure_ascii=False) 쓴다. 상위 폴더가 없으면 만들고, 쓴 경로를 돌려준다.
def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


# 분할 하나의 요약(건수, total_tokens 분포, 방해 문서 개수별 빈도, 고유 entity 수)을 만든다. 구축 보고서 5절에 들어간다.
def split_summary(insts: list[dict]) -> dict:
    n_dis = collections.Counter(sum(d["role"] == "distractor" for d in x["docs"]) for x in insts)
    return {"n": len(insts), "total_tokens": _pcts([x["total_tokens"] for x in insts]),
            "n_distractors": dict(sorted(n_dis.items())), "unique_entities": len({x["entity"] for x in insts})}


# dev 인스턴스에서 seed로 n건을 뽑아 사람이 채우는 수동 검사 시트(마크다운, 체크박스 3개)를 쓴다.
# 각 항목에 conflict_a·conflict_b 문서를 그대로 싣고 방해 문서는 앞 200자만 싣는다.
# 주의: conflict_a/b 역할이 있다고 가정하므로 none 유형 분할을 넣으면 KeyError가 난다. build_all은 dev만 넘긴다.
def write_manual_check(path: Path, dev: list[dict], n: int = 20, seed: int = 42) -> Path:
    pick = random.Random(f"{seed}-manual").sample(dev, min(n, len(dev)))
    out = ["# dev 수동 검사 시트", "",
           f"dev.jsonl에서 seed {seed}로 무작위 {len(pick)}건. 체크박스는 검사자가 직접 채운다.", ""]
    for j, x in enumerate(pick, 1):
        g = x["gold"]
        role = {d["role"]: d for d in x["docs"]}
        out += [f"## {j}. {x['id']} (entity: {x['entity']}, relation: {x.get('relation')})", "",
                f"- 질문: {x['question']}", f"- answer_a: {g['answer_a']}", f"- answer_b: {g.get('answer_b')}", "",
                f"### conflict_a ({role['conflict_a']['doc_id']})", "", role["conflict_a"]["text"], "",
                f"### conflict_b ({role['conflict_b']['doc_id']})", "", role["conflict_b"]["text"], "",
                "### distractors (앞 200자)", ""]
        out += [f"- [{d['doc_id']}] {d['text'][:200].replace(chr(10), ' ')}" for d in x["docs"] if d["role"] == "distractor"]
        out += ["", "- [ ] 두 문서가 같은 질문에 서로 다른 답을 준다",
                "- [ ] distractor 중 가짜 충돌(같은 질문에 다른 답)을 만드는 문서가 있다",
                "- [ ] 한 문서 안에 양쪽 정보(answer_a와 answer_b)가 섞여 있다", "- 메모:", ""]
    path.write_text("\n".join(out), encoding="utf-8")
    return path


# 객체를 들여쓰기 2칸 JSON 코드 블록(```json) 문자열로 만든다. 구축 보고서 본문에 넣는 용도다.
def _md_json(obj) -> str:
    return "```json\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```"


# v2 데이터 구축 전체를 실행한다: 길이 통계 → ConflictBank 분할 → WikiContradict test_ood2 → 파일 쓰기 → 보고서 → 봉인.
# dev·dev_distractor는 data/processed/에, test_id_pool·test_ood1·test_ood2·distractor는 data/sealed/에 쓴다.
# 오늘 날짜를 붙인 <날짜>_build_report.md와 <날짜>_dev_manual_check.md를 만들고, seal=True면 test 계열 4개 파일을 seal_files로 봉인한다.
# 화면에는 집계(분할 크기, 분할 간 subject 중복 수, 보고서 경로)만 찍는다. `python src/data_build.py build`로 실행한다.
def build_all(seed: int = 42, window: tuple[int, int] = TOKEN_WINDOW, pool_size: int = 50000,
              stats_n: int = 20000, raw_path: Path = CB_RAW, seal: bool = True) -> dict:
    """Build every v2 set, write processed/sealed files, report, and seal. Prints only aggregates."""
    today = dt.date.today().isoformat()
    index = cb.load_index(raw_path, CB_INDEX_CACHE)
    print(f"index: {len(index)} records", flush=True)
    stats = cb_length_stats(raw_path, index, stats_n, seed, windows=((100, 400), tuple(window)))
    print("length stats done", flush=True)
    sets, cb_rep = build_conflictbank(raw_path, index, window=window, pool_size=pool_size, seed=seed)
    print("conflictbank sets done", flush=True)
    ood2, wc_rep = build_wikicontradict(wc.load_rows(), wc.dev50_keys(DEV_50_PATH), seed)

    paths = {
        "dev": write_jsonl(OUTPUT_DIR / "dev.jsonl", sets["dev"]),
        "dev_distractor": write_jsonl(OUTPUT_DIR / "dev_distractor.jsonl", sets["dev_distractor"]),
        "test_id_pool": write_jsonl(SEALED_DIR / "test_id_pool.jsonl", sets["test_id_pool"]),
        "test_ood1": write_jsonl(SEALED_DIR / "test_ood1.jsonl", sets["test_ood1"]),
        "test_ood2": write_jsonl(SEALED_DIR / "test_ood2.jsonl", ood2),
        "distractor": write_jsonl(SEALED_DIR / "distractor.jsonl", sets["distractor"]),
    }
    all_sets = {**sets, "test_ood2": ood2}
    summaries = {k: split_summary(v) for k, v in all_sets.items()}
    cb_entities = {k: {x["entity"] for x in v} for k, v in sets.items()}
    names = list(cb_entities)
    overlaps = {f"{a}&{b}": len(cb_entities[a] & cb_entities[b])
                for i, a in enumerate(names) for b in names[i + 1:]}
    write_manual_check(OUTPUT_DIR / f"{today}_dev_manual_check.md", sets["dev"], seed=seed)

    report = OUTPUT_DIR / f"{today}_build_report.md"
    report.write_text("\n\n".join([
        f"# CCD-G1 데이터 구축 보고 ({today})",
        f"- seed: {seed}\n- 토큰화: cl100k_base (`budget.count_tokens`)\n"
        f"- ConflictBank 원본: `{raw_path.relative_to(ROOT_DIR)}` ({raw_path.stat().st_size:,} bytes, JSON Lines)\n"
        f"- 레코드 수: {cb_rep['n_records']:,}, 고유 subject 수: {cb_rep['n_subjects']:,}",
        "## 1. evidence 길이 분포 (무작위 레코드 표본)", _md_json(stats),
        f"## 2. 선택한 길이 창\n\nevidence 문서 하나당 {window[0]}–{window[1]} 토큰(cl100k). "
        "충돌 쌍은 두 문서 모두, distractor 문서도 같은 창을 적용했다.",
        "## 3. ConflictBank 분할 (subject 단위, 한 subject당 최대 1개 레코드)\n\n"
        "subject를 seed로 섞은 뒤 dev → dev_distractor → test_id_pool → test_ood1 → distractor 순으로 채운다. "
        "각 subject의 레코드를 섞어 처음으로 모든 필터를 통과한 레코드 1개를 쓴다. none 유형도 misinformation 쌍 "
        "필터를 통과한 레코드에서 뽑아 같은 모집단을 유지했다. 아래 rejects는 검사한 레코드 기준이다 "
        "(필터 순서: missing_field → same_answer → span_missing → mixed → length_window).",
        _md_json({k: v for k, v in cb_rep.items() if k not in ("pool", "short_distractor_instances")}),
        f"세트 간 subject 중복 수 (모두 0이어야 함): {json.dumps(overlaps)}",
        "## 4. distractor 선택 방법\n\n"
        f"- 후보 풀: 어떤 질문 인스턴스에도 쓰이지 않은 subject의 레코드에서 무작위 {pool_size:,}개를 뽑아 "
        "default_evidence를 사용(길이 창 적용). subject의 md5 첫 바이트 홀짝으로 dev 쪽 풀과 test 쪽 풀을 나눠, "
        "dev·dev_distractor와 test 계열이 같은 distractor 문서를 공유하지 않게 했다.\n"
        "- 순위: rank_bm25 BM25Okapi, 질문 텍스트를 쿼리로 사용(소문자 `\\w+` 토큰, 불용어 제거 없음).\n"
        "- 제외: 인스턴스의 subject·object·replaced_object 문자열 중 하나라도 대소문자 무시 단어 경계 일치로 포함한 문서, "
        "같은 subject 문서, 한 인스턴스 안에서 같은 subject의 두 번째 문서.\n"
        "- 개수: 인스턴스마다 seed 고정 randint(3, 6).",
        _md_json({"pool": cb_rep["pool"], "short_distractor_instances": cb_rep["short_distractor_instances"]}),
        "## 5. 세트별 요약 (생성 시점 기록; 봉인 후 sealed 파일은 다시 열지 않는다)", _md_json(summaries),
        "## 6. WikiContradict (test_ood2)\n\n"
        "정제 단계: context1==context2 또는 samepassage=Same 제외 → dev_50.jsonl에 쓰인 행 제외"
        "((question, WikipediaArticleTitle) 일치; dev_50에 question_ID가 없어 이 키로 매핑) → 빈 답·동일 답 제외. "
        "distractor 풀은 CSV 전체 행의 context(다른 article만, 제목·두 답 문자열 포함 문서 제외)이며, "
        "다른 test_ood2 인스턴스의 context가 distractor로 들어갈 수 있다(소스 규모 제약).",
        _md_json(wc_rep),
        "## 7. 계획서 v2 대비 변경점",
        "\n".join([
            f"- evidence 길이 창: 계획서 100–400 토큰 → {window[0]}–{window[1]} 토큰 (근거: 1절 분포).",
            "- Test-ID: 1,000건 대신 순서 고정 2,000건 풀(test_id_pool)을 봉인. G1에서 정한 N만큼 앞에서부터 사용.",
            "- dev_distractor 100건 추가(dev 쪽 충돌 없음 대조군, 계획서 표에는 없음).",
            "- 자동 제외 규칙 추가: 정답 문자열이 해당 문서에 없는 레코드(span_missing), 한 문서에 상대측 답이 "
            "함께 나오는 레코드(mixed). 실행 가이드의 '수동 검사 후 제외 규칙'을 문자열 일치로 먼저 적용한 것이다.",
            "- 질문: ConflictBank 원본 `question`(자유형 문장) 사용. 원본의 객관식 options는 쓰지 않는다.",
            "- none 유형 인스턴스의 conflict_positions는 default 문서 위치 1개만 담는다(공통 스키마의 [int,int]와 다름).",
            "- WikiContradict distractor: build_dev_50의 무작위 선택 대신 BM25 상위 선택.",
        ]),
    ]) + "\n", encoding="utf-8")

    sealed = None
    if seal:
        sealed = seal_files([paths[k] for k in ("test_id_pool", "test_ood1", "test_ood2", "distractor")])
    print(json.dumps({"sizes": {k: v["n"] for k, v in summaries.items()}, "overlaps": overlaps,
                      "report": str(report)}, ensure_ascii=False), flush=True)
    return {"paths": paths, "report": report, "sealed": sealed}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dev50"
    if cmd == "dev50":
        build_dev_50()
    elif cmd == "stats":  # length distribution only, to choose TOKEN_WINDOW before building
        idx = cb.load_index(CB_RAW, CB_INDEX_CACHE)
        print(json.dumps(cb_length_stats(CB_RAW, idx, windows=((100, 400), (300, 600), (350, 650), (400, 700))), indent=2))
    elif cmd == "build":
        build_all()
