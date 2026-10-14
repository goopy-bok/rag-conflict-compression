# rag-conflict-compression

Code and results for measuring **whether inter-document knowledge conflicts survive LLM context compression in RAG**.

When retrieved documents give incompatible answers to the same question (answer A vs. answer B), an upstream compressor shrinks the context to a token budget of `B = floor(0.3 × source tokens)` before a downstream reader answers. We measure how often both sides survive compression, and how the compression instruction affects that.

> RAG 검색 문서끼리 답이 충돌할 때, LLM이 문맥을 30%로 압축하는 단계에서 양쪽 답이 살아남는지를 측정하는 실험 코드와 결과입니다. 압축 지시문(C0~C3)에 따라 충돌 보존율(CSR)이 어떻게 달라지는지 비교합니다.

**Status:** preliminary research code (master's thesis in progress). The results below are **preliminary**, not a final verdict. They come from a single seed and r = 0.3 only. The LLM judge has not yet been validated against human labels for this compressor.

## Pipeline

```
retrieved docs (2 conflicting + 3–6 BM25 distractors, ~3,035 tokens/item)
  → upstream compressor (prompt condition C0–C3, budget r = 0.3, one retry if over budget)
  → downstream reader (sees only the compressed context)
  → independent eval judge (CSR, Signal, D, FCR)
```

| Condition | Compression instruction |
|---|---|
| C0 | Plain: "keep only the information needed to answer" |
| C1 | C0 + "if documents conflict, preserve both sides" |
| C2 | Hand-written structured manual + 3 fixed few-shot examples (v0, untuned) |
| C3-P | ProTeGi-optimized instruction (C2 examples kept fixed) |
| C3-M | MIPROv2-optimized instruction (C2 examples kept fixed) |

| Metric | Meaning |
|---|---|
| CSR | Both answers A and B present in the compressed text, undistorted, within budget |
| Signal | Compressed text explicitly marks the conflict |
| D | Downstream answer acknowledges the conflict (`CONFLICT_ACK`) |
| FCR | On conflict-free inputs, the compressor invents a conflict |

Roles in the Phi-4 runs (one model family per role): compressor and reader `microsoft/phi-4`, optimization judge `meta-llama/llama-3.1-8b-instruct`, reflection LM `openai/gpt-4o-mini`, eval judge `qwen/qwen-2.5-72b-instruct`. All calls go through OpenRouter (`configs/models_openrouter.yaml`).

## Repository layout

```
src/                    experiment code
  data_build.py         build items from ConflictBank / WikiContradict (filters, BM25 distractors, splits, seal)
  distractors.py        BM25 distractor retrieval
  seal.py               SHA-256 sealing of test splits, prompt freezing, test-access log
  run.py                harness CLI: run / estimate / freeze
  compressors/llm.py    prompt compressor (C0–C3), output normalization, one budget retry
  budget.py             cl100k token counting and budget
  llm_client.py         Ollama / Splash / OpenRouter client with cache and cost cap
  judge.py              CSR / FCR / downstream reader / D judges
  apo_protegi.py        C3-P (ProTeGi port)
  apo_mipro.py          C3-M (MIPROv2 via DSPy)
  apo_gepa.py           C3-G (GEPA, used in the earlier phi-3.5 pilot)
  apo_common.py, dspy_adapter.py   shared optimizer pieces
  stats.py              Wilson CI, paired bootstrap, exact McNemar, Holm, exact binomial, Fisher
  ...                   preflight / gate / G0 judge-validation / labelling tools
scripts/
  env_openrouter.sh           environment for the Phi-4 runs
  run_openrouter_bench.sh     first Phi-4 run (results/openrouter_phi4)
  run_openrouter_expanded.py  expanded run, N = 1,354 items (results/openrouter_phi4_expanded)
  run_lite.sh                 earlier local phi-3.5 pilot (G1-lite)
configs/                model roles and condition → prompt mapping
prompts/                compression, judge and downstream prompts (prompts/openrouter: optimized C3-P, C3-M)
tests/                  pytest suite (offline, no network)
data/SEALED.md          SHA-256 of the sealed test splits (data itself not included)
data/processed/apo_split.json   train/val ids used for prompt optimization
results/openrouter_phi4/          first Phi-4 run (test_id 200, ood1 100, ood2 154, distractor 100)
results/openrouter_phi4_expanded/ expanded Phi-4 run (test_id 600, ood1 300, ood2 154, distractor 300)
reports/openrouter/     optimization logs (ProTeGi trace, MIPROv2 / ProTeGi run logs)
```

Each result file is named `<condition>_<split>_r0.3_s0.jsonl`, with one item per line. The main fields are `question`, `gold.answer_a`, `gold.answer_b`, `compressed`, `judge.*`, `csr`, `signal`, `false_conflict`, and `downstream.*`. `FINAL_BENCHMARK_REPORT.md` in each results folder holds the statistics, in Korean. `FROZEN_PROMPTS.json` records the prompt hashes used, and `api_cost_log.jsonl` records per-call token and cost figures.

## Preliminary results (expanded Phi-4 run, test_id n = 600, r = 0.3, seed 0)

| Condition | CSR | 95% CI (Wilson) | D |
|---|---|---|---|
| C0 | 30.0% (180/600) | 26.5–33.8 | 121/600 |
| C1 | 62.2% (373/600) | 58.2–66.0 | 305/600 |
| C2 | 83.2% (499/600) | 80.0–85.9 | 514/600 |
| C3-P | 84.7% (508/600) | 81.6–87.3 | 510/600 |
| C3-M | 87.0% (522/600) | 84.1–89.5 | 539/600 |

- C1 − C0 = +32.2 pp. Exact McNemar p < 0.001.
- C3-M − C2 = +3.8 pp. McNemar p = 0.013 before correction and 0.053 after Holm correction. C3-P − C2 = +1.5 pp (n.s.).
- **False conflicts on conflict-free inputs are high for the structured prompts.** FCR is C2 159/300, C3-P 187/300, and C3-M 134/300, versus 1/300 for C0 and C1. The FCR judge sees the reference answer, so treat these numbers as provisional.
- WikiContradict (OOD-2) is lower for every condition. It overlaps an earlier pilot set (45/154 items) and is clustered by article, so its significance is not claimed.

See `results/openrouter_phi4_expanded/FINAL_BENCHMARK_REPORT.md` for the full tables and caveats.

## Setup

```bash
uv sync                    # Python 3.12, dependencies from uv.lock
uv run pytest -q           # offline unit tests (3 tests that need data/processed/dev.jsonl skip until the data is built)
cp .env.example .env       # add OPENROUTER_API_KEY to run real calls
```

## Rebuilding the data

The datasets are **not redistributed** here. To rebuild them:

1. Download ConflictBank QA (`QA_dataset.json` from the Hugging Face dataset `Warrieryes/CB_qa`) into `data/raw/conflictbank/`.
2. Put the WikiContradict CSV at `data/raw/wikicontradict_all.csv`, or set `CCD_WIKI_CSV`.
3. Run `uv run python src/data_build.py`. The builder is deterministic (seed 42).
4. Compare your sealed files against the hashes in `data/SEALED.md`.

## Running the Phi-4 experiment

```bash
source scripts/env_openrouter.sh                 # isolates configs, prompts, results and cache
bash scripts/run_openrouter_bench.sh             # 5 conditions × 4 splits (cost-capped, resumable)
uv run python scripts/run_openrouter_expanded.py # expanded run on unused sealed items
```

Test splits can only be read after the prompts are frozen (`src/run.py freeze`) and the seal still verifies. Every access is appended to `test_access_log.jsonl`.

## Data sources

- ConflictBank: Su et al., *ConflictBank: A Benchmark for Evaluating the Influence of Knowledge Conflicts in LLM*, NeurIPS 2024 Datasets & Benchmarks. arXiv:2408.12076.
- WikiContradict: Hou et al., *WikiContradict: A Benchmark for Evaluating LLMs on Real-World Knowledge Conflicts from Wikipedia*, NeurIPS 2024 Datasets & Benchmarks. arXiv:2406.13805.

Please follow the original licenses of these datasets. The result files in `results/` contain questions, answers and compressed text derived from them.

## License

The code is released under the MIT License (see `LICENSE`). Dataset-derived content follows the original dataset licenses.
