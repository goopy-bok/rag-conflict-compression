"""
Few-shot fixation check for the C3 runs (execution guide v2 section 7.5).

  uv run python src/fewshot_check.py                 # every reports/apo_log_*.json
  uv run python src/fewshot_check.py --mock          # reports/mock/apo_log_*.json
  uv run python src/fewshot_check.py --logs reports/apo_log_C3-G_seed0.json ...

Per APO log:
  1. examples: the sha256 set of examples blocks that reached compressor calls (log key
     "examples_sha256_seen", or "examples_seen" in the ProTeGi log) must be exactly
     {sha256(render_examples(c2_fewshot.json))}; for C3-G1 exactly {sha256("")}. Missing key = fail.
  2. demos: every predictor of the saved DSPy program (program_path) has no demos. Runs without a
     DSPy program (ProTeGi) pass with a note.
  3. recorded only: does the final instruction contain an example-like block ([Examples], ## Example k,
     [Question], [Compressed]) that the start instruction did not contain.
Writes reports/fewshot_check.json (mock: reports/mock/fewshot_check.json). Exit 1 if 1 or 2 fails
for any run, or if no log is found.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from apo_common import expected_examples_block, sha256_text  # noqa: E402

EXAMPLE_BLOCK = re.compile(r"\[Examples\]|##\s*Example\s*\d|\[Question\]|\[Compressed\]", re.I)
DEFAULT_PROMPT = {"C3-P": "c3p.txt", "C3-M": "c3m.txt"}


def _path(p) -> Path | None:
    return None if not p else (Path(p) if Path(p).is_absolute() else ROOT_DIR / p)


def check_log(log: dict, fewshot_path: Path = ROOT_DIR / "prompts" / "c2_fewshot.json",
              log_dir: Path | None = None) -> dict:
    cond = log.get("condition", "?")
    uses_fewshot = log.get("uses_fewshot", cond != "C3-G1")
    expected = sha256_text(expected_examples_block(uses_fewshot, fewshot_path))
    seen = log.get("examples_sha256_seen", log.get("examples_seen"))
    if seen is None:
        c1 = {"ok": False, "reason": "examples hashes not recorded in the log"}
    else:
        c1 = {"ok": sorted(seen) == [expected], "expected": expected, "seen": sorted(seen)}

    prog = _path(log.get("program_path"))
    if prog is None:
        c2 = {"ok": True, "note": "no DSPy program in this run (demos not applicable)"}
    elif not prog.exists():
        c2 = {"ok": False, "reason": f"program file missing: {prog}"}
    else:
        state = json.loads(prog.read_text(encoding="utf-8"))
        demos = {k: len(v.get("demos", [])) for k, v in state.items() if isinstance(v, dict) and "demos" in v}
        c2 = {"ok": bool(demos) and not any(demos.values()), "demos": demos}

    final = _path(log.get("final_instruction_path"))
    if final is None and cond in DEFAULT_PROMPT and log_dir is not None:
        final = (ROOT_DIR / "prompts" / ("mock" if log_dir.name == "mock" else "") / DEFAULT_PROMPT[cond])
    start = _path(log.get("start_instruction_path"))
    final_text = final.read_text(encoding="utf-8") if final and final.exists() else None
    start_text = start.read_text(encoding="utf-8") if start and start.exists() else ""
    c3 = {"final_instruction_path": str(final) if final else None,
          "new_example_block": None if final_text is None else
          bool(EXAMPLE_BLOCK.search(final_text)) and not EXAMPLE_BLOCK.search(start_text)}
    return {"condition": cond, "seed": log.get("seed"), "examples_input_fixed": c1, "demos_empty": c2,
            "example_block_in_instruction": c3, "valid": c1["ok"] and c2["ok"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", type=Path)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    rdir = ROOT_DIR / "reports" / ("mock" if a.mock else "")
    logs = a.logs or sorted(rdir.glob("apo_log_*.json"))
    results = [check_log(json.loads(Path(p).read_text(encoding="utf-8")), log_dir=Path(p).parent) | {"log": str(p)}
               for p in logs]
    out = a.out or rdir / "fewshot_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"all_valid": bool(results) and all(r["valid"] for r in results), "runs": results}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in results:
        print(f"{r['condition']:7} seed={r['seed']}  examples_fixed={r['examples_input_fixed']['ok']}  "
              f"demos_empty={r['demos_empty']['ok']}  new_example_block="
              f"{r['example_block_in_instruction']['new_example_block']}  -> {'VALID' if r['valid'] else 'INVALID'}")
    if not results:
        print(f"no APO logs found in {rdir}")
    print(f"saved {out}")
    return 0 if report["all_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
