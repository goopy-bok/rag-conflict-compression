"""
C3' (C3-V): the final C3-G instruction of each seed plus one extractive line, no further
optimization (execution guide v2 section 7.7). Evaluated with the same c2_fewshot.json examples.

  uv run python src/apo_variants.py --seeds 0 1 2          # prompts/c3g_seed{k}.txt -> prompts/c3v_seed{k}.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
VERBATIM_LINE = "Copy sentences verbatim from the documents; do not paraphrase."


def make_variant(seed: int, prompts_dir: Path = ROOT_DIR / "prompts") -> Path:
    src = prompts_dir / f"c3g_seed{seed}.txt"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found; run apo_gepa.py --condition C3-G --seed {seed} first")
    dst = prompts_dir / f"c3v_seed{seed}.txt"
    dst.write_text(src.read_text(encoding="utf-8").rstrip("\n") + "\n" + VERBATIM_LINE + "\n", encoding="utf-8")
    return dst


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--prompts-dir", type=Path, default=ROOT_DIR / "prompts")
    a = ap.parse_args(argv)
    for s in a.seeds:
        print(f"saved {make_variant(s, a.prompts_dir)}")


if __name__ == "__main__":
    main()
