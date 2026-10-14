"""
G1-lite Test-ID subset (preregistration section 7.1): the first N lines of the sealed pool.

  uv run python src/make_test_subset.py --split test_id --n 200

Copies the first N lines of data/sealed/test_id_pool.jsonl to data/sealed/test_id.jsonl as bytes (no parsing,
nothing printed from the file) and appends a seal section to data/SEALED.md that keeps every file of the
previous section plus test_id.jsonl. If test_id.jsonl already exists it must equal those N lines (same
SHA-256); otherwise this stops with an error. Other splits are cut with `run.py run --limit` (no file).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import seal  # noqa: E402

POOL, OUT = Path("data/sealed/test_id_pool.jsonl"), Path("data/sealed/test_id.jsonl")


def head_bytes(path: Path, n: int) -> bytes:
    lines = []
    with path.open("rb") as f:
        for line in f:
            lines.append(line)
            if len(lines) == n:
                break
    if len(lines) < n:
        raise SystemExit(f"{path} has {len(lines)} lines, fewer than {n}")
    return b"".join(lines)


def make_test_id(n: int, root: Path = seal.ROOT_DIR) -> Path:
    seal.verify_sealed(root)  # the pool (and everything else sealed) must be unchanged
    pool, out = root / POOL, root / OUT
    data = head_bytes(pool, n)
    sha = hashlib.sha256(data).hexdigest()
    if out.exists():
        if seal._sha256_file(out) != sha:
            raise seal.SealError(f"{out} exists and differs from the first {n} lines of {pool}; not overwritten")
        print(f"{out} already equals the first {n} lines of the pool")
    else:
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(out)
        print(f"wrote {out} (first {n} lines of {pool.name})")
    latest = seal._latest_seal(root)
    if latest.get(str(OUT)) != sha:
        paths = [Path(p) if Path(p).is_absolute() else root / p for p in latest if p != str(OUT)] + [out]
        seal.seal_files(paths, root)
        print(f"sealed {len(paths)} files in {root / 'data' / 'SEALED.md'}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, required=True)
    a = ap.parse_args(argv)
    if a.split != "test_id":
        raise SystemExit(f"no file is made for {a.split!r}; use `run.py run --split {a.split} --limit {a.n}`")
    make_test_id(a.n)


if __name__ == "__main__":
    main()
