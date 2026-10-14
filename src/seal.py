"""Test-set sealing: file hashes, prompt freezing, and a gate for test-split access.

- seal_files      : record SHA-256 + line count of sealed files in data/SEALED.md (appends; history kept)
- verify_sealed   : raise if any file differs from the latest SEALED.md section
- freeze_prompts  : record final prompt hashes (+ git HEAD when the prompts are committed) in
                    results/FROZEN_PROMPTS.json
- check_test_access: test-type splits require verify_sealed + frozen prompts that still match;
                    every attempt is appended to results/test_access_log.jsonl
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import warnings
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SEALED_SPLITS = {"test_id", "test_id_pool", "test_ood1", "test_ood2", "distractor"}
_ROW = re.compile(r"^\| (\S+) \| (\d+) \| ([0-9a-f]{64}) \|$")


# 결과 폴더 경로를 돌려준다. 환경변수 CCD_RESULTS_DIR가 있으면 그것을, 없으면 <root>/results를 쓴다.
# FROZEN_PROMPTS.json과 test_access_log.jsonl이 이 폴더에 놓인다.
def _results_dir(root: Path = ROOT_DIR) -> Path:
    return Path(os.environ.get("CCD_RESULTS_DIR") or root / "results")


# 봉인 기록(data/SEALED.md)이 없거나, 봉인 파일이 없어지거나 해시가 바뀌었을 때 쓰는 예외다.
class SealError(RuntimeError):
    pass


# 파일을 1MB 단위로 읽어 SHA-256 16진 문자열을 돌려준다. 봉인 해시와 프롬프트 동결 해시 모두 이 값이다.
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 파일의 줄 수를 센다. jsonl 봉인 기록의 lines 열(인스턴스 수)로 쓴다.
def _line_count(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


# 경로를 root 기준 상대경로 문자열로 바꾼다. root 밖에 있으면 절대경로 문자열을 그대로 돌려준다.
def _rel(path: Path, root: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(root.resolve()))
    except ValueError:
        return str(path)


# 현재 시각을 시간대가 붙은 ISO 8601 문자열(초 단위)로 돌려준다. 봉인·동결·접근 로그의 시각 기록용이다.
def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


# 주어진 파일들의 SHA-256과 줄 수를 data/SEALED.md에 새 '## Seal <시각>' 절로 덧붙인다(이전 절은 이력으로 남는다).
# 반환: {상대경로: {"sha256", "lines"}}. data_build.build_all이 test 계열 4개 파일을,
# make_test_subset.py가 test_id(test_id_pool 앞 n줄)를 만든 뒤 기존 봉인 파일과 함께 다시 봉인할 때 호출한다.
def seal_files(paths, root: Path = ROOT_DIR) -> dict:
    """Append a seal section to <root>/data/SEALED.md. Returns {relpath: {"sha256", "lines"}}."""
    entries = {_rel(p, root): {"sha256": _sha256_file(Path(p)), "lines": _line_count(Path(p))} for p in paths}
    md = root / "data" / "SEALED.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    head = md.read_text(encoding="utf-8") if md.exists() else (
        "# SEALED test files\n\n"
        "Each section is one seal event; the LAST section is authoritative for verify_sealed().\n"
        "Earlier sections are kept as history. Do not edit rows by hand.\n")
    rows = "\n".join(f"| {k} | {v['lines']} | {v['sha256']} |" for k, v in entries.items())
    md.write_text(head.rstrip("\n") + f"\n\n## Seal {_now()}\n\n| path | lines | sha256 |\n|---|---|---|\n{rows}\n",
                  encoding="utf-8")
    return entries


# SEALED.md의 마지막 '## Seal' 절에서 표 행을 읽어 {경로: sha256}을 돌려준다.
# 파일이 없거나, 절이 없거나, 마지막 절에 행이 없으면 SealError를 낸다.
def _latest_seal(root: Path) -> dict[str, str]:
    md = root / "data" / "SEALED.md"
    if not md.exists():
        raise SealError(f"{md} not found; nothing has been sealed")
    section = md.read_text(encoding="utf-8").rsplit("\n## Seal ", 1)
    if len(section) < 2:
        raise SealError(f"{md} has no seal section")
    rows = {m.group(1): m.group(3) for line in section[1].splitlines() if (m := _ROW.match(line.strip()))}
    if not rows:
        raise SealError(f"{md} latest section has no rows")
    return rows


# 마지막 봉인 절에 적힌 파일마다 존재 여부와 현재 해시를 확인하고, 하나라도 없거나 바뀌었으면 SealError를 낸다.
# 상대경로는 root 기준으로 푼다. check_test_access, make_test_subset.py, pretest_gate.py(K6)가 호출한다.
def verify_sealed(root: Path = ROOT_DIR) -> None:
    """Raise SealError if any file in the latest seal section is missing or changed."""
    problems = []
    for rel, sha in _latest_seal(root).items():
        p = Path(rel) if Path(rel).is_absolute() else root / rel
        if not p.exists():
            problems.append(f"missing: {rel}")
        elif _sha256_file(p) != sha:
            problems.append(f"hash mismatch: {rel}")
    if problems:
        raise SealError("; ".join(problems))


# paths가 모두 git에 추적되고 수정되지 않았을 때만 HEAD 커밋 해시를 돌려주고, 아니면 None을 돌려준다.
# git 실행 실패(저장소 아님, git 없음, 10초 시간 초과 등)도 None으로 처리한다.
def _git_commit_if_clean(paths: list[Path], cwd: Path) -> str | None:
    """git HEAD if every path is tracked and unmodified; otherwise None."""
    try:
        # cwd에서 git 하위 명령을 실행하는 내부 도우미다. 실패하면 check=True에 따라 예외를 던진다.
        def git(*args):
            return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=10)
        git("ls-files", "--error-unmatch", *map(str, paths))
        if git("status", "--porcelain", "--", *map(str, paths)).stdout.strip():
            return None
        return git("rev-parse", "HEAD").stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


# 최종 프롬프트 파일들의 SHA-256과 동결 시각, git 커밋(깨끗할 때만, 아니면 null)을 results/FROZEN_PROMPTS.json에 쓴다.
# 커밋을 얻지 못하면 경고만 내고 계속한다. run.py의 freeze 하위 명령이 호출하며, 이후 _check_frozen이 이 기록과 대조한다.
def freeze_prompts(prompt_paths, root: Path = ROOT_DIR) -> Path:
    paths = [Path(p).resolve() for p in prompt_paths]
    commit = _git_commit_if_clean(paths, root)
    if commit is None:
        warnings.warn("prompt files are not committed (or repo untracked); git_commit recorded as null")
    out = _results_dir(root) / "FROZEN_PROMPTS.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "frozen_at": _now(),
        "git_commit": commit,
        "prompts": {_rel(p, root): _sha256_file(p) for p in paths},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


# FROZEN_PROMPTS.json이 있고, 기록된 프롬프트 파일이 모두 존재하며 해시가 그대로인지 확인한다.
# 어느 하나라도 어긋나면 PermissionError를 낸다. check_test_access의 두 번째 관문이다.
def _check_frozen(root: Path) -> None:
    fp = _results_dir(root) / "FROZEN_PROMPTS.json"
    if not fp.exists():
        raise PermissionError(f"{fp} not found; freeze final prompts before touching test splits")
    frozen = json.loads(fp.read_text(encoding="utf-8"))["prompts"]
    for rel, sha in frozen.items():
        p = Path(rel) if Path(rel).is_absolute() else root / rel
        if not p.exists():
            raise PermissionError(f"frozen prompt missing: {rel}")
        if _sha256_file(p) != sha:
            raise PermissionError(f"frozen prompt changed since freeze: {rel}")


# test 계열 분할을 열기 전의 관문이다. SEALED_SPLITS에 있거나 이름이 "test"로 시작하는 분할만 검사하고 나머지는 그냥 통과한다.
# verify_sealed(봉인 해시)와 _check_frozen(프롬프트 동결)을 모두 통과해야 허용하며, 실패하면 PermissionError를 낸다.
# 허용·거부와 관계없이 시도마다 results/test_access_log.jsonl에 한 줄을 남긴다. run.py의 load_split이 호출한다.
def check_test_access(split: str, root: Path = ROOT_DIR) -> None:
    """No-op for dev splits. For test-type splits raise PermissionError unless the seal holds
    and the frozen prompts are unchanged. Every test-type attempt is logged."""
    if not (split in SEALED_SPLITS or split.startswith("test")):
        return
    reason = None
    try:
        try:
            verify_sealed(root)
        except SealError as e:
            raise PermissionError(f"seal check failed: {e}") from e
        _check_frozen(root)
    except PermissionError as e:
        reason = str(e)
    log = _results_dir(root) / "test_access_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": _now(), "split": split, "allowed": reason is None, "reason": reason},
                           ensure_ascii=False) + "\n")
    if reason:
        raise PermissionError(f"test access denied for split '{split}': {reason}")
