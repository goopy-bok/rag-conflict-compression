"""
Preflight for an unattended G1-lite run (amendment 2.2-c). No LLM calls.

  uv run python src/preflight.py                 # results/preflight.json; exit 0 pass, 1 fail
  uv run python src/preflight.py --allow-swap    # swap usage recorded but not a failure

P1 roles: the compressor is a ROLE_TABLE candidate, the roles equal assign_roles(compressor) (configure_roles
   was applied), and the family rule holds.
P2 G0 input: <PILOT_DIR>/<compressor>_C0.jsonl and _C1.jsonl hold >= 25 records each
   (env PILOT_DIR, default results/pilot_h2h; G0 samples 50 = 25 per condition).
P3 Ollama: the server answers, every Ollama role model is installed, and a role model whose thinking default is
   on has `think` set in configs/models.yaml (otherwise thinking would silently use the output budget).
P4 Splash (only when a role uses it): `splash` on PATH and the manifest glob matches exactly one file.
P5 resources: free disk >= 30 GB; swap used < 90% of total (failure unless --allow-swap); memory pressure recorded.
P6 seal intact and test unopened (pretest_gate.check_test_unopened); on a resume after the pre-test gate passed
   and the prompts were frozen, the test accesses of the run itself are accepted.
P7 unit tests pass (pytest -q).
P8 caffeinate is available.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from configure_roles import ROLE_TABLE, SPLASH, assign_roles, check_family_rule, model_families  # noqa: E402
from llm_client import load_models_config, local_transport, read_splash_manifest  # noqa: E402
from pretest_gate import check_test_unopened  # noqa: E402

MIN_PILOT_PER_CONDITION = 25
MIN_FREE_DISK_GB = 30
MAX_SWAP_USED = 0.90
OUT = ROOT_DIR / "results" / "preflight.json"


def check_roles(cfg: dict) -> dict:
    roles = cfg.get("roles") or {}
    comp = roles.get("compressor")
    if comp not in ROLE_TABLE:
        return {"ok": False, "reason": f"compressor {comp!r} is not a ROLE_TABLE candidate"}
    want = assign_roles(comp)
    diff = {k: (roles.get(k), v) for k, v in want.items()
            if k != "downstream" and roles.get(k) != v}
    if roles.get("downstream") != comp:  # load_models_config resolves same_as_compressor
        diff["downstream"] = (roles.get("downstream"), comp)
    fam = check_family_rule(roles, model_families())
    return {"ok": not diff and not fam, "compressor": comp, "mismatch": diff, "family_rule": fam}


def check_pilot(compressor: str, pilot_dir: Path) -> dict:
    stem = compressor.replace(":", "-")
    counts = {}
    for c in ("C0", "C1"):
        p = pilot_dir / f"{stem}_{c}.jsonl"
        counts[c] = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip()) if p.exists() else 0
    return {"ok": all(n >= MIN_PILOT_PER_CONDITION for n in counts.values()), "dir": str(pilot_dir), "records": counts}


def _installed(name: str, tags: set[str]) -> bool:
    return name in tags or (":" not in name and f"{name}:latest" in tags)


def check_ollama(cfg: dict, transport=None) -> dict:
    url = (cfg.get("ollama") or {}).get("url", "http://localhost:11434")
    send = transport or (lambda path, payload: local_transport(url + path, payload, 10))
    models = cfg.get("models") or {}
    role_models = sorted({m for m in (cfg.get("roles") or {}).values()
                          if m and (models.get(m) or {}).get("provider") == "ollama"})
    try:
        tags = {t["name"] for t in send("/api/tags", None).get("models", [])}
        version = send("/api/version", None).get("version")
    except Exception as e:  # noqa: BLE001 - reported as a failure
        return {"ok": False, "reason": f"Ollama not reachable: {e}"}
    missing = [m for m in role_models if not _installed(m, tags)]
    thinking_unset = []
    for m in role_models:
        if m in missing:
            continue
        show = send("/api/show", {"model": m})
        default_on = "thinking" in (show.get("capabilities") or []) and (show.get("thinking") or {}).get("default", True)
        if default_on and (models.get(m) or {}).get("think") is None:
            thinking_unset.append(m)
    return {"ok": not missing and not thinking_unset, "version": version, "role_models": role_models,
            "missing": missing, "thinking_default_on_without_think": thinking_unset}


def check_splash(cfg: dict) -> dict:
    if SPLASH not in set((cfg.get("roles") or {}).values()):
        return {"ok": True, "note": "no role uses Splash"}
    problems = []
    if not shutil.which("splash"):
        problems.append("splash not on PATH")
    try:
        read_splash_manifest(((cfg.get("models") or {}).get(SPLASH) or {}).get("manifest_path", ""))
    except Exception as e:  # noqa: BLE001
        problems.append(str(e))
    return {"ok": not problems, "problems": problems}


def parse_swap(text: str) -> dict:
    """`sysctl -n vm.swapusage` -> {"total_mb", "used_mb"}; e.g. 'total = 26624.00M  used = 26130.19M  free = ...'."""
    v = {k: float(x) for k, x in re.findall(r"(total|used)\s*=\s*([\d.]+)M", text)}
    return {"total_mb": v.get("total", 0.0), "used_mb": v.get("used", 0.0)}


def check_resources(allow_swap: bool = False, swap_text: str | None = None, free_bytes: int | None = None) -> dict:
    free_gb = (shutil.disk_usage(ROOT_DIR).free if free_bytes is None else free_bytes) / 1e9
    if swap_text is None:
        swap_text = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    swap = parse_swap(swap_text)
    frac = swap["used_mb"] / swap["total_mb"] if swap["total_mb"] else 0.0
    mp = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout if shutil.which("memory_pressure") else ""
    m = re.search(r"free percentage:\s*(\d+)%", mp)
    swap_ok = frac < MAX_SWAP_USED or allow_swap
    res = {"ok": free_gb >= MIN_FREE_DISK_GB and swap_ok, "free_disk_gb": round(free_gb, 1),
           "swap_used_fraction": round(frac, 3), "swap": swap, "memory_free_percent": int(m.group(1)) if m else None,
           "allow_swap": allow_swap}
    if frac >= MAX_SWAP_USED:
        res["note"] = (f"swap {frac:.0%} used: a reboot (or quitting large apps) before an unattended run is "
                       "recommended" + ("; accepted by --allow-swap" if allow_swap else ""))
    return res


def check_seal_state(root: Path = ROOT_DIR) -> dict:
    """P6. Before the test: seal intact and no allowed test access. On a resume after the pre-test gate passed
    and the prompts were frozen, allowed test accesses are the legitimate test run, so only the seal is checked."""
    res = check_test_unopened(root)
    if res["ok"] or "allowed_accesses" not in res:
        return res
    gate = root / "results" / "pretest_gate.json"
    gate_ok = gate.exists() and bool(json.loads(gate.read_text(encoding="utf-8")).get("pass"))
    frozen = (root / "results" / "FROZEN_PROMPTS.json").exists()
    if gate_ok and frozen:
        return {"ok": True, "allowed_accesses": res["allowed_accesses"],
                "note": "test already opened after the pre-test gate passed and prompts were frozen (resume)"}
    return {**res, "reason": "test was opened without a passed pre-test gate and frozen prompts"}


def check_tests() -> dict:
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], cwd=ROOT_DIR,
                       capture_output=True, text=True)
    tail = [line for line in p.stdout.splitlines() if line.strip()][-1:] or [""]
    return {"ok": p.returncode == 0, "summary": re.sub(r"\x1b\[[0-9;]*m", "", tail[0])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-swap", action="store_true")
    ap.add_argument("--skip-tests", action="store_true", help="for a quick recheck; the run_lite preflight runs them")
    a = ap.parse_args(argv)
    cfg = load_models_config()
    pilot_dir = Path(os.environ.get("PILOT_DIR", "results/pilot_h2h"))
    pilot_dir = pilot_dir if pilot_dir.is_absolute() else ROOT_DIR / pilot_dir
    checks = {"P1_roles": check_roles(cfg),
              "P2_g0_input": check_pilot((cfg.get("roles") or {}).get("compressor") or "", pilot_dir),
              "P3_ollama": check_ollama(cfg),
              "P4_splash": check_splash(cfg),
              "P5_resources": check_resources(a.allow_swap),
              "P6_seal": check_seal_state(),
              "P7_tests": {"ok": True, "note": "skipped (--skip-tests)"} if a.skip_tests else check_tests(),
              "P8_caffeinate": {"ok": bool(shutil.which("caffeinate"))}}
    res = {"pass": all(c["ok"] for c in checks.values()), "checks": checks}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in checks.items():
        detail = v.get("reason") or v.get("note") or {
            x: v[x] for x in ("mismatch", "family_rule", "records", "missing", "thinking_default_on_without_think",
                              "problems", "summary") if v.get(x)} or ""
        print(f"{k:18s} {'PASS' if v['ok'] else 'FAIL'}  {detail}")
    print(f"preflight: {'PASS' if res['pass'] else 'FAIL'} -> {OUT}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
