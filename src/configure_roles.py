"""
Role assignment after the compressor selection pilot (preregistration 2026-10-09, section 2.3).

  uv run python src/configure_roles.py --compressor qwen2.5:7b

Rewrites only the `roles:` block of configs/models.yaml (comments elsewhere kept); the previous file
is copied to configs/models.yaml.bak, then the family rule (check_family_rule) is checked on the written
file. D judge = eval judge (test) / opt judge (dev, G1-G2) is handled by run.default_judge_role, not here.
Judge fallbacks after G0 (2.4): fallback() gives the next model; it is applied by hand (write_roles).

JSON mode (llm_client): a model holding a judge role and a non-judge role at once gets no JSON format.
The revised 2.3 table has no such model (tests/test_splash.py). Within the 2.4 orders the only case is
compressor qwen2.5:14b with opt_judge falling back to qwen2.5:14b (allowed by the family rule: the opt judge
may share the compressor's family) -> that model then runs its judge calls without JSON format. Eval-judge
fallbacks cannot collide (a shared model would share the family, which the rule forbids).
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_CONFIG = SCRIPT_DIR.parent / "configs" / "models.yaml"
sys.path.insert(0, str(SCRIPT_DIR))
SPLASH = "incoai/Qwen3.8-27B-Splash"

# Preregistration 2.3 table (revised 2026-10-09, before the pilot): compressor -> (opt judge,
# instruction generator, eval judge).
ROLE_TABLE = {
    "llama3.1:8b": ("gemma2:9b", "mistral-small", SPLASH),
    "phi3.5": ("gemma2:9b", "mistral-small", SPLASH),
    "qwen2.5:7b": ("gemma2:9b", SPLASH, "mistral-small"),
    "qwen2.5:14b": ("gemma2:9b", SPLASH, "mistral-small"),
    "gemma2:9b": ("llama3.1:8b", "mistral-small", SPLASH),
    "gemma4:e4b": ("llama3.1:8b", "mistral-small", SPLASH),  # amendment 2.2-b; same row as the other Google compressor
}
# Preregistration 2.4: replacements after a G0 failure, tried after the table model, in this order.
FALLBACK_ORDER = {"eval_judge": ("qwen2.5:14b", "gemma2:27b", "mistral-small"),
                  "opt_judge": ("qwen2.5:14b", "gemma2:27b")}
ROLE_ORDER = ("compressor", "opt_judge", "eval_judge", "downstream", "reflection", "aux_compressor")


# 선택된 압축기 모델명으로 사전등록 2.3 표(ROLE_TABLE)를 찾아 역할 사전을 만든다.
# 반환: compressor·opt_judge·eval_judge·reflection 모델명, downstream="same_as_compressor", aux_compressor=None.
# ROLE_TABLE에 없는 모델이면 ValueError를 낸다. pilot_select.py·preflight.py도 이 함수로 기대 역할을 만든다.
def assign_roles(compressor_model: str) -> dict:
    """Roles for a selected compressor (2.3). Downstream generator = the compressor; reflection =
    instruction generator for GEPA, ProTeGi and MIPROv2; no auxiliary compressor."""
    if compressor_model not in ROLE_TABLE:
        raise ValueError(f"{compressor_model!r} is not a compressor candidate; choose from {list(ROLE_TABLE)}")
    opt_judge, reflection, eval_judge = ROLE_TABLE[compressor_model]
    return {"compressor": compressor_model, "opt_judge": opt_judge, "eval_judge": eval_judge,
            "downstream": "same_as_compressor", "reflection": reflection, "aux_compressor": None}


# models.yaml의 models 항목을 읽어 {모델명: family} 사전을 돌려준다. family가 없으면 값은 None이다.
def model_families(path: Path = MODELS_CONFIG) -> dict[str, str | None]:
    models = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("models") or {}
    return {m: (v or {}).get("family") for m, v in models.items()}


# 사전등록 2.3의 계열 규칙을 검사한다: eval_judge의 family가 compressor·opt_judge·reflection 중 어느 것과도 같으면 안 된다.
# 위반 설명 문자열 목록을 돌려주며, 빈 목록이면 규칙을 지킨 것이다. family가 없는 모델도 위반으로 센다.
# downstream과 aux_compressor는 검사하지 않는다. preflight.py와 main()이 이 함수로 최종 배정을 확인한다.
def check_family_rule(roles: dict, families: dict[str, str | None] | None = None) -> list[str]:
    """2.3 rule: family(eval_judge) not in families of {compressor, opt_judge, reflection}. Returns the
    violations (empty list = rule holds). A model without a family in models.yaml is a violation."""
    families = model_families() if families is None else families
    fam = {r: families.get(roles.get(r)) for r in ("compressor", "opt_judge", "reflection", "eval_judge")}
    bad = [f"{r} {roles.get(r)!r} has no family in models.yaml" for r, f in fam.items() if f is None]
    bad += [f"eval_judge {roles['eval_judge']!r} and {r} {roles[r]!r} are both {fam[r]}"
            for r in ("compressor", "opt_judge", "reflection") if fam["eval_judge"] and fam[r] == fam["eval_judge"]]
    return bad


# G0 실패 뒤 판정자 역할(eval_judge 또는 opt_judge)의 다음 후보 모델을 고른다(사전등록 2.4).
# 2.3 표의 모델 → FALLBACK_ORDER[role] 순서로 보며, failed_models에 있거나 계열 규칙을 깨는 모델은 건너뛴다.
# 후보가 다 떨어지면 None(실험 중단·보고)을 돌려준다. 그 밖의 role 값이면 FALLBACK_ORDER에서 KeyError가 난다.
# src 안에서는 호출하는 곳이 없고 tests/test_splash.py만 쓴다. 결과는 write_roles로 사람이 직접 반영한다.
def fallback(role: str, compressor: str, failed_models, roles: dict | None = None,
             families: dict[str, str | None] | None = None) -> str | None:
    """Next model for a judge role after G0 failures (2.4): the 2.3 table model, then FALLBACK_ORDER[role],
    skipping models in failed_models and models that would break check_family_rule. `roles` = current
    assignment (default: the 2.3 table), so a second replacement sees the first one. None = order exhausted
    (2.3: stop the experiment and report)."""
    roles = dict(roles or assign_roles(compressor))
    families = model_families() if families is None else families
    for m in (ROLE_TABLE[compressor][2 if role == "eval_judge" else 0], *FALLBACK_ORDER[role]):
        if m not in failed_models and not check_family_rule({**roles, role: m}, families):
            return m
    return None


# 역할 사전을 ROLE_ORDER 순서의 YAML `roles:` 블록 문자열로 만든다. None 값은 null로 쓴다.
def render_roles(roles: dict) -> str:
    return "roles:\n" + "".join(f"  {k}: {'null' if roles[k] is None else roles[k]}\n" for k in ROLE_ORDER)


# models.yaml에서 최상위 `roles:` 블록 하나만 render_roles 결과로 바꿔 쓰고, 다른 부분(주석 포함)은 그대로 둔다.
# 쓰기 전에 원본을 models.yaml.bak으로 복사하며(이전 .bak은 덮어씀), 그 백업 경로를 돌려준다.
# `roles:` 블록이 정확히 하나가 아니면 SystemExit로 멈춘다.
def write_roles(roles: dict, path: Path = MODELS_CONFIG) -> Path:
    text = path.read_text(encoding="utf-8")
    block = re.compile(r"^roles:\n(?:[ \t]+\S.*\n)*", re.M)  # indented lines up to the first blank line
    if len(block.findall(text)) != 1:
        raise SystemExit(f"expected exactly one top-level 'roles:' block in {path}")
    backup = path.with_name(path.name + ".bak")
    shutil.copy2(path, backup)
    path.write_text(block.sub(lambda m: render_roles(roles), text), encoding="utf-8")
    return backup


# CLI 진입점이다. --compressor로 역할을 배정해 models.yaml에 쓰고, llm_client.load_models_config로
# same_as_compressor를 풀어 다시 읽은 역할에 check_family_rule을 적용한다. 위반이 있으면 SystemExit로 끝난다.
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compressor", required=True, choices=list(ROLE_TABLE))
    a = ap.parse_args(argv)
    roles = assign_roles(a.compressor)
    backup = write_roles(roles, MODELS_CONFIG)
    print(render_roles(roles), end="")
    print(f"written {MODELS_CONFIG} (backup {backup})")
    from llm_client import load_models_config  # roles as written, same_as_compressor resolved
    bad = check_family_rule(load_models_config(MODELS_CONFIG)["roles"])
    if bad:
        raise SystemExit("family rule (2.3) violated: " + "; ".join(bad))
    print("family rule (2.3): ok")


if __name__ == "__main__":
    main()
