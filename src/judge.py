"""
Judges for CCD-G1 (execution guide v2, section 6).

  CSRJudge            compressed context -> supported/distorted flags -> CSR, CSR-lenient, CSR-strict
  FCRJudge            no-conflict instance -> did compression invent a conflict?
  DownstreamGenerator fixed neutral reader over the compressed context
  DJudge              downstream answer -> DEFINITIVE / CONFLICT_ACK / INSUFFICIENT / OTHER (D = CONFLICT_ACK)
  CCDJudge            placeholder until the D1-D7 taxonomy is fixed

Every LLM call goes through `client.call(model=, system=, prompt=, max_tokens=, temperature=0.0)
-> (text, meta)` (LLMClient contract, common brief). Nothing here imports llm_client.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
PROMPTS_DIR = ROOT_DIR / "prompts"
sys.path.insert(0, str(SCRIPT_DIR))


# prompts/ 아래 파일을 읽어 '#'로 시작하는 줄(출처 메모)을 빼고 돌려준다. judge 세 종류와 하류 생성기의 시스템 프롬프트가
# 여기서 온다. 줄 맨 앞이 '#'이면 프롬프트 본문의 마크다운 제목 줄도 함께 빠진다.
def load_prompt(name: str) -> str:
    """Prompt file text with '#' comment lines (provenance notes) removed."""
    text = (PROMPTS_DIR / name).read_text(encoding="utf-8")
    return "\n".join(l for l in text.splitlines() if not l.startswith("#")).strip()


# ----------------------------------------------------------------- JSON parsing

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_OBJ = re.compile(r"\{.*\}", re.S)


# judge 응답을 dict로 읽고 읽은 방식(json | regex | fail)을 함께 돌려준다. 코드펜스 제거 → 전체 json.loads → {...} 구간
# → 키별 정규식 순으로 시도한다. 정규식 경로는 요청한 키가 모두 잡혀야 regex이고, 아니면 잡힌 키만 담아 fail이다.
# CSRJudge·FCRJudge는 이 방식 값을 parse_ok·parse_method에 쓰고, DJudge는 버린다.
def parse_json_reply(text: str, bool_keys: tuple[str, ...] = (), str_keys: tuple[str, ...] = ()) -> tuple[dict, str]:
    """Return (parsed, method), method in json | regex | fail.

    Order: strip code fence -> json.loads -> outermost {...} -> per-key regex.
    Regex counts as success only if every key in bool_keys + str_keys is found.
    """
    t = _FENCE.sub("", (text or "").strip()).strip()
    for cand in (t, *(m.group(0) for m in _OBJ.finditer(t))):
        try:
            d = json.loads(cand)
            if isinstance(d, dict):
                return d, "json"
        except (json.JSONDecodeError, ValueError):
            pass
    d: dict = {}
    for k in bool_keys:
        m = re.search(rf'"?{k}"?\s*:\s*"?(true|false)', t, re.I)
        if m:
            d[k] = m.group(1).lower() == "true"
    for k in str_keys:
        m = re.search(rf'"?{k}"?\s*:\s*"([^"]*)', t, re.I)
        if m:
            d[k] = m.group(1)
    if d and all(k in d for k in (*bool_keys, *str_keys)):
        return d, "regex"
    return d, "fail"


# 판정 값을 bool로 바꾼다. 문자열은 "true"/"yes"/"1"(대소문자·앞뒤 공백 무시)만 True, 그 외 타입은 bool()로 바꾼다.
def _as_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "1"}
    return bool(v)


# --------------------------------------------- value matching (CSR-strict)
# Ported from CAR-exp experiments/classify.py (norm, _numbers, _kind, _date_parts,
# _has_name, value_in_text). Change: NFKC normalisation first, so "m²" matches "m2".

_MARKDOWN = re.compile(r"[*_`#>~\[\]]")
_NUM_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?")
_YEAR = re.compile(r"\b(?:1\d{3}|2\d{3})\b")
_ISO = re.compile(r"\b(1\d{3}|2\d{3})-(\d{1,2})-(\d{1,2})\b")
_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]
_DATEY = re.compile(r"^[\s\d\-‐-―/.,]+$")


# CSR-strict 값 대조용 정규화: NFKC → 소문자 → 마크다운 기호·쉼표 제거 → a-z·0-9·점 외 문자를 공백으로 → 숫자 사이가
# 아닌 점 제거 → 공백 정리. 라틴 소문자·숫자만 남으므로 한글 등 비라틴 문자는 모두 지워진다.
def norm(text: str | None) -> str:
    if not text:
        return ""
    t = _MARKDOWN.sub(" ", unicodedata.normalize("NFKC", text).lower())
    t = t.replace(",", "")
    t = re.sub(r"[^a-z0-9.]+", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# 텍스트 속 숫자 토큰을 쉼표를 뺀 문자열 집합으로 돌려준다. 소수는 끝의 0과 점을 떼어 "2.50"과 "2.5"를 같게 본다.
def _numbers(text: str) -> set[str]:
    out = set()
    for m in _NUM_TOKEN.finditer(text or ""):
        s = m.group(0).replace(",", "").rstrip(".")
        if s:
            out.add(s.rstrip("0").rstrip(".") if "." in s else s)
    return out


# 정답 값의 겉모양으로 종류를 정한다: ISO 날짜·월 이름+연도·숫자/구분자+연도는 DATE, 숫자가 있으면 NUMBER, 그 외 NAME.
# value_in_text가 대조 방식을 고를 때 쓰고, validate_judge.py도 가져다 쓴다.
def value_kind(value: str) -> str:
    """DATE | NUMBER | NAME, from the value's surface form."""
    v = unicodedata.normalize("NFKC", value).strip()
    if _ISO.search(v):
        return "DATE"
    low = v.lower()
    if any(m in low for m in _MONTHS) and _YEAR.search(v):
        return "DATE"
    if _DATEY.match(v) and _YEAR.search(v):
        return "DATE"
    if any(ch.isdigit() for ch in v):
        return "NUMBER"
    return "NAME"


# DATE 값에서 (연도 목록, 월, 일)을 뽑는다. ISO 형식이면 그 세 부분을, 아니면 연도는 4자리 연도 전부, 월은 영어 월 이름,
# 일은 월이 있을 때만 1~2자리 숫자 중 첫 번째로 잡는다. 월·일은 앞자리 0을 뗀 문자열이다.
def _date_parts(value: str) -> tuple[list[str], str | None, str | None]:
    iso = _ISO.search(value)
    if iso:
        return [iso.group(1)], str(int(iso.group(2))), str(int(iso.group(3)))
    years = _YEAR.findall(value)
    low = value.lower()
    month = next((str(i + 1) for i, m in enumerate(_MONTHS) if m in low), None)
    day = None
    if month:
        rest = [n for n in re.findall(r"\b\d{1,2}\b", value)]
        day = str(int(rest[0])) if rest else None
    return years, month, day


# NAME 값 대조: 정규화된 값이 단어 경계로 text에 있으면 True. 여러 단어 값이면 4자 이상인 마지막 단어(성)만으로도
# 일치로 보되, 그 성이 다른 후보(other)에도 있으면 이 완화 규칙을 쓰지 않는다.
def _has_name(value_n: str, text_n: str, other_n: str) -> bool:
    if not value_n:
        return False
    if f" {value_n} " in f" {text_n} ":
        return True
    parts = value_n.split()
    if len(parts) >= 2:
        surname = parts[-1]
        if len(surname) >= 4 and f" {surname} " not in f" {other_n} ":
            return f" {surname} " in f" {text_n} "
    return False


# text가 정답 값 value를 언급하는지 판정한다. DATE는 연도 전부·월(이름 또는 숫자)·일이 있어야 하고, NUMBER는 값의 숫자
# 집합이 text 숫자 집합에 모두 들어 있어야 하며, NAME은 _has_name으로 본다. other(다른 후보 답)는 성만 맞는 애매한
# 일치를 막는 데 쓴다. span_present의 값 대조 부분이며 validate_judge.py도 직접 호출한다.
def value_in_text(value: str | None, text: str, other: str | None = None) -> bool:
    """Does `text` mention `value`? `other` vetoes an ambiguous surname match."""
    if not value:
        return False
    value = unicodedata.normalize("NFKC", value)
    text = unicodedata.normalize("NFKC", text or "")
    kind = value_kind(value)
    text_n = norm(text)
    if kind == "DATE":
        years, month, day = _date_parts(value)
        nums = _numbers(text_n)
        if not years or not all(y in nums for y in years):
            return False
        if month and not (_MONTHS[int(month) - 1] in text.lower() or month in nums):
            return False
        if day and day not in nums:
            return False
        return True
    if kind == "NUMBER":
        want = _numbers(norm(value))
        return bool(want) and want <= _numbers(text_n)
    return _has_name(norm(value), text_n, norm(other))


# CSR-strict용 존재 확인: 정규화한 span이 정규화한 text에 단어 경계로 들어 있거나 value_in_text가 True면 True.
# CSRJudge.evaluate가 압축문에 후보 A·B의 근거(span_a/span_b, 없으면 answer_a/answer_b)가 남았는지 볼 때 쓴다.
def span_present(span: str, text: str, other: str | None = None) -> bool:
    """Normalised substring match, or value match (numbers, dates, names)."""
    return (bool(norm(span)) and f" {norm(span)} " in f" {norm(text)} ") or value_in_text(span, text, other)


# ------------------------------------------------------------------- judges

CSR_KEYS = ("a_supported", "a_distorted", "b_supported", "b_distorted", "conflict_signaled")


# 충돌 문항용 judge. 압축문만 보고 두 후보 답 A·B가 각각 지지되는지(supported)·왜곡됐는지(distorted)와 충돌을
# 알렸는지(conflict_signaled)를 LLM에 묻는다.
class CSRJudge:
    # 시스템 프롬프트를 prompts/judge_csr.txt(기본값, prompt_file로 교체 가능)에서 load_prompt로 읽는다.
    def __init__(self, client, model: str, prompt_file: str = "judge_csr.txt"):
        self.client, self.model = client, model
        self.system_prompt = load_prompt(prompt_file)

    # [QUESTION]·[CONTEXT](압축문)·[CANDIDATE ANSWERS](gold의 answer_a·answer_b)로 judge 사용자 프롬프트를 만든다.
    def build_prompt(self, instance: dict, compressed_text: str) -> str:
        g = instance["gold"]
        return (
            f"[QUESTION]\n{instance['question']}\n\n"
            f"[CONTEXT]\n{compressed_text}\n\n"
            f"[CANDIDATE ANSWERS]\n"
            f"Candidate A: {g['answer_a']}\n"
            f"Candidate B: {g['answer_b']}\n\n"
            f"Evaluate Candidate A and Candidate B according to the system instructions. Return JSON only."
        )

    # 압축문 1건을 판정해 레코드의 judge 필드가 될 dict를 돌려준다(max_tokens 400). over_budget이면 judge를 부르지 않고
    # 모든 플래그 False, csr·csr_lenient·csr_strict 0으로 돌려준다(meta.judge_skipped=True). 응답에 없는 키는 False로 채운다.
    # csr = A·B 모두 supported이고 둘 다 distorted 아님, csr_lenient = A·B 모두 supported(왜곡 무시),
    # csr_strict = csr이면서 span_present로 A·B 근거가 압축문에 문자열로 확인됨. conflict_signaled는 CSR 계산에 쓰지 않고
    # run.py가 레코드의 signal 필드로 옮긴다.
    def evaluate(self, instance: dict, compressed_text: str, over_budget: bool) -> dict:
        out = {"judge_model": self.model, "raw_judge": None, **{k: False for k in CSR_KEYS},
               "csr": 0, "csr_lenient": 0, "csr_strict": 0, "rationale": "",
               "parse_ok": True, "parse_method": None, "meta": {"judge_skipped": True}}
        if over_budget:  # guide v2 s4: over budget -> CSR = 0, judge not called
            out["rationale"] = "over_budget: judge not called"
            return out

        text, meta = self.client.call(model=self.model, system=self.system_prompt,
                                      prompt=self.build_prompt(instance, compressed_text),
                                      max_tokens=400, temperature=0.0)
        parsed, method = parse_json_reply(text, bool_keys=CSR_KEYS, str_keys=("rationale",))
        flags = {k: _as_bool(parsed.get(k, False)) for k in CSR_KEYS}
        g = instance["gold"]
        csr = int(flags["a_supported"] and flags["b_supported"]
                  and not flags["a_distorted"] and not flags["b_distorted"])
        has_a = span_present(g.get("span_a") or g["answer_a"], compressed_text, g["answer_b"])
        has_b = span_present(g.get("span_b") or g["answer_b"], compressed_text, g["answer_a"])
        out.update(flags, raw_judge=parsed, rationale=str(parsed.get("rationale", "")),
                   csr=csr, csr_lenient=int(flags["a_supported"] and flags["b_supported"]),
                   csr_strict=int(csr == 1 and has_a and has_b),
                   parse_ok=method != "fail" and all(k in parsed for k in CSR_KEYS), parse_method=method,
                   meta={**(meta or {}), "judge_skipped": False})
        return out


# 무충돌(distractor, conflict_type == "none") 문항용 judge. 압축이 원래 없던 충돌을 만들어 냈는지(false_conflict)를 판정한다.
class FCRJudge:
    # 시스템 프롬프트를 prompts/judge_fcr.txt에서 읽는다(파일명 고정).
    def __init__(self, client, model: str):
        self.client, self.model = client, model
        self.system_prompt = load_prompt("judge_fcr.txt")

    # [QUESTION]·[REFERENCE ANSWER](gold.answer_a)·[CONTEXT](압축문)로 묻고(max_tokens 200) false_conflict·rationale·parse_ok를
    # 돌려준다. over_budget이면 호출 없이 false_conflict=False로 기록하고, 파싱에서 키를 못 찾아도 False가 된다(parse_ok=False).
    def evaluate(self, instance: dict, compressed_text: str, over_budget: bool) -> dict:
        if over_budget:
            return {"false_conflict": False, "rationale": "over_budget: judge not called",
                    "parse_ok": True, "meta": {"judge_skipped": True}}
        prompt = (f"[QUESTION]\n{instance['question']}\n\n"
                  f"[REFERENCE ANSWER]\n{instance['gold']['answer_a']}\n\n"
                  f"[CONTEXT]\n{compressed_text}\n\nReturn JSON only.")
        text, meta = self.client.call(model=self.model, system=self.system_prompt, prompt=prompt,
                                      max_tokens=200, temperature=0.0)
        parsed, method = parse_json_reply(text, bool_keys=("false_conflict",))
        return {"false_conflict": _as_bool(parsed.get("false_conflict", False)),
                "rationale": str(parsed.get("rationale", "")),
                "parse_ok": method != "fail" and "false_conflict" in parsed,
                "meta": {**(meta or {}), "judge_skipped": False}}


# 하류 독자 모델. 원문 없이 압축문만 보고 질문에 답한다. 시스템 프롬프트는 prompts/downstream.txt다.
class DownstreamGenerator:
    # client·모델·max_tokens(기본 300)를 저장하고 시스템 프롬프트를 prompts/downstream.txt에서 읽는다.
    def __init__(self, client, model: str, max_tokens: int = 300):
        self.client, self.model, self.max_tokens = client, model, max_tokens
        self.system_prompt = load_prompt("downstream.txt")

    # "Context: / Question: / Answer:" 형식으로 한 번 호출해 답변 텍스트를 그대로 돌려준다. 레코드의 downstream.answer가 되며,
    # 호출 meta는 버린다.
    def answer(self, instance: dict, compressed_text: str) -> str:
        prompt = f"Context:\n{compressed_text}\n\nQuestion: {instance['question']}\nAnswer:"
        text, _ = self.client.call(model=self.model, system=self.system_prompt, prompt=prompt,
                                   max_tokens=self.max_tokens, temperature=0.0)
        return text


D_CATEGORIES = {"DEFINITIVE", "CONFLICT_ACK", "INSUFFICIENT", "OTHER"}
D_VALUES = {"A", "B", "NONE", "OTHER"}


# 하류 답변을 DEFINITIVE·CONFLICT_ACK·INSUFFICIENT·OTHER로 분류하고, 답이 확정한 후보(A·B·NONE·OTHER)를 뽑는다.
# CONFLICT_ACK일 때만 D=1이다. 레코드 downstream의 D·category·committed_value가 여기서 나온다.
class DJudge:
    """CAR-exp run_judge.py ported: same text, same category/value sets. D = 1 iff CONFLICT_ACK."""

    # prompts/judge_d.txt('#' 줄 제거 후)를 '---USER---'로 나눠 앞은 시스템 프롬프트, 뒤는 사용자 프롬프트 템플릿으로 쓴다.
    # 구분자가 정확히 한 번 나오지 않으면 언패킹에서 ValueError가 난다.
    def __init__(self, client, model: str):
        self.client, self.model = client, model
        self.system_prompt, self.user_template = load_prompt("judge_d.txt").split("---USER---")
        self.system_prompt, self.user_template = self.system_prompt.strip(), self.user_template.strip()

    # 템플릿에 질문·후보 A/B(없으면 "N/A")·하류 답을 채워 호출하고(max_tokens 120), category·committed_value를 대문자로 읽는다.
    # 허용 집합 밖의 category는 OTHER로 바꾸고 parse_ok=False, committed_value도 집합 밖이면 OTHER다. d = int(category == "CONFLICT_ACK").
    def evaluate(self, instance: dict, answer: str) -> dict:
        g = instance["gold"]
        prompt = self.user_template.format(question=instance["question"], val_A=g.get("answer_a") or "N/A",
                                           val_B=g.get("answer_b") or "N/A", response=answer)
        text, meta = self.client.call(model=self.model, system=self.system_prompt, prompt=prompt,
                                      max_tokens=120, temperature=0.0)
        parsed, _ = parse_json_reply(text, str_keys=("category", "committed_value"))
        cat = str(parsed.get("category", "")).strip().upper()
        val = str(parsed.get("committed_value", "NONE")).strip().upper()
        ok = cat in D_CATEGORIES
        cat = cat if ok else "OTHER"
        val = val if val in D_VALUES else "OTHER"
        return {"category": cat, "committed_value": val, "d": int(cat == "CONFLICT_ACK"),
                "parse_ok": ok, "meta": meta or {}}


# D1~D7 분류 체계가 확정되기 전의 자리표시자. run.py 파이프라인은 이 클래스를 쓰지 않는다(참조는 tests/test_judge.py뿐).
class CCDJudge:
    # client·model을 저장만 한다.
    def __init__(self, client=None, model: str | None = None):
        self.client, self.model = client, model

    # 호출하면 항상 NotImplementedError를 낸다.
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("CCD D1–D7 정의 확정 필요")
