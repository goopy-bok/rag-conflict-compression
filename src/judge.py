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


def load_prompt(name: str) -> str:
    """Prompt file text with '#' comment lines (provenance notes) removed."""
    text = (PROMPTS_DIR / name).read_text(encoding="utf-8")
    return "\n".join(l for l in text.splitlines() if not l.startswith("#")).strip()


# ----------------------------------------------------------------- JSON parsing

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_OBJ = re.compile(r"\{.*\}", re.S)


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


def norm(text: str | None) -> str:
    if not text:
        return ""
    t = _MARKDOWN.sub(" ", unicodedata.normalize("NFKC", text).lower())
    t = t.replace(",", "")
    t = re.sub(r"[^a-z0-9.]+", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _numbers(text: str) -> set[str]:
    out = set()
    for m in _NUM_TOKEN.finditer(text or ""):
        s = m.group(0).replace(",", "").rstrip(".")
        if s:
            out.add(s.rstrip("0").rstrip(".") if "." in s else s)
    return out


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


def span_present(span: str, text: str, other: str | None = None) -> bool:
    """Normalised substring match, or value match (numbers, dates, names)."""
    return (bool(norm(span)) and f" {norm(span)} " in f" {norm(text)} ") or value_in_text(span, text, other)


# ------------------------------------------------------------------- judges

CSR_KEYS = ("a_supported", "a_distorted", "b_supported", "b_distorted", "conflict_signaled")


class CSRJudge:
    def __init__(self, client, model: str, prompt_file: str = "judge_csr.txt"):
        self.client, self.model = client, model
        self.system_prompt = load_prompt(prompt_file)

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


class FCRJudge:
    def __init__(self, client, model: str):
        self.client, self.model = client, model
        self.system_prompt = load_prompt("judge_fcr.txt")

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


class DownstreamGenerator:
    def __init__(self, client, model: str, max_tokens: int = 300):
        self.client, self.model, self.max_tokens = client, model, max_tokens
        self.system_prompt = load_prompt("downstream.txt")

    def answer(self, instance: dict, compressed_text: str) -> str:
        prompt = f"Context:\n{compressed_text}\n\nQuestion: {instance['question']}\nAnswer:"
        text, _ = self.client.call(model=self.model, system=self.system_prompt, prompt=prompt,
                                   max_tokens=self.max_tokens, temperature=0.0)
        return text


D_CATEGORIES = {"DEFINITIVE", "CONFLICT_ACK", "INSUFFICIENT", "OTHER"}
D_VALUES = {"A", "B", "NONE", "OTHER"}


class DJudge:
    """CAR-exp run_judge.py ported: same text, same category/value sets. D = 1 iff CONFLICT_ACK."""

    def __init__(self, client, model: str):
        self.client, self.model = client, model
        self.system_prompt, self.user_template = load_prompt("judge_d.txt").split("---USER---")
        self.system_prompt, self.user_template = self.system_prompt.strip(), self.user_template.strip()

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


class CCDJudge:
    def __init__(self, client=None, model: str | None = None):
        self.client, self.model = client, model

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("CCD D1–D7 정의 확정 필요")
