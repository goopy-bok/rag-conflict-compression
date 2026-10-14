"""
Deterministic offline LLM stand-in (no network, no cost). Used by LLMClient(mode="mock").

Role detection (case-insensitive, on system + "\n" + prompt; first matching rule wins).
Keywords are the JSON field names / labels each judge must ask for, so the rules survive
prompt-file rewording as long as the output schema stays the same.
  0a. GEPA reflection : "write a new instruction for the assistant" (gepa InstructionProposalSignature
                        default template) -> the current instruction (first ``` block) plus one
                        deterministic extra line, inside ``` fences (what gepa's output_extractor reads)
  0b. DSPy ChatAdapter: "[[ ## completed ## ]]" marker -> every requested output field as
                        "[[ ## name ## ]]" sections + "[[ ## completed ## ]]". Field values: "compressed"
                        -> documents cut to the budget; names containing "instruction" -> the input
                        instruction plus one line; typed fields -> a valid literal; others -> short text
  1. D judge      : "committed_value" or "conflict_ack"  -> {"category", "committed_value"}
  2. FCR judge    : "false_conflict"                      -> {"false_conflict", "rationale"}
  3. CSR judge    : "a_supported"                         -> {"a_supported","a_distorted","b_supported",
                                                               "b_distorted","conflict_signaled","rationale"}
  4. Compressor   : "at most <N> tokens" in prompt, or ("compress" in system and "[Documents]" in prompt)
                    -> text of the LAST "[Documents]" block of the prompt, truncated to N-1 tokens
                       (cl100k_base; N = last "at most N tokens" match, default max_tokens)
  5. otherwise    : downstream generator -> short one-line answer
Judge values vary deterministically with sha256(model, system, prompt), so outputs are
reproducible but not constant.
"""
from __future__ import annotations

import hashlib
import json
import re

from budget import get_encoder

_AT_MOST = re.compile(r"at most (\d+) tokens", re.I)
_FIELD = re.compile(r"\[\[ ## (\w+) ## \]\]")
_REQUESTED = re.compile(r"`\[\[ ## (\w+) ## \]\]`(?: \(must be formatted as a valid Python (\w+))?")
_GEPA_MARK = "write a new instruction for the assistant"
_TYPED = {"int": "0", "float": "0.0", "bool": "False", "list": "[]", "dict": "{}"}


def _cut(text: str, limit: int) -> str:
    return get_encoder().decode(get_encoder().encode(text)[: max(limit - 1, 0)])


def _gepa_reflection(prompt: str, h: int) -> str:
    current = prompt.split("```", 2)[1].strip() if prompt.count("```") >= 2 else ""
    return f"```\n{current}\nRevision {h % 1000}: keep every answer value together with its document id.\n```"


def _chat_adapter(prompt: str, h: int) -> str:
    sections, pos = {}, list(_FIELD.finditer(prompt))
    for m, nxt in zip(pos, pos[1:] + [None]):
        sections[m.group(1)] = prompt[m.end(): nxt.start() if nxt else len(prompt)].strip()
    tail = prompt.rsplit("Respond with the corresponding output fields", 1)[-1]
    out = []
    for name, typ in _REQUESTED.findall(tail):
        if name == "completed":
            continue
        if name == "compressed":
            budget = re.search(r"\d+", sections.get("budget", ""))
            val = _cut(sections.get("documents", ""), int(budget.group(0)) if budget else 100)
        elif "instruction" in name:
            base = next((sections[k] for k in ("basic_instruction", "current_instruction", "instruction")
                         if sections.get(k)), "Compress the documents for the question.")
            val = f"{base}\nKeep every answer value together with its document id. (mock variant {h % 1000})"
        else:
            val = _TYPED.get(typ, f"mock {name} {h % 1000}")
        out.append(f"[[ ## {name} ## ]]\n{val}")
    return "\n\n".join(out + ["[[ ## completed ## ]]"])


def _h(*parts: str) -> int:
    return int(hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest(), 16)


def _compress(prompt: str, max_tokens: int) -> str:
    found = _AT_MOST.findall(prompt)
    limit = int(found[-1]) if found else max_tokens
    docs = prompt.rsplit("[Documents]", 1)[-1].split("[Budget]", 1)[0].strip()
    return _cut(docs, limit)


def respond(model: str, system: str, prompt: str, max_tokens: int) -> str:
    text = f"{system}\n{prompt}".lower()
    h = _h(model, system, prompt)
    bit = lambda i: bool((h >> i) & 1)  # noqa: E731

    if _GEPA_MARK in text:
        return _gepa_reflection(prompt, h)
    if "[[ ## completed ## ]]" in f"{system}\n{prompt}":
        return _chat_adapter(prompt, h)
    if "committed_value" in text or "conflict_ack" in text:
        cat = ["DEFINITIVE", "CONFLICT_ACK", "INSUFFICIENT", "OTHER"][h % 4]
        val = ("A" if bit(2) else "B") if cat == "DEFINITIVE" else "NONE"
        return json.dumps({"category": cat, "committed_value": val})
    if "false_conflict" in text:
        return json.dumps({"false_conflict": h % 4 == 0, "rationale": "mock"})
    if "a_supported" in text:
        return json.dumps({
            "a_supported": h % 4 != 0, "a_distorted": (h >> 2) % 8 == 0,
            "b_supported": bit(5), "b_distorted": (h >> 6) % 8 == 0,
            "conflict_signaled": bit(9), "rationale": "mock",
        })
    if _AT_MOST.search(prompt) or ("compress" in system.lower() and "[documents]" in text):
        return _compress(prompt, max_tokens)
    return f"Mock answer {h % 1000}."
