"""
C0-C3 LLM prompt compressor (execution guide v2, section 5).
One class; conditions differ only by the instruction file and the optional fixed few-shot file.

Few-shot examples live outside the instruction text (examples_path, JSON list) so that APO for
C3 can rewrite the instruction but never the examples. Each example is
  {"question": str, "docs": [{"doc_id": str, "text": str}, ...], "budget": int, "output": str}
and is rendered with render_examples() in front of the user prompt as an "[Examples]" block.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from budget import count_tokens, is_over_budget  # noqa: E402

# Output normalization N (preregistration amendment 2.2-b). Applied to every compressor output right after the
# model call, before tokens are counted, so budget, retry and everything downstream see the same text for all
# conditions. Deterministic; a model that already answers in plain text is left unchanged.
_DOC_ID = re.compile(r"^d\d+$", re.I)
_CONTROL_TOKENS = ("<tool_call>", "</tool_call>", "<|im_start|>", "<|im_end|>", "<|endoftext|>",
                   "<start_of_turn>", "<end_of_turn>")
_TRAILING_META = re.compile(r"(?:\n[ \t]*\(?[ \t]*(?:tokens?|token count|word count)\b[^\n]*)+\s*$", re.I)
_TRAILING_REFS = re.compile(r"\n+[ \t]*(?:\*\*)?references?(?:\*\*)?[ \t]*:.*\Z", re.I | re.S)
# Amendment 2.2-g: a line that starts the input/few-shot template again ('## Example 4', '[Question]', bare '##').
_TEMPLATE_LEAK = re.compile(r"^[ \t]*(?:#{1,6}[ \t]*(?:Example\b.*)?|\[(?:Examples?|Question|Documents|Budget|Compressed)\].*)"
                            r"[ \t]*$", re.I | re.M)


def _json_strings(x, out: list[str]) -> None:
    """Every string in a parsed JSON value, keys included, except keys that are bare document ids (d3, D0)."""
    if isinstance(x, str):
        out.append(x)
    elif isinstance(x, dict):
        for k, v in x.items():
            if not _DOC_ID.match(k):
                out.append(k)
            _json_strings(v, out)
    elif isinstance(x, list):
        for v in x:
            _json_strings(v, out)


def normalize_output(text: str) -> str:
    """N: (1) drop chat control tokens, (2) if the whole output is a JSON object/array, replace it by its strings
    in order (doc-id keys dropped; an object with no text left becomes ''), unparseable JSON stays as is,
    (3) cut a run-on into the template: content followed by a line that starts the input/few-shot template again
    is cut at that line (amendment 2.2-g); an output that starts with the template (an echo) is left as is,
    (4) drop a trailing 'Tokens: n' line and a trailing 'References:' block."""
    s = text
    for tok in _CONTROL_TOKENS:
        s = s.replace(tok, "")
    s = s.strip()
    if s[:1] in ("{", "["):
        try:
            parsed = json.loads(s)
        except ValueError:
            parsed = None
        if isinstance(parsed, (dict, list)):
            strings: list[str] = []
            _json_strings(parsed, strings)
            s = "\n".join(t.strip() for t in strings if t and t.strip())
    m = _TEMPLATE_LEAK.search(s)
    if m and m.start() > 0:
        s = s[:m.start()].rstrip()
    s = _TRAILING_META.sub("", s)
    s = _TRAILING_REFS.sub("", s)
    return s.strip()


def docs_block(docs: list[dict]) -> str:
    out = ""
    for d in docs:
        out += f"[{d['doc_id']}] {d['text']}\n\n"
    return out.strip()


def format_fields(question: str, documents: str, budget: int) -> str:
    """Compressor input from already-rendered fields (shared with the DSPy HarnessAdapter)."""
    return (
        f"[Question]\n{question}\n\n"
        f"[Documents]\n{documents}\n\n"
        f"[Budget]\nYour output must be at most {budget} tokens."
    )


def format_input(instance: dict, budget: int) -> str:
    """Unified compressor input (identical across conditions; pilot cache keys depend on it)."""
    return format_fields(instance["question"], docs_block(instance["docs"]), budget)


def assemble_prompt(examples_block: str, body: str) -> str:
    return f"{examples_block}\n\n{body}" if examples_block else body


def compress_max_tokens(budget: int) -> int:
    return max(budget * 2, 300)


def render_examples(examples: list[dict]) -> str:
    """Fixed few-shot block. Empty list -> empty string."""
    if not examples:
        return ""
    parts = ["[Examples]"]
    for k, ex in enumerate(examples, start=1):
        parts.append(f"## Example {k}\n{format_input(ex, ex['budget'])}\n\n[Compressed]\n{ex['output']}")
    return "\n\n".join(parts)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retry_note(out_tokens: int, budget: int) -> str:
    return (f"Your previous output was {out_tokens} tokens; rewrite it in at most {budget} tokens, "
            f"dropping distractor content first.")


class PromptCompressor:
    def __init__(self, condition: str, instruction_path: str | Path | None, model: str, client,
                 examples_path: str | Path | None = None, *, instruction_text: str | None = None,
                 examples: list[dict] | None = None):
        """instruction_text / examples: in-memory alternatives to the two files (APO scoring)."""
        self.condition = condition
        self.model = model
        self.client = client
        self.examples_sha256 = None
        self.examples_block = ""
        if instruction_text is not None:
            self.system_prompt = instruction_text.strip()
            self.instruction_sha256 = hashlib.sha256(instruction_text.encode("utf-8")).hexdigest()
        else:
            instruction_path = Path(instruction_path)
            if not instruction_path.exists():
                raise FileNotFoundError(f"{condition}: instruction file not found: {instruction_path}")
            self.system_prompt = instruction_path.read_text(encoding="utf-8").strip()
            self.instruction_sha256 = _sha256(instruction_path)
        if examples is not None:
            self.examples_block = render_examples(examples)
            self.examples_sha256 = hashlib.sha256(self.examples_block.encode("utf-8")).hexdigest()
        elif examples_path is not None:
            examples_path = Path(examples_path)
            if not examples_path.exists():
                raise FileNotFoundError(f"{condition}: examples file not found: {examples_path}")
            self.examples_sha256 = _sha256(examples_path)
            self.examples_block = render_examples(json.loads(examples_path.read_text(encoding="utf-8")))

    def build_prompt(self, instance: dict, budget: int) -> str:
        return assemble_prompt(self.examples_block, format_input(instance, budget))

    def compress(self, instance: dict, budget: int) -> dict:
        user_prompt = self.build_prompt(instance, budget)
        max_tokens = compress_max_tokens(budget)

        text, meta1 = self.client.call(model=self.model, system=self.system_prompt,
                                       prompt=user_prompt, max_tokens=max_tokens, temperature=0.0)
        raw_text, text = text, normalize_output(text)
        out_tokens = count_tokens(text)
        retried, meta2 = False, None

        # Budget retry rule (guide v2 section 5): exactly one retry.
        if is_over_budget(out_tokens, budget):
            retried = True
            text, meta2 = self.client.call(
                model=self.model, system=self.system_prompt,
                prompt=f"{user_prompt}\n\n{retry_note(out_tokens, budget)}",
                max_tokens=max_tokens, temperature=0.0)
            raw_text, text = text, normalize_output(text)
            out_tokens = count_tokens(text)

        metas = [m for m in (meta1, meta2) if m]
        return {
            "condition": self.condition,
            "text": text.strip(),
            "raw_text": raw_text.strip(),  # model output before N, of the final call
            "tokens": out_tokens,
            "budget": budget,
            "over_budget": is_over_budget(out_tokens, budget),
            "retried": retried,
            "meta": {
                "model": self.model,
                "api_calls": len(metas),
                "call1_cached": meta1.get("cached"),
                "call2_cached": meta2.get("cached") if meta2 else None,
                "legacy_cache": any(m.get("legacy_cache") for m in metas),
                "dry_run": any(m.get("dry_run") for m in metas),
                "latency_ms": sum(m.get("latency_ms") or 0 for m in metas),
                "model_digest": meta1.get("model_digest"),
                "ollama": [m["ollama"] for m in metas if m.get("ollama")],
                "instruction_sha256": self.instruction_sha256,
                "examples_sha256": self.examples_sha256,
            },
        }
