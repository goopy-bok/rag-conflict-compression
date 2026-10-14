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


# normalize_output의 (2)단계 보조 함수: 파싱된 JSON 값을 재귀로 돌며 문자열을 등장 순서대로 out에 덧붙인다.
# 사전 키도 넣되 d3·D0 같은 문서 id 키(_DOC_ID)는 뺀다. 반환값 없이 out 리스트를 직접 바꾼다.
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


# 압축기 출력 정규화 N(사전등록 수정 2.2-b·2.2-g). compress가 모델 응답마다 토큰을 세기 전에 적용하므로, 예산 판정·
# 재시도·judge 입력·결과 JSONL의 compressed가 모두 정규화된 텍스트 기준이다. 순서: 제어 토큰 제거 → 출력 전체가 JSON이면
# 문자열만 줄바꿈으로 잇기 → 입력/예시 틀(## Example, [Question] 등)이 다시 시작되는 줄부터 절단(첫머리부터 틀이면
# 그대로 둠) → 끝의 Tokens: 줄·References: 블록 제거. 코드펜스(```)는 따로 제거하지 않는다.
# dspy_adapter.py·pilot_select.py도 이 함수를 쓴다.
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


# 문서 목록을 "[doc_id] 본문" 단락으로 이어 붙여 [Documents] 칸의 내용을 만든다.
def docs_block(docs: list[dict]) -> str:
    out = ""
    for d in docs:
        out += f"[{d['doc_id']}] {d['text']}\n\n"
    return out.strip()


# [Question]·[Documents]·[Budget] 세 칸으로 된 압축기 입력 본문을 만든다. dspy_adapter.py가 같은 형식을 내려고 가져다 쓴다.
def format_fields(question: str, documents: str, budget: int) -> str:
    """Compressor input from already-rendered fields (shared with the DSPy HarnessAdapter)."""
    return (
        f"[Question]\n{question}\n\n"
        f"[Documents]\n{documents}\n\n"
        f"[Budget]\nYour output must be at most {budget} tokens."
    )


# 문항 dict(question, docs)와 예산으로 압축기 입력 본문을 만든다. 조건과 무관하게 같으므로, 조건 간 프롬프트 차이는
# 시스템 지시문과 예시 블록에서만 생긴다. 이 문자열이 바뀌면 파일럿 캐시 키도 바뀐다(docstring).
def format_input(instance: dict, budget: int) -> str:
    """Unified compressor input (identical across conditions; pilot cache keys depend on it)."""
    return format_fields(instance["question"], docs_block(instance["docs"]), budget)


# 예시 블록이 있으면 그 뒤에 빈 줄을 두고 본문을 붙이고, 없으면 본문만 돌려준다. conditions.yaml 기준으로 예시
# 파일이 지정된 조건은 C2·C3-G·C3-M·C3-P·C3-V다.
def assemble_prompt(examples_block: str, body: str) -> str:
    return f"{examples_block}\n\n{body}" if examples_block else body


# 압축 호출의 max_tokens: 예산의 2배와 300 중 큰 값.
def compress_max_tokens(budget: int) -> int:
    return max(budget * 2, 300)


# 고정 few-shot 예시를 "[Examples]" 아래 "## Example k" + format_input 본문 + "[Compressed]" 출력 형식으로 렌더링한다.
# 빈 목록이면 빈 문자열. normalize_output의 _TEMPLATE_LEAK이 출력에서 잘라 내는 줄이 바로 이 틀의 머리줄들이다.
def render_examples(examples: list[dict]) -> str:
    """Fixed few-shot block. Empty list -> empty string."""
    if not examples:
        return ""
    parts = ["[Examples]"]
    for k, ex in enumerate(examples, start=1):
        parts.append(f"## Example {k}\n{format_input(ex, ex['budget'])}\n\n[Compressed]\n{ex['output']}")
    return "\n\n".join(parts)


# 파일 바이트의 SHA-256 16진 문자열. 레코드 prompt_sha256(지시문·예시 파일 해시)의 원천이다.
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 예산 초과 재시도 때 원래 사용자 프롬프트 끝에 붙이는 안내문(직전 토큰 수, 예산, 방해 내용부터 버리라는 지시).
# apo_estimate.py·dspy_adapter.py도 같은 문구를 쓴다.
def retry_note(out_tokens: int, budget: int) -> str:
    return (f"Your previous output was {out_tokens} tokens; rewrite it in at most {budget} tokens, "
            f"dropping distractor content first.")


# C0~C3 모든 조건이 공유하는 LLM 압축기. 조건 간 차이는 시스템 지시문(instruction)과 선택적 고정 few-shot 블록뿐이다.
class PromptCompressor:
    # 지시문은 instruction_text(메모리, APO 채점용)가 있으면 그것을, 없으면 instruction_path 파일을 읽는다. 예시도
    # examples(메모리)가 examples_path보다 우선한다. 해시는 파일이면 파일 바이트, 메모리면 텍스트(예시는 렌더링된 블록)
    # 기준이다. 파일이 없으면 FileNotFoundError. 이 해시들이 compress 반환 meta의 instruction_sha256·examples_sha256이 된다.
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

    # 예시 블록 + 문항 입력을 합친 사용자 프롬프트를 만든다(시스템 프롬프트는 self.system_prompt로 따로 보낸다).
    def build_prompt(self, instance: dict, budget: int) -> str:
        return assemble_prompt(self.examples_block, format_input(instance, budget))

    # 문항 1건을 budget 토큰 안으로 압축한다. temperature 0으로 호출하고 normalize_output 후 count_tokens로 센다.
    # 예산 초과면 원 프롬프트에 retry_note를 붙여 정확히 한 번 다시 호출하고, 두 번째 결과로 대체한다(초과가 계속돼도 첫 결과로
    # 돌아가지 않음). 반환값의 text·tokens·over_budget·retried가 run.py 레코드의 compressed·out_tokens·over_budget·retried가
    # 된다. over_budget은 최종 텍스트 기준이며, True면 CSRJudge·FCRJudge가 LLM을 부르지 않는다. raw_text는 마지막 호출의 정규화 전 출력이다.
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
