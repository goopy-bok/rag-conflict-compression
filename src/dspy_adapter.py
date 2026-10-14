"""
DSPy layer for the C3 optimizers (GEPA, MIPROv2). Checked against the installed DSPy 3.4.0 source.

Cost control: every LM call DSPy or GEPA makes goes through LLMClient, never through LiteLLM.
  ClientLM(dspy.BaseLM)  legacy forward() plugin (dspy/clients/base_lm.py, engines/legacy_engine.py:
                         a BaseLM subclass that overrides forward() is run by LegacyEngine, which
                         calls forward(prompt=, messages=, **kwargs) and adds no retries or caching).
                         forward -> LLMClient.call_chat, so mock/dry_run/execute, the cost cap, the JSONL
                         cache and the cost log all apply. DSPy 3.4 marks this interface deprecated
                         (removal planned for 3.5); the warning is silenced here, the pin is dspy 3.4.x.
  HarnessAdapter         formats the compressor call exactly like the harness (PromptCompressor):
                         system = instruction, user = examples block + [Question]/[Documents]/[Budget]
                         (+ retry note). The optimized instruction is therefore searched in the same
                         prompt format in which it is evaluated, and calls share the harness cache keys.
  BudgetedCompressor     dspy.Module with one dspy.Predict(Compress); the one-retry budget rule is in
                         forward() (guide 7.1).
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # no model-price download if litellm is imported

import dspy  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from apo_common import judge_and_score, record_examples  # noqa: E402
from budget import count_tokens, is_over_budget  # noqa: E402
from compressors.llm import (assemble_prompt, compress_max_tokens, docs_block, format_fields,  # noqa: E402
                             normalize_output, retry_note)

warnings.filterwarnings("ignore", message="Implementing custom LMs through BaseLM.forward", category=DeprecationWarning)


# DSPy가 하는 모든 LM 호출을 LiteLLM 대신 LLMClient로 보내는 BaseLM 하위 클래스다.
# 그래서 mock/execute 모드, 비용 상한, 캐시, 비용 로그가 본 시험과 같은 경로로 적용된다.
# apo_mipro·apo_gepa에서 과제 모델(compressor 역할)과 제안/반성 모델(reflection 역할)로 쓰인다.
class ClientLM(dspy.BaseLM):
    """dspy LM whose only transport is LLMClient. Exceptions from the client are recorded in
    self.errors (shared by copies) before re-raising, because DSPy's Evaluate turns program
    exceptions into failure scores and would otherwise hide them (e.g. BudgetExceeded)."""

    # 모델 이름·온도·max_tokens를 BaseLM에 넘기되 DSPy 자체 캐시와 재시도는 끈다(cache=False, num_retries=0). 오류 목록 errors를 만든다.
    def __init__(self, client, model: str, temperature: float = 0.0, max_tokens: int = 1000):
        super().__init__(model=model, model_type="chat", temperature=temperature, max_tokens=max_tokens,
                         cache=False, num_retries=0)
        self.client = client
        self.errors: list[str] = []

    # DSPy legacy 경로가 부르는 실제 호출. 생성자 kwargs 위에 호출 kwargs(None 제외)를 덮고, prompt만 오면 user 메시지 하나로 감싸
    # client.call_chat을 부른다. rollout_id가 있으면 cache_salt "rollout:<id>"로 캐시 키를 나눈다.
    # 응답은 OpenAI 형식 dict(choices, usage)로 돌려주며, usage의 입력 토큰은 meta의 추정치다. 예외는 errors에 적고 다시 던진다.
    def forward(self, prompt=None, messages=None, **kwargs):
        merged = {**self.kwargs, **{k: v for k, v in kwargs.items() if v is not None}}
        messages = messages or [{"role": "user", "content": prompt}]
        rollout = merged.get("rollout_id")
        try:
            text, meta = self.client.call_chat(
                self.model, messages, max_tokens=int(merged.get("max_tokens") or 1000),
                temperature=float(merged.get("temperature") or 0.0),
                cache_salt=None if rollout is None else f"rollout:{rollout}")
        except Exception as e:
            self.errors.append(f"{type(e).__name__}: {e}")
            raise
        out_tok = meta.get("output_tokens", meta.get("output_tokens_est", 0))
        return {"model": self.model,
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": meta.get("input_tokens_est", 0), "completion_tokens": out_tok,
                          "total_tokens": meta.get("input_tokens_est", 0) + out_tok}}

    # 비동기 인터페이스. forward를 그대로 동기 호출한다.
    async def aforward(self, prompt=None, messages=None, **kwargs):
        return self.forward(prompt=prompt, messages=messages, **kwargs)

    # 깊은 복사 때 자기 자신을 돌려준다. LLMClient가 잠금을 가지고 있어서이며, 그 결과 프로그램 복사본들이 같은 client와 errors를 공유한다.
    def __deepcopy__(self, memo):  # LLMClient holds a lock; programs may be deep-copied
        return self


# 압축 호출 메시지를 본 시험 하네스(PromptCompressor)와 같은 형식으로 만드는 ChatAdapter다.
# system = 지시문, user = 예시 블록 + [Question]/[Documents]/[Budget] (+ 재시도 안내)이며 DSPy의 필드 마커 형식은 쓰지 않는다.
# 그래서 최적화 중 지시문이 시험 때와 같은 프롬프트 형식 안에서 탐색된다.
class HarnessAdapter(dspy.ChatAdapter):
    # retry에 재시도 안내 문장(retry_note)을 받는다. 빈 문자열이면 첫 시도다. JSON adapter 대체 경로는 끈다.
    def __init__(self, retry: str = ""):
        super().__init__(use_json_adapter_fallback=False)
        self.retry = retry

    # DSPy 데모(demos)가 있으면 ValueError를 낸다. 3-shot은 examples 입력 필드로만 들어가야 하기 때문이다.
    # 예시 블록 해시를 record_examples로 기록하고, assemble_prompt·format_fields로 user 메시지를 만들어 [system, user] 목록을 돌려준다.
    def format(self, signature, demos, inputs):
        if demos:
            raise ValueError("HarnessAdapter: demos must stay empty (few-shot goes through the examples field)")
        record_examples(inputs.get("examples") or "")
        user = assemble_prompt(inputs.get("examples") or "",
                               format_fields(inputs["question"], inputs["documents"], inputs["budget"]))
        if self.retry:
            user = f"{user}\n\n{self.retry}"
        return [{"role": "system", "content": signature.instructions.strip()}, {"role": "user", "content": user}]

    # 모델 응답 전체를 그대로 compressed 필드로 돌려준다(필드 파싱 없음). 출력 정규화는 BudgetedCompressor._call이 한다.
    def parse(self, signature, completion):
        return {"compressed": completion}


# 압축 과제의 DSPy 시그니처. 입력은 examples·question·documents·budget, 출력은 compressed다.
# docstring은 자리표시이고 BudgetedCompressor가 with_instructions로 시작 지시문을 넣는다. C3 최적화기가 탐색하는 대상이 이 instructions다.
class Compress(dspy.Signature):
    """(replaced by the start instruction)"""
    examples: str = dspy.InputField(desc="fixed worked examples; may be empty")
    question: str = dspy.InputField()
    documents: str = dspy.InputField(desc="[d0] ... [d1] ...")
    budget: int = dspy.InputField(desc="max output tokens")
    compressed: str = dspy.OutputField()


# 지시문 하나와 고정 예시 블록을 가진 압축 모듈(dspy.Predict 하나)이다. MIPROv2·GEPA가 최적화하는 프로그램이며,
# 예산 초과 시 1회 재시도 규칙을 forward에 담아 하네스와 같은 조건으로 출력을 만든다. 결과 지시문은 program_instruction으로 꺼낸다.
class BudgetedCompressor(dspy.Module):
    # 고정 예시 블록을 보관하고, 시작 지시문(앞뒤 공백 제거)을 Compress 시그니처의 instructions로 넣은 Predict를 만든다.
    def __init__(self, instruction: str, examples_block: str):
        super().__init__()
        self.examples = examples_block
        self.predict = dspy.Predict(Compress.with_instructions(instruction.strip()))

    # 압축 호출 한 번. 이 호출에서만 HarnessAdapter(retry)를 쓰도록 context를 바꾸고,
    # max_tokens = compress_max_tokens(budget)(= max(2 × budget, 300)), temperature 0.0으로 부른다.
    # 응답은 PromptCompressor.compress와 같은 normalize_output으로 정규화해 돌려준다.
    def _call(self, question, documents, budget, retry=""):
        with dspy.context(adapter=HarnessAdapter(retry)):
            out = self.predict(examples=self.examples, question=question, documents=documents, budget=budget,
                               config={"max_tokens": compress_max_tokens(budget), "temperature": 0.0})
        return normalize_output(out.compressed or "")  # same N as PromptCompressor.compress (amendment 2.2-b)

    # 압축 후 토큰 수가 예산을 넘으면 retry_note를 붙여 정확히 한 번 다시 부른다. 지표는 재시도 후 최종 출력만 본다.
    # 반환 Prediction: compressed(앞뒤 공백 제거), out_tokens, over_budget(재시도 후에도 초과인지), retried.
    def forward(self, question, documents, budget):
        text = self._call(question, documents, budget)
        n, retried = count_tokens(text), False
        if is_over_budget(n, budget):  # guide 5 / 7.1: exactly one retry, metric sees the final output
            text, retried = self._call(question, documents, budget, retry_note(n, budget)), True
            n = count_tokens(text)
        return dspy.Prediction(compressed=text.strip(), out_tokens=n, over_budget=is_over_budget(n, budget),
                               retried=retried)


# (문항, r) 목록을 dspy.Example 목록으로 바꾼다. id·r·question·문서 블록(docs_block)·예산(compute_budget(total_tokens, r))을 담고
# 입력 필드는 question·documents·budget이다. examples 입력은 BudgetedCompressor가 직접 채운다.
def to_examples(items: list[tuple[dict, float]]) -> list[dspy.Example]:
    from budget import compute_budget
    return [dspy.Example(id=inst["id"], r=r, question=inst["question"], documents=docs_block(inst["docs"]),
                         budget=compute_budget(inst["total_tokens"], r)).with_inputs("question", "documents", "budget")
            for inst, r in items]


# DSPy 최적화기용 지표 함수를 만든다. gold.id로 원 문항을 찾아 apo_common.judge_and_score로 채점한다.
# feedback=True면 Prediction(score, feedback)을 돌려주는 GEPA용, False면 float 점수만 돌려주는 MIPROv2용이다.
def make_metric(instances: dict[str, dict], client, judge_model: str, *, feedback: bool):
    """GEPA metric (feedback=True -> dspy.Prediction(score, feedback)) or MIPROv2 metric (float).
    The score and feedback come from apo_common.judge_and_score (single definition)."""
    # DSPy 지표 시그니처(gold, pred, trace, pred_name, pred_trace)를 따르며 trace 관련 인자는 쓰지 않는다.
    # 예산은 gold.budget에서, 출력·토큰 수·초과 여부는 pred에서 읽는다.
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        s = judge_and_score(instances[gold.id], pred.compressed, pred.out_tokens, gold.budget, pred.over_budget,
                            client, judge_model)
        return dspy.Prediction(score=s["score"], feedback=s["feedback"]) if feedback else s["score"]
    return metric


# 프로그램의 predictor별 데모 개수를 {이름: 개수}로 돌려준다. 데모가 붙지 않았는지(0) 로그에 남기는 용도다.
def program_demos(program: dspy.Module) -> dict[str, int]:
    return {name: len(p.demos) for name, p in program.named_predictors()}


# 프로그램 predict 시그니처의 instructions(최적화된 지시문)를 앞뒤 공백 제거해 돌려준다. self.predict가 있는 BudgetedCompressor 구조를 전제한다.
def program_instruction(program: dspy.Module) -> str:
    return program.predict.signature.instructions.strip()
