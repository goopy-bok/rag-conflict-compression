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


class ClientLM(dspy.BaseLM):
    """dspy LM whose only transport is LLMClient. Exceptions from the client are recorded in
    self.errors (shared by copies) before re-raising, because DSPy's Evaluate turns program
    exceptions into failure scores and would otherwise hide them (e.g. BudgetExceeded)."""

    def __init__(self, client, model: str, temperature: float = 0.0, max_tokens: int = 1000):
        super().__init__(model=model, model_type="chat", temperature=temperature, max_tokens=max_tokens,
                         cache=False, num_retries=0)
        self.client = client
        self.errors: list[str] = []

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

    async def aforward(self, prompt=None, messages=None, **kwargs):
        return self.forward(prompt=prompt, messages=messages, **kwargs)

    def __deepcopy__(self, memo):  # LLMClient holds a lock; programs may be deep-copied
        return self


class HarnessAdapter(dspy.ChatAdapter):
    def __init__(self, retry: str = ""):
        super().__init__(use_json_adapter_fallback=False)
        self.retry = retry

    def format(self, signature, demos, inputs):
        if demos:
            raise ValueError("HarnessAdapter: demos must stay empty (few-shot goes through the examples field)")
        record_examples(inputs.get("examples") or "")
        user = assemble_prompt(inputs.get("examples") or "",
                               format_fields(inputs["question"], inputs["documents"], inputs["budget"]))
        if self.retry:
            user = f"{user}\n\n{self.retry}"
        return [{"role": "system", "content": signature.instructions.strip()}, {"role": "user", "content": user}]

    def parse(self, signature, completion):
        return {"compressed": completion}


class Compress(dspy.Signature):
    """(replaced by the start instruction)"""
    examples: str = dspy.InputField(desc="fixed worked examples; may be empty")
    question: str = dspy.InputField()
    documents: str = dspy.InputField(desc="[d0] ... [d1] ...")
    budget: int = dspy.InputField(desc="max output tokens")
    compressed: str = dspy.OutputField()


class BudgetedCompressor(dspy.Module):
    def __init__(self, instruction: str, examples_block: str):
        super().__init__()
        self.examples = examples_block
        self.predict = dspy.Predict(Compress.with_instructions(instruction.strip()))

    def _call(self, question, documents, budget, retry=""):
        with dspy.context(adapter=HarnessAdapter(retry)):
            out = self.predict(examples=self.examples, question=question, documents=documents, budget=budget,
                               config={"max_tokens": compress_max_tokens(budget), "temperature": 0.0})
        return normalize_output(out.compressed or "")  # same N as PromptCompressor.compress (amendment 2.2-b)

    def forward(self, question, documents, budget):
        text = self._call(question, documents, budget)
        n, retried = count_tokens(text), False
        if is_over_budget(n, budget):  # guide 5 / 7.1: exactly one retry, metric sees the final output
            text, retried = self._call(question, documents, budget, retry_note(n, budget)), True
            n = count_tokens(text)
        return dspy.Prediction(compressed=text.strip(), out_tokens=n, over_budget=is_over_budget(n, budget),
                               retried=retried)


def to_examples(items: list[tuple[dict, float]]) -> list[dspy.Example]:
    from budget import compute_budget
    return [dspy.Example(id=inst["id"], r=r, question=inst["question"], documents=docs_block(inst["docs"]),
                         budget=compute_budget(inst["total_tokens"], r)).with_inputs("question", "documents", "budget")
            for inst, r in items]


def make_metric(instances: dict[str, dict], client, judge_model: str, *, feedback: bool):
    """GEPA metric (feedback=True -> dspy.Prediction(score, feedback)) or MIPROv2 metric (float).
    The score and feedback come from apo_common.judge_and_score (single definition)."""
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        s = judge_and_score(instances[gold.id], pred.compressed, pred.out_tokens, gold.budget, pred.over_budget,
                            client, judge_model)
        return dspy.Prediction(score=s["score"], feedback=s["feedback"]) if feedback else s["score"]
    return metric


def program_demos(program: dspy.Module) -> dict[str, int]:
    return {name: len(p.demos) for name, p in program.named_predictors()}


def program_instruction(program: dspy.Module) -> str:
    return program.predict.signature.instructions.strip()
