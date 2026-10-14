"""
Budget & Token counting utility for CCD-G1.
Fixed on cl100k_base tokenizer.
"""
from __future__ import annotations
import os
import math

os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.expanduser("~/.cache/tiktoken"))

import tiktoken

_ENCODER = None

# cl100k_base 인코더를 처음 호출할 때 한 번만 만들어 모듈 전역 _ENCODER에 캐시하고 돌려준다.
# count_tokens 외에 mock_llm.py도 이 인코더로 출력을 토큰 단위로 자른다.
def get_encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER

# 텍스트의 cl100k_base 토큰 수를 센다. 빈 문자열·None이면 인코더를 부르지 않고 0을 돌려준다.
# 문서 길이 창(data_build), 인스턴스 total_tokens, 압축 출력 길이 측정이 모두 이 함수 하나를 쓴다.
def count_tokens(text: str) -> int:
    """Count tokens in text using cl100k_base."""
    if not text:
        return 0
    enc = get_encoder()
    return len(enc.encode(text))

# 압축 예산 B = floor(total_tokens × r)를 계산한다. r 기본값은 0.3이다.
# run.py가 인스턴스의 total_tokens로 호출해 압축기에 넘길 토큰 상한을 정한다.
def compute_budget(total_tokens: int, r: float = 0.3) -> int:
    """Compute token budget: B = floor(total_tokens * r)."""
    return math.floor(total_tokens * r)

# 출력 토큰 수가 예산을 엄격히 넘는지(out_tokens > budget) 판정한다. 예산과 같으면 초과가 아니다.
# compressors/llm.py와 dspy_adapter.py가 재시도 여부와 over_budget 기록에 쓴다.
def is_over_budget(out_tokens: int, budget: int) -> bool:
    """Return True if out_tokens exceeds budget."""
    return out_tokens > budget
