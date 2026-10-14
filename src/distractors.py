"""BM25 distractor retrieval with entity-overlap exclusion."""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from sources.conflictbank import mentions


# 소문자로 바꾼 뒤 정규식 \w+ 로 단어를 뽑는다. 불용어 제거나 어간 처리는 하지 않는다.
# BM25 색인(문서)과 질의(질문) 양쪽에 같은 규칙을 적용한다.
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# 방해(distractor) 문서 후보 풀을 BM25Okapi로 색인해 두고, 질문별로 관련도 높은 문서를 고른다.
# data_build.py의 build_conflictbank(dev/test 쪽 풀 각각)와 build_wikicontradict가 만든다.
class DistractorPool:
    """docs: [{"key", "text", "group"}]. group = subject (ConflictBank) or article title (WikiContradict)."""

    # docs 목록({key, text, group})을 그대로 보관하고, 각 text를 tokenize해 BM25 색인을 만든다.
    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(d["text"]) for d in docs])

    # query(질문)에 대한 BM25 점수 내림차순으로 문서를 훑어 최대 k개를 고른다.
    # exclude_strings(subject·두 정답 등)를 단어 경계 일치로 포함한 문서, exclude_groups에 속한 문서,
    # 이미 고른 문서와 같은 group의 문서는 건너뛴다. 후보가 모자라면 k개 미만을 돌려준다.
    # 주의: depth로 앞부분과 나머지를 나눠 두 번 훑지만 순서가 그대로 이어지므로, 결과는 전체를 한 번 훑는 것과 같다.
    def pick(self, query: str, k: int, exclude_strings: list[str], exclude_groups: set[str] = frozenset(),
             depth: int = 300) -> list[dict]:
        """Top-k BM25 docs for query, skipping docs that mention any exclude string, docs from
        excluded groups, and a second doc from the same group. Fewer than k if the pool runs out."""
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores, kind="stable")
        out, seen = [], set(exclude_groups)
        for pass_order in (order[:depth], order[depth:]):  # deep scan only when the head is exhausted
            for i in pass_order:
                d = self.docs[i]
                if d["group"] in seen or any(mentions(d["text"], s) for s in exclude_strings):
                    continue
                out.append(d)
                seen.add(d["group"])
                if len(out) == k:
                    return out
        return out
