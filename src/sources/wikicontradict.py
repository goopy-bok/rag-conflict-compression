"""WikiContradict loader and cleaning for Test-OOD-2."""
from __future__ import annotations

import csv
import os
import json
from pathlib import Path

WIKI_CSV = Path(os.environ.get("CCD_WIKI_CSV", Path(__file__).resolve().parents[2] / "data" / "raw" / "wikicontradict_all.csv"))


def load_rows(path: Path = WIKI_CSV) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:  # fields contain newlines
        return list(csv.DictReader(f))


def dev50_keys(dev50_path: Path) -> set[tuple[str, str]]:
    """dev_50.jsonl has no question_ID; rows are identified by (question, article title)."""
    keys = set()
    with open(dev50_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            keys.add((r["question"].strip(), r["article_title"].strip()))
    return keys


def clean(rows: list[dict], exclude_keys: set[tuple[str, str]]) -> tuple[list[dict], dict]:
    """Drop same-passage rows, dev_50 rows, and rows without two distinct answers.

    Returns (kept_rows, step_counts)."""
    steps = {"total": len(rows)}
    kept = [r for r in rows
            if r["context1"].strip() != r["context2"].strip() and r["samepassage"].strip().lower() != "same"]
    steps["after_same_passage"] = len(kept)
    kept = [r for r in kept
            if (r["question"].strip(), r["WikipediaArticleTitle"].strip()) not in exclude_keys]
    steps["after_dev50"] = len(kept)
    kept = [r for r in kept
            if r["question"].strip() and r["answer1"].strip() and r["answer2"].strip()
            and r["answer1"].strip().lower() != r["answer2"].strip().lower()]
    steps["after_answer_check"] = len(kept)
    return kept, steps
