"""ConflictBank QA loader (HF `Warrieryes/CB_qa`, file `QA_dataset.json`).

The file is JSON Lines (one record per line, ~5.8 GB), so it is streamed line by
line; json.loads per line keeps memory flat (ijson is unnecessary for JSONL).
Records are addressed by byte offset so selected records can be re-read with seek.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# Evidence field per conflict type. Note the upstream key typo for misinformation.
EVIDENCE_KEY = {
    "default": "default_evidence",
    "misinformation": "misinformation_conflict_evidence_evidence",
    "temporal": "temporal_conflict_evidence",
}
CLAIM_KEY = {
    "default": "default_claim",
    "misinformation": "misinformation_conflict_claim",
    "temporal": "temporal_conflict_claim",
}


def build_index(path: Path) -> list[tuple[int, int, str]]:
    """One streaming pass: [(line_no, byte_offset, subject)] for every record."""
    index = []
    with open(path, "rb") as f:
        offset = 0
        for line_no, line in enumerate(f):
            if line.strip():
                index.append((line_no, offset, json.loads(line)["subject"]))
            offset += len(line)
    return index


def load_index(path: Path, cache: Path | None = None) -> list[tuple[int, int, str]]:
    """build_index with an optional JSON cache (the full pass takes minutes)."""
    if cache and cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text())]
    index = build_index(path)
    if cache:
        cache.write_text(json.dumps(index))
    return index


def read_record(f, offset: int) -> dict:
    """Read the record at byte offset from a file opened in 'rb'."""
    f.seek(offset)
    return json.loads(f.readline())


def mentions(text: str, s: str) -> bool:
    """Case-insensitive whole-token match of s in text (avoids 'male' in 'female')."""
    s = (s or "").strip()
    if not s:
        return False
    return re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", text or "", re.IGNORECASE) is not None


def check_pair(rec: dict, ctype: str) -> str | None:
    """Return the rejection reason for using (default, ctype) evidence as a conflict pair, or None.

    Reasons (applied in this order):
      missing_field  - question/object/replaced_object/evidence empty
      same_answer    - object equals replaced_object
      span_missing   - object not in default evidence, or replaced_object not in conflict evidence
      mixed          - default evidence names replaced_object, or conflict evidence names object
                       (one document carrying both sides; reported in the ConflictBank paper)
    """
    a, b = rec.get("object", ""), rec.get("replaced_object", "")
    ev_a, ev_b = rec.get(EVIDENCE_KEY["default"], ""), rec.get(EVIDENCE_KEY[ctype], "")
    if not (rec.get("question") and a and b and ev_a and ev_b):
        return "missing_field"
    if a.strip().lower() == b.strip().lower():
        return "same_answer"
    if not (mentions(ev_a, a) and mentions(ev_b, b)):
        return "span_missing"
    if mentions(ev_a, b) or mentions(ev_b, a):
        return "mixed"
    return None
