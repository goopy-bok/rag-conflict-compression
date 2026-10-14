"""Data build + sealing tests. No network: fake ConflictBank JSONL and fake WikiContradict CSV."""
import csv
import json
import random

import pytest

import data_build as db
import seal
from sources import conflictbank as cb
from sources import wikicontradict as wc

FILLER = " ".join(["the archive records describe several events across the years"] * 4)
SIZES = {"dev": (5, "misinformation"), "dev_distractor": (3, "none"), "test_id_pool": (8, "misinformation"),
         "test_ood1": (4, "temporal"), "distractor": (4, "none")}


def _rec(i, subj, obj, rep, mixed=False, span_missing=False):
    a = f"{subj} worked for {obj}. {FILLER}" + (f" Some say {rep}." if mixed else "")
    return {
        "relation": "P108", "subject": subj, "object": obj, "replaced_object": rep,
        "question": f"Which organization did {subj} work for?",
        "default_claim": f"{subj} worked for {obj}.", "default_evidence": a if not span_missing else FILLER,
        "misinformation_conflict_claim": f"{subj} worked for {rep}.",
        "misinformation_conflict_evidence_evidence": f"{subj} was employed at {rep}. {FILLER}",
        "temporal_conflict_time_span": ["2030", "2040"],
        "temporal_conflict_claim": f"{subj} worked for {rep} from 2030 to 2040.",
        "temporal_conflict_evidence": f"From 2030 to 2040 {subj} served {rep}. {FILLER}",
        "semantic_conflict_evidence": "ignored", "options": ["x"], "correct_option": "A",
    }


@pytest.fixture
def cb_file(tmp_path):
    recs = []
    for i in range(90):
        subj = f"Person{i} Surname{i}"
        recs.append(_rec(i, subj, f"Org{i % 30}", f"Org{(i + 7) % 30}"))
        if i % 9 == 0:  # second record for some subjects, one of them invalid
            recs.append(_rec(i, subj, f"Org{(i + 1) % 30}", f"Org{(i + 2) % 30}", mixed=True))
    recs.append(_rec(999, "Mixed Only", "Org1", "Org2", mixed=True))
    recs.append(_rec(998, "Span Missing", "Org3", "Org4", span_missing=True))
    p = tmp_path / "QA_dataset.json"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return p


def _build(cb_file, seed=42):
    index = cb.build_index(cb_file)
    return db.build_conflictbank(cb_file, index, sizes=SIZES, window=(1, 10000), pool_size=10000, seed=seed)


def test_check_pair_rules():
    assert cb.check_pair(_rec(0, "A B", "Org1", "Org2"), "misinformation") is None
    assert cb.check_pair(_rec(0, "A B", "Org1", "Org2", mixed=True), "misinformation") == "mixed"
    assert cb.check_pair(_rec(0, "A B", "Org1", "Org2", span_missing=True), "temporal") == "span_missing"
    assert cb.check_pair(_rec(0, "A B", "Org1", "Org1"), "misinformation") == "same_answer"
    assert not cb.mentions("female", "male") and cb.mentions("a Male x", "male")


def test_schema_split_distractors(cb_file):
    sets, rep = _build(cb_file)
    assert {k: len(v) for k, v in sets.items()} == {k: n for k, (n, _) in SIZES.items()}
    subjects = {k: {x["entity"] for x in v} for k, v in sets.items()}
    names = list(subjects)
    for i, a in enumerate(names):  # entity-level split: no subject in two sets
        for b in names[i + 1:]:
            assert not subjects[a] & subjects[b], (a, b)
    all_subj = set().union(*subjects.values())
    assert "Mixed Only" not in all_subj and "Span Missing" not in all_subj

    for split, insts in sets.items():
        ctype = SIZES[split][1]
        for x in insts:
            assert isinstance(x["id"], str) and x["source"] == "conflictbank" and x["conflict_type"] == ctype
            assert isinstance(x["question"], str) and isinstance(x["total_tokens"], int)
            roles = [d["role"] for d in x["docs"]]
            assert [d["doc_id"] for d in x["docs"]] == [f"d{i}" for i in range(len(x["docs"]))]
            # conflict_positions points exactly at the non-distractor docs
            assert x["conflict_positions"] == [i for i, r in enumerate(roles) if r != "distractor"]
            if ctype == "none":
                assert sorted(r for r in roles if r != "distractor") == ["default"]
                assert set(x["gold"]) == {"answer_a", "span_a"}
            else:
                assert sorted(r for r in roles if r != "distractor") == ["conflict_a", "conflict_b"]
                g = x["gold"]
                assert g["answer_a"] != g["answer_b"] and g["span_a"] == g["answer_a"]
                assert cb.mentions(x["docs"][x["conflict_positions"][0]]["text"] + x["docs"][x["conflict_positions"][1]]["text"], g["span_b"])
            if ctype == "temporal":
                assert x["gold"]["time_span"] == ["2030", "2040"]
            n_dis = roles.count("distractor")
            assert n_dis <= 6 and len(x["distractor_keys"]) == n_dis
            # total_tokens = cl100k of the rendered '[dN] text' block
            assert x["total_tokens"] == db.count_tokens("\n\n".join(f"[{d['doc_id']}] {d['text']}" for d in x["docs"]))
            # distractors never mention the instance's entities
            ents = [x["entity"], x["gold"]["answer_a"], x["gold"].get("answer_b")]
            ents = [e for e in ents if e]
            for d in x["docs"]:
                if d["role"] == "distractor":
                    assert not any(cb.mentions(d["text"], e) for e in ents)
                    assert d["text"].split(" worked for ")[0] not in all_subj  # pool subjects disjoint
    assert sum(len(v) for v in sets.values()) > 0 and rep["pool"]["sampled"] > 0
    # most instances get the requested 3-6 distractors from the fake pool
    assert sum(3 <= sum(d["role"] == "distractor" for d in x["docs"]) for v in sets.values() for x in v) >= 20


def test_dev_test_pools_disjoint(cb_file):
    sets, _ = _build(cb_file)
    dev_keys = {k for s in db.DEV_SIDE for x in sets[s] for k in x["distractor_keys"]}
    test_keys = {k for s in sets if s not in db.DEV_SIDE for x in sets[s] for k in x["distractor_keys"]}
    assert not dev_keys & test_keys


def test_deterministic(cb_file):
    assert _build(cb_file, seed=7)[0] == _build(cb_file, seed=7)[0]
    assert _build(cb_file, seed=7)[0] != _build(cb_file, seed=8)[0]


def _wiki_csv(tmp_path):
    cols = ["question_ID", "question", "context1", "context2", "answer1", "answer2", "contradictType",
            "samepassage", "merged_context", "ref_answer", "WikipediaArticleTitle", "url"]
    rows = [
        ["1", "When was X born?", "X was born in 1900.\nLine two.", "X was born in 1901.", "1900", "1901", "Explicit", "Different", "", "", "X", ""],
        ["2", "Where is Y?", "Y is in Paris and also Lyon.", "Y is in Paris and also Lyon.", "Paris", "Lyon", "Explicit", "Different", "", "", "Y", ""],
        ["3", "Who owns Z?", "Z is owned by A; others say B.", "Z owned by B.", "A", "B", "Implicit (reasoning required)", "Same", "", "", "Z", ""],
        ["4", "How tall is W?", "W is 10 m.", "W is 12 m.", "10 m", "12 m", "Explicit", "Different", "", "", "W", ""],
        ["5", "What color is V?", "V is red.", "V is blue.", "red", "blue", "Explicit", "Different", "", "", "V", ""],
        ["6", "Unrelated?", "Q is a town in Peru.", "Q has 300 people.", "a", "b", "Explicit", "Different", "", "", "Q", ""],
    ]
    p = tmp_path / "wc.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    return p


def test_wikicontradict_cleaning(tmp_path):
    rows = wc.load_rows(_wiki_csv(tmp_path))
    assert rows[0]["context1"].endswith("Line two.")  # embedded newline survives
    insts, rep = db.build_wikicontradict(rows, exclude_keys={("How tall is W?", "W")}, seed=1)
    assert rep["steps"] == {"total": 6, "after_same_passage": 4, "after_dev50": 3, "after_answer_check": 3}
    assert {x["question_ID"] for x in insts} == {"1", "5", "6"}
    for x in insts:
        assert x["conflict_type"] == "wiki_natural" and x["source"] == "wikicontradict"
        for d in x["docs"]:
            if d["role"] == "distractor":
                assert x["entity"] + " " not in d["text"]


def _seal_root(tmp_path):
    (tmp_path / "data" / "sealed").mkdir(parents=True)
    (tmp_path / "prompts").mkdir()
    f = tmp_path / "data" / "sealed" / "test_ood1.jsonl"
    f.write_text('{"id": 1}\n{"id": 2}\n')
    p = tmp_path / "prompts" / "c3.txt"
    p.write_text("final prompt")
    return f, p


def test_seal_verify(tmp_path):
    f, _ = _seal_root(tmp_path)
    entries = seal.seal_files([f], root=tmp_path)
    assert entries["data/sealed/test_ood1.jsonl"]["lines"] == 2
    seal.verify_sealed(root=tmp_path)
    seal.seal_files([f], root=tmp_path)  # second seal keeps history
    assert (tmp_path / "data" / "SEALED.md").read_text().count("## Seal ") == 2
    f.write_text('{"id": 1}\n')
    with pytest.raises(seal.SealError):
        seal.verify_sealed(root=tmp_path)


def test_check_test_access(tmp_path):
    f, p = _seal_root(tmp_path)
    seal.seal_files([f], root=tmp_path)
    seal.check_test_access("dev", root=tmp_path)  # dev: no gate
    with pytest.raises(PermissionError, match="FROZEN_PROMPTS"):
        seal.check_test_access("test_ood1", root=tmp_path)
    with pytest.warns(UserWarning):
        out = seal.freeze_prompts([p], root=tmp_path)
    assert json.loads(out.read_text())["git_commit"] is None
    seal.check_test_access("test_ood1", root=tmp_path)
    p.write_text("edited prompt")
    with pytest.raises(PermissionError, match="changed"):
        seal.check_test_access("distractor", root=tmp_path)
    p.write_text("final prompt")
    f.write_text("tampered\n")
    with pytest.raises(PermissionError, match="seal"):
        seal.check_test_access("test_id", root=tmp_path)
    log = [json.loads(l) for l in (tmp_path / "results" / "test_access_log.jsonl").read_text().splitlines()]
    assert [e["allowed"] for e in log] == [False, True, False, False]
    assert all(e["split"] and e["time"] for e in log)
