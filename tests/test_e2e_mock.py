"""Full harness pass with MockLLM on 3 tiny instances (2 conflict + 1 distractor). No network."""
import json

from budget import count_tokens
from run import run_condition


def _inst(i, ctype, a, b=None):
    docs = [{"doc_id": f"d{k}", "text": f"Document {k} about item {i}. " * 20, "role": "distractor"}
            for k in range(3)]
    docs[0].update(text=f"Item {i} was founded in {a}. " * 10, role="conflict_a" if b else "default")
    gold = {"answer_a": a, "span_a": a}
    if b:
        docs[1].update(text=f"Item {i} was founded in {b}. " * 10, role="conflict_b")
        gold.update(answer_b=b, span_b=b)
    return {"id": f"t{i}", "source": "conflictbank", "conflict_type": ctype, "question": f"When was item {i} founded?",
            "docs": docs, "gold": gold, "conflict_positions": [0, 1] if b else [],
            "entity": f"item {i}", "total_tokens": sum(count_tokens(d["text"]) for d in docs)}


def test_mock_pipeline_end_to_end(tmp_path, make_client):
    data = tmp_path / "mini.jsonl"
    rows = [_inst(1, "misinformation", "1990", "1995"), _inst(2, "temporal", "2001", "2011"),
            _inst(3, "none", "1970")]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))

    client = make_client("mock")
    recs = run_condition("C0", "dev", 0.3, client, seed=0, results_dir=tmp_path / "out", data_path=data)

    out = tmp_path / "out" / "C0_dev_r0.3_s0.jsonl"
    assert out.exists() and not (tmp_path / "out" / "C0_dev_r0.3_s0.jsonl.partial").exists()
    saved = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["id"] for r in saved] == ["t1", "t2", "t3"] == [r["id"] for r in recs]

    for r in saved:
        for k in ("condition", "split", "r", "seed", "budget", "out_tokens", "over_budget", "retried",
                  "compressed", "judge", "judge_model", "downstream", "conflict_positions",
                  "prompt_sha256", "meta"):
            assert k in r, k
        assert r["condition"] == "C0" and r["seed"] == 0 and r["r"] == 0.3
        assert r["out_tokens"] <= r["budget"] and not r["over_budget"]
        assert r["meta"]["mode"] == "mock"
        assert len(r["prompt_sha256"]["instruction"]) == 64 and r["prompt_sha256"]["examples"] is None
        assert r["downstream"]["D"] in (0, 1)
        assert r["downstream"]["category"] in {"DEFINITIVE", "CONFLICT_ACK", "INSUFFICIENT", "OTHER"}

    conflict, distractor = saved[:2], saved[2]
    for r in conflict:
        assert r["csr"] in (0, 1) and r["csr_strict"] in (0, 1) and r["false_conflict"] is None
    assert distractor["csr"] is None and isinstance(distractor["false_conflict"], bool)

    s = client.summary()
    # per instance: 1 compress + 1 judge + 1 downstream + 1 D judge (mock never exceeds budget)
    assert s["calls_total"] == 12 and s["usd_spent"] == 0
