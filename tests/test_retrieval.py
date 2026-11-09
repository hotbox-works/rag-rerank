import csv
import os

import retrieval
from conftest import make_hits


def _get_batch(scores):
    hits = make_hits()
    return lambda offset, k: [{**h, "t5": scores.get(h["docid"], 0.0)} for h in hits[offset:offset + k]]


def test_run_loop_stops_at_first_batch_over_threshold():
    res = retrieval.run_loop(_get_batch({"d7": 0.99, "d2": 0.5}), threshold=0.98)
    assert res["stopped_early"] and res["n_scored"] == 10 and res["batches"] == 2
    assert [h["docid"] for h in res["selected"]] == ["d7"]  # only one hit passes the threshold


def test_run_loop_fallback_top_and_none():
    res = retrieval.run_loop(_get_batch({"d3": 0.6, "d12": 0.7}), threshold=0.98)
    assert res["fallback"] and res["n_scored"] == 15
    assert [h["docid"] for h in res["selected"]] == ["d12", "d3"]
    none = retrieval.run_loop(_get_batch({"d3": 0.6}), threshold=0.98, fallback="none")
    assert none["selected"] == []


def test_sweep_reproduces_saved_csv(dev_records):
    """Regression check: replaying the saved scores gives exactly the committed sweep.csv."""
    path = retrieval.SWEEP_PATH
    if not os.path.exists(path):
        return
    with open(path) as f:
        saved = list(csv.DictReader(f))
    rows = retrieval.sweep(dev_records)
    assert len(rows) == len(saved)
    for r, s in zip(rows, saved):
        for col in retrieval.SWEEP_COLUMNS:
            assert retrieval._csv_cell(col, r[col]) == s[col], (col, r, s)


def test_score_questions_tolerates_missing_answer(tmp_path):
    """Test questions have no answer field."""
    from conftest import FakeRetriever, FakeScorer

    out = tmp_path / "s.jsonl"
    retrieval.score_questions([{"_id": "x", "question": "q"}], FakeRetriever(make_hits()), FakeScorer({}), str(out))
    rec = retrieval.read_jsonl(str(out))[0]
    assert rec["gold"] == "" and len(rec["hits"]) == 15
