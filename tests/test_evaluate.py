import json
import os

import evaluate
import pipeline
import retrieval


def test_metrics_and_outcomes():
    rows = []
    for pred, gold, ctx_has_gold in (("Paris", "Paris", True), ("the city of Paris", "Paris", True),
                                      ("London", "Paris", True), ("London", "Paris", False), ("yes", "no", None)):
        r = {"answer": pred, "gold": gold, "type": "bridge", "gold_in_context": ctx_has_gold, "fallback": False,
             "closed_book": False, "n_scored": 5, "contexts": [{}], "timing": {}}
        r.update(evaluate.score_prediction(pred, gold))
        rows.append(r)
    assert [evaluate.outcome(r) for r in rows] == ["correct", "partial", "reader_error", "retrieval_miss",
                                                   "ungradable"]
    m = evaluate.compute_metrics(rows)
    assert m["n"] == 5 and abs(m["em"] - 20.0) < 1e-9 and m["pct_correct"] == 20.0
    assert abs(m["gold_in_context"] - 75.0) < 1e-9


def test_runner_end_to_end_with_fake_llm(tmp_path, fake_llm, dev_records, dev_questions):
    runner = evaluate.Runner(llm=fake_llm, summary_cache=pipeline.SummaryCache(path=None), runs_dir=str(tmp_path),
                             batch_size=4)
    qs = dev_questions[:10]
    spec = {"settings": {"threshold": 0.98, "keep": 2, "summarize": True}, "scores": retrieval.SCORES_PATH}
    m = runner.run("t", spec, qs, verbose=False)
    assert m["n"] == 10 and os.path.exists(tmp_path / "t" / "metrics.json")
    preds = retrieval.read_jsonl(str(tmp_path / "t" / "predictions.jsonl"))
    by_id = {r["id"]: r for r in dev_records}
    for p in preds:  # FakeLLM answers with the first context's title, which must be the top selected passage
        assert p["answer"] == p["selected"][0]["title"] and p["selected"][0]["docid"] in \
            {h["docid"] for h in by_id[p["id"]]["hits"]}
        assert all(c["text"].startswith("S(") for c in p["contexts"])
    n_calls = len(fake_llm.calls)
    runner.run("t", spec, qs, verbose=False)  # resume: nothing left to answer
    assert len(fake_llm.calls) == n_calls
    m2 = runner.run("t", spec, dev_questions[:12], verbose=False)
    assert m2["n"] == 12


def test_closed_book_needs_no_scores(tmp_path, fake_llm, dev_questions):
    runner = evaluate.Runner(llm=fake_llm, summary_cache=pipeline.SummaryCache(path=None), runs_dir=str(tmp_path))
    spec = {"settings": {"mode": "closed_book", "summarize": False}, "scores": "/does/not/exist"}
    m = runner.run("cb", spec, dev_questions[:3], verbose=False)
    assert m["closed_book_rate"] == 100.0


def test_registry_and_best(tmp_path):
    exps = evaluate.experiments()
    for names in evaluate.GROUPS.values():
        assert all(n in exps for n in names)
    for spec in exps.values():
        pipeline.check_settings(spec["settings"])
    rows = [{"name": "sweep_a", "n": 100, "f1": 40.0, "em": 30.0, "threshold": 0.9, "cap": 15, "keep": 2},
            {"name": "sweep_b", "n": 100, "f1": 42.0, "em": 31.0, "threshold": 0.98, "cap": 15, "keep": 3},
            {"name": "sweep_c_limit20", "n": 20, "f1": 60.0, "em": 50.0, "threshold": 0.5, "cap": 15, "keep": 1}]
    best, settings = evaluate.pick_best(rows)
    assert best["name"] == "sweep_b" and settings == {"threshold": 0.98, "cap": 15, "keep": 3}
    evaluate.write_table(rows, str(tmp_path / "r.csv"))
    assert open(tmp_path / "r.csv").readline().startswith("name,mode")


def test_overrides():
    assert evaluate._parse_overrides(["keep=3", "threshold=0.9", "summarize=false", "fallback=bm25"]) == \
        {"keep": 3, "threshold": 0.9, "summarize": False, "fallback": "bm25"}


def test_predict_test_writes_hotpot_format(tmp_path, fake_llm, dev_records, dev_questions):
    runner = evaluate.Runner(llm=fake_llm, summary_cache=pipeline.SummaryCache(path=None), runs_dir=str(tmp_path))
    qs = [{k: v for k, v in q.items() if k != "answer"} for q in dev_questions[:5]]  # test has no answers
    out = tmp_path / "test_predictions.json"
    evaluate.predict_test(runner, {"settings": {"keep": 2, "summarize": False}}, qs, str(out), retrieval.SCORES_PATH)
    pred = json.load(open(out))
    assert set(pred) == {"answer", "sp"} and len(pred["answer"]) == 5
