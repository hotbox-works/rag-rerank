import pytest

import generator
import pipeline
import retrieval
from conftest import FakeRetriever, FakeScorer, make_hits


def live(fake_llm, scores, **settings):
    sel = pipeline.LiveSelector(FakeRetriever(make_hits()), FakeScorer(scores))
    return pipeline.Pipeline(sel, fake_llm, **settings)


def test_adaptive_raw_selects_best_scored(fake_llm):
    res = live(fake_llm, {"d4": 0.99, "d2": 0.995}, summarize=False, keep=2).answer("q")
    assert [h["docid"] for h in res["selected"]] == ["d2", "d4"]
    assert res["answer"] == "T2" and res["n_scored"] == 5 and res["stopped_early"]
    assert res["contexts"][0]["text"] == "text 2"
    assert fake_llm.n_prompts("summary") == 0


def test_summaries_are_injected_and_cached(fake_llm):
    pipe = live(fake_llm, {"d1": 0.99}, summarize=True, keep=1)
    res = pipe.answer("q")
    assert res["contexts"] == [{"title": "T1", "text": "S(T1)"}] and res["new_summaries"] == 1
    pipe.answer("q")
    assert fake_llm.n_prompts("summary") == 1  # second call reused the cache


def test_drop_irrelevant(fake_llm):
    pipe = live(fake_llm, {"d1": 0.99, "d2": 0.99}, summarize=True, keep=2, drop_irrelevant=True)
    pipe.cache.add_many([("q", {"docid": "d1"}, generator.NOT_RELEVANT), ("q", {"docid": "d2"}, "useful")])
    assert [c["title"] for c in pipe.answer("q")["contexts"]] == ["T2"]


@pytest.mark.parametrize("fallback, expected", [("top", ["d9", "d3"]), ("bm25", ["d1", "d2"]), ("closed_book", [])])
def test_fallbacks(fake_llm, fallback, expected):
    res = live(fake_llm, {"d3": 0.4, "d9": 0.5}, summarize=False, keep=2, fallback=fallback).answer("q")
    assert res["fallback"] and [h["docid"] for h in res["selected"]] == expected
    assert res["closed_book"] == (expected == [])
    assert res["answer"] == ("unknown" if not expected else "T" + expected[0][1:])


def test_bm25_and_closed_book_modes(fake_llm):
    res = live(fake_llm, {}, mode="bm25", keep=3, summarize=False).answer("q")
    assert [h["rank"] for h in res["selected"]] == [1, 2, 3] and res["n_scored"] == 0
    res = live(fake_llm, {}, mode="closed_book").answer("q")
    assert res["closed_book"] and res["answer"] == "unknown"


def test_bad_settings_rejected(fake_llm):
    with pytest.raises(ValueError):
        live(fake_llm, {}, fallback="nope")
    with pytest.raises(ValueError):
        live(fake_llm, {}, tresh=0.5)


def test_replay_matches_live(fake_llm, dev_records):
    """Replaying saved scores selects exactly what a live run with the same scores would."""
    for rec in dev_records[:40]:
        hits = retrieval.bm25_order(rec)
        live_sel = pipeline.LiveSelector(FakeRetriever([{k: v for k, v in h.items() if k != "t5"} for h in hits]),
                                         FakeScorer({h["docid"]: h["t5"] for h in hits}))
        replay = pipeline.ReplaySelector([rec])
        for settings in ({"threshold": 0.98, "keep": 2}, {"threshold": 0.5, "keep": 3, "cap": 10},
                         {"threshold": 1.01, "keep": 2, "fallback": "bm25"}):
            s = pipeline.check_settings(settings)
            a = live_sel.select(rec["question"], rec["id"], s)
            b = replay.select(rec["question"], rec["id"], s)
            assert [h["docid"] for h in a["selected"]] == [h["docid"] for h in b["selected"]]
            assert a["n_scored"] == b["n_scored"] and a["fallback"] == b["fallback"]


def test_summary_cache_persists(tmp_path):
    path = str(tmp_path / "s.jsonl")
    c = pipeline.SummaryCache(path, "m")
    c.add_many([("q", {"docid": "d1"}, "sum")])
    assert pipeline.SummaryCache(path, "m").get("q", {"docid": "d1"}) == "sum"
    assert pipeline.SummaryCache(path, "other-model").get("q", {"docid": "d1"}) is None
