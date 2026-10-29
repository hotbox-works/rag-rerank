import random

import retrieval
import train_scorer


def test_split_is_deterministic_and_disjoint(dev_questions):
    a_train, a_val = train_scorer.split_questions(dev_questions)
    b_train, b_val = train_scorer.split_questions(dev_questions)
    assert [q["_id"] for q in a_val] == [q["_id"] for q in b_val]
    assert not {q["_id"] for q in a_train} & {q["_id"] for q in a_val}
    assert len(a_val) == int(len(dev_questions) * train_scorer.VAL_FRACTION)


def _hits(texts):
    return [{"rank": i + 1, "docid": str(i), "title": "", "text": t} for i, t in enumerate(texts)]


def test_build_pools_labels_and_skips():
    qs = [{"_id": "a", "question": "q", "answer": "Paris"}, {"_id": "b", "question": "q", "answer": "yes"},
          {"_id": "c", "question": "q", "answer": "Rome"}]
    cache = {"a": _hits(["x", "in paris today", "y", "Paris again", "z"]), "b": _hits(["x"]), "c": _hits(["x", "y"])}
    pools, skipped = train_scorer.build_pools(qs, cache)
    assert skipped == {"yes_no": 1, "no_positive": 1}
    assert [h["docid"] for h in pools[0]["pos"]] == ["1", "3"] and len(pools[0]["neg"]) == 3


def test_sample_examples_ratio():
    pools = [{"question": "q", "pos": [{"docid": "p"}], "neg": [{"docid": str(i)} for i in range(10)]}]
    ex = train_scorer.sample_examples(pools, 3, random.Random(0))
    assert sum(lab for *_, lab in ex) == 1 and len(ex) == 4
    assert {"0"} <= {p["docid"] for _, p, lab in ex if not lab}  # the hardest negative is always included


def test_ranking_metrics_match_dev_recall(dev_records):
    """Labelling dev with gold_in and ranking by saved monoT5 scores reproduces 44.0% recall at 2."""
    rows = [{"hits": train_scorer.label_hits(r["gold"], r["hits"])} for r in dev_records
            if retrieval.gradable(r["gold"])]
    m = train_scorer.ranking_metrics(rows)
    assert abs(m["gold@2"] - 44.0) < 0.01 and abs(m["oracle@15"] - 57.24) < 0.01
