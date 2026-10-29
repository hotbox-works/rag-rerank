"""Real models on a GPU node: RUN_GPU_TESTS=1 python -m pytest tests/test_gpu_smoke.py -s"""
import pytest

import pipeline
import retrieval

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def live_pipe():
    return pipeline.build_live(summary_cache_path=None)


def test_t5_scores_are_probabilities(live_pipe):
    scorer = live_pipe.selector.scorer
    s = scorer.score_batch("Who wrote Hamlet?", [{"title": "Hamlet", "text": "Hamlet is a tragedy by William "
                                                  "Shakespeare."}, {"title": "Cat", "text": "Cats are mammals."}])
    assert all(0 <= x <= 1 for x in s) and s[0] > s[1]



def test_answer_three_dev_questions(live_pipe):
    for q in retrieval.read_jsonl(retrieval.DATA_PATH, 3):
        res = live_pipe.answer(q["question"], qid=q["_id"])
        print(q["question"], "| gold:", q["answer"], "| pred:", res["answer"], "|", res["timing"])
        assert res["answer"] and res["selected"]
        assert len(res["answer"].split()) < 20


def test_module_level_answer():
    assert isinstance(pipeline.answer("What is the capital of France?", summarize=False), str)
