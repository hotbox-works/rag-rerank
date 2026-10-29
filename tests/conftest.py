"""Shared fakes. Nothing here needs a GPU, Java, or model downloads."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import retrieval  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a GPU, Java and model downloads (set RUN_GPU_TESTS=1)")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_GPU_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="set RUN_GPU_TESTS=1 on a GPU node to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


class FakeLLM:
    """Answers "Answer: <first context title or 'unknown'>" and summarizes as "S(<title>)". Records every call."""

    def __init__(self):
        self.calls = []

    def chat(self, conversations, max_new_tokens=32):
        self.calls.append(conversations)
        out = []
        for conv in conversations:
            user = conv[-1]["content"]
            if user.startswith("Passage ("):
                title = user[len("Passage ("):user.index("):")]
                out.append(f"S({title})")
            elif "Document 1 (" in user:
                title = user.split("Document 1 (", 1)[1].split("):", 1)[0]
                out.append(f"Answer: {title}")
            else:
                out.append("Answer: unknown")
        return out

    def n_prompts(self, kind):
        convs = [c for batch in self.calls for c in batch]
        if kind == "summary":
            return sum(c[-1]["content"].startswith("Passage (") for c in convs)
        return sum(not c[-1]["content"].startswith("Passage (") for c in convs)


class FakeScorer:
    """Returns fixed scores by docid (default 0)."""

    def __init__(self, scores):
        self.scores = scores
        self.n_scored = 0

    def score_batch(self, query, passages):
        self.n_scored += len(passages)
        return [self.scores.get(p["docid"], 0.0) for p in passages]


def make_hits(n=15):
    return [{"rank": i + 1, "docid": f"d{i + 1}", "title": f"T{i + 1}", "text": f"text {i + 1}", "bm25": 20.0 - i}
            for i in range(n)]


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, query, offset=0, k=5, qid=None):
        return [dict(h) for h in self.hits[offset:offset + k]]


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture(scope="session")
def dev_records():
    if not os.path.exists(retrieval.SCORES_PATH):
        pytest.skip("outputs/t5_scores.jsonl not present")
    return retrieval.read_jsonl(retrieval.SCORES_PATH)


@pytest.fixture(scope="session")
def dev_questions():
    return retrieval.read_jsonl(retrieval.DATA_PATH)
