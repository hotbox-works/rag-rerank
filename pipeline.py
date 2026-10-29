"""The full system: BM25 -> monoT5 adaptive selection -> Llama summaries -> Llama answer.

    from pipeline import answer
    answer("Which band formed first, Oasis or Blur?")          # loads everything on first call

    python pipeline.py "Which band formed first, Oasis or Blur?" --keep 2 --threshold 0.98

Selection runs in one of two ways:
  * LiveSelector retrieves and scores on demand (any question, real latency).
  * ReplaySelector reads monoT5 scores already saved by `retrieval.py score`, so experiments over
    thresholds, caps and keep values only pay for Llama. Both return the same fields.

Settings (all keyword arguments to Pipeline):
  mode       "adaptive" (BM25 + monoT5 loop), "bm25" (plain BM25 top-keep), or "closed_book" (no documents)
  threshold, cap, keep   adaptive loop settings, as in retrieval.run_loop
  fallback   when nothing reaches the threshold: "top" (best-scored anyway), "bm25" (BM25 top-keep),
             or "closed_book" (answer with no documents)
  summarize  condense each selected passage with Llama before injection (True) or inject raw text (False)
  drop_irrelevant   with summarize, leave out passages Llama marks "Not relevant."
"""
import argparse
import hashlib
import json
import os
import time

import generator
import retrieval
from utils import config

SUMMARY_CACHE_PATH = os.path.join(config.DIR_OUTPUTS, "summaries.jsonl")

DEFAULT_SETTINGS = {"mode": "adaptive", "threshold": retrieval.DEFAULT_THRESHOLD, "cap": retrieval.MAX_DOCS,
                    "keep": retrieval.DEFAULT_KEEP, "fallback": "top", "summarize": True, "drop_irrelevant": False}
MODES = ("adaptive", "bm25", "closed_book")
FALLBACKS = ("top", "bm25", "closed_book")


def check_settings(settings):
    unknown = set(settings) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    s = {**DEFAULT_SETTINGS, **settings}
    if s["mode"] not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if s["fallback"] not in FALLBACKS:
        raise ValueError(f"fallback must be one of {FALLBACKS}")
    return s


def apply_fallback(res, bm25_hits, keep, fallback):
    """Replace `res["selected"]` when the loop fell back. run_loop already chose the best-scored hits ("top")."""
    if res["fallback"] and fallback == "bm25":
        res["selected"] = [dict(h) for h in bm25_hits[:keep]]
    elif res["fallback"] and fallback == "closed_book":
        res["selected"] = []
    return res


def _bm25_result(hits, keep):
    return {"selected": [dict(h) for h in hits[:keep]], "n_scored": 0, "batches": 0, "stopped_early": False,
            "fallback": False, "best_score": None}


# Selection

class LiveSelector:
    """Retrieves and scores per question. `scorer` can be None for mode="bm25"."""

    def __init__(self, retriever, scorer=None):
        self.retriever = retriever
        self.scorer = scorer

    def select(self, question, qid, s):
        if s["mode"] == "closed_book":
            return {**_bm25_result([], 0), "retrieve_s": 0.0, "score_s": 0.0}
        t0 = time.perf_counter()
        if s["mode"] == "bm25":
            res = _bm25_result(self.retriever.retrieve(question, 0, s["keep"], qid=qid), s["keep"])
            res.update(retrieve_s=time.perf_counter() - t0, score_s=0.0)
            return res
        if self.scorer is None:
            raise ValueError("mode='adaptive' needs a scorer")
        loop_fallback = "none" if s["fallback"] == "closed_book" else "top"
        res = retrieval.adaptive_retrieve(question, self.retriever, self.scorer, qid=qid, threshold=s["threshold"],
                                          max_docs=s["cap"], keep=s["keep"], fallback=loop_fallback)
        if res["fallback"] and s["fallback"] == "bm25":
            bm25_hits = self.retriever.retrieve(question, 0, s["keep"], qid=qid)
            apply_fallback(res, bm25_hits, s["keep"], "bm25")
        return res


class ReplaySelector:
    """Selection from records written by `retrieval.score_questions`, keyed by question id. No model runs."""

    def __init__(self, records):
        self.records = {r["id"]: r for r in records}

    @classmethod
    def from_path(cls, path=retrieval.SCORES_PATH):
        return cls(retrieval.read_jsonl(path))

    def select(self, question, qid, s):
        if s["mode"] == "closed_book":
            return {**_bm25_result([], 0), "retrieve_s": 0.0, "score_s": 0.0}
        if qid not in self.records:
            raise KeyError(f"No saved scores for question {qid}. Run `python retrieval.py score` on this split.")
        record = self.records[qid]
        hits = retrieval.bm25_order(record)
        if s["mode"] == "bm25":
            return {**_bm25_result(hits, s["keep"]), "retrieve_s": 0.0, "score_s": 0.0}
        res = retrieval.select_from_record(record, s["threshold"], retrieval.BATCH_SIZE, s["cap"], s["keep"], "top")
        apply_fallback(res, hits, s["keep"], s["fallback"])
        res.update(retrieve_s=0.0, score_s=res.pop("est_score_s", 0.0))
        return res


# Summary cache

class SummaryCache:
    """Summaries keyed by (model, prompt, question, passage), appended to a JSONL file so every run reuses them."""

    def __init__(self, path=SUMMARY_CACHE_PATH, model_name=generator.LLAMA_MODEL):
        self.path = path
        self.salt = model_name + "\n" + generator.SUMMARY_PROMPT
        self.data = {}
        if path and os.path.exists(path):
            for row in retrieval.read_jsonl(path):
                self.data[row["key"]] = row["summary"]

    def key(self, question, passage):
        pid = passage.get("docid") or f"{passage.get('title', '')}\n{passage.get('text', '')}"
        return hashlib.sha1(f"{self.salt}\n{question}\n{pid}".encode()).hexdigest()

    def get(self, question, passage):
        return self.data.get(self.key(question, passage))

    def add_many(self, items):
        """`items` is a list of (question, passage, summary)."""
        rows = []
        for q, p, summary in items:
            k = self.key(q, p)
            self.data[k] = summary
            rows.append({"key": k, "docid": p.get("docid"), "summary": summary})
        if self.path and rows:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)


# Pipeline

class Pipeline:
    def __init__(self, selector, llm, summary_cache=None, **settings):
        self.selector = selector
        self.llm = llm
        self.settings = check_settings(settings)
        self.cache = summary_cache if summary_cache is not None else SummaryCache(path=None)

    def with_settings(self, **settings):
        """A pipeline sharing the same models and cache with different settings."""
        return Pipeline(self.selector, self.llm, self.cache, **{**self.settings, **settings})

    def _summaries(self, questions, selections):
        todo, seen = [], set()
        for q, res in zip(questions, selections):
            for p in res["selected"]:
                k = self.cache.key(q["question"], p)
                if self.cache.get(q["question"], p) is None and k not in seen:
                    seen.add(k)
                    todo.append((q["question"], p))
        t0 = time.perf_counter()
        if todo:
            summaries = generator.summarize_batch(self.llm, todo)
            self.cache.add_many([(q, p, s) for (q, p), s in zip(todo, summaries)])
        return time.perf_counter() - t0, len(todo)

    def _contexts(self, question, selected):
        s = self.settings
        contexts = []
        for p in selected:
            text = p.get("text", "")
            if s["summarize"]:
                text = self.cache.get(question, p)
                if s["drop_irrelevant"] and text == generator.NOT_RELEVANT:
                    continue
            contexts.append({"title": p.get("title", ""), "text": text})
        return contexts

    def answer_batch(self, questions):
        """Answer a list of {"_id", "question"} dicts. Llama runs batched; per-question LLM times are averages."""
        s = self.settings
        selections = [self.selector.select(q["question"], q.get("_id"), s) for q in questions]
        summarize_s, n_new = self._summaries(questions, selections) if s["summarize"] else (0.0, 0)
        items = [(q["question"], self._contexts(q["question"], res["selected"])) for q, res in zip(questions, selections)]
        t0 = time.perf_counter()
        answers, raw = generator.generate_batch(self.llm, items)
        generate_s = time.perf_counter() - t0
        n = max(len(questions), 1)
        out = []
        for q, res, (_, contexts), a, r in zip(questions, selections, items, answers, raw):
            out.append({
                "id": q.get("_id"), "question": q["question"], "answer": a, "raw": r,
                "selected": [{k: h.get(k) for k in ("docid", "title", "rank", "t5", "text")} for h in res["selected"]],
                "contexts": contexts, "closed_book": not contexts, "n_scored": res["n_scored"],
                "fallback": res["fallback"], "stopped_early": res["stopped_early"], "best_score": res["best_score"],
                "timing": {"retrieve_s": res.get("retrieve_s", 0.0), "score_s": res.get("score_s", 0.0),
                           "summarize_s": summarize_s / n, "generate_s": generate_s / n},
                "new_summaries": n_new,
            })
        return out

    def answer(self, question, qid=None):
        t0 = time.perf_counter()
        res = self.answer_batch([{"_id": qid, "question": question}])[0]
        res["timing"]["total_s"] = time.perf_counter() - t0
        return res


def build_live(index=retrieval.INDEX, cache_path=retrieval.CACHE_PATH, t5_model=retrieval.T5_MODEL,
               llama_model=generator.LLAMA_MODEL, backend=generator.LLAMA_BACKEND,
               summary_cache_path=SUMMARY_CACHE_PATH, **settings):
    """Load BM25, monoT5 and Llama for live use. `cache_path` only speeds up questions already cached."""
    retriever = retrieval.Retriever(index, cache_path=cache_path)
    scorer = retrieval.T5Scorer(t5_model)
    llm = generator.Llama(llama_model, backend)
    return Pipeline(LiveSelector(retriever, scorer), llm, SummaryCache(summary_cache_path, llama_model), **settings)


_DEFAULT = None


def answer(query, **settings):
    """Answer one question with the default system. Returns the answer string."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = build_live()
    pipe = _DEFAULT.with_settings(**settings) if settings else _DEFAULT
    return pipe.answer(query)["answer"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--mode", choices=MODES, default=DEFAULT_SETTINGS["mode"])
    ap.add_argument("--threshold", type=float, default=DEFAULT_SETTINGS["threshold"])
    ap.add_argument("--cap", type=int, default=DEFAULT_SETTINGS["cap"])
    ap.add_argument("--keep", type=int, default=DEFAULT_SETTINGS["keep"])
    ap.add_argument("--fallback", choices=FALLBACKS, default=DEFAULT_SETTINGS["fallback"])
    ap.add_argument("--no-summarize", action="store_true")
    ap.add_argument("--t5-model", default=retrieval.T5_MODEL)
    ap.add_argument("--backend", default=generator.LLAMA_BACKEND, choices=["hf", "vllm"])
    args = ap.parse_args()
    pipe = build_live(t5_model=args.t5_model, backend=args.backend, mode=args.mode, threshold=args.threshold,
                      cap=args.cap, keep=args.keep, fallback=args.fallback, summarize=not args.no_summarize)
    res = pipe.answer(args.question)
    for c in res["contexts"]:
        print(f"[{c['title']}] {c['text']}")
    print(f"\nanswer: {res['answer']}\nraw: {res['raw']!r}\nfallback: {res['fallback']}  docs scored: {res['n_scored']}")
    print("timing:", {k: round(v, 3) for k, v in res["timing"].items()})


if __name__ == "__main__":
    main()
