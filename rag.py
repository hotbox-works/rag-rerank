"""Retrieval-augmented QA on HotpotQA: BM25 -> monoT5 adaptive selection -> Llama, and its EM/F1 evaluation.

    python rag.py eval                                  # every setting in SETTINGS, on all dev questions
    python rag.py eval --settings adaptive bm25_top2 --limit 100
    python rag.py ask "Which magazine was started first?"   # live pipeline for one question (Pyserini + GPU)

`RAGPipeline` runs live: monoT5 scores BM25 hits in batches until one is confidently relevant, and the
selected passages go to Llama. Evaluation replays that same loop (`retrieval.run_loop`) on the scores saved by
`retrieval.py score`, so monoT5 doesn't run again and every setting sees identical scores.
"""
import argparse
import csv
import json
import os
import time

import retrieval
from hotpot_eval import exact_match_score, f1_score
from model import LLAMA_BATCH_SIZE, LLAMA_MODEL, LlamaReader
from utils import config

PRED_DIR = os.path.join(config.DIR_OUTPUTS, "llama")
EVAL_PATH = os.path.join(config.DIR_OUTPUTS, "llama_eval.csv")

SETTINGS = {
    "closed_book": {"source": "none"},
    "bm25_top2": {"source": "bm25", "k": 2},
    "bm25_top8": {"source": "bm25", "k": 8},
    "monot5_top2": {"source": "monot5", "k": 2},
    "monot5_top8": {"source": "monot5", "k": 8},
    "adaptive": {"source": "adaptive", "threshold": retrieval.DEFAULT_THRESHOLD, "max_docs": retrieval.MAX_DOCS,
                 "keep": retrieval.DEFAULT_KEEP, "fallback": retrieval.DEFAULT_FALLBACK},
}


# Live pipeline

class RAGPipeline:
    def __init__(self, retriever, scorer, reader, threshold=retrieval.DEFAULT_THRESHOLD, max_docs=retrieval.MAX_DOCS,
                 keep=retrieval.DEFAULT_KEEP, fallback=retrieval.DEFAULT_FALLBACK):
        self.retriever = retriever
        self.scorer = scorer
        self.reader = reader
        self.threshold = threshold
        self.max_docs = max_docs
        self.keep = keep
        self.fallback = fallback

    def retrieve(self, question, qid=None):
        return retrieval.adaptive_retrieve(question, self.retriever, self.scorer, qid=qid, threshold=self.threshold,
                                           max_docs=self.max_docs, keep=self.keep, fallback=self.fallback)

    def answer_many(self, questions):
        """Answer dev-style dicts with a "question" (and optionally "_id" to use the BM25 cache)."""
        results = [self.retrieve(q["question"], q.get("_id")) for q in questions]
        t0 = time.perf_counter()
        replies = self.reader.generate([q["question"] for q in questions], [r["selected"] for r in results])
        generate_s = (time.perf_counter() - t0) / max(len(questions), 1)
        return [{"question": q["question"], "answer": reply["answer"], "raw": reply["raw"], "passages": r["selected"],
                 "n_scored": r["n_scored"], "stopped_early": r["stopped_early"], "fallback": r["fallback"],
                 "retrieve_s": r["retrieve_s"], "score_s": r["score_s"], "generate_s": generate_s}
                for q, r, reply in zip(questions, results, replies)]

    def answer(self, question, qid=None):
        return self.answer_many([{"question": question, "_id": qid}])[0]


# Evaluation

def select_passages(record, setting):
    """Passages a setting gives Llama for one record from `retrieval.score_questions`, and how many monoT5 scored."""
    hits = retrieval.bm25_order(record)
    source = setting["source"]
    if source == "none":
        return [], 0
    if source == "bm25":
        return hits[:setting["k"]], 0
    if source == "monot5":
        return sorted(hits, key=lambda h: (-h["t5"], h["rank"]))[:setting["k"]], len(hits)
    if source == "adaptive":
        res = retrieval.select_from_record(record, setting["threshold"], retrieval.BATCH_SIZE, setting["max_docs"],
                                           setting["keep"], setting["fallback"])
        return res["selected"], res["n_scored"]
    raise ValueError(f"Unknown source {source!r}")


def run_setting(name, records, reader=None, pred_dir=PRED_DIR, overwrite=False, chunk_size=128):
    """Answer every record under one setting, appending predictions to <pred_dir>/<name>.jsonl.

    Records already answered are skipped unless `overwrite` is set, so `reader` is only needed for new work.
    Returns the predictions for `records`.
    """
    setting = SETTINGS[name]
    path = os.path.join(pred_dir, f"{name}.jsonl")
    done = set()
    if os.path.exists(path) and not overwrite:
        done = {p["id"] for p in retrieval.read_jsonl(path)}
    todo = [r for r in records if r["id"] not in done]
    print(f"{name}: {len(records)} questions, {len(todo)} to answer")
    if todo and reader is None:
        raise ValueError(f"{name} has {len(todo)} unanswered questions; pass a LlamaReader.")
    os.makedirs(pred_dir, exist_ok=True)
    with open(path, "a" if done else "w") as f:
        for start in range(0, len(todo), chunk_size):
            chunk = todo[start:start + chunk_size]
            selections = [select_passages(r, setting) for r in chunk]
            t0 = time.perf_counter()
            replies = reader.generate([r["question"] for r in chunk], [passages for passages, _ in selections])
            generate_s = (time.perf_counter() - t0) / len(chunk)
            for r, (passages, n_scored), reply in zip(chunk, selections, replies):
                f.write(json.dumps({"id": r["id"], "question": r["question"], "gold": r["gold"], "type": r["type"],
                                    "pred": reply["answer"], "raw": reply["raw"],
                                    "docids": [p["docid"] for p in passages], "n_scored": n_scored,
                                    "gold_in_context": retrieval.gold_in(r["gold"], passages),
                                    "generate_s": generate_s}) + "\n")
            f.flush()
            print(f"  {min(start + chunk_size, len(todo))}/{len(todo)}  {generate_s:.2f}s per question")
    ids = {r["id"] for r in records}
    return [p for p in retrieval.read_jsonl(path) if p["id"] in ids]


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def score_predictions(preds):
    """EM and F1 (x100, hotpot_eval.py) overall and per question type, plus context and cost statistics.

    answer_in_context is over questions whose answer isn't yes/no, like the retrieval sweep.
    """
    row = {"n": len(preds),
           "em": 100 * _mean(exact_match_score(p["pred"], p["gold"]) for p in preds),
           "f1": 100 * _mean(f1_score(p["pred"], p["gold"])[0] for p in preds)}
    for qtype in ("bridge", "comparison"):
        subset = [p for p in preds if p["type"] == qtype]
        row[f"em_{qtype}"] = 100 * _mean(exact_match_score(p["pred"], p["gold"]) for p in subset)
        row[f"f1_{qtype}"] = 100 * _mean(f1_score(p["pred"], p["gold"])[0] for p in subset)
    row["answer_in_context"] = 100 * _mean(p["gold_in_context"] for p in preds if p["gold_in_context"] is not None)
    row["avg_passages"] = _mean(len(p["docids"]) for p in preds)
    row["avg_docs_scored"] = _mean(p["n_scored"] for p in preds)
    row["generate_s"] = _mean(p["generate_s"] for p in preds)
    return row


def evaluate(records, reader=None, settings=tuple(SETTINGS), pred_dir=PRED_DIR, out_path=EVAL_PATH, overwrite=False):
    """Run and score each setting on `records`, writing one row per setting to `out_path`."""
    rows = [{"setting": name, **score_predictions(run_setting(name, records, reader, pred_dir, overwrite))}
            for name in settings]
    write_eval_csv(rows, out_path)
    return rows


def write_eval_csv(rows, path=EVAL_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: f"{v:.2f}" if isinstance(v, float) else v for k, v in r.items()} for r in rows)


def format_eval(rows):
    lines = [f"{'setting':<12} {'n':>5} {'EM':>6} {'F1':>6} {'EM_br':>6} {'EM_cmp':>6} {'ctx%':>6} {'psg':>4} "
             f"{'scored':>6} {'gen_s':>6}"]
    for r in rows:
        lines.append(f"{r['setting']:<12} {r['n']:>5} {r['em']:>6.2f} {r['f1']:>6.2f} {r['em_bridge']:>6.2f} "
                     f"{r['em_comparison']:>6.2f} {r['answer_in_context']:>6.2f} {r['avg_passages']:>4.1f} "
                     f"{r['avg_docs_scored']:>6.2f} {r['generate_s']:>6.2f}")
    return "\n".join(lines)


# Command line

def cmd_eval(args):
    records = retrieval.read_jsonl(args.scores, args.limit)
    reader = None if args.score_only else LlamaReader(args.model, args.batch_size)
    rows = evaluate(records, reader, args.settings, args.pred_dir, args.out, args.overwrite)
    print(format_eval(rows))


def cmd_ask(args):
    from utils import env

    env.ensure_java()
    retriever = retrieval.Retriever(cache_path=args.cache)
    pipeline = RAGPipeline(retriever, retrieval.T5Scorer(), LlamaReader(args.model, args.batch_size))
    res = pipeline.answer(args.question)
    for i, p in enumerate(res["passages"], 1):
        print(f"[{i}] {p['title']} (monoT5 {p['t5']:.3f})")
    print(f"Scored {res['n_scored']} passages. Answer: {res['answer']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("eval", help="answer and score the dev questions under each setting")
    ev.add_argument("--settings", nargs="*", choices=list(SETTINGS), default=list(SETTINGS))
    ev.add_argument("--scores", default=retrieval.SCORES_PATH)
    ev.add_argument("--pred-dir", default=PRED_DIR)
    ev.add_argument("--out", default=EVAL_PATH)
    ev.add_argument("--limit", type=int, default=None)
    ev.add_argument("--overwrite", action="store_true")
    ev.add_argument("--score-only", action="store_true", help="score saved predictions without loading Llama")
    ask = sub.add_parser("ask", help="answer one question with the live pipeline")
    ask.add_argument("question")
    ask.add_argument("--cache", default=retrieval.CACHE_PATH)
    for p in (ev, ask):
        p.add_argument("--model", default=LLAMA_MODEL)
        p.add_argument("--batch-size", type=int, default=LLAMA_BATCH_SIZE, help="questions per Llama generate() call")
    args = ap.parse_args()
    {"eval": cmd_eval, "ask": cmd_ask}[args.cmd](args)


if __name__ == "__main__":
    main()
