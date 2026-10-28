"""BM25 retrieval, monoT5 scoring, and adaptive early stopping for HotpotQA.

The notebook imports these functions directly. The same three steps also run from the shell:

    python retrieval.py cache   # BM25 top 15 per dev question (Pyserini + Java 21)
    python retrieval.py score   # monoT5 relevance for every cached passage (GPU)
    python retrieval.py sweep   # replay adaptive stopping offline per threshold and cap

Paths and the index come from .env (see config.py).
"""
import argparse
import json
import os
import re
import string
import time

from utils import config

DATA_PATH = os.path.join(config.DIR_DATA, "dev.jsonl")
OUT_DIR = config.DIR_OUTPUTS
CACHE_PATH = os.path.join(OUT_DIR, "bm25_top15.json")
SCORES_PATH = os.path.join(OUT_DIR, "t5_scores.jsonl")
SWEEP_PATH = os.path.join(OUT_DIR, "sweep.csv")

INDEX = config.PYSERINI_INDEX
T5_MODEL = "castorini/monot5-large-msmarco"

BATCH_SIZE = 5  # passages retrieved and scored per step of the adaptive loop
MAX_DOCS = 15  # BM25 depth, and the largest cap
T5_BATCH_SIZE = 15  # passages per monoT5 forward pass
DEFAULT_THRESHOLD = 0.98
DEFAULT_KEEP = 2
DEFAULT_FALLBACK = "top"
DEFAULT_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995, 1.01]
DEFAULT_CAPS = [5, 10, 15]

PUNCTUATION = set(string.punctuation)


# I/O

def read_jsonl(path, limit=None):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


# BM25 retrieval

def parse_doc(raw):
    title, text = "", raw or ""
    try:
        d = json.loads(text)
        title = d.get("title", "") or ""
        text = d.get("text") or d.get("contents") or ""
    except (ValueError, TypeError, AttributeError):
        pass
    if not title:
        first, _, rest = text.partition("\n")
        if rest and len(first.split()) <= 15:
            title, text = first.strip().strip('"'), rest
    words = text.split()
    return title.strip(), " ".join(words[:300])


def open_searcher(index=INDEX):
    from pyserini.search.lucene import LuceneSearcher

    if os.path.isdir(index):
        return LuceneSearcher(index)
    return LuceneSearcher.from_prebuilt_index(index)


def make_hit(searcher, hit, rank):
    doc = searcher.doc(hit.docid)
    raw = (doc.raw() or doc.contents()) if doc is not None else ""
    title, text = parse_doc(raw)
    return {"rank": rank, "docid": hit.docid, "title": title, "text": text, "bm25": float(hit.score)}


class Retriever:
    """Serves BM25 hits from a cache keyed by question id, searching live on a miss."""

    def __init__(self, index=INDEX, cache=None, cache_path=None, searcher=None):
        self.index = index
        self.searcher = searcher
        if cache is None and cache_path and os.path.exists(cache_path):
            cache = read_json(cache_path)
        self.cache = cache or {}
        self._memo = {}

    def _search(self, query, depth):
        key = (query, depth)
        if key not in self._memo:
            if self.searcher is None:
                self.searcher = open_searcher(self.index)
            hits = self.searcher.search(query, k=depth)
            self._memo[key] = [make_hit(self.searcher, h, i + 1) for i, h in enumerate(hits)]
        return self._memo[key]

    def retrieve(self, query, offset=0, k=BATCH_SIZE, qid=None):
        if qid is not None and qid in self.cache:
            hits = self.cache[qid]
            if offset + k <= len(hits):
                return [dict(h) for h in hits[offset:offset + k]]
        depth = max(MAX_DOCS, offset + k)
        return [dict(h) for h in self._search(query, depth)[offset:offset + k]]


def build_cache(questions, searcher=None, index=INDEX, cache_path=CACHE_PATH, depth=MAX_DOCS,
                threads=8, overwrite=False, chunk_size=256):
    """Retrieve the top `depth` BM25 hits for each question, saving to `cache_path` after every chunk."""
    if searcher is None:
        searcher = open_searcher(index)
    cache = {} if overwrite or not os.path.exists(cache_path) else read_json(cache_path)
    todo = [q for q in questions if q["_id"] not in cache]
    print(f"{len(questions)} questions, {len(todo)} to retrieve")
    for start in range(0, len(todo), chunk_size):
        chunk = todo[start:start + chunk_size]
        res = searcher.batch_search([q["question"] for q in chunk], [q["_id"] for q in chunk],
                                    k=depth, threads=threads)
        for q in chunk:
            cache[q["_id"]] = [make_hit(searcher, h, i + 1) for i, h in enumerate(res.get(q["_id"], []))]
        write_json(cache, cache_path)
        print(f"{min(start + chunk_size, len(todo))}/{len(todo)}")
    return cache


# monoT5 scoring

class T5Scorer:
    def __init__(self, model_name=T5_MODEL, tokenizer_name=None, batch_size=T5_BATCH_SIZE,
                 max_length=512, dtype=None, device=None):
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = "bfloat16" if self.device == "cuda" and torch.cuda.is_bf16_supported() else "float32"
        try:
            self.tok = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        except (OSError, ValueError):
            self.tok = AutoTokenizer.from_pretrained("t5-large")
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=getattr(torch, dtype)).to(self.device).eval()
        self.true_id = self.tok.convert_tokens_to_ids("▁true")
        self.false_id = self.tok.convert_tokens_to_ids("▁false")
        if self.tok.unk_token_id in (self.true_id, self.false_id):
            raise ValueError("The tokenizer has no true/false tokens. Use a T5 tokenizer with a SentencePiece vocabulary.")
        self.batch_size = batch_size
        self.max_length = max_length

    @staticmethod
    def format_input(query, passage):
        if isinstance(passage, dict):
            body = f"{passage.get('title', '')}. {passage.get('text', '')}".strip(". ")
        else:
            body = str(passage)
        return f"Query: {query} Document: {body} Relevant:"

    def score_batch(self, query, passages):
        """P(relevant) for each passage: softmax over the first decoded token's true/false logits."""
        torch = self.torch
        out = []
        for start in range(0, len(passages), self.batch_size):
            chunk = passages[start:start + self.batch_size]
            enc = self.tok([self.format_input(query, p) for p in chunk], padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            dec = torch.full((len(chunk), 1), self.model.config.decoder_start_token_id,
                             dtype=torch.long, device=self.device)
            with torch.inference_mode():
                logits = self.model(**enc, decoder_input_ids=dec).logits[:, 0, [self.false_id, self.true_id]]
            out.extend(torch.softmax(logits.float(), dim=-1)[:, 1].tolist())
        return out

    def score(self, query, passage):
        return self.score_batch(query, [passage])[0]


def score_questions(questions, retriever, scorer, scores_path=SCORES_PATH, depth=MAX_DOCS, overwrite=False):
    """Score the top `depth` passages for each question and append one record per question to `scores_path`.

    Questions already in `scores_path` are skipped unless `overwrite` is set.
    """
    done = set()
    if os.path.exists(scores_path) and not overwrite:
        done = {r["id"] for r in read_jsonl(scores_path)}
    todo = [q for q in questions if q["_id"] not in done]
    print(f"{len(questions)} questions, {len(todo)} to score")
    os.makedirs(os.path.dirname(scores_path) or ".", exist_ok=True)
    t_all = time.perf_counter()
    with open(scores_path, "a" if done else "w") as f:
        for i, q in enumerate(todo):
            hits = retriever.retrieve(q["question"], 0, depth, qid=q["_id"])
            t0 = time.perf_counter()
            scores = scorer.score_batch(q["question"], hits)
            dt = time.perf_counter() - t0
            for h, s in zip(hits, scores):
                h["t5"] = float(s)
            f.write(json.dumps({"id": q["_id"], "question": q["question"], "gold": q["answer"],
                                "type": q.get("type", ""), "score_s": dt, "hits": hits}) + "\n")
            if (i + 1) % 50 == 0:
                f.flush()
                print(f"{i + 1}/{len(todo)}  {(time.perf_counter() - t_all) / (i + 1):.2f}s per question")


# Adaptive early stopping

def run_loop(get_batch, threshold=DEFAULT_THRESHOLD, batch_size=BATCH_SIZE, max_docs=MAX_DOCS,
             keep=DEFAULT_KEEP, fallback=DEFAULT_FALLBACK):
    """Score passages `batch_size` at a time until one reaches `threshold` or `max_docs` are scored.

    `get_batch(offset, k)` returns up to k hits with a "t5" score. Selects up to `keep` hits at or above
    the threshold; if none qualify, `fallback="top"` takes the best `keep` anyway and "none" takes nothing.
    """
    if fallback not in ("top", "none"):
        raise ValueError("fallback must be 'top' or 'none'")
    scored, batches, offset, stopped = [], 0, 0, False
    while offset < max_docs:
        k = min(batch_size, max_docs - offset)
        hits = get_batch(offset, k)
        if not hits:
            break
        scored.extend(hits)
        batches += 1
        offset += len(hits)
        if any(h["t5"] >= threshold for h in hits):
            stopped = True
            break
        if len(hits) < k:
            break
    ranked = sorted(scored, key=lambda h: (-h["t5"], h["rank"]))
    above = [h for h in ranked if h["t5"] >= threshold]
    if above:
        selected = above[:keep]
    elif fallback == "top":
        selected = ranked[:keep]
    else:
        selected = []
    return {"selected": selected, "ranked": ranked, "n_scored": len(scored), "batches": batches,
            "stopped_early": stopped, "fallback": not above,
            "best_score": ranked[0]["t5"] if ranked else None}


def adaptive_retrieve(query, retriever, scorer, qid=None, threshold=DEFAULT_THRESHOLD,
                      batch_size=BATCH_SIZE, max_docs=MAX_DOCS, keep=DEFAULT_KEEP, fallback=DEFAULT_FALLBACK):
    """Run the adaptive loop live, retrieving and scoring each batch on demand."""
    timing = {"retrieve_s": 0.0, "score_s": 0.0}

    def get_batch(offset, k):
        t0 = time.perf_counter()
        hits = retriever.retrieve(query, offset, k, qid=qid)
        timing["retrieve_s"] += time.perf_counter() - t0
        if not hits:
            return []
        t0 = time.perf_counter()
        scores = scorer.score_batch(query, hits)
        timing["score_s"] += time.perf_counter() - t0
        for h, s in zip(hits, scores):
            h["t5"] = float(s)
        return hits

    t0 = time.perf_counter()
    res = run_loop(get_batch, threshold, batch_size, max_docs, keep, fallback)
    res["latency_s"] = time.perf_counter() - t0
    res.update(timing)
    return res


def select_from_record(record, threshold=DEFAULT_THRESHOLD, batch_size=BATCH_SIZE, max_docs=MAX_DOCS,
                       keep=DEFAULT_KEEP, fallback=DEFAULT_FALLBACK):
    """Replay the adaptive loop on a record from `score_questions`, without running the model."""
    hits = bm25_order(record)
    per_doc = record.get("score_s", 0.0) / max(len(hits), 1)

    def get_batch(offset, k):
        return [dict(h) for h in hits[offset:offset + k]]

    res = run_loop(get_batch, threshold, batch_size, max_docs, keep, fallback)
    res["est_score_s"] = per_doc * res["n_scored"]
    return res


def bm25_order(record):
    return sorted(record["hits"], key=lambda h: h["rank"])


# Evaluation

def normalize_answer(s):
    s = "".join(ch for ch in s.lower() if ch not in PUNCTUATION)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def gradable(gold):
    """Whether `gold_in` can check this answer: yes/no and empty answers can't be string-matched."""
    g = normalize_answer(gold)
    return bool(g) and g not in ("yes", "no")


def gold_in(gold, passages):
    """Whether the normalized gold answer appears in any passage, or None if the answer isn't gradable."""
    if not gradable(gold):
        return None
    g = normalize_answer(gold)
    return any(g in normalize_answer(f"{p.get('title', '')} {p.get('text', '')}") for p in passages)


SWEEP_COLUMNS = ["cap", "threshold", "fallback_rate", "avg_docs_scored", "stopped_early_rate",
                 "gold_in_selected", "gold_in_bm25_topkeep"]


def sweep(records, thresholds=DEFAULT_THRESHOLDS, caps=DEFAULT_CAPS, keep=DEFAULT_KEEP, fallback=DEFAULT_FALLBACK):
    """One row per (cap, threshold). Rates are over all records; gold_* are over gradable records only."""
    n = max(len(records), 1)
    graded = [r for r in records if gradable(r["gold"])]
    n_graded = max(len(graded), 1)
    bm25_hits = sum(gold_in(r["gold"], bm25_order(r)[:keep]) for r in graded)
    rows = []
    for cap in caps:
        for th in thresholds:
            results = [select_from_record(r, th, BATCH_SIZE, cap, keep, fallback) for r in records]
            selected_hits = sum(gold_in(r["gold"], res["selected"])
                                for r, res in zip(records, results) if gradable(r["gold"]))
            rows.append({"cap": cap, "threshold": th,
                         "fallback_rate": 100 * sum(res["fallback"] for res in results) / n,
                         "avg_docs_scored": sum(res["n_scored"] for res in results) / n,
                         "stopped_early_rate": 100 * sum(res["stopped_early"] for res in results) / n,
                         "gold_in_selected": 100 * selected_hits / n_graded,
                         "gold_in_bm25_topkeep": 100 * bm25_hits / n_graded})
    return rows


def _csv_cell(col, value):
    if col == "threshold":
        return f"{value:g}"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def write_sweep_csv(rows, path=SWEEP_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(SWEEP_COLUMNS) + "\n")
        for r in rows:
            f.write(",".join(_csv_cell(c, r[c]) for c in SWEEP_COLUMNS) + "\n")


def format_sweep(rows):
    lines = [f"{'cap':>4} {'thr':>6} {'fallback%':>10} {'docs':>6} {'early%':>7} {'gold_sel%':>10} {'gold_bm25%':>11}"]
    for r in rows:
        lines.append(f"{r['cap']:>4} {r['threshold']:>6g} {r['fallback_rate']:>10.2f} {r['avg_docs_scored']:>6.2f} "
                     f"{r['stopped_early_rate']:>7.2f} {r['gold_in_selected']:>10.2f} {r['gold_in_bm25_topkeep']:>11.2f}")
    return "\n".join(lines)


# Command line

def cmd_cache(args):
    build_cache(read_jsonl(args.data, args.limit), index=args.index, cache_path=args.cache,
                depth=args.depth, threads=args.threads, overwrite=args.overwrite)


def cmd_score(args):
    if not os.path.exists(args.cache):
        raise SystemExit(f"Missing {args.cache}. Run the cache step first.")
    retriever = Retriever(args.index, cache_path=args.cache)
    scorer = T5Scorer(args.model, args.tokenizer, args.t5_batch_size, dtype=args.dtype)
    score_questions(read_jsonl(args.data, args.limit), retriever, scorer, args.scores, overwrite=args.overwrite)


def cmd_sweep(args):
    records = read_jsonl(args.scores, args.limit)
    rows = sweep(records, args.thresholds, args.caps, args.keep, args.fallback)
    write_sweep_csv(rows, args.out)
    print(f"{len(records)} questions, keep={args.keep}, fallback={args.fallback}")
    print(format_sweep(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["cache", "score", "sweep"])
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--index", default=INDEX, help="Pyserini prebuilt index name or local index directory")
    ap.add_argument("--cache", default=CACHE_PATH)
    ap.add_argument("--scores", default=SCORES_PATH)
    ap.add_argument("--out", default=SWEEP_PATH)
    ap.add_argument("--depth", type=int, default=MAX_DOCS, help="BM25 hits cached per question")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--model", default=T5_MODEL)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--t5-batch-size", type=int, default=T5_BATCH_SIZE, help="passages per monoT5 forward pass")
    ap.add_argument("--thresholds", type=float, nargs="*", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--caps", type=int, nargs="*", default=DEFAULT_CAPS)
    ap.add_argument("--keep", type=int, default=DEFAULT_KEEP)
    ap.add_argument("--fallback", choices=["top", "none"], default=DEFAULT_FALLBACK)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    {"cache": cmd_cache, "score": cmd_score, "sweep": cmd_sweep}[args.cmd](args)


if __name__ == "__main__":
    main()
