import argparse
import json
import os
import re
import string
import sys
import time

DATA_PATH = "data/dev.jsonl"
OUT_DIR = "outputs"
CACHE_PATH = os.path.join(OUT_DIR, "bm25_top15.json")
SCORES_PATH = os.path.join(OUT_DIR, "t5_scores.jsonl")
SWEEP_PATH = os.path.join(OUT_DIR, "sweep.csv")

BATCH_SIZE = 5
MAX_DOCS = 15
DEFAULT_THRESHOLD = 0.98
DEFAULT_KEEP = 2
DEFAULT_FALLBACK = "top"


def load_dev(limit=None, path=DATA_PATH):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


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
    text = " ".join(text.split())
    return title.strip(), " ".join(text.split()[:300])


class Retriever:
    def __init__(self, index="wikipedia-dpr-100w", cache_path=None):
        self.cache = {}
        self.searcher = None
        self.index = index
        self._memo = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self.cache = json.load(f)

    def _load_searcher(self):
        if self.searcher is None:
            from pyserini.search.lucene import LuceneSearcher
            if os.path.isdir(self.index):
                self.searcher = LuceneSearcher(self.index)
            else:
                self.searcher = LuceneSearcher.from_prebuilt_index(self.index)
        return self.searcher

    def _hit(self, searcher, h, rank):
        doc = searcher.doc(h.docid)
        raw = (doc.raw() or doc.contents()) if doc is not None else ""
        title, text = parse_doc(raw)
        return {"rank": rank, "docid": h.docid, "title": title, "text": text, "bm25": float(h.score)}

    def _search(self, query, depth):
        key = (query, depth)
        if key not in self._memo:
            searcher = self._load_searcher()
            hits = searcher.search(query, k=depth)
            self._memo[key] = [self._hit(searcher, h, i + 1) for i, h in enumerate(hits)]
        return self._memo[key]

    def retrieve(self, query, offset=0, k=BATCH_SIZE, qid=None):
        if qid is not None and qid in self.cache:
            hits = self.cache[qid]
            if offset + k <= len(hits):
                return [dict(h) for h in hits[offset:offset + k]]
        depth = max(MAX_DOCS, offset + k)
        return [dict(h) for h in self._search(query, depth)[offset:offset + k]]


class T5Scorer:
    def __init__(self, model_name="castorini/monot5-large-msmarco", tokenizer_name=None,
                 batch_size=16, max_length=512, dtype=None, device=None):
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
        if self.true_id == self.tok.unk_token_id or self.false_id == self.tok.unk_token_id:
            sys.exit("The tokenizer has no true/false tokens. Use a T5 tokenizer with a SentencePiece vocabulary.")
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


def run_loop(get_batch, threshold=DEFAULT_THRESHOLD, batch_size=BATCH_SIZE, max_docs=MAX_DOCS,
             keep=DEFAULT_KEEP, fallback=DEFAULT_FALLBACK):
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
    used_fallback = not above
    if above:
        selected = above[:keep]
    elif fallback == "top":
        selected = ranked[:keep]
    else:
        selected = []
    return {"selected": selected, "ranked": ranked, "n_scored": len(scored), "batches": batches,
            "stopped_early": stopped, "fallback": used_fallback,
            "best_score": ranked[0]["t5"] if ranked else None}


def adaptive_retrieve(query, retriever, scorer, qid=None, threshold=DEFAULT_THRESHOLD,
                      batch_size=BATCH_SIZE, max_docs=MAX_DOCS, keep=DEFAULT_KEEP, fallback=DEFAULT_FALLBACK):
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
    hits = sorted(record["hits"], key=lambda h: h["rank"])
    per_doc = record.get("score_s", 0.0) / max(len(hits), 1)

    def get_batch(offset, k):
        return [dict(h) for h in hits[offset:offset + k]]

    res = run_loop(get_batch, threshold, batch_size, max_docs, keep, fallback)
    res["est_score_s"] = per_doc * res["n_scored"]
    return res


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def gold_in(gold, passages):
    g = normalize_answer(gold)
    if not g or g in ("yes", "no"):
        return None
    return any(g in normalize_answer(f"{p.get('title', '')} {p.get('text', '')}") for p in passages)


def cmd_cache(args):
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(args.index) if os.path.isdir(args.index) else LuceneSearcher.from_prebuilt_index(args.index)
    retr = Retriever(args.index)
    retr.searcher = searcher
    dev = load_dev(args.limit, args.data)
    cache = {}
    if os.path.exists(args.cache) and not args.overwrite:
        with open(args.cache) as f:
            cache = json.load(f)
    todo = [ex for ex in dev if ex["_id"] not in cache]
    print(f"{len(dev)} questions, {len(todo)} to retrieve")
    for start in range(0, len(todo), 256):
        chunk = todo[start:start + 256]
        res = searcher.batch_search([ex["question"] for ex in chunk], [ex["_id"] for ex in chunk],
                                    k=args.depth, threads=args.threads)
        for ex in chunk:
            cache[ex["_id"]] = [retr._hit(searcher, h, i + 1) for i, h in enumerate(res.get(ex["_id"], []))]
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        with open(args.cache, "w") as f:
            json.dump(cache, f)
        print(f"{min(start + 256, len(todo))}/{len(todo)}")


def cmd_score(args):
    if not os.path.exists(args.cache):
        sys.exit(f"Missing {args.cache}. Run the cache step first in the Pyserini environment.")
    retr = Retriever(cache_path=args.cache)
    scorer = T5Scorer(args.model, args.tokenizer, args.batch_size, dtype=args.dtype)
    dev = load_dev(args.limit, args.data)
    done = set()
    if os.path.exists(args.scores) and not args.overwrite:
        with open(args.scores) as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    mode = "a" if done else "w"
    todo = [ex for ex in dev if ex["_id"] not in done]
    print(f"{len(dev)} questions, {len(todo)} to score")
    t_all = time.perf_counter()
    with open(args.scores, mode) as f:
        for i, ex in enumerate(todo):
            hits = retr.retrieve(ex["question"], 0, MAX_DOCS, qid=ex["_id"])
            t0 = time.perf_counter()
            scores = scorer.score_batch(ex["question"], hits)
            dt = time.perf_counter() - t0
            for h, s in zip(hits, scores):
                h["t5"] = float(s)
            f.write(json.dumps({"id": ex["_id"], "question": ex["question"], "gold": ex["answer"],
                                "type": ex.get("type", ""), "score_s": dt, "hits": hits}) + "\n")
            if (i + 1) % 50 == 0:
                f.flush()
                print(f"{i + 1}/{len(todo)}  {(time.perf_counter() - t_all) / (i + 1):.2f}s per question")


def load_scores(path, limit=None):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def sweep(records, thresholds, caps, keep=DEFAULT_KEEP, fallback=DEFAULT_FALLBACK):
    rows = []
    for cap in caps:
        for th in thresholds:
            fb = docs = early = 0
            hit = tot = base_hit = 0
            for r in records:
                res = select_from_record(r, th, BATCH_SIZE, cap, keep, fallback)
                fb += res["fallback"]
                docs += res["n_scored"]
                early += res["stopped_early"]
                g = gold_in(r["gold"], res["selected"])
                if g is not None:
                    tot += 1
                    hit += g
                    base_hit += gold_in(r["gold"], sorted(r["hits"], key=lambda h: h["rank"])[:keep])
            n = max(len(records), 1)
            rows.append({"cap": cap, "threshold": th, "fallback_rate": 100 * fb / n,
                         "avg_docs_scored": docs / n, "stopped_early_rate": 100 * early / n,
                         "gold_in_selected": 100 * hit / max(tot, 1),
                         "gold_in_bm25_topkeep": 100 * base_hit / max(tot, 1)})
    return rows


def cmd_sweep(args):
    records = load_scores(args.scores, args.limit)
    rows = sweep(records, args.thresholds, args.caps, args.keep, args.fallback)
    cols = list(rows[0].keys())
    with open(args.out, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:g}" if c == "threshold" else f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    print(f"{len(records)} questions, keep={args.keep}, fallback={args.fallback}")
    print(f"{'cap':>4} {'thr':>6} {'fallback%':>10} {'docs':>6} {'early%':>7} {'gold_sel%':>10} {'gold_bm25%':>11}")
    for r in rows:
        print(f"{r['cap']:>4} {r['threshold']:>6g} {r['fallback_rate']:>10.2f} {r['avg_docs_scored']:>6.2f} "
              f"{r['stopped_early_rate']:>7.2f} {r['gold_in_selected']:>10.2f} {r['gold_in_bm25_topkeep']:>11.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "score", "sweep"])
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--index", default="wikipedia-dpr-100w")
    ap.add_argument("--cache", default=CACHE_PATH)
    ap.add_argument("--scores", default=SCORES_PATH)
    ap.add_argument("--out", default=SWEEP_PATH)
    ap.add_argument("--depth", type=int, default=MAX_DOCS)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--model", default="castorini/monot5-large-msmarco")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--thresholds", type=float, nargs="*",
                    default=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995, 1.01])
    ap.add_argument("--caps", type=int, nargs="*", default=[5, 10, 15])
    ap.add_argument("--keep", type=int, default=DEFAULT_KEEP)
    ap.add_argument("--fallback", choices=["top", "none"], default=DEFAULT_FALLBACK)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    {"cache": cmd_cache, "score": cmd_score, "sweep": cmd_sweep}[args.cmd](args)


if __name__ == "__main__":
    main()
