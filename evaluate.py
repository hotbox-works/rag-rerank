"""End-to-end experiments on the HotpotQA dev set, scored with hotpot_eval.py (answer EM/F1 only).

Each run writes outputs/runs/<name>/{config.json, predictions.jsonl, metrics.json}. Reruns skip questions already
in predictions.jsonl, so an interrupted run continues where it stopped.

    python evaluate.py list                       # experiments and groups
    python evaluate.py run --group baselines      # closed book, BM25 top-2 / top-8
    python evaluate.py run --group sweep          # threshold x keep grid, raw passages (Llama answers only)
    python evaluate.py best                       # pick the best sweep setting by F1 -> best_settings.json
    python evaluate.py run --group ablations      # fallback and summarizer ablations on the best setting
    python evaluate.py run --group finetuned      # same, with the fine-tuned scorer's saved scores
    python evaluate.py table                      # every run -> results.csv
    python evaluate.py latency --n 100            # live, one question at a time, adaptive vs. no early stop
    python evaluate.py test --exp final           # test-set predictions in HotpotQA format

Add --limit 20 to any run for a quick check. Passage selection is replayed from saved monoT5 scores
(retrieval.py score), so runs only pay for Llama. Summaries are cached in summaries.jsonl and shared across runs.
"""
import argparse
import json
import os
import statistics
import time

import generator
import pipeline
import retrieval
from hotpot_eval import exact_match_score, f1_score
from utils import config

RUNS_DIR = os.path.join(config.DIR_OUTPUTS, "runs")
RESULTS_PATH = os.path.join(config.DIR_OUTPUTS, "results.csv")
BEST_PATH = os.path.join(config.DIR_OUTPUTS, "best_settings.json")
LATENCY_PATH = os.path.join(config.DIR_OUTPUTS, "latency.csv")
TEST_SCORES_PATH = os.path.join(config.DIR_OUTPUTS, "t5_scores_test.jsonl")
TEST_CACHE_PATH = os.path.join(config.DIR_OUTPUTS, "bm25_top15_test.json")
TEST_PRED_PATH = os.path.join(config.DIR_OUTPUTS, "test_predictions.json")
FT_SCORES_PATH = os.path.join(config.DIR_OUTPUTS, "t5_scores_ft.jsonl")

EVAL_BATCH_SIZE = 32  # questions per Llama batch

SWEEP_THRESHOLDS = [0.5, 0.9, 0.98, 1.01]
SWEEP_KEEPS = [1, 2, 3, 5, 8]
SWEEP_CAP = 15


# Experiments

def _sweep_name(th, keep, cap=SWEEP_CAP):
    return f"sweep_th{th:g}_cap{cap}_k{keep}"


def load_best():
    """Best sweep setting if `best` has been run, else the current defaults."""
    if os.path.exists(BEST_PATH):
        return retrieval.read_json(BEST_PATH)["settings"]
    return {"threshold": retrieval.DEFAULT_THRESHOLD, "cap": retrieval.MAX_DOCS, "keep": retrieval.DEFAULT_KEEP}


def experiments():
    """name -> {"settings": Pipeline settings, "scores": saved monoT5 scores to replay}. Rebuilt on each call so
    the ablations pick up best_settings.json."""
    exps = {
        "closed_book": {"settings": {"mode": "closed_book", "summarize": False}},
        "bm25_k2_raw": {"settings": {"mode": "bm25", "keep": 2, "summarize": False}},
        "bm25_k8_raw": {"settings": {"mode": "bm25", "keep": 8, "summarize": False}},
    }
    for th in SWEEP_THRESHOLDS:
        for keep in SWEEP_KEEPS:
            exps[_sweep_name(th, keep)] = {"settings": {"threshold": th, "cap": SWEEP_CAP, "keep": keep,
                                                       "fallback": "top", "summarize": False}}
    best = {k: v for k, v in load_best().items() if k in ("threshold", "cap", "keep")}
    for fb in pipeline.FALLBACKS:
        exps[f"best_fallback_{fb}_raw"] = {"settings": {**best, "fallback": fb, "summarize": False}}
    exps["best_raw"] = {"settings": {**best, "fallback": "top", "summarize": False}}
    exps["best_sum"] = {"settings": {**best, "fallback": "top", "summarize": True}}
    exps["best_sum_drop"] = {"settings": {**best, "fallback": "top", "summarize": True, "drop_irrelevant": True}}
    exps["best_nostop_raw"] = {"settings": {**best, "threshold": 1.01, "fallback": "top", "summarize": False}}
    exps["ft_best_raw"] = {"settings": {**best, "fallback": "top", "summarize": False}, "scores": FT_SCORES_PATH}
    exps["ft_best_sum"] = {"settings": {**best, "fallback": "top", "summarize": True}, "scores": FT_SCORES_PATH}
    exps["ft_nostop_raw"] = {"settings": {**best, "threshold": 1.01, "fallback": "top", "summarize": False},
                             "scores": FT_SCORES_PATH}
    # The fine-tuned scorer is less confident than monoT5 (at 0.98 it almost never stops early), so a threshold
    # tuned for monoT5 doesn't transfer. 0.9 scores about as many passages as monoT5 does at 0.98.
    exps["ft_th0.9_raw"] = {"settings": {**best, "threshold": 0.9, "fallback": "top", "summarize": False},
                            "scores": FT_SCORES_PATH}
    for spec in exps.values():
        spec.setdefault("scores", retrieval.SCORES_PATH)
    return exps


GROUPS = {
    "baselines": ["closed_book", "bm25_k2_raw", "bm25_k8_raw"],
    "sweep": [_sweep_name(th, k) for th in SWEEP_THRESHOLDS for k in SWEEP_KEEPS],
    "ablations": [f"best_fallback_{fb}_raw" for fb in pipeline.FALLBACKS]
                 + ["best_raw", "best_nostop_raw", "best_sum", "best_sum_drop"],
    "finetuned": ["ft_nostop_raw", "ft_th0.9_raw", "ft_best_raw", "ft_best_sum"],
}


# Metrics

def score_prediction(pred, gold):
    f1, prec, recall = f1_score(pred, gold)
    return {"em": float(exact_match_score(pred, gold)), "f1": f1, "prec": prec, "recall": recall}


def outcome(row):
    """correct / partial / retrieval_miss / reader_error / ungradable, for error analysis."""
    if row["em"] == 1.0:
        return "correct"
    found = row.get("gold_in_context")
    if found is None:
        return "ungradable" if row["f1"] == 0 else "partial"
    if row["f1"] > 0:
        return "partial"
    return "reader_error" if found else "retrieval_miss"


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def compute_metrics(rows):
    """Aggregate per-question rows (as written to predictions.jsonl). EM/F1 are percentages, like the README."""
    out = {"n": len(rows)}
    for k in ("em", "f1", "prec", "recall"):
        out[k] = 100 * _mean(r[k] for r in rows)
    for t in sorted({r.get("type", "") for r in rows}):
        sub = [r for r in rows if r.get("type", "") == t]
        out[f"em_{t or 'unknown'}"] = 100 * _mean(r["em"] for r in sub)
        out[f"f1_{t or 'unknown'}"] = 100 * _mean(r["f1"] for r in sub)
    graded = [r for r in rows if r.get("gold_in_context") is not None]
    out["gold_in_context"] = 100 * _mean(bool(r["gold_in_context"]) for r in graded)
    out["fallback_rate"] = 100 * _mean(r["fallback"] for r in rows)
    out["closed_book_rate"] = 100 * _mean(r["closed_book"] for r in rows)
    out["avg_docs_scored"] = _mean(r["n_scored"] for r in rows)
    out["avg_contexts"] = _mean(len(r["contexts"]) for r in rows)
    for k in ("retrieve_s", "score_s", "summarize_s", "generate_s"):
        out[k] = _mean(r["timing"].get(k, 0.0) for r in rows)
    counts = {}
    for r in rows:
        counts[outcome(r)] = counts.get(outcome(r), 0) + 1
    for k in ("correct", "partial", "retrieval_miss", "reader_error", "ungradable"):
        out[f"pct_{k}"] = 100 * counts.get(k, 0) / max(len(rows), 1)
    return out


def finish_row(res, q):
    """Add the gold answer, scores and gold-in-context check to a Pipeline result."""
    gold = q.get("answer", "")
    row = {**res, "gold": gold, "type": q.get("type", ""), **score_prediction(res["answer"], gold)}
    row["gold_in_context"] = retrieval.gold_in(gold, res["selected"]) if res["selected"] else (
        None if not retrieval.gradable(gold) else False)
    for h in row["selected"]:
        h.pop("text", None)  # keep predictions.jsonl small; contexts holds what Llama saw
    return row


# Running

class Runner:
    """Holds one Llama and one summary cache for every run in a session."""

    def __init__(self, llm=None, backend=generator.LLAMA_BACKEND, summary_cache=None, runs_dir=RUNS_DIR,
                 batch_size=EVAL_BATCH_SIZE):
        self._llm = llm
        self.backend = backend
        self.cache = summary_cache if summary_cache is not None else pipeline.SummaryCache()
        self.runs_dir = runs_dir
        self.batch_size = batch_size
        self._selectors = {}

    @property
    def llm(self):
        if self._llm is None:
            self._llm = generator.Llama(backend=self.backend)
        return self._llm

    def selector(self, scores_path):
        if scores_path not in self._selectors:
            if not os.path.exists(scores_path):
                raise SystemExit(f"Missing {scores_path}. Score this split with monoT5 first.")
            self._selectors[scores_path] = pipeline.ReplaySelector.from_path(scores_path)
        return self._selectors[scores_path]

    def run(self, name, spec, questions, overwrite=False, verbose=True):
        out_dir = os.path.join(self.runs_dir, name)
        pred_path = os.path.join(out_dir, "predictions.jsonl")
        os.makedirs(out_dir, exist_ok=True)
        settings = pipeline.check_settings(spec["settings"])
        cfg = {"name": name, "settings": settings, "scores": spec["scores"], "llama": generator.LLAMA_MODEL,
               "answer_prompt": generator.ANSWER_PROMPT, "summary_prompt": generator.SUMMARY_PROMPT}
        retrieval.write_json(cfg, os.path.join(out_dir, "config.json"))

        rows = [] if overwrite or not os.path.exists(pred_path) else retrieval.read_jsonl(pred_path)
        wanted = {q["_id"] for q in questions}
        rows = [r for r in rows if r["id"] in wanted]
        done = {r["id"] for r in rows}
        todo = [q for q in questions if q["_id"] not in done]
        if verbose:
            print(f"[{name}] {len(questions)} questions, {len(todo)} to answer")
        if todo:
            needs_scores = settings["mode"] != "closed_book"
            selector = self.selector(spec["scores"]) if needs_scores else pipeline.ReplaySelector([])
            pipe = pipeline.Pipeline(selector, self.llm, self.cache, **settings)
            t_all = time.perf_counter()
            with open(pred_path, "w" if not rows else "a") as f:
                for start in range(0, len(todo), self.batch_size):
                    chunk = todo[start:start + self.batch_size]
                    new = [finish_row(res, q) for res, q in zip(pipe.answer_batch(chunk), chunk)]
                    f.writelines(json.dumps(r) + "\n" for r in new)
                    f.flush()
                    rows.extend(new)
                    if verbose:
                        n = min(start + self.batch_size, len(todo))
                        print(f"  {n}/{len(todo)}  {(time.perf_counter() - t_all) / n:.2f}s per question")
        metrics = compute_metrics(rows)
        metrics.update(name=name, scores=os.path.basename(spec["scores"]), **settings)
        retrieval.write_json(metrics, os.path.join(out_dir, "metrics.json"))
        if verbose:
            print(f"[{name}] EM {metrics['em']:.2f}  F1 {metrics['f1']:.2f}  gold_in_context "
                  f"{metrics['gold_in_context']:.2f}  fallback {metrics['fallback_rate']:.1f}%")
        return metrics


# Tables

TABLE_COLUMNS = ["name", "mode", "scores", "threshold", "cap", "keep", "fallback", "summarize", "drop_irrelevant", "n",
                 "em", "f1", "em_bridge", "f1_bridge", "em_comparison", "f1_comparison", "gold_in_context",
                 "fallback_rate", "closed_book_rate", "avg_docs_scored", "avg_contexts", "score_s", "summarize_s",
                 "generate_s", "pct_correct", "pct_partial", "pct_retrieval_miss", "pct_reader_error"]


def collect_metrics(runs_dir=RUNS_DIR):
    rows = []
    if not os.path.isdir(runs_dir):
        return rows
    for name in sorted(os.listdir(runs_dir)):
        path = os.path.join(runs_dir, name, "metrics.json")
        if os.path.exists(path):
            rows.append(retrieval.read_json(path))
    return rows


def _cell(value):
    if isinstance(value, float):
        return f"{value:.4g}" if abs(value) < 1 else f"{value:.2f}"
    return "" if value is None else str(value)


def write_table(rows, path=RESULTS_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(TABLE_COLUMNS) + "\n")
        for r in rows:
            f.write(",".join(_cell(r.get(c)) for c in TABLE_COLUMNS) + "\n")


def format_table(rows):
    lines = [f"{'run':<28} {'n':>5} {'EM':>6} {'F1':>6} {'gold_ctx':>8} {'fallbk%':>7} {'docs':>5} {'ctx':>4}"]
    for r in sorted(rows, key=lambda r: -r["f1"]):
        lines.append(f"{r['name']:<28} {r['n']:>5} {r['em']:>6.2f} {r['f1']:>6.2f} {r['gold_in_context']:>8.2f} "
                     f"{r['fallback_rate']:>7.1f} {r['avg_docs_scored']:>5.2f} {r['avg_contexts']:>4.1f}")
    return "\n".join(lines)


def pick_best(rows, prefix="sweep_", min_n=None):
    """Best run by F1 (EM breaks ties) among names starting with `prefix`."""
    cands = [r for r in rows if r["name"].startswith(prefix) and "_limit" not in r["name"]]
    if not cands:
        raise SystemExit(f"No finished full runs named {prefix}*. Run the sweep group first.")
    min_n = max(r["n"] for r in cands) if min_n is None else min_n
    cands = [r for r in cands if r["n"] >= min_n]
    best = max(cands, key=lambda r: (r["f1"], r["em"]))
    return best, {"threshold": best["threshold"], "cap": best["cap"], "keep": best["keep"]}


# Latency

def measure_latency(questions, pipe_settings, n=100, use_bm25_cache=False, backend=generator.LLAMA_BACKEND,
                    path=LATENCY_PATH):
    """Answer `n` questions one at a time with live retrieval and scoring, for each named setting.

    `pipe_settings` maps a label to Pipeline settings. Summaries are not cached here, so the timings are real.
    """
    cache_path = retrieval.CACHE_PATH if use_bm25_cache else None
    base = pipeline.build_live(cache_path=cache_path, backend=backend, summary_cache_path=None)
    rows = []
    for label, settings in pipe_settings.items():
        pipe = base.with_settings(**settings)
        pipe.cache = pipeline.SummaryCache(path=None)
        pipe.answer(questions[-1]["question"])  # warm-up on a question outside the timed set
        if hasattr(pipe.selector.retriever, "_memo"):
            pipe.selector.retriever._memo.clear()  # otherwise later settings reuse earlier BM25 results (0 s)
        for q in questions[:n]:
            res = pipe.answer(q["question"], qid=q["_id"])
            rows.append({"setting": label, "id": q["_id"], "n_scored": res["n_scored"], **res["timing"],
                         **score_prediction(res["answer"], q.get("answer", ""))})
        sub = [r for r in rows if r["setting"] == label]
        print(f"[{label}] mean {statistics.mean(r['total_s'] for r in sub):.3f}s  "
              f"median {statistics.median(r['total_s'] for r in sub):.3f}s  "
              f"docs {statistics.mean(r['n_scored'] for r in sub):.2f}")
    cols = ["setting", "id", "n_scored", "retrieve_s", "score_s", "summarize_s", "generate_s", "total_s", "em", "f1"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(_cell(r.get(c)) for c in cols) + "\n")
    return rows


# Test predictions

def predict_test(runner, spec, questions, path=TEST_PRED_PATH, scores_path=TEST_SCORES_PATH):
    """Answer the test split with the given experiment's settings, replaying monoT5 scores saved for test."""
    name = "test_" + os.path.splitext(os.path.basename(path))[0]
    runner.run(name, {**spec, "scores": scores_path}, questions)
    rows = retrieval.read_jsonl(os.path.join(runner.runs_dir, name, "predictions.jsonl"))
    retrieval.write_json({"answer": {r["id"]: r["answer"] for r in rows}, "sp": {}}, path)
    print(f"Wrote {len(rows)} answers to {path}")


# Command line

def _parse_overrides(pairs):
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def _names(args, exps):
    names = list(args.exp or [])
    for g in args.group or []:
        names += GROUPS[g]
    missing = [n for n in names if n not in exps]
    if missing:
        raise SystemExit(f"Unknown experiments: {missing}. See `python evaluate.py list`.")
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["list", "run", "best", "table", "latency", "test"])
    ap.add_argument("--exp", nargs="*", help="experiment names")
    ap.add_argument("--group", nargs="*", choices=sorted(GROUPS))
    ap.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override settings for every run, e.g. keep=3")
    ap.add_argument("--data", default=retrieval.DATA_PATH)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n", type=int, default=100, help="questions for the latency test")
    ap.add_argument("--use-bm25-cache", action="store_true", help="latency: serve BM25 from the dev cache")
    ap.add_argument("--backend", default=generator.LLAMA_BACKEND, choices=["hf", "vllm"])
    ap.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)
    ap.add_argument("--test-scores", default=TEST_SCORES_PATH, help="test: saved monoT5 scores for the test split")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    exps = experiments()
    overrides = _parse_overrides(args.set)

    if args.cmd == "list":
        for g, names in GROUPS.items():
            print(f"{g}: {' '.join(names)}")
        print("\nBest setting used by ablations/finetuned:", load_best())
    elif args.cmd == "run":
        questions = retrieval.read_jsonl(args.data, args.limit)
        runner = Runner(backend=args.backend, batch_size=args.batch_size)
        for name in _names(args, exps):
            spec = {**exps[name], "settings": {**exps[name]["settings"], **overrides}}
            suffix = "".join(f"_{k}{v}" for k, v in sorted(overrides.items()))  # never mix settings in one folder
            runner.run(name + suffix + ("" if not args.limit else f"_limit{args.limit}"), spec, questions,
                       args.overwrite)
        print(format_table(collect_metrics()))
    elif args.cmd == "best":
        rows = collect_metrics()
        best, settings = pick_best(rows)
        retrieval.write_json({"from_run": best["name"], "em": best["em"], "f1": best["f1"], "settings": settings},
                             BEST_PATH)
        print(f"Best sweep run {best['name']}: EM {best['em']:.2f} F1 {best['f1']:.2f} -> {settings}")
    elif args.cmd == "table":
        rows = collect_metrics()
        write_table(rows)
        print(format_table(rows))
        print(f"\nWrote {RESULTS_PATH}")
    elif args.cmd == "latency":
        questions = retrieval.read_jsonl(args.data)
        best = load_best()
        # The best setting may already be "no early stop" (threshold 1.01), so time a real early-stopping
        # threshold alongside it.
        settings = {"no_early_stop": {**best, "threshold": 1.01, "summarize": False, **overrides},
                    "adaptive_t0.98": {**best, "threshold": 0.98, "summarize": False, **overrides},
                    "no_early_stop_sum": {**best, "threshold": 1.01, "summarize": True, **overrides}}
        measure_latency(questions, settings, args.n, args.use_bm25_cache, args.backend)
    elif args.cmd == "test":
        name = (args.exp or ["final"])[0]
        if name == "final":
            spec = {"settings": {**load_best(), "fallback": "top", "summarize": False, **overrides}}
        else:
            spec = {**exps[name], "settings": {**exps[name]["settings"], **overrides}}
        questions = retrieval.read_jsonl(retrieval.TEST_PATH, args.limit)
        predict_test(Runner(backend=args.backend, batch_size=args.batch_size), spec, questions,
                     scores_path=args.test_scores)


if __name__ == "__main__":
    main()
