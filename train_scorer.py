"""Fine-tune the T5 relevance scorer on HotpotQA train questions.

train.jsonl has questions and answers but no labelled passages, so labels come from distant supervision:
a BM25 passage is positive if the normalized gold answer appears in it, negative otherwise. Yes/no answers
can't be matched, so those questions are left out. 10% of train is held out to choose hyperparameters,
so dev stays untouched until the final comparison.

    python train_scorer.py data                    # BM25 top 30 for train questions -> candidate pools (Java)
    python train_scorer.py base                    # off-the-shelf monoT5 on the held-out split (the baseline)
    python train_scorer.py train --lr 3e-5 --epochs 2 --neg-ratio 3
    python train_scorer.py sweep                   # grid over lr x neg-ratio (x init) -> scorer_sweep.csv
    python train_scorer.py rescore --model outputs/scorer_runs/<run>/best   # dev -> t5_scores_ft.jsonl, sweep_ft.csv

Training targets the first decoded token ("true" or "false"), the same token `T5Scorer` reads at inference.
"""
import argparse
import itertools
import json
import os
import random
import time

import retrieval
from utils import config

TRAIN_CACHE_PATH = os.path.join(config.DIR_OUTPUTS, "bm25_train_top30.json")
POOLS_PATH = os.path.join(config.DIR_OUTPUTS, "t5_train_pools.jsonl")
VAL_PATH = os.path.join(config.DIR_OUTPUTS, "t5_val.jsonl")
RUNS_DIR = os.path.join(config.DIR_OUTPUTS, "scorer_runs")
SWEEP_PATH = os.path.join(config.DIR_OUTPUTS, "scorer_sweep.csv")
FT_SCORES_PATH = os.path.join(config.DIR_OUTPUTS, "t5_scores_ft.jsonl")
FT_SWEEP_PATH = os.path.join(config.DIR_OUTPUTS, "sweep_ft.csv")

DATA_DEPTH = 30  # BM25 hits per train question to draw negatives from
VAL_DEPTH = retrieval.MAX_DOCS  # held-out questions are ranked over the same 15 passages as dev
VAL_FRACTION = 0.1
MAX_POS = 2  # positives kept per question
SEED = 0

DEFAULTS = {"init": retrieval.T5_MODEL, "lr": 3e-5, "epochs": 2, "neg_ratio": 3, "batch_size": 8, "grad_accum": 2,
            "warmup": 0.1, "weight_decay": 0.01, "max_length": 512, "grad_checkpointing": False, "seed": SEED}
SWEEP_GRID = {"lr": [1e-5, 3e-5], "neg_ratio": [1, 3]}


# Data

def split_questions(questions, val_fraction=VAL_FRACTION, seed=SEED):
    """Deterministic train/held-out split by shuffled question id."""
    ids = sorted(q["_id"] for q in questions)
    random.Random(seed).shuffle(ids)
    val_ids = set(ids[:int(len(ids) * val_fraction)])
    return [q for q in questions if q["_id"] not in val_ids], [q for q in questions if q["_id"] in val_ids]


def label_hits(gold, hits):
    """Copy of `hits` with label 1 where the gold answer string appears in the passage."""
    return [{**h, "label": int(bool(retrieval.gold_in(gold, [h])))} for h in hits]


def build_pools(questions, cache, max_pos=MAX_POS):
    """One pool per question with at least one positive: positives (best BM25 rank first) and negatives."""
    pools, skipped = [], {"yes_no": 0, "no_positive": 0}
    for q in questions:
        if not retrieval.gradable(q["answer"]):
            skipped["yes_no"] += 1
            continue
        hits = label_hits(q["answer"], cache.get(q["_id"], []))
        pos = [h for h in hits if h["label"]][:max_pos]
        neg = [h for h in hits if not h["label"]]
        if not pos:
            skipped["no_positive"] += 1
            continue
        keep = ("docid", "title", "text", "rank")
        pools.append({"id": q["_id"], "question": q["question"], "gold": q["answer"],
                      "pos": [{k: h[k] for k in keep} for h in pos], "neg": [{k: h[k] for k in keep} for h in neg]})
    return pools, skipped


def build_val(questions, cache, depth=VAL_DEPTH):
    """Held-out questions with their top `depth` hits labelled, for ranking evaluation."""
    rows = []
    for q in questions:
        if retrieval.gradable(q["answer"]):
            rows.append({"id": q["_id"], "question": q["question"], "gold": q["answer"],
                         "hits": label_hits(q["answer"], cache.get(q["_id"], [])[:depth])})
    return rows


def sample_examples(pools, neg_ratio, rng):
    """(question, passage, label) examples: every positive plus `neg_ratio` negatives per positive.

    Half the negatives come from the highest-ranked BM25 negatives (hard), the rest at random from the pool.
    """
    out = []
    for p in pools:
        n_neg = min(len(p["neg"]), neg_ratio * len(p["pos"]))
        n_hard = n_neg // 2
        hard = p["neg"][:n_hard]
        rest = p["neg"][n_hard:]
        negs = hard + rng.sample(rest, min(len(rest), n_neg - n_hard))
        out += [(p["question"], h, 1) for h in p["pos"]] + [(p["question"], h, 0) for h in negs]
    rng.shuffle(out)
    return out


# Evaluation

def ranking_metrics(rows, ks=(1, 2, 5)):
    """Over held-out rows whose hits carry "t5" and "label": answer recall at k after reranking, and MRR."""
    out, rr = {f"gold@{k}": 0.0 for k in ks}, 0.0
    usable = [r for r in rows if r["hits"]]
    for r in usable:
        ranked = sorted(r["hits"], key=lambda h: (-h["t5"], h["rank"]))
        for k in ks:
            out[f"gold@{k}"] += any(h["label"] for h in ranked[:k])
        first = next((i for i, h in enumerate(ranked) if h["label"]), None)
        rr += 0.0 if first is None else 1.0 / (first + 1)
    n = max(len(usable), 1)
    out = {k: 100 * v / n for k, v in out.items()}
    out["mrr"] = rr / n
    out["oracle@15"] = 100 * sum(any(h["label"] for h in r["hits"]) for r in usable) / n
    return out


def evaluate_scorer(scorer, val_rows):
    rows = []
    for r in val_rows:
        scores = scorer.score_batch(r["question"], r["hits"]) if r["hits"] else []
        rows.append({**r, "hits": [{**h, "t5": float(s)} for h, s in zip(r["hits"], scores)]})
    return ranking_metrics(rows)


def scorer_from_model(model, tok, batch_size=retrieval.T5_BATCH_SIZE, max_length=512):
    """A retrieval.T5Scorer around an already-loaded model, so validation uses the exact inference code."""
    import torch

    s = retrieval.T5Scorer.__new__(retrieval.T5Scorer)
    s.torch, s.device, s.tok, s.model = torch, next(model.parameters()).device, tok, model
    s.true_id, s.false_id = tok.convert_tokens_to_ids("▁true"), tok.convert_tokens_to_ids("▁false")
    s.batch_size, s.max_length = batch_size, max_length
    return s


# Training

def load_tokenizer(name):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(name)
    except (OSError, ValueError):
        return AutoTokenizer.from_pretrained("t5-large")


def run_name(hp):
    init = os.path.basename(hp["init"].rstrip("/")).replace("monot5-large-msmarco", "monot5")
    return f"{init}_lr{hp['lr']:g}_ep{hp['epochs']}_neg{hp['neg_ratio']}"


def train(hp, pools, val_rows, runs_dir=RUNS_DIR, save=True):
    """Fine-tune from hp["init"]; validate after each epoch and keep the checkpoint with the best gold@2."""
    import torch
    from transformers import T5ForConditionalGeneration, get_linear_schedule_with_warmup

    hp = {**DEFAULTS, **hp}
    name = run_name(hp)
    out_dir = os.path.join(runs_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    retrieval.write_json(hp, os.path.join(out_dir, "config.json"))
    torch.manual_seed(hp["seed"])
    rng = random.Random(hp["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(hp["init"])
    model = T5ForConditionalGeneration.from_pretrained(hp["init"], torch_dtype=torch.float32).to(device)
    if hp["grad_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    true_id, false_id = tok.convert_tokens_to_ids("▁true"), tok.convert_tokens_to_ids("▁false")
    n_per_epoch = len(sample_examples(pools, hp["neg_ratio"], random.Random(0)))
    steps = (n_per_epoch // (hp["batch_size"] * hp["grad_accum"]) + 1) * hp["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, int(hp["warmup"] * steps), steps)
    use_bf16 = device == "cuda" and retrieval.native_bf16(torch)  # fp32 on V100 (fp16 overflows in T5)

    log, best = [], None
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        examples = sample_examples(pools, hp["neg_ratio"], rng)
        t0, total, n_batches = time.perf_counter(), 0.0, 0
        for b, start in enumerate(range(0, len(examples), hp["batch_size"])):
            chunk = examples[start:start + hp["batch_size"]]
            enc = tok([retrieval.T5Scorer.format_input(q, p) for q, p, _ in chunk], padding=True, truncation=True,
                      max_length=hp["max_length"], return_tensors="pt").to(device)
            dec = torch.full((len(chunk), 1), model.config.decoder_start_token_id, dtype=torch.long, device=device)
            y = torch.tensor([true_id if lab else false_id for _, _, lab in chunk], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(**enc, decoder_input_ids=dec).logits[:, 0, :]
            loss = torch.nn.functional.cross_entropy(logits.float(), y)
            (loss / hp["grad_accum"]).backward()
            total += loss.item()
            n_batches += 1
            if (b + 1) % hp["grad_accum"] == 0 or start + hp["batch_size"] >= len(examples):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if n_batches % 200 == 0:
                print(f"  epoch {epoch} batch {n_batches}  loss {total / n_batches:.4f}  "
                      f"{(time.perf_counter() - t0) / n_batches:.2f}s/batch")
        model.eval()
        metrics = evaluate_scorer(scorer_from_model(model, tok, max_length=hp["max_length"]), val_rows)
        row = {"epoch": epoch, "train_loss": total / max(n_batches, 1), "train_examples": len(examples),
               "epoch_s": time.perf_counter() - t0, **metrics}
        log.append(row)
        with open(os.path.join(out_dir, "log.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{name}] epoch {epoch}: loss {row['train_loss']:.4f}  gold@1 {metrics['gold@1']:.2f}  "
              f"gold@2 {metrics['gold@2']:.2f}  mrr {metrics['mrr']:.4f}")
        if best is None or metrics["gold@2"] > best["gold@2"]:
            best = row
            if save:
                model.save_pretrained(os.path.join(out_dir, "best"))
                tok.save_pretrained(os.path.join(out_dir, "best"))
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    result = {"name": name, **{k: hp[k] for k in ("init", "lr", "epochs", "neg_ratio", "batch_size", "grad_accum")},
              "best_epoch": best["epoch"], **{k: v for k, v in best.items() if k.startswith("gold@") or k == "mrr"},
              "checkpoint": os.path.join(out_dir, "best")}
    retrieval.write_json(result, os.path.join(out_dir, "result.json"))
    return result


SWEEP_COLUMNS = ["name", "init", "lr", "epochs", "neg_ratio", "batch_size", "grad_accum", "best_epoch", "gold@1",
                 "gold@2", "gold@5", "mrr", "checkpoint"]


def write_sweep(rows, path=SWEEP_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(SWEEP_COLUMNS) + "\n")
        for r in rows:
            f.write(",".join(f"{r.get(c):.4g}" if isinstance(r.get(c), float) else str(r.get(c, ""))
                             for c in SWEEP_COLUMNS) + "\n")


def collect_runs(runs_dir=RUNS_DIR):
    rows = []
    if os.path.isdir(runs_dir):
        for name in sorted(os.listdir(runs_dir)):
            path = os.path.join(runs_dir, name, "result.json")
            if os.path.exists(path):
                rows.append(retrieval.read_json(path))
    return rows


# Command line

def load_data():
    if not (os.path.exists(POOLS_PATH) and os.path.exists(VAL_PATH)):
        raise SystemExit(f"Missing {POOLS_PATH}. Run `python train_scorer.py data` first.")
    return retrieval.read_jsonl(POOLS_PATH), retrieval.read_jsonl(VAL_PATH)


def cmd_data(args):
    questions = retrieval.read_jsonl(args.data, args.limit)
    train_q, val_q = split_questions(questions, args.val_fraction)
    cache = retrieval.build_cache(questions, index=args.index, cache_path=args.cache, depth=args.depth,
                                  threads=args.threads)
    pools, skipped = build_pools(train_q, cache)
    val = build_val(val_q, cache)
    for path, rows in ((POOLS_PATH, pools), (VAL_PATH, val)):
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    n_pos = sum(len(p["pos"]) for p in pools)
    print(f"train: {len(train_q)} questions -> {len(pools)} pools, {n_pos} positives "
          f"(skipped {skipped['yes_no']} yes/no, {skipped['no_positive']} with no positive in top {args.depth})")
    print(f"held out: {len(val_q)} questions -> {len(val)} gradable; "
          f"oracle@15 {ranking_metrics([{**r, 'hits': [{**h, 't5': 0.0} for h in r['hits']]} for r in val])['oracle@15']:.2f}")


def cmd_base(args):
    _, val = load_data()
    scorer = retrieval.T5Scorer(args.init)
    m = evaluate_scorer(scorer, val)
    bm25 = ranking_metrics([{**r, "hits": [{**h, "t5": -h["rank"]} for h in r["hits"]]} for r in val])
    result = {"name": "bm25_order", "init": "", "lr": 0, "epochs": 0, "neg_ratio": 0, **bm25}
    base = {"name": "base_" + os.path.basename(args.init.rstrip("/")), "init": args.init, "lr": 0, "epochs": 0,
            "neg_ratio": 0, **m}
    for r in (result, base):
        os.makedirs(os.path.join(RUNS_DIR, r["name"]), exist_ok=True)
        retrieval.write_json(r, os.path.join(RUNS_DIR, r["name"], "result.json"))
        print(f"{r['name']}: gold@1 {r['gold@1']:.2f}  gold@2 {r['gold@2']:.2f}  gold@5 {r['gold@5']:.2f}  "
              f"mrr {r['mrr']:.4f}")


def _hp(args):
    return {k: getattr(args, k) for k in DEFAULTS}


def cmd_train(args):
    pools, val = load_data()
    print(train(_hp(args), pools, val))
    write_sweep(collect_runs())


def cmd_sweep(args):
    pools, val = load_data()
    done = {r["name"] for r in collect_runs()}
    inits = args.inits or [args.init]
    for init, lr, neg in itertools.product(inits, SWEEP_GRID["lr"], SWEEP_GRID["neg_ratio"]):
        hp = {**_hp(args), "init": init, "lr": lr, "neg_ratio": neg}
        if run_name({**DEFAULTS, **hp}) in done and not args.overwrite:
            print("skip", run_name({**DEFAULTS, **hp}))
            continue
        train(hp, pools, val)
    rows = collect_runs()
    write_sweep(rows)
    trained = [r for r in rows if r.get("epochs")]
    if trained:
        best = max(trained, key=lambda r: r["gold@2"])
        print(f"Best: {best['name']} gold@2 {best['gold@2']:.2f} -> {best['checkpoint']}")


def cmd_rescore(args):
    dev = retrieval.read_jsonl(retrieval.DATA_PATH, args.limit)
    retriever = retrieval.Retriever(cache_path=retrieval.CACHE_PATH)
    scorer = retrieval.T5Scorer(args.model)
    retrieval.score_questions(dev, retriever, scorer, scores_path=args.scores, overwrite=args.overwrite)
    rows = retrieval.sweep(retrieval.read_jsonl(args.scores))
    retrieval.write_sweep_csv(rows, args.sweep_out)
    print(retrieval.format_sweep(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["data", "base", "train", "sweep", "rescore"])
    ap.add_argument("--data", default=retrieval.TRAIN_PATH)
    ap.add_argument("--index", default=retrieval.INDEX)
    ap.add_argument("--cache", default=TRAIN_CACHE_PATH)
    ap.add_argument("--depth", type=int, default=DATA_DEPTH)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--init", default=DEFAULTS["init"], help="checkpoint to start from (monoT5 or google-t5/t5-large)")
    ap.add_argument("--inits", nargs="*", help="sweep: several starting checkpoints")
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--neg-ratio", type=int, default=DEFAULTS["neg_ratio"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum"])
    ap.add_argument("--warmup", type=float, default=DEFAULTS["warmup"])
    ap.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    ap.add_argument("--max-length", type=int, default=DEFAULTS["max_length"])
    ap.add_argument("--grad-checkpointing", action="store_true", help="lower memory (use on an A30)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--model", help="rescore: fine-tuned checkpoint directory")
    ap.add_argument("--scores", default=FT_SCORES_PATH)
    ap.add_argument("--sweep-out", default=FT_SWEEP_PATH)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.cmd == "rescore" and not args.model:
        raise SystemExit("rescore needs --model <checkpoint directory>")
    {"data": cmd_data, "base": cmd_base, "train": cmd_train, "sweep": cmd_sweep, "rescore": cmd_rescore}[args.cmd](args)


if __name__ == "__main__":
    main()
