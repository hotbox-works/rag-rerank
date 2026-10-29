"""Figures and tables for the report and slides, built from whatever outputs exist so far.

    python report.py            # -> outputs/report/{results.md, *.png}

Every section is skipped quietly if its inputs aren't there yet, so this can be rerun after each step:
retrieval figures need t5_scores.jsonl and sweep.csv; end-to-end figures need evaluate.py runs;
the scorer figure needs train_scorer.py runs; the latency figure needs latency.csv.
"""
import argparse
import csv
import os

import retrieval
from utils import config

REPORT_DIR = os.path.join(config.DIR_OUTPUTS, "report")
FT_SCORES_PATH = os.path.join(config.DIR_OUTPUTS, "t5_scores_ft.jsonl")
FT_SWEEP_PATH = os.path.join(config.DIR_OUTPUTS, "sweep_ft.csv")

# Parts 1 to 3 (README), Llama 3.1 8B Instruct on the same 1,110 dev questions.
BASELINES = [("Naive prompting", 5.23, 12.37), ("Instruct prompting (V2)", 24.59, 32.20),
             ("Chain-of-thought", 23.78, 32.02), ("Random few-shot ICL", 23.06, 30.67),
             ("Retrieval-based ICL", 25.32, 34.01), ("Wikipedia RAG (BM25, n=8)", 30.90, 41.68)]

# Runs shown in the headline chart, with readable labels. Missing runs are skipped.
KEY_RUNS = [("closed_book", "Closed book (this prompt)"), ("bm25_k2_raw", "BM25 top-2"),
            ("bm25_k8_raw", "BM25 top-8"), ("best_nostop_raw", "monoT5 rerank, no early stop"),
            ("best_raw", "Adaptive (best setting)"), ("best_sum", "Adaptive + summaries"),
            ("ft_nostop_raw", "Fine-tuned T5, no early stop"), ("ft_th0.9_raw", "Adaptive, fine-tuned T5 (t=0.9)"),
            ("ft_best_sum", "Adaptive, fine-tuned T5 + summaries")]

# Reference categorical palette (light mode), slots in fixed order; ink and surface tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
SEQ = "Blues"


# Helpers

def plt_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK_2,
        "ytick.color": INK_2, "font.size": 10, "legend.frameon": False, "lines.linewidth": 2,
        "axes.axisbelow": True})
    return plt


def save(fig, name, out_dir):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if v not in ("", None) and k not in ("name", "setting", "id", "mode", "fallback",
                                                                       "scores", "init", "checkpoint") else v
            except ValueError:
                pass
    return rows


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(f"{c:.2f}" if isinstance(c, float) else str(c) for c in r) + " |")
    return "\n".join(out)


def recall_at_k(records, order, ks=range(1, 16)):
    """Answer recall at each k over gradable questions, with hits ordered by BM25 rank or monoT5 score."""
    graded = [r for r in records if retrieval.gradable(r["gold"])]
    key = (lambda h: h["rank"]) if order == "bm25" else (lambda h: (-h["t5"], h["rank"]))
    ranked = [(r["gold"], sorted(r["hits"], key=key)) for r in graded]
    return {k: 100 * sum(bool(retrieval.gold_in(g, hs[:k])) for g, hs in ranked) / max(len(ranked), 1) for k in ks}


def load_runs():
    import evaluate

    return {r["name"]: r for r in evaluate.collect_metrics()}


# Sections

def section_retrieval(out_dir, plt):
    if not os.path.exists(retrieval.SCORES_PATH):
        return ""
    records = retrieval.read_jsonl(retrieval.SCORES_PATH)
    curves = {"BM25 order": recall_at_k(records, "bm25"), "monoT5 rerank": recall_at_k(records, "t5")}
    if os.path.exists(FT_SCORES_PATH):
        curves["Fine-tuned T5 rerank"] = recall_at_k(retrieval.read_jsonl(FT_SCORES_PATH), "t5")
    fig, ax = plt.subplots(figsize=(7, 4))
    for (label, c), color in zip(curves.items(), SERIES):
        ks = list(c)
        ax.plot(ks, [c[k] for k in ks], color=color, label=label, marker="o", markersize=4)
        ax.annotate(f"{c[2]:.1f}", (2, c[2]), textcoords="offset points", xytext=(6, -10 if "BM25" in label else 4),
                    color=INK_2, fontsize=8)
    ax.set_xlabel("Passages kept (k)")
    ax.set_ylabel("Gold answer in kept passages (%)")
    ax.set_title("Answer recall by passages kept, dev (yes/no excluded)")
    ax.set_xticks([1, 2, 3, 5, 8, 10, 15])
    ax.legend(loc="lower right")
    save(fig, "retrieval_recall.png", out_dir)
    ks = [1, 2, 3, 5, 8, 10, 15]
    table = md_table(["Order"] + [f"@{k}" for k in ks], [[n] + [c[k] for k in ks] for n, c in curves.items()])
    t5 = [r["score_s"] for r in records]
    return (f"## Retrieval and reranking\n\n![](retrieval_recall.png)\n\n{table}\n\n"
            f"monoT5 scores 15 passages in {sum(t5) / len(t5):.3f}s per question on average "
            f"({len(records)} dev questions, {sum(not retrieval.gradable(r['gold']) for r in records)} yes/no "
            f"excluded from recall).\n")


def section_sweep(out_dir, plt):
    parts = []
    for path, label, fname in ((retrieval.SWEEP_PATH, "monoT5", "sweep_tradeoff.png"),
                               (FT_SWEEP_PATH, "fine-tuned T5", "sweep_tradeoff_ft.png")):
        rows = read_csv(path)
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        caps = sorted({r["cap"] for r in rows})
        for cap, color in zip(caps, SERIES):
            sub = sorted([r for r in rows if r["cap"] == cap], key=lambda r: r["avg_docs_scored"])
            ax.plot([r["avg_docs_scored"] for r in sub], [r["gold_in_selected"] for r in sub], color=color,
                    marker="o", markersize=5, label=f"cap {int(cap)}")
            if cap == max(caps):
                for r in sub:
                    if r["threshold"] in (0.5, 0.9, 0.98, 0.995, 1.01):
                        ax.annotate(f"t={r['threshold']:g}", (r["avg_docs_scored"], r["gold_in_selected"]),
                                    textcoords="offset points", xytext=(4, -12), fontsize=7, color=INK_2)
        base = rows[0]["gold_in_bm25_topkeep"]
        ax.axhline(base, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate(f"BM25 top-2: {base:.1f}", (ax.get_xlim()[0], base), textcoords="offset points",
                    xytext=(4, 4), fontsize=8, color=INK_2)
        ax.set_xlabel("Passages scored per question (avg)")
        ax.set_ylabel("Gold answer in selected 2 (%)")
        ax.set_title(f"Early stopping trade-off ({label})")
        ax.legend(loc="center right")
        save(fig, fname, out_dir)
        cap15 = [r for r in rows if r["cap"] == max(caps)]
        table = md_table(["threshold", "fallback %", "docs scored", "stopped early %", "gold in selected %"],
                         [[f"{r['threshold']:g}", r["fallback_rate"], r["avg_docs_scored"], r["stopped_early_rate"],
                           r["gold_in_selected"]] for r in cap15])
        parts.append(f"### Threshold sweep, {label} (cap {int(max(caps))}, keep 2)\n\n![]({fname})\n\n{table}\n")
    return ("## Adaptive early stopping (offline replay)\n\n" + "\n".join(parts)) if parts else ""


def section_e2e(out_dir, plt, runs):
    key = [(n, label) for n, label in KEY_RUNS if n in runs]
    if not key:
        return ""
    bars = [(label, em, f1, "Parts 1-3") for label, em, f1 in BASELINES[1:]]
    bars += [(label, runs[n]["em"], runs[n]["f1"], "This project") for n, label in key]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(bars) + 1.2))
    ys = list(range(len(bars)))[::-1]
    h = 0.38
    ax.barh([y + h / 2 for y in ys], [b[2] for b in bars], height=h, color=SERIES[0], label="F1")
    ax.barh([y - h / 2 for y in ys], [b[1] for b in bars], height=h, color=SERIES[1], label="EM")
    for y, b in zip(ys, bars):
        ax.text(b[2] + 0.5, y + h / 2, f"{b[2]:.1f}", va="center", fontsize=7, color=INK_2)
        ax.text(b[1] + 0.5, y - h / 2, f"{b[1]:.1f}", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(ys)
    ax.set_yticklabels([b[0] for b in bars])
    for tick, b in zip(ax.get_yticklabels(), bars):
        tick.set_color(INK if b[3] == "This project" else MUTED)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Score (%)")
    ax.set_title("HotpotQA dev, Llama 3.1 8B Instruct (grey labels: Parts 1-3)")
    ax.legend(loc="lower right")
    save(fig, "e2e_results.png", out_dir)

    rows = [[label, "Parts 1-3", em, f1, "", ""] for label, em, f1 in BASELINES]
    rows += [[label, n, runs[n]["em"], runs[n]["f1"], runs[n]["gold_in_context"], runs[n]["fallback_rate"]]
             for n, label in key]
    table = md_table(["System", "Run", "EM", "F1", "Gold in context %", "Fallback %"], rows)
    type_rows = [[label, runs[n].get("em_bridge", 0.0), runs[n].get("f1_bridge", 0.0),
                  runs[n].get("em_comparison", 0.0), runs[n].get("f1_comparison", 0.0)] for n, label in key]
    types = md_table(["System", "EM bridge", "F1 bridge", "EM comparison", "F1 comparison"], type_rows)

    # Error analysis, stacked
    cats = [("pct_correct", "Correct (EM)"), ("pct_partial", "Partial (F1 > 0)"),
            ("pct_reader_error", "Reader error (answer in context)"), ("pct_retrieval_miss", "Answer not in context")]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(key) + 1.4))
    ys = list(range(len(key)))[::-1]
    left = [0.0] * len(key)
    for (col, label), color in zip(cats, SERIES):
        vals = [runs[n].get(col, 0.0) for n, _ in key]
        ax.barh(ys, vals, left=left, color=color, label=label, height=0.6, edgecolor=SURFACE, linewidth=2)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(ys)
    ax.set_yticklabels([label for _, label in key])
    ax.set_xlabel("Questions (%)  (remainder: yes/no answered wrong)")
    ax.set_title("Where answers go wrong")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18 - 0.02 * len(key)), ncol=2, fontsize=8)
    save(fig, "error_analysis.png", out_dir)
    err = md_table(["System"] + [label for _, label in cats],
                   [[label] + [runs[n].get(c, 0.0) for c, _ in cats] for n, label in key])
    return (f"## End-to-end results\n\n![](e2e_results.png)\n\n{table}\n\n### By question type\n\n{types}\n\n"
            f"### Error analysis\n\n![](error_analysis.png)\n\n{err}\n\nRetrieval miss: wrong, and the gold "
            f"answer is not in any injected passage. Reader error: wrong although it is.\n")


def section_e2e_sweep(out_dir, plt, runs):
    import evaluate

    cells = {(th, k): runs[evaluate._sweep_name(th, k)] for th in evaluate.SWEEP_THRESHOLDS
             for k in evaluate.SWEEP_KEEPS if evaluate._sweep_name(th, k) in runs}
    if not cells:
        return ""
    ths, ks = evaluate.SWEEP_THRESHOLDS, evaluate.SWEEP_KEEPS
    import numpy as np

    grid = np.full((len(ths), len(ks)), np.nan)
    for (th, k), r in cells.items():
        grid[ths.index(th), ks.index(k)] = r["f1"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    im = ax.imshow(grid, cmap=SEQ, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(ks)), [str(k) for k in ks])
    ax.set_yticks(range(len(ths)), ["1.01 (no stop)" if t > 1 else f"{t:g}" for t in ths])
    ax.set_xlabel("Passages injected (keep)")
    ax.set_ylabel("Threshold")
    ax.set_title("F1 by threshold and keep (cap 15, raw passages)")
    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    for i in range(len(ths)):
        for j in range(len(ks)):
            if not np.isnan(grid[i, j]):
                dark = (grid[i, j] - vmin) / max(vmax - vmin, 1e-9) > 0.6
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color=SURFACE if dark else INK)
    fig.colorbar(im, ax=ax, label="F1")
    save(fig, "e2e_sweep_heatmap.png", out_dir)
    rows = [[f"{th:g}", k, r["em"], r["f1"], r["gold_in_context"], r["avg_docs_scored"], r["fallback_rate"]]
            for (th, k), r in sorted(cells.items())]
    table = md_table(["threshold", "keep", "EM", "F1", "gold in context %", "docs scored", "fallback %"], rows)
    best = ""
    if os.path.exists(evaluate.BEST_PATH):
        b = retrieval.read_json(evaluate.BEST_PATH)
        best = f"\nChosen setting: {b['settings']} (from {b['from_run']}, F1 {b['f1']:.2f}).\n"
    return f"## End-to-end sweep\n\n![](e2e_sweep_heatmap.png)\n\n{table}\n{best}"


def section_ablations(runs):
    groups = [("Fallback when nothing passes the threshold",
               [("best_fallback_top_raw", "Best-scored passages anyway"), ("best_fallback_bm25_raw", "BM25 top-k"),
                ("best_fallback_closed_book_raw", "No documents (closed book)")]),
              ("Summarizer", [("best_raw", "Raw passages"), ("best_sum", "Llama summaries"),
                              ("best_sum_drop", "Summaries, drop 'Not relevant'")]),
              ("Early stopping", [("best_raw", "Adaptive (best threshold)"), ("best_nostop_raw", "Score all 15")]),
              ("Scorer, no early stopping", [("best_nostop_raw", "monoT5 (MS MARCO)"),
                                             ("ft_nostop_raw", "Fine-tuned on HotpotQA train")]),
              ("Scorer, matched effort", [("best_raw", "monoT5, best threshold"),
                                          ("ft_th0.9_raw", "Fine-tuned, threshold 0.9")])]
    out = []
    for title, items in groups:
        rows = [[label, runs[n]["em"], runs[n]["f1"], runs[n]["gold_in_context"], runs[n]["avg_contexts"],
                 runs[n]["avg_docs_scored"]] for n, label in items if n in runs]
        if len(rows) > 1:
            out.append(f"### {title}\n\n" + md_table(["Variant", "EM", "F1", "Gold in context %", "Contexts",
                                                      "Docs scored"], rows) + "\n")
    return ("## Ablations\n\n" + "\n".join(out)) if out else ""


def section_scorer(out_dir, plt):
    import train_scorer

    rows = train_scorer.collect_runs()
    if not rows:
        return ""
    rows = sorted(rows, key=lambda r: r["gold@2"])
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(rows) + 1.2))
    colors = [MUTED if not r.get("epochs") else SERIES[0] for r in rows]
    ax.barh(range(len(rows)), [r["gold@2"] for r in rows], color=colors, height=0.6)
    for i, r in enumerate(rows):
        ax.text(r["gold@2"] + 0.3, i, f"{r['gold@2']:.1f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(rows)), [r["name"] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Gold answer in top 2 after reranking (%), held-out train questions")
    ax.set_title("T5 scorer fine-tuning (grey: not fine-tuned)")
    save(fig, "scorer_sweep.png", out_dir)
    table = md_table(["Run", "lr", "neg ratio", "best epoch", "gold@1", "gold@2", "gold@5", "MRR"],
                     [[r["name"], f"{r.get('lr', 0):g}", r.get("neg_ratio", ""), r.get("best_epoch", ""),
                       r["gold@1"], r["gold@2"], r["gold@5"], r["mrr"]] for r in rows[::-1]])
    return f"## Scorer fine-tuning\n\n![](scorer_sweep.png)\n\n{table}\n"


def section_latency(out_dir, plt):
    import evaluate

    rows = read_csv(evaluate.LATENCY_PATH)
    if not rows:
        return ""
    settings = list(dict.fromkeys(r["setting"] for r in rows))
    stages = [("retrieve_s", "BM25"), ("score_s", "monoT5"), ("summarize_s", "Summaries"), ("generate_s", "Answer")]
    means = {s: {k: sum(r[k] for r in rows if r["setting"] == s) / sum(r["setting"] == s for r in rows)
                 for k, _ in stages + [("total_s", ""), ("n_scored", "")]} for s in settings}
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(settings) + 1.4))
    left = [0.0] * len(settings)
    for (k, label), color in zip(stages, SERIES):
        vals = [means[s][k] for s in settings]
        ax.barh(range(len(settings)), vals, left=left, color=color, label=label, height=0.6, edgecolor=SURFACE,
                linewidth=2)
        left = [a + b for a, b in zip(left, vals)]
    for i, s in enumerate(settings):
        ax.text(left[i] + 0.02, i, f"{means[s]['total_s']:.2f}s", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(settings)), settings)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Seconds per question (mean, live, one at a time)")
    ax.set_title("Latency by stage")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=4, fontsize=8)
    save(fig, "latency.png", out_dir)
    table = md_table(["Setting", "BM25 s", "monoT5 s", "Summaries s", "Answer s", "Total s", "Docs scored"],
                     [[s] + [means[s][k] for k, _ in stages] + [means[s]["total_s"], means[s]["n_scored"]]
                      for s in settings])
    return f"## Latency\n\n![](latency.png)\n\n{table}\n"


def section_examples(runs, n=4):
    """Failure cases from the best adaptive run, for slides."""
    import evaluate

    name = next((x for x in ("best_raw", "best_nostop_raw", "ft_nostop_raw", "best_sum") if x in runs), None)
    if name is None:
        return ""
    rows = retrieval.read_jsonl(os.path.join(evaluate.RUNS_DIR, name, "predictions.jsonl"))
    out = [f"## Example cases ({name})\n"]
    for cat, title in (("correct", "Correct"), ("reader_error", "Reader errors"),
                       ("retrieval_miss", "Retrieval misses")):
        picked = [r for r in rows if evaluate.outcome(r) == cat][:n]
        if picked:
            out.append(f"### {title}\n")
            for r in picked:
                titles = ", ".join(c["title"] for c in r["contexts"]) or "none"
                out.append(f"- **Q:** {r['question']}  \n  **Gold:** {r['gold']}  **Predicted:** {r['answer']}  \n"
                           f"  Passages: {titles}")
            out.append("")
    return "\n".join(out)


def build(out_dir=REPORT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    plt = plt_setup()
    runs = load_runs()
    sections = [section_retrieval(out_dir, plt), section_sweep(out_dir, plt), section_scorer(out_dir, plt),
                section_e2e_sweep(out_dir, plt, runs), section_e2e(out_dir, plt, runs), section_ablations(runs),
                section_latency(out_dir, plt), section_examples(runs)]
    body = "\n".join(s for s in sections if s)
    path = os.path.join(out_dir, "results.md")
    with open(path, "w") as f:
        f.write("# Project 2 results\n\nGenerated by `python report.py` from the files in `outputs/`.\n\n" + body)
    print(f"Wrote {path} and {len([f for f in os.listdir(out_dir) if f.endswith('.png')])} figures")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=REPORT_DIR)
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
