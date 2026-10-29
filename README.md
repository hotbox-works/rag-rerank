# Project 2

## Setup

You need a Jupyter session on an A40 or A30 GPU node and a Java 21+ JDK (Pyserini needs it; nothing is downloaded for you).

1. Copy the settings template:
   ```bash
   cp .env.example .env
   ```

2. In `.env`, set `JAVA_HOME` to your JDK and `HF_TOKEN` to a Hugging Face token with access to Llama 3.1 8B Instruct (both required). The other settings put caches and outputs under `DIR_SCRATCH`, which defaults to `$RCAC_SCRATCH`. Change them if you want things elsewhere. A variable set in your shell overrides `.env`.

## Getting started

Open `project2.ipynb` with this folder as the working directory and run the cells in order:

1. **Setup**: prints the settings in use and checks `JAVA_HOME` and the GPU. Install packages first with `pip install -e ".[dev]"` (add `vllm` for faster generation).
2. **BM25**: retrieves the top 15 Wikipedia passages for each dev question and saves them to `bm25_top15.json`.
3. **monoT5**: scores every passage and saves the scores to `t5_scores.jsonl`. If you rerun it, it continues where it stopped.
4. **Sweep**: replays adaptive early stopping for each threshold and cap and saves the results to `sweep.csv`.
5. **Pipeline smoke test**: answers three dev questions live with BM25, monoT5, and Llama.
6. **End-to-end baselines and sweep**: closed book, BM25 top-k, and a threshold x keep grid, scored with EM/F1.
7. **Ablations**: fallback strategy, early stopping vs. scoring all 15, raw passages vs. Llama summaries, on the best setting.
8. **Scorer fine-tuning**: results of `train_scorer.py` (run from a terminal), then the fine-tuned scorer end to end.
9. **Latency**: live, one question at a time, per stage.
10. **Test predictions**: `test_predictions.json` in HotpotQA format.
11. **Report**: `results.csv` plus figures and tables in `outputs/report/`.
12. **Zip**: packs the outputs into `project2_results.zip`.

All outputs go to `DIR_OUTPUTS` (`outputs/` by default).

## Code

| File | What it does |
|---|---|
| `retrieval.py` | BM25 retrieval (`Retriever.retrieve(query, offset, k=5)`), monoT5 `score(query, doc)`, the adaptive loop, offline sweep |
| `generator.py` | Llama 3.1 8B Instruct: query-focused passage summaries, answer prompt, answer parsing |
| `pipeline.py` | `answer(query)`: adaptive retrieval, then summaries, then the answer; live or replayed from saved scores |
| `train_scorer.py` | T5 scorer training data from `train.jsonl` (distant labels), fine-tuning, hyperparameter sweep, dev rescoring |
| `evaluate.py` | Named experiments, resumable runs, EM/F1 and error analysis, results table, latency, test predictions |
| `report.py` | Figures and `results.md` for the report and slides |
| `tests/` | `python -m pytest` (no GPU needed); `RUN_GPU_TESTS=1 python -m pytest tests/test_gpu_smoke.py -s` on a GPU node |

## Running without the notebook

```bash
python retrieval.py cache && python retrieval.py score && python retrieval.py sweep
python pipeline.py "Which band formed first, Oasis or Blur?"           # one live answer
python evaluate.py run --group baselines sweep --limit 50                # quick check
python evaluate.py run --group baselines sweep                           # full dev set
python evaluate.py best
python evaluate.py run --group ablations
python train_scorer.py data && python train_scorer.py base && python train_scorer.py sweep
python train_scorer.py rescore --model outputs/scorer_runs/<best>/best
python evaluate.py run --group finetuned
python evaluate.py latency --n 100
python evaluate.py table && python report.py
```

For the test set, build its BM25 cache and scores first (`python retrieval.py cache --data data/test.jsonl --cache outputs/bm25_top15_test.json`, then `score` with the same `--data`/`--cache` and `--scores outputs/t5_scores_test.jsonl`), then `python evaluate.py test --exp final`.

## Baselines from Parts 1 to 3

All baselines use Llama 3.1 8B Instruct on the same 1,110 HotpotQA dev questions, scored with `hotpot_eval.py`.

| Baseline | Part | Setting | EM | F1 |
|---|---|---|---|---|
| Naive prompting | 1 | Raw question | 5.23 | 12.37 |
| Instruct prompting | 1 | Best version (V2) | 24.59 | 32.20 |
| Chain-of-thought prompting | 1 | Step by step, final answer line | 23.78 | 32.02 |
| Random few-shot ICL | 2 | n = 2 | 23.06 | 30.67 |
| Retrieval-based ICL | 2 | BM25 demos, n = 10 | 25.32 | 34.01 |
| Wikipedia RAG | 3 | BM25 passages, n = 8, context before question | 30.90 | 41.68 |
