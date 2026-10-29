# Project 2

## Setup

You need a Jupyter session on an A40 or A30 GPU node and a Java 21+ JDK (Pyserini needs it; nothing is downloaded for you).

1. Copy the settings template:
   ```bash
   cp .env.example .env
   ```

2. In `.env`, set `JAVA_HOME` to your JDK (required). The other settings put caches and outputs under `DIR_SCRATCH`, which defaults to `$RCAC_SCRATCH`. Change them if you want things elsewhere. A variable set in your shell overrides `.env`.

3. Request access to [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on Hugging Face. The notebook logs you in when you reach the Llama step.

## Getting started

Open `project2.ipynb` with this folder as the working directory and run the cells in order:

1. **Setup**: prints the settings in use and checks `JAVA_HOME` and the GPU.
2. **BM25**: retrieves the top 15 Wikipedia passages for each dev question and saves them to `bm25_top15.json`.
3. **monoT5**: scores every passage and saves the scores to `t5_scores.jsonl`. If you rerun it, it continues where it stopped.
4. **Sweep**: replays adaptive early stopping for each threshold and cap and saves the results to `sweep.csv`.
5. **Llama**: logs in to Hugging Face and loads Llama 3.1 8B Instruct (`model.py`).
6. **Evaluate**: answers every dev question under each retrieval setting (no passages, BM25, monoT5 rerank, adaptive monoT5) and saves EM/F1 to `llama_eval.csv`. Each answer is saved to `llama/<setting>.jsonl`, so a rerun continues where it stopped.
7. **Live pipeline**: runs BM25, then monoT5, then Llama on a few questions, as the pipeline would on new input (`rag.RAGPipeline`).
8. **Zip**: packs the outputs into `project2_retrieval_results.zip`.

All outputs go to `DIR_OUTPUTS` (`outputs/` by default). You can run the same steps without the notebook: `python retrieval.py cache`, then `score`, then `sweep`, then `python rag.py eval`. To answer a single question, run `python rag.py ask "..."`.

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
