# Baselines from Parts 1 to 3

All baselines use Llama 3.1 8B Instruct on the same 1,110 HotpotQA dev questions, scored with `hotpot_eval.py`.
`report.py` puts these next to this project's runs in `outputs/report/results.md` and `e2e_results.png`.

| Baseline | Part | Setting | EM | F1 |
|---|---|---|---|---|
| Naive prompting | 1 | Raw question | 5.23 | 12.37 |
| Instruct prompting | 1 | Best version (V2) | 24.59 | 32.20 |
| Chain-of-thought prompting | 1 | Step by step, final answer line | 23.78 | 32.02 |
| Random few-shot ICL | 2 | n = 2 | 23.06 | 30.67 |
| Retrieval-based ICL | 2 | BM25 demos, n = 10 | 25.32 | 34.01 |
| Wikipedia RAG | 3 | BM25 passages, n = 8, context before question | 30.90 | 41.68 |
