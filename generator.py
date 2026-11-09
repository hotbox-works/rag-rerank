"""Llama 3.1 8B Instruct: query-focused passage summaries and answer generation.

Both steps share one loaded model. Prompts are module constants so the report can quote them exactly.
Decoding is greedy everywhere, so reruns give the same text.

    python generator.py "Who directed the film that won Best Picture in 1998?"   # closed-book check

Paths and the model come from .env (see config.py): LLAMA_MODEL, LLAMA_BACKEND, HF_TOKEN.
"""
import argparse
import re
import time

from utils import config

LLAMA_MODEL = config.LLAMA_MODEL
LLAMA_BACKEND = config.LLAMA_BACKEND

LLM_BATCH_SIZE = 16  # prompts per generate() call with the transformers backend
MAX_PROMPT_TOKENS = 3072
SUMMARY_MAX_TOKENS = 96
ANSWER_MAX_TOKENS = 32
NOT_RELEVANT = "Not relevant."


# Prompts

SYSTEM_PROMPT = ("You are a precise question answering assistant. Answers are short phrases such as a name, "
                 "a date, a number, or yes/no, never full sentences.")

SUMMARY_PROMPT = """Passage ({title}):
{text}

Question: {question}

Write at most two sentences stating only the facts from the passage that help answer the question. \
Copy names, dates and numbers exactly as written. Do not answer the question yourself. \
If the passage has nothing useful for the question, write exactly: {not_relevant}"""

ANSWER_PROMPT = """Use the documents below to answer the question. The question may need facts from more than \
one document. If the documents do not contain the answer, answer from your own knowledge.

{context}

Question: {question}
Reply with one line in the form "Answer: <answer>", using as few words as possible."""

CLOSED_BOOK_PROMPT = """Answer the question.

Question: {question}
Reply with one line in the form "Answer: <answer>", using as few words as possible."""


def passage_body(passage):
    if isinstance(passage, dict):
        return passage.get("title", "") or "", passage.get("text", "") or ""
    return "", str(passage)


def summary_messages(question, passage):
    title, text = passage_body(passage)
    user = SUMMARY_PROMPT.format(title=title or "untitled", text=text, question=question, not_relevant=NOT_RELEVANT)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def format_context(contexts):
    """`contexts` is a list of {"title", "text"} dicts, where text is either the raw passage or its summary."""
    return "\n\n".join(f"Document {i + 1} ({c.get('title') or 'untitled'}): {c.get('text', '')}"
                       for i, c in enumerate(contexts))


def answer_messages(question, contexts):
    if contexts:
        user = ANSWER_PROMPT.format(context=format_context(contexts), question=question)
    else:
        user = CLOSED_BOOK_PROMPT.format(question=question)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# Output parsing

_ANSWER_RE = re.compile(r"answer\s*:\s*(.+)", re.IGNORECASE)
_YESNO_RE = re.compile(r"^(yes|no)(?:\s*[,.;:!]|\s*$)", re.IGNORECASE)


def parse_answer(text):
    """Pull the short answer out of a generation: the text after the last "Answer:", else the first line.

    "Yes, both are ..." becomes "yes"; "No Doubt" stays as it is because no punctuation follows "No".
    """
    text = (text or "").strip()
    matches = _ANSWER_RE.findall(text)
    ans = matches[-1] if matches else (text.splitlines()[0] if text else "")
    ans = ans.strip().splitlines()[0].strip() if ans.strip() else ""
    ans = ans.strip("*").strip()
    if len(ans) >= 2 and ans[0] == ans[-1] and ans[0] in "\"'":
        ans = ans[1:-1].strip()
    m = _YESNO_RE.match(ans)
    if m:
        return m.group(1).lower()
    return ans.rstrip(".").strip()


def clean_summary(text):
    text = (text or "").strip()
    return NOT_RELEVANT if text.lower().startswith(NOT_RELEVANT.lower().rstrip(".")) else text


# Model

class Llama:
    """Greedy chat generation over a batch of conversations, with a transformers or vLLM backend."""

    def __init__(self, model_name=LLAMA_MODEL, backend=LLAMA_BACKEND, batch_size=LLM_BATCH_SIZE,
                 max_prompt_tokens=MAX_PROMPT_TOKENS, dtype="bfloat16", device=None, gpu_memory_utilization=0.6):
        self.model_name = model_name
        self.backend = backend
        self.batch_size = batch_size
        self.max_prompt_tokens = max_prompt_tokens
        if backend == "vllm":
            from vllm import LLM

            self.llm = LLM(model=model_name, dtype=dtype, gpu_memory_utilization=gpu_memory_utilization,
                           max_model_len=max_prompt_tokens + 256, seed=0)
            self.tok = self.llm.get_tokenizer()
        elif backend == "hf":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.torch = torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.tok.padding_side = "left"
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=getattr(torch, dtype)).to(self.device).eval()
            eos = [self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|eot_id|>")]
            self.eos_ids = [i for i in eos if i is not None and i != self.tok.unk_token_id]
        else:
            raise ValueError("backend must be 'hf' or 'vllm'")

    def chat(self, conversations, max_new_tokens=ANSWER_MAX_TOKENS):
        """Return one generated string per conversation (a list of {"role", "content"} messages)."""
        if not conversations:
            return []
        if self.backend == "vllm":
            from vllm import SamplingParams

            params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
            outs = self.llm.chat(conversations, params, use_tqdm=False)
            return [o.outputs[0].text.strip() for o in outs]
        torch = self.torch
        prompts = [self.tok.apply_chat_template(c, add_generation_prompt=True, tokenize=False) for c in conversations]
        out = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start:start + self.batch_size]
            enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_prompt_tokens, add_special_tokens=False).to(self.device)
            with torch.inference_mode():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                          top_p=None, pad_token_id=self.tok.pad_token_id, eos_token_id=self.eos_ids)
            new = gen[:, enc["input_ids"].shape[1]:]
            out.extend(t.strip() for t in self.tok.batch_decode(new, skip_special_tokens=True))
        return out


# Steps used by the pipeline

def summarize_batch(llm, pairs, max_new_tokens=SUMMARY_MAX_TOKENS):
    """Summaries for a list of (question, passage) pairs, in order."""
    return [clean_summary(t) for t in llm.chat([summary_messages(q, p) for q, p in pairs], max_new_tokens)]


def summarize(llm, question, passage):
    return summarize_batch(llm, [(question, passage)])[0]


def generate_batch(llm, items, max_new_tokens=ANSWER_MAX_TOKENS):
    """Answers for a list of (question, contexts) items. Returns (parsed answers, raw generations)."""
    raw = llm.chat([answer_messages(q, ctx) for q, ctx in items], max_new_tokens)
    return [parse_answer(r) for r in raw], raw


def generate(llm, question, contexts):
    answers, raw = generate_batch(llm, [(question, contexts)])
    return answers[0], raw[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--model", default=LLAMA_MODEL)
    ap.add_argument("--backend", default=LLAMA_BACKEND, choices=["hf", "vllm"])
    args = ap.parse_args()
    llm = Llama(args.model, args.backend)
    t0 = time.perf_counter()
    answer, raw = generate(llm, args.question, [])
    print(f"raw: {raw!r}\nanswer: {answer!r}\n{time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
