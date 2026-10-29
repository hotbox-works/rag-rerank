"""Llama 3.1 8B Instruct as the reader: answers a HotpotQA question from a set of passages.

The model is gated on Hugging Face, so log in (or set HF_TOKEN) before loading it.
"""
import re

LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_BATCH_SIZE = 16  # questions per generate() call
MAX_NEW_TOKENS = 32

SYSTEM_PROMPT = ("You answer trivia questions that may need facts from more than one source. Reply with only the "
                 "final answer: a short phrase such as a name, date or number, or yes or no. Do not explain.")


def format_passages(passages):
    return "\n\n".join(f"[{i}] {p.get('title', '')}: {p.get('text', '')}" for i, p in enumerate(passages, 1))


def build_messages(question, passages=()):
    """Chat messages for one question, with the passages before the question (the best layout in Part 3)."""
    if passages:
        user = (f"Passages:\n{format_passages(passages)}\n\n"
                "Answer the question using the passages. If they don't contain the answer, use what you know.\n\n"
                f"Question: {question}\nAnswer:")
    else:
        user = f"Question: {question}\nAnswer:"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def clean_answer(text):
    """First line of the reply, without an "Answer:" prefix, quotes, bold markers or a trailing period.

    A reply that opens with "Yes," or "No." and goes on to explain is cut to the yes or no.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    answer = lines[0] if lines else ""
    answer = re.sub(r"^(final answer|answer)\s*[:\-]\s*", "", answer, flags=re.I)
    answer = answer.strip(" *\"'").rstrip(".").strip()
    yes_no = re.match(r"(yes|no)([,.;:!]|$)", answer, flags=re.I)
    return yes_no.group(1).lower() if yes_no else answer


class LlamaReader:
    def __init__(self, model_name=LLAMA_MODEL, batch_size=LLAMA_BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS,
                 dtype=None, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = "bfloat16" if self.device == "cuda" and torch.cuda.is_bf16_supported() else "float32"
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=getattr(torch, dtype)).to(self.device).eval()
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

    def prompt(self, question, passages=()):
        return self.tok.apply_chat_template(build_messages(question, passages), tokenize=False,
                                            add_generation_prompt=True)

    def generate(self, questions, passages_list=None):
        """Greedy-decode an answer per question. Returns dicts with the cleaned "answer" and the "raw" reply.

        Prompts are batched in order of length so each batch needs little padding.
        """
        passages_list = passages_list or [()] * len(questions)
        prompts = [self.prompt(q, p) for q, p in zip(questions, passages_list)]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        out = [None] * len(prompts)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            enc = self.tok([prompts[i] for i in idx], padding=True, add_special_tokens=False,
                           return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          temperature=None, top_p=None, pad_token_id=self.tok.pad_token_id)
            replies = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, raw in zip(idx, replies):
                out[i] = {"answer": clean_answer(raw), "raw": raw}
        return out

    def answer(self, question, passages=()):
        return self.generate([question], [passages])[0]
