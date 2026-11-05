"""LoRA supervised fine-tuning on chat-formatted JSONL (CPU-friendly, no Trainer magic).

Loss is applied only to assistant tokens: the last assistant message by default, or every
assistant message when an example sets ``"train_on": "all_assistant"`` (full-history format).

    python -m sssm.train.sft --base ~/hf/Qwen2.5-0.5B-Instruct --data data/sssm_train.jsonl \
        --out models/sssm-qwen0.5b --epochs 1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def _ids(tok, messages: list[dict], add_generation_prompt: bool = False) -> list[int]:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return tok(text, add_special_tokens=False)["input_ids"]


def encode(tok, messages: list[dict], train_on: str, max_len: int) -> tuple[list[int], list[int]]:
    ids = _ids(tok, messages)
    labels = [-100] * len(ids)
    targets = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if train_on != "all_assistant":
        targets = targets[-1:]
    for i in targets:
        start = len(_ids(tok, messages[:i], add_generation_prompt=True))
        end = len(_ids(tok, messages[: i + 1]))
        labels[start:end] = ids[start:end]
    return ids[:max_len], labels[:max_len]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="HF model dir or hub id")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--max-train-tokens", type=int, default=0, help="stop after this many tokens (0 = off)")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    torch.manual_seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=dtype).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=a.rank,
            lora_alpha=2 * a.rank,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    rows = [json.loads(line) for line in Path(a.data).read_text().splitlines()]
    data = [encode(tok, r["messages"], r.get("train_on", "last"), a.max_len) for r in rows]
    n_tok = sum(len(x) for x, _ in data)
    n_sup = sum(sum(1 for t in y if t != -100) for _, y in data)
    print(f"{len(data)} examples, {n_tok} tokens ({n_sup} supervised)")

    total_steps = math.ceil(len(data) * a.epochs)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 10) * max(0.05, 1 - s / max(1, total_steps // a.grad_accum))
    )
    model.train()
    order, seen_tok, t0, running = [], 0, time.time(), []
    for step in range(total_steps):
        if not order:
            order = list(range(len(data)))
            random.Random(a.seed + step).shuffle(order)
        ids, labels = data[order.pop()]
        x = torch.tensor([ids], device=device)
        y = torch.tensor([labels], device=device)
        loss = model(input_ids=x, labels=y).loss
        (loss / a.grad_accum).backward()
        running.append(loss.item())
        seen_tok += len(ids)
        if (step + 1) % a.grad_accum == 0 or step + 1 == total_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        if (step + 1) % 20 == 0 or step + 1 == total_steps:
            el = time.time() - t0
            print(
                f"step {step + 1}/{total_steps} loss {sum(running) / len(running):.4f} "
                f"tok {seen_tok} ({seen_tok / el:.0f} tok/s) {el / 60:.1f} min",
                flush=True,
            )
            running = []
        if a.max_train_tokens and seen_tok >= a.max_train_tokens:
            print(f"reached token budget {seen_tok}")
            break

    merged = model.merge_and_unload()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    meta = {**vars(a), "examples": len(data), "tokens_seen": seen_tok, "minutes": (time.time() - t0) / 60}
    (Path(a.out) / "train_meta.json").write_text(json.dumps(meta))
    print(f"saved merged model to {a.out}")


if __name__ == "__main__":
    main()
