# SSSM: Semantic State Space Model prototype (small local models)

A small, runnable test of the **Semantic State Space Model (SSSM)** idea from NVIDIA's
*"You don't need long context"* ([arXiv 2611.05306](https://arxiv.org/abs/2611.05306), copy in `paper/`):

> A SSSM is a model post-trained to (1) comprehend an input language description of a state of a
> conversation, (2) reason about the appropriate next action, (3) take an action, (4) register its outcome,
> and (5) update the language description of the conversation's state [...] so that a separate inference
> instance of the SSSM can continue the conversation toward completion.

The paper only proposes SSSMs. Its closest experiment is **Single-Action Compaction (SAC)**, and this project
builds on that and adds the missing piece: a model *post-trained* for the SSSM loop. The comparison is against
a normal local chat model that keeps the whole conversation in context.

## What is compared

| mode | what the model sees at each step | flag |
|---|---|---|
| **FHA** (normal chat) | the whole conversation so far (every action and tool output) | `--modes fha` |
| **EoCC** | the whole conversation until it passes a token threshold, then the model writes a state summary and starts over | `--modes eocc` |
| **SSSM** | a fresh context holding only the task and the current written state. The model takes one action, sees the result, rewrites the state, and the context is thrown away | `--modes sssm` |

Each mode can run on the **base** model (prompted only) or on a **LoRA-fine-tuned** model:
`sssm-sft` is trained on SSSM-format traces, and `fha-sft` is a control trained with the same token budget
on the same tasks in full-history format. This way any gain from SSSM training can be told apart from a
gain from training on the task.

The state follows the paper's Appendix A guidance (goal, completion criteria, plan, facts and artifacts,
last step, next step):

```
GOAL: ...
DONE WHEN: ...
PLAN: ...
FACTS: ...
LAST STEP: ...
NEXT: ...
```

## Tasks (deterministic and graded automatically)

The agent has the tools `list_files()`, `read_file(name)`, `calc(expr)`, and `submit(answer)`. The number of
steps needed grows with the **horizon**.

* `ledger`: keep a running balance for one account across N transaction files that are full of distractors.
* `pointer`: follow a chain of N files, where each file names the next one, and collect a letter from each.
* `cart`: read N customer messages one at a time (`next_message()`) and keep a shopping cart up to date. This
  family is **held out** from training to test generalization.

## Hardware and models

Everything runs on the CPU and fits in a few GB of RAM. Qwen2.5-0.5B-Instruct (about 0.4 GB at Q4_K_M) is the
default for training. 1.5B and 3B also work for evaluation (`--model qwen2.5:1.5b-instruct`). LoRA SFT of the
0.5B model takes about 4 to 6 GB of RAM on the CPU. Pick a model by available RAM:

| free RAM | inference | fine-tuning |
|---|---|---|
| 4 GB | qwen2.5:0.5b-instruct | 0.5B with `--max-len 1024` |
| 8 GB | qwen2.5:1.5b-instruct | 0.5B |
| 16 GB+ | qwen2.5:3b-instruct | 1.5B |

## Setup

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:0.5b-instruct
python3 -m venv .venv && .venv/bin/pip install -e '.[report,train,dev]'
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ~/hf/Qwen2.5-0.5B-Instruct
# only needed to import fine-tuned models into Ollama at Q4_K_M:
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
.venv/bin/pip install -e ~/llama.cpp/gguf-py
cmake -B ~/llama.cpp/build -S ~/llama.cpp && cmake --build ~/llama.cpp/build --target llama-quantize -j
```

## Running

```bash
# quick prompted comparison with no training
.venv/bin/python -m sssm.run --model qwen2.5:0.5b-instruct --modes fha,sssm \
    --families ledger,pointer --horizons 4,8 --seeds 0,1 --out results/quick.jsonl
.venv/bin/python -m sssm.report results/quick.jsonl --md results/quick.md --png results/quick.png

# full pipeline: gold traces -> LoRA SFT (SSSM + FHA control) -> Ollama import -> eval -> report
HORIZONS=4,8 SEEDS=0,1 scripts/run_experiment.sh

.venv/bin/pytest -q && .venv/bin/ruff check .
```

Code map: `sssm/agents.py` (FHA / EoCC / SSSM loops), `sssm/tasks.py` (environments and graders),
`sssm/llm.py` (Ollama client and token accounting), `sssm/train/gold.py` (oracle traces),
`sssm/train/sft.py` (LoRA SFT), `sssm/run.py`, `sssm/report.py`, `scripts/`.

## Results (CPU-only VM, 8 cores, Qwen2.5 at Q4_K_M, 2 seeds x horizons 4/8/16 x 3 families = 18 episodes per row)

Full tables: `results/qwen0.5b.md`, `results/qwen1.5b.md`. Plots: `results/*.png`. The raw per-episode JSONL files
include full traces, so every state the SSSM wrote can be inspected.

**Qwen2.5-0.5B**

| model / mode | accuracy | partial credit | peak ctx mean/max | prompt tok/episode | LLM calls | s/episode |
|---|---|---|---|---|---|---|
| base / FHA (normal chat) | 0% | 10.7% | 2800 / 16533 | 53.7k | 19.4 | 41.7 |
| base / SSSM (prompted SAC) | 0% | 0.0% | 859 / 1573 | 35.9k | 66.9 | 83.0 |
| FHA-SFT / FHA (control, 150k tok) | 5.6% | 48.8% | 1311 / 2547 | 10.7k | 12.4 | 6.7 |
| SSSM-SFT / SSSM (150k tok, about 0.25 epoch) | 0% | 1.9% | 883 / 1129 | 33.0k | 57.5 | 83.9 |
| SSSM-SFT-ep1 / SSSM (1 epoch, about 600k tok) | 5.6% | 34.5% | 872 / 1382 | 8.4k | 15.3 | 15.6 |

**Qwen2.5-1.5B (base only, no fine-tuning)**

| model / mode | accuracy | partial credit | peak ctx mean/max | prompt tok/episode | LLM calls | s/episode |
|---|---|---|---|---|---|---|
| base / FHA (normal chat) | 11.1% | 31.8% | 634 / 2561 | 7.5k | 10.2 | 4.6 |
| base / SSSM (prompted SAC) | 0% | 1.0% | 913 / 1323 | 31.7k | 56.9 | 155.7 |

(Base 1.5B FHA often submits early, for example after one failed `calc`, so it has short contexts and is fast.
It still beats prompted SSSM, which never fills in the state.)

What this shows:

* **The context bound holds.** In SSSM mode, peak context stays flat at about 0.6–1.4k tokens at every horizon,
  while full-history context grows with the horizon. For base FHA on ledger h=16 it reached 16.5k, which is past
  the 16k window.
* **Prompting alone does not make a small model an SSSM.** Base 0.5B and 1.5B do not keep the state up to date.
  They copy the placeholders forward or repeat the last action until they hit the step limit. This matches the
  paper's point that SSSM behavior has to be post-trained.
* **Post-training teaches the loop.** The 1-epoch SSSM-SFT model reads each file once, updates GOAL, FACTS,
  LAST STEP and NEXT correctly, and submits after the right number of steps. A fresh context then continues from
  the state alone. On average it uses fewer prompt tokens per episode than the full-history control (8.4k vs
  10.7k) with a smaller peak context. The gap grows with the horizon: at horizon 16, peak context is about 0.85k
  vs 1.5k on pointer and 1.1k vs 2.4k on ledger.
* **What still fails is in-head arithmetic and copying, not the state mechanism.** A typical miss: the state says
  `-8 -22 = -65`. Both fine-tuned models make this kind of error, so exact accuracy is near zero for both. The
  equal-token SSSM model (150k tokens) is undertrained. The control sees each training episode about 2.25 times,
  while the SSSM model sees about a quarter of its data, because every SSSM example repeats the system prompt and
  state. Its training cost per episode is about 7.5x higher (601k vs 67k tokens for the same 80 episodes).
* **Caveats:** 2 seeds per cell, so differences of one episode are noise. Partial credit for ledger scales with
  the size of the answer, so it is only comparable within a horizon. Resident GiB is 0.00 for two rows because of a
  name-matching bug in the measurement, which has since been fixed. On CPU, SSSM costs about 2 LLM calls per
  action, so wall time is higher at small horizons.

Next steps that would likely matter: train the SSSM to call `calc()` instead of doing arithmetic in its head;
run 2–3 epochs, or a 1.5B/3B base on a GPU; add more seeds.
