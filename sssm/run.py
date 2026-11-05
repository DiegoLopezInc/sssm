"""Run FHA / EoCC / SSSM episodes against a local model and append results to JSONL.

Example:
    python -m sssm.run --model qwen2.5:1.5b-instruct --modes fha,sssm \
        --families ledger,pointer,cart --horizons 4,8,16 --seeds 0,1,2 --out results/q15.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from .agents import run_fha, run_sssm
from .llm import ChatClient
from .tasks import make_task


def loaded_model_bytes(base_url: str, model: str) -> int | None:
    """Resident size Ollama reports for the loaded model (weights + KV cache + buffers)."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/ps", timeout=10) as r:
            for m in json.loads(r.read()).get("models", []):
                names = {m.get("name"), m.get("model")}
                if model in names or f"{model}:latest" in names:
                    return int(m.get("size", 0))
    except OSError:
        return None
    return None


def done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        r = json.loads(line)
        keys.add((r["model"], r["mode"], r["family"], r["horizon"], r["seed"]))
    return keys


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen2.5:1.5b-instruct")
    p.add_argument("--label", default=None, help="name stored in results (defaults to --model)")
    p.add_argument("--base-url", default="http://localhost:11434")
    p.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    p.add_argument("--modes", default="fha,sssm", help="comma list of fha,eocc,sssm")
    p.add_argument("--families", default="ledger,pointer,cart")
    p.add_argument("--horizons", default="4,8,16")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--num-ctx-fha", type=int, default=16384, help="KV-cache size for fha/eocc")
    p.add_argument("--num-ctx-sssm", type=int, default=4096, help="KV-cache size for sssm")
    p.add_argument("--eocc-threshold", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--out", default="results/results.jsonl")
    p.add_argument("--no-trace", action="store_true", help="drop per-step traces from output")
    a = p.parse_args(argv)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    label = a.label or a.model
    skip = done_keys(out)
    for mode in a.modes.split(","):
        num_ctx = a.num_ctx_sssm if mode == "sssm" else a.num_ctx_fha
        llm = ChatClient(a.model, a.base_url, a.backend, num_ctx=num_ctx, temperature=a.temperature)
        for family in a.families.split(","):
            for horizon in map(int, a.horizons.split(",")):
                for seed in map(int, a.seeds.split(",")):
                    if (label, mode, family, horizon, seed) in skip:
                        continue
                    llm.seed = seed
                    task = make_task(family, horizon, seed)
                    if mode == "sssm":
                        ep = run_sssm(task, llm)
                    else:
                        threshold = a.eocc_threshold if mode == "eocc" else None
                        ep = run_fha(task, llm, compact_threshold=threshold)
                    rec = ep.record()
                    rec["model"] = label
                    rec["num_ctx"] = num_ctx
                    rec["truncation_risk"] = rec["peak_context"] >= num_ctx
                    if a.backend == "ollama":
                        rec["resident_bytes"] = loaded_model_bytes(a.base_url, a.model)
                    if a.no_trace:
                        rec.pop("trace")
                    with out.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(
                        f"{label:24s} {mode:5s} {family:8s} h={horizon:<3d} seed={seed} "
                        f"correct={rec['correct']!s:5s} steps={rec['steps']:<3d} "
                        f"peak_ctx={rec['peak_context']:<6d} tokens_in={rec['prompt_tokens']:<7d} "
                        f"t={rec['wall_seconds']}s",
                        flush=True,
                    )


if __name__ == "__main__":
    sys.exit(main())
