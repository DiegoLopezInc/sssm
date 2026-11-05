#!/usr/bin/env bash
# End-to-end SSSM experiment on a small local model.
#   1. build gold trajectories (cart family held out to test generalization)
#   2. LoRA-SFT the same base model on the SSSM format and, as a control, the FHA format
#   3. import both into Ollama at the base model's quantization
#   4. evaluate base-FHA, base-SSSM(prompted), SFT-SSSM and SFT-FHA; write the report
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
HF_BASE=${HF_BASE:-$HOME/hf/Qwen2.5-0.5B-Instruct}   # HF weights used for fine-tuning
OLLAMA_BASE=${OLLAMA_BASE:-qwen2.5:0.5b-instruct}   # same model as served by Ollama
TAG=${TAG:-qwen0.5b}
TRAIN_FAMILIES=${TRAIN_FAMILIES:-ledger,pointer}
EPISODES=${EPISODES:-40}
TOKENS=${TOKENS:-150000}          # equal training-token budget for both formats
EVAL_FAMILIES=${EVAL_FAMILIES:-ledger,pointer,cart}
HORIZONS=${HORIZONS:-4,8,16}
SEEDS=${SEEDS:-0,1,2}
OUT=results/$TAG.jsonl

for fmt in sssm fha; do
  [ -f data/${fmt}_train.jsonl ] || $PY -m sssm.train.gold --format $fmt --families $TRAIN_FAMILIES \
      --horizons 3-8 --episodes $EPISODES --out data/${fmt}_train.jsonl
  [ -d models/$fmt-$TAG ] || $PY -m sssm.train.sft --base "$HF_BASE" --data data/${fmt}_train.jsonl \
      --out models/$fmt-$TAG --epochs 3 --max-train-tokens $TOKENS 2>&1 | tee logs/sft-$fmt-$TAG.log
  ollama list | grep -q "^$fmt-sft-$TAG" || PY=$PY scripts/export_ollama.sh models/$fmt-$TAG $fmt-sft-$TAG "$OLLAMA_BASE"
done

COMMON=(--families $EVAL_FAMILIES --horizons $HORIZONS --seeds $SEEDS --out $OUT)
$PY -m sssm.run --model "$OLLAMA_BASE" --label "$TAG-base" --modes fha,sssm "${COMMON[@]}"
$PY -m sssm.run --model sssm-sft-$TAG --label "$TAG-sssm-sft" --modes sssm "${COMMON[@]}"
$PY -m sssm.run --model fha-sft-$TAG --label "$TAG-fha-sft" --modes fha "${COMMON[@]}"
$PY -m sssm.report $OUT --md results/$TAG.md --png results/$TAG.png
