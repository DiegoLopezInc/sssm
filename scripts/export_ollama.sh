#!/usr/bin/env bash
# Import a merged HF checkpoint into Ollama at the same quantization (Q4_K_M) and with the
# same chat template as the base Ollama model, so the fine-tuned SSSM is compared like-for-like.
#   scripts/export_ollama.sh models/sssm-qwen0.5b sssm-sft-qwen0.5b qwen2.5:0.5b-instruct
# Needs a llama.cpp checkout with llama-quantize built (LLAMA_CPP, default ~/llama.cpp):
#   git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
#   pip install -e ~/llama.cpp/gguf-py
#   cmake -B ~/llama.cpp/build -S ~/llama.cpp && cmake --build ~/llama.cpp/build --target llama-quantize -j
set -euo pipefail
SRC=$(realpath "$1"); NAME=$2; BASE=${3:-qwen2.5:0.5b-instruct}; QUANT=${QUANT:-Q4_K_M}
LLAMA_CPP=${LLAMA_CPP:-$HOME/llama.cpp}
PY=${PY:-python3}
"$PY" "$LLAMA_CPP/convert_hf_to_gguf.py" "$SRC" --outfile "$SRC/model-f16.gguf" --outtype f16
"$LLAMA_CPP/build/bin/llama-quantize" "$SRC/model-f16.gguf" "$SRC/model-$QUANT.gguf" "$QUANT"
rm -f "$SRC/model-f16.gguf"
MF=$(mktemp)
echo "FROM $SRC/model-$QUANT.gguf" > "$MF"
ollama show "$BASE" --modelfile | grep -v -E '^(FROM|#)' >> "$MF"
ollama create "$NAME" -f "$MF"
rm -f "$MF"
ollama list | grep "$NAME"
