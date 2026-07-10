#!/usr/bin/env bash
set -euo pipefail

# Requires Hugging Face login and approved access to meta-llama/Llama-3.1-8B.
# This script intentionally runs only base and SEQ methods, not LLMC methods.

cd "$HOME/SEQ_Clean"
source .venv-seq/bin/activate

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/${TS}"
mkdir -p "$OUT_DIR"
echo "Output directory: $OUT_DIR"

python run_compare_matrix.py \
  --models "meta-llama/Llama-3.1-8B" \
  --device auto \
  --methods "base,seq" \
  --benchmarks "ppl,hellaswag,arc_easy,arc_challenge,piqa,winogrande,lambada_openai" \
  --experiments_file experiments.yaml \
  --output_dir "$OUT_DIR" \
  --lm_eval_num_fewshot 0 \
  --lm_eval_batch_size 1 \
  --lm_eval_fail_policy warn
