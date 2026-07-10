#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/SEQ_Clean"
source .venv-seq/bin/activate

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results_smoke/${TS}"
mkdir -p "$OUT_DIR"
echo "Output directory: $OUT_DIR"

python run_compare_matrix.py \
  --models "facebook/opt-125m" \
  --device auto \
  --methods "base,seq" \
  --benchmarks "ppl" \
  --experiments_file experiments.yaml \
  --output_dir "$OUT_DIR" \
  --lm_eval_num_fewshot 0 \
  --lm_eval_batch_size 1 \
  --lm_eval_fail_policy warn

find "$OUT_DIR" -maxdepth 4 -type f | sort | head -100
