#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/SEQ_Clean"
source .venv-seq/bin/activate

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results_smoke_lmeval/${TS}"
mkdir -p "$OUT_DIR"
echo "Output directory: $OUT_DIR"

python run_compare_matrix.py \
  --models "facebook/opt-125m" \
  --device auto \
  --methods "base" \
  --benchmarks "hellaswag" \
  --experiments_file experiments.yaml \
  --output_dir "$OUT_DIR" \
  --lm_eval_num_fewshot 0 \
  --lm_eval_batch_size 1 \
  --lm_eval_fail_policy warn

find "$OUT_DIR" -maxdepth 5 -type f | sort | head -120
