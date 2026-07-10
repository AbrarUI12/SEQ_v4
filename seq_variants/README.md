# SEQ Variants

Run the smoke test without loading large models:

```bash
python seq_variants/smoke_test_variants.py
```

| Method name | Folder | Change |
|---|---|---|
| seq / seq_v0 | SEQ-v0 | current SEQ |
| seq_v1 | SEQ-v1 | activation masking only |
| seq_v2 | SEQ-v2 | v1 + thresholds 0.80/0.80 |
| seq_v3 | SEQ-v3 | v2 + all attention projections >= INT8 |
| seq_v4 | SEQ-v4 | v3 + gate/down projections >= INT8 |
| seq_v5 | SEQ-v5 | risk-score policy |

## Example 1: Run One Variant

```bash
TS=$(date +%Y%m%d_%H%M%S)

python run_compare_matrix.py \
  --models "meta-llama/Llama-3.2-3B" \
  --device auto \
  --methods "base,seq_v5" \
  --benchmarks "ppl" \
  --experiments_file experiments_seq_variants.yaml \
  --output_dir "results/${TS}" \
  --lm_eval_num_fewshot 0 \
  --lm_eval_batch_size 1 \
  --lm_eval_fail_policy warn \
  --llmc_repo /mnt/d/LightCompress \
  --llmc_venv /mnt/d/LightCompress/.venv-llmc \
  --llmc_model_type Llama \
  --llmc_save_mode fake \
  --llmc_calib_dataset wikitext2 \
  --llmc_eval_dataset wikitext2 \
  --llmc_calib_samples 32 \
  --llmc_calib_seq_len 512 \
  --llmc_eval_seq_len 2048
```

## Example 2: Run All Variants

```bash
TS=$(date +%Y%m%d_%H%M%S)

python run_compare_matrix.py \
  --models "meta-llama/Llama-3.2-3B" \
  --device auto \
  --methods "base,seq_v0,seq_v1,seq_v2,seq_v3,seq_v4,seq_v5" \
  --benchmarks "ppl,hellaswag,arc_easy,arc_challenge,piqa,winogrande,lambada_openai" \
  --experiments_file experiments_seq_variants.yaml \
  --output_dir "results/${TS}" \
  --lm_eval_num_fewshot 0 \
  --lm_eval_batch_size 1 \
  --lm_eval_fail_policy warn \
  --llmc_repo /mnt/d/LightCompress \
  --llmc_venv /mnt/d/LightCompress/.venv-llmc \
  --llmc_model_type Llama \
  --llmc_save_mode fake \
  --llmc_calib_dataset wikitext2 \
  --llmc_eval_dataset wikitext2 \
  --llmc_calib_samples 32 \
  --llmc_calib_seq_len 512 \
  --llmc_eval_seq_len 2048
```
