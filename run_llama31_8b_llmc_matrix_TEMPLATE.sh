# TEMPLATE ONLY. LightCompress was not found during setup.
# Do not run until <ACTUAL_LIGHTCOMPRESS_PATH> is replaced with a real repo path
# and that repo has a working .venv-llmc.
#
# Notes from current validation docs:
# - Stable first LLMC set: base,seq,gptq_llmc,smoothquant_llmc,rtn_llmc
# - Broader set with llm_int8_llmc: base,seq,gptq_llmc,smoothquant_llmc,rtn_llmc,llm_int8_llmc
# - omniquant_llmc is recognized but not recommended/possibly disabled in notes.

cd "$HOME/SEQ_Clean"
source .venv-seq/bin/activate

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/${TS}"
mkdir -p "$OUT_DIR"
echo "Output directory: $OUT_DIR"

python run_compare_matrix.py \
  --models "meta-llama/Llama-3.1-8B" \
  --device auto \
  --methods "base,seq,gptq_llmc,smoothquant_llmc,rtn_llmc,llm_int8_llmc,omniquant_llmc" \
  --benchmarks "ppl,hellaswag,arc_easy,arc_challenge,piqa,winogrande,lambada_openai" \
  --experiments_file experiments.yaml \
  --output_dir "$OUT_DIR" \
  --lm_eval_num_fewshot 0 \
  --lm_eval_batch_size 1 \
  --lm_eval_fail_policy warn \
  --llmc_repo "<ACTUAL_LIGHTCOMPRESS_PATH>" \
  --llmc_venv "<ACTUAL_LIGHTCOMPRESS_PATH>/.venv-llmc" \
  --llmc_model_type Llama \
  --llmc_save_mode fake \
  --llmc_calib_dataset wikitext2 \
  --llmc_eval_dataset wikitext2 \
  --llmc_calib_samples 32 \
  --llmc_calib_seq_len 512 \
  --llmc_eval_seq_len 2048
