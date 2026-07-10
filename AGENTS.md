# AGENTS.md

## Project overview

- SEQ means Silver Entropy Quantization.
- This repository is a Python research codebase for post-training, inference-oriented LLM quantization using entropy-guided mixed precision.
- The core SEQ workflow collects weight and activation entropy, assigns INT4/INT8/FP16 tiers, applies bitsandbytes-backed replacements, and compares baseline versus quantized behavior.
- Evaluation covers memory/size, effective bits, perplexity, generation robustness probes, MMLU/zero-shot helpers, optional EleutherAI `lm-eval`, and compare-matrix runs against external quantization methods.
- Models visible in the repo include TinyLlama, Llama/Llama-2/Llama-3-family paths, Qwen2 inference support in code, OPT smoke configs for LLMC, and a local `models/Llama-3.1-8B/` snapshot.

## Repository layout

- `seq_core/`: SEQ-owned implementation.
  - `pipeline.py`: main SEQ experiment runner; CLI supports `--experiment`, `--all`, `--model_name`, and `--experiments_file`.
  - `entropy_metrics.py`: weight and activation entropy collection plus table serialization.
  - `precision_policy.py`: entropy-to-tier assignment, INT4/INT8/FP16 protections, and policy constraint checks.
  - `quantize_model.py`: bitsandbytes Linear4bit/Linear8bitLt replacement, verification, save/reload helpers, and effective-bit accounting.
- `benchmarks/`: evaluation and reporting helpers.
  - `ppl_eval.py`: standalone PPL runner.
  - `evaluation_suite.py`: baseline/quantized suite for PPL, tail risk, JSON stress, temperature sweep, long context, latency/memory, size, MMLU, zero-shot, and lm-eval.
  - `seq_lm_eval.py`: EleutherAI lm-evaluation-harness wrapper and result normalizer.
  - `multiple_choice_eval.py`: MMLU, ARC, and PIQA-style local evaluation helpers.
  - `reporting.py` and `plotting.py`: markdown/JSON reports and figures.
- `run_compare_matrix.py`: current public comparison runner for `base`, `seq`, direct `omniquant`, and LLMC-backed methods.
- `experiments.yaml`: main experiment config, defaulting to `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, canonical WikiText-2 PPL, MMLU, zero-shot, and compare-method settings.
- `experiments.smoke.yaml`: reduced smoke config with shorter sequence lengths, fewer examples, and smaller latency settings.
- `calibration_prompts.json`, `calibration_prompts_2.json`, `eval_prompts.json`: calibration and generation-evaluation prompt inputs.
- `third_party_quant/`: external quantization integrations.
  - `adapters/omniquant_adapter.py`: subprocess wrapper for pinned upstream OmniQuant.
  - `adapters/llmc_adapter.py` and `llmc_compare.py`: LightCompress/LLMC config rendering, subprocess execution, log parsing, and summary writing.
  - `llmc_templates/` and `llmc_smoke_configs/`: LLMC method templates and `facebook/opt-125m` smoke configs.
  - `run_llmc_smoke.sh`: WSL/Linux helper for LLMC smoke runs.
  - `envs/omniquant-upstream.environment.yml`: separate OmniQuant environment spec.
- `docs/`: focused docs for `seq_core` and lm-eval integration.
- `SETUP_CLEAN_SEQ.md`, `TRANSFER_TO_NEW_PC.md`, `LLMC_INTEGRATION_DISCOVERY.md`, `OMNIQUANT_UPSTREAM_ENV_SETUP.md`: setup, transfer, and integration notes. Treat these as evidence, but check code when commands disagree.
- `runs/`, `reports/`, `results/`, `artifacts/`: generated outputs. Do not edit or delete them unless the task explicitly asks.

## Environment setup

- Main SEQ setup is documented for Python 3.12 in `SETUP_CLEAN_SEQ.md`.
- Windows PowerShell SEQ environment:

```powershell
cd E:\SEQ_Clean
py -3.12 -m venv .venv-seq
.\.venv-seq\Scripts\python.exe -m pip install --upgrade pip
.\.venv-seq\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv-seq\Scripts\python.exe -m pip install accelerate==1.12.0 bitsandbytes==0.49.1 datasets==4.5.0 hf-xet==1.2.0 matplotlib==3.10.8 numpy==2.2.6 pyyaml==6.0.3 safetensors==0.7.0 sentencepiece==0.2.1 tokenizers==0.22.2 tiktoken==0.12.0 tqdm==4.67.1 transformers==4.57.6
```

- Linux/WSL SEQ environment:

```bash
cd /mnt/e/SEQ_Clean
python3.12 -m venv .venv-seq
source .venv-seq/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install accelerate==1.12.0 bitsandbytes==0.49.1 datasets==4.5.0 hf-xet==1.2.0 matplotlib==3.10.8 numpy==2.2.6 pyyaml==6.0.3 safetensors==0.7.0 sentencepiece==0.2.1 tokenizers==0.22.2 tiktoken==0.12.0 tqdm==4.67.1 transformers==4.57.6
```

- `requirements.clean.lock.txt` is a pinned dependency snapshot. Because PyTorch uses the CUDA wheel index, prefer the setup commands above over blindly running one unqualified `pip install -r`.
- Optional lm-eval install:

```powershell
.\.venv-seq\Scripts\python.exe -m pip install -r requirements.lm_eval.txt
```

```bash
python -m pip install -r requirements.lm_eval.txt
```

- LLMC/LightCompress is external and side-by-side, not inside this repo:
  - Windows path seen in docs: `E:\LightCompress`
  - WSL path seen in docs: `/mnt/e/LightCompress`
  - LLMC venv: `/mnt/e/LightCompress/.venv-llmc`
  - Activate with `source /mnt/e/LightCompress/.venv-llmc/bin/activate`
- Direct OmniQuant uses a separate environment named `omniquant-upstream`; see `OMNIQUANT_UPSTREAM_ENV_SETUP.md` and `third_party_quant/envs/omniquant-upstream.environment.yml`.

## Common commands

- Lightweight import/CUDA checks:

```powershell
.\.venv-seq\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
.\.venv-seq\Scripts\python.exe -c "import transformers, bitsandbytes, yaml, numpy; print('imports ok')"
```

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
python -c "import transformers, bitsandbytes, yaml, numpy; print('imports ok')"
```

- Validate LLMC smoke configs without running quantization:

```bash
python third_party_quant/validate_llmc_smoke_configs.py
```

- SEQ smoke run. This loads a model and can still take time/GPU memory:

```powershell
.\.venv-seq\Scripts\python.exe -m seq_core.pipeline --experiment main --experiments_file experiments.smoke.yaml
```

```bash
python -m seq_core.pipeline --experiment main --experiments_file experiments.smoke.yaml
```

- Main SEQ run. Long-running:

```powershell
.\.venv-seq\Scripts\python.exe -m seq_core.pipeline --experiment main
```

```bash
python -m seq_core.pipeline --experiment main
```

- Standalone canonical WikiText-2 PPL. Long-running for real models:

```powershell
.\.venv-seq\Scripts\python.exe -m benchmarks.ppl_eval --models "meta-llama/Llama-3.2-1B" --device auto --ppl_mode canonical --ppl_dataset wikitext2 --ppl_split test --ppl_full_corpus true --ppl_seq_len 2048
```

- Compare-matrix base PPL smoke:

```bash
python run_compare_matrix.py --models "facebook/opt-125m" --methods "base" --benchmarks "ppl" --experiments_file experiments.smoke.yaml --output_dir results
```

- Compare-matrix SEQ or lm-eval examples. Treat as long-running unless explicitly requested:

```bash
python run_compare_matrix.py --models "meta-llama/Llama-3.2-1B" --methods "base,seq" --benchmarks "ppl" --experiments_file experiments.smoke.yaml --output_dir results
python run_compare_matrix.py --models "meta-llama/Llama-3.2-1B" --methods "base" --benchmarks "hellaswag" --lm_eval_limit 5 --lm_eval_num_fewshot 0 --lm_eval_batch_size 1 --experiments_file experiments.smoke.yaml --output_dir results
```

- LLMC PPL-first compare shape. Long-running; run from WSL with the LLMC venv:

```bash
cd /mnt/e/SEQ_Clean
source /mnt/e/LightCompress/.venv-llmc/bin/activate
python run_compare_matrix.py --models "meta-llama/Llama-2-7b" --methods "base,seq,gptq_llmc,smoothquant_llmc,awq_llmc,rtn_llmc" --benchmarks "ppl" --experiments_file experiments.smoke.yaml --output_dir results --llmc_repo /mnt/e/LightCompress --llmc_venv /mnt/e/LightCompress/.venv-llmc --llmc_model_type Llama --llmc_save_mode none --llmc_calib_dataset wikitext2 --llmc_eval_dataset wikitext2 --llmc_calib_samples 32 --llmc_calib_seq_len 512 --llmc_eval_seq_len 512
```

- There is no configured formatter, linter, or test runner in the root. Use focused Python syntax/import checks or existing small validators rather than inventing project-wide test commands.

## Experiment workflow

1. Select a base model and method list. Current compare methods are `base`, `seq`, `omniquant`, `gptq_llmc`, `smoothquant_llmc`, `awq_llmc`, `rtn_llmc`, `llm_int8_llmc`, `spinquant_llmc`, and `omniquant_llmc`.
2. Choose calibration and evaluation settings from `experiments.yaml` or `experiments.smoke.yaml`.
3. Run base evaluation for PPL, size, latency/memory, lm-eval, or other requested metrics.
4. Run SEQ quantization/evaluation through `seq_core.pipeline` or `run_compare_matrix.py`.
5. Run external baselines only through their adapters; do not reimplement LLMC or OmniQuant internals in SEQ core.
6. Compare PPL, memory, disk size, effective bits, and benchmark metrics.
7. Save and inspect outputs under `runs/`, `reports/`, or `results/compare_real__.../`.

- Datasets and benchmarks present in config/code include WikiText-2 PPL, optional C4/PTB support in LLMC notes, MMLU, HellaSwag, ARC-Easy, ARC-Challenge, PIQA, Winogrande, LAMBADA OpenAI, local tail-risk/json/temperature/long-context probes, latency/memory, size, and quantization accounting.
- GSM8K, TruthfulQA, IFEval, HumanEval, and LongBench are not configured in the current repo evidence. Add them only with an explicit task and dependency plan.

## Coding conventions

- Prefer small, targeted edits. Avoid broad refactors unless explicitly requested.
- Preserve CLI argument names, output paths, JSON keys, CSV columns, and result schemas.
- Do not silently change metric definitions, calibration data, seeds, sequence lengths, or benchmark limits.
- Keep experiment outputs reproducible. Use existing `seeds.global` and `seeds.eval` fields when adding config-driven behavior.
- Keep `seq_core` free of external-baseline internals. LLMC and OmniQuant work belongs under `third_party_quant/`.
- Use structured parsers for YAML/JSON/CSV. Avoid ad hoc string manipulation for config and result files.
- Do not modify generated result files, model caches, checkpoints, or local model snapshots unless the task specifically asks.

## Evaluation and validation

- Before changing quantization or benchmark logic, run the smallest relevant smoke check available.
- Safe local checks include:

```bash
python third_party_quant/validate_llmc_smoke_configs.py
python -m py_compile run_compare_matrix.py seq_core/pipeline.py benchmarks/evaluation_suite.py benchmarks/seq_lm_eval.py
```

- For SEQ logic changes, prefer `experiments.smoke.yaml` before `experiments.yaml`.
- For compare-matrix changes, start with `facebook/opt-125m`, `--benchmarks "ppl"`, and one method.
- For lm-eval changes, start with `--benchmarks "hellaswag" --lm_eval_limit 5 --lm_eval_batch_size 1`.
- Do not run full LLM benchmarks, gated-model downloads, Llama-2/3 multi-hour jobs, or LLMC/OmniQuant real quantization unless explicitly requested.
- If GPU, model access, token, or memory limits prevent validation, state that in the final response and list the command that should be run.

## Data, model, and credential safety

- Never commit Hugging Face tokens, API keys, local cache paths, model weights, generated checkpoints, or large datasets.
- Use environment variables or the existing Hugging Face auth flow for tokens.
- Gated Hugging Face models such as Meta Llama may fail because of access/token permission, not because of SEQ code.
- Keep generated outputs in ignored paths: `artifacts/`, `runs/`, `reports/`, `results/`, `results_*`, `compare_real__*`, and LLMC save directories.
- Do not delete experiment results, model caches, checkpoints, or downloaded model snapshots unless explicitly asked.

## Known pitfalls

- There is no root `README.md` in this checkout. Use `SETUP_CLEAN_SEQ.md`, `docs/`, and the current code as primary local guidance.
- Some lm-eval docs are stale relative to `seq_core.pipeline`: `pipeline.py` currently does not accept CLI flags such as `--metrics`, `--seq-only`, `--lm-eval-only`, or `--lm-eval-tasks`. Use `run_compare_matrix.py` for those comparison/lm-eval workflows, or enable lm-eval through config.
- Bare Windows `python.exe` may not have torch/lm-eval installed. Prefer `.venv-seq` for SEQ or the WSL LLMC venv for compare-matrix/LLMC work.
- `requirements.clean.lock.txt` expects `huggingface_hub==0.36.2`; mixing newer `huggingface-hub` with `transformers==4.57.6` can break imports.
- CUDA is required for the bitsandbytes INT4/INT8 path. If `torch.cuda.is_available()` is false, fix the environment before debugging quantization code.
- Full-corpus canonical PPL and lm-eval tasks can be slow and memory-heavy.
- LLMC save paths must be fresh; the adapter normally removes stale generated save paths, while static upstream smoke configs may fail if their save path already exists.
- `llm_int8_llmc` is implemented but documented as having a current OPT smoke failure. `spinquant_llmc` and `omniquant_llmc` are recognized but intentionally disabled/not recommended in current validation notes.
- Direct OmniQuant depends on the pinned `third_party_quant/OmniQuant/` checkout and a separate `omniquant-upstream` environment; AutoGPTQ real-quant support may require compiler/CUDA toolkit setup.
- On Windows, the upstream OmniQuant checkout has a case-only image filename collision under `imgs/`; do not treat that as a SEQ algorithm change.
- WSL paths are common in LLMC docs and configs. Convert carefully between `E:\SEQ_Clean` and `/mnt/e/SEQ_Clean`.

## Agent behavior rules

- First inspect relevant files before editing. Prefer `rg` and targeted reads.
- Make the smallest safe change and keep it within the requested scope.
- Explain changed files and validation performed.
- Do not fabricate benchmark results. Report only observed metrics or clearly label commands as not run.
- Do not modify generated result files unless the task specifically asks for it.
- Ask before adding new heavy dependencies or new external checkouts.
- For long-running commands, provide the command but do not run it automatically unless explicitly instructed.
