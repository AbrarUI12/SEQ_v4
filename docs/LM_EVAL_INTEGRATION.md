# lm-eval Harness Integration

SEQ keeps its built-in evaluation suite by default. EleutherAI lm-evaluation-harness is an optional second evaluation layer for academic tasks such as HellaSwag, ARC, PIQA, Winogrande, and LAMBADA.

## Install

```powershell
.\.venv-seq\Scripts\python.exe -m pip install -r requirements.lm_eval.txt
```

The base SEQ environment does not require lm-eval. If lm-eval is requested but missing, SEQ writes machine-readable skipped/error files under the run's `lm_eval` directory according to `fail_policy`.

## Metric Selection

Metric groups can be selected from config or CLI:

- `seq_core`
- `ppl`
- `tail_risk`
- `json_stress`
- `temperature_sweep`
- `long_context`
- `latency_memory`
- `size`
- `quant_accounting`
- `lm_eval`
- `all`

CLI overrides config:

```powershell
.\.venv-seq\Scripts\python.exe -m seq_core.pipeline --experiment main --metrics seq_core,lm_eval,latency_memory
.\.venv-seq\Scripts\python.exe -m seq_core.pipeline --experiment main --skip-metrics long_context,temperature_sweep
.\.venv-seq\Scripts\python.exe -m seq_core.pipeline --experiment main --seq-only
.\.venv-seq\Scripts\python.exe -m seq_core.pipeline --experiment main --lm-eval-only --lm-eval-tasks hellaswag --lm-eval-limit 5
```

Default behavior is unchanged: SEQ built-in metrics run and lm-eval is disabled unless config or CLI enables it.

## lm-eval Presets

`experiments.yaml` defines:

- `smoke`: `hellaswag`, limit 10
- `standard`: `hellaswag`, `arc_easy`, `arc_challenge`, `piqa`, `winogrande`
- `paper`: standard plus `lambada_openai`

Examples:

```powershell
.\.venv-seq\Scripts\python.exe -m seq_core.pipeline --experiment main --lm-eval --lm-eval-preset smoke
.\.venv-seq\Scripts\python.exe -m seq_core.pipeline --experiment main --lm-eval --lm-eval-tasks hellaswag,arc_easy,piqa --lm-eval-limit 10
```

## Outputs

When lm-eval is requested, each eval directory contains:

- `lm_eval/lm_eval_raw.json`
- `lm_eval/lm_eval_summary.json`
- `lm_eval/lm_eval_command.txt`
- `lm_eval/lm_eval_status.json`

The normalized summary includes stable flattened keys such as:

```text
lm_eval__hellaswag__acc
lm_eval__hellaswag__acc_norm
lm_eval__arc_easy__acc
lm_eval__status
lm_eval__tasks
```

## Quantized Model Limitation

SEQ quantization currently replaces modules in memory with bitsandbytes-backed modules. lm-eval's Hugging Face backend normally reloads a model from `pretrained=<name-or-path>`, and the saved SEQ quantized checkpoint is not a safe drop-in reload target for that path.

By default:

- baseline lm-eval uses the Hugging Face model name/path;
- SEQ quantized lm-eval uses the already-instantiated in-memory quantized model through `lm_eval.models.huggingface.HFLM`;
- `evaluation.lm_eval.quantized_reloadable` still means only: the saved artifact is safe to reload through the CLI/path-based Hugging Face route.

This preserves the working SEQ quantized runtime path and avoids relying on an unsafe checkpoint reload.

`run_compare_matrix.py` uses the same policy for `seq`: `base` continues to use the CLI/path route, while `seq` consumes the in-memory lm-eval summary produced by the SEQ pipeline.

## Smoke Commands

SEQ-only smoke:

```powershell
.\.venv-seq\Scripts\python.exe -m seq_core.pipeline --experiment main --experiments_file experiments.smoke.yaml --seq-only
```

lm-eval smoke on the baseline model:

```powershell
.\.venv-seq\Scripts\python.exe -m seq_core.pipeline --experiment main --experiments_file experiments.smoke.yaml --lm-eval-only --lm-eval-tasks hellaswag --lm-eval-limit 5
```

Combined smoke:

```powershell
.\.venv-seq\Scripts\python.exe -m seq_core.pipeline --experiment main --experiments_file experiments.smoke.yaml --metrics seq_core,lm_eval,latency_memory,quant_accounting --lm-eval-tasks hellaswag --lm-eval-limit 5
```

Missing lm-eval behavior:

```powershell
.\.venv-seq\Scripts\python.exe -m seq_core.pipeline --experiment main --experiments_file experiments.smoke.yaml --lm-eval --lm-eval-fail-policy warn --lm-eval-tasks hellaswag --lm-eval-limit 5
```

## Compare Runner Commands

Compare-matrix baseline lm-eval:

```powershell
python run_compare_matrix.py --models "meta-llama/Llama-3.2-1B" --device auto --methods "base" --benchmarks "hellaswag" --lm_eval_limit 5 --lm_eval_num_fewshot 0 --lm_eval_batch_size 1 --experiments_file experiments.smoke.yaml --output_dir "results/${TS}"
python run_compare_matrix.py --models "meta-llama/Llama-3.2-1B" --device auto --methods "base" --benchmarks "hellaswag,arc_easy,piqa" --lm_eval_num_fewshot 0 --lm_eval_batch_size 1 --experiments_file experiments.smoke.yaml --output_dir "results/${TS}"
```

Compare-matrix SEQ quantized lm-eval:

```powershell
python run_compare_matrix.py --models "meta-llama/Llama-3.2-1B" --device auto --methods "seq" --benchmarks "hellaswag" --lm_eval_limit 5 --lm_eval_num_fewshot 0 --lm_eval_batch_size 1 --experiments_file experiments.smoke.yaml --output_dir "results/${TS}"
```

For `seq`, the runner performs the real SEQ quantization pipeline and then evaluates the live quantized model in memory. Successful summaries report `lm_eval_source: in_memory_hflm`.
