# SEQ Core Package

This folder contains the SEQ-owned implementation.

## Entry Point

Run from the repository root:

```powershell
.\.venv-seq\Scripts\python.exe -m seq_core.pipeline --experiment main
```

Smoke run:

```powershell
.\.venv-seq\Scripts\python.exe -m seq_core.pipeline --experiment main --experiments_file experiments.smoke.yaml
```

## Module Map

- `pipeline.py`: main SEQ orchestration
- `entropy_metrics.py`: weight and activation entropy
- `precision_policy.py`: INT4/INT8/FP16 tier policy and protections
- `quantize_model.py`: bitsandbytes replacement and effective-bit accounting
- `evaluation_suite.py`: baseline and quantized evaluation suite
- `benchmarks.py`: perplexity, parameter counts, disk-size summaries
- `metrics_utils.py`: small metric helpers
- `multiple_choice_eval.py`: optional MMLU and zero-shot task evaluation
- `plotting.py`: run figures
- `reporting.py`: markdown/JSON report assembly
- `ppl_eval.py`: standalone PPL runner using the same canonical/proxy implementation
- `compare_methods.py`: external compare-method orchestration such as pinned upstream OmniQuant

Third-party methods belong outside this folder, under `third_party_quant/`.

## Standalone PPL

Canonical WikiText-2 full-corpus PPL:

```powershell
.\.venv-seq\Scripts\python.exe -m seq_core.ppl_eval `
  --models "meta-llama/Llama-3.2-1B" `
  --device auto `
  --ppl_mode canonical `
  --ppl_dataset wikitext2 `
  --ppl_split test `
  --ppl_full_corpus true `
  --ppl_seq_len 2048
```
