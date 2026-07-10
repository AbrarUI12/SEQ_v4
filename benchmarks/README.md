# Benchmark Support Package

This folder contains evaluation, benchmark, plotting, and report helpers used by
the SEQ pipeline.

## Module Map

- `core.py`: perplexity, parameter counts, disk-size summaries, bench summaries
- `evaluation_suite.py`: baseline and quantized evaluation suite
- `eval_config.py`: metric and lm-eval configuration helpers
- `metrics_utils.py`: small metric helpers
- `multiple_choice_eval.py`: optional MMLU and zero-shot task evaluation
- `plotting.py`: run figures
- `reporting.py`: markdown/JSON report assembly
- `ppl_eval.py`: standalone PPL runner
- `seq_lm_eval.py`: optional EleutherAI lm-evaluation-harness adapter

## Standalone PPL

```powershell
.\.venv-seq\Scripts\python.exe -m benchmarks.ppl_eval `
  --models "meta-llama/Llama-3.2-1B" `
  --device auto `
  --ppl_mode canonical `
  --ppl_dataset wikitext2 `
  --ppl_split test `
  --ppl_full_corpus true `
  --ppl_seq_len 2048
```

## Compare Matrix

Run from the repository root:

```powershell
.\.venv-seq\Scripts\python.exe run_compare_matrix.py `
  --models "meta-llama/Llama-3.2-1B" `
  --device auto `
  --methods "seq,omniquant" `
  --benchmarks "hellaswag,arc_easy,piqa" `
  --lm_eval_batch_size 1 `
  --degeneracy_mode old
```

`seq` is executed through `seq_core/`. `omniquant` is executed through the
pinned upstream checkout in `third_party_quant/OmniQuant/`. Benchmark names are
EleutherAI lm-evaluation-harness task names.
