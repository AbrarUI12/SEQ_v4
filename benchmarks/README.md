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

