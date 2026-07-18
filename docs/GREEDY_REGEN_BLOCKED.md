# Corrected greedy regeneration — blocked preflight

Date: 2026-07-18. The old `runs/final_hqq_greedy` and
`runs/final_gptqgreedy` outputs are preserved and excluded from the corrected
comparison because they were generated before the greedy-order fix.

## Discovered checkpoints

| model | exact checkpoint path | status |
|---|---|---|
| `meta-llama/Llama-3.2-3B` | `runs/final_llmc/Llama-3.2-3B/gptq/artifacts/fake_quant_model` | valid (`config.json`, `model.safetensors`) |
| `meta-llama/Llama-3.2-1B` | `runs/final_llmc/Llama-3.2-1B/gptq/artifacts/fake_quant_model` | valid (`config.json`, `model.safetensors`) |
| `meta-llama/Llama-3.1-8B` | — | **missing** |

No Llama-3.1-8B fake-quant checkpoint exists under `runs/`; its GPTQ run
directory contains only `channel_pareto.json` and Markdown output.

## Commands executed

```bash
cd /mnt/d/Abrar/SEQ/seq_v4
source .venv-seq/bin/activate
python -m pytest -q
python analysis/build_comparison.py --help
python analysis/plot_final_results.py --help
python scripts/enrich_llmc_baseline_storage.py \
  --input results/final_baselines_comparison.json \
  --output results/final_baselines_weight_only.json --bits 4
python analysis/build_comparison.py \
  --sweeps runs/final_hqq_residual_accounted runs/final_gptqbase \
    runs/final_gptq_actscale runs/final_hqq_uniform runs/final_hqq_value \
  --baselines results/final_baselines_weight_only.json \
  --signals act_max,residual_rms,residual_max,greedy,tier_alloc,random,act_scale \
  --out docs/COMPARISON.md --csv results/final_comparison.csv \
  --json results/final_comparison.json
python analysis/plot_final_results.py --input results/final_comparison.csv \
  --output-dir figures/final_corrected
scripts/run_fixed_greedy_sweeps.sh
```

The final command stops during preflight with:

```text
MISSING GPTQ CHECKPOINT MAPPING: meta-llama/Llama-3.1-8B
```

No GPU greedy sweep was launched. The exact preflight output is preserved in
`runs/final_greedy_fixed/run.log`.
