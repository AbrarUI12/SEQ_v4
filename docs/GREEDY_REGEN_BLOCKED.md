# Corrected greedy regeneration — now runs on available models (1B, 3B)

Date: 2026-07-18 (updated). The old `runs/final_hqq_greedy` /
`runs/final_gptqgreedy` outputs are preserved and excluded from the corrected
comparison because they were generated before the greedy-order fix.

## What changed

The earlier preflight aborted the **entire** rerun because Llama-3.1-8B has no
LLMC fake-quant checkpoint. `scripts/run_fixed_greedy_sweeps.sh` now **warns and
skips** any model without a valid checkpoint and proceeds with the rest, so the
decisive interaction-aware selection gate runs on **1B and 3B** without waiting on
8B. The runner also evaluates both selection modes on both bases:

- `--select greedy` — interaction-aware, iterative
- `--select greedy_indep` — the same objective's first-step gains, **no** iterative
  update (the interaction-free ablation added in `seq_core/greedy_select.py`)

on the GPTQ base (`gptq_llmc`) and the data-free HQQ base, at fractions
`0,0.02,0.05,0.1,0.2`.

## Discovered checkpoints

| model | checkpoint path | status |
|---|---|---|
| `meta-llama/Llama-3.2-3B` | `runs/final_llmc/Llama-3.2-3B/gptq/artifacts/fake_quant_model` | valid → **runs** |
| `meta-llama/Llama-3.2-1B` | `runs/final_llmc/Llama-3.2-1B/gptq/artifacts/fake_quant_model` | valid → **runs** |
| `meta-llama/Llama-3.1-8B` | — | missing → **skipped** (deferred until the gate passes on 1B/3B) |

## Run the gate

```bash
cd "/path/to/SEQ-clean-v4" && source .venv-seq/bin/activate
bash scripts/run_fixed_greedy_sweeps.sh \
  --gptq-root runs/final/llmc --output-root runs/final/sweeps --resume
# then regenerate the table on CPU with the new selectors included:
python analysis/build_comparison.py \
  --sweeps runs/final_greedy_fixed runs/final_hqq_residual_accounted runs/final_gptqbase \
    runs/final_gptq_actscale runs/final_hqq_uniform runs/final_hqq_value \
  --baselines results/final_baselines_weight_only.json \
  --signals greedy,greedy_indep,residual_max,residual_rms,act_max,act_scale,random \
  --out docs/COMPARISON.md --csv results/final_comparison.csv --json results/final_comparison.json
python analysis/plot_final_results.py --input results/final_comparison.csv --output-dir figures/final_corrected
```

## Decision rule (the gate)

At matched actual weight bits, on **both** 1B and 3B across budgets: if `greedy`
does not consistently beat `greedy_indep`, `residual_max`, and `random`, the
iterative cross-column interaction term buys nothing → report it as a negative /
ablation result and make the **audit** the paper. If it does beat them cleanly →
acquire the 8B checkpoint and add OWQ / CLAQ / SpQR column baselines + latency.
