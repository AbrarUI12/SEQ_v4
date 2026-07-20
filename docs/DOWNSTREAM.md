# F5 — Downstream task accuracy at the paper operating points

Zero-shot accuracy (acc_norm where the task reports it, else acc), WikiText-2-matched checkpoints. Higher is better. `avg` is the macro-average across tasks.

## meta-llama/Llama-3.2-1B

| point | bits | arc_challenge | arc_easy | hellaswag | lambada_openai | piqa | winogrande | avg | note |
|---|---|---|---|---|---|---|---|---|---|
| fp16 | 16 | 31.50 | 61.50 | 58.00 | 60.00 | 76.50 | 60.50 | 58.00 | FP16 reference |
| gptq4 | 4 | 34.98 | 61.62 | 62.11 | 58.06 | 73.56 | 59.75 | 58.35 | LightCompress GPTQ-4 base (strong error-compensated base) |
| hqq4 | 4 | 35.00 | 59.00 | 54.50 | 56.50 | 75.50 | 62.50 | 57.17 | data-free RTN/HQQ-4 base (frac 0 = no protection) |
| resmax_gptq | 4.82 | 35.32 | 61.41 | 61.81 | 60.62 | 73.94 | 61.25 | 59.06 | safe activation-magnitude protection on GPTQ (3B Pareto frontier point) |
| greedy_gptq | 4.82 | 35.58 | 60.82 | 61.97 | 59.01 | 73.45 | 61.17 | 58.67 | CATASTROPHIC control: residual-driven greedy on GPTQ (confirms F3 antagonism downstream) |
| best_hqq | 7.7 | 32.00 | 64.00 | 57.50 | 61.00 | 75.00 | 63.50 | 58.83 | best SEQ point on the data-free HQQ base (F1) |

### Paired contrasts (Δ accuracy points, 95% CI)

- **resmax_gptq − gptq4** (safe activation-magnitude protection is >= GPTQ-4 (F4)): macro Δ = +0.71 [+0.28, +1.16] pts — paired bootstrap.
- **greedy_gptq − gptq4** (residual-driven protection is catastrophic on GPTQ (F3)): macro Δ = +0.32 [-0.07, +0.73] pts — paired bootstrap.
- **best_hqq − hqq4** (protection helps a data-free base (F1)): macro Δ = +1.67 [-0.08, +3.33] pts — paired bootstrap.

## meta-llama/Llama-3.2-3B

| point | bits | arc_challenge | arc_easy | hellaswag | lambada_openai | piqa | winogrande | avg | note |
|---|---|---|---|---|---|---|---|---|---|
| fp16 | 16 | 46.42 | 72.14 | 74.16 | 70.17 | 78.07 | 69.61 | 68.43 | FP16 reference |
| gptq4 | 4 | 41.00 | 67.50 | 66.50 | 70.50 | 75.50 | 66.50 | 64.58 | LightCompress GPTQ-4 base (strong error-compensated base) |
| hqq4 | 4 | 45.31 | 70.03 | 72.34 | 67.48 | 76.61 | 68.59 | 66.72 | data-free RTN/HQQ-4 base (frac 0 = no protection) |
| resmax_gptq | 4.82 | 40.50 | 70.50 | 65.50 | 70.50 | 76.50 | 67.50 | 65.17 | safe activation-magnitude protection on GPTQ (3B Pareto frontier point) |
| greedy_gptq | 4.82 | 41.50 | 68.50 | 65.50 | 73.00 | 76.00 | 69.50 | 65.67 | CATASTROPHIC control: residual-driven greedy on GPTQ (confirms F3 antagonism downstream) |
| best_hqq | 7.7 | 45.99 | 71.63 | 73.34 | 69.63 | 77.86 | 69.14 | 67.93 | best SEQ point on the data-free HQQ base (F1) |

### Paired contrasts (Δ accuracy points, 95% CI)

- **resmax_gptq − gptq4** (safe activation-magnitude protection is >= GPTQ-4 (F4)): macro Δ = +0.58 [-0.83, +1.92] pts — paired bootstrap.
- **greedy_gptq − gptq4** (residual-driven protection is catastrophic on GPTQ (F3)): macro Δ = +1.08 [-0.08, +2.25] pts — paired bootstrap.
- **best_hqq − hqq4** (protection helps a data-free base (F1)): macro Δ = +1.21 [+0.77, +1.63] pts — paired bootstrap.
