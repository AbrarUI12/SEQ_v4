# F5 — Downstream task accuracy at the paper operating points

Zero-shot accuracy (acc_norm where the task reports it, else acc), WikiText-2-matched checkpoints. Higher is better. `avg` is the macro-average across tasks.

## meta-llama/Llama-3.2-1B

| point | bits | arc_challenge | arc_easy | hellaswag | lambada_openai | piqa | winogrande | avg | note |
|---|---|---|---|---|---|---|---|---|---|
| fp16 | 16 | 31.50 | 61.50 | 58.00 | 60.00 | 76.50 | 60.50 | 58.00 | FP16 reference |
| hqq4 | 4 | 35.00 | 59.00 | 54.50 | 56.50 | 75.50 | 62.50 | 57.17 | data-free RTN/HQQ-4 base (frac 0 = no protection) |
| best_hqq | 7.7 | 32.00 | 64.00 | 57.50 | 61.00 | 75.00 | 63.50 | 58.83 | best SEQ point on the data-free HQQ base (F1) |

### Paired contrasts (Δ accuracy points, 95% CI)

- **best_hqq − hqq4** (protection helps a data-free base (F1)): macro Δ = +1.67 [-0.08, +3.33] pts — paired bootstrap.

## meta-llama/Llama-3.2-3B

| point | bits | arc_challenge | arc_easy | hellaswag | lambada_openai | piqa | winogrande | avg | note |
|---|---|---|---|---|---|---|---|---|---|
| fp16 | 16 | 46.42 | 72.14 | 74.16 | 70.17 | 78.07 | 69.61 | 68.43 | FP16 reference |
| hqq4 | 4 | 45.31 | 70.03 | 72.34 | 67.48 | 76.61 | 68.59 | 66.72 | data-free RTN/HQQ-4 base (frac 0 = no protection) |
| best_hqq | 7.7 | 45.99 | 71.63 | 73.34 | 69.63 | 77.86 | 69.14 | 67.93 | best SEQ point on the data-free HQQ base (F1) |

### Paired contrasts (Δ accuracy points, 95% CI)

- **best_hqq − hqq4** (protection helps a data-free base (F1)): macro Δ = +1.21 [+0.77, +1.63] pts — paired bootstrap.
