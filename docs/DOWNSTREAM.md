# F5 — Downstream task accuracy at the paper operating points

Zero-shot accuracy (acc_norm where the task reports it, else acc), WikiText-2-matched checkpoints. Higher is better. `avg` is the macro-average across tasks.

## meta-llama/Llama-3.2-1B

| point | bits | arc_challenge | arc_easy | hellaswag | lambada_openai | piqa | winogrande | avg | note |
|---|---|---|---|---|---|---|---|---|---|
| fp16 | 16 | 31.50 | 61.50 | 58.00 | 60.00 | 76.50 | 60.50 | 58.00 | FP16 reference |
| gptq4 | 4 | 34.98 | 61.62 | 62.11 | 58.06 | 73.56 | 59.75 | 58.35 | LightCompress GPTQ-4 base (strong error-compensated base) |
| hqq4 | 4 | 35.00 | 59.00 | 54.50 | 56.50 | 75.50 | 62.50 | 57.17 | data-free RTN/HQQ-4 base (frac 0 = no protection) |
| resmax_gptq | 4.82 | 35.32 | 61.41 | 61.81 | 60.62 | 73.94 | 61.25 | 59.06 | safe activation-magnitude protection on GPTQ (3B Pareto frontier point) |
| greedy_gptq | 4.82 | 35.58 | 60.82 | 61.97 | 59.01 | 73.45 | 61.17 | 58.67 | F3 probe: residual-driven greedy on GPTQ. Catastrophic in PPL and REPRODUCES on the regenerated base (63.95 on 1B / 51.68 on 3B) with verify_materialized delta~0 => not an export bug; but Llama-2-7B is healthy => F3 is model-dependent. Downstream-healthy is being closed by reloading the on-disk checkpoint - see FINDINGS_PAPER 7.2 |
| best_hqq | 7.7 | 32.00 | 64.00 | 57.50 | 61.00 | 75.00 | 63.50 | 58.83 | best SEQ point on the data-free HQQ base (F1) |

### Paired contrasts (Δ accuracy points, 95% CI)

- **resmax_gptq − gptq4** (safe activation-magnitude protection is >= GPTQ-4 (F4)): macro Δ = +0.71 [+0.28, +1.16] pts — paired bootstrap.
- **greedy_gptq − gptq4** (F3 pre-registration (catastrophic on GPTQ) - FALSIFIED downstream, see FINDINGS_PAPER 7.2): macro Δ = +0.32 [-0.07, +0.73] pts — paired bootstrap.
- **best_hqq − hqq4** (protection helps a data-free base (F1)): macro Δ = +1.67 [-0.08, +3.33] pts — paired bootstrap.

## meta-llama/Llama-3.2-3B

| point | bits | arc_challenge | arc_easy | hellaswag | lambada_openai | piqa | winogrande | avg | note |
|---|---|---|---|---|---|---|---|---|---|
| fp16 | 16 | 46.42 | 72.14 | 74.16 | 70.17 | 78.07 | 69.61 | 68.43 | FP16 reference |
| gptq4 | 4 | 44.20 | 69.91 | 73.24 | 68.54 | 77.04 | 69.69 | 67.10 | LightCompress GPTQ-4 base (strong error-compensated base) |
| hqq4 | 4 | 45.90 | 69.78 | 72.62 | 68.04 | 76.39 | 68.43 | 66.86 | data-free RTN/HQQ-4 base (frac 0 = no protection) |
| resmax_gptq | 4.82 | 45.31 | 72.35 | 73.27 | 68.45 | 76.39 | 69.06 | 67.47 | safe activation-magnitude protection on GPTQ (3B Pareto frontier point) |
| greedy_gptq | 4.82 | 44.80 | 71.46 | 73.25 | 69.05 | 77.26 | 69.85 | 67.61 | F3 probe: residual-driven greedy on GPTQ. Catastrophic in PPL and REPRODUCES on the regenerated base (63.95 on 1B / 51.68 on 3B) with verify_materialized delta~0 => not an export bug; but Llama-2-7B is healthy => F3 is model-dependent. Downstream-healthy is being closed by reloading the on-disk checkpoint - see FINDINGS_PAPER 7.2 |
| best_hqq | 7.7 | 46.67 | 72.05 | 73.39 | 69.53 | 77.97 | 69.22 | 68.14 | best SEQ point on the data-free HQQ base (F1) |
| random_hqq | 7.7 | 45.90 | 70.29 | 73.03 | 69.11 | 76.61 | 68.35 | 67.21 | matched-bit random-channel control on HQQ (7.70 bits, seed 1234). Pairs with best_hqq for the matched-bit F1 test (paper 3). random_hqq{,_s2,_s3} give channel-set uncertainty. |

### Paired contrasts (Δ accuracy points, 95% CI)

- **resmax_gptq − gptq4** (safe activation-magnitude protection is >= GPTQ-4 (F4)): macro Δ = +0.37 [-0.06, +0.77] pts — paired bootstrap.
- **greedy_gptq − gptq4** (F3 pre-registration (catastrophic on GPTQ) - FALSIFIED downstream, see FINDINGS_PAPER 7.2): macro Δ = +0.51 [+0.13, +0.86] pts — paired bootstrap.
- **best_hqq − hqq4** (protection helps a data-free base (F1)): macro Δ = +1.28 [+0.84, +1.68] pts — paired bootstrap.
- **best_hqq − random_hqq** (matched-bit (7.70) activation-signal protection beats a random-channel control downstream (F1 sharpening, mirrors paper 3)): macro Δ = +0.92 [+0.50, +1.34] pts — paired bootstrap.
