# SEQ vs baselines — actual-bits comparison

Points from SEQ sweeps + external baselines, sorted by **weight-only bits/param** — quantized linear weights plus their inline overhead (group scales/zeros, FP16/INT8 protection residual, channel index), divided by the quantized-linear parameter count. Embeddings, lm_head, norms and biases are FP16 in every method, common to the axis, and excluded — so this axis is directly comparable to GPTQ-4 = 4.0. ★ = on the Pareto frontier (no method has both fewer bits and lower PPL).

## meta-llama/Llama-3.2-1B  (FP16 PPL 9.7572)

Axis = **weight-only bits/param** (quantized linear weights + inline overhead; FP16 embeddings/lm_head/norms excluded, common to all methods — so it is comparable to GPTQ-4 = 4.0). *full-model bits* is the deployment average including FP16 embeddings, shown for reference only, not the frontier axis.

| method | weight bits | nominal bits | full-model bits | PPL | Δ vs FP16 | frontier |
|---|---|---|---|---|---|---|
| AWQ-4 g128 | 4.00 | 4.00 | — | 11.278 | +1.520 | ★ |
| RTN-4 g128 | 4.00 | 4.00 | — | 11.710 | +1.953 |  |
| GPTQ-4 g128 | 4.29 | 4.00 | 6.78 | 10.363 | +0.606 | ★ |
| HQQ-4 uniform | 4.50 | 4.00 | — | 11.187 | +1.430 |  |
| HQQ-5 uniform | 5.00 | 5.00 | — | 10.064 | +0.306 | ★ |
| HQQ-6 uniform | 6.00 | 6.00 | — | 9.829 | +0.072 | ★ |
| HQQ-8 uniform | 8.00 | 8.00 | — | 9.762 | +0.005 | ★ |
| FP16 | 16.00 | 16.00 | — | 9.757 | +0.000 | ★ |

> **No SEQ point is on the frontier** — a baseline dominates it.

## meta-llama/Llama-3.2-3B  (FP16 PPL 7.8167)

Axis = **weight-only bits/param** (quantized linear weights + inline overhead; FP16 embeddings/lm_head/norms excluded, common to all methods — so it is comparable to GPTQ-4 = 4.0). *full-model bits* is the deployment average including FP16 embeddings, shown for reference only, not the frontier axis.

| method | weight bits | nominal bits | full-model bits | PPL | Δ vs FP16 | frontier |
|---|---|---|---|---|---|---|
| HQQ-4 uniform | 4.00 | 4.00 | — | 8.387 | +0.570 | ★ |
| AWQ-4 g128 | 4.00 | 4.00 | — | 8.405 | +0.588 |  |
| RTN-4 g128 | 4.00 | 4.00 | — | 8.498 | +0.682 |  |
| GPTQ-4 g128 | 4.28 | 4.00 | 5.72 | 8.304 | +0.487 | ★ |
| HQQ-5 uniform | 7.46 | 5.00 | — | 7.957 | +0.140 | ★ |
| HQQ-6 uniform | 8.46 | 6.00 | — | 7.845 | +0.028 | ★ |
| HQQ-8 uniform | 10.46 | 8.00 | — | 7.820 | +0.003 | ★ |
| FP16 | 16.00 | 16.00 | — | 7.817 | +0.000 | ★ |

> **No SEQ point is on the frontier** — a baseline dominates it.
