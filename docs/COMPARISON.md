# SEQ vs baselines — actual-bits comparison

Points from SEQ sweeps + external baselines, sorted by **weight-only bits/param** — quantized linear weights plus their inline overhead (group scales/zeros, FP16/INT8 protection residual, channel index), divided by the quantized-linear parameter count. Embeddings, lm_head, norms and biases are FP16 in every method, common to the axis, and excluded — so this axis is directly comparable to GPTQ-4 = 4.0. ★ = on the Pareto frontier (no method has both fewer bits and lower PPL).

## meta-llama/Llama-3.2-1B  (FP16 PPL 9.7572)

Axis = **weight-only bits/param** (quantized linear weights + inline overhead; FP16 embeddings/lm_head/norms excluded, common to all methods — so it is comparable to GPTQ-4 = 4.0). *full-model bits* is the deployment average including FP16 embeddings, shown for reference only, not the frontier axis.

| method | weight bits | nominal bits | full-model bits | PPL | Δ vs FP16 | frontier |
|---|---|---|---|---|---|---|
| AWQ-4 g128 | 4.00 | 4.00 | — | 11.278 | +1.520 | ★ |
| RTN-4 g128 | 4.00 | 4.00 | — | 11.710 | +1.953 |  |
| GPTQ-4 g128 | 4.29 | 4.00 | 6.78 | 10.363 | +0.606 | ★ |
| SEQ:act_scale(gptq_llmc-4b k=0.0) | 4.50 | 4.00 | 7.90 | 10.557 | +0.800 |  |
| SEQ:act_max(gptq_llmc-4b k=0.0) | 4.50 | 4.00 | 7.90 | 10.557 | +0.800 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.0) | 4.50 | 4.00 | 7.90 | 10.557 | +0.800 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.0) | 4.50 | 4.00 | 7.90 | 10.557 | +0.800 |  |
| SEQ:random(gptq_llmc-4b k=0.0) | 4.50 | 4.00 | 7.90 | 10.557 | +0.800 |  |
| SEQ:act_max(hqq-4b k=0.0) | 4.50 | 4.00 | 7.90 | 11.187 | +1.430 |  |
| SEQ:residual_rms(hqq-4b k=0.0) | 4.50 | 4.00 | 7.90 | 11.187 | +1.430 |  |
| SEQ:residual_max(hqq-4b k=0.0) | 4.50 | 4.00 | 7.90 | 11.187 | +1.430 |  |
| SEQ:random(hqq-4b k=0.0) | 4.50 | 4.00 | 7.90 | 11.187 | +1.430 |  |
| SEQ:act_max(hqq-4b k=0.0) | 4.50 | 4.00 | 4.50 | 11.187 | +1.430 |  |
| HQQ-4 uniform | 4.50 | 4.00 | — | 11.187 | +1.430 |  |
| SEQ:tier_alloc(hqq-4b [budget=0.25]) | 4.76 | 4.07 | 8.16 | 10.631 | +0.873 |  |
| SEQ:random(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 8.22 | 10.412 | +0.654 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 8.22 | 10.423 | +0.666 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 8.22 | 10.434 | +0.677 |  |
| SEQ:act_max(hqq-4b k=0.02) | 4.82 | 4.24 | 8.22 | 10.516 | +0.759 |  |
| SEQ:residual_max(hqq-4b k=0.02) | 4.82 | 4.24 | 8.22 | 10.523 | +0.766 |  |
| SEQ:residual_rms(hqq-4b k=0.02) | 4.82 | 4.24 | 8.22 | 10.615 | +0.858 |  |
| SEQ:act_max(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 8.22 | 10.711 | +0.954 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 8.22 | 10.930 | +1.172 |  |
| SEQ:random(hqq-4b k=0.02) | 4.82 | 4.24 | 8.22 | 11.160 | +1.403 |  |
| HQQ-5 uniform | 5.00 | 5.00 | — | 10.064 | +0.306 | ★ |
| SEQ:tier_alloc(hqq-4b [budget=0.5]) | 5.01 | 4.13 | 8.41 | 10.592 | +0.835 |  |
| SEQ:random(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 8.71 | 10.413 | +0.656 |  |
| SEQ:residual_max(hqq-4b k=0.05) | 5.30 | 4.60 | 8.71 | 10.422 | +0.665 |  |
| SEQ:act_max(hqq-4b k=0.05) | 5.30 | 4.60 | 8.71 | 10.422 | +0.665 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 8.71 | 10.436 | +0.678 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 8.71 | 10.436 | +0.679 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 8.71 | 10.476 | +0.719 |  |
| SEQ:residual_rms(hqq-4b k=0.05) | 5.30 | 4.60 | 8.71 | 10.568 | +0.810 |  |
| SEQ:act_max(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 8.71 | 10.687 | +0.930 |  |
| SEQ:random(hqq-4b k=0.05) | 5.30 | 4.60 | 8.71 | 11.132 | +1.375 |  |
| SEQ:tier_alloc(hqq-4b [budget=1.0]) | 5.53 | 4.27 | 8.93 | 10.555 | +0.798 |  |
| HQQ-6 uniform | 6.00 | 6.00 | — | 9.829 | +0.072 | ★ |
| SEQ:act_max(hqq-4b k=0.1) | 6.10 | 5.20 | 9.50 | 10.325 | +0.568 |  |
| SEQ:residual_max(hqq-4b k=0.1) | 6.10 | 5.20 | 9.50 | 10.337 | +0.580 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 9.50 | 10.413 | +0.656 |  |
| SEQ:random(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 9.50 | 10.432 | +0.675 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 9.50 | 10.441 | +0.684 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 9.50 | 10.484 | +0.727 |  |
| SEQ:residual_rms(hqq-4b k=0.1) | 6.10 | 5.20 | 9.50 | 10.500 | +0.743 |  |
| SEQ:act_max(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 9.50 | 10.646 | +0.888 |  |
| SEQ:random(hqq-4b k=0.1) | 6.10 | 5.20 | 9.50 | 11.089 | +1.332 |  |
| SEQ:tier_alloc(hqq-4b [budget=2.0]) | 6.56 | 4.54 | 9.96 | 10.479 | +0.721 |  |
| SEQ:act_max(hqq-4b k=0.2) | 7.70 | 6.40 | 11.11 | 10.238 | +0.481 |  |
| SEQ:residual_max(hqq-4b k=0.2) | 7.70 | 6.40 | 11.11 | 10.243 | +0.486 |  |
| SEQ:residual_rms(hqq-4b k=0.2) | 7.70 | 6.40 | 11.11 | 10.395 | +0.637 |  |
| SEQ:random(hqq-4b k=0.2) | 7.70 | 6.40 | 11.11 | 10.993 | +1.236 |  |
| HQQ-8 uniform | 8.00 | 8.00 | — | 9.762 | +0.005 | ★ |
| FP16 | 16.00 | 16.00 | — | 9.757 | +0.000 | ★ |

> **No SEQ point is on the frontier** — a baseline dominates it. At ~7.7 bits, best SEQ (10.238) **loses to** HQQ-8 uniform (9.762).

## meta-llama/Llama-3.2-3B  (FP16 PPL 7.8167)

Axis = **weight-only bits/param** (quantized linear weights + inline overhead; FP16 embeddings/lm_head/norms excluded, common to all methods — so it is comparable to GPTQ-4 = 4.0). *full-model bits* is the deployment average including FP16 embeddings, shown for reference only, not the frontier axis.

| method | weight bits | nominal bits | full-model bits | PPL | Δ vs FP16 | frontier |
|---|---|---|---|---|---|---|
| HQQ-4 uniform | 4.00 | 4.00 | — | 8.387 | +0.570 | ★ |
| AWQ-4 g128 | 4.00 | 4.00 | — | 8.405 | +0.588 |  |
| RTN-4 g128 | 4.00 | 4.00 | — | 8.498 | +0.682 |  |
| GPTQ-4 g128 | 4.28 | 4.00 | 5.72 | 8.304 | +0.487 | ★ |
| SEQ:act_scale(gptq_llmc-4b k=0.0) | 4.50 | 4.00 | 6.46 | 8.171 | +0.355 | ★ |
| SEQ:residual_max(gptq_llmc-4b k=0.0) | 4.50 | 4.00 | 6.46 | 8.171 | +0.355 | ★ |
| SEQ:random(gptq_llmc-4b k=0.0) | 4.50 | 4.00 | 6.46 | 8.171 | +0.355 | ★ |
| SEQ:residual_max(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 6.79 | 8.100 | +0.283 | ★ |
| SEQ:random(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 6.79 | 8.181 | +0.364 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 6.79 | 8.586 | +0.769 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 7.26 | 8.100 | +0.283 |  |
| SEQ:random(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 7.26 | 8.220 | +0.404 |  |
| SEQ:act_max(hqq-5b k=0.0) | 5.50 | 5.00 | 7.46 | 7.957 | +0.140 | ★ |
| SEQ:residual_max(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 8.07 | 8.101 | +0.284 |  |
| SEQ:random(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 8.07 | 9.048 | +1.231 |  |
| SEQ:act_max(hqq-6b k=0.0) | 6.50 | 6.00 | 8.46 | 7.845 | +0.028 | ★ |
| HQQ-5 uniform | 7.46 | 5.00 | — | 7.957 | +0.140 |  |
| HQQ-6 uniform | 8.46 | 6.00 | — | 7.845 | +0.028 |  |
| SEQ:act_max(hqq-8b k=0.0) | 8.50 | 8.00 | 10.46 | 7.820 | +0.003 | ★ |
| HQQ-8 uniform | 10.46 | 8.00 | — | 7.820 | +0.003 |  |
| FP16 | 16.00 | 16.00 | — | 7.817 | +0.000 | ★ |

> SEQ is on the Pareto frontier (7 point(s)). At ~8.5 bits, best SEQ (7.820) **beats** HQQ-6 uniform (7.845).
