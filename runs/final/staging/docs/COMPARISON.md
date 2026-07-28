# SEQ vs baselines — actual-bits comparison

Points from SEQ sweeps + external baselines, sorted by **weight-only bits/param** — quantized linear weights plus their inline overhead (group scales/zeros, FP16/INT8 protection residual, channel index), divided by the quantized-linear parameter count. Embeddings, lm_head, norms and biases are FP16 in every method, common to the axis, and excluded — so this axis is directly comparable to GPTQ-4 = 4.0. ★ = on the Pareto frontier (no method has both fewer bits and lower PPL).

## meta-llama/Llama-3.2-1B  (FP16 PPL 9.7572)

Axis = **weight-only bits/param** (quantized linear weights + inline overhead; FP16 embeddings/lm_head/norms excluded, common to all methods — so it is comparable to GPTQ-4 = 4.0). *full-model bits* is the deployment average including FP16 embeddings, shown for reference only, not the frontier axis.

| method | weight bits | nominal bits | full-model bits | PPL | Δ vs FP16 | frontier |
|---|---|---|---|---|---|---|
| GPTQ-4 g128 | 4.00 | 4.00 | — | 10.431 | +0.673 | ★ |
| AWQ-4 g128 | 4.00 | 4.00 | — | 11.263 | +1.506 |  |
| RTN-4 g128 | 4.00 | 4.00 | — | 11.711 | +1.954 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.0) | 4.25 | 4.00 | 10.15 | 10.557 | +0.800 |  |
| SEQ:residual_max(hqq-4b k=0.0) | 4.50 | 4.00 | 10.35 | 11.072 | +1.315 |  |
| HQQ-4 uniform | 4.50 | 4.00 | 10.35 | 11.072 | +1.315 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 10.40 | 10.423 | +0.666 | ★ |
| SEQ:residual_rms(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 10.40 | 10.434 | +0.677 |  |
| SEQ:random(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 10.40 | 10.557 [10.158, 10.957] | +0.800 |  |
| SEQ:act_max(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 10.40 | 10.711 | +0.954 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 10.40 | 10.930 | +1.172 |  |
| SEQ:greedy_indep(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 10.40 | 11.275 | +1.518 |  |
| SEQ:greedy(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 10.40 | 63.946 | +54.189 |  |
| SEQ:tier_alloc(hqq-4b [budget=0.25]) | 4.76 | 4.07 | 8.16 | 10.631 | +0.873 |  |
| SEQ:greedy(hqq-4b k=0.02) | 4.82 | 4.24 | 10.60 | 10.398 | +0.641 | ★ |
| SEQ:greedy_indep(hqq-4b k=0.02) | 4.82 | 4.24 | 10.60 | 10.406 | +0.649 |  |
| SEQ:residual_max(hqq-4b k=0.02) | 4.82 | 4.24 | 10.60 | 10.432 | +0.675 |  |
| SEQ:act_max(hqq-4b k=0.02) | 4.82 | 4.24 | 10.60 | 10.434 | +0.677 |  |
| SEQ:residual_rms(hqq-4b k=0.02) | 4.82 | 4.24 | 10.60 | 10.512 | +0.755 |  |
| SEQ:act_scale(hqq-4b k=0.02) | 4.82 | 4.24 | 10.60 | 10.521 | +0.763 |  |
| SEQ:random(hqq-4b k=0.02) | 4.82 | 4.24 | 10.60 | 11.052 [11.048, 11.055] | +1.294 |  |
| SEQ:tier_alloc(hqq-4b [budget=0.5]) | 5.01 | 4.13 | 8.41 | 10.592 | +0.835 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 10.78 | 10.436 | +0.678 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 10.78 | 10.436 | +0.679 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 10.78 | 10.476 | +0.719 |  |
| SEQ:random(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 10.78 | 10.553 [10.232, 10.874] | +0.796 |  |
| SEQ:act_max(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 10.78 | 10.687 | +0.930 |  |
| SEQ:greedy_indep(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 10.78 | 11.283 | +1.526 |  |
| SEQ:greedy(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 10.78 | 66.839 | +57.082 |  |
| SEQ:greedy(hqq-4b k=0.05) | 5.30 | 4.60 | 10.98 | 10.289 | +0.532 | ★ |
| SEQ:greedy_indep(hqq-4b k=0.05) | 5.30 | 4.60 | 10.98 | 10.312 | +0.555 |  |
| SEQ:act_max(hqq-4b k=0.05) | 5.30 | 4.60 | 10.98 | 10.329 | +0.572 |  |
| SEQ:residual_max(hqq-4b k=0.05) | 5.30 | 4.60 | 10.98 | 10.332 | +0.575 |  |
| SEQ:residual_rms(hqq-4b k=0.05) | 5.30 | 4.60 | 10.98 | 10.465 | +0.708 |  |
| SEQ:act_scale(hqq-4b k=0.05) | 5.30 | 4.60 | 10.98 | 10.467 | +0.710 |  |
| SEQ:random(hqq-4b k=0.05) | 5.30 | 4.60 | 10.98 | 11.025 [11.016, 11.034] | +1.268 |  |
| SEQ:act_max(hqq-5b k=0.0) | 5.50 | 5.00 | 8.90 | 10.064 | +0.306 | ★ |
| HQQ-5 uniform | 5.50 | 5.00 | 8.90 | 10.064 | +0.306 | ★ |
| SEQ:tier_alloc(hqq-4b [budget=1.0]) | 5.53 | 4.27 | 8.93 | 10.555 | +0.798 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 11.41 | 10.413 | +0.656 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 11.41 | 10.441 | +0.684 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 11.41 | 10.484 | +0.727 |  |
| SEQ:random(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 11.41 | 10.485 [10.290, 10.680] | +0.728 |  |
| SEQ:act_max(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 11.41 | 10.646 | +0.888 |  |
| SEQ:greedy_indep(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 11.41 | 11.251 | +1.494 |  |
| SEQ:greedy(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 11.41 | 77.987 | +68.230 |  |
| SEQ:greedy(hqq-4b k=0.1) | 6.10 | 5.20 | 11.61 | 10.216 | +0.459 |  |
| SEQ:greedy_indep(hqq-4b k=0.1) | 6.10 | 5.20 | 11.61 | 10.239 | +0.481 |  |
| SEQ:act_max(hqq-4b k=0.1) | 6.10 | 5.20 | 11.61 | 10.246 | +0.489 |  |
| SEQ:residual_max(hqq-4b k=0.1) | 6.10 | 5.20 | 11.61 | 10.250 | +0.493 |  |
| SEQ:residual_rms(hqq-4b k=0.1) | 6.10 | 5.20 | 11.61 | 10.400 | +0.643 |  |
| SEQ:act_scale(hqq-4b k=0.1) | 6.10 | 5.20 | 11.61 | 10.403 | +0.646 |  |
| SEQ:random(hqq-4b k=0.1) | 6.10 | 5.20 | 11.61 | 10.976 [10.943, 11.011] | +1.219 |  |
| SEQ:act_max(hqq-6b k=0.0) | 6.50 | 6.00 | 9.90 | 9.829 | +0.072 | ★ |
| HQQ-6 uniform | 6.50 | 6.00 | 9.90 | 9.829 | +0.072 | ★ |
| SEQ:tier_alloc(hqq-4b [budget=2.0]) | 6.56 | 4.54 | 9.96 | 10.479 | +0.721 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 12.67 | 10.355 | +0.598 |  |
| SEQ:act_max(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 12.67 | 10.383 | +0.626 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 12.67 | 10.453 | +0.695 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 12.67 | 10.479 | +0.722 |  |
| SEQ:random(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 12.67 | 10.508 [10.334, 10.682] | +0.751 |  |
| SEQ:greedy_indep(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 12.67 | 11.385 | +1.628 |  |
| SEQ:greedy(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 12.67 | 64.995 | +55.238 |  |
| SEQ:greedy(hqq-4b k=0.2) | 7.70 | 6.40 | 12.87 | 10.128 | +0.371 |  |
| SEQ:greedy_indep(hqq-4b k=0.2) | 7.70 | 6.40 | 12.87 | 10.155 | +0.397 |  |
| SEQ:residual_max(hqq-4b k=0.2) | 7.70 | 6.40 | 12.87 | 10.157 | +0.400 |  |
| SEQ:act_max(hqq-4b k=0.2) | 7.70 | 6.40 | 12.87 | 10.158 | +0.401 |  |
| SEQ:residual_rms(hqq-4b k=0.2) | 7.70 | 6.40 | 12.87 | 10.308 | +0.550 |  |
| SEQ:act_scale(hqq-4b k=0.2) | 7.70 | 6.40 | 12.87 | 10.309 | +0.552 |  |
| SEQ:random(hqq-4b k=0.2) | 7.70 | 6.40 | 12.87 | 10.884 [10.836, 10.933] | +1.127 |  |
| SEQ:act_max(hqq-8b k=0.0) | 8.50 | 8.00 | 11.90 | 9.762 | +0.005 | ★ |
| HQQ-8 uniform | 8.50 | 8.00 | 11.90 | 9.762 | +0.005 | ★ |
| FP16 | 16.00 | 16.00 | — | 9.757 | +0.000 | ★ |

> SEQ is on the Pareto frontier (6 point(s)). At ~8.5 bits, best SEQ (9.762) **loses to** HQQ-8 uniform (9.762).

## meta-llama/Llama-3.2-3B  (FP16 PPL 7.8167)

Axis = **weight-only bits/param** (quantized linear weights + inline overhead; FP16 embeddings/lm_head/norms excluded, common to all methods — so it is comparable to GPTQ-4 = 4.0). *full-model bits* is the deployment average including FP16 embeddings, shown for reference only, not the frontier axis.

| method | weight bits | nominal bits | full-model bits | PPL | Δ vs FP16 | frontier |
|---|---|---|---|---|---|---|
| GPTQ-4 g128 | 4.00 | 4.00 | — | 8.271 | +0.455 | ★ |
| AWQ-4 g128 | 4.00 | 4.00 | — | 8.409 | +0.593 |  |
| RTN-4 g128 | 4.00 | 4.00 | — | 8.497 | +0.680 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.0) | 4.25 | 4.00 | 7.65 | 8.171 | +0.355 | ★ |
| SEQ:residual_max(hqq-4b k=0.0) | 4.50 | 4.00 | 7.87 | 8.328 | +0.511 |  |
| HQQ-4 uniform | 4.50 | 4.00 | 7.87 | 8.328 | +0.511 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 7.94 | 8.100 | +0.283 | ★ |
| SEQ:residual_rms(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 7.94 | 8.146 | +0.329 |  |
| SEQ:random(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 7.94 | 8.170 [8.143, 8.197] | +0.353 |  |
| SEQ:act_max(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 7.94 | 8.181 | +0.365 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 7.94 | 8.586 | +0.769 |  |
| SEQ:greedy_indep(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 7.94 | 41.502 | +33.685 |  |
| SEQ:greedy(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 7.94 | 51.683 | +43.866 |  |
| SEQ:greedy(hqq-4b k=0.02) | 4.82 | 4.24 | 8.16 | 8.101 | +0.284 |  |
| SEQ:greedy_indep(hqq-4b k=0.02) | 4.82 | 4.24 | 8.16 | 8.101 | +0.285 |  |
| SEQ:act_max(hqq-4b k=0.02) | 4.82 | 4.24 | 8.16 | 8.108 | +0.292 |  |
| SEQ:residual_max(hqq-4b k=0.02) | 4.82 | 4.24 | 8.16 | 8.110 | +0.293 |  |
| SEQ:residual_rms(hqq-4b k=0.02) | 4.82 | 4.24 | 8.16 | 8.141 | +0.324 |  |
| SEQ:act_scale(hqq-4b k=0.02) | 4.82 | 4.24 | 8.16 | 8.146 | +0.329 |  |
| SEQ:random(hqq-4b k=0.02) | 4.82 | 4.24 | 8.16 | 8.318 [8.312, 8.325] | +0.501 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 8.36 | 8.100 | +0.283 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 8.36 | 8.153 | +0.336 |  |
| SEQ:act_max(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 8.36 | 8.173 | +0.356 |  |
| SEQ:random(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 8.36 | 8.193 [8.091, 8.295] | +0.376 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 8.36 | 8.204 | +0.388 |  |
| SEQ:greedy_indep(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 8.36 | 43.296 | +35.479 |  |
| SEQ:greedy(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 8.36 | 55.047 | +47.230 |  |
| SEQ:greedy(hqq-4b k=0.05) | 5.30 | 4.60 | 8.58 | 8.064 | +0.247 | ★ |
| SEQ:greedy_indep(hqq-4b k=0.05) | 5.30 | 4.60 | 8.58 | 8.064 | +0.247 | ★ |
| SEQ:act_max(hqq-4b k=0.05) | 5.30 | 4.60 | 8.58 | 8.075 | +0.259 |  |
| SEQ:residual_max(hqq-4b k=0.05) | 5.30 | 4.60 | 8.58 | 8.078 | +0.261 |  |
| SEQ:residual_rms(hqq-4b k=0.05) | 5.30 | 4.60 | 8.58 | 8.111 | +0.294 |  |
| SEQ:act_scale(hqq-4b k=0.05) | 5.30 | 4.60 | 8.58 | 8.114 | +0.297 |  |
| SEQ:random(hqq-4b k=0.05) | 5.30 | 4.60 | 8.58 | 8.304 [8.297, 8.312] | +0.488 |  |
| SEQ:act_max(hqq-5b k=0.0) | 5.50 | 5.00 | 7.46 | 7.957 | +0.140 | ★ |
| HQQ-5 uniform | 5.50 | 5.00 | 7.46 | 7.957 | +0.140 | ★ |
| SEQ:residual_max(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 9.06 | 8.101 | +0.284 |  |
| SEQ:act_max(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 9.06 | 8.136 | +0.319 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 9.06 | 8.160 | +0.343 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 9.06 | 8.187 | +0.370 |  |
| SEQ:random(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 9.06 | 8.464 [7.207, 9.721] | +0.647 |  |
| SEQ:greedy(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 9.06 | 38.385 | +30.568 |  |
| SEQ:greedy_indep(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 9.06 | 44.829 | +37.012 |  |
| SEQ:greedy(hqq-4b k=0.1) | 6.10 | 5.20 | 9.28 | 8.031 | +0.214 |  |
| SEQ:greedy_indep(hqq-4b k=0.1) | 6.10 | 5.20 | 9.28 | 8.035 | +0.219 |  |
| SEQ:act_max(hqq-4b k=0.1) | 6.10 | 5.20 | 9.28 | 8.045 | +0.228 |  |
| SEQ:residual_max(hqq-4b k=0.1) | 6.10 | 5.20 | 9.28 | 8.047 | +0.230 |  |
| SEQ:residual_rms(hqq-4b k=0.1) | 6.10 | 5.20 | 9.28 | 8.087 | +0.270 |  |
| SEQ:act_scale(hqq-4b k=0.1) | 6.10 | 5.20 | 9.28 | 8.090 | +0.273 |  |
| SEQ:random(hqq-4b k=0.1) | 6.10 | 5.20 | 9.28 | 8.281 [8.275, 8.286] | +0.464 |  |
| SEQ:act_max(hqq-6b k=0.0) | 6.50 | 6.00 | 8.46 | 7.845 | +0.028 | ★ |
| HQQ-6 uniform | 6.50 | 6.00 | 8.46 | 7.845 | +0.028 | ★ |
| SEQ:residual_max(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 10.46 | 8.075 | +0.259 |  |
| SEQ:act_max(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 10.46 | 8.101 | +0.285 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 10.46 | 8.139 | +0.322 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 10.46 | 8.154 | +0.337 |  |
| SEQ:random(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 10.46 | 8.336 [7.899, 8.773] | +0.519 |  |
| SEQ:greedy(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 10.46 | 40.652 | +32.835 |  |
| SEQ:greedy_indep(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 10.46 | 43.039 | +35.223 |  |
| SEQ:greedy(hqq-4b k=0.2) | 7.70 | 6.40 | 10.68 | 7.994 | +0.178 |  |
| SEQ:greedy_indep(hqq-4b k=0.2) | 7.70 | 6.40 | 10.68 | 8.002 | +0.185 |  |
| SEQ:residual_max(hqq-4b k=0.2) | 7.70 | 6.40 | 10.68 | 8.007 | +0.190 |  |
| SEQ:act_max(hqq-4b k=0.2) | 7.70 | 6.40 | 10.68 | 8.007 | +0.190 |  |
| SEQ:residual_rms(hqq-4b k=0.2) | 7.70 | 6.40 | 10.68 | 8.040 | +0.224 |  |
| SEQ:act_scale(hqq-4b k=0.2) | 7.70 | 6.40 | 10.68 | 8.046 | +0.229 |  |
| SEQ:random(hqq-4b k=0.2) | 7.70 | 6.40 | 10.68 | 8.198 [7.983, 8.413] | +0.381 |  |
| SEQ:act_max(hqq-8b k=0.0) | 8.50 | 8.00 | 10.46 | 7.820 | +0.003 | ★ |
| HQQ-8 uniform | 8.50 | 8.00 | 10.46 | 7.820 | +0.003 | ★ |
| FP16 | 16.00 | 16.00 | — | 7.817 | +0.000 | ★ |

> SEQ is on the Pareto frontier (7 point(s)). At ~8.5 bits, best SEQ (7.820) **loses to** HQQ-8 uniform (7.820).

## Qwen/Qwen2.5-3B  (FP16 PPL 8.0304)

Axis = **weight-only bits/param** (quantized linear weights + inline overhead; FP16 embeddings/lm_head/norms excluded, common to all methods — so it is comparable to GPTQ-4 = 4.0). *full-model bits* is the deployment average including FP16 embeddings, shown for reference only, not the frontier axis.

| method | weight bits | nominal bits | full-model bits | PPL | Δ vs FP16 | frontier |
|---|---|---|---|---|---|---|
| GPTQ-4 g128 | 4.00 | 4.00 | — | 8.291 | +0.261 | ★ |
| SEQ:residual_max(gptq_llmc-4b k=0.0) | 4.25 | 4.00 | 7.05 | 8.290 | +0.260 | ★ |
| SEQ:residual_max(hqq-4b k=0.0) | 4.50 | 4.00 | 7.27 | 8.666 | +0.635 |  |
| HQQ-4 uniform | 4.50 | 4.00 | 7.27 | 8.666 | +0.635 |  |
| SEQ:random(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 7.34 | 8.292 [8.275, 8.309] | +0.261 |  |
| SEQ:greedy(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 7.34 | 8.335 | +0.304 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 7.34 | 8.339 | +0.309 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 7.34 | 8.351 | +0.320 |  |
| SEQ:greedy_indep(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 7.34 | 8.357 | +0.327 |  |
| SEQ:act_max(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 7.34 | 8.366 | +0.335 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.02) | 4.57 | 4.24 | 7.34 | 8.373 | +0.342 |  |
| SEQ:greedy_indep(hqq-4b k=0.02) | 4.82 | 4.24 | 7.56 | 8.272 | +0.241 | ★ |
| SEQ:greedy(hqq-4b k=0.02) | 4.82 | 4.24 | 7.56 | 8.274 | +0.243 |  |
| SEQ:residual_rms(hqq-4b k=0.02) | 4.82 | 4.24 | 7.56 | 8.279 | +0.249 |  |
| SEQ:act_max(hqq-4b k=0.02) | 4.82 | 4.24 | 7.56 | 8.286 | +0.255 |  |
| SEQ:residual_max(hqq-4b k=0.02) | 4.82 | 4.24 | 7.56 | 8.287 | +0.257 |  |
| SEQ:act_scale(hqq-4b k=0.02) | 4.82 | 4.24 | 7.56 | 8.300 | +0.270 |  |
| SEQ:random(hqq-4b k=0.02) | 4.82 | 4.24 | 7.56 | 8.654 [8.640, 8.667] | +0.623 |  |
| SEQ:random(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 7.77 | 8.302 [8.253, 8.351] | +0.272 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 7.77 | 8.326 | +0.296 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 7.77 | 8.332 | +0.301 |  |
| SEQ:act_max(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 7.77 | 8.352 | +0.321 |  |
| SEQ:greedy(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 7.77 | 8.354 | +0.323 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 7.77 | 8.357 | +0.326 |  |
| SEQ:greedy_indep(gptq_llmc-4b k=0.05) | 5.05 | 4.60 | 7.77 | 8.370 | +0.339 |  |
| SEQ:greedy(hqq-4b k=0.05) | 5.30 | 4.60 | 8.00 | 8.245 | +0.214 | ★ |
| SEQ:greedy_indep(hqq-4b k=0.05) | 5.30 | 4.60 | 8.00 | 8.246 | +0.216 |  |
| SEQ:residual_rms(hqq-4b k=0.05) | 5.30 | 4.60 | 8.00 | 8.261 | +0.231 |  |
| SEQ:residual_max(hqq-4b k=0.05) | 5.30 | 4.60 | 8.00 | 8.263 | +0.232 |  |
| SEQ:act_max(hqq-4b k=0.05) | 5.30 | 4.60 | 8.00 | 8.267 | +0.237 |  |
| SEQ:act_scale(hqq-4b k=0.05) | 5.30 | 4.60 | 8.00 | 8.281 | +0.251 |  |
| SEQ:random(hqq-4b k=0.05) | 5.30 | 4.60 | 8.00 | 8.645 [8.622, 8.669] | +0.615 |  |
| SEQ:act_max(hqq-5b k=0.0) | 5.50 | 5.00 | 7.11 | 8.241 | +0.210 | ★ |
| HQQ-5 uniform | 5.50 | 5.00 | 7.11 | 8.241 | +0.210 | ★ |
| SEQ:residual_max(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 8.49 | 8.296 | +0.266 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 8.49 | 8.307 | +0.277 |  |
| SEQ:act_max(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 8.49 | 8.320 | +0.289 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 8.49 | 8.320 | +0.290 |  |
| SEQ:random(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 8.49 | 8.340 [8.307, 8.374] | +0.310 |  |
| SEQ:greedy(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 8.49 | 8.342 | +0.311 |  |
| SEQ:greedy_indep(gptq_llmc-4b k=0.1) | 5.85 | 5.20 | 8.49 | 8.364 | +0.334 |  |
| SEQ:greedy(hqq-4b k=0.1) | 6.10 | 5.20 | 8.71 | 8.217 | +0.187 | ★ |
| SEQ:greedy_indep(hqq-4b k=0.1) | 6.10 | 5.20 | 8.71 | 8.220 | +0.189 |  |
| SEQ:residual_rms(hqq-4b k=0.1) | 6.10 | 5.20 | 8.71 | 8.238 | +0.208 |  |
| SEQ:act_max(hqq-4b k=0.1) | 6.10 | 5.20 | 8.71 | 8.239 | +0.209 |  |
| SEQ:residual_max(hqq-4b k=0.1) | 6.10 | 5.20 | 8.71 | 8.239 | +0.209 |  |
| SEQ:act_scale(hqq-4b k=0.1) | 6.10 | 5.20 | 8.71 | 8.241 | +0.211 |  |
| SEQ:random(hqq-4b k=0.1) | 6.10 | 5.20 | 8.71 | 8.617 [8.613, 8.622] | +0.587 |  |
| SEQ:act_max(hqq-6b k=0.0) | 6.50 | 6.00 | 8.11 | 8.080 | +0.050 | ★ |
| HQQ-6 uniform | 6.50 | 6.00 | 8.11 | 8.080 | +0.050 | ★ |
| SEQ:residual_max(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 9.93 | 8.263 | +0.232 |  |
| SEQ:act_max(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 9.93 | 8.282 | +0.251 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 9.93 | 8.287 | +0.256 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 9.93 | 8.287 | +0.256 |  |
| SEQ:greedy(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 9.93 | 8.325 | +0.294 |  |
| SEQ:greedy_indep(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 9.93 | 8.363 | +0.333 |  |
| SEQ:random(gptq_llmc-4b k=0.2) | 7.45 | 6.40 | 9.93 | 8.366 [8.261, 8.471] | +0.336 |  |
| SEQ:greedy(hqq-4b k=0.2) | 7.70 | 6.40 | 10.15 | 8.187 | +0.157 |  |
| SEQ:greedy_indep(hqq-4b k=0.2) | 7.70 | 6.40 | 10.15 | 8.191 | +0.161 |  |
| SEQ:residual_max(hqq-4b k=0.2) | 7.70 | 6.40 | 10.15 | 8.199 | +0.169 |  |
| SEQ:act_max(hqq-4b k=0.2) | 7.70 | 6.40 | 10.15 | 8.200 | +0.169 |  |
| SEQ:act_scale(hqq-4b k=0.2) | 7.70 | 6.40 | 10.15 | 8.208 | +0.177 |  |
| SEQ:residual_rms(hqq-4b k=0.2) | 7.70 | 6.40 | 10.15 | 8.210 | +0.180 |  |
| SEQ:random(hqq-4b k=0.2) | 7.70 | 6.40 | 10.15 | 8.526 [8.404, 8.649] | +0.496 |  |
| SEQ:act_max(hqq-8b k=0.0) | 8.50 | 8.00 | 10.11 | 8.036 | +0.005 | ★ |
| HQQ-8 uniform | 8.50 | 8.00 | 10.11 | 8.036 | +0.005 | ★ |
| FP16 | 16.00 | 16.00 | — | 8.030 | +0.000 | ★ |

> SEQ is on the Pareto frontier (7 point(s)). At ~8.5 bits, best SEQ (8.036) **loses to** HQQ-8 uniform (8.036).

## Qwen/Qwen3-4B-Base

Axis = **weight-only bits/param** (quantized linear weights + inline overhead; FP16 embeddings/lm_head/norms excluded, common to all methods — so it is comparable to GPTQ-4 = 4.0). *full-model bits* is the deployment average including FP16 embeddings, shown for reference only, not the frontier axis.

| method | weight bits | nominal bits | full-model bits | PPL | Δ vs FP16 | frontier |
|---|---|---|---|---|---|---|
| GPTQ-4 g128 | 4.00 | 4.00 | — | 8.134 | — | ★ |

> **No SEQ point is on the frontier** — a baseline dominates it.

## meta-llama/Llama-2-7b-hf

Axis = **weight-only bits/param** (quantized linear weights + inline overhead; FP16 embeddings/lm_head/norms excluded, common to all methods — so it is comparable to GPTQ-4 = 4.0). *full-model bits* is the deployment average including FP16 embeddings, shown for reference only, not the frontier axis.

| method | weight bits | nominal bits | full-model bits | PPL | Δ vs FP16 | frontier |
|---|---|---|---|---|---|---|
| GPTQ-4 g128 | 4.00 | 4.00 | — | 5.590 | — | ★ |

> **No SEQ point is on the frontier** — a baseline dominates it.

## mistralai/Mistral-7B-v0.3

Axis = **weight-only bits/param** (quantized linear weights + inline overhead; FP16 embeddings/lm_head/norms excluded, common to all methods — so it is comparable to GPTQ-4 = 4.0). *full-model bits* is the deployment average including FP16 embeddings, shown for reference only, not the frontier axis.

| method | weight bits | nominal bits | full-model bits | PPL | Δ vs FP16 | frontier |
|---|---|---|---|---|---|---|
| GPTQ-4 g128 | 4.00 | 4.00 | — | 5.482 | — | ★ |

> **No SEQ point is on the frontier** — a baseline dominates it.
