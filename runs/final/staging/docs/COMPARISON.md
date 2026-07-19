# SEQ vs baselines — actual-bits comparison

Points from SEQ sweeps + external baselines, sorted by **weight-only bits/param** — quantized linear weights plus their inline overhead (group scales/zeros, FP16/INT8 protection residual, channel index), divided by the quantized-linear parameter count. Embeddings, lm_head, norms and biases are FP16 in every method, common to the axis, and excluded — so this axis is directly comparable to GPTQ-4 = 4.0. ★ = on the Pareto frontier (no method has both fewer bits and lower PPL).

## meta-llama/Llama-3.2-1B  (FP16 PPL 9.7571)

Axis = **weight-only bits/param** (quantized linear weights + inline overhead; FP16 embeddings/lm_head/norms excluded, common to all methods — so it is comparable to GPTQ-4 = 4.0). *full-model bits* is the deployment average including FP16 embeddings, shown for reference only, not the frontier axis.

| method | weight bits | nominal bits | full-model bits | PPL | Δ vs FP16 | frontier |
|---|---|---|---|---|---|---|
| AWQ-4 g128 | 4.00 | 4.00 | — | 11.263 | +1.506 | ★ |
| RTN-4 g128 | 4.00 | 4.00 | — | 11.711 | +1.954 |  |
| GPTQ-4 g128 | 4.29 | 4.00 | 6.78 | 10.405 | +0.648 | ★ |
| SEQ:residual_max(gptq_llmc-4b k=0.0) | 4.50 | 4.00 | 7.90 | 10.779 | +1.022 |  |
| SEQ:residual_max(hqq-4b k=0.0) | 4.50 | 4.00 | 7.90 | 11.187 | +1.430 |  |
| HQQ-4 uniform | 4.50 | 4.00 | 7.90 | 11.187 | +1.430 |  |
| SEQ:tier_alloc(hqq-4b [budget=0.25]) | 4.76 | 4.07 | 8.16 | 10.631 | +0.874 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 8.22 | 10.391 | +0.634 | ★ |
| SEQ:residual_rms(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 8.22 | 10.400 | +0.643 |  |
| SEQ:greedy(hqq-4b k=0.02) | 4.82 | 4.24 | 8.22 | 10.495 | +0.738 |  |
| SEQ:greedy_indep(hqq-4b k=0.02) | 4.82 | 4.24 | 8.22 | 10.506 | +0.749 |  |
| SEQ:residual_max(hqq-4b k=0.02) | 4.82 | 4.24 | 8.22 | 10.533 | +0.776 |  |
| SEQ:act_max(hqq-4b k=0.02) | 4.82 | 4.24 | 8.22 | 10.538 | +0.781 |  |
| SEQ:residual_rms(hqq-4b k=0.02) | 4.82 | 4.24 | 8.22 | 10.614 | +0.857 |  |
| SEQ:act_scale(hqq-4b k=0.02) | 4.82 | 4.24 | 8.22 | 10.621 | +0.864 |  |
| SEQ:random(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 8.22 | 10.680 [10.409, 10.952] | +0.923 |  |
| SEQ:act_max(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 8.22 | 10.692 | +0.935 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 8.22 | 10.935 | +1.178 |  |
| SEQ:random(hqq-4b k=0.02) | 4.82 | 4.24 | 8.22 | 11.164 [11.155, 11.174] | +1.407 |  |
| SEQ:greedy_indep(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 8.22 | 15.639 | +5.882 |  |
| SEQ:greedy(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 8.22 | 104.163 | +94.406 |  |
| SEQ:tier_alloc(hqq-4b [budget=0.5]) | 5.01 | 4.13 | 8.41 | 10.592 | +0.835 |  |
| SEQ:greedy(hqq-4b k=0.05) | 5.30 | 4.60 | 8.71 | 10.381 | +0.624 | ★ |
| SEQ:residual_max(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 8.71 | 10.400 | +0.643 |  |
| SEQ:greedy_indep(hqq-4b k=0.05) | 5.30 | 4.60 | 8.71 | 10.405 | +0.648 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 8.71 | 10.424 | +0.667 |  |
| SEQ:act_max(hqq-4b k=0.05) | 5.30 | 4.60 | 8.71 | 10.425 | +0.668 |  |
| SEQ:residual_max(hqq-4b k=0.05) | 5.30 | 4.60 | 8.71 | 10.428 | +0.671 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 8.71 | 10.456 | +0.699 |  |
| SEQ:residual_rms(hqq-4b k=0.05) | 5.30 | 4.60 | 8.71 | 10.567 | +0.810 |  |
| SEQ:act_scale(hqq-4b k=0.05) | 5.30 | 4.60 | 8.71 | 10.571 | +0.814 |  |
| SEQ:act_max(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 8.71 | 10.657 | +0.900 |  |
| SEQ:random(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 8.71 | 10.689 [10.282, 11.097] | +0.932 |  |
| SEQ:random(hqq-4b k=0.05) | 5.30 | 4.60 | 8.71 | 11.133 [11.123, 11.143] | +1.376 |  |
| SEQ:greedy_indep(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 8.71 | 14.884 | +5.127 |  |
| SEQ:greedy(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 8.71 | 98.410 | +88.653 |  |
| SEQ:act_max(hqq-5b k=0.0) | 5.50 | 5.00 | 8.90 | 10.064 | +0.307 | ★ |
| HQQ-5 uniform | 5.50 | 5.00 | 8.90 | 10.064 | +0.307 | ★ |
| SEQ:tier_alloc(hqq-4b [budget=1.0]) | 5.53 | 4.27 | 8.93 | 10.555 | +0.798 |  |
| SEQ:greedy(hqq-4b k=0.1) | 6.10 | 5.20 | 9.50 | 10.299 | +0.542 |  |
| SEQ:greedy_indep(hqq-4b k=0.1) | 6.10 | 5.20 | 9.50 | 10.323 | +0.566 |  |
| SEQ:act_max(hqq-4b k=0.1) | 6.10 | 5.20 | 9.50 | 10.332 | +0.575 |  |
| SEQ:residual_max(hqq-4b k=0.1) | 6.10 | 5.20 | 9.50 | 10.337 | +0.580 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 9.50 | 10.391 | +0.634 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 9.50 | 10.411 | +0.654 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 9.50 | 10.460 | +0.703 |  |
| SEQ:residual_rms(hqq-4b k=0.1) | 6.10 | 5.20 | 9.50 | 10.501 | +0.744 |  |
| SEQ:act_scale(hqq-4b k=0.1) | 6.10 | 5.20 | 9.50 | 10.504 | +0.746 |  |
| SEQ:random(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 9.50 | 10.602 [10.130, 11.075] | +0.845 |  |
| SEQ:act_max(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 9.50 | 10.638 | +0.881 |  |
| SEQ:random(hqq-4b k=0.1) | 6.10 | 5.20 | 9.50 | 11.080 [11.035, 11.124] | +1.323 |  |
| SEQ:greedy_indep(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 9.50 | 14.864 | +5.107 |  |
| SEQ:greedy(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 9.50 | 103.532 | +93.775 |  |
| SEQ:act_max(hqq-6b k=0.0) | 6.50 | 6.00 | 9.90 | 9.829 | +0.072 | ★ |
| HQQ-6 uniform | 6.50 | 6.00 | 9.90 | 9.829 | +0.072 | ★ |
| SEQ:tier_alloc(hqq-4b [budget=2.0]) | 6.56 | 4.54 | 9.96 | 10.479 | +0.722 |  |
| SEQ:greedy(hqq-4b k=0.2) | 7.70 | 6.40 | 11.11 | 10.207 | +0.450 |  |
| SEQ:act_max(hqq-4b k=0.2) | 7.70 | 6.40 | 11.11 | 10.227 | +0.470 |  |
| SEQ:greedy_indep(hqq-4b k=0.2) | 7.70 | 6.40 | 11.11 | 10.230 | +0.473 |  |
| SEQ:residual_max(hqq-4b k=0.2) | 7.70 | 6.40 | 11.11 | 10.232 | +0.475 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.2) | 7.70 | 6.40 | 11.11 | 10.350 | +0.592 |  |
| SEQ:act_max(gptq_llmc-4b k=0.2) | 7.70 | 6.40 | 11.11 | 10.362 | +0.605 |  |
| SEQ:residual_rms(hqq-4b k=0.2) | 7.70 | 6.40 | 11.11 | 10.395 | +0.638 |  |
| SEQ:act_scale(hqq-4b k=0.2) | 7.70 | 6.40 | 11.11 | 10.399 | +0.642 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.2) | 7.70 | 6.40 | 11.11 | 10.403 | +0.646 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.2) | 7.70 | 6.40 | 11.11 | 10.449 | +0.692 |  |
| SEQ:random(gptq_llmc-4b k=0.2) | 7.70 | 6.40 | 11.11 | 10.649 [10.251, 11.048] | +0.892 |  |
| SEQ:random(hqq-4b k=0.2) | 7.70 | 6.40 | 11.11 | 10.975 [10.932, 11.018] | +1.218 |  |
| SEQ:greedy_indep(gptq_llmc-4b k=0.2) | 7.70 | 6.40 | 11.11 | 15.607 | +5.850 |  |
| SEQ:greedy(gptq_llmc-4b k=0.2) | 7.70 | 6.40 | 11.11 | 106.824 | +97.067 |  |
| SEQ:act_max(hqq-8b k=0.0) | 8.50 | 8.00 | 11.90 | 9.762 | +0.005 | ★ |
| HQQ-8 uniform | 8.50 | 8.00 | 11.90 | 9.762 | +0.005 | ★ |
| FP16 | 16.00 | 16.00 | — | 9.757 | +0.000 | ★ |

> SEQ is on the Pareto frontier (5 point(s)). At ~8.5 bits, best SEQ (9.762) **loses to** HQQ-8 uniform (9.762).

## meta-llama/Llama-3.2-3B  (FP16 PPL 7.8166)

Axis = **weight-only bits/param** (quantized linear weights + inline overhead; FP16 embeddings/lm_head/norms excluded, common to all methods — so it is comparable to GPTQ-4 = 4.0). *full-model bits* is the deployment average including FP16 embeddings, shown for reference only, not the frontier axis.

| method | weight bits | nominal bits | full-model bits | PPL | Δ vs FP16 | frontier |
|---|---|---|---|---|---|---|
| AWQ-4 g128 | 4.00 | 4.00 | — | 8.409 | +0.593 | ★ |
| RTN-4 g128 | 4.00 | 4.00 | — | 8.497 | +0.680 |  |
| GPTQ-4 g128 | 4.28 | 4.00 | 6.16 | 8.326 | +0.510 | ★ |
| SEQ:residual_max(gptq_llmc-4b k=0.0) | 4.50 | 4.00 | 6.46 | 8.162 | +0.345 | ★ |
| SEQ:residual_max(hqq-4b k=0.0) | 4.50 | 4.00 | 6.46 | 8.387 | +0.571 |  |
| HQQ-4 uniform | 4.50 | 4.00 | 6.46 | 8.387 | +0.571 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 6.79 | 8.098 | +0.282 | ★ |
| SEQ:residual_rms(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 6.79 | 8.137 | +0.321 |  |
| SEQ:greedy(hqq-4b k=0.02) | 4.82 | 4.24 | 6.79 | 8.149 | +0.333 |  |
| SEQ:greedy_indep(hqq-4b k=0.02) | 4.82 | 4.24 | 6.79 | 8.151 | +0.334 |  |
| SEQ:act_max(hqq-4b k=0.02) | 4.82 | 4.24 | 6.79 | 8.160 | +0.344 |  |
| SEQ:residual_max(hqq-4b k=0.02) | 4.82 | 4.24 | 6.79 | 8.161 | +0.344 |  |
| SEQ:random(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 6.79 | 8.161 [8.128, 8.193] | +0.344 |  |
| SEQ:act_max(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 6.79 | 8.168 | +0.352 |  |
| SEQ:residual_rms(hqq-4b k=0.02) | 4.82 | 4.24 | 6.79 | 8.200 | +0.384 |  |
| SEQ:act_scale(hqq-4b k=0.02) | 4.82 | 4.24 | 6.79 | 8.205 | +0.388 |  |
| SEQ:random(hqq-4b k=0.02) | 4.82 | 4.24 | 6.79 | 8.376 [8.370, 8.382] | +0.559 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 6.79 | 8.577 | +0.760 |  |
| SEQ:greedy_indep(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 6.79 | 44.878 | +37.062 |  |
| SEQ:greedy(gptq_llmc-4b k=0.02) | 4.82 | 4.24 | 6.79 | 55.345 | +47.528 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 7.26 | 8.098 | +0.281 | ★ |
| SEQ:greedy_indep(hqq-4b k=0.05) | 5.30 | 4.60 | 7.26 | 8.111 | +0.294 |  |
| SEQ:greedy(hqq-4b k=0.05) | 5.30 | 4.60 | 7.26 | 8.112 | +0.295 |  |
| SEQ:act_max(hqq-4b k=0.05) | 5.30 | 4.60 | 7.26 | 8.125 | +0.309 |  |
| SEQ:residual_max(hqq-4b k=0.05) | 5.30 | 4.60 | 7.26 | 8.126 | +0.310 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 7.26 | 8.147 | +0.330 |  |
| SEQ:residual_rms(hqq-4b k=0.05) | 5.30 | 4.60 | 7.26 | 8.168 | +0.351 |  |
| SEQ:act_max(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 7.26 | 8.168 | +0.351 |  |
| SEQ:act_scale(hqq-4b k=0.05) | 5.30 | 4.60 | 7.26 | 8.171 | +0.354 |  |
| SEQ:random(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 7.26 | 8.193 [8.099, 8.286] | +0.376 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 7.26 | 8.210 | +0.393 |  |
| SEQ:random(hqq-4b k=0.05) | 5.30 | 4.60 | 7.26 | 8.361 [8.351, 8.371] | +0.544 |  |
| SEQ:greedy_indep(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 7.26 | 45.597 | +37.781 |  |
| SEQ:greedy(gptq_llmc-4b k=0.05) | 5.30 | 4.60 | 7.26 | 47.193 | +39.377 |  |
| SEQ:act_max(hqq-5b k=0.0) | 5.50 | 5.00 | 7.46 | 7.957 | +0.140 | ★ |
| HQQ-5 uniform | 5.50 | 5.00 | 7.46 | 7.957 | +0.140 | ★ |
| SEQ:greedy(hqq-4b k=0.1) | 6.10 | 5.20 | 8.07 | 8.073 | +0.257 |  |
| SEQ:greedy_indep(hqq-4b k=0.1) | 6.10 | 5.20 | 8.07 | 8.076 | +0.260 |  |
| SEQ:act_max(hqq-4b k=0.1) | 6.10 | 5.20 | 8.07 | 8.091 | +0.274 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 8.07 | 8.094 | +0.277 |  |
| SEQ:residual_max(hqq-4b k=0.1) | 6.10 | 5.20 | 8.07 | 8.095 | +0.278 |  |
| SEQ:act_max(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 8.07 | 8.134 | +0.317 |  |
| SEQ:residual_rms(hqq-4b k=0.1) | 6.10 | 5.20 | 8.07 | 8.142 | +0.325 |  |
| SEQ:act_scale(hqq-4b k=0.1) | 6.10 | 5.20 | 8.07 | 8.146 | +0.329 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 8.07 | 8.148 | +0.331 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 8.07 | 8.189 | +0.373 |  |
| SEQ:random(hqq-4b k=0.1) | 6.10 | 5.20 | 8.07 | 8.332 [8.323, 8.342] | +0.516 |  |
| SEQ:random(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 8.07 | 8.434 [7.337, 9.530] | +0.617 |  |
| SEQ:greedy(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 8.07 | 46.027 | +38.210 |  |
| SEQ:greedy_indep(gptq_llmc-4b k=0.1) | 6.10 | 5.20 | 8.07 | 46.778 | +38.961 |  |
| SEQ:act_max(hqq-6b k=0.0) | 6.50 | 6.00 | 8.46 | 7.845 | +0.028 | ★ |
| HQQ-6 uniform | 6.50 | 6.00 | 8.46 | 7.845 | +0.028 | ★ |
| SEQ:greedy(hqq-4b k=0.2) | 7.70 | 6.40 | 9.67 | 8.028 | +0.211 |  |
| SEQ:greedy_indep(hqq-4b k=0.2) | 7.70 | 6.40 | 9.67 | 8.037 | +0.221 |  |
| SEQ:act_max(hqq-4b k=0.2) | 7.70 | 6.40 | 9.67 | 8.047 | +0.230 |  |
| SEQ:residual_max(hqq-4b k=0.2) | 7.70 | 6.40 | 9.67 | 8.048 | +0.232 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.2) | 7.70 | 6.40 | 9.67 | 8.070 | +0.253 |  |
| SEQ:residual_rms(hqq-4b k=0.2) | 7.70 | 6.40 | 9.67 | 8.090 | +0.274 |  |
| SEQ:act_scale(hqq-4b k=0.2) | 7.70 | 6.40 | 9.67 | 8.096 | +0.279 |  |
| SEQ:act_max(gptq_llmc-4b k=0.2) | 7.70 | 6.40 | 9.67 | 8.106 | +0.290 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.2) | 7.70 | 6.40 | 9.67 | 8.129 | +0.312 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.2) | 7.70 | 6.40 | 9.67 | 8.152 | +0.336 |  |
| SEQ:random(hqq-4b k=0.2) | 7.70 | 6.40 | 9.67 | 8.243 [8.025, 8.462] | +0.427 |  |
| SEQ:random(gptq_llmc-4b k=0.2) | 7.70 | 6.40 | 9.67 | 8.342 [7.907, 8.777] | +0.525 |  |
| SEQ:greedy(gptq_llmc-4b k=0.2) | 7.70 | 6.40 | 9.67 | 43.209 | +35.392 |  |
| SEQ:greedy_indep(gptq_llmc-4b k=0.2) | 7.70 | 6.40 | 9.67 | 45.410 | +37.594 |  |
| SEQ:act_max(hqq-8b k=0.0) | 8.50 | 8.00 | 10.46 | 7.820 | +0.003 | ★ |
| HQQ-8 uniform | 8.50 | 8.00 | 10.46 | 7.820 | +0.003 | ★ |
| FP16 | 16.00 | 16.00 | — | 7.817 | +0.000 | ★ |

> SEQ is on the Pareto frontier (6 point(s)). At ~8.5 bits, best SEQ (7.820) **loses to** HQQ-8 uniform (7.820).
